"""Vector index abstraction + Faiss default.

The index stores *only* (record_id, vector) pairs — the full record body lives
in the :class:`~paper2manim.emb.store.MemoryStore`. We keep them separate so:

* Tests can mock the index without faking a Pydantic model.
* The store stays portable (SQLite is fine on any host); only retrieval needs
  Faiss, which can be swapped for sklearn / hnswlib / pgvector by writing
  another :class:`VectorIndex` impl.
"""

from __future__ import annotations

import logging
import math
import pickle
from pathlib import Path
from typing import Protocol, runtime_checkable

from paper2manim.emb.exceptions import VectorIndexError

log = logging.getLogger(__name__)


@runtime_checkable
class VectorIndex(Protocol):
    """Maintains (record_id, unit_vector) pairs and answers top-k queries."""

    @property
    def dim(self) -> int: ...

    def add(self, record_id: str, vector: list[float]) -> None: ...

    def remove(self, record_id: str) -> None: ...

    def query(self, vector: list[float], k: int) -> list[tuple[str, float]]:
        """Return up to ``k`` ``(record_id, similarity)`` pairs, highest sim first."""
        ...

    def size(self) -> int: ...

    def save(self, path: Path) -> None: ...

    def load(self, path: Path) -> None: ...


# ---- Faiss default ----


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm <= 0.0:
        return vec
    return [v / norm for v in vec]


class FaissVectorIndex:
    """Cosine similarity via ``IndexFlatIP`` on L2-normalized vectors.

    Faiss only stores int64 row ids, so we keep a side dict
    ``{int_row_id ↔ str_record_id}`` to round-trip Pydantic ``MemoryRecord.id``.

    Removal: Faiss's flat indexes support ``remove_ids`` natively, but the API
    requires an ``IDSelector``. We implement remove by rebuilding from the
    remaining live vectors — fine at our scale (< 10K records expected).
    """

    def __init__(self, dim: int) -> None:
        if dim <= 0:
            raise ValueError("FaissVectorIndex dim must be positive")
        self._dim = int(dim)
        self._index = None  # type: ignore[assignment]
        self._id_to_row: dict[str, int] = {}
        self._row_to_id: dict[int, str] = {}
        self._next_row: int = 0
        self._init_index()

    # ---- internals ----

    def _init_index(self) -> None:
        try:
            import faiss  # type: ignore
        except ImportError as exc:
            raise VectorIndexError(
                "faiss is required for FaissVectorIndex. pip install faiss-cpu "
                "(or inject a custom VectorIndex)."
            ) from exc
        self._faiss = faiss
        self._index = faiss.IndexFlatIP(self._dim)

    def _vec_to_numpy(self, vector: list[float]) -> object:
        if len(vector) != self._dim:
            raise VectorIndexError(
                f"vector dim mismatch: got {len(vector)}, index expects {self._dim}"
            )
        import numpy as np

        return np.asarray([_l2_normalize(vector)], dtype="float32")

    # ---- Protocol ----

    @property
    def dim(self) -> int:
        return self._dim

    def add(self, record_id: str, vector: list[float]) -> None:
        if record_id in self._id_to_row:
            # Idempotent re-add: replace by remove + add to keep vector fresh.
            self.remove(record_id)
        arr = self._vec_to_numpy(vector)
        assert self._index is not None
        self._index.add(arr)
        row = self._next_row
        self._next_row += 1
        self._id_to_row[record_id] = row
        self._row_to_id[row] = record_id

    def remove(self, record_id: str) -> None:
        if record_id not in self._id_to_row:
            return
        # Rebuild without the removed vector. For our scale this is fine; if we
        # ever push past 10K records we can switch to faiss IDMap2 + remove_ids.
        keep: list[tuple[str, list[float]]] = []
        # Extract current vectors via reconstruct + skip target row.
        target_row = self._id_to_row[record_id]
        assert self._index is not None
        for rid, row in self._id_to_row.items():
            if rid == record_id:
                continue
            vec = self._index.reconstruct(row).tolist()
            keep.append((rid, vec))
        # Reset and re-add the survivors.
        self._index.reset()
        self._id_to_row.clear()
        self._row_to_id.clear()
        self._next_row = 0
        del target_row  # silence linter; useful for debugging
        for rid, vec in keep:
            arr = self._vec_to_numpy(vec)
            self._index.add(arr)
            r = self._next_row
            self._next_row += 1
            self._id_to_row[rid] = r
            self._row_to_id[r] = rid

    def query(self, vector: list[float], k: int) -> list[tuple[str, float]]:
        if k <= 0 or self.size() == 0:
            return []
        arr = self._vec_to_numpy(vector)
        assert self._index is not None
        k_eff = min(k, self.size())
        sims, rows = self._index.search(arr, k_eff)
        out: list[tuple[str, float]] = []
        for sim, row in zip(sims[0].tolist(), rows[0].tolist(), strict=True):
            if row < 0:
                continue
            rid = self._row_to_id.get(int(row))
            if rid is None:
                continue
            out.append((rid, float(sim)))
        return out

    def size(self) -> int:
        return len(self._id_to_row)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        assert self._index is not None
        self._faiss.write_index(self._index, str(path))
        meta = {
            "dim": self._dim,
            "id_to_row": self._id_to_row,
            "row_to_id": self._row_to_id,
            "next_row": self._next_row,
        }
        path.with_suffix(path.suffix + ".meta").write_bytes(pickle.dumps(meta))

    def load(self, path: Path) -> None:
        if not path.exists():
            raise VectorIndexError(f"Faiss index file not found: {path}")
        meta_path = path.with_suffix(path.suffix + ".meta")
        if not meta_path.exists():
            raise VectorIndexError(f"Faiss meta sidecar missing: {meta_path}")
        try:
            meta = pickle.loads(meta_path.read_bytes())
        except (pickle.UnpicklingError, EOFError, KeyError, ValueError, TypeError) as exc:
            raise VectorIndexError(f"Faiss meta unreadable at {meta_path}: {exc}") from exc
        try:
            dim = int(meta["dim"])
        except (KeyError, TypeError, ValueError) as exc:
            raise VectorIndexError(f"Faiss meta missing dim field: {exc}") from exc
        if dim != self._dim:
            raise VectorIndexError(f"Saved index dim {dim} != configured dim {self._dim}")
        try:
            self._index = self._faiss.read_index(str(path))
        except Exception as exc:  # noqa: BLE001
            raise VectorIndexError(f"Faiss read_index failed for {path}: {exc}") from exc
        self._id_to_row = dict(meta["id_to_row"])
        self._row_to_id = {int(k): v for k, v in meta["row_to_id"].items()}
        self._next_row = int(meta["next_row"])


# ---- Fallback: in-memory brute-force (no faiss needed) ----


class InMemoryVectorIndex:
    """Brute-force cosine index used when faiss isn't available.

    Suitable for tests and very small EMBs (< 1K records).
    """

    def __init__(self, dim: int) -> None:
        self._dim = int(dim)
        self._vectors: dict[str, list[float]] = {}

    @property
    def dim(self) -> int:
        return self._dim

    def add(self, record_id: str, vector: list[float]) -> None:
        if len(vector) != self._dim:
            raise VectorIndexError(
                f"vector dim mismatch: got {len(vector)}, index expects {self._dim}"
            )
        self._vectors[record_id] = _l2_normalize(vector)

    def remove(self, record_id: str) -> None:
        self._vectors.pop(record_id, None)

    def query(self, vector: list[float], k: int) -> list[tuple[str, float]]:
        if k <= 0 or not self._vectors:
            return []
        if len(vector) != self._dim:
            raise VectorIndexError(
                f"vector dim mismatch: got {len(vector)}, index expects {self._dim}"
            )
        q = _l2_normalize(vector)
        scored = [
            (rid, sum(a * b for a, b in zip(q, v, strict=True)))
            for rid, v in self._vectors.items()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def size(self) -> int:
        return len(self._vectors)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps({"dim": self._dim, "vectors": self._vectors}))

    def load(self, path: Path) -> None:
        if not path.exists():
            raise VectorIndexError(f"index file not found: {path}")
        try:
            blob = pickle.loads(path.read_bytes())
        except (pickle.UnpicklingError, EOFError, KeyError, ValueError, TypeError) as exc:
            raise VectorIndexError(f"failed to deserialize index at {path}: {exc}") from exc
        try:
            dim = int(blob["dim"])
            vectors = dict(blob["vectors"])
        except (KeyError, TypeError, ValueError) as exc:
            raise VectorIndexError(f"index blob missing expected fields: {exc}") from exc
        if dim != self._dim:
            raise VectorIndexError(f"Saved index dim {dim} != configured dim {self._dim}")
        self._vectors = vectors
