"""High-level EMB facade: store + index + embedder composed.

Most callers only touch :class:`EpisodicMemoryBank` — never the underlying
components directly. The facade handles three things the components don't:

1. Encoding ``context.task_text`` into ``context.task_embedding`` on write.
2. Keeping the vector index in sync with the store after every write / delete.
3. Persisting the index alongside the SQLite db so a fresh process picks up
   right where the previous one left off.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

from paper2manim.emb.embedder import (
    Embedder,
    HashEmbedder,
    SentenceTransformersEmbedder,
)
from paper2manim.emb.exceptions import EmbedderError, EMBError, VectorIndexError
from paper2manim.emb.index import (
    FaissVectorIndex,
    InMemoryVectorIndex,
    VectorIndex,
)
from paper2manim.emb.schema import (
    Context,
    FailureBody,
    MemoryRecord,
    Polarity,
    Provenance,
    SuccessBody,
)
from paper2manim.emb.store import (
    InMemoryMemoryStore,
    MemoryStore,
    SQLiteMemoryStore,
)

log = logging.getLogger(__name__)


class RetrievedRecord:
    """A query hit — the full ``MemoryRecord`` plus the similarity score.

    Wrapping it in a thin class (rather than returning a raw tuple) keeps call
    sites readable in the graph / coder / distillation modules.
    """

    __slots__ = ("record", "similarity")

    def __init__(self, record: MemoryRecord, similarity: float) -> None:
        self.record = record
        self.similarity = float(similarity)

    def __repr__(self) -> str:
        return (
            f"RetrievedRecord(id={self.record.id!r}, "
            f"polarity={self.record.polarity}, similarity={self.similarity:.3f})"
        )


class EpisodicMemoryBank:
    """Composable EMB. Inject custom store / index / embedder for tests.

    Default behaviour:

    * Build a per-polarity Faiss index keyed by record id so we never bleed
      success records into a failure-only query.
    * Re-use the stored ``context.task_embedding`` if present; otherwise call
      the embedder. This means the embedder's identity matters across runs —
      mixing different embedders against the same store will produce
      inconsistent retrieval.
    * Persist index files next to the SQLite db (``<dir>/success.index`` and
      ``<dir>/failure.index``).
    """

    def __init__(
        self,
        store: MemoryStore,
        embedder: Embedder,
        *,
        success_index: VectorIndex | None = None,
        failure_index: VectorIndex | None = None,
        store_dir: Path | None = None,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._success_index: VectorIndex = success_index or InMemoryVectorIndex(embedder.dim)
        self._failure_index: VectorIndex = failure_index or InMemoryVectorIndex(embedder.dim)
        self._dir = store_dir
        # Hydrate indices from store on first construction. Faiss / InMemory
        # indices both support add() so this is uniform across impls.
        self._rebuild_indices_from_store()

    # ---- internals ----

    def _index_for(self, polarity: Polarity) -> VectorIndex:
        return self._success_index if polarity == "success" else self._failure_index

    def _ensure_embedding(self, ctx: Context) -> Context:
        if ctx.task_embedding and len(ctx.task_embedding) == self._embedder.dim:
            return ctx
        try:
            vec = self._embedder.encode_one(ctx.task_text)
        except EmbedderError:
            raise
        except Exception as exc:  # noqa: BLE001 — wrap upstream into our taxonomy
            raise EmbedderError(f"encode failed: {type(exc).__name__}: {exc}") from exc
        return ctx.model_copy(update={"task_embedding": vec})

    def _rebuild_indices_from_store(self) -> None:
        # Cheap on EMB load (≤ 1K records expected); fast path for prod is
        # incremental add() in :meth:`put`. Re-hydration is needed because the
        # SQLite db survives process restarts while in-memory indices don't.
        for rec in self._store.all():
            if not rec.context.task_embedding:
                continue
            idx = self._index_for(rec.polarity)
            # Skip records already present in a freshly-loaded index. Without
            # this guard, FaissVectorIndex.add() sees the id in `_id_to_row`
            # and falls into its remove+add path — a full O(N) rebuild per
            # record, so rehydrating N records costs O(N²).
            if getattr(idx, "_id_to_row", None) is not None and rec.id in idx._id_to_row:
                continue
            if getattr(idx, "_vectors", None) is not None and rec.id in idx._vectors:
                continue
            try:
                idx.add(rec.id, rec.context.task_embedding)
            except VectorIndexError as exc:
                log.warning("[emb] skip rehydrate id=%s: %s", rec.id, exc)

    # ---- API ----

    def put(self, record: MemoryRecord) -> str:
        """Persist a record + register its embedding in the right index.

        Returns the record id. The caller can pre-compute
        ``record.context.task_embedding`` to skip the embedder call (useful
        when batch-encoding for bootstrap runs).
        """
        if record.context.task_embedding and len(record.context.task_embedding) != self._embedder.dim:
            raise EMBError(
                f"context.task_embedding dim {len(record.context.task_embedding)} != "
                f"embedder.dim {self._embedder.dim}"
            )
        ctx = self._ensure_embedding(record.context)
        rec = record.model_copy(update={"context": ctx})
        # Dedupe: if a record with the same (run_id, scene_id, polarity,
        # extraction_source) already exists, reuse its id so the row is
        # replaced in place — both in the store and (via remove+add) in the
        # vector index. Stores without find_id_by_provenance get no dedupe.
        existing_id = None
        finder = getattr(self._store, "find_id_by_provenance", None)
        if finder is not None:
            try:
                existing_id = finder(
                    rec.provenance.run_id,
                    rec.provenance.scene_id,
                    rec.polarity,
                    rec.provenance.extraction_source,
                    rec.provenance.transition_ordinal,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("[emb] find_id_by_provenance failed: %s", exc)
        if existing_id and existing_id != rec.id:
            # Remove the old vector row (different id), then rewrite the record
            # with the existing id so the new vector lands under the same key.
            self._index_for(rec.polarity).remove(existing_id)
            rec = rec.model_copy(update={"id": existing_id})
        self._store.put(rec)
        idx = self._index_for(rec.polarity)
        idx.add(rec.id, ctx.task_embedding)
        log.info(
            "[emb] put id=%s polarity=%s scene_role=%s source=%s",
            rec.id,
            rec.polarity,
            rec.context.scene_role,
            rec.context.source_paper or "<unknown>",
        )
        return rec.id

    def get(self, record_id: str) -> MemoryRecord:
        return self._store.get(record_id)

    def delete(self, record_id: str) -> None:
        try:
            rec = self._store.get(record_id)
        except Exception:  # noqa: BLE001
            return
        self._index_for(rec.polarity).remove(record_id)
        self._store.delete(record_id)

    def query(
        self,
        text: str,
        polarity: Polarity,
        k: int = 3,
        *,
        bump_hit: bool = True,
    ) -> list[RetrievedRecord]:
        """Return up to ``k`` highest-similarity records of the given polarity.

        ``bump_hit`` increments hit_count + last_used so the EMB can later
        identify cold records for pruning (proposal §9 risk #2).
        """
        if k <= 0:
            return []
        try:
            qvec = self._embedder.encode_one(text)
        except Exception as exc:  # noqa: BLE001
            raise EmbedderError(f"encode failed: {type(exc).__name__}: {exc}") from exc
        hits = self._index_for(polarity).query(qvec, k=k)
        out: list[RetrievedRecord] = []
        for rid, sim in hits:
            try:
                rec = self._store.get(rid)
            except Exception as exc:  # noqa: BLE001
                log.warning("[emb] dangling index id=%s: %s", rid, exc)
                continue
            out.append(RetrievedRecord(rec, sim))
            if bump_hit:
                try:
                    self._store.bump_hit(rid)
                except Exception:  # noqa: BLE001
                    pass
        return out

    def all(self, *, polarity: Polarity | None = None) -> list[MemoryRecord]:
        return self._store.all(polarity=polarity)

    def count(self, *, polarity: Polarity | None = None) -> int:
        return self._store.count(polarity=polarity)

    def stats(self) -> dict[str, int]:
        return {
            "total": self._store.count(),
            "success": self._store.count(polarity="success"),
            "failure": self._store.count(polarity="failure"),
            "success_indexed": self._success_index.size(),
            "failure_indexed": self._failure_index.size(),
        }

    def save_indices(self) -> None:
        """Persist Faiss/InMemory index files alongside the SQLite db.

        Safe to call at any time; the SQLite store is always authoritative.
        """
        if self._dir is None:
            return
        try:
            self._success_index.save(self._dir / "success.index")
            self._failure_index.save(self._dir / "failure.index")
        except Exception as exc:  # noqa: BLE001
            log.warning("[emb] save_indices failed: %s", exc)

    # ---- bulk helpers ----

    def add_many(self, records: Iterable[MemoryRecord]) -> list[str]:
        ids: list[str] = []
        for r in records:
            ids.append(self.put(r))
        return ids


# ---- Convenience constructors ----


def build_default_emb(
    base_dir: Path | str,
    *,
    use_faiss: bool = True,
    use_real_embedder: bool = True,
    embedder_model: str | None = None,
) -> EpisodicMemoryBank:
    """Construct an EMB with the recommended defaults.

    Layout under ``base_dir``::

        base_dir/
        ├── memory.db
        ├── success.index  (Faiss native + .meta pickle sidecar)
        └── failure.index

    Toggling ``use_real_embedder=False`` swaps in :class:`HashEmbedder`, which
    avoids the ~80 MB sentence-transformers download. Useful for tests, CI
    smoke runs, and offline bootstrap experiments where retrieval quality
    isn't being measured yet.
    """
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    store = SQLiteMemoryStore(base / "memory.db")
    if use_real_embedder:
        embedder: Embedder = SentenceTransformersEmbedder(
            embedder_model or "sentence-transformers/all-MiniLM-L6-v2"
        )
    else:
        embedder = HashEmbedder(dim=64)
    if use_faiss:
        try:
            s_idx: VectorIndex = FaissVectorIndex(embedder.dim)
            f_idx: VectorIndex = FaissVectorIndex(embedder.dim)
        except VectorIndexError as exc:
            log.warning("[emb] Faiss unavailable (%s); falling back to in-memory index", exc)
            s_idx = InMemoryVectorIndex(embedder.dim)
            f_idx = InMemoryVectorIndex(embedder.dim)
    else:
        s_idx = InMemoryVectorIndex(embedder.dim)
        f_idx = InMemoryVectorIndex(embedder.dim)
    # Load existing index files if present, otherwise rely on facade's
    # store-driven rebuild.
    s_path = base / "success.index"
    f_path = base / "failure.index"
    if s_path.exists():
        try:
            s_idx.load(s_path)
        except VectorIndexError as exc:
            log.warning("[emb] success index reload failed (%s); rebuilding from store", exc)
    if f_path.exists():
        try:
            f_idx.load(f_path)
        except VectorIndexError as exc:
            log.warning("[emb] failure index reload failed (%s); rebuilding from store", exc)
    return EpisodicMemoryBank(
        store=store,
        embedder=embedder,
        success_index=s_idx,
        failure_index=f_idx,
        store_dir=base,
    )


def build_in_memory_emb(*, embedder: Embedder | None = None) -> EpisodicMemoryBank:
    """Volatile EMB for tests; no disk side effects."""
    emb = embedder or HashEmbedder(dim=32)
    return EpisodicMemoryBank(
        store=InMemoryMemoryStore(),
        embedder=emb,
        success_index=InMemoryVectorIndex(emb.dim),
        failure_index=InMemoryVectorIndex(emb.dim),
        store_dir=None,
    )


__all__ = [
    "EpisodicMemoryBank",
    "RetrievedRecord",
    "build_default_emb",
    "build_in_memory_emb",
    # Re-export key types so callers can `from paper2manim.emb.manager import ...`
    "Context",
    "FailureBody",
    "MemoryRecord",
    "Provenance",
    "SuccessBody",
]
