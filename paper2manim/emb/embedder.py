"""Embedder abstraction + sentence-transformers default.

Encoding is the one slow / dependency-heavy step in the EMB write path. Putting
it behind a small Protocol lets tests inject a deterministic mock without
loading the 80 MB MiniLM checkpoint.
"""

from __future__ import annotations

import hashlib
import logging
import math
from typing import Any, Protocol, runtime_checkable

from paper2manim.emb.exceptions import EmbedderError

log = logging.getLogger(__name__)


@runtime_checkable
class Embedder(Protocol):
    """Text → fixed-dim vector. Vectors should be L2-unit-normalized."""

    @property
    def dim(self) -> int: ...

    def encode(self, texts: list[str]) -> list[list[float]]: ...

    def encode_one(self, text: str) -> list[float]: ...

    def spec(self) -> dict[str, Any]:
        """Serializable identity (``kind``, ``model``, ``dim``).

        Persisted alongside the EMB store so different consumers (batch /
        ``emb`` CLI / next-batch re-open) can't bleed e.g. a 64-d HashEmbedder
        over a 384-d ST store's records.
        """
        ...


# ---- Default: sentence-transformers ----


_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_DEFAULT_DIM = 384


class SentenceTransformersEmbedder:
    """Wrap a ``sentence_transformers.SentenceTransformer`` model.

    The actual model is loaded lazily on the first ``encode`` call. This keeps
    ``import paper2manim.emb`` cheap and lets tests skip the dependency by
    using :class:`HashEmbedder` instead.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        *,
        device: str | None = None,
        normalize: bool = True,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._normalize = normalize
        self._model = None
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._load()
        assert self._dim is not None
        return self._dim

    def _load(self) -> None:
        try:
            # Lazy import: sentence-transformers pulls in transformers + torch,
            # both heavy. Reserve the cost until someone actually encodes.
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbedderError(
                "sentence-transformers is required for the default embedder. "
                "pip install sentence-transformers (or inject a custom Embedder)."
            ) from exc
        try:
            self._model = SentenceTransformer(self._model_name, device=self._device)
        except Exception as exc:  # noqa: BLE001 — model load can fail in many ways
            raise EmbedderError(
                f"Failed to load sentence-transformers model '{self._model_name}': "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        try:
            self._dim = int(self._model.get_sentence_embedding_dimension())
        except Exception:  # noqa: BLE001
            self._dim = _DEFAULT_DIM
        log.info(
            "[embedder] loaded %s dim=%d device=%s",
            self._model_name,
            self._dim,
            self._device or "auto",
        )

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._model is None:
            self._load()
        assert self._model is not None
        vecs = self._model.encode(
            texts,
            normalize_embeddings=self._normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vecs]

    def encode_one(self, text: str) -> list[float]:
        return self.encode([text])[0]

    def spec(self) -> dict[str, Any]:
        # Avoid forcing model load just to report the spec — fall back to the
        # documented default dim so ``emb stats`` can stay cheap on a fresh
        # process that hasn't called encode() yet.
        dim = self._dim if self._dim is not None else _DEFAULT_DIM
        return {"kind": "sentence-transformers", "model": self._model_name, "dim": int(dim)}


# ---- Test-friendly default: deterministic hash embedder ----


class HashEmbedder:
    """Deterministic, dependency-free embedder for tests / offline dev.

    Maps each input string to a stable unit vector by hashing it to a seed,
    then drawing ``dim`` pseudo-random floats. Different strings give nearly
    orthogonal vectors with high probability, so nearest-neighbour search
    still meaningfully separates them — sufficient for unit tests.
    """

    def __init__(self, dim: int = 64) -> None:
        if dim <= 0:
            raise ValueError("HashEmbedder dim must be positive")
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def encode_one(self, text: str) -> list[float]:
        h = hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()
        seed = int.from_bytes(h, "big")
        # Mulberry32-style PRNG — repeatable across Python versions.
        state = seed & 0xFFFFFFFF
        vec: list[float] = []
        for _ in range(self._dim):
            state = (state + 0x6D2B79F5) & 0xFFFFFFFF
            t = state
            t = ((t ^ (t >> 15)) * (t | 1)) & 0xFFFFFFFF
            t ^= (t + ((t ^ (t >> 7)) * (t | 61))) & 0xFFFFFFFF
            x = ((t ^ (t >> 14)) & 0xFFFFFFFF) / 0x100000000  # [0,1)
            vec.append(x * 2.0 - 1.0)  # [-1, 1)
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [self.encode_one(t) for t in texts]

    def spec(self) -> dict[str, Any]:
        return {"kind": "hash", "dim": int(self._dim)}
