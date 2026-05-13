"""Episodic Memory Bank (EMB) — dual-channel self-learning store.

The EMB persists two polarities of records:

* ``success``: Final high-score scenes (Rationale + complete code + frame hash)
  produced by §4.4a of the research proposal. Retrieved as soft-constraint
  *Reference Examples* during Coder generation.
* ``failure``: ``before → after`` reflection transitions that strictly improved
  VLM score (or moved from render-error to render-success). Distilled into
  structured Lessons (trigger / root_cause / fix_recipe / anti & good example).
  Retrieved as hard-constraint *Known Pitfalls*.

The bank composes three independent abstractions so any layer can be swapped:

* :class:`Embedder` — text → fixed-dim vector
* :class:`VectorIndex` — vector ANN store
* :class:`MemoryStore` — durable record store (full payload, hit_count,
  provenance)

Default stack: ``SentenceTransformersEmbedder`` (all-MiniLM-L6-v2, 384-d) +
``FaissVectorIndex`` (IndexFlatIP + L2-normalized vectors = cosine) +
``SQLiteMemoryStore`` (WAL mode, single ``memory_records`` table).

The high-level :class:`EpisodicMemoryBank` facade is the only entry point most
callers need.
"""

from __future__ import annotations

from paper2manim.emb.exceptions import (
    EmbedderError,
    EMBError,
    RecordNotFoundError,
    StoreError,
    VectorIndexError,
)
from paper2manim.emb.manager import EpisodicMemoryBank, build_default_emb
from paper2manim.emb.schema import (
    Context,
    FailureBody,
    MemoryRecord,
    Polarity,
    Provenance,
    SuccessBody,
)

__all__ = [
    "Context",
    "EMBError",
    "EmbedderError",
    "EpisodicMemoryBank",
    "FailureBody",
    "MemoryRecord",
    "Polarity",
    "Provenance",
    "RecordNotFoundError",
    "StoreError",
    "SuccessBody",
    "VectorIndexError",
    "build_default_emb",
]
