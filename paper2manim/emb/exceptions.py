"""EMB-specific exceptions."""

from __future__ import annotations


class EMBError(Exception):
    """Base class for all EMB failures."""


class StoreError(EMBError):
    """Persistence-layer error (SQLite locked / corrupted / missing schema)."""


class VectorIndexError(EMBError):
    """Vector-index error (Faiss load failure, dim mismatch, etc.)."""


class EmbedderError(EMBError):
    """Embedding model failed (model load / inference error)."""


class RecordNotFoundError(EMBError, KeyError):
    """Lookup by record id failed."""
