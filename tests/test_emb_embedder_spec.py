"""Regression: issue #27 Bug B — embedder identity pinning.

Before the fix, ``build_default_emb`` would happily construct a fresh
HashEmbedder(64) on top of a store whose records were written with
sentence-transformers (384-d). The downstream ``_rebuild_indices_from_store``
then ``add()``-failed every record with ``vector dim mismatch`` and silently
log-warned them away — leaving an EMB that *looked* alive (SQLite records
present) but retrieved nothing (``success_indexed=0``).

These tests pin down four invariants:

1. A fresh store writes ``embedder.json`` on creation.
2. Subsequent opens use the pinned spec, ignoring caller flags that disagree.
3. Legacy stores (no spec file, but records with embeddings) get a spec
   backfilled from the first record's embedding dim.
4. Genuine dim mismatch on rehydrate now raises ``VectorIndexError`` instead
   of silently dropping records.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from paper2manim.emb import (
    Context,
    MemoryRecord,
    Provenance,
    SuccessBody,
)
from paper2manim.emb.embedder import HashEmbedder
from paper2manim.emb.exceptions import VectorIndexError
from paper2manim.emb.manager import (
    EMBEDDER_SPEC_FILENAME,
    EpisodicMemoryBank,
    build_default_emb,
)
from paper2manim.emb.store import SQLiteMemoryStore


def _spec_path(base: Path) -> Path:
    return base / EMBEDDER_SPEC_FILENAME


def _put_success(emb: EpisodicMemoryBank, text: str, run_id: str = "r-1") -> str:
    return emb.put(
        MemoryRecord(
            polarity="success",
            context=Context(task_text=text, scene_role="background"),
            body=SuccessBody(rationale="r", code_full="from manim import *\n"),
            provenance=Provenance(run_id=run_id, scene_id="S0"),
        )
    )


class TestFreshStorePinning:
    def test_fresh_hash_store_writes_spec(self, tmp_path):
        base = tmp_path / "emb"
        build_default_emb(base, use_faiss=False, use_real_embedder=False)
        spec = json.loads(_spec_path(base).read_text(encoding="utf-8"))
        assert spec == {"kind": "hash", "dim": 64}

    def test_fresh_st_store_writes_spec(self, tmp_path):
        """We never actually load ST here — ``build_default_emb`` only
        constructs the lazy wrapper, so we can write the spec without torch
        on disk."""
        base = tmp_path / "emb_st"
        build_default_emb(base, use_faiss=False, use_real_embedder=True)
        spec = json.loads(_spec_path(base).read_text(encoding="utf-8"))
        assert spec["kind"] == "sentence-transformers"
        assert spec["dim"] == 384
        assert spec["model"] == "sentence-transformers/all-MiniLM-L6-v2"


class TestPinnedSpecOverridesCaller:
    def test_hash_store_ignores_use_real_embedder_true(self, tmp_path, caplog):
        """Reproducer of issue #27 Bug B in inverted form: hash-bootstrapped
        store re-opened by a caller asking for ST should NOT silently switch."""
        base = tmp_path / "emb"
        # Bootstrap with hash@64 + write a real record so rebuild has work.
        emb1 = build_default_emb(base, use_faiss=False, use_real_embedder=False)
        _put_success(emb1, "intro to attention")
        emb1.save_indices()

        # Re-open asking for ST — pinned hash@64 should win.
        with caplog.at_level("WARNING", logger="paper2manim.emb.manager"):
            emb2 = build_default_emb(base, use_faiss=False, use_real_embedder=True)
        assert any("pinned to hash" in m for m in caplog.messages), caplog.messages
        # And critically: index actually has the record (no silent drop).
        assert emb2.stats()["success_indexed"] == 1

    def test_hash_store_ignores_custom_dim_change(self, tmp_path):
        """Caller can't change embedder dim by re-opening. A hash@64 store
        re-opened with use_real_embedder=False (default hash) still gives
        hash@64, not hash@128 or anything else."""
        base = tmp_path / "emb"
        build_default_emb(base, use_faiss=False, use_real_embedder=False)
        emb2 = build_default_emb(base, use_faiss=False, use_real_embedder=False)
        # No way to peek at private embedder, but stats works:
        assert emb2.stats()["total"] == 0


class TestLegacyStoreBackfill:
    def test_legacy_hash64_store_backfills(self, tmp_path):
        """Simulate a pre-fix store: SQLite with records carrying 64-dim
        embeddings, but no ``embedder.json`` on disk."""
        base = tmp_path / "legacy_hash"
        base.mkdir()
        # Hand-roll the store directly so build_default_emb isn't involved
        # in writing the spec.
        store = SQLiteMemoryStore(base / "memory.db")
        embedder = HashEmbedder(dim=64)
        rec = MemoryRecord(
            polarity="success",
            context=Context(
                task_text="intro",
                scene_role="background",
                task_embedding=embedder.encode_one("intro"),
            ),
            body=SuccessBody(rationale="r", code_full="from manim import *\n"),
            provenance=Provenance(run_id="r-legacy", scene_id="S0"),
        )
        store.put(rec)
        assert not _spec_path(base).exists()  # legacy: no spec yet

        emb = build_default_emb(base, use_faiss=False, use_real_embedder=False)
        # Spec was backfilled + persisted.
        assert _spec_path(base).exists()
        spec = json.loads(_spec_path(base).read_text())
        assert spec == {"kind": "hash", "dim": 64}
        # And the index actually hydrated — not silently dropped.
        assert emb.stats()["success_indexed"] == 1

    def test_legacy_st384_store_backfills_to_st(self, tmp_path):
        """Records with 384-dim embeddings (the ST default) are inferred as
        sentence-transformers, matching the original issue #27 scenario."""
        base = tmp_path / "legacy_st"
        base.mkdir()
        store = SQLiteMemoryStore(base / "memory.db")
        # Use HashEmbedder(384) just as a stand-in to produce a 384-d vector
        # without loading torch — the backfill only inspects the dim.
        stand_in = HashEmbedder(dim=384)
        rec = MemoryRecord(
            polarity="success",
            context=Context(
                task_text="text",
                scene_role="background",
                task_embedding=stand_in.encode_one("text"),
            ),
            body=SuccessBody(rationale="r", code_full="from manim import *\n"),
            provenance=Provenance(run_id="r-legacy-st", scene_id="S0"),
        )
        store.put(rec)

        # Caller asks for hash, but the legacy data is 384-d → backfill MUST
        # win (otherwise we'd be back to issue #27 Bug B).
        emb = build_default_emb(base, use_faiss=False, use_real_embedder=False)
        spec = json.loads(_spec_path(base).read_text())
        assert spec["kind"] == "sentence-transformers"
        assert spec["dim"] == 384
        assert emb.stats()["success_indexed"] == 1


class TestRehydrateStrictness:
    def test_rebuild_raises_on_dim_mismatch(self, tmp_path):
        """If something corrupts the spec / records so that rebuild can't
        ``add()`` a record, we raise — never silently log-and-skip."""
        base = tmp_path / "corrupt"
        base.mkdir()
        # Plant a record with a 384-d vector but pin spec to hash@64. This
        # mimics what happened pre-fix: spec says one dim, records have
        # another. After the fix, ``build_default_emb`` would normally
        # *correct* the spec to match the record (backfill); we bypass that
        # by writing the spec ourselves first.
        _spec_path(base).write_text(
            json.dumps({"kind": "hash", "dim": 64}), encoding="utf-8"
        )
        store = SQLiteMemoryStore(base / "memory.db")
        stand_in = HashEmbedder(dim=384)
        rec = MemoryRecord(
            polarity="success",
            context=Context(
                task_text="x",
                scene_role="background",
                task_embedding=stand_in.encode_one("x"),
            ),
            body=SuccessBody(rationale="r", code_full="from manim import *\n"),
            provenance=Provenance(run_id="r-corrupt", scene_id="S0"),
        )
        store.put(rec)

        with pytest.raises(VectorIndexError, match="rehydrate failed"):
            build_default_emb(base, use_faiss=False, use_real_embedder=False)


class TestEmbCliReadOnlyOnStStore:
    def test_open_emb_helper_uses_pinned_spec(self, tmp_path, caplog):
        """``cli_emb._open_emb`` used to hardcode ``use_real_embedder=False``,
        which clobbered ST-pinned stores. After the fix it should defer to
        the pinned spec, so ``stats()`` on an ST-pinned store returns the
        rehydrated records instead of silently zero."""
        from paper2manim.cli_emb import _open_emb

        base = tmp_path / "st_pinned"
        base.mkdir()
        # Stand in for an ST-pinned store: spec says ST@384, records carry
        # 384-d vectors. No torch needed because the embedder is lazy and
        # neither ``stats()`` nor ``_rebuild_indices_from_store`` calls
        # ``encode()``.
        _spec_path(base).write_text(
            json.dumps(
                {
                    "kind": "sentence-transformers",
                    "model": "sentence-transformers/all-MiniLM-L6-v2",
                    "dim": 384,
                }
            ),
            encoding="utf-8",
        )
        store = SQLiteMemoryStore(base / "memory.db")
        stand_in = HashEmbedder(dim=384)
        for i in range(3):
            store.put(
                MemoryRecord(
                    polarity="success",
                    context=Context(
                        task_text=f"task-{i}",
                        scene_role="background",
                        task_embedding=stand_in.encode_one(f"task-{i}"),
                    ),
                    body=SuccessBody(rationale="r", code_full="from manim import *\n"),
                    provenance=Provenance(run_id=f"r-{i}", scene_id="S0"),
                )
            )

        emb = _open_emb(str(base))
        stats = emb.stats()
        assert stats["total"] == 3
        assert stats["success_indexed"] == 3, (
            "Bug B regression: ST-pinned store re-opened via CLI must hydrate "
            "all records into the index, not silently drop them."
        )
