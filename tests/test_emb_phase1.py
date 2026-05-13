"""Phase 1 tests: EMB schema, store, index, embedder, facade."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from paper2manim.emb import (
    Context,
    FailureBody,
    MemoryRecord,
    Provenance,
    SuccessBody,
)
from paper2manim.emb.embedder import HashEmbedder
from paper2manim.emb.exceptions import (
    EmbedderError,
    RecordNotFoundError,
    StoreError,
    VectorIndexError,
)
from paper2manim.emb.index import InMemoryVectorIndex
from paper2manim.emb.manager import build_default_emb, build_in_memory_emb
from paper2manim.emb.store import InMemoryMemoryStore, SQLiteMemoryStore

# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #


class TestSchema:
    def test_success_record_round_trips(self):
        rec = MemoryRecord(
            polarity="success",
            context=Context(task_text="intro to vectors", scene_role="background"),
            body=SuccessBody(rationale="staged write-in", code_full="from manim import *\n"),
        )
        payload = rec.model_dump_json()
        clone = MemoryRecord.model_validate_json(payload)
        assert clone.polarity == "success"
        assert isinstance(clone.body, SuccessBody)
        assert clone.body.rationale == "staged write-in"

    def test_failure_record_round_trips(self):
        rec = MemoryRecord(
            polarity="failure",
            context=Context(task_text="text overlaps with axes"),
            body=FailureBody(
                trigger_pattern="Axes + Text",
                root_cause="default position collision",
                fix_recipe="next_to(axes, UP)",
            ),
            provenance=Provenance(validated=True, before_score=2.0, after_score=3.5),
        )
        clone = MemoryRecord.model_validate_json(rec.model_dump_json())
        assert clone.polarity == "failure"
        assert isinstance(clone.body, FailureBody)
        assert clone.provenance.validated is True
        assert clone.provenance.after_score == 3.5

    def test_polarity_body_mismatch_rejected(self):
        # Pydantic v2 smart-union coerces a FailureBody-shaped payload before
        # model_post_init runs, so the construction succeeds in the union
        # resolution and then trips our explicit ``polarity == body`` check.
        # That either raises ValueError (post_init path) or ValidationError
        # (if pydantic refused the union coercion entirely). Accept either.
        with pytest.raises((ValueError, ValidationError)):
            MemoryRecord(
                polarity="success",
                context=Context(task_text="x"),
                body=FailureBody(
                    trigger_pattern="t",
                    root_cause="r",
                    fix_recipe="f",
                ),
            )

    def test_extra_fields_forbidden(self):
        # extra fields should be rejected per ConfigDict(extra="forbid")
        with pytest.raises(ValidationError):
            Context(task_text="x", bogus="nope")  # type: ignore[call-arg]


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #


def _success_rec(text: str = "x") -> MemoryRecord:
    return MemoryRecord(
        polarity="success",
        context=Context(task_text=text),
        body=SuccessBody(rationale="r", code_full="c"),
    )


def _failure_rec(text: str = "y", before: float = 2.0, after: float = 3.0) -> MemoryRecord:
    return MemoryRecord(
        polarity="failure",
        context=Context(task_text=text),
        body=FailureBody(trigger_pattern="t", root_cause="rc", fix_recipe="f"),
        provenance=Provenance(validated=True, before_score=before, after_score=after),
    )


class TestSQLiteStore:
    def test_put_get_delete(self, tmp_path):
        store = SQLiteMemoryStore(tmp_path / "memory.db")
        rec = _success_rec("alpha")
        store.put(rec)
        assert store.count() == 1
        got = store.get(rec.id)
        assert got.id == rec.id
        assert got.body.rationale == "r"
        store.delete(rec.id)
        assert store.count() == 0
        with pytest.raises(RecordNotFoundError):
            store.get(rec.id)

    def test_polarity_filtering(self, tmp_path):
        store = SQLiteMemoryStore(tmp_path / "memory.db")
        store.put(_success_rec("a"))
        store.put(_success_rec("b"))
        store.put(_failure_rec("c"))
        assert store.count() == 3
        assert store.count(polarity="success") == 2
        assert store.count(polarity="failure") == 1
        assert len(store.all(polarity="success")) == 2

    def test_bump_hit_increments_and_persists(self, tmp_path):
        store = SQLiteMemoryStore(tmp_path / "memory.db")
        rec = _success_rec()
        store.put(rec)
        store.bump_hit(rec.id, now=12345.0)
        got = store.get(rec.id)
        assert got.provenance.hit_count == 1
        assert got.provenance.last_used == 12345.0
        store.bump_hit(rec.id, now=12346.0)
        assert store.get(rec.id).provenance.hit_count == 2

    def test_idempotent_put_overwrites_body(self, tmp_path):
        store = SQLiteMemoryStore(tmp_path / "memory.db")
        rec = _success_rec("first")
        store.put(rec)
        rec2 = rec.model_copy(deep=True)
        rec2.body = SuccessBody(rationale="updated", code_full="c")
        store.put(rec2)
        got = store.get(rec.id)
        assert got.body.rationale == "updated"
        assert store.count() == 1

    def test_persists_across_instances(self, tmp_path):
        db = tmp_path / "memory.db"
        s1 = SQLiteMemoryStore(db)
        rec = _success_rec()
        s1.put(rec)
        s1.close()
        s2 = SQLiteMemoryStore(db)
        assert s2.count() == 1
        assert s2.get(rec.id).id == rec.id


class TestInMemoryStore:
    def test_basic_crud(self):
        store = InMemoryMemoryStore()
        rec = _success_rec()
        store.put(rec)
        assert store.count() == 1
        got = store.get(rec.id)
        assert got.id == rec.id
        store.delete(rec.id)
        with pytest.raises(RecordNotFoundError):
            store.get(rec.id)

    def test_get_returns_deep_copy(self):
        """Mutating a returned record must not corrupt the store."""
        store = InMemoryMemoryStore()
        rec = _success_rec()
        store.put(rec)
        got = store.get(rec.id)
        got.body.rationale = "MUTATED"
        again = store.get(rec.id)
        assert again.body.rationale == "r"  # unchanged


# --------------------------------------------------------------------------- #
# Embedder
# --------------------------------------------------------------------------- #


class TestHashEmbedder:
    def test_dim_respected(self):
        e = HashEmbedder(dim=32)
        v = e.encode_one("hello")
        assert len(v) == 32
        assert e.dim == 32

    def test_deterministic(self):
        e = HashEmbedder(dim=16)
        assert e.encode_one("same") == e.encode_one("same")

    def test_unit_norm(self):
        e = HashEmbedder(dim=16)
        v = e.encode_one("anything")
        n = math.sqrt(sum(x * x for x in v))
        assert abs(n - 1.0) < 1e-6

    def test_different_strings_yield_different_vectors(self):
        e = HashEmbedder(dim=16)
        assert e.encode_one("a") != e.encode_one("b")

    def test_encode_batch(self):
        e = HashEmbedder(dim=8)
        out = e.encode(["x", "y", "z"])
        assert len(out) == 3
        assert all(len(v) == 8 for v in out)


# --------------------------------------------------------------------------- #
# Vector index
# --------------------------------------------------------------------------- #


class TestInMemoryVectorIndex:
    def test_add_query(self):
        idx = InMemoryVectorIndex(dim=4)
        idx.add("a", [1.0, 0.0, 0.0, 0.0])
        idx.add("b", [0.0, 1.0, 0.0, 0.0])
        hits = idx.query([1.0, 0.0, 0.0, 0.0], k=2)
        assert hits[0][0] == "a"
        # a should have similarity ~1.0, b ~0.0
        assert hits[0][1] > 0.99
        assert hits[1][0] == "b"

    def test_remove_excludes_from_results(self):
        idx = InMemoryVectorIndex(dim=2)
        idx.add("a", [1.0, 0.0])
        idx.add("b", [0.0, 1.0])
        idx.remove("a")
        hits = idx.query([1.0, 0.0], k=2)
        assert [h[0] for h in hits] == ["b"]

    def test_dim_mismatch_raises(self):
        idx = InMemoryVectorIndex(dim=4)
        with pytest.raises(VectorIndexError):
            idx.add("a", [1.0, 0.0])

    def test_query_empty_returns_empty(self):
        idx = InMemoryVectorIndex(dim=4)
        assert idx.query([1.0, 0.0, 0.0, 0.0], k=3) == []

    def test_persistence_round_trip(self, tmp_path):
        idx = InMemoryVectorIndex(dim=3)
        idx.add("a", [1.0, 0.0, 0.0])
        idx.add("b", [0.0, 1.0, 0.0])
        p = tmp_path / "idx.pkl"
        idx.save(p)

        idx2 = InMemoryVectorIndex(dim=3)
        idx2.load(p)
        assert idx2.size() == 2
        hits = idx2.query([1.0, 0.0, 0.0], k=1)
        assert hits[0][0] == "a"


# --------------------------------------------------------------------------- #
# EpisodicMemoryBank facade
# --------------------------------------------------------------------------- #


class TestFacade:
    def test_put_and_query_polarity_isolated(self):
        emb = build_in_memory_emb()
        s_rec = _success_rec("intro pythagoras")
        f_rec = _failure_rec("axes text overlap")
        s_id = emb.put(s_rec)
        f_id = emb.put(f_rec)
        assert emb.count() == 2

        s_hits = emb.query("pythagoras intro", polarity="success", k=5)
        f_hits = emb.query("axes overlap", polarity="failure", k=5)
        assert len(s_hits) == 1 and s_hits[0].record.id == s_id
        assert len(f_hits) == 1 and f_hits[0].record.id == f_id

    def test_query_returns_empty_when_nothing_indexed(self):
        emb = build_in_memory_emb()
        assert emb.query("anything", polarity="success", k=3) == []
        assert emb.query("anything", polarity="failure", k=3) == []

    def test_put_skips_re_encoding_when_embedding_present(self):
        e = HashEmbedder(dim=16)
        rec = MemoryRecord(
            polarity="success",
            context=Context(task_text="x", task_embedding=e.encode_one("x")),
            body=SuccessBody(rationale="r", code_full="c"),
        )
        emb = build_in_memory_emb(embedder=e)
        rid = emb.put(rec)
        # The stored embedding must match exactly the precomputed one
        got = emb.get(rid)
        assert got.context.task_embedding == rec.context.task_embedding

    def test_query_bumps_hit_count(self):
        emb = build_in_memory_emb()
        rid = emb.put(_success_rec())
        emb.query("anything", polarity="success", k=3)
        emb.query("anything", polarity="success", k=3)
        got = emb.get(rid)
        assert got.provenance.hit_count == 2
        assert got.provenance.last_used is not None

    def test_query_with_bump_hit_false_does_not_bump(self):
        emb = build_in_memory_emb()
        rid = emb.put(_success_rec())
        emb.query("x", polarity="success", k=3, bump_hit=False)
        assert emb.get(rid).provenance.hit_count == 0

    def test_delete_removes_from_index_and_store(self):
        emb = build_in_memory_emb()
        rid = emb.put(_success_rec())
        emb.delete(rid)
        assert emb.count() == 0
        assert emb.query("x", polarity="success", k=3) == []

    def test_stats(self):
        emb = build_in_memory_emb()
        emb.put(_success_rec("a"))
        emb.put(_success_rec("b"))
        emb.put(_failure_rec("c"))
        st = emb.stats()
        assert st["total"] == 3
        assert st["success"] == 2
        assert st["failure"] == 1
        assert st["success_indexed"] == 2
        assert st["failure_indexed"] == 1


class TestPersistedFacade:
    def test_save_load_cycle(self, tmp_path):
        base = tmp_path / "emb"
        emb1 = build_default_emb(base, use_faiss=False, use_real_embedder=False)
        rid_s = emb1.put(_success_rec("alpha"))
        rid_f = emb1.put(_failure_rec("beta"))
        emb1.save_indices()
        st1 = emb1.stats()
        assert st1["total"] == 2

        # Fresh process: re-open the EMB on the same dir.
        emb2 = build_default_emb(base, use_faiss=False, use_real_embedder=False)
        st2 = emb2.stats()
        assert st2 == st1
        # Records remain queryable.
        assert emb2.get(rid_s).id == rid_s
        assert emb2.get(rid_f).id == rid_f
        # Index is hot, retrieval works without rebuild.
        s_hits = emb2.query("alpha", polarity="success", k=3)
        assert s_hits and s_hits[0].record.id == rid_s

    def test_index_rebuilt_from_store_if_file_corrupt(self, tmp_path):
        base = tmp_path / "emb"
        emb1 = build_default_emb(base, use_faiss=False, use_real_embedder=False)
        rid = emb1.put(_success_rec("gamma"))
        emb1.save_indices()
        # Corrupt the index sidecar.
        (base / "success.index").write_bytes(b"not a pickle")
        # Reload — should warn but recover by rebuilding from SQLite.
        emb2 = build_default_emb(base, use_faiss=False, use_real_embedder=False)
        assert emb2.stats()["success"] == 1
        assert emb2.get(rid).id == rid


# --------------------------------------------------------------------------- #
# Exceptions wired up sensibly
# --------------------------------------------------------------------------- #


def test_store_error_inheritance():
    assert issubclass(StoreError, Exception)
    assert issubclass(EmbedderError, Exception)
    assert issubclass(VectorIndexError, Exception)
    assert issubclass(RecordNotFoundError, KeyError)


# --------------------------------------------------------------------------- #
# Provenance dedup
# --------------------------------------------------------------------------- #


def _make_record(*, run_id: str, scene_id: str, extraction_source: str, task: str) -> MemoryRecord:
    return MemoryRecord(
        polarity="success",
        context=Context(task_text=task, scene_role="background"),
        body=SuccessBody(rationale=task, code_full="from manim import *\n"),
        provenance=Provenance(
            run_id=run_id, scene_id=scene_id, extraction_source=extraction_source
        ),
    )


class TestProvenanceDedup:
    def test_inmemory_store_finds_id_by_provenance(self):
        store = InMemoryMemoryStore()
        rec = _make_record(run_id="r1", scene_id="S1", extraction_source="high_score_scene", task="x")
        store.put(rec)
        found = store.find_id_by_provenance("r1", "S1", "success", "high_score_scene")
        assert found == rec.id
        assert store.find_id_by_provenance("", "S1", "success", "high_score_scene") is None
        assert store.find_id_by_provenance("r1", "S1", "failure", "high_score_scene") is None

    def test_sqlite_store_finds_id_by_provenance(self, tmp_path):
        store = SQLiteMemoryStore(tmp_path / "memory.db")
        rec = _make_record(run_id="r1", scene_id="S1", extraction_source="high_score_scene", task="x")
        store.put(rec)
        assert store.find_id_by_provenance("r1", "S1", "success", "high_score_scene") == rec.id
        assert store.find_id_by_provenance("r1", "S1", "success", "visual_reflection") is None

    def test_manager_put_reuses_id_for_same_provenance(self):
        emb = build_in_memory_emb()
        rec1 = _make_record(
            run_id="r1", scene_id="S1", extraction_source="high_score_scene", task="first"
        )
        id1 = emb.put(rec1)
        # A second record with same provenance tuple but different body should
        # collapse onto the same id rather than accumulate.
        rec2 = _make_record(
            run_id="r1", scene_id="S1", extraction_source="high_score_scene", task="second"
        )
        id2 = emb.put(rec2)
        assert id1 == id2
        assert emb.count(polarity="success") == 1
        # Body of the surviving record reflects the latest write.
        assert emb.get(id1).body.rationale == "second"
        # Different extraction_source must NOT collapse.
        rec3 = _make_record(
            run_id="r1", scene_id="S1", extraction_source="visual_reflection", task="third"
        )
        id3 = emb.put(rec3)
        assert id3 != id1
        assert emb.count(polarity="success") == 2

    def test_empty_provenance_does_not_dedup(self):
        # Records with empty run_id / scene_id (e.g. manually inserted seeds)
        # are exempt from the unique constraint and accumulate normally.
        emb = build_in_memory_emb()
        a = _make_record(run_id="", scene_id="", extraction_source="manual", task="a")
        b = _make_record(run_id="", scene_id="", extraction_source="manual", task="b")
        ida = emb.put(a)
        idb = emb.put(b)
        assert ida != idb
        assert emb.count(polarity="success") == 2
