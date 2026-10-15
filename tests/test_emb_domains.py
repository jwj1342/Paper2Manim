"""EMB domain tags, schema migration, and read-only behavior.

Covers:
- Context.domain round-trip via SQLite store (incl. ALTER TABLE migration)
- retrieve_for_scene domain_filter + skip_success / skip_failure toggles
- consolidate_run honours domain + skip_success / skip_failure
- emb_consolidate_node short-circuits on emb_readonly
"""

from __future__ import annotations

import sqlite3

from paper2manim.emb.manager import build_default_emb, build_in_memory_emb
from paper2manim.emb.retrieval import retrieve_for_scene
from paper2manim.emb.schema import (
    Context,
    FailureBody,
    MemoryRecord,
    Provenance,
    SuccessBody,
)


def _mk_success(text: str, *, domain: str, paper: str = "arxiv:000.0") -> MemoryRecord:
    return MemoryRecord(
        polarity="success",
        context=Context(task_text=text, domain=domain, source_paper=paper),
        body=SuccessBody(rationale="r", code_full="from manim import *"),
        provenance=Provenance(extraction_source="high_score_scene", validated=True),
    )


def _mk_failure(text: str, *, domain: str) -> MemoryRecord:
    return MemoryRecord(
        polarity="failure",
        context=Context(task_text=text, domain=domain),
        body=FailureBody(trigger_pattern="overlap", root_cause="x", fix_recipe="y"),
        provenance=Provenance(extraction_source="visual_reflection", validated=True),
    )


# ---------- Schema / store ----------


def test_context_domain_default_is_empty():
    ctx = Context(task_text="hello")
    assert ctx.domain == ""


def test_context_domain_persisted_through_sqlite(tmp_path):
    emb = build_default_emb(tmp_path / "emb_a", use_faiss=False, use_real_embedder=False)
    rec = _mk_success("intro to attention", domain="cs")
    emb.put(rec)
    loaded = emb.get(rec.id)
    assert loaded.context.domain == "cs"


def test_sqlite_migration_adds_domain_column_to_legacy_db(tmp_path):
    """Legacy DBs (no domain col) must upgrade transparently on next open."""
    base = tmp_path / "emb_legacy"
    base.mkdir()
    db_path = base / "memory.db"
    # Older schema = current schema minus the ``domain`` column + idx_domain.
    legacy_schema = """
        CREATE TABLE memory_records (
            id                 TEXT PRIMARY KEY,
            polarity           TEXT NOT NULL CHECK (polarity IN ('success','failure')),
            run_id             TEXT NOT NULL DEFAULT '',
            scene_id           TEXT NOT NULL DEFAULT '',
            extraction_source  TEXT NOT NULL DEFAULT '',
            transition_ordinal INTEGER NOT NULL DEFAULT 0,
            context_json       TEXT NOT NULL,
            body_json          TEXT NOT NULL,
            provenance_json    TEXT NOT NULL,
            created_at         REAL NOT NULL,
            updated_at         REAL NOT NULL
        );
    """
    with sqlite3.connect(db_path) as conn:
        conn.executescript(legacy_schema)
    # Opening through build_default_emb should run _MIGRATIONS_SQL.
    emb = build_default_emb(base, use_faiss=False, use_real_embedder=False)
    rec = _mk_success("post-migration insert", domain="math")
    emb.put(rec)  # would error if column wasn't added
    loaded = emb.get(rec.id)
    assert loaded.context.domain == "math"


# ---------- retrieval channel + domain filter ----------


def test_retrieve_for_scene_skip_success_returns_no_success_hits():
    emb = build_in_memory_emb()
    emb.put(_mk_success("attention overview", domain="cs"))
    emb.put(_mk_failure("attention overview", domain="cs"))
    bundle = retrieve_for_scene(
        emb, "attention overview", k_success=2, k_failure=2, skip_success=True
    )
    assert bundle.success == []
    assert len(bundle.failure) >= 1


def test_retrieve_for_scene_skip_failure_returns_no_failure_hits():
    emb = build_in_memory_emb()
    emb.put(_mk_success("attention overview", domain="cs"))
    emb.put(_mk_failure("attention overview", domain="cs"))
    bundle = retrieve_for_scene(
        emb, "attention overview", k_success=2, k_failure=2, skip_failure=True
    )
    assert bundle.failure == []
    assert len(bundle.success) >= 1


def test_retrieve_for_scene_domain_filter_drops_other_domains():
    emb = build_in_memory_emb()
    emb.put(_mk_success("intro to fields", domain="cs"))
    emb.put(_mk_success("intro to fields", domain="physics"))
    bundle = retrieve_for_scene(
        emb, "intro to fields", k_success=5, k_failure=0, domain_filter="physics"
    )
    assert all(h.record.context.domain == "physics" for h in bundle.success)
    assert any(h.record.context.domain == "physics" for h in bundle.success)


# ---------- consolidate_run channel + domain ----------


def test_consolidate_run_skip_success_writes_only_failures(tmp_path, monkeypatch):
    """When skip_success=True, distill_success_records must not be invoked."""
    from paper2manim.emb import distill as distill_mod

    called = {"success": False, "failure": False}

    def fake_success(*a, **kw):
        called["success"] = True
        return []

    def fake_failure(*a, **kw):
        called["failure"] = True
        return []

    monkeypatch.setattr(distill_mod, "distill_success_records", fake_success)
    monkeypatch.setattr(distill_mod, "distill_failure_records", fake_failure)

    emb = build_in_memory_emb()
    distill_mod.consolidate_run("r1", emb, domain="cs", skip_success=True)
    assert called["success"] is False
    assert called["failure"] is True


def test_consolidate_run_skip_failure_writes_only_success(tmp_path, monkeypatch):
    from paper2manim.emb import distill as distill_mod

    called = {"success": False, "failure": False}
    monkeypatch.setattr(
        distill_mod, "distill_success_records",
        lambda *a, **kw: called.__setitem__("success", True) or [],
    )
    monkeypatch.setattr(
        distill_mod, "distill_failure_records",
        lambda *a, **kw: called.__setitem__("failure", True) or [],
    )

    emb = build_in_memory_emb()
    distill_mod.consolidate_run("r1", emb, skip_failure=True)
    assert called["success"] is True
    assert called["failure"] is False


def test_consolidate_run_passes_domain_to_distillers(tmp_path, monkeypatch):
    from paper2manim.emb import distill as distill_mod

    captured = {}

    def fake_success(*a, **kw):
        captured["success_domain"] = kw.get("domain")
        return []

    def fake_failure(*a, **kw):
        captured["failure_domain"] = kw.get("domain")
        return []

    monkeypatch.setattr(distill_mod, "distill_success_records", fake_success)
    monkeypatch.setattr(distill_mod, "distill_failure_records", fake_failure)

    emb = build_in_memory_emb()
    distill_mod.consolidate_run("r1", emb, domain="quantum")
    assert captured["success_domain"] == "quantum"
    assert captured["failure_domain"] == "quantum"


# ---------- emb_consolidate_node readonly short-circuit ----------


def test_emb_consolidate_node_readonly_skips(monkeypatch):
    from paper2manim.graphs import generation as generate_mod

    called = {"consolidate": False}

    def boom(*a, **kw):
        called["consolidate"] = True
        return {}

    monkeypatch.setattr(generate_mod, "consolidate_run", boom)
    state = {
        "run_id": "r1",
        "emb_enabled": True,
        "emb_readonly": True,
        "emb_store_path": "/tmp/never-used",
    }
    out = generate_mod.emb_consolidate_node(state)
    assert out == {}
    assert called["consolidate"] is False
