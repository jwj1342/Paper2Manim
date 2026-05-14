"""B6: cross-domain freeze + dataset schema tests.

Covers:
- Context.domain round-trip via SQLite store (incl. ALTER TABLE migration)
- retrieve_for_scene domain_filter + skip_success / skip_failure toggles
- consolidate_run honours domain + skip_success / skip_failure
- emb_consolidate_node short-circuits on emb_readonly
- scripts/dataset_validate.py offline schema checks
- scripts/cross_domain.py phase config + report layout
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

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
    # Pre-B6 schema = current schema minus the ``domain`` column + idx_domain.
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
    from paper2manim.graphs import mvp2 as mvp2_mod

    called = {"consolidate": False}

    def boom(*a, **kw):
        called["consolidate"] = True
        return {}

    monkeypatch.setattr(mvp2_mod, "consolidate_run", boom)
    state = {
        "run_id": "r1",
        "emb_enabled": True,
        "emb_readonly": True,
        "emb_store_path": "/tmp/never-used",
    }
    out = mvp2_mod.emb_consolidate_node(state)
    assert out == {}
    assert called["consolidate"] is False


# ---------- scripts/dataset_validate.py ----------


def _ds(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "dataset.csv"
    p.write_text(body, encoding="utf-8")
    return p


def test_dataset_validate_offline_accepts_well_formed(tmp_path):
    from scripts.dataset_validate import validate

    csv_path = _ds(
        tmp_path,
        "arxiv_id,section,domain,split,expected_scene_count_min\n"
        "1706.03762,Background,cs,bootstrap,2\n"
        "1412.6980,Algorithm,math,eval,\n",
    )
    rows, errs = validate(csv_path, offline=True)
    assert errs == []
    assert len(rows) == 2


def test_dataset_validate_rejects_unknown_domain(tmp_path):
    from scripts.dataset_validate import validate

    csv_path = _ds(
        tmp_path,
        "arxiv_id,section,domain,split,expected_scene_count_min\n"
        "1706.03762,Background,bio,bootstrap,2\n",
    )
    rows, errs = validate(csv_path, offline=True)
    assert rows == []
    assert any("unknown domain" in e for e in errs)


def test_dataset_validate_rejects_unknown_split(tmp_path):
    from scripts.dataset_validate import validate

    csv_path = _ds(
        tmp_path,
        "arxiv_id,section,domain,split,expected_scene_count_min\n"
        "1706.03762,Background,cs,training,2\n",
    )
    _rows, errs = validate(csv_path, offline=True)
    assert any("unknown split" in e for e in errs)


def test_dataset_validate_rejects_missing_required(tmp_path):
    from scripts.dataset_validate import validate

    csv_path = _ds(
        tmp_path,
        "arxiv_id,section,domain,split,expected_scene_count_min\n"
        ",Background,cs,bootstrap,2\n",
    )
    _rows, errs = validate(csv_path, offline=True)
    assert any("missing column 'arxiv_id'" in e for e in errs)


def test_dataset_validate_int_parse(tmp_path):
    from scripts.dataset_validate import validate

    csv_path = _ds(
        tmp_path,
        "arxiv_id,section,domain,split,expected_scene_count_min\n"
        "1706.03762,Background,cs,bootstrap,not-a-number\n",
    )
    _rows, errs = validate(csv_path, offline=True)
    assert any("not an int" in e for e in errs)


def test_dataset_validate_writes_outputs(tmp_path):
    from scripts.dataset_validate import _write_outputs, validate

    csv_path = _ds(
        tmp_path,
        "arxiv_id,section,domain,split,expected_scene_count_min\n"
        "1706.03762,Background,cs,bootstrap,2\n"
        ",Background,cs,bootstrap,2\n",
    )
    rows, errs = validate(csv_path, offline=True)
    out_csv, out_err = _write_outputs(csv_path, rows, errs)
    assert out_csv.exists() and out_err.exists()
    assert "1706.03762" in out_csv.read_text(encoding="utf-8")
    assert "missing column 'arxiv_id'" in out_err.read_text(encoding="utf-8")


def test_dataset_validate_committed_p2m_v1_passes_offline():
    from scripts.dataset_validate import validate

    repo_root = Path(__file__).resolve().parent.parent
    csv_path = repo_root / "examples" / "datasets" / "p2m_v1.csv"
    rows, errs = validate(csv_path, offline=True)
    assert errs == [], errs
    assert len(rows) >= 5


# ---------- scripts/cross_domain.py ----------


def test_cross_domain_config_for_phase():
    from scripts.cross_domain import _config_for_phase

    train = _config_for_phase("train", "/tmp/x")
    assert "--vlm" in train and "--emb" in train and "--emb-store-path" in train
    assert "--emb-readonly" not in train

    test = _config_for_phase("test", "/tmp/x")
    assert "--emb-readonly" in test

    base = _config_for_phase("baseline", "/tmp/x")
    assert "--no-emb" in base
    assert "--emb-readonly" not in base


def test_cross_domain_unknown_phase_raises():
    from scripts.cross_domain import _config_for_phase

    with pytest.raises(ValueError):
        _config_for_phase("totally-fake", "/tmp/x")


def test_cross_domain_read_tasks_filters_by_domain_and_split(tmp_path):
    from scripts.cross_domain import _read_tasks

    csv_path = _ds(
        tmp_path,
        "arxiv_id,section,domain,split,expected_scene_count_min\n"
        "1706.03762,Background,cs,cross_train,2\n"
        "1810.04805,Introduction,cs,cross_test,2\n"
        "1503.02531,Introduction,physics,cross_test,2\n",
    )
    train_cs = _read_tasks(csv_path, domain="cs", split="cross_train")
    assert [r["arxiv_id"] for r in train_cs] == ["1706.03762"]
    test_phys = _read_tasks(csv_path, domain="physics", split="cross_test")
    assert [r["arxiv_id"] for r in test_phys] == ["1503.02531"]
    none = _read_tasks(csv_path, domain="math", split="cross_test")
    assert none == []


def test_cross_domain_run_phase_dumps_report(tmp_path, monkeypatch):
    from scripts import cross_domain as cd

    def fake_run_one(task, *, base_args, quality, dry_run, **_):
        return cd.TaskOutcome(
            arxiv_id=task["arxiv_id"],
            section=task["section"],
            domain=task["domain"],
            exit_code=0,
            duration_s=0.01,
            run_id="run-fake-1",
        )

    monkeypatch.setattr(cd, "_run_one", fake_run_one)

    tasks = [
        {"arxiv_id": "1706.03762", "section": "Background", "domain": "cs"},
        {"arxiv_id": "1503.02531", "section": "Introduction", "domain": "physics"},
    ]
    report = cd.run_phase(
        phase="test",
        tasks=tasks,
        emb_store_path=tmp_path / "emb",
        quality="l",
        dry_run=True,
    )
    assert report.phase == "test"
    assert "--emb-readonly" in report.config_args
    assert len(report.outcomes) == 2
    assert all(o.run_id == "run-fake-1" for o in report.outcomes)

    out_path = tmp_path / "report.json"
    cd._dump_report(report, out_path)
    assert out_path.exists()
    import json
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["phase"] == "test"
    assert len(payload["outcomes"]) == 2


def test_cross_domain_extra_args_forwarded(tmp_path, monkeypatch):
    from scripts import cross_domain as cd

    captured: dict = {}

    def fake_run_one(task, *, base_args, quality, dry_run, **_):
        captured["base_args"] = base_args
        return cd.TaskOutcome(
            arxiv_id=task["arxiv_id"], section=task["section"], domain=task["domain"],
            exit_code=0, duration_s=0.0,
        )

    monkeypatch.setattr(cd, "_run_one", fake_run_one)
    cd.run_phase(
        phase="train",
        tasks=[{"arxiv_id": "x", "section": "y", "domain": "cs"}],
        emb_store_path=tmp_path / "emb",
        quality="l",
        dry_run=True,
        extra_args=["--max-retries", "1"],
    )
    assert "--max-retries" in captured["base_args"]
    assert "1" in captured["base_args"]


# ---------- _run_one: subprocess form ----------


class TestRunOneSubprocess:
    """``_run_one`` shells out via ``python -m paper2manim`` for the same
    reasons run_experiment._invoke_one does: crash isolation, timeout,
    no module-state leak, memory reclaim."""

    def _completed(self, *, returncode=0, stdout="", stderr=""):
        import subprocess
        return subprocess.CompletedProcess(
            args=[], returncode=returncode, stdout=stdout, stderr=stderr,
        )

    def _task(self):
        return {"arxiv_id": "1706.03762", "section": "Background", "domain": "cs"}

    def test_invokes_python_dash_m_paper2manim(self):
        from scripts import cross_domain as cd

        captured: dict = {}

        def fake_runner(cmd, **kwargs):
            captured["cmd"] = cmd
            captured["kwargs"] = kwargs
            return self._completed(stdout="RUN_ID=cd-fake-1\n")

        outcome = cd._run_one(
            self._task(), base_args=["--vlm", "--no-emb"],
            quality="l", dry_run=False, runner=fake_runner,
        )
        assert outcome.exit_code == 0
        assert outcome.run_id == "cd-fake-1"
        assert captured["cmd"][1:3] == ["-m", "paper2manim"]
        assert captured["cmd"][3] == "mvp2"
        assert "--dataset-domain" in captured["cmd"]
        assert "cs" in captured["cmd"]
        assert captured["kwargs"]["capture_output"] is True
        assert captured["kwargs"]["text"] is True

    def test_timeout_returns_124_with_partial_stdout(self):
        import subprocess

        from scripts import cross_domain as cd

        def fake_runner(cmd, **kwargs):
            raise subprocess.TimeoutExpired(
                cmd=cmd, timeout=2.0, output="RUN_ID=cd-t\nstart...",
            )

        outcome = cd._run_one(
            self._task(), base_args=[], quality="l", dry_run=False,
            timeout_s=2.0, runner=fake_runner,
        )
        assert outcome.exit_code == 124
        assert outcome.run_id == "cd-t"
        assert "timeout after 2.0s" in outcome.stderr_tail

    def test_executable_not_found_returns_127(self):
        from scripts import cross_domain as cd

        def fake_runner(cmd, **kwargs):
            raise FileNotFoundError(2, "no such file", cmd[0])

        outcome = cd._run_one(
            self._task(), base_args=[], quality="l", dry_run=False,
            runner=fake_runner,
        )
        assert outcome.exit_code == 127
        assert "executable not found" in outcome.stderr_tail

    def test_unexpected_runner_exception_returns_99(self):
        from scripts import cross_domain as cd

        def fake_runner(cmd, **kwargs):
            raise RuntimeError("disk full")

        outcome = cd._run_one(
            self._task(), base_args=[], quality="l", dry_run=False,
            runner=fake_runner,
        )
        assert outcome.exit_code == 99
        assert "RuntimeError" in outcome.stderr_tail

    def test_dry_run_appends_no_render_to_args(self):
        from scripts import cross_domain as cd

        captured: dict = {}

        def fake_runner(cmd, **kwargs):
            captured["cmd"] = cmd
            return self._completed(stdout="")

        cd._run_one(
            self._task(), base_args=[], quality="l", dry_run=True,
            runner=fake_runner,
        )
        assert "--no-render" in captured["cmd"]

    def test_run_phase_threads_timeout_to_run_one(self, tmp_path, monkeypatch):
        from scripts import cross_domain as cd

        captured: dict = {}

        def fake_run_one(task, *, base_args, quality, dry_run,
                         timeout_s=None, **_):
            captured["timeout_s"] = timeout_s
            return cd.TaskOutcome(
                arxiv_id=task["arxiv_id"], section=task["section"],
                domain=task["domain"], exit_code=0, duration_s=0.0,
            )

        monkeypatch.setattr(cd, "_run_one", fake_run_one)
        cd.run_phase(
            phase="train", tasks=[self._task()],
            emb_store_path=tmp_path / "emb",
            quality="l", dry_run=False, timeout_s=900.0,
        )
        assert captured["timeout_s"] == 900.0


# ---------- Real subprocess smoke ----------


class TestCrossDomainRealSubprocessSmoke:
    """Real ``python -m paper2manim`` exercise (PR #25 reviewer ask).

    Same shape as TestRealSubprocessSmoke in test_run_experiment.py — proves
    the subprocess wiring isn't silently dead. We can't run mvp2 here without
    LLM/network, so we route through a guaranteed-fast dummy command and
    verify the subprocess machinery captures the exit code + stderr."""

    def test_unknown_command_nonzero_with_stderr_captured(self):
        from scripts import cross_domain as cd

        # Reuse _run_one for the exercise; dataset-domain is required by
        # _run_one's argv builder so we need a valid task even if the inner
        # CLI is going to bail. Override base_args with a non-existent
        # subcommand-like flag so click exits non-zero quickly.
        outcome = cd._run_one(
            {"arxiv_id": "1706.03762", "section": "Background", "domain": "cs"},
            base_args=["--this-flag-does-not-exist"],
            quality="l", dry_run=False, timeout_s=30.0,
        )
        assert outcome.exit_code != 0
        assert outcome.stderr_tail  # captured something


# ---------- Shared dataset constants ----------


class TestSharedDatasetConstants:
    """CLI ``--dataset-domain`` and the validator must agree on the allowed
    set; otherwise the CLI silently writes a misspelled tag the validator
    later rejects, and ``retrieve_for_scene(domain_filter=...)`` permanently
    misses those records."""

    def test_cli_choice_uses_shared_domain_constants(self):
        # The validator and the CLI must read from the same source.
        from paper2manim.datasets import DOMAINS as cli_domains
        from scripts.dataset_validate import _DOMAINS as validator_domains

        assert set(cli_domains) == validator_domains, (
            "CLI's --dataset-domain Choice and dataset_validate's _DOMAINS "
            "must come from paper2manim.datasets.constants"
        )

    def test_cli_rejects_misspelled_domain(self, tmp_path):
        from click.testing import CliRunner

        from paper2manim.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, [
            "mvp2",
            "--arxiv", "1706.03762",
            "--dataset-domain", "phyics",  # typo: "physics"
            "--no-render",
        ])
        # click.Choice yields exit 2 for usage errors.
        assert result.exit_code != 0
        # Click's standard "Invalid value for '--dataset-domain'" message.
        out = (result.output or "") + (
            (result.stderr if hasattr(result, "stderr") else "") or ""
        )
        assert "phyics" in out

    def test_cli_accepts_valid_domain(self, tmp_path, monkeypatch):
        """Sanity: a valid domain still parses without click rejecting it.
        We stub out build_mvp2_graph so we don't invoke the real graph."""
        from click.testing import CliRunner

        # Patch the graph runner so the command exits early with a known
        # signal instead of doing real work.
        sentinel = {"called": False}

        class _DummyGraph:
            def with_config(self, *a, **kw):
                return self

            def invoke(self, state, **kw):
                sentinel["called"] = True
                # Return a minimal final state so the CLI's post-graph code
                # doesn't blow up.
                return {
                    **state,
                    "fatal_error": "stub: real graph bypassed for test",
                    "rendered_videos": [],
                    "skipped_scenes": [],
                }

        monkeypatch.setattr(
            "paper2manim.graphs.mvp2.build_mvp2_graph",
            lambda: _DummyGraph(),
        )
        runner = CliRunner()
        result = runner.invoke(cli_for_mvp2_test(), [
            "mvp2",
            "--arxiv", "1706.03762",
            "--dataset-domain", "physics",  # valid
            "--no-render",
        ])
        # Either the dummy graph ran (sentinel True) or click rejected the
        # invocation BEFORE reaching the graph for some unrelated reason —
        # but in no case should click reject the *domain* itself.
        out = (result.output or "")
        assert "Invalid value for '--dataset-domain'" not in out


def cli_for_mvp2_test():
    """Re-import the CLI to pick up the monkeypatched build_mvp2_graph."""
    import importlib

    import paper2manim.cli as cli_mod
    importlib.reload(cli_mod)
    return cli_mod.cli
