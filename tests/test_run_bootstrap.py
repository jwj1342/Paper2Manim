"""B2 unit tests for ``scripts/run_bootstrap.py`` — RUN_ID stdout parsing."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# scripts/ is not a package; load run_bootstrap by file path.
_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_bootstrap.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_bootstrap", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_bootstrap"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def rb():
    return _load_module()


# --------------------------------------------------------------------------- #
# parse_run_id
# --------------------------------------------------------------------------- #


def test_parse_run_id_extracts_marker_line(rb):
    out = "Some intro\nRUN_ID=20260513-091013-fcb3a7\nMore noise\n"
    assert rb.parse_run_id(out) == "20260513-091013-fcb3a7"


def test_parse_run_id_handles_empty_or_none(rb):
    assert rb.parse_run_id("") is None
    assert rb.parse_run_id(None) is None


def test_parse_run_id_returns_first_match_when_multiple_present(rb):
    """Defensive: if for some reason two RUN_ID lines slip in (e.g., a CLI
    that re-prints), take the first — that's the run_id of the *outer*
    invocation we're tracking."""
    out = "RUN_ID=first-run\n... more output ...\nRUN_ID=second-run\n"
    assert rb.parse_run_id(out) == "first-run"


def test_parse_run_id_ignores_lookalikes(rb):
    """The marker is anchored at line start in MULTILINE mode, so embedded
    matches like 'see RUN_ID=foo for details' do NOT count."""
    out = "log: see RUN_ID=embedded for details\n"
    assert rb.parse_run_id(out) is None


def test_parse_run_id_accepts_rich_run_id_chars(rb):
    """run_id format is YYYYMMDD-HHMMSS-<6hex> in production, but the parser
    accepts the broader \\w+\\-+ class for dryrun/test ids."""
    out = "RUN_ID=dryrun-20260513-091013-abcd12\n"
    assert rb.parse_run_id(out) == "dryrun-20260513-091013-abcd12"


# --------------------------------------------------------------------------- #
# TaskOutcome carries run_id
# --------------------------------------------------------------------------- #


def test_task_outcome_dataclass_has_run_id_field(rb):
    fields = {f.name for f in __import__("dataclasses").fields(rb.TaskOutcome)}
    assert "run_id" in fields, (
        "BootstrapReport summary serializer expects this field; without it the "
        "downstream plot_evolution can't link batches to runs."
    )


def test_summary_includes_run_id(rb):
    """The summary() dict must surface run_id per task so external scripts can
    walk batches without reading the dataclass directly."""
    report = rb.BootstrapReport(started_at=0.0, finished_at=1.0, tasks=[
        rb.TaskOutcome(
            arxiv_id="1706.03762", section="Background",
            exit_code=0, duration_s=12.3, stdout_tail="ok",
            run_id="20260513-091013-abc123",
        ),
    ])
    s = report.summary()
    assert s["tasks"][0]["run_id"] == "20260513-091013-abc123"
    assert s["tasks"][0]["ok"] is True
