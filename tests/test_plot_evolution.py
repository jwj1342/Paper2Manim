"""B3 tests for ``scripts/plot_evolution.py`` aggregation layer.

We don't render PNGs here — matplotlib is heavy and rendering quality is
human-judged. Instead we test the data-shape contracts:

* ``read_manifest`` rejects malformed manifests.
* ``aggregate_manifest_by_config`` walks the manifest correctly and discovers
  per-(config, task_idx) groups; missing trace.jsonl files are dropped.
* ``_bootstrap_mean_ci`` returns sensible bounds and degrades gracefully on
  small / empty samples.
* ``build_curves`` converts grouped RunStats into per-config curves.
* End-to-end: ``main(--manifest ...)`` writes a CSV with both aggregated and
  per-run sections.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from paper2manim.artifacts import run_dir as artifact_run_dir

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "plot_evolution.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("plot_evolution", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["plot_evolution"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def pe():
    return _load_module()


# --------------------------------------------------------------------------- #
# Helpers — synthesize a 3-dim 0-100 trace per run
# --------------------------------------------------------------------------- #


def _write_synthetic_trace(
    run_id: str,
    *,
    pass_at_first: bool,
    final_avg: float,
    reflection_iters: int = 0,
) -> str:
    """Lay down a minimal trace.jsonl + attempt files that ``aggregate_run``
    will summarize as the requested per-run quantities.

    The ``scores`` dict carries BOTH the new 3-dim 0-100 keys (B1) AND the
    legacy 6-dim 1-5 keys, with ``final_avg`` plugged into both. This keeps
    the test schema-agnostic: ``_avg_score`` will pick whichever set matches
    its current ``_SCORE_KEYS`` and produce ``final_avg`` either way. So the
    test passes both on a stale main (legacy schema active) and on a
    main that already merged B1 (new schema active) — the assertion
    ``score.mean == final_avg`` holds in both worlds.
    """
    rd = artifact_run_dir(run_id)
    events = []
    for i in range(reflection_iters + 1):
        events.append({
            "node": "render", "scene": "S1",
            "iter": i, "v_rev": 0,
            "status": "success" if i == reflection_iters else "error",
            "category": None,
        })
    avg = int(final_avg)
    events.append({
        "node": "vlm_review", "scene": "S1", "v_rev": 0,
        "decision": "pass" if pass_at_first else "revise",
        "scores": {
            # New (post-B1) 3-dim 0-100 schema
            "logic_flow": avg, "layout_occlusion": avg, "accuracy": avg,
            # Legacy 6-dim 1-5 schema (kept so the test passes on pre-B1 main)
            "paper_alignment": avg, "visual_clarity": avg, "readability": avg,
            "layout_balance": avg, "visual_focus": avg, "animation_perceived": avg,
        },
    })
    trace = rd / "trace.jsonl"
    trace.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")
    (rd / "attempts").mkdir(exist_ok=True)
    (rd / "attempts" / f"{reflection_iters:02d}_S1.py").write_text("# stub", encoding="utf-8")
    return run_id


# --------------------------------------------------------------------------- #
# read_manifest validation
# --------------------------------------------------------------------------- #


class TestReadManifest:
    def test_rejects_missing_file(self, pe, tmp_path):
        with pytest.raises(SystemExit, match="manifest not found"):
            pe.read_manifest(tmp_path / "nope.json")

    def test_rejects_missing_required_fields(self, pe, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"configs": ["A"]}), encoding="utf-8")
        with pytest.raises(SystemExit, match="missing required field"):
            pe.read_manifest(bad)

    def test_accepts_minimal_valid(self, pe, tmp_path):
        good = tmp_path / "good.json"
        good.write_text(json.dumps({
            "configs": ["A"], "seeds": [1], "tasks": [], "runs": [],
        }), encoding="utf-8")
        m = pe.read_manifest(good)
        assert m["configs"] == ["A"]


# --------------------------------------------------------------------------- #
# Bootstrap CI behavior on edge cases
# --------------------------------------------------------------------------- #


class TestBootstrapCI:
    def test_zero_samples_returns_nan_no_ci(self, pe):
        mean, lo, hi = pe._bootstrap_mean_ci([])
        assert mean != mean  # nan
        assert lo is None and hi is None

    def test_one_sample_returns_value_no_ci(self, pe):
        mean, lo, hi = pe._bootstrap_mean_ci([42.0])
        assert mean == 42.0
        assert lo is None and hi is None

    def test_two_samples_produces_ci(self, pe):
        mean, lo, hi = pe._bootstrap_mean_ci([0.4, 0.6], seed=42)
        assert 0.4 <= mean <= 0.6
        assert lo is not None and hi is not None
        assert lo <= mean <= hi
        # CI must be inside the data envelope
        assert 0.4 <= lo and hi <= 0.6

    def test_identical_samples_zero_width_ci(self, pe):
        """All-equal data → CI collapses to the point."""
        mean, lo, hi = pe._bootstrap_mean_ci([5.0, 5.0, 5.0])
        assert mean == 5.0
        assert lo == 5.0 and hi == 5.0


# --------------------------------------------------------------------------- #
# aggregate_manifest_by_config + build_curves
# --------------------------------------------------------------------------- #


class TestAggregateManifest:
    def test_groups_runs_by_config_and_task_idx(self, pe):
        # Three runs: A seed1 task0, A seed2 task0, B seed1 task0
        rid_a1 = _write_synthetic_trace("rid_pe_a1", pass_at_first=True, final_avg=80.0)
        rid_a2 = _write_synthetic_trace("rid_pe_a2", pass_at_first=False, final_avg=70.0)
        rid_b1 = _write_synthetic_trace("rid_pe_b1", pass_at_first=True, final_avg=85.0)
        manifest = {
            "configs": ["A", "B"], "seeds": [1, 2],
            "tasks": [{"arxiv_id": "x", "section": "y", "domain": ""}],
            "runs": [
                {"config": "A", "seed": 1, "task_idx": 0, "run_id": rid_a1},
                {"config": "A", "seed": 2, "task_idx": 0, "run_id": rid_a2},
                {"config": "B", "seed": 1, "task_idx": 0, "run_id": rid_b1},
            ],
            "cli_args": {},
        }
        flat, grouped = pe.aggregate_manifest_by_config(manifest)
        assert len(flat) == 3
        assert "A" in grouped and "B" in grouped
        assert len(grouped["A"][0]) == 2  # both A seeds
        assert len(grouped["B"][0]) == 1

    def test_skips_runs_with_missing_run_id(self, pe):
        manifest = {
            "configs": ["A"], "seeds": [1], "tasks": [],
            "runs": [
                {"config": "A", "seed": 1, "task_idx": 0, "run_id": None},
                {"config": "A", "seed": 1, "task_idx": 0, "run_id": ""},
            ],
            "cli_args": {},
        }
        flat, grouped = pe.aggregate_manifest_by_config(manifest)
        assert flat == []
        assert grouped == {}

    def test_skips_runs_whose_trace_is_missing(self, pe):
        manifest = {
            "configs": ["A"], "seeds": [1], "tasks": [],
            "runs": [{"config": "A", "seed": 1, "task_idx": 0,
                      "run_id": "rid_pe_does_not_exist_anywhere"}],
            "cli_args": {},
        }
        flat, grouped = pe.aggregate_manifest_by_config(manifest)
        assert flat == []
        assert grouped == {}


class TestBuildCurves:
    def test_curves_have_three_metrics_per_config(self, pe):
        rid = _write_synthetic_trace("rid_pe_curves", pass_at_first=True, final_avg=90.0)
        grouped = {"A": {0: [pe.aggregate_run(rid)]}}
        curves = pe.build_curves(grouped)
        assert "A" in curves
        assert set(curves["A"].keys()) == {
            "pass_at_1", "mean_reflection_iters", "mean_final_score"
        }
        # X axis is task_idx + 1
        assert curves["A"]["pass_at_1"][0].cumulative_idx == 1

    def test_pass_at_1_is_one_when_all_seeds_passed(self, pe):
        rid_a = _write_synthetic_trace("rid_pe_curves_a", pass_at_first=True, final_avg=90.0)
        rid_b = _write_synthetic_trace("rid_pe_curves_b", pass_at_first=True, final_avg=80.0)
        grouped = {"C": {0: [pe.aggregate_run(rid_a), pe.aggregate_run(rid_b)]}}
        curves = pe.build_curves(grouped)
        pa1 = curves["C"]["pass_at_1"][0]
        assert pa1.mean == 1.0
        assert pa1.n_seeds == 2

    def test_pass_at_1_is_half_when_one_of_two_passes(self, pe):
        rid_a = _write_synthetic_trace("rid_pe_curves_p1", pass_at_first=True, final_avg=90.0)
        rid_b = _write_synthetic_trace("rid_pe_curves_p2", pass_at_first=False, final_avg=70.0)
        grouped = {"C": {0: [pe.aggregate_run(rid_a), pe.aggregate_run(rid_b)]}}
        curves = pe.build_curves(grouped)
        pa1 = curves["C"]["pass_at_1"][0]
        assert pa1.mean == 0.5

    def test_score_curves_pick_highest_vlm_avg(self, pe):
        rid = _write_synthetic_trace("rid_pe_curves_score", pass_at_first=True, final_avg=88.0)
        grouped = {"A": {0: [pe.aggregate_run(rid)]}}
        curves = pe.build_curves(grouped)
        score = curves["A"]["mean_final_score"][0]
        # synthetic trace has scores all 88 → avg 88.0
        assert score.mean == pytest.approx(88.0)


# --------------------------------------------------------------------------- #
# main() end-to-end with a manifest
# --------------------------------------------------------------------------- #


class TestMainManifestMode:
    def test_manifest_mode_writes_csv_with_two_sections(self, pe, tmp_path):
        rid = _write_synthetic_trace("rid_pe_main", pass_at_first=True, final_avg=85.0)
        manifest_path = tmp_path / "exp" / "manifest.json"
        manifest_path.parent.mkdir(parents=True)
        manifest_path.write_text(json.dumps({
            "configs": ["C"], "seeds": [1],
            "tasks": [{"arxiv_id": "x", "section": "y", "domain": "cs"}],
            "runs": [{"config": "C", "seed": 1, "task_idx": 0, "run_id": rid}],
            "cli_args": {},
        }), encoding="utf-8")
        out_dir = tmp_path / "out"
        rc = pe.main(["--manifest", str(manifest_path), "--out-dir", str(out_dir)])
        assert rc == 0
        csv_text = (out_dir / "hero_plot.csv").read_text(encoding="utf-8")
        assert "# section: aggregated" in csv_text
        assert "# section: per_run" in csv_text
        assert "C,1,pass_at_1" in csv_text

    def test_manifest_mode_with_zero_runs_does_not_crash(self, pe, tmp_path):
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps({
            "configs": ["A"], "seeds": [1], "tasks": [], "runs": [],
            "cli_args": {},
        }), encoding="utf-8")
        out_dir = tmp_path / "out_empty"
        rc = pe.main(["--manifest", str(manifest_path), "--out-dir", str(out_dir)])
        assert rc == 0
        # CSV gets the section headers even with no data
        assert (out_dir / "hero_plot.csv").exists()


# --------------------------------------------------------------------------- #
# Legacy fallback mode
# --------------------------------------------------------------------------- #


class TestLegacyMode:
    def test_legacy_mode_scans_runs_dir(self, pe, tmp_path):
        # Per the conftest fixture, runs/ is tmp_path/runs. Lay down one run
        # there and confirm the script picks it up.
        _write_synthetic_trace("rid_pe_legacy", pass_at_first=True, final_avg=70.0)
        out_dir = tmp_path / "legacy_out"
        rc = pe.main(["--out-dir", str(out_dir)])
        assert rc == 0
        csv_text = (out_dir / "hero_plot.csv").read_text(encoding="utf-8")
        assert "rid_pe_legacy" in csv_text
