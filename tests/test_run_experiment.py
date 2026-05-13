"""B2 unit tests for ``scripts/run_experiment.py``.

Cover the runner's wiring layer:

1. ``ABLATION_PRESETS`` resolves the three core configs and the ablation names.
2. ``build_cli_args`` injects per-config flags + per-(seed,config) EMB path.
3. ``run_one`` in dry-run mode emits a synthetic RUN_ID and full RunOutcome.
4. ``main`` writes a complete manifest with N = len(tasks)*len(configs)*len(seeds)
   runs and overwrites cleanly when re-run.

We never actually invoke the real ``paper2manim mvp2`` here — either ``--dry-run``
or a CliRunner monkeypatch — so the tests are fast and don't need an LLM.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from paper2manim.ablations import ABLATION_PRESETS, known_presets, resolve

# scripts/ isn't a package; load by file path.
_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_experiment.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_experiment", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_experiment"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def re_mod():
    return _load_module()


# --------------------------------------------------------------------------- #
# ABLATION_PRESETS
# --------------------------------------------------------------------------- #


class TestPresets:
    def test_three_core_configs_present(self):
        for k in ("A", "B", "C"):
            assert k in ABLATION_PRESETS, f"core proposal §5 RQ1 config {k!r} missing"

    def test_a_is_zero_shot(self):
        assert resolve("A") == ("--no-vlm", "--no-emb")

    def test_b_is_reflection_only(self):
        assert resolve("B") == ("--vlm", "--no-emb")

    def test_c_is_full_proposal(self):
        assert resolve("C") == ("--vlm", "--emb")

    def test_unknown_preset_raises(self):
        with pytest.raises(KeyError, match="Unknown ablation preset"):
            resolve("Z_does_not_exist")

    def test_known_presets_includes_ablations(self):
        names = known_presets()
        assert "C_no_success_channel" in names
        assert "C_no_failure_channel" in names


# --------------------------------------------------------------------------- #
# build_cli_args
# --------------------------------------------------------------------------- #


class TestBuildCliArgs:
    def _args(self, **overrides):
        # Mimic argparse.Namespace shape used by main().
        import argparse
        defaults = dict(
            quality="l",
            max_retries=2,
            no_render=False,
            allow_render_on_login=False,
            max_visual_revisions=2,
            emb_store_base=None,
            emb_theta_high=None,
            emb_failure_min_margin=None,
            emb_fake_embedder=False,
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_config_a_no_emb_no_vlm(self, re_mod):
        task = re_mod.TaskSpec(arxiv_id="1706.03762", section="Background")
        argv = re_mod.build_cli_args("A", 1, task, self._args(), emb_store_path=None)
        assert "--no-vlm" in argv
        assert "--no-emb" in argv
        assert "--emb" not in argv  # not the substring; check exact tokens
        assert "--emb-store-path" not in argv

    def test_config_b_has_vlm_no_emb(self, re_mod):
        task = re_mod.TaskSpec(arxiv_id="1706.03762")
        argv = re_mod.build_cli_args("B", 1, task, self._args(), emb_store_path=None)
        assert "--vlm" in argv
        assert "--no-emb" in argv

    def test_config_c_uses_emb_store_path_when_provided(self, re_mod, tmp_path):
        task = re_mod.TaskSpec(arxiv_id="1706.03762", section="Method")
        store = tmp_path / "emb_seed_1"
        argv = re_mod.build_cli_args("C", 1, task, self._args(), emb_store_path=store)
        assert "--vlm" in argv
        assert "--emb" in argv
        # path is paired with its flag
        i = argv.index("--emb-store-path")
        assert argv[i + 1] == str(store)

    def test_config_c_emb_thresholds_pass_through(self, re_mod, tmp_path):
        task = re_mod.TaskSpec(arxiv_id="1706.03762")
        argv = re_mod.build_cli_args(
            "C", 1, task,
            self._args(emb_theta_high=80.0, emb_failure_min_margin=5.0),
            emb_store_path=tmp_path / "x",
        )
        i = argv.index("--emb-theta-high")
        assert argv[i + 1] == "80.0"
        i = argv.index("--emb-failure-min-margin")
        assert argv[i + 1] == "5.0"

    def test_config_a_ignores_emb_thresholds(self, re_mod):
        """A doesn't have --emb in its preset; threshold flags should not
        leak in even if the user passed them globally."""
        task = re_mod.TaskSpec(arxiv_id="1706.03762")
        argv = re_mod.build_cli_args(
            "A", 1, task, self._args(emb_theta_high=80.0), emb_store_path=None
        )
        assert "--emb-theta-high" not in argv

    def test_section_optional(self, re_mod):
        task = re_mod.TaskSpec(arxiv_id="1706.03762", section=None)
        argv = re_mod.build_cli_args("A", 1, task, self._args(), emb_store_path=None)
        assert "--section" not in argv

    def test_no_render_passthrough(self, re_mod):
        task = re_mod.TaskSpec(arxiv_id="1706.03762")
        argv = re_mod.build_cli_args(
            "A", 1, task, self._args(no_render=True), emb_store_path=None
        )
        assert "--no-render" in argv

    def test_emb_path_for_returns_none_for_no_emb_configs(self, re_mod):
        import argparse
        ns = argparse.Namespace(emb_store_base="runs/_emb_exp1")
        assert re_mod.emb_path_for(ns, "A", 1) is None
        assert re_mod.emb_path_for(ns, "B", 1) is None

    def test_emb_path_for_partitions_by_config_and_seed(self, re_mod, tmp_path):
        import argparse
        ns = argparse.Namespace(emb_store_base=str(tmp_path / "emb"))
        a = re_mod.emb_path_for(ns, "C", 1)
        b = re_mod.emb_path_for(ns, "C", 2)
        c = re_mod.emb_path_for(ns, "C_no_success_channel", 1)
        assert a != b != c
        assert "C_seed_1" in str(a)
        assert "C_seed_2" in str(b)
        assert "C_no_success_channel_seed_1" in str(c)


# --------------------------------------------------------------------------- #
# Dry-run end-to-end: main() writes a manifest without invoking real CLI
# --------------------------------------------------------------------------- #


class TestMainDryRun:
    def test_main_writes_full_manifest_in_dry_run(self, re_mod, tmp_path):
        out_dir = tmp_path / "exp_test"
        rc = re_mod.main([
            "--tasks", "1706.03762:Background", "1810.04805:Intro",
            "--configs", "A,B,C",
            "--seeds", "1,2",
            "--quality", "l",
            "--max-retries", "2",
            "--max-visual-revisions", "2",
            "--out-dir", str(out_dir),
            "--dry-run",
        ])
        assert rc == 0
        manifest = json.loads((out_dir / "manifest.json").read_text())

        # 2 tasks × 3 configs × 2 seeds = 12 runs
        assert manifest["experiment_name"] == "exp_test"
        assert manifest["configs"] == ["A", "B", "C"]
        assert manifest["seeds"] == [1, 2]
        assert len(manifest["tasks"]) == 2
        assert len(manifest["runs"]) == 12
        # Every run should carry a (synthetic) run_id in dry-run mode
        for run in manifest["runs"]:
            assert run["run_id"] is not None
            assert run["run_id"].startswith("dryrun-")
            assert run["exit_code"] == 0
            assert run["config"] in ("A", "B", "C")
            assert run["seed"] in (1, 2)
            assert run["task_idx"] in (0, 1)

    def test_main_overwrites_existing_manifest(self, re_mod, tmp_path):
        out_dir = tmp_path / "exp_overwrite"
        argv = [
            "--tasks", "1706.03762:Background",
            "--configs", "A",
            "--seeds", "1",
            "--out-dir", str(out_dir),
            "--dry-run",
        ]
        re_mod.main(argv)
        first = json.loads((out_dir / "manifest.json").read_text())

        re_mod.main(argv)  # second run with same args
        second = json.loads((out_dir / "manifest.json").read_text())

        # Second invocation overwrites the file, doesn't append: same shape,
        # different (synthetic) run_ids and timestamps.
        assert len(first["runs"]) == 1
        assert len(second["runs"]) == 1
        assert first["runs"][0]["run_id"] != second["runs"][0]["run_id"]

    def test_main_rejects_unknown_config(self, re_mod, tmp_path):
        out_dir = tmp_path / "exp_bad"
        with pytest.raises(KeyError, match="Unknown ablation preset"):
            re_mod.main([
                "--tasks", "1706.03762:Background",
                "--configs", "X_unknown",
                "--out-dir", str(out_dir),
                "--dry-run",
            ])

    def test_csv_loading_with_domain_column(self, re_mod, tmp_path):
        csv_path = tmp_path / "tasks.csv"
        csv_path.write_text(
            "arxiv_id,section,domain\n"
            "1706.03762,Background,cs\n"
            "1810.04805,Introduction,cs\n",
            encoding="utf-8",
        )
        out_dir = tmp_path / "exp_csv"
        rc = re_mod.main([
            "--tasks-csv", str(csv_path),
            "--configs", "A",
            "--seeds", "1",
            "--out-dir", str(out_dir),
            "--dry-run",
        ])
        assert rc == 0
        manifest = json.loads((out_dir / "manifest.json").read_text())
        assert manifest["tasks"][0]["domain"] == "cs"
        assert manifest["runs"][0]["domain"] == "cs"
