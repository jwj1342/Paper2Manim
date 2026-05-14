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


# --------------------------------------------------------------------------- #
# _invoke_one: subprocess plumbing
# --------------------------------------------------------------------------- #


class TestInvokeOneSubprocess:
    """Real (non-dry) ``_invoke_one`` shells out via ``python -m paper2manim``.

    These tests exercise the subprocess-shaped contract via a fake ``runner``
    callable so they stay fast and don't actually fork.
    """

    def _fake_completed(self, *, returncode=0, stdout="", stderr=""):
        import subprocess
        return subprocess.CompletedProcess(
            args=[], returncode=returncode, stdout=stdout, stderr=stderr,
        )

    def test_invokes_python_dash_m_paper2manim(self, re_mod):
        captured: dict = {}

        def fake_runner(cmd, **kwargs):
            captured["cmd"] = cmd
            captured["kwargs"] = kwargs
            return self._fake_completed(stdout="RUN_ID=fake-1\n")

        exit_code, output, err = re_mod._invoke_one(
            ["mvp2", "--arxiv", "1706.03762"],
            dry_run=False, runner=fake_runner,
        )
        assert exit_code == 0
        assert "RUN_ID=fake-1" in output
        assert err is None
        # Subprocess form: [python, -m, paper2manim, ...cli_args]
        assert captured["cmd"][1:3] == ["-m", "paper2manim"]
        assert captured["cmd"][3:] == ["mvp2", "--arxiv", "1706.03762"]
        assert captured["kwargs"]["capture_output"] is True
        assert captured["kwargs"]["text"] is True

    def test_timeout_returns_124_with_partial_stdout(self, re_mod):
        import subprocess

        def fake_runner(cmd, **kwargs):
            raise subprocess.TimeoutExpired(
                cmd=cmd, timeout=5.0, output="RUN_ID=part-1\nstart...\n",
            )

        exit_code, output, err = re_mod._invoke_one(
            ["mvp2", "--arxiv", "x"],
            dry_run=False, timeout_s=5.0, runner=fake_runner,
        )
        assert exit_code == 124, "should mirror GNU timeout's exit code"
        assert "RUN_ID=part-1" in output, "partial stdout must survive timeout"
        assert "timeout after 5.0s" in (err or "")

    def test_timeout_with_bytes_stdout_decodes(self, re_mod):
        """Some Python versions hand back bytes in TimeoutExpired.stdout when
        ``text=True`` is bypassed by an early kill; we should not crash."""
        import subprocess

        def fake_runner(cmd, **kwargs):
            raise subprocess.TimeoutExpired(
                cmd=cmd, timeout=1.0, output=b"RUN_ID=b\xc3\xa9\n",
            )

        exit_code, output, err = re_mod._invoke_one(
            ["mvp2", "--arxiv", "x"],
            dry_run=False, timeout_s=1.0, runner=fake_runner,
        )
        assert exit_code == 124
        # Latin-1-ish bytes round-tripped without ascii decode error
        assert "RUN_ID=b" in output

    def test_executable_not_found_returns_127(self, re_mod):
        def fake_runner(cmd, **kwargs):
            raise FileNotFoundError(2, "No such file or directory", cmd[0])

        exit_code, _, err = re_mod._invoke_one(
            ["mvp2"], dry_run=False, runner=fake_runner,
        )
        assert exit_code == 127
        assert "executable not found" in (err or "")

    def test_unexpected_runner_exception_returns_2(self, re_mod):
        def fake_runner(cmd, **kwargs):
            raise RuntimeError("disk full")

        exit_code, _, err = re_mod._invoke_one(
            ["mvp2"], dry_run=False, runner=fake_runner,
        )
        assert exit_code == 2
        assert "RuntimeError" in (err or "")

    def test_nonzero_exit_captures_stderr_tail_in_err(self, re_mod):
        def fake_runner(cmd, **kwargs):
            return self._fake_completed(
                returncode=1,
                stdout="RUN_ID=t-2\n",
                stderr="Traceback (most recent call last):\n... RuntimeError: synthetic\n",
            )

        exit_code, output, err = re_mod._invoke_one(
            ["mvp2"], dry_run=False, runner=fake_runner,
        )
        assert exit_code == 1
        assert "RUN_ID=t-2" in output
        assert err is not None and "exit 1" in err
        assert "synthetic" in err, (
            "stderr tail should be captured into err so the manifest carries "
            "actionable diagnostics, not just an exit code"
        )

    def test_run_one_threads_per_task_timeout(self, re_mod, monkeypatch):
        """Plumbing: ``per_task_timeout`` from args reaches ``_invoke_one``."""
        import argparse

        captured: dict = {}

        def fake_invoke(cli_args, *, dry_run, timeout_s=None, **_):
            captured["timeout_s"] = timeout_s
            captured["dry_run"] = dry_run
            return 0, "RUN_ID=plumb-1\n", None

        monkeypatch.setattr(re_mod, "_invoke_one", fake_invoke)

        ns = argparse.Namespace(
            quality="l", max_retries=2, no_render=True,
            allow_render_on_login=False, max_visual_revisions=2,
            emb_store_base=None, emb_theta_high=None,
            emb_failure_min_margin=None, emb_fake_embedder=False,
            dry_run=False, per_task_timeout=600.0,
        )
        outcome = re_mod.run_one(
            "A", 1, 0, re_mod.TaskSpec(arxiv_id="1706.03762"), ns,
        )
        assert outcome.exit_code == 0
        assert captured["timeout_s"] == 600.0
        assert captured["dry_run"] is False

    def test_run_one_default_no_timeout_when_arg_missing(self, re_mod, monkeypatch):
        """Backwards-compat: a Namespace without ``per_task_timeout`` (older
        callers) must not crash; ``_invoke_one`` should see ``timeout_s=None``."""
        import argparse

        captured: dict = {}

        def fake_invoke(cli_args, *, dry_run, timeout_s=None, **_):
            captured["timeout_s"] = timeout_s
            return 0, "RUN_ID=p\n", None

        monkeypatch.setattr(re_mod, "_invoke_one", fake_invoke)
        ns = argparse.Namespace(
            quality="l", max_retries=2, no_render=True,
            allow_render_on_login=False, max_visual_revisions=2,
            emb_store_base=None, emb_theta_high=None,
            emb_failure_min_margin=None, emb_fake_embedder=False,
            dry_run=False,  # no per_task_timeout attr at all
        )
        re_mod.run_one("A", 1, 0, re_mod.TaskSpec(arxiv_id="x"), ns)
        assert captured["timeout_s"] is None


# --------------------------------------------------------------------------- #
# Real subprocess smoke test — proves the wiring isn't silently dead
# --------------------------------------------------------------------------- #


class TestRealSubprocessSmoke:
    """Reviewer's ask (PR #23): a real non-dry-run smoke test.

    We invoke ``paper2manim info`` (no LLM, no Manim, < 1s). It exercises:
    - ``paper2manim/__main__.py`` → ``cli()`` resolution
    - ``subprocess.run`` argv plumbing + capture_output + text decoding
    - exit-code propagation
    All four are the non-dry codepath that prior CI never covered.
    """

    def test_python_dash_m_paper2manim_info_returns_json(self, re_mod):
        exit_code, stdout, err = re_mod._invoke_one(
            ["info"], dry_run=False, timeout_s=30.0,
        )
        assert exit_code == 0, f"err={err!r}\nstdout={stdout!r}"
        # ``info`` prints a JSON object with PAPER2MANIM_RUNS_DIR.
        assert "PAPER2MANIM_RUNS_DIR" in stdout
        assert err is None

    def test_python_dash_m_paper2manim_unknown_command_nonzero_with_stderr(
        self, re_mod
    ):
        exit_code, _stdout, err = re_mod._invoke_one(
            ["this-command-does-not-exist"], dry_run=False, timeout_s=30.0,
        )
        # Click exits 2 for usage errors. Either way, must be nonzero and
        # carry diagnostic stderr — never silently exit 0.
        assert exit_code != 0
        assert err is not None and ("exit" in err)
