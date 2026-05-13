#!/usr/bin/env python3
"""Drive a controlled experiment over (config × seed × task) triples.

This is the proposal §5 RQ1 / §8.3 driver. It runs the same task list under
several named configurations (``A`` zero-shot / ``B`` reflection-only /
``C`` reflection + EMB; plus §8.3 ablations defined in
``paper2manim.ablations``) and writes a single ``manifest.json`` linking
each run_id back to ``(config, seed, task_idx)``. Downstream
``scripts/plot_evolution.py`` reads that manifest to draw per-config curves
with confidence bands.

Usage:

    # Inline tasks
    python scripts/run_experiment.py \
      --tasks 1706.03762:Background 1810.04805:Introduction \
      --configs A,B,C --seeds 1,2 \
      --quality l --max-retries 2 --max-visual-revisions 2 \
      --emb-store-base runs/_emb_exp1 \
      --out-dir runs/exp1

    # CSV-driven (recommended for experiments)
    python scripts/run_experiment.py \
      --tasks-csv examples/datasets/p2m_v1.csv \
      --configs A,B,C --seeds 1,2,3 \
      --out-dir runs/exp_main

    # Smoke (no LLM): proves the runner wires up without burning credits
    python scripts/run_experiment.py --tasks 1706.03762:Background --configs A \
      --dry-run --out-dir runs/exp_smoke

The dry-run mode synthesizes a fake ``run_id``, skips the actual CLI invoke,
and still writes a complete manifest — useful for CI and for validating
the runner's wiring before a long real experiment.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from click.testing import CliRunner

from paper2manim.ablations import known_presets
from paper2manim.ablations import resolve as resolve_preset
from paper2manim.cli import cli as paper2manim_cli

# Re-use bootstrap's RUN_ID parser so both drivers stay in lockstep.
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from run_bootstrap import parse_run_id  # type: ignore[no-redef]
except Exception:  # noqa: BLE001
    # Fallback: inline copy. Keeps this script importable when scripts/ isn't
    # on sys.path (e.g., when launched from inside an IDE that strips it).
    import re as _re

    _RUN_ID_RE = _re.compile(r"^RUN_ID=([\w\-]+)\s*$", _re.MULTILINE)

    def parse_run_id(stdout: str) -> str | None:  # type: ignore[no-redef]
        if not stdout:
            return None
        m = _RUN_ID_RE.search(stdout)
        return m.group(1) if m else None


log = logging.getLogger("run_experiment")


# --------------------------------------------------------------------------- #
# Task spec
# --------------------------------------------------------------------------- #


@dataclass
class TaskSpec:
    arxiv_id: str
    section: str | None = None
    domain: str = ""  # populated by --tasks-csv if the column is present

    @classmethod
    def from_inline(cls, raw: str) -> TaskSpec:
        if ":" in raw:
            a, s = raw.split(":", 1)
            return cls(arxiv_id=a.strip(), section=s.strip() or None)
        return cls(arxiv_id=raw.strip())

    @classmethod
    def from_csv_row(cls, row: dict[str, str]) -> TaskSpec:
        # Tolerate either ``arxiv_id`` or ``id`` headers.
        a = (row.get("arxiv_id") or row.get("id") or "").strip()
        s = (row.get("section") or "").strip()
        d = (row.get("domain") or "").strip()
        if not a:
            raise ValueError(f"task row missing arxiv_id: {row!r}")
        return cls(arxiv_id=a, section=s or None, domain=d)


def load_tasks(args: argparse.Namespace) -> list[TaskSpec]:
    tasks: list[TaskSpec] = []
    if args.tasks_csv:
        path = Path(args.tasks_csv)
        if not path.exists():
            raise SystemExit(f"tasks CSV not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                # Skip comment-style rows (header line starting with #).
                if not row or any(
                    (k or "").lstrip().startswith("#") for k in row.keys() if k
                ):
                    continue
                tasks.append(TaskSpec.from_csv_row(row))
    for raw in args.tasks or []:
        tasks.append(TaskSpec.from_inline(raw))
    if not tasks:
        raise SystemExit("no tasks specified (use --tasks-csv or --tasks)")
    return tasks


# --------------------------------------------------------------------------- #
# Run record + manifest
# --------------------------------------------------------------------------- #


@dataclass
class RunOutcome:
    config: str
    seed: int
    task_idx: int
    arxiv_id: str
    section: str | None
    domain: str
    run_id: str | None
    exit_code: int
    started_at: float
    finished_at: float
    error: str | None = None

    @property
    def duration_s(self) -> float:
        return round(self.finished_at - self.started_at, 2)

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["duration_s"] = self.duration_s
        return d


@dataclass
class ExperimentManifest:
    """Top-level artifact written to ``<out_dir>/manifest.json``.

    Every field is JSON-serializable so downstream consumers (plot_evolution,
    notebooks, dataset cards) can load this with ``json.load`` and rely on
    the schema without round-tripping through pydantic.
    """

    experiment_name: str
    started_at: float
    finished_at: float = 0.0
    configs: list[str] = field(default_factory=list)
    seeds: list[int] = field(default_factory=list)
    tasks: list[dict[str, Any]] = field(default_factory=list)
    runs: list[RunOutcome] = field(default_factory=list)
    cli_args: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": round(self.finished_at - self.started_at, 2),
            "configs": list(self.configs),
            "seeds": list(self.seeds),
            "tasks": list(self.tasks),
            "runs": [r.to_dict() for r in self.runs],
            "cli_args": dict(self.cli_args),
        }


def write_manifest(manifest: ExperimentManifest, out_path: Path) -> Path:
    """Atomically overwrite ``out_path`` with the manifest JSON.

    Atomic so an interrupted experiment can't leave a half-written file that
    later crashes ``plot_evolution.py``.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    tmp.replace(out_path)
    return out_path


# --------------------------------------------------------------------------- #
# CLI arg construction per (config, seed, task)
# --------------------------------------------------------------------------- #


def build_cli_args(
    config: str,
    seed: int,
    task: TaskSpec,
    args: argparse.Namespace,
    *,
    emb_store_path: Path | None,
) -> list[str]:
    """Translate one (config, seed, task) triple into a ``paper2manim mvp2`` argv."""
    out: list[str] = ["mvp2", "--arxiv", task.arxiv_id]
    if task.section:
        out.extend(["--section", task.section])
    out.extend(["--quality", args.quality])
    if args.max_retries is not None:
        out.extend(["--max-retries", str(args.max_retries)])
    if args.no_render:
        out.append("--no-render")
    if args.allow_render_on_login:
        out.append("--allow-render-on-login")
    out.extend(["--max-visual-revisions", str(args.max_visual_revisions)])
    out.extend(resolve_preset(config))
    if "--emb" in out and emb_store_path is not None:
        out.extend(["--emb-store-path", str(emb_store_path)])
    if args.emb_theta_high is not None and "--emb" in out:
        out.extend(["--emb-theta-high", str(args.emb_theta_high)])
    if args.emb_failure_min_margin is not None and "--emb" in out:
        out.extend(["--emb-failure-min-margin", str(args.emb_failure_min_margin)])
    if args.emb_fake_embedder and "--emb" in out:
        out.append("--emb-fake-embedder")
    return out


def emb_path_for(args: argparse.Namespace, config: str, seed: int) -> Path | None:
    """Per-(seed, config) EMB path so configs / seeds don't pollute each other.

    We only carve out a directory for configs that actually use the EMB; ``A``
    and ``B`` get ``None`` so the CLI uses its no-store default (and
    ``--emb`` isn't in their flag list anyway).
    """
    if "--emb" not in resolve_preset(config):
        return None
    base = Path(args.emb_store_base) if args.emb_store_base else None
    if base is None:
        return None
    return base / f"{config}_seed_{seed}"


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


def _invoke_one(
    cli_args: list[str], *, dry_run: bool, runner: CliRunner | None = None
) -> tuple[int, str, str | None]:
    """Run one ``paper2manim`` invocation.

    Returns ``(exit_code, stdout, error_str)``. In dry-run mode synthesizes a
    fake stdout containing a ``RUN_ID=dryrun-<ts>-<rand>`` marker so the rest
    of the pipeline stays exercised.
    """
    if dry_run:
        import secrets
        import time as _t
        fake = f"dryrun-{_t.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"
        return 0, f"RUN_ID={fake}\n[dry-run] would have invoked: {' '.join(cli_args)}\n", None

    rn = runner or CliRunner()
    try:
        result = rn.invoke(paper2manim_cli, cli_args, catch_exceptions=True)
    except Exception as exc:  # noqa: BLE001
        return 2, "", f"runner raised: {type(exc).__name__}: {exc}"
    err = None
    if result.exit_code != 0:
        err = f"exit {result.exit_code}"
        if result.exception is not None:
            err += f" ({type(result.exception).__name__}: {result.exception})"
    return int(result.exit_code), result.output or "", err


def run_one(
    config: str,
    seed: int,
    task_idx: int,
    task: TaskSpec,
    args: argparse.Namespace,
) -> RunOutcome:
    cli_args = build_cli_args(
        config, seed, task, args, emb_store_path=emb_path_for(args, config, seed)
    )
    log.info(
        "[experiment] %s | seed=%d | task[%d]=%s§%s",
        config, seed, task_idx, task.arxiv_id, task.section or "<all>",
    )
    started = time.time()
    exit_code, output, err = _invoke_one(cli_args, dry_run=args.dry_run)
    finished = time.time()
    rid = parse_run_id(output)
    log.info(
        "[experiment] done %s seed=%d task[%d] exit=%d (%.1fs) run_id=%s",
        config, seed, task_idx, exit_code, finished - started, rid or "<none>",
    )
    return RunOutcome(
        config=config,
        seed=seed,
        task_idx=task_idx,
        arxiv_id=task.arxiv_id,
        section=task.section,
        domain=task.domain,
        run_id=rid,
        exit_code=exit_code,
        started_at=started,
        finished_at=finished,
        error=err,
    )


# --------------------------------------------------------------------------- #
# main()
# --------------------------------------------------------------------------- #


def _comma_split(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tasks", nargs="*", help="Inline tasks (ID or ID:Section).")
    parser.add_argument(
        "--tasks-csv",
        help="CSV with header row containing arxiv_id, optional section, domain.",
    )
    parser.add_argument(
        "--configs",
        default="A,B,C",
        help=f"Comma-separated preset names. Known: {', '.join(known_presets())}",
    )
    parser.add_argument(
        "--seeds",
        default="1",
        help="Comma-separated integer seeds, e.g. '1,2,3' for n=3 replicates per config.",
    )
    parser.add_argument("--quality", choices=["l", "m", "h"], default="l")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--max-visual-revisions", type=int, default=2)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--allow-render-on-login", action="store_true")
    parser.add_argument(
        "--emb-store-base",
        default=None,
        help="Directory under which per-(config,seed) EMB stores are created. "
        "Only used by configs that have --emb in their preset.",
    )
    parser.add_argument("--emb-theta-high", type=float, default=None)
    parser.add_argument("--emb-failure-min-margin", type=float, default=None)
    parser.add_argument("--emb-fake-embedder", action="store_true")
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for manifest.json (and any future per-config rollups).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't actually run paper2manim mvp2; synthesize fake run_ids and "
        "still write a full manifest. Useful for CI and runner-wiring smoke tests.",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    configs = _comma_split(args.configs)
    if not configs:
        raise SystemExit("--configs cannot be empty")
    for c in configs:
        # Resolve early so a typo fails before we burn LLM tokens.
        resolve_preset(c)
    seeds = [int(s) for s in _comma_split(args.seeds)]
    if not seeds:
        raise SystemExit("--seeds cannot be empty")

    tasks = load_tasks(args)
    log.info(
        "[experiment] %d task(s) × %d config(s) × %d seed(s) = %d run(s)",
        len(tasks), len(configs), len(seeds), len(tasks) * len(configs) * len(seeds),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    manifest = ExperimentManifest(
        experiment_name=out_dir.name,
        started_at=time.time(),
        configs=configs,
        seeds=seeds,
        tasks=[
            {"arxiv_id": t.arxiv_id, "section": t.section, "domain": t.domain}
            for t in tasks
        ],
        cli_args=vars(args),
    )

    # Outer loop: config first, then seed, then task. This order keeps an
    # EMB-using seed grow monotonically inside its directory before the next
    # seed starts fresh — important for §4 evolution semantics.
    for config in configs:
        for seed in seeds:
            for task_idx, task in enumerate(tasks):
                outcome = run_one(config, seed, task_idx, task, args)
                manifest.runs.append(outcome)
                # Persist after every run so an interrupted experiment leaves a
                # usable partial manifest behind.
                manifest.finished_at = time.time()
                write_manifest(manifest, manifest_path)

    summary = {
        "n_total": len(manifest.runs),
        "n_ok": sum(1 for r in manifest.runs if r.exit_code == 0),
        "n_with_run_id": sum(1 for r in manifest.runs if r.run_id),
        "duration_s": round(manifest.finished_at - manifest.started_at, 1),
    }
    log.info("[experiment] DONE: %s manifest=%s", summary, manifest_path)
    return 0 if summary["n_ok"] == summary["n_total"] else 1


if __name__ == "__main__":
    sys.exit(main())
