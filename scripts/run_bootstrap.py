#!/usr/bin/env python3
"""Drive a bootstrap batch of paper-section runs to populate the EMB.

This is the practical embodiment of MVP 3.0's cold-start phase (proposal §4.0):
the EMB starts empty, and we let it self-learn by running a curated batch of
arXiv sections through the full pipeline with ``--emb`` enabled.

Usage:

    # Inline list
    python scripts/run_bootstrap.py --tasks 1706.03762:Background 1810.04805:Introduction \\
        --quality l --max-retries 2 --max-visual-revisions 2 --vlm

    # CSV file (one ``arxiv_id,section`` per line; ``#`` for comments)
    python scripts/run_bootstrap.py --tasks-csv examples/bootstrap_v1.csv --vlm

The script invokes the existing ``paper2manim mvp2`` CLI in-process via
``CliRunner``, so all flags except ``--pdf`` / ``--arxiv`` / ``--section`` pass
through directly.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from click.testing import CliRunner

from paper2manim.cli import cli as paper2manim_cli
from paper2manim.config.env import settings

log = logging.getLogger("bootstrap")


@dataclass
class TaskSpec:
    arxiv_id: str
    section: str | None = None

    @classmethod
    def from_inline(cls, raw: str) -> TaskSpec:
        """Parse ``ID`` or ``ID:Section`` syntax."""
        if ":" in raw:
            a, s = raw.split(":", 1)
            return cls(arxiv_id=a.strip(), section=s.strip() or None)
        return cls(arxiv_id=raw.strip())

    @classmethod
    def from_csv_row(cls, row: list[str]) -> TaskSpec:
        a = row[0].strip()
        s = row[1].strip() if len(row) > 1 else ""
        return cls(arxiv_id=a, section=s or None)


_RUN_ID_RE = __import__("re").compile(r"^RUN_ID=([\w\-]+)\s*$", __import__("re").MULTILINE)


def parse_run_id(stdout: str) -> str | None:
    """Extract ``run_id`` from the ``RUN_ID=<id>`` marker line emitted by
    ``paper2manim mvp1`` / ``mvp2``. Returns ``None`` if not found (e.g.,
    when the CLI exited before the marker, or this is a pre-marker version)."""
    if not stdout:
        return None
    m = _RUN_ID_RE.search(stdout)
    return m.group(1) if m else None


@dataclass
class TaskOutcome:
    arxiv_id: str
    section: str | None
    exit_code: int
    duration_s: float
    stdout_tail: str
    run_id: str | None = None  # extracted from RUN_ID=... marker; None if absent
    error: str | None = None


@dataclass
class BootstrapReport:
    started_at: float
    finished_at: float = 0.0
    tasks: list[TaskOutcome] = field(default_factory=list)

    def summary(self) -> dict:
        n_ok = sum(1 for t in self.tasks if t.exit_code == 0)
        return {
            "n_total": len(self.tasks),
            "n_ok": n_ok,
            "n_failed": len(self.tasks) - n_ok,
            "total_duration_s": round(self.finished_at - self.started_at, 1),
            "tasks": [
                {
                    "arxiv_id": t.arxiv_id,
                    "section": t.section,
                    "ok": t.exit_code == 0,
                    "duration_s": round(t.duration_s, 1),
                    "run_id": t.run_id,
                }
                for t in self.tasks
            ],
        }


def load_tasks(args: argparse.Namespace) -> list[TaskSpec]:
    tasks: list[TaskSpec] = []
    if args.tasks_csv:
        path = Path(args.tasks_csv)
        if not path.exists():
            raise SystemExit(f"tasks CSV not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            for row in csv.reader(f):
                if not row or row[0].lstrip().startswith("#"):
                    continue
                tasks.append(TaskSpec.from_csv_row(row))
    for raw in args.tasks or []:
        tasks.append(TaskSpec.from_inline(raw))
    if not tasks:
        raise SystemExit("no tasks specified (use --tasks-csv or --tasks)")
    return tasks


def build_cli_args(task: TaskSpec, args: argparse.Namespace) -> list[str]:
    """Translate a TaskSpec + global args into a paper2manim mvp2 CLI argv."""
    out = ["mvp2", "--arxiv", task.arxiv_id]
    if task.section:
        out.extend(["--section", task.section])
    if args.quality:
        out.extend(["--quality", args.quality])
    if args.max_retries is not None:
        out.extend(["--max-retries", str(args.max_retries)])
    if args.no_render:
        out.append("--no-render")
    if args.allow_render_on_login:
        out.append("--allow-render-on-login")
    if args.vlm:
        out.append("--vlm")
    out.extend(["--max-visual-revisions", str(args.max_visual_revisions)])
    # EMB flags — always-on for bootstrap, that's the whole point.
    out.append("--emb")
    if args.emb_store_path:
        out.extend(["--emb-store-path", args.emb_store_path])
    out.extend(["--emb-theta-high", str(args.emb_theta_high)])
    if args.emb_llm_distill:
        out.append("--emb-llm-distill")
    if args.emb_fake_embedder:
        out.append("--emb-fake-embedder")
    return out


def run_one(task: TaskSpec, args: argparse.Namespace) -> TaskOutcome:
    cli_args = build_cli_args(task, args)
    runner = CliRunner()
    log.info("[bootstrap] starting %s :: %s", task.arxiv_id, task.section or "<all>")
    t0 = time.time()
    try:
        result = runner.invoke(paper2manim_cli, cli_args, catch_exceptions=True)
    except Exception as exc:  # noqa: BLE001
        return TaskOutcome(
            arxiv_id=task.arxiv_id,
            section=task.section,
            exit_code=2,
            duration_s=time.time() - t0,
            stdout_tail="",
            error=f"runner raised: {type(exc).__name__}: {exc}",
        )
    duration = time.time() - t0
    tail = (result.output or "")[-1200:]
    err = None
    if result.exit_code != 0:
        err = f"exit {result.exit_code}"
        if result.exception is not None:
            err += f" ({type(result.exception).__name__}: {result.exception})"
    log.info(
        "[bootstrap] done %s :: %s (exit=%d, %.1fs)",
        task.arxiv_id,
        task.section or "<all>",
        result.exit_code,
        duration,
    )
    return TaskOutcome(
        arxiv_id=task.arxiv_id,
        section=task.section,
        exit_code=int(result.exit_code),
        duration_s=duration,
        stdout_tail=tail,
        run_id=parse_run_id(result.output or ""),
        error=err,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--tasks",
        nargs="*",
        help="Inline tasks (ID or ID:Section). Combined with --tasks-csv if both given.",
    )
    parser.add_argument("--tasks-csv", help="Path to CSV with arxiv_id,section per row.")
    parser.add_argument("--quality", choices=["l", "m", "h"], default="l")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--allow-render-on-login", action="store_true")
    parser.add_argument("--vlm", action="store_true")
    parser.add_argument("--max-visual-revisions", type=int, default=2)
    parser.add_argument("--emb-store-path", default=None)
    parser.add_argument("--emb-theta-high", type=float, default=85.0,
                        help="0-100 scale; matches paper2manim mvp2 default.")
    parser.add_argument("--emb-llm-distill", action="store_true")
    parser.add_argument("--emb-fake-embedder", action="store_true")
    parser.add_argument(
        "--report-path",
        default=None,
        help="Where to write the JSON summary. Defaults to runs/bootstrap_<ts>.json.",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    tasks = load_tasks(args)
    log.info("[bootstrap] %d tasks queued", len(tasks))
    report = BootstrapReport(started_at=time.time())
    for i, task in enumerate(tasks, start=1):
        log.info("[bootstrap] (%d/%d) %s", i, len(tasks), task.arxiv_id)
        report.tasks.append(run_one(task, args))
    report.finished_at = time.time()

    report_path = (
        Path(args.report_path)
        if args.report_path
        else Path(settings.PAPER2MANIM_RUNS_DIR)
        / f"bootstrap_{time.strftime('%Y%m%d-%H%M%S')}.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report.summary(), indent=2), encoding="utf-8")
    summary = report.summary()
    log.info(
        "[bootstrap] DONE: %d/%d ok in %ss (report=%s)",
        summary["n_ok"],
        summary["n_total"],
        summary["total_duration_s"],
        report_path,
    )
    return 0 if summary["n_failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
