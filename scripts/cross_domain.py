"""Cross-domain EMB freeze experiment driver (proposal §5 RQ3).

Two phases on the same task pool CSV (see ``examples/datasets/p2m_v1_schema.md``):

  1. ``train``    — grow EMB on tasks where ``domain == --train-domain`` and
                    ``split == cross_train``. Writes to ``--emb-store-path``.
  2. ``test``     — freeze that EMB (``--emb-readonly``) and run on tasks where
                    ``domain == --test-domain`` and ``split == cross_test``.
  3. ``baseline`` — same test set but ``--no-emb``; lets you compute the lift
                    attributable to the frozen cross-domain memory.

Each phase writes a JSON report capturing per-task exit code / duration / RUN_ID
(if echoed by the CLI). The script does not aggregate across phases — pair-up is
left to downstream notebooks so we don't bake a particular metric in here.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

_PHASES = ("train", "test", "baseline")


@dataclass
class TaskOutcome:
    arxiv_id: str
    section: str
    domain: str
    exit_code: int
    duration_s: float
    run_id: str = ""
    stderr_tail: str = ""


@dataclass
class PhaseReport:
    phase: str
    config_args: list[str]
    emb_store_path: str
    started_at: float
    finished_at: float = 0.0
    outcomes: list[TaskOutcome] = field(default_factory=list)


def _read_tasks(csv_path: Path, *, domain: str, split: str) -> list[dict]:
    rows: list[dict] = []
    text = "\n".join(
        line for line in csv_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if not text:
        return rows
    reader = csv.DictReader(text.splitlines())
    for r in reader:
        if (r.get("domain") or "").strip() == domain and (r.get("split") or "").strip() == split:
            rows.append(r)
    return rows


def _run_one(
    task: dict,
    *,
    base_args: list[str],
    quality: str,
    dry_run: bool,
    timeout_s: float | None = None,
    runner: Callable[..., subprocess.CompletedProcess] | None = None,
    python: str | None = None,
) -> TaskOutcome:
    """Run one ``paper2manim mvp2`` invocation in an isolated subprocess.

    See ``scripts/run_experiment.py::_invoke_one`` — same rationale: the
    in-process ``click.testing.CliRunner`` form leaks module state across
    tasks, has no crash isolation, no enforceable timeout, and accumulates
    Manim memory. For an experiment driver that runs N×M×K tasks unattended,
    those are real failure modes, not theoretical ones.

    Exit codes on subprocess failures (mirrors GNU/shell convention so reports
    stay readable):

    - ``124`` — :class:`subprocess.TimeoutExpired`
    - ``127`` — interpreter / package not found
    -  ``99`` — any other unexpected runner exception (kept distinct from 1/2
       which the CLI itself emits, so post-hoc triage can tell the two apart)

    The ``runner`` parameter is a test seam: any callable with the
    ``subprocess.run`` signature works; production callers leave it ``None``.
    """
    args = [
        "mvp2",
        "--arxiv", task["arxiv_id"],
        "--section", task["section"],
        "--quality", quality,
        "--dataset-domain", task["domain"],
        *base_args,
    ]
    if dry_run:
        args.append("--no-render")

    started = time.time()
    runner_fn = runner or subprocess.run
    cmd = [python or sys.executable, "-m", "paper2manim", *args]
    out = ""
    err_tail = ""
    try:
        result = runner_fn(
            cmd, capture_output=True, text=True, timeout=timeout_s,
        )
        exit_code = int(result.returncode)
        out = result.stdout or ""
        err_tail = (result.stderr or "").strip()[-300:]
    except subprocess.TimeoutExpired as exc:
        exit_code = 124
        partial = exc.stdout if isinstance(exc.stdout, str) else (
            exc.stdout.decode("utf-8", errors="replace") if exc.stdout else ""
        )
        out = partial or ""
        err_tail = f"timeout after {timeout_s}s"
    except FileNotFoundError as exc:
        exit_code = 127
        err_tail = f"executable not found: {exc}"
    except Exception as exc:  # noqa: BLE001
        exit_code = 99
        err_tail = repr(exc)[-300:]
    duration = time.time() - started

    run_id = ""
    for line in out.splitlines():
        if line.startswith("RUN_ID="):
            run_id = line.split("=", 1)[1].strip()
            break

    return TaskOutcome(
        arxiv_id=task["arxiv_id"],
        section=task["section"],
        domain=task["domain"],
        exit_code=exit_code,
        duration_s=duration,
        run_id=run_id,
        stderr_tail=err_tail,
    )


def _config_for_phase(phase: str, emb_store_path: str) -> list[str]:
    if phase == "train":
        return ["--vlm", "--emb", "--emb-store-path", emb_store_path]
    if phase == "test":
        return ["--vlm", "--emb", "--emb-store-path", emb_store_path, "--emb-readonly"]
    if phase == "baseline":
        return ["--vlm", "--no-emb"]
    raise ValueError(phase)


def run_phase(
    *,
    phase: str,
    tasks: list[dict],
    emb_store_path: Path,
    quality: str,
    dry_run: bool,
    extra_args: list[str] | None = None,
    timeout_s: float | None = None,
) -> PhaseReport:
    if phase not in _PHASES:
        raise ValueError(f"unknown phase {phase!r}; expected one of {_PHASES}")
    base = _config_for_phase(phase, str(emb_store_path)) + list(extra_args or [])
    report = PhaseReport(
        phase=phase,
        config_args=base,
        emb_store_path=str(emb_store_path),
        started_at=time.time(),
    )
    for task in tasks:
        report.outcomes.append(_run_one(
            task, base_args=base, quality=quality,
            dry_run=dry_run, timeout_s=timeout_s,
        ))
    report.finished_at = time.time()
    return report


def _dump_report(report: PhaseReport, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(report)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=list(_PHASES))
    p.add_argument("--tasks-csv", type=Path, required=True)
    p.add_argument("--emb-store-path", type=Path, required=True)
    p.add_argument("--report-out", type=Path, required=True)
    p.add_argument("--train-domain", help="filter tasks for train phase")
    p.add_argument("--test-domain", help="filter tasks for test/baseline phase")
    p.add_argument("--quality", default="l", choices=["l", "m", "h"])
    p.add_argument("--dry-run", action="store_true", help="pass --no-render to skip Manim")
    p.add_argument(
        "--extra-arg", action="append", default=[],
        help="extra arg forwarded to mvp2 (repeatable, e.g. --extra-arg=--max-retries=2)",
    )
    p.add_argument(
        "--per-task-timeout",
        type=float,
        default=None,
        help="Kill each mvp2 invocation after N seconds (default: no limit). "
        "On timeout the task records exit_code=124 and the phase keeps going.",
    )
    args = p.parse_args(argv)

    if args.phase == "train":
        if not args.train_domain:
            print("error: train phase requires --train-domain", file=sys.stderr)
            return 2
        tasks = _read_tasks(args.tasks_csv, domain=args.train_domain, split="cross_train")
    else:
        if not args.test_domain:
            print(f"error: {args.phase} phase requires --test-domain", file=sys.stderr)
            return 2
        tasks = _read_tasks(args.tasks_csv, domain=args.test_domain, split="cross_test")

    if not tasks:
        print(f"warning: no tasks matched filters (phase={args.phase})", file=sys.stderr)

    extra: list[str] = []
    for raw in args.extra_arg:
        extra.extend(raw.split(" ") if " " in raw else [raw])

    report = run_phase(
        phase=args.phase,
        tasks=tasks,
        emb_store_path=args.emb_store_path,
        quality=args.quality,
        dry_run=args.dry_run,
        extra_args=extra,
        timeout_s=args.per_task_timeout,
    )
    _dump_report(report, args.report_out)
    failed = sum(1 for o in report.outcomes if o.exit_code != 0)
    print(f"{args.phase}: ran {len(report.outcomes)} tasks, {failed} non-zero exits -> {args.report_out}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
