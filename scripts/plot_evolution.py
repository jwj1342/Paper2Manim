#!/usr/bin/env python3
"""Aggregate ``runs/<run_id>/trace.jsonl`` into Hero Plot data.

Three quantities tracked as a function of *cumulative paper-sections processed*:

* **Pass@1**: fraction of scenes whose first VLM review returned ``decision=pass``
  (only counted when VLM was on). Skipped runs are excluded.
* **Avg reflection rounds**: mean number of *text* reflection retries per scene
  (i.e. mean ``iter`` of the final successful render at ``v_rev=0``).
* **VLM avg score**: mean of the *final* (highest) VLM avg score across scenes.

Outputs:

* ``hero_plot.png`` — three-panel matplotlib figure (or single panel if
  matplotlib import fails / matplotlib not installed, fall back to a CSV).
* ``hero_plot.csv`` — the raw aggregated table, one row per run.

Run from repo root:

    python scripts/plot_evolution.py --out hero_plot.png

Filter to a specific bootstrap report:

    python scripts/plot_evolution.py --bootstrap-report runs/bootstrap_2026...json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from paper2manim.config.env import settings
from paper2manim.emb.distill import parse_trace

log = logging.getLogger("plot_evolution")


@dataclass
class RunStats:
    run_id: str
    started_ts: float
    n_scenes: int
    n_pass_at_1: int
    n_scored_scenes: int
    mean_reflection_iters: float
    mean_final_score: float
    had_vlm: bool

    def pass_at_1(self) -> float | None:
        if not self.had_vlm or self.n_scored_scenes == 0:
            return None
        return self.n_pass_at_1 / self.n_scored_scenes

    def to_csv_row(self, idx: int) -> dict:
        return {
            "cumulative_idx": idx,
            "run_id": self.run_id,
            "n_scenes": self.n_scenes,
            "pass_at_1": "" if self.pass_at_1() is None else f"{self.pass_at_1():.3f}",
            "mean_reflection_iters": f"{self.mean_reflection_iters:.3f}",
            "mean_final_score": f"{self.mean_final_score:.3f}",
            "had_vlm": int(self.had_vlm),
            "started_ts": self.started_ts,
        }


def _run_started_ts(run_id: str) -> float:
    """Use the trace file mtime as a stable run-ordering key."""
    p = Path(settings.PAPER2MANIM_RUNS_DIR) / run_id / "trace.jsonl"
    return p.stat().st_mtime if p.exists() else 0.0


def aggregate_run(run_id: str) -> RunStats | None:
    scenes = parse_trace(run_id)
    if not scenes:
        return None
    n_pass_at_1 = 0
    n_scored = 0
    reflection_iters: list[int] = []
    final_scores: list[float] = []
    had_vlm = False
    for sc in scenes.values():
        # Pass@1: did the first VLM review (v_rev=0) say "pass"?
        v0 = [r for r in sc.vlm_reviews if r.v_rev == 0]
        if v0:
            had_vlm = True
            n_scored += 1
            if v0[0].decision.lower() == "pass":
                n_pass_at_1 += 1
        # Reflection iters: highest iter among successful v_rev=0 renders
        successes = [r for r in sc.renders if r.v_rev == 0 and r.status == "success"]
        if successes:
            reflection_iters.append(max(r.iter_idx for r in successes))
        # Final score = highest avg across all v_revs
        if sc.vlm_reviews:
            final_scores.append(max(r.avg_score for r in sc.vlm_reviews))
    return RunStats(
        run_id=run_id,
        started_ts=_run_started_ts(run_id),
        n_scenes=len(scenes),
        n_pass_at_1=n_pass_at_1,
        n_scored_scenes=n_scored,
        mean_reflection_iters=(sum(reflection_iters) / len(reflection_iters))
        if reflection_iters
        else 0.0,
        mean_final_score=(sum(final_scores) / len(final_scores)) if final_scores else 0.0,
        had_vlm=had_vlm,
    )


def discover_run_ids() -> list[str]:
    base = Path(settings.PAPER2MANIM_RUNS_DIR)
    if not base.exists():
        return []
    out: list[str] = []
    for d in base.iterdir():
        if not d.is_dir():
            continue
        if (d / "trace.jsonl").exists():
            out.append(d.name)
    return out


def run_ids_from_bootstrap_report(path: Path) -> list[str]:
    """Bootstrap reports don't carry run ids directly; fall back to scanning
    runs/ for trace files newer than the report's start time."""
    data = json.loads(path.read_text(encoding="utf-8"))
    # The report has tasks but not the per-task run_id (each task uses
    # ``new_run_id``). We scan instead, filtered by mtime ≥ earliest task
    # if available — but the structure we wrote doesn't have task timestamps,
    # so for now we just return all runs with trace.jsonl. This is good
    # enough until we extend ``BootstrapReport`` to record run_ids.
    del data
    return discover_run_ids()


def write_csv(stats: list[RunStats], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "cumulative_idx",
                "run_id",
                "n_scenes",
                "pass_at_1",
                "mean_reflection_iters",
                "mean_final_score",
                "had_vlm",
                "started_ts",
            ],
        )
        w.writeheader()
        for i, s in enumerate(stats, start=1):
            w.writerow(s.to_csv_row(i))


def draw_plot(stats: list[RunStats], out_png: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed; skipping PNG output (CSV is the SoT)")
        return
    if not stats:
        log.warning("no stats to plot")
        return
    xs = list(range(1, len(stats) + 1))
    pass_at_1 = [s.pass_at_1() for s in stats]
    refl = [s.mean_reflection_iters for s in stats]
    score = [s.mean_final_score for s in stats]
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    # Pass@1 — drop None values (runs without VLM scoring)
    xs_p, ys_p = zip(*[(x, y) for x, y in zip(xs, pass_at_1, strict=False) if y is not None], strict=False) if any(
        y is not None for y in pass_at_1
    ) else ([], [])
    if xs_p:
        axes[0].plot(xs_p, ys_p, marker="o")
    axes[0].set_ylabel("Pass@1 (first VLM review = pass)")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(xs, refl, marker="o", color="tab:orange")
    axes[1].set_ylabel("Mean text-reflection iters / scene")
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(xs, score, marker="o", color="tab:green")
    axes[2].set_ylabel("Mean final VLM avg score")
    axes[2].set_ylim(0.0, 5.0)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlabel("# paper-sections processed (cumulative)")
    fig.suptitle("EMB Evolution Curve (Hero Plot)")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    log.info("wrote %s", out_png)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--bootstrap-report",
        default=None,
        help="Optional bootstrap report JSON; restricts which run_ids are aggregated.",
    )
    parser.add_argument("--out", default="hero_plot.png", help="PNG output path.")
    parser.add_argument(
        "--csv-out", default=None, help="CSV output path (defaults to <out>.csv)."
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    if args.bootstrap_report:
        run_ids = run_ids_from_bootstrap_report(Path(args.bootstrap_report))
    else:
        run_ids = discover_run_ids()
    log.info("found %d run(s) with trace.jsonl", len(run_ids))

    stats: list[RunStats] = []
    for rid in run_ids:
        s = aggregate_run(rid)
        if s is not None:
            stats.append(s)
    stats.sort(key=lambda s: s.started_ts)

    out_png = Path(args.out)
    csv_out = Path(args.csv_out) if args.csv_out else out_png.with_suffix(".csv")
    write_csv(stats, csv_out)
    draw_plot(stats, out_png)
    log.info("wrote %s (rows=%d)", csv_out, len(stats))
    return 0


if __name__ == "__main__":
    sys.exit(main())
