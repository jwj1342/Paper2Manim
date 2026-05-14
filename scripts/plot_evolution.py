#!/usr/bin/env python3
"""Aggregate ``runs/<run_id>/trace.jsonl`` into Hero Plot data.

Two operating modes:

1. **Manifest mode** (``--manifest <path>``) — read the experiment manifest
   from ``scripts/run_experiment.py`` and group by ``(config, seed)``.
   Produces per-config curves with mean ± 95% bootstrap CI, exactly the
   form proposal §5 RQ1 calls for. Also dumps a per-(config) EMB
   hit-count histogram when an EMB store path is recoverable.

2. **Legacy / fallback mode** (no ``--manifest``) — scan
   ``$PAPER2MANIM_RUNS_DIR`` for every ``runs/<run_id>/trace.jsonl`` and
   emit a single un-grouped curve. Same behavior as the pre-B3 script,
   kept so ad-hoc bootstrap runs that didn't go through run_experiment.py
   still produce a plot.

Three quantities tracked as a function of *cumulative paper-sections processed*:

* **Pass@1**: fraction of scenes whose first VLM review returned ``decision=pass``
  (only counted when VLM was on). Skipped runs are excluded.
* **Avg reflection rounds**: mean number of *text* reflection retries per scene
  (i.e. mean ``iter`` of the final successful render at ``v_rev=0``).
* **VLM avg score**: mean of the *final* (highest) VLM avg score across scenes.

Outputs (under ``--out-dir``, default ``./``):

* ``hero_plot.png`` — three-panel matplotlib figure.
* ``hero_plot.csv`` — raw aggregated table; one row per (config, x) in
  manifest mode, one row per run in legacy mode.
* ``emb_hits.png`` — per-config EMB hit-count distribution (manifest mode
  only). Skipped silently when no EMB store path can be located.

Examples:

    # Hero plot from a finished run_experiment.py manifest
    python scripts/plot_evolution.py --manifest runs/exp_main/manifest.json \\
        --out-dir runs/exp_main

    # Legacy: scan all runs/ for any trace.jsonl, single curve
    python scripts/plot_evolution.py --out-dir /tmp/legacy_plot
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from paper2manim.config import env as _env_mod
from paper2manim.emb.distill import parse_trace


def _runs_dir() -> Path:
    """Resolve the runs/ directory from the live ``settings`` attribute.

    Looking it up at call time (rather than capturing the singleton at module
    import) lets the test conftest's ``_isolated_runs_dir`` fixture redirect
    everything to ``tmp_path`` cleanly. The cost is one attribute lookup per
    call, which is negligible compared to filesystem work this script does.
    """
    return Path(_env_mod.settings.PAPER2MANIM_RUNS_DIR)

log = logging.getLogger("plot_evolution")


# --------------------------------------------------------------------------- #
# Per-run aggregation (shared by both modes)
# --------------------------------------------------------------------------- #


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

    def to_csv_row(self, idx: int, *, config: str = "", seed: int = -1) -> dict:
        return {
            "cumulative_idx": idx,
            "config": config,
            "seed": seed,
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
    p = _runs_dir() / run_id / "trace.jsonl"
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
    base = _runs_dir()
    if not base.exists():
        return []
    out: list[str] = []
    for d in base.iterdir():
        if not d.is_dir():
            continue
        if (d / "trace.jsonl").exists():
            out.append(d.name)
    return out


# --------------------------------------------------------------------------- #
# Manifest mode: read run_experiment.py manifest, group by (config, seed)
# --------------------------------------------------------------------------- #


@dataclass
class ConfigCurvePoint:
    """One (config, x) data point with mean + 95% bootstrap CI across seeds."""

    config: str
    cumulative_idx: int  # x axis: cumulative paper-sections processed within a seed
    metric: str  # "pass_at_1" | "mean_reflection_iters" | "mean_final_score"
    mean: float | None  # None means no usable measurements at this point
    ci_low: float | None  # 2.5th percentile of bootstrap means; None if n_seeds < 2
    ci_high: float | None
    n_seeds: int  # how many seeds contributed at this x


def read_manifest(path: Path) -> dict[str, Any]:
    """Load and lightly-validate ``manifest.json`` (B2 format)."""
    if not path.exists():
        raise SystemExit(f"manifest not found: {path}")
    obj = json.loads(path.read_text(encoding="utf-8"))
    for required in ("configs", "seeds", "tasks", "runs"):
        if required not in obj:
            raise SystemExit(
                f"manifest at {path} missing required field {required!r}; "
                f"is this a run_experiment.py manifest?"
            )
    return obj


def _bootstrap_mean_ci(
    values: list[float], *, n_resamples: int = 1000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float | None, float | None]:
    """Bootstrap percentile CI for the mean of ``values``.

    Returns ``(mean, ci_low, ci_high)``. With < 2 samples there's no CI; we
    return ``(mean, None, None)`` so the plot renders the point but no band.

    Stdlib-only — no numpy dependency. With ``n_resamples=1000`` the cost is
    O(n*1000) which is trivially fast for n < 100.
    """
    n = len(values)
    if n == 0:
        return float("nan"), None, None
    mean = sum(values) / n
    if n < 2:
        return mean, None, None
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(n_resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo_idx = max(0, int(math.floor((alpha / 2) * n_resamples)))
    hi_idx = min(n_resamples - 1, int(math.ceil((1 - alpha / 2) * n_resamples)) - 1)
    return mean, means[lo_idx], means[hi_idx]


def aggregate_manifest_by_config(
    manifest: dict[str, Any]
) -> tuple[list[RunStats], dict[str, dict[int, list[RunStats]]]]:
    """Group manifest runs by ``config``, then by ``task_idx`` (= x axis).

    Returns ``(all_runs_flat, grouped)`` where ``grouped[config][task_idx]`` is
    the list of per-seed RunStats. ``all_runs_flat`` carries ``config`` and
    ``seed`` annotations so the CSV writer can dump the unrolled long-format
    table.
    """
    flat: list[RunStats] = []
    grouped: dict[str, dict[int, list[RunStats]]] = {}
    for run in manifest["runs"]:
        rid = run.get("run_id")
        if not rid:
            continue
        stats = aggregate_run(rid)
        if stats is None:
            continue
        # Stash config / seed on the RunStats via a sidecar dict keyed on id().
        stats_with_meta = stats
        flat.append(stats_with_meta)
        # Tag the RunStats so the CSV writer can read it back later.
        stats.__dict__["_config"] = run.get("config", "")
        stats.__dict__["_seed"] = int(run.get("seed", -1))
        config = run.get("config", "")
        task_idx = int(run.get("task_idx", 0))
        grouped.setdefault(config, {}).setdefault(task_idx, []).append(stats)
    return flat, grouped


def build_curves(
    grouped: dict[str, dict[int, list[RunStats]]]
) -> dict[str, dict[str, list[ConfigCurvePoint]]]:
    """Convert per-(config, x, seed) RunStats into per-config curves with CI.

    Returns ``out[config][metric]`` = list of ConfigCurvePoint sorted by x.
    metric ∈ {pass_at_1, mean_reflection_iters, mean_final_score}.
    """
    out: dict[str, dict[str, list[ConfigCurvePoint]]] = {}
    for config, by_x in grouped.items():
        out[config] = {
            "pass_at_1": [],
            "mean_reflection_iters": [],
            "mean_final_score": [],
        }
        for x in sorted(by_x.keys()):
            samples = by_x[x]
            # X axis interpretation: task_idx is 0-based; +1 so the curve
            # starts at "1 task processed" matching the proposal Hero Plot.
            display_x = x + 1
            # Pass@1: drop None values (runs without VLM)
            pa1_vals = [s.pass_at_1() for s in samples]
            pa1_vals = [v for v in pa1_vals if v is not None]
            mean, lo, hi = _bootstrap_mean_ci(pa1_vals) if pa1_vals else (None, None, None)
            out[config]["pass_at_1"].append(
                ConfigCurvePoint(config, display_x, "pass_at_1",
                                 mean, lo, hi, len(pa1_vals))
            )
            # mean_reflection_iters: always defined
            ri_vals = [s.mean_reflection_iters for s in samples]
            mean, lo, hi = _bootstrap_mean_ci(ri_vals)
            out[config]["mean_reflection_iters"].append(
                ConfigCurvePoint(config, display_x, "mean_reflection_iters",
                                 mean, lo, hi, len(ri_vals))
            )
            # mean_final_score
            fs_vals = [s.mean_final_score for s in samples]
            mean, lo, hi = _bootstrap_mean_ci(fs_vals)
            out[config]["mean_final_score"].append(
                ConfigCurvePoint(config, display_x, "mean_final_score",
                                 mean, lo, hi, len(fs_vals))
            )
    return out


# --------------------------------------------------------------------------- #
# CSV / plot writers
# --------------------------------------------------------------------------- #


def write_csv_legacy(stats: list[RunStats], out: Path) -> None:
    """Pre-B3 single-curve CSV format. Kept for fallback mode."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "cumulative_idx", "config", "seed", "run_id", "n_scenes",
                "pass_at_1", "mean_reflection_iters", "mean_final_score",
                "had_vlm", "started_ts",
            ],
        )
        w.writeheader()
        for i, s in enumerate(stats, start=1):
            w.writerow(s.to_csv_row(i))


def write_csv_grouped(
    flat: list[RunStats], curves: dict[str, dict[str, list[ConfigCurvePoint]]], out: Path
) -> None:
    """Manifest-mode CSV: per-(config, x) aggregated rows AND a long-format
    per-(config, seed, run_id) table after."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        # Section 1: aggregated rows (one per config × x × metric)
        f.write("# section: aggregated\n")
        w = csv.writer(f)
        w.writerow(["config", "cumulative_idx", "metric",
                    "mean", "ci_low", "ci_high", "n_seeds"])
        for _config, by_metric in curves.items():
            for _metric, points in by_metric.items():
                for p in points:
                    w.writerow([
                        p.config, p.cumulative_idx, p.metric,
                        f"{p.mean:.4f}" if p.mean is not None else "",
                        f"{p.ci_low:.4f}" if p.ci_low is not None else "",
                        f"{p.ci_high:.4f}" if p.ci_high is not None else "",
                        p.n_seeds,
                    ])
        # Section 2: per-run long format
        f.write("\n# section: per_run\n")
        w = csv.DictWriter(
            f,
            fieldnames=[
                "config", "seed", "run_id", "n_scenes",
                "pass_at_1", "mean_reflection_iters", "mean_final_score",
                "had_vlm", "started_ts",
            ],
        )
        w.writeheader()
        for s in flat:
            w.writerow({
                "config": s.__dict__.get("_config", ""),
                "seed": s.__dict__.get("_seed", -1),
                "run_id": s.run_id,
                "n_scenes": s.n_scenes,
                "pass_at_1": "" if s.pass_at_1() is None else f"{s.pass_at_1():.3f}",
                "mean_reflection_iters": f"{s.mean_reflection_iters:.3f}",
                "mean_final_score": f"{s.mean_final_score:.3f}",
                "had_vlm": int(s.had_vlm),
                "started_ts": s.started_ts,
            })


def draw_plot_legacy(stats: list[RunStats], out_png: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed; CSV is the SoT")
        return
    if not stats:
        log.warning("no stats to plot")
        return
    xs = list(range(1, len(stats) + 1))
    pass_at_1 = [s.pass_at_1() for s in stats]
    refl = [s.mean_reflection_iters for s in stats]
    score = [s.mean_final_score for s in stats]
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    if any(y is not None for y in pass_at_1):
        xs_p = [x for x, y in zip(xs, pass_at_1, strict=False) if y is not None]
        ys_p = [y for y in pass_at_1 if y is not None]
        axes[0].plot(xs_p, ys_p, marker="o")
    axes[0].set_ylabel("Pass@1")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(xs, refl, marker="o", color="tab:orange")
    axes[1].set_ylabel("Mean text-reflection iters / scene")
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(xs, score, marker="o", color="tab:green")
    axes[2].set_ylabel("Mean final VLM avg score")
    axes[2].set_ylim(0.0, 100.0)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlabel("# paper-sections processed (cumulative)")
    fig.suptitle("EMB Evolution Curve (Hero Plot — legacy mode, single curve)")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    log.info("wrote %s", out_png)


_CONFIG_COLORS = {
    "A": "tab:gray",
    "B": "tab:orange",
    "C": "tab:green",
}


def draw_plot_grouped(
    curves: dict[str, dict[str, list[ConfigCurvePoint]]], out_png: Path
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed; CSV is the SoT")
        return
    if not curves:
        log.warning("no curves to plot")
        return
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    metrics = (
        ("pass_at_1", "Pass@1 (first VLM review = pass)", (0.0, 1.0)),
        ("mean_reflection_iters", "Mean text-reflection iters / scene", None),
        ("mean_final_score", "Mean final VLM avg score (0-100)", (0.0, 100.0)),
    )
    for ax, (metric, ylabel, ylim) in zip(axes, metrics, strict=False):
        for config, by_metric in curves.items():
            points = [p for p in by_metric[metric] if p.mean is not None]
            if not points:
                continue
            xs = [p.cumulative_idx for p in points]
            ys = [p.mean for p in points]
            color = _CONFIG_COLORS.get(config)
            ax.plot(xs, ys, marker="o", label=config, color=color)
            # CI band where available
            xs_ci = [p.cumulative_idx for p in points if p.ci_low is not None]
            lo = [p.ci_low for p in points if p.ci_low is not None]
            hi = [p.ci_high for p in points if p.ci_low is not None]
            if xs_ci:
                ax.fill_between(xs_ci, lo, hi, alpha=0.15, color=color)
        ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
    axes[-1].set_xlabel("# paper-sections processed (cumulative, per seed)")
    fig.suptitle("EMB Evolution — Hero Plot (mean ± 95% bootstrap CI across seeds)")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    log.info("wrote %s", out_png)


# --------------------------------------------------------------------------- #
# EMB hits panel (manifest mode only)
# --------------------------------------------------------------------------- #


def _emb_paths_from_manifest(manifest: dict[str, Any]) -> dict[str, Path]:
    """Walk manifest cli_args to recover the EMB store base path.

    Returns ``{config: path}`` for configs whose preset includes ``--emb``.
    Pre-B6, just one `<emb_store_base>/<config>_seed_<seed>` per (config, seed).
    """
    cli = manifest.get("cli_args") or {}
    base = cli.get("emb_store_base")
    if not base:
        return {}
    out: dict[str, Path] = {}
    from paper2manim.ablations import resolve as _resolve_preset
    for config in manifest.get("configs") or []:
        try:
            flags = _resolve_preset(config)
        except KeyError:
            continue
        if "--emb" not in flags:
            continue
        # Use seed=1's path as the canonical EMB store for histogram purposes
        # (each seed has its own; for Hero Plot summary we want one
        # representative). Manifest mode users wanting per-seed histograms can
        # call paper2manim emb stats directly.
        seeds = manifest.get("seeds") or [1]
        out[config] = Path(base) / f"{config}_seed_{seeds[0]}"
    return out


def draw_emb_hits_plot(
    emb_paths: dict[str, Path], out_png: Path, *, seed: int | None = None
) -> None:
    """``seed`` only affects the figure title — the EMB store path lookup
    happens upstream in :func:`_emb_paths_from_manifest`. Passing the seed
    lets the title match the actual store being plotted; defaults to ``?``
    when callers don't have one (legacy ``--out`` path or ad-hoc invocation)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if not emb_paths:
        return
    from paper2manim.emb.manager import build_default_emb
    fig, axes = plt.subplots(
        2, max(1, len(emb_paths)), figsize=(4 * max(1, len(emb_paths)), 6), sharey=True,
    )
    if len(emb_paths) == 1:
        axes = [[axes[0]], [axes[1]]]
    for col_idx, (config, path) in enumerate(sorted(emb_paths.items())):
        if not path.exists():
            log.info("[emb_hits] %s store missing at %s; skipping", config, path)
            continue
        try:
            emb = build_default_emb(str(path), use_real_embedder=False)
        except Exception as exc:  # noqa: BLE001
            log.warning("[emb_hits] %s open failed: %s", config, exc)
            continue
        for row_idx, polarity in enumerate(("success", "failure")):
            recs = emb.all(polarity=polarity)
            counts = [r.provenance.hit_count for r in recs] or [0]
            ax = axes[row_idx][col_idx]
            ax.hist(counts, bins=range(0, max(counts) + 2))
            ax.set_title(f"{config} · {polarity}")
            ax.set_xlabel("hit_count")
            if col_idx == 0:
                ax.set_ylabel("# records")
    seed_label = "?" if seed is None else seed
    fig.suptitle(f"EMB hit-count distribution per config (seed {seed_label})")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    log.info("wrote %s", out_png)


# --------------------------------------------------------------------------- #
# main()
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--manifest",
        default=None,
        help="Path to a run_experiment.py manifest.json. When set, plot is "
        "grouped by (config, seed) with bootstrap CI; otherwise legacy mode "
        "scans runs/ for any trace.jsonl.",
    )
    parser.add_argument("--out", default=None, help="(legacy) PNG output path.")
    parser.add_argument("--csv-out", default=None, help="(legacy) CSV output path.")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory for hero_plot.png + hero_plot.csv + emb_hits.png. "
        "If unset, falls back to the per-flag --out / --csv-out for "
        "backward compatibility.",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    out_dir = Path(args.out_dir) if args.out_dir else None

    if args.manifest:
        manifest = read_manifest(Path(args.manifest))
        flat, grouped = aggregate_manifest_by_config(manifest)
        curves = build_curves(grouped)
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            csv_path = out_dir / "hero_plot.csv"
            png_path = out_dir / "hero_plot.png"
            emb_png = out_dir / "emb_hits.png"
        else:
            csv_path = Path(args.csv_out or "hero_plot.csv")
            png_path = Path(args.out or "hero_plot.png")
            emb_png = png_path.with_name("emb_hits.png")
        write_csv_grouped(flat, curves, csv_path)
        draw_plot_grouped(curves, png_path)
        # Same seed _emb_paths_from_manifest used (manifest seeds[0]) so the
        # title and the data on the panel agree.
        seeds_for_emb = manifest.get("seeds") or [1]
        draw_emb_hits_plot(
            _emb_paths_from_manifest(manifest), emb_png, seed=seeds_for_emb[0]
        )
        log.info(
            "wrote %s + %s (configs=%d, runs_aggregated=%d)",
            csv_path, png_path, len(curves), len(flat),
        )
        return 0

    # ---- legacy / fallback mode ----
    run_ids = discover_run_ids()
    log.info("found %d run(s) with trace.jsonl", len(run_ids))
    stats: list[RunStats] = []
    for rid in run_ids:
        s = aggregate_run(rid)
        if s is not None:
            stats.append(s)
    stats.sort(key=lambda s: s.started_ts)

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        png_path = out_dir / "hero_plot.png"
        csv_path = out_dir / "hero_plot.csv"
    else:
        png_path = Path(args.out or "hero_plot.png")
        csv_path = Path(args.csv_out) if args.csv_out else png_path.with_suffix(".csv")
    write_csv_legacy(stats, csv_path)
    draw_plot_legacy(stats, png_path)
    log.info("wrote %s (rows=%d)", csv_path, len(stats))
    return 0


if __name__ == "__main__":
    sys.exit(main())
