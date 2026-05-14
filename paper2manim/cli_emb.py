"""``paper2manim emb`` — inspection + health subcommand group.

Five subcommands:

* ``stats`` — total counts per polarity + hit-count quantiles + cold-record
  share. The first line of defense for "is the EMB doing anything?".
* ``list`` — paginated table of records, filterable by polarity / source paper.
* ``show <id>`` — full record dump (context + body + provenance) as JSON.
* ``prune`` — list (default) or delete cold records. Conservative defaults
  (``--cold-hit-threshold 0 --max-age-days 30``) and dry-run by default
  so users see what would be removed before committing.
* ``retest`` — sample N success records and re-score them with the live VLM,
  flagging records whose new score is significantly below their stored
  ``vlm_score``. Observation-only — actual deletion goes through ``prune``.
  Marked ``[EXPERIMENTAL]`` because it depends on the original montage
  being on disk under ``runs/<run_id>/vlm_frames/``, which isn't always
  true after long-running experiments archive their output.

The group is registered into the top-level ``paper2manim`` CLI in
``paper2manim/cli.py`` via ``cli.add_command(emb_group)`` so it shows up
under ``paper2manim --help`` next to ``mvp1`` / ``mvp2`` / ``info``.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.table import Table

from paper2manim.config.env import settings
from paper2manim.emb.manager import EpisodicMemoryBank, build_default_emb
from paper2manim.emb.schema import MemoryRecord

log = logging.getLogger(__name__)

console = Console()


# --------------------------------------------------------------------------- #
# Path resolution
# --------------------------------------------------------------------------- #


def _default_store_path() -> Path:
    """Same convention as ``paper2manim mvp2 --emb-store-path`` default."""
    return Path(settings.PAPER2MANIM_RUNS_DIR) / "_emb"


def _resolve_store(store_path: str | None) -> Path:
    p = Path(store_path) if store_path else _default_store_path()
    return p


def _open_emb(store_path: str | None) -> EpisodicMemoryBank:
    """Open the EMB at ``store_path`` for inspection.

    Defers embedder choice to the store's pinned ``embedder.json`` spec
    (issue #27 Bug B) — forcing ``HashEmbedder(64)`` here used to blow away
    rehydration on stores that were written with sentence-transformers.
    The ST embedder is lazy-loaded on first ``encode()``, so read-only ops
    like ``stats`` / ``list`` / ``show`` still don't pay the torch import.
    """
    p = _resolve_store(store_path)
    if not p.exists():
        raise click.ClickException(
            f"EMB store not found at {p}. Run something with --emb first, "
            f"or pass --store-path explicitly."
        )
    return build_default_emb(str(p))


# --------------------------------------------------------------------------- #
# stats
# --------------------------------------------------------------------------- #


@dataclass
class HitStats:
    """Hit-count distribution summary used by both ``stats`` and ``prune``."""

    n: int
    n_zero: int
    minimum: int
    p25: int
    median: int
    p75: int
    maximum: int

    @classmethod
    def of(cls, hits: list[int]) -> HitStats:
        if not hits:
            return cls(n=0, n_zero=0, minimum=0, p25=0, median=0, p75=0, maximum=0)
        s = sorted(hits)
        n = len(s)

        def pct(p: float) -> int:
            # Linear-interpolation percentile, matching numpy default ('linear')
            # for a smooth small-n behavior.
            i = (n - 1) * p
            lo = int(i)
            hi = min(lo + 1, n - 1)
            frac = i - lo
            return int(round(s[lo] * (1 - frac) + s[hi] * frac))

        return cls(
            n=n,
            n_zero=sum(1 for x in s if x == 0),
            minimum=s[0],
            p25=pct(0.25),
            median=pct(0.50),
            p75=pct(0.75),
            maximum=s[-1],
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "n": self.n,
            "n_zero": self.n_zero,
            "min": self.minimum,
            "p25": self.p25,
            "median": self.median,
            "p75": self.p75,
            "max": self.maximum,
        }


def collect_stats(emb: EpisodicMemoryBank) -> dict[str, Any]:
    """Build the dict that ``paper2manim emb stats`` prints (and that
    callers / tests can introspect without screen-scraping)."""
    base = emb.stats()
    out: dict[str, Any] = dict(base)
    now = time.time()
    for polarity in ("success", "failure"):
        recs = emb.all(polarity=polarity)
        hits = [r.provenance.hit_count for r in recs]
        last_used_ages = [
            (now - r.provenance.last_used) / 86400.0
            for r in recs
            if r.provenance.last_used is not None
        ]
        out[f"{polarity}_hit_stats"] = HitStats.of(hits).to_dict()
        if last_used_ages:
            out[f"{polarity}_last_used_age_days"] = {
                "min": round(min(last_used_ages), 2),
                "mean": round(sum(last_used_ages) / len(last_used_ages), 2),
                "max": round(max(last_used_ages), 2),
                "n_with_last_used": len(last_used_ages),
            }
        else:
            out[f"{polarity}_last_used_age_days"] = None
    return out


# --------------------------------------------------------------------------- #
# Pruning logic
# --------------------------------------------------------------------------- #


def _store_db_path(store_path: Path) -> Path:
    return store_path / "memory.db"


def _read_created_at_map(store_path: Path) -> dict[str, float]:
    """Pull ``id → created_at`` from the SQLite store directly.

    The MemoryStore protocol doesn't expose ``created_at`` (it's a row-level
    timestamp internal to the SQLite impl), so we read it ourselves rather
    than threading it through the abstraction.
    """
    db = _store_db_path(store_path)
    if not db.exists():
        return {}
    out: dict[str, float] = {}
    conn = sqlite3.connect(str(db))
    try:
        for rid, ts in conn.execute("SELECT id, created_at FROM memory_records"):
            out[rid] = float(ts)
    finally:
        conn.close()
    return out


def select_cold_records(
    emb: EpisodicMemoryBank,
    store_path: Path,
    *,
    cold_hit_threshold: int,
    max_age_days: float,
    polarity: str | None = None,
    now: float | None = None,
) -> list[MemoryRecord]:
    """Return records that should be pruned.

    Selection criteria (must satisfy ALL):

    * ``provenance.hit_count <= cold_hit_threshold`` (default 0 = never used)
    * Last activity age > ``max_age_days``. The age is ``now - last_used`` if
      ``last_used`` is set, else ``now - created_at`` from the SQLite row.
      This guards against pruning brand-new records that simply haven't been
      retrieved yet.
    """
    cutoff_seconds = max_age_days * 86400.0
    n = now if now is not None else time.time()
    created_map = _read_created_at_map(store_path)
    out: list[MemoryRecord] = []
    for rec in emb.all(polarity=polarity):
        if rec.provenance.hit_count > cold_hit_threshold:
            continue
        ref_ts = rec.provenance.last_used or created_map.get(rec.id)
        if ref_ts is None:
            # No information about freshness — be conservative and skip.
            continue
        if (n - ref_ts) <= cutoff_seconds:
            continue
        out.append(rec)
    return out


# --------------------------------------------------------------------------- #
# Click group
# --------------------------------------------------------------------------- #


_STORE_PATH_OPT = click.option(
    "--store-path",
    "store_path",
    default=None,
    type=click.Path(),
    help="EMB store directory. Defaults to $PAPER2MANIM_RUNS_DIR/_emb.",
)


@click.group("emb")
def emb_group() -> None:
    """Episodic Memory Bank inspection + health utilities."""


@emb_group.command("stats")
@_STORE_PATH_OPT
@click.option("--json", "as_json", is_flag=True, help="Emit raw JSON instead of a table.")
def stats_cmd(store_path: str | None, as_json: bool) -> None:
    """Total counts, hit-count quantiles, and last-used freshness."""
    emb = _open_emb(store_path)
    data = collect_stats(emb)
    if as_json:
        click.echo(json.dumps(data, indent=2, default=str))
        return
    table = Table(title=f"EMB stats — {_resolve_store(store_path)}")
    table.add_column("metric")
    table.add_column("value", overflow="fold")
    for k in ("total", "success", "failure", "success_indexed", "failure_indexed"):
        table.add_row(k, str(data.get(k, "-")))
    for polarity in ("success", "failure"):
        h = data.get(f"{polarity}_hit_stats", {})
        if h.get("n", 0) == 0:
            continue
        table.add_row(
            f"{polarity} hit_count",
            f"n={h['n']} zero={h['n_zero']} min={h['min']} "
            f"p25={h['p25']} med={h['median']} p75={h['p75']} max={h['max']}",
        )
        age = data.get(f"{polarity}_last_used_age_days")
        if age:
            table.add_row(
                f"{polarity} last_used age (days)",
                f"min={age['min']} mean={age['mean']} max={age['max']} "
                f"n={age['n_with_last_used']}",
            )
    console.print(table)


@emb_group.command("list")
@_STORE_PATH_OPT
@click.option(
    "--polarity",
    type=click.Choice(["success", "failure", "all"]),
    default="all",
    show_default=True,
)
@click.option(
    "--source-paper",
    default=None,
    help="Substring filter on context.source_paper (case-insensitive).",
)
@click.option("--limit", default=20, type=int, show_default=True)
def list_cmd(
    store_path: str | None, polarity: str, source_paper: str | None, limit: int
) -> None:
    """Paginated record listing."""
    emb = _open_emb(store_path)
    pol = None if polarity == "all" else polarity
    recs = emb.all(polarity=pol)  # type: ignore[arg-type]
    if source_paper:
        needle = source_paper.lower()
        recs = [r for r in recs if needle in (r.context.source_paper or "").lower()]
    recs = recs[:limit]
    table = Table(title=f"EMB records ({len(recs)} shown)")
    table.add_column("id8")
    table.add_column("polarity")
    table.add_column("scene_role")
    table.add_column("source")
    table.add_column("hits", justify="right")
    table.add_column("last_used", overflow="fold")
    table.add_column("score", justify="right")
    for r in recs:
        last = r.provenance.last_used
        last_s = "-" if last is None else time.strftime("%Y-%m-%d %H:%M", time.localtime(last))
        score = r.provenance.vlm_score
        if score is None:
            score = r.provenance.after_score
        table.add_row(
            r.id[:8],
            r.polarity,
            r.context.scene_role,
            f"{r.context.source_paper}{(' / ' + r.context.source_section) if r.context.source_section else ''}",
            str(r.provenance.hit_count),
            last_s,
            "-" if score is None else f"{float(score):.1f}",
        )
    console.print(table)


@emb_group.command("show")
@_STORE_PATH_OPT
@click.argument("record_id")
def show_cmd(store_path: str | None, record_id: str) -> None:
    """Full record dump as JSON. ``RECORD_ID`` may be a prefix; the first
    matching id is shown."""
    emb = _open_emb(store_path)
    # Allow prefix lookup for ergonomics.
    matches = [r for r in emb.all() if r.id.startswith(record_id)]
    if not matches:
        raise click.ClickException(f"no record matches id prefix {record_id!r}")
    rec = matches[0]
    payload = {
        "id": rec.id,
        "polarity": rec.polarity,
        "context": {
            **rec.context.model_dump(exclude={"task_embedding"}),
            "task_embedding_dim": len(rec.context.task_embedding),
        },
        "body": rec.body.model_dump(),
        "provenance": rec.provenance.model_dump(),
    }
    click.echo(json.dumps(payload, indent=2, default=str, ensure_ascii=False))


@emb_group.command("prune")
@_STORE_PATH_OPT
@click.option(
    "--cold-hit-threshold",
    type=int,
    default=0,
    show_default=True,
    help="Records with hit_count <= this are pruning candidates.",
)
@click.option(
    "--max-age-days",
    type=float,
    default=30.0,
    show_default=True,
    help="Records last used (or created) > this many days ago are candidates.",
)
@click.option(
    "--polarity",
    type=click.Choice(["success", "failure", "all"]),
    default="all",
    show_default=True,
)
@click.option(
    "--apply",
    "apply_changes",
    is_flag=True,
    help="Actually delete the candidates. Without this, the command is a "
    "no-op preview (the default — pruning is destructive).",
)
def prune_cmd(
    store_path: str | None,
    cold_hit_threshold: int,
    max_age_days: float,
    polarity: str,
    apply_changes: bool,
) -> None:
    """List or delete cold records (proposal §9 risk #2 mitigation)."""
    emb = _open_emb(store_path)
    pol = None if polarity == "all" else polarity
    cold = select_cold_records(
        emb,
        _resolve_store(store_path),
        cold_hit_threshold=cold_hit_threshold,
        max_age_days=max_age_days,
        polarity=pol,
    )
    table = Table(title=f"prune candidates ({len(cold)})")
    table.add_column("id8")
    table.add_column("polarity")
    table.add_column("hits", justify="right")
    table.add_column("source", overflow="fold")
    for r in cold:
        table.add_row(
            r.id[:8], r.polarity, str(r.provenance.hit_count),
            f"{r.context.source_paper}",
        )
    console.print(table)
    if not apply_changes:
        click.echo(
            f"[dry-run] would delete {len(cold)} record(s). "
            f"Pass --apply to commit."
        )
        return
    deleted = 0
    for r in cold:
        try:
            emb.delete(r.id)
            deleted += 1
        except Exception as exc:  # noqa: BLE001
            log.warning("[prune] delete failed for %s: %s", r.id[:8], exc)
    emb.save_indices()
    click.echo(f"deleted {deleted}/{len(cold)} record(s)")


@emb_group.command("retest")
@_STORE_PATH_OPT
@click.option("--sample", type=int, default=10, show_default=True)
@click.option(
    "--score-margin",
    type=float,
    default=5.0,
    show_default=True,
    help="Flag records whose new score is below stored vlm_score by more than this margin.",
)
def retest_cmd(store_path: str | None, sample: int, score_margin: float) -> None:
    """[EXPERIMENTAL] Re-score N success records with the live VLM and flag
    candidates for memory-decay (§9 risk #2)."""
    emb = _open_emb(store_path)
    successes = [r for r in emb.all(polarity="success")]
    if not successes:
        click.echo("no success records to retest")
        return
    import random
    rng = random.Random(0)
    sampled = rng.sample(successes, k=min(sample, len(successes)))
    decayed: list[tuple[str, float, float]] = []
    unverifiable: list[str] = []
    try:
        from paper2manim.agents.vlm_scene_reviewer import review_scene
    except Exception as exc:  # noqa: BLE001
        raise click.ClickException(f"VLM agent unavailable: {exc}") from exc
    for r in sampled:
        old = r.provenance.vlm_score
        if old is None:
            unverifiable.append(r.id[:8])
            continue
        # Reconstruct the montage path. Convention: runs/<run_id>/vlm_frames/<scene>_v<n>.png
        # ``final_v_rev`` is the v_rev whose VLM score was stored as ``vlm_score``;
        # comparing the new score against ``vlm_score`` is only meaningful if we
        # re-score the same frame. Older records (pre-PR introducing the field)
        # default to 0, which matches their actual final_v_rev.
        v_rev = r.provenance.final_v_rev
        montage = (
            Path(settings.PAPER2MANIM_RUNS_DIR)
            / r.provenance.run_id / "vlm_frames"
            / f"{r.provenance.scene_id}_v{v_rev}.png"
        )
        if not montage.exists():
            unverifiable.append(r.id[:8])
            continue
        try:
            review = review_scene(
                {"name": r.provenance.scene_id, "description": r.context.task_text},
                str(montage),
            )
            new = float(review.get("average_score") or 0.0)
        except Exception as exc:  # noqa: BLE001
            log.warning("[retest] %s VLM call failed: %s", r.id[:8], exc)
            unverifiable.append(r.id[:8])
            continue
        if (old - new) > score_margin:
            decayed.append((r.id[:8], old, new))
    table = Table(title=f"retest sample={len(sampled)} margin={score_margin}")
    table.add_column("id8")
    table.add_column("old_score", justify="right")
    table.add_column("new_score", justify="right")
    table.add_column("delta", justify="right")
    for rid, old, new in decayed:
        table.add_row(rid, f"{old:.1f}", f"{new:.1f}", f"{new - old:+.1f}")
    console.print(table)
    click.echo(
        f"decayed candidates: {len(decayed)}/{len(sampled)} "
        f"unverifiable: {len(unverifiable)}"
    )


__all__ = [
    "emb_group",
    "collect_stats",
    "select_cold_records",
    "HitStats",
]
