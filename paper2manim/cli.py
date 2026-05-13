"""paper2manim CLI: `paper2manim mvp1` and `paper2manim mvp2`."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from paper2manim.artifacts import new_run_id, run_dir, save_input
from paper2manim.config.env import settings
from paper2manim.logging_setup import setup_logging
from paper2manim.parsers.text import load_text
from paper2manim.state import PaperState

console = Console()
log = logging.getLogger(__name__)


def _on_compute_node() -> bool:
    return bool(os.environ.get("SLURM_JOB_ID"))


def _block_login_node_render(allow: bool) -> None:
    """Refuse to render on a login node unless --allow-render-on-login is passed.

    Per global CLAUDE.md HPC rules: login nodes are not allowed to do heavy compute.
    """
    if _on_compute_node() or allow:
        return
    console.print(
        "[bold red]Refusing to render on a login node.[/bold red] "
        "Either run `salloc --account=aip-zhouyang --time=02:00:00 --cpus-per-task=4 --mem=16G` "
        "first, or pass `--no-render` to generate code only, or `--allow-render-on-login` to override."
    )
    sys.exit(2)


def _print_summary(state: PaperState) -> None:
    table = Table(title=f"Paper2Manim run {state.get('run_id', '<unknown>')}")
    table.add_column("Field")
    table.add_column("Value", overflow="fold")
    sb = state.get("storyboard") or {}
    table.add_row("title", sb.get("title", "-"))
    table.add_row("scenes", str(len(sb.get("scenes", []))))
    table.add_row("attempts", str(len(state.get("attempts", []))))
    table.add_row("rendered_videos", str(len(state.get("rendered_videos", []))))
    skipped = state.get("skipped_scenes") or []
    table.add_row("skipped_scenes", ", ".join(skipped) if skipped else "-")
    table.add_row("final_video_path", state.get("final_video_path") or "-")
    table.add_row("fatal_error", state.get("fatal_error") or "-")
    console.print(table)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Verbose logging (DEBUG level)")
def cli(verbose: bool) -> None:
    """LLM multi-agent pipeline: paper -> Manim animation."""
    setup_logging(level=logging.DEBUG if verbose else logging.INFO)


@cli.command()
@click.option(
    "--input",
    "input_arg",
    required=True,
    help="Plain text or path to a .txt file.",
)
@click.option("--quality", default=None, type=click.Choice(["l", "m", "h"]))
@click.option("--no-render", is_flag=True, help="Generate code only; skip manim render.")
@click.option(
    "--allow-render-on-login",
    is_flag=True,
    help="Override the login-node guard (NOT recommended on Vulcan).",
)
def mvp1(input_arg: str, quality: str | None, no_render: bool, allow_render_on_login: bool) -> None:
    """MVP 1.0: short text -> single-scene Manim video."""
    if not no_render:
        _block_login_node_render(allow_render_on_login)
    from paper2manim.graphs.mvp1 import build_mvp1_graph

    text = load_text(input_arg)
    run_id = new_run_id()
    save_input(run_id, raw_text=text)
    state: PaperState = {
        "run_id": run_id,
        "input_kind": "text",
        "raw_text": text,
        "attempts": [],
        "rendered_videos": [],
        "max_retries": settings.PAPER2MANIM_MAX_RETRIES,
        "iter_count": 0,
        "current_scene_idx": 0,
        "quality": quality or settings.PAPER2MANIM_QUALITY,  # type: ignore[typeddict-item]
        "skip_render": no_render,
    }
    console.print(f"[cyan]MVP 1.0 run {run_id}[/cyan]: {text[:80]}...")
    # Stable plain-text marker so external drivers (run_experiment.py,
    # run_bootstrap.py) can grep the run_id back without parsing rich output.
    click.echo(f"RUN_ID={run_id}")
    g = build_mvp1_graph()
    final = g.invoke(state)
    _print_summary(final)
    console.print(f"[green]Run dir:[/green] {run_dir(run_id)}")


@cli.command()
@click.option(
    "--pdf",
    "pdf_path",
    default=None,
    type=click.Path(exists=True),
    help="Local PDF file. Parsed via Marker.",
)
@click.option(
    "--arxiv",
    "arxiv_spec",
    default=None,
    help="arXiv id/URL (e.g. '1706.03762' or 'https://arxiv.org/abs/1706.03762v2'). "
    "Uses author's LaTeX source when available; falls back to Marker on the PDF.",
)
@click.option(
    "--section",
    "arxiv_section",
    default=None,
    help="Optional substring of a \\section{...} title to slice from the arXiv source "
    "(e.g. 'Method'). Ignored for --pdf.",
)
@click.option("--quality", default=None, type=click.Choice(["l", "m", "h"]))
@click.option("--max-retries", default=None, type=int)
@click.option("--no-render", is_flag=True)
@click.option("--allow-render-on-login", is_flag=True)
@click.option(
    "--vlm/--no-vlm",
    "vlm_enabled",
    default=False,
    help="Enable VLM multi-dim scoring loop on rendered scenes (requires config.yaml with vision_checker).",
)
@click.option(
    "--max-visual-revisions",
    default=2,
    type=int,
    show_default=True,
    help="Per-scene cap on visual revision passes when --vlm is on.",
)
@click.option(
    "--emb/--no-emb",
    "emb_enabled",
    default=False,
    help="Enable Episodic Memory Bank: retrieve past success/failure records before coding and consolidate at end of run (proposal §4.1 + §4.4).",
)
@click.option(
    "--emb-store-path",
    default=None,
    type=click.Path(),
    help="Directory for EMB persistence (memory.db + faiss indices). Defaults to $PAPER2MANIM_RUNS_DIR/_emb.",
)
@click.option(
    "--emb-theta-high",
    default=85.0,
    type=float,
    show_default=True,
    help="Success-record acceptance threshold on the 0-100 avg VLM score "
    "(proposal §4.2 3-dim canonical schema). Sits just below the §4.3 "
    "auto-pass threshold (90), so bypass-passes still land in EMB.success. "
    "Lower to ~70 during bootstrap when VLM signal is noisy.",
)
@click.option(
    "--emb-failure-min-margin",
    default=5.0,
    type=float,
    show_default=True,
    help="Minimum (after_score - before_score) on the 0-100 scale for a VLM "
    "transition to qualify as a validated failure record. Larger = fewer but "
    "cleaner records. Set ~0.5 to keep every strict improvement.",
)
@click.option(
    "--emb-llm-distill/--no-emb-llm-distill",
    "emb_use_llm_distillers",
    default=False,
    help="Use LLM-backed rationale_writer + lesson_distiller during consolidation (uses extra API calls). Off by default.",
)
@click.option(
    "--emb-fake-embedder",
    "emb_use_real_embedder",
    flag_value=False,
    default=True,
    help="Use the dependency-free HashEmbedder instead of sentence-transformers. Useful for CI / offline bootstrap.",
)
@click.option(
    "--scene-parallelism",
    default=1,
    type=int,
    show_default=True,
    help="Max concurrent scenes per paper (LangGraph Send fan-out). 1 = serial behavior identical to pre-refactor.",
)
@click.option(
    "--render-concurrency",
    default=None,
    type=int,
    help="Max concurrent Manim subprocesses across all scenes. Defaults to unbounded; recommended 2-4 when --scene-parallelism > 1.",
)
@click.option(
    "--llm-rps",
    default=None,
    type=float,
    help="Global LLM calls-per-second cap (token bucket, shared across all agents). Defaults to unlimited.",
)
def mvp2(
    pdf_path: str | None,
    arxiv_spec: str | None,
    arxiv_section: str | None,
    quality: str | None,
    max_retries: int | None,
    no_render: bool,
    allow_render_on_login: bool,
    vlm_enabled: bool,
    max_visual_revisions: int,
    emb_enabled: bool,
    emb_store_path: str | None,
    emb_theta_high: float,
    emb_failure_min_margin: float,
    emb_use_llm_distillers: bool,
    emb_use_real_embedder: bool,
    scene_parallelism: int,
    render_concurrency: int | None,
    llm_rps: float | None,
) -> None:
    """MVP 2.0: paper -> multi-scene video with reflection loop.

    Input: exactly one of --pdf <path> or --arxiv <id|url>.
    """
    if bool(pdf_path) == bool(arxiv_spec):
        raise click.UsageError("Provide exactly one of --pdf or --arxiv.")
    if not no_render:
        _block_login_node_render(allow_render_on_login)
    from paper2manim import concurrency
    from paper2manim.graphs.mvp2 import build_mvp2_graph

    # Configure process-global throttles BEFORE building the graph so that the
    # first get_llm() / render() call inside any scene branch sees them.
    concurrency.configure(
        scene_parallelism=scene_parallelism,
        render_concurrency=render_concurrency,
        llm_rps=llm_rps,
    )

    run_id = new_run_id()
    # Resolve EMB path now so the user sees the final location in logs even on
    # the default branch. Layout: <runs>/_emb/{memory.db,success.index,...}
    resolved_emb_path = emb_store_path or str(Path(settings.PAPER2MANIM_RUNS_DIR) / "_emb")
    state: PaperState = {
        "run_id": run_id,
        "attempts": [],
        "rendered_videos": [],
        "skipped_scenes": [],
        "scene_reports": [],
        "max_retries": max_retries or settings.PAPER2MANIM_MAX_RETRIES,
        "quality": quality or settings.PAPER2MANIM_QUALITY,  # type: ignore[typeddict-item]
        "skip_render": no_render,
        "vlm_enabled": vlm_enabled,
        "max_visual_revisions": max_visual_revisions,
        "visual_revision_decisions": [],
        "emb_enabled": emb_enabled,
        "emb_store_path": resolved_emb_path if emb_enabled else None,
        "emb_theta_high": emb_theta_high,
        "emb_failure_min_margin": emb_failure_min_margin,
        "emb_use_llm_distillers": emb_use_llm_distillers,
        "emb_use_faiss": True,
        "emb_use_real_embedder": emb_use_real_embedder,
        "retrieved_success": [],
        "retrieved_failure": [],
        "emb_writes": [],
    }
    if emb_enabled:
        console.print(f"[cyan]EMB enabled[/cyan] — store={resolved_emb_path}, theta_high={emb_theta_high}")
    if pdf_path:
        save_input(run_id, pdf_path=pdf_path)
        state["input_kind"] = "pdf"
        state["pdf_path"] = str(Path(pdf_path).resolve())
        console.print(f"[cyan]MVP 2.0 run {run_id}[/cyan] (pdf): {pdf_path}")
    else:
        (run_dir(run_id) / "input.arxiv.txt").write_text(
            f"{arxiv_spec}\nsection={arxiv_section or ''}\n", encoding="utf-8"
        )
        state["input_kind"] = "arxiv"
        state["arxiv_spec"] = arxiv_spec
        state["arxiv_section"] = arxiv_section
        tag = f" §{arxiv_section}" if arxiv_section else ""
        console.print(f"[cyan]MVP 2.0 run {run_id}[/cyan] (arxiv): {arxiv_spec}{tag}")
    # Stable plain-text marker so external drivers (run_experiment.py,
    # run_bootstrap.py) can grep the run_id back without parsing rich output.
    click.echo(f"RUN_ID={run_id}")
    # The parent graph is shallow (parser → summarizer → storyboarder →
    # run_scene fan-out → concat → emb_consolidate); recursion limit just
    # needs to cover the linear depth plus the Send fan-out step. The per-scene
    # recursion budget is set inside ``run_scene_node`` against the compiled
    # scene subgraph, not against this limit.
    recursion_limit = 50
    g = build_mvp2_graph()
    final = g.invoke(state, config={"recursion_limit": recursion_limit})
    _print_summary(final)
    console.print(f"[green]Run dir:[/green] {run_dir(run_id)}")


@cli.command()
def info() -> None:
    """Print configuration and environment status."""
    console.print(
        json.dumps(
            {
                "MIMO_BASE_URL": settings.MIMO_BASE_URL,
                "MIMO_API_KEY_set": bool(settings.MIMO_API_KEY),
                "PAPER2MANIM_RUNS_DIR": str(settings.PAPER2MANIM_RUNS_DIR),
                "PAPER2MANIM_DEFAULT_MODEL": settings.PAPER2MANIM_DEFAULT_MODEL,
                "PAPER2MANIM_MAX_RETRIES": settings.PAPER2MANIM_MAX_RETRIES,
                "PAPER2MANIM_QUALITY": settings.PAPER2MANIM_QUALITY,
                "on_compute_node": _on_compute_node(),
                "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    cli()
