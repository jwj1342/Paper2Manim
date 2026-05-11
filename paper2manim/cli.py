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
    g = build_mvp1_graph()
    final = g.invoke(state)
    _print_summary(final)
    console.print(f"[green]Run dir:[/green] {run_dir(run_id)}")


@cli.command()
@click.option("--pdf", "pdf_path", required=True, type=click.Path(exists=True))
@click.option("--quality", default=None, type=click.Choice(["l", "m", "h"]))
@click.option("--max-retries", default=None, type=int)
@click.option("--no-render", is_flag=True)
@click.option("--allow-render-on-login", is_flag=True)
def mvp2(
    pdf_path: str,
    quality: str | None,
    max_retries: int | None,
    no_render: bool,
    allow_render_on_login: bool,
) -> None:
    """MVP 2.0: full PDF -> multi-scene video with reflection loop."""
    if not no_render:
        _block_login_node_render(allow_render_on_login)
    from paper2manim.graphs.mvp2 import build_mvp2_graph

    run_id = new_run_id()
    save_input(run_id, pdf_path=pdf_path)
    state: PaperState = {
        "run_id": run_id,
        "input_kind": "pdf",
        "pdf_path": str(Path(pdf_path).resolve()),
        "attempts": [],
        "rendered_videos": [],
        "skipped_scenes": [],
        "current_scene_idx": 0,
        "iter_count": 0,
        "max_retries": max_retries or settings.PAPER2MANIM_MAX_RETRIES,
        "quality": quality or settings.PAPER2MANIM_QUALITY,  # type: ignore[typeddict-item]
        "skip_render": no_render,
    }
    console.print(f"[cyan]MVP 2.0 run {run_id}[/cyan]: {pdf_path}")
    g = build_mvp2_graph()
    final = g.invoke(state, config={"recursion_limit": 80})
    _print_summary(final)
    console.print(f"[green]Run dir:[/green] {run_dir(run_id)}")


@cli.command()
def info() -> None:
    """Print configuration and environment status."""
    console.print(json.dumps({
        "MIMO_BASE_URL": settings.MIMO_BASE_URL,
        "MIMO_API_KEY_set": bool(settings.MIMO_API_KEY),
        "PAPER2MANIM_RUNS_DIR": str(settings.PAPER2MANIM_RUNS_DIR),
        "PAPER2MANIM_DEFAULT_MODEL": settings.PAPER2MANIM_DEFAULT_MODEL,
        "PAPER2MANIM_MAX_RETRIES": settings.PAPER2MANIM_MAX_RETRIES,
        "PAPER2MANIM_QUALITY": settings.PAPER2MANIM_QUALITY,
        "on_compute_node": _on_compute_node(),
        "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID"),
    }, indent=2))


if __name__ == "__main__":
    cli()
