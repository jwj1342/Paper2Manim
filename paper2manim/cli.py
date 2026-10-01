"""ManimAgent generation and episodic-memory CLI."""

from __future__ import annotations

import json
import logging
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


def _preflight_voiceover(voiceover_enabled: bool, no_render: bool) -> None:
    """Validate voiceover prerequisites before starting the graph.

    Fails fast with ``click.UsageError`` so the user doesn't wait through a
    full render only to discover a missing TTS config.
    """
    if not voiceover_enabled:
        return

    if no_render:
        raise click.UsageError(
            "--no-render and --voiceover are incompatible: "
            "voiceover needs rendered videos to align audio to."
        )

    from paper2manim.llm import tts_config as get_tts_config

    cfg = get_tts_config()
    if cfg is None:
        raise click.UsageError(
            "--voiceover requires a 'tts:' block in config.yaml. "
            "Add the block or pass --no-voiceover."
        )

    # Validate provider is supported (does not make a network call).
    from paper2manim.infrastructure.tts.factory import build_tts_client

    try:
        build_tts_client(cfg)
    except RuntimeError as exc:
        raise click.UsageError(f"TTS configuration error: {exc}") from exc


def _print_summary(state: PaperState) -> None:
    table = Table(title=f"Paper2Manim run {state.get('run_id', '<unknown>')}")
    table.add_column("Field")
    table.add_column("Value", overflow="fold")
    sb = state.get("storyboard") or {}
    table.add_row("title", sb.get("title", "-"))
    table.add_row("scenes", str(len(sb.get("scenes", []))))
    table.add_row("attempts", str(len(state.get("attempts", []))))
    table.add_row(
        "reflection_rounds",
        str(sum(report.get("reflection_rounds", 0) for report in state.get("scene_reports", []))),
    )
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


# Register subcommand groups. Imported here (not at top of file) so the
# heavy EMB stack does not load until a command needs it.
from paper2manim.cli_emb import emb_group as _emb_group  # noqa: E402

cli.add_command(_emb_group)


@cli.command()
@click.option("--input", "input_arg", default=None, help="Section text or path to a .txt file.")
@click.option(
    "--scene-role",
    default="BACKGROUND",
    type=click.Choice(["BACKGROUND", "METHOD", "EXPERIMENT", "CONCLUSION"], case_sensitive=False),
    show_default=True,
)
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
    "(e.g. 'Method'). Requires --arxiv.",
)
@click.option("--quality", default=None, type=click.Choice(["l", "m", "h"]))
@click.option("--max-retries", default=None, type=click.IntRange(min=0))
@click.option("--no-render", is_flag=True)
@click.option(
    "--vlm/--no-vlm",
    "vlm_enabled",
    default=True,
    help="Enable VLM multi-dim scoring loop on rendered scenes (requires config.yaml with vision_checker).",
)
@click.option(
    "--max-visual-revisions",
    default=2,
    type=click.IntRange(min=0),
    show_default=True,
    help="Per-scene visual-revision budget (paper: 2).",
)
@click.option(
    "--emb/--no-emb",
    "emb_enabled",
    default=True,
    help="Enable Episodic Memory Bank: retrieve past success/failure records before coding and consolidate at end of run.",
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
    help="Minimum average VLM score for storing a positive memory record.",
)
@click.option(
    "--emb-failure-min-margin",
    default=5.0,
    type=float,
    show_default=True,
    help="Minimum VLM score improvement for storing a visual repair lesson.",
)
@click.option(
    "--emb-fake-embedder",
    "emb_use_real_embedder",
    flag_value=False,
    default=True,
    help="Use the dependency-free HashEmbedder instead of sentence-transformers. For offline development checks.",
)
@click.option(
    "--emb-readonly/--no-emb-readonly",
    "emb_readonly",
    default=False,
    help="Read existing memories without adding records or updating hit counters.",
)
@click.option("--domain", default=None, help="Optional domain label stored with memories.")
@click.option(
    "--emb-no-success-channel",
    "emb_no_success_channel",
    is_flag=True,
    default=False,
    help="Disable EMB.success on both retrieve and write.",
)
@click.option(
    "--emb-no-failure-channel",
    "emb_no_failure_channel",
    is_flag=True,
    default=False,
    help="Disable EMB.failure on both retrieve and write.",
)
@click.option(
    "--voiceover/--no-voiceover",
    "voiceover_enabled",
    default=False,
    help="Enable voiceover narration + TTS synthesis.",
)
@click.option(
    "--tts-voice",
    default=None,
    help="Override TTS voice (defaults to config.yaml tts.voice).",
)
@click.option(
    "--tts-speed",
    default=None,
    type=float,
    help="TTS speed multiplier (defaults to config.yaml tts.speed).",
)
@click.option(
    "--voiceover-language",
    default="en",
    show_default=True,
    help="Language for narration text.",
)
@click.option(
    "--voiceover-strict/--voiceover-best-effort",
    "voiceover_strict",
    default=True,
    help="Strict mode: TTS/alignment failures are fatal. Best-effort: degrade gracefully.",
)
@click.option(
    "--scene-parallelism",
    default=1,
    type=click.IntRange(min=1),
    show_default=True,
    help="Max concurrent scenes per paper (LangGraph Send fan-out). 1 = serial execution.",
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
def generate(
    input_arg: str | None,
    scene_role: str,
    pdf_path: str | None,
    arxiv_spec: str | None,
    arxiv_section: str | None,
    quality: str | None,
    max_retries: int | None,
    no_render: bool,
    vlm_enabled: bool,
    max_visual_revisions: int,
    emb_enabled: bool,
    emb_store_path: str | None,
    emb_theta_high: float,
    emb_failure_min_margin: float,
    emb_use_real_embedder: bool,
    emb_readonly: bool,
    domain: str | None,
    emb_no_success_channel: bool,
    emb_no_failure_channel: bool,
    voiceover_enabled: bool,
    tts_voice: str | None,
    tts_speed: float | None,
    voiceover_language: str,
    voiceover_strict: bool,
    scene_parallelism: int,
    render_concurrency: int | None,
    llm_rps: float | None,
) -> None:
    """Generate a paper-section animation with visual reflection and dual-channel memory.

    Input: exactly one of --input, --pdf, or --arxiv.
    """
    _preflight_voiceover(voiceover_enabled, no_render)
    if sum(x is not None for x in (input_arg, pdf_path, arxiv_spec)) != 1:
        raise click.UsageError("Provide exactly one of --input, --pdf, or --arxiv.")
    if arxiv_section and not arxiv_spec:
        raise click.UsageError("--section requires --arxiv; use --input for local section text.")
    if no_render and emb_enabled:
        emb_readonly = True
    from paper2manim import concurrency
    from paper2manim.graphs.generation import build_generation_graph

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
        "max_retries": settings.PAPER2MANIM_MAX_RETRIES if max_retries is None else max_retries,
        "quality": quality or settings.PAPER2MANIM_QUALITY,  # type: ignore[typeddict-item]
        "skip_render": no_render,
        "vlm_enabled": vlm_enabled,
        "max_visual_revisions": max_visual_revisions,
        "visual_revision_decisions": [],
        "emb_enabled": emb_enabled,
        "emb_store_path": resolved_emb_path if emb_enabled else None,
        "emb_theta_high": emb_theta_high,
        "emb_failure_min_margin": emb_failure_min_margin,
        "emb_use_faiss": True,
        "emb_use_real_embedder": emb_use_real_embedder,
        "retrieved_success": [],
        "retrieved_failure": [],
        "emb_writes": [],
        "emb_readonly": emb_readonly,
        "domain": domain,
        "scene_role": scene_role.upper(),
        "emb_no_success_channel": emb_no_success_channel,
        "emb_no_failure_channel": emb_no_failure_channel,
        # Voiceover / TTS
        "voiceover_enabled": voiceover_enabled,
        "voiceover_strict": voiceover_strict,
        "voiceover_language": voiceover_language,
        "narration_plan": None,
        "vo_tts_voice_override": tts_voice,
        "vo_tts_speed_override": tts_speed,
    }
    if emb_enabled:
        console.print(
            f"[cyan]EMB enabled[/cyan] — store={resolved_emb_path}, theta_high={emb_theta_high}"
        )
    if input_arg is not None:
        text = load_text(input_arg)
        if not text.strip():
            raise click.UsageError("--input must contain nonempty text.")
        save_input(run_id, raw_text=text)
        state["input_kind"] = "text"
        state["raw_text"] = text
        console.print(f"[cyan]ManimAgent run {run_id}[/cyan] (text): {text[:80]}")
    elif pdf_path:
        save_input(run_id, pdf_path=pdf_path)
        state["input_kind"] = "pdf"
        state["pdf_path"] = str(Path(pdf_path).resolve())
        console.print(f"[cyan]ManimAgent run {run_id}[/cyan] (pdf): {pdf_path}")
    else:
        (run_dir(run_id) / "input.arxiv.txt").write_text(
            f"{arxiv_spec}\nsection={arxiv_section or ''}\n", encoding="utf-8"
        )
        state["input_kind"] = "arxiv"
        state["arxiv_spec"] = arxiv_spec
        state["arxiv_section"] = arxiv_section
        tag = f" §{arxiv_section}" if arxiv_section else ""
        console.print(f"[cyan]ManimAgent run {run_id}[/cyan] (arxiv): {arxiv_spec}{tag}")
    # Stable marker for batch callers.
    click.echo(f"RUN_ID={run_id}")
    # The parent graph is shallow (parser → summarizer → storyboarder →
    # run_scene fan-out → concat → emb_consolidate); recursion limit just
    # needs to cover the linear depth plus the Send fan-out step. The per-scene
    # recursion budget is set inside ``run_scene_node`` against the compiled
    # scene subgraph, not against this limit.
    recursion_limit = 50
    g = build_generation_graph()
    final = g.invoke(
        state, config={"recursion_limit": recursion_limit, "max_concurrency": scene_parallelism}
    )
    _print_summary(final)
    console.print(f"[green]Run dir:[/green] {run_dir(run_id)}")
    if final.get("fatal_error"):
        raise click.ClickException(final["fatal_error"])


@cli.command()
def info() -> None:
    """Print configuration and environment status."""
    from paper2manim.llm import CANONICAL_ROLES, current_model, current_provider

    console.print(
        json.dumps(
            {
                "provider": current_provider(),
                "models": {role: current_model(role) for role in CANONICAL_ROLES},
                "PAPER2MANIM_RUNS_DIR": str(settings.PAPER2MANIM_RUNS_DIR),
                "PAPER2MANIM_MAX_RETRIES": settings.PAPER2MANIM_MAX_RETRIES,
                "PAPER2MANIM_QUALITY": settings.PAPER2MANIM_QUALITY,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    cli()
