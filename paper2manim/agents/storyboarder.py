"""Storyboarder agent: source text/summary -> structured Storyboard JSON."""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.exceptions import OutputParserException
from pydantic import ValidationError

from paper2manim.artifacts import append_trace, save_json
from paper2manim.llm import get_llm, safe_structured_invoke
from paper2manim.prompts import load_prompt
from paper2manim.schemas import StoryboardModel
from paper2manim.state import PaperState

log = logging.getLogger(__name__)


def _format_figure_manifest(figures: list[dict]) -> str:
    """Render a concise text manifest so the storyboarder LLM can route figures to scenes.

    Format per figure:
      fig_001 type=schematic redrawable=True salience=high
        summary: encoder-decoder architecture overview
        elements: encoder, decoder, attention

    Figures with semantics=None (VLM disabled / failed) get only fig_id + a note.
    """
    lines: list[str] = []
    for f in figures:
        fig_id = f.get("fig_id", "?")
        sem = f.get("semantics") or {}
        if not sem:
            lines.append(f"- {fig_id} (no semantics; can still embed as ImageMobject if needed)")
            continue
        ft = sem.get("fig_type", "?")
        rd = sem.get("redrawable")
        sal = sem.get("salience", "?")
        summary = sem.get("one_line_summary", "")
        elements = sem.get("key_elements", []) or []
        line = f"- {fig_id} type={ft} redrawable={rd} salience={sal}"
        if summary:
            line += f"\n    summary: {summary}"
        if elements:
            line += f"\n    elements: {', '.join(elements)}"
        lines.append(line)
    return "\n".join(lines)


def _format_table_manifest(tables: list[dict]) -> str:
    """Render a concise text manifest so the storyboarder LLM can decide which scene uses which table.

    Format per table:
      tab_001 [markdown] header=[col1, col2] n_rows=3
        first_row: col1=val1, col2=val2

    If structured parse failed (header/rows are None), fall back to a short raw_md preview.
    """
    lines: list[str] = []
    for t in tables:
        tab_id = t.get("tab_id", "?")
        fmt = t.get("fmt", "?")
        header = t.get("header")
        rows = t.get("rows") or []
        if header:
            n_rows = len(rows)
            head_str = ", ".join(header)
            line = f"- {tab_id} [{fmt}] header=[{head_str}] n_rows={n_rows}"
            if rows:
                first = rows[0]
                # strict=False: malformed tables may have shorter rows; truncate to
                # the shared prefix rather than crashing in a manifest-preview path.
                preview = ", ".join(
                    f"{h}={v}" for h, v in zip(header, first, strict=False)
                )
                line += f"\n    first_row: {preview}"
        else:
            preview = (t.get("raw_md", "") or "")[:120].replace("\n", " ")
            line = f"- {tab_id} [{fmt}] raw_preview: {preview!r}"
        lines.append(line)
    return "\n".join(lines)


def storyboarder_node(state: PaperState) -> dict[str, Any]:
    sb_text = (
        json.dumps(state["summary"], ensure_ascii=False, indent=2)
        if state.get("summary")
        else state.get("raw_text", "")
    )
    if not sb_text:
        return {"fatal_error": "storyboarder: no input (raw_text and summary both empty)"}

    # Phase 1: when the parser preserved structured tables, surface them as a manifest
    # so the storyboarder can route specific tables into specific scenes via
    # SceneModel.referenced_tables.
    tables = state.get("tables") or []
    if tables:
        sb_text = (
            sb_text
            + "\n\n## Available tables (route via SceneModel.referenced_tables)\n"
            + _format_table_manifest(tables)
        )

    # Phase 2b: when figure_understander has enriched figures with semantics, surface a
    # similar manifest so the storyboarder can route specific figures into specific scenes
    # via SceneModel.referenced_figures. Coder decides redraw-vs-embed from the semantics.
    figures = state.get("figures") or []
    if figures:
        sb_text = (
            sb_text
            + "\n\n## Available figures (route via SceneModel.referenced_figures)\n"
            + _format_figure_manifest(figures)
        )

    system = load_prompt("storyboarder")
    llm = get_llm("flash", temperature=0.3)
    log.info(
        "[storyboarder] input chars=%d, tables_in_manifest=%d, figures_in_manifest=%d",
        len(sb_text),
        len(tables),
        len(figures),
    )
    try:
        sb = safe_structured_invoke(
            llm, StoryboardModel, [("system", system), ("user", sb_text)], retries=1
        )
    except (OutputParserException, ValidationError) as exc:
        return {"fatal_error": f"storyboarder: schema parse failed: {type(exc).__name__}: {str(exc)[:200]}"}
    sb_dict = sb.model_dump()

    if state.get("run_id"):
        save_json(state["run_id"], "storyboard", sb_dict)
        append_trace(
            state["run_id"],
            "storyboarder",
            {"title": sb_dict["title"], "n_scenes": len(sb_dict["scenes"])},
        )

    return {"storyboard": sb_dict, "current_scene_idx": 0, "iter_count": 0}
