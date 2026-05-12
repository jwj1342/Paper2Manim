"""Coder agent: Storyboard scene + (optional error feedback) -> Manim Python code."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from paper2manim.artifacts import append_trace, save_attempt_code
from paper2manim.llm import get_llm
from paper2manim.prompts import load_prompt
from paper2manim.state import PaperState

log = logging.getLogger(__name__)

_PY_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)\n```", re.DOTALL)


def extract_python_block(text: str) -> str:
    """Extract the first ```python ... ``` block; fall back to the full text stripped."""
    m = _PY_BLOCK_RE.search(text)
    if m:
        return m.group(1).strip()
    # If the model returned bare code, return as-is
    return text.strip()


def _resolve_referenced_tables(scene: dict, all_tables: list[dict]) -> list[dict]:
    """Look up the TableAsset dicts referenced by this scene's `referenced_tables` ids."""
    ref_ids = scene.get("referenced_tables") or []
    if not ref_ids:
        return []
    by_id = {t.get("tab_id"): t for t in all_tables}
    return [by_id[r] for r in ref_ids if r in by_id]


def _format_table_for_coder(tab: dict) -> str:
    """Render one TableAsset in a compact, coder-friendly JSON block."""
    payload = {
        "tab_id": tab.get("tab_id"),
        "header": tab.get("header"),
        "rows": tab.get("rows"),
    }
    if payload["header"] is None or payload["rows"] is None:
        # Structured parse failed — fall back to the raw block so coder can still try
        payload["raw_md"] = tab.get("raw_md")
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _resolve_referenced_figures(scene: dict, all_figures: list[dict]) -> list[dict]:
    """Look up the FigureAsset dicts referenced by this scene's `referenced_figures` ids."""
    ref_ids = scene.get("referenced_figures") or []
    if not ref_ids:
        return []
    by_id = {f.get("fig_id"): f for f in all_figures}
    return [by_id[r] for r in ref_ids if r in by_id]


def _format_figure_for_coder(fig: dict) -> str:
    """Render one FigureAsset in a compact, coder-friendly JSON block.

    The coder decides between two rendering paths based on the payload it sees:
    - ``recipe`` present → reproduce via Manim using the FigureRecipe template
    - ``recipe`` is None → embed the original via ``ImageMobject(path)``
    """
    sem = fig.get("semantics") or {}
    payload = {
        "fig_id": fig.get("fig_id"),
        "path": fig.get("path"),
        "fig_type": sem.get("fig_type"),
        "redrawable": sem.get("redrawable"),
        "one_line_summary": sem.get("one_line_summary"),
        "key_elements": sem.get("key_elements", []),
        "recipe": fig.get("recipe"),  # None when not redrawable or VLM produced no recipe
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _build_user_prompt(state: PaperState) -> str:
    sb = state.get("storyboard")
    if not sb:
        raise ValueError("coder: storyboard missing in state")
    idx = state.get("current_scene_idx", 0)
    scene = sb["scenes"][idx]

    blocks: list[str] = []
    blocks.append("## Current scene to implement\n")
    blocks.append(json.dumps(scene, ensure_ascii=False, indent=2))

    # Phase 1: when the scene routed tables, inject the structured data + a pointer to
    # the table-rendering rule in manim_skill_rules.md. Tables stay opt-in per scene.
    referenced = _resolve_referenced_tables(scene, state.get("tables") or [])
    if referenced:
        blocks.append("\n## Tables to render in this scene\n")
        blocks.append(
            "Render each table below using Manim's `Table` mobject "
            "(see Project conventions → Rendering tables). "
            "Use the structured `header` and `rows` verbatim — do not invent or omit cells.\n\n"
        )
        for tab in referenced:
            blocks.append(_format_table_for_coder(tab))
            blocks.append("\n")

    # Phase 2b/3: figures get one of two treatments based on `redrawable`. The coder
    # consults the JSON payload below + Project conventions → "Embedding paper figures"
    # vs "Reproducing figures via FigureRecipe".
    referenced_figs = _resolve_referenced_figures(scene, state.get("figures") or [])
    if referenced_figs:
        blocks.append("\n## Figures to show in this scene\n")
        blocks.append(
            "For each figure below: if `recipe` is present, REPRODUCE it in Manim using the "
            "FigureRecipe template (see Project conventions → Reproducing figures). "
            "If `recipe` is null, EMBED the original via `ImageMobject(path)` "
            "(see Project conventions → Embedding paper figures). "
            "Use the figure's `one_line_summary` to write a 1-sentence on-screen caption.\n\n"
        )
        for fig in referenced_figs:
            blocks.append(_format_figure_for_coder(fig))
            blocks.append("\n")

    blocks.append("\n## Project conventions\n")
    blocks.append(load_prompt("manim_skill_rules"))

    if state.get("error_feedback"):
        ef = state["error_feedback"]
        blocks.append("\n## Previous attempt failed — fix and retry\n")
        blocks.append(
            f"- category: {ef.get('render_result', {}).get('category')}\n"
            f"- error: {ef.get('render_result', {}).get('error_message')}\n"
            f"- reviewer hint: {ef.get('hint')}\n"
        )
        if state.get("current_code"):
            blocks.append("\n### Previous code (failed)\n```python\n")
            blocks.append(state["current_code"])
            blocks.append("\n```\n")
        rr = ef.get("render_result", {})
        if rr.get("traceback_tail"):
            blocks.append("\n### stderr (tail)\n```\n")
            blocks.append(rr["traceback_tail"])
            blocks.append("\n```\n")
        if rr.get("source_excerpt"):
            blocks.append("\n### Source excerpt around failing line\n```\n")
            for ln in rr["source_excerpt"]:
                blocks.append(f"{ln['line']:4d}: {ln['code']}\n")
            blocks.append("```\n")
        if rr.get("tex_log_excerpt"):
            blocks.append("\n### LaTeX log (key '!' lines)\n```\n")
            blocks.append(rr["tex_log_excerpt"])
            blocks.append("\n```\n")
    blocks.append(
        "\n## Output format\nReturn ONE ```python ... ``` block containing a complete, runnable script. "
        "Define a single Scene subclass whose name matches the requested scene.name exactly.\n"
    )
    return "".join(blocks)


def coder_node(state: PaperState) -> dict[str, Any]:
    sb = state.get("storyboard")
    if not sb:
        return {"fatal_error": "coder: storyboard missing"}
    idx = state.get("current_scene_idx", 0)
    if idx >= len(sb["scenes"]):
        return {"fatal_error": f"coder: scene index {idx} out of range"}

    scene = sb["scenes"][idx]
    user = _build_user_prompt(state)
    system = load_prompt("coder")
    llm = get_llm("flash", temperature=0.1)
    log.info(
        "[coder] scene=%s iter=%d prompt_chars=%d",
        scene["name"],
        state.get("iter_count", 0),
        len(user),
    )
    raw = llm.invoke([("system", system), ("user", user)]).content
    code = extract_python_block(raw if isinstance(raw, str) else str(raw))

    if state.get("run_id"):
        save_attempt_code(state["run_id"], scene["name"], state.get("iter_count", 0), code)
        append_trace(
            state["run_id"],
            "coder",
            {"scene": scene["name"], "iter": state.get("iter_count", 0), "code_chars": len(code)},
        )
    return {"current_code": code}
