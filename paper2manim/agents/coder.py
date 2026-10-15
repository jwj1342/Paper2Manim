"""Coder agent: Storyboard scene + (optional error feedback) -> Manim Python code.

When the EMB is enabled, the graph populates ``state["retrieved_success"]`` and
``state["retrieved_failure"]`` ahead of this node; we inject them as
*Reference Examples* (soft guidance) and *Known Pitfalls* (hard constraints)
respectively. Empty banks retain both prompt slots with an explicit empty marker.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any

from paper2manim.artifacts import append_trace, save_attempt_code
from paper2manim.emb.retrieval import (
    render_known_pitfalls_block,
    render_reference_examples_block,
)
from paper2manim.llm import get_llm
from paper2manim.prompts import load_prompt

if TYPE_CHECKING:
    from paper2manim.graphs.scene_graph import SceneState

log = logging.getLogger(__name__)

_PY_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)\n```", re.DOTALL)


def extract_python_block(text: str) -> str:
    """Extract the first ```python ... ``` block; fall back to the full text stripped."""
    m = _PY_BLOCK_RE.search(text)
    if m:
        return m.group(1).strip()
    # If the model returned bare code, return as-is
    return text.strip()


def _build_user_prompt(state: SceneState) -> str:
    sb = state.get("storyboard")
    if not sb:
        raise ValueError("coder: storyboard missing in state")
    idx = state.get("current_scene_idx", 0)
    scene = sb["scenes"][idx]

    blocks: list[str] = []
    blocks.append("## Current scene to implement\n")
    blocks.append(json.dumps(scene, ensure_ascii=False, indent=2))

    blocks.append("\n## Project conventions\n")
    blocks.append(load_prompt("manim_skill_rules"))

    # Keep the same two slots for empty and populated banks.
    if state.get("emb_enabled"):
        blocks.append("\n" + render_reference_examples_block(state.get("retrieved_success") or []))
        blocks.append("\n" + render_known_pitfalls_block(state.get("retrieved_failure") or []))

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


def coder_node(state: SceneState) -> dict[str, Any]:
    sb = state.get("storyboard")
    if not sb:
        return {"fatal_error": "coder: storyboard missing"}
    idx = state.get("current_scene_idx", 0)
    if idx >= len(sb["scenes"]):
        return {"fatal_error": f"coder: scene index {idx} out of range"}

    scene = sb["scenes"][idx]
    user = _build_user_prompt(state)
    system = load_prompt("coder")
    llm = get_llm("scene_coder", temperature=0.1)
    log.info(
        "[coder] scene=%s iter=%d prompt_chars=%d",
        scene["name"],
        state.get("iter_count", 0),
        len(user),
    )
    raw = llm.invoke([("system", system), ("user", user)]).content
    code = extract_python_block(raw if isinstance(raw, str) else str(raw))

    v_rev = state.get("vlm_revision_count", 0)
    artifact_scene = scene["name"] if v_rev == 0 else f"{scene['name']}_v{v_rev}"
    if state.get("run_id"):
        save_attempt_code(state["run_id"], artifact_scene, state.get("iter_count", 0), code)
        append_trace(
            state["run_id"],
            "coder",
            {
                "scene": scene["name"],
                "iter": state.get("iter_count", 0),
                "code_chars": len(code),
                "is_retry": bool(state.get("error_feedback")),
                "v_rev": v_rev,
            },
        )
    return {
        "current_code": code,
        "text_retry_count": state.get("text_retry_count", 0)
        + int(bool(state.get("error_feedback"))),
    }
