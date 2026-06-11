"""Narrator agent: summary + storyboard -> structured NarrationPlan JSON.

Runs before the scene fan-out in MVP 2.0 (between storyboarder and
fan_out_scenes), and between storyboarder and coder in MVP 1.0. The narrator
needs the global paper summary and the full storyboard to produce coherent
multi-scene narration — it cannot run per-scene.

Voiceover text never enters the Manim code path. Generated code stays visual;
all audio work (TTS, alignment, mux) happens in the ``assemble_av`` node after
render succeeds.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.exceptions import OutputParserException
from pydantic import ValidationError

from paper2manim.artifacts import append_trace, save_json
from paper2manim.llm import get_llm, safe_structured_invoke
from paper2manim.prompts import load_prompt
from paper2manim.schemas.narration import NarrationPlanModel
from paper2manim.state import PaperState

log = logging.getLogger(__name__)


def _build_narrator_prompt(state: PaperState) -> str:
    """Build the user prompt: summary JSON + storyboard JSON + language hint."""
    blocks: list[str] = []

    summary = state.get("summary")
    if summary:
        blocks.append("## Paper summary\n```json\n")
        blocks.append(json.dumps(summary, ensure_ascii=False, indent=2))
        blocks.append("\n```\n")

    sb = state.get("storyboard")
    if sb:
        blocks.append("## Storyboard\n```json\n")
        blocks.append(json.dumps(sb, ensure_ascii=False, indent=2))
        blocks.append("\n```\n")

    language = state.get("voiceover_language") or "en"
    blocks.append(f"## Target language\n{language}\n")

    blocks.append(
        "\n## Task\nGenerate the NarrationPlan JSON matching the schema "
        "described in the system prompt. One scene entry per storyboard scene, "
        f"in the same order. Use language='{language}' for every scene.\n"
    )
    return "".join(blocks)


def narrator_node(state: PaperState) -> dict[str, Any]:
    """Generate a narration plan from the summary and storyboard.

    Returns ``{"narration_plan": ...}`` on success, or
    ``{"fatal_error": ...}`` when the LLM output cannot be parsed.
    """
    sb = state.get("storyboard")
    if not sb:
        return {"fatal_error": "narrator: storyboard missing"}
    if not state.get("summary") and not state.get("raw_text"):
        return {"fatal_error": "narrator: summary and raw_text both missing"}

    system = load_prompt("narrator")
    user = _build_narrator_prompt(state)
    llm = get_llm("scene_planner", temperature=0.3)

    log.info(
        "[narrator] scenes=%d language=%s prompt_chars=%d",
        len(sb.get("scenes", [])),
        state.get("voiceover_language") or "en",
        len(user),
    )

    try:
        plan = safe_structured_invoke(
            llm, NarrationPlanModel, [("system", system), ("user", user)], retries=1
        )
    except (OutputParserException, ValidationError) as exc:
        return {
            "fatal_error": (
                f"narrator: schema parse failed: {type(exc).__name__}: {str(exc)[:200]}"
            )
        }

    plan_dict = plan.model_dump()

    # Coerce every scene's language to the requested language so a stray LLM
    # output doesn't produce a mixed-language narration.json.
    requested_lang = (state.get("voiceover_language") or "en").strip()
    for s in plan_dict.get("scenes", []):
        s["language"] = requested_lang

    if state.get("run_id"):
        save_json(state["run_id"], "narration_plan", plan_dict)
        append_trace(
            state["run_id"],
            "narrator",
            {
                "title": plan_dict.get("title"),
                "n_scenes": len(plan_dict.get("scenes", [])),
                "language": requested_lang,
            },
        )

    return {"narration_plan": plan_dict}
