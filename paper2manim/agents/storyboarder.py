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


def storyboarder_node(state: PaperState) -> dict[str, Any]:
    sb_text = (
        json.dumps(state["summary"], ensure_ascii=False, indent=2)
        if state.get("summary")
        else state.get("raw_text", "")
    )
    if not sb_text:
        return {"fatal_error": "storyboarder: no input (raw_text and summary both empty)"}

    system = load_prompt("storyboarder")
    llm = get_llm("flash", temperature=0.3)
    log.info("[storyboarder] input chars=%d", len(sb_text))
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
