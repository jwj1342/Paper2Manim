"""MVP 2.0 Summarizer agent: parsed markdown -> SummaryModel JSON."""

from __future__ import annotations

import logging
from typing import Any

from paper2manim.artifacts import append_trace, save_json
from paper2manim.llm import get_llm
from paper2manim.prompts import load_prompt
from paper2manim.schemas import SummaryModel
from paper2manim.state import PaperState

log = logging.getLogger(__name__)

# 经验阈值：>30K chars 用 pro 模型（256K context 还够，但 reviewer/coder 还要嵌进来，留余量）
_USE_PRO_CHAR_THRESHOLD = 30_000


def summarizer_node(state: PaperState) -> dict[str, Any]:
    md = state.get("parsed_markdown")
    if not md:
        return {"fatal_error": "summarizer: parsed_markdown missing"}

    model_alias = "pro" if len(md) > _USE_PRO_CHAR_THRESHOLD else "flash"
    log.info("[summarizer] model=%s md_chars=%d", model_alias, len(md))
    system = load_prompt("summarizer")
    llm = get_llm(model_alias, temperature=0.2, max_tokens=4096).with_structured_output(
        SummaryModel
    )
    summary: SummaryModel = llm.invoke([("system", system), ("user", md)])
    summary_dict = summary.model_dump()

    if state.get("run_id"):
        save_json(state["run_id"], "summary", summary_dict)
        append_trace(
            state["run_id"],
            "summarizer",
            {"model": model_alias, "title": summary_dict["title"]},
        )
    return {"summary": summary_dict}
