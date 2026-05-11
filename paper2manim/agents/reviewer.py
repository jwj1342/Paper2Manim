"""MVP 2.0 Reviewer: inspect last RenderResult, decide retry/give_up, give a hint."""

from __future__ import annotations

import json
import logging
from typing import Any

from paper2manim.artifacts import append_trace
from paper2manim.llm import get_llm
from paper2manim.prompts import load_prompt
from paper2manim.state import PaperState

log = logging.getLogger(__name__)


def _build_review_prompt(state: PaperState) -> str:
    attempts = state.get("attempts", [])
    last = attempts[-1] if attempts else {}
    rr = last.get("render_result", {})

    blocks: list[str] = []
    blocks.append(f"## Iteration {state.get('iter_count', 0)} of {state.get('max_retries', 3)}\n")
    blocks.append(f"Scene: `{rr.get('scene')}`\n")
    blocks.append(f"Status: `{rr.get('status')}` Category: `{rr.get('category')}`\n")
    blocks.append(f"Error: {rr.get('error_message')}\n")
    if rr.get("traceback_tail"):
        blocks.append("\n### Traceback tail\n```\n" + rr["traceback_tail"] + "\n```\n")
    if rr.get("source_excerpt"):
        blocks.append("\n### Source excerpt\n```\n")
        for ln in rr["source_excerpt"]:
            blocks.append(f"{ln['line']:4d}: {ln['code']}\n")
        blocks.append("```\n")
    if rr.get("tex_log_excerpt"):
        blocks.append("\n### TeX log\n```\n" + rr["tex_log_excerpt"] + "\n```\n")
    if len(attempts) >= 2:
        blocks.append("\n### Prior attempts (categories)\n")
        for a in attempts[-3:-1]:
            arr = a.get("render_result", {})
            blocks.append(f"- iter {a.get('iter')}: {arr.get('category')} / {arr.get('error_message')}\n")
    blocks.append(
        '\nRespond with strict JSON: {"decision": "retry" | "give_up", "hint": "..."}\n'
        "If the same category fails twice in a row with same error_message -> give_up.\n"
    )
    return "".join(blocks)


def reviewer_node(state: PaperState) -> dict[str, Any]:
    attempts = state.get("attempts", [])
    if not attempts:
        return {"error_feedback": None}
    last = attempts[-1]
    rr = last.get("render_result", {})

    # Success short-circuit — no LLM call needed
    if rr.get("status") == "success":
        log.info("[reviewer] scene %s succeeded; no review needed", rr.get("scene"))
        return {"error_feedback": None}

    # Hard cap
    iter_count = state.get("iter_count", 0)
    max_retries = state.get("max_retries", 3)
    if iter_count >= max_retries:
        log.warning(
            "[reviewer] hard cap hit (iter=%d max=%d); giving up on %s",
            iter_count,
            max_retries,
            rr.get("scene"),
        )
        ef = {
            "decision": "give_up",
            "hint": f"max_retries={max_retries} exhausted",
            "render_result": rr,
        }
        if state.get("run_id"):
            append_trace(
                state["run_id"], "reviewer", {"decision": "give_up", "scene": rr.get("scene")}
            )
        return {"error_feedback": ef, "iter_count": iter_count + 1}

    # LLM review
    system = load_prompt("reviewer")
    user = _build_review_prompt(state)
    llm = get_llm("flash", temperature=0.0)
    raw = llm.invoke([("system", system), ("user", user)]).content
    decision, hint = _parse_review(raw)

    ef = {"decision": decision, "hint": hint, "render_result": rr}
    if state.get("run_id"):
        append_trace(
            state["run_id"],
            "reviewer",
            {"decision": decision, "scene": rr.get("scene"), "iter": iter_count},
        )
    log.info(
        "[reviewer] scene=%s iter=%d decision=%s hint=%r",
        rr.get("scene"),
        iter_count,
        decision,
        (hint or "")[:120],
    )
    return {"error_feedback": ef, "iter_count": iter_count + 1}


def _parse_review(raw: Any) -> tuple[str, str]:
    """Best-effort JSON parse; fall back to regex."""
    text = raw if isinstance(raw, str) else str(raw)
    # try to grab JSON object
    try:
        # strip code fences
        import re

        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            obj = json.loads(m.group(0))
            decision = obj.get("decision", "retry")
            if decision not in {"retry", "give_up"}:
                decision = "retry"
            return decision, str(obj.get("hint", "")).strip()
    except (json.JSONDecodeError, ValueError):
        pass
    # fallback: heuristic
    decision = "give_up" if "give_up" in text.lower() else "retry"
    return decision, text.strip()[:500]
