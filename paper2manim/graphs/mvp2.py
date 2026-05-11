"""MVP 2.0 graph: PDF -> Marker -> Summarizer -> Storyboarder ->
                 [init_scene -> coder -> render -> reviewer]* -> concat -> END.

Reflection loop is the conditional edge `should_retry` after `reviewer`.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from langgraph.graph import END, StateGraph

from paper2manim.agents.coder import coder_node
from paper2manim.agents.reviewer import reviewer_node
from paper2manim.agents.storyboarder import storyboarder_node
from paper2manim.agents.summarizer import summarizer_node
from paper2manim.artifacts import (
    append_trace,
    copy_final_video,
    run_dir,
    save_attempt_result,
)
from paper2manim.parsers.marker import parse_pdf
from paper2manim.sandbox.concat import concat_videos
from paper2manim.sandbox.render import render
from paper2manim.state import PaperState

log = logging.getLogger(__name__)


def parser_node(state: PaperState) -> dict[str, Any]:
    pdf = state.get("pdf_path")
    if not pdf:
        return {"fatal_error": "parser: pdf_path missing"}
    md = parse_pdf(pdf)
    if state.get("run_id"):
        (run_dir(state["run_id"]) / "parsed.md").write_text(md, encoding="utf-8")
        append_trace(state["run_id"], "parser", {"chars": len(md)})
    return {"parsed_markdown": md}


def init_scene_node(state: PaperState) -> dict[str, Any]:
    """Reset per-scene fields when starting a new scene."""
    sb = state.get("storyboard")
    if not sb:
        return {"fatal_error": "init_scene: storyboard missing"}
    idx = state.get("current_scene_idx", 0)
    log.info(
        "[init_scene] starting scene %d/%d: %s",
        idx + 1,
        len(sb["scenes"]),
        sb["scenes"][idx]["name"] if idx < len(sb["scenes"]) else "<oob>",
    )
    return {"iter_count": 0, "error_feedback": None, "current_code": None, "fatal_error": None}


def render_node(state: PaperState) -> dict[str, Any]:
    if state.get("skip_render"):
        return {}
    sb = state["storyboard"]
    idx = state["current_scene_idx"]
    scene = sb["scenes"][idx]
    iter_idx = state.get("iter_count", 0)

    workdir = run_dir(state["run_id"]) / "attempts" / f"work_{scene['name']}_{iter_idx:02d}"
    workdir.mkdir(parents=True, exist_ok=True)
    rr = render(
        state["current_code"],
        scene["name"],
        quality=state.get("quality", "l"),
        workdir=workdir,
    )
    save_attempt_result(state["run_id"], scene["name"], iter_idx, rr)
    append_trace(
        state["run_id"],
        "render",
        {"scene": scene["name"], "iter": iter_idx, "status": rr["status"], "category": rr.get("category")},
    )
    attempt = {
        "iter": iter_idx,
        "scene": scene["name"],
        "code": state["current_code"],
        "render_result": rr,
        "reviewer_decision": None,
        "reviewer_hint": None,
    }
    return {"attempts": [attempt]}


def advance_scene_node(state: PaperState) -> dict[str, Any]:
    """If last scene succeeded, append its mp4 to rendered_videos. Then bump scene index."""
    attempts = state.get("attempts", [])
    out: dict[str, Any] = {"current_scene_idx": state.get("current_scene_idx", 0) + 1}
    if attempts:
        last = attempts[-1]
        rr = last.get("render_result", {})
        if rr.get("status") == "success" and rr.get("video_path"):
            out["rendered_videos"] = [rr["video_path"]]
            log.info("[advance] scene %s OK -> %s", rr["scene"], rr["video_path"])
        else:
            log.warning("[advance] scene %s gave up after %d attempt(s)", rr.get("scene"), len(attempts))
    out["fatal_error"] = None  # reset; one bad scene shouldn't kill the whole run
    return out


def concat_node(state: PaperState) -> dict[str, Any]:
    videos = state.get("rendered_videos", [])
    if not videos:
        return {"fatal_error": "concat: no successful scenes to concatenate"}
    out_path = run_dir(state["run_id"]) / "final" / "output.mp4"
    final = concat_videos(videos, out_path)
    append_trace(
        state["run_id"], "concat", {"n_videos": len(videos), "final": str(final)}
    )
    return {"final_video_path": str(final)}


# ---- Conditional edges ----


def should_retry(state: PaperState) -> Literal["coder", "advance"]:
    """After reviewer: retry coder vs. advance to next scene."""
    if state.get("fatal_error"):
        return "advance"
    attempts = state.get("attempts", [])
    if not attempts:
        return "advance"
    last = attempts[-1]
    if last.get("render_result", {}).get("status") == "success":
        return "advance"
    if state.get("iter_count", 0) >= state.get("max_retries", 3):
        return "advance"
    ef = state.get("error_feedback") or {}
    if ef.get("decision") == "give_up":
        return "advance"
    return "coder"


def has_more_scenes(state: PaperState) -> Literal["init_scene", "concat"]:
    sb = state.get("storyboard")
    if not sb:
        return "concat"
    return "init_scene" if state["current_scene_idx"] < len(sb["scenes"]) else "concat"


def build_mvp2_graph():
    g = StateGraph(PaperState)
    g.add_node("parser", parser_node)
    g.add_node("summarizer", summarizer_node)
    g.add_node("storyboarder", storyboarder_node)
    g.add_node("init_scene", init_scene_node)
    g.add_node("coder", coder_node)
    g.add_node("render", render_node)
    g.add_node("reviewer", reviewer_node)
    g.add_node("advance", advance_scene_node)
    g.add_node("concat", concat_node)

    g.set_entry_point("parser")
    g.add_edge("parser", "summarizer")
    g.add_edge("summarizer", "storyboarder")
    g.add_edge("storyboarder", "init_scene")
    g.add_edge("init_scene", "coder")
    g.add_edge("coder", "render")
    g.add_edge("render", "reviewer")
    g.add_conditional_edges("reviewer", should_retry, {"coder": "coder", "advance": "advance"})
    g.add_conditional_edges(
        "advance", has_more_scenes, {"init_scene": "init_scene", "concat": "concat"}
    )
    g.add_edge("concat", END)
    return g.compile()
