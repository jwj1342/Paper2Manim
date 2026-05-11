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
from paper2manim.parsers import parse_arxiv, parse_local_pdf
from paper2manim.sandbox.concat import concat_videos
from paper2manim.sandbox.render import render
from paper2manim.state import PaperState

log = logging.getLogger(__name__)


def parser_node(state: PaperState) -> dict[str, Any]:
    kind = state.get("input_kind")
    try:
        if kind == "arxiv":
            spec = state.get("arxiv_spec")
            if not spec:
                return {"fatal_error": "parser: arxiv_spec missing"}
            parsed = parse_arxiv(spec, section=state.get("arxiv_section"))
        elif kind == "pdf":
            pdf = state.get("pdf_path")
            if not pdf:
                return {"fatal_error": "parser: pdf_path missing"}
            parsed = parse_local_pdf(pdf)
        else:
            return {"fatal_error": f"parser: unsupported input_kind {kind!r} for MVP 2.0"}
    except Exception as exc:  # arxiv download error / Marker import error / etc.
        return {"fatal_error": f"parser: {type(exc).__name__}: {exc}"}

    if state.get("run_id"):
        ext = "tex" if parsed.fmt == "latex" else "md"
        (run_dir(state["run_id"]) / f"parsed.{ext}").write_text(parsed.text, encoding="utf-8")
        append_trace(
            state["run_id"],
            "parser",
            {"chars": len(parsed.text), "fmt": parsed.fmt, "source": parsed.source},
        )
    return {
        "parsed_markdown": parsed.text,
        "parsed_format": parsed.fmt,
        "parser_source": parsed.source,
    }


def init_scene_node(state: PaperState) -> dict[str, Any]:
    """Reset per-scene fields when starting a new scene.

    Does NOT touch ``fatal_error`` — that field is reserved for graph-level
    fatal errors that must propagate to END. Per-scene give_ups are recorded
    via ``skipped_scenes`` (populated by ``advance_scene_node``).
    """
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
    return {"iter_count": 0, "error_feedback": None, "current_code": None}


def render_node(state: PaperState) -> dict[str, Any]:
    # Defensive guards: graph may reach here with missing state if fatal_error
    # propagation hasn't kicked in yet, or if a prior node set an inconsistent state.
    if state.get("skip_render") or state.get("fatal_error"):
        return {}
    sb = state.get("storyboard")
    if not sb or "scenes" not in sb:
        return {"fatal_error": "render: storyboard missing or malformed"}
    idx = state.get("current_scene_idx", 0)
    if idx >= len(sb["scenes"]):
        return {"fatal_error": f"render: scene_idx {idx} out of range (n_scenes={len(sb['scenes'])})"}
    if not state.get("current_code"):
        return {"fatal_error": "render: current_code is empty"}
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
    """If last scene succeeded, append its mp4 to ``rendered_videos``; otherwise
    record the scene name in ``skipped_scenes``. Then bump scene index.

    Does NOT touch ``fatal_error`` — see ``init_scene_node`` docstring.
    """
    attempts = state.get("attempts", [])
    out: dict[str, Any] = {"current_scene_idx": state.get("current_scene_idx", 0) + 1}
    if attempts:
        last = attempts[-1]
        rr = last.get("render_result", {})
        scene_name = rr.get("scene", "<unknown>")
        if rr.get("status") == "success" and rr.get("video_path"):
            out["rendered_videos"] = [rr["video_path"]]
            log.info("[advance] scene %s OK -> %s", scene_name, rr["video_path"])
        else:
            out["skipped_scenes"] = [scene_name]
            log.warning(
                "[advance] scene %s gave up after %d attempt(s); recorded in skipped_scenes",
                scene_name,
                len(attempts),
            )
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


def _is_fatal(next_node: str):
    """Build a conditional edge that routes to END when fatal_error is set,
    otherwise to ``next_node``."""

    def predicate(state: PaperState) -> Literal["END", "next"]:
        return "END" if state.get("fatal_error") else "next"

    return predicate, {"END": END, "next": next_node}


def should_retry(state: PaperState) -> Literal["coder", "advance"]:
    """After reviewer: retry coder vs. advance to next scene."""
    if state.get("fatal_error"):
        # A fatal error from a downstream guard (e.g., render_node missing inputs)
        # is treated like a give_up for the current scene; advance will skip it.
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

    # Upstream stages: any fatal_error short-circuits to END instead of
    # cascading into per-scene nodes that can't recover.
    for src, dst in [
        ("parser", "summarizer"),
        ("summarizer", "storyboarder"),
        ("storyboarder", "init_scene"),
    ]:
        pred, mapping = _is_fatal(dst)
        g.add_conditional_edges(src, pred, mapping)

    g.add_edge("init_scene", "coder")
    g.add_edge("coder", "render")
    g.add_edge("render", "reviewer")
    g.add_conditional_edges("reviewer", should_retry, {"coder": "coder", "advance": "advance"})
    g.add_conditional_edges(
        "advance", has_more_scenes, {"init_scene": "init_scene", "concat": "concat"}
    )
    g.add_edge("concat", END)
    return g.compile()
