"""MVP 1.0 graph: text -> storyboarder -> coder -> render -> END.

No reflection loop; if render fails the run terminates with fatal_error set.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, StateGraph

from paper2manim.agents.coder import coder_node
from paper2manim.agents.storyboarder import storyboarder_node
from paper2manim.artifacts import (
    append_trace,
    copy_final_video,
    run_dir,
    save_attempt_result,
)
from paper2manim.sandbox.render import render
from paper2manim.state import PaperState

log = logging.getLogger(__name__)


def render_node(state: PaperState) -> dict[str, Any]:
    if state.get("skip_render"):
        log.info("[render] skip_render set — not invoking manim")
        return {}

    sb = state.get("storyboard")
    if not sb:
        return {"fatal_error": "render: storyboard missing"}
    if not state.get("current_code"):
        return {"fatal_error": "render: current_code missing"}

    scene = sb["scenes"][state.get("current_scene_idx", 0)]
    workdir = run_dir(state["run_id"]) / "attempts" / f"work_{scene['name']}_00"
    workdir.mkdir(parents=True, exist_ok=True)

    rr = render(
        state["current_code"],
        scene["name"],
        quality=state.get("quality", "l"),
        workdir=workdir,
    )

    save_attempt_result(state["run_id"], scene["name"], state.get("iter_count", 0), rr)
    append_trace(
        state["run_id"],
        "render",
        {"scene": scene["name"], "status": rr["status"], "category": rr.get("category")},
    )

    attempt = {
        "iter": state.get("iter_count", 0),
        "scene": scene["name"],
        "code": state["current_code"],
        "render_result": rr,
        "reviewer_decision": None,
        "reviewer_hint": None,
    }

    if rr["status"] != "success":
        return {
            "attempts": [attempt],
            "fatal_error": f"render failed: category={rr.get('category')} msg={rr.get('error_message')}",
        }

    final_path = copy_final_video(state["run_id"], rr["video_path"], "output.mp4")
    return {
        "attempts": [attempt],
        "rendered_videos": [str(rr["video_path"])],
        "final_video_path": str(final_path),
    }


def build_mvp1_graph():
    g = StateGraph(PaperState)
    g.add_node("storyboarder", storyboarder_node)
    g.add_node("coder", coder_node)
    g.add_node("render", render_node)
    g.set_entry_point("storyboarder")
    g.add_edge("storyboarder", "coder")
    g.add_edge("coder", "render")
    g.add_edge("render", END)
    return g.compile()
