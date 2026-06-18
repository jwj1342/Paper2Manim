"""MVP 1.0 graph: text -> storyboarder -> [narrator] -> coder -> render -> assemble_av -> END.

No reflection loop; if render fails the run terminates with fatal_error set.
Narrator runs only when ``voiceover_enabled`` is True (fix-plan §Mod 1).
AV assembly delegates to :mod:`paper2manim.voiceover.assembly`.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from langgraph.graph import END, StateGraph

from paper2manim.agents.coder import coder_node
from paper2manim.agents.narrator import narrator_node
from paper2manim.agents.storyboarder import storyboarder_node
from paper2manim.artifacts import append_trace, copy_final_video, run_dir, save_attempt_result
from paper2manim.llm import tts_config as get_tts_config
from paper2manim.sandbox.render import render
from paper2manim.state import PaperState
from paper2manim.voiceover.assembly import assemble_voiceover

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

    out: dict[str, Any] = {
        "attempts": [attempt],
        "rendered_videos": [str(rr["video_path"])],
        "rendered_scene_videos": [
            {
                "scene": scene["name"],
                "video_path": str(rr["video_path"]),
                "duration_s": None,
            }
        ],
    }

    # When voiceover is off, set final_video_path here (pre-voiceover behaviour).
    if not state.get("voiceover_enabled"):
        final_path = copy_final_video(state["run_id"], rr["video_path"], "output.mp4")
        out["final_video_path"] = str(final_path)
        out["silent_video_path"] = str(final_path)

    return out


def assemble_av_node(state: PaperState) -> dict[str, Any]:
    """Single-scene AV assembly — delegates to shared helper."""
    voiceover_enabled = bool(state.get("voiceover_enabled", False))
    if not voiceover_enabled:
        return {}

    run_id = state["run_id"]
    scene_videos = state.get("rendered_scene_videos", [])
    if not scene_videos:
        return {"fatal_error": "assemble_av: no rendered scene videos"}

    result = assemble_voiceover(
        run_id=run_id,
        scene_videos=scene_videos,
        narration_plan=state.get("narration_plan"),
        tts_cfg=get_tts_config(),
        voice_override=state.get("vo_tts_voice_override"),
        speed_override=state.get("vo_tts_speed_override"),
        language=state.get("voiceover_language") or "en",
        strict=bool(state.get("voiceover_strict", True)),
    )

    out: dict[str, Any] = {
        "final_video_path": result.final_video_path,
        "silent_video_path": result.silent_video_path,
        "narrated_video_path": result.narrated_video_path,
        "final_audio_path": result.final_audio_path,
        "tts_audio_paths": result.tts_audio_paths,
    }
    if result.fatal_error:
        out["fatal_error"] = result.fatal_error
    if result.warnings:
        out["voiceover_warnings"] = result.warnings
    return out


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #


def _post_storyboarder(state: PaperState) -> Literal["narrator", "coder", "END"]:
    """Route storyboarder: voiceover ON → narrator, else → coder."""
    if state.get("fatal_error"):
        return "END"  # type: ignore[return-value]
    if state.get("voiceover_enabled"):
        return "narrator"
    return "coder"


# --------------------------------------------------------------------------- #
# Graph builder
# --------------------------------------------------------------------------- #


def build_mvp1_graph():
    g = StateGraph(PaperState)
    g.add_node("storyboarder", storyboarder_node)
    g.add_node("narrator", narrator_node)
    g.add_node("coder", coder_node)
    g.add_node("render", render_node)
    g.add_node("assemble_av", assemble_av_node)
    g.set_entry_point("storyboarder")

    # Storyboarder → narrator (voiceover) or coder (no voiceover) or END (fatal).
    g.add_conditional_edges(
        "storyboarder",
        _post_storyboarder,
        {"narrator": "narrator", "coder": "coder", "END": END},
    )
    g.add_edge("narrator", "coder")
    g.add_edge("coder", "render")
    g.add_edge("render", "assemble_av")
    g.add_edge("assemble_av", END)
    return g.compile()
