"""Per-scene LangGraph subgraph.

The MVP 2.0 graph fans out scenes through ``langgraph.types.Send``. Each scene
runs its own compiled :class:`StateGraph` over a :class:`SceneState` — the
inner loop of coder / render / reviewer / VLM / visual-revise. Splitting the
per-scene work into its own graph lets multiple scenes run concurrently in
LangGraph's thread pool while sharing zero mutable state through the parent
``PaperState`` (everything passes through reducers).

Wire-up belongs in :mod:`paper2manim.graphs.mvp2`; this module is intentionally
self-contained so it can be unit-tested in isolation.
"""

from __future__ import annotations

import logging
import operator
import threading
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from paper2manim.agents.coder import coder_node
from paper2manim.agents.reviewer import reviewer_node
from paper2manim.agents.visual_revision_agent import revise_code
from paper2manim.agents.vlm_scene_reviewer import review_scene
from paper2manim.artifacts import (
    append_trace,
    run_dir,
    save_attempt_code,
    save_attempt_result,
)
from paper2manim.emb.manager import EpisodicMemoryBank, build_default_emb
from paper2manim.emb.retrieval import retrieve_for_scene, summarize_bundle
from paper2manim.sandbox.render import render
from paper2manim.state import Attempt, Scene, Storyboard
from paper2manim.utils.frame_sampler import FrameSamplerError, sample_frames_montage

log = logging.getLogger(__name__)


# ---- EMB cache (process-global, thread-safe) ----
# A scene fan-out fires N concurrent Send branches; without a lock they would
# all race to construct the same EMB. The lock guards the cache *lookup +
# construction*, not the cached object's own thread-safety (that is handled
# inside :class:`EpisodicMemoryBank` and the SQLite store).
_EMB_CACHE_LOCK = threading.Lock()
_EMB_CACHE: dict[str, EpisodicMemoryBank] = {}


def _reset_emb_cache() -> None:
    """Drop all cached EMB instances. Tests use this for isolation."""
    with _EMB_CACHE_LOCK:
        _EMB_CACHE.clear()


# --------------------------------------------------------------------------- #
# Scene-local state schema
# --------------------------------------------------------------------------- #


class SceneState(TypedDict, total=False):
    """Per-scene state passed to the scene subgraph.

    Carries (a) read-only paper context that the existing agent nodes already
    look up by key (``storyboard``, ``summary``, ``run_id``), (b) the per-scene
    mutables that used to live on ``PaperState`` (current_code, iter_count,
    error_feedback, vlm_revision_count, current_montage_path,
    last_visual_review), and (c) reducer-accumulated within-scene records
    (``attempts``, ``visual_revision_decisions``). The outputs that bubble up
    to ``PaperState`` (rendered_videos, skipped_scenes, scene_reports) are
    materialized by the Send target wrapper in :mod:`mvp2`.
    """

    # ---- Inputs (set by fan_out_scenes; read-only inside the subgraph) ----
    run_id: str
    storyboard: Storyboard
    current_scene_idx: int
    scene: Scene
    summary: dict | None
    quality: Literal["l", "m", "h"]
    skip_render: bool
    max_retries: int
    max_visual_revisions: int
    vlm_enabled: bool

    emb_enabled: bool
    emb_store_path: str | None
    emb_use_faiss: bool
    emb_use_real_embedder: bool
    emb_instance: object | None

    # ---- Per-scene mutables (managed by inner loop) ----
    current_code: str | None
    iter_count: int
    error_feedback: dict | None
    vlm_revision_count: int
    current_montage_path: str | None
    last_visual_review: dict | None

    # ---- Per-scene retrieved (output of emb_retrieve) ----
    retrieved_success: list[dict]
    retrieved_failure: list[dict]

    # ---- Reducer fields (accumulate within this scene only) ----
    attempts: Annotated[list[Attempt], operator.add]
    # Each entry is ``{"scene": scene_name, "decision": "pass|revise|fail"}``,
    # matching the parent-graph schema introduced in PR #15. The scene
    # subgraph appends the dict; the surrounding ``run_scene_node`` flushes
    # the per-scene reducer up to ``PaperState.visual_revision_decisions``.
    visual_revision_decisions: Annotated[list[dict[str, str]], operator.add]


# --------------------------------------------------------------------------- #
# EMB lookup (read by emb_retrieve_node)
# --------------------------------------------------------------------------- #


def _emb_for_state(state: SceneState) -> EpisodicMemoryBank | None:
    """Resolve the EMB instance for this scene, or ``None`` when off."""
    if not state.get("emb_enabled"):
        return None
    injected = state.get("emb_instance")
    if isinstance(injected, EpisodicMemoryBank):
        return injected
    path = state.get("emb_store_path")
    if not path:
        return None
    path_s = str(path)
    with _EMB_CACHE_LOCK:
        if path_s not in _EMB_CACHE:
            try:
                _EMB_CACHE[path_s] = build_default_emb(
                    path_s,
                    use_faiss=bool(state.get("emb_use_faiss", True)),
                    use_real_embedder=bool(state.get("emb_use_real_embedder", True)),
                )
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "[emb] build_default_emb failed for %s: %s — disabling EMB",
                    path_s,
                    exc,
                )
                return None
        return _EMB_CACHE[path_s]


# --------------------------------------------------------------------------- #
# Per-scene node functions (migrated from mvp2.py)
# --------------------------------------------------------------------------- #


def emb_retrieve_node(state: SceneState) -> dict[str, Any]:
    """Query EMB for top-k success / failure records for this scene."""
    empty = {"retrieved_success": [], "retrieved_failure": []}
    emb = _emb_for_state(state)
    if emb is None:
        return empty
    scene = state.get("scene") or {}
    scene_text = (scene.get("description") or scene.get("name") or "").strip()
    if not scene_text:
        return empty
    bundle = retrieve_for_scene(emb, scene_text, k_success=2, k_failure=3)
    wire = bundle.to_state_dict()
    try:
        append_trace(
            state["run_id"],
            "emb_retrieve",
            {
                "scene": scene.get("name", "<unknown>"),
                "stats": emb.stats(),
                "hits": summarize_bundle(bundle),
            },
        )
    except Exception:  # noqa: BLE001
        pass
    return {
        "retrieved_success": wire["success"],
        "retrieved_failure": wire["failure"],
    }


def render_node(state: SceneState) -> dict[str, Any]:
    """Render the current Manim code for this scene."""
    if state.get("skip_render"):
        return {}
    scene = state.get("scene")
    if not scene:
        return {}
    if not state.get("current_code"):
        # Coder failed silently — record a synthetic error so the reviewer can
        # decide retry/give_up rather than silently advancing.
        return {
            "attempts": [
                {
                    "iter": state.get("iter_count", 0),
                    "scene": scene["name"],
                    "code": "",
                    "render_result": {
                        "status": "error",
                        "category": "python",
                        "scene": scene["name"],
                        "error_message": "render: current_code is empty",
                    },
                    "reviewer_decision": None,
                    "reviewer_hint": None,
                }
            ]
        }
    iter_idx = state.get("iter_count", 0)
    vrev = state.get("vlm_revision_count", 0)
    workdir_tag = f"{iter_idx:02d}" if vrev == 0 else f"{iter_idx:02d}_v{vrev}"
    workdir = run_dir(state["run_id"]) / "attempts" / f"work_{scene['name']}_{workdir_tag}"
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
        {
            "scene": scene["name"],
            "iter": iter_idx,
            "v_rev": vrev,
            "status": rr["status"],
            "category": rr.get("category"),
        },
    )
    attempt: Attempt = {
        "iter": iter_idx,
        "scene": scene["name"],
        "code": state["current_code"],
        "render_result": rr,
        "reviewer_decision": None,
        "reviewer_hint": None,
    }
    return {"attempts": [attempt]}


def frame_sampler_node(state: SceneState) -> dict[str, Any]:
    attempts = state.get("attempts", [])
    if not attempts:
        return {"current_montage_path": None}
    rr = attempts[-1].get("render_result", {})
    video_path = rr.get("video_path")
    scene_name = rr.get("scene", "<unknown>")
    if not video_path:
        log.warning("[frame_sampler] no video_path for %s — skipping", scene_name)
        return {"current_montage_path": None}
    vrev = state.get("vlm_revision_count", 0)
    out_png = run_dir(state["run_id"]) / "vlm_frames" / f"{scene_name}_v{vrev}.png"
    try:
        sample_frames_montage(Path(video_path), out_png, n_frames=4)
    except FrameSamplerError as exc:
        log.warning("[frame_sampler] failed for %s: %s", scene_name, exc)
        return {"current_montage_path": None}
    append_trace(
        state["run_id"],
        "frame_sampler",
        {"scene": scene_name, "v_rev": vrev, "montage": str(out_png)},
    )
    return {"current_montage_path": str(out_png)}


def vlm_review_node(state: SceneState) -> dict[str, Any]:
    montage = state.get("current_montage_path")
    scene = state.get("scene") or {}
    scene_name = scene.get("name") if scene else "<unknown>"
    if not montage or not scene:
        log.warning("[vlm_review] missing montage or scene; auto-pass")
        review = {
            "scene_id": scene_name,
            "decision": "pass",
            "scores": {},
            "revision_instruction": "",
        }
        return {
            "last_visual_review": review,
            "visual_revision_decisions": [{"scene": scene_name, "decision": "pass"}],
        }
    summary = state.get("summary")
    try:
        review = review_scene(scene, montage, summary=summary, scene_idx=state.get("current_scene_idx", 0))
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "[vlm_review] %s raised %s: %s — auto-pass to keep graph moving",
            scene.get("name"),
            type(exc).__name__,
            exc,
        )
        review = {
            "scene_id": scene.get("name"),
            "decision": "pass",
            "scores": {},
            "revision_instruction": "",
            "raw_response": f"{type(exc).__name__}: {exc}",
        }
    append_trace(
        state["run_id"],
        "vlm_review",
        {
            "scene": scene.get("name"),
            "v_rev": state.get("vlm_revision_count", 0),
            "decision": review.get("decision"),
            "scores": review.get("scores"),
        },
    )
    return {
        "last_visual_review": review,
        "visual_revision_decisions": [
            {"scene": scene_name, "decision": str(review.get("decision", "pass"))}
        ],
    }


def visual_revise_node(state: SceneState) -> dict[str, Any]:
    scene = state.get("scene")
    if not scene:
        return {}
    review = state.get("last_visual_review") or {}
    current = state.get("current_code") or ""
    new_count = state.get("vlm_revision_count", 0) + 1
    log.info("[visual_revise] scene=%s v_rev=%d", scene["name"], new_count)
    try:
        new_code = revise_code(scene, current, review, summary=state.get("summary"))
    except Exception as exc:  # noqa: BLE001
        log.warning("[visual_revise] %s raised %s — keeping prior code", scene["name"], exc)
        new_code = current
    if state.get("run_id"):
        save_attempt_code(
            state["run_id"], f"{scene['name']}_v{new_count}", state.get("iter_count", 0), new_code
        )
        append_trace(state["run_id"], "visual_revise", {"scene": scene["name"], "v_rev": new_count})
    return {"current_code": new_code, "vlm_revision_count": new_count}


# --------------------------------------------------------------------------- #
# Routing predicates
# --------------------------------------------------------------------------- #


def post_reviewer_route(state: SceneState) -> Literal["coder", "frame_sampler", "end"]:
    attempts = state.get("attempts", [])
    if not attempts:
        return "end"
    last = attempts[-1]
    rr = last.get("render_result", {})
    if rr.get("status") == "success":
        if state.get("vlm_enabled") and rr.get("video_path"):
            return "frame_sampler"
        return "end"
    if state.get("iter_count", 0) >= state.get("max_retries", 3):
        return "end"
    ef = state.get("error_feedback") or {}
    if ef.get("decision") == "give_up":
        return "end"
    return "coder"


def post_vlm_route(state: SceneState) -> Literal["visual_revise", "end"]:
    review = state.get("last_visual_review") or {}
    decision = str(review.get("decision") or "pass").lower()
    if decision == "pass":
        return "end"
    if decision == "fail":
        # Soft-fail: render succeeded, so the scene is "done" for the
        # rendered_videos list. Strict-fail would require a CLI flag, kept
        # consistent with the original mvp2.py behavior.
        return "end"
    # decision == "revise"
    if state.get("vlm_revision_count", 0) >= state.get("max_visual_revisions", 2):
        log.info("[vlm] cap reached for %s; ending scene", review.get("scene_id"))
        return "end"
    return "visual_revise"


# --------------------------------------------------------------------------- #
# Graph builder
# --------------------------------------------------------------------------- #


def build_scene_graph():
    """Compile the per-scene subgraph.

    Topology mirrors the original mvp2 inner loop:
    ``emb_retrieve → coder → render → reviewer →
        (retry → coder | frame_sampler → vlm_review →
            (visual_revise → render | END))``
    """
    g = StateGraph(SceneState)
    g.add_node("emb_retrieve", emb_retrieve_node)
    g.add_node("coder", coder_node)
    g.add_node("render", render_node)
    g.add_node("reviewer", reviewer_node)
    g.add_node("frame_sampler", frame_sampler_node)
    g.add_node("vlm_review", vlm_review_node)
    g.add_node("visual_revise", visual_revise_node)

    g.add_edge(START, "emb_retrieve")
    g.add_edge("emb_retrieve", "coder")
    g.add_edge("coder", "render")
    g.add_edge("render", "reviewer")
    g.add_conditional_edges(
        "reviewer",
        post_reviewer_route,
        {"coder": "coder", "frame_sampler": "frame_sampler", "end": END},
    )
    g.add_edge("frame_sampler", "vlm_review")
    g.add_conditional_edges(
        "vlm_review",
        post_vlm_route,
        {"visual_revise": "visual_revise", "end": END},
    )
    g.add_edge("visual_revise", "render")
    return g.compile()


# Compile lazily so importers (and tests) don't pay LangGraph construction
# cost on import. Cached at first call.
_COMPILED_SCENE_GRAPH = None
_COMPILED_LOCK = threading.Lock()


def get_compiled_scene_graph():
    global _COMPILED_SCENE_GRAPH
    with _COMPILED_LOCK:
        if _COMPILED_SCENE_GRAPH is None:
            _COMPILED_SCENE_GRAPH = build_scene_graph()
    return _COMPILED_SCENE_GRAPH


__all__ = [
    "SceneState",
    "build_scene_graph",
    "emb_retrieve_node",
    "frame_sampler_node",
    "get_compiled_scene_graph",
    "post_reviewer_route",
    "post_vlm_route",
    "render_node",
    "visual_revise_node",
    "vlm_review_node",
    "_reset_emb_cache",
]
