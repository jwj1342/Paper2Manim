"""MVP 2.0 graph: PDF/arXiv -> parser -> summarizer -> storyboarder ->
                 [init_scene -> coder -> render -> reviewer ->
                    (error: reflection back to coder)
                    (success: optional frame_sampler -> vlm_review ->
                              (revise: visual_revise -> render)
                              (pass / fail / cap: advance)
                 ]* -> concat -> END.

Two interleaved retry loops:

* **Reflection** (existing) — render error feedback fed back into ``coder``.
  Capped at ``max_retries`` per scene.
* **VLM Multi-Dim Scoring** (new) — once a scene renders, frame_sampler builds a
  montage and ``vlm_reviewer`` returns a 6-dim verdict. ``revise`` routes to
  ``visual_reviser`` (which rewrites the script) → ``render`` again. Capped at
  ``max_visual_revisions`` per scene.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

from langgraph.graph import END, StateGraph

from paper2manim.agents.coder import coder_node
from paper2manim.agents.reviewer import reviewer_node
from paper2manim.agents.storyboarder import storyboarder_node
from paper2manim.agents.summarizer import summarizer_node
from paper2manim.agents.visual_revision_agent import revise_code
from paper2manim.agents.vlm_scene_reviewer import review_scene
from paper2manim.artifacts import (
    append_trace,
    run_dir,
    save_attempt_code,
    save_attempt_result,
)
from paper2manim.emb.distill import consolidate_run, infer_source_metadata
from paper2manim.emb.manager import EpisodicMemoryBank, build_default_emb
from paper2manim.emb.retrieval import retrieve_for_scene, summarize_bundle
from paper2manim.parsers import parse_arxiv, parse_local_pdf
from paper2manim.sandbox.concat import concat_videos
from paper2manim.sandbox.render import render
from paper2manim.state import PaperState
from paper2manim.utils.frame_sampler import FrameSamplerError, sample_frames_montage

log = logging.getLogger(__name__)


# ---- EMB lookup ----------------------------------------------------------- #
# Path-keyed cache so we don't reload SQLite + sentence-transformers on every
# scene. Tests can inject a pre-built EMB via ``state['emb_instance']`` to
# bypass the cache entirely.
_EMB_CACHE: dict[str, EpisodicMemoryBank] = {}


def _reset_emb_cache() -> None:
    """Drop all cached EMB instances. Used by tests; safe in production too."""
    _EMB_CACHE.clear()


def _emb_for_state(state: PaperState) -> EpisodicMemoryBank | None:
    """Resolve the EMB instance for the current run, or ``None`` when off."""
    if not state.get("emb_enabled"):
        return None
    injected = state.get("emb_instance")
    if isinstance(injected, EpisodicMemoryBank):
        return injected
    path = state.get("emb_store_path")
    if not path:
        return None
    path_s = str(path)
    if path_s not in _EMB_CACHE:
        try:
            _EMB_CACHE[path_s] = build_default_emb(
                path_s,
                use_faiss=bool(state.get("emb_use_faiss", True)),
                use_real_embedder=bool(state.get("emb_use_real_embedder", True)),
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("[emb] build_default_emb failed for %s: %s — disabling EMB", path_s, exc)
            return None
    return _EMB_CACHE[path_s]


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


def emb_retrieve_node(state: PaperState) -> dict[str, Any]:
    """Query the EMB for top-k success / failure records for the current scene.

    Always returns valid ``retrieved_success`` / ``retrieved_failure`` lists
    so the Coder can read them unconditionally. When the EMB is disabled,
    empty, or unavailable we return empty lists — equivalent to zero-shot.
    """
    empty = {"retrieved_success": [], "retrieved_failure": []}
    emb = _emb_for_state(state)
    if emb is None:
        return empty
    sb = state.get("storyboard") or {}
    scenes = sb.get("scenes", [])
    idx = state.get("current_scene_idx", 0)
    if idx >= len(scenes):
        return empty
    scene = scenes[idx]
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
                "scene": scene["name"],
                "stats": emb.stats(),
                "hits": summarize_bundle(bundle),
            },
        )
    except Exception:  # noqa: BLE001 — trace failures must not block the graph
        pass
    return {
        "retrieved_success": wire["success"],
        "retrieved_failure": wire["failure"],
    }


def emb_consolidate_node(state: PaperState) -> dict[str, Any]:
    """End-of-run §4.4 sink: distill the trace into success/failure records.

    Idempotent over re-runs of the same ``run_id`` only in that re-running it
    will append duplicate records (no dedupe layer yet); call it exactly once
    per finished run, which is what the graph does.
    """
    if not state.get("emb_enabled"):
        return {}
    emb = _emb_for_state(state)
    if emb is None:
        return {}
    run_id = state.get("run_id")
    if not run_id:
        return {}
    source_paper, source_section = infer_source_metadata(state)
    use_llm = bool(state.get("emb_use_llm_distillers", False))
    rw = None
    ld = None
    if use_llm:
        # Lazy import: agents.rationale_writer + agents.lesson_distiller both
        # import the LLM factory, which we'd rather skip when the distillers
        # are off.
        from paper2manim.agents.lesson_distiller import distill_lesson_llm
        from paper2manim.agents.rationale_writer import write_rationale_llm

        rw = write_rationale_llm
        ld = distill_lesson_llm
    theta = float(state.get("emb_theta_high", 4.0))
    fail_margin = float(state.get("emb_failure_min_margin", 0.5))
    try:
        report = consolidate_run(
            run_id,
            emb,
            state=dict(state),
            theta_high=theta,
            source_paper=source_paper,
            source_section=source_section,
            rationale_writer=rw,
            lesson_distiller=ld,
            failure_min_margin=fail_margin,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("[emb_consolidate] failed for run %s: %s", run_id, exc)
        return {}
    return {"emb_writes": [report.to_dict()]}


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
    return {
        "iter_count": 0,
        "error_feedback": None,
        "current_code": None,
        "vlm_revision_count": 0,
        "current_montage_path": None,
        "last_visual_review": None,
    }


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
        return {
            "fatal_error": f"render: scene_idx {idx} out of range (n_scenes={len(sb['scenes'])})"
        }
    if not state.get("current_code"):
        return {"fatal_error": "render: current_code is empty"}
    scene = sb["scenes"][idx]
    iter_idx = state.get("iter_count", 0)
    vrev = state.get("vlm_revision_count", 0)
    # Workdir name includes both counters so visual revisions don't trample the
    # original render's artifacts.
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
    attempt = {
        "iter": iter_idx,
        "scene": scene["name"],
        "code": state["current_code"],
        "render_result": rr,
        "reviewer_decision": None,
        "reviewer_hint": None,
    }
    return {"attempts": [attempt]}


def frame_sampler_node(state: PaperState) -> dict[str, Any]:
    """Build a horizontal montage of N frames from the latest scene mp4."""
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


def vlm_review_node(state: PaperState) -> dict[str, Any]:
    """Run the VLM multi-dim review on the latest montage."""
    montage = state.get("current_montage_path")
    sb = state.get("storyboard") or {}
    idx = state.get("current_scene_idx", 0)
    scenes = sb.get("scenes", [])
    if not montage or idx >= len(scenes):
        # No montage / out of range — treat as auto-pass so we don't block.
        log.warning("[vlm_review] missing montage or scene; auto-pass")
        review = {
            "scene_id": scenes[idx]["name"] if idx < len(scenes) else "<unknown>",
            "decision": "pass",
            "scores": {},
            "revision_instruction": "",
        }
        return {"last_visual_review": review, "visual_revision_decisions": ["pass"]}
    scene = scenes[idx]
    summary = state.get("summary")
    try:
        review = review_scene(scene, montage, summary=summary, scene_idx=idx)
    except Exception as exc:
        log.warning(
            "[vlm_review] %s raised %s: %s — auto-pass to keep graph moving",
            scene["name"],
            type(exc).__name__,
            exc,
        )
        review = {
            "scene_id": scene["name"],
            "decision": "pass",
            "scores": {},
            "revision_instruction": "",
            "raw_response": f"{type(exc).__name__}: {exc}",
        }
    append_trace(
        state["run_id"],
        "vlm_review",
        {
            "scene": scene["name"],
            "v_rev": state.get("vlm_revision_count", 0),
            "decision": review.get("decision"),
            "scores": review.get("scores"),
        },
    )
    return {
        "last_visual_review": review,
        "visual_revision_decisions": [str(review.get("decision", "pass"))],
    }


def visual_revise_node(state: PaperState) -> dict[str, Any]:
    """Apply the VLM's revision_instruction to the current Manim source."""
    sb = state.get("storyboard") or {}
    idx = state.get("current_scene_idx", 0)
    scenes = sb.get("scenes", [])
    if idx >= len(scenes):
        return {"fatal_error": f"visual_revise: scene_idx {idx} out of range"}
    scene = scenes[idx]
    review = state.get("last_visual_review") or {}
    current = state.get("current_code") or ""
    new_count = state.get("vlm_revision_count", 0) + 1
    log.info("[visual_revise] scene=%s v_rev=%d", scene["name"], new_count)
    try:
        new_code = revise_code(scene, current, review, summary=state.get("summary"))
    except Exception as exc:
        log.warning("[visual_revise] %s raised %s — keeping prior code", scene["name"], exc)
        new_code = current
    if state.get("run_id"):
        # Save under a distinct tag so we can compare pre/post revisions in the run dir.
        save_attempt_code(
            state["run_id"], f"{scene['name']}_v{new_count}", state.get("iter_count", 0), new_code
        )
        append_trace(state["run_id"], "visual_revise", {"scene": scene["name"], "v_rev": new_count})
    return {"current_code": new_code, "vlm_revision_count": new_count}


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
    append_trace(state["run_id"], "concat", {"n_videos": len(videos), "final": str(final)})
    return {"final_video_path": str(final)}


# ---- Conditional edges ----


def _is_fatal(next_node: str):
    """Build a conditional edge that routes to END when fatal_error is set,
    otherwise to ``next_node``."""

    def predicate(state: PaperState) -> Literal["END", "next"]:
        return "END" if state.get("fatal_error") else "next"

    return predicate, {"END": END, "next": next_node}


def post_reviewer_route(state: PaperState) -> Literal["coder", "frame_sampler", "advance"]:
    """After reviewer: pick reflection retry, visual review entry, or advance."""
    if state.get("fatal_error"):
        return "advance"
    attempts = state.get("attempts", [])
    if not attempts:
        return "advance"
    last = attempts[-1]
    rr = last.get("render_result", {})
    if rr.get("status") == "success":
        # Render passed → optionally enter the VLM loop, otherwise advance.
        if state.get("vlm_enabled") and rr.get("video_path"):
            return "frame_sampler"
        return "advance"
    # Render failed → reflection retry logic
    if state.get("iter_count", 0) >= state.get("max_retries", 3):
        return "advance"
    ef = state.get("error_feedback") or {}
    if ef.get("decision") == "give_up":
        return "advance"
    return "coder"


def post_vlm_route(state: PaperState) -> Literal["visual_revise", "advance"]:
    """After vlm_review: revise (if not at cap) or advance."""
    review = state.get("last_visual_review") or {}
    decision = str(review.get("decision") or "pass").lower()
    if decision == "pass":
        return "advance"
    if decision == "fail":
        # Soft-fail policy (intentional): a "fail" verdict from the VLM is logged
        # in trace.jsonl + last_visual_review but does NOT exclude the scene from
        # rendered_videos — the render itself succeeded, and at this VLM
        # maturity (Claude rarely self-passes, see docs/vlm_experiment.md) a
        # strict-fail policy would frequently leave final/output.mp4 empty. If
        # we ever want strict-fail, gate it on a CLI flag and have
        # advance_scene_node consult last_visual_review.decision.
        return "advance"
    # decision == "revise"
    if state.get("vlm_revision_count", 0) >= state.get("max_visual_revisions", 2):
        log.info("[vlm] cap reached for %s; advancing", review.get("scene_id"))
        return "advance"
    return "visual_revise"


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
    g.add_node("emb_retrieve", emb_retrieve_node)
    g.add_node("coder", coder_node)
    g.add_node("render", render_node)
    g.add_node("reviewer", reviewer_node)
    g.add_node("frame_sampler", frame_sampler_node)
    g.add_node("vlm_review", vlm_review_node)
    g.add_node("visual_revise", visual_revise_node)
    g.add_node("advance", advance_scene_node)
    g.add_node("concat", concat_node)
    g.add_node("emb_consolidate", emb_consolidate_node)

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

    # EMB retrieval runs once per scene, between init_scene and coder. The
    # reviewer's retry edge skips it on purpose — within-scene retries reuse
    # the same retrieved records.
    g.add_edge("init_scene", "emb_retrieve")
    g.add_edge("emb_retrieve", "coder")
    g.add_edge("coder", "render")
    g.add_edge("render", "reviewer")
    g.add_conditional_edges(
        "reviewer",
        post_reviewer_route,
        {"coder": "coder", "frame_sampler": "frame_sampler", "advance": "advance"},
    )
    g.add_edge("frame_sampler", "vlm_review")
    g.add_conditional_edges(
        "vlm_review",
        post_vlm_route,
        {"visual_revise": "visual_revise", "advance": "advance"},
    )
    # Visual revision rewrites the code; jump straight to render, bypassing
    # the text-error reflection path (reviewer would short-circuit anyway).
    g.add_edge("visual_revise", "render")
    g.add_conditional_edges(
        "advance", has_more_scenes, {"init_scene": "init_scene", "concat": "concat"}
    )
    # Consolidation runs once at the very end. It's a no-op when emb_enabled
    # is False, so leaving it unconditionally on the graph adds negligible
    # overhead to non-EMB runs.
    g.add_edge("concat", "emb_consolidate")
    g.add_edge("emb_consolidate", END)
    return g.compile()
