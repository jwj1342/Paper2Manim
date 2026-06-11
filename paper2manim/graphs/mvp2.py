"""MVP 2.0 graph: PDF/arXiv -> parser -> summarizer -> storyboarder ->
                 [Send fan-out × N_scenes -> scene_graph] ->
                 concat -> emb_consolidate -> END.

Within each scene, the inner reflection / VLM loops live in
:mod:`paper2manim.graphs.scene_graph`. This file owns only the paper-level
plumbing: parsing, summarization, storyboarding, scene fan-out, concat, and
end-of-run EMB consolidation.

Concurrency at the per-scene layer is provided by ``langgraph.types.Send``:
``fan_out_scenes`` emits one Send per scene; LangGraph's internal executor
runs the resulting branches concurrently up to the step's configured
parallelism. Shared resources (Manim render, LLM rate) are bounded by
:mod:`paper2manim.concurrency`.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from langgraph.graph import END, StateGraph
from langgraph.types import Send

from paper2manim.agents.narrator import narrator_node
from paper2manim.agents.storyboarder import storyboarder_node
from paper2manim.agents.summarizer import summarizer_node
from paper2manim.artifacts import append_trace, run_dir
from paper2manim.emb.distill import consolidate_run, infer_source_metadata
from paper2manim.graphs.scene_graph import _emb_for_state as _scene_emb_for_state
from paper2manim.graphs.scene_graph import get_compiled_scene_graph
from paper2manim.llm import tts_config as get_tts_config
from paper2manim.parsers import parse_arxiv, parse_local_pdf
from paper2manim.state import PaperState
from paper2manim.voiceover.assembly import assemble_voiceover

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


# --------------------------------------------------------------------------- #
# Scene fan-out
# --------------------------------------------------------------------------- #


def _make_scene_payload(state: PaperState, idx: int) -> dict[str, Any]:
    """Build the SceneState payload passed via Send for scene #idx.

    The payload carries the read-only paper context that the per-scene nodes
    look up by key (``storyboard``, ``summary``, ``run_id``, etc.) plus the
    per-scene mutables initialized to their starting values.
    """
    sb = state.get("storyboard") or {"title": "", "scenes": []}
    scene = sb["scenes"][idx] if idx < len(sb["scenes"]) else {"name": "<oob>", "description": ""}
    return {
        # Read-only paper context
        "run_id": state.get("run_id", ""),
        "storyboard": sb,
        "current_scene_idx": idx,
        "scene": scene,
        "summary": state.get("summary"),
        "quality": state.get("quality", "l"),
        "skip_render": bool(state.get("skip_render", False)),
        "max_retries": int(state.get("max_retries", 3)),
        "max_visual_revisions": int(state.get("max_visual_revisions", 2)),
        "vlm_enabled": bool(state.get("vlm_enabled", False)),
        # EMB context (forwarded so each scene's emb_retrieve resolves the same
        # bank the parent configured)
        "emb_enabled": bool(state.get("emb_enabled", False)),
        "emb_store_path": state.get("emb_store_path"),
        "emb_use_faiss": bool(state.get("emb_use_faiss", True)),
        "emb_use_real_embedder": bool(state.get("emb_use_real_embedder", True)),
        "emb_instance": state.get("emb_instance"),
        # B6 channel ablation: forwarded so per-scene emb_retrieve_node sees
        # the same toggles. Write-side toggles stay on PaperState (used by
        # parent emb_consolidate_node).
        "emb_no_success_channel": bool(state.get("emb_no_success_channel", False)),
        "emb_no_failure_channel": bool(state.get("emb_no_failure_channel", False)),
        # Per-scene mutables — start fresh for each fan-out branch
        "current_code": None,
        "iter_count": 0,
        "error_feedback": None,
        "vlm_revision_count": 0,
        "current_montage_path": None,
        "last_visual_review": None,
        "retrieved_success": [],
        "retrieved_failure": [],
        "attempts": [],
        "visual_revision_decisions": [],
    }


def fan_out_scenes(state: PaperState) -> list[Send]:
    """Emit one ``Send`` per storyboard scene, targeting ``run_scene``.

    LangGraph's Pregel engine schedules the resulting branches on its thread
    pool; the per-scene cap is set by the step's recursion limit and by the
    process-global throttles in :mod:`paper2manim.concurrency`.
    """
    if state.get("fatal_error"):
        return []
    sb = state.get("storyboard") or {"scenes": []}
    scenes = sb.get("scenes", [])
    if not scenes:
        return []
    log.info("[fan_out] %d scene(s)", len(scenes))
    return [Send("run_scene", _make_scene_payload(state, i)) for i in range(len(scenes))]


def run_scene_node(payload: dict[str, Any]) -> dict[str, Any]:
    """Send target: invoke the compiled scene subgraph for one scene.

    Returns updates that bubble back to ``PaperState`` via reducers:
    ``attempts``, ``rendered_videos``, ``skipped_scenes``,
    ``visual_revision_decisions``, ``retrieved_success`` / ``retrieved_failure``
    (latest), and ``scene_reports`` (one summary dict per scene).
    """
    scene = payload.get("scene") or {}
    scene_name = scene.get("name", "<unknown>")
    log.info("[run_scene] %s starting", scene_name)
    scene_graph = get_compiled_scene_graph()
    # Per-scene recursion budget: worst-case = 1 emb_retrieve + (1+max_retries)
    # × (coder+render+reviewer) + (frame_sampler+vlm_review+visual_revise) ×
    # max_visual_revisions. Give a healthy multiplier.
    max_retries = int(payload.get("max_retries", 3))
    max_vrev = int(payload.get("max_visual_revisions", 2))
    recursion_limit = max(60, 6 + 3 * (max_retries + 1) + 5 * max_vrev + 4)
    try:
        final = scene_graph.invoke(payload, config={"recursion_limit": recursion_limit})
    except Exception as exc:  # noqa: BLE001
        log.warning("[run_scene] %s raised %s — skipping scene", scene_name, exc)
        return {
            "skipped_scenes": [scene_name],
            "scene_reports": [
                {
                    "scene": scene_name,
                    "status": "fatal",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            ],
        }

    attempts = final.get("attempts", [])
    last_rr = attempts[-1].get("render_result", {}) if attempts else {}
    renditions = final.get("scene_renditions", []) or []
    # NB: per-scene fields like ``retrieved_success`` / ``retrieved_failure`` /
    # ``current_code`` are deliberately NOT bubbled back to PaperState. Two
    # parallel branches each returning their own values would trigger
    # LangGraph's InvalidUpdateError on those non-reducer keys. Per-scene
    # context lives inside SceneState; PaperState only sees the reducer-merged
    # outputs (attempts, rendered_videos, skipped_scenes, scene_reports,
    # visual_revision_decisions).
    updates: dict[str, Any] = {
        "attempts": attempts,
        "visual_revision_decisions": final.get("visual_revision_decisions", []),
        "scene_reports": [
            {
                "scene": scene_name,
                "n_attempts": len(attempts),
                "final_status": last_rr.get("status"),
                "v_revs": final.get("vlm_revision_count", 0),
                # Per-scene VLM verdict (post-parallelism, this is the place
                # to read scene-keyed VLM decisions from PaperState).
                "last_visual_review": final.get("last_visual_review"),
            }
        ],
    }
    if last_rr.get("status") == "success" and last_rr.get("video_path"):
        # Best-of-N: when the VLM scored multiple renditions, ship the
        # highest-scoring video — not the most recent. The visual-revise loop
        # sometimes produces a strictly worse v(N+1) (see
        # docs/vlm_experiment.md regression case); shipping the latest in that
        # case puts a worse video in ``rendered_videos`` AND a worse code in
        # EMB.success (because emb/distill.py:find_scored_scenes already picks
        # the highest-VLM-avg v_rev). Aligning ``rendered_videos`` with the
        # same v_rev keeps the two artifacts coherent.
        chosen = _pick_best_rendition(renditions, scene_name) or {
            "video_path": last_rr["video_path"],
            "v_rev": final.get("vlm_revision_count", 0),
            "avg_score": None,
        }
        chosen_video = chosen.get("video_path") or last_rr["video_path"]
        updates["rendered_videos"] = [chosen_video]
        # Structured record for voiceover: {scene, video_path, duration_s}.
        # Duration is probed later in assemble_av; store None here as a
        # placeholder so the scene→video mapping is available regardless.
        updates["rendered_scene_videos"] = [
            {
                "scene": scene_name,
                "video_path": chosen_video,
                "duration_s": None,
            }
        ]
        if chosen_video != last_rr.get("video_path"):
            try:
                append_trace(
                    final.get("run_id", ""),
                    "advance",
                    {
                        "scene": scene_name,
                        "chose": {
                            "v_rev": chosen.get("v_rev"),
                            "avg_score": chosen.get("avg_score"),
                            "video_path": chosen_video,
                        },
                        "skipped_last": {
                            "v_rev": final.get("vlm_revision_count", 0),
                            "video_path": last_rr.get("video_path"),
                        },
                    },
                )
            except Exception:  # noqa: BLE001
                pass
            log.info(
                "[run_scene] %s best-of-N picked v%s (score=%s) over latest v%d",
                scene_name,
                chosen.get("v_rev"),
                chosen.get("avg_score"),
                final.get("vlm_revision_count", 0),
            )
        else:
            log.info("[run_scene] %s OK -> %s", scene_name, chosen_video)
    else:
        updates["skipped_scenes"] = [scene_name]
        log.warning(
            "[run_scene] %s gave up after %d attempt(s); recorded in skipped_scenes",
            scene_name,
            len(attempts),
        )
    return updates


def _pick_best_rendition(
    renditions: list[dict], scene_name: str
) -> dict | None:
    """Return the highest-``avg_score`` rendition for ``scene_name``, or None.

    Renditions without a usable ``video_path`` or with ``avg_score is None``
    are not eligible — those signal frame-sampler failure or auto-pass
    branches where the VLM never produced a comparable score. With ties on
    score, prefer the earliest v_rev so we don't reward a tie-with-regression.
    Returns ``None`` when no eligible rendition exists; the caller falls back
    to "last attempt" semantics in that case.
    """
    eligible = [
        r for r in renditions
        if r.get("scene") == scene_name
        and r.get("video_path")
        and r.get("avg_score") is not None
    ]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda r: (float(r["avg_score"]), -int(r.get("v_rev", 0))),
    )


# --------------------------------------------------------------------------- #
# AV assembly — concat (silent) + optional voiceover (TTS + alignment + mux)
# --------------------------------------------------------------------------- #


def assemble_av_node(state: PaperState) -> dict[str, Any]:
    """Assemble the final video: concat scene mp4s, and optionally add voiceover.

    Delegates to :func:`paper2manim.voiceover.assembly.assemble_voiceover`
    for the voiceover path. When voiceover is off, this is a thin wrapper
    around ``concat_videos`` (identical to the old ``concat_node``).
    """
    voiceover_enabled = bool(state.get("voiceover_enabled", False))

    scene_videos = state.get("rendered_scene_videos", [])
    if not scene_videos:
        # Fallback for legacy callers that only populate rendered_videos.
        videos = state.get("rendered_videos", [])
        scene_videos = [
            {"scene": f"scene_{i:02d}", "video_path": vp, "duration_s": None}
            for i, vp in enumerate(videos)
        ]
        if not scene_videos:
            return {"fatal_error": "assemble_av: no successful scenes to concatenate"}

    result = assemble_voiceover(
        run_id=state["run_id"],
        scene_videos=scene_videos,
        narration_plan=state.get("narration_plan") if voiceover_enabled else None,
        tts_cfg=get_tts_config() if voiceover_enabled else None,
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


def emb_consolidate_node(state: PaperState) -> dict[str, Any]:
    """End-of-run §4.4 sink: distill the trace into success/failure records.

    ``emb_readonly`` (B6) short-circuits this node — required by RQ3
    cross-domain test phase so the frozen Domain-A EMB doesn't absorb
    Domain-B records mid-experiment. The retrieve path stays active so prior
    records still inject into the Coder prompt; only WRITE is frozen.
    """
    if not state.get("emb_enabled"):
        return {}
    if state.get("emb_readonly"):
        try:
            append_trace(
                state.get("run_id", ""), "emb_consolidate",
                {"skipped": True, "reason": "emb_readonly"},
            )
        except Exception:  # noqa: BLE001
            pass
        return {}
    # The scene-level cache + the parent share the same `_EMB_CACHE` dict in
    # scene_graph; reuse its resolver so we don't double-build.
    emb = _scene_emb_for_state(
        {  # type: ignore[arg-type]
            "emb_enabled": True,
            "emb_instance": state.get("emb_instance"),
            "emb_store_path": state.get("emb_store_path"),
            "emb_use_faiss": state.get("emb_use_faiss", True),
            "emb_use_real_embedder": state.get("emb_use_real_embedder", True),
        }
    )
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
        from paper2manim.agents.lesson_distiller import distill_lesson_llm
        from paper2manim.agents.rationale_writer import write_rationale_llm

        rw = write_rationale_llm
        ld = distill_lesson_llm
    # Defaults match emb/distill.py on the proposal §4.2 0–100 schema; CLI
    # `--emb-theta-high` / `--emb-failure-min-margin` should already be in this
    # range, but a stale state dict (older test fixture, hand-built invocation)
    # falls back to these.
    theta = float(state.get("emb_theta_high", 85.0))
    fail_margin = float(state.get("emb_failure_min_margin", 5.0))
    domain = state.get("dataset_domain") or ""
    skip_success = bool(state.get("emb_no_success_channel", False))
    skip_failure = bool(state.get("emb_no_failure_channel", False))
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
            domain=domain,
            skip_success=skip_success,
            skip_failure=skip_failure,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("[emb_consolidate] failed for run %s: %s", run_id, exc)
        return {}
    return {"emb_writes": [report.to_dict()]}


# --------------------------------------------------------------------------- #
# Conditional edges
# --------------------------------------------------------------------------- #


def _is_fatal(next_node: str):
    def predicate(state: PaperState) -> Literal["END", "next"]:
        return "END" if state.get("fatal_error") else "next"

    return predicate, {"END": END, "next": next_node}


def _post_storyboarder(state: PaperState) -> list[Send] | str:
    """Route storyboarder → narrator (if voiceover) or directly fan-out.

    When voiceover is disabled, skip the narrator entirely to avoid an
    unnecessary LLM call and keep ``--no-voiceover`` identical to the
    pre-voiceover code path.
    """
    if state.get("fatal_error"):
        return END  # type: ignore[return-value]
    sb = state.get("storyboard") or {"scenes": []}
    if not sb.get("scenes"):
        return END  # type: ignore[return-value]
    if state.get("voiceover_enabled"):
        return "narrator"
    return fan_out_scenes(state)


def _post_narrator(state: PaperState) -> list[Send] | str:
    """After narrator: if fatal or no scenes → END; otherwise fan-out to scenes."""
    if state.get("fatal_error"):
        return END  # type: ignore[return-value]
    sb = state.get("storyboard") or {"scenes": []}
    if not sb.get("scenes"):
        return END  # type: ignore[return-value]
    return fan_out_scenes(state)


# --------------------------------------------------------------------------- #
# Graph builder
# --------------------------------------------------------------------------- #


def build_mvp2_graph():
    g = StateGraph(PaperState)
    g.add_node("parser", parser_node)
    g.add_node("summarizer", summarizer_node)
    g.add_node("storyboarder", storyboarder_node)
    g.add_node("narrator", narrator_node)
    g.add_node("run_scene", run_scene_node)
    g.add_node("assemble_av", assemble_av_node)
    g.add_node("emb_consolidate", emb_consolidate_node)

    g.set_entry_point("parser")

    for src, dst in [("parser", "summarizer"), ("summarizer", "storyboarder")]:
        pred, mapping = _is_fatal(dst)
        g.add_conditional_edges(src, pred, mapping)

    # Storyboarder → narrator (voiceover on) or direct fan-out to run_scene
    # (voiceover off). Both routes end at END on fatal/no-scenes.
    g.add_conditional_edges(
        "storyboarder",
        _post_storyboarder,
        {"narrator": "narrator", "run_scene": "run_scene", END: END},
    )

    # Narrator → fan-out to run_scene (or END on fatal/no-scenes).
    g.add_conditional_edges(
        "narrator", _post_narrator, {"run_scene": "run_scene", END: END}
    )

    # After all run_scene branches finish, assemble_av → emb_consolidate → END.
    g.add_edge("run_scene", "assemble_av")
    g.add_edge("assemble_av", "emb_consolidate")
    g.add_edge("emb_consolidate", END)
    return g.compile()


__all__ = [
    "assemble_av_node",
    "build_mvp2_graph",
    "emb_consolidate_node",
    "fan_out_scenes",
    "parser_node",
    "run_scene_node",
]
