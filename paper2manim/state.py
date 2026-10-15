"""LangGraph State schema — single source of truth for all nodes.

Paper-level outputs use reducers to merge independent scene branches.
Scene-local mutable values live in SceneState.
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

# ---- Substructures (TypedDict mirror of pydantic schemas) ----


class Scene(TypedDict):
    name: str
    description: str
    duration_hint: float
    paper_claim: str
    paper_evidence: str
    final_takeaway: str


class Storyboard(TypedDict):
    title: str
    scenes: list[Scene]


class SourceLine(TypedDict):
    line: int
    code: str


class RenderResult(TypedDict, total=False):
    status: Literal["success", "error"]
    category: Literal["python", "latex", "manim_runtime", "timeout", "unknown"] | None
    exit_code: int
    scene: str
    video_path: str | None
    error_type: str | None
    error_message: str | None
    traceback_tail: str | None
    source_excerpt: list[SourceLine] | None
    tex_log_excerpt: str | None
    workdir: str


class Attempt(TypedDict, total=False):
    iter: int
    scene: str
    code: str
    render_result: RenderResult
    reviewer_decision: Literal["retry", "done", "give_up"] | None
    reviewer_hint: str | None
    v_rev: int


# ---- Top-level State ----


class PaperState(TypedDict, total=False):
    # ---- Inputs ----
    run_id: str
    input_kind: Literal["text", "pdf", "arxiv"]
    raw_text: str | None  # Local section text
    pdf_path: str | None  # Local PDF
    arxiv_spec: str | None  # arXiv id / url / "arXiv:1706.03762"
    arxiv_section: str | None  # optional: slice a single \section{...} from source

    # ---- Generation pipeline parsing/summarization ----
    # `parsed_markdown` holds the parser's flattened text regardless of source format.
    # Use `parsed_format` to disambiguate: "markdown" (Marker) vs "latex" (arXiv source).
    parsed_markdown: str | None
    parsed_format: Literal["markdown", "latex"] | None
    parser_source: str | None  # origin tag, e.g. "arxiv-src:1706.03762"
    summary: dict | None  # SummaryModel.model_dump()

    # ---- Storyboard ----
    storyboard: Storyboard | None

    # ---- Reflection caps ----
    attempts: Annotated[list[Attempt], operator.add]  # reducer: append across scenes
    max_retries: int

    # ---- Output ----
    rendered_videos: Annotated[list[str], operator.add]  # mp4 paths in scene order
    skipped_scenes: Annotated[list[str], operator.add]  # scenes that gave up after retries
    final_video_path: str | None

    # ---- VLM caps + reducer outputs ----
    vlm_enabled: bool
    # Per-scene cap, enforced inside the scene subgraph.
    max_visual_revisions: int
    # Each entry is ``{"scene": scene_name, "decision": "pass|revise|fail"}``.
    # The reducer keeps appending across the whole run; slicing by ``scene``
    # gives you per-scene history without needing a per-scene reset. (Schema
    # introduced in PR #15.)
    visual_revision_decisions: Annotated[list[dict[str, str]], operator.add]
    # Scenes the VLM couldn't actually review — missing montage, VLM API
    # exception, or visual_revise_node hitting an exception. Distinct from:
    #   - ``skipped_scenes`` (text-reflection give_up after max_retries)
    #   - a recorded ``decision == "fail"`` in ``visual_revision_decisions``
    #     (the VLM did review and explicitly judged the scene unusable)
    # Reducer prevents earlier skips from being overwritten by later branches.
    vlm_skipped_scenes: Annotated[list[str], operator.add]
    # One structured summary per scene (generate parallel path only) — flushed
    # back from each Send branch by ``scene_graph`` and reduced via
    # ``operator.add``. Includes the final ``last_visual_review`` dict so
    # post-run inspectors don't lose per-scene VLM detail (previously
    # available as a top-level field, now scoped to SceneState).
    scene_reports: Annotated[list[dict], operator.add]

    # ---- Episodic Memory Bank (ManimAgent) ----
    # Enabled by `--emb`. `emb_store_path` points at the directory holding
    # `memory.db` + `{success,failure}.index`. The retrieve / consolidate
    # nodes look this up on each invocation and cache the loaded EMB by path.
    # `emb_instance` is a test-only escape hatch — when present, nodes use it
    # directly and skip the path-based cache.
    emb_enabled: bool
    emb_store_path: str | None
    # Both thresholds live on the three-axis scoring 0-100 schema. CLI defaults are
    # 85.0 / 5.0 (see paper2manim.cli generate). DO NOT default these to old 1-5
    # values when synthesizing test states — the production gate would never fire.
    emb_theta_high: float  # success-record acceptance threshold (0-100 avg score)
    emb_failure_min_margin: float  # min (after-before) gap for a failure record
    emb_use_faiss: bool
    emb_use_real_embedder: bool
    emb_instance: object | None  # test injection; opaque so TypedDict typecheck stays cheap
    retrieved_success: list[dict]  # wire-format records (no embedding payload)
    retrieved_failure: list[dict]
    emb_writes: Annotated[list[dict], operator.add]  # consolidation reports, one per run/scene
    # Read-only memory and optional channel switches.
    emb_readonly: bool
    domain: str | None
    task_text: str
    scene_role: str
    emb_k_success: int
    emb_k_failure: int
    emb_no_success_channel: bool
    emb_no_failure_channel: bool

    # ---- Voiceover / TTS ----
    # ``narration_plan`` is set by the narrator node before scene fan-out.
    # ``voiceover_enabled`` gates TTS synthesis + mux in assemble_av.
    # ``voiceover_strict`` (default True) makes TTS/alignment failures fatal;
    # when False, best-effort degradation (silence padding, speed-up) keeps the
    # graph moving and writes warnings into ``voiceover_warnings``.
    narration_plan: dict | None
    voiceover_enabled: bool
    voiceover_strict: bool
    voiceover_language: str | None
    # CLI overrides for TTS config (--tts-voice, --tts-speed).
    # When set, these take precedence over config.yaml tts.voice / tts.speed.
    vo_tts_voice_override: str | None
    vo_tts_speed_override: float | None
    # Accumulated across scenes by the assemble_av node (single writer; no fan-out).
    tts_audio_paths: list[str]
    final_audio_path: str | None
    # ``silent_video_path`` is always produced (or copied from the existing
    # concat output). ``narrated_video_path`` is only set when voiceover +
    # mux succeed. When voiceover is off, ``narrated_video_path`` is None
    # and ``final_video_path`` = ``silent_video_path``.
    silent_video_path: str | None
    narrated_video_path: str | None
    voiceover_warnings: Annotated[list[dict], operator.add]
    # Structured scene video records: {scene, video_path, duration_s}.
    # Populated by run_scene_node alongside rendered_videos so assemble_av
    # can look up the actual video → scene mapping without re-parsing paths.
    rendered_scene_videos: Annotated[list[dict], operator.add]

    # ---- Control flags ----
    # `fatal_error` is for graph-level fatal errors only (parser/summarizer/storyboarder/
    # missing-input failures). It triggers early exit to END. Do NOT use this for
    # per-scene give_up — that goes into `skipped_scenes` via the reviewer flow.
    fatal_error: str | None
    quality: Literal["l", "m", "h"]
    skip_render: bool  # set by CLI --no-render
