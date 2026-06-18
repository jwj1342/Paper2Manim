"""LangGraph State schema — single source of truth for all nodes.

合一设计：MVP 1.0 仅用部分字段；MVP 2.0 全用。`total=False` 让所有字段可选，
`Annotated[..., operator.add]` 标记需要累积的字段（reflection 历史）。
State 必须可 JSON 序列化（LangGraph checkpoint / LangSmith trace 才能正常工作）。
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

# ---- Substructures (TypedDict mirror of pydantic schemas) ----


class Scene(TypedDict):
    name: str
    description: str
    duration_hint: float


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


# ---- Top-level State ----


class PaperState(TypedDict, total=False):
    # ---- Inputs ----
    run_id: str
    input_kind: Literal["text", "pdf", "arxiv"]
    raw_text: str | None  # MVP 1.0
    pdf_path: str | None  # MVP 2.0 local PDF
    arxiv_spec: str | None  # MVP 2.0 arxiv id / url / "arXiv:1706.03762"
    arxiv_section: str | None  # optional: slice a single \section{...} from source

    # ---- MVP 2.0 parsing/summarization ----
    # `parsed_markdown` holds the parser's flattened text regardless of source format.
    # Use `parsed_format` to disambiguate: "markdown" (Marker) vs "latex" (arXiv source).
    parsed_markdown: str | None
    parsed_format: Literal["markdown", "latex"] | None
    parser_source: str | None  # origin tag, e.g. "arxiv-src:1706.03762"
    summary: dict | None  # SummaryModel.model_dump()

    # ---- Storyboard ----
    storyboard: Storyboard | None

    # ---- Per-scene mutables (LEGACY: mvp1 only) ----
    # mvp2's per-scene loop now runs in a sub-graph with its own state shape
    # (``paper2manim.graphs.scene_graph.SceneState``). These fields remain on
    # PaperState so the simpler mvp1 graph (no fan-out) keeps working without
    # a state-shape refactor. mvp2 nodes never read or write them; LangGraph
    # would otherwise reject the parallel updates from each Send branch.
    current_scene_idx: int
    current_code: str | None
    iter_count: int
    error_feedback: dict | None

    # ---- Reflection caps ----
    attempts: Annotated[list[Attempt], operator.add]  # reducer: append across scenes
    max_retries: int

    # ---- Output ----
    rendered_videos: Annotated[list[str], operator.add]  # mp4 paths in scene order
    skipped_scenes: Annotated[list[str], operator.add]  # scenes that gave up after retries
    final_video_path: str | None

    # ---- VLM caps + reducer outputs ----
    vlm_enabled: bool
    # Per-scene cap; enforced inside the scene subgraph (mvp2 parallel path)
    # or inline by the legacy mvp1 nodes.
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
    # One structured summary per scene (mvp2 parallel path only) — flushed
    # back from each Send branch by ``scene_graph`` and reduced via
    # ``operator.add``. Includes the final ``last_visual_review`` dict so
    # post-run inspectors don't lose per-scene VLM detail (previously
    # available as a top-level field, now scoped to SceneState).
    scene_reports: Annotated[list[dict], operator.add]

    # ---- Episodic Memory Bank (MVP 3.0 §4.1 + §4.4) ----
    # Enabled by `--emb`. `emb_store_path` points at the directory holding
    # `memory.db` + `{success,failure}.index`. The retrieve / consolidate
    # nodes look this up on each invocation and cache the loaded EMB by path.
    # `emb_instance` is a test-only escape hatch — when present, nodes use it
    # directly and skip the path-based cache.
    emb_enabled: bool
    emb_store_path: str | None
    # Both thresholds live on the proposal §4.2 0-100 schema. CLI defaults are
    # 85.0 / 5.0 (see paper2manim.cli mvp2). DO NOT default these to old 1-5
    # values when synthesizing test states — the production gate would never fire.
    emb_theta_high: float  # success-record acceptance threshold (0-100 avg score)
    emb_failure_min_margin: float  # min (after-before) gap for a failure record
    emb_use_llm_distillers: bool  # if True, rationale_writer + lesson_distiller call the LLM
    emb_use_faiss: bool
    emb_use_real_embedder: bool
    emb_instance: object | None  # test injection; opaque so TypedDict typecheck stays cheap
    retrieved_success: list[dict]  # wire-format records (no embedding payload)
    retrieved_failure: list[dict]
    emb_writes: Annotated[list[dict], operator.add]  # consolidation reports, one per run/scene
    # B6: cross-domain freeze + channel ablations.
    # ``emb_readonly`` (--emb-readonly): emb_consolidate_node returns immediately,
    # so the EMB grows in train phase only. Required by RQ3 cross-domain test.
    # ``dataset_domain`` (--dataset-domain cs|math|...): tags every record this
    # run writes, so retrieval can later filter by ``Context.domain``.
    # ``emb_no_success_channel`` / ``emb_no_failure_channel``: §8.3 Ablation E
    # — disable one polarity for both retrieval (scene_graph.emb_retrieve_node)
    # and consolidation (mvp2.emb_consolidate_node).
    emb_readonly: bool
    dataset_domain: str | None
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
