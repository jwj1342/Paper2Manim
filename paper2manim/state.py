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

    # ---- Per-scene loop state ----
    current_scene_idx: int
    current_code: str | None

    # ---- Reflection ----
    attempts: Annotated[list[Attempt], operator.add]  # reducer: append
    error_feedback: dict | None  # ErrorFeedback.model_dump() or None
    max_retries: int
    iter_count: int

    # ---- Output ----
    rendered_videos: Annotated[list[str], operator.add]  # mp4 paths in scene order
    skipped_scenes: Annotated[list[str], operator.add]  # scene names that gave up after retries
    final_video_path: str | None

    # ---- VLM Multi-Dim Scoring loop (MVP 3.0) ----
    # `vlm_enabled` is a per-run toggle (set by CLI / build_mvp2_graph) that decides
    # whether to insert frame_sampler + vlm_reviewer between render-success and advance.
    # Counts and decisions are per-scene scoped (reset by init_scene_node when
    # starting a new scene).
    vlm_enabled: bool
    vlm_revision_count: int  # how many visual revisions have run on the current scene
    max_visual_revisions: int  # cap; advance once exceeded
    # Each entry is ``{"scene": scene_name, "decision": "pass|revise|fail"}``.
    # The reducer keeps appending across the whole run; slicing by ``scene``
    # gives you per-scene history without needing a per-scene reset.
    visual_revision_decisions: Annotated[list[dict[str, str]], operator.add]
    # Scenes the VLM couldn't actually review — missing montage, VLM API
    # exception, or visual_revise_node hitting an exception. Distinct from:
    #   - ``skipped_scenes`` (text-reflection give_up after max_retries)
    #   - a recorded ``decision == "fail"`` in ``visual_revision_decisions``
    #     (the VLM did review and explicitly judged the scene unusable)
    # The reducer accumulates across scenes; without it, the dict-merge
    # default would silently overwrite earlier skips. Reported by Copilot
    # review on PR #15.
    vlm_skipped_scenes: Annotated[list[str], operator.add]
    last_visual_review: dict | None  # the full review payload from vlm_scene_reviewer
    current_montage_path: str | None  # latest frame montage produced for this scene

    # ---- Episodic Memory Bank (MVP 3.0 §4.1 + §4.4) ----
    # Enabled by `--emb`. `emb_store_path` points at the directory holding
    # `memory.db` + `{success,failure}.index`. The retrieve / consolidate
    # nodes look this up on each invocation and cache the loaded EMB by path.
    # `emb_instance` is a test-only escape hatch — when present, nodes use it
    # directly and skip the path-based cache.
    emb_enabled: bool
    emb_store_path: str | None
    emb_theta_high: float  # success-record acceptance threshold (avg score)
    emb_failure_min_margin: float  # min (after-before) score gap for a failure record
    emb_use_llm_distillers: bool  # if True, rationale_writer + lesson_distiller call the LLM
    emb_use_faiss: bool
    emb_use_real_embedder: bool
    emb_instance: object | None  # test injection; opaque so TypedDict typecheck stays cheap
    retrieved_success: list[dict]  # wire-format records (no embedding payload)
    retrieved_failure: list[dict]
    emb_writes: Annotated[list[dict], operator.add]  # consolidation reports, one per run/scene

    # ---- Control flags ----
    # `fatal_error` is for graph-level fatal errors only (parser/summarizer/storyboarder/
    # missing-input failures). It triggers early exit to END. Do NOT use this for
    # per-scene give_up — that goes into `skipped_scenes` via the reviewer flow.
    fatal_error: str | None
    quality: Literal["l", "m", "h"]
    skip_render: bool  # set by CLI --no-render
