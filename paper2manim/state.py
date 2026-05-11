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
    input_kind: Literal["text", "pdf"]
    raw_text: str | None  # MVP 1.0
    pdf_path: str | None  # MVP 2.0

    # ---- MVP 2.0 parsing/summarization ----
    parsed_markdown: str | None
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

    # ---- Control flags ----
    # `fatal_error` is for graph-level fatal errors only (parser/summarizer/storyboarder/
    # missing-input failures). It triggers early exit to END. Do NOT use this for
    # per-scene give_up — that goes into `skipped_scenes` via the reviewer flow.
    fatal_error: str | None
    quality: Literal["l", "m", "h"]
    skip_render: bool  # set by CLI --no-render
