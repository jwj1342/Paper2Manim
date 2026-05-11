from typing import Literal

from pydantic import BaseModel


class SourceLine(BaseModel):
    line: int
    code: str


class RenderResultModel(BaseModel):
    status: Literal["success", "error"]
    category: Literal["python", "latex", "manim_runtime", "timeout", "unknown"] | None = None
    exit_code: int
    scene: str
    video_path: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    traceback_tail: str | None = None
    source_excerpt: list[SourceLine] | None = None
    tex_log_excerpt: str | None = None
    workdir: str


class ErrorFeedback(BaseModel):
    """Reviewer 给 Coder 的结构化反馈"""

    decision: Literal["retry", "give_up"]
    hint: str
    render_result: RenderResultModel
