from __future__ import annotations

from pathlib import Path
from typing import Protocol

from paper2manim.infrastructure.models.message import ModelMessage
from paper2manim.infrastructure.models.options import ModelCallOptions
from paper2manim.infrastructure.models.response import ModelResponse


class BaseModel(Protocol):
    name: str
    supports_vision: bool
    supports_thinking: bool

    async def ainvoke(
        self,
        messages: list[ModelMessage],
        *,
        images: list[str | Path] | None = None,
        options: ModelCallOptions | None = None,
    ) -> ModelResponse:
        ...

    def invoke(
        self,
        messages: list[ModelMessage],
        *,
        images: list[str | Path] | None = None,
        options: ModelCallOptions | None = None,
    ) -> ModelResponse:
        ...
