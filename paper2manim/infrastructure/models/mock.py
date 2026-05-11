from __future__ import annotations

from pathlib import Path
from typing import Any

from paper2manim.infrastructure.models.message import ModelMessage
from paper2manim.infrastructure.models.options import ModelCallOptions
from paper2manim.infrastructure.models.response import ModelResponse


class MockModel:
    def __init__(
        self,
        name: str = "mock",
        *,
        supports_vision: bool = False,
        supports_thinking: bool = False,
        fixed_response: str | None = None,
    ) -> None:
        self.name = name
        self.supports_vision = supports_vision
        self.supports_thinking = supports_thinking
        self._fixed_response = fixed_response
        self._call_count = 0
        self._last_messages: list[ModelMessage] | None = None
        self._last_images: list[str | Path] | None = None

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def last_messages(self) -> list[ModelMessage] | None:
        return self._last_messages

    async def ainvoke(
        self,
        messages: list[ModelMessage],
        *,
        images: list[str | Path] | None = None,
        options: ModelCallOptions | None = None,
    ) -> ModelResponse:
        return self.invoke(messages, images=images, options=options)

    def invoke(
        self,
        messages: list[ModelMessage],
        *,
        images: list[str | Path] | None = None,
        options: ModelCallOptions | None = None,
    ) -> ModelResponse:
        self._call_count += 1
        self._last_messages = messages
        self._last_images = images

        if self._fixed_response is not None:
            text = self._fixed_response
        elif self.supports_vision and images:
            text = self._vision_text()
        else:
            text = self._text()

        return ModelResponse(
            text=text,
            raw={"mock": True},
            usage={"mock_call_count": self._call_count},
            model=self.name,
            provider="mock",
            finish_reason="stop",
            metadata={"call_count": self._call_count},
        )

    def _text(self) -> str:
        return '{"result": "mock text response"}'

    def _vision_text(self) -> str:
        return (
            '{"scene_id": "mock_scene", "decision": "pass", '
            '"scores": {"paper_alignment": 4, "visual_clarity": 4, '
            '"readability": 4, "layout_balance": 4, "visual_focus": 4, '
            '"animation_perceived": 4}, "issues": [], '
            '"paper_alignment_notes": "mock review", '
            '"revision_instruction": "none", "requires_replanning": false}'
        )

    def set_response(self, text: str) -> None:
        self._fixed_response = text

    def __repr__(self) -> str:
        return f"MockModel(name={self.name!r}, vision={self.supports_vision})"
