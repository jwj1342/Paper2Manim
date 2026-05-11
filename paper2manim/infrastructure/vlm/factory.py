from __future__ import annotations

from paper2manim.config.settings import Settings
from paper2manim.infrastructure.vlm.client import VLMClient
from paper2manim.infrastructure.vlm.doubao_vlm_client import DoubaoVLMClient
from paper2manim.infrastructure.vlm.mock_vlm_client import MockVLMClient


def build_vlm_client(settings: Settings | None, *, mock: bool = False) -> VLMClient | None:
    if settings is None or settings.vlm is None:
        return None
    if mock or settings.visual_review.use_mock_vlm:
        return MockVLMClient()
    provider = settings.vlm.provider.lower()
    if provider in {"volcengine_ark", "ark", "doubao"}:
        return DoubaoVLMClient(settings.vlm)
    raise RuntimeError(f"Unsupported VLM provider: {settings.vlm.provider}")
