"""Pick the right :class:`TTSClient` for a :class:`TTSConfig`."""

from __future__ import annotations

from paper2manim.config.model_config import TTSConfig
from paper2manim.infrastructure.tts.client import TTSClient
from paper2manim.infrastructure.tts.edge_tts_client import EdgeTTSClient
from paper2manim.infrastructure.tts.mock_tts_client import MockTTSClient
from paper2manim.infrastructure.tts.openai_tts_client import OpenAITTSSClient


def build_tts_client(cfg: TTSConfig) -> TTSClient:
    """Return a TTS client for ``cfg``.

    Raises ``RuntimeError`` for unsupported providers.
    """
    if cfg.provider == "openai":
        return OpenAITTSSClient(cfg)
    if cfg.provider == "edge":
        return EdgeTTSClient(cfg)
    if cfg.provider == "mock":
        return MockTTSClient()
    raise RuntimeError(
        f"Unsupported TTS provider '{cfg.provider}'. "
        "Use 'openai', 'edge', or 'mock'."
    )
