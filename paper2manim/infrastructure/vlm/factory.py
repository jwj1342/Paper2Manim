"""Pick the right :class:`VLMClient` for a :class:`ModelConfig`."""

from __future__ import annotations

from paper2manim.config.model_config import ModelConfig
from paper2manim.infrastructure.vlm.anthropic_vlm_client import AnthropicVLMClient
from paper2manim.infrastructure.vlm.client import VLMClient
from paper2manim.infrastructure.vlm.openai_vlm_client import OpenAICompatibleVLMClient


def build_vlm_client(cfg: ModelConfig) -> VLMClient:
    if not cfg.supports_vision:
        raise RuntimeError(
            f"Model '{cfg.name}' is not declared supports_vision=true; refusing to build a VLM client."
        )
    if cfg.provider in {"openai_compatible", "azure_foundry"}:
        return OpenAICompatibleVLMClient(cfg)
    if cfg.provider == "anthropic":
        return AnthropicVLMClient(cfg)
    raise RuntimeError(
        f"Unsupported VLM provider '{cfg.provider}' on '{cfg.name}'. "
        "Use 'openai_compatible', 'azure_foundry', or 'anthropic'."
    )
