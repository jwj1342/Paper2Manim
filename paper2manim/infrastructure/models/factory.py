from __future__ import annotations

from paper2manim.config.model_config import ModelConfig
from paper2manim.infrastructure.models.base import BaseModel
from paper2manim.infrastructure.models.mock import MockModel
from paper2manim.infrastructure.models.openai_compatible import OpenAICompatibleModel


def create_model(config: ModelConfig) -> BaseModel:
    provider = config.provider.lower()

    if provider == "openai_compatible":
        return OpenAICompatibleModel(config)

    if provider == "mock":
        return MockModel(
            name=config.name,
            supports_vision=config.supports_vision,
            supports_thinking=config.supports_thinking,
        )

    raise ValueError(
        f"Unsupported model provider '{config.provider}' "
        f"for model '{config.name}'. "
        f"Supported providers: openai_compatible, mock."
    )
