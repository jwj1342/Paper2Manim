from __future__ import annotations

from paper2manim.config.model_config import ModelSettings
from paper2manim.infrastructure.models.base import BaseModel
from paper2manim.infrastructure.models.factory import create_model


class ModelRegistry:
    def __init__(self, settings: ModelSettings) -> None:
        self._settings = settings
        self._instances: dict[str, BaseModel] = {}

    def get(self, role: str) -> BaseModel:
        model_config = self._settings.model_for_role(role)

        if role == "vision_checker" and not model_config.supports_vision:
            raise ValueError(
                f"Role 'vision_checker' requires a vision-capable model. "
                f"Model '{model_config.name}' has supports_vision=false."
            )

        name = model_config.name
        if name not in self._instances:
            self._instances[name] = create_model(model_config)

        return self._instances[name]

    @property
    def settings(self) -> ModelSettings:
        return self._settings

    def __repr__(self) -> str:
        return (
            f"ModelRegistry(roles={sorted(self._settings.roles)}, "
            f"cached={sorted(self._instances)})"
        )
