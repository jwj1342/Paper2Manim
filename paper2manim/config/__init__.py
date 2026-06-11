"""Configuration entry point.

Two layers, both production-active:

1. **Environment-based settings** (``env.py``) — pydantic-settings reading
   ``.env`` for API keys (MIMO_*, LANGSMITH_*) and runtime defaults
   (PAPER2MANIM_*). Exposed as the module-level ``settings`` singleton via
   ``from paper2manim.config.env import settings`` (we intentionally don't
   re-export the instance here to keep import paths explicit).

2. **YAML-based model registry** (``model_config.py`` + ``config_loader.py``)
   — multi-provider model definitions and role-to-model mapping loaded from
   ``config.yaml``. Consumed by :mod:`paper2manim.llm` for role-routed LLM /
   VLM clients.
"""

from paper2manim.config.config_loader import load_model_settings
from paper2manim.config.env import (
    PROJECT_ROOT,
    Settings,
    get_settings,
)
from paper2manim.config.model_config import ModelConfig, ModelSettings, TTSConfig

__all__ = [
    "Settings",
    "get_settings",
    "PROJECT_ROOT",
    "ModelConfig",
    "ModelSettings",
    "TTSConfig",
    "load_model_settings",
]
