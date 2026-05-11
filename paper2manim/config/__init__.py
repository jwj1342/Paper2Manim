"""Unified configuration entry point.

This package keeps two configuration layers side-by-side:

1. **Environment-based settings** (``env.py``) — pydantic-settings reading
   ``.env`` for API keys and runtime defaults. Used by the original
   ``main``-branch agents/graphs via ``from paper2manim.config.env import settings``.

2. **YAML-based model & application settings** (``model_config.py``,
   ``config_loader.py``, ``settings.py``) — adopted from the collaborator's
   ``fix/api-client-config`` branch (D3 of issue #1) for multi-provider
   model registry and richer application config.

The collaborator's ``Settings`` (app-level dataclass) is re-exported as
``AppSettings`` to avoid colliding with the env-based pydantic ``Settings``.
"""

# Env-based settings (main branch's pydantic-settings API).
# NOTE: we intentionally do NOT re-export the ``settings`` instance here, because
# the collaborator's ``settings.py`` submodule (re-exported below) would shadow
# it. Callers wanting the env singleton should do
# ``from paper2manim.config.env import settings`` or call ``get_settings()``.
from paper2manim.config.env import (
    PROJECT_ROOT,
    Settings,
    get_settings,
)

# New: YAML-based model registry + role mapping
from paper2manim.config.model_config import ModelConfig, ModelSettings
from paper2manim.config.config_loader import load_model_settings

# New: YAML-based application settings (collaborator's)
from paper2manim.config.settings import (
    ProviderDefaults,
    Settings as AppSettings,
    VLMConfig,
    VisualReviewConfig,
    load_dotenv,
    load_settings as load_app_settings,
)

__all__ = [
    # env-based (main)
    "Settings",
    "get_settings",
    "PROJECT_ROOT",
    # YAML-based model registry
    "ModelConfig",
    "ModelSettings",
    "load_model_settings",
    # YAML-based application settings (collaborator)
    "AppSettings",
    "ProviderDefaults",
    "VLMConfig",
    "VisualReviewConfig",
    "load_app_settings",
    "load_dotenv",
]
