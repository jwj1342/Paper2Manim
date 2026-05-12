"""Settings loaded from environment / .env via pydantic-settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# env.py lives inside the ``config/`` package, so the repository root is three
# levels up: .../Paper2Manim/paper2manim/config/env.py -> .../Paper2Manim
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    # ----- Unified LLM config (preferred) -----
    # Empty LLM_PROVIDER triggers auto-detect from whichever *_API_KEY is set.
    LLM_PROVIDER: str = Field(default="", description="mimo | deepseek | doubao | openai; empty = auto-detect")
    LLM_API_KEY: str = Field(default="", description="Generic API key; overrides provider-specific keys")
    LLM_BASE_URL: str = Field(default="", description="Override provider default base_url")
    LLM_MODEL_FLASH: str = Field(default="", description="Override provider default flash-tier model id")
    LLM_MODEL_PRO: str = Field(default="", description="Override provider default pro-tier model id")

    # ----- Provider-specific keys (legacy + alternatives) -----
    # MiMo (Xiaomi Token Plan)
    MIMO_API_KEY: str = Field(default="", description="tp- prefixed key from MiMo-API.txt")
    MIMO_BASE_URL: str = "https://token-plan-cn.xiaomimimo.com/v1"
    # DeepSeek
    DEEPSEEK_API_KEY: str = ""
    # Volcengine Ark (Doubao). VLM_API_KEY is also accepted as a fallback for provider=doubao.
    DOUBAO_API_KEY: str = ""
    # OpenAI
    OPENAI_API_KEY: str = ""

    # ----- VLM (MVP 3.0 scene critic; vision-only path) -----
    # These fields predate the unified config; for provider=doubao they're reused
    # as a key/model fallback so users with an existing Volcengine setup don't
    # have to duplicate values.
    VLM_API_KEY: str = ""
    VLM_BASE_URL: str = ""
    VLM_MODEL: str = ""
    VLM_ENABLE: bool = False

    # Project paths and defaults
    PAPER2MANIM_RUNS_DIR: Path = PROJECT_ROOT / "runs"
    PAPER2MANIM_DEFAULT_MODEL: str = "flash"  # flash | pro
    PAPER2MANIM_MAX_RETRIES: int = 3
    PAPER2MANIM_QUALITY: str = "l"  # l | m | h

    # Optional: LangSmith
    LANGCHAIN_TRACING_V2: bool = False
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_PROJECT: str = "paper2manim"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


# Convenience module-level handle
settings = get_settings()
