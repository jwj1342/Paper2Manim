"""Settings loaded from environment / .env via pydantic-settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    # MiMo (Xiaomi Token Plan)
    MIMO_API_KEY: str = Field(default="", description="tp- prefixed key from MiMo-API.txt")
    MIMO_BASE_URL: str = "https://token-plan-cn.xiaomimimo.com/v1"

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
