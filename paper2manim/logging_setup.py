"""Logging setup with optional LangSmith tracing hook."""

from __future__ import annotations

import logging
import os

from paper2manim.config.env import settings


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Quiet down very noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    if settings.LANGCHAIN_TRACING_V2 and settings.LANGSMITH_API_KEY:
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGCHAIN_API_KEY", settings.LANGSMITH_API_KEY)
        os.environ.setdefault("LANGCHAIN_PROJECT", settings.LANGSMITH_PROJECT)
        logging.getLogger(__name__).info(
            "LangSmith tracing enabled (project=%s)", settings.LANGSMITH_PROJECT
        )
