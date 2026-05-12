"""VLM client facade — graph-side entry point for the VLM infrastructure.

Mirrors ``paper2manim/llm.py`` so graph-side agents (``paper2manim/agents/*``) can
get a VLM client without importing from ``paper2manim/infrastructure``. Returns
``None`` when VLM is not enabled in settings, so callers can short-circuit
gracefully rather than treating it as fatal.
"""

from __future__ import annotations

import logging

from paper2manim.config import load_app_settings
from paper2manim.infrastructure.vlm.client import VLMClient
from paper2manim.infrastructure.vlm.factory import build_vlm_client

log = logging.getLogger(__name__)


def get_vlm(*, mock: bool = False) -> VLMClient | None:
    """Build a VLMClient from the current app settings.

    Returns
    -------
    VLMClient | None
        ``None`` when VLM is unconfigured (``vlm.enable=false``, missing API key,
        etc.). Callers should treat ``None`` as "VLM disabled — skip this step",
        not as a fatal error.
    """
    try:
        settings = load_app_settings()
    except Exception as exc:  # noqa: BLE001 — config loading is best-effort
        log.warning("[vlm] could not load app settings: %s", exc)
        return None
    client = build_vlm_client(settings, mock=mock)
    if client is None:
        log.info("[vlm] disabled (no vlm section in settings)")
    return client
