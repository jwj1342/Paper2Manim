"""VLM client Protocol shared by all provider-specific implementations."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Protocol


class VLMClient(Protocol):
    """Submit one ``(prompt, image)`` review and return raw JSON-bearing text.

    Implementations are free to retry / wrap the underlying transport, but must
    not modify the prompt or post-process the response. JSON extraction is the
    caller's job (see :func:`paper2manim.utils.text_utils.extract_json_object`).
    """

    def review_scene(self, prompt: str, image_path: str | Path) -> str: ...


def encode_image_base64(path: Path) -> tuple[str, str]:
    """Return ``(base64_data, media_type)`` for a PNG/JPEG file."""
    raw = path.read_bytes()
    media_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return base64.b64encode(raw).decode("ascii"), media_type


def encode_image_data_url(path: Path) -> str:
    """Return a ``data:image/<ext>;base64,<...>`` URL for OpenAI-style ``image_url``."""
    encoded, media = encode_image_base64(path)
    return f"data:{media};base64,{encoded}"
