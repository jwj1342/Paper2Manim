"""TTS client Protocol shared by all provider-specific implementations.

Analogous to :class:`paper2manim.infrastructure.vlm.client.VLMClient` but for
audio synthesis instead of vision-language scoring.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class TTSClient(Protocol):
    """Synthesize one text string as an audio file.

    Implementations are free to retry / wrap the underlying transport, but must
    write a standard audio file to ``output_path``. The caller chooses the
    format (``audio_format``); implementations should validate that the provider
    actually supports it.
    """

    def synthesize(
        self,
        text: str,
        output_path: Path,
        *,
        voice: str,
        speed: float = 1.0,
        audio_format: str = "wav",
    ) -> Path:
        """Write synthesized speech to ``output_path`` and return the path."""
        ...
