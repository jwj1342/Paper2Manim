"""TTS adapter for OpenAI ``/audio/speech`` endpoint.

Uses the OpenAI Python SDK (``openai.audio.speech.create``) which supports
``gpt-4o-mini-tts`` and ``tts-1`` / ``tts-1-hd`` models.
"""

from __future__ import annotations

import logging
from pathlib import Path

from openai import OpenAI

from paper2manim.config.model_config import TTSConfig

log = logging.getLogger(__name__)

# OpenAI TTS models that support the ``instructions`` parameter for tone / style.
_INSTRUCTION_MODELS = {"gpt-4o-mini-tts"}


class OpenAITTSSClient:
    """Synthesize speech via OpenAI's TTS API.

    ``gpt-4o-mini-tts`` is the recommended model — it supports ``instructions``
    for tone control. Legacy ``tts-1`` / ``tts-1-hd`` work without instructions.
    """

    def __init__(self, cfg: TTSConfig) -> None:
        self._cfg = cfg
        self._client = OpenAI(
            api_key=cfg.api_key, base_url=cfg.base_url or None, timeout=cfg.timeout
        )

    def synthesize(
        self,
        text: str,
        output_path: Path,
        *,
        voice: str,
        speed: float = 1.0,
        audio_format: str = "wav",
    ) -> Path:
        fmt = audio_format.lower()
        # OpenAI TTS returns: mp3, opus, aac, flac, wav, pcm.
        # Map "wav" → "wav" (supported), "mp3" → "mp3", etc.
        if fmt not in {"mp3", "opus", "aac", "flac", "wav", "pcm"}:
            log.warning(
                "[tts.openai] unsupported format %r; falling back to wav", fmt
            )
            fmt = "wav"

        kwargs: dict = {
            "model": self._cfg.model,
            "voice": voice,
            "input": text,
            "speed": speed,
            "response_format": fmt,
        }
        # gpt-4o-mini-tts supports instructions for tone guidance.
        if self._cfg.model in _INSTRUCTION_MODELS:
            kwargs["instructions"] = (
                "Speak in a clear, educational tone suitable for an academic "
                "video narration. Pace yourself naturally — not too fast, not "
                "too slow. Pronounce technical terms carefully."
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        log.info(
            "[tts.openai] model=%s voice=%s speed=%.2f chars=%d",
            self._cfg.model,
            voice,
            speed,
            len(text),
        )
        with self._client.audio.speech.with_streaming_response.create(
            **kwargs,
        ) as response:
            response.stream_to_file(str(output_path))

        return output_path
