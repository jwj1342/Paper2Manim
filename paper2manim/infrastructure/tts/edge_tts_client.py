"""TTS adapter for Microsoft Edge TTS (free, no API key required).

Uses the ``edge-tts`` library which communicates with Microsoft's free
TTS service. Supports 100+ voices across many languages.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from paper2manim.config.model_config import TTSConfig

log = logging.getLogger(__name__)


class EdgeTTSClient:
    """Synthesize speech via Microsoft Edge TTS (free).

    ``voice`` should be a ShortName like ``en-US-AriaNeural``.
    ``audio_format`` should be ``wav`` (Edge TTS outputs mp3 by default
    but we convert via ffmpeg in the caller — here we save as mp3 and
    the alignment step handles it transparently).
    """

    def __init__(self, cfg: TTSConfig) -> None:
        self._cfg = cfg

    def synthesize(
        self,
        text: str,
        output_path: Path,
        *,
        voice: str,
        speed: float = 1.0,
        audio_format: str = "wav",
    ) -> Path:
        import edge_tts

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # edge-tts rate: "+0%" = normal, "+50%" = faster, "-50%" = slower
        rate_pct = int((speed - 1.0) * 100)
        rate_str = f"{rate_pct:+d}%" if rate_pct != 0 else "+0%"

        voice_name = voice or "en-US-AriaNeural"

        # edge-tts saves as mp3; convert to wav if requested
        tmp_path = output_path.with_suffix(".mp3")

        log.info(
            "[tts.edge] voice=%s rate=%s chars=%d",
            voice_name, rate_str, len(text),
        )

        async def _synthesize():
            communicate = edge_tts.Communicate(
                text,
                voice_name,
                rate=rate_str,
            )
            await communicate.save(str(tmp_path))

        try:
            asyncio.run(_synthesize())
        except RuntimeError:
            # Handle nested event loop (graph runs in thread pool)
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(_synthesize())
            finally:
                loop.close()

        if audio_format.lower() == "wav" and tmp_path.suffix == ".mp3":
            # Convert mp3 to wav via ffmpeg
            import subprocess

            cmd = [
                "ffmpeg", "-y",
                "-i", str(tmp_path),
                "-acodec", "pcm_s16le",
                "-ar", "24000",
                "-ac", "1",
                str(output_path),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if proc.returncode != 0:
                log.warning(
                    "[tts.edge] ffmpeg mp3→wav conversion failed: %s",
                    proc.stderr[-200:],
                )
                # Fall back to mp3
                tmp_path.rename(output_path)
                return output_path
            tmp_path.unlink()
        else:
            tmp_path.rename(output_path)

        return output_path
