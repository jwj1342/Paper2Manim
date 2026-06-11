"""Mock TTS client for unit tests — generates deterministic WAV files.

Returns a minimal valid WAV (silence) that ffprobe can parse, so graph tests
can exercise the full voiceover pipeline without a real TTS call.
"""

from __future__ import annotations

import struct
import wave
from pathlib import Path


class MockTTSClient:
    """Generates a deterministic WAV file.

    ``default_duration_s`` controls the length of the silence WAV.
    ``calls`` records every synthesis call for test assertions.
    """

    def __init__(self, *, default_duration_s: float = 2.0) -> None:
        self.default_duration_s = float(default_duration_s)
        self.calls: list[dict] = []

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
        if fmt != "wav":
            raise ValueError(
                f"MockTTSClient only supports 'wav' format, got {fmt!r}"
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        duration = self.default_duration_s / max(speed, 0.25)
        self._write_silence_wav(output_path, duration)
        self.calls.append(
            {
                "text": text,
                "output_path": str(output_path),
                "voice": voice,
                "speed": speed,
                "duration_s": duration,
            }
        )
        return output_path

    @staticmethod
    def _write_silence_wav(path: Path, duration_s: float) -> None:
        """Write a minimal 16-bit mono PCM WAV with ``duration_s`` seconds of silence."""
        sample_rate = 24000
        num_channels = 1
        sample_width = 2  # 16-bit
        num_samples = int(sample_rate * duration_s)

        with wave.open(str(path), "w") as wf:
            wf.setnchannels(num_channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            # Write silence (zero samples)
            wf.writeframes(struct.pack(f"<{num_samples}h", *([0] * num_samples)))
