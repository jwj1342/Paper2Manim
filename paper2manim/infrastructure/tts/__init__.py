"""Text-to-speech client adapters that take ``(text, output_path) -> Path``.

Three provider styles are supported:

* ``openai`` — OpenAI ``/audio/speech`` endpoint (``gpt-4o-mini-tts``, ``tts-1``).
* ``edge`` — Microsoft Edge TTS (free, no API key, 100+ voices).
* ``mock`` — deterministic silence WAV for tests and CI.

All return a standard audio file path; the caller never has to think about
provider-specific transports.
"""

from paper2manim.infrastructure.tts.client import TTSClient
from paper2manim.infrastructure.tts.edge_tts_client import EdgeTTSClient
from paper2manim.infrastructure.tts.factory import build_tts_client
from paper2manim.infrastructure.tts.mock_tts_client import MockTTSClient

__all__ = ["TTSClient", "build_tts_client", "EdgeTTSClient", "MockTTSClient"]
