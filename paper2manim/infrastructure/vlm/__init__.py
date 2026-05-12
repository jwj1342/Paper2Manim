"""Vision-language client adapters that take ``(prompt, image_path) -> JSON text``.

Two provider styles are supported, selected by :attr:`ModelConfig.provider`:

* ``openai_compatible`` — OpenAI ``/chat/completions`` with ``image_url`` content
  block (Doubao Ark, OpenAI, MiMo if it ever gains vision).
* ``anthropic`` — Anthropic Messages API with ``image`` content block; sends
  ``Authorization: Bearer`` when ``ModelConfig.auth_style == "bearer"`` (Azure).

Both implementations encode the local image as a base64 data URL / payload, so
the caller never has to think about uploads.
"""

from paper2manim.infrastructure.vlm.client import VLMClient
from paper2manim.infrastructure.vlm.factory import build_vlm_client
from paper2manim.infrastructure.vlm.mock_vlm_client import MockVLMClient

__all__ = ["VLMClient", "build_vlm_client", "MockVLMClient"]
