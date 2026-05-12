"""VLM adapter for OpenAI-style ``/chat/completions`` (e.g. Doubao Ark, OpenAI)."""

from __future__ import annotations

from pathlib import Path

from openai import OpenAI

from paper2manim.config.model_config import ModelConfig
from paper2manim.infrastructure.vlm.client import encode_image_data_url


class OpenAICompatibleVLMClient:
    def __init__(self, cfg: ModelConfig) -> None:
        self._cfg = cfg
        self._client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url, timeout=cfg.timeout)

    def review_scene(self, prompt: str, image_path: str | Path) -> str:
        url = encode_image_data_url(Path(image_path))
        resp = self._client.chat.completions.create(
            model=self._cfg.model,
            max_tokens=self._cfg.max_tokens,
            temperature=self._cfg.temperature,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": url}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        return resp.choices[0].message.content or ""
