"""VLM adapter for OpenAI-style ``/chat/completions`` (e.g. Doubao Ark, OpenAI)."""

from __future__ import annotations

from pathlib import Path

from openai import OpenAI

from paper2manim.config.model_config import ModelConfig
from paper2manim.infrastructure.vlm.client import encode_image_data_url


class OpenAICompatibleVLMClient:
    def __init__(self, cfg: ModelConfig) -> None:
        self._cfg = cfg
        base_url = cfg.base_url
        if cfg.provider == "azure_foundry":
            base_url = cfg.base_url.rstrip("/") + "/openai/v1"
        self._client = OpenAI(api_key=cfg.api_key, base_url=base_url, timeout=cfg.timeout)

    def review_scene(self, prompt: str, image_path: str | Path) -> str:
        url = encode_image_data_url(Path(image_path))
        kwargs: dict = {
            "model": self._cfg.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": url}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }
        if self._cfg.provider == "azure_foundry":
            kwargs["max_completion_tokens"] = self._cfg.max_tokens
        else:
            kwargs["max_tokens"] = self._cfg.max_tokens
        if not self._cfg.omit_temperature:
            kwargs["temperature"] = self._cfg.temperature
        resp = self._client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""
