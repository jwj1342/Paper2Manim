"""VLM adapter for the Anthropic Messages API (including Azure-hosted Claude).

When ``ModelConfig.auth_style == "bearer"`` the Bearer header is injected via
``Anthropic(auth_token=...)``, which is what Azure's anthropic route requires;
otherwise the standard ``x-api-key`` header is sent.
"""

from __future__ import annotations

from pathlib import Path

from anthropic import Anthropic

from paper2manim.config.model_config import ModelConfig
from paper2manim.infrastructure.vlm.client import encode_image_base64


class AnthropicVLMClient:
    def __init__(self, cfg: ModelConfig) -> None:
        self._cfg = cfg
        if cfg.auth_style == "bearer":
            self._client = Anthropic(auth_token=cfg.api_key, base_url=cfg.base_url, timeout=cfg.timeout)
        else:
            self._client = Anthropic(api_key=cfg.api_key, base_url=cfg.base_url, timeout=cfg.timeout)

    def review_scene(self, prompt: str, image_path: str | Path) -> str:
        data, media = encode_image_base64(Path(image_path))
        kwargs: dict = {
            "model": self._cfg.model,
            "max_tokens": self._cfg.max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media,
                                "data": data,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }
        if not self._cfg.omit_temperature:
            kwargs["temperature"] = self._cfg.temperature
        resp = self._client.messages.create(**kwargs)
        # Concatenate any text blocks the model emits.
        return "".join(blk.text for blk in resp.content if getattr(blk, "type", None) == "text")
