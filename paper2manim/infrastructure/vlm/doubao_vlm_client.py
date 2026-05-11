from __future__ import annotations

from pathlib import Path
from typing import Any

from paper2manim.config.settings import VLMConfig
from paper2manim.infrastructure.vlm.client import BaseVLMHTTPClient, VLMClient, VLMChatMessage, image_to_data_url


class DoubaoVLMClient(BaseVLMHTTPClient):
    def __init__(self, config: VLMConfig) -> None:
        super().__init__(config)

    def review_scene(self, prompt: str, image_path: str | Path) -> str:
        return self.review_images(prompt, [Path(image_path)], response_format="json")

    def review_images(
        self,
        prompt: str,
        image_paths: list[Path],
        *,
        response_format: str = "json",
    ) -> str:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for path in image_paths:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_to_data_url(path)},
                }
            )

        messages = [VLMChatMessage(role="user", content=content)]
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": message.role, "content": message.content} for message in messages
            ],
            "temperature": self.config.temperature,
            "stream": False,
        }
        # Doubao VLM does not support response_format=json_object; rely on prompt + JSON parsing.

        data = self._post_json(
            "/chat/completions",
            payload,
            {"Authorization": f"Bearer {self.config.api_key}"},
        )
        try:
            choice = data["choices"][0]
            finish_reason = choice.get("finish_reason")
            if finish_reason not in (None, "stop"):
                raise RuntimeError(
                    f"{self.config.provider} VLM completion did not finish cleanly: "
                    f"{finish_reason}"
                )
            content = choice["message"]["content"]
            if content is None:
                raise RuntimeError(
                    f"{self.config.provider} VLM returned empty content: {data}"
                )
            return str(content)
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"Unexpected {self.config.provider} VLM API response: {data}"
            ) from exc
