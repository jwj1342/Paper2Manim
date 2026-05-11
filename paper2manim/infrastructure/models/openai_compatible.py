from __future__ import annotations

from pathlib import Path
from typing import Any

from paper2manim.config.model_config import ModelConfig
from paper2manim.infrastructure.llm.client import (
    LLMConfig,
    OpenAICompatibleClient,
)
from paper2manim.infrastructure.models.message import ModelMessage
from paper2manim.infrastructure.models.options import ModelCallOptions
from paper2manim.infrastructure.models.response import ModelResponse


class OpenAICompatibleModel:
    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self._client = OpenAICompatibleClient(
            LLMConfig(
                provider=config.name,
                api_key=config.api_key,
                base_url=config.base_url,
                model=config.model,
                timeout=config.timeout,
                max_retries=2,
                api_style="openai",
            )
        )

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def supports_vision(self) -> bool:
        return self._config.supports_vision

    @property
    def supports_thinking(self) -> bool:
        return self._config.supports_thinking

    async def ainvoke(
        self,
        messages: list[ModelMessage],
        *,
        images: list[str | Path] | None = None,
        options: ModelCallOptions | None = None,
    ) -> ModelResponse:
        return self.invoke(messages, images=images, options=options)

    def invoke(
        self,
        messages: list[ModelMessage],
        *,
        images: list[str | Path] | None = None,
        options: ModelCallOptions | None = None,
    ) -> ModelResponse:
        temperature = _resolve(options, "temperature", self._config.temperature)
        max_tokens = _resolve(options, "max_tokens", self._config.max_tokens)
        response_format = _resolve(options, "response_format", None)

        payload: dict[str, Any] = {
            "model": self._config.model,
            "messages": self._build_payload(messages, images),
            "stream": False,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            payload["response_format"] = {"type": response_format}

        data = self._client._post_json(
            "/chat/completions",
            payload,
            {"Authorization": f"Bearer {self._config.api_key}"},
        )

        try:
            choice = data["choices"][0]
            finish_reason = choice.get("finish_reason")
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"Unexpected {self.name} API response: {data}"
            ) from exc

        return ModelResponse(
            text=str(content) if content else "",
            raw=data,
            usage=_extract_usage(data),
            model=data.get("model"),
            provider="openai_compatible",
            finish_reason=finish_reason,
        )

    def _build_payload(
        self,
        messages: list[ModelMessage],
        images: list[str | Path] | None,
    ) -> list[dict[str, Any]]:
        if not images:
            return [{"role": msg.role, "content": msg.content} for msg in messages]

        if len(messages) > 1:
            return [
                {"role": msg.role, "content": msg.content}
                for msg in messages[:-1]
            ] + [_vision_payload(messages[-1], images)]

        last = messages[-1]
        return [_vision_payload(last, images)]

    def __repr__(self) -> str:
        return (
            f"OpenAICompatibleModel(name={self.name!r}, model={self._config.model!r}, "
            f"vision={self.supports_vision})"
        )


def _vision_payload(
    message: ModelMessage, images: list[str | Path]
) -> dict[str, Any]:
    parts: list[dict[str, Any]] = [
        {"type": "text", "text": message.content}
    ]
    for image_path in images:
        encoded = _encode_image(Path(image_path))
        parts.append({
            "type": "image_url",
            "image_url": {"url": encoded, "detail": "auto"},
        })
    return {"role": message.role, "content": parts}


def _encode_image(path: Path) -> str:
    import base64

    raw = path.read_bytes()
    encoded = base64.b64encode(raw).decode("ascii")
    suffix = path.suffix.lower().lstrip(".")
    media_type = "jpeg" if suffix in {"jpg", "jpeg"} else suffix or "jpeg"
    return f"data:image/{media_type};base64,{encoded}"


def _resolve(
    options: ModelCallOptions | None,
    key: str,
    default: Any,
) -> Any:
    if options is None:
        return default
    value = getattr(options, key, None)
    return default if value is None else value


def _extract_usage(data: dict[str, Any]) -> dict[str, Any] | None:
    usage = data.get("usage")
    if isinstance(usage, dict):
        return dict(usage)
    return None
