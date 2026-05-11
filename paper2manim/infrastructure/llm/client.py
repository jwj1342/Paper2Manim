from __future__ import annotations

import http.client
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


class LLMClient(Protocol):
    def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int = 4000,
        response_format: dict[str, str] | None = None,
        reasoning_effort: str = "high",
    ) -> str:
        """Return the assistant text for one chat completion."""
        ...


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    api_key: str
    base_url: str
    model: str
    timeout: int = 120
    max_retries: int = 2
    api_style: str | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)


class BaseHTTPClient:
    retry_statuses = {307, 408, 409, 425, 429, 500, 502, 503, 504}

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.base_url = config.base_url.rstrip("/")

    def _post_json(
        self,
        path: str,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        base_headers = {
            "Content-Type": "application/json",
            "Accept-Encoding": "identity",
            "Connection": "close",
            **headers,
        }
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers=base_headers,
            method="POST",
        )
        use_no_proxy = False
        last_error: str | None = None
        no_proxy_opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({})
        )
        for attempt in range(self.config.max_retries + 1):
            try:
                if use_no_proxy:
                    response_ctx = no_proxy_opener.open(
                        request, timeout=self.config.timeout
                    )
                else:
                    response_ctx = urllib.request.urlopen(
                        request, timeout=self.config.timeout
                    )
                with response_ctx as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                if exc.code in {307, 308}:
                    location = exc.headers.get("Location") or exc.headers.get(
                        "location"
                    )
                    if location:
                        redirected_url = urllib.parse.urljoin(
                            request.full_url, location
                        )
                        if redirected_url != request.full_url:
                            request = urllib.request.Request(
                                redirected_url,
                                data=body,
                                headers=base_headers,
                                method="POST",
                            )
                            continue
                    if not use_no_proxy and urllib.request.getproxies():
                        use_no_proxy = True
                        continue
                if (
                    exc.code in self.retry_statuses
                    and attempt < self.config.max_retries
                ):
                    time.sleep(1.5 * (attempt + 1))
                    continue
                detail = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"{self.config.provider} API HTTP {exc.code}: {detail}"
                ) from exc
            except (http.client.IncompleteRead, TimeoutError, urllib.error.URLError) as exc:
                last_error = str(exc)
                if not use_no_proxy and urllib.request.getproxies():
                    use_no_proxy = True
                    continue
                if attempt >= self.config.max_retries:
                    raise RuntimeError(
                        f"{self.config.provider} API request failed: {exc}"
                    ) from exc
                time.sleep(1.5 * (attempt + 1))

        detail = f": {last_error}" if last_error else "."
        raise RuntimeError(f"{self.config.provider} API request failed{detail}")


class OpenAICompatibleClient(BaseHTTPClient):
    """Small stdlib client for providers that expose OpenAI-style chat completions."""

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int = 4000,
        response_format: dict[str, str] | None = None,
        reasoning_effort: str = "high",
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [message.__dict__ for message in messages],
            "stream": False,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        payload.update(self.config.extra_body)
        if (
            "thinking" in payload
            and payload["thinking"].get("type") == "enabled"
            and "reasoning_effort" not in payload
        ):
            payload["reasoning_effort"] = reasoning_effort
        if response_format is not None:
            payload["response_format"] = response_format

        data = self._post_json(
            "/chat/completions",
            payload,
            {"Authorization": f"Bearer {self.config.api_key}"},
        )
        try:
            choice = data["choices"][0]
            finish_reason = choice.get("finish_reason")
            if finish_reason != "stop":
                raise RuntimeError(
                    f"{self.config.provider} completion did not finish cleanly: "
                    f"{finish_reason}"
                )
            content = choice["message"]["content"]
            if content is None:
                raise RuntimeError(
                    f"{self.config.provider} returned empty content: {data}"
                )
            return str(content)
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"Unexpected {self.config.provider} API response: {data}"
            ) from exc


class AnthropicClient(BaseHTTPClient):
    def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int = 4000,
        response_format: dict[str, str] | None = None,
        reasoning_effort: str = "high",
    ) -> str:
        system, turns = _split_system_messages(messages)
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": _anthropic_role(message.role), "content": message.content}
                for message in turns
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system:
            payload["system"] = system
        payload.update(self.config.extra_body)

        data = self._post_json(
            "/v1/messages",
            payload,
            {
                "x-api-key": self.config.api_key,
                "anthropic-version": "2023-06-01",
            },
        )
        try:
            stop_reason = data.get("stop_reason")
            if stop_reason not in {"end_turn", "stop_sequence"}:
                raise RuntimeError(
                    f"{self.config.provider} completion did not finish cleanly: "
                    f"{stop_reason}"
                )
            parts = data["content"]
            text = "".join(
                str(part.get("text", ""))
                for part in parts
                if part.get("type") == "text"
            )
            if not text:
                raise RuntimeError(
                    f"{self.config.provider} returned empty content: {data}"
                )
            return text
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                f"Unexpected {self.config.provider} API response: {data}"
            ) from exc


class GeminiClient(BaseHTTPClient):
    def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int = 4000,
        response_format: dict[str, str] | None = None,
        reasoning_effort: str = "high",
    ) -> str:
        system, turns = _split_system_messages(messages)
        generation_config: dict[str, Any] = {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        }
        if response_format and response_format.get("type") == "json_object":
            generation_config["responseMimeType"] = "application/json"

        payload: dict[str, Any] = {
            "contents": [
                {
                    "role": _gemini_role(message.role),
                    "parts": [{"text": message.content}],
                }
                for message in turns
            ],
            "generationConfig": generation_config,
        }
        if system:
            payload["system_instruction"] = {"parts": [{"text": system}]}
        payload.update(self.config.extra_body)

        quoted_model = urllib.parse.quote(self.config.model, safe="")
        data = self._post_json(
            f"/models/{quoted_model}:generateContent",
            payload,
            {"x-goog-api-key": self.config.api_key},
        )
        try:
            candidate = data["candidates"][0]
            finish_reason = candidate.get("finishReason")
            if finish_reason and finish_reason != "STOP":
                raise RuntimeError(
                    f"{self.config.provider} completion did not finish cleanly: "
                    f"{finish_reason}"
                )
            parts = candidate["content"]["parts"]
            text = "".join(str(part.get("text", "")) for part in parts)
            if not text:
                raise RuntimeError(
                    f"{self.config.provider} returned empty content: {data}"
                )
            return text
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"Unexpected {self.config.provider} API response: {data}"
            ) from exc


def build_llm_client(config: LLMConfig) -> LLMClient:
    style = (config.api_style or "openai").lower()
    if style == "openai":
        return OpenAICompatibleClient(config)
    if style == "anthropic":
        return AnthropicClient(config)
    if style == "gemini":
        return GeminiClient(config)
    raise ValueError(f"Unsupported API style '{style}'.")


def _split_system_messages(
    messages: list[ChatMessage],
) -> tuple[str, list[ChatMessage]]:
    system_parts = [message.content for message in messages if message.role == "system"]
    turns = [message for message in messages if message.role != "system"]
    return "\n\n".join(system_parts), turns


def _anthropic_role(role: str) -> str:
    return "assistant" if role == "assistant" else "user"


def _gemini_role(role: str) -> str:
    return "model" if role == "assistant" else "user"
