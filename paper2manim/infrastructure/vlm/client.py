from __future__ import annotations

import base64
import http.client
import json
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from paper2manim.config.settings import VLMConfig


class VLMClient(Protocol):
    def review_scene(self, prompt: str, image_path: str | Path) -> str:
        ...

    def review_images(
        self,
        prompt: str,
        image_paths: list[Path],
        *,
        response_format: str = "json",
    ) -> str:
        ...


@dataclass(frozen=True)
class VLMChatMessage:
    role: str
    content: list[dict[str, Any]]


class BaseVLMHTTPClient:
    retry_statuses = {307, 408, 409, 425, 429, 500, 502, 503, 504}

    def __init__(self, config: VLMConfig) -> None:
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
        no_proxy_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        for attempt in range(self.config.max_retries + 1):
            try:
                if use_no_proxy:
                    response_ctx = no_proxy_opener.open(request, timeout=self.config.timeout)
                else:
                    response_ctx = urllib.request.urlopen(request, timeout=self.config.timeout)
                with response_ctx as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                if exc.code in {307, 308}:
                    location = exc.headers.get("Location") or exc.headers.get("location")
                    if location:
                        redirected_url = urllib.parse.urljoin(request.full_url, location)
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
                if exc.code in self.retry_statuses and attempt < self.config.max_retries:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                detail = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"{self.config.provider} VLM HTTP {exc.code}: {detail}"
                ) from exc
            except (http.client.IncompleteRead, TimeoutError, socket.timeout, urllib.error.URLError) as exc:
                last_error = str(exc)
                if not use_no_proxy and urllib.request.getproxies():
                    use_no_proxy = True
                    continue
                if attempt >= self.config.max_retries:
                    raise RuntimeError(
                        f"{self.config.provider} VLM request failed: {exc}"
                    ) from exc
                time.sleep(1.5 * (attempt + 1))

        detail = f": {last_error}" if last_error else "."
        raise RuntimeError(f"{self.config.provider} VLM request failed{detail}")


def image_to_data_url(path: Path) -> str:
    raw = path.read_bytes()
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"
