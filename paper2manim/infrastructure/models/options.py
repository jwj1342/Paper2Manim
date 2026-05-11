from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelCallOptions:
    temperature: float | None = None
    max_tokens: int | None = None
    response_format: str | None = None
    timeout: int | None = None
    extra: dict[str, Any] | None = None
