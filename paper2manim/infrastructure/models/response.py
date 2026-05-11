from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelResponse:
    text: str
    raw: Any | None = None
    usage: dict[str, Any] | None = None
    model: str | None = None
    provider: str | None = None
    finish_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
