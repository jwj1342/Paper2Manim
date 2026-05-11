"""MiMo (Xiaomi Token Plan) client factory — thin wrapper over langchain-openai.ChatOpenAI."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Literal, TypeVar

from langchain_core.exceptions import OutputParserException
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError

from paper2manim.config import get_settings

log = logging.getLogger(__name__)
_T = TypeVar("_T", bound=BaseModel)

# Real MiMo (Xiaomi Token Plan) model IDs as returned by /v1/models — all lowercase
# with dotted version. The PascalCase names (MiMo-V2.5-Pro) are display-only.
# Verified against token-plan-cn.xiaomimimo.com/v1/models.
_MODEL_MAP = {
    "flash": "mimo-v2.5",       # default lightweight reasoning (storyboarder/coder/reviewer)
    "pro": "mimo-v2.5-pro",     # long-context / heavier reasoning (summarizer on long PDFs)
    # Fallbacks if v2.5 has rate limits / outages
    "v2": "mimo-v2-pro",
    "v2-omni": "mimo-v2-omni",
}


def get_llm(
    model: Literal["flash", "pro", "v2", "v2-omni"] = "flash",
    *,
    temperature: float = 0.2,
    max_tokens: int = 8192,
    timeout: float = 120,
    **kwargs,
) -> ChatOpenAI:
    """Build a MiMo-backed ChatOpenAI client.

    Examples
    --------
    >>> get_llm("flash").invoke("hi").content
    >>> get_llm("pro").with_structured_output(MyPydanticModel).invoke([("system", ...), ("user", ...)])
    """
    if model not in _MODEL_MAP:
        raise ValueError(f"Unknown model alias '{model}'. Choose 'flash' or 'pro'.")
    s = get_settings()
    if not s.MIMO_API_KEY:
        raise RuntimeError(
            "MIMO_API_KEY is empty. Copy .env.example to .env and fill in the tp- key from MiMo-API.txt."
        )
    return ChatOpenAI(
        model=_MODEL_MAP[model],
        api_key=s.MIMO_API_KEY,
        base_url=s.MIMO_BASE_URL,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        **kwargs,
    )


def safe_structured_invoke(
    llm: ChatOpenAI,
    model_cls: type[_T],
    messages: Sequence[tuple[str, str]],
    *,
    retries: int = 1,
) -> _T:
    """Invoke ``llm.with_structured_output(model_cls)`` with one automatic retry on
    schema/parse failures.

    Wraps the common pattern where an LLM occasionally returns commentary plus JSON,
    or JSON that fails pydantic validation. We retry once with a strict reminder.

    Raises
    ------
    OutputParserException / ValidationError
        If both attempts fail; callers should catch and convert to ``fatal_error``.
    """
    structured = llm.with_structured_output(model_cls)
    last_exc: Exception | None = None
    msgs: list[tuple[str, str]] = list(messages)
    for attempt in range(retries + 1):
        try:
            return structured.invoke(msgs)  # type: ignore[return-value]
        except (OutputParserException, ValidationError) as exc:
            last_exc = exc
            log.warning(
                "[llm.safe_structured_invoke] attempt %d/%d failed for %s: %s",
                attempt + 1,
                retries + 1,
                model_cls.__name__,
                str(exc)[:300],
            )
            if attempt < retries and msgs and msgs[-1][0] == "user":
                msgs = msgs[:-1] + [
                    (
                        "user",
                        msgs[-1][1]
                        + "\n\nIMPORTANT: respond with STRICT JSON matching the schema. "
                        "No prose, no commentary, no markdown fences.",
                    )
                ]
    # All retries exhausted
    assert last_exc is not None
    raise last_exc
