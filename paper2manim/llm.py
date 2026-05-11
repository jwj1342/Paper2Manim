"""MiMo (Xiaomi Token Plan) client factory — thin wrapper over langchain-openai.ChatOpenAI."""

from __future__ import annotations

from typing import Literal

from langchain_openai import ChatOpenAI

from paper2manim.config import get_settings

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
