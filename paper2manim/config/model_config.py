from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    name: str
    display_name: str
    provider: str
    model: str
    api_key: str
    base_url: str
    temperature: float = 0.2
    max_tokens: int = 4096
    timeout: int = 600
    supports_vision: bool = False
    supports_thinking: bool = False
    # "header_api_key" sends Authorization the way each SDK defaults to:
    # OpenAI-compatible -> Bearer (in OpenAI SDK), Anthropic -> x-api-key.
    # "bearer" forces Authorization: Bearer <key> via default_headers — required
    # for Azure-hosted Claude on the Anthropic Messages endpoint.
    auth_style: str = "header_api_key"
    # Some models (e.g. Claude Opus 4.7) reject `temperature` outright. Set
    # true and the LLM/VLM factories will omit the parameter at call time.
    omit_temperature: bool = False

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ModelConfig:
        name = _require_str(value, "name")
        api_key = _require_str(value, "api_key")
        return cls(
            name=name,
            display_name=str(value.get("display_name") or name),
            provider=str(value.get("provider") or "openai_compatible").strip(),
            model=str(value.get("model") or "").strip(),
            api_key=api_key,
            base_url=str(value.get("base_url") or "").strip(),
            temperature=_float(value.get("temperature"), 0.2),
            max_tokens=_int(value.get("max_tokens"), 4096),
            timeout=_int(value.get("timeout"), 600),
            supports_vision=bool(value.get("supports_vision", False)),
            supports_thinking=bool(value.get("supports_thinking", False)),
            auth_style=str(value.get("auth_style") or "header_api_key").strip(),
            omit_temperature=bool(value.get("omit_temperature", False)),
        )

    def __repr__(self) -> str:
        return (
            f"ModelConfig(name={self.name!r}, provider={self.provider!r}, "
            f"model={self.model!r}, base_url={self.base_url!r}, "
            f"supports_vision={self.supports_vision}, "
            f"supports_thinking={self.supports_thinking}, "
            f"auth_style={self.auth_style!r})"
        )


@dataclass(frozen=True)
class ModelSettings:
    models: dict[str, ModelConfig]
    roles: dict[str, str]

    @classmethod
    def from_dicts(
        cls,
        models: list[ModelConfig],
        roles: dict[str, str],
    ) -> ModelSettings:
        by_name = {model.name: model for model in models}
        return cls(models=by_name, roles=roles)

    def model_for_role(self, role: str) -> ModelConfig:
        model_name = self.roles.get(role)
        if model_name is None:
            available = ", ".join(sorted(self.roles))
            raise KeyError(f"Unknown model role '{role}'. Available roles: {available}")
        model = self.models.get(model_name)
        if model is None:
            known = ", ".join(sorted(self.models))
            raise KeyError(
                f"Model '{model_name}' (role '{role}') not found in model definitions. "
                f"Defined models: {known}"
            )
        return model


def _require_str(value: dict[str, Any], key: str) -> str:
    raw = value.get(key)
    if raw is None:
        raise KeyError(f"Model config is missing required field '{key}'.")
    return str(raw).strip()


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
