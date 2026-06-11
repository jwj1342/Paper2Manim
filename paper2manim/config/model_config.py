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
    # Azure AI Foundry project endpoints use an api-version query parameter.
    api_version: str = ""

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
            api_version=str(value.get("api_version") or "").strip(),
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
    tts_config: TTSConfig | None = None

    @classmethod
    def from_dicts(
        cls,
        models: list[ModelConfig],
        roles: dict[str, str],
        *,
        tts_config: TTSConfig | None = None,
    ) -> ModelSettings:
        by_name = {model.name: model for model in models}
        return cls(models=by_name, roles=roles, tts_config=tts_config)

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


@dataclass(frozen=True)
class TTSConfig:
    """Voiceover text-to-speech configuration (independent from chat/VLM models).

    Read from the ``tts:`` block in config.yaml. The TTS provider surface
    (audio output, streaming, voice selection) is fundamentally different
    from the chat completion surface, so TTS gets its own config struct
    rather than reusing :class:`ModelConfig`.
    """

    provider: str  # "openai" (extensible to "azure" / "elevenlabs")
    model: str
    api_key: str
    base_url: str = ""
    voice: str = "alloy"
    audio_format: str = "wav"
    speed: float = 1.0
    timeout: int = 120

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TTSConfig:
        provider = str(value.get("provider") or "openai").strip()
        model = str(value.get("model") or "").strip()
        api_key = str(value.get("api_key") or "").strip()
        if not model:
            raise KeyError("tts config is missing required field 'model'.")
        # api_key: required for openai, optional for edge and mock.
        if provider == "openai" and not api_key:
            raise KeyError(
                "tts config with provider='openai' requires 'api_key'."
            )
        return cls(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=str(value.get("base_url") or "").strip(),
            voice=str(value.get("voice") or "alloy").strip(),
            audio_format=str(value.get("audio_format") or "wav").strip(),
            speed=_float(value.get("speed"), 1.0),
            timeout=_int(value.get("timeout"), 120),
        )


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
