from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

from paper2manim.config.model_config import ModelConfig, ModelSettings

_ENV_VAR_RE = re.compile(r"\$([A-Z_][A-Z0-9_]*)")


def load_model_settings(path: str | Path) -> ModelSettings:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model configuration file not found: {path}. "
            f"Copy config.example.yaml to {path} and fill in your API keys."
        )

    data = _read_yaml(path)
    raw_models = _require_list(data, "models")
    raw_roles = _require_dict(data, "model_roles")

    models = [_parse_model(item, index) for index, item in enumerate(raw_models)]
    roles = {str(key): str(value).strip() for key, value in raw_roles.items()}

    _validate_roles_exist(roles, models)
    _validate_vision_checker(roles, models)

    return ModelSettings.from_dicts(models, roles)


def _parse_model(raw: Any, index: int) -> ModelConfig:
    if not isinstance(raw, dict):
        raise TypeError(f"Model at index {index} must be a mapping, got {type(raw).__name__}.")
    raw_copy = dict(raw)
    raw_copy["api_key"] = _resolve_env(raw_copy.get("api_key", ""))
    return ModelConfig.from_dict(raw_copy)


def _resolve_env(value: Any) -> str:
    text = str(value).strip()
    match = _ENV_VAR_RE.fullmatch(text)
    if match:
        var_name = match.group(1)
        resolved = os.environ.get(var_name, "")
        if not resolved:
            raise RuntimeError(
                f"Environment variable '{var_name}' is not set. "
                f"It is required by a model configuration."
            )
        return resolved
    return text


def _validate_roles_exist(roles: dict[str, str], models: list[ModelConfig]) -> None:
    model_names = {model.name for model in models}
    for role, model_name in roles.items():
        if model_name not in model_names:
            available = ", ".join(sorted(model_names))
            raise ValueError(
                f"Model role '{role}' references model '{model_name}', "
                f"but no model with that name is defined. "
                f"Defined models: {available}"
            )


def _validate_vision_checker(roles: dict[str, str], models: list[ModelConfig]) -> None:
    vision_model_name = roles.get("vision_checker")
    if vision_model_name is None:
        return
    by_name = {model.name: model for model in models}
    vision_model = by_name.get(vision_model_name)
    if vision_model is None:
        return
    if not vision_model.supports_vision:
        raise ValueError(
            f"Role 'vision_checker' points to model '{vision_model.name}', "
            f"but it has supports_vision=false. "
            f"A vision-checker model must support vision."
        )


def _read_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise TypeError(f"Config file {path} must contain a YAML mapping.")
    return raw


def _require_list(data: dict[str, Any], key: str) -> list[Any]:
    value = data.get(key)
    if value is None:
        raise KeyError(f"Config file is missing required key '{key}'.")
    if not isinstance(value, list):
        raise TypeError(f"Config key '{key}' must be a list, got {type(value).__name__}.")
    return value


def _require_dict(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if value is None:
        raise KeyError(f"Config file is missing required key '{key}'.")
    if not isinstance(value, dict):
        raise TypeError(f"Config key '{key}' must be a mapping, got {type(value).__name__}.")
    return value
