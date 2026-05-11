from __future__ import annotations

import json
import re
from typing import Any


CODE_FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json_object(text: str) -> dict[str, Any]:
    value = extract_json_value(text)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object.")
    return value


def extract_json_array(text: str) -> list[Any]:
    value = extract_json_value(text)
    if not isinstance(value, list):
        raise ValueError("Expected a JSON array.")
    return value


def extract_json_value(text: str) -> Any:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        match = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL | re.IGNORECASE)
        if match:
            cleaned = match.group(1).strip()

    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        object_start = cleaned.find("{")
        object_end = cleaned.rfind("}")
        array_start = cleaned.find("[")
        array_end = cleaned.rfind("]")
        if array_start != -1 and (object_start == -1 or array_start < object_start):
            start, end = array_start, array_end
        else:
            start, end = object_start, object_end
        if start == -1 or end == -1 or end <= start:
            raise
        value = json.loads(cleaned[start : end + 1])
    return value


def extract_python_code(text: str) -> str:
    match = CODE_FENCE_RE.search(text)
    code = match.group(1) if match else text
    return code.strip() + "\n"
