"""VLM scene reviewer — feed (scene spec, frame montage) → 6-dim scored review.

Uses :func:`paper2manim.llm.get_vlm` to honor the YAML-bound ``vision_checker``
role. Returns a normalized review dict the graph can route on. JSON is parsed
defensively; on any parse failure we synthesize a conservative ``revise``
verdict so the loop can still proceed.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from paper2manim.config.config_loader import load_model_settings
from paper2manim.config.model_config import ModelSettings
from paper2manim.infrastructure.vlm.client import VLMClient
from paper2manim.infrastructure.vlm.factory import build_vlm_client
from paper2manim.prompts import load_prompt

log = logging.getLogger(__name__)


_SCORE_KEYS = (
    "paper_alignment",
    "visual_clarity",
    "readability",
    "layout_balance",
    "visual_focus",
    "animation_perceived",
)


def _extract_first_json_object(raw: str) -> str | None:
    """Return the first top-level ``{...}`` substring with balanced braces.

    Replaces a greedy ``r"\\{.*\\}"`` regex that could span two JSON objects when
    a chatty model emits e.g. a thinking trace followed by the answer, producing
    invalid input for ``json.loads``. We walk the string once tracking depth and
    ignoring braces inside double-quoted strings (with simple ``\\"`` escape
    handling) — good enough for the JSON shape our prompt asks for.
    """
    in_str = False
    escape = False
    depth = 0
    start = -1
    for i, ch in enumerate(raw):
        if escape:
            escape = False
            continue
        if in_str:
            if ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start >= 0:
                return raw[start : i + 1]
    return None


def _coerce_int(value: Any, default: int = 1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_scores(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {k: 1 for k in _SCORE_KEYS}
    return {k: _coerce_int(value.get(k), 1) for k in _SCORE_KEYS}


def _conservative_review(scene_id: str, raw: str, error: str) -> dict[str, Any]:
    """Used when the VLM response is unparseable — keeps the graph moving."""
    return {
        "scene_id": scene_id,
        "decision": "revise",
        "scores": {k: 1 for k in _SCORE_KEYS},
        "issues": [
            {
                "type": "other",
                "severity": "high",
                "evidence": f"VLM response could not be parsed: {error}",
                "suggestion": "Retry the scene with a conservative layout fix.",
            }
        ],
        "paper_alignment_notes": "VLM response unparsed.",
        "revision_instruction": "Make conservative layout fixes: reduce text density, enlarge main visual, increase spacing.",
        "requires_replanning": False,
        "raw_response": raw,
    }


def parse_vlm_response(raw: str, scene_id: str) -> dict[str, Any]:
    """Best-effort JSON extraction — never raises."""
    blob = _extract_first_json_object(raw or "")
    if not blob:
        return _conservative_review(scene_id, raw, "no JSON object found")
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError as exc:
        return _conservative_review(scene_id, raw, f"json.JSONDecodeError: {exc}")
    decision = str(obj.get("decision") or "revise").strip().lower()
    if decision not in {"pass", "revise", "fail"}:
        decision = "revise"
    return {
        "scene_id": str(obj.get("scene_id") or scene_id),
        "decision": decision,
        "scores": _coerce_scores(obj.get("scores")),
        "issues": obj.get("issues") if isinstance(obj.get("issues"), list) else [],
        "paper_alignment_notes": str(obj.get("paper_alignment_notes") or ""),
        "revision_instruction": str(obj.get("revision_instruction") or ""),
        "requires_replanning": bool(obj.get("requires_replanning", False)),
        "raw_response": raw,
    }


def _build_scene_spec_payload(
    scene: dict[str, Any],
    *,
    summary: dict[str, Any] | None,
    scene_idx: int,
) -> dict[str, Any]:
    """Project our (storyboard scene + summary) onto the SceneSpec fields the VLM prompt expects."""
    paper_claim = ""
    paper_evidence: list[str] = []
    if summary:
        paper_claim = (
            (summary.get("key_contributions") or [""])[0]
            if summary.get("key_contributions")
            else ""
        )
        paper_evidence = list(summary.get("main_concepts") or [])
    return {
        "scene_id": scene.get("name") or f"scene_{scene_idx}",
        "title": scene.get("name") or "",
        "paper_role": "explanatory",
        "paper_claim": paper_claim,
        "paper_evidence": paper_evidence,
        "visual_mapping": scene.get("description", ""),
        "main_visual_object": scene.get("description", "").split(".")[0][:120],
        "animation_beats": [scene.get("description", "")],
        "final_takeaway": scene.get("description", "").split(".")[-1].strip()[:200],
    }


def review_scene(
    scene: dict[str, Any],
    montage_path: Path | str,
    *,
    summary: dict[str, Any] | None = None,
    scene_idx: int = 0,
    client: VLMClient | None = None,
) -> dict[str, Any]:
    """Run one VLM review pass.

    The graph normally lets ``client`` default to the YAML-routed VLM; tests
    inject a :class:`MockVLMClient` to exercise the loop without a real call.
    """
    if client is None:
        client = _build_default_vlm_client()
    spec = _build_scene_spec_payload(scene, summary=summary, scene_idx=scene_idx)
    system_prompt = load_prompt("vlm_scene_reviewer")
    payload = f"SceneSpec JSON:\n{json.dumps(spec, ensure_ascii=False, indent=2)}"
    full_prompt = f"{system_prompt}\n\n{payload}"
    log.info("[vlm_review] scene=%s montage=%s", spec["scene_id"], montage_path)
    raw = client.review_scene(full_prompt, montage_path)
    return parse_vlm_response(raw, spec["scene_id"])


def _build_default_vlm_client() -> VLMClient:
    """Materialize the YAML-bound vision_checker model into a VLM client."""
    from paper2manim.config import PROJECT_ROOT

    config_path = PROJECT_ROOT / "config.yaml"
    if not config_path.exists():
        raise RuntimeError(
            "VLM review requires config.yaml with a supports_vision=true model "
            "bound to role 'vision_checker'."
        )
    settings: ModelSettings = load_model_settings(config_path)
    cfg = settings.model_for_role("vision_checker")
    return build_vlm_client(cfg)
