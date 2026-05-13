"""VLM scene reviewer — feed (scene spec, frame montage) → 3-dim scored review.

Resolves the ``vision_checker`` model via
:func:`paper2manim.llm.vision_checker_config` (cached YAML settings) and
materializes a client through
:func:`paper2manim.infrastructure.vlm.factory.build_vlm_client`. Returns a
normalized review dict the graph can route on. JSON is parsed defensively; on
any parse failure we synthesize a conservative ``revise`` verdict so the loop
can still proceed.

The schema is the proposal §4.2 canonical form: three dimensions
(``logic_flow``, ``layout_occlusion``, ``accuracy``) on a 0–100 scale, with a
``decision`` of ``pass | revise | fail``. Scenes whose average score is ≥ 90
are auto-upgraded to ``pass`` even if the VLM said ``revise``, so the loop
doesn't keep grinding when the model is reluctant to self-pass.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from paper2manim.infrastructure.vlm.client import VLMClient
from paper2manim.infrastructure.vlm.factory import build_vlm_client
from paper2manim.llm import vision_checker_config
from paper2manim.prompts import load_prompt

log = logging.getLogger(__name__)


_SCORE_KEYS = (
    "logic_flow",
    "layout_occlusion",
    "accuracy",
)

# Average-score threshold above which a "revise" verdict is auto-upgraded to
# "pass". Without this the model rarely self-passes and every scene burns the
# full revision cap; see Issue #12 / docs/vlm_experiment.md baseline.
_AUTO_PASS_AVG = 90.0

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


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_score_or_none(value: Any) -> int | None:
    """Return a [0,100]-clamped int, ``None`` for missing/garbage values.

    Used by :func:`_coerce_scores` so that a VLM response missing one dimension
    records ``null`` instead of silently bottoming out at ``0`` and dragging the
    scene average down. The 0–100 clamp protects downstream consumers from a
    model that exceeds the schema (e.g. emits ``120`` or ``-5``).
    """
    if value is None:
        return None
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    if n < 0:
        return 0
    if n > 100:
        return 100
    return n


def _coerce_scores(value: Any) -> dict[str, int | None]:
    if not isinstance(value, dict):
        return {k: None for k in _SCORE_KEYS}
    return {k: _coerce_score_or_none(value.get(k)) for k in _SCORE_KEYS}


def _average_score(scores: dict[str, int | None]) -> float | None:
    """Return the mean of non-None scores, or ``None`` if everything is missing."""
    present = [v for v in scores.values() if v is not None]
    if not present:
        return None
    return sum(present) / len(present)


def _conservative_review(scene_id: str, raw: str, error: str) -> dict[str, Any]:
    """Used when the VLM response is unparseable — keeps the graph moving.

    All three dimensions are deliberately filled with ``0`` (not ``None``) because
    "model produced garbage" is itself the worst possible quality signal, and we
    want this entry to *look* bad in the trend table rather than be excluded
    from it.
    """
    return {
        "scene_id": scene_id,
        "decision": "revise",
        "scores": {k: 0 for k in _SCORE_KEYS},
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
    """Best-effort JSON extraction — never raises.

    Applies the ≥ :data:`_AUTO_PASS_AVG` auto-pass bypass: if the VLM said
    ``revise`` but the average of the three dimensions is high enough, treat
    it as a pass. The original decision is kept on ``raw_decision`` for the
    trace so we can audit how often the bypass fires.
    """
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
    scores = _coerce_scores(obj.get("scores"))
    raw_decision = decision
    avg = _average_score(scores)
    if decision == "revise" and avg is not None and avg >= _AUTO_PASS_AVG:
        decision = "pass"
    return {
        "scene_id": str(obj.get("scene_id") or scene_id),
        "decision": decision,
        "raw_decision": raw_decision,
        "scores": scores,
        "average_score": avg,
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
    """Project our (storyboard scene + summary) onto the SceneSpec fields the VLM prompt expects.

    paper_claim resolution, in order:
      1. ``scene["paper_claim"]`` if the storyboarder filled it (new schema).
      2. ``summary["key_contributions"][scene_idx]`` if that contribution exists.
      3. ``scene["description"]`` truncated as a last-resort claim.

    This avoids the prior bug where every scene was judged against
    ``key_contributions[0]`` regardless of which scene it was.
    """
    paper_evidence: list[str] = []
    if summary:
        paper_evidence = list(summary.get("main_concepts") or [])

    scene_claim = (scene.get("paper_claim") or "").strip()
    if not scene_claim and summary:
        contribs = summary.get("key_contributions") or []
        if scene_idx < len(contribs):
            scene_claim = str(contribs[scene_idx])
    if not scene_claim:
        scene_claim = (scene.get("description") or "").strip()[:240]

    return {
        "scene_id": scene.get("name") or f"scene_{scene_idx}",
        "title": scene.get("name") or "",
        "paper_role": "explanatory",
        "paper_claim": scene_claim,
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
    """Materialize the YAML-bound vision_checker model into a VLM client.

    Pulls the :class:`ModelConfig` from :func:`paper2manim.llm.vision_checker_config`,
    which returns the cached YAML settings — re-reading config.yaml on every
    review (5 scenes × 1–3 reviews = ~15 redundant parses) is wasteful.
    """
    return build_vlm_client(vision_checker_config())
