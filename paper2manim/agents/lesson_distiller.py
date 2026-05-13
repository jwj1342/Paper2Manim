"""LLM-driven Failure Pattern Lesson distiller.

Plugs into :data:`paper2manim.emb.distill.LessonDistiller` so the §4.4b
consolidation path can substitute it for the cheap default. The LLM receives
the before/after code + the validated diagnostic (traceback or VLM remark) and
must return a JSON object matching :class:`paper2manim.emb.schema.FailureBody`.

On any parse / API failure we fall back to
:func:`paper2manim.emb.distill.default_lesson_distiller`, which still returns
a valid FailureBody (built from line-diff heuristics). This means the
consolidation pipeline is robust to LLM outages — we may store a less
informative lesson, but we never lose the transition.
"""

from __future__ import annotations

import json
import logging
import re

from paper2manim.agents.vlm_scene_reviewer import _extract_first_json_object
from paper2manim.emb.distill import (
    TextTransition,
    VisualTransition,
    default_lesson_distiller,
)
from paper2manim.emb.schema import FailureBody
from paper2manim.llm import get_llm
from paper2manim.prompts import load_prompt

log = logging.getLogger(__name__)

_MAX_CODE_CHARS = 3500  # before + after each get this; final_code stays in EMB for fidelity


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


_FIELD_LIMITS = {
    "trigger_pattern": 400,
    "root_cause": 400,
    "fix_recipe": 400,
    "code_anti_example": 800,
    "code_good_example": 800,
    "vlm_diagnostic": 1000,
}


def _coerce_failure_body(payload: dict) -> FailureBody:
    """Map a possibly-noisy JSON payload onto a FailureBody.

    Missing fields become empty strings rather than triggering a Pydantic
    error, so a partial LLM answer still produces a usable record.
    """
    return FailureBody(
        trigger_pattern=_truncate(str(payload.get("trigger_pattern") or ""), _FIELD_LIMITS["trigger_pattern"])
        or "unknown trigger",
        root_cause=_truncate(str(payload.get("root_cause") or ""), _FIELD_LIMITS["root_cause"])
        or "unknown root cause",
        fix_recipe=_truncate(str(payload.get("fix_recipe") or ""), _FIELD_LIMITS["fix_recipe"])
        or "see code_good_example",
        code_anti_example=_truncate(
            str(payload.get("code_anti_example") or ""), _FIELD_LIMITS["code_anti_example"]
        ),
        code_good_example=_truncate(
            str(payload.get("code_good_example") or ""), _FIELD_LIMITS["code_good_example"]
        ),
        vlm_diagnostic=_truncate(
            str(payload.get("vlm_diagnostic") or ""), _FIELD_LIMITS["vlm_diagnostic"]
        ),
    )


def distill_lesson_llm(
    transition: VisualTransition | TextTransition,
    scene_description: str,
) -> FailureBody:
    """Call the LLM to produce a Lesson; fall back to heuristic on any error."""
    system = load_prompt("lesson_distiller")
    user = _build_user_prompt(transition, scene_description)
    try:
        llm = get_llm("render_fixer", temperature=0.0, max_tokens=1500)
        raw = llm.invoke([("system", system), ("user", user)]).content
        text = raw if isinstance(raw, str) else str(raw)
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "[lesson_distiller] LLM call failed for %s (%s); using default",
            transition.scene,
            exc,
        )
        return default_lesson_distiller(transition, scene_description)
    blob = _extract_first_json_object(text)
    if not blob:
        log.warning(
            "[lesson_distiller] no JSON object in LLM output for %s; using default",
            transition.scene,
        )
        return default_lesson_distiller(transition, scene_description)
    try:
        payload = json.loads(blob)
    except json.JSONDecodeError as exc:
        log.warning(
            "[lesson_distiller] JSON parse failed for %s (%s); using default",
            transition.scene,
            exc,
        )
        return default_lesson_distiller(transition, scene_description)
    if not isinstance(payload, dict):
        log.warning(
            "[lesson_distiller] LLM returned non-object JSON for %s; using default",
            transition.scene,
        )
        return default_lesson_distiller(transition, scene_description)
    return _coerce_failure_body(payload)


def _build_user_prompt(
    transition: VisualTransition | TextTransition, scene_description: str
) -> str:
    parts: list[str] = []
    parts.append(f"## Scene name\n{transition.scene}\n")
    parts.append(f"\n## Scene description\n{scene_description.strip() or '<missing>'}\n")
    if isinstance(transition, VisualTransition):
        parts.append(
            f"\n## Transition type\nvisual_reflection — VLM avg score "
            f"{transition.before_score:.2f} → {transition.after_score:.2f}\n"
        )
        parts.append(
            f"\n## Original VLM revision_instruction\n{transition.revision_instruction.strip() or '<empty>'}\n"
        )
    else:
        parts.append(
            f"\n## Transition type\ntext_reflection — render error (category={transition.error_category!r}) → success\n"
        )
        parts.append(f"\n## Error message\n{transition.error_message.strip() or '<empty>'}\n")
        if transition.traceback_tail:
            parts.append(f"\n## Traceback tail\n```\n{transition.traceback_tail.strip()}\n```\n")
    parts.append(
        f"\n## Before code (FAILING / LOW-SCORING)\n```python\n"
        f"{_truncate(transition.before_code, _MAX_CODE_CHARS)}\n```\n"
    )
    parts.append(
        f"\n## After code (FIXED / IMPROVED)\n```python\n"
        f"{_truncate(transition.after_code, _MAX_CODE_CHARS)}\n```\n"
    )
    parts.append("\n## Task\nReturn the Lesson JSON object now.")
    return "".join(parts)


# Useful for tests / dry-run distillation:
def _looks_like_json_object(s: str) -> bool:
    return bool(re.match(r"^\s*\{", s))
