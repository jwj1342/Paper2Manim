"""Visual revision agent — takes a failed visual review and rewrites the scene code.

This is the "fix the visuals" side of the VLM loop. Distinct from
``agents.coder`` because:

* The input is a working-but-visually-poor script (not a crash), plus a
  ``revision_instruction`` from the VLM reviewer.
* The model is bound to the ``visual_reviser`` role in YAML, so it can be a
  different (or the same) text model than the one used for first-shot coding.

Output is the entire revised Python source (no markdown fence is required;
fences are stripped if present).
"""

from __future__ import annotations

import logging
import re
from typing import Any

from paper2manim.llm import get_llm
from paper2manim.prompts import load_prompt

log = logging.getLogger(__name__)

_PY_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)\n```", re.DOTALL)


def _strip_fence(text: str) -> str:
    m = _PY_BLOCK_RE.search(text)
    return m.group(1).strip() if m else text.strip()


def revise_code(
    scene: dict[str, Any],
    current_code: str,
    review: dict[str, Any],
    *,
    summary: dict[str, Any] | None = None,
) -> str:
    """Return the new Manim source string."""
    system_prompt = load_prompt("visual_revision_agent")
    scene_id = scene.get("name") or "<unknown>"
    parts: list[str] = []
    parts.append(f"## SceneSpec\n```json\n{_compact_json({'name': scene_id, 'description': scene.get('description', '')})}\n```\n")
    if summary:
        parts.append(f"## Paper summary\n```json\n{_compact_json(summary)}\n```\n")
    parts.append("## Current Manim code (renders but looks bad)\n```python\n")
    parts.append(current_code)
    parts.append("\n```\n")
    parts.append("## VisualReviewResult\n```json\n")
    parts.append(_compact_json({
        "decision": review.get("decision"),
        "scores": review.get("scores"),
        "issues": review.get("issues"),
        "revision_instruction": review.get("revision_instruction"),
        "paper_alignment_notes": review.get("paper_alignment_notes"),
    }))
    parts.append("\n```\n")
    user = "".join(parts)

    llm = get_llm("visual_reviser", temperature=0.1)
    log.info("[visual_revision] scene=%s decision=%s", scene_id, review.get("decision"))
    raw = llm.invoke([("system", system_prompt), ("user", user)]).content
    return _strip_fence(raw if isinstance(raw, str) else str(raw))


def _compact_json(payload: Any) -> str:
    import json
    return json.dumps(payload, ensure_ascii=False, indent=2)
