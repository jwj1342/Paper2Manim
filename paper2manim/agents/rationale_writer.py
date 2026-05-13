"""LLM-driven High-Score Rationale writer.

Plugs into :data:`paper2manim.emb.distill.RationaleWriter` so the §4.4a
consolidation path can substitute it for the cheap default.

If the scene has a frame montage (i.e. VLM was on) we pass the image to a VLM;
otherwise we degrade to a text-only LLM call against the final code. Either
way the output is a single paragraph of prose suitable as in-context guidance.
"""

from __future__ import annotations

import logging
from pathlib import Path

from paper2manim.emb.distill import ScoredScene
from paper2manim.llm import get_llm
from paper2manim.prompts import load_prompt

log = logging.getLogger(__name__)

_MAX_CODE_CHARS = 4000  # cap to keep prompt small; final_code is the SoT for the record


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def write_rationale_llm(scored: ScoredScene, scene_description: str) -> str:
    """Call the LLM bound to ``final_summarizer`` (or ``flash`` fallback) to
    produce a rationale string. Falls back to the deterministic
    :func:`paper2manim.emb.distill.default_rationale_writer` on any error."""
    system = load_prompt("rationale_writer")
    code_snippet = _truncate(scored.final_code or "", _MAX_CODE_CHARS)
    user_parts = [
        f"## Scene name\n{scored.name}\n",
        f"\n## Scene description\n{scene_description.strip() or '<missing>'}\n",
        f"\n## Final score (avg)\n{scored.final_score:.2f}/5\n"
        if scored.had_vlm_review
        else "\n## Final score\nVLM was disabled for this run\n",
        f"\n## Final Manim code\n```python\n{code_snippet}\n```\n",
        "\n## Task\nWrite the rationale described in the system prompt now.",
    ]
    user = "".join(user_parts)
    try:
        # Use the same role as the summarizer for short prose tasks; fall back
        # to ``flash`` when ``final_summarizer`` isn't bound. ``get_llm`` resolves
        # the alias to whichever model the user pointed at.
        llm = get_llm("final_summarizer", temperature=0.3, max_tokens=400)
        raw = llm.invoke([("system", system), ("user", user)]).content
        text = raw if isinstance(raw, str) else str(raw)
        return _clean_rationale(text)
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "[rationale_writer] LLM call failed for %s (%s); using default",
            scored.name,
            exc,
        )
        # Deferred import to avoid a circular module dependency.
        from paper2manim.emb.distill import default_rationale_writer

        return default_rationale_writer(scored, scene_description)


def _clean_rationale(text: str) -> str:
    """Strip code fences / surrounding quotes the model sometimes adds."""
    t = text.strip()
    if t.startswith("```"):
        # Drop the first and last fence lines.
        lines = t.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    if (t.startswith('"') and t.endswith('"')) or (t.startswith("'") and t.endswith("'")):
        t = t[1:-1].strip()
    return _truncate(t, 400)


# Helper exposing the path of the montage (kept for future VLM-grounded
# rationale: pass the image to a vision model). Wired but not yet used because
# get_vlm() requires a separate VLM-bound role; the text-only path is enough
# for the v1 pipeline and is cheaper.
def _montage_path_for(scored: ScoredScene) -> Path | None:
    return scored.final_montage_path
