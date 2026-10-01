"""LLM-driven High-Score Rationale writer.

Plugs into :data:`paper2manim.emb.distill.RationaleWriter` so the positive memory
consolidation path can substitute it for the cheap default.

The writer receives the selected code, scene description, and VLM score.
"""

from __future__ import annotations

import logging

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
    """Distill the selected high-scoring code into a rationale of at most 400 characters."""
    system = load_prompt("rationale_writer")
    code_snippet = _truncate(scored.final_code or "", _MAX_CODE_CHARS)
    user_parts = [
        f"## Scene name\n{scored.name}\n",
        f"\n## Scene description\n{scene_description.strip() or '<missing>'}\n",
        f"\n## Final score (avg)\n{scored.final_score:.2f}/100\n"
        if scored.had_vlm_review
        else "\n## Final score\nVLM was disabled for this run\n",
        f"\n## Final Manim code\n```python\n{code_snippet}\n```\n",
        "\n## Task\nWrite the rationale described in the system prompt now.",
    ]
    user = "".join(user_parts)
    llm = get_llm("final_summarizer", temperature=0.3, max_tokens=400)
    raw = llm.invoke([("system", system), ("user", user)]).content
    text = raw if isinstance(raw, str) else str(raw)
    rationale = _clean_rationale(text)
    if not rationale:
        raise ValueError("rationale_writer returned empty text")
    return rationale


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
