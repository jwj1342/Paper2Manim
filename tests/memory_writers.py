"""Deterministic memory writers for API-free tests."""

from paper2manim.emb.distill import ScoredScene, TextTransition, VisualTransition
from paper2manim.emb.schema import FailureBody


def mock_rationale_writer(scored: ScoredScene, scene_description: str) -> str:
    """Fallback rationale used when no LLM writer is injected.

    Useful for offline bootstrap runs where we don't want to spend LLM tokens
    on prose. The text retains enough structure (score + montage path) to be
    informative when retrieved later.
    """
    montage = str(scored.final_montage_path) if scored.final_montage_path else "<none>"
    return (
        f"High-scoring scene '{scored.name}' (avg={scored.final_score:.2f}, "
        f"v_rev={scored.final_v_rev}, montage={montage}). "
        f"Scene description: {scene_description.strip()[:300]}"
    )


def mock_lesson_distiller(
    transition: VisualTransition | TextTransition, scene_description: str
) -> FailureBody:
    """Fallback lesson body — minimal but valid, no LLM needed.

    Pulls anti / good examples by extracting the first significantly differing
    lines from before / after code. Used to isolate pipeline tests from model calls.
    """
    if isinstance(transition, VisualTransition):
        trigger = f"Visual scene with low layout/clarity score (avg={transition.before_score:.2f})"
        root_cause = transition.revision_instruction.strip()[:300] or "VLM flagged for revision"
        fix_recipe = f"Apply: {transition.revision_instruction.strip()[:300]}"
        diagnostic = transition.revision_instruction
    else:
        trigger = (
            f"Render fails with category={transition.error_category!r} "
            f"during {scene_description.strip()[:120]}"
        )
        root_cause = transition.error_message[:300] or "render error"
        fix_recipe = "See code_good_example for the working version."
        diagnostic = transition.traceback_tail
    anti = _first_distinct_lines(transition.before_code, transition.after_code, n=8)
    good = _first_distinct_lines(transition.after_code, transition.before_code, n=8)
    return FailureBody(
        trigger_pattern=trigger,
        root_cause=root_cause,
        fix_recipe=fix_recipe,
        code_anti_example=anti,
        code_good_example=good,
        vlm_diagnostic=diagnostic[:1000],
    )


def _first_distinct_lines(a: str, b: str, *, n: int = 8) -> str:
    """Return up to ``n`` lines from ``a`` that don't appear verbatim in ``b``."""
    bset = set(b.splitlines())
    out: list[str] = []
    for line in a.splitlines():
        if line.strip() and line not in bset:
            out.append(line)
            if len(out) >= n:
                break
    return "\n".join(out)
