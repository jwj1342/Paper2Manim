"""figure_understander agent (Phase 2 + 3).

Reads ``state.figures``, calls a VLM once per figure, and attaches both
``FigureSemantics`` (classification — fig_type / redrawable / salience / summary)
and an optional ``FigureRecipe`` (reconstruction layout for redrawable figures)
to each figure under the ``semantics`` and ``recipe`` keys respectively.

Failures degrade gracefully: a single bad figure ends up with both keys
``None`` rather than aborting the run, and a disabled VLM (``get_vlm()``
returns ``None``) short-circuits the whole node. A defensive consistency
check drops the recipe when ``semantics.redrawable`` is False, regardless of
what the VLM returned.

This node NEVER sets ``fatal_error`` — figure understanding is enrichment.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from paper2manim.artifacts import append_trace
from paper2manim.prompts import load_prompt
from paper2manim.schemas.figure import FigureUnderstanding
from paper2manim.state import PaperState
from paper2manim.vlm import get_vlm

log = logging.getLogger(__name__)


def _safe_parse_understanding(raw: Any) -> tuple[dict | None, dict | None]:
    """Best-effort JSON parse + FigureUnderstanding validation.

    Returns ``(semantics_dict, recipe_dict)``. Either or both may be ``None``:
    - ``(None, None)`` — couldn't even parse the response
    - ``(semantics, None)`` — classification OK; recipe missing/invalid OR redrawable=False
    - ``(semantics, recipe)`` — full understanding
    """
    text = raw if isinstance(raw, str) else str(raw)
    # Strip code fences / surrounding prose: grab the first {...} block
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None, None
    try:
        parsed = FigureUnderstanding.model_validate_json(m.group(0))
    except ValidationError as exc:
        log.warning("[figure_understander] schema validation failed: %s", str(exc)[:200])
        return None, None
    semantics = parsed.semantics.model_dump()
    recipe = parsed.recipe.model_dump() if parsed.recipe is not None else None
    # Defensive consistency check: redrawable=False should imply recipe=None
    if recipe is not None and not parsed.semantics.redrawable:
        log.info(
            "[figure_understander] dropping recipe (semantics.redrawable=False) for consistency"
        )
        recipe = None
    return semantics, recipe


def figure_understander_node(state: PaperState) -> dict[str, Any]:
    figures = state.get("figures") or []
    if not figures:
        log.info("[figure_understander] no figures in state — skipping")
        return {}

    vlm = get_vlm()
    if vlm is None:
        log.info("[figure_understander] VLM disabled — skipping (figures keep semantics=None)")
        return {}

    system = load_prompt("figure_understander")
    enriched: list[dict] = []
    n_ok = 0
    n_with_recipe = 0
    for fig in figures:
        new_fig = dict(fig)  # shallow copy — never mutate the state dict in place
        path = fig.get("path")
        if not path or not Path(path).exists():
            log.warning(
                "[figure_understander] %s has no readable path; skipping",
                fig.get("fig_id"),
            )
            new_fig["semantics"] = None
            new_fig["recipe"] = None
            enriched.append(new_fig)
            continue
        try:
            raw = vlm.review_scene(system, path)
            semantics, recipe = _safe_parse_understanding(raw)
        except Exception as exc:  # noqa: BLE001 — VLM call must not crash the run
            log.warning(
                "[figure_understander] VLM call failed for %s: %s: %s",
                fig.get("fig_id"),
                type(exc).__name__,
                str(exc)[:200],
            )
            semantics, recipe = None, None
        new_fig["semantics"] = semantics
        new_fig["recipe"] = recipe
        if semantics is not None:
            n_ok += 1
        if recipe is not None:
            n_with_recipe += 1
        enriched.append(new_fig)

    if state.get("run_id"):
        append_trace(
            state["run_id"],
            "figure_understander",
            {
                "n_figures": len(figures),
                "n_understood": n_ok,
                "n_with_recipe": n_with_recipe,
            },
        )
    log.info(
        "[figure_understander] %d/%d figures understood, %d with recipe",
        n_ok,
        len(figures),
        n_with_recipe,
    )
    return {"figures": enriched}
