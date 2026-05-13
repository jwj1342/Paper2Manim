"""RAG layer — query the EMB for the current SceneSpec and render prompt blocks.

Two responsibilities, kept in this single module so call sites (the graph node
and the Coder agent) can stay simple:

1. :func:`retrieve_for_scene` — given a scene description and an EMB, return
   the top-k records of each polarity.
2. :func:`render_reference_examples_block` / :func:`render_known_pitfalls_block`
   — format retrieved records into self-contained markdown blocks safe to
   inject into the Coder prompt.

The output of (1) is :class:`RetrievalBundle`; converting it to plain dicts via
:meth:`RetrievalBundle.to_state_dict` is what the graph node will store on
PaperState (the TypedDict can't carry Pydantic models cleanly).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from paper2manim.emb.manager import EpisodicMemoryBank, RetrievedRecord

log = logging.getLogger(__name__)


# Hard limits — keep injected blocks compact so they don't blow the prompt.
_MAX_CODE_FULL_CHARS = 1200
_MAX_CODE_FRAGMENT_CHARS = 400
_MAX_RATIONALE_CHARS = 400


@dataclass
class RetrievalBundle:
    success: list[RetrievedRecord] = field(default_factory=list)
    failure: list[RetrievedRecord] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.success and not self.failure

    def to_state_dict(self) -> dict[str, list[dict[str, Any]]]:
        """Wire-format suitable for PaperState (no Pydantic instances)."""
        return {
            "success": [_record_to_state_dict(r) for r in self.success],
            "failure": [_record_to_state_dict(r) for r in self.failure],
        }


def _record_to_state_dict(retrieved: RetrievedRecord) -> dict[str, Any]:
    rec = retrieved.record
    # Strip the embedding from the wire form — it's large and the consumer
    # (Coder prompt builder) never needs it.
    body = rec.body.model_dump()
    context = rec.context.model_dump()
    context.pop("task_embedding", None)
    return {
        "id": rec.id,
        "polarity": rec.polarity,
        "similarity": retrieved.similarity,
        "context": context,
        "body": body,
        "provenance": rec.provenance.model_dump(),
    }


def retrieve_for_scene(
    emb: EpisodicMemoryBank,
    scene_text: str,
    *,
    k_success: int = 2,
    k_failure: int = 3,
    bump_hit: bool = True,
) -> RetrievalBundle:
    """Top-k for each polarity. Empty EMB → empty bundle, no exception."""
    try:
        s_hits = emb.query(scene_text, polarity="success", k=k_success, bump_hit=bump_hit)
        f_hits = emb.query(scene_text, polarity="failure", k=k_failure, bump_hit=bump_hit)
    except Exception as exc:  # noqa: BLE001 — never let RAG crash the graph
        log.warning("[emb.retrieve] query failed (%s) — degrading to zero-shot", exc)
        return RetrievalBundle()
    return RetrievalBundle(success=s_hits, failure=f_hits)


# --------------------------------------------------------------------------- #
# Markdown rendering — used by the Coder agent
# --------------------------------------------------------------------------- #


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _format_source(provenance: dict[str, Any] | Any, context: dict[str, Any] | Any) -> str:
    """Build a short ``paper:section / scene_id`` tag for one record."""
    if hasattr(context, "model_dump"):
        context = context.model_dump()
    if hasattr(provenance, "model_dump"):
        provenance = provenance.model_dump()
    paper = (context or {}).get("source_paper", "") or "<unknown source>"
    section = (context or {}).get("source_section", "")
    scene_id = (provenance or {}).get("scene_id", "")
    pieces = [paper]
    if section:
        pieces.append(section)
    if scene_id:
        pieces.append(scene_id)
    return " / ".join(pieces)


def render_reference_examples_block(
    records: Iterable[RetrievedRecord | dict[str, Any]],
) -> str:
    """Format success records into the *Reference Examples* prompt section.

    Returns an empty string when there are no records, so the caller can
    safely concat it unconditionally.
    """
    items = list(records)
    if not items:
        return ""
    lines: list[str] = []
    lines.append(
        "## Reference Examples (past successful scenes; treat as guidance, not literal copy)\n"
    )
    for i, item in enumerate(items, start=1):
        rec = _unwrap(item)
        body = rec["body"]
        source = _format_source(rec.get("provenance"), rec.get("context"))
        sim = float(rec.get("similarity", 0.0))
        rationale = _truncate(str(body.get("rationale", "")), _MAX_RATIONALE_CHARS)
        code = _truncate(str(body.get("code_full", "")), _MAX_CODE_FULL_CHARS)
        lines.append(f"\n### Example {i} — similarity {sim:.2f}, source: {source}\n")
        if rationale:
            lines.append(f"Rationale: {rationale}\n")
        if code:
            lines.append(f"\n```python\n{code}\n```\n")
    return "".join(lines)


def render_known_pitfalls_block(
    records: Iterable[RetrievedRecord | dict[str, Any]],
) -> str:
    """Format failure records into the *Known Pitfalls* prompt section.

    These are hard constraints: the LLM should treat the anti-examples as
    things to avoid. Each pitfall keeps its anti / good fragments separate so
    the model has a concrete repair pattern to copy.
    """
    items = list(records)
    if not items:
        return ""
    lines: list[str] = []
    lines.append(
        "## Known Pitfalls (validated failure→success transitions; AVOID the anti-pattern)\n"
    )
    for i, item in enumerate(items, start=1):
        rec = _unwrap(item)
        body = rec["body"]
        source = _format_source(rec.get("provenance"), rec.get("context"))
        sim = float(rec.get("similarity", 0.0))
        trigger = _truncate(str(body.get("trigger_pattern", "")), 300)
        cause = _truncate(str(body.get("root_cause", "")), 300)
        fix = _truncate(str(body.get("fix_recipe", "")), 300)
        anti = _truncate(str(body.get("code_anti_example", "")), _MAX_CODE_FRAGMENT_CHARS)
        good = _truncate(str(body.get("code_good_example", "")), _MAX_CODE_FRAGMENT_CHARS)
        lines.append(f"\n### Pitfall {i} — similarity {sim:.2f}, source: {source}\n")
        lines.append(f"- Trigger: {trigger}\n")
        lines.append(f"- Root cause: {cause}\n")
        lines.append(f"- Fix: {fix}\n")
        if anti:
            lines.append(f"\nAnti-example (do NOT do this):\n```python\n{anti}\n```\n")
        if good:
            lines.append(f"\nGood example (do this instead):\n```python\n{good}\n```\n")
    return "".join(lines)


def _unwrap(item: RetrievedRecord | dict[str, Any]) -> dict[str, Any]:
    """Convert either a live RetrievedRecord or a wire-format dict into a dict.

    Lets the Coder prompt builder work with whichever form is in PaperState
    without branching at every call site.
    """
    if isinstance(item, RetrievedRecord):
        return _record_to_state_dict(item)
    if isinstance(item, dict):
        # Already wire-format. Defensive copy so callers can mutate freely.
        return {
            "id": item.get("id", ""),
            "polarity": item.get("polarity", ""),
            "similarity": float(item.get("similarity", 0.0)),
            "context": dict(item.get("context") or {}),
            "body": dict(item.get("body") or {}),
            "provenance": dict(item.get("provenance") or {}),
        }
    raise TypeError(f"unexpected retrieved item type: {type(item).__name__}")


# --------------------------------------------------------------------------- #
# Pretty-printer for the trace (used by emb_retrieve_node when logging)
# --------------------------------------------------------------------------- #


def summarize_bundle(bundle: RetrievalBundle | dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Compact summary that's safe to log without leaking full code bodies."""
    if isinstance(bundle, RetrievalBundle):
        bundle_dict = bundle.to_state_dict()
    else:
        bundle_dict = bundle
    out: dict[str, Any] = {}
    for polarity in ("success", "failure"):
        recs = bundle_dict.get(polarity, [])
        out[polarity] = [
            {"id": r.get("id", "")[:8], "similarity": round(r.get("similarity", 0.0), 3)}
            for r in recs
        ]
    return out


__all__ = [
    "RetrievalBundle",
    "render_known_pitfalls_block",
    "render_reference_examples_block",
    "retrieve_for_scene",
    "summarize_bundle",
]
