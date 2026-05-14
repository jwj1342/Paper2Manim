"""Distillation pipeline — pure logic that turns a finished run into EMB records.

The pipeline has two layers:

1. **Trace parsing** (this module) — walks ``runs/<run_id>/trace.jsonl`` plus the
   ``attempts/`` and ``vlm_frames/`` directories to extract every render +
   VLM review event into typed dataclasses. No LLM calls.

2. **Record distillation** — applies the §4.4 quality gates (success ≥ θ_high,
   failure transitions must satisfy ``after_score > before_score``) and calls
   the injected ``rationale_writer`` / ``lesson_distiller`` to produce the
   final ``MemoryRecord`` bodies.

The two layers are split so we can unit-test the parser with synthetic
trace.jsonl files, and unit-test the distillers with mock writers — without
touching disk in either case beyond ``tmp_path``.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from paper2manim.artifacts import run_dir
from paper2manim.emb.manager import EpisodicMemoryBank
from paper2manim.emb.schema import (
    Context,
    FailureBody,
    MemoryRecord,
    Provenance,
    SuccessBody,
)

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Trace event dataclasses
# --------------------------------------------------------------------------- #


@dataclass
class RenderEvent:
    """One render call. ``v_rev == 0`` means a first-pass / text-reflection
    render; ``v_rev >= 1`` means a visual revision of the same scene."""

    scene: str
    iter_idx: int
    v_rev: int
    status: str  # "success" | "error"
    category: str | None  # python / latex / manim_runtime / timeout / unknown / None
    raw: dict[str, Any]


@dataclass
class VLMReviewEvent:
    scene: str
    v_rev: int
    decision: str  # pass | revise | fail
    scores: dict[str, int]
    avg_score: float
    raw: dict[str, Any]


@dataclass
class SceneEvents:
    """Everything we know about one scene after the run finished."""

    name: str
    renders: list[RenderEvent] = field(default_factory=list)
    vlm_reviews: list[VLMReviewEvent] = field(default_factory=list)


@dataclass
class ScoredScene:
    """A scene that reached the end of the per-scene loop successfully."""

    name: str
    final_v_rev: int
    final_score: float  # avg across scoring dims; 0.0 if VLM was off
    final_code: str
    final_montage_path: Path | None
    final_video_path: Path | None
    had_vlm_review: bool


@dataclass
class TextTransition:
    """An ``error → success`` transition produced by text reflection.

    Always validated by construction: we only emit a TextTransition when the
    next-iter render flipped from error to success.
    """

    scene: str
    before_iter: int
    after_iter: int
    before_code: str
    after_code: str
    error_category: str | None
    error_message: str
    traceback_tail: str


@dataclass
class VisualTransition:
    """A VLM ``low-score → high-score`` transition.

    Only emitted when ``after_score > before_score`` strictly. The graph also
    sometimes produces ``v -> v+1`` pairs where the second is *worse*; those
    do not pass the §4.4b validation and are filtered here.
    """

    scene: str
    before_v_rev: int
    after_v_rev: int
    before_code: str
    after_code: str
    before_score: float
    after_score: float
    revision_instruction: str


# --------------------------------------------------------------------------- #
# Trace parser
# --------------------------------------------------------------------------- #


# Proposal §4.2 canonical 3-dim 0-100 schema, mirrored from
# paper2manim/agents/vlm_scene_reviewer.py:_SCORE_KEYS. Keep these two in sync —
# if the reviewer's schema drifts (back to 6-dim 1-5, etc.) the consolidate
# pipeline silently writes zero records, which is exactly the bug fixed here.
_SCORE_KEYS = (
    "logic_flow",
    "layout_occlusion",
    "accuracy",
)


def _avg_score(scores: dict[str, int | None] | None) -> float:
    """Mean of the three canonical dimensions on a 0–100 scale.

    Skips ``None`` entries so a partial review (one dim missing) still produces
    a defensible average over the present dims, matching how
    :func:`paper2manim.agents.vlm_scene_reviewer.parse_vlm_response` treats
    missing dimensions. Returns 0.0 only when *no* dim has a numeric value —
    that case is what trips the consolidation gate, which is the intended
    behavior for unparseable VLM output.
    """
    if not scores:
        return 0.0
    nums: list[float] = []
    for k, v in scores.items():
        if k not in _SCORE_KEYS or v is None:
            continue
        try:
            nums.append(float(v))
        except (TypeError, ValueError):
            continue
    return sum(nums) / len(nums) if nums else 0.0


def _read_trace(run_id: str) -> list[dict[str, Any]]:
    """Yield every JSONL line from ``runs/<run_id>/trace.jsonl``."""
    trace_path = run_dir(run_id) / "trace.jsonl"
    if not trace_path.exists():
        return []
    out: list[dict[str, Any]] = []
    with trace_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                log.warning("[distill] skipping malformed trace line: %s", exc)
    return out


def _read_attempt_code(run_id: str, scene: str, iter_idx: int, v_rev: int) -> str:
    """Load the saved code for one (scene, iter, v_rev) tuple, or '' if missing.

    Layout convention (mirrors ``agents.coder.save_attempt_code`` /
    ``visual_revise_node``):

    * v_rev == 0:        ``attempts/{iter:02d}_{scene}.py``
    * v_rev >= 1:        ``attempts/{iter:02d}_{scene}_v{v_rev}.py``
    """
    name = scene if v_rev == 0 else f"{scene}_v{v_rev}"
    path = run_dir(run_id) / "attempts" / f"{iter_idx:02d}_{name}.py"
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        log.warning("[distill] failed to read %s: %s", path, exc)
        return ""


def _read_attempt_render_result(
    run_id: str, scene: str, iter_idx: int
) -> dict[str, Any] | None:
    """Load the ``.render.json`` sidecar for one (scene, iter) tuple.

    Note: ``save_attempt_result`` only writes one file per (scene, iter); v_rev
    values share that file, so for visual revisions the JSON reflects the
    *latest* render of that text iter. Good enough for traceback excerpts on
    text transitions; visual transitions use VLM scores instead.
    """
    path = run_dir(run_id) / "attempts" / f"{iter_idx:02d}_{scene}.render.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("[distill] failed to parse %s: %s", path, exc)
        return None


def _read_montage_path(run_id: str, scene: str, v_rev: int) -> Path | None:
    p = run_dir(run_id) / "vlm_frames" / f"{scene}_v{v_rev}.png"
    return p if p.exists() else None


def parse_trace(run_id: str) -> dict[str, SceneEvents]:
    """Group the trace.jsonl events by scene name."""
    events = _read_trace(run_id)
    by_scene: dict[str, SceneEvents] = {}
    for ev in events:
        node = ev.get("node")
        scene = ev.get("scene")
        if not scene:
            continue
        bucket = by_scene.setdefault(scene, SceneEvents(name=scene))
        if node == "render":
            bucket.renders.append(
                RenderEvent(
                    scene=scene,
                    iter_idx=int(ev.get("iter", 0)),
                    v_rev=int(ev.get("v_rev", 0)),
                    status=str(ev.get("status", "")),
                    category=ev.get("category"),
                    raw=ev,
                )
            )
        elif node == "vlm_review":
            scores = ev.get("scores") or {}
            bucket.vlm_reviews.append(
                VLMReviewEvent(
                    scene=scene,
                    v_rev=int(ev.get("v_rev", 0)),
                    decision=str(ev.get("decision") or ""),
                    scores=scores,
                    avg_score=_avg_score(scores),
                    raw=ev,
                )
            )
    return by_scene


# --------------------------------------------------------------------------- #
# Transition finders
# --------------------------------------------------------------------------- #


def find_text_transitions(
    run_id: str, scenes: Iterable[SceneEvents]
) -> list[TextTransition]:
    """Pairs of (render_error iter=k) → (render_success iter=k+1) within a scene.

    Only emits when the success render is also at ``v_rev == 0`` — text
    reflection always happens before any visual revision in our graph.
    """
    out: list[TextTransition] = []
    for sc in scenes:
        # Iter-ordered renders at v_rev 0 only (text reflection lane).
        v0 = sorted(
            [r for r in sc.renders if r.v_rev == 0],
            key=lambda r: r.iter_idx,
        )
        for prev, nxt in zip(v0, v0[1:], strict=False):
            if prev.status != "error" or nxt.status != "success":
                continue
            before_code = _read_attempt_code(run_id, sc.name, prev.iter_idx, 0)
            after_code = _read_attempt_code(run_id, sc.name, nxt.iter_idx, 0)
            if not before_code or not after_code:
                continue
            rr = _read_attempt_render_result(run_id, sc.name, prev.iter_idx) or {}
            out.append(
                TextTransition(
                    scene=sc.name,
                    before_iter=prev.iter_idx,
                    after_iter=nxt.iter_idx,
                    before_code=before_code,
                    after_code=after_code,
                    error_category=prev.category,
                    error_message=str(rr.get("error_message") or "")[:1000],
                    traceback_tail=str(rr.get("traceback_tail") or "")[:2000],
                )
            )
    return out


def find_visual_transitions(
    run_id: str,
    scenes: Iterable[SceneEvents],
    *,
    final_text_iter_by_scene: dict[str, int] | None = None,
    min_margin: float = 5.0,
) -> list[VisualTransition]:
    """VLM (v=k, score=s_k) → (v=k+1, score=s_{k+1}) where s_{k+1} − s_k ≥ ``min_margin``.

    ``min_margin`` (default 5.0 on the 0–100 avg scale, mirroring the proposal
    §4.2 schema) filters out marginal score wiggles that don't represent a real
    learnable transition. Set to a tiny epsilon (e.g. 0.5) to keep every
    strictly-improved pair.

    ``final_text_iter_by_scene`` maps each scene name to the iter_idx that the
    text reflection settled on — that's the iter under which all v_rev files
    were saved (see ``visual_revise_node``). If absent we infer by taking the
    largest iter with a successful v_rev=0 render.
    """
    if final_text_iter_by_scene is None:
        final_text_iter_by_scene = {}
        for sc in scenes:
            for r in sorted(sc.renders, key=lambda x: x.iter_idx):
                if r.v_rev == 0 and r.status == "success":
                    final_text_iter_by_scene[sc.name] = r.iter_idx
                    break
    out: list[VisualTransition] = []
    for sc in scenes:
        iter_for_scene = final_text_iter_by_scene.get(sc.name)
        if iter_for_scene is None:
            continue
        reviews = sorted(sc.vlm_reviews, key=lambda r: r.v_rev)
        for prev, nxt in zip(reviews, reviews[1:], strict=False):
            if nxt.v_rev != prev.v_rev + 1:
                continue
            if (nxt.avg_score - prev.avg_score) < min_margin:
                continue
            before_code = _read_attempt_code(run_id, sc.name, iter_for_scene, prev.v_rev)
            after_code = _read_attempt_code(run_id, sc.name, iter_for_scene, nxt.v_rev)
            if not before_code or not after_code:
                continue
            instr = str(prev.raw.get("revision_instruction") or "")
            out.append(
                VisualTransition(
                    scene=sc.name,
                    before_v_rev=prev.v_rev,
                    after_v_rev=nxt.v_rev,
                    before_code=before_code,
                    after_code=after_code,
                    before_score=prev.avg_score,
                    after_score=nxt.avg_score,
                    revision_instruction=instr,
                )
            )
    return out


def find_scored_scenes(
    run_id: str, scenes: Iterable[SceneEvents]
) -> list[ScoredScene]:
    """One ScoredScene per scene that ended with a successful render.

    Picks the highest-scoring VLM review (when present) as the final state.
    When VLM was off, the ``final_score`` is 0.0 — callers must opt in to
    accepting these via ``theta_high <= 0.0``.
    """
    out: list[ScoredScene] = []
    for sc in scenes:
        # Find the latest text-iter where v_rev=0 render succeeded.
        success_v0 = [r for r in sc.renders if r.v_rev == 0 and r.status == "success"]
        if not success_v0:
            continue
        text_iter = max(r.iter_idx for r in success_v0)
        # Pick best VLM review (highest avg). If none, fall back to v_rev=0 code.
        if sc.vlm_reviews:
            best = max(sc.vlm_reviews, key=lambda r: r.avg_score)
            final_v_rev = best.v_rev
            final_score = best.avg_score
            had_vlm = True
        else:
            final_v_rev = 0
            final_score = 0.0
            had_vlm = False
        final_code = _read_attempt_code(run_id, sc.name, text_iter, final_v_rev)
        if not final_code:
            # Fallback to v_rev=0 if the visual-revision file is gone.
            final_code = _read_attempt_code(run_id, sc.name, text_iter, 0)
        final_montage = _read_montage_path(run_id, sc.name, final_v_rev)
        rr = _read_attempt_render_result(run_id, sc.name, text_iter) or {}
        video = rr.get("video_path")
        out.append(
            ScoredScene(
                name=sc.name,
                final_v_rev=final_v_rev,
                final_score=final_score,
                final_code=final_code,
                final_montage_path=final_montage,
                final_video_path=Path(video) if video else None,
                had_vlm_review=had_vlm,
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Distillation — Memory record builders
# --------------------------------------------------------------------------- #


RationaleWriter = Callable[[ScoredScene, str], str]
"""Args: (scored_scene, scene_description). Returns the natural-language rationale."""

LessonDistiller = Callable[[VisualTransition | TextTransition, str], FailureBody]
"""Args: (transition, scene_description). Returns a FailureBody."""


def default_rationale_writer(scored: ScoredScene, scene_description: str) -> str:
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


def default_lesson_distiller(
    transition: VisualTransition | TextTransition, scene_description: str
) -> FailureBody:
    """Fallback lesson body — minimal but valid, no LLM needed.

    Pulls anti / good examples by extracting the first significantly differing
    lines from before / after code. Good enough as a placeholder until the
    real ``agents.lesson_distiller`` is wired up.
    """
    if isinstance(transition, VisualTransition):
        trigger = (
            f"Visual scene with low layout/clarity score "
            f"(avg={transition.before_score:.2f})"
        )
        root_cause = (
            transition.revision_instruction.strip()[:300] or "VLM flagged for revision"
        )
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


def _scene_description_lookup(state: dict[str, Any] | None) -> dict[str, str]:
    """Build {scene_name -> description} from a PaperState-like dict."""
    if not state:
        return {}
    sb = state.get("storyboard") or {}
    out: dict[str, str] = {}
    for s in sb.get("scenes", []):
        if isinstance(s, dict) and s.get("name"):
            out[s["name"]] = str(s.get("description", ""))
    return out


def _domain_tags_from_text(text: str) -> list[str]:
    """Cheap keyword-based domain tagger — used until we wire a real classifier.

    Picks at most three tags from a small hand-curated vocabulary that maps to
    the three target domains in the proposal (CS / math / physics).
    """
    text_l = text.lower()
    candidates = [
        ("transformer", ["transformer", "attention head", "self-attention"]),
        ("attention", ["attention", "softmax(qk", "query key value"]),
        ("optimization", ["gradient descent", "loss function", "optimizer"]),
        ("graph", ["graph neural", "gnn", "node embedding"]),
        ("probability", ["probability", "distribution", "bayes"]),
        ("linear-algebra", ["matrix", "vector", "eigen", "tensor"]),
        ("quantum", ["quantum", "qubit", "hamiltonian"]),
        ("dynamics", ["dynamics", "equation of motion", "lagrangian"]),
    ]
    tags: list[str] = []
    for tag, kws in candidates:
        if any(kw in text_l for kw in kws):
            tags.append(tag)
        if len(tags) >= 3:
            break
    return tags


def _scene_role_from_name(name: str) -> str:
    """Heuristic scene_role: looks at the storyboarder's PascalCase scene name.

    The proposal §4.4 keeps ``scene_role`` coarse so retrieval can ablate by
    section type (background / method / experiment / conclusion). Storyboarder
    output tends to embed these tokens in scene names (e.g. ``TitleIntro``,
    ``ScaledDotProductAttention``, ``TakeawayConclusion``). When the name
    doesn't hint at a role we fall back to ``unknown``.
    """
    n = name.lower()
    if any(k in n for k in ("intro", "title", "background", "motivation")):
        return "background"
    if any(k in n for k in ("conclusion", "takeaway", "summary")):
        return "conclusion"
    if any(k in n for k in ("experiment", "result", "benchmark")):
        return "experiment"
    return "method"


# --------------------------------------------------------------------------- #
# Top-level distill functions
# --------------------------------------------------------------------------- #


def distill_success_records(
    run_id: str,
    *,
    state: dict[str, Any] | None = None,
    theta_high: float = 85.0,
    source_paper: str = "",
    source_section: str = "",
    rationale_writer: RationaleWriter | None = None,
    domain: str = "",
) -> list[MemoryRecord]:
    """One ``MemoryRecord`` per scene whose final avg score ≥ ``theta_high``.

    Default ``theta_high=85.0`` matches the proposal §4.2 0–100 schema; it sits
    just below the §4.3 auto-pass threshold (90) so scenes the VLM
    bypass-passes also qualify as success records. If ``theta_high <= 0``,
    scenes without VLM scoring still qualify (useful for MVP 2.0-style
    bootstrap where the VLM loop is disabled).

    ``domain`` (B6) is stamped on every Context for RQ3 cross-domain
    experiments. Empty string = "untagged".
    """
    rw = rationale_writer or default_rationale_writer
    desc = _scene_description_lookup(state)
    scenes = parse_trace(run_id)
    scored = find_scored_scenes(run_id, scenes.values())
    out: list[MemoryRecord] = []
    for sc in scored:
        if theta_high > 0 and sc.final_score < theta_high:
            continue
        if not sc.final_code:
            continue
        scene_desc = desc.get(sc.name, "")
        rationale = rw(sc, scene_desc)
        body = SuccessBody(rationale=rationale, code_full=sc.final_code)
        ctx = Context(
            task_text=scene_desc or sc.name,
            scene_role=_scene_role_from_name(sc.name),
            domain_tags=_domain_tags_from_text(scene_desc or sc.name),
            domain=domain,
            source_paper=source_paper,
            source_section=source_section,
        )
        prov = Provenance(
            run_id=run_id,
            scene_id=sc.name,
            extraction_source="high_score_scene",
            validated=True,
            vlm_score=sc.final_score if sc.had_vlm_review else None,
            # Pin the v_rev whose montage corresponds to ``vlm_score`` so
            # ``paper2manim emb retest`` can re-score the right frame instead
            # of always loading ``_v0.png``.
            final_v_rev=sc.final_v_rev,
        )
        out.append(MemoryRecord(polarity="success", context=ctx, body=body, provenance=prov))
    return out


def distill_failure_records(
    run_id: str,
    *,
    state: dict[str, Any] | None = None,
    source_paper: str = "",
    source_section: str = "",
    lesson_distiller: LessonDistiller | None = None,
    include_text_transitions: bool = True,
    include_visual_transitions: bool = True,
    failure_min_margin: float = 5.0,
    domain: str = "",
) -> list[MemoryRecord]:
    """One record per validated transition. ``after_score − before_score ≥
    failure_min_margin`` enforced inside :func:`find_visual_transitions`; text
    transitions are validated by ``error → success``. Default 5.0 is on the
    proposal §4.2 0–100 scale.

    ``domain`` (B6) stamps every Context for RQ3 cross-domain experiments.
    """
    ld = lesson_distiller or default_lesson_distiller
    desc = _scene_description_lookup(state)
    scenes = parse_trace(run_id)
    text_ts: list[TextTransition] = (
        find_text_transitions(run_id, scenes.values()) if include_text_transitions else []
    )
    visual_ts: list[VisualTransition] = (
        find_visual_transitions(
            run_id, scenes.values(), min_margin=failure_min_margin
        )
        if include_visual_transitions
        else []
    )

    out: list[MemoryRecord] = []
    for vt in visual_ts:
        scene_desc = desc.get(vt.scene, "")
        body = ld(vt, scene_desc)
        ctx = Context(
            task_text=scene_desc or vt.scene,
            scene_role=_scene_role_from_name(vt.scene),
            domain_tags=_domain_tags_from_text(scene_desc or vt.scene),
            domain=domain,
            source_paper=source_paper,
            source_section=source_section,
        )
        prov = Provenance(
            run_id=run_id,
            scene_id=vt.scene,
            extraction_source="visual_reflection",
            # before_v_rev keeps v0→v1 and v1→v2 of the same scene as separate
            # records instead of letting the second silently overwrite the first
            # under INSERT OR REPLACE.
            transition_ordinal=vt.before_v_rev,
            validated=True,
            before_score=vt.before_score,
            after_score=vt.after_score,
        )
        out.append(MemoryRecord(polarity="failure", context=ctx, body=body, provenance=prov))
    for tt in text_ts:
        scene_desc = desc.get(tt.scene, "")
        body = ld(tt, scene_desc)
        ctx = Context(
            task_text=scene_desc or tt.scene,
            scene_role=_scene_role_from_name(tt.scene),
            domain_tags=_domain_tags_from_text(scene_desc or tt.scene),
            domain=domain,
            source_paper=source_paper,
            source_section=source_section,
        )
        # before_score=0, after_score=1 stands in for "error → success" since
        # text transitions don't have continuous scores.
        prov = Provenance(
            run_id=run_id,
            scene_id=tt.scene,
            extraction_source="text_reflection",
            # after_iter keeps multiple text reflections within the same scene
            # distinct (would only happen if a scene actually produced multiple
            # error→success transitions on v_rev=0, but defensive in case).
            transition_ordinal=tt.after_iter,
            validated=True,
            before_score=0.0,
            after_score=1.0,
        )
        out.append(MemoryRecord(polarity="failure", context=ctx, body=body, provenance=prov))
    return out


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #


@dataclass
class ConsolidationReport:
    n_success_written: int = 0
    n_failure_written: int = 0
    success_ids: list[str] = field(default_factory=list)
    failure_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_success_written": self.n_success_written,
            "n_failure_written": self.n_failure_written,
            "success_ids": list(self.success_ids),
            "failure_ids": list(self.failure_ids),
        }


def consolidate_run(
    run_id: str,
    emb: EpisodicMemoryBank,
    *,
    state: dict[str, Any] | None = None,
    theta_high: float = 85.0,
    source_paper: str = "",
    source_section: str = "",
    rationale_writer: RationaleWriter | None = None,
    lesson_distiller: LessonDistiller | None = None,
    failure_min_margin: float = 5.0,
    domain: str = "",
    skip_success: bool = False,
    skip_failure: bool = False,
) -> ConsolidationReport:
    """End-to-end §4.4 sink for one paper-section run.

    Reads trace.jsonl + attempts/ from disk; calls the injected writers (which
    may hit an LLM/VLM) to materialize bodies; writes every produced record to
    ``emb``. Idempotent over repeated calls within a process **only insofar as
    new records are appended with fresh UUIDs** — there is no dedupe yet.

    ``domain`` (B6) stamps every Context for RQ3 cross-domain. ``skip_success``
    / ``skip_failure`` (B6, §8.3 Ablation E) bypass the corresponding distill
    call entirely so the channel ablation works on the WRITE side too —
    paired with the same flags in :func:`retrieve_for_scene` for the READ side.
    """
    report = ConsolidationReport()
    succ: list[MemoryRecord] = []
    fail: list[MemoryRecord] = []
    if not skip_success:
        succ = distill_success_records(
            run_id,
            state=state,
            theta_high=theta_high,
            source_paper=source_paper,
            source_section=source_section,
            rationale_writer=rationale_writer,
            domain=domain,
        )
    if not skip_failure:
        fail = distill_failure_records(
            run_id,
            state=state,
            source_paper=source_paper,
            source_section=source_section,
            lesson_distiller=lesson_distiller,
            failure_min_margin=failure_min_margin,
            domain=domain,
        )
    for rec in succ:
        try:
            rid = emb.put(rec)
            report.success_ids.append(rid)
            report.n_success_written += 1
        except Exception as exc:  # noqa: BLE001
            log.warning("[consolidate] failed to write success record: %s", exc)
    for rec in fail:
        try:
            rid = emb.put(rec)
            report.failure_ids.append(rid)
            report.n_failure_written += 1
        except Exception as exc:  # noqa: BLE001
            log.warning("[consolidate] failed to write failure record: %s", exc)
    emb.save_indices()
    log.info(
        "[consolidate] run=%s wrote %d success + %d failure records",
        run_id,
        report.n_success_written,
        report.n_failure_written,
    )
    return report


# --------------------------------------------------------------------------- #
# Inferring source_paper / source_section from PaperState
# --------------------------------------------------------------------------- #


_ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5}|[a-z\-]+/\d{7})", re.IGNORECASE)


def infer_source_metadata(state: dict[str, Any] | None) -> tuple[str, str]:
    """Guess ``(source_paper, source_section)`` from a finished PaperState.

    Returns empty strings when the input kind doesn't carry an arxiv id (e.g.
    raw text MVP 1.0 runs).
    """
    if not state:
        return "", ""
    kind = state.get("input_kind")
    if kind == "arxiv":
        spec = str(state.get("arxiv_spec") or "")
        m = _ARXIV_ID_RE.search(spec)
        paper = f"arxiv:{m.group(0)}" if m else f"arxiv:{spec}"
        section = str(state.get("arxiv_section") or "")
        return paper, section
    if kind == "pdf":
        pdf = state.get("pdf_path") or ""
        return f"pdf:{Path(str(pdf)).name}" if pdf else "", ""
    return "", ""
