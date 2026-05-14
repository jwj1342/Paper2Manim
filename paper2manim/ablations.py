"""Experiment configuration presets for proposal §5 RQ1 + §8.3 ablations.

These named bundles map to the exact ``paper2manim mvp2`` CLI flags each
config needs. They live here (rather than in ``scripts/run_experiment.py``)
so the same dictionary is callable from tests, notebooks, and any future
batch runner without depending on the script module.

**Naming convention**:

* Single capital letters ``A``, ``B``, ``C`` are the three head-to-head
  configurations from the proposal §5 RQ1 main experiment.
* ``C_no_*`` variants are §8.3 Ablation E (channel-isolation) — they need
  flags introduced by branch B6 (``--emb-no-success-channel`` /
  ``--emb-no-failure-channel`` / ``--emb-readonly``); attempting to use
  them before B6 lands will surface as ``UsageError`` from click, which is
  the desired loud failure.
* ``C_bootstrap_<N>`` variants are §8.3 Ablation C (bootstrap batch size).
  These are *the same as* ``C`` — the bootstrap-batch dimension is set by
  the experiment runner, not by per-run flags — so they're listed here
  only for naming symmetry and resolve to the same flag list.

When adding a new preset, prefer composing existing flag fragments via the
``_BASE_*`` constants below so the diff stays small and the per-config
delta stays obvious.
"""

from __future__ import annotations

# Common flag bundles. Keep these short and orthogonal so presets are
# readable as set-arithmetic.
_BASE_NO_VLM = ("--no-vlm",)
_BASE_VLM = ("--vlm",)
_BASE_NO_EMB = ("--no-emb",)
_BASE_EMB = ("--emb",)


# Each preset is a tuple of CLI flags appended to ``mvp2 --arxiv ID`` (or
# ``mvp2 --pdf PATH``). Common per-experiment knobs (--quality / --max-retries
# / --max-visual-revisions / --emb-store-path / --dataset-domain) are filled
# in by ``scripts/run_experiment.py`` from its own argparse.
ABLATION_PRESETS: dict[str, tuple[str, ...]] = {
    # ---- §5 RQ1 main ----
    "A": (*_BASE_NO_VLM, *_BASE_NO_EMB),  # zero-shot baseline
    "B": (*_BASE_VLM, *_BASE_NO_EMB),  # reflection-only (VLM judge but no EMB)
    "C": (*_BASE_VLM, *_BASE_EMB),  # full proposal — reflection + EMB
    # ---- §8.3 Ablation B (VLM signal value) — same as A vs C, named for clarity ----
    "C_no_vlm": (*_BASE_NO_VLM, *_BASE_EMB),  # EMB enabled but VLM off — degenerate; see B6
    # ---- §8.3 Ablation E (channel isolation) — needs B6 flags ----
    "C_no_success_channel": (*_BASE_VLM, *_BASE_EMB, "--emb-no-success-channel"),
    "C_no_failure_channel": (*_BASE_VLM, *_BASE_EMB, "--emb-no-failure-channel"),
    # ---- §8.3 Ablation C (bootstrap-size sensitivity) — alias for C ----
    "C_bootstrap_0": (*_BASE_VLM, *_BASE_EMB),
    "C_bootstrap_10": (*_BASE_VLM, *_BASE_EMB),
    "C_bootstrap_50": (*_BASE_VLM, *_BASE_EMB),
    "C_bootstrap_100": (*_BASE_VLM, *_BASE_EMB),
}


def resolve(preset_name: str) -> tuple[str, ...]:
    """Return the flag tuple for ``preset_name`` or raise KeyError with a
    helpful message listing all known presets."""
    if preset_name not in ABLATION_PRESETS:
        raise KeyError(
            f"Unknown ablation preset {preset_name!r}. "
            f"Known: {sorted(ABLATION_PRESETS.keys())}. "
            f"Add new presets in paper2manim/ablations.py."
        )
    return ABLATION_PRESETS[preset_name]


def known_presets() -> list[str]:
    return sorted(ABLATION_PRESETS.keys())


__all__ = ["ABLATION_PRESETS", "resolve", "known_presets"]
