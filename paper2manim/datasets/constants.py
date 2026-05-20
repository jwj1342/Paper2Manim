"""Frozen enumerations for the lightweight task dataset schema.

When growing these (e.g. adding a new domain), update ``examples/datasets/
p2m_v1_schema.md`` in the same PR so curators don't have to guess.
"""

from __future__ import annotations

# Tuples (not sets) so click.Choice keeps a stable order in --help output.
DOMAINS: tuple[str, ...] = ("cs", "math", "physics", "quantum", "econ")
CANONICAL_SPLITS: tuple[str, ...] = (
    "memory_build",
    "fixed_probe",
    "test_holdout",
    "cross_train",
    "cross_test",
)
LEGACY_SPLIT_ALIASES: dict[str, str] = {
    "bootstrap": "memory_build",
    "eval": "fixed_probe",
}
SPLITS: tuple[str, ...] = CANONICAL_SPLITS + tuple(LEGACY_SPLIT_ALIASES)
