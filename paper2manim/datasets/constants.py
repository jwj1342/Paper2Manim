"""Frozen enumerations for the v1 task dataset schema.

When growing these (e.g. adding a new domain), update ``examples/datasets/
p2m_v1_schema.md`` in the same PR so curators don't have to guess.
"""

from __future__ import annotations

# Tuples (not sets) so click.Choice keeps a stable order in --help output.
DOMAINS: tuple[str, ...] = ("cs", "math", "physics", "quantum", "econ")
SPLITS: tuple[str, ...] = ("bootstrap", "eval", "cross_train", "cross_test")
