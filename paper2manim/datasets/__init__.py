"""Constants and helpers for Paper2Manim task datasets (proposal §8.1).

Single source of truth for domain / split enums. CLI (``--dataset-domain``)
and validator (``scripts/dataset_validate.py``) both import from here so they
stay in lockstep — typos like ``--dataset-domain phyics`` would silently
write a misspelled tag into EMB records and make ``retrieve_for_scene
(domain_filter='physics')`` permanently miss them.
"""

from __future__ import annotations

from .constants import DOMAINS, SPLITS
from .csv_utils import strip_comment_lines

__all__ = ["DOMAINS", "SPLITS", "strip_comment_lines"]
