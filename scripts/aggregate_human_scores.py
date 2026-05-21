"""Aggregate blind human-score JSONL annotations.

Rows may include a direct ``condition`` field or a hidden condition object
exported after annotation collection. The script reports the metrics used by
the EMNLP experiment: Human Pass@1, Human Quality, per-dimension means, fatal
flag frequencies, and binary inter-rater agreement.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DIMENSIONS = (
    "paper_alignment",
    "key_claim_coverage",
    "visual_robustness",
    "animation_flow",
    "first_attempt_usability",
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _condition(row: dict[str, Any]) -> str:
    if row.get("condition"):
        return str(row["condition"])
    hidden = row.get("condition_hidden") or {}
    return str(hidden.get("system") or "UNKNOWN")


def _fleiss_kappa_binary(groups: dict[str, list[bool]]) -> float | None:
    usable = [values for values in groups.values() if len(values) >= 2]
    if not usable:
        return None
    rater_counts = {len(values) for values in usable}
    if len(rater_counts) != 1:
        return None
    n = rater_counts.pop()
    category_totals = Counter()
    agreement_per_item = []
    for values in usable:
        counts = Counter(bool(value) for value in values)
        category_totals.update(counts)
        agreement_per_item.append((sum(count * count for count in counts.values()) - n) / (n * (n - 1)))
    p_bar = statistics.mean(agreement_per_item)
    total = len(usable) * n
    p_yes = category_totals[True] / total
    p_no = category_totals[False] / total
    p_expected = p_yes * p_yes + p_no * p_no
    if p_expected == 1:
        return 1.0
    return (p_bar - p_expected) / (1 - p_expected)


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pass_by_output: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        by_condition[_condition(row)].append(row)
        pass_by_output[str(row["blind_output_id"])].append(bool(row.get("human_pass_at_1")))

    results = {}
    for condition, condition_rows in sorted(by_condition.items()):
        pass_values = [bool(row.get("human_pass_at_1")) for row in condition_rows]
        per_dimension = {
            dim: statistics.mean(float((row.get("scores") or {}).get(dim, 0)) for row in condition_rows)
            for dim in DIMENSIONS
        }
        quality_values = [
            statistics.mean(float((row.get("scores") or {}).get(dim, 0)) for dim in DIMENSIONS)
            for row in condition_rows
        ]
        fatal_counts = Counter()
        for row in condition_rows:
            fatal_counts.update(flag for flag, active in (row.get("fatal_flags") or {}).items() if active)
        results[condition] = {
            "annotations": len(condition_rows),
            "videos": len({row["blind_output_id"] for row in condition_rows}),
            "human_pass_at_1": statistics.mean(pass_values) if pass_values else None,
            "human_quality_score": statistics.mean(quality_values) if quality_values else None,
            "per_dimension_means": per_dimension,
            "fatal_flag_frequencies": {
                flag: count / len(condition_rows) for flag, count in sorted(fatal_counts.items())
            },
        }

    return {
        "conditions": results,
        "inter_rater_agreement": {
            "human_pass_at_1_fleiss_kappa": _fleiss_kappa_binary(pass_by_output),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("human_scores_jsonl", type=Path)
    args = parser.parse_args()
    print(json.dumps(aggregate(_load_jsonl(args.human_scores_jsonl)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
