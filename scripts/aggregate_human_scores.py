"""Aggregate blind human-score JSONL annotations.

Rows may include a direct ``condition`` field or a hidden condition object
exported after annotation collection. The script reports the metrics used by
the EMNLP experiment: Human Pass@1, Human Quality, per-dimension means, fatal
flag frequencies, and binary inter-rater agreement.
"""

from __future__ import annotations

import argparse
import json
import math
import random
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


def _rank(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][1] == ordered[i][1]:
            j += 1
        rank = (i + j + 1) / 2
        for original_idx, _ in ordered[i:j]:
            ranks[original_idx] = rank
        i = j
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    return _pearson(_rank(xs), _rank(ys))


def _cohen_kappa_binary(pairs: list[tuple[bool, bool]]) -> float | None:
    if not pairs:
        return None
    observed = sum(a == b for a, b in pairs) / len(pairs)
    p_a_yes = sum(a for a, _ in pairs) / len(pairs)
    p_b_yes = sum(b for _, b in pairs) / len(pairs)
    expected = p_a_yes * p_b_yes + (1 - p_a_yes) * (1 - p_b_yes)
    if expected == 1:
        return 1.0
    return (observed - expected) / (1 - expected)


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


def _krippendorff_alpha_ordinal(groups: dict[str, list[int]]) -> float | None:
    usable = {item: values for item, values in groups.items() if len(values) >= 2}
    if not usable:
        return None
    categories = sorted({value for values in usable.values() for value in values})
    if len(categories) < 2:
        return 1.0
    distance = {(a, b): (a - b) ** 2 for a in categories for b in categories}
    observed_num = 0.0
    observed_den = 0
    pooled: list[int] = []
    for values in usable.values():
        pooled.extend(values)
        for i, a in enumerate(values):
            for j, b in enumerate(values):
                if i == j:
                    continue
                observed_num += distance[(a, b)]
                observed_den += 1
    if observed_den == 0 or len(pooled) < 2:
        return None
    expected_num = 0.0
    expected_den = 0
    for i, a in enumerate(pooled):
        for j, b in enumerate(pooled):
            if i == j:
                continue
            expected_num += distance[(a, b)]
            expected_den += 1
    if expected_den == 0:
        return None
    observed = observed_num / observed_den
    expected = expected_num / expected_den
    if expected == 0:
        return 1.0
    return 1 - observed / expected


def _quality(row: dict[str, Any]) -> float:
    return statistics.mean(float((row.get("scores") or {}).get(dim, 0)) for dim in DIMENSIONS)


def _vlm_quality(row: dict[str, Any]) -> float | None:
    scores = row.get("vlm_scores") or row.get("vlm_score")
    if isinstance(scores, dict):
        values = [
            float(value)
            for key, value in scores.items()
            if not isinstance(value, bool) and isinstance(value, int | float) and "pass" not in str(key).lower()
        ]
        return statistics.mean(values) if values else None
    if isinstance(scores, int | float):
        return float(scores)
    return None


def _vlm_pass(row: dict[str, Any]) -> bool | None:
    for key in ("vlm_pass_at_1", "vlm_pass", "vlm_verdict_pass"):
        if key in row:
            return bool(row[key])
    scores = row.get("vlm_scores")
    if isinstance(scores, dict) and "pass_at_1" in scores:
        return bool(scores["pass_at_1"])
    return None


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def _cluster_bootstrap_ci(
    rows: list[dict[str, Any]],
    *,
    cluster_key: str,
    metric,
    iterations: int,
    seed: int,
) -> dict[str, float | None]:
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        clusters[str(row.get(cluster_key) or row.get("task_id") or row.get("blind_output_id"))].append(row)
    keys = sorted(clusters)
    if len(keys) < 2:
        point = metric(rows) if rows else None
        return {"mean": point, "ci95_low": None, "ci95_high": None, "clusters": len(keys)}
    rng = random.Random(seed)
    samples = []
    for _ in range(iterations):
        sampled_rows: list[dict[str, Any]] = []
        for key in (rng.choice(keys) for _ in keys):
            sampled_rows.extend(clusters[key])
        samples.append(metric(sampled_rows))
    return {
        "mean": metric(rows),
        "ci95_low": _percentile(samples, 0.025),
        "ci95_high": _percentile(samples, 0.975),
        "clusters": len(keys),
    }


def _task_paper_map(dataset_index: Path | None) -> dict[str, str]:
    if dataset_index is None:
        return {}
    index = json.loads(dataset_index.read_text(encoding="utf-8"))
    root = dataset_index.parent
    if index.get("single_json_path"):
        payload = json.loads((root / index["single_json_path"]).read_text(encoding="utf-8"))
        rows = list(payload.get("tasks") or [])
        rows.extend(payload.get("holdout_tasks") or [])
    else:
        rows = _load_jsonl(root / index.get("tasks_path", "tasks.jsonl"))
        holdout = index.get("tasks_holdout_path")
        if holdout:
            rows.extend(_load_jsonl(root / holdout))
    return {str(row["task_id"]): str(row["paper_id"]) for row in rows}


def aggregate(
    rows: list[dict[str, Any]],
    *,
    task_to_paper: dict[str, str] | None = None,
    bootstrap_iterations: int = 2000,
    bootstrap_seed: int = 1,
) -> dict[str, Any]:
    task_to_paper = task_to_paper or {}
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pass_by_output: dict[str, list[bool]] = defaultdict(list)
    ordinal_by_output = {dim: defaultdict(list) for dim in DIMENSIONS}
    for row in rows:
        row.setdefault("paper_id", task_to_paper.get(str(row.get("task_id")), row.get("paper_id")))
        by_condition[_condition(row)].append(row)
        pass_by_output[str(row["blind_output_id"])].append(bool(row.get("human_pass_at_1")))
        for dim in DIMENSIONS:
            if dim in (row.get("scores") or {}):
                ordinal_by_output[dim][str(row["blind_output_id"])].append(int((row.get("scores") or {})[dim]))

    results = {}
    for condition, condition_rows in sorted(by_condition.items()):
        pass_values = [bool(row.get("human_pass_at_1")) for row in condition_rows]
        per_dimension = {
            dim: statistics.mean(float((row.get("scores") or {}).get(dim, 0)) for row in condition_rows)
            for dim in DIMENSIONS
        }
        quality_values = [_quality(row) for row in condition_rows]
        fatal_counts = Counter()
        for row in condition_rows:
            fatal_counts.update(flag for flag, active in (row.get("fatal_flags") or {}).items() if active)
        results[condition] = {
            "annotations": len(condition_rows),
            "videos": len({row["blind_output_id"] for row in condition_rows}),
            "human_pass_at_1": statistics.mean(pass_values) if pass_values else None,
            "human_quality_score": statistics.mean(quality_values) if quality_values else None,
            "human_pass_at_1_cluster_bootstrap_ci": _cluster_bootstrap_ci(
                condition_rows,
                cluster_key="paper_id",
                metric=lambda sample: statistics.mean(bool(row.get("human_pass_at_1")) for row in sample),
                iterations=bootstrap_iterations,
                seed=bootstrap_seed,
            ),
            "human_quality_score_cluster_bootstrap_ci": _cluster_bootstrap_ci(
                condition_rows,
                cluster_key="paper_id",
                metric=lambda sample: statistics.mean(_quality(row) for row in sample),
                iterations=bootstrap_iterations,
                seed=bootstrap_seed,
            ),
            "per_dimension_means": per_dimension,
            "fatal_flag_frequencies": {
                flag: count / len(condition_rows) for flag, count in sorted(fatal_counts.items())
            },
        }

    human_by_output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    vlm_by_output: dict[str, dict[str, Any]] = {}
    for row in rows:
        output_id = str(row["blind_output_id"])
        human_by_output[output_id].append(row)
        if _vlm_quality(row) is not None or _vlm_pass(row) is not None:
            vlm_by_output.setdefault(output_id, row)
    quality_pairs = []
    pass_pairs = []
    for output_id, human_rows in human_by_output.items():
        vlm_row = vlm_by_output.get(output_id)
        if not vlm_row:
            continue
        vlm_quality = _vlm_quality(vlm_row)
        if vlm_quality is not None:
            quality_pairs.append((vlm_quality, statistics.mean(_quality(row) for row in human_rows)))
        vlm_pass = _vlm_pass(vlm_row)
        if vlm_pass is not None:
            human_majority = statistics.mean(bool(row.get("human_pass_at_1")) for row in human_rows) >= 0.5
            pass_pairs.append((vlm_pass, human_majority))

    return {
        "conditions": results,
        "inter_rater_agreement": {
            "human_pass_at_1_fleiss_kappa": _fleiss_kappa_binary(pass_by_output),
            "ordinal_krippendorff_alpha": {
                dim: _krippendorff_alpha_ordinal(groups) for dim, groups in ordinal_by_output.items()
            },
        },
        "vlm_human_agreement": {
            "quality_pearson_r": _pearson([x for x, _ in quality_pairs], [y for _, y in quality_pairs]),
            "quality_spearman_rho": _spearman([x for x, _ in quality_pairs], [y for _, y in quality_pairs]),
            "pass_at_1_cohen_kappa": _cohen_kappa_binary(pass_pairs),
            "paired_outputs": max(len(quality_pairs), len(pass_pairs)),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("human_scores_jsonl", type=Path)
    parser.add_argument("--dataset-index", type=Path)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=1)
    args = parser.parse_args()
    print(
        json.dumps(
            aggregate(
                _load_jsonl(args.human_scores_jsonl),
                task_to_paper=_task_paper_map(args.dataset_index),
                bootstrap_iterations=args.bootstrap_iterations,
                bootstrap_seed=args.bootstrap_seed,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
