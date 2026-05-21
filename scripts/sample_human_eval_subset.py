"""Sample fixed-probe tasks for human scoring after target audit.

By default, only tasks with ``target_annotation_status == human_audited`` are
eligible. Use ``--pending-audit`` to inspect the provisional stratified sample
that should be audited next.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _stratum(task: dict[str, Any]) -> tuple[str, str, str, str]:
    strata = task.get("human_eval_strata") or {}
    return (
        str(strata.get("domain") or task.get("domain")),
        str(strata.get("scene_role") or task.get("scene_role")),
        str(strata.get("category") or task.get("category")),
        str(strata.get("difficulty") or task.get("difficulty")),
    )


def sample(tasks: list[dict[str, Any]], *, count: int, seed: int, pending_audit: bool) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    candidates = []
    for task in tasks:
        if task.get("split") != "fixed_probe":
            continue
        if (task.get("eval_only") or {}).get("hydration_status") != "ok":
            continue
        if pending_audit:
            if not task.get("human_eval_candidate_pending_audit"):
                continue
        elif not (task.get("human_eval_candidate") and task.get("target_annotation_status") == "human_audited"):
            continue
        candidates.append(task)

    by_stratum: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for task in candidates:
        by_stratum[_stratum(task)].append(task)
    for rows in by_stratum.values():
        rng.shuffle(rows)

    selected: list[dict[str, Any]] = []
    while len(selected) < count and by_stratum:
        for key in sorted(list(by_stratum)):
            rows = by_stratum[key]
            if rows:
                selected.append(rows.pop())
                if len(selected) == count:
                    break
            if not rows:
                del by_stratum[key]
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_index", type=Path)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--pending-audit", action="store_true")
    args = parser.parse_args()

    index = json.loads(args.dataset_index.read_text(encoding="utf-8"))
    tasks = _load_jsonl(args.dataset_index.parent / index.get("tasks_path", "tasks.jsonl"))
    count = int((index.get("human_scoring_plan") or {}).get("task_sample_count", 25))
    for task in sample(tasks, count=count, seed=args.seed, pending_audit=args.pending_audit):
        print(json.dumps({"task_id": task["task_id"], "human_eval_strata": task["human_eval_strata"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
