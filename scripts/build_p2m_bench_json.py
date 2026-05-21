"""Build the single-file P2M-Bench release artifact.

The source layout remains normalized for editing and validation:
``dataset_index.json`` points to JSONL tasks and per-paper metadata files.
This script packages those release-facing pieces into one JSON file while
leaving output-level human scores in their sidecar annotation files.

Usage:
    python scripts/build_p2m_bench_json.py data/p2m_bench_v2/dataset_index.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL: {exc}") from exc
    return rows


def _load_papers(path: Path) -> dict[str, dict[str, Any]]:
    papers: dict[str, dict[str, Any]] = {}
    for paper_path in sorted(path.glob("*.json")):
        paper = json.loads(paper_path.read_text(encoding="utf-8"))
        paper_id = str(paper.get("paper_id") or paper_path.stem.replace("_", "."))
        papers[paper_id] = paper
    return papers


def build_single_json(index_path: Path) -> dict[str, Any]:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    root = index_path.parent
    existing_single_json = root / index.get("single_json_path", "p2m_bench_v2.json")
    source_tasks_path = root / index.get("tasks_path", "tasks.jsonl")

    if index.get("single_json_path") and existing_single_json.exists() and not source_tasks_path.exists():
        payload = json.loads(existing_single_json.read_text(encoding="utf-8"))
        payload["metadata"] = {**payload.get("metadata", {}), **index}
        return payload

    tasks = _load_jsonl(source_tasks_path)
    holdout_path = index.get("tasks_holdout_path")
    holdout_tasks = _load_jsonl(root / holdout_path) if holdout_path else []
    papers = _load_papers(root / index.get("papers_path", "papers/"))

    return {
        "dataset": index.get("dataset", "P2M-Bench"),
        "version": index.get("version"),
        "format": "single_json",
        "description": index.get("description"),
        "metadata": index,
        "release_policy": {
            **index.get("release_policy", {}),
            "human_scores_included": False,
            "human_scores_sidecar": index.get("manifest_paths", {}).get(
                "human_scores", "data/p2m_bench_v2/annotations/human_scores.jsonl"
            ),
        },
        "tasks": tasks,
        "holdout_tasks": holdout_tasks,
        "papers": papers,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_index", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="Defaults to <dataset root>/p2m_bench_v2.json.",
    )
    args = parser.parse_args()

    payload = build_single_json(args.dataset_index)
    out_path = args.output or args.dataset_index.parent / "p2m_bench_v2.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
