"""Write concrete publication cutoff strata into P2M-Bench task files."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _stratify(rows: list[dict[str, Any]], cutoff: date) -> None:
    for row in rows:
        publish_date = date.fromisoformat(str(row.get("paper_publish_date") or row.get("contamination", {}).get("paper_publish_date")))
        contamination = dict(row.get("contamination") or {})
        contamination["cutoff_reference"] = cutoff.isoformat()
        contamination["paper_publish_date"] = publish_date.isoformat()
        contamination["publication_stratum"] = "pre_cutoff" if publish_date < cutoff else "post_cutoff"
        contamination["used_for_cutoff_stratification"] = True
        row["contamination"] = contamination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_index", type=Path)
    parser.add_argument("--cutoff-date", default="2023-10-01")
    args = parser.parse_args()

    cutoff = date.fromisoformat(args.cutoff_date)
    index = json.loads(args.dataset_index.read_text(encoding="utf-8"))
    root = args.dataset_index.parent
    paths = [root / index.get("tasks_path", "tasks.jsonl")]
    if index.get("tasks_holdout_path"):
        paths.append(root / index["tasks_holdout_path"])
    for path in paths:
        rows = _load_jsonl(path)
        _stratify(rows, cutoff)
        _write_jsonl(path, rows)

    index["contamination_cutoff_date"] = cutoff.isoformat()
    args.dataset_index.write_text(json.dumps(index, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
