"""Validate a Paper2Manim task dataset CSV against the v1 schema.

Schema reference: ``examples/datasets/p2m_v1_schema.md``.

Usage::

    python scripts/dataset_validate.py examples/datasets/p2m_v1.csv
    python scripts/dataset_validate.py path/to.csv --offline   # skip arXiv fetches

Outputs (next to the input file):
- ``<csv>.validated.csv`` — only the rows that passed
- ``<csv>.errors.txt`` — one line per failed row with reason

In ``--offline`` mode only schema-level checks run (column presence, enum membership,
int parse). With network, each row's ``arxiv_id``/``section`` is also probed via
``parse_arxiv``; ``SourceUnavailable`` is recorded as ``unreachable`` and is *not*
treated as a hard failure (those rows are still emitted to the validated set so the
user can decide).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_DOMAINS = {"cs", "math", "physics", "quantum", "econ"}
_SPLITS = {"bootstrap", "eval", "cross_train", "cross_test"}
_REQUIRED_COLS = ("arxiv_id", "section", "domain", "split")


def _strip_comment_lines(path: Path) -> list[str]:
    out: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        out.append(raw)
    return out


def _check_row(row: dict, idx: int) -> str | None:
    for col in _REQUIRED_COLS:
        if not (row.get(col) or "").strip():
            return f"row {idx}: missing column '{col}'"
    if row["domain"] not in _DOMAINS:
        return f"row {idx}: unknown domain '{row['domain']}' (allowed: {sorted(_DOMAINS)})"
    if row["split"] not in _SPLITS:
        return f"row {idx}: unknown split '{row['split']}' (allowed: {sorted(_SPLITS)})"
    raw = (row.get("expected_scene_count_min") or "").strip()
    if raw:
        try:
            n = int(raw)
            if n < 0:
                return f"row {idx}: expected_scene_count_min must be >= 0"
        except ValueError:
            return f"row {idx}: expected_scene_count_min '{raw}' is not an int"
    return None


def _probe_arxiv(arxiv_id: str, section: str) -> str | None:
    """Return None on success, or an error tag string."""
    try:
        from paper2manim.parsers import parse_arxiv
        from paper2manim.parsers.arxiv_source import SourceUnavailable
    except Exception as exc:  # noqa: BLE001
        return f"import-error:{exc}"
    try:
        result = parse_arxiv(arxiv_id, section=section, allow_pdf_fallback=False)
    except SourceUnavailable as exc:
        return f"unreachable:{exc}"
    except Exception as exc:  # noqa: BLE001
        return f"fetch-error:{exc}"
    if section and section.lower() not in result.text.lower():
        return f"section-not-found:{section}"
    if len(result.text) > 4000:
        return f"too-long:{len(result.text)}>4000"
    return None


def validate(csv_path: Path, *, offline: bool) -> tuple[list[dict], list[str]]:
    lines = _strip_comment_lines(csv_path)
    if not lines:
        return [], [f"{csv_path}: file is empty after stripping comments"]
    reader = csv.DictReader(lines)
    if reader.fieldnames is None:
        return [], [f"{csv_path}: missing header row"]
    missing = [c for c in _REQUIRED_COLS if c not in reader.fieldnames]
    if missing:
        return [], [f"{csv_path}: header missing columns {missing}"]

    validated: list[dict] = []
    errors: list[str] = []
    for i, row in enumerate(reader, start=1):
        err = _check_row(row, i)
        if err:
            errors.append(err)
            continue
        if not offline:
            tag = _probe_arxiv(row["arxiv_id"].strip(), row["section"].strip())
            if tag and tag.startswith("unreachable:"):
                # soft warn — keep row, append a note column
                row["_note"] = tag
            elif tag:
                errors.append(f"row {i}: {tag}")
                continue
        validated.append(row)
    return validated, errors


def _write_outputs(csv_path: Path, validated: list[dict], errors: list[str]) -> tuple[Path, Path]:
    out_csv = csv_path.with_suffix(csv_path.suffix + ".validated.csv")
    out_err = csv_path.with_suffix(csv_path.suffix + ".errors.txt")
    if validated:
        cols = list(validated[0].keys())
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            writer.writerows(validated)
    else:
        out_csv.write_text("", encoding="utf-8")
    out_err.write_text("\n".join(errors) + ("\n" if errors else ""), encoding="utf-8")
    return out_csv, out_err


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv", type=Path)
    p.add_argument("--offline", action="store_true", help="skip arxiv fetch probes")
    args = p.parse_args(argv)

    if not args.csv.exists():
        print(f"error: file not found: {args.csv}", file=sys.stderr)
        return 2
    validated, errors = validate(args.csv, offline=args.offline)
    out_csv, out_err = _write_outputs(args.csv, validated, errors)
    print(f"validated: {len(validated)} -> {out_csv}")
    print(f"errors:    {len(errors)} -> {out_err}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
