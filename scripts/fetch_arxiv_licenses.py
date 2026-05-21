"""Fetch arXiv license URLs from abstract pages and write P2M-Bench metadata.

The arXiv Atom API does not consistently expose the selected license. This
script reads the license link from each paper's abstract page instead.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path
from typing import Any


LICENSE_RE = re.compile(r'<div class="abs-license">\s*<a href="([^"]+)"', re.IGNORECASE)
LICENSE_LABELS = {
    "http://arxiv.org/licenses/nonexclusive-distrib/1.0/": "arXiv.org perpetual, non-exclusive license 1.0",
    "https://arxiv.org/licenses/nonexclusive-distrib/1.0/": "arXiv.org perpetual, non-exclusive license 1.0",
    "http://creativecommons.org/licenses/by/4.0/": "CC BY 4.0",
    "https://creativecommons.org/licenses/by/4.0/": "CC BY 4.0",
    "http://creativecommons.org/licenses/by-sa/4.0/": "CC BY-SA 4.0",
    "https://creativecommons.org/licenses/by-sa/4.0/": "CC BY-SA 4.0",
    "http://creativecommons.org/licenses/by-nc-sa/4.0/": "CC BY-NC-SA 4.0",
    "https://creativecommons.org/licenses/by-nc-sa/4.0/": "CC BY-NC-SA 4.0",
    "http://creativecommons.org/licenses/by-nc-nd/4.0/": "CC BY-NC-ND 4.0",
    "https://creativecommons.org/licenses/by-nc-nd/4.0/": "CC BY-NC-ND 4.0",
    "http://arxiv.org/licenses/assumed-1991-2003/": "arXiv legacy assumed license (1991-2003)",
    "https://arxiv.org/licenses/assumed-1991-2003/": "arXiv legacy assumed license (1991-2003)",
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _fetch_license(arxiv_id: str, *, timeout: int) -> tuple[str, str]:
    url = f"https://arxiv.org/abs/{arxiv_id}"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Paper2Manim metadata audit (mailto:metadata@example.invalid)"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        html = response.read().decode("utf-8", errors="replace")
    match = LICENSE_RE.search(html)
    if not match:
        return "unknown", ""
    license_url = match.group(1)
    return LICENSE_LABELS.get(license_url, license_url), license_url


def _paper_paths(root: Path) -> list[Path]:
    return sorted((root / "papers").glob("*.json"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_index", type=Path)
    parser.add_argument("--delay-seconds", type=float, default=1.0)
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()

    index = json.loads(args.dataset_index.read_text(encoding="utf-8"))
    root = args.dataset_index.parent
    licenses: dict[str, tuple[str, str]] = {}
    for paper_path in _paper_paths(root):
        paper = json.loads(paper_path.read_text(encoding="utf-8"))
        arxiv_id = str(paper["arxiv_id"])
        label, url = _fetch_license(arxiv_id, timeout=args.timeout)
        paper["paper_license"] = label
        paper["paper_license_url"] = url
        paper["paper_license_source"] = "arxiv_abs_page"
        paper_path.write_text(json.dumps(paper, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        licenses[arxiv_id] = (label, url)
        time.sleep(args.delay_seconds)

    for rel_key in ("tasks_path", "tasks_holdout_path"):
        rel = index.get(rel_key)
        if not rel:
            continue
        path = root / rel
        rows = _load_jsonl(path)
        for row in rows:
            label, url = licenses.get(str(row.get("arxiv_id")), (row.get("paper_license", "unknown"), ""))
            row["paper_license"] = label
            row["paper_license_url"] = url
            row["paper_license_source"] = "arxiv_abs_page" if url else "unknown"
        _write_jsonl(path, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
