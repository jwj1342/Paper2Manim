"""Validate the P2M-Bench v2.1 experiment artefact.

Base validation checks that the dataset is a clean experimental instrument for
the ManimAgent fixed-probe EMB snapshot protocol. Use ``--strict-paper-ready``
after run manifests and human scores have been collected.

Usage:
    python scripts/validate_p2m_bench.py data/p2m_bench_v2/dataset_index.json
    python scripts/validate_p2m_bench.py data/p2m_bench_v2/dataset_index.json --strict-paper-ready
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


HEADLINE_SPLITS = ("memory_build", "fixed_probe", "cross_test")
MATCHED_SPLITS = ("memory_build", "fixed_probe")
EVAL_ONLY_FIELDS = {
    "main_topics",
    "key_claims",
    "reference_scene_plan",
    "human_rubric",
    "vlm_diagnostic_rubric",
    "human_scores",
    "human_pass_at_1",
    "fatal_flags",
    "human_quality_score",
    "condition_hidden",
}
FORBIDDEN_SECTION_MARKERS = (
    "author contribution",
    "acknowledg",
    "references",
    "supplementary material",
    "checklist",
    "broader impacts",
)
GENERIC_ALGORITHM_PHRASES = (
    "animate the algorithm step by step",
    "highlight the update, stopping condition",
    "stopping condition",
)
GENERIC_DRAFT_BEATS = (
    "introduce the concept",
    "why it matters locally",
    "show the relation among the named entities",
    "end with the local takeaway",
)
MAIN_CONDITIONS = {
    "VLM Reflection Only",
    "EMB@0",
    "EMB@50",
    "EMB@100",
    "EMB@200",
    "EMB@400",
}
ABLATIONS = {
    "Full EMB",
    "No success channel",
    "No failure channel",
    "Empty EMB",
    "High success threshold",
    "Low success threshold",
    "Strict failure gate",
    "Loose failure gate",
    "Top-1 retrieval",
}
CONDITION_FILENAME_MARKERS = (
    "vlm",
    "reflection",
    "emb",
    "snapshot",
    "seed",
    "condition",
)


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
                raise AssertionError(f"{path}:{line_no}: invalid JSONL: {exc}") from exc
    return rows


def _nested_keys(obj: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(obj, dict):
        for key, value in obj.items():
            keys.add(str(key))
            keys |= _nested_keys(value)
    elif isinstance(obj, list):
        for value in obj:
            keys |= _nested_keys(value)
    return keys


def _stratum(task: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(task.get("domain")),
        str(task.get("scene_role")),
        str(task.get("category")),
        str(task.get("difficulty")),
    )


def _tasks_for(tasks: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    return [task for task in tasks if task.get("split") == split]


def assert_no_paper_overlap(tasks: list[dict[str, Any]], split_a: str, split_b: str) -> None:
    papers_a = {str(t["paper_id"]) for t in _tasks_for(tasks, split_a)}
    papers_b = {str(t["paper_id"]) for t in _tasks_for(tasks, split_b)}
    overlap = papers_a & papers_b
    if overlap:
        raise AssertionError(f"paper overlap between {split_a} and {split_b}: {sorted(overlap)[:8]}")


def _check_split_counts(index: dict[str, Any], tasks: list[dict[str, Any]]) -> None:
    actual = Counter(t.get("split") for t in tasks if t.get("split") != "test_holdout_debug")
    for split, expected in (index.get("splits") or {}).items():
        if actual.get(split, 0) != expected:
            raise AssertionError(f"{split}: declared {expected}, actual {actual.get(split, 0)}")


def _check_distribution_match(index: dict[str, Any], tasks: list[dict[str, Any]]) -> None:
    scope = index.get("matched_distribution_scope") or {}
    domains = set(scope.get("domains") or [])

    def scoped(split: str) -> list[dict[str, Any]]:
        rows = _tasks_for(tasks, split)
        return [task for task in rows if not domains or task.get("domain") in domains]

    grouped = {split: scoped(split) for split in MATCHED_SPLITS}
    if not all(grouped.values()):
        raise AssertionError("memory_build and fixed_probe must both be non-empty")
    counters = {split: Counter(_stratum(task) for task in rows) for split, rows in grouped.items()}
    sizes = {split: len(rows) for split, rows in grouped.items()}
    all_strata = set(counters["memory_build"]) | set(counters["fixed_probe"])
    missing = [s for s in all_strata if counters["memory_build"][s] == 0 or counters["fixed_probe"][s] == 0]
    if missing:
        raise AssertionError(f"matched splits do not share strata: {missing[:5]}")
    l1 = sum(
        abs(counters["memory_build"][s] / sizes["memory_build"] - counters["fixed_probe"][s] / sizes["fixed_probe"])
        for s in all_strata
    )
    if l1 > 0.25:
        raise AssertionError(f"memory_build/fixed_probe distribution mismatch too high: L1={l1:.3f}")


def _check_headline_tasks(tasks: list[dict[str, Any]]) -> None:
    for task in tasks:
        if task.get("split") not in HEADLINE_SPLITS:
            continue
        if task.get("exclude_from_main"):
            raise AssertionError(f"{task['task_id']}: headline task is excluded from main")
        status = (task.get("eval_only") or {}).get("hydration_status")
        if status != "ok":
            raise AssertionError(f"{task['task_id']}: headline hydration_status must be ok, got {status!r}")
        section = " ".join(
            str(x or "")
            for x in (
                task.get("section"),
                ((task.get("model_input") or {}).get("target_unit") or {}).get("title"),
            )
        ).lower()
        if any(marker in section for marker in FORBIDDEN_SECTION_MARKERS):
            raise AssertionError(f"{task['task_id']}: forbidden non-content section")


def _check_isolation(index: dict[str, Any], tasks: list[dict[str, Any]]) -> None:
    forbidden = EVAL_ONLY_FIELDS | set(index.get("field_isolation", {}).get("eval_only") or [])
    for task in tasks:
        if "model_visible" in task:
            raise AssertionError(f"{task['task_id']}: legacy model_visible field is not allowed in v2.1 tasks")
        model_keys = _nested_keys(task.get("model_input") or {})
        leaked = forbidden & model_keys
        if leaked:
            raise AssertionError(f"{task['task_id']}: eval-only keys leaked into model_input: {sorted(leaked)}")
        memory_record = task.get("memory_source_record") or task.get("emb_construction_record") or {}
        leaked_memory = forbidden & _nested_keys(memory_record)
        if leaked_memory:
            raise AssertionError(f"{task['task_id']}: EMB-never-write keys leaked into memory record: {sorted(leaked_memory)}")


def _check_reference_plans(tasks: list[dict[str, Any]]) -> None:
    for task in tasks:
        plan = " ".join(str(item.get("beat", item)) for item in (task.get("eval_only") or {}).get("reference_scene_plan") or [])
        if any(phrase in plan.lower() for phrase in GENERIC_DRAFT_BEATS):
            raise AssertionError(f"{task['task_id']}: generic draft reference_scene_plan appears in release task")
        if task.get("category") not in {"Algorithm", "Architecture"} and task.get("scene_role") != "METHOD":
            if any(phrase in plan.lower() for phrase in GENERIC_ALGORITHM_PHRASES):
                raise AssertionError(f"{task['task_id']}: generic algorithm wording on non-algorithm task")


def _check_annotations(tasks: list[dict[str, Any]], *, strict: bool) -> None:
    for task in tasks:
        if task.get("target_annotation_status") == "llm_draft":
            raise AssertionError(f"{task['task_id']}: llm_draft target annotations cannot appear in release-facing tasks")
        source = (task.get("eval_only") or {}).get("annotation_source")
        if source == "scripted_synthetic_simulation":
            raise AssertionError(f"{task['task_id']}: scripted synthetic annotations must live in _draft/, not release tasks")
        if task.get("output_annotation_status") != "not_run":
            continue
        labels = {"human_scores", "human_pass_at_1", "fatal_flags", "human_quality_score"} & set((task.get("eval_only") or {}))
        if labels:
            raise AssertionError(f"{task['task_id']}: human output labels present before evaluation: {sorted(labels)}")
        if task.get("human_eval_candidate") and task.get("split") != "fixed_probe":
            raise AssertionError(f"{task['task_id']}: human_eval_candidate must be fixed_probe")
    if strict:
        candidates = [t for t in _tasks_for(tasks, "fixed_probe") if t.get("human_eval_candidate")]
        if len(candidates) < 25:
            raise AssertionError(f"strict: need at least 25 fixed_probe human_eval_candidate tasks, found {len(candidates)}")


def _check_contamination(tasks: list[dict[str, Any]]) -> None:
    allowed = {"pre_cutoff", "post_cutoff"}
    for task in tasks:
        contamination = task.get("contamination") or {}
        if contamination.get("publication_stratum") not in allowed:
            raise AssertionError(f"{task['task_id']}: invalid publication_stratum")
        if not contamination.get("paper_publish_date"):
            raise AssertionError(f"{task['task_id']}: missing contamination paper_publish_date")
        cutoff_reference = str(contamination.get("cutoff_reference") or "")
        if cutoff_reference == "underlying_llm_cutoff":
            raise AssertionError(f"{task['task_id']}: cutoff_reference must be a concrete date")
        try:
            publish_date = tuple(map(int, str(contamination["paper_publish_date"]).split("-")))
            cutoff_date = tuple(map(int, cutoff_reference.split("-")))
        except Exception as exc:  # noqa: BLE001
            raise AssertionError(f"{task['task_id']}: cutoff_reference and paper_publish_date must be YYYY-MM-DD") from exc
        expected = "pre_cutoff" if publish_date < cutoff_date else "post_cutoff"
        if contamination["publication_stratum"] != expected:
            raise AssertionError(f"{task['task_id']}: publication_stratum does not match cutoff_reference")


def _check_paper_licenses(tasks: list[dict[str, Any]]) -> None:
    licenses = {str(task.get("paper_license") or "") for task in tasks}
    for task in tasks:
        if not task.get("paper_license") or not task.get("paper_license_url"):
            raise AssertionError(f"{task['task_id']}: missing paper license metadata")
        if task.get("paper_license_source") != "arxiv_abs_page":
            raise AssertionError(f"{task['task_id']}: paper_license_source must be arxiv_abs_page")
    if len(tasks) > 1 and len(licenses) == 1:
        raise AssertionError("paper_license is constant across all release tasks; expected per-paper arXiv license metadata")


def _check_snapshot_plausibility(index: dict[str, Any], tasks: list[dict[str, Any]]) -> None:
    targets = index.get("snapshot_targets") or []
    if set(targets) != {0, 50, 100, 200, 400}:
        raise AssertionError(f"snapshot_targets must be exactly [0,50,100,200,400], got {targets}")
    expected = sum(int(task.get("expected_min_memory_records") or 0) for task in _tasks_for(tasks, "memory_build"))
    if expected < max(targets):
        raise AssertionError(f"memory stream cannot plausibly reach EMB@{max(targets)}: expected {expected}")


def _check_release_policy(index: dict[str, Any], tasks: list[dict[str, Any]]) -> None:
    if (index.get("release_policy") or {}).get("raw_full_paper_text_in_release") is not False:
        raise AssertionError("release_policy.raw_full_paper_text_in_release must be false")
    for task in tasks:
        model_input = task.get("model_input") or {}
        if "paper_full_text" in model_input:
            raise AssertionError(f"{task['task_id']}: paper_full_text appears in model_input")
        source_text = task.get("source_paper_text") or {}
        if source_text.get("raw_paper_text_available_in_release"):
            raise AssertionError(f"{task['task_id']}: raw paper text marked release-available")


def _check_probe_manifest(path: Path, tasks_by_id: dict[str, dict[str, Any]]) -> None:
    if not path.exists():
        return
    for row in _load_jsonl(path):
        task = tasks_by_id.get(row.get("task_id"))
        if not task:
            raise AssertionError(f"{path}: unknown task_id {row.get('task_id')}")
        if task.get("split") != "fixed_probe":
            raise AssertionError(f"{path}: probe manifest task must be fixed_probe: {row.get('task_id')}")
        if row.get("condition") not in MAIN_CONDITIONS:
            raise AssertionError(f"{path}: unsupported condition {row.get('condition')}")
        if row.get("writes_to_emb") is not False:
            raise AssertionError(f"{path}: fixed probe runs must be read-only")


def _check_snapshot_manifest(path: Path, tasks_by_id: dict[str, dict[str, Any]], targets: set[int]) -> None:
    if not path.exists():
        return
    observed: set[int] = set()
    for row in _load_jsonl(path):
        record_count = int(row.get("record_count_total") or 0)
        for target in targets:
            if record_count >= target and str(row.get("snapshot_id", "")).endswith(f"_{target}"):
                observed.add(target)
        if record_count > 0 and (int(row.get("record_count_success") or 0) <= 0 or int(row.get("record_count_failure") or 0) <= 0):
            raise AssertionError(f"{path}: non-empty snapshots need success and failure records")
        for task_id in row.get("source_task_ids") or [row.get("last_memory_build_task_id")]:
            if task_id and tasks_by_id.get(task_id, {}).get("split") != "memory_build":
                raise AssertionError(f"{path}: snapshot source task is not memory_build: {task_id}")
    if observed and not observed.issuperset(targets):
        raise AssertionError(f"{path}: missing snapshot targets {sorted(targets - observed)}")


def _check_blind_outputs(path: Path, tasks_by_id: dict[str, dict[str, Any]]) -> None:
    if not path.exists():
        return
    for row in _load_jsonl(path):
        if row.get("task_id") not in tasks_by_id:
            raise AssertionError(f"{path}: unknown task_id {row.get('task_id')}")
        if row.get("attempt_idx") != 0 or row.get("is_first_attempt") is not True:
            raise AssertionError(f"{path}: blind outputs must expose first-attempt videos only")
        rater_path = str(row.get("video_path_for_rater") or "").lower()
        if any(marker in rater_path for marker in CONDITION_FILENAME_MARKERS):
            raise AssertionError(f"{path}: rater filename reveals condition: {rater_path}")


def _check_optional_manifest_paths(root: Path, index: dict[str, Any], tasks: list[dict[str, Any]]) -> None:
    tasks_by_id = {task["task_id"]: task for task in tasks}
    paths = index.get("manifest_paths") or {}
    targets = set(map(int, index.get("snapshot_targets") or []))
    checks = {
        "snapshot_manifest": lambda p: _check_snapshot_manifest(p, tasks_by_id, targets),
        "probe_run_manifest": lambda p: _check_probe_manifest(p, tasks_by_id),
        "blind_output_sidecar": lambda p: _check_blind_outputs(p, tasks_by_id),
    }
    for key, check in checks.items():
        rel = paths.get(key)
        if rel and "<experiment_id>" not in rel:
            check(root / rel)


def _check_strict_outputs(root: Path, index: dict[str, Any]) -> None:
    paths = index.get("manifest_paths") or {}
    required = ("human_scores",)
    for key in required:
        rel = paths.get(key)
        if not rel:
            raise AssertionError(f"strict: missing manifest path for {key}")
        path = root.parent.parent / rel if rel.startswith("data/") else root / rel
        if not path.exists():
            raise AssertionError(f"strict: missing required file {path}")
        if not _load_jsonl(path):
            raise AssertionError(f"strict: required file is empty {path}")


def validate(index_path: Path, *, strict: bool = False) -> None:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    root = index_path.parent
    if index.get("single_json_path"):
        payload_path = root / index["single_json_path"]
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        tasks = payload.get("tasks") or []
        holdout = payload.get("holdout_tasks") or []
        index = payload.get("metadata") or index
    elif index.get("format") == "single_json" and "tasks" in index:
        tasks = index.get("tasks") or []
        holdout = index.get("holdout_tasks") or []
        index = index.get("metadata") or index
    else:
        tasks = _load_jsonl(root / index.get("tasks_path", "tasks.jsonl"))
        holdout_path = index.get("tasks_holdout_path")
        holdout = _load_jsonl(root / holdout_path) if holdout_path else []
    if holdout and any(task.get("split") != "test_holdout_debug" for task in holdout):
        raise AssertionError("holdout tasks must contain only test_holdout_debug tasks")
    if any(task.get("split") == "test_holdout_debug" for task in tasks):
        raise AssertionError("release tasks_path must not include quarantined test_holdout_debug tasks")
    duplicate_ids = [task_id for task_id, count in Counter(t.get("task_id") for t in tasks).items() if count > 1]
    if duplicate_ids:
        raise AssertionError(f"duplicate task_id values: {duplicate_ids[:10]}")

    _check_split_counts(index, tasks)
    for split_a, split_b in (("memory_build", "fixed_probe"), ("memory_build", "cross_test"), ("fixed_probe", "cross_test")):
        assert_no_paper_overlap(tasks, split_a, split_b)
    _check_distribution_match(index, tasks)
    _check_headline_tasks(tasks)
    _check_isolation(index, tasks)
    _check_reference_plans(tasks)
    _check_annotations(tasks, strict=strict)
    _check_contamination(tasks)
    _check_paper_licenses(tasks)
    _check_snapshot_plausibility(index, tasks)
    _check_release_policy(index, tasks)
    _check_optional_manifest_paths(root, index, tasks)
    if strict:
        _check_strict_outputs(root, index)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_index", type=Path)
    parser.add_argument("--strict-paper-ready", action="store_true")
    args = parser.parse_args(argv)
    try:
        validate(args.dataset_index, strict=args.strict_paper_ready)
    except Exception as exc:  # noqa: BLE001
        print(f"validation failed: {exc}", file=sys.stderr)
        return 1
    mode = "strict paper-ready" if args.strict_paper_ready else "base"
    print(f"validation passed ({mode}): {args.dataset_index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
