# P2M-Bench v2.1

This directory contains the release-facing dataset artefact for the ManimAgent
EMNLP experiment. The legacy full-text single JSON source was used only as a
migration input and is not part of this release-facing layout.

Entry point:

```bash
python scripts/validate_p2m_bench.py data/p2m_bench_v2/dataset_index.json
```

The base validator checks the fixed-probe EMB snapshot instrument:

- paper-disjoint `memory_build`, `fixed_probe`, and `cross_test` splits
- matched `memory_build` / `fixed_probe` distributions on the CS subset
- `hydration_status == "ok"` for all headline tasks
- no full-paper text in model-visible release fields
- eval-only fields kept out of model input and EMB writeback
- snapshot target plausibility for `EMB@0/50/100/200/400`
- schemas for fixed-probe runs, blind outputs, human scores, ablations,
  online evolution, cross-domain transfer, and retrieval inspection

Strict paper-ready validation is intentionally stronger:

```bash
python scripts/validate_p2m_bench.py data/p2m_bench_v2/dataset_index.json --strict-paper-ready
```

It should pass only after real target audits, output-level human scores, and
experiment run manifests have been collected. LLM-generated draft annotations
are not marked as human-audited in this dataset.

Layout:

- `tasks.jsonl` contains only the 112 headline tasks.
- `tasks_holdout.jsonl` contains the 213 quarantined `test_holdout_debug`
  fallback tasks. They are excluded from EMB construction, fixed-probe
  evaluation, human scoring, and paper headline claims.
- `_draft/eval_annotations.jsonl` contains scripted draft `key_claims` and
  `reference_scene_plan` values. These are placeholders for audit planning, not
  release-facing ground truth.

Contamination stratification uses the concrete cutoff date `2023-10-01`.
`publication_stratum` is computed from each task's `paper_publish_date` against
that cutoff.

Known limitations:

- Fixed-probe paper-level generalization is based on 13 papers / 33 tasks.
  Report `aggregate_human_scores.py --dataset-index ...` outputs with the
  cluster bootstrap confidence intervals, and interpret adjacent EMB snapshot
  conditions cautiously when intervals overlap.
- The matched distribution check is meaningful only on the CS subset in v2.1.
  Math and physics cells are retained as coverage probes but are too small for
  a matched-distribution claim.
- Scene roles are skewed toward BACKGROUND and CONCLUSION tasks. METHOD and
  EXPERIMENT coverage should be expanded before making broad claims about
  formula-heavy layout or experiment-result animation.
- `sample_human_eval_subset.py` emits rows only after target audit metadata is
  populated. Before then it exits with a stderr message rather than silently
  returning an empty sample.
