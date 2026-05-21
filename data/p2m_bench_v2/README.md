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
- matched `memory_build` / `fixed_probe` distributions
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
