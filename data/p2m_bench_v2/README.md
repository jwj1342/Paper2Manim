# P2M-Bench v2.1

This directory is the release-facing dataset layout.

- `p2m_bench_v2.json` is the final single-file dataset. It contains metadata,
  headline tasks, quarantined holdout tasks, and paper metadata.
- `dataset_index.json` is a small pointer file for tooling.
- `annotations/human_scores.jsonl` is reserved for real output-level human
  scores and stays separate from the model-visible dataset.

Validate the final dataset:

```bash
python scripts/validate_p2m_bench.py data/p2m_bench_v2/dataset_index.json
```

Human scores are intentionally not embedded in `p2m_bench_v2.json`.
