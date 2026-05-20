# Paper2Manim Task Dataset Minimal CSV Schema

This lightweight CSV drives current experiment runners such as
`scripts/run_experiment.py` and `scripts/cross_domain.py`. Each row is one
`paper2manim mvp2 --arxiv <id> --section <name>` invocation.

The filename remains `p2m_v1_schema.md` for compatibility with existing
references, but the split semantics are aligned with P2M-Bench v2.

## Required Columns

| column | type | required | notes |
|---|---|---:|---|
| `id` | str | recommended | Stable task id. If missing, derive from `arxiv_id + section`. |
| `arxiv_id` | str | yes | Bare id (`1706.03762`) or `arXiv:` form. `parse_arxiv` accepts both. |
| `section` | str | yes | Exact or near-exact section title passed to `paper2manim mvp2 --section`. |
| `domain` | enum | yes | `cs / math / physics / quantum / econ`. |
| `split` | enum | yes | Canonical v2 split or supported legacy alias. |
| `expected_scene_count_min` | int | optional | Sanity floor only; blank means no check. |

## Recommended v2 Columns

These columns are optional for current runners but recommended for main
experiment manifests:

| column | type | notes |
|---|---|---|
| `stream_idx` | int/null | Frozen order for `memory_build`. |
| `probe_idx` | int/null | Frozen order for `fixed_probe`. |
| `scene_role` | enum | `BACKGROUND / METHOD / EXPERIMENT / CONCLUSION`. |
| `category` | enum | `Concept / Equation / Algorithm / Figure / Architecture / Experiment`. |
| `difficulty` | enum | `easy / medium / hard`. |
| `human_eval_candidate` | bool | Whether outputs from this task may enter blind human scoring. |

## Canonical v2 Splits

| split | meaning |
|---|---|
| `memory_build` | Ordered stream used only to grow EMB and save snapshots. |
| `fixed_probe` | Held-out tasks used for read-only evaluation of VLM Reflection Only and frozen EMB snapshots. |
| `test_holdout` | Reserved final set; do not tune on it. |
| `cross_train` | Optional appendix: grow EMB on one domain. |
| `cross_test` | Optional appendix: frozen cross-domain probe. |

## Legacy Aliases

Legacy CSVs remain valid:

| legacy split | canonical split |
|---|---|
| `bootstrap` | `memory_build` |
| `eval` | `fixed_probe` |
| `cross_train` | `cross_train` |
| `cross_test` | `cross_test` |

New dataset files and paper text should use `memory_build` and `fixed_probe`.

## Field Isolation

The minimal CSV must not contain answer fields. Evaluation-only fields such as
`key_claims`, `reference_scene_plan`, `human_rubric`, human scores, and fatal
flags belong in a separate evaluation or human-scoring sidecar and must never
enter model prompts or EMB writes.

## Domain Values

Domain values are single-token lowercase strings. They are used by
`--dataset-domain` and by EMB context metadata, so typos must fail validation.
Allowed values are imported from `paper2manim.datasets.constants`.
