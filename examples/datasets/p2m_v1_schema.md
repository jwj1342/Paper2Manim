# Paper2Manim Task Dataset v1 Schema

Drives experiment runners (`scripts/run_experiment.py`, `scripts/cross_domain.py`).
Each row is one `paper2manim mvp2 --arxiv <id> --section <name>` invocation.

## Columns

| column | type | required | notes |
|---|---|---|---|
| `arxiv_id` | str | yes | bare id (`1706.03762`) or `arXiv:` form. `parse_arxiv` accepts both. |
| `section` | str | yes | exact `\section{...}` title slice; case-sensitive match |
| `domain` | enum | yes | one of `cs / math / physics / quantum / econ` |
| `split` | enum | yes | one of `bootstrap / eval / cross_train / cross_test` |
| `expected_scene_count_min` | int | optional | sanity floor; storyboarder usually returns ≥ this many scenes; blank → no check |

## Splits — what they mean

- **bootstrap** — used by `scripts/run_experiment.py` to grow EMB and measure A/B/C learning curves on the *same* domain.
- **eval** — held-out tasks for measuring cross-task generalization within a domain (no EMB writes).
- **cross_train** — feeds the *train* phase of `scripts/cross_domain.py`; EMB grows on this domain.
- **cross_test** — feeds the *test* phase of `scripts/cross_domain.py` with `--emb-readonly`; the EMB built on `cross_train` is frozen, then transferred to a different domain.

## Domain values

Single-token, lowercase. Used both for split-level filtering and for the `Context.domain` column written into EMB records (so retrieval can later filter by domain). The five values above are the v1 ceiling — if you need a new one, update `_DOMAINS` in `scripts/dataset_validate.py`.
