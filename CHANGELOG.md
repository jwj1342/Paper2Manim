# Changelog

All notable changes to this project are tracked here. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning is loose because there's no formal release yet.

## [Unreleased]

### Removed
- ~1,997 lines of D3-era DDD scaffolding that LangGraph never adopted: `paper2manim/domain/`, `paper2manim/infrastructure/{models,llm,rendering,documents,storage,tracing,video}/`, `paper2manim/utils/{prompt_loader,text_utils}.py`, plus empty package shells `paper2manim/{apps,application,runtime}/` and `tests/{unit,integration}/`. Verified zero external callers before deletion. Resolves long-standing D3 debt in `docs/progress.md` §To Do E.
- Three `PaperState` fields (`vlm_revision_count` / `last_visual_review` / `current_montage_path`) labeled "mvp1 only" but never read off `PaperState` — mvp1 has no VLM nodes and mvp2's fan-out keeps them on `SceneState`.

### Added
- `CONTRIBUTING.md` — onboarding + coding conventions + known debt index.
- `CHANGELOG.md` — this file.
- `pyproject.toml` `[tool.mypy]` baseline config (pragmatic, not strict). Run `mypy` locally to see the ~26 outstanding union-attr findings.
- `pyproject.toml` `[tool.coverage.run]` + `[tool.coverage.report]` config + `pytest-cov` in `dev` extras. `fail_under` deliberately unset; quantify first, tighten later.
- `pyproject.toml` `[tool.pytest.ini_options].filterwarnings` to silence two unactionable upstream warnings (`LangChainPendingDeprecationWarning` for `allowed_objects`, `PydanticSerializationUnexpectedValue`).

### Fixed
- pytest run is now silent — was emitting 1 warning per run from `langgraph/cache/base/__init__.py`.

---

## Pre-changelog history (selected merged PRs)

These landed before this file existed. Listed by merge order for traceability.

### Research scaffolding (RQ1 / RQ3 + EMB health CLI)
- **#29** `feat(graph): visual best-of-N` — ship highest-scored rendition rather than the latest, fixing the v2-worse-than-v1 regression observed in `docs/vlm_experiment.md` §5 #3.
- **#28** `fix(emb,exp): #27 三处 bug` — domain pass-through, CSV comment stripping, embedder spec dim mismatch.
- **#25** `feat(emb): cross-domain freeze + dataset schema` — `scripts/cross_domain.py` train/test/baseline driver; `examples/datasets/p2m_v1_schema.md`; `--emb-readonly` / `--dataset-domain` flags.
- **#24** `feat(plot): per-config Hero Plot + 95% bootstrap CI + EMB hits panel` — `scripts/plot_evolution.py` manifest mode.
- **#23** `feat(exp): A/B/C 实验对照 runner + RUN_ID stdout 标记` — `scripts/run_experiment.py` driving proposal §5 RQ1.
- **#22** `feat(cli): paper2manim emb {stats,list,show,prune,retest}` — EMB health CLI; resolves #17 cold-record pruning by `hit_count` / `last_used`.
- **#21** `fix(emb): distill 评分 schema 对齐到 proposal §4.2 (3 维 0-100)`.

### MVP 3.0 §4 — Episodic Memory Bank backend
- **#19** `perf(graph): MVP 2.0 图计算并行化` — `fan_out_scenes` Send×N + per-scene subgraph + `RENDER_SEMAPHORE` + `TokenBucket`; CLI `--scene-parallelism` / `--render-concurrency` / `--llm-rps`. Default off = zero behavioral change.
- **#16** `feat(emb): dual-channel Episodic Memory Bank backend` — schema + SQLite store + Faiss/in-memory index + sentence-transformers/HashEmbedder + manager + distill + retrieval; CLI `--emb` family.

### MVP 3.0 §4.2 — VLM judge
- **#15** `feat(vlm): 评分 schema 收敛 3 维 × 0-100 + auto-pass bypass` — resolves #12 schema canonicalization.
- **#11** `feat(llm): multi-provider 工厂 + VLM 接图` — YAML-routed `openai_compatible` / `anthropic` providers + Azure bearer + `omit_temperature` for Claude Opus 4.7.

### MVP 2.0 + plumbing
- **#10** `docs(proposal): add Failure Pattern Memory (4b) to dual-channel EMB`.
- **#8** `docs: refocus Research Proposal on VLM-driven episodic memory evolution`.
- **#7** `docs: sync progress.md + README with arXiv parser and CI/CD landed`.
- **#6** `fix(ci): install pangocairo/cairo system deps for manimpango build`.
- **#5** `docs: rewrite repo-automation to describe settings, not commands`.
- **#2-#4** Dependabot infra version bumps.
