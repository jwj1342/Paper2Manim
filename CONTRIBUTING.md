# Contributing to Paper2Manim

Thanks for considering a contribution. This is a research-track repository — the bar is "doesn't break the Hero Plot run" rather than "production-grade". Below is the minimum you need to know to land a change.

> Deeper context lives in [`docs/repo-automation.md`](./docs/repo-automation.md) (branch protection, CI matrix, Dependabot) and [`docs/progress.md`](./docs/progress.md) (current debt + roadmap). Skim those before tackling anything non-trivial.

## Quick start

```bash
git clone <repo> Paper2Manim && cd Paper2Manim
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # adds pytest, pytest-cov, ruff, mypy
pip install -e ".[emb]"          # add only if you'll touch the EMB backend
cp .env.example .env             # fill in MIMO_API_KEY or your provider key
```

## Local checks before opening a PR

```bash
ruff check paper2manim tests scripts        # must be clean
pytest -m "not slow"                        # 336+ tests, ~10s
pytest --cov=paper2manim                    # optional, no enforced threshold
mypy                                         # advisory; see "Known debt" below
```

CI runs the first two on py3.11 + py3.12. PRs are merged via squash, so a clean linear commit on your branch is welcome but not required.

## Where things live

| You want to ... | Look in |
|---|---|
| Add a new LLM agent | `paper2manim/agents/<name>.py` + `prompts/<name>.md` + wire it in `graphs/scene_graph.py` or `graphs/mvp2.py` |
| Change the LangGraph topology | `paper2manim/graphs/` (`mvp1.py` linear, `mvp2.py` fan-out parent, `scene_graph.py` per-scene subgraph) |
| Tune a prompt | `prompts/*.md` — hot-loaded; no reinstall needed |
| Add an experiment driver | `scripts/` (`run_experiment.py`, `cross_domain.py`, `run_bootstrap.py`) |
| Touch the EMB | `paper2manim/emb/` (schema / store / index / embedder / manager / distill / retrieval) |
| Add CLI flags | `paper2manim/cli.py` (parent flags) or `paper2manim/cli_emb.py` (EMB subcommands) |
| Adjust render sandbox | `paper2manim/sandbox/render.py` (subprocess + rlimit) and `quality/manim_static_checker.py` (AST blacklist) |
| Add LangGraph state fields | `paper2manim/state.py` (PaperState) **and** `paper2manim/graphs/scene_graph.py` (SceneState) when relevant |

## Coding conventions

- **No new comments unless the *why* is non-obvious.** Identifier names should already say *what*. Don't reference PR numbers or "added for X flow" — that rots fast; commit messages already have the context.
- **Don't restore the DDD ghosts.** `paper2manim/domain/` / `paper2manim/infrastructure/{models,llm,rendering}/` were removed in the cleanup PR — they're unused parallel scaffolding. If you need a new abstraction, prefer extending the active LangGraph state + agent contract.
- **`infrastructure/vlm/`** is the only live piece of the legacy `infrastructure/` tree — keep it that way.
- **Multi-provider config** flows through `config.yaml` → `paper2manim/config/model_config.py` → `paper2manim/llm.py`. The env-`MIMO_API_KEY` path is a fallback for users without `config.yaml`. Do not hardcode provider names elsewhere.
- **Prompts are external + hot-loaded.** Never inline a long prompt string in `agents/*.py`; write it in `prompts/<name>.md` and read via `paper2manim.prompts`.
- **Per-scene state in fan-out** lives on `SceneState` (`scene_graph.py`), not `PaperState`. Reading scene-level fields off `PaperState` after fan-out is wrong — the reducers don't see them.

## Test policy

- **`@pytest.mark.slow`** for anything that needs a real LLM or real Manim render. CI skips these. Default suite uses mocks via `tests/conftest.py::mock_llm`.
- **One test file per agent / module.** Subgraph or graph-level integration tests go in `tests/test_graph_*.py`.
- **Don't mock LangGraph internals.** Build a tiny real subgraph and run it; LangGraph is fast in-process.
- **CSV-driven experiment runners** (`scripts/run_experiment.py`, `scripts/cross_domain.py`) ship with `--dry-run` for testing the wiring without burning credits — use that in tests.

## Branching + PRs

- `main` is protected: 1 approval + 3 CI checks (pytest py3.11 / pytest py3.12 / CodeQL) + strict-up-to-date + no force push.
- Open PRs from feature branches: `feat/<slug>`, `fix/<slug>`, `docs/<slug>`, `chore/<slug>`.
- Squash merge; commit message body becomes the PR description if you `gh pr create --fill`.
- Keep PRs focused — one logical change per PR. Cross-cutting cleanups (lint sweep, doc sync) are fine as their own PR.

## Known debt (don't make it worse)

Tracked in [`docs/progress.md`](./docs/progress.md) §To Do. Headlines:

- **mypy** currently surfaces ~26 union-attr / annotation findings; the config (`[tool.mypy]` in `pyproject.toml`) deliberately ships in a pragmatic-not-strict mode. Don't add new typing-ignored functions; do fix neighboring errors when you touch a file.
- **Coverage** is configured but not enforced (`fail_under` intentionally unset). Run `pytest --cov` locally to see what your change moves.
- **Hero Plot main experiment** (RQ1) has the runner + plotter wired, but real ≥200-paper-section data hasn't been collected. If you're adding to `scripts/run_experiment.py`, keep `--dry-run` working.
- **CHANGELOG.md** entries should be appended under `## [Unreleased]` for any user-visible change.

## Secrets

Never commit `.env`, `MiMo-API.txt`, or any `tp-`-prefixed token. `.gitignore` covers the common cases and GitHub Secret Scanning + Push Protection are enabled, but the human-side rule still stands.

Real CI secrets live in `Settings → Secrets and variables → Actions`; fork PRs do **not** receive them — this is GitHub's policy, not a misconfiguration.
