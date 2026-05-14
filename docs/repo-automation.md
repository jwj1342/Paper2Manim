# 仓库自动化与协作规范

本仓库使用 GitHub 原生功能做持续集成、依赖管理、安全扫描与分支保护。本文说明**当前启用了什么**、**怎么用**、以及**贡献者要注意什么**。

---

## 1. CI / CD 总览

| 工作流 | 文件 | 触发 | 作用 |
|---|---|---|---|
| **CI**（lint + 测试） | `.github/workflows/ci.yml` | push 到 `main` / 任何 PR | ruff + pytest（mock 测试，~2 分钟） |
| **CodeQL** 安全扫描 | `.github/workflows/codeql.yml` | push / PR + 每周一 06:00 UTC | 静态安全分析（SQL 注入、命令注入、不安全反序列化等） |
| **Dependabot** 依赖更新 | `.github/dependabot.yml` | 每周一 / 每月（Actions） | 自动开 PR 升级 `pyproject.toml` 和 `actions/*` 版本 |

> Nightly 端到端 workflow（真实 LLM + 真实渲染）**暂未启用**。等 MVP 1.0 跑通后再加，避免 flaky CI 和 API 费用。

---

## 2. CI 详解（`ci.yml`）

### 跑什么

- **Python matrix**：3.11 + 3.12 并行
- **ruff check**：lint，**失败则 PR 红**
- **ruff format --check**：格式漂移，目前**仅 warning**（`continue-on-error: true`）
- **pytest -m "not slow"**：跑 mock 测试套件（当前 193 个），**失败则 PR 红**

### 不跑什么

- `@pytest.mark.slow` 标记的真实渲染测试（需要 LaTeX 工具链 + 几分钟）
- 任何调用真实 MiMo API 的测试（CI 注入了假 key `tp-fake-ci-key`，真实调用会立即 401）
- 任何依赖 arxiv.org / mermaid.ink 的网络测试

### 本地复现 CI 流程

```bash
pip install -e ".[dev]"
ruff check paper2manim tests
pytest -m "not slow" -v
```

PR 前跑一下能省一次 CI 来回。

### concurrency

同一分支后续 push 会**取消上一次未完成的 run**，避免排队浪费分钟数。

---

## 3. 分支保护（`main`）

> GitHub 自 2023 起主推 **Rulesets** 替代旧的 Branch protection rules，两者功能等价、并存。本仓库目前用经典 Branch protection 实现，未来可平滑迁移到 Rulesets。

**当前 `main` 分支已生效的约束**：

| 约束 | 作用 |
|---|---|
| 必须通过 PR 合入 | 任何人（含 owner）不能 `git push origin main` 直推 |
| 1 个 approval | PR 必须有一个 approving review |
| Dismiss stale reviews | 新 commit push 后旧的 approval 失效，需要重新 review |
| 必须通过的 status checks | `pytest + ruff (py3.11)`、`pytest + ruff (py3.12)`、`Analyze (python)` —— 三个全绿才能 merge |
| Strict（up-to-date）| PR 分支必须先 rebase / merge main 才能合 |
| 必须解决所有 review conversation | 未 resolved 的 comment 会阻塞 merge |
| 禁止 force push | `git push -f` 被拒 |
| 禁止删除分支 | `main` 不能被删 |
| 不允许 bypass（含 admin）| owner 自己也得走 PR 流程 |

**触发场景**：
- `git push origin main` 直推 → `remote rejected`
- CI 红或 CodeQL 红 → Merge 按钮灰掉
- 强制 push / 删除 `main` → 被拒

**紧急绕过**：极少数情况下（例如 CI workflow 自身坏了导致永远不绿），临时在 Settings 里把 "Do not allow bypassing" 取消，merge 后立刻恢复。

---

## 4. Code security

公开仓库 GitHub 默认就开了一批安全功能，无需点：

| 功能 | 状态 | 作用 |
|---|---|---|
| Dependency graph | ✅ 默认 | 解析 `pyproject.toml`，构建依赖关系图，供 alerts / CodeQL 用 |
| Secret scanning + push protection | ✅ 默认（公开仓库） | 检测意外提交的 token / key（如 `tp-...`、AWS key），并在 `git push` 时拦截 |
| CodeQL code scanning | ✅ 通过 `.github/workflows/codeql.yml` | 静态分析 Python 代码的安全漏洞（注入、不安全反序列化等） |

发现的问题在 **Security** tab 列出，PR 里以注释形式提示。误报可 dismiss 并选原因，会被记住。

---

## 5. Dependabot 依赖更新

每周一会自动开 PR 升级依赖，**最多同时 5 个**。PR 命名格式 `chore(deps): bump foo from 1.2.3 to 1.3.0`。

分组策略：
- `langchain*` / `langgraph*` 合并成一个 PR（避免相互不兼容的小版本各开一个）
- `pytest*` / `ruff` / `mypy` 合并成一个 dev-tools PR
- 其余每个包独立 PR

**怎么处理**：
- 看一眼 changelog（PR description 里有）
- 等 CI 绿
- 如果是 minor / patch，直接 merge
- 如果是 major（比如 langgraph 2.x），认真看 breaking changes，可能要改代码

**安全更新（Dependabot alerts）** 独立于版本升级 PR，会立即开 PR 不受 weekly 限制。

---

## 6. 推荐协作流程

```bash
# 1. 起新分支
git checkout -b feat/short-name

# 2. 开发，本地跑测试
pytest -m "not slow"
ruff check paper2manim tests

# 3. push + 开 PR
git push -u origin feat/short-name
gh pr create --fill   # 或在 GitHub UI 上开

# 4. 等 CI 绿，若有 review 等 approve
# 5. Merge（Squash 推荐：保持 main 线性历史）
# 6. 删分支：本地 git branch -d，远端 GitHub UI 上点一下
```

---

## 7. Secrets 管理

- **绝不 commit `.env`**：`.gitignore` 已经覆盖
- 真实 API key 走 `Settings → Secrets and variables → Actions`
- workflow 里用 `${{ secrets.MIMO_API_KEY }}` 读取
- **PR 来自 fork 时，secrets 不会注入**（GitHub 安全策略）。意味着外部贡献者的 PR 跑不通需要 secrets 的 step——这是正常行为，不是 bug

---

## 8. 未来可加（按需）

| 想做 | 怎么加 | 何时 |
|---|---|---|
| Nightly 端到端真实跑 | 新 workflow，`on.schedule` + `workflow_dispatch`，注入 MIMO key | MVP 1.0 demo 稳定后 |
| 真实 Manim 渲染冒烟测试 | CI 里 `apt install texlive-latex-extra` + 跑 `@pytest.mark.slow` | 当 mock 测试不够 |
| PR 自动 review | 安装 Claude GitHub App | PR 流量上来再说 |
| 自动发版 | `release-please` 或手动 tag | 准备发 v0.1.0 时 |
| 覆盖率报告 | `pytest --cov` + Codecov action | 想盯指标了再加 |

---

## 9. 当前状态与待办

### 已生效

| 项 | 在哪 | 作用 |
|---|---|---|
| CI workflow | `.github/workflows/ci.yml` | push / PR 时跑 ruff + 46 个 mock 测试，py3.11/3.12 双 matrix |
| CodeQL workflow | `.github/workflows/codeql.yml` | push / PR + 每周一基线扫描 Python 安全漏洞 |
| Dependabot config | `.github/dependabot.yml` | 周更 pip / 月更 actions，分组 PR |
| Branch protection (main) | Settings → Branches | 见第 3 节：必须 PR + 3 个 check 全绿 + 1 approval + 不允许 bypass |
| Workflow 默认权限 | Settings → Actions → General | 只读，且禁止 workflow 自动 approve PR |
| Dependency graph + Secret scanning + push protection | Settings → Code security（公开仓库默认） | 见第 4 节 |

### 还没开（建议补上）

| 项 | 在哪 | 为什么要开 |
|---|---|---|
| **Dependabot alerts** | Settings → Code security → Dependabot alerts → Enable | 没开的话依赖里有 CVE 你也不会知道；这是 Dependabot 安全更新的前提 |
| **Dependabot security updates** | 同上 → Dependabot security updates → Enable | CVE 出现时自动开 PR 升级，**不受**每周 5 个 PR 上限的限制 |
| **合并策略收紧** | Settings → General → Pull Requests | 现状是 merge commit / squash / rebase 全允许；建议**只留 Squash**，并勾上 "Automatically delete head branches"，强制 main 线性历史 |

### 不影响功能但可选

| 项 | 说明 |
|---|---|
| 迁移到 Rulesets | 把现有 Branch protection 换成新版 Ruleset。功能等价，多出"叠加 / dry-run / 管 tag"等能力。当前没必要，等需要管 tag 或多分支策略时再迁 |
| Required linear history | 如果第二条待办（只留 squash）做了，这个就自动满足；如果想保留 rebase 也允许，可单独开启这条 |
| 私有仓库的 Secret scanning | 当前是公开仓库自动开；如果将来转私有，需要 GitHub Advanced Security（付费） |
