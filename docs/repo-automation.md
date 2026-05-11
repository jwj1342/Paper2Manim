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
- **pytest -m "not slow"**：跑 mock 测试套件（当前 46 个），**失败则 PR 红**

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

## 3. 分支保护规则（**需要手动在 GitHub Settings 启用**）

GitHub Actions 文件本身只定义"跑什么"，**"不通过不能合"是分支保护规则**，必须在仓库 Settings 里点一次。建议配置：

**路径**：`Settings → Branches → Branch protection rules → Add rule`

**规则名**：`main`

勾选：

- [x] **Require a pull request before merging**
  - [x] Require approvals: **1**（个人项目可设 0，但建议至少 1）
  - [x] Dismiss stale pull request approvals when new commits are pushed
- [x] **Require status checks to pass before merging**
  - [x] Require branches to be up to date before merging
  - 必选 checks（在 PR 跑过一次后这里才会出现选项）：
    - `pytest + ruff (py3.11)`
    - `pytest + ruff (py3.12)`
    - `Analyze (python)` (CodeQL)
- [x] **Require conversation resolution before merging**
- [x] **Do not allow bypassing the above settings**（包括 admin，强烈建议）
- [ ] Require signed commits（可选，HPC 环境配 GPG 签名较麻烦，跳过）
- [ ] Require linear history（可选；想强制 rebase 合并就开）

**Restrict who can push to matching branches**：勾上并清空允许列表 → 任何人（包括 owner）都必须通过 PR 才能动 `main`。

### 效果

- 直接 `git push origin main`：**被拒绝**（remote rejected）
- PR 红了：**Merge 按钮灰掉**
- 强制 push：**被拒绝**

### 紧急绕过

如果真的需要紧急修复（例如 CI 本身坏了导致永远红），有两条路：
1. 临时在 Settings 把"Do not allow bypassing"取消，merge 后立刻恢复
2. 在另一个 PR 里修复 CI，让它先合（如果 CI 失败的不是 CI 自己，没救只能走 1）

---

## 4. CodeQL 安全扫描

第一次启用需要在 `Settings → Code security and analysis` 里：
- 开启 **Code scanning** → 选 "Default" 或 "Advanced"（我们用 advanced，即 workflow 文件版）

发现的问题在 **Security tab** 列出，PR 里也会以注释形式提示。

误报很常见，可以在 dismiss 时选原因（"False positive" / "Used in tests" 等），dismiss 会被记住。

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

## 9. 一次性手动配置 checklist

第一次设置仓库时按顺序点：

- [ ] `Settings → Code security and analysis` → 开 Dependabot alerts / Dependabot security updates / Code scanning
- [ ] 第一次 push 触发 CI 跑一遍，让 `pytest + ruff (py3.11)` 等 check 名字出现
- [ ] `Settings → Branches → Add branch protection rule` for `main`（见第 3 节）
- [ ] `Settings → Actions → General` → "Workflow permissions" 设为 "Read repository contents and packages permissions"（最小权限原则）
- [ ] （可选）`Settings → Pull Requests` → 关掉 "Allow merge commits"，只保留 Squash，强制 linear history
