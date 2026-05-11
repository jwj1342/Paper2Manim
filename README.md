# Paper2Manim

> 探索"学术论文 → 高质量动画讲解视频"的端到端自动化路径。

本仓库是一项以"程序化教学动画生成 Agent 如何**摆脱失忆、持续进化**"为研究问题的代码实现。完整研究动机、四阶段进化框架与 Research Questions 见 **[`docs/ResearchProposal.md`](./docs/ResearchProposal.md)**。

## 路线图

| 阶段 | 状态 | 输入 | 处理 | 输出 | 核心研究主张 |
|---|---|---|---|---|---|
| **MVP 1.0** | 已验证 | 短文本（摘要 / 单定理） | Storyboarder → Coder → Render | 15–30 秒视频 | 验证最短端到端链路可行 |
| **MVP 2.0** | 进行中 | 完整 PDF | Parser(Marker) → Summarizer → Storyboarder → Coder ⇄ Reviewer 反思闭环 → Concat | 1–2 分钟多场景视频 | 反思机制对 Pass@1 的提升 |
| MVP 3.0 | 计划中 | 教学任务 + 检索到的历史经验 | + VLM-as-Judge 视觉打分 + Episodic Memory Bank 沉淀 + RAG 冷启动 | 持续进化的多场景视频 | **自进化机制 + 跨域记忆迁移**（详见提案 §4） |

**当前状态**：MVP 1.0 端到端已验证（LLM 生成代码 + 沙盒渲染均成功）。MVP 2.0 的全部 agents、graph 拓扑、反思 conditional edge 已实现并通过单测，待真实论文跑通验收。MVP 3.0 的 VLM Critic 与 visual revision 智能体脚手架已合入（见 `paper2manim/infrastructure/vlm/` 与 `paper2manim/agents/vlm_scene_reviewer.py`），尚未接入 `graphs/mvp2.py`，留待 issue #1 的 D2 / D5 讨论收敛后落地。详细进度与 To Do 清单见 [`docs/progress.md`](./docs/progress.md)。

> **新协作者请直接阅读 [`docs/getting-started.md`](./docs/getting-started.md)**——一份在普通笔记本 / 服务器上从零跑通的详尽入门指南，含三大平台依赖、API key 申请、第一个 demo、看视频、改 prompt、常见报错排查。

## 环境要求

| 依赖 | 用途 | 备注 |
|---|---|---|
| **Python ≥ 3.11, < 3.13** | 主语言 | 已在 3.11 上验证 |
| **ffmpeg** | Manim 视频编码 | 系统包管理器装即可 |
| **LaTeX**（推荐 TeX Live full / MacTeX / TinyTeX） | Manim 公式渲染 | Manim 文档列出的最小包集见下文 |
| **MiMo API key** | LLM 调用 | 从 [MiMo 控制台](https://www.xiaomimimo.com/) 申请 Token Plan，`tp-` 前缀 |
| 网络 | 调用 LLM API + 首次安装时编译 `skia-pathops` | 完全离线环境需要预编译 wheel |

按平台安装系统依赖：

```bash
# Ubuntu / Debian
sudo apt install ffmpeg python3.11 python3.11-venv \
    texlive texlive-latex-extra texlive-fonts-extra texlive-science \
    tipa cm-super dvisvgm

# macOS (Homebrew)
brew install ffmpeg python@3.11
brew install --cask mactex-no-gui   # or: install TinyTeX via R

# Windows
# 推荐用 WSL2 + Ubuntu 路径；原生 Windows 可走 conda + MikTeX，但未在本项目中验证。
```

无 sudo 的环境下，可装[用户级 TinyTeX](https://yihui.org/tinytex/)：
```bash
curl -sL https://yihui.org/tinytex/install-bin-unix.sh | sh
~/.TinyTeX/bin/x86_64-linux/tlmgr install \
    standalone preview doublestroke ms setspace rsfs relsize ragged2e \
    fundus-calligra microtype wasysym physics babel-english \
    cm-super xcolor amsmath amssymb dvisvgm
export PATH=$HOME/.TinyTeX/bin/x86_64-linux:$PATH
```

## 安装

```bash
# 1) 拉代码
git clone <this-repo> Paper2Manim
cd Paper2Manim

# 2) 创建并激活虚拟环境
python -m venv .venv
source .venv/bin/activate           # macOS / Linux
# .venv\Scripts\activate            # Windows PowerShell

# 3) 安装 paper2manim 与依赖
pip install --upgrade pip wheel
pip install -e .                    # MVP 1.0 所需
pip install -e ".[mvp2]"            # 加 MVP 2.0：Marker PDF 解析（首次会下载 ~3GB 模型权重到 ~/.cache/huggingface）
pip install -e ".[dev]"             # 加开发工具：pytest / ruff / mypy
```

> 提示：Manim 的 `skia-pathops` 依赖在 PyPI 上没有 Linux 预编译 wheel，会从源码编译并通过 git 拉 `chromium.googlesource.com` 上的 skia 子模块。首次安装请确保该域名可达，预计 5–10 分钟。

## 配置 API key 与环境变量

我们用 [`pydantic-settings`](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) 从项目根目录的 **`.env`** 文件读取配置。仓库提供了 [`.env.example`](./.env.example) 作为模板，按以下步骤生成本地配置：

```bash
cp .env.example .env
# 用任意编辑器打开 .env，把 MIMO_API_KEY 改成你自己的 tp- key
```

`.env` 已加入 `.gitignore`，不会被误提交。每个变量的含义都在 `.env.example` 里有详细注释，最关键的两个：

| 变量 | 必填？ | 说明 |
|---|---|---|
| `MIMO_API_KEY` | 必填 | MiMo Token Plan 的 `tp-` 前缀 key |
| `MIMO_BASE_URL` | 默认即可 | `https://token-plan-cn.xiaomimimo.com/v1`（**注意不是** `api.xiaomimimo.com`，那个端点不认 `tp-` key） |
| `PAPER2MANIM_DEFAULT_MODEL` | 默认即可 | `flash`（→ `mimo-v2.5`）/ `pro`（→ `mimo-v2.5-pro`）/ `v2`（→ `mimo-v2-pro`，备用） |
| `PAPER2MANIM_MAX_RETRIES` | 默认即可 | MVP 2.0 反思循环每个 scene 的最大重试轮数（默认 3） |
| `PAPER2MANIM_QUALITY` | 默认即可 | Manim 渲染质量 `l`/`m`/`h`（480p15 / 720p30 / 1080p60） |
| `PAPER2MANIM_RUNS_DIR` | 默认即可 | 运行产物目录，默认是项目根下的 `runs/` |

环境变量的读取在 import `paper2manim.config` 时一次完成。激活 venv 后，从项目根目录运行 CLI 即可，**不需要**手工 `export` 任何变量；`.env` 会被自动加载。

确认配置生效：
```bash
paper2manim info
# {"MIMO_BASE_URL": "...", "MIMO_API_KEY_set": true, ...}
```

## 使用方法

### MVP 1.0：短文本 → 单场景视频

输入是一段普通文本（一篇论文摘要 / 一个核心定理的描述）。系统会调用 LLM 生成 1 个 scene 的 storyboard 与 Manim 代码，然后在沙盒里渲染。

```bash
# 用一个示例输入跑通
paper2manim mvp1 --input examples/mvp1/pythagorean.txt

# 直接传文本字符串
paper2manim mvp1 --input "Newton's second law: F = m·a, applied to a sliding block."

# 提高视频质量（更慢）
paper2manim mvp1 --input examples/mvp1/fourier_intuition.txt --quality m

# 仅生成代码、跳过渲染（快速调试 prompt 时用）
paper2manim mvp1 --input examples/mvp1/eulers_identity.txt --no-render
```

### MVP 2.0：完整论文 → 多场景视频（带反思纠错）

输入两种来源任选其一：

1. **`--arxiv <id|url>` （推荐）**：直接抓 arXiv 上作者上传的 LaTeX 源码，自动展开 `\input{}` / `\include{}` 并去注释。公式 100% 准确（原始 LaTeX 而非 OCR），双栏顺序天然正确。可选 `--section <name>` 截单节（按 `\section{...}` 标题做大小写不敏感子串匹配）。作者只上传了 PDF 时自动回退到 Marker 解析 PDF。
2. **`--pdf <path>`**：本地 PDF，走 Marker（首次下 ~3GB 模型权重）。

得到论文文本后，Summarizer / Storyboarder 拆出 2–5 个 scene，逐 scene 让 Coder 写代码并由 Reviewer 在沙盒里检验，失败则带着结构化错误反馈让 Coder 修复，最多重试 `PAPER2MANIM_MAX_RETRIES` 次。最后用 ffmpeg 把成功的 scene 串成最终视频。

```bash
# arXiv 源码（全文）
paper2manim mvp2 --arxiv 1706.03762

# 只跑某一节（截 §3.2）
paper2manim mvp2 --arxiv 1706.03762 --section "Scaled Dot-Product Attention"

# arXiv URL 也行
paper2manim mvp2 --arxiv https://arxiv.org/abs/2401.12345v2

# 本地 PDF（Marker 路径）
paper2manim mvp2 --pdf path/to/your-paper.pdf

# 调整反思轮数 + 渲染质量
paper2manim mvp2 --arxiv 1706.03762 --max-retries 5 --quality m

# 只生成代码不渲染（验证 prompt + 反思逻辑）
paper2manim mvp2 --arxiv 1706.03762 --no-render
```

> 网络要求：`--arxiv` 需要能访问 `arxiv.org`。HPC 计算节点的 squid 代理可能拒绝该域名 → 在登录节点跑 `--no-render` 部分；渲染部分（不需要外网）再去 salloc 计算节点。

### 输出工件

每次运行会创建 `runs/<run_id>/`（`run_id` 是 `YYYYMMDD-HHMMSS-<6-hex>` 格式）：

```
runs/20260510-080654-3a1159/
├── input.txt                  # 输入备份
├── parsed.md                  # MVP 2.0：Marker 输出的 markdown
├── summary.json               # MVP 2.0：论文结构化摘要
├── storyboard.json            # 分镜脚本（含每个 scene 的 name / description / duration）
├── attempts/                  # 每轮尝试的代码 + 渲染结果
│   ├── 00_PythagorasIntro.py
│   ├── 00_PythagorasIntro.render.json
│   ├── 01_PythagorasIntro.py             # 反思后的第二轮（如果有）
│   └── ...
├── final/
│   └── output.mp4             # 最终视频
└── trace.jsonl                # 每个 graph 节点的 input/output 流水
```

调试时优先看：`storyboard.json`（LLM 怎么拆的剧本）→ `attempts/*.py`（生成的代码）→ `attempts/*.render.json`（结构化错误信息）→ `trace.jsonl`（节点级时间线）。

### 测试

```bash
pytest -m "not slow"     # 默认：mock LLM 与 mock render，~2s 跑完
pytest                   # 含 slow 标记的真实渲染冒烟测试
```

## 项目目录

仓库自顶向下的组织如下。每个目录的职责都做了清晰切分，方便协作者按兴趣深入。

```
Paper2Manim/
├── README.md                  ← 你正在看的文件
├── pyproject.toml             ← Python 包元数据 + 依赖（含 [mvp2] / [dev] extras）
├── requirements.txt           ← pip freeze 锁定的依赖（用于精确复现）
├── requirements-dev.txt       ← 开发工具子集（pytest / ruff / mypy）
├── .env.example               ← 环境变量模板（复制为 .env 后填 MIMO_API_KEY）
├── .gitignore                 ← 屏蔽 .venv / runs / media / .env / MiMo-API.txt
│
├── .github/                  ← GitHub 自动化配置
│   ├── workflows/ci.yml         pytest + ruff，push / PR 触发，py3.11/3.12
│   ├── workflows/codeql.yml     Python 安全扫描，PR + 每周基线
│   └── dependabot.yml           依赖自动升级（pip 周更，actions 月更）
│
├── docs/                      ← 项目文档
│   ├── ResearchProposal.md      研究提案：失忆 Agent → VLM-as-Judge + 情景记忆库 + 进化曲线
│   ├── getting-started.md       面向新协作者的非 HPC 入门指南（含常见问题）
│   ├── progress.md              当前进度 + To Do + 已知问题 + 验收基线
│   ├── graphs.md                LangGraph 拓扑可视化（Mermaid，GitHub 自动渲染）
│   ├── graphs/                  原始 .mmd 文件
│   └── repo-automation.md       CI/CD、分支保护、Code security、协作流程参考
│
├── paper2manim/               ← 主 Python 包（pip install -e . 后可 import）
│   ├── __init__.py
│   ├── cli.py                   click 命令行入口；子命令 mvp1 / mvp2 / info
│   ├── config/                  pydantic-settings 包
│   │   ├── env.py                 从 .env 读取运行时配置（MIMO_API_KEY 等）
│   │   └── settings.py            YAML-based AppSettings（多 provider 模型注册）
│   ├── llm.py                   MiMo (Xiaomi Token Plan) 客户端工厂
│   │                            含模型 alias：flash → mimo-v2.5, pro → mimo-v2.5-pro
│   │                            + safe_structured_invoke（C3 修复，schema drift 重试）
│   ├── state.py                 LangGraph 共享状态 TypedDict (PaperState)
│   │                            所有 agent 节点的输入输出契约都在这里
│   ├── prompts.py               从 prompts/*.md 读取 system prompt（热加载）
│   ├── artifacts.py             runs/<run_id>/ 目录管理 + trace.jsonl 流水
│   ├── logging_setup.py         logging + 可选 LangSmith hook
│   │
│   ├── agents/                  ← LLM 智能体（一个文件一个 agent）
│   │   ├── storyboarder.py        text/summary → Storyboard JSON（MVP 1.0）
│   │   ├── coder.py               scene + error_feedback → Manim 代码（MVP 1.0）
│   │   ├── summarizer.py          markdown → 关键贡献 / 公式 / 概念（MVP 2.0）
│   │   ├── reviewer.py            RenderResult → retry / give_up + hint（MVP 2.0）
│   │   ├── vlm_scene_reviewer.py  渲染帧 → VisualReviewResult（MVP 3.0 脚手架，未接图）
│   │   └── visual_revision_agent.py  失败的 VisualReview → 修订后代码（MVP 3.0 脚手架）
│   │
│   ├── parsers/                 ← 输入解析器
│   │   ├── __init__.py            分派器：parse_arxiv / parse_local_pdf；ParsedInput 命名元组
│   │   ├── text.py                MVP 1.0：plain text 直通
│   │   ├── arxiv_source.py        MVP 2.0：抓 arxiv.org/e-print/<id>，flatten \input，可选截 \section
│   │   └── marker.py              MVP 2.0：用 Marker 把 PDF 转 markdown（arxiv 无源码时的兜底）
│   │
│   ├── sandbox/                 ← Manim 代码的隔离执行 + 错误结构化
│   │   ├── render.py              subprocess 调 manim CLI；含 setrlimit + 静态预检
│   │   ├── classify.py            stderr 分类：python / latex / manim_runtime / timeout
│   │   └── concat.py              ffmpeg concat demuxer 拼多 scene 为单 mp4
│   │
│   ├── quality/                 ← 代码安全 / 静态检查
│   │   └── manim_static_checker.py  AST 黑名单（os/subprocess/eval/exec/Paragraph 等）
│   │                                + 结构校验（Scene 子类、≥2 self.play）
│   │                                在 render() 入口处先于 subprocess 拦截
│   │
│   ├── graphs/                  ← LangGraph 工作流定义
│   │   ├── mvp1.py                线性拓扑：storyboarder → coder → render → END
│   │   └── mvp2.py                含反思 conditional edge 的多 scene 拓扑
│   │                              （reviewer 后分流：retry → coder | advance → next scene）
│   │
│   ├── schemas/                 ← Pydantic 数据契约
│   │   ├── storyboard.py          SceneModel / StoryboardModel（PascalCase 校验）
│   │   ├── summary.py             SummaryModel + FormulaItem（MVP 2.0）
│   │   └── error_feedback.py      RenderResultModel / ErrorFeedback
│   │
│   ├── domain/                  ← 合作者引入的 DDD 领域模型（独立于 LangGraph state）
│   │   └── models.py              SceneSpec / PaperVideoPlan / VisualReviewResult 等
│   │                              当前为 MVP 3.0 VLM 子系统服务，未与 PaperState 融合
│   │
│   ├── infrastructure/          ← 多 provider 抽象层（合作者）
│   │   ├── models/                openai_compatible / mock 模型客户端 + factory + registry
│   │   ├── llm/                   LLMClient（基于 models/ 的高层包装）
│   │   ├── vlm/                   VLMClient Protocol + Doubao（豆包）+ Mock 实现
│   │   └── rendering/             另一份 manim 渲染器（与 sandbox/render.py 并存）
│   │
│   └── utils/                   ← 通用工具
│       ├── prompt_loader.py       从 repo-root prompts/ 加载 + 拼接 global_system.md
│       └── text_utils.py          ```python``` 块抽取 / JSON 抽取等
│
├── prompts/                   ← 所有 LLM 系统提示词（外置 / 热加载）
│   ├── global_system.md         所有 agent 共享的人设头（被 load_agent_prompt 拼接）
│   ├── storyboarder.md          剧本导演人设 + JSON schema + 一个 one-shot 示例
│   ├── coder.md                 Manim 工程师人设 + 0.20 API 限制 + 反思修复模板
│   ├── manim_skill_rules.md     Manim 0.20 API 速查（被 coder.md 引用）
│   ├── summarizer.md            论文摘要器（MVP 2.0）
│   ├── reviewer.md              代码审阅器（MVP 2.0），含 retry / give_up 启发
│   ├── vlm_scene_reviewer.md    渲染帧视觉评审 prompt（MVP 3.0）
│   └── visual_revision_agent.md 据视觉反馈修订代码 prompt（MVP 3.0）
│
├── config.example.yaml        ← AppSettings YAML 模板（多 provider 模型注册表）
│
├── tests/                     ← pytest 测试套件（46/46 通过）
│   ├── conftest.py              共享 fixtures（隔离 runs 目录、mock LLM）
│   ├── test_classify.py         错误分类用例 + 静态检查器集成（D1/D4）
│   ├── test_llm_client.py       MiMo client 边界条件（key 缺失、未知 alias）
│   ├── test_storyboarder.py     mock LLM 验证结构化输出
│   ├── test_coder.py            python 块抽取、error_feedback 回灌进 prompt
│   ├── test_graph_mvp1.py       端到端 mock：含 / 不含 render 两条路径
│   ├── test_graph_mvp2.py       反思闭环：两次失败后第三次成功 + max_retries give_up
│   └── test_arxiv_source.py     id 变体解析 / flatten / 截节 / tarball 解压（18 用例）
│
├── examples/                  ← 输入样例
│   ├── mvp1/                    5 个固定短文本 demo（pythagorean / fourier / euler / newton / linear-regression）
│   └── mvp2/                    （留空，由用户放入真实论文 PDF）
│
├── scripts/                   ← 可选辅助脚本（普通用户用不到，仅 HPC 加速用）
│   ├── setup_env.sh             模块加载 + venv 创建 + pip install 一条龙
│   ├── install_tinytex.sh       无 sudo 装用户级 LaTeX
│   ├── render_node.sh           sbatch 模板（Slurm 集群批量渲染）
│   └── smoke_test.sh            最小 demo 冒烟脚本
│
└── runs/                      ← 运行时产物（gitignore；每次运行一个子目录）
    └── <run_id>/                YYYYMMDD-HHMMSS-<6hex>
        ├── input.{txt,pdf}
        ├── parsed.md             ← MVP 2.0 only
        ├── summary.json          ← MVP 2.0 only
        ├── storyboard.json
        ├── attempts/             ← 每个 scene 的代码 + 渲染结果
        │   ├── 00_<Scene>.py
        │   ├── 00_<Scene>.render.json
        │   └── ...
        ├── final/output.mp4      ← 最终视频
        └── trace.jsonl           ← graph 节点级流水
```

**速查：我想 X，应该看哪里？**

| 你想 ... | 去看 |
|---|---|
| 跑第一个 demo | [`docs/getting-started.md`](./docs/getting-started.md) |
| 改 LLM 行为 | `prompts/<agent>.md`（不需要重装） |
| 加新 agent | `paper2manim/agents/`、`paper2manim/state.py`、`paper2manim/graphs/mvp2.py` |
| 改沙盒资源限制 / 错误分类 | `paper2manim/sandbox/render.py` 与 `classify.py` |
| 知道当前进度 / 待办 | [`docs/progress.md`](./docs/progress.md) |
| 看研究背景 | [`docs/ResearchProposal.md`](./docs/ResearchProposal.md) |
| 看 graph 拓扑图 | [`docs/graphs.md`](./docs/graphs.md) |
| 了解 CI / 分支保护 / PR 流程 | [`docs/repo-automation.md`](./docs/repo-automation.md) |
| 排查一次失败的运行 | `runs/<run_id>/storyboard.json` → `attempts/*.py` → `attempts/*.render.json` → `trace.jsonl` |

## 与协作者的工作约定

- **不要把 `.env` 或 `MiMo-API.txt` 提交到仓库**——这两个文件已被 `.gitignore` 屏蔽。
- 改 prompt 不必改代码：`prompts/*.md` 是热加载的，diff 友好。
- 加新 agent 时遵循已有 pattern：`agents/<name>.py` 写一个 `xxx_node(state) -> dict` 函数；在 `graphs/mvpN.py` 里挂到 `StateGraph` 上；把它对外的输入输出字段加到 `state.py` 的 `PaperState`。
- 跑一次复杂 PDF 之前先用 `--no-render` 验证 LLM 端的产物（节省渲染开销）。
- **`main` 受分支保护**：不能直推，必须开 PR，CI（`pytest + ruff` py3.11/3.12 + CodeQL）三个 check 必过才能合。详细说明见 [`docs/repo-automation.md`](./docs/repo-automation.md)。

## 参考与致谢

- [Manim Community Edition](https://www.manim.community/) — 数学动画引擎
- [LangGraph](https://www.langchain.com/langgraph) — 多智能体工作流编排
- [Marker](https://github.com/VikParuchuri/marker) — 学术 PDF → markdown 解析
- 设计上受 [Manimator](https://arxiv.org/abs/2507.14306)、[Code2Video](https://arxiv.org/abs/2510.01174)、[manim-generator](https://github.com/makefinks/manim-generator) 等工作启发
