# Progress

> 更新时间：2026-05-11（合入合作者 fix 分支：static checker + 多 provider 抽象 + VLM 脚手架）

## 总览

| 模块 | 状态 | 备注 |
|---|---|---|
| 项目骨架 | 完成 | pyproject / .env / .gitignore / README / docs |
| 核心模块（state/llm/config/prompts/artifacts/cli） | 完成 | config 已包化（env.py + settings.py） |
| Sandbox（render/classify/concat） | 完成 | subprocess + rlimit + 静态预检（quality/manim_static_checker） |
| MVP 1.0 agents（storyboarder, coder） | 完成 | 端到端验证通过 |
| MVP 2.0 agents（summarizer, reviewer） | 完成 | 单测通过；待真实 PDF 验收 |
| MVP 3.0 脚手架（VLM critic + visual revision） | 已合入 | 模块就位；尚未接入 graphs/mvp2.py |
| Graphs（mvp1, mvp2） | 完成 | mvp2 含 should_retry / has_more_scenes 两个 conditional edges |
| Prompts（8 个） | 完成 | 外置在 `prompts/*.md`，含 global_system 共享头 |
| 多 provider 抽象（infrastructure/models, llm, vlm） | 合入待整合 | 与 main 的 `llm.py` 并存；后续在 D3 决议后统一 |
| 测试 | 完成 | 28/28 单测；slow 标记的真实渲染冒烟测已规划 |
| 端到端验收（MVP 1.0） | 完成 | pythagorean 输入 → 10.47s mp4 |
| 端到端验收（MVP 2.0） | 进行中 | 反思 cycle 单测已通过；待真实论文跑通 |

---

## Done

### 项目骨架与工程基础
- `pyproject.toml`（PEP 621，含 `mvp2` / `dev` extras）
- `.env.example`（详注释模板）+ `.env`（本地 key，gitignore）
- `.gitignore`（venv / runs / media / .env / MiMo-API.txt）
- `README.md`（研究导向；非 HPC 安装；详细 .env 说明；目录树）
- `docs/getting-started.md`（跨平台入门，包含常见问题排查）
- `docs/ResearchProposal.md`（研究提案原文）
- `requirements.txt`（pip freeze 锁定，71 个包）

### 核心模块
- `paper2manim/state.py` — LangGraph TypedDict `PaperState`，含 reflection 累积器（`Annotated[list[Attempt], operator.add]`）、per-scene 状态字段、控制位
- `paper2manim/llm.py` — MiMo 客户端工厂（OpenAI 兼容 base_url + flash/pro/v2 三个 alias，懒加载 settings）
- `paper2manim/config.py` — pydantic-settings 读取 `.env`
- `paper2manim/prompts.py` — 从 `prompts/*.md` 加载（lru_cache 但 hash 在每次进程内）
- `paper2manim/artifacts.py` — `runs/<run_id>/` 目录管理 + `trace.jsonl` 流水
- `paper2manim/logging_setup.py` — logging.basicConfig + 可选 LangSmith hook
- `paper2manim/cli.py` — click 子命令 `mvp1` / `mvp2` / `info`，含 `--no-render` / `--quality` / `--max-retries` / login-node guard

### Schemas
- `paper2manim/schemas/storyboard.py` — `SceneModel` / `StoryboardModel`（含 PascalCase 校验）
- `paper2manim/schemas/summary.py` — `SummaryModel` + `FormulaItem`
- `paper2manim/schemas/error_feedback.py` — `RenderResultModel` / `ErrorFeedback`

### Sandbox
- `paper2manim/sandbox/render.py` — `render(code, scene, *, quality, wall_timeout, cpu_seconds, mem_mb, fsize_mb)`，subprocess + `setrlimit`，强制 `--renderer=cairo` 适配无显示节点
- `paper2manim/sandbox/classify.py` — 错误分类：`python` / `latex` / `manim_runtime` / `timeout` / `unknown`，并提供 `extract_traceback_tail` / `excerpt_source` / `extract_tex_log_errors` / `first_error_line`
- `paper2manim/sandbox/concat.py` — ffmpeg concat demuxer 拼多 scene mp4，失败时回退到 re-encode

### Agents
- `paper2manim/agents/storyboarder.py` — 复用 `with_structured_output(StoryboardModel)`，兼容文本 / 摘要两种入口
- `paper2manim/agents/coder.py` — 含 `extract_python_block` 抓 ` ```python ` 块；prompt 里把上一轮失败的 code + structured error_feedback + tex log 全塞进去
- `paper2manim/agents/summarizer.py` — 长 markdown 自动切到 `mimo-v2.5-pro`（>30K chars 阈值）
- `paper2manim/agents/reviewer.py` — 短路 success；硬 cap 命中直接 give_up；其余调 LLM 出 `{decision, hint}` JSON

### Graphs
- `paper2manim/graphs/mvp1.py` — `storyboarder → coder → render → END` 线性拓扑
- `paper2manim/graphs/mvp2.py` — `parser → summarizer → storyboarder → init_scene → coder → render → reviewer → [retry: coder | advance: advance_scene] → [more: init_scene | done: concat] → END`，两个 conditional edges 实现反思闭环

### Prompts（外置，热加载）
- `prompts/storyboarder.md`（含 MVP 1 / 2 切换规则与一个 Pythagoras one-shot）
- `prompts/coder.md`（Manim 0.20 API 限制 + 反思迭代时如何结合 error_feedback）
- `prompts/manim_skill_rules.md`（Manim API 速查表，被 coder.md 引用）
- `prompts/summarizer.md`（输出 SummaryModel JSON 严格 schema）
- `prompts/reviewer.md`（决策启发：连续两次同类错 → give_up；含正反例）

### Tests（pytest）
- `tests/conftest.py` — 隔离 runs 目录、注入 mock `MIMO_API_KEY`、`mock_llm` fixture monkeypatch 所有 agents 的 `get_llm`
- `tests/test_classify.py`（9 用例：每类错误、tail 截取、行号定位、源码摘录）
- `tests/test_llm_client.py`（key 缺失报错、模型名验证、未知 alias）
- `tests/test_storyboarder.py`（结构化输出 + 缺输入报错）
- `tests/test_coder.py`（fence 抽取、no-fence 兜底、error_feedback 回灌进 prompt）
- `tests/test_graph_mvp1.py`（端到端 mock 跑通；render 成功 / skip 两条路径）
- `tests/test_graph_mvp2.py`（**反思闭环关键**：两次 latex error 后第三次成功；max_retries=1 give_up 回归）
- 当前结果：**25/25 通过**（含 3 条 CRITICAL fix 回归测试）

### 端到端验收
- **MVP 1.0** 真实跑通：`examples/mvp1/pythagorean.txt` → `runs/20260510-080654-3a1159/final/output.mp4`，时长 10.47s，854×480@15fps，h264，87.7 KB
  - LLM 端：MiMo `mimo-v2.5` 通过（storyboarder + coder 各 1 次调用）
  - Render 端：manim 0.20.1 + 系统 LaTeX → mp4，无任何 fallback

### 文档
- `README.md` — 研究导向、跨平台、目录树详注
- `docs/getting-started.md` — 详细入门（含 §5 常见问题与排查）
- `docs/progress.md` — 本文件
- `docs/ResearchProposal.md` — 研究提案

### 工程脚手架（HPC 友好但非必需）
- `scripts/setup_env.sh`、`install_tinytex.sh`、`render_node.sh`、`smoke_test.sh`
- 这些只是辅助脚本；非 HPC 用户走 `pip install -e ".[dev]"` 即可，无需 source 任何脚本

### 初次代码 review 后的 CRITICAL 修复
基于 Claude 端 review（codex 因 squid 拒 `auth.openai.com` 暂不可用）的 3 个 CRITICAL 问题已修：

- **C1**（`graphs/mvp2.py:render_node`）— 前序节点失败时 `state["storyboard"]` 直接 KeyError 崩栈。改为 `.get()` + 三层守卫（storyboard 缺失 / scene idx 越界 / current_code 空）→ 返回 `fatal_error` 而非 raise。
- **C2**（`state.py` + `graphs/mvp2.py`）— `fatal_error` 字段同时表达"全局致命"与"per-scene give_up"两种语义，`init_scene_node` / `advance_scene_node` 一律重置抹掉了 parser/summarizer/storyboarder 的真实致命错。改：
  - 新增 `state.skipped_scenes: Annotated[list[str], operator.add]` 字段记录单 scene 放弃
  - 删除 `init_scene_node` / `advance_scene_node` 的 `fatal_error: None` 重置
  - graph 在 parser → summarizer → storyboarder 三处加 `_is_fatal` conditional edge，`fatal_error` 一旦置位直接跳到 END
- **C3**（`agents/storyboarder.py` + `summarizer.py` + `llm.py`）— `with_structured_output(...).invoke(...)` 在 LLM schema drift 时直接抛 `OutputParserException` / `ValidationError`，整个 run 崩。新增 `llm.safe_structured_invoke()` helper（重试一次 + 严格 JSON 提示），storyboarder/summarizer 改用并将异常转为 `fatal_error`。

回归测试 3 条（`test_storyboarder_fatal_on_validation_error`、`test_mvp2_early_exit_on_parser_fatal`、`test_mvp2_render_node_guards_missing_storyboard`）已通过，单测总数 22 → 25。

剩余 9 个 HIGH 与 12 个 MEDIUM 见 To Do.F「工程债」。

### 合入合作者 fix 分支（issue #1 D1 / D3 / D4 + D2/D5 脚手架）

通过三次主题化 merge 把 `fix/api-client-config` 上的代码合入 main：

1. **D3 — 多 provider 模型抽象**（commit `22d833b`）：`paper2manim/infrastructure/models/` + `infrastructure/llm/client.py` + `config/settings.py`（YAML AppSettings）+ `config.example.yaml`。与 main 的 `paper2manim/llm.py` 并存，未替换 LangGraph 节点中的客户端调用。`config.py` 重构为 `config/` 包；`config/__init__.py` 同时 re-export 旧的 env 配置与新的 AppSettings，确保 `from paper2manim.config import AppSettings` / `from paper2manim.config.env import settings` 都通。
2. **D1 + D4 — 渲染前静态检查**（commit `113b791`）：`paper2manim/quality/manim_static_checker.py`，在 `sandbox/render.py` 进入 subprocess 前先做 AST 黑名单扫描（os/subprocess/socket/shutil/eval/exec/Path.write_text/Paragraph 等）+ 结构校验（必须 `from manim import *`、必须有 `Scene` 子类、必须 ≥2 个 `self.play(`）。失败转为 `StaticCheckError` 错误结果，不启动渲染子进程。D1 调整：解禁 `Tex` / `MathTex`（公式必须），保留 `Paragraph` 禁用（防 wall-of-text）。新增 3 条集成测试覆盖。
3. **D2 / D5 脚手架 — VLM 视觉评审**（commit `519308f`）：`paper2manim/infrastructure/vlm/`（VLMClient Protocol + Doubao/豆包 实现 + Mock + factory）、`paper2manim/agents/vlm_scene_reviewer.py`、`paper2manim/agents/visual_revision_agent.py`、`paper2manim/domain/models.py`（1005 行领域模型：SceneSpec / PaperVideoPlan / VisualReviewResult 等）、`paper2manim/utils/{prompt_loader,text_utils}.py`、新增 `prompts/{global_system,vlm_scene_reviewer,visual_revision_agent}.md`。**未接入 graphs/mvp2.py**，待 issue #1 的 D2 / D5 讨论收敛后再决定整合策略。

整合后回归：`pytest` **28/28 通过**；`paper2manim.{config.env,config,llm,state,graphs.mvp1,graphs.mvp2,sandbox.render,quality.manim_static_checker,utils.prompt_loader}` 全部 import 通过；`prompt_loader.PROMPT_DIR` 正确指向 repo-root `prompts/`（合并时调整为 `parents[2]`，避开与 `paper2manim/prompts.py` 模块同名）。

---

## In Progress

### MVP 2.0 真实论文验收
- 安装 `marker-pdf` + 预下载 ~3GB 模型权重（首次自动）
- 准备一篇真实论文 PDF（候选：Attention Is All You Need 的 Section 3 摘出来约 5 页 PDF）
- 端到端跑通 `paper2manim mvp2 --pdf <path>`，确认：
  - 反思 cycle 在真实场景下能触发并修复
  - concat 出来的多 scene mp4 可播放
  - `runs/<id>/trace.jsonl` 完整记录

---

## To Do

### A. MVP 2.0 收尾（短期）
- [x] **arXiv 源码路径**（方案 C）：`paper2manim/parsers/arxiv_source.py` + 分派 `parsers/__init__.py`；CLI 加 `--arxiv <id|url>` / `--section <name>`；state 增 `input_kind="arxiv"` + `arxiv_spec` / `arxiv_section` / `parsed_format` / `parser_source`；summarizer prompt 增加 LaTeX 输入分支；18 条新单测（id 解析 / flatten / 截节 / tarball 解压）。author 只传 PDF 时自动回退 Marker（`SourceUnavailable`）。
- [ ] 安装 marker-pdf 并预下载模型（仅当用 `--pdf` 或 arXiv 回退路径时需要）
- [ ] 端到端跑通：`paper2manim mvp2 --arxiv 1706.03762 --section "Scaled Dot-Product Attention"`（需能访问 arxiv.org）
- [ ] 端到端跑通：`paper2manim mvp2 --pdf examples/mvp2/<paper>.pdf`
- [ ] 同一论文的双路径对比：`--arxiv` vs `--pdf`，记录 storyboard / summary 差异
- [ ] 用一个故意 LaTeX 错的输入观察反思 retry 是否真触发并成功修复（也可手改 prompt 制造）
- [ ] 测量同样输入下：`max_retries=0`（disable reflection）vs `max_retries=3` 的成功率差异

### B. CLI / 工具改进（中期）
- [ ] `--resume <run_id>`：从 `runs/<id>/storyboard.json` 恢复，跳过已成功 scene
- [ ] `--scene <SceneName>`：只针对单个 scene 重跑 coder + render
- [ ] cost tracking：每次 LLM 调用累加 prompt/completion tokens，最后打印 + 写入 `runs/<id>/cost.json`
- [ ] 加 `runs/<id>/run.log` 把 logging 输出落盘
- [ ] MiMo 限速时的 retry/backoff（langchain 自带 retry，但需要 tune）

### C. Prompt 工程
- [ ] coder.md 加 negative examples（曾经出过的错；e.g. `\R` 替换 `\mathbb{R}`、`mobj.center` 替换 `mobj.get_center()`）
- [ ] reviewer.md 加 few-shot：成功案例 + 故意 give_up 案例
- [ ] summarizer.md 调到能稳定提 ≤5 个公式 + 配自然语言解释
- [ ] storyboarder 增加节奏感的提示（开场 / 主体 / 收尾）

### D. 评估基准（贡献三，长期）
- [ ] 收集 30+ 篇论文（CS / 数学 / 物理 / 生物各 ~7 篇）
- [ ] 定义"内容保真度"指标：LLM-as-judge + 抽样人工核验
- [ ] 定义"视觉连贯性"指标：渲染帧 + VLM Critic 打分
- [ ] 定义"代码可执行性"指标：Pass@1 / Pass@3
- [ ] 与 baseline 对比：(1) zero-shot 单 LLM；(2) 无反思的多 agent；(3) 完整 paper2manim
- [ ] 写论文实验章节

### E. MVP 3.0 探索（远期）
- [x] VLM Critic 脚手架（agents/vlm_scene_reviewer.py + infrastructure/vlm/）已合入，未接图
- [ ] 把 `vlm_scene_reviewer` + `visual_revision_agent` 接入 `graphs/mvp2.py` 的反思闭环（D2 / D5 决议后）
- [ ] 把 `paper2manim.domain` 的富领域模型与 `state.py` 的 `PaperState` 调和（沿用 TypedDict + reducer 还是改成 Pydantic）
- [ ] 把 `paper2manim.llm` 与 `paper2manim.infrastructure.llm.client` 收敛到单一客户端层（D3）
- [ ] 图表抽取：原图嵌入 `ImageMobject`；或用代码复现图表
- [ ] TTS 集成：ElevenLabs / 系统 TTS；按 storyboard 时间戳同步
- [ ] camera：MovingCameraScene 高级运镜
- [ ] 风格化：用户传入风格描述（"3Blue1Brown 风格"），影响配色 / 字体 / 节奏

### F. 工程债
- [ ] Pydantic UserWarning（`PydanticSerializationUnexpectedValue` from langchain-openai `with_structured_output`）— 无害但烦人
- [ ] mypy strict 跑通
- [ ] 单测覆盖率 → 80%（当前未量化，估计 60–70%）
- [ ] CI（GitHub Actions / GitLab CI）：lint + fast tests on PR
- [ ] CONTRIBUTING.md / CHANGELOG.md

---

## Known Issues

- **HPC 计算节点 squid 代理白名单**：在某些集群（如 Vulcan）的计算节点上，`api.xiaomimimo.com` / `chromium.googlesource.com` / `arxiv.org` 都会被代理拒绝（403）。表现：装 manim 失败（skia-pathops）+ 调 LLM 失败 + `--arxiv` 抓不到源码。
  - **绕开**：在登录节点（外网更宽松）装好 manim、跑 LLM 节点（parser / summarizer / storyboarder / coder / reviewer）；只在计算节点跑 manim render（不需要外网）。当前 CLI 还没拆分阶段执行（见 To Do B 的 `--resume <run_id>`）。
  - **`--arxiv` 失败模式**：`parser_node` 会把任何下载异常吞成 `fatal_error`，整图通过 `_is_fatal` 早退到 END，不会让进程崩。
- **Pydantic 序列化 UserWarning**：`langchain-openai==0.3.35` 配合 `pydantic==2.13` 调 `with_structured_output(Model)` 时会在调用栈里打印一个 `PydanticSerializationUnexpectedValue` 警告，不影响功能。已观察到在 storyboarder/summarizer 节点出现。
- **Marker 首跑慢 + 占空间大**：MVP 2.0 的 PDF 解析首次会下载 ~3GB 模型权重到 `~/.cache/huggingface/`。
- **Manim OpenGL 在无显示节点崩溃**：已通过强制 `--renderer=cairo` 解决，但若用户显式指定 `--renderer=opengl` 不在 sandbox 控制范围内（暂无该选项）。

---

## Acceptance Baseline（验收基线）

### MVP 1.0
- [x] `paper2manim mvp1 --input <file>` 端到端跑通
- [x] 输出 ≥ 10 秒、可播放、≥ 480p 的 mp4
- [ ] 5 个 demo 输入中至少 4 / 5 一次跑通（已跑 1 个，剩 4 个待验证）

### MVP 2.0
- [x] 反思 retry 单测通过（`test_mvp2_reflection_succeeds_after_two_retries`）
- [x] give_up 单测通过（`test_mvp2_reflection_gives_up_at_cap`）
- [ ] 1 篇真实论文端到端跑通，concat 出 ≥ 60 秒视频
- [ ] 反思 cycle 在至少一次真实失败上成功修复
