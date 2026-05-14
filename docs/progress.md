# Progress

> 更新时间：2026-05-13（图执行从串行改并行——`fan_out_scenes` Send×N + render/LLM throttles；测试基线 189/189）
>
> 历史里程碑：MVP 2.0 真实论文验收 → YAML 多 provider + 多模态标识 → MVP 3.0 阶段 2/3 VLM 接图（`docs/vlm_experiment.md`）→ proposal §4.2 canonical schema 落地 → 阶段 4 EMB 后端落地（双通道 success/failure + RAG 注入 + 蒸馏）→ 图并行计算。

## 总览

| 模块 | 状态 | 备注 |
|---|---|---|
| 项目骨架 | 完成 | pyproject / .env / .gitignore / README / docs |
| 核心模块（state/llm/config/prompts/artifacts/cli） | 完成 | `llm.py` 走 YAML `config.yaml` → `ModelSettings` 路由，env-mimo 兜底；config.yaml 存在但解析失败时**显式抛 RuntimeError**而非静默回退 |
| Sandbox（render/classify/concat） | 完成 | subprocess + rlimit + 静态预检（quality/manim_static_checker） |
| MVP 1.0 agents（storyboarder, coder） | 完成 | 端到端验证通过 |
| MVP 2.0 agents（summarizer, reviewer） | 完成 | 反思 cycle 单测 + 真实论文端到端验收均通过 |
| MVP 3.0 阶段 2/3（VLM 多维评分接图） | 完成 | `agents/vlm_scene_reviewer` + `visual_revision_agent` + `utils/frame_sampler` 已接入 `graphs/mvp2.py`；CLI `--vlm` 开启；评分 schema = proposal §4.2 canonical 3 维 × 0-100（logic_flow / layout_occlusion / accuracy）+ avg≥90 auto-pass bypass；mock-VLM 闭环单测 8 条 + JSON 解析单测 10 条；真实实验数据见 `docs/vlm_experiment.md` |
| MVP 3.0 阶段 4（EMB / 自进化）后端 | 完成 | `paper2manim/emb/`：dual-channel schema（success / failure 各自 body 共享 context+provenance 头）+ SQLite store（provenance-keyed dedup）+ Faiss / in-memory index + sentence-transformers / HashEmbedder + `EpisodicMemoryBank` facade + `distill.consolidate_run` + `retrieval.retrieve_for_scene`；CLI `--emb` 开启 |
| 图计算并行化 | 完成 | `graphs/scene_graph.py` 新建 per-scene 子图（`SceneState` TypedDict + emb_retrieve / coder / render / reviewer / frame_sampler / vlm_review / visual_revise 完整闭环）；父图 `mvp2.py` 改 `fan_out_scenes` 发 N 个 `Send`，N 个 scene 并发跑；`concurrency.py` 提供 `RENDER_SEMAPHORE` + LLM `TokenBucket`；CLI `--scene-parallelism` / `--render-concurrency` / `--llm-rps`；默认全关 = 串行 = 行为零差异 |
| Graphs（mvp1, mvp2） | 完成 | mvp1 保留串行；mvp2 改 fan-out（`parser → summarizer → storyboarder → Send×N run_scene → concat → emb_consolidate`），per-scene 包含完整反思 + VLM + EMB 检索闭环 |
| Prompts | 完成 | 外置在 `prompts/*.md`，含 `vlm_scene_reviewer.md` + `visual_revision_agent.md` + `global_system.md` |
| 多 provider + 多模态标识（YAML） | 完成 | `config.example.yaml` 模板（所有字段留空 + `$ENV_VAR` 引用）；`ModelConfig.supports_vision` / `auth_style` / `omit_temperature` 字段；`get_llm(role)` + `get_vlm()` 双入口；`provider` ∈ {openai_compatible, anthropic}，anthropic 支持 bearer auth（Azure Claude）；canonical roles ＝ {global_reader, scene_planner, scene_coder, render_fixer, final_summarizer, visual_reviser, vision_checker}，legacy alias（flash/pro/v2/v2-omni）继续可用 |
| 测试 | 完成 | **189/189** 单测全绿（pytest -q）；含 `test_concurrency`（24）+ `test_llm_proxy`（13）+ `test_artifacts_concurrency`（4）+ `test_scene_graph`（17）+ EMB 套件 + 既有 VLM/parser/graph 测试 |
| 输入解析（arXiv 源码 + 本地 PDF 兜底） | 完成 | `parsers/arxiv_source.py` + 分派 `parsers/__init__.py`；18 条新单测；SourceUnavailable 自动回退 Marker |
| CI/CD（GitHub Actions） | 完成 | `.github/workflows/ci.yml`（pytest + ruff, py3.11/3.12 matrix）+ `codeql.yml`（每周 + 每次 PR） |
| 分支保护 + Dependabot | 完成 | main 强制 PR + 3 个 check 必过 + 禁 force push；Dependabot 周更 pip / 月更 actions |
| 端到端验收（MVP 1.0） | 完成 | pythagorean 输入 → 10.47s mp4（`runs/20260510-080654-3a1159/`，磁盘保留） |
| 端到端验收（MVP 2.0） | 完成 | arXiv `1706.03762 §Background` → 5 scenes 全成功 → concat 79.0s mp4（`runs/20260512-213215-4d4572/`，runs/ 已 gitignore，本地未保留） |
| 端到端验收（MVP 3.0 阶段 2/3） | 完成 | 同一输入 + `--vlm` → 5 scenes × 3 reviews（含 2 visual revisions）→ concat 69.1s mp4（`runs/20260512-221822-08f182/`，runs/ 已 gitignore，本地未保留）；评分趋势见 `docs/vlm_experiment.md`；Claude Opus 4.7 复跑同输入 → 84.4s mp4，0 fatal_error |
| 端到端验收（图并行） | 进行中 | `--scene-parallelism>1` + `--llm-rps` + `--render-concurrency` 已可端到端运行；并发等价性、scene 失败隔离、render semaphore enforcement 在单测层全覆盖；并行下相对串行的实际端到端时延加速比待补一组对照实验 |

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
- `paper2manim/state.py` — LangGraph TypedDict `PaperState`，含 reflection 累积器（`Annotated[list[Attempt], operator.add]`）、per-scene 状态字段、控制位（含 `vlm_enabled` / `vlm_revision_count` / `max_visual_revisions` / `current_montage_path` / `last_visual_review` / `visual_revision_decisions`）
- `paper2manim/llm.py` — **YAML 多 provider 工厂**：`config.yaml` 存在则按 canonical role（global_reader / scene_planner / scene_coder / render_fixer / final_summarizer / visual_reviser / vision_checker）路由到 `openai_compatible` 或 `anthropic` provider；legacy alias（flash/pro/v2/v2-omni）通过 `_LEGACY_ALIAS_TO_ROLE` 兼容旧 agent 代码；无 config.yaml 时走 env-MiMo 兜底（`MIMO_API_KEY`）；config.yaml 存在但解析失败时**显式抛 RuntimeError**（不静默回退，避免供应商被悄悄切换）；`safe_structured_invoke` 用 `method="function_calling"` 同时兼容 Doubao Ark + OpenAI + Anthropic；`get_vlm()` 强制 `supports_vision=true`；支持 `omit_temperature` 兼容 Claude Opus 4.7（temperature 被弃用）；`_seed_env_from_dotenv` 让 YAML 里的 `$ENV_VAR` 引用能解析 `.env` 中未在 pydantic-settings 字段声明的 key（如 `AZURE_CLAUDE_API_KEY`）
- `paper2manim/config/` 包 — 两层，均 production-active：`config/env.py` 用 pydantic-settings 读 `.env`（含 `MIMO_*` / `LLM_*` / `VLM_*` 兼容字段，全部 default=""），出口 `settings` 单例；`config/model_config.py` 定义 `ModelConfig`（name / provider / model / base_url / api_key / supports_vision / auth_style / omit_temperature）+ `ModelSettings`（model_roles dict + `model_for_role()` 查表）；`config/config_loader.py` 读 `config.yaml` + `$ENV_VAR` 展开 + role 表校验（`vision_checker` 必须 `supports_vision=true`）。**已清理**：D3 merge 时引入的 `config/settings.py`（`AppSettings` / 第三条配置轨道，~322 行死代码）已删除，`config/__init__.py` 收敛到只暴露 env + model_config 两套出口
- `paper2manim/prompts.py` — 从 `prompts/*.md` 加载（lru_cache 但 hash 在每次进程内）
- `paper2manim/artifacts.py` — `runs/<run_id>/` 目录管理 + `trace.jsonl` 流水
- `paper2manim/logging_setup.py` — logging.basicConfig + 可选 LangSmith hook
- `paper2manim/cli.py` — click 子命令 `mvp1` / `mvp2` / `info`，含 `--no-render` / `--quality` / `--max-retries` / `--vlm` / `--no-vlm` / `--max-visual-revisions`（mvp2，default=2）/ `--pdf` 或 `--arxiv <id|url>` / `--section <name>` / `--allow-render-on-login` / login-node guard；mvp2 启动时根据 `max_retries × max_visual_revisions × n_scenes` 估算 recursion_limit 给 LangGraph（最小 200）

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
- `paper2manim/agents/summarizer.py` — 长 markdown 自动切到 `pro`/`final_summarizer`（>30K chars 阈值）
- `paper2manim/agents/reviewer.py` — 短路 success；硬 cap 命中直接 give_up；其余调 LLM 出 `{decision, hint}` JSON
- `paper2manim/agents/vlm_scene_reviewer.py` — proposal §4.2 canonical 3 维评分（`logic_flow` / `layout_occlusion` / `accuracy`，0-100 分），返回 `{scene_id, decision: pass|revise|fail, raw_decision, scores, average_score, revision_instruction}`；`_extract_first_json_object` 用平衡花括号扫描兼容"思考链 + JSON"双输出 LLM；`parse_vlm_response` 内置 **avg ≥ 90 auto-pass bypass**：VLM 说 revise 但 3 维均分 ≥ 90 → 升级为 pass，`raw_decision` 保留原始判决供 trace 审计
- `paper2manim/agents/visual_revision_agent.py` — 根据 VLM 的 `revision_instruction` 重写 Manim 源码；异常时回退原 code，不中断 graph

### Graphs
- `paper2manim/graphs/mvp1.py` — `storyboarder → coder → render → END` 线性拓扑
- `paper2manim/graphs/mvp2.py` — `parser → summarizer → storyboarder → init_scene → coder → render → reviewer → [retry: coder | frame_sampler（VLM 开启时）| advance] → [revise: visual_revise → render | advance] → [more: init_scene | done: concat] → END`，两套交错闭环：
  - **文本反思闭环**：render 失败 → reviewer 判 retry/give_up，cap=`max_retries`
  - **VLM 视觉反思闭环**：render 成功 → frame_sampler 抽 4 帧 hstack 成 montage → vlm_review 3 维评分 + auto-pass bypass → revise 时回到 visual_revise→render，cap=`max_visual_revisions`
  - `post_vlm_route` 对 `decision="fail"` 采用 **soft-fail 策略**（不剔出 rendered_videos，避免最终视频为空，因为 Claude 极少自发判 pass，见 `docs/vlm_experiment.md` §4.2）
  - 三个上游 conditional edges（`parser`/`summarizer`/`storyboarder`）一旦置 `fatal_error` 直接早退到 END

### Prompts（外置，热加载）
- `prompts/storyboarder.md`（含 MVP 1 / 2 切换规则与一个 Pythagoras one-shot）
- `prompts/coder.md`（Manim 0.20 API 限制 + 反思迭代时如何结合 error_feedback）
- `prompts/manim_skill_rules.md`（Manim API 速查表，被 coder.md 引用）
- `prompts/summarizer.md`（输出 SummaryModel JSON 严格 schema；含 LaTeX / Markdown 双输入分支提示）
- `prompts/reviewer.md`（决策启发：连续两次同类错 → give_up；含正反例）
- `prompts/global_system.md`（全局系统 prompt，被 VLM 相关 agent 共用）
- `prompts/vlm_scene_reviewer.md`（3 维 × 0-100 评分严格 schema + decision 启发 + 每维 0/100 锚点定义）
- `prompts/visual_revision_agent.md`（根据 VLM revision_instruction 修改 Manim 源码的 prompt）

### Tests（pytest）
- `tests/conftest.py` — 隔离 runs 目录、固定 `LLM_PROVIDER=mimo` + `MIMO_API_KEY=tp-test-key`、把 YAML loader 指向不存在路径（确保测试不被本地 dev `config.yaml` 干扰）、`mock_llm` fixture monkeypatch 所有 agents 的 `get_llm`
- `tests/test_arxiv_source.py`（**18 用例**：7 种 id/URL 变体含 `hep-th/9901001` + flatten `\input` / `\include` + strip 注释保留 `\%` + 大小写不敏感截节 + tarball 解压 + 非法 PDF blob 拒绝）
- `tests/test_classify.py`（**11 用例**：5 类错误分类 + tail 截取 + 行号定位 + 源码摘录 + 3 条静态预检集成测试：禁 forbidden import / 允许 Tex+MathTex / 禁 Paragraph）
- `tests/test_llm_client.py`（**4 用例**：env-MiMo 兜底 key 缺失 / 返回 ChatOpenAI 类型 / 未知 alias 报错 / **malformed config.yaml 显式抛 RuntimeError**）
- `tests/test_storyboarder.py`（**3 用例**：结构化输出 + 缺输入 fatal + ValidationError 转 fatal_error）
- `tests/test_coder.py`（**4 用例**：fence 抽取、no-fence 兜底、error_feedback 回灌、storyboard 路径）
- `tests/test_graph_mvp1.py`（**2 用例**：端到端 mock 跑通；render 成功 / skip 两条路径）
- `tests/test_graph_mvp2.py`（**4 用例**：反思闭环两次 latex error 后第三次成功；max_retries=1 give_up；parser fatal 早退；render_node 缺 storyboard 守卫）
- `tests/test_graph_mvp2_vlm.py`（**8 用例**：pass 短路 / revise→pass / 触 cap / vlm 关闭 / VLM 判 fail 不剔出 rendered_videos（soft-fail）/ vlm_review 抛异常时 auto-pass / 跨 scene 混合 verdict / **avg ≥ 90 auto-pass bypass 端到端**）
- `tests/test_vlm_response_parse.py`（**10 用例**：两个 JSON 对象输入只取第一个 / 嵌套花括号 / 字符串内花括号 / 不平衡输入返回 None / chatty 模型先思考再 JSON / 缺失维度记 None / [0,100] clamp / avg ≥ 90 auto-pass / avg < 90 不 auto-pass / 全空 scores 不 auto-pass）
- 当前结果：**65/65 通过**（pytest -q，ruff clean）

### 端到端验收
- **MVP 1.0** 真实跑通：`examples/mvp1/pythagorean.txt` → `runs/20260510-080654-3a1159/final/output.mp4`，时长 10.47s，854×480@15fps，h264，87.7 KB
  - LLM 端：MiMo `mimo-v2.5` 通过（storyboarder + coder 各 1 次调用）
  - Render 端：manim 0.20.1 + 系统 LaTeX → mp4，无任何 fallback

### 文档
- `README.md` — 研究导向、跨平台、目录树详注
- `docs/getting-started.md` — 详细入门（含 §5 常见问题与排查）
- `docs/progress.md` — 本文件
- `docs/ResearchProposal.md` — 研究提案（VLM-driven episodic memory evolution 已重聚焦，含 Failure Pattern Memory 4b dual-channel EMB）
- `docs/vlm_experiment.md` — MVP 3.0 阶段 2/3 首次真实端到端实验记录（arXiv 1706.03762 §Background × Azure Claude Sonnet 4.6 × cap=2）
- `docs/graphs.md` + `docs/graphs/{mvp1,mvp2}.mmd` — LangGraph 拓扑可视化（Mermaid 源码 + GitHub 自动渲染）
- `docs/repo-automation.md` — CI/CD、分支保护、Code security、Dependabot、协作流程的状态参考

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
   > **后续清理**：`config/settings.py`（`AppSettings` / `ProviderDefaults` / `VLMConfig` / `VisualReviewConfig` / `load_settings`）在后续 audit 中被发现零外部调用——PR #11 落地 YAML 路由后被 `config/model_config.py` + `config/config_loader.py` 完全替代。整个文件 + `__init__.py` 的相关 re-export 已删除（resolves config-debt D4 / D13 / D14）。`infrastructure/llm/client.py` 和 `infrastructure/models/` 仍待清理。
2. **D1 + D4 — 渲染前静态检查**（commit `113b791`）：`paper2manim/quality/manim_static_checker.py`，在 `sandbox/render.py` 进入 subprocess 前先做 AST 黑名单扫描（os/subprocess/socket/shutil/eval/exec/Path.write_text/Paragraph 等）+ 结构校验（必须 `from manim import *`、必须有 `Scene` 子类、必须 ≥2 个 `self.play(`）。失败转为 `StaticCheckError` 错误结果，不启动渲染子进程。D1 调整：解禁 `Tex` / `MathTex`（公式必须），保留 `Paragraph` 禁用（防 wall-of-text）。新增 3 条集成测试覆盖。
3. **D2 / D5 脚手架 — VLM 视觉评审**（commit `519308f`）：`paper2manim/infrastructure/vlm/`（VLMClient Protocol + Doubao/豆包 实现 + Mock + factory）、`paper2manim/agents/vlm_scene_reviewer.py`、`paper2manim/agents/visual_revision_agent.py`、`paper2manim/domain/models.py`（1005 行领域模型：SceneSpec / PaperVideoPlan / VisualReviewResult 等）、`paper2manim/utils/{prompt_loader,text_utils}.py`、新增 `prompts/{global_system,vlm_scene_reviewer,visual_revision_agent}.md`。**未接入 graphs/mvp2.py**，待 issue #1 的 D2 / D5 讨论收敛后再决定整合策略。

整合后回归：`pytest` **28/28 通过**；`paper2manim.{config.env,config,llm,state,graphs.mvp1,graphs.mvp2,sandbox.render,quality.manim_static_checker,utils.prompt_loader}` 全部 import 通过；`prompt_loader.PROMPT_DIR` 正确指向 repo-root `prompts/`（合并时调整为 `parents[2]`，避开与 `paper2manim/prompts.py` 模块同名）。

> **后续演进**：D2/D5 的 Doubao VLM client 已在 PR #11 中被替换为 `openai_compatible` + `anthropic` 双 adapter（详见下面 `PR #11 — multi-provider LLM 工厂 + VLM 接图正式 land` 段）。这条历史记录保留以备追溯，但 `infrastructure/vlm/doubao_vlm_client.py` 文件本身已不再存在。

### arXiv 源码解析路径（方案 C）

`paper2manim/parsers/arxiv_source.py`：抓 `arxiv.org/e-print/<id>` 返回的 tarball / single-gz / gzipped-PDF 三种 blob 形态；递归展开 `\input{}` / `\include{}`（max_depth=6）；去注释保留 `\%`；按 `\section{...}` 大小写不敏感子串截单节。`paper2manim/parsers/__init__.py` 暴露 `parse_arxiv()` / `parse_local_pdf()` 分派器，arXiv 抓不到源码（`SourceUnavailable`）时自动回退 Marker 解析 PDF。CLI 加 `--arxiv <id|url>` / `--section <name>`；state 增 `input_kind="arxiv"` + `arxiv_spec` / `arxiv_section` / `parsed_format` / `parser_source`；summarizer prompt 增加 LaTeX 输入分支提示（让模型忽略 `\label / \ref / \cite / preamble`）。`graphs/mvp2.py` 的 `parser_node` 改为分派 + 把任何下载异常吞成 `fatal_error`，由 `_is_fatal` 早退到 END。新增 18 条单测覆盖 id 解析（7 种变体含 `hep-th/9901001`）、flatten、截节、tarball 解压。`docs/graphs.md` + `docs/graphs/{mvp1,mvp2}.mmd` 落 LangGraph 拓扑可视化。

### GitHub CI/CD + 分支保护

三个 workflow + Dependabot + 分支保护一次落到位：

- `.github/workflows/ci.yml`：push / PR 时跑 ruff + pytest（`-m "not slow"`），py3.11/3.12 双 matrix；concurrency cancel；apt 装 `libcairo2-dev libpango1.0-dev pkg-config ffmpeg`（manimpango 需要 pangocairo C 扩展）；注入 `MIMO_API_KEY=tp-fake-ci-key` 防止 CI 误调真 API
- `.github/workflows/codeql.yml`：Python 静态安全扫描，push / PR + 每周一基线
- `.github/dependabot.yml`：周更 pip / 月更 actions；langchain* 与 dev-tools 分组合并 PR；目前已成功跑通 3 个 actions 升级 PR（checkout 4→6 / setup-python 5→6 / codeql-action 3→4）
- Classic Branch protection on `main`：必须 PR + 3 个 status check（`pytest + ruff (py3.11)` / `(py3.12)` / `Analyze (python)`）全绿 + strict（up-to-date）+ dismiss stale + 必须解决所有 review conversation + 禁 force push + 禁删除分支
- Code security（公开仓库自动 / 手动开）：Dependency graph、Secret scanning + Push protection、Dependabot alerts、Dependabot security updates、Grouped security updates、Private vulnerability reporting、Copilot Autofix 全开
- 文档：`docs/repo-automation.md` 把上述配置以"已设了什么 + 作用 + 还差什么"的形式记录下来；ruff 顺手在合并代码库上修了 42 个旧 lint 问题（F401 / I001 / UP037 等）

### PR #11 — multi-provider LLM 工厂 + VLM 接图正式 land（commit `2335236`）

这是一次主题相关、分四个 commit 推进的整合（首个 commit 走 env-`PROVIDER_TABLE` 思路，被后续 commit 收敛到 YAML 路由）：

1. **`feat(llm)` — multi-provider 工厂**（中间形态，被后续 commit 替换）：把 `paper2manim/llm.py` 从 MiMo-only `ChatOpenAI` 推广到 mimo / deepseek / doubao / openai 四个 env-PROVIDER_TABLE。`tests/test_llm_client.py` 从 3 条扩到 13 条；`tests/conftest.py` 固定 `LLM_PROVIDER=mimo` 防止本地 dev `.env` 干扰。
2. **`feat(vlm)` — MVP 3.0 阶段 2/3 VLM 接图**：`graphs/mvp2.py` 接入 `frame_sampler → vlm_review → visual_revise → render` 闭环（cap=`--max-visual-revisions`），文本反思闭环不变。同时把 `llm.py` 从 env-PROVIDER_TABLE **重写为 YAML 优先**（`config.yaml` → `ModelConfig` → role 表）+ env-MiMo 兜底，`ModelConfig` 加 `supports_vision` / `auth_style ∈ {header_api_key, bearer}` 支持 Azure-hosted Claude 走 `Authorization: Bearer` header。`infrastructure/vlm/`：拆出 `openai_vlm_client.py` + `anthropic_vlm_client.py`，**删掉 `doubao_vlm_client.py`**（被 `openai_compatible` 通用 adapter 覆盖）。CLI 加 `--vlm` / `--max-visual-revisions`，并把 LangGraph `recursion_limit` 提到 200 给双闭环留余量。50/50 测试通过。
3. **`fix(llm)` — Claude Opus 4.7 兼容**：Azure-hosted Opus 4.7 拒绝 `temperature` 参数（400 invalid_request_error: deprecated）。`ModelConfig` 加 `omit_temperature: bool = False`，在 `_build_yaml_client`（openai_compatible + anthropic）和两个 VLM client 三处 call site 都条件性跳过 temperature kwarg。Opus 4.7 端到端复跑 arXiv 1706.03762 §Background → 5/5 scenes、14 attempts、concat 84.4s mp4、0 fatal_error。
4. **`fix` — Copilot review hits**：
   - `agents/vlm_scene_reviewer.py`：旧的 `r"\{.*\}"` greedy regex 在 LLM 输出 "thinking trace JSON + 真答案 JSON" 双对象时会拼成非法 JSON 进 `json.loads`，然后退化到 `_conservative_review`。换成平衡花括号扫描 `_extract_first_json_object`（跟 depth、跳过双引号内的 `{` / `}`）。
   - `llm.py`：`config.yaml` 存在但解析失败时，原来打 warning 然后静默回退 env-MiMo，会把流量悄悄切换到完全不同的供应商。改为**抛 RuntimeError**，让用户立刻看到 typo 或 `$ENV_VAR` 错引；只有 `config.yaml` **不存在**时才走 env-MiMo 兜底。
   - 新增 5 条 `tests/test_vlm_response_parse.py` + 1 条 `test_malformed_config_yaml_raises_loudly`，总数 50/50 → 56/56。
5. **`fix(ci)` — CI 绿 + VLM robustness**：`pyproject.toml` 显式声明 `anthropic` / `langchain-anthropic`（CI 之前在 `factory.py` 急加载 `AnthropicVLMClient` 时 ImportError 失败）；再加 3 条 mvp2_vlm robustness 测试（fail decision soft-pass / vlm 抛异常 auto-pass / 跨 scene 混合 verdict）。最终 59/59 通过。

整合带来的关键状态变化：
- `paper2manim/infrastructure/vlm/` 现在只有 `{client.py, openai_vlm_client.py, anthropic_vlm_client.py, mock_vlm_client.py, factory.py, __init__.py}`，`doubao_vlm_client.py` 已退役
- `tests/` 总数：**46 → 59**（+5 vlm_response_parse, +3 mvp2_vlm robustness, +1 malformed config, +4 重组 llm_client）
- `config.yaml` 现在是 YAML 路由的**唯一入口**，存在即必须可解析；不存在则走纯 env-MiMo 路径

### PR #16 — MVP 3.0 阶段 4 EMB 后端 land

proposal §4.4 描述的双通道 Episodic Memory Bank 后端整套落地：

- **`paper2manim/emb/schema.py`** — Pydantic v2 dual-body schema：success/failure 共享 `(context, provenance)` 头，body 按 polarity 分（`SuccessBody` 含 rationale + final_code；`FailureBody` 含 lesson + anti-/good-example pair + score delta）
- **`paper2manim/emb/store.py`** — SQLite store，provenance-keyed dedup（`source_paper + source_section + scene_id + transition_ordinal`）防止重复跑同一 section 时 EMB 单调膨胀
- **`paper2manim/emb/index/`** — Faiss-or-fallback in-memory；sentence-transformers `all-MiniLM-L6-v2` + HashEmbedder 兜底（首次启动 / CI / 无网络场景）
- **`paper2manim/emb/manager.py`** — `EpisodicMemoryBank` facade：`add_success / add_failure / query` 加 `hit_count` / `last_used` / `first_seen` provenance 字段（issue #17 cold-record pruning 即基于此）
- **`paper2manim/emb/distill.py`** — `consolidate_run`：扫一次 `trace.jsonl` + `attempts/`，按 `theta_high`（默认 4.0）筛 success records、按 `failure_min_margin`（默认 0.5）筛 failure transitions，调用注入的 `rationale_writer` / `lesson_distiller`（LLM 或 mock）写 body
- **`paper2manim/emb/retrieval.py`** — `retrieve_for_scene`：scene 描述向量 → top-k success + top-k failure → `to_state_dict()` 输出 wire-format
- **`graphs/mvp2.py`** — `emb_retrieve` 节点（先 per-scene，PR #19 后下放到 scene 子图）+ `emb_consolidate` 节点（父图末端）
- **`agents/coder.py`** — `## Reference Examples` / `## Known Pitfalls` 注入位置（夹在 `## Project conventions` 与 `## Previous attempt failed` 之间）
- **CLI** — `--emb` / `--emb-store-path` / `--emb-theta-high` / `--emb-failure-min-margin` / `--emb-use-llm-distillers`

未做、open issue 跟进：
- [#17 §9 Cold record pruning by hit_count / last_used](https://github.com/jwj1342/Paper2Manim/issues/17)
- [#18 [EMB] A/B protocol for RAG injection position in Coder prompt](https://github.com/jwj1342/Paper2Manim/issues/18)

### PR #19 — 图执行从串行改并行

把 MVP 2.0 per-scene 循环抽到 compiled sub-StateGraph，父图通过 `langgraph.types.Send` fan-out N 个 scene 同时跑：

- **`paper2manim/graphs/scene_graph.py`**（新）— `SceneState` TypedDict + 完整 per-scene 闭环（`emb_retrieve` → `coder` ↔ `render` ↔ `reviewer` → `frame_sampler` → `vlm_review` ↔ `visual_revise`）
- **`paper2manim/graphs/mvp2.py`**（重写）— `parser → summarizer → storyboarder → fan_out_scenes → [Send×N → run_scene] → concat → emb_consolidate`。删 `init_scene_node` / `advance_scene_node`，初始化挪到 `_make_scene_payload`，成功视频汇总挪到 `run_scene_node` 的 SceneState→PaperState reducer
- **`paper2manim/concurrency.py`**（新）— `RENDER_SEMAPHORE`（限并发 Manim 子进程）+ `TokenBucket`（限 LLM RPS）；两者都是 module-global，`None` 时纯 passthrough 零开销
- **`paper2manim/sandbox/render.py`** + **`paper2manim/llm.py`** — `render()` 调 `render_slot()`；`get_llm` / `get_vlm` 包成 `RateLimitedLLM` 透明代理（gated 在 `invoke / ainvoke / stream / batch / with_structured_output / __or__ / bind_tools`）
- **`paper2manim/artifacts.py`** — `append_trace` 加 `threading.Lock`，并发写 `trace.jsonl` 不交错
- **`paper2manim/state.py`** — 加 `scene_reports` reducer；`vlm_revision_count` / `last_visual_review` / `current_montage_path` 这三个变量保留在 PaperState 是给 mvp1 串行路径，mvp2 fan-out 不再读写它们（迁到 SceneState）
- **CLI** — `--scene-parallelism` / `--render-concurrency` / `--llm-rps`，**默认全关 = 串行 = 行为零差异**
- **Tests** — `test_concurrency` / `test_llm_proxy` / `test_artifacts_concurrency` / `test_scene_graph` 共 35 条新增

Schema 收敛：`visual_revision_decisions` 从 `list[str]` 改成 PR #15 的 `list[dict[str, str]]`，scene 子图同步 emit 新形态。

---

## In Progress

- **图并行的端到端时延加速比**：单测层等价性已覆盖；真实多 scene 论文下 `--scene-parallelism={1,3,5}` × `--llm-rps={off,1,3}` 的对照实验待补，写到 `docs/parallelism_experiment.md`

---

## To Do

### A. MVP 2.0 收尾（短期）
- [x] **arXiv 源码路径**（方案 C）：`paper2manim/parsers/arxiv_source.py` + 分派 `parsers/__init__.py`；CLI 加 `--arxiv <id|url>` / `--section <name>`；state 增 `input_kind="arxiv"` + `arxiv_spec` / `arxiv_section` / `parsed_format` / `parser_source`；summarizer prompt 增加 LaTeX 输入分支；18 条新单测（id 解析 / flatten / 截节 / tarball 解压）。author 只传 PDF 时自动回退 Marker（`SourceUnavailable`）。
- [x] 端到端跑通：`paper2manim mvp2 --arxiv 1706.03762 --section "Background"`（5 scenes 一次过 → 79s mp4，`runs/20260512-213215-4d4572/`）
- [ ] 安装 marker-pdf 并预下载模型（仅当用 `--pdf` 路径时需要）
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

### E. MVP 3.0 探索

#### 阶段 2/3 已完成（VLM 反思闭环）
- [x] VLM Critic 脚手架（agents/vlm_scene_reviewer.py + infrastructure/vlm/`{openai,anthropic,mock}_vlm_client.py`）
- [x] 把 `vlm_scene_reviewer` + `visual_revision_agent` 接入 `graphs/mvp2.py` 的反思闭环（节点：`frame_sampler` → `vlm_review` → `visual_revise` → `render` → ...，cap 由 `--max-visual-revisions` 控制）
- [x] YAML 多 provider + `supports_vision` flag（`config.example.yaml` + `ModelConfig`），双 provider：`openai_compatible` / `anthropic`（含 Azure bearer 与 `omit_temperature` 兼容 Opus 4.7）
- [x] frame_sampler：ffmpeg 抽 N 帧 hstack 成 montage PNG
- [x] mock-VLM 闭环单测（**7 条**：pass 短路 / revise→pass / 触 cap / vlm_enabled=False 跳过 / fail decision soft-pass / vlm 抛异常 auto-pass / 跨 scene 混合 verdict）
- [x] VLM 输出 JSON 解析鲁棒性（10 条：双 JSON 对象只取首个 / 嵌套花括号 / 字符串内花括号 / 不平衡输入 / chatty 模型先思考再答案 / 缺失维度记 None / [0,100] clamp / avg ≥ 90 auto-pass / avg < 90 不 auto-pass / 全空 scores 不 auto-pass）
- [x] config.yaml 解析失败 fail-loud 而非静默回退（防止流量被悄悄切换供应商）
- [x] 真实实验：5 scenes / Azure Claude Sonnet 4.6 / cap=2，平均 Δavg=+0.20，详见 `docs/vlm_experiment.md`
- [x] Opus 4.7 端到端验证：5/5 scenes, 14 attempts, concat 84.4s mp4, 0 fatal_error

#### 阶段 4（EMB / 自进化）— **暂不考虑**
> Proposal §4 + §7 的核心创新点（情景记忆库 + 知识蒸馏 + 进化曲线）保留作为未来工作的描述锚点；本仓库当前**不实现** EMB / 检索 / High-Score Rationale 写入。任何 PR / commit 中提及"自进化"或"EMB"仅作 proposal 引用，不构成实现承诺。

#### 阶段 2/3 已知不足（在 `docs/vlm_experiment.md` 详述）
- [x] ~~评分 schema 对齐 proposal~~ — issue #12 已解：收敛为 3 维（logic_flow / layout_occlusion / accuracy，0-100）
- [x] ~~阈值化 pass~~ — issue #12 顺带做：`parse_vlm_response` 内 avg ≥ 90 → auto-upgrade 为 pass，raw_decision 保留供 trace 审计
- [ ] **best-of-N 保留**：当前 visual_revise 闭环只保留最后一版；遇到 v1>v2 的情况（如旧 baseline TitleIntro 中 v1=2.83, v2=2.50），最终输出反而劣化
- [ ] **VLM 与渲染分辨率耦合**：layout_occlusion 维度在 480p15 下容易被压低，VLM 让 coder 改 layout 也救不回 → 未来需要把渲染质量从 schema 里解耦

#### 其他探索（长期）
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
- [x] CI（GitHub Actions）：lint + fast tests on PR（见 `.github/workflows/ci.yml`、`docs/repo-automation.md`）
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
- [x] 1 篇真实论文端到端跑通，concat 出 ≥ 60 秒视频（arXiv `1706.03762 §Background` → 79.0s mp4，5 scenes）
- [ ] 反思 cycle 在至少一次真实失败上成功修复（实测中 Claude Sonnet 4.6 一次过 5/5，没触发文本反思 retry —— 该项需用更弱模型或人为注入错误验证）

### MVP 3.0 阶段 2/3（VLM 反思闭环）
- [x] mock-VLM 闭环单测通过（`tests/test_graph_mvp2_vlm.py` 8 条：pass / revise→pass / cap / vlm 关闭 / fail soft-pass / 异常 auto-pass / 混合 verdict / avg ≥ 90 auto-pass）
- [x] VLM JSON 解析鲁棒性单测通过（`tests/test_vlm_response_parse.py` 10 条）
- [x] VLM 在真实场景下成功被调用并落盘 3 维评分 + montage（旧 6 维 baseline `runs/20260512-221822-08f182/`，新 3 维 baseline 见 `docs/vlm_experiment.md`）
- [x] 至少一个 scene 的 visual revision 真正改进评分（旧 6 维 baseline TakeawayConclusion: 2.17 → 2.83，Δ=+0.67；新 3 维 baseline 见 `docs/vlm_experiment.md` §3）
- [x] 完整 5 scenes × cap=2 闭环跑完不崩 + concat 输出可播放 mp4（Sonnet 4.6: 69.1s；Opus 4.7: 84.4s）
