# P2M-Bench: Dataset Design for Paper2Manim

## 1. 数据集结构

### 1.1 两层结构：paper context + target_unit

每条样本以 `(paper_id, target_unit)` 为主键。同一篇论文可派生多条局部任务，例如 *Attention Is All You Need* 可派生 `Scaled Dot-Product Attention`、`Multi-Head Attention`、`Architecture Figure` 三条。模型生成时只拿到：

```text
paper_full_text
+ target_unit
+ scene_role
+ domain / source_type 等非答案型元数据
+ required_prior_context（仅在需要前置视觉状态时）
```

这一结构与 `PaperState` 基本对齐：`paper_full_text` 对应 `state["full_text"]`，`target_unit` 对应当前可视化任务。`main_topics`、`key_claims`、`reference_scene_plan`、完整 rubric 等字段只属于评测 / 展示端，不能进入模型输入或 EMB。

### 1.2 主表字段（HuggingFace parquet）

| 字段 | 类型 | 必填 | 含义 |
|---|---|---|---|
| `id` | str | 是 | `<paper_id>_<unit_kind>_<unit_index>`，主键 |
| `category` | enum | 是 | `Concept / Equation / Algorithm / Figure / Architecture / Experiment` |
| `domain` | enum | 是 | `cs / math / physics / quantum / econ` |
| `paper_id` | str | 是 | arXiv id（兼容 `hep-th/9901001` 旧式 id）或本地 PDF SHA-1 |
| `paper_title` | str | 是 | 论文标题 |
| `paper_publish_date` | date | 是 | 数据污染分析用；`cross_domain` / `test` 优先采用 2025-01 之后发表的论文 |
| `paper_license` | str | 是 | arXiv / publisher license；用于 HF dataset card 和分发权说明 |
| `paper_full_text` | str | 是 | 清洗后的论文文本；LaTeX 优先，回退 Marker PDF |
| `target_unit` | struct | 是 | 见 [§1.3](#13-target_unit-结构) |
| `source_type` | enum | 是 | 主论文使用 `section`；扩展支持 `subsection / equation / figure / table / algorithm / full_paper` |
| `scene_role` | enum | 是 | `BACKGROUND / METHOD / EXPERIMENT / CONCLUSION`；对齐论文中的 task role `r`，也进入 EMB embedding |
| `track` | enum | 是 | `local / global`；Hero Plot 主干只使用 `track=local`，`global` 独立成次要 track |
| `main_topics` | list[str] | 是 | 3-5 个核心知识点；评测 / 展示字段，不进入 Storyboarder prompt 或 EMB query |
| `split` | enum | 是 | `bootstrap / steady_state / cross_domain / test` |
| `is_human_gold_candidate` | bool | 否 | 是否可进入 output-level human scoring sidecar；不是 task split |
| `task_idx` | int | stream only | 主论文唯一冻结任务序号；A/B/C 配置必须共用 identical task stream |
| `task_idx_easy` | int | optional | 附录 curriculum 轨迹，按 `easy_to_hard` 冻结 |
| `task_idx_blocked` | int | optional | 附录领域分块轨迹；先跑一个领域再切到另一个领域，用于展示 domain shift |
| `task_idx_interleaved` | int | optional | 附录鲁棒性轨迹；按 `(domain, source_type)` stratified shuffle 冻结 |
| `difficulty` | enum | 是 | `easy / medium / hard`，用于分层报告 |
| `isomorphic_pair_id` | str \| null | 否 | 跨域同构任务对 id；方法论文正文展示 5-10 对 qualitative cases |
| `isomorphism_type` | enum \| null | 否 | `full / partial`；至少 1/3 为 `partial`，避免过干净的 cherry-pick |

`category` 表示可视化意图，`source_type` 表示原文形态。二者不是 1:1：例如 Transformer 架构图可同时是 `source_type=figure` 与 `category=Architecture`。

### 1.3 `target_unit` 结构

```json
{
  "title": "Scaled Dot-Product Attention",
  "text": "<原文摘录, <=4K chars>",
  "type": "section | subsection | equation | figure | table | algorithm | full_paper",
  "anchor": {
    "section": "3.2.1",
    "equation_label": "eq:scaled-dot",
    "figure_label": "fig:transformer-arch",
    "page": 3
  },
  "required_prior_context": "",
  "prior_objects": []
}
```

`anchor` 字段均为 optional，按 `type` 取需要的子集。主论文实验只要求 `type=section` 且文本长度 `<=4K chars`；其它类型用于后续更细粒度 benchmark。`required_prior_context` 和 `prior_objects` 只在 `algorithm` 或多 scene 片段需要前置视觉状态时填写，用来避免模型因缺失上下文而被错误扣分；普通 section / subsection / equation 可留空。

### 1.4 每条样本的完整标注

```json
{
  "id": "1706.03762_method_3_2_1",
  "category": "Concept",
  "domain": "cs",
  "split": "steady_state",
  "is_human_gold_candidate": true,
  "task_idx": 12,
  "task_idx_easy": 12,
  "task_idx_blocked": 31,
  "task_idx_interleaved": 47,
  "difficulty": "medium",
  "scene_role": "METHOD",
  "paper_id": "1706.03762",
  "paper_title": "Attention Is All You Need",
  "paper_publish_date": "2017-06-12",
  "paper_license": "arXiv.org perpetual, non-exclusive license",
  "track": "local",
  "target_unit": {
    "title": "Section 3.2.1 Scaled Dot-Product Attention",
    "text": "We call our particular attention 'Scaled Dot-Product Attention' ...",
    "type": "section",
    "anchor": { "section": "3.2.1" },
    "required_prior_context": "",
    "prior_objects": []
  },
  "isomorphic_pair_id": null,
  "isomorphism_type": null,
  "main_topics": ["QK^T", "softmax scaling", "weighted sum"],
  "key_claims": [
    "QK^T measures pairwise similarity between queries and keys",
    "Scaling by sqrt(d_k) prevents softmax saturation at large d_k",
    "The output is a weighted sum of V with weights from softmax"
  ],
  "reference_scene_plan": [
    { "beat": "Show Q, K, V as three colored matrix blocks" },
    { "beat": "Compute QK^T, visualize as similarity heatmap" },
    { "beat": "Divide by sqrt(d_k), highlight scaling effect on softmax" },
    { "beat": "Apply softmax, then weight-sum with V" }
  ],
  "evaluation_rubric": {
    "logic_flow": ["Beats appear in causal order"],
    "layout_occlusion": ["Matrices are labeled and non-overlapping"],
    "accuracy": ["softmax denominator is sqrt(d_k), not d_k"]
  }
}
```

`key_claims`、`evaluation_rubric`、`reference_scene_plan` 由研究生标注者独立完成；LLM 只能作为草标助手，所有 LLM 草标必须经至少两名人工逐条 review 后进入 ground truth。

### 1.5 Ground truth 的边界

P2M-Bench 不提供唯一标准 Manim 视频。它提供的是论文理解材料、局部可视化锚点、关键信息点、参考分镜和评分 rubric。

`reference_scene_plan` 的 storyboard 对齐度作为独立指标报告，不进入 `Pass@1` 阈值。正文可用一句话描述为：weighted sum of order alignment (Kendall tau) and embedding similarity (Hungarian matching)。具体权重、embedding 模型和实现细节放附录或 release notes。

### 1.6 `full_paper` 的独立 track

`source_type=full_paper` 与 paper-section、equation、figure、algorithm 等局部任务不是同质样本。它要求模型把整篇论文压成多 scene 叙事，失败模式会混入 context overflow、长程规划衰减、渲染超时等因素。为避免 Hero Plot 被这种异类任务放大噪声，主实验约束为：

- Track 1: Local Tasks。主论文只使用 paper-section tasks；equation、figure、algorithm、table、subsection 是兼容扩展，承载 RQ1 Hero Plot 与 RQ3 跨域实验时必须保持同一任务粒度。
- Track 2: Global Narrative。仅包含 `source_type=full_paper`，作为次要实验或 future work；若报告，只在冻结 EMB 后做 zero-shot，单独列结果。

`global` track 暂不定义 `narrative_coherence` / `symbol_consistency` 等专属评分维度。方法论文正文只需说明 global narrative generation 的设置与局限，完整长视频 benchmark 留给后续工作。

## 2. 实验需求映射

### 2.1 RQ × 字段映射

| Proposal | 需要的数据信号 | P2M-Bench 字段 |
|---|---|---|
| RQ1 Hero Plot | 有序 paper-section 任务流 | `split=steady_state`, `track=local`, `task_idx` |
| RQ1 bootstrap regime | EMB 为空时的预热流 | `split=bootstrap`, `task_idx` |
| RQ1 domain shift 副图 | 领域切换点明确的任务流 | optional `task_idx_blocked` |
| RQ1 鲁棒性附录 | 嘈杂任务流 | optional `task_idx_interleaved` |
| RQ2 VLM-Human 一致性 | 100 个 converged videos，三名专家逐维打分 | output-level sidecar `human_scores` |
| RQ2 维度对齐 | Logic / Layout / Accuracy 共用维度 | `evaluation_rubric` |
| RQ3 跨域泛化 | Domain A steady-state EMB → Domain B held-out set | `domain` + `split=cross_domain` |
| RQ3 structure mapping 案例 | 少量同构任务对 | `isomorphic_pair_id` |
| Ablations A-E | 同任务集多配置复跑 | `id`, `split`, runner preset |

主论文 Figure 3 使用单一 frozen task stream，A/B/C 配置必须共享 identical task stream。三条 curriculum 共用同一批样本，不增加标注成本，但属于附录鲁棒性分析：`task_idx_blocked` 用于展示领域切换后的 zero-shot 跃升或退化，`task_idx_interleaved` 用于报告嘈杂任务流下的稳定性，不能替代主论文的 `task_idx`。

同构任务对用于回应“跨域提升是否只是复用了代码片段”的质疑。dataset 维护一小组跨域同构案例，至少包含若干部分同构样本，方法论文正文展示 5-10 对案例和一张小表，不把它扩展成完整 benchmark 结论。

### 2.2 数据污染缓解

arXiv 论文很可能进入主流 LLM 预训练语料，P2M-Bench 必须显式报告污染风险。`cross_domain` / `test` 优先采用 2025-01 之后发表的论文，并通过 `paper_publish_date` 字段按 cutoff 前后分层报告结果。所有主表样本保留发表日期，论文正文需说明该策略只能缓解预训练污染，不能证明模型完全未见过论文内容。

### 2.3 与 EMB 字段对齐

P2M-Bench 字段可以映射到 EMB `Context`，让检索使用任务元数据加权：

| P2M-Bench | EMB `Context` |
|---|---|
| `target_unit.text` | `task_text` |
| `scene_role` | `scene_role` |
| `target_unit.type` | `scene_role` 的补充粒度，不替代主论文 role |
| `domain` | `domain_tags` |
| `paper_id` | `source_paper` |
| `target_unit.anchor.section` | `source_section` |

严禁把 `main_topics`、`key_claims`、`reference_scene_plan` 或 rubric 细则写入 EMB。它们是评测端 ground truth / 展示字段，进入检索会污染 RQ1 和 RQ3。

### 2.4 Loader 红线

实验 loader 必须物理隐藏评测字段。模型侧只暴露：

```text
paper_full_text
target_unit
scene_role
domain/source_type 等非答案型元数据
required_prior_context（仅在需要前置视觉状态时）
```

评测侧才可读取：

```text
main_topics
key_claims
reference_scene_plan
evaluation_rubric
human_scores
```

这条约束是本文与人工种子 skill-library 工作区分的关键：EMB 只能由系统在任务流上自学习生成，不能预置人工分镜或答案。

### 2.5 指标口径

| 指标 | 数据集提供 | 备注 |
|---|---|---|
| Task-level Pass@1 | 每个 task 的首轮 attempt trace + VLM 三维分 | 若首轮 attempt 的聚合 VLM 分数 `>= θpass`，该 task 记为 Pass@1；`θpass` 是报告指标阈值 |
| L-Pass@1 | `task_idx` + sliding window + task-level Pass@1 | 按最近 `w` 个 task 的 Pass@1 均值随累计任务数作图；bootstrap 单独报告，不进 headline steady-state window |
| Pass@K | `id` + reflection trace | 最多 `Tmax=5` 轮 |
| 平均反思轮数 | `id` | runner trace 统计 |
| VLM 平均分 | `evaluation_rubric` 三维 | Logic / Layout / Accuracy |
| Human-VLM 一致性 | sidecar `human_scores` | Pearson / Spearman / Cohen's kappa |
| Domain-Transfer Gain | `domain` x `split` | `cross_domain` 中比较 frozen EMB vs no EMB；可按 cutoff 前后和 difficulty 分层报告 |
| Human Ceiling | 20 条小样本专家完成任务 | 附录报告人类 Pass@1 / 三维均分 |

主论文草稿中 `θwrite=90` 用于 convergence 和 EMB 写入门槛；它不必与 `θpass` 绑定。若论文决定沿用同一数值，应写成 `θpass=θwrite=90`；若后续做人类校准，则只调整 `θpass`，不改变 EMB 写入门槛。

EMB 健康度指标属于方法评估章节，不放在 dataset doc 主体中。

## 3. Split 定义

| Split | 规模 | 用途 |
|---|---:|---|
| `bootstrap` | 50-100 | EMB 为空的预热流；单独报告，不进入 headline L-Pass@1 |
| `steady_state` | ~150 | Hero Plot 主曲线；仅 `track=local`，A/B/C 共用 identical task stream |
| `cross_domain` | 50 | Domain B held-out set；quantum / microeconomics 等未见域 |
| `test` | 50 | 最终保留集，不调参 |

`human_gold` 不是 task split，而是 generated output 的标注子集。主论文协议从 converged videos 中按 configuration × domain 分层抽样 100 条，由三名数学或计算机教师沿 Logic Flow / Layout-Occlusion / Accuracy 三轴独立盲评，并计算 Pearson / Spearman / Cohen's kappa。

为避免 ceiling effect，human scoring sidecar 必须为入选任务保存 `attempt_0`、`attempt_mid`、`attempt_final` 三阶段视频和逐维人工分数。主论文 Table 5 仍报告 converged videos；附录报告三阶段 attempt 相关性，展示 VLM-Human 一致性是否覆盖失败、中间态和收敛态。

Sidecar 结构示例：

```json
{
  "run_id": "exp_main_C_seed_0",
  "task_id": "1706.03762_method_3_2_1",
  "attempt": "final",
  "video_path": "runs/exp_main_C_seed_0/1706.03762_method_3_2_1/final.mp4",
  "human_scores": {
    "logic_flow": [90, 85, 88],
    "layout_occlusion": [82, 80, 85],
    "accuracy": [95, 92, 94]
  }
}
```

## 4. 标注、IAA 与发布合规

标注流程采用轻量方法论文规格：两名标注者独立标注 `key_claims`、`evaluation_rubric`、`reference_scene_plan`，第三人仲裁分歧。主论文的人类一致性实验沿用 100 converged videos 协议；dataset 标注本身可先抽 30-50 条 pilot 样本，将逐维评分离散化后计算 Cohen's kappa，并在附录说明仲裁流程。无需同时报告 Krippendorff alpha 和 Kendall tau。

发布时需要补齐：

| 项 | 位置 |
|---|---|
| 标注来源声明 | 正文 1 句 + 附录半页 |
| 数据污染缓解 | 正文 2-3 句 |
| License 说明 | `paper_license` 字段 + HF dataset card |
| Responsible NLP Checklist | 按 EMNLP 模板填写 dataset 相关问题 |
| Human Ceiling | 附录 20 条小样本表 |

不需要完整 Gebru-style Datasheet；Responsible NLP Checklist 足够覆盖方法论文提交要求。

## 5. 后续实现项

文档 merge 后需要单独 tracking：

| 项 | 目的 |
|---|---|
| Dataset loader | 物理隔离模型输入字段与评测字段 |
| Storyboard scoring | 附录级实现 storyboard alignment |
| HF dataset card | 说明 schema、license、污染缓解、Responsible NLP Checklist |
| Human scoring sidecar | 保存 output-level human subset 的 `attempt_0/mid/final` 和人工逐维分 |
| Curriculum generator | 生成并冻结 `task_idx_easy`、`task_idx_blocked`、`task_idx_interleaved` |
| Isomorphic-pair manifest | 维护一小组跨域同构 qualitative cases 及 `full/partial` 标记 |

## 6. 一句话总结

P2M-Bench 应该短而硬：用 paper-section 五元组支撑 RQ1/RQ2/RQ3 与 ablation，严格隔离 ground truth 字段，并补足数据污染、标注来源、IAA、license 和 human ceiling 这些审稿防御点。多 curriculum 与同构任务对只作为附录鲁棒性 / qualitative analysis，`full_paper` 独立成 global track；CLI/parser 细节和 EMB 健康度评测不放在 dataset doc 主体中。
