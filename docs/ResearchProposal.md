# Research Proposal

**面向教学动画生成的自进化多智能体系统：基于 VLM 反馈的情景记忆库**

> Self-Evolving Multi-Agent System for Educational Animation Generation via VLM-Driven Episodic Memory

> 更新：2026-05-13 — §4 取消人工种子库的设定，改为统一管线下的自学习 EMB（详见 §4.1 / §4.4）；冷启动重新定义为 EMB 规模较小的早期段，代码路径与稳态期一致；§8.1 数据集相应调整为 paper-section 任务池；§8.3 Ablation C 调整为 bootstrap 批次大小消融。

---

## 1. Motivation：失忆的打工人

当前用 LLM 生成教学/解释性动画代码（Manim 等程序化动画）的工作，无论是 zero-shot 单模型方案，还是基于多智能体反思（如 Code2Video、manim-generator）的复杂管线，本质上都是**一个"失忆的打工人"**：每次拿到"解释傅里叶变换"这种需求，系统都要从零开始猜测怎么排版、怎么分镜、怎么避免遮挡、怎么把 LaTeX 写对。即使这一轮通过反复试错最终成功，**下一轮遇到结构相似的"解释概率分布"任务时，它依然会重蹈覆辙——犯一模一样的排版错、写一模一样的崩溃代码**。

换句话说，现有方法在**单任务内**（intra-task）已经做到了"反思-修正"的闭环，但在**任务之间**（inter-task）没有任何知识沉淀。每一个成功视频背后那些"哪个公式该左对齐""动画过渡多久才不晕"的可迁移经验，在 run 结束的那一刻就被永久丢弃了。

我们认为，**这正是把 LLM Agent 用于程序化视觉创作时最大的浪费，也是最大的研究机会**。

## 2. 核心研究问题

> **能否让一个程序化动画生成 Agent，在不更新模型权重的前提下，仅通过外部"经验记忆"的持续沉淀，把教学视频生成的一次成功率（Pass@1）和视觉质量随使用时长持续提升——即实现真正的"终身学习"？**

围绕这个问题，我们提出**三条具体的研究问题（RQs）**作为论文骨架（详见 §5）。

## 3. 核心思想：从"反思智能体"到"会进化的智能体"

我们的方案在已有反思架构（Reviewer-Coder loop）基础上引入两个关键组件：

1. **情景记忆库（Episodic Memory Bank, EMB）**——一个外置的、可检索的知识库，**双通道**沉淀：成功 scene 的 `<SceneSpec, Rationale, Final Code, VLM Final Score, 帧 hash>` 作为正向示例（4a）；反思过程中验证过的 failure→success 转化作为负向规则（4b）。EMB 完全由系统在 paper-section 任务流上自学习生成，不预置任何人工种子或外部视频库（详见 §4）。
2. **视觉奖励模型（Vision-Language Reward Model, VLRM）**——一个强 VLM（GPT-4o / Gemini-1.5-Pro / Claude-4 Opus），扮演**自主裁判**角色，多维度评估渲染后的视频帧并给出诊断报告，是决定哪些经验值得入库的把关人。

二者闭环作用：**生成 → 渲染 → VLM 打分 → 反思修改 → 再打分 → 高分入库**。随着记忆库扩充，下一轮同类任务的初版生成质量被检索增强直接拔高，反思轮数下降，Pass@1 攀升——形成可观测的"**进化曲线（Evolution Curve）**"。

这本质上是一种 **Prompt-based RLAIF（Reinforcement Learning from AI Feedback）**：VLM 是 reward model，记忆库是 policy 优化的外化载体。我们不动模型权重，但系统行为在持续改进。

---

## 4. 系统结构（The Evolutionary Framework）

系统在概念上由三层耦合而成：任务内的反思闭环（§4.2 + §4.3，VLM 多维打分 + 局部迭代）、跨任务的记忆沉淀闭环（§4.4，双通道写入 EMB），以及由前两者共同驱动的检索增强生成（§4.1，从 EMB 召回 in-context 示例与失败规则）。

EMB 从空集开始，由 paper-section 任务流持续填充，没有独立的种子构建阶段——所谓冷启动只是 EMB 规模较小的早期段，其代码路径与稳态期完全一致。

### 4.1 检索增强生成（RAG over EMB）

收到新输入（例如一篇论文的 §Background 节）后，系统先经 storyboarder 将该节切分成若干 SceneSpec。对每个 SceneSpec：

1. 计算 task_embedding（输入文本 + scene_role 拼接后过 sentence encoder）。
2. 在 EMB.success 中检索 top-k 最近邻，作为正向示例注入 Coder prompt 的 Reference Examples 段（软约束）。
3. 在 EMB.failure 中检索 top-k 最近邻，作为反向规则注入 Coder prompt 的 Known Pitfalls 段（硬约束）。
4. Coder 据此生成初版代码，进入 §4.2 的 VLM 评估。

EMB 规模较小时（系统刚启动的前若干任务），检索结果信号弱、注入价值小，等效于退化为 zero-shot。但 §4.4 的写入闭环始终运行，使 EMB 规模随任务数单调增长——这正是进化曲线在 §4.1 的直接体现。

与已有 RAG-augmented code generation 的关键差别不在检索算法本身，而在被检索的语料是系统自学习产物，不依赖人工构建。

### 4.2 VLM 多维度打分（VLRM-as-a-Judge）

把初版视频的**关键帧序列** + 原始**教学文本** + **storyboard** 一起喂给 VLM。VLM 在三个独立维度上打分：

| 维度 | 评估什么 |
|---|---|
| **Logic Flow（逻辑连贯性）** | 分镜顺序是否符合教学递进；动画过渡是否突兀 |
| **Layout / Occlusion（布局与遮挡）** | 元素是否重叠、超出画面、被公式挡住 |
| **Accuracy（教学准确性）** | 公式是否正确、概念可视化是否引导误解 |

输出**结构化反馈**：`{score: 0–100, per_dim_scores: {...}, diagnostics: ["坐标轴标签和右侧球体重叠了", "动画过渡 0.3 秒太快, 观众跟不上"]}`。

> 实现注记：本节描述的 3 维 × 0-100 schema 已是当前代码的 canonical 形态（`paper2manim/agents/vlm_scene_reviewer.py:33-37`、PR #15）；早期 6 维 × 1-5 baseline 仅保留在 `docs/vlm_experiment.md` 附录 §A 作对照。

### 4.3 反思与局部迭代（Reflection & Iterative Refinement）

Coder 接收 VLM 的诊断，**只改有问题的部分**（局部 patch，而非整段重写），重新渲染，再喂 VLM 评分。循环进行直到：

- VLM 分数 ≥ 优秀阈值（默认 90），或
- 达到 `max_iter`（默认 5）。

> 与已有工作的差别：现有 reflection 只看"代码是否跑通"（Pass@1 binary），我们看"视觉是否合格"（VLM 连续打分），反馈信号信息量高一个数量级。

### 4.4 记忆沉淀（Memory Consolidation）—— **核心创新**

每一次 paper-section 任务收敛后，系统从 trace 中同时蒸馏两路记忆：成功 scene 的最终高分版本写入 EMB.success（4a），反思过程中验证过的 failure→success 转化写入 EMB.failure（4b）。两路写入共享 context 与 provenance 字段骨架，仅 body 字段按极性分歧。

#### 4a：正向沉淀（Success Rationale Memory）

scene 通过 §4.3 收敛、最终 VLM 分数 ≥ θ_high（默认 90/100）时：

1. 让 VLM 写一段 **High-Score Rationale**，自然语言描述为什么该 scene 是好示例（如：用渐入动画让公式逐项浮现，避免了一次性堆叠的视觉压力）。
2. 打包 `<SceneSpec, Rationale, Final Code, VLM Final Score, 帧 hash>` 写入 EMB.success，同时建立 embedding 索引。
3. 下一次同类任务到来时，§4.1 的检索直接召回该条目，初版代码质量被喂答案般拔高。

#### 4b：负向沉淀（Failure Pattern Memory）

反思过程中产生的每一次 failure→success 转化都是潜在的 Lesson。具体来源：

- 文本反思：render error / LaTeX log 触发 reviewer retry，后续版本渲染成功。
- 视觉反思：VLM 低分诊断触发 visual_revise，后续版本 VLM 分数严格上升。

只有同时满足以下两个条件的转化才入 EMB.failure：

1. before / after 来自相邻的两次 attempt（确保因果性可归因）。
2. after_score 严格高于 before_score（验证 fix 真正生效）。

校验失败的转化（after ≤ before，例如 `docs/vlm_experiment.md` §4.3 中 TitleIntro 的 v1=2.83 → v2=2.50 案例）不入库，避免坏记忆污染——这是 EMB 质量的写入端硬过滤，比 §9 的定期重测更早。

满足条件的转化经 LLM 蒸馏成结构化 Lesson：

```json
{
  "polarity": "failure",
  "context": {
    "task_text": "...",
    "task_embedding": [...],
    "scene_role": "method",
    "domain_tags": ["transformer", "attention"],
    "source_paper": "arxiv:1706.03762",
    "source_section": "Background"
  },
  "body": {
    "trigger_pattern":   "scene contains Axes + Text positioned at default ORIGIN",
    "root_cause":        "default Text placement collides with Axes origin",
    "fix_recipe":        "use .next_to(axes, UP, buff=0.5) or explicit .move_to() away from origin",
    "code_anti_example": "Text('label')",
    "code_good_example": "Text('label').next_to(axes, UP, buff=0.5)",
    "vlm_diagnostic":    "<原始 VLM 报告片段>"
  },
  "provenance": {
    "run_id": "...",
    "scene_id": "...",
    "extraction_source": "visual_reflection",
    "validated": true,
    "before_score": 2.17,
    "after_score": 2.83,
    "hit_count": 0,
    "first_seen": "..."
  }
}
```

下一轮任务的代码生成**前**（§4.1 的检索环节），系统按当前 scene 描述检索 top-k 相关 Lesson，作为约束规则注入 Coder prompt 的 Known Pitfalls 段。

#### 双桶共享 Schema

EMB.success 与 EMB.failure 共享 context 字段（task_text / task_embedding / scene_role / domain_tags / source_paper / source_section）和 provenance 元数据骨架（run_id / scene_id / hit_count / first_seen / last_used）。两桶的差异仅在 body 与写入条件：

| 字段 | success | failure |
|---|---|---|
| body | Rationale + 完整代码 + 关键代码片段 + 帧 hash | trigger_pattern + root_cause + fix_recipe + anti & good example |
| 注入位置 | Coder prompt 的 Reference Examples 段（软约束） | Coder prompt 的 Known Pitfalls 段（硬约束） |
| 写入条件 | scene 最终分数 ≥ θ_high | after_score > before_score 且 before/after 相邻 |

共享 schema 头使两桶在检索接口、索引基础设施、provenance 追踪上完全对称——从工程角度可以把 EMB 理解为单一 store + polarity 元数据字段。

#### 4a 与 4b 的范式差别

正向沉淀产出**自然语言 Rationale + 完整代码示例**，靠 in-context 软约束；失败沉淀产出**结构化规则**，作为硬约束注入 prompt。两路信号正交——**正向压缩"该怎么写"的探索空间，负向消解"不该怎么写"的重复试错**。

#### 进化效应

随着任务量累积，平均反思轮数应当**单调下降**，Pass@1 应当**单调上升**。我们把这条曲线作为本工作的**核心实验图（Hero Plot）**。4a 与 4b 是该曲线背后两条**独立可量化**的收敛驱动力——同一类错误经反思捕获并校验通过后即**全局免疫**（4b 贡献），而不仅仅是"找到一个最像的过去成功案例"（4a 贡献）。

```
            ↑ Pass@1
       0.9 ┤                       ●━━━●━━●
           │                  ●━●━━
       0.7 ┤             ●━●━━
           │        ●━●━━
       0.5 ┤   ●━●━━
           │●━━
       0.3 ┤
           └────────────────────────────────→ # paper-sections processed
            0    50    100   200   500
```

x 轴是累计处理的 paper-section 数。Hero Plot 上没有冷启动 / 稳态期的断点——整条曲线就是单一管线在 EMB 规模递增下的单调进化轨迹。

---

## 5. Research Questions

围绕"进化机制是否真的有效、用什么衡量、能否泛化"三个角度：

### RQ1：进化机制的有效性（Self-Evolution Efficacy）

> 在不更新模型权重的前提下，由 VLM 反馈驱动的"记忆库持续扩充"机制，能否显著提升 LLM 在零样本（Zero-shot）和一次通过（Pass@1）生成教学类动画代码 / 视频时的成功率与视觉质量？

**回答**："你的进化机制真的让系统变聪明了吗？"——这是论文的**主结论**。

**实验设计**：固定底层 LLM 不变，比较三种配置在同一份 200 任务测试集上的表现曲线：
- (A) Zero-shot baseline（无记忆，无反思）
- (B) Reflection-only（无记忆，仅 intra-task 反思）
- (C) **Ours: Reflection + EMB**（完整四阶段闭环）

**Hero Plot**：Pass@1 / 平均反思轮数 / VLM 平均分 vs. 累计处理任务数。

### RQ2：VLM 裁判的可靠性（Reliability of VLM-as-a-Judge）

> 在长视频代码生成任务中，VLM 作为自主奖励模型给出的自动化评分与诊断反馈，与人类专家的审美和教学评估（Human Preference）一致性有多高？

**回答**："用 VLM 替代人类来驱动进化，靠不靠谱？"——这是论文的**方法学正当性证明**，目前 Agent / RLAIF 领域评委极为关心。

**实验设计**：抽样 100 条视频，邀请教学专家（数学/CS 教师）独立打分；计算 VLM 与人类的 Pearson / Spearman / Cohen's κ；按维度（Logic / Layout / Accuracy）分别报告。

### RQ3：记忆的跨域泛化能力（Cross-Domain Generalization of Memory）

> Agent 在特定学科（如线性代数）中"进化"积累的排版经验和动画策略，能否成功泛化到未见过的新学科（如量子物理、微观经济学）的可视化任务？

**回答**："系统学到的是死记硬背的代码，还是通用的高级视觉排版策略？"——这是论文的**深度与启发性**所在。

**实验设计**：在 Domain A（线性代数）上跑 200 任务进化 EMB；冻结 EMB；在 Domain B（量子物理）的 50 个未见过的任务上测 Pass@1 vs. 不带 EMB 的同配置。如有显著提升，证明记忆学到的是 transferable layout / pacing 策略而非领域代码模板。

---

## 6. 为什么这个故事对 EMNLP（及同档顶会）有吸引力

1. **踩中 RLAIF 风口**：实质是一种 prompt-based RLAIF——VLM 是 reward model，EMB 是 policy 的外化承载。在不动权重的前提下做出"会学习"的系统，是近两年 Agent 论文最受关注的方向之一。
2. **避免纯工程口水仗**：我们不是去比"谁画图 API 调得更花哨"，而是探讨**LLM 如何通过反思与外部记忆，在多模态任务上做终身学习（Lifelong Learning）**——理论深度足够支撑顶会篇幅。
3. **实验图极具说服力**：Hero Plot 是"随使用次数增加，Pass@1 单调爬升、反思轮数单调下降"的进化曲线——这种"系统在自己学习"的可视化在顶会上极有冲击力。
4. **跨域泛化的反直觉性**：如果 RQ3 成立——"用线性代数的记忆能帮量子物理画得更好"——这是一个对评委非常有说服力的洞见。
5. **零人工标注的方法学清洁度**：EMB 完全由系统在真实任务流上自学习产生，不依赖任何人工挑选的种子三元组或外部视频库——这点在 §10 的对比表上与 Voyager 等已有 skill-library 工作显著区分。

---

## 7. MVP 演进路线图（与代码仓库对应）

为了让"故事"落到能跑的代码上，我们把工程实现拆成三个 MVP，对应本仓库的 `paper2manim/graphs/`：MVP 1.0 → `mvp1.py`、MVP 2.0 / 3.0 共用父图 `mvp2.py`，MVP 3.0 的 per-scene 反思 + EMB 检索闭环在 `scene_graph.py` 子图里实现（PR #19 把 per-scene 循环抽出独立子图以支持 Send 并行）。**注意**：本提案的核心创新（§4.4 自进化）落在 MVP 3.0；MVP 1.0 / 2.0 是必要的能力前置铺垫。

### MVP 1.0：最短链路打通（PoC）

**对应阶段**：1（无 RAG 的退化版）+ 3（无 VLM 的退化版）。

- **输入**：短文本（论文摘要 / 单一核心定理描述）。
- **流程**：Storyboarder → Coder → Render。
- **失败处理**：单次失败即终止，不反思。
- **目标**：验证 LLM + Manim 沙盒端到端能产出可播放的 15–30 秒视频。
- **当前状态**：**已验证**。

### MVP 2.0：单任务反思闭环（Code-level Reflection）

**对应阶段**：1（无 RAG）+ 2（退化为编译错误反馈，非 VLM）+ 3。

- **输入**：完整论文（PDF / arXiv 源码）。
- **流程**：Parser → Summarizer → Storyboarder → 【Coder ↔ Render-error-Reviewer】×N → Concat。
- **关键差异**：阶段 2 的"VLM 视觉裁判"在 2.0 里**退化为编译错误反馈**（Python traceback / LaTeX log），用最便宜的信号先把反思 loop 工程跑通。
- **目标**：在不引入 VLM 成本的前提下，先验证反思机制对**代码层成功率**的提升。
- **当前状态**：**单测全通，真实论文端到端验收均通过**。

### MVP 3.0：VLM 闭环 + 记忆库（The Real Research Version）

**对应阶段**：§4 完整结构（4.1–4.4）。

- **输入**：教学任务描述（论文 / 知识点 / 用户自然语言需求），主流通道是 paper-section（与 MVP 2.0 一致）。
- **流程**：**EMB 检索（§4.1）→ Coder → Render → VLM 评分（§4.2）→ Reflection 局部修改（§4.3）→ 双通道沉淀（§4.4）**。
- **关键差异**：
  - §4.2 接入真实 VLM（已落代码，详见 `docs/vlm_experiment.md`）。
  - §4.4 实现双桶 EMB + 蒸馏管线 + 检索注入。
  - 引入"进化曲线"作为系统级核心评估指标。
- **冷启动**：bootstrap 批次（前 50–100 个 paper-section 任务）让 EMB 从空集起步；该批次内 Pass@1 偏低是预期行为，主要目的是产出初始记忆。bootstrap 段与稳态期使用完全相同的代码路径，差别仅在 EMB 规模。
- **当前状态**：VLM 反思闭环（§4.2 + §4.3）已 land；§4.1 的检索与 §4.4 的双通道沉淀也已落地（PR #16 `paper2manim/emb/`：dual-channel schema + SQLite store + Faiss 检索 + 蒸馏 + RAG 注入，CLI `--emb` 开启）。剩余工程工作：cold-record pruning（[#17](https://github.com/jwj1342/Paper2Manim/issues/17)）与 RAG 注入位置 A/B（[#18](https://github.com/jwj1342/Paper2Manim/issues/18)）。本提案的全部核心创新点（RQ1/2/3）都在此版本上完成实验。

---

## 8. 评估计划

### 8.1 数据集

- **任务池**（约 300 条 paper-section）：从 arXiv 抓取 CS / 数学 / 物理三个学科的论文，按 `\section{...}` 切片；优先抽取 Background / Method / Experiment / Conclusion 四种 section_role，长度 ≤4K chars。覆盖至少 100 篇 paper 以保证主题多样性。
- **Bootstrap 批次**（前 50–100 条）：任意采样自任务池，用于让 EMB 从空集起步；该批次内**不计入** Hero Plot 主曲线，主要目的是产出初始记忆。
- **稳态评估批次**（接下来约 150 条）：用于绘制 Hero Plot 主体段——观察 Pass@1 / 平均反思轮数 / VLM 平均分随 EMB 规模的变化。
- **跨域测试集**（50 条）：来自训练池**未出现**的两个学科（量子物理、微观经济学）；用于 RQ3。
- 整套数据采集流程不需要任何人工三元组标注或外部视频库。

### 8.2 指标

| 指标 | 含义 | 用于哪个 RQ |
|---|---|---|
| **Pass@1** | 不依赖反思、首次生成即渲染成功且 VLM 分数 ≥ θ 的比例 | RQ1 主指标 |
| **Pass@K** | 至多 K 轮反思后达标的比例 | RQ1 辅助 |
| **平均反思轮数** | 单任务进入阶段 3 后的平均迭代次数 | RQ1 / 进化曲线 |
| **VLM 平均分** | VLM 给出的 0–100 分均值（逐维度 + 加权综合） | RQ1 / RQ2 |
| **Human-VLM 一致性** | Pearson / Spearman / Cohen's κ | RQ2 |
| **Domain-Transfer Gain** | `Pass@1(Domain B with frozen EMB) - Pass@1(Domain B no EMB)` | RQ3 |

### 8.3 关键消融

- **Ablation A**：去掉 EMB（退回 MVP 2.0）——验证 §4.4 的必要性。
- **Ablation B**：去掉 VLM 反馈，仅用代码运行错误（退回 §4.2 的退化版本）——验证 VLM 信号的增量价值。
- **Ablation C**：bootstrap 批次大小 ∈ {0, 10, 50, 100}——验证 EMB 早期规模对评估期 Pass@1 的影响。这是原 proposal 中"种子记忆数量"消融的等价版，因新设计没有外部种子，改为评估前预跑的任务数，语义相同。
- **Ablation D**：把 VLRM 换成更小的开源 VLM（如 Qwen-VL）——验证 RQ2 的鲁棒性。
- **Ablation E**：双通道分离消融——分别关闭 4a（正向 Rationale Memory）和 4b（Failure Pattern Memory），量化两路信号各自的边际贡献。预期：两路独立关闭后反思轮数下降斜率均显著变缓但不归零，证明它们正交且互补；同时关闭则退化为 Ablation A。

---

## 9. 风险与缓解

| 风险 | 缓解 |
|---|---|
| VLM 评分与人类不一致（RQ2 失败） | 提前用 30 条小样本预试；若一致性低，用 Domain expert annotation 作为 ground truth 微调 prompt |
| 记忆库出现"坏记忆"导致后续生成劣化 | 写入端硬过滤：4b 要求 `after_score > before_score`，4a 要求 final ≥ θ_high；运行时引入"记忆退化检测"——定期重测旧记忆在新 VLM 下的得分，分数下降则降权或剔除 |
| EMB 早期规模为 0 时检索几乎是 no-op，bootstrap 批次内 Pass@1 偏低 | 这是预期行为而非 bug——bootstrap 段不计入 Hero Plot 主曲线，且 4a/4b 沉淀正是从该段开始累积。论文 §Methodology 单独说明该段为非评估段 |
| Domain B 上 EMB 反而拖累（RQ3 反直觉但可能） | 即使是 negative result 也具学术价值——揭示记忆迁移的边界条件 |
| VLM API 调用成本 | 用本地 Qwen-VL 跑大规模实验，GPT-4o 仅在最终评估和 Ablation D 用 |

---

## 10. 与已有工作的比较

| 工作 | 反思机制 | VLM 裁判 | 跨任务记忆 | 进化曲线评估 |
|---|---|---|---|---|
| Manimator (2025) | ❌ | ❌ | ❌ | ❌ |
| Code2Video (2025) | ✅ (multi-agent) | ❌ (rule-based) | ❌ | ❌ |
| manim-generator | ✅ (compile-error) | ❌ | ❌ | ❌ |
| Voyager (Minecraft, NeurIPS'23) | ✅ | ❌ | ✅ (skill library) | 部分 |
| **Ours** | ✅ | **✅** | **✅ (dual-channel: success rationale + validated failure patterns)** † | **✅ (核心指标)** |

† EMB 完全由系统在 paper-section 任务流上自学习生成，不依赖任何人工标注的三元组或外部视频库；这是与 Voyager 等已有 skill-library 工作的关键区分点。

我们与 Voyager 在"外部技能 / 记忆库"的思想上有共鸣，但 Voyager 在游戏环境内，奖励是规则化的稀疏信号；我们在多模态视觉创作任务上，奖励是 VLM 连续打分——这是更困难、也更接近真实"教学质量"的场景。

---

## 11. 时间表（建议）

| 阶段 | 周次 | 产出 |
|---|---|---|
| MVP 2.0 收尾 | W1–W2 | 端到端跑通 ≥3 篇真实论文（已 land） |
| MVP 3.0 §4.2/4.3（VLM Judge + Reflection） | W3–W4 | 接入 VLM、reflection loop 用 VLM 信号（已在 PR #11 land + PR #15 schema 收敛 3 维×0-100） |
| MVP 3.0 §4.4（EMB store + 蒸馏） | W5–W6 | 双桶 schema、检索接口、4a/4b 蒸馏管线（已在 PR #16 land） |
| MVP 3.0 §4.1（RAG 注入） | W7 | EMB 检索结果注入 Coder prompt（已 land）；跑 bootstrap 50 任务 |
| **图计算并行化** | W7 平行线 | per-scene fan-out + render/LLM throttle；把后续 Hero Plot 实验循环从天级压到小时级（PR #19） |
| RQ1 + RQ2 主实验 | W8–W10 | 稳态 150 任务跑 Hero Plot；三组配置对比 + 100 条人类标注一致性 |
| RQ3 跨域实验 | W11 | Domain A → Domain B 迁移 |
| Ablation + 写作 | W12–W14 | EMNLP submission |

---

## 12. 一句话总结

> **我们让程序化动画生成 Agent 不再"失忆"。每一次成功的视频都成为下一次的起点，每一次反思中验证过的失败修复都成为下一次的硬约束——系统行为在不动权重的情况下持续进化。这是把 LLM 用作创作主体时最值得探索的下一步。**
