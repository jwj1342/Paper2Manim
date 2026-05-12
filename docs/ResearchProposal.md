# Research Proposal

**面向教学动画生成的自进化多智能体系统：基于 VLM 反馈的情景记忆库**

> Self-Evolving Multi-Agent System for Educational Animation Generation via VLM-Driven Episodic Memory

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

1. **情景记忆库（Episodic Memory Bank, EMB）**——一个外置的、可检索的知识库，**双通道**沉淀：既存储成功视频背后的 `<教学文本 → 视觉策略 → Manim 代码>` 三元组作为**正向示例**，也存储反思过程中诊断出的失败模式与修复规则作为**负向规则**（详见阶段 4）。
2. **视觉奖励模型（Vision-Language Reward Model, VLRM）**——一个强 VLM（GPT-4o / Gemini-1.5-Pro / Claude-4 Opus），扮演**自主裁判**角色，多维度评估渲染后的视频帧并给出诊断报告，是决定"哪些经验值得入库"的把关人。

二者闭环作用：**生成 → 渲染 → VLM 打分 → 反思修改 → 再打分 → 高分入库**。随着记忆库扩充，下一轮同类任务的初版生成质量被检索增强直接拔高，反思轮数下降，Pass@1 攀升——形成可观测的"**进化曲线（Evolution Curve）**"。

这本质上是一种 **Prompt-based RLAIF（Reinforcement Learning from AI Feedback）**：VLM 是 reward model，记忆库是 policy 优化的外化载体。我们不动模型权重，但系统行为在持续改进。

---

## 4. 四阶段进化框架（The Evolutionary Framework）

下面四个阶段构成一个严密的闭环。**阶段 1–3 是单次任务（intra-task）的"反思闭环"，阶段 4 是跨任务（inter-task）的"进化闭环"——这才是本工作的核心创新。**

### 阶段 1：冷启动与检索增强生成（RAG from Seed Memory）

**种子记忆**：人工精选少量（约 50 条）高质量的 3Blue1Brown / Manim 社区精品视频片段，从中提取 `<教学文本, 视觉策略总结, Manim 代码>` 三元组作为冷启动的"专家先验"。

**检索增强**：收到新输入（例如"解释正态分布"）时，先在 EMB 中做语义检索，召回最相似的若干历史记忆，把它们的"视觉策略总结"+"成功代码片段"作为 in-context examples 注入 prompt，指导 Coder 生成初版代码 → 渲染初版视频。

> 越往后跑，EMB 越大，初版质量越高——这是阶段 4"自进化"在阶段 1 的直接体现。

### 阶段 2：VLM 多维度打分（VLRM-as-a-Judge）

把初版视频的**关键帧序列** + 原始**教学文本** + **storyboard** 一起喂给 VLM。VLM 在三个独立维度上打分：

| 维度 | 评估什么 |
|---|---|
| **Logic Flow（逻辑连贯性）** | 分镜顺序是否符合教学递进；动画过渡是否突兀 |
| **Layout / Occlusion（布局与遮挡）** | 元素是否重叠、超出画面、被公式挡住 |
| **Accuracy（教学准确性）** | 公式是否正确、概念可视化是否引导误解 |

输出**结构化反馈**：`{score: 0–100, per_dim_scores: {...}, diagnostics: ["坐标轴标签和右侧球体重叠了", "动画过渡 0.3 秒太快, 观众跟不上"]}`。

### 阶段 3：反思与局部迭代（Reflection & Iterative Refinement）

Coder 接收 VLM 的诊断，**只改有问题的部分**（局部 patch，而非整段重写），重新渲染，再喂 VLM 评分。循环进行直到：

- VLM 分数 ≥ 优秀阈值（默认 90），或
- 达到 `max_iter`（默认 5）。

> 与已有工作的差别：现有 reflection 只看"代码是否跑通"（Pass@1 binary），我们看"视觉是否合格"（VLM 连续打分），反馈信号信息量高一个数量级。

### 阶段 4：记忆沉淀与自进化（Memory Consolidation & Self-Evolution）—— **核心创新**

阶段 3 收敛后系统执行**双通道知识蒸馏**：成功视频沉淀为正向示例（4a），反思过程中暴露的失败模式沉淀为负向规则（4b）。

#### 4a：正向沉淀（Success Rationale Memory）

视频一旦通过阶段 3 的高分校验，系统执行：

1. 让 VLM 写一段"**High-Score Rationale**"——为什么这是个好视频（"用渐入动画让公式逐项浮现，避免了一次性堆叠的视觉压力"）。
2. 打包 `<Input Text, Rationale, Final Code, VLM Final Score, 关键帧 hash>` 写入 EMB，同时建立 embedding 索引。
3. 下一次同类任务到来时，阶段 1 的检索直接召回这条记忆，初版代码质量被"喂答案"般拔高。

#### 4b：负向沉淀（Failure Pattern Memory）

仅靠正向沉淀有一个盲区：**反思 loop 在收敛过程中产生的负向信号被完全丢弃**。VLM 报告的每一个具体诊断（"`MathTex` 与 `Axes` 原点重叠"、"动画过渡 0.3s 太快观众跟不上"）、Reviewer 抓到的每一个 traceback（"`set_stroke()` got unexpected keyword `dash_pattern`"），都是"未来不该再犯"的可复用知识——目前 run 结束就蒸发。

我们把这些反思事件**追溯到代码层级**（具体的 Mobject、API 调用、或视觉症状），蒸馏成 "**Lesson**" 形态的结构化规则：

```json
{
  "lesson_id": "L0042",
  "trigger_pattern": "scene contains Axes + Text positioned at default ORIGIN",
  "root_cause": "default Text placement collides with Axes origin",
  "fix_recipe": "use .next_to(axes, UP, buff=0.5) or explicit .move_to() away from origin",
  "code_anti_example": "Text('label')",
  "code_good_example": "Text('label').next_to(axes, UP, buff=0.5)",
  "tags": ["layout", "axes", "occlusion"],
  "source_runs": ["run_20260512_..."],
  "hit_count": 1
}
```

在下一轮任务的代码生成**前**（阶段 1 的检索环节），系统按当前 scene 描述 + 上一轮（若有）失败上下文检索 top-k 相关 Lesson，作为约束规则注入 Coder prompt 的 "Known pitfalls" 段。

**与 4a 的范式差别**：正向沉淀产出**自然语言 Rationale + 完整代码示例**，靠 in-context 软约束；失败沉淀产出**结构化规则**，作为硬约束注入 prompt。两路信号正交——**正向压缩"该怎么写"的探索空间，负向消解"不该怎么写"的重复试错**。

#### 进化效应

随着任务量累积，平均反思轮数应当**单调下降**，Pass@1 应当**单调上升**。我们把这条曲线作为本工作的**核心实验图（Hero Plot）**。4a 与 4b 是该曲线背后两条**独立可量化**的收敛驱动力——同一类错误被 Reviewer 发现一次后即**全局免疫**（4b 贡献），而不仅仅是"找到一个最像的过去成功案例"（4a 贡献）。

```
            ↑ Pass@1
       0.9 ┤                       ●━━━●━━●
           │                  ●━●━━
       0.7 ┤             ●━●━━
           │        ●━●━━
       0.5 ┤   ●━●━━
           │●━━
       0.3 ┤
           └────────────────────────────────→ # tasks processed
            0    50    100   200   500
```

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

---

## 7. MVP 演进路线图（与代码仓库对应）

为了让"故事"落到能跑的代码上，我们把工程实现拆成三个 MVP，与本仓库的 `paper2manim/graphs/mvp{1,2,3}.py` 一一对应。**注意**：本提案的核心创新（阶段 4 自进化）落在 MVP 3.0；MVP 1.0 / 2.0 是必要的能力前置铺垫。

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
- **当前状态**：**单测全通，待真实论文端到端验收**。

### MVP 3.0：VLM 闭环 + 记忆库（The Real Research Version）

**对应阶段**：1 + 2 + 3 + **4**（完整四阶段框架）。

- **输入**：教学任务描述（论文 / 知识点 / 用户自然语言需求）。
- **流程**：**EMB 检索 → Coder → Render → VLM 评分 → Reflection 局部修改 → 高分入 EMB**。
- **关键差异**：
  - 阶段 2 接入真实 VLM（候选：GPT-4o / Gemini-1.5-Pro / 豆包视觉模型）。
  - 阶段 4 实现知识蒸馏 + 记忆库写入 + 检索 in-context 注入。
  - 引入"进化曲线"作为系统级核心评估指标。
- **当前状态**：VLM 客户端、scene reviewer、visual revision agent 已落代码（见 `paper2manim/infrastructure/vlm/`、`paper2manim/agents/{vlm_scene_reviewer, visual_revision_agent}.py`），**尚未接入 graph、尚无 EMB**。本提案的全部核心创新点（RQ1/2/3）都在此版本上完成实验。

---

## 8. 评估计划

### 8.1 数据集

- **种子数据**（人工）：从 3Blue1Brown / Manim 社区精选 50 条优质视频片段，标注 `<text, strategy, code>` 三元组，作为 EMB 冷启动 + Memory 质量上限的参照。
- **训练 / 进化任务池**（200 条）：覆盖 CS / 数学 / 物理三个学科，每个学科 ~70 条；每条是一段 200–400 字的概念解释。
- **跨域测试集**（50 条）：来自训练池**未出现**的两个学科（量子物理、微观经济学）；用于 RQ3。

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

- **Ablation A**：去掉 EMB（退回 MVP 2.0）——验证阶段 4 的必要性。
- **Ablation B**：去掉 VLM 反馈，仅用代码运行错误（退回阶段 2 的退化版本）——验证 VLM 信号的增量价值。
- **Ablation C**：种子记忆数量 ∈ {0, 10, 50, 200}——验证冷启动门槛。
- **Ablation D**：把 VLRM 换成更小的开源 VLM（如 Qwen-VL）——验证 RQ2 的鲁棒性。
- **Ablation E**：双通道分离消融——分别关闭 4a（正向 Rationale Memory）和 4b（Failure Pattern Memory），量化两路信号各自的边际贡献。预期：两路独立关闭后反思轮数下降斜率均显著变缓但不归零，证明它们正交且互补；同时关闭则退化为 Ablation A。

---

## 9. 风险与缓解

| 风险 | 缓解 |
|---|---|
| VLM 评分与人类不一致（RQ2 失败） | 提前用 30 条小样本预试；若一致性低，用 Domain expert annotation 作为 ground truth 微调 prompt |
| 记忆库出现"坏记忆"导致后续生成劣化 | 引入"记忆退化检测"：定期重测旧记忆在新 VLM 下的得分；分数下降则降权或剔除 |
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
| **Ours** | ✅ | **✅** | **✅ (dual-channel: success rationale + failure patterns)** | **✅ (核心指标)** |

我们与 Voyager 在"外部技能 / 记忆库"的思想上有共鸣，但 Voyager 在游戏环境内，奖励是规则化的稀疏信号；我们在多模态视觉创作任务上，奖励是 VLM 连续打分——这是更困难、也更接近真实"教学质量"的场景。

---

## 11. 时间表（建议）

| 阶段 | 周次 | 产出 |
|---|---|---|
| MVP 2.0 收尾 | W1–W2 | 端到端跑通 ≥3 篇真实论文 |
| MVP 3.0 阶段 2（VLM Judge） | W3–W4 | 接入 VLM、reflection loop 用 VLM 信号、跑 50 任务定性看 |
| MVP 3.0 阶段 4（EMB） | W5–W7 | 实现检索 + 知识蒸馏写入、跑通 200 任务进化曲线 |
| RQ1 + RQ2 主实验 | W8–W10 | 三组配置对比 + 人类标注一致性 |
| RQ3 跨域实验 | W11 | Domain A → Domain B 迁移 |
| Ablation + 写作 | W12–W14 | EMNLP submission |

---

## 12. 一句话总结

> **我们让程序化动画生成 Agent 不再"失忆"。每一次成功的视频都成为下一次的起点，系统行为在不动权重的情况下持续进化——这是把 LLM 用作创作主体时最值得探索的下一步。**
