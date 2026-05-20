# Paper2Manim Human Evaluation Rubric / 人类评测量表

Object: first-attempt paper-section animation outputs  
对象：论文片段生成的 first-attempt 动画结果

---

## 0. Goal / 评测目标

The goal is to judge whether a first-attempt animation correctly and clearly visualizes the given paper unit.

本量表用于判断：第一次生成的动画是否正确、清楚地可视化了给定论文片段。

This rubric is aligned with the paper’s core claim: ManimAgent should reduce repeated cross-task layout, pacing, and content mistakes by reusing accumulated episodic memory.

该量表对齐论文核心主张：ManimAgent 应通过跨任务积累的 EMB 经验，减少重复出现的布局、节奏和内容错误。

---

## 1. What the rater sees / 评审会看到什么

You will see:

- paper title / 论文标题
- target unit title and text / 目标片段标题与文本
- scene role / 场景角色
- key claims / 关键论点
- reference scene plan / 参考分镜
- generated first-attempt video / 第一次生成的视频

You will not see:

- system name / 系统名称
- baseline or EMB condition / baseline 或 EMB 条件
- EMB size / EMB 大小
- VLM score / VLM 分数
- prompt, code, or revision trace / 提示词、代码或修复轨迹

The reference scene plan is guidance, not a required script. Do not penalize a video only because it uses a different but valid visualization.

参考分镜是评分参考，不是唯一标准答案。只要视频用另一种方式正确表达论文片段，不应因此扣分。

---

## 2. Human Pass@1 / 人类 Pass@1

**Question / 问题**

Would you accept this first-attempt animation as a usable visualization of the given paper unit?

你是否认为这个 first-attempt 动画已经可以作为该论文片段的可用可视化结果？

Choose one / 请选择：

- Yes / 是
- No / 否

Choose **Yes** only if the video satisfies all of the following:

只有大体满足以下条件时才选择 **是**：

1. The core paper content is not seriously wrong. / 核心论文内容没有严重错误。
2. The main visual is readable. / 主要画面可读。
3. The animation order is understandable. / 动画顺序可以理解。
4. There is no fatal layout, formula, or mismatch error. / 没有致命的布局、公式或内容不匹配错误。
5. The video would not require major repair before being used as a first-pass output. / 作为 first-pass 输出不需要大修。

---

## 3. Five rating dimensions / 五个评分维度

Use integer scores from 1 to 5.

请使用 1 到 5 的整数分。

### Q1. Paper Alignment / 论文内容对齐

Does the animation faithfully match the target paper unit?

动画是否忠实对齐目标论文片段？

| Score | English | 中文 |
|---|---|---|
| 1 | Unrelated or seriously wrong. | 基本无关或严重错误。 |
| 2 | Major content errors that affect understanding. | 有影响理解的重大内容错误。 |
| 3 | Generally related, but with noticeable omissions or misunderstandings. | 大体相关，但有明显遗漏或误解。 |
| 4 | Mostly correct with minor issues. | 基本正确，只有轻微问题。 |
| 5 | Accurate and faithful to the target unit. | 准确、忠实地表达目标片段。 |

### Q2. Key-Claim Coverage / 关键论点覆盖

Does the animation cover the key claims of the target unit?

动画是否覆盖目标片段的关键论点？

| Score | English | 中文 |
|---|---|---|
| 1 | Covers almost none of the key claims. | 几乎没有覆盖关键论点。 |
| 2 | Covers only a superficial or minor point. | 只覆盖表面或次要内容。 |
| 3 | Covers some key claims but misses important ones. | 覆盖部分关键点，但遗漏明显。 |
| 4 | Covers most key claims. | 覆盖大部分关键论点。 |
| 5 | Covers the central claims clearly and completely. | 清楚、完整覆盖核心论点。 |

### Q3. Visual Robustness / 视觉稳健性

Is the rendered scene readable and free from common visual failures?

渲染画面是否可读，并避免常见视觉失败？

| Score | English | 中文 |
|---|---|---|
| 1 | Unreadable due to severe overlap, cropping, blank screen, or chaos. | 因严重重叠、裁切、空白或混乱而不可读。 |
| 2 | Many layout problems; readability is clearly affected. | 多处布局问题，明显影响阅读。 |
| 3 | Understandable overall, but cluttered or visually weak. | 大体能懂，但拥挤或视觉重点弱。 |
| 4 | Clear with only minor layout issues. | 清楚，只有轻微布局问题。 |
| 5 | Clean, readable, and visually unambiguous. | 干净、可读、视觉重点明确。 |

### Q4. Animation Flow / 动画流程

Does the animation unfold in a coherent order?

动画是否按连贯顺序展开？

| Score | English | 中文 |
|---|---|---|
| 1 | Chaotic; the sequence cannot be followed. | 顺序混乱，无法跟随。 |
| 2 | Some steps exist, but transitions are confusing. | 有一些步骤，但衔接混乱。 |
| 3 | Basic order is present, but some steps are missing or abrupt. | 有基本顺序，但有缺步或跳跃。 |
| 4 | The order is reasonable and easy to follow. | 顺序合理，容易跟随。 |
| 5 | Smooth step-by-step progression to the final takeaway. | 步骤自然推进到最终要点。 |

### Q5. First-Attempt Usability / 首轮可用性

How usable is this as a first-attempt output before any repair?

在未经后续修复前，这个 first-attempt 输出的可用程度如何？

| Score | English | 中文 |
|---|---|---|
| 1 | Unusable; requires complete regeneration. | 不可用，需要完全重做。 |
| 2 | Requires major repair. | 需要大幅修复。 |
| 3 | Borderline; usable only after several fixes. | 勉强可用，但需要多处修复。 |
| 4 | Mostly usable with minor fixes. | 基本可用，只需小修。 |
| 5 | Usable as-is or with negligible fixes. | 可以直接使用，或只需极小修改。 |

---

## 4. Failure flags / 失败类型勾选

Check all that apply.

请勾选所有适用项。

- [ ] unrelated to target unit / 与目标片段无关
- [ ] major formula, symbol, variable, or factual error / 公式、符号、变量或事实严重错误
- [ ] missing central claim / 遗漏核心论点
- [ ] severe overlap or occlusion / 严重重叠或遮挡
- [ ] cropped or off-screen object / 对象被裁切或跑出画面
- [ ] unreadable text or labels / 文字或标签不可读
- [ ] animation order is confusing / 动画顺序混乱
- [ ] too static or no meaningful animation / 过于静态或缺少有效动画
- [ ] too text-heavy / 文字堆叠过多
- [ ] empty, broken, or unplayable video / 视频为空、损坏或无法播放
- [ ] other / 其他：__________

If a fatal issue makes the video unusable, Human Pass@1 should be **No**.

如果致命问题导致视频不可用，Human Pass@1 应选择 **否**。

---

## 5. Final scoring form / 最终打分表

Sample ID / 样本 ID：__________  
Video ID / 视频 ID：__________  
Rater ID / 评审 ID：__________

Human Pass@1 / 人类 Pass@1：

- [ ] Yes / 是
- [ ] No / 否

Scores / 分数：

| Dimension / 维度 | Score 1–5 |
|---|---|
| Paper Alignment / 论文内容对齐 | ___ |
| Key-Claim Coverage / 关键论点覆盖 | ___ |
| Visual Robustness / 视觉稳健性 | ___ |
| Animation Flow / 动画流程 | ___ |
| First-Attempt Usability / 首轮可用性 | ___ |

Failure flags / 失败类型：

- [ ] unrelated to target unit / 与目标片段无关
- [ ] major formula/symbol/factual error / 公式、符号或事实严重错误
- [ ] missing central claim / 遗漏核心论点
- [ ] severe overlap or occlusion / 严重重叠或遮挡
- [ ] cropped or off-screen object / 裁切或跑出画面
- [ ] unreadable text or labels / 文字或标签不可读
- [ ] confusing animation order / 动画顺序混乱
- [ ] too static / 过于静态
- [ ] too text-heavy / 文字过多
- [ ] empty/broken/unplayable / 空、损坏或无法播放
- [ ] other / 其他：__________

Optional comment / 可选评论：

____________________________________________________________

---

## 6. Aggregation rules / 汇总规则

For researchers only.

供研究者汇总使用。

- Human Pass@1: majority vote across raters.
- Human Quality Score: average of the five 1–5 scores, then average across raters.
- Report dimension-level scores and failure-flag frequencies in the appendix.
- Reflection Depth comes from system logs, not human raters.
- VLM scores are auxiliary diagnostics, not the primary human quality evidence.

- Human Pass@1：多名评审多数投票。
- Human Quality Score：五个维度取平均，再对评审取平均。
- 分维度分数和失败类型频率放附录。
- Reflection Depth 来自系统日志，不由人类评审打分。
- VLM 分数只作为辅助诊断，不作为主要人类质量证据。
