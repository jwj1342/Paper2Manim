# P2M-Bench: Dataset & Benchmark for Paper-to-Manim Educational Animation

## 1. 数据集结构

### 1.1 两层结构：paper context + target_unit

每条样本以 `(paper_id, target_unit)` 为主键。同一篇论文可派生多条样本（例如 *Attention Is All You Need* 派生 `Scaled Dot-Product Attention`、`Multi-Head Attention`、`Architecture Figure` 三条）。这与 proposal §4.1 的 "EMB 检索粒度 = scene/task" 直接对应——同一篇论文的多个 target_unit 在 EMB 里走独立检索 key，不会互相串扰。

模型在生成时拿到：

```text
paper_full_text        ←  整篇论文背景（建立语境，但 ≠ 必须可视化全部）
+ target_unit          ←  当前要解释的局部
+ main_topics          ←  3-5 个核心知识点（监督 storyboarder 选材）
```

这一层结构与 `paper2manim/state.py` 的 `PaperState` 几乎一一对应：`paper_full_text` → `state["full_text"]`，`target_unit` → `state["arxiv_section"]`（subsection 情况）或新增 `state["target_unit"]`（其余情况）。

### 1.2 主表字段（HuggingFace parquet）

| 字段 | 类型 | 必填 | 含义 / 与 proposal 的对应 |
|---|---|---|---|
| `id` | str | ✅ | `<paper_id>_<unit_kind>_<unit_index>`，主键 |
| `category` | enum | ✅ | `Concept / Equation / Algorithm / Figure / Architecture / Experiment` — 教学任务类型 |
| `domain` | enum | ✅ | `cs / math / physics / quantum / econ`（与 `paper2manim/datasets/constants.py::DOMAINS` 同步） |
| `paper_id` | str | ✅ | arXiv id（兼容 `hep-th/9901001` 旧式 id）或本地 PDF SHA-1 |
| `paper_title` | str | ✅ | 论文标题 |
| `paper_full_text` | str | ✅ | 整篇清洗后文本（LaTeX 优先，回退 Marker PDF）。HF parquet inline；本地 dev 走 `papers/<paper_id>.txt` sidecar |
| `target_unit` | struct | ✅ | 见 [§1.3](#13-target_unit-结构) |
| `source_type` | enum | ✅ | `subsection / equation / figure / table / algorithm / full_paper` |
| `main_topics` | list[str] | ✅ | 3-5 个核心知识点；驱动 storyboarder 选材，也用作 EMB 检索辅助 key |

| `split` | enum | ✅ | `dev / test / hard_test / human_gold`（[§4](#4-与当前-p2m_v1csv-的关系迁移路径) 给出与 `bootstrap/eval/cross_train/cross_test` 的映射） |

### 1.3 `target_unit` 结构

```json
{
  "title":  "Scaled Dot-Product Attention",
  "text":   "<原文摘录, ≤4K chars>",
  "type":   "subsection | equation | figure | table | algorithm | full_paper",
  "anchor": {
    "section":      "3.2.1",
    "equation_label": "eq:scaled-dot",
    "figure_label":   "fig:transformer-arch",
    "page":         3
  }
}
```

`anchor` 的所有字段都 optional，按 `type` 取需要的子集：subsection 用 `section`、equation 用 `equation_label`、figure 用 `figure_label`、algorithm 用 `section + page`。`anchor` 让 parser 能精确定位（即使原文有命名冲突）。

### 1.4 每条样本的完整标注

```json
{
  "id": "1706.03762_eq_scaled_dot",
  "category": "Equation",
  "domain": "cs",
  "paper_id": "1706.03762",
  "paper_title": "Attention Is All You Need",
  "paper_full_text": "...",
  "paper_sections": [
    { "title": "Background", "text": "..." },
    { "title": "Model Architecture", "text": "..." }
  ],
  "target_unit": {
    "title": "Scaled Dot-Product Attention",
    "text": "We call our particular attention 'Scaled Dot-Product Attention' ...",
    "type": "equation",
    "anchor": { "section": "3.2.1", "equation_label": "eq:scaled-dot" }
  },
  "equations":   [ { "label": "eq:scaled-dot", "tex": "\\text{Attention}(Q,K,V) = \\text{softmax}(QK^\\top/\\sqrt{d_k})V" } ],
  "figures":     [ { "label": "fig:transformer-arch", "caption": "..." } ],
  "key_claims":  [
    "QK^T measures pairwise similarity between queries and keys",
    "Scaling by sqrt(d_k) prevents softmax saturation at large d_k",
    "The output is a weighted sum of V with weights from softmax"
  ],
  "core_entities":          [ "Query (Q)", "Key (K)", "Value (V)", "softmax", "d_k" ],
  "common_misunderstandings": [
    "Dropping the sqrt(d_k) scaling is harmless — it is not; gradients vanish.",
    "Q, K, V are three separate matrices learned independently — they are projections of the same input."
  ],
  "reference_scene_plan": [
    { "beat": "Show Q, K, V as three colored matrix blocks" },
    { "beat": "Compute QK^T, visualize as similarity heatmap" },
    { "beat": "Divide by sqrt(d_k), highlight scaling effect on softmax" },
    { "beat": "Apply softmax, then weight-sum with V" }
  ],
  "evaluation_rubric": {
    "logic_flow":       [ "Beats appear in causal order (Q,K,V → QK^T → scale → softmax → weighted sum)" ],
    "layout_occlusion": [ "Matrices labeled and non-overlapping", "Equation legible at 480p15" ],
    "accuracy":         [ "softmax denominator is sqrt(d_k), not d_k", "Output dim matches V's columns" ]
  }
}
```

### 1.5 Ground truth 的哲学

P2M-Bench **不提供** "唯一标准 Manim 视频"。它提供的是：

```text
1. 整篇论文理解材料 (paper_full_text, paper_sections)
2. 局部讲解锚点      (target_unit, anchor)
3. 关键知识点        (key_claims, core_entities, common_misunderstandings)
4. 参考分镜          (reference_scene_plan)         ← 评测端 reference, 见下方 §2.4 红线
5. 评测 rubric       (evaluation_rubric)            ← VLM-as-Judge / Human 共用
```

这样每条样本能评测 5 件事：

1. 是否理解整篇论文的背景与目标（covered by `paper_full_text` 输入完整）
2. 是否准确定位并解释 `target_unit`（covered by `key_claims` × VLM/Human 打分）
3. 是否产出合理的教学分镜（covered by `reference_scene_plan` × storyboard 对齐度）
4. 是否产出可执行 Manim 代码（covered by 现有 `Pass@1` / `Pass@K` 指标）
5. 视频是否语义忠实、画面清晰、有教学性（covered by `evaluation_rubric` × VLM 三维分）

### 1.6 `full_paper` 任务专属指标

`source_type=full_paper` 与其它 5 种 scene-local 类型有本质差别：它要求模型把整篇论文压成一支 60-90s 视频，跨多个 scene 串成连贯讲解。scene-local 的三维（logic_flow / layout_occlusion / accuracy）只评单 scene，**评不到跨 scene 的衔接质量**。因此 `full_paper` 样本的 `evaluation_rubric` 在标准三维之外额外要求两维：

| 维度 | 评的是什么 | 0 锚点 | 100 锚点 |
|---|---|---|---|
| `narrative_coherence` | 章节之间的过渡是否符合教学递进；前一 scene 留下的悬念是否被下一 scene 拾起 | 章节像独立短片随机拼接 | 像一段连续讲解，承接关系明确 |
| `symbol_consistency` | 同一符号 / 变量 / 颜色编码在不同 scene 中是否保持一致 | 同一 Q 矩阵换三种颜色三种字体 | 全片符号 / 颜色 / 标记体系自洽 |

实施约束：
- 这两维**只对** `source_type=full_paper` 样本生效；scene-local 样本不评（避免对单 scene 任务硬塞跨 scene 指标）
- VLM 评分时需输入**整片视频的关键帧 montage**（所有 scene 各抽 1-2 帧拼接），而非单 scene 4 帧——这要求 `paper2manim/utils/frame_sampler.py` 增加 `whole_video_montage` 模式
- §5 把 `full_paper` 截成 "abstract+intro+conclusion 三段" 是为绕 recursion limit 的工程妥协；引入这两维后**必须改回完整论文**作为评测输入——否则评 `narrative_coherence` 没意义。length cap 8K 是 storyboarder 的输入上限，不应反过来截 paper

---

## 2. 与 proposal 实验需求的契合度

### 2.1 RQ × 字段映射

| Proposal | 需要的数据信号 | P2M-Bench 字段 | 状态 |
|---|---|---|---|
| **RQ1** Hero Plot — Pass@1 / 反思轮数 / VLM 均分 vs. 累计任务数 | 200+ 同质任务流 | `id` × `split=dev` 的有序序列 | ✅ 直接支持 |
| **RQ1** 主曲线三组 (A/B/C) 对比 | 同一份任务集跑 3 配置 | 任意 split | ✅ 直接支持，无需新字段 |
| **RQ2** VLM-Human 一致性 (Pearson / κ) | 人工逐维打分 ≥ 100 视频 | `split=human_gold` 的子集，独立存 `human_scores` parquet | ✅ 新增 `split=human_gold`，配套外存 |
| **RQ2** 维度对齐 (Logic / Layout / Accuracy) | VLM 与人类共用同套维度 | `evaluation_rubric.{logic_flow, layout_occlusion, accuracy}` | ✅ 与 `prompts/vlm_scene_reviewer.md` 完全对应 |
| **RQ3** 跨域泛化 (Domain A → Domain B) | 训练域 vs. 未见域明确分离 | `domain` ∈ {cs, math} 训练，∈ {quantum, econ} 测试 | ✅ 复用 `domain` 字段，配 `split=hard_test` |
| **§8.3 Ablation A** 去 EMB | 同一份任务集，配置驱动 | 任意 split + preset `A` vs `C` | ✅ 直接支持 |
| **§8.3 Ablation B** 去 VLM | 同上 | 任意 split + preset `A` vs `B` | ✅ 直接支持 |
| **§8.3 Ablation C** bootstrap 批次大小 ∈ {0,10,50,100} | 按顺序前 N 条不计入 Hero Plot | `split=dev` 内顺序前 N 条 | ✅ 由 runner 切片，不需要新字段 |
| **§8.3 Ablation D** 换小 VLM | 同任务集,配置驱动 | 任意 split | ✅ 直接支持 |
| **§8.3 Ablation E** 4a / 4b 通道分离 | 同上 | 任意 split + preset `C_no_success_channel` / `C_no_failure_channel` | ✅ 已在 `paper2manim/ablations.py` 注册 |

### 2.2 与现有指标对齐

| 指标 (proposal §8.2) | 需要 P2M-Bench 提供 | 字段 |
|---|---|---|
| Pass@1 | 阈值 θ 由谁定 | `evaluation_rubric.thresholds` (可选，默认 avg≥85) |
| 平均反思轮数 | 任务标识即可 | `id` |
| VLM 平均分 | 三维维度定义 | `evaluation_rubric` 三维 |
| Human-VLM 一致性 | 同维度的人类分 | sidecar `human_scores/<id>.json` |
| Domain-Transfer Gain | domain 标签 + 训/测划分 | `domain` × `split` |

### 2.3 与 EMB 字段对齐（proposal §4.4）

P2M-Bench 字段 → EMB `Context` 字段 一一映射，让 RAG 检索可用样本元数据加权：

| P2M-Bench | EMB `Context`（`paper2manim/emb/schema.py`） |
|---|---|
| `target_unit.text` | `task_text` |
| `target_unit.type` | `scene_role`（subsection / equation / ... ）|
| `domain` | `domain_tags` |
| `paper_id` | `source_paper` |
| `target_unit.anchor.section` | `source_section` |
| `main_topics` | 拼接进 `task_text` 让 embedder 看到 |

⚠️ **不要把 `key_claims` / `reference_scene_plan` 写入 EMB**——它们是评测端 ground truth，写进去等于让 Coder 在 RAG 阶段直接拿到答案，污染 RQ1 主结论。

### 2.4 🚨 红线：`reference_scene_plan` ≠ training input

Proposal §4 / §6 / §10 反复强调："EMB 完全由系统在 paper-section 任务流上自学习生成，**不预置任何人工种子或外部视频库**"——这是本文与 Voyager 等已有 skill-library 工作的**关键区分点**，写在论文 §10 的对比表里。

P2M-Bench 引入 `reference_scene_plan` / `evaluation_rubric` 时**必须**遵守以下约束：

- ❌ **不得**作为 in-context examples 注入 Coder prompt（即使在 bootstrap 阶段）
- ❌ **不得**作为 EMB.success 的初始种子（即使在 Ablation C bootstrap=100 的极端配置）
- ❌ **不得**作为 storyboarder 的提示（storyboarder 只看 `paper_full_text` + `target_unit.text`）
- ✅ **只能**在评测时用于：（a）storyboard 对齐度的自动打分；（b）VLM/Human 打分的 anchor 参照
- ✅ **可以**在论文 §8 数据集描述里展示样例，但模型 pipeline 跑实验时取不到

这条约束需要在 `paper2manim/datasets/loader.py`（待新增）的接口层强制：split=dev/test/hard_test 时，loader **物理隐藏** `reference_scene_plan` 和 `evaluation_rubric.thresholds` 之外的字段，给模型只暴露 `(paper_full_text, target_unit, main_topics)` 三件套。

### 2.5 HF 展示表（README / dataset card）

| id | category | domain | paper_title | target_unit | source_type | main_topics | num_ref_scenes | reference_image |
|---:|---|---|---|---|---|---|---:|---|
| 0 | Equation | cs | Attention Is All You Need | Scaled Dot-Product Attention | equation | QK^T; softmax; weighted sum | 4 | 🖼 |
| 1 | Architecture | cs | Attention Is All You Need | Transformer Encoder | full_paper | attention; feed-forward; residual | 6 | 🖼 |
| 2 | Algorithm | math | Adam | Algorithm 1 | algorithm | momentum; adaptive lr | 5 | 🖼 |
| 3 | Figure | cs | NeRF | Volume Rendering Figure | figure | ray sampling; density; color | 5 | 🖼 |

### 2.6 EMB 健康度评测

proposal §4.4 的 EMB 不只是黑盒——它会暴露 `hit_count / last_used / first_seen` 等可观测信号（已实现，见 `paper2manim/emb/manager.py` 与 `paper2manim emb stats` CLI）。P2M-Bench 需要给出**用什么数据评 EMB 本身好不好用**的口径，否则 RQ1 的 Pass@1 提升可能掩盖 "EMB 实际几乎没被检索命中" 的问题。

| 评测项 | 定义 | 用 P2M-Bench 哪些字段做金标 |
|---|---|---|
| **Retrieval Precision@k** | 检索回的 top-k 条目中，与当前 task 在 `(domain, source_type, main_topics 重合数 ≥ 1)` 三元组上至少一致的比例 | `domain`, `source_type`, `main_topics` 作弱监督 ground truth |
| **Lesson Reuse Rate** | 全实验跑完后，`EMB.failure.hit_count > 0` 的条目占总条目比例；过低说明 4b 在写入但没在被检索 | EMB 自身 `provenance.hit_count` |
| **Success Memory Half-Life** | 一条 EMB.success 从首次写入到 `last_used` 超过 N 天的比例（衰减检测） | EMB 自身 `provenance.first_seen` / `last_used` |
| **4b Cross-Task Generalization** | 一条 failure pattern 是否能在 `(paper_id 不同, source_type 相同)` 的下游任务上被检索并命中 | `paper_id`, `source_type` 字段 |
| **EMB-Free Counterfactual Gain** | 同一 task 在 preset `C` vs `A` 下的 ΔPass@1，按 `domain × source_type` 切片报告 | `domain`, `source_type` 用于切片 |

实施位置：新增 `scripts/emb_health_report.py`，输入 `runs/exp_*/manifest.json` + `runs/_emb_exp1/<config>_seed_<n>/store.sqlite`，输出每个 (config, seed) 的健康度 5 项指标表，写入 `docs/emb-health-report.md`。这与 RQ1 Hero Plot 是正交评测，可同一份实验数据复用，不增加额外跑成本。

---

## 3. Split 定义

| Split | 规模 | 用途 | proposal 对应 | 现有 split 别名 |
|---|---:|---|---|---|
| `dev` | ~150 | Hero Plot 主曲线（A/B/C 对比） | §5 RQ1 / §8.1 稳态评估批次 | `bootstrap` ∪ `eval` |
| `test` | 50 | 论文报数前的最终保留集，不调参 | §8 (隐含) | — (新增) |
| `hard_test` | 50 | 跨域 (Domain B) + 长论文 + 含算法证明的难样本 | §5 RQ3 / §8.1 跨域测试集 | `cross_test` |
| `human_gold` | 100 | 人类专家逐维打分子集 | §5 RQ2 | — (新增) |

`human_gold` 与其他三个 split 正交——它从 `dev ∪ test ∪ hard_test` 中按 domain × source_type 分层抽样 100 条，附带 `human_scores/<id>.json`。这样不浪费样本：跑 RQ1 主实验时 `human_gold` 子集也产出 trace，可直接喂 RQ2。

---

## 4. 与当前 `p2m_v1.csv` 的关系（迁移路径）

### 4.1 字段差异

| 现 `p2m_v1.csv` | P2M-Bench | 处理 |
|---|---|---|
| `arxiv_id` | `paper_id` | 直接重命名 |
| `section` | `target_unit.anchor.section` + `target_unit.title` | 拆 |
| `domain` | `domain` | 不动 |
| `split ∈ {bootstrap, eval, cross_train, cross_test}` | `split ∈ {dev, test, hard_test, human_gold}` | 见映射表 |
| `expected_scene_count_min` | 删（用 `num_reference_scenes` 替） | — |
| — | `category / source_type / main_topics / key_claims / ...` | 新增 |

### 4.2 split 映射

| 现 split | 新 split | 备注 |
|---|---|---|
| `bootstrap` | `dev`（前 50–100 条） | `scripts/run_experiment.py` 用 task_idx 切片标记 bootstrap 段 |
| `eval` | `dev`（其余） | 同上，bootstrap 之后即稳态评估段 |
| `cross_train` | `dev`（domain=cs/math 的子集） | 训练域，与上面同表，不再单列 |
| `cross_test` | `hard_test` | 直接重命名 |

⚠️ `paper2manim/datasets/constants.py::SPLITS` 与 `--dataset-domain` 的 click.Choice 列表需同步更新，相关 8 条测试（`tests/test_cross_domain_and_dataset.py`）会失败，需补 fixture。

### 4.3 实施清单（最终版，一步到位）

**Code 改动**：
- 新增 `paper2manim/datasets/schema.py`（Pydantic v2 model 描述 §1.4 完整 JSON；含 `target_unit.type` 与 `target_unit.anchor` 字段集的一致性校验）
- 新增 `paper2manim/datasets/loader.py`：从 parquet / 多文件 jsonl 加载，对 `split ∈ {dev, test, hard_test}` 自动剥离 `reference_scene_plan` / `key_claims` / `common_misunderstandings` / `evaluation_rubric` 整条；只在 `mode="evaluation"` 显式调用时返回完整记录
- 更新 `paper2manim/datasets/constants.py`：`SPLITS = ("dev", "test", "hard_test", "human_gold")`；`DOMAINS` 不变
- 扩展 `paper2manim/parsers/arxiv_source.py`：实现 §5 表里 5 种 extractor（`extract_equation` / `extract_figure` / `extract_table` / `extract_algorithm` / `extract_full_paper`）
- 扩展 CLI：`paper2manim mvp2 --task <record.json>` 是新的主入口，所有 `source_type` 走同一路径；旧 `--arxiv / --section` 作 subsection 简写保留
- 新增 `paper2manim/utils/frame_sampler.py::whole_video_montage`，用于 `source_type=full_paper` 样本的 cross-scene VLM 评分
- 新增 `paper2manim/agents/full_paper_vlm_reviewer.py`：在三维基础上额外评 `narrative_coherence` + `symbol_consistency`（§1.6）
- 新增 `scripts/score_against_rubric.py`：把 trace 中的 VLM 分与样本 `evaluation_rubric` 对齐打分
- 新增 `scripts/emb_health_report.py`：输出 §2.6 五项指标，写入 `docs/emb-health-report.md`

**数据标注（300 条目标）**：
- Split 分布：`dev=150 / test=50 / hard_test=50 / human_gold=100`（`human_gold` 与前三 split 重叠，按 domain × source_type 分层抽样）
- Domain 分布：`cs ≥ 80 / math ≥ 60 / physics ≥ 50 / quantum ≥ 50 / econ ≥ 30`（quantum/econ 仅出现在 `hard_test`，承担 RQ3 跨域）
- Source_type 分布：`subsection ≥ 100 / equation ≥ 60 / figure ≥ 40 / algorithm ≥ 40 / table ≥ 30 / full_paper ≥ 30`
- 每条样本附 `annotation_log.jsonl`（双标 + 仲裁记录，见 §6.2）
- `human_gold` 100 条按 `human_scores/<id>.json` sidecar 单独存（3 维 × 0-100 × ≥2 标注者）

**测试覆盖**：
- `tests/test_p2m_bench_schema.py`：Pydantic 校验（missing required field / 未知 enum / `target_unit.type` 与 `anchor` 字段集一致性）
- `tests/test_p2m_bench_loader.py`：§6.4 的 4 条红线单测
- `tests/test_emb_health_report.py`：五项指标算法正确性 + 边界情况（空 EMB / 单条 EMB）
- `tests/test_score_against_rubric.py`：rubric ↔ score sidecar 读写、维度对齐

**发布**：
- HuggingFace 数据集 `Paper2Manim/p2m-bench`：主表 parquet + `papers/<paper_id>.txt` 全文 sidecar + `human_scores/<id>.json` + `annotation_log/<id>.jsonl`
- HF dataset card：§1.4 完整 schema + §2 RQ 映射表 + §6 标注流程 + §2.4 ⚠️ 红线声明
- 仓库 `examples/datasets/p2m_v1.csv` 删除，由 `paper2manim/datasets/loader.py` 直接读 HF

**实验闭环（投稿前要全部跑完）**：
- RQ1 Hero Plot：`A/B/C × ≥3 seeds × dev 150 条`，写 `docs/hero-plot.md`
- RQ2 VLM-Human 一致性：`human_gold` 100 条 × Pearson r + Spearman ρ + Cohen's κ，写 `docs/rq2-vlm-human.md`
- RQ3 跨域：cs/math 上 train EMB → frozen-EMB 在 quantum/econ `hard_test` 50 条上 test，写 `docs/rq3-cross-domain.md`
- 五个 ablation（A/B/C/D/E）：全在 `dev` 150 条上跑，写 `docs/ablations.md`

---

## 5. CLI 与 parser 的扩展点

当前 `paper2manim mvp2 --arxiv ID --section NAME` 只覆盖 `source_type=subsection`。其余 5 种 `source_type` 需要：

| source_type | parser 改动 | CLI 改动 |
|---|---|---|
| `subsection` | 已支持 | 已支持 |
| `equation` | `parsers/arxiv_source.py` 增 `extract_equation(label)`：返回包含该 `\label{eq:...}` 的 `equation` / `align` 环境 + 前后 200 chars 上下文 | `--target-unit equation:eq:scaled-dot` |
| `figure` | 增 `extract_figure(label)`：返回 `\caption{...}` + 引用该 fig 的所有段落 | `--target-unit figure:fig:transformer-arch` |
| `algorithm` | 增 `extract_algorithm(label_or_index)`：抓 `algorithm`/`algorithmic` 环境 | `--target-unit algorithm:1` |
| `table` | 类似 figure | `--target-unit table:tab:results` |
| `full_paper` | 抓整篇 abstract+intro+conclusion；length cap 拉到 8K | `--target-unit full_paper` |

实施建议：把 CLI 入口收敛到 `--task <path-to-p2m-bench-record.json>`，所有 source_type 走同一入口；旧 `--arxiv / --section` 保留为 backward-compat 简写。

---


## 6. 一句话总结

> P2M-Bench 把 `(arxiv_id, section)` 升级为 `(整篇论文, 待讲解 target_unit, 关键知识点, 评测 rubric, 参考分镜)` 五元组，让一份数据同时驱动 proposal 的 RQ1 主曲线、RQ2 人工一致性、RQ3 跨域泛化与全部五个 ablation；同时通过 loader 层的物理字段隐藏（§2.4 红线 + §6.4 单测兜底），保证 EMB 自学习的洁癖（proposal §4 / §10 的核心区分点）不被新引入的 ground truth 字段破坏。`full_paper` 样本额外引入 narrative_coherence + symbol_consistency 两维评跨 scene 衔接（§1.6）；EMB 健康度评测（§2.6）与 Hero Plot 正交共用同一份实验数据；标注流程与 IAA 口径（§6）封死评委两个常见拒稿理由。
