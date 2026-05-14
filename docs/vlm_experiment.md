# VLM Multi-Dim Scoring 引入实验

> 2026-05-13 · 对应 [ResearchProposal.md](./ResearchProposal.md) §4.2（阶段 2：VLM 多维打分）和 §7（MVP 3.0 阶段 2/3）。
>
> 本文档记录 VLM 反思闭环在 `graphs/mvp2.py` 接图后的真实端到端实验。**两次 baseline 都保留**：
>
> - **新 baseline (2026-05-13)**：proposal §4.2 canonical schema = 3 维 × 0-100（logic_flow / layout_occlusion / accuracy）+ avg ≥ 90 auto-pass bypass，VLM = Claude Opus 4.7（Azure）。详见 §1–§4。
> - **旧 baseline (2026-05-12)**：早期 6 维 × 1-5 schema（已废弃），VLM = Claude Sonnet 4.6（Azure）。仅保留作对照，见 [§A 附录](#a-附录旧-6-维--1-5-baseline2026-05-12)。
>
> EMB（proposal §4 阶段 4）**不在本实验范围**。

---

## 1. 实验设置（新 3 维 baseline）

| 项 | 值 |
|---|---|
| 输入 | arXiv `1706.03762`（*Attention Is All You Need*）`§Background` 节，2119 chars LaTeX 源 |
| 文本 LLM（storyboarder / coder / summarizer / reviewer / visual_reviser） | `claude-opus-4-7-1` via Azure（`https://claude-models-yangxu-resource.openai.azure.com/anthropic`，bearer auth，`omit_temperature: true`） |
| VLM（vision_checker） | 同上 — 同一模型承担文本生成 + 视觉评审 |
| 文本反思 cap (`--max-retries`) | 2 |
| 视觉反思 cap (`--max-visual-revisions`) | 2 |
| 渲染质量 | `l`（480p15）|
| 每 scene 抽帧数（frame_sampler n_frames） | 4，hstack 横向拼接 |
| 评分 schema | proposal §4.2 canonical 3 维 × 0-100 |
| auto-pass bypass | `parse_vlm_response`: decision == "revise" && avg(scores) ≥ 90 → 升级 pass，原始 verdict 保留在 `raw_decision` |
| Run dir | `runs/20260513-152258-45dd9a/` |
| 输出视频 | `final/output.mp4`，**69.8s · 854×480@15 h264 · 1.04 MB**，5/5 scenes 全成功 concat |
| 端到端耗时 | ≈7 分 20 秒（15:22:58 → 15:30:18） |

**评分 schema 三维定义**（与 `prompts/vlm_scene_reviewer.md` 对齐）：

| 维度 | 含义 | 0 锚点 | 100 锚点 |
|---|---|---|---|
| `logic_flow` | 视觉是否端到端讲清 SceneSpec.paper_claim；动画 beat 是否首尾连贯 | 画的是无关内容 | 渲染帧本身就让观众理解 claim |
| `layout_occlusion` | 可读性 + 元素之间的空间关系；扣分项：text overlap、cropped、label collision、wall-of-text、empty space | 不可读 | 干净布局，主视觉无歧义 |
| `accuracy` | 数学符号 / 公式 / 坐标轴 / 比例 / 标签是否正确且与 paper_claim 一致 | 有可见错误（错号、错变量、坐标轴颠倒） | 屏幕上无任何事实错误 |

---

## 2. 每 scene × 每 review 3 维评分表

由 `runs/20260513-152258-45dd9a/trace.jsonl` 提取（节点 `vlm_review`），每行包含 `raw_decision`/`average_score` 供 auto-pass bypass 审计：

| Scene | v_rev | decision | raw_decision | logic_flow | layout_occlusion | accuracy | avg |
|---|---|---|---|---|---|---|---|
| TitleScene | 0 | **pass** | pass | 88 | 85 | 95 | 89.33 |
| ArchitectureComparison | 0 | revise | revise | 60 | 45 | 70 | 58.33 |
| ArchitectureComparison | 1 | revise | revise | 70 | 45 | 75 | 63.33 |
| ArchitectureComparison | 2 | revise | revise | 70 | 45 | 75 | 63.33 |
| ScaledDotProductAttention | 0 | revise | revise | 55 | 35 | 60 | 50.00 |
| ScaledDotProductAttention | 1 | revise | revise | 72 | 65 | 85 | 74.00 |
| ScaledDotProductAttention | 2 | revise | revise | 78 | 70 | 85 | 77.67 |
| MultiHeadAttention | 0 | revise | revise | 70 | 55 | 60 | 61.67 |
| MultiHeadAttention | 1 | revise | revise | 70 | 55 | 60 | 61.67 |
| MultiHeadAttention | 2 | revise | revise | 70 | 55 | 72 | 65.67 |
| Takeaway | 0 | revise | revise | 78 | 55 | 80 | 71.00 |
| Takeaway | 1 | **pass** | pass | 92 | 90 | 95 | 92.33 |

总计 **12 vlm_review events / 7 visual_revise events**。

---

## 3. 趋势汇总

| Scene | init_avg (v0) | final_avg | Δ | visual_revise 次数 | 出口原因 |
|---|---|---|---|---|---|
| TitleScene | 89.33 | 89.33 | +0.00 | 0 | VLM 自发 pass |
| ArchitectureComparison | 58.33 | 63.33 | +5.00 | 2 | cap 触发 |
| ScaledDotProductAttention | 50.00 | 77.67 | **+27.67** | 2 | cap 触发 |
| MultiHeadAttention | 61.67 | 65.67 | +4.00 | 2 | cap 触发 |
| Takeaway | 71.00 | 92.33 | **+21.33** | 1 | VLM 自发 pass |
| **平均（仅 revise 类 4 scenes）** | 60.25 | 74.75 | **+14.50** | 1.75 | — |
| **平均（全 5 scenes）** | 66.07 | 77.67 | **+11.60** | 1.4 | — |

`runs/20260513-152258-45dd9a/vlm_frames/<scene>_v{0,1,2}.png` 保留了每轮 montage，可肉眼复核 VLM 判断。

---

## 4. 关键观察

### 4.1 Schema 收敛带来"自发 pass"的解锁

旧 baseline 中 Sonnet 4.6 在 6 维 × 1-5 schema 下 **0/5 scenes 自发判 pass**，全部 cap 触发；新 baseline 中 Opus 4.7 在 3 维 × 0-100 schema 下 **2/5 scenes 自发判 pass**（TitleScene 一次过、Takeaway 1 次 revision 后）。要分清两个变量：

- **模型差异**（Sonnet 4.6 → Opus 4.7）：Opus 整体更"果断"，会给到 88-95 的高分而不是只给 3-4
- **schema 差异**（6 维 × 1-5 → 3 维 × 0-100）：维度更少且各维独立，0-100 让 "85" 这种"还行但不是顶配"的判断有地方落

这次实验两个变量同时变了，无法严格归因。**未来需要做 schema ablation**（同一模型分别在新旧 schema 跑同输入）。

### 4.2 auto-pass bypass 在本次 baseline 中 **0 次触发**

理由：所有 12 个 reviews 里 `raw_decision == decision` 全部成立。最接近触发的是 **TitleScene avg=89.33**（差 0.67 就过 90 阈值），但 VLM 自己就判了 pass，bypass 没机会上场。

这说明：
1. Opus 4.7 在 3 维 × 0-100 schema 下**自发 pass 行为已经够积极**，bypass 当前更像"防御性兜底"而非"流量驱动器"
2. 阈值 90 偏高 — 真到 90 的时候 VLM 大概率已经自己 pass 了。要让 bypass 真正起作用，要么降阈值（如 80），要么换更保守的 VLM 模型。本 PR 暂不动，留待数据更多时再调

`tests/test_vlm_response_parse.py` 的 4 条 bypass 单测保证 wiring 是对的；真实触发数据等下一个 baseline 收集。

### 4.3 Revision 效率显著提升

- 平均 Δavg = **+14.5 分**（仅 revise 类 4 scenes），相当于满分 100 中 14.5%
- 最大正向：**ScaledDotProductAttention +27.67**（50.00 → 77.67，2 次 revision）— Opus 把数学公式从乱→可读
- 次最大：**Takeaway +21.33**（71.00 → 92.33，1 次 revision）— 唯一一个 revision 后真的"进 pass 区"的 scene
- 反例存在但量级很小：MultiHeadAttention v0→v1 完全平移（61.67→61.67），同分但具体 score 分布变了；layout_occlusion=55 在 3 个 revision 里都没动 → VLM 反复指出 layout 问题但 coder 改不动

对照旧 baseline 的 Δavg=+0.20/5 = **4%**，新 baseline +14.5/100 = **14.5%**。**3 维 schema 信号噪比显著好**，但样本 N=5，统计意义弱。

### 4.4 成本 / 耗时

- 端到端：**7 分 20 秒**（旧 baseline 13.5 分钟，几乎减半）
- 主因：2 个 scenes 在 cap 之前 pass 了（TitleScene 0 次 revision，Takeaway 1 次），少跑了 2 × ~50-60s 的 revise→re-render 周期
- 每 scene 增加成本：仍然是 1× frame_sampler（local ffmpeg，~3-5s）+ 1× VLM review（Claude vision，~10-15s）+ 1× LLM rewrite（Claude text，~15-20s）+ 1× re-render（~3s）≈ 35-45s per revision（Opus 4.7 略慢于 Sonnet 4.6 但差距不显著）
- 真实 visual_revise 调用：**7 次**（4 cap-scenes × 2 + Takeaway × 1 − ArchitectureComparison/ScaledDotProduct/MultiHead 各-0 = 7）— 旧 baseline 是 10 次
- VLM review 调用：**12 次**（5 scenes 的 v0 + 4 scenes 的 v1 + 3 scenes 的 v2 = 12）— 旧 baseline 是 15 次

---

## 5. 已识别的不足 → 改进方向

| # | 不足 | 状态 / 改进 |
|---|---|---|
| 1 | ~~评分 schema 与 proposal §4.2 不一致（6 维×1-5 vs 3 维×0-100）~~ | ✅ **issue #12 已解**：本 PR 收敛到 3 维 × 0-100；prompt + agent + mock + tests 全配套 |
| 2 | ~~Claude 极少自发 pass，每 scene 跑满 cap~~ | ✅ **本 PR 顺带解**：3 维 schema + auto-pass bypass（avg≥90）；本次 baseline 2/5 自发 pass，无需 bypass 上场 |
| 3 | **best-of-N 没保留** — 最后一版未必最好 | 🟡 未解：当前实现只保留最后一版 rendered video；遇到 v1 > v2 时（旧 baseline TitleIntro v1=2.83→v2=2.50；新 baseline 未出现明显 case）输出反而劣化。需在 state 累积 (rendered_video, avg) 列表，advance 时挑 max |
| 4 | 部分维度受限于渲染分辨率 | 🟡 未解：`layout_occlusion` 在 480p15 下天然吃亏（ArchitectureComparison 三次 revision 都卡在 45）。三个改进选项：(a) 渲染升 720p30 后再喂 VLM；(b) prompt 里告诉 VLM "renderer is 480p15, do not penalize subpixel readability"；(c) 给 layout_occlusion 加权重折扣 |
| 5 | 单一 VLM = 单一 model，没法做 RQ2（Human-VLM 一致性） | 🟡 future work；暂不在本 PR 范围 |
| 6 | ~~EMB 完全没做~~ | ✅ **PR #16 已 land**：`paper2manim/emb/` 双通道 schema + SQLite store + Faiss 检索 + 蒸馏 + RAG 注入；CLI `--emb` 开启。剩余 cold-record pruning（#17）与 RAG 注入位置 A/B（#18），不在本实验范围 |
| 7 | auto-pass bypass 阈值 90 在本次 baseline 触发 0 次 | 🟡 数据不足，留待更多 baseline 后再调（候选：降到 80） |

---

## 6. 复现命令

```bash
# 1) 把 .env / config.yaml 配好（见 config.example.yaml；vision_checker role 必须指向 supports_vision=true 的模型）
cp config.example.yaml config.yaml
# 在 config.yaml 里把 vision_checker / scene_coder / final_summarizer 等 role 全指向一个 supports_vision=true 的 model
# 把 $TEXT_FLASH_API_KEY / $VISION_API_KEY / $AZURE_CLAUDE_API_KEY 等环境变量填到 .env

# 2) 验证 yaml 路由
python -c "from paper2manim.llm import vision_checker_config; print(vision_checker_config().name, vision_checker_config().supports_vision)"

# 3) 跑实验
paper2manim mvp2 --arxiv 1706.03762 --section Background \
  --quality l --max-retries 2 --vlm --max-visual-revisions 2 \
  --allow-render-on-login

# 4) 提取 3 维评分表（含 raw_decision 与 average_score 字段）
python <<'PY'
import json, sys
from pathlib import Path

run_dir = Path(sys.argv[1])
rows = [json.loads(l) for l in (run_dir / "trace.jsonl").open()]
for r in rows:
    if r.get("node") != "vlm_review":
        continue
    s = r.get("scores") or {}
    print(f"{r['scene']:<28} v={r.get('v_rev')} {r.get('decision'):<8} "
          f"raw={r.get('raw_decision','?'):<7} "
          f"lf={s.get('logic_flow','?')} "
          f"lo={s.get('layout_occlusion','?')} "
          f"ac={s.get('accuracy','?')} "
          f"avg={r.get('average_score', 0):.2f}")
PY  runs/<your_run_id>
```

---

## A. 附录：旧 6 维 × 1-5 baseline（2026-05-12）

> 保留作 schema 收敛前的对照。旧 schema 已废弃，详见 §1 与 §4.1。

### A.1 旧 baseline 实验设置

| 项 | 值 |
|---|---|
| VLM | `claude-sonnet-4-6-1` via Azure |
| Schema | 6 维 × 1-5（paper_alignment / visual_clarity / readability / layout_balance / visual_focus / animation_perceived） |
| Run dir | `runs/20260512-221822-08f182/`（runs/ 已 gitignore，本地未保留） |
| 输出视频 | 69.1s · 854×480@15 h264 · 1.3 MB |
| 端到端耗时 | ≈13.5 分钟（22:18 → 22:31） |

### A.2 旧 6 维评分表

| Scene | v_rev | decision | pa | vc | rd | lb | vf | ap | avg |
|---|---|---|---|---|---|---|---|---|---|
| TitleIntro | 0 | revise | 3 | 3 | 3 | 2 | 2 | 3 | 2.67 |
| TitleIntro | 1 | revise | 4 | 3 | 3 | 2 | 2 | 3 | 2.83 |
| TitleIntro | 2 | revise | 3 | 3 | 2 | 2 | 2 | 3 | 2.50 |
| SequentialVsParallel | 0 | revise | 3 | 2 | 2 | 2 | 2 | 2 | 2.17 |
| SequentialVsParallel | 1 | revise | 3 | 2 | 2 | 3 | 2 | 3 | 2.50 |
| SequentialVsParallel | 2 | revise | 3 | 2 | 2 | 2 | 2 | 3 | 2.33 |
| ScaledDotProductAttention | 0 | revise | 3 | 3 | 2 | 2 | 3 | 3 | 2.67 |
| ScaledDotProductAttention | 1 | revise | 3 | 3 | 2 | 2 | 2 | 3 | 2.50 |
| ScaledDotProductAttention | 2 | revise | 3 | 3 | 2 | 2 | 3 | 3 | 2.67 |
| MultiHeadAttention | 0 | revise | 3 | 2 | 2 | 2 | 2 | 3 | 2.33 |
| MultiHeadAttention | 1 | revise | 3 | 3 | 2 | 3 | 3 | 3 | 2.83 |
| MultiHeadAttention | 2 | revise | 3 | 3 | 2 | 2 | 3 | 3 | 2.67 |
| TakeawayConclusion | 0 | revise | 3 | 2 | 2 | 2 | 2 | 2 | 2.17 |
| TakeawayConclusion | 1 | revise | 3 | 2 | 2 | 2 | 2 | 2 | 2.17 |
| TakeawayConclusion | 2 | revise | 4 | 3 | 3 | 2 | 2 | 3 | 2.83 |

旧趋势汇总：平均 Δavg = **+0.20 / scene**（满分 5 中的 +4%），5/5 scenes 都 cap 触发，0 个自发 pass。
