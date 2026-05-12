# VLM Multi-Dim Scoring 引入实验

> 2026-05-12 · 对应 [ResearchProposal.md](./ResearchProposal.md) §4.2（阶段 2：VLM 多维打分）和 §7（MVP 3.0 阶段 2/3）。
>
> 本文档记录 VLM 反思闭环在 `graphs/mvp2.py` 接图后的**首次真实端到端实验**，作为后续把 schema/阈值/best-of-N 等改进项收敛的基线参考。EMB（proposal §4 阶段 4）**不在本实验范围**。

---

## 1. 实验设置

| 项 | 值 |
|---|---|
| 输入 | arXiv `1706.03762`（*Attention Is All You Need*）`§Background` 节，2119 chars LaTeX 源 |
| 文本 LLM（storyboarder / coder / summarizer / reviewer / visual_reviser） | `claude-sonnet-4-6-1` via Azure（`https://claude-models-yangxu-resource.openai.azure.com/anthropic`，bearer auth） |
| VLM（vision_checker） | 同上 — 同一模型承担文本生成 + 视觉评审两套职责 |
| 文本反思 cap (`--max-retries`) | 2 |
| 视觉反思 cap (`--max-visual-revisions`) | 2 |
| 渲染质量 | `l`（480p15）|
| 每 scene 抽帧数（frame_sampler n_frames） | 4，hstack 横向拼接 |
| Run dir | `runs/20260512-221822-08f182/` |
| 输出视频 | `final/output.mp4`，**69.1s · 854×480@15 h264 · 1.3 MB**，5/5 scenes 全成功 concat |

**评分 schema（当前实现，与 proposal §4.2 偏离）**

实现为 6 维 × 1-5 分（来自 `prompts/vlm_scene_reviewer.md` 历史 schema）：

| 维度 | 缩写 | 含义 |
|---|---|---|
| paper_alignment | pa | 视觉是否真表达了 SceneSpec.paper_claim |
| visual_clarity | vc | 主视觉对象是否清晰可辨 |
| readability | rd | 文字可读性（大小、对比） |
| layout_balance | lb | 元素布局松紧 / 空白率 |
| visual_focus | vf | 视觉焦点是否突出 |
| animation_perceived | ap | 帧间动画感知到的变化是否清晰 |

> **Proposal §4.2 描述的是 3 维 × 0-100**（Logic Flow / Layout-Occlusion / Accuracy）。两者之间需要做一次收敛 — 见本文 §4「不足」。

---

## 2. 每 scene × 每 review 6 维评分表

由 `runs/20260512-221822-08f182/trace.jsonl` 提取（节点 `vlm_review`）。

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

---

## 3. 趋势汇总

| Scene | init_avg (v0) | final_avg (v2) | Δ | cap 触发的 visual revisions |
|---|---|---|---|---|
| TitleIntro | 2.67 | 2.50 | **−0.17** | 2 |
| SequentialVsParallel | 2.17 | 2.33 | +0.17 | 2 |
| ScaledDotProductAttention | 2.67 | 2.67 | +0.00 | 2 |
| MultiHeadAttention | 2.33 | 2.67 | +0.33 | 2 |
| TakeawayConclusion | 2.17 | 2.83 | **+0.67** | 2 |
| **平均** | 2.40 | 2.60 | **+0.20** | — |

`runs/20260512-221822-08f182/vlm_frames/<scene>_v{0,1,2}.png` 保留了每轮 montage，可肉眼复核 VLM 判断。

---

## 4. 关键观察

### 4.1 闭环工程实现是好的
- 每 scene 都按预期跑完 `render → frame_sampler → vlm_review → visual_revise → render → vlm_review → ...`
- `trace.jsonl` 完整记录每步评分 + montage 路径
- 5/5 scenes 最终 concat 成功，没崩、没 skip
- mock-VLM 单测（`tests/test_graph_mvp2_vlm.py`）覆盖 pass / revise→pass / cap / 关闭四条路径

### 4.2 VLM 当 critic 极为严苛
- **5/5 scenes 在 cap 内没有一个被 VLM 自发判为 `pass`** — 所有都是 cap 触发后 advance
- 初评分集中在 2.17-2.67（"acceptable" 下界），revision 后也只在 2.3-2.85 徘徊
- 主要扣分维度：`readability / layout_balance / visual_focus` 长期 ≤2，但这些受限于 480p15 渲染分辨率，VLM 让 coder 改 layout 救不回硬件分辨率
- 如果 proposal §4.2 的 ≥90/100 阈值（折合 ≥4.5/5）是真实目标，本实验下没有任何 scene 达标

### 4.3 Revision 有用，但效率低
- 平均 Δavg = +0.20 / scene（样本 N=5，统计意义弱）
- 最大正向：`TakeawayConclusion` +0.67（2.17 → 2.83）— 说明 VLM 反馈是有信号的
- 反例：`TitleIntro` 出现 **v1 (2.83) > v2 (2.50)** 的"改差了"情况，但当前实现只保留最后一版 → 输出反而劣化

### 4.4 成本侧
- 真实运行耗时：**≈13.5 分钟**（22:18 → 22:31），对比无 VLM 的 1 分 30 秒
- 每 scene 增加成本：1× frame_sampler（local ffmpeg，~3-5s）+ 1× VLM review（Claude vision，~25s）+ 1× LLM rewrite（Claude text，~20s）+ 1× re-render（~3s）≈ 50-60s per revision
- 5 scenes × 2 revisions = 10 次完整 revision，约 8-10 分钟纯 VLM/LLM 调用

---

## 5. 已识别的不足 → 改进方向

| # | 不足 | 改进 |
|---|---|---|
| 1 | 评分 schema 与 proposal §4.2 不一致（6 维×1-5 vs 3 维×0-100） | 选其一收敛，更新 prompt + `agents/vlm_scene_reviewer.py` 解析逻辑 + `tests/test_graph_mvp2_vlm.py` mock |
| 2 | best-of-N 没保留 — 最后一版未必最好 | 在 state 中累积 (rendered_video, score) 列表；advance 时挑 max；TitleIntro 这种"改差了"的 case 不再丢失初版 |
| 3 | Claude 几乎不自发 pass | 加 `avg ≥ θ` 旁路 pass（如 θ=3.5），与 VLM 自判 OR'd |
| 4 | 部分维度受限于渲染分辨率 | (a) 渲染升 720p30 后再喂 VLM；(b) prompt 里告诉 VLM "renderer is 480p15, do not penalize subpixel readability"；(c) 只取 `paper_alignment / visual_clarity / animation_perceived` 三维做 retry 决策，剩三维仅记录 |
| 5 | 单一 VLM = 单一 model，没法做 RQ2（Human-VLM 一致性） | 暂不在本 PR 范围；future work |
| 6 | EMB 完全没做 | proposal §4 阶段 4，**暂不考虑** |

---

## 6. 复现命令

```bash
# 1) 把 .env / config.yaml 配好（见 config.example.yaml；vision_checker role 必须指向 supports_vision=true 的模型）
cp config.example.yaml config.yaml
# 在 config.yaml 里把 vision_checker / scene_coder / final_summarizer 等 role 全指向一个 supports_vision=true 的 model
# 把 $TEXT_FLASH_API_KEY / $VISION_API_KEY 等环境变量填到 .env

# 2) 验证 yaml 路由
python -c "from paper2manim.llm import current_provider, current_model, is_vision_capable; \
  print(current_provider(), current_model('scene_coder'), is_vision_capable('vision_checker'))"

# 3) 跑实验
paper2manim mvp2 --arxiv 1706.03762 --section Background \
  --quality l --max-retries 2 --vlm --max-visual-revisions 2 \
  --allow-render-on-login

# 4) 提取评分表
python -c "
import json, sys
with open(sys.argv[1]) as f:
    rows = [json.loads(l) for l in f if json.loads(l).get('node') == 'vlm_review']
for r in rows:
    s = r['scores']; avg = sum(s.values())/6
    print(f\"{r['scene']:<28} v={r['v_rev']} {r['decision']:<8} pa={s['paper_alignment']} vc={s['visual_clarity']} rd={s['readability']} lb={s['layout_balance']} vf={s['visual_focus']} ap={s['animation_perceived']} avg={avg:.2f}\")
" runs/<your_run_id>/trace.jsonl
```
