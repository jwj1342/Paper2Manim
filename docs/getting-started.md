# 入门指南（非 HPC）

适用范围：Linux / macOS / Windows + WSL 的个人开发机或私有服务器。

跟着这份指南走完一遍，你将能在自己的电脑上：
- 跑通 **MVP 1.0**：把一段文字喂给 Paper2Manim，得到一个约 15 秒的 Manim 动画 mp4。
- （可选）准备好 **MVP 2.0** 的环境，把一篇 PDF 跑成多场景视频。

预计耗时：**首次 30–45 分钟**（其中 5–10 分钟在编译 `skia-pathops`）。之后跑一个新输入只需要几十秒到一两分钟。

> 如果你只想最快地看到一个视频结果，可以直接跳到 §2，照抄命令即可。

---

## 1. 先做一次性的准备

### 1.1 系统依赖

请确认下面这些命令在 shell 里能直接调用。括号里是常见包名，按你的系统装。

| 命令 | 用途 | Ubuntu/Debian | macOS (brew) | Windows |
|---|---|---|---|---|
| `python3.11` 或 `python3.12` | 主语言 | `python3.11 python3.11-venv` | `python@3.11` | 推荐 WSL2 装 Ubuntu；原生可用 [python.org installer](https://www.python.org/downloads/) |
| `ffmpeg` | Manim 编码视频 | `ffmpeg` | `ffmpeg` | `winget install ffmpeg` 或 WSL 的 apt |
| `latex`, `dvisvgm` | Manim 渲染数学公式 | `texlive texlive-latex-extra texlive-fonts-extra texlive-science tipa cm-super dvisvgm` | `--cask mactex-no-gui` 或 TinyTeX | WSL 同 Ubuntu / 原生用 [MikTeX](https://miktex.org) |
| `git`, `curl` | 拉代码、装 TinyTeX | 通常已有 | 通常已有 | 通常已有 |

无 sudo 权限时（共享服务器 / 集群）可以装[用户级 TinyTeX](https://yihui.org/tinytex/)：

```bash
curl -sL https://yihui.org/tinytex/install-bin-unix.sh | sh
export PATH=$HOME/.TinyTeX/bin/x86_64-linux:$PATH    # macOS 改成 universal-darwin
tlmgr install standalone preview doublestroke ms setspace rsfs relsize \
              ragged2e fundus-calligra microtype wasysym physics babel-english \
              cm-super xcolor amsmath amssymb dvisvgm
```

### 1.2 Python 版本

```bash
python3.11 --version    # 期望 3.11.x
# 或
python3.12 --version    # 期望 3.12.x
```

> Python 3.13 暂不支持（Manim 0.20.1 的部分 C 扩展尚未提供 3.13 wheel；正式适配后会更新）。
>
> 系统 Python 不对时推荐用 [pyenv](https://github.com/pyenv/pyenv)、conda 或 [uv](https://github.com/astral-sh/uv) 任意一种隔离一个干净的 3.11/3.12。

### 1.3 申请 MiMo API key

1. 到 [小米 MiMo 开放平台](https://www.xiaomimimo.com/) 注册账号、订阅 Token Plan。
2. 拿到一个 `tp-` 开头的 key（形如 `tp-abc...`），妥善保管，**不要发到聊天群或提交到 git**。

> 项目用到的 API 是 OpenAI 兼容的 `/v1/chat/completions`，base URL 是 `https://token-plan-cn.xiaomimimo.com/v1`。我们已经在代码里写死了正确的 endpoint，你只需要把 key 填进 `.env`。

---

## 2. 第一次运行（10 分钟）

### 2.1 拉代码

```bash
git clone <this-repo-url> Paper2Manim
cd Paper2Manim
```

### 2.2 创建并激活虚拟环境

```bash
python3.11 -m venv .venv             # 或 python3.12

# macOS / Linux:
source .venv/bin/activate

# Windows PowerShell:
.venv\Scripts\Activate.ps1

# Windows cmd:
.venv\Scripts\activate.bat
```

激活成功后，shell 提示符前会出现 `(.venv)`。后续所有命令都在这个激活的 venv 里跑。

### 2.3 装依赖

```bash
pip install --upgrade pip wheel
pip install -e ".[dev]"
```

预期最后一行：`Successfully installed paper2manim-0.1.0`。

> **首次安装时 `skia-pathops` 会从源码编译，耗时 5–10 分钟**，期间会从 `chromium.googlesource.com` 下载 skia 子模块（约 700 MB），需要联网。完全离线环境请预先制作 wheel。
>
> 如果编译卡住或失败，跳到 §5 常见问题。

### 2.4 配 API key

```bash
cp .env.example .env
```

用编辑器打开 `.env`，把这一行：
```
MIMO_API_KEY=tp-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```
改成你自己的 key。其它字段先保持默认。

> `.env` 已经在 `.gitignore` 里，不会被误提交。文件权限建议 `chmod 600 .env`。

验证配置：
```bash
paper2manim info
```
应输出类似：
```json
{
  "MIMO_BASE_URL": "https://token-plan-cn.xiaomimimo.com/v1",
  "MIMO_API_KEY_set": true,
  ...
}
```
关键是 `MIMO_API_KEY_set: true`。

### 2.5 跑 Hello World

```bash
paper2manim mvp1 --input examples/mvp1/pythagorean.txt
```

正常会看到（大约 30–60 秒）：

```
MVP 1.0 run 20260510-080654-3a1159: The Pythagorean theorem ...
[storyboarder] input chars=410          ← 调用 MiMo 出剧本
[coder] scene=PythagorasIntro iter=0    ← 调用 MiMo 写代码
[render] manim ... -ql ...              ← 调用 manim 渲染
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Field            ┃ Value                   ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ title            │ The Pythagorean Theorem │
│ scenes           │ 1                       │
│ ...              │ ...                     │
│ final_video_path │ runs/.../output.mp4     │
└──────────────────┴─────────────────────────┘
```

如果中间报错，跳到 §5 常见问题。

### 2.6 看视频

视频在 `runs/<run_id>/final/output.mp4`：

```bash
# macOS:
open runs/*/final/output.mp4

# Linux 桌面:
xdg-open runs/*/final/output.mp4

# WSL:
explorer.exe $(wslpath -w runs/*/final/output.mp4)

# Windows PowerShell:
Start-Process runs\*\final\output.mp4

# 远程服务器（无显示）→ 拉到本地看:
scp user@server:/path/to/Paper2Manim/runs/.../final/output.mp4 .
```

应该看到一个十几秒的视频：黑底，标题 "The Pythagorean Theorem" 淡入，画一个直角三角形（蓝/绿/红三色边），写出 $a^2+b^2=c^2$，最后淡出。

**如果看到这个，整条 LLM → Manim 流水线就通了。**

---

## 3. 多玩一玩

### 3.1 换一个输入

仓库带了 5 个示例文本：

```bash
ls examples/mvp1/
# eulers_identity.txt   fourier_intuition.txt
# linear_regression_intuition.txt   newton_second_law.txt   pythagorean.txt

paper2manim mvp1 --input examples/mvp1/fourier_intuition.txt
```

直接给字符串也行：

```bash
paper2manim mvp1 --input "The chain rule: d/dx[f(g(x))] = f'(g(x)) g'(x)."
```

写一个新的 `.txt` 放进 `examples/mvp1/` 也是同样的用法。每段输入控制在 **3–5 句话、< 500 字符**效果最稳。

### 3.2 调整渲染质量

| 选项 | 分辨率/帧率 | 一个简单 scene 的渲染耗时 |
|---|---|---|
| `--quality l`（默认） | 480p15 | 10–30 秒 |
| `--quality m` | 720p30 | 30–90 秒 |
| `--quality h` | 1080p60 | 1–4 分钟 |

调 prompt 时用 `l`，确认效果好了再 `m`/`h` 出最终成品。

### 3.3 仅生成代码、跳过渲染

调 prompt 时如果不想等渲染：

```bash
paper2manim mvp1 --input examples/mvp1/pythagorean.txt --no-render
# 然后看 runs/<run_id>/attempts/00_*.py 即可
```

### 3.4 改 prompt

`prompts/*.md` 里的内容是热加载的——每次跑都会重新读，不需要 `pip install -e .` 重装。直接改 `prompts/storyboarder.md` 或 `prompts/coder.md`，再跑一次就生效。

```bash
# 建议在 git 里建分支再改，方便对比效果
git checkout -b prompt-tweak-v1
$EDITOR prompts/coder.md
paper2manim mvp1 --input examples/mvp1/pythagorean.txt
```

### 3.5 进 MVP 2.0：跑一篇真实论文

输入两种来源任选其一：**`--arxiv` 是推荐路径**（直接抓作者上传的 LaTeX 源码，公式 100% 准确、双栏顺序天然正确、无需 3GB Marker 权重）；`--pdf` 是兜底（走 Marker 解析）。

**路径 A：`--arxiv`（推荐）**

不需要装额外依赖：

```bash
# 整篇论文
paper2manim mvp2 --arxiv 1706.03762

# 只跑某一节（按 \section{...} 标题做大小写不敏感子串匹配，强烈推荐先用这个跑通）
paper2manim mvp2 --arxiv 1706.03762 --section "Background"

# arXiv URL 也接受
paper2manim mvp2 --arxiv https://arxiv.org/abs/2401.12345v2
```

作者只上传了 PDF（无 LaTeX 源码）时，会自动回退到 `--pdf` 路径。

**路径 B：`--pdf`（本地 PDF）**

需要先装 Marker：

```bash
pip install -e ".[mvp2]"     # 装 marker-pdf；首次会下载 ~3GB 模型权重到 ~/.cache/huggingface
mkdir -p examples/mvp2
cp /your/path/some_paper.pdf examples/mvp2/
paper2manim mvp2 --pdf examples/mvp2/some_paper.pdf
```

期望流水线（两条路径合流后一样）：

1. **Parser** 拿到论文文本（arXiv flatten `\input{}` 或 Marker markdown）
2. **Summarizer** 提取关键贡献 / 公式 / 主概念
3. **Storyboarder** 拆 2–5 个 scene
4. 每个 scene 在自己的子图里独立跑：**Coder ⇄ Render ⇄ Reviewer** 文本反思（最多 `PAPER2MANIM_MAX_RETRIES`，默认 3 次）
5. **Concat** ffmpeg 把成功的 scene 串成最终 mp4

默认所有 scene 串行。要并行加速看 §3.7。

### 3.6 打开 VLM 视觉反思（MVP 3.0 阶段 2/3）

`--vlm` 会在每个 scene 渲染成功后再加一层视觉反思闭环：`frame_sampler` 抽 4 帧拼成 montage → `vlm_review` 按 3 维 × 0-100 打分（`logic_flow` / `layout_occlusion` / `accuracy`，avg ≥ 90 自动 pass）→ `visual_revise` 改代码 → 重渲染。

```bash
paper2manim mvp2 --arxiv 1706.03762 --section "Background" --vlm
# 视觉反思每 scene 最多 2 轮，用 --max-visual-revisions N 调
paper2manim mvp2 --arxiv 1706.03762 --section "Background" --vlm --max-visual-revisions 3
```

VLM 需要在 `config.yaml` 里配 `supports_vision: true` 的模型（例如 Claude Opus 4.7 / Sonnet 4.6 via Azure，或 GPT-4o）。参考 `config.example.yaml`。

### 3.7 打开 Episodic Memory Bank（MVP 3.0 阶段 4，自进化）

`--emb` 会在每个 scene 写代码前从历史库 RAG 检索 top-k 成功示例 + 失败教训注入 Coder prompt，并在 run 结束时把本次的高分 scene + 失败→成功转化蒸馏回库。第一次跑库是空的、行为等价于关闭；跑多篇之后才显现进化。

```bash
paper2manim mvp2 --arxiv 1706.03762 --section "Background" --vlm --emb
# 默认库放在 $PAPER2MANIM_RUNS_DIR/_emb；CI / 离线时加 --emb-fake-embedder 跳过 sentence-transformers
paper2manim mvp2 --arxiv 1706.03762 --vlm --emb --emb-fake-embedder
```

### 3.8 并行化（多 scene 并发）

```bash
# 5 个 scene 同时跑；建议同时给一个 render-concurrency 上限避免 manim 把 CPU 打爆
paper2manim mvp2 --arxiv 1706.03762 --scene-parallelism 5 --render-concurrency 3

# 共用 LLM API 的全局 RPS 上限（防 429）
paper2manim mvp2 --arxiv 1706.03762 --scene-parallelism 5 --llm-rps 1.5
```

默认 `--scene-parallelism 1`（串行，与 PR #19 之前行为零差异）；`--render-concurrency` / `--llm-rps` 默认不限。

### 3.9 其它常用调试 flag

- `--max-retries N`：给文本反思更多 / 更少机会（默认 3）
- `--no-render`：只跑 LLM 端（parser → summarizer → storyboarder → coder），不调 manim。验证 prompt 时省渲染开销
- `paper2manim -v mvp2 ...`：DEBUG 日志

---

## 4. 看懂输出文件

每次运行都会创建一个独立目录 `runs/<run_id>/`。建议按下面顺序排查：

```
runs/20260510-080654-3a1159/
├── input.txt                  ← 输入备份
├── parsed.md                  ← (MVP 2.0) Marker 输出
├── summary.json               ← (MVP 2.0) 论文结构化摘要
├── storyboard.json            ← LLM 拆出来的分镜剧本（看这个能秒懂 LLM 是不是听懂了）
├── attempts/                  ← 每个 scene 的代码 + 渲染结果
│   ├── 00_PythagorasIntro.py            ← 第 1 轮代码
│   ├── 00_PythagorasIntro.render.json   ← 第 1 轮渲染结果（成功就是空错误）
│   ├── 01_PythagorasIntro.py            ← 反思后的第 2 轮（仅 MVP 2.0 触发）
│   └── ...
├── final/
│   └── output.mp4             ← 最终视频
└── trace.jsonl                ← 每个 graph 节点的 input/output 时间线
```

排查思路：
- 视频不对 → 看 `attempts/<latest>.py` 看 LLM 写了啥 Manim 代码
- 渲染失败 → 看 `attempts/<latest>.render.json`，里面有结构化的 `category` / `traceback_tail` / `tex_log_excerpt`
- LLM 输出离谱 → 看 `storyboard.json` 里 LLM 是不是误解了输入

---

## 5. 常见问题

### Q1: `pip install` 时 `skia-pathops` 编译失败

错误形如：`Failed to build skia-pathops`。原因通常是网络拉不到 chromium 的 skia 源码。

```bash
# 1. 检查网络能否访问 chromium.googlesource.com
curl -sI https://chromium.googlesource.com | head -1   # 应该 200

# 2. 如果在公司/学校代理后面，确认 https_proxy 设了
echo $https_proxy

# 3. 实在拉不到：可以从一台能访问的机器上 pip wheel 出 skia-pathops 的 wheel 拷过来
#    pip wheel skia-pathops -w ./wheels
#    然后在目标机：pip install ./wheels/skia_pathops-*.whl
```

### Q2: `openai.AuthenticationError: 401 Invalid API Key`

三件事按顺序检查：

1. **base_url**：`paper2manim info` 应该显示 `MIMO_BASE_URL: https://token-plan-cn.xiaomimimo.com/v1`。如果是 `api.xiaomimimo.com`，会拒掉 `tp-` key。修 `.env`：
   ```
   MIMO_BASE_URL=https://token-plan-cn.xiaomimimo.com/v1
   ```
2. **key 本身**：登录 MiMo 控制台确认 key 没过期，必要时新建一个。
3. **拼写**：key 头要是完整的 `tp-...`，不要漏掉 `tp-` 前缀。

直接 curl 验证（最快）：
```bash
KEY=<你的 key>
curl -sS https://token-plan-cn.xiaomimimo.com/v1/models -H "Authorization: Bearer $KEY"
# 应该返回一个 JSON 列出 mimo-v2.5, mimo-v2.5-pro 等
```

### Q3: `LatexError: ! Undefined control sequence ...`

LaTeX 缺包。装 TinyTeX 时多带几个：

```bash
tlmgr install amsmath amssymb amsfonts cm-super doublestroke physics
```

或者直接装全套：
```bash
sudo apt install texlive-full     # 体积大但一劳永逸
```

### Q4: `from manim import *` 失败

```bash
which python              # 确认是 .venv 里的
pip list | grep -i manim  # 应能看到 manim 0.20.1
```

如果不在 venv 里：`source .venv/bin/activate` 重新激活。

### Q5: 渲染特别慢

- 默认 `--quality l`（480p15）一个简单 scene 应在 30 秒内
- 第一次跑 manim 时它会预编译一些 cache（位置 `~/.manim_cache/`），后续会快
- 复杂 scene 才慢；测试时让 LLM 写"简单 scene"

### Q6: API 返回慢、调用频繁出错

MiMo 平台可能有 RPS 限速。`PAPER2MANIM_DEFAULT_MODEL=flash`（→ `mimo-v2.5`）已经是较小较快的模型；只有 `summarizer` 在长 PDF 时才升到 `pro`（`mimo-v2.5-pro`）。如果还是慢，换一个时间段再试。

### Q7: 想看更详细的日志

```bash
paper2manim -v mvp1 --input ...    # DEBUG 级日志
```

也可以看 `runs/<run_id>/trace.jsonl`——它把每个节点的输入输出都按行记录了。

---

## 6. 下一步

- 读 [`docs/progress.md`](./progress.md) 了解项目当前进度与待办
- 读 [`docs/ResearchProposal.md`](./ResearchProposal.md) 了解研究背景与目标
- 读 `paper2manim/state.py` 理解 LangGraph 的 state schema
- 读 `paper2manim/graphs/mvp2.py` 理解反思闭环的拓扑
- 读 `prompts/*.md` 了解每个 agent 的人设和职责，并尝试改它们看看效果
- 跑一遍单测：`pytest -m "not slow" -v`
