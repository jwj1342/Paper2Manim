<div align="center">

<h1>🎬 ManimAgent</h1>

<h3>面向视觉教育的自进化多模态智能体</h3>

<p><b>EMNLP 2026 · 主会</b></p>

<p>
<a href="https://jwj1342.github.io">Wenjia Jiang</a><sup>1</sup>,
<a href="https://zongyuan-cai.github.io/">Zongyuan Cai</a><sup>2</sup>,
<a href="https://github.com/rirko">Yuanhang Shao</a><sup>2</sup>,
<a href="https://winterccc.github.io/">Chenru Wang</a><sup>3</sup>,
<a href="https://burgerio-823.github.io">Boyan Han</a><sup>2</sup>,
<a href="https://github.com/KMnO4-zx">Zhixue Song</a><sup>4</sup>,
Keyu Chen<sup>5</sup>,
<a href="https://answ.in/">Shengwei An</a><sup>3</sup>,
<a href="https://yangxuntu.github.io/">Xu Yang</a><sup>2</sup>,
<a href="https://homepage.zhouyang.me/">Zhou Yang</a><sup>1</sup>
</p>

<p>
<sup>1</sup>University of Alberta &nbsp;
<sup>2</sup>Southeast University &nbsp;
<sup>3</sup>Virginia Tech &nbsp;
<sup>4</sup>Emotion Machine Inc. &nbsp;
<sup>5</sup>Vivavia Inc.
</p>

<img src="assets/affiliations.png" alt="University of Alberta, Southeast University, Virginia Tech, Emotion Machine, Vivavia" width="520">

<p>
<a href="https://2026.emnlp.org/"><img src="https://img.shields.io/badge/EMNLP%202026-Main%20Conference-1f6feb?style=for-the-badge" alt="EMNLP 2026 Main Conference"></a>
<a href="https://arxiv.org/abs/2606.30296"><img src="https://img.shields.io/badge/arXiv-2606.30296-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv 2606.30296"></a>
<a href="https://manimagent.github.io/"><img src="https://img.shields.io/badge/Project-Page-2ea44f?style=for-the-badge&logo=githubpages&logoColor=white" alt="项目主页"></a>
</p>

<p>
<a href="https://github.com/jwj1342/Paper2Manim/actions/workflows/ci.yml"><img src="https://github.com/jwj1342/Paper2Manim/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
<img src="https://img.shields.io/badge/python-3.11%20%7C%203.12-3776ab?logo=python&logoColor=white" alt="Python 3.11 | 3.12">
<a href="https://www.manim.community/"><img src="https://img.shields.io/badge/Manim%20CE-0.20.1-e07a5f" alt="Manim CE 0.20.1"></a>
<a href="https://www.langchain.com/langgraph"><img src="https://img.shields.io/badge/LangGraph-1.x-1c3c3c?logo=langchain&logoColor=white" alt="LangGraph 1.x"></a>
</p>

<p><a href="README.md">English</a> · <b>中文</b></p>

<p>
<a href="#-亮点">亮点</a> ·
<a href="#-实验结果">实验结果</a> ·
<a href="#-工作流程">工作流程</a> ·
<a href="#-快速开始">快速开始</a> ·
<a href="#-引用">引用</a>
</p>

</div>

> **一句话总结** —— 反思机制能让智能体在*单个任务内*纠正错误，但这些经验在进入下一个任务前就被丢弃了，这就是**跨任务遗忘**。ManimAgent 用一个从自身任务流中成长起来的双通道**情景记忆库（Episodic Memory Bank，EMB）**把经验延续下去，既不更新模型权重，也不依赖人工种子数据。

<p align="center">
  <img src="assets/teaser.png" alt="仅靠反思的智能体会在任务之间遗忘，ManimAgent 通过情景记忆库存储并检索经验" width="100%">
</p>
<p align="center"><sub><b>左：</b>仅靠反思的智能体在当前任务中修正了错误，却在下一个任务中重蹈覆辙。<b>右：</b>ManimAgent 将经验写入记忆库，在下一个场景中一次做对。</sub></p>

ManimAgent 将科学论文中的章节转化为 [Manim](https://www.manim.community/) 教学动画。它先规划分镜，再生成并渲染场景代码，修复运行错误，并通过视觉语言模型检查画面质量。双通道 EMB 保存成功示例和经过验证的修复经验，供后续任务检索使用。

Python 包和命令行工具的名称为 `paper2manim`。

## ✨ 亮点

- 🧠 **跨任务记忆，无需更新权重。** 以 VLM 评分作为奖励信号，配合完全由智能体自身任务流积累的双通道 EMB，在不微调、不依赖人工种子数据的前提下弥合任务之间的经验断层。
- ⚖️ **两个互补的记忆通道。** LLM 蒸馏器将每条反思轨迹转化为成功经验（**M+**，作为软性*参考示例*）或结构化的失败教训（**M−**，作为硬性*已知易错点*），只有满足各自的质量门槛才会写入。
- 🔬 **固定探针评估。** 在只读模式下评估冻结的 EMB 快照，与无记忆、同等预算 RAG 和打乱记忆等基线对比，并采用盲评人工打分。

## 📊 实验结果

<div align="center">

| 指标 | EMB@0 | Manim-code RAG | **EMB@200** |
| :--- | :---: | :---: | :---: |
| 人工 Pass@1 (%) ↑ | 62.0 | 83.3 | **84.9** |
| 反思轮数 ↓ | 12.2 | 11.4 | **6.5** |
| 人工质量评分（1–5）↑ | 3.26 | 3.65 | **3.88** |

</div>

<p align="center">
  <img src="assets/results.jpg" alt="EMB 规模从 0 增长到 200 时，人工 Pass@1、反思轮数和人工质量评分的变化" width="100%">
</p>
<p align="center"><sub>EMB@K 表示规模为 K 的冻结记忆快照。随着记忆增长，盲评人工 Pass@1 和质量评分持续上升，反思轮数持续下降，并在 K = 200 时超过静态 Manim-code RAG 基线。</sub></p>

## 🧩 工作流程

<p align="center">
  <img src="assets/pipeline.png" alt="ManimAgent 流程：成功记忆与失败记忆引导 Coder、Render 和 Reviewer 智能体" width="100%">
</p>

1. **规划分镜：** 将原文的科学论点、支撑证据和最终要点保留在场景计划中。
2. **生成与渲染：** 检索两条正向示例和三条负向经验，再生成并渲染 Manim 代码。
3. **反思与修订：** 修复渲染错误，并通过四张关键帧评估逻辑流程、布局与遮挡、科学准确性，评分范围为 0–100。
4. **整理记忆：** 保存评分不低于 85 的正向示例、从运行失败到成功的修复经验，以及评分提升至少 5 分的视觉修复经验。

EMB 使用冻结的 `all-MiniLM-L6-v2` 编码器，通过 Faiss 进行余弦相似度检索。正向和负向通道分别向代码生成智能体提供参考示例与已知易错模式。新建记忆库初始为空，随后在多次运行中持续积累经验。

## 🚀 快速开始

使用 Python 3.11 或 3.12。渲染需要 `ffmpeg`、Cairo/Pango，以及包含 `dvisvgm` 的 LaTeX 环境。系统依赖的安装方法见[入门指南](docs/getting-started.md)。

```bash
git clone https://github.com/jwj1342/Paper2Manim.git
cd Paper2Manim
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[emb]"
cp config.example.yaml config.yaml
cp .env.example .env
```

在 `.env` 中设置 `MANIMAGENT_API_KEY`。示例配置按照论文设置，为所有文本和视觉角色使用同一个 GPT-5.5 接口。若需使用其他模型，请修改 `config.yaml` 中的模型名称、接口地址、服务提供方和角色绑定；视觉评审模型必须支持图像输入。

> [!NOTE]
> 模型选择必须通过 `config.yaml` 配置。配置缺失或无效时会报错，程序不会自动加载示例配置。

使用本地章节文本生成动画：

```bash
paper2manim generate --input path/to/section.txt --scene-role METHOD
```

也可以从 arXiv 论文中选择一个章节：

```bash
paper2manim generate --arxiv 1706.03762 --section "Background"
```

默认启用 VLM 评审、EMB 检索和 LLM 记忆蒸馏。每个场景最多进行两次文本重试和两次视觉修订。最终视频采用各场景中评分最高的已渲染版本；评分相同时保留较早的版本。运行文件保存在 `runs/<run_id>/`，记忆默认存储在 `runs/_emb/`。

## 📚 使用说明与文档

| 文档 | 内容 |
| :--- | :--- |
| [入门指南](docs/getting-started.md) | 安装、模型配置、输入输出和可选配音 |
| [系统架构](docs/architecture.md) | 智能体、场景循环、记忆检索与整理 |
| [贡献指南](CONTRIBUTING.md) | 开发环境与测试方法 |

```bash
paper2manim generate --help
paper2manim emb --help
```

[`examples/text/`](examples/text/) 提供简短文本示例。[`scripts/`](scripts/) 包含环境配置、冒烟测试和 PDF 批量渲染脚本。配置 TTS 后，可通过 `--voiceover` 启用配音。

## 📝 引用

如果 ManimAgent 对你的研究有帮助，欢迎引用：

```bibtex
@inproceedings{jiang2026manimagent,
  title     = {ManimAgent: Self-Evolving Multimodal Agents for Visual Education},
  author    = {Wenjia Jiang and Zongyuan Cai and Yuanhang Shao and Chenru Wang
               and Boyan Han and Zhixue Song and Keyu Chen and Shengwei An
               and Xu Yang and Zhou Yang},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year      = {2026},
  publisher = {Association for Computational Linguistics},
  url       = {https://arxiv.org/abs/2606.30296}
}
```

<details>
<summary>arXiv 版本</summary>

```bibtex
@misc{jiang2026manimagentselfevolvingmultimodalagents,
  title         = {ManimAgent: Self-Evolving Multimodal Agents for Visual Education},
  author        = {Wenjia Jiang and Zongyuan Cai and Yuanhang Shao and Chenru Wang
                   and Boyan Han and Zhixue Song and Keyu Chen and Shengwei An
                   and Xu Yang and Zhou Yang},
  year          = {2026},
  eprint        = {2606.30296},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2606.30296}
}
```

</details>

## 🙏 致谢

ManimAgent 基于 [Manim Community Edition](https://www.manim.community/) 和 [LangGraph](https://www.langchain.com/langgraph) 构建，并使用 [Marker](https://github.com/datalab-to/marker) 提供可选的 PDF 解析功能。
