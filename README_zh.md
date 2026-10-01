# ManimAgent

**面向视觉教育的自进化多模态智能体** · EMNLP 2026 主会

[项目主页](https://manimagent.github.io/) · [论文](https://arxiv.org/abs/2606.30296) · [English](README.md)

ManimAgent 将科学论文中的章节转化为 Manim 教学动画。它先规划分镜，再生成并渲染场景代码，修复运行错误，并通过视觉语言模型检查画面质量。双通道情景记忆库（Episodic Memory Bank，EMB）保存成功示例和经过验证的修复经验，供后续任务检索使用。

Python 包和命令行工具的名称为 `paper2manim`。

## 快速开始

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

模型选择必须通过 `config.yaml` 配置。配置缺失或无效时会报错，程序不会自动加载示例配置。

使用本地章节文本生成动画：

```bash
paper2manim generate --input path/to/section.txt --scene-role METHOD
```

也可以从 arXiv 论文中选择一个章节：

```bash
paper2manim generate --arxiv 1706.03762 --section "Background"
```

默认启用 VLM 评审、EMB 检索和 LLM 记忆蒸馏。每个场景最多进行两次文本重试和两次视觉修订。最终视频采用各场景中评分最高的已渲染版本；评分相同时保留较早的版本。运行文件保存在 `runs/<run_id>/`，记忆默认存储在 `runs/_emb/`。

## 工作流程

1. **规划分镜：** 将原文的科学论点、支撑证据和最终要点保留在场景计划中。
2. **生成与渲染：** 检索两条正向示例和三条负向经验，再生成并渲染 Manim 代码。
3. **反思与修订：** 修复渲染错误，并通过四张关键帧评估逻辑流程、布局与遮挡、科学准确性，评分范围为 0–100。
4. **整理记忆：** 保存评分不低于 85 的正向示例、从运行失败到成功的修复经验，以及评分提升至少 5 分的视觉修复经验。

EMB 使用冻结的 `all-MiniLM-L6-v2` 编码器，通过 Faiss 进行余弦相似度检索。正向和负向通道分别向代码生成智能体提供参考示例与已知易错模式。新建记忆库初始为空，随后在多次运行中持续积累经验。

## 使用说明与文档

- [入门指南](docs/getting-started.md)：安装、模型配置、输入输出和可选配音。
- [系统架构](docs/architecture.md)：智能体、场景循环、记忆检索与整理。
- [贡献指南](CONTRIBUTING.md)：开发环境与测试方法。

```bash
paper2manim generate --help
paper2manim emb --help
```

[`examples/text/`](examples/text/) 提供简短文本示例。[`scripts/`](scripts/) 包含环境配置、冒烟测试和 PDF 批量渲染脚本。配置 TTS 后，可通过 `--voiceover` 启用配音。

## 引用

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

## 致谢

ManimAgent 基于 [Manim Community Edition](https://www.manim.community/) 和 [LangGraph](https://www.langchain.com/langgraph) 构建，并使用 [Marker](https://github.com/datalab-to/marker) 提供可选的 PDF 解析功能。
