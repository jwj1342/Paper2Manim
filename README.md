# ManimAgent

**Self-Evolving Multimodal Agents for Visual Education** · EMNLP 2026 Main Conference

[Project page](https://manimagent.github.io/) · [Paper](https://arxiv.org/abs/2606.30296) · [中文](README_zh.md)

ManimAgent turns scientific paper sections into educational Manim animations. It plans a storyboard, generates and renders scene code, repairs execution errors, and uses a vision-language model to review visual quality. A dual-channel Episodic Memory Bank (EMB) stores successful examples and validated repair lessons for later tasks.

The Python package and command-line interface are named `paper2manim`.

## Quick start

Use Python 3.11 or 3.12. Rendering requires `ffmpeg`, Cairo/Pango, and LaTeX with `dvisvgm`; see [Getting started](docs/getting-started.md) for system dependencies.

```bash
git clone https://github.com/jwj1342/Paper2Manim.git
cd Paper2Manim
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[emb]"
cp config.example.yaml config.yaml
cp .env.example .env
```

Set `MANIMAGENT_API_KEY` in `.env`. The example configuration uses GPT-5.5 for all text and vision roles, following the paper. To use another model, update its name, endpoint, provider, and role bindings in `config.yaml`; the vision reviewer requires a vision-capable model.

Generate an animation from local section text:

```bash
paper2manim generate --input path/to/section.txt --scene-role METHOD
```

Or select a section from an arXiv paper:

```bash
paper2manim generate --arxiv 1706.03762 --section "Background"
```

VLM review, EMB retrieval, and LLM memory distillation are enabled by default. Each scene allows up to two text retries and two visual revisions. The final video uses the highest-scoring rendered version of each scene; ties favor the earlier version. Runs are saved under `runs/<run_id>/`, and memories are stored in `runs/_emb/` by default.

## How it works

1. **Storyboard:** preserve the source claim, evidence, and final takeaway in a scene plan.
2. **Generate and render:** retrieve two positive examples and three negative lessons, then produce and render Manim code.
3. **Reflect:** repair render errors and review four keyframes for logic flow, layout/occlusion, and scientific accuracy on a 0–100 scale.
4. **Consolidate:** store positive examples scoring at least 85, crash-to-success repairs, and visual repairs improving the score by at least 5 points.

The EMB uses a frozen `all-MiniLM-L6-v2` encoder and cosine retrieval through Faiss. Its positive and negative channels provide reference examples and known pitfalls to the coder. New banks start empty and accumulate experience across runs.

## Usage and documentation

- [Getting started](docs/getting-started.md): installation, model configuration, inputs, outputs, and optional narration.
- [Architecture](docs/architecture.md): agents, scene loops, memory retrieval, and consolidation.
- [Contributing](CONTRIBUTING.md): development setup and tests.

```bash
paper2manim generate --help
paper2manim emb --help
```

[`examples/text/`](examples/text/) contains small text inputs. [`scripts/`](scripts/) contains environment setup, a smoke test, and a PDF batch renderer. Optional voiceover is enabled with `--voiceover` after configuring TTS.

## Citation

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

## Acknowledgements

ManimAgent builds on [Manim Community Edition](https://www.manim.community/), [LangGraph](https://www.langchain.com/langgraph), and [Marker](https://github.com/datalab-to/marker) for optional PDF parsing.
