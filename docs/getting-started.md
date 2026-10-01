# Getting started

ManimAgent is the research system; `paper2manim` is its Python package and CLI.

## Install

Use Python 3.11 or 3.12. On Ubuntu or Debian, install rendering dependencies:

```bash
sudo apt update
sudo apt install ffmpeg libcairo2-dev libpango1.0-dev pkg-config \
  texlive texlive-latex-extra texlive-fonts-extra texlive-science \
  tipa cm-super dvisvgm
```

On Windows, use WSL2 and the Linux commands above. On macOS, install the corresponding Cairo/Pango dependencies, `ffmpeg`, and MacTeX or TinyTeX. The [Manim installation guide](https://docs.manim.community/en/stable/installation.html) describes platform dependencies.

```bash
git clone https://github.com/jwj1342/Paper2Manim.git
cd Paper2Manim
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[emb]"
```

| Optional extra | Purpose |
|---|---|
| `pip install -e ".[pdf]"` | Parse local PDFs or arXiv papers without LaTeX source through Marker |
| `pip install -e ".[voiceover]"` | Use Edge TTS for narration |
| `pip install -e ".[dev]"` | Run tests and linting |

The EMB encoder and Marker download their model weights when first used.

## Configure models

```bash
cp config.example.yaml config.yaml
cp .env.example .env
```

Set `MANIMAGENT_API_KEY` in `.env`. The template binds every text and vision role to one GPT-5.5 endpoint. You can change `model`, `base_url`, `provider`, and `model_roles` for your service. The model bound to `vision_checker` must support image input and declare `supports_vision: true`.

API keys written as `$ENV_VAR` are resolved from the environment or `.env`. Keep `.env` and `config.yaml` out of Git. Set `omit_temperature: true` if your endpoint rejects the temperature parameter.

## Generate an animation

For a local paper section, pass its text directly and select the pedagogical role:

```bash
paper2manim generate --input path/to/section.txt --scene-role METHOD
```

`--input` also accepts literal text. Supported roles are `BACKGROUND`, `METHOD`, `EXPERIMENT`, and `CONCLUSION`.

```bash
paper2manim generate --input examples/text/pythagorean.txt
paper2manim generate --arxiv 1706.03762 --section "Background"
paper2manim generate --pdf path/to/paper.pdf
```

Provide exactly one of `--input`, `--arxiv`, or `--pdf`. `--section` selects a section from arXiv LaTeX source and requires `--arxiv`. If arXiv has no source and a section was requested, supply the section text with `--input`. Without `--section`, arXiv can fall back to PDF parsing. Local PDFs use Marker; install the `pdf` extra first. For a specific section of a local paper, use `--input` with its extracted text.

VLM review, EMB retrieval, and LLM memory distillation are enabled by default. New memory banks start empty. The default bank lives in `runs/_emb/`; use `--emb-store-path path/to/bank` to select another location. `--emb-readonly` retrieves existing records without writing new experience or changing hit counters.

To generate code without rendering:

```bash
paper2manim generate --input examples/text/pythagorean.txt --no-render --no-emb
```

`--no-render` saves generated scene code and skips rendering, video assembly, and memory consolidation. `--no-vlm` disables visual review; `--no-emb` disables memory. `--max-retries` and `--max-visual-revisions` each default to 2. `--quality l|m|h` selects Manim output quality.

## Optional narration and concurrency

Configure the commented `tts:` block in `config.yaml`, install the chosen backend, and enable voiceover:

```bash
pip install -e ".[voiceover]"
paper2manim generate --input path/to/section.txt --voiceover
```

Voiceover cannot be combined with `--no-render`. `--tts-voice`, `--tts-speed`, and `--voiceover-language` customize speech. `--voiceover-best-effort` allows a run to retain video when narration assembly fails.

`--scene-parallelism N` controls concurrent scene generation. `--render-concurrency N` limits Manim subprocesses, and `--llm-rps N` limits model request rate.

```bash
paper2manim generate --help
paper2manim emb --help
```

## Outputs

The CLI prints `RUN_ID=<run_id>` and saves files under `runs/<run_id>/`. Set `PAPER2MANIM_RUNS_DIR` to change the output root.

| Path | Contents |
|---|---|
| `input.txt`, `input.pdf.path`, or `input.arxiv.txt` | Input text, PDF path, or arXiv request |
| `parsed.tex` or `parsed.md` | Parsed source text |
| `summary.json` | Summary for arXiv/PDF input |
| `storyboard.json` | Scene plan |
| `attempts/` | Code and render sidecars for text attempts and visual revisions |
| `vlm_frames/` | Keyframe montages |
| `final/output.mp4` | Assembled silent video |
| `final/output_narrated.mp4`, `final/narration.json`, `final/audio/` | Narrated video, metadata, and audio when enabled |
| `trace.jsonl` | Execution trace |

For failures, inspect the printed error, `attempts/*.render.json`, and `trace.jsonl`. See [Architecture](architecture.md) for the generation and memory flow.
