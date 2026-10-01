#!/bin/bash
# 端到端最小 ManimAgent 示例，用于安装后快速冒烟

set -euo pipefail
cd "$(dirname "$0")/.."

source scripts/setup_env.sh

echo "[smoke] step 1/3: import check"
python -c "import manim, langgraph, langchain_openai, click; print('imports ok')"

echo "[smoke] step 2/3: model connectivity"
python -c "
from dotenv import load_dotenv; load_dotenv()
from paper2manim.llm import get_llm
print(get_llm('scene_coder').invoke('Reply with exactly: OK').content)
"

echo "[smoke] step 3/3: end-to-end ManimAgent (pythagorean.txt)"
paper2manim generate --input examples/text/pythagorean.txt --quality l

echo "[smoke] done. Check runs/ for the generated mp4."
