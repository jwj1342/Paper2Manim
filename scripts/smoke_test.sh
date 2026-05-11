#!/bin/bash
# 端到端最小 demo（MVP 1.0），用于安装后快速冒烟
# 必须在 salloc 计算节点跑（manim 渲染重计算）

set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "[smoke] WARNING: not on a compute node. Render will be blocked unless --allow-render-on-login."
fi

source scripts/setup_env.sh

echo "[smoke] step 1/3: import check"
python -c "import manim, langgraph, langchain_openai, click; print('imports ok')"

echo "[smoke] step 2/3: MiMo connectivity"
python -c "
from dotenv import load_dotenv; load_dotenv()
from paper2manim.llm import get_llm
print(get_llm('flash').invoke('Reply with exactly: OK').content)
"

echo "[smoke] step 3/3: end-to-end MVP 1.0 (pythagorean.txt)"
paper2manim mvp1 --input examples/mvp1/pythagorean.txt --quality l

echo "[smoke] done. Check runs/ for the generated mp4."
