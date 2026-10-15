#!/bin/bash
# Paper2Manim — 使用已安装的 CLI 顺序渲染本地 PDF。
# 用法：bash scripts/render_papers.sh path/to/paper.pdf [path/to/another.pdf ...]

set -euo pipefail

if [[ $# -eq 0 ]]; then
  echo "Usage: bash scripts/render_papers.sh PDF [PDF ...]" >&2
  exit 2
fi

for pdf in "$@"; do
  echo "[render_papers] rendering $pdf"
  paper2manim generate --pdf "$pdf" --quality m
done
