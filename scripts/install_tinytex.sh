#!/bin/bash
# 用户级 TinyTeX 安装 + Manim 推荐 LaTeX 包
# 仅在集群无 texlive 模块时跑一次（Vulcan 当前情形）

set -euo pipefail

if command -v latex >/dev/null 2>&1; then
  echo "[tinytex] latex already available at $(which latex); nothing to do."
  exit 0
fi

if [[ ! -d "$HOME/.TinyTeX" ]]; then
  echo "[tinytex] downloading installer..."
  curl -sL https://yihui.org/tinytex/install-bin-unix.sh | sh
fi

export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH"

echo "[tinytex] installing Manim-recommended LaTeX packages..."
tlmgr install \
  standalone preview doublestroke ms setspace rsfs relsize ragged2e \
  fundus-calligra microtype wasysym physics babel-english \
  cm-super xcolor amsmath amssymb dvisvgm latex-tools \
  geometry adjustbox tools

echo "[tinytex] done. PATH already augmented for this shell."
echo "[tinytex] Add this to your ~/.bashrc / shell init for permanence:"
echo "          export PATH=\"\$HOME/.TinyTeX/bin/x86_64-linux:\$PATH\""
