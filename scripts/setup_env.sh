#!/bin/bash
# Paper2Manim — 创建并激活本地虚拟环境，安装项目依赖。
# 用法：source scripts/setup_env.sh
#       PAPER2MANIM_PDF=1 source scripts/setup_env.sh   # 同时安装 Marker
#       PAPER2MANIM_DEV=1 source scripts/setup_env.sh   # 同时安装开发工具
#       PAPER2MANIM_PYTHON=python3.11 source scripts/setup_env.sh

set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
venv_dir="$project_dir/.venv"

if [[ ! -d "$venv_dir" ]]; then
  echo "[setup] creating venv at $venv_dir"
  "${PAPER2MANIM_PYTHON:-python3}" -m venv "$venv_dir"
fi

# shellcheck disable=SC1091
source "$venv_dir/bin/activate"

python -m pip install --upgrade pip wheel

extras=(emb)
if [[ "${PAPER2MANIM_PDF:-0}" == "1" ]]; then
  extras+=(pdf)
fi
if [[ "${PAPER2MANIM_DEV:-0}" == "1" ]]; then
  extras+=(dev)
fi

install_target="$project_dir"
if [[ ${#extras[@]} -gt 0 ]]; then
  extra_names="$(IFS=,; echo "${extras[*]}")"
  install_target+="[$extra_names]"
fi

echo "[setup] installing paper2manim and dependencies"
python -m pip install -e "$install_target"

if ! command -v latex >/dev/null 2>&1; then
  if [[ -d "$HOME/.TinyTeX" ]]; then
    export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH"
  fi
fi
latex_path=$(command -v latex || echo NOT_INSTALLED)
manim_path=$(command -v manim || echo NOT_INSTALLED)

echo "[setup] done."
echo "        python=$(command -v python) ($(python --version 2>&1))"
echo "        manim=$manim_path"
echo "        latex=$latex_path"

if [[ "$latex_path" == "NOT_INSTALLED" ]]; then
  echo "[setup] HINT: latex not on PATH. To install user-level TinyTeX:"
  echo "        bash scripts/install_tinytex.sh"
fi
