#!/bin/bash
# Paper2Manim — HPC 环境装载（登录节点 / 计算节点共用）
# 用法：source scripts/setup_env.sh                      # 默认 MVP 1，仅装 paper2manim 自己 + LLM 依赖
#       PAPER2MANIM_INSTALL_MANIM=1 source scripts/setup_env.sh   # 同时装 manim（仅登录节点能成）
#       PAPER2MANIM_MVP=2 source scripts/setup_env.sh   # 同时装 marker (MVP 2)
#       PAPER2MANIM_DEV=1 source scripts/setup_env.sh   # 同时装 pytest/ruff/mypy
#
# 装 manim 需要外网编译 skia-pathops（chromium.googlesource.com），
# 集群计算节点的 squid 代理通常拒绝该域 → 必须在登录节点装一次，venv 在所有节点共享。

set -euo pipefail

PROJ=/scratch/jwj/research/Paper2Manim
VENV=$PROJ/.venv
ON_LOGIN_NODE=true
[[ -n "${SLURM_JOB_ID:-}" ]] && ON_LOGIN_NODE=false

# 1) Lmod modules（动态确认过：Vulcan 有 python/3.11.5、ffmpeg/7.1.1，无 cairo/pango/texlive 模块）
module purge
module load StdEnv/2023 python/3.11.5 ffmpeg/7.1.1

# 2) 创建 venv（仅首次）
if [[ ! -d "$VENV" ]]; then
  echo "[setup] creating venv at $VENV"
  python -m venv "$VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"
unset PIP_PREFIX

# 3) pip / wheel
pip install --no-index --upgrade pip wheel >/dev/null 2>&1 \
  || pip install --upgrade pip wheel >/dev/null 2>&1

# 4) 装 LLM/agent 核心依赖
#    集群 wheelhouse 有 langgraph/langchain-core，但 langchain-openai 0.3.x 需走 PyPI。
#    PyPI（pypi.org / files.pythonhosted.org）通过 squid 都允许。
echo "[setup] installing core LLM/agent deps"
pip install \
    "langgraph>=1.0,<2.0" \
    "langchain-core>=0.3,<0.4" \
    "langchain-openai>=0.3,<0.4" \
    "openai>=1.50" \
    "pydantic>=2.7" \
    "pydantic-settings>=2.4" \
    "click>=8.1" \
    "python-dotenv>=1.0" \
    "rich>=13"

# 5) Manim（可选）—— 仅在登录节点编译 skia-pathops 才能成功
if [[ "${PAPER2MANIM_INSTALL_MANIM:-0}" == "1" ]]; then
  if ! $ON_LOGIN_NODE; then
    echo "[setup] WARN: PAPER2MANIM_INSTALL_MANIM=1 but you're on a compute node."
    echo "        skia-pathops compilation needs chromium.googlesource.com which the squid proxy"
    echo "        blocks on compute nodes. Run this once on a LOGIN node instead."
    echo "        Continuing without manim install."
  else
    echo "[setup] installing manim (will compile skia-pathops; needs ~5-10 min)"
    pip install "manim==0.20.1"
  fi
fi

# 6) MVP 2.0 增量：marker-pdf
if [[ "${PAPER2MANIM_MVP:-1}" == "2" ]]; then
  echo "[setup] installing marker-pdf for MVP 2.0"
  pip install marker-pdf
fi

# 7) Dev 工具
if [[ "${PAPER2MANIM_DEV:-0}" == "1" ]]; then
  echo "[setup] installing dev tools"
  pip install --no-index pytest pytest-mock ruff mypy 2>/dev/null \
    || pip install pytest pytest-mock ruff mypy
fi

# 8) 安装项目本身（editable，仅自身代码，不拖依赖）
pip install -e "$PROJ" --no-deps >/dev/null

# 9) LaTeX 检查
if ! command -v latex >/dev/null 2>&1; then
  if [[ -d "$HOME/.TinyTeX" ]]; then
    export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH"
  fi
fi
LATEX_PATH=$(command -v latex || echo NOT_INSTALLED)
MANIM_PATH=$(command -v manim || echo NOT_INSTALLED)

# 10) 状态报告
echo ""
echo "[setup] done."
echo "        host=$(hostname)  on_login_node=$ON_LOGIN_NODE  SLURM_JOB_ID=${SLURM_JOB_ID:-<login>}"
echo "        python=$(which python)  ($(python --version 2>&1))"
echo "        manim=$MANIM_PATH"
echo "        latex=$LATEX_PATH"
echo ""

if [[ "$MANIM_PATH" == "NOT_INSTALLED" ]]; then
  echo "[setup] HINT: manim not installed. To install (login node only):"
  echo "        PAPER2MANIM_INSTALL_MANIM=1 source scripts/setup_env.sh"
fi
if [[ "$LATEX_PATH" == "NOT_INSTALLED" ]]; then
  echo "[setup] HINT: latex not on PATH. To install user-level TinyTeX:"
  echo "        scripts/install_tinytex.sh"
fi
