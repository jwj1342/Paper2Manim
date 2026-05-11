#!/bin/bash
#SBATCH --account=aip-zhouyang
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --job-name=paper2manim
#SBATCH --output=runs/slurm-%A_%a.log
#SBATCH --array=0-0     # 改成 0-N 来批量跑多篇论文

# Paper2Manim — sbatch 批量论文渲染（MVP 2.0）
# 用法：
#   sbatch scripts/render_node.sh              # 单篇（修改下面 PAPERS 列表）
#   sbatch --array=0-2 scripts/render_node.sh  # 批量

set -euo pipefail

PAPERS=(
  examples/mvp2/transformer_section3_only.pdf
  # examples/mvp2/attention_is_all_you_need.pdf
)

PDF=${PAPERS[$SLURM_ARRAY_TASK_ID]}
cd /scratch/jwj/research/Paper2Manim

PAPER2MANIM_MVP=2 source scripts/setup_env.sh
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

echo "[render_node] rendering $PDF on $(hostname)"
paper2manim mvp2 --pdf "$PDF" --quality m
