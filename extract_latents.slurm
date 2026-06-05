#!/bin/bash
#SBATCH -J extract_latents
#SBATCH -p iris-hi
#SBATCH --account=iris
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH -o /iris/u/rbhowmik/logs/%x-%j.out
#SBATCH -e /iris/u/rbhowmik/logs/%x-%j.err

set -euo pipefail

cd /iris/u/rbhowmik/
source ./cache.sh
cd openpi/

XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 uv run python examples/robomimic/extract_latents.py
