#!/bin/bash
#SBATCH -J train
#SBATCH -p iris-hi
#SBATCH --account=iris
#SBATCH --nodelist=iris-hgx-1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH -o /iris/u/rbhowmik/logs/%x-%j.out
#SBATCH -e /iris/u/rbhowmik/logs/%x-%j.err

set -euo pipefail

cd /iris/u/rbhowmik/
source ./cache.sh
cd openpi/


echo "Running on host: $(hostname)"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# uv run scripts/train.py \
#   --config pi0_libero_low_mem_finetune \
#   --exp-name lora_finetune

# XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_robomimic_square_ph_image_lora --exp-name=robomimic_2 --overwrite --batch-size=32 --num-train-steps=20000
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_robomimic_square_ph_image_lora_30ep --exp-name=robomimic_30ep_lora_2 --overwrite --batch-size=32
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_robomimic_square_ph_image_lora_100ep_random_post --exp-name=robomimic_100ep_random_post_100ep --overwrite --batch-size=32
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_robomimic_square_ph_image_lora_combined_grasp --exp-name=robomimic_combined_grasp_24k --overwrite --batch-size=32
