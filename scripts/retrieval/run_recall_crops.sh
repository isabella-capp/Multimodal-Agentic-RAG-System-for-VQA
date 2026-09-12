#!/bin/bash
#SBATCH --job-name=recall_crops
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_A40_45G|gpu_L40S_45G
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/homes/%u/cvcs2026/logs/recall_crops_%j.out
#SBATCH --error=/homes/%u/cvcs2026/logs/recall_crops_%j.err
#SBATCH --account=cvcs2026

set -euo pipefail
PROJECT_DIR="/homes/$USER/cvcs2026"
export HF_HOME="/work/cvcs2026/recursive_retrievers/hf_cache/huggingface"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export PATH="$HOME/.local/bin:$PATH"
unset SSL_CERT_DIR
cd "$PROJECT_DIR"; mkdir -p logs outputs/retrieval

uv run python src/retrieval/experiments/compute_recall_crops.py \
    --output outputs/retrieval/recall_crops.jsonl
