#!/bin/bash
#SBATCH --job-name=recall_topk
#SBATCH --partition=all_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_RTX_A5000_24G|gpu_RTX6000_24G
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/homes/%u/cvcs2026/logs/recall_%j.out
#SBATCH --error=/homes/%u/cvcs2026/logs/recall_%j.err
#SBATCH --account=cvcs2026

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

export HF_HOME="/work/cvcs2026/recursive_retrievers/hf_cache/huggingface"
export HF_HUB_OFFLINE=1
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
unset SSL_CERT_DIR

CODE_DIR="${CODE_DIR:-$PROJECT_DIR}"
cd "$PROJECT_DIR"
mkdir -p logs outputs/retrieval

uv run python "$CODE_DIR"/src/retrieval/experiments/compute_recall.py \
    --output outputs/retrieval/retrieval_topk50.jsonl \
    --top-k 50
