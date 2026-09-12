#!/bin/bash
#SBATCH --job-name=recall_text
#SBATCH --partition=all_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=logs/recall_text_%j.out
#SBATCH --error=logs/recall_text_%j.err
#SBATCH --account=cvcs2026

set -euo pipefail
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CODE_DIR="${CODE_DIR:-$PROJECT_DIR}"
export PYTHONUNBUFFERED=1
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/work/cvcs2026/recursive_retrievers/$USER/.uv_cache}"
mkdir -p "$UV_CACHE_DIR"
unset SSL_CERT_DIR
cd "$PROJECT_DIR"; mkdir -p "${LOG_DIR:-logs}" outputs/retrieval
uv run python "$CODE_DIR"/src/retrieval/experiments/compute_recall_text.py "$@"
