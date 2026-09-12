#!/bin/bash
#SBATCH --job-name=naming_probe
#SBATCH --partition=all_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/naming_%j.out
#SBATCH --error=logs/naming_%j.err
#SBATCH --account=cvcs2026

set -euo pipefail

MODEL="google/gemini-2.5-flash"
TAG="gemini25flash"
LIMIT=1000

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

export HF_HOME="/work/cvcs2026/recursive_retrievers/hf_cache/huggingface"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export PATH="$HOME/.local/bin:$PATH"
unset SSL_CERT_DIR

CODE_DIR="${CODE_DIR:-$PROJECT_DIR}"
cd "$PROJECT_DIR"
mkdir -p logs outputs/agentic

: "${LLM_API_KEY:?LLM_API_KEY is not set — export it and submit with 'sbatch --export=ALL'}"

uv run python "$CODE_DIR"/src/agent/experiments/naming_probe.py \
    --model-name "$MODEL" \
    --base-url "https://openrouter.ai/api/v1" \
    --output "outputs/agentic/naming_probe_${TAG}.jsonl" \
    --limit "$LIMIT" \
    --concurrency 8
