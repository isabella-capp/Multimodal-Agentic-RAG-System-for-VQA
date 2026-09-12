#!/bin/bash
#SBATCH --job-name=naming_crops
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_A40_45G|gpu_L40S_45G
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=logs/naming_crops_%j.out
#SBATCH --error=logs/naming_crops_%j.err
#SBATCH --account=cvcs2026

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
TAG="${TAG:-qwen3vl8b}"
GPU_UTIL=0.85          # nothing else on this GPU: no retriever, no reranker
MAX_LEN=32768
VARIANTS="${VARIANTS:-full}"
GUESSES="${GUESSES:-1}"      # space-separated list: sweeps them
STYLES="${STYLES:-diverse}"  # space-separated list of prompt styles
UPSCALE="${UPSCALE:-1}"
LIMIT="${LIMIT:-1000}"
CONCURRENCY=8

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
VENV="/homes/$USER/vllm_venv"
CODE_DIR="${CODE_DIR:-$PROJECT_DIR}"
OUT_DIR="outputs/agentic/$TAG/${RUN_ID:-manual}"

FLAGS=()
[ "$UPSCALE" = "0" ] && FLAGS+=(--no-upscale)
[ -n "${BOXES:-}" ] && FLAGS+=(--boxes "$BOXES")

export HF_HOME="/work/cvcs2026/recursive_retrievers/hf_cache/huggingface"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/work/cvcs2026/recursive_retrievers/$USER/.uv_cache}"
mkdir -p "$UV_CACHE_DIR"
unset SSL_CERT_DIR

cd "$PROJECT_DIR"
mkdir -p "${LOG_DIR:-logs}" "$OUT_DIR"
source "$CODE_DIR/scripts/lib/vllm.sh"

echo "variants: $VARIANTS   guesses: $GUESSES   upscale: $UPSCALE"
ensure_vllm_venv
serve_model "$MODEL" "$GPU_UTIL" "$MAX_LEN"

for G in $GUESSES; do
  for S in $STYLES; do
    echo "################ guesses=$G style=$S"
    uv run python "$CODE_DIR"/src/agent/experiments/naming_probe.py \
        --model-name "$MODEL" --base-url "$BASE_URL" \
        --output "$OUT_DIR/naming_g${G}_${S}.jsonl" \
        --variants "$VARIANTS" --guesses "$G" --style "$S" \
        --limit "$LIMIT" --concurrency "$CONCURRENCY" \
        "${FLAGS[@]}"
  done
done

stop_model
cat "$CODE_DIR/RUN_INFO" 2>/dev/null || echo "code: live tree"
