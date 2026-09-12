#!/bin/bash
#SBATCH --job-name=sweep
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_A40_45G|gpu_L40S_45G
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=logs/sweep_%j.out
#SBATCH --error=logs/sweep_%j.err
#SBATCH --account=cvcs2026

set -euo pipefail

LOCAL_MODELS=(
    "Qwen/Qwen2.5-VL-3B-Instruct|qwen25vl3b|0.35|32768"
    "Qwen/Qwen2.5-VL-7B-Instruct|qwen25vl7b|0.45|32768"
    "Qwen/Qwen3-VL-8B-Instruct|qwen3vl8b|0.50|32768"
)
REMOTE_MODEL="qwen/qwen3-vl-235b-a22b-instruct"
REMOTE_TAG="qwen3vl235b"
LIMIT=1000
CONCURRENCY=4

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
VENV="/homes/$USER/vllm_venv"
CODE_DIR="${CODE_DIR:-$PROJECT_DIR}"
OUT_DIR="outputs/agentic/sweep"

export HF_HOME="/work/cvcs2026/recursive_retrievers/hf_cache/huggingface"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export PATH="$HOME/.local/bin:$PATH"
export TFHUB_CACHE_DIR="/work/cvcs2026/recursive_retrievers/tfhub_cache"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/work/cvcs2026/recursive_retrievers/$USER/.uv_cache}"
mkdir -p "$UV_CACHE_DIR"
unset SSL_CERT_DIR

cd "$PROJECT_DIR"
mkdir -p "${LOG_DIR:-logs}" "$OUT_DIR"
source "$CODE_DIR/scripts/lib/vllm.sh"

evaluate() {
    local model="$1" base_url="$2" tag="$3"
    echo "################ $tag  ($model)"

    uv run python "$CODE_DIR"/src/agent/experiments/naming_probe.py \
        --model-name "$model" --base-url "$base_url" \
        --output "$OUT_DIR/naming_${tag}.jsonl" \
        --limit "$LIMIT" --concurrency 8 || true

    local pred="$OUT_DIR/predictions_${tag}.jsonl"
    uv run python "$CODE_DIR"/src/agent/run_inference.py \
        --model-name "$model" --base-url "$base_url" \
        --output "$pred" --limit "$LIMIT" \
        --concurrency "$CONCURRENCY" --debug-samples 3 || true

    (cd "$PROJECT_DIR/evqa_eval" && uv run python "$CODE_DIR/evqa_eval/score_evqa.py" \
        --predictions "../$pred" --output "../$OUT_DIR/results_${tag}.json") || true
}

ensure_vllm_venv

for entry in "${LOCAL_MODELS[@]}"; do
    IFS='|' read -r model tag gpu_util max_len <<< "$entry"
    if [ -f "$OUT_DIR/results_${tag}.json" ]; then
        echo "################ $tag — already scored, skipping"
        continue
    fi
    if serve_model "$model" "$gpu_util" "$max_len"; then
        evaluate "$model" "$BASE_URL" "$tag"
    else
        echo "vLLM failed to start for $model — skipping."
    fi
    stop_model
done

if [ -n "${LLM_API_KEY:-}" ]; then
    if [ -f "$OUT_DIR/results_${REMOTE_TAG}.json" ]; then
        echo "################ $REMOTE_TAG — already scored, skipping"
    else
        evaluate "$REMOTE_MODEL" "https://openrouter.ai/api/v1" "$REMOTE_TAG"
    fi
else
    echo "LLM_API_KEY not set — skipping $REMOTE_TAG."
fi

echo "################ summary"
for f in "$OUT_DIR"/results_*.json; do
    [ -e "$f" ] && echo "--- $f" && cat "$f"
done
