#!/bin/bash
#SBATCH --job-name=ablation
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_A40_45G|gpu_L40S_45G
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=logs/ablation_%j.out
#SBATCH --error=logs/ablation_%j.err
#SBATCH --account=cvcs2026

set -euo pipefail

MODEL="Qwen/Qwen3-VL-8B-Instruct"
TAG="qwen3vl8b"
GPU_UTIL=0.50
MAX_LEN=36864
SPLIT="${SPLIT:-val}"
if [ "$SPLIT" = "test" ]; then
    JSON="/work/cvcs2026/encyclopedic/encyclopedic_test_subset.json"
    TOP_K_VALUES="10 20 50"
    TOP_N_VALUES="10 20 30"
else
    JSON="$PROJECT_DIR/data/encyclopedic_val_split.json"
    TOP_K_VALUES="5 10 20 50 80"
    TOP_N_VALUES="5 10 20 30 40 60"
fi
CONCURRENCY=8

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
VENV="/homes/$USER/vllm_venv"
CODE_DIR="${CODE_DIR:-$PROJECT_DIR}"
OUT_DIR="outputs/ablation/$TAG/$SPLIT"

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

ensure_vllm_venv
serve_model "$MODEL" "$GPU_UTIL" "$MAX_LEN"

echo "################ grid on $SPLIT: k=[$TOP_K_VALUES] x n=[$TOP_N_VALUES]"
uv run python "$CODE_DIR"/src/ablation/run_ablation_cross.py \
    --model-name "$MODEL" --base-url "$BASE_URL" \
    --val-json "$JSON" \
    --output-dir "$OUT_DIR" \
    --top-k-values $TOP_K_VALUES \
    --rerank-top-n-values $TOP_N_VALUES \
    --concurrency "$CONCURRENCY" --debug-samples 1

stop_model

echo "################ BEM scoring"
for pred in "$OUT_DIR"/predictions_*.jsonl; do
    [ -e "$pred" ] || continue
    base=$(basename "$pred" .jsonl)
    result="$OUT_DIR/results_BEM_${base#predictions_}.json"
    [ -f "$result" ] && { echo "  [skip] $result"; continue; }
    (cd "$PROJECT_DIR/evqa_eval" && uv run python "$CODE_DIR/evqa_eval/score_evqa.py" \
        --predictions "../$pred" --output "../$result") || true
done

echo "################ summary"
uv run python "$CODE_DIR"/src/ablation/aggregate_ablation.py \
    --results-dir "$OUT_DIR" \
    --output "$OUT_DIR/ablation_summary_BEM.json"
cat "$OUT_DIR/ablation_summary_BEM.json" 2>/dev/null | head -30
