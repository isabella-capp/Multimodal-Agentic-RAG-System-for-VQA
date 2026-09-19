#!/bin/bash
#SBATCH --job-name=abc
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_A40_45G|gpu_L40S_45G
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=05:00:00
#SBATCH --output=logs/abc_%j.out
#SBATCH --error=logs/abc_%j.err
#SBATCH --account=cvcs2026

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
TAG="${TAG:-qwen3vl8b}"
GPU_UTIL="${GPU_UTIL:-0.50}"
MAX_LEN="${MAX_LEN:-32768}"
TOP_K="${TOP_K:-20}"
TOP_N="${TOP_N:-20}"
CONCURRENCY="${CONCURRENCY:-8}"
C_CONCURRENCY="${C_CONCURRENCY:-$CONCURRENCY}"
FINAL_PASS="${FINAL_PASS:-1}"
FP=(); [ "$FINAL_PASS" = "1" ] && FP=(--final-pass)

RETRIEVAL_STRATEGY="${RETRIEVAL_STRATEGY:-rrf}"
TOOLS_STRATEGY="${TOOLS_STRATEGY:-rrf}"
PREVIEW_STRATEGY="${PREVIEW_STRATEGY:-rrf}"
FINAL_STRATEGY="${FINAL_STRATEGY:-bge}"

NAMING_GUESSES="${NAMING_GUESSES:-3}"
NAMING_LIMIT="${NAMING_LIMIT:-1}"
TEXT_LIMIT="${TEXT_LIMIT:-5}"
TEXT_GATE="${TEXT_GATE--1}"
GATE=(); [ -n "${TEXT_GATE:-}" ] && GATE=(--text-gate "$TEXT_GATE")
PREVIEW="${PREVIEW:-8}"
SETTINGS="${SETTINGS:-A B C}"

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV="/homes/$USER/vllm_venv"
CODE_DIR="${CODE_DIR:-$PROJECT_DIR}"
OUT_DIR="outputs/abc/$TAG/${RUN_ID:-manual}"

if [ "${SMOKE:-0}" = "1" ]; then
    LIMIT=(--limit 5); DEBUG=5; OUT_DIR="$OUT_DIR/smoke"
else
    LIMIT=(); DEBUG=3
fi

export HF_HOME="/work/cvcs2026/recursive_retrievers/hf_cache/huggingface"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export CROSS_ENCODER_MODEL="${CROSS_ENCODER_MODEL:-BAAI/bge-reranker-v2-m3}"
export VLLM_USE_FLASHINFER_SAMPLER=0
export PATH="$HOME/.local/bin:$PATH"
export TFHUB_CACHE_DIR="/work/cvcs2026/recursive_retrievers/tfhub_cache"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/work/cvcs2026/recursive_retrievers/$USER/.uv_cache}"
mkdir -p "$UV_CACHE_DIR"
unset SSL_CERT_DIR

cd "$PROJECT_DIR"
mkdir -p "${LOG_DIR:-logs}" "$OUT_DIR"
source "$CODE_DIR/scripts/lib/vllm.sh"

echo "model: $MODEL   reranker: $CROSS_ENCODER_MODEL   settings: $SETTINGS"
echo "gate: ${TEXT_GATE:-off}   concurrency: A/B=$CONCURRENCY C=$C_CONCURRENCY   B: $RETRIEVAL_STRATEGY   C: $TOOLS_STRATEGY/$PREVIEW_STRATEGY/$FINAL_STRATEGY   tp: ${TP:-1}"
ensure_vllm_venv
serve_model "$MODEL" "$GPU_UTIL" "$MAX_LEN" "${NEED_GB:-25}"
[ -n "${RETRIEVER_GPU:-}" ] && export CUDA_VISIBLE_DEVICES="$RETRIEVER_GPU"

for S in $SETTINGS; do
    [ -f "$OUT_DIR/results_$S.json" ] && { echo "################ $S — already scored"; continue; }
    echo "################ $S"
    case "$S" in
    A)
        uv run python "$CODE_DIR"/src/vlm/run_inference.py \
            --model-name "$MODEL" --base-url "$BASE_URL" \
            --output "$OUT_DIR/predictions_A.jsonl"\
            --concurrency "$CONCURRENCY" --debug-samples "$DEBUG" "${LIMIT[@]}"
        ;;
    B)
        uv run python "$CODE_DIR"/src/vlm/run_inference.py \
            --model-name "$MODEL" --base-url "$BASE_URL" \
            --output "$OUT_DIR/predictions_B.jsonl"\
            --use-retrieval --top-k "$TOP_K" --rerank-top-n "$TOP_N" \
            --retrieval-strategy "$RETRIEVAL_STRATEGY" \
            --use-naming --naming-guesses "$NAMING_GUESSES" --naming-limit "$NAMING_LIMIT" \
            --use-text --text-limit "$TEXT_LIMIT" "${GATE[@]}" \
            --concurrency "$CONCURRENCY" --debug-samples "$DEBUG" "${LIMIT[@]}"
        ;;
    C)
        uv run python "$CODE_DIR"/src/agent/run_inference.py \
            --model-name "$MODEL" --base-url "$BASE_URL" \
            --output "$OUT_DIR/predictions_C.jsonl" \
            "${FP[@]}" \
            --preview "$PREVIEW" "${GATE[@]}" \
            --top-k "$TOP_K" --rerank-top-n "$TOP_N" \
            --tools-strategy "$TOOLS_STRATEGY" \
            --preview-strategy "$PREVIEW_STRATEGY" \
            --final-strategy "$FINAL_STRATEGY" \
            --concurrency "$C_CONCURRENCY" --debug-samples "$DEBUG" "${LIMIT[@]}"
        ;;
    esac
    (cd "$PROJECT_DIR/evqa_eval" && uv run python "$CODE_DIR/evqa_eval/score_evqa.py" \
        --predictions "../$OUT_DIR/predictions_$S.jsonl" \
        --output "../$OUT_DIR/results_$S.json") || true
done

stop_model

echo "################ summary  ($MODEL)"
uv run python "$CODE_DIR"/src/ablation/summarise.py "$OUT_DIR"
cat "$CODE_DIR/RUN_INFO" 2>/dev/null || echo "code: live tree"
