#!/bin/bash
#SBATCH --job-name=baseline_b
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_A40_45G|gpu_L40S_45G
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/baseline_b_%j.out
#SBATCH --error=logs/baseline_b_%j.err
#SBATCH --account=cvcs2026

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
TAG="${TAG:-qwen3vl8b}"
GPU_UTIL="${GPU_UTIL:-0.50}"
MAX_LEN=32768
CONCURRENCY=8

ARMS="${ARMS:-B Bplus Btext Bgated}"
ORACLE="${ORACLE:-0}"   # upper bound: puts the gold article in the pool

TOP_K="${TOP_K:-20}"
TOP_N="${TOP_N:-20}"
BM25_TOP_M="${BM25_TOP_M:-50}"
NAMING_LIMIT="${NAMING_LIMIT:-1}"    # articles kept per guess
NAMING_GUESSES="${NAMING_GUESSES:-3}"
TEXT_LIMIT="${TEXT_LIMIT:-5}"
POOL_ARTICLES="${POOL_ARTICLES:-20}"
TEXT_GATE="${TEXT_GATE:--1}"
RETRIEVAL_STRATEGY="${RETRIEVAL_STRATEGY:-rrf}"

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
VENV="/homes/$USER/vllm_venv"
CODE_DIR="${CODE_DIR:-$PROJECT_DIR}"
OUT_DIR="outputs/baselines/$TAG/${RUN_ID:-manual}"

ORACLE_FLAG=(); [ "$ORACLE" = "1" ] && ORACLE_FLAG=(--oracle)

if [ "${SMOKE:-0}" = "1" ]; then
    LIMIT=(--limit 5); DEBUG="${DEBUG:-5}"; OUT_DIR="$OUT_DIR/smoke"
else
    LIMIT=(); DEBUG="${DEBUG:-3}"
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

echo "arms: $ARMS   reranker: $CROSS_ENCODER_MODEL"
ensure_vllm_venv
serve_model "$MODEL" "$GPU_UTIL" "$MAX_LEN" "${NEED_GB:-25}"

[ -n "${RETRIEVER_GPU:-}" ] && export CUDA_VISIBLE_DEVICES="$RETRIEVER_GPU"

for ARM in $ARMS; do
    CHANNELS=(); RETRIEVAL=(--use-retrieval)
    case "$ARM" in
        A) RETRIEVAL=() ;;   # no retrieval at all: what the weights alone know
        Bplus) CHANNELS=(--use-naming --naming-limit "$NAMING_LIMIT" --naming-guesses "$NAMING_GUESSES") ;;
        Btext) CHANNELS=(--use-naming --naming-limit "$NAMING_LIMIT" --naming-guesses "$NAMING_GUESSES"
                         --use-text --text-limit "$TEXT_LIMIT") ;;
        Bgated) CHANNELS=(--use-naming --naming-limit "$NAMING_LIMIT" --naming-guesses "$NAMING_GUESSES"
                          --use-text --text-limit "$TEXT_LIMIT"
                          --text-gate "$TEXT_GATE") ;;
    esac
    echo "################ $ARM  (top-k=$TOP_K top-n=$TOP_N bm25-m=$BM25_TOP_M)$([ "$LEGACY" = 1 ] && echo ", legacy prompt")"
    uv run python "$CODE_DIR"/src/vlm/run_inference.py \
        --model-name "$MODEL" --base-url "$BASE_URL" \
        --output "$OUT_DIR/predictions_$ARM.jsonl" \
        "${RETRIEVAL[@]}" --top-k "$TOP_K" --rerank-top-n "$TOP_N" --bm25-top-m "$BM25_TOP_M" \
        --retrieval-strategy "$RETRIEVAL_STRATEGY" \
        --concurrency "$CONCURRENCY" --debug-samples "$DEBUG" \
        "${CHANNELS[@]}" "${ORACLE_FLAG[@]}" "${LIMIT[@]}"
done

stop_model

echo "################ scoring"
for ARM in $ARMS; do
    (cd "$PROJECT_DIR/evqa_eval" && uv run python "$CODE_DIR/evqa_eval/score_evqa.py" \
        --predictions "../$OUT_DIR/predictions_$ARM.jsonl" \
        --output "../$OUT_DIR/results_$ARM.json") || true
    uv run python "$CODE_DIR"/src/retrieval/experiments/analyse_pool.py \
        --predictions "$OUT_DIR/predictions_$ARM.jsonl" \
        --output "$OUT_DIR/pool_$ARM.json" || true
done

echo "################ summary"
python3 -c "
import json
for arm in '$ARMS'.split():
    try:
        r = json.load(open('$OUT_DIR/results_%s.json' % arm))
        p = json.load(open('$OUT_DIR/pool_%s.json' % arm))['percent']
        print(f\"  {arm:6s} BEM={r['accuracy_overall']:.4f}   right article: \"
              f\"image={p['image']}%  name={p['name']}%  union={p['union']}%\")
    except Exception as e:
        print(f'  {arm}: n/a ({e})')
" 2>/dev/null || true
cat "$CODE_DIR/RUN_INFO" 2>/dev/null || echo "code: live tree"
