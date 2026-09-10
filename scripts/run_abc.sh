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
#
# 96G, not the 32G the single-setting runners use: that is enough for an 8B at
# TP=1 and not for a 30B at TP=2, where two resident vLLM workers plus the FAISS
# index plus EVA-CLIP-8B loading peaked past 35G and the job was OOM-killed the
# moment B started. The L40S nodes have 515G, so the headroom costs nothing.

set -euo pipefail

# A, B and C at their best configurations, against ONE vLLM server in one job:
# same weights, same endpoint, same examples, so they differ only in method.
# Two runs of the same configuration land within ~0.3 points, and the gaps we
# care about are that size, so anything measured across separate jobs is noise.
#
#   scripts/submit.sh scripts/run_abc.sh
#
# On a bigger model, split the weights over two cards with TP and leave a third
# for EVA-CLIP and the reranker. The L40S nodes carry 4 GPUs each and there are
# nine of them, so this both fits and queues:
#
#   MODEL=Qwen/Qwen3-VL-30B-A3B-Instruct TAG=qwen3vl30b GPU_UTIL=0.85 \
#     TP=2 VLLM_GPU=0,1 RETRIEVER_GPU=2 NEED_GB=70 \
#     scripts/submit.sh scripts/run_abc.sh \
#       --constraint=gpu_L40S_45G --gres=gpu:3 --time=16:00:00
#
# Not the 96 GB RTXPro6000B nodes, even though one card would hold the model:
# they are Blackwell (sm_120) and the project's torch is pinned to cu124, which
# stops at sm_90 — vLLM serves there once scripts/setup/warm_flashinfer.sh has
# run, but EVA-CLIP dies with "no kernel image is available for execution on the
# device" the moment B or C starts. L40S is Ada, where the stack is proven.

MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
TAG="${TAG:-qwen3vl8b}"
GPU_UTIL="${GPU_UTIL:-0.50}"
MAX_LEN="${MAX_LEN:-32768}"
TOP_K="${TOP_K:-20}"
TOP_N="${TOP_N:-20}"
CONCURRENCY="${CONCURRENCY:-8}"

# How the tools rank what the agent reads mid-loop. The preview and the final
# pass have their own strategies, defaulted in fusion.Ranking — they are not the
# same operation and the best value differs for each.
TOOLS_STRATEGY="${TOOLS_STRATEGY:-rrf}"

# B: image + three name guesses + the text channel behind the cross-encoder gate
NAMING_GUESSES="${NAMING_GUESSES:-3}"
NAMING_LIMIT="${NAMING_LIMIT:-1}"
TEXT_LIMIT="${TEXT_LIMIT:-5}"
TEXT_GATE="${TEXT_GATE:--1}"
# C: one search tool, a passage preview, the same gate, answer from the pipeline
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
            --output "$OUT_DIR/predictions_A.jsonl" --legacy-prompt \
            --concurrency "$CONCURRENCY" --debug-samples "$DEBUG" "${LIMIT[@]}"
        ;;
    B)
        uv run python "$CODE_DIR"/src/vlm/run_inference.py \
            --model-name "$MODEL" --base-url "$BASE_URL" \
            --output "$OUT_DIR/predictions_B.jsonl" --legacy-prompt \
            --use-retrieval --top-k "$TOP_K" --rerank-top-n "$TOP_N" \
            --use-naming --naming-guesses "$NAMING_GUESSES" --naming-limit "$NAMING_LIMIT" \
            --use-text --text-limit "$TEXT_LIMIT" --text-gate "$TEXT_GATE" \
            --concurrency "$CONCURRENCY" --debug-samples "$DEBUG" "${LIMIT[@]}"
        ;;
    C)
        uv run python "$CODE_DIR"/src/agent/run_inference.py \
            --model-name "$MODEL" --base-url "$BASE_URL" \
            --output "$OUT_DIR/predictions_C.jsonl" \
            --final-pass --legacy-prompt \
            --preview "$PREVIEW" --text-gate "$TEXT_GATE" \
            --top-k "$TOP_K" --rerank-top-n "$TOP_N" \
            --tools-strategy "$TOOLS_STRATEGY" \
            --concurrency 4 --debug-samples "$DEBUG" "${LIMIT[@]}"
        ;;
    esac
    (cd "$PROJECT_DIR/evqa_eval" && uv run python "$CODE_DIR/evqa_eval/score_evqa.py" \
        --predictions "../$OUT_DIR/predictions_$S.jsonl" \
        --output "../$OUT_DIR/results_$S.json") || true
done

stop_model

echo "################ summary  ($MODEL)"
python3 -c "
import json, os
for s in '$SETTINGS'.split():
    r = '$OUT_DIR/results_%s.json' % s
    m = '$OUT_DIR/predictions_%s.meta.json' % s
    try:
        a = json.load(open(r))
        t = json.load(open(m)).get('avg_seconds_per_example') if os.path.exists(m) else None
        print(f\"  {s}: {a['accuracy_overall']:.4f}   {a['accuracy_by_type']}   {t or '?'} s/example\")
    except Exception as e:
        print(f'  {s}: n/a ({e})')
"
cat "$CODE_DIR/RUN_INFO" 2>/dev/null || echo "code: live tree"
