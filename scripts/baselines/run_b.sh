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
#
# The retrieval baseline, in its two arms, on one vLLM server.
#
#   A      no retrieval — the reference for what the weights alone know
#   B      image retrieval only — the reference pipeline
#   B+     same, plus one naming call whose resolved articles join the pool
#   Btext  same as B+, plus the question searched against the paragraph index
#   Bgated same channels as Btext, but the text search runs only where the first
#          pass looks empty-handed
#
# The three arms are the three ways into the KB. Only the last does not go
# through the model, and it is the one that moves the ceiling: the right article
# is in the pool 40.6% of the time with the image alone, 46.4% adding the name,
# 58.7% adding the text search.
#
# Btext runs the text search on every example, which is why Bgated exists: it
# took the pool from 133 paragraphs to 197 while the cross-encoder still returns
# twenty, so the channel gained 55 points where it alone found the article and
# lost 9 on the 465 where the article was already there — a wash.
#
# Merging the channels by rank into a pool of fixed size was tried first and is
# worse than either: capped at twenty articles, every text or name article
# displaces an image one and image coverage fell from 41.1% to 26.8%. The pool
# is not the thing to shrink. Bgated runs the second channel only where the
# first pass scores badly, which keeps the coverage and skips most of the noise.
#
# B+ differs from B by a single flag: same code path, same prompt, same
# reranker, same top-n. It exists so the agentic comparison means something —
# the agent can enter the KB by name and B cannot, so without B+ a win for C
# would only show that the name channel works.
#
#   scripts/submit.sh scripts/baselines/run_b.sh                       # every arm
#   ARMS=Bgated scripts/submit.sh scripts/baselines/run_b.sh           # the best one
#   LEGACY=0 scripts/submit.sh scripts/baselines/run_b.sh              # answer-format block
#   NAMING_GUESSES=1 scripts/submit.sh scripts/baselines/run_b.sh      # one name guess
#   CROSS_ENCODER_MODEL=BAAI/bge-reranker-base scripts/submit.sh …     # the smaller reranker
#
# Both arms run in one job on purpose: two runs of the same configuration a week
# apart scored 0.395 and 0.392, so a gap under ~0.3 points is not a result
# unless the arms were measured side by side.

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
TAG="${TAG:-qwen3vl8b}"
GPU_UTIL="${GPU_UTIL:-0.50}"
MAX_LEN=32768
CONCURRENCY=8

ARMS="${ARMS:-B Bplus Btext Bgated}"

TOP_K="${TOP_K:-20}"
TOP_N="${TOP_N:-20}"
BM25_TOP_M="${BM25_TOP_M:-50}"
NAMING_LIMIT="${NAMING_LIMIT:-1}"    # articles kept per guess
NAMING_GUESSES="${NAMING_GUESSES:-3}"
TEXT_LIMIT="${TEXT_LIMIT:-5}"
POOL_ARTICLES="${POOL_ARTICLES:-20}"
TEXT_GATE="${TEXT_GATE:--1}"
# rrf scored 0.4760 twice against 0.4660 +/- 0.0026 for three runs of bm25_bge,
# so it is the default. The per-example paired test does not separate them
# (+0.011, CI [-0.009, +0.032]), which is worth knowing before quoting the point
# difference; what carries it is that two independent rrf runs landed on the same
# value. The gate is unaffected either way — both send the whole pool to the
# cross-encoder, and measured, it opens 400 times against 409.
RETRIEVAL_STRATEGY="${RETRIEVAL_STRATEGY:-rrf}"
LEGACY="${LEGACY:-1}"
DIRECT="${DIRECT:-0}"

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
VENV="/homes/$USER/vllm_venv"
CODE_DIR="${CODE_DIR:-$PROJECT_DIR}"
OUT_DIR="outputs/baselines/$TAG/${RUN_ID:-manual}"

PROMPT=()
[ "$LEGACY" = "1" ] && PROMPT=(--legacy-prompt)
[ "$DIRECT" = "1" ] && PROMPT=(--direct-prompt)

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

echo "arms: $ARMS   reranker: $CROSS_ENCODER_MODEL   legacy-prompt: $LEGACY"
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
        "${CHANNELS[@]}" "${PROMPT[@]}" "${LIMIT[@]}"
done

stop_model

echo "################ scoring"
for ARM in $ARMS; do
    (cd "$PROJECT_DIR/evqa_eval" && uv run python "$CODE_DIR/evqa_eval/score_evqa.py" \
        --predictions "../$OUT_DIR/predictions_$ARM.jsonl" \
        --output "../$OUT_DIR/results_$ARM.json") || true
    # where the right article came from: the image ranking, the name, or neither
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
