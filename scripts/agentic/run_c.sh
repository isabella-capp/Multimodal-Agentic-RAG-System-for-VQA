#!/bin/bash
#SBATCH --job-name=agentic
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_A40_45G|gpu_L40S_45G
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=logs/agentic_%j.out
#SBATCH --error=logs/agentic_%j.err
#SBATCH --account=cvcs2026
#
# Setting C alone, for iterating on the agent. A and B do not touch the agent's
# code, so re-running them per variant would spend two thirds of a job
# reproducing numbers we already have — use scripts/run_abc.sh for the reference
# table, this one for every attempt after it.
#
#   VARIANT=hedge scripts/submit.sh scripts/agentic/run_c.sh
#
# Run the full 1000 by default. Loading vLLM, EVA-CLIP, FAISS and the reranker
# costs ~14 minutes whatever you do, while inference over all 1000 examples takes
# ~29: a 5-example SMOKE=1 spends the same 14 minutes to produce 9 seconds of
# signal. Only worth it to catch something obviously broken after a model change.

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
TAG="${TAG:-qwen3vl8b}"

# Every default below is our best measured C (0.470 +/- 0.003 on the 8B), so a
# bare run reproduces it and a variant is one variable away. They were 0 for a
# long time and we passed the real values by hand each time, which is how a C
# ended up measured against B with the wrong retrieval mode.
#
# To vary one, set it: PREVIEW=0, UNIFIED=0, TEXT_GATE= (empty turns the gate
# off), FINAL_PASS=0, WITH_READ=0.
GPU_UTIL="${GPU_UTIL:-0.50}"      # the retriever and reranker share this GPU
MAX_LEN="${MAX_LEN:-32768}"
TOP_K="${TOP_K:-20}"
TOP_N="${TOP_N:-20}"
BM25_TOP_M="${BM25_TOP_M:-50}"
CONCURRENCY=4
TEXT_GATE="${TEXT_GATE--1}"   # no colon: TEXT_GATE= (empty) turns the gate off
FINAL_PASS="${FINAL_PASS:-1}"
LEGACY="${LEGACY:-1}"
DIRECT="${DIRECT:-0}"
PREVIEW="${PREVIEW:-8}"
TEXT_LIMIT="${TEXT_LIMIT:-5}"
MAX_NAMES="${MAX_NAMES:-4}"
LOOKUP_LIMIT="${LOOKUP_LIMIT:-3}"
MAX_IT="${MAX_IT:-12}"
# One strategy per ranking operation, from fusion.STRATEGIES. They are three
# different operations and the best value differs for each, which is the whole
# reason they are separate:
#
#   TOOLS_STRATEGY    what the agent reads mid-loop.  rrf.
#   PREVIEW_STRATEGY  the passages shown beside the image candidates.  bge —
#                     and never measured against anything else, because it was
#                     hard-coded until now. The ablation swept how MANY passages
#                     (4/8/16) while the ranking that picks them was fixed.
#   FINAL_STRATEGY    the ranking the answer is generated from.  bge 0.4740
#                     against rrf 0.4610. Note B goes the other way on what is
#                     nominally the same operation — rrf 0.4760 twice against
#                     bm25_bge 0.4660 — which we cannot yet explain.
# minimal = the two tools; legacy = the four-tool interface, for the ablation.
TOOL_SET="${TOOL_SET:-minimal}"
TOOLS_STRATEGY="${TOOLS_STRATEGY:-rrf}"
PREVIEW_STRATEGY="${PREVIEW_STRATEGY:-bge}"
FINAL_STRATEGY="${FINAL_STRATEGY:-bge}"

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
VENV="/homes/$USER/vllm_venv"
CODE_DIR="${CODE_DIR:-$PROJECT_DIR}"
OUT_DIR="${RESUME_DIR:-outputs/agentic/$TAG/${RUN_ID:-manual}}"

FORCE=()
[ -n "$TEXT_GATE" ] && FORCE+=(--text-gate "$TEXT_GATE")
[ "$FINAL_PASS" = "1" ] && FORCE+=(--final-pass)
[ "$LEGACY" = "1" ] && FORCE+=(--legacy-prompt)
[ "$DIRECT" = "1" ] && FORCE+=(--direct-prompt)
[ "$PREVIEW" != "0" ] && FORCE+=(--preview "$PREVIEW")

if [ "${SMOKE:-0}" = "1" ]; then
    LIMIT=(--limit 5); DEBUG="${DEBUG:-5}"; OUT_DIR="$OUT_DIR/smoke"
else
    LIMIT=(); DEBUG="${DEBUG:-3}"   # raise it when a variant needs its tool sequences read
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

echo "reranker: $CROSS_ENCODER_MODEL   strategies: $TOOLS_STRATEGY/$PREVIEW_STRATEGY/$FINAL_STRATEGY   bm25-m: $BM25_TOP_M   gate: ${TEXT_GATE:-off}"
ensure_vllm_venv
serve_model "$MODEL" "$GPU_UTIL" "$MAX_LEN" "${NEED_GB:-25}"

# With VLLM_GPU set, the server has its own card; everything the Python
# side loads goes on the other one.
[ -n "${RETRIEVER_GPU:-}" ] && export CUDA_VISIBLE_DEVICES="$RETRIEVER_GPU"

echo "################ C — agentic  ($MODEL${VARIANT:+, variant $VARIANT}, strategies=$TOOLS_STRATEGY/$PREVIEW_STRATEGY/$FINAL_STRATEGY)"
MODE_SUFFIX=""
MODE_SUFFIX="_${TOOLS_STRATEGY}"
uv run python "$CODE_DIR"/src/agent/run_inference.py \
    --model-name "$MODEL" --base-url "$BASE_URL" \
    --output "$OUT_DIR/predictions_C${MODE_SUFFIX}.jsonl" \
    --top-k "$TOP_K" --rerank-top-n "$TOP_N" --bm25-top-m "$BM25_TOP_M" \
    --max-iterations "$MAX_IT" --text-limit "$TEXT_LIMIT" \
    --max-names "$MAX_NAMES" --lookup-limit "$LOOKUP_LIMIT" \
    --tool-set "$TOOL_SET" \
    --tools-strategy "$TOOLS_STRATEGY" \
    --preview-strategy "$PREVIEW_STRATEGY" \
    --final-strategy "$FINAL_STRATEGY" \
    --concurrency "$CONCURRENCY" --debug-samples "$DEBUG" \
    "${FORCE[@]}" "${LIMIT[@]}"

stop_model

echo "################ scoring"
(cd "$PROJECT_DIR/evqa_eval" && uv run python "$CODE_DIR/evqa_eval/score_evqa.py" \
    --predictions "../$OUT_DIR/predictions_C${MODE_SUFFIX}.jsonl" \
    --output "../$OUT_DIR/results_C${MODE_SUFFIX}.json") || true

echo "################ summary"
python3 -c "import json;print('  C${MODE_SUFFIX}:', json.load(open('$OUT_DIR/results_C${MODE_SUFFIX}.json'))['accuracy_overall'])" 2>/dev/null || echo "  C${MODE_SUFFIX}: n/a"
echo "  tool use:"
python3 -c "
import json; d=json.load(open('$OUT_DIR/predictions_C${MODE_SUFFIX}.metrics.json'))
print('   ', {k: d[k] for k in ('tool_called_pct','avg_tool_calls','avg_paragraphs_read','errors')})
print('    first tool:', d['first_tool_pct'])
for k,v in (d.get('tool_usage') or {}).items():
    print(f'    {k:18s} calls={v[\"calls\"]:4d} miss={v[\"miss_pct\"]}%')
" 2>/dev/null || true
cat "$CODE_DIR/RUN_INFO" 2>/dev/null || echo "code: live tree (not submitted via scripts/submit.sh)"
