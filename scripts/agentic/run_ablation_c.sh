#!/bin/bash
#SBATCH --job-name=abl_c
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_A40_45G|gpu_L40S_45G
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --output=logs/abl_c_%j.out
#SBATCH --error=logs/abl_c_%j.err
#SBATCH --account=cvcs2026
#
# One-at-a-time sweep of the parameters the best C uses without ever having
# tuned them. Everything else is held at the best configuration, so each line
# reads as "what this knob is worth", not as a search over a grid we cannot
# afford — nine settings on 1000 examples is already twelve hours.
#
#   scripts/submit.sh scripts/agentic/run_ablation_c.sh
#
# One vLLM server for the whole sweep: loading it costs 14 minutes, and paying
# that nine times would be most of a run. Configurations are appended to their
# own output file, so a job killed by the walltime resumes where it stopped.

set -euo pipefail

MODEL="Qwen/Qwen3-VL-8B-Instruct"
TAG="qwen3vl8b"
GPU_UTIL=0.50
MAX_LEN=32768
CONCURRENCY=4
LIMIT="${LIMIT:-1000}"

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
VENV="/homes/$USER/vllm_venv"
CODE_DIR="${CODE_DIR:-$PROJECT_DIR}"
# The resume guard keys on the results file, so a smoke run must not write where
# the real one will look: five-example results would make it skip every config.
OUT_DIR="outputs/agentic/$TAG/ablation$([ "$LIMIT" = 1000 ] || echo "_n$LIMIT")"

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
ensure_vllm_venv
serve_model "$MODEL" "$GPU_UTIL" "$MAX_LEN"

# name                    preview  text  names  lookup  gate   iters
CONFIGS="
reference                 8        5     4      3       -1     12
preview4                  4        5     4      3       -1     12
preview16                 16       5     4      3       -1     12
text3                     8        3     4      3       -1     12
text10                    8        10    4      3       -1     12
names2                    8        5     2      3       -1     12
names8                    8        5     8      3       -1     12
lookup5                   8        5     4      5       -1     12
gate-2                    8        5     4      3       -2     12
gate0                     8        5     4      3        0     12
iters6                    8        5     4      3       -1      6
"

echo "$CONFIGS" | while read -r NAME PREVIEW TEXTL NAMES LOOKUP GATE ITERS; do
    [ -z "${NAME:-}" ] && continue
    OUT="$OUT_DIR/predictions_$NAME.jsonl"
    RES="$OUT_DIR/results_$NAME.json"
    [ -f "$RES" ] && { echo "################ $NAME — already scored, skipping"; continue; }
    echo "################ $NAME  preview=$PREVIEW text=$TEXTL names=$NAMES lookup=$LOOKUP gate=$GATE iters=$ITERS"
    uv run python "$CODE_DIR"/src/agent/run_inference.py \
        --model-name "$MODEL" --base-url "$BASE_URL" --output "$OUT" \
        --final-pass --legacy-prompt \
        --top-k 20 --rerank-top-n 20 --tools-strategy bge \
        --preview "$PREVIEW" --text-limit "$TEXTL" --max-names "$NAMES" \
        --lookup-limit "$LOOKUP" --text-gate "$GATE" --max-iterations "$ITERS" \
        --concurrency "$CONCURRENCY" --debug-samples 0 --limit "$LIMIT" || continue
    (cd "$PROJECT_DIR/evqa_eval" && uv run python "$CODE_DIR/evqa_eval/score_evqa.py" \
        --predictions "../$OUT" --output "../$RES") || true
done

stop_model

echo "################ summary"
python3 -c "
import json, glob, os
rows=[]
for f in sorted(glob.glob('$OUT_DIR/results_*.json')):
    n=os.path.basename(f)[8:-5]
    try:
        r=json.load(open(f))
        m=f.replace('results_','predictions_').replace('.json','.metrics.json')
        M=json.load(open(m)) if os.path.exists(m) else {}
        rows.append((n, r['accuracy_overall'], M.get('avg_tool_calls',0), M.get('avg_seconds_per_example',0)))
    except Exception as e: print(f'  {n}: {e}')
print(f\"{'config':22s} {'BEM':>7} {'tool':>6} {'sec':>7}\")
for n,a,t,s in sorted(rows, key=lambda x:-x[1]):
    print(f'  {n:20s} {a:7.4f} {t:6.2f} {s:7.1f}')
"
cat "$CODE_DIR/RUN_INFO" 2>/dev/null || echo "code: live tree"
