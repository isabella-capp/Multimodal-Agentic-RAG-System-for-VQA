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
SWEEP="${SWEEP:-$(date +%Y%m%d)}"
OUT_DIR="outputs/agentic/$TAG/ablation-$SWEEP$([ "$LIMIT" = 1000 ] || echo "_n$LIMIT")"

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

BASE="--top-k 20 --rerank-top-n 20 --bm25-top-m 50
      --preview 8 --text-gate -1 --max-iterations 12
      --text-limit 5 --max-names 4 --lookup-limit 3
      --tool-set two --tools-strategy rrf
      --preview-strategy bge --final-strategy bge"   # the pre-sweep default

CONFIGS="
reference     |--final-pass
preview_rrf   |--final-pass --preview-strategy rrf
preview_bm25  |--final-pass --preview-strategy bm25_bge
final_rrf     |--final-pass --final-strategy rrf
final_bm25    |--final-pass --final-strategy bm25_bge
tools_bge     |--final-pass --tools-strategy bge
nofinalpass   |
fourtools     |--final-pass --tool-set four
all_rrf       |--final-pass --preview-strategy rrf --final-strategy rrf
"

echo "$CONFIGS" | while IFS='|' read -r NAME EXTRA; do
    NAME=$(echo "$NAME" | tr -d ' ')
    [ -z "${NAME:-}" ] && continue
    OUT="$OUT_DIR/predictions_$NAME.jsonl"
    RES="$OUT_DIR/results_$NAME.json"
    [ -f "$RES" ] && { echo "################ $NAME — already scored, skipping"; continue; }
    echo "################ $NAME   $EXTRA"
    uv run python "$CODE_DIR"/src/agent/run_inference.py \
        --model-name "$MODEL" --base-url "$BASE_URL" --output "$OUT" \
        $BASE $EXTRA \
        --concurrency "$CONCURRENCY" --debug-samples 0 --limit "$LIMIT" || continue
    (cd "$PROJECT_DIR/evqa_eval" && uv run python "$CODE_DIR/evqa_eval/score_evqa.py" \
        --predictions "../$OUT" --output "../$RES") || true
done

stop_model

echo "################ summary"
uv run python "$CODE_DIR"/src/ablation/summarise.py "$OUT_DIR"
cat "$CODE_DIR/RUN_INFO" 2>/dev/null || echo "code: live tree"
