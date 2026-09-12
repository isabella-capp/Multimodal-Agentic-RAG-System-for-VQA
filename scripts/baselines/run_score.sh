#!/bin/bash
#SBATCH --job-name=score
#SBATCH --partition=all_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/score_%j.out
#SBATCH --error=logs/score_%j.err
#SBATCH --account=cvcs2026

set -euo pipefail
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CODE_DIR="${CODE_DIR:-$PROJECT_DIR}"
export HF_HOME="/work/cvcs2026/recursive_retrievers/hf_cache/huggingface"
export TFHUB_CACHE_DIR="/work/cvcs2026/recursive_retrievers/tfhub_cache"
export PYTHONUNBUFFERED=1
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/work/cvcs2026/recursive_retrievers/$USER/.uv_cache}"
mkdir -p "$UV_CACHE_DIR"
unset SSL_CERT_DIR
cd "$PROJECT_DIR"; mkdir -p "${LOG_DIR:-logs}"

: "${PREDS:?set PREDS to the prediction files to score}"
for P in $PREDS; do
    R="${P%.jsonl}"; R="${R/predictions_/results_}.json"
    echo "################ $P"
    (cd "$PROJECT_DIR/evqa_eval" && uv run python "$CODE_DIR/evqa_eval/score_evqa.py" \
        --predictions "../$P" --output "../$R") || true
    uv run python "$CODE_DIR"/src/retrieval/experiments/analyse_pool.py \
        --predictions "$P" --output "${P%.jsonl}.pool.json" || true
done
echo "################ summary"
python3 -c "
import json, sys
for p in '''$PREDS'''.split():
    r = p[:-6].replace('predictions_', 'results_') + '.json'
    try:
        d = json.load(open(r)); print(f\"  {p.split('/')[-2]:26s} {p.split('/')[-1]:22s} BEM={d['accuracy_overall']:.4f}\")
    except Exception as e: print(f'  {p}: n/a ({e})')
"
