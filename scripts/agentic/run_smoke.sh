#!/bin/bash
#SBATCH --job-name=smoke
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_A40_45G|gpu_L40S_45G
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/smoke_%j.out
#SBATCH --error=logs/smoke_%j.err
#SBATCH --account=cvcs2026

set -euo pipefail

MODELS=(
    "qwen/qwen3-vl-8b-instruct|qwen3vl8b"
    "qwen/qwen3-vl-235b-a22b-instruct|qwen3vl235b"
    "google/gemini-2.5-flash|gemini25flash"
)
LIMIT=5

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
OUT_DIR="outputs/agentic/smoke"

export HF_HOME="/work/cvcs2026/recursive_retrievers/hf_cache/huggingface"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export PATH="$HOME/.local/bin:$PATH"
unset SSL_CERT_DIR

CODE_DIR="${CODE_DIR:-$PROJECT_DIR}"
cd "$PROJECT_DIR"
mkdir -p "${LOG_DIR:-logs}" "$OUT_DIR"

: "${LLM_API_KEY:?LLM_API_KEY is not set — export it and submit with 'sbatch --export=ALL'}"

for entry in "${MODELS[@]}"; do
    model="${entry%%|*}"
    tag="${entry##*|}"
    pred="$OUT_DIR/${tag}.jsonl"
    echo "################ $tag  ($model)"
    rm -f "$pred" "$OUT_DIR/${tag}.metrics.json"  # the eval resumes, so start clean

    uv run python "$CODE_DIR"/src/agent/run_inference.py \
        --model-name "$model" \
        --base-url "https://openrouter.ai/api/v1" \
        --output "$pred" \
        --limit "$LIMIT" --concurrency 1 --debug-samples "$LIMIT" || true
done

echo "################ verdict"
for entry in "${MODELS[@]}"; do
    tag="${entry##*|}"
    echo "--- $tag"
    python3 - "$OUT_DIR/${tag}.metrics.json" <<'PY'
import json, sys
try:
    m = json.load(open(sys.argv[1]))
except Exception as e:
    print(f"  no metrics ({e})")
    raise SystemExit
print(f"  tool_called_pct={m['tool_called_pct']}  avg_tool_calls={m['avg_tool_calls']}  errors={m['errors']}")
for err, n in (m.get("error_types") or {}).items():
    print(f"  [{n}x] {err[:170]}")
print("  => OK" if m["errors"] == 0 and m["tool_called_pct"] > 0 else "  => UNUSABLE")
PY
done
