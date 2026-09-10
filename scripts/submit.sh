#!/bin/bash

set -euo pipefail

[ $# -ge 1 ] || { echo "usage: scripts/submit.sh <script.sh> [sbatch args...]"; exit 1; }
SCRIPT="$1"; shift

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
[ -f "$SCRIPT" ] || { echo "no such script: $SCRIPT"; exit 1; }

RUN_ID="$(date +%Y%m%d-%H%M%S)${VARIANT:+-$VARIANT}"
SNAP="$REPO/runs/$RUN_ID"
LOG_DIR="logs/$RUN_ID"          # everything this run writes ends up here
mkdir -p "$SNAP" "$REPO/$LOG_DIR"
cp -r src scripts "$SNAP"/

mkdir -p "$SNAP/evqa_eval"
cp evqa_eval/*.py evqa_eval/pyproject.toml evqa_eval/uv.lock "$SNAP/evqa_eval"/
# The venv itself is reused from the repo — snapshotting 1.8 GB per run makes no
# sense. The lock files record what it was supposed to contain, which is enough
# to rebuild it or to notice it changed.
cp pyproject.toml uv.lock "$SNAP"/ 2>/dev/null || true

# What the snapshot corresponds to, for the times it matters later.
{
    echo "run_id:  $RUN_ID"
    echo "variant: ${VARIANT:-<none>}"
    echo "commit:  $(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "branch:  $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
    echo "dirty:   $(git status --porcelain 2>/dev/null | wc -l) files"
    echo "script:  $SCRIPT"
    echo "vllm:    $("/homes/$USER/vllm_venv/bin/vllm" --version 2>/dev/null | head -1 || echo "not built yet")"
    echo "sbatch:  $*"
    # The knobs, as they were at launch. Without these the snapshot says which
    # code ran but not which configuration, and the two are equally necessary:
    # two 1-GPU C jobs launched a minute apart differed only by TEXT_GATE, which
    # was recoverable solely from the predictions meta.json once they finished.
    # Empty means "set but empty" (TEXT_GATE= turns the gate off), which is not
    # the same as absent, so print it only when the name is in the environment.
    for k in MODEL TAG SETTINGS ARMS GPU_UTIL MAX_LEN TP VLLM_GPU RETRIEVER_GPU \
             NEED_GB CROSS_ENCODER_MODEL RETRIEVAL_STRATEGY \
             TOOLS_STRATEGY PREVIEW_STRATEGY FINAL_STRATEGY \
             TOP_K TOP_N BM25_TOP_M \
             TEXT_GATE TEXT_LIMIT PREVIEW MAX_NAMES LOOKUP_LIMIT MAX_IT \
             NAMING_GUESSES NAMING_LIMIT FINAL_PASS LEGACY DIRECT \
             CONCURRENCY LIMIT SMOKE RESUME_DIR; do
        [ -z "${!k+x}" ] || echo "env:     $k=${!k}"
    done
} > "$SNAP/RUN_INFO"
git diff HEAD > "$SNAP/uncommitted.diff" 2>/dev/null || true


export CODE_DIR="$SNAP" RUN_ID="$RUN_ID" LOG_DIR="$LOG_DIR"
JOB=$(sbatch --parsable \
      --output="$LOG_DIR/%x_%j.out" --error="$LOG_DIR/%x_%j.err" \
      "$@" "$SCRIPT")
echo "submitted $JOB"
echo "  code: runs/$RUN_ID"
echo "  logs: $LOG_DIR/"
echo "  variant: ${VARIANT:-<none>}"
echo "$JOB" > "$SNAP/JOB_ID"
