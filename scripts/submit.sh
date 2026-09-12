#!/bin/bash

set -euo pipefail

[ $# -ge 1 ] || { echo "usage: scripts/submit.sh <script.sh> [sbatch args...]"; exit 1; }
SCRIPT="$1"; shift
SCRIPT_LIB="scripts/lib/vllm.sh"   # sourced by the runners, and carries knobs too

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
cp pyproject.toml uv.lock "$SNAP"/ 2>/dev/null || true

{
    echo "run_id:  $RUN_ID"
    echo "variant: ${VARIANT:-<none>}"
    echo "commit:  $(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "branch:  $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
    echo "dirty:   $(git status --porcelain 2>/dev/null | wc -l) files"
    echo "script:  $SCRIPT"
    echo "vllm:    $("/homes/$USER/vllm_venv/bin/vllm" --version 2>/dev/null | head -1 || echo "not built yet")"
    echo "sbatch:  $*"
    for k in $(grep -ohE '\$\{[A-Z][A-Z0-9_]*[:-]' "$SCRIPT" "$SCRIPT_LIB" 2>/dev/null |
               tr -d '${:-' | sort -u); do
        case "$k" in SLURM*|PATH|HOME|USER|PWD|SHELL|TMPDIR) continue ;; esac
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
