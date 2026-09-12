#!/bin/bash
#SBATCH --job-name=fi_warm
#SBATCH --partition=all_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=192G
#SBATCH --cpus-per-task=8
#SBATCH --time=05:00:00
#SBATCH --output=logs/fi_warm_%j.out
#SBATCH --error=logs/fi_warm_%j.err
#SBATCH --account=cvcs2026

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
VENV="/homes/$USER/vllm_venv"

export PYTHONUNBUFFERED=1
unset SSL_CERT_DIR

CUDA_HOME=$(ls -d /homes/admin/spack/opt/spack/linux-x86_64_v2/cuda-1[3-9].*/ \
                  /homes/admin/spack/opt/spack/linux-x86_64_v2/cuda-12.9*/ 2>/dev/null |
            sort -V | tail -1 || true)
CUDA_HOME="${CUDA_HOME%/}"
[ -x "$CUDA_HOME/bin/nvcc" ] || { echo "no nvcc >= 12.9 found"; exit 1; }
export CUDA_HOME
export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH"
export FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-12.0}"
export FLASHINFER_NVCC_THREADS="${SLURM_CPUS_PER_TASK:-8}"

cd "$PROJECT_DIR"
mkdir -p "${LOG_DIR:-logs}"

echo "nvcc:  $("$CUDA_HOME/bin/nvcc" --version | grep -o 'release [0-9.]*')"
echo "archs: $FLASHINFER_CUDA_ARCH_LIST   threads: $FLASHINFER_NVCC_THREADS"

"$VENV/bin/python" - <<'PY'
import time
from flashinfer.compilation_context import CompilationContext
from flashinfer.fused_moe.core import gen_cutlass_fused_moe_sm120_module

print("target archs:", CompilationContext().TARGET_CUDA_ARCHS, flush=True)
t = time.time()
gen_cutlass_fused_moe_sm120_module(False).build_and_load()
print(f"fused_moe_120 built in {time.time() - t:.0f}s", flush=True)
PY

echo "cache:"
du -sh "$HOME/.cache/flashinfer" 2>/dev/null || true
