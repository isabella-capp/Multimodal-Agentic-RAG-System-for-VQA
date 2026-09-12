#!/bin/bash
#SBATCH --job-name=hf_download
#SBATCH --partition=all_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=/homes/%u/cvcs2026/logs/hf_download_%j.out
#SBATCH --error=/homes/%u/cvcs2026/logs/hf_download_%j.err
#SBATCH --account=cvcs2026

set -euo pipefail
export HF_HOME="/work/cvcs2026/recursive_retrievers/hf_cache/huggingface"
export PATH="$HOME/.local/bin:$PATH"
unset SSL_CERT_DIR
cd "/homes/$USER/cvcs2026"; mkdir -p logs

MODEL="${1:-Qwen/Qwen2.5-VL-7B-Instruct}"
echo "downloading $MODEL into $HF_HOME ..."
uv run python -c "
from huggingface_hub import snapshot_download
print('done:', snapshot_download('$MODEL'))
"