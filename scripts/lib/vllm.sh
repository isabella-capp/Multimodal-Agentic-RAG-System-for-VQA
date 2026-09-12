
PORT=$((10000 + ${SLURM_JOB_ID:-0} % 20000))
BASE_URL="http://localhost:$PORT/v1"

VLLM_PID=""
STAGED_DIR=""

vllm_cleanup() {
    [ -n "$VLLM_PID" ] && kill "$VLLM_PID" 2>/dev/null || true
    [ -n "$STAGED_DIR" ] && rm -rf "$STAGED_DIR" || true
}
trap vllm_cleanup EXIT

_stage_weights() {
    local model="$1" need_gb="$2" snap avail
    snap=$(ls -d "$HF_HOME"/hub/"models--${model//\//--}"/snapshots/*/ 2>/dev/null | head -1)
    [ -n "$snap" ] || return 1
    avail=$(df -BG --output=avail "${TMPDIR:-/tmp}" 2>/dev/null | tail -1 | tr -dc '0-9')
    [ "${avail:-0}" -ge "$need_gb" ] || return 1

    STAGED_DIR="${TMPDIR:-/tmp}/$(basename "$snap")_${SLURM_JOB_ID:-0}"
    mkdir -p "$STAGED_DIR"
    echo "staging weights to node-local disk ..."
    cp -rL "$snap". "$STAGED_DIR"/ && return 0
    rm -rf "$STAGED_DIR"; STAGED_DIR=""; return 1
}

_pick_cuda_home() {
    local best
    best=$(ls -d /homes/admin/spack/opt/spack/linux-x86_64_v2/cuda-1[3-9].*/ \
                 /homes/admin/spack/opt/spack/linux-x86_64_v2/cuda-12.9*/ 2>/dev/null |
           sort -V | tail -1 || true)
    [ -n "$best" ] && [ -x "${best}bin/nvcc" ] || return 0
    export CUDA_HOME="${best%/}"
    export PATH="$CUDA_HOME/bin:$PATH"
    echo "CUDA_HOME -> $CUDA_HOME ($("$CUDA_HOME/bin/nvcc" --version | grep -o 'release [0-9.]*'))"
}

serve_model() {
    local model="$1" gpu_util="${2:-0.50}" max_len="${3:-32768}" need_gb="${4:-25}"
    local serve="$model" ticks=240 extra=()

    _pick_cuda_home
    export PATH="$VENV/bin:$PATH"
    export FLASHINFER_NVCC_THREADS="${SLURM_CPUS_PER_TASK:-4}"

    if _stage_weights "$model" "$need_gb"; then
        serve="$STAGED_DIR"
    else
        echo "serving from \$HF_HOME (wider wait window)"
        ticks=1200
    fi

    case "$model" in
        *Qwen2.5-VL*) extra=(--chat-template "${CODE_DIR:-$PROJECT_DIR}/scripts/agentic/qwen2.5-vl-tool-chat-template.jinja") ;;
    esac

    [ "${TP:-1}" -gt 1 ] && extra+=(--tensor-parallel-size "$TP") || true

    CUDA_VISIBLE_DEVICES="${VLLM_GPU:-${CUDA_VISIBLE_DEVICES:-}}" \
    "$VENV/bin/vllm" serve "$serve" --port "$PORT" \
        --served-model-name "$model" \
        --gpu-memory-utilization "$gpu_util" --max-model-len "$max_len" \
        --enable-auto-tool-choice --tool-call-parser hermes "${extra[@]}" \
        --safetensors-load-strategy=prefetch \
        > "${LOG_DIR:-logs}/vllm_$(basename "$model")_${SLURM_JOB_ID:-0}.log" 2>&1 &
    VLLM_PID=$!

    echo "waiting for vLLM on port $PORT (up to $((ticks / 6)) min) ..."
    for _ in $(seq 1 "$ticks"); do
        curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1 && break
        sleep 10
    done
    curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1 || { echo "vLLM did not start"; return 1; }

    if ! curl -sf "http://localhost:$PORT/v1/models" | grep -q "\"$model\""; then
        echo "port $PORT answers but does not serve $model — refusing to run."
        curl -sf "http://localhost:$PORT/v1/models" || true
        return 1
    fi
    echo "vLLM ready on port $PORT serving $model"
}

stop_model() {
    vllm_cleanup
    VLLM_PID=""; STAGED_DIR=""
    sleep 20  # let the GPU drain before the next server
}

ensure_vllm_venv() {
    [ -x "$VENV/bin/vllm" ] && return 0
    echo "creating vLLM venv at $VENV ..."
    uv venv "$VENV" --python 3.12 --clear
    uv pip install --python "$VENV/bin/python" "vllm==0.25.1"
}
