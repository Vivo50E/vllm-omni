#!/usr/bin/env bash
# KV-Offload serving benchmark for Qwen2.5-Omni-7B (dense 7B thinker).
#
# Uses the DEFAULT Qwen2.5-Omni pipeline + --stage-overrides to inject
# per-arm knobs on the thinker (stage 0), instead of a hand-written YAML.
#
# Arms (all run with GPU prefix-caching OFF so LMCache is isolated):
#   baseline     : no cache            -> full recompute every request
#   lmcache_only : LMCache KV offload  -> CPU->GPU restore on hit
#   prefix_only  : GPU prefix caching  -> free in-GPU hit (reference ceiling)
#
# Usage: bash bench_serving_qwen25.sh [arm ...]   (default: all three)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

VENV=/localhome/local-yiqix/repos/venv
VLLM="$VENV/bin/vllm"
MODEL="${MODEL:-Qwen/Qwen2.5-Omni-7B}"
PORT="${PORT:-46355}"
OUT_DIR="$SCRIPT_DIR/results_qwen25"
mkdir -p "$OUT_DIR"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
# flashinfer JIT needs nvcc 13.0; /usr/bin/nvcc is 11.5 and fails to compile.
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
export PATH="$CUDA_HOME/bin:$VENV/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONHASHSEED=0

# Workload: small set of long prefixes reused many times (high reuse so the
# cache is actually exercised). reuse factor = NUM_PROMPTS / PR_NUM_PREFIXES.
NUM_PROMPTS="${NUM_PROMPTS:-256}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-8}"
PR_PREFIX_LEN="${PR_PREFIX_LEN:-8192}"
PR_SUFFIX_LEN="${PR_SUFFIX_LEN:-256}"
PR_NUM_PREFIXES="${PR_NUM_PREFIXES:-32}"
OUTPUT_LEN="${OUTPUT_LEN:-64}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# AUDIO=1 -> full text+audio pipeline with audio metrics (TTFP/RTF); else text-only.
if [ "${AUDIO:-0}" = "1" ]; then
    EXTRA_BODY='{"modalities": ["text", "audio"]}'
    PCT_METRICS="ttft,tpot,itl,e2el,audio_ttfp,audio_rtf"
else
    EXTRA_BODY='{"modalities": ["text"]}'
    PCT_METRICS="ttft,tpot,itl,e2el"
fi

# Capacity knobs. GPU_MEM_UTIL shrinks the in-GPU prefix cache; LMCACHE_CPU_GB
# sizes the CPU tier. To make LMCache beat prefix_only, size the reused working
# set above the GPU cache but below the CPU cache.
GPU_MEM_UTIL="${GPU_MEM_UTIL:-}"          # empty -> deploy default (0.8)
LMCACHE_CPU_GB="${LMCACHE_CPU_GB:-40}"
LMCACHE_JSON='{"kv_store_config":{"lmcache_config":{"chunk_size":256,"local_cpu":true,"max_local_cpu_size":'"$LMCACHE_CPU_GB"',"skip_mm_storage":true}}}'

# Optional "gpu_memory_utilization":<v> fragment, applied to every arm equally.
GPU_FRAG=""
[ -n "$GPU_MEM_UTIL" ] && GPU_FRAG=',"gpu_memory_utilization":'"$GPU_MEM_UTIL"

overrides_for_arm() {
    local arm=$1
    case "$arm" in
        baseline)
            echo '{"0":{"max_num_seqs":64,"enable_prefix_caching":false'"$GPU_FRAG"'}}'
            ;;
        prefix_only)
            echo '{"0":{"max_num_seqs":64,"enable_prefix_caching":true'"$GPU_FRAG"'}}'
            ;;
        lmcache_only)
            echo '{"0":{"max_num_seqs":64,"enable_prefix_caching":false'"$GPU_FRAG"',"omni_kv_config":'"$LMCACHE_JSON"'}}'
            ;;
        *)
            echo "Unknown arm: $arm" >&2; exit 1 ;;
    esac
}

SERVER_PID=""
start_server() {
    local overrides=$1
    local log_path=$2
    echo "[bench] starting server, overrides=$overrides"
    setsid "$VLLM" serve "$MODEL" --omni \
        --port "$PORT" \
        --stage-overrides "$overrides" \
        > "$log_path" 2>&1 &
    SERVER_PID=$!
    echo "[bench] server PID=$SERVER_PID, waiting for readiness..."
    for i in $(seq 1 600); do
        if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
            echo "[bench] server ready after ${i}s"; return 0
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "[bench] server process died early; tail of log:" >&2
            tail -40 "$log_path" >&2; return 1
        fi
        sleep 1
    done
    echo "[bench] server failed to start within 600s" >&2
    tail -40 "$log_path" >&2
    kill -9 -- "-$SERVER_PID" 2>/dev/null || true
    return 1
}

stop_server() {
    if [ -n "${SERVER_PID:-}" ]; then
        echo "[bench] stopping server PGID=$SERVER_PID"
        kill -9 -- "-$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=""
    fi
    sleep 2
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sort -u | while read -r pid; do
        [ -z "$pid" ] && continue
        kill -9 "$pid" 2>/dev/null || true
    done
    for i in $(seq 1 30); do
        min_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | sort -n | head -1)
        [ "${min_free:-0}" -gt 70000 ] && break
        sleep 2
    done
    echo "[bench] GPU free: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr '\n' ' ')MiB"
}

run_arm() {
    local arm=$1
    local server_log="$OUT_DIR/server_${arm}_${TIMESTAMP}.log"
    local bench_log="$OUT_DIR/bench_${arm}_${TIMESTAMP}.log"
    echo ""
    echo "======================== ARM: $arm ========================"
    start_server "$(overrides_for_arm "$arm")" "$server_log"

    echo "[bench] running prefix_repetition: prefixes=$PR_NUM_PREFIXES prompts=$NUM_PROMPTS reuse=$((NUM_PROMPTS/PR_NUM_PREFIXES))x"
    "$VLLM" bench serve \
        --omni \
        --dataset-name prefix_repetition \
        --port "$PORT" \
        --max-concurrency "$MAX_CONCURRENCY" \
        --model "$MODEL" \
        --endpoint /v1/chat/completions \
        --backend openai-chat-omni \
        --num-prompts "$NUM_PROMPTS" \
        --prefix-repetition-prefix-len "$PR_PREFIX_LEN" \
        --prefix-repetition-output-len "$OUTPUT_LEN" \
        --prefix-repetition-suffix-len "$PR_SUFFIX_LEN" \
        --prefix-repetition-num-prefixes "$PR_NUM_PREFIXES" \
        --ignore-eos \
        --percentile-metrics "$PCT_METRICS" \
        --extra_body "$EXTRA_BODY" \
        2>&1 | tee "$bench_log"

    stop_server
    echo "[bench] ARM $arm done -> $bench_log"
}

cd "$ROOT"
ARMS="${*:-baseline lmcache_only prefix_only}"
trap stop_server EXIT
for arm in $ARMS; do run_arm "$arm"; done
echo ""
echo "All done. Results in $OUT_DIR"
