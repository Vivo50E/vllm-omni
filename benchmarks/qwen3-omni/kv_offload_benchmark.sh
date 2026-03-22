#!/bin/bash
# Qwen3-Omni-30B KV Cache Offloading Benchmark
#
# Compares throughput/latency with KV offloading ON vs OFF
#
# Usage:
#   bash kv_offload_benchmark.sh
#
# Environment variables:
#   GPU_DEVICE       - GPU index (default: 0,1)
#   MODEL            - Model path (default: Qwen/Qwen3-Omni-30B-A3B-Instruct)
#   NUM_PROMPTS      - Prompts per concurrency level (default: 50)
#   CONCURRENCY      - Space-separated concurrency levels (default: "1 4 10")
#   PORT_ON          - Port for KV offload ON server (default: 8000)
#   PORT_OFF         - Port for KV offload OFF server (default: 8001)
#   CPU_OFFLOAD_GB   - CPU memory for offloading in GB (default: 10.0)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$PROJECT_ROOT"

GPU_DEVICE="${GPU_DEVICE:-0,1}"
MODEL="${MODEL:-Qwen/Qwen3-Omni-30B-A3B-Instruct}"
NUM_PROMPTS="${NUM_PROMPTS:-50}"
CONCURRENCY="${CONCURRENCY:-1 4 10}"
NUM_WARMUPS="${NUM_WARMUPS:-3}"
PORT_ON="${PORT_ON:-8000}"
PORT_OFF="${PORT_OFF:-8001}"
CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-10.0}"
RESULT_DIR="${SCRIPT_DIR}/results"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

STAGE_CONFIG_ON="${SCRIPT_DIR}/qwen3_omni_kv_offload_on.yaml"
STAGE_CONFIG_OFF="${SCRIPT_DIR}/qwen3_omni_kv_offload_off.yaml"

mkdir -p "${RESULT_DIR}"

echo "============================================================"
echo " Qwen3-Omni-30B KV Cache Offloading Benchmark"
echo "============================================================"
echo " GPU:            ${GPU_DEVICE}"
echo " Model:          ${MODEL}"
echo " Prompts:        ${NUM_PROMPTS}"
echo " Concurrency:    ${CONCURRENCY}"
echo " CPU Offload:    ${CPU_OFFLOAD_GB} GB"
echo " Port (on/off):  ${PORT_ON} / ${PORT_OFF}"
echo " Results:        ${RESULT_DIR}"
echo "============================================================"

cleanup() {
    echo "Cleaning up servers..."
    kill "$PID_ON" 2>/dev/null || true
    kill "$PID_OFF" 2>/dev/null || true
    wait "$PID_ON" 2>/dev/null || true
    wait "$PID_OFF" 2>/dev/null || true
}
trap cleanup EXIT

wait_for_server() {
    local port=$1
    local name=$2
    local max_wait=600
    local elapsed=0
    echo "Waiting for ${name} server on port ${port}..."
    while ! curl -s "http://localhost:${port}/health" >/dev/null 2>&1; do
        sleep 5
        elapsed=$((elapsed + 5))
        if [ $elapsed -ge $max_wait ]; then
            echo "ERROR: ${name} server failed to start within ${max_wait}s"
            exit 1
        fi
    done
    echo "${name} server ready (${elapsed}s)"
}

# ---- Phase 1: Start KV Offload ON server ----
echo ""
echo "[Phase 1] Starting KV Offload ON server on port ${PORT_ON}..."
CUDA_VISIBLE_DEVICES=${GPU_DEVICE} vllm-omni serve "${MODEL}" \
    --stage-configs-path "${STAGE_CONFIG_ON}" \
    --host 0.0.0.0 --port "${PORT_ON}" \
    --trust-remote-code --omni \
    > "${RESULT_DIR}/server_on_${TIMESTAMP}.log" 2>&1 &
PID_ON=$!

wait_for_server "${PORT_ON}" "kv_offload_on"

echo "[Phase 1] Benchmarking KV Offload ON..."
# shellcheck disable=SC2086
python "${SCRIPT_DIR}/bench_kv_offload.py" \
    --host 127.0.0.1 --port "${PORT_ON}" \
    --config-name "kv_offload_on" \
    --num-prompts "${NUM_PROMPTS}" \
    --max-concurrency ${CONCURRENCY} \
    --num-warmups "${NUM_WARMUPS}" \
    --result-dir "${RESULT_DIR}"

echo "[Phase 1] Stopping KV Offload ON server..."
kill "$PID_ON" 2>/dev/null || true
wait "$PID_ON" 2>/dev/null || true
sleep 5

# ---- Phase 2: Start KV Offload OFF server ----
echo ""
echo "[Phase 2] Starting KV Offload OFF server on port ${PORT_OFF}..."
CUDA_VISIBLE_DEVICES=${GPU_DEVICE} vllm-omni serve "${MODEL}" \
    --stage-configs-path "${STAGE_CONFIG_OFF}" \
    --host 0.0.0.0 --port "${PORT_OFF}" \
    --trust-remote-code --omni \
    > "${RESULT_DIR}/server_off_${TIMESTAMP}.log" 2>&1 &
PID_OFF=$!

wait_for_server "${PORT_OFF}" "kv_offload_off"

echo "[Phase 2] Benchmarking KV Offload OFF..."
# shellcheck disable=SC2086
python "${SCRIPT_DIR}/bench_kv_offload.py" \
    --host 127.0.0.1 --port "${PORT_OFF}" \
    --config-name "kv_offload_off" \
    --num-prompts "${NUM_PROMPTS}" \
    --max-concurrency ${CONCURRENCY} \
    --num-warmups "${NUM_WARMUPS}" \
    --result-dir "${RESULT_DIR}"

echo "[Phase 2] Stopping KV Offload OFF server..."
kill "$PID_OFF" 2>/dev/null || true
wait "$PID_OFF" 2>/dev/null || true

# ---- Phase 3: Plot results ----
echo ""
echo "[Phase 3] Generating plots..."

# Find the latest result files
RESULT_ON=$(ls -t "${RESULT_DIR}"/bench_kv_offload_on_*.json 2>/dev/null | head -1)
RESULT_OFF=$(ls -t "${RESULT_DIR}"/bench_kv_offload_off_*.json 2>/dev/null | head -1)

if [ -z "$RESULT_ON" ] || [ -z "$RESULT_OFF" ]; then
    echo "ERROR: Could not find result files. Check logs in ${RESULT_DIR}/"
    exit 1
fi

echo "  ON results:  ${RESULT_ON}"
echo "  OFF results: ${RESULT_OFF}"

# Generate comparison plots
python "${SCRIPT_DIR}/plot_kv_offload.py" \
    --off "${RESULT_OFF}" \
    --on "${RESULT_ON}" \
    --output "${RESULT_DIR}/qwen3_omni_kv_offload_comparison.png"

echo ""
echo "============================================================"
echo " Benchmark complete!"
echo " Results: ${RESULT_DIR}/"
echo " Plot: ${RESULT_DIR}/qwen3_omni_kv_offload_comparison.png"
echo "============================================================"
