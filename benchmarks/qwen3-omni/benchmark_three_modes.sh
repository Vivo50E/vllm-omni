#!/bin/bash
# Qwen3-Omni-30B Three-Mode Benchmark: Baseline vs Offload vs LMCache
#
# Tests three configurations at c=1, 4, 10:
#   - baseline: No KV offload, no LMCache
#   - offload:  KV offload only
#   - lmcache:  LMCache only
#
# Usage:
#   bash benchmark_three_modes.sh
#
# Environment variables:
#   GPU_DEVICE       - GPU indices (default: 0,1)
#   MODEL            - Model path (default: Qwen/Qwen3-Omni-30B-A3B-Instruct)
#   NUM_PROMPTS      - Prompts per concurrency level (default: 10)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$PROJECT_ROOT"

GPU_DEVICE="${GPU_DEVICE:-0,1}"
MODEL="${MODEL:-Qwen/Qwen3-Omni-30B-A3B-Instruct}"
NUM_PROMPTS="${NUM_PROMPTS:-10}"
CONCURRENCY="1 4 10"
NUM_WARMUPS="${NUM_WARMUPS:-3}"
BASE_PORT=8000
RESULT_DIR="${SCRIPT_DIR}/results"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# Stage configs for three modes
STAGE_CONFIG_BASELINE="${SCRIPT_DIR}/qwen3_omni_baseline.yaml"
STAGE_CONFIG_OFFLOAD="${SCRIPT_DIR}/qwen3_omni_kv_offload_on.yaml"
STAGE_CONFIG_LMCACHE="${SCRIPT_DIR}/qwen3_omni_lmcache.yaml"

mkdir -p "${RESULT_DIR}"

echo "============================================================"
echo " Qwen3-Omni-30B Three-Mode Benchmark"
echo "============================================================"
echo " GPU:         ${GPU_DEVICE}"
echo " Model:       ${MODEL}"
echo " Prompts:     ${NUM_PROMPTS}"
echo " Concurrency: ${CONCURRENCY}"
echo " Results:     ${RESULT_DIR}"
echo "============================================================"

cleanup() {
    echo "Cleaning up servers..."
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

wait_for_server() {
    local port=$1
    local name=$2
    local max_wait=1200
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

run_benchmark() {
    local mode=$1
    local config=$2
    local port=$3

    echo ""
    echo "============================================================"
    echo " Testing Mode: ${mode}"
    echo "============================================================"

    echo "Starting ${mode} server on port ${port}..."
    HF_HOME=/tmp/hf_home CUDA_VISIBLE_DEVICES=${GPU_DEVICE} .venv/bin/python -m vllm_omni.entrypoints.cli.main serve "${MODEL}" \
        --stage-configs-path "${config}" \
        --host 0.0.0.0 --port "${port}" \
        --trust-remote-code --omni \
        > "${RESULT_DIR}/server_${mode}_${TIMESTAMP}.log" 2>&1 &
    SERVER_PID=$!

    wait_for_server "${port}" "${mode}"

    echo "Benchmarking ${mode}..."
    # shellcheck disable=SC2086
    "${PROJECT_ROOT}/.venv/bin/python" "${SCRIPT_DIR}/bench_kv_offload.py" \
        --host 127.0.0.1 --port "${port}" \
        --config-name "${mode}" \
        --num-prompts "${NUM_PROMPTS}" \
        --max-concurrency ${CONCURRENCY} \
        --num-warmups "${NUM_WARMUPS}" \
        --result-dir "${RESULT_DIR}"

    echo "Stopping ${mode} server..."
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    sleep 5
}

# Run benchmarks for all three modes (baseline -> lmcache -> offload)
run_benchmark "baseline" "${STAGE_CONFIG_BASELINE}" $((BASE_PORT + 0))
run_benchmark "lmcache" "${STAGE_CONFIG_LMCACHE}" $((BASE_PORT + 1))
run_benchmark "offload" "${STAGE_CONFIG_OFFLOAD}" $((BASE_PORT + 2))

# Generate comparison
echo ""
echo "============================================================"
echo " Generating comparison table..."
echo "============================================================"

# Find the latest result files
RESULT_BASELINE=$(ls -t "${RESULT_DIR}"/bench_baseline_*.json 2>/dev/null | head -1)
RESULT_OFFLOAD=$(ls -t "${RESULT_DIR}"/bench_offload_*.json 2>/dev/null | head -1)
RESULT_LMCACHE=$(ls -t "${RESULT_DIR}"/bench_lmcache_*.json 2>/dev/null | head -1)

if [ -z "$RESULT_BASELINE" ] || [ -z "$RESULT_OFFLOAD" ] || [ -z "$RESULT_LMCACHE" ]; then
    echo "ERROR: Could not find result files. Check logs in ${RESULT_DIR}/"
    exit 1
fi

echo "  Baseline results: ${RESULT_BASELINE}"
echo "  Offload results:  ${RESULT_OFFLOAD}"
echo "  LMCache results:  ${RESULT_LMCACHE}"

# Generate comparison table
"${PROJECT_ROOT}/.venv/bin/python" "${SCRIPT_DIR}/compare_three_modes.py" \
    --baseline "${RESULT_BASELINE}" \
    --offload "${RESULT_OFFLOAD}" \
    --lmcache "${RESULT_LMCACHE}" \
    --output "${RESULT_DIR}/three_mode_comparison_${TIMESTAMP}.txt"

echo ""
echo "============================================================"
echo " Benchmark complete!"
echo " Results: ${RESULT_DIR}/"
echo " Comparison: ${RESULT_DIR}/three_mode_comparison_${TIMESTAMP}.txt"
echo "============================================================"
