#!/bin/bash
# Qwen3-Omni-30B PR #1330 vs Main Benchmark Comparison
#
# Compares KV offload PR branch against main branch
#
# Usage:
#   bash compare_pr_benchmark.sh
#
# Environment variables:
#   GPU_DEVICE       - GPU index (default: 0,1)
#   MODEL            - Model path (default: Qwen/Qwen3-Omni-30B-A3B-Instruct)
#   NUM_PROMPTS      - Prompts per concurrency level (default: 10)
#   CONCURRENCY      - Space-separated concurrency levels (default: "1 4")
#   PORT_PR          - Port for PR server (default: 8000)
#   PORT_MAIN        - Port for Main server (default: 8001)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

GPU_DEVICE="${GPU_DEVICE:-0,1}"
MODEL="${MODEL:-Qwen/Qwen3-Omni-30B-A3B-Instruct}"
NUM_PROMPTS="${NUM_PROMPTS:-10}"
CONCURRENCY="${CONCURRENCY:-1 4}"
NUM_WARMUPS="${NUM_WARMUPS:-3}"
PORT_PR="${PORT_PR:-8000}"
PORT_MAIN="${PORT_MAIN:-8001}"
RESULT_DIR="${SCRIPT_DIR}/results"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# Stage configs - use the one with KV offload for PR, basic one for main
STAGE_CONFIG_PR="${SCRIPT_DIR}/qwen3_omni_kv_offload_on.yaml"
STAGE_CONFIG_MAIN="vllm_omni/model_executor/stage_configs/qwen3_omni_moe.yaml"

mkdir -p "${RESULT_DIR}"

echo "============================================================"
echo " Qwen3-Omni-30B: PR #1330 vs Main Benchmark"
echo "============================================================"
echo " GPU:            ${GPU_DEVICE}"
echo " Model:          ${MODEL}"
echo " Prompts:        ${NUM_PROMPTS}"
echo " Concurrency:    ${CONCURRENCY}"
echo " Port (PR/Main): ${PORT_PR} / ${PORT_MAIN}"
echo " Results:        ${RESULT_DIR}"
echo "============================================================"

# Save current branch
CURRENT_BRANCH=$(cd "$PROJECT_ROOT" && git rev-parse --abbrev-ref HEAD)
echo "Current branch: ${CURRENT_BRANCH}"

cleanup() {
    echo "Cleaning up servers..."
    kill "$PID_PR" 2>/dev/null || true
    kill "$PID_MAIN" 2>/dev/null || true
    wait "$PID_PR" 2>/dev/null || true
    wait "$PID_MAIN" 2>/dev/null || true

    # Return to original branch
    if [ -n "${CURRENT_BRANCH}" ]; then
        cd "$PROJECT_ROOT"
        git checkout "${CURRENT_BRANCH}" 2>/dev/null || true
    fi
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

# ---- Phase 1: Test PR branch (kv_offload) ----
echo ""
echo "[Phase 1] Testing PR #1330 (kv_offload branch) with KV offloading..."
cd "$PROJECT_ROOT"

# Ensure we're on kv_offload branch
git checkout kv_offload || {
    echo "ERROR: Failed to checkout kv_offload branch"
    exit 1
}

# Reinstall to ensure we have the latest code
echo "Installing PR version..."
pip install -e . -q

echo "Starting PR server on port ${PORT_PR}..."
CUDA_VISIBLE_DEVICES=${GPU_DEVICE} vllm-omni serve "${MODEL}" \
    --stage-configs-path "${STAGE_CONFIG_PR}" \
    --host 0.0.0.0 --port "${PORT_PR}" \
    --trust-remote-code --omni \
    > "${RESULT_DIR}/server_pr_${TIMESTAMP}.log" 2>&1 &
PID_PR=$!

wait_for_server "${PORT_PR}" "PR"

echo "Benchmarking PR #1330..."
# shellcheck disable=SC2086
python "${SCRIPT_DIR}/bench_kv_offload.py" \
    --host 127.0.0.1 --port "${PORT_PR}" \
    --config-name "pr_1330" \
    --num-prompts "${NUM_PROMPTS}" \
    --max-concurrency ${CONCURRENCY} \
    --num-warmups "${NUM_WARMUPS}" \
    --result-dir "${RESULT_DIR}"

echo "Stopping PR server..."
kill "$PID_PR" 2>/dev/null || true
wait "$PID_PR" 2>/dev/null || true
sleep 5

# ---- Phase 2: Test Main branch ----
echo ""
echo "[Phase 2] Testing Main branch (baseline without KV offload)..."
cd "$PROJECT_ROOT"

git checkout main || {
    echo "ERROR: Failed to checkout main branch"
    exit 1
}

# Reinstall to ensure we have the main version
echo "Installing Main version..."
pip install -e . -q

echo "Starting Main server on port ${PORT_MAIN}..."
CUDA_VISIBLE_DEVICES=${GPU_DEVICE} vllm-omni serve "${MODEL}" \
    --stage-configs-path "${STAGE_CONFIG_MAIN}" \
    --host 0.0.0.0 --port "${PORT_MAIN}" \
    --trust-remote-code --omni \
    > "${RESULT_DIR}/server_main_${TIMESTAMP}.log" 2>&1 &
PID_MAIN=$!

wait_for_server "${PORT_MAIN}" "Main"

echo "Benchmarking Main..."
# shellcheck disable=SC2086
python "${SCRIPT_DIR}/bench_kv_offload.py" \
    --host 127.0.0.1 --port "${PORT_MAIN}" \
    --config-name "main" \
    --num-prompts "${NUM_PROMPTS}" \
    --max-concurrency ${CONCURRENCY} \
    --num-warmups "${NUM_WARMUPS}" \
    --result-dir "${RESULT_DIR}"

echo "Stopping Main server..."
kill "$PID_MAIN" 2>/dev/null || true
wait "$PID_MAIN" 2>/dev/null || true

# Return to original branch
cd "$PROJECT_ROOT"
git checkout "${CURRENT_BRANCH}"
pip install -e . -q

# ---- Phase 3: Generate comparison ----
echo ""
echo "[Phase 3] Generating comparison table..."

# Find the latest result files
RESULT_PR=$(ls -t "${RESULT_DIR}"/bench_pr_1330_*.json 2>/dev/null | head -1)
RESULT_MAIN=$(ls -t "${RESULT_DIR}"/bench_main_*.json 2>/dev/null | head -1)

if [ -z "$RESULT_PR" ] || [ -z "$RESULT_MAIN" ]; then
    echo "ERROR: Could not find result files. Check logs in ${RESULT_DIR}/"
    exit 1
fi

echo "  PR results:   ${RESULT_PR}"
echo "  Main results: ${RESULT_MAIN}"

# Generate comparison table
python "${SCRIPT_DIR}/compare_results.py" \
    --pr "${RESULT_PR}" \
    --main "${RESULT_MAIN}" \
    --output "${RESULT_DIR}/pr_vs_main_${TIMESTAMP}.txt"

echo ""
echo "============================================================"
echo " Benchmark complete!"
echo " Results: ${RESULT_DIR}/"
echo " Comparison: ${RESULT_DIR}/pr_vs_main_${TIMESTAMP}.txt"
echo "============================================================"
