#!/bin/bash
set -e

PROJECT_ROOT="/tmp/vllm-omni-benchmark/vllm-omni"
SCRIPT_DIR="${PROJECT_ROOT}/benchmarks/qwen3-omni"
RESULT_DIR="${SCRIPT_DIR}/results"
PORT=8100
NUM_PROMPTS=50
CONCURRENCY_LEVELS=(1 4 10)

export HF_HOME=/tmp/hf_home
export CUDA_VISIBLE_DEVICES=0,1

mkdir -p "${RESULT_DIR}"

kill_server() {
    echo "Killing existing vllm processes..."
    pkill -f "vllm_omni.entrypoints" 2>/dev/null || true
    sleep 5
    pkill -9 -f "vllm_omni.entrypoints" 2>/dev/null || true
    sleep 3
}

start_server() {
    local config_name=$1
    local config_file="${SCRIPT_DIR}/qwen3_omni_${config_name}.yaml"

    echo ""
    echo "=========================================="
    echo "Starting server with config: ${config_name}"
    echo "=========================================="

    cd "${PROJECT_ROOT}"
    nohup .venv/bin/python -m vllm_omni.entrypoints.cli.main serve \
        Qwen/Qwen3-Omni-30B-A3B-Instruct \
        --omni \
        --stage-configs-path "${config_file}" \
        --port ${PORT} &> "/tmp/server_${config_name}.log" &

    echo "Waiting for server to be ready..."
    for i in {1..180}; do
        if curl -s "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then
            echo "Server ready after ${i} seconds"
            sleep 5
            return 0
        fi
        [ $((i % 30)) -eq 0 ] && echo "  Still waiting... (${i}s)"
        sleep 1
    done
    echo "ERROR: Server failed to start!"
    tail -100 "/tmp/server_${config_name}.log"
    return 1
}

run_single_benchmark() {
    local config_name=$1
    local concurrency=$2
    local timestamp=$(date +%Y%m%d_%H%M%S)

    echo ""
    echo "--- Running: ${config_name} @ concurrency=${concurrency} ---"

    cd "${PROJECT_ROOT}"
    .venv/bin/python -m vllm_omni.entrypoints.cli.main bench serve \
        --backend openai-chat-omni \
        --base-url "http://127.0.0.1:${PORT}" \
        --endpoint "/v1/chat/completions" \
        --model "Qwen/Qwen3-Omni-30B-A3B-Instruct" \
        --num-prompts ${NUM_PROMPTS} \
        --max-concurrency ${concurrency} \
        --ignore-eos \
        --save-result \
        --result-dir "${RESULT_DIR}" \
        --result-filename "bench_${config_name}_c${concurrency}_${timestamp}.json" \
        2>&1 | tee -a "${RESULT_DIR}/bench_${config_name}_log.txt"
}

CONFIGS=("baseline" "lmcache" "kv_offload")

echo "=========================================="
echo "Full Benchmark Starting at $(date)"
echo "NUM_PROMPTS=${NUM_PROMPTS}"
echo "CONCURRENCY=${CONCURRENCY_LEVELS[*]}"
echo "CONFIGS=${CONFIGS[*]}"
echo "=========================================="

for config in "${CONFIGS[@]}"; do
    kill_server
    if ! start_server "${config}"; then
        echo "ERROR: Failed to start ${config}, skipping..."
        continue
    fi
    for c in "${CONCURRENCY_LEVELS[@]}"; do
        run_single_benchmark "${config}" "${c}"
    done
done

kill_server

echo ""
echo "=========================================="
echo "All benchmarks complete at $(date)"
echo "Results in: ${RESULT_DIR}"
ls -la "${RESULT_DIR}"/*.json 2>/dev/null || echo "No JSON files found"
echo "=========================================="
