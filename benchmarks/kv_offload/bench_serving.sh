#!/usr/bin/env bash
# KV-Offload serving benchmark using vllm bench serve.
#
# Usage:
#   bash benchmarks/kv_offload/bench_serving.sh [arm]
#   arm: baseline | prefix_only | lmcache_only  (default: runs all)
#
# Requires: 2 GPUs, model weights at $MODEL_PATH

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="${PYTHON:-/tmp/miniforge3/envs/vllm312/bin/python}"
VLLM="${VLLM:-/tmp/miniforge3/envs/vllm312/bin/vllm}"

MODEL="${MODEL:-Qwen/Qwen3-Omni-30B-A3B-Instruct}"
PORT="${PORT:-46354}"
OUT_DIR="$SCRIPT_DIR/results"
mkdir -p "$OUT_DIR"

# Cache dirs (avoid NFS home)
export HF_HOME="${HF_HOME:-/tmp/local-yiqix-hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_HUB_CACHE="$HF_HOME/hub"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/tmp/local-yiqix-vllm_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/local-yiqix-xdg_cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/local-yiqix-triton_cache}"
export TORCH_HOME="${TORCH_HOME:-/tmp/local-yiqix-torch_home}"
export SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0
export PYTHONHASHSEED=0
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
CONDA_LIB="/tmp/miniforge3/envs/vllm312/lib"
export LD_LIBRARY_PATH="${CONDA_LIB}:${LD_LIBRARY_PATH:-}"

# Benchmark parameters
NUM_PROMPTS="${NUM_PROMPTS:-100}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-10}"
RANDOM_INPUT_LEN="${RANDOM_INPUT_LEN:-400}"
RANDOM_OUTPUT_LEN="${RANDOM_OUTPUT_LEN:-100}"
RANDOM_PREFIX_LEN="${RANDOM_PREFIX_LEN:-200}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# --- YAML generators ---

gen_yaml_baseline() {
    cat <<'YAML'
async_chunk: true
stage_args:
  - stage_id: 0
    stage_type: llm
    runtime:
      devices: "0"
    engine_args:
      model_stage: thinker
      max_num_seqs: 64
      model_arch: Qwen3OmniMoeForConditionalGeneration
      worker_type: ar
      scheduler_cls: vllm_omni.core.sched.omni_ar_scheduler.OmniARScheduler
      gpu_memory_utilization: 0.9
      enforce_eager: false
      trust_remote_code: true
      engine_output_type: latent
      distributed_executor_backend: "mp"
      enable_prefix_caching: false
      max_num_batched_tokens: 32768
      hf_config_name: thinker_config
      tensor_parallel_size: 1
      custom_process_next_stage_input_func: vllm_omni.model_executor.stage_input_processors.qwen3_omni.thinker2talker_async_chunk
    final_output: true
    final_output_type: text
    is_comprehension: true
    output_connectors:
      to_stage_1: connector_of_shared_memory
    default_sampling_params:
      temperature: 0.0
      max_tokens: 2048
      seed: 42
      detokenize: true
      repetition_penalty: 1.05

  - stage_id: 1
    stage_type: llm
    runtime:
      devices: "1"
    engine_args:
      model_stage: talker
      max_num_seqs: 64
      model_arch: Qwen3OmniMoeForConditionalGeneration
      worker_type: ar
      scheduler_cls: vllm_omni.core.sched.omni_ar_scheduler.OmniARScheduler
      gpu_memory_utilization: 0.6
      enforce_eager: false
      trust_remote_code: true
      engine_output_type: latent
      enable_prefix_caching: false
      max_num_batched_tokens: 32768
      distributed_executor_backend: "mp"
      hf_config_name: talker_config
      custom_process_next_stage_input_func: vllm_omni.model_executor.stage_input_processors.qwen3_omni.talker2code2wav_async_chunk
    engine_input_source: [0]
    input_connectors:
      from_stage_0: connector_of_shared_memory
    default_sampling_params:
      temperature: 0.9
      top_k: 50
      max_tokens: 4096
      seed: 42
      detokenize: false
      repetition_penalty: 1.05
      stop_token_ids: [2150]

  - stage_id: 2
    stage_type: llm
    runtime:
      devices: "1"
    engine_args:
      model_stage: code2wav
      max_num_seqs: 64
      model_arch: Qwen3OmniMoeForConditionalGeneration
      worker_type: generation
      scheduler_cls: vllm_omni.core.sched.omni_generation_scheduler.OmniGenerationScheduler
      enforce_eager: true
      trust_remote_code: true
      async_scheduling: false
      enable_prefix_caching: false
      engine_output_type: audio
      gpu_memory_utilization: 0.1
      distributed_executor_backend: "mp"
      max_num_batched_tokens: 51200
      hf_config_name: thinker_config
    engine_input_source: [1]
    final_output: true
    final_output_type: audio
    default_sampling_params:
      temperature: 0.0
      top_p: 1.0
      top_k: -1
      max_tokens: 65536
      seed: 42
      detokenize: true
      repetition_penalty: 1.1

runtime:
  connectors:
    connector_of_shared_memory:
      name: SharedMemoryConnector
      extra:
        codec_chunk_frames: 25
        codec_left_context_frames: 25
YAML
}

gen_yaml_prefix_only() {
    # Same as baseline but enable_prefix_caching: true on thinker
    gen_yaml_baseline | sed 's/enable_prefix_caching: false/enable_prefix_caching: true/' | head -n 38 | sed '$ a\      enable_prefix_caching: true'
}

gen_yaml_lmcache_only() {
    # prefix_caching + LMCache connector on thinker
    gen_yaml_baseline | sed '/enable_prefix_caching: false/{
        s/enable_prefix_caching: false/enable_prefix_caching: true/
    }' | sed '/^      enable_prefix_caching: true$/a\      omni_kv_config:\n        kv_store_config:\n          lmcache_config:\n            chunk_size: 256\n            local_cpu: true\n            max_local_cpu_size: 30'
}

# Better approach: generate per-arm yaml properly
write_yaml() {
    local arm=$1
    local out=$2
    local base_yaml
    base_yaml=$(gen_yaml_baseline)

    case "$arm" in
        baseline)
            echo "$base_yaml" > "$out"
            ;;
        prefix_only)
            echo "$base_yaml" | sed '0,/enable_prefix_caching: false/{s/enable_prefix_caching: false/enable_prefix_caching: true/}' > "$out"
            ;;
        lmcache_only)
            # Enable prefix caching on thinker (first occurrence only) + add omni_kv_config
            echo "$base_yaml" | sed '0,/enable_prefix_caching: false/{s/enable_prefix_caching: false/enable_prefix_caching: true/}' | \
            sed '0,/enable_prefix_caching: true/{//a\      omni_kv_config:\n        kv_store_config:\n          lmcache_config:\n            chunk_size: '"${LMCACHE_CHUNK_SIZE:-64}"'\n            local_cpu: true\n            max_local_cpu_size: 30\n            skip_mm_storage: '"${LMCACHE_SKIP_MM_STORAGE:-true}"'
            }' > "$out"
            ;;
        lmcache_no_prefix)
            # LMCache enabled, prefix caching OFF on thinker (first stage only)
            echo "$base_yaml" | \
            sed '0,/enable_prefix_caching: false/{//a\      omni_kv_config:\n        kv_store_config:\n          lmcache_config:\n            chunk_size: '"${LMCACHE_CHUNK_SIZE:-64}"'\n            local_cpu: true\n            max_local_cpu_size: 30\n            skip_mm_storage: '"${LMCACHE_SKIP_MM_STORAGE:-true}"'
            }' > "$out"
            ;;
        *)
            echo "Unknown arm: $arm" >&2; exit 1
            ;;
    esac
}

# --- Server lifecycle ---

start_server() {
    local yaml_path=$1
    local log_path=$2
    echo "[bench] Starting server with config: $yaml_path"
    # setsid puts the server in its own process group so we can kill it transitively
    setsid $VLLM serve "$MODEL" --omni \
        --port "$PORT" \
        --stage-configs-path "$yaml_path" \
        > "$log_path" 2>&1 &
    SERVER_PID=$!
    echo "[bench] Server PID=$SERVER_PID (PGID=$SERVER_PID), waiting for readiness..."

    # Wait for server to be ready
    for i in $(seq 1 300); do
        if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
            echo "[bench] Server ready after ${i}s"
            return 0
        fi
        sleep 1
    done
    echo "[bench] Server failed to start within 300s" >&2
    kill -9 -- "-$SERVER_PID" 2>/dev/null || true
    return 1
}

stop_server() {
    if [ -n "${SERVER_PID:-}" ]; then
        echo "[bench] Stopping server PGID=$SERVER_PID"
        # Negative PID = kill the entire process group (catches all multiprocessing children)
        kill -9 -- "-$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=""
    fi
    # Backstop: kill any GPU compute apps that survived (across all GPUs)
    sleep 2
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sort -u | while read pid; do
        [ -z "$pid" ] && continue
        [ "$pid" = "$$" ] && continue
        kill -9 "$pid" 2>/dev/null || true
    done
    # Wait for GPU memory to be fully released across all GPUs
    for i in $(seq 1 30); do
        min_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | sort -n | head -1)
        [ "${min_free:-0}" -gt 70000 ] && break
        sleep 2
    done
    echo "[bench] GPU memory freed: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr '\n' ' ') MiB free"
}

# --- Run benchmark for one arm ---

run_arm() {
    local arm=$1
    local yaml_path="$OUT_DIR/stage_config_${arm}.yaml"
    local server_log="$OUT_DIR/server_${arm}_${TIMESTAMP}.log"
    local bench_log="$OUT_DIR/bench_${arm}_${TIMESTAMP}.log"

    echo ""
    echo "========================================================"
    echo "  ARM: $arm"
    echo "========================================================"

    write_yaml "$arm" "$yaml_path"
    start_server "$yaml_path" "$server_log"

    echo "[bench] Running benchmark: $arm (input_modality=${INPUT_MODALITY:-text})"
    $VLLM bench serve \
        --omni \
        --dataset-name random-mm \
        --port "$PORT" \
        --max-concurrency "$MAX_CONCURRENCY" \
        --model "$MODEL" \
        --endpoint /v1/chat/completions \
        --backend openai-chat-omni \
        --num-prompts "$NUM_PROMPTS" \
        --random-input-len "$RANDOM_INPUT_LEN" \
        --random-output-len "$RANDOM_OUTPUT_LEN" \
        --random-prefix-len "$RANDOM_PREFIX_LEN" \
        --random-mm-base-items-per-request "${MM_BASE_ITEMS:-0}" \
        --ignore-eos \
        --percentile-metrics ttft,tpot,itl,e2el,audio_ttfp,audio_rtf \
        --extra_body '{"modalities": ["text", "audio"]}' \
        2>&1 | tee "$bench_log"

    stop_server
    echo "[bench] ARM $arm done. Results: $bench_log"
}

# --- Main ---

cd "$ROOT"

ARMS="${1:-baseline prefix_only lmcache_only}"

trap stop_server EXIT

for arm in $ARMS; do
    run_arm "$arm"
done

echo ""
echo "========================================================"
echo "All benchmarks complete. Results in: $OUT_DIR"
echo "========================================================"
