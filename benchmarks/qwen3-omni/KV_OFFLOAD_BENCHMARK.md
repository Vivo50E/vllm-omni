# Qwen3-Omni-30B KV Cache Offloading Benchmark

This benchmark suite evaluates the performance impact of KV cache CPU offloading on Qwen3-Omni-30B model as introduced in PR #1330.

## Overview

The benchmark compares two configurations:
- **KV Offload OFF**: Baseline configuration without CPU offloading
- **KV Offload ON**: With CPU offloading enabled (10GB CPU memory)

## Metrics Measured

Based on the [Qwen3-Omni TTS Performance Blog](https://github.com/Shirley125/vllm-omni/blob/cbf8e5033d6d58c12dd3d33748cf8cc80ad217da/docs/design/qwen3_omni_tts_performance_blog.md):

1. **TTFP (Time to First Packet)**: Latency before the first audio chunk is received (ms)
2. **E2E Latency**: Total end-to-end request latency (ms)
3. **RTF (Real-Time Factor)**: Processing speed relative to audio duration (lower is better)
4. **Request Throughput**: Number of requests processed per second

## Test Configurations

- **Concurrency Levels**: 1, 4, 10 (configurable)
- **Number of Prompts**: 50 per concurrency level (configurable)
- **CPU Offload Size**: 10GB (configurable)
- **GPUs**: 2 GPUs (default: 0,1)

## Quick Start

### Prerequisites

```bash
# Install vllm-omni (from kv_offload branch)
cd /home/yiqix/vllm-omni
pip install -e .

# Install dependencies
pip install aiohttp numpy tqdm matplotlib
```

### Run Benchmark

```bash
cd /home/yiqix/vllm-omni/benchmarks/qwen3-omni

# Run with default settings
bash kv_offload_benchmark.sh

# Or customize settings
GPU_DEVICE="0,1" \
MODEL="Qwen/Qwen3-Omni-MoE-30B-A3B-Instruct" \
NUM_PROMPTS=100 \
CONCURRENCY="1 4 10" \
CPU_OFFLOAD_GB=10.0 \
bash kv_offload_benchmark.sh
```

### Environment Variables

- `GPU_DEVICE`: GPU indices (default: "0,1")
- `MODEL`: Model name or path (default: "Qwen/Qwen3-Omni-MoE-30B-A3B-Instruct")
- `NUM_PROMPTS`: Number of prompts per concurrency level (default: 50)
- `CONCURRENCY`: Space-separated concurrency levels (default: "1 4 10")
- `CPU_OFFLOAD_GB`: CPU memory for offloading in GB (default: 10.0)
- `PORT_ON`: Port for KV offload ON server (default: 8000)
- `PORT_OFF`: Port for KV offload OFF server (default: 8001)

## Output

The benchmark generates:

1. **JSON Results**: `results/bench_kv_offload_{on|off}_TIMESTAMP.json`
2. **Comparison Plot**: `results/qwen3_omni_kv_offload_comparison.png`
3. **Server Logs**: `results/server_{on|off}_TIMESTAMP.log`
4. **Console Summary**: Table showing performance differences

## Files

- `kv_offload_benchmark.sh`: Main benchmark orchestration script
- `bench_kv_offload.py`: Python client for TTS benchmarking
- `plot_kv_offload.py`: Visualization script
- `qwen3_omni_kv_offload_on.yaml`: Stage config WITH KV offloading
- `qwen3_omni_kv_offload_off.yaml`: Stage config WITHOUT KV offloading

## Example Output

```
============================================================
BENCHMARK SUMMARY
============================================================
Concurrency  Metric              OFF             ON              Change
--------------------------------------------------------------------------------
1            TTFP (ms)           125.3           118.7           -5.3%
             E2E Latency (ms)    450.2           438.5           -2.6%
             RTF                 0.234           0.228           -2.6%
             Throughput (req/s)  2.22            2.28            +2.7%
--------------------------------------------------------------------------------
4            TTFP (ms)           156.8           145.2           -7.4%
             E2E Latency (ms)    512.3           495.1           -3.4%
             RTF                 0.267           0.258           -3.4%
             Throughput (req/s)  7.81            8.08            +3.5%
============================================================
```

## Notes

- The benchmark runs sequentially: first KV offload ON, then OFF
- Each phase includes warmup requests before measurement
- Results are saved with timestamps to avoid overwriting
- The plot shows percentage changes with color-coded annotations
- Server logs are saved for debugging if needed

## Troubleshooting

### Server fails to start
- Check GPU availability and memory
- Verify model is downloaded and accessible
- Review server logs in `results/server_*.log`

### Out of memory errors
- Reduce `NUM_PROMPTS` or `CONCURRENCY`
- Adjust `gpu_memory_utilization` in stage configs
- Increase `CPU_OFFLOAD_GB` for KV offload configuration

### Import errors
- Ensure all dependencies are installed
- Verify vllm-omni is installed from the kv_offload branch
- Check Python environment is activated

## Related Documentation

- [PR #1330: KV Cache CPU Offloading](https://github.com/vllm-project/vllm-omni/pull/1330)
- [Qwen3-Omni TTS Performance Blog](https://github.com/Shirley125/vllm-omni/blob/cbf8e5033d6d58c12dd3d33748cf8cc80ad217da/docs/design/qwen3_omni_tts_performance_blog.md)
