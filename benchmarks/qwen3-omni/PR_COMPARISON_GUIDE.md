# PR #1330 vs Main Benchmark Comparison Guide

This guide shows how to benchmark PR #1330 (KV Cache CPU Offloading) against the main branch.

## Overview

The comparison benchmark:
1. Tests PR #1330 (kv_offload branch) with KV offloading enabled
2. Tests Main branch without KV offloading
3. Generates a formatted comparison table with performance deltas

## Output Format

The benchmark produces a comparison table like this:

```
Metric              PR #1330 c=1    Main c=1    Delta      PR #1330 c=4    Main c=4    Delta
=========================================================================================
completed           10/10           10/10       =          40/40           40/40       =
RTF                 0.283           0.299       -5.4%      0.780           0.890       -12.4%
TTFT (ms)           71.4            89.6        -20.3%     2047            1363        +50.2%
E2EL (ms)           7659            8027        -4.6%      23163           26394       -12.2%
throughput (req/s)  0.131           0.125       +4.8%      0.166           0.146       +13.7%
```

## Metrics Explained

- **completed**: Number of successful requests out of total
- **RTF**: Real-Time Factor (processing time / audio duration, lower is better)
- **TTFT**: Time To First Token/Packet in milliseconds (lower is better)
- **E2EL**: End-to-End Latency in milliseconds (lower is better)
- **throughput**: Requests processed per second (higher is better)

Delta percentages:
- **Negative values** (green) = PR is faster/better (for latency/RTF)
- **Positive values** (red) = PR is slower/worse (for latency/RTF)
- For throughput, positive is better (more requests/sec)

## Quick Start

### Prerequisites

```bash
# Ensure you have both branches available
cd /home/yiqix/vllm-omni
git fetch origin

# Ensure dependencies are installed
pip install aiohttp numpy tqdm matplotlib
```

### Run Comparison Benchmark

```bash
cd /home/yiqix/vllm-omni/benchmarks/qwen3-omni

# Run with default settings (c=1, c=4, 10 prompts each)
bash compare_pr_benchmark.sh

# Or customize
GPU_DEVICE="0,1" \
MODEL="Qwen/Qwen3-Omni-MoE-30B-A3B-Instruct" \
NUM_PROMPTS=10 \
CONCURRENCY="1 4" \
bash compare_pr_benchmark.sh
```

### Environment Variables

- `GPU_DEVICE`: GPU indices (default: "0,1")
- `MODEL`: Model path (default: "Qwen/Qwen3-Omni-MoE-30B-A3B-Instruct")
- `NUM_PROMPTS`: Number of prompts per concurrency level (default: 10)
- `CONCURRENCY`: Space-separated concurrency levels (default: "1 4")
- `PORT_PR`: Port for PR server (default: 8000)
- `PORT_MAIN`: Port for Main server (default: 8001)

## How It Works

The benchmark script:

1. **Saves your current branch** - Returns to it at the end
2. **Phase 1: Test PR #1330**
   - Checks out `kv_offload` branch
   - Reinstalls vllm-omni from PR branch
   - Starts server with KV offloading enabled
   - Runs benchmarks for each concurrency level
   - Stops server

3. **Phase 2: Test Main**
   - Checks out `main` branch
   - Reinstalls vllm-omni from main
   - Starts server without KV offloading
   - Runs same benchmarks
   - Stops server

4. **Phase 3: Compare Results**
   - Generates formatted comparison table
   - Shows deltas for each metric
   - Saves detailed statistics

5. **Cleanup**
   - Returns to your original branch
   - Reinstalls from original branch

## Output Files

All results are saved in `results/` directory:

- `bench_pr_1330_TIMESTAMP.json` - PR benchmark data
- `bench_main_TIMESTAMP.json` - Main benchmark data
- `pr_vs_main_TIMESTAMP.txt` - Formatted comparison table
- `server_pr_TIMESTAMP.log` - PR server logs
- `server_main_TIMESTAMP.log` - Main server logs

## Example Session

```bash
$ cd /home/yiqix/vllm-omni/benchmarks/qwen3-omni
$ bash compare_pr_benchmark.sh

============================================================
 Qwen3-Omni-30B: PR #1330 vs Main Benchmark
============================================================
 GPU:            0,1
 Model:          Qwen/Qwen3-Omni-MoE-30B-A3B-Instruct
 Prompts:        10
 Concurrency:    1 4
 Port (PR/Main): 8000 / 8001
 Results:        results/
============================================================
Current branch: kv_offload

[Phase 1] Testing PR #1330 (kv_offload branch) with KV offloading...
...
[Phase 2] Testing Main branch (baseline without KV offload)...
...
[Phase 3] Generating comparison table...

Metric              PR #1330 c=1    Main c=1    Delta      PR #1330 c=4    Main c=4    Delta
=========================================================================================
completed           10/10           10/10       =          40/40           40/40       =
RTF                 0.283           0.299       -5.4%      0.780           0.890       -12.4%
TTFT (ms)           71.4            89.6        -20.3%     2047            1363        +50.2%
E2EL (ms)           7659            8027        -4.6%      23163           26394       -12.2%
throughput (req/s)  0.131           0.125       +4.8%      0.166           0.146       +13.7%

============================================================
 Benchmark complete!
 Results: results/
 Comparison: results/pr_vs_main_20260320_062345.txt
============================================================
```

## Manual Comparison

If you want to compare existing results manually:

```bash
python compare_results.py \
    --pr results/bench_pr_1330_20260320_123456.json \
    --main results/bench_main_20260320_123456.json \
    --output results/custom_comparison.txt
```

## Troubleshooting

### Branch checkout fails
- Ensure you have committed or stashed changes in your working directory
- The script will try to return to your original branch even on error

### Installation fails between branches
- Check that both branches are compatible with your environment
- You may need to manually install dependencies for each branch

### Server fails to start
- Check GPU memory availability
- Review server logs in `results/server_*.log`
- Ensure model is downloaded and accessible

### Results don't match concurrency levels
- Verify NUM_PROMPTS and CONCURRENCY are set correctly
- Check that both benchmarks completed successfully
- Review JSON result files

## Notes

- The script automatically switches between branches and reinstalls
- Your original branch is restored at the end
- Server logs are saved for debugging
- All operations are sequential to avoid port conflicts
- The comparison uses the same prompts for both configurations

## Related Files

- `compare_pr_benchmark.sh` - Main orchestration script
- `bench_kv_offload.py` - Benchmark client (reused from KV offload benchmark)
- `compare_results.py` - Table generation script
- `qwen3_omni_kv_offload_on.yaml` - PR config with KV offloading
- `vllm_omni/model_executor/stage_configs/qwen3_omni_moe.yaml` - Main config
