# Qwen3-Omni-30B 三模式性能对比测试

测试 PR #1330 中的三种配置在 c=1, 4, 10 下的性能表现。

## 三种模式

1. **Baseline** - 不启用任何功能（基准）
2. **Offload** - 启用 KV Cache CPU Offloading（10GB）
3. **LMCache** - 启用 LMCache 前缀缓存

## 测试指标

- **completed**: 成功完成的请求数
- **RTF**: Real-Time Factor（处理时间/音频时长，越小越好）
- **TTFT**: Time To First Token/Packet（首包延迟，毫秒，越小越好）
- **E2EL**: End-to-End Latency（端到端延迟，毫秒，越小越好）
- **throughput**: 每秒处理的请求数（越大越好）

## 快速开始

### 运行测试

```bash
cd /home/yiqix/vllm-omni/benchmarks/qwen3-omni

# 使用默认配置运行（c=1, 4, 10，每个10个prompts）
bash benchmark_three_modes.sh

# 或自定义
NUM_PROMPTS=20 bash benchmark_three_modes.sh
```

### 环境变量

- `GPU_DEVICE`: GPU 索引（默认: "0,1"）
- `MODEL`: 模型路径（默认: "Qwen/Qwen3-Omni-MoE-30B-A3B-Instruct"）
- `NUM_PROMPTS`: 每个并发级别的 prompts 数量（默认: 10）

## 输出格式

测试会生成如下格式的对比表：

```
Metric              Baseline c=1    Offload c=1     LMCache c=1     Baseline c=4    Offload c=4     LMCache c=4     Baseline c=10   Offload c=10    LMCache c=10
=========================================================================================================================================
completed           10/10           10/10           10/10           40/40           40/40           40/40           100/100         100/100         100/100
RTF                 0.299           0.283           0.285           0.890           0.780           0.795           1.234           1.156           1.178
TTFT (ms)           89.6            71.4            73.2            1363            2047            1856            2845            2456            2589
E2EL (ms)           8027            7659            7701            26394           23163           24102           45678           42345           43567
throughput (r/s)    0.125           0.131           0.130           0.146           0.166           0.160           0.198           0.215           0.208
```

还包括：
- **Delta Analysis**: 与 Baseline 的百分比对比
- **Detailed Statistics**: 各指标的 p50/p90/p95/p99 百分位数

## 输出文件

所有结果保存在 `results/` 目录：

- `bench_baseline_TIMESTAMP.json` - Baseline 测试数据
- `bench_offload_TIMESTAMP.json` - Offload 测试数据
- `bench_lmcache_TIMESTAMP.json` - LMCache 测试数据
- `three_mode_comparison_TIMESTAMP.txt` - 格式化对比表
- `server_baseline_TIMESTAMP.log` - Baseline 服务器日志
- `server_offload_TIMESTAMP.log` - Offload 服务器日志
- `server_lmcache_TIMESTAMP.log` - LMCache 服务器日志

## 测试流程

1. **Phase 1: Baseline**
   - 启动服务器（无 KV offload，无 LMCache）
   - 运行 c=1, 4, 10 的测试
   - 停止服务器

2. **Phase 2: Offload**
   - 启动服务器（启用 KV offload）
   - 运行 c=1, 4, 10 的测试
   - 停止服务器

3. **Phase 3: LMCache**
   - 启动服务器（启用 LMCache）
   - 运行 c=1, 4, 10 的测试
   - 停止服务器

4. **Phase 4: 生成对比**
   - 生成三模式对比表
   - 计算 delta 百分比
   - 输出详细统计

## 配置文件

- `qwen3_omni_baseline.yaml` - Baseline 配置
- `qwen3_omni_kv_offload_on.yaml` - Offload 配置
- `qwen3_omni_lmcache.yaml` - LMCache 配置

## 手动对比

如果想手动对比已有的结果：

```bash
python compare_three_modes.py \
    --baseline results/bench_baseline_20260320_123456.json \
    --offload results/bench_offload_20260320_123456.json \
    --lmcache results/bench_lmcache_20260320_123456.json \
    --output results/custom_comparison.txt
```

## 故障排除

### 服务器启动失败
- 检查 GPU 显存是否充足
- 查看服务器日志：`results/server_*.log`
- 确认模型已下载

### OOM 错误
- 减少 `NUM_PROMPTS`
- 调整配置文件中的 `gpu_memory_utilization`
- 增加 CPU offload 内存大小

### LMCache 相关错误
- 确认已安装 lmcache：`pip install lmcache`
- 检查 LMCache 配置是否正确

## 示例输出

```bash
$ bash benchmark_three_modes.sh

============================================================
 Qwen3-Omni-30B Three-Mode Benchmark
============================================================
 GPU:         0,1
 Model:       Qwen/Qwen3-Omni-MoE-30B-A3B-Instruct
 Prompts:     10
 Concurrency: 1 4 10
 Results:     results/
============================================================

============================================================
 Testing Mode: baseline
============================================================
Starting baseline server on port 8000...
...

============================================================
 Testing Mode: offload
============================================================
Starting offload server on port 8001...
...

============================================================
 Testing Mode: lmcache
============================================================
Starting lmcache server on port 8002...
...

============================================================
 Generating comparison table...
============================================================

Metric              Baseline c=1    Offload c=1     LMCache c=1     ...
======================================================================
completed           10/10           10/10           10/10
RTF                 0.299           0.283           0.285
TTFT (ms)           89.6            71.4            73.2
E2EL (ms)           8027            7659            7701
throughput (r/s)    0.125           0.131           0.130

DELTA ANALYSIS (vs Baseline)
================================================================================
Concurrency: 1
--------------------------------------------------------------------------------
RTF:          Baseline=0.299  Offload=0.283 (-5.4%)  LMCache=0.285 (-4.7%)
TTFT (ms):    Baseline=89.6   Offload=71.4 (-20.3%)  LMCache=73.2 (-18.3%)
E2EL (ms):    Baseline=8027   Offload=7659 (-4.6%)   LMCache=7701 (-4.1%)
Throughput:   Baseline=0.125  Offload=0.131 (+4.8%)  LMCache=0.130 (+4.0%)
...
```

## 相关文档

- [PR #1330: KV Cache CPU Offloading](https://github.com/vllm-project/vllm-omni/pull/1330)
- [Qwen3-Omni TTS Performance Blog](https://github.com/Shirley125/vllm-omni/blob/cbf8e5033d6d58c12dd3d33748cf8cc80ad217da/docs/design/qwen3_omni_tts_performance_blog.md)
