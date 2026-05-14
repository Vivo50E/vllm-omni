# KV Cache CPU Offloading Benchmark — Qwen3-Omni-30B E2E Pipeline

**Platform:** 2x NVIDIA A100 80 GB PCIe
**Model:** `Qwen/Qwen3-Omni-30B-A3B-Instruct`
**Stack:** vLLM 0.19.0 · vLLM-Omni · PyTorch 2.10.0+cu128
**Date:** 2026-04-23

---

## Setup

### Pipeline stages

```
[Thinker — GPU 0]  Qwen3-Omni 30B MoE, text comprehension  (gpu_memory_utilization=0.9)
       ↓  hidden states + token embeddings
[Talker  — GPU 1]  Qwen3-Omni 8.6B AR, TTS token generation (gpu_memory_utilization=0.6)
       ↓  codec tokens
[Code2Wav — GPU 1] Qwen3-Omni 0.4B, waveform synthesis     (gpu_memory_utilization=0.1)
       ↓
Audio output (24 kHz PCM)
```

### Benchmark arms

| Arm | Prefix caching | KV connector | CPU store |
|---|---|---|---|
| `baseline` | off | none | — |
| `prefix_only` | on (GPU only) | none | — |
| `lmcache_only` | on | LMCacheConnectorV1 | 30 GB |

**LMCache config** (`lmcache_only` arm):
```yaml
chunk_size: 256
local_cpu: true
max_local_cpu_size: 30  # GB
```

### Benchmark method

`vllm serve` + `vllm bench serve` with `--backend openai-chat-omni` (streaming API).
Async chunk enabled for all arms. Each arm: start server, run benchmark, stop server.

### Workload

- `--dataset-name random-mm`
- `--random-input-len 20000` (~20K tokens per request)
- `--random-prefix-len 10000` (50% shared prefix)
- `--random-output-len 100`
- `--num-prompts 100`
- `--max-concurrency 10`
- `--extra-body '{"modalities": ["text", "audio"]}'`

GPU KV capacity: ~98K tokens. With concurrency 10, up to 200K tokens active simultaneously — forces GPU prefix cache eviction.

---

## Results

### E2E Pipeline (20K input tokens, 10K shared prefix, async chunk)

| Metric | baseline | prefix_only | lmcache_only |
|---|---|---|---|
| **Mean TTFT (ms)** | 49,945 | 16,531 (-67%) | **15,317 (-69%)** |
| **Mean TTFP (ms)** | 66,405 | 34,607 (-48%) | **33,073 (-50%)** |
| **Mean E2EL (ms)** | 91,924 | 119,782 | 118,778 |
| **Mean RTF** | 1.89 | 1.16 (-39%) | **1.08 (-43%)** |
| **Median RTF** | 1.41 | 0.89 | **0.89** |
| **P99 RTF** | 13.02 | 5.52 (-58%) | **3.07 (-76%)** |
| **Audio throughput (s/s)** | 6.74 | 11.18 (+66%) | **11.14 (+65%)** |
| Mean TPOT (ms) | — | — | — |
| Request throughput (r/s) | — | — | — |

> **TTFT** = time to first thinker text token (real streaming TTFT via OpenAI API)
> **TTFP** = time to first audio packet
> **E2EL** = end-to-end latency from request submission to final audio output
> **RTF** = E2EL (s) / audio duration (s) — lower is better; RTF < 1 = realtime
> **Audio throughput** = total audio seconds produced / wall clock seconds

### Thinker-Only (8.6K input tokens, offline benchmark)

| Arm | c | Throughput (r/s) | TTFT unique (ms) | TTFT revisit (ms) | KV hit% |
|---|---|---|---|---|---|
| `baseline` | 1 | 1.007 | 14,949 | 9,860 | 0.0% |
| `baseline` | 4 | 1.343 | 11,221 | 7,388 | 0.0% |
| `prefix_only` | 1 | 0.904 | 16,853 | 10,800 | 0.0% |
| `prefix_only` | 4 | 1.145 | 13,379 | 8,447 | 0.0% |
| `lmcache_only` | 1 | 1.033 | 17,421 | **6,781** | **97.2%** |
| `lmcache_only` | 4 | 1.333 | 14,446 | **4,309** | **97.2%** |

**Thinker revisit TTFT improvement:**
- c=1: 9,860 ms -> **6,781 ms** (**31% faster**)
- c=4: 7,388 ms -> **4,309 ms** (**42% faster**)

Thinker-only test uses two-phase design: 15 unique prompts (~8,663 tokens each, total ~130K > 98K GPU KV capacity) fill and evict GPU KV blocks, then 10 revisit prompts measure LMCache benefit. `max_tokens = 32`.

---

## Key findings

### LMCache outperforms GPU prefix cache under memory pressure

At 20K input tokens with concurrency 10, GPU KV cache (~98K tokens) is far exceeded. LMCache's CPU store (30 GB) retains evicted KV chunks and serves them on revisit:

- **TTFT**: lmcache -69% vs baseline, -7% vs prefix_only
- **TTFP**: lmcache -50% vs baseline, -4% vs prefix_only
- **P99 RTF**: lmcache **3.07** vs prefix_only 5.52 (**-44% long-tail improvement**)

The P99 RTF improvement is particularly significant — LMCache reduces worst-case latency by preventing full prefill recomputation when GPU prefix cache entries are evicted under high memory pressure.

### GPU prefix cache is effective but bounded

`prefix_only` already delivers strong improvements (TTFT -67%, RTF -39%) via GPU-level KV block reuse. However, under heavy memory pressure, GPU eviction degrades tail latency (P99 RTF = 5.52). LMCache's CPU second-level cache rescues these cases.

### Audio throughput scales with prefix caching

Both `prefix_only` and `lmcache_only` achieve ~66% higher audio throughput (11.1 vs 6.7 audio-s/wall-s). By skipping prefill recomputation, the thinker frees GPU cycles for more concurrent audio generation.

### Hidden-state cache for talker correctness

`OmniOffloadHiddenStateCache` accumulates thinker hidden states per-request across decode steps. On KV restore, the thinker only runs on new tokens, so `thinker_hidden_states` is partial. The cache merges the cached prefix with the partial to reconstruct the full-sequence representation that the talker requires.

---

## Reproduction

```bash
cd vllm-omni

# 1. Start server (for each arm, modify stage config yaml accordingly)
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni \
  --port 46354 \
  --stage-configs-path ./stage_config.yaml

# 2. Run benchmark
vllm bench serve --omni \
  --dataset-name random-mm \
  --port 46354 \
  --max-concurrency 10 \
  --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --endpoint /v1/chat/completions \
  --backend openai-chat-omni \
  --num-prompts 100 \
  --random-input-len 20000 \
  --random-output-len 100 \
  --random-prefix-len 10000 \
  --ignore-eos \
  --percentile-metrics ttft,tpot,itl,e2el,audio_ttfp,audio_rtf \
  --extra-body '{"modalities": ["text", "audio"]}'

# Or use the automated script:
bash benchmarks/kv_offload/bench_serving.sh 'baseline prefix_only lmcache_only'
```

Results are written to `benchmarks/kv_offload/results/`.
