# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
KV Cache CPU Offload Benchmark for Qwen3-Omni-30B-A3B-Instruct.

Measures the benefit of OffloadingConnector + LMCacheConnectorV1 under GPU
memory pressure using a two-phase revisit workload:

  Phase 1 (unique requests): fills and evicts GPU KV blocks, populating the
    CPU offload store.
  Phase 2 (revisit requests): replays the same prompts; LMCache serves KV
    from CPU, skipping thinker prefill recomputation.

Three arms are compared:

  baseline        – no prefix caching, no connector
  prefix_only     – GPU prefix caching only (degrades to baseline under eviction)
  offload_lmcache – OffloadingConnector + LMCacheConnectorV1 (survives eviction)

The primary metrics are TTFT and E2EL for revisit requests (Phase 2), where
the LMCache benefit is most visible, plus overall audio-s/wall-s throughput.

Usage:
    # Full benchmark (thinker + E2E, all arms, c=1 and c=4):
    python bench_kv_offload.py

    # Quick smoke test (thinker only, baseline, 3 prompts):
    python bench_kv_offload.py --mode thinker --arms baseline \\
        --concurrency 1 --n-unique 3 --n-revisit 1 \\
        --target-tokens 1000 --max-tokens 16 --trials 1

    # E2E only, offload arm, c=4:
    python bench_kv_offload.py --mode e2e --arms offload_lmcache --concurrency 4
"""

from __future__ import annotations

import os

# LMCache uses Python's builtin hash() for token chunk keys.  Without a fixed
# seed the Worker and Scheduler sub-processes get different random seeds,
# causing store/lookup key mismatch and 0% cache hits.
os.environ.setdefault("PYTHONHASHSEED", "0")

import argparse
import datetime
import pickle
import signal
import subprocess as _sp
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Optional

import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
AUDIO_SAMPLE_RATE = 24_000  # code2wav outputs 24 kHz

SEEDS = [
    "quantum mechanics and wave-particle duality",
    "deep learning transformer architectures",
    "Renaissance art and the Medici family",
    "plate tectonics and earthquake dynamics",
    "Roman Empire economic trade routes",
    "CRISPR gene editing therapeutic applications",
    "supply chain optimization in semiconductor manufacturing",
    "climate feedback loops and arctic amplification",
    "Byzantine medieval manuscript illumination",
    "exoplanet atmospheric spectroscopy methods",
    "Byzantine fault tolerance in distributed systems",
    "Ottoman Empire administrative structure",
    "photosynthesis biochemical light reactions",
    "stochastic calculus and Black-Scholes model",
    "Medieval Islamic golden age mathematics",
    "French Revolution socioeconomic causes",
    "volcanic eruption pyroclastic flow dynamics",
    "nuclear fusion plasma confinement techniques",
    "ancient Mesopotamian irrigation agriculture",
    "coral reef bleaching thermal stress biology",
]

FILLER_PARA = (
    "The rapid advancement of modern technology has fundamentally transformed "
    "how societies organize information, communicate across vast distances, and "
    "make decisions under uncertainty. From the earliest mechanical calculators "
    "to contemporary neural networks processing billions of parameters, each era "
    "of innovation has introduced new paradigms that challenge existing assumptions. "
    "Engineers and scientists work collaboratively across disciplines, drawing on "
    "mathematics, physics, biology, and social sciences to develop systems that "
    "are both powerful and robust. The interplay between theory and practice drives "
    "progress: theoretical breakthroughs enable practical applications, while real-world "
    "constraints motivate new mathematical frameworks. Understanding these dynamics is "
    "essential for anyone seeking to contribute meaningfully to technological progress."
)

# ---------------------------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------------------------

def _approx_tokens(text: str) -> int:
    import re
    return int(len(re.findall(r"\S+", text)) * 1.35)


def build_prompt(seed: str, target_tokens: int) -> str:
    """Build a chat-template prompt padded to ~target_tokens."""
    question = (
        f"Based on the context about {seed}, provide a comprehensive technical "
        "analysis covering fundamental principles, historical development, current "
        "state of the art, key challenges, and future research directions."
    )
    header = f"Topic: {seed}\n\nContext:\n"
    footer = f"\n\nQuestion: {question}"
    fixed = _approx_tokens(header + footer)
    reps = max(1, (target_tokens - fixed) // _approx_tokens(FILLER_PARA))
    body = "\n\n".join(f"[Section {i + 1}] {FILLER_PARA}" for i in range(reps))
    user_content = header + body + footer
    # Qwen3-Omni chat template (required for thinker2talker stage input processor)
    return (
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def build_workload(n_unique: int, n_revisit: int, target_tokens: int) -> list[dict]:
    unique = [
        {"id": f"unique_{i:03d}", "prompt": build_prompt(SEEDS[i], target_tokens), "is_revisit": False}
        for i in range(n_unique)
    ]
    revisit = [
        {"id": f"revisit_{j:03d}", "prompt": unique[j]["prompt"], "is_revisit": True}
        for j in range(n_revisit)
    ]
    return unique + revisit

# ---------------------------------------------------------------------------
# Stage config builders
# ---------------------------------------------------------------------------

def _thinker_engine_args(arm: str, concurrency: int, max_tokens: int, mode: str) -> dict:
    args: dict = {
        "model_stage": "thinker",
        "model_arch": "Qwen3OmniMoeForConditionalGeneration",
        "worker_type": "ar",
        "scheduler_cls": "vllm_omni.core.sched.omni_ar_scheduler.OmniARScheduler",
        "gpu_memory_utilization": 0.9,
        "enforce_eager": False,
        "trust_remote_code": True,
        "enable_prefix_caching": arm != "baseline",
        "max_num_batched_tokens": 32768,
        "hf_config_name": "thinker_config",
        "tensor_parallel_size": 1,
        "distributed_executor_backend": "mp",
        "max_num_seqs": concurrency,
        "engine_output_type": "text" if mode == "thinker" else "latent",
    }
    if arm == "offload_only":
        args["omni_kv_config"] = {
            "kv_store_config": {
                "enable_offload": True,
                "max_cpu_memory_gb": 20.0,
            }
        }
    elif arm == "lmcache_only":
        args["omni_kv_config"] = {
            "kv_store_config": {
                "lmcache_config": {
                    "chunk_size": 256,
                    "local_cpu": True,
                    "max_local_cpu_size": 30,
                },
            }
        }
    elif arm == "offload_lmcache":
        args["omni_kv_config"] = {
            "kv_store_config": {
                "enable_offload": True,
                "max_cpu_memory_gb": 20.0,
                "lmcache_config": {
                    "chunk_size": 256,
                    "local_cpu": True,
                    "max_local_cpu_size": 30,
                },
            }
        }
    return args


def make_stage_config(arm: str, mode: str, concurrency: int, max_tokens: int) -> dict:
    thinker_ea = _thinker_engine_args(arm, concurrency, max_tokens, mode)
    thinker_stage = {
        "stage_id": 0,
        "stage_type": "llm",
        "runtime": {"devices": "0"},
        "engine_args": thinker_ea,
        "is_comprehension": True,
        "final_output": True,
        "final_output_type": "text",
        "default_sampling_params": {
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "seed": 42,
            "detokenize": True,
            "repetition_penalty": 1.05,
        },
    }
    if mode == "thinker":
        return {"async_chunk": False, "stage_args": [thinker_stage]}
    if mode == "e2e_async":
        return {"async_chunk": True, "stage_args": [thinker_stage, talker_stage, code2wav_stage]}

    talker_stage = {
        "stage_id": 1,
        "stage_type": "llm",
        "runtime": {"devices": "1"},
        "engine_args": {
            "model_stage": "talker",
            "model_arch": "Qwen3OmniMoeForConditionalGeneration",
            "worker_type": "ar",
            "scheduler_cls": "vllm_omni.core.sched.omni_ar_scheduler.OmniARScheduler",
            "gpu_memory_utilization": 0.6,
            "enforce_eager": False,
            "trust_remote_code": True,
            "engine_output_type": "latent",
            "enable_prefix_caching": False,
            "max_num_batched_tokens": 32768,
            "distributed_executor_backend": "mp",
            "hf_config_name": "talker_config",
            "max_num_seqs": 64,
        },
        "engine_input_source": [0],
        "custom_process_input_func": (
            "vllm_omni.model_executor.stage_input_processors.qwen3_omni.thinker2talker"
        ),
        "default_sampling_params": {
            "temperature": 0.9,
            "top_k": 50,
            "max_tokens": 4096,
            "seed": 42,
            "detokenize": False,
            "repetition_penalty": 1.05,
            "stop_token_ids": [2150],
        },
    }
    code2wav_stage = {
        "stage_id": 2,
        "stage_type": "llm",
        "runtime": {"devices": "1"},
        "engine_args": {
            "model_stage": "code2wav",
            "model_arch": "Qwen3OmniMoeForConditionalGeneration",
            "worker_type": "generation",
            "scheduler_cls": (
                "vllm_omni.core.sched.omni_generation_scheduler.OmniGenerationScheduler"
            ),
            "enforce_eager": True,
            "trust_remote_code": True,
            "async_scheduling": False,
            "enable_prefix_caching": False,
            "engine_output_type": "audio",
            "gpu_memory_utilization": 0.1,
            "distributed_executor_backend": "mp",
            "max_num_batched_tokens": 100000,
            "hf_config_name": "thinker_config",
            "max_num_seqs": 64,
        },
        "engine_input_source": [1],
        "custom_process_input_func": (
            "vllm_omni.model_executor.stage_input_processors.qwen3_omni.talker2code2wav"
        ),
        "final_output": True,
        "final_output_type": "audio",
        "default_sampling_params": {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "max_tokens": 65536,
            "seed": 42,
            "detokenize": True,
            "repetition_penalty": 1.1,
        },
    }
    return {"async_chunk": False, "stage_args": [thinker_stage, talker_stage, code2wav_stage]}

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ReqResult:
    req_id: str
    is_revisit: bool
    submit_ts: float
    first_text_ts: Optional[float]   # TTFT: first thinker text token
    first_audio_ts: Optional[float]  # TTFP: first audio packet
    finish_ts: float
    prompt_tokens: int
    cached_tokens: int
    audio_frames: int = 0

    @property
    def e2el_ms(self) -> float:
        return (self.finish_ts - self.submit_ts) * 1000.0

    @property
    def ttft_ms(self) -> Optional[float]:
        if self.first_text_ts is None:
            return None
        return (self.first_text_ts - self.submit_ts) * 1000.0

    @property
    def ttfp_ms(self) -> Optional[float]:
        if self.first_audio_ts is None:
            return None
        return (self.first_audio_ts - self.submit_ts) * 1000.0

    @property
    def audio_duration_s(self) -> float:
        return self.audio_frames / AUDIO_SAMPLE_RATE

    @property
    def rtf(self) -> Optional[float]:
        if self.audio_frames == 0:
            return None
        return self.e2el_ms / 1000.0 / self.audio_duration_s


@dataclass
class BenchmarkConfig:
    arm: str
    mode: str
    concurrency: int
    n_unique: int
    n_revisit: int
    target_tokens: int
    max_tokens: int
    model: str
    stage_init_timeout: int = 300


@dataclass
class BenchmarkResult:
    cfg: BenchmarkConfig
    req_results: list[ReqResult]
    wall_start: float
    wall_end: float

    @property
    def _unique(self) -> list[ReqResult]:
        return [r for r in self.req_results if not r.is_revisit]

    @property
    def _revisit(self) -> list[ReqResult]:
        return [r for r in self.req_results if r.is_revisit]

    @staticmethod
    def _mean(vals: list) -> Optional[float]:
        return sum(vals) / len(vals) if vals else None

    @property
    def throughput_rps(self) -> float:
        elapsed = self.wall_end - self.wall_start
        return len(self.req_results) / elapsed if elapsed > 0 else 0.0

    @property
    def throughput_audio_s(self) -> Optional[float]:
        total = sum(r.audio_duration_s for r in self.req_results)
        elapsed = self.wall_end - self.wall_start
        return total / elapsed if elapsed > 0 and total > 0 else None

    # --- unique (Phase 1) ---
    @property
    def unique_ttft_ms(self) -> Optional[float]:
        return self._mean([r.ttft_ms for r in self._unique if r.ttft_ms is not None])

    @property
    def unique_e2el_ms(self) -> Optional[float]:
        return self._mean([r.e2el_ms for r in self._unique])

    # --- revisit (Phase 2) ---
    @property
    def revisit_ttft_ms(self) -> Optional[float]:
        return self._mean([r.ttft_ms for r in self._revisit if r.ttft_ms is not None])

    @property
    def revisit_ttfp_ms(self) -> Optional[float]:
        return self._mean([r.ttfp_ms for r in self._revisit if r.ttfp_ms is not None])

    @property
    def revisit_e2el_ms(self) -> Optional[float]:
        return self._mean([r.e2el_ms for r in self._revisit])

    @property
    def revisit_rtf(self) -> Optional[float]:
        return self._mean([r.rtf for r in self._revisit if r.rtf is not None])

    @property
    def kv_hit_pct(self) -> float:
        """Cache hit% on revisit requests (where hits are expected)."""
        rs = self._revisit
        total_p = sum(r.prompt_tokens for r in rs)
        total_c = sum(r.cached_tokens for r in rs)
        return 100.0 * total_c / total_p if total_p > 0 else 0.0

# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def _collect_outputs(gen, submit_ts: float, mode: str) -> tuple:
    first_text: dict[str, float] = {}
    first_audio: dict[str, float] = {}
    finish: dict[str, float] = {}
    prompt_toks: dict[str, int] = {}
    cached_toks: dict[str, int] = {}
    audio_frames: dict[str, int] = {}

    for stage_out in gen:
        now = time.monotonic()
        rid = stage_out.request_id
        ft = stage_out.final_output_type

        if ft == "text":
            first_text.setdefault(rid, now)
            finish[rid] = now
            pt = stage_out.prompt_token_ids
            prompt_toks[rid] = len(pt) if pt is not None else 0
            cached_toks[rid] = stage_out.num_cached_tokens or 0
            print(
                f"  [{rid}] text  prompt={prompt_toks[rid]}  "
                f"cached={cached_toks[rid]}  "
                f"ttft={1000*(now - submit_ts):.0f}ms",
                flush=True,
            )

        if ft == "audio":
            first_audio.setdefault(rid, now)
            finish[rid] = now
            audio_out = None
            if stage_out.outputs:
                mm = getattr(stage_out.outputs[0], "multimodal_output", {}) or {}
                audio_out = mm.get("audio")
            if audio_out is not None and hasattr(audio_out, "shape"):
                audio_frames[rid] = int(audio_out.shape[-1])
            dur = audio_frames.get(rid, 0) / AUDIO_SAMPLE_RATE
            print(f"  [{rid}] audio  frames={audio_frames.get(rid, 0)}  dur={dur:.1f}s", flush=True)

    return first_text, first_audio, finish, prompt_toks, cached_toks, audio_frames


def _make_req_results(
    rids: list[str],
    is_revisit: bool,
    submit_ts: float,
    wall_end: float,
    first_text: dict,
    first_audio: dict,
    finish: dict,
    prompt_toks: dict,
    cached_toks: dict,
    audio_frames: dict,
) -> list[ReqResult]:
    return [
        ReqResult(
            req_id=rid,
            is_revisit=is_revisit,
            submit_ts=submit_ts,
            first_text_ts=first_text.get(rid),
            first_audio_ts=first_audio.get(rid),
            finish_ts=finish.get(rid, wall_end),
            prompt_tokens=prompt_toks.get(rid, 0),
            cached_tokens=cached_toks.get(rid, 0),
            audio_frames=audio_frames.get(rid, 0),
        )
        for rid in rids
    ]


def _run_benchmark_worker(cfg: BenchmarkConfig, result_path: str) -> None:
    from vllm import SamplingParams
    from vllm_omni.entrypoints.omni import Omni

    workload = build_workload(cfg.n_unique, cfg.n_revisit, cfg.target_tokens)
    unique_prompts  = [p for p in workload if not p["is_revisit"]]
    revisit_prompts = [p for p in workload if p["is_revisit"]]

    cfg_data = make_stage_config(cfg.arm, cfg.mode, cfg.concurrency, cfg.max_tokens)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False,
        prefix=f"kv_bench_{cfg.arm}_{cfg.mode}_c{cfg.concurrency}_",
    )
    yaml.dump(cfg_data, tmp, default_flow_style=False, sort_keys=False)
    tmp.flush()
    cfg_path = tmp.name

    print(f"\n{'='*65}", flush=True)
    print(f"  ARM={cfg.arm}  MODE={cfg.mode}  C={cfg.concurrency}", flush=True)
    print(f"  Phase 1: {len(unique_prompts)} unique  → stress GPU KV / populate offload", flush=True)
    print(f"  Phase 2: {len(revisit_prompts)} revisit → measure LMCache benefit", flush=True)
    print(f"{'='*65}", flush=True)

    num_stages = 1 if cfg.mode == "thinker" else 3
    sampling_params = [
        SamplingParams(temperature=0.0, max_tokens=cfg.max_tokens, seed=42,
                       detokenize=True, repetition_penalty=1.05),
        SamplingParams(temperature=0.9, top_k=50, max_tokens=4096, seed=42,
                       detokenize=False, repetition_penalty=1.05, stop_token_ids=[2150]),
        SamplingParams(temperature=0.0, top_p=1.0, top_k=-1, max_tokens=65536,
                       seed=42, detokenize=True, repetition_penalty=1.1),
    ][:num_stages]

    def _inputs(prompts):
        if cfg.mode in ("e2e", "e2e_async"):
            return [{"prompt": p["prompt"], "modalities": ["text", "audio"]} for p in prompts]
        return [{"prompt": p["prompt"]} for p in prompts]

    omni = Omni(model=cfg.model, stage_configs_path=cfg_path, log_stats=True,
                stage_init_timeout=cfg.stage_init_timeout)

    # In e2e_async mode, disable FINAL_ONLY so thinker streams per-token
    # outputs and we can measure real TTFT.
    if cfg.mode == "e2e_async":
        omni._set_final_only_for_llm_stages = lambda sp: list(sp)

    # Phase 1: unique
    print(f"\n--- Phase 1: {len(unique_prompts)} unique requests ---", flush=True)
    p1_ts = time.monotonic()
    p1 = _collect_outputs(omni.generate(_inputs(unique_prompts), sampling_params), p1_ts, cfg.mode)
    p1_end = time.monotonic()
    p1_text, p1_audio, p1_finish, p1_pt, p1_ct, p1_frames = p1
    print(f"  Phase 1 done in {p1_end - p1_ts:.1f}s", flush=True)

    # Phase 2: revisit
    print(f"\n--- Phase 2: {len(revisit_prompts)} revisit requests ---", flush=True)
    p2_ts = time.monotonic()
    p2 = _collect_outputs(omni.generate(_inputs(revisit_prompts), sampling_params), p2_ts, cfg.mode)
    p2_end = time.monotonic()
    p2_text, p2_audio, p2_finish, p2_pt, p2_ct, p2_frames = p2
    print(f"  Phase 2 done in {p2_end - p2_ts:.1f}s", flush=True)

    omni.close()
    os.unlink(cfg_path)

    unique_results  = _make_req_results(sorted(p1_finish), False, p1_ts, p1_end, *p1)
    revisit_results = _make_req_results(sorted(p2_finish), True,  p2_ts, p2_end, *p2)

    result = BenchmarkResult(
        cfg=cfg,
        req_results=unique_results + revisit_results,
        wall_start=p1_ts,
        wall_end=p2_end,
    )
    with open(result_path, "wb") as f:
        pickle.dump(result, f)


def _kill_gpu_orphans() -> None:
    """Kill VLLM worker processes that outlived the trial subprocess."""
    try:
        out = _sp.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        own_pid = os.getpid()
        for line in out.stdout.splitlines():
            try:
                pid = int(line.strip())
            except ValueError:
                continue
            if pid == own_pid:
                continue
            try:
                os.kill(pid, signal.SIGKILL)
                print(f"[bench] Killed orphaned GPU process {pid}", flush=True)
            except ProcessLookupError:
                pass
    except Exception as exc:
        print(f"[bench] GPU orphan cleanup failed: {exc}", flush=True)


def run_one_trial(cfg: BenchmarkConfig) -> BenchmarkResult:
    """Run one trial in an isolated subprocess to avoid CUDA context leaks."""
    import multiprocessing

    result_file = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False, prefix="kv_bench_")
    result_path = result_file.name
    result_file.close()

    ctx = multiprocessing.get_context("spawn")
    proc = ctx.Process(target=_run_benchmark_worker, args=(cfg, result_path))
    proc.start()
    proc.join()
    _kill_gpu_orphans()

    if os.path.exists(result_path) and os.path.getsize(result_path) > 0:
        try:
            with open(result_path, "rb") as f:
                result = pickle.load(f)
            os.unlink(result_path)
            if proc.exitcode != 0:
                print(
                    f"[bench] Subprocess exited {proc.exitcode} (cleanup crash); "
                    "results captured.", flush=True,
                )
            return result
        except Exception:
            pass

    if proc.exitcode != 0:
        raise RuntimeError(f"Trial subprocess failed with exit code {proc.exitcode}.")

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _f(val, fmt=".0f") -> str:
    """Format a metric value, returning '—' for None."""
    if val is None:
        return "—"
    return format(val, fmt)


def _thinker_table(rows: list[BenchmarkResult]) -> str:
    header = (
        "| Arm | c | Throughput (r/s) "
        "| TTFT unique (ms) | TTFT revisit (ms) "
        "| E2EL unique (ms) | E2EL revisit (ms) | KV hit% |"
    )
    sep = "|---|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"| {r.cfg.arm} | {r.cfg.concurrency} | {r.throughput_rps:.3f} "
            f"| {_f(r.unique_ttft_ms)} | {_f(r.revisit_ttft_ms)} "
            f"| {_f(r.unique_e2el_ms)} | {_f(r.revisit_e2el_ms)} "
            f"| {r.kv_hit_pct:.1f}% |"
        )
    return "\n".join(lines)


def _e2e_table(rows: list[BenchmarkResult]) -> str:
    header = (
        "| Arm | c "
        "| TTFT unique (ms) | TTFT revisit (ms) | TTFP revisit (ms) "
        "| E2EL unique (ms) | E2EL revisit (ms) | RTF revisit "
        "| Throughput (audio-s/wall-s) | KV hit% |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"| {r.cfg.arm} | {r.cfg.concurrency} "
            f"| {_f(r.unique_ttft_ms)} | {_f(r.revisit_ttft_ms)} | {_f(r.revisit_ttfp_ms)} "
            f"| {_f(r.unique_e2el_ms)} | {_f(r.revisit_e2el_ms)} | {_f(r.revisit_rtf, '.3f')} "
            f"| {_f(r.throughput_audio_s, '.2f')} | {r.kv_hit_pct:.1f}% |"
        )
    return "\n".join(lines)


def generate_report(
    thinker_results: list[BenchmarkResult],
    e2e_results: list[BenchmarkResult],
    args: argparse.Namespace,
) -> str:
    try:
        import vllm
        vllm_ver = vllm.__version__
    except Exception:
        vllm_ver = "unknown"
    try:
        import vllm_omni
        omni_ver = vllm_omni.__version__
    except Exception:
        omni_ver = "unknown"
    try:
        import torch
        torch_ver = torch.__version__
    except Exception:
        torch_ver = "unknown"

    n_total = args.n_unique + args.n_revisit
    revisit_pct = 100.0 * args.n_revisit / n_total

    report = f"""# KV Cache CPU Offload Benchmark

**{args.model} · {args.hw_desc}**
*Generated: {datetime.date.today().isoformat()}*

---

## Setup

| Component | Version |
|---|---|
| vLLM | {vllm_ver} |
| vllm-omni | {omni_ver} |
| PyTorch | {torch_ver} |

| Parameter | Value |
|---|---|
| n_unique | {args.n_unique} |
| n_revisit | {args.n_revisit} ({revisit_pct:.0f}% of total) |
| target_tokens | {args.target_tokens:,} |
| max_tokens (thinker) | {args.max_tokens} |
| gpu_memory_utilization | 0.9 |

**Arms:**

| Arm | prefix caching | KV connector | CPU store |
|---|---|---|---|
| `baseline` | off | none | — |
| `prefix_only` | on | none (GPU only) | — |
| `lmcache_only` | on | LMCacheConnectorV1 | 30 GB |

**Workload design:**
Phase 1 ({args.n_unique} unique × ~{args.target_tokens:,} tokens) fills and evicts GPU KV blocks,
populating the CPU offload store.
Phase 2 ({args.n_revisit} revisit) replays the same prompts; LMCache serves KV from CPU,
skipping thinker prefill recomputation. Metrics below are split by phase.

---
"""

    if thinker_results:
        report += f"""
## Thinker-Only Results

Single GPU (device 0), thinker stage only.

{_thinker_table(thinker_results)}

---
"""

    if e2e_results:
        report += f"""
## E2E Pipeline Results

```
[Thinker — GPU 0]  30B MoE, text comprehension
       ↓  hidden states + embeddings
[Talker  — GPU 1]  8.6B, TTS token generation
       ↓  codec tokens
[Code2Wav — GPU 1] 0.4B, waveform synthesis
       ↓  audio output (24 kHz PCM)
```

{_e2e_table(e2e_results)}

---
"""

    report += """
*TTFT = time to first thinker text token.*
*TTFP = time to first audio packet.*
*RTF = E2EL_s / audio_duration_s (lower is better; <1.0 means faster than realtime).*
*Throughput = total audio seconds produced / wall clock seconds (both phases).*
*KV hit% = cached_tokens / prompt_tokens for revisit requests only.*
"""
    return report

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KV Cache CPU Offload Benchmark for Qwen3-Omni",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--mode", choices=["thinker", "e2e", "e2e_async", "both"], default="both")
    parser.add_argument(
        "--arms", nargs="+", default=["baseline", "prefix_only", "lmcache_only"],
        choices=["baseline", "prefix_only", "offload_only", "lmcache_only", "offload_lmcache"],
    )
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 4],
                        help="max_num_seqs values to sweep")
    parser.add_argument("--n-unique", type=int, default=15,
                        help="Unique prompts in Phase 1 (stress GPU KV)")
    parser.add_argument("--n-revisit", type=int, default=10,
                        help="Revisit prompts in Phase 2 (measure LMCache benefit)")
    parser.add_argument("--target-tokens", type=int, default=8663,
                        help="Approximate prompt length per unique request")
    parser.add_argument("--max-tokens", type=int, default=32,
                        help="Max thinker output tokens (short decode to isolate TTFT)")
    parser.add_argument("--trials", type=int, default=2,
                        help="Trials per config; last trial is reported")
    parser.add_argument("--stage-init-timeout", type=int, default=300,
                        help="Seconds to wait for each stage to initialize (use 900+ for e2e/both)")
    parser.add_argument("--hw-desc", default="2× NVIDIA A100 80 GB PCIe")
    parser.add_argument("--report-out", default="benchmark_report.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    modes = []
    if args.mode in ("thinker", "both"):
        modes.append("thinker")
    if args.mode in ("e2e", "both"):
        modes.append("e2e")

    thinker_results: list[BenchmarkResult] = []
    e2e_results: list[BenchmarkResult] = []

    for mode in modes:
        for arm in args.arms:
            for c in args.concurrency:
                cfg = BenchmarkConfig(
                    arm=arm, mode=mode, concurrency=c,
                    n_unique=args.n_unique, n_revisit=args.n_revisit,
                    target_tokens=args.target_tokens, max_tokens=args.max_tokens,
                    model=args.model,
                    stage_init_timeout=args.stage_init_timeout,
                )
                last = None
                for trial in range(args.trials):
                    print(f"\n[Trial {trial + 1}/{args.trials}]  {arm}  {mode}  c={c}", flush=True)
                    last = run_one_trial(cfg)
                    print(
                        f"  → {last.throughput_rps:.3f} r/s  "
                        f"kv_hit(revisit)={last.kv_hit_pct:.1f}%  "
                        f"ttft_unique={_f(last.unique_ttft_ms)}ms  "
                        f"ttft_revisit={_f(last.revisit_ttft_ms)}ms",
                        flush=True,
                    )
                    if mode == "e2e" and last.revisit_rtf is not None:
                        print(
                            f"     rtf_revisit={last.revisit_rtf:.3f}  "
                            f"audio-s/wall-s={_f(last.throughput_audio_s, '.2f')}",
                            flush=True,
                        )

                if mode == "thinker":
                    thinker_results.append(last)
                else:
                    e2e_results.append(last)

    report = generate_report(thinker_results, e2e_results, args)

    print("\n" + "=" * 70)
    print(report)
    print("=" * 70)

    with open(args.report_out, "w") as f:
        f.write(report)
    print(f"\nReport saved to {args.report_out}", flush=True)


if __name__ == "__main__":
    main()
