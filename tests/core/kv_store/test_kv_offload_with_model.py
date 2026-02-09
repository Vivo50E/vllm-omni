"""
Local E2E test: verify KV offload works with a real model.

Usage:
    # Text-only (thinker stage only, single GPU)
    python tests/core/kv_store/test_kv_offload_with_model.py

    # Custom model
    python tests/core/kv_store/test_kv_offload_with_model.py --model Qwen/Qwen2.5-Omni-3B

    # Custom stage config
    python tests/core/kv_store/test_kv_offload_with_model.py --stage-config path/to/config.yaml
"""

import argparse
import logging
import tempfile
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("kv_offload_test")


def build_stage_config_with_offload(base_config_path: str | None, model: str) -> str:
    """Build a temp stage config YAML with KV offload enabled on thinker stage."""

    if base_config_path and Path(base_config_path).exists():
        with open(base_config_path) as f:
            config = yaml.safe_load(f)
    else:
        # Minimal single-stage (thinker only, text output) config
        config = {
            "stage_args": [
                {
                    "stage_id": 0,
                    "runtime": {"process": True, "devices": "0", "max_batch_size": 1},
                    "engine_args": {
                        "model_stage": "thinker",
                        "model_arch": "Qwen2_5OmniForConditionalGeneration",
                        "worker_type": "ar",
                        "scheduler_cls": "vllm_omni.core.sched.omni_ar_scheduler.OmniARScheduler",
                        "max_model_len": 512,
                        "max_num_batched_tokens": 512,
                        "max_num_seqs": 4,
                        "gpu_memory_utilization": 0.8,
                        "skip_mm_profiling": True,
                        "enforce_eager": True,
                        "trust_remote_code": True,
                        "engine_output_type": "text",
                        "enable_prefix_caching": False,
                    },
                    "is_comprehension": True,
                    "final_output": True,
                    "final_output_type": "text",
                    "default_sampling_params": {
                        "temperature": 0.0,
                        "max_tokens": 64,
                        "seed": 42,
                    },
                }
            ],
            "runtime": {
                "enabled": True,
                "defaults": {"window_size": -1, "max_inflight": 1},
                "edges": [],
            },
        }

    # Inject omni_kv_config into the first AR stage
    for stage in config["stage_args"]:
        ea = stage.get("engine_args", {})
        if ea.get("worker_type") == "ar":
            ea["omni_kv_config"] = {
                "kv_store_config": {
                    "enable_offload": True,
                    "backend_type": "cpu",
                    "max_cpu_memory_gb": 2.0,
                }
            }
            logger.info("Injected KV offload config into stage %s", stage["stage_id"])
            break

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(config, tmp, default_flow_style=False)
    tmp.flush()
    logger.info("Stage config written to %s", tmp.name)
    return tmp.name


def run_test(model: str, stage_config: str | None, num_prompts: int):
    from vllm_omni.entrypoints.omni import Omni

    # 1. Build config with offload
    config_path = build_stage_config_with_offload(stage_config, model)
    logger.info("=" * 60)
    logger.info("Model: %s", model)
    logger.info("Stage config: %s", config_path)
    logger.info("Num prompts: %d", num_prompts)
    logger.info("=" * 60)

    # 2. Start Omni
    logger.info("Starting Omni engine...")
    omni = Omni(
        model=model,
        stage_configs_path=config_path,
        stage_init_timeout=300,
        batch_timeout=5,
        init_timeout=300,
        log_stats=True,
    )

    # 3. Generate
    prompts = [
        {"prompt": f"<|im_start|>user\nCount from 1 to {i + 5}.<|im_end|>\n<|im_start|>assistant\n"}
        for i in range(num_prompts)
    ]
    sampling_params_list = [st.default_sampling_params for st in omni.stage_list]

    logger.info("Sending %d prompts...", num_prompts)
    outputs = omni.generate(prompts, sampling_params_list)

    # 4. Check outputs
    logger.info("-" * 60)
    logger.info("RESULTS")
    logger.info("-" * 60)
    for out in outputs:
        if out.final_output_type == "text" and out.request_output:
            for ro in out.request_output:
                text = ro.outputs[0].text if ro.outputs else "(empty)"
                logger.info("  [%s] %s", ro.request_id, text[:120])

    # 5. Check KV store backend stats (if accessible)
    # The backend lives inside the scheduler subprocess, so we log from there.
    # Look for log lines like "[Omni] KV store backend enabled" and
    # "[Omni] Offloaded KV for" / "[Omni] Prefetched KV for" in the output.
    logger.info("-" * 60)
    logger.info(
        "Check logs above for:\n"
        "  - '[Omni] KV store backend enabled'  (backend was created)\n"
        "  - '[Omni] Offloaded KV for'          (offload happened)\n"
        "  - '[Omni] Prefetched KV for'         (prefetch happened)\n"
        "  - '[Omni] Stored KV for'             (LMCache store)\n"
        "  - '[Omni] Loaded KV for'             (LMCache load)\n"
        "If no offload/prefetch lines appear, it means GPU memory was\n"
        "sufficient for all requests (no pressure). Try:\n"
        "  - Lowering gpu_memory_utilization in config\n"
        "  - Increasing --num-prompts\n"
        "  - Using longer prompts"
    )
    logger.info("-" * 60)

    omni.close()
    logger.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test KV offload with a real model")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Omni-3B", help="Model name or path")
    parser.add_argument("--stage-config", default=None, help="Base stage config YAML (optional)")
    parser.add_argument("--num-prompts", type=int, default=3, help="Number of prompts to send")
    args = parser.parse_args()

    run_test(args.model, args.stage_config, args.num_prompts)
