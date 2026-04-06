"""
E2E test: verify MultiConnector (OffloadingConnector + LMCacheConnector) works
with a real model via the omni_kv_config YAML surface.

Usage:
    # Default: Qwen2.5-Omni-3B, offload only
    python tests/engine/test_kv_offload_with_model.py

    # Offload + LMCache (requires lmcache installed)
    python tests/engine/test_kv_offload_with_model.py --mode offload+lmcache

    # Custom model
    python tests/engine/test_kv_offload_with_model.py --model Qwen/Qwen2.5-Omni-3B
"""

import argparse
import logging
import tempfile

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("multi_connector_test")

MODES = {
    "offload": {
        "kv_store_config": {
            "enable_offload": True,
            "max_cpu_memory_gb": 2.0,
        }
    },
    "lmcache": {
        "kv_store_config": {
            "lmcache_config": {
                "config_file": "",  # uses default LMCache config
            }
        }
    },
    "offload+lmcache": {
        "kv_store_config": {
            "enable_offload": True,
            "max_cpu_memory_gb": 2.0,
            "lmcache_config": {
                "config_file": "",
            },
        }
    },
}


def build_stage_config(model: str, mode: str) -> str:
    """Build a temp stage config YAML with the specified KV config mode."""
    omni_kv_config = MODES[mode]

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
                    "omni_kv_config": omni_kv_config,
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

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(config, tmp, default_flow_style=False)
    tmp.flush()
    logger.info("Stage config written to %s", tmp.name)
    logger.info("Mode: %s | omni_kv_config: %s", mode, omni_kv_config)
    return tmp.name


def run_test(model: str, mode: str, num_prompts: int):
    from vllm_omni.entrypoints.omni import Omni

    config_path = build_stage_config(model, mode)

    logger.info("=" * 60)
    logger.info("Model      : %s", model)
    logger.info("Mode       : %s", mode)
    logger.info("Num prompts: %d", num_prompts)
    logger.info("=" * 60)

    logger.info("Starting Omni engine...")
    omni = Omni(
        model=model,
        stage_configs_path=config_path,
        stage_init_timeout=300,
        batch_timeout=5,
        init_timeout=300,
        log_stats=True,
    )

    prompts = [
        {"prompt": f"<|im_start|>user\nCount from 1 to {i + 5}.<|im_end|>\n<|im_start|>assistant\n"}
        for i in range(num_prompts)
    ]
    sampling_params_list = [st.default_sampling_params for st in omni.stage_list]

    logger.info("Sending %d prompts...", num_prompts)
    outputs = omni.generate(prompts, sampling_params_list)

    logger.info("-" * 60)
    logger.info("RESULTS")
    logger.info("-" * 60)
    ok = False
    for out in outputs:
        if out.final_output_type == "text" and out.request_output:
            for ro in out.request_output:
                text = ro.outputs[0].text if ro.outputs else "(empty)"
                logger.info("  [%s] %s", ro.request_id, text[:120])
                if text.strip():
                    ok = True

    if ok:
        logger.info("PASS: Got non-empty text output")
    else:
        logger.error("FAIL: No text output generated")

    omni.close()
    logger.info("Done!")
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test MultiConnector with a real model")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Omni-3B", help="Model name or path")
    parser.add_argument("--mode", default="offload", choices=list(MODES.keys()), help="KV config mode")
    parser.add_argument("--num-prompts", type=int, default=3, help="Number of prompts to send")
    args = parser.parse_args()

    success = run_test(args.model, args.mode, args.num_prompts)
    exit(0 if success else 1)
