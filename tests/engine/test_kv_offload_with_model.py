"""
E2E test: verify MultiConnector (OffloadingConnector + LMCacheConnector) works
with a real model via the omni_kv_config YAML surface.
"""

import tempfile

import pytest
import yaml

pytestmark = [pytest.mark.advanced_model, pytest.mark.omni, pytest.mark.cuda]

DEFAULT_MODEL = "Qwen/Qwen2.5-Omni-3B"

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

LMCACHE_MODES = {"lmcache", "offload+lmcache"}


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
    return tmp.name


def _run(model: str, mode: str, num_prompts: int = 3) -> bool:
    from vllm_omni.entrypoints.omni import Omni

    config_path = build_stage_config(model, mode)
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
    sampling_params_list = omni.default_sampling_params_list
    outputs = omni.generate(prompts, sampling_params_list)
    omni.close()

    return any(
        out.request_output and out.request_output.outputs and out.request_output.outputs[0].text.strip()
        for out in outputs
        if out.final_output_type == "text"
    )


@pytest.mark.parametrize("mode", list(MODES.keys()))
def test_kv_offload_modes(mode):
    if mode in LMCACHE_MODES:
        pytest.importorskip("lmcache", reason="lmcache not installed")
    assert _run(DEFAULT_MODEL, mode), f"No text output for mode={mode}"
