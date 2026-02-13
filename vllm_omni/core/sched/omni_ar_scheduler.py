from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from time import time
from typing import Any

from vllm.compilation.cuda_graph import CUDAGraphStat
from vllm.distributed.kv_events import KVEventBatch
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler as VLLMScheduler
from vllm.v1.core.sched.utils import remove_all
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs
from vllm.v1.metrics.perf import PerfStats
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.spec_decode.metrics import SpecDecodingStats

from vllm_omni.core.kv_store import OmniKvStoreFactory
from vllm_omni.core.sched.output import OmniSchedulerOutput
from vllm_omni.distributed.omni_connectors.transfer_adapter.chunk_transfer_adapter import (
    OmniChunkTransferAdapter,
)

logger = init_logger(__name__)


@dataclass
class KVCacheTransferData:
    request_id: str
    layer_blocks: dict[str, Any]
    block_ids: list[int]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OmniARScheduler(VLLMScheduler):
    """
    OmniARScheduler: Scheduler for vLLM-Omni multimodal processing.

    This scheduler extends vLLM's scheduler to support multimodal and
    non-autoregressive processing with additional fields and methods
    specific to vLLM-Omni.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Track requests that need KV cache transfer when finished
        # Value is {"seq_len": int, "block_ids": list[int]}
        self.requests_needing_kv_transfer: dict[str, dict[str, Any]] = {}

        # Track requests waiting for KV transfer (blocks not freed yet)
        self.waiting_for_transfer_free: set[str] = set()

        # Track ACTIVE transfers (submitted to runner but not yet acked via kv_extracted_req_ids)
        self.active_kv_transfers: set[str] = set()

        # [Omni] Pre-parse KV transfer criteria
        self.kv_transfer_criteria = self._get_kv_transfer_criteria()

        # Track requests that have already triggered prefill transfer to avoid duplicates
        self.transfer_triggered_requests: set[str] = set()
        model_config = self.vllm_config.model_config
        self.chunk_transfer_adapter = None
        if getattr(model_config, "async_chunk", False):
            self.chunk_transfer_adapter = OmniChunkTransferAdapter(self.vllm_config)

        # [Omni] KV store backend for CPU offloading
        self.kv_store_backend = self._create_kv_store_backend()
        self.offloaded_requests: set[str] = set()  # KV on CPU, request in waiting queue
        self.offloaded_metadata: dict[str, dict] = {}  # {req_id: {"seq_len": int}}

        # [Omni] Two-phase offload protocol state
        self.offload_pending_extract: set[str] = set()  # Phase 1: waiting for runner extraction
        self.offload_pending_free: set[str] = set()  # Phase 2: extraction done, pending block free
        self.offload_usage_threshold: float = self._get_offload_config("offload_usage_threshold", 0.85)
        self.max_offloads_per_step: int = self._get_offload_config("max_offloads_per_step", 2)

    def _get_offload_config(self, key: str, default: Any = None) -> Any:
        omni_kv_config = getattr(self.vllm_config.model_config, "omni_kv_config", None)
        if not omni_kv_config:
            return default
        kv_store_config = (
            omni_kv_config.get("kv_store_config", {})
            if isinstance(omni_kv_config, dict)
            else getattr(omni_kv_config, "kv_store_config", {}) or {}
        )
        return kv_store_config.get(key, default) if isinstance(kv_store_config, dict) else default

    def _create_kv_store_backend(self):
        """Create KV store backend from omni_kv_config.kv_store_config."""
        omni_kv_config = getattr(self.vllm_config.model_config, "omni_kv_config", None)
        if not omni_kv_config:
            return None

        kv_store_config = (
            omni_kv_config.get("kv_store_config", None)
            if isinstance(omni_kv_config, dict)
            else getattr(omni_kv_config, "kv_store_config", None)
        )
        if not kv_store_config or not kv_store_config.get("enable_offload"):
            return None

        try:
            backend = OmniKvStoreFactory.create("x", kv_store_config)
            if backend:
                logger.info(
                    "[Omni] KV store backend enabled: %s",
                    backend,
                )
            return backend
        except Exception as e:
            logger.warning("[Omni] Failed to create KV store backend: %s", e)
            return None

    def _get_kv_transfer_criteria(self) -> dict | None:
        # Note: vllm_config is available in Scheduler after super().__init__
        if not hasattr(self, "vllm_config"):
            return None

        omni_kv_config = getattr(self.vllm_config.model_config, "omni_kv_config", None)
        if omni_kv_config:
            if isinstance(omni_kv_config, dict):
                return omni_kv_config.get("kv_transfer_criteria", None)
            else:
                return getattr(omni_kv_config, "kv_transfer_criteria", None)
        return None

    def _process_kv_transfer_trigger(self, request: Request, new_token_ids: list[int]) -> bool:
        """
        Check triggers and process side effects (marking transfer).
        Returns True if request should be STOPPED.
        Returns False if request should continue (even if transfer was triggered).
        """
        if not self.kv_transfer_criteria:
            return False

        if request.request_id in self.waiting_for_transfer_free:
            return False

        criteria_type = self.kv_transfer_criteria.get("type")

        # Universal duplicate check for once semantics
        if request.request_id in self.transfer_triggered_requests:
            return False

        if criteria_type == "prefill_finished":
            if request.num_computed_tokens >= request.num_prompt_tokens:
                logger.debug(f"[Omni] Request {request.request_id} triggered prefill_finished transfer (Non-Stop)")
                self.transfer_triggered_requests.add(request.request_id)
                self._mark_request_for_kv_transfer(request.request_id, request.num_computed_tokens)

                # Return False means "Do NOT stop the request" -> Continue Decoding
                return False

        elif criteria_type == "special_token":
            target_token_id = self.kv_transfer_criteria.get("token_id")
            if target_token_id is not None and target_token_id in new_token_ids:
                logger.debug(f"[Omni] Request {request.request_id} triggered special_token criteria (Non-Stop)")

                self.transfer_triggered_requests.add(request.request_id)

                # Calculate precise snapshot length (trim to sentinel)
                # Find the FIRST occurrence of the sentinel
                try:
                    idx = new_token_ids.index(target_token_id)
                    # seq_len = tokens_before_this_step + idx + 1 (include sentinel)
                    # request.num_computed_tokens already includes ALL new_token_ids
                    # so we subtract (len(new_token_ids) - (idx + 1))
                    tokens_to_exclude = len(new_token_ids) - (idx + 1)
                    snapshot_len = request.num_computed_tokens - tokens_to_exclude
                except ValueError:
                    snapshot_len = request.num_computed_tokens

                # Trigger Transfer
                self._mark_request_for_kv_transfer(request.request_id, snapshot_len)

                # Do NOT stop request
                return False

        return False

    def _execute_pending_frees(self) -> None:
        """Phase 2 of two-phase offload: free GPU blocks for extracted requests.

        Called at the start of schedule(), BEFORE super().schedule(), so that
        freed blocks are available for new/waiting requests in the same cycle.
        """
        if not self.offload_pending_free:
            return

        for req_id in list(self.offload_pending_free):
            request = self.requests.get(req_id)
            if request is None:
                # Request already finished/aborted
                self.offload_pending_free.discard(req_id)
                continue

            seq_len = request.num_computed_tokens
            # Free GPU blocks
            self.kv_cache_manager.free(request)
            self.encoder_cache_manager.free(request)
            # Preserve num_computed_tokens so resume path allocates all blocks
            # but only schedules 1 new token (not full recompute)
            request.status = RequestStatus.PREEMPTED
            request.num_computed_tokens = max(seq_len - 1, 0)
            request.spec_token_ids.clear()
            request.num_preemptions += 1
            # Remove from running and prepend to waiting
            self.running = remove_all(self.running, {request})
            self.waiting.prepend_request(request)
            # Track as offloaded
            self.offloaded_requests.add(req_id)
            self.offloaded_metadata[req_id] = {"seq_len": seq_len}
            logger.info(
                "[Omni] Offload Phase 2: freed blocks for %s, preserved %d computed tokens",
                req_id,
                request.num_computed_tokens,
            )

        self.offload_pending_free.clear()

    def schedule(self) -> SchedulerOutput:  # type: ignore[override]
        if self.chunk_transfer_adapter:
            self.chunk_transfer_adapter.process_pending_chunks(self.waiting, self.running)

        try:
            scheduler_output = super().schedule()
        finally:
            if self.chunk_transfer_adapter:
                # Add request waiting for chunk to the waiting and running queue
                self.chunk_transfer_adapter.restore_queues(self.waiting, self.running)
        try:
            # Late import to avoid circulars in some launch modes
            from .output import OmniNewRequestData

            # Rewrap base NewRequestData entries with OmniNewRequestData,
            # enriching with request-level payloads
            new_list = []
            for nr in scheduler_output.scheduled_new_reqs:
                req_id = getattr(nr, "req_id", None)
                request = self.requests.get(req_id) if req_id else None
                # Build omni entry preserving all base fields
                omni_nr = OmniNewRequestData(
                    req_id=nr.req_id,
                    external_req_id=(getattr(request, "external_req_id", None) if request else None),
                    prompt_token_ids=nr.prompt_token_ids,
                    mm_features=nr.mm_features,
                    sampling_params=nr.sampling_params,
                    pooling_params=nr.pooling_params,
                    block_ids=nr.block_ids,
                    num_computed_tokens=nr.num_computed_tokens,
                    lora_request=nr.lora_request,
                    # Enrich with omni payloads from the live request object
                    prompt_embeds=(getattr(request, "prompt_embeds", None) if request else None),
                    additional_information=(getattr(request, "additional_information", None) if request else None),
                )
                new_list.append(omni_nr)

            scheduler_output.scheduled_new_reqs = new_list  # type: ignore[assignment]
            if self.chunk_transfer_adapter:
                self.chunk_transfer_adapter.postprocess_scheduler_output(scheduler_output, self.requests)
            # Add information about requests needing KV cache transfer
            finished_reqs = self.get_finished_requests_needing_kv_transfer()

            # [Omni] Generate KV offload/prefetch decisions
            kv_offload_decisions = None
            if self.kv_store_backend is not None:
                # DEBUG: log usage and queue sizes every step
                _usage = self.kv_cache_manager.usage
                _n_running = len(self.running)
                _n_waiting = len(self.waiting)
                if _n_running > 0 and _n_waiting > 0:
                    logger.info(
                        "[Omni] Offload check: usage=%.4f, running=%d, waiting=%d, threshold=%.4f",
                        _usage,
                        _n_running,
                        _n_waiting,
                        self.offload_usage_threshold,
                    )
                kv_offload_decisions = self._generate_offload_decisions(scheduler_output)
        except Exception:
            # If anything goes wrong, leave the original output unchanged
            init_logger(__name__).exception("Failed to wrap scheduled_new_reqs with OmniNewRequestData")
            finished_reqs = {}
            kv_offload_decisions = None

        # Wrap in omni scheduler output to carry transfer metadata.
        base_fields = SchedulerOutput.__dataclass_fields__.keys()
        base_data = {name: getattr(scheduler_output, name) for name in base_fields}
        omni_output = OmniSchedulerOutput(
            **base_data,
            finished_requests_needing_kv_transfer=finished_reqs,
            kv_offload_decisions=kv_offload_decisions,
        )
        return omni_output

    def _generate_offload_decisions(self, scheduler_output: SchedulerOutput) -> dict[str, Any]:
        """Generate offload/prefetch decisions using two-phase protocol.

        Offload (Phase 1): When GPU KV usage exceeds threshold and there are
        waiting requests, select running decode requests to offload (largest KV
        first). Runner extracts KV to CPU and acks via kv_extracted_req_ids.

        Load: When an offloaded request gets re-scheduled from waiting, load
        its KV from CPU into newly allocated GPU blocks.

        Returns:
            {"offload_req_ids": [...], "load_req_ids": [...],
             "load_new_block_ids": {req_id: list[int]}}
        """
        offload_ids: list[str] = []
        load_ids: list[str] = []
        load_new_block_ids: dict[str, list[int]] = {}

        scheduled = set(scheduler_output.num_scheduled_tokens.keys())

        # --- Load: offloaded requests with CPU backup that got re-scheduled ---
        for req_id in list(self.offloaded_requests):
            if req_id in scheduled:
                load_ids.append(req_id)
                try:
                    block_ids_tuple = self.kv_cache_manager.get_block_ids(req_id)
                    if block_ids_tuple:
                        load_new_block_ids[req_id] = list(block_ids_tuple[0])
                except Exception:
                    logger.warning("[Omni] Failed to get new block IDs for load of %s", req_id)
                self.offloaded_requests.discard(req_id)
                self.offloaded_metadata.pop(req_id, None)

        # --- Offload: proactive threshold-based trigger ---
        if self.kv_cache_manager.usage >= self.offload_usage_threshold and len(self.waiting) > 0:
            # Build exclusion set
            excluded = (
                self.offload_pending_extract
                | self.offload_pending_free
                | self.offloaded_requests
                | self.waiting_for_transfer_free
                | self.active_kv_transfers
            )
            # Select candidates: running decode requests, largest KV first
            candidates = []
            for req in self.running:
                rid = req.request_id
                if rid in scheduled and rid not in excluded and req.num_computed_tokens > req.num_prompt_tokens:
                    candidates.append(req)
            candidates.sort(key=lambda r: r.num_computed_tokens, reverse=True)

            for req in candidates[: self.max_offloads_per_step]:
                offload_ids.append(req.request_id)
                self.offload_pending_extract.add(req.request_id)

            if offload_ids:
                logger.info(
                    "[Omni] KV usage %.1f%% >= %.1f%% threshold, offloading %d requests: %s",
                    self.kv_cache_manager.usage * 100,
                    self.offload_usage_threshold * 100,
                    len(offload_ids),
                    offload_ids,
                )

        if load_ids:
            logger.info("[Omni] Prefetching %d offloaded requests: %s", len(load_ids), load_ids)

        return {
            "offload_req_ids": offload_ids,
            "load_req_ids": load_ids,
            "load_new_block_ids": load_new_block_ids,
        }

    def _preempt_request(self, request: Request, timestamp: float) -> None:
        """Override preemption to preserve num_computed_tokens when CPU backup exists.

        Standard preemption (base class) resets num_computed_tokens=0, forcing
        full recomputation.  If KV is backed up on CPU, we preserve
        num_computed_tokens (minus 1) so the KVTransfer path (scheduler.py:629)
        kicks in on resume — base scheduler allocates new blocks and schedules
        only 1 new token instead of full prefill.
        """
        req_id = request.request_id
        has_backup = (
            self.kv_store_backend is not None and self.kv_store_backend.load_kv(req_id, target_device="cpu") is not None
        )

        if has_backup:
            seq_len = request.num_computed_tokens
            # Do block free + status change like base, but preserve num_computed_tokens
            self.kv_cache_manager.free(request)
            self.encoder_cache_manager.free(request)
            request.status = RequestStatus.PREEMPTED
            request.num_computed_tokens = max(seq_len - 1, 0)
            request.spec_token_ids.clear()
            request.num_preemptions += 1
            self.waiting.prepend_request(request)
            self.offloaded_requests.add(req_id)
            self.offloaded_metadata[req_id] = {"seq_len": seq_len}
            logger.info(
                "[Omni] Preempted %s with CPU backup: preserved %d computed tokens",
                req_id,
                request.num_computed_tokens,
            )
        else:
            # No backup — fall back to standard preemption (recompute from scratch)
            super()._preempt_request(request, timestamp)

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        sampled_token_ids = model_runner_output.sampled_token_ids
        logprobs = model_runner_output.logprobs
        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        pooler_outputs = model_runner_output.pooler_output
        num_nans_in_logits = model_runner_output.num_nans_in_logits
        kv_connector_output = model_runner_output.kv_connector_output
        cudagraph_stats: CUDAGraphStat | None = model_runner_output.cudagraph_stats

        perf_stats: PerfStats | None = None
        if self.perf_metrics and self.perf_metrics.is_enabled():
            perf_stats = self.perf_metrics.get_step_perf_stats_per_gpu(scheduler_output)

        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
        spec_decoding_stats: SpecDecodingStats | None = None
        kv_connector_stats: KVConnectorStats | None = (
            kv_connector_output.kv_connector_stats if kv_connector_output else None
        )
        if kv_connector_stats and self.connector:
            kv_stats = self.connector.get_kv_connector_stats()
            if kv_stats:
                kv_connector_stats = kv_connector_stats.aggregate(kv_stats)

        failed_kv_load_req_ids = None
        if kv_connector_output and kv_connector_output.invalid_block_ids:
            # These blocks contain externally computed tokens that failed to
            # load. Identify affected requests and adjust their computed token
            # count to trigger recomputation of the invalid blocks.
            failed_kv_load_req_ids = self._handle_invalid_blocks(kv_connector_output.invalid_block_ids)

        # NOTE(woosuk): As len(num_scheduled_tokens) can be up to 1K or more,
        # the below loop can be a performance bottleneck. We should do our best
        # to avoid expensive operations inside the loop.
        stopped_running_reqs: set[Request] = set()
        stopped_preempted_reqs: set[Request] = set()
        for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
            assert num_tokens_scheduled > 0
            if failed_kv_load_req_ids and req_id in failed_kv_load_req_ids:
                # Skip requests that were recovered from KV load failure
                continue
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # The request is already finished. This can happen if the
                # request is aborted while the model is executing it (e.g.,
                # in pipeline parallelism or async scheduling).
                continue

            req_index = model_runner_output.req_id_to_index[req_id]
            generated_token_ids = sampled_token_ids[req_index] if sampled_token_ids else []

            scheduled_spec_token_ids = scheduler_output.scheduled_spec_decode_tokens.get(req_id)
            if scheduled_spec_token_ids:
                num_draft_tokens = len(scheduled_spec_token_ids)
                num_accepted = len(generated_token_ids) - 1
                num_rejected = num_draft_tokens - num_accepted
                # num_computed_tokens represents the number of tokens
                # processed in the current step, considering scheduled
                # tokens and rejections. If some tokens are rejected,
                # num_computed_tokens is decreased by the number of rejected
                # tokens.
                if request.num_computed_tokens > 0:
                    request.num_computed_tokens -= num_rejected
                # If async scheduling, num_output_placeholders also includes
                # the scheduled spec tokens count and so is similarly adjusted.
                if request.num_output_placeholders > 0:
                    request.num_output_placeholders -= num_rejected
                spec_decoding_stats = self.make_spec_decoding_stats(
                    spec_decoding_stats,
                    num_draft_tokens=num_draft_tokens,
                    num_accepted_tokens=num_accepted,
                    num_invalid_spec_tokens=scheduler_output.num_invalid_spec_tokens,
                    request_id=req_id,
                )

            stopped = False
            new_logprobs = None
            new_token_ids = generated_token_ids
            pooler_output = pooler_outputs[req_index] if pooler_outputs else None
            kv_transfer_params = None
            status_before_stop = request.status
            finish_reason = None
            routed_experts = None

            # Check for stop and update request status.
            if new_token_ids:
                new_token_ids, stopped = self._update_request_with_output(request, new_token_ids)
            elif request.pooling_params and pooler_output is not None:
                # Pooling stops as soon as there is output.
                request.status = RequestStatus.FINISHED_STOPPED
                stopped = True

            # If criteria returns True, it means we must STOP the request.
            # If criteria returns False, it might have triggered a background
            # transfer (e.g. prefill finished / special token) but continues decoding.
            if not stopped and self._process_kv_transfer_trigger(request, new_token_ids):
                stopped = True

            if stopped:
                routed_experts = self._get_routed_experts(request)

                # Capture finish_reason BEFORE _handle_stopped_request, which may
                # reset the status to WAITING for streaming requests that continue.
                finish_reason = request.get_finished_reason()
                finished = self._handle_stopped_request(request)
                if finished:
                    kv_transfer_params = self._free_request(request)
                if status_before_stop == RequestStatus.RUNNING:
                    stopped_running_reqs.add(request)
                elif status_before_stop == RequestStatus.WAITING_FOR_CHUNK:
                    # In async chunk mode, request may be in either queue.
                    # Remove from both to avoid stale queue entries.
                    stopped_running_reqs.add(request)
                    stopped_preempted_reqs.add(request)
                else:
                    stopped_preempted_reqs.add(request)

            # Extract sample logprobs if needed.
            if request.sampling_params is not None and request.sampling_params.logprobs is not None and logprobs:
                new_logprobs = logprobs.slice_request(req_index, len(new_token_ids))

            if new_token_ids and self.structured_output_manager.should_advance(request):
                struct_output_request = request.structured_output_request
                assert struct_output_request is not None
                assert struct_output_request.grammar is not None
                ok = struct_output_request.grammar.accept_tokens(req_id, new_token_ids)
                if not ok:
                    logger.warning(
                        "Unexpected: grammar rejected tokens %s for request %s.",
                        new_token_ids,
                        req_id,
                    )

            if num_nans_in_logits is not None and req_id in num_nans_in_logits:
                request.num_nans_in_logits = num_nans_in_logits[req_id]

            # Get prompt logprobs for this request.
            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
            if new_token_ids or pooler_output is not None or kv_transfer_params or stopped:
                # Add EngineCoreOutput for this Request.
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=new_token_ids,
                        finish_reason=finish_reason,
                        new_logprobs=new_logprobs,
                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                        pooling_output=pooler_output,
                        stop_reason=request.stop_reason,
                        events=request.take_events(),
                        kv_transfer_params=kv_transfer_params,
                        trace_headers=request.trace_headers,
                        num_cached_tokens=request.num_cached_tokens,
                        num_external_computed_tokens=request.num_external_computed_tokens,
                        routed_experts=routed_experts,
                        num_nans_in_logits=request.num_nans_in_logits,
                    )
                )
                if self.chunk_transfer_adapter is not None:
                    self.chunk_transfer_adapter.save_async(pooler_output, request)
            else:
                # Invariant: EngineCore returns no partial prefill outputs.
                assert not prompt_logprobs_tensors

        # Remove the stopped requests from the running and waiting queues.
        if stopped_running_reqs:
            self.running = remove_all(self.running, stopped_running_reqs)
        if stopped_preempted_reqs:
            # This is a rare case and unlikely to impact performance.
            self.waiting.remove_requests(stopped_preempted_reqs)

        # [Main] Handle failed KV load requests
        if failed_kv_load_req_ids and not self.recompute_kv_load_failures:
            requests = [self.requests[req_id] for req_id in failed_kv_load_req_ids]
            self.finish_requests(failed_kv_load_req_ids, RequestStatus.FINISHED_ERROR)
            for request in requests:
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=request.request_id,
                        new_token_ids=[],
                        finish_reason=request.get_finished_reason(),
                        events=request.take_events(),
                        trace_headers=request.trace_headers,
                        num_cached_tokens=request.num_cached_tokens,
                    )
                )
                if self.chunk_transfer_adapter is not None:
                    self.chunk_transfer_adapter.cleanup(
                        request.request_id,
                        getattr(request, "external_req_id", None),
                    )

        # [Omni] Cleanup state for finished requests
        for req in stopped_running_reqs:
            rid = req.request_id
            if rid not in self.waiting_for_transfer_free:
                self.transfer_triggered_requests.discard(rid)
                self.active_kv_transfers.discard(rid)
            # Clean up offload state
            self.offload_pending_extract.discard(rid)
            self.offload_pending_free.discard(rid)
            if rid in self.offloaded_requests:
                self.offloaded_requests.discard(rid)
                self.offloaded_metadata.pop(rid, None)
            if self.kv_store_backend is not None:
                try:
                    self.kv_store_backend.evict_kv(rid)
                except Exception:
                    pass

        # Same for preempted
        for req in stopped_preempted_reqs:
            rid = req.request_id
            if rid not in self.waiting_for_transfer_free:
                self.transfer_triggered_requests.discard(rid)
                self.active_kv_transfers.discard(rid)
            # Clean up offload state
            self.offload_pending_extract.discard(rid)
            self.offload_pending_free.discard(rid)
        # KV Connector: update state for finished KV Transfers.
        if kv_connector_output:
            self._update_from_kv_xfer_finished(kv_connector_output)

        # collect KV cache events from KV cache manager
        events = self.kv_cache_manager.take_events()

        # collect KV cache events from connector
        if self.connector is not None:
            connector_events = self.connector.take_events()
            if connector_events:
                if events is None:
                    events = list(connector_events)
                else:
                    events.extend(connector_events)

        # publish collected KV cache events
        if events:
            batch = KVEventBatch(ts=time(), events=events)
            self.kv_event_publisher.publish(batch)

        # Create EngineCoreOutputs for all clients that have requests with
        # outputs in this step.
        engine_core_outputs = {client_index: EngineCoreOutputs(outputs=outs) for client_index, outs in outputs.items()}

        finished_req_ids = self.finished_req_ids_dict
        if finished_req_ids:
            # Include ids of requests that finished since last outputs
            # were sent.
            for client_index, finished_set in finished_req_ids.items():
                # Set finished request set in EngineCoreOutputs for this client.
                if (eco := engine_core_outputs.get(client_index)) is not None:
                    eco.finished_requests = finished_set
                else:
                    engine_core_outputs[client_index] = EngineCoreOutputs(finished_requests=finished_set)
            finished_req_ids.clear()

        if (stats := self.make_stats(spec_decoding_stats, kv_connector_stats, cudagraph_stats, perf_stats)) is not None:
            # Return stats to only one of the front-ends.
            if (eco := next(iter(engine_core_outputs.values()), None)) is None:
                # We must return the stats even if there are no request
                # outputs this step.
                engine_core_outputs[0] = eco = EngineCoreOutputs()
            eco.scheduler_stats = stats

        # Process extraction acks from runner (inter-stage KV transfers + offload)
        try:
            kv_extracted_ids = getattr(model_runner_output, "kv_extracted_req_ids", None)
            if kv_extracted_ids:
                for req_id in kv_extracted_ids:
                    # [Omni] Offload ack: move from pending_extract to pending_free
                    if req_id in self.offload_pending_extract:
                        self.offload_pending_extract.discard(req_id)
                        self.offload_pending_free.add(req_id)
                        logger.debug("[Omni] Offload extraction confirmed for %s, pending free", req_id)
                        continue  # Not an inter-stage transfer ack

                    if req_id in self.active_kv_transfers:
                        self.active_kv_transfers.remove(req_id)
                        logger.debug("[Omni] KV Transfer finished for %s", req_id)

                    if req_id in self.waiting_for_transfer_free:
                        req = self.requests.get(req_id)
                        if req:
                            self.kv_cache_manager.free(req)
                            if req_id in self.requests:
                                del self.requests[req_id]
                            self.transfer_triggered_requests.discard(req_id)
                            self.active_kv_transfers.discard(req_id)
                            logger.debug("Freed blocks for %s after transfer extraction", req_id)
                        self.waiting_for_transfer_free.remove(req_id)
        except Exception:
            init_logger(__name__).exception("Failed to process extraction acks")

        return engine_core_outputs

    def _free_request(self, request: Request, delay_free_blocks: bool = False) -> dict[str, Any] | None:
        # TODO(wzliu)! for offline mode, we should not end process until all data is transferred
        """Mark a request as finished and free its resources."""
        assert request.is_finished()

        # 1. Standard cleanup parts from base _free_request
        connector_delay_free_blocks = False
        kv_xfer_params = None
        if self.connector is not None:
            connector_delay_free_blocks, kv_xfer_params = self._connector_finished(request)

        self.encoder_cache_manager.free(request)
        request_id = request.request_id
        self.finished_req_ids.add(request_id)
        if self.finished_req_ids_dict is not None:
            self.finished_req_ids_dict[request.client_index].add(request_id)

        # 2. Omni Specific: Check if we need to transfer KV
        if self._should_transfer_kv_for_request(request_id):
            already_triggered = request_id in self.transfer_triggered_requests
            is_active = request_id in self.active_kv_transfers

            if already_triggered:
                if is_active:
                    # It triggered but hasn't finished yet. We MUST wait.
                    logger.debug(f"[Omni] Request {request_id} finished but transfer is still ACTIVE. Waiting.")
                    self.waiting_for_transfer_free.add(request_id)
                    # We do NOT mark for transfer again, just wait.
                    kv_xfer_params = None  # No new transfer params
                    return kv_xfer_params
                else:
                    logger.debug(
                        f"[Omni] Request {request_id} finished and transfer no longer ACTIVE (extracted/acked). "
                        "Freeing immediately."
                    )
            else:
                self.waiting_for_transfer_free.add(request_id)
                self._mark_request_for_kv_transfer(request_id, request.num_computed_tokens)
                # Return KV transfer metadata so it propagates to RequestOutput
                if request_id in self.requests_needing_kv_transfer:
                    transfer_data = self.requests_needing_kv_transfer[request_id]
                    kv_xfer_params = {
                        "past_key_values": transfer_data["block_ids"],
                        "kv_metadata": {"seq_len": transfer_data["seq_len"], "block_ids": transfer_data["block_ids"]},
                    }
                    # Also update request.additional_information for good measure
                    add_info = getattr(request, "additional_information", None)
                    # If additional_information is an AdditionalInformationPayload-like object,
                    # unpack list_data into a plain dict.
                    if (
                        add_info is not None
                        and hasattr(add_info, "entries")
                        and isinstance(getattr(add_info, "entries"), dict)
                    ):
                        request.additional_information = {
                            k: getattr(v, "list_data")
                            for k, v in getattr(add_info, "entries").items()
                            if getattr(v, "list_data", None) is not None
                        }
                        add_info = request.additional_information
                    if add_info is None:
                        request.additional_information = {}
                        add_info = request.additional_information
                    if isinstance(add_info, dict):
                        add_info.update(kv_xfer_params)

                return kv_xfer_params

        # 3. Standard Freeing
        delay_free_blocks |= connector_delay_free_blocks
        if not delay_free_blocks:
            self._free_blocks(request)

        return kv_xfer_params

    def _free_blocks(self, request: Request):
        # Helper to match base class structure if not directly available
        # VLLMScheduler has _free_blocks
        super()._free_blocks(request)

    def _mark_request_for_kv_transfer(self, req_id: str, seq_len: int) -> None:
        """Mark a request as needing KV cache transfer when it finishes."""
        # Avoid duplicate marking (if already pending in queue)
        if req_id in self.requests_needing_kv_transfer:
            return

        if self._should_transfer_kv_for_request(req_id):
            # [Omni] Get block IDs from KVCacheManager
            try:
                block_ids_tuple = self.kv_cache_manager.get_block_ids(req_id)
                if block_ids_tuple and len(block_ids_tuple) > 0:
                    block_ids = block_ids_tuple[0]

                    # [Omni] Fix: Truncate blocks to match seq_len snapshot
                    # We need to know block_size. Usually in self.cache_config.block_size
                    # Note: vllm_config might not be directly available, check scheduler_config or cache_config
                    if hasattr(self, "cache_config") and hasattr(self.cache_config, "block_size"):
                        block_size = self.cache_config.block_size
                    elif hasattr(self, "scheduler_config") and hasattr(
                        self.scheduler_config, "block_size"
                    ):  # Some versions
                        block_size = self.scheduler_config.block_size
                    else:
                        raise ValueError("Block size not found in cache_config or scheduler_config")

                    # ceil(seq_len / block_size)
                    num_blocks = (seq_len + block_size - 1) // block_size
                    if len(block_ids) > num_blocks:
                        logger.debug(
                            f"[Omni] Truncating blocks for {req_id} from {len(block_ids)} "
                            f"to {num_blocks} (seq_len={seq_len})"
                        )
                        block_ids = block_ids[:num_blocks]

                else:
                    block_ids = []
            except Exception as e:
                init_logger(__name__).warning(f"Failed to get block IDs for {req_id}: {e}")
                block_ids = []

            self.requests_needing_kv_transfer[req_id] = {"seq_len": seq_len, "block_ids": block_ids}
            logger.debug(f"Marked request {req_id} for KV cache transfer (len={seq_len}, blocks={len(block_ids)})")

    def _should_transfer_kv_for_request(self, req_id: str) -> bool:
        """Determine if a request should trigger KV cache transfer."""
        need_send = False
        # Try to read from vLLM Config (where YAML config is typically loaded)
        # Check for omni_kv_config attribute
        omni_kv_config = getattr(self.vllm_config.model_config, "omni_kv_config", None)
        if omni_kv_config:
            # omni_kv_config could be an object or a dict
            if isinstance(omni_kv_config, dict):
                need_send = omni_kv_config.get("need_send_cache", False)
            else:
                need_send = getattr(omni_kv_config, "need_send_cache", False)
        return need_send

    def has_requests(self) -> bool:
        """Check if there are any requests to process, including KV transfers."""
        # [Omni] Also check for pending KV transfers
        if self.requests_needing_kv_transfer or self.active_kv_transfers or self.waiting_for_transfer_free:
            return True
        return super().has_requests()

    def has_finished_requests(self) -> bool:
        """Check if there are any finished requests (including those needing KV transfer)."""
        if self.requests_needing_kv_transfer or self.active_kv_transfers or self.waiting_for_transfer_free:
            return True
        return super().has_finished_requests()

    def has_unfinished_requests(self) -> bool:
        """Check if there are any unfinished requests (including those needing KV transfer)."""
        # [Omni] Also check for pending KV transfers to ensure the engine loop continues
        # MUST verify waiting_for_transfer_free and active_kv_transfers
        # Otherwise engine loop might exit before transfer Ack is received.
        if self.requests_needing_kv_transfer or self.active_kv_transfers or self.waiting_for_transfer_free:
            return True
        return super().has_unfinished_requests()

    def get_finished_requests_needing_kv_transfer(self) -> dict[str, dict]:
        """Get and clear the list of requests needing KV cache transfer.
        Returns dict: {req_id: {"seq_len": int, "block_ids": list[int]}}
        """
        requests = self.requests_needing_kv_transfer.copy()

        # Mark these requests as ACTIVE (sent to runner)
        self.active_kv_transfers.update(requests.keys())

        self.requests_needing_kv_transfer.clear()
        return requests
