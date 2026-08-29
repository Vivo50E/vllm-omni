# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LMCache hidden-state store/restore for the AR stage.

Mixed into ``GPUARModelRunner`` only. The shared ``OmniGPUModelRunner`` base
keeps thin no-op hooks (``_setup_lmcache_hidden_state_offload`` and
``_drop_hs_pending_state``) so non-AR runners carry none of this.
"""

import os

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

# Canonical layer_idx for the final hidden-state tap. Using a fixed sentinel (not
# a per-model layer count) keeps LMCache's hidden_state_layers allowlist stable
# as taps are added/removed.
_LMCACHE_HS_HIDDEN_LAYER_IDX: int = -1


def _hs_layer_idx(layer_key: str) -> int:
    """Map a layer_key to the canonical LMCache layer_idx."""
    return _LMCACHE_HS_HIDDEN_LAYER_IDX if layer_key == "hidden" else int(layer_key)


class LMCacheHiddenStateMixin:
    """Store/restore per-layer hidden states in LMCache alongside KV."""

    def _setup_lmcache_hidden_state_offload(self) -> None:
        """Init HS-offload state and discover the mm taps from the talker config."""
        self._hs_pending_buffer: dict[str, dict[str, list[torch.Tensor]]] = {}
        self._hs_saved_boundary: dict[str, int] = {}
        self._lmcache_hs_mm_keys: tuple[str, ...] = ()
        omni_kv = getattr(getattr(self, "model_config", None), "omni_kv_config", None)
        kv_store = omni_kv.get("kv_store_config") if isinstance(omni_kv, dict) else None
        self._has_lmcache = isinstance(kv_store, dict) and bool(kv_store.get("lmcache_config"))
        if not self._has_lmcache:
            return
        # Qwen3-Omni captures layer 0 (word embeddings) + accept_hidden_layer;
        # Qwen2.5-Omni has none, leaving only the final "hidden" tap cached.
        talker_config = getattr(getattr(self, "model", None), "talker_config", None)
        accept_layer = getattr(talker_config, "accept_hidden_layer", None)
        if accept_layer is not None:
            self._lmcache_hs_mm_keys = ("0", str(int(accept_layer)))
        logger.info("LMCache hidden state store/restore enabled (mm_keys=%s)", self._lmcache_hs_mm_keys)

    def _get_lmcache_adapter(self):
        """Lazily find the LMCacheConnectorV1Impl adapter from the KV connector."""
        cached = getattr(self, "_lmcache_adapter_cached", None)
        if cached is not None:
            return cached
        try:
            from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group

            if not has_kv_transfer_group():
                return None
            connector = get_kv_transfer_group()
            # MultiConnector: search sub-connectors
            if hasattr(connector, "_connectors"):
                for c in connector._connectors:
                    impl = getattr(c, "_lmcache_engine", None)
                    if impl is not None and hasattr(impl, "lmcache_engine"):
                        self._lmcache_adapter_cached = impl
                        return impl
            # Direct LMCacheConnectorV1
            impl = getattr(connector, "_lmcache_engine", None)
            if impl is not None and hasattr(impl, "lmcache_engine"):
                self._lmcache_adapter_cached = impl
                return impl
        except Exception:
            pass
        return None

    def _maybe_store_hs_to_lmcache(
        self,
        hidden_states: torch.Tensor,
        multimodal_outputs: dict | None,
        num_tokens_unpadded: int,
        scheduler_output,
    ):
        """Buffer per-step HS and flush full chunks to LMCache on chunk boundary crossings.

        LMCache's HS pool is chunk-aligned (same boundaries as KV). We buffer this
        request's HS rows since the last flushed chunk boundary and only call
        ``store_hidden_states`` when a new chunk completes, mirroring KV's
        "wait for full chunk" semantic.
        """
        if not self._has_lmcache:
            return
        adapter = self._get_lmcache_adapter()
        if adapter is None or not hasattr(adapter, "lmcache_engine"):
            return
        engine = adapter.lmcache_engine
        if engine is None:
            return
        hs_store = engine.hidden_state_store
        if hs_store is None:
            return

        chunk_size = int(getattr(engine.config, "chunk_size", None) or 256)

        layers_to_store: dict[str, torch.Tensor] = {}
        mm_layers: dict = {}
        if isinstance(multimodal_outputs, dict):
            hs_dict = multimodal_outputs.get("hidden_states")
            if isinstance(hs_dict, dict) and isinstance(hs_dict.get("layers"), dict):
                mm_layers = hs_dict["layers"]
        for key in self._lmcache_hs_mm_keys:
            t = mm_layers.get(key)
            if t is None:
                try:
                    t = mm_layers.get(int(key))
                except (TypeError, ValueError):
                    t = None
            if isinstance(t, torch.Tensor):
                layers_to_store[key] = t
        if isinstance(hidden_states, torch.Tensor):
            layers_to_store["hidden"] = hidden_states

        if not layers_to_store:
            return

        hs_cpu_by_layer: dict[str, torch.Tensor] = {
            k: t[:num_tokens_unpadded].detach().to("cpu").contiguous() for k, t in layers_to_store.items()
        }

        for req_id in self.input_batch.req_ids:
            sched = scheduler_output.num_scheduled_tokens.get(req_id, 0)
            if sched <= 0:
                continue
            req_idx = self.input_batch.req_id_to_index[req_id]
            start = int(self.query_start_loc.cpu[req_idx])
            num_computed = int(self.input_batch.num_computed_tokens_cpu[req_idx])
            total = num_computed + sched

            # Preemption/reset regressed this request; drop stale buffered HS.
            if num_computed < self._hs_saved_boundary.get(req_id, 0):
                self._drop_hs_pending_state(req_id)

            # Every layer must cover this request's rows; skip otherwise to avoid
            # buffering a misaligned/partial chunk.
            if any(int(hs_cpu_by_layer[k].shape[0]) < start + sched for k in layers_to_store):
                continue

            req_buf = self._hs_pending_buffer.setdefault(req_id, {})
            for layer_key in layers_to_store:
                req_buf.setdefault(layer_key, []).append(hs_cpu_by_layer[layer_key][start : start + sched])

            saved_boundary = self._hs_saved_boundary.get(req_id, 0)
            new_boundary = (total // chunk_size) * chunk_size
            if new_boundary <= saved_boundary:
                continue

            chunk_rows = new_boundary - saved_boundary
            seg_token_ids = self.input_batch.token_ids_cpu[req_idx, :new_boundary].tolist()
            full_bufs = {k: (torch.cat(b, dim=0) if len(b) > 1 else b[0]) for k, b in req_buf.items()}
            if any(int(fb.shape[0]) < chunk_rows for fb in full_bufs.values()):
                continue

            all_stored = True
            expected_chunks = chunk_rows // chunk_size
            for layer_key, full_buf in full_bufs.items():
                try:
                    stored = hs_store.store_hidden_states(
                        seg_token_ids,
                        full_buf[:chunk_rows],
                        layer_idx=_hs_layer_idx(layer_key),
                        token_offset=saved_boundary,
                    )
                except Exception:
                    logger.exception("LMCache: store_hidden_states failed (req_id=%s layer=%s)", req_id, layer_key)
                    all_stored = False
                    continue
                # A full HS pool stops the store early and returns normally, so
                # the count is the only signal that a chunk did not persist.
                if int(stored) != expected_chunks:
                    logger.error(
                        "LMCache: stored %d of %d HS chunks (req_id=%s layer=%s); the HS pool "
                        "is likely full. Not advancing the boundary so the chunk is retried.",
                        int(stored),
                        expected_chunks,
                        req_id,
                        layer_key,
                    )
                    all_stored = False
            # Only trim buffers and advance the boundary once every layer's chunk
            # is persisted, so a failure retries the same boundary next step.
            if not all_stored:
                continue
            for layer_key, full_buf in full_bufs.items():
                remainder = full_buf[chunk_rows:]
                req_buf[layer_key] = [remainder] if remainder.shape[0] > 0 else []
            self._hs_saved_boundary[req_id] = new_boundary

    def _drop_hs_pending_state(self, req_id: str) -> None:
        """Discard buffered HS / saved-boundary / restored state for ``req_id``."""
        self._hs_pending_buffer.pop(req_id, None)
        self._hs_saved_boundary.pop(req_id, None)
        restored_mm = getattr(self, "_restored_mm", None)
        if restored_mm is not None:
            restored_mm.pop(req_id, None)

    def _maybe_write_hs_restore_marker(self, req_id: str, num_computed: int, layers: dict) -> None:
        """Test-only hook: record a restore event when OMNI_HS_RESTORE_MARKER_PATH is set."""
        marker_path = os.environ.get("OMNI_HS_RESTORE_MARKER_PATH")
        if not marker_path:
            return
        try:
            with open(marker_path, "a", encoding="utf-8") as fp:
                fp.write(f"{req_id}\t{num_computed}\t{','.join(sorted(layers.keys()))}\n")
        except OSError:
            pass

    def _maybe_restore_hs_from_lmcache(self, scheduler_output=None):
        """Restore per-layer hidden states from LMCache for KV-hit new requests.

        All-or-nothing: if any required layer is missing or shorter than the KV
        prefix, nothing is prepended (the talker would otherwise get truncated
        conditioning). Stashes into ``_restored_mm`` for the pooler payload build.
        """
        if not self._has_lmcache or scheduler_output is None:
            return
        adapter = self._get_lmcache_adapter()
        if adapter is None or not hasattr(adapter, "lmcache_engine"):
            return
        engine = adapter.lmcache_engine
        if engine is None:
            return
        hs_store = engine.hidden_state_store
        if hs_store is None:
            return

        # Per-request restored HS, consumed (popped) by _build_omni_pooler_payload.
        # Not wiped wholesale each step: with async omni output the pooler build is
        # deferred, so a later step's restore must not clear an earlier entry.
        if getattr(self, "_restored_mm", None) is None:
            self._restored_mm: dict[str, dict[str, torch.Tensor]] = {}
        for new_req in scheduler_output.scheduled_new_reqs:
            if new_req.num_computed_tokens <= 0:
                continue
            req_id = new_req.req_id
            req_idx = self.input_batch.req_id_to_index.get(req_id)
            if req_idx is None:
                continue
            num_computed = new_req.num_computed_tokens
            chunk_sz = int(getattr(engine.config, "chunk_size", None) or 256)
            prompt_tokens = int(self.input_batch.num_prompt_tokens[req_idx])
            # Round up to the next chunk boundary so retrieval keys match offload.
            aligned_up = ((num_computed + chunk_sz - 1) // chunk_sz) * chunk_sz
            retrieve_len = min(prompt_tokens, aligned_up)
            if retrieve_len <= 0:
                continue
            lookup_ids = self.input_batch.token_ids_cpu[req_idx, :retrieve_len].tolist()

            layers: dict[str, torch.Tensor] = {}
            incomplete = False
            for layer_key in (*self._lmcache_hs_mm_keys, "hidden"):
                hs = hs_store.retrieve_hidden_states(lookup_ids, layer_idx=_hs_layer_idx(layer_key))
                if hs is None or int(hs.shape[0]) < num_computed:
                    incomplete = True
                    break
                layers[layer_key] = hs[:num_computed]

            if incomplete:
                # KV is already restored but HS is not fully available; prepending
                # would give the talker truncated conditioning. Fail loud, write nothing.
                num_external = int(getattr(new_req, "num_external_computed_tokens", 0) or 0)
                logger.error(
                    "LMCache: incomplete HS restore for req_id=%s (num_computed=%d, "
                    "%d of them from the connector); skipping prepend. Size the HS pool "
                    ">= KV pool or disable HS offload.",
                    req_id,
                    num_computed,
                    num_external,
                )
                # Writing nothing is not enough: the merge still reads this
                # request's slots, which hold leftovers for blocks the connector
                # filled rather than this engine.
                if num_external > 0 and self.omni_prefix_cache is not None:
                    self.omni_prefix_cache.drop_prefix_cached_new_req_id(req_id)
                continue
            if not layers:
                continue

            logger.debug(
                "LMCache: restored HS (req_id=%s, num_prefix_tokens=%d, layers=%s)",
                req_id,
                num_computed,
                list(layers.keys()),
            )
            self._maybe_write_hs_restore_marker(req_id, num_computed, layers)
            # mm layers use the flattened payload key; "hidden" stays as-is.
            remapped = {(lk if lk == "hidden" else f"hidden_states.layer_{lk}"): hs for lk, hs in layers.items()}
            # Exactly one consumer: the merge already picks up the slots we
            # write, so stashing for the pooler payload too prepends twice.
            if self.omni_prefix_cache is not None:
                for cache_key, hs in remapped.items():
                    self.omni_prefix_cache.write_restored_hidden_states(req_idx, self.input_batch, cache_key, hs)
            else:
                self._restored_mm[req_id] = remapped
