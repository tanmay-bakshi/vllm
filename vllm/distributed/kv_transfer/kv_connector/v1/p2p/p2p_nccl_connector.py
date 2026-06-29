# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import threading
import time
import traceback
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import regex as re
import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine import (
    P2pNcclEngine,
)
from vllm.distributed.parallel_state import get_world_group
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import MLACommonMetadata
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class ReqMeta:
    # Request Id
    request_id: str
    # Request block ids, keyed by KV cache group.
    block_ids_by_group: tuple[torch.Tensor, ...]
    # Request num tokens
    num_tokens: int

    @staticmethod
    def make_meta(
        request_id: str,
        token_ids: list[int],
        block_ids: tuple[list[int], ...],
    ) -> "ReqMeta":
        block_ids_by_group = tuple(
            torch.tensor(group_block_ids, dtype=torch.int64)
            for group_block_ids in block_ids
        )
        return ReqMeta(
            request_id=request_id,
            block_ids_by_group=block_ids_by_group,
            num_tokens=len(token_ids),
        )


@dataclass
class LayerBundleEntry:
    """KV tensor slice queued for a bundled P2P transfer.

    :ivar layer_name: Layer name associated with the tensor slice.
    :ivar tensor: Flattenable KV tensor slice for the layer.
    """

    layer_name: str
    tensor: torch.Tensor


@dataclass
class LayerBundleLoadPlan:
    """Validated shape contract for loading one P2P KV bundle."""

    bundle_index: int
    tensor_id: str
    layer_names: list[str]
    expected_entries: list[
        tuple[str, torch.Tensor, torch.Tensor, torch.Size, int]
    ]
    expected_dtype: torch.dtype
    expected_numel: int
    expected_nbytes: int


@dataclass
class PrefetchedLayerBundle:
    """Staged receive-store result for one P2P KV bundle."""

    plan: LayerBundleLoadPlan | None
    bundle: torch.Tensor | None
    error: str | None = None
    staged_nbytes: int = 0
    defer_to_sync: bool = False


@dataclass
class PrefetchReservation:
    """Memory reservation outcome for a staged P2P prefetch."""

    succeeded: bool
    reason: str | None = None
    defer_to_sync: bool = False


@dataclass
class P2pNcclConnectorMetadata(KVConnectorMetadata):
    requests: list[ReqMeta]

    def __init__(self):
        self.requests = []

    def add_request(
        self,
        request_id: str,
        token_ids: list[int],
        block_ids: tuple[list[int], ...],
    ) -> None:
        self.requests.append(
            ReqMeta.make_meta(request_id, token_ids, block_ids)
        )


class P2pNcclConnector(KVConnectorBase_V1, SupportsHMA):
    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        self._block_size = vllm_config.cache_config.block_size
        self._requests_need_load: dict[str, Any] = {}
        self._block_ids_with_load_errors: set[int] = set()
        self._timing_enabled = os.environ.get("VLLM_P2P_NCCL_TIMING", "0") == "1"
        self._trace_events_enabled = (
            os.environ.get("VLLM_P2P_NCCL_TRACE", "0") == "1"
        )
        self._request_timing_enabled = (
            os.environ.get("VLLM_P2P_NCCL_REQUEST_TIMING", "0") == "1"
            or self._trace_events_enabled
        )
        self._timing_lock = threading.Lock()
        self._timing_stats: dict[str, dict[str, float]] = {}
        self._request_timing_stats: dict[str, dict[str, dict[str, float]]] = {}
        self._request_milestone_once_events: dict[str, set[str]] = {}
        self._load_kv_async_enabled = (
            os.environ.get("VLLM_P2P_NCCL_LOAD_KV_ASYNC", "0") == "1"
        )
        self._pipelined_load_requested = (
            os.environ.get("VLLM_P2P_NCCL_PIPELINED_LOAD", "0") == "1"
        )
        self._pipelined_load_enabled = (
            self._pipelined_load_requested and self._load_kv_async_enabled is False
        )
        if self._pipelined_load_requested and self._load_kv_async_enabled:
            logger.warning(
                "P2pNcclConnector ignoring pipelined load because async KV "
                "load scheduling is enabled"
            )
        self._pipelined_prefetch_requested = (
            os.environ.get("VLLM_P2P_NCCL_PIPELINED_PREFETCH", "0") == "1"
        )
        self._pipelined_prefetch_enabled = (
            self._pipelined_prefetch_requested and self._pipelined_load_enabled
        )
        if self._pipelined_prefetch_requested and self._pipelined_load_enabled is False:
            logger.warning(
                "P2pNcclConnector ignoring pipelined prefetch because "
                "pipelined KV load is not enabled"
            )
        self._pipelined_prefetch_window = int(
            os.environ.get("VLLM_P2P_NCCL_PIPELINED_PREFETCH_WINDOW", "1")
        )
        if self._pipelined_prefetch_window < 1:
            self._pipelined_prefetch_enabled = False
            logger.warning(
                "P2pNcclConnector ignoring pipelined prefetch because "
                "prefetch window is less than 1"
            )
        max_staged_bytes_text = os.environ.get(
            "VLLM_P2P_NCCL_PIPELINED_PREFETCH_MAX_STAGED_BYTES"
        )
        if max_staged_bytes_text is None:
            self._pipelined_prefetch_max_staged_bytes = max(
                1,
                int(float(self._kv_transfer_config.kv_buffer_size) // 4),
            )
        else:
            self._pipelined_prefetch_max_staged_bytes = int(
                float(max_staged_bytes_text)
            )
        if self._pipelined_prefetch_max_staged_bytes < 1:
            self._pipelined_prefetch_enabled = False
            logger.warning(
                "P2pNcclConnector ignoring pipelined prefetch because "
                "max staged bytes is less than 1"
            )
        self._finished_recving_request_ids: set[str] = set()
        self._pending_async_load_requests: dict[str, ReqMeta] = {}
        self._pending_async_load_request_started_at: dict[str, float] = {}
        self._pending_async_load_expected_tensor_ids: dict[str, tuple[str, ...]] = {}
        self._pending_async_load_seen_tensor_ids: dict[str, set[str]] = {}
        self._pending_async_load_all_tensors_visible_at: dict[str, float] = {}
        self._pending_async_load_once_events: dict[str, set[str]] = {}
        self._load_kv_async_pending_timeout_s = float(
            os.environ.get(
                "VLLM_P2P_NCCL_LOAD_KV_PENDING_TIMEOUT_S",
                "300",
            )
        )
        if self._load_kv_async_pending_timeout_s < 0.0:
            self._load_kv_async_pending_timeout_s = 0.0
        self._load_kv_async_max_ready_loads_per_step = int(
            os.environ.get(
                "VLLM_P2P_NCCL_LOAD_KV_ASYNC_MAX_READY_PER_STEP",
                "0",
            )
        )
        if self._load_kv_async_max_ready_loads_per_step < 0:
            self._load_kv_async_max_ready_loads_per_step = 0
        self._load_kv_async_max_cap_pending_requests = int(
            os.environ.get(
                "VLLM_P2P_NCCL_LOAD_KV_ASYNC_MAX_CAP_PENDING",
                "0",
            )
        )
        if self._load_kv_async_max_cap_pending_requests < 0:
            self._load_kv_async_max_cap_pending_requests = 0
        self._load_kv_async_max_cap_ready_requests = int(
            os.environ.get(
                "VLLM_P2P_NCCL_LOAD_KV_ASYNC_MAX_CAP_READY",
                "0",
            )
        )
        if self._load_kv_async_max_cap_ready_requests < 0:
            self._load_kv_async_max_cap_ready_requests = 0
        self.is_producer = self._kv_transfer_config.is_kv_producer
        bundle_layer_count = int(
            self._kv_transfer_config.get_from_extra_config(
                "p2p_bundle_layer_count", "1"
            )
        )
        if bundle_layer_count < 1:
            bundle_layer_count = 1
        self._bundle_layer_count = bundle_layer_count
        if (
            self._pipelined_prefetch_enabled
            and self._is_layer_bundling_enabled() is False
        ):
            self._pipelined_prefetch_enabled = False
            logger.warning(
                "P2pNcclConnector ignoring pipelined prefetch because "
                "layer bundling is not enabled"
            )
        self._pending_layer_bundles: dict[str, list[LayerBundleEntry]] = {}
        self._pending_bundle_remote_addresses: dict[str, str] = {}
        self._next_bundle_indices: dict[str, int] = {}
        self._active_bundle_request_ids: set[str] = set()
        self._pipelined_load_requests: dict[str, ReqMeta] = {}
        self._pipelined_load_remote_addresses: dict[str, str] = {}
        self._pipelined_transfer_layers: list[tuple[str, torch.Tensor]] = []
        self._pipelined_transfer_layer_indices: dict[str, int] = {}
        self._pipelined_attn_metadata: AttentionMetadata | None = None
        self._pipelined_loaded_bundles: set[tuple[str, int]] = set()
        self._pipelined_loaded_layers: set[tuple[str, str]] = set()
        self._pipelined_prefetch_cv = threading.Condition()
        self._pipelined_prefetch_queue: list[tuple[str, int]] = []
        self._pipelined_prefetch_inflight: set[tuple[str, int]] = set()
        self._pipelined_prefetched_bundles: dict[
            tuple[str, int],
            PrefetchedLayerBundle,
        ] = {}
        self._pipelined_prefetch_staged_bytes = 0
        self._pipelined_prefetch_thread: threading.Thread | None = None
        if self._pipelined_prefetch_enabled:
            self._pipelined_prefetch_thread = threading.Thread(
                target=self._pipelined_prefetch_worker,
                daemon=True,
            )
            self._pipelined_prefetch_thread.start()
        self._consumer_transfer_layer_order_logged = False
        self._producer_transfer_layer_order_logged = False
        self._producer_transfer_layer_order: list[str] = []
        self._group_block_sizes = tuple(
            int(group.kv_cache_spec.block_size)
            for group in kv_cache_config.kv_cache_groups
        )
        self.chunked_prefill: dict[
            str, tuple[tuple[list[int], ...], list[int] | None]
        ] = {}
        self._layer_name_to_group_index = {
            layer_name: group_index
            for group_index, group in enumerate(kv_cache_config.kv_cache_groups)
            for layer_name in group.layer_names
        }

        self._rank = get_world_group().rank if role == KVConnectorRole.WORKER else 0
        self._local_rank = (
            get_world_group().local_rank if role == KVConnectorRole.WORKER else 0
        )

        self.p2p_nccl_engine = (
            P2pNcclEngine(
                local_rank=self._local_rank,
                config=self._kv_transfer_config,
                hostname="",
                port_offset=self._rank,
            )
            if role == KVConnectorRole.WORKER
            else None
        )

    # ==============================
    # Worker-side methods
    # ==============================

    @staticmethod
    def _concat_block_ids_by_group(
        prefix: tuple[list[int], ...],
        suffix: tuple[list[int], ...],
    ) -> tuple[list[int], ...]:
        if len(prefix) != len(suffix):
            raise ValueError(
                "P2pNcclConnector received mismatched KV cache group counts: "
                f"{len(prefix)} != {len(suffix)}"
            )
        return tuple(
            prefix_group + suffix_group
            for prefix_group, suffix_group in zip(prefix, suffix)
        )

    @staticmethod
    def _block_count_for_token_count(token_count: int, block_size: int) -> int:
        """Get the number of KV blocks needed for a token count.

        :param token_count: Number of tokens represented by a transfer.
        :param block_size: Number of tokens represented by one KV block.
        :returns: Required KV block count.
        """

        if token_count <= 0:
            return 0
        return (token_count + block_size - 1) // block_size

    def _group_block_size(self, group_index: int) -> int:
        """Get the KV block size for a KV cache group.

        :param group_index: KV cache group index.
        :returns: Number of tokens represented by one KV block.
        """

        if group_index < len(self._group_block_sizes):
            return self._group_block_sizes[group_index]
        return self._block_size

    def _truncate_block_ids_by_token_count(
        self,
        block_ids: tuple[list[int], ...],
        token_count: int,
    ) -> tuple[list[int], ...]:
        """Trim block IDs to the blocks required by a token prefix.

        :param block_ids: Full per-group block IDs.
        :param token_count: Token prefix length.
        :returns: Per-group block IDs covering the token prefix.
        """

        return tuple(
            group_block_ids[
                : self._block_count_for_token_count(
                    token_count, self._group_block_size(group_index)
                )
            ]
            for group_index, group_block_ids in enumerate(block_ids)
        )

    def _external_token_ids_for_remote_decode(
        self,
        prompt_token_ids: list[int] | None,
    ) -> list[int]:
        """Get the prompt token prefix imported by the decode worker.

        :param prompt_token_ids: Full prompt token IDs.
        :returns: Token IDs whose KV blocks are transferred to decode.
        """

        if prompt_token_ids is None:
            return []
        if len(prompt_token_ids) == 0:
            return []
        return prompt_token_ids[:-1]

    def _add_producer_request_meta(
        self,
        meta: P2pNcclConnectorMetadata,
        request_id: str,
        prompt_token_ids: list[int] | None,
        block_ids: tuple[list[int], ...],
    ) -> None:
        """Add producer metadata for the KV prefix imported by decode.

        :param meta: Metadata object being built.
        :param request_id: Request identifier.
        :param prompt_token_ids: Full prompt token IDs.
        :param block_ids: Full per-group request block IDs.
        """

        transfer_token_ids = self._external_token_ids_for_remote_decode(
            prompt_token_ids
        )
        if len(transfer_token_ids) == 0:
            return
        transfer_block_ids = self._truncate_block_ids_by_token_count(
            block_ids,
            len(transfer_token_ids),
        )
        meta.add_request(
            request_id=request_id,
            token_ids=transfer_token_ids,
            block_ids=transfer_block_ids,
        )

    def _add_consumer_request_meta(
        self,
        meta: P2pNcclConnectorMetadata,
        request_id: str,
        token_ids: list[int],
        block_ids: tuple[list[int], ...],
        num_external_tokens: int,
    ) -> None:
        """Add consumer metadata for the KV prefix imported from prefill.

        :param meta: Metadata object being built.
        :param request_id: Request identifier.
        :param token_ids: Request token IDs containing the imported prefix.
        :param block_ids: Per-group block IDs allocated for imported KV.
        :param num_external_tokens: Number of prefix tokens imported from prefill.
        """

        if num_external_tokens <= 0:
            return
        if len(token_ids) < num_external_tokens:
            logger.error(
                "P2pNcclConnector cannot describe consumer KV import for "
                "request_id:%s because only %d token ids are available for %d "
                "external tokens",
                request_id,
                len(token_ids),
                num_external_tokens,
            )
            return
        transfer_token_ids = token_ids[:num_external_tokens]
        transfer_block_ids = self._truncate_block_ids_by_token_count(
            block_ids,
            num_external_tokens,
        )
        meta.add_request(
            request_id=request_id,
            token_ids=transfer_token_ids,
            block_ids=transfer_block_ids,
        )

    @staticmethod
    def _transfer_request_id(request_id: str) -> str:
        match = re.search(r"(.*_[0-9a-f]{32})(?:-.+)?$", request_id)
        if match is None:
            return request_id
        return match.group(1)

    def _tensor_id(self, request_id: str, layer_name: str) -> str:
        return f"{self._transfer_request_id(request_id)}#{layer_name}"

    def _bundle_tensor_id(self, request_id: str, bundle_index: int) -> str:
        return f"{self._transfer_request_id(request_id)}#__p2p_bundle_{bundle_index}"

    @staticmethod
    def _tensor_nbytes(tensor: torch.Tensor | None) -> int:
        if tensor is None:
            return 0
        return tensor.element_size() * tensor.numel()

    @staticmethod
    def _shape_numel(shape: torch.Size) -> int:
        numel = 1
        for dim in shape:
            numel *= int(dim)
        return numel

    def _is_layer_bundling_enabled(self) -> bool:
        return self._bundle_layer_count > 1

    def _record_timing(
        self,
        name: str,
        elapsed_s: float,
        nbytes: int = 0,
    ) -> None:
        if self._timing_enabled is False:
            return
        with self._timing_lock:
            stats = self._timing_stats.setdefault(
                name,
                {
                    "count": 0.0,
                    "seconds": 0.0,
                    "max_seconds": 0.0,
                    "bytes": 0.0,
                },
            )
            stats["count"] += 1.0
            stats["seconds"] += elapsed_s
            stats["max_seconds"] = max(stats["max_seconds"], elapsed_s)
            stats["bytes"] += float(nbytes)

    def _record_request_timing(
        self,
        request_id: str,
        name: str,
        elapsed_s: float,
        nbytes: int = 0,
    ) -> None:
        if self._request_timing_enabled is False:
            return
        transfer_request_id = self._transfer_request_id(request_id)
        with self._timing_lock:
            request_stats = self._request_timing_stats.setdefault(
                transfer_request_id,
                {},
            )
            stats = request_stats.setdefault(
                name,
                {
                    "count": 0.0,
                    "seconds": 0.0,
                    "max_seconds": 0.0,
                    "bytes": 0.0,
                },
            )
            stats["count"] += 1.0
            stats["seconds"] += elapsed_s
            stats["max_seconds"] = max(stats["max_seconds"], elapsed_s)
            stats["bytes"] += float(nbytes)

    def _emit_trace_event(
        self,
        event: str,
        request_id: str | None = None,
        tensor_id: str | None = None,
        nbytes: int = 0,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if self._trace_events_enabled is False:
            return

        payload: dict[str, Any] = {
            "event": event,
            "rank": self._rank,
            "role": "producer" if self.is_producer else "consumer",
            "time_ns": time.time_ns(),
        }
        if request_id is not None:
            payload["request_id"] = self._transfer_request_id(request_id)
        if tensor_id is not None:
            payload["tensor_id"] = tensor_id
        if nbytes > 0:
            payload["bytes"] = nbytes
        if extra is not None:
            payload.update(extra)

        logger.info(
            "P2P NCCL connector trace event %s",
            json.dumps(payload, sort_keys=True),
        )

    def _emit_request_milestone(
        self,
        request_id: str,
        milestone: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if self._request_timing_enabled is False:
            return

        payload: dict[str, Any] = {
            "event": "p2p_request_milestone",
            "milestone": milestone,
            "monotonic_ns": time.monotonic_ns(),
            "rank": self._rank,
            "request_id": self._transfer_request_id(request_id),
            "role": "producer" if self.is_producer else "consumer",
            "time_ns": time.time_ns(),
        }
        if extra is not None:
            payload.update(extra)

        logger.info(
            "P2P NCCL request milestone %s",
            json.dumps(payload, sort_keys=True),
        )

    def _emit_request_milestone_once(
        self,
        request_id: str,
        milestone: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if self._request_timing_enabled is False:
            return

        transfer_request_id = self._transfer_request_id(request_id)
        with self._timing_lock:
            milestone_names = self._request_milestone_once_events.setdefault(
                transfer_request_id,
                set(),
            )
            if milestone in milestone_names:
                return
            milestone_names.add(milestone)
        self._emit_request_milestone(request_id, milestone, extra)

    def _pop_timing_stats(self) -> dict[str, dict[str, float]]:
        if self._timing_enabled is False:
            return {}
        with self._timing_lock:
            stats = self._timing_stats
            self._timing_stats = {}
        return stats

    def _pop_request_timing_stats(
        self,
    ) -> dict[str, dict[str, dict[str, float]]]:
        if self._request_timing_enabled is False:
            return {}
        with self._timing_lock:
            stats = self._request_timing_stats
            self._request_timing_stats = {}
        return stats

    def _log_timing_stats(self, reason: str) -> None:
        stats = self._pop_timing_stats()
        if len(stats) == 0:
            return
        role = "producer" if self.is_producer else "consumer"
        logger.info(
            "P2P NCCL connector timing stats, role:%s, rank:%d, reason:%s, "
            "stats:%s",
            role,
            self._rank,
            reason,
            json.dumps(stats, sort_keys=True),
        )

    def _log_request_timing_stats(
        self,
        reason: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        stats = self._pop_request_timing_stats()
        if len(stats) == 0:
            return
        role = "producer" if self.is_producer else "consumer"
        for request_id, request_stats in sorted(stats.items()):
            payload: dict[str, Any] = {
                "event": "p2p_request_timing_stats",
                "rank": self._rank,
                "reason": reason,
                "request_id": request_id,
                "role": role,
                "stats": request_stats,
                "time_ns": time.time_ns(),
            }
            if extra is not None:
                payload.update(extra)
            logger.info(
                "P2P NCCL request timing stats %s",
                json.dumps(payload, sort_keys=True),
            )

    def _get_layer_block_ids(self, request: ReqMeta, layer_name: str) -> torch.Tensor:
        group_index = self._layer_name_to_group_index.get(layer_name)
        if group_index is None:
            group_index = self._get_shared_layer_group_index(layer_name)
        if group_index is None:
            raise KeyError(
                f"P2pNcclConnector has no KV cache group for layer {layer_name}"
            )
        if group_index >= len(request.block_ids_by_group):
            raise IndexError(
                "P2pNcclConnector metadata does not contain block ids for KV "
                f"cache group {group_index}"
            )
        return request.block_ids_by_group[group_index]

    def _record_kv_load_failure(self, request: ReqMeta, layer_name: str) -> None:
        block_ids = self._get_layer_block_ids(request, layer_name)
        self._block_ids_with_load_errors.update(
            int(block_id) for block_id in block_ids.tolist()
        )

    def _record_request_kv_load_failure(self, request: ReqMeta) -> None:
        for block_ids in request.block_ids_by_group:
            self._block_ids_with_load_errors.update(
                int(block_id) for block_id in block_ids.tolist()
            )

    def _track_pending_async_load_request(
        self,
        request: ReqMeta,
        now: float,
    ) -> None:
        if request.request_id not in self._pending_async_load_request_started_at:
            self._pending_async_load_request_started_at[request.request_id] = now
            self._emit_request_milestone_once(
                request.request_id,
                "consumer_pending_registered",
                extra={"num_tokens": request.num_tokens},
            )
            self._record_request_timing(
                request.request_id,
                "load_kv_pending_registered",
                0.0,
            )
        self._pending_async_load_requests[request.request_id] = request

    def _drop_pending_async_load_request(self, request_id: str) -> None:
        self._pending_async_load_requests.pop(request_id, None)
        self._pending_async_load_request_started_at.pop(request_id, None)
        self._pending_async_load_expected_tensor_ids.pop(request_id, None)
        self._pending_async_load_seen_tensor_ids.pop(request_id, None)
        self._pending_async_load_all_tensors_visible_at.pop(request_id, None)
        self._pending_async_load_once_events.pop(request_id, None)

    def _pending_async_load_elapsed_s(
        self,
        request_id: str,
        now: float,
    ) -> float:
        started_at = self._pending_async_load_request_started_at.get(request_id)
        if started_at is None:
            return 0.0
        return now - started_at

    def _expected_load_tensor_ids(
        self,
        request: ReqMeta,
        transfer_layers: list[tuple[str, torch.Tensor]],
    ) -> list[str]:
        if self._is_layer_bundling_enabled():
            bundle_count = (
                len(transfer_layers) + self._bundle_layer_count - 1
            ) // self._bundle_layer_count
            return [
                self._bundle_tensor_id(request.request_id, bundle_index)
                for bundle_index in range(bundle_count)
            ]
        return [
            self._tensor_id(request.request_id, layer_name)
            for layer_name, _ in transfer_layers
        ]

    def _record_async_load_event_once(
        self,
        request_id: str,
        name: str,
        elapsed_s: float,
    ) -> None:
        if self._request_timing_enabled is False:
            return
        events = self._pending_async_load_once_events.setdefault(request_id, set())
        if name in events:
            return
        events.add(name)
        self._record_request_timing(request_id, name, elapsed_s)

    def _record_expected_async_load_tensors_once(
        self,
        request_id: str,
        tensor_ids: list[str],
    ) -> None:
        if self._request_timing_enabled is False:
            return
        if request_id in self._pending_async_load_expected_tensor_ids:
            return
        self._pending_async_load_expected_tensor_ids[request_id] = tuple(tensor_ids)
        self._emit_request_milestone_once(
            request_id,
            "consumer_expected_tensors_registered",
            extra={"expected_tensor_count": len(tensor_ids)},
        )
        for _tensor_id in tensor_ids:
            self._record_request_timing(
                request_id,
                "load_kv_expected_tensor",
                0.0,
            )

    def _record_async_load_visible_tensors(
        self,
        request_id: str,
        tensor_ids: list[str],
        visible_tensor_ids: list[str],
        elapsed_s: float,
    ) -> None:
        if self._request_timing_enabled is False:
            return
        if len(visible_tensor_ids) == 0:
            return
        seen_tensor_ids = self._pending_async_load_seen_tensor_ids.setdefault(
            request_id,
            set(),
        )
        seen_count_before = len(seen_tensor_ids)
        for tensor_id in visible_tensor_ids:
            if tensor_id in seen_tensor_ids:
                continue
            seen_tensor_ids.add(tensor_id)
            self._record_request_timing(
                request_id,
                "load_kv_tensor_visible_after_pending",
                elapsed_s,
            )
            self._emit_request_milestone(
                request_id,
                "consumer_tensor_visible",
                extra={
                    "elapsed_seconds": elapsed_s,
                    "seen_tensor_count": len(seen_tensor_ids),
                    "tensor_id": tensor_id,
                    "total_tensor_count": len(tensor_ids),
                },
            )
        if seen_count_before == 0 and len(seen_tensor_ids) > 0:
            self._emit_request_milestone_once(
                request_id,
                "consumer_first_tensor_visible",
                extra={
                    "elapsed_seconds": elapsed_s,
                    "seen_tensor_count": len(seen_tensor_ids),
                    "total_tensor_count": len(tensor_ids),
                },
            )
        self._record_async_load_event_once(
            request_id,
            "load_kv_first_tensor_visible_after_pending",
            elapsed_s,
        )
        if len(seen_tensor_ids) == len(tensor_ids):
            self._pending_async_load_all_tensors_visible_at.setdefault(
                request_id,
                time.monotonic(),
            )
            self._emit_request_milestone_once(
                request_id,
                "consumer_all_tensors_visible",
                extra={
                    "elapsed_seconds": elapsed_s,
                    "total_tensor_count": len(tensor_ids),
                },
            )
            self._record_async_load_event_once(
                request_id,
                "load_kv_all_tensors_visible_after_pending",
                elapsed_s,
            )

    def _record_async_load_ready_visible_to_marked_ready(
        self,
        request_id: str,
        ready_now: float,
    ) -> None:
        visible_at = self._pending_async_load_all_tensors_visible_at.get(request_id)
        if visible_at is None:
            return
        self._record_async_load_event_once(
            request_id,
            "load_kv_all_tensors_visible_to_request_marked_ready",
            max(0.0, ready_now - visible_at),
        )

    def _record_async_load_budget_skip(
        self,
        request: ReqMeta,
        ready_loads_this_step: int,
        remaining_request_count: int,
    ) -> None:
        if self._request_timing_enabled is False:
            return
        ready_now = time.monotonic()
        visible_at = self._pending_async_load_all_tensors_visible_at.get(
            request.request_id,
        )
        ready_age_s = 0.0
        if visible_at is not None:
            ready_age_s = max(0.0, ready_now - visible_at)
        self._record_request_timing(
            request.request_id,
            "load_kv_ready_skipped_by_budget",
            0.0,
        )
        self._record_request_timing(
            request.request_id,
            "load_kv_ready_skipped_by_budget_age",
            ready_age_s,
        )
        self._emit_request_milestone(
            request.request_id,
            "consumer_ready_skipped_by_budget",
            extra={
                "ready_age_seconds": ready_age_s,
                "ready_loads_this_step": ready_loads_this_step,
                "remaining_request_count": remaining_request_count,
            },
        )

    def _record_async_load_cap_disabled_by_pending_threshold(
        self,
        requests_to_load: list[ReqMeta],
    ) -> None:
        if self._request_timing_enabled is False:
            return
        for request in requests_to_load:
            self._record_async_load_event_once(
                request.request_id,
                "load_kv_ready_cap_disabled_by_pending_threshold",
                0.0,
            )
            self._emit_request_milestone_once(
                request.request_id,
                "consumer_ready_cap_disabled_by_pending_threshold",
                extra={
                    "max_cap_pending_requests": (
                        self._load_kv_async_max_cap_pending_requests
                    ),
                    "max_ready_loads_per_step": (
                        self._load_kv_async_max_ready_loads_per_step
                    ),
                    "pending_request_count": len(requests_to_load),
                },
            )

    def _record_async_load_cap_disabled_by_ready_threshold(
        self,
        ready_requests: list[ReqMeta],
    ) -> None:
        if self._request_timing_enabled is False:
            return
        for request in ready_requests:
            self._record_async_load_event_once(
                request.request_id,
                "load_kv_ready_cap_disabled_by_ready_threshold",
                0.0,
            )
            self._emit_request_milestone_once(
                request.request_id,
                "consumer_ready_cap_disabled_by_ready_threshold",
                extra={
                    "max_cap_ready_requests": (
                        self._load_kv_async_max_cap_ready_requests
                    ),
                    "max_ready_loads_per_step": (
                        self._load_kv_async_max_ready_loads_per_step
                    ),
                    "ready_request_count": len(ready_requests),
                },
            )

    def _emit_async_load_step(
        self,
        requests_to_load: list[ReqMeta],
        ready_requests: list[ReqMeta],
        loaded_requests: list[ReqMeta],
        not_ready_request_count: int,
        timed_out_request_count: int,
        cap_disabled_by_pending_threshold: bool,
        cap_disabled_by_ready_threshold: bool,
    ) -> None:
        if self._request_timing_enabled is False:
            return
        ready_remaining_count = len(ready_requests) - len(loaded_requests)
        if ready_remaining_count < 0:
            ready_remaining_count = 0
        cap_configured = self._load_kv_async_max_ready_loads_per_step > 0
        cap_hit = cap_configured and ready_remaining_count > 0
        payload: dict[str, Any] = {
            "cap_configured": cap_configured,
            "cap_disabled_by_pending_threshold": (
                cap_disabled_by_pending_threshold
            ),
            "cap_disabled_by_ready_threshold": cap_disabled_by_ready_threshold,
            "cap_hit": cap_hit,
            "event": "p2p_async_load_step",
            "loaded_request_count": len(loaded_requests),
            "loop_returned_early": cap_hit,
            "max_cap_pending_requests": (
                self._load_kv_async_max_cap_pending_requests
            ),
            "max_cap_ready_requests": self._load_kv_async_max_cap_ready_requests,
            "max_ready_loads_per_step": (
                self._load_kv_async_max_ready_loads_per_step
            ),
            "monotonic_ns": time.monotonic_ns(),
            "not_ready_request_count": not_ready_request_count,
            "pending_request_count": len(requests_to_load),
            "rank": self._rank,
            "ready_remaining_count": ready_remaining_count,
            "ready_request_count": len(ready_requests),
            "role": "producer" if self.is_producer else "consumer",
            "time_ns": time.time_ns(),
            "timed_out_request_count": timed_out_request_count,
        }
        logger.info(
            "P2P NCCL async load step %s",
            json.dumps(payload, sort_keys=True),
        )

    def _async_load_requests_for_current_step(
        self,
        requests_to_load: list[ReqMeta],
        transfer_layers: list[tuple[str, torch.Tensor]],
        now: float,
    ) -> list[ReqMeta]:
        """Choose fully visible async requests to load in this scheduler pass.

        :param requests_to_load: Pending async remote-KV requests.
        :param transfer_layers: Decode-worker KV cache layers to receive.
        :param now: Current monotonic timestamp.
        :returns: Requests whose full remote KV should be loaded this pass.
        """

        ready_requests: list[ReqMeta] = []
        not_ready_request_count = 0
        timed_out_request_count = 0
        for request in requests_to_load:
            if self._request_load_tensors_available(request, transfer_layers):
                ready_requests.append(request)
                continue

            pending_s = self._pending_async_load_elapsed_s(
                request.request_id,
                now,
            )
            if pending_s >= self._load_kv_async_pending_timeout_s:
                logger.error(
                    "P2pNcclConnector remote-KV tensors did not arrive "
                    "before pending timeout, request_id:%s, pending_s:%.6f, "
                    "timeout_s:%.6f",
                    request.request_id,
                    pending_s,
                    self._load_kv_async_pending_timeout_s,
                )
                self._record_request_timing(
                    request.request_id,
                    "load_kv_pending_timeout",
                    pending_s,
                )
                self._record_request_kv_load_failure(request)
                self._drop_pending_async_load_request(request.request_id)
                timed_out_request_count += 1
                continue

            self._record_request_timing(
                request.request_id,
                "load_kv_not_ready",
                0.0,
            )
            not_ready_request_count += 1

        max_ready_loads_this_step = self._load_kv_async_max_ready_loads_per_step
        if max_ready_loads_this_step <= 0:
            self._emit_async_load_step(
                requests_to_load,
                ready_requests,
                ready_requests,
                not_ready_request_count,
                timed_out_request_count,
                cap_disabled_by_pending_threshold=False,
                cap_disabled_by_ready_threshold=False,
            )
            return ready_requests

        if (
            self._load_kv_async_max_cap_pending_requests > 0
            and len(requests_to_load) > self._load_kv_async_max_cap_pending_requests
        ):
            self._record_async_load_cap_disabled_by_pending_threshold(
                requests_to_load,
            )
            self._emit_async_load_step(
                requests_to_load,
                ready_requests,
                ready_requests,
                not_ready_request_count,
                timed_out_request_count,
                cap_disabled_by_pending_threshold=True,
                cap_disabled_by_ready_threshold=False,
            )
            return ready_requests

        if (
            self._load_kv_async_max_cap_ready_requests > 0
            and len(ready_requests) > self._load_kv_async_max_cap_ready_requests
        ):
            self._record_async_load_cap_disabled_by_ready_threshold(
                ready_requests,
            )
            self._emit_async_load_step(
                requests_to_load,
                ready_requests,
                ready_requests,
                not_ready_request_count,
                timed_out_request_count,
                cap_disabled_by_pending_threshold=False,
                cap_disabled_by_ready_threshold=True,
            )
            return ready_requests

        if len(ready_requests) <= max_ready_loads_this_step:
            self._emit_async_load_step(
                requests_to_load,
                ready_requests,
                ready_requests,
                not_ready_request_count,
                timed_out_request_count,
                cap_disabled_by_pending_threshold=False,
                cap_disabled_by_ready_threshold=False,
            )
            return ready_requests

        skipped_requests = ready_requests[max_ready_loads_this_step:]
        for request in skipped_requests:
            self._record_async_load_budget_skip(
                request,
                max_ready_loads_this_step,
                len(skipped_requests),
            )
        loaded_requests = ready_requests[:max_ready_loads_this_step]
        self._emit_async_load_step(
            requests_to_load,
            ready_requests,
            loaded_requests,
            not_ready_request_count,
            timed_out_request_count,
            cap_disabled_by_pending_threshold=False,
            cap_disabled_by_ready_threshold=False,
        )
        return loaded_requests

    def _request_load_tensors_available(
        self,
        request: ReqMeta,
        transfer_layers: list[tuple[str, torch.Tensor]],
    ) -> bool:
        assert self.p2p_nccl_engine is not None
        tensor_ids = self._expected_load_tensor_ids(request, transfer_layers)
        if len(tensor_ids) == 0:
            return False
        all_available = self.p2p_nccl_engine.has_tensors(tensor_ids)
        if self._request_timing_enabled is False:
            return all_available

        now = time.monotonic()
        elapsed_s = self._pending_async_load_elapsed_s(request.request_id, now)
        self._record_expected_async_load_tensors_once(
            request.request_id,
            tensor_ids,
        )
        if all_available:
            self._record_async_load_visible_tensors(
                request.request_id,
                tensor_ids,
                tensor_ids,
                elapsed_s,
            )
            return True

        visible_tensor_ids = [
            tensor_id
            for tensor_id in tensor_ids
            if self.p2p_nccl_engine.has_tensors([tensor_id]) is True
        ]
        self._record_async_load_visible_tensors(
            request.request_id,
            tensor_ids,
            visible_tensor_ids,
            elapsed_s,
        )
        return False

    def _wait_for_load_tensor_available(
        self,
        request: ReqMeta,
        tensor_id: str,
    ) -> bool:
        """Wait for a remote KV tensor to appear in the receive store.

        :param request: Request metadata.
        :param tensor_id: Remote KV tensor identifier.
        :returns: Whether the tensor appeared before the configured timeout.
        """

        assert self.p2p_nccl_engine is not None

        wait_started = time.perf_counter()
        deadline = time.monotonic() + self._load_kv_async_pending_timeout_s
        while self.p2p_nccl_engine.has_tensors([tensor_id]) is False:
            now = time.monotonic()
            if now >= deadline:
                elapsed_s = time.perf_counter() - wait_started
                self._record_request_timing(
                    request.request_id,
                    "load_kv_tensor_available_timeout",
                    elapsed_s,
                )
                return False
            time.sleep(min(0.001, max(0.0, deadline - now)))
        elapsed_s = time.perf_counter() - wait_started
        self._record_request_timing(
            request.request_id,
            "load_kv_wait_tensor_available",
            elapsed_s,
        )
        return True

    def _extracted_kv_shape(
        self,
        layer: torch.Tensor,
        block_ids: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
    ) -> torch.Size | None:
        if isinstance(attn_metadata, MLACommonMetadata) or layer.shape[1] == 2:
            return torch.Size((len(block_ids), *layer.shape[1:]))
        if layer.shape[0] == 2:
            return torch.Size((layer.shape[0], len(block_ids), *layer.shape[2:]))
        return None

    def _extract_kv_from_layer(
        self,
        layer: torch.Tensor,
        block_ids: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
    ) -> torch.Tensor | None:
        if isinstance(attn_metadata, MLACommonMetadata) or layer.shape[1] == 2:
            return layer[block_ids, ...]
        if layer.shape[0] == 2:
            return layer[:, block_ids, ...]
        return None

    def _inject_kv_into_layer(
        self,
        layer: torch.Tensor,
        kv_cache: torch.Tensor,
        block_ids: torch.Tensor,
        request_id: str,
        attn_metadata: AttentionMetadata | None,
    ) -> bool:
        if isinstance(attn_metadata, MLACommonMetadata) or layer.shape[1] == 2:
            num_block = kv_cache.shape[0]
            self.check_tensors_except_dim(layer, kv_cache, 0)
            if len(block_ids) == num_block:
                layer[block_ids, ...] = kv_cache
                return True
            logger.error(
                "P2pNcclConnector refusing to inject mismatched KV cache, "
                "block_ids:%d, num_block:%d, request_id:%s",
                len(block_ids),
                num_block,
                request_id,
            )
            return False

        if layer.shape[0] == 2:
            num_block = kv_cache.shape[1]
            self.check_tensors_except_dim(layer, kv_cache, 1)
            if len(block_ids) == num_block:
                layer[:, block_ids, ...] = kv_cache
                return True
            logger.error(
                "P2pNcclConnector refusing to inject mismatched KV cache, "
                "block_ids:%d, num_block:%d, request_id:%s",
                len(block_ids),
                num_block,
                request_id,
            )
            return False

        return False

    def _get_shared_target_layer_name(self, layer_name: str) -> str | None:
        static_context = self._vllm_config.compilation_config.static_forward_context
        layer = static_context.get(layer_name)
        if layer is None:
            return None
        target_layer_name = getattr(layer, "kv_sharing_target_layer_name", None)
        if isinstance(target_layer_name, str) is False:
            return None
        return target_layer_name

    def _is_kv_shared_layer(self, layer_name: str) -> bool:
        return self._get_shared_target_layer_name(layer_name) is not None

    def _get_shared_layer_group_index(self, layer_name: str) -> int | None:
        target_layer_name = self._get_shared_target_layer_name(layer_name)
        if target_layer_name is None:
            return None
        candidate_layer_names = [target_layer_name]
        if target_layer_name.startswith("language_model."):
            candidate_layer_names.append(
                target_layer_name.removeprefix("language_model.")
            )
        else:
            candidate_layer_names.append(f"language_model.{target_layer_name}")
        for candidate_layer_name in candidate_layer_names:
            group_index = self._layer_name_to_group_index.get(candidate_layer_name)
            if group_index is not None:
                self._layer_name_to_group_index[layer_name] = group_index
                return group_index
        return None

    def _transfer_layers_from_forward_context(
        self,
        forward_context: "ForwardContext",
    ) -> list[tuple[str, torch.Tensor]]:
        transfer_layers: list[tuple[str, torch.Tensor]] = []
        for layer_name in forward_context.no_compile_layers:
            layer = forward_context.no_compile_layers[layer_name]
            kv_cache = getattr(layer, "kv_cache", None)
            if kv_cache is None:
                continue
            if self._is_kv_shared_layer(layer_name):
                continue
            transfer_layers.append((layer_name, kv_cache))
        if self._consumer_transfer_layer_order_logged is False:
            self._consumer_transfer_layer_order_logged = True
            logger.info(
                "P2P NCCL transfer layer order, role:%s, rank:%d, layers:%s",
                "consumer",
                self._rank,
                [layer_name for layer_name, _ in transfer_layers],
            )
        return transfer_layers

    def _load_single_layer_tensor(
        self,
        request: ReqMeta,
        layer_name: str,
        layer: torch.Tensor,
        remote_address: str,
        attn_metadata: AttentionMetadata | None,
    ) -> bool:
        assert self.p2p_nccl_engine is not None

        kv_cache = self.p2p_nccl_engine.recv_tensor(
            self._tensor_id(request.request_id, layer_name), remote_address
        )

        if kv_cache is None:
            logger.warning("🚧kv_cache is None, %s", request.request_id)
            self._record_kv_load_failure(request, layer_name)
            return False

        inject_started = time.perf_counter()
        loaded = self._inject_kv_into_layer(
            layer,
            kv_cache,
            self._get_layer_block_ids(request, layer_name),
            request.request_id,
            attn_metadata,
        )
        self._record_timing(
            "load_kv_inject",
            time.perf_counter() - inject_started,
            self._tensor_nbytes(kv_cache),
        )
        if loaded is False:
            self._record_kv_load_failure(request, layer_name)
        return loaded

    def _load_layer_bundles(
        self,
        request: ReqMeta,
        remote_address: str,
        transfer_layers: list[tuple[str, torch.Tensor]],
        attn_metadata: AttentionMetadata | None,
    ) -> bool:
        bundle_count = (
            len(transfer_layers) + self._bundle_layer_count - 1
        ) // self._bundle_layer_count
        all_loaded = True
        for bundle_index in range(bundle_count):
            loaded = self._load_one_layer_bundle(
                request,
                remote_address,
                transfer_layers,
                attn_metadata,
                bundle_index,
                fail_closed=False,
            )
            if loaded is False:
                all_loaded = False
        return all_loaded

    def _load_one_layer_bundle(
        self,
        request: ReqMeta,
        remote_address: str,
        transfer_layers: list[tuple[str, torch.Tensor]],
        attn_metadata: AttentionMetadata | None,
        bundle_index: int,
        fail_closed: bool,
    ) -> bool:
        assert self.p2p_nccl_engine is not None

        plan = self._build_layer_bundle_load_plan(
            request,
            transfer_layers,
            attn_metadata,
            bundle_index,
        )
        if plan is None:
            return True
        self._emit_trace_event(
            "consumer_bundle_load_start",
            request_id=request.request_id,
            tensor_id=plan.tensor_id,
            extra={
                "bundle_index": bundle_index,
                "layer_names": plan.layer_names,
            },
        )
        if (
            fail_closed
            and self._wait_for_load_tensor_available(request, plan.tensor_id)
            is False
        ):
            self._fail_pipelined_load(
                request,
                plan.layer_names,
                f"timed out waiting for bundle {bundle_index}",
            )
        recv_started = time.perf_counter()
        bundle = self.p2p_nccl_engine.recv_tensor(plan.tensor_id, remote_address)
        self._record_request_timing(
            request.request_id,
            "load_kv_bundle_recv",
            time.perf_counter() - recv_started,
            plan.expected_nbytes,
        )
        return self._validate_and_inject_layer_bundle(
            request,
            plan,
            bundle,
            attn_metadata,
            fail_closed,
        )

    def _build_layer_bundle_load_plan(
        self,
        request: ReqMeta,
        transfer_layers: list[tuple[str, torch.Tensor]],
        attn_metadata: AttentionMetadata | None,
        bundle_index: int,
    ) -> LayerBundleLoadPlan | None:
        start = bundle_index * self._bundle_layer_count
        bundle_layers = transfer_layers[start : start + self._bundle_layer_count]
        if len(bundle_layers) == 0:
            return None

        expected_entries: list[
            tuple[str, torch.Tensor, torch.Tensor, torch.Size, int]
        ] = []
        expected_dtype: torch.dtype | None = None
        expected_numel = 0
        for layer_name, layer in bundle_layers:
            block_ids = self._get_layer_block_ids(request, layer_name)
            expected_shape = self._extracted_kv_shape(
                layer,
                block_ids,
                attn_metadata,
            )
            if expected_shape is None:
                raise NotImplementedError(
                    "P2pNcclConnector cannot bundle unsupported KV layout "
                    f"for layer {layer_name}"
                )
            if expected_dtype is None:
                expected_dtype = layer.dtype
            elif layer.dtype != expected_dtype:
                raise TypeError(
                    "P2pNcclConnector cannot bundle mixed KV dtypes: "
                    f"{layer.dtype} != {expected_dtype}"
                )
            numel = self._shape_numel(expected_shape)
            expected_entries.append(
                (layer_name, layer, block_ids, expected_shape, numel)
            )
            expected_numel += numel

        tensor_id = self._bundle_tensor_id(request.request_id, bundle_index)
        expected_nbytes = expected_numel * expected_entries[0][1].element_size()
        layer_names = [entry[0] for entry in expected_entries]
        assert expected_dtype is not None
        return LayerBundleLoadPlan(
            bundle_index=bundle_index,
            tensor_id=tensor_id,
            layer_names=layer_names,
            expected_entries=expected_entries,
            expected_dtype=expected_dtype,
            expected_numel=expected_numel,
            expected_nbytes=expected_nbytes,
        )

    def _validate_and_inject_layer_bundle(
        self,
        request: ReqMeta,
        plan: LayerBundleLoadPlan,
        bundle: torch.Tensor | None,
        attn_metadata: AttentionMetadata | None,
        fail_closed: bool,
    ) -> bool:
        if bundle is None:
            for layer_name in plan.layer_names:
                logger.warning(
                    "P2pNcclConnector received empty KV bundle for "
                    "request_id:%s, bundle_index:%d, layer_name:%s",
                    request.request_id,
                    plan.bundle_index,
                    layer_name,
                )
                self._record_kv_load_failure(request, layer_name)
            if fail_closed:
                self._fail_pipelined_load(
                    request,
                    plan.layer_names,
                    f"received empty bundle {plan.bundle_index}",
                )
            return False
        self._emit_trace_event(
            "consumer_bundle_received",
            request_id=request.request_id,
            tensor_id=plan.tensor_id,
            nbytes=self._tensor_nbytes(bundle),
            extra={
                "bundle_index": plan.bundle_index,
                "expected_numel": plan.expected_numel,
            },
        )
        if bundle.dtype != plan.expected_dtype:
            logger.error(
                "P2pNcclConnector received mismatched KV bundle dtype, "
                "request_id:%s, bundle_index:%d, dtype:%s, expected:%s",
                request.request_id,
                plan.bundle_index,
                bundle.dtype,
                plan.expected_dtype,
            )
            for layer_name in plan.layer_names:
                self._record_kv_load_failure(request, layer_name)
            if fail_closed:
                self._fail_pipelined_load(
                    request,
                    plan.layer_names,
                    f"received mismatched bundle {plan.bundle_index} dtype",
                )
            return False

        flat_bundle = bundle.reshape(-1)
        if flat_bundle.numel() != plan.expected_numel:
            logger.error(
                "P2pNcclConnector received mismatched KV bundle size, "
                "request_id:%s, bundle_index:%d, numel:%d, expected:%d",
                request.request_id,
                plan.bundle_index,
                flat_bundle.numel(),
                plan.expected_numel,
            )
            for layer_name in plan.layer_names:
                self._record_kv_load_failure(request, layer_name)
            if fail_closed:
                self._fail_pipelined_load(
                    request,
                    plan.layer_names,
                    f"received mismatched bundle {plan.bundle_index} size",
                )
            return False

        all_loaded = True
        offset = 0
        for (
            layer_name,
            layer,
            block_ids,
            expected_shape,
            numel,
        ) in plan.expected_entries:
            split_started = time.perf_counter()
            kv_cache = flat_bundle.narrow(0, offset, numel).view(expected_shape)
            split_elapsed_s = time.perf_counter() - split_started
            self._record_timing(
                "load_kv_bundle_split",
                split_elapsed_s,
                self._tensor_nbytes(kv_cache),
            )
            self._record_request_timing(
                request.request_id,
                "load_kv_bundle_split",
                split_elapsed_s,
                self._tensor_nbytes(kv_cache),
            )
            offset += numel

            inject_started = time.perf_counter()
            loaded = self._inject_kv_into_layer(
                layer,
                kv_cache,
                block_ids,
                request.request_id,
                attn_metadata,
            )
            inject_elapsed_s = time.perf_counter() - inject_started
            self._record_timing(
                "load_kv_inject",
                inject_elapsed_s,
                self._tensor_nbytes(kv_cache),
            )
            self._record_request_timing(
                request.request_id,
                "load_kv_inject",
                inject_elapsed_s,
                self._tensor_nbytes(kv_cache),
            )
            if loaded is False:
                self._record_kv_load_failure(request, layer_name)
                all_loaded = False
        if fail_closed and all_loaded is False:
            self._fail_pipelined_load(
                request,
                plan.layer_names,
                f"failed to inject bundle {plan.bundle_index}",
            )
        self._emit_trace_event(
            "consumer_bundle_injected",
            request_id=request.request_id,
            tensor_id=plan.tensor_id,
            nbytes=self._tensor_nbytes(bundle),
            extra={"bundle_index": plan.bundle_index},
        )
        return all_loaded

    def _reserve_pipelined_prefetch_bytes(
        self,
        request: ReqMeta,
        nbytes: int,
    ) -> PrefetchReservation:
        if self._pipelined_prefetch_enabled is False:
            return PrefetchReservation(
                succeeded=False,
                reason="pipelined prefetch is disabled",
            )
        if nbytes > self._pipelined_prefetch_max_staged_bytes:
            self._record_request_timing(
                request.request_id,
                "pipelined_prefetch_reserve_oversize",
                0.0,
                nbytes,
            )
            return PrefetchReservation(
                succeeded=False,
                reason=(
                    f"requested_nbytes:{nbytes}, "
                    f"max_staged_nbytes:"
                    f"{self._pipelined_prefetch_max_staged_bytes}"
                ),
                defer_to_sync=True,
            )

        wait_started = time.perf_counter()
        deadline = time.monotonic() + self._load_kv_async_pending_timeout_s
        with self._pipelined_prefetch_cv:
            if request.request_id not in self._pipelined_load_requests:
                return PrefetchReservation(
                    succeeded=False,
                    reason=(
                        "request is no longer active while reserving bytes, "
                        f"requested_nbytes:{nbytes}, "
                        f"staged_nbytes:"
                        f"{self._pipelined_prefetch_staged_bytes}, "
                        f"max_staged_nbytes:"
                        f"{self._pipelined_prefetch_max_staged_bytes}, "
                        f"queue_depth:{len(self._pipelined_prefetch_queue)}, "
                        f"inflight_bundles:"
                        f"{len(self._pipelined_prefetch_inflight)}, "
                        f"staged_bundles:"
                        f"{len(self._pipelined_prefetched_bundles)}"
                    ),
                )
            while (
                self._pipelined_prefetch_staged_bytes + nbytes
                > self._pipelined_prefetch_max_staged_bytes
            ):
                if request.request_id not in self._pipelined_load_requests:
                    elapsed_s = time.perf_counter() - wait_started
                    self._record_request_timing(
                        request.request_id,
                        "pipelined_prefetch_reserve_cancelled",
                        elapsed_s,
                        nbytes,
                    )
                    return PrefetchReservation(
                        succeeded=False,
                        reason=(
                            "request is no longer active while reserving bytes, "
                            f"requested_nbytes:{nbytes}, "
                            f"staged_nbytes:"
                            f"{self._pipelined_prefetch_staged_bytes}, "
                            f"max_staged_nbytes:"
                            f"{self._pipelined_prefetch_max_staged_bytes}, "
                            f"queue_depth:{len(self._pipelined_prefetch_queue)}, "
                            f"inflight_bundles:"
                            f"{len(self._pipelined_prefetch_inflight)}, "
                            f"staged_bundles:"
                            f"{len(self._pipelined_prefetched_bundles)}"
                        ),
                    )
                now = time.monotonic()
                if now >= deadline:
                    elapsed_s = time.perf_counter() - wait_started
                    self._record_request_timing(
                        request.request_id,
                        "pipelined_prefetch_reserve_timeout",
                        elapsed_s,
                        nbytes,
                    )
                    return PrefetchReservation(
                        succeeded=False,
                        reason=(
                            "timed out reserving staged bytes, "
                            f"wait_seconds:{elapsed_s:.6f}, "
                            f"requested_nbytes:{nbytes}, "
                            f"staged_nbytes:"
                            f"{self._pipelined_prefetch_staged_bytes}, "
                            f"max_staged_nbytes:"
                            f"{self._pipelined_prefetch_max_staged_bytes}, "
                            f"queue_depth:{len(self._pipelined_prefetch_queue)}, "
                            f"inflight_bundles:"
                            f"{len(self._pipelined_prefetch_inflight)}, "
                            f"staged_bundles:"
                            f"{len(self._pipelined_prefetched_bundles)}"
                        ),
                        defer_to_sync=True,
                    )
                self._pipelined_prefetch_cv.wait(
                    timeout=min(0.001, max(0.0, deadline - now))
                )
            self._pipelined_prefetch_staged_bytes += nbytes
            self._pipelined_prefetch_cv.notify_all()
        elapsed_s = time.perf_counter() - wait_started
        self._record_request_timing(
            request.request_id,
            "pipelined_prefetch_reserve",
            elapsed_s,
            nbytes,
        )
        return PrefetchReservation(succeeded=True)

    def _release_pipelined_prefetch_bytes(self, nbytes: int) -> None:
        if nbytes <= 0:
            return
        with self._pipelined_prefetch_cv:
            self._release_pipelined_prefetch_bytes_locked(nbytes)

    def _release_pipelined_prefetch_bytes_locked(self, nbytes: int) -> None:
        if nbytes <= 0:
            return
        self._pipelined_prefetch_staged_bytes = max(
            0,
            self._pipelined_prefetch_staged_bytes - nbytes,
        )
        self._pipelined_prefetch_cv.notify_all()

    def _release_prefetched_layer_bundle(
        self,
        prefetched: PrefetchedLayerBundle,
    ) -> None:
        self._release_pipelined_prefetch_bytes(prefetched.staged_nbytes)
        prefetched.staged_nbytes = 0

    def _queue_pipelined_bundle_prefetch(
        self,
        request_id: str,
        bundle_index: int,
    ) -> None:
        if self._pipelined_prefetch_enabled is False:
            return
        if request_id not in self._pipelined_load_requests:
            return
        if bundle_index < 0 or bundle_index >= self._pipelined_bundle_count():
            return

        key = (request_id, bundle_index)
        with self._pipelined_prefetch_cv:
            if key in self._pipelined_prefetched_bundles:
                return
            if key in self._pipelined_prefetch_inflight:
                return
            if key in self._pipelined_loaded_bundles:
                return
            self._pipelined_prefetch_inflight.add(key)
            self._pipelined_prefetch_queue.append(key)
            self._pipelined_prefetch_cv.notify()
        self._record_request_timing(
            request_id,
            "pipelined_prefetch_queued",
            0.0,
        )

    def _pipelined_prefetch_worker(self) -> None:
        while True:
            with self._pipelined_prefetch_cv:
                while len(self._pipelined_prefetch_queue) == 0:
                    self._pipelined_prefetch_cv.wait()
                request_id, bundle_index = self._pipelined_prefetch_queue.pop(0)

            key = (request_id, bundle_index)
            result = self._prefetch_layer_bundle(request_id, bundle_index)
            with self._pipelined_prefetch_cv:
                self._pipelined_prefetch_inflight.discard(key)
                if request_id in self._pipelined_load_requests:
                    self._pipelined_prefetched_bundles[key] = result
                else:
                    self._release_pipelined_prefetch_bytes_locked(
                        result.staged_nbytes
                    )
                    result.staged_nbytes = 0
                self._pipelined_prefetch_cv.notify_all()

    def _wait_for_prefetch_tensor_available(
        self,
        request: ReqMeta,
        tensor_id: str,
    ) -> bool:
        assert self.p2p_nccl_engine is not None

        wait_started = time.perf_counter()
        deadline = time.monotonic() + self._load_kv_async_pending_timeout_s
        while self.p2p_nccl_engine.has_tensors([tensor_id]) is False:
            if request.request_id not in self._pipelined_load_requests:
                elapsed_s = time.perf_counter() - wait_started
                self._record_request_timing(
                    request.request_id,
                    "pipelined_prefetch_cancelled",
                    elapsed_s,
                )
                return False
            now = time.monotonic()
            if now >= deadline:
                elapsed_s = time.perf_counter() - wait_started
                self._record_request_timing(
                    request.request_id,
                    "pipelined_prefetch_tensor_available_timeout",
                    elapsed_s,
                )
                return False
            time.sleep(min(0.001, max(0.0, deadline - now)))
        elapsed_s = time.perf_counter() - wait_started
        self._record_request_timing(
            request.request_id,
            "pipelined_prefetch_wait_tensor_available",
            elapsed_s,
        )
        return True

    def _prefetch_layer_bundle(
        self,
        request_id: str,
        bundle_index: int,
    ) -> PrefetchedLayerBundle:
        assert self.p2p_nccl_engine is not None

        request = self._pipelined_load_requests.get(request_id)
        remote_address = self._pipelined_load_remote_addresses.get(request_id)
        if request is None:
            return PrefetchedLayerBundle(
                plan=None,
                bundle=None,
                error="request is no longer active",
            )
        if remote_address is None:
            return PrefetchedLayerBundle(
                plan=None,
                bundle=None,
                error="missing remote address",
            )

        reserved_nbytes = 0
        try:
            plan = self._build_layer_bundle_load_plan(
                request,
                self._pipelined_transfer_layers,
                self._pipelined_attn_metadata,
                bundle_index,
            )
            if plan is None:
                return PrefetchedLayerBundle(
                    plan=None,
                    bundle=None,
                    error=f"empty bundle {bundle_index}",
                )
            if (
                self._wait_for_prefetch_tensor_available(request, plan.tensor_id)
                is False
            ):
                return PrefetchedLayerBundle(
                    plan=plan,
                    bundle=None,
                    error=f"timed out waiting for bundle {bundle_index}",
                )
            reservation = self._reserve_pipelined_prefetch_bytes(
                request,
                plan.expected_nbytes,
            )
            if reservation.succeeded is False:
                logger.error(
                    "P2pNcclConnector pipelined prefetch reserve failed, "
                    "request_id:%s, bundle_index:%d, defer_to_sync:%s, "
                    "reason:%s",
                    request_id,
                    bundle_index,
                    reservation.defer_to_sync,
                    reservation.reason,
                )
                if reservation.defer_to_sync:
                    return PrefetchedLayerBundle(
                        plan=plan,
                        bundle=None,
                        error=reservation.reason,
                        defer_to_sync=True,
                    )
                return PrefetchedLayerBundle(
                    plan=plan,
                    bundle=None,
                    error=(
                        f"failed to reserve staged bytes for bundle {bundle_index}: "
                        f"{reservation.reason}"
                    ),
                )
            reserved_nbytes = plan.expected_nbytes
            recv_started = time.perf_counter()
            bundle = self.p2p_nccl_engine.recv_tensor(
                plan.tensor_id,
                remote_address,
            )
            recv_elapsed_s = time.perf_counter() - recv_started
            self._record_request_timing(
                request_id,
                "pipelined_prefetch_bundle_recv",
                recv_elapsed_s,
                plan.expected_nbytes,
            )
            if bundle is None:
                self._release_pipelined_prefetch_bytes(reserved_nbytes)
                return PrefetchedLayerBundle(
                    plan=plan,
                    bundle=None,
                    error=f"received empty bundle {bundle_index}",
                )
            flat_bundle = bundle.reshape(-1)
            if bundle.dtype != plan.expected_dtype:
                self._release_pipelined_prefetch_bytes(reserved_nbytes)
                return PrefetchedLayerBundle(
                    plan=plan,
                    bundle=None,
                    error=f"received mismatched bundle {bundle_index} dtype",
                )
            if flat_bundle.numel() != plan.expected_numel:
                self._release_pipelined_prefetch_bytes(reserved_nbytes)
                return PrefetchedLayerBundle(
                    plan=plan,
                    bundle=None,
                    error=f"received mismatched bundle {bundle_index} size",
                )
            self._record_request_timing(
                request_id,
                "pipelined_prefetch_staged",
                0.0,
                self._tensor_nbytes(bundle),
            )
            return PrefetchedLayerBundle(
                plan=plan,
                bundle=bundle,
                staged_nbytes=reserved_nbytes,
            )
        except Exception:
            self._release_pipelined_prefetch_bytes(reserved_nbytes)
            error = traceback.format_exc()
            logger.error(
                "P2pNcclConnector pipelined prefetch failed, "
                "request_id:%s, bundle_index:%d, traceback:%s",
                request_id,
                bundle_index,
                error,
            )
            return PrefetchedLayerBundle(plan=None, bundle=None, error=error)

    def _wait_for_prefetched_layer_bundle(
        self,
        request: ReqMeta,
        bundle_index: int,
    ) -> PrefetchedLayerBundle:
        key = (request.request_id, bundle_index)
        self._queue_pipelined_bundle_prefetch(request.request_id, bundle_index)

        wait_started = time.perf_counter()
        deadline = time.monotonic() + self._load_kv_async_pending_timeout_s
        with self._pipelined_prefetch_cv:
            while key not in self._pipelined_prefetched_bundles:
                if request.request_id not in self._pipelined_load_requests:
                    elapsed_s = time.perf_counter() - wait_started
                    self._record_request_timing(
                        request.request_id,
                        "pipelined_prefetch_wait_cancelled",
                        elapsed_s,
                    )
                    return PrefetchedLayerBundle(
                        plan=None,
                        bundle=None,
                        error="request is no longer active",
                    )
                now = time.monotonic()
                if now >= deadline:
                    elapsed_s = time.perf_counter() - wait_started
                    self._record_request_timing(
                        request.request_id,
                        "pipelined_prefetch_wait_timeout",
                        elapsed_s,
                    )
                    return PrefetchedLayerBundle(
                        plan=None,
                        bundle=None,
                        error=f"timed out waiting for bundle {bundle_index}",
                    )
                self._pipelined_prefetch_cv.wait(
                    timeout=min(0.001, max(0.0, deadline - now))
                )
            result = self._pipelined_prefetched_bundles.pop(key)
        elapsed_s = time.perf_counter() - wait_started
        self._record_request_timing(
            request.request_id,
            "pipelined_prefetch_wait",
            elapsed_s,
            0 if result.bundle is None else self._tensor_nbytes(result.bundle),
        )
        return result

    def _drop_pipelined_load_request(self, request_id: str) -> None:
        self._pipelined_load_requests.pop(request_id, None)
        self._pipelined_load_remote_addresses.pop(request_id, None)
        self._pipelined_loaded_bundles = {
            key for key in self._pipelined_loaded_bundles if key[0] != request_id
        }
        self._pipelined_loaded_layers = {
            key for key in self._pipelined_loaded_layers if key[0] != request_id
        }
        with self._pipelined_prefetch_cv:
            self._pipelined_prefetch_queue = [
                key
                for key in self._pipelined_prefetch_queue
                if key[0] != request_id
            ]
            self._pipelined_prefetch_inflight = {
                key
                for key in self._pipelined_prefetch_inflight
                if key[0] != request_id
            }
            released_nbytes = sum(
                value.staged_nbytes
                for key, value in self._pipelined_prefetched_bundles.items()
                if key[0] == request_id
            )
            self._pipelined_prefetched_bundles = {
                key: value
                for key, value in self._pipelined_prefetched_bundles.items()
                if key[0] != request_id
            }
            self._pipelined_prefetch_staged_bytes = max(
                0,
                self._pipelined_prefetch_staged_bytes - released_nbytes,
            )
            self._pipelined_prefetch_cv.notify_all()

    def _fail_pipelined_load(
        self,
        request: ReqMeta,
        layer_names: list[str],
        reason: str,
    ) -> None:
        if len(layer_names) == 0:
            self._record_request_kv_load_failure(request)
        for layer_name in layer_names:
            self._record_kv_load_failure(request, layer_name)
        self._drop_pipelined_load_request(request.request_id)
        raise RuntimeError(
            "P2pNcclConnector pipelined KV load failed before attention, "
            f"request_id:{request.request_id}, reason:{reason}"
        )

    def _prepare_pipelined_load(
        self,
        metadata: P2pNcclConnectorMetadata,
        transfer_layers: list[tuple[str, torch.Tensor]],
        attn_metadata: AttentionMetadata | None,
    ) -> None:
        self._pipelined_transfer_layers = transfer_layers
        self._pipelined_transfer_layer_indices = {
            layer_name: index for index, (layer_name, _) in enumerate(transfer_layers)
        }
        self._pipelined_attn_metadata = attn_metadata
        registered_request_ids: list[str] = []
        for request in metadata.requests:
            request_id = request.request_id
            ip, port = self.parse_request_id(request_id, False)
            self._pipelined_load_requests[request_id] = request
            self._pipelined_load_remote_addresses[request_id] = (
                ip + ":" + str(port + self._rank)
            )
            self._record_request_timing(
                request_id,
                "pipelined_load_registered",
                0.0,
            )
            registered_request_ids.append(request_id)
        if self._pipelined_prefetch_enabled:
            for bundle_index in range(self._pipelined_prefetch_window):
                for request_id in registered_request_ids:
                    self._queue_pipelined_bundle_prefetch(
                        request_id,
                        bundle_index,
                    )

    def _pipelined_bundle_count(self) -> int:
        if len(self._pipelined_transfer_layers) == 0:
            return 0
        return (
            len(self._pipelined_transfer_layers) + self._bundle_layer_count - 1
        ) // self._bundle_layer_count

    def _is_pipelined_request_complete(self, request_id: str) -> bool:
        if self._is_layer_bundling_enabled():
            bundle_count = self._pipelined_bundle_count()
            for bundle_index in range(bundle_count):
                if (request_id, bundle_index) not in self._pipelined_loaded_bundles:
                    return False
            return True
        for layer_name in self._pipelined_transfer_layer_indices:
            if (request_id, layer_name) not in self._pipelined_loaded_layers:
                return False
        return True

    def _queue_layer_bundle(
        self,
        request_id: str,
        layer_name: str,
        kv_cache: torch.Tensor,
        remote_address: str,
    ) -> None:
        assert self.p2p_nccl_engine is not None

        existing_remote_address = self._pending_bundle_remote_addresses.get(request_id)
        if (
            existing_remote_address is not None
            and existing_remote_address != remote_address
        ):
            raise ValueError(
                "P2pNcclConnector received multiple remote addresses for one "
                f"request bundle: {existing_remote_address} != {remote_address}"
            )

        self._pending_bundle_remote_addresses[request_id] = remote_address
        self.p2p_nccl_engine.prewarm_remote(remote_address)
        self._active_bundle_request_ids.add(request_id)
        entries = self._pending_layer_bundles.setdefault(request_id, [])
        entries.append(LayerBundleEntry(layer_name=layer_name, tensor=kv_cache))
        if len(entries) >= self._bundle_layer_count:
            self._flush_layer_bundle(request_id)

    def _flush_layer_bundle(self, request_id: str) -> None:
        assert self.p2p_nccl_engine is not None

        entries = self._pending_layer_bundles.pop(request_id, [])
        if len(entries) == 0:
            return
        remote_address = self._pending_bundle_remote_addresses.get(request_id)
        if remote_address is None:
            raise ValueError(
                "P2pNcclConnector cannot flush a KV bundle without a remote "
                f"address for request_id:{request_id}"
            )

        dtype = entries[0].tensor.dtype
        for entry in entries:
            if entry.tensor.dtype != dtype:
                raise TypeError(
                    "P2pNcclConnector cannot bundle mixed KV dtypes: "
                    f"{entry.tensor.dtype} != {dtype}"
                )

        bundle_index = self._next_bundle_indices.get(request_id, 0)
        self._next_bundle_indices[request_id] = bundle_index + 1
        tensor_id = self._bundle_tensor_id(request_id, bundle_index)
        layer_names = [entry.layer_name for entry in entries]
        self._emit_request_milestone(
            request_id,
            "producer_bundle_pack_start",
            extra={
                "bundle_index": bundle_index,
                "layer_count": len(layer_names),
            },
        )
        self._emit_trace_event(
            "producer_bundle_pack_start",
            request_id=request_id,
            tensor_id=tensor_id,
            extra={"bundle_index": bundle_index, "layer_names": layer_names},
        )
        pack_started = time.perf_counter()
        if len(entries) == 1:
            bundle = entries[0].tensor.reshape(-1).contiguous()
        else:
            bundle = torch.cat([entry.tensor.reshape(-1) for entry in entries])
        bundle_nbytes = self._tensor_nbytes(bundle)
        pack_elapsed_s = time.perf_counter() - pack_started
        self._record_timing(
            "save_kv_bundle_pack",
            pack_elapsed_s,
            bundle_nbytes,
        )
        self._record_request_timing(
            request_id,
            "save_kv_bundle_pack",
            pack_elapsed_s,
            bundle_nbytes,
        )
        self._emit_request_milestone(
            request_id,
            "producer_bundle_pack_done",
            extra={
                "bundle_index": bundle_index,
                "bytes": bundle_nbytes,
                "elapsed_seconds": pack_elapsed_s,
                "layer_count": len(layer_names),
            },
        )
        self._emit_trace_event(
            "producer_bundle_pack_done",
            request_id=request_id,
            tensor_id=tensor_id,
            nbytes=bundle_nbytes,
            extra={"bundle_index": bundle_index, "layer_names": layer_names},
        )

        enqueue_started = time.perf_counter()
        sent = self.p2p_nccl_engine.send_tensor(
            tensor_id, bundle, remote_address
        )
        enqueue_elapsed_s = time.perf_counter() - enqueue_started
        self._record_timing(
            "save_kv_bundle_send_tensor",
            enqueue_elapsed_s,
            bundle_nbytes,
        )
        self._record_request_timing(
            request_id,
            "save_kv_bundle_send_tensor",
            enqueue_elapsed_s,
            bundle_nbytes,
        )
        self._emit_request_milestone(
            request_id,
            "producer_bundle_send_returned",
            extra={
                "bundle_index": bundle_index,
                "bytes": bundle_nbytes,
                "elapsed_seconds": enqueue_elapsed_s,
                "send_type": self._kv_transfer_config.get_from_extra_config(
                    "send_type", "PUT_ASYNC"
                ),
                "sent": sent,
            },
        )
        self._emit_trace_event(
            "producer_bundle_send_tensor_returned",
            request_id=request_id,
            tensor_id=tensor_id,
            nbytes=bundle_nbytes,
            extra={
                "bundle_index": bundle_index,
                "sent": sent,
                "send_type": self._kv_transfer_config.get_from_extra_config(
                    "send_type", "PUT_ASYNC"
                ),
            },
        )
        if sent is False:
            logger.error(
                "P2pNcclConnector failed to enqueue/send KV bundle for "
                "request_id:%s, bundle_index:%d, layers:%s",
                self._transfer_request_id(request_id),
                bundle_index,
                layer_names,
            )

    def _flush_all_layer_bundles(self) -> list[str]:
        flushed_request_ids: list[str] = []
        for request_id in list(self._active_bundle_request_ids):
            self._flush_layer_bundle(request_id)
            bundle_count = self._next_bundle_indices.get(request_id, 0)
            self._emit_request_milestone(
                request_id,
                "producer_all_bundles_flushed",
                extra={"bundle_count": bundle_count},
            )
            flushed_request_ids.append(request_id)
            self._pending_bundle_remote_addresses.pop(request_id, None)
            self._next_bundle_indices.pop(request_id, None)
        self._active_bundle_request_ids.clear()
        return flushed_request_ids

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        """Start loading the KV cache from the connector buffer to vLLM's
        paged KV buffer.

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation

        Note:
            The number of elements in kv_caches and layer_names should be
            the same.
        """

        # Only consumer/decode loads KV Cache
        if self.is_producer:
            return

        assert self.p2p_nccl_engine is not None

        attn_metadata = forward_context.attn_metadata

        # Get the metadata
        metadata: KVConnectorMetadata = self._get_connector_metadata()
        assert isinstance(metadata, P2pNcclConnectorMetadata)

        if metadata is None:
            return

        now = time.monotonic()
        if self._load_kv_async_enabled:
            for request in metadata.requests:
                self._track_pending_async_load_request(request, now)
            requests_to_load = list(self._pending_async_load_requests.values())
        else:
            requests_to_load = metadata.requests

        transfer_layers = self._transfer_layers_from_forward_context(forward_context)
        if len(requests_to_load) > 0 and len(transfer_layers) == 0:
            for request in requests_to_load:
                logger.error(
                    "P2pNcclConnector found no transfer layers for remote-KV "
                    "load, request_id:%s",
                    request.request_id,
                )
                self._record_request_kv_load_failure(request)
                self._drop_pending_async_load_request(request.request_id)
                self._drop_pipelined_load_request(request.request_id)
            if (
                self._pipelined_load_enabled
                and self._load_kv_async_enabled is False
            ):
                raise RuntimeError(
                    "P2pNcclConnector cannot pipeline remote-KV load because "
                    "the forward context has no transfer layers"
                )
            return

        if (
            self._pipelined_load_enabled
            and self._load_kv_async_enabled is False
        ):
            if len(requests_to_load) > 0:
                self._prepare_pipelined_load(
                    metadata,
                    transfer_layers,
                    attn_metadata,
                )
            return

        if self._load_kv_async_enabled:
            requests_to_load = self._async_load_requests_for_current_step(
                requests_to_load,
                transfer_layers,
                now,
            )

        # Load the KV for each request each layer
        for request in requests_to_load:
            request_id = request.request_id
            self._emit_request_milestone(
                request_id,
                "consumer_load_kv_start",
                extra={"num_transfer_layers": len(transfer_layers)},
            )
            ip, port = self.parse_request_id(request_id, False)
            remote_address = ip + ":" + str(port + self._rank)
            if self._is_layer_bundling_enabled():
                loaded = self._load_layer_bundles(
                    request,
                    remote_address,
                    transfer_layers,
                    attn_metadata,
                )
                if self._load_kv_async_enabled and loaded:
                    ready_now = time.monotonic()
                    ready_elapsed_s = self._pending_async_load_elapsed_s(
                        request_id,
                        ready_now,
                    )
                    self._record_async_load_event_once(
                        request_id,
                        "load_kv_request_marked_ready_after_pending",
                        ready_elapsed_s,
                    )
                    self._record_async_load_ready_visible_to_marked_ready(
                        request_id,
                        ready_now,
                    )
                    self._emit_request_milestone_once(
                        request_id,
                        "consumer_request_marked_ready",
                        extra={"elapsed_seconds": ready_elapsed_s},
                    )
                    self._finished_recving_request_ids.add(request_id)
                    self._drop_pending_async_load_request(request_id)
                elif self._load_kv_async_enabled:
                    logger.error(
                        "P2pNcclConnector did not mark request ready after "
                        "failed bundled KV load, request_id:%s",
                        request_id,
                    )
                    self._drop_pending_async_load_request(request_id)
                continue
            loaded_all = True
            for layer_name, layer in transfer_layers:
                loaded = self._load_single_layer_tensor(
                    request,
                    layer_name,
                    layer,
                    remote_address,
                    attn_metadata,
                )
                if loaded is False:
                    loaded_all = False
            if self._load_kv_async_enabled and loaded_all:
                ready_now = time.monotonic()
                ready_elapsed_s = self._pending_async_load_elapsed_s(
                    request_id,
                    ready_now,
                )
                self._record_async_load_event_once(
                    request_id,
                    "load_kv_request_marked_ready_after_pending",
                    ready_elapsed_s,
                )
                self._record_async_load_ready_visible_to_marked_ready(
                    request_id,
                    ready_now,
                )
                self._emit_request_milestone_once(
                    request_id,
                    "consumer_request_marked_ready",
                    extra={"elapsed_seconds": ready_elapsed_s},
                )
                self._finished_recving_request_ids.add(request_id)
                self._drop_pending_async_load_request(request_id)
            elif self._load_kv_async_enabled:
                logger.error(
                    "P2pNcclConnector did not mark request ready after failed "
                    "KV load, request_id:%s",
                    request_id,
                )
                self._drop_pending_async_load_request(request_id)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Blocking until the KV for a specific layer is loaded into vLLM's
        paged buffer.

        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer
        """
        if self.is_producer:
            return
        if self._pipelined_load_enabled is False:
            return
        if self._load_kv_async_enabled:
            return
        if len(self._pipelined_load_requests) == 0:
            return
        layer_index = self._pipelined_transfer_layer_indices.get(layer_name)
        if layer_index is None:
            return

        if self._is_layer_bundling_enabled():
            bundle_index = layer_index // self._bundle_layer_count
            bundle_start = bundle_index * self._bundle_layer_count
            bundle_layers = self._pipelined_transfer_layers[
                bundle_start : bundle_start + self._bundle_layer_count
            ]
            bundle_layer_names = [
                bundle_layer_name for bundle_layer_name, _ in bundle_layers
            ]
            for request_id, request in list(self._pipelined_load_requests.items()):
                hook_started = time.perf_counter()
                bundle_key = (request_id, bundle_index)
                if bundle_key in self._pipelined_loaded_bundles:
                    continue
                remote_address = self._pipelined_load_remote_addresses.get(request_id)
                if remote_address is None:
                    self._fail_pipelined_load(
                        request,
                        bundle_layer_names,
                        "missing remote address",
                    )
                if self._pipelined_prefetch_enabled:
                    prefetched = self._wait_for_prefetched_layer_bundle(
                        request,
                        bundle_index,
                    )
                    if prefetched.defer_to_sync:
                        self._record_request_timing(
                            request_id,
                            "pipelined_prefetch_sync_fallback",
                            0.0,
                        )
                        loaded = self._load_one_layer_bundle(
                            request,
                            remote_address,
                            self._pipelined_transfer_layers,
                            self._pipelined_attn_metadata,
                            bundle_index,
                            fail_closed=True,
                        )
                    elif prefetched.error is not None or prefetched.plan is None:
                        self._fail_pipelined_load(
                            request,
                            bundle_layer_names,
                            prefetched.error or "missing prefetched bundle",
                        )
                    else:
                        try:
                            loaded = self._validate_and_inject_layer_bundle(
                                request,
                                prefetched.plan,
                                prefetched.bundle,
                                self._pipelined_attn_metadata,
                                fail_closed=True,
                            )
                        finally:
                            self._release_prefetched_layer_bundle(prefetched)
                else:
                    loaded = self._load_one_layer_bundle(
                        request,
                        remote_address,
                        self._pipelined_transfer_layers,
                        self._pipelined_attn_metadata,
                        bundle_index,
                        fail_closed=True,
                    )
                if loaded is False:
                    self._fail_pipelined_load(
                        request,
                        bundle_layer_names,
                        f"failed to load bundle {bundle_index}",
                    )
                self._pipelined_loaded_bundles.add(bundle_key)
                for bundle_layer_name in bundle_layer_names:
                    self._pipelined_loaded_layers.add(
                        (request_id, bundle_layer_name)
                    )
                if self._pipelined_prefetch_enabled:
                    self._queue_pipelined_bundle_prefetch(
                        request_id,
                        bundle_index + self._pipelined_prefetch_window,
                    )
                if self._is_pipelined_request_complete(request_id):
                    self._drop_pipelined_load_request(request_id)
                self._record_request_timing(
                    request_id,
                    "pipelined_wait_for_layer_load",
                    time.perf_counter() - hook_started,
                )
            return

        for request_id, request in list(self._pipelined_load_requests.items()):
            layer_key = (request_id, layer_name)
            if layer_key in self._pipelined_loaded_layers:
                continue
            remote_address = self._pipelined_load_remote_addresses.get(request_id)
            if remote_address is None:
                self._fail_pipelined_load(
                    request,
                    [layer_name],
                    "missing remote address",
                )
            tensor_id = self._tensor_id(request_id, layer_name)
            if self._wait_for_load_tensor_available(request, tensor_id) is False:
                self._fail_pipelined_load(
                    request,
                    [layer_name],
                    f"timed out waiting for layer {layer_name}",
                )
            layer = self._pipelined_transfer_layers[layer_index][1]
            loaded = self._load_single_layer_tensor(
                request,
                layer_name,
                layer,
                remote_address,
                self._pipelined_attn_metadata,
            )
            if loaded is False:
                self._fail_pipelined_load(
                    request,
                    [layer_name],
                    f"failed to load layer {layer_name}",
                )
            self._pipelined_loaded_layers.add(layer_key)
            if self._is_pipelined_request_complete(request_id):
                self._drop_pipelined_load_request(request_id)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        """Start saving the KV cache of the layer from vLLM's paged buffer
        to the connector.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """

        # Only producer/prefill saves KV Cache
        if not self.is_producer:
            return

        assert self.p2p_nccl_engine is not None
        if self._is_kv_shared_layer(layer_name):
            return

        connector_metadata = self._get_connector_metadata()
        assert isinstance(connector_metadata, P2pNcclConnectorMetadata)
        if (
            self._producer_transfer_layer_order_logged is False
            and len(connector_metadata.requests) > 0
        ):
            self._producer_transfer_layer_order.append(layer_name)
        for request in connector_metadata.requests:
            request_id = request.request_id
            ip, port = self.parse_request_id(request_id, True)
            remote_address = ip + ":" + str(port + self._rank)

            layer_block_ids = self._get_layer_block_ids(request, layer_name)
            extract_started = time.perf_counter()
            kv_cache = self._extract_kv_from_layer(
                kv_layer,
                layer_block_ids,
                attn_metadata,
            )
            extract_elapsed_s = time.perf_counter() - extract_started
            self._record_timing(
                "save_kv_extract",
                extract_elapsed_s,
                self._tensor_nbytes(kv_cache),
            )
            self._record_request_timing(
                request_id,
                "save_kv_extract",
                extract_elapsed_s,
                self._tensor_nbytes(kv_cache),
            )
            if kv_cache is None:
                raise NotImplementedError(
                    "P2pNcclConnector cannot save unsupported KV layout "
                    f"for layer {layer_name}"
                )
            if self._is_layer_bundling_enabled():
                self._queue_layer_bundle(
                    request_id,
                    layer_name,
                    kv_cache,
                    remote_address,
                )
                continue
            enqueue_started = time.perf_counter()
            sent = self.p2p_nccl_engine.send_tensor(
                self._tensor_id(request_id, layer_name), kv_cache, remote_address
            )
            enqueue_elapsed_s = time.perf_counter() - enqueue_started
            self._record_timing(
                "save_kv_send_tensor",
                enqueue_elapsed_s,
                self._tensor_nbytes(kv_cache),
            )
            self._record_request_timing(
                request_id,
                "save_kv_send_tensor",
                enqueue_elapsed_s,
                self._tensor_nbytes(kv_cache),
            )
            if sent is False:
                logger.error(
                    "P2pNcclConnector failed to enqueue/send KV tensor for "
                    "request_id:%s, layer_name:%s",
                    self._transfer_request_id(request_id),
                    layer_name,
                )

    def wait_for_save(self):
        if self.is_producer:
            assert self.p2p_nccl_engine is not None
            flushed_request_ids: list[str] = []
            if self._is_layer_bundling_enabled():
                flushed_request_ids = self._flush_all_layer_bundles()
            if (
                self._producer_transfer_layer_order_logged is False
                and len(self._producer_transfer_layer_order) > 0
            ):
                self._producer_transfer_layer_order_logged = True
                logger.info(
                    "P2P NCCL transfer layer order, role:%s, rank:%d, layers:%s",
                    "producer",
                    self._rank,
                    self._producer_transfer_layer_order,
                )
            for request_id in flushed_request_ids:
                self._emit_request_milestone(request_id, "producer_wait_for_sent_start")
            wait_started = time.perf_counter()
            self.p2p_nccl_engine.wait_for_sent()
            wait_elapsed_s = time.perf_counter() - wait_started
            for request_id in flushed_request_ids:
                self._emit_request_milestone(
                    request_id,
                    "producer_wait_for_sent_done",
                    extra={"wait_for_save_seconds": wait_elapsed_s},
                )
            self._record_timing(
                "wait_for_save",
                wait_elapsed_s,
            )
            self._log_timing_stats("wait_for_save")
            self._log_request_timing_stats(
                "wait_for_save",
                extra={"wait_for_save_seconds": wait_elapsed_s},
            )
            return

        for request_id in sorted(self._finished_recving_request_ids):
            self._emit_request_milestone(
                request_id,
                "consumer_post_forward_reached",
            )

    def get_finished(
        self, finished_req_ids: set[str], **kwargs: Any
    ) -> tuple[set[str] | None, set[str] | None]:
        """
        Notifies worker-side connector ids of requests that have
        finished generating tokens.

        Returns:
            ids of requests that have finished asynchronous transfer,
            tuple of (sending/saving ids, recving/loading ids).
            The finished saves/sends req ids must belong to a set provided in a
            call to this method (this call or a prior one).
        """

        assert self.p2p_nccl_engine is not None

        no_compile_layers = self._vllm_config.compilation_config.static_forward_context
        finished_transfer_req_ids = {
            self._transfer_request_id(request_id) for request_id in finished_req_ids
        }
        finished_sending, engine_finished_recving = self.p2p_nccl_engine.get_finished(
            finished_transfer_req_ids, no_compile_layers
        )
        finished_recving = set(engine_finished_recving or ())
        if not self.is_producer and self._load_kv_async_enabled:
            finished_recving.update(self._finished_recving_request_ids)
            self._finished_recving_request_ids.clear()
        self._log_timing_stats("get_finished")
        self._log_request_timing_stats("get_finished")
        return finished_sending, finished_recving or None

    def get_block_ids_with_load_errors(self) -> set[int]:
        block_ids = self._block_ids_with_load_errors
        self._block_ids_with_load_errors = set()
        if len(block_ids) > 0:
            logger.error(
                "P2pNcclConnector reporting %d block ids with KV load errors",
                len(block_ids),
            )
        return block_ids

    # ==============================
    # Scheduler-side methods
    # ==============================

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        """
        Get number of new tokens that can be loaded from the
        external KV cache beyond the num_computed_tokens.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            the number of tokens that can be loaded from the
            external KV cache beyond what is already computed.
        """
        if self.is_producer:
            return 0, False

        prompt_token_ids = (
            request.prompt_token_ids if request.prompt_token_ids is not None else []
        )
        num_external_tokens = len(prompt_token_ids) - 1 - num_computed_tokens

        if num_external_tokens < 0:
            num_external_tokens = 0

        return num_external_tokens, (
            self._load_kv_async_enabled and num_external_tokens > 0
        )

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        """
        Update KVConnector state after block allocation.
        """
        if not self.is_producer and num_external_tokens > 0:
            self._requests_need_load[request.request_id] = (
                request,
                blocks.get_block_ids(),
                num_external_tokens,
            )

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        """Build the connector metadata for this step.

        This function should NOT modify any fields in the scheduler_output.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """

        meta = P2pNcclConnectorMetadata()

        for new_req in scheduler_output.scheduled_new_reqs:
            if self.is_producer:
                num_scheduled_tokens = (scheduler_output.num_scheduled_tokens)[
                    new_req.req_id
                ]
                num_tokens = num_scheduled_tokens + new_req.num_computed_tokens
                # the request's prompt is chunked prefill
                prompt_token_count = (
                    len(new_req.prompt_token_ids)
                    if new_req.prompt_token_ids is not None
                    else 0
                )
                if num_tokens < prompt_token_count:
                    # 'CachedRequestData' has no attribute 'prompt_token_ids'
                    self.chunked_prefill[new_req.req_id] = (
                        new_req.block_ids,
                        new_req.prompt_token_ids,
                    )
                    continue
                # the request's prompt is not chunked prefill
                self._add_producer_request_meta(
                    meta=meta,
                    request_id=new_req.req_id,
                    block_ids=new_req.block_ids,
                    prompt_token_ids=new_req.prompt_token_ids,
                )
                continue
            if new_req.req_id in self._requests_need_load:
                request, block_ids, num_external_tokens = self._requests_need_load.pop(
                    new_req.req_id
                )
                self._add_consumer_request_meta(
                    meta=meta,
                    request_id=new_req.req_id,
                    token_ids=request.all_token_ids,
                    block_ids=block_ids,
                    num_external_tokens=num_external_tokens,
                )

        cached_reqs = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached_reqs.req_ids):
            num_computed_tokens = cached_reqs.num_computed_tokens[i]
            new_block_ids = cached_reqs.new_block_ids[i]
            resumed_from_preemption = req_id in cached_reqs.resumed_req_ids

            if self.is_producer:
                num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
                num_tokens = num_scheduled_tokens + num_computed_tokens
                assert req_id in self.chunked_prefill
                existing_block_ids, prompt_token_ids = self.chunked_prefill[req_id]
                if resumed_from_preemption:
                    assert new_block_ids is not None
                    block_ids = new_block_ids
                elif new_block_ids is not None:
                    block_ids = self._concat_block_ids_by_group(
                        existing_block_ids,
                        new_block_ids,
                    )
                else:
                    block_ids = existing_block_ids
                assert prompt_token_ids is not None
                # the request's prompt is chunked prefill again
                if num_tokens < len(prompt_token_ids):
                    self.chunked_prefill[req_id] = (block_ids, prompt_token_ids)
                    continue
                # the request's prompt is all prefilled finally
                self._add_producer_request_meta(
                    meta=meta,
                    request_id=req_id,
                    block_ids=block_ids,
                    prompt_token_ids=prompt_token_ids,
                )
                self.chunked_prefill.pop(req_id, None)
                continue

            # NOTE(rob): here we rely on the resumed requests being
            # the first N requests in the list scheduled_cache_reqs.
            if not resumed_from_preemption:
                break
            if req_id in self._requests_need_load:
                request, block_ids, num_external_tokens = self._requests_need_load.pop(
                    req_id
                )
                self._add_consumer_request_meta(
                    meta=meta,
                    request_id=req_id,
                    token_ids=request.all_token_ids[num_computed_tokens:],
                    block_ids=block_ids,
                    num_external_tokens=num_external_tokens,
                )

        if not self.is_producer and self._load_kv_async_enabled:
            for req_id, (request, block_ids, num_external_tokens) in list(
                self._requests_need_load.items()
            ):
                self._add_consumer_request_meta(
                    meta=meta,
                    request_id=req_id,
                    token_ids=request.all_token_ids,
                    block_ids=block_ids,
                    num_external_tokens=num_external_tokens,
                )

        self._requests_need_load.clear()
        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Called when a request has finished, before its blocks are freed.

        Returns:
            True if the request is being saved/sent asynchronously and blocks
            should not be freed until the request_id is returned from
            get_finished().
            Optional KVTransferParams to be included in the request outputs
            returned by the engine.
        """

        return self.request_finished_all_groups(request, (block_ids,))

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        """Called when a request has finished for all KV cache groups."""

        self.chunked_prefill.pop(request.request_id, None)
        self._drop_pending_async_load_request(request.request_id)
        self._drop_pipelined_load_request(request.request_id)

        return False, None

    # ==============================
    # Static methods
    # ==============================

    @staticmethod
    def parse_request_id(request_id: str, is_prefill=True) -> tuple[str, int]:
        # Regular expression to match the string hostname and integer port
        if is_prefill:
            pattern = r"___decode_addr_(.*):(\d+)"
        else:
            pattern = r"___prefill_addr_(.*):(\d+)___"

        # Use re.search to find the pattern in the request_id
        match = re.search(pattern, request_id)
        if match:
            # Extract the ranks
            ip = match.group(1)
            port = int(match.group(2))

            return ip, port
        raise ValueError(f"Request id {request_id} does not contain hostname and port")

    @staticmethod
    def check_tensors_except_dim(tensor1, tensor2, dim):
        shape1 = tensor1.size()
        shape2 = tensor2.size()

        if len(shape1) != len(shape2) or not all(
            s1 == s2 for i, (s1, s2) in enumerate(zip(shape1, shape2)) if i != dim
        ):
            raise NotImplementedError(
                "Currently, only symmetric TP is supported. Asymmetric TP, PP,"
                "and others will be supported in future PRs."
            )
