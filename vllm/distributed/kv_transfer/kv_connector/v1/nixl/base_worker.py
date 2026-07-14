# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Base worker-side logic for the NIXL connector."""

import json
import logging
import os
import queue
import threading
import time
import traceback
import uuid
from collections import defaultdict, deque
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Never, cast

import msgspec
import numpy as np
import torch
import zmq

from vllm.distributed.kv_transfer.integrity import (
    IntegrityIdentity,
    IntegrityPayloadKind,
    IntegrityStage,
)
from vllm.distributed.kv_transfer.kv_connector.utils import (
    BlockIds,
    EngineId,
    EngineTransferInfo,
    TransferTopology,
    get_current_attn_backends,
    kv_postprocess_blksize_and_layout_on_receive,
    kv_postprocess_blksize_on_receive,
    kv_postprocess_layout_on_receive,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import CopyBlocksOp
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    GET_META_MSG,
    HeartbeatInfo,
    NixlAgentMetadata,
    NixlConnectorMetadata,
    NixlHandshakePayload,
    ReqId,
    ReqMeta,
    TransferHandle,
    compute_nixl_compatibility_hash,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.stats import (
    NixlKVConnectorStats,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import (
    TPMapping,
    _is_attention_spec,
    _is_ssm_spec,
    compute_tp_mapping,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.utils import (
    _NIXL_SUPPORTED_DEVICE,
    get_representative_spec_type,
    zmq_ctx,
)
from vllm.distributed.kv_transfer.kv_connector.v1.ssm_conv_transfer_utils import (
    MambaConvSplitInfo,
    derive_mamba_conv_split,
)
from vllm.distributed.kv_transfer.nixl_fingerprint import NixlDeviceFingerprinter
from vllm.distributed.kv_transfer.nixl_localization import (
    IntegrityLeafKey,
    LocalizationArtifactWriter,
    LocalizationError,
    LocalizationMode,
    NixlCaptureRecord,
    NixlEventRecord,
    NixlIntegrityLeaf,
    NixlLocalizationConfig,
    NixlRegionDescriptor,
    NixlSourceContract,
    NixlSourceManifest,
    NixlSourceManifestRecord,
    NixlSourceRoster,
    build_fingerprint_leaf,
    build_integrity_identity,
    build_integrity_leaf,
    compute_semantic_contract_digest,
    leaf_source_key,
    seal_source_manifest,
    validate_source_manifest_structure,
)
from vllm.distributed.kv_transfer.staging_ownership import (
    CoalescedStagingPlan,
    HandleState,
    StagingRangeAllocator,
    StagingSafetyError,
)
from vllm.distributed.nixl_utils import NixlWrapper, nixl_agent_config
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.network_utils import make_zmq_path
from vllm.v1.attention.backends.utils import get_kv_cache_layout
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.outputs import KVTransferFailure, KVTransferFailureReason
from vllm.v1.worker.block_table import BlockTable
from vllm.v1.worker.utils import select_common_block_size

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


@dataclass(frozen=True, slots=True)
class _LocalizationLeafSpec:
    """Canonical lineage and placement for one pending device fingerprint."""

    identity: IntegrityIdentity
    local_block_id: int | None
    destination_half: int | None
    rank_slot: int | None


class NixlBaseConnectorWorker:
    """Base implementation of Worker side methods shared by pull and push."""

    def _compute_desc_ids(
        self,
        block_ids: BlockIds,
        dst_num_blocks: int,
        block_size_ratio: float | None,
        physical_blocks_per_logical: int,
    ) -> np.ndarray:
        """Compute NIXL descriptor IDs for given block IDs."""
        num_fa_regions = self.num_regions
        num_ssm_regions = 0
        if self._has_mamba:
            assert self._conv_decomp is not None
            # NIXL regions per SSM layer = conv sub-projections + 1 SSM temporal
            # (Mamba2/GDN: 3+1=4; Mamba1: 1+1=2).
            ssm_regions_per_layer = len(self._conv_decomp.local_conv_offsets) + 1
            num_ssm_regions = len(self.block_len_per_layer) * ssm_regions_per_layer

        num_blocks = dst_num_blocks
        if block_size_ratio is not None:
            num_blocks = int(num_blocks * block_size_ratio)
        num_fa_descs = num_fa_regions * num_blocks

        # All-attention fast path: single vectorized broadcast.
        if num_ssm_regions == 0:
            # NOTE (NickLucche) With HMA, every kv group has the same number of layers
            # and layers from different groups share the same kv tensor.
            # eg block_ids=[[1, 2], [3]]->blocks [1, 2] need to be
            # read across all regions, same for [3], but group0-group1 blocks will
            # always differ (different areas). Therefore we can just flatten the
            # block_ids and compute the descs ids for all groups at once.
            block_arr = np.concatenate(block_ids)[None, :]
            region_ids = np.arange(num_fa_regions)[:, None]
            return (region_ids * num_blocks + block_arr).flatten()

        # Compute desc ids per group using the right stride: FA descs have
        # num_blocks entries per region (kernel granularity), SSM descs have
        # logical_blocks entries per region (no kernel splitting).
        logical_blocks = num_blocks // physical_blocks_per_logical
        all_descs: list[np.ndarray] = []
        for i, group in enumerate(block_ids):
            group_arr = np.asarray(group)
            if _is_attention_spec(self._group_spec_types[i]):
                fa_region_ids = np.arange(num_fa_regions)[:, None]
                all_descs.append(
                    (fa_region_ids * num_blocks + group_arr[None, :]).flatten()
                )
            elif _is_ssm_spec(self._group_spec_types[i]):
                # NOTE (NickLucche) SSM and Attention block regions can
                # be exchanged arbitrarily by manager.  Therefore, descs
                # are laid out as:
                #   [descs_fa (all regions) | descs_ssm (all regions)].
                # num_fa_descs offset must be computed per-engine since
                # P and D can have different num_blocks (and thus
                # different FA desc counts).
                ssm_region_ids = np.arange(num_ssm_regions)[:, None]
                all_descs.append(
                    (
                        ssm_region_ids * logical_blocks
                        + group_arr[None, :]
                        + num_fa_descs
                    ).flatten()
                )
            else:
                raise ValueError(
                    f"Unknown spec type {self._group_spec_types[i]} at index {i}"
                )

        return np.concatenate(all_descs)

    def _build_local_splits_from_plan(
        self,
        plan: TPMapping,
        src_blocks_data: list[tuple[int, int, int]],
        num_fa_descs: int,
    ) -> Iterator[list[tuple[int, int, int]]]:
        """Build split handle data for P_TP > D_TP scenario.

        num_fa_descs is the boundary between FA and SSM descriptors.
        Split counts are derived from source_ranks_per_group lengths.
        FA uses rank_to_attention_slot for the slot offset;
        SSM uses the rank's positional index.
        """
        fa_idx = next(
            i for i, t in enumerate(self._group_spec_types) if _is_attention_spec(t)
        )
        fa_num_splits = len(plan.source_ranks_per_group[fa_idx])

        has_ssm_descs = num_fa_descs < len(src_blocks_data)
        ssm_idx = next(
            (i for i, t in enumerate(self._group_spec_types) if _is_ssm_spec(t)),
            None,
        )
        ssm_num_splits = (
            len(plan.source_ranks_per_group[ssm_idx])
            if has_ssm_descs and ssm_idx is not None
            else 0
        )

        # Per-FA-descriptor replicate flag, in _build_fa_local emission order.
        fa_desc_replicated = self._fa_desc_replicated(num_fa_descs)

        for p_idx, p_rank in enumerate(plan.all_source_ranks):
            fa_slot = plan.rank_to_attention_slot.get(p_rank, 0)

            handle: list[tuple[int, int, int]] = []
            for j, (addr, local_len, dev) in enumerate(src_blocks_data):
                if j < num_fa_descs:
                    if fa_desc_replicated[j]:
                        # REPLICATE (MLA): whole block written on every rank.
                        handle.append((addr, local_len, dev))
                    else:
                        # SPLIT (full-attn): this rank's head slice.
                        chunk = local_len // fa_num_splits
                        handle.append((addr + fa_slot * chunk, chunk, dev))
                else:
                    chunk = local_len // ssm_num_splits
                    handle.append((addr + p_idx * chunk, chunk, dev))
            yield handle

    def _fa_desc_replicated(self, num_fa_descs: int) -> list[bool]:
        """Per-FA-descriptor replicate flag, in _build_fa_local emission order
        (region-major; K then optional V per region). Length ``num_fa_descs``.
        """
        assert self.transfer_topo is not None
        n_regions = len(self.block_len_per_layer)
        if n_regions == 0 or self.num_regions == 0:
            return [False] * num_fa_descs
        nblk = num_fa_descs // self.num_regions
        virtually_split = self.transfer_topo.virtually_split_kv_in_blocks
        flags: list[bool] = []
        for i in range(n_regions):
            replicated = self._is_region_replicated(i)
            num_streams = 1 if replicated or not virtually_split else 2
            flags.extend([replicated] * (num_streams * nblk))
        assert len(flags) == num_fa_descs, (
            f"FA desc flags {len(flags)} != num_fa_descs {num_fa_descs}"
        )
        return flags

    def _is_region_replicated(self, region_idx: int) -> bool:
        """Whether region ``region_idx`` is transferred REPLICATE vs SPLIT.

        REPLICATE (MLA): identical on every rank, whole block read from one
        rank at offset 0, key-only. SPLIT (full-attn): head-sharded across TP.
        Defaults to SPLIT when the per-region map is unset (e.g. tests that set
        block_len_per_layer without register_kv_caches).
        """
        return region_idx < len(self._region_is_mla) and self._region_is_mla[region_idx]

    @staticmethod
    def _parse_phase_separation_config(value: object) -> bool:
        """Validate the transfer/decode phase-separation switch.

        :param value: Raw connector-extra configuration value.
        :returns: Validated boolean switch.
        :raises ValueError: If the value is not exactly a boolean.
        """
        if type(value) is not bool:
            raise ValueError("phase_separate_transfer_decode must be a boolean")
        return value

    def __init__(
        self,
        vllm_config: "VllmConfig",
        engine_id: str,
        kv_cache_config: "KVCacheConfig",
    ) -> None:
        nixl_wrapper_cls = NixlWrapper
        if nixl_wrapper_cls is None:
            logger.error("NIXL is not available")
            raise RuntimeError("NIXL is not available")
        logger.info("Initializing NIXL wrapper")
        logger.info("Initializing NIXL worker %s", engine_id)

        # Config.
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        # mypy will complain on re-assignment otherwise.
        self.block_size: int = cast(int, vllm_config.cache_config.block_size)

        if vllm_config.kv_transfer_config is None:
            raise ValueError("kv_transfer_config must be set for NixlConnector")
        self.kv_transfer_config = vllm_config.kv_transfer_config
        phase_separation_key = "phase_separate_transfer_decode"
        self._phase_separation_instrumented = (
            phase_separation_key in self.kv_transfer_config.kv_connector_extra_config
        )
        self._phase_separate_transfer_decode = self._parse_phase_separation_config(
            self.kv_transfer_config.get_from_extra_config(
                phase_separation_key,
                False,
            )
        )
        if (
            self._phase_separate_transfer_decode
            and self.kv_transfer_config.kv_role != "kv_consumer"
        ):
            raise ValueError(
                "phase_separate_transfer_decode requires kv_role='kv_consumer'"
            )
        self._transfer_phase_active = False
        self._transfer_phase_epoch = 0
        self._deferred_phase_sending: set[ReqId] = set()
        self._deferred_phase_recving: set[ReqId] = set()
        self._transfer_phase_plan_count = 0
        self._transfer_phase_handle_count = 0
        self._transfer_phase_byte_count = 0
        self._transfer_phase_violation_count = 0
        self._transfer_phase_started_ns = 0
        self._transfer_phase_entry_sync_ns = 0
        self._transfer_phase_records: list[tuple[str, dict[str, object]]] = []
        if self._phase_separate_transfer_decode:
            logger.info(
                "[transfer-decode-phase] %s",
                json.dumps(
                    {
                        "enabled": True,
                        "engine_id": engine_id,
                        "event": "configured",
                    },
                    sort_keys=True,
                ),
            )

        self.nixl_backends = vllm_config.kv_transfer_config.get_from_extra_config(
            "backends", ["UCX"]
        )
        kv_lease_duration: int = vllm_config.kv_transfer_config.get_from_extra_config(
            "kv_lease_duration", 30
        )
        # NOTE (NickLucche): For now we use a hardcoded value for a simpler interface.
        self._lease_extension = kv_lease_duration * 2 // 3
        self._heartbeat_interval = max(kv_lease_duration // 6, 1)
        self._heartbeat_targets: dict[EngineId, HeartbeatInfo] = {}
        self._last_heartbeat_time = 0.0

        self._is_hma_required = (
            not vllm_config.scheduler_config.disable_hybrid_kv_cache_manager
            and any(
                not isinstance(g.kv_cache_spec, FullAttentionSpec)
                for g in kv_cache_config.kv_cache_groups
            )
        )
        self.kv_cache_config = kv_cache_config
        self._layer_to_group_index = {
            layer_name: group_index
            for group_index, group in enumerate(kv_cache_config.kv_cache_groups)
            for layer_name in group.layer_names
        }
        layer_count = sum(
            len(group.layer_names) for group in kv_cache_config.kv_cache_groups
        )
        if len(self._layer_to_group_index) != layer_count:
            raise LocalizationError("KV cache layer belongs to multiple groups")
        self._layer_specs = {
            layer: group.kv_cache_spec
            for group in kv_cache_config.kv_cache_groups
            for layer in group.layer_names
        }
        self.hma_group_size = len(kv_cache_config.kv_cache_tensors)

        # ---- Model state (derived from model config) ----
        mamba_ssm_size = (0, 0)
        # Conv state sub-projection decomposition (None when no Mamba).
        # The transfer requires DS (dim, state_len) conv layout so that
        # conv sub-projections are contiguous in memory.
        self._conv_decomp: MambaConvSplitInfo | None = None
        self._has_mamba = any(
            isinstance(g.kv_cache_spec, MambaSpec)
            for g in kv_cache_config.kv_cache_groups
        )
        if self._has_mamba:
            assert self._is_hma_required
            from vllm.model_executor.layers.mamba.mamba_utils import (
                is_conv_state_dim_first,
            )

            assert is_conv_state_dim_first(), (
                "3-read Mamba conv transfer requires DS conv state layout. "
                "Set VLLM_SSM_CONV_STATE_LAYOUT=DS"
            )
            mamba_spec = next(
                spec
                for spec in self._layer_specs.values()
                if isinstance(spec, MambaSpec)
            )
            self._conv_decomp = derive_mamba_conv_split(
                mamba_spec,
                vllm_config.parallel_config.tensor_parallel_size,
            )
            mamba_ssm_size = self._conv_decomp.ssm_sizes
        self._mamba_ssm_size = mamba_ssm_size

        # Agent.
        non_ucx_backends = [b for b in self.nixl_backends if b != "UCX"]
        # Configure NIXL num_threads to avoid UAR exhaustion on Mellanox NICs.
        # Each UCX thread allocates UARs (doorbell pages) via DevX, and
        # excessive NIXL UAR usage can exhaust NIC UAR space. This can cause
        # components like NVSHMEM (used by DeepEP kernels) to fail during RDMA
        # initialization with "mlx5dv_devx_alloc_uar" errors.
        # Ref: https://network.nvidia.com/files/doc-2020/ethernet-adapters-programming-manual.pdf#page=63
        num_threads = vllm_config.kv_transfer_config.get_from_extra_config(
            "num_threads", 4
        )
        if nixl_agent_config is None:
            config = None
        else:
            # Enable telemetry by default for NIXL 0.7.1 and above.
            config = (
                nixl_agent_config(backends=self.nixl_backends, capture_telemetry=True)
                if len(non_ucx_backends) > 0
                else nixl_agent_config(num_threads=num_threads, capture_telemetry=True)
            )

        self.nixl_wrapper = nixl_wrapper_cls(str(uuid.uuid4()), config)
        # Map of engine_id -> {rank0: agent_name0, rank1: agent_name1..}.
        self._remote_agents: dict[EngineId, dict[int, str]] = defaultdict(dict)

        # Metadata.
        self.engine_id: EngineId = engine_id
        self.tp_rank = get_tensor_model_parallel_rank()
        self.world_size = get_tensor_model_parallel_world_size()
        self._registration_generation = uuid.uuid4().hex

        self.num_blocks = kv_cache_config.num_blocks
        self.enable_permute_local_kv = False
        self.enable_heterogeneous_attn_post_process = False

        # KV Caches and nixl tracking data.
        self.device_type = current_platform.device_type
        self.kv_buffer_device: str = vllm_config.kv_transfer_config.kv_buffer_device
        if self.device_type not in _NIXL_SUPPORTED_DEVICE:
            raise RuntimeError(f"{self.device_type} is not supported.")
        elif self.kv_buffer_device not in _NIXL_SUPPORTED_DEVICE[self.device_type]:
            raise RuntimeError(
                f"{self.device_type} with {self.kv_buffer_device} kv_buffer "
                "is not supported."
            )
        self.device_kv_caches: dict[str, torch.Tensor] = {}

        # Coalesced pull path (VLLM_NIXL_COALESCED_PULL=1): with remote
        # TP > local TP, the stock pull scatters one descriptor per
        # (block, region, K/V, remote shard) -- the local head-slice
        # destinations are strided, so descriptors bottom out at
        # local_block_len/|tp_ratio| bytes (16KB here) and transfers run
        # latency-bound at ~0.25% of fabric bandwidth, with descriptor
        # posting/reaping stalling the engine thread. This path instead
        # reads whole remote blocks in contiguous-run descriptors into a
        # registered staging buffer (O(regions x runs) descriptors) and
        # re-scatters into the real cache with one strided GPU copy per
        # (region, shard) at completion, before the request is released
        # to the scheduler. Byte placement is identical to stock.
        self.coalesce_pull = os.environ.get("VLLM_NIXL_COALESCED_PULL", "0") == "1"
        self.coalesce_staging_mb = int(
            os.environ.get("VLLM_NIXL_COALESCED_STAGING_MB", "12288")
        )
        self._coalesce_warn_after_s = float(
            self.kv_transfer_config.get_from_extra_config(
                "coalesce_staging_warn_after_s", 30.0
            )
        )
        self._coalesce_fail_after_s = float(
            self.kv_transfer_config.get_from_extra_config(
                "coalesce_staging_fail_after_s", 300.0
            )
        )
        if (
            self._coalesce_warn_after_s <= 0
            or self._coalesce_fail_after_s <= self._coalesce_warn_after_s
        ):
            raise ValueError(
                "coalesce_staging_fail_after_s must exceed the positive "
                "coalesce_staging_warn_after_s"
            )
        self._staging_buf: torch.Tensor | None = None
        self._staging_allocator: StagingRangeAllocator | None = None
        # Requests waiting for a staging range are serviced FIFO as completed
        # plans free ranges. Parking is safe:
        # to the rest of the engine a parked request is
        # indistinguishable from an in-flight transfer (blocks stay
        # held until we report done_recving), and it beats the
        # alternative -- falling back to the stock path costs ~20x in
        # transfer time, while a range frees in ~100-300ms.
        self._coalesce_pending: deque = deque()
        # Coalesced handles, scatter geometry, and staging lifetime have one
        # generation-scoped owner. Stock transfers remain in
        # _recving_transfers because they never write the staging registration.
        self._coalesce_plans: dict[ReqId, CoalescedStagingPlan] = {}
        self._coalesce_owner_sequence = 0
        self._sp_flags_cache: list[bool] | None = None
        # canonicalized (nb, row_bytes) uint8 views of each region's
        # physical storage (registered tensors are permuted VIEWS of the
        # HND-contiguous base; stride-sorting recovers it). None until
        # built; False-y build failure disables coalescing.
        self._region_rows: list[torch.Tensor] | None = None
        # engine_id -> rank -> (block_lens, num_blocks, device_id) from
        # the handshake metadata (needed to build raw range descriptors)
        self._remote_layout: dict[EngineId, dict[int, tuple]] = defaultdict(dict)
        self._remote_regions: dict[
            EngineId, dict[int, tuple[NixlRegionDescriptor, ...]]
        ] = defaultdict(dict)
        self._remote_registration_generations: dict[EngineId, dict[int, str]] = (
            defaultdict(dict)
        )
        self._remote_source_semantics: dict[
            EngineId, dict[int, tuple[tuple[int, ...], tuple[int, ...]]]
        ] = defaultdict(dict)
        # region index -> registered cache tensor (scatter destinations)
        self._region_tensors: list[torch.Tensor] = []
        self._region_descriptors: tuple[NixlRegionDescriptor, ...] = ()
        self._localization_config = NixlLocalizationConfig.from_environment()
        self._localization_fingerprinter = (
            NixlDeviceFingerprinter()
            if self._localization_config.mode is LocalizationMode.FINGERPRINT
            else None
        )
        self._localization_writer = (
            LocalizationArtifactWriter(
                self._localization_config,
                self.engine_id,
                self.tp_rank,
                self.world_size,
                self.model_config.get_total_num_kv_heads(),
            )
            if self._localization_config.enabled
            else None
        )
        self._localization_source_rosters: dict[ReqId, NixlSourceRoster] = {}
        self._localization_pre_read_plans: dict[ReqId, dict[str, Any]] = {}
        self._localization_zero_recorded: set[ReqId] = set()
        self._localization_terminal_recorded: set[ReqId] = set()

        # cpu kv buffer for xfer
        # used when device memory can not be registered under nixl
        self.host_xfer_buffers: dict[str, torch.Tensor] = {}
        if self.device_type == "cpu":
            self.use_host_buffer = False
        else:
            self.use_host_buffer = self.kv_buffer_device == "cpu"

        # reserve different cores for start_load_kv() from model_forward()
        if self.device_type == "cpu":
            numa_core_list = current_platform.discover_numa_topology()
            # setup one last core in each numa for kv transfer.
            rsv_cores_for_kv = [
                max(each_numa_core_list) for each_numa_core_list in numa_core_list
            ]

            if rsv_cores_for_kv:
                if not hasattr(os, "sched_setaffinity"):
                    raise NotImplementedError(
                        "os.sched_setaffinity is not available on this platform"
                    )
                os.sched_setaffinity(0, rsv_cores_for_kv)

        # support for oot platform which can't register nixl memory
        # type based on kv_buffer_device
        nixl_memory_type = current_platform.get_nixl_memory_type()
        if nixl_memory_type is None:
            if self.kv_buffer_device in ["cuda", "xpu"]:
                nixl_memory_type = "VRAM"
            elif self.kv_buffer_device == "cpu":
                nixl_memory_type = "DRAM"
        if nixl_memory_type is None:
            raise RuntimeError(
                f"{self.device_type} with {self.kv_buffer_device} kv_buffer "
                "is not supported."
            )
        self.nixl_memory_type = nixl_memory_type

        # Note: host xfer buffer ops when use_host_buffer is True
        self.copy_blocks: CopyBlocksOp | None = None

        # Map of engine_id -> kv_caches_base_addr. For TP case, each local
        self.device_id: int = 0
        # Current rank may pull from multiple remote TP workers.
        # EngineId, dict[int, list[int]] -> engine_id, tp_rank, base_addr_for_layer
        self.kv_caches_base_addr = defaultdict[EngineId, dict[int, list[int]]](dict)

        # Number of NIXL regions. Currently one region per cache
        # (so 1 per layer for MLA, otherwise 2 per layer)
        self.num_regions = 0

        # nixl_prepped_dlist_handle.
        self.src_xfer_handles_by_block_size: dict[int, int] = {}
        # Populated dynamically during handshake based on remote configuration.
        # Keep track of regions at different tp_ratio values. tp_ratio->handles
        self.src_xfer_handles_by_tp_ratio: dict[int, list[int]] = {}
        # Map of engine_id -> {tp_rank: nixl_prepped_dlist_handle (int)}.
        self.dst_xfer_side_handles = defaultdict[EngineId, dict[int, int]](dict)

        # Map of engine_id -> num_blocks. All ranks in the same deployment will
        # have the same number of blocks.
        self.dst_num_blocks: dict[EngineId, int] = {}
        self._registered_descs: list[Any] = []

        # In progress transfers.
        # [req_id -> list[handle]]
        self._recving_metadata: dict[ReqId, ReqMeta] = {}
        self._recving_transfers = defaultdict[ReqId, list[TransferHandle]](list)
        # Track the expiration time of requests that are waiting to be sent.
        self._reqs_to_send: dict[ReqId, float] = {}
        # Release fence: remote rids whose producer blocks are known to be
        # released after all expected consumers completed. Bounded FIFO. A pull
        # must never be issued for, nor committed against, a released rid.
        self._released_rids: dict[str, float] = {}
        # Consumer side: completed pulls per rid; the rid is released once
        # this reaches the request's expected_consumers.
        self._rid_completion_counts: dict[str, int] = {}
        # ---- resident-KV checksum auditor (VLLM_GEMMA4_KV_AUDIT) ----
        # Snapshot content sums of immutable prompt rows at pull commit;
        # re-verify periodically. Debug instrument, off by default.
        self._audit_enabled = os.environ.get("VLLM_GEMMA4_KV_AUDIT", "0") == "1"
        self._audit_interval = int(
            os.environ.get("VLLM_GEMMA4_KV_AUDIT_INTERVAL", "16")
        )
        self._audit_groups = {
            int(x)
            for x in os.environ.get("VLLM_GEMMA4_KV_AUDIT_GROUPS", "10,11").split(",")
            if x.strip()
        }
        self._audit_tail_exclude = int(os.environ.get("VLLM_GEMMA4_KV_AUDIT_TAIL", "2"))
        self._audit_selftest = int(os.environ.get("VLLM_GEMMA4_KV_AUDIT_SELFTEST", "0"))
        # rid -> dict(rows=LongTensor, sums=[n_regions, K] int64, t=float)
        self._audit_state: dict[str, dict] = {}
        # (rid, audit_rows, all_rows) committed this step, snapshot at
        # the end of get_finished (after any post-receive processing).
        self._audit_pending: list[tuple[str, list[int], list[int]]] = []
        # physical row -> (owner rid, since, alive)
        self._audit_row_owner: dict[int, tuple[str, float, bool]] = {}
        # audited physical row -> rid currently auditing it
        self._audit_row_to_rid: dict[int, str] = {}
        self._audit_step = 0
        self._audit_regions: list[int] | None = None
        self._audit_snap_count = 0
        if self._audit_enabled:
            logger.info(
                "[kv-audit] ENABLED interval=%d groups=%s tail=%d selftest=%d",
                self._audit_interval,
                sorted(self._audit_groups),
                self._audit_tail_exclude,
                self._audit_selftest,
            )
        # Set of requests that have been part of a batch, regardless of status.
        self._reqs_to_process: set[ReqId] = set()

        # Invalid blocks from failed NIXL operations (thread-safe queue of block ids)
        self._invalid_block_ids: queue.Queue[set[int]] = queue.Queue()
        # Receive failures may originate in the background handshake thread.
        # Only the main thread mutates transfer and request ownership state.
        self._failed_recv_outcomes: queue.Queue[
            tuple[ReqId, KVTransferFailureReason, TransferHandle | None]
        ] = queue.Queue()
        self._failed_recv_pending: dict[ReqId, KVTransferFailure] = {}
        self._completed_failed_recv_outcomes: queue.Queue[
            tuple[ReqId, KVTransferFailure]
        ] = queue.Queue()

        # Handshake metadata of this worker for NIXL transfers.
        self.xfer_handshake_metadata: NixlHandshakePayload | None = None
        # Background thread for initializing new NIXL handshakes.
        self._handshake_initiation_executor = ThreadPoolExecutor(
            # NIXL is not guaranteed to be thread-safe, limit 1 worker.
            max_workers=1,
            thread_name_prefix="vllm-nixl-handshake-initiator",
        )
        self._ready_requests = queue.Queue[tuple[ReqId, ReqMeta]]()
        self._handshake_futures: dict[EngineId, Future[dict[int, str]]] = {}
        # Protects _handshake_futures and _remote_agents.
        self._handshake_lock = threading.RLock()

        # TTL-based eviction of stale remote engine state.
        self._engine_last_active: dict[EngineId, float] = {}
        self._engine_ttl: float = vllm_config.kv_transfer_config.get_from_extra_config(
            "engine_ttl", 3600.0
        )

        self.use_mla = self.model_config.use_mla

        # Get the attention backend from the first layer
        # NOTE (NickLucche) models with multiple backends are not supported yet
        self.attn_backends = get_current_attn_backends(vllm_config)
        self.backend_name = self.attn_backends[0].get_name()

        self.kv_cache_layout = get_kv_cache_layout()
        self.host_buffer_kv_cache_layout = self.kv_cache_layout
        logger.info(
            "Detected attention backend(s) %s",
            [backend.get_name() for backend in self.attn_backends],
        )
        logger.info("Detected kv cache layout %s", self.kv_cache_layout)

        # lazy initialized in register_kv_caches
        self.compat_hash: str | None = None
        self.transfer_topo: TransferTopology | None = None

        # With heterogeneous TP, P must wait for all assigned D TP workers to
        # finish reading before safely freeing the blocks.
        self.consumer_notification_counts_by_req = defaultdict[ReqId, int](int)
        self.xfer_stats = NixlKVConnectorStats()

        self._physical_blocks_per_logical_kv_block = 1
        self._sync_block_size_with_kernel()

        # Unwrap UniformTypeKVCacheSpecs to get the representative spec type
        self._group_spec_types = tuple(
            get_representative_spec_type(g.kv_cache_spec)
            for g in self.kv_cache_config.kv_cache_groups
        )
        _skip = os.environ.get("VLLM_GEMMA4_SKIP_PULL_GROUPS", "")
        self._skip_pull_groups: set[int] = {
            int(x) for x in _skip.split(",") if x.strip()
        }
        for _gi, _g in enumerate(self.kv_cache_config.kv_cache_groups):
            logger.info(
                "[kv-group] i=%d block_size=%s n_layers=%d first=%s skip_pull=%s",
                _gi,
                getattr(_g.kv_cache_spec, "block_size", "?"),
                len(_g.layer_names),
                _g.layer_names[0] if _g.layer_names else "-",
                _gi in self._skip_pull_groups,
            )

        # Per-region MLA flag, 1:1 with block_len_per_layer. True -> REPLICATE
        # (MLA), False -> SPLIT (head-sharded full-attn). Mixed only for models
        # combining both (e.g. GQA main + MLA Eagle-3 draft).
        self._region_is_mla = list[bool]()

        # Enable different block lengths for different layers *only* when MLA is used.
        # This is not used for SSM layers, which use the counterpart `mamba_ssm_size`.
        self.block_len_per_layer = list[int]()

        # Per-engine TP mappings. Generated during handshake.
        self.tp_mappings: dict[EngineId, TPMapping] = {}

        self.enforce_compat_hash = self.kv_transfer_config.get_from_extra_config(
            "enforce_handshake_compat", True
        )

    def _sync_block_size_with_kernel(self) -> None:
        backends = get_current_attn_backends(self.vllm_config)
        kernel_block_size = select_common_block_size(self.block_size, backends)
        # Number of blocks not accounting for kernel block mismatches
        self._logical_num_blocks = self.num_blocks
        if self.block_size != kernel_block_size:
            logger.info_once(
                "User-specified logical block size (%s) does not match"
                " physical kernel block size (%s). Using the latter.",
                self.block_size,
                kernel_block_size,
            )
            assert self.block_size > kernel_block_size
            self._physical_blocks_per_logical_kv_block = (
                self.block_size // kernel_block_size
            )
            self.block_size = kernel_block_size
            self.num_blocks *= self._physical_blocks_per_logical_kv_block

    def _nixl_handshake(
        self,
        host: str,
        port: int,
        remote_tp_size: int,
        expected_engine_id: str,
    ) -> dict[int, str]:
        """Do a NIXL handshake with a remote instance."""

        # the first time we connect to a remote agent.
        # be careful, the handshake happens in a background thread.
        # it does not have an active cuda context until any cuda runtime
        # call is made. when UCX fails to find a valid cuda context, it will
        # disable any cuda ipc communication, essentially disabling any NVLink
        # communication.
        # when we are using device buffers, we need to set the device
        # explicitly to make sure the handshake background thread has a valid
        # cuda context.
        if not self.use_host_buffer:
            current_platform.set_device(self.device_id)

        # When target instance TP > local TP, we need to perform multiple
        # handshakes. Do it in a single background job for simplicity.
        # Regardless, only handshake with the remote TP rank(s) that current
        # local rank will read from. Note that With homogeneous TP,
        # this happens to be the same single rank_i.
        assert self.transfer_topo is not None
        p_remote_ranks = self.transfer_topo.handshake_target_ranks(remote_tp_size)
        remote_rank_to_agent_name = {}
        path = make_zmq_path("tcp", host, port)

        with zmq_ctx(zmq.REQ, path) as sock:
            for remote_rank in p_remote_ranks:
                logger.debug(
                    "Querying metadata on path: %s at remote tp rank %s",
                    path,
                    remote_rank,
                )

                start_time = time.perf_counter()
                # Send query for the request.
                msg = msgspec.msgpack.encode((GET_META_MSG, remote_rank))
                # Set receive timeout to 5 seconds to avoid hanging on dead server
                sock.setsockopt(zmq.RCVTIMEO, 5000)  # milliseconds
                sock.send(msg)
                handshake_bytes = sock.recv()

                # Decode handshake payload to get compatibility hash
                handshake_decoder = msgspec.msgpack.Decoder(NixlHandshakePayload)
                try:
                    handshake_payload = handshake_decoder.decode(handshake_bytes)
                except (msgspec.DecodeError, msgspec.ValidationError) as e:
                    raise RuntimeError(
                        f"Failed to decode NixlHandshakePayload. This likely indicates "
                        f"an incompatibility between connector version. Error: {e}"
                    ) from e

                got_metadata_time = time.perf_counter()
                logger.debug(
                    "NIXL handshake: get metadata took: %s",
                    got_metadata_time - start_time,
                )

                # Check compatibility hash BEFORE decoding agent metadata
                assert self.compat_hash is not None
                if (
                    self.enforce_compat_hash
                    and handshake_payload.compatibility_hash != self.compat_hash
                ):
                    raise RuntimeError(
                        f"NIXL compatibility hash mismatch. "
                        f"Local: {self.compat_hash}, "
                        f"Remote: {handshake_payload.compatibility_hash}. "
                        f"Prefill and decode instances have incompatible "
                        f"configurations. This may be due to: different vLLM versions,"
                        f" models, dtypes, KV cache layouts, attention backends, etc. "
                        f"Both instances must use identical configurations."
                        f"Disable this check using "
                        f'--kv-transfer-config \'{{"kv_connector_extra_config": '
                        f'{{"enforce_handshake_compat": false}}}}\''
                    )

                logger.info(
                    "NIXL compatibility check passed (hash: %s)",
                    handshake_payload.compatibility_hash,
                )

                # Decode agent metadata
                metadata_decoder = msgspec.msgpack.Decoder(NixlAgentMetadata)
                try:
                    metadata = metadata_decoder.decode(
                        handshake_payload.agent_metadata_bytes
                    )
                except (msgspec.DecodeError, msgspec.ValidationError) as e:
                    # This should not happen if hash matched
                    raise RuntimeError(
                        f"Failed to decode NixlAgentMetadata. Error: {e}"
                    ) from e

                # Ensure engine id matches.
                if metadata.engine_id != expected_engine_id:
                    raise RuntimeError(
                        f"Remote NIXL agent engine ID mismatch. "
                        f"Expected {expected_engine_id},"
                        f"received {metadata.engine_id}."
                    )

                # Register Remote agent.
                remote_agent_name = self.add_remote_agent(
                    metadata, remote_rank, remote_tp_size
                )
                setup_agent_time = time.perf_counter()
                logger.debug(
                    "NIXL handshake: add agent took: %s",
                    setup_agent_time - got_metadata_time,
                )
                remote_rank_to_agent_name[remote_rank] = remote_agent_name
        return remote_rank_to_agent_name

    def initialize_host_xfer_buffer(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """
        Initialize transfer buffer in CPU mem for accelerators
        NOT directly supported by NIXL (e.g., tpu)
        """
        xfer_buffers: dict[str, torch.Tensor] = {}
        inv_order = [0, 1, 3, 2, 4]
        try:
            for layer_name, kv_cache in kv_caches.items():
                kv_shape = kv_cache.shape
                kv_dtype = kv_cache.dtype
                permute_shape = False
                if (
                    self.kv_cache_layout == "NHD"
                    and self.vllm_config.kv_transfer_config is not None
                    and self.vllm_config.kv_transfer_config.enable_permute_local_kv
                ):
                    logger.info_once(
                        "'enable_permute_local_kv' flag is enabled while "
                        "device KV Layout is NHD. Init host buffer with"
                        " HND to better support Decode/Prefill TP_ratio > 1."
                    )
                    # Since NHD will not support Decode/Prefill TP_ratio > 1,
                    # we can leverage host_buffer for permute
                    self.host_buffer_kv_cache_layout = "HND"
                    kv_shape = (
                        tuple(kv_shape[i] for i in inv_order)
                        if not self.use_mla
                        else kv_shape
                    )
                    permute_shape = not self.use_mla

                xfer_buffers[layer_name] = torch.empty(
                    kv_shape, dtype=kv_dtype, device="cpu"
                )
                if permute_shape:
                    xfer_buffers[layer_name] = xfer_buffers[layer_name].permute(
                        inv_order
                    )
        except MemoryError as e:
            logger.error("NIXLConnectorWorker gets %s.", e)
            raise

        self.host_xfer_buffers = xfer_buffers

    def set_host_xfer_buffer_ops(self, copy_operation: CopyBlocksOp):
        """Assign copy (d2h, h2d) operations when host buffer is used."""
        # Set a no-op if the host buffer is not cpu.
        if self.kv_buffer_device != "cpu":
            return
        # Set a no-op if self.device_type is 'cpu'.
        if self.device_type == "cpu":
            return
        assert self.use_host_buffer
        self.copy_blocks = copy_operation

    def _log_failure(
        self,
        failure_type: str,
        req_id: str | None,
        msg: str = "",
        error: Exception | None = None,
        meta: ReqMeta | None = None,
        **extra_context,
    ):
        """Log transfer failure with structured context for easier debugging."""
        context: dict[str, Any] = {
            "failure_type": failure_type,
            "request_id": req_id,
            "engine_id": self.engine_id,
        }
        if meta is None and req_id is not None:
            # Try to get metadata from in progress transfers when not provided
            meta = self._recving_metadata.get(req_id)

        if meta and meta.remote:
            context.update(
                {
                    "remote_engine_id": meta.remote.engine_id,
                    "remote_request_id": meta.remote.request_id,
                    "remote_host": meta.remote.host,
                    "remote_port": meta.remote.port,
                    "num_local_blocks": sum(
                        len(group) for group in meta.local_block_ids
                    ),
                    "num_remote_blocks": sum(
                        len(group) for group in meta.remote.block_ids
                    ),
                    "local_block_ids_sample": meta.local_block_ids[0][:10]
                    if meta.local_block_ids
                    else [],
                }
            )

        context.update(extra_context)
        if msg:
            failure_type = f"{failure_type}. {msg}"

        logger.error(
            "NIXL transfer failure: %s | Context: %s",
            failure_type,
            context,
            exc_info=error is not None,
            stacklevel=2,
        )

    def _ensure_handshake(
        self,
        engine_id: EngineId,
        host: str,
        port: int,
        tp_size: int,
    ) -> Future[dict[int, str]] | None:
        """
        Ensure a handshake is in-flight (or already done) for *engine_id*.

        Returns the ``Future`` if a handshake is pending (or was just
        started), or ``None`` if the handshake already completed
        successfully.  Callers can attach per-request callbacks to the
        returned future.
        Failures to handshake are logged and the request is marked as failed.
        """
        self._evict_stale_engines()
        with self._handshake_lock:
            if engine_id in self._remote_agents:
                return None
            fut = self._handshake_futures.get(engine_id)
            if fut is not None:
                return fut
            fut = self._handshake_initiation_executor.submit(
                self._nixl_handshake,
                host,
                port,
                tp_size,
                engine_id,
            )
            self._handshake_futures[engine_id] = fut

            def done_callback(f: Future[dict[int, str]], eid=engine_id):
                with self._handshake_lock:
                    del self._handshake_futures[eid]
                    try:
                        self._remote_agents[eid] = f.result()
                        self._engine_last_active[eid] = time.perf_counter()
                    except Exception as e:
                        self._log_failure(
                            failure_type="handshake_setup_failed",
                            req_id=None,
                            error=e,
                            remote_engine_id=eid,
                        )

            fut.add_done_callback(done_callback)
            return fut

    def _background_nixl_handshake(
        self, req_id: str, remote_engine_id: EngineId, meta: ReqMeta
    ):
        # Do NIXL handshake in background and add to _ready_requests when done.
        assert meta.remote is not None
        fut = self._ensure_handshake(
            remote_engine_id,
            meta.remote.host,
            meta.remote.port,
            meta.tp_size,
        )
        if fut is None:
            # Already handshaked — only happens if caller does not pre-check.
            self._ready_requests.put((req_id, meta))
            return

        # Check handshake success before proceeding with request.
        def request_ready(f: Future[Any], entry=(req_id, meta)):
            try:
                f.result()
                self._ready_requests.put(entry)
            except Exception as e:
                self._log_failure(
                    failure_type="handshake_failed",
                    req_id=req_id,
                    error=e,
                    meta=meta,
                )
                self._handle_failed_transfer(req_id, None)

        fut.add_done_callback(request_ready)

    def register_cross_layers_kv_caches(self, kv_cache: torch.Tensor) -> None:
        """Register a cross-layers KV cache tensor with NIXL.

        `use_uniform_kv_cache()` guarantees a single KV cache group whose
        layers all share the same `AttentionSpec`, so any layer name from
        `_layer_specs` yields the correct per-layer spec for `page_size_bytes`.
        """
        first_layer = next(iter(self._layer_specs))
        # Forwarding a real layer name rather than a synthetic key
        self.register_kv_caches({first_layer: kv_cache})

    def _register_packed_kv_cache(
        self,
        storage: torch.UntypedStorage,
    ) -> None:
        """Register a packed KV cache as a single NIXL region.

        The packed allocation interleaves all layers per block, so each
        block_stride-byte chunk is one logical block.  We register 1
        NIXL region and create 1 descriptor per block.
        """
        self.transfer_topo = TransferTopology(
            tp_rank=self.tp_rank,
            tp_size=self.world_size,
            block_size=self.block_size,
            engine_id=self.engine_id,
            is_mla=self.use_mla,
            total_num_kv_heads=self.model_config.get_total_num_kv_heads(),
            attn_backends=self.attn_backends,
            tensor_shape=None,
            is_mamba=self._has_mamba,
        )
        self.compat_hash = compute_nixl_compatibility_hash(
            self.vllm_config,
            self.backend_name,
            self.transfer_topo.cross_layers_blocks,
        )

        total_size = storage.nbytes()
        block_stride = total_size // self.num_blocks
        base_addr = storage.data_ptr()
        device_id = storage.device.index
        assert device_id is not None

        logger.info(
            "Registering packed KV cache: total_size=%s, block_stride=%s, "
            "num_blocks=%s, num_regions=1",
            total_size,
            block_stride,
            self.num_blocks,
        )

        self.device_id = device_id
        caches_data = [(base_addr, total_size, self.device_id, "")]

        self.block_len_per_layer = [block_stride]
        self._region_descriptors = (
            NixlRegionDescriptor(
                semantic_name="packed_cross_layer_storage",
                group_indices=tuple(range(len(self.kv_cache_config.kv_cache_groups))),
                group_semantic_names=tuple(
                    (
                        group_index,
                        f"packed_cross_layer_storage:group_{group_index}",
                    )
                    for group_index in range(len(self.kv_cache_config.kv_cache_groups))
                ),
                base_address=base_addr,
                registered_bytes=total_size,
                row_bytes=block_stride,
                shape=(self.num_blocks, block_stride),
                strides=(block_stride, 1),
                dtype="uint8",
                element_size_bytes=1,
                layout="packed",
            ),
        )
        self.num_regions = 1
        self.num_descs = self.num_blocks
        self.kv_caches_base_addr[self.engine_id][self.tp_rank] = [base_addr]

        descs = self.nixl_wrapper.get_reg_descs(caches_data, self.nixl_memory_type)
        self.nixl_wrapper.register_memory(descs, backends=self.nixl_backends)
        self._registered_descs.append(descs)

        self.dst_num_blocks[self.engine_id] = self.num_blocks

        self.src_xfer_handles_by_block_size[self.block_size], (self.src_blocks_data) = (
            self.register_local_xfer_handler(self.block_size)
        )

        agent_metadata = NixlAgentMetadata(
            engine_id=self.engine_id,
            agent_metadata=self.nixl_wrapper.get_agent_metadata(),
            device_id=self.device_id,
            kv_caches_base_addr=(
                self.kv_caches_base_addr[self.engine_id][self.tp_rank]
            ),
            num_blocks=self.num_blocks,
            block_lens=self.block_len_per_layer,
            kv_cache_layout=self.kv_cache_layout,
            block_size=self.block_size,
            ssm_sizes=self._mamba_ssm_size,
            attn_backend_name=self.backend_name,
            physical_blocks_per_logical_kv_block=(
                self._physical_blocks_per_logical_kv_block
            ),
            registration_generation=self._registration_generation,
            regions=self._region_descriptors,
            source_group_planes=tuple(
                1 if flag else 2 for flag in self._sp_group_flags()
            ),
            physical_group_token_capacities=(self._physical_group_token_capacities()),
        )
        assert self.compat_hash is not None
        encoder = msgspec.msgpack.Encoder()
        self.xfer_handshake_metadata = NixlHandshakePayload(
            compatibility_hash=self.compat_hash,
            agent_metadata_bytes=encoder.encode(agent_metadata),
        )

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register the KV Cache data in nixl."""

        # Detect packed allocation: all tensors are strided views into the
        # same backing storage (different data_ptr but same storage).
        # This happens with DSv4-style contiguous per-block packing.
        if len(kv_caches) > 1 and not self._has_mamba:
            storage = next(iter(kv_caches.values())).untyped_storage()
            storage_ptrs = {
                cache.untyped_storage().data_ptr() for cache in kv_caches.values()
            }
            data_ptrs = {cache.data_ptr() for cache in kv_caches.values()}
            if len(storage_ptrs) == 1 and len(data_ptrs) > 1:
                self._register_packed_kv_cache(storage)
                self.device_kv_caches = kv_caches
                return

        self.transfer_topo = TransferTopology(
            tp_rank=self.tp_rank,
            tp_size=self.world_size,
            block_size=self.block_size,
            engine_id=self.engine_id,
            is_mla=self.use_mla,
            total_num_kv_heads=self.model_config.get_total_num_kv_heads(),
            attn_backends=self.attn_backends,
            # SSM States come in tuples (ssm, conv)
            tensor_shape=next(iter(kv_caches.values())).shape
            if not self._has_mamba
            else None,
            is_mamba=self._has_mamba,
        )
        self.compat_hash = compute_nixl_compatibility_hash(
            self.vllm_config, self.backend_name, self.transfer_topo.cross_layers_blocks
        )

        if self.use_host_buffer:
            self.initialize_host_xfer_buffer(kv_caches=kv_caches)
            assert len(self.host_xfer_buffers) == len(kv_caches), (
                f"host_buffer: {len(self.host_xfer_buffers)}, "
                f"kv_caches: {len(kv_caches)}"
            )
            xfer_buffers = self.host_xfer_buffers
        else:
            xfer_buffers = kv_caches
            assert not self.host_xfer_buffers, (
                "host_xfer_buffer should not be initialized when "
                f"kv_buffer_device is {self.kv_buffer_device}"
            )

        logger.info(
            "Registering KV_Caches. use_mla: %s, kv_buffer_device: %s, "
            "use_host_buffer: %s",
            self.use_mla,
            self.kv_buffer_device,
            self.use_host_buffer,
        )

        caches_data = []
        # With hybrid allocator, layers can share a kv cache tensor
        seen_base_addresses = []
        region_semantic_names: dict[int, list[str]] = defaultdict(list)
        region_group_indices: dict[int, set[int]] = defaultdict(set)
        region_group_semantic_names: dict[int, dict[int, list[str]]] = {}
        region_registered_bytes: dict[int, int] = {}

        # Note(tms): I modified this from the original region setup code.
        # K and V are now in different regions. Advantage is that we can
        # elegantly support MLA and any cases where the K and V tensors
        # are non-contiguous (it's not locally guaranteed that they will be)
        # Disadvantage is that the encoded NixlAgentMetadata is now larger
        # (roughly 8KB vs 5KB).
        # Conversely for FlashInfer, K and V are registered in the same region
        # to better exploit the memory layout (ie num_blocks is the first dim).
        tensor_size_bytes = None

        for layer_name, cache_or_caches in xfer_buffers.items():
            # NOTE (NickLucche) Hybrid SSM models assume a layout that is similar to
            # that of FI, with block laid out as in `get_backend_aware_kv_block_len`.
            # However, physical page_size may differ when kernel requires a specific
            # block size. This leads to SSM and FA layers having different num_blocks.
            # `_physical_blocks_per_logical_kv_block` ratio is used to adjust for this.
            layer_spec = self._layer_specs.get(layer_name)
            if layer_spec is None:
                logger.debug(
                    "Skipping layer %s as no KVCache spec is present. "
                    "This is likely because the layer is sharing its KV cache",
                    layer_name,
                )
                continue
            if isinstance(layer_spec, UniformTypeKVCacheSpecs):
                # MLA DSv32 Indexer case: UniformTypeKVCacheSpecs merges kv_cache_specs
                layer_spec = layer_spec.kv_cache_specs[layer_name]
            cache_list = self.transfer_topo.get_transfer_cache_regions(
                cache_or_caches, layer_spec
            )
            # `layer_spec.page_size_bytes` only accounts for logical page_size, that is
            # the page_size assuming constant `self._logical_num_blocks`.
            physical_page_size = (
                layer_spec.page_size_bytes
                if isinstance(layer_spec, MambaSpec)
                else layer_spec.page_size_bytes
                // self._physical_blocks_per_logical_kv_block
            )
            # For when registering multiple tensors eg K/V in separate regions.
            physical_page_size = physical_page_size // len(cache_list)
            if self.transfer_topo._cross_layers_blocks:
                # When cross-layers blocks are used, multiply by number of layers
                physical_page_size = physical_page_size * len(
                    self.kv_cache_config.kv_cache_tensors
                )
            num_blocks = (
                self._logical_num_blocks
                if isinstance(layer_spec, MambaSpec)
                else self.num_blocks
            )
            # `page_size` accounts for physical blocks, st KVCache is always
            # [`num_blocks` * `page_size`]
            curr_tensor_size_bytes = num_blocks * physical_page_size

            # TODO (NickLucche) we could eventually unify how we handle FA/FI regions,
            # registering a single tensor for both K/V and splitting logically like FI.
            for cache_index, cache in enumerate(cache_list):
                base_addr = cache.data_ptr()
                region_semantic_names[base_addr].append(
                    f"{layer_name}:transfer_region_{cache_index}"
                )
                group_index = self._layer_to_group_index.get(layer_name)
                if group_index is None:
                    raise LocalizationError(
                        f"registered KV layer {layer_name} has no cache-group owner"
                    )
                region_group_indices[base_addr].add(group_index)
                names_by_group = region_group_semantic_names.setdefault(
                    base_addr,
                    {},
                )
                names_by_group.setdefault(group_index, []).append(
                    f"{layer_name}:transfer_region_{cache_index}"
                )
                if base_addr in seen_base_addresses:
                    # NOTE (NickLucche) HMA employs memory pooling to share tensors
                    # across groups. This results in skipping all tensors but the ones
                    # pointed to by group0. Also, generally we will have more blocks
                    # per tensor but fewer regions.
                    logger.debug("Skipping %s because it's already seen", layer_name)
                    continue
                logger.debug(
                    "Registering layer %s with cache shape: %s", layer_name, cache.shape
                )
                seen_base_addresses.append(base_addr)
                region_registered_bytes[base_addr] = curr_tensor_size_bytes
                self._region_tensors.append(cache)
                # Only record non-Mamba page sizes.
                if isinstance(layer_spec, MambaSpec):
                    self.block_len_per_layer.append(
                        physical_page_size // self._physical_blocks_per_logical_kv_block
                    )
                else:
                    self.block_len_per_layer.append(physical_page_size)
                is_mla_region = isinstance(
                    layer_spec, (MLAAttentionSpec, SlidingWindowMLASpec)
                )
                self._region_is_mla.append(is_mla_region)

                if not is_mla_region:
                    if tensor_size_bytes is None:
                        tensor_size_bytes = curr_tensor_size_bytes
                    assert tensor_size_bytes == curr_tensor_size_bytes, (
                        "All non-MLA kv cache tensors must have the same size"
                    )

                if cache.shape[0] != num_blocks:
                    raise AssertionError(
                        "All kv cache tensors must have the same number of "
                        f"blocks; layer={layer_name}, "
                        f"expected_num_blocks={num_blocks}, "
                        f"cache_shape={tuple(cache.shape)}, "
                        f"cache_stride={tuple(cache.stride())}, "
                        f"layer_spec={type(layer_spec).__name__}, "
                        f"backend={self.backend_name}, "
                        "all_backends="
                        f"{[backend.get_name() for backend in self.attn_backends]}, "
                        f"kv_cache_layout={self.kv_cache_layout}, "
                        "blocks_first="
                        f"{self.transfer_topo.is_kv_layout_blocks_first}"
                    )

                # Need to make sure the device ID is non-negative for NIXL,
                # Torch uses -1 to indicate CPU tensors.
                self.device_id = max(cache.get_device(), 0)
                caches_data.append(
                    (base_addr, curr_tensor_size_bytes, self.device_id, "")
                )

        logger.debug(
            "Different block lengths collected: %s", set(self.block_len_per_layer)
        )
        assert (
            len(self.block_len_per_layer)
            == len(seen_base_addresses)
            == len(self._region_is_mla)
        )
        registered_layout = (
            self.kv_cache_layout
            if not self.use_host_buffer
            else self.host_buffer_kv_cache_layout
        )
        self._region_descriptors = tuple(
            NixlRegionDescriptor(
                semantic_name="|".join(sorted(region_semantic_names[base_addr])),
                group_indices=tuple(sorted(region_group_indices[base_addr])),
                group_semantic_names=tuple(
                    (group_index, "|".join(sorted(names)))
                    for group_index, names in sorted(
                        region_group_semantic_names[base_addr].items()
                    )
                ),
                base_address=base_addr,
                registered_bytes=region_registered_bytes[base_addr],
                row_bytes=self.block_len_per_layer[index],
                shape=tuple(int(dim) for dim in self._region_tensors[index].shape),
                strides=tuple(
                    int(stride) for stride in self._region_tensors[index].stride()
                ),
                dtype=str(self._region_tensors[index].dtype),
                element_size_bytes=int(self._region_tensors[index].element_size()),
                layout=registered_layout,
            )
            for index, base_addr in enumerate(seen_base_addresses)
        )

        self.kv_caches_base_addr[self.engine_id][self.tp_rank] = seen_base_addresses
        self.num_regions = len(caches_data)

        if self.transfer_topo.virtually_split_kv_in_blocks:
            # NOTE (NickLucche) When FlashInfer is used, memory is registered
            # with joint KV for each block. This minimizes the overhead in
            # registerMem allowing faster descs queries. In order to be able to
            # split on kv_heads dim as required by heterogeneous TP, one must
            # be able to index K/V separately. Hence we double the number
            # of 'virtual' regions here and halve `block_len` below.
            # Similarly for Mamba layers, we register SSM+Conv as a single region and
            # then duplicate it logically to be able to index SSM/Conv separately.
            # Exception: key-only REPLICATE regions (MLA) have no V half, so
            # they contribute a single desc stream and are not doubled.
            self.num_regions = sum(
                1 if self._is_region_replicated(i) else 2
                for i in range(len(self._region_is_mla))
            )

        # Total local FA descriptors (boundary between FA and mamba descs).
        self.num_descs = self.num_regions * self.num_blocks

        descs = self.nixl_wrapper.get_reg_descs(caches_data, self.nixl_memory_type)
        logger.debug("Registering descs: %s", caches_data)
        self.nixl_wrapper.register_memory(descs, backends=self.nixl_backends)
        logger.debug("Done registering descs")
        self._registered_descs.append(descs)

        self.device_kv_caches = kv_caches
        self.dst_num_blocks[self.engine_id] = self.num_blocks

        if self._has_mamba:
            logger.info(
                "Hybrid SSM registration: num_blocks=%s, "
                "logical_num_blocks=%s, ratio=%s, num_regions=%s, "
                "num_descs=%s, mamba_ssm_size=%s, block_len_per_layer=%s",
                self.num_blocks,
                self._logical_num_blocks,
                self._physical_blocks_per_logical_kv_block,
                self.num_regions,
                self.num_descs,
                self._mamba_ssm_size,
                set(self.block_len_per_layer),
            )

        # Register local/src descr for NIXL xfer.
        self.src_xfer_handles_by_block_size[self.block_size], self.src_blocks_data = (
            self.register_local_xfer_handler(self.block_size)
        )

        # After KV Caches registered, listen for new connections.
        agent_metadata = NixlAgentMetadata(
            engine_id=self.engine_id,
            agent_metadata=self.nixl_wrapper.get_agent_metadata(),
            device_id=self.device_id,
            kv_caches_base_addr=self.kv_caches_base_addr[self.engine_id][self.tp_rank],
            num_blocks=self.num_blocks,
            block_lens=self.block_len_per_layer,
            kv_cache_layout=self.kv_cache_layout
            if not self.use_host_buffer
            else self.host_buffer_kv_cache_layout,
            block_size=self.block_size,
            ssm_sizes=self._mamba_ssm_size,
            attn_backend_name=self.backend_name,
            physical_blocks_per_logical_kv_block=(
                self._physical_blocks_per_logical_kv_block
            ),
            registration_generation=self._registration_generation,
            regions=self._region_descriptors,
            source_group_planes=tuple(
                1 if flag else 2 for flag in self._sp_group_flags()
            ),
            physical_group_token_capacities=(self._physical_group_token_capacities()),
        )
        # Wrap metadata in payload with hash for defensive decoding
        assert self.compat_hash is not None
        encoder = msgspec.msgpack.Encoder()
        self.xfer_handshake_metadata = NixlHandshakePayload(
            compatibility_hash=self.compat_hash,
            agent_metadata_bytes=encoder.encode(agent_metadata),
        )

    def _build_mamba_local(
        self,
        base_addresses: list[int],
        block_size_ratio: int,
    ) -> list[tuple[int, int, int]]:
        """Build desc regions (conv sub-projections + ssm) per layer for
        local mamba blocks with DS conv layout."""
        assert block_size_ratio == 1, (
            "Mamba 3-read transfer with block_size_ratio != 1 is not tested. "
            f"Got block_size_ratio={block_size_ratio}."
        )
        assert self._conv_decomp is not None
        conv_offsets = self._conv_decomp.local_conv_offsets
        conv_size, ssm_size = self._mamba_ssm_size
        num_blocks = self._logical_num_blocks * block_size_ratio
        physical_per_logical = self._physical_blocks_per_logical_kv_block

        result: list[tuple[int, int, int]] = []
        for i, base_addr in enumerate(base_addresses):
            # Jump one page_size, but ssm page_size may be bigger when kernel
            # locks block size to a specific value (physical_per_logical scale).
            page_stride = (
                self.block_len_per_layer[i] // block_size_ratio * physical_per_logical
            )
            for off, sz in conv_offsets:
                for blk in range(num_blocks):
                    result.append(
                        (base_addr + blk * page_stride + off, sz, self.device_id)
                    )
            # SSM temporal state follows the conv state.
            for blk in range(num_blocks):
                result.append(
                    (
                        base_addr + blk * page_stride + conv_size,
                        ssm_size,
                        self.device_id,
                    )
                )
        return result

    def _build_mamba_remote(
        self,
        nixl_agent_meta: NixlAgentMetadata,
        tp_ratio: int,
        transfer_info: EngineTransferInfo,
    ) -> list[tuple[int, int, int]]:
        """Build remote desc regions (conv sub-projections + ssm) per layer.
        For hetero-TP, each D rank reads only its sub-projection slice from
        the P rank."""
        assert self._conv_decomp is not None
        effective_ratio = max(tp_ratio, 1)
        # Mamba conv state is always TP-sharded, even when attention KV
        # is replicated (num_kv_heads < tp_size).
        local_offset = self.tp_rank % effective_ratio
        conv_size_remote = nixl_agent_meta.ssm_sizes[0]

        conv_offsets = self._conv_decomp.remote_conv_offsets(local_offset, tp_ratio)
        if tp_ratio >= 1:
            ssm_read_size = self._mamba_ssm_size[1]
        else:
            ssm_read_size = nixl_agent_meta.ssm_sizes[1]

        remote_physical_per_logical = transfer_info.remote_physical_blocks_per_logical
        num_blocks = nixl_agent_meta.num_blocks // remote_physical_per_logical
        device_id = nixl_agent_meta.device_id

        result: list[tuple[int, int, int]] = []
        # NOTE (ZhanqiuHu): use per-layer block_lens[i], not [0], in case
        # block lengths vary across layers (e.g. MLA).
        for i, base_addr in enumerate(nixl_agent_meta.kv_caches_base_addr):
            page_stride = nixl_agent_meta.block_lens[i] * remote_physical_per_logical
            for off, sz in conv_offsets:
                for blk in range(num_blocks):
                    result.append((base_addr + blk * page_stride + off, sz, device_id))
            # SSM temporal state is also TP-sharded on the heads dimension.
            for blk in range(num_blocks):
                ssm_addr = (
                    base_addr
                    + blk * page_stride
                    + conv_size_remote
                    + local_offset * ssm_read_size
                )
                result.append((ssm_addr, ssm_read_size, device_id))
        return result

    def _build_fa_local(
        self,
        base_addresses: list[int],
        block_size_ratio: int,
    ) -> list[tuple[int, int, int]]:
        """Build local FA descriptors for all layers."""
        assert self.transfer_topo is not None
        num_blocks = self.num_blocks * block_size_ratio
        result: list[tuple[int, int, int]] = []
        for i, base_addr in enumerate(base_addresses):
            kv_block_len = (
                self.get_backend_aware_kv_block_len(
                    layer_idx=i, first_split=True, mamba_view=False
                )
                // block_size_ratio
            )
            page_stride = self.block_len_per_layer[i] // block_size_ratio
            for block_id in range(num_blocks):
                block_offset = block_id * page_stride
                addr = base_addr + block_offset
                result.append((addr, kv_block_len, self.device_id))

            if (
                self.transfer_topo.virtually_split_kv_in_blocks
                and not self._is_region_replicated(i)
            ):
                # Separate and interleave K/V regions to maintain the same
                # descs ordering. This is needed for selecting contiguous heads
                # when split across TP ranks. (Skipped for key-only REPLICATE.)
                second_split = self.get_backend_aware_kv_block_len(
                    layer_idx=i, first_split=False, mamba_view=False
                )
                for block_id in range(num_blocks):
                    block_offset = block_id * page_stride
                    addr = base_addr + block_offset
                    v_addr = addr + kv_block_len
                    result.append((v_addr, second_split, self.device_id))
        return result

    def _build_fa_remote(
        self,
        plan: TPMapping,
        nixl_agent_meta: NixlAgentMetadata,
        block_size_ratio: int,
    ) -> list[tuple[int, int, int]]:
        """Build remote FA descriptors for all layers."""
        assert self.transfer_topo is not None
        fa_group_idx = next(
            i for i, t in enumerate(self._group_spec_types) if _is_attention_spec(t)
        )
        # SPLIT regions read their head slice from this many remote ranks at a
        # per-rank offset; REPLICATE regions read the whole block once.
        split_reads = len(plan.source_ranks_per_group[fa_group_idx])
        num_blocks = nixl_agent_meta.num_blocks
        result: list[tuple[int, int, int]] = []
        for i, base_addr in enumerate(nixl_agent_meta.kv_caches_base_addr):
            replicated = self._is_region_replicated(i)
            # Read our whole local region size from remote..
            local_block_len = self.get_backend_aware_kv_block_len(
                layer_idx=i, first_split=True, mamba_view=False
            )
            remote_kv_block_len = local_block_len // block_size_ratio
            if block_size_ratio > 1:
                # ..using remote kv_block_len as transfer unit
                local_block_len = remote_kv_block_len

            # REPLICATE reads the whole block once at offset 0; SPLIT gathers
            # its head slice from `split_reads` remote ranks at a per-rank offset.
            num_reads = 1 if replicated else split_reads
            rank_offset = (
                0 if replicated else plan.rank_offset_factor * remote_kv_block_len
            )
            local_block_len = local_block_len // num_reads

            page_size = nixl_agent_meta.block_lens[i]
            for block_id in range(num_blocks):
                block_offset = block_id * page_size
                # For each block, grab the kv heads chunk belonging to current local
                # tp rank of size local_block_len.
                addr = base_addr + block_offset + rank_offset
                result.append((addr, local_block_len, nixl_agent_meta.device_id))

            emits_v = self.transfer_topo.virtually_split_kv_in_blocks and not replicated
            if emits_v:
                # With FlashInfer index V separately to allow head splitting.
                second_split = self.get_backend_aware_kv_block_len(
                    layer_idx=i, first_split=False, mamba_view=False
                )
                second_split = second_split // num_reads
                for block_id in range(num_blocks):
                    block_offset = block_id * page_size
                    addr = base_addr + block_offset + rank_offset
                    # Hop over the first split of remote page, K, to read V.
                    v_addr = addr + nixl_agent_meta.block_lens[i] // 2
                    result.append((v_addr, second_split, nixl_agent_meta.device_id))
        return result

    def register_local_xfer_handler(
        self,
        block_size: int,
    ) -> tuple[int, list[tuple[int, int, int]]]:
        """
        Function used for register local xfer handler with local block_size or
        Remote block_size.

        When local block_size is same as remote block_size, we use local block_size
        to register local_xfer_handler during init.

        When remote block size is less than local block size, we need to use
        register another local_xfer_handler using remote block len to ensure
        data copy correctness.
        """
        assert self.transfer_topo is not None
        block_size_ratio = self.block_size // block_size
        local_base_addresses = self.kv_caches_base_addr[self.engine_id][self.tp_rank]

        blocks_data = self._build_fa_local(local_base_addresses, block_size_ratio)
        logger.debug(
            "Created %s blocks for src engine %s and rank %s on device id %s",
            len(blocks_data),
            self.engine_id,
            self.tp_rank,
            self.device_id,
        )
        if self._has_mamba:
            assert self.num_descs == len(blocks_data)
            # TODO (ZhanqiuHu): For homogeneous TP (tp_ratio == 1), the 3-descs split
            # is unnecessary — a single conv desc per block suffices.  Consider
            # adding a fast path that falls back to the standard 2-region
            # registration (_build_fa_local mamba=True) when no hetero-TP
            # remote has been seen.  Currently we always register 4 regions
            # because local descs are created before knowing the remote TP.
            logger.debug("Registering local Mamba descriptors (4 regions/layer)")
            blocks_data.extend(
                self._build_mamba_local(local_base_addresses, block_size_ratio)
            )

        descs = self.nixl_wrapper.get_xfer_descs(blocks_data, self.nixl_memory_type)
        # NIXL_INIT_AGENT to be used for preparations of local descs.
        return self.nixl_wrapper.prep_xfer_dlist("NIXL_INIT_AGENT", descs), blocks_data

    def add_remote_agent(
        self,
        nixl_agent_meta: NixlAgentMetadata,
        remote_tp_rank: int = 0,
        remote_tp_size: int = 1,
    ) -> str:
        """
        Add the remote NIXL agent and prepare the descriptors for reading cache
        blocks from remote.

        In particular, handle both homogeneous and heterogeneous TP. The former
        requires local rank_i to read from remote rank_i.
        The latter, in the case of D.world_size < P.world_size, requires that a
        local (D) TP worker reads from multiple remote (P) TP workers.
        Conversely, assuming D.world_size > P.world_size, two or more local TP
        workers will read from a single remote TP worker.

        Here's an example for the last case described above (non-MLA):

        rank_offset     p_remote_tp_rank
        (kv split no)
        --------------------------------
            0                 0      Worker0  ---- 1st half of KV ----> Worker0  [ KV Cache ]
                                                                        /
            1                 0      Worker1  ---- 2nd half of KV -----/

            0                 1      Worker2  ---- 1st half of KV ----> Worker1  [ KV Cache ]
                                                                        /
            1                 1      Worker3  ---- 2nd half of KV -----/


                                Decoder TP workers                     Prefix TP workers
                                  (world_size=4)                         (world_size=2)
                                                 tp_ratio = 4 // 2 = 2

        Considering the KV Caches, if P-Worker_i has cache size [2, num_blocksP, kv_heads, block_size, head_dim]
        then D-Worker_j has [2, num_blocksD, kv_heads//tp_ratio, block_size, head_dim]. Mind the "HND" layout format.
        Assuming num_blocksD >= num_blocksP, D-Worker0 reads from P-Worker0 by preparing the kv_heads//tp_ratio
        first heads from all the slots of all the blocks. D-Worker1 will do the same, but reading the second split
        along the kv_heads dimension, and so forth until "tp_ratio" D TP workers have pulled from P-Worker0.

        Note that the above will also hold true for the homogeneous TP case, where tp_ratio evaluates to 1.

        Regarding MLA case, the cache is replicated across TP workers so the rank_offset will just always be 0
        so that the whole cache is shared by "tp_ratio" D TP workers.

        For Mamba hetero-TP, both tp_ratio > 0 (D_TP > P_TP) and
        tp_ratio < 0 (P_TP > D_TP) are supported by the 3-read transfer.
        """  # noqa: E501
        engine_id = nixl_agent_meta.engine_id
        # TODO re-evaluate refreshing for scaling/recovery
        if remote_tp_rank in self._remote_agents.get(engine_id, {}):
            logger.debug(
                "Remote agent with engine_id %s and rank"
                "%s already exchanged metadata, skip handshake.",
                engine_id,
                remote_tp_rank,
            )
            return self._remote_agents[engine_id][remote_tp_rank]

        ### Register remote engine in TransferTopology (idempotent).
        assert self.transfer_topo is not None
        transfer_topo = self.transfer_topo
        physical_blocks_per_logical = (
            nixl_agent_meta.physical_blocks_per_logical_kv_block
        )
        transfer_info = EngineTransferInfo(
            remote_tp_size=remote_tp_size,
            remote_block_size=nixl_agent_meta.block_size,
            remote_block_len=nixl_agent_meta.block_lens[0],
            remote_physical_blocks_per_logical=physical_blocks_per_logical,
        )
        transfer_topo.register_remote_engine(engine_id, transfer_info)
        logger.info("Transfer plan: %s", transfer_topo.describe(engine_id))

        self.tp_mappings[engine_id] = compute_tp_mapping(
            transfer_topology=transfer_topo,
            remote_tp_size=remote_tp_size,
            group_spec_types=self._group_spec_types,
        )

        remote_agent_name = self.nixl_wrapper.add_remote_agent(
            nixl_agent_meta.agent_metadata
        )

        # Create dst descs and xfer side handles. TP workers have same #blocks
        # so we only register once per engine_id.
        # Example:
        # block_size_ratio > 1:
        # remote:               | 0| 1| 2| 3| 4| 5| 6| 7| 8| 9|10|11|12|
        # local origin:|          0|          1|          8|         12|
        # local mapped:| 0| 1| 2| 3| 4| 5| 6| 7| 8| 9|10|11|12|13|14|15|
        block_size_ratio = transfer_topo.block_size_ratio(nixl_agent_meta.block_size)

        if engine_id not in self.dst_num_blocks:
            self.dst_num_blocks[engine_id] = nixl_agent_meta.num_blocks

        self._validate_remote_agent_handshake(nixl_agent_meta, remote_tp_size)

        # Keep track of remote agent kv caches base addresses.
        self.kv_caches_base_addr[engine_id][remote_tp_rank] = (
            nixl_agent_meta.kv_caches_base_addr
        )
        # Retain per-region layout facts for the coalesced pull path
        # (raw range descriptors are built from these at transfer time).
        self._remote_layout[engine_id][remote_tp_rank] = (
            list(nixl_agent_meta.block_lens),
            nixl_agent_meta.num_blocks,
            nixl_agent_meta.device_id,
        )
        self._remote_regions[engine_id][remote_tp_rank] = nixl_agent_meta.regions
        self._remote_registration_generations[engine_id][remote_tp_rank] = (
            nixl_agent_meta.registration_generation
        )
        self._remote_source_semantics[engine_id][remote_tp_rank] = (
            nixl_agent_meta.source_group_planes,
            nixl_agent_meta.physical_group_token_capacities,
        )

        # This is 1 when P and D `--tensor-parallel-size` match. Otherwise,
        # this is the ratio between the two sizes.
        tp_ratio = transfer_topo.tp_ratio(remote_tp_size)

        logger.debug(
            "Registering remote agent (%s, rank %s) memory regions with tp_ratio %s",
            engine_id,
            remote_tp_rank,
            tp_ratio,
        )

        plan = self.tp_mappings[engine_id]

        ### (Optional) Register local agent memory regions. MLA is not split.
        if (
            tp_ratio < 0
            and not self.use_mla
            and tp_ratio not in self.src_xfer_handles_by_tp_ratio
        ):
            # Remote tp_size > local tp_size: read from multiple remote ranks.
            # Logically "split" own regions into |tp_ratio| chunks. Mind that
            # we only do this once per remote tp_size (replica-friendly).
            self.src_xfer_handles_by_tp_ratio[tp_ratio] = []

            for handle_data in self._build_local_splits_from_plan(
                plan,
                self.src_blocks_data,
                self.num_descs,
            ):
                descs = self.nixl_wrapper.get_xfer_descs(
                    handle_data, self.nixl_memory_type
                )
                handle = self.nixl_wrapper.prep_xfer_dlist("NIXL_INIT_AGENT", descs)
                self.src_xfer_handles_by_tp_ratio[tp_ratio].append(handle)

        ### Register remote agent memory regions
        # With homogeneous TP, D pulls the whole kv cache from corresponding rank. With
        # heterogeneous TP, prepare the descriptors by splitting the P KV cache along
        # kv_head dim, of D worker's kv_head size (D>P).
        # Eg. PTP1 DTP2 => P0 KV:[block0-KV_0 | block0-KV_1..].

        # Register all remote blocks, but only the corresponding kv heads.
        blocks_data = self._build_fa_remote(
            plan,
            nixl_agent_meta,
            block_size_ratio,
        )
        logger.debug(
            "Created %s blocks for dst engine %s with remote rank %s and local rank %s",
            len(blocks_data),
            engine_id,
            remote_tp_rank,
            self.tp_rank,
        )
        if self._has_mamba:
            logger.debug(
                "Registering remote Mamba blocks for engine %s rank %s",
                engine_id,
                remote_tp_rank,
            )
            blocks_data.extend(
                self._build_mamba_remote(
                    nixl_agent_meta,
                    tp_ratio,
                    transfer_info,
                )
            )

        # Register with NIXL.
        descs = self.nixl_wrapper.get_xfer_descs(blocks_data, self.nixl_memory_type)
        self.dst_xfer_side_handles[engine_id][remote_tp_rank] = (
            self.nixl_wrapper.prep_xfer_dlist(remote_agent_name, descs)
        )

        if block_size_ratio > 1:
            # when prefill with smaller block_size, we need to init a
            # new handler with same block_len to match
            self.src_xfer_handles_by_block_size[nixl_agent_meta.block_size] = (
                self.register_local_xfer_handler(nixl_agent_meta.block_size)[0]
            )

        return remote_agent_name

    def _validate_remote_agent_handshake(
        self, nixl_agent_meta: NixlAgentMetadata, remote_tp_size: int
    ):
        """
        Validate the remote agent handshake metadata ensuring the
        invariants hold true.
        """
        remote_engine_id = nixl_agent_meta.engine_id

        assert self.transfer_topo is not None
        remote_info = self.transfer_topo.get_engine_info(remote_engine_id)
        assert remote_info.remote_tp_size == remote_tp_size

        tp_ratio = self.transfer_topo.tp_ratio(remote_tp_size)
        block_size_ratio = self.transfer_topo.block_size_ratio(
            nixl_agent_meta.block_size
        )
        # num_kv_heads > tp_size with P_TP > D_TP not supported for non-mamba.
        # Mamba models can have replicated FA KV with tp_ratio < 0.
        # MLA models do not need to handle kv replication.
        if not self.use_mla and not self._has_mamba:
            assert not (
                tp_ratio < 0 and self.transfer_topo.is_kv_replicated(remote_engine_id)
            )

        remote_physical_per_logical = (
            nixl_agent_meta.physical_blocks_per_logical_kv_block
        )
        if (
            self._has_mamba
            and remote_physical_per_logical
            != self._physical_blocks_per_logical_kv_block
            and self.vllm_config.cache_config.enable_prefix_caching
        ):
            raise RuntimeError(
                "Prefix caching with heterogeneous physical_blocks_per_logical "
                "is not supported for Mamba hybrid models. "
                f"Local: {self._physical_blocks_per_logical_kv_block}, "
                f"Remote: {remote_physical_per_logical}. "
                "Disable prefix caching with --no-enable-prefix-caching."
            )

        if self._is_hma_required:
            assert block_size_ratio == 1, (
                "HMA does not support different remote block size yet"
            )
        kv_cache_layout = (
            self.kv_cache_layout
            if not self.use_host_buffer
            else self.host_buffer_kv_cache_layout
        )
        if not self.use_mla and nixl_agent_meta.kv_cache_layout != kv_cache_layout:
            if (
                self.kv_transfer_config.enable_permute_local_kv
                and nixl_agent_meta.kv_cache_layout == "HND"
            ):
                logger.info(
                    "Remote is HND and local is NHD, enabled additional permute "
                    "on local device KV."
                )
                assert not self._is_hma_required, (
                    "HMA does not support block size post processing"
                )
                self.enable_permute_local_kv = True
            else:
                raise RuntimeError(
                    "Heterogeneous TP expects same kv_cache_layout. "
                    "Or enable experimental feature to use HND to NHD support by "
                    "setting 'enable_permute_local_kv'=True in --kv-transfer-config."
                )
        # if remote_agent used attn is not same as local,
        # hint heterogenuous attn post process
        if (
            nixl_agent_meta.attn_backend_name != self.backend_name
            and self.backend_name in ["CPU_ATTN"]
        ):
            if self._is_hma_required:
                raise RuntimeError(
                    "heterogeneous attn post process is not supported with HMA"
                )
            logger.info(
                "[Experimental] CPU_ATTN backend is used, "
                "hint heterogeneous attn post process"
            )
            self.enable_heterogeneous_attn_post_process = True

        # Heterogeneous TP requires head-splitting, which only works with
        # HND layout. MLA and replicated-KV cases don't split on heads.
        # Mamba doesn't support heterogeneous TP.
        if (
            abs(tp_ratio) != 1
            and not self.use_mla
            and not self.transfer_topo.is_kv_replicated(remote_engine_id)
            and kv_cache_layout != "HND"
            and not self.enable_permute_local_kv
        ):
            raise RuntimeError(
                "Heterogeneous TP head-dimension splitting requires contiguous heads. "
                "Use HND layout on the prefill side."
            )

        # Per-region block_len validation enforcing the P/D invariant.
        # REPLICATE regions (MLA, or a whole-model MLA / replicated-KV transfer)
        # only allow the number of blocks to differ; SPLIT regions scale with
        # the per-rank KV head ratio rather than the raw tp_ratio, because GQA
        # replication caps per-rank heads at 1 when tp > total_kv_heads
        # (issue #45330). Mamba uses the ssm_sizes counterpart, so skip here.
        if not self._has_mamba:
            assert len(self.block_len_per_layer) == len(nixl_agent_meta.block_lens), (
                "Number of KV layers must match between prefill and decode"
            )
            model_replicated = self.use_mla or self.transfer_topo.is_kv_replicated(
                remote_engine_id
            )
            total_kv_heads = self.transfer_topo.total_num_kv_heads
            local_heads = self.transfer_topo.local_physical_heads
            remote_heads = max(1, total_kv_heads // remote_tp_size)
            for i, local_len in enumerate(self.block_len_per_layer):
                replicated = model_replicated or self._is_region_replicated(i)
                remote_len = nixl_agent_meta.block_lens[i]
                if replicated:
                    assert local_len // block_size_ratio == remote_len, (
                        "KV cache sizes must match between P and D when "
                        f"replicated (region {i}: local={local_len}, "
                        f"remote={remote_len}, bsr={block_size_ratio})."
                    )
                elif tp_ratio > 0:
                    assert (
                        remote_len
                        == (local_len * remote_heads // local_heads) // block_size_ratio
                    ), (
                        f"SPLIT region {i}: remote P KV block_len {remote_len} "
                        f"must equal local {local_len} * remote_heads "
                        f"{remote_heads} // local_heads {local_heads} "
                        f"// block_size_ratio {block_size_ratio}."
                    )
                else:
                    assert block_size_ratio == 1, (
                        "Different local/remote block sizes are not supported "
                        "when P TP > D TP."
                    )
                    assert remote_len == local_len * remote_heads // local_heads, (
                        f"SPLIT region {i}: remote P KV block_len {remote_len} "
                        f"must equal local {local_len} * remote_heads "
                        f"{remote_heads} // local_heads {local_heads}."
                    )

        # TP workers that handhshake with same remote have same #blocks.
        assert self.dst_num_blocks[remote_engine_id] == nixl_agent_meta.num_blocks
        # Same number of regions/~layers.
        assert len(nixl_agent_meta.kv_caches_base_addr) == len(self.block_len_per_layer)

    def sync_recved_kv_to_device(self, req_id: str, meta: ReqMeta):
        """copy recved kv from host buffer to device."""
        assert self.use_host_buffer
        assert self.copy_blocks is not None

        local_block_ids = meta.local_physical_block_ids
        # TODO (NickLucche) D2H<>H2D ops could benefit from coalescing io across groups
        for group_block_ids in local_block_ids:
            self.copy_blocks(
                self.host_xfer_buffers,
                self.device_kv_caches,
                group_block_ids,
                group_block_ids,
                "h2d",
            )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "synced recved kv of request[%s] to device kv buffer,"
                "local_block_ids: %s. ",
                req_id,
                ",".join(map(str, local_block_ids)),
            )

    def save_kv_to_host(self, metadata: NixlConnectorMetadata):
        """copy kv from device to host buffer."""
        assert self.use_host_buffer
        assert self.copy_blocks is not None

        for req_id, meta in metadata.reqs_to_save.items():
            meta.local_physical_block_ids = self._logical_to_kernel_block_ids(
                meta.local_block_ids
            )
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "save_load_kv for request[%s] to host xfer buffer."
                    "local_block_ids: %s. ",
                    req_id,
                    ",".join(map(str, meta.local_physical_block_ids)),
                )
            # blocking
            for group_block_ids in meta.local_physical_block_ids:
                self.copy_blocks(
                    self.device_kv_caches,
                    self.host_xfer_buffers,
                    group_block_ids,
                    group_block_ids,
                    "d2h",
                )

    def post_process_device_kv_on_receive(
        self,
        block_size_ratio: int,
        block_ids_list: list[list[int]],
    ):
        """
        Post process device kv cache after receiving from remote.

        3 types of post processing supported:
            * kv_cache_postprocess_layout => convert from HND to NHD
            * kv_cache_postprocess_blksize => convert from small block size
              to large block size
            * kv_cache_postprocess_blksize_and_layout => convert from small
              block size to large block size and convert from HND to NHD

        """
        if len(self.device_kv_caches) == 0:
            return
        assert block_size_ratio >= 1, "Only nP < nD supported currently."
        assert self.transfer_topo is not None
        if self.enable_permute_local_kv and block_size_ratio > 1:
            logger.debug(
                "Post-processing device kv cache on receive by converting "
                "block_size with %sx bigger and permuting layout from HND"
                " to NHD.",
                block_size_ratio,
            )
        elif self.enable_permute_local_kv:
            logger.debug(
                "Post-processing device kv cache on receive by permuting layout"
                "from HND to NHD."
            )
        else:
            logger.debug(
                "Post-processing device kv cache on receive by converting "
                "block_size with %sx bigger.",
                block_size_ratio,
            )

        split_k_and_v = self.transfer_topo.split_k_and_v

        for block_ids in block_ids_list:
            indices = torch.tensor(block_ids, device=self.device_type, dtype=torch.long)

            for _, cache_or_caches in self.device_kv_caches.items():
                cache_list = cache_or_caches if split_k_and_v else [cache_or_caches]
                for cache in cache_list:
                    if self.enable_permute_local_kv and block_size_ratio > 1:
                        kv_postprocess_blksize_and_layout_on_receive(
                            cache, indices, block_size_ratio
                        )
                    elif self.enable_permute_local_kv:
                        kv_postprocess_layout_on_receive(cache, indices)
                    else:
                        kv_postprocess_blksize_on_receive(
                            cache, indices, block_size_ratio
                        )

    def post_process_device_kv_on_receive_heterogeneous_attn(
        self, block_ids: list[int]
    ):
        """
        Post process device kv cache after receiving from remote
        for heterogeneous attention.
        """
        assert self.enable_heterogeneous_attn_post_process

        indices = torch.tensor(block_ids, device=self.device_type, dtype=torch.long)

        for _, cache_or_caches in self.device_kv_caches.items():
            blocks_to_update = cache_or_caches.index_select(1, indices)
            current_platform.pack_kv_cache(
                key=blocks_to_update[0],
                value=blocks_to_update[1],
                key_cache=cache_or_caches[0],
                value_cache=cache_or_caches[1],
                block_ids=block_ids,
                indices=indices,
            )

    # ------------------------------------------------------------------
    # Diagnostic P-to-D source snapshots
    # ------------------------------------------------------------------

    def _localization_capture_source_rosters(
        self,
        rosters: dict[ReqId, NixlSourceRoster],
    ) -> None:
        """Retain newly offered producer allocations until read completion.

        :param rosters: Exact post-clipping logical block rosters from the
            scheduler.
        """
        if self._localization_config.enabled is False:
            if len(rosters) > 0:
                raise LocalizationError(
                    "source rosters arrived while localization is disabled"
                )
            return
        for req_id, logical_roster in rosters.items():
            if self._localization_config.enabled_for(req_id) is False:
                continue
            if req_id in self._localization_source_rosters:
                raise LocalizationError(f"duplicate source roster for {req_id}")
            physical_groups = self._logical_to_kernel_block_ids(
                [list(group) for group in logical_roster.block_ids]
            )
            physical_capacities = self._physical_group_token_capacities()
            if len(logical_roster.group_token_capacities) != len(
                physical_groups
            ) or len(physical_capacities) != len(physical_groups):
                raise LocalizationError(
                    f"source roster token-capacity mismatch for {req_id}"
                )
            for group_index, physical_capacity in enumerate(physical_capacities):
                factor = (
                    1
                    if isinstance(
                        self.kv_cache_config.kv_cache_groups[group_index].kv_cache_spec,
                        MambaSpec,
                    )
                    else self._physical_blocks_per_logical_kv_block
                )
                if (
                    logical_roster.group_token_capacities[group_index]
                    != physical_capacity * factor
                ):
                    raise LocalizationError(
                        f"source roster group {group_index} token capacity differs "
                        "from the registered physical contract"
                    )
            physical_roster = NixlSourceRoster(
                offer_generation=logical_roster.offer_generation,
                iteration=logical_roster.iteration,
                valid_token_extent=logical_roster.valid_token_extent,
                group_token_capacities=physical_capacities,
                block_ids=tuple(
                    tuple(int(block_id) for block_id in group)
                    for group in physical_groups
                ),
            )
            self._localization_source_rosters[req_id] = physical_roster

    def _localization_materialize_fingerprint_leaves(
        self,
        batches: list[torch.Tensor],
        specs: list[_LocalizationLeafSpec],
    ) -> tuple[NixlIntegrityLeaf, ...]:
        """Copy compact device fingerprints and attach canonical metadata.

        :param batches: Device fingerprint result inputs in leaf order.
        :param specs: Canonical identity and placement in the same order.
        :returns: Materialized localization leaves.
        """
        if self._localization_fingerprinter is None:
            raise LocalizationError("fingerprint mode has no device fingerprinter")
        fingerprints = self._localization_fingerprinter.fingerprints_to_digests(batches)
        if len(fingerprints) != len(specs):
            raise LocalizationError(
                "device fingerprint cardinality differs from leaf metadata"
            )
        return tuple(
            build_fingerprint_leaf(
                identity=spec.identity,
                fingerprint=fingerprint,
                local_block_id=spec.local_block_id,
                destination_half=spec.destination_half,
                rank_slot=spec.rank_slot,
            )
            for spec, fingerprint in zip(specs, fingerprints, strict=True)
        )

    def _localization_fingerprint_rows(self, rows: torch.Tensor) -> torch.Tensor:
        """Fingerprint one bounded payload batch without retaining its bytes.

        :param rows: Contiguous two-dimensional byte payloads.
        :returns: Compact device-resident fingerprint rows.
        """
        if self._localization_fingerprinter is None:
            raise LocalizationError("fingerprint mode has no device fingerprinter")
        return self._localization_fingerprinter.fingerprint_rows(rows)

    @staticmethod
    def _localization_leaf_mapping(
        leaves: tuple[NixlIntegrityLeaf, ...],
    ) -> dict[IntegrityLeafKey, tuple[int, int, int]]:
        """Build the decoder placement map for materialized capture leaves.

        :param leaves: Capture leaves with complete decoder placement metadata.
        :returns: Source keys mapped to local block, rank slot, and destination half.
        """
        mapping: dict[IntegrityLeafKey, tuple[int, int, int]] = {}
        for leaf in leaves:
            if (
                leaf.local_block_id is None
                or leaf.rank_slot is None
                or leaf.destination_half is None
            ):
                raise LocalizationError(
                    "decoder capture leaf lacks complete placement metadata"
                )
            mapping[leaf_source_key(leaf)] = (
                leaf.local_block_id,
                leaf.rank_slot,
                leaf.destination_half,
            )
        return mapping

    def _localization_capture_source_manifest(
        self,
        req_id: ReqId,
        roster: NixlSourceRoster,
        stage: IntegrityStage,
    ) -> NixlSourceManifest:
        """Hash one P-rank allocation with bounded host-copy memory.

        :param req_id: Producer request identifier.
        :param roster: Exact physical block roster.
        :param stage: Completed-read source checkpoint.
        :returns: Compact per-region manifest.
        """
        if stage is not IntegrityStage.SOURCE_POST:
            raise LocalizationError("producer snapshots are only valid at SOURCE_POST")
        if not self._coalesce_region_rows():
            raise LocalizationError("source KV regions are not canonicalizable")
        assert self._region_rows is not None
        if len(self._region_rows) == 0:
            raise LocalizationError("source snapshot has no registered KV regions")
        if len(self._region_descriptors) != len(self._region_rows):
            raise LocalizationError("source region descriptor cardinality mismatch")

        start_ns = time.perf_counter_ns()
        first_device = self._region_rows[0].device.type
        if first_device != "cpu":
            torch.accelerator.synchronize()

        source_group_planes = tuple(1 if flag else 2 for flag in self._sp_group_flags())
        positions = [
            (
                group_index,
                source_position,
                int(block_id),
                roster.group_token_capacities[group_index],
            )
            for group_index, group in enumerate(roster.block_ids)
            for source_position, block_id in enumerate(group)
        ]
        leaves: list[NixlIntegrityLeaf] = []
        fingerprint_batches: list[torch.Tensor] = []
        fingerprint_specs: list[_LocalizationLeafSpec] = []
        fingerprint_mode = (
            self._localization_config.mode is LocalizationMode.FINGERPRINT
        )
        copied_bytes = 0
        hashed_bytes = 0
        for region_index, rows in enumerate(self._region_rows):
            row_bytes = int(rows.shape[1])
            descriptor = self._region_descriptors[region_index]
            region_positions = [
                position
                for position in positions
                if position[0] in descriptor.group_indices
            ]
            if row_bytes <= 0:
                raise LocalizationError("source region row length must be positive")
            if descriptor.row_bytes != row_bytes:
                raise LocalizationError("source semantic row length mismatch")
            rows_per_chunk = max(
                1,
                self._localization_config.copy_chunk_bytes // row_bytes,
            )
            for chunk_start in range(0, len(region_positions), rows_per_chunk):
                chunk_positions = region_positions[
                    chunk_start : chunk_start + rows_per_chunk
                ]
                for _, _, block_id, _ in chunk_positions:
                    expected_address = (
                        descriptor.base_address + block_id * descriptor.row_bytes
                    )
                    actual_address = int(rows[block_id].data_ptr())
                    if actual_address != expected_address:
                        raise LocalizationError(
                            "source semantic row does not match registered interval"
                        )
                    if (
                        expected_address + descriptor.row_bytes
                        > descriptor.base_address + descriptor.registered_bytes
                    ):
                        raise LocalizationError(
                            "source row extends beyond registered memory"
                        )
                indices = torch.tensor(
                    [position[2] for position in chunk_positions],
                    device=rows.device,
                    dtype=torch.long,
                )
                selected_rows = rows.index_select(0, indices)
                host_rows = (
                    None if fingerprint_mode else selected_rows.cpu().contiguous()
                )
                if fingerprint_mode is False:
                    copied_bytes += len(chunk_positions) * row_bytes
                wire_specs: list[_LocalizationLeafSpec] = []
                commit_specs: tuple[
                    list[_LocalizationLeafSpec], list[_LocalizationLeafSpec]
                ] = ([], [])
                for row_index, (
                    group_index,
                    source_position,
                    block_id,
                    group_token_capacity,
                ) in enumerate(chunk_positions):
                    payload = (
                        None
                        if host_rows is None
                        else memoryview(host_rows[row_index].numpy()).cast("B")
                    )
                    semantic_contract_digest = compute_semantic_contract_digest(
                        region=descriptor,
                        group_index=group_index,
                        group_token_capacity=group_token_capacity,
                        source_plane_contract=source_group_planes[group_index],
                    )
                    wire_identity = build_integrity_identity(
                        config=self._localization_config,
                        producer_engine_id=self.engine_id,
                        producer_request_id=req_id,
                        registration_generation=self._registration_generation,
                        semantic_contract_digest=semantic_contract_digest,
                        offer_generation=roster.offer_generation,
                        iteration=roster.iteration,
                        source_rank=self.tp_rank,
                        region_index=region_index,
                        group_index=group_index,
                        plane_index=-1,
                        source_position=source_position,
                        remote_block_id=block_id,
                        valid_token_extent=roster.valid_token_extent,
                        group_token_capacity=group_token_capacity,
                        payload_kind=IntegrityPayloadKind.WIRE,
                        byte_length=row_bytes,
                    )
                    if fingerprint_mode:
                        wire_specs.append(
                            _LocalizationLeafSpec(
                                identity=wire_identity,
                                local_block_id=None,
                                destination_half=None,
                                rank_slot=None,
                            )
                        )
                    else:
                        assert payload is not None
                        leaves.append(
                            build_integrity_leaf(
                                identity=wire_identity,
                                payload=payload,
                                local_block_id=None,
                                destination_half=None,
                                rank_slot=None,
                            )
                        )
                    hashed_bytes += row_bytes
                    commit_bytes = row_bytes // 2
                    for source_plane in (0, 1):
                        commit_identity = build_integrity_identity(
                            config=self._localization_config,
                            producer_engine_id=self.engine_id,
                            producer_request_id=req_id,
                            registration_generation=self._registration_generation,
                            semantic_contract_digest=semantic_contract_digest,
                            offer_generation=roster.offer_generation,
                            iteration=roster.iteration,
                            source_rank=self.tp_rank,
                            region_index=region_index,
                            group_index=group_index,
                            plane_index=source_plane,
                            source_position=source_position,
                            remote_block_id=block_id,
                            valid_token_extent=roster.valid_token_extent,
                            group_token_capacity=group_token_capacity,
                            payload_kind=IntegrityPayloadKind.COMMIT,
                            byte_length=commit_bytes,
                        )
                        if fingerprint_mode:
                            commit_specs[source_plane].append(
                                _LocalizationLeafSpec(
                                    identity=commit_identity,
                                    local_block_id=None,
                                    destination_half=None,
                                    rank_slot=None,
                                )
                            )
                        else:
                            assert payload is not None
                            plane_start = source_plane * commit_bytes
                            leaves.append(
                                build_integrity_leaf(
                                    identity=commit_identity,
                                    payload=payload[
                                        plane_start : plane_start + commit_bytes
                                    ],
                                    local_block_id=None,
                                    destination_half=None,
                                    rank_slot=None,
                                )
                            )
                        hashed_bytes += commit_bytes

                if fingerprint_mode:
                    fingerprint_batches.append(
                        self._localization_fingerprint_rows(selected_rows)
                    )
                    fingerprint_specs.extend(wire_specs)
                    commit_rows = selected_rows.view(
                        len(chunk_positions), 2, row_bytes // 2
                    )
                    for source_plane in (0, 1):
                        fingerprint_batches.append(
                            self._localization_fingerprint_rows(
                                commit_rows[:, source_plane, :].contiguous()
                            )
                        )
                        fingerprint_specs.extend(commit_specs[source_plane])

        if fingerprint_mode:
            leaves.extend(
                self._localization_materialize_fingerprint_leaves(
                    fingerprint_batches,
                    fingerprint_specs,
                )
            )
            copied_bytes = len(leaves) * 32

        duration_ns = time.perf_counter_ns() - start_ns
        manifest = seal_source_manifest(
            NixlSourceManifest(
                schema_version=IntegrityIdentity.SCHEMA_VERSION,
                fingerprint_algorithm=self._localization_config.fingerprint_algorithm,
                run_id=self._localization_config.run_id,
                transport_arm=self._localization_config.transport_arm,
                producer_engine_id=self.engine_id,
                producer_request_id=req_id,
                registration_generation=self._registration_generation,
                offer_generation=roster.offer_generation,
                iteration=roster.iteration,
                source_rank=self.tp_rank,
                region_lengths=tuple(int(rows.shape[1]) for rows in self._region_rows),
                regions=self._region_descriptors,
                source_group_planes=source_group_planes,
                valid_token_extent=roster.valid_token_extent,
                group_token_capacities=roster.group_token_capacities,
                block_ids=roster.block_ids,
                observer=True,
                copied_bytes=copied_bytes,
                hashed_bytes=hashed_bytes,
                duration_ns=duration_ns,
                manifest_digest=b"",
                leaves=tuple(leaves),
            )
        )
        manifest_errors = validate_source_manifest_structure(manifest)
        if len(manifest_errors) > 0:
            raise LocalizationError(
                f"invalid source manifest for {(req_id, self.tp_rank)}: "
                f"{manifest_errors[:8]}"
            )
        self._localization_write_source_capture(manifest, stage)
        logger.warning(
            "[p2d-localize] stage=%s rid=%s rank=%d copied_mib=%.1f "
            "hashed_mib=%.1f duration_s=%.3f manifest=%s observer=true",
            stage.value,
            req_id,
            self.tp_rank,
            copied_bytes / (1024 * 1024),
            hashed_bytes / (1024 * 1024),
            duration_ns / 1_000_000_000,
            manifest.manifest_digest.hex(),
        )
        return manifest

    def _localization_write_source_capture(
        self,
        manifest: NixlSourceManifest,
        stage: IntegrityStage,
    ) -> None:
        """Write one complete source manifest as a first-class artifact.

        :param manifest: Source manifest to preserve.
        :param stage: Completed-read source checkpoint.
        """
        if self._localization_writer is None:
            raise LocalizationError("enabled localization has no artifact writer")
        self._localization_writer.write(
            NixlSourceManifestRecord(
                record_type=NixlSourceManifestRecord.RECORD_TYPE,
                stage=stage,
                manifest=manifest,
            )
        )

    def _localization_capture_source_post(self, req_id: ReqId) -> None:
        """Capture a completed-read source allocation before final release.

        :param req_id: Producer request whose pages are still pinned.
        """
        if self._localization_config.enabled_for(req_id) is False:
            return
        roster = self._localization_source_rosters.get(req_id)
        if roster is None:
            raise LocalizationError(
                f"completed target read has no retained source roster for {req_id}"
            )
        self._localization_capture_source_manifest(
            req_id,
            roster,
            IntegrityStage.SOURCE_POST,
        )
        del self._localization_source_rosters[req_id]

    def _localization_capture_staging(
        self,
        req_id: ReqId,
        ownership: CoalescedStagingPlan,
        stage: IntegrityStage,
        barrier: str,
    ) -> None:
        """Capture staged source rows at one configured transfer checkpoint.

        :param req_id: Decoder child request identifier.
        :param ownership: Sealed coalesced transfer and placement owner.
        :param stage: Staging checkpoint being observed.
        :param barrier: Exact observer ordering applied before the capture.
        """
        if self._localization_config.enabled_for(req_id) is False:
            return
        if self._staging_buf is None:
            raise LocalizationError("staging capture has no staging allocation")
        plan = ownership.scatter
        contracts = tuple(plan["source_contracts"])
        positions = plan["transfer_order"]
        n_pos = int(plan["n_pos"])
        n_ranks = int(plan["n_ranks"])
        if len(contracts) != n_ranks:
            raise LocalizationError(f"staging capture lacks contracts for {req_id}")
        fingerprint_mode = (
            self._localization_config.mode is LocalizationMode.FINGERPRINT
        )
        for rank_index, source_rank in enumerate(plan["source_ranks"]):
            start_ns = time.perf_counter_ns()
            contract = contracts[rank_index]
            if contract.source_rank != int(source_rank):
                raise LocalizationError("staging contract rank order differs")
            rank_slot = int(plan["slots"][rank_index])
            leaves: list[NixlIntegrityLeaf] = []
            fingerprint_batches: list[torch.Tensor] = []
            fingerprint_specs: list[_LocalizationLeafSpec] = []
            mapping: dict[IntegrityLeafKey, tuple[int, int, int]] = {}
            copied_bytes = 0
            hashed_bytes = 0
            for region_index, row_bytes_raw in enumerate(plan["blens"]):
                row_bytes = int(row_bytes_raw)
                region_start = ownership.lease.offset + int(
                    plan["region_off"][region_index]
                )
                region_size = n_ranks * n_pos * row_bytes
                region = self._staging_buf[
                    region_start : region_start + region_size
                ].view(n_ranks, n_pos, row_bytes)[rank_index]
                rows_per_chunk = max(
                    1,
                    self._localization_config.copy_chunk_bytes // row_bytes,
                )
                descriptor = contract.regions[region_index]
                owned_indices = [
                    index
                    for index, position in enumerate(positions)
                    if int(position.group_index) in descriptor.group_indices
                ]
                for chunk_start in range(0, len(owned_indices), rows_per_chunk):
                    selected = owned_indices[chunk_start : chunk_start + rows_per_chunk]
                    selected_tensor = torch.tensor(
                        selected,
                        device=region.device,
                        dtype=torch.long,
                    )
                    selected_rows = region.index_select(
                        0,
                        selected_tensor,
                    )
                    host_rows = (
                        None if fingerprint_mode else selected_rows.cpu().contiguous()
                    )
                    if fingerprint_mode is False:
                        copied_bytes += len(selected) * row_bytes
                    wire_specs: list[_LocalizationLeafSpec] = []
                    commit_specs: list[_LocalizationLeafSpec] = []
                    commit_row_indices: list[int] = []
                    for row_index, position_index in enumerate(selected):
                        position = positions[position_index]
                        payload = (
                            None
                            if host_rows is None
                            else memoryview(host_rows[row_index].numpy()).cast("B")
                        )
                        semantic_contract_digest = compute_semantic_contract_digest(
                            region=contract.regions[region_index],
                            group_index=int(position.group_index),
                            group_token_capacity=int(position.group_token_capacity),
                            source_plane_contract=(
                                contract.source_group_planes[int(position.group_index)]
                            ),
                        )
                        wire_identity = build_integrity_identity(
                            config=self._localization_config,
                            producer_engine_id=contract.producer_engine_id,
                            producer_request_id=contract.producer_request_id,
                            registration_generation=(contract.registration_generation),
                            semantic_contract_digest=semantic_contract_digest,
                            offer_generation=contract.offer_generation,
                            iteration=contract.iteration,
                            source_rank=int(source_rank),
                            region_index=region_index,
                            group_index=int(position.group_index),
                            plane_index=-1,
                            source_position=int(position.source_position),
                            remote_block_id=int(position.remote_block_id),
                            valid_token_extent=int(position.valid_token_extent),
                            group_token_capacity=int(position.group_token_capacity),
                            payload_kind=IntegrityPayloadKind.WIRE,
                            byte_length=row_bytes,
                        )
                        if fingerprint_mode:
                            wire_specs.append(
                                _LocalizationLeafSpec(
                                    identity=wire_identity,
                                    local_block_id=int(position.local_block_id),
                                    destination_half=int(position.plane_index),
                                    rank_slot=rank_slot,
                                )
                            )
                        else:
                            assert payload is not None
                            wire_leaf = build_integrity_leaf(
                                identity=wire_identity,
                                payload=payload,
                                local_block_id=int(position.local_block_id),
                                destination_half=int(position.plane_index),
                                rank_slot=rank_slot,
                            )
                            leaves.append(wire_leaf)
                            mapping[leaf_source_key(wire_leaf)] = (
                                int(position.local_block_id),
                                rank_slot,
                                int(position.plane_index),
                            )
                        hashed_bytes += row_bytes
                        if int(position.plane_index) < 0:
                            continue
                        commit_bytes = row_bytes // 2
                        commit_identity = build_integrity_identity(
                            config=self._localization_config,
                            producer_engine_id=contract.producer_engine_id,
                            producer_request_id=contract.producer_request_id,
                            registration_generation=(contract.registration_generation),
                            semantic_contract_digest=semantic_contract_digest,
                            offer_generation=contract.offer_generation,
                            iteration=contract.iteration,
                            source_rank=int(source_rank),
                            region_index=region_index,
                            group_index=int(position.group_index),
                            plane_index=0,
                            source_position=int(position.source_position),
                            remote_block_id=int(position.remote_block_id),
                            valid_token_extent=int(position.valid_token_extent),
                            group_token_capacity=int(position.group_token_capacity),
                            payload_kind=IntegrityPayloadKind.COMMIT,
                            byte_length=commit_bytes,
                        )
                        if fingerprint_mode:
                            commit_specs.append(
                                _LocalizationLeafSpec(
                                    identity=commit_identity,
                                    local_block_id=int(position.local_block_id),
                                    destination_half=int(position.plane_index),
                                    rank_slot=rank_slot,
                                )
                            )
                            commit_row_indices.append(row_index)
                        else:
                            assert payload is not None
                            commit_leaf = build_integrity_leaf(
                                identity=commit_identity,
                                payload=payload[:commit_bytes],
                                local_block_id=int(position.local_block_id),
                                destination_half=int(position.plane_index),
                                rank_slot=rank_slot,
                            )
                            leaves.append(commit_leaf)
                            mapping[leaf_source_key(commit_leaf)] = (
                                int(position.local_block_id),
                                rank_slot,
                                int(position.plane_index),
                            )
                        hashed_bytes += commit_bytes
                    if fingerprint_mode:
                        fingerprint_batches.append(
                            self._localization_fingerprint_rows(selected_rows)
                        )
                        fingerprint_specs.extend(wire_specs)
                        if len(commit_specs) > 0:
                            commit_indices = torch.tensor(
                                commit_row_indices,
                                device=selected_rows.device,
                                dtype=torch.long,
                            )
                            fingerprint_batches.append(
                                self._localization_fingerprint_rows(
                                    selected_rows.index_select(0, commit_indices)[
                                        :, : row_bytes // 2
                                    ].contiguous()
                                )
                            )
                            fingerprint_specs.extend(commit_specs)
            if fingerprint_mode:
                materialized = self._localization_materialize_fingerprint_leaves(
                    fingerprint_batches,
                    fingerprint_specs,
                )
                leaves.extend(materialized)
                copied_bytes = len(leaves) * 32
                mapping = self._localization_leaf_mapping(materialized)
            self._localization_finish_capture(
                req_id=req_id,
                contract=contract,
                stage=stage,
                barrier=barrier,
                leaves=tuple(leaves),
                mapping=mapping,
                copied_bytes=copied_bytes,
                hashed_bytes=hashed_bytes,
                duration_ns=time.perf_counter_ns() - start_ns,
            )

    def _localization_capture_destination(
        self,
        req_id: ReqId,
        plan: dict[str, Any],
        stage: IntegrityStage,
        barrier: str,
    ) -> None:
        """Capture reconstructed source shards in destination cache rows.

        :param req_id: Decoder child request identifier.
        :param plan: Sealed coalesced transfer and placement plan.
        :param stage: Post-scatter or pre-first-read checkpoint.
        :param barrier: Exact observer ordering applied before the capture.
        """
        if self._localization_config.enabled_for(req_id) is False:
            return
        if self._region_rows is None:
            raise LocalizationError("destination capture has no canonical rows")
        contracts = tuple(plan["source_contracts"])
        positions = plan["transfer_order"]
        n_ranks = int(plan["n_ranks"])
        if len(contracts) != n_ranks:
            raise LocalizationError(f"destination capture lacks contracts for {req_id}")
        fingerprint_mode = (
            self._localization_config.mode is LocalizationMode.FINGERPRINT
        )
        for rank_index, source_rank in enumerate(plan["source_ranks"]):
            start_ns = time.perf_counter_ns()
            contract = contracts[rank_index]
            if contract.source_rank != int(source_rank):
                raise LocalizationError("destination contract rank order differs")
            rank_slot = int(plan["slots"][rank_index])
            leaves: list[NixlIntegrityLeaf] = []
            fingerprint_batches: list[torch.Tensor] = []
            fingerprint_specs: list[_LocalizationLeafSpec] = []
            mapping: dict[IntegrityLeafKey, tuple[int, int, int]] = {}
            copied_bytes = 0
            hashed_bytes = 0
            for region_index, flat in enumerate(self._region_rows):
                row_bytes = int(plan["blens"][region_index])
                chunk_bytes = row_bytes // 2
                descriptor = self._region_descriptors[region_index]
                dual_indices = [
                    index
                    for index, position in enumerate(positions)
                    if int(position.group_index) in descriptor.group_indices
                    and int(position.plane_index) < 0
                ]
                single_indices = [
                    index
                    for index, position in enumerate(positions)
                    if int(position.group_index) in descriptor.group_indices
                    and int(position.plane_index) >= 0
                ]
                destination = flat.view(flat.shape[0], 2, n_ranks, chunk_bytes)
                rows_per_chunk = max(
                    1,
                    self._localization_config.copy_chunk_bytes // row_bytes,
                )
                for chunk_start in range(0, len(dual_indices), rows_per_chunk):
                    selected = dual_indices[chunk_start : chunk_start + rows_per_chunk]
                    local_indices = torch.tensor(
                        [int(positions[index].local_block_id) for index in selected],
                        device=flat.device,
                        dtype=torch.long,
                    )
                    selected_rows = (
                        destination[local_indices, :, rank_slot, :]
                        .contiguous()
                        .view(len(selected), row_bytes)
                    )
                    host_rows = None if fingerprint_mode else selected_rows.cpu()
                    if fingerprint_mode is False:
                        copied_bytes += len(selected) * row_bytes
                    batch_specs: list[_LocalizationLeafSpec] = []
                    for row_index, position_index in enumerate(selected):
                        position = positions[position_index]
                        payload = (
                            None
                            if host_rows is None
                            else memoryview(host_rows[row_index].numpy()).cast("B")
                        )
                        semantic_contract_digest = compute_semantic_contract_digest(
                            region=contract.regions[region_index],
                            group_index=int(position.group_index),
                            group_token_capacity=int(position.group_token_capacity),
                            source_plane_contract=(
                                contract.source_group_planes[int(position.group_index)]
                            ),
                        )
                        identity = build_integrity_identity(
                            config=self._localization_config,
                            producer_engine_id=contract.producer_engine_id,
                            producer_request_id=contract.producer_request_id,
                            registration_generation=(contract.registration_generation),
                            semantic_contract_digest=semantic_contract_digest,
                            offer_generation=contract.offer_generation,
                            iteration=contract.iteration,
                            source_rank=int(source_rank),
                            region_index=region_index,
                            group_index=int(position.group_index),
                            plane_index=-1,
                            source_position=int(position.source_position),
                            remote_block_id=int(position.remote_block_id),
                            valid_token_extent=int(position.valid_token_extent),
                            group_token_capacity=int(position.group_token_capacity),
                            payload_kind=IntegrityPayloadKind.WIRE,
                            byte_length=row_bytes,
                        )
                        if fingerprint_mode:
                            batch_specs.append(
                                _LocalizationLeafSpec(
                                    identity=identity,
                                    local_block_id=int(position.local_block_id),
                                    destination_half=-1,
                                    rank_slot=rank_slot,
                                )
                            )
                        else:
                            assert payload is not None
                            leaf = build_integrity_leaf(
                                identity=identity,
                                payload=payload,
                                local_block_id=int(position.local_block_id),
                                destination_half=-1,
                                rank_slot=rank_slot,
                            )
                            leaves.append(leaf)
                            mapping[leaf_source_key(leaf)] = (
                                int(position.local_block_id),
                                rank_slot,
                                -1,
                            )
                        hashed_bytes += row_bytes
                    if fingerprint_mode:
                        fingerprint_batches.append(
                            self._localization_fingerprint_rows(selected_rows)
                        )
                        fingerprint_specs.extend(batch_specs)

                single_rows_per_chunk = max(
                    1,
                    self._localization_config.copy_chunk_bytes // chunk_bytes,
                )
                destination_single = flat.view(flat.shape[0], n_ranks, 2, chunk_bytes)
                for chunk_start in range(
                    0,
                    len(single_indices),
                    single_rows_per_chunk,
                ):
                    selected = single_indices[
                        chunk_start : chunk_start + single_rows_per_chunk
                    ]
                    local_indices = torch.tensor(
                        [int(positions[index].local_block_id) for index in selected],
                        device=flat.device,
                        dtype=torch.long,
                    )
                    destination_halves = torch.tensor(
                        [int(positions[index].plane_index) for index in selected],
                        device=flat.device,
                        dtype=torch.long,
                    )
                    selected_rows = destination_single[
                        local_indices,
                        rank_slot,
                        destination_halves,
                        :,
                    ].contiguous()
                    host_rows = None if fingerprint_mode else selected_rows.cpu()
                    if fingerprint_mode is False:
                        copied_bytes += len(selected) * chunk_bytes
                    batch_specs = []
                    for row_index, position_index in enumerate(selected):
                        position = positions[position_index]
                        payload = (
                            None
                            if host_rows is None
                            else memoryview(host_rows[row_index].numpy()).cast("B")
                        )
                        semantic_contract_digest = compute_semantic_contract_digest(
                            region=contract.regions[region_index],
                            group_index=int(position.group_index),
                            group_token_capacity=int(position.group_token_capacity),
                            source_plane_contract=(
                                contract.source_group_planes[int(position.group_index)]
                            ),
                        )
                        identity = build_integrity_identity(
                            config=self._localization_config,
                            producer_engine_id=contract.producer_engine_id,
                            producer_request_id=contract.producer_request_id,
                            registration_generation=(contract.registration_generation),
                            semantic_contract_digest=semantic_contract_digest,
                            offer_generation=contract.offer_generation,
                            iteration=contract.iteration,
                            source_rank=int(source_rank),
                            region_index=region_index,
                            group_index=int(position.group_index),
                            plane_index=0,
                            source_position=int(position.source_position),
                            remote_block_id=int(position.remote_block_id),
                            valid_token_extent=int(position.valid_token_extent),
                            group_token_capacity=int(position.group_token_capacity),
                            payload_kind=IntegrityPayloadKind.COMMIT,
                            byte_length=chunk_bytes,
                        )
                        if fingerprint_mode:
                            batch_specs.append(
                                _LocalizationLeafSpec(
                                    identity=identity,
                                    local_block_id=int(position.local_block_id),
                                    destination_half=int(position.plane_index),
                                    rank_slot=rank_slot,
                                )
                            )
                        else:
                            assert payload is not None
                            leaf = build_integrity_leaf(
                                identity=identity,
                                payload=payload,
                                local_block_id=int(position.local_block_id),
                                destination_half=int(position.plane_index),
                                rank_slot=rank_slot,
                            )
                            leaves.append(leaf)
                            mapping[leaf_source_key(leaf)] = (
                                int(position.local_block_id),
                                rank_slot,
                                int(position.plane_index),
                            )
                        hashed_bytes += chunk_bytes
                    if fingerprint_mode:
                        fingerprint_batches.append(
                            self._localization_fingerprint_rows(selected_rows)
                        )
                        fingerprint_specs.extend(batch_specs)
            if fingerprint_mode:
                materialized = self._localization_materialize_fingerprint_leaves(
                    fingerprint_batches,
                    fingerprint_specs,
                )
                leaves.extend(materialized)
                copied_bytes = len(leaves) * 32
                mapping = self._localization_leaf_mapping(materialized)
            self._localization_finish_capture(
                req_id=req_id,
                contract=contract,
                stage=stage,
                barrier=barrier,
                leaves=tuple(leaves),
                mapping=mapping,
                copied_bytes=copied_bytes,
                hashed_bytes=hashed_bytes,
                duration_ns=time.perf_counter_ns() - start_ns,
            )

    def _localization_finish_capture(
        self,
        *,
        req_id: ReqId,
        contract: NixlSourceContract,
        stage: IntegrityStage,
        barrier: str,
        leaves: tuple[NixlIntegrityLeaf, ...],
        mapping: dict[IntegrityLeafKey, tuple[int, int, int]],
        copied_bytes: int,
        hashed_bytes: int,
        duration_ns: int,
    ) -> None:
        """Persist one complete rank and stage observation for offline proof."""
        if self._localization_writer is None:
            raise LocalizationError("enabled localization has no artifact writer")
        if len(leaves) != len(mapping) or set(map(leaf_source_key, leaves)) != set(
            mapping
        ):
            raise LocalizationError(
                f"{stage.value} capture mapping differs from its leaf set"
            )
        self._localization_writer.write(
            NixlCaptureRecord(
                record_type=NixlCaptureRecord.RECORD_TYPE,
                schema_version=contract.schema_version,
                fingerprint_algorithm=contract.fingerprint_algorithm,
                stage=stage,
                run_id=contract.run_id,
                transport_arm=contract.transport_arm,
                producer_engine_id=contract.producer_engine_id,
                producer_request_id=contract.producer_request_id,
                registration_generation=contract.registration_generation,
                offer_generation=contract.offer_generation,
                iteration=contract.iteration,
                child_request_id=req_id,
                source_rank=contract.source_rank,
                observer_engine_id=self.engine_id,
                observer_rank=self.tp_rank,
                observer=True,
                copied_bytes=copied_bytes,
                hashed_bytes=hashed_bytes,
                duration_ns=duration_ns,
                barrier=barrier,
                leaves=leaves,
            )
        )
        logger.warning(
            "[p2d-localize] stage=%s child=%s producer=%s source_rank=%d "
            "copied_mib=%.1f hashed_mib=%.1f duration_s=%.3f observer=true",
            stage.value,
            req_id,
            contract.producer_request_id,
            contract.source_rank,
            copied_bytes / (1024 * 1024),
            hashed_bytes / (1024 * 1024),
            duration_ns / 1_000_000_000,
        )

    def _localization_capture_pre_read(self, req_ids: set[ReqId]) -> None:
        """Capture destination rows after transfer drain and before model forward.

        :param req_ids: Requests admitted to the current model batch.
        """
        if self._localization_config.enabled is False:
            return
        if len(self._localization_pre_read_plans) == 0:
            return
        if len(self._localization_pre_read_plans) != 1:
            raise LocalizationError(
                "multiple localization targets are pending first-read capture"
            )
        req_id, plan = next(iter(self._localization_pre_read_plans.items()))
        if self._localization_config.enabled_for(req_id) is False:
            raise LocalizationError("pre-read plan is outside the localization target")
        if req_id not in req_ids:
            return
        self._localization_pre_read_plans.pop(req_id)
        self._localization_capture_destination(
            req_id,
            plan,
            IntegrityStage.PRE_READ,
            "after_transfer_phase_drain_before_model_forward",
        )
        contracts = tuple(plan["source_contracts"])
        if len(contracts) == 0:
            raise LocalizationError("pre-read plan has no source contracts")
        producer = contracts[0]
        is_sham = self._localization_config.mode is LocalizationMode.SHAM
        self._localization_record_event(
            code=("SHAM_CAPTURE_COMPLETE" if is_sham else "CAPTURE_COMPLETE"),
            evidentiary=False,
            child_request_id=req_id,
            producer_engine_id=producer.producer_engine_id,
            producer_request_id=producer.producer_request_id,
            detail="all decoder stages captured; offline source comparison pending",
        )

    def _localization_record_event(
        self,
        *,
        code: str,
        evidentiary: bool,
        child_request_id: ReqId | None,
        producer_engine_id: str | None,
        producer_request_id: str | None,
        detail: str,
    ) -> None:
        """Write one explicit child terminal or exclusion outcome.

        :param code: Stable outcome code.
        :param evidentiary: Whether the child supports byte-integrity claims.
        :param child_request_id: Decoder child, if the event is child-scoped.
        :param producer_engine_id: Source engine lineage.
        :param producer_request_id: Source request lineage.
        :param detail: Human-readable outcome detail.
        """
        if (
            child_request_id is None
            or self._localization_config.enabled_for(child_request_id) is False
        ):
            return
        if child_request_id in self._localization_terminal_recorded:
            return
        if self._localization_writer is None:
            raise LocalizationError("enabled localization has no artifact writer")
        self._localization_writer.write(
            NixlEventRecord(
                record_type=NixlEventRecord.RECORD_TYPE,
                schema_version=IntegrityIdentity.SCHEMA_VERSION,
                run_id=self._localization_config.run_id,
                transport_arm=self._localization_config.transport_arm,
                code=code,
                evidentiary=evidentiary,
                producer_engine_id=producer_engine_id,
                producer_request_id=producer_request_id,
                child_request_id=child_request_id,
                observer_engine_id=self.engine_id,
                observer_rank=self.tp_rank,
                detail=detail,
                created_ns=time.time_ns(),
            )
        )
        self._localization_terminal_recorded.add(child_request_id)

    # ------------------------------------------------------------------
    # Coalesced pull: staging buffer + completion scatter
    # ------------------------------------------------------------------

    def _staging_init(self) -> bool:
        """Lazily allocate and NIXL-register the staging buffer.

        :returns: Whether staging is initialized and available.
        """
        if self._staging_buf is not None:
            return True
        size = self.coalesce_staging_mb * 1024 * 1024
        try:
            dev = next(iter(self.device_kv_caches.values())).device
            # Successful plans overwrite the complete lease before scatter.
            # The allocation therefore has no initialization writer to order.
            self._staging_buf = torch.empty(size, dtype=torch.uint8, device=dev)
            self.nixl_wrapper.register_memory(
                [(self._staging_buf.data_ptr(), size, self.device_id, "")],
                self.nixl_memory_type,
            )
        except Exception:
            logger.error(
                "coalesced pull: staging init failed; disabling\n%s",
                traceback.format_exc(),
            )
            self.coalesce_pull = False
            self._staging_buf = None
            return False
        self._staging_allocator = StagingRangeAllocator(size)
        logger.info("coalesced pull: %sMB staging registered", self.coalesce_staging_mb)
        return True

    def _create_coalesced_plan(
        self,
        req_id: ReqId,
        size: int,
        source_ranks: tuple[int, ...],
        remote_engine_id: EngineId,
        scatter: dict[str, Any],
    ) -> CoalescedStagingPlan | None:
        """Atomically allocate staging and install its native owner.

        :param req_id: Decoder request identifier.
        :param size: Required staging bytes.
        :param source_ranks: Complete set of independently posted ranks.
        :param remote_engine_id: Remote engine owning the source registration.
        :param scatter: Transfer and placement geometry retained by the plan.
        :returns: Installed plan, or ``None`` when the pool is busy.
        """
        if self._staging_allocator is None:
            raise StagingSafetyError("coalesced staging allocator is not initialized")
        if req_id in self._coalesce_plans:
            raise StagingSafetyError(f"request {req_id} already owns staging")
        if len(remote_engine_id) == 0:
            raise StagingSafetyError("coalesced staging requires a remote engine")
        if self._phase_separate_transfer_decode and not self._transfer_phase_active:
            self._transfer_phase_violation_count += 1
            raise StagingSafetyError(
                "coalesced staging allocation is forbidden outside the transfer phase"
            )
        owner_id = f"{self.engine_id}:{self.tp_rank}:{self._coalesce_owner_sequence}"
        self._coalesce_owner_sequence += 1
        plan = self._staging_allocator.create_plan(
            owner_id=owner_id,
            request_id=req_id,
            size=size,
            source_ranks=source_ranks,
            remote_engine_id=remote_engine_id,
            scatter=scatter,
            warn_after_s=self._coalesce_warn_after_s,
            fail_after_s=self._coalesce_fail_after_s,
        )
        if plan is not None:
            self._coalesce_plans[req_id] = plan
            if self._phase_separate_transfer_decode:
                self._transfer_phase_plan_count += 1
                self._transfer_phase_byte_count += plan.lease.size
        return plan

    def _release_coalesced_plan(self, plan: CoalescedStagingPlan) -> None:
        """Release one plan only after its complete quiescence proof.

        :param plan: Exact active plan to reclaim.
        """
        if self._staging_allocator is None:
            raise StagingSafetyError("coalesced staging allocator disappeared")
        if self._coalesce_plans.get(plan.request_id) is not plan:
            raise StagingSafetyError("coalesced staging request owner mismatch")
        self._staging_allocator.release(plan)
        del self._coalesce_plans[plan.request_id]

    def _discard_completed_coalesced_plan(self, request_id: ReqId) -> None:
        """Reclaim a successful native plan whose request failed before scatter.

        :param request_id: Decoder request whose bytes must not be published.
        :raises StagingSafetyError: If any native writer is not proven quiescent.
        """
        plan = self._coalesce_plans.get(request_id)
        if plan is None:
            return
        if not plan.ready_to_scatter:
            raise StagingSafetyError(
                "A failed receive cannot discard an unquiesced staging plan: "
                f"{plan.describe()}"
            )
        plan.fail("request failed after native completion and before scatter")
        self._release_coalesced_plan(plan)

    def _release_quiescent_failed_coalesced_plan(
        self,
        plan: CoalescedStagingPlan,
    ) -> None:
        """Release a failed plan only after proving every writer quiescent.

        Handles that were prepared but never posted still own native resources,
        as do completed handles that have not yet passed through the polling
        cleanup. Resource cleanup must therefore precede allocator reuse.

        :param plan: Failed, reusable staging owner.
        :raises StagingSafetyError: If cleanup cannot preserve the reuse proof.
        """
        if plan.operation_failed is False or plan.reusable is False:
            raise StagingSafetyError(
                f"A coalesced failure is not safely reclaimable: {plan.describe()}"
            )
        try:
            for source_rank, slot in plan.slots.items():
                if slot.native_handle is None or slot.native_released:
                    continue
                if slot.state not in {
                    HandleState.DONE,
                    HandleState.SEALED_UNPOSTED,
                }:
                    raise StagingSafetyError(
                        "A reusable coalesced plan retained an unexpected handle: "
                        f"{plan.describe()}"
                    )
                self.nixl_wrapper.release_xfer_handle(slot.native_handle)
                if slot.state is HandleState.DONE:
                    plan.mark_native_released(source_rank)
                # ``native_released`` is evidence for a posted handle's DONE
                # cleanup. SEALED_UNPOSTED already proves that this handle
                # never became a writer, and this plan is retired immediately.
        except Exception as error:
            stacktrace = traceback.format_exc()
            plan.tombstone("failed coalesced native resource cleanup\n" + stacktrace)
            self._fail_coalesced_plan(
                plan,
                "failed coalesced native resource cleanup\n" + stacktrace,
                error,
            )
        self._release_coalesced_plan(plan)

    def _finish_quiescent_coalesced_failure(
        self,
        plan: CoalescedStagingPlan,
        reason: str,
        error: BaseException | None = None,
    ) -> None:
        """Turn a provably pre-write failure into a request terminal.

        :param plan: Plan whose logical operation failed.
        :param reason: Stable failure detail.
        :param error: Native exception, if one was raised.
        :raises StagingSafetyError: If any possible writer remains uncertain.
        """
        if plan.operation_failed is False:
            plan.fail(reason)
        if plan.posting_sealed is False:
            plan.seal_posting()
        if plan.reusable is False:
            self._fail_coalesced_plan(plan, reason, error)
        self._release_quiescent_failed_coalesced_plan(plan)
        logger.error(
            "coalesced staging failed before any live writer remained: "
            "request=%s reason=%s",
            plan.request_id,
            reason,
            exc_info=error is not None,
        )
        self._handle_failed_transfer(plan.request_id, None)

    def _resolve_failed_coalesced_plan(self, request_id: ReqId) -> bool:
        """Resolve staging ownership before publishing a failed terminal.

        :param request_id: Decoder request with a pending failed receive.
        :returns: Whether no coalesced writer can still touch its destination.
        :raises StagingSafetyError: If a failed plan lost its reuse proof.
        """
        plan = self._coalesce_plans.get(request_id)
        if plan is None:
            return True
        if plan.ready_to_scatter:
            self._discard_completed_coalesced_plan(request_id)
            return True
        if plan.operation_failed is False:
            return False
        if plan.reusable is False:
            self._fail_coalesced_plan(
                plan,
                plan.failure_reason or "coalesced operation failed",
            )
        self._release_quiescent_failed_coalesced_plan(plan)
        return True

    def _initialize_and_post_coalesced(
        self,
        plan: CoalescedStagingPlan,
        source_rank: int,
        local_descs: Any,
        remote_descs: Any,
        remote_agent: str,
        notification_id: bytes,
    ) -> bool:
        """Prepare, attach, and post one rank without losing native ownership.

        :param plan: Owner installed before native preparation.
        :param source_rank: Producer rank being posted.
        :param local_descs: NIXL local descriptor list.
        :param remote_descs: NIXL remote descriptor list.
        :param remote_agent: NIXL remote agent identity.
        :param notification_id: Completion notification payload.
        :returns: Whether the source rank was posted successfully.
        """
        try:
            handle = self.nixl_wrapper.initialize_xfer(
                "READ",
                local_descs,
                remote_descs,
                remote_agent,
                notification_id,
            )
            plan.attach_handle(source_rank, handle)
        except Exception as error:
            stacktrace = traceback.format_exc()
            if plan.slots[source_rank].state is HandleState.PREPARING:
                plan.record_prepare_failure(
                    source_rank,
                    f"rank {source_rank} native preparation raised\n{stacktrace}",
                )
                self._finish_quiescent_coalesced_failure(
                    plan,
                    f"rank {source_rank} native preparation raised\n{stacktrace}",
                    error,
                )
                return False
            else:
                plan.tombstone(
                    f"rank {source_rank} preparation transition failed\n{stacktrace}"
                )
            self._fail_coalesced_plan(
                plan,
                f"rank {source_rank} native preparation raised\n{stacktrace}",
                error,
            )

        try:
            plan.begin_post(source_rank)
            self._assert_transfer_post_allowed(coalesced=True)
            status = self.nixl_wrapper.transfer(handle)
            plan.record_post_result(source_rank, status)
        except Exception as error:
            stacktrace = traceback.format_exc()
            slot = plan.slots[source_rank]
            if slot.state is HandleState.POSTING:
                plan.record_post_exception(
                    source_rank,
                    f"rank {source_rank} native post raised\n{stacktrace}",
                )
            elif slot.state is HandleState.PREPARED:
                self._finish_quiescent_coalesced_failure(
                    plan,
                    f"rank {source_rank} failed before native post\n{stacktrace}",
                    error,
                )
                return False
            else:
                plan.tombstone(
                    f"rank {source_rank} post transition failed in "
                    f"state {slot.state}\n{stacktrace}"
                )
            self._fail_coalesced_plan(
                plan,
                f"rank {source_rank} native post raised\n{stacktrace}",
                error,
            )

        if plan.operation_failed:
            self._fail_coalesced_plan(
                plan,
                plan.failure_reason or "native post failed",
            )
        return True

    def _fail_coalesced_plan(
        self,
        plan: CoalescedStagingPlan,
        reason: str,
        error: BaseException | None = None,
    ) -> Never:
        """Tombstone a plan and fail the decoder before publication.

        :param plan: Plan whose safety proof failed.
        :param reason: Stable failure detail.
        :param error: Native exception, if one was raised.
        :raises StagingSafetyError: Always, after retaining the plan and handles.
        """
        if plan.operation_failed is False:
            plan.fail(reason)
        if plan.posting_sealed is False:
            plan.seal_posting()
        logger.error(
            "coalesced staging fail-stop: %s reason=%s",
            plan.describe(),
            reason,
            exc_info=error is not None,
        )
        failure = StagingSafetyError(
            f"coalesced staging safety proof failed: {plan.describe()} reason={reason}"
        )
        if error is None:
            raise failure
        raise failure from error

    def _poll_coalesced_plans(self) -> set[ReqId]:
        """Poll coalesced owners without releasing any possible writer.

        :returns: Requests whose every native handle authoritatively reached DONE.
        """
        done_req_ids: set[ReqId] = set()
        now = time.monotonic()
        for plan in tuple(self._coalesce_plans.values()):
            if plan.warning_due(now):
                logger.error("coalesced staging still active: %s", plan.describe(now))
                plan.mark_warning_emitted()
            if plan.operation_failed:
                self._fail_coalesced_plan(
                    plan,
                    plan.failure_reason or "coalesced operation failed",
                )
            for source_rank, slot in plan.slots.items():
                if slot.state is HandleState.PROC:
                    if slot.native_handle is None:
                        plan.tombstone(f"rank {source_rank} lost its native handle")
                        self._fail_coalesced_plan(
                            plan,
                            f"rank {source_rank} lost its native handle",
                        )
                    try:
                        status = self.nixl_wrapper.check_xfer_state(slot.native_handle)
                    except Exception as error:
                        stacktrace = traceback.format_exc()
                        plan.record_query_exception(
                            source_rank,
                            f"rank {source_rank} status query raised\n{stacktrace}",
                        )
                        self._fail_coalesced_plan(
                            plan,
                            f"rank {source_rank} status query raised\n{stacktrace}",
                            error,
                        )
                    plan.record_query_result(source_rank, status)
                if slot.state in {HandleState.ERR, HandleState.UNKNOWN}:
                    self._fail_coalesced_plan(
                        plan,
                        f"rank {source_rank} status became {slot.state.value}",
                    )
                if slot.state is not HandleState.DONE or slot.native_released:
                    continue
                if slot.native_handle is None:
                    self._fail_coalesced_plan(
                        plan,
                        f"rank {source_rank} DONE state lost its native handle",
                    )
                try:
                    telemetry = self.nixl_wrapper.get_xfer_telemetry(slot.native_handle)
                    self.xfer_stats.record_transfer(telemetry)
                    self.nixl_wrapper.release_xfer_handle(slot.native_handle)
                except Exception as error:
                    stacktrace = traceback.format_exc()
                    self._fail_coalesced_plan(
                        plan,
                        f"rank {source_rank} DONE handle cleanup raised\n{stacktrace}",
                        error,
                    )
                else:
                    plan.mark_native_released(source_rank)

            if plan.fail_deadline_expired(now):
                plan.tombstone("native transfer exceeded the fail-stop deadline")
                self._fail_coalesced_plan(
                    plan,
                    "native transfer exceeded the fail-stop deadline",
                )
            if plan.ready_to_scatter:
                done_req_ids.add(plan.request_id)
        return done_req_ids

    def _sp_group_flags(self) -> list[bool]:
        """Per-KV-cache-group single-plane flags (F2b: kv_planes==1
        groups need K-half 2:1 pulls; the stock path is invalid)."""
        if self._sp_flags_cache is None:
            self._sp_flags_cache = [
                getattr(g.kv_cache_spec, "kv_planes", 2) == 1
                for g in self.kv_cache_config.kv_cache_groups
            ]
        return self._sp_flags_cache

    def _physical_group_token_capacities(self) -> tuple[int, ...]:
        """Return each cache group's token capacity per registered row.

        :returns: Physical token capacities in cache-group order.
        """
        capacities: list[int] = []
        for group in self.kv_cache_config.kv_cache_groups:
            spec = group.kv_cache_spec
            factor = (
                1
                if isinstance(spec, MambaSpec)
                else self._physical_blocks_per_logical_kv_block
            )
            if spec.block_size <= 0 or spec.block_size % factor != 0:
                raise LocalizationError(
                    "cache-group token capacity cannot be represented by "
                    "physical kernel blocks"
                )
            capacities.append(int(spec.block_size // factor))
        return tuple(capacities)

    def _coalesce_region_rows(self) -> bool:
        """Canonicalize every region tensor to a (num_blocks, row_bytes)
        uint8 view of its physical storage. Registered cache tensors are
        permuted views of the HND-contiguous base (pages outermost);
        stride-sorting the dims recovers it. Any region that does not
        canonicalize this way disables the coalesced path -- checked
        here, before any transfer is ever posted."""
        if self._region_rows is not None:
            return True
        rows: list[torch.Tensor] = []
        for i, cache in enumerate(self._region_tensors):
            order = sorted(range(cache.dim()), key=lambda d: -cache.stride(d))
            phys = cache.permute(order)
            if not phys.is_contiguous() or phys.shape[0] != cache.shape[0]:
                logger.warning(
                    "coalesced pull: region %s not canonicalizable "
                    "(shape %s strides %s); disabling",
                    i,
                    tuple(cache.shape),
                    tuple(cache.stride()),
                )
                self.coalesce_pull = False
                return False
            flat = phys.view(torch.uint8).view(phys.shape[0], -1)
            if flat.shape[1] != self.block_len_per_layer[i]:
                logger.warning(
                    "coalesced pull: region %s row bytes %s != block_len %s; disabling",
                    i,
                    flat.shape[1],
                    self.block_len_per_layer[i],
                )
                self.coalesce_pull = False
                return False
            rows.append(flat)
        self._region_rows = rows
        return True

    def _coalesced_scatter(self, req_id: ReqId) -> None:
        """Scatter one completed staging generation into the destination cache.

        The device-wide completion fence proves that asynchronous scatter reads
        no longer touch staging before its lease is returned to the allocator.

        :param req_id: Decoder request whose completed generation should scatter.
        """
        plan = self._coalesce_plans.get(req_id)
        if plan is None:
            return
        geometry = plan.scatter
        plan.begin_device_read()
        scatter_error: BaseException | None = None
        scatter_traceback: str | None = None
        observation_error: BaseException | None = None
        observation_traceback: str | None = None
        try:
            assert self._staging_buf is not None
            assert self._region_rows is not None
            if (
                self._localization_config.enabled_for(req_id)
                and self._localization_config.mode is not LocalizationMode.FINGERPRINT
            ):
                self._localization_capture_staging(
                    req_id,
                    plan,
                    IntegrityStage.STAGING_RAW,
                    "nixl_done_without_added_device_wide_sync",
                )
                if self._staging_buf.device.type != "cpu":
                    torch.accelerator.synchronize()
                self._localization_capture_staging(
                    req_id,
                    plan,
                    IntegrityStage.STAGING_FENCED_CONTROL,
                    "device_synchronize_observer_control_not_gdr_flush",
                )
            idx = torch.tensor(
                geometry["lpos"], device=self._staging_buf.device, dtype=torch.long
            )
            n_pos, n_ranks = geometry["n_pos"], geometry["n_ranks"]
            halves = geometry.get("sp_half")
            hv = None
            if halves is not None and any(h >= 0 for h in halves):
                hv = torch.tensor(
                    halves, device=self._staging_buf.device, dtype=torch.long
                )
                dual_m = hv < 0
                sp_m = ~dual_m
                idx_d = idx[dual_m]
                idx_s = idx[sp_m]
                h_s = hv[sp_m]
            for i, flat in enumerate(self._region_rows):
                blen = geometry["blens"][i]
                chunk = blen // 2
                base = plan.lease.offset + geometry["region_off"][i]
                reg = self._staging_buf[base : base + n_ranks * n_pos * blen]
                reg = reg.view(n_ranks, n_pos, 2, chunk)
                dest = flat.view(flat.shape[0], 2, n_ranks, chunk)
                if hv is None:
                    for r in range(n_ranks):
                        dest[:, :, geometry["slots"][r], :][idx] = reg[r]
                    continue
                # F2b: single-plane positions carry (local row, half);
                # the row layout is (rank-shard, halves of 64 tok, K
                # chunk) -- write only the staged K half. Dual positions
                # keep the standard (K/V, rank-shard, chunk) write.
                dest_sp = flat.view(flat.shape[0], n_ranks, 2, chunk)
                for r in range(n_ranks):
                    sl = geometry["slots"][r]
                    if idx_d.numel():
                        dest[:, :, sl, :][idx_d] = reg[r][dual_m]
                    if idx_s.numel():
                        dest_sp[idx_s, sl, h_s] = reg[r][sp_m][:, 0]
            if self._audit_enabled and geometry.get("audit_rows"):
                self._audit_pending.append(
                    (req_id, geometry["audit_rows"], geometry["lpos"])
                )
        except Exception as error:
            scatter_error = error
            scatter_traceback = traceback.format_exc()

        # NIXL writers are quiescent at this point, but the CUDA scatter reads
        # staging asynchronously. A failed synchronization is not a release
        # fence and must leave the allocation permanently owned.
        try:
            torch.cuda.synchronize()
        except Exception as error:
            stacktrace = traceback.format_exc()
            reason = (
                "device synchronization failed after staging readers began\n"
                + stacktrace
            )
            plan.tombstone(reason)
            self._fail_coalesced_plan(
                plan,
                reason,
                error,
            )
        if (
            scatter_error is None
            and self._localization_config.enabled_for(req_id)
            and self._localization_config.mode is LocalizationMode.FINGERPRINT
        ):
            try:
                self._localization_capture_staging(
                    req_id,
                    plan,
                    IntegrityStage.STAGING_POST_SCATTER,
                    "post_scatter_device_synchronize_before_staging_release",
                )
                self._localization_capture_destination(
                    req_id,
                    geometry,
                    IntegrityStage.DESTINATION,
                    "post_scatter_device_synchronize_before_publication",
                )
                self._localization_pre_read_plans[req_id] = geometry
            except Exception as error:
                observation_error = error
                observation_traceback = traceback.format_exc()

            try:
                torch.cuda.synchronize()
            except Exception as error:
                stacktrace = traceback.format_exc()
                reason = (
                    "device synchronization failed after localization readers "
                    "began\n" + stacktrace
                )
                plan.tombstone(reason)
                self._fail_coalesced_plan(
                    plan,
                    reason,
                    error,
                )

        plan.mark_device_quiescent()

        if scatter_error is not None:
            plan.fail("staging scatter failed before publication")
            self._release_coalesced_plan(plan)
            logger.error(
                "coalesced staging scatter failed after safe device quiescence: %s\n%s",
                plan.describe(),
                scatter_traceback,
            )
            raise StagingSafetyError(
                "coalesced staging scatter failed before scheduler publication"
            ) from scatter_error

        if observation_error is not None:
            assert observation_traceback is not None
            plan.fail("post-scatter observation failed before publication")
            self._release_coalesced_plan(plan)
            raise StagingSafetyError(
                "post-scatter observation failed before publication\n"
                + observation_traceback
            ) from observation_error

        try:
            if (
                self._localization_config.enabled_for(req_id)
                and self._localization_config.mode is not LocalizationMode.FINGERPRINT
            ):
                self._localization_capture_destination(
                    req_id,
                    geometry,
                    IntegrityStage.DESTINATION,
                    "post_scatter_device_synchronize_before_publication",
                )
                self._localization_pre_read_plans[req_id] = geometry
        except Exception as error:
            stacktrace = traceback.format_exc()
            reason = "destination observation failed before publication\n" + stacktrace
            plan.fail(reason)
            self._release_coalesced_plan(plan)
            raise StagingSafetyError(
                "coalesced destination observation failed before publication\n"
                + stacktrace
            ) from error

        self._release_coalesced_plan(plan)

    def _begin_transfer_phase(self) -> None:
        """Close prior device work and authorize one transfer-only phase.

        :raises StagingSafetyError: If the preceding phase was not fully consumed
            or device quiescence cannot be established.
        """
        if not self._phase_separate_transfer_decode:
            return

        stock_handle_count = sum(
            len(handles) for handles in self._recving_transfers.values()
        )
        if (
            self._transfer_phase_active
            or len(self._deferred_phase_sending) > 0
            or len(self._deferred_phase_recving) > 0
            or len(self._coalesce_plans) > 0
            or len(self._coalesce_pending) > 0
            or stock_handle_count > 0
        ):
            self._transfer_phase_violation_count += 1
            raise StagingSafetyError(
                "transfer phase began before the preceding phase was consumed"
            )

        sync_started_ns = time.monotonic_ns()
        try:
            torch.accelerator.synchronize()
        except Exception as error:
            self._transfer_phase_violation_count += 1
            raise StagingSafetyError(
                "device synchronization failed before the transfer phase\n"
                + traceback.format_exc()
            ) from error

        self._transfer_phase_entry_sync_ns = time.monotonic_ns() - sync_started_ns
        self._transfer_phase_epoch += 1
        self._transfer_phase_plan_count = 0
        self._transfer_phase_handle_count = 0
        self._transfer_phase_byte_count = 0
        self._transfer_phase_violation_count = 0
        self._transfer_phase_started_ns = time.monotonic_ns()
        self._transfer_phase_active = True

    def _assert_transfer_post_allowed(self, *, coalesced: bool) -> None:
        """Authorize one native post at the transfer/compute boundary.

        :param coalesced: Whether the post is owned by a coalesced staging plan.
        :raises StagingSafetyError: If a post could overlap model execution or
            escape generation-scoped staging ownership.
        """
        if not self._phase_separate_transfer_decode:
            return
        if not self._transfer_phase_active or not coalesced:
            self._transfer_phase_violation_count += 1
            raise StagingSafetyError(
                "native transfer post is forbidden outside the owned transfer phase"
            )
        self._transfer_phase_handle_count += 1

    def _drain_transfer_phase(self) -> None:
        """Drain native receives, scatter, and device work before model execution.

        :raises StagingSafetyError: If work escapes the owned coalesced path or
            the phase cannot establish complete quiescence.
        """
        if not self._phase_separate_transfer_decode:
            return
        if not self._transfer_phase_active:
            self._transfer_phase_violation_count += 1
            raise StagingSafetyError("transfer phase drain has no active phase")

        while True:
            stock_handle_count = sum(
                len(handles) for handles in self._recving_transfers.values()
            )
            if stock_handle_count > 0:
                self._transfer_phase_violation_count += 1
                raise StagingSafetyError(
                    "stock receive handle escaped the coalesced transfer phase"
                )

            done_sending, done_recving = self._get_finished(service_pending=True)
            self._deferred_phase_sending.update(done_sending)
            self._deferred_phase_recving.update(done_recving)

            stock_handle_count = sum(
                len(handles) for handles in self._recving_transfers.values()
            )
            active_plan_count = len(self._coalesce_plans)
            pending_request_count = len(self._coalesce_pending)
            if stock_handle_count > 0:
                self._transfer_phase_violation_count += 1
                raise StagingSafetyError(
                    "stock receive handle appeared while draining the transfer phase"
                )
            if active_plan_count == 0 and pending_request_count == 0:
                break
            if active_plan_count == 0:
                self._transfer_phase_violation_count += 1
                raise StagingSafetyError(
                    "parked receive work has no active staging owner"
                )
            time.sleep(0.001)

        exit_sync_started_ns = time.monotonic_ns()
        try:
            torch.accelerator.synchronize()
        except Exception as error:
            self._transfer_phase_violation_count += 1
            raise StagingSafetyError(
                "device synchronization failed after the transfer phase\n"
                + traceback.format_exc()
            ) from error
        exit_sync_ns = time.monotonic_ns() - exit_sync_started_ns

        stock_handle_count = sum(
            len(handles) for handles in self._recving_transfers.values()
        )
        if (
            len(self._coalesce_plans) > 0
            or len(self._coalesce_pending) > 0
            or stock_handle_count > 0
        ):
            self._transfer_phase_violation_count += 1
            raise StagingSafetyError(
                "transfer work appeared after the compute-boundary synchronization"
            )

        finished_ns = time.monotonic_ns()
        self._transfer_phase_active = False
        self._transfer_phase_records.append(
            (
                "transfer-decode-phase",
                {
                    "active_plan_count": 0,
                    "byte_count": self._transfer_phase_byte_count,
                    "deferred_recving_count": len(self._deferred_phase_recving),
                    "deferred_sending_count": len(self._deferred_phase_sending),
                    "engine_id": self.engine_id,
                    "entry_sync_ns": self._transfer_phase_entry_sync_ns,
                    "epoch": self._transfer_phase_epoch,
                    "event": "compute_boundary",
                    "exit_sync_ns": exit_sync_ns,
                    "handle_count": self._transfer_phase_handle_count,
                    "pending_request_count": 0,
                    "plan_count": self._transfer_phase_plan_count,
                    "rank": self.tp_rank,
                    "stock_handle_count": 0,
                    "transfer_phase_ns": (
                        finished_ns - self._transfer_phase_started_ns
                    ),
                    "violations": self._transfer_phase_violation_count,
                },
            )
        )

    def _record_transfer_decode_boundary(self) -> None:
        """Record a non-mutating control-arm snapshot before model execution."""
        if (
            not self._phase_separation_instrumented
            or self._phase_separate_transfer_decode
        ):
            return

        possible_writer_states = {
            HandleState.POSTING,
            HandleState.PROC,
            HandleState.UNKNOWN,
            HandleState.ERR,
        }
        potential_plan_count = 0
        potential_byte_count = 0
        potential_handle_count = 0
        resident_byte_count = 0
        for plan in self._coalesce_plans.values():
            resident_byte_count += plan.lease.size
            plan_handle_count = sum(
                slot.state in possible_writer_states for slot in plan.slots.values()
            )
            potential_handle_count += plan_handle_count
            if plan_handle_count == 0:
                continue
            potential_plan_count += 1
            potential_byte_count += plan.lease.size
        stock_handle_count = sum(
            len(handles) for handles in self._recving_transfers.values()
        )
        self._transfer_phase_records.append(
            (
                "transfer-decode-boundary",
                {
                    "engine_id": self.engine_id,
                    "event": "compute_boundary_snapshot",
                    "pending_request_count": len(self._coalesce_pending),
                    "potential_active_handle_count": potential_handle_count,
                    "potential_byte_count": potential_byte_count,
                    "potential_overlap": (
                        potential_handle_count > 0 or stock_handle_count > 0
                    ),
                    "potential_plan_count": potential_plan_count,
                    "rank": self.tp_rank,
                    "resident_byte_count": resident_byte_count,
                    "resident_plan_count": len(self._coalesce_plans),
                    "stock_handle_count": stock_handle_count,
                },
            )
        )

    def _flush_transfer_phase_records(self) -> None:
        """Emit compute-boundary evidence after model execution has completed."""
        if len(self._transfer_phase_records) == 0:
            return
        records = tuple(self._transfer_phase_records)
        for marker, record in records:
            logger.info("[%s] %s", marker, json.dumps(record, sort_keys=True))
        del self._transfer_phase_records[: len(records)]

    def get_finished(self) -> tuple[set[str], set[str]]:
        """Publish transfer completions at the post-forward boundary.

        :returns: Requests done sending and receiving on this worker.
        """
        if not self._phase_separate_transfer_decode:
            result = self._get_finished(service_pending=True)
            if self._phase_separation_instrumented:
                self._flush_transfer_phase_records()
            return result
        if (
            self._transfer_phase_active
            or len(self._coalesce_plans) > 0
            or len(self._coalesce_pending) > 0
            or sum(len(handles) for handles in self._recving_transfers.values()) > 0
        ):
            self._transfer_phase_violation_count += 1
            raise StagingSafetyError(
                "post-forward completion observed undrained transfer work"
            )

        done_sending, done_recving = self._get_finished(service_pending=False)
        done_sending.update(self._deferred_phase_sending)
        done_recving.update(self._deferred_phase_recving)
        self._deferred_phase_sending.clear()
        self._deferred_phase_recving.clear()
        if self._phase_separation_instrumented:
            self._flush_transfer_phase_records()
        return done_sending, done_recving

    def _get_finished(
        self,
        *,
        service_pending: bool,
    ) -> tuple[set[str], set[str]]:
        """
        Get requests that are done sending or recving on this specific worker.
        The scheduler process (via the MultiprocExecutor) will use this output
        to track which workers are done.

        :param service_pending: Whether freed staging may launch parked pulls.
        :returns: Requests done sending and receiving on this worker.
        """
        self._service_heartbeats()
        assert self.transfer_topo is not None
        done_sending = self._get_new_notifs()
        done_recving = self._pop_done_transfers(self._recving_transfers)

        while not self._failed_recv_outcomes.empty():
            try:
                req_id, reason, handle = self._failed_recv_outcomes.get_nowait()
            except queue.Empty:
                break
            self._record_failed_receive(req_id, reason)
            if handle is not None:
                self.nixl_wrapper.release_xfer_handle(handle)
            self.xfer_stats.record_failed_transfer()

        for req_id in tuple(self._failed_recv_pending):
            plan = self._coalesce_plans.get(req_id)
            if plan is not None and plan.operation_failed:
                self._resolve_failed_coalesced_plan(req_id)

        done_recving.update(self._poll_coalesced_plans())

        failed_recv_reqs = set[ReqId]()
        for req_id in self._failed_recv_pending:
            if req_id in self._recving_transfers:
                continue
            if self._resolve_failed_coalesced_plan(req_id) is False:
                continue
            failed_recv_reqs.add(req_id)
        done_recving.update(failed_recv_reqs)

        if len(done_sending) > 0 or len(done_recving) > 0:
            logger.debug(
                "Rank %s, get_finished: %s requests done sending "
                "and %s requests done recving (%s failed)",
                self.tp_rank,
                len(done_sending),
                len(done_recving),
                len(failed_recv_reqs),
            )

        block_ids_for_blocksize_post_process = defaultdict(list)
        block_ids_for_heterogeneous_attn_post_process = list[list[int]]()
        for req_id in done_recving:
            # clean up metadata for completed requests
            meta = self._recving_metadata.pop(req_id, None)
            assert meta is not None, f"{req_id} not found in recving_metadata list"

            # Release fence: this pull completed for a rid whose producer
            # blocks were already released after all expected consumers
            # completed. The bytes may come from reused pages, so fail the
            # request instead of committing and publishing them.
            # Full-prefix-hit requests (empty local ids) read nothing and
            # are exempt.
            if (
                meta.remote is not None
                and meta.remote.request_id in self._released_rids
                and req_id not in failed_recv_reqs
                and sum(len(g) for g in meta.local_physical_block_ids) > 0
            ):
                logger.error(
                    "[release-fence] pull for %s completed after remote "
                    "request %s was released; failing instead of "
                    "committing.",
                    req_id,
                    meta.remote.request_id,
                )
                self._record_failed_receive(
                    req_id,
                    KVTransferFailureReason.INTEGRITY,
                    meta,
                )
                failed_recv_reqs.add(req_id)
            elif meta.remote is not None and req_id not in failed_recv_reqs:
                # Count this completion; the producer frees its blocks at
                # expected_consumers completions, so mirror that release
                # point locally.
                rid = meta.remote.request_id
                n_done = self._rid_completion_counts.get(rid, 0) + 1
                self._rid_completion_counts[rid] = n_done
                if n_done >= meta.remote.expected_consumers:
                    self._rid_completion_counts.pop(rid, None)
                    self._mark_rid_released(rid)

            # Skip KV sync and post-processing for failed requests
            if req_id in failed_recv_reqs:
                logger.warning(
                    "Skipping KV post-processing for failed request %s",
                    req_id,
                )
                self._discard_completed_coalesced_plan(req_id)
                continue

            assert meta.remote is not None
            if req_id in self._coalesce_plans:
                # coalesced pull: staged bytes -> real cache before the
                # scheduler sees the request as loaded
                self._coalesced_scatter(req_id)
            if self.use_host_buffer:
                self.sync_recved_kv_to_device(req_id, meta)

            # post processing for heteroblocksize
            remote_info = self.transfer_topo.get_engine_info(meta.remote.engine_id)
            block_size_ratio = self.transfer_topo.block_size_ratio(
                remote_info.remote_block_size
            )
            if not self.use_mla and (
                block_size_ratio > 1 or self.enable_permute_local_kv
            ):
                assert not self._is_hma_required
                block_ids_for_blocksize_post_process[block_size_ratio].append(
                    meta.local_physical_block_ids[0]
                )
            # post processing for heterogeneous attention
            if self.enable_heterogeneous_attn_post_process:
                block_ids_for_heterogeneous_attn_post_process.append(
                    meta.local_physical_block_ids[0]
                )
        for (
            block_size_ratio,
            block_ids_list,
        ) in block_ids_for_blocksize_post_process.items():
            self.post_process_device_kv_on_receive(block_size_ratio, block_ids_list)

        for block_ids in block_ids_for_heterogeneous_attn_post_process:
            self.post_process_device_kv_on_receive_heterogeneous_attn(block_ids)

        for req_id in failed_recv_reqs:
            failure = self._failed_recv_pending.pop(req_id)
            self._completed_failed_recv_outcomes.put((req_id, failure))

        self._sync_device_after_mamba_recv(done_recving, failed_recv_reqs)

        self._mark_overdue_leases(time.perf_counter())

        # coalesced pull: completed scatters freed staging; start
        # transfers for requests parked on the staging pool
        if service_pending and self._coalesce_pending:
            self._coalesce_service_pending()

        self._audit_tick(failed_recv_reqs)

        return done_sending, done_recving

    def _mark_overdue_leases(self, now: float) -> None:
        """Consume elapsed deadlines without releasing producer ownership.

        :param now: Current monotonic timestamp.
        """
        for req_id, expires in tuple(self._reqs_to_send.items()):
            if now < expires:
                continue
            if req_id not in self._reqs_to_process:
                raise RuntimeError(
                    f"Lease deadline for {req_id} has no producer ownership pin"
                )
            count = self._producer_completion_count(req_id)
            self.xfer_stats.record_kv_expired_req()
            logger.warning(
                "KV lease overdue for request %s after %d remote completion "
                "notification(s); retaining blocks until every expected "
                "consumer finishes.",
                req_id,
                count,
            )
            del self._reqs_to_send[req_id]

    def _producer_completion_count(self, req_id: ReqId) -> int:
        """Return completed remote obligations for one producer request.

        :param req_id: Producer request identifier.
        :returns: Number of completion proofs observed by this worker.
        """
        return self.consumer_notification_counts_by_req.get(req_id, 0)

    # ------------------------------------------------------------------
    # Resident-KV checksum auditor (VLLM_GEMMA4_KV_AUDIT)
    # ------------------------------------------------------------------

    def _audit_map_regions(self) -> list[int]:
        """Region indices serving the audited groups' layers, plus
        region 0 as a canary. HMA broadcast pulls write every region for
        every position, so any region detects transfer-side stomps; the
        owning group's regions additionally detect decode-side ones."""
        if self._audit_regions is not None:
            return self._audit_regions
        base_to_region = {t.data_ptr(): i for i, t in enumerate(self._region_tensors)}
        regs: set[int] = {0} if self._region_tensors else set()
        groups = self.kv_cache_config.kv_cache_groups
        for gi in sorted(self._audit_groups):
            if gi >= len(groups):
                continue
            for lname in groups[gi].layer_names:
                cache_or_caches = self.device_kv_caches.get(lname)
                if cache_or_caches is None:
                    continue
                caches = (
                    cache_or_caches
                    if isinstance(cache_or_caches, (list, tuple))
                    else [cache_or_caches]
                )
                for c in caches:
                    r = base_to_region.get(c.data_ptr())
                    if r is not None:
                        regs.add(r)
        self._audit_regions = sorted(regs)
        logger.info(
            "[kv-audit] auditing regions %s for groups %s",
            self._audit_regions,
            sorted(self._audit_groups),
        )
        return self._audit_regions

    def _audit_checksum(self, rows: torch.Tensor) -> torch.Tensor:
        """:param rows: LongTensor of physical row indices.
        :returns: [n_audit_regions, len(rows)] int64 content sums."""
        assert self._region_rows is not None
        outs = []
        for r in self._audit_map_regions():
            flat = self._region_rows[r]
            got = flat[rows]
            if got.shape[1] % 4 == 0:
                sums = got.view(torch.int32).sum(dim=1, dtype=torch.int64)
            else:
                sums = got.sum(dim=1, dtype=torch.int64)
            outs.append(sums)
        return torch.stack(outs)

    def _audit_drop(self, rid: str, alive: bool) -> None:
        """Retire one rid's audit state; journal its rows as freed."""
        st = self._audit_state.pop(rid, None)
        if st is None:
            return
        now = time.perf_counter()
        for p in st["rows"].tolist():
            if self._audit_row_to_rid.get(p) == rid:
                del self._audit_row_to_rid[p]
            owner = self._audit_row_owner.get(p)
            if owner is not None and owner[0] == rid:
                self._audit_row_owner[p] = (rid, now, alive)

    def _audit_retire(self, metadata) -> None:
        """Drop audit state for requests the scheduler finished; their
        rows may be legitimately reallocated from now on."""
        if not self._audit_enabled:
            return
        for rid in getattr(metadata, "audit_finished", None) or ():
            self._audit_drop(rid, alive=False)

    def _audit_poison(self, req_id: str) -> None:
        """Self-test: flip bytes in one audited row so the next verify
        pass MUST report a mismatch (proves the instrument detects)."""
        st = self._audit_state.get(req_id)
        if st is None or st["rows"].numel() == 0:
            return
        assert self._region_rows is not None
        r0 = self._audit_map_regions()[-1]
        row = int(st["rows"][st["rows"].numel() // 2])
        self._region_rows[r0][row, 128:144] ^= 0x01
        logger.error(
            "[kv-audit-selftest] poisoned region %d row %d of %s; a "
            "MISMATCH report for this row must follow",
            r0,
            row,
            req_id,
        )

    def _audit_tick(self, failed_recv_reqs: set[str]) -> None:
        """Snapshot rows committed this step; periodically re-verify
        all audited requests' immutable rows."""
        if not self._audit_enabled or self._region_rows is None:
            return
        if self._audit_pending:
            dev = self._region_rows[0].device
            now = time.perf_counter()
            for req_id, arows, all_rows in self._audit_pending:
                if req_id in failed_recv_reqs or len(arows) == 0:
                    continue
                # ROW-TAKEOVER: this pull wrote rows another live audited
                # request still owns -- double-assignment, the bug class
                # itself. Report and retire the trampled entry (its
                # content is gone; a MISMATCH would only be noise).
                taken: dict[str, list[int]] = {}
                for p in all_rows:
                    old = self._audit_row_to_rid.get(p)
                    if old is not None and old != req_id:
                        taken.setdefault(old, []).append(p)
                for old_rid, rows_taken in taken.items():
                    logger.error(
                        "[kv-audit] ROW-TAKEOVER: pull for %s wrote %d "
                        "rows still audited for LIVE request %s "
                        "(first=%s) -- double-assigned KV pages",
                        req_id,
                        len(rows_taken),
                        old_rid,
                        rows_taken[:8],
                    )
                    self._audit_drop(old_rid, alive=True)
                rows_t = torch.tensor(sorted(set(arows)), device=dev, dtype=torch.long)
                self._audit_state[req_id] = {
                    "rows": rows_t,
                    "sums": self._audit_checksum(rows_t),
                    "t": now,
                }
                for p in arows:
                    self._audit_row_to_rid[p] = req_id
                for p in all_rows:
                    self._audit_row_owner[p] = (req_id, now, True)
                self._audit_snap_count += 1
                logger.info(
                    "[kv-audit] snapshot %s: %d immutable rows x %d "
                    "regions (%d reqs audited)",
                    req_id,
                    rows_t.numel(),
                    len(self._audit_map_regions()),
                    len(self._audit_state),
                )
                if (
                    self._audit_selftest
                    and self._audit_snap_count == self._audit_selftest
                ):
                    self._audit_poison(req_id)
            self._audit_pending.clear()
        self._audit_step += 1
        if self._audit_step % self._audit_interval or not self._audit_state:
            return
        for req_id, st in list(self._audit_state.items()):
            cur = self._audit_checksum(st["rows"])
            bad = (cur != st["sums"]).any(dim=0)
            n_bad = int(bad.sum())
            if n_bad == 0:
                continue
            idx = bad.nonzero().flatten()[:8]
            rows = st["rows"][idx].tolist()
            owners = {p: self._audit_row_owner.get(p) for p in rows}
            regions_hit = (cur[:, idx] != st["sums"][:, idx]).sum(dim=1).tolist()
            logger.error(
                "[kv-audit] MISMATCH req=%s rows_bad=%d/%d first=%s "
                "region_hit_counts=%s owners=%s age=%.1fs step=%d",
                req_id,
                n_bad,
                st["rows"].numel(),
                rows,
                regions_hit,
                owners,
                time.perf_counter() - st["t"],
                self._audit_step,
            )
            # Re-arm to current content so each distinct stomp logs once.
            st["sums"] = cur

    def _coalesce_service_pending(self) -> None:
        """Overridden by the pull worker; push has no pull staging."""
        return

    def _sync_device_after_mamba_recv(
        self,
        done_recving: set[str],
        failed_recv_reqs: set[str],
    ) -> None:
        """Synchronize ROCm direct-GPU Mamba receives before model execution."""
        if (
            not current_platform.is_rocm()
            or not self._has_mamba
            or self.use_host_buffer
            or not (done_recving - failed_recv_reqs)
        ):
            return

        torch.accelerator.synchronize()

    def _get_new_notifs(self) -> set[str]:
        """Get req_ids which got a remote xfer notification.

        Subclasses must implement this to handle mode-specific notifications.
        """
        raise NotImplementedError

    def _mark_rid_released(self, rid: str) -> None:
        self._released_rids[rid] = time.perf_counter()
        if len(self._released_rids) > 8192:
            for k in list(self._released_rids)[:2048]:
                del self._released_rids[k]

    def _handle_heartbeat(self, payload: str) -> None:
        """Extend leases for requests referenced in a heartbeat.

        Args:
            payload: comma-separated P-side request IDs, e.g.
                     "req_abc,req_def".
        """
        new_expiry = time.perf_counter() + self._lease_extension
        for req_id in payload.split(","):
            if req_id in self._reqs_to_send:
                old = self._reqs_to_send[req_id]
                self._reqs_to_send[req_id] = max(old, new_expiry)
                logger.debug(
                    "Heartbeat extended lease for request %s "
                    "by %ds (old_expiry=%.1f, new_expiry=%.1f)",
                    req_id,
                    self._lease_extension,
                    old,
                    new_expiry,
                )

    def _pop_done_transfers(
        self,
        transfers: dict[str, list[int]],
        failed_transfer_handler: Callable[[str, int | None], None] | None = None,
    ) -> set[str]:
        """
        Pop completed xfers by checking for DONE state.
        Args:
            transfers: dict of req_id -> list[running_xfer]
        Returns:
            set of req_ids that have all done xfers
        """
        if failed_transfer_handler is None:
            failed_transfer_handler = self._handle_failed_transfer

        done_req_ids: set[str] = set()
        for req_id, handles in list(transfers.items()):
            in_progress = []
            for handle in handles:
                try:
                    xfer_state = self.nixl_wrapper.check_xfer_state(handle)
                    if xfer_state == "DONE":
                        # Get telemetry from NIXL
                        res = self.nixl_wrapper.get_xfer_telemetry(handle)
                        self.xfer_stats.record_transfer(res)
                        self.nixl_wrapper.release_xfer_handle(handle)
                    elif xfer_state == "PROC":
                        in_progress.append(handle)
                        continue
                    else:
                        self._log_failure(
                            failure_type="transfer_failed",
                            msg="Handling failed transfer",
                            req_id=req_id,
                            xfer_state=xfer_state,
                        )
                        failed_transfer_handler(req_id, handle)
                except Exception as e:
                    self._log_failure(
                        failure_type="transfer_exception",
                        msg="Handling failed transfer",
                        req_id=req_id,
                        error=e,
                    )
                    failed_transfer_handler(req_id, handle)

            if not in_progress:
                # Only report request as completed when all transfers are done.
                done_req_ids.add(req_id)
                del transfers[req_id]
            else:
                transfers[req_id] = in_progress
        return done_req_ids

    def _make_failed_receive(
        self,
        req_id: ReqId,
        reason: KVTransferFailureReason,
        meta: ReqMeta | None = None,
    ) -> KVTransferFailure:
        """Build a request-scoped failure over every local cache group.

        :param req_id: Decoder request identifier.
        :param reason: Failure classification.
        :param meta: Optional metadata retained by the current caller.
        :returns: Typed failure with every affected logical block ID.
        """
        if meta is None:
            meta = self._recving_metadata.get(req_id)
        invalid_block_ids = (
            frozenset(
                block_id
                for group_block_ids in meta.local_block_ids
                for block_id in group_block_ids
            )
            if meta is not None
            else frozenset()
        )
        return KVTransferFailure(
            reason=reason,
            invalid_block_ids=invalid_block_ids,
        )

    def _record_failed_receive(
        self,
        req_id: ReqId,
        reason: KVTransferFailureReason,
        meta: ReqMeta | None = None,
    ) -> None:
        """Merge one failure into the main-thread request outcome.

        :param req_id: Decoder request identifier.
        :param reason: Failure classification.
        :param meta: Optional metadata retained by the current caller.
        """
        failure = self._make_failed_receive(req_id, reason, meta)
        existing = self._failed_recv_pending.get(req_id)
        self._failed_recv_pending[req_id] = (
            failure if existing is None else existing.aggregate(failure)
        )

    def _handle_failed_transfer(
        self,
        req_id: str,
        handle: int | None,
        reason: KVTransferFailureReason = KVTransferFailureReason.TRANSFER,
    ) -> None:
        """Queue a receive failure for main-thread ownership processing.

        :param req_id: Decoder request identifier.
        :param handle: Native transfer handle, if one was created.
        :param reason: Failure classification.
        """
        meta = self._recving_metadata.get(req_id)
        remote = meta.remote if meta is not None else None
        self._localization_record_event(
            code="TRANSFER_ABORTED",
            evidentiary=False,
            child_request_id=req_id,
            producer_engine_id=(remote.engine_id if remote is not None else None),
            producer_request_id=(remote.request_id if remote is not None else None),
            detail="transfer failed before complete diagnostic capture",
        )
        self._localization_pre_read_plans.pop(req_id, None)
        self._failed_recv_outcomes.put((req_id, reason, handle))

    def _handle_failed_sending_transfer(
        self,
        _req_id: str,
        handle: int | None,
    ) -> None:
        """Release a terminal send failure without reporting a failed receive.

        :param _req_id: Sending request identifier retained for handler symmetry.
        :param handle: Native transfer handle, if one was created.
        """
        if handle is not None:
            self.nixl_wrapper.release_xfer_handle(handle)
        self.xfer_stats.record_failed_transfer()

    def _update_heartbeat_targets(self, metadata: NixlConnectorMetadata) -> None:
        """Apply a complete scheduler-side heartbeat ownership snapshot.

        :param metadata: Connector metadata for the current model step.
        """
        if metadata.heartbeat_snapshot is not None:
            self._heartbeat_targets = metadata.heartbeat_snapshot

    def _service_heartbeats(self) -> None:
        """Dispatch retained heartbeat targets when their cadence is due."""
        if len(self._heartbeat_targets) == 0:
            return
        now = time.perf_counter()
        if now - self._last_heartbeat_time < self._heartbeat_interval:
            return

        self._last_heartbeat_time = now
        self._dispatch_heartbeat_targets(dict(self._heartbeat_targets))

    def _dispatch_heartbeat_targets(
        self,
        targets: dict[EngineId, HeartbeatInfo],
    ) -> None:
        """Dispatch one due heartbeat batch on the current worker thread.

        :param targets: Producer engines and request ownership to renew.
        """
        self._send_heartbeat_targets(targets)

    def _send_heartbeat_targets(
        self,
        targets: dict[EngineId, HeartbeatInfo],
    ) -> None:
        """Send heartbeat notifications for one retained ownership snapshot.

        :param targets: Producer engines and request ownership to renew.
        """
        for engine_id, hb_info in targets.items():
            # Proactive handshake (this request may still be in waiting queue) so
            # the **next** heartbeat for this remote can go through.
            if (
                self._ensure_handshake(
                    engine_id, hb_info.host, hb_info.port, hb_info.tp_size
                )
                is not None
            ):
                continue  # handshake is still pending

            # Build the heartbeat message: "HB:req1,req2,..."
            hb_msg = ("HB:" + ",".join(sorted(hb_info.request_refcounts))).encode()
            for agent_name in self._remote_agents[engine_id].values():
                try:
                    self.nixl_wrapper.send_notif(agent_name, notif_msg=hb_msg)
                except Exception:
                    logger.warning(
                        "Failed to send heartbeat to engine %s\n%s",
                        engine_id,
                        traceback.format_exc(),
                    )

    def get_mapped_blocks(
        self, block_ids: np.ndarray, block_size_ratio: int
    ) -> np.ndarray:
        """
          Calculates the new set of block IDs by mapping every element
          in the (potentially sparse) input array.
          Example: block_ids=[0, 2], block_size_ratio=2
        get_mapped_blocks    0     1     [2     3]     4     5
              # remote is |h0-b0|h1-b0||h0-b1|h1-b1||h0-b1|h1-b1||
              # local is  |h0-b0......||h1-b0......||h2-b0........
        local_block_ids         0           [1]           2
        """
        if block_ids.size == 0:
            return np.array([], dtype=np.int64)

        start_ids = block_ids * block_size_ratio
        offsets = np.arange(block_size_ratio)
        mapped_2d = start_ids[:, None] + offsets[None, :]

        return mapped_2d.flatten().astype(np.int64)

    def _logical_to_kernel_block_ids(self, block_ids: BlockIds) -> BlockIds:
        """
        Convert logical block ids to kernel physical block ids.
        This is required when the logical block size (the one set by the user)
        does not match the one required by the attn backend.
        """
        if self._physical_blocks_per_logical_kv_block == 1:
            # Noop when physical and logical block sizes are the same
            return block_ids
        block_arange = np.arange(0, self._physical_blocks_per_logical_kv_block).reshape(
            1, -1
        )
        # Mamba blocks have no logical<>physical discrepancy
        group_specs = self.kv_cache_config.kv_cache_groups
        return [
            BlockTable.map_to_kernel_blocks(
                np.array(group),
                self._physical_blocks_per_logical_kv_block,
                block_arange,
            ).tolist()
            if not isinstance(group_specs[i].kv_cache_spec, MambaSpec)
            else group
            for i, group in enumerate(block_ids)
        ]

    def _apply_prefix_caching(
        self,
        local_block_ids: BlockIds,
        remote_block_ids: BlockIds,
        remote_physical_per_logical: int,
    ) -> tuple[BlockIds, list]:
        """Apply prefix caching by trimming local/remote block ID lists.

        For non-Mamba models: end-trim remote to match local count, so that
        already-cached prefix blocks are skipped in the transfer.

        For Mamba hybrid (prefix caching not yet supported): front-trim both
        to the minimum count to handle kernel block count discrepancies from
        logical block rounding in heterogeneous TP.
        """
        # Partial prefix cache hit: just read uncomputed blocks.
        # Skip mamba groups — their blocks represent full state (conv+ssm),
        # not per-token data, so trimming would corrupt the transfer.
        remote_block_ids = list(remote_block_ids)
        if not self._has_mamba:
            sp_flags = self._sp_group_flags()
            for i, remote_group in enumerate(remote_block_ids):
                if i in self._skip_pull_groups:
                    remote_block_ids[i] = []
                    local_block_ids[i] = []
                    continue
                num_local_blocks = len(local_block_ids[i])
                # F2b single-plane groups: local blocks hold 2x remote
                # tokens, so a fully-uncached request legitimately has
                # len(local) ~= len(remote)/2. The end-trim must keep
                # 2x len(local) remote blocks or it silently drops the
                # FIRST HALF of the context (found via margin collapse +
                # position-scrambled needle retrieval, 2026-07-08).
                factor = 2 if sp_flags[i] else 1
                total_local_blocks = (len(remote_group) + factor - 1) // factor
                assert num_local_blocks <= total_local_blocks
                cached_local_blocks = total_local_blocks - num_local_blocks
                remote_start = factor * cached_local_blocks
                if num_local_blocks != len(remote_group):
                    logger.info(
                        "[prefix-trim] group=%s len_local=%d len_remote=%d "
                        "remote_start=%d first_local=%s first_remote=%s",
                        i,
                        num_local_blocks,
                        len(remote_group),
                        remote_start,
                        local_block_ids[i][:2] if num_local_blocks else [],
                        remote_group[:2],
                    )
                remote_block_ids[i] = remote_group[remote_start:]
        else:
            # (NOTE: ZhanqiuHu) Mamba hybrid: no prefix caching support so far.HeteroTP
            # can cause different kernel block counts due to logical block rounding.
            # Example: 640 prompt tokens, kernel_block_size=64
            #   remote physical_per_logical=10, local physical_per_logical=6
            #   remote logical ids from kv_transfer_params = [0]
            #   local logical ids allocated = [0, 1]
            #   remote kernel blocks: [0..9]  (1*10=10)
            #   local kernel blocks:  [0..11] (2*6=12)
            #   actual data blocks = ceil(640/64) = 10, trim both to 10
            # Vice versa (remote physical_per_logical=6, local=10):
            #   remote logical ids = [0, 1], local logical ids = [0]
            #   remote kernel blocks: [0..11] (2*6=12)
            #   local kernel blocks:  [0..9]  (1*10=10)
            #   actual data blocks = ceil(640/64) = 10, trim both to 10
            local_block_ids = list(local_block_ids)
            for i, remote_group in enumerate(remote_block_ids):
                num_local_blocks = len(local_block_ids[i])
                num_remote_blocks = len(remote_group)
                if (
                    _is_ssm_spec(self._group_spec_types[i])
                    and num_local_blocks < num_remote_blocks
                ):
                    # NOTE (NickLucche): With prefix caching on SSM, (remote) blocks
                    # prior to the last one are placeholders (null blocks). Mind that
                    # this doesn't really impact transfer, as we only still care about
                    # the last "block", the full in-place state.
                    assert num_local_blocks == 1, "SSM can only have one local block"
                    remote_block_ids[i] = remote_group[-num_local_blocks:]
                elif (
                    self._physical_blocks_per_logical_kv_block
                    == remote_physical_per_logical
                    and num_local_blocks < num_remote_blocks
                ):
                    # Partial prefix cache hit for FA group.
                    remote_block_ids[i] = remote_group[-num_local_blocks:]
                else:
                    # TODO Handle prefix caching with different block_sizes
                    max_padding = max(
                        self._physical_blocks_per_logical_kv_block,
                        remote_physical_per_logical,
                    )
                    assert abs(num_local_blocks - num_remote_blocks) < max_padding, (
                        f"Group {i}: |{num_local_blocks} - "
                        f"{num_remote_blocks}| >= {max_padding}"
                    )
                    num_blocks = min(num_local_blocks, num_remote_blocks)
                    local_block_ids[i] = local_block_ids[i][:num_blocks]
                    remote_block_ids[i] = remote_group[:num_blocks]
        return local_block_ids, remote_block_ids

    def _logical_to_remote_kernel_block_ids(
        self, block_ids: BlockIds, remote_physical_per_logical: int
    ) -> BlockIds:
        """Map logical block IDs to physical kernel block IDs on the remote.

        Args:
            block_ids: per-group lists of logical block IDs.
            remote_physical_per_logical: remote engine's physical blocks
                per logical block.

        Returns:
            Same structure with FA groups expanded (each logical block L
            becomes kernel blocks [L*remote_physical_per_logical, ..
            L*remote_physical_per_logical +
            remote_physical_per_logical - 1]).
            Mamba groups are passed through unchanged.
        """
        if remote_physical_per_logical == 1:
            return block_ids
        remote_arange = np.arange(remote_physical_per_logical).reshape(1, -1)
        group_specs = self.kv_cache_config.kv_cache_groups
        result = [
            BlockTable.map_to_kernel_blocks(
                np.array(group),
                remote_physical_per_logical,
                remote_arange,
            ).tolist()
            if not isinstance(group_specs[i].kv_cache_spec, MambaSpec)
            else group
            for i, group in enumerate(block_ids)
        ]
        return result

    def get_backend_aware_kv_block_len(
        self, layer_idx: int, first_split: bool = True, mamba_view: bool = False
    ) -> int:
        """
        Get the block length for one K/V element (K and V have the same size).

        For FA and other backends, this is equal to the length of the whole
        block, as K and V are in separate regions.
        For FlashInfer, this is half the length of the whole block, as K and V
        share the same region.
        Similarly, for SSM-based models, state and conv are interleaved, but crucially
        the their size differs.
        Reference diagram:
                            KVCacheTensor (Shared)
                               /       \\
                              /         \\
                             /           \\
        Attention (FlashInfer) View      Mamba View
                  |                          |
                  |                          |
           +-------------------+         +-------------------+
           | KVCacheTensor     |         | KVCacheTensor      |
           |                   |         |                    |
           |<----- page ------>|         |<----- page ------->|
           |       size        |         |       size         |
           |  Key 0  |  Val 0  |         |Conv 0  |   SSM 0   |
           |  Key 1  |  Val 1  |         |Conv 1  |   SSM 1   |
           |   ...   |   ...   |         |  ...   |    ...    |
           | Key N-2 | Val N-2 |         |Conv N-2|   SSM N-2 |
           | Key N-1 | Val N-1 |         |Conv N-1|   SSM N-1 |
           +-------------------+         +--------------------+
           |1st_split-2nd_split|         |1st_split-2nd_split |
        """
        assert self.transfer_topo is not None
        virtually_split = self.transfer_topo.virtually_split_kv_in_blocks
        if virtually_split and mamba_view:
            block_len = self._mamba_ssm_size[not first_split]
        else:
            half_block = virtually_split and not self._is_region_replicated(layer_idx)
            block_len = self.block_len_per_layer[layer_idx] // (2 if half_block else 1)
        return block_len

    def get_kv_connector_stats(self) -> KVConnectorStats | None:
        """
        Get the KV transfer stats for the connector.
        """
        # Clear stats for next iteration
        if not self.xfer_stats.is_empty():
            return self.xfer_stats.clone_and_reset()
        return None

    def get_block_ids_with_load_errors(self) -> set[int]:
        """
        Return and clear the set of block IDs that failed to load.

        This is called by the scheduler to identify blocks that need
        to be retried after a NIXL transfer failure.
        """
        # Drain the queue (thread-safe, no lock needed).
        result: set[int] = set()
        while not self._invalid_block_ids.empty():
            try:
                result.update(self._invalid_block_ids.get_nowait())
            except queue.Empty:
                break
        return result

    def get_failed_recving(self) -> dict[str, KVTransferFailure]:
        """Return and clear completed request-scoped receive failures.

        :returns: Terminal failures keyed by request ID.
        """
        result: dict[str, KVTransferFailure] = {}
        while not self._completed_failed_recv_outcomes.empty():
            try:
                req_id, failure = self._completed_failed_recv_outcomes.get_nowait()
            except queue.Empty:
                break
            existing = result.get(req_id)
            result[req_id] = (
                failure if existing is None else existing.aggregate(failure)
            )
        return result

    def _evict_stale_engines(self) -> None:
        """Scan for and evict remote engines that have exceeded their TTL.

        Called from the main thread in when a new remote engine appears.
        We can only go OOM as we discover and register a new remote, therefore we make
        sure we clean up stale engine data structures before then. This invariant
        prevents us from using background threads, though memory usage is not guaranteed
        to be "optimal" until a new handshake is performed.

        Pending handshakes do not have an ``_engine_last_active`` entry yet.
        Coalesced owners are checked explicitly because a long-running native
        transfer can outlive the last start-time activity update.
        """
        # NOTE (NickLucche): This does NOT currently prevent OOMing if a huge number
        # of remote engines is registered all at once (adding a background cleanup
        # thread wouldnt help either).
        # If that scenario is plausible, we can follow up with an LRU eviction policy.
        if self._engine_ttl <= 0:
            return

        now = time.perf_counter()
        for eid, last_active in list(self._engine_last_active.items()):
            active_plans = [
                plan
                for plan in self._coalesce_plans.values()
                if plan.remote_engine_id == eid
            ]
            if len(active_plans) > 0:
                self._engine_last_active[eid] = now
                logger.warning(
                    "Skipping remote-engine eviction while staging is owned: %s",
                    "; ".join(plan.describe(now) for plan in active_plans),
                )
                continue
            if now - last_active > self._engine_ttl:
                self._cleanup_remote_engine(eid)

    def _cleanup_remote_engine(
        self, engine_id: EngineId, *, log_eviction: bool = True
    ) -> None:
        """Remove all state for a single remote engine.

        Releases NIXL resources (dlist handles, remote agents) and clears
        all per-engine data structures. Used by both TTL eviction and
        shutdown.
        """
        assert engine_id in self._remote_agents
        unsafe_plans = [
            plan
            for plan in self._coalesce_plans.values()
            if plan.remote_engine_id == engine_id and plan.reusable is False
        ]
        if len(unsafe_plans) > 0:
            raise StagingSafetyError(
                "remote-engine cleanup would release resources with live staging "
                + "; ".join(plan.describe() for plan in unsafe_plans)
            )

        for handle in self.dst_xfer_side_handles.pop(engine_id).values():
            self.nixl_wrapper.release_dlist_handle(handle)
        for agent_name in self._remote_agents.pop(engine_id).values():
            self.nixl_wrapper.remove_remote_agent(agent_name)

        del self.kv_caches_base_addr[engine_id]
        del self.dst_num_blocks[engine_id]
        del self.tp_mappings[engine_id]
        self._remote_layout.pop(engine_id, None)
        self._remote_regions.pop(engine_id, None)
        self._remote_registration_generations.pop(engine_id, None)
        if self.transfer_topo is not None:
            self.transfer_topo.unregister_remote_engine(engine_id)

        last_active = self._engine_last_active.pop(engine_id)
        if log_eviction:
            logger.info(
                "Evicted stale remote engine %s (inactive for %.1fs).",
                engine_id,
                time.perf_counter() - last_active,
            )

    def __del__(self) -> None:
        try:
            self.shutdown()
        except StagingSafetyError:
            logger.critical(
                "Refusing in-process NIXL teardown with unresolved staging ownership",
                exc_info=True,
            )

    def shutdown(self) -> None:
        """Shutdown the connector worker."""
        if not hasattr(self, "_handshake_initiation_executor"):
            # error happens during init, no need to shutdown
            return
        unsafe_plans = [
            plan for plan in self._coalesce_plans.values() if plan.reusable is False
        ]
        if len(unsafe_plans) > 0:
            raise StagingSafetyError(
                "in-process shutdown cannot quiesce coalesced staging: "
                + "; ".join(plan.describe() for plan in unsafe_plans)
            )
        if self._localization_writer is not None:
            self._localization_writer.close()
            self._localization_writer = None
        self._handshake_initiation_executor.shutdown(wait=False)
        for handles in self._recving_transfers.values():
            for handle in handles:
                self.nixl_wrapper.release_xfer_handle(handle)
        self._recving_transfers.clear()
        for handle in self.src_xfer_handles_by_block_size.values():
            self.nixl_wrapper.release_dlist_handle(handle)
        self.src_xfer_handles_by_block_size.clear()
        for handles in self.src_xfer_handles_by_tp_ratio.values():
            for handle in handles:
                self.nixl_wrapper.release_dlist_handle(handle)
        self.src_xfer_handles_by_tp_ratio.clear()
        for engine_id in list(self._remote_agents):
            self._cleanup_remote_engine(engine_id, log_eviction=False)
        for desc in self._registered_descs:
            self.nixl_wrapper.deregister_memory(desc)
        self._registered_descs.clear()
