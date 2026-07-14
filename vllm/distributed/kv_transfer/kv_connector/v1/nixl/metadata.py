# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Metadata dataclasses and helpers for the NIXL connector."""

from dataclasses import dataclass
from typing import Any

import msgspec

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds, EngineId
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorHandshakeMetadata,
    KVConnectorMetadata,
)
from vllm.distributed.kv_transfer.nixl_localization import (
    NixlRegionDescriptor,
    NixlSourceRoster,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

TransferHandle = int
ReqId = str

GET_META_MSG = b"get_meta_msg"

# Push-mode (WRITE-based) registration notification.
# Sent worker-to-worker over NIXL: D worker -> P worker, encoded as
# PUSH_REG_NOTIF_PREFIX + msgpack(registration_data).
PUSH_REG_NOTIF_PREFIX = b"PUSH_REG:"
PULL_READ_COMPLETE_PREFIX = b"PULL_READ_COMPLETE:"
PULL_OFFER_CANCELLATION_CONTROL_PREFIX = b"PULL_OFFER_CANCELLATION:"
#
# NIXL Connector Version
#
# Increment this version whenever there is an incompatible change to:
#   - NixlAgentMetadata schema
#   - kv_transfer_params schema or semantics
#   - NIXL transfer protocol or wire format
#   - KV cache memory layout or block organization
#   - Any other change that breaks P/D interoperability
#
# Version History:
#   1: Initial version with compatibility checking
#   2: Add remote_request_id to kv_transfer_params
#   3: Add physical_blocks_per_logical_kv_block to NixlAgentMetadata
#   4: Add KV block lease renewal through heartbeats
#   5: Add gated P-to-D source integrity manifests
#   6: Replace source gating with post-transfer source references
#   7: Add producer-owned leases and idempotent pull completion proofs
#   8: Add exact decoder-rank whole-offer cancellation proofs
#
NIXL_CONNECTOR_VERSION: int = 8


@dataclass
class NixlAgentMetadata:
    engine_id: str
    agent_metadata: bytes
    kv_caches_base_addr: list[int]
    device_id: int
    num_blocks: int
    block_lens: list[int]
    kv_cache_layout: str
    block_size: int
    ssm_sizes: tuple[int, int]
    attn_backend_name: str
    physical_blocks_per_logical_kv_block: int
    registration_generation: str = ""
    regions: tuple[NixlRegionDescriptor, ...] = ()
    source_group_planes: tuple[int, ...] = ()
    physical_group_token_capacities: tuple[int, ...] = ()


@dataclass
class NixlHandshakePayload(KVConnectorHandshakeMetadata):
    """
    Wrapper for NIXL handshake sent over the wire.

    Enables two-phase decoding for graceful compatibility checking:
    1. Decode NixlHandshakePayload to get compatibility_hash
    2. Compute local hash and compare
    3. Only if hashes match, decode agent_metadata_bytes

    This prevents decoder errors when NixlAgentMetadata schema is
    incompatible, allowing graceful failure with clear error message.
    """

    compatibility_hash: str
    agent_metadata_bytes: bytes  # NixlAgentMetadata encoded


def compute_nixl_compatibility_hash(
    vllm_config: VllmConfig, attn_backend_name: str, cross_layers_blocks: bool
) -> str:
    """
    Compute compatibility hash for NIXL KV transfer.

    Hash only the factors that affect whether two NIXL instances can
    successfully transfer KV cache data.

    Factors included:
    - vLLM version and NIXL connector version
    - Model architecture (name, dtype, KV heads, layers)
    - KV cache format (dtype, sliding window)
    - Attention backend

    Note: Factors like tensor_parallel_size, block_size, and kv_cache_layout
    are validated at runtime in _validate_remote_agent_handshake and are not
    included in this hash to support heterogeneous deployments.

    Note - the set of factors are likely to evolve significantly over
    time to be more or less permissive.

    Returns:
        SHA-256 hex digest
    """
    from vllm import __version__ as vllm_version
    from vllm.config.utils import hash_factors

    model_config = vllm_config.model_config
    cache_config = vllm_config.cache_config
    is_hma_enabled = not vllm_config.scheduler_config.disable_hybrid_kv_cache_manager

    factors = {
        # Version compatibility
        "vllm_version": vllm_version,
        "nixl_connector_version": NIXL_CONNECTOR_VERSION,
        # Model architecture - affects KV cache shape
        "model": model_config.model,
        "dtype": str(model_config.dtype),
        "num_kv_heads": model_config.get_total_num_kv_heads(),
        "head_size": model_config.get_head_size(),
        "num_hidden_layers": model_config.get_total_num_hidden_layers(),
        # Attention backend and KV cache dtype affect memory layout
        "attn_backend_name": attn_backend_name,
        "cache_dtype": str(cache_config.cache_dtype),
        "cross_layers_blocks": cross_layers_blocks,
        "is_hma_enabled": is_hma_enabled,
    }

    compat_hash = hash_factors(factors)
    logger.debug(
        "NIXL compatibility hash: %s (model=%s, dtype=%s, num_kv_heads=%d, "
        "cache_dtype=%s, attn_backend=%s)",
        compat_hash,
        factors["model"],
        factors["dtype"],
        factors["num_kv_heads"],
        factors["cache_dtype"],
        attn_backend_name,
    )
    return compat_hash


@dataclass
class HeartbeatInfo:
    """Heartbeat ownership for one producer engine.

    :ivar request_refcounts: Active decoder consumers grouped by producer request.
    :ivar host: Producer side-channel host.
    :ivar port: Producer side-channel port.
    :ivar tp_size: Producer tensor-parallel size.
    """

    request_refcounts: dict[ReqId, int]
    host: str
    port: int
    tp_size: int


@dataclass(frozen=True)
class ProducerLease:
    """Producer ownership contract for remotely readable KV blocks.

    :ivar deadline: Monotonic liveness deadline for operational reporting.
    :ivar expected_consumers: Logical decoder children that may read the blocks.
    :ivar consumer_tp_size: Decoder tensor-parallel size covered by the lease.
    """

    deadline: float
    expected_consumers: int
    consumer_tp_size: int


class PullReadComplete(msgspec.Struct, frozen=True, array_like=True):
    """Idempotent proof that one decoder rank finished one logical read.

    :ivar producer_request_id: Request whose producer pages were read.
    :ivar consumer_request_id: Concrete decoder request for diagnostics.
    :ivar consumer_index: Stable parallel-sampling child index.
    :ivar consumer_rank: Decoder tensor-parallel rank that completed the read.
    :ivar consumer_tp_size: Decoder tensor-parallel world size.
    :ivar expected_consumers: Decoder view of the producer's consumer contract.
    """

    producer_request_id: ReqId
    consumer_request_id: ReqId
    consumer_index: int
    consumer_rank: int
    consumer_tp_size: int
    expected_consumers: int


class PullOfferCancelled(msgspec.Struct, frozen=True, array_like=True):
    """Proof that one decoder rank admitted no consumer from an offer.

    :ivar producer_request_id: Producer request whose pages remain offered.
    :ivar consumer_rank: Decoder tensor-parallel rank making the assertion.
    :ivar consumer_tp_size: Decoder tensor-parallel world size.
    :ivar expected_consumers: Decoder view of the producer-owned contract.
    """

    producer_request_id: ReqId
    consumer_rank: int
    consumer_tp_size: int
    expected_consumers: int


class PullOfferCancellationControl(msgspec.Struct, frozen=True, array_like=True):
    """Side-channel request carrying one decoder-rank cancellation proof.

    :ivar producer_ranks: Exact producer ranks covered by the decoder rank.
    :ivar proof: Whole-offer cancellation proof to deliver to those ranks.
    """

    producer_ranks: tuple[int, ...]
    proof: PullOfferCancelled


class PullOfferCancellationAck(msgspec.Struct, frozen=True, array_like=True):
    """Producer acknowledgement for a queued cancellation control request.

    :ivar producer_request_id: Producer offer accepted by the control plane.
    :ivar producer_ranks: Exact producer ranks that will receive the proof.
    :ivar accepted: Whether the complete control request was queued atomically.
    """

    producer_request_id: ReqId
    producer_ranks: tuple[int, ...]
    accepted: bool


@dataclass
class RemoteMeta:
    block_ids: BlockIds
    host: str
    port: int
    engine_id: str
    request_id: str
    remote_num_tokens: int = 0
    # Immutable producer-owned decoder topology. The producer releases
    # pages only after every exact child/rank obligation is proven complete.
    expected_consumers: int = 1
    consumer_tp_size: int = 1
    p2d_run_id: str | None = None
    p2d_transport_arm: str | None = None
    p2d_offer_generation: int | None = None
    p2d_iteration: int | None = None


@dataclass
class ReqMeta:
    local_block_ids: BlockIds
    # To be used when logical block size does not match the kernel block size
    local_physical_block_ids: BlockIds
    tp_size: int
    remote: RemoteMeta | None = None
    # Remote block size, discovered during NIXL handshake (push mode).
    remote_block_size: int | None = None


class NixlConnectorMetadata(KVConnectorMetadata):
    def __init__(self):
        self.reqs_to_recv: dict[ReqId, ReqMeta] = {}
        self.reqs_to_save: dict[ReqId, ReqMeta] = {}
        self.reqs_to_send: dict[ReqId, ProducerLease] = {}
        self.offer_cancellations_by_rank: dict[int, tuple[PullOfferCancelled, ...]] = {}
        # P-side block rosters retained until completed remote reads are observed.
        self.source_rosters: dict[ReqId, NixlSourceRoster] = {}
        # Requests that will execute a model forward with this metadata. This
        # is distinct from producer-side lease tracking in reqs_in_batch.
        self.scheduled_request_ids: set[ReqId] = set()
        self.reqs_in_batch: set[ReqId] = set()
        self.reqs_not_processed: set[ReqId] = set()
        # A complete replacement of the D worker's heartbeat targets. None means
        # the scheduler-side ownership state has not changed on this step.
        self.heartbeat_snapshot: dict[EngineId, HeartbeatInfo] | None = None
        # Push mode (D side): registration data the D worker should send to
        # P workers via NIXL notification on this step.
        self.push_registrations: dict[ReqId, dict[str, Any]] = {}
        # Push mode (P side): newly finished request blocks to be matched
        # against pending D registrations on the P worker.
        self.push_finished_blocks: dict[ReqId, BlockIds] = {}
        # KV-audit: consumer request ids that finished on the scheduler
        # this step (any finish status). The worker retires audit state
        # for these; unknown ids are ignored.
        self.audit_finished: set[ReqId] = set()

    def _add_new_req(
        self,
        local_block_ids: BlockIds,
        kv_transfer_params: dict[str, Any],
    ) -> ReqMeta:
        return ReqMeta(
            local_block_ids=local_block_ids,
            local_physical_block_ids=local_block_ids,
            # P workers don't need to receive tp_size from proxy here.
            tp_size=kv_transfer_params.get("tp_size", 1),
            remote_block_size=kv_transfer_params.get("remote_block_size"),
        )

    def add_new_req_to_save(
        self,
        request_id: ReqId,
        local_block_ids: BlockIds,
        kv_transfer_params: dict[str, Any],
    ):
        self.reqs_to_save[request_id] = self._add_new_req(
            local_block_ids, kv_transfer_params
        )

    def add_new_req_to_recv(
        self,
        request_id: ReqId,
        local_block_ids: BlockIds,
        kv_transfer_params: dict[str, Any],
    ):
        expected_consumers = kv_transfer_params.get("expected_consumers", 1)
        consumer_tp_size = kv_transfer_params.get("consumer_tp_size", 1)
        if type(expected_consumers) is not int or expected_consumers < 1:
            raise ValueError("expected_consumers must be a positive integer")
        if type(consumer_tp_size) is not int or consumer_tp_size < 1:
            raise ValueError("consumer_tp_size must be a positive integer")
        req = self._add_new_req(local_block_ids, kv_transfer_params)
        req.remote = RemoteMeta(
            block_ids=kv_transfer_params["remote_block_ids"],
            engine_id=kv_transfer_params["remote_engine_id"],
            request_id=kv_transfer_params["remote_request_id"],
            remote_num_tokens=int(kv_transfer_params["remote_num_tokens"]),
            host=kv_transfer_params["remote_host"],
            port=kv_transfer_params["remote_port"],
            expected_consumers=expected_consumers,
            consumer_tp_size=consumer_tp_size,
            p2d_run_id=kv_transfer_params.get("p2d_run_id"),
            p2d_transport_arm=kv_transfer_params.get("p2d_transport_arm"),
            p2d_offer_generation=kv_transfer_params.get("p2d_offer_generation"),
            p2d_iteration=kv_transfer_params.get("p2d_iteration"),
        )
        self.reqs_to_recv[request_id] = req
