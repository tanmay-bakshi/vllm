# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Metadata dataclasses and helpers for the NIXL connector."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import msgspec

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.coalesced_layout import rank_major_slot_base
from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds, EngineId
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorHandshakeMetadata,
    KVConnectorMetadata,
)
from vllm.distributed.kv_transfer.nixl_contracts import (
    NixlRegionDescriptor,
    NixlSourceRoster,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

TransferHandle = int
ReqId = str
RemoteOfferKey = tuple[EngineId, ReqId, int | None]

GET_META_MSG = b"get_meta_msg"

# Push-mode (WRITE-based) registration notification.
# Sent worker-to-worker over NIXL: D worker -> P worker, encoded as
# PUSH_REG_NOTIF_PREFIX + msgpack(registration_data).
PUSH_REG_NOTIF_PREFIX = b"PUSH_REG:"
PULL_READ_COMPLETE_PREFIX = b"PULL_READ_COMPLETE:"
PULL_OFFER_CANCELLATION_CONTROL_PREFIX = b"PULL_OFFER_CANCELLATION:"
PACKED_WRITE_REQUEST_PREFIX = b"PACKED_WRITE_REQUEST:"
PACKED_WRITE_ARRIVED_PREFIX = b"PACKED_WRITE_ARRIVED:"
PACKED_WRITE_FAILED_BEFORE_WRITE_PREFIX = b"PACKED_WRITE_FAILED_BEFORE_WRITE:"
PACKED_WRITE_SOURCE_DRAINED_PREFIX = b"PACKED_WRITE_SOURCE_DRAINED:"
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
#   9: Require producer-rank, KV-region, and cache-group handshake contracts
#   10: Add bounded producer-packed pull pools and chunk ownership messages
#   11: Replace packed READ with producer-initiated packed WRITE
#       and authenticated decoder source-drain proofs; generation-scope every
#       terminal and publish the producer-committed source-retirement floor
#
NIXL_CONNECTOR_VERSION: int = 11


def _require_packed_int(name: str, value: int, *, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(
            f"{name} must be an integer greater than or equal to {minimum}"
        )


def _require_packed_text(name: str, value: str) -> None:
    if type(value) is not str or len(value) == 0:
        raise ValueError(f"{name} must be a non-empty string")


def _require_packed_digest(name: str, value: str) -> None:
    if type(value) is not str or len(value) != 64:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _validate_packed_write_pool_geometry(
    *,
    label: str,
    registration_generation: str,
    base_address: int,
    registered_bytes: int,
    slot_size_bytes: int,
    slot_count: int,
    device_id: int,
    alignment_bytes: int,
) -> None:
    """Validate geometry shared by producer and consumer pools."""
    _require_packed_text(
        f"{label} registration_generation",
        registration_generation,
    )
    _require_packed_int(f"{label} base_address", base_address, minimum=1)
    _require_packed_int(f"{label} registered_bytes", registered_bytes, minimum=1)
    _require_packed_int(f"{label} slot_size_bytes", slot_size_bytes, minimum=1)
    _require_packed_int(f"{label} slot_count", slot_count, minimum=1)
    _require_packed_int(f"{label} device_id", device_id)
    _require_packed_int(f"{label} alignment_bytes", alignment_bytes, minimum=1)
    if alignment_bytes & (alignment_bytes - 1) != 0:
        raise ValueError(f"{label} alignment_bytes must be a power of two")
    if base_address % alignment_bytes != 0:
        raise ValueError(f"{label} base_address is not aligned")
    if slot_size_bytes % alignment_bytes != 0:
        raise ValueError(f"{label} slot_size_bytes is not aligned")
    if registered_bytes != slot_size_bytes * slot_count:
        raise ValueError(f"{label} registered_bytes must exactly cover every slot")
    registration_end = base_address + registered_bytes
    if registration_end <= base_address or registration_end > (1 << 64):
        raise ValueError(f"{label} registration exceeds the uint64 address space")


class PackedWriteProducerPoolGeometry(msgspec.Struct, frozen=True, array_like=True):
    """One producer rank's registered flat gather pool."""

    registration_generation: str
    base_address: int
    registered_bytes: int
    slot_size_bytes: int
    slot_count: int
    device_id: int
    alignment_bytes: int

    def __post_init__(self) -> None:
        _validate_packed_write_pool_geometry(
            label="packed WRITE producer pool",
            registration_generation=self.registration_generation,
            base_address=self.base_address,
            registered_bytes=self.registered_bytes,
            slot_size_bytes=self.slot_size_bytes,
            slot_count=self.slot_count,
            device_id=self.device_id,
            alignment_bytes=self.alignment_bytes,
        )


class PackedWriteConsumerPoolGeometry(msgspec.Struct, frozen=True, array_like=True):
    """One decoder rank's registered rank-major receive pool."""

    registration_generation: str
    base_address: int
    registered_bytes: int
    slot_size_bytes: int
    slot_count: int
    source_tp_size: int
    rank_stride_bytes: int
    device_id: int
    alignment_bytes: int

    def __post_init__(self) -> None:
        _validate_packed_write_pool_geometry(
            label="packed WRITE consumer pool",
            registration_generation=self.registration_generation,
            base_address=self.base_address,
            registered_bytes=self.registered_bytes,
            slot_size_bytes=self.slot_size_bytes,
            slot_count=self.slot_count,
            device_id=self.device_id,
            alignment_bytes=self.alignment_bytes,
        )
        _require_packed_int(
            "packed WRITE consumer pool source_tp_size",
            self.source_tp_size,
            minimum=1,
        )
        _require_packed_int(
            "packed WRITE consumer pool rank_stride_bytes",
            self.rank_stride_bytes,
            minimum=1,
        )
        if self.rank_stride_bytes % self.alignment_bytes != 0:
            raise ValueError(
                "packed WRITE consumer pool rank_stride_bytes is not aligned"
            )
        expected_slot_size = self.source_tp_size * self.rank_stride_bytes
        if self.slot_size_bytes != expected_slot_size:
            raise ValueError(
                "packed WRITE consumer pool slot_size_bytes must exactly cover "
                "every source-rank slab"
            )


class PackedWriteSourceSelection(msgspec.Struct, frozen=True, array_like=True):
    """One exact group-absolute source interval."""

    group_index: int
    source_position_start: int
    position_count: int

    def __post_init__(self) -> None:
        _require_packed_int("source selection group_index", self.group_index)
        _require_packed_int(
            "source selection source_position_start",
            self.source_position_start,
        )
        _require_packed_int("source selection position_count", self.position_count)


class PackedWriteConsumerEndpoint(msgspec.Struct, frozen=True, array_like=True):
    """Authenticated decoder agent and registered WRITE destination pool."""

    consumer_engine_id: str
    consumer_rank: int
    consumer_tp_size: int
    registration_generation: str
    agent_metadata: bytes
    consumer_pool: PackedWriteConsumerPoolGeometry

    def __post_init__(self) -> None:
        _require_packed_text("reply consumer_engine_id", self.consumer_engine_id)
        _require_packed_int("reply consumer_rank", self.consumer_rank)
        _require_packed_int("reply consumer_tp_size", self.consumer_tp_size, minimum=1)
        if self.consumer_rank >= self.consumer_tp_size:
            raise ValueError("reply consumer_rank is outside consumer_tp_size")
        _require_packed_text(
            "reply registration_generation",
            self.registration_generation,
        )
        if type(self.agent_metadata) is not bytes or len(self.agent_metadata) == 0:
            raise ValueError("consumer endpoint agent_metadata must be non-empty bytes")
        if type(self.consumer_pool) is not PackedWriteConsumerPoolGeometry:
            raise ValueError("consumer endpoint pool must be typed")
        if self.consumer_pool.registration_generation != self.registration_generation:
            raise ValueError(
                "consumer endpoint and pool registration generations differ"
            )


class PackedWriteChunkIdentity(msgspec.Struct, frozen=True, array_like=True):
    """Immutable identity shared by every message for one packed chunk."""

    producer_engine_id: str
    producer_request_id: ReqId
    offer_generation: int
    producer_rank: int
    producer_tp_size: int
    consumer_engine_id: str
    consumer_request_id: ReqId
    consumer_rank: int
    consumer_tp_size: int
    source_plan_digest: str
    packed_plan_digest: str
    chunk_plan_digest: str
    chunk_ordinal: int
    chunk_count: int
    valid_token_extent: int
    exact_bytes: int

    def __post_init__(self) -> None:
        _require_packed_text("producer_engine_id", self.producer_engine_id)
        _require_packed_text("producer_request_id", self.producer_request_id)
        _require_packed_int("offer_generation", self.offer_generation, minimum=1)
        _require_packed_int("producer_rank", self.producer_rank)
        _require_packed_int("producer_tp_size", self.producer_tp_size, minimum=1)
        if self.producer_rank >= self.producer_tp_size:
            raise ValueError("producer_rank is outside producer_tp_size")
        _require_packed_text("consumer_engine_id", self.consumer_engine_id)
        _require_packed_text("consumer_request_id", self.consumer_request_id)
        _require_packed_int("consumer_rank", self.consumer_rank)
        _require_packed_int("consumer_tp_size", self.consumer_tp_size, minimum=1)
        if self.consumer_rank >= self.consumer_tp_size:
            raise ValueError("consumer_rank is outside consumer_tp_size")
        _require_packed_digest("source_plan_digest", self.source_plan_digest)
        _require_packed_digest("packed_plan_digest", self.packed_plan_digest)
        _require_packed_digest("chunk_plan_digest", self.chunk_plan_digest)
        _require_packed_int("chunk_ordinal", self.chunk_ordinal)
        _require_packed_int("chunk_count", self.chunk_count, minimum=1)
        if self.chunk_ordinal >= self.chunk_count:
            raise ValueError("chunk_ordinal is outside chunk_count")
        _require_packed_int("valid_token_extent", self.valid_token_extent, minimum=1)
        _require_packed_int("exact_bytes", self.exact_bytes, minimum=1)


class PackedWriteSlotBinding(msgspec.Struct, frozen=True, array_like=True):
    """Generation-scoped ownership of one decoder receive slot."""

    pool_registration_generation: str
    slot_index: int
    slot_generation: int
    payload_bytes: int

    def __post_init__(self) -> None:
        _require_packed_text(
            "pool_registration_generation",
            self.pool_registration_generation,
        )
        _require_packed_int("slot_index", self.slot_index)
        _require_packed_int("slot_generation", self.slot_generation, minimum=1)
        _require_packed_int("payload_bytes", self.payload_bytes, minimum=1)


class PackedWriteCommand(msgspec.Struct, frozen=True, array_like=True):
    """Exact rankful chunk and destination generation for one WRITE."""

    chunk: PackedWriteChunkIdentity
    destination: PackedWriteSlotBinding

    def __post_init__(self) -> None:
        if type(self.chunk) is not PackedWriteChunkIdentity:
            raise ValueError("packed WRITE command chunk must be typed")
        if type(self.destination) is not PackedWriteSlotBinding:
            raise ValueError("packed WRITE command destination must be typed")
        if self.destination.payload_bytes != self.chunk.exact_bytes:
            raise ValueError(
                "packed WRITE destination byte count does not match its chunk"
            )


class PackedWriteRequest(msgspec.Struct, frozen=True, array_like=True):
    """Decoder command for one producer-packed WRITE into an exact slot."""

    command: PackedWriteCommand
    consumer_endpoint: PackedWriteConsumerEndpoint
    source_selections: tuple[PackedWriteSourceSelection, ...]

    def __post_init__(self) -> None:
        if type(self.command) is not PackedWriteCommand:
            raise ValueError("packed WRITE request command must be typed")
        if type(self.consumer_endpoint) is not PackedWriteConsumerEndpoint:
            raise ValueError("packed WRITE consumer endpoint must be typed")
        chunk = self.command.chunk
        endpoint = self.consumer_endpoint
        if (
            endpoint.consumer_engine_id != chunk.consumer_engine_id
            or endpoint.consumer_rank != chunk.consumer_rank
            or endpoint.consumer_tp_size != chunk.consumer_tp_size
        ):
            raise ValueError("packed WRITE consumer endpoint does not match its chunk")
        packed_write_destination_address(self.command, endpoint.consumer_pool)
        if type(self.source_selections) is not tuple:
            raise ValueError("packed WRITE source selections must be a tuple")
        if len(self.source_selections) == 0:
            raise ValueError("packed WRITE source selections must not be empty")
        for expected_group_index, selection in enumerate(self.source_selections):
            if type(selection) is not PackedWriteSourceSelection:
                raise ValueError("packed WRITE source selection must be typed")
            if selection.group_index != expected_group_index:
                raise ValueError(
                    "packed WRITE source selections must cover canonical groups "
                    "exactly once"
                )
        if all(selection.position_count == 0 for selection in self.source_selections):
            raise ValueError("packed WRITE source selections contain no positions")


class PackedWriteArrived(msgspec.Struct, frozen=True, array_like=True):
    """NIXL-attached proof that one WRITE is visible in decoder staging."""

    command: PackedWriteCommand

    def __post_init__(self) -> None:
        if type(self.command) is not PackedWriteCommand:
            raise ValueError("packed WRITE arrival command must be typed")


class PackedWriteFailureCode(StrEnum):
    """Producer failures that prove an exact WRITE was never posted."""

    POOL_TIMEOUT = "pool_timeout"
    INVALID_REQUEST = "invalid_request"
    SOURCE_UNAVAILABLE = "source_unavailable"
    PACK_FAILED = "pack_failed"
    ENDPOINT_UNAVAILABLE = "endpoint_unavailable"
    WRITE_PREPARE_FAILED = "write_prepare_failed"
    INTERNAL_ERROR = "internal_error"


class PackedWriteFailedBeforeWrite(msgspec.Struct, frozen=True, array_like=True):
    """Producer proof that one exact command will never post a WRITE."""

    command: PackedWriteCommand
    code: PackedWriteFailureCode
    reason: str

    def __post_init__(self) -> None:
        if type(self.command) is not PackedWriteCommand:
            raise ValueError("packed WRITE failure command must be typed")
        if type(self.code) is not PackedWriteFailureCode:
            raise ValueError("packed WRITE failure code must be typed")
        _require_packed_text("packed WRITE failure reason", self.reason)
        if len(self.reason) > 1024:
            raise ValueError("packed WRITE failure reason exceeds 1024 characters")


class PackedWriteSourceDrained(msgspec.Struct, frozen=True, array_like=True):
    """Decoder proof that every published chunk for one request is terminal."""

    anchor: PackedWriteCommand
    published_chunk_count: int

    def __post_init__(self) -> None:
        if type(self.anchor) is not PackedWriteCommand:
            raise ValueError("packed WRITE source-drain anchor must be typed")
        _require_packed_int(
            "packed WRITE published_chunk_count",
            self.published_chunk_count,
            minimum=1,
        )
        if self.anchor.chunk.chunk_ordinal != 0:
            raise ValueError("packed WRITE source-drain anchor must be chunk zero")
        if self.published_chunk_count > self.anchor.chunk.chunk_count:
            raise ValueError(
                "packed WRITE published chunk count exceeds the canonical plan"
            )


def validate_packed_write_slot_binding(
    slot: PackedWriteSlotBinding,
    pool: PackedWriteConsumerPoolGeometry,
) -> None:
    """Validate one destination lease against its endpoint's pool geometry."""
    if type(slot) is not PackedWriteSlotBinding:
        raise ValueError("packed WRITE destination slot must be typed")
    if type(pool) is not PackedWriteConsumerPoolGeometry:
        raise ValueError("packed WRITE destination pool must be typed")
    if slot.pool_registration_generation != pool.registration_generation:
        raise ValueError("packed WRITE slot names a stale pool generation")
    if slot.slot_index >= pool.slot_count:
        raise ValueError("packed WRITE slot index is outside the pool")
    if slot.payload_bytes > pool.rank_stride_bytes:
        raise ValueError("packed WRITE payload exceeds its source-rank slab")


def packed_write_destination_address(
    command: PackedWriteCommand,
    consumer_pool: PackedWriteConsumerPoolGeometry,
) -> int:
    """Resolve one authenticated rank-major WRITE destination address.

    :param command: Exact chunk and destination-slot generation.
    :param consumer_pool: Decoder receive-pool geometry from its endpoint.
    :returns: Base address of this producer rank's destination slab.
    """
    if type(command) is not PackedWriteCommand:
        raise ValueError("packed WRITE command must be typed")
    if type(consumer_pool) is not PackedWriteConsumerPoolGeometry:
        raise ValueError("packed WRITE consumer pool must be typed")
    if command.chunk.producer_tp_size != consumer_pool.source_tp_size:
        raise ValueError(
            "packed WRITE source TP size does not match its destination pool"
        )
    validate_packed_write_slot_binding(command.destination, consumer_pool)
    slot_base = (
        consumer_pool.base_address
        + command.destination.slot_index * consumer_pool.slot_size_bytes
    )
    return rank_major_slot_base(
        slot_base,
        command.chunk.producer_rank,
        consumer_pool.rank_stride_bytes,
    )


def validate_packed_write_request_replay(
    expected: PackedWriteRequest,
    replay: PackedWriteRequest,
) -> None:
    """Require an exact immutable replay of one previously accepted command."""
    if type(expected) is not PackedWriteRequest:
        raise ValueError("expected packed WRITE request must be typed")
    if type(replay) is not PackedWriteRequest:
        raise ValueError("replayed packed WRITE request must be typed")
    if expected != replay:
        raise ValueError("packed WRITE request replay changed authenticated fields")


def validate_packed_write_arrived(
    request: PackedWriteRequest,
    arrived: PackedWriteArrived,
) -> None:
    """Validate a native-attached arrival against its exact request."""
    if type(request) is not PackedWriteRequest:
        raise ValueError("packed WRITE arrival request must be typed")
    if type(arrived) is not PackedWriteArrived:
        raise ValueError("packed WRITE arrival must be typed")
    if arrived.command != request.command:
        raise ValueError("packed WRITE arrival does not match its request")


def validate_packed_write_failed_before_write(
    request: PackedWriteRequest,
    failed: PackedWriteFailedBeforeWrite,
) -> None:
    """Validate a no-WRITE failure proof against its exact request."""
    if type(request) is not PackedWriteRequest:
        raise ValueError("packed WRITE failure request must be typed")
    if type(failed) is not PackedWriteFailedBeforeWrite:
        raise ValueError("packed WRITE pre-post failure must be typed")
    if failed.command != request.command:
        raise ValueError("packed WRITE failure does not match its request")


def encode_packed_write_request_notification(request: PackedWriteRequest) -> bytes:
    """Encode one typed decoder packed-WRITE request."""
    if type(request) is not PackedWriteRequest:
        raise ValueError("packed WRITE request must be typed")
    return PACKED_WRITE_REQUEST_PREFIX + msgspec.msgpack.encode(request)


def decode_packed_write_request_notification(notification: bytes) -> PackedWriteRequest:
    """Decode one typed decoder packed-WRITE request."""
    if type(notification) is not bytes or not notification.startswith(
        PACKED_WRITE_REQUEST_PREFIX
    ):
        raise ValueError("notification is not a packed WRITE request")
    return msgspec.msgpack.decode(
        notification[len(PACKED_WRITE_REQUEST_PREFIX) :],
        type=PackedWriteRequest,
    )


def encode_packed_write_arrived_notification(arrived: PackedWriteArrived) -> bytes:
    """Encode one native-attached packed-WRITE arrival proof."""
    if type(arrived) is not PackedWriteArrived:
        raise ValueError("packed WRITE arrival must be typed")
    return PACKED_WRITE_ARRIVED_PREFIX + msgspec.msgpack.encode(arrived)


def decode_packed_write_arrived_notification(notification: bytes) -> PackedWriteArrived:
    """Decode one native-attached packed-WRITE arrival proof."""
    if type(notification) is not bytes or not notification.startswith(
        PACKED_WRITE_ARRIVED_PREFIX
    ):
        raise ValueError("notification is not a packed WRITE arrival")
    return msgspec.msgpack.decode(
        notification[len(PACKED_WRITE_ARRIVED_PREFIX) :],
        type=PackedWriteArrived,
    )


def encode_packed_write_failed_before_write_notification(
    failed: PackedWriteFailedBeforeWrite,
) -> bytes:
    """Encode one producer proof that an exact WRITE was never posted."""
    if type(failed) is not PackedWriteFailedBeforeWrite:
        raise ValueError("packed WRITE pre-post failure must be typed")
    return PACKED_WRITE_FAILED_BEFORE_WRITE_PREFIX + msgspec.msgpack.encode(failed)


def decode_packed_write_failed_before_write_notification(
    notification: bytes,
) -> PackedWriteFailedBeforeWrite:
    """Decode one producer proof that an exact WRITE was never posted."""
    if type(notification) is not bytes or not notification.startswith(
        PACKED_WRITE_FAILED_BEFORE_WRITE_PREFIX
    ):
        raise ValueError("notification is not a packed WRITE pre-post failure")
    return msgspec.msgpack.decode(
        notification[len(PACKED_WRITE_FAILED_BEFORE_WRITE_PREFIX) :],
        type=PackedWriteFailedBeforeWrite,
    )


def encode_packed_write_source_drained_notification(
    drained: PackedWriteSourceDrained,
) -> bytes:
    """Encode one decoder proof that all of its published chunks drained."""
    if type(drained) is not PackedWriteSourceDrained:
        raise ValueError("packed WRITE source-drain proof must be typed")
    return PACKED_WRITE_SOURCE_DRAINED_PREFIX + msgspec.msgpack.encode(drained)


def decode_packed_write_source_drained_notification(
    notification: bytes,
) -> PackedWriteSourceDrained:
    """Decode one packed-WRITE source-drain proof."""
    if type(notification) is not bytes or not notification.startswith(
        PACKED_WRITE_SOURCE_DRAINED_PREFIX
    ):
        raise ValueError("notification is not a packed WRITE source-drain proof")
    return msgspec.msgpack.decode(
        notification[len(PACKED_WRITE_SOURCE_DRAINED_PREFIX) :],
        type=PackedWriteSourceDrained,
    )


@dataclass
class NixlAgentMetadata:
    engine_id: str
    tp_rank: int
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
    packed_write_producer_pool: PackedWriteProducerPoolGeometry | None = None
    packed_write_consumer_pool: PackedWriteConsumerPoolGeometry | None = None


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
    :ivar offer_generation: Exact producer allocation generation.
    :ivar consumer_request_id: Concrete decoder request for diagnostics.
    :ivar consumer_index: Stable parallel-sampling child index.
    :ivar consumer_rank: Decoder tensor-parallel rank that completed the read.
    :ivar consumer_tp_size: Decoder tensor-parallel world size.
    :ivar expected_consumers: Decoder view of the producer's consumer contract.
    """

    producer_request_id: ReqId
    offer_generation: int
    consumer_request_id: ReqId
    consumer_index: int
    consumer_rank: int
    consumer_tp_size: int
    expected_consumers: int

    def __post_init__(self) -> None:
        _require_packed_text("completion producer_request_id", self.producer_request_id)
        _require_packed_int(
            "completion offer_generation", self.offer_generation, minimum=1
        )


class PullOfferCancelled(msgspec.Struct, frozen=True, array_like=True):
    """Proof that one decoder rank admitted no consumer from an offer.

    :ivar producer_request_id: Producer request whose pages remain offered.
    :ivar offer_generation: Exact producer allocation generation.
    :ivar consumer_rank: Decoder tensor-parallel rank making the assertion.
    :ivar consumer_tp_size: Decoder tensor-parallel world size.
    :ivar expected_consumers: Decoder view of the producer-owned contract.
    """

    producer_request_id: ReqId
    offer_generation: int
    consumer_rank: int
    consumer_tp_size: int
    expected_consumers: int

    def __post_init__(self) -> None:
        _require_packed_text(
            "offer cancellation producer_request_id",
            self.producer_request_id,
        )
        _require_packed_int(
            "offer cancellation offer_generation",
            self.offer_generation,
            minimum=1,
        )


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
    :ivar offer_generation: Exact producer allocation generation.
    :ivar producer_ranks: Exact producer ranks that will receive the proof.
    :ivar accepted: Whether the complete control request was queued atomically.
    """

    producer_request_id: ReqId
    offer_generation: int
    producer_ranks: tuple[int, ...]
    accepted: bool

    def __post_init__(self) -> None:
        _require_packed_text(
            "offer cancellation acknowledgement producer_request_id",
            self.producer_request_id,
        )
        _require_packed_int(
            "offer cancellation acknowledgement offer_generation",
            self.offer_generation,
            minimum=1,
        )


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
    source_offer_generation: int | None = None
    source_retired_through: int = 0
    p2d_run_id: str | None = None
    p2d_transport_arm: str | None = None
    p2d_iteration: int | None = None

    @property
    def offer_key(self) -> RemoteOfferKey:
        """Return the generation-scoped producer allocation identity."""
        return (
            self.engine_id,
            self.request_id,
            self.source_offer_generation,
        )


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
        # Highest gap-free source generation committed by the P scheduler.
        self.source_retired_through: int = 0
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
        source_offer_generation = kv_transfer_params.get("source_offer_generation")
        source_retired_through = kv_transfer_params.get("source_retired_through", 0)
        if source_offer_generation is not None and (
            type(source_offer_generation) is not int or source_offer_generation < 1
        ):
            raise ValueError("source_offer_generation must be a positive integer")
        if type(source_retired_through) is not int or source_retired_through < 0:
            raise ValueError("source_retired_through must be a non-negative integer")
        if (source_offer_generation is None and source_retired_through != 0) or (
            source_offer_generation is not None
            and source_retired_through >= source_offer_generation
        ):
            raise ValueError("source retirement floor must precede its offer")
        req.remote = RemoteMeta(
            block_ids=kv_transfer_params["remote_block_ids"],
            engine_id=kv_transfer_params["remote_engine_id"],
            request_id=kv_transfer_params["remote_request_id"],
            remote_num_tokens=int(kv_transfer_params["remote_num_tokens"]),
            host=kv_transfer_params["remote_host"],
            port=kv_transfer_params["remote_port"],
            expected_consumers=expected_consumers,
            consumer_tp_size=consumer_tp_size,
            source_offer_generation=source_offer_generation,
            source_retired_through=source_retired_through,
            p2d_run_id=kv_transfer_params.get("p2d_run_id"),
            p2d_transport_arm=kv_transfer_params.get("p2d_transport_arm"),
            p2d_iteration=kv_transfer_params.get("p2d_iteration"),
        )
        self.reqs_to_recv[request_id] = req
