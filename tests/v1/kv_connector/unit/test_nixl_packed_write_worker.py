# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-only integration proofs for producer-initiated packed WRITE workers."""

from collections import deque
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

import msgspec
import pytest

from vllm.distributed.kv_transfer.coalesced_layout import (
    CanonicalSourceTransferPlan,
    CoalescedTransferPlan,
    GroupTransferRoster,
    PackedTransferPlan,
    RegionOwnership,
    SourceGroupTransferRoster,
    SourceRegionOwnership,
    bind_coalesced_transfer_destinations,
    build_canonical_source_plan,
    build_packed_transfer_plan,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    PACKED_WRITE_ARRIVED_PREFIX,
    PACKED_WRITE_FAILED_BEFORE_WRITE_PREFIX,
    PACKED_WRITE_REQUEST_PREFIX,
    PACKED_WRITE_SOURCE_DRAINED_PREFIX,
    PackedWriteArrived,
    PackedWriteChunkIdentity,
    PackedWriteCommand,
    PackedWriteConsumerEndpoint,
    PackedWriteConsumerPoolGeometry,
    PackedWriteFailedBeforeWrite,
    PackedWriteFailureCode,
    PackedWriteProducerPoolGeometry,
    PackedWriteRequest,
    PackedWriteSlotBinding,
    PackedWriteSourceDrained,
    PackedWriteSourceSelection,
    ProducerLease,
    PullReadComplete,
    RemoteMeta,
    ReqMeta,
    decode_packed_write_arrived_notification,
    decode_packed_write_failed_before_write_notification,
    decode_packed_write_request_notification,
    decode_packed_write_source_drained_notification,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.packed_write_config import (
    PackedWriteConfig,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_worker import (
    CompletedPullContract,
    NixlPullConnectorWorker,
    _AuthenticatedPackedSourceDrained,
    _ConsumerPackedTerminal,
    _ProducerPackedOperation,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import ReadSpec
from vllm.distributed.kv_transfer.nixl_contracts import (
    NixlRegionDescriptor,
    NixlSourceRoster,
)
from vllm.distributed.kv_transfer.packed_write_ownership import (
    PackedWriteProducerPool,
)
from vllm.distributed.kv_transfer.staging_ownership import StagingSafetyError
from vllm.v1.outputs import KVTransferFailureReason

_MIB = 1024 * 1024
_CHUNK_BYTES_PER_RANK = 64 * _MIB
_ROW_BYTES = 32 * _MIB
_SOURCE_TP_SIZE = 4
_CONSUMER_SLOT_BYTES = _SOURCE_TP_SIZE * _CHUNK_BYTES_PER_RANK
_SOURCE_RANKS = tuple(range(_SOURCE_TP_SIZE))


@dataclass(slots=True)
class _FakeBuffer:
    """Expose registered-buffer geometry without allocating its payload."""

    address: int
    size_bytes: int

    def data_ptr(self) -> int:
        """Return the synthetic registration base address."""
        return self.address

    def numel(self) -> int:
        """Return the synthetic byte length."""
        return self.size_bytes


@dataclass(slots=True)
class _FakeLaunch:
    """Provide deterministic host-side completion for an asynchronous launch."""

    complete: bool = True
    duration_ms: float = 1.25

    def is_complete(self) -> bool:
        """Return the configured completion state."""
        return self.complete

    def gpu_duration_ms(self) -> float:
        """Return deterministic synthetic GPU duration."""
        if self.complete is False:
            raise RuntimeError("launch is not complete")
        return self.duration_ms


def _config(
    *,
    producer_slot_count: int = 1,
    consumer_slot_count: int = 2,
    warn_after_s: float = 1.0,
    fail_after_s: float = 2.0,
) -> PackedWriteConfig:
    """Build the smallest supported packed-WRITE process contract.

    :param producer_slot_count: Producer gather slot count.
    :param consumer_slot_count: Decoder rank-major slot count.
    :param warn_after_s: Operational warning age.
    :param fail_after_s: Fail-closed watchdog age.
    :returns: Enabled validated configuration.
    """
    return PackedWriteConfig(
        enabled=True,
        chunk_bytes_per_rank=_CHUNK_BYTES_PER_RANK,
        min_descriptors_per_rank=1,
        producer_slot_count=producer_slot_count,
        consumer_slot_count=consumer_slot_count,
        alignment_bytes=256,
        warn_after_s=warn_after_s,
        fail_after_s=fail_after_s,
    )


def _producer_geometry(
    producer_rank: int,
    *,
    slot_count: int = 1,
) -> PackedWriteProducerPoolGeometry:
    """Build one producer-rank gather-pool registration.

    :param producer_rank: Rank whose synthetic address space is represented.
    :param slot_count: Number of reusable producer slots.
    :returns: Valid producer pool geometry.
    """
    return PackedWriteProducerPoolGeometry(
        registration_generation=f"producer-registration-{producer_rank}",
        base_address=0x1_0000_0000 + producer_rank * 0x1000_0000,
        registered_bytes=slot_count * _CHUNK_BYTES_PER_RANK,
        slot_size_bytes=_CHUNK_BYTES_PER_RANK,
        slot_count=slot_count,
        device_id=producer_rank,
        alignment_bytes=256,
    )


def _consumer_geometry(
    *,
    engine_id: str = "decode-a",
    base_address: int = 0x4_0000_0000,
    slot_count: int = 2,
) -> PackedWriteConsumerPoolGeometry:
    """Build one decoder rank-major receive-pool registration.

    :param engine_id: Decoder identity used to scope its generation.
    :param base_address: Synthetic registered base address.
    :param slot_count: Number of reusable decoder slots.
    :returns: Valid consumer pool geometry.
    """
    return PackedWriteConsumerPoolGeometry(
        registration_generation=f"{engine_id}-consumer-registration",
        base_address=base_address,
        registered_bytes=slot_count * _CONSUMER_SLOT_BYTES,
        slot_size_bytes=_CONSUMER_SLOT_BYTES,
        slot_count=slot_count,
        source_tp_size=_SOURCE_TP_SIZE,
        rank_stride_bytes=_CHUNK_BYTES_PER_RANK,
        device_id=5,
        alignment_bytes=256,
    )


def _plans() -> tuple[
    CanonicalSourceTransferPlan,
    CoalescedTransferPlan,
    PackedTransferPlan,
]:
    """Build a full, full, tail chunk sequence over one canonical region.

    :returns: Matching source, destination, and bounded packed plans.
    """
    remote_blocks = (0, 1, 2, 3, 4)
    source_plan = build_canonical_source_plan(
        source_tp_size=_SOURCE_TP_SIZE,
        source_ranks=_SOURCE_RANKS,
        groups=(
            SourceGroupTransferRoster(
                group_index=0,
                source_position_start=0,
                remote_block_ids=remote_blocks,
            ),
        ),
        regions=(
            SourceRegionOwnership(
                region_index=0,
                group_indices=(0,),
                source_row_count=8,
                row_bytes=_ROW_BYTES,
            ),
        ),
    )
    destination_plan = bind_coalesced_transfer_destinations(
        source_plan,
        rank_slots=_SOURCE_RANKS,
        groups=(
            GroupTransferRoster(
                group_index=0,
                source_position_start=0,
                destination_plane_count=2,
                local_block_ids=remote_blocks,
                remote_block_ids=remote_blocks,
            ),
        ),
        regions=(
            RegionOwnership(
                region_index=0,
                group_indices=(0,),
                source_row_count=8,
                destination_row_count=8,
                row_bytes=_ROW_BYTES,
            ),
        ),
    )
    packed_plan = build_packed_transfer_plan(
        source_plan,
        max_chunk_bytes_per_rank=_CHUNK_BYTES_PER_RANK,
    )
    assert tuple(chunk.rank_stride_bytes for chunk in packed_plan.chunks) == (
        _CHUNK_BYTES_PER_RANK,
        _CHUNK_BYTES_PER_RANK,
        _ROW_BYTES,
    )
    return source_plan, destination_plan, packed_plan


def _consumer_endpoint(
    *,
    engine_id: str = "decode-a",
    base_address: int = 0x4_0000_0000,
) -> PackedWriteConsumerEndpoint:
    """Build one exact decoder endpoint.

    :param engine_id: Decoder engine identity.
    :param base_address: Decoder receive-pool base address.
    :returns: Generation-bound endpoint.
    """
    geometry = _consumer_geometry(engine_id=engine_id, base_address=base_address)
    return PackedWriteConsumerEndpoint(
        consumer_engine_id=engine_id,
        consumer_rank=0,
        consumer_tp_size=1,
        registration_generation=geometry.registration_generation,
        agent_metadata=f"{engine_id}-agent-metadata".encode(),
        consumer_pool=geometry,
    )


def _request(
    *,
    producer_rank: int = 0,
    chunk_index: int = 0,
    offer_generation: int = 7,
    consumer_engine_id: str = "decode-a",
    consumer_request_id: str = "decode-request",
    destination: PackedWriteSlotBinding | None = None,
    base_address: int = 0x4_0000_0000,
) -> PackedWriteRequest:
    """Build one self-consistent producer-rank packed-WRITE command.

    :param producer_rank: Addressed producer rank.
    :param chunk_index: Canonical chunk ordinal.
    :param offer_generation: Exact retained producer allocation generation.
    :param consumer_engine_id: Addressed decoder engine.
    :param consumer_request_id: Decoder request identity.
    :param destination: Optional consumer slot binding.
    :param base_address: Decoder receive-pool base address.
    :returns: Valid request matching the canonical test plans.
    """
    source_plan, _, packed_plan = _plans()
    chunk = packed_plan.chunks[chunk_index]
    endpoint = _consumer_endpoint(
        engine_id=consumer_engine_id,
        base_address=base_address,
    )
    binding = destination
    if binding is None:
        binding = PackedWriteSlotBinding(
            pool_registration_generation=(
                endpoint.consumer_pool.registration_generation
            ),
            slot_index=0,
            slot_generation=1,
            payload_bytes=chunk.rank_stride_bytes,
        )
    identity = PackedWriteChunkIdentity(
        producer_engine_id="prefill",
        producer_request_id="prefill-request",
        offer_generation=offer_generation,
        producer_rank=producer_rank,
        producer_tp_size=_SOURCE_TP_SIZE,
        consumer_engine_id=consumer_engine_id,
        consumer_request_id=consumer_request_id,
        consumer_rank=0,
        consumer_tp_size=1,
        source_plan_digest=source_plan.digest,
        packed_plan_digest=packed_plan.digest,
        chunk_plan_digest=chunk.digest,
        chunk_ordinal=chunk_index,
        chunk_count=len(packed_plan.chunks),
        valid_token_extent=80,
        exact_bytes=chunk.rank_stride_bytes,
    )
    return PackedWriteRequest(
        command=PackedWriteCommand(chunk=identity, destination=binding),
        consumer_endpoint=endpoint,
        source_selections=(
            PackedWriteSourceSelection(
                group_index=0,
                source_position_start=0,
                position_count=5,
            ),
        ),
    )


def _producer_operation(
    worker: NixlPullConnectorWorker,
    request: PackedWriteRequest,
) -> _ProducerPackedOperation:
    """Admit one producer operation with an already-complete gather.

    :param worker: Synthetic producer worker.
    :param request: Exact decoder command.
    :returns: Live producer operation.
    """
    pool = worker._packed_producer_pool
    assert pool is not None
    admission = pool.admit(request, "consumer-agent")
    assert admission.lease is not None
    source_plan, _, packed_plan = _plans()
    return _ProducerPackedOperation(
        request=request,
        source_plan=source_plan,
        packed_plan=packed_plan,
        lease=admission.lease,
        pack_launch=cast(Any, _FakeLaunch()),
        consumer_agent="consumer-agent",
        created_at=10.0,
        stage_started_at=10.0,
    )


def _producer_worker(
    request: PackedWriteRequest,
) -> tuple[NixlPullConnectorWorker, _ProducerPackedOperation]:
    """Build a producer worker around one admitted operation.

    :param request: Exact decoder command to own.
    :returns: Synthetic worker and its live operation.
    """
    worker = cast(
        NixlPullConnectorWorker,
        object.__new__(NixlPullConnectorWorker),
    )
    producer_rank = request.command.chunk.producer_rank
    geometry = _producer_geometry(producer_rank)
    worker._packed_write_config = _config()
    worker.kv_transfer_config = SimpleNamespace(kv_role="kv_producer")
    worker.tp_rank = producer_rank
    worker.world_size = _SOURCE_TP_SIZE
    worker.engine_id = "prefill"
    worker.device_id = producer_rank
    worker.nixl_memory_type = "VRAM"
    worker._packed_write_producer_pool_geometry = geometry
    worker._packed_write_producer_pool_buffer = _FakeBuffer(
        geometry.base_address,
        geometry.registered_bytes,
    )
    worker._packed_write_producer_pack_stream = object()
    worker._packed_producer_pool = PackedWriteProducerPool(
        geometry,
        producer_rank=producer_rank,
    )
    worker._packed_producer_pending = deque()
    worker._packed_producer_operations = {}
    worker._packed_completion_bindings = {}
    worker._pull_completion_states = {}
    worker._buffered_packed_source_drained = {}
    worker._buffered_pull_completions = {}
    worker._buffered_offer_cancellations = {}
    worker._reqs_to_process = set()
    worker._reqs_to_send = {}
    worker._completed_pull_contracts = {}
    worker._local_source_retired_through = 0
    worker._packed_source_ready_events = {}
    worker._source_rosters = {}
    worker._packed_consumer_agents = {
        request.consumer_endpoint: "consumer-agent",
    }
    worker._packed_consumer_agent_last_active = {
        request.consumer_endpoint: 0.0,
    }
    worker._engine_ttl = 0
    worker._remote_agents = {}
    worker.nixl_wrapper = MagicMock()
    worker.nixl_wrapper.get_xfer_descs.side_effect = lambda descriptors, _memory_type: (
        tuple(descriptors)
    )
    worker.nixl_wrapper.initialize_xfer.return_value = "write-handle"
    worker.nixl_wrapper.transfer.return_value = "PROC"
    worker.nixl_wrapper.check_xfer_state.return_value = "DONE"
    worker.nixl_wrapper.get_xfer_telemetry.return_value = None
    worker.xfer_stats = MagicMock()
    operation = _producer_operation(worker, request)
    worker._packed_producer_operations[request.command.chunk] = operation
    return worker, operation


def _producer_source_drain_contract(
    *,
    published_chunk_count: int = 2,
    remembered_chunk_count: int | None = None,
) -> tuple[
    NixlPullConnectorWorker,
    tuple[PackedWriteRequest, ...],
    _AuthenticatedPackedSourceDrained,
]:
    """Build one generation-bound producer contract and its drain proof.

    :param published_chunk_count: Published chunk prefix named by the proof.
    :param remembered_chunk_count: Prefix already terminal in producer history.
    :returns: Producer worker, exact commands, and authenticated drain proof.
    """
    if remembered_chunk_count is None:
        remembered_chunk_count = published_chunk_count
    requests = tuple(
        _request(producer_rank=0, chunk_index=chunk_index)
        for chunk_index in range(published_chunk_count)
    )
    worker, _ = _producer_worker(requests[0])
    geometry = worker._packed_write_producer_pool_geometry
    assert geometry is not None
    worker._packed_producer_pool = PackedWriteProducerPool(
        geometry,
        producer_rank=0,
    )
    worker._packed_producer_operations.clear()
    worker._source_rosters = {"prefill-request": _source_roster()}
    worker._reqs_to_process = {"prefill-request"}
    worker._reqs_to_send = {"prefill-request": 100.0}
    worker._localization_capture_source_post = MagicMock()
    worker._install_pull_completion_state(
        "prefill-request",
        ProducerLease(
            deadline=100.0,
            expected_consumers=1,
            consumer_tp_size=1,
        ),
    )
    pool = worker._packed_producer_pool
    assert pool is not None
    for request in requests:
        worker._bind_packed_completion(request, "consumer-agent")
    for request in requests[:remembered_chunk_count]:
        pool.remember_failed_before_admission(
            request,
            "consumer-agent",
            PackedWriteFailedBeforeWrite(
                command=request.command,
                code=PackedWriteFailureCode.PACK_FAILED,
                reason="known terminal before source drain",
            ),
        )
    authenticated = _AuthenticatedPackedSourceDrained(
        drained=PackedWriteSourceDrained(
            anchor=requests[0].command,
            published_chunk_count=published_chunk_count,
        ),
        consumer_agent="consumer-agent",
    )
    return worker, requests, authenticated


def _source_roster() -> NixlSourceRoster:
    """Build the retained source roster matching the canonical test plan."""
    return NixlSourceRoster(
        offer_generation=7,
        iteration=0,
        expected_consumers=1,
        valid_token_extent=80,
        group_token_capacities=(16,),
        block_ids=((0, 1, 2, 3, 4),),
    )


def _source_region() -> NixlRegionDescriptor:
    """Build the retained producer region matching the canonical test plan."""
    return NixlRegionDescriptor(
        semantic_name="region-0",
        group_indices=(0,),
        group_semantic_names=((0, "group-0-region-0"),),
        base_address=0x8_0000_0000,
        registered_bytes=8 * _ROW_BYTES,
        row_bytes=_ROW_BYTES,
        shape=(8, _ROW_BYTES),
        strides=(_ROW_BYTES, 1),
        dtype="torch.uint8",
        element_size_bytes=1,
        layout="HND",
    )


def _consumer_worker(
    *,
    engine_id: str = "decode-a",
    base_address: int = 0x4_0000_0000,
) -> NixlPullConnectorWorker:
    """Build a decoder worker with two bounded rank-major slots.

    :param engine_id: Decoder engine identity.
    :param base_address: Synthetic consumer registration base.
    :returns: Synthetic consumer worker.
    """
    worker = cast(
        NixlPullConnectorWorker,
        object.__new__(NixlPullConnectorWorker),
    )
    geometry = _consumer_geometry(
        engine_id=engine_id,
        base_address=base_address,
    )
    worker._packed_write_config = _config()
    worker.kv_transfer_config = SimpleNamespace(kv_role="kv_consumer")
    worker.engine_id = engine_id
    worker.tp_rank = 0
    worker.world_size = 1
    worker._registration_generation = geometry.registration_generation
    worker._packed_write_consumer_pool_geometry = geometry
    worker._packed_write_consumer_pool_buffer = _FakeBuffer(
        geometry.base_address,
        geometry.registered_bytes,
    )
    worker._packed_write_scatter_stream = object()
    worker._packed_consumer_pool = None
    worker._packed_consumer_requests = {}
    worker._packed_consumer_chunks = {}
    worker._packed_terminal_notifications = deque()
    worker._packed_consumer_terminals = {}
    worker._packed_done_recving = set()
    worker._remote_agents = {
        "prefill": {rank: f"producer-agent-{rank}" for rank in _SOURCE_RANKS}
    }
    worker._released_remote_offers = {}
    worker._remote_offer_completion_counts = {}
    worker._remote_source_retired_through = {}
    worker._remote_source_consumption_proven = {}
    worker._remote_packed_write_producer_pools = {
        "prefill": {rank: _producer_geometry(rank) for rank in _SOURCE_RANKS}
    }
    worker._region_rows = (MagicMock(name="destination-region"),)
    worker._phase_separate_transfer_decode = True
    worker._transfer_phase_plan_count = 0
    worker._transfer_phase_byte_count = 0
    worker.nixl_wrapper = MagicMock()
    worker.nixl_wrapper.get_agent_metadata.return_value = (
        f"{engine_id}-agent-metadata".encode()
    )
    worker.xfer_stats = MagicMock()
    worker._record_failed_receive = MagicMock()
    return worker


def _consumer_inputs(
    *,
    expected_consumers: int = 1,
) -> tuple[
    ReqMeta,
    list[ReadSpec],
    CanonicalSourceTransferPlan,
    CoalescedTransferPlan,
]:
    """Build matching decoder request inputs.

    :param expected_consumers: Producer offer's decoder-child quorum size.
    :returns: Metadata, rank quorum, source plan, and destination plan.
    """
    source_plan, destination_plan, _ = _plans()
    meta = ReqMeta(
        local_block_ids=([0, 1, 2, 3, 4],),
        local_physical_block_ids=([0, 1, 2, 3, 4],),
        tp_size=_SOURCE_TP_SIZE,
        remote=RemoteMeta(
            block_ids=([0, 1, 2, 3, 4],),
            host="127.0.0.1",
            port=1234,
            engine_id="prefill",
            request_id="prefill-request",
            remote_num_tokens=80,
            expected_consumers=expected_consumers,
            source_offer_generation=7,
        ),
    )
    read_specs = [
        ReadSpec(
            remote_rank=rank,
            local_block_ids=([0, 1, 2, 3, 4],),
            remote_block_ids=([0, 1, 2, 3, 4],),
        )
        for rank in _SOURCE_RANKS
    ]
    return meta, read_specs, source_plan, destination_plan


def _start_consumer_request(
    worker: NixlPullConnectorWorker,
    *,
    request_id: str = "decode-request",
    expected_consumers: int = 1,
) -> None:
    """Install the canonical three-chunk decoder request.

    :param worker: Synthetic decoder worker.
    :param request_id: Decoder child identity.
    :param expected_consumers: Producer offer's decoder-child quorum size.
    """
    meta, read_specs, source_plan, destination_plan = _consumer_inputs(
        expected_consumers=expected_consumers
    )
    result = worker._packed_write_request(
        req_id=request_id,
        meta=meta,
        read_specs=read_specs,
        source_plan=source_plan,
        destination_plan=destination_plan,
        logical_bytes=source_plan.transfer_size_bytes,
        direct_descriptor_count=128,
    )
    assert result == "posted"


def _published_requests(worker: NixlPullConnectorWorker) -> list[PackedWriteRequest]:
    """Decode every producer command emitted by a decoder worker.

    :param worker: Synthetic decoder worker.
    :returns: Commands in publication order.
    """
    return [
        decode_packed_write_request_notification(send.kwargs["notif_msg"])
        for send in worker.nixl_wrapper.send_notif.call_args_list
        if send.kwargs["notif_msg"].startswith(PACKED_WRITE_REQUEST_PREFIX)
    ]


def _source_drains(
    worker: NixlPullConnectorWorker,
) -> list[tuple[str, PackedWriteSourceDrained]]:
    """Decode every final packed source-drain proof emitted by a decoder.

    :param worker: Synthetic decoder worker.
    :returns: Native destination agent and decoded proof pairs.
    """
    return [
        (
            send.args[0],
            decode_packed_write_source_drained_notification(send.kwargs["notif_msg"]),
        )
        for send in worker.nixl_wrapper.send_notif.call_args_list
        if send.kwargs["notif_msg"].startswith(PACKED_WRITE_SOURCE_DRAINED_PREFIX)
    ]


def _accept_arrivals(
    worker: NixlPullConnectorWorker,
    requests: tuple[PackedWriteRequest, ...],
) -> None:
    """Queue an authenticated ARRIVED quorum.

    :param worker: Synthetic decoder worker.
    :param requests: Exact published rank commands.
    """
    for request in requests:
        rank = request.command.chunk.producer_rank
        worker._accept_packed_arrived_notification(
            PackedWriteArrived(command=request.command),
            f"producer-agent-{rank}",
        )


@pytest.mark.cpu_test
def test_producer_reconstructs_pending_command_before_launching_exact_pack() -> None:
    """Pending admission derives its gather from retained producer truth."""
    request = _request(producer_rank=1, chunk_index=2)
    worker, _ = _producer_worker(request)
    geometry = worker._packed_write_producer_pool_geometry
    assert geometry is not None
    worker._packed_producer_pool = PackedWriteProducerPool(
        geometry,
        producer_rank=1,
    )
    worker._packed_producer_operations.clear()
    worker._source_rosters = {"prefill-request": _source_roster()}
    source_ready_event = object()
    worker._packed_source_ready_events = {
        "prefill-request": cast(Any, source_ready_event)
    }
    worker._region_descriptors = (_source_region(),)
    worker._region_rows = (MagicMock(name="source-region"),)
    pack_launch = _FakeLaunch(complete=False)

    worker._accept_packed_request_notification(request, "consumer-agent")
    with patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_worker."
        "launch_packed_chunk_pack",
        return_value=pack_launch,
    ) as launch_pack:
        worker._service_packed_producer()

    assert len(worker._packed_producer_pending) == 0
    operation = worker._packed_producer_operations[request.command.chunk]
    assert operation.request == request
    assert operation.pack_launch is pack_launch
    assert operation.lease.payload_bytes == _ROW_BYTES
    assert operation.source_plan.digest == request.command.chunk.source_plan_digest
    assert operation.packed_plan.digest == request.command.chunk.packed_plan_digest
    launch_pack.assert_called_once()
    assert launch_pack.call_args.args[4] == 2
    assert launch_pack.call_args.kwargs == {
        "destination_base_offset_bytes": 0,
        "source_ready_event": source_ready_event,
    }
    worker.nixl_wrapper.initialize_xfer.assert_not_called()


@pytest.mark.cpu_test
def test_producer_floor_rejects_retired_command_before_binding_or_admission() -> None:
    """Producer ingress ignores retired generations but admits the next one."""
    retired = _request(offer_generation=7)
    worker, _ = _producer_worker(retired)
    geometry = worker._packed_write_producer_pool_geometry
    assert geometry is not None
    worker._packed_producer_pool = PackedWriteProducerPool(
        geometry,
        producer_rank=0,
    )
    worker._packed_producer_operations.clear()
    worker._local_source_retired_through = 7

    worker._accept_packed_request_notification(retired, "consumer-agent")

    assert len(worker._packed_producer_pending) == 0
    assert worker._packed_completion_bindings == {}
    assert worker._packed_producer_pool.free_slot_count == 1
    worker.nixl_wrapper.send_notif.assert_not_called()

    current = _request(offer_generation=8)
    worker._accept_packed_request_notification(current, "consumer-agent")

    assert tuple(pending.request for pending in worker._packed_producer_pending) == (
        current,
    )
    assert len(worker._packed_completion_bindings) == 1
    assert worker._packed_producer_pool.free_slot_count == 1


@pytest.mark.cpu_test
def test_producer_releases_native_handle_before_slot_and_replays_arrived() -> None:
    """DONE, handle release, slot release, and replay remain strictly ordered."""
    request = _request(producer_rank=2)
    worker, operation = _producer_worker(request)
    pool = worker._packed_producer_pool
    assert pool is not None
    events: list[str] = []
    original_mark_released = pool.mark_write_released
    original_complete_arrived = pool.complete_arrived

    def release_handle(handle: object) -> None:
        """Record native handle release while the source slot remains owned."""
        assert handle == "write-handle"
        assert pool.free_slot_count == 0
        events.append("native-release")

    def mark_released(lease: object) -> None:
        """Require native release before crossing the ownership boundary."""
        assert events == ["native-release"]
        original_mark_released(cast(Any, lease))
        events.append("write-released")

    def complete_arrived(lease: object) -> PackedWriteArrived:
        """Require both release boundaries before freeing the gather slot."""
        assert events == ["native-release", "write-released"]
        arrived = original_complete_arrived(cast(Any, lease))
        events.append("slot-free")
        return arrived

    worker.nixl_wrapper.release_xfer_handle.side_effect = release_handle
    with (
        patch.object(pool, "mark_write_released", side_effect=mark_released),
        patch.object(pool, "complete_arrived", side_effect=complete_arrived),
    ):
        worker._service_packed_producer()

    assert events == ["native-release", "write-released", "slot-free"]
    assert pool.free_slot_count == 1
    assert len(worker._packed_producer_operations) == 0
    initialize = worker.nixl_wrapper.initialize_xfer.call_args
    assert initialize.args[0] == "WRITE"
    assert initialize.args[3] == "consumer-agent"
    attached = decode_packed_write_arrived_notification(initialize.args[4])
    assert attached == PackedWriteArrived(command=request.command)
    expected_destination = (
        request.consumer_endpoint.consumer_pool.base_address
        + request.command.destination.slot_index
        * request.consumer_endpoint.consumer_pool.slot_size_bytes
        + request.command.chunk.producer_rank
        * request.consumer_endpoint.consumer_pool.rank_stride_bytes
    )
    assert initialize.args[2] == (
        (
            expected_destination,
            request.command.chunk.exact_bytes,
            request.consumer_endpoint.consumer_pool.device_id,
        ),
    )
    worker.nixl_wrapper.send_notif.assert_not_called()

    worker._source_rosters.clear()
    worker._packed_source_ready_events.clear()
    worker._accept_packed_request_notification(request, "consumer-agent")
    worker._service_packed_producer()

    worker.nixl_wrapper.send_notif.assert_called_once()
    replay = worker.nixl_wrapper.send_notif.call_args.kwargs["notif_msg"]
    assert replay.startswith(PACKED_WRITE_ARRIVED_PREFIX)
    assert decode_packed_write_arrived_notification(replay) == attached
    assert operation.write_status == "DONE"


@pytest.mark.cpu_test
def test_producer_ingress_validates_active_replays_before_reconstruction() -> None:
    """An active identity can neither queue twice nor emit a false no-WRITE proof."""
    request = _request(producer_rank=1)
    worker, _ = _producer_worker(request)

    worker._accept_packed_request_notification(request, "consumer-agent")
    assert len(worker._packed_producer_pending) == 0

    changed = msgspec.structs.replace(
        request,
        source_selections=(
            PackedWriteSourceSelection(
                group_index=0,
                source_position_start=1,
                position_count=4,
            ),
        ),
    )
    with pytest.raises(StagingSafetyError, match="authenticated fields"):
        worker._accept_packed_request_notification(changed, "consumer-agent")

    assert len(worker._packed_producer_pending) == 0
    worker.nixl_wrapper.send_notif.assert_not_called()


@pytest.mark.cpu_test
def test_producer_rejects_commands_after_source_contract_retirement() -> None:
    """A forgotten terminal cannot be replaced by a contradictory late failure."""
    request = _request(producer_rank=1)
    worker, operation = _producer_worker(request)
    pool = worker._packed_producer_pool
    assert pool is not None
    pool.fail_before_write(
        operation.lease,
        PackedWriteFailedBeforeWrite(
            command=request.command,
            code=PackedWriteFailureCode.PACK_FAILED,
            reason="known pre-WRITE failure",
        ),
    )
    worker._packed_producer_operations.clear()
    assert pool.retire_offer("prefill-request", 7) == 1
    worker._completed_pull_contracts[("prefill-request", 7)] = CompletedPullContract(
        expected_consumers=1,
        consumer_tp_size=1,
        offer_generation=7,
        terminal_mode="read_complete",
    )

    with pytest.raises(StagingSafetyError, match="released source contract"):
        worker._accept_packed_request_notification(request, "consumer-agent")

    assert len(worker._packed_producer_pending) == 0
    worker.nixl_wrapper.send_notif.assert_not_called()


@pytest.mark.cpu_test
def test_producer_source_drain_releases_exact_contract_and_replays() -> None:
    """One exact terminal prefix releases once and remains replay-idempotent."""
    worker, requests, authenticated = _producer_source_drain_contract()

    assert (
        worker._record_packed_source_drained(authenticated, allow_buffer=False)
        == "completed"
    )
    assert "prefill-request" not in worker._pull_completion_states
    assert "prefill-request" not in worker._reqs_to_process
    assert "prefill-request" not in worker._source_rosters
    assert worker._packed_completion_bindings == {}
    completed = worker._completed_pull_contracts[("prefill-request", 7)]
    assert completed.offer_generation == 7
    assert completed.packed_completions == frozenset({authenticated})
    worker._localization_capture_source_post.assert_called_once_with("prefill-request")
    pool = worker._packed_producer_pool
    assert pool is not None
    assert pool.terminal_for_replay(requests[0], "consumer-agent") is None

    worker._accept_packed_request_notification(requests[0], "consumer-agent")
    assert len(worker._packed_producer_pending) == 0
    changed_endpoint = _request(
        producer_rank=0,
        chunk_index=0,
        base_address=0x5_0000_0000,
    )
    with (
        patch.object(
            worker,
            "_packed_consumer_agent",
            return_value="consumer-agent",
        ),
        pytest.raises(StagingSafetyError, match="released source contract"),
    ):
        worker._accept_packed_request_notification(
            changed_endpoint,
            "consumer-agent",
        )

    next_generation = _request(
        producer_rank=0,
        chunk_index=0,
        offer_generation=8,
    )
    worker._accept_packed_request_notification(
        next_generation,
        "consumer-agent",
    )
    assert tuple(pending.request for pending in worker._packed_producer_pending) == (
        next_generation,
    )

    assert (
        worker._record_packed_source_drained(authenticated, allow_buffer=False)
        == "recorded"
    )
    changed = _AuthenticatedPackedSourceDrained(
        drained=msgspec.structs.replace(
            authenticated.drained,
            published_chunk_count=1,
        ),
        consumer_agent="consumer-agent",
    )
    with pytest.raises(StagingSafetyError, match="replay changed exact fields"):
        worker._record_packed_source_drained(changed, allow_buffer=False)


@pytest.mark.cpu_test
def test_producer_source_drain_rejects_wrong_sender_and_anchor() -> None:
    """Native sender and canonical chunk-zero command remain immutable."""
    worker, requests, authenticated = _producer_source_drain_contract()
    wrong_sender = _AuthenticatedPackedSourceDrained(
        drained=authenticated.drained,
        consumer_agent="different-consumer-agent",
    )
    with pytest.raises(StagingSafetyError, match="sender differs"):
        worker._record_packed_source_drained(wrong_sender, allow_buffer=False)

    changed_anchor = msgspec.structs.replace(
        requests[0].command,
        destination=msgspec.structs.replace(
            requests[0].command.destination,
            slot_generation=requests[0].command.destination.slot_generation + 1,
        ),
    )
    changed = _AuthenticatedPackedSourceDrained(
        drained=PackedWriteSourceDrained(
            anchor=changed_anchor,
            published_chunk_count=2,
        ),
        consumer_agent="consumer-agent",
    )
    with pytest.raises(StagingSafetyError, match="anchor differs"):
        worker._record_packed_source_drained(changed, allow_buffer=False)


@pytest.mark.cpu_test
def test_producer_source_drain_rejects_stale_offer_generation() -> None:
    """A proof cannot cross the generation-scoped accepted-command binding."""
    worker, requests, authenticated = _producer_source_drain_contract()
    stale_identity = msgspec.structs.replace(
        requests[0].command.chunk,
        offer_generation=requests[0].command.chunk.offer_generation + 1,
    )
    stale = _AuthenticatedPackedSourceDrained(
        drained=PackedWriteSourceDrained(
            anchor=msgspec.structs.replace(
                authenticated.drained.anchor,
                chunk=stale_identity,
            ),
            published_chunk_count=2,
        ),
        consumer_agent="consumer-agent",
    )

    with pytest.raises(StagingSafetyError, match="no accepted command binding"):
        worker._record_packed_source_drained(stale, allow_buffer=False)


@pytest.mark.cpu_test
def test_legacy_completion_cannot_bypass_packed_source_drain() -> None:
    """A legacy read proof cannot release an obligation bound to packed WRITE."""
    worker, _, _ = _producer_source_drain_contract()
    proof = PullReadComplete(
        producer_request_id="prefill-request",
        offer_generation=7,
        consumer_request_id="decode-request",
        consumer_index=0,
        consumer_rank=0,
        consumer_tp_size=1,
        expected_consumers=1,
    )

    with pytest.raises(StagingSafetyError, match="cannot satisfy a packed WRITE"):
        worker._record_pull_completion(proof, allow_buffer=False)
    state = worker._pull_completion_states["prefill-request"]
    assert state.read_completions == set()
    assert state.packed_completions == {}


@pytest.mark.cpu_test
def test_producer_source_drain_waits_for_its_complete_terminal_prefix() -> None:
    """Early drain waits only while producer actors can still finish the prefix."""
    worker, requests, authenticated = _producer_source_drain_contract(
        remembered_chunk_count=1
    )
    worker._packed_producer_pending.append(SimpleNamespace(request=requests[1]))

    assert (
        worker._record_packed_source_drained(authenticated, allow_buffer=False)
        == "waiting"
    )
    assert worker._buffered_packed_source_drained == {
        "prefill-request": {authenticated}
    }
    state = worker._pull_completion_states["prefill-request"]
    assert state.read_completions == set()

    worker._packed_producer_pending.clear()
    pool = worker._packed_producer_pool
    assert pool is not None
    pool.remember_failed_before_admission(
        requests[1],
        "consumer-agent",
        PackedWriteFailedBeforeWrite(
            command=requests[1].command,
            code=PackedWriteFailureCode.PACK_FAILED,
            reason="second terminal completed after early drain",
        ),
    )
    assert (
        worker._record_packed_source_drained(authenticated, allow_buffer=False)
        == "completed"
    )
    assert worker._buffered_packed_source_drained == {}

    incomplete, _, incomplete_proof = _producer_source_drain_contract(
        remembered_chunk_count=1
    )
    with pytest.raises(StagingSafetyError, match="lacks its terminal prefix"):
        incomplete._record_packed_source_drained(
            incomplete_proof,
            allow_buffer=False,
        )


@pytest.mark.cpu_test
def test_consumer_terminal_history_trims_only_retired_requests() -> None:
    """Long-lived decoders keep a bounded replay window without evicting live work."""
    worker = _consumer_worker()
    requests = tuple(
        _request(
            producer_rank=0,
            consumer_request_id=f"decode-request-{index}",
        )
        for index in range(6)
    )
    for request in requests:
        worker._packed_consumer_terminals[request.command.chunk] = (
            _ConsumerPackedTerminal(
                request=request,
                terminal=PackedWriteArrived(command=request.command),
            )
        )
    worker._packed_consumer_requests["decode-request-0"] = cast(
        Any,
        SimpleNamespace(
            identities_by_chunk=((requests[0].command.chunk,),),
        ),
    )

    module = "vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_worker"
    with (
        patch(f"{module}._MAX_PACKED_WRITE_CONSUMER_TERMINALS", 6),
        patch(f"{module}._PACKED_WRITE_CONSUMER_TERMINAL_TRIM", 2),
    ):
        worker._ensure_packed_consumer_terminal_capacity(1)

    retained = set(worker._packed_consumer_terminals)
    assert requests[0].command.chunk in retained
    assert requests[1].command.chunk not in retained
    assert requests[2].command.chunk not in retained
    assert len(retained) == 4


@pytest.mark.cpu_test
@pytest.mark.parametrize("shared_with_handshake", [False, True])
def test_packed_shutdown_releases_only_dynamic_decoder_agents(
    shared_with_handshake: bool,
) -> None:
    """Quiescent teardown neither leaks nor double-removes imported agents."""
    request = _request(producer_rank=0)
    worker, operation = _producer_worker(request)
    pool = worker._packed_producer_pool
    assert pool is not None
    pool.fail_before_write(
        operation.lease,
        PackedWriteFailedBeforeWrite(
            command=request.command,
            code=PackedWriteFailureCode.PACK_FAILED,
            reason="quiescent test cleanup",
        ),
    )
    worker._packed_producer_operations.clear()
    pool.retire_offer("prefill-request", 7)
    if shared_with_handshake:
        worker._remote_agents = {"decode-a": {0: "consumer-agent"}}

    worker._packed_shutdown_cleanup()

    assert worker._packed_consumer_agents == {}
    assert worker._packed_consumer_agent_last_active == {}
    if shared_with_handshake:
        worker.nixl_wrapper.remove_remote_agent.assert_not_called()
    else:
        worker.nixl_wrapper.remove_remote_agent.assert_called_once_with(
            "consumer-agent"
        )


@pytest.mark.cpu_test
def test_producer_native_prepare_failure_sends_exact_failed_before_write() -> None:
    """A definitive preparation failure emits a typed replayable terminal."""
    request = _request(producer_rank=1)
    worker, operation = _producer_worker(request)
    pool = worker._packed_producer_pool
    assert pool is not None
    worker.nixl_wrapper.initialize_xfer.side_effect = RuntimeError("prepare failed")

    worker._advance_packed_producer_pack(operation)

    assert pool.free_slot_count == 1
    worker.nixl_wrapper.transfer.assert_not_called()
    worker.nixl_wrapper.release_xfer_handle.assert_not_called()
    worker.nixl_wrapper.send_notif.assert_called_once()
    notification = worker.nixl_wrapper.send_notif.call_args.kwargs["notif_msg"]
    assert notification.startswith(PACKED_WRITE_FAILED_BEFORE_WRITE_PREFIX)
    failed = decode_packed_write_failed_before_write_notification(notification)
    assert failed.command == request.command
    assert failed.code is PackedWriteFailureCode.WRITE_PREPARE_FAILED
    assert failed.reason.startswith("packed WRITE native preparation failed")
    assert len(failed.reason) <= 1024
    assert pool.terminal_for_replay(request, "consumer-agent") == failed


@pytest.mark.cpu_test
def test_producer_uncertain_submission_tombstones_without_false_failure() -> None:
    """An exception across native submission never claims that no WRITE posted."""
    request = _request(producer_rank=3)
    worker, operation = _producer_worker(request)
    pool = worker._packed_producer_pool
    assert pool is not None
    worker.nixl_wrapper.transfer.side_effect = RuntimeError("submission uncertain")

    with pytest.raises(StagingSafetyError, match="submission became uncertain"):
        worker._advance_packed_producer_pack(operation)

    assert pool.free_slot_count == 0
    assert pool.has_registered_ownership
    assert pool.request_for_identity(request.command.chunk) == request
    assert pool.terminal_for_replay(request, "consumer-agent") is None
    worker.nixl_wrapper.send_notif.assert_not_called()
    worker.nixl_wrapper.release_xfer_handle.assert_not_called()


@pytest.mark.cpu_test
def test_producer_done_handle_release_failure_tombstones_slot() -> None:
    """DONE without successful native-handle release remains permanently owned."""
    request = _request(producer_rank=0)
    worker, operation = _producer_worker(request)
    pool = worker._packed_producer_pool
    assert pool is not None
    worker._advance_packed_producer_pack(operation)
    worker.nixl_wrapper.release_xfer_handle.side_effect = RuntimeError("release failed")

    with pytest.raises(StagingSafetyError, match="release failed after DONE"):
        worker._advance_packed_producer_write(operation)

    assert pool.free_slot_count == 0
    assert pool.has_registered_ownership
    assert pool.terminal_for_replay(request, "consumer-agent") is None
    worker.nixl_wrapper.send_notif.assert_not_called()


@pytest.mark.cpu_test
def test_producer_routes_two_decoders_by_exact_endpoint_and_sender() -> None:
    """Independent decoder registrations cannot alias or spoof one another."""
    first = _request(
        consumer_engine_id="decode-a",
        consumer_request_id="decode-a-request",
        base_address=0x4_0000_0000,
    )
    second = _request(
        consumer_engine_id="decode-b",
        consumer_request_id="decode-b-request",
        base_address=0x5_0000_0000,
    )
    worker, _ = _producer_worker(first)
    worker._packed_producer_operations.clear()
    worker._packed_producer_pool = None
    worker._packed_consumer_agents.clear()
    worker._packed_consumer_agent_last_active.clear()
    agent_by_metadata = {
        first.consumer_endpoint.agent_metadata: b"decoder-agent-a",
        second.consumer_endpoint.agent_metadata: b"decoder-agent-b",
    }
    worker.nixl_wrapper.add_remote_agent.side_effect = agent_by_metadata.__getitem__

    worker._accept_packed_request_notification(first, "decoder-agent-a")
    worker._accept_packed_request_notification(second, "decoder-agent-b")

    assert tuple(pending.request for pending in worker._packed_producer_pending) == (
        first,
        second,
    )
    assert worker._packed_consumer_agents == {
        first.consumer_endpoint: "decoder-agent-a",
        second.consumer_endpoint: "decoder-agent-b",
    }
    with pytest.raises(StagingSafetyError, match="authenticated endpoint"):
        worker._accept_packed_request_notification(second, "decoder-agent-a")


@pytest.mark.cpu_test
def test_consumer_exact_duplicate_at_arrival_quorum_is_idempotent() -> None:
    """A queued duplicate cannot re-enter ownership after scatter begins."""
    worker = _consumer_worker()
    _start_consumer_request(worker)
    owner = worker._packed_consumer_requests["decode-request"]
    first = owner.active_chunks[0]
    _accept_arrivals(worker, first.requests)
    duplicate = first.requests[-1]
    worker._accept_packed_arrived_notification(
        PackedWriteArrived(command=duplicate.command),
        "producer-agent-3",
    )

    with patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_worker."
        "launch_packed_chunk_scatter",
        return_value=_FakeLaunch(complete=False),
    ) as scatter:
        worker._consume_packed_terminal_notifications()

    scatter.assert_called_once()
    assert len(worker._packed_terminal_notifications) == 0
    assert set(first.terminals_by_rank) == set(_SOURCE_RANKS)
    assert first.scatter_launch is not None


@pytest.mark.cpu_test
def test_consumer_full_full_tail_scatter_reuses_slot_generation_atomically() -> None:
    """Three chunks scatter only after quorum and publish after the tail."""
    worker = _consumer_worker()
    launches = [_FakeLaunch(), _FakeLaunch(), _FakeLaunch()]
    with patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_worker."
        "launch_packed_chunk_scatter",
        side_effect=launches,
    ) as scatter:
        _start_consumer_request(worker)
        owner = worker._packed_consumer_requests["decode-request"]
        assert tuple(owner.active_chunks) == (0, 1)
        assert tuple(owner.pending_chunk_indices) == (2,)
        assert len(_published_requests(worker)) == 8

        for chunk_index in (0, 1):
            _accept_arrivals(worker, owner.active_chunks[chunk_index].requests)
        duplicate = owner.active_chunks[0].requests[-1]
        worker._accept_packed_arrived_notification(
            PackedWriteArrived(command=duplicate.command),
            "producer-agent-3",
        )
        assert scatter.call_count == 0
        worker._service_packed_consumer()

        owner = worker._packed_consumer_requests["decode-request"]
        assert tuple(owner.active_chunks) == (2,)
        tail = owner.active_chunks[2]
        assert tail.lease.slot_index == 0
        assert tail.lease.binding.slot_generation == 2
        assert tail.lease.binding.payload_bytes == _ROW_BYTES
        assert len(_published_requests(worker)) == 12
        assert worker._packed_done_recving == set()

        _accept_arrivals(worker, tail.requests)
        worker._service_packed_consumer()

    assert "decode-request" not in worker._packed_consumer_requests
    assert worker._packed_done_recving == {"decode-request"}
    assert worker._packed_consumer_pool is not None
    assert worker._packed_consumer_pool.free_slot_count == 2
    assert scatter.call_count == 3
    assert [scatter_call.args[4] for scatter_call in scatter.call_args_list] == [
        0,
        1,
        2,
    ]
    assert [
        scatter_call.kwargs["staging_base_offset_bytes"]
        for scatter_call in scatter.call_args_list
    ] == [0, _CONSUMER_SLOT_BYTES, 0]
    assert all(
        scatter_call.kwargs["staging_rank_stride_bytes"] == _CHUNK_BYTES_PER_RANK
        for scatter_call in scatter.call_args_list
    )
    source_drains = _source_drains(worker)
    assert [agent for agent, _drained in source_drains] == [
        f"producer-agent-{rank}" for rank in _SOURCE_RANKS
    ]
    assert [
        drained.anchor.chunk.producer_rank for _agent, drained in source_drains
    ] == [*_SOURCE_RANKS]
    assert all(
        drained.anchor.chunk.chunk_ordinal == 0 and drained.published_chunk_count == 3
        for _agent, drained in source_drains
    )
    assert all(
        send.kwargs["notif_msg"] != b"source-release"
        for send in worker.nixl_wrapper.send_notif.call_args_list
    )
    worker._record_failed_receive.assert_not_called()
    worker.xfer_stats.record_packed_write.assert_called_once()

    retired = next(iter(worker._packed_consumer_terminals.values()))
    assert isinstance(retired.terminal, PackedWriteArrived)
    queued_before = len(worker._packed_terminal_notifications)
    worker._accept_packed_arrived_notification(
        cast(PackedWriteArrived, retired.terminal),
        f"producer-agent-{retired.request.command.chunk.producer_rank}",
    )
    assert len(worker._packed_terminal_notifications) == queued_before
    conflicting = PackedWriteFailedBeforeWrite(
        command=retired.request.command,
        code=PackedWriteFailureCode.PACK_FAILED,
        reason="impossible conflicting terminal",
    )
    with pytest.raises(StagingSafetyError, match="conflicting terminals"):
        worker._accept_packed_failed_notification(
            conflicting,
            f"producer-agent-{retired.request.command.chunk.producer_rank}",
        )


@pytest.mark.cpu_test
def test_consumer_failure_drains_every_published_rank_before_discard() -> None:
    """One rank failure cancels pending chunks but drains all published chunks."""
    worker = _consumer_worker()
    with patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_worker."
        "launch_packed_chunk_scatter",
    ) as scatter:
        _start_consumer_request(worker)
        owner = worker._packed_consumer_requests["decode-request"]
        first = owner.active_chunks[0]
        second = owner.active_chunks[1]
        first_rank_zero = first.requests[0]
        worker._accept_packed_failed_notification(
            PackedWriteFailedBeforeWrite(
                command=first_rank_zero.command,
                code=PackedWriteFailureCode.PACK_FAILED,
                reason="rank zero pack failed",
            ),
            "producer-agent-0",
        )
        _accept_arrivals(worker, first.requests[1:])
        worker._service_packed_consumer()

        owner = worker._packed_consumer_requests["decode-request"]
        meta = owner.meta
        assert owner.failure_reason is not None
        assert len(owner.pending_chunk_indices) == 0
        assert tuple(owner.active_chunks) == (1,)
        assert len(_published_requests(worker)) == 8
        assert worker._packed_done_recving == set()
        worker._record_failed_receive.assert_not_called()
        assert all(
            not send.kwargs["notif_msg"].startswith(PACKED_WRITE_SOURCE_DRAINED_PREFIX)
            for send in worker.nixl_wrapper.send_notif.call_args_list
        )

        _accept_arrivals(worker, second.requests)
        worker._service_packed_consumer()

    scatter.assert_not_called()
    assert "decode-request" not in worker._packed_consumer_requests
    assert worker._packed_done_recving == {"decode-request"}
    assert worker._packed_consumer_pool is not None
    assert worker._packed_consumer_pool.free_slot_count == 2
    assert len(worker._packed_consumer_terminals) == 8
    worker._record_failed_receive.assert_called_once()
    assert worker._record_failed_receive.call_args.args == (
        "decode-request",
        KVTransferFailureReason.TRANSFER,
        _consumer_inputs()[0],
    )
    source_drains = _source_drains(worker)
    assert len(source_drains) == _SOURCE_TP_SIZE
    assert all(
        drained.anchor.chunk.chunk_ordinal == 0 and drained.published_chunk_count == 2
        for _agent, drained in source_drains
    )
    assert meta.remote is not None
    assert worker._remote_source_consumption_proven == {
        "decode-request": meta.remote.offer_key
    }
    assert worker._consume_remote_source_consumption_proof("decode-request", meta)
    assert worker._remote_offer_completion_counts == {}
    assert meta.remote.offer_key in worker._released_remote_offers


@pytest.mark.cpu_test
def test_mixed_packed_children_fence_only_after_both_source_drains() -> None:
    """A successful and failed packed child jointly satisfy a two-child offer."""
    worker = _consumer_worker()
    success_id = "0_decode-request"
    failed_id = "1_decode-request"
    with patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_worker."
        "launch_packed_chunk_scatter",
        side_effect=(_FakeLaunch(), _FakeLaunch(), _FakeLaunch()),
    ):
        _start_consumer_request(
            worker,
            request_id=success_id,
            expected_consumers=2,
        )
        success = worker._packed_consumer_requests[success_id]
        success_meta = success.meta
        for chunk_index in (0, 1):
            _accept_arrivals(worker, success.active_chunks[chunk_index].requests)
        worker._service_packed_consumer()
        success = worker._packed_consumer_requests[success_id]
        _accept_arrivals(worker, success.active_chunks[2].requests)
        worker._service_packed_consumer()

    assert success_meta.remote is not None
    offer_key = success_meta.remote.offer_key
    assert worker._remote_source_consumption_proven == {success_id: offer_key}
    assert worker._consume_remote_source_consumption_proof(
        success_id,
        success_meta,
    )
    assert worker._remote_offer_completion_counts == {offer_key: 1}
    assert offer_key not in worker._released_remote_offers

    _start_consumer_request(
        worker,
        request_id=failed_id,
        expected_consumers=2,
    )
    failed = worker._packed_consumer_requests[failed_id]
    failed_meta = failed.meta
    first = failed.active_chunks[0]
    second = failed.active_chunks[1]
    worker._accept_packed_failed_notification(
        PackedWriteFailedBeforeWrite(
            command=first.requests[0].command,
            code=PackedWriteFailureCode.PACK_FAILED,
            reason="rank zero pack failed",
        ),
        "producer-agent-0",
    )
    _accept_arrivals(worker, first.requests[1:])
    worker._service_packed_consumer()
    _accept_arrivals(worker, second.requests)
    worker._service_packed_consumer()

    assert failed_meta.remote is not None
    assert failed_meta.remote.offer_key == offer_key
    assert worker._remote_source_consumption_proven == {failed_id: offer_key}
    assert worker._consume_remote_source_consumption_proof(failed_id, failed_meta)
    assert worker._remote_offer_completion_counts == {}
    assert offer_key in worker._released_remote_offers


@pytest.mark.cpu_test
def test_consumer_rejects_wrong_rank_sender_before_terminal_queueing() -> None:
    """A valid command from the wrong native rank identity is rejected."""
    worker = _consumer_worker()
    _start_consumer_request(worker)
    owner = worker._packed_consumer_requests["decode-request"]
    request = owner.active_chunks[0].requests[2]

    with pytest.raises(StagingSafetyError, match="authenticated rank agent"):
        worker._accept_packed_arrived_notification(
            PackedWriteArrived(command=request.command),
            "producer-agent-1",
        )

    assert len(worker._packed_terminal_notifications) == 0
    assert owner.active_chunks[0].terminals_by_rank == {}


@pytest.mark.cpu_test
def test_consumer_uncertain_publication_tombstones_request_and_slot() -> None:
    """A command-send exception prevents slot reuse and successful exposure."""
    worker = _consumer_worker()
    worker.nixl_wrapper.send_notif.side_effect = [None, None, RuntimeError("send")]

    with pytest.raises(StagingSafetyError, match="publication became uncertain"):
        _start_consumer_request(worker)

    owner = worker._packed_consumer_requests["decode-request"]
    assert owner.tracker.tombstoned
    assert worker._packed_consumer_pool is not None
    assert worker._packed_consumer_pool.free_slot_count == 1
    assert worker._packed_consumer_pool.has_registered_ownership
    assert worker._packed_done_recving == set()


@pytest.mark.cpu_test
def test_producer_and_consumer_watchdogs_tombstone_uncertain_work() -> None:
    """Both actor watchdogs fail closed without making a slot reusable."""
    producer_request = _request()
    producer, operation = _producer_worker(producer_request)
    operation.stage_started_at = 0.0
    with pytest.raises(StagingSafetyError, match="producer pack timeout"):
        producer._service_packed_producer_watchdogs()
    assert producer._packed_producer_pool is not None
    assert producer._packed_producer_pool.free_slot_count == 0

    consumer = _consumer_worker()
    _start_consumer_request(consumer)
    owner = consumer._packed_consumer_requests["decode-request"]
    owner.active_chunks[0].admitted_at = 0.0
    with pytest.raises(StagingSafetyError, match="WRITE arrival timeout"):
        consumer._service_packed_consumer_watchdogs()
    assert owner.tracker.tombstoned
    assert consumer._packed_consumer_pool is not None
    assert consumer._packed_consumer_pool.free_slot_count == 0
