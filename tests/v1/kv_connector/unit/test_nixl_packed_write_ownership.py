# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Behavioral proofs for producer-initiated packed-WRITE ownership."""

from dataclasses import replace

import msgspec
import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
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
)
from vllm.distributed.kv_transfer.packed_write_ownership import (
    ConsumerSlotLease,
    PackedWriteChunkKey,
    PackedWriteConsumerPool,
    PackedWriteProducerPool,
    PackedWriteRequestTracker,
    ProducerSlotLease,
)
from vllm.distributed.kv_transfer.staging_ownership import StagingSafetyError

_CONSUMER_AGENT = "decoder-agent"
_PAYLOAD_BYTES = 512
_RANK_STRIDE_BYTES = 1024
_SOURCE_TP_SIZE = 2


def _digest(character: str) -> str:
    """Build a test SHA-256 identity.

    :param character: One lowercase hexadecimal character.
    :returns: A syntactically valid digest.
    """
    return character * 64


def _producer_geometry(*, slot_count: int = 2) -> PackedWriteProducerPoolGeometry:
    """Build one registered producer gather pool.

    :param slot_count: Number of independently owned gather slots.
    :returns: Valid producer pool geometry.
    """
    return PackedWriteProducerPoolGeometry(
        registration_generation="producer-registration",
        base_address=0x100000,
        registered_bytes=slot_count * _RANK_STRIDE_BYTES,
        slot_size_bytes=_RANK_STRIDE_BYTES,
        slot_count=slot_count,
        device_id=0,
        alignment_bytes=256,
    )


def _consumer_geometry(
    *,
    slot_count: int = 2,
    source_tp_size: int = _SOURCE_TP_SIZE,
    registration_generation: str = "consumer-registration",
) -> PackedWriteConsumerPoolGeometry:
    """Build one registered rank-major decoder receive pool.

    :param slot_count: Number of independently owned receive slots.
    :param source_tp_size: Producer ranks represented by every slot.
    :param registration_generation: Pool-lifetime identity.
    :returns: Valid consumer pool geometry.
    """
    slot_size_bytes = source_tp_size * _RANK_STRIDE_BYTES
    return PackedWriteConsumerPoolGeometry(
        registration_generation=registration_generation,
        base_address=0x200000,
        registered_bytes=slot_count * slot_size_bytes,
        slot_size_bytes=slot_size_bytes,
        slot_count=slot_count,
        source_tp_size=source_tp_size,
        rank_stride_bytes=_RANK_STRIDE_BYTES,
        device_id=1,
        alignment_bytes=256,
    )


def _identity(
    producer_rank: int,
    *,
    offer_generation: int = 1,
    consumer_request_id: str = "decoder-request",
    chunk_ordinal: int = 0,
    chunk_count: int = 2,
    chunk_digest: str | None = None,
) -> PackedWriteChunkIdentity:
    """Build one rankful chunk identity.

    :param producer_rank: Producer rank addressed by the command.
    :param offer_generation: Retained source allocation generation.
    :param consumer_request_id: Complete decoder request identity.
    :param chunk_ordinal: Canonical bounded chunk ordinal.
    :param chunk_count: Complete bounded chunk count.
    :param chunk_digest: Optional exact chunk digest override.
    :returns: Valid immutable chunk identity.
    """
    digest = (
        _digest("c")
        if chunk_digest is None and chunk_ordinal == 0
        else _digest("d")
        if chunk_digest is None
        else chunk_digest
    )
    return PackedWriteChunkIdentity(
        producer_engine_id="prefill",
        producer_request_id="prefill-request",
        offer_generation=offer_generation,
        producer_rank=producer_rank,
        producer_tp_size=_SOURCE_TP_SIZE,
        consumer_engine_id="decode",
        consumer_request_id=consumer_request_id,
        consumer_rank=0,
        consumer_tp_size=1,
        source_plan_digest=_digest("a"),
        packed_plan_digest=_digest("b"),
        chunk_plan_digest=digest,
        chunk_ordinal=chunk_ordinal,
        chunk_count=chunk_count,
        valid_token_extent=2048,
        exact_bytes=_PAYLOAD_BYTES,
    )


def _binding(
    geometry: PackedWriteConsumerPoolGeometry,
    *,
    slot_index: int = 0,
    slot_generation: int = 1,
) -> PackedWriteSlotBinding:
    """Build one exact decoder destination lease.

    :param geometry: Consumer pool that owns the slot.
    :param slot_index: Physical slot index.
    :param slot_generation: Monotonic reuse generation.
    :returns: Valid destination binding.
    """
    return PackedWriteSlotBinding(
        pool_registration_generation=geometry.registration_generation,
        slot_index=slot_index,
        slot_generation=slot_generation,
        payload_bytes=_PAYLOAD_BYTES,
    )


def _request(
    producer_rank: int,
    *,
    geometry: PackedWriteConsumerPoolGeometry | None = None,
    binding: PackedWriteSlotBinding | None = None,
    offer_generation: int = 1,
    consumer_request_id: str = "decoder-request",
    chunk_ordinal: int = 0,
    chunk_count: int = 2,
    chunk_digest: str | None = None,
) -> PackedWriteRequest:
    """Build one exact producer-rank command.

    :param producer_rank: Producer rank addressed by the command.
    :param geometry: Consumer destination pool override.
    :param binding: Destination slot override.
    :param offer_generation: Retained source allocation generation.
    :param consumer_request_id: Complete decoder request identity.
    :param chunk_ordinal: Canonical bounded chunk ordinal.
    :param chunk_count: Complete bounded chunk count.
    :param chunk_digest: Optional exact chunk digest override.
    :returns: Valid packed-WRITE request.
    """
    consumer_geometry = _consumer_geometry() if geometry is None else geometry
    destination = _binding(consumer_geometry) if binding is None else binding
    identity = _identity(
        producer_rank,
        offer_generation=offer_generation,
        consumer_request_id=consumer_request_id,
        chunk_ordinal=chunk_ordinal,
        chunk_count=chunk_count,
        chunk_digest=chunk_digest,
    )
    return PackedWriteRequest(
        command=PackedWriteCommand(
            chunk=identity,
            destination=destination,
        ),
        consumer_endpoint=PackedWriteConsumerEndpoint(
            consumer_engine_id="decode",
            consumer_rank=0,
            consumer_tp_size=1,
            registration_generation=consumer_geometry.registration_generation,
            agent_metadata=b"decoder-agent-metadata",
            consumer_pool=consumer_geometry,
        ),
        source_selections=(
            PackedWriteSourceSelection(
                group_index=0,
                source_position_start=8,
                position_count=64,
            ),
            PackedWriteSourceSelection(
                group_index=1,
                source_position_start=4,
                position_count=32,
            ),
        ),
    )


def _failed(
    request: PackedWriteRequest,
    *,
    code: PackedWriteFailureCode = PackedWriteFailureCode.PACK_FAILED,
    reason: str = "packing failed before WRITE",
) -> PackedWriteFailedBeforeWrite:
    """Build one exact no-WRITE terminal.

    :param request: Request whose command failed.
    :param code: Exact failure category.
    :param reason: Bounded diagnostic reason.
    :returns: Valid failure terminal.
    """
    return PackedWriteFailedBeforeWrite(
        command=request.command,
        code=code,
        reason=reason,
    )


def _reserve_and_bind(
    pool: PackedWriteConsumerPool,
    *,
    offer_generation: int = 1,
    consumer_request_id: str = "decoder-request",
    chunk_ordinal: int = 0,
    chunk_count: int = 2,
) -> tuple[ConsumerSlotLease, tuple[PackedWriteRequest, ...]]:
    """Reserve a decoder slot and bind its complete rank quorum.

    :param pool: Consumer pool to reserve.
    :param offer_generation: Retained source allocation generation.
    :param consumer_request_id: Complete decoder request identity.
    :param chunk_ordinal: Canonical bounded chunk ordinal.
    :param chunk_count: Complete bounded chunk count.
    :returns: Exact slot lease and rank request quorum.
    """
    first_identity = _identity(
        0,
        offer_generation=offer_generation,
        consumer_request_id=consumer_request_id,
        chunk_ordinal=chunk_ordinal,
        chunk_count=chunk_count,
    )
    chunk = PackedWriteChunkKey.from_identity(first_identity)
    lease = pool.reserve(chunk, _PAYLOAD_BYTES)
    assert lease is not None
    requests = tuple(
        _request(
            producer_rank,
            geometry=pool.geometry,
            binding=lease.binding,
            offer_generation=offer_generation,
            consumer_request_id=consumer_request_id,
            chunk_ordinal=chunk_ordinal,
            chunk_count=chunk_count,
        )
        for producer_rank in range(_SOURCE_TP_SIZE)
    )
    pool.bind_requests(lease, requests)
    return lease, requests


def _publish_all(
    pool: PackedWriteConsumerPool,
    lease: ConsumerSlotLease,
) -> None:
    """Publish and seal every bound producer rank.

    :param pool: Consumer pool that owns the lease.
    :param lease: Exact publishing generation.
    """
    for producer_rank in range(_SOURCE_TP_SIZE):
        pool.mark_rank_published(lease, producer_rank)
    pool.seal_publication(lease)


def test_producer_waits_for_native_done_before_reusing_a_slot() -> None:
    pool = PackedWriteProducerPool(_producer_geometry(slot_count=1), producer_rank=0)
    request = _request(0)
    admission = pool.admit(request, _CONSUMER_AGENT)
    assert admission.lease is not None
    assert admission.newly_admitted

    pool.mark_pack_complete(admission.lease)
    pool.begin_write_post(admission.lease)
    pool.record_write_posted(admission.lease, "PROC")
    with pytest.raises(StagingSafetyError, match="WRITE_DONE"):
        pool.mark_write_released(admission.lease)
    with pytest.raises(StagingSafetyError, match="WRITE_RELEASED"):
        pool.complete_arrived(admission.lease)
    assert pool.free_slot_count == 0

    pool.record_write_query(admission.lease, "PROC")
    pool.record_write_query(admission.lease, "DONE")
    with pytest.raises(StagingSafetyError, match="WRITE_RELEASED"):
        pool.complete_arrived(admission.lease)
    pool.mark_write_released(admission.lease)
    arrived = pool.complete_arrived(admission.lease)

    assert arrived == PackedWriteArrived(command=request.command)
    assert pool.free_slot_count == 1
    assert pool.has_registered_ownership is False


def test_producer_immediate_done_and_terminal_replay_are_exact() -> None:
    pool = PackedWriteProducerPool(_producer_geometry(), producer_rank=0)
    request = _request(0)
    admission = pool.admit(request, _CONSUMER_AGENT)
    assert admission.lease is not None
    duplicate = pool.admit(request, _CONSUMER_AGENT)
    assert duplicate.lease == admission.lease
    assert duplicate.newly_admitted is False

    pool.mark_pack_complete(admission.lease)
    pool.begin_write_post(admission.lease)
    pool.record_write_posted(admission.lease, "DONE")
    pool.mark_write_released(admission.lease)
    arrived = pool.complete_arrived(admission.lease)
    assert pool.terminal_for_replay(request, _CONSUMER_AGENT) == arrived
    replay = pool.admit(request, _CONSUMER_AGENT)

    assert replay.lease is None
    assert replay.terminal == arrived
    assert replay.newly_admitted is False


def test_producer_terminal_lookup_never_allocates_and_rejects_conflicts() -> None:
    pool = PackedWriteProducerPool(_producer_geometry(slot_count=1), producer_rank=0)
    request = _request(0)
    admission = pool.admit(request, _CONSUMER_AGENT)
    assert admission.lease is not None
    failure = _failed(request)
    pool.fail_before_write(admission.lease, failure)
    assert pool.free_slot_count == 1

    assert pool.terminal_for_replay(request, _CONSUMER_AGENT) == failure
    assert pool.free_slot_count == 1
    unseen = _request(
        0,
        offer_generation=2,
        consumer_request_id="unseen-request",
    )
    assert pool.terminal_for_replay(unseen, _CONSUMER_AGENT) is None
    assert pool.free_slot_count == 1

    changed = msgspec.structs.replace(
        request,
        source_selections=(
            PackedWriteSourceSelection(0, 7, 65),
            request.source_selections[1],
        ),
    )
    with pytest.raises(StagingSafetyError, match="authenticated fields"):
        pool.terminal_for_replay(changed, _CONSUMER_AGENT)
    with pytest.raises(StagingSafetyError, match="consumer agent"):
        pool.terminal_for_replay(request, "different-decoder-agent")


def test_producer_active_lookup_validates_without_queueing_or_allocating() -> None:
    pool = PackedWriteProducerPool(_producer_geometry(slot_count=1), producer_rank=0)
    request = _request(0)
    admission = pool.admit(request, _CONSUMER_AGENT)
    assert admission.lease is not None

    assert pool.active_for_replay(request, _CONSUMER_AGENT)
    assert pool.active_slot_count == 1
    changed = msgspec.structs.replace(
        request,
        source_selections=(
            PackedWriteSourceSelection(0, 7, 65),
            request.source_selections[1],
        ),
    )
    with pytest.raises(StagingSafetyError, match="authenticated fields"):
        pool.active_for_replay(changed, _CONSUMER_AGENT)
    with pytest.raises(StagingSafetyError, match="consumer agent"):
        pool.active_for_replay(request, "different-decoder-agent")

    pool.fail_before_write(admission.lease, _failed(request))
    assert pool.active_for_replay(request, _CONSUMER_AGENT) is False


def test_producer_replay_history_retires_only_one_quiescent_offer() -> None:
    pool = PackedWriteProducerPool(_producer_geometry(slot_count=1), producer_rank=0)
    request = _request(0)
    admission = pool.admit(request, _CONSUMER_AGENT)
    assert admission.lease is not None

    with pytest.raises(StagingSafetyError, match="cannot retire while active"):
        pool.retire_offer("prefill-request", 1)

    failure = _failed(request)
    pool.fail_before_write(admission.lease, failure)
    next_generation = _request(0, offer_generation=2)
    next_failure = _failed(next_generation)
    pool.remember_failed_before_admission(
        next_generation,
        _CONSUMER_AGENT,
        next_failure,
    )
    assert pool.terminal_for_replay(request, _CONSUMER_AGENT) == failure
    assert pool.retire_offer("prefill-request", 1) == 1
    assert pool.terminal_for_replay(request, _CONSUMER_AGENT) is None
    assert pool.terminal_for_replay(next_generation, _CONSUMER_AGENT) == next_failure
    assert pool.retire_offer("prefill-request", 1) == 0
    assert pool.retire_offer("prefill-request", 2) == 1


def test_producer_validates_one_exact_terminal_prefix_for_source_drain() -> None:
    pool = PackedWriteProducerPool(_producer_geometry(), producer_rank=0)
    requests = tuple(
        _request(0, chunk_ordinal=chunk_ordinal) for chunk_ordinal in range(2)
    )
    for request in requests:
        pool.remember_failed_before_admission(
            request,
            _CONSUMER_AGENT,
            _failed(request),
        )
    drained = PackedWriteSourceDrained(
        anchor=requests[0].command,
        published_chunk_count=2,
    )

    assert pool.validate_source_drained(drained, _CONSUMER_AGENT)
    assert pool.validate_source_drained(drained, _CONSUMER_AGENT)
    with pytest.raises(StagingSafetyError, match="sender differs"):
        pool.validate_source_drained(drained, "different-decoder-agent")

    changed_anchor = msgspec.structs.replace(
        requests[0].command,
        destination=msgspec.structs.replace(
            requests[0].command.destination,
            slot_generation=requests[0].command.destination.slot_generation + 1,
        ),
    )
    with pytest.raises(StagingSafetyError, match="anchor differs"):
        pool.validate_source_drained(
            PackedWriteSourceDrained(
                anchor=changed_anchor,
                published_chunk_count=2,
            ),
            _CONSUMER_AGENT,
        )


def test_producer_source_drain_cannot_skip_or_invent_terminal_history() -> None:
    pool = PackedWriteProducerPool(_producer_geometry(), producer_rank=0)
    first = _request(0, chunk_ordinal=0)
    second = _request(0, chunk_ordinal=1)
    pool.remember_failed_before_admission(first, _CONSUMER_AGENT, _failed(first))
    incomplete = PackedWriteSourceDrained(
        anchor=first.command,
        published_chunk_count=2,
    )

    assert pool.validate_source_drained(incomplete, _CONSUMER_AGENT) is False

    pool.remember_failed_before_admission(second, _CONSUMER_AGENT, _failed(second))
    with pytest.raises(StagingSafetyError, match="omits a published"):
        pool.validate_source_drained(
            PackedWriteSourceDrained(
                anchor=first.command,
                published_chunk_count=1,
            ),
            _CONSUMER_AGENT,
        )

    stale = _request(0, offer_generation=2, chunk_ordinal=0)
    assert (
        pool.validate_source_drained(
            PackedWriteSourceDrained(
                anchor=stale.command,
                published_chunk_count=1,
            ),
            _CONSUMER_AGENT,
        )
        is False
    )


def test_producer_rejects_request_or_agent_conflicts_on_replay() -> None:
    pool = PackedWriteProducerPool(_producer_geometry(), producer_rank=0)
    request = _request(0)
    admission = pool.admit(request, _CONSUMER_AGENT)
    assert admission.lease is not None
    changed_request = msgspec.structs.replace(
        request,
        source_selections=(
            PackedWriteSourceSelection(0, 7, 65),
            request.source_selections[1],
        ),
    )

    with pytest.raises(StagingSafetyError, match="authenticated fields"):
        pool.admit(changed_request, _CONSUMER_AGENT)
    with pytest.raises(StagingSafetyError, match="consumer agent"):
        pool.admit(request, "different-decoder-agent")

    pool.fail_before_write(admission.lease, _failed(request))
    with pytest.raises(StagingSafetyError, match="authenticated fields"):
        pool.admit(changed_request, _CONSUMER_AGENT)
    with pytest.raises(StagingSafetyError, match="consumer agent"):
        pool.admit(request, "different-decoder-agent")


def test_producer_backpressure_is_bounded_and_generation_scoped() -> None:
    pool = PackedWriteProducerPool(_producer_geometry(slot_count=1), producer_rank=0)
    first = pool.admit(_request(0), _CONSUMER_AGENT)
    assert first.lease is not None
    blocked = pool.admit(
        _request(0, offer_generation=2, consumer_request_id="second-request"),
        _CONSUMER_AGENT,
    )
    assert blocked.lease is None
    assert blocked.terminal is None
    assert blocked.newly_admitted is False

    pool.fail_before_write(first.lease, _failed(_request(0)))
    second = pool.admit(
        _request(0, offer_generation=2, consumer_request_id="second-request"),
        _CONSUMER_AGENT,
    )
    assert second.lease is not None
    assert second.lease.slot_index == first.lease.slot_index
    assert second.lease.slot_generation == first.lease.slot_generation + 1
    with pytest.raises(StagingSafetyError, match="stale"):
        pool.mark_pack_complete(first.lease)


@pytest.mark.parametrize("complete_pack", [False, True])
def test_producer_failure_before_write_releases_and_replays(
    complete_pack: bool,
) -> None:
    pool = PackedWriteProducerPool(_producer_geometry(slot_count=1), producer_rank=0)
    request = _request(0)
    admission = pool.admit(request, _CONSUMER_AGENT)
    assert admission.lease is not None
    if complete_pack:
        pool.mark_pack_complete(admission.lease)
    failure = _failed(request)

    pool.fail_before_write(admission.lease, failure)
    replay = pool.admit(request, _CONSUMER_AGENT)

    assert replay.terminal == failure
    assert pool.free_slot_count == 1


def test_producer_cannot_claim_pre_write_failure_after_post_boundary() -> None:
    pool = PackedWriteProducerPool(_producer_geometry(slot_count=1), producer_rank=0)
    request = _request(0)
    admission = pool.admit(request, _CONSUMER_AGENT)
    assert admission.lease is not None
    pool.mark_pack_complete(admission.lease)
    pool.begin_write_post(admission.lease)

    with pytest.raises(StagingSafetyError, match="POSTING_WRITE"):
        pool.fail_before_write(admission.lease, _failed(request))
    pool.tombstone(admission.lease, "transfer outcome is uncertain")

    assert pool.has_registered_ownership
    assert pool.free_slot_count == 0
    with pytest.raises(StagingSafetyError, match="tombstoned"):
        pool.admit(request, _CONSUMER_AGENT)


def test_producer_handle_release_failure_tombstones_done_write() -> None:
    pool = PackedWriteProducerPool(_producer_geometry(slot_count=1), producer_rank=0)
    admission = pool.admit(_request(0), _CONSUMER_AGENT)
    assert admission.lease is not None
    pool.mark_pack_complete(admission.lease)
    pool.begin_write_post(admission.lease)
    pool.record_write_posted(admission.lease, "DONE")

    pool.tombstone(admission.lease, "native handle release failed")

    assert pool.free_slot_count == 0
    assert pool.has_registered_ownership
    with pytest.raises(StagingSafetyError, match="TOMBSTONED"):
        pool.mark_write_released(admission.lease)


@pytest.mark.parametrize(
    ("phase", "status"),
    [("post", "ERR"), ("query", "ERR")],
)
def test_producer_uncertain_native_status_tombstones(
    phase: str,
    status: str,
) -> None:
    pool = PackedWriteProducerPool(_producer_geometry(slot_count=1), producer_rank=0)
    admission = pool.admit(_request(0), _CONSUMER_AGENT)
    assert admission.lease is not None
    pool.mark_pack_complete(admission.lease)
    pool.begin_write_post(admission.lease)
    if phase == "query":
        pool.record_write_posted(admission.lease, "PROC")

    with pytest.raises(StagingSafetyError, match="native packed WRITE"):
        if phase == "post":
            pool.record_write_posted(admission.lease, status)
        else:
            pool.record_write_query(admission.lease, status)

    assert pool.has_registered_ownership
    assert pool.free_slot_count == 0


def test_producer_rejects_wrong_rank_and_oversized_payload() -> None:
    pool = PackedWriteProducerPool(_producer_geometry(), producer_rank=0)
    with pytest.raises(StagingSafetyError, match="wrong producer rank"):
        pool.admit(_request(1), _CONSUMER_AGENT)

    small_geometry = PackedWriteProducerPoolGeometry(
        registration_generation="small-producer-registration",
        base_address=0x300000,
        registered_bytes=256,
        slot_size_bytes=256,
        slot_count=1,
        device_id=0,
        alignment_bytes=256,
    )
    small_pool = PackedWriteProducerPool(small_geometry, producer_rank=0)
    with pytest.raises(StagingSafetyError, match="exceeds producer slot"):
        small_pool.admit(_request(0), _CONSUMER_AGENT)


def test_consumer_reservation_is_bounded_exact_and_generation_scoped() -> None:
    pool = PackedWriteConsumerPool(_consumer_geometry(slot_count=1))
    first_key = PackedWriteChunkKey.from_identity(_identity(0))
    first = pool.reserve(first_key, _PAYLOAD_BYTES)
    assert first is not None
    assert (
        pool.reserve(
            PackedWriteChunkKey.from_identity(
                _identity(0, offer_generation=2, consumer_request_id="second-request")
            ),
            _PAYLOAD_BYTES,
        )
        is None
    )
    with pytest.raises(StagingSafetyError, match="already owns"):
        pool.reserve(first_key, _PAYLOAD_BYTES)

    requests = tuple(
        _request(rank, geometry=pool.geometry, binding=first.binding)
        for rank in range(_SOURCE_TP_SIZE)
    )
    pool.bind_requests(first, requests)
    _publish_all(pool, first)
    for request in requests:
        pool.record_failed_before_write(first, _failed(request))
    pool.discard_terminal(first)

    second_key = PackedWriteChunkKey.from_identity(
        _identity(0, offer_generation=2, consumer_request_id="second-request")
    )
    second = pool.reserve(second_key, _PAYLOAD_BYTES)
    assert second is not None
    assert second.slot_index == first.slot_index
    assert second.binding.slot_generation == first.binding.slot_generation + 1
    with pytest.raises(StagingSafetyError, match="stale"):
        pool.bind_requests(first, requests)


def test_consumer_rejects_payload_and_source_topology_mismatch() -> None:
    pool = PackedWriteConsumerPool(_consumer_geometry())
    key = PackedWriteChunkKey.from_identity(_identity(0))
    with pytest.raises(ValueError, match="equal the chunk"):
        pool.reserve(key, _PAYLOAD_BYTES // 2)

    wrong_topology = replace(key, producer_tp_size=4)
    with pytest.raises(StagingSafetyError, match="source topology"):
        pool.reserve(wrong_topology, _PAYLOAD_BYTES)


def test_consumer_binding_is_atomic_idempotent_and_immutable() -> None:
    pool = PackedWriteConsumerPool(_consumer_geometry(slot_count=1))
    key = PackedWriteChunkKey.from_identity(_identity(0))
    lease = pool.reserve(key, _PAYLOAD_BYTES)
    assert lease is not None
    requests = tuple(
        _request(rank, geometry=pool.geometry, binding=lease.binding)
        for rank in range(_SOURCE_TP_SIZE)
    )

    with pytest.raises(StagingSafetyError, match="incomplete"):
        pool.bind_requests(lease, requests[:1])
    with pytest.raises(StagingSafetyError, match="not bound"):
        pool.request_for_rank(lease, 0)

    pool.bind_requests(lease, requests)
    pool.bind_requests(lease, requests)
    changed = msgspec.structs.replace(
        requests[0],
        source_selections=(
            PackedWriteSourceSelection(0, 7, 65),
            requests[0].source_selections[1],
        ),
    )
    with pytest.raises(StagingSafetyError, match="rebound inconsistently"):
        pool.bind_requests(lease, (changed, requests[1]))

    pool.mark_rank_published(lease, 0)
    with pytest.raises(StagingSafetyError, match="rebound inconsistently"):
        pool.bind_requests(lease, (changed, requests[1]))


def test_consumer_publication_requires_each_exact_rank_once() -> None:
    pool = PackedWriteConsumerPool(_consumer_geometry())
    lease, _ = _reserve_and_bind(pool)

    pool.mark_rank_published(lease, 0)
    with pytest.raises(StagingSafetyError, match="published twice"):
        pool.mark_rank_published(lease, 0)
    with pytest.raises(StagingSafetyError, match="publication quorum"):
        pool.seal_publication(lease)

    pool.mark_rank_published(lease, 1)
    pool.seal_publication(lease)


def test_consumer_all_arrived_quorum_is_the_only_scatter_path() -> None:
    pool = PackedWriteConsumerPool(_consumer_geometry(slot_count=1))
    lease, requests = _reserve_and_bind(pool)
    _publish_all(pool, lease)

    arrived0 = PackedWriteArrived(command=requests[0].command)
    assert pool.record_arrived(lease, arrived0) is False
    assert pool.record_arrived(lease, arrived0) is False
    with pytest.raises(StagingSafetyError, match="all-ARRIVED"):
        pool.begin_scatter(lease)

    assert pool.record_arrived(
        lease,
        PackedWriteArrived(command=requests[1].command),
    )
    assert pool.terminal_complete(lease)
    assert pool.has_failure(lease) is False
    pool.begin_scatter(lease)
    completed = pool.complete_scatter(lease)

    assert completed == lease.chunk
    assert pool.free_slot_count == 1


def test_consumer_mixed_terminal_quorum_discards_without_scatter() -> None:
    pool = PackedWriteConsumerPool(_consumer_geometry(slot_count=1))
    lease, requests = _reserve_and_bind(pool)
    _publish_all(pool, lease)

    assert (
        pool.record_arrived(
            lease,
            PackedWriteArrived(command=requests[0].command),
        )
        is False
    )
    assert pool.record_failed_before_write(lease, _failed(requests[1]))
    assert pool.has_failure(lease)
    with pytest.raises(StagingSafetyError, match="all-ARRIVED"):
        pool.begin_scatter(lease)

    assert pool.discard_terminal(lease) == lease.chunk
    assert pool.free_slot_count == 1


def test_consumer_terminal_replays_are_exact_and_mutually_exclusive() -> None:
    pool = PackedWriteConsumerPool(_consumer_geometry())
    lease, requests = _reserve_and_bind(pool)
    _publish_all(pool, lease)
    failure = _failed(requests[0])

    assert pool.record_failed_before_write(lease, failure) is False
    assert pool.record_failed_before_write(lease, failure) is False
    changed_failure = _failed(
        requests[0],
        code=PackedWriteFailureCode.INTERNAL_ERROR,
        reason="different exact terminal",
    )
    with pytest.raises(StagingSafetyError, match="changed fields"):
        pool.record_failed_before_write(lease, changed_failure)
    with pytest.raises(StagingSafetyError, match="both arrived and failed"):
        pool.record_arrived(
            lease,
            PackedWriteArrived(command=requests[0].command),
        )


def test_consumer_rejects_terminal_for_a_different_command() -> None:
    pool = PackedWriteConsumerPool(_consumer_geometry())
    lease, requests = _reserve_and_bind(pool)
    _publish_all(pool, lease)
    other = _request(
        0,
        geometry=pool.geometry,
        binding=lease.binding,
        offer_generation=2,
    )

    with pytest.raises(StagingSafetyError, match="published command"):
        pool.record_arrived(
            lease,
            PackedWriteArrived(command=other.command),
        )
    assert pool.request_for_rank(lease, 0) == requests[0]


def test_consumer_cannot_discard_an_incomplete_terminal_quorum() -> None:
    pool = PackedWriteConsumerPool(_consumer_geometry())
    lease, requests = _reserve_and_bind(pool)
    _publish_all(pool, lease)
    pool.record_failed_before_write(lease, _failed(requests[0]))

    with pytest.raises(StagingSafetyError, match="every published rank"):
        pool.discard_terminal(lease)
    assert pool.has_registered_ownership


def test_consumer_publication_uncertainty_tombstones_without_reuse() -> None:
    pool = PackedWriteConsumerPool(_consumer_geometry(slot_count=1))
    lease, _ = _reserve_and_bind(pool)
    pool.mark_rank_published(lease, 0)
    pool.tombstone(lease, "notification send outcome is uncertain")

    assert pool.has_registered_ownership
    assert pool.free_slot_count == 0
    with pytest.raises(StagingSafetyError, match="TOMBSTONED"):
        pool.mark_rank_published(lease, 1)


def test_consumer_scatter_uncertainty_tombstones_without_reuse() -> None:
    pool = PackedWriteConsumerPool(_consumer_geometry(slot_count=1))
    lease, requests = _reserve_and_bind(pool)
    _publish_all(pool, lease)
    for request in requests:
        pool.record_arrived(lease, PackedWriteArrived(command=request.command))
    pool.begin_scatter(lease)
    pool.tombstone(lease, "scatter completion is uncertain")

    assert pool.has_registered_ownership
    assert pool.free_slot_count == 0
    with pytest.raises(StagingSafetyError, match="TOMBSTONED"):
        pool.complete_scatter(lease)


def test_tracker_publishes_only_after_one_exact_complete_chunk_set() -> None:
    chunk0 = PackedWriteChunkKey.from_identity(_identity(0, chunk_ordinal=0))
    chunk1 = PackedWriteChunkKey.from_identity(_identity(0, chunk_ordinal=1))
    tracker = PackedWriteRequestTracker.create(chunk0)

    assert tracker.record_chunk_complete(chunk1) is False
    assert tracker.record_chunk_complete(chunk1) is False
    assert tracker.record_chunk_complete(chunk0)
    assert tracker.publication_eligible
    tracker.mark_published()

    assert tracker.publication_eligible is False
    with pytest.raises(StagingSafetyError, match="not publishable"):
        tracker.mark_published()
    with pytest.raises(StagingSafetyError, match="cannot fail"):
        tracker.fail()
    with pytest.raises(StagingSafetyError, match="cannot tombstone"):
        tracker.tombstone()


@pytest.mark.parametrize("terminal_state", ["failed", "tombstoned"])
def test_tracker_never_publishes_failed_or_uncertain_requests(
    terminal_state: str,
) -> None:
    chunk0 = PackedWriteChunkKey.from_identity(_identity(0, chunk_ordinal=0))
    chunk1 = PackedWriteChunkKey.from_identity(_identity(0, chunk_ordinal=1))
    tracker = PackedWriteRequestTracker.create(chunk0)
    if terminal_state == "failed":
        tracker.fail()
    else:
        tracker.tombstone()

    assert tracker.record_chunk_complete(chunk0) is False
    assert tracker.record_chunk_complete(chunk1) is False
    assert tracker.publication_eligible is False
    with pytest.raises(StagingSafetyError, match="not publishable"):
        tracker.mark_published()


def test_tracker_rejects_chunk_conflicts_and_cross_request_completion() -> None:
    chunk0 = PackedWriteChunkKey.from_identity(_identity(0, chunk_ordinal=0))
    tracker = PackedWriteRequestTracker.create(chunk0)
    tracker.record_chunk_complete(chunk0)
    conflicting = replace(chunk0, chunk_plan_digest=_digest("e"))
    with pytest.raises(StagingSafetyError, match="conflicted"):
        tracker.record_chunk_complete(conflicting)

    other_request = replace(chunk0, consumer_request_id="different-request")
    with pytest.raises(StagingSafetyError, match="request identity"):
        tracker.record_chunk_complete(other_request)
    out_of_range = replace(chunk0, chunk_ordinal=chunk0.chunk_count)
    with pytest.raises(StagingSafetyError, match="outside its request"):
        tracker.record_chunk_complete(out_of_range)


def test_tracker_must_be_created_from_canonical_chunk_zero() -> None:
    chunk1 = PackedWriteChunkKey.from_identity(_identity(0, chunk_ordinal=1))
    with pytest.raises(ValueError, match="start at chunk zero"):
        PackedWriteRequestTracker.create(chunk1)


def test_forged_stale_producer_lease_cannot_address_from_the_end() -> None:
    pool = PackedWriteProducerPool(_producer_geometry(slot_count=1), producer_rank=0)
    request = _request(0)
    admission = pool.admit(request, _CONSUMER_AGENT)
    assert admission.lease is not None
    forged = ProducerSlotLease(
        slot_index=-1,
        slot_generation=admission.lease.slot_generation,
        source_address=admission.lease.source_address,
        payload_bytes=admission.lease.payload_bytes,
        chunk=admission.lease.chunk,
    )

    with pytest.raises(StagingSafetyError, match="outside its pool"):
        pool.mark_pack_complete(forged)
