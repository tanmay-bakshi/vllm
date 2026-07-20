# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the bounded NIXL packed-WRITE wire contract."""

import msgspec
import pytest

import vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata as metadata_module
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NIXL_CONNECTOR_VERSION,
    PACKED_WRITE_ARRIVED_PREFIX,
    PACKED_WRITE_FAILED_BEFORE_WRITE_PREFIX,
    PACKED_WRITE_REQUEST_PREFIX,
    PACKED_WRITE_SOURCE_DRAINED_PREFIX,
    NixlAgentMetadata,
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
    decode_packed_write_arrived_notification,
    decode_packed_write_failed_before_write_notification,
    decode_packed_write_request_notification,
    decode_packed_write_source_drained_notification,
    encode_packed_write_arrived_notification,
    encode_packed_write_failed_before_write_notification,
    encode_packed_write_request_notification,
    encode_packed_write_source_drained_notification,
    packed_write_destination_address,
    validate_packed_write_arrived,
    validate_packed_write_failed_before_write,
    validate_packed_write_request_replay,
    validate_packed_write_slot_binding,
)

_MIB = 1024 * 1024
_ALIGNMENT_BYTES = 256
_RANK_STRIDE_BYTES = 256 * _MIB
_SOURCE_TP_SIZE = 4
_CONSUMER_SLOT_SIZE_BYTES = _SOURCE_TP_SIZE * _RANK_STRIDE_BYTES
_SLOT_COUNT = 2


def _digest(character: str) -> str:
    return character * 64


def _producer_pool() -> PackedWriteProducerPoolGeometry:
    return PackedWriteProducerPoolGeometry(
        registration_generation="producer-registration",
        base_address=0x2_0000_0000,
        registered_bytes=_SLOT_COUNT * _RANK_STRIDE_BYTES,
        slot_size_bytes=_RANK_STRIDE_BYTES,
        slot_count=_SLOT_COUNT,
        device_id=2,
        alignment_bytes=_ALIGNMENT_BYTES,
    )


def _consumer_pool() -> PackedWriteConsumerPoolGeometry:
    return PackedWriteConsumerPoolGeometry(
        registration_generation="consumer-registration",
        base_address=0x4_0000_0000,
        registered_bytes=_SLOT_COUNT * _CONSUMER_SLOT_SIZE_BYTES,
        slot_size_bytes=_CONSUMER_SLOT_SIZE_BYTES,
        slot_count=_SLOT_COUNT,
        source_tp_size=_SOURCE_TP_SIZE,
        rank_stride_bytes=_RANK_STRIDE_BYTES,
        device_id=5,
        alignment_bytes=_ALIGNMENT_BYTES,
    )


def _identity() -> PackedWriteChunkIdentity:
    return PackedWriteChunkIdentity(
        producer_engine_id="prefill",
        producer_request_id="producer-request",
        offer_generation=5,
        producer_rank=2,
        producer_tp_size=_SOURCE_TP_SIZE,
        consumer_engine_id="decode",
        consumer_request_id="consumer-request",
        consumer_rank=0,
        consumer_tp_size=1,
        source_plan_digest=_digest("a"),
        packed_plan_digest=_digest("b"),
        chunk_plan_digest=_digest("c"),
        chunk_ordinal=1,
        chunk_count=3,
        valid_token_extent=2048,
        exact_bytes=192 * _MIB,
    )


def _destination() -> PackedWriteSlotBinding:
    return PackedWriteSlotBinding(
        pool_registration_generation="consumer-registration",
        slot_index=1,
        slot_generation=7,
        payload_bytes=192 * _MIB,
    )


def _command(
    *,
    chunk: PackedWriteChunkIdentity | None = None,
    destination: PackedWriteSlotBinding | None = None,
) -> PackedWriteCommand:
    return PackedWriteCommand(
        chunk=_identity() if chunk is None else chunk,
        destination=_destination() if destination is None else destination,
    )


def _endpoint() -> PackedWriteConsumerEndpoint:
    return PackedWriteConsumerEndpoint(
        consumer_engine_id="decode",
        consumer_rank=0,
        consumer_tp_size=1,
        registration_generation="consumer-registration",
        agent_metadata=b"decoder-agent-metadata",
        consumer_pool=_consumer_pool(),
    )


def _selections() -> tuple[PackedWriteSourceSelection, ...]:
    return (
        PackedWriteSourceSelection(
            group_index=0,
            source_position_start=8,
            position_count=120,
        ),
        PackedWriteSourceSelection(
            group_index=1,
            source_position_start=4,
            position_count=60,
        ),
    )


def _request(
    *,
    command: PackedWriteCommand | None = None,
    consumer_endpoint: PackedWriteConsumerEndpoint | None = None,
    source_selections: tuple[PackedWriteSourceSelection, ...] | None = None,
) -> PackedWriteRequest:
    return PackedWriteRequest(
        command=_command() if command is None else command,
        consumer_endpoint=_endpoint()
        if consumer_endpoint is None
        else consumer_endpoint,
        source_selections=_selections()
        if source_selections is None
        else source_selections,
    )


def _source_drained(
    *,
    anchor: PackedWriteCommand | None = None,
    published_chunk_count: int = 2,
) -> PackedWriteSourceDrained:
    return PackedWriteSourceDrained(
        anchor=(
            _command(chunk=msgspec.structs.replace(_identity(), chunk_ordinal=0))
            if anchor is None
            else anchor
        ),
        published_chunk_count=published_chunk_count,
    )


@pytest.mark.cpu_test
def test_connector_version_and_dual_handshake_pools_round_trip() -> None:
    assert NIXL_CONNECTOR_VERSION == 11
    agent_metadata = NixlAgentMetadata(
        engine_id="both",
        tp_rank=0,
        agent_metadata=b"agent",
        kv_caches_base_addr=[0x8_0000_0000],
        device_id=3,
        num_blocks=128,
        block_lens=[64],
        kv_cache_layout="NHD",
        block_size=16,
        ssm_sizes=(0, 0),
        attn_backend_name="FLASH_ATTN",
        physical_blocks_per_logical_kv_block=1,
        registration_generation="kv-registration",
        packed_write_producer_pool=_producer_pool(),
        packed_write_consumer_pool=_consumer_pool(),
    )

    decoded = msgspec.msgpack.decode(
        msgspec.msgpack.encode(agent_metadata),
        type=NixlAgentMetadata,
    )

    assert decoded.packed_write_producer_pool == _producer_pool()
    assert decoded.packed_write_consumer_pool == _consumer_pool()


@pytest.mark.cpu_test
def test_typed_notifications_round_trip_with_exact_prefixes() -> None:
    request = _request()
    arrived = PackedWriteArrived(command=request.command)
    failed = PackedWriteFailedBeforeWrite(
        command=request.command,
        code=PackedWriteFailureCode.PACK_FAILED,
        reason="pack kernel failed before WRITE submission",
    )
    drained = _source_drained()

    request_notification = encode_packed_write_request_notification(request)
    arrived_notification = encode_packed_write_arrived_notification(arrived)
    failed_notification = encode_packed_write_failed_before_write_notification(failed)
    drained_notification = encode_packed_write_source_drained_notification(drained)

    assert request_notification.startswith(PACKED_WRITE_REQUEST_PREFIX)
    assert arrived_notification.startswith(PACKED_WRITE_ARRIVED_PREFIX)
    assert failed_notification.startswith(PACKED_WRITE_FAILED_BEFORE_WRITE_PREFIX)
    assert drained_notification.startswith(PACKED_WRITE_SOURCE_DRAINED_PREFIX)
    assert decode_packed_write_request_notification(request_notification) == request
    assert decode_packed_write_arrived_notification(arrived_notification) == arrived
    assert (
        decode_packed_write_failed_before_write_notification(failed_notification)
        == failed
    )
    assert (
        decode_packed_write_source_drained_notification(drained_notification) == drained
    )


@pytest.mark.cpu_test
def test_notification_encoders_require_typed_messages() -> None:
    with pytest.raises(ValueError, match="request must be typed"):
        encode_packed_write_request_notification("request")
    with pytest.raises(ValueError, match="arrival must be typed"):
        encode_packed_write_arrived_notification("arrived")
    with pytest.raises(ValueError, match="failure must be typed"):
        encode_packed_write_failed_before_write_notification("failed")
    with pytest.raises(ValueError, match="source-drain proof must be typed"):
        encode_packed_write_source_drained_notification("drained")


@pytest.mark.cpu_test
def test_wire_surface_has_only_the_write_terminal_contract() -> None:
    arrived_fields = {
        field.name for field in msgspec.structs.fields(PackedWriteArrived)
    }
    failed_fields = {
        field.name for field in msgspec.structs.fields(PackedWriteFailedBeforeWrite)
    }
    drained_fields = {
        field.name for field in msgspec.structs.fields(PackedWriteSourceDrained)
    }

    assert arrived_fields == {"command"}
    assert failed_fields == {"command", "code", "reason"}
    assert drained_fields == {"anchor", "published_chunk_count"}
    for obsolete_name in (
        "PackedPullReady",
        "PackedPullReadConsumed",
        "PackedPullCancelled",
        "PackedPullFailed",
        "encode_packed_pull_request_notification",
    ):
        assert obsolete_name not in metadata_module.__dict__


@pytest.mark.cpu_test
def test_array_like_wire_field_order_is_explicit_and_stable() -> None:
    def field_names(struct_type: type[msgspec.Struct]) -> tuple[str, ...]:
        return tuple(field.name for field in msgspec.structs.fields(struct_type))

    assert field_names(PackedWriteProducerPoolGeometry) == (
        "registration_generation",
        "base_address",
        "registered_bytes",
        "slot_size_bytes",
        "slot_count",
        "device_id",
        "alignment_bytes",
    )
    assert field_names(PackedWriteConsumerPoolGeometry) == (
        "registration_generation",
        "base_address",
        "registered_bytes",
        "slot_size_bytes",
        "slot_count",
        "source_tp_size",
        "rank_stride_bytes",
        "device_id",
        "alignment_bytes",
    )
    assert field_names(PackedWriteConsumerEndpoint) == (
        "consumer_engine_id",
        "consumer_rank",
        "consumer_tp_size",
        "registration_generation",
        "agent_metadata",
        "consumer_pool",
    )
    assert field_names(PackedWriteChunkIdentity) == (
        "producer_engine_id",
        "producer_request_id",
        "offer_generation",
        "producer_rank",
        "producer_tp_size",
        "consumer_engine_id",
        "consumer_request_id",
        "consumer_rank",
        "consumer_tp_size",
        "source_plan_digest",
        "packed_plan_digest",
        "chunk_plan_digest",
        "chunk_ordinal",
        "chunk_count",
        "valid_token_extent",
        "exact_bytes",
    )
    assert field_names(PackedWriteSlotBinding) == (
        "pool_registration_generation",
        "slot_index",
        "slot_generation",
        "payload_bytes",
    )
    assert field_names(PackedWriteCommand) == ("chunk", "destination")
    assert field_names(PackedWriteRequest) == (
        "command",
        "consumer_endpoint",
        "source_selections",
    )
    assert field_names(PackedWriteSourceDrained) == (
        "anchor",
        "published_chunk_count",
    )


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"published_chunk_count": 0}, "published_chunk_count"),
        ({"published_chunk_count": 4}, "exceeds the canonical plan"),
        ({"published_chunk_count": True}, "published_chunk_count"),
        ({"anchor": _command()}, "anchor must be chunk zero"),
    ],
)
def test_source_drained_rejects_malformed_published_prefix(
    changes: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        msgspec.structs.replace(_source_drained(), **changes)


@pytest.mark.cpu_test
def test_source_drained_decode_rejects_malformed_or_wrong_wire_messages() -> None:
    encoded = msgspec.to_builtins(_source_drained())
    encoded[1] = 0
    with pytest.raises(msgspec.ValidationError, match="published_chunk_count"):
        decode_packed_write_source_drained_notification(
            PACKED_WRITE_SOURCE_DRAINED_PREFIX + msgspec.msgpack.encode(encoded)
        )

    with pytest.raises(msgspec.ValidationError):
        decode_packed_write_source_drained_notification(
            PACKED_WRITE_SOURCE_DRAINED_PREFIX + msgspec.msgpack.encode([])
        )
    with pytest.raises(ValueError, match="not a packed WRITE source-drain proof"):
        decode_packed_write_source_drained_notification(
            PACKED_WRITE_ARRIVED_PREFIX + msgspec.msgpack.encode(encoded)
        )


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("producer_engine_id", "", "producer_engine_id"),
        ("producer_request_id", "", "producer_request_id"),
        ("offer_generation", 0, "offer_generation"),
        ("producer_rank", _SOURCE_TP_SIZE, "outside producer_tp_size"),
        ("producer_tp_size", 0, "producer_tp_size"),
        ("consumer_engine_id", "", "consumer_engine_id"),
        ("consumer_request_id", "", "consumer_request_id"),
        ("consumer_rank", 1, "outside consumer_tp_size"),
        ("consumer_tp_size", 0, "consumer_tp_size"),
        ("source_plan_digest", "A" * 64, "lowercase SHA-256"),
        ("packed_plan_digest", "b" * 63, "lowercase SHA-256"),
        ("chunk_plan_digest", "z" * 64, "lowercase SHA-256"),
        ("chunk_ordinal", 3, "outside chunk_count"),
        ("chunk_count", 0, "chunk_count"),
        ("valid_token_extent", 0, "valid_token_extent"),
        ("exact_bytes", 0, "exact_bytes"),
        ("exact_bytes", True, "exact_bytes"),
    ],
)
def test_chunk_identity_rejects_malformed_namespace_fields(
    field: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        msgspec.structs.replace(_identity(), **{field: value})


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"base_address": 0}, "base_address"),
        ({"base_address": 0x2_0000_0001}, "not aligned"),
        ({"registered_bytes": _RANK_STRIDE_BYTES}, "exactly cover"),
        ({"slot_size_bytes": _RANK_STRIDE_BYTES - 1}, "not aligned"),
        ({"slot_count": 0}, "slot_count"),
        ({"device_id": -1}, "device_id"),
        ({"alignment_bytes": 192}, "power of two"),
        ({"base_address": (1 << 64) - _ALIGNMENT_BYTES}, "uint64"),
    ],
)
def test_producer_pool_rejects_invalid_flat_slot_geometry(
    changes: dict[str, object],
    message: str,
) -> None:
    values: dict[str, object] = {
        "registration_generation": "producer-registration",
        "base_address": 0x2_0000_0000,
        "registered_bytes": _SLOT_COUNT * _RANK_STRIDE_BYTES,
        "slot_size_bytes": _RANK_STRIDE_BYTES,
        "slot_count": _SLOT_COUNT,
        "device_id": 2,
        "alignment_bytes": _ALIGNMENT_BYTES,
    }
    values.update(changes)

    with pytest.raises(ValueError, match=message):
        PackedWriteProducerPoolGeometry(**values)


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"source_tp_size": 0}, "source_tp_size"),
        ({"rank_stride_bytes": 0}, "rank_stride_bytes"),
        ({"rank_stride_bytes": _RANK_STRIDE_BYTES + 1}, "not aligned"),
        (
            {
                "slot_size_bytes": _CONSUMER_SLOT_SIZE_BYTES // 2,
                "registered_bytes": _SLOT_COUNT * (_CONSUMER_SLOT_SIZE_BYTES // 2),
            },
            "every source-rank slab",
        ),
        ({"registered_bytes": _CONSUMER_SLOT_SIZE_BYTES}, "exactly cover"),
        ({"alignment_bytes": 0}, "alignment_bytes"),
    ],
)
def test_consumer_pool_rejects_invalid_rank_major_geometry(
    changes: dict[str, object],
    message: str,
) -> None:
    values: dict[str, object] = {
        "registration_generation": "consumer-registration",
        "base_address": 0x4_0000_0000,
        "registered_bytes": _SLOT_COUNT * _CONSUMER_SLOT_SIZE_BYTES,
        "slot_size_bytes": _CONSUMER_SLOT_SIZE_BYTES,
        "slot_count": _SLOT_COUNT,
        "source_tp_size": _SOURCE_TP_SIZE,
        "rank_stride_bytes": _RANK_STRIDE_BYTES,
        "device_id": 5,
        "alignment_bytes": _ALIGNMENT_BYTES,
    }
    values.update(changes)

    with pytest.raises(ValueError, match=message):
        PackedWriteConsumerPoolGeometry(**values)


@pytest.mark.cpu_test
def test_endpoint_binds_agent_identity_and_consumer_pool_generation() -> None:
    endpoint = _endpoint()

    assert endpoint.consumer_pool == _consumer_pool()
    with pytest.raises(ValueError, match="agent_metadata"):
        msgspec.structs.replace(endpoint, agent_metadata=b"")
    with pytest.raises(ValueError, match="outside consumer_tp_size"):
        msgspec.structs.replace(endpoint, consumer_rank=1)
    with pytest.raises(ValueError, match="generations differ"):
        msgspec.structs.replace(
            endpoint,
            consumer_pool=msgspec.structs.replace(
                endpoint.consumer_pool,
                registration_generation="other-registration",
            ),
        )


@pytest.mark.cpu_test
def test_request_rejects_endpoint_identity_and_source_tp_mismatches() -> None:
    with pytest.raises(ValueError, match="endpoint does not match"):
        _request(
            consumer_endpoint=msgspec.structs.replace(
                _endpoint(),
                consumer_engine_id="other-decode",
            )
        )

    tp2_pool = PackedWriteConsumerPoolGeometry(
        registration_generation="consumer-registration",
        base_address=0x4_0000_0000,
        registered_bytes=_SLOT_COUNT * 2 * _RANK_STRIDE_BYTES,
        slot_size_bytes=2 * _RANK_STRIDE_BYTES,
        slot_count=_SLOT_COUNT,
        source_tp_size=2,
        rank_stride_bytes=_RANK_STRIDE_BYTES,
        device_id=5,
        alignment_bytes=_ALIGNMENT_BYTES,
    )
    with pytest.raises(ValueError, match="source TP size"):
        _request(
            consumer_endpoint=msgspec.structs.replace(
                _endpoint(),
                consumer_pool=tp2_pool,
            )
        )


@pytest.mark.cpu_test
def test_destination_binding_rejects_stale_or_unsafe_slot_leases() -> None:
    pool = _consumer_pool()
    validate_packed_write_slot_binding(_destination(), pool)

    with pytest.raises(ValueError, match="stale pool generation"):
        validate_packed_write_slot_binding(
            msgspec.structs.replace(
                _destination(),
                pool_registration_generation="stale-registration",
            ),
            pool,
        )
    with pytest.raises(ValueError, match="outside the pool"):
        validate_packed_write_slot_binding(
            msgspec.structs.replace(_destination(), slot_index=_SLOT_COUNT),
            pool,
        )
    with pytest.raises(ValueError, match="exceeds its source-rank slab"):
        validate_packed_write_slot_binding(
            msgspec.structs.replace(
                _destination(),
                payload_bytes=_RANK_STRIDE_BYTES + 1,
            ),
            pool,
        )


@pytest.mark.cpu_test
def test_command_requires_one_exact_payload_size() -> None:
    with pytest.raises(ValueError, match="byte count does not match"):
        _command(
            destination=msgspec.structs.replace(
                _destination(),
                payload_bytes=_identity().exact_bytes - _ALIGNMENT_BYTES,
            )
        )


@pytest.mark.cpu_test
def test_destination_address_is_derived_from_slot_and_producer_rank() -> None:
    pool = _consumer_pool()
    slot_base = pool.base_address + _destination().slot_index * pool.slot_size_bytes

    for producer_rank in range(_SOURCE_TP_SIZE):
        chunk = msgspec.structs.replace(_identity(), producer_rank=producer_rank)
        command = _command(chunk=chunk)
        assert packed_write_destination_address(command, pool) == (
            slot_base + producer_rank * pool.rank_stride_bytes
        )


@pytest.mark.cpu_test
def test_request_allows_empty_groups_but_rejects_noncanonical_or_empty_work() -> None:
    request = _request(
        source_selections=(
            PackedWriteSourceSelection(
                group_index=0,
                source_position_start=8,
                position_count=0,
            ),
            PackedWriteSourceSelection(
                group_index=1,
                source_position_start=4,
                position_count=60,
            ),
        )
    )
    assert request.source_selections[0].position_count == 0

    with pytest.raises(ValueError, match="canonical groups"):
        _request(
            source_selections=(
                PackedWriteSourceSelection(
                    group_index=1,
                    source_position_start=0,
                    position_count=1,
                ),
            )
        )
    with pytest.raises(ValueError, match="contain no positions"):
        _request(
            source_selections=(
                PackedWriteSourceSelection(
                    group_index=0,
                    source_position_start=8,
                    position_count=0,
                ),
            )
        )


@pytest.mark.cpu_test
def test_typed_decode_rejects_semantically_invalid_or_untyped_requests() -> None:
    encoded = msgspec.to_builtins(_request())
    encoded[0][0][12] = encoded[0][0][13]
    notification = PACKED_WRITE_REQUEST_PREFIX + msgspec.msgpack.encode(encoded)

    with pytest.raises(msgspec.ValidationError, match="chunk_ordinal"):
        decode_packed_write_request_notification(notification)
    with pytest.raises(msgspec.ValidationError):
        decode_packed_write_request_notification(
            PACKED_WRITE_REQUEST_PREFIX + msgspec.msgpack.encode({"command": []})
        )
    with pytest.raises(msgspec.ValidationError):
        decode_packed_write_request_notification(
            PACKED_WRITE_REQUEST_PREFIX
            + msgspec.msgpack.encode(msgspec.to_builtins(_request()) + ["extra"])
        )


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "notification",
    [
        PACKED_WRITE_ARRIVED_PREFIX + b"\x90",
        b"PACKED_PULL_REQUEST:\x90",
        bytearray(PACKED_WRITE_REQUEST_PREFIX + b"\x90"),
    ],
)
def test_request_decode_rejects_wrong_or_obsolete_prefixes(
    notification: bytes,
) -> None:
    with pytest.raises(ValueError, match="not a packed WRITE request"):
        decode_packed_write_request_notification(notification)


@pytest.mark.cpu_test
def test_request_replay_requires_every_authenticated_field_to_be_identical() -> None:
    request = _request()
    validate_packed_write_request_replay(request, request)

    changed_chunk = msgspec.structs.replace(
        request.command.chunk,
        chunk_ordinal=0,
    )
    changed_endpoint = msgspec.structs.replace(
        request.consumer_endpoint,
        agent_metadata=b"different-decoder-agent-metadata",
    )
    changed_pool = msgspec.structs.replace(
        request.consumer_endpoint.consumer_pool,
        base_address=request.consumer_endpoint.consumer_pool.base_address
        + _ALIGNMENT_BYTES,
    )
    changed_generation_pool = msgspec.structs.replace(
        request.consumer_endpoint.consumer_pool,
        registration_generation="next-consumer-registration",
    )
    changed_generation_destination = msgspec.structs.replace(
        request.command.destination,
        pool_registration_generation="next-consumer-registration",
    )
    changed_generation_command = _command(
        destination=changed_generation_destination,
    )
    changed_generation_endpoint = msgspec.structs.replace(
        request.consumer_endpoint,
        registration_generation="next-consumer-registration",
        consumer_pool=changed_generation_pool,
    )
    replays = (
        _request(command=_command(chunk=changed_chunk)),
        _request(
            command=_command(
                destination=msgspec.structs.replace(
                    request.command.destination,
                    slot_generation=request.command.destination.slot_generation + 1,
                )
            )
        ),
        _request(consumer_endpoint=changed_endpoint),
        _request(
            consumer_endpoint=msgspec.structs.replace(
                request.consumer_endpoint,
                consumer_pool=changed_pool,
            )
        ),
        _request(
            command=changed_generation_command,
            consumer_endpoint=changed_generation_endpoint,
        ),
        _request(
            source_selections=(
                msgspec.structs.replace(
                    request.source_selections[0],
                    source_position_start=request.source_selections[
                        0
                    ].source_position_start
                    + 1,
                ),
                request.source_selections[1],
            )
        ),
    )
    for replay in replays:
        with pytest.raises(ValueError, match="changed authenticated fields"):
            validate_packed_write_request_replay(request, replay)


@pytest.mark.cpu_test
def test_terminals_require_the_exact_published_command() -> None:
    request = _request()
    arrived = PackedWriteArrived(command=request.command)
    failed = PackedWriteFailedBeforeWrite(
        command=request.command,
        code=PackedWriteFailureCode.WRITE_PREPARE_FAILED,
        reason="NIXL descriptor preparation failed",
    )
    validate_packed_write_arrived(request, arrived)
    validate_packed_write_failed_before_write(request, failed)

    changed_command = _command(
        destination=msgspec.structs.replace(
            request.command.destination,
            slot_generation=request.command.destination.slot_generation + 1,
        )
    )
    with pytest.raises(ValueError, match="arrival does not match"):
        validate_packed_write_arrived(
            request,
            PackedWriteArrived(command=changed_command),
        )
    with pytest.raises(ValueError, match="failure does not match"):
        validate_packed_write_failed_before_write(
            request,
            PackedWriteFailedBeforeWrite(
                command=changed_command,
                code=PackedWriteFailureCode.WRITE_PREPARE_FAILED,
                reason="NIXL descriptor preparation failed",
            ),
        )


@pytest.mark.cpu_test
def test_failed_before_write_requires_a_bounded_typed_reason() -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        PackedWriteFailedBeforeWrite(
            command=_command(),
            code=PackedWriteFailureCode.INTERNAL_ERROR,
            reason="",
        )
    with pytest.raises(ValueError, match="exceeds 1024"):
        PackedWriteFailedBeforeWrite(
            command=_command(),
            code=PackedWriteFailureCode.INTERNAL_ERROR,
            reason="x" * 1025,
        )
    with pytest.raises(ValueError, match="code must be typed"):
        PackedWriteFailedBeforeWrite(
            command=_command(),
            code="internal_error",
            reason="failure",
        )

    encoded = msgspec.to_builtins(
        PackedWriteFailedBeforeWrite(
            command=_command(),
            code=PackedWriteFailureCode.INTERNAL_ERROR,
            reason="failure",
        )
    )
    encoded[1] = "post_write_failure"
    with pytest.raises(msgspec.ValidationError):
        decode_packed_write_failed_before_write_notification(
            PACKED_WRITE_FAILED_BEFORE_WRITE_PREFIX + msgspec.msgpack.encode(encoded)
        )
