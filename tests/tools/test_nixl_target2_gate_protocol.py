import base64

import pytest

from tools.gemma4_pd.nixl_micro_rig.protocol import ProtocolError
from tools.gemma4_pd.nixl_micro_rig.target2_gate_protocol import (
    GateBatchCompletePayload,
    GateBatchPostPayload,
    GateBatchPreparePayload,
    GateBatchReadyPayload,
    GateConsumerHelloPayload,
    GatePackCommandPayload,
    GatePackReadyPayload,
    GateProducerHelloPayload,
)


def _opaque(value: bytes) -> str:
    return base64.b64encode(value).decode()


@pytest.mark.parametrize(
    "payload",
    [
        GateProducerHelloPayload(
            agent_metadata=_opaque(b"producer"),
            source_base_addresses=(100, 200),
            source_region_bytes=(10, 20),
            pack_slot_base_addresses=(300, 400),
            pack_slot_bytes=64 * 1024 * 1024,
            logical_device=3,
        ),
        GateConsumerHelloPayload(
            agent_metadata=_opaque(b"consumer"),
            receive_slot_base_addresses=(500, 600),
            receive_slot_bytes=64 * 1024 * 1024,
            logical_device=0,
        ),
        GateBatchPreparePayload(
            case_name="packed_read-observed_c64_fragmentation-64mib-q2",
            arm="packed_read",
            run_count=424,
            expected_descriptors_per_rank=2120,
            chunk_bytes=64 * 1024 * 1024,
            in_flight_depth=2,
            batch_index=1,
            measured=True,
            payload_iterations=(1000, 1001),
        ),
        GateBatchReadyPayload(source_plan_digests=("a" * 64, "b" * 64)),
        GatePackCommandPayload(
            request_index=1,
            chunk_index=3,
            slot_index=1,
            notification_id=_opaque(b"notification"),
        ),
        GatePackReadyPayload(
            request_index=1,
            chunk_index=3,
            slot_index=1,
            slot_generation=2,
            payload_bytes=61,
            pack_gpu_ms=0.25,
            native_done=True,
            native_telemetry={
                "backend": "UCX",
                "start_time_us": 1,
                "post_duration_us": 2,
                "transfer_duration_us": 3,
                "total_bytes": 61,
                "descriptor_count": 1,
            },
        ),
        GateBatchCompletePayload(success=True),
        GateBatchPostPayload(
            sources_verified=True,
            notifications_seen=8,
            final_slot_states=("free", "free"),
        ),
    ],
)
def test_target2_payloads_round_trip_exactly(payload: object) -> None:
    assert type(payload).from_json(payload.to_json()) == payload


def test_target2_prepare_requires_unique_iteration_per_request() -> None:
    payload = GateBatchPreparePayload(
        case_name="case",
        arm="packed_read",
        run_count=1,
        expected_descriptors_per_rank=10,
        chunk_bytes=64 * 1024 * 1024,
        in_flight_depth=2,
        batch_index=0,
        measured=False,
        payload_iterations=(7, 7),
    )

    with pytest.raises(ProtocolError, match="uniquely cover"):
        GateBatchPreparePayload.from_json(payload.to_json())


def test_target2_ready_binds_native_completion_to_exact_telemetry() -> None:
    payload = GatePackReadyPayload(
        request_index=0,
        chunk_index=0,
        slot_index=0,
        slot_generation=1,
        payload_bytes=1024,
        pack_gpu_ms=0.1,
        native_done=False,
        native_telemetry={"backend": "UCX"},
    )

    with pytest.raises(ProtocolError, match="must not carry telemetry"):
        GatePackReadyPayload.from_json(payload.to_json())


def test_target2_payloads_reject_pool_escape_and_unknown_keys() -> None:
    command = GatePackCommandPayload(
        request_index=0,
        chunk_index=0,
        slot_index=0,
        notification_id=_opaque(b"notification"),
    ).to_json()
    command["slot_index"] = 2
    with pytest.raises(ProtocolError, match="two-slot pool"):
        GatePackCommandPayload.from_json(command)

    hello = GateConsumerHelloPayload(
        agent_metadata=_opaque(b"consumer"),
        receive_slot_base_addresses=(100, 200),
        receive_slot_bytes=1024,
        logical_device=0,
    ).to_json()
    hello["unexpected"] = True
    with pytest.raises(ProtocolError, match="keys differ"):
        GateConsumerHelloPayload.from_json(hello)
