"""Typed control messages for the Target 2 fixed-byte transport gate."""

import base64
import binascii
import math
from dataclasses import dataclass
from typing import ClassVar, Self

from tools.gemma4_pd.nixl_micro_rig.protocol import ControlPayload, ProtocolError


def _object(value: object, *, keys: frozenset[str], label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(type(key) is str for key in value):
        raise ProtocolError(f"{label} must be an object")
    if frozenset(value) != keys:
        raise ProtocolError(f"{label} keys differ: {frozenset(value) ^ keys}")
    return value


def _integer(value: object, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ProtocolError(f"{label} must be an integer at least {minimum}")
    return value


def _text(value: object, *, label: str) -> str:
    if type(value) is not str or len(value) == 0:
        raise ProtocolError(f"{label} must be a non-empty string")
    return value


def _boolean(value: object, *, label: str) -> bool:
    if type(value) is not bool:
        raise ProtocolError(f"{label} must be a boolean")
    return value


def _finite(value: object, *, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
    ):
        raise ProtocolError(f"{label} must be a finite non-negative number")
    return float(value)


def _integer_tuple(
    value: object, *, label: str, minimum: int = 0, nonempty: bool = True
) -> tuple[int, ...]:
    if not isinstance(value, list) or (nonempty and len(value) == 0):
        raise ProtocolError(f"{label} must be a non-empty array")
    return tuple(
        _integer(item, label=f"{label}[{index}]", minimum=minimum)
        for index, item in enumerate(value)
    )


def _text_tuple(value: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) == 0:
        raise ProtocolError(f"{label} must be a non-empty array")
    return tuple(
        _text(item, label=f"{label}[{index}]") for index, item in enumerate(value)
    )


def _base64(value: object, *, label: str) -> str:
    text = _text(value, label=label)
    try:
        decoded = base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ProtocolError(f"{label} must contain canonical base64") from error
    if len(decoded) == 0 or base64.b64encode(decoded).decode() != text:
        raise ProtocolError(f"{label} must contain non-empty canonical base64")
    return text


def _native_telemetry(value: object) -> dict[str, object]:
    payload = _object(
        value,
        keys=frozenset(
            {
                "backend",
                "start_time_us",
                "post_duration_us",
                "transfer_duration_us",
                "total_bytes",
                "descriptor_count",
            }
        ),
        label="native telemetry",
    )
    return {
        "backend": _text(payload["backend"], label="native_telemetry.backend"),
        "start_time_us": _finite(
            payload["start_time_us"], label="native_telemetry.start_time_us"
        ),
        "post_duration_us": _finite(
            payload["post_duration_us"], label="native_telemetry.post_duration_us"
        ),
        "transfer_duration_us": _finite(
            payload["transfer_duration_us"],
            label="native_telemetry.transfer_duration_us",
        ),
        "total_bytes": _integer(
            payload["total_bytes"], label="native_telemetry.total_bytes", minimum=1
        ),
        "descriptor_count": _integer(
            payload["descriptor_count"],
            label="native_telemetry.descriptor_count",
            minimum=1,
        ),
    }


@dataclass(frozen=True)
class GateProducerHelloPayload(ControlPayload):
    """Advertise source registrations and the bounded producer pack pool."""

    message_type: ClassVar[str] = "target2_producer_hello"

    agent_metadata: str
    source_base_addresses: tuple[int, ...]
    source_region_bytes: tuple[int, ...]
    pack_slot_base_addresses: tuple[int, ...]
    pack_slot_bytes: int
    logical_device: int

    def to_json(self) -> dict[str, object]:
        """Encode the producer advertisement.

        :returns: Exact JSON payload.
        """
        return {
            "agent_metadata": self.agent_metadata,
            "source_base_addresses": list(self.source_base_addresses),
            "source_region_bytes": list(self.source_region_bytes),
            "pack_slot_base_addresses": list(self.pack_slot_base_addresses),
            "pack_slot_bytes": self.pack_slot_bytes,
            "logical_device": self.logical_device,
        }

    @classmethod
    def from_json(cls, value: object) -> Self:
        """Decode a producer advertisement.

        :param value: Decoded JSON value.
        :returns: Validated payload.
        """
        payload = _object(
            value,
            keys=frozenset(
                {
                    "agent_metadata",
                    "source_base_addresses",
                    "source_region_bytes",
                    "pack_slot_base_addresses",
                    "pack_slot_bytes",
                    "logical_device",
                }
            ),
            label="target2 producer hello",
        )
        source_bases = _integer_tuple(
            payload["source_base_addresses"],
            label="source_base_addresses",
            minimum=1,
        )
        source_bytes = _integer_tuple(
            payload["source_region_bytes"],
            label="source_region_bytes",
            minimum=1,
        )
        pack_bases = _integer_tuple(
            payload["pack_slot_base_addresses"],
            label="pack_slot_base_addresses",
            minimum=1,
        )
        if len(source_bases) != len(source_bytes):
            raise ProtocolError("source address and length counts differ")
        if len(pack_bases) != 2:
            raise ProtocolError("producer pack pool must contain exactly two slots")
        return cls(
            agent_metadata=_base64(payload["agent_metadata"], label="agent_metadata"),
            source_base_addresses=source_bases,
            source_region_bytes=source_bytes,
            pack_slot_base_addresses=pack_bases,
            pack_slot_bytes=_integer(
                payload["pack_slot_bytes"], label="pack_slot_bytes", minimum=1
            ),
            logical_device=_integer(payload["logical_device"], label="logical_device"),
        )


@dataclass(frozen=True)
class GateConsumerHelloPayload(ControlPayload):
    """Advertise complete rank-major decoder slots for packed WRITE."""

    message_type: ClassVar[str] = "target2_consumer_hello"

    agent_metadata: str
    receive_slot_base_addresses: tuple[int, ...]
    receive_slot_bytes: int
    logical_device: int

    def to_json(self) -> dict[str, object]:
        """Encode the consumer advertisement.

        :returns: Exact JSON payload.
        """
        return {
            "agent_metadata": self.agent_metadata,
            "receive_slot_base_addresses": list(self.receive_slot_base_addresses),
            "receive_slot_bytes": self.receive_slot_bytes,
            "logical_device": self.logical_device,
        }

    @classmethod
    def from_json(cls, value: object) -> Self:
        """Decode a consumer advertisement.

        :param value: Decoded JSON value.
        :returns: Validated payload.
        """
        payload = _object(
            value,
            keys=frozenset(
                {
                    "agent_metadata",
                    "receive_slot_base_addresses",
                    "receive_slot_bytes",
                    "logical_device",
                }
            ),
            label="target2 consumer hello",
        )
        receive_bases = _integer_tuple(
            payload["receive_slot_base_addresses"],
            label="receive_slot_base_addresses",
            minimum=1,
        )
        if len(receive_bases) != 2:
            raise ProtocolError("consumer receive pool must contain exactly two slots")
        return cls(
            agent_metadata=_base64(payload["agent_metadata"], label="agent_metadata"),
            receive_slot_base_addresses=receive_bases,
            receive_slot_bytes=_integer(
                payload["receive_slot_bytes"], label="receive_slot_bytes", minimum=1
            ),
            logical_device=_integer(payload["logical_device"], label="logical_device"),
        )


@dataclass(frozen=True)
class GateBatchPreparePayload(ControlPayload):
    """Identify one independently warmed or measured gate batch."""

    message_type: ClassVar[str] = "target2_batch_prepare"

    case_name: str
    arm: str
    run_count: int
    expected_descriptors_per_rank: int
    chunk_bytes: int
    in_flight_depth: int
    batch_index: int
    measured: bool
    payload_iterations: tuple[int, ...]

    def to_json(self) -> dict[str, object]:
        """Encode the batch contract.

        :returns: Exact JSON payload.
        """
        return {
            "case_name": self.case_name,
            "arm": self.arm,
            "run_count": self.run_count,
            "expected_descriptors_per_rank": self.expected_descriptors_per_rank,
            "chunk_bytes": self.chunk_bytes,
            "in_flight_depth": self.in_flight_depth,
            "batch_index": self.batch_index,
            "measured": self.measured,
            "payload_iterations": list(self.payload_iterations),
        }

    @classmethod
    def from_json(cls, value: object) -> Self:
        """Decode a batch contract.

        :param value: Decoded JSON value.
        :returns: Validated payload.
        """
        payload = _object(
            value,
            keys=frozenset(
                {
                    "case_name",
                    "arm",
                    "run_count",
                    "expected_descriptors_per_rank",
                    "chunk_bytes",
                    "in_flight_depth",
                    "batch_index",
                    "measured",
                    "payload_iterations",
                }
            ),
            label="target2 batch prepare",
        )
        depth = _integer(payload["in_flight_depth"], label="in_flight_depth", minimum=1)
        iterations = _integer_tuple(
            payload["payload_iterations"], label="payload_iterations"
        )
        if len(iterations) != depth or len(set(iterations)) != depth:
            raise ProtocolError(
                "payload_iterations must uniquely cover the in-flight batch"
            )
        return cls(
            case_name=_text(payload["case_name"], label="case_name"),
            arm=_text(payload["arm"], label="arm"),
            run_count=_integer(payload["run_count"], label="run_count", minimum=1),
            expected_descriptors_per_rank=_integer(
                payload["expected_descriptors_per_rank"],
                label="expected_descriptors_per_rank",
                minimum=1,
            ),
            chunk_bytes=_integer(payload["chunk_bytes"], label="chunk_bytes"),
            in_flight_depth=depth,
            batch_index=_integer(payload["batch_index"], label="batch_index"),
            measured=_boolean(payload["measured"], label="measured"),
            payload_iterations=iterations,
        )


@dataclass(frozen=True)
class GateBatchReadyPayload(ControlPayload):
    """Prove that every disjoint request source is populated and verified."""

    message_type: ClassVar[str] = "target2_batch_ready"

    source_plan_digests: tuple[str, ...]

    def to_json(self) -> dict[str, object]:
        """Encode source readiness.

        :returns: Exact JSON payload.
        """
        return {"source_plan_digests": list(self.source_plan_digests)}

    @classmethod
    def from_json(cls, value: object) -> Self:
        """Decode source readiness.

        :param value: Decoded JSON value.
        :returns: Validated payload.
        """
        payload = _object(
            value,
            keys=frozenset({"source_plan_digests"}),
            label="target2 batch ready",
        )
        digests = _text_tuple(
            payload["source_plan_digests"], label="source_plan_digests"
        )
        if any(
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            for digest in digests
        ):
            raise ProtocolError("source plan digests must contain lowercase SHA-256")
        return cls(source_plan_digests=digests)


@dataclass(frozen=True)
class GatePackCommandPayload(ControlPayload):
    """Request one exact logical chunk in one bounded slot."""

    message_type: ClassVar[str] = "target2_pack_command"

    request_index: int
    chunk_index: int
    slot_index: int
    notification_id: str

    def to_json(self) -> dict[str, object]:
        """Encode the pack command.

        :returns: Exact JSON payload.
        """
        return {
            "request_index": self.request_index,
            "chunk_index": self.chunk_index,
            "slot_index": self.slot_index,
            "notification_id": self.notification_id,
        }

    @classmethod
    def from_json(cls, value: object) -> Self:
        """Decode a pack command.

        :param value: Decoded JSON value.
        :returns: Validated payload.
        """
        payload = _object(
            value,
            keys=frozenset(
                {"request_index", "chunk_index", "slot_index", "notification_id"}
            ),
            label="target2 pack command",
        )
        slot_index = _integer(payload["slot_index"], label="slot_index")
        if slot_index >= 2:
            raise ProtocolError("slot_index must address the two-slot pool")
        return cls(
            request_index=_integer(payload["request_index"], label="request_index"),
            chunk_index=_integer(payload["chunk_index"], label="chunk_index"),
            slot_index=slot_index,
            notification_id=_base64(
                payload["notification_id"], label="notification_id"
            ),
        )


@dataclass(frozen=True)
class GatePackReadyPayload(ControlPayload):
    """Report a packed slot ready for READ or a completed WRITE."""

    message_type: ClassVar[str] = "target2_pack_ready"

    request_index: int
    chunk_index: int
    slot_index: int
    slot_generation: int
    payload_bytes: int
    pack_gpu_ms: float
    native_done: bool
    native_telemetry: dict[str, object]

    def to_json(self) -> dict[str, object]:
        """Encode producer packing and optional WRITE telemetry.

        :returns: Exact JSON payload.
        """
        return {
            "request_index": self.request_index,
            "chunk_index": self.chunk_index,
            "slot_index": self.slot_index,
            "slot_generation": self.slot_generation,
            "payload_bytes": self.payload_bytes,
            "pack_gpu_ms": self.pack_gpu_ms,
            "native_done": self.native_done,
            "native_telemetry": self.native_telemetry,
        }

    @classmethod
    def from_json(cls, value: object) -> Self:
        """Decode producer packing evidence.

        :param value: Decoded JSON value.
        :returns: Validated payload.
        """
        payload = _object(
            value,
            keys=frozenset(
                {
                    "request_index",
                    "chunk_index",
                    "slot_index",
                    "slot_generation",
                    "payload_bytes",
                    "pack_gpu_ms",
                    "native_done",
                    "native_telemetry",
                }
            ),
            label="target2 pack ready",
        )
        slot_index = _integer(payload["slot_index"], label="slot_index")
        if slot_index >= 2:
            raise ProtocolError("slot_index must address the two-slot pool")
        native_done = _boolean(payload["native_done"], label="native_done")
        raw_telemetry = payload["native_telemetry"]
        if not isinstance(raw_telemetry, dict):
            raise ProtocolError("native_telemetry must be an object")
        if native_done:
            telemetry = _native_telemetry(raw_telemetry)
        else:
            if len(raw_telemetry) > 0:
                raise ProtocolError("non-native readiness must not carry telemetry")
            telemetry = {}
        return cls(
            request_index=_integer(payload["request_index"], label="request_index"),
            chunk_index=_integer(payload["chunk_index"], label="chunk_index"),
            slot_index=slot_index,
            slot_generation=_integer(
                payload["slot_generation"], label="slot_generation", minimum=1
            ),
            payload_bytes=_integer(
                payload["payload_bytes"], label="payload_bytes", minimum=1
            ),
            pack_gpu_ms=_finite(payload["pack_gpu_ms"], label="pack_gpu_ms"),
            native_done=native_done,
            native_telemetry=telemetry,
        )


@dataclass(frozen=True)
class GateBatchCompletePayload(ControlPayload):
    """Declare decoder completion after every native and device operation."""

    message_type: ClassVar[str] = "target2_batch_complete"

    success: bool

    def to_json(self) -> dict[str, object]:
        """Encode batch completion.

        :returns: Exact JSON payload.
        """
        return {"success": self.success}

    @classmethod
    def from_json(cls, value: object) -> Self:
        """Decode batch completion.

        :param value: Decoded JSON value.
        :returns: Validated payload.
        """
        payload = _object(
            value,
            keys=frozenset({"success"}),
            label="target2 batch complete",
        )
        return cls(success=_boolean(payload["success"], label="success"))


@dataclass(frozen=True)
class GateBatchPostPayload(ControlPayload):
    """Report producer source immutability and slot release evidence."""

    message_type: ClassVar[str] = "target2_batch_post"

    sources_verified: bool
    notifications_seen: int
    final_slot_states: tuple[str, ...]

    def to_json(self) -> dict[str, object]:
        """Encode post-transfer producer evidence.

        :returns: Exact JSON payload.
        """
        return {
            "sources_verified": self.sources_verified,
            "notifications_seen": self.notifications_seen,
            "final_slot_states": list(self.final_slot_states),
        }

    @classmethod
    def from_json(cls, value: object) -> Self:
        """Decode post-transfer producer evidence.

        :param value: Decoded JSON value.
        :returns: Validated payload.
        """
        payload = _object(
            value,
            keys=frozenset(
                {"sources_verified", "notifications_seen", "final_slot_states"}
            ),
            label="target2 batch post",
        )
        states = _text_tuple(payload["final_slot_states"], label="final_slot_states")
        if len(states) != 2:
            raise ProtocolError("batch post must report exactly two slot states")
        return cls(
            sources_verified=_boolean(
                payload["sources_verified"], label="sources_verified"
            ),
            notifications_seen=_integer(
                payload["notifications_seen"], label="notifications_seen"
            ),
            final_slot_states=states,
        )
