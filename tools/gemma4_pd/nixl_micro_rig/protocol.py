"""Strict, length-prefixed JSON protocol for independent micro-rig roles."""

import base64
import binascii
import json
import math
import socket
import struct
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar, Self, TypeVar


class ProtocolError(RuntimeError):
    """Report malformed, stale, duplicated, or truncated control traffic."""


_HEADER = struct.Struct("!Q")
_MAX_MESSAGE_BYTES = 64 * 1024 * 1024
_PROTOCOL_VERSION = 1
_ROLES = frozenset({"producer", "consumer"})
_ENVELOPE_KEYS = {
    "protocol_version",
    "type",
    "run_id",
    "config_fingerprint",
    "transport_arm",
    "iteration",
    "sender_role",
    "sender_rank",
    "sequence",
    "payload",
}


def _require_object(
    value: object, *, keys: frozenset[str], label: str
) -> dict[str, object]:
    """Require a JSON object with an exact key vocabulary.

    :param value: Decoded JSON value.
    :param keys: Exact required keys.
    :param label: Diagnostic field path.
    :returns: Validated object.
    :raises ProtocolError: If the value or keys differ.
    """
    if not isinstance(value, dict) or not all(type(key) is str for key in value):
        raise ProtocolError(f"{label} must be a JSON object")
    actual_keys = frozenset(value.keys())
    if actual_keys != keys:
        raise ProtocolError(f"{label} keys differ: {actual_keys ^ keys}")
    return value


def _require_json_object(value: object, *, label: str) -> dict[str, object]:
    """Require an open-schema object containing only strict JSON values.

    :param value: Decoded JSON value.
    :param label: Diagnostic field path.
    :returns: Validated object.
    :raises ProtocolError: If the value is not a strict JSON object.
    """
    if not isinstance(value, dict) or not all(type(key) is str for key in value):
        raise ProtocolError(f"{label} must be a JSON object")
    for key, item in value.items():
        _require_json_value(item, label=f"{label}.{key}")
    return value


def _require_json_value(value: object, *, label: str) -> None:
    """Reject Python or non-finite values outside strict JSON.

    :param value: Candidate JSON value.
    :param label: Diagnostic field path.
    :raises ProtocolError: If the value is not strict JSON.
    """
    if value is None or type(value) in {bool, int, str}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ProtocolError(f"{label} must be finite")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _require_json_value(item, label=f"{label}[{index}]")
        return
    if isinstance(value, dict) and all(type(key) is str for key in value):
        for key, item in value.items():
            _require_json_value(item, label=f"{label}.{key}")
        return
    raise ProtocolError(f"{label} is not a strict JSON value")


def _require_string(value: object, *, label: str, allow_empty: bool = False) -> str:
    """Require an exact JSON string.

    :param value: Decoded JSON value.
    :param label: Diagnostic field path.
    :param allow_empty: Whether the empty string is valid.
    :returns: Validated string.
    :raises ProtocolError: If the value is not an allowed string.
    """
    if type(value) is not str:
        raise ProtocolError(f"{label} must be a string")
    if not allow_empty and len(value) == 0:
        raise ProtocolError(f"{label} must not be empty")
    return value


def _require_integer(value: object, *, label: str, minimum: int | None = None) -> int:
    """Require an exact JSON integer, excluding booleans.

    :param value: Decoded JSON value.
    :param label: Diagnostic field path.
    :param minimum: Optional inclusive lower bound.
    :returns: Validated integer.
    :raises ProtocolError: If the value is not an allowed integer.
    """
    if type(value) is not int:
        raise ProtocolError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ProtocolError(f"{label} must be at least {minimum}")
    return value


def _require_boolean(value: object, *, label: str) -> bool:
    """Require an exact JSON boolean.

    :param value: Decoded JSON value.
    :param label: Diagnostic field path.
    :returns: Validated boolean.
    :raises ProtocolError: If the value is not a boolean.
    """
    if type(value) is not bool:
        raise ProtocolError(f"{label} must be a boolean")
    return value


def _require_base64(value: object, *, label: str) -> str:
    """Require non-empty canonical base64 text.

    :param value: Decoded JSON value.
    :param label: Diagnostic field path.
    :returns: Validated base64 string.
    :raises ProtocolError: If the value is malformed.
    """
    text = _require_string(value, label=label)
    try:
        decoded = base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ProtocolError(f"{label} must be valid base64") from error
    if len(decoded) == 0 or base64.b64encode(decoded).decode() != text:
        raise ProtocolError(f"{label} must be non-empty canonical base64")
    return text


class ControlPayload(ABC):
    """Define one exact control-message payload schema."""

    message_type: ClassVar[str]

    @abstractmethod
    def to_json(self) -> dict[str, object]:
        """Encode the payload as a strict JSON object.

        :returns: JSON object ready for framing.
        """

    @classmethod
    @abstractmethod
    def from_json(cls, value: object) -> Self:
        """Validate and decode one payload.

        :param value: Decoded JSON value.
        :returns: Typed payload.
        :raises ProtocolError: If the payload differs from the schema.
        """


@dataclass(frozen=True)
class DigestRow:
    """Compact source-observation identity and digest."""

    source_rank: int
    region_index: int
    group_index: int
    source_position: int
    remote_block_id: int
    digest: str

    def to_json(self) -> list[int | str]:
        """Encode one compact digest row.

        :returns: Six-element JSON array.
        """
        return [
            self.source_rank,
            self.region_index,
            self.group_index,
            self.source_position,
            self.remote_block_id,
            self.digest,
        ]

    @classmethod
    def from_json(cls, value: object) -> Self:
        """Decode one compact digest row.

        :param value: Decoded JSON value.
        :returns: Validated digest row.
        :raises ProtocolError: If the row differs from its exact schema.
        """
        if not isinstance(value, list) or len(value) != 6:
            raise ProtocolError("prepared.digests[] must be a six-element array")
        digest = _require_string(value[5], label="prepared.digests[].digest")
        if len(digest) != 32 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ProtocolError(
                "prepared.digests[].digest must be lowercase BLAKE2b-128 hex"
            )
        return cls(
            source_rank=_require_integer(
                value[0], label="prepared.digests[].source_rank", minimum=0
            ),
            region_index=_require_integer(
                value[1], label="prepared.digests[].region_index", minimum=0
            ),
            group_index=_require_integer(
                value[2], label="prepared.digests[].group_index", minimum=0
            ),
            source_position=_require_integer(
                value[3], label="prepared.digests[].source_position", minimum=0
            ),
            remote_block_id=_require_integer(
                value[4], label="prepared.digests[].remote_block_id", minimum=0
            ),
            digest=digest,
        )


@dataclass(frozen=True)
class HelloPayload(ControlPayload):
    """Advertise one producer agent and its registered source regions."""

    message_type: ClassVar[str] = "hello"

    agent_metadata: str
    base_addresses: tuple[int, ...]
    logical_device: int
    region_bytes: tuple[int, ...]
    attestation: dict[str, object]

    def to_json(self) -> dict[str, object]:
        """Encode the producer advertisement.

        :returns: Exact hello payload object.
        """
        return {
            "agent_metadata": self.agent_metadata,
            "base_addresses": list(self.base_addresses),
            "logical_device": self.logical_device,
            "region_bytes": list(self.region_bytes),
            "attestation": self.attestation,
        }

    @classmethod
    def from_json(cls, value: object) -> Self:
        """Decode a producer advertisement.

        :param value: Decoded JSON value.
        :returns: Validated hello payload.
        """
        payload = _require_object(
            value,
            keys=frozenset(
                {
                    "agent_metadata",
                    "base_addresses",
                    "logical_device",
                    "region_bytes",
                    "attestation",
                }
            ),
            label="hello payload",
        )
        raw_addresses = payload["base_addresses"]
        raw_region_bytes = payload["region_bytes"]
        if not isinstance(raw_addresses, list) or len(raw_addresses) == 0:
            raise ProtocolError("hello.base_addresses must be a non-empty array")
        if not isinstance(raw_region_bytes, list) or len(raw_region_bytes) == 0:
            raise ProtocolError("hello.region_bytes must be a non-empty array")
        addresses = tuple(
            _require_integer(item, label="hello.base_addresses[]", minimum=0)
            for item in raw_addresses
        )
        region_bytes = tuple(
            _require_integer(item, label="hello.region_bytes[]", minimum=1)
            for item in raw_region_bytes
        )
        if len(addresses) != len(region_bytes):
            raise ProtocolError("hello region address and length counts differ")
        return cls(
            agent_metadata=_require_base64(
                payload["agent_metadata"], label="hello.agent_metadata"
            ),
            base_addresses=addresses,
            logical_device=_require_integer(
                payload["logical_device"], label="hello.logical_device", minimum=0
            ),
            region_bytes=region_bytes,
            attestation=_require_json_object(
                payload["attestation"], label="hello.attestation"
            ),
        )


@dataclass(frozen=True)
class PreparePayload(ControlPayload):
    """Identify one exact transfer iteration and notification lineage."""

    message_type: ClassVar[str] = "prepare"

    scenario: str
    scenario_iteration: int
    producer_request_id: str
    child_request_id: str
    notification_id: str

    def to_json(self) -> dict[str, object]:
        """Encode the transfer preparation request.

        :returns: Exact prepare payload object.
        """
        return {
            "scenario": self.scenario,
            "scenario_iteration": self.scenario_iteration,
            "producer_request_id": self.producer_request_id,
            "child_request_id": self.child_request_id,
            "notification_id": self.notification_id,
        }

    @classmethod
    def from_json(cls, value: object) -> Self:
        """Decode a transfer preparation request.

        :param value: Decoded JSON value.
        :returns: Validated prepare payload.
        """
        payload = _require_object(
            value,
            keys=frozenset(
                {
                    "scenario",
                    "scenario_iteration",
                    "producer_request_id",
                    "child_request_id",
                    "notification_id",
                }
            ),
            label="prepare payload",
        )
        return cls(
            scenario=_require_string(payload["scenario"], label="prepare.scenario"),
            scenario_iteration=_require_integer(
                payload["scenario_iteration"],
                label="prepare.scenario_iteration",
                minimum=0,
            ),
            producer_request_id=_require_string(
                payload["producer_request_id"], label="prepare.producer_request_id"
            ),
            child_request_id=_require_string(
                payload["child_request_id"], label="prepare.child_request_id"
            ),
            notification_id=_require_base64(
                payload["notification_id"], label="prepare.notification_id"
            ),
        )


@dataclass(frozen=True)
class PreparedPayload(ControlPayload):
    """Report producer-side source observations after source preparation."""

    message_type: ClassVar[str] = "prepared"

    digests: tuple[DigestRow, ...]
    observation_count: int

    def to_json(self) -> dict[str, object]:
        """Encode producer-side source observations.

        :returns: Exact prepared payload object.
        """
        return {
            "digests": [row.to_json() for row in self.digests],
            "observation_count": self.observation_count,
        }

    @classmethod
    def from_json(cls, value: object) -> Self:
        """Decode producer-side source observations.

        :param value: Decoded JSON value.
        :returns: Validated prepared payload.
        """
        payload = _require_object(
            value,
            keys=frozenset({"digests", "observation_count"}),
            label="prepared payload",
        )
        raw_digests = payload["digests"]
        if not isinstance(raw_digests, list):
            raise ProtocolError("prepared.digests must be an array")
        digests = tuple(DigestRow.from_json(row) for row in raw_digests)
        observation_count = _require_integer(
            payload["observation_count"],
            label="prepared.observation_count",
            minimum=0,
        )
        if observation_count != len(digests):
            raise ProtocolError("prepared observation count differs from digest rows")
        return cls(digests=digests, observation_count=observation_count)


@dataclass(frozen=True)
class CompletePayload(ControlPayload):
    """Acknowledge the consumer's transfer and notification submission."""

    message_type: ClassVar[str] = "complete"

    success: bool
    notification_id: str

    def to_json(self) -> dict[str, object]:
        """Encode consumer completion.

        :returns: Exact complete payload object.
        """
        return {
            "success": self.success,
            "notification_id": self.notification_id,
        }

    @classmethod
    def from_json(cls, value: object) -> Self:
        """Decode consumer completion.

        :param value: Decoded JSON value.
        :returns: Validated completion payload.
        """
        payload = _require_object(
            value,
            keys=frozenset({"success", "notification_id"}),
            label="complete payload",
        )
        return cls(
            success=_require_boolean(payload["success"], label="complete.success"),
            notification_id=_require_base64(
                payload["notification_id"], label="complete.notification_id"
            ),
        )


@dataclass(frozen=True)
class SourcePostPayload(ControlPayload):
    """Report producer immutability and notification evidence."""

    message_type: ClassVar[str] = "source_post"

    matches_pre: bool
    notification_seen: bool

    def to_json(self) -> dict[str, object]:
        """Encode producer post-transfer evidence.

        :returns: Exact source-post payload object.
        """
        return {
            "matches_pre": self.matches_pre,
            "notification_seen": self.notification_seen,
        }

    @classmethod
    def from_json(cls, value: object) -> Self:
        """Decode producer post-transfer evidence.

        :param value: Decoded JSON value.
        :returns: Validated source-post payload.
        """
        payload = _require_object(
            value,
            keys=frozenset({"matches_pre", "notification_seen"}),
            label="source_post payload",
        )
        return cls(
            matches_pre=_require_boolean(
                payload["matches_pre"], label="source_post.matches_pre"
            ),
            notification_seen=_require_boolean(
                payload["notification_seen"], label="source_post.notification_seen"
            ),
        )


@dataclass(frozen=True)
class StopPayload(ControlPayload):
    """Request orderly producer teardown."""

    message_type: ClassVar[str] = "stop"

    def to_json(self) -> dict[str, object]:
        """Encode the stop request.

        :returns: Empty stop payload.
        """
        return {}

    @classmethod
    def from_json(cls, value: object) -> Self:
        """Decode the stop request.

        :param value: Decoded JSON value.
        :returns: Validated stop payload.
        """
        _require_object(value, keys=frozenset(), label="stop payload")
        return cls()


@dataclass(frozen=True)
class StoppedPayload(ControlPayload):
    """Acknowledge orderly producer teardown."""

    message_type: ClassVar[str] = "stopped"

    def to_json(self) -> dict[str, object]:
        """Encode the stopped acknowledgement.

        :returns: Empty stopped payload.
        """
        return {}

    @classmethod
    def from_json(cls, value: object) -> Self:
        """Decode the stopped acknowledgement.

        :param value: Decoded JSON value.
        :returns: Validated stopped payload.
        """
        _require_object(value, keys=frozenset(), label="stopped payload")
        return cls()


PayloadT = TypeVar("PayloadT", bound=ControlPayload)


def _validate_identity(
    *,
    run_id: object,
    config_fingerprint: object,
    transport_arm: object,
    role: object,
    rank: object,
) -> tuple[str, str, str, str, int]:
    """Validate immutable endpoint lineage.

    :param run_id: Campaign UUID.
    :param config_fingerprint: SHA-256 configuration identity.
    :param transport_arm: Fresh-process transport arm.
    :param role: Producer or consumer role.
    :param rank: Role-local rank.
    :returns: Validated lineage fields.
    :raises ProtocolError: If any field is malformed.
    """
    run_id_text = _require_string(run_id, label="run_id")
    try:
        parsed_run_id = uuid.UUID(run_id_text)
    except ValueError as error:
        raise ProtocolError("run_id must be a UUID") from error
    if str(parsed_run_id) != run_id_text:
        raise ProtocolError("run_id must use canonical lowercase UUID text")
    fingerprint = _require_string(config_fingerprint, label="config_fingerprint")
    if len(fingerprint) != 64 or any(
        character not in "0123456789abcdef" for character in fingerprint
    ):
        raise ProtocolError("config_fingerprint must be lowercase SHA-256")
    arm = _require_string(transport_arm, label="transport_arm")
    role_name = _require_string(role, label="role")
    if role_name not in _ROLES:
        raise ProtocolError(f"unsupported protocol role: {role_name}")
    role_rank = _require_integer(rank, label="rank", minimum=0)
    return run_id_text, fingerprint, arm, role_name, role_rank


@dataclass
class JsonChannel:
    """Exchange bounded JSON with exact campaign lineage and sequencing.

    :ivar connection: Connected TCP socket owned by the channel.
    :ivar run_id: Campaign UUID shared by both endpoints.
    :ivar config_fingerprint: SHA-256 configuration identity.
    :ivar transport_arm: Fresh-process UCX arm.
    :ivar local_role: Role emitted by this endpoint.
    :ivar local_rank: Rank emitted by this endpoint.
    :ivar remote_role: Only accepted peer role.
    :ivar remote_rank: Only accepted peer rank.
    :ivar timeout_seconds: Socket send/receive timeout.
    """

    connection: socket.socket
    run_id: str
    config_fingerprint: str
    transport_arm: str
    local_role: str
    local_rank: int
    remote_role: str
    remote_rank: int
    timeout_seconds: float
    _send_sequence: int = field(default=0, init=False)
    _receive_sequence: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        """Validate immutable endpoint identity and configure timeout."""
        (
            self.run_id,
            self.config_fingerprint,
            self.transport_arm,
            self.local_role,
            self.local_rank,
        ) = _validate_identity(
            run_id=self.run_id,
            config_fingerprint=self.config_fingerprint,
            transport_arm=self.transport_arm,
            role=self.local_role,
            rank=self.local_rank,
        )
        _, _, _, self.remote_role, self.remote_rank = _validate_identity(
            run_id=self.run_id,
            config_fingerprint=self.config_fingerprint,
            transport_arm=self.transport_arm,
            role=self.remote_role,
            rank=self.remote_rank,
        )
        if type(self.timeout_seconds) not in {int, float} or not math.isfinite(
            self.timeout_seconds
        ):
            raise ProtocolError("timeout_seconds must be a finite number")
        if self.timeout_seconds <= 0:
            raise ProtocolError("timeout_seconds must be positive")
        self.connection.settimeout(self.timeout_seconds)

    def _context(self, *, direction: str, message_type: str, iteration: int) -> str:
        """Render exact endpoint context for transport failures.

        :param direction: Send or receive operation.
        :param message_type: Typed control message identity.
        :param iteration: Expected campaign iteration.
        :returns: Single-line diagnostic context.
        """
        return (
            f"direction={direction} local={self.local_role}:{self.local_rank} "
            f"remote={self.remote_role}:{self.remote_rank} run={self.run_id} "
            f"arm={self.transport_arm} iteration={iteration} "
            f"type={message_type}"
        )

    def _receive_exact(self, size: int, *, message_type: str, iteration: int) -> bytes:
        """Read an exact frame segment with contextual transport errors.

        :param size: Required byte count.
        :param message_type: Expected typed control message identity.
        :param iteration: Expected campaign iteration.
        :returns: Exact received bytes.
        :raises ProtocolError: If the peer times out, fails, or closes early.
        """
        chunks: list[bytes] = []
        remaining = size
        context = self._context(
            direction="receive", message_type=message_type, iteration=iteration
        )
        while remaining > 0:
            try:
                chunk = self.connection.recv(remaining)
            except TimeoutError as error:
                raise ProtocolError(
                    f"control receive timed out with {remaining} bytes outstanding; "
                    f"{context}"
                ) from error
            except OSError as error:
                raise ProtocolError(
                    f"control receive failed with {remaining} bytes outstanding; "
                    f"{context}: {error}"
                ) from error
            if len(chunk) == 0:
                raise ProtocolError(
                    f"control connection closed with {remaining} bytes outstanding; "
                    f"{context}"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def send(self, payload: ControlPayload, *, iteration: int) -> None:
        """Send one strictly framed typed control message.

        :param payload: Type-specific payload object.
        :param iteration: Campaign iteration, or -1 for setup/teardown.
        :raises ProtocolError: If the message or transport is invalid.
        """
        validated_iteration = _require_integer(iteration, label="iteration", minimum=-1)
        payload_json = type(payload).from_json(payload.to_json()).to_json()
        message_type = payload.message_type
        envelope: dict[str, object] = {
            "protocol_version": _PROTOCOL_VERSION,
            "type": message_type,
            "run_id": self.run_id,
            "config_fingerprint": self.config_fingerprint,
            "transport_arm": self.transport_arm,
            "iteration": validated_iteration,
            "sender_role": self.local_role,
            "sender_rank": self.local_rank,
            "sequence": self._send_sequence,
            "payload": payload_json,
        }
        try:
            encoded = json.dumps(
                envelope, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        except (TypeError, ValueError) as error:
            raise ProtocolError("control payload is not strict JSON") from error
        if len(encoded) > _MAX_MESSAGE_BYTES:
            raise ProtocolError(f"control message is too large: {len(encoded)} bytes")
        context = self._context(
            direction="send", message_type=message_type, iteration=iteration
        )
        try:
            self.connection.sendall(_HEADER.pack(len(encoded)) + encoded)
        except TimeoutError as error:
            raise ProtocolError(f"control send timed out; {context}") from error
        except OSError as error:
            raise ProtocolError(f"control send failed; {context}: {error}") from error
        self._send_sequence += 1

    def receive(self, payload_type: type[PayloadT], *, iteration: int) -> PayloadT:
        """Receive one typed message and reject replay or schema drift.

        :param payload_type: Required type-specific payload schema.
        :param iteration: Required campaign iteration.
        :returns: Validated typed payload.
        :raises ProtocolError: If framing, lineage, sequence, or payload differ.
        """
        expected_iteration = _require_integer(iteration, label="iteration", minimum=-1)
        message_type = payload_type.message_type
        size = _HEADER.unpack(
            self._receive_exact(
                _HEADER.size,
                message_type=message_type,
                iteration=expected_iteration,
            )
        )[0]
        if size > _MAX_MESSAGE_BYTES:
            raise ProtocolError(f"control message is too large: {size} bytes")
        encoded = self._receive_exact(
            size, message_type=message_type, iteration=expected_iteration
        )
        try:
            decoded: object = json.loads(encoded)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ProtocolError(f"invalid control JSON: {error}") from error
        message = _require_object(
            decoded, keys=frozenset(_ENVELOPE_KEYS), label="control envelope"
        )
        protocol_version = _require_integer(
            message["protocol_version"],
            label="control.protocol_version",
            minimum=0,
        )
        received_type = _require_string(message["type"], label="control.type")
        (
            received_run_id,
            received_fingerprint,
            received_arm,
            received_role,
            received_rank,
        ) = _validate_identity(
            run_id=message["run_id"],
            config_fingerprint=message["config_fingerprint"],
            transport_arm=message["transport_arm"],
            role=message["sender_role"],
            rank=message["sender_rank"],
        )
        received_iteration = _require_integer(
            message["iteration"], label="control.iteration", minimum=-1
        )
        received_sequence = _require_integer(
            message["sequence"], label="control.sequence", minimum=0
        )
        expected_fields: tuple[tuple[str, object, object], ...] = (
            ("protocol_version", protocol_version, _PROTOCOL_VERSION),
            ("type", received_type, message_type),
            ("run_id", received_run_id, self.run_id),
            ("config_fingerprint", received_fingerprint, self.config_fingerprint),
            ("transport_arm", received_arm, self.transport_arm),
            ("iteration", received_iteration, expected_iteration),
            ("sender_role", received_role, self.remote_role),
            ("sender_rank", received_rank, self.remote_rank),
            ("sequence", received_sequence, self._receive_sequence),
        )
        for key, actual, expected in expected_fields:
            if actual != expected:
                raise ProtocolError(
                    f"control {key} is {actual!r}, expected {expected!r}"
                )
        payload = payload_type.from_json(message["payload"])
        self._receive_sequence += 1
        return payload

    def close(self) -> None:
        """Close the owned socket."""
        self.connection.close()

    def __enter__(self) -> "JsonChannel":
        """Return the owned channel for a context-managed exchange.

        :returns: This channel.
        """
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        """Close the owned socket when leaving the exchange.

        :param exc_type: Active exception type, when present.
        :param exc: Active exception, when present.
        :param traceback: Active traceback, when present.
        """
        self.close()
