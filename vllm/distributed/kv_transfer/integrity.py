# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Canonical integrity records for NIXL transfer diagnostics."""

import hashlib
import struct
from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar


class IntegrityStage(StrEnum):
    """Observation point for a transferred payload."""

    SOURCE_PRE = "source_pre"
    SOURCE_POST = "source_post"
    STAGING_RAW = "staging_raw"
    STAGING_FENCED_CONTROL = "staging_fenced_control"
    DESTINATION = "destination"
    PRE_READ = "pre_read"


class IntegrityPayloadKind(StrEnum):
    """Byte contract represented by an integrity leaf."""

    WIRE = "wire"
    COMMIT = "commit"


@dataclass(frozen=True, slots=True)
class IntegrityIdentity:
    """Stage-independent identity of one ordered transfer payload.

    :ivar run_id: Identifier shared by every participant in one diagnostic run.
    :ivar transport_arm: Human-readable transport configuration under test.
    :ivar producer_engine_id: Engine that owns the source allocation.
    :ivar producer_request_id: Producer-side request that pins the allocation.
    :ivar registration_generation: Incarnation of the registered source regions.
    :ivar semantic_contract_digest: Canonical region/group layout contract.
    :ivar offer_generation: Generation of the producer's transfer offer.
    :ivar iteration: Repetition number within the transport arm.
    :ivar source_rank: Tensor-parallel rank that owns the source bytes.
    :ivar region_index: Registered source-region index.
    :ivar group_index: KV-cache group that owns the source position.
    :ivar plane_index: Source payload plane (K=0, V=1), or ``-1`` when combined.
    :ivar source_position: Physical position ordinal within the group roster.
    :ivar remote_block_id: Producer physical block identifier.
    :ivar valid_token_extent: Exact request token extent represented by the offer.
    :ivar group_token_capacity: Runtime token capacity of one source group block.
    :ivar payload_kind: Wire bytes or the subset committed at the destination.
    :ivar byte_length: Exact number of bytes covered by the digest.
    """

    SCHEMA_VERSION: ClassVar[int] = 1

    run_id: str
    transport_arm: str
    producer_engine_id: str
    producer_request_id: str
    registration_generation: str
    semantic_contract_digest: bytes
    offer_generation: int
    iteration: int
    source_rank: int
    region_index: int
    group_index: int
    plane_index: int
    source_position: int
    remote_block_id: int
    valid_token_extent: int
    group_token_capacity: int
    payload_kind: IntegrityPayloadKind
    byte_length: int

    def __post_init__(self) -> None:
        """Validate values used in the canonical digest domain."""
        if len(self.run_id) == 0:
            raise ValueError("run_id must not be empty")
        if len(self.transport_arm) == 0:
            raise ValueError("transport_arm must not be empty")
        if len(self.producer_engine_id) == 0:
            raise ValueError("producer_engine_id must not be empty")
        if len(self.producer_request_id) == 0:
            raise ValueError("producer_request_id must not be empty")
        if len(self.registration_generation) == 0:
            raise ValueError("registration_generation must not be empty")
        if len(self.semantic_contract_digest) != 32:
            raise ValueError("semantic_contract_digest must contain exactly 256 bits")

        non_negative_fields = (
            self.offer_generation,
            self.iteration,
            self.source_rank,
            self.region_index,
            self.group_index,
            self.source_position,
            self.remote_block_id,
            self.valid_token_extent,
            self.group_token_capacity,
            self.byte_length,
        )
        if any(type(value) is not int for value in non_negative_fields):
            raise TypeError("integrity identity integers must be exact int values")
        if any(value < 0 for value in non_negative_fields):
            raise ValueError("integrity identity integers must be non-negative")
        if type(self.plane_index) is not int:
            raise TypeError("plane_index must be an exact int value")
        if self.plane_index < -1:
            raise ValueError("plane_index must be -1, 0, or 1")
        if self.plane_index > 1:
            raise ValueError("plane_index must be -1, 0, or 1")


@dataclass(frozen=True, slots=True)
class IntegrityObservation:
    """One stage-specific observation of an integrity identity.

    The observation fields are deliberately excluded from the digest domain.
    Source, staging, and destination observations must remain comparable even
    though their local request and destination block identifiers differ.

    :ivar stage: Point at which the payload was observed.
    :ivar identity: Stage-independent source identity and byte contract.
    :ivar digest: BLAKE2b-128 digest of the canonical identity and exact bytes.
    :ivar child_request_id: Consumer request lineage, absent at the source.
    :ivar observer_engine_id: Engine that recorded the observation.
    :ivar observer_rank: Tensor-parallel rank that recorded the observation.
    :ivar local_block_id: Consumer physical block identifier, absent at source.
    """

    stage: IntegrityStage
    identity: IntegrityIdentity
    digest: bytes
    child_request_id: str | None
    observer_engine_id: str
    observer_rank: int
    local_block_id: int | None

    def __post_init__(self) -> None:
        """Validate observation-only lineage fields."""
        if len(self.digest) != 16:
            raise ValueError("integrity digest must contain exactly 128 bits")
        if len(self.observer_engine_id) == 0:
            raise ValueError("observer_engine_id must not be empty")
        if self.observer_rank < 0:
            raise ValueError("observer_rank must be non-negative")
        if self.local_block_id is not None and self.local_block_id < 0:
            raise ValueError("local_block_id must be non-negative")
        if self.stage in (IntegrityStage.SOURCE_PRE, IntegrityStage.SOURCE_POST):
            if self.child_request_id is not None or self.local_block_id is not None:
                raise ValueError(
                    "source observations cannot carry consumer-local lineage"
                )
        elif self.child_request_id is None or len(self.child_request_id) == 0:
            raise ValueError("consumer observations require child_request_id")


def _frame_text(value: str) -> bytes:
    encoded = value.encode("utf-8")
    if len(encoded) > 0xFFFFFFFF:
        raise ValueError("integrity identity text field is too large")
    return struct.pack(">I", len(encoded)) + encoded


def canonical_identity_bytes(identity: IntegrityIdentity) -> bytes:
    """Serialize an integrity identity without interpreter-dependent state.

    :param identity: Identity to serialize.
    :returns: Canonical big-endian binary representation.
    """
    payload_kind = identity.payload_kind.value
    return b"".join(
        (
            struct.pack(">H", IntegrityIdentity.SCHEMA_VERSION),
            _frame_text(identity.run_id),
            _frame_text(identity.transport_arm),
            _frame_text(identity.producer_engine_id),
            _frame_text(identity.producer_request_id),
            _frame_text(identity.registration_generation),
            identity.semantic_contract_digest,
            struct.pack(
                ">QQQIIiIIQQQQ",
                identity.offer_generation,
                identity.iteration,
                identity.source_rank,
                identity.region_index,
                identity.group_index,
                identity.plane_index,
                identity.source_position,
                len(payload_kind),
                identity.remote_block_id,
                identity.valid_token_extent,
                identity.group_token_capacity,
                identity.byte_length,
            ),
            payload_kind.encode("ascii"),
        )
    )


def compute_integrity_digest(
    identity: IntegrityIdentity,
    payload: bytes | bytearray | memoryview,
) -> bytes:
    """Hash one exact, ordered payload with a canonical identity domain.

    BLAKE2b is a cryptographic hash, truncated here to a 128-bit diagnostic
    digest. The result is probabilistic collision resistance, not a proof of
    byte equality.

    :param identity: Stage-independent identity of the payload.
    :param payload: Contiguous bytes in their transmitted logical order.
    :returns: Sixteen-byte BLAKE2b digest.
    :raises ValueError: If the payload is non-contiguous or has the wrong size.
    """
    payload_view = memoryview(payload)
    if payload_view.contiguous is False:
        raise ValueError("integrity payload must be contiguous")
    byte_view = payload_view.cast("B")
    if byte_view.nbytes != identity.byte_length:
        raise ValueError(
            f"integrity payload has {byte_view.nbytes} bytes, "
            f"expected {identity.byte_length}"
        )

    hasher = hashlib.blake2b(digest_size=16, person=b"vllm-p2d-v1")
    hasher.update(canonical_identity_bytes(identity))
    hasher.update(byte_view)
    return hasher.digest()


def observations_match(
    expected: IntegrityObservation,
    actual: IntegrityObservation,
) -> bool:
    """Return whether two observations cover and contain the same payload.

    :param expected: Reference observation.
    :param actual: Observation to compare against the reference.
    :returns: ``True`` only when identity and digest both match.
    """
    return expected.identity == actual.identity and expected.digest == actual.digest
