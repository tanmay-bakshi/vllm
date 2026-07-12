# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for canonical NIXL transfer-integrity records."""

import struct

import pytest

from vllm.distributed.kv_transfer.integrity import (
    IntegrityIdentity,
    IntegrityObservation,
    IntegrityPayloadKind,
    IntegrityStage,
    canonical_identity_bytes,
    compute_integrity_digest,
    observations_match,
)


def _identity(**overrides: object) -> IntegrityIdentity:
    values: dict[str, object] = {
        "run_id": "run-37b",
        "transport_arm": "tcp-shm-cuda-copy",
        "producer_engine_id": "prefill",
        "producer_request_id": "producer-rid",
        "offer_generation": 3,
        "iteration": 7,
        "source_rank": 2,
        "region_index": 9,
        "group_index": 11,
        "source_position": 41,
        "remote_block_id": 32768,
        "payload_kind": IntegrityPayloadKind.WIRE,
        "byte_length": 8,
    }
    values.update(overrides)
    return IntegrityIdentity(**values)  # type: ignore[arg-type]


@pytest.mark.cpu_test
def test_digest_catches_order_and_compensating_changes() -> None:
    """Order and compensating mutations defeat linear sums, not this digest."""
    identity = _identity()
    baseline = struct.pack(">II", 1, 4)
    permuted = struct.pack(">II", 4, 1)
    compensated = struct.pack(">II", 2, 3)

    assert sum(struct.unpack(">II", baseline)) == sum(
        struct.unpack(">II", permuted)
    )
    assert sum(struct.unpack(">II", baseline)) == sum(
        struct.unpack(">II", compensated)
    )

    baseline_digest = compute_integrity_digest(identity, baseline)
    assert compute_integrity_digest(identity, permuted) != baseline_digest
    assert compute_integrity_digest(identity, compensated) != baseline_digest


@pytest.mark.cpu_test
def test_digest_catches_single_byte_flip() -> None:
    """Every payload byte participates in the digest."""
    identity = _identity()
    baseline = bytes(range(8))
    flipped = bytearray(baseline)
    flipped[-1] ^= 0x80

    assert compute_integrity_digest(identity, baseline) != compute_integrity_digest(
        identity, flipped
    )


@pytest.mark.cpu_test
def test_identity_domain_changes_digest() -> None:
    """Source geometry and lineage are part of the digest domain."""
    payload = bytes(range(8))
    baseline = compute_integrity_digest(_identity(), payload)

    for field, value in (
        ("run_id", "run-other"),
        ("transport_arm", "cuda-ipc"),
        ("producer_engine_id", "prefill-other"),
        ("producer_request_id", "rid-other"),
        ("offer_generation", 4),
        ("iteration", 8),
        ("source_rank", 3),
        ("region_index", 10),
        ("group_index", 12),
        ("source_position", 42),
        ("remote_block_id", 32769),
        ("payload_kind", IntegrityPayloadKind.COMMIT),
    ):
        changed = compute_integrity_digest(_identity(**{field: value}), payload)
        assert changed != baseline


@pytest.mark.cpu_test
def test_stage_and_consumer_lineage_do_not_change_comparison_identity() -> None:
    """Stage-local lineage is recorded without poisoning cross-stage equality."""
    identity = _identity()
    digest = compute_integrity_digest(identity, bytes(range(8)))
    source = IntegrityObservation(
        stage=IntegrityStage.SOURCE,
        identity=identity,
        digest=digest,
        child_request_id=None,
        observer_engine_id="prefill",
        observer_rank=2,
        local_block_id=None,
    )
    staging = IntegrityObservation(
        stage=IntegrityStage.STAGING,
        identity=identity,
        digest=digest,
        child_request_id="decode-child",
        observer_engine_id="decoder",
        observer_rank=0,
        local_block_id=101,
    )

    assert observations_match(source, staging)


@pytest.mark.cpu_test
def test_canonical_identity_is_stable() -> None:
    """Canonical serialization has a fixed compatibility fingerprint."""
    assert canonical_identity_bytes(_identity()).hex() == (
        "00010000000772756e2d333762000000117463702d73686d2d637564612d636f7079"
        "0000000770726566696c6c0000000c70726f64756365722d72696400000000000000"
        "0300000000000000070000000000000002000000090000000b0000002900000004"
        "0000000000008000000000000000000877697265"
    )


@pytest.mark.cpu_test
def test_payload_length_is_enforced() -> None:
    """A digest cannot silently cover a truncated or extended payload."""
    with pytest.raises(ValueError, match="expected 8"):
        compute_integrity_digest(_identity(), b"short")


@pytest.mark.cpu_test
def test_observation_requires_stage_appropriate_lineage() -> None:
    """Source and consumer observations cannot masquerade as each other."""
    identity = _identity()
    digest = compute_integrity_digest(identity, bytes(range(8)))
    with pytest.raises(ValueError, match="cannot carry consumer-local"):
        IntegrityObservation(
            stage=IntegrityStage.SOURCE,
            identity=identity,
            digest=digest,
            child_request_id="child",
            observer_engine_id="prefill",
            observer_rank=2,
            local_block_id=1,
        )
    with pytest.raises(ValueError, match="require child_request_id"):
        IntegrityObservation(
            stage=IntegrityStage.DESTINATION,
            identity=identity,
            digest=digest,
            child_request_id=None,
            observer_engine_id="decoder",
            observer_rank=0,
            local_block_id=1,
        )
