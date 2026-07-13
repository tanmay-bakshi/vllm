# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for low-perturbation NIXL device fingerprints."""

import struct

import pytest
import torch

from vllm.distributed.kv_transfer.nixl_fingerprint import (
    NixlDeviceFingerprinter,
)

_UINT32_MASK = (1 << 32) - 1
_MASK_SEEDS = (
    0x243F6A88,
    0x85A308D3,
    0x13198A2E,
    0x03707344,
)
_COEFFICIENT_SEEDS = (
    0xA4093822,
    0x299F31D0,
    0x082EFA98,
    0xEC4E6C89,
)


def _splitmix32(value: int) -> int:
    value = (value + 0x9E3779B9) & _UINT32_MASK
    value = ((value ^ (value >> 16)) * 0x21F0AAAD) & _UINT32_MASK
    value = ((value ^ (value >> 15)) * 0x735A2D97) & _UINT32_MASK
    return value ^ (value >> 15)


def _as_int32(value: int) -> int:
    value &= _UINT32_MASK
    if value < 1 << 31:
        return value
    return value - (1 << 32)


def _reference_fingerprint(payload: bytes) -> tuple[int, int, int, int]:
    words = struct.unpack(f"<{len(payload) // 4}i", payload)
    lanes: list[int] = []
    for mask_seed, coefficient_seed in zip(
        _MASK_SEEDS,
        _COEFFICIENT_SEEDS,
        strict=True,
    ):
        total = 0
        for position, word in enumerate(words):
            mask = _as_int32(_splitmix32(position ^ mask_seed))
            coefficient = _as_int32(_splitmix32(position ^ coefficient_seed) | 1)
            product = _as_int32((word ^ mask) * coefficient)
            total += product
        lanes.append(total)
    return lanes[0], lanes[1], lanes[2], lanes[3]


@pytest.mark.cpu_test
def test_fingerprint_matches_scalar_reference() -> None:
    """The tensor implementation preserves the specified 32-bit arithmetic."""
    payloads = torch.tensor(
        [
            list(range(16)),
            [255, 0, 127, 128, 11, 22, 33, 44, 55, 66, 77, 88, 99, 1, 2, 3],
        ],
        dtype=torch.uint8,
    )

    fingerprints = NixlDeviceFingerprinter().fingerprint_rows(payloads)

    assert fingerprints.dtype is torch.int64
    assert fingerprints.device == payloads.device
    assert fingerprints.is_contiguous()
    assert fingerprints.shape == (2, 4)
    assert fingerprints.tolist() == [
        [-5873781097, 1258166376, 4130139304, 404894561],
        [3351345397, 19736646, 2660118832, 628287823],
    ]
    assert [tuple(row) for row in fingerprints.tolist()] == [
        _reference_fingerprint(bytes(row)) for row in payloads.tolist()
    ]


@pytest.mark.cpu_test
def test_results_return_little_endian_digests_in_row_order() -> None:
    """Compact result batches preserve row order across one host handoff."""
    first = torch.arange(24, dtype=torch.uint8).reshape(2, 12)
    second = torch.arange(16, dtype=torch.uint8).reshape(1, 16)
    fingerprinter = NixlDeviceFingerprinter()

    digests = fingerprinter.fingerprints_to_digests(
        (
            fingerprinter.fingerprint_rows(first),
            fingerprinter.fingerprint_rows(second),
        )
    )

    expected_rows = first.tolist() + second.tolist()
    assert len(digests) == 3
    assert all(len(digest) == 32 for digest in digests)
    assert [struct.unpack("<4q", digest) for digest in digests] == [
        _reference_fingerprint(bytes(row)) for row in expected_rows
    ]


@pytest.mark.cpu_test
def test_position_and_every_payload_byte_affect_fingerprint() -> None:
    """The fixed diagnostic examples catch reordering and single-byte damage."""
    baseline = torch.arange(32, dtype=torch.uint8).reshape(1, 32)
    reordered = baseline.reshape(1, 8, 4).flip(1).reshape(1, 32)
    damaged = baseline.clone()
    damaged[0, -1] ^= 0x80
    fingerprinter = NixlDeviceFingerprinter()

    baseline_fingerprint = fingerprinter.fingerprint_rows(baseline)

    assert not torch.equal(
        fingerprinter.fingerprint_rows(reordered),
        baseline_fingerprint,
    )
    assert not torch.equal(
        fingerprinter.fingerprint_rows(damaged),
        baseline_fingerprint,
    )


@pytest.mark.cpu_test
def test_parameters_are_cached_by_device_and_word_count() -> None:
    """Repeated geometries reuse parameters while distinct widths do not."""
    fingerprinter = NixlDeviceFingerprinter()

    fingerprinter.fingerprint_rows(torch.zeros((2, 8), dtype=torch.uint8))
    original_parameters = fingerprinter._parameter_cache[(torch.device("cpu"), 2)]
    fingerprinter.fingerprint_rows(torch.ones((1, 8), dtype=torch.uint8))
    fingerprinter.fingerprint_rows(torch.zeros((1, 12), dtype=torch.uint8))

    assert len(fingerprinter._parameter_cache) == 2
    assert (
        fingerprinter._parameter_cache[(torch.device("cpu"), 2)] is original_parameters
    )
    assert torch.all(original_parameters.coefficients.bitwise_and(1) == 1).item()


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("payloads", "error", "message"),
    [
        ([1, 2, 3, 4], TypeError, "torch.Tensor"),
        (torch.zeros((1, 4), dtype=torch.int32), TypeError, "torch.uint8"),
        (torch.zeros(4, dtype=torch.uint8), ValueError, "two-dimensional"),
        (torch.zeros((0, 4), dtype=torch.uint8), ValueError, "at least one row"),
        (torch.zeros((1, 0), dtype=torch.uint8), ValueError, "must not be empty"),
        (torch.zeros((1, 6), dtype=torch.uint8), ValueError, "divisible by 4"),
        (
            torch.zeros((2, 8), dtype=torch.uint8)[:, ::2],
            ValueError,
            "contiguous",
        ),
        (
            torch.zeros(9, dtype=torch.uint8)[1:].reshape(1, 8),
            ValueError,
            "4-byte aligned",
        ),
    ],
)
def test_invalid_payloads_are_rejected(
    payloads: object,
    error: type[Exception],
    message: str,
) -> None:
    """Malformed payloads cannot silently change fingerprint semantics."""
    with pytest.raises(error, match=message):
        NixlDeviceFingerprinter().fingerprint_rows(payloads)  # type: ignore[arg-type]


@pytest.mark.cpu_test
def test_invalid_or_missing_fingerprint_batches_are_rejected() -> None:
    """Digest encoding accepts only compact result batches."""
    fingerprinter = NixlDeviceFingerprinter()
    with pytest.raises(ValueError, match="at least one"):
        fingerprinter.fingerprints_to_digests(())
    with pytest.raises(TypeError, match="torch.int64"):
        fingerprinter.fingerprints_to_digests((torch.zeros((1, 4), dtype=torch.int32),))
    with pytest.raises(ValueError, match=r"\[rows, 4\]"):
        fingerprinter.fingerprints_to_digests((torch.zeros((1, 3), dtype=torch.int64),))
