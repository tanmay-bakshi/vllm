# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Low-perturbation device fingerprints for NIXL diagnostics."""

import struct
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import torch


@dataclass(frozen=True, slots=True)
class _FingerprintParameters:
    """Cached position-dependent parameters for one payload geometry.

    :ivar masks: Four lanes of signed 32-bit XOR masks.
    :ivar coefficients: Four lanes of signed 32-bit odd coefficients.
    """

    masks: torch.Tensor
    coefficients: torch.Tensor


class NixlDeviceFingerprinter:
    """Compute compact, order-sensitive fingerprints on a tensor's device.

    The four-lane result is a noncryptographic probabilistic fingerprint. It is
    intended to localize accidental transfer corruption while copying only 32
    bytes per payload row to the host. Equality is strong diagnostic evidence,
    not proof that two payloads are identical or protection against an
    adversarially chosen collision.
    """

    _LANE_COUNT: ClassVar[int] = 4
    _WORD_BYTES: ClassVar[int] = 4
    _UINT32_MASK: ClassVar[int] = (1 << 32) - 1
    _MASK_SEEDS: ClassVar[tuple[int, ...]] = (
        0x243F6A88,
        0x85A308D3,
        0x13198A2E,
        0x03707344,
    )
    _COEFFICIENT_SEEDS: ClassVar[tuple[int, ...]] = (
        0xA4093822,
        0x299F31D0,
        0x082EFA98,
        0xEC4E6C89,
    )

    _parameter_cache: dict[tuple[torch.device, int], _FingerprintParameters]

    def __init__(self) -> None:
        """Initialize an empty device-parameter cache."""
        self._parameter_cache = {}

    @classmethod
    def _splitmix32(cls, values: torch.Tensor) -> torch.Tensor:
        """Apply the SplitMix32 finalizer to nonnegative 32-bit values.

        :param values: Signed 64-bit tensor whose values fit in 32 bits.
        :returns: Signed 64-bit tensor containing unsigned 32-bit results.
        """
        values = torch.bitwise_and(values + 0x9E3779B9, cls._UINT32_MASK)
        values = torch.bitwise_and(
            torch.bitwise_xor(values, torch.bitwise_right_shift(values, 16))
            * 0x21F0AAAD,
            cls._UINT32_MASK,
        )
        values = torch.bitwise_and(
            torch.bitwise_xor(values, torch.bitwise_right_shift(values, 15))
            * 0x735A2D97,
            cls._UINT32_MASK,
        )
        return torch.bitwise_xor(
            values,
            torch.bitwise_right_shift(values, 15),
        )

    @classmethod
    def _build_parameters(
        cls,
        device: torch.device,
        word_count: int,
    ) -> _FingerprintParameters:
        """Build four deterministic parameter lanes on the target device.

        :param device: Device on which fingerprinting will execute.
        :param word_count: Number of 32-bit words in each payload row.
        :returns: Position-dependent XOR masks and odd coefficients.
        """
        positions = torch.arange(word_count, dtype=torch.int64, device=device)
        mask_lanes: list[torch.Tensor] = []
        coefficient_lanes: list[torch.Tensor] = []
        for mask_seed, coefficient_seed in zip(
            cls._MASK_SEEDS,
            cls._COEFFICIENT_SEEDS,
            strict=True,
        ):
            mask_input = torch.bitwise_xor(positions, mask_seed)
            coefficient_input = torch.bitwise_xor(positions, coefficient_seed)
            mask_lanes.append(cls._splitmix32(mask_input).to(torch.int32))
            coefficient_lanes.append(
                torch.bitwise_or(
                    cls._splitmix32(coefficient_input),
                    1,
                ).to(torch.int32)
            )
        return _FingerprintParameters(
            masks=torch.stack(mask_lanes),
            coefficients=torch.stack(coefficient_lanes),
        )

    def _parameters_for(
        self,
        device: torch.device,
        word_count: int,
    ) -> _FingerprintParameters:
        """Return cached parameters for a device and payload geometry.

        :param device: Device on which fingerprinting will execute.
        :param word_count: Number of 32-bit words in each payload row.
        :returns: Cached or newly generated fingerprint parameters.
        """
        cache_key = (device, word_count)
        parameters = self._parameter_cache.get(cache_key)
        if parameters is not None:
            return parameters
        parameters = self._build_parameters(device, word_count)
        self._parameter_cache[cache_key] = parameters
        return parameters

    @classmethod
    def _validate_payloads(cls, payloads: torch.Tensor) -> None:
        """Validate one batch of fixed-width byte payloads.

        :param payloads: Candidate two-dimensional byte tensor.
        :raises TypeError: If *payloads* is not a tensor or has the wrong dtype.
        :raises ValueError: If its shape or storage layout is unsupported.
        """
        if not isinstance(payloads, torch.Tensor):
            raise TypeError("fingerprint payloads must be a torch.Tensor")
        if payloads.dtype is not torch.uint8:
            raise TypeError("fingerprint payloads must have dtype torch.uint8")
        if payloads.ndim != 2:
            raise ValueError("fingerprint payloads must be two-dimensional")
        if payloads.shape[0] == 0:
            raise ValueError("fingerprint payloads must contain at least one row")
        if payloads.shape[1] == 0:
            raise ValueError("fingerprint payload rows must not be empty")
        if payloads.shape[1] % cls._WORD_BYTES != 0:
            raise ValueError("fingerprint payload byte length must be divisible by 4")
        if not payloads.is_contiguous():
            raise ValueError("fingerprint payloads must be contiguous")
        if payloads.storage_offset() % cls._WORD_BYTES != 0:
            raise ValueError("fingerprint payload storage must be 4-byte aligned")

    def fingerprint_rows(self, payloads: torch.Tensor) -> torch.Tensor:
        """Fingerprint fixed-width byte payloads without leaving their device.

        Bytes are reinterpreted as signed 32-bit words. Each lane computes
        ``sum(((word XOR mask) * odd_coefficient), dtype=int64)`` with distinct,
        position-dependent SplitMix32 parameters. The signed 32-bit product
        intentionally wraps before the signed 64-bit reduction.

        :param payloads: Contiguous ``[rows, bytes]`` uint8 tensor whose row
            width is positive and divisible by four.
        :returns: Contiguous ``[rows, 4]`` int64 tensor on the input device.
        """
        self._validate_payloads(payloads)
        row_count, byte_count = payloads.shape
        word_count = byte_count // self._WORD_BYTES
        words = payloads.view(torch.int32).view(row_count, word_count)
        parameters = self._parameters_for(payloads.device, word_count)
        fingerprints = torch.empty(
            (row_count, self._LANE_COUNT),
            dtype=torch.int64,
            device=payloads.device,
        )
        for lane_index in range(self._LANE_COUNT):
            weighted_words = torch.bitwise_xor(
                words,
                parameters.masks[lane_index],
            )
            weighted_words.mul_(parameters.coefficients[lane_index])
            fingerprints[:, lane_index] = weighted_words.sum(
                dim=1,
                dtype=torch.int64,
            )
        return fingerprints

    @classmethod
    def _validate_fingerprints(cls, fingerprints: torch.Tensor) -> None:
        """Validate one batch of device-computed fingerprints.

        :param fingerprints: Candidate two-dimensional fingerprint tensor.
        :raises TypeError: If *fingerprints* is not an int64 tensor.
        :raises ValueError: If its shape or storage layout is unsupported.
        """
        if not isinstance(fingerprints, torch.Tensor):
            raise TypeError("fingerprints must be a torch.Tensor")
        if fingerprints.dtype is not torch.int64:
            raise TypeError("fingerprints must have dtype torch.int64")
        if fingerprints.ndim != 2 or fingerprints.shape[1] != cls._LANE_COUNT:
            raise ValueError("fingerprints must have shape [rows, 4]")
        if fingerprints.shape[0] == 0:
            raise ValueError("fingerprints must contain at least one row")
        if not fingerprints.is_contiguous():
            raise ValueError("fingerprints must be contiguous")

    def fingerprints_to_digests(
        self,
        fingerprint_batches: Sequence[torch.Tensor],
    ) -> list[bytes]:
        """Encode device fingerprint batches with one compact host copy.

        Call :meth:`fingerprint_rows` as each bounded payload chunk becomes
        available, then pass only those compact results here. The fingerprints
        are concatenated in sequence order before one tensor is copied to CPU.
        Each returned digest is exactly 32 bytes in little-endian lane order.

        :param fingerprint_batches: Nonempty sequence of compatible ``[rows, 4]``
            int64 tensors.
        :returns: One 32-byte digest for every input row, in input order.
        :raises TypeError: If a batch is not an int64 tensor.
        :raises ValueError: If batches have invalid shapes or different devices.
        """
        if len(fingerprint_batches) == 0:
            raise ValueError("at least one fingerprint batch is required")
        first_batch = fingerprint_batches[0]
        self._validate_fingerprints(first_batch)
        for batch in fingerprint_batches[1:]:
            self._validate_fingerprints(batch)
            if batch.device != first_batch.device:
                raise ValueError("fingerprint batches must share a device")

        fingerprints = (
            first_batch
            if len(fingerprint_batches) == 1
            else torch.cat(tuple(fingerprint_batches), dim=0)
        )
        host_fingerprints = fingerprints.cpu()
        return [
            struct.pack("<4q", *(int(value) for value in row))
            for row in host_fingerprints.tolist()
        ]
