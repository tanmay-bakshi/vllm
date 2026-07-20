# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validated process configuration for adaptive producer-initiated packed writes."""

import math
import os
from dataclasses import dataclass

_MIB = 1024 * 1024
_SUPPORTED_CHUNK_MIB = (64, 128, 256, 512)


def _parse_boolean(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    if value == "0":
        return False
    if value == "1":
        return True
    raise ValueError(f"{name} must be exactly '0' or '1'")


def _parse_integer(name: str, default: int, *, minimum: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error
    if parsed < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return parsed


def _parse_float(name: str, default: float, *, minimum: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"{name} must be numeric") from error
    if parsed < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return parsed


@dataclass(frozen=True, slots=True)
class PackedWriteConfig:
    """Bound memory and select the adaptive packed transport.

    :ivar enabled: Whether packed transport resources and protocol are active.
    :ivar chunk_bytes_per_rank: Maximum bytes packed by one producer rank.
    :ivar min_descriptors_per_rank: Direct-plan threshold selecting packing.
    :ivar producer_slot_count: Registered reusable slots on each producer rank.
    :ivar consumer_slot_count: Registered rank-major slots on each decoder.
    :ivar alignment_bytes: Required address and slot alignment.
    :ivar warn_after_s: One-shot age warning for a live packed operation.
    :ivar fail_after_s: Age after which an unambiguous stall fails closed.
    """

    enabled: bool
    chunk_bytes_per_rank: int
    min_descriptors_per_rank: int
    producer_slot_count: int
    consumer_slot_count: int
    alignment_bytes: int
    warn_after_s: float
    fail_after_s: float

    def __post_init__(self) -> None:
        if self.chunk_bytes_per_rank not in tuple(
            chunk_mib * _MIB for chunk_mib in _SUPPORTED_CHUNK_MIB
        ):
            raise ValueError("packed write chunk size must be 64, 128, 256, or 512 MiB")
        if self.min_descriptors_per_rank < 1:
            raise ValueError("packed write descriptor threshold must be positive")
        if self.producer_slot_count < 1 or self.consumer_slot_count < 1:
            raise ValueError("packed write slot counts must be positive")
        if (
            self.alignment_bytes < 1
            or (self.alignment_bytes & (self.alignment_bytes - 1)) != 0
        ):
            raise ValueError("packed write alignment must be a positive power of two")
        if self.chunk_bytes_per_rank % self.alignment_bytes != 0:
            raise ValueError("packed write chunk size must satisfy its alignment")
        if (
            math.isfinite(self.warn_after_s) is False
            or math.isfinite(self.fail_after_s) is False
            or self.warn_after_s <= 0
            or self.fail_after_s <= self.warn_after_s
        ):
            raise ValueError("packed write failure age must exceed its warning age")

    @classmethod
    def from_environment(cls) -> "PackedWriteConfig":
        """Read one complete packed-write process contract.

        :returns: Validated immutable configuration.
        """
        chunk_mib = _parse_integer(
            "VLLM_NIXL_PACKED_WRITE_CHUNK_MB",
            256,
            minimum=1,
        )
        return cls(
            enabled=_parse_boolean("VLLM_NIXL_PACKED_WRITE", False),
            chunk_bytes_per_rank=chunk_mib * _MIB,
            min_descriptors_per_rank=_parse_integer(
                "VLLM_NIXL_PACKED_WRITE_MIN_DESCRIPTORS_PER_RANK",
                1024,
                minimum=1,
            ),
            producer_slot_count=_parse_integer(
                "VLLM_NIXL_PACKED_WRITE_PRODUCER_SLOTS",
                2,
                minimum=1,
            ),
            consumer_slot_count=_parse_integer(
                "VLLM_NIXL_PACKED_WRITE_CONSUMER_SLOTS",
                2,
                minimum=1,
            ),
            alignment_bytes=256,
            warn_after_s=_parse_float(
                "VLLM_NIXL_PACKED_WRITE_WARN_AFTER_S",
                30.0,
                minimum=0.001,
            ),
            fail_after_s=_parse_float(
                "VLLM_NIXL_PACKED_WRITE_FAIL_AFTER_S",
                300.0,
                minimum=0.001,
            ),
        )

    @property
    def producer_pool_bytes(self) -> int:
        """Return registered bytes on each producer rank.

        :returns: Complete per-rank producer pool size.
        """
        return self.producer_slot_count * self.chunk_bytes_per_rank

    def consumer_pool_bytes(self, source_tp_size: int) -> int:
        """Return registered rank-major bytes on one decoder.

        :param source_tp_size: Producer ranks represented in every slot.
        :returns: Complete decoder receive-pool size.
        """
        if type(source_tp_size) is not int or source_tp_size < 1:
            raise ValueError("source_tp_size must be a positive integer")
        return self.consumer_slot_count * source_tp_size * self.chunk_bytes_per_rank
