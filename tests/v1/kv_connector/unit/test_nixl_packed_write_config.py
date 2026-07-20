# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for adaptive packed-write process configuration."""

from typing import cast
from unittest.mock import patch

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.packed_write_config import (
    PackedWriteConfig,
)


def test_default_packed_write_configuration_is_bounded_and_disabled() -> None:
    with patch.dict("os.environ", {}, clear=True):
        config = PackedWriteConfig.from_environment()

    assert config.enabled is False
    assert config.chunk_bytes_per_rank == 256 * 1024 * 1024
    assert config.producer_slot_count == 2
    assert config.consumer_slot_count == 2
    assert config.producer_pool_bytes == 512 * 1024 * 1024
    assert config.consumer_pool_bytes(4) == 2 * 1024 * 1024 * 1024
    assert config.min_descriptors_per_rank == 1024


def test_environment_selects_only_supported_exact_geometry() -> None:
    environment = {
        "VLLM_NIXL_PACKED_WRITE": "1",
        "VLLM_NIXL_PACKED_WRITE_CHUNK_MB": "128",
        "VLLM_NIXL_PACKED_WRITE_MIN_DESCRIPTORS_PER_RANK": "2118",
        "VLLM_NIXL_PACKED_WRITE_PRODUCER_SLOTS": "3",
        "VLLM_NIXL_PACKED_WRITE_CONSUMER_SLOTS": "4",
        "VLLM_NIXL_PACKED_WRITE_WARN_AFTER_S": "2.5",
        "VLLM_NIXL_PACKED_WRITE_FAIL_AFTER_S": "7.5",
    }
    with patch.dict("os.environ", environment, clear=True):
        config = PackedWriteConfig.from_environment()

    assert config.enabled is True
    assert config.chunk_bytes_per_rank == 128 * 1024 * 1024
    assert config.min_descriptors_per_rank == 2118
    assert config.producer_pool_bytes == 384 * 1024 * 1024
    assert config.consumer_pool_bytes(4) == 2 * 1024 * 1024 * 1024
    assert config.warn_after_s == 2.5
    assert config.fail_after_s == 7.5


@pytest.mark.parametrize(
    ("name", "value", "match"),
    [
        ("VLLM_NIXL_PACKED_WRITE", "true", "exactly '0' or '1'"),
        ("VLLM_NIXL_PACKED_WRITE_CHUNK_MB", "96", "64, 128, 256, or 512"),
        (
            "VLLM_NIXL_PACKED_WRITE_MIN_DESCRIPTORS_PER_RANK",
            "0",
            "at least 1",
        ),
        ("VLLM_NIXL_PACKED_WRITE_PRODUCER_SLOTS", "zero", "integer"),
        ("VLLM_NIXL_PACKED_WRITE_FAIL_AFTER_S", "nan", "warning age"),
    ],
)
def test_invalid_environment_fails_closed(
    name: str,
    value: str,
    match: str,
) -> None:
    with (
        patch.dict("os.environ", {name: value}, clear=True),
        pytest.raises(ValueError, match=match),
    ):
        PackedWriteConfig.from_environment()


@pytest.mark.parametrize("source_tp_size", [0, -1, True, 1.5])
def test_consumer_pool_rejects_invalid_source_topology(
    source_tp_size: object,
) -> None:
    with patch.dict("os.environ", {}, clear=True):
        config = PackedWriteConfig.from_environment()
    with pytest.raises(ValueError, match="positive integer"):
        config.consumer_pool_bytes(cast(int, source_tp_size))
