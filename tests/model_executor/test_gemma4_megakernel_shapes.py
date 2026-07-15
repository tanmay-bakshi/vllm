# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.model_executor.models.gemma4_megakernel_shapes import (
    HIGH_M_KERNEL_ROWS,
    RowBlock,
    kernel_shapes,
    plan_shape,
    row_offsets,
)

pytestmark = pytest.mark.cpu_test


def test_shape_within_row_limit_remains_one_exact_kernel() -> None:
    assert plan_shape((16, 7)) == (RowBlock((16, 7), 0, 0),)


def test_384_token_shape_uses_one_high_m_kernel() -> None:
    assert plan_shape((16, 24)) == (RowBlock((16, 24), 0, 0),)


def test_high_m_kernel_preserves_request_geometry() -> None:
    assert plan_shape((12, 32)) == (RowBlock((12, 32), 0, 0),)


def test_high_m_kernel_allows_one_request_larger_than_one_tile() -> None:
    assert plan_shape((256, 1)) == (RowBlock((256, 1), 0, 0),)


def test_ragged_shape_uses_the_minimum_number_of_balanced_blocks() -> None:
    assert plan_shape((16, 15)) == (
        RowBlock((16, 8), 0, 0),
        RowBlock((16, 7), 8, 128),
    )


def test_power_of_two_partition_reuses_existing_kernel_shape() -> None:
    assert plan_shape((12, 24)) == (
        RowBlock((12, 8), 0, 0),
        RowBlock((12, 8), 8, 96),
        RowBlock((12, 8), 16, 192),
    )


def test_non_power_of_two_partition_is_balanced() -> None:
    assert plan_shape((12, 27)) == (
        RowBlock((12, 9), 0, 0),
        RowBlock((12, 9), 9, 108),
        RowBlock((12, 9), 18, 216),
    )


def test_kernel_shapes_deduplicate_components() -> None:
    served_shapes = (
        (16, 1),
        (16, 2),
        (16, 4),
        (16, 8),
        (16, 15),
        (16, 24),
    )

    assert kernel_shapes(served_shapes) == (
        (16, 1),
        (16, 2),
        (16, 4),
        (16, 7),
        (16, 8),
        (16, 24),
    )


def test_row_offsets_cover_ragged_and_full_blocks() -> None:
    assert row_offsets(((16, 15), (16, 24))) == (0, 128)


def test_plans_cover_all_supported_request_counts_exactly() -> None:
    for query_length in (1, 4, 8, 12, 16):
        for num_requests in range(1, 33):
            blocks = plan_shape((query_length, num_requests))
            requests_per_block = 128 // query_length
            total_rows = query_length * num_requests
            expected_blocks = 1
            if total_rows > 128 and total_rows not in HIGH_M_KERNEL_ROWS:
                expected_blocks = (
                    num_requests + requests_per_block - 1
                ) // requests_per_block
            assert len(blocks) == expected_blocks
            next_request = 0
            for block in blocks:
                assert block.request_offset == next_request
                assert block.row_offset == query_length * next_request
                block_query_length, block_requests = block.kernel_shape
                assert block_query_length == query_length
                block_rows = block_query_length * block_requests
                assert block_rows <= 128 or block_rows in HIGH_M_KERNEL_ROWS
                next_request += block_requests
            assert next_request == num_requests


@pytest.mark.parametrize("shape", [(0, 1), (1, 0), (-1, 1), (1, -1)])
def test_invalid_shape_is_rejected(shape: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="shape dimensions must be positive"):
        plan_shape(shape)


def test_query_longer_than_one_kernel_is_rejected() -> None:
    with pytest.raises(ValueError, match="single-tile kernel limit"):
        plan_shape((129, 1))


def test_shape_larger_than_serving_buffers_is_rejected() -> None:
    with pytest.raises(ValueError, match="serving limit"):
        plan_shape((1, 513))
