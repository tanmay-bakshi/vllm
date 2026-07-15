# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shape planning for Gemma 4 fused decode kernels."""

from dataclasses import dataclass
from itertools import combinations_with_replacement

MAX_STANDARD_KERNEL_ROWS = 128
HIGH_M_KERNEL_ROWS = frozenset((256, 384, 512))
MAX_SERVED_ROWS = max(HIGH_M_KERNEL_ROWS)

Shape = tuple[int, int]


@dataclass(frozen=True, slots=True)
class RowBlock:
    """One contiguous fused-kernel launch within a served decode shape.

    :ivar kernel_shape: Query length and request count compiled into the kernel.
    :ivar request_offset: First request handled by the launch.
    :ivar row_offset: First flattened token row handled by the launch.
    """

    kernel_shape: Shape
    request_offset: int
    row_offset: int


def plan_shape(shape: Shape) -> tuple[RowBlock, ...]:
    """Plan exact fused-kernel launches for one uniform decode shape.

    Shapes supported by either the single-tile or measured high-M tactics
    retain their exact shape. Other shapes use the minimum possible number of
    single-tile launches. When that launch count admits an exact power-of-two
    partition, the plan prefers it so graph capture buckets reuse compiled
    kernels. Other request counts use balanced blocks, limiting the plan to at
    most two component shapes.

    :param shape: Query length and request count for the complete decode step.
    :returns: Contiguous launches covering every request exactly once.
    :raises ValueError: If either shape dimension is not positive or one
        request cannot fit in a kernel launch.
    """
    query_length, num_requests = shape
    if query_length < 1 or num_requests < 1:
        raise ValueError(f"shape dimensions must be positive: {shape}")
    total_rows = query_length * num_requests
    if total_rows > MAX_SERVED_ROWS:
        raise ValueError(
            f"shape exceeds the {MAX_SERVED_ROWS}-row serving limit: {shape}"
        )
    if total_rows in HIGH_M_KERNEL_ROWS:
        return (RowBlock(shape, 0, 0),)
    if query_length > MAX_STANDARD_KERNEL_ROWS:
        raise ValueError(
            "query length exceeds the single-tile kernel limit of "
            f"{MAX_STANDARD_KERNEL_ROWS}: {shape}"
        )

    requests_per_block = max(1, MAX_STANDARD_KERNEL_ROWS // query_length)
    if num_requests <= requests_per_block:
        block_sizes = [num_requests]
    else:
        num_blocks = (num_requests + requests_per_block - 1) // requests_per_block
        power_sizes = tuple(
            1 << exponent
            for exponent in range(requests_per_block.bit_length())
            if 1 << exponent <= requests_per_block
        )
        power_partitions = (
            partition
            for partition in combinations_with_replacement(power_sizes, num_blocks)
            if sum(partition) == num_requests
        )
        power_partition = min(
            power_partitions,
            key=lambda partition: (
                len(set(partition)),
                max(partition) - min(partition),
                partition,
            ),
            default=None,
        )
        if power_partition is not None:
            block_sizes = sorted(power_partition, reverse=True)
        else:
            block_size, larger_blocks = divmod(num_requests, num_blocks)
            block_sizes = [block_size + 1] * larger_blocks
            block_sizes.extend([block_size] * (num_blocks - larger_blocks))

    blocks: list[RowBlock] = []
    request_offset = 0
    for block_size in block_sizes:
        blocks.append(
            RowBlock(
                kernel_shape=(query_length, block_size),
                request_offset=request_offset,
                row_offset=query_length * request_offset,
            )
        )
        request_offset += block_size
    return tuple(blocks)


def kernel_shapes(served_shapes: tuple[Shape, ...]) -> tuple[Shape, ...]:
    """Return the unique kernel shapes required by served decode shapes.

    :param served_shapes: Complete uniform decode shapes exposed by the runner.
    :returns: Kernel shapes in deterministic order.
    """
    return tuple(
        sorted(
            {
                block.kernel_shape
                for shape in served_shapes
                for block in plan_shape(shape)
            }
        )
    )


def row_offsets(served_shapes: tuple[Shape, ...]) -> tuple[int, ...]:
    """Return the unique row offsets needed by all served decode shapes.

    :param served_shapes: Complete uniform decode shapes exposed by the runner.
    :returns: Row offsets in deterministic order.
    """
    return tuple(
        sorted(
            {block.row_offset for shape in served_shapes for block in plan_shape(shape)}
        )
    )
