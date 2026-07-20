# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end tests for producer gather and decoder scatter around packed WRITEs."""

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from vllm.distributed.kv_transfer.coalesced_layout import (
    CanonicalSourceTransferPlan,
    CoalescedTransferPlan,
    GroupTransferRoster,
    PackedTransferPlan,
    RegionOwnership,
    SourceGroupTransferRoster,
    SourceRegionOwnership,
    build_canonical_source_plan,
    build_coalesced_transfer_plan,
    build_packed_transfer_plan,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.coalesced_pack import (
    launch_packed_chunk_pack,
    pack_chunk_reference,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.coalesced_scatter import (
    launch_packed_chunk_scatter,
    scatter_coalesced_reference,
    scatter_packed_chunk_reference,
)
from vllm.triton_utils import HAS_TRITON

_CANARY = 0xD7
_REDZONE_BYTES = 9
_STAGING_BASE = 11
_MAX_CHUNK_BYTES_PER_RANK = 28
_STAGING_RANK_STRIDE_BYTES = 34


def _plans() -> tuple[
    CanonicalSourceTransferPlan,
    CoalescedTransferPlan,
    PackedTransferPlan,
]:
    """Build odd-prefix, mixed-plane, multi-region transfer plans.

    :returns: Canonical source, destination-bound, and packed plans.
    """
    destination_groups = (
        GroupTransferRoster(
            group_index=0,
            source_position_start=0,
            destination_plane_count=2,
            local_block_ids=(2, 0, 3),
            remote_block_ids=(5, 3, 6),
        ),
        GroupTransferRoster(
            group_index=1,
            source_position_start=3,
            destination_plane_count=1,
            local_block_ids=(1, 2),
            remote_block_ids=(8, 4, 7),
        ),
        GroupTransferRoster(
            group_index=2,
            source_position_start=0,
            destination_plane_count=2,
            local_block_ids=(2, 0),
            remote_block_ids=(9, 1),
        ),
    )
    destination_regions = (
        RegionOwnership(
            region_index=0,
            group_indices=(0,),
            source_row_count=12,
            destination_row_count=4,
            row_bytes=10,
        ),
        RegionOwnership(
            region_index=1,
            group_indices=(1,),
            source_row_count=12,
            destination_row_count=4,
            row_bytes=14,
        ),
        RegionOwnership(
            region_index=2,
            group_indices=(2,),
            source_row_count=12,
            destination_row_count=3,
            row_bytes=18,
        ),
    )
    source_ranks = (3, 1, 0, 2)
    source_plan = build_canonical_source_plan(
        source_tp_size=4,
        source_ranks=source_ranks,
        groups=tuple(
            SourceGroupTransferRoster(
                group_index=group.group_index,
                source_position_start=group.source_position_start,
                remote_block_ids=group.remote_block_ids,
            )
            for group in destination_groups
        ),
        regions=tuple(
            SourceRegionOwnership(
                region_index=region.region_index,
                group_indices=region.group_indices,
                source_row_count=region.source_row_count,
                row_bytes=region.row_bytes,
            )
            for region in destination_regions
        ),
    )
    destination_plan = build_coalesced_transfer_plan(
        source_tp_size=4,
        source_ranks=source_ranks,
        rank_slots=(2, 0, 3, 1),
        groups=destination_groups,
        regions=destination_regions,
    )
    packed_plan = build_packed_transfer_plan(
        source_plan,
        max_chunk_bytes_per_rank=_MAX_CHUNK_BYTES_PER_RANK,
    )
    return source_plan, destination_plan, packed_plan


def _rank_sources(
    source_plan: CanonicalSourceTransferPlan,
) -> tuple[tuple[torch.Tensor, ...], ...]:
    """Build deterministic registered source regions for every producer rank.

    :param source_plan: Canonical source geometry.
    :returns: Region tensors grouped by source-rank transfer position.
    """
    rank_sources: list[tuple[torch.Tensor, ...]] = []
    for source_rank in source_plan.source_ranks:
        regions: list[torch.Tensor] = []
        for region in source_plan.regions:
            source_bytes = (
                region.ownership.source_row_count * region.ownership.row_bytes
            )
            values = (
                torch.arange(source_bytes, dtype=torch.int64)
                + source_rank * 43
                + region.ownership.region_index * 67
            ) % 251
            regions.append(values.to(torch.uint8))
        rank_sources.append(tuple(regions))
    return tuple(rank_sources)


def _direct_staging(
    source_plan: CanonicalSourceTransferPlan,
    destination_plan: CoalescedTransferPlan,
    rank_sources: tuple[tuple[torch.Tensor, ...], ...],
) -> torch.Tensor:
    """Materialize the direct rank-major reference transfer.

    :param source_plan: Canonical source selection.
    :param destination_plan: Destination-bound direct layout.
    :param rank_sources: Registered source regions for every producer rank.
    :returns: Direct staging allocation surrounded by canaries.
    """
    staging = torch.full(
        (_STAGING_BASE + destination_plan.staging_size_bytes + _REDZONE_BYTES,),
        _CANARY,
        dtype=torch.uint8,
    )
    for rank_index, sources in enumerate(rank_sources):
        for source_region in source_plan.regions:
            row_bytes = source_region.ownership.row_bytes
            source = sources[source_region.ownership.region_index]
            destination_start = (
                _STAGING_BASE
                + destination_plan.rank_offset(rank_index)
                + source_region.offset_within_rank
            )
            for position_index, position in enumerate(source_region.positions):
                source_start = position.remote_block_id * row_bytes
                row_start = destination_start + position_index * row_bytes
                staging[row_start : row_start + row_bytes].copy_(
                    source[source_start : source_start + row_bytes]
                )
    return staging


def _destination_backings(
    plan: CoalescedTransferPlan,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Allocate exact destination views surrounded by canaries.

    :param plan: Destination-bound transfer plan.
    :returns: Destination backings followed by exact region views.
    """
    backings: list[torch.Tensor] = []
    destinations: list[torch.Tensor] = []
    for region in plan.regions:
        destination_bytes = (
            region.ownership.destination_row_count
            * len(plan.source_ranks)
            * region.ownership.row_bytes
        )
        backing = torch.full(
            (destination_bytes + 2 * _REDZONE_BYTES,),
            _CANARY,
            dtype=torch.uint8,
        )
        backings.append(backing)
        destinations.append(
            backing[_REDZONE_BYTES : _REDZONE_BYTES + destination_bytes]
        )
    return tuple(backings), tuple(destinations)


def _packed_staging_size(source_rank_count: int) -> int:
    """Return a guarded fixed-lane packed staging allocation size.

    :param source_rank_count: Number of source-rank lanes.
    :returns: Required byte count.
    """
    return (
        _STAGING_BASE
        + (source_rank_count - 1) * _STAGING_RANK_STRIDE_BYTES
        + _MAX_CHUNK_BYTES_PER_RANK
        + _REDZONE_BYTES
    )


def _assert_destination_redzones(backings: tuple[torch.Tensor, ...]) -> None:
    """Verify that packed scatter did not touch adjacent destination storage.

    :param backings: Guarded destination backing tensors.
    """
    for backing in backings:
        assert torch.all(backing[:_REDZONE_BYTES] == _CANARY)
        assert torch.all(backing[-_REDZONE_BYTES:] == _CANARY)


def _assert_packed_staging_guards(
    staging: torch.Tensor,
    source_rank_count: int,
    allowed_bytes_per_rank: int,
) -> None:
    """Verify all bytes outside fixed producer-rank lanes remain canaries.

    :param staging: Guarded packed staging allocation.
    :param source_rank_count: Number of source-rank lanes.
    :param allowed_bytes_per_rank: Writable bytes at the start of every lane.
    """
    writable = torch.zeros(staging.numel(), dtype=torch.bool)
    for rank_index in range(source_rank_count):
        rank_start = _STAGING_BASE + rank_index * _STAGING_RANK_STRIDE_BYTES
        writable[rank_start : rank_start + allowed_bytes_per_rank] = True
    assert torch.all(staging[~writable] == _CANARY)


def _replace_chunk(
    packed_plan: PackedTransferPlan,
    selected_chunk_index: int,
    **changes: object,
) -> PackedTransferPlan:
    """Replace one immutable chunk without repairing any integrity digest.

    :param packed_plan: Original valid packed plan.
    :param selected_chunk_index: Chunk to corrupt.
    :param changes: Fields to replace on the selected chunk.
    :returns: Packed plan containing the corrupted chunk.
    """
    chunks = list(packed_plan.chunks)
    chunks[selected_chunk_index] = replace(
        chunks[selected_chunk_index],
        **changes,
    )
    return replace(packed_plan, chunks=tuple(chunks))


def _malformed_plan(
    packed_plan: PackedTransferPlan,
    case: str,
) -> PackedTransferPlan:
    """Construct one deliberately malformed packed plan.

    :param packed_plan: Original valid packed plan.
    :param case: Corruption case name.
    :returns: Corrupted immutable plan.
    """
    chunk = packed_plan.chunks[0]
    region_slice = chunk.region_slices[0]
    if case == "source_digest":
        return replace(packed_plan, source_digest="0" * 64)
    if case == "plan_digest":
        return replace(packed_plan, digest="0" * 64)
    if case == "chunk_digest":
        return _replace_chunk(packed_plan, 0, digest="0" * 64)
    if case == "chunk_ordinal":
        return _replace_chunk(packed_plan, 0, chunk_index=1)
    if case == "chunk_stride":
        rank_stride_bytes = chunk.rank_stride_bytes + 2
        return _replace_chunk(
            packed_plan,
            0,
            rank_stride_bytes=rank_stride_bytes,
            transfer_size_bytes=rank_stride_bytes * len(packed_plan.source_ranks),
        )
    if case == "slice_start":
        corrupted_slice = replace(region_slice, position_start=-1)
    elif case == "slice_size":
        corrupted_slice = replace(
            region_slice,
            size_bytes=region_slice.size_bytes + 2,
        )
    elif case == "slice_offset":
        corrupted_slice = replace(region_slice, offset_within_rank=1)
    elif case == "coverage":
        return replace(packed_plan, chunks=packed_plan.chunks[:-1])
    else:
        raise ValueError(f"unknown malformed-plan case: {case}")
    return _replace_chunk(
        packed_plan,
        0,
        region_slices=(corrupted_slice, *chunk.region_slices[1:]),
    )


@pytest.mark.cpu_test
def test_packed_pipeline_matches_direct_reference_and_preserves_redzones() -> None:
    """Match the direct oracle across chunks, planes, ranks, and regions."""
    source_plan, destination_plan, packed_plan = _plans()
    assert source_plan.digest == destination_plan.source_digest
    rank_sources = _rank_sources(source_plan)
    source_snapshots = tuple(
        tuple(source.clone() for source in sources) for sources in rank_sources
    )

    direct_staging = _direct_staging(
        source_plan,
        destination_plan,
        rank_sources,
    )
    direct_staging_snapshot = direct_staging.clone()
    direct_backings, direct_destinations = _destination_backings(destination_plan)
    scatter_coalesced_reference(
        direct_staging,
        direct_destinations,
        destination_plan,
        staging_base_offset_bytes=_STAGING_BASE,
    )

    packed_backings, packed_destinations = _destination_backings(destination_plan)
    packed_staging = torch.full(
        (_packed_staging_size(len(source_plan.source_ranks)),),
        _CANARY,
        dtype=torch.uint8,
    )
    for chunk in packed_plan.chunks:
        packed_staging.fill_(_CANARY)
        for rank_index, sources in enumerate(rank_sources):
            pack_chunk_reference(
                packed_staging,
                sources,
                source_plan,
                packed_plan,
                chunk.chunk_index,
                destination_base_offset_bytes=(
                    _STAGING_BASE + rank_index * _STAGING_RANK_STRIDE_BYTES
                ),
            )
        _assert_packed_staging_guards(
            packed_staging,
            len(source_plan.source_ranks),
            chunk.rank_stride_bytes,
        )
        packed_staging_snapshot = packed_staging.clone()
        scatter_packed_chunk_reference(
            packed_staging,
            packed_destinations,
            destination_plan,
            packed_plan,
            chunk.chunk_index,
            staging_base_offset_bytes=_STAGING_BASE,
            staging_rank_stride_bytes=_STAGING_RANK_STRIDE_BYTES,
        )
        assert torch.equal(packed_staging, packed_staging_snapshot)

    assert torch.equal(direct_staging, direct_staging_snapshot)
    _assert_destination_redzones(direct_backings)
    _assert_destination_redzones(packed_backings)
    for direct, packed in zip(direct_backings, packed_backings, strict=True):
        assert torch.equal(packed, direct)
    for sources, snapshots in zip(rank_sources, source_snapshots, strict=True):
        for source, snapshot in zip(sources, snapshots, strict=True):
            assert torch.equal(source, snapshot)

    single_plane_destination = packed_destinations[1].view(4, 4, 2, 7)
    assert torch.all(single_plane_destination[1, :, 0] == _CANARY)


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("source_digest", "source digest"),
        ("plan_digest", "plan digest"),
        ("chunk_digest", "chunk digest"),
        ("chunk_ordinal", "indices"),
        ("chunk_stride", "rank stride"),
        ("slice_start", "position_start"),
        ("slice_size", "slice size"),
        ("slice_offset", "byte-contiguous"),
        ("coverage", "cover every"),
    ],
)
def test_pack_and_scatter_reject_malformed_plans_before_writing(
    case: str,
    match: str,
) -> None:
    """Reject every corrupted binding before gather or scatter mutates memory."""
    source_plan, destination_plan, packed_plan = _plans()
    malformed_plan = _malformed_plan(packed_plan, case)
    rank_sources = _rank_sources(source_plan)
    staging = torch.full(
        (_packed_staging_size(len(source_plan.source_ranks)),),
        _CANARY,
        dtype=torch.uint8,
    )
    staging_snapshot = staging.clone()
    backings, destinations = _destination_backings(destination_plan)
    backing_snapshots = tuple(backing.clone() for backing in backings)

    with pytest.raises(ValueError, match=match):
        pack_chunk_reference(
            staging,
            rank_sources[0],
            source_plan,
            malformed_plan,
            0,
            destination_base_offset_bytes=_STAGING_BASE,
        )
    assert torch.equal(staging, staging_snapshot)

    with pytest.raises(ValueError, match=match):
        scatter_packed_chunk_reference(
            staging,
            destinations,
            destination_plan,
            malformed_plan,
            0,
            staging_base_offset_bytes=_STAGING_BASE,
            staging_rank_stride_bytes=_STAGING_RANK_STRIDE_BYTES,
        )
    assert torch.equal(staging, staging_snapshot)
    for backing, snapshot in zip(backings, backing_snapshots, strict=True):
        assert torch.equal(backing, snapshot)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not HAS_TRITON,
    reason="CUDA and Triton are required",
)
def test_cuda_packed_pipeline_matches_direct_cpu_reference() -> None:
    """Match CUDA gather and scatter to the complete direct CPU oracle."""
    source_plan, destination_plan, packed_plan = _plans()
    rank_sources_cpu = _rank_sources(source_plan)
    direct_staging = _direct_staging(
        source_plan,
        destination_plan,
        rank_sources_cpu,
    )
    direct_backings, direct_destinations = _destination_backings(destination_plan)
    scatter_coalesced_reference(
        direct_staging,
        direct_destinations,
        destination_plan,
        staging_base_offset_bytes=_STAGING_BASE,
    )

    rank_sources_cuda = tuple(
        tuple(source.cuda() for source in sources) for sources in rank_sources_cpu
    )
    staging = torch.full(
        (_packed_staging_size(len(source_plan.source_ranks)),),
        _CANARY,
        dtype=torch.uint8,
        device="cuda",
    )
    cuda_backings = tuple(backing.cuda() for backing in direct_backings)
    for backing in cuda_backings:
        backing.fill_(_CANARY)
    cuda_destinations = tuple(
        backing[_REDZONE_BYTES : backing.numel() - _REDZONE_BYTES]
        for backing in cuda_backings
    )
    stream = torch.cuda.Stream(device=staging.device)
    stream.wait_stream(torch.cuda.current_stream(staging.device))
    pack_launches = []
    scatter_launches = []
    with (
        patch.object(
            torch.cuda,
            "synchronize",
            side_effect=AssertionError("device-wide synchronization is forbidden"),
        ),
        patch.object(
            torch.accelerator,
            "synchronize",
            side_effect=AssertionError("accelerator synchronization is forbidden"),
        ),
    ):
        for chunk in packed_plan.chunks:
            for rank_index, sources in enumerate(rank_sources_cuda):
                pack_launches.append(
                    launch_packed_chunk_pack(
                        staging,
                        sources,
                        source_plan,
                        packed_plan,
                        chunk.chunk_index,
                        stream,
                        destination_base_offset_bytes=(
                            _STAGING_BASE + rank_index * _STAGING_RANK_STRIDE_BYTES
                        ),
                    )
                )
            scatter_launches.append(
                launch_packed_chunk_scatter(
                    staging,
                    cuda_destinations,
                    destination_plan,
                    packed_plan,
                    chunk.chunk_index,
                    stream,
                    staging_base_offset_bytes=_STAGING_BASE,
                    staging_rank_stride_bytes=_STAGING_RANK_STRIDE_BYTES,
                )
            )

    scatter_launches[-1].completion_event.synchronize()
    assert all(launch.is_complete() for launch in pack_launches)
    assert all(launch.is_complete() for launch in scatter_launches)
    _assert_packed_staging_guards(
        staging.cpu(),
        len(source_plan.source_ranks),
        _MAX_CHUNK_BYTES_PER_RANK,
    )
    for actual, expected in zip(cuda_backings, direct_backings, strict=True):
        assert torch.equal(actual.cpu(), expected)
    for cuda_sources, cpu_sources in zip(
        rank_sources_cuda,
        rank_sources_cpu,
        strict=True,
    ):
        for actual, expected in zip(cuda_sources, cpu_sources, strict=True):
            assert torch.equal(actual.cpu(), expected)
