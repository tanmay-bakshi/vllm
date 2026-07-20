# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for canonical coalesced NIXL transport layouts."""

from dataclasses import FrozenInstanceError, replace

import pytest

from vllm.distributed.kv_transfer.coalesced_layout import (
    DUAL_PLANE_DESTINATION_HALF,
    CanonicalSourceTransferPlan,
    CoalescedTransferPlan,
    GroupTransferRoster,
    RegionOwnership,
    RemoteBlockRun,
    SourceGroupTransferRoster,
    SourceRegionOwnership,
    TransferPosition,
    bind_coalesced_transfer_destinations,
    build_canonical_source_plan,
    build_coalesced_transfer_plan,
    build_packed_transfer_plan,
    rank_major_slot_base,
)


def _group(
    group_index: int = 0,
    *,
    source_position_start: int = 0,
    destination_plane_count: int = 2,
    local_block_ids: tuple[int, ...] = (20, 21, 22),
    remote_block_ids: tuple[int, ...] = (10, 11, 12),
) -> GroupTransferRoster:
    return GroupTransferRoster(
        group_index=group_index,
        source_position_start=source_position_start,
        destination_plane_count=destination_plane_count,
        local_block_ids=local_block_ids,
        remote_block_ids=remote_block_ids,
    )


def _region(
    region_index: int = 0,
    *,
    group_indices: tuple[int, ...] = (0,),
    source_row_count: int = 64,
    destination_row_count: int = 64,
    row_bytes: int = 8,
) -> RegionOwnership:
    return RegionOwnership(
        region_index=region_index,
        group_indices=group_indices,
        source_row_count=source_row_count,
        destination_row_count=destination_row_count,
        row_bytes=row_bytes,
    )


def _plan(
    *,
    source_tp_size: int = 4,
    source_ranks: tuple[int, ...] = (1, 3),
    rank_slots: tuple[int, ...] = (1, 0),
    groups: tuple[GroupTransferRoster, ...] | None = None,
    regions: tuple[RegionOwnership, ...] | None = None,
) -> CoalescedTransferPlan:
    return build_coalesced_transfer_plan(
        source_tp_size=source_tp_size,
        source_ranks=source_ranks,
        rank_slots=rank_slots,
        groups=groups if groups is not None else (_group(),),
        regions=regions if regions is not None else (_region(),),
    )


def _source_plan(
    *,
    source_tp_size: int = 4,
    source_ranks: tuple[int, ...] = (1, 3),
    groups: tuple[GroupTransferRoster, ...] | None = None,
    regions: tuple[RegionOwnership, ...] | None = None,
) -> CanonicalSourceTransferPlan:
    destination_groups = groups if groups is not None else (_group(),)
    destination_regions = regions if regions is not None else (_region(),)
    return build_canonical_source_plan(
        source_tp_size=source_tp_size,
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


@pytest.mark.cpu_test
def test_rank_major_slot_base_uses_the_selected_stride_exactly_once() -> None:
    slot_base = 0x1_0000_0000
    rank_stride_bytes = 256 * 1024 * 1024

    assert tuple(
        rank_major_slot_base(slot_base, rank, rank_stride_bytes) for rank in range(4)
    ) == (
        slot_base,
        slot_base + 256 * 1024 * 1024,
        slot_base + 512 * 1024 * 1024,
        slot_base + 768 * 1024 * 1024,
    )


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("slot_base", "rank", "rank_stride_bytes"),
    [
        (0, 0, 1),
        (True, 0, 1),
        (1, -1, 1),
        (1, True, 1),
        (1, 0, 0),
        (1, 0, True),
        ((1 << 64) - 1, 1, 1),
        ((1 << 64) - 1, 0, 2),
    ],
)
def test_rank_major_slot_base_rejects_invalid_or_overflowing_addresses(
    slot_base: int,
    rank: int,
    rank_stride_bytes: int,
) -> None:
    with pytest.raises(ValueError):
        rank_major_slot_base(slot_base, rank, rank_stride_bytes)


def _production_groups(
    position_counts: tuple[int, ...],
) -> tuple[GroupTransferRoster, ...]:
    groups: list[GroupTransferRoster] = []
    remote_cursor = 0
    local_cursor = 0
    for group_index, position_count in enumerate(position_counts):
        destination_plane_count = 2
        local_position_count = position_count
        remote_block_ids = tuple(range(remote_cursor, remote_cursor + position_count))
        local_block_ids = tuple(
            range(local_cursor, local_cursor + local_position_count)
        )
        groups.append(
            _group(
                group_index,
                destination_plane_count=destination_plane_count,
                local_block_ids=local_block_ids,
                remote_block_ids=remote_block_ids,
            )
        )
        remote_cursor += position_count
        local_cursor += local_position_count
    return tuple(groups)


def _production_regions(
    *,
    all_groups_per_region: bool = False,
    region_count: int = 10,
    row_count: int = 64_000,
) -> tuple[RegionOwnership, ...]:
    return tuple(
        _region(
            region_index,
            group_indices=(
                tuple(range(13))
                if all_groups_per_region
                else tuple(range(12))
                if region_index < 5
                else (12,)
            ),
            source_row_count=row_count,
            destination_row_count=row_count,
            row_bytes=65_536,
        )
        for region_index in range(region_count)
    )


@pytest.mark.cpu_test
def test_rank_major_layout_prunes_many_to_many_region_ownership() -> None:
    groups = (
        _group(
            0,
            local_block_ids=(20, 21, 22),
            remote_block_ids=(7, 8, 20),
        ),
        _group(
            1,
            local_block_ids=(30, 31),
            remote_block_ids=(9, 10),
        ),
        _group(
            2,
            local_block_ids=(40,),
            remote_block_ids=(30,),
        ),
    )
    regions = (
        _region(0, group_indices=(0, 1), row_bytes=8),
        _region(1, group_indices=(2,), row_bytes=16),
        _region(2, group_indices=(0, 2), row_bytes=4),
    )

    plan = _plan(groups=groups, regions=regions)

    assert [position.group_index for position in plan.regions[0].positions] == [
        0,
        0,
        1,
        1,
        0,
    ]
    assert [position.remote_block_id for position in plan.regions[0].positions] == [
        7,
        8,
        9,
        10,
        20,
    ]
    assert plan.regions[0].runs == (
        RemoteBlockRun(remote_block_id=7, position_count=4, position_start=0),
        RemoteBlockRun(remote_block_id=20, position_count=1, position_start=4),
    )
    assert [position.group_index for position in plan.regions[1].positions] == [2]
    assert [position.group_index for position in plan.regions[2].positions] == [
        0,
        0,
        0,
        2,
    ]

    assert [region.offset_within_rank for region in plan.regions] == [0, 40, 56]
    assert plan.rank_stride_bytes == 72
    assert plan.staging_size_bytes == 144
    assert plan.rank_offset(0) == 0
    assert plan.rank_offset(1) == 72
    assert plan.region_offset(0, 2) == 56
    assert plan.region_offset(1, 0) == 72
    assert plan.region_offset(1, 2) == 128


@pytest.mark.cpu_test
def test_production_2k_layout_is_exactly_half_the_legacy_cross_product() -> None:
    position_counts = (64,) * 10 + (65, 65, 33)
    groups = _production_groups(position_counts)
    regions = _production_regions()

    plan = _plan(
        source_tp_size=4,
        source_ranks=(0, 1, 2, 3),
        rank_slots=(0, 1, 2, 3),
        groups=groups,
        regions=regions,
    )

    legacy_bytes = 4 * sum(position_counts) * 10 * 65_536
    assert tuple(len(region.positions) for region in plan.regions) == (
        770,
        770,
        770,
        770,
        770,
        33,
        33,
        33,
        33,
        33,
    )
    assert plan.staging_size_bytes == 1_052_508_160
    assert plan.staging_size_bytes * 2 == legacy_bytes
    assert legacy_bytes == 2_105_016_320


@pytest.mark.cpu_test
def test_production_max_context_layout_is_exactly_13600_mib() -> None:
    position_counts = (64,) * 10 + (4_096, 4_096, 2_048)

    plan = _plan(
        source_tp_size=4,
        source_ranks=(0, 1, 2, 3),
        rank_slots=(0, 1, 2, 3),
        groups=_production_groups(position_counts),
        regions=_production_regions(),
    )

    assert plan.rank_stride_bytes == 3_565_158_400
    assert plan.staging_size_bytes == 14_260_633_600
    assert plan.staging_size_bytes == 13_600 * 1024 * 1024


@pytest.mark.cpu_test
def test_all_groups_region_has_no_ownership_pruning_opportunity() -> None:
    position_counts = (64,) * 10 + (65, 65, 33)

    plan = _plan(
        source_tp_size=4,
        source_ranks=(0, 1, 2, 3),
        rank_slots=(0, 1, 2, 3),
        groups=_production_groups(position_counts),
        regions=_production_regions(
            all_groups_per_region=True,
            region_count=1,
        ),
    )

    legacy_bytes = 4 * sum(position_counts) * 65_536
    assert len(plan.regions[0].positions) == sum(position_counts)
    assert plan.staging_size_bytes == legacy_bytes


@pytest.mark.cpu_test
def test_full_prefix_hit_preserves_regions_in_a_zero_byte_plan() -> None:
    groups = _production_groups((0,) * 13)

    plan = _plan(
        source_tp_size=4,
        source_ranks=(0, 1, 2, 3),
        rank_slots=(0, 1, 2, 3),
        groups=groups,
        regions=_production_regions(row_count=1),
    )

    assert len(plan.regions) == 10
    assert all(region.positions == () for region in plan.regions)
    assert all(region.runs == () for region in plan.regions)
    assert all(region.offset_within_rank == 0 for region in plan.regions)
    assert plan.rank_stride_bytes == 0
    assert plan.staging_size_bytes == 0


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("source_position_start", "remote_block_ids", "local_block_ids", "expected"),
    [
        (0, (10, 11, 12, 13), (20, 21), ((20, 0), (20, 1), (21, 0), (21, 1))),
        (2, (10, 11, 12), (20, 21), ((20, 0), (20, 1), (21, 0))),
        (3, (10, 11, 12), (20, 21), ((20, 1), (21, 0), (21, 1))),
        (5, (10,), (20,), ((20, 1),)),
    ],
)
def test_single_plane_uses_absolute_source_position_for_destination_half(
    source_position_start: int,
    remote_block_ids: tuple[int, ...],
    local_block_ids: tuple[int, ...],
    expected: tuple[tuple[int, int], ...],
) -> None:
    group = _group(
        source_position_start=source_position_start,
        destination_plane_count=1,
        local_block_ids=local_block_ids,
        remote_block_ids=remote_block_ids,
    )

    plan = _plan(groups=(group,))

    assert (
        tuple(
            (position.local_block_id, position.destination_half)
            for position in plan.regions[0].positions
        )
        == expected
    )
    assert tuple(
        position.source_position for position in plan.regions[0].positions
    ) == tuple(
        range(source_position_start, source_position_start + len(remote_block_ids))
    )


@pytest.mark.cpu_test
def test_dual_plane_positions_have_no_destination_half() -> None:
    plan = _plan()

    assert plan.regions[0].positions == (
        TransferPosition(0, 0, 10, 20, DUAL_PLANE_DESTINATION_HALF),
        TransferPosition(0, 1, 11, 21, DUAL_PLANE_DESTINATION_HALF),
        TransferPosition(0, 2, 12, 22, DUAL_PLANE_DESTINATION_HALF),
    )


@pytest.mark.cpu_test
def test_duplicate_remote_reads_fail_before_run_formation() -> None:
    group = _group(
        local_block_ids=(20, 21, 22, 23),
        remote_block_ids=(8, 7, 8, 7),
    )

    with pytest.raises(ValueError, match="reads remote block 7 more than once"):
        _plan(groups=(group,))


@pytest.mark.cpu_test
def test_overlapping_destination_writes_fail_before_transport() -> None:
    groups = (
        _group(0, local_block_ids=(20,), remote_block_ids=(10,)),
        _group(1, local_block_ids=(20,), remote_block_ids=(11,)),
    )

    with pytest.raises(ValueError, match="multiple source positions"):
        _plan(
            groups=groups,
            regions=(_region(group_indices=(0, 1)),),
        )


@pytest.mark.cpu_test
def test_zero_position_regions_remain_in_the_canonical_layout() -> None:
    groups = (
        _group(0, local_block_ids=(), remote_block_ids=()),
        _group(1, local_block_ids=(5,), remote_block_ids=(6,)),
    )
    regions = (
        _region(0, group_indices=(0,), row_bytes=8),
        _region(1, group_indices=(1,), row_bytes=16),
        _region(2, group_indices=(0,), row_bytes=32),
    )

    plan = _plan(groups=groups, regions=regions)

    assert plan.regions[0].positions == ()
    assert plan.regions[0].runs == ()
    assert plan.regions[0].offset_within_rank == 0
    assert plan.regions[1].offset_within_rank == 0
    assert plan.regions[2].offset_within_rank == 16
    assert plan.regions[2].positions == ()
    assert plan.rank_stride_bytes == 16
    assert plan.staging_size_bytes == 32


@pytest.mark.cpu_test
def test_plan_and_nested_records_are_immutable() -> None:
    plan = _plan()

    with pytest.raises(FrozenInstanceError):
        plan.rank_stride_bytes = 0
    with pytest.raises(FrozenInstanceError):
        plan.regions[0].offset_within_rank = 0
    with pytest.raises(FrozenInstanceError):
        plan.regions[0].positions[0].local_block_id = 0


@pytest.mark.cpu_test
def test_digest_is_deterministic_and_binds_semantic_layout() -> None:
    plan = _plan()

    assert _plan().digest == plan.digest
    assert len(plan.digest) == 64
    variants = (
        _plan(source_ranks=(0, 2), rank_slots=(1, 0)),
        _plan(rank_slots=(0, 1)),
        _plan(groups=(replace(_group(), source_position_start=4),)),
        _plan(groups=(replace(_group(), local_block_ids=(23, 24, 25)),)),
        _plan(groups=(replace(_group(), remote_block_ids=(11, 12, 13)),)),
        _plan(regions=(replace(_region(), row_bytes=16),)),
    )
    assert all(variant.digest != plan.digest for variant in variants)


@pytest.mark.cpu_test
def test_canonical_source_binding_preserves_direct_layout_and_digest() -> None:
    groups = (_group(),)
    regions = (_region(),)
    source_plan = _source_plan(groups=groups, regions=regions)

    bound_plan = bind_coalesced_transfer_destinations(
        source_plan,
        rank_slots=(1, 0),
        groups=groups,
        regions=regions,
    )
    compatibility_plan = _plan(groups=groups, regions=regions)

    assert bound_plan == compatibility_plan
    assert bound_plan.source_digest == source_plan.digest
    assert bound_plan.digest == (
        "5cac20e70b360908ee3786436283312073952c5055ff52c12c3671abdefae1d0"
    )


@pytest.mark.cpu_test
def test_source_digest_is_destination_independent_and_direct_digest_is_not() -> None:
    groups = (_group(),)
    regions = (_region(),)
    destination_variant_groups = (replace(_group(), local_block_ids=(23, 24, 25)),)
    destination_variant_regions = (replace(_region(), destination_row_count=96),)
    source_plan = _source_plan(groups=groups, regions=regions)
    independently_built_source_plan = _source_plan(
        groups=destination_variant_groups,
        regions=destination_variant_regions,
    )

    assert independently_built_source_plan == source_plan
    direct_plan = bind_coalesced_transfer_destinations(
        source_plan,
        rank_slots=(1, 0),
        groups=groups,
        regions=regions,
    )
    destination_variant_plan = bind_coalesced_transfer_destinations(
        source_plan,
        rank_slots=(0, 1),
        groups=destination_variant_groups,
        regions=destination_variant_regions,
    )

    assert direct_plan.source_digest == destination_variant_plan.source_digest
    assert direct_plan.digest != destination_variant_plan.digest


@pytest.mark.cpu_test
def test_destination_binding_rejects_a_different_source_selection() -> None:
    source_plan = _source_plan()
    mismatched_groups = (replace(_group(), remote_block_ids=(11, 12, 13)),)

    with pytest.raises(ValueError, match="does not match the canonical source plan"):
        bind_coalesced_transfer_destinations(
            source_plan,
            rank_slots=(1, 0),
            groups=mismatched_groups,
            regions=(_region(),),
        )


@pytest.mark.cpu_test
def test_packed_chunks_are_bounded_row_aligned_and_cover_every_position_once() -> None:
    groups = (
        _group(
            0,
            local_block_ids=(20, 21, 22),
            remote_block_ids=(2, 3, 8),
        ),
        _group(
            1,
            local_block_ids=(30, 31),
            remote_block_ids=(10, 11),
        ),
    )
    regions = (
        _region(0, group_indices=(0,), row_bytes=8),
        _region(1, group_indices=(1,), row_bytes=12),
    )
    source_plan = _source_plan(
        source_ranks=(0, 2),
        groups=groups,
        regions=regions,
    )

    packed_plan = build_packed_transfer_plan(
        source_plan,
        max_chunk_bytes_per_rank=20,
    )

    assert tuple(chunk.rank_stride_bytes for chunk in packed_plan.chunks) == (
        16,
        20,
        12,
    )
    assert tuple(
        tuple(
            (
                region_slice.region_index,
                region_slice.position_start,
                region_slice.position_count,
                region_slice.offset_within_rank,
                region_slice.size_bytes,
            )
            for region_slice in chunk.region_slices
        )
        for chunk in packed_plan.chunks
    ) == (
        ((0, 0, 2, 0, 16),),
        ((0, 2, 1, 0, 8), (1, 0, 1, 8, 12)),
        ((1, 1, 1, 0, 12),),
    )

    covered_positions: list[tuple[int, int]] = []
    for chunk in packed_plan.chunks:
        expected_offset = 0
        assert chunk.rank_stride_bytes <= packed_plan.max_chunk_bytes_per_rank
        assert chunk.transfer_size_bytes == (
            chunk.rank_stride_bytes * len(source_plan.source_ranks)
        )
        for region_slice in chunk.region_slices:
            region = source_plan.regions[region_slice.region_index]
            assert region_slice.offset_within_rank == expected_offset
            assert region_slice.size_bytes == (
                region_slice.position_count * region.ownership.row_bytes
            )
            covered_positions.extend(
                (region_slice.region_index, position_index)
                for position_index in range(
                    region_slice.position_start,
                    region_slice.position_start + region_slice.position_count,
                )
            )
            expected_offset += region_slice.size_bytes
        assert expected_offset == chunk.rank_stride_bytes

    expected_positions = [
        (region.ownership.region_index, position_index)
        for region in source_plan.regions
        for position_index in range(len(region.positions))
    ]
    assert covered_positions == expected_positions
    assert packed_plan.rank_stride_bytes == source_plan.rank_stride_bytes == 48
    assert packed_plan.transfer_size_bytes == source_plan.transfer_size_bytes == 96


@pytest.mark.cpu_test
def test_packed_digest_is_deterministic_and_bound_to_source_and_limit() -> None:
    groups = (
        _group(
            0,
            local_block_ids=(20, 21, 22),
            remote_block_ids=(2, 3, 8),
        ),
        _group(
            1,
            local_block_ids=(30, 31),
            remote_block_ids=(10, 11),
        ),
    )
    regions = (
        _region(0, group_indices=(0,), row_bytes=8),
        _region(1, group_indices=(1,), row_bytes=12),
    )
    source_plan = _source_plan(groups=groups, regions=regions)
    packed_plan = build_packed_transfer_plan(
        source_plan,
        max_chunk_bytes_per_rank=20,
    )

    assert (
        build_packed_transfer_plan(
            source_plan,
            max_chunk_bytes_per_rank=20,
        )
        == packed_plan
    )
    assert packed_plan.source_digest == source_plan.digest
    assert len(packed_plan.digest) == 64
    assert all(len(chunk.digest) == 64 for chunk in packed_plan.chunks)

    different_limit_plan = build_packed_transfer_plan(
        source_plan,
        max_chunk_bytes_per_rank=21,
    )
    assert different_limit_plan.chunks == packed_plan.chunks
    assert different_limit_plan.digest != packed_plan.digest

    changed_groups = (
        replace(groups[0], remote_block_ids=(3, 4, 9)),
        groups[1],
    )
    changed_source_plan = _source_plan(groups=changed_groups, regions=regions)
    changed_packed_plan = build_packed_transfer_plan(
        changed_source_plan,
        max_chunk_bytes_per_rank=20,
    )
    assert changed_packed_plan.source_digest != packed_plan.source_digest
    assert changed_packed_plan.digest != packed_plan.digest
    assert tuple(chunk.digest for chunk in changed_packed_plan.chunks) != tuple(
        chunk.digest for chunk in packed_plan.chunks
    )


@pytest.mark.cpu_test
def test_zero_byte_source_has_a_valid_empty_packed_plan() -> None:
    source_plan = _source_plan(
        groups=(_group(local_block_ids=(), remote_block_ids=()),),
        regions=(_region(source_row_count=1),),
    )

    packed_plan = build_packed_transfer_plan(
        source_plan,
        max_chunk_bytes_per_rank=8,
    )

    assert packed_plan.chunks == ()
    assert packed_plan.rank_stride_bytes == 0
    assert packed_plan.transfer_size_bytes == 0
    assert packed_plan.source_digest == source_plan.digest


@pytest.mark.cpu_test
@pytest.mark.parametrize("max_chunk_bytes_per_rank", [0, True])
def test_packed_plan_rejects_invalid_chunk_bounds(
    max_chunk_bytes_per_rank: int,
) -> None:
    with pytest.raises(ValueError, match="max_chunk_bytes_per_rank"):
        build_packed_transfer_plan(
            _source_plan(),
            max_chunk_bytes_per_rank=max_chunk_bytes_per_rank,
        )


@pytest.mark.cpu_test
def test_packed_plan_rejects_a_row_larger_than_the_chunk_bound() -> None:
    with pytest.raises(ValueError, match="row size 8 exceeds"):
        build_packed_transfer_plan(
            _source_plan(),
            max_chunk_bytes_per_rank=7,
        )


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"source_tp_size": 0}, "source_tp_size"),
        ({"source_ranks": ()}, "non-empty"),
        ({"source_ranks": (1, 1)}, "duplicates"),
        ({"source_ranks": (1, 4)}, "outside the source TP"),
        ({"rank_slots": (0,)}, "match source_ranks"),
        ({"rank_slots": (0, 0)}, "complete bijection"),
        ({"rank_slots": (0, 2)}, "complete bijection"),
        ({"groups": ()}, "groups must be a non-empty tuple"),
        ({"regions": ()}, "regions must be a non-empty tuple"),
        (
            {"groups": (replace(_group(), group_index=1),)},
            "group indices must be canonical",
        ),
        (
            {"groups": (replace(_group(), destination_plane_count=3),)},
            "plane count",
        ),
        (
            {"groups": (replace(_group(), destination_plane_count=True),)},
            "destination_plane_count",
        ),
        (
            {"groups": (replace(_group(), source_position_start=-1),)},
            "source_position_start",
        ),
        (
            {"groups": (replace(_group(), local_block_ids=(20, 21)),)},
            "local blocks, expected 3",
        ),
        (
            {
                "groups": (
                    replace(
                        _group(),
                        destination_plane_count=1,
                        local_block_ids=(20,),
                    ),
                )
            },
            "local blocks, expected 2",
        ),
        (
            {"regions": (replace(_region(), region_index=1),)},
            "region indices must be canonical",
        ),
        (
            {"regions": (replace(_region(), group_indices=()),)},
            "must own at least one group",
        ),
        (
            {"regions": (replace(_region(), group_indices=(0, 0)),)},
            "sorted and unique",
        ),
        (
            {"regions": (replace(_region(), group_indices=(1,)),)},
            "out-of-range group",
        ),
        (
            {"regions": (replace(_region(), row_bytes=7),)},
            "row size must be even",
        ),
        (
            {"regions": (replace(_region(), source_row_count=12),)},
            "remote block 12",
        ),
        (
            {"regions": (replace(_region(), destination_row_count=22),)},
            "local block 22",
        ),
    ],
)
def test_invalid_layouts_fail_closed(kwargs: dict[str, object], match: str) -> None:
    defaults: dict[str, object] = {
        "source_tp_size": 4,
        "source_ranks": (1, 3),
        "rank_slots": (1, 0),
        "groups": (_group(),),
        "regions": (_region(),),
    }
    defaults.update(kwargs)

    with pytest.raises(ValueError, match=match):
        build_coalesced_transfer_plan(**defaults)


@pytest.mark.cpu_test
def test_region_ownership_must_cover_every_group() -> None:
    groups = (_group(0), _group(1))

    with pytest.raises(ValueError, match="cover every group"):
        _plan(groups=groups, regions=(_region(group_indices=(0,)),))


@pytest.mark.cpu_test
def test_region_rejects_mixed_destination_plane_layouts() -> None:
    groups = (
        _group(0, destination_plane_count=2),
        _group(
            1,
            destination_plane_count=1,
            local_block_ids=(30, 31),
            remote_block_ids=(13, 14, 15),
        ),
    )

    with pytest.raises(ValueError, match="mixes destination plane layouts"):
        _plan(groups=groups, regions=(_region(group_indices=(0, 1)),))


@pytest.mark.cpu_test
def test_regions_accept_uniform_single_and_dual_plane_owners() -> None:
    groups = (
        _group(
            0,
            destination_plane_count=1,
            local_block_ids=(20,),
            remote_block_ids=(10, 11),
        ),
        _group(
            1,
            destination_plane_count=1,
            local_block_ids=(21,),
            remote_block_ids=(12, 13),
        ),
        _group(
            2,
            destination_plane_count=2,
            local_block_ids=(30, 31),
            remote_block_ids=(20, 21),
        ),
        _group(
            3,
            destination_plane_count=2,
            local_block_ids=(32, 33),
            remote_block_ids=(22, 23),
        ),
    )
    regions = (
        _region(0, group_indices=(0, 1)),
        _region(1, group_indices=(2, 3)),
    )

    plan = _plan(groups=groups, regions=regions)

    assert tuple(
        {
            groups[position.group_index].destination_plane_count
            for position in region.positions
        }
        for region in plan.regions
    ) == ({1}, {2})


@pytest.mark.cpu_test
def test_plan_offset_accessors_reject_out_of_range_indices() -> None:
    plan = _plan()

    with pytest.raises(IndexError, match="rank index"):
        plan.rank_offset(-1)
    with pytest.raises(IndexError, match="rank index"):
        plan.region_offset(2, 0)
    with pytest.raises(IndexError, match="region index"):
        plan.region_offset(0, 1)
