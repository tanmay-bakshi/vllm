# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Canonical transport layouts for coalesced KV-cache reads."""

import hashlib
import json
from dataclasses import dataclass

DUAL_PLANE_DESTINATION_HALF = -1


@dataclass(frozen=True, slots=True)
class GroupTransferRoster:
    """Exact post-prefix-transfer roster for one KV cache group.

    :ivar group_index: Canonical cache-group index.
    :ivar source_position_start: Absolute first selected source position.
    :ivar destination_plane_count: One for packed half-row placement, two for
        complete K/V-row placement.
    :ivar local_block_ids: Selected destination block roster.
    :ivar remote_block_ids: Selected source block roster.
    """

    group_index: int
    source_position_start: int
    destination_plane_count: int
    local_block_ids: tuple[int, ...]
    remote_block_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RegionOwnership:
    """Authoritative group ownership and row geometry for one region.

    :ivar region_index: Canonical registered-region index.
    :ivar group_indices: Cache groups whose rows exist in this region.
    :ivar source_row_count: Registered source rows.
    :ivar destination_row_count: Registered destination rows.
    :ivar row_bytes: Bytes in one producer-rank source row.
    """

    region_index: int
    group_indices: tuple[int, ...]
    source_row_count: int
    destination_row_count: int
    row_bytes: int


@dataclass(frozen=True, slots=True)
class TransferPosition:
    """One source-row to destination-row placement.

    :ivar group_index: Owning cache-group index.
    :ivar source_position: Absolute position within the source group roster.
    :ivar remote_block_id: Registered producer row.
    :ivar local_block_id: Registered decoder row.
    :ivar destination_half: Packed destination half, or
        :data:`DUAL_PLANE_DESTINATION_HALF` for a complete row.
    """

    group_index: int
    source_position: int
    remote_block_id: int
    local_block_id: int
    destination_half: int


@dataclass(frozen=True, slots=True)
class RemoteBlockRun:
    """A maximal consecutive source-row run within one region.

    :ivar remote_block_id: First source row in the run.
    :ivar position_count: Consecutive row count.
    :ivar position_start: First packed-position index in the region.
    """

    remote_block_id: int
    position_count: int
    position_start: int


@dataclass(frozen=True, slots=True)
class RegionTransferLayout:
    """Packed positions and runs for one region within every rank slab.

    :ivar ownership: Authenticated semantic and physical region contract.
    :ivar offset_within_rank: Region byte offset within each rank slab.
    :ivar positions: Source-sorted placement records.
    :ivar runs: Maximal contiguous source runs over ``positions``.
    """

    ownership: RegionOwnership
    offset_within_rank: int
    positions: tuple[TransferPosition, ...]
    runs: tuple[RemoteBlockRun, ...]

    @property
    def size_bytes(self) -> int:
        """Return this region's size within one rank slab."""
        return len(self.positions) * self.ownership.row_bytes


@dataclass(frozen=True, slots=True)
class CoalescedTransferPlan:
    """Canonical rank-major staging layout for one coalesced request.

    :ivar source_tp_size: Complete producer tensor-parallel size.
    :ivar source_ranks: Participating producer ranks in staging order.
    :ivar rank_slots: Decoder attention slots paired with ``source_ranks``.
    :ivar groups: Exact selected group rosters.
    :ivar regions: Region-owned packed layouts.
    :ivar rank_stride_bytes: Bytes reserved for each producer rank.
    :ivar staging_size_bytes: Exact complete staging allocation size.
    :ivar digest: SHA-256 identity of every semantic placement field.
    """

    source_tp_size: int
    source_ranks: tuple[int, ...]
    rank_slots: tuple[int, ...]
    groups: tuple[GroupTransferRoster, ...]
    regions: tuple[RegionTransferLayout, ...]
    rank_stride_bytes: int
    staging_size_bytes: int
    digest: str

    def rank_offset(self, rank_index: int) -> int:
        """Return the start of a source rank's staging slab.

        :param rank_index: Position of the source rank in ``source_ranks``.
        :returns: Byte offset from the start of the request's staging allocation.
        """
        if rank_index < 0 or rank_index >= len(self.source_ranks):
            raise IndexError("rank index is outside the transfer plan")
        return rank_index * self.rank_stride_bytes

    def region_offset(self, rank_index: int, region_index: int) -> int:
        """Return a region's offset within a source rank's staging slab.

        :param rank_index: Position of the source rank in ``source_ranks``.
        :param region_index: Canonical region index.
        :returns: Byte offset from the start of the request's staging allocation.
        """
        if region_index < 0 or region_index >= len(self.regions):
            raise IndexError("region index is outside the transfer plan")
        return (
            self.rank_offset(rank_index) + self.regions[region_index].offset_within_rank
        )


def _require_int(name: str, value: int, *, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(
            f"{name} must be an integer greater than or equal to {minimum}"
        )


def _require_block_ids(name: str, block_ids: tuple[int, ...]) -> None:
    if type(block_ids) is not tuple:
        raise ValueError(f"{name} must be a tuple")
    for index, block_id in enumerate(block_ids):
        _require_int(f"{name}[{index}]", block_id)


def _validate_groups(groups: tuple[GroupTransferRoster, ...]) -> None:
    if type(groups) is not tuple or len(groups) == 0:
        raise ValueError("groups must be a non-empty tuple")
    for expected_index, group in enumerate(groups):
        if type(group) is not GroupTransferRoster:
            raise ValueError("groups must contain GroupTransferRoster records")
        _require_int(f"groups[{expected_index}].group_index", group.group_index)
        if group.group_index != expected_index:
            raise ValueError("group indices must be canonical and contiguous")
        _require_int(
            f"groups[{expected_index}].source_position_start",
            group.source_position_start,
        )
        _require_int(
            f"groups[{expected_index}].destination_plane_count",
            group.destination_plane_count,
            minimum=1,
        )
        if group.destination_plane_count not in (1, 2):
            raise ValueError("destination plane count must be one or two")
        _require_block_ids(
            f"groups[{expected_index}].local_block_ids",
            group.local_block_ids,
        )
        _require_block_ids(
            f"groups[{expected_index}].remote_block_ids",
            group.remote_block_ids,
        )
        remote_count = len(group.remote_block_ids)
        if group.destination_plane_count == 2:
            expected_local_count = remote_count
        elif remote_count == 0:
            expected_local_count = 0
        else:
            first_local_position = group.source_position_start // 2
            final_source_position = group.source_position_start + remote_count - 1
            expected_local_count = final_source_position // 2 - first_local_position + 1
        if len(group.local_block_ids) != expected_local_count:
            raise ValueError(
                f"group {expected_index} has {len(group.local_block_ids)} local "
                f"blocks, expected {expected_local_count}"
            )


def _validate_regions(
    regions: tuple[RegionOwnership, ...],
    groups: tuple[GroupTransferRoster, ...],
) -> None:
    if type(regions) is not tuple or len(regions) == 0:
        raise ValueError("regions must be a non-empty tuple")
    owned_groups: set[int] = set()
    for expected_index, region in enumerate(regions):
        if type(region) is not RegionOwnership:
            raise ValueError("regions must contain RegionOwnership records")
        _require_int(f"regions[{expected_index}].region_index", region.region_index)
        if region.region_index != expected_index:
            raise ValueError("region indices must be canonical and contiguous")
        if type(region.group_indices) is not tuple or len(region.group_indices) == 0:
            raise ValueError(f"region {expected_index} must own at least one group")
        if tuple(sorted(set(region.group_indices))) != region.group_indices:
            raise ValueError(
                f"region {expected_index} group indices must be sorted and unique"
            )
        for group_index in region.group_indices:
            _require_int(
                f"regions[{expected_index}].group_indices entry",
                group_index,
            )
            if group_index >= len(groups):
                raise ValueError(
                    f"region {expected_index} owns out-of-range group {group_index}"
                )
            owned_groups.add(group_index)
        plane_counts = {
            groups[group_index].destination_plane_count
            for group_index in region.group_indices
        }
        if len(plane_counts) != 1:
            raise ValueError(f"region {expected_index} mixes destination plane layouts")
        _require_int(
            f"regions[{expected_index}].source_row_count",
            region.source_row_count,
            minimum=1,
        )
        _require_int(
            f"regions[{expected_index}].destination_row_count",
            region.destination_row_count,
            minimum=1,
        )
        _require_int(
            f"regions[{expected_index}].row_bytes",
            region.row_bytes,
            minimum=1,
        )
        if region.row_bytes % 2 != 0:
            raise ValueError(f"region {expected_index} row size must be even")
    if owned_groups != set(range(len(groups))):
        raise ValueError("region ownership must cover every group")


def _validate_ranks(
    source_tp_size: int,
    source_ranks: tuple[int, ...],
    rank_slots: tuple[int, ...],
) -> None:
    _require_int("source_tp_size", source_tp_size, minimum=1)
    if type(source_ranks) is not tuple or len(source_ranks) == 0:
        raise ValueError("source_ranks must be a non-empty tuple")
    if type(rank_slots) is not tuple or len(rank_slots) != len(source_ranks):
        raise ValueError("rank_slots must match source_ranks exactly")
    for index, source_rank in enumerate(source_ranks):
        _require_int(f"source_ranks[{index}]", source_rank)
        if source_rank >= source_tp_size:
            raise ValueError(f"source rank {source_rank} is outside the source TP")
    if len(set(source_ranks)) != len(source_ranks):
        raise ValueError("source_ranks contains duplicates")
    for index, rank_slot in enumerate(rank_slots):
        _require_int(f"rank_slots[{index}]", rank_slot)
    if tuple(sorted(rank_slots)) != tuple(range(len(source_ranks))):
        raise ValueError("rank_slots must be a complete bijection")


def _group_positions(group: GroupTransferRoster) -> tuple[TransferPosition, ...]:
    positions: list[TransferPosition] = []
    first_local_position = group.source_position_start // 2
    for roster_index, remote_block_id in enumerate(group.remote_block_ids):
        source_position = group.source_position_start + roster_index
        if group.destination_plane_count == 1:
            local_index = source_position // 2 - first_local_position
            destination_half = source_position % 2
        else:
            local_index = roster_index
            destination_half = DUAL_PLANE_DESTINATION_HALF
        positions.append(
            TransferPosition(
                group_index=group.group_index,
                source_position=source_position,
                remote_block_id=remote_block_id,
                local_block_id=group.local_block_ids[local_index],
                destination_half=destination_half,
            )
        )
    return tuple(positions)


def _validate_position_bounds(
    region: RegionOwnership,
    positions: tuple[TransferPosition, ...],
) -> None:
    for position in positions:
        if position.remote_block_id >= region.source_row_count:
            raise ValueError(
                f"region {region.region_index} remote block "
                f"{position.remote_block_id} is outside its source rows"
            )
        if position.local_block_id >= region.destination_row_count:
            raise ValueError(
                f"region {region.region_index} local block "
                f"{position.local_block_id} is outside its destination rows"
            )


def _validate_destination_writes(
    region: RegionOwnership,
    positions: tuple[TransferPosition, ...],
) -> None:
    occupied_halves: dict[int, int] = {}
    for position in positions:
        write_mask = (
            0b11
            if position.destination_half == DUAL_PLANE_DESTINATION_HALF
            else 1 << position.destination_half
        )
        prior_mask = occupied_halves.get(position.local_block_id, 0)
        if prior_mask & write_mask != 0:
            raise ValueError(
                f"region {region.region_index} maps multiple source positions "
                f"to destination block {position.local_block_id} half mask "
                f"{write_mask:#04b}"
            )
        occupied_halves[position.local_block_id] = prior_mask | write_mask


def _validate_source_reads(
    region: RegionOwnership,
    positions: tuple[TransferPosition, ...],
) -> None:
    source_owners: dict[int, int] = {}
    for position in positions:
        prior_group = source_owners.get(position.remote_block_id)
        if prior_group is not None:
            raise ValueError(
                f"region {region.region_index} reads remote block "
                f"{position.remote_block_id} more than once through groups "
                f"{prior_group} and {position.group_index}"
            )
        source_owners[position.remote_block_id] = position.group_index


def _form_runs(positions: tuple[TransferPosition, ...]) -> tuple[RemoteBlockRun, ...]:
    if len(positions) == 0:
        return ()
    runs: list[RemoteBlockRun] = []
    run_start = 0
    for position_index in range(1, len(positions) + 1):
        if (
            position_index < len(positions)
            and positions[position_index].remote_block_id
            == positions[position_index - 1].remote_block_id + 1
        ):
            continue
        runs.append(
            RemoteBlockRun(
                remote_block_id=positions[run_start].remote_block_id,
                position_count=position_index - run_start,
                position_start=run_start,
            )
        )
        run_start = position_index
    return tuple(runs)


def _digest_payload(
    source_tp_size: int,
    source_ranks: tuple[int, ...],
    rank_slots: tuple[int, ...],
    groups: tuple[GroupTransferRoster, ...],
    regions: tuple[RegionTransferLayout, ...],
    rank_stride_bytes: int,
    staging_size_bytes: int,
) -> tuple[object, ...]:
    return (
        "vllm.nixl.coalesced-transfer-layout.v1",
        source_tp_size,
        source_ranks,
        rank_slots,
        tuple(
            (
                group.group_index,
                group.source_position_start,
                group.destination_plane_count,
                group.local_block_ids,
                group.remote_block_ids,
            )
            for group in groups
        ),
        tuple(
            (
                region.ownership.region_index,
                region.ownership.group_indices,
                region.ownership.source_row_count,
                region.ownership.destination_row_count,
                region.ownership.row_bytes,
                region.offset_within_rank,
                tuple(
                    (
                        position.group_index,
                        position.source_position,
                        position.remote_block_id,
                        position.local_block_id,
                        position.destination_half,
                    )
                    for position in region.positions
                ),
                tuple(
                    (
                        run.remote_block_id,
                        run.position_count,
                        run.position_start,
                    )
                    for run in region.runs
                ),
            )
            for region in regions
        ),
        rank_stride_bytes,
        staging_size_bytes,
    )


def build_coalesced_transfer_plan(
    *,
    source_tp_size: int,
    source_ranks: tuple[int, ...],
    rank_slots: tuple[int, ...],
    groups: tuple[GroupTransferRoster, ...],
    regions: tuple[RegionOwnership, ...],
) -> CoalescedTransferPlan:
    """Build and validate a canonical coalesced transfer plan.

    Prefix trimming and request clipping are deliberately outside this function.
    ``groups`` must contain the exact selected rosters and their absolute starting
    positions in the producer rosters.

    :param source_tp_size: Producer tensor-parallel world size.
    :param source_ranks: Producer ranks participating in this decoder read.
    :param rank_slots: Destination attention slots paired with ``source_ranks``.
    :param groups: Canonical group-local selected block rosters.
    :param regions: Canonical authoritative region ownership and geometry.
    :returns: An immutable rank-major staging plan.
    """
    _validate_ranks(source_tp_size, source_ranks, rank_slots)
    _validate_groups(groups)
    _validate_regions(regions, groups)

    positions_by_group = tuple(_group_positions(group) for group in groups)
    region_layouts: list[RegionTransferLayout] = []
    region_offset = 0
    for region in regions:
        unsorted_positions = tuple(
            position
            for group_index in region.group_indices
            for position in positions_by_group[group_index]
        )
        positions = tuple(
            sorted(unsorted_positions, key=lambda position: position.remote_block_id)
        )
        _validate_position_bounds(region, positions)
        _validate_source_reads(region, positions)
        _validate_destination_writes(region, positions)
        layout = RegionTransferLayout(
            ownership=region,
            offset_within_rank=region_offset,
            positions=positions,
            runs=_form_runs(positions),
        )
        region_layouts.append(layout)
        region_offset += layout.size_bytes

    frozen_regions = tuple(region_layouts)
    staging_size_bytes = region_offset * len(source_ranks)
    payload = _digest_payload(
        source_tp_size,
        source_ranks,
        rank_slots,
        groups,
        frozen_regions,
        region_offset,
        staging_size_bytes,
    )
    encoded_payload = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
    return CoalescedTransferPlan(
        source_tp_size=source_tp_size,
        source_ranks=source_ranks,
        rank_slots=rank_slots,
        groups=groups,
        regions=frozen_regions,
        rank_stride_bytes=region_offset,
        staging_size_bytes=staging_size_bytes,
        digest=hashlib.sha256(encoded_payload).hexdigest(),
    )
