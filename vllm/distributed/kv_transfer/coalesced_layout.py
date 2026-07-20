# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Canonical transport layouts for coalesced KV-cache reads."""

import hashlib
import json
from dataclasses import dataclass

DUAL_PLANE_DESTINATION_HALF = -1
_UINT64_LIMIT = 1 << 64


def rank_major_slot_base(
    slot_base: int,
    rank: int,
    rank_stride_bytes: int,
) -> int:
    """Return one producer-rank slab within a shared receive slot.

    :param slot_base: Base address of the complete rank-major slot.
    :param rank: Zero-based producer rank.
    :param rank_stride_bytes: Reserved bytes between adjacent rank slabs.
    :returns: Base address of the selected producer-rank slab.
    :raises ValueError: If the inputs are malformed or the address overflows.
    """
    _require_int("slot_base", slot_base, minimum=1)
    _require_int("rank", rank)
    _require_int("rank_stride_bytes", rank_stride_bytes, minimum=1)
    rank_base = slot_base + rank * rank_stride_bytes
    rank_end = rank_base + rank_stride_bytes
    if rank_base >= _UINT64_LIMIT or rank_end > _UINT64_LIMIT:
        raise ValueError("rank-major slot address exceeds the uint64 address space")
    return rank_base


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
class SourceGroupTransferRoster:
    """Destination-independent source roster for one KV cache group.

    :ivar group_index: Canonical cache-group index.
    :ivar source_position_start: Absolute first selected source position.
    :ivar remote_block_ids: Selected source block roster.
    """

    group_index: int
    source_position_start: int
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
class SourceRegionOwnership:
    """Destination-independent ownership and row geometry for one region.

    :ivar region_index: Canonical registered-region index.
    :ivar group_indices: Cache groups whose rows exist in this region.
    :ivar source_row_count: Registered source rows.
    :ivar row_bytes: Bytes in one producer-rank source row.
    """

    region_index: int
    group_indices: tuple[int, ...]
    source_row_count: int
    row_bytes: int


@dataclass(frozen=True, slots=True)
class SourceTransferPosition:
    """One selected source row before destination placement is known.

    :ivar group_index: Owning cache-group index.
    :ivar source_position: Absolute position within the source group roster.
    :ivar remote_block_id: Registered producer row.
    """

    group_index: int
    source_position: int
    remote_block_id: int


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
class SourceRegionTransferLayout:
    """Canonical selected source rows for one region within every rank slab.

    :ivar ownership: Destination-independent source-region contract.
    :ivar offset_within_rank: Region byte offset within each source-rank slab.
    :ivar positions: Source-sorted selected positions.
    :ivar runs: Maximal contiguous source runs over ``positions``.
    """

    ownership: SourceRegionOwnership
    offset_within_rank: int
    positions: tuple[SourceTransferPosition, ...]
    runs: tuple[RemoteBlockRun, ...]

    @property
    def size_bytes(self) -> int:
        """Return this region's size within one source-rank slab."""
        return len(self.positions) * self.ownership.row_bytes


@dataclass(frozen=True, slots=True)
class CanonicalSourceTransferPlan:
    """Destination-independent rank-major source transfer layout.

    :ivar source_tp_size: Complete producer tensor-parallel size.
    :ivar source_ranks: Participating producer ranks in transfer order.
    :ivar groups: Exact selected source-group rosters.
    :ivar regions: Region-owned source layouts.
    :ivar rank_stride_bytes: Bytes contributed by each producer rank.
    :ivar transfer_size_bytes: Bytes contributed by all producer ranks.
    :ivar digest: SHA-256 identity of the source selection and geometry.
    """

    source_tp_size: int
    source_ranks: tuple[int, ...]
    groups: tuple[SourceGroupTransferRoster, ...]
    regions: tuple[SourceRegionTransferLayout, ...]
    rank_stride_bytes: int
    transfer_size_bytes: int
    digest: str

    def rank_offset(self, rank_index: int) -> int:
        """Return the start of a source rank's transfer slab.

        :param rank_index: Position of the source rank in ``source_ranks``.
        :returns: Byte offset from the start of the complete transfer.
        """
        if rank_index < 0 or rank_index >= len(self.source_ranks):
            raise IndexError("rank index is outside the source transfer plan")
        return rank_index * self.rank_stride_bytes

    def region_offset(self, rank_index: int, region_index: int) -> int:
        """Return a region's offset within a source rank's transfer slab.

        :param rank_index: Position of the source rank in ``source_ranks``.
        :param region_index: Canonical region index.
        :returns: Byte offset from the start of the complete transfer.
        """
        if region_index < 0 or region_index >= len(self.regions):
            raise IndexError("region index is outside the source transfer plan")
        return (
            self.rank_offset(rank_index) + self.regions[region_index].offset_within_rank
        )


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
    :ivar source_digest: SHA-256 identity of the canonical source layout.
    :ivar digest: SHA-256 identity of every semantic placement field.
    """

    source_tp_size: int
    source_ranks: tuple[int, ...]
    rank_slots: tuple[int, ...]
    groups: tuple[GroupTransferRoster, ...]
    regions: tuple[RegionTransferLayout, ...]
    rank_stride_bytes: int
    staging_size_bytes: int
    source_digest: str
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


@dataclass(frozen=True, slots=True)
class PackedRegionSlice:
    """One contiguous region-position slice within a packed rank chunk.

    :ivar region_index: Canonical source-region index.
    :ivar position_start: First canonical position in the source region.
    :ivar position_count: Number of complete source rows in the slice.
    :ivar offset_within_rank: Slice byte offset within each packed rank slab.
    :ivar size_bytes: Slice size within one packed rank slab.
    """

    region_index: int
    position_start: int
    position_count: int
    offset_within_rank: int
    size_bytes: int


@dataclass(frozen=True, slots=True)
class PackedTransferChunk:
    """One bounded rank-major packed transfer chunk.

    :ivar chunk_index: Canonical chunk sequence index.
    :ivar region_slices: Ordered complete-row slices in each rank slab.
    :ivar rank_stride_bytes: Bytes transferred for each producer rank.
    :ivar transfer_size_bytes: Bytes transferred for all producer ranks.
    :ivar digest: SHA-256 identity of this chunk and its source layout.
    """

    chunk_index: int
    region_slices: tuple[PackedRegionSlice, ...]
    rank_stride_bytes: int
    transfer_size_bytes: int
    digest: str


@dataclass(frozen=True, slots=True)
class PackedTransferPlan:
    """Bounded row-aligned chunks over a canonical source transfer plan.

    :ivar source_digest: SHA-256 identity of the canonical source layout.
    :ivar source_ranks: Participating producer ranks in transfer order.
    :ivar max_chunk_bytes_per_rank: Per-rank byte ceiling for every chunk.
    :ivar chunks: Canonical chunk sequence.
    :ivar rank_stride_bytes: Total bytes transferred for each producer rank.
    :ivar transfer_size_bytes: Total bytes transferred for all producer ranks.
    :ivar digest: SHA-256 identity of the source layout and chunk sequence.
    """

    source_digest: str
    source_ranks: tuple[int, ...]
    max_chunk_bytes_per_rank: int
    chunks: tuple[PackedTransferChunk, ...]
    rank_stride_bytes: int
    transfer_size_bytes: int
    digest: str


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


def _validate_source_groups(
    groups: tuple[SourceGroupTransferRoster, ...],
) -> None:
    if type(groups) is not tuple or len(groups) == 0:
        raise ValueError("groups must be a non-empty tuple")
    for expected_index, group in enumerate(groups):
        if type(group) is not SourceGroupTransferRoster:
            raise ValueError("groups must contain SourceGroupTransferRoster records")
        _require_int(f"groups[{expected_index}].group_index", group.group_index)
        if group.group_index != expected_index:
            raise ValueError("group indices must be canonical and contiguous")
        _require_int(
            f"groups[{expected_index}].source_position_start",
            group.source_position_start,
        )
        _require_block_ids(
            f"groups[{expected_index}].remote_block_ids",
            group.remote_block_ids,
        )


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


def _validate_source_regions(
    regions: tuple[SourceRegionOwnership, ...],
    groups: tuple[SourceGroupTransferRoster, ...],
) -> None:
    if type(regions) is not tuple or len(regions) == 0:
        raise ValueError("regions must be a non-empty tuple")
    owned_groups: set[int] = set()
    for expected_index, region in enumerate(regions):
        if type(region) is not SourceRegionOwnership:
            raise ValueError("regions must contain SourceRegionOwnership records")
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
        _require_int(
            f"regions[{expected_index}].source_row_count",
            region.source_row_count,
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


def _validate_source_ranks(
    source_tp_size: int,
    source_ranks: tuple[int, ...],
) -> None:
    _require_int("source_tp_size", source_tp_size, minimum=1)
    if type(source_ranks) is not tuple or len(source_ranks) == 0:
        raise ValueError("source_ranks must be a non-empty tuple")
    for index, source_rank in enumerate(source_ranks):
        _require_int(f"source_ranks[{index}]", source_rank)
        if source_rank >= source_tp_size:
            raise ValueError(f"source rank {source_rank} is outside the source TP")
    if len(set(source_ranks)) != len(source_ranks):
        raise ValueError("source_ranks contains duplicates")


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


def _source_groups_from_destinations(
    groups: tuple[GroupTransferRoster, ...],
) -> tuple[SourceGroupTransferRoster, ...]:
    return tuple(
        SourceGroupTransferRoster(
            group_index=group.group_index,
            source_position_start=group.source_position_start,
            remote_block_ids=group.remote_block_ids,
        )
        for group in groups
    )


def _source_regions_from_destinations(
    regions: tuple[RegionOwnership, ...],
) -> tuple[SourceRegionOwnership, ...]:
    return tuple(
        SourceRegionOwnership(
            region_index=region.region_index,
            group_indices=region.group_indices,
            source_row_count=region.source_row_count,
            row_bytes=region.row_bytes,
        )
        for region in regions
    )


def _source_group_positions(
    group: SourceGroupTransferRoster,
) -> tuple[SourceTransferPosition, ...]:
    return tuple(
        SourceTransferPosition(
            group_index=group.group_index,
            source_position=group.source_position_start + roster_index,
            remote_block_id=remote_block_id,
        )
        for roster_index, remote_block_id in enumerate(group.remote_block_ids)
    )


def _bind_source_positions(
    positions: tuple[SourceTransferPosition, ...],
    groups: tuple[GroupTransferRoster, ...],
) -> tuple[TransferPosition, ...]:
    bound_positions: list[TransferPosition] = []
    for position in positions:
        group = groups[position.group_index]
        roster_index = position.source_position - group.source_position_start
        if group.destination_plane_count == 1:
            first_local_position = group.source_position_start // 2
            local_index = position.source_position // 2 - first_local_position
            destination_half = position.source_position % 2
        else:
            local_index = roster_index
            destination_half = DUAL_PLANE_DESTINATION_HALF
        bound_positions.append(
            TransferPosition(
                group_index=position.group_index,
                source_position=position.source_position,
                remote_block_id=position.remote_block_id,
                local_block_id=group.local_block_ids[local_index],
                destination_half=destination_half,
            )
        )
    return tuple(bound_positions)


def _validate_source_position_bounds(
    region: SourceRegionOwnership,
    positions: tuple[SourceTransferPosition, ...],
) -> None:
    for position in positions:
        if position.remote_block_id >= region.source_row_count:
            raise ValueError(
                f"region {region.region_index} remote block "
                f"{position.remote_block_id} is outside its source rows"
            )


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


def _validate_remote_reads(
    region_index: int,
    owners_and_blocks: tuple[tuple[int, int], ...],
) -> None:
    source_owners: dict[int, int] = {}
    for group_index, remote_block_id in owners_and_blocks:
        prior_group = source_owners.get(remote_block_id)
        if prior_group is not None:
            raise ValueError(
                f"region {region_index} reads remote block "
                f"{remote_block_id} more than once through groups "
                f"{prior_group} and {group_index}"
            )
        source_owners[remote_block_id] = group_index


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
    _validate_remote_reads(
        region.region_index,
        tuple(
            (position.group_index, position.remote_block_id) for position in positions
        ),
    )


def _validate_canonical_source_reads(
    region: SourceRegionOwnership,
    positions: tuple[SourceTransferPosition, ...],
) -> None:
    _validate_remote_reads(
        region.region_index,
        tuple(
            (position.group_index, position.remote_block_id) for position in positions
        ),
    )


def _form_remote_block_runs(
    remote_block_ids: tuple[int, ...],
) -> tuple[RemoteBlockRun, ...]:
    if len(remote_block_ids) == 0:
        return ()
    runs: list[RemoteBlockRun] = []
    run_start = 0
    for position_index in range(1, len(remote_block_ids) + 1):
        if (
            position_index < len(remote_block_ids)
            and remote_block_ids[position_index]
            == remote_block_ids[position_index - 1] + 1
        ):
            continue
        runs.append(
            RemoteBlockRun(
                remote_block_id=remote_block_ids[run_start],
                position_count=position_index - run_start,
                position_start=run_start,
            )
        )
        run_start = position_index
    return tuple(runs)


def _form_source_runs(
    positions: tuple[SourceTransferPosition, ...],
) -> tuple[RemoteBlockRun, ...]:
    return _form_remote_block_runs(
        tuple(position.remote_block_id for position in positions)
    )


def _hash_payload(payload: tuple[object, ...]) -> str:
    encoded_payload = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(encoded_payload).hexdigest()


def _source_digest_payload(
    source_tp_size: int,
    source_ranks: tuple[int, ...],
    groups: tuple[SourceGroupTransferRoster, ...],
    regions: tuple[SourceRegionTransferLayout, ...],
    rank_stride_bytes: int,
    transfer_size_bytes: int,
) -> tuple[object, ...]:
    return (
        "vllm.nixl.canonical-source-transfer-layout.v1",
        source_tp_size,
        source_ranks,
        tuple(
            (
                group.group_index,
                group.source_position_start,
                group.remote_block_ids,
            )
            for group in groups
        ),
        tuple(
            (
                region.ownership.region_index,
                region.ownership.group_indices,
                region.ownership.source_row_count,
                region.ownership.row_bytes,
                region.offset_within_rank,
                tuple(
                    (
                        position.group_index,
                        position.source_position,
                        position.remote_block_id,
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
        transfer_size_bytes,
    )


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


def build_canonical_source_plan(
    *,
    source_tp_size: int,
    source_ranks: tuple[int, ...],
    groups: tuple[SourceGroupTransferRoster, ...],
    regions: tuple[SourceRegionOwnership, ...],
) -> CanonicalSourceTransferPlan:
    """Build a destination-independent canonical source transfer plan.

    :param source_tp_size: Producer tensor-parallel world size.
    :param source_ranks: Producer ranks participating in the transfer.
    :param groups: Canonical selected producer block rosters.
    :param regions: Canonical producer region ownership and geometry.
    :returns: An immutable rank-major source plan.
    """
    _validate_source_ranks(source_tp_size, source_ranks)
    _validate_source_groups(groups)
    _validate_source_regions(regions, groups)

    positions_by_group = tuple(_source_group_positions(group) for group in groups)
    region_layouts: list[SourceRegionTransferLayout] = []
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
        _validate_source_position_bounds(region, positions)
        _validate_canonical_source_reads(region, positions)
        layout = SourceRegionTransferLayout(
            ownership=region,
            offset_within_rank=region_offset,
            positions=positions,
            runs=_form_source_runs(positions),
        )
        region_layouts.append(layout)
        region_offset += layout.size_bytes

    frozen_regions = tuple(region_layouts)
    transfer_size_bytes = region_offset * len(source_ranks)
    digest = _hash_payload(
        _source_digest_payload(
            source_tp_size,
            source_ranks,
            groups,
            frozen_regions,
            region_offset,
            transfer_size_bytes,
        )
    )
    return CanonicalSourceTransferPlan(
        source_tp_size=source_tp_size,
        source_ranks=source_ranks,
        groups=groups,
        regions=frozen_regions,
        rank_stride_bytes=region_offset,
        transfer_size_bytes=transfer_size_bytes,
        digest=digest,
    )


def bind_coalesced_transfer_destinations(
    source_plan: CanonicalSourceTransferPlan,
    *,
    rank_slots: tuple[int, ...],
    groups: tuple[GroupTransferRoster, ...],
    regions: tuple[RegionOwnership, ...],
) -> CoalescedTransferPlan:
    """Bind decoder-local destinations to a canonical source plan.

    :param source_plan: Destination-independent producer selection and geometry.
    :param rank_slots: Destination attention slots paired with ``source_ranks``.
    :param groups: Matching source rosters with decoder-local placements.
    :param regions: Matching source geometry with decoder-local row bounds.
    :returns: An immutable rank-major staging plan.
    """
    if type(source_plan) is not CanonicalSourceTransferPlan:
        raise ValueError("source_plan must be a CanonicalSourceTransferPlan")
    _validate_ranks(
        source_plan.source_tp_size,
        source_plan.source_ranks,
        rank_slots,
    )
    _validate_groups(groups)
    _validate_regions(regions, groups)

    source_ownership = tuple(region.ownership for region in source_plan.regions)
    if (
        _source_groups_from_destinations(groups) != source_plan.groups
        or _source_regions_from_destinations(regions) != source_ownership
    ):
        raise ValueError("destination binding does not match the canonical source plan")

    region_layouts: list[RegionTransferLayout] = []
    for region in regions:
        source_region = source_plan.regions[region.region_index]
        positions = _bind_source_positions(
            source_region.positions,
            groups,
        )
        _validate_position_bounds(region, positions)
        _validate_source_reads(region, positions)
        _validate_destination_writes(region, positions)
        region_layouts.append(
            RegionTransferLayout(
                ownership=region,
                offset_within_rank=source_region.offset_within_rank,
                positions=positions,
                runs=source_region.runs,
            )
        )

    frozen_regions = tuple(region_layouts)
    payload = _digest_payload(
        source_plan.source_tp_size,
        source_plan.source_ranks,
        rank_slots,
        groups,
        frozen_regions,
        source_plan.rank_stride_bytes,
        source_plan.transfer_size_bytes,
    )
    return CoalescedTransferPlan(
        source_tp_size=source_plan.source_tp_size,
        source_ranks=source_plan.source_ranks,
        rank_slots=rank_slots,
        groups=groups,
        regions=frozen_regions,
        rank_stride_bytes=source_plan.rank_stride_bytes,
        staging_size_bytes=source_plan.transfer_size_bytes,
        source_digest=source_plan.digest,
        digest=_hash_payload(payload),
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
    source_plan = build_canonical_source_plan(
        source_tp_size=source_tp_size,
        source_ranks=source_ranks,
        groups=_source_groups_from_destinations(groups),
        regions=_source_regions_from_destinations(regions),
    )
    return bind_coalesced_transfer_destinations(
        source_plan,
        rank_slots=rank_slots,
        groups=groups,
        regions=regions,
    )


def _packed_chunk_digest_payload(
    source_digest: str,
    source_ranks: tuple[int, ...],
    chunk: PackedTransferChunk,
) -> tuple[object, ...]:
    return (
        "vllm.nixl.packed-transfer-chunk.v1",
        source_digest,
        source_ranks,
        chunk.chunk_index,
        tuple(
            (
                region_slice.region_index,
                region_slice.position_start,
                region_slice.position_count,
                region_slice.offset_within_rank,
                region_slice.size_bytes,
            )
            for region_slice in chunk.region_slices
        ),
        chunk.rank_stride_bytes,
        chunk.transfer_size_bytes,
    )


def _packed_plan_digest_payload(
    packed_plan: PackedTransferPlan,
) -> tuple[object, ...]:
    return (
        "vllm.nixl.packed-transfer-layout.v1",
        packed_plan.source_digest,
        packed_plan.source_ranks,
        packed_plan.max_chunk_bytes_per_rank,
        tuple(
            (
                chunk.chunk_index,
                tuple(
                    (
                        region_slice.region_index,
                        region_slice.position_start,
                        region_slice.position_count,
                        region_slice.offset_within_rank,
                        region_slice.size_bytes,
                    )
                    for region_slice in chunk.region_slices
                ),
                chunk.rank_stride_bytes,
                chunk.transfer_size_bytes,
                chunk.digest,
            )
            for chunk in packed_plan.chunks
        ),
        packed_plan.rank_stride_bytes,
        packed_plan.transfer_size_bytes,
    )


def _build_packed_chunk(
    *,
    source_plan: CanonicalSourceTransferPlan,
    chunk_index: int,
    region_slices: tuple[PackedRegionSlice, ...],
    rank_stride_bytes: int,
) -> PackedTransferChunk:
    transfer_size_bytes = rank_stride_bytes * len(source_plan.source_ranks)
    chunk = PackedTransferChunk(
        chunk_index=chunk_index,
        region_slices=region_slices,
        rank_stride_bytes=rank_stride_bytes,
        transfer_size_bytes=transfer_size_bytes,
        digest="",
    )
    return PackedTransferChunk(
        chunk_index=chunk.chunk_index,
        region_slices=chunk.region_slices,
        rank_stride_bytes=chunk.rank_stride_bytes,
        transfer_size_bytes=chunk.transfer_size_bytes,
        digest=_hash_payload(
            _packed_chunk_digest_payload(
                source_plan.digest,
                source_plan.source_ranks,
                chunk,
            )
        ),
    )


def build_packed_transfer_plan(
    source_plan: CanonicalSourceTransferPlan,
    *,
    max_chunk_bytes_per_rank: int,
) -> PackedTransferPlan:
    """Partition a canonical source plan into bounded complete-row chunks.

    Chunk bounds apply independently to each participating source rank. A row is
    never divided between chunks, and every source position appears exactly once.

    :param source_plan: Canonical destination-independent source plan.
    :param max_chunk_bytes_per_rank: Maximum bytes in one rank's chunk slab.
    :returns: An immutable ordered packed-chunk plan.
    :raises ValueError: If a selected row cannot fit within the chunk bound.
    """
    if type(source_plan) is not CanonicalSourceTransferPlan:
        raise ValueError("source_plan must be a CanonicalSourceTransferPlan")
    _require_int(
        "max_chunk_bytes_per_rank",
        max_chunk_bytes_per_rank,
        minimum=1,
    )

    chunks: list[PackedTransferChunk] = []
    pending_slices: list[PackedRegionSlice] = []
    pending_size_bytes = 0
    for region in source_plan.regions:
        position_start = 0
        position_total = len(region.positions)
        if position_total == 0:
            continue
        row_bytes = region.ownership.row_bytes
        if row_bytes > max_chunk_bytes_per_rank:
            raise ValueError(
                f"region {region.ownership.region_index} row size {row_bytes} "
                f"exceeds per-rank chunk bound {max_chunk_bytes_per_rank}"
            )

        while position_start < position_total:
            available_bytes = max_chunk_bytes_per_rank - pending_size_bytes
            available_rows = available_bytes // row_bytes
            if available_rows == 0:
                chunks.append(
                    _build_packed_chunk(
                        source_plan=source_plan,
                        chunk_index=len(chunks),
                        region_slices=tuple(pending_slices),
                        rank_stride_bytes=pending_size_bytes,
                    )
                )
                pending_slices = []
                pending_size_bytes = 0
                continue

            position_count = min(
                available_rows,
                position_total - position_start,
            )
            size_bytes = position_count * row_bytes
            pending_slices.append(
                PackedRegionSlice(
                    region_index=region.ownership.region_index,
                    position_start=position_start,
                    position_count=position_count,
                    offset_within_rank=pending_size_bytes,
                    size_bytes=size_bytes,
                )
            )
            pending_size_bytes += size_bytes
            position_start += position_count
            if pending_size_bytes == max_chunk_bytes_per_rank:
                chunks.append(
                    _build_packed_chunk(
                        source_plan=source_plan,
                        chunk_index=len(chunks),
                        region_slices=tuple(pending_slices),
                        rank_stride_bytes=pending_size_bytes,
                    )
                )
                pending_slices = []
                pending_size_bytes = 0

    if len(pending_slices) > 0:
        chunks.append(
            _build_packed_chunk(
                source_plan=source_plan,
                chunk_index=len(chunks),
                region_slices=tuple(pending_slices),
                rank_stride_bytes=pending_size_bytes,
            )
        )

    frozen_chunks = tuple(chunks)
    rank_stride_bytes = sum(chunk.rank_stride_bytes for chunk in frozen_chunks)
    transfer_size_bytes = sum(chunk.transfer_size_bytes for chunk in frozen_chunks)
    if rank_stride_bytes != source_plan.rank_stride_bytes:
        raise RuntimeError("packed chunks do not conserve source bytes per rank")
    if transfer_size_bytes != source_plan.transfer_size_bytes:
        raise RuntimeError("packed chunks do not conserve complete transfer bytes")
    packed_plan = PackedTransferPlan(
        source_digest=source_plan.digest,
        source_ranks=source_plan.source_ranks,
        max_chunk_bytes_per_rank=max_chunk_bytes_per_rank,
        chunks=frozen_chunks,
        rank_stride_bytes=rank_stride_bytes,
        transfer_size_bytes=transfer_size_bytes,
        digest="",
    )
    result = PackedTransferPlan(
        source_digest=packed_plan.source_digest,
        source_ranks=packed_plan.source_ranks,
        max_chunk_bytes_per_rank=packed_plan.max_chunk_bytes_per_rank,
        chunks=packed_plan.chunks,
        rank_stride_bytes=packed_plan.rank_stride_bytes,
        transfer_size_bytes=packed_plan.transfer_size_bytes,
        digest=_hash_payload(_packed_plan_digest_payload(packed_plan)),
    )
    validate_packed_transfer_plan(
        result,
        source_digest=source_plan.digest,
        source_ranks=source_plan.source_ranks,
        region_position_counts=tuple(
            len(region.positions) for region in source_plan.regions
        ),
        region_row_bytes=tuple(
            region.ownership.row_bytes for region in source_plan.regions
        ),
        rank_stride_bytes=source_plan.rank_stride_bytes,
        transfer_size_bytes=source_plan.transfer_size_bytes,
    )
    return result


def validate_packed_transfer_plan(
    packed_plan: PackedTransferPlan,
    *,
    source_digest: str,
    source_ranks: tuple[int, ...],
    region_position_counts: tuple[int, ...],
    region_row_bytes: tuple[int, ...],
    rank_stride_bytes: int,
    transfer_size_bytes: int,
) -> None:
    """Validate a packed plan against trusted canonical source geometry.

    :param packed_plan: Candidate packed transfer plan.
    :param source_digest: Trusted canonical source-layout digest.
    :param source_ranks: Trusted producer-rank transfer order.
    :param region_position_counts: Trusted selected-row count per region.
    :param region_row_bytes: Trusted producer row size per region.
    :param rank_stride_bytes: Trusted total selected bytes per producer rank.
    :param transfer_size_bytes: Trusted total selected bytes across all ranks.
    :raises ValueError: If any binding, structure, coverage, or digest differs.
    """
    if type(packed_plan) is not PackedTransferPlan:
        raise ValueError("packed_plan must be a PackedTransferPlan")
    if packed_plan.source_digest != source_digest:
        raise ValueError("packed plan differs from the canonical source digest")
    if packed_plan.source_ranks != source_ranks:
        raise ValueError("packed source ranks differ from the canonical source ranks")
    if type(region_position_counts) is not tuple or type(region_row_bytes) is not tuple:
        raise ValueError("packed source region geometry must use tuples")
    if len(region_position_counts) != len(region_row_bytes):
        raise ValueError("packed source region geometry lengths differ")
    for region_index, (position_count, row_bytes) in enumerate(
        zip(region_position_counts, region_row_bytes, strict=True)
    ):
        _require_int(
            f"region_position_counts[{region_index}]",
            position_count,
        )
        _require_int(
            f"region_row_bytes[{region_index}]",
            row_bytes,
            minimum=1,
        )
    _require_int("rank_stride_bytes", rank_stride_bytes)
    _require_int("transfer_size_bytes", transfer_size_bytes)
    if transfer_size_bytes != rank_stride_bytes * len(source_ranks):
        raise ValueError("canonical transfer size differs from its source ranks")
    _require_int(
        "packed_plan.max_chunk_bytes_per_rank",
        packed_plan.max_chunk_bytes_per_rank,
        minimum=1,
    )
    _require_int(
        "packed_plan.rank_stride_bytes",
        packed_plan.rank_stride_bytes,
    )
    _require_int(
        "packed_plan.transfer_size_bytes",
        packed_plan.transfer_size_bytes,
    )
    if type(packed_plan.chunks) is not tuple:
        raise ValueError("packed plan chunks must be a tuple")
    if packed_plan.rank_stride_bytes != rank_stride_bytes:
        raise ValueError("packed plan does not conserve source bytes per rank")
    if packed_plan.transfer_size_bytes != transfer_size_bytes:
        raise ValueError("packed plan does not conserve complete transfer bytes")

    expected_region_index = 0
    expected_position_start = 0
    while (
        expected_region_index < len(region_position_counts)
        and region_position_counts[expected_region_index] == 0
    ):
        expected_region_index += 1

    observed_rank_bytes = 0
    observed_transfer_bytes = 0
    for expected_chunk_index, chunk in enumerate(packed_plan.chunks):
        if type(chunk) is not PackedTransferChunk:
            raise ValueError("packed plan chunks must be PackedTransferChunk records")
        if chunk.chunk_index != expected_chunk_index:
            raise ValueError("packed chunk indices must be canonical and contiguous")
        if type(chunk.region_slices) is not tuple or len(chunk.region_slices) == 0:
            raise ValueError("packed chunks must contain at least one region slice")
        _require_int(
            f"packed chunk {expected_chunk_index} rank_stride_bytes",
            chunk.rank_stride_bytes,
            minimum=1,
        )
        if chunk.rank_stride_bytes > packed_plan.max_chunk_bytes_per_rank:
            raise ValueError("packed chunk exceeds its per-rank byte bound")
        _require_int(
            f"packed chunk {expected_chunk_index} transfer_size_bytes",
            chunk.transfer_size_bytes,
            minimum=1,
        )
        expected_chunk_transfer_bytes = chunk.rank_stride_bytes * len(source_ranks)
        if chunk.transfer_size_bytes != expected_chunk_transfer_bytes:
            raise ValueError("packed chunk transfer size differs from its rank slabs")

        expected_slice_offset = 0
        previous_region_index: int | None = None
        for region_slice in chunk.region_slices:
            if type(region_slice) is not PackedRegionSlice:
                raise ValueError("packed chunks must contain PackedRegionSlice records")
            _require_int("packed slice region_index", region_slice.region_index)
            _require_int("packed slice position_start", region_slice.position_start)
            _require_int(
                "packed slice position_count",
                region_slice.position_count,
                minimum=1,
            )
            _require_int(
                "packed slice offset_within_rank",
                region_slice.offset_within_rank,
            )
            _require_int(
                "packed slice size_bytes",
                region_slice.size_bytes,
                minimum=1,
            )
            if region_slice.region_index >= len(region_position_counts):
                raise ValueError("packed slice region index is outside the source plan")
            if region_slice.region_index != expected_region_index:
                raise ValueError("packed slices leave a source-region gap or overlap")
            if region_slice.position_start != expected_position_start:
                raise ValueError("packed slices leave a source-position gap or overlap")
            if previous_region_index == region_slice.region_index:
                raise ValueError(
                    "adjacent packed slices from one region must be merged"
                )
            if region_slice.offset_within_rank != expected_slice_offset:
                raise ValueError("packed slices are not byte-contiguous")

            position_end = region_slice.position_start + region_slice.position_count
            region_position_count = region_position_counts[region_slice.region_index]
            if position_end > region_position_count:
                raise ValueError("packed slice exceeds its canonical source region")
            expected_size_bytes = (
                region_slice.position_count
                * region_row_bytes[region_slice.region_index]
            )
            if region_slice.size_bytes != expected_size_bytes:
                raise ValueError("packed slice size differs from complete source rows")

            expected_slice_offset += region_slice.size_bytes
            expected_position_start = position_end
            previous_region_index = region_slice.region_index
            if expected_position_start != region_position_count:
                continue
            expected_region_index += 1
            expected_position_start = 0
            while (
                expected_region_index < len(region_position_counts)
                and region_position_counts[expected_region_index] == 0
            ):
                expected_region_index += 1

        if expected_slice_offset != chunk.rank_stride_bytes:
            raise ValueError("packed chunk rank stride differs from its slices")
        expected_chunk_digest = _hash_payload(
            _packed_chunk_digest_payload(
                packed_plan.source_digest,
                packed_plan.source_ranks,
                chunk,
            )
        )
        if chunk.digest != expected_chunk_digest:
            raise ValueError("packed chunk digest differs from its structure")
        observed_rank_bytes += chunk.rank_stride_bytes
        observed_transfer_bytes += chunk.transfer_size_bytes

        if expected_chunk_index + 1 >= len(packed_plan.chunks):
            continue
        if expected_region_index >= len(region_position_counts):
            raise ValueError("packed chunks extend beyond the canonical source plan")
        next_row_bytes = region_row_bytes[expected_region_index]
        if (
            chunk.rank_stride_bytes + next_row_bytes
            <= packed_plan.max_chunk_bytes_per_rank
        ):
            raise ValueError("packed chunk ends before its per-rank capacity boundary")

    if expected_region_index != len(region_position_counts):
        raise ValueError("packed chunks do not cover every canonical source position")
    if observed_rank_bytes != packed_plan.rank_stride_bytes:
        raise ValueError("packed chunk strides do not match the packed plan")
    if observed_transfer_bytes != packed_plan.transfer_size_bytes:
        raise ValueError("packed chunk sizes do not match the packed plan")
    expected_plan_digest = _hash_payload(_packed_plan_digest_payload(packed_plan))
    if packed_plan.digest != expected_plan_digest:
        raise ValueError("packed plan digest differs from its structure")
