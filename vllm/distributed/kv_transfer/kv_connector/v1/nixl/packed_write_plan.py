# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Producer-side reconstruction of bounded packed-write plans."""

from dataclasses import dataclass

from vllm.distributed.kv_transfer.coalesced_layout import (
    CanonicalSourceTransferPlan,
    PackedTransferChunk,
    PackedTransferPlan,
    SourceGroupTransferRoster,
    SourceRegionOwnership,
    build_canonical_source_plan,
    build_packed_transfer_plan,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    PackedWriteChunkIdentity,
    PackedWriteRequest,
    PackedWriteSourceSelection,
)
from vllm.distributed.kv_transfer.nixl_contracts import (
    NixlRegionDescriptor,
    NixlSourceRoster,
)


@dataclass(frozen=True, slots=True)
class ProducerPackedWritePlan:
    """A producer-authenticated source, packed, and selected chunk plan.

    :ivar source_plan: Canonical source selection reconstructed from retained
        producer state.
    :ivar packed_plan: Canonical bounded partition of the source selection.
    :ivar chunk: Exact chunk named by the decoder request.
    """

    source_plan: CanonicalSourceTransferPlan
    packed_plan: PackedTransferPlan
    chunk: PackedTransferChunk


def _require_int(name: str, value: int, *, minimum: int = 0) -> None:
    """Require a plain integer no smaller than ``minimum``.

    :param name: Field name used in validation errors.
    :param value: Candidate integer.
    :param minimum: Inclusive lower bound.
    :raises ValueError: If the candidate is not a valid integer.
    """
    if type(value) is not int or value < minimum:
        raise ValueError(
            f"{name} must be an integer greater than or equal to {minimum}"
        )


def _require_text(name: str, value: str) -> None:
    """Require a non-empty plain string.

    :param name: Field name used in validation errors.
    :param value: Candidate string.
    :raises ValueError: If the candidate is not a non-empty plain string.
    """
    if type(value) is not str or len(value) == 0:
        raise ValueError(f"{name} must be a non-empty string")


def _validate_roster(roster: NixlSourceRoster) -> None:
    """Validate the retained producer allocation contract.

    :param roster: Producer-owned physical block roster.
    :raises ValueError: If the roster is incomplete or malformed.
    """
    if type(roster) is not NixlSourceRoster:
        raise ValueError("roster must be a NixlSourceRoster")
    _require_int("roster.offer_generation", roster.offer_generation, minimum=1)
    _require_int("roster.iteration", roster.iteration)
    _require_int("roster.expected_consumers", roster.expected_consumers, minimum=1)
    _require_int("roster.valid_token_extent", roster.valid_token_extent, minimum=1)
    if (
        type(roster.group_token_capacities) is not tuple
        or len(roster.group_token_capacities) == 0
    ):
        raise ValueError("roster group token capacities must be a non-empty tuple")
    if type(roster.block_ids) is not tuple:
        raise ValueError("roster block IDs must be a tuple")
    if len(roster.block_ids) != len(roster.group_token_capacities):
        raise ValueError("roster group capacity and block cardinalities differ")
    for group_index, (capacity, block_ids) in enumerate(
        zip(
            roster.group_token_capacities,
            roster.block_ids,
            strict=True,
        )
    ):
        _require_int(
            f"roster.group_token_capacities[{group_index}]",
            capacity,
            minimum=1,
        )
        if type(block_ids) is not tuple:
            raise ValueError(f"roster group {group_index} block IDs must be a tuple")
        for position, block_id in enumerate(block_ids):
            _require_int(
                f"roster.block_ids[{group_index}][{position}]",
                block_id,
            )
        if len(set(block_ids)) != len(block_ids):
            raise ValueError(f"roster group {group_index} contains duplicate blocks")
    if all(len(block_ids) == 0 for block_ids in roster.block_ids):
        raise ValueError("roster does not retain any source blocks")


def _validate_region_descriptors(
    regions: tuple[NixlRegionDescriptor, ...],
    *,
    group_count: int,
) -> None:
    """Validate local registered-region geometry used by the pack kernel.

    :param regions: Producer-local registered KV region descriptors.
    :param group_count: Number of canonical source groups.
    :raises ValueError: If the region contract is incomplete or unsafe.
    """
    if type(regions) is not tuple or len(regions) == 0:
        raise ValueError("source regions must be a non-empty tuple")
    _require_int("group_count", group_count, minimum=1)

    covered_groups: set[int] = set()
    semantic_names: set[str] = set()
    address_ranges: list[tuple[int, int, int]] = []
    source_row_count: int | None = None
    maximum_address = (1 << 64) - 1
    for region_index, region in enumerate(regions):
        if type(region) is not NixlRegionDescriptor:
            raise ValueError("source regions must contain NixlRegionDescriptor records")
        _require_text(f"region {region_index} semantic_name", region.semantic_name)
        if region.semantic_name in semantic_names:
            raise ValueError(
                f"region {region_index} duplicates semantic name "
                f"{region.semantic_name!r}"
            )
        semantic_names.add(region.semantic_name)

        if type(region.group_indices) is not tuple or len(region.group_indices) == 0:
            raise ValueError(f"region {region_index} has no cache-group owner")
        if tuple(sorted(set(region.group_indices))) != region.group_indices:
            raise ValueError(
                f"region {region_index} group owners must be sorted and unique"
            )
        for owner in region.group_indices:
            _require_int(f"region {region_index} group owner", owner)
            if owner >= group_count:
                raise ValueError(
                    f"region {region_index} owns out-of-range group {owner}"
                )
        if type(region.group_semantic_names) is not tuple:
            raise ValueError(
                f"region {region_index} group semantic names must be a tuple"
            )
        semantic_owners = tuple(owner for owner, _ in region.group_semantic_names)
        if semantic_owners != region.group_indices:
            raise ValueError(
                f"region {region_index} semantic owners differ from physical owners"
            )
        for owner, semantic_name in region.group_semantic_names:
            _require_int(f"region {region_index} semantic owner", owner)
            _require_text(
                f"region {region_index} group {owner} semantic name",
                semantic_name,
            )
        covered_groups.update(region.group_indices)

        _require_int(
            f"region {region_index} base_address", region.base_address, minimum=1
        )
        _require_int(
            f"region {region_index} registered_bytes",
            region.registered_bytes,
            minimum=1,
        )
        _require_int(f"region {region_index} row_bytes", region.row_bytes, minimum=1)
        if region.row_bytes % 2 != 0:
            raise ValueError(f"region {region_index} row size must be even")
        if region.registered_bytes % region.row_bytes != 0:
            raise ValueError(f"region {region_index} registration is not row-aligned")
        registered_row_count = region.registered_bytes // region.row_bytes
        if source_row_count is None:
            source_row_count = registered_row_count
        elif source_row_count != registered_row_count:
            raise ValueError("source regions have different physical row counts")

        if type(region.shape) is not tuple or len(region.shape) == 0:
            raise ValueError(f"region {region_index} has an invalid shape")
        if type(region.strides) is not tuple or len(region.strides) != len(
            region.shape
        ):
            raise ValueError(f"region {region_index} has an invalid stride rank")
        for dimension, extent in enumerate(region.shape):
            _require_int(
                f"region {region_index} shape[{dimension}]",
                extent,
                minimum=1,
            )
        for dimension, stride in enumerate(region.strides):
            _require_int(
                f"region {region_index} strides[{dimension}]",
                stride,
                minimum=1,
            )
        if region.shape[0] != registered_row_count:
            raise ValueError(
                f"region {region_index} leading dimension differs from its "
                "registered rows"
            )
        _require_text(f"region {region_index} dtype", region.dtype)
        _require_text(f"region {region_index} layout", region.layout)
        _require_int(
            f"region {region_index} element_size_bytes",
            region.element_size_bytes,
            minimum=1,
        )
        storage_elements = 1 + sum(
            (extent - 1) * stride
            for extent, stride in zip(region.shape, region.strides, strict=True)
        )
        if storage_elements * region.element_size_bytes > region.registered_bytes:
            raise ValueError(
                f"region {region_index} tensor view exceeds its registration"
            )

        registration_end = region.base_address + region.registered_bytes
        if (
            registration_end <= region.base_address
            or registration_end > maximum_address
        ):
            raise ValueError(
                f"region {region_index} registration exceeds the uint64 address space"
            )
        address_ranges.append((region.base_address, registration_end, region_index))

    if covered_groups != set(range(group_count)):
        raise ValueError("source region ownership does not cover every group")
    address_ranges.sort()
    for previous, current in zip(address_ranges, address_ranges[1:]):
        if previous[1] > current[0]:
            raise ValueError(f"source regions {previous[2]} and {current[2]} overlap")


def _validate_producer_identity(
    identity: PackedWriteChunkIdentity,
    *,
    producer_engine_id: str,
    producer_rank: int,
    producer_tp_size: int,
) -> None:
    """Bind a request to the receiving producer process.

    :param identity: Request-carried packed chunk identity.
    :param producer_engine_id: Receiving producer engine identity.
    :param producer_rank: Receiving producer tensor-parallel rank.
    :param producer_tp_size: Receiving producer tensor-parallel world size.
    :raises ValueError: If the request addresses another producer.
    """
    _require_text("producer_engine_id", producer_engine_id)
    _require_int("producer_tp_size", producer_tp_size, minimum=1)
    _require_int("producer_rank", producer_rank)
    if producer_rank >= producer_tp_size:
        raise ValueError("producer_rank is outside producer_tp_size")
    observed = (
        identity.producer_engine_id,
        identity.producer_rank,
        identity.producer_tp_size,
    )
    expected = (producer_engine_id, producer_rank, producer_tp_size)
    if observed != expected:
        raise ValueError(
            "packed write request addresses a different producer identity: "
            f"observed={observed}, expected={expected}"
        )


def _reconstruct_source_groups(
    roster: NixlSourceRoster,
    selections: tuple[PackedWriteSourceSelection, ...],
) -> tuple[SourceGroupTransferRoster, ...]:
    """Reconstruct exact canonical suffix selections from retained blocks.

    :param roster: Producer-owned physical block roster.
    :param selections: Decoder-advertised group-absolute intervals.
    :returns: Canonical selected source groups.
    :raises ValueError: If any interval is not the exact retained suffix.
    """
    if type(selections) is not tuple:
        raise ValueError("source selections must be a tuple")
    if len(selections) != len(roster.block_ids):
        raise ValueError("source selection count differs from the retained roster")

    groups: list[SourceGroupTransferRoster] = []
    positive_group_count = 0
    for expected_group_index, (selection, retained_blocks) in enumerate(
        zip(selections, roster.block_ids, strict=True)
    ):
        if type(selection) is not PackedWriteSourceSelection:
            raise ValueError("source selections must be typed")
        if selection.group_index != expected_group_index:
            raise ValueError("source selections are not in canonical group order")
        expected_end = selection.source_position_start + selection.position_count
        if expected_end != len(retained_blocks):
            raise ValueError(
                f"group {expected_group_index} source selection is not the exact "
                "retained suffix"
            )
        selected_blocks = retained_blocks[selection.source_position_start :]
        if len(selected_blocks) != selection.position_count:
            raise ValueError(
                f"group {expected_group_index} source selection is out of range"
            )
        if selection.position_count > 0:
            positive_group_count += 1
        groups.append(
            SourceGroupTransferRoster(
                group_index=expected_group_index,
                source_position_start=selection.source_position_start,
                remote_block_ids=selected_blocks,
            )
        )
    if positive_group_count == 0:
        raise ValueError("source selections contain no retained positions")
    return tuple(groups)


def _source_region_ownership(
    regions: tuple[NixlRegionDescriptor, ...],
) -> tuple[SourceRegionOwnership, ...]:
    """Project trusted registered descriptors into canonical source geometry.

    :param regions: Validated producer-local registered regions.
    :returns: Address-independent canonical source-region ownership.
    """
    return tuple(
        SourceRegionOwnership(
            region_index=region_index,
            group_indices=region.group_indices,
            source_row_count=region.shape[0],
            row_bytes=region.row_bytes,
        )
        for region_index, region in enumerate(regions)
    )


def _validate_request_identity(
    request: PackedWriteRequest,
    *,
    roster: NixlSourceRoster,
    source_plan: CanonicalSourceTransferPlan,
    packed_plan: PackedTransferPlan,
) -> PackedTransferChunk:
    """Validate request digests and chunk geometry against reconstruction.

    :param request: Decoder request for one producer-packed chunk.
    :param roster: Producer-owned physical block roster.
    :param source_plan: Reconstructed canonical source selection.
    :param packed_plan: Reconstructed bounded packed plan.
    :returns: The exact reconstructed chunk named by the request.
    :raises ValueError: If any decoder assertion differs from producer state.
    """
    identity = request.command.chunk
    if identity.offer_generation != roster.offer_generation:
        raise ValueError("packed write request names a stale source offer")
    if identity.valid_token_extent != roster.valid_token_extent:
        raise ValueError("packed write request token extent differs from its source")
    if identity.source_plan_digest != source_plan.digest:
        raise ValueError("packed write request source-plan digest differs")
    if identity.packed_plan_digest != packed_plan.digest:
        raise ValueError("packed write request packed-plan digest differs")
    if identity.chunk_count != len(packed_plan.chunks):
        raise ValueError("packed write request chunk count differs")
    if identity.chunk_ordinal >= len(packed_plan.chunks):
        raise ValueError("packed write request chunk ordinal is out of range")
    chunk = packed_plan.chunks[identity.chunk_ordinal]
    if identity.chunk_plan_digest != chunk.digest:
        raise ValueError("packed write request chunk-plan digest differs")
    if identity.exact_bytes != chunk.rank_stride_bytes:
        raise ValueError("packed write request byte count differs from its rank chunk")
    return chunk


def reconstruct_producer_packed_write_plan(
    *,
    roster: NixlSourceRoster,
    regions: tuple[NixlRegionDescriptor, ...],
    request: PackedWriteRequest,
    producer_engine_id: str,
    producer_rank: int,
    producer_tp_size: int,
    max_chunk_bytes_per_rank: int,
) -> ProducerPackedWritePlan:
    """Reconstruct and authenticate one decoder-requested producer pack.

    The decoder supplies only canonical group intervals and cryptographic plan
    identities. The producer derives every block ID and every region slice from
    retained local state, then accepts the request only when the independent
    reconstruction matches byte-for-byte.

    :param roster: Retained physical source roster for ``producer_request_id``.
    :param regions: Producer-local registered KV region descriptors.
    :param request: Typed decoder request for one packed chunk.
    :param producer_engine_id: Receiving producer engine identity.
    :param producer_rank: Receiving producer tensor-parallel rank.
    :param producer_tp_size: Receiving producer tensor-parallel world size.
    :param max_chunk_bytes_per_rank: Configured producer slot payload bound.
    :returns: Authenticated source, packed, and selected chunk plans.
    :raises ValueError: If any request assertion differs from retained state.
    """
    _validate_roster(roster)
    if type(request) is not PackedWriteRequest:
        raise ValueError("request must be a PackedWriteRequest")
    _validate_producer_identity(
        request.command.chunk,
        producer_engine_id=producer_engine_id,
        producer_rank=producer_rank,
        producer_tp_size=producer_tp_size,
    )
    _require_int(
        "max_chunk_bytes_per_rank",
        max_chunk_bytes_per_rank,
        minimum=1,
    )
    group_count = len(roster.block_ids)
    groups = _reconstruct_source_groups(roster, request.source_selections)
    _validate_region_descriptors(regions, group_count=group_count)
    source_plan = build_canonical_source_plan(
        source_tp_size=producer_tp_size,
        source_ranks=tuple(range(producer_tp_size)),
        groups=groups,
        regions=_source_region_ownership(regions),
    )
    packed_plan = build_packed_transfer_plan(
        source_plan,
        max_chunk_bytes_per_rank=max_chunk_bytes_per_rank,
    )
    chunk = _validate_request_identity(
        request,
        roster=roster,
        source_plan=source_plan,
        packed_plan=packed_plan,
    )
    return ProducerPackedWritePlan(
        source_plan=source_plan,
        packed_plan=packed_plan,
        chunk=chunk,
    )
