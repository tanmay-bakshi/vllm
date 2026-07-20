# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for producer-side packed-write plan reconstruction."""

import msgspec
import pytest

from vllm.distributed.kv_transfer.coalesced_layout import (
    CanonicalSourceTransferPlan,
    PackedTransferPlan,
    SourceGroupTransferRoster,
    SourceRegionOwnership,
    build_canonical_source_plan,
    build_packed_transfer_plan,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    PackedWriteChunkIdentity,
    PackedWriteCommand,
    PackedWriteConsumerEndpoint,
    PackedWriteConsumerPoolGeometry,
    PackedWriteRequest,
    PackedWriteSlotBinding,
    PackedWriteSourceSelection,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.packed_write_plan import (
    ProducerPackedWritePlan,
    reconstruct_producer_packed_write_plan,
)
from vllm.distributed.kv_transfer.nixl_contracts import (
    NixlRegionDescriptor,
    NixlSourceRoster,
)

_CHUNK_BYTES_PER_RANK = 16
_PRODUCER_ENGINE_ID = "prefill-engine"
_PRODUCER_RANK = 2
_PRODUCER_TP_SIZE = 4


def _roster() -> NixlSourceRoster:
    """Build retained physical source state with three cache groups.

    :returns: A valid producer roster.
    """
    return NixlSourceRoster(
        offer_generation=7,
        iteration=0,
        expected_consumers=2,
        valid_token_extent=32755,
        group_token_capacities=(16, 16, 16),
        block_ids=(
            (5, 6, 7, 8),
            (0, 3),
            (4, 9, 10),
        ),
    )


def _region(
    region_index: int,
    *,
    group_indices: tuple[int, ...],
    row_bytes: int,
) -> NixlRegionDescriptor:
    """Build one internally consistent registered source region.

    :param region_index: Canonical test-region index.
    :param group_indices: Groups backed by the region.
    :param row_bytes: Bytes in one physical source row.
    :returns: A valid registered-region descriptor.
    """
    row_count = 16
    return NixlRegionDescriptor(
        semantic_name=f"region-{region_index}",
        group_indices=group_indices,
        group_semantic_names=tuple(
            (group_index, f"group-{group_index}-region-{region_index}")
            for group_index in group_indices
        ),
        base_address=0x100000 + region_index * 0x10000,
        registered_bytes=row_count * row_bytes,
        row_bytes=row_bytes,
        shape=(row_count, row_bytes),
        strides=(row_bytes, 1),
        dtype="torch.uint8",
        element_size_bytes=1,
        layout="HND",
    )


def _regions() -> tuple[NixlRegionDescriptor, ...]:
    """Build multi-region ownership for every canonical group.

    :returns: Valid producer-local source regions.
    """
    return (
        _region(0, group_indices=(0, 1), row_bytes=8),
        _region(1, group_indices=(2,), row_bytes=12),
    )


def _selections() -> tuple[PackedWriteSourceSelection, ...]:
    """Build suffix selections including one zero-count canonical group.

    :returns: Exact decoder-advertised source intervals.
    """
    return (
        PackedWriteSourceSelection(
            group_index=0,
            source_position_start=2,
            position_count=2,
        ),
        PackedWriteSourceSelection(
            group_index=1,
            source_position_start=2,
            position_count=0,
        ),
        PackedWriteSourceSelection(
            group_index=2,
            source_position_start=1,
            position_count=2,
        ),
    )


def _plans(
    roster: NixlSourceRoster,
    regions: tuple[NixlRegionDescriptor, ...],
    selections: tuple[PackedWriteSourceSelection, ...],
    *,
    max_chunk_bytes_per_rank: int = _CHUNK_BYTES_PER_RANK,
) -> tuple[CanonicalSourceTransferPlan, PackedTransferPlan]:
    """Independently build expected canonical and packed plans.

    :param roster: Retained physical source roster.
    :param regions: Registered source regions.
    :param selections: Exact suffix selections.
    :param max_chunk_bytes_per_rank: Packed chunk bound.
    :returns: Expected canonical source and packed plans.
    """
    source_plan = build_canonical_source_plan(
        source_tp_size=_PRODUCER_TP_SIZE,
        source_ranks=tuple(range(_PRODUCER_TP_SIZE)),
        groups=tuple(
            SourceGroupTransferRoster(
                group_index=selection.group_index,
                source_position_start=selection.source_position_start,
                remote_block_ids=roster.block_ids[selection.group_index][
                    selection.source_position_start :
                ],
            )
            for selection in selections
        ),
        regions=tuple(
            SourceRegionOwnership(
                region_index=region_index,
                group_indices=region.group_indices,
                source_row_count=region.shape[0],
                row_bytes=region.row_bytes,
            )
            for region_index, region in enumerate(regions)
        ),
    )
    return source_plan, build_packed_transfer_plan(
        source_plan,
        max_chunk_bytes_per_rank=max_chunk_bytes_per_rank,
    )


def _request(
    *,
    roster: NixlSourceRoster | None = None,
    regions: tuple[NixlRegionDescriptor, ...] | None = None,
    selections: tuple[PackedWriteSourceSelection, ...] | None = None,
    chunk_ordinal: int = 1,
    max_chunk_bytes_per_rank: int = _CHUNK_BYTES_PER_RANK,
) -> PackedWriteRequest:
    """Build a request whose identities match producer reconstruction.

    :param roster: Optional retained source roster override.
    :param regions: Optional registered source-region override.
    :param selections: Optional source-selection override.
    :param chunk_ordinal: Packed chunk requested from this producer rank.
    :param max_chunk_bytes_per_rank: Packed chunk bound used by the decoder.
    :returns: A fully self-consistent packed-write request.
    """
    source_roster = _roster() if roster is None else roster
    source_regions = _regions() if regions is None else regions
    source_selections = _selections() if selections is None else selections
    source_plan, packed_plan = _plans(
        source_roster,
        source_regions,
        source_selections,
        max_chunk_bytes_per_rank=max_chunk_bytes_per_rank,
    )
    chunk = packed_plan.chunks[chunk_ordinal]
    registration_generation = "decoder-registration-generation"
    identity = PackedWriteChunkIdentity(
        producer_engine_id=_PRODUCER_ENGINE_ID,
        producer_request_id="prefill-request",
        offer_generation=source_roster.offer_generation,
        producer_rank=_PRODUCER_RANK,
        producer_tp_size=_PRODUCER_TP_SIZE,
        consumer_engine_id="decode-engine",
        consumer_request_id="decode-request",
        consumer_rank=0,
        consumer_tp_size=1,
        source_plan_digest=source_plan.digest,
        packed_plan_digest=packed_plan.digest,
        chunk_plan_digest=chunk.digest,
        chunk_ordinal=chunk_ordinal,
        chunk_count=len(packed_plan.chunks),
        valid_token_extent=source_roster.valid_token_extent,
        exact_bytes=chunk.rank_stride_bytes,
    )
    return PackedWriteRequest(
        command=PackedWriteCommand(
            chunk=identity,
            destination=PackedWriteSlotBinding(
                pool_registration_generation=registration_generation,
                slot_index=1,
                slot_generation=3,
                payload_bytes=chunk.rank_stride_bytes,
            ),
        ),
        consumer_endpoint=PackedWriteConsumerEndpoint(
            consumer_engine_id="decode-engine",
            consumer_rank=0,
            consumer_tp_size=1,
            registration_generation=registration_generation,
            agent_metadata=b"decoder-agent-metadata",
            consumer_pool=PackedWriteConsumerPoolGeometry(
                registration_generation=registration_generation,
                base_address=0x200000,
                registered_bytes=2 * _PRODUCER_TP_SIZE * max_chunk_bytes_per_rank,
                slot_size_bytes=_PRODUCER_TP_SIZE * max_chunk_bytes_per_rank,
                slot_count=2,
                source_tp_size=_PRODUCER_TP_SIZE,
                rank_stride_bytes=max_chunk_bytes_per_rank,
                device_id=0,
                alignment_bytes=4,
            ),
        ),
        source_selections=source_selections,
    )


def _reconstruct(
    request: PackedWriteRequest,
    *,
    roster: NixlSourceRoster | None = None,
    regions: tuple[NixlRegionDescriptor, ...] | None = None,
    producer_engine_id: str = _PRODUCER_ENGINE_ID,
    producer_rank: int = _PRODUCER_RANK,
    producer_tp_size: int = _PRODUCER_TP_SIZE,
    max_chunk_bytes_per_rank: int = _CHUNK_BYTES_PER_RANK,
) -> ProducerPackedWritePlan:
    """Invoke producer reconstruction with canonical defaults.

    :param request: Packed request to validate.
    :param roster: Optional producer roster override.
    :param regions: Optional producer region override.
    :param producer_engine_id: Receiving producer engine identity.
    :param producer_rank: Receiving producer rank.
    :param producer_tp_size: Receiving producer world size.
    :param max_chunk_bytes_per_rank: Receiving producer chunk bound.
    :returns: Reconstructed producer plan.
    """
    return reconstruct_producer_packed_write_plan(
        roster=_roster() if roster is None else roster,
        regions=_regions() if regions is None else regions,
        request=request,
        producer_engine_id=producer_engine_id,
        producer_rank=producer_rank,
        producer_tp_size=producer_tp_size,
        max_chunk_bytes_per_rank=max_chunk_bytes_per_rank,
    )


def test_reconstructs_exact_source_plan_and_requested_chunk() -> None:
    request = _request()
    expected_source, expected_packed = _plans(
        _roster(),
        _regions(),
        _selections(),
    )

    reconstructed = _reconstruct(request)

    assert reconstructed.source_plan == expected_source
    assert reconstructed.packed_plan == expected_packed
    assert reconstructed.chunk == expected_packed.chunks[1]
    assert reconstructed.source_plan.source_ranks == (0, 1, 2, 3)
    assert reconstructed.source_plan.groups[0].remote_block_ids == (7, 8)
    assert reconstructed.source_plan.groups[1].remote_block_ids == ()
    assert reconstructed.source_plan.groups[2].remote_block_ids == (9, 10)
    assert len(reconstructed.packed_plan.chunks) == 3
    assert reconstructed.chunk.rank_stride_bytes == 12


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("offer_generation", 8, "stale source offer"),
        ("valid_token_extent", 32754, "token extent"),
        ("source_plan_digest", "a" * 64, "source-plan digest"),
        ("packed_plan_digest", "b" * 64, "packed-plan digest"),
        ("chunk_plan_digest", "c" * 64, "chunk-plan digest"),
        ("chunk_count", 4, "chunk count"),
        ("exact_bytes", 13, "byte count"),
    ],
)
def test_rejects_every_untrusted_plan_assertion(
    field: str,
    value: int | str,
    message: str,
) -> None:
    request = _request()
    changed_identity = msgspec.structs.replace(
        request.command.chunk,
        **{field: value},
    )
    changed_destination = request.command.destination
    if field == "exact_bytes":
        changed_destination = msgspec.structs.replace(
            changed_destination,
            payload_bytes=value,
        )
    changed_command = msgspec.structs.replace(
        request.command,
        chunk=changed_identity,
        destination=changed_destination,
    )
    changed_request = msgspec.structs.replace(
        request,
        command=changed_command,
    )

    with pytest.raises(ValueError, match=message):
        _reconstruct(changed_request)


@pytest.mark.parametrize(
    ("expected", "message"),
    [
        ({"producer_engine_id": "other-prefill"}, "different producer identity"),
        ({"producer_rank": 1}, "different producer identity"),
        ({"producer_tp_size": 5}, "different producer identity"),
    ],
)
def test_rejects_request_addressed_to_another_producer(
    expected: dict[str, int | str],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _reconstruct(_request(), **expected)


@pytest.mark.parametrize(
    "selection",
    [
        PackedWriteSourceSelection(
            group_index=0,
            source_position_start=1,
            position_count=2,
        ),
        PackedWriteSourceSelection(
            group_index=0,
            source_position_start=5,
            position_count=0,
        ),
    ],
)
def test_rejects_non_suffix_and_out_of_range_selections(
    selection: PackedWriteSourceSelection,
) -> None:
    valid = _request()
    changed_selections = (selection, *valid.source_selections[1:])
    changed_request = msgspec.structs.replace(
        valid,
        source_selections=changed_selections,
    )

    with pytest.raises(ValueError, match="exact retained suffix"):
        _reconstruct(changed_request)


def test_rejects_zero_count_selection_that_does_not_name_the_suffix_end() -> None:
    valid = _request()
    changed_selections = (
        valid.source_selections[0],
        PackedWriteSourceSelection(
            group_index=1,
            source_position_start=1,
            position_count=0,
        ),
        valid.source_selections[2],
    )
    changed_request = msgspec.structs.replace(
        valid,
        source_selections=changed_selections,
    )

    with pytest.raises(ValueError, match="exact retained suffix"):
        _reconstruct(changed_request)


def test_rejects_selection_group_count_different_from_retained_roster() -> None:
    roster = msgspec.structs.replace(
        _roster(),
        group_token_capacities=(16, 16, 16, 16),
        block_ids=(*_roster().block_ids, (11,)),
    )

    with pytest.raises(ValueError, match="selection count"):
        _reconstruct(_request(), roster=roster)


def test_rejects_retained_offer_and_extent_different_from_request() -> None:
    request = _request()

    with pytest.raises(ValueError, match="stale source offer"):
        _reconstruct(
            request,
            roster=msgspec.structs.replace(_roster(), offer_generation=6),
        )
    with pytest.raises(ValueError, match="token extent"):
        _reconstruct(
            request,
            roster=msgspec.structs.replace(_roster(), valid_token_extent=2048),
        )


def test_rejects_decoder_and_producer_chunk_bound_disagreement() -> None:
    request = _request(max_chunk_bytes_per_rank=24)

    with pytest.raises(ValueError, match="packed-plan digest"):
        _reconstruct(request)


@pytest.mark.parametrize(
    ("regions", "message"),
    [
        (
            (
                msgspec.structs.replace(
                    _regions()[0],
                    shape=(15, 8),
                ),
                _regions()[1],
            ),
            "leading dimension",
        ),
        (
            (
                msgspec.structs.replace(
                    _regions()[0],
                    group_indices=(0,),
                    group_semantic_names=((0, "group-0-region-0"),),
                ),
                _regions()[1],
            ),
            "does not cover every group",
        ),
        (
            (
                _regions()[0],
                msgspec.structs.replace(
                    _regions()[1],
                    base_address=_regions()[0].base_address + 16,
                ),
            ),
            "overlap",
        ),
    ],
)
def test_rejects_malformed_local_region_contracts(
    regions: tuple[NixlRegionDescriptor, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _reconstruct(_request(), regions=regions)


def test_rejects_source_block_outside_registered_rows() -> None:
    roster = msgspec.structs.replace(
        _roster(),
        block_ids=((5, 6, 7, 16), *_roster().block_ids[1:]),
    )

    with pytest.raises(ValueError, match="outside its source rows"):
        _reconstruct(_request(), roster=roster)


def test_allows_sliding_window_roster_smaller_than_valid_token_extent() -> None:
    roster = msgspec.structs.replace(
        _roster(),
        valid_token_extent=131072,
    )
    request = _request(roster=roster)

    assert _reconstruct(request, roster=roster).source_plan.groups[
        0
    ].remote_block_ids == (
        7,
        8,
    )
