# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for fused canonical NIXL coalesced scatter."""

from contextlib import nullcontext
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from vllm.distributed.kv_transfer.coalesced_layout import (
    CoalescedTransferPlan,
    GroupTransferRoster,
    RegionOwnership,
    build_coalesced_transfer_plan,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import (
    coalesced_scatter as scatter_module,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.coalesced_scatter import (
    ScatterEnqueueError,
    launch_coalesced_scatter,
    scatter_coalesced_reference,
)
from vllm.triton_utils import HAS_TRITON

_CANARY = 0xD7
_STAGING_BASE = 5
_REDZONE_BYTES = 7
_PRODUCTION_ROW_BYTES = 64 * 1024
_KERNEL_BLOCK_WIDTHS = (512, 1024, 2048, 4096)


def _production_plan() -> CoalescedTransferPlan:
    """Build a mixed-plane, multi-region TP4 production layout.

    :returns: Canonical plan with an empty region and nonstandard row widths.
    """
    groups = (
        GroupTransferRoster(
            group_index=0,
            source_position_start=0,
            destination_plane_count=2,
            local_block_ids=(2, 0),
            remote_block_ids=(10, 11),
        ),
        GroupTransferRoster(
            group_index=1,
            source_position_start=2,
            destination_plane_count=1,
            local_block_ids=(1,),
            remote_block_ids=(20, 21),
        ),
        GroupTransferRoster(
            group_index=2,
            source_position_start=0,
            destination_plane_count=2,
            local_block_ids=(),
            remote_block_ids=(),
        ),
    )
    regions = (
        RegionOwnership(
            region_index=0,
            group_indices=(0,),
            source_row_count=64,
            destination_row_count=4,
            row_bytes=10,
        ),
        RegionOwnership(
            region_index=1,
            group_indices=(1,),
            source_row_count=64,
            destination_row_count=3,
            row_bytes=14,
        ),
        RegionOwnership(
            region_index=2,
            group_indices=(2,),
            source_row_count=64,
            destination_row_count=2,
            row_bytes=18,
        ),
    )
    return build_coalesced_transfer_plan(
        source_tp_size=4,
        source_ranks=(3, 1, 0, 2),
        rank_slots=(2, 0, 3, 1),
        groups=groups,
        regions=regions,
    )


def _empty_plan() -> CoalescedTransferPlan:
    """Build a valid TP4 plan whose only region has no positions.

    :returns: Canonical zero-position plan.
    """
    return build_coalesced_transfer_plan(
        source_tp_size=4,
        source_ranks=(0, 1, 2, 3),
        rank_slots=(0, 1, 2, 3),
        groups=(
            GroupTransferRoster(
                group_index=0,
                source_position_start=0,
                destination_plane_count=2,
                local_block_ids=(),
                remote_block_ids=(),
            ),
        ),
        regions=(
            RegionOwnership(
                region_index=0,
                group_indices=(0,),
                source_row_count=1,
                destination_row_count=2,
                row_bytes=6,
            ),
        ),
    )


def _production_row_plan() -> CoalescedTransferPlan:
    """Build the mixed-plane test plan with production-width cache rows.

    :returns: Canonical plan whose source rows are exactly 64 KiB.
    """
    plan = _production_plan()
    return build_coalesced_transfer_plan(
        source_tp_size=plan.source_tp_size,
        source_ranks=plan.source_ranks,
        rank_slots=plan.rank_slots,
        groups=plan.groups,
        regions=tuple(
            replace(region.ownership, row_bytes=_PRODUCTION_ROW_BYTES)
            for region in plan.regions
        ),
    )


def _large_offset_plan() -> CoalescedTransferPlan:
    """Build one-row geometry whose destination begins at exactly 4 GiB.

    :returns: Canonical dual-plane plan with 64-bit source and destination
        offsets.
    """
    destination_block_id = (1 << 32) // (4 * _PRODUCTION_ROW_BYTES)
    return build_coalesced_transfer_plan(
        source_tp_size=4,
        source_ranks=(3, 1, 0, 2),
        rank_slots=(2, 0, 3, 1),
        groups=(
            GroupTransferRoster(
                group_index=0,
                source_position_start=0,
                destination_plane_count=2,
                local_block_ids=(destination_block_id,),
                remote_block_ids=(0,),
            ),
        ),
        regions=(
            RegionOwnership(
                region_index=0,
                group_indices=(0,),
                source_row_count=1,
                destination_row_count=destination_block_id + 1,
                row_bytes=_PRODUCTION_ROW_BYTES,
            ),
        ),
    )


def _staging(plan: CoalescedTransferPlan) -> torch.Tensor:
    """Build rank-major staging bytes with request redzones.

    :param plan: Canonical plan governing the allocation.
    :returns: Initialized CPU staging tensor.
    """
    staging = torch.full(
        (_STAGING_BASE + plan.staging_size_bytes + _REDZONE_BYTES,),
        _CANARY,
        dtype=torch.uint8,
    )
    for rank_index in range(4):
        for region in plan.regions:
            for position_index in range(len(region.positions)):
                row_bytes = region.ownership.row_bytes
                row_start = (
                    _STAGING_BASE
                    + plan.rank_offset(rank_index)
                    + region.offset_within_rank
                    + position_index * row_bytes
                )
                values = (
                    torch.arange(row_bytes, dtype=torch.int64)
                    + rank_index * 47
                    + region.ownership.region_index * 29
                    + position_index * 13
                ) % 211
                staging[row_start : row_start + row_bytes] = values.to(torch.uint8)
    return staging


def _destination_backings(
    plan: CoalescedTransferPlan,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Allocate exact destination views surrounded by canaries.

    :param plan: Canonical plan describing destination capacities.
    :returns: Backing tensors followed by their exact region views.
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
            (2 * _REDZONE_BYTES + destination_bytes,),
            _CANARY,
            dtype=torch.uint8,
        )
        backings.append(backing)
        destinations.append(
            backing[_REDZONE_BYTES : _REDZONE_BYTES + destination_bytes]
        )
    return tuple(backings), tuple(destinations)


def _assert_redzones(backings: tuple[torch.Tensor, ...]) -> None:
    """Assert that every destination allocation retained both canaries.

    :param backings: Destination tensors containing leading and trailing guards.
    """
    for backing in backings:
        assert torch.all(backing[:_REDZONE_BYTES] == _CANARY)
        assert torch.all(backing[-_REDZONE_BYTES:] == _CANARY)


@pytest.mark.cpu_test
def test_reference_scatter_obeys_rank_major_canonical_layout() -> None:
    """Scatter dual and single-plane rows without touching adjacent storage."""
    plan = _production_plan()
    staging = _staging(plan)
    staging_before = staging.clone()
    backings, destinations = _destination_backings(plan)

    scatter_coalesced_reference(
        staging,
        destinations,
        plan,
        staging_base_offset_bytes=_STAGING_BASE,
    )

    assert torch.equal(staging, staging_before)
    _assert_redzones(backings)

    dual_region = plan.regions[0]
    dual = destinations[0].view(4, 2, 4, 5)
    for rank_index, rank_slot in enumerate(plan.rank_slots):
        rank_region_start = (
            _STAGING_BASE
            + plan.rank_offset(rank_index)
            + dual_region.offset_within_rank
        )
        for position_index, local_block_id in enumerate((2, 0)):
            source_start = rank_region_start + position_index * 10
            assert torch.equal(
                dual[local_block_id, 0, rank_slot],
                staging[source_start : source_start + 5],
            )
            assert torch.equal(
                dual[local_block_id, 1, rank_slot],
                staging[source_start + 5 : source_start + 10],
            )
    assert torch.all(dual[1] == _CANARY)
    assert torch.all(dual[3] == _CANARY)

    single_region = plan.regions[1]
    single = destinations[1].view(3, 4, 2, 7)
    for rank_index, rank_slot in enumerate(plan.rank_slots):
        rank_region_start = (
            _STAGING_BASE
            + plan.rank_offset(rank_index)
            + single_region.offset_within_rank
        )
        for position_index, destination_half in enumerate((0, 1)):
            source_start = rank_region_start + position_index * 14
            assert torch.equal(
                single[1, rank_slot, destination_half],
                staging[source_start : source_start + 7],
            )
        assert torch.all(single[1, rank_slot, 0] != _CANARY)
        assert torch.all(single[1, rank_slot, 1] != _CANARY)
    assert torch.all(single[0] == _CANARY)
    assert torch.all(single[2] == _CANARY)
    assert torch.all(destinations[2] == _CANARY)


@pytest.mark.cpu_test
def test_reference_scatter_preserves_unselected_half_after_odd_prefix_trim() -> None:
    """Place an odd-start single-plane suffix without clobbering cached bytes."""
    plan = build_coalesced_transfer_plan(
        source_tp_size=4,
        source_ranks=(2, 0, 3, 1),
        rank_slots=(1, 3, 0, 2),
        groups=(
            GroupTransferRoster(
                group_index=0,
                source_position_start=3,
                destination_plane_count=1,
                local_block_ids=(1, 2),
                remote_block_ids=(10, 11, 12),
            ),
        ),
        regions=(
            RegionOwnership(
                region_index=0,
                group_indices=(0,),
                source_row_count=64,
                destination_row_count=4,
                row_bytes=10,
            ),
        ),
    )
    staging = _staging(plan)
    backings, destinations = _destination_backings(plan)

    scatter_coalesced_reference(
        staging,
        destinations,
        plan,
        staging_base_offset_bytes=_STAGING_BASE,
    )

    _assert_redzones(backings)
    destination = destinations[0].view(4, 4, 2, 5)
    expected_placements = ((1, 1), (2, 0), (2, 1))
    for rank_index, rank_slot in enumerate(plan.rank_slots):
        rank_start = _STAGING_BASE + plan.rank_offset(rank_index)
        for position_index, (local_block_id, destination_half) in enumerate(
            expected_placements
        ):
            source_start = rank_start + position_index * 10
            assert torch.equal(
                destination[local_block_id, rank_slot, destination_half],
                staging[source_start : source_start + 5],
            )
        assert torch.all(destination[1, rank_slot, 0] == _CANARY)
    assert torch.all(destination[0] == _CANARY)
    assert torch.all(destination[3] == _CANARY)


@pytest.mark.cpu_test
def test_zero_position_region_is_a_valid_no_op() -> None:
    """Keep zero-position regions in the plan without fabricating work."""
    plan = _empty_plan()
    staging = torch.full((11,), _CANARY, dtype=torch.uint8)
    backings, destinations = _destination_backings(plan)

    scatter_coalesced_reference(
        staging,
        destinations,
        plan,
        staging_base_offset_bytes=3,
    )

    assert torch.all(staging == _CANARY)
    assert torch.all(destinations[0] == _CANARY)
    _assert_redzones(backings)


@pytest.mark.cpu_test
def test_scatter_rejects_non_tp4_and_tampered_canonical_geometry() -> None:
    """Fail closed before touching tensors when canonical geometry diverges."""
    plan = _production_plan()
    staging = _staging(plan)
    _, destinations = _destination_backings(plan)

    subset_plan = build_coalesced_transfer_plan(
        source_tp_size=4,
        source_ranks=(1, 3),
        rank_slots=(1, 0),
        groups=plan.groups,
        regions=tuple(region.ownership for region in plan.regions),
    )
    with pytest.raises(ValueError, match="all four TP4 source ranks"):
        scatter_coalesced_reference(
            staging,
            destinations,
            subset_plan,
            staging_base_offset_bytes=_STAGING_BASE,
        )

    tampered_region = replace(plan.regions[1], offset_within_rank=21)
    tampered_plan = replace(
        plan,
        regions=(plan.regions[0], tampered_region, plan.regions[2]),
    )
    with pytest.raises(ValueError, match="packed within each rank slab"):
        scatter_coalesced_reference(
            staging,
            destinations,
            tampered_plan,
            staging_base_offset_bytes=_STAGING_BASE,
        )

    short_destination = destinations[0][:-1]
    with pytest.raises(ValueError, match="expected exactly"):
        scatter_coalesced_reference(
            staging,
            (short_destination, *destinations[1:]),
            plan,
            staging_base_offset_bytes=_STAGING_BASE,
        )

    with pytest.raises(ValueError, match="staging allocation exceeds"):
        scatter_coalesced_reference(
            staging,
            destinations,
            plan,
            staging_base_offset_bytes=_STAGING_BASE + _REDZONE_BYTES + 1,
        )


@pytest.mark.cpu_test
def test_validation_failure_precedes_every_cuda_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rejected launch cannot create an ambiguous device reader."""
    event_calls: list[object] = []

    def reject(*args: object, **kwargs: object) -> None:
        raise ValueError("invalid scatter")

    def create_event(*args: object, **kwargs: object) -> object:
        event_calls.append(object())
        return event_calls[-1]

    monkeypatch.setattr(scatter_module, "validate_coalesced_scatter", reject)
    monkeypatch.setattr(scatter_module.torch.cuda, "Event", create_event)

    with pytest.raises(ValueError, match="invalid scatter"):
        launch_coalesced_scatter(
            object(),
            (object(),),
            _production_plan(),
            object(),
            staging_base_offset_bytes=0,
        )

    assert event_calls == []


@pytest.mark.cpu_test
def test_partial_enqueue_error_returns_event_owned_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A synchronous kernel-launch failure retains and fences prior work."""

    class FakeEvent:
        """Minimal timing event with deterministic completion."""

        recorded: bool

        def __init__(self, *, enable_timing: bool) -> None:
            assert enable_timing
            self.recorded = False

        def record(self, stream: object) -> None:
            self.recorded = True

        def query(self) -> bool:
            return self.recorded

        def elapsed_time(self, end_event: "FakeEvent") -> float:
            assert self.recorded and end_event.recorded
            return 2.5

    class FakeTensor:
        """Tensor-shaped keepalive used without touching a CUDA runtime."""

        device: object = object()

        def reshape(self, *shape: int) -> "FakeTensor":
            return self

    class FailingKernel:
        """Triton-shaped launcher that fails after one complete enqueue."""

        launch_count: int = 0

        def __getitem__(self, grid: tuple[int, ...]):
            def launch(*args: object, **kwargs: object) -> None:
                self.launch_count += 1
                if self.launch_count == 2:
                    raise RuntimeError("injected kernel launch failure")

            return launch

    events: list[FakeEvent] = []

    def create_event(*, enable_timing: bool) -> FakeEvent:
        event = FakeEvent(enable_timing=enable_timing)
        events.append(event)
        return event

    monkeypatch.setattr(
        scatter_module,
        "validate_coalesced_scatter",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(scatter_module.torch.cuda, "Event", create_event)
    monkeypatch.setattr(
        scatter_module.torch.cuda,
        "stream",
        lambda stream: nullcontext(),
    )
    monkeypatch.setattr(
        scatter_module.torch,
        "tensor",
        lambda *args, **kwargs: FakeTensor(),
    )
    monkeypatch.setattr(
        scatter_module,
        "_coalesced_scatter_kernel",
        FailingKernel(),
    )

    with pytest.raises(ScatterEnqueueError) as caught:
        launch_coalesced_scatter(
            FakeTensor(),
            (FakeTensor(), FakeTensor(), FakeTensor()),
            _production_plan(),
            object(),
            staging_base_offset_bytes=0,
        )

    recovery = caught.value.recovery_launch
    assert recovery is not None
    assert recovery.enqueued_region_count == 1
    assert recovery.is_complete()
    assert recovery.gpu_duration_ms() == 2.5
    assert len(recovery._keepalive) == 6
    assert len(events) == 2


@pytest.mark.skipif(
    not torch.cuda.is_available() or not HAS_TRITON,
    reason="CUDA and Triton are required",
)
@pytest.mark.parametrize("kernel_block_bytes", _KERNEL_BLOCK_WIDTHS)
def test_cuda_scatter_matches_production_rows_without_device_wide_sync(
    monkeypatch: pytest.MonkeyPatch,
    kernel_block_bytes: int,
) -> None:
    """Compile production-width rows and match the mixed-plane CPU oracle."""
    monkeypatch.setattr(
        scatter_module,
        "_KERNEL_BLOCK_BYTES",
        kernel_block_bytes,
    )
    plan = _production_row_plan()
    staging_cpu = _staging(plan)
    reference_backings, reference_destinations = _destination_backings(plan)
    scatter_coalesced_reference(
        staging_cpu,
        reference_destinations,
        plan,
        staging_base_offset_bytes=_STAGING_BASE,
    )

    staging_cuda = staging_cpu.cuda()
    cuda_backings = tuple(backing.cuda() for backing in reference_backings)
    for backing in cuda_backings:
        backing.fill_(_CANARY)
    cuda_destinations = tuple(
        backing[_REDZONE_BYTES : backing.numel() - _REDZONE_BYTES]
        for backing in cuda_backings
    )
    stream = torch.cuda.Stream(device=staging_cuda.device)
    stream.wait_stream(torch.cuda.current_stream(staging_cuda.device))
    warmup = launch_coalesced_scatter(
        staging_cuda,
        cuda_destinations,
        plan,
        stream,
        staging_base_offset_bytes=_STAGING_BASE,
    )
    warmup.completion_event.synchronize()

    for backing in cuda_backings:
        backing.fill_(_CANARY)
    stream.wait_stream(torch.cuda.current_stream(staging_cuda.device))

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
        launch = launch_coalesced_scatter(
            staging_cuda,
            cuda_destinations,
            plan,
            stream,
            staging_base_offset_bytes=_STAGING_BASE,
        )

    assert launch.enqueued_region_count == 2
    launch.completion_event.synchronize()
    assert launch.is_complete()
    assert launch.gpu_duration_ms() >= 0.0
    assert torch.equal(staging_cuda.cpu(), staging_cpu)
    for actual, expected in zip(cuda_backings, reference_backings, strict=True):
        assert torch.equal(actual.cpu(), expected)


@pytest.mark.optional
@pytest.mark.skipif(
    not torch.cuda.is_available() or not HAS_TRITON,
    reason="CUDA and Triton are required",
)
def test_cuda_scatter_uses_64_bit_source_and_destination_offsets() -> None:
    """Read and write beyond 4 GiB without truncating kernel byte offsets."""
    plan = _large_offset_plan()
    staging_base = (1 << 32) + 37
    staging_size = staging_base + plan.staging_size_bytes + _REDZONE_BYTES
    destination_size = (
        plan.regions[0].ownership.destination_row_count
        * len(plan.source_ranks)
        * _PRODUCTION_ROW_BYTES
    )
    required_free_bytes = staging_size + destination_size + 1024**3
    free_bytes, _ = torch.cuda.mem_get_info()
    if free_bytes < required_free_bytes:
        pytest.skip(
            "64-bit scatter proof requires "
            f"{required_free_bytes / 1024**3:.2f} GiB free CUDA memory"
        )

    device = torch.device("cuda", torch.cuda.current_device())
    staging = torch.empty(staging_size, dtype=torch.uint8, device=device)
    destination_backing = torch.empty(
        destination_size + 2 * _REDZONE_BYTES,
        dtype=torch.uint8,
        device=device,
    )
    destination = destination_backing[
        _REDZONE_BYTES : _REDZONE_BYTES + destination_size
    ]
    destination_backing[:_REDZONE_BYTES].fill_(_CANARY)
    destination_backing[-_REDZONE_BYTES:].fill_(_CANARY)
    wrapped_source_start = staging_base & ((1 << 32) - 1)
    staging[
        wrapped_source_start : wrapped_source_start + plan.staging_size_bytes
    ].fill_(0x39)

    expected_rows: list[torch.Tensor] = []
    source_row = torch.arange(
        _PRODUCTION_ROW_BYTES,
        dtype=torch.int64,
        device=device,
    )
    for rank_index in range(len(plan.source_ranks)):
        expected = ((source_row + rank_index * 47) % 251).to(torch.uint8)
        source_start = staging_base + plan.rank_offset(rank_index)
        staging[source_start : source_start + _PRODUCTION_ROW_BYTES].copy_(expected)
        expected_rows.append(expected)

    destination_row_bytes = len(plan.source_ranks) * _PRODUCTION_ROW_BYTES
    target_start = destination_size - destination_row_bytes
    preceding_start = target_start - destination_row_bytes

    def reset_destination_guards() -> None:
        destination[:destination_row_bytes].fill_(_CANARY)
        destination[preceding_start:].fill_(_CANARY)

    reset_destination_guards()
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    warmup = launch_coalesced_scatter(
        staging,
        (destination,),
        plan,
        stream,
        staging_base_offset_bytes=staging_base,
    )
    warmup.completion_event.synchronize()

    reset_destination_guards()
    stream.wait_stream(torch.cuda.current_stream(device))
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
        launch = launch_coalesced_scatter(
            staging,
            (destination,),
            plan,
            stream,
            staging_base_offset_bytes=staging_base,
        )

    launch.completion_event.synchronize()
    assert launch.is_complete()
    assert launch.gpu_duration_ms() >= 0.0
    assert torch.all(destination_backing[:_REDZONE_BYTES] == _CANARY)
    assert torch.all(destination_backing[-_REDZONE_BYTES:] == _CANARY)
    assert torch.all(destination[:destination_row_bytes] == _CANARY)
    assert torch.all(destination[preceding_start:target_start] == _CANARY)
    target = destination[target_start:].view(
        2,
        len(plan.source_ranks),
        _PRODUCTION_ROW_BYTES // 2,
    )
    for rank_index, rank_slot in enumerate(plan.rank_slots):
        expected = expected_rows[rank_index]
        assert torch.equal(target[0, rank_slot], expected[: _PRODUCTION_ROW_BYTES // 2])
        assert torch.equal(target[1, rank_slot], expected[_PRODUCTION_ROW_BYTES // 2 :])
        source_start = staging_base + plan.rank_offset(rank_index)
        assert torch.equal(
            staging[source_start : source_start + _PRODUCTION_ROW_BYTES],
            expected,
        )
    assert torch.all(
        staging[wrapped_source_start : wrapped_source_start + plan.staging_size_bytes]
        == 0x39
    )
