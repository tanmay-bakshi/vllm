# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Asynchronous TP4-to-TP1 scatter for canonical coalesced NIXL reads."""

import traceback
from dataclasses import dataclass, field

import torch

from vllm.distributed.kv_transfer.coalesced_layout import (
    DUAL_PLANE_DESTINATION_HALF,
    CoalescedTransferPlan,
)
from vllm.triton_utils import HAS_TRITON, tl, triton

_SOURCE_RANK_COUNT = 4
_KERNEL_BLOCK_BYTES = 4096


class ScatterEnqueueError(RuntimeError):
    """Report a scatter launch failure and any available quiescence proof."""

    recovery_launch: "ScatterLaunch | None"

    def __init__(
        self,
        message: str,
        recovery_launch: "ScatterLaunch | None",
    ) -> None:
        """Create a contextual asynchronous launch error.

        :param message: Launch and recovery traceback evidence.
        :param recovery_launch: Event owning any work enqueued before the
            failure, or ``None`` when the stream could not provide a completion
            proof.
        """
        super().__init__(message)
        self.recovery_launch = recovery_launch


@dataclass(frozen=True, slots=True)
class ScatterLaunch:
    """Own one asynchronous scatter until its completion event becomes ready.

    The handle retains every tensor allocated or consumed by the kernels. Its
    owner must keep the handle alive until :meth:`is_complete` returns ``True``.

    :ivar start_event: Timing event recorded before region work is enqueued.
    :ivar completion_event: Event recorded after every region kernel on the
        caller-supplied stream.
    :ivar enqueued_region_count: Number of non-empty region kernels enqueued.
    """

    start_event: torch.cuda.Event
    completion_event: torch.cuda.Event
    enqueued_region_count: int
    _keepalive: tuple[torch.Tensor, ...] = field(repr=False)

    def is_complete(self) -> bool:
        """Return whether every scatter kernel has completed.

        :returns: ``True`` after the completion event has executed.
        """
        return self.completion_event.query()

    def gpu_duration_ms(self) -> float:
        """Return event-measured GPU duration after completion.

        :returns: Milliseconds between the launch boundary and completion event.
        :raises RuntimeError: If the completion event is not ready.
        """
        if not self.is_complete():
            raise RuntimeError("scatter GPU duration is unavailable before completion")
        return self.start_event.elapsed_time(self.completion_event)


@triton.jit
def _coalesced_scatter_kernel(
    staging_ptr,
    destination_ptr,
    region_metadata_ptr,
    staging_region_offset_bytes,
    rank_stride_bytes,
    position_count,
    rank_slot_0,
    rank_slot_1,
    rank_slot_2,
    rank_slot_3,
    SOURCE_ROW_BYTES: tl.constexpr,
    SOURCE_PLANE_BYTES: tl.constexpr,
    DESTINATION_ROW_BYTES: tl.constexpr,
    SOURCE_RANK_COUNT: tl.constexpr,
    BLOCK_BYTES: tl.constexpr,
):
    position = tl.program_id(0).to(tl.int64)
    source_rank = tl.program_id(1).to(tl.int64)

    local_block_id = tl.load(region_metadata_ptr + position).to(tl.int64)
    destination_half = tl.load(region_metadata_ptr + position_count + position).to(
        tl.int64
    )
    rank_slot = tl.where(
        source_rank == 0,
        rank_slot_0,
        tl.where(
            source_rank == 1,
            rank_slot_1,
            tl.where(source_rank == 2, rank_slot_2, rank_slot_3),
        ),
    ).to(tl.int64)
    is_dual_plane = destination_half < 0
    copy_bytes = tl.where(is_dual_plane, SOURCE_ROW_BYTES, SOURCE_PLANE_BYTES)
    lane_offsets = tl.arange(0, BLOCK_BYTES)
    for byte_start in tl.range(0, SOURCE_ROW_BYTES, BLOCK_BYTES):
        byte_offsets = byte_start + lane_offsets
        byte_mask = byte_offsets < copy_bytes

        source_offset = (
            staging_region_offset_bytes
            + source_rank * rank_stride_bytes
            + position * SOURCE_ROW_BYTES
            + byte_offsets
        )
        values = tl.load(staging_ptr + source_offset, mask=byte_mask)

        source_plane = byte_offsets // SOURCE_PLANE_BYTES
        source_plane_offset = byte_offsets - source_plane * SOURCE_PLANE_BYTES
        dual_plane_offset = (
            source_plane * SOURCE_RANK_COUNT * SOURCE_PLANE_BYTES
            + rank_slot * SOURCE_PLANE_BYTES
            + source_plane_offset
        )
        single_plane_offset = (
            rank_slot * SOURCE_ROW_BYTES
            + destination_half * SOURCE_PLANE_BYTES
            + byte_offsets
        )
        destination_offset = local_block_id * DESTINATION_ROW_BYTES + tl.where(
            is_dual_plane, dual_plane_offset, single_plane_offset
        )
        tl.store(destination_ptr + destination_offset, values, mask=byte_mask)


def scatter_coalesced_reference(
    staging: torch.Tensor,
    destinations: tuple[torch.Tensor, ...],
    plan: CoalescedTransferPlan,
    *,
    staging_base_offset_bytes: int,
) -> None:
    """Apply the canonical coalesced scatter synchronously on CPU tensors.

    This implementation is the executable layout specification used by host
    tests. It intentionally favors explicit indexing over vectorization.

    :param staging: Contiguous CPU ``uint8`` staging tensor.
    :param destinations: Contiguous CPU ``uint8`` destination tensors, one per
        canonical region.
    :param plan: The canonical plan shared with descriptor construction,
        localization, and scatter.
    :param staging_base_offset_bytes: Byte offset of this request's allocation
        within ``staging``.
    """
    _validate_tensors(
        staging,
        destinations,
        plan,
        staging_base_offset_bytes,
        "cpu",
    )

    for destination, region in zip(destinations, plan.regions, strict=True):
        source_row_bytes = region.ownership.row_bytes
        source_plane_bytes = source_row_bytes // 2
        destination_row_bytes = _SOURCE_RANK_COUNT * source_row_bytes
        destination_flat = destination.reshape(-1)

        for source_rank_index, rank_slot in enumerate(plan.rank_slots):
            region_source_start = (
                staging_base_offset_bytes
                + plan.rank_offset(source_rank_index)
                + region.offset_within_rank
            )
            for position_index, position in enumerate(region.positions):
                source_start = region_source_start + position_index * source_row_bytes
                destination_row_start = position.local_block_id * destination_row_bytes

                if position.destination_half == DUAL_PLANE_DESTINATION_HALF:
                    for source_plane in range(2):
                        source_plane_start = (
                            source_start + source_plane * source_plane_bytes
                        )
                        destination_start = (
                            destination_row_start
                            + source_plane * _SOURCE_RANK_COUNT * source_plane_bytes
                            + rank_slot * source_plane_bytes
                        )
                        destination_flat[
                            destination_start : destination_start + source_plane_bytes
                        ].copy_(
                            staging[
                                source_plane_start : source_plane_start
                                + source_plane_bytes
                            ]
                        )
                    continue

                destination_start = (
                    destination_row_start
                    + rank_slot * source_row_bytes
                    + position.destination_half * source_plane_bytes
                )
                destination_flat[
                    destination_start : destination_start + source_plane_bytes
                ].copy_(staging[source_start : source_start + source_plane_bytes])


def launch_coalesced_scatter(
    staging: torch.Tensor,
    destinations: tuple[torch.Tensor, ...],
    plan: CoalescedTransferPlan,
    stream: torch.cuda.Stream,
    *,
    staging_base_offset_bytes: int,
) -> ScatterLaunch:
    """Enqueue the canonical coalesced scatter on a caller-owned CUDA stream.

    The function never synchronizes a stream or device. The returned handle is
    both the completion proof and the lifetime owner for launch metadata.

    :param staging: Contiguous CUDA ``uint8`` staging tensor.
    :param destinations: Contiguous CUDA ``uint8`` destination tensors, one per
        canonical region.
    :param plan: The canonical plan shared with descriptor construction,
        localization, and scatter.
    :param stream: CUDA stream on which metadata copies, kernels, and the
        completion event are enqueued.
    :param staging_base_offset_bytes: Byte offset of this request's allocation
        within ``staging``.
    :returns: An ownership handle for the asynchronous launch.
    :raises RuntimeError: If Triton is unavailable.
    """
    validate_coalesced_scatter(
        staging,
        destinations,
        plan,
        stream,
        staging_base_offset_bytes=staging_base_offset_bytes,
    )

    keepalive: list[torch.Tensor] = [staging, *destinations]
    enqueued_region_count = 0
    start_event = torch.cuda.Event(enable_timing=True)
    try:
        with torch.cuda.stream(stream):
            start_event.record(stream)
            for destination, region in zip(destinations, plan.regions, strict=True):
                position_count = len(region.positions)
                if position_count == 0:
                    continue

                region_metadata = torch.tensor(
                    (
                        tuple(position.local_block_id for position in region.positions),
                        tuple(
                            position.destination_half for position in region.positions
                        ),
                    ),
                    dtype=torch.int64,
                    device=staging.device,
                )
                keepalive.append(region_metadata)

                source_row_bytes = region.ownership.row_bytes
                source_plane_bytes = source_row_bytes // 2
                destination_row_bytes = _SOURCE_RANK_COUNT * source_row_bytes
                grid = (
                    position_count,
                    _SOURCE_RANK_COUNT,
                )
                _coalesced_scatter_kernel[grid](
                    staging,
                    destination.reshape(-1),
                    region_metadata,
                    staging_base_offset_bytes + region.offset_within_rank,
                    plan.rank_stride_bytes,
                    position_count,
                    *plan.rank_slots,
                    SOURCE_ROW_BYTES=source_row_bytes,
                    SOURCE_PLANE_BYTES=source_plane_bytes,
                    DESTINATION_ROW_BYTES=destination_row_bytes,
                    SOURCE_RANK_COUNT=_SOURCE_RANK_COUNT,
                    BLOCK_BYTES=_KERNEL_BLOCK_BYTES,
                    num_warps=8,
                )
                enqueued_region_count += 1

            completion_event = torch.cuda.Event(enable_timing=True)
            completion_event.record(stream)
    except Exception as error:
        launch_traceback = traceback.format_exc()
        recovery_launch: ScatterLaunch | None = None
        try:
            with torch.cuda.stream(stream):
                recovery_event = torch.cuda.Event(enable_timing=True)
                recovery_event.record(stream)
            recovery_launch = ScatterLaunch(
                start_event=start_event,
                completion_event=recovery_event,
                enqueued_region_count=enqueued_region_count,
                _keepalive=tuple(keepalive),
            )
            recovery_traceback = ""
        except Exception:
            recovery_traceback = (
                "\ncompletion-event recovery also failed\n" + traceback.format_exc()
            )
        raise ScatterEnqueueError(
            "scatter enqueue failed after validation\n"
            + launch_traceback
            + recovery_traceback,
            recovery_launch,
        ) from error

    return ScatterLaunch(
        start_event=start_event,
        completion_event=completion_event,
        enqueued_region_count=enqueued_region_count,
        _keepalive=tuple(keepalive),
    )


def validate_coalesced_scatter(
    staging: torch.Tensor,
    destinations: tuple[torch.Tensor, ...],
    plan: CoalescedTransferPlan,
    stream: torch.cuda.Stream,
    *,
    staging_base_offset_bytes: int,
) -> None:
    """Validate a CUDA scatter before any staging reader is enqueued.

    :param staging: Contiguous CUDA ``uint8`` staging tensor.
    :param destinations: Contiguous CUDA ``uint8`` destination tensors.
    :param plan: Canonical coalesced transfer plan.
    :param stream: Stream that will own the scatter launch.
    :param staging_base_offset_bytes: Request allocation offset in ``staging``.
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton is required for CUDA coalesced scatter")
    _validate_tensors(
        staging,
        destinations,
        plan,
        staging_base_offset_bytes,
        "cuda",
    )
    if torch.device(stream.device) != staging.device:
        raise ValueError("the caller-supplied stream must match the tensor device")


def _validate_tensors(
    staging: torch.Tensor,
    destinations: tuple[torch.Tensor, ...],
    plan: CoalescedTransferPlan,
    staging_base_offset_bytes: int,
    device_type: str,
) -> None:
    """Validate tensors and the production TP4 subset of the canonical plan.

    :param staging: Candidate staging tensor.
    :param destinations: Candidate region destination tensors.
    :param plan: Canonical scatter plan.
    :param staging_base_offset_bytes: Candidate request-allocation offset.
    :param device_type: Required PyTorch device type.
    """
    _validate_plan(plan)
    if type(staging_base_offset_bytes) is not int or staging_base_offset_bytes < 0:
        raise ValueError("staging_base_offset_bytes must be a non-negative integer")
    if staging.device.type != device_type:
        raise ValueError(f"staging must be on a {device_type} device")
    if staging.dtype is not torch.uint8:
        raise TypeError("staging must have dtype torch.uint8")
    if not staging.is_contiguous():
        raise ValueError("staging must be contiguous")
    if staging.ndim != 1:
        raise ValueError("staging must be a one-dimensional byte tensor")
    if len(destinations) != len(plan.regions):
        raise ValueError("destinations and canonical regions must have equal lengths")

    staging_end = staging_base_offset_bytes + plan.staging_size_bytes
    if staging_end > staging.numel():
        raise ValueError("the canonical staging allocation exceeds the tensor")

    for region_index, (destination, region) in enumerate(
        zip(destinations, plan.regions, strict=True)
    ):
        if destination.device != staging.device:
            raise ValueError(
                f"destination region {region_index} must share the staging device"
            )
        if destination.dtype is not torch.uint8:
            raise TypeError(
                f"destination region {region_index} must have dtype torch.uint8"
            )
        if not destination.is_contiguous():
            raise ValueError(f"destination region {region_index} must be contiguous")

        destination_size_bytes = (
            region.ownership.destination_row_count
            * _SOURCE_RANK_COUNT
            * region.ownership.row_bytes
        )
        if destination.numel() != destination_size_bytes:
            raise ValueError(
                f"destination region {region_index} has {destination.numel()} "
                f"bytes, expected exactly {destination_size_bytes}"
            )


def _validate_plan(plan: CoalescedTransferPlan) -> None:
    """Fail closed if canonical geometry does not describe TP4-to-TP1.

    :param plan: Canonical plan to validate for the production scatter kernel.
    """
    if plan.source_tp_size != _SOURCE_RANK_COUNT:
        raise ValueError("coalesced scatter requires a TP4 source plan")
    if tuple(sorted(plan.source_ranks)) != tuple(range(_SOURCE_RANK_COUNT)):
        raise ValueError("coalesced scatter requires all four TP4 source ranks")
    if tuple(sorted(plan.rank_slots)) != tuple(range(_SOURCE_RANK_COUNT)):
        raise ValueError("rank_slots must be a complete TP4-to-TP1 bijection")
    if len(plan.regions) == 0:
        raise ValueError("the canonical plan must contain at least one region")

    expected_region_offset = 0
    for expected_region_index, region in enumerate(plan.regions):
        ownership = region.ownership
        if ownership.region_index != expected_region_index:
            raise ValueError("canonical region indices must be contiguous")
        if ownership.row_bytes <= 0 or ownership.row_bytes % 2 != 0:
            raise ValueError("canonical source row bytes must be positive and even")
        if ownership.destination_row_count <= 0:
            raise ValueError("canonical destination row count must be positive")
        if region.offset_within_rank != expected_region_offset:
            raise ValueError("canonical regions must be packed within each rank slab")

        occupied_halves: dict[int, int] = {}
        for position in region.positions:
            if (
                position.local_block_id < 0
                or position.local_block_id >= ownership.destination_row_count
            ):
                raise ValueError("canonical local block ID is outside its region")
            if position.destination_half not in (
                DUAL_PLANE_DESTINATION_HALF,
                0,
                1,
            ):
                raise ValueError("canonical destination half must be -1, 0, or 1")

            write_mask = (
                0b11
                if position.destination_half == DUAL_PLANE_DESTINATION_HALF
                else 1 << position.destination_half
            )
            prior_mask = occupied_halves.get(position.local_block_id, 0)
            if prior_mask & write_mask != 0:
                raise ValueError(
                    "canonical region contains overlapping destination writes"
                )
            occupied_halves[position.local_block_id] = prior_mask | write_mask

        expected_region_offset += len(region.positions) * ownership.row_bytes

    if plan.rank_stride_bytes != expected_region_offset:
        raise ValueError("canonical rank stride does not match its packed regions")
    expected_staging_size = _SOURCE_RANK_COUNT * plan.rank_stride_bytes
    if plan.staging_size_bytes != expected_staging_size:
        raise ValueError("canonical staging size does not match its TP4 rank slabs")
