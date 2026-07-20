# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Asynchronous producer gather for bounded producer-initiated NIXL writes."""

import traceback
from dataclasses import dataclass, field

import torch

from vllm.distributed.kv_transfer.coalesced_layout import (
    CanonicalSourceTransferPlan,
    PackedTransferChunk,
    PackedTransferPlan,
    validate_packed_transfer_plan,
)
from vllm.triton_utils import HAS_TRITON, tl, triton

_KERNEL_BLOCK_BYTES = 4096


class PackEnqueueError(RuntimeError):
    """Report a pack launch failure and any available quiescence proof."""

    recovery_launch: "PackLaunch | None"

    def __init__(
        self,
        message: str,
        recovery_launch: "PackLaunch | None",
    ) -> None:
        super().__init__(message)
        self.recovery_launch = recovery_launch


@dataclass(frozen=True, slots=True)
class PackLaunch:
    """Own one asynchronous pack operation through its completion event."""

    start_event: torch.cuda.Event
    completion_event: torch.cuda.Event
    enqueued_region_count: int
    _keepalive: tuple[torch.Tensor, ...] = field(repr=False)

    def is_complete(self) -> bool:
        """Return whether every gather kernel has completed."""
        return self.completion_event.query()

    def gpu_duration_ms(self) -> float:
        """Return event-measured GPU duration after completion."""
        if self.is_complete() is False:
            raise RuntimeError("pack GPU duration is unavailable before completion")
        return self.start_event.elapsed_time(self.completion_event)


@triton.jit
def _coalesced_pack_kernel(
    source_ptr,
    destination_ptr,
    remote_block_ids_ptr,
    destination_offset_bytes,
    position_count,
    ROW_BYTES: tl.constexpr,
    BLOCK_BYTES: tl.constexpr,
):
    position = tl.program_id(0).to(tl.int64)
    remote_block_id = tl.load(remote_block_ids_ptr + position).to(tl.int64)
    lane_offsets = tl.arange(0, BLOCK_BYTES)
    for byte_start in tl.range(0, ROW_BYTES, BLOCK_BYTES):
        byte_offsets = byte_start + lane_offsets
        byte_mask = byte_offsets < ROW_BYTES
        values = tl.load(
            source_ptr + remote_block_id * ROW_BYTES + byte_offsets,
            mask=byte_mask,
        )
        tl.store(
            destination_ptr
            + destination_offset_bytes
            + position * ROW_BYTES
            + byte_offsets,
            values,
            mask=byte_mask,
        )


def pack_chunk_reference(
    destination: torch.Tensor,
    sources: tuple[torch.Tensor, ...],
    source_plan: CanonicalSourceTransferPlan,
    packed_plan: PackedTransferPlan,
    chunk_index: int,
    *,
    destination_base_offset_bytes: int,
) -> None:
    """Gather one canonical source chunk synchronously on CPU tensors."""
    chunk = _validate_pack_tensors(
        destination,
        sources,
        source_plan,
        packed_plan,
        chunk_index,
        destination_base_offset_bytes,
        "cpu",
    )
    for region_slice in chunk.region_slices:
        source_region = source_plan.regions[region_slice.region_index]
        source = sources[region_slice.region_index].reshape(-1)
        position_end = region_slice.position_start + region_slice.position_count
        positions = source_region.positions[region_slice.position_start : position_end]
        row_bytes = source_region.ownership.row_bytes
        destination_start = (
            destination_base_offset_bytes + region_slice.offset_within_rank
        )
        for position_index, position in enumerate(positions):
            source_start = position.remote_block_id * row_bytes
            packed_start = destination_start + position_index * row_bytes
            destination[packed_start : packed_start + row_bytes].copy_(
                source[source_start : source_start + row_bytes]
            )


def launch_packed_chunk_pack(
    destination: torch.Tensor,
    sources: tuple[torch.Tensor, ...],
    source_plan: CanonicalSourceTransferPlan,
    packed_plan: PackedTransferPlan,
    chunk_index: int,
    stream: torch.cuda.Stream,
    *,
    destination_base_offset_bytes: int,
    source_ready_event: torch.cuda.Event | None = None,
) -> PackLaunch:
    """Enqueue one canonical source gather on a caller-owned CUDA stream."""
    if HAS_TRITON is False:
        raise RuntimeError("Triton is required for CUDA packed gather")
    chunk = _validate_pack_tensors(
        destination,
        sources,
        source_plan,
        packed_plan,
        chunk_index,
        destination_base_offset_bytes,
        "cuda",
    )
    if torch.device(stream.device) != destination.device:
        raise ValueError("the caller-supplied stream must match the tensor device")

    keepalive: list[torch.Tensor] = [destination, *sources]
    enqueued_region_count = 0
    start_event = torch.cuda.Event(enable_timing=True)
    try:
        with torch.cuda.stream(stream):
            if source_ready_event is not None:
                stream.wait_event(source_ready_event)
            start_event.record(stream)
            for region_slice in chunk.region_slices:
                source_region = source_plan.regions[region_slice.region_index]
                position_end = region_slice.position_start + region_slice.position_count
                positions = source_region.positions[
                    region_slice.position_start : position_end
                ]
                remote_block_ids = torch.tensor(
                    tuple(position.remote_block_id for position in positions),
                    dtype=torch.int64,
                    device=destination.device,
                )
                keepalive.append(remote_block_ids)
                _coalesced_pack_kernel[(region_slice.position_count,)](
                    sources[region_slice.region_index].reshape(-1),
                    destination,
                    remote_block_ids,
                    destination_base_offset_bytes + region_slice.offset_within_rank,
                    region_slice.position_count,
                    ROW_BYTES=source_region.ownership.row_bytes,
                    BLOCK_BYTES=_KERNEL_BLOCK_BYTES,
                    num_warps=8,
                )
                enqueued_region_count += 1
            completion_event = torch.cuda.Event(enable_timing=True)
            completion_event.record(stream)
    except Exception as error:
        launch_traceback = traceback.format_exc()
        recovery_launch: PackLaunch | None = None
        try:
            with torch.cuda.stream(stream):
                recovery_event = torch.cuda.Event(enable_timing=True)
                recovery_event.record(stream)
            recovery_launch = PackLaunch(
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
        raise PackEnqueueError(
            "packed gather enqueue failed after validation\n"
            + launch_traceback
            + recovery_traceback,
            recovery_launch,
        ) from error

    return PackLaunch(
        start_event=start_event,
        completion_event=completion_event,
        enqueued_region_count=enqueued_region_count,
        _keepalive=tuple(keepalive),
    )


def _validate_pack_tensors(
    destination: torch.Tensor,
    sources: tuple[torch.Tensor, ...],
    source_plan: CanonicalSourceTransferPlan,
    packed_plan: PackedTransferPlan,
    chunk_index: int,
    destination_base_offset_bytes: int,
    device_type: str,
) -> PackedTransferChunk:
    """Validate one source gather before any destination write begins."""
    if type(source_plan) is not CanonicalSourceTransferPlan:
        raise ValueError("source_plan must be a CanonicalSourceTransferPlan")
    validate_packed_transfer_plan(
        packed_plan,
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
    if type(chunk_index) is not int or not 0 <= chunk_index < len(packed_plan.chunks):
        raise ValueError("packed chunk index is outside the plan")
    if (
        type(destination_base_offset_bytes) is not int
        or destination_base_offset_bytes < 0
    ):
        raise ValueError("destination_base_offset_bytes must be non-negative")
    if destination.device.type != device_type:
        raise ValueError(f"destination must be on a {device_type} device")
    if destination.dtype is not torch.uint8 or destination.ndim != 1:
        raise TypeError("destination must be a one-dimensional uint8 tensor")
    if destination.is_contiguous() is False:
        raise ValueError("destination must be contiguous")
    if len(sources) != len(source_plan.regions):
        raise ValueError("sources and canonical regions must have equal lengths")

    chunk = packed_plan.chunks[chunk_index]
    if destination_base_offset_bytes + chunk.rank_stride_bytes > destination.numel():
        raise ValueError("packed chunk exceeds its destination slot")

    for region_index, (source, region) in enumerate(
        zip(sources, source_plan.regions, strict=True)
    ):
        if source.device != destination.device:
            raise ValueError(
                f"source region {region_index} must share the destination device"
            )
        if source.dtype is not torch.uint8 or source.is_contiguous() is False:
            raise TypeError(f"source region {region_index} must be contiguous uint8")
        expected_bytes = region.ownership.source_row_count * region.ownership.row_bytes
        if source.numel() != expected_bytes:
            raise ValueError(
                f"source region {region_index} has {source.numel()} bytes, "
                f"expected exactly {expected_bytes}"
            )
    return chunk
