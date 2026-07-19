# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reproducibly qualify and tune the production coalesced scatter kernel."""

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TypedDict, cast

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
    launch_coalesced_scatter,
)
from vllm.triton_utils import HAS_TRITON

SCHEMA_VERSION = 1
PRODUCTION_ROW_BYTES = 64 * 1024
SOURCE_TP_SIZE = 4
SOURCE_ROW_COUNT = 64_000
KERNEL_BLOCK_WIDTHS = (512, 1024, 2048, 4096)
BASELINE_KERNEL_BLOCK_WIDTH = 1024
BOUNDARY_EXTENSION_WIDTHS = {512: 256, 4096: 8192}
WARMUP_COUNT = 5
SAMPLE_ROUND_COUNT = 21
MEASUREMENT_SEED = 0x37B4C0A1
PER_SHAPE_REJECTION_RATIO = 1.03
NOISE_RETENTION_SPEEDUP = 1.02
ONE_SIDED_MEDIAN_UPPER_ORDER_INDEX = 14
ONE_SIDED_MEDIAN_CONFIDENCE = (
    sum(
        math.comb(SAMPLE_ROUND_COUNT, count)
        for count in range(ONE_SIDED_MEDIAN_UPPER_ORDER_INDEX + 1)
    )
    / 2**SAMPLE_ROUND_COUNT
)
REDZONE_BYTES = 4096
MINIMUM_WORKING_SET_BYTES = 256 * 1024 * 1024
CANARY = 0xD7


@dataclass(frozen=True, slots=True)
class ScatterShape:
    """Define one production-semantic scatter load shape.

    :ivar name: Stable shape identity used in artifacts.
    :ivar group_position_counts: Selected dual-plane positions for all thirteen
        Gemma 4 cache groups.
    """

    name: str
    group_position_counts: tuple[int, ...]


SCATTER_SHAPES = (
    ScatterShape("minimum", (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)),
    ScatterShape(
        "exact_2k",
        (64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 65, 65, 33),
    ),
    ScatterShape(
        "maximum",
        (64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 4096, 4096, 2048),
    ),
)


class SampleRecord(TypedDict):
    """One raw event-fenced scatter timing observation."""

    schema_version: int
    shape: str
    kernel_block_bytes: int
    sample_round: int
    order_within_round: int
    staging_base_offset_bytes: int
    wire_bytes: int
    gpu_duration_ms: float
    wall_duration_ms: float
    gpu_gbps: float
    wall_gbps: float


class ShapeMedian(TypedDict):
    """Median timing and baseline normalization for one shape."""

    median_gpu_duration_ms: float
    median_wall_duration_ms: float
    gpu_duration_ratio_to_1024: float
    wall_duration_ratio_to_1024: float
    gpu_ratio_one_sided_upper: float
    wall_ratio_one_sided_upper: float
    normalized_duration_ratio: float


class CandidateSummary(TypedDict):
    """Deterministic selection evidence for one kernel width."""

    eligible: bool
    rejected_shapes: list[str]
    shapes: dict[str, ShapeMedian]
    geometric_mean_normalized_duration: float
    geometric_mean_speedup: float


class SelectionSummary(TypedDict):
    """Complete deterministic kernel-width selection result."""

    schema_version: int
    baseline_kernel_block_bytes: int
    rejection_ratio: float
    noise_retention_speedup: float
    paired_median_one_sided_confidence: float
    shape_order: list[str]
    candidates: dict[str, CandidateSummary]
    selected_kernel_block_bytes: int
    selection_reason: str
    boundary_extension_required: bool
    boundary_extension_kernel_block_bytes: int | None


class CorrectnessRecord(TypedDict):
    """One full-byte and redzone result for a kernel width and shape."""

    schema_version: int
    shape: str
    kernel_block_bytes: int
    wire_bytes: int
    exact_bytes_equal: bool
    semantic_probes_equal: bool
    redzones_intact: bool


class CommandCapture(TypedDict):
    """Captured command outcome used in environment evidence."""

    argv: list[str]
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True, slots=True)
class InitialArtifactReceipt:
    """Authenticated identity and extension request from an initial campaign.

    :ivar manifest_sha256: Digest named by the sealed terminal marker.
    :ivar selection_sha256: Digest of the replayed initial selection.
    :ivar selected_kernel_block_bytes: Provisional initial boundary winner.
    :ivar extension_kernel_block_bytes: Exact adjacent width requested by the
        initial selection.
    """

    manifest_sha256: str
    selection_sha256: str
    selected_kernel_block_bytes: int
    extension_kernel_block_bytes: int


@dataclass(frozen=True, slots=True)
class _ValidatedArtifact:
    """Internal result of physical artifact validation and selection replay."""

    summary: SelectionSummary
    manifest_sha256: str
    selection_sha256: str
    campaign_kind: str
    lineage: dict[str, object] | None


@dataclass(slots=True)
class _StagingBuffers:
    """Own aligned staging storage and its deterministic working-set offsets."""

    backing: torch.Tensor
    tensor: torch.Tensor
    offsets: tuple[int, ...]


@dataclass(slots=True)
class _DestinationBuffers:
    """Own exact destination views and their aligned guard storage."""

    backings: tuple[torch.Tensor, ...]
    tensors: tuple[torch.Tensor, ...]


def build_shape_plan(shape: ScatterShape) -> CoalescedTransferPlan:
    """Build compact IDs with exact Gemma 4 ownership and transfer volume.

    :param shape: Production-semantic position counts.
    :returns: Canonical rank-major, region-pruned TP4-to-TP1 plan.
    """
    if len(shape.name) == 0:
        raise ValueError("scatter shape name must not be empty")
    if len(shape.group_position_counts) != 13:
        raise ValueError("scatter shapes must describe all thirteen cache groups")
    if any(count < 0 for count in shape.group_position_counts):
        raise ValueError("scatter shape position counts must be non-negative")

    partition_cursors = [0, 0]
    groups: list[GroupTransferRoster] = []
    for group_index, position_count in enumerate(shape.group_position_counts):
        partition_index = 0 if group_index < 12 else 1
        start = partition_cursors[partition_index]
        stop = start + position_count
        groups.append(
            GroupTransferRoster(
                group_index=group_index,
                source_position_start=0,
                destination_plane_count=2,
                local_block_ids=tuple(range(start, stop)),
                remote_block_ids=tuple(range(start, stop)),
            )
        )
        partition_cursors[partition_index] = stop

    regions = tuple(
        RegionOwnership(
            region_index=region_index,
            group_indices=tuple(range(12)) if region_index < 5 else (12,),
            source_row_count=SOURCE_ROW_COUNT,
            destination_row_count=partition_cursors[0 if region_index < 5 else 1] + 1,
            row_bytes=PRODUCTION_ROW_BYTES,
        )
        for region_index in range(10)
    )
    return build_coalesced_transfer_plan(
        source_tp_size=SOURCE_TP_SIZE,
        source_ranks=tuple(range(SOURCE_TP_SIZE)),
        rank_slots=tuple(range(SOURCE_TP_SIZE)),
        groups=tuple(groups),
        regions=regions,
    )


def _validate_kernel_block_widths(kernel_block_widths: tuple[int, ...]) -> None:
    """Validate the closed width matrix used by measurement and replay.

    :param kernel_block_widths: Candidate widths in canonical order.
    :raises ValueError: If the matrix cannot support baseline normalization.
    """
    if (
        len(kernel_block_widths) < 2
        or len(set(kernel_block_widths)) != len(kernel_block_widths)
        or any(width <= 0 for width in kernel_block_widths)
        or tuple(sorted(kernel_block_widths)) != kernel_block_widths
        or BASELINE_KERNEL_BLOCK_WIDTH not in kernel_block_widths
    ):
        raise ValueError(
            "kernel widths must be sorted, unique, positive, and include the baseline"
        )


def extension_kernel_block_widths(
    extension_kernel_block_bytes: int,
) -> tuple[int, ...]:
    """Return the one closed extension matrix admitted by the protocol.

    :param extension_kernel_block_bytes: Adjacent width requested by a sealed
        initial boundary selection.
    :returns: Original matrix plus the requested adjacent width.
    """
    if extension_kernel_block_bytes not in BOUNDARY_EXTENSION_WIDTHS.values():
        raise ValueError("extension width is not an adjacent protocol width")
    return tuple(sorted((*KERNEL_BLOCK_WIDTHS, extension_kernel_block_bytes)))


def measurement_order(
    shape_index: int,
    kernel_block_widths: tuple[int, ...] = KERNEL_BLOCK_WIDTHS,
) -> tuple[tuple[int, ...], ...]:
    """Return the fixed-seed randomized round-robin width order.

    :param shape_index: Canonical index within :data:`SCATTER_SHAPES`.
    :param kernel_block_widths: Complete sorted width matrix for this campaign.
    :returns: One complete width permutation per measurement round.
    """
    if shape_index < 0 or shape_index >= len(SCATTER_SHAPES):
        raise ValueError("shape index is outside the tuning matrix")
    _validate_kernel_block_widths(kernel_block_widths)
    generator = random.Random(MEASUREMENT_SEED + shape_index)
    rounds: list[tuple[int, ...]] = []
    for _ in range(SAMPLE_ROUND_COUNT):
        widths = list(kernel_block_widths)
        generator.shuffle(widths)
        rounds.append(tuple(widths))
    return tuple(rounds)


def _geometric_mean(values: list[float]) -> float:
    if len(values) == 0 or any(value <= 0.0 for value in values):
        raise ValueError("geometric mean requires positive observations")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _paired_median_and_upper(values: list[float]) -> tuple[float, float]:
    """Return a paired median and its exact one-sided confidence bound.

    For 21 independent paired ratios, the fifteenth order statistic is a
    distribution-free 96.08% upper confidence bound for the population median.

    :param values: Positive per-round candidate-to-baseline ratios.
    :returns: Sample median followed by the one-sided upper bound.
    """
    if len(values) != SAMPLE_ROUND_COUNT:
        raise ValueError("paired ratio count differs from the fixed protocol")
    ordered = sorted(values)
    return (
        float(statistics.median(ordered)),
        ordered[ONE_SIDED_MEDIAN_UPPER_ORDER_INDEX],
    )


def select_kernel_width(
    samples: tuple[SampleRecord, ...],
    kernel_block_widths: tuple[int, ...] = KERNEL_BLOCK_WIDTHS,
    *,
    permit_boundary_extension: bool = True,
) -> SelectionSummary:
    """Apply the fixed no-regression and noise-retention selection policy.

    :param samples: Complete raw timing matrix.
    :param kernel_block_widths: Complete sorted candidate matrix.
    :param permit_boundary_extension: Whether a material initial boundary winner
        emits one adjacent extension request.
    :returns: Deterministic per-candidate medians, rejection evidence, and choice.
    """
    _validate_kernel_block_widths(kernel_block_widths)
    shape_names = [shape.name for shape in SCATTER_SHAPES]
    expected_sample_count = (
        len(SCATTER_SHAPES) * len(kernel_block_widths) * SAMPLE_ROUND_COUNT
    )
    if len(samples) != expected_sample_count:
        raise ValueError(
            f"raw matrix has {len(samples)} samples, expected {expected_sample_count}"
        )
    expected_wire_bytes = {
        shape.name: build_shape_plan(shape).staging_size_bytes
        for shape in SCATTER_SHAPES
    }
    shape_indices = {
        shape.name: shape_index for shape_index, shape in enumerate(SCATTER_SHAPES)
    }
    records_by_round: dict[tuple[str, int], list[SampleRecord]] = {}
    for sample in samples:
        shape_name = sample["shape"]
        width = sample["kernel_block_bytes"]
        sample_round = sample["sample_round"]
        if sample["schema_version"] != SCHEMA_VERSION:
            raise ValueError("raw sample schema version differs from the tuner")
        if shape_name not in shape_indices:
            raise ValueError(f"raw sample has unknown shape {shape_name!r}")
        if width not in kernel_block_widths:
            raise ValueError(f"raw sample has unknown kernel width {width}")
        if sample_round < 0 or sample_round >= SAMPLE_ROUND_COUNT:
            raise ValueError("raw sample round is outside the fixed protocol")
        if sample["wire_bytes"] != expected_wire_bytes[shape_name]:
            raise ValueError("raw sample wire bytes differ from canonical geometry")
        if sample["staging_base_offset_bytes"] < 0:
            raise ValueError("raw sample staging offset must be non-negative")
        gpu_ms = sample["gpu_duration_ms"]
        wall_ms = sample["wall_duration_ms"]
        if (
            math.isfinite(gpu_ms) is False
            or math.isfinite(wall_ms) is False
            or gpu_ms <= 0.0
            or wall_ms <= 0.0
        ):
            raise ValueError("scatter timing observations must be finite and positive")
        expected_gpu_gbps = sample["wire_bytes"] / 1_000_000 / gpu_ms
        expected_wall_gbps = sample["wire_bytes"] / 1_000_000 / wall_ms
        if math.isclose(sample["gpu_gbps"], expected_gpu_gbps) is False:
            raise ValueError("raw GPU throughput differs from bytes and duration")
        if math.isclose(sample["wall_gbps"], expected_wall_gbps) is False:
            raise ValueError("raw wall throughput differs from bytes and duration")
        records_by_round.setdefault((shape_name, sample_round), []).append(sample)

    for shape_name, shape_index in shape_indices.items():
        expected_orders = measurement_order(shape_index, kernel_block_widths)
        for sample_round, expected_order in enumerate(expected_orders):
            round_records = records_by_round.get((shape_name, sample_round), [])
            if len(round_records) != len(kernel_block_widths):
                raise ValueError(
                    f"{shape_name} round {sample_round} is not a complete width round"
                )
            observed_order = tuple(
                record["kernel_block_bytes"]
                for record in sorted(
                    round_records,
                    key=lambda record: record["order_within_round"],
                )
            )
            observed_indices = sorted(
                record["order_within_round"] for record in round_records
            )
            if observed_indices != list(range(len(kernel_block_widths))):
                raise ValueError("raw round order indices are not canonical")
            if observed_order != expected_order:
                raise ValueError("raw round differs from the fixed-seed schedule")

    medians: dict[int, dict[str, tuple[float, float]]] = {}
    for width in kernel_block_widths:
        medians[width] = {}
        for shape_name in shape_names:
            selected = [
                sample
                for sample in samples
                if sample["kernel_block_bytes"] == width
                and sample["shape"] == shape_name
            ]
            if len(selected) != SAMPLE_ROUND_COUNT:
                raise ValueError(
                    f"{shape_name}/{width} has {len(selected)} samples, "
                    f"expected {SAMPLE_ROUND_COUNT}"
                )
            sample_rounds = sorted(sample["sample_round"] for sample in selected)
            if sample_rounds != list(range(SAMPLE_ROUND_COUNT)):
                raise ValueError(f"{shape_name}/{width} sample rounds are incomplete")
            gpu_values = [sample["gpu_duration_ms"] for sample in selected]
            wall_values = [sample["wall_duration_ms"] for sample in selected]
            medians[width][shape_name] = (
                float(statistics.median(gpu_values)),
                float(statistics.median(wall_values)),
            )

    candidates: dict[str, CandidateSummary] = {}
    for width in kernel_block_widths:
        rejected_shapes: list[str] = []
        shape_summaries: dict[str, ShapeMedian] = {}
        normalized_ratios: list[float] = []
        for shape_name in shape_names:
            gpu_median, wall_median = medians[width][shape_name]
            candidate_samples = sorted(
                (
                    sample
                    for sample in samples
                    if sample["kernel_block_bytes"] == width
                    and sample["shape"] == shape_name
                ),
                key=lambda sample: sample["sample_round"],
            )
            baseline_samples = sorted(
                (
                    sample
                    for sample in samples
                    if sample["kernel_block_bytes"] == BASELINE_KERNEL_BLOCK_WIDTH
                    and sample["shape"] == shape_name
                ),
                key=lambda sample: sample["sample_round"],
            )
            gpu_ratio, gpu_upper = _paired_median_and_upper(
                [
                    candidate["gpu_duration_ms"] / baseline["gpu_duration_ms"]
                    for candidate, baseline in zip(
                        candidate_samples,
                        baseline_samples,
                        strict=True,
                    )
                ]
            )
            wall_ratio, wall_upper = _paired_median_and_upper(
                [
                    candidate["wall_duration_ms"] / baseline["wall_duration_ms"]
                    for candidate, baseline in zip(
                        candidate_samples,
                        baseline_samples,
                        strict=True,
                    )
                ]
            )
            normalized_ratio = math.sqrt(gpu_ratio * wall_ratio)
            if (
                gpu_upper > PER_SHAPE_REJECTION_RATIO
                or wall_upper > PER_SHAPE_REJECTION_RATIO
            ):
                rejected_shapes.append(shape_name)
            normalized_ratios.append(normalized_ratio)
            shape_summaries[shape_name] = {
                "median_gpu_duration_ms": gpu_median,
                "median_wall_duration_ms": wall_median,
                "gpu_duration_ratio_to_1024": gpu_ratio,
                "wall_duration_ratio_to_1024": wall_ratio,
                "gpu_ratio_one_sided_upper": gpu_upper,
                "wall_ratio_one_sided_upper": wall_upper,
                "normalized_duration_ratio": normalized_ratio,
            }
        overall_ratio = _geometric_mean(normalized_ratios)
        candidates[str(width)] = {
            "eligible": len(rejected_shapes) == 0,
            "rejected_shapes": rejected_shapes,
            "shapes": shape_summaries,
            "geometric_mean_normalized_duration": overall_ratio,
            "geometric_mean_speedup": 1.0 / overall_ratio,
        }

    eligible_widths = [
        width for width in kernel_block_widths if candidates[str(width)]["eligible"]
    ]
    if BASELINE_KERNEL_BLOCK_WIDTH not in eligible_widths:
        raise AssertionError("the normalized baseline cannot reject itself")
    best_width = min(
        eligible_widths,
        key=lambda width: (
            candidates[str(width)]["geometric_mean_normalized_duration"],
            0 if width == BASELINE_KERNEL_BLOCK_WIDTH else 1,
            width,
        ),
    )
    best_speedup = candidates[str(best_width)]["geometric_mean_speedup"]
    if (
        best_width != BASELINE_KERNEL_BLOCK_WIDTH
        and best_speedup <= NOISE_RETENTION_SPEEDUP
    ):
        selected_width = BASELINE_KERNEL_BLOCK_WIDTH
        reason = (
            f"retained {BASELINE_KERNEL_BLOCK_WIDTH}: best eligible width "
            f"{best_width} improved the geometric mean by only "
            f"{(best_speedup - 1.0) * 100:.6f}%, within the 2% noise band"
        )
    else:
        selected_width = best_width
        reason = (
            f"selected {best_width}: eligible minimum equal-weight geometric-mean "
            "duration across GPU and wall time for all shapes"
        )

    boundary_extension_width = BOUNDARY_EXTENSION_WIDTHS.get(selected_width)
    boundary_extension_required = (
        permit_boundary_extension
        and kernel_block_widths == KERNEL_BLOCK_WIDTHS
        and boundary_extension_width is not None
        and candidates[str(selected_width)]["geometric_mean_speedup"]
        > NOISE_RETENTION_SPEEDUP
    )
    if boundary_extension_required is False:
        boundary_extension_width = None
    if boundary_extension_required:
        reason += (
            f"; provisional boundary winner requires {boundary_extension_width}-byte "
            "extension before production selection"
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "baseline_kernel_block_bytes": BASELINE_KERNEL_BLOCK_WIDTH,
        "rejection_ratio": PER_SHAPE_REJECTION_RATIO,
        "noise_retention_speedup": NOISE_RETENTION_SPEEDUP,
        "paired_median_one_sided_confidence": ONE_SIDED_MEDIAN_CONFIDENCE,
        "shape_order": shape_names,
        "candidates": candidates,
        "selected_kernel_block_bytes": selected_width,
        "selection_reason": reason,
        "boundary_extension_required": boundary_extension_required,
        "boundary_extension_kernel_block_bytes": boundary_extension_width,
    }


def _destination_bytes(plan: CoalescedTransferPlan) -> int:
    return sum(
        region.ownership.destination_row_count
        * SOURCE_TP_SIZE
        * region.ownership.row_bytes
        for region in plan.regions
    )


def _working_set_window_count(shape: ScatterShape, plan: CoalescedTransferPlan) -> int:
    if shape.name != "minimum":
        return 1
    return max(
        1,
        (MINIMUM_WORKING_SET_BYTES + plan.staging_size_bytes - 1)
        // plan.staging_size_bytes,
    )


def _allocate_staging(
    shape: ScatterShape,
    plan: CoalescedTransferPlan,
    device: torch.device,
) -> _StagingBuffers:
    window_count = _working_set_window_count(shape, plan)
    capacity = window_count * plan.staging_size_bytes
    backing = torch.empty(
        capacity + 2 * REDZONE_BYTES,
        dtype=torch.uint8,
        device=device,
    )
    tensor = backing[REDZONE_BYTES : REDZONE_BYTES + capacity]
    backing[:REDZONE_BYTES].fill_(CANARY)
    backing[-REDZONE_BYTES:].fill_(CANARY)

    rows_per_window = plan.staging_size_bytes // PRODUCTION_ROW_BYTES
    rows = tensor.view(window_count, rows_per_window, PRODUCTION_ROW_BYTES)
    row_indices = torch.arange(rows_per_window, dtype=torch.int64, device=device)
    key_tags = ((row_indices % 127) + 1).to(torch.uint8)
    value_tags = ((row_indices % 87) + 128).to(torch.uint8)
    half = PRODUCTION_ROW_BYTES // 2
    rows[:, :, :half].copy_(
        key_tags.view(1, rows_per_window, 1).expand(window_count, -1, half)
    )
    rows[:, :, half:].copy_(
        value_tags.view(1, rows_per_window, 1).expand(window_count, -1, half)
    )
    return _StagingBuffers(
        backing=backing,
        tensor=tensor,
        offsets=tuple(
            window_index * plan.staging_size_bytes
            for window_index in range(window_count)
        ),
    )


def _allocate_destinations(
    plan: CoalescedTransferPlan,
    device: torch.device,
) -> _DestinationBuffers:
    backings: list[torch.Tensor] = []
    tensors: list[torch.Tensor] = []
    for region in plan.regions:
        size = (
            region.ownership.destination_row_count
            * SOURCE_TP_SIZE
            * region.ownership.row_bytes
        )
        backing = torch.empty(
            size + 2 * REDZONE_BYTES,
            dtype=torch.uint8,
            device=device,
        )
        backings.append(backing)
        tensors.append(backing[REDZONE_BYTES : REDZONE_BYTES + size])
    buffers = _DestinationBuffers(tuple(backings), tuple(tensors))
    _reset_destinations(buffers)
    return buffers


def _reset_destinations(buffers: _DestinationBuffers) -> None:
    for backing, tensor in zip(buffers.backings, buffers.tensors, strict=True):
        backing[:REDZONE_BYTES].fill_(CANARY)
        backing[-REDZONE_BYTES:].fill_(CANARY)
        tensor.fill_(CANARY)


def _redzones_intact(
    staging: _StagingBuffers,
    destinations: _DestinationBuffers,
) -> bool:
    if not torch.all(staging.backing[:REDZONE_BYTES] == CANARY).item():
        return False
    if not torch.all(staging.backing[-REDZONE_BYTES:] == CANARY).item():
        return False
    for backing in destinations.backings:
        if not torch.all(backing[:REDZONE_BYTES] == CANARY).item():
            return False
        if not torch.all(backing[-REDZONE_BYTES:] == CANARY).item():
            return False
    return True


def _semantic_probes_equal(
    plan: CoalescedTransferPlan,
    destinations: _DestinationBuffers,
) -> bool:
    device = destinations.tensors[0].device
    half = PRODUCTION_ROW_BYTES // 2
    probes = torch.tensor((0, half // 2, half - 1), dtype=torch.long, device=device)
    rows_per_rank = plan.rank_stride_bytes // PRODUCTION_ROW_BYTES
    for region, destination in zip(plan.regions, destinations.tensors, strict=True):
        destination_view = destination.view(
            region.ownership.destination_row_count,
            2,
            SOURCE_TP_SIZE,
            half,
        )
        if not torch.all(destination_view[-1] == CANARY).item():
            return False
        if len(region.positions) == 0:
            continue
        local_ids = torch.tensor(
            tuple(position.local_block_id for position in region.positions),
            dtype=torch.long,
            device=device,
        )
        region_row_offset = region.offset_within_rank // PRODUCTION_ROW_BYTES
        position_indices = torch.arange(
            len(region.positions),
            dtype=torch.int64,
            device=device,
        )
        for rank_index, rank_slot in enumerate(plan.rank_slots):
            global_rows = (
                rank_index * rows_per_rank + region_row_offset + position_indices
            )
            expected_key = ((global_rows % 127) + 1).to(torch.uint8)
            expected_value = ((global_rows % 87) + 128).to(torch.uint8)
            actual_key = destination_view[local_ids, 0, rank_slot][:, probes]
            actual_value = destination_view[local_ids, 1, rank_slot][:, probes]
            if not torch.equal(actual_key, expected_key.view(-1, 1).expand(-1, 3)):
                return False
            if not torch.equal(
                actual_value,
                expected_value.view(-1, 1).expand(-1, 3),
            ):
                return False
    return True


def _launch_once(
    *,
    staging: _StagingBuffers,
    destinations: _DestinationBuffers,
    plan: CoalescedTransferPlan,
    stream: torch.cuda.Stream,
    staging_base_offset_bytes: int,
) -> tuple[float, float]:
    stream.wait_stream(torch.cuda.current_stream(staging.tensor.device))
    wall_start = time.perf_counter()
    launch = launch_coalesced_scatter(
        staging.tensor,
        destinations.tensors,
        plan,
        stream,
        staging_base_offset_bytes=staging_base_offset_bytes,
    )
    launch.completion_event.synchronize()
    wall_duration_ms = (time.perf_counter() - wall_start) * 1000
    gpu_duration_ms = launch.gpu_duration_ms()
    if gpu_duration_ms <= 0.0 or wall_duration_ms <= 0.0:
        raise RuntimeError("scatter timings must be positive")
    return gpu_duration_ms, wall_duration_ms


def _run_correctness(
    *,
    shape: ScatterShape,
    plan: CoalescedTransferPlan,
    staging: _StagingBuffers,
    candidate: _DestinationBuffers,
    device: torch.device,
    kernel_block_widths: tuple[int, ...],
) -> tuple[CorrectnessRecord, ...]:
    reference = _allocate_destinations(plan, device)
    stream = torch.cuda.Stream(device=device)
    original_width = scatter_module._KERNEL_BLOCK_BYTES
    try:
        scatter_module._KERNEL_BLOCK_BYTES = BASELINE_KERNEL_BLOCK_WIDTH
        _launch_once(
            staging=staging,
            destinations=reference,
            plan=plan,
            stream=stream,
            staging_base_offset_bytes=staging.offsets[0],
        )
        reference_semantic = _semantic_probes_equal(plan, reference)
        reference_redzones = _redzones_intact(staging, reference)
        if reference_semantic is False or reference_redzones is False:
            raise RuntimeError(f"baseline correctness failed for {shape.name}")

        records: list[CorrectnessRecord] = []
        for width in kernel_block_widths:
            _reset_destinations(candidate)
            scatter_module._KERNEL_BLOCK_BYTES = width
            _launch_once(
                staging=staging,
                destinations=candidate,
                plan=plan,
                stream=stream,
                staging_base_offset_bytes=staging.offsets[0],
            )
            exact_bytes_equal = all(
                torch.equal(actual, expected)
                for actual, expected in zip(
                    candidate.tensors,
                    reference.tensors,
                    strict=True,
                )
            )
            semantic_probes_equal = _semantic_probes_equal(plan, candidate)
            redzones_intact = _redzones_intact(staging, candidate)
            record: CorrectnessRecord = {
                "schema_version": SCHEMA_VERSION,
                "shape": shape.name,
                "kernel_block_bytes": width,
                "wire_bytes": plan.staging_size_bytes,
                "exact_bytes_equal": exact_bytes_equal,
                "semantic_probes_equal": semantic_probes_equal,
                "redzones_intact": redzones_intact,
            }
            records.append(record)
            if (
                exact_bytes_equal is False
                or semantic_probes_equal is False
                or redzones_intact is False
            ):
                raise RuntimeError(
                    f"scatter correctness failed for {shape.name}/{width}: {record}"
                )
        return tuple(records)
    finally:
        scatter_module._KERNEL_BLOCK_BYTES = original_width


def _benchmark_shape(
    *,
    shape_index: int,
    shape: ScatterShape,
    plan: CoalescedTransferPlan,
    device: torch.device,
    kernel_block_widths: tuple[int, ...],
) -> tuple[tuple[CorrectnessRecord, ...], tuple[SampleRecord, ...]]:
    staging = _allocate_staging(shape, plan, device)
    candidate = _allocate_destinations(plan, device)
    correctness = _run_correctness(
        shape=shape,
        plan=plan,
        staging=staging,
        candidate=candidate,
        device=device,
        kernel_block_widths=kernel_block_widths,
    )
    gc.collect()
    torch.cuda.empty_cache()

    stream = torch.cuda.Stream(device=device)
    original_width = scatter_module._KERNEL_BLOCK_BYTES
    try:
        for width_index, width in enumerate(kernel_block_widths):
            scatter_module._KERNEL_BLOCK_BYTES = width
            for warmup_index in range(WARMUP_COUNT):
                offset_index = (
                    warmup_index * len(kernel_block_widths) + width_index
                ) % len(staging.offsets)
                _launch_once(
                    staging=staging,
                    destinations=candidate,
                    plan=plan,
                    stream=stream,
                    staging_base_offset_bytes=staging.offsets[offset_index],
                )

        records: list[SampleRecord] = []
        width_indices = {
            width: width_index
            for width_index, width in enumerate(kernel_block_widths)
        }
        warmup_launch_count = WARMUP_COUNT * len(kernel_block_widths)
        for sample_round, order in enumerate(
            measurement_order(shape_index, kernel_block_widths)
        ):
            for order_index, width in enumerate(order):
                scatter_module._KERNEL_BLOCK_BYTES = width
                offset_index = (
                    warmup_launch_count
                    + sample_round * len(kernel_block_widths)
                    + width_indices[width]
                ) % len(staging.offsets)
                staging_offset = staging.offsets[offset_index]
                gpu_ms, wall_ms = _launch_once(
                    staging=staging,
                    destinations=candidate,
                    plan=plan,
                    stream=stream,
                    staging_base_offset_bytes=staging_offset,
                )
                wire_gigabytes = plan.staging_size_bytes / 1_000_000_000
                records.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "shape": shape.name,
                        "kernel_block_bytes": width,
                        "sample_round": sample_round,
                        "order_within_round": order_index,
                        "staging_base_offset_bytes": staging_offset,
                        "wire_bytes": plan.staging_size_bytes,
                        "gpu_duration_ms": gpu_ms,
                        "wall_duration_ms": wall_ms,
                        "gpu_gbps": wire_gigabytes / (gpu_ms / 1000),
                        "wall_gbps": wire_gigabytes / (wall_ms / 1000),
                    }
                )
        if _redzones_intact(staging, candidate) is False:
            raise RuntimeError(f"scatter redzone changed during {shape.name} timing")
        return correctness, tuple(records)
    finally:
        scatter_module._KERNEL_BLOCK_BYTES = original_width


def _capture_command(argv: list[str]) -> CommandCapture:
    try:
        result = subprocess.run(argv, check=False, capture_output=True, text=True)
    except FileNotFoundError as error:
        return {
            "argv": argv,
            "returncode": 127,
            "stdout": "",
            "stderr": str(error),
        }
    return {
        "argv": argv,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            chunk = source.read(1024 * 1024)
            if len(chunk) == 0:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _environment(
    plans: tuple[CoalescedTransferPlan, ...],
    device: torch.device,
    kernel_block_widths: tuple[int, ...],
    lineage: dict[str, object] | None,
) -> dict[str, object]:
    properties = torch.cuda.get_device_properties(device)
    scatter_path = Path(scatter_module.__file__).resolve()
    tool_path = Path(__file__).resolve()
    return {
        "schema_version": SCHEMA_VERSION,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
        "cwd": str(Path.cwd()),
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": torch.version.cuda,
        "triton_version": _package_version("triton"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR"),
        "cuda_module_loading": os.environ.get("CUDA_MODULE_LOADING"),
        "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "device": {
            "logical_index": device.index,
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "multiprocessor_count": properties.multi_processor_count,
            "compute_capability": [properties.major, properties.minor],
        },
        "protocol": {
            "campaign_kind": "initial" if lineage is None else "boundary_extension",
            "kernel_block_widths": list(kernel_block_widths),
            "baseline_kernel_block_width": BASELINE_KERNEL_BLOCK_WIDTH,
            "num_warps": 8,
            "compile_correctness_launches_per_width_per_shape": 1,
            "warmups_per_width_per_shape": WARMUP_COUNT,
            "sample_rounds_per_width_per_shape": SAMPLE_ROUND_COUNT,
            "measurement_seed": MEASUREMENT_SEED,
            "minimum_working_set_bytes": MINIMUM_WORKING_SET_BYTES,
            "rejection_ratio": PER_SHAPE_REJECTION_RATIO,
            "noise_retention_speedup": NOISE_RETENTION_SPEEDUP,
            "paired_median_one_sided_upper_order_index": (
                ONE_SIDED_MEDIAN_UPPER_ORDER_INDEX
            ),
            "paired_median_one_sided_confidence": ONE_SIDED_MEDIAN_CONFIDENCE,
            "lineage": lineage,
        },
        "shapes": [
            {
                "name": shape.name,
                "group_position_counts": list(shape.group_position_counts),
                "plan_digest": plan.digest,
                "rank_stride_bytes": plan.rank_stride_bytes,
                "wire_bytes": plan.staging_size_bytes,
                "region_position_count": sum(
                    len(region.positions) for region in plan.regions
                ),
                "kernel_program_count": SOURCE_TP_SIZE
                * sum(len(region.positions) for region in plan.regions),
                "nonempty_region_launch_count": sum(
                    1 for region in plan.regions if len(region.positions) > 0
                ),
                "destination_bytes": _destination_bytes(plan),
            }
            for shape, plan in zip(SCATTER_SHAPES, plans, strict=True)
        ],
        "source_files": {
            str(scatter_path): _sha256(scatter_path),
            str(tool_path): _sha256(tool_path),
        },
        "git_head": _capture_command(["git", "rev-parse", "HEAD"]),
        "git_status": _capture_command(["git", "status", "--short"]),
        "nvidia_smi_gpu": _capture_command(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.total,memory.free,temperature.gpu,"
                "clocks.sm,clocks.mem,power.draw",
                "--format=csv,noheader,nounits",
            ]
        ),
        "nvidia_smi_processes": _capture_command(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ]
        ),
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _append_jsonl(path: Path, values: tuple[object, ...]) -> None:
    with path.open("a") as destination:
        for value in values:
            destination.write(json.dumps(value, sort_keys=True) + "\n")
        destination.flush()


def _read_json_object(path: Path, context: str) -> dict[str, object]:
    """Read one physically regular JSON object.

    :param path: Evidence path to read.
    :param context: Stable diagnostic identity.
    :returns: Parsed JSON object.
    :raises ValueError: If the path or payload is not canonical evidence.
    """
    if path.is_symlink() or path.is_file() is False:
        raise ValueError(f"{context} is not a physical regular file")
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{context} is not valid JSON: {error}") from error
    if not isinstance(value, dict) or any(
        not isinstance(key, str) for key in value
    ):
        raise ValueError(f"{context} is not a JSON object")
    return value


def _read_jsonl_objects(path: Path, context: str) -> tuple[dict[str, object], ...]:
    """Read a non-empty physical JSON-lines evidence file.

    :param path: Evidence path to read.
    :param context: Stable diagnostic identity.
    :returns: Parsed records in physical order.
    :raises ValueError: If any line is empty or is not a JSON object.
    """
    if path.is_symlink() or path.is_file() is False:
        raise ValueError(f"{context} is not a physical regular file")
    try:
        lines = path.read_text().splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"{context} is not readable text: {error}") from error
    if len(lines) == 0 or any(len(line) == 0 for line in lines):
        raise ValueError(f"{context} must contain non-empty JSON lines")
    records: list[dict[str, object]] = []
    for line_index, line in enumerate(lines):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{context} line {line_index + 1} is not valid JSON: {error}"
            ) from error
        if not isinstance(value, dict) or any(
            not isinstance(key, str) for key in value
        ):
            raise ValueError(f"{context} line {line_index + 1} is not an object")
        records.append(value)
    return tuple(records)


def _require_sha256(value: object, context: str) -> str:
    """Validate one lowercase SHA-256 identity.

    :param value: Parsed identity.
    :param context: Stable diagnostic identity.
    :returns: Validated digest.
    :raises ValueError: If the identity is malformed.
    """
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} is not a lowercase SHA-256 digest")
    return value


def _terminal_manifest_sha256(path: Path, marker_name: str) -> str:
    """Read the exact manifest identity from one terminal marker.

    :param path: Marker path.
    :param marker_name: Expected marker filename.
    :returns: Manifest digest named by the marker.
    :raises ValueError: If the marker is not canonical.
    """
    if path.is_symlink() or path.is_file() is False:
        raise ValueError(f"{marker_name} is not a physical regular file")
    try:
        payload = path.read_text()
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"{marker_name} is not readable text: {error}") from error
    prefix = "manifest_sha256="
    if payload.startswith(prefix) is False or payload.endswith("\n") is False:
        raise ValueError(f"{marker_name} is malformed")
    if payload.count("\n") != 1:
        raise ValueError(f"{marker_name} contains unexpected fields")
    return _require_sha256(payload[len(prefix) : -1], marker_name)


def _validate_manifest(artifact_directory: Path) -> tuple[str, str]:
    """Authenticate every physical file sealed by an artifact manifest.

    :param artifact_directory: Campaign evidence directory.
    :returns: Terminal marker name and manifest digest.
    :raises ValueError: If the directory is unsealed, mutable, or inconsistent.
    """
    if artifact_directory.is_symlink() or artifact_directory.is_dir() is False:
        raise ValueError("scatter tuning artifact is not a physical directory")
    terminal_markers = tuple(
        marker
        for marker in ("COMPLETE", "EXTENSION_REQUIRED")
        if (artifact_directory / marker).exists()
    )
    if len(terminal_markers) != 1:
        raise ValueError("scatter tuning artifact has no unique terminal marker")
    if any(
        (artifact_directory / marker).exists()
        for marker in ("RUNNING", "FAILED")
    ):
        raise ValueError("scatter tuning artifact contains a non-terminal marker")

    marker_name = terminal_markers[0]
    manifest_sha256 = _terminal_manifest_sha256(
        artifact_directory / marker_name,
        marker_name,
    )
    manifest_path = artifact_directory / "manifest.json"
    if _sha256(manifest_path) != manifest_sha256:
        raise ValueError("scatter tuning manifest differs from its terminal marker")
    manifest = _read_json_object(manifest_path, "scatter tuning manifest")
    if set(manifest) != {"schema_version", "files"}:
        raise ValueError("scatter tuning manifest fields differ from the protocol")
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("scatter tuning manifest schema differs from the tuner")
    files = manifest["files"]
    if not isinstance(files, dict) or any(
        not isinstance(name, str) for name in files
    ):
        raise ValueError("scatter tuning manifest files is not an object")
    evidence_files = {
        "environment.json",
        "environment-final.json",
        "correctness.jsonl",
        "samples.jsonl",
        "selection.json",
    }
    if set(files) != evidence_files:
        raise ValueError("scatter tuning evidence file roster differs")
    expected_directory_entries = evidence_files | {"manifest.json", marker_name}
    if {entry.name for entry in artifact_directory.iterdir()} != (
        expected_directory_entries
    ):
        raise ValueError("scatter tuning artifact contains an unsealed path")
    for name in sorted(evidence_files):
        digest = _require_sha256(files[name], f"scatter tuning {name} digest")
        path = artifact_directory / name
        if path.is_symlink() or path.is_file() is False:
            raise ValueError(f"scatter tuning {name} is not a physical regular file")
        if _sha256(path) != digest:
            raise ValueError(f"scatter tuning {name} differs from its manifest")
    return marker_name, manifest_sha256


def _validate_correctness_records(
    records: tuple[dict[str, object], ...],
    kernel_block_widths: tuple[int, ...],
) -> None:
    """Validate the full width-by-shape correctness and redzone matrix.

    :param records: Parsed correctness observations.
    :param kernel_block_widths: Exact candidate matrix.
    :raises ValueError: If any expected proof is absent or false.
    """
    expected_keys = {
        "schema_version",
        "shape",
        "kernel_block_bytes",
        "wire_bytes",
        "exact_bytes_equal",
        "semantic_probes_equal",
        "redzones_intact",
    }
    expected_wire_bytes = {
        shape.name: build_shape_plan(shape).staging_size_bytes
        for shape in SCATTER_SHAPES
    }
    observed: set[tuple[str, int]] = set()
    for record in records:
        if set(record) != expected_keys:
            raise ValueError("scatter correctness record fields differ")
        shape = record["shape"]
        width = record["kernel_block_bytes"]
        if not isinstance(shape, str) or shape not in expected_wire_bytes:
            raise ValueError("scatter correctness record has an unknown shape")
        if type(width) is not int or width not in kernel_block_widths:
            raise ValueError("scatter correctness record has an unknown width")
        identity = (shape, width)
        if identity in observed:
            raise ValueError("scatter correctness matrix has a duplicate record")
        observed.add(identity)
        if (
            record["schema_version"] != SCHEMA_VERSION
            or record["wire_bytes"] != expected_wire_bytes[shape]
            or record["exact_bytes_equal"] is not True
            or record["semantic_probes_equal"] is not True
            or record["redzones_intact"] is not True
        ):
            raise ValueError("scatter correctness proof is false or inconsistent")
    expected = {
        (shape.name, width)
        for shape in SCATTER_SHAPES
        for width in kernel_block_widths
    }
    if observed != expected:
        raise ValueError("scatter correctness matrix is incomplete")


def _validated_artifact(artifact_directory: Path) -> _ValidatedArtifact:
    """Authenticate one campaign and deterministically replay its selection.

    :param artifact_directory: Sealed initial or final artifact directory.
    :returns: Validated selection, identities, and lineage.
    :raises ValueError: If any evidence differs from the closed protocol.
    """
    marker_name, manifest_sha256 = _validate_manifest(artifact_directory)
    environment = _read_json_object(
        artifact_directory / "environment.json",
        "scatter tuning environment",
    )
    if environment.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("scatter tuning environment schema differs")
    protocol = environment.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("scatter tuning environment protocol is not an object")
    raw_widths = protocol.get("kernel_block_widths")
    if not isinstance(raw_widths, list) or any(
        type(width) is not int for width in raw_widths
    ):
        raise ValueError("scatter tuning environment widths are malformed")
    kernel_block_widths = tuple(raw_widths)
    _validate_kernel_block_widths(kernel_block_widths)
    campaign_kind_value = protocol.get("campaign_kind", "initial")
    if not isinstance(campaign_kind_value, str):
        raise ValueError("scatter tuning campaign kind is malformed")
    campaign_kind = campaign_kind_value
    lineage_value = protocol.get("lineage")
    if lineage_value is not None and not isinstance(lineage_value, dict):
        raise ValueError("scatter tuning lineage is not an object")
    lineage = lineage_value

    if campaign_kind == "initial":
        if kernel_block_widths != KERNEL_BLOCK_WIDTHS or lineage is not None:
            raise ValueError("initial scatter tuning protocol differs")
        permit_boundary_extension = True
    elif campaign_kind == "boundary_extension":
        if not isinstance(lineage, dict):
            raise ValueError("boundary extension has no authenticated lineage")
        expected_lineage_keys = {
            "initial_manifest_sha256",
            "initial_selection_sha256",
            "initial_selected_kernel_block_bytes",
            "extension_kernel_block_bytes",
        }
        if set(lineage) != expected_lineage_keys:
            raise ValueError("boundary extension lineage fields differ")
        _require_sha256(
            lineage["initial_manifest_sha256"],
            "initial scatter tuning manifest identity",
        )
        _require_sha256(
            lineage["initial_selection_sha256"],
            "initial scatter tuning selection identity",
        )
        extension_width = lineage["extension_kernel_block_bytes"]
        if type(extension_width) is not int:
            raise ValueError("boundary extension width is malformed")
        if kernel_block_widths != extension_kernel_block_widths(extension_width):
            raise ValueError("boundary extension matrix differs from its request")
        selected_width = lineage["initial_selected_kernel_block_bytes"]
        if (
            type(selected_width) is not int
            or BOUNDARY_EXTENSION_WIDTHS.get(selected_width) != extension_width
        ):
            raise ValueError("boundary extension lineage is not adjacent")
        permit_boundary_extension = False
    else:
        raise ValueError("scatter tuning campaign kind is unknown")

    sample_objects = _read_jsonl_objects(
        artifact_directory / "samples.jsonl",
        "scatter tuning samples",
    )
    expected_sample_keys = set(SampleRecord.__required_keys__)
    for record in sample_objects:
        if set(record) != expected_sample_keys:
            raise ValueError("scatter timing sample fields differ")
    samples = cast(tuple[SampleRecord, ...], sample_objects)
    replayed = select_kernel_width(
        samples,
        kernel_block_widths,
        permit_boundary_extension=permit_boundary_extension,
    )
    recorded = _read_json_object(
        artifact_directory / "selection.json",
        "scatter tuning selection",
    )
    if recorded != replayed:
        raise ValueError("scatter tuning selection differs from deterministic replay")

    expected_marker = (
        "EXTENSION_REQUIRED"
        if replayed["boundary_extension_required"]
        else "COMPLETE"
    )
    if campaign_kind == "boundary_extension":
        expected_marker = "COMPLETE"
    if marker_name != expected_marker:
        raise ValueError("scatter tuning terminal marker differs from its selection")
    _validate_correctness_records(
        _read_jsonl_objects(
            artifact_directory / "correctness.jsonl",
            "scatter tuning correctness",
        ),
        kernel_block_widths,
    )
    environment_final = _read_json_object(
        artifact_directory / "environment-final.json",
        "scatter tuning final environment",
    )
    if environment_final.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("scatter tuning final environment schema differs")
    return _ValidatedArtifact(
        summary=replayed,
        manifest_sha256=manifest_sha256,
        selection_sha256=_sha256(artifact_directory / "selection.json"),
        campaign_kind=campaign_kind,
        lineage=lineage,
    )


def validate_campaign_artifact(
    artifact_directory: Path,
    initial_artifact_directory: Path | None = None,
) -> SelectionSummary:
    """Validate and replay a sealed scatter tuning campaign.

    :param artifact_directory: Sealed campaign artifact.
    :param initial_artifact_directory: Initial artifact required when validating
        final extension lineage across both physical artifacts.
    :returns: Deterministically replayed selection.
    :raises ValueError: If evidence or optional cross-artifact lineage differs.
    """
    validated = _validated_artifact(artifact_directory)
    if initial_artifact_directory is None:
        return validated.summary
    if validated.campaign_kind != "boundary_extension":
        raise ValueError("initial lineage was supplied for a non-extension campaign")
    receipt = authenticate_initial_extension_artifact(initial_artifact_directory)
    expected_lineage: dict[str, object] = {
        "initial_manifest_sha256": receipt.manifest_sha256,
        "initial_selection_sha256": receipt.selection_sha256,
        "initial_selected_kernel_block_bytes": (
            receipt.selected_kernel_block_bytes
        ),
        "extension_kernel_block_bytes": receipt.extension_kernel_block_bytes,
    }
    if validated.lineage != expected_lineage:
        raise ValueError("boundary extension lineage differs from its initial artifact")
    return validated.summary


def authenticate_initial_extension_artifact(
    artifact_directory: Path,
) -> InitialArtifactReceipt:
    """Authenticate the single adjacent request from an initial campaign.

    :param artifact_directory: Sealed initial ``EXTENSION_REQUIRED`` artifact.
    :returns: Cryptographic lineage and exact requested extension width.
    :raises ValueError: If the artifact is final, invalid, or not an initial run.
    """
    validated = _validated_artifact(artifact_directory)
    summary = validated.summary
    extension_width = summary["boundary_extension_kernel_block_bytes"]
    if (
        validated.campaign_kind != "initial"
        or summary["boundary_extension_required"] is False
        or extension_width is None
    ):
        raise ValueError("initial artifact does not request a boundary extension")
    return InitialArtifactReceipt(
        manifest_sha256=validated.manifest_sha256,
        selection_sha256=validated.selection_sha256,
        selected_kernel_block_bytes=summary["selected_kernel_block_bytes"],
        extension_kernel_block_bytes=extension_width,
    )


def _run_campaign(
    artifact_directory: Path,
    kernel_block_widths: tuple[int, ...],
    lineage: dict[str, object] | None,
) -> SelectionSummary:
    """Execute one immutable local-GPU tuning matrix.

    :param artifact_directory: New directory that will own all raw evidence.
    :param kernel_block_widths: Closed candidate matrix.
    :param lineage: Authenticated initial identity for a final extension.
    :returns: Deterministic selection summary.
    """
    if artifact_directory.exists():
        raise FileExistsError(
            f"artifact directory already exists: {artifact_directory}"
        )
    if torch.cuda.is_available() is False or HAS_TRITON is False:
        raise RuntimeError("CUDA and Triton are required for scatter tuning")
    _validate_kernel_block_widths(kernel_block_widths)
    if lineage is None and kernel_block_widths != KERNEL_BLOCK_WIDTHS:
        raise ValueError("an initial campaign must use the canonical width matrix")
    if lineage is not None:
        extension_width = lineage.get("extension_kernel_block_bytes")
        if type(extension_width) is not int or kernel_block_widths != (
            extension_kernel_block_widths(extension_width)
        ):
            raise ValueError("extension lineage differs from the campaign matrix")

    artifact_directory.mkdir(parents=True)
    running_marker = artifact_directory / "RUNNING"
    running_marker.write_text(f"pid={os.getpid()}\n")
    try:
        plans = tuple(build_shape_plan(shape) for shape in SCATTER_SHAPES)
        device = torch.device("cuda", torch.cuda.current_device())
        _write_json(
            artifact_directory / "environment.json",
            _environment(plans, device, kernel_block_widths, lineage),
        )

        required_peak_bytes = max(
            plan.staging_size_bytes + 2 * _destination_bytes(plan) + 2 * 1024**3
            for plan in plans
        )
        free_bytes, _ = torch.cuda.mem_get_info(device)
        if free_bytes < required_peak_bytes:
            raise RuntimeError(
                f"scatter tuning requires {required_peak_bytes} free bytes, "
                f"observed {free_bytes}"
            )

        sample_path = artifact_directory / "samples.jsonl"
        correctness_path = artifact_directory / "correctness.jsonl"
        samples: list[SampleRecord] = []
        for shape_index, (shape, plan) in enumerate(
            zip(SCATTER_SHAPES, plans, strict=True)
        ):
            correctness, shape_samples = _benchmark_shape(
                shape_index=shape_index,
                shape=shape,
                plan=plan,
                device=device,
                kernel_block_widths=kernel_block_widths,
            )
            _append_jsonl(correctness_path, tuple(correctness))
            _append_jsonl(sample_path, tuple(shape_samples))
            samples.extend(shape_samples)
            gc.collect()
            torch.cuda.empty_cache()
        summary = select_kernel_width(
            tuple(samples),
            kernel_block_widths,
            permit_boundary_extension=lineage is None,
        )
        _write_json(artifact_directory / "selection.json", summary)
        _write_json(
            artifact_directory / "environment-final.json",
            {
                "schema_version": SCHEMA_VERSION,
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "nvidia_smi_gpu": _capture_command(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,uuid,name,memory.total,memory.free,"
                        "temperature.gpu,clocks.sm,clocks.mem,power.draw",
                        "--format=csv,noheader,nounits",
                    ]
                ),
                "nvidia_smi_processes": _capture_command(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                        "--format=csv,noheader,nounits",
                    ]
                ),
            },
        )
        evidence_files = (
            "environment.json",
            "environment-final.json",
            "correctness.jsonl",
            "samples.jsonl",
            "selection.json",
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "files": {
                name: _sha256(artifact_directory / name) for name in evidence_files
            },
        }
        _write_json(artifact_directory / "manifest.json", manifest)
        manifest_sha256 = _sha256(artifact_directory / "manifest.json")
        running_marker.unlink()
        terminal_marker = (
            "EXTENSION_REQUIRED"
            if summary["boundary_extension_required"]
            else "COMPLETE"
        )
        (artifact_directory / terminal_marker).write_text(
            f"manifest_sha256={manifest_sha256}\n"
        )
        return summary
    except Exception:
        failure = traceback.format_exc()
        if running_marker.exists():
            running_marker.unlink()
        (artifact_directory / "FAILED").write_text(failure)
        raise


def run_campaign(artifact_directory: Path) -> SelectionSummary:
    """Execute the immutable initial local-GPU tuning campaign.

    :param artifact_directory: New directory that will own all raw evidence.
    :returns: Deterministic initial selection or adjacent extension request.
    """
    return _run_campaign(artifact_directory, KERNEL_BLOCK_WIDTHS, None)


def run_extension_campaign(
    *,
    initial_artifact_directory: Path,
    extension_kernel_block_bytes: int,
    artifact_directory: Path,
) -> SelectionSummary:
    """Execute the one closed extension requested by a sealed initial run.

    The initial artifact is fully authenticated and replayed before the output
    directory is created. The final campaign collects an entirely fresh paired
    matrix over the original widths and the requested adjacent width.

    :param initial_artifact_directory: Sealed initial extension request.
    :param extension_kernel_block_bytes: Exact requested adjacent width.
    :param artifact_directory: New final artifact directory.
    :returns: Deterministic final selection with no recursive extension.
    :raises ValueError: If the explicit request differs from the sealed request.
    """
    receipt = authenticate_initial_extension_artifact(initial_artifact_directory)
    if extension_kernel_block_bytes != receipt.extension_kernel_block_bytes:
        raise ValueError(
            "explicit extension width differs from the authenticated initial request"
        )
    lineage: dict[str, object] = {
        "initial_manifest_sha256": receipt.manifest_sha256,
        "initial_selection_sha256": receipt.selection_sha256,
        "initial_selected_kernel_block_bytes": (
            receipt.selected_kernel_block_bytes
        ),
        "extension_kernel_block_bytes": receipt.extension_kernel_block_bytes,
    }
    summary = _run_campaign(
        artifact_directory,
        extension_kernel_block_widths(extension_kernel_block_bytes),
        lineage,
    )
    validate_campaign_artifact(artifact_directory, initial_artifact_directory)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-directory",
        type=Path,
        help="new directory for raw timings and deterministic selection evidence",
    )
    parser.add_argument(
        "--extend-initial-artifact",
        type=Path,
        help="sealed initial EXTENSION_REQUIRED artifact to authenticate and extend",
    )
    parser.add_argument(
        "--extension-kernel-block-bytes",
        type=int,
        help="exact adjacent width requested by the sealed initial artifact",
    )
    parser.add_argument(
        "--validate-artifact",
        type=Path,
        help="sealed initial or final artifact to replay without using CUDA",
    )
    parser.add_argument(
        "--validation-initial-artifact",
        type=Path,
        help="initial artifact used to authenticate final extension lineage",
    )
    return parser


def main() -> None:
    """Run the command-line scatter tuning campaign."""
    parser = _parser()
    arguments = parser.parse_args()
    if arguments.validate_artifact is not None:
        if (
            arguments.artifact_directory is not None
            or arguments.extend_initial_artifact is not None
            or arguments.extension_kernel_block_bytes is not None
        ):
            parser.error("artifact validation cannot also execute a campaign")
        summary = validate_campaign_artifact(
            arguments.validate_artifact,
            arguments.validation_initial_artifact,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    if arguments.validation_initial_artifact is not None:
        parser.error("--validation-initial-artifact requires --validate-artifact")
    if arguments.artifact_directory is None:
        parser.error("--artifact-directory is required when executing a campaign")
    if arguments.extend_initial_artifact is None:
        if arguments.extension_kernel_block_bytes is not None:
            parser.error(
                "--extension-kernel-block-bytes requires --extend-initial-artifact"
            )
        summary = run_campaign(arguments.artifact_directory)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    if arguments.extension_kernel_block_bytes is None:
        parser.error(
            "--extend-initial-artifact requires --extension-kernel-block-bytes"
        )
    summary = run_extension_campaign(
        initial_artifact_directory=arguments.extend_initial_artifact,
        extension_kernel_block_bytes=arguments.extension_kernel_block_bytes,
        artifact_directory=arguments.artifact_directory,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
