# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stats and Prometheus metrics for the NIXL connector."""

import copy
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self, cast

import numpy as np
from prometheus_client import Counter, Histogram

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics,
    KVConnectorStats,
    PromMetric,
    PromMetricT,
)
from vllm.v1.metrics.utils import create_metric_per_engine

if TYPE_CHECKING:
    from vllm.distributed.nixl_utils import nixlXferTelemetry


_PLAN_LOGICAL_BYTES = "coalesced_plan_logical_bytes"
_PLAN_WIRE_BYTES = "coalesced_plan_wire_bytes"
_PLAN_ELIDED_BYTES = "coalesced_plan_elided_bytes"
_PLAN_LAYOUT_DURATION = "coalesced_plan_layout_duration"
_PLAN_NUM_DESCRIPTORS = "coalesced_plan_num_descriptors"
_PLAN_NATIVE_TRANSFER_DURATION_SUM = "coalesced_plan_native_transfer_duration_sum"
_PLAN_NATIVE_TRANSFER_DURATION_MAX = "coalesced_plan_native_transfer_duration_max"
_PLAN_NATIVE_POST_DURATION_SUM = "coalesced_plan_native_post_duration_sum"
_PLAN_NATIVE_POST_DURATION_MAX = "coalesced_plan_native_post_duration_max"
_PLAN_SCATTER_DURATION = "coalesced_plan_scatter_duration"
_PLAN_SCATTER_GPU_DURATION = "coalesced_plan_scatter_gpu_duration"
_PLAN_STAGING_RESIDENCY = "coalesced_plan_staging_residency"


@dataclass(frozen=True, slots=True)
class NixlCoalescedPlanTelemetry:
    """One completed coalesced transfer plan's production telemetry.

    Native duration sums measure aggregate backend work across rank handles.
    Native duration maxima are critical-path proxies only because the handles
    do not provide a synchronized wall-clock interval.

    :ivar logical_bytes: Bytes the unpruned logical transfer would contain.
    :ivar wire_bytes: Bytes described by all native rank handles.
    :ivar layout_duration_seconds: Wall time spent building the canonical plan.
    :ivar descriptor_count: Total descriptors posted across native handles.
    :ivar native_transfer_duration_sum_seconds: Sum of native transfer
        durations across rank handles.
    :ivar native_transfer_duration_max_seconds: Maximum native transfer
        duration among rank handles.
    :ivar native_post_duration_sum_seconds: Sum of native post durations across
        rank handles.
    :ivar native_post_duration_max_seconds: Maximum native post duration among
        rank handles.
    :ivar scatter_duration_seconds: Wall time from scatter enqueue until the
        worker observed its completion event ready.
    :ivar scatter_gpu_duration_seconds: Exact CUDA start-to-end event duration
        for the scatter kernels, or ``None`` when CUDA timing was unavailable.
    :ivar staging_residency_seconds: Wall time from staging lease acquisition
        until every release precondition was established.
    """

    logical_bytes: int
    wire_bytes: int
    layout_duration_seconds: float
    descriptor_count: int
    native_transfer_duration_sum_seconds: float
    native_transfer_duration_max_seconds: float
    native_post_duration_sum_seconds: float
    native_post_duration_max_seconds: float
    scatter_duration_seconds: float
    scatter_gpu_duration_seconds: float | None
    staging_residency_seconds: float

    def __post_init__(self) -> None:
        """Validate units and cross-field invariants."""
        _require_non_negative_int("logical_bytes", self.logical_bytes)
        _require_non_negative_int("wire_bytes", self.wire_bytes)
        _require_non_negative_int("descriptor_count", self.descriptor_count)
        if self.wire_bytes > self.logical_bytes:
            raise ValueError("wire_bytes cannot exceed logical_bytes")
        for name, value in (
            ("layout_duration_seconds", self.layout_duration_seconds),
            (
                "native_transfer_duration_sum_seconds",
                self.native_transfer_duration_sum_seconds,
            ),
            (
                "native_transfer_duration_max_seconds",
                self.native_transfer_duration_max_seconds,
            ),
            (
                "native_post_duration_sum_seconds",
                self.native_post_duration_sum_seconds,
            ),
            (
                "native_post_duration_max_seconds",
                self.native_post_duration_max_seconds,
            ),
            ("scatter_duration_seconds", self.scatter_duration_seconds),
            ("staging_residency_seconds", self.staging_residency_seconds),
        ):
            _require_duration(name, value)
        if self.scatter_gpu_duration_seconds is not None:
            _require_duration(
                "scatter_gpu_duration_seconds",
                self.scatter_gpu_duration_seconds,
            )
        if (
            self.native_transfer_duration_max_seconds
            > self.native_transfer_duration_sum_seconds
        ):
            raise ValueError("native transfer maximum cannot exceed its sum")
        if (
            self.native_post_duration_max_seconds
            > self.native_post_duration_sum_seconds
        ):
            raise ValueError("native post maximum cannot exceed its sum")

    @property
    def elided_bytes(self) -> int:
        """Return bytes removed from the logical transfer.

        :returns: Exact difference between logical and wire bytes.
        """
        return self.logical_bytes - self.wire_bytes

    @classmethod
    def from_native_transfers(
        cls,
        *,
        logical_bytes: int,
        layout_duration_seconds: float,
        native_transfers: tuple["nixlXferTelemetry", ...],
        scatter_duration_seconds: float,
        scatter_gpu_duration_seconds: float | None,
        staging_residency_seconds: float,
    ) -> Self:
        """Aggregate native per-handle telemetry into one typed plan record.

        NIXL reports durations in microseconds. This factory converts them to
        seconds, matching the existing connector metric contract.

        :param logical_bytes: Bytes the unpruned logical transfer would contain.
        :param layout_duration_seconds: Wall time spent building the plan.
        :param native_transfers: Terminal native telemetry for every rank handle.
        :param scatter_duration_seconds: Scatter enqueue-to-completion-observation
            wall time.
        :param scatter_gpu_duration_seconds: Exact CUDA start-to-end event
            duration for the scatter kernels, or ``None`` when unavailable.
        :param staging_residency_seconds: Lease acquisition-to-release-ready
            wall time.
        :returns: Validated plan telemetry with summed native byte, descriptor,
            and duration observations.
        """
        transfer_durations = tuple(
            float(transfer.xferDuration) / 1e6 for transfer in native_transfers
        )
        post_durations = tuple(
            float(transfer.postDuration) / 1e6 for transfer in native_transfers
        )
        return cls(
            logical_bytes=logical_bytes,
            wire_bytes=sum(int(transfer.totalBytes) for transfer in native_transfers),
            layout_duration_seconds=layout_duration_seconds,
            descriptor_count=sum(
                int(transfer.descCount) for transfer in native_transfers
            ),
            native_transfer_duration_sum_seconds=sum(transfer_durations),
            native_transfer_duration_max_seconds=max(transfer_durations, default=0.0),
            native_post_duration_sum_seconds=sum(post_durations),
            native_post_duration_max_seconds=max(post_durations, default=0.0),
            scatter_duration_seconds=scatter_duration_seconds,
            scatter_gpu_duration_seconds=scatter_gpu_duration_seconds,
            staging_residency_seconds=staging_residency_seconds,
        )


def _require_non_negative_int(name: str, value: int) -> None:
    """Require an exact, non-negative integer.

    :param name: Field name for validation errors.
    :param value: Candidate integer.
    """
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _require_duration(name: str, value: float) -> None:
    """Require a finite, non-negative duration in seconds.

    :param name: Field name for validation errors.
    :param value: Candidate duration.
    """
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite, non-negative duration")


@dataclass
class NixlKVConnectorStats(KVConnectorStats):
    """Container for transfer performance metrics"""

    def __post_init__(self):
        if not self.data:
            # Empty container init, no data is passed in.
            self.reset()

    def reset(self):
        # Must be serializable
        self.data: dict[str, list[float | int]] = {
            "transfer_duration": [],
            "post_duration": [],
            "bytes_transferred": [],
            "num_descriptors": [],
            "num_failed_transfers": [],
            "num_failed_notifications": [],
            "num_kv_expired_reqs": [],
            _PLAN_LOGICAL_BYTES: [],
            _PLAN_WIRE_BYTES: [],
            _PLAN_ELIDED_BYTES: [],
            _PLAN_LAYOUT_DURATION: [],
            _PLAN_NUM_DESCRIPTORS: [],
            _PLAN_NATIVE_TRANSFER_DURATION_SUM: [],
            _PLAN_NATIVE_TRANSFER_DURATION_MAX: [],
            _PLAN_NATIVE_POST_DURATION_SUM: [],
            _PLAN_NATIVE_POST_DURATION_MAX: [],
            _PLAN_SCATTER_DURATION: [],
            _PLAN_SCATTER_GPU_DURATION: [],
            _PLAN_STAGING_RESIDENCY: [],
        }

    def record_transfer(self, res: "nixlXferTelemetry"):
        # Keep metrics units consistent with rest of the code: time us->s
        self.data["transfer_duration"].append(res.xferDuration / 1e6)
        self.data["post_duration"].append(res.postDuration / 1e6)
        self.data["bytes_transferred"].append(res.totalBytes)
        self.data["num_descriptors"].append(res.descCount)

    def record_coalesced_plan(self, telemetry: NixlCoalescedPlanTelemetry) -> None:
        """Record one completed canonical coalesced transfer plan.

        :param telemetry: Validated plan-level byte, descriptor, and duration
            observations.
        """
        self.data[_PLAN_LOGICAL_BYTES].append(telemetry.logical_bytes)
        self.data[_PLAN_WIRE_BYTES].append(telemetry.wire_bytes)
        self.data[_PLAN_ELIDED_BYTES].append(telemetry.elided_bytes)
        self.data[_PLAN_LAYOUT_DURATION].append(telemetry.layout_duration_seconds)
        self.data[_PLAN_NUM_DESCRIPTORS].append(telemetry.descriptor_count)
        self.data[_PLAN_NATIVE_TRANSFER_DURATION_SUM].append(
            telemetry.native_transfer_duration_sum_seconds
        )
        self.data[_PLAN_NATIVE_TRANSFER_DURATION_MAX].append(
            telemetry.native_transfer_duration_max_seconds
        )
        self.data[_PLAN_NATIVE_POST_DURATION_SUM].append(
            telemetry.native_post_duration_sum_seconds
        )
        self.data[_PLAN_NATIVE_POST_DURATION_MAX].append(
            telemetry.native_post_duration_max_seconds
        )
        self.data[_PLAN_SCATTER_DURATION].append(telemetry.scatter_duration_seconds)
        if telemetry.scatter_gpu_duration_seconds is not None:
            self.data[_PLAN_SCATTER_GPU_DURATION].append(
                telemetry.scatter_gpu_duration_seconds
            )
        self.data[_PLAN_STAGING_RESIDENCY].append(telemetry.staging_residency_seconds)

    def record_failed_transfer(self):
        """Record a failed NIXL transfer operation."""
        self.data["num_failed_transfers"].append(1)

    def record_failed_notification(self):
        """Record a failed NIXL notification (send_notif)."""
        self.data["num_failed_notifications"].append(1)

    def record_kv_expired_req(self):
        """Record a request whose KV liveness deadline elapsed."""
        self.data["num_kv_expired_reqs"].append(1)

    def clone_and_reset(self) -> "NixlKVConnectorStats":
        old = copy.copy(self)
        self.reset()
        return old

    def is_empty(self) -> bool:
        # Do not discard metrics update that are entirely failures related.
        return (
            self.num_successful_transfers == 0
            and self.num_coalesced_plans == 0
            and len(self.data["num_failed_transfers"]) == 0
            and len(self.data["num_failed_notifications"]) == 0
            and len(self.data["num_kv_expired_reqs"]) == 0
        )

    def aggregate(self, other: KVConnectorStats) -> KVConnectorStats:
        if not other.is_empty():
            for k, v in other.data.items():
                accumulator = self.data[k]
                assert isinstance(accumulator, list)
                accumulator.extend(v)
        return self

    def reduce(self) -> dict[str, int | float]:
        # Compute compact representative stats suitable for CLI logging
        if self.num_successful_transfers == 0:
            # CLI logging only reports successful transfers stats. If all requests in
            # the interval were unsuccessful, Prom will report failures stats instead.
            reduced: dict[str, int | float] = {
                "Num successful transfers": 0,
                "Avg xfer time (ms)": 0,
                "P90 xfer time (ms)": 0,
                "Avg post time (ms)": 0,
                "P90 post time (ms)": 0,
                "Avg MB per transfer": 0,
                "Throughput (MB/s)": 0,
                "Avg number of descriptors": 0,
            }
        else:
            xfer_time = np.asarray(self.data["transfer_duration"])
            post_time = np.asarray(self.data["post_duration"])
            mb = np.asarray(self.data["bytes_transferred"]) / 2**20
            descs = np.asarray(self.data["num_descriptors"], dtype=np.uint32)
            n = len(descs)
            assert n == self.num_successful_transfers

            total_mb = mb.sum()
            total_time_seconds = xfer_time.sum()
            throughput_mb_s = (
                total_mb / total_time_seconds if total_time_seconds > 0 else 0.0
            )
            reduced = {
                "Num successful transfers": n,
                "Avg xfer time (ms)": round(xfer_time.mean() * 1e3, 3),
                "P90 xfer time (ms)": round(
                    np.percentile(xfer_time, 90).item() * 1e3, 3
                ),
                "Avg post time (ms)": round(post_time.mean() * 1e3, 3),
                "P90 post time (ms)": round(
                    np.percentile(post_time, 90).item() * 1e3, 3
                ),
                "Avg MB per transfer": round(total_mb / n, 3),
                "Throughput (MB/s)": round(throughput_mb_s, 3),
                "Avg number of descriptors": round(descs.mean(), 1),
            }
        reduced.update(self._reduce_coalesced_plans())
        return reduced

    def _reduce_coalesced_plans(self) -> dict[str, int | float]:
        """Reduce canonical-plan observations for CLI logging.

        :returns: Plan counts, byte efficiency, and mean phase durations.
        """
        if self.num_coalesced_plans == 0:
            return {"Num coalesced plans": 0}

        logical_bytes = np.asarray(self.data[_PLAN_LOGICAL_BYTES], dtype=np.uint64)
        wire_bytes = np.asarray(self.data[_PLAN_WIRE_BYTES], dtype=np.uint64)
        elided_bytes = np.asarray(self.data[_PLAN_ELIDED_BYTES], dtype=np.uint64)
        total_logical_bytes = int(logical_bytes.sum())
        total_elided_bytes = int(elided_bytes.sum())
        elided_percent = (
            100.0 * total_elided_bytes / total_logical_bytes
            if total_logical_bytes > 0
            else 0.0
        )

        def mean_milliseconds(key: str) -> float:
            values = np.asarray(self.data[key])
            return round(values.mean() * 1e3, 3)

        reduced: dict[str, int | float] = {
            "Num coalesced plans": self.num_coalesced_plans,
            "Avg logical MB per coalesced plan": round(logical_bytes.mean() / 2**20, 3),
            "Avg wire MB per coalesced plan": round(wire_bytes.mean() / 2**20, 3),
            "Coalesced bytes elided (%)": round(elided_percent, 3),
            "Avg coalesced layout time (ms)": mean_milliseconds(_PLAN_LAYOUT_DURATION),
            "Avg coalesced descriptors": round(
                np.asarray(self.data[_PLAN_NUM_DESCRIPTORS]).mean(), 1
            ),
            "Avg coalesced native xfer sum (ms)": mean_milliseconds(
                _PLAN_NATIVE_TRANSFER_DURATION_SUM
            ),
            "Avg coalesced native xfer max proxy (ms)": mean_milliseconds(
                _PLAN_NATIVE_TRANSFER_DURATION_MAX
            ),
            "Avg coalesced native post sum (ms)": mean_milliseconds(
                _PLAN_NATIVE_POST_DURATION_SUM
            ),
            "Avg coalesced native post max proxy (ms)": mean_milliseconds(
                _PLAN_NATIVE_POST_DURATION_MAX
            ),
            "Avg coalesced scatter time (ms)": mean_milliseconds(
                _PLAN_SCATTER_DURATION
            ),
            "Num coalesced scatter GPU timings": len(
                self.data[_PLAN_SCATTER_GPU_DURATION]
            ),
            "Avg coalesced staging residency (ms)": mean_milliseconds(
                _PLAN_STAGING_RESIDENCY
            ),
        }
        if len(self.data[_PLAN_SCATTER_GPU_DURATION]) > 0:
            reduced["Avg coalesced scatter GPU time (ms)"] = mean_milliseconds(
                _PLAN_SCATTER_GPU_DURATION
            )
        return reduced

    @property
    def num_successful_transfers(self) -> int:
        return len(self.data["transfer_duration"])

    @property
    def num_coalesced_plans(self) -> int:
        """Return the number of complete plan-level observations.

        :returns: Number of recorded coalesced plans.
        """
        return len(self.data[_PLAN_LOGICAL_BYTES])


class NixlPromMetrics(KVConnectorPromMetrics):
    def __init__(
        self,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ):
        super().__init__(vllm_config, metric_types, labelnames, per_engine_labelvalues)

        buckets = [
            0.001,
            0.005,
            0.01,
            0.025,
            0.05,
            0.075,
            0.1,
            0.2,
            0.3,
            0.5,
            0.75,
            1.0,
            5.0,
        ]
        histogram_cls = cast(type[Histogram], self._histogram_cls)
        nixl_histogram_xfer_time = histogram_cls(
            name="vllm:nixl_xfer_time_seconds",
            documentation="Histogram of transfer duration for NIXL KV Cache transfers.",
            buckets=buckets[1:],
            labelnames=labelnames,
        )
        self.nixl_histogram_xfer_time = cast(
            dict[int, Histogram],
            create_metric_per_engine(
                nixl_histogram_xfer_time, self.per_engine_labelvalues
            ),
        )
        nixl_histogram_post_time = histogram_cls(
            name="vllm:nixl_post_time_seconds",
            documentation="Histogram of transfer post time for NIXL KV"
            " Cache transfers.",
            buckets=buckets,
            labelnames=labelnames,
        )
        self.nixl_histogram_post_time = cast(
            dict[int, Histogram],
            create_metric_per_engine(
                nixl_histogram_post_time, self.per_engine_labelvalues
            ),
        )
        # uniform 2kb to 16gb range
        buckets = [2 ** (10 + i) for i in range(1, 25, 2)]
        nixl_histogram_bytes_transferred = histogram_cls(
            name="vllm:nixl_bytes_transferred",
            documentation="Histogram of bytes transferred per NIXL KV Cache transfers.",
            buckets=buckets,
            labelnames=labelnames,
        )
        self.nixl_histogram_bytes_transferred = cast(
            dict[int, Histogram],
            create_metric_per_engine(
                nixl_histogram_bytes_transferred, self.per_engine_labelvalues
            ),
        )
        buckets = [
            10,
            20,
            30,
            50,
            75,
            100,
            200,
            400,
            1000,
            2000,
            4000,
            10000,
            20000,
            50000,
        ]
        nixl_histogram_num_descriptors = histogram_cls(
            name="vllm:nixl_num_descriptors",
            documentation="Histogram of number of descriptors per NIXL"
            "  KV Cache transfers.",
            buckets=buckets,
            labelnames=labelnames,
        )
        self.nixl_histogram_num_descriptors = cast(
            dict[int, Histogram],
            create_metric_per_engine(
                nixl_histogram_num_descriptors, self.per_engine_labelvalues
            ),
        )

        byte_buckets = [
            0,
            *[2 ** (10 + i) for i in range(1, 25, 2)],
            2**34,
            2**35,
        ]
        duration_buckets = [
            0.000001,
            0.000005,
            0.00001,
            0.000025,
            0.00005,
            0.0001,
            0.00025,
            0.0005,
            0.001,
            0.005,
            0.01,
            0.025,
            0.05,
            0.1,
            0.25,
            0.5,
            1.0,
            5.0,
            10.0,
            30.0,
            60.0,
        ]
        plan_histogram_specs: dict[str, tuple[str, str, list[float | int]]] = {
            _PLAN_LOGICAL_BYTES: (
                "vllm:nixl_coalesced_plan_logical_bytes",
                "Logical bytes per canonical coalesced NIXL transfer plan.",
                byte_buckets,
            ),
            _PLAN_WIRE_BYTES: (
                "vllm:nixl_coalesced_plan_wire_bytes",
                "Native descriptor bytes per canonical coalesced NIXL transfer plan.",
                byte_buckets,
            ),
            _PLAN_ELIDED_BYTES: (
                "vllm:nixl_coalesced_plan_elided_bytes",
                "Logical bytes elided per canonical coalesced NIXL transfer plan.",
                byte_buckets,
            ),
            _PLAN_LAYOUT_DURATION: (
                "vllm:nixl_coalesced_plan_layout_time_seconds",
                "Canonical coalesced NIXL plan construction duration.",
                duration_buckets,
            ),
            _PLAN_NUM_DESCRIPTORS: (
                "vllm:nixl_coalesced_plan_num_descriptors",
                "Native descriptor count per canonical coalesced NIXL transfer plan.",
                buckets,
            ),
            _PLAN_NATIVE_TRANSFER_DURATION_SUM: (
                "vllm:nixl_coalesced_plan_native_xfer_time_sum_seconds",
                "Sum of native rank-handle transfer durations per coalesced plan.",
                duration_buckets,
            ),
            _PLAN_NATIVE_TRANSFER_DURATION_MAX: (
                "vllm:nixl_coalesced_plan_native_xfer_time_max_seconds",
                "Maximum native rank-handle transfer duration per coalesced plan; "
                "a critical-path proxy, not synchronized wall latency.",
                duration_buckets,
            ),
            _PLAN_NATIVE_POST_DURATION_SUM: (
                "vllm:nixl_coalesced_plan_native_post_time_sum_seconds",
                "Sum of native rank-handle post durations per coalesced plan.",
                duration_buckets,
            ),
            _PLAN_NATIVE_POST_DURATION_MAX: (
                "vllm:nixl_coalesced_plan_native_post_time_max_seconds",
                "Maximum native rank-handle post duration per coalesced plan; "
                "a critical-path proxy, not synchronized wall latency.",
                duration_buckets,
            ),
            _PLAN_SCATTER_DURATION: (
                "vllm:nixl_coalesced_plan_scatter_time_seconds",
                "Scatter enqueue-to-completion-observation wall duration per "
                "coalesced plan.",
                duration_buckets,
            ),
            _PLAN_SCATTER_GPU_DURATION: (
                "vllm:nixl_coalesced_plan_scatter_gpu_time_seconds",
                "Exact CUDA-event scatter duration per coalesced plan.",
                duration_buckets,
            ),
            _PLAN_STAGING_RESIDENCY: (
                "vllm:nixl_coalesced_plan_staging_residency_seconds",
                "Staging lease acquisition-to-release-ready duration per "
                "coalesced plan.",
                duration_buckets,
            ),
        }
        self.coalesced_plan_histograms: dict[str, dict[int, Histogram]] = {}
        for data_key, (
            metric_name,
            documentation,
            metric_buckets,
        ) in plan_histogram_specs.items():
            histogram = histogram_cls(
                name=metric_name,
                documentation=documentation,
                buckets=metric_buckets,
                labelnames=labelnames,
            )
            self.coalesced_plan_histograms[data_key] = cast(
                dict[int, Histogram],
                create_metric_per_engine(histogram, self.per_engine_labelvalues),
            )

        counter_cls = cast(type[Counter], self._counter_cls)
        counter_nixl_num_failed_transfers = counter_cls(
            name="vllm:nixl_num_failed_transfers",
            documentation="Number of failed NIXL KV Cache transfers.",
            labelnames=labelnames,
        )
        self.counter_nixl_num_failed_transfers = cast(
            dict[int, Counter],
            create_metric_per_engine(
                counter_nixl_num_failed_transfers, self.per_engine_labelvalues
            ),
        )
        counter_nixl_num_failed_notifications = counter_cls(
            name="vllm:nixl_num_failed_notifications",
            documentation="Number of failed NIXL KV Cache notifications.",
            labelnames=labelnames,
        )
        self.counter_nixl_num_failed_notifications = cast(
            dict[int, Counter],
            create_metric_per_engine(
                counter_nixl_num_failed_notifications, self.per_engine_labelvalues
            ),
        )

        counter_nixl_num_kv_expired_reqs = counter_cls(
            name="vllm:nixl_num_kv_expired_reqs",
            documentation=(
                "Number of source requests whose KV liveness deadline elapsed "
                "before authoritative transfer completion."
            ),
            labelnames=labelnames,
        )
        self.counter_nixl_num_kv_expired_reqs = cast(
            dict[int, Counter],
            create_metric_per_engine(
                counter_nixl_num_kv_expired_reqs, self.per_engine_labelvalues
            ),
        )

    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0):
        for prom_obj, list_item_key in zip(
            [
                self.nixl_histogram_xfer_time,
                self.nixl_histogram_post_time,
                self.nixl_histogram_bytes_transferred,
                self.nixl_histogram_num_descriptors,
            ],
            [
                "transfer_duration",
                "post_duration",
                "bytes_transferred",
                "num_descriptors",
            ],
        ):
            for list_item in transfer_stats_data[list_item_key]:
                prom_obj[engine_idx].observe(list_item)
        for counter_obj, counter_item_key in zip(
            [
                self.counter_nixl_num_failed_transfers,
                self.counter_nixl_num_failed_notifications,
                self.counter_nixl_num_kv_expired_reqs,
            ],
            ["num_failed_transfers", "num_failed_notifications", "num_kv_expired_reqs"],
        ):
            for list_item in transfer_stats_data[counter_item_key]:
                counter_obj[engine_idx].inc(list_item)
        for data_key, histogram in self.coalesced_plan_histograms.items():
            for observation in transfer_stats_data[data_key]:
                histogram[engine_idx].observe(observation)
