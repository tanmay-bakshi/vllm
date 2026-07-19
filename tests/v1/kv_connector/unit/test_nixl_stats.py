# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for NIXL transfer-plan telemetry."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from prometheus_client import Counter, Gauge, Histogram

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.stats import (
    NixlCoalescedPlanTelemetry,
    NixlKVConnectorStats,
    NixlPromMetrics,
)


@dataclass(frozen=True, slots=True)
class _NativeTelemetry:
    """Minimal native NIXL telemetry used by the aggregation factory."""

    xferDuration: int
    postDuration: int
    totalBytes: int
    descCount: int


class _FakeMetric:
    """Capture Prometheus construction and observation without a registry."""

    def __init__(self, **kwargs: object) -> None:
        """Initialize a fake metric.

        :param kwargs: Prometheus metric constructor arguments.
        """
        self.kwargs = kwargs
        self.children: list[_FakeMetric] = []
        self.observed: list[int | float] = []
        self.increments: list[int | float] = []
        self.labelvalues: tuple[object, ...] = ()

    def labels(self, *labelvalues: object) -> "_FakeMetric":
        """Create a labeled child metric.

        :param labelvalues: Values bound to the configured labels.
        :returns: New fake child metric.
        """
        child = _FakeMetric(**self.kwargs)
        child.labelvalues = labelvalues
        self.children.append(child)
        return child

    def observe(self, value: int | float) -> None:
        """Capture one histogram observation.

        :param value: Observed value.
        """
        self.observed.append(value)

    def inc(self, value: int | float) -> None:
        """Capture one counter increment.

        :param value: Counter increment.
        """
        self.increments.append(value)


def _plan_telemetry(
    scatter_gpu_duration_seconds: float | None = 0.00011,
) -> NixlCoalescedPlanTelemetry:
    """Build one plan record from four native rank handles.

    :param scatter_gpu_duration_seconds: CUDA timing, or ``None`` when its
        event query was unavailable after quiescence.
    :returns: Validated plan-level telemetry.
    """
    native_transfers = (
        _NativeTelemetry(1000, 100, 100, 2),
        _NativeTelemetry(2000, 200, 200, 3),
        _NativeTelemetry(3000, 300, 300, 5),
        _NativeTelemetry(4000, 400, 400, 7),
    )
    return NixlCoalescedPlanTelemetry.from_native_transfers(
        logical_bytes=1600,
        layout_duration_seconds=0.0002,
        native_transfers=native_transfers,  # type: ignore[arg-type]
        scatter_duration_seconds=0.0003,
        scatter_gpu_duration_seconds=scatter_gpu_duration_seconds,
        staging_residency_seconds=0.02,
    )


@pytest.mark.cpu_test
def test_native_factory_aggregates_one_complete_plan() -> None:
    """Derive wire work and explicit sum/max duration semantics."""
    telemetry = _plan_telemetry()

    assert telemetry.logical_bytes == 1600
    assert telemetry.wire_bytes == 1000
    assert telemetry.elided_bytes == 600
    assert telemetry.descriptor_count == 17
    assert telemetry.native_transfer_duration_sum_seconds == pytest.approx(0.01)
    assert telemetry.native_transfer_duration_max_seconds == pytest.approx(0.004)
    assert telemetry.native_post_duration_sum_seconds == pytest.approx(0.001)
    assert telemetry.native_post_duration_max_seconds == pytest.approx(0.0004)
    assert telemetry.scatter_duration_seconds == pytest.approx(0.0003)
    assert telemetry.scatter_gpu_duration_seconds == pytest.approx(0.00011)


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"logical_bytes": -1}, "logical_bytes"),
        ({"wire_bytes": 1601}, "cannot exceed"),
        ({"descriptor_count": 1.5}, "descriptor_count"),
        ({"layout_duration_seconds": float("nan")}, "layout_duration_seconds"),
        ({"scatter_gpu_duration_seconds": -0.1}, "scatter_gpu_duration_seconds"),
        (
            {
                "native_transfer_duration_sum_seconds": 0.1,
                "native_transfer_duration_max_seconds": 0.2,
            },
            "transfer maximum",
        ),
        (
            {
                "native_post_duration_sum_seconds": 0.1,
                "native_post_duration_max_seconds": 0.2,
            },
            "post maximum",
        ),
    ],
)
def test_plan_telemetry_rejects_invalid_observations(
    changes: dict[str, object],
    match: str,
) -> None:
    """Reject inconsistent byte, descriptor, and duration observations."""
    values: dict[str, object] = {
        "logical_bytes": 1000,
        "wire_bytes": 800,
        "layout_duration_seconds": 0.1,
        "descriptor_count": 4,
        "native_transfer_duration_sum_seconds": 0.4,
        "native_transfer_duration_max_seconds": 0.1,
        "native_post_duration_sum_seconds": 0.04,
        "native_post_duration_max_seconds": 0.01,
        "scatter_duration_seconds": 0.02,
        "scatter_gpu_duration_seconds": 0.01,
        "staging_residency_seconds": 0.6,
    }
    values.update(changes)

    with pytest.raises(ValueError, match=match):
        NixlCoalescedPlanTelemetry(**values)  # type: ignore[arg-type]


@pytest.mark.cpu_test
def test_stats_record_reduce_clone_and_aggregate_plan_rows() -> None:
    """Keep plan columns aligned through reduction, snapshots, and aggregation."""
    first = NixlKVConnectorStats()
    first.record_coalesced_plan(_plan_telemetry())
    assert not first.is_empty()
    assert first.num_successful_transfers == 0
    assert first.num_coalesced_plans == 1

    reduced = first.reduce()
    assert reduced["Num successful transfers"] == 0
    assert reduced["Num coalesced plans"] == 1
    assert reduced["Avg logical MB per coalesced plan"] == round(1600 / 2**20, 3)
    assert reduced["Avg wire MB per coalesced plan"] == round(1000 / 2**20, 3)
    assert reduced["Coalesced bytes elided (%)"] == 37.5
    assert reduced["Avg coalesced descriptors"] == 17.0
    assert reduced["Avg coalesced native xfer sum (ms)"] == 10.0
    assert reduced["Avg coalesced native xfer max proxy (ms)"] == 4.0
    assert reduced["Avg coalesced scatter time (ms)"] == 0.3
    assert reduced["Num coalesced scatter GPU timings"] == 1
    assert reduced["Avg coalesced scatter GPU time (ms)"] == 0.11
    assert reduced["Avg coalesced staging residency (ms)"] == 20.0

    snapshot = first.clone_and_reset()
    assert snapshot.num_coalesced_plans == 1
    assert first.is_empty()

    second = NixlKVConnectorStats()
    second.record_coalesced_plan(_plan_telemetry())
    snapshot.aggregate(second)
    assert snapshot.num_coalesced_plans == 2
    assert len(snapshot.data["coalesced_plan_staging_residency"]) == 2


@pytest.mark.cpu_test
def test_missing_gpu_timing_preserves_other_plan_telemetry() -> None:
    """An unavailable CUDA duration is omitted rather than replaced by wall time."""
    missing = NixlKVConnectorStats()
    missing.record_coalesced_plan(_plan_telemetry(None))

    reduced = missing.reduce()
    assert reduced["Num coalesced plans"] == 1
    assert reduced["Num coalesced scatter GPU timings"] == 0
    assert "Avg coalesced scatter GPU time (ms)" not in reduced
    assert missing.data["coalesced_plan_scatter_gpu_duration"] == []
    assert missing.data["coalesced_plan_scatter_duration"] == [0.0003]

    measured = NixlKVConnectorStats()
    measured.record_coalesced_plan(_plan_telemetry())
    missing.aggregate(measured)

    reduced = missing.reduce()
    assert reduced["Num coalesced plans"] == 2
    assert reduced["Num coalesced scatter GPU timings"] == 1
    assert reduced["Avg coalesced scatter GPU time (ms)"] == 0.11


@pytest.mark.cpu_test
def test_prometheus_omits_only_an_unavailable_gpu_timing() -> None:
    """Missing event timing leaves no false zero or wall-time histogram sample."""
    prom_metrics = NixlPromMetrics(
        vllm_config=SimpleNamespace(kv_transfer_config=None),  # type: ignore[arg-type]
        metric_types={
            Gauge: _FakeMetric,
            Counter: _FakeMetric,
            Histogram: _FakeMetric,
        },
        labelnames=["model_name", "engine"],
        per_engine_labelvalues={0: ["model", "0"]},
    )
    stats = NixlKVConnectorStats()
    stats.record_coalesced_plan(_plan_telemetry(None))

    prom_metrics.observe(stats.data)

    gpu_histogram = prom_metrics.coalesced_plan_histograms[
        "coalesced_plan_scatter_gpu_duration"
    ][0]
    wall_histogram = prom_metrics.coalesced_plan_histograms[
        "coalesced_plan_scatter_duration"
    ][0]
    logical_histogram = prom_metrics.coalesced_plan_histograms[
        "coalesced_plan_logical_bytes"
    ][0]
    assert gpu_histogram.observed == []
    assert wall_histogram.observed == [pytest.approx(0.0003)]
    assert logical_histogram.observed == [1600]


@pytest.mark.cpu_test
def test_existing_native_transfer_metrics_keep_their_contract() -> None:
    """Retain existing per-handle units, reduction names, and Prom observations."""
    stats = NixlKVConnectorStats()
    native = _NativeTelemetry(2000, 500, 2**20, 8)
    stats.record_transfer(native)  # type: ignore[arg-type]

    reduced = stats.reduce()
    assert reduced["Num successful transfers"] == 1
    assert reduced["Avg xfer time (ms)"] == 2.0
    assert reduced["Avg post time (ms)"] == 0.5
    assert reduced["Avg MB per transfer"] == 1.0
    assert reduced["Avg number of descriptors"] == 8.0
    assert reduced["Num coalesced plans"] == 0


@pytest.mark.cpu_test
def test_prometheus_observes_every_plan_metric_on_the_selected_engine() -> None:
    """Export one aligned plan row through the full Prometheus surface."""
    prom_metrics = NixlPromMetrics(
        vllm_config=SimpleNamespace(kv_transfer_config=None),  # type: ignore[arg-type]
        metric_types={
            Gauge: _FakeMetric,
            Counter: _FakeMetric,
            Histogram: _FakeMetric,
        },
        labelnames=["model_name", "engine"],
        per_engine_labelvalues={
            0: ["model", "0"],
            1: ["model", "1"],
        },
    )
    stats = NixlKVConnectorStats()
    stats.record_coalesced_plan(_plan_telemetry())

    prom_metrics.observe(stats.data, engine_idx=1)

    expected = {
        "coalesced_plan_logical_bytes": (
            "vllm:nixl_coalesced_plan_logical_bytes",
            1600,
        ),
        "coalesced_plan_wire_bytes": (
            "vllm:nixl_coalesced_plan_wire_bytes",
            1000,
        ),
        "coalesced_plan_elided_bytes": (
            "vllm:nixl_coalesced_plan_elided_bytes",
            600,
        ),
        "coalesced_plan_layout_duration": (
            "vllm:nixl_coalesced_plan_layout_time_seconds",
            0.0002,
        ),
        "coalesced_plan_num_descriptors": (
            "vllm:nixl_coalesced_plan_num_descriptors",
            17,
        ),
        "coalesced_plan_native_transfer_duration_sum": (
            "vllm:nixl_coalesced_plan_native_xfer_time_sum_seconds",
            0.01,
        ),
        "coalesced_plan_native_transfer_duration_max": (
            "vllm:nixl_coalesced_plan_native_xfer_time_max_seconds",
            0.004,
        ),
        "coalesced_plan_native_post_duration_sum": (
            "vllm:nixl_coalesced_plan_native_post_time_sum_seconds",
            0.001,
        ),
        "coalesced_plan_native_post_duration_max": (
            "vllm:nixl_coalesced_plan_native_post_time_max_seconds",
            0.0004,
        ),
        "coalesced_plan_scatter_duration": (
            "vllm:nixl_coalesced_plan_scatter_time_seconds",
            0.0003,
        ),
        "coalesced_plan_scatter_gpu_duration": (
            "vllm:nixl_coalesced_plan_scatter_gpu_time_seconds",
            0.00011,
        ),
        "coalesced_plan_staging_residency": (
            "vllm:nixl_coalesced_plan_staging_residency_seconds",
            0.02,
        ),
    }
    for data_key, (metric_name, observation) in expected.items():
        engine_zero = prom_metrics.coalesced_plan_histograms[data_key][0]
        engine_one = prom_metrics.coalesced_plan_histograms[data_key][1]
        assert engine_zero.observed == []
        assert engine_one.observed == [pytest.approx(observation)]
        assert engine_one.kwargs["name"] == metric_name
        assert engine_one.labelvalues == ("model", "1")
