# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host tests for deterministic production scatter tuning."""

import hashlib
import json
from pathlib import Path

import pytest

from tools.gemma4_pd.tune_coalesced_scatter import (
    BASELINE_KERNEL_BLOCK_WIDTH,
    KERNEL_BLOCK_WIDTHS,
    SAMPLE_ROUND_COUNT,
    SCATTER_SHAPES,
    SampleRecord,
    SelectionSummary,
    authenticate_initial_extension_artifact,
    build_shape_plan,
    extension_kernel_block_widths,
    measurement_order,
    run_extension_campaign,
    select_kernel_width,
    validate_campaign_artifact,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import (
    coalesced_scatter as scatter_module,
)


def test_production_scatter_width_matches_qualified_selection() -> None:
    """Pin the production kernel to the closed five-width campaign winner."""

    assert scatter_module._KERNEL_BLOCK_BYTES == 4096


def _samples(
    durations: dict[int, dict[str, tuple[float, float]]],
    kernel_block_widths: tuple[int, ...] = KERNEL_BLOCK_WIDTHS,
) -> tuple[SampleRecord, ...]:
    """Build a complete constant-median timing matrix.

    :param durations: GPU and wall milliseconds by width and shape.
    :returns: Complete raw selection input.
    """
    records: list[SampleRecord] = []
    for shape_index, shape in enumerate(SCATTER_SHAPES):
        wire_bytes = build_shape_plan(shape).staging_size_bytes
        for sample_round, order in enumerate(
            measurement_order(shape_index, kernel_block_widths)
        ):
            for order_index, width in enumerate(order):
                gpu_ms, wall_ms = durations[width][shape.name]
                records.append(
                    {
                        "schema_version": 1,
                        "shape": shape.name,
                        "kernel_block_bytes": width,
                        "sample_round": sample_round,
                        "order_within_round": order_index,
                        "staging_base_offset_bytes": 0,
                        "wire_bytes": wire_bytes,
                        "gpu_duration_ms": gpu_ms,
                        "wall_duration_ms": wall_ms,
                        "gpu_gbps": wire_bytes / 1_000_000 / gpu_ms,
                        "wall_gbps": wire_bytes / 1_000_000 / wall_ms,
                    }
                )
    return tuple(records)


def _sha256(path: Path) -> str:
    """Return one fixture file digest.

    :param path: File to hash.
    :returns: Lowercase SHA-256 digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    """Write JSON with the tuner's canonical formatting.

    :param path: Fixture path.
    :param value: JSON-compatible value.
    """
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _seal_artifact(
    path: Path,
    *,
    samples: tuple[SampleRecord, ...],
    kernel_block_widths: tuple[int, ...],
    selection: SelectionSummary,
    campaign_kind: str,
    lineage: dict[str, object] | None,
) -> tuple[str, str]:
    """Build a minimal complete artifact accepted by the public validator.

    :param path: New fixture artifact directory.
    :param samples: Complete raw timing matrix.
    :param kernel_block_widths: Exact campaign width matrix.
    :param selection: Deterministically replayed selection.
    :param campaign_kind: Initial or closed extension campaign kind.
    :param lineage: Optional authenticated initial identity.
    :returns: Manifest and selection SHA-256 identities.
    """
    path.mkdir()
    _write_json(
        path / "environment.json",
        {
            "schema_version": 1,
            "protocol": {
                "campaign_kind": campaign_kind,
                "kernel_block_widths": list(kernel_block_widths),
                "lineage": lineage,
            },
        },
    )
    _write_json(path / "environment-final.json", {"schema_version": 1})
    with (path / "samples.jsonl").open("w") as destination:
        for sample in samples:
            destination.write(json.dumps(sample, sort_keys=True) + "\n")
    with (path / "correctness.jsonl").open("w") as destination:
        for shape in SCATTER_SHAPES:
            wire_bytes = build_shape_plan(shape).staging_size_bytes
            for width in kernel_block_widths:
                destination.write(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "shape": shape.name,
                            "kernel_block_bytes": width,
                            "wire_bytes": wire_bytes,
                            "exact_bytes_equal": True,
                            "semantic_probes_equal": True,
                            "redzones_intact": True,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
    _write_json(path / "selection.json", selection)
    evidence_files = (
        "environment.json",
        "environment-final.json",
        "correctness.jsonl",
        "samples.jsonl",
        "selection.json",
    )
    _write_json(
        path / "manifest.json",
        {
            "schema_version": 1,
            "files": {name: _sha256(path / name) for name in evidence_files},
        },
    )
    manifest_sha256 = _sha256(path / "manifest.json")
    marker = (
        "EXTENSION_REQUIRED"
        if selection["boundary_extension_required"]
        else "COMPLETE"
    )
    (path / marker).write_text(f"manifest_sha256={manifest_sha256}\n")
    return manifest_sha256, _sha256(path / "selection.json")


def _uniform_durations(
    by_width: dict[int, tuple[float, float]],
) -> dict[int, dict[str, tuple[float, float]]]:
    """Expand one duration pair across every canonical shape.

    :param by_width: GPU and wall milliseconds by kernel width.
    :returns: Complete width and shape timing map.
    """
    return {
        width: {shape.name: duration for shape in SCATTER_SHAPES}
        for width, duration in by_width.items()
    }


def _replace_duration(
    sample: SampleRecord,
    gpu_ms: float,
    wall_ms: float,
) -> None:
    """Replace one raw duration while preserving its throughput identity.

    :param sample: Mutable raw timing record.
    :param gpu_ms: Replacement CUDA-event duration.
    :param wall_ms: Replacement launch-to-event duration.
    """
    sample["gpu_duration_ms"] = gpu_ms
    sample["wall_duration_ms"] = wall_ms
    sample["gpu_gbps"] = sample["wire_bytes"] / 1_000_000 / gpu_ms
    sample["wall_gbps"] = sample["wire_bytes"] / 1_000_000 / wall_ms


@pytest.mark.cpu_test
def test_shape_plans_match_exact_production_volumes() -> None:
    """Keep minimum, exact-2K, and maximum geometry authoritative."""
    expected = {
        "minimum": (327_680, 1_310_720, 5, 5, (1, 0)),
        "exact_2k": (263_127_040, 1_052_508_160, 4015, 10, (770, 33)),
        "maximum": (3_565_158_400, 14_260_633_600, 54_400, 10, (8832, 2048)),
    }
    for shape in SCATTER_SHAPES:
        plan = build_shape_plan(shape)
        rank_stride, wire_bytes, positions, launches, partitions = expected[shape.name]
        assert plan.rank_stride_bytes == rank_stride
        assert plan.staging_size_bytes == wire_bytes
        assert sum(len(region.positions) for region in plan.regions) == positions
        assert sum(len(region.positions) > 0 for region in plan.regions) == launches
        assert len(plan.regions[0].positions) == partitions[0]
        assert len(plan.regions[5].positions) == partitions[1]
        assert plan.regions[0].ownership.destination_row_count == partitions[0] + 1
        assert plan.regions[5].ownership.destination_row_count == partitions[1] + 1
        assert all(
            len(region.runs) == (1 if len(region.positions) > 0 else 0)
            for region in plan.regions
        )


@pytest.mark.cpu_test
def test_measurement_order_is_fixed_seed_balanced_round_robin() -> None:
    """Preserve randomized order while sampling every width once per round."""
    order = measurement_order(0)
    assert len(order) == SAMPLE_ROUND_COUNT
    assert order[:3] == (
        (512, 2048, 1024, 4096),
        (1024, 512, 2048, 4096),
        (1024, 2048, 512, 4096),
    )
    assert all(
        tuple(sorted(round_order)) == KERNEL_BLOCK_WIDTHS for round_order in order
    )
    assert measurement_order(0) == order
    assert measurement_order(1) != order


@pytest.mark.cpu_test
def test_selection_rejects_any_per_shape_regression_over_three_percent() -> None:
    """A fast aggregate cannot conceal one materially slower shape."""
    durations = _uniform_durations(
        {
            512: (9.0, 9.0),
            1024: (10.0, 10.0),
            2048: (9.5, 9.5),
            4096: (10.2, 10.2),
        }
    )
    durations[512]["maximum"] = (10.4, 10.0)

    summary = select_kernel_width(_samples(durations))

    assert summary["candidates"]["512"]["eligible"] is False
    assert summary["candidates"]["512"]["rejected_shapes"] == ["maximum"]
    assert summary["selected_kernel_block_bytes"] == 2048


@pytest.mark.cpu_test
def test_selection_retains_1024_when_best_gain_is_within_noise_band() -> None:
    """Avoid changing production width for a sub-two-percent aggregate gain."""
    durations = _uniform_durations(
        {
            512: (9.82, 9.82),
            1024: (10.0, 10.0),
            2048: (10.1, 10.1),
            4096: (10.2, 10.2),
        }
    )

    summary = select_kernel_width(_samples(durations))

    assert summary["selected_kernel_block_bytes"] == BASELINE_KERNEL_BLOCK_WIDTH
    assert "within the 2% noise band" in summary["selection_reason"]


@pytest.mark.cpu_test
def test_selection_uses_equal_weight_geometric_mean_beyond_noise() -> None:
    """Choose the eligible cross-shape geometric-mean winner deterministically."""
    durations = _uniform_durations(
        {
            512: (9.8, 9.8),
            1024: (10.0, 10.0),
            2048: (9.0, 9.0),
            4096: (10.2, 10.2),
        }
    )

    summary = select_kernel_width(_samples(durations))

    assert summary["selected_kernel_block_bytes"] == 2048
    assert summary["boundary_extension_required"] is False
    assert summary["candidates"]["2048"]["geometric_mean_speedup"] == pytest.approx(
        10 / 9
    )


@pytest.mark.cpu_test
def test_selection_uses_exact_paired_upper_confidence_bound() -> None:
    """Reject credible regressions without letting a few outliers dominate."""
    durations = _uniform_durations(
        {
            512: (10.0, 10.0),
            1024: (10.0, 10.0),
            2048: (10.1, 10.1),
            4096: (10.2, 10.2),
        }
    )
    samples = list(_samples(durations))
    candidate = [
        sample
        for sample in samples
        if sample["shape"] == "maximum" and sample["kernel_block_bytes"] == 512
    ]
    for sample in candidate[:6]:
        _replace_duration(sample, 10.5, 10.5)

    six_outliers = select_kernel_width(tuple(samples))

    assert six_outliers["candidates"]["512"]["eligible"] is True
    assert six_outliers["paired_median_one_sided_confidence"] == pytest.approx(
        0.9608230590820312
    )

    _replace_duration(candidate[6], 10.5, 10.5)
    seven_outliers = select_kernel_width(tuple(samples))

    assert seven_outliers["candidates"]["512"]["eligible"] is False
    assert seven_outliers["candidates"]["512"]["rejected_shapes"] == ["maximum"]


@pytest.mark.cpu_test
def test_material_boundary_winner_requires_adjacent_extension() -> None:
    """Do not present a materially winning search boundary as final."""
    durations = _uniform_durations(
        {
            512: (9.0, 9.0),
            1024: (10.0, 10.0),
            2048: (9.5, 9.5),
            4096: (10.1, 10.1),
        }
    )

    summary = select_kernel_width(_samples(durations))

    assert summary["selected_kernel_block_bytes"] == 512
    assert summary["boundary_extension_required"] is True
    assert summary["boundary_extension_kernel_block_bytes"] == 256
    assert "provisional boundary winner" in summary["selection_reason"]


@pytest.mark.cpu_test
def test_selection_rejects_incomplete_raw_matrix() -> None:
    """Never select a width from a missing sample round."""
    durations = _uniform_durations(
        {
            512: (10.0, 10.0),
            1024: (10.0, 10.0),
            2048: (10.0, 10.0),
            4096: (10.0, 10.0),
        }
    )
    samples = _samples(durations)

    with pytest.raises(ValueError, match="raw matrix has"):
        select_kernel_width(samples[:-1])


@pytest.mark.cpu_test
def test_selection_rejects_reordered_raw_round() -> None:
    """Make the fixed-seed execution order part of the evidence contract."""
    durations = _uniform_durations(
        {
            512: (10.0, 10.0),
            1024: (10.0, 10.0),
            2048: (10.0, 10.0),
            4096: (10.0, 10.0),
        }
    )
    samples = list(_samples(durations))
    samples[0]["order_within_round"] = 1
    samples[1]["order_within_round"] = 0

    with pytest.raises(ValueError, match="fixed-seed schedule"):
        select_kernel_width(tuple(samples))


@pytest.mark.cpu_test
def test_extension_order_is_a_fixed_fresh_five_width_matrix() -> None:
    """Pair every original width with the one authenticated adjacent width."""
    widths = extension_kernel_block_widths(8192)
    order = measurement_order(0, widths)

    assert widths == (512, 1024, 2048, 4096, 8192)
    assert order[:3] == (
        (512, 2048, 1024, 4096, 8192),
        (512, 8192, 2048, 1024, 4096),
        (2048, 8192, 4096, 1024, 512),
    )
    assert all(tuple(sorted(round_order)) == widths for round_order in order)
    assert measurement_order(0, widths) == order


@pytest.mark.cpu_test
def test_final_extension_selection_is_closed_at_the_derived_width() -> None:
    """A final adjacent winner cannot recursively request an open-ended search."""
    widths = extension_kernel_block_widths(8192)
    samples = _samples(
        _uniform_durations(
            {
                512: (9.8, 9.8),
                1024: (10.0, 10.0),
                2048: (9.5, 9.5),
                4096: (8.8, 8.8),
                8192: (8.0, 8.0),
            }
        ),
        widths,
    )

    summary = select_kernel_width(
        samples,
        widths,
        permit_boundary_extension=False,
    )

    assert summary["selected_kernel_block_bytes"] == 8192
    assert summary["boundary_extension_required"] is False
    assert summary["boundary_extension_kernel_block_bytes"] is None


@pytest.mark.cpu_test
def test_sealed_extension_authenticates_lineage_and_replays_selection(
    tmp_path: Path,
) -> None:
    """Bind a closed final artifact to the exact replayed initial request."""
    initial_samples = _samples(
        _uniform_durations(
            {
                512: (9.8, 9.8),
                1024: (10.0, 10.0),
                2048: (9.5, 9.5),
                4096: (8.5, 8.5),
            }
        )
    )
    initial_selection = select_kernel_width(initial_samples)
    initial_path = tmp_path / "initial"
    initial_manifest, initial_selection_sha = _seal_artifact(
        initial_path,
        samples=initial_samples,
        kernel_block_widths=KERNEL_BLOCK_WIDTHS,
        selection=initial_selection,
        campaign_kind="initial",
        lineage=None,
    )

    receipt = authenticate_initial_extension_artifact(initial_path)

    assert receipt.manifest_sha256 == initial_manifest
    assert receipt.selection_sha256 == initial_selection_sha
    assert receipt.selected_kernel_block_bytes == 4096
    assert receipt.extension_kernel_block_bytes == 8192

    lineage: dict[str, object] = {
        "initial_manifest_sha256": receipt.manifest_sha256,
        "initial_selection_sha256": receipt.selection_sha256,
        "initial_selected_kernel_block_bytes": (
            receipt.selected_kernel_block_bytes
        ),
        "extension_kernel_block_bytes": receipt.extension_kernel_block_bytes,
    }
    widths = extension_kernel_block_widths(8192)
    final_samples = _samples(
        _uniform_durations(
            {
                512: (9.8, 9.8),
                1024: (10.0, 10.0),
                2048: (9.5, 9.5),
                4096: (8.8, 8.8),
                8192: (8.0, 8.0),
            }
        ),
        widths,
    )
    final_selection = select_kernel_width(
        final_samples,
        widths,
        permit_boundary_extension=False,
    )
    final_path = tmp_path / "final"
    _seal_artifact(
        final_path,
        samples=final_samples,
        kernel_block_widths=widths,
        selection=final_selection,
        campaign_kind="boundary_extension",
        lineage=lineage,
    )

    replayed = validate_campaign_artifact(final_path, initial_path)

    assert replayed == final_selection


@pytest.mark.cpu_test
def test_extension_rejects_a_different_explicit_request_before_cuda(
    tmp_path: Path,
) -> None:
    """Never create output for a request not authorized by initial evidence."""
    initial_samples = _samples(
        _uniform_durations(
            {
                512: (9.8, 9.8),
                1024: (10.0, 10.0),
                2048: (9.5, 9.5),
                4096: (8.5, 8.5),
            }
        )
    )
    initial_selection = select_kernel_width(initial_samples)
    initial_path = tmp_path / "initial"
    _seal_artifact(
        initial_path,
        samples=initial_samples,
        kernel_block_widths=KERNEL_BLOCK_WIDTHS,
        selection=initial_selection,
        campaign_kind="initial",
        lineage=None,
    )
    output_path = tmp_path / "final"

    with pytest.raises(ValueError, match="explicit extension width differs"):
        run_extension_campaign(
            initial_artifact_directory=initial_path,
            extension_kernel_block_bytes=256,
            artifact_directory=output_path,
        )

    assert output_path.exists() is False
