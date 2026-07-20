from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from tools.gemma4_pd.nixl_micro_rig import launcher
from tools.gemma4_pd.nixl_micro_rig.config import RigConfig, load_config
from tools.gemma4_pd.nixl_micro_rig.data import (
    fill_source_rows,
    verify_destination_rows,
)
from tools.gemma4_pd.nixl_micro_rig.geometry import TransferPlan, build_plan
from tools.gemma4_pd.nixl_micro_rig.target2_gate import (
    C1_OBSERVED_REGIME,
    C64_OBSERVED_REGIME,
    CONTIGUOUS_REGIME,
    BufferSlotLease,
    FragmentationRegime,
    GateArm,
    GateCase,
    GateConfigurationError,
    SlotState,
    _target2_decision,
    attest_cuda_ipc_write_protocol,
    build_gate_packed_plan,
    build_gate_plan,
    build_gate_request_plans,
    build_gate_source_plan,
    describe_gate_case,
    focused_write_conformance_plan,
    gate_matrix,
    packed_descriptors,
    summarize_gate_records,
    verify_packed_chunk,
)
from vllm.distributed.kv_transfer.coalesced_layout import rank_major_slot_base
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.coalesced_pack import (
    pack_chunk_reference,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.coalesced_scatter import (
    scatter_packed_chunk_reference,
)

CONFIG_PATH = (
    Path(__file__).parents[2]
    / "tools"
    / "gemma4_pd"
    / "nixl_micro_rig"
    / "gemma4_tp4_to_tp1_2k.json"
)


def _tiny_plan() -> tuple[RigConfig, TransferPlan]:
    config = load_config(CONFIG_PATH)
    region = replace(config.regions[0], name="tiny", row_bytes=16)
    group = replace(
        config.groups[0],
        index=0,
        remote_position_count=4,
        local_position_count=4,
        owned_region_indices=(0,),
    )
    scenario = replace(
        config.scenarios[0],
        name="tiny",
        source_start_block=2,
        source_end_block=6,
        run_count=2,
        iterations=1,
        staging_offsets_mib=(0,),
        replay_manifest=None,
    )
    tiny = replace(
        config,
        source_block_count=16,
        valid_token_extent=4,
        staging_capacity_mib=1,
        regions=(region,),
        groups=(group,),
        scenarios=(scenario,),
    )
    return tiny, build_plan(tiny, scenario)


@pytest.mark.parametrize(
    ("regime", "expected_descriptors"),
    [
        (CONTIGUOUS_REGIME, 10),
        (C1_OBSERVED_REGIME, 625),
        (C64_OBSERVED_REGIME, 2120),
    ],
)
def test_exact_2k_fragmentation_regimes_are_calibrated(
    regime: FragmentationRegime, expected_descriptors: int
) -> None:
    config = load_config(CONFIG_PATH)
    plan = build_gate_plan(config, config.scenarios[0], regime)

    assert plan.logical_position_count == 803
    assert plan.staging_bytes == 1_052_508_160
    assert plan.descriptors_per_handle == expected_descriptors


@pytest.mark.parametrize(
    ("chunk_mib", "expected_chunks", "expected_descriptors"),
    [(64, 4, 16), (128, 2, 8), (256, 1, 4), (512, 1, 4)],
)
def test_packed_layout_is_bounded_lossless_and_contiguous(
    chunk_mib: int, expected_chunks: int, expected_descriptors: int
) -> None:
    config = load_config(CONFIG_PATH)
    plan = build_gate_plan(config, config.scenarios[0], C64_OBSERVED_REGIME)
    chunk_bytes = chunk_mib * 1024 * 1024
    layout = build_gate_packed_plan(plan, chunk_bytes)

    assert layout.rank_stride_bytes == 263_127_040
    assert layout.transfer_size_bytes == 1_052_508_160
    assert len(layout.chunks) == expected_chunks
    assert len(layout.chunks) * plan.rank_count == expected_descriptors
    assert all(chunk.rank_stride_bytes <= chunk_bytes for chunk in layout.chunks)
    assert [chunk.chunk_index for chunk in layout.chunks] == list(
        range(expected_chunks)
    )

    covered = [
        (region_slice.region_index, position_index)
        for chunk in layout.chunks
        for region_slice in chunk.region_slices
        for position_index in range(
            region_slice.position_start,
            region_slice.position_start + region_slice.position_count,
        )
    ]
    expected = [
        (region_index, position_index)
        for region_index, region in enumerate(plan.transport.regions)
        for position_index in range(len(region.positions))
    ]
    assert covered == expected


def test_in_flight_request_plans_use_disjoint_source_rows() -> None:
    config = load_config(CONFIG_PATH)
    plans = build_gate_request_plans(
        config,
        config.scenarios[0],
        C64_OBSERVED_REGIME,
        in_flight_depth=8,
    )

    source_sets = [
        {
            position.remote_block_id
            for region in plan.transport.regions
            for position in region.positions
        }
        for plan in plans
    ]
    assert all(len(source_ids) == 803 for source_ids in source_sets)
    assert sum(len(source_ids) for source_ids in source_sets) == len(
        set().union(*source_sets)
    )
    assert all(plan.staging_bytes == 1_052_508_160 for plan in plans)
    assert all(plan.descriptors_per_handle == 2120 for plan in plans)


def test_cpu_pack_verify_and_chunk_scatter_preserve_exact_payload() -> None:
    config, plan = _tiny_plan()
    chunk_bytes = 32
    layout = build_gate_packed_plan(plan, chunk_bytes)
    source_plan = build_gate_source_plan(plan)
    iteration = 7
    sources: list[tuple[torch.Tensor, ...]] = []
    destination = torch.zeros(
        config.source_block_count * plan.rank_count * config.regions[0].row_bytes,
        dtype=torch.uint8,
    )

    for chunk in layout.chunks:
        packed = torch.full((plan.rank_count * chunk_bytes,), 0xA5, dtype=torch.uint8)
        for source_rank in range(plan.rank_count):
            source_regions = (
                torch.zeros(
                    config.source_block_count * config.regions[0].row_bytes,
                    dtype=torch.uint8,
                ),
            )
            fill_source_rows(
                config=config,
                plan=plan,
                source_rank=source_rank,
                iteration=iteration,
                regions=source_regions,
            )
            sources.append(source_regions)
            pack_chunk_reference(
                packed,
                source_regions,
                source_plan,
                layout,
                chunk.chunk_index,
                destination_base_offset_bytes=source_rank * chunk_bytes,
            )
            verify_packed_chunk(
                plan=plan,
                packed_plan=layout,
                chunk_index=chunk.chunk_index,
                source_rank=source_rank,
                iteration=iteration,
                packed=packed,
                packed_base_offset_bytes=source_rank * chunk_bytes,
            )
            guard_start = source_rank * chunk_bytes + chunk.rank_stride_bytes
            guard_end = (source_rank + 1) * chunk_bytes
            assert torch.all(packed[guard_start:guard_end] == 0xA5)
        scatter_packed_chunk_reference(
            packed,
            (destination,),
            plan.transport,
            layout,
            chunk.chunk_index,
            staging_base_offset_bytes=0,
            staging_rank_stride_bytes=chunk_bytes,
        )

    verify_destination_rows(
        config=config,
        plan=plan,
        iteration=iteration,
        destinations=(destination,),
    )
    assert len(sources) == plan.rank_count * len(layout.chunks)


def test_packed_descriptor_pair_has_one_exact_extent() -> None:
    _, plan = _tiny_plan()
    chunk = build_gate_packed_plan(plan, chunk_bytes=32).chunks[-1]

    local, remote = packed_descriptors(
        local_base=1000,
        local_device=0,
        remote_base=2000,
        remote_device=3,
        chunk=chunk,
    )

    assert local == [(1000, chunk.rank_stride_bytes, 0)]
    assert remote == [(2000, chunk.rank_stride_bytes, 3)]


def test_buffer_slot_requires_native_and_device_quiescence_before_reuse() -> None:
    slot = BufferSlotLease(slot_index=1)
    generation = slot.begin_device_write(request_index=3, chunk_index=2)
    assert generation == 1
    slot.begin_native_transfer()
    with pytest.raises(RuntimeError, match="native_transfer"):
        slot.begin_device_write(request_index=4, chunk_index=0)
    slot.begin_device_read()
    with pytest.raises(RuntimeError, match="device_reader"):
        slot.begin_device_write(request_index=4, chunk_index=0)
    slot.release_after_device()
    assert slot.state is SlotState.FREE
    assert slot.begin_device_write(request_index=4, chunk_index=0) == 2


def test_buffer_slot_tombstone_is_permanent() -> None:
    slot = BufferSlotLease(slot_index=0)
    slot.begin_device_write(request_index=0, chunk_index=0)
    slot.begin_native_transfer()
    slot.tombstone("native completion became ambiguous")

    assert slot.state is SlotState.TOMBSTONED
    with pytest.raises(RuntimeError, match="tombstoned"):
        slot.begin_device_write(request_index=1, chunk_index=0)


def test_buffer_slot_native_writer_requires_device_quiescence_before_reuse() -> None:
    slot = BufferSlotLease(slot_index=0)
    assert slot.begin_native_write(request_index=0, chunk_index=0) == 1
    with pytest.raises(RuntimeError, match="native_transfer"):
        slot.begin_native_write(request_index=1, chunk_index=0)
    slot.begin_device_read()
    slot.release_after_device()
    assert slot.begin_native_write(request_index=1, chunk_index=0) == 2


def test_gate_matrix_keeps_direct_control_and_all_packed_directions() -> None:
    cases = gate_matrix(
        regimes=(C64_OBSERVED_REGIME,),
        chunk_mib=(64, 256),
        in_flight_depths=(1, 8),
        warmup_batches=2,
        measured_batches=5,
    )

    assert len(cases) == 10
    assert sum(case.arm is GateArm.DIRECT_READ for case in cases) == 2
    assert sum(case.arm is GateArm.PACKED_READ for case in cases) == 4
    assert sum(case.arm is GateArm.PACKED_WRITE for case in cases) == 4
    assert all(case.warmup_batches == 2 for case in cases)
    assert all(case.measured_batches == 5 for case in cases)


def test_gate_case_rejects_ambiguous_chunk_semantics() -> None:
    with pytest.raises(GateConfigurationError, match="direct_read"):
        GateCase(
            arm=GateArm.DIRECT_READ,
            fragmentation=CONTIGUOUS_REGIME,
            chunk_bytes=64 * 1024 * 1024,
            in_flight_depth=1,
        )
    with pytest.raises(GateConfigurationError, match="64, 128, 256, or 512"):
        GateCase(
            arm=GateArm.PACKED_READ,
            fragmentation=CONTIGUOUS_REGIME,
            chunk_bytes=96 * 1024 * 1024,
            in_flight_depth=1,
        )


def test_gate_description_reports_memory_and_descriptor_reduction() -> None:
    config = load_config(CONFIG_PATH)
    case = GateCase(
        arm=GateArm.PACKED_READ,
        fragmentation=C64_OBSERVED_REGIME,
        chunk_bytes=128 * 1024 * 1024,
        in_flight_depth=8,
    )
    description = describe_gate_case(config, config.scenarios[0], case)

    assert description["request_bytes"] == 1_052_508_160
    assert description["direct_descriptors_per_request"] == 8480
    assert description["packed_descriptors_per_request"] == 8
    assert description["consumer_receive_bytes"] == 1024 * 1024 * 1024


def test_target2_preflight_accounts_for_every_guarded_pool_slot() -> None:
    config = load_config(CONFIG_PATH)
    maximum_chunk_bytes = 512 * 1024 * 1024
    additional = launcher._target2_additional_memory(config, maximum_chunk_bytes)
    guarded_slot_bytes = maximum_chunk_bytes + 2 * 1024 * 1024

    assert all(
        additional[device] == 2 * guarded_slot_bytes
        for device in config.producer_devices
    )
    consumer_slot_bytes = 4 * maximum_chunk_bytes + 2 * 1024 * 1024
    assert additional[config.consumer_device] == 2 * consumer_slot_bytes


def test_rank_major_slot_base_uses_the_active_candidate_stride() -> None:
    slot_base = 0x1000_0000_0000

    assert rank_major_slot_base(slot_base, 0, 64 * 1024 * 1024) == slot_base
    assert rank_major_slot_base(slot_base, 1, 64 * 1024 * 1024) == (
        slot_base + 64 * 1024 * 1024
    )
    assert rank_major_slot_base(slot_base, 1, 512 * 1024 * 1024) == (
        slot_base + 512 * 1024 * 1024
    )


def test_focused_write_conformance_does_not_reopen_candidate_selection() -> None:
    config = load_config(CONFIG_PATH)

    plan = focused_write_conformance_plan(config, config.scenarios[0])

    assert plan["selection_rerun"] is False
    assert plan["selected_candidate"] == {
        "arm": GateArm.PACKED_WRITE.value,
        "chunk_bytes_per_rank": 256 * 1024 * 1024,
        "source_tp_size": 4,
        "decoder_count": 2,
        "producer_slot_count": 2,
        "consumer_slot_count": 2,
        "alignment_bytes": 256,
    }
    cases = plan["cases"]
    assert isinstance(cases, list)
    exact, tail = cases
    assert exact["rank_bytes"] == 263_127_040
    assert exact["chunk_payload_bytes_per_rank"] == [263_127_040]
    assert exact["decoder_route"] == [0, 1, 0, 1, 0, 1]
    assert tail["rank_bytes"] == 2 * 256 * 1024 * 1024 + 64 * 1024
    assert tail["chunk_payload_bytes_per_rank"] == [
        256 * 1024 * 1024,
        256 * 1024 * 1024,
        64 * 1024,
    ]
    assert tail["chunk_count"] == 3
    assert tail["decoder_route"] == [0, 1, 0]
    assert all(case["producer_slot_reuse_proved"] for case in cases)
    assert all(case["consumer_slot_reuse_proved"] for case in cases)
    assert all(
        task["descriptor_count"] == 1 for case in cases for task in case["tasks"]
    )


def test_cuda_ipc_write_attestation_requires_every_producer_log(
    tmp_path: Path,
) -> None:
    protocol_log = (
        "[1784556793.091727] [host:286911:0] | ucp_context_0 intra-node cfg#2 | "
        "remote memory write by ucp_put*(multi) from cuda/GPU0 to cuda/dev[0] |\n"
        "[1784556793.091728] [host:286911:0] "
        "+----------------+----------------+----------+\n"
        "[1784556793.091729] [host:286911:0] | 0..inf | zero-copy | "
        "cuda_ipc/cuda |\n"
    )
    for rank in range(4):
        (tmp_path / f"producer-{rank}.log").write_text(protocol_log)

    evidence = attest_cuda_ipc_write_protocol(tmp_path, producer_count=4)

    assert evidence["operation"] == "remote_memory_write"
    assert evidence["source_memory"] == evidence["destination_memory"] == "cuda"
    assert evidence["protocol"] == "cuda_ipc/cuda"
    producers = evidence["producers"]
    assert isinstance(producers, list)
    assert [producer["rank"] for producer in producers] == [0, 1, 2, 3]

    (tmp_path / "producer-2.log").write_text(
        "remote memory write from host to host\n0..inf | zero-copy | tcp/lo\n"
    )
    with pytest.raises(RuntimeError, match="producer 2 did not prove"):
        attest_cuda_ipc_write_protocol(tmp_path, producer_count=4)


def test_spawn_environment_prebinds_gpu_runtime_and_restores_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spawn bootstrap inherits the role contract without mutating its parent."""
    config = load_config(CONFIG_PATH)
    device_uuids = {device: f"GPU-{device}" for device in range(8)}
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "parent-visible")
    monkeypatch.setenv("UCX_TLS", "parent-ucx")
    monkeypatch.delenv("UCX_RNDV_SCHEME", raising=False)

    with launcher._arm_spawn_environment(
        config,
        "cuda_ipc",
        "consumer",
        device_uuids,
    ) as visibility:
        assert visibility == device_uuids[config.consumer_device]
        assert launcher.os.environ["CUDA_VISIBLE_DEVICES"] == visibility
        assert launcher.os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
        assert (
            launcher.os.environ["UCX_TLS"] == config.transport_arm("cuda_ipc").ucx_tls
        )

    assert launcher.os.environ["CUDA_VISIBLE_DEVICES"] == "parent-visible"
    assert launcher.os.environ["UCX_TLS"] == "parent-ucx"
    assert "UCX_RNDV_SCHEME" not in launcher.os.environ


def test_target2_processes_start_inside_role_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The child OS process inherits CUDA state before spawn imports begin."""
    config = load_config(CONFIG_PATH)
    device_uuids = {device: f"GPU-{device}" for device in range(8)}
    started: list[tuple[str, dict[str, str]]] = []

    class FakeProcess:
        """Minimal successful spawn-process surface."""

        exitcode: int | None

        def __init__(
            self,
            *,
            target: Callable[..., object],
            name: str,
            args: tuple[object, ...],
        ) -> None:
            self.exitcode = None
            self.name = name
            self.target = target
            self.args = args

        def start(self) -> None:
            started.append((self.name, dict(launcher.os.environ)))
            self.exitcode = 0

        def join(self, timeout: float | None = None) -> None:
            del timeout

        def is_alive(self) -> bool:
            return False

        def terminate(self) -> None:
            raise AssertionError("a successful fake process must not terminate")

        def kill(self) -> None:
            raise AssertionError("a successful fake process must not be killed")

    class FakeContext:
        """Construct fake processes with the spawn-compatible signature."""

        def Process(
            self,
            *,
            target: Callable[..., object],
            name: str,
            args: tuple[object, ...],
        ) -> FakeProcess:
            return FakeProcess(target=target, name=name, args=args)

    monkeypatch.setattr(
        launcher.multiprocessing,
        "get_context",
        lambda method: FakeContext() if method == "spawn" else None,
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "parent-visible")
    monkeypatch.setenv("UCX_TLS", "parent-ucx")

    launcher._run_target2_arm(
        config_path=CONFIG_PATH,
        config=config,
        arm_name="cuda_ipc",
        run_id="spawn-environment-test",
        case_specs=(),
        artifact_directory=tmp_path,
        device_uuids=device_uuids,
    )

    assert len(started) == 5
    producer_visibility = ",".join(
        device_uuids[device] for device in config.producer_devices
    )
    for name, environment in started:
        expected = (
            device_uuids[config.consumer_device]
            if name == "target2-gate-d0"
            else producer_visibility
        )
        assert environment["CUDA_VISIBLE_DEVICES"] == expected
        assert environment["UCX_TLS"] == config.transport_arm("cuda_ipc").ucx_tls
    assert launcher.os.environ["CUDA_VISIBLE_DEVICES"] == "parent-visible"
    assert launcher.os.environ["UCX_TLS"] == "parent-ucx"


def test_target2_source_identity_binds_gate_and_production_primitives() -> None:
    identity = launcher._target2_source_identity()
    records = identity["files"]
    relative_paths = {record["path"] for record in records}

    assert len(identity["aggregate_sha256"]) == 64
    assert "tools/gemma4_pd/nixl_micro_rig/target2_gate_roles.py" in relative_paths
    assert (
        "vllm/distributed/kv_transfer/kv_connector/v1/nixl/coalesced_pack.py"
        in relative_paths
    )
    assert (
        "vllm/distributed/kv_transfer/kv_connector/v1/nixl/coalesced_scatter.py"
        in relative_paths
    )


def test_target2_postflight_preserves_protected_processes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(CONFIG_PATH)
    uuids = {device: f"GPU-{device}" for device in range(8)}
    protected = {
        5: [],
        6: [
            {
                "gpu": 6,
                "gpu_uuid": uuids[6],
                "pid": 600,
                "name": "protected-6",
            }
        ],
        7: [
            {
                "gpu": 7,
                "gpu_uuid": uuids[7],
                "pid": 700,
                "name": "protected-7",
            }
        ],
    }
    preflight = {
        "inventory": {
            device: {"uuid": gpu_uuid, "free_mib": 1000}
            for device, gpu_uuid in uuids.items()
        },
        "protected_compute_processes": protected,
    }
    compute_rows = [
        [uuids[6], "600", "protected-6"],
        [uuids[7], "700", "protected-7"],
    ]
    monkeypatch.setattr(
        launcher,
        "_nvidia_smi",
        lambda *fields, compute_apps=False: compute_rows if compute_apps else [],
    )

    result = launcher._target2_postflight(config, preflight)

    assert result["selected_devices_clear"] == [0, 1, 2, 3, 4]
    assert result["protected_compute_processes"] == protected


def test_gate_summary_requires_complete_integrity_and_compares_to_direct() -> None:
    config = load_config(CONFIG_PATH)
    cases = (
        GateCase(
            arm=GateArm.DIRECT_READ,
            fragmentation=C64_OBSERVED_REGIME,
            chunk_bytes=0,
            in_flight_depth=1,
            warmup_batches=0,
            measured_batches=1,
        ),
        GateCase(
            arm=GateArm.PACKED_READ,
            fragmentation=C64_OBSERVED_REGIME,
            chunk_bytes=64 * 1024 * 1024,
            in_flight_depth=1,
            warmup_batches=0,
            measured_batches=1,
        ),
    )
    plan = build_gate_plan(config, config.scenarios[0], C64_OBSERVED_REGIME)
    chunks = build_gate_packed_plan(plan, 64 * 1024 * 1024).chunks
    direct_native = [
        {
            "backend": "UCX",
            "start_time_us": 1,
            "post_duration_us": 2,
            "transfer_duration_us": 100,
            "total_bytes": plan.transport.rank_stride_bytes,
            "descriptor_count": 2120,
            "request_index": 0,
            "rank": rank,
        }
        for rank in range(4)
    ]
    packed_native = [
        {
            "backend": "UCX",
            "start_time_us": 1,
            "post_duration_us": 2,
            "transfer_duration_us": 50,
            "total_bytes": chunk.rank_stride_bytes,
            "descriptor_count": 1,
            "request_index": 0,
            "chunk_index": chunk.chunk_index,
            "rank": rank,
        }
        for chunk in chunks
        for rank in range(4)
    ]
    common = {
        "fragmentation": C64_OBSERVED_REGIME.name,
        "source_run_count": C64_OBSERVED_REGIME.run_count,
        "direct_descriptors_per_rank": 2120,
        "in_flight_depth": 1,
        "batch_index": 0,
        "measured": True,
        "payload_iterations": [0],
        "request_bytes": 1_052_508_160,
        "total_batch_bytes": 1_052_508_160,
    }
    records = [
        {
            **common,
            "case_index": 0,
            "case_name": cases[0].name,
            "arm": GateArm.DIRECT_READ.value,
            "chunk_bytes": 0,
            "elapsed_seconds": 1.0,
            "native_handles": direct_native,
            "pack_gpu_ms": [],
            "scatter_gpu_ms": [1.0],
            "packed_window_limit": 0,
            "maximum_active_packed_tasks": 0,
            "integrity_mode": "all_staging_and_final_scatter",
            "final_slot_states": ["free"],
        },
        {
            **common,
            "case_index": 1,
            "case_name": cases[1].name,
            "arm": GateArm.PACKED_READ.value,
            "chunk_bytes": 64 * 1024 * 1024,
            "elapsed_seconds": 0.5,
            "native_handles": packed_native,
            "pack_gpu_ms": [
                {
                    "request_index": 0,
                    "chunk_index": chunk.chunk_index,
                    "rank": rank,
                    "duration_ms": 0.1,
                }
                for chunk in chunks
                for rank in range(4)
            ],
            "scatter_gpu_ms": [
                {
                    "request_index": 0,
                    "chunk_index": chunk.chunk_index,
                    "rank": rank,
                    "duration_ms": 0.2,
                }
                for chunk in chunks
                for rank in range(4)
            ],
            "packed_window_limit": 2,
            "maximum_active_packed_tasks": 2,
            "integrity_mode": "final_request",
            "final_slot_states": ["free"] * 8,
        },
    ]

    summary = summarize_gate_records(cases, records)

    assert summary["integrity_verdict"] == (
        "exact_warmups_and_measured_wire_final_destination"
    )
    assert summary["target2_decision"]["verdict"] == ("not_evaluable_incomplete_matrix")
    comparisons = summary["direct_relative_comparisons"]
    assert isinstance(comparisons, list)
    assert comparisons[0]["direct_over_packed_speed_ratio"] == 2.0

    records[1]["final_slot_states"][0] = "native_transfer"
    with pytest.raises(RuntimeError, match="retained a live slot"):
        summarize_gate_records(cases, records)


def test_target2_decision_uses_activation_regime_and_bounded_tie_break() -> None:
    depths = (1, 2, 4, 8)
    summaries: list[dict[str, object]] = [
        {
            "arm": GateArm.DIRECT_READ.value,
            "fragmentation": C64_OBSERVED_REGIME.name,
            "in_flight_depth": depth,
            "chunk_bytes": 0,
            "median_elapsed_seconds": 1.0,
        }
        for depth in depths
    ]
    elapsed_by_candidate = {
        (GateArm.PACKED_READ.value, 64): 0.95,
        (GateArm.PACKED_READ.value, 128): 0.804,
        (GateArm.PACKED_READ.value, 256): 0.8,
        (GateArm.PACKED_READ.value, 512): 0.81,
        (GateArm.PACKED_WRITE.value, 64): 0.92,
        (GateArm.PACKED_WRITE.value, 128): 0.9,
        (GateArm.PACKED_WRITE.value, 256): 0.88,
        (GateArm.PACKED_WRITE.value, 512): 0.89,
    }
    for (arm, chunk_mib), elapsed in elapsed_by_candidate.items():
        summaries.extend(
            {
                "arm": arm,
                "fragmentation": C64_OBSERVED_REGIME.name,
                "in_flight_depth": depth,
                "chunk_bytes": chunk_mib * 1024 * 1024,
                "median_elapsed_seconds": elapsed,
            }
            for depth in depths
        )

    decision = _target2_decision(summaries)

    assert decision["verdict"] == "pass"
    selected = decision["selected_candidate"]
    assert selected["arm"] == GateArm.PACKED_READ.value
    assert selected["chunk_bytes"] == 128 * 1024 * 1024
