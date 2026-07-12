import ast
import json
import socket
import struct
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from tools.gemma4_pd.nixl_micro_rig import launcher
from tools.gemma4_pd.nixl_micro_rig.config import (
    ConfigError,
    RigConfig,
    load_config,
)
from tools.gemma4_pd.nixl_micro_rig.data import (
    compute_rig_semantic_contract_digest,
    expected_plane_bytes,
    fill_destination_canary_rows,
    verify_destination_canary_rows,
)
from tools.gemma4_pd.nixl_micro_rig.geometry import (
    ReplayGroup,
    apply_group_prefix_trim,
    build_configured_plan,
    build_plan,
    describe_plan,
)
from tools.gemma4_pd.nixl_micro_rig.integrity_check import (
    integrity_contract_self_test,
)
from tools.gemma4_pd.nixl_micro_rig.ownership import (
    HandleState,
    StagingGeneration,
    StagingRangeAllocator,
)
from tools.gemma4_pd.nixl_micro_rig.protocol import (
    JsonChannel,
    PreparePayload,
    ProtocolError,
    SourcePostPayload,
)
from tools.gemma4_pd.nixl_micro_rig.roles import (
    _prepared_descriptors,
    _staging_guard_ranges,
)

CONFIG_PATH = (
    Path(__file__).parents[2]
    / "tools"
    / "gemma4_pd"
    / "nixl_micro_rig"
    / "gemma4_tp4_to_tp1.json"
)


def test_default_geometry_matches_captured_storm37b_handshakes() -> None:
    config = load_config(CONFIG_PATH)
    scenario = config.scenario("starts_at_2gib")
    plan = build_plan(config, scenario)

    assert plan.position_count == 2672
    assert config.valid_token_extent == 16240
    assert tuple(group.token_capacity for group in config.groups) == (16,) * 13
    assert plan.rank_count == 4
    assert plan.staging_bytes == 7_004_487_680
    assert plan.staging_bytes // (1024 * 1024) == 6680
    assert plan.source_registration_bytes_per_rank == 41_943_040_000
    assert plan.destination_registration_bytes == 167_772_160_000
    assert plan.region_offsets == tuple(
        region_index * 700_448_768 for region_index in range(10)
    )
    assert plan.replay_manifest is None


def test_semantic_contract_binds_group_region_and_ownership() -> None:
    config = load_config(CONFIG_PATH)

    owned = compute_rig_semantic_contract_digest(config, 0, 0)
    other_group = compute_rig_semantic_contract_digest(config, 1, 0)
    transport_only = compute_rig_semantic_contract_digest(config, 12, 0)
    other_region = compute_rig_semantic_contract_digest(config, 0, 1)

    assert len(owned) == 32
    assert len({owned, other_group, transport_only, other_region}) == 4


@pytest.mark.parametrize(
    ("scenario_name", "run_count", "descriptor_count", "split", "last_block"),
    [
        ("ends_before_2gib", 1, 10, False, 32767),
        ("starts_at_2gib", 1, 10, False, 35439),
        ("starts_after_2gib", 1, 10, False, 35440),
        ("straddles_2gib", 1, 10, False, 34103),
        ("observed_high_contiguous", 1, 10, False, 54641),
        ("observed_ceiling_102_runs", 102, 1020, False, 54641),
        ("split_boundary_103_runs", 103, 1030, True, 54641),
        ("observed_104_runs", 104, 1040, True, 54641),
        ("low_split_boundary", 103, 1030, True, 3797),
    ],
)
def test_run_split_and_source_extent_geometry(
    scenario_name: str,
    run_count: int,
    descriptor_count: int,
    split: bool,
    last_block: int,
) -> None:
    config = load_config(CONFIG_PATH)
    description = describe_plan(config, config.scenario(scenario_name))

    assert description["run_count"] == run_count
    assert description["descriptors_per_handle"] == descriptor_count
    assert description["crosses_nixl_1024_descriptor_split"] is split
    assert description["last_source_block"] == last_block


def test_group_prefix_trim_keeps_remote_suffix_and_original_local_pairing() -> None:
    config = load_config(CONFIG_PATH)
    groups = list(config.groups)
    groups[0] = replace(groups[0], remote_position_count=66)
    config = replace(config, groups=tuple(groups))
    replay: list[ReplayGroup] = []
    local_cursor = 0
    for group in config.groups:
        local_ids = tuple(
            range(local_cursor, local_cursor + group.local_position_count)
        )
        suffix = tuple(range(1000 + local_cursor, 1000 + local_cursor + len(local_ids)))
        prefix = (17, 19) if group.index == 0 else ()
        replay.append(
            ReplayGroup(
                group_index=group.index,
                remote_block_ids=prefix + suffix,
                local_block_ids=local_ids,
            )
        )
        local_cursor += len(local_ids)

    pairings = apply_group_prefix_trim(config, tuple(replay))

    assert pairings[0].remote_block_id == 1000
    assert pairings[0].local_block_id == 0
    assert pairings[0].source_position == 2
    assert all(pair.remote_block_id not in {17, 19} for pair in pairings)


def test_default_replay_exercises_stable_sort_and_high_scatter_rows() -> None:
    config = load_config(CONFIG_PATH)
    plan = build_plan(config, config.scenario("observed_high_contiguous"))

    assert plan.original_pairings != plan.sorted_pairings
    assert min(plan.local_block_ids) >= 49152
    original_pairs = {
        pair.remote_block_id: pair.local_block_id for pair in plan.original_pairings
    }
    assert all(
        original_pairs[pair.remote_block_id] == pair.local_block_id
        for pair in plan.sorted_pairings
    )


def test_duplicate_remote_ids_across_groups_retain_stable_pairing_order() -> None:
    config = load_config(CONFIG_PATH)
    scenario = config.scenario("starts_at_2gib")
    base = build_plan(config, scenario)
    by_group: list[list] = [[] for _ in config.groups]
    for pair in base.original_pairings:
        by_group[pair.group_index].append(pair)
    duplicate_id = by_group[0][0].remote_block_id
    group_one_remote = [pair.remote_block_id for pair in by_group[1]]
    group_one_remote[0] = duplicate_id
    replay = tuple(
        ReplayGroup(
            group_index=group.index,
            remote_block_ids=tuple(
                group_one_remote
                if group.index == 1
                else [pair.remote_block_id for pair in by_group[group.index]]
            ),
            local_block_ids=tuple(
                pair.local_block_id for pair in by_group[group.index]
            ),
        )
        for group in config.groups
    )
    replay_scenario = replace(scenario, replay_manifest="capture.json")
    plan = build_plan(config, replay_scenario, replay)
    duplicates = [
        pair for pair in plan.sorted_pairings if pair.remote_block_id == duplicate_id
    ]

    assert [pair.group_index for pair in duplicates] == [0, 1]
    assert duplicates[0].local_block_id != duplicates[1].local_block_id
    duplicate_positions = [
        position
        for position, pair in enumerate(plan.sorted_pairings)
        if pair.remote_block_id == duplicate_id
    ]
    common = {
        "plan": plan,
        "position_count": 1,
        "source_rank": 0,
        "region_index": 0,
        "plane_index": 0,
        "iteration": 0,
        "row_bytes": config.regions[0].row_bytes,
        "device": torch.device("cpu"),
    }
    assert torch.equal(
        expected_plane_bytes(position_start=duplicate_positions[0], **common),
        expected_plane_bytes(position_start=duplicate_positions[1], **common),
    )
    description = describe_plan(config, replay_scenario, plan)
    assert description["plan_provenance"] == "capture.json"
    assert description["run_count"] == len(plan.runs)


def test_full_prefix_hit_is_zero_byte_and_non_evidentiary() -> None:
    config = load_config(CONFIG_PATH)
    zero_groups = tuple(
        replace(group, local_position_count=0) for group in config.groups
    )
    zero_scenario = replace(
        config.scenarios[0],
        name="full_prefix",
        source_start_block=0,
        source_end_block=0,
        run_count=0,
        iterations=1,
        staging_offsets_mib=(0,),
    )
    config = replace(config, groups=zero_groups, scenarios=(zero_scenario,))
    description = describe_plan(config, zero_scenario)

    assert description["position_count"] == 0
    assert description["staging_bytes"] == 0
    assert description["has_content_evidence"] is False
    assert description["first_source_block"] is None
    assert description["last_source_block"] is None


def test_configured_replay_is_loaded_for_plan_and_description(tmp_path: Path) -> None:
    config = load_config(CONFIG_PATH)
    scenario = config.scenario("starts_at_2gib")
    synthetic = build_plan(config, scenario)
    by_group: list[list] = [[] for _ in config.groups]
    for pairing in synthetic.original_pairings:
        by_group[pairing.group_index].append(pairing)
    replay_groups = []
    replay = []
    for group in config.groups:
        suffix = tuple(pair.remote_block_id for pair in by_group[group.index])
        prefix_count = group.remote_position_count - len(suffix)
        remote_ids = tuple(0 for _ in range(prefix_count)) + suffix
        local_ids = tuple(pair.local_block_id for pair in by_group[group.index])
        replay_groups.append(
            {
                "group_index": group.index,
                "remote_block_ids": list(remote_ids),
                "local_block_ids": list(local_ids),
            }
        )
        replay.append(
            ReplayGroup(
                group_index=group.index,
                remote_block_ids=remote_ids,
                local_block_ids=local_ids,
            )
        )
    replay_path = tmp_path / "request-plan.json"
    replay_path.write_text(json.dumps({"schema_version": 1, "groups": replay_groups}))
    replay_scenario = replace(scenario, replay_manifest=replay_path.name)

    configured = build_configured_plan(
        tmp_path / "config.json", config, replay_scenario
    )

    assert configured == build_plan(config, replay_scenario, tuple(replay))
    assert (
        describe_plan(config, replay_scenario, configured)["plan_provenance"]
        == replay_path.name
    )
    with pytest.raises(ConfigError, match="loaded plan"):
        describe_plan(config, replay_scenario)


def test_stable_source_sort_preserves_remote_local_pairing() -> None:
    config = load_config(CONFIG_PATH)
    scenario = config.scenario("starts_at_2gib")
    base = build_plan(config, scenario)
    by_group: list[list] = [[] for _ in config.groups]
    for pair in base.original_pairings:
        by_group[pair.group_index].append(pair)
    replay = tuple(
        ReplayGroup(
            group_index=group.index,
            remote_block_ids=tuple(
                pair.remote_block_id for pair in reversed(by_group[group.index])
            ),
            local_block_ids=tuple(
                pair.local_block_id for pair in by_group[group.index]
            ),
        )
        for group in config.groups
    )
    plan = build_plan(config, replace(scenario, replay_manifest="capture.json"), replay)
    original_map = {
        pair.remote_block_id: pair.local_block_id for pair in plan.original_pairings
    }

    assert tuple(pair.remote_block_id for pair in plan.sorted_pairings) == tuple(
        sorted(original_map)
    )
    assert all(
        original_map[pair.remote_block_id] == pair.local_block_id
        for pair in plan.sorted_pairings
    )


def test_configuration_rejects_unknown_keys() -> None:
    value = json.loads(CONFIG_PATH.read_text())
    value["surprise"] = True

    with pytest.raises(ConfigError, match="unknown keys"):
        RigConfig.from_json(value)


def test_configuration_rejects_protected_device() -> None:
    value = json.loads(CONFIG_PATH.read_text())
    value["consumer_device"] = 6

    with pytest.raises(ConfigError, match="hard-limited"):
        RigConfig.from_json(value)


def test_configuration_cannot_remove_immutable_gpu_denylist() -> None:
    value = json.loads(CONFIG_PATH.read_text())
    value["protected_devices"] = []

    with pytest.raises(ConfigError, match="immutable"):
        RigConfig.from_json(value)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_handshake_manifest", "../capture.json", "relative path"),
        ("source_handshake_sha256", "A" * 64, "lowercase SHA-256"),
        ("control_port", 65535, "exceed"),
    ],
)
def test_configuration_rejects_unsafe_evidence_inputs(
    field: str, value: object, message: str
) -> None:
    config_value = json.loads(CONFIG_PATH.read_text())
    config_value[field] = value

    with pytest.raises(ConfigError, match=message):
        RigConfig.from_json(config_value)


def test_configuration_rejects_handshake_transcription_drift(tmp_path: Path) -> None:
    value = json.loads(CONFIG_PATH.read_text())
    value["regions"][3]["row_bytes"] = 131072
    for manifest_name in (
        "storm37b-p-handshake.json",
        "storm37b-d1-handshake.json",
    ):
        (tmp_path / manifest_name).write_bytes(
            (CONFIG_PATH.parent / manifest_name).read_bytes()
        )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(value))

    with pytest.raises(ConfigError, match="block_lens"):
        load_config(config_path)


def test_integrity_contract_negative_controls() -> None:
    digests = integrity_contract_self_test()

    assert len(digests) == 4
    assert len(set(digests.values())) == 4


def test_pattern_oracle_changes_across_rank_region_plane_and_iteration() -> None:
    config = load_config(CONFIG_PATH)
    plan = build_plan(config, config.scenario("starts_at_2gib"))
    common = {
        "plan": plan,
        "position_start": 0,
        "position_count": 1,
        "row_bytes": config.regions[0].row_bytes,
        "device": torch.device("cpu"),
    }
    baseline = expected_plane_bytes(
        source_rank=0, region_index=0, plane_index=0, iteration=0, **common
    )

    for changes in (
        {"source_rank": 1, "region_index": 0, "plane_index": 0, "iteration": 0},
        {"source_rank": 0, "region_index": 1, "plane_index": 0, "iteration": 0},
        {"source_rank": 0, "region_index": 0, "plane_index": 1, "iteration": 0},
        {"source_rank": 0, "region_index": 0, "plane_index": 0, "iteration": 1},
    ):
        assert not torch.equal(baseline, expected_plane_bytes(**changes, **common))


def test_destination_and_staging_canaries_detect_out_of_target_writes() -> None:
    config = load_config(CONFIG_PATH)
    region = replace(config.regions[0], row_bytes=16)
    group = replace(
        config.groups[0],
        index=0,
        remote_position_count=2,
        local_position_count=2,
        owned_region_indices=(0,),
    )
    scenario = replace(
        config.scenarios[0],
        name="tiny",
        source_start_block=0,
        source_end_block=1,
        run_count=1,
        iterations=1,
        staging_offsets_mib=(0,),
        replay_manifest=None,
    )
    tiny = replace(
        config,
        producer_devices=(0,),
        consumer_device=1,
        source_block_count=8,
        staging_capacity_mib=1,
        regions=(region,),
        groups=(group,),
        scenarios=(scenario,),
    )
    plan = build_plan(tiny, scenario)
    destination = torch.empty(
        tiny.source_block_count * plan.rank_count * region.row_bytes,
        dtype=torch.uint8,
    )
    destinations = (destination,)
    fill_destination_canary_rows(
        config=tiny,
        plan=plan,
        destinations=destinations,
        value=0x5A,
    )
    verify_destination_canary_rows(
        config=tiny,
        plan=plan,
        destinations=destinations,
        value=0x5A,
    )
    destination[plan.local_block_ids[0] * region.row_bytes] = 0
    with pytest.raises(RuntimeError, match="before scatter"):
        verify_destination_canary_rows(
            config=tiny,
            plan=plan,
            destinations=destinations,
            value=0x5A,
        )

    assert _staging_guard_ranges(capacity=100, offset=20, size=40) == (
        (0, 20),
        (60, 100),
    )
    assert _staging_guard_ranges(capacity=100, offset=0, size=40) == ((40, 100),)


def test_prepared_descriptor_geometry_matches_tp4_to_tp1_mapping() -> None:
    local = _prepared_descriptors(
        base_addresses=[1000],
        row_bytes=[64],
        block_count=2,
        rank_count=4,
        rank=2,
        device_id=0,
        destination=True,
    )
    remote = _prepared_descriptors(
        base_addresses=[1000],
        row_bytes=[64],
        block_count=2,
        rank_count=4,
        rank=2,
        device_id=2,
        destination=False,
    )

    assert local.tolist() == [
        [1064, 32, 0],
        [1320, 32, 0],
        [1192, 32, 0],
        [1448, 32, 0],
    ]
    assert remote.tolist() == [
        [1000, 32, 2],
        [1064, 32, 2],
        [1032, 32, 2],
        [1096, 32, 2],
    ]


def test_preflight_rejects_foreign_compute_pid(monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_config(CONFIG_PATH)

    def fake_nvidia_smi(*fields: str, compute_apps: bool = False) -> list[list[str]]:
        if compute_apps:
            return [["uuid-4", "12345", "foreign-server"]]
        return [[str(index), f"uuid-{index}", "300000"] for index in range(8)]

    monkeypatch.setattr(launcher, "_nvidia_smi", fake_nvidia_smi)
    with pytest.raises(RuntimeError, match="foreign compute"):
        launcher.preflight(config)


def test_cuda_visibility_is_bound_to_preflighted_gpu_uuids() -> None:
    config = load_config(CONFIG_PATH)
    device_uuids = {device: f"GPU-authoritative-{device}" for device in range(8)}

    assert launcher._cuda_visibility(config, "producer", device_uuids) == ",".join(
        device_uuids[device] for device in config.producer_devices
    )
    assert (
        launcher._cuda_visibility(config, "consumer", device_uuids)
        == device_uuids[config.consumer_device]
    )
    del device_uuids[config.consumer_device]
    with pytest.raises(RuntimeError, match="lacks selected GPU UUID"):
        launcher._cuda_visibility(config, "consumer", device_uuids)


def test_partial_post_failure_never_reuses_live_staging_generation() -> None:
    allocator = StagingRangeAllocator(capacity=1024)
    generation = allocator.reserve(generation=7, offset=0, size=512, rank_count=4)
    assert not generation.reusable
    with pytest.raises(RuntimeError, match="still has writers"):
        generation.release()

    generation.record_posted(
        rank=0, initial_state=HandleState.DONE, native_handle_token="rank-0"
    )
    generation.record_posted(
        rank=1, initial_state=HandleState.PROC, native_handle_token="rank-1"
    )
    assert not generation.reusable
    generation.record_posted(
        rank=2, initial_state=HandleState.ERR, native_handle_token="rank-2"
    )
    generation.seal_posting()

    assert generation.operation_failed
    assert not generation.reusable
    with pytest.raises(RuntimeError, match="still has writers"):
        allocator.release(7)
    with pytest.raises(RuntimeError, match="overlaps"):
        allocator.reserve(generation=8, offset=0, size=512, rank_count=4)

    generation.update(rank=1, state=HandleState.DONE)
    assert not generation.reusable
    generation.update(rank=2, state=HandleState.CANCEL_ACK)
    assert generation.reusable
    allocator.release(7)
    assert generation.released

    allocator.reserve(generation=8, offset=0, size=512, rank_count=4)
    with pytest.raises(RuntimeError, match="late completion"):
        allocator.require_active_writer(generation=7, rank=1)


def test_post_after_sealed_failure_boundary_is_rejected() -> None:
    generation = StagingGeneration.create(generation=9, rank_count=4)
    generation.record_posted(
        rank=0, initial_state=HandleState.ERR, native_handle_token="rank-0"
    )
    generation.seal_posting()

    with pytest.raises(RuntimeError, match="after sealing"):
        generation.record_posted(
            rank=1, initial_state=HandleState.PROC, native_handle_token="rank-1"
        )


def test_out_of_range_rank_cannot_extend_generation() -> None:
    generation = StagingGeneration.create(generation=10, rank_count=4)

    with pytest.raises(ValueError, match="outside"):
        generation.record_posted(
            rank=4,
            initial_state=HandleState.PROC,
            native_handle_token="rank-4",
        )


@pytest.mark.xfail(
    strict=True,
    reason="production drops coalesced staging on first ERR while a sibling is PROC",
)
def test_current_production_partial_post_release_is_safe() -> None:
    source_path = (
        Path(__file__).parents[2]
        / "vllm"
        / "distributed"
        / "kv_transfer"
        / "kv_connector"
        / "v1"
        / "nixl"
        / "base_worker.py"
    )
    syntax = ast.parse(source_path.read_text(), filename=str(source_path))
    method_node = next(
        node
        for node in ast.walk(syntax)
        if isinstance(node, ast.FunctionDef) and node.name == "_coalesce_drop_plan"
    )
    method_node.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[method_node], type_ignores=[]))
    namespace: dict[str, object] = {"ReqId": str}
    exec(compile(module, str(source_path), "exec"), namespace)
    production_drop_plan = namespace["_coalesce_drop_plan"]

    class FakeWorker:
        def __init__(self) -> None:
            self._coalesce_plans = {"request": {"off": 128, "size": 256}}
            self.released: list[tuple[int, int]] = []

        def _staging_release(self, offset: int, size: int) -> None:
            self.released.append((offset, size))

    worker = FakeWorker()
    sibling_state = HandleState.PROC
    production_drop_plan(worker, "request")

    assert not (len(worker.released) > 0 and sibling_state is HandleState.PROC)


def test_unknown_native_submission_requires_terminal_or_cancel_ack() -> None:
    generation = StagingGeneration.create(generation=8, rank_count=4)
    generation.record_post_exception(rank=0, submission_token="submit-rank-0")
    generation.seal_posting()

    assert not generation.reusable
    with pytest.raises(RuntimeError, match="still has writers"):
        generation.release()
    generation.update(rank=0, state=HandleState.CANCEL_ACK)
    assert generation.reusable


def test_protocol_round_trip_and_versioning() -> None:
    first, second = socket.socketpair()
    kwargs = {
        "run_id": "12345678-1234-5678-1234-567812345678",
        "config_fingerprint": "a" * 64,
        "transport_arm": "cuda_copy",
        "timeout_seconds": 1.0,
    }
    with (
        JsonChannel(
            first,
            local_role="producer",
            local_rank=2,
            remote_role="consumer",
            remote_rank=0,
            **kwargs,
        ) as sender,
        JsonChannel(
            second,
            local_role="consumer",
            local_rank=0,
            remote_role="producer",
            remote_rank=2,
            **kwargs,
        ) as receiver,
    ):
        payload = PreparePayload(
            scenario="boundary",
            scenario_iteration=4,
            producer_request_id="producer-request",
            child_request_id="child-request",
            notification_id="bm90aWZpY2F0aW9u",
        )
        sender.send(payload, iteration=4)
        assert receiver.receive(PreparePayload, iteration=4) == payload


def test_protocol_rejects_missing_message_type() -> None:
    first, second = socket.socketpair()
    with pytest.raises(ProtocolError, match="run_id"):
        JsonChannel(
            first,
            run_id="not-a-uuid",
            config_fingerprint="a" * 64,
            transport_arm="cuda_copy",
            local_role="producer",
            local_rank=0,
            remote_role="consumer",
            remote_rank=0,
            timeout_seconds=1.0,
        )
    second.close()


def test_protocol_rejects_stale_iteration() -> None:
    first, second = socket.socketpair()
    kwargs = {
        "run_id": "12345678-1234-5678-1234-567812345678",
        "config_fingerprint": "a" * 64,
        "transport_arm": "cuda_copy",
        "timeout_seconds": 1.0,
    }
    with (
        JsonChannel(
            first,
            local_role="producer",
            local_rank=1,
            remote_role="consumer",
            remote_rank=0,
            **kwargs,
        ) as sender,
        JsonChannel(
            second,
            local_role="consumer",
            local_rank=0,
            remote_role="producer",
            remote_rank=1,
            **kwargs,
        ) as receiver,
    ):
        sender.send(
            SourcePostPayload(matches_pre=True, notification_seen=True),
            iteration=3,
        )
        with pytest.raises(ProtocolError, match="iteration"):
            receiver.receive(SourcePostPayload, iteration=4)


def test_protocol_rejects_duplicate_sequence() -> None:
    first, second = socket.socketpair()
    kwargs = {
        "run_id": "12345678-1234-5678-1234-567812345678",
        "config_fingerprint": "a" * 64,
        "transport_arm": "cuda_copy",
        "timeout_seconds": 1.0,
    }
    with (
        JsonChannel(
            first,
            local_role="producer",
            local_rank=1,
            remote_role="consumer",
            remote_rank=0,
            **kwargs,
        ) as sender,
        JsonChannel(
            second,
            local_role="consumer",
            local_rank=0,
            remote_role="producer",
            remote_rank=1,
            **kwargs,
        ) as receiver,
    ):
        payload = SourcePostPayload(matches_pre=True, notification_seen=True)
        sender.send(payload, iteration=0)
        receiver.receive(SourcePostPayload, iteration=0)
        sender._send_sequence = 0
        sender.send(payload, iteration=1)
        with pytest.raises(ProtocolError, match="sequence"):
            receiver.receive(SourcePostPayload, iteration=1)


def _valid_source_post_envelope() -> dict[str, object]:
    return {
        "protocol_version": 1,
        "type": "source_post",
        "run_id": "12345678-1234-5678-1234-567812345678",
        "config_fingerprint": "a" * 64,
        "transport_arm": "cuda_copy",
        "iteration": 0,
        "sender_role": "producer",
        "sender_rank": 1,
        "sequence": 0,
        "payload": {"matches_pre": True, "notification_seen": True},
    }


def _send_raw_control(connection: socket.socket, envelope: dict[str, object]) -> None:
    encoded = json.dumps(envelope, separators=(",", ":")).encode()
    connection.sendall(struct.pack("!Q", len(encoded)) + encoded)


def _consumer_channel(connection: socket.socket) -> JsonChannel:
    return JsonChannel(
        connection,
        run_id="12345678-1234-5678-1234-567812345678",
        config_fingerprint="a" * 64,
        transport_arm="cuda_copy",
        local_role="consumer",
        local_rank=0,
        remote_role="producer",
        remote_rank=1,
        timeout_seconds=0.1,
    )


@pytest.mark.parametrize(
    "field",
    ["protocol_version", "iteration", "sender_rank", "sequence"],
)
def test_protocol_rejects_boolean_integer_envelope_fields(field: str) -> None:
    first, second = socket.socketpair()
    envelope = _valid_source_post_envelope()
    envelope[field] = True
    _send_raw_control(first, envelope)
    with (
        _consumer_channel(second) as receiver,
        pytest.raises(ProtocolError, match="integer"),
    ):
        receiver.receive(SourcePostPayload, iteration=0)
    first.close()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("run_id", True, "run_id must be a string"),
        ("sender_role", 1, "role must be a string"),
        ("type", 1, "control.type must be a string"),
    ],
)
def test_protocol_rejects_wrong_envelope_value_types(
    field: str, value: object, message: str
) -> None:
    first, second = socket.socketpair()
    envelope = _valid_source_post_envelope()
    envelope[field] = value
    _send_raw_control(first, envelope)
    with (
        _consumer_channel(second) as receiver,
        pytest.raises(ProtocolError, match=message),
    ):
        receiver.receive(SourcePostPayload, iteration=0)
    first.close()


@pytest.mark.parametrize(
    ("field", "value"),
    [("matches_pre", 1), ("notification_seen", "true")],
)
def test_protocol_rejects_wrong_payload_value_types(field: str, value: object) -> None:
    first, second = socket.socketpair()
    envelope = _valid_source_post_envelope()
    payload = envelope["payload"]
    assert isinstance(payload, dict)
    payload[field] = value
    _send_raw_control(first, envelope)
    with (
        _consumer_channel(second) as receiver,
        pytest.raises(ProtocolError, match="boolean"),
    ):
        receiver.receive(SourcePostPayload, iteration=0)
    first.close()


def test_protocol_closure_reports_exact_endpoint_context() -> None:
    first, second = socket.socketpair()
    first.close()
    with (
        _consumer_channel(second) as receiver,
        pytest.raises(
            ProtocolError,
            match=("local=consumer:0 remote=producer:1 .*iteration=7 type=source_post"),
        ),
    ):
        receiver.receive(SourcePostPayload, iteration=7)
