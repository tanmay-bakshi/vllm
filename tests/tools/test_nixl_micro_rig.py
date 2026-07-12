import json
import socket
from dataclasses import replace
from pathlib import Path

import pytest

from tools.gemma4_pd.nixl_micro_rig.config import (
    ConfigError,
    RigConfig,
    load_config,
)
from tools.gemma4_pd.nixl_micro_rig.geometry import (
    ReplayGroup,
    apply_group_prefix_trim,
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
    production_drop_plan_would_release,
)
from tools.gemma4_pd.nixl_micro_rig.protocol import JsonChannel, ProtocolError

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
    assert plan.rank_count == 4
    assert plan.staging_bytes == 7_004_487_680
    assert plan.staging_bytes // (1024 * 1024) == 6680
    assert plan.source_registration_bytes_per_rank == 41_943_040_000
    assert plan.destination_registration_bytes == 167_772_160_000
    assert plan.region_offsets == tuple(
        region_index * 700_448_768 for region_index in range(10)
    )


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
    plan = build_plan(config, replace(scenario, replay_manifest="capture.json"), replay)
    duplicates = [
        pair for pair in plan.sorted_pairings if pair.remote_block_id == duplicate_id
    ]

    assert [pair.group_index for pair in duplicates] == [0, 1]
    assert duplicates[0].local_block_id != duplicates[1].local_block_id


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
    states = {
        0: HandleState.PROC,
        1: HandleState.ERR,
        2: HandleState.UNPOSTED,
        3: HandleState.UNPOSTED,
    }
    production_releases = production_drop_plan_would_release(states)
    sibling_can_still_write = states[0] is HandleState.PROC

    assert not (production_releases and sibling_can_still_write)


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
    with JsonChannel(first) as sender, JsonChannel(second) as receiver:
        sender.send({"type": "hello", "metadata": "abc", "count": 4})
        assert receiver.receive() == {
            "type": "hello",
            "metadata": "abc",
            "count": 4,
        }


def test_protocol_rejects_missing_message_type() -> None:
    first, second = socket.socketpair()
    with (
        JsonChannel(first) as sender,
        JsonChannel(second),
        pytest.raises(ProtocolError, match="requires a string type"),
    ):
        sender.send({"count": 4})
