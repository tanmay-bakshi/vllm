import hashlib
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
    ObservationContext,
    compact_digests,
    compute_rig_semantic_contract_digest,
    destination_observations,
    expected_plane_bytes,
    fill_destination_canary_rows,
    fill_source_rows,
    source_observations,
    staging_observations,
    verify_destination_canary_rows,
    verify_destination_rows,
    verify_staging_rows,
)
from tools.gemma4_pd.nixl_micro_rig.geometry import (
    ReplayGroup,
    build_configured_plan,
    build_group_rosters,
    build_plan,
    describe_plan,
)
from tools.gemma4_pd.nixl_micro_rig.integrity_check import (
    integrity_contract_self_test,
)
from tools.gemma4_pd.nixl_micro_rig.protocol import (
    JsonChannel,
    PreparePayload,
    ProtocolError,
    SourcePostPayload,
)
from tools.gemma4_pd.nixl_micro_rig.roles import (
    _prepared_descriptors,
    _raw_descriptors,
    _scatter,
    _staging_guard_ranges,
)
from vllm.distributed.kv_transfer.coalesced_layout import (
    CoalescedTransferPlan,
    GroupTransferRoster,
    RegionOwnership,
    build_coalesced_transfer_plan,
)
from vllm.distributed.kv_transfer.integrity import IntegrityPayloadKind
from vllm.distributed.kv_transfer.staging_ownership import (
    StagingRangeAllocator,
    StagingSafetyError,
)

CONFIG_PATH = (
    Path(__file__).parents[2]
    / "tools"
    / "gemma4_pd"
    / "nixl_micro_rig"
    / "gemma4_tp4_to_tp1.json"
)
PRODUCTION_CONFIG_PATH = CONFIG_PATH.with_name("gemma4_tp4_to_tp1_2k.json")

EXPECTED_CONFIG_IDENTITIES = {
    CONFIG_PATH.name: (
        "9a03cef6426f4395cf8491fec7716e79aafd7d0fcfae5f09769afa4004cb0b38",
        "f05cec19e50426f9a437ccd5b360545dde9e7f61140df635cf35ec62bc4c0d8e",
    ),
    PRODUCTION_CONFIG_PATH.name: (
        "f2b87d7826c6ded812fc5fa3edf31a9f57da17e6f8e94fa5d7a00413a4d5dec0",
        "335f0a6c72693267cdd3830c319c61c1c37d38b65fed66a2ed3d743ca41a4933",
    ),
}


def _allocator_layout(
    rank_count: int, *, has_positions: bool = True
) -> CoalescedTransferPlan:
    block_ids = (0,) if has_positions else ()
    return build_coalesced_transfer_plan(
        source_tp_size=rank_count,
        source_ranks=tuple(range(rank_count)),
        rank_slots=tuple(range(rank_count)),
        groups=(
            GroupTransferRoster(
                group_index=0,
                source_position_start=0,
                destination_plane_count=2,
                local_block_ids=block_ids,
                remote_block_ids=block_ids,
            ),
        ),
        regions=(
            RegionOwnership(
                region_index=0,
                group_indices=(0,),
                source_row_count=1,
                destination_row_count=1,
                row_bytes=512 // rank_count,
            ),
        ),
    )


def test_default_geometry_matches_legacy_storm37b_physical_capture() -> None:
    config = load_config(CONFIG_PATH)
    scenario = config.scenario("starts_at_2gib")
    plan = build_plan(config, scenario)

    assert plan.logical_position_count == 2672
    assert plan.region_position_count == 13360
    assert config.valid_token_extent == 16240
    assert tuple(group.token_capacity for group in config.groups) == (16,) * 13
    assert plan.rank_count == 4
    assert plan.staging_bytes == 3_502_243_840
    assert plan.staging_bytes // (1024 * 1024) == 3340
    assert plan.source_registration_bytes_per_rank == 41_943_040_000
    assert plan.destination_registration_bytes == 167_772_160_000
    assert tuple(region.offset_within_rank for region in plan.transport.regions) == (
        0,
        174_981_120,
        349_962_240,
        524_943_360,
        699_924_480,
        874_905_600,
        875_036_672,
        875_167_744,
        875_298_816,
        875_429_888,
    )
    assert plan.transport.rank_stride_bytes == 875_560_960
    assert plan.replay_manifest is None
    evidence = describe_plan(config, scenario, plan)["handshake_evidence"]
    assert evidence == {
        "legacy_physical_capture": {
            "source_manifest": "storm37b-p-handshake.json",
            "destination_manifest": "storm37b-d1-handshake.json",
            "sha256_authenticated": True,
            "evidence_scope": "legacy_storm37b_physical_geometry_only",
            "connector_v11_semantics_authenticated": False,
        },
        "connector_v11_semantic_fixture": {
            "manifest": "connector-v11-semantic-fixtures.json",
            "profile": "legacy-storm37b-roster-static-projection",
            "sha256_authenticated": True,
            "evidence_scope": "static_model_free_connector_v11_contract_fixture",
            "runtime_capture_authenticated": False,
        },
    }


def test_production_2k_profile_matches_exact_gemma_transport_geometry() -> None:
    config = load_config(PRODUCTION_CONFIG_PATH)
    scenario = config.scenario("production_2k_contiguous")
    plan = build_plan(config, scenario)

    assert config.valid_token_extent == 2048
    assert tuple(group.remote_position_count for group in config.groups) == (
        (64,) * 10 + (65, 65, 33)
    )
    assert tuple(group.local_position_count for group in config.groups) == (
        (64,) * 10 + (65, 65, 33)
    )
    assert tuple(group.destination_plane_count for group in config.groups) == (
        (2,) * 13
    )
    assert tuple(group.token_capacity for group in config.groups) == (
        (16,) * 10 + (32, 32, 64)
    )
    assert [len(region.positions) for region in plan.transport.regions] == [
        770,
        770,
        770,
        770,
        770,
        33,
        33,
        33,
        33,
        33,
    ]
    assert plan.logical_position_count == 803
    assert plan.region_position_count == 4015
    assert plan.staging_bytes == 1_052_508_160
    assert plan.staging_bytes == 4015 * 4 * 65_536
    assert describe_plan(config, scenario, plan)["staging_mib"] == 1003.75
    assert plan.descriptors_per_handle == 10

    dflash_positions = plan.transport.regions[5].positions
    assert all(position.group_index == 12 for position in dflash_positions)
    assert all(position.destination_half == -1 for position in dflash_positions)
    assert len({position.local_block_id for position in dflash_positions}) == 33


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
        ("owned_ceiling_203_runs", 203, 1020, False, 54641),
        ("owned_split_boundary_204_runs", 204, 1025, True, 54641),
        ("owned_205_runs", 205, 1030, True, 54641),
        ("low_owned_split_boundary", 204, 1025, True, 3898),
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

    assert config.scenario(scenario_name).run_count == run_count
    assert description["region_run_count"] == descriptor_count
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

    rosters = build_group_rosters(config, tuple(replay))

    assert rosters[0].remote_block_ids[0] == 1000
    assert rosters[0].local_block_ids[0] == 0
    assert rosters[0].source_position_start == 2
    assert all(
        block_id not in {17, 19}
        for roster in rosters
        for block_id in roster.remote_block_ids
    )


def test_default_replay_exercises_stable_sort_and_high_scatter_rows() -> None:
    config = load_config(CONFIG_PATH)
    plan = build_plan(config, config.scenario("observed_high_contiguous"))

    original_remote_ids = tuple(
        block_id
        for group in plan.transport.groups[:12]
        for block_id in group.remote_block_ids
    )
    sorted_remote_ids = tuple(
        position.remote_block_id for position in plan.transport.regions[0].positions
    )
    assert original_remote_ids != sorted_remote_ids
    assert (
        min(
            block_id
            for group in plan.transport.groups
            for block_id in group.local_block_ids
        )
        >= 49152
    )
    original_pairs = {
        remote_block_id: local_block_id
        for group in plan.transport.groups
        for remote_block_id, local_block_id in zip(
            group.remote_block_ids, group.local_block_ids, strict=True
        )
    }
    assert all(
        original_pairs[position.remote_block_id] == position.local_block_id
        for region in plan.transport.regions
        for position in region.positions
    )


def test_duplicate_remote_ids_across_owned_groups_are_rejected() -> None:
    config = load_config(CONFIG_PATH)
    scenario = config.scenario("starts_at_2gib")
    base = build_plan(config, scenario)
    duplicate_id = base.transport.groups[0].remote_block_ids[0]
    group_one_remote = list(base.transport.groups[1].remote_block_ids)
    group_one_remote[0] = duplicate_id
    replay = tuple(
        ReplayGroup(
            group_index=group.index,
            remote_block_ids=tuple(
                group_one_remote
                if group.index == 1
                else base.transport.groups[group.index].remote_block_ids
            ),
            local_block_ids=base.transport.groups[group.index].local_block_ids,
        )
        for group in config.groups
    )
    replay_scenario = replace(scenario, replay_manifest="capture.json")
    with pytest.raises(ValueError, match="reads remote block"):
        build_plan(config, replay_scenario, replay)


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
    replay_groups = []
    replay = []
    for group in config.groups:
        roster = synthetic.transport.groups[group.index]
        suffix = roster.remote_block_ids
        prefix_count = group.remote_position_count - len(suffix)
        remote_ids = tuple(0 for _ in range(prefix_count)) + suffix
        local_ids = roster.local_block_ids
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

    rebuilt = build_plan(config, replay_scenario, tuple(replay))
    assert configured.transport == rebuilt.transport
    assert configured.scenario_name == rebuilt.scenario_name
    assert configured.replay_manifest == rebuilt.replay_manifest
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
    replay = tuple(
        ReplayGroup(
            group_index=group.index,
            remote_block_ids=tuple(
                reversed(base.transport.groups[group.index].remote_block_ids)
            ),
            local_block_ids=base.transport.groups[group.index].local_block_ids,
        )
        for group in config.groups
    )
    plan = build_plan(config, replace(scenario, replay_manifest="capture.json"), replay)
    original_map = {
        remote_block_id: local_block_id
        for group in plan.transport.groups
        for remote_block_id, local_block_id in zip(
            group.remote_block_ids, group.local_block_ids, strict=True
        )
    }

    assert tuple(
        position.remote_block_id for position in plan.transport.regions[0].positions
    ) == tuple(
        sorted(
            remote_block_id
            for group in plan.transport.groups[:12]
            for remote_block_id in group.remote_block_ids
        )
    )
    assert all(
        original_map[position.remote_block_id] == position.local_block_id
        for region in plan.transport.regions
        for position in region.positions
    )


def test_configuration_rejects_unknown_keys() -> None:
    value = json.loads(CONFIG_PATH.read_text())
    value["surprise"] = True

    with pytest.raises(ConfigError, match="unknown keys"):
        RigConfig.from_json(value)


@pytest.mark.parametrize("config_path", [CONFIG_PATH, PRODUCTION_CONFIG_PATH])
def test_configuration_identity_protects_unused_gpu_five(config_path: Path) -> None:
    expected_file_sha256, expected_fingerprint = EXPECTED_CONFIG_IDENTITIES[
        config_path.name
    ]
    payload = config_path.read_bytes()
    config = load_config(config_path)

    assert hashlib.sha256(payload).hexdigest() == expected_file_sha256
    assert config.fingerprint == expected_fingerprint
    assert config.protected_devices == (5, 6, 7)


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
        ("legacy_source_handshake_manifest", "../capture.json", "relative path"),
        ("legacy_source_handshake_sha256", "A" * 64, "lowercase SHA-256"),
        ("semantic_handshake_manifest", "../fixture.json", "relative path"),
        ("semantic_handshake_sha256", "A" * 64, "lowercase SHA-256"),
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
        "connector-v11-semantic-fixtures.json",
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
    destination[
        plan.transport.regions[0].positions[0].local_block_id * region.row_bytes
    ] = 0
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


def test_rank_major_owned_staging_matches_source_integrity_leaves() -> None:
    config = load_config(CONFIG_PATH)
    regions = (
        replace(config.regions[0], name="owned_0", row_bytes=16),
        replace(config.regions[1], name="owned_1", row_bytes=16),
    )
    groups = (
        replace(
            config.groups[0],
            index=0,
            remote_position_count=2,
            local_position_count=2,
            owned_region_indices=(0,),
        ),
        replace(
            config.groups[1],
            index=1,
            remote_position_count=2,
            local_position_count=2,
            owned_region_indices=(1,),
        ),
    )
    scenario = replace(
        config.scenarios[0],
        name="owned_tiny",
        source_start_block=0,
        source_end_block=3,
        run_count=1,
        iterations=1,
        staging_offsets_mib=(0,),
        replay_manifest=None,
    )
    tiny = replace(
        config,
        producer_devices=(0, 1),
        consumer_device=2,
        source_block_count=16,
        staging_capacity_mib=1,
        regions=regions,
        groups=groups,
        scenarios=(scenario,),
    )
    plan = build_plan(tiny, scenario)
    context = ObservationContext(
        run_id="run-owned",
        transport_arm="cuda_copy",
        producer_engine_id="producer",
        producer_request_id="producer-request",
        offer_generation=1,
        iteration=3,
        child_request_id="child-request",
        consumer_engine_id="consumer",
    )
    sources: list[tuple[torch.Tensor, ...]] = []
    source_leaves = []
    for source_rank in range(plan.rank_count):
        source_regions = tuple(
            torch.zeros(tiny.source_block_count * region.row_bytes, dtype=torch.uint8)
            for region in tiny.regions
        )
        fill_source_rows(
            config=tiny,
            plan=plan,
            source_rank=source_rank,
            iteration=context.iteration,
            regions=source_regions,
        )
        source_leaves.extend(
            source_observations(
                config=tiny,
                plan=plan,
                context=context,
                source_rank=source_rank,
                regions=source_regions,
            )
        )
        sources.append(source_regions)

    staging = torch.zeros(plan.staging_bytes, dtype=torch.uint8)
    for source_rank, source_regions in enumerate(sources):
        for region_index, region in enumerate(tiny.regions):
            positions = plan.transport.regions[region_index].positions
            block_ids = torch.tensor(
                tuple(position.remote_block_id for position in positions),
                dtype=torch.long,
            )
            selected = source_regions[region_index].view(
                tiny.source_block_count, region.row_bytes
            )[block_ids]
            start = plan.transport.region_offset(source_rank, region_index)
            staging[start : start + selected.numel()].copy_(selected.flatten())

    verify_staging_rows(
        config=tiny,
        plan=plan,
        iteration=context.iteration,
        staging=staging,
        staging_offset=0,
    )
    staged_leaves = staging_observations(
        config=tiny,
        plan=plan,
        context=context,
        staging=staging,
        staging_offset=0,
    )

    assert [
        tuple(position.group_index for position in region.positions)
        for region in plan.transport.regions
    ] == [(0, 0), (1, 1)]
    assert sorted(compact_digests(source_leaves)) == sorted(
        compact_digests(staged_leaves)
    )


def test_single_plane_scatter_and_integrity_use_absolute_position_halves() -> None:
    config = load_config(PRODUCTION_CONFIG_PATH)
    region = replace(config.regions[5], name="single_plane", row_bytes=16)
    group = replace(
        config.groups[12],
        index=0,
        destination_plane_count=1,
        remote_position_count=3,
        local_position_count=2,
        owned_region_indices=(0,),
    )
    scenario = replace(
        config.scenarios[0],
        name="single_plane_tiny",
        source_start_block=0,
        source_end_block=2,
        run_count=1,
        iterations=1,
        staging_offsets_mib=(0,),
    )
    tiny = replace(
        config,
        source_block_count=8,
        staging_capacity_mib=1,
        regions=(region,),
        groups=(group,),
        scenarios=(scenario,),
    )
    plan = build_plan(tiny, scenario)
    context = ObservationContext(
        run_id="run-single-plane",
        transport_arm="cuda_ipc",
        producer_engine_id="producer",
        producer_request_id="producer-request",
        offer_generation=2,
        iteration=5,
        child_request_id="child-request",
        consumer_engine_id="consumer",
    )
    source_leaves = []
    staging = torch.zeros(plan.staging_bytes, dtype=torch.uint8)
    for source_rank in range(plan.rank_count):
        source_regions = (
            torch.zeros(tiny.source_block_count * region.row_bytes, dtype=torch.uint8),
        )
        fill_source_rows(
            config=tiny,
            plan=plan,
            source_rank=source_rank,
            iteration=context.iteration,
            regions=source_regions,
        )
        source_leaves.extend(
            source_observations(
                config=tiny,
                plan=plan,
                context=context,
                source_rank=source_rank,
                regions=source_regions,
            )
        )
        positions = plan.transport.regions[0].positions
        block_ids = torch.tensor(
            tuple(position.remote_block_id for position in positions),
            dtype=torch.long,
        )
        selected = source_regions[0].view(tiny.source_block_count, region.row_bytes)[
            block_ids
        ]
        start = plan.transport.region_offset(source_rank, 0)
        staging[start : start + selected.numel()].copy_(selected.flatten())
    destination = torch.zeros(
        tiny.source_block_count * plan.rank_count * region.row_bytes,
        dtype=torch.uint8,
    )

    _scatter(
        plan=plan,
        staging=staging,
        staging_offset=0,
        destinations=(destination,),
    )
    verify_destination_rows(
        config=tiny,
        plan=plan,
        iteration=context.iteration,
        destinations=(destination,),
    )
    staged_leaves = staging_observations(
        config=tiny,
        plan=plan,
        context=context,
        staging=staging,
        staging_offset=0,
    )
    destination_leaves = destination_observations(
        config=tiny,
        plan=plan,
        context=context,
        destinations=(destination,),
    )

    positions = plan.transport.regions[0].positions
    assert sorted(
        (position.source_position, position.destination_half) for position in positions
    ) == [(0, 0), (1, 1), (2, 0)]
    assert all(
        leaf.identity.byte_length == region.row_bytes // 2 for leaf in source_leaves
    )
    assert all(leaf.identity.plane_index == 0 for leaf in source_leaves)
    assert all(
        leaf.identity.payload_kind is IntegrityPayloadKind.COMMIT
        for leaf in source_leaves
    )
    assert sorted(compact_digests(source_leaves)) == sorted(
        compact_digests(staged_leaves)
    )
    assert sorted(compact_digests(source_leaves)) == sorted(
        compact_digests(destination_leaves)
    )
    trailing_position = next(
        position for position in positions if position.source_position == 2
    )
    destination_rows = destination.view(
        tiny.source_block_count,
        plan.rank_count,
        2,
        region.row_bytes // 2,
    )
    assert torch.all(destination_rows[trailing_position.local_block_id, :, 1] == 0)


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


def test_raw_descriptors_use_canonical_rank_major_region_runs() -> None:
    config = load_config(CONFIG_PATH)
    plan = build_plan(config, config.scenario("starts_at_2gib"))
    staging = torch.empty(1, dtype=torch.uint8)
    staging_offset = 4096
    remote_bases = [10_000_000_000 + index * 1_000_000 for index in range(10)]

    local, remote = _raw_descriptors(
        config=config,
        plan=plan,
        staging=staging,
        staging_offset=staging_offset,
        rank=2,
        remote_bases=remote_bases,
        remote_device=2,
    )

    assert len(local) == plan.descriptors_per_handle == 10
    assert local[0] == (
        staging.data_ptr() + staging_offset + plan.transport.region_offset(2, 0),
        2670 * 65_536,
        -1,
    )
    assert local[5] == (
        staging.data_ptr() + staging_offset + plan.transport.region_offset(2, 5),
        2 * 65_536,
        -1,
    )
    assert remote[0] == (
        remote_bases[0] + 32_768 * 65_536,
        2670 * 65_536,
        2,
    )
    assert remote[5] == (
        remote_bases[5] + 35_438 * 65_536,
        2 * 65_536,
        2,
    )


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
    ownership = allocator.create_plan(
        owner_id="plan-7",
        request_id="request-7",
        generation=7,
        offset=0,
        layout=_allocator_layout(4),
        layout_duration_seconds=0.001,
    )
    assert ownership is not None
    for rank, status in ((0, "DONE"), (1, "PROC"), (2, "ERR")):
        ownership.begin_prepare(rank)
        ownership.attach_handle(rank, object())
        ownership.begin_post(rank)
        ownership.record_post_result(rank, status)
    ownership.seal_posting()

    assert ownership.operation_failed
    assert ownership.permanently_tombstoned
    assert not ownership.reusable
    with pytest.raises(StagingSafetyError, match="not quiescent"):
        allocator.release(ownership)
    assert (
        allocator.create_plan(
            owner_id="plan-8",
            request_id="request-8",
            generation=8,
            offset=0,
            layout=_allocator_layout(4),
            layout_duration_seconds=0.001,
        )
        is None
    )

    ownership.record_query_result(1, "DONE")
    assert not ownership.native_quiescent
    assert not ownership.reusable


def test_post_after_sealed_failure_boundary_is_rejected() -> None:
    allocator = StagingRangeAllocator(capacity=1024)
    ownership = allocator.create_plan(
        owner_id="plan-9",
        request_id="request-9",
        layout=_allocator_layout(4),
        layout_duration_seconds=0.001,
    )
    assert ownership is not None
    ownership.begin_prepare(0)
    ownership.attach_handle(0, object())
    ownership.begin_post(0)
    ownership.record_post_result(0, "ERR")
    ownership.seal_posting()

    with pytest.raises(RuntimeError, match="after sealing"):
        ownership.begin_prepare(1)


def test_out_of_range_rank_cannot_extend_generation() -> None:
    allocator = StagingRangeAllocator(capacity=1024)
    ownership = allocator.create_plan(
        owner_id="plan-10",
        request_id="request-10",
        layout=_allocator_layout(4),
        layout_duration_seconds=0.001,
    )
    assert ownership is not None

    with pytest.raises(ValueError, match="not owned"):
        ownership.begin_prepare(4)


def test_unknown_native_submission_remains_tombstoned_after_done() -> None:
    allocator = StagingRangeAllocator(capacity=1024)
    ownership = allocator.create_plan(
        owner_id="plan-unknown",
        request_id="request-unknown",
        layout=_allocator_layout(1),
        layout_duration_seconds=0.001,
    )
    assert ownership is not None
    ownership.begin_prepare(0)
    ownership.attach_handle(0, object())
    ownership.begin_post(0)
    ownership.record_post_exception(0, "post raised")
    ownership.seal_posting()

    ownership.record_query_result(0, "DONE")
    assert ownership.native_quiescent
    assert not ownership.reusable
    with pytest.raises(StagingSafetyError, match="not quiescent"):
        allocator.release(ownership)


def test_successful_generation_requires_device_quiescence_after_scatter() -> None:
    allocator = StagingRangeAllocator(capacity=1024)
    ownership = allocator.create_plan(
        owner_id="plan-success",
        request_id="request-success",
        layout=_allocator_layout(2),
        layout_duration_seconds=0.001,
    )
    assert ownership is not None
    for rank in ownership.slots:
        ownership.begin_prepare(rank)
        ownership.attach_handle(rank, object())
        ownership.begin_post(rank)
        ownership.record_post_result(rank, "DONE")
    ownership.seal_posting()
    for rank in ownership.slots:
        ownership.record_native_telemetry(rank, {"rank": rank})
        ownership.mark_native_released(rank)

    assert ownership.ready_to_scatter
    with pytest.raises(StagingSafetyError, match="no preceding device reader"):
        ownership.mark_device_quiescent()
    ownership.begin_device_read()
    assert not ownership.reusable
    ownership.mark_device_quiescent()
    allocator.release(ownership)
    assert ownership.released
    with pytest.raises(StagingSafetyError, match="late completion"):
        allocator.require_active(ownership.lease.generation)


def test_zero_byte_plan_does_not_consume_allocator_capacity() -> None:
    allocator = StagingRangeAllocator(capacity=1024)
    with pytest.raises(ValueError, match="size"):
        allocator.create_plan(
            owner_id="invalid",
            request_id="invalid",
            layout=_allocator_layout(1, has_positions=False),
            layout_duration_seconds=0.001,
        )
    assert allocator.free_bytes == 1024


def test_prepare_failure_is_quiescent_but_still_logically_failed() -> None:
    allocator = StagingRangeAllocator(capacity=1024)
    ownership = allocator.create_plan(
        owner_id="prepare-failure",
        request_id="prepare-failure",
        layout=_allocator_layout(2),
        layout_duration_seconds=0.001,
    )
    assert ownership is not None
    ownership.begin_prepare(0)
    ownership.record_prepare_failure(0, "prepare failed")
    ownership.seal_posting()

    assert ownership.operation_failed
    assert ownership.permanently_tombstoned is False
    assert ownership.native_quiescent
    allocator.release(ownership)


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
