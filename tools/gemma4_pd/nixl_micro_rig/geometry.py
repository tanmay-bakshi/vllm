"""Pure transfer-plan construction for the NIXL transport micro-rig."""

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from tools.gemma4_pd.nixl_micro_rig.config import (
    ConfigError,
    GroupConfig,
    RigConfig,
    ScenarioConfig,
)
from tools.gemma4_pd.nixl_micro_rig.handshake import (
    STATIC_SEMANTIC_EVIDENCE_SCOPE,
)
from vllm.distributed.kv_transfer.coalesced_layout import (
    CoalescedTransferPlan,
    GroupTransferRoster,
    RegionOwnership,
    RemoteBlockRun,
    build_coalesced_transfer_plan,
)


@dataclass(frozen=True, slots=True)
class ReplayGroup:
    """Captured group-wise block-ID roster before prefix trimming.

    :ivar group_index: KV-cache group index.
    :ivar remote_block_ids: P physical IDs before suffix trimming.
    :ivar local_block_ids: D uncached suffix IDs.
    """

    group_index: int
    remote_block_ids: tuple[int, ...]
    local_block_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class TransferPlan:
    """Rig metadata around the production canonical transport plan.

    :ivar scenario_name: Source scenario identity.
    :ivar transport: Canonical rank-major, region-owned transport plan.
    :ivar source_registration_bytes_per_rank: Registered source bytes per P rank.
    :ivar destination_registration_bytes: Registered D destination bytes.
    :ivar layout_duration_seconds: Wall time spent building ``transport``.
    :ivar replay_manifest: Captured plan source, absent for synthetic geometry.
    """

    scenario_name: str
    transport: CoalescedTransferPlan
    source_registration_bytes_per_rank: int
    destination_registration_bytes: int
    layout_duration_seconds: float
    replay_manifest: str | None

    @property
    def logical_position_count(self) -> int:
        """Return selected positions across the group-local rosters.

        :returns: Logical positions before region ownership expansion.
        """
        return sum(len(group.remote_block_ids) for group in self.transport.groups)

    @property
    def region_position_count(self) -> int:
        """Return positions physically transferred across all regions per rank.

        :returns: Sum of canonical per-region position counts.
        """
        return sum(len(region.positions) for region in self.transport.regions)

    @property
    def descriptors_per_handle(self) -> int:
        """Return canonical NIXL descriptors in each source-rank handle.

        :returns: Sum of maximal runs across owned regions.
        """
        return sum(len(region.runs) for region in self.transport.regions)

    @property
    def rank_count(self) -> int:
        """Return participating producer rank count.

        :returns: Number of rank-major staging slabs.
        """
        return len(self.transport.source_ranks)

    @property
    def staging_bytes(self) -> int:
        """Return exact request staging size.

        :returns: Canonical staging allocation size.
        """
        return self.transport.staging_size_bytes


def _balanced_parts(total: int, part_count: int) -> tuple[int, ...]:
    """Partition an integer into near-equal non-negative parts.

    :param total: Total value to distribute.
    :param part_count: Number of output parts.
    :returns: Parts whose sum is ``total``.
    """
    if part_count == 0:
        return ()
    base, remainder = divmod(total, part_count)
    return tuple(base + (1 if index < remainder else 0) for index in range(part_count))


def build_block_runs(
    scenario: ScenarioConfig,
    position_count: int,
) -> tuple[RemoteBlockRun, ...]:
    """Construct an exact synthetic source-ID span and run geometry.

    :param scenario: Source geometry configuration.
    :param position_count: Number of selected positions.
    :returns: Maximal consecutive runs in ascending source order.
    """
    if position_count == 0:
        return ()
    selected_run_count = scenario.run_count
    if selected_run_count <= 0 or selected_run_count > position_count:
        raise ConfigError("synthetic run count must be within the position count")
    span = scenario.source_end_block - scenario.source_start_block + 1
    gap_count = span - position_count
    if gap_count < selected_run_count - 1:
        raise ConfigError("synthetic source span cannot separate every run")
    run_lengths = _balanced_parts(position_count, selected_run_count)
    gaps = _balanced_parts(gap_count, selected_run_count - 1)
    runs: list[RemoteBlockRun] = []
    source_block = scenario.source_start_block
    position_offset = 0
    for run_index, block_count in enumerate(run_lengths):
        runs.append(
            RemoteBlockRun(
                remote_block_id=source_block,
                position_count=block_count,
                position_start=position_offset,
            )
        )
        source_block += block_count
        position_offset += block_count
        if run_index < len(gaps):
            source_block += gaps[run_index]
    if source_block - 1 != scenario.source_end_block:
        raise AssertionError("run construction did not reach the declared source end")
    return tuple(runs)


def _coprime_stride(position_count: int, preferred: int) -> int:
    if position_count <= 1:
        return 1
    stride = min(preferred, position_count - 1)
    while math.gcd(stride, position_count) != 1:
        stride -= 1
    return stride


def _ownership_partitions(config: RigConfig) -> tuple[tuple[int, ...], ...]:
    partitions: dict[tuple[int, ...], list[int]] = {}
    for group in config.groups:
        partitions.setdefault(group.owned_region_indices, []).append(group.index)
    return tuple(tuple(group_indices) for group_indices in partitions.values())


def _selected_remote_start(group: GroupConfig) -> int:
    remote_positions_per_local = 2 if group.destination_plane_count == 1 else 1
    total_local_positions = (
        group.remote_position_count + remote_positions_per_local - 1
    ) // remote_positions_per_local
    cached_local_positions = total_local_positions - group.local_position_count
    return remote_positions_per_local * cached_local_positions


def _selected_remote_count(group: GroupConfig) -> int:
    return group.remote_position_count - _selected_remote_start(group)


def _synthetic_replay(
    config: RigConfig, scenario: ScenarioConfig
) -> tuple[ReplayGroup, ...]:
    """Construct deterministic group rosters for owned-region fragmentation.

    The configured run geometry is divided into contiguous source-ID slices by
    ownership partition, then independently permuted before group assignment.
    Stable region-local sorting reconstructs each partition's exact slice.

    :param config: Complete rig configuration.
    :param scenario: Selected source geometry scenario.
    :returns: Group inputs before remote suffix trimming.
    """
    logical_position_count = sum(
        _selected_remote_count(group) for group in config.groups
    )
    runs = build_block_runs(scenario, logical_position_count)
    all_selected_ids = tuple(
        block_id
        for run in runs
        for block_id in range(
            run.remote_block_id,
            run.remote_block_id + run.position_count,
        )
    )
    selected_by_group: list[tuple[int, ...]] = [() for _ in config.groups]
    partition_cursor = 0
    for partition_index, group_indices in enumerate(_ownership_partitions(config)):
        partition_position_count = sum(
            _selected_remote_count(config.groups[group_index])
            for group_index in group_indices
        )
        partition_ids_sorted = all_selected_ids[
            partition_cursor : partition_cursor + partition_position_count
        ]
        partition_cursor += partition_position_count
        if partition_position_count == 0:
            continue
        stride = _coprime_stride(partition_position_count, 1049)
        rotation = (17 + partition_index * 31) % partition_position_count
        selected_ids = tuple(
            partition_ids_sorted[
                (position * stride + rotation) % partition_position_count
            ]
            for position in range(partition_position_count)
        )
        cursor = 0
        for group_index in group_indices:
            count = _selected_remote_count(config.groups[group_index])
            selected_by_group[group_index] = selected_ids[cursor : cursor + count]
            cursor += count

    total_local_position_count = sum(
        group.local_position_count for group in config.groups
    )
    local_pool_start = config.source_block_count - total_local_position_count - 4096
    if local_pool_start < 0:
        local_pool_start = 0
    local_ids_sorted = tuple(
        range(local_pool_start, local_pool_start + total_local_position_count)
    )
    local_stride = _coprime_stride(total_local_position_count, 791)
    local_rotation = 31 % max(1, total_local_position_count)
    local_ids_all = tuple(
        local_ids_sorted[
            (position * local_stride + local_rotation) % total_local_position_count
        ]
        for position in range(total_local_position_count)
    )

    prefix_cursor = config.source_block_count - 1
    local_cursor = 0
    replay: list[ReplayGroup] = []
    for group in config.groups:
        prefix_count = _selected_remote_start(group)
        prefix_ids: list[int] = []
        while len(prefix_ids) < prefix_count:
            if prefix_cursor not in all_selected_ids:
                prefix_ids.append(prefix_cursor)
            prefix_cursor -= 1
            if prefix_cursor < 0:
                raise ConfigError("unable to allocate synthetic trimmed prefix IDs")
        local_ids = local_ids_all[
            local_cursor : local_cursor + group.local_position_count
        ]
        replay.append(
            ReplayGroup(
                group_index=group.index,
                remote_block_ids=tuple(prefix_ids) + selected_by_group[group.index],
                local_block_ids=local_ids,
            )
        )
        local_cursor += group.local_position_count
    return tuple(replay)


def load_replay_manifest(path: Path) -> tuple[ReplayGroup, ...]:
    """Load an actual group-wise remote/local roster capture.

    :param path: Versioned JSON replay manifest.
    :returns: Captured group inputs before prefix trimming.
    :raises ConfigError: If the manifest shape is invalid.
    """
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ConfigError(f"invalid replay JSON in {path}: {error}") from error
    if not isinstance(value, dict) or set(value) != {"schema_version", "groups"}:
        raise ConfigError("replay manifest must contain schema_version and groups")
    if value["schema_version"] != 1:
        raise ConfigError("unsupported replay manifest schema_version")
    group_values = value["groups"]
    if not isinstance(group_values, list):
        raise ConfigError("replay manifest groups must be an array")
    result: list[ReplayGroup] = []
    for array_index, group_value in enumerate(group_values):
        if not isinstance(group_value, dict):
            raise ConfigError(f"replay groups[{array_index}] must be an object")
        expected_keys = {"group_index", "remote_block_ids", "local_block_ids"}
        if set(group_value) != expected_keys:
            raise ConfigError(
                f"replay groups[{array_index}] keys differ from {expected_keys}"
            )
        group_index = group_value["group_index"]
        remote_ids = group_value["remote_block_ids"]
        local_ids = group_value["local_block_ids"]
        if isinstance(group_index, bool) or not isinstance(group_index, int):
            raise ConfigError("replay group_index must be an integer")
        if not isinstance(remote_ids, list) or not isinstance(local_ids, list):
            raise ConfigError("replay block IDs must be arrays")
        if not all(
            isinstance(item, int) and not isinstance(item, bool) for item in remote_ids
        ):
            raise ConfigError("replay remote block IDs must be integers")
        if not all(
            isinstance(item, int) and not isinstance(item, bool) for item in local_ids
        ):
            raise ConfigError("replay local block IDs must be integers")
        result.append(
            ReplayGroup(
                group_index=group_index,
                remote_block_ids=tuple(remote_ids),
                local_block_ids=tuple(local_ids),
            )
        )
    return tuple(result)


def build_group_rosters(
    config: RigConfig,
    replay: tuple[ReplayGroup, ...],
) -> tuple[GroupTransferRoster, ...]:
    """Build exact canonical group rosters after prefix trimming.

    :param config: Complete rig configuration.
    :param replay: Remote/local group IDs before prefix trimming.
    :returns: Canonical selected group rosters.
    :raises ConfigError: If the replay differs from the configured shape.
    """
    if len(replay) != len(config.groups):
        raise ConfigError("replay group count differs from configured groups")
    rosters: list[GroupTransferRoster] = []
    for group, captured in zip(config.groups, replay, strict=True):
        if captured.group_index != group.index:
            raise ConfigError("replay groups must be contiguous and ordered")
        if len(captured.remote_block_ids) != group.remote_position_count:
            raise ConfigError(
                f"group {group.index} remote count differs from configuration"
            )
        if len(captured.local_block_ids) != group.local_position_count:
            raise ConfigError(
                f"group {group.index} local count differs from configuration"
            )
        trim_offset = _selected_remote_start(group)
        remote_suffix = captured.remote_block_ids[trim_offset:]
        rosters.append(
            GroupTransferRoster(
                group_index=group.index,
                source_position_start=trim_offset,
                destination_plane_count=group.destination_plane_count,
                local_block_ids=captured.local_block_ids,
                remote_block_ids=remote_suffix,
            )
        )
    return tuple(rosters)


def _region_ownership(config: RigConfig) -> tuple[RegionOwnership, ...]:
    return tuple(
        RegionOwnership(
            region_index=region_index,
            group_indices=tuple(
                group.index
                for group in config.groups
                if region_index in group.owned_region_indices
            ),
            source_row_count=config.source_block_count,
            destination_row_count=config.source_block_count,
            row_bytes=region.row_bytes,
        )
        for region_index, region in enumerate(config.regions)
    )


def build_plan(
    config: RigConfig,
    scenario: ScenarioConfig,
    replay: tuple[ReplayGroup, ...] | None = None,
) -> TransferPlan:
    """Build the canonical transport and destination-scatter plan.

    :param config: Complete rig configuration.
    :param scenario: Selected source geometry scenario.
    :param replay: Optional captured group roster before prefix trimming.
    :returns: Transfer plan used by every independent runtime role.
    :raises ConfigError: If geometry or capacity is inconsistent.
    """
    layout_start = time.perf_counter()
    if replay is None:
        replay = _synthetic_replay(config, scenario)
    group_rosters = build_group_rosters(config, replay)
    rank_count = len(config.producer_devices)
    try:
        transport = build_coalesced_transfer_plan(
            source_tp_size=rank_count,
            source_ranks=tuple(range(rank_count)),
            rank_slots=tuple(range(rank_count)),
            groups=group_rosters,
            regions=_region_ownership(config),
        )
    except ValueError as error:
        raise ConfigError(f"canonical transport plan is invalid: {error}") from error

    capacity_bytes = config.staging_capacity_mib * 1024 * 1024
    for offset_mib in scenario.staging_offsets_mib:
        end = offset_mib * 1024 * 1024 + transport.staging_size_bytes
        if end > capacity_bytes:
            raise ConfigError(
                f"scenario {scenario.name!r} staging offset {offset_mib}MiB "
                f"ends at {end} bytes, past capacity {capacity_bytes}"
            )

    source_registration_bytes_per_rank = sum(
        config.source_block_count * region.row_bytes for region in config.regions
    )
    destination_registration_bytes = sum(
        config.source_block_count * rank_count * region.row_bytes
        for region in config.regions
    )
    return TransferPlan(
        scenario_name=scenario.name,
        transport=transport,
        source_registration_bytes_per_rank=source_registration_bytes_per_rank,
        destination_registration_bytes=destination_registration_bytes,
        layout_duration_seconds=time.perf_counter() - layout_start,
        replay_manifest=scenario.replay_manifest,
    )


def build_configured_plan(
    config_path: Path,
    config: RigConfig,
    scenario: ScenarioConfig,
) -> TransferPlan:
    """Build a scenario from its configured synthetic or captured source.

    :param config_path: Configuration path used to resolve replay manifests.
    :param config: Complete rig configuration.
    :param scenario: Selected source geometry scenario.
    :returns: Exact configured transfer plan.
    """
    replay = None
    if scenario.replay_manifest is not None:
        replay = load_replay_manifest(config_path.parent / scenario.replay_manifest)
    return build_plan(config, scenario, replay)


def describe_plan(
    config: RigConfig,
    scenario: ScenarioConfig,
    plan: TransferPlan | None = None,
) -> dict[str, object]:
    """Build a JSON-serializable canonical plan description.

    :param config: Complete rig configuration.
    :param scenario: Selected source geometry scenario.
    :param plan: Exact configured plan, or a synthetic plan when omitted.
    :returns: Geometry, boundary, registration, and scatter facts.
    """
    if plan is None:
        if scenario.replay_manifest is not None:
            raise ConfigError(
                "replay-backed scenarios require their loaded plan for description"
            )
        plan = build_plan(config, scenario)
    if plan.scenario_name != scenario.name:
        raise ConfigError("described plan does not belong to the scenario")

    transport = plan.transport
    selected_remote_ids = tuple(
        block_id for group in transport.groups for block_id in group.remote_block_ids
    )
    two_gib = 2 * 1024**3
    source_region_ends = (
        [(max(selected_remote_ids) + 1) * region.row_bytes for region in config.regions]
        if len(selected_remote_ids) > 0
        else []
    )
    staging_slabs: list[dict[str, int | str | bool]] = []
    for rank_index, source_rank in enumerate(transport.source_ranks):
        for region_index, region in enumerate(config.regions):
            region_layout = transport.regions[region_index]
            start = transport.region_offset(rank_index, region_index)
            staging_slabs.append(
                {
                    "region": region.name,
                    "rank": source_rank,
                    "start": start,
                    "end": start + region_layout.size_bytes,
                    "starts_above_2gib": start >= two_gib,
                    "crosses_2gib": start < two_gib < start + region_layout.size_bytes,
                }
            )
    return {
        "scenario": scenario.name,
        "handshake_evidence": {
            "legacy_physical_capture": {
                "source_manifest": config.legacy_source_handshake_manifest,
                "destination_manifest": (config.legacy_destination_handshake_manifest),
                "sha256_authenticated": True,
                "evidence_scope": "legacy_storm37b_physical_geometry_only",
                "connector_v9_semantics_authenticated": False,
            },
            "connector_v9_semantic_fixture": {
                "manifest": config.semantic_handshake_manifest,
                "profile": config.semantic_handshake_profile,
                "sha256_authenticated": True,
                "evidence_scope": STATIC_SEMANTIC_EVIDENCE_SCOPE,
                "runtime_capture_authenticated": False,
            },
        },
        "plan_provenance": (
            "synthetic" if plan.replay_manifest is None else plan.replay_manifest
        ),
        "plan_digest": transport.digest,
        "position_count": plan.logical_position_count,
        "region_position_count": plan.region_position_count,
        "region_run_count": plan.descriptors_per_handle,
        "descriptors_per_handle": plan.descriptors_per_handle,
        "crosses_nixl_1024_descriptor_split": plan.descriptors_per_handle >= 1024,
        "has_content_evidence": plan.logical_position_count > 0,
        "first_source_block": min(selected_remote_ids)
        if len(selected_remote_ids) > 0
        else None,
        "last_source_block": max(selected_remote_ids)
        if len(selected_remote_ids) > 0
        else None,
        "source_registration_bytes_per_rank": plan.source_registration_bytes_per_rank,
        "destination_registration_bytes": plan.destination_registration_bytes,
        "source_region_ends": source_region_ends,
        "source_regions_crossing_2gib": sum(
            1 for end in source_region_ends if end > two_gib
        ),
        "staging_bytes": plan.staging_bytes,
        "staging_mib": plan.staging_bytes / (1024 * 1024),
        "staging_capacity_mib": config.staging_capacity_mib,
        "staging_offsets_mib": list(scenario.staging_offsets_mib),
        "rank_stride_bytes": transport.rank_stride_bytes,
        "region_offsets_within_rank": [
            region.offset_within_rank for region in transport.regions
        ],
        "region_position_counts": [
            len(region.positions) for region in transport.regions
        ],
        "region_runs": [
            [asdict(run) for run in region.runs] for region in transport.regions
        ],
        "group_counts": [
            {
                "group_index": group.index,
                "remote_before_trim": group.remote_position_count,
                "local_after_trim": group.local_position_count,
                "owned_region_indices": list(group.owned_region_indices),
            }
            for group in config.groups
        ],
        "group_roster_head": [
            {
                "group_index": group.group_index,
                "source_position_start": group.source_position_start,
                "remote_block_ids": list(group.remote_block_ids[:8]),
                "local_block_ids": list(group.local_block_ids[:8]),
            }
            for group in transport.groups
        ],
        "region_position_head": [
            [asdict(position) for position in region.positions[:8]]
            for region in transport.regions
        ],
        "staging_slabs": staging_slabs,
    }
