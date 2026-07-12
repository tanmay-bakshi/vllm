"""Pure transfer-plan construction for the NIXL transport micro-rig."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from tools.gemma4_pd.nixl_micro_rig.config import (
    ConfigError,
    RigConfig,
    ScenarioConfig,
)


@dataclass(frozen=True)
class BlockRun:
    """Describe one maximal consecutive source block-ID run.

    :ivar start_block: First source block ID.
    :ivar block_count: Consecutive block count.
    :ivar position_offset: First stable-sorted transfer position.
    """

    start_block: int
    block_count: int
    position_offset: int


@dataclass(frozen=True)
class GroupPairing:
    """Preserve one original remote/local pair through source sorting.

    :ivar group_index: KV-cache group that owns the pair.
    :ivar group_position: Position ordinal after group-wise prefix trimming.
    :ivar remote_block_id: P physical source block.
    :ivar local_block_id: D physical destination block.
    :ivar source_position: Position in the original flattened group roster.
    """

    group_index: int
    group_position: int
    remote_block_id: int
    local_block_id: int
    source_position: int


@dataclass(frozen=True)
class ReplayGroup:
    """Captured group-wise block-ID pairing before prefix trimming.

    :ivar group_index: KV-cache group index.
    :ivar remote_block_ids: P physical IDs before suffix trimming.
    :ivar local_block_ids: D uncached suffix IDs.
    """

    group_index: int
    remote_block_ids: tuple[int, ...]
    local_block_ids: tuple[int, ...]


@dataclass(frozen=True)
class TransferPlan:
    """Describe one vLLM-compatible coalesced pull geometry.

    :ivar scenario_name: Source scenario identity.
    :ivar original_pairings: Post-trim pairs in group-flattened order.
    :ivar sorted_pairings: Same pairs after stable remote-ID sorting.
    :ivar runs: Maximal consecutive source-ID runs.
    :ivar region_offsets: Byte offsets of regions within one staged plan.
    :ivar staging_bytes: Exact bytes occupied by one request plan.
    :ivar source_registration_bytes_per_rank: Registered source bytes per P rank.
    :ivar destination_registration_bytes: Registered D destination bytes.
    :ivar descriptors_per_handle: Raw descriptors in each P-rank NIXL handle.
    :ivar rank_count: Independent P agents and TP ranks.
    :ivar destination_block_count: Physical rows in every D region.
    :ivar replay_manifest: Captured plan source, absent for synthetic geometry.
    """

    scenario_name: str
    original_pairings: tuple[GroupPairing, ...]
    sorted_pairings: tuple[GroupPairing, ...]
    runs: tuple[BlockRun, ...]
    region_offsets: tuple[int, ...]
    staging_bytes: int
    source_registration_bytes_per_rank: int
    destination_registration_bytes: int
    descriptors_per_handle: int
    rank_count: int
    destination_block_count: int
    replay_manifest: str | None

    @property
    def position_count(self) -> int:
        """Return the flattened post-trim position count.

        :returns: Number of staged source rows per rank and region.
        """
        return len(self.sorted_pairings)

    @property
    def block_ids(self) -> tuple[int, ...]:
        """Return stable-sorted P source block IDs.

        :returns: Remote block IDs in staging order.
        """
        return tuple(pair.remote_block_id for pair in self.sorted_pairings)

    @property
    def local_block_ids(self) -> tuple[int, ...]:
        """Return D scatter rows paired with sorted source IDs.

        :returns: Local block IDs in staging order.
        """
        return tuple(pair.local_block_id for pair in self.sorted_pairings)


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
    scenario: ScenarioConfig, position_count: int
) -> tuple[BlockRun, ...]:
    """Construct the requested exact source-ID span and run geometry.

    :param scenario: Source geometry configuration.
    :param position_count: Number of selected post-trim positions.
    :returns: Maximal consecutive runs in ascending source order.
    """
    if position_count == 0:
        return ()
    span = scenario.source_end_block - scenario.source_start_block + 1
    gap_count = span - position_count
    run_lengths = _balanced_parts(position_count, scenario.run_count)
    gaps = _balanced_parts(gap_count, scenario.run_count - 1)
    runs: list[BlockRun] = []
    source_block = scenario.source_start_block
    position_offset = 0
    for run_index, block_count in enumerate(run_lengths):
        runs.append(
            BlockRun(
                start_block=source_block,
                block_count=block_count,
                position_offset=position_offset,
            )
        )
        source_block += block_count
        position_offset += block_count
        if run_index < len(gaps):
            source_block += gaps[run_index]
    if source_block - 1 != scenario.source_end_block:
        raise AssertionError("run construction did not reach the declared source end")
    return tuple(runs)


def _synthetic_replay(
    config: RigConfig, scenario: ScenarioConfig
) -> tuple[ReplayGroup, ...]:
    """Construct group-wise input while retaining the production trim shape.

    Selected remote IDs are assigned in original group order. The plan builder
    still carries every remote/local pair through the same stable sort used by
    vLLM. Captured replay manifests can replace this synthetic assignment.

    :param config: Complete rig configuration.
    :param scenario: Selected source geometry scenario.
    :returns: Group inputs before remote suffix trimming.
    """
    position_count = sum(group.local_position_count for group in config.groups)
    runs = build_block_runs(scenario, position_count)
    selected_ids_sorted = [
        block_id
        for run in runs
        for block_id in range(run.start_block, run.start_block + run.block_count)
    ]
    remote_stride = 1049
    remote_rotation = 17
    selected_ids = [
        selected_ids_sorted[
            (position * remote_stride + remote_rotation) % position_count
        ]
        for position in range(position_count)
    ]
    local_pool_start = config.source_block_count - position_count - 4096
    if local_pool_start < 0:
        local_pool_start = 0
    local_ids_sorted = list(range(local_pool_start, local_pool_start + position_count))
    local_stride = 791
    local_rotation = 31
    local_ids_all = [
        local_ids_sorted[(position * local_stride + local_rotation) % position_count]
        for position in range(position_count)
    ]
    selected = set(selected_ids_sorted)
    prefix_cursor = config.source_block_count - 1
    remote_cursor = 0
    local_cursor = 0
    replay: list[ReplayGroup] = []
    for group in config.groups:
        prefix_count = group.remote_position_count - group.local_position_count
        prefix_ids: list[int] = []
        while len(prefix_ids) < prefix_count:
            if prefix_cursor not in selected:
                prefix_ids.append(prefix_cursor)
            prefix_cursor -= 1
            if prefix_cursor < 0:
                raise ConfigError("unable to allocate synthetic trimmed prefix IDs")
        suffix_ids = selected_ids[
            remote_cursor : remote_cursor + group.local_position_count
        ]
        local_ids = tuple(
            local_ids_all[local_cursor : local_cursor + group.local_position_count]
        )
        replay.append(
            ReplayGroup(
                group_index=group.index,
                remote_block_ids=tuple(prefix_ids) + tuple(suffix_ids),
                local_block_ids=local_ids,
            )
        )
        remote_cursor += group.local_position_count
        local_cursor += group.local_position_count
    return tuple(replay)


def load_replay_manifest(path: Path) -> tuple[ReplayGroup, ...]:
    """Load an actual group-wise remote/local pairing capture.

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


def apply_group_prefix_trim(
    config: RigConfig, replay: tuple[ReplayGroup, ...]
) -> tuple[GroupPairing, ...]:
    """Apply vLLM's non-Mamba remote-suffix trim group by group.

    :param config: Complete rig configuration.
    :param replay: Remote/local group IDs before prefix trimming.
    :returns: Original flattened pairings after trimming.
    :raises ConfigError: If the replay is inconsistent with the handshake shape.
    """
    if len(replay) != len(config.groups):
        raise ConfigError("replay group count differs from configured groups")
    result: list[GroupPairing] = []
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
        trim_offset = len(captured.remote_block_ids) - len(captured.local_block_ids)
        if len(captured.local_block_ids) == 0:
            remote_suffix: tuple[int, ...] = ()
        else:
            remote_suffix = captured.remote_block_ids[-len(captured.local_block_ids) :]
        for group_position, (remote_id, local_id) in enumerate(
            zip(remote_suffix, captured.local_block_ids, strict=True)
        ):
            if remote_id < 0 or remote_id >= config.source_block_count:
                raise ConfigError(
                    f"group {group.index} remote block ID is out of range"
                )
            if local_id < 0 or local_id >= config.source_block_count:
                raise ConfigError(f"group {group.index} local block ID is out of range")
            result.append(
                GroupPairing(
                    group_index=group.index,
                    group_position=group_position,
                    remote_block_id=remote_id,
                    local_block_id=local_id,
                    source_position=trim_offset + group_position,
                )
            )
    return tuple(result)


def _runs_from_sorted_pairs(
    pairings: tuple[GroupPairing, ...],
) -> tuple[BlockRun, ...]:
    """Find maximal consecutive remote-ID runs after stable sorting.

    :param pairings: Pairings in stable remote-ID order.
    :returns: Maximal descriptor runs.
    """
    runs: list[BlockRun] = []
    if len(pairings) == 0:
        return ()
    run_start = 0
    for index in range(1, len(pairings) + 1):
        at_end = index == len(pairings)
        consecutive = False
        if not at_end:
            consecutive = (
                pairings[index].remote_block_id
                == pairings[index - 1].remote_block_id + 1
            )
        if at_end or not consecutive:
            runs.append(
                BlockRun(
                    start_block=pairings[run_start].remote_block_id,
                    block_count=index - run_start,
                    position_offset=run_start,
                )
            )
            run_start = index
    return tuple(runs)


def build_plan(
    config: RigConfig,
    scenario: ScenarioConfig,
    replay: tuple[ReplayGroup, ...] | None = None,
) -> TransferPlan:
    """Build the raw coalesced pull and exact destination-scatter plan.

    :param config: Complete rig configuration.
    :param scenario: Selected source geometry scenario.
    :param replay: Optional captured group pairing before prefix trimming.
    :returns: Transfer plan used by every independent runtime role.
    :raises ConfigError: If the replay or declared geometry is inconsistent.
    """
    if replay is None:
        replay = _synthetic_replay(config, scenario)
    original_pairings = apply_group_prefix_trim(config, replay)
    sorted_pairings = tuple(
        sorted(original_pairings, key=lambda pair: pair.remote_block_id)
    )
    runs = _runs_from_sorted_pairs(sorted_pairings)
    if scenario.replay_manifest is None:
        if len(runs) != scenario.run_count:
            raise AssertionError("synthetic source construction changed run count")
        if len(sorted_pairings) > 0:
            if sorted_pairings[0].remote_block_id != scenario.source_start_block:
                raise AssertionError(
                    "synthetic source construction changed first block"
                )
            if sorted_pairings[-1].remote_block_id != scenario.source_end_block:
                raise AssertionError("synthetic source construction changed last block")

    rank_count = len(config.producer_devices)
    region_offsets: list[int] = []
    staging_bytes = 0
    source_registration_bytes_per_rank = 0
    destination_registration_bytes = 0
    for region in config.regions:
        region_offsets.append(staging_bytes)
        staging_bytes += len(sorted_pairings) * rank_count * region.row_bytes
        source_registration_bytes_per_rank += (
            config.source_block_count * region.row_bytes
        )
        destination_registration_bytes += (
            config.source_block_count * rank_count * region.row_bytes
        )
    return TransferPlan(
        scenario_name=scenario.name,
        original_pairings=original_pairings,
        sorted_pairings=sorted_pairings,
        runs=runs,
        region_offsets=tuple(region_offsets),
        staging_bytes=staging_bytes,
        source_registration_bytes_per_rank=source_registration_bytes_per_rank,
        destination_registration_bytes=destination_registration_bytes,
        descriptors_per_handle=len(runs) * len(config.regions),
        rank_count=rank_count,
        destination_block_count=config.source_block_count,
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
    """Build a JSON-serializable plan description.

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
    two_gib = 2 * 1024**3
    source_region_ends = (
        [(plan.block_ids[-1] + 1) * region.row_bytes for region in config.regions]
        if plan.position_count > 0
        else []
    )
    staging_slabs: list[dict[str, int | str | bool]] = []
    for region_index, region in enumerate(config.regions):
        rank_slab_bytes = plan.position_count * region.row_bytes
        for rank in range(plan.rank_count):
            start = plan.region_offsets[region_index] + rank * rank_slab_bytes
            staging_slabs.append(
                {
                    "region": region.name,
                    "rank": rank,
                    "start": start,
                    "end": start + rank_slab_bytes,
                    "starts_above_2gib": start >= two_gib,
                    "crosses_2gib": start < two_gib < start + rank_slab_bytes,
                }
            )
    return {
        "scenario": scenario.name,
        "plan_provenance": (
            "synthetic" if plan.replay_manifest is None else plan.replay_manifest
        ),
        "position_count": plan.position_count,
        "run_count": len(plan.runs),
        "descriptors_per_handle": plan.descriptors_per_handle,
        "crosses_nixl_1024_descriptor_split": plan.descriptors_per_handle >= 1024,
        "has_content_evidence": plan.position_count > 0,
        "first_source_block": plan.block_ids[0] if plan.position_count > 0 else None,
        "last_source_block": plan.block_ids[-1] if plan.position_count > 0 else None,
        "source_registration_bytes_per_rank": plan.source_registration_bytes_per_rank,
        "destination_registration_bytes": plan.destination_registration_bytes,
        "source_region_ends": source_region_ends,
        "source_regions_crossing_2gib": sum(
            1 for end in source_region_ends if end > two_gib
        ),
        "staging_bytes": plan.staging_bytes,
        "staging_mib": plan.staging_bytes // (1024 * 1024),
        "staging_capacity_mib": config.staging_capacity_mib,
        "staging_offsets_mib": list(scenario.staging_offsets_mib),
        "region_offsets": list(plan.region_offsets),
        "runs": [asdict(run) for run in plan.runs],
        "group_counts": [
            {
                "group_index": group.index,
                "remote_before_trim": group.remote_position_count,
                "local_after_trim": group.local_position_count,
                "owned_region_indices": list(group.owned_region_indices),
            }
            for group in config.groups
        ],
        "original_pairing_head": [asdict(pair) for pair in plan.original_pairings[:8]],
        "sorted_pairing_head": [asdict(pair) for pair in plan.sorted_pairings[:8]],
        "staging_slabs": staging_slabs,
    }
