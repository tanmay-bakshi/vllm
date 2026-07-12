"""Deterministic payloads and integrity observations for GPU runtime roles."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from tools.gemma4_pd.nixl_micro_rig.config import RigConfig
from tools.gemma4_pd.nixl_micro_rig.geometry import TransferPlan
from tools.gemma4_pd.nixl_micro_rig.semantic_contract import (
    compute_rig_semantic_contract_digest,
)
from vllm.distributed.kv_transfer.integrity import (
    IntegrityIdentity,
    IntegrityObservation,
    IntegrityPayloadKind,
    IntegrityStage,
    compute_integrity_digest,
)

_MASK32 = 0xFFFFFFFF


@dataclass(frozen=True)
class ObservationContext:
    """Lineage shared by every leaf in one transfer iteration.

    :ivar run_id: Campaign UUID.
    :ivar transport_arm: Fresh-process UCX arm.
    :ivar producer_engine_id: Shared P engine identity.
    :ivar producer_request_id: P-side request identity.
    :ivar offer_generation: Offer and registration generation.
    :ivar iteration: Global campaign iteration.
    :ivar child_request_id: D child request identity.
    :ivar consumer_engine_id: D observer identity.
    """

    run_id: str
    transport_arm: str
    producer_engine_id: str
    producer_request_id: str
    offer_generation: int
    iteration: int
    child_request_id: str
    consumer_engine_id: str


def _mix32(values: torch.Tensor) -> torch.Tensor:
    """Apply an exact 32-bit avalanche permutation.

    :param values: Signed int64 tensor carrying unsigned 32-bit values.
    :returns: Mixed values restricted to 32 bits.
    """
    values = (values ^ (values >> 16)) * 0x7FEB352D
    values &= _MASK32
    values = (values ^ (values >> 15)) * 0x846CA68B
    values &= _MASK32
    return (values ^ (values >> 16)) & _MASK32


def expected_plane_bytes(
    *,
    plan: TransferPlan,
    position_start: int,
    position_count: int,
    source_rank: int,
    region_index: int,
    plane_index: int,
    iteration: int,
    row_bytes: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate an independent, identity-rich K or V payload oracle.

    The pattern is a pure function of the physical source address contract:
    iteration, rank, region, remote block, K/V half, and word offset. Repeated
    remote IDs therefore produce identical bytes even when distinct group
    positions refer to the same physical row. Every operation is integer and
    bit-exact across the source and consumer GPUs.

    :param plan: Stable-sorted transfer plan.
    :param position_start: First sorted transfer position.
    :param position_count: Number of rows in the batch.
    :param source_rank: Independent P rank.
    :param region_index: Physical registration region.
    :param plane_index: Zero for K and one for V.
    :param iteration: Global campaign iteration.
    :param row_bytes: Full P row bytes.
    :param device: CUDA device that receives the oracle.
    :returns: Contiguous uint8 tensor shaped ``(position_count, row_bytes / 2)``.
    """
    if plane_index not in {0, 1}:
        raise ValueError("plane_index must be 0 or 1")
    half_bytes = row_bytes // 2
    words_per_half = half_bytes // 4
    selected = plan.sorted_pairings[position_start : position_start + position_count]
    row_keys = []
    for pairing in selected:
        value = iteration * 0x9E3779B1
        value ^= source_rank * 0x85EBCA77
        value ^= region_index * 0xC2B2AE3D
        value ^= plane_index * 0x27D4EB2F
        value ^= pairing.remote_block_id * 0xB55A4F09
        row_keys.append(value & _MASK32)
    keys = torch.tensor(row_keys, dtype=torch.int64, device=device).view(-1, 1)
    word_offsets = torch.arange(words_per_half, dtype=torch.int64, device=device).view(
        1, -1
    )
    values = _mix32(keys ^ word_offsets)
    return values.to(torch.int32).view(torch.uint8).reshape(position_count, half_bytes)


def fill_source_rows(
    *,
    config: RigConfig,
    plan: TransferPlan,
    source_rank: int,
    iteration: int,
    regions: tuple[torch.Tensor, ...],
    batch_rows: int = 64,
) -> None:
    """Fill only selected P rows with deterministic K/V anti-patterns.

    :param config: Complete rig configuration.
    :param plan: Stable-sorted transfer plan.
    :param source_rank: P rank that owns ``regions``.
    :param iteration: Global campaign iteration.
    :param regions: Full registered P source tensors.
    :param batch_rows: Rows generated per GPU batch.
    """
    if len(regions) != len(config.regions):
        raise ValueError("source region count differs from configuration")
    device = regions[0].device
    for region_index, (region_config, region) in enumerate(
        zip(config.regions, regions, strict=True)
    ):
        rows = region.view(config.source_block_count, 2, region_config.row_bytes // 2)
        for position_start in range(0, plan.position_count, batch_rows):
            count = min(batch_rows, plan.position_count - position_start)
            block_ids = torch.tensor(
                plan.block_ids[position_start : position_start + count],
                dtype=torch.long,
                device=device,
            )
            for plane_index in (0, 1):
                expected = expected_plane_bytes(
                    plan=plan,
                    position_start=position_start,
                    position_count=count,
                    source_rank=source_rank,
                    region_index=region_index,
                    plane_index=plane_index,
                    iteration=iteration,
                    row_bytes=region_config.row_bytes,
                    device=device,
                )
                rows[block_ids, plane_index] = expected


def verify_source_rows(
    *,
    config: RigConfig,
    plan: TransferPlan,
    source_rank: int,
    iteration: int,
    regions: tuple[torch.Tensor, ...],
    batch_rows: int = 64,
) -> None:
    """Byte-compare registered P rows against an independent oracle.

    :param config: Complete rig configuration.
    :param plan: Stable-sorted transfer plan.
    :param source_rank: P rank that owns ``regions``.
    :param iteration: Global campaign iteration.
    :param regions: Full registered P source tensors.
    :param batch_rows: Rows verified per GPU batch.
    :raises RuntimeError: At the first rank/region/plane mismatch.
    """
    device = regions[0].device
    for region_index, (region_config, region) in enumerate(
        zip(config.regions, regions, strict=True)
    ):
        rows = region.view(config.source_block_count, 2, region_config.row_bytes // 2)
        for position_start in range(0, plan.position_count, batch_rows):
            count = min(batch_rows, plan.position_count - position_start)
            block_ids = torch.tensor(
                plan.block_ids[position_start : position_start + count],
                dtype=torch.long,
                device=device,
            )
            for plane_index in (0, 1):
                expected = expected_plane_bytes(
                    plan=plan,
                    position_start=position_start,
                    position_count=count,
                    source_rank=source_rank,
                    region_index=region_index,
                    plane_index=plane_index,
                    iteration=iteration,
                    row_bytes=region_config.row_bytes,
                    device=device,
                )
                if not torch.equal(rows[block_ids, plane_index], expected):
                    raise RuntimeError(
                        "source oracle mismatch at "
                        f"rank={source_rank} region={region_index} "
                        f"plane={plane_index} position={position_start}"
                    )


def verify_staging_rows(
    *,
    config: RigConfig,
    plan: TransferPlan,
    iteration: int,
    staging: torch.Tensor,
    staging_offset: int,
    batch_rows: int = 64,
) -> None:
    """Byte-compare raw D staging against the independent payload oracle.

    :param config: Complete rig configuration.
    :param plan: Stable-sorted transfer plan.
    :param iteration: Global campaign iteration.
    :param staging: Full registered D staging allocation.
    :param staging_offset: Allocation-generation byte offset.
    :param batch_rows: Rows verified per GPU batch.
    :raises RuntimeError: At the first rank/region/plane mismatch.
    """
    device = staging.device
    for region_index, region_config in enumerate(config.regions):
        start = staging_offset + plan.region_offsets[region_index]
        byte_count = plan.rank_count * plan.position_count * region_config.row_bytes
        rows = staging[start : start + byte_count].view(
            plan.rank_count,
            plan.position_count,
            2,
            region_config.row_bytes // 2,
        )
        for rank in range(plan.rank_count):
            for position_start in range(0, plan.position_count, batch_rows):
                count = min(batch_rows, plan.position_count - position_start)
                for plane_index in (0, 1):
                    expected = expected_plane_bytes(
                        plan=plan,
                        position_start=position_start,
                        position_count=count,
                        source_rank=rank,
                        region_index=region_index,
                        plane_index=plane_index,
                        iteration=iteration,
                        row_bytes=region_config.row_bytes,
                        device=device,
                    )
                    actual = rows[
                        rank,
                        position_start : position_start + count,
                        plane_index,
                    ]
                    if not torch.equal(actual, expected):
                        raise RuntimeError(
                            "staging oracle mismatch at "
                            f"rank={rank} region={region_index} "
                            f"plane={plane_index} position={position_start}"
                        )


def fill_destination_canary_rows(
    *,
    config: RigConfig,
    plan: TransferPlan,
    destinations: tuple[torch.Tensor, ...],
    value: int,
    batch_rows: int = 64,
) -> None:
    """Seed every destination row that raw transport must leave untouched.

    :param config: Complete rig configuration.
    :param plan: Stable-sorted transfer plan.
    :param destinations: Full registered D destination regions.
    :param value: Byte sentinel for this allocation generation.
    :param batch_rows: Destination rows seeded per GPU batch.
    """
    if value < 0 or value > 0xFF:
        raise ValueError("destination canary must be one byte")
    if len(destinations) != len(config.regions):
        raise ValueError("destination region count differs from configuration")
    device = destinations[0].device
    for region_config, destination in zip(config.regions, destinations, strict=True):
        rows = destination.view(
            config.source_block_count,
            2,
            plan.rank_count,
            region_config.row_bytes // 2,
        )
        for position_start in range(0, plan.position_count, batch_rows):
            local_ids = torch.tensor(
                plan.local_block_ids[position_start : position_start + batch_rows],
                dtype=torch.long,
                device=device,
            )
            rows[local_ids] = value


def verify_destination_canary_rows(
    *,
    config: RigConfig,
    plan: TransferPlan,
    destinations: tuple[torch.Tensor, ...],
    value: int,
    batch_rows: int = 64,
) -> None:
    """Prove raw NIXL reads did not write the prepared destination regions.

    :param config: Complete rig configuration.
    :param plan: Stable-sorted transfer plan.
    :param destinations: Full registered D destination regions.
    :param value: Byte sentinel seeded before native submission.
    :param batch_rows: Destination rows verified per GPU batch.
    :raises RuntimeError: If any destination byte changed before scatter.
    """
    device = destinations[0].device
    for region_index, (region_config, destination) in enumerate(
        zip(config.regions, destinations, strict=True)
    ):
        rows = destination.view(
            config.source_block_count,
            2,
            plan.rank_count,
            region_config.row_bytes // 2,
        )
        for position_start in range(0, plan.position_count, batch_rows):
            local_ids = torch.tensor(
                plan.local_block_ids[position_start : position_start + batch_rows],
                dtype=torch.long,
                device=device,
            )
            if not torch.all(rows.index_select(0, local_ids) == value).item():
                raise RuntimeError(
                    "destination changed before scatter at "
                    f"region={region_index} position={position_start}"
                )


def verify_destination_rows(
    *,
    config: RigConfig,
    plan: TransferPlan,
    iteration: int,
    destinations: tuple[torch.Tensor, ...],
    batch_rows: int = 64,
) -> None:
    """Byte-compare exact D scatter rows against an independent oracle.

    :param config: Complete rig configuration.
    :param plan: Stable-sorted transfer plan.
    :param iteration: Global campaign iteration.
    :param destinations: Full registered D destination regions.
    :param batch_rows: Rows verified per GPU batch.
    :raises RuntimeError: At the first rank/region/plane mismatch.
    """
    device = destinations[0].device
    for region_index, (region_config, destination) in enumerate(
        zip(config.regions, destinations, strict=True)
    ):
        rows = destination.view(
            config.source_block_count,
            2,
            plan.rank_count,
            region_config.row_bytes // 2,
        )
        for position_start in range(0, plan.position_count, batch_rows):
            count = min(batch_rows, plan.position_count - position_start)
            local_ids = torch.tensor(
                plan.local_block_ids[position_start : position_start + count],
                dtype=torch.long,
                device=device,
            )
            for rank in range(plan.rank_count):
                for plane_index in (0, 1):
                    expected = expected_plane_bytes(
                        plan=plan,
                        position_start=position_start,
                        position_count=count,
                        source_rank=rank,
                        region_index=region_index,
                        plane_index=plane_index,
                        iteration=iteration,
                        row_bytes=region_config.row_bytes,
                        device=device,
                    )
                    actual = rows[local_ids, plane_index, rank]
                    if not torch.equal(actual, expected):
                        raise RuntimeError(
                            "destination oracle mismatch at "
                            f"rank={rank} region={region_index} "
                            f"plane={plane_index} position={position_start}"
                        )


def _observation(
    *,
    config: RigConfig,
    context: ObservationContext,
    stage: IntegrityStage,
    plan: TransferPlan,
    source_rank: int,
    region_index: int,
    sorted_position: int,
    payload: memoryview,
) -> IntegrityObservation:
    """Hash one physical source row under its exact logical lineage.

    :param config: Complete typed rig configuration.
    :param context: Shared iteration lineage.
    :param stage: Source, staging, or destination observation point.
    :param plan: Exact stable-sorted transfer plan.
    :param source_rank: Producer rank owning the source bytes.
    :param region_index: Registered source-region index.
    :param sorted_position: Position within stable remote-ID order.
    :param payload: Exact source-comparable row bytes.
    :returns: Canonical integrity observation.
    """
    pairing = plan.sorted_pairings[sorted_position]
    group = config.groups[pairing.group_index]
    identity = IntegrityIdentity(
        run_id=context.run_id,
        transport_arm=context.transport_arm,
        producer_engine_id=context.producer_engine_id,
        producer_request_id=context.producer_request_id,
        registration_generation=(
            f"{context.run_id}:{context.transport_arm}:"
            f"producer-rank-{source_rank}:registration-1"
        ),
        semantic_contract_digest=compute_rig_semantic_contract_digest(
            config,
            pairing.group_index,
            region_index,
        ),
        offer_generation=context.offer_generation,
        iteration=context.iteration,
        source_rank=source_rank,
        region_index=region_index,
        group_index=pairing.group_index,
        plane_index=-1,
        source_position=pairing.group_position,
        remote_block_id=pairing.remote_block_id,
        valid_token_extent=config.valid_token_extent,
        group_token_capacity=group.token_capacity,
        payload_kind=IntegrityPayloadKind.WIRE,
        byte_length=payload.nbytes,
    )
    is_source = stage in (IntegrityStage.SOURCE_PRE, IntegrityStage.SOURCE_POST)
    return IntegrityObservation(
        stage=stage,
        identity=identity,
        digest=compute_integrity_digest(identity, payload),
        child_request_id=None if is_source else context.child_request_id,
        observer_engine_id=(
            context.producer_engine_id if is_source else context.consumer_engine_id
        ),
        observer_rank=source_rank if is_source else 0,
        local_block_id=(
            None if is_source else plan.sorted_pairings[sorted_position].local_block_id
        ),
    )


def source_observations(
    *,
    config: RigConfig,
    plan: TransferPlan,
    context: ObservationContext,
    stage: IntegrityStage,
    source_rank: int,
    regions: tuple[torch.Tensor, ...],
) -> list[IntegrityObservation]:
    """Capture strong source leaves from selected registered rows.

    :param config: Complete rig configuration.
    :param plan: Stable-sorted transfer plan.
    :param context: Iteration lineage.
    :param stage: Explicit pre-transfer or post-transfer source checkpoint.
    :param source_rank: P rank that owns ``regions``.
    :param regions: Full registered P source tensors.
    :returns: Region-major, position-ordered source observations.
    """
    if stage not in (IntegrityStage.SOURCE_PRE, IntegrityStage.SOURCE_POST):
        raise ValueError("source observations require a source integrity stage")
    observations: list[IntegrityObservation] = []
    device = regions[0].device
    block_ids = torch.tensor(plan.block_ids, dtype=torch.long, device=device)
    for region_index, (region_config, region) in enumerate(
        zip(config.regions, regions, strict=True)
    ):
        rows = region.view(config.source_block_count, region_config.row_bytes)
        selected = rows.index_select(0, block_ids).cpu()
        selected_bytes = memoryview(selected.numpy()).cast("B")
        for position in range(plan.position_count):
            start = position * region_config.row_bytes
            payload = selected_bytes[start : start + region_config.row_bytes]
            observations.append(
                _observation(
                    config=config,
                    context=context,
                    stage=stage,
                    plan=plan,
                    source_rank=source_rank,
                    region_index=region_index,
                    sorted_position=position,
                    payload=payload,
                )
            )
    return observations


def staging_observations(
    *,
    config: RigConfig,
    plan: TransferPlan,
    context: ObservationContext,
    staging: torch.Tensor,
    staging_offset: int,
) -> list[IntegrityObservation]:
    """Capture strong leaves from raw coalesced staging bytes.

    :param config: Complete rig configuration.
    :param plan: Stable-sorted transfer plan.
    :param context: Iteration lineage.
    :param staging: Full registered D staging allocation.
    :param staging_offset: Allocation-generation byte offset.
    :returns: Rank/region/position observations.
    """
    observations: list[IntegrityObservation] = []
    for region_index, region_config in enumerate(config.regions):
        region_start = staging_offset + plan.region_offsets[region_index]
        byte_count = plan.rank_count * plan.position_count * region_config.row_bytes
        host = staging[region_start : region_start + byte_count].cpu()
        host_bytes = memoryview(host.numpy()).cast("B")
        for rank in range(plan.rank_count):
            rank_start = rank * plan.position_count * region_config.row_bytes
            for position in range(plan.position_count):
                start = rank_start + position * region_config.row_bytes
                payload = host_bytes[start : start + region_config.row_bytes]
                observations.append(
                    _observation(
                        config=config,
                        context=context,
                        stage=IntegrityStage.STAGING_RAW,
                        plan=plan,
                        source_rank=rank,
                        region_index=region_index,
                        sorted_position=position,
                        payload=payload,
                    )
                )
    return observations


def destination_observations(
    *,
    config: RigConfig,
    plan: TransferPlan,
    context: ObservationContext,
    destinations: tuple[torch.Tensor, ...],
) -> list[IntegrityObservation]:
    """Capture source-comparable leaves from exact scattered D rows.

    :param config: Complete rig configuration.
    :param plan: Stable-sorted transfer plan.
    :param context: Iteration lineage.
    :param destinations: Full registered D destination regions.
    :returns: Rank/region/position observations in source wire order.
    """
    observations: list[IntegrityObservation] = []
    device = destinations[0].device
    local_ids = torch.tensor(plan.local_block_ids, dtype=torch.long, device=device)
    for region_index, (region_config, destination) in enumerate(
        zip(config.regions, destinations, strict=True)
    ):
        chunk = region_config.row_bytes // 2
        rows = destination.view(config.source_block_count, 2, plan.rank_count, chunk)
        selected = rows.index_select(0, local_ids)
        wire_order = selected.permute(2, 0, 1, 3).contiguous().cpu()
        host_bytes = memoryview(wire_order.numpy()).cast("B")
        for rank in range(plan.rank_count):
            rank_start = rank * plan.position_count * region_config.row_bytes
            for position in range(plan.position_count):
                start = rank_start + position * region_config.row_bytes
                payload = host_bytes[start : start + region_config.row_bytes]
                observations.append(
                    _observation(
                        config=config,
                        context=context,
                        stage=IntegrityStage.DESTINATION,
                        plan=plan,
                        source_rank=rank,
                        region_index=region_index,
                        sorted_position=position,
                        payload=payload,
                    )
                )
    return observations


def observation_key(
    observation: IntegrityObservation,
) -> tuple[int, int, int, int, int]:
    """Return the compact rank/region/source-position key.

    :param observation: Integrity observation.
    :returns: Rank, region, group, source ordinal, and remote block key.
    """
    identity = observation.identity
    return (
        identity.source_rank,
        identity.region_index,
        identity.group_index,
        identity.source_position,
        identity.remote_block_id,
    )


def compact_digests(
    observations: list[IntegrityObservation],
) -> list[list[int | str]]:
    """Encode observations for bounded control-plane comparison.

    :param observations: Full schema observations retained in local artifacts.
    :returns: Compact identity and digest rows.
    """
    return [
        [
            observation.identity.source_rank,
            observation.identity.region_index,
            observation.identity.group_index,
            observation.identity.source_position,
            observation.identity.remote_block_id,
            observation.digest.hex(),
        ]
        for observation in observations
    ]


def write_observations(
    path: Path,
    observations: list[IntegrityObservation],
    *,
    config_fingerprint: str,
    input_bundle_fingerprint: str,
    scenario: str,
) -> None:
    """Persist full schema observations as append-only JSON Lines.

    :param path: Output artifact path.
    :param observations: Observations to persist.
    :param config_fingerprint: SHA-256 identity of the untouched full config.
    :param input_bundle_fingerprint: SHA-256 identity of every immutable input.
    :param scenario: Selected scenario identity.
    """
    with path.open("a") as output:
        for observation in observations:
            record = asdict(observation)
            record["stage"] = observation.stage.value
            record["digest"] = observation.digest.hex()
            record["config_fingerprint"] = config_fingerprint
            record["input_bundle_fingerprint"] = input_bundle_fingerprint
            record["scenario"] = scenario
            identity = record["identity"]
            if not isinstance(identity, dict):
                raise AssertionError(
                    "dataclass identity did not serialize as an object"
                )
            identity["semantic_contract_digest"] = (
                observation.identity.semantic_contract_digest.hex()
            )
            identity["payload_kind"] = observation.identity.payload_kind.value
            record["evidence_status"] = observation.evidence_status.value
            output.write(json.dumps(record, sort_keys=True) + "\n")
