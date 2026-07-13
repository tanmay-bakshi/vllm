# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline validation for authoritative P-to-D localization artifacts."""

import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, TypeAlias

import msgspec

from vllm.distributed.kv_transfer.integrity import (
    IntegrityIdentity,
    IntegrityPayloadKind,
    IntegrityStage,
)
from vllm.distributed.kv_transfer.nixl_localization import (
    LOCALIZATION_ARTIFACT_MAGIC,
    LOCALIZATION_FRAME_PERSON,
    LOCALIZATION_ROOT_PERSON,
    IntegrityLeafKey,
    LocalizationError,
    LocalizationMode,
    NixlCaptureRecord,
    NixlEventRecord,
    NixlPlanPosition,
    NixlPlanRecord,
    NixlSessionRecord,
    NixlSourceManifest,
    NixlSourceManifestRecord,
    NixlTerminalRecord,
    compute_semantic_contract_digest,
    leaf_source_key,
    localization_request_id_base,
    locate_subsequence,
    select_source_manifest,
    validate_capture,
    validate_source_manifest_structure,
)

ArtifactRecord: TypeAlias = (
    NixlPlanRecord
    | NixlCaptureRecord
    | NixlSourceManifestRecord
    | NixlEventRecord
)
SourceLineage: TypeAlias = tuple[str, str, str, int, int, int]
ObserverKey: TypeAlias = tuple[str, str, int]
CaptureKey: TypeAlias = tuple[str, str, int, int, IntegrityStage]


@dataclass(frozen=True, slots=True)
class LocalizationArtifact:
    """One completely framed and authenticated process artifact.

    :ivar path: Artifact path.
    :ivar session: Process session header.
    :ivar records: Non-terminal process records in frame order.
    :ivar terminal: Final completeness marker.
    """

    path: Path
    session: NixlSessionRecord
    records: tuple[ArtifactRecord, ...]
    terminal: NixlTerminalRecord


@dataclass(frozen=True, slots=True)
class LocalizationDivergence:
    """First content or placement divergence for one observed source rank.

    :ivar child_request_id: Decoder child request.
    :ivar observer_engine_id: Decoder engine.
    :ivar observer_rank: Decoder tensor-parallel rank.
    :ivar source_rank: Producer tensor-parallel rank.
    :ivar edge: First failing observation edge.
    :ivar errors: Exact content, cardinality, or placement mismatches.
    """

    child_request_id: str
    observer_engine_id: str
    observer_rank: int
    source_rank: int
    edge: str
    errors: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LocalizationValidationReport:
    """Cross-process verdict for one complete localization artifact set.

    :ivar artifact_count: Number of complete process artifacts.
    :ivar physical_pull_count: Number of child/decoder-engine pull objects.
    :ivar verified_pull_count: Number of clean evidentiary physical pulls.
    :ivar excluded_outcome_count: Complete non-evidentiary child outcomes.
    :ivar terminal_outcome_count: Number of child/decoder-engine outcomes.
    :ivar divergences: First localized divergence per affected source rank.
    :ivar errors: Framing, lineage, completeness, or protocol errors.
    """

    artifact_count: int
    physical_pull_count: int
    verified_pull_count: int
    excluded_outcome_count: int
    terminal_outcome_count: int
    divergences: tuple[LocalizationDivergence, ...]
    errors: tuple[str, ...]

    @property
    def passed(self) -> bool:
        """Return whether the set is complete and contains no divergence.

        :returns: ``True`` only for a clean, complete instrumented trace.
        """
        return (
            len(self.errors) == 0
            and len(self.divergences) == 0
            and self.physical_pull_count > 0
            and self.verified_pull_count == self.physical_pull_count
        )


def _read_exact(file: BinaryIO, size: int, context: str) -> bytes:
    """Read an exact frame component.

    :param file: Open binary artifact.
    :param size: Required byte count.
    :param context: Component name for diagnostics.
    :returns: Exact bytes.
    :raises LocalizationError: If the artifact ends early.
    """
    data = file.read(size)
    if len(data) != size:
        raise LocalizationError(
            f"truncated localization artifact while reading {context}: "
            f"{len(data)}/{size} bytes"
        )
    return data


def _decode_record(
    payload: bytes,
    sequence: int,
) -> NixlSessionRecord | ArtifactRecord | NixlTerminalRecord:
    """Decode one record after selecting its exact array-like schema.

    :param payload: MessagePack frame payload.
    :param sequence: Frame sequence for diagnostics.
    :returns: Typed artifact record.
    :raises LocalizationError: If the record type or schema is malformed.
    """
    try:
        raw = msgspec.msgpack.decode(payload)
    except msgspec.DecodeError as error:
        raise LocalizationError(
            f"frame {sequence} is not valid MessagePack"
        ) from error
    if not isinstance(raw, list) or len(raw) == 0 or not isinstance(raw[0], str):
        raise LocalizationError(f"frame {sequence} has no record discriminator")
    record_type = raw[0]
    schema: type[
        NixlSessionRecord
        | NixlPlanRecord
        | NixlCaptureRecord
        | NixlSourceManifestRecord
        | NixlEventRecord
        | NixlTerminalRecord
    ]
    if record_type == NixlSessionRecord.RECORD_TYPE:
        schema = NixlSessionRecord
    elif record_type == NixlPlanRecord.RECORD_TYPE:
        schema = NixlPlanRecord
    elif record_type == NixlCaptureRecord.RECORD_TYPE:
        schema = NixlCaptureRecord
    elif record_type == NixlSourceManifestRecord.RECORD_TYPE:
        schema = NixlSourceManifestRecord
    elif record_type == NixlEventRecord.RECORD_TYPE:
        schema = NixlEventRecord
    elif record_type == NixlTerminalRecord.RECORD_TYPE:
        schema = NixlTerminalRecord
    else:
        raise LocalizationError(
            f"frame {sequence} has unknown record type {record_type!r}"
        )
    try:
        return msgspec.msgpack.decode(payload, type=schema)
    except (msgspec.DecodeError, msgspec.ValidationError) as error:
        raise LocalizationError(
            f"frame {sequence} does not match {record_type!r} schema"
        ) from error


def read_localization_artifact(
    path: str | Path,
    *,
    max_frame_bytes: int = 1024 * 1024 * 1024,
) -> LocalizationArtifact:
    """Read and authenticate one complete process artifact.

    :param path: Framed MessagePack artifact.
    :param max_frame_bytes: Defensive upper bound for one record payload.
    :returns: Complete typed artifact.
    :raises LocalizationError: If framing, checksums, sequence, root, or terminal
        completeness is invalid.
    """
    artifact_path = Path(path)
    if max_frame_bytes <= 0:
        raise ValueError("max_frame_bytes must be positive")
    sequence = 0
    root_hasher = hashlib.blake2b(
        digest_size=32,
        person=LOCALIZATION_ROOT_PERSON,
    )
    session: NixlSessionRecord | None = None
    records: list[ArtifactRecord] = []
    with artifact_path.open("rb") as file:
        while True:
            magic = file.read(len(LOCALIZATION_ARTIFACT_MAGIC))
            if len(magic) == 0:
                raise LocalizationError("localization artifact has no terminal frame")
            if len(magic) != len(LOCALIZATION_ARTIFACT_MAGIC):
                raise LocalizationError("truncated localization artifact frame magic")
            if magic != LOCALIZATION_ARTIFACT_MAGIC:
                raise LocalizationError(
                    f"frame {sequence} has invalid localization artifact magic"
                )
            sequence_bytes = _read_exact(file, 8, "frame sequence")
            length_bytes = _read_exact(file, 8, "frame length")
            checksum = _read_exact(file, 32, "frame checksum")
            frame_sequence = struct.unpack(">Q", sequence_bytes)[0]
            payload_length = struct.unpack(">Q", length_bytes)[0]
            if frame_sequence != sequence:
                raise LocalizationError(
                    f"artifact frame sequence is {frame_sequence}, expected {sequence}"
                )
            if payload_length > max_frame_bytes:
                raise LocalizationError(
                    f"frame {sequence} payload exceeds {max_frame_bytes} bytes"
                )
            payload = _read_exact(file, payload_length, "frame payload")
            expected_checksum = hashlib.blake2b(
                sequence_bytes + payload,
                digest_size=32,
                person=LOCALIZATION_FRAME_PERSON,
            ).digest()
            if checksum != expected_checksum:
                raise LocalizationError(f"frame {sequence} checksum mismatch")
            record = _decode_record(payload, sequence)
            if isinstance(record, NixlTerminalRecord):
                if session is None:
                    raise LocalizationError("terminal frame precedes session header")
                if record.schema_version != IntegrityIdentity.SCHEMA_VERSION:
                    raise LocalizationError("terminal record schema mismatch")
                if record.record_count != sequence:
                    raise LocalizationError(
                        "terminal record count does not match preceding frames"
                    )
                if record.root_digest != root_hasher.digest():
                    raise LocalizationError("artifact terminal root digest mismatch")
                if len(file.read(1)) != 0:
                    raise LocalizationError(
                        "artifact contains bytes after terminal frame"
                    )
                return LocalizationArtifact(
                    path=artifact_path,
                    session=session,
                    records=tuple(records),
                    terminal=record,
                )
            root_hasher.update(checksum)
            if sequence == 0:
                if not isinstance(record, NixlSessionRecord):
                    raise LocalizationError("first artifact record is not a session")
                session = record
            else:
                if isinstance(record, NixlSessionRecord):
                    raise LocalizationError("artifact contains a duplicate session")
                records.append(record)
            sequence += 1


def _source_lineage(manifest: NixlSourceManifest) -> SourceLineage:
    """Return the complete source-allocation lineage.

    :param manifest: Producer manifest.
    :returns: Engine, request, registration, offer, iteration, and rank.
    """
    return (
        manifest.producer_engine_id,
        manifest.producer_request_id,
        manifest.registration_generation,
        manifest.offer_generation,
        manifest.iteration,
        manifest.source_rank,
    )


def _record_scope_errors(
    artifact: LocalizationArtifact,
    record: ArtifactRecord,
) -> tuple[str, ...]:
    """Validate a record against the process session that emitted it.

    :param artifact: Containing process artifact.
    :param record: Record to validate.
    :returns: Scope and observer-lineage errors.
    """
    session = artifact.session
    errors: list[str] = []
    if isinstance(record, NixlSourceManifestRecord):
        manifest = record.manifest
        if record.stage not in (
            IntegrityStage.SOURCE_PRE,
            IntegrityStage.SOURCE_POST,
        ):
            errors.append("source manifest record has a non-source stage")
        if manifest.run_id != session.run_id:
            errors.append("source manifest run differs from its session")
        if manifest.transport_arm != session.transport_arm:
            errors.append("source manifest arm differs from its session")
        if manifest.producer_engine_id != session.engine_id:
            errors.append("source manifest engine differs from its session")
        if manifest.source_rank != session.rank:
            errors.append("source manifest rank differs from its session")
        if (
            localization_request_id_base(manifest.producer_request_id)
            != session.target_request_id
        ):
            errors.append("source manifest request differs from its session target")
        return tuple(errors)

    if record.run_id != session.run_id:
        errors.append(f"{record.record_type} run differs from its session")
    if record.transport_arm != session.transport_arm:
        errors.append(f"{record.record_type} arm differs from its session")
    if record.observer_engine_id != session.engine_id:
        errors.append(f"{record.record_type} engine differs from its session")
    if record.observer_rank != session.rank:
        errors.append(f"{record.record_type} rank differs from its session")
    if (
        record.child_request_id is None
        or localization_request_id_base(record.child_request_id)
        != session.target_request_id
    ):
        errors.append(f"{record.record_type} child differs from its session target")
    if (
        record.producer_request_id is not None
        and localization_request_id_base(record.producer_request_id)
        != session.target_request_id
    ):
        errors.append(f"{record.record_type} source differs from its session target")
    return tuple(errors)


def _expected_plan_positions(
    plan: NixlPlanRecord,
) -> tuple[dict[tuple[int, int, int], tuple[int, int]], tuple[str, ...]]:
    """Reconstruct canonical mapping before transfer-order sorting.

    :param plan: Decoder transfer plan.
    :returns: Source tuple to local block and destination half, plus errors.
    """
    expected: dict[tuple[int, int, int], tuple[int, int]] = {}
    errors: list[str] = []
    group_count = len(plan.raw_remote_groups)
    if (
        len(plan.selected_remote_groups) != group_count
        or len(plan.selected_local_groups) != group_count
        or len(plan.destination_group_planes) != group_count
    ):
        return {}, ("plan group cardinality mismatch",)
    for group_index in range(group_count):
        raw = list(plan.raw_remote_groups[group_index])
        remote = list(plan.selected_remote_groups[group_index])
        local = list(plan.selected_local_groups[group_index])
        try:
            source_start = locate_subsequence(raw, remote)
        except LocalizationError as error:
            errors.append(f"plan group {group_index}: {error}")
            continue
        planes = plan.destination_group_planes[group_index]
        if planes == 2:
            if len(local) != len(remote):
                errors.append(f"plan group {group_index} dual-plane length mismatch")
                continue
            local_for_position = local
            halves = [-1] * len(remote)
        elif planes == 1:
            if len(remote) > 2 * len(local):
                errors.append(f"plan group {group_index} single-plane length mismatch")
                continue
            local_for_position = [local[index // 2] for index in range(len(remote))]
            halves = [
                (source_start + index) % 2 for index in range(len(remote))
            ]
        else:
            errors.append(f"plan group {group_index} has invalid plane contract")
            continue
        for index, block_id in enumerate(remote):
            key = (group_index, source_start + index, block_id)
            if key in expected:
                errors.append(f"plan contains duplicate canonical source tuple {key}")
                continue
            expected[key] = (local_for_position[index], halves[index])
    return expected, tuple(errors)


def _plan_errors(
    plan: NixlPlanRecord,
    source_by_digest: dict[bytes, NixlSourceManifest],
) -> tuple[str, ...]:
    """Validate one exact transfer plan against its source manifests.

    :param plan: Decoder plan artifact.
    :param source_by_digest: SOURCE_PRE manifests keyed by sealed digest.
    :returns: Plan, rank, semantic, geometry, and placement errors.
    """
    errors: list[str] = []
    rank_count = len(plan.source_ranks)
    if (
        len(plan.rank_slots) != rank_count
        or len(plan.registration_generations) != rank_count
        or len(plan.source_manifest_digests) != rank_count
        or len(plan.regions) != rank_count
    ):
        errors.append("plan source-rank cardinality mismatch")
        return tuple(errors)
    if len(set(plan.source_ranks)) != rank_count:
        errors.append("plan contains duplicate source ranks")
    if sorted(plan.rank_slots) != list(range(rank_count)):
        errors.append("plan rank slots are not a complete bijection")
    if plan.rank_slot_contract != tuple(sorted(plan.rank_slot_contract)):
        errors.append("plan rank-slot contract is not canonical")
    contract = dict(plan.rank_slot_contract)
    if (
        len(contract) != len(plan.rank_slot_contract)
        or set(contract) != set(plan.source_ranks)
        or sorted(contract.values()) != list(range(rank_count))
    ):
        errors.append("plan rank-slot contract is not a complete bijection")
    else:
        for source_rank, rank_slot in zip(
            plan.source_ranks,
            plan.rank_slots,
            strict=True,
        ):
            if contract[source_rank] != rank_slot:
                errors.append(
                    f"plan source rank {source_rank} violates rank-slot contract"
                )

    manifests: list[NixlSourceManifest] = []
    for index, source_rank in enumerate(plan.source_ranks):
        digest = plan.source_manifest_digests[index]
        manifest = source_by_digest.get(digest)
        if manifest is None:
            errors.append(f"plan source rank {source_rank} references no SOURCE_PRE")
            continue
        manifests.append(manifest)
        if manifest.source_rank != source_rank:
            errors.append(f"plan source rank {source_rank} manifest rank mismatch")
        if manifest.registration_generation != plan.registration_generations[index]:
            errors.append(
                f"plan source rank {source_rank} registration generation mismatch"
            )
        if manifest.regions != plan.regions[index]:
            errors.append(f"plan source rank {source_rank} region roster mismatch")
        if (
            manifest.run_id != plan.run_id
            or manifest.transport_arm != plan.transport_arm
            or manifest.producer_engine_id != plan.producer_engine_id
            or manifest.producer_request_id != plan.producer_request_id
            or manifest.offer_generation != plan.offer_generation
            or manifest.iteration != plan.iteration
            or manifest.block_ids != plan.raw_remote_groups
            or manifest.region_lengths != plan.region_lengths
        ):
            errors.append(f"plan source rank {source_rank} lineage mismatch")
    if len(manifests) == rank_count and rank_count > 0:
        reference = manifests[0]
        for manifest in manifests[1:]:
            if (
                manifest.valid_token_extent != reference.valid_token_extent
                or manifest.group_token_capacities
                != reference.group_token_capacities
                or manifest.source_group_planes != reference.source_group_planes
            ):
                errors.append("plan source ranks disagree on semantic token contract")
        if len(plan.destination_group_planes) != len(
            reference.group_token_capacities
        ):
            errors.append("plan destination plane-contract cardinality mismatch")

    expected_positions, mapping_errors = _expected_plan_positions(plan)
    errors.extend(mapping_errors)
    actual_positions: dict[tuple[int, int, int], NixlPlanPosition] = {}
    for position in plan.transfer_order:
        key = (
            position.group_index,
            position.source_position,
            position.remote_block_id,
        )
        if key in actual_positions:
            errors.append(f"plan transfer order duplicates source tuple {key}")
            continue
        actual_positions[key] = position
    if set(actual_positions) != set(expected_positions):
        errors.append("plan transfer order differs from selected group rosters")
    reference = manifests[0] if len(manifests) > 0 else None
    for key in set(actual_positions) & set(expected_positions):
        position = actual_positions[key]
        if (position.local_block_id, position.plane_index) != expected_positions[key]:
            errors.append(f"plan destination mapping mismatch {key}")
        if reference is not None:
            group_index = position.group_index
            if (
                position.valid_token_extent != reference.valid_token_extent
                or position.group_token_capacity
                != reference.group_token_capacities[group_index]
            ):
                errors.append(f"plan token contract mismatch {key}")

    remote_order = [position.remote_block_id for position in plan.transfer_order]
    if remote_order != sorted(remote_order):
        errors.append("plan transfer order is not sorted by remote block id")
    expected_runs: list[tuple[int, int, int]] = []
    if len(remote_order) > 0:
        run_start = 0
        for index in range(1, len(remote_order) + 1):
            if (
                index == len(remote_order)
                or remote_order[index] != remote_order[index - 1] + 1
            ):
                expected_runs.append(
                    (remote_order[run_start], index - run_start, run_start)
                )
                run_start = index
    if tuple(expected_runs) != plan.runs:
        errors.append("plan coalesced runs do not match transfer order")

    expected_offsets: list[int] = []
    expected_size = 0
    position_count = len(plan.transfer_order)
    for row_bytes in plan.region_lengths:
        expected_offsets.append(expected_size)
        expected_size += position_count * rank_count * row_bytes
    if tuple(expected_offsets) != plan.region_offsets:
        errors.append("plan region offsets do not match transfer geometry")
    if expected_size != plan.staging_size:
        errors.append("plan staging size does not match transfer geometry")
    if plan.staging_offset < 0:
        errors.append("plan staging offset is negative")
    if len(plan.local_regions) != len(plan.region_lengths):
        errors.append("plan local region cardinality mismatch")
    elif len(manifests) > 0:
        for source_region, local_region in zip(
            manifests[0].regions,
            plan.local_regions,
            strict=True,
        ):
            if (
                source_region.semantic_name != local_region.semantic_name
                or source_region.group_indices != local_region.group_indices
                or source_region.group_semantic_names
                != local_region.group_semantic_names
                or source_region.dtype != local_region.dtype
                or source_region.element_size_bytes
                != local_region.element_size_bytes
                or source_region.layout != local_region.layout
            ):
                errors.append("plan local/source semantic region mismatch")
            if local_region.row_bytes != rank_count * source_region.row_bytes:
                errors.append("plan local/source region row geometry mismatch")
            for role, region in (
                ("source", source_region),
                ("local", local_region),
            ):
                if (
                    len(region.shape) == 0
                    or len(region.shape) != len(region.strides)
                    or region.registered_bytes
                    != region.shape[0] * region.row_bytes
                    or region.strides[0] * region.element_size_bytes
                    != region.row_bytes
                ):
                    errors.append(f"plan {role} region is not row canonical")
    return tuple(errors)


def validate_localization_plan(
    plan: NixlPlanRecord,
    manifests: tuple[NixlSourceManifest, ...],
) -> tuple[str, ...]:
    """Validate one plan against an explicit SOURCE_PRE manifest set.

    :param plan: Decoder plan artifact.
    :param manifests: Exact source manifests referenced by the plan.
    :returns: Plan and semantic mapping errors.
    """
    return _plan_errors(
        plan,
        {manifest.manifest_digest: manifest for manifest in manifests},
    )


def _capture_contract(
    plan: NixlPlanRecord,
    manifest: NixlSourceManifest,
    stage: IntegrityStage,
) -> tuple[
    set[IntegrityLeafKey],
    dict[IntegrityLeafKey, tuple[int, int, int]],
    int,
    int,
]:
    """Reconstruct exact semantic leaves and byte counts for one stage.

    :param plan: Canonical decoder transfer plan.
    :param manifest: Referenced SOURCE_PRE manifest.
    :param stage: Decoder observation stage.
    :returns: Leaf keys, placement mapping, copied bytes, and hashed bytes.
    """
    rank_index = plan.source_ranks.index(manifest.source_rank)
    rank_slot = plan.rank_slots[rank_index]
    keys: set[IntegrityLeafKey] = set()
    mapping: dict[IntegrityLeafKey, tuple[int, int, int]] = {}
    copied_bytes = 0
    hashed_bytes = 0
    source_leaves = {leaf_source_key(leaf): leaf for leaf in manifest.leaves}
    for region_index, region in enumerate(manifest.regions):
        for position in plan.transfer_order:
            group_index = position.group_index
            if group_index not in region.group_indices:
                continue
            if stage in (
                IntegrityStage.STAGING_RAW,
                IntegrityStage.STAGING_FENCED_CONTROL,
            ):
                copied_bytes += region.row_bytes
                contracts = [(-1, IntegrityPayloadKind.WIRE)]
                if plan.destination_group_planes[group_index] == 1:
                    contracts.append((0, IntegrityPayloadKind.COMMIT))
            elif stage in (IntegrityStage.DESTINATION, IntegrityStage.PRE_READ):
                if plan.destination_group_planes[group_index] == 1:
                    copied_bytes += region.row_bytes // 2
                    contracts = [(0, IntegrityPayloadKind.COMMIT)]
                else:
                    copied_bytes += region.row_bytes
                    contracts = [(-1, IntegrityPayloadKind.WIRE)]
            else:
                raise ValueError(f"unsupported decoder capture stage {stage}")
            for plane_index, payload_kind in contracts:
                semantic_contract_digest = compute_semantic_contract_digest(
                    region=region,
                    group_index=group_index,
                    group_token_capacity=position.group_token_capacity,
                    source_plane_contract=manifest.source_group_planes[
                        group_index
                    ],
                )
                key: IntegrityLeafKey = (
                    region_index,
                    group_index,
                    position.source_position,
                    position.remote_block_id,
                    plane_index,
                    position.valid_token_extent,
                    position.group_token_capacity,
                    semantic_contract_digest,
                    payload_kind.value,
                )
                source_leaf = source_leaves.get(key)
                if source_leaf is None:
                    raise LocalizationError(
                        f"capture contract has no source leaf {key}"
                    )
                keys.add(key)
                mapping[key] = (
                    position.local_block_id,
                    rank_slot,
                    position.plane_index,
                )
                hashed_bytes += source_leaf.byte_length
    return keys, mapping, copied_bytes, hashed_bytes


def _capture_errors(
    plan: NixlPlanRecord,
    manifest: NixlSourceManifest,
    capture: NixlCaptureRecord,
    *,
    compare_digests: bool,
) -> tuple[str, ...]:
    """Validate one decoder observation against source and plan.

    :param plan: Canonical decoder transfer plan.
    :param manifest: Referenced SOURCE_PRE manifest.
    :param capture: Stage capture.
    :param compare_digests: Whether this is an evidentiary trace arm.
    :returns: Exact lineage, byte, cardinality, content, and mapping errors.
    """
    errors: list[str] = []
    if (
        capture.run_id != plan.run_id
        or capture.transport_arm != plan.transport_arm
        or capture.producer_engine_id != plan.producer_engine_id
        or capture.producer_request_id != plan.producer_request_id
        or capture.registration_generation != manifest.registration_generation
        or capture.source_manifest_digest != manifest.manifest_digest
        or capture.offer_generation != plan.offer_generation
        or capture.iteration != plan.iteration
        or capture.child_request_id != plan.child_request_id
        or capture.source_rank != manifest.source_rank
        or capture.observer_engine_id != plan.observer_engine_id
        or capture.observer_rank != plan.observer_rank
    ):
        errors.append("capture lineage differs from plan or SOURCE_PRE")
    keys, mapping, copied_bytes, hashed_bytes = _capture_contract(
        plan,
        manifest,
        capture.stage,
    )
    selected_manifest = select_source_manifest(manifest, keys)
    errors.extend(
        validate_capture(
            selected_manifest,
            capture.leaves,
            mapping,
            compare_digests=compare_digests,
        )
    )
    if capture.copied_bytes != copied_bytes:
        errors.append("capture copied-byte count differs from semantic contract")
    if capture.hashed_bytes != hashed_bytes:
        errors.append("capture hashed-byte count differs from semantic contract")
    if capture.observer is False:
        errors.append("capture is not labeled as an observer")
    expected_barriers = {
        IntegrityStage.STAGING_RAW: "nixl_done_without_added_device_wide_sync",
        IntegrityStage.STAGING_FENCED_CONTROL: (
            "device_synchronize_observer_control_not_gdr_flush"
        ),
        IntegrityStage.DESTINATION: (
            "post_scatter_device_synchronize_before_publication"
        ),
        IntegrityStage.PRE_READ: (
            "first_operation_in_start_load_kv_before_new_dma_or_forward"
        ),
    }
    if capture.barrier != expected_barriers[capture.stage]:
        errors.append("capture barrier label differs from stage contract")
    return tuple(errors)


def _first_divergence(
    plan: NixlPlanRecord,
    manifest: NixlSourceManifest,
    captures: dict[CaptureKey, NixlCaptureRecord],
    *,
    compare_digests: bool,
) -> LocalizationDivergence | None:
    """Find the earliest failing decoder edge for one source rank.

    :param plan: Decoder plan.
    :param manifest: SOURCE_PRE manifest.
    :param captures: Complete capture index.
    :param compare_digests: Whether the arm carries content evidence.
    :returns: Earliest divergence, or ``None`` when all present stages match.
    """
    ordered_edges = (
        (IntegrityStage.STAGING_RAW, "source_pre->staging_raw"),
        (
            IntegrityStage.STAGING_FENCED_CONTROL,
            "staging_raw->staging_fenced_control",
        ),
        (IntegrityStage.DESTINATION, "staging_fenced_control->destination"),
        (IntegrityStage.PRE_READ, "destination->pre_read"),
    )
    for stage, edge in ordered_edges:
        key: CaptureKey = (
            plan.child_request_id,
            plan.observer_engine_id,
            plan.observer_rank,
            manifest.source_rank,
            stage,
        )
        capture = captures.get(key)
        if capture is None:
            continue
        try:
            errors = _capture_errors(
                plan,
                manifest,
                capture,
                compare_digests=compare_digests,
            )
        except (LocalizationError, ValueError) as error:
            errors = (str(error),)
        if len(errors) > 0:
            return LocalizationDivergence(
                child_request_id=plan.child_request_id,
                observer_engine_id=plan.observer_engine_id,
                observer_rank=plan.observer_rank,
                source_rank=manifest.source_rank,
                edge=edge,
                errors=errors,
            )
    return None


def validate_localization_artifacts(
    paths: list[str | Path] | tuple[str | Path, ...],
) -> LocalizationValidationReport:
    """Validate a complete multi-process localization artifact set.

    The validator rejects partial files, mixed runs, duplicate processes,
    stale manifests, rank-incomplete pulls, ambiguous mappings, missing terminal
    outcomes, and any content or placement divergence.

    :param paths: Every P and D worker artifact from one diagnostic run.
    :returns: Structured complete-set verdict.
    """
    errors: list[str] = []
    artifacts: list[LocalizationArtifact] = []
    for raw_path in paths:
        try:
            artifacts.append(read_localization_artifact(raw_path))
        except (OSError, LocalizationError) as error:
            errors.append(f"{Path(raw_path)}: {error}")
    if len(artifacts) == 0:
        if len(errors) == 0:
            errors.append("no localization artifacts were supplied")
        return LocalizationValidationReport(
            artifact_count=0,
            physical_pull_count=0,
            verified_pull_count=0,
            excluded_outcome_count=0,
            terminal_outcome_count=0,
            divergences=(),
            errors=tuple(errors),
        )

    reference_session = artifacts[0].session
    sessions: dict[tuple[str, int], LocalizationArtifact] = {}
    source_pre_by_digest: dict[bytes, NixlSourceManifest] = {}
    source_pre_by_lineage: dict[SourceLineage, NixlSourceManifest] = {}
    source_post_by_lineage: dict[SourceLineage, NixlSourceManifest] = {}
    plans: dict[ObserverKey, NixlPlanRecord] = {}
    captures: dict[CaptureKey, NixlCaptureRecord] = {}
    events: dict[ObserverKey, NixlEventRecord] = {}

    for artifact in artifacts:
        session = artifact.session
        session_key = (session.engine_id, session.rank)
        if session_key in sessions:
            errors.append(f"duplicate process artifact for {session_key}")
        else:
            sessions[session_key] = artifact
        if (
            session.schema_version != IntegrityIdentity.SCHEMA_VERSION
            or session.run_id != reference_session.run_id
            or session.transport_arm != reference_session.transport_arm
            or session.mode is not reference_session.mode
            or session.target_request_id != reference_session.target_request_id
            or len(session.target_request_id) == 0
            or localization_request_id_base(session.target_request_id)
            != session.target_request_id
            or session.observer is False
            or session.claim_scope != "instrumented_only"
        ):
            errors.append(f"mixed or invalid session metadata in {artifact.path}")
        for record in artifact.records:
            for scope_error in _record_scope_errors(artifact, record):
                errors.append(f"{artifact.path}: {scope_error}")
            if isinstance(record, NixlSourceManifestRecord):
                manifest = record.manifest
                structure_errors = validate_source_manifest_structure(manifest)
                for structure_error in structure_errors:
                    errors.append(
                        f"{artifact.path}: {record.stage.value}: {structure_error}"
                    )
                lineage = _source_lineage(manifest)
                if record.stage is IntegrityStage.SOURCE_PRE:
                    if lineage in source_pre_by_lineage:
                        errors.append(f"duplicate SOURCE_PRE lineage {lineage}")
                    else:
                        source_pre_by_lineage[lineage] = manifest
                    if manifest.manifest_digest in source_pre_by_digest:
                        errors.append("duplicate SOURCE_PRE manifest digest")
                    else:
                        source_pre_by_digest[manifest.manifest_digest] = manifest
                else:
                    if lineage in source_post_by_lineage:
                        errors.append(f"duplicate SOURCE_POST lineage {lineage}")
                    else:
                        source_post_by_lineage[lineage] = manifest
            elif isinstance(record, NixlPlanRecord):
                key: ObserverKey = (
                    record.child_request_id,
                    record.observer_engine_id,
                    record.observer_rank,
                )
                if key in plans:
                    errors.append(f"duplicate transfer plan {key}")
                else:
                    plans[key] = record
            elif isinstance(record, NixlCaptureRecord):
                if record.child_request_id is None:
                    errors.append("decoder capture has no child request id")
                    continue
                key: CaptureKey = (
                    record.child_request_id,
                    record.observer_engine_id,
                    record.observer_rank,
                    record.source_rank,
                    record.stage,
                )
                if key in captures:
                    errors.append(f"duplicate decoder capture {key}")
                else:
                    captures[key] = record
            else:
                if record.child_request_id is None:
                    errors.append("localization event has no child request id")
                    continue
                key = (
                    record.child_request_id,
                    record.observer_engine_id,
                    record.observer_rank,
                )
                if key in events:
                    errors.append(f"duplicate child terminal event {key}")
                else:
                    events[key] = record

    for lineage, pre_manifest in source_pre_by_lineage.items():
        post_manifest = source_post_by_lineage.get(lineage)
        if post_manifest is None:
            errors.append(f"SOURCE_PRE has no SOURCE_POST bookend {lineage}")
            continue
        post_errors = validate_capture(pre_manifest, post_manifest.leaves, None)
        for post_error in post_errors:
            errors.append(f"source_pre->source_post {lineage}: {post_error}")
    for lineage in set(source_post_by_lineage) - set(source_pre_by_lineage):
        errors.append(f"SOURCE_POST has no SOURCE_PRE {lineage}")

    divergences: list[LocalizationDivergence] = []
    capture_stages = (
        IntegrityStage.STAGING_RAW,
        IntegrityStage.STAGING_FENCED_CONTROL,
        IntegrityStage.DESTINATION,
        IntegrityStage.PRE_READ,
    )
    for observer_key, plan in plans.items():
        plan_errors = _plan_errors(plan, source_pre_by_digest)
        for plan_error in plan_errors:
            errors.append(f"plan {observer_key}: {plan_error}")
        event = events.get(observer_key)
        completed = event is not None and event.code in (
            "VERIFIED_PRE_READ",
            "SHAM_PRE_READ_COMPLETE",
        )
        if event is None:
            errors.append(f"plan has no terminal child outcome {observer_key}")
        if len(plan_errors) > 0:
            continue
        for source_rank, digest in zip(
            plan.source_ranks,
            plan.source_manifest_digests,
            strict=True,
        ):
            manifest = source_pre_by_digest.get(digest)
            if manifest is None:
                continue
            divergence = _first_divergence(
                plan,
                manifest,
                captures,
                compare_digests=reference_session.mode is LocalizationMode.TRACE,
            )
            if divergence is not None:
                divergences.append(divergence)
            if completed:
                for stage in capture_stages:
                    capture_key: CaptureKey = (
                        plan.child_request_id,
                        plan.observer_engine_id,
                        plan.observer_rank,
                        source_rank,
                        stage,
                    )
                    if capture_key not in captures:
                        errors.append(f"verified pull is missing capture {capture_key}")

    for capture_key in captures:
        observer_key = (capture_key[0], capture_key[1], capture_key[2])
        if observer_key not in plans:
            errors.append(f"decoder capture has no canonical plan {capture_key}")

    allowed_outcomes = {
        "VERIFIED_PRE_READ",
        "SHAM_PRE_READ_COMPLETE",
        "NON_EVIDENTIARY_ZERO_BYTE",
        "TRANSFER_ABORTED",
        "REQUEST_ABORTED",
        "TRANSFER_RETRY",
    }
    child_engines = {(key[0], key[1]) for key in events}
    if len(child_engines) == 0:
        errors.append("artifact set contains no child terminal outcomes")
    if len(source_pre_by_lineage) == 0:
        errors.append("artifact set contains no SOURCE_PRE manifests")
    for child_id, observer_engine in child_engines:
        expected_ranks = {
            rank for engine, rank in sessions if engine == observer_engine
        }
        child_events = {
            rank: event
            for (child, engine, rank), event in events.items()
            if child == child_id and engine == observer_engine
        }
        if set(child_events) != expected_ranks:
            errors.append(
                f"child {(child_id, observer_engine)} terminal ranks are incomplete"
            )
        outcome_codes = {event.code for event in child_events.values()}
        if len(outcome_codes) != 1:
            errors.append(
                f"child {(child_id, observer_engine)} ranks disagree on outcome"
            )
        for event in child_events.values():
            if event.code not in allowed_outcomes:
                errors.append(f"unknown localization terminal outcome {event.code}")
            if event.code == "VERIFIED_PRE_READ" and event.evidentiary is False:
                errors.append("verified pre-read outcome is not evidentiary")
            if event.code != "VERIFIED_PRE_READ" and event.evidentiary:
                errors.append(f"non-verified outcome {event.code} is evidentiary")
        child_plan_keys = {
            key for key in plans if key[0] == child_id and key[1] == observer_engine
        }
        child_capture_keys = {
            key
            for key in captures
            if key[0] == child_id and key[1] == observer_engine
        }
        if outcome_codes in (
            {"VERIFIED_PRE_READ"},
            {"SHAM_PRE_READ_COMPLETE"},
        ):
            if {key[2] for key in child_plan_keys} != expected_ranks:
                errors.append(
                    f"verified child {(child_id, observer_engine)} plans are incomplete"
                )
        elif outcome_codes == {"NON_EVIDENTIARY_ZERO_BYTE"} and (
            len(child_plan_keys) > 0 or len(child_capture_keys) > 0
        ):
            errors.append("zero-byte child contains physical-pull evidence")

    for observer_key in plans:
        if observer_key not in events:
            continue
        event = events[observer_key]
        if (
            event.code == "VERIFIED_PRE_READ"
            and reference_session.mode is LocalizationMode.SHAM
        ):
            errors.append("sham arm cannot emit an evidentiary verified outcome")
        if (
            event.code == "SHAM_PRE_READ_COMPLETE"
            and reference_session.mode is LocalizationMode.TRACE
        ):
            errors.append("trace arm cannot emit a sham completion outcome")

    physical_pulls = {(key[0], key[1]) for key in plans}
    verified_pulls = {
        (key[0], key[1])
        for key, event in events.items()
        if event.code == "VERIFIED_PRE_READ"
        and (key[0], key[1]) in physical_pulls
    }
    excluded_outcomes = {
        (key[0], key[1])
        for key, event in events.items()
        if event.code != "VERIFIED_PRE_READ"
    }
    return LocalizationValidationReport(
        artifact_count=len(artifacts),
        physical_pull_count=len(physical_pulls),
        verified_pull_count=len(verified_pulls),
        excluded_outcome_count=len(excluded_outcomes),
        terminal_outcome_count=len(child_engines),
        divergences=tuple(divergences),
        errors=tuple(errors),
    )
