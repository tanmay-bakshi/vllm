# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline validation for authoritative P-to-D localization artifacts."""

import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, TypeAlias

import msgspec

from vllm.distributed.kv_transfer.coalesced_layout import (
    CoalescedTransferPlan,
    GroupTransferRoster,
    RegionOwnership,
    build_coalesced_transfer_plan,
)
from vllm.distributed.kv_transfer.integrity import (
    IntegrityIdentity,
    IntegrityPayloadKind,
    IntegrityStage,
)
from vllm.distributed.kv_transfer.nixl_contracts import NixlRegionDescriptor
from vllm.distributed.kv_transfer.nixl_localization import (
    LOCALIZATION_ARTIFACT_MAGIC,
    LOCALIZATION_FRAME_PERSON,
    LOCALIZATION_ROOT_PERSON,
    IntegrityLeafKey,
    LocalizationError,
    LocalizationFingerprintAlgorithm,
    LocalizationMode,
    NixlCaptureRecord,
    NixlEventRecord,
    NixlPlanPosition,
    NixlPlanRecord,
    NixlPlanRun,
    NixlRegionPlan,
    NixlSessionRecord,
    NixlSourceContract,
    NixlSourceManifest,
    NixlSourceManifestRecord,
    NixlTerminalRecord,
    compute_semantic_contract_digest,
    localization_child_index,
    localization_fingerprint_size,
    localization_producer_target,
    localization_request_id_base,
    localization_request_target,
    localization_stage_barrier,
    select_source_manifest,
    source_contract_from_manifest,
    validate_capture,
    validate_source_contract_structure,
    validate_source_manifest_structure,
)

ArtifactRecord: TypeAlias = (
    NixlPlanRecord | NixlCaptureRecord | NixlSourceManifestRecord | NixlEventRecord
)
SourceLineage: TypeAlias = tuple[str, str, str, str, str, int, int, int]
ObserverKey: TypeAlias = tuple[str, str, int]
CaptureKey: TypeAlias = tuple[ObserverKey, SourceLineage, IntegrityStage]


def _capture_stages(mode: LocalizationMode) -> tuple[IntegrityStage, ...]:
    """Return the complete decoder checkpoint sequence for one mode.

    :param mode: Observer mode recorded by every process session.
    :returns: Ordered decoder capture stages.
    """
    if mode is LocalizationMode.FINGERPRINT:
        return (
            IntegrityStage.STAGING_POST_SCATTER,
            IntegrityStage.DESTINATION,
            IntegrityStage.PRE_READ,
        )
    return (
        IntegrityStage.STAGING_RAW,
        IntegrityStage.STAGING_FENCED_CONTROL,
        IntegrityStage.DESTINATION,
        IntegrityStage.PRE_READ,
    )


@dataclass(frozen=True, slots=True)
class LocalizationArtifact:
    """One completely framed and integrity-checked process artifact.

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
        raise LocalizationError(f"frame {sequence} is not valid MessagePack") from error
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
    """Read and verify one complete process artifact.

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
    :returns: Run, arm, engine, request, registration, offer, iteration, and rank.
    """
    return (
        manifest.run_id,
        manifest.transport_arm,
        manifest.producer_engine_id,
        manifest.producer_request_id,
        manifest.registration_generation,
        manifest.offer_generation,
        manifest.iteration,
        manifest.source_rank,
    )


def _contract_lineage(contract: NixlSourceContract) -> SourceLineage:
    """Return the complete lineage encoded by a decoder source contract.

    :param contract: Content-free decoder-side source contract.
    :returns: Run, arm, engine, request, registration, offer, iteration, and rank.
    """
    return (
        contract.run_id,
        contract.transport_arm,
        contract.producer_engine_id,
        contract.producer_request_id,
        contract.registration_generation,
        contract.offer_generation,
        contract.iteration,
        contract.source_rank,
    )


def _capture_lineage(capture: NixlCaptureRecord) -> SourceLineage:
    """Return the complete source lineage encoded by a decoder capture.

    :param capture: Decoder capture record.
    :returns: Run, arm, engine, request, registration, offer, iteration, and rank.
    """
    return (
        capture.run_id,
        capture.transport_arm,
        capture.producer_engine_id,
        capture.producer_request_id,
        capture.registration_generation,
        capture.offer_generation,
        capture.iteration,
        capture.source_rank,
    )


def _derive_p2d_attention_assignment(
    source_world_size: int,
    decoder_world_size: int,
    decoder_rank: int,
    total_num_kv_heads: int,
) -> tuple[tuple[int, int], ...]:
    """Derive the canonical P-rank and staging-slot assignment.

    This diagnostic observes the non-MLA, non-Mamba coalesced path where the
    prefill tensor-parallel world is strictly larger than the decoder world.
    Replicated GQA ranks are deduplicated by retaining the first P rank for each
    physical KV head, matching the transfer topology without trusting the plan.

    :param source_world_size: Producer tensor-parallel process count.
    :param decoder_world_size: Decoder tensor-parallel process count.
    :param decoder_rank: Decoder rank whose assignment is requested.
    :param total_num_kv_heads: Model-wide KV-head count.
    :returns: Canonical ``(source_rank, rank_slot)`` pairs.
    :raises LocalizationError: If the topology is outside the diagnostic scope.
    """
    if source_world_size <= decoder_world_size:
        raise LocalizationError(
            "localization requires producer TP greater than decoder TP"
        )
    if source_world_size % decoder_world_size != 0:
        raise LocalizationError("producer TP is not divisible by decoder TP")
    if decoder_rank < 0 or decoder_rank >= decoder_world_size:
        raise LocalizationError("decoder rank is outside its tensor-parallel world")
    if total_num_kv_heads <= 0:
        raise LocalizationError("localization requires a positive KV-head count")

    source_ranks_per_decoder = source_world_size // decoder_world_size
    source_start = decoder_rank * source_ranks_per_decoder
    seen_heads: set[int] = set()
    assignment: list[tuple[int, int]] = []
    for source_rank in range(
        source_start,
        source_start + source_ranks_per_decoder,
    ):
        kv_head = source_rank * total_num_kv_heads // source_world_size
        if kv_head in seen_heads:
            continue
        seen_heads.add(kv_head)
        assignment.append((source_rank, len(assignment)))
    if len(assignment) == 0:
        raise LocalizationError("topology derives no producer ranks")
    return tuple(assignment)


def _region_semantic_signature(
    region: NixlRegionDescriptor,
) -> tuple[object, ...]:
    """Return rank-independent registered-region semantics.

    :param region: NIXL region descriptor.
    :returns: Semantic and geometric fields excluding the rank-local address.
    """
    return (
        region.semantic_name,
        region.group_indices,
        region.group_semantic_names,
        region.registered_bytes,
        region.row_bytes,
        region.shape,
        region.strides,
        region.dtype,
        region.element_size_bytes,
        region.layout,
    )


def _source_request_signature(contract: NixlSourceContract) -> tuple[object, ...]:
    """Return the request contract shared by every participating P rank.

    :param contract: One rank-local producer contract.
    :returns: Request lineage, semantic geometry, and physical block roster.
    """
    return (
        contract.schema_version,
        contract.fingerprint_algorithm,
        contract.run_id,
        contract.transport_arm,
        contract.producer_engine_id,
        contract.producer_request_id,
        contract.offer_generation,
        contract.iteration,
        contract.expected_consumers,
        contract.region_lengths,
        tuple(_region_semantic_signature(region) for region in contract.regions),
        contract.source_group_planes,
        contract.valid_token_extent,
        contract.group_token_capacities,
        contract.block_ids,
    )


def _decoder_request_signature(plan: NixlPlanRecord) -> tuple[object, ...]:
    """Return rank-independent decoder placement semantics for one request.

    :param plan: Decoder-rank transfer plan.
    :returns: Semantic destination geometry and source-position selection.
    """
    return (
        plan.source_tp_size,
        plan.destination_group_planes,
        plan.destination_group_token_capacities,
        tuple(_region_semantic_signature(region) for region in plan.local_regions),
        tuple(len(group) for group in plan.untrimmed_local_groups),
        plan.skipped_groups,
        plan.selected_remote_groups,
        tuple(len(group) for group in plan.selected_local_groups),
        tuple(
            (
                region_plan.region_index,
                region_plan.offset_within_rank,
                tuple(
                    (
                        position.group_index,
                        position.source_position,
                        position.remote_block_id,
                        position.valid_token_extent,
                        position.group_token_capacity,
                        position.destination_half,
                    )
                    for position in region_plan.positions
                ),
                region_plan.runs,
            )
            for region_plan in plan.region_plans
        ),
        plan.rank_stride_bytes,
    )


def _plan_topology_errors(
    plan: NixlPlanRecord,
    engine_world_sizes: dict[str, int],
    engine_total_num_kv_heads: dict[str, int],
) -> tuple[str, ...]:
    """Validate a plan from independently recorded P and D session topology.

    :param plan: Decoder transfer plan.
    :param engine_world_sizes: Session-derived tensor-parallel world sizes.
    :param engine_total_num_kv_heads: Session-derived model KV-head counts.
    :returns: Producer/decoder topology and assignment errors.
    """
    if len(plan.source_contracts) == 0:
        return ()
    if len(plan.rank_slots) != len(plan.source_contracts):
        return ()
    producer_engines = {
        contract.producer_engine_id for contract in plan.source_contracts
    }
    if len(producer_engines) != 1:
        return ("plan source contracts name multiple producer engines",)
    producer_engine = next(iter(producer_engines))
    if producer_engine == plan.observer_engine_id:
        return ("plan producer and decoder engines are identical",)
    source_world_size = engine_world_sizes.get(producer_engine)
    decoder_world_size = engine_world_sizes.get(plan.observer_engine_id)
    source_kv_heads = engine_total_num_kv_heads.get(producer_engine)
    decoder_kv_heads = engine_total_num_kv_heads.get(plan.observer_engine_id)
    if source_world_size is None or decoder_world_size is None:
        return ("plan topology has no complete producer/decoder sessions",)
    if plan.source_tp_size != source_world_size:
        return ("plan source TP size differs from producer session world size",)
    if source_kv_heads is None or decoder_kv_heads is None:
        return ("plan topology has no producer/decoder KV-head contract",)
    if source_kv_heads != decoder_kv_heads:
        return ("producer and decoder sessions disagree on total KV heads",)
    try:
        expected = _derive_p2d_attention_assignment(
            source_world_size,
            decoder_world_size,
            plan.observer_rank,
            source_kv_heads,
        )
    except LocalizationError as error:
        return (str(error),)
    actual = tuple(
        (contract.source_rank, rank_slot)
        for contract, rank_slot in zip(
            plan.source_contracts,
            plan.rank_slots,
            strict=True,
        )
    )
    if actual != expected:
        return (
            "plan source-rank partition or rank slots differ from "
            f"session-derived topology: {actual} != {expected}",
        )
    return ()


def _cross_decoder_rank_plan_errors(
    child_request_id: str,
    observer_engine_id: str,
    rank_plans: dict[int, NixlPlanRecord],
    engine_world_sizes: dict[str, int],
    engine_total_num_kv_heads: dict[str, int],
) -> tuple[str, ...]:
    """Validate request consistency and the complete D-rank partition.

    :param child_request_id: Decoder child request shared by the plans.
    :param observer_engine_id: Decoder engine shared by the plans.
    :param rank_plans: Decoder-rank keyed plans.
    :param engine_world_sizes: Session-derived tensor-parallel world sizes.
    :param engine_total_num_kv_heads: Session-derived model KV-head counts.
    :returns: Cross-rank request-contract and topology-partition errors.
    """
    errors: list[str] = []
    decoder_world_size = engine_world_sizes.get(observer_engine_id)
    if decoder_world_size is None:
        return ("decoder plan set has no session-derived world size",)
    expected_decoder_ranks = set(range(decoder_world_size))
    if set(rank_plans) != expected_decoder_ranks:
        return (
            f"child {(child_request_id, observer_engine_id)} plan ranks are "
            "incomplete for topology validation",
        )
    if any(
        len(plan.source_contracts) == 0
        or len(plan.rank_slots) != len(plan.source_contracts)
        for plan in rank_plans.values()
    ):
        return ()

    request_signatures = {
        _source_request_signature(contract)
        for plan in rank_plans.values()
        for contract in plan.source_contracts
    }
    if len(request_signatures) != 1:
        errors.append(
            f"child {(child_request_id, observer_engine_id)} decoder ranks "
            "disagree on the producer request contract"
        )
    decoder_signatures = {
        _decoder_request_signature(plan) for plan in rank_plans.values()
    }
    if len(decoder_signatures) != 1:
        errors.append(
            f"child {(child_request_id, observer_engine_id)} decoder ranks "
            "disagree on destination request geometry"
        )

    contracts_by_source_rank: dict[int, NixlSourceContract] = {}
    for plan in rank_plans.values():
        for contract in plan.source_contracts:
            existing = contracts_by_source_rank.get(contract.source_rank)
            if existing is not None and existing != contract:
                errors.append(
                    f"child {(child_request_id, observer_engine_id)} decoder "
                    f"ranks disagree on source-rank contract {contract.source_rank}"
                )
            else:
                contracts_by_source_rank[contract.source_rank] = contract

    producer_engines = {
        contract.producer_engine_id
        for plan in rank_plans.values()
        for contract in plan.source_contracts
    }
    if len(producer_engines) != 1:
        return tuple(errors)
    producer_engine = next(iter(producer_engines))
    source_world_size = engine_world_sizes.get(producer_engine)
    source_kv_heads = engine_total_num_kv_heads.get(producer_engine)
    decoder_kv_heads = engine_total_num_kv_heads.get(observer_engine_id)
    if (
        source_world_size is None
        or source_kv_heads is None
        or decoder_kv_heads is None
        or source_kv_heads != decoder_kv_heads
    ):
        return tuple(errors)
    try:
        expected_partition = tuple(
            (
                decoder_rank,
                _derive_p2d_attention_assignment(
                    source_world_size,
                    decoder_world_size,
                    decoder_rank,
                    source_kv_heads,
                ),
            )
            for decoder_rank in range(decoder_world_size)
        )
    except LocalizationError as error:
        errors.append(str(error))
        return tuple(errors)
    actual_partition = tuple(
        (
            decoder_rank,
            tuple(
                (contract.source_rank, rank_slot)
                for contract, rank_slot in zip(
                    rank_plans[decoder_rank].source_contracts,
                    rank_plans[decoder_rank].rank_slots,
                    strict=True,
                )
            ),
        )
        for decoder_rank in range(decoder_world_size)
    )
    if actual_partition != expected_partition:
        errors.append(
            f"child {(child_request_id, observer_engine_id)} source-rank "
            "partition differs from session-derived P-to-D topology"
        )
    return tuple(errors)


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
        if record.stage is not IntegrityStage.SOURCE_POST:
            errors.append("localization source record is not SOURCE_POST")
        if manifest.run_id != session.run_id:
            errors.append("source manifest run differs from its session")
        if manifest.fingerprint_algorithm is not session.fingerprint_algorithm:
            errors.append("source manifest algorithm differs from its session")
        if manifest.transport_arm != session.transport_arm:
            errors.append("source manifest arm differs from its session")
        if manifest.producer_engine_id != session.engine_id:
            errors.append("source manifest engine differs from its session")
        if manifest.source_rank != session.rank:
            errors.append("source manifest rank differs from its session")
        if (
            localization_producer_target(
                manifest.producer_request_id,
                session.target_request_ids,
            )
            is None
        ):
            errors.append("source manifest request differs from its session target")
        return tuple(errors)

    if isinstance(record, NixlPlanRecord):
        if record.observer_engine_id != session.engine_id:
            errors.append("plan engine differs from its session")
        if record.observer_rank != session.rank:
            errors.append("plan rank differs from its session")
        child_target = localization_request_target(
            record.child_request_id,
            session.target_request_ids,
        )
        if child_target is None:
            errors.append("plan child differs from its session target")
        for contract in record.source_contracts:
            if contract.fingerprint_algorithm is not session.fingerprint_algorithm:
                errors.append("plan source algorithm differs from its session")
            if contract.run_id != session.run_id:
                errors.append("plan source contract run differs from its session")
            if contract.transport_arm != session.transport_arm:
                errors.append("plan source contract arm differs from its session")
            producer_target = localization_producer_target(
                contract.producer_request_id,
                session.target_request_ids,
            )
            if producer_target is None or producer_target != child_target:
                errors.append("plan source differs from its session target")
        return tuple(errors)

    if record.schema_version != IntegrityIdentity.SCHEMA_VERSION:
        errors.append(f"{record.record_type} schema mismatch")
    if (
        isinstance(record, NixlCaptureRecord)
        and record.fingerprint_algorithm is not session.fingerprint_algorithm
    ):
        errors.append("capture algorithm differs from its session")
    if isinstance(record, NixlCaptureRecord) and record.stage not in _capture_stages(
        session.mode
    ):
        errors.append("capture stage differs from its session mode")
    if record.run_id != session.run_id:
        errors.append(f"{record.record_type} run differs from its session")
    if record.transport_arm != session.transport_arm:
        errors.append(f"{record.record_type} arm differs from its session")
    if record.observer_engine_id != session.engine_id:
        errors.append(f"{record.record_type} engine differs from its session")
    if record.observer_rank != session.rank:
        errors.append(f"{record.record_type} rank differs from its session")
    child_target = (
        None
        if record.child_request_id is None
        else localization_request_target(
            record.child_request_id,
            session.target_request_ids,
        )
    )
    if child_target is None:
        errors.append(f"{record.record_type} child differs from its session target")
    if (
        record.producer_request_id is not None
        and localization_producer_target(
            record.producer_request_id,
            session.target_request_ids,
        )
        != child_target
    ):
        errors.append(f"{record.record_type} source differs from its session target")
    return tuple(errors)


def _target_coverage_errors(
    target_request_ids: tuple[str, ...],
    source_manifests: tuple[NixlSourceManifest, ...],
    events: dict[ObserverKey, NixlEventRecord],
) -> tuple[str, ...]:
    """Prove every configured parent has its complete logical child set.

    :param target_request_ids: Complete configured parent allowlist.
    :param source_manifests: Every authoritative SOURCE_POST manifest.
    :param events: Decoder-rank terminal outcomes keyed by observer identity.
    :returns: Missing-parent, cardinality, child-form, and duplication errors.
    """
    errors: list[str] = []
    consumer_counts: dict[str, set[int]] = {
        target_request_id: set() for target_request_id in target_request_ids
    }
    for manifest in source_manifests:
        target_request_id = localization_producer_target(
            manifest.producer_request_id,
            target_request_ids,
        )
        if target_request_id is not None:
            consumer_counts[target_request_id].add(manifest.expected_consumers)

    logical_children: dict[str, set[tuple[str, str]]] = {
        target_request_id: set() for target_request_id in target_request_ids
    }
    for child_request_id, observer_engine_id, _ in events:
        target_request_id = localization_request_target(
            child_request_id,
            target_request_ids,
        )
        if target_request_id is not None:
            logical_children[target_request_id].add(
                (child_request_id, observer_engine_id)
            )

    for target_request_id in target_request_ids:
        counts = consumer_counts[target_request_id]
        if len(counts) == 0:
            errors.append(
                f"configured target {target_request_id!r} has no SOURCE_POST manifest"
            )
            continue
        if len(counts) != 1:
            errors.append(
                f"configured target {target_request_id!r} has inconsistent "
                f"expected-consumer counts: {sorted(counts)}"
            )
            continue
        expected_consumers = next(iter(counts))
        children_by_index: dict[int, set[tuple[str, str]]] = {}
        for child_request_id, observer_engine_id in logical_children[target_request_id]:
            stable_child_id = localization_request_id_base(child_request_id)
            child_index = localization_child_index(
                child_request_id,
                target_request_id,
            )
            if child_index is None:
                errors.append(
                    f"target {target_request_id!r} has an invalid child identity "
                    f"{child_request_id!r}"
                )
                continue
            direct_parent = stable_child_id == target_request_id
            if expected_consumers == 1 and not direct_parent:
                errors.append(
                    f"single-consumer target {target_request_id!r} uses prefixed "
                    f"child {child_request_id!r}"
                )
                continue
            if expected_consumers > 1 and direct_parent:
                errors.append(
                    f"parallel target {target_request_id!r} uses bare parent "
                    f"{child_request_id!r} as a child"
                )
                continue
            if child_index >= expected_consumers:
                errors.append(
                    f"target {target_request_id!r} child index {child_index} exceeds "
                    f"its {expected_consumers}-consumer contract"
                )
                continue
            children_by_index.setdefault(child_index, set()).add(
                (child_request_id, observer_engine_id)
            )

        expected_indices = set(range(expected_consumers))
        actual_indices = set(children_by_index)
        if actual_indices != expected_indices:
            errors.append(
                f"configured target {target_request_id!r} child indices are "
                f"incomplete: {sorted(actual_indices)} != {sorted(expected_indices)}"
            )
        for child_index, children in children_by_index.items():
            if len(children) != 1:
                errors.append(
                    f"configured target {target_request_id!r} child index "
                    f"{child_index} maps to multiple logical decoder children: "
                    f"{sorted(children)}"
                )
    return tuple(errors)


def _artifact_chronology_errors(
    artifact: LocalizationArtifact,
) -> tuple[str, ...]:
    """Validate plan, capture, and terminal-event order within one process.

    :param artifact: Complete process artifact whose frame order is preserved.
    :returns: Child-local chronology errors.
    """
    child_records: dict[ObserverKey, list[tuple[int, ArtifactRecord]]] = {}
    for index, record in enumerate(artifact.records):
        observer_key: ObserverKey | None = None
        if isinstance(record, NixlPlanRecord) or (
            isinstance(record, NixlCaptureRecord | NixlEventRecord)
            and record.child_request_id is not None
        ):
            assert record.child_request_id is not None
            observer_key = (
                record.child_request_id,
                record.observer_engine_id,
                record.observer_rank,
            )
        if observer_key is not None:
            child_records.setdefault(observer_key, []).append((index, record))

    errors: list[str] = []
    ordered_stages = _capture_stages(artifact.session.mode)
    for observer_key, records in child_records.items():
        event_records = [
            (index, record)
            for index, record in records
            if isinstance(record, NixlEventRecord)
        ]
        if len(event_records) > 1:
            errors.append(f"child {observer_key} has multiple terminal events")
            continue
        if len(event_records) == 0:
            continue
        event_index, event = event_records[0]
        if any(
            index > event_index and not isinstance(record, NixlEventRecord)
            for index, record in records
        ):
            errors.append(f"child {observer_key} has records after its terminal event")
        if event.code != "CAPTURE_COMPLETE":
            continue

        plan_indices = [
            index for index, record in records if isinstance(record, NixlPlanRecord)
        ]
        stage_indices = {
            stage: [
                index
                for index, record in records
                if isinstance(record, NixlCaptureRecord) and record.stage is stage
            ]
            for stage in ordered_stages
        }
        if len(plan_indices) != 1 or any(
            len(indices) == 0 for indices in stage_indices.values()
        ):
            continue
        ordered_ranges = [plan_indices]
        ordered_ranges.extend(stage_indices[stage] for stage in ordered_stages)
        ordered_ranges.append([event_index])
        if any(
            max(before) >= min(after)
            for before, after in zip(
                ordered_ranges,
                ordered_ranges[1:],
            )
        ):
            errors.append(f"child {observer_key} capture stages are out of order")
    return tuple(errors)


def _coalesced_plan_inputs(
    plan: NixlPlanRecord,
) -> tuple[
    tuple[GroupTransferRoster, ...],
    tuple[RegionOwnership, ...],
    tuple[str, ...],
]:
    """Reconstruct canonical builder inputs from independent artifact facts.

    :param plan: Decoder transfer plan.
    :returns: Group rosters, region ownership, and structural errors.
    """
    errors: list[str] = []
    if len(plan.source_contracts) == 0:
        return (), (), ("plan has no source contracts",)
    reference = plan.source_contracts[0]
    group_count = len(reference.block_ids)
    if (
        len(plan.selected_remote_groups) != group_count
        or len(plan.selected_local_groups) != group_count
        or len(plan.untrimmed_local_groups) != group_count
        or len(plan.destination_group_planes) != group_count
        or len(plan.destination_group_token_capacities) != group_count
    ):
        return (), (), ("plan group cardinality mismatch",)

    skipped_groups = set(plan.skipped_groups)
    groups: list[GroupTransferRoster] = []
    for group_index in range(group_count):
        raw = reference.block_ids[group_index]
        untrimmed_local = plan.untrimmed_local_groups[group_index]
        remote = plan.selected_remote_groups[group_index]
        local = plan.selected_local_groups[group_index]
        planes = plan.destination_group_planes[group_index]
        if group_index in skipped_groups:
            if len(remote) > 0 or len(local) > 0:
                errors.append(
                    f"plan group {group_index} is skipped but has selected blocks"
                )
        else:
            if local != untrimmed_local:
                errors.append(
                    f"plan group {group_index} changed its decoder allocation"
                )
            factor = 2 if planes == 1 else 1
            expected_remote_count = min(len(raw), factor * len(untrimmed_local))
            expected_remote = (
                raw[-expected_remote_count:] if expected_remote_count > 0 else ()
            )
            if remote != expected_remote:
                errors.append(
                    f"plan group {group_index} remote selection is not the "
                    "required suffix"
                )
        groups.append(
            GroupTransferRoster(
                group_index=group_index,
                source_position_start=len(raw) - len(remote),
                destination_plane_count=planes,
                local_block_ids=local,
                remote_block_ids=remote,
            )
        )

    if len(plan.local_regions) != len(reference.regions):
        errors.append("plan local region cardinality mismatch")
        return tuple(groups), (), tuple(errors)
    regions: list[RegionOwnership] = []
    for region_index, (source_region, local_region) in enumerate(
        zip(reference.regions, plan.local_regions, strict=True)
    ):
        if (
            source_region.semantic_name != local_region.semantic_name
            or source_region.group_indices != local_region.group_indices
            or source_region.group_semantic_names != local_region.group_semantic_names
            or source_region.dtype != local_region.dtype
            or source_region.element_size_bytes != local_region.element_size_bytes
            or source_region.layout != local_region.layout
        ):
            errors.append("plan local/source semantic region mismatch")
        row_counts: list[int] = []
        for role, region in (
            ("source", source_region),
            ("local", local_region),
        ):
            if (
                len(region.shape) == 0
                or len(region.shape) != len(region.strides)
                or region.row_bytes <= 0
                or region.registered_bytes <= 0
                or region.registered_bytes != region.shape[0] * region.row_bytes
                or region.strides[0] * region.element_size_bytes != region.row_bytes
            ):
                errors.append(f"plan {role} region is not row canonical")
                row_counts.append(0)
            else:
                row_counts.append(region.registered_bytes // region.row_bytes)
        if 0 in row_counts:
            continue
        expected_local_row_bytes = len(plan.source_contracts) * source_region.row_bytes
        if local_region.row_bytes != expected_local_row_bytes:
            errors.append(
                f"plan local region {region_index} row length differs from the "
                "participating source-rank geometry"
            )
        regions.append(
            RegionOwnership(
                region_index=region_index,
                group_indices=source_region.group_indices,
                source_row_count=row_counts[0],
                destination_row_count=row_counts[1],
                row_bytes=source_region.row_bytes,
            )
        )
    return tuple(groups), tuple(regions), tuple(errors)


def _artifact_region_plans(
    plan: NixlPlanRecord,
    rebuilt: CoalescedTransferPlan,
) -> tuple[NixlRegionPlan, ...]:
    """Convert a rebuilt canonical plan into its artifact representation.

    :param plan: Decoder transfer plan carrying token semantics.
    :param rebuilt: Independently rebuilt transport layout.
    :returns: Exact typed region plans expected in the artifact.
    """
    reference = plan.source_contracts[0]
    return tuple(
        NixlRegionPlan(
            region_index=region.ownership.region_index,
            offset_within_rank=region.offset_within_rank,
            positions=tuple(
                NixlPlanPosition(
                    group_index=position.group_index,
                    source_position=position.source_position,
                    remote_block_id=position.remote_block_id,
                    valid_token_extent=reference.valid_token_extent,
                    group_token_capacity=(
                        reference.group_token_capacities[position.group_index]
                    ),
                    local_block_id=position.local_block_id,
                    destination_half=position.destination_half,
                )
                for position in region.positions
            ),
            runs=tuple(
                NixlPlanRun(
                    remote_block_id=run.remote_block_id,
                    position_count=run.position_count,
                    position_start=run.position_start,
                )
                for run in region.runs
            ),
        )
        for region in rebuilt.regions
    )


def _plan_errors(
    plan: NixlPlanRecord,
) -> tuple[str, ...]:
    """Validate one exact transfer plan and its content-free source contracts.

    :param plan: Decoder plan artifact.
    :returns: Plan, rank, semantic, geometry, and placement errors.
    """
    errors: list[str] = []
    if plan.schema_version != IntegrityIdentity.SCHEMA_VERSION:
        errors.append("plan schema mismatch")
    rank_count = len(plan.source_contracts)
    if rank_count == 0:
        errors.append("plan has no source contracts")
        return tuple(errors)
    if len(plan.rank_slots) != rank_count:
        errors.append("plan source-rank cardinality mismatch")
        return tuple(errors)
    source_contracts_valid = True
    for index, contract in enumerate(plan.source_contracts):
        contract_errors = validate_source_contract_structure(contract)
        if len(contract_errors) > 0:
            source_contracts_valid = False
        for contract_error in contract_errors:
            errors.append(f"plan source contract {index}: {contract_error}")
    if source_contracts_valid is False:
        return tuple(errors)

    source_ranks = tuple(contract.source_rank for contract in plan.source_contracts)
    if len(set(source_ranks)) != rank_count:
        errors.append("plan contains duplicate source ranks")
    if sorted(plan.rank_slots) != list(range(rank_count)):
        errors.append("plan rank slots are not a complete bijection")
    if plan.skipped_groups != tuple(sorted(set(plan.skipped_groups))):
        errors.append("plan skipped groups are not canonical")
    if len(plan.skipped_groups) > 0:
        errors.append("plan skips cache groups and is non-evidentiary")
    if any(
        group_index < 0 or group_index >= len(plan.source_contracts[0].block_ids)
        for group_index in plan.skipped_groups
    ):
        errors.append("plan contains an invalid skipped group")
    for group_index, blocks in enumerate(plan.untrimmed_local_groups):
        if any(type(block_id) is not int or block_id < 0 for block_id in blocks):
            errors.append(f"plan local group {group_index} has an invalid block id")
        if len(set(blocks)) != len(blocks):
            errors.append(f"plan local group {group_index} has duplicate block ids")
    reference = plan.source_contracts[0]
    if len(reference.regions) == 0:
        errors.append("plan source contract has no registered regions")
    if not any(len(group) > 0 for group in reference.block_ids):
        errors.append("plan source contract has no physical source blocks")
    reference_region_semantics = tuple(
        (
            region.semantic_name,
            region.group_indices,
            region.group_semantic_names,
            region.registered_bytes,
            region.row_bytes,
            region.shape,
            region.strides,
            region.dtype,
            region.element_size_bytes,
            region.layout,
        )
        for region in reference.regions
    )
    for source_contract in plan.source_contracts[1:]:
        if (
            source_contract.run_id != reference.run_id
            or source_contract.fingerprint_algorithm
            is not reference.fingerprint_algorithm
            or source_contract.transport_arm != reference.transport_arm
            or source_contract.producer_engine_id != reference.producer_engine_id
            or source_contract.producer_request_id != reference.producer_request_id
            or source_contract.offer_generation != reference.offer_generation
            or source_contract.iteration != reference.iteration
            or source_contract.expected_consumers != reference.expected_consumers
            or source_contract.region_lengths != reference.region_lengths
            or source_contract.source_group_planes != reference.source_group_planes
            or source_contract.valid_token_extent != reference.valid_token_extent
            or source_contract.group_token_capacities
            != reference.group_token_capacities
            or source_contract.block_ids != reference.block_ids
        ):
            errors.append("plan source ranks disagree on transfer contract")
        source_region_semantics = tuple(
            (
                region.semantic_name,
                region.group_indices,
                region.group_semantic_names,
                region.registered_bytes,
                region.row_bytes,
                region.shape,
                region.strides,
                region.dtype,
                region.element_size_bytes,
                region.layout,
            )
            for region in source_contract.regions
        )
        if source_region_semantics != reference_region_semantics:
            errors.append("plan source ranks disagree on semantic regions")
    if plan.destination_group_planes != reference.source_group_planes:
        errors.append("plan destination/source plane contracts differ")
    if plan.destination_group_token_capacities != reference.group_token_capacities:
        errors.append("plan destination/source token capacities differ")
    if len(plan.destination_group_planes) != len(reference.group_token_capacities):
        errors.append("plan destination plane-contract cardinality mismatch")
    if plan.source_tp_size <= 0:
        errors.append("plan source TP size is not positive")
    if any(
        source_rank < 0 or source_rank >= plan.source_tp_size
        for source_rank in source_ranks
    ):
        errors.append("plan source rank is outside source_tp_size")
    if plan.staging_offset < 0:
        errors.append("plan staging offset is negative")

    groups, regions, input_errors = _coalesced_plan_inputs(plan)
    errors.extend(input_errors)
    if len(groups) > 0 and len(regions) == len(reference.regions):
        try:
            rebuilt = build_coalesced_transfer_plan(
                source_tp_size=plan.source_tp_size,
                source_ranks=source_ranks,
                rank_slots=plan.rank_slots,
                groups=groups,
                regions=regions,
            )
        except ValueError as error:
            errors.append(f"plan cannot reconstruct canonical layout: {error}")
        else:
            expected_region_plans = _artifact_region_plans(plan, rebuilt)
            if len(plan.region_plans) != len(expected_region_plans):
                errors.append("plan region-plan cardinality differs from ownership")
            else:
                for expected, actual in zip(
                    expected_region_plans,
                    plan.region_plans,
                    strict=True,
                ):
                    if actual.region_index != expected.region_index:
                        errors.append("plan region indices are not canonical")
                    if actual.offset_within_rank != expected.offset_within_rank:
                        errors.append(
                            f"plan region {expected.region_index} packed offset "
                            "differs from canonical rank slab"
                        )
                    if actual.positions != expected.positions:
                        errors.append(
                            f"plan region {expected.region_index} positions differ "
                            "from ownership-filtered canonical layout"
                        )
                    if actual.runs != expected.runs:
                        errors.append(
                            f"plan region {expected.region_index} runs differ from "
                            "its canonical positions"
                        )
            if plan.rank_stride_bytes != rebuilt.rank_stride_bytes:
                errors.append("plan rank stride differs from packed region geometry")
            if plan.staging_size != rebuilt.staging_size_bytes:
                errors.append(
                    "plan staging size differs from canonical rank-major size"
                )
            if plan.layout_digest != rebuilt.digest:
                errors.append("plan layout digest differs from reconstructed layout")
            if sum(len(region.positions) for region in rebuilt.regions) == 0:
                errors.append("physical transfer plan has no transfer positions")
    return tuple(errors)


def validate_localization_plan(
    plan: NixlPlanRecord,
) -> tuple[str, ...]:
    """Validate one plan against its embedded source contracts.

    :param plan: Decoder plan artifact.
    :returns: Plan and semantic mapping errors.
    """
    return _plan_errors(plan)


def _capture_contract(
    plan: NixlPlanRecord,
    source_contract: NixlSourceContract,
    stage: IntegrityStage,
) -> tuple[
    set[IntegrityLeafKey],
    dict[IntegrityLeafKey, tuple[int, int, int]],
    int,
    int,
]:
    """Reconstruct exact semantic leaves and byte counts for one stage.

    :param plan: Canonical decoder transfer plan.
    :param source_contract: Decoder's content-free source contract.
    :param stage: Decoder observation stage.
    :returns: Leaf keys, placement mapping, copied bytes, and hashed bytes.
    """
    source_ranks = tuple(contract.source_rank for contract in plan.source_contracts)
    rank_index = source_ranks.index(source_contract.source_rank)
    rank_slot = plan.rank_slots[rank_index]
    keys: set[IntegrityLeafKey] = set()
    mapping: dict[IntegrityLeafKey, tuple[int, int, int]] = {}
    copied_bytes = 0
    hashed_bytes = 0
    for region_index, region in enumerate(source_contract.regions):
        region_plan = plan.region_plans[region_index]
        for position in region_plan.positions:
            group_index = position.group_index
            if stage in (
                IntegrityStage.STAGING_RAW,
                IntegrityStage.STAGING_FENCED_CONTROL,
                IntegrityStage.STAGING_POST_SCATTER,
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
                    source_plane_contract=source_contract.source_group_planes[
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
                keys.add(key)
                mapping[key] = (
                    position.local_block_id,
                    rank_slot,
                    position.destination_half,
                )
                hashed_bytes += (
                    region.row_bytes
                    if payload_kind is IntegrityPayloadKind.WIRE
                    else region.row_bytes // 2
                )
    if (
        source_contract.fingerprint_algorithm
        is LocalizationFingerprintAlgorithm.POSITION_WEIGHTED_WORDS_256_V1
    ):
        copied_bytes = len(keys) * localization_fingerprint_size(
            source_contract.fingerprint_algorithm
        )
    return keys, mapping, copied_bytes, hashed_bytes


def _capture_errors(
    plan: NixlPlanRecord,
    source_contract: NixlSourceContract,
    manifest: NixlSourceManifest,
    capture: NixlCaptureRecord,
    *,
    compare_digests: bool,
) -> tuple[str, ...]:
    """Validate one decoder observation against source and plan.

    :param plan: Canonical decoder transfer plan.
    :param source_contract: Decoder's content-free source contract.
    :param manifest: Authoritative producer SOURCE_POST manifest.
    :param capture: Stage capture.
    :param compare_digests: Whether this is an evidentiary trace arm.
    :returns: Exact lineage, byte, cardinality, content, and mapping errors.
    """
    errors: list[str] = []
    if (
        _capture_lineage(capture) != _contract_lineage(source_contract)
        or _source_lineage(manifest) != _contract_lineage(source_contract)
        or capture.child_request_id != plan.child_request_id
        or capture.source_rank != manifest.source_rank
        or capture.observer_engine_id != plan.observer_engine_id
        or capture.observer_rank != plan.observer_rank
        or capture.fingerprint_algorithm is not manifest.fingerprint_algorithm
        or source_contract.fingerprint_algorithm is not manifest.fingerprint_algorithm
    ):
        errors.append("capture lineage differs from plan or SOURCE_POST")
    keys, mapping, copied_bytes, hashed_bytes = _capture_contract(
        plan,
        source_contract,
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
    fingerprint_size = localization_fingerprint_size(capture.fingerprint_algorithm)
    if any(len(leaf.digest) != fingerprint_size for leaf in capture.leaves):
        errors.append("capture fingerprint length differs from its algorithm")
    if compare_digests is False and any(
        leaf.digest != b"\x00" * fingerprint_size for leaf in capture.leaves
    ):
        errors.append("sham capture contains a non-redacted digest")
    if capture.copied_bytes != copied_bytes:
        errors.append("capture copied-byte count differs from semantic contract")
    if capture.hashed_bytes != hashed_bytes:
        errors.append("capture hashed-byte count differs from semantic contract")
    if capture.observer is False:
        errors.append("capture is not labeled as an observer")
    if capture.barrier != localization_stage_barrier(capture.stage):
        errors.append("capture barrier label differs from stage contract")
    return tuple(errors)


def _first_divergence(
    plan: NixlPlanRecord,
    source_contract: NixlSourceContract,
    manifest: NixlSourceManifest,
    captures: dict[CaptureKey, NixlCaptureRecord],
    *,
    compare_digests: bool,
) -> LocalizationDivergence | None:
    """Find the earliest failing decoder edge for one source rank.

    :param plan: Decoder plan.
    :param source_contract: Decoder's content-free source contract.
    :param manifest: Authoritative producer SOURCE_POST manifest.
    :param captures: Complete capture index.
    :param compare_digests: Whether the arm carries content evidence.
    :returns: Earliest divergence, or ``None`` when all present stages match.
    """
    ordered_edges: tuple[tuple[IntegrityStage, str], ...]
    if (
        source_contract.fingerprint_algorithm
        is LocalizationFingerprintAlgorithm.POSITION_WEIGHTED_WORDS_256_V1
    ):
        ordered_edges = (
            (
                IntegrityStage.STAGING_POST_SCATTER,
                "source_post_reference_vs_staging_post_scatter",
            ),
            (IntegrityStage.DESTINATION, "staging_post_scatter->destination"),
            (IntegrityStage.PRE_READ, "destination->pre_read"),
        )
    else:
        ordered_edges = (
            (
                IntegrityStage.STAGING_RAW,
                "source_post_reference_vs_staging_raw",
            ),
            (
                IntegrityStage.STAGING_FENCED_CONTROL,
                "staging_raw->staging_fenced_control",
            ),
            (IntegrityStage.DESTINATION, "staging_fenced_control->destination"),
            (IntegrityStage.PRE_READ, "destination->pre_read"),
        )
    observer_key: ObserverKey = (
        plan.child_request_id,
        plan.observer_engine_id,
        plan.observer_rank,
    )
    lineage = _contract_lineage(source_contract)
    if any(
        (observer_key, lineage, stage) not in captures for stage, _ in ordered_edges
    ):
        return None
    for stage, edge in ordered_edges:
        key: CaptureKey = (observer_key, lineage, stage)
        capture = captures.get(key)
        if capture is None:
            continue
        try:
            errors = _capture_errors(
                plan,
                source_contract,
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
    source_post_by_lineage: dict[SourceLineage, NixlSourceManifest] = {}
    plans: dict[ObserverKey, NixlPlanRecord] = {}
    captures: dict[CaptureKey, NixlCaptureRecord] = {}
    events: dict[ObserverKey, NixlEventRecord] = {}
    engine_world_sizes: dict[str, int] = {}
    engine_total_num_kv_heads: dict[str, int] = {}

    for artifact in artifacts:
        session = artifact.session
        session_key = (session.engine_id, session.rank)
        if session_key in sessions:
            errors.append(f"duplicate process artifact for {session_key}")
        else:
            sessions[session_key] = artifact
        if (
            session.schema_version != IntegrityIdentity.SCHEMA_VERSION
            or session.fingerprint_algorithm
            is not reference_session.fingerprint_algorithm
            or session.run_id != reference_session.run_id
            or session.transport_arm != reference_session.transport_arm
            or session.mode is not reference_session.mode
            or session.mode is LocalizationMode.OFF
            or (
                session.mode is LocalizationMode.FINGERPRINT
                and session.fingerprint_algorithm
                is not LocalizationFingerprintAlgorithm.POSITION_WEIGHTED_WORDS_256_V1
            )
            or (
                session.mode is not LocalizationMode.FINGERPRINT
                and session.fingerprint_algorithm
                is not LocalizationFingerprintAlgorithm.BLAKE2B_128
            )
            or session.target_request_ids != reference_session.target_request_ids
            or len(session.target_request_ids) == 0
            or tuple(sorted(session.target_request_ids)) != session.target_request_ids
            or len(set(session.target_request_ids)) != len(session.target_request_ids)
            or any(
                len(target_request_id) == 0
                or localization_request_id_base(target_request_id) != target_request_id
                for target_request_id in session.target_request_ids
            )
            or any(
                separator == "_"
                and child_index.isascii()
                and child_index.isdecimal()
                and parent_request_id in session.target_request_ids
                for target_request_id in session.target_request_ids
                for child_index, separator, parent_request_id in (
                    target_request_id.partition("_"),
                )
            )
            or session.world_size <= 0
            or session.rank < 0
            or session.rank >= session.world_size
            or session.total_num_kv_heads <= 0
            or session.observer is False
            or session.claim_scope != "instrumented_only"
        ):
            errors.append(f"mixed or invalid session metadata in {artifact.path}")
        recorded_world_size = engine_world_sizes.get(session.engine_id)
        if recorded_world_size is None:
            engine_world_sizes[session.engine_id] = session.world_size
        elif recorded_world_size != session.world_size:
            errors.append(f"engine {session.engine_id} sessions disagree on world size")
        recorded_kv_heads = engine_total_num_kv_heads.get(session.engine_id)
        if recorded_kv_heads is None:
            engine_total_num_kv_heads[session.engine_id] = session.total_num_kv_heads
        elif recorded_kv_heads != session.total_num_kv_heads:
            errors.append(
                f"engine {session.engine_id} sessions disagree on total KV heads"
            )
        for chronology_error in _artifact_chronology_errors(artifact):
            errors.append(f"{artifact.path}: {chronology_error}")
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
                if manifest.observer is False:
                    errors.append(
                        f"{artifact.path}: SOURCE_POST is not labeled as an observer"
                    )
                if record.stage is not IntegrityStage.SOURCE_POST:
                    continue
                lineage = _source_lineage(manifest)
                if lineage in source_post_by_lineage:
                    errors.append(f"duplicate SOURCE_POST lineage {lineage}")
                else:
                    source_post_by_lineage[lineage] = manifest
            elif isinstance(record, NixlPlanRecord):
                plan_key: ObserverKey = (
                    record.child_request_id,
                    record.observer_engine_id,
                    record.observer_rank,
                )
                if plan_key in plans:
                    errors.append(f"duplicate transfer plan {plan_key}")
                else:
                    plans[plan_key] = record
            elif isinstance(record, NixlCaptureRecord):
                if record.child_request_id is None:
                    errors.append("decoder capture has no child request id")
                    continue
                if record.stage not in (
                    IntegrityStage.STAGING_RAW,
                    IntegrityStage.STAGING_FENCED_CONTROL,
                    IntegrityStage.STAGING_POST_SCATTER,
                    IntegrityStage.DESTINATION,
                    IntegrityStage.PRE_READ,
                ):
                    errors.append(
                        f"decoder capture has unsupported stage {record.stage.value}"
                    )
                    continue
                observer_key: ObserverKey = (
                    record.child_request_id,
                    record.observer_engine_id,
                    record.observer_rank,
                )
                capture_key: CaptureKey = (
                    observer_key,
                    _capture_lineage(record),
                    record.stage,
                )
                if capture_key in captures:
                    errors.append(f"duplicate decoder capture {capture_key}")
                else:
                    captures[capture_key] = record
            else:
                if record.child_request_id is None:
                    errors.append("localization event has no child request id")
                    continue
                event_key: ObserverKey = (
                    record.child_request_id,
                    record.observer_engine_id,
                    record.observer_rank,
                )
                if event_key in events:
                    errors.append(f"duplicate child terminal event {event_key}")
                else:
                    events[event_key] = record

    for engine_id, world_size in engine_world_sizes.items():
        if world_size <= 0:
            continue
        actual_ranks = {rank for engine, rank in sessions if engine == engine_id}
        expected_ranks = set(range(world_size))
        if actual_ranks != expected_ranks:
            errors.append(
                f"engine {engine_id} process ranks are incomplete: "
                f"{sorted(actual_ranks)} != {sorted(expected_ranks)}"
            )

    errors.extend(
        _target_coverage_errors(
            reference_session.target_request_ids,
            tuple(source_post_by_lineage.values()),
            events,
        )
    )

    divergences: list[LocalizationDivergence] = []
    verified_observers: set[ObserverKey] = set()
    capture_stages = _capture_stages(reference_session.mode)
    plan_groups: dict[tuple[str, str], dict[int, NixlPlanRecord]] = {}
    for observer_key, plan in plans.items():
        plan_groups.setdefault(observer_key[:2], {})[observer_key[2]] = plan
    for (child_request_id, observer_engine_id), rank_plans in plan_groups.items():
        for plan_error in _cross_decoder_rank_plan_errors(
            child_request_id,
            observer_engine_id,
            rank_plans,
            engine_world_sizes,
            engine_total_num_kv_heads,
        ):
            errors.append(
                f"child {(child_request_id, observer_engine_id)}: {plan_error}"
            )

    for observer_key, plan in plans.items():
        plan_errors = (
            *_plan_errors(plan),
            *_plan_topology_errors(
                plan,
                engine_world_sizes,
                engine_total_num_kv_heads,
            ),
        )
        for plan_error in plan_errors:
            errors.append(f"plan {observer_key}: {plan_error}")
        event = events.get(observer_key)
        completed = (
            event is not None
            and event.code == "CAPTURE_COMPLETE"
            and event.evidentiary is False
        )
        if event is None:
            errors.append(f"plan has no terminal child outcome {observer_key}")
        if len(plan_errors) > 0:
            continue
        if not completed:
            continue
        reference_contract = plan.source_contracts[0]
        if (
            event is None
            or event.producer_engine_id != reference_contract.producer_engine_id
            or event.producer_request_id != reference_contract.producer_request_id
        ):
            errors.append(
                f"plan {observer_key} terminal source differs from its contract"
            )
            continue
        plan_verified: bool = True
        for source_contract in plan.source_contracts:
            lineage = _contract_lineage(source_contract)
            source_manifest = source_post_by_lineage.get(lineage)
            if source_manifest is None:
                errors.append(f"plan {observer_key} has no SOURCE_POST for {lineage}")
                plan_verified = False
                continue
            if source_contract_from_manifest(source_manifest) != source_contract:
                errors.append(
                    f"plan {observer_key} SOURCE_POST contract differs for {lineage}"
                )
                plan_verified = False
                continue
            divergence = _first_divergence(
                plan,
                source_contract,
                source_manifest,
                captures,
                compare_digests=reference_session.mode is not LocalizationMode.SHAM,
            )
            if divergence is not None:
                divergences.append(divergence)
                plan_verified = False
            if completed:
                for stage in capture_stages:
                    required_capture_key: CaptureKey = (
                        observer_key,
                        lineage,
                        stage,
                    )
                    if required_capture_key not in captures:
                        errors.append(
                            f"completed pull is missing capture {required_capture_key}"
                        )
                        plan_verified = False
        if plan_verified and reference_session.mode is not LocalizationMode.SHAM:
            verified_observers.add(observer_key)

    for indexed_capture_key in captures:
        observer_key, lineage, _ = indexed_capture_key
        capture_plan = plans.get(observer_key)
        if capture_plan is None:
            errors.append(
                f"decoder capture has no canonical plan {indexed_capture_key}"
            )
            continue
        plan_lineages = {
            _contract_lineage(contract) for contract in capture_plan.source_contracts
        }
        if lineage not in plan_lineages:
            errors.append(
                "decoder capture source lineage is absent from its plan "
                f"{indexed_capture_key}"
            )

    referenced_source_lineages = {
        _contract_lineage(contract)
        for plan in plans.values()
        for contract in plan.source_contracts
    }
    zero_byte_sources = {
        (
            event.run_id,
            event.transport_arm,
            event.producer_engine_id,
            event.producer_request_id,
        )
        for event in events.values()
        if event.code == "NON_EVIDENTIARY_ZERO_BYTE"
    }
    for lineage in source_post_by_lineage:
        if (
            lineage not in referenced_source_lineages
            and lineage[:4] not in zero_byte_sources
        ):
            errors.append(f"SOURCE_POST has no decoder transfer plan {lineage}")

    allowed_outcomes = {
        "CAPTURE_COMPLETE",
        "SHAM_CAPTURE_COMPLETE",
        "NON_EVIDENTIARY_ZERO_BYTE",
        "TRANSFER_ABORTED",
        "REQUEST_ABORTED",
        "TRANSFER_RETRY",
    }
    child_engines = {(key[0], key[1]) for key in events}
    if len(child_engines) == 0:
        errors.append("artifact set contains no child terminal outcomes")
    for child_id, observer_engine in child_engines:
        expected_ranks = set(range(engine_world_sizes.get(observer_engine, 0)))
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
            if event.evidentiary:
                errors.append(f"runtime terminal outcome {event.code} claims evidence")
        event_sources = {
            (event.producer_engine_id, event.producer_request_id)
            for event in child_events.values()
        }
        if len(event_sources) != 1:
            errors.append(
                f"child {(child_id, observer_engine)} ranks disagree on source"
            )
        child_plan_keys = {
            key for key in plans if key[0] == child_id and key[1] == observer_engine
        }
        child_capture_keys = {
            key
            for key in captures
            if key[0][0] == child_id and key[0][1] == observer_engine
        }
        if outcome_codes == {"CAPTURE_COMPLETE"}:
            if {key[2] for key in child_plan_keys} != expected_ranks:
                errors.append(
                    f"captured child {(child_id, observer_engine)} plans are incomplete"
                )
        elif outcome_codes == {"NON_EVIDENTIARY_ZERO_BYTE"} and (
            len(child_plan_keys) > 0 or len(child_capture_keys) > 0
        ):
            errors.append("zero-byte child contains physical-pull evidence")

    if len(errors) > 0:
        verified_observers.clear()
    physical_pulls = {(key[0], key[1]) for key in plans}
    verified_pulls: set[tuple[str, str]] = set()
    for child_id, observer_engine in physical_pulls:
        expected_ranks = set(range(engine_world_sizes.get(observer_engine, 0)))
        verified_ranks = {
            rank
            for child, engine, rank in verified_observers
            if child == child_id and engine == observer_engine
        }
        if len(expected_ranks) > 0 and verified_ranks == expected_ranks:
            verified_pulls.add((child_id, observer_engine))
    excluded_outcomes = child_engines - verified_pulls
    return LocalizationValidationReport(
        artifact_count=len(artifacts),
        physical_pull_count=len(physical_pulls),
        verified_pull_count=len(verified_pulls),
        excluded_outcome_count=len(excluded_outcomes),
        terminal_outcome_count=len(child_engines),
        divergences=tuple(divergences),
        errors=tuple(errors),
    )
