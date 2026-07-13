# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Diagnostic protocol and artifacts for P-to-D KV localization."""

import hashlib
import os
import re
import struct
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import BinaryIO, ClassVar, Protocol, TypeAlias

import msgspec

from vllm.distributed.kv_transfer.integrity import (
    IntegrityIdentity,
    IntegrityPayloadKind,
    IntegrityStage,
    compute_integrity_digest,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorWorkerMetadata,
)

GET_SOURCE_MANIFEST_MSG = b"get_source_manifest_v1"
LOCALIZATION_ARTIFACT_MAGIC = b"P2DLOC01"
LOCALIZATION_FRAME_PERSON = b"vllm-p2d-frame"
LOCALIZATION_ROOT_PERSON = b"vllm-p2d-root1"
_RANDOMIZED_REQUEST_ID_SUFFIX = re.compile(r"-[0-9a-f]{8}$")

IntegrityLeafKey: TypeAlias = tuple[
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    bytes,
    str,
]


class LocalizationMode(StrEnum):
    """Runtime behavior of the localization observer."""

    OFF = "off"
    TRACE = "trace"
    SHAM = "sham"


class ManifestStatus(StrEnum):
    """Producer response to a decoder manifest query."""

    READY = "ready"
    PENDING = "pending"
    REJECTED = "rejected"


class LocalizationError(RuntimeError):
    """Fail-closed error in the diagnostic localization protocol."""


def localization_request_id_base(request_id: str) -> str:
    """Remove vLLM's per-engine random suffix from a request identifier.

    Prefill and decode independently append ``-<8 hex>`` to the same stable
    request identifier. Localization selects that stable identity while its
    artifacts retain each engine's exact internal identifier.

    :param request_id: Stable or engine-internal request identifier.
    :returns: Stable request identifier shared by prefill and decode.
    """
    return _RANDOMIZED_REQUEST_ID_SUFFIX.sub("", request_id)


@dataclass(frozen=True, slots=True)
class NixlLocalizationConfig:
    """Validated process configuration for localization diagnostics.

    :ivar mode: Trace, sham observer-control, or disabled mode.
    :ivar run_id: Identifier shared by producer and decoder processes.
    :ivar transport_arm: Human-readable transport configuration.
    :ivar target_request_id: Exact stable request identifier to observe before
        vLLM appends its per-engine random suffix.
    :ivar artifact_dir: Directory receiving framed MessagePack artifacts.
    :ivar manifest_timeout_s: Maximum time D may wait before failing closed.
    :ivar copy_chunk_bytes: Upper bound for one device-to-host observer copy.
    :ivar strict_zero_byte: Whether non-evidentiary zero-byte hits fail the request.
    """

    mode: LocalizationMode
    run_id: str
    transport_arm: str
    target_request_id: str
    artifact_dir: Path | None
    manifest_timeout_s: float
    copy_chunk_bytes: int
    strict_zero_byte: bool

    def __post_init__(self) -> None:
        """Validate the relationship between mode and request scope."""
        if self.mode is LocalizationMode.OFF:
            if len(self.target_request_id) > 0:
                raise ValueError("disabled localization cannot select a request")
            return
        if len(self.target_request_id) == 0:
            raise ValueError("enabled localization requires a target request")
        if (
            localization_request_id_base(self.target_request_id)
            != self.target_request_id
        ):
            raise ValueError("localization target must not include a random suffix")

    @property
    def enabled(self) -> bool:
        """Return whether the observer and source gate are active."""
        return self.mode is not LocalizationMode.OFF

    def enabled_for(self, request_id: str) -> bool:
        """Return whether *request_id* is the configured observation target.

        :param request_id: Internal vLLM request identifier.
        :returns: Whether localization is enabled for the request.
        """
        return (
            self.enabled
            and localization_request_id_base(request_id) == self.target_request_id
        )

    @classmethod
    def from_environment(cls) -> "NixlLocalizationConfig":
        """Build and validate configuration from explicit environment values.

        :returns: Validated localization configuration.
        :raises ValueError: If an enabled configuration is incomplete or invalid.
        """
        raw_mode = os.environ.get("VLLM_NIXL_P2D_LOCALIZATION", "off")
        try:
            mode = LocalizationMode(raw_mode)
        except ValueError as error:
            choices = ", ".join(item.value for item in LocalizationMode)
            raise ValueError(
                f"VLLM_NIXL_P2D_LOCALIZATION must be one of {choices}"
            ) from error

        if mode is LocalizationMode.OFF:
            return cls(
                mode=mode,
                run_id="off",
                transport_arm="off",
                target_request_id="",
                artifact_dir=None,
                manifest_timeout_s=0.0,
                copy_chunk_bytes=64 * 1024 * 1024,
                strict_zero_byte=False,
            )

        run_id = os.environ.get("VLLM_NIXL_P2D_RUN_ID", "")
        transport_arm = os.environ.get("VLLM_NIXL_P2D_TRANSPORT_ARM", "")
        target_request_id = os.environ.get(
            "VLLM_NIXL_P2D_TARGET_REQUEST_ID",
            "",
        )
        artifact_dir_text = os.environ.get("VLLM_NIXL_P2D_ARTIFACT_DIR", "")
        if len(run_id) == 0:
            raise ValueError("VLLM_NIXL_P2D_RUN_ID is required")
        if len(transport_arm) == 0:
            raise ValueError("VLLM_NIXL_P2D_TRANSPORT_ARM is required")
        if len(target_request_id) == 0:
            raise ValueError("VLLM_NIXL_P2D_TARGET_REQUEST_ID is required")
        if len(artifact_dir_text) == 0:
            raise ValueError("VLLM_NIXL_P2D_ARTIFACT_DIR is required")

        timeout_s = float(os.environ.get("VLLM_NIXL_P2D_MANIFEST_TIMEOUT_S", "30"))
        chunk_mb = int(os.environ.get("VLLM_NIXL_P2D_COPY_CHUNK_MB", "64"))
        strict_zero_byte_text = os.environ.get(
            "VLLM_NIXL_P2D_STRICT_ZERO_BYTE",
            "0",
        )
        if timeout_s <= 0.0:
            raise ValueError("manifest timeout must be positive")
        if chunk_mb <= 0:
            raise ValueError("copy chunk size must be positive")
        if strict_zero_byte_text not in ("0", "1"):
            raise ValueError("strict zero-byte mode must be 0 or 1")

        return cls(
            mode=mode,
            run_id=run_id,
            transport_arm=transport_arm,
            target_request_id=target_request_id,
            artifact_dir=Path(artifact_dir_text),
            manifest_timeout_s=timeout_s,
            copy_chunk_bytes=chunk_mb * 1024 * 1024,
            strict_zero_byte=strict_zero_byte_text == "1",
        )


class NixlIntegrityLeaf(msgspec.Struct, array_like=True, frozen=True):
    """Compact digest and mapping record for one region and source position."""

    region_index: int
    group_index: int
    source_position: int
    remote_block_id: int
    valid_token_extent: int
    group_token_capacity: int
    semantic_contract_digest: bytes
    local_block_id: int | None
    plane_index: int
    destination_half: int | None
    rank_slot: int | None
    payload_kind: IntegrityPayloadKind
    byte_length: int
    digest: bytes


class NixlSourceRoster(msgspec.Struct, array_like=True, frozen=True):
    """Exact producer block roster awaiting a pre-transfer snapshot."""

    offer_generation: int
    iteration: int
    valid_token_extent: int
    group_token_capacities: tuple[int, ...]
    block_ids: tuple[tuple[int, ...], ...]


class NixlRegionDescriptor(msgspec.Struct, array_like=True, frozen=True):
    """Semantic and physical identity of one registered KV region."""

    semantic_name: str
    group_indices: tuple[int, ...]
    group_semantic_names: tuple[tuple[int, str], ...]
    base_address: int
    registered_bytes: int
    row_bytes: int
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    dtype: str
    element_size_bytes: int
    layout: str


class NixlSourceManifest(msgspec.Struct, array_like=True, frozen=True):
    """One P-rank source snapshot served to gated decoders."""

    schema_version: int
    run_id: str
    transport_arm: str
    producer_engine_id: str
    producer_request_id: str
    registration_generation: str
    offer_generation: int
    iteration: int
    source_rank: int
    region_lengths: tuple[int, ...]
    regions: tuple[NixlRegionDescriptor, ...]
    source_group_planes: tuple[int, ...]
    valid_token_extent: int
    group_token_capacities: tuple[int, ...]
    block_ids: tuple[tuple[int, ...], ...]
    observer: bool
    copied_bytes: int
    hashed_bytes: int
    duration_ns: int
    manifest_digest: bytes
    leaves: tuple[NixlIntegrityLeaf, ...]


class NixlSourceManifestResponse(msgspec.Struct, array_like=True, frozen=True):
    """Versioned response to a decoder's source-manifest query."""

    schema_version: int
    status: ManifestStatus
    detail: str
    manifests: tuple[NixlSourceManifest, ...]


@dataclass(slots=True)
class NixlLocalizationWorkerMetadata(KVConnectorWorkerMetadata):
    """Source manifests returned from P workers to the P scheduler.

    :ivar manifests: Producer request and generation to per-rank manifests.
    """

    manifests: dict[tuple[str, int], dict[int, NixlSourceManifest]]

    def aggregate(
        self,
        other: KVConnectorWorkerMetadata,
    ) -> "NixlLocalizationWorkerMetadata":
        """Merge disjoint rank manifests from one engine step.

        :param other: Metadata emitted by another tensor-parallel worker.
        :returns: New aggregate without mutating either input.
        :raises LocalizationError: If workers disagree about one rank manifest.
        """
        if not isinstance(other, NixlLocalizationWorkerMetadata):
            raise TypeError("cannot aggregate non-localization worker metadata")
        merged = {
            key: dict(rank_manifests)
            for key, rank_manifests in self.manifests.items()
        }
        for key, rank_manifests in other.manifests.items():
            target = merged.setdefault(key, {})
            overlap = set(target) & set(rank_manifests)
            if len(overlap) > 0:
                raise LocalizationError(
                    f"duplicate P-rank source manifests for {key}: {sorted(overlap)}"
                )
            target.update(rank_manifests)
        return NixlLocalizationWorkerMetadata(manifests=merged)


class NixlPlanPosition(msgspec.Struct, array_like=True, frozen=True):
    """Canonical source-to-destination mapping for one transfer position."""

    group_index: int
    source_position: int
    remote_block_id: int
    valid_token_extent: int
    group_token_capacity: int
    local_block_id: int
    plane_index: int


class NixlCaptureRecord(msgspec.Struct, array_like=True, frozen=True):
    """Compact artifact for one stage, child, and source rank."""

    RECORD_TYPE: ClassVar[str] = "capture"

    record_type: str
    schema_version: int
    stage: IntegrityStage
    run_id: str
    transport_arm: str
    producer_engine_id: str
    producer_request_id: str
    registration_generation: str
    source_manifest_digest: bytes
    offer_generation: int
    iteration: int
    child_request_id: str | None
    source_rank: int
    observer_engine_id: str
    observer_rank: int
    observer: bool
    copied_bytes: int
    hashed_bytes: int
    duration_ns: int
    barrier: str
    leaves: tuple[NixlIntegrityLeaf, ...]


class NixlPlanRecord(msgspec.Struct, array_like=True, frozen=True):
    """Exact raw and transfer-order mapping retained for placement checks."""

    RECORD_TYPE: ClassVar[str] = "plan"

    record_type: str
    schema_version: int
    run_id: str
    transport_arm: str
    producer_engine_id: str
    producer_request_id: str
    registration_generations: tuple[str, ...]
    source_manifest_digests: tuple[bytes, ...]
    offer_generation: int
    iteration: int
    child_request_id: str
    observer_engine_id: str
    observer_rank: int
    source_ranks: tuple[int, ...]
    rank_slots: tuple[int, ...]
    rank_slot_contract: tuple[tuple[int, int], ...]
    destination_group_planes: tuple[int, ...]
    region_lengths: tuple[int, ...]
    regions: tuple[tuple[NixlRegionDescriptor, ...], ...]
    local_regions: tuple[NixlRegionDescriptor, ...]
    raw_remote_groups: tuple[tuple[int, ...], ...]
    selected_remote_groups: tuple[tuple[int, ...], ...]
    selected_local_groups: tuple[tuple[int, ...], ...]
    transfer_order: tuple[NixlPlanPosition, ...]
    runs: tuple[tuple[int, int, int], ...]
    region_offsets: tuple[int, ...]
    staging_offset: int
    staging_size: int


class NixlSessionRecord(msgspec.Struct, array_like=True, frozen=True):
    """Header proving observer configuration for one process artifact."""

    RECORD_TYPE: ClassVar[str] = "session"

    record_type: str
    schema_version: int
    run_id: str
    transport_arm: str
    mode: LocalizationMode
    target_request_id: str
    engine_id: str
    rank: int
    pid: int
    observer: bool
    claim_scope: str
    copy_chunk_bytes: int
    strict_zero_byte: bool
    created_ns: int


class NixlSourceManifestRecord(msgspec.Struct, array_like=True, frozen=True):
    """Source PRE or POST manifest preserved as a first-class artifact."""

    RECORD_TYPE: ClassVar[str] = "source_manifest"

    record_type: str
    stage: IntegrityStage
    manifest: NixlSourceManifest


class NixlEventRecord(msgspec.Struct, array_like=True, frozen=True):
    """Explicit non-capture outcome retained in the diagnostic artifact."""

    RECORD_TYPE: ClassVar[str] = "event"

    record_type: str
    schema_version: int
    run_id: str
    transport_arm: str
    code: str
    evidentiary: bool
    producer_engine_id: str | None
    producer_request_id: str | None
    child_request_id: str | None
    observer_engine_id: str
    observer_rank: int
    detail: str
    created_ns: int


class NixlTerminalRecord(msgspec.Struct, array_like=True, frozen=True):
    """Completeness marker and root for every preceding artifact frame."""

    RECORD_TYPE: ClassVar[str] = "terminal"

    record_type: str
    schema_version: int
    record_count: int
    root_digest: bytes
    completed_ns: int


class _DigestHasher(Protocol):
    """Minimal interface shared by the artifact root hash implementations."""

    def update(self, data: bytes) -> None:
        """:param data: Bytes to add to the running digest."""

    def digest(self) -> bytes:
        """:returns: Current digest without consuming the hasher."""


class LocalizationArtifactWriter:
    """Append-only length-framed MessagePack writer for one worker process."""

    _SAFE_NAME: ClassVar[re.Pattern[str]] = re.compile(r"[^A-Za-z0-9_.-]+")

    _config: NixlLocalizationConfig
    _file: BinaryIO
    _path: Path
    _sequence: int
    _root_hasher: _DigestHasher
    _closed: bool

    def __init__(
        self,
        config: NixlLocalizationConfig,
        engine_id: str,
        rank: int,
    ) -> None:
        """Open a unique artifact and write its session header.

        :param config: Enabled localization configuration.
        :param engine_id: Local engine identifier.
        :param rank: Local tensor-parallel rank.
        """
        if config.enabled is False or config.artifact_dir is None:
            raise ValueError("artifact writer requires enabled localization")
        self._config = config
        config.artifact_dir.mkdir(parents=True, exist_ok=True)
        pid = os.getpid()
        safe_engine = self._SAFE_NAME.sub("_", engine_id)
        safe_run = self._SAFE_NAME.sub("_", config.run_id)
        self._path = config.artifact_dir / (
            f"{safe_run}.{safe_engine}.rank{rank}.pid{pid}.p2d.msgpack"
        )
        self._file = self._path.open("xb", buffering=0)
        self._sequence = 0
        self._root_hasher = hashlib.blake2b(
            digest_size=32,
            person=LOCALIZATION_ROOT_PERSON,
        )
        self._closed = False
        self.write(
            NixlSessionRecord(
                record_type=NixlSessionRecord.RECORD_TYPE,
                schema_version=IntegrityIdentity.SCHEMA_VERSION,
                run_id=config.run_id,
                transport_arm=config.transport_arm,
                mode=config.mode,
                target_request_id=config.target_request_id,
                engine_id=engine_id,
                rank=rank,
                pid=pid,
                observer=True,
                claim_scope="instrumented_only",
                copy_chunk_bytes=config.copy_chunk_bytes,
                strict_zero_byte=config.strict_zero_byte,
                created_ns=time.time_ns(),
            )
        )

    @property
    def path(self) -> Path:
        """Return the unique artifact path.

        :returns: Open process artifact path.
        """
        return self._path

    def write(
        self,
        record: (
            NixlSessionRecord
            | NixlPlanRecord
            | NixlCaptureRecord
            | NixlSourceManifestRecord
            | NixlEventRecord
        ),
    ) -> None:
        """Write one complete framed record.

        :param record: Typed artifact record.
        """
        if self._closed:
            raise ValueError("localization artifact is already closed")
        artifact_record = record
        if (
            self._config.mode is LocalizationMode.SHAM
            and isinstance(record, NixlCaptureRecord)
        ):
            redacted_leaves = tuple(
                NixlIntegrityLeaf(
                    region_index=leaf.region_index,
                    group_index=leaf.group_index,
                    source_position=leaf.source_position,
                    remote_block_id=leaf.remote_block_id,
                    valid_token_extent=leaf.valid_token_extent,
                    group_token_capacity=leaf.group_token_capacity,
                    semantic_contract_digest=leaf.semantic_contract_digest,
                    local_block_id=leaf.local_block_id,
                    plane_index=leaf.plane_index,
                    destination_half=leaf.destination_half,
                    rank_slot=leaf.rank_slot,
                    payload_kind=leaf.payload_kind,
                    byte_length=leaf.byte_length,
                    digest=b"\x00" * 16,
                )
                for leaf in record.leaves
            )
            artifact_record = NixlCaptureRecord(
                record_type=record.record_type,
                schema_version=record.schema_version,
                stage=record.stage,
                run_id=record.run_id,
                transport_arm=record.transport_arm,
                producer_engine_id=record.producer_engine_id,
                producer_request_id=record.producer_request_id,
                registration_generation=record.registration_generation,
                source_manifest_digest=record.source_manifest_digest,
                offer_generation=record.offer_generation,
                iteration=record.iteration,
                child_request_id=record.child_request_id,
                source_rank=record.source_rank,
                observer_engine_id=record.observer_engine_id,
                observer_rank=record.observer_rank,
                observer=record.observer,
                copied_bytes=record.copied_bytes,
                hashed_bytes=record.hashed_bytes,
                duration_ns=record.duration_ns,
                barrier=record.barrier,
                leaves=redacted_leaves,
            )
        self._write_frame(artifact_record)

    def _write_frame(
        self,
        record: (
            NixlSessionRecord
            | NixlPlanRecord
            | NixlCaptureRecord
            | NixlSourceManifestRecord
            | NixlEventRecord
            | NixlTerminalRecord
        ),
    ) -> None:
        """Write one frame and extend the running artifact root.

        :param record: Typed record, including the writer-owned terminal record.
        """
        payload = msgspec.msgpack.encode(record)
        sequence_bytes = struct.pack(">Q", self._sequence)
        checksum = hashlib.blake2b(
            sequence_bytes + payload,
            digest_size=32,
            person=LOCALIZATION_FRAME_PERSON,
        ).digest()
        frame = (
            LOCALIZATION_ARTIFACT_MAGIC
            + sequence_bytes
            + struct.pack(">Q", len(payload))
        )
        frame += checksum + payload
        written = self._file.write(frame)
        if written != len(frame):
            raise OSError(f"short localization artifact write: {written}/{len(frame)}")
        self._root_hasher.update(checksum)
        self._sequence += 1

    def close(self) -> None:
        """Close the process artifact."""
        if self._closed:
            return
        root_digest = self._root_hasher.digest()
        self._write_frame(
            NixlTerminalRecord(
                record_type=NixlTerminalRecord.RECORD_TYPE,
                schema_version=IntegrityIdentity.SCHEMA_VERSION,
                record_count=self._sequence,
                root_digest=root_digest,
                completed_ns=time.time_ns(),
            )
        )
        os.fsync(self._file.fileno())
        self._file.close()
        self._closed = True


def compute_semantic_contract_digest(
    *,
    region: NixlRegionDescriptor,
    group_index: int,
    group_token_capacity: int,
    source_plane_contract: int,
) -> bytes:
    """Hash the exact source region/group semantic interpretation.

    :param region: Registered source-region descriptor.
    :param group_index: Owning KV cache group.
    :param group_token_capacity: Runtime token capacity of one source block.
    :param source_plane_contract: Number of source payload planes for the group.
    :returns: Thirty-two-byte BLAKE2b semantic contract digest.
    :raises LocalizationError: If the descriptor does not name the group exactly.
    """
    names = dict(region.group_semantic_names)
    if len(names) != len(region.group_semantic_names):
        raise LocalizationError("region has duplicate group semantic names")
    group_semantic_name = names.get(group_index)
    if group_semantic_name is None:
        raise LocalizationError(
            f"region semantic contract has no owner group {group_index}"
        )
    payload = msgspec.msgpack.encode(
        (
            IntegrityIdentity.SCHEMA_VERSION,
            group_index,
            group_semantic_name,
            region.semantic_name,
            region.group_indices,
            region.shape,
            region.strides,
            region.dtype,
            region.element_size_bytes,
            region.layout,
            region.row_bytes,
            group_token_capacity,
            source_plane_contract,
        )
    )
    return hashlib.blake2b(
        payload,
        digest_size=32,
        person=b"vllm-p2d-sem-v1",
    ).digest()


def build_integrity_identity(
    *,
    config: NixlLocalizationConfig,
    producer_engine_id: str,
    producer_request_id: str,
    registration_generation: str,
    semantic_contract_digest: bytes,
    offer_generation: int,
    iteration: int,
    source_rank: int,
    region_index: int,
    group_index: int,
    plane_index: int,
    source_position: int,
    remote_block_id: int,
    valid_token_extent: int,
    group_token_capacity: int,
    payload_kind: IntegrityPayloadKind,
    byte_length: int,
) -> IntegrityIdentity:
    """Build the shared digest identity from NIXL lineage.

    :returns: Canonical stage-independent identity.
    """
    return IntegrityIdentity(
        run_id=config.run_id,
        transport_arm=config.transport_arm,
        producer_engine_id=producer_engine_id,
        producer_request_id=producer_request_id,
        registration_generation=registration_generation,
        semantic_contract_digest=semantic_contract_digest,
        offer_generation=offer_generation,
        iteration=iteration,
        source_rank=source_rank,
        region_index=region_index,
        group_index=group_index,
        plane_index=plane_index,
        source_position=source_position,
        remote_block_id=remote_block_id,
        valid_token_extent=valid_token_extent,
        group_token_capacity=group_token_capacity,
        payload_kind=payload_kind,
        byte_length=byte_length,
    )


def compute_source_manifest_digest(manifest: NixlSourceManifest) -> bytes:
    """Hash a source manifest, excluding its self-referential digest field.

    :param manifest: Manifest to authenticate against accidental corruption,
        truncation, or substitution.
    :returns: Thirty-two-byte BLAKE2b digest.
    """
    payload = msgspec.msgpack.encode(
        (
            manifest.schema_version,
            manifest.run_id,
            manifest.transport_arm,
            manifest.producer_engine_id,
            manifest.producer_request_id,
            manifest.registration_generation,
            manifest.offer_generation,
            manifest.iteration,
            manifest.source_rank,
            manifest.region_lengths,
            manifest.regions,
            manifest.source_group_planes,
            manifest.valid_token_extent,
            manifest.group_token_capacities,
            manifest.block_ids,
            manifest.observer,
            manifest.copied_bytes,
            manifest.hashed_bytes,
            manifest.duration_ns,
            manifest.leaves,
        )
    )
    return hashlib.blake2b(
        payload,
        digest_size=32,
        person=b"vllm-p2d-man-v1",
    ).digest()


def seal_source_manifest(manifest: NixlSourceManifest) -> NixlSourceManifest:
    """Return an otherwise identical manifest carrying its content digest.

    :param manifest: Unsealed manifest whose digest field is ignored.
    :returns: Sealed manifest.
    """
    return NixlSourceManifest(
        schema_version=manifest.schema_version,
        run_id=manifest.run_id,
        transport_arm=manifest.transport_arm,
        producer_engine_id=manifest.producer_engine_id,
        producer_request_id=manifest.producer_request_id,
        registration_generation=manifest.registration_generation,
        offer_generation=manifest.offer_generation,
        iteration=manifest.iteration,
        source_rank=manifest.source_rank,
        region_lengths=manifest.region_lengths,
        regions=manifest.regions,
        source_group_planes=manifest.source_group_planes,
        valid_token_extent=manifest.valid_token_extent,
        group_token_capacities=manifest.group_token_capacities,
        block_ids=manifest.block_ids,
        observer=manifest.observer,
        copied_bytes=manifest.copied_bytes,
        hashed_bytes=manifest.hashed_bytes,
        duration_ns=manifest.duration_ns,
        manifest_digest=compute_source_manifest_digest(manifest),
        leaves=manifest.leaves,
    )


def build_integrity_leaf(
    *,
    identity: IntegrityIdentity,
    payload: bytes | bytearray | memoryview,
    local_block_id: int | None,
    destination_half: int | None,
    rank_slot: int | None,
) -> NixlIntegrityLeaf:
    """Hash payload bytes and retain their placement metadata.

    :returns: Compact integrity leaf.
    """
    return NixlIntegrityLeaf(
        region_index=identity.region_index,
        group_index=identity.group_index,
        source_position=identity.source_position,
        remote_block_id=identity.remote_block_id,
        valid_token_extent=identity.valid_token_extent,
        group_token_capacity=identity.group_token_capacity,
        semantic_contract_digest=identity.semantic_contract_digest,
        local_block_id=local_block_id,
        plane_index=identity.plane_index,
        destination_half=destination_half,
        rank_slot=rank_slot,
        payload_kind=identity.payload_kind,
        byte_length=identity.byte_length,
        digest=compute_integrity_digest(identity, payload),
    )


def leaf_source_key(leaf: NixlIntegrityLeaf) -> IntegrityLeafKey:
    """Return a stage-independent key for one leaf.

    :returns: Region, group, position, block, plane, and payload-kind tuple.
    """
    return (
        leaf.region_index,
        leaf.group_index,
        leaf.source_position,
        leaf.remote_block_id,
        leaf.plane_index,
        leaf.valid_token_extent,
        leaf.group_token_capacity,
        leaf.semantic_contract_digest,
        leaf.payload_kind.value,
    )


def select_source_manifest(
    manifest: NixlSourceManifest,
    keys: set[IntegrityLeafKey],
) -> NixlSourceManifest:
    """Create a sealed child-specific view of a full producer manifest.

    :param manifest: Full source registration manifest.
    :param keys: Exact leaves selected by prefix trimming and the D layout.
    :returns: Sealed manifest containing exactly the requested leaves.
    :raises LocalizationError: If a requested leaf is absent.
    """
    available = {leaf_source_key(leaf) for leaf in manifest.leaves}
    missing = keys - available
    if len(missing) > 0:
        raise LocalizationError(f"selected source leaves are absent: {sorted(missing)}")
    selected = tuple(
        leaf for leaf in manifest.leaves if leaf_source_key(leaf) in keys
    )
    return seal_source_manifest(
        NixlSourceManifest(
            schema_version=manifest.schema_version,
            run_id=manifest.run_id,
            transport_arm=manifest.transport_arm,
            producer_engine_id=manifest.producer_engine_id,
            producer_request_id=manifest.producer_request_id,
            registration_generation=manifest.registration_generation,
            offer_generation=manifest.offer_generation,
            iteration=manifest.iteration,
            source_rank=manifest.source_rank,
            region_lengths=manifest.region_lengths,
            regions=manifest.regions,
            source_group_planes=manifest.source_group_planes,
            valid_token_extent=manifest.valid_token_extent,
            group_token_capacities=manifest.group_token_capacities,
            block_ids=manifest.block_ids,
            observer=manifest.observer,
            copied_bytes=manifest.copied_bytes,
            hashed_bytes=manifest.hashed_bytes,
            duration_ns=manifest.duration_ns,
            manifest_digest=b"",
            leaves=selected,
        )
    )


def validate_source_manifest_structure(
    manifest: NixlSourceManifest,
) -> tuple[str, ...]:
    """Validate source-manifest cardinality, uniqueness, and leaf contracts.

    :param manifest: Sealed producer manifest.
    :returns: Structural errors; empty means the manifest is internally sound.
    """
    errors: list[str] = []
    if manifest.schema_version != IntegrityIdentity.SCHEMA_VERSION:
        errors.append("source manifest schema mismatch")
    if len(manifest.region_lengths) != len(manifest.regions):
        errors.append("source manifest region cardinality mismatch")
    num_groups = len(manifest.block_ids)
    if len(manifest.source_group_planes) != num_groups:
        errors.append("source manifest plane-contract cardinality mismatch")
    if len(manifest.group_token_capacities) != num_groups:
        errors.append("source manifest token-capacity cardinality mismatch")
    if any(planes not in (1, 2) for planes in manifest.source_group_planes):
        errors.append("source manifest contains an invalid source plane contract")
    if manifest.valid_token_extent <= 0:
        errors.append("source manifest has an invalid request token extent")
    keys = [leaf_source_key(leaf) for leaf in manifest.leaves]
    if len(set(keys)) != len(keys):
        errors.append("source manifest contains duplicate leaves")

    owned_groups: set[int] = set()
    for region_index, region in enumerate(manifest.regions):
        region_groups = tuple(sorted(set(region.group_indices)))
        if len(region.group_indices) == 0:
            errors.append(f"source region {region_index} has no semantic owners")
        if region.group_indices != region_groups:
            errors.append(f"source region {region_index} owners are not canonical")
        if any(group < 0 or group >= num_groups for group in region.group_indices):
            errors.append(f"source region {region_index} has an invalid owner")
        semantic_names = tuple(sorted(region.group_semantic_names))
        if region.group_semantic_names != semantic_names:
            errors.append(
                f"source region {region_index} semantic names are not canonical"
            )
        if {group for group, _ in region.group_semantic_names} != set(
            region.group_indices
        ):
            errors.append(
                f"source region {region_index} semantic names differ from owners"
            )
        if any(len(name) == 0 for _, name in region.group_semantic_names):
            errors.append(f"source region {region_index} has an empty semantic name")
        if (
            len(region.shape) == 0
            or len(region.shape) != len(region.strides)
            or region.element_size_bytes <= 0
        ):
            errors.append(f"source region {region_index} has invalid tensor geometry")
        elif (
            region.shape[0] <= 0
            or region.registered_bytes != region.shape[0] * region.row_bytes
            or region.strides[0] * region.element_size_bytes != region.row_bytes
        ):
            errors.append(
                f"source region {region_index} physical geometry is not row canonical"
            )
        owned_groups.update(region.group_indices)
    if owned_groups != set(range(num_groups)):
        errors.append("source semantic regions do not cover every cache group")

    for group_index, _ in enumerate(manifest.block_ids):
        if group_index >= len(manifest.group_token_capacities):
            continue
        capacity = manifest.group_token_capacities[group_index]
        if capacity <= 0:
            errors.append(f"source group {group_index} has invalid token capacity")

    expected_keys: set[IntegrityLeafKey] = set()
    expected_lengths: dict[IntegrityLeafKey, int] = {}
    expected_copied_bytes = 0
    expected_hashed_bytes = 0
    for region_index, region in enumerate(manifest.regions):
        if region_index >= len(manifest.region_lengths):
            continue
        if region.row_bytes != manifest.region_lengths[region_index]:
            errors.append(f"source region {region_index} row length mismatch")
        if region.row_bytes <= 0 or region.row_bytes % 2 != 0:
            errors.append(f"source region {region_index} has invalid row length")
            continue
        for group_index, blocks in enumerate(manifest.block_ids):
            if group_index not in region.group_indices:
                continue
            if group_index >= len(manifest.group_token_capacities):
                continue
            try:
                semantic_contract_digest = compute_semantic_contract_digest(
                    region=region,
                    group_index=group_index,
                    group_token_capacity=(
                        manifest.group_token_capacities[group_index]
                    ),
                    source_plane_contract=manifest.source_group_planes[
                        group_index
                    ],
                )
            except LocalizationError:
                errors.append(
                    f"source region {region_index} group {group_index} "
                    "semantic contract is invalid"
                )
                continue
            for source_position, block_id in enumerate(blocks):
                for plane_index, payload_kind, byte_length in (
                    (-1, IntegrityPayloadKind.WIRE, region.row_bytes),
                    (0, IntegrityPayloadKind.COMMIT, region.row_bytes // 2),
                    (1, IntegrityPayloadKind.COMMIT, region.row_bytes // 2),
                ):
                    key = (
                        region_index,
                        group_index,
                        source_position,
                        block_id,
                        plane_index,
                        manifest.valid_token_extent,
                        manifest.group_token_capacities[group_index],
                        semantic_contract_digest,
                        payload_kind.value,
                    )
                    expected_keys.add(key)
                    expected_lengths[key] = byte_length
                expected_copied_bytes += region.row_bytes
                expected_hashed_bytes += 2 * region.row_bytes

    actual = {leaf_source_key(leaf): leaf for leaf in manifest.leaves}
    if set(actual) != expected_keys:
        errors.append("source manifest leaf set does not match its exact roster")
    for key in set(actual) & expected_keys:
        leaf = actual[key]
        if leaf.byte_length != expected_lengths[key]:
            errors.append(f"source leaf length mismatch {key}")
        if len(leaf.digest) != 16:
            errors.append(f"source leaf digest length mismatch {key}")
        if len(leaf.semantic_contract_digest) != 32:
            errors.append(f"source semantic contract digest length mismatch {key}")
        if (
            leaf.local_block_id is not None
            or leaf.destination_half is not None
            or leaf.rank_slot is not None
        ):
            errors.append(f"source leaf carries destination mapping {key}")
    if manifest.copied_bytes != expected_copied_bytes:
        errors.append("source manifest copied-byte count mismatch")
    if manifest.hashed_bytes != expected_hashed_bytes:
        errors.append("source manifest hashed-byte count mismatch")
    if compute_source_manifest_digest(manifest) != manifest.manifest_digest:
        errors.append("source manifest content digest mismatch")
    return tuple(errors)


def resolve_source_manifest_request(
    request: list[object] | tuple[object, ...],
    config: NixlLocalizationConfig,
    stored_manifests: dict[
        tuple[str, int], dict[int, NixlSourceManifest]
    ],
    available_ranks: set[int] | None = None,
) -> NixlSourceManifestResponse:
    """Resolve one untrusted decoder query without mutating producer state.

    :param request: Decoded side-channel request.
    :param config: Producer localization configuration.
    :param stored_manifests: Prepared manifests keyed by request and offer.
    :param available_ranks: Producer ranks valid for this engine incarnation.
    :returns: Ready, pending, or fail-closed rejection response.
    """
    def rejected(detail: str) -> NixlSourceManifestResponse:
        return NixlSourceManifestResponse(
            schema_version=IntegrityIdentity.SCHEMA_VERSION,
            status=ManifestStatus.REJECTED,
            detail=detail,
            manifests=(),
        )

    if len(request) != 8 or request[0] != GET_SOURCE_MANIFEST_MSG:
        return rejected("invalid source-manifest request schema")
    (
        _,
        run_id,
        transport_arm,
        producer_engine_id,
        producer_request_id,
        offer_generation,
        iteration,
        required_ranks_raw,
    ) = request
    if (
        not isinstance(run_id, str)
        or len(run_id) == 0
        or not isinstance(transport_arm, str)
        or len(transport_arm) == 0
        or not isinstance(producer_engine_id, str)
        or len(producer_engine_id) == 0
        or not isinstance(producer_request_id, str)
        or len(producer_request_id) == 0
        or type(offer_generation) is not int
        or offer_generation < 0
        or type(iteration) is not int
        or iteration < 0
        or not isinstance(required_ranks_raw, (list, tuple))
    ):
        return rejected("invalid source-manifest request fields")
    if len(required_ranks_raw) == 0 or any(
        type(rank) is not int or rank < 0 for rank in required_ranks_raw
    ):
        return rejected("invalid required source ranks")
    required_ranks = tuple(int(rank) for rank in required_ranks_raw)
    if len(set(required_ranks)) != len(required_ranks):
        return rejected("duplicate required source ranks")
    if available_ranks is not None and not set(required_ranks).issubset(
        available_ranks
    ):
        return rejected("required source ranks are outside the producer topology")
    if config.enabled is False:
        return rejected("producer localization observer is disabled")
    if config.enabled_for(producer_request_id) is False:
        return rejected("producer request is outside the localization target")
    if run_id != config.run_id or transport_arm != config.transport_arm:
        return rejected("localization run or transport arm mismatch")

    key = (producer_request_id, offer_generation)
    rank_manifests = stored_manifests.get(key)
    if rank_manifests is None:
        return NixlSourceManifestResponse(
            schema_version=IntegrityIdentity.SCHEMA_VERSION,
            status=ManifestStatus.PENDING,
            detail="waiting for required P-rank snapshots",
            manifests=(),
        )
    stored_ranks = set(rank_manifests)
    required_rank_set = set(required_ranks)
    if not required_rank_set.issubset(stored_ranks):
        return NixlSourceManifestResponse(
            schema_version=IntegrityIdentity.SCHEMA_VERSION,
            status=ManifestStatus.PENDING,
            detail="waiting for required P-rank snapshots",
            manifests=(),
        )

    manifests = tuple(rank_manifests[rank] for rank in required_ranks)
    reference = manifests[0]
    reference_semantics = (
        reference.region_lengths,
        tuple(
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
        ),
        reference.source_group_planes,
        reference.valid_token_extent,
        reference.group_token_capacities,
        reference.block_ids,
    )
    for rank, manifest in zip(required_ranks, manifests, strict=True):
        structure_errors = validate_source_manifest_structure(manifest)
        if len(structure_errors) > 0:
            return rejected(
                f"invalid source manifest rank {rank}: {structure_errors[0]}"
            )
        if (
            manifest.run_id != run_id
            or manifest.transport_arm != transport_arm
            or manifest.producer_engine_id != producer_engine_id
            or manifest.producer_request_id != producer_request_id
            or manifest.offer_generation != offer_generation
            or manifest.iteration != iteration
            or manifest.source_rank != rank
            or manifest.observer is False
        ):
            return rejected(f"source manifest lineage mismatch on rank {rank}")
        semantics = (
            manifest.region_lengths,
            tuple(
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
                for region in manifest.regions
            ),
            manifest.source_group_planes,
            manifest.valid_token_extent,
            manifest.group_token_capacities,
            manifest.block_ids,
        )
        if semantics != reference_semantics:
            return rejected("source ranks disagree on semantic transfer contract")
    return NixlSourceManifestResponse(
        schema_version=IntegrityIdentity.SCHEMA_VERSION,
        status=ManifestStatus.READY,
        detail="all required P-rank snapshots are ready",
        manifests=manifests,
    )


def validate_capture(
    manifest: NixlSourceManifest,
    actual_leaves: tuple[NixlIntegrityLeaf, ...],
    expected_mapping: dict[IntegrityLeafKey, tuple[int, int, int]] | None,
    *,
    compare_digests: bool = True,
) -> tuple[str, ...]:
    """Validate content, exact cardinality, and optional destination mapping.

    :param manifest: Producer source manifest for one source rank.
    :param actual_leaves: Stage observations to validate.
    :param expected_mapping: Source key to local block, rank slot, and half.
    :param compare_digests: Whether to compare content digests. Sham artifacts
        disable this while retaining exact cardinality and placement checks.
    :returns: Human-readable mismatches; empty means the capture agrees.
    """
    expected = {leaf_source_key(leaf): leaf for leaf in manifest.leaves}
    actual: dict[IntegrityLeafKey, NixlIntegrityLeaf] = {}
    errors: list[str] = []
    for leaf in actual_leaves:
        key = leaf_source_key(leaf)
        if key in actual:
            errors.append(f"duplicate actual leaf {key}")
            continue
        actual[key] = leaf

    expected_keys = set(expected)
    if expected_mapping is not None and set(expected_mapping) != expected_keys:
        return ("canonical mapping keys differ from manifest leaves",)
    actual_keys = set(actual)
    for key in sorted(expected_keys - actual_keys):
        errors.append(f"missing actual leaf {key}")
    for key in sorted(actual_keys - expected_keys):
        errors.append(f"unexpected actual leaf {key}")
    for key in sorted(expected_keys & actual_keys):
        expected_leaf = expected[key]
        actual_leaf = actual[key]
        if expected_leaf.byte_length != actual_leaf.byte_length:
            errors.append(f"byte-length mismatch {key}")
        if compare_digests and expected_leaf.digest != actual_leaf.digest:
            errors.append(f"digest mismatch {key}")
        if expected_mapping is None:
            continue
        mapping = expected_mapping.get(key)
        if mapping is None:
            errors.append(f"missing canonical mapping {key}")
            continue
        if (
            actual_leaf.local_block_id,
            actual_leaf.rank_slot,
            actual_leaf.destination_half,
        ) != mapping:
            errors.append(f"destination mapping mismatch {key}")
    return tuple(errors)


def locate_subsequence(source: list[int], selected: list[int]) -> int:
    """Locate one exact selected run inside the unsorted source roster.

    :param source: Original physical block roster.
    :param selected: Prefix-cache-selected contiguous sub-roster.
    :returns: Starting source position.
    :raises LocalizationError: If the selection is absent or ambiguous.
    """
    if len(selected) == 0:
        return 0
    limit = len(source) - len(selected) + 1
    matches = [
        start
        for start in range(max(limit, 0))
        if source[start : start + len(selected)] == selected
    ]
    if len(matches) != 1:
        raise LocalizationError(
            "selected remote blocks are not one unambiguous source sub-roster"
        )
    return matches[0]
