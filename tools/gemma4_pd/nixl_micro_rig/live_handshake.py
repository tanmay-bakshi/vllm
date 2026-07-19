"""Capture and replay live connector-v9 handshake evidence."""

import hashlib
import inspect
import json
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import msgspec
from zmq.constants import SocketOption, SocketType
from zmq.error import ZMQError
from zmq.sugar.context import Context

from tools.gemma4_pd.nixl_micro_rig.config import RigConfig
from vllm.config import KVTransferConfig, VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import TransferTopology
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import (
    NixlBaseConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    GET_META_MSG,
    NIXL_CONNECTOR_VERSION,
    NixlAgentMetadata,
    NixlHandshakePayload,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import (
    compute_tp_mapping,
)
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig

LIVE_HANDSHAKE_SCHEMA_VERSION = 1
LIVE_HANDSHAKE_EVIDENCE_SCOPE = "live_connector_v9_wire_capture"
_EXPECTED_ATTENTION_BACKEND = "FLASHINFER_GEMMA4_TRTLLM_GEN"


class LiveHandshakeError(RuntimeError):
    """Report an invalid, unauthenticated, or incompatible live capture."""


class CapturedCodeIdentity(
    msgspec.Struct,
    frozen=True,
    forbid_unknown_fields=True,
):
    """Bind a capture to one clean Git commit and tree.

    :ivar commit: Exact Git commit identity.
    :ivar tree: Exact Git tree identity.
    :ivar worktree_clean: Whether the complete worktree matched the commit.
    """

    commit: str
    tree: str
    worktree_clean: bool


class CapturedRigIdentity(
    msgspec.Struct,
    frozen=True,
    forbid_unknown_fields=True,
):
    """Bind a capture to one authenticated micro-rig contract.

    :ivar file_sha256: Exact configuration-file digest.
    :ivar fingerprint: Parsed typed-configuration fingerprint.
    :ivar file_name: Informational configuration basename.
    """

    file_sha256: str
    fingerprint: str
    file_name: str


class CapturedModelIdentity(
    msgspec.Struct,
    frozen=True,
    forbid_unknown_fields=True,
):
    """Bind replay facts to the authenticated deployed model configuration.

    :ivar config_payload: Exact deployed Hugging Face ``config.json`` bytes.
    :ivar config_sha256: Independently expected configuration digest.
    :ivar file_name: Informational configuration basename.
    :ivar model_type: Declared Hugging Face model type.
    :ivar architectures: Declared model architectures before launch overrides.
    :ivar total_num_kv_heads: Total KV-head count derived from the exact payload.
    """

    config_payload: bytes
    config_sha256: str
    file_name: str
    model_type: str
    architectures: tuple[str, ...]
    total_num_kv_heads: int


class CapturedWirePayload(
    msgspec.Struct,
    frozen=True,
    forbid_unknown_fields=True,
):
    """Preserve one exact rank response and its semantic identities.

    :ivar rank: Endpoint tensor-parallel rank.
    :ivar payload: Exact outer ``NixlHandshakePayload`` MessagePack bytes.
    :ivar payload_sha256: Digest of ``payload``.
    :ivar compatibility_hash: Decoded connector compatibility hash.
    :ivar engine_id: Decoded engine identity.
    :ivar normalized_contract_sha256: Address-independent rank-contract digest.
    :ivar registration_generation_sha256: Digest of the volatile registration
        generation preserved inside ``payload``.
    """

    rank: int
    payload: bytes
    payload_sha256: str
    compatibility_hash: str
    engine_id: str
    normalized_contract_sha256: str
    registration_generation_sha256: str


class CapturedEndpoint(
    msgspec.Struct,
    frozen=True,
    forbid_unknown_fields=True,
):
    """Describe one queried scheduler endpoint.

    :ivar host: Exact queried host.
    :ivar port: Exact queried port.
    :ivar engine_id: Independently expected engine identity.
    :ivar tensor_parallel_size: Exact endpoint TP size.
    :ivar payloads: Rank-ordered wire responses.
    """

    host: str
    port: int
    engine_id: str
    tensor_parallel_size: int
    payloads: tuple[CapturedWirePayload, ...]


class ValidatorReplayContract(
    msgspec.Struct,
    frozen=True,
    forbid_unknown_fields=True,
):
    """Record decoder facts not carried explicitly by the wire payload.

    :ivar total_num_kv_heads: Model-derived total KV-head count.
    :ivar local_tensor_parallel_size: Decoder TP size used by replay.
    :ivar group_spec_kinds: Exact group classification used to derive TP mapping.
    :ivar hybrid_memory_allocator_required: Whether the live decoder uses HMA.
    :ivar use_mla: Whether the model uses MLA.
    :ivar has_mamba: Whether the model contains Mamba cache groups.
    :ivar use_host_buffer: Whether the decoder receives through a host buffer.
    :ivar enable_permute_local_kv: Whether replay permits KV-layout permutation.
    :ivar enable_prefix_caching: Whether prefix caching affects the contract.
    """

    total_num_kv_heads: int
    local_tensor_parallel_size: int
    group_spec_kinds: tuple[str, ...]
    hybrid_memory_allocator_required: bool
    use_mla: bool
    has_mamba: bool
    use_host_buffer: bool
    enable_permute_local_kv: bool
    enable_prefix_caching: bool


class LiveHandshakeCapture(
    msgspec.Struct,
    frozen=True,
    forbid_unknown_fields=True,
):
    """Self-contained live wire capture with an offline replay contract.

    :ivar schema_version: Evidence schema version.
    :ivar evidence_scope: Explicit live-wire evidence classification.
    :ivar connector_version: Connector version of the capturing code.
    :ivar captured_at: UTC capture timestamp.
    :ivar code: Clean source identity.
    :ivar rig: Typed production-geometry identity.
    :ivar model: Authenticated deployed model configuration.
    :ivar replay: Decoder-side production-validator projection.
    :ivar producer: Complete four-rank producer endpoint.
    :ivar decoder: Exact TP1 decoder endpoint.
    """

    schema_version: int
    evidence_scope: str
    connector_version: int
    captured_at: str
    code: CapturedCodeIdentity
    rig: CapturedRigIdentity
    model: CapturedModelIdentity
    replay: ValidatorReplayContract
    producer: CapturedEndpoint
    decoder: CapturedEndpoint


@dataclass(frozen=True, slots=True)
class CaptureWriteResult:
    """Report the immutable capture path and externally sealable digest.

    :ivar path: Fresh read-only capture file.
    :ivar sha256: Digest that an external evidence manifest must preserve.
    """

    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class CaptureVerificationResult:
    """Report successful authentication and production-validator replay.

    :ivar capture_sha256: Authenticated capture digest.
    :ivar code_commit: Bound code commit.
    :ivar code_tree: Bound code tree.
    :ivar compatibility_hash: Shared live P/D compatibility hash.
    :ivar producer_engine_id: Captured P engine identity.
    :ivar decoder_engine_id: Captured D engine identity.
    :ivar producer_rank_count: Complete P rank count.
    :ivar validator_replay: Terminal replay verdict.
    """

    capture_sha256: str
    code_commit: str
    code_tree: str
    compatibility_hash: str
    producer_engine_id: str
    decoder_engine_id: str
    producer_rank_count: int
    validator_replay: str


class _ReplayAttentionBackend:
    """Supply the cache orientation needed by :class:`TransferTopology`."""

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[int, ...]:
        """Return a synthetic K/V-first shape matching the captured HND cache.

        :param num_blocks: Registered row count.
        :param block_size: Physical tokens per block.
        :param num_kv_heads: Local physical KV heads.
        :param head_size: Elements per KV head.
        :returns: K/V-first cache shape.
        """
        return (2, num_blocks, num_kv_heads, block_size, head_size)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: str, label: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise LiveHandshakeError(f"{label} must contain lowercase SHA-256 hex")


def _require_git_object_id(value: str, label: str) -> None:
    if len(value) not in (40, 64) or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise LiveHandshakeError(
            f"{label} must contain lowercase SHA-1 or SHA-256 Git object hex"
        )


def _model_identity(
    config_path: Path,
    expected_sha256: str,
) -> CapturedModelIdentity:
    """Authenticate and decode the deployed model's KV-head contract.

    :param config_path: Exact deployed Hugging Face ``config.json``.
    :param expected_sha256: Digest from the independently authenticated model
        source manifest.
    :returns: Exact model bytes and typed facts needed by replay.
    :raises LiveHandshakeError: If the digest or required model fields differ.
    """
    payload = config_path.read_bytes()
    return _model_identity_from_payload(config_path, payload, expected_sha256)


def _model_identity_from_payload(
    config_path: Path,
    payload: bytes,
    expected_sha256: str,
) -> CapturedModelIdentity:
    """Decode exact model bytes after authenticating their external digest.

    :param config_path: Informational model-configuration path.
    :param payload: Exact model-configuration bytes.
    :param expected_sha256: Independently authenticated digest.
    :returns: Exact model bytes and typed replay facts.
    :raises LiveHandshakeError: If the digest or required fields differ.
    """
    _require_sha256(expected_sha256, "expected model-config digest")
    actual_sha256 = _sha256(payload)
    if actual_sha256 != expected_sha256:
        raise LiveHandshakeError(
            f"model-config SHA-256 {actual_sha256} differs from {expected_sha256}"
        )
    try:
        decoded = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LiveHandshakeError("model config is not valid UTF-8 JSON") from error
    if not isinstance(decoded, dict):
        raise LiveHandshakeError("model config root must be an object")

    model_type = decoded.get("model_type")
    architectures = decoded.get("architectures")
    if not isinstance(model_type, str) or len(model_type) == 0:
        raise LiveHandshakeError("model config has no model_type")
    if (
        not isinstance(architectures, list)
        or len(architectures) == 0
        or any(
            not isinstance(architecture, str) or len(architecture) == 0
            for architecture in architectures
        )
    ):
        raise LiveHandshakeError("model config has no valid architectures")

    kv_head_values: list[int] = []
    for candidate in (decoded, decoded.get("text_config")):
        if not isinstance(candidate, dict):
            continue
        value = candidate.get("num_key_value_heads")
        if type(value) is int and value > 0:
            kv_head_values.append(value)
    if len(kv_head_values) == 0:
        raise LiveHandshakeError("model config has no positive num_key_value_heads")
    if len(set(kv_head_values)) != 1:
        raise LiveHandshakeError("model config declares inconsistent KV-head counts")

    return CapturedModelIdentity(
        config_payload=payload,
        config_sha256=actual_sha256,
        file_name=config_path.name,
        model_type=model_type,
        architectures=tuple(architectures),
        total_num_kv_heads=kv_head_values[0],
    )


def _git_identity(code_root: Path) -> CapturedCodeIdentity:
    """Return the clean Git identity for the executing candidate checkout.

    :param code_root: Candidate Git worktree.
    :returns: Commit, tree, and clean-state proof.
    :raises LiveHandshakeError: If Git fails or tracked files are dirty.
    """

    def invoke(*arguments: str) -> str:
        try:
            result = subprocess.run(
                ["git", "-C", str(code_root), *arguments],
                check=True,
                capture_output=True,
                text=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError) as error:
            raise LiveHandshakeError(
                f"failed to authenticate Git identity at {code_root}"
            ) from error
        return result.stdout.strip()

    resolved_root = code_root.resolve()
    loaded_sources = (
        Path(__file__).resolve(),
        Path(inspect.getfile(NixlBaseConnectorWorker)).resolve(),
    )
    for source in loaded_sources:
        try:
            source.relative_to(resolved_root)
        except ValueError as error:
            raise LiveHandshakeError(
                f"loaded source {source} is outside candidate root {resolved_root}"
            ) from error

    status = invoke("status", "--porcelain", "--untracked-files=all")
    if len(status) > 0:
        raise LiveHandshakeError("capture code root is not a clean worktree")
    commit = invoke("rev-parse", "HEAD^{commit}")
    tree = invoke("rev-parse", "HEAD^{tree}")
    _require_git_object_id(commit, "Git commit")
    _require_git_object_id(tree, "Git tree")
    return CapturedCodeIdentity(commit=commit, tree=tree, worktree_clean=True)


def _decode_wire_payload(
    payload: bytes,
) -> tuple[NixlHandshakePayload, NixlAgentMetadata]:
    """Decode both typed connector-v9 MessagePack envelopes.

    :param payload: Exact scheduler response.
    :returns: Outer handshake and inner agent metadata.
    :raises LiveHandshakeError: If either typed payload is invalid.
    """
    try:
        handshake = msgspec.msgpack.decode(payload, type=NixlHandshakePayload)
        metadata = msgspec.msgpack.decode(
            handshake.agent_metadata_bytes,
            type=NixlAgentMetadata,
        )
    except (msgspec.DecodeError, msgspec.ValidationError) as error:
        raise LiveHandshakeError(
            "endpoint returned invalid connector-v9 metadata"
        ) from error
    return handshake, metadata


def _region_contract(metadata: NixlAgentMetadata) -> tuple[tuple[object, ...], ...]:
    return tuple(
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
        for region in metadata.regions
    )


def _normalized_contract_bytes(metadata: NixlAgentMetadata) -> bytes:
    """Serialize the same address-independent fields compared across P ranks.

    :param metadata: Decoded rank metadata.
    :returns: Canonical JSON contract bytes.
    """
    contract = {
        "engine_id": metadata.engine_id,
        "attn_backend_name": metadata.attn_backend_name,
        "ssm_sizes": metadata.ssm_sizes,
        "kv_cache_layout": metadata.kv_cache_layout,
        "block_size": metadata.block_size,
        "physical_blocks_per_logical_kv_block": (
            metadata.physical_blocks_per_logical_kv_block
        ),
        "block_lens": metadata.block_lens,
        "num_blocks": metadata.num_blocks,
        "source_group_planes": metadata.source_group_planes,
        "physical_group_token_capacities": (metadata.physical_group_token_capacities),
        "region_geometry": _region_contract(metadata),
    }
    return json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()


def _capture_wire_payload(rank: int, payload: bytes) -> CapturedWirePayload:
    handshake, metadata = _decode_wire_payload(payload)
    if metadata.tp_rank != rank:
        raise LiveHandshakeError(
            "wire metadata rank differs from the queried endpoint rank; "
            f"metadata={metadata.tp_rank}, queried={rank}"
        )
    return CapturedWirePayload(
        rank=rank,
        payload=payload,
        payload_sha256=_sha256(payload),
        compatibility_hash=handshake.compatibility_hash,
        engine_id=metadata.engine_id,
        normalized_contract_sha256=_sha256(_normalized_contract_bytes(metadata)),
        registration_generation_sha256=_sha256(
            metadata.registration_generation.encode()
        ),
    )


def _fetch_payload(host: str, port: int, rank: int, timeout_seconds: float) -> bytes:
    """Read one rank from the connector's ordinary metadata endpoint.

    The request is the same read-only ``GET_META_MSG`` used by decoder workers.
    It does not import an agent, register memory, or post a native operation.

    :param host: Scheduler side-channel host.
    :param port: Scheduler side-channel port.
    :param rank: Endpoint TP rank.
    :param timeout_seconds: Send and receive deadline.
    :returns: Exact scheduler response bytes.
    :raises LiveHandshakeError: If transport fails or returns an empty payload.
    """
    if len(host) == 0 or port <= 0 or port > 65535 or rank < 0:
        raise LiveHandshakeError("handshake endpoint identity is invalid")
    if timeout_seconds <= 0:
        raise LiveHandshakeError("handshake capture timeout must be positive")
    timeout_ms = max(1, int(timeout_seconds * 1000))
    context = Context()
    socket = context.socket(SocketType.REQ)
    socket.setsockopt(SocketOption.LINGER, 0)
    socket.setsockopt(SocketOption.SNDTIMEO, timeout_ms)
    socket.setsockopt(SocketOption.RCVTIMEO, timeout_ms)
    try:
        socket.connect(f"tcp://{host}:{port}")
        socket.send(msgspec.msgpack.encode((GET_META_MSG, rank)))
        payload = socket.recv()
    except ZMQError as error:
        raise LiveHandshakeError(
            f"handshake query failed for {host}:{port} rank {rank}"
        ) from error
    finally:
        socket.close()
        context.term()
    if len(payload) == 0:
        raise LiveHandshakeError(
            f"handshake endpoint returned an empty rank {rank} response"
        )
    return payload


def _metadata_by_rank(endpoint: CapturedEndpoint) -> dict[int, NixlAgentMetadata]:
    expected_ranks = tuple(range(endpoint.tensor_parallel_size))
    observed_ranks = tuple(payload.rank for payload in endpoint.payloads)
    if observed_ranks != expected_ranks:
        raise LiveHandshakeError(
            f"endpoint rank roster differs: expected={expected_ranks}, "
            f"observed={observed_ranks}"
        )
    result: dict[int, NixlAgentMetadata] = {}
    for captured in endpoint.payloads:
        if _sha256(captured.payload) != captured.payload_sha256:
            raise LiveHandshakeError(
                f"rank {captured.rank} wire-payload digest differs"
            )
        handshake, metadata = _decode_wire_payload(captured.payload)
        if metadata.tp_rank != captured.rank:
            raise LiveHandshakeError(
                f"rank {captured.rank} metadata rank differs from the endpoint"
            )
        if handshake.compatibility_hash != captured.compatibility_hash:
            raise LiveHandshakeError(
                f"rank {captured.rank} compatibility hash differs from its payload"
            )
        if (
            metadata.engine_id != endpoint.engine_id
            or metadata.engine_id != captured.engine_id
        ):
            raise LiveHandshakeError(
                f"rank {captured.rank} engine identity differs from the endpoint"
            )
        normalized_sha256 = _sha256(_normalized_contract_bytes(metadata))
        if normalized_sha256 != captured.normalized_contract_sha256:
            raise LiveHandshakeError(
                f"rank {captured.rank} normalized contract digest differs"
            )
        generation_sha256 = _sha256(metadata.registration_generation.encode())
        if generation_sha256 != captured.registration_generation_sha256:
            raise LiveHandshakeError(
                f"rank {captured.rank} registration-generation digest differs"
            )
        result[captured.rank] = metadata
    return result


def _expected_region_owners(config: RigConfig) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(
            group.index
            for group in config.groups
            if region_index in group.owned_region_indices
        )
        for region_index in range(len(config.regions))
    )


def _validate_metadata_geometry(
    *,
    config: RigConfig,
    metadata: NixlAgentMetadata,
    row_scale: int,
    expected_device: int,
) -> None:
    """Validate one live rank against exact production physical geometry.

    :param config: Authenticated rig contract.
    :param metadata: Decoded live rank metadata.
    :param row_scale: One for P and four for TP1 D.
    :param expected_device: Role-local CUDA device ordinal.
    :raises LiveHandshakeError: If any physical or semantic fact differs.
    """
    expected_rows = tuple(region.row_bytes * row_scale for region in config.regions)
    expected_planes = tuple(group.destination_plane_count for group in config.groups)
    expected_capacities = tuple(group.token_capacity for group in config.groups)
    expected_owners = _expected_region_owners(config)
    if metadata.device_id != expected_device:
        raise LiveHandshakeError(
            f"rank device {metadata.device_id} differs from expected {expected_device}"
        )
    if metadata.num_blocks != config.source_block_count:
        raise LiveHandshakeError(
            "live rank block count differs from production geometry"
        )
    if tuple(metadata.block_lens) != expected_rows:
        raise LiveHandshakeError(
            "live rank row lengths differ from production geometry"
        )
    if metadata.kv_cache_layout != "HND":
        raise LiveHandshakeError("live rank does not use the production HND layout")
    if metadata.block_size != 16:
        raise LiveHandshakeError("live rank physical block size differs from 16")
    if metadata.ssm_sizes != (0, 0):
        raise LiveHandshakeError("Gemma 4 live rank unexpectedly advertises SSM state")
    if metadata.attn_backend_name != _EXPECTED_ATTENTION_BACKEND:
        raise LiveHandshakeError("live rank attention backend differs from production")
    if metadata.physical_blocks_per_logical_kv_block != 1:
        raise LiveHandshakeError("live rank physical-to-logical block ratio differs")
    if metadata.source_group_planes != expected_planes:
        raise LiveHandshakeError("live rank group-plane contract differs")
    if metadata.physical_group_token_capacities != expected_capacities:
        raise LiveHandshakeError("live rank group token capacities differ")
    if len(metadata.registration_generation) == 0:
        raise LiveHandshakeError("live rank registration generation is empty")
    if len(metadata.agent_metadata) == 0:
        raise LiveHandshakeError("live rank opaque NIXL agent metadata is empty")
    if len(metadata.regions) != len(config.regions):
        raise LiveHandshakeError("live rank region cardinality differs")
    if metadata.kv_caches_base_addr != [
        region.base_address for region in metadata.regions
    ]:
        raise LiveHandshakeError("live rank base-address roster differs from regions")
    for region_index, (region, row_bytes, owners) in enumerate(
        zip(metadata.regions, expected_rows, expected_owners, strict=True)
    ):
        if region.group_indices != owners:
            raise LiveHandshakeError(
                f"live region {region_index} ownership differs from production"
            )
        if region.row_bytes != row_bytes:
            raise LiveHandshakeError(f"live region {region_index} row bytes differ")
        if region.registered_bytes != config.source_block_count * row_bytes:
            raise LiveHandshakeError(
                f"live region {region_index} registered length differs"
            )
        if region.base_address <= 0 or region.shape[0] != config.source_block_count:
            raise LiveHandshakeError(
                f"live region {region_index} address or row shape is invalid"
            )
        if len(region.semantic_name) == 0 or len(region.group_semantic_names) == 0:
            raise LiveHandshakeError(
                f"live region {region_index} semantic identity is incomplete"
            )


def _validate_live_contract(
    config: RigConfig,
    capture: LiveHandshakeCapture,
) -> tuple[dict[int, NixlAgentMetadata], NixlAgentMetadata, str]:
    if capture.schema_version != LIVE_HANDSHAKE_SCHEMA_VERSION:
        raise LiveHandshakeError(
            f"unsupported live-handshake schema {capture.schema_version}"
        )
    if capture.evidence_scope != LIVE_HANDSHAKE_EVIDENCE_SCOPE:
        raise LiveHandshakeError("capture evidence scope is not live connector-v9 wire")
    if (
        capture.connector_version != NIXL_CONNECTOR_VERSION
        or NIXL_CONNECTOR_VERSION != 9
    ):
        raise LiveHandshakeError(
            "capture connector version differs from loaded v9 code"
        )
    if capture.producer.tensor_parallel_size != len(config.producer_devices):
        raise LiveHandshakeError("producer TP size differs from the rig contract")
    if capture.decoder.tensor_parallel_size != 1:
        raise LiveHandshakeError("Target 1 decoder capture must be TP1")
    replay = capture.replay
    if replay.total_num_kv_heads != capture.model.total_num_kv_heads:
        raise LiveHandshakeError("replay KV-head count differs from model evidence")
    if replay.total_num_kv_heads < capture.producer.tensor_parallel_size:
        raise LiveHandshakeError("model KV heads cannot represent four unique P ranks")
    if replay.local_tensor_parallel_size != capture.decoder.tensor_parallel_size:
        raise LiveHandshakeError("replay local TP size differs from decoder endpoint")
    if replay.group_spec_kinds != ("attention",) * len(config.groups):
        raise LiveHandshakeError("Gemma 4 replay requires only attention cache groups")
    expected_replay_flags = (
        replay.hybrid_memory_allocator_required,
        replay.use_mla,
        replay.has_mamba,
        replay.use_host_buffer,
        replay.enable_permute_local_kv,
        replay.enable_prefix_caching,
    )
    if expected_replay_flags != (True, False, False, False, False, False):
        raise LiveHandshakeError("decoder replay flags differ from production")

    producer_metadata = _metadata_by_rank(capture.producer)
    decoder_metadata_by_rank = _metadata_by_rank(capture.decoder)
    decoder_metadata = decoder_metadata_by_rank[0]
    for rank, metadata in producer_metadata.items():
        _validate_metadata_geometry(
            config=config,
            metadata=metadata,
            row_scale=1,
            expected_device=rank,
        )
    _validate_metadata_geometry(
        config=config,
        metadata=decoder_metadata,
        row_scale=len(config.producer_devices),
        expected_device=0,
    )

    producer_contracts = {
        captured.normalized_contract_sha256 for captured in capture.producer.payloads
    }
    if len(producer_contracts) != 1:
        raise LiveHandshakeError("producer ranks have different normalized contracts")
    compatibility_hashes = {
        captured.compatibility_hash
        for endpoint in (capture.producer, capture.decoder)
        for captured in endpoint.payloads
    }
    if len(compatibility_hashes) != 1:
        raise LiveHandshakeError("live P and D compatibility hashes differ")

    for rank, producer in producer_metadata.items():
        for region_index, (source, destination) in enumerate(
            zip(producer.regions, decoder_metadata.regions, strict=True)
        ):
            source_identity = (
                source.semantic_name,
                source.group_indices,
                source.group_semantic_names,
                source.dtype,
                source.element_size_bytes,
            )
            destination_identity = (
                destination.semantic_name,
                destination.group_indices,
                destination.group_semantic_names,
                destination.dtype,
                destination.element_size_bytes,
            )
            if source_identity != destination_identity:
                raise LiveHandshakeError(
                    f"producer rank {rank} region {region_index} semantic identity "
                    "differs from the decoder"
                )
    return producer_metadata, decoder_metadata, next(iter(compatibility_hashes))


def _replay_production_validator(
    *,
    config: RigConfig,
    capture: LiveHandshakeCapture,
    producer_metadata: dict[int, NixlAgentMetadata],
    decoder_metadata: NixlAgentMetadata,
) -> None:
    """Invoke the production all-rank validator with captured decoder state.

    :param config: Authenticated rig contract.
    :param capture: Typed live capture.
    :param producer_metadata: Exact decoded P rank roster.
    :param decoder_metadata: Exact decoded D rank metadata.
    :raises LiveHandshakeError: If production validation rejects the capture.
    """
    worker = cast(
        NixlBaseConnectorWorker,
        object.__new__(NixlBaseConnectorWorker),
    )
    worker.engine_id = decoder_metadata.engine_id
    worker.tp_rank = 0
    worker.block_size = decoder_metadata.block_size
    worker.num_blocks = decoder_metadata.num_blocks
    worker.block_len_per_layer = list(decoder_metadata.block_lens)
    worker._region_descriptors = decoder_metadata.regions
    worker._region_is_mla = [False] * len(decoder_metadata.regions)
    worker._has_mamba = False
    worker.use_mla = False
    worker.use_host_buffer = False
    worker.kv_cache_layout = decoder_metadata.kv_cache_layout
    worker.host_buffer_kv_cache_layout = decoder_metadata.kv_cache_layout
    worker.backend_name = decoder_metadata.attn_backend_name
    worker.enable_permute_local_kv = False
    worker.enable_heterogeneous_attn_post_process = False
    worker._is_hma_required = True
    worker._physical_blocks_per_logical_kv_block = (
        decoder_metadata.physical_blocks_per_logical_kv_block
    )
    worker._sp_flags_cache = [
        plane_count == 1 for plane_count in decoder_metadata.source_group_planes
    ]
    worker.kv_transfer_config = cast(
        KVTransferConfig,
        SimpleNamespace(enable_permute_local_kv=False),
    )
    worker.vllm_config = cast(
        VllmConfig,
        SimpleNamespace(cache_config=SimpleNamespace(enable_prefix_caching=False)),
    )
    worker.kv_cache_config = cast(
        KVCacheConfig,
        SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(
                    kv_cache_spec=SimpleNamespace(
                        block_size=capacity,
                        kv_planes=plane_count,
                    )
                )
                for capacity, plane_count in zip(
                    decoder_metadata.physical_group_token_capacities,
                    decoder_metadata.source_group_planes,
                    strict=True,
                )
            ]
        ),
    )
    worker._group_spec_types = (FullAttentionSpec,) * len(config.groups)
    worker.transfer_topo = TransferTopology(
        tp_rank=0,
        tp_size=capture.replay.local_tensor_parallel_size,
        block_size=decoder_metadata.block_size,
        engine_id=decoder_metadata.engine_id,
        is_mla=False,
        is_mamba=False,
        total_num_kv_heads=capture.replay.total_num_kv_heads,
        attn_backends=[cast(type[AttentionBackend], _ReplayAttentionBackend)],
        tensor_shape=None,
    )
    worker.dst_num_blocks = {worker.engine_id: worker.num_blocks}
    worker._remote_agents = defaultdict(dict)
    worker._remote_regions = defaultdict(dict)
    worker._remote_layout = defaultdict(dict)
    worker._remote_source_semantics = defaultdict(dict)
    worker._remote_rank_contracts = defaultdict(dict)
    worker.tp_mappings = {}
    plan = compute_tp_mapping(
        worker.transfer_topo,
        capture.producer.tensor_parallel_size,
        worker._group_spec_types,
    )
    try:
        worker._validate_remote_handshake_roster(
            producer_metadata,
            capture.producer.tensor_parallel_size,
            plan,
        )
    except RuntimeError as error:
        raise LiveHandshakeError(
            "captured roster failed the production all-rank validator"
        ) from error


def _validate_capture(
    *,
    config_path: Path,
    config: RigConfig,
    code_root: Path,
    capture: LiveHandshakeCapture,
    expected_model_config_sha256: str,
) -> tuple[str, CapturedCodeIdentity]:
    rig_payload = config_path.read_bytes()
    if _sha256(rig_payload) != capture.rig.file_sha256:
        raise LiveHandshakeError("capture rig file digest differs")
    if config.fingerprint != capture.rig.fingerprint:
        raise LiveHandshakeError("capture typed rig fingerprint differs")
    model_path = Path(capture.model.file_name)
    replayed_model = _model_identity_from_payload(
        model_path,
        capture.model.config_payload,
        expected_model_config_sha256,
    )
    if replayed_model != capture.model:
        raise LiveHandshakeError("capture model identity differs from its payload")
    code = _git_identity(code_root)
    if code != capture.code:
        raise LiveHandshakeError("capture code identity differs from the checkout")
    producer_metadata, decoder_metadata, compatibility_hash = _validate_live_contract(
        config,
        capture,
    )
    _replay_production_validator(
        config=config,
        capture=capture,
        producer_metadata=producer_metadata,
        decoder_metadata=decoder_metadata,
    )
    return compatibility_hash, code


def capture_live_handshake(
    *,
    config_path: Path,
    config: RigConfig,
    code_root: Path,
    producer_host: str,
    producer_port: int,
    producer_engine_id: str | None,
    decoder_host: str,
    decoder_port: int,
    decoder_engine_id: str | None,
    model_config_path: Path,
    expected_model_config_sha256: str,
    output_path: Path,
    timeout_seconds: float,
) -> CaptureWriteResult:
    """Capture one live TP4 P endpoint and one exact TP1 D endpoint.

    :param config_path: Authenticated exact-2K rig configuration.
    :param config: Parsed rig configuration.
    :param code_root: Clean candidate Git checkout.
    :param producer_host: P scheduler side-channel host.
    :param producer_port: P scheduler side-channel port.
    :param producer_engine_id: Optional independently expected P engine identity.
    :param decoder_host: D scheduler side-channel host.
    :param decoder_port: D scheduler side-channel port.
    :param decoder_engine_id: Optional independently expected D engine identity.
    :param model_config_path: Authenticated deployed model ``config.json``.
    :param expected_model_config_sha256: Independently preserved model-config
        digest.
    :param output_path: Fresh immutable evidence file.
    :param timeout_seconds: Per-rank query deadline.
    :returns: Output path and externally sealable SHA-256.
    :raises LiveHandshakeError: If capture or immediate replay fails.
    """
    if producer_engine_id is not None and len(producer_engine_id) == 0:
        raise LiveHandshakeError("expected producer engine identity is empty")
    if decoder_engine_id is not None and len(decoder_engine_id) == 0:
        raise LiveHandshakeError("expected decoder engine identity is empty")
    model = _model_identity(model_config_path, expected_model_config_sha256)
    code = _git_identity(code_root)
    producer_payloads = tuple(
        _capture_wire_payload(
            rank,
            _fetch_payload(producer_host, producer_port, rank, timeout_seconds),
        )
        for rank in range(len(config.producer_devices))
    )
    decoder_payloads = (
        _capture_wire_payload(
            0,
            _fetch_payload(decoder_host, decoder_port, 0, timeout_seconds),
        ),
    )
    captured_producer_engine_ids = {payload.engine_id for payload in producer_payloads}
    if len(captured_producer_engine_ids) != 1:
        raise LiveHandshakeError("producer ranks report different engine identities")
    captured_producer_engine_id = next(iter(captured_producer_engine_ids))
    captured_decoder_engine_id = decoder_payloads[0].engine_id
    if (
        producer_engine_id is not None
        and producer_engine_id != captured_producer_engine_id
    ):
        raise LiveHandshakeError("captured producer engine identity differs")
    if (
        decoder_engine_id is not None
        and decoder_engine_id != captured_decoder_engine_id
    ):
        raise LiveHandshakeError("captured decoder engine identity differs")
    if captured_producer_engine_id == captured_decoder_engine_id:
        raise LiveHandshakeError("producer and decoder engine identities must differ")
    capture = LiveHandshakeCapture(
        schema_version=LIVE_HANDSHAKE_SCHEMA_VERSION,
        evidence_scope=LIVE_HANDSHAKE_EVIDENCE_SCOPE,
        connector_version=NIXL_CONNECTOR_VERSION,
        captured_at=datetime.now(UTC).isoformat(),
        code=code,
        rig=CapturedRigIdentity(
            file_sha256=_sha256(config_path.read_bytes()),
            fingerprint=config.fingerprint,
            file_name=config_path.name,
        ),
        model=model,
        replay=ValidatorReplayContract(
            total_num_kv_heads=model.total_num_kv_heads,
            local_tensor_parallel_size=1,
            group_spec_kinds=("attention",) * len(config.groups),
            hybrid_memory_allocator_required=True,
            use_mla=False,
            has_mamba=False,
            use_host_buffer=False,
            enable_permute_local_kv=False,
            enable_prefix_caching=False,
        ),
        producer=CapturedEndpoint(
            host=producer_host,
            port=producer_port,
            engine_id=captured_producer_engine_id,
            tensor_parallel_size=len(config.producer_devices),
            payloads=producer_payloads,
        ),
        decoder=CapturedEndpoint(
            host=decoder_host,
            port=decoder_port,
            engine_id=captured_decoder_engine_id,
            tensor_parallel_size=1,
            payloads=decoder_payloads,
        ),
    )
    _validate_capture(
        config_path=config_path,
        config=config,
        code_root=code_root,
        capture=capture,
        expected_model_config_sha256=expected_model_config_sha256,
    )
    encoded = msgspec.json.encode(capture)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("xb") as output:
            output.write(encoded)
    except FileExistsError as error:
        raise LiveHandshakeError(
            f"capture path already exists: {output_path}"
        ) from error
    output_path.chmod(0o444)
    return CaptureWriteResult(path=output_path, sha256=_sha256(encoded))


def verify_live_handshake_capture(
    *,
    capture_path: Path,
    expected_sha256: str,
    config_path: Path,
    config: RigConfig,
    code_root: Path,
    expected_model_config_sha256: str,
) -> CaptureVerificationResult:
    """Authenticate a preserved capture and replay the production validator.

    :param capture_path: Preserved live capture.
    :param expected_sha256: Digest from an independent evidence manifest.
    :param config_path: Authenticated exact-2K rig configuration.
    :param config: Parsed rig configuration.
    :param code_root: Exact candidate checkout bound by the capture.
    :param expected_model_config_sha256: Independently preserved model-config
        digest.
    :returns: Terminal authentication and validator verdict.
    :raises LiveHandshakeError: If any identity or contract differs.
    """
    _require_sha256(expected_sha256, "expected capture digest")
    encoded = capture_path.read_bytes()
    capture_sha256 = _sha256(encoded)
    if capture_sha256 != expected_sha256:
        raise LiveHandshakeError(
            f"capture SHA-256 {capture_sha256} differs from {expected_sha256}"
        )
    try:
        capture = msgspec.json.decode(encoded, type=LiveHandshakeCapture)
    except (msgspec.DecodeError, msgspec.ValidationError) as error:
        raise LiveHandshakeError("live handshake capture schema is invalid") from error
    compatibility_hash, code = _validate_capture(
        config_path=config_path,
        config=config,
        code_root=code_root,
        capture=capture,
        expected_model_config_sha256=expected_model_config_sha256,
    )
    return CaptureVerificationResult(
        capture_sha256=capture_sha256,
        code_commit=code.commit,
        code_tree=code.tree,
        compatibility_hash=compatibility_hash,
        producer_engine_id=capture.producer.engine_id,
        decoder_engine_id=capture.decoder.engine_id,
        producer_rank_count=len(capture.producer.payloads),
        validator_replay="passed_without_native_mutation",
    )


def encode_result(result: CaptureWriteResult | CaptureVerificationResult) -> str:
    """Encode one CLI result as stable readable JSON.

    :param result: Capture or verification result.
    :returns: Sorted indented JSON.
    """
    if isinstance(result, CaptureWriteResult):
        value: dict[str, object] = {
            "path": str(result.path),
            "sha256": result.sha256,
        }
    else:
        value = {
            "capture_sha256": result.capture_sha256,
            "code_commit": result.code_commit,
            "code_tree": result.code_tree,
            "compatibility_hash": result.compatibility_hash,
            "producer_engine_id": result.producer_engine_id,
            "decoder_engine_id": result.decoder_engine_id,
            "producer_rank_count": result.producer_rank_count,
            "validator_replay": result.validator_replay,
        }
    return json.dumps(value, indent=2, sort_keys=True)


def write_result_output(
    output_path: Path,
    result: CaptureWriteResult | CaptureVerificationResult,
) -> None:
    """Write one deterministic machine-readable result to a fresh artifact.

    Process logs may precede CLI output during vLLM imports. Production gates
    therefore consume this exclusive result file, never stdout.

    :param output_path: Fresh read-only result artifact.
    :param result: Capture or verification result.
    :raises LiveHandshakeError: If the result path already exists.
    """
    encoded = f"{encode_result(result)}\n".encode()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("xb") as output:
            output.write(encoded)
    except FileExistsError as error:
        raise LiveHandshakeError(
            f"result path already exists: {output_path}"
        ) from error
    output_path.chmod(0o444)
