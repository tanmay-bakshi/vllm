# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for authoritative P-to-D localization artifacts."""

import hashlib
import queue
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import msgspec
import pytest
import torch

from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.coalesced_layout import (
    GroupTransferRoster,
    RegionOwnership,
    build_coalesced_transfer_plan,
)
from vllm.distributed.kv_transfer.integrity import (
    IntegrityIdentity,
    IntegrityPayloadKind,
    IntegrityStage,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import (
    _CoalescedLocalizationPlan,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    RemoteMeta,
    ReqMeta,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_scheduler import (
    NixlPullConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_worker import (
    NixlPullConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import ReadSpec
from vllm.distributed.kv_transfer.nixl_contracts import (
    NixlRegionDescriptor,
    NixlSourceRoster,
)
from vllm.distributed.kv_transfer.nixl_localization import (
    LocalizationArtifactWriter,
    LocalizationError,
    LocalizationFingerprintAlgorithm,
    LocalizationMode,
    NixlCaptureRecord,
    NixlEventRecord,
    NixlIntegrityLeaf,
    NixlLocalizationConfig,
    NixlPlanPosition,
    NixlPlanRecord,
    NixlPlanRun,
    NixlRegionPlan,
    NixlSourceContract,
    NixlSourceManifest,
    NixlSourceManifestRecord,
    build_fingerprint_leaf,
    build_integrity_identity,
    build_integrity_leaf,
    compute_semantic_contract_digest,
    leaf_source_key,
    localization_child_index,
    localization_fingerprint_size,
    localization_producer_target,
    localization_request_target,
    localization_stage_barrier,
    seal_source_manifest,
    source_contract_from_manifest,
    validate_source_contract_structure,
    validate_source_manifest_structure,
)
from vllm.distributed.kv_transfer.nixl_localization_validator import (
    read_localization_artifact,
    validate_localization_artifacts,
    validate_localization_plan,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)

HTTP_TARGET_REQUEST_ID = "p2d-phase2db-separated-score2-20260713-p000-s000637"
TARGET_REQUEST_ID_BASE = f"chatcmpl-{HTTP_TARGET_REQUEST_ID}"


@pytest.mark.parametrize(
    ("stage", "barrier"),
    (
        (
            IntegrityStage.STAGING_RAW,
            "nixl_done_without_added_device_wide_sync",
        ),
        (
            IntegrityStage.STAGING_FENCED_CONTROL,
            "device_synchronize_observer_control_not_gdr_flush",
        ),
        (
            IntegrityStage.STAGING_POST_SCATTER,
            "scatter_completion_event_before_staging_release",
        ),
        (
            IntegrityStage.DESTINATION,
            "scatter_completion_event_before_publication",
        ),
        (
            IntegrityStage.PRE_READ,
            "after_transfer_phase_drain_before_model_forward",
        ),
    ),
)
def test_localization_stage_barrier_is_canonical(
    stage: IntegrityStage,
    barrier: str,
) -> None:
    """Each decoder capture stage must have one exact ordering contract."""
    assert localization_stage_barrier(stage) == barrier


@pytest.mark.parametrize(
    "stage",
    (IntegrityStage.SOURCE_PRE, IntegrityStage.SOURCE_POST),
)
def test_localization_stage_barrier_rejects_source_stages(
    stage: IntegrityStage,
) -> None:
    """Producer observations must not masquerade as decoder barriers."""
    with pytest.raises(ValueError, match="unsupported decoder localization stage"):
        localization_stage_barrier(stage)


@pytest.mark.cpu_test
def test_worker_construction_initializes_localization_model_topology(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Construct an enabled worker with model topology available to its writer."""
    monkeypatch.setenv("VLLM_NIXL_P2D_LOCALIZATION", "trace")
    monkeypatch.setenv("VLLM_NIXL_P2D_RUN_ID", "constructor-regression")
    monkeypatch.setenv("VLLM_NIXL_P2D_TRANSPORT_ARM", "unit-test")
    monkeypatch.setenv(
        "VLLM_NIXL_P2D_TARGET_REQUEST_IDS_JSON",
        msgspec.json.encode([TARGET_REQUEST_ID_BASE]).decode(),
    )
    monkeypatch.setenv("VLLM_NIXL_P2D_ARTIFACT_DIR", str(tmp_path))

    model_config = MagicMock()
    model_config.use_mla = False
    model_config.get_total_num_kv_heads.return_value = 8
    kv_transfer_config = KVTransferConfig(
        kv_connector="NixlConnector",
        kv_role="kv_consumer",
        kv_buffer_device="cuda",
    )
    vllm_config = MagicMock()
    vllm_config.model_config = model_config
    vllm_config.cache_config.block_size = 16
    vllm_config.kv_transfer_config = kv_transfer_config
    vllm_config.scheduler_config.disable_hybrid_kv_cache_manager = False
    vllm_config.parallel_config.tensor_parallel_size = 4

    kv_cache_config = KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer0"],
                FullAttentionSpec(
                    block_size=16,
                    num_kv_heads=2,
                    head_size=16,
                    dtype=torch.float16,
                ),
            )
        ],
    )
    platform = MagicMock()
    platform.device_type = "cuda"
    platform.get_nixl_memory_type.return_value = "VRAM"
    backend = MagicMock()
    backend.get_name.return_value = "TEST_ATTENTION"

    with (
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker.NixlWrapper",
            MagicMock,
        ),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker.nixl_agent_config",
            None,
        ),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker.current_platform",
            platform,
        ),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker.get_tensor_model_parallel_rank",
            return_value=2,
        ),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker.get_tensor_model_parallel_world_size",
            return_value=4,
        ),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker.get_current_attn_backends",
            return_value=[backend],
        ),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker.select_common_block_size",
            return_value=16,
        ),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker.get_kv_cache_layout",
            return_value="HND",
        ),
    ):
        worker = NixlPullConnectorWorker(
            vllm_config,
            "decoder",
            kv_cache_config,
        )

    try:
        assert worker.model_config is model_config
        assert worker._localization_writer is not None
        artifact_path = worker._localization_writer.path
    finally:
        worker.shutdown()

    artifact = read_localization_artifact(artifact_path)
    assert artifact.session.rank == 2
    assert artifact.session.world_size == 4
    assert artifact.session.total_num_kv_heads == 8


def _config(
    artifact_dir: Path,
    *,
    run_id: str = "run-localization",
    mode: LocalizationMode = LocalizationMode.TRACE,
    target_request_ids: tuple[str, ...] = (TARGET_REQUEST_ID_BASE,),
) -> NixlLocalizationConfig:
    return NixlLocalizationConfig(
        mode=mode,
        run_id=run_id,
        transport_arm="tcp-shm-cuda-copy",
        target_request_ids=target_request_ids,
        artifact_dir=artifact_dir,
        copy_chunk_bytes=64 * 1024 * 1024,
        strict_zero_byte=False,
    )


def _region(
    *,
    base_address: int = 0x100000,
    group_semantic_name: str = "model.layers.0:transfer_region_0",
    row_bytes: int = 8,
    shape: tuple[int, ...] | None = None,
    strides: tuple[int, ...] | None = None,
) -> NixlRegionDescriptor:
    region_shape = shape if shape is not None else (128, row_bytes)
    region_strides = strides if strides is not None else (row_bytes, 1)
    return NixlRegionDescriptor(
        semantic_name=group_semantic_name,
        group_indices=(0,),
        group_semantic_names=((0, group_semantic_name),),
        base_address=base_address,
        registered_bytes=region_shape[0] * row_bytes,
        row_bytes=row_bytes,
        shape=region_shape,
        strides=region_strides,
        dtype="torch.uint8",
        element_size_bytes=1,
        layout="HND",
    )


def _manifest(
    config: NixlLocalizationConfig,
    *,
    region: NixlRegionDescriptor | None = None,
    blocks: tuple[int, ...] = (10, 11),
    source_rank: int = 0,
    registration_generation: str = "registration-1",
    producer_request_id: str | None = None,
    source_planes: int = 2,
    expected_consumers: int = 1,
) -> NixlSourceManifest:
    source_region = region if region is not None else _region()
    request_id = (
        producer_request_id
        if producer_request_id is not None
        else config.target_request_ids[0]
    )
    contract_digest = compute_semantic_contract_digest(
        region=source_region,
        group_index=0,
        group_token_capacity=64,
        source_plane_contract=source_planes,
    )
    leaves: list[NixlIntegrityLeaf] = []
    for source_position, block_id in enumerate(blocks):
        payload = bytes(
            (block_id + offset) % 256 for offset in range(source_region.row_bytes)
        )
        commit_bytes = source_region.row_bytes // 2
        for plane_index, payload_kind, plane_payload in (
            (-1, IntegrityPayloadKind.WIRE, payload),
            (0, IntegrityPayloadKind.COMMIT, payload[:commit_bytes]),
            (1, IntegrityPayloadKind.COMMIT, payload[commit_bytes:]),
        ):
            identity = build_integrity_identity(
                config=config,
                producer_engine_id="prefill",
                producer_request_id=request_id,
                registration_generation=registration_generation,
                semantic_contract_digest=contract_digest,
                offer_generation=1,
                iteration=0,
                source_rank=source_rank,
                region_index=0,
                group_index=0,
                plane_index=plane_index,
                source_position=source_position,
                remote_block_id=block_id,
                valid_token_extent=100,
                group_token_capacity=64,
                payload_kind=payload_kind,
                byte_length=len(plane_payload),
            )
            if config.mode is LocalizationMode.FINGERPRINT:
                leaves.append(
                    build_fingerprint_leaf(
                        identity=identity,
                        fingerprint=hashlib.sha256(plane_payload).digest(),
                        local_block_id=None,
                        destination_half=None,
                        rank_slot=None,
                    )
                )
            else:
                leaves.append(
                    build_integrity_leaf(
                        identity=identity,
                        payload=plane_payload,
                        local_block_id=None,
                        destination_half=None,
                        rank_slot=None,
                    )
                )
    copied_bytes = len(blocks) * source_region.row_bytes
    if config.mode is LocalizationMode.FINGERPRINT:
        copied_bytes = len(leaves) * localization_fingerprint_size(
            config.fingerprint_algorithm
        )
    return seal_source_manifest(
        NixlSourceManifest(
            schema_version=IntegrityIdentity.SCHEMA_VERSION,
            fingerprint_algorithm=config.fingerprint_algorithm,
            run_id=config.run_id,
            transport_arm=config.transport_arm,
            producer_engine_id="prefill",
            producer_request_id=request_id,
            registration_generation=registration_generation,
            offer_generation=1,
            iteration=0,
            expected_consumers=expected_consumers,
            source_rank=source_rank,
            region_lengths=(source_region.row_bytes,),
            regions=(source_region,),
            source_group_planes=(source_planes,),
            valid_token_extent=100,
            group_token_capacities=(64,),
            block_ids=(blocks,),
            observer=True,
            copied_bytes=copied_bytes,
            hashed_bytes=len(blocks) * 2 * source_region.row_bytes,
            duration_ns=1000,
            manifest_digest=b"",
            leaves=tuple(leaves),
        )
    )


def _plan(
    contract: NixlSourceContract,
    *,
    source_contracts: tuple[NixlSourceContract, ...] | None = None,
    rank_slots: tuple[int, ...] | None = None,
    selected_remote: tuple[int, ...] | None = None,
    selected_local: tuple[int, ...] | None = None,
    destination_planes: int = 2,
    positions: tuple[NixlPlanPosition, ...] | None = None,
    local_region: NixlRegionDescriptor | None = None,
    child_request_id: str | None = None,
    observer_rank: int = 0,
    source_tp_size: int | None = None,
) -> NixlPlanRecord:
    contracts = source_contracts if source_contracts is not None else (contract,)
    slots = rank_slots if rank_slots is not None else tuple(range(len(contracts)))
    remote = selected_remote if selected_remote is not None else contract.block_ids[0]
    source_start = len(contract.block_ids[0]) - len(remote)
    if selected_local is None:
        if destination_planes == 2 or len(remote) == 0:
            local_count = len(remote)
        else:
            first_local_position = source_start // 2
            final_source_position = source_start + len(remote) - 1
            local_count = final_source_position // 2 - first_local_position + 1
        selected_local = tuple(100 + index for index in range(local_count))
    resolved_local_region = (
        local_region if local_region is not None else contract.regions[0]
    )
    resolved_source_tp_size = (
        source_tp_size
        if source_tp_size is not None
        else max(source_contract.source_rank for source_contract in contracts) + 1
    )
    layout = build_coalesced_transfer_plan(
        source_tp_size=resolved_source_tp_size,
        source_ranks=tuple(
            source_contract.source_rank for source_contract in contracts
        ),
        rank_slots=slots,
        groups=(
            GroupTransferRoster(
                group_index=0,
                source_position_start=source_start,
                destination_plane_count=destination_planes,
                local_block_ids=selected_local,
                remote_block_ids=remote,
            ),
        ),
        regions=(
            RegionOwnership(
                region_index=0,
                group_indices=contract.regions[0].group_indices,
                source_row_count=(
                    contract.regions[0].registered_bytes
                    // contract.regions[0].row_bytes
                ),
                destination_row_count=(
                    resolved_local_region.registered_bytes
                    // resolved_local_region.row_bytes
                ),
                row_bytes=contract.regions[0].row_bytes,
            ),
        ),
    )
    region_layout = layout.regions[0]
    canonical_positions = tuple(
        NixlPlanPosition(
            group_index=position.group_index,
            source_position=position.source_position,
            remote_block_id=position.remote_block_id,
            valid_token_extent=contract.valid_token_extent,
            group_token_capacity=contract.group_token_capacities[0],
            local_block_id=position.local_block_id,
            destination_half=position.destination_half,
        )
        for position in region_layout.positions
    )
    return NixlPlanRecord(
        record_type=NixlPlanRecord.RECORD_TYPE,
        schema_version=IntegrityIdentity.SCHEMA_VERSION,
        source_tp_size=resolved_source_tp_size,
        source_contracts=contracts,
        child_request_id=(
            child_request_id
            if child_request_id is not None
            else contract.producer_request_id
        ),
        observer_engine_id="decoder",
        observer_rank=observer_rank,
        rank_slots=slots,
        destination_group_planes=(destination_planes,),
        destination_group_token_capacities=contract.group_token_capacities,
        local_regions=(resolved_local_region,),
        untrimmed_local_groups=(selected_local,),
        skipped_groups=(),
        selected_remote_groups=(remote,),
        selected_local_groups=(selected_local,),
        region_plans=(
            NixlRegionPlan(
                region_index=0,
                offset_within_rank=region_layout.offset_within_rank,
                positions=positions if positions is not None else canonical_positions,
                runs=tuple(
                    NixlPlanRun(
                        remote_block_id=run.remote_block_id,
                        position_count=run.position_count,
                        position_start=run.position_start,
                    )
                    for run in region_layout.runs
                ),
            ),
        ),
        rank_stride_bytes=layout.rank_stride_bytes,
        layout_digest=layout.digest,
        staging_offset=0,
        staging_size=layout.staging_size_bytes,
    )


def _retained_localization_plan(
    contract: NixlSourceContract,
) -> _CoalescedLocalizationPlan:
    """Build the typed runtime plan retained until the target model forward."""
    groups: list[GroupTransferRoster] = []
    for group_index, remote_block_ids in enumerate(contract.block_ids):
        destination_plane_count = contract.source_group_planes[group_index]
        local_count = (
            len(remote_block_ids)
            if destination_plane_count == 2
            else (len(remote_block_ids) + 1) // 2
        )
        groups.append(
            GroupTransferRoster(
                group_index=group_index,
                source_position_start=0,
                destination_plane_count=destination_plane_count,
                local_block_ids=tuple(range(local_count)),
                remote_block_ids=remote_block_ids,
            )
        )
    regions = tuple(
        RegionOwnership(
            region_index=region_index,
            group_indices=region.group_indices,
            source_row_count=region.shape[0],
            destination_row_count=region.shape[0],
            row_bytes=region.row_bytes,
        )
        for region_index, region in enumerate(contract.regions)
    )
    layout = build_coalesced_transfer_plan(
        source_tp_size=max(contract.source_rank + 1, 1),
        source_ranks=(contract.source_rank,),
        rank_slots=(0,),
        groups=tuple(groups),
        regions=regions,
    )
    return _CoalescedLocalizationPlan(
        layout=layout,
        source_contracts=(contract,),
    )


def _runtime_region(
    region_index: int,
    group_index: int,
    row_bytes: int,
) -> NixlRegionDescriptor:
    """Build one ownership-specific runtime capture region."""
    semantic_name = f"group-{group_index}:transfer-region-{region_index}"
    return NixlRegionDescriptor(
        semantic_name=semantic_name,
        group_indices=(group_index,),
        group_semantic_names=((group_index, semantic_name),),
        base_address=0x100000 + region_index * 0x10000,
        registered_bytes=128 * row_bytes,
        row_bytes=row_bytes,
        shape=(128, row_bytes),
        strides=(row_bytes, 1),
        dtype="torch.uint8",
        element_size_bytes=1,
        layout="HND",
    )


def _runtime_contract(
    config: NixlLocalizationConfig,
    source_rank: int,
    regions: tuple[NixlRegionDescriptor, ...],
) -> NixlSourceContract:
    """Build one complete two-group runtime localization contract."""
    return NixlSourceContract(
        schema_version=IntegrityIdentity.SCHEMA_VERSION,
        fingerprint_algorithm=config.fingerprint_algorithm,
        run_id=config.run_id,
        transport_arm=config.transport_arm,
        producer_engine_id="prefill",
        producer_request_id=TARGET_REQUEST_ID_BASE,
        registration_generation=f"registration-{source_rank}",
        offer_generation=7,
        iteration=3,
        expected_consumers=1,
        source_rank=source_rank,
        region_lengths=tuple(region.row_bytes for region in regions),
        regions=regions,
        source_group_planes=(2, 1),
        valid_token_extent=173,
        group_token_capacities=(64, 32),
        block_ids=((10, 11), (20, 21, 22, 23)),
    )


def _runtime_capture_worker(
    tmp_path: Path,
    mode: LocalizationMode = LocalizationMode.TRACE,
) -> tuple[
    NixlPullConnectorWorker,
    SimpleNamespace,
    _CoalescedLocalizationPlan,
]:
    """Build rank-major staging bytes for an ownership-pruned runtime plan."""
    config = _config(tmp_path, mode=mode)
    regions = (
        _runtime_region(0, 0, 8),
        _runtime_region(1, 1, 6),
    )
    contracts = tuple(
        _runtime_contract(config, source_rank, regions) for source_rank in (2, 0)
    )
    layout = build_coalesced_transfer_plan(
        source_tp_size=4,
        source_ranks=(2, 0),
        rank_slots=(1, 0),
        groups=(
            GroupTransferRoster(
                group_index=0,
                source_position_start=0,
                destination_plane_count=2,
                local_block_ids=(2, 3),
                remote_block_ids=(10, 11),
            ),
            GroupTransferRoster(
                group_index=1,
                source_position_start=1,
                destination_plane_count=1,
                local_block_ids=(4, 5),
                remote_block_ids=(21, 22, 23),
            ),
        ),
        regions=(
            RegionOwnership(
                region_index=0,
                group_indices=(0,),
                source_row_count=128,
                destination_row_count=8,
                row_bytes=8,
            ),
            RegionOwnership(
                region_index=1,
                group_indices=(1,),
                source_row_count=128,
                destination_row_count=8,
                row_bytes=6,
            ),
        ),
    )
    runtime_plan = _CoalescedLocalizationPlan(
        layout=layout,
        source_contracts=contracts,
    )
    ownership = SimpleNamespace(
        layout=layout,
        lease=SimpleNamespace(offset=11),
    )
    staging = torch.full(
        (ownership.lease.offset + layout.staging_size_bytes + 7,),
        0xEE,
        dtype=torch.uint8,
    )
    for rank_index in range(len(layout.source_ranks)):
        for region in layout.regions:
            row_bytes = region.ownership.row_bytes
            for position_index in range(len(region.positions)):
                row_start = (
                    ownership.lease.offset
                    + layout.region_offset(
                        rank_index,
                        region.ownership.region_index,
                    )
                    + position_index * row_bytes
                )
                value = rank_index * 64 + region.ownership.region_index * 16
                value += position_index * 4
                staging[row_start : row_start + row_bytes] = (
                    torch.arange(row_bytes, dtype=torch.uint8) + value
                )

    worker = object.__new__(NixlPullConnectorWorker)
    worker._localization_config = config
    worker._staging_buf = staging
    worker._coalesced_localization_plans = {TARGET_REQUEST_ID_BASE: runtime_plan}
    worker._localization_finish_capture = MagicMock()
    if mode is LocalizationMode.FINGERPRINT:
        fingerprinter = MagicMock()
        fingerprinter.fingerprint_rows.side_effect = lambda rows: rows.clone()
        fingerprinter.fingerprints_to_digests.side_effect = lambda batches: [
            hashlib.sha256(bytes(row.tolist())).digest()
            for batch in batches
            for row in batch
        ]
        worker._localization_fingerprinter = fingerprinter
    return worker, ownership, runtime_plan


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "mode",
    [LocalizationMode.TRACE, LocalizationMode.FINGERPRINT],
)
def test_runtime_capture_uses_canonical_rank_major_region_layout(
    tmp_path: Path,
    mode: LocalizationMode,
) -> None:
    """Staging and destination captures agree on every committed payload."""
    worker, ownership, runtime_plan = _runtime_capture_worker(tmp_path, mode)
    layout = runtime_plan.layout

    worker._localization_capture_staging(
        TARGET_REQUEST_ID_BASE,
        ownership,
        IntegrityStage.STAGING_RAW,
        "unit-test-staging-boundary",
    )

    staging_calls = worker._localization_finish_capture.call_args_list
    assert len(staging_calls) == len(layout.source_ranks)
    staging_leaves_by_rank = {
        call.kwargs["contract"].source_rank: {
            leaf_source_key(leaf): leaf for leaf in call.kwargs["leaves"]
        }
        for call in staging_calls
    }
    for source_rank, rank_slot in zip(
        layout.source_ranks,
        layout.rank_slots,
        strict=True,
    ):
        leaves = tuple(staging_leaves_by_rank[source_rank].values())
        assert len(leaves) == 8
        assert {leaf.rank_slot for leaf in leaves} == {rank_slot}
        single_plane = sorted(
            (
                leaf.source_position,
                leaf.remote_block_id,
                leaf.local_block_id,
                leaf.destination_half,
            )
            for leaf in leaves
            if leaf.region_index == 1
            and leaf.payload_kind is IntegrityPayloadKind.COMMIT
        )
        assert single_plane == [
            (1, 21, 4, 1),
            (2, 22, 5, 0),
            (3, 23, 5, 1),
        ]

    destination_rows = [
        torch.zeros(
            (
                region.ownership.destination_row_count,
                len(layout.source_ranks) * region.ownership.row_bytes,
            ),
            dtype=torch.uint8,
        )
        for region in layout.regions
    ]
    for rank_index, rank_slot in enumerate(layout.rank_slots):
        for region in layout.regions:
            region_index = region.ownership.region_index
            row_bytes = region.ownership.row_bytes
            chunk_bytes = row_bytes // 2
            source_start = ownership.lease.offset + layout.region_offset(
                rank_index,
                region_index,
            )
            source_rows = worker._staging_buf[
                source_start : source_start + region.size_bytes
            ].view(len(region.positions), row_bytes)
            for position_index, position in enumerate(region.positions):
                if position.destination_half < 0:
                    destination = destination_rows[region_index].view(
                        region.ownership.destination_row_count,
                        2,
                        len(layout.source_ranks),
                        chunk_bytes,
                    )
                    destination[
                        position.local_block_id,
                        :,
                        rank_slot,
                        :,
                    ] = source_rows[position_index].view(2, chunk_bytes)
                    continue
                destination = destination_rows[region_index].view(
                    region.ownership.destination_row_count,
                    len(layout.source_ranks),
                    2,
                    chunk_bytes,
                )
                destination[
                    position.local_block_id,
                    rank_slot,
                    position.destination_half,
                    :,
                ] = source_rows[position_index, :chunk_bytes]

    worker._region_rows = destination_rows
    worker._region_descriptors = tuple(
        contract_region for contract_region in runtime_plan.source_contracts[0].regions
    )
    worker._localization_finish_capture.reset_mock()
    worker._localization_capture_destination(
        TARGET_REQUEST_ID_BASE,
        runtime_plan,
        IntegrityStage.DESTINATION,
        "unit-test-scatter-boundary",
    )

    destination_calls = worker._localization_finish_capture.call_args_list
    assert len(destination_calls) == len(layout.source_ranks)
    for call in destination_calls:
        source_rank = call.kwargs["contract"].source_rank
        destination_leaves = call.kwargs["leaves"]
        assert len(destination_leaves) == 5
        staging_leaves = staging_leaves_by_rank[source_rank]
        for leaf in destination_leaves:
            staging_leaf = staging_leaves[leaf_source_key(leaf)]
            assert leaf.digest == staging_leaf.digest
            assert leaf.local_block_id == staging_leaf.local_block_id
            assert leaf.destination_half == staging_leaf.destination_half
            assert leaf.rank_slot == staging_leaf.rank_slot


@pytest.mark.cpu_test
def test_runtime_localization_install_rejects_contract_geometry_drift(
    tmp_path: Path,
) -> None:
    """The retained runtime contract cannot diverge from canonical geometry."""
    worker, _, runtime_plan = _runtime_capture_worker(tmp_path)
    worker._coalesced_localization_plans = {}

    worker._install_coalesced_localization_plan(
        TARGET_REQUEST_ID_BASE,
        runtime_plan.layout,
        runtime_plan.source_contracts,
    )
    assert worker._coalesced_localization_plans == {
        TARGET_REQUEST_ID_BASE: runtime_plan
    }

    worker._coalesced_localization_plans = {}
    mismatched_regions = (
        runtime_plan.source_contracts[0].regions[0],
        _runtime_region(1, 1, 8),
    )
    mismatched_contracts = tuple(
        _runtime_contract(
            worker._localization_config,
            source_rank,
            mismatched_regions,
        )
        for source_rank in runtime_plan.layout.source_ranks
    )
    with pytest.raises(
        LocalizationError,
        match="region contract differs from canonical layout",
    ):
        worker._install_coalesced_localization_plan(
            TARGET_REQUEST_ID_BASE,
            runtime_plan.layout,
            mismatched_contracts,
        )
    assert worker._coalesced_localization_plans == {}


def _plan_from_layout(
    contract: NixlSourceContract,
    local_regions: tuple[NixlRegionDescriptor, ...],
    groups: tuple[GroupTransferRoster, ...],
) -> NixlPlanRecord:
    """Encode one canonical transport layout as a diagnostic plan record."""
    ownership = tuple(
        RegionOwnership(
            region_index=region_index,
            group_indices=source_region.group_indices,
            source_row_count=(
                source_region.registered_bytes // source_region.row_bytes
            ),
            destination_row_count=(
                local_region.registered_bytes // local_region.row_bytes
            ),
            row_bytes=source_region.row_bytes,
        )
        for region_index, (source_region, local_region) in enumerate(
            zip(contract.regions, local_regions, strict=True)
        )
    )
    layout = build_coalesced_transfer_plan(
        source_tp_size=1,
        source_ranks=(contract.source_rank,),
        rank_slots=(0,),
        groups=groups,
        regions=ownership,
    )
    return NixlPlanRecord(
        record_type=NixlPlanRecord.RECORD_TYPE,
        schema_version=IntegrityIdentity.SCHEMA_VERSION,
        source_tp_size=layout.source_tp_size,
        source_contracts=(contract,),
        child_request_id=contract.producer_request_id,
        observer_engine_id="decoder",
        observer_rank=0,
        rank_slots=layout.rank_slots,
        destination_group_planes=tuple(
            group.destination_plane_count for group in groups
        ),
        destination_group_token_capacities=contract.group_token_capacities,
        local_regions=local_regions,
        untrimmed_local_groups=tuple(group.local_block_ids for group in groups),
        skipped_groups=(),
        selected_remote_groups=tuple(group.remote_block_ids for group in groups),
        selected_local_groups=tuple(group.local_block_ids for group in groups),
        region_plans=tuple(
            NixlRegionPlan(
                region_index=region.ownership.region_index,
                offset_within_rank=region.offset_within_rank,
                positions=tuple(
                    NixlPlanPosition(
                        group_index=position.group_index,
                        source_position=position.source_position,
                        remote_block_id=position.remote_block_id,
                        valid_token_extent=contract.valid_token_extent,
                        group_token_capacity=(
                            contract.group_token_capacities[position.group_index]
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
            for region in layout.regions
        ),
        rank_stride_bytes=layout.rank_stride_bytes,
        layout_digest=layout.digest,
        staging_offset=0,
        staging_size=layout.staging_size_bytes,
    )


def _mapped_wire_leaf(
    source_leaf: NixlIntegrityLeaf,
    position: NixlPlanPosition,
    rank_slot: int,
) -> NixlIntegrityLeaf:
    return NixlIntegrityLeaf(
        region_index=source_leaf.region_index,
        group_index=source_leaf.group_index,
        source_position=source_leaf.source_position,
        remote_block_id=source_leaf.remote_block_id,
        valid_token_extent=source_leaf.valid_token_extent,
        group_token_capacity=source_leaf.group_token_capacity,
        semantic_contract_digest=source_leaf.semantic_contract_digest,
        local_block_id=position.local_block_id,
        plane_index=source_leaf.plane_index,
        destination_half=position.destination_half,
        rank_slot=rank_slot,
        payload_kind=source_leaf.payload_kind,
        byte_length=source_leaf.byte_length,
        digest=source_leaf.digest,
    )


def _capture(
    manifest: NixlSourceManifest,
    plan: NixlPlanRecord,
    stage: IntegrityStage,
    *,
    corrupt: bool = False,
) -> NixlCaptureRecord:
    source_wires = {
        (
            leaf.region_index,
            leaf.group_index,
            leaf.source_position,
            leaf.remote_block_id,
        ): leaf
        for leaf in manifest.leaves
        if leaf.payload_kind is IntegrityPayloadKind.WIRE
    }
    source_ranks = tuple(contract.source_rank for contract in plan.source_contracts)
    rank_slot = plan.rank_slots[source_ranks.index(manifest.source_rank)]
    leaves = tuple(
        _mapped_wire_leaf(
            source_wires[
                (
                    region_plan.region_index,
                    position.group_index,
                    position.source_position,
                    position.remote_block_id,
                )
            ],
            position,
            rank_slot,
        )
        for region_plan in plan.region_plans
        for position in region_plan.positions
    )
    if corrupt:
        first = leaves[0]
        leaves = (
            NixlIntegrityLeaf(
                region_index=first.region_index,
                group_index=first.group_index,
                source_position=first.source_position,
                remote_block_id=first.remote_block_id,
                valid_token_extent=first.valid_token_extent,
                group_token_capacity=first.group_token_capacity,
                semantic_contract_digest=first.semantic_contract_digest,
                local_block_id=first.local_block_id,
                plane_index=first.plane_index,
                destination_half=first.destination_half,
                rank_slot=first.rank_slot,
                payload_kind=first.payload_kind,
                byte_length=first.byte_length,
                digest=bytes((first.digest[0] ^ 1,)) + first.digest[1:],
            ),
            *leaves[1:],
        )
    copied_bytes = sum(
        len(region_plan.positions) * manifest.region_lengths[region_plan.region_index]
        for region_plan in plan.region_plans
    )
    hashed_bytes = copied_bytes
    if (
        manifest.fingerprint_algorithm
        is LocalizationFingerprintAlgorithm.POSITION_WEIGHTED_WORDS_256_V1
    ):
        copied_bytes = len(leaves) * localization_fingerprint_size(
            manifest.fingerprint_algorithm
        )
    return NixlCaptureRecord(
        record_type=NixlCaptureRecord.RECORD_TYPE,
        schema_version=IntegrityIdentity.SCHEMA_VERSION,
        fingerprint_algorithm=manifest.fingerprint_algorithm,
        stage=stage,
        run_id=manifest.run_id,
        transport_arm=manifest.transport_arm,
        producer_engine_id=manifest.producer_engine_id,
        producer_request_id=manifest.producer_request_id,
        registration_generation=manifest.registration_generation,
        offer_generation=manifest.offer_generation,
        iteration=manifest.iteration,
        child_request_id=plan.child_request_id,
        source_rank=manifest.source_rank,
        observer_engine_id=plan.observer_engine_id,
        observer_rank=plan.observer_rank,
        observer=True,
        copied_bytes=copied_bytes,
        hashed_bytes=hashed_bytes,
        duration_ns=1000,
        barrier=localization_stage_barrier(stage),
        leaves=leaves,
    )


def _write_trace(
    artifact_dir: Path,
    *,
    corrupt_stage: IntegrityStage | None = None,
    corrupt_barrier_stage: IntegrityStage | None = None,
    include_event: bool = True,
    producer_request_id: str | None = None,
    child_request_id: str | None = None,
    decoder_world_size: int = 1,
    rank_slots: tuple[int, ...] | None = None,
    complete_decoder_world: bool = False,
    source_block_rosters: dict[int, tuple[int, ...]] | None = None,
    mode: LocalizationMode = LocalizationMode.TRACE,
    target_request_ids: tuple[str, ...] = (TARGET_REQUEST_ID_BASE,),
    expected_consumers: int = 1,
    recorded_source_tp_size: int | None = None,
) -> tuple[tuple[Path, ...], NixlSourceManifest, NixlPlanRecord]:
    config = _config(
        artifact_dir,
        mode=mode,
        target_request_ids=target_request_ids,
    )
    source_world_size = decoder_world_size * 2
    manifests = tuple(
        _manifest(
            config,
            region=_region(base_address=0x100000 + source_rank * 0x10000),
            blocks=(
                source_block_rosters[source_rank]
                if source_block_rosters is not None
                and source_rank in source_block_rosters
                else (10, 11)
            ),
            source_rank=source_rank,
            registration_generation=f"registration-{source_rank}",
            producer_request_id=producer_request_id,
            expected_consumers=expected_consumers,
        )
        for source_rank in range(source_world_size)
    )
    source_writers: list[LocalizationArtifactWriter] = []
    for source_rank, manifest in enumerate(manifests):
        source_writer = LocalizationArtifactWriter(
            config,
            "prefill",
            source_rank,
            source_world_size,
            8,
        )
        source_writer.write(
            NixlSourceManifestRecord(
                record_type=NixlSourceManifestRecord.RECORD_TYPE,
                stage=IntegrityStage.SOURCE_POST,
                manifest=manifest,
            )
        )
        source_writer.close()
        source_writers.append(source_writer)

    decoder_ranks = range(decoder_world_size) if complete_decoder_world else range(1)
    decoder_writers: list[LocalizationArtifactWriter] = []
    plans: list[NixlPlanRecord] = []
    for decoder_rank in decoder_ranks:
        source_start = decoder_rank * 2
        participating_manifests = manifests[source_start : source_start + 2]
        contracts = tuple(
            source_contract_from_manifest(manifest)
            for manifest in participating_manifests
        )
        plan = _plan(
            contracts[0],
            source_contracts=contracts,
            rank_slots=rank_slots,
            local_region=_region(
                base_address=0x200000 + decoder_rank * 0x10000,
                row_bytes=16,
            ),
            child_request_id=child_request_id,
            observer_rank=decoder_rank,
            source_tp_size=(
                recorded_source_tp_size
                if recorded_source_tp_size is not None
                else source_world_size
            ),
        )
        plans.append(plan)

        decoder_writer = LocalizationArtifactWriter(
            config,
            "decoder",
            decoder_rank,
            decoder_world_size,
            8,
        )
        decoder_writer.write(plan)
        capture_stages = (
            (
                IntegrityStage.STAGING_POST_SCATTER,
                IntegrityStage.DESTINATION,
                IntegrityStage.PRE_READ,
            )
            if mode is LocalizationMode.FINGERPRINT
            else (
                IntegrityStage.STAGING_RAW,
                IntegrityStage.STAGING_FENCED_CONTROL,
                IntegrityStage.DESTINATION,
                IntegrityStage.PRE_READ,
            )
        )
        for stage in capture_stages:
            for manifest in participating_manifests:
                capture = _capture(
                    manifest,
                    plan,
                    stage,
                    corrupt=(manifest.source_rank == 0 and stage is corrupt_stage),
                )
                if manifest.source_rank == 0 and stage is corrupt_barrier_stage:
                    capture = msgspec.structs.replace(
                        capture,
                        barrier="stale_runtime_ordering_claim",
                    )
                decoder_writer.write(capture)
        if include_event:
            manifest = participating_manifests[0]
            decoder_writer.write(
                NixlEventRecord(
                    record_type=NixlEventRecord.RECORD_TYPE,
                    schema_version=IntegrityIdentity.SCHEMA_VERSION,
                    run_id=config.run_id,
                    transport_arm=config.transport_arm,
                    code="CAPTURE_COMPLETE",
                    evidentiary=False,
                    producer_engine_id=manifest.producer_engine_id,
                    producer_request_id=manifest.producer_request_id,
                    child_request_id=plan.child_request_id,
                    observer_engine_id=plan.observer_engine_id,
                    observer_rank=plan.observer_rank,
                    detail=(
                        "all decoder stages captured; offline source comparison pending"
                    ),
                    created_ns=time.time_ns(),
                )
            )
        decoder_writer.close()
        decoder_writers.append(decoder_writer)
    paths = tuple(writer.path for writer in source_writers) + (
        tuple(writer.path for writer in decoder_writers)
    )
    return paths, manifests[0], plans[0]


@pytest.mark.cpu_test
def test_config_requires_and_selects_exact_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_NIXL_P2D_LOCALIZATION", "trace")
    monkeypatch.setenv("VLLM_NIXL_P2D_RUN_ID", "targeted-run")
    monkeypatch.setenv("VLLM_NIXL_P2D_TRANSPORT_ARM", "cuda-copy")
    monkeypatch.setenv("VLLM_NIXL_P2D_ARTIFACT_DIR", str(tmp_path))
    monkeypatch.delenv("VLLM_NIXL_P2D_TARGET_REQUEST_IDS_JSON", raising=False)

    with pytest.raises(ValueError, match="TARGET_REQUEST_IDS_JSON is required"):
        NixlLocalizationConfig.from_environment()

    monkeypatch.setenv(
        "VLLM_NIXL_P2D_TARGET_REQUEST_IDS_JSON",
        msgspec.json.encode(
            [TARGET_REQUEST_ID_BASE, "chatcmpl-second-target"]
        ).decode(),
    )
    config = NixlLocalizationConfig.from_environment()

    assert config.enabled
    assert config.target_request_ids == (
        TARGET_REQUEST_ID_BASE,
        "chatcmpl-second-target",
    )
    assert config.enabled_for(TARGET_REQUEST_ID_BASE)
    assert config.enabled_for(f"{TARGET_REQUEST_ID_BASE}-deadbeef")
    assert config.enabled_for_producer(TARGET_REQUEST_ID_BASE)
    assert config.enabled_for_producer(f"{TARGET_REQUEST_ID_BASE}-deadbeef")
    assert all(
        config.enabled_for(f"{choice}_{TARGET_REQUEST_ID_BASE}-deadbeef")
        for choice in range(8)
    )
    assert config.enabled_for("7_chatcmpl-second-target-deadbeef")
    assert config.enabled_for_producer(f"4_{TARGET_REQUEST_ID_BASE}-deadbeef") is False
    assert (
        localization_request_target(
            f"4_{TARGET_REQUEST_ID_BASE}-deadbeef",
            config.target_request_ids,
        )
        == TARGET_REQUEST_ID_BASE
    )
    assert (
        localization_producer_target(
            f"4_{TARGET_REQUEST_ID_BASE}-deadbeef",
            config.target_request_ids,
        )
        is None
    )
    assert (
        localization_child_index(
            f"4_{TARGET_REQUEST_ID_BASE}-deadbeef",
            TARGET_REQUEST_ID_BASE,
        )
        == 4
    )
    assert config.enabled_for(f"{TARGET_REQUEST_ID_BASE}-DEADBEEF") is False
    assert config.enabled_for(f"{TARGET_REQUEST_ID_BASE}-r1-deadbeef") is False
    assert config.enabled_for(f"{TARGET_REQUEST_ID_BASE}-deadbee") is False
    assert config.enabled_for(f"{TARGET_REQUEST_ID_BASE}-deadbeef0") is False
    assert config.enabled_for(f"4_5_{TARGET_REQUEST_ID_BASE}-deadbeef") is False
    assert config.enabled_for(HTTP_TARGET_REQUEST_ID) is False
    direct_target_config = _config(
        tmp_path,
        target_request_ids=("0_chatcmpl-direct-target",),
    )
    assert (
        localization_request_target(
            "0_chatcmpl-direct-target-deadbeef",
            direct_target_config.target_request_ids,
        )
        == "0_chatcmpl-direct-target"
    )
    with pytest.raises(ValueError, match="must not include a random suffix"):
        _config(
            tmp_path,
            target_request_ids=(f"{TARGET_REQUEST_ID_BASE}-deadbeef",),
        )
    with pytest.raises(ValueError, match="unique and sorted"):
        _config(
            tmp_path,
            target_request_ids=("chatcmpl-z", "chatcmpl-a"),
        )
    with pytest.raises(ValueError, match="ambiguous parent/child"):
        _config(
            tmp_path,
            target_request_ids=("4_chatcmpl-parent", "chatcmpl-parent"),
        )


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "target_request_ids_json",
    (
        "{}",
        "[]",
        '["chatcmpl-target",1]',
        '["chatcmpl-z","chatcmpl-a"]',
        '["chatcmpl-target","chatcmpl-target"]',
        '["chatcmpl-target-deadbeef"]',
    ),
)
def test_config_rejects_invalid_target_allowlists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_request_ids_json: str,
) -> None:
    """The environment cannot weaken or ambiguously encode request scope."""
    monkeypatch.setenv("VLLM_NIXL_P2D_LOCALIZATION", "trace")
    monkeypatch.setenv("VLLM_NIXL_P2D_RUN_ID", "targeted-run")
    monkeypatch.setenv("VLLM_NIXL_P2D_TRANSPORT_ARM", "cuda-copy")
    monkeypatch.setenv("VLLM_NIXL_P2D_ARTIFACT_DIR", str(tmp_path))
    monkeypatch.setenv(
        "VLLM_NIXL_P2D_TARGET_REQUEST_IDS_JSON",
        target_request_ids_json,
    )

    with pytest.raises(ValueError):
        NixlLocalizationConfig.from_environment()


@pytest.mark.cpu_test
def test_normal_consumer_metadata_names_only_the_actual_forward() -> None:
    """Transfer admission and model-forward membership remain distinct."""
    child_request_id = f"{TARGET_REQUEST_ID_BASE}-22222222"
    scheduler = object.__new__(NixlPullConnectorScheduler)
    scheduler._is_hma_required = False
    scheduler.is_bidirectional_kv_xfer_enabled = False
    scheduler.use_host_buffer = False
    scheduler._reqs_need_recv = {}
    scheduler._reqs_need_save = {}
    scheduler._reqs_need_send = {}
    scheduler._source_rosters = {}
    scheduler._source_offer_generation = 0
    scheduler._active_source_offer_generations = {}
    scheduler._sparse_retired_source_generations = set()
    scheduler._source_retired_through = 0
    scheduler._reqs_in_batch = set()
    scheduler._reqs_not_processed = set()
    scheduler._audit_finished_reqs = set()
    scheduler._heartbeat_by_engine = {}
    scheduler._heartbeat_snapshot_dirty = False
    scheduler._parallel_pull_flights = {}
    scheduler._offer_cancellation_queue = queue.Queue()

    request = MagicMock()
    request.request_id = child_request_id
    request.kv_transfer_params = {
        "do_remote_prefill": True,
        "do_remote_decode": False,
        "remote_block_ids": ((10, 11),),
        "remote_engine_id": "prefill",
        "remote_request_id": f"{TARGET_REQUEST_ID_BASE}-11111111",
        "remote_host": "127.0.0.1",
        "remote_port": 5601,
        "remote_num_tokens": 128,
        "tp_size": 4,
    }
    blocks = MagicMock()
    blocks.get_unhashed_block_ids_all_groups.return_value = ((100, 101),)
    scheduler.update_state_after_alloc(request, blocks, num_external_tokens=128)

    admission_output = MagicMock()
    admission_output.num_scheduled_tokens = {}
    admission = scheduler.build_connector_meta(admission_output)

    assert child_request_id in admission.reqs_to_recv
    assert admission.scheduled_request_ids == set()
    assert admission.reqs_in_batch == set()

    forward_output = MagicMock()
    forward_output.num_scheduled_tokens = {child_request_id: 1}
    forward = scheduler.build_connector_meta(forward_output)

    assert forward.reqs_to_recv == {}
    assert forward.scheduled_request_ids == {child_request_id}
    assert forward.reqs_in_batch == set()


@pytest.mark.cpu_test
def test_pre_read_waits_for_the_target_forward(tmp_path: Path) -> None:
    """Unrelated forwards preserve a plan; its target consumes it exactly once."""
    child_request_id = f"{TARGET_REQUEST_ID_BASE}-22222222"
    producer_request_id = f"{TARGET_REQUEST_ID_BASE}-11111111"
    manifest = _manifest(
        _config(tmp_path),
        producer_request_id=producer_request_id,
    )
    contract = source_contract_from_manifest(manifest)
    plan = _retained_localization_plan(contract)
    worker = object.__new__(NixlPullConnectorWorker)
    worker._localization_config = _config(tmp_path)
    worker._localization_pre_read_plans = {child_request_id: plan}
    capture = MagicMock()
    record = MagicMock()
    worker._localization_capture_destination = capture
    worker._localization_record_event = record

    worker._localization_capture_pre_read({"chatcmpl-unrelated"})

    assert worker._localization_pre_read_plans == {child_request_id: plan}
    capture.assert_not_called()
    record.assert_not_called()

    worker._localization_capture_pre_read({child_request_id})

    capture.assert_called_once_with(
        child_request_id,
        plan,
        IntegrityStage.PRE_READ,
        "after_transfer_phase_drain_before_model_forward",
    )
    record.assert_called_once_with(
        code="CAPTURE_COMPLETE",
        evidentiary=False,
        child_request_id=child_request_id,
        producer_engine_id="prefill",
        producer_request_id=producer_request_id,
        detail="all decoder stages captured; offline source comparison pending",
    )
    assert worker._localization_pre_read_plans == {}


@pytest.mark.cpu_test
def test_pre_read_captures_all_scheduled_parallel_children(tmp_path: Path) -> None:
    """One model batch captures every scheduled target and preserves the rest."""
    first_child_request_id = f"3_{TARGET_REQUEST_ID_BASE}-22222222"
    second_child_request_id = f"4_{TARGET_REQUEST_ID_BASE}-33333333"
    waiting_child_request_id = f"5_{TARGET_REQUEST_ID_BASE}-44444444"
    producer_request_id = f"{TARGET_REQUEST_ID_BASE}-11111111"
    manifest = _manifest(
        _config(tmp_path),
        producer_request_id=producer_request_id,
    )
    contract = source_contract_from_manifest(manifest)
    first_plan = _retained_localization_plan(contract)
    second_plan = _retained_localization_plan(contract)
    waiting_plan = _retained_localization_plan(contract)
    worker = object.__new__(NixlPullConnectorWorker)
    worker._localization_config = _config(tmp_path)
    worker._localization_pre_read_plans = {
        first_child_request_id: first_plan,
        second_child_request_id: second_plan,
        waiting_child_request_id: waiting_plan,
    }
    capture = MagicMock()
    record = MagicMock()
    worker._localization_capture_destination = capture
    worker._localization_record_event = record

    worker._localization_capture_pre_read(
        {first_child_request_id, second_child_request_id}
    )

    assert capture.call_count == 2
    capture.assert_any_call(
        first_child_request_id,
        first_plan,
        IntegrityStage.PRE_READ,
        "after_transfer_phase_drain_before_model_forward",
    )
    capture.assert_any_call(
        second_child_request_id,
        second_plan,
        IntegrityStage.PRE_READ,
        "after_transfer_phase_drain_before_model_forward",
    )
    assert record.call_count == 2
    assert worker._localization_pre_read_plans == {
        waiting_child_request_id: waiting_plan
    }


def _contract_worker(
    artifact_dir: Path,
) -> tuple[
    NixlPullConnectorWorker,
    str,
    ReqMeta,
    list[ReadSpec],
    NixlSourceContract,
]:
    """Build one structurally complete decoder contract fixture.

    :param artifact_dir: Localization artifact directory.
    :returns: Worker, child id, transfer metadata, read specs, and contract.
    """
    config = _config(artifact_dir)
    producer_request_id = f"{TARGET_REQUEST_ID_BASE}-11111111"
    child_request_id = f"{TARGET_REQUEST_ID_BASE}-22222222"
    manifest = _manifest(config, producer_request_id=producer_request_id)
    contract = source_contract_from_manifest(manifest)
    worker = object.__new__(NixlPullConnectorWorker)
    worker._localization_config = config
    worker._region_descriptors = manifest.regions
    worker._sp_group_flags = MagicMock(return_value=[False])
    worker._physical_group_token_capacities = MagicMock(
        return_value=manifest.group_token_capacities
    )
    worker._remote_layout = {"prefill": {0: ([8], manifest.regions[0].shape[0], 0)}}
    worker._remote_regions = {"prefill": {0: manifest.regions}}
    worker._remote_registration_generations = {
        "prefill": {0: manifest.registration_generation}
    }
    worker._remote_source_semantics = {
        "prefill": {
            0: (
                manifest.source_group_planes,
                manifest.group_token_capacities,
            )
        }
    }
    worker.kv_caches_base_addr = {"prefill": {0: [manifest.regions[0].base_address]}}
    metadata = ReqMeta(
        local_block_ids=((100, 101),),
        local_physical_block_ids=((100, 101),),
        tp_size=1,
        remote=RemoteMeta(
            block_ids=manifest.block_ids,
            host="127.0.0.1",
            port=5601,
            engine_id=manifest.producer_engine_id,
            request_id=manifest.producer_request_id,
            remote_num_tokens=manifest.valid_token_extent,
            p2d_run_id=config.run_id,
            p2d_transport_arm=config.transport_arm,
            source_offer_generation=manifest.offer_generation,
            p2d_iteration=manifest.iteration,
        ),
    )
    read_specs = [
        ReadSpec(
            remote_rank=0,
            local_block_ids=[[100, 101]],
            remote_block_ids=[[10, 11]],
        )
    ]
    return worker, child_request_id, metadata, read_specs, contract


@pytest.mark.cpu_test
def test_source_contract_is_built_from_request_and_handshake_lineage(
    tmp_path: Path,
) -> None:
    worker, req_id, metadata, read_specs, expected = _contract_worker(tmp_path)

    contracts = worker._localization_build_source_contracts(
        req_id,
        metadata,
        read_specs,
        expected.block_ids,
        expected.region_lengths,
    )

    assert contracts == (expected,)
    assert validate_source_contract_structure(contracts[0]) == ()


@pytest.mark.cpu_test
def test_source_contract_rejects_nonphysical_region_geometry(tmp_path: Path) -> None:
    """Artifact validation independently rejects impossible registration facts."""
    contract = source_contract_from_manifest(_manifest(_config(tmp_path)))
    invalid_region = msgspec.structs.replace(
        contract.regions[0],
        base_address=True,
        shape=(),
        strides=(),
    )
    invalid_contract = msgspec.structs.replace(
        contract,
        regions=(invalid_region,),
    )

    errors = validate_source_contract_structure(invalid_contract)

    assert any("geometry contains a non-integer" in error for error in errors)
    assert any("invalid tensor geometry" in error for error in errors)


@pytest.mark.cpu_test
def test_source_contract_rejects_mismatched_request_lineage(tmp_path: Path) -> None:
    worker, req_id, metadata, read_specs, expected = _contract_worker(tmp_path)
    assert metadata.remote is not None
    metadata.remote.p2d_run_id = "different-run"

    with pytest.raises(LocalizationError, match="mismatched localization lineage"):
        worker._localization_build_source_contracts(
            req_id,
            metadata,
            read_specs,
            expected.block_ids,
            expected.region_lengths,
        )


@pytest.mark.cpu_test
def test_source_contract_rejects_cross_target_lineage(tmp_path: Path) -> None:
    """Allowlist membership cannot pair a child with another target's producer."""
    worker, req_id, metadata, read_specs, expected = _contract_worker(tmp_path)
    worker._localization_config = _config(
        tmp_path,
        target_request_ids=(TARGET_REQUEST_ID_BASE, "chatcmpl-z-target"),
    )
    assert metadata.remote is not None
    metadata.remote.request_id = "chatcmpl-z-target-11111111"

    with pytest.raises(LocalizationError, match="mismatched localization lineage"):
        worker._localization_build_source_contracts(
            req_id,
            metadata,
            read_specs,
            expected.block_ids,
            expected.region_lengths,
        )


@pytest.mark.cpu_test
def test_source_contract_rejects_handshake_outside_native_registration(
    tmp_path: Path,
) -> None:
    worker, req_id, metadata, read_specs, expected = _contract_worker(tmp_path)
    worker.kv_caches_base_addr["prefill"][0] = [
        expected.regions[0].base_address + expected.regions[0].row_bytes
    ]

    with pytest.raises(LocalizationError, match="native registration"):
        worker._localization_build_source_contracts(
            req_id,
            metadata,
            read_specs,
            expected.block_ids,
            expected.region_lengths,
        )


@pytest.mark.cpu_test
def test_completed_source_roster_captures_source_post_and_remains_leased(
    tmp_path: Path,
) -> None:
    req_id = f"{TARGET_REQUEST_ID_BASE}-11111111"
    roster = NixlSourceRoster(
        offer_generation=1,
        iteration=0,
        expected_consumers=1,
        valid_token_extent=100,
        group_token_capacities=(64,),
        block_ids=((10, 11),),
    )
    worker = object.__new__(NixlPullConnectorWorker)
    worker._localization_config = _config(tmp_path)
    worker._source_rosters = {req_id: roster}
    capture = MagicMock()
    worker._localization_capture_source_manifest = capture

    worker._localization_capture_source_post(req_id)

    capture.assert_called_once_with(req_id, roster, IntegrityStage.SOURCE_POST)
    assert worker._source_rosters == {req_id: roster}


@pytest.mark.cpu_test
def test_complete_trace_validates(tmp_path: Path) -> None:
    """A complete source, plan, four-stage, and terminal trace is accepted."""
    paths, _, _ = _write_trace(tmp_path)
    report = validate_localization_artifacts(paths)

    assert report.passed
    assert report.physical_pull_count == 1
    assert report.verified_pull_count == 1


@pytest.mark.cpu_test
def test_complete_fingerprint_trace_validates(tmp_path: Path) -> None:
    """A compact post-scatter trace proves the full P-to-D path."""
    paths, _, _ = _write_trace(tmp_path, mode=LocalizationMode.FINGERPRINT)

    report = validate_localization_artifacts(paths)

    assert report.passed
    assert report.physical_pull_count == 1
    assert report.verified_pull_count == 1


@pytest.mark.cpu_test
def test_validator_requires_every_configured_target(tmp_path: Path) -> None:
    """An allowlisted parent with no source or child evidence is incomplete."""
    missing_target = "chatcmpl-z-target"
    paths, _, _ = _write_trace(
        tmp_path,
        mode=LocalizationMode.FINGERPRINT,
        target_request_ids=(TARGET_REQUEST_ID_BASE, missing_target),
    )

    report = validate_localization_artifacts(paths)

    assert report.passed is False
    assert any(
        f"configured target {missing_target!r} has no SOURCE_POST manifest" in error
        for error in report.errors
    )


@pytest.mark.cpu_test
def test_validator_requires_every_parallel_child_index(tmp_path: Path) -> None:
    """A producer-declared n=8 request cannot pass with only one child."""
    paths, _, _ = _write_trace(
        tmp_path,
        mode=LocalizationMode.FINGERPRINT,
        child_request_id=f"0_{TARGET_REQUEST_ID_BASE}-abcdef12",
        expected_consumers=8,
    )

    report = validate_localization_artifacts(paths)

    assert report.passed is False
    assert any(
        "child indices are incomplete: [0] != [0, 1, 2, 3, 4, 5, 6, 7]" in error
        for error in report.errors
    )


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("expected_consumers", "child_request_id", "error_fragment"),
    (
        (
            1,
            f"0_{TARGET_REQUEST_ID_BASE}-abcdef12",
            "single-consumer target",
        ),
        (
            8,
            f"{TARGET_REQUEST_ID_BASE}-abcdef12",
            "parallel target",
        ),
    ),
)
def test_validator_requires_canonical_child_form(
    tmp_path: Path,
    expected_consumers: int,
    child_request_id: str,
    error_fragment: str,
) -> None:
    """Consumer cardinality independently determines the child-ID form."""
    paths, _, _ = _write_trace(
        tmp_path,
        mode=LocalizationMode.FINGERPRINT,
        child_request_id=child_request_id,
        expected_consumers=expected_consumers,
    )

    report = validate_localization_artifacts(paths)

    assert report.passed is False
    assert any(error_fragment in error for error in report.errors)


@pytest.mark.cpu_test
def test_complete_interleaved_multi_target_trace_validates(tmp_path: Path) -> None:
    """One process artifact can prove independent pulls for several parents."""
    target_request_ids = (TARGET_REQUEST_ID_BASE, "chatcmpl-z-target")
    config = _config(
        tmp_path,
        mode=LocalizationMode.FINGERPRINT,
        target_request_ids=target_request_ids,
    )
    source_writers = tuple(
        LocalizationArtifactWriter(config, "prefill", rank, 2, 8) for rank in range(2)
    )
    manifests_by_target: list[tuple[NixlSourceManifest, ...]] = []
    for target_index, target_request_id in enumerate(target_request_ids):
        manifests = tuple(
            _manifest(
                config,
                region=_region(base_address=0x100000 + source_rank * 0x10000),
                source_rank=source_rank,
                registration_generation=f"registration-{source_rank}",
                producer_request_id=f"{target_request_id}-{target_index + 1:08x}",
                expected_consumers=8,
            )
            for source_rank in range(2)
        )
        manifests_by_target.append(manifests)
        for source_writer, manifest in zip(
            source_writers,
            manifests,
            strict=True,
        ):
            source_writer.write(
                NixlSourceManifestRecord(
                    record_type=NixlSourceManifestRecord.RECORD_TYPE,
                    stage=IntegrityStage.SOURCE_POST,
                    manifest=manifest,
                )
            )
    for source_writer in source_writers:
        source_writer.close()

    decoder_writer = LocalizationArtifactWriter(config, "decoder", 0, 1, 8)
    for target_index, manifests in enumerate(manifests_by_target):
        contracts = tuple(
            source_contract_from_manifest(manifest) for manifest in manifests
        )
        for child_index in range(8):
            plan = _plan(
                contracts[0],
                source_contracts=contracts,
                local_region=_region(base_address=0x200000, row_bytes=16),
                child_request_id=(
                    f"{child_index}_{target_request_ids[target_index]}-abcdef12"
                ),
            )
            decoder_writer.write(plan)
            for stage in (
                IntegrityStage.STAGING_POST_SCATTER,
                IntegrityStage.DESTINATION,
                IntegrityStage.PRE_READ,
            ):
                for manifest in manifests:
                    decoder_writer.write(_capture(manifest, plan, stage))
            decoder_writer.write(
                NixlEventRecord(
                    record_type=NixlEventRecord.RECORD_TYPE,
                    schema_version=IntegrityIdentity.SCHEMA_VERSION,
                    run_id=config.run_id,
                    transport_arm=config.transport_arm,
                    code="CAPTURE_COMPLETE",
                    evidentiary=False,
                    producer_engine_id=manifests[0].producer_engine_id,
                    producer_request_id=manifests[0].producer_request_id,
                    child_request_id=plan.child_request_id,
                    observer_engine_id=plan.observer_engine_id,
                    observer_rank=plan.observer_rank,
                    detail=(
                        "all decoder stages captured; offline source comparison pending"
                    ),
                    created_ns=time.time_ns(),
                )
            )
    decoder_writer.close()

    paths = tuple(writer.path for writer in source_writers) + (decoder_writer.path,)
    report = validate_localization_artifacts(paths)

    assert report.passed
    assert report.physical_pull_count == 16
    assert report.verified_pull_count == 16
    assert (
        read_localization_artifact(decoder_writer.path).session.target_request_ids
        == target_request_ids
    )


@pytest.mark.cpu_test
def test_validator_rejects_cross_target_plan_lineage(tmp_path: Path) -> None:
    """A child and producer from different allowed parents cannot share a plan."""
    target_request_ids = (TARGET_REQUEST_ID_BASE, "chatcmpl-z-target")
    config = _config(tmp_path, target_request_ids=target_request_ids)
    manifest = _manifest(
        config,
        producer_request_id=f"{TARGET_REQUEST_ID_BASE}-11111111",
    )
    source_writer = LocalizationArtifactWriter(config, "prefill", 0, 1, 8)
    source_writer.write(
        NixlSourceManifestRecord(
            record_type=NixlSourceManifestRecord.RECORD_TYPE,
            stage=IntegrityStage.SOURCE_POST,
            manifest=manifest,
        )
    )
    source_writer.close()
    decoder_writer = LocalizationArtifactWriter(config, "decoder", 0, 1, 8)
    decoder_writer.write(
        _plan(
            source_contract_from_manifest(manifest),
            child_request_id="4_chatcmpl-z-target-22222222",
        )
    )
    decoder_writer.close()

    report = validate_localization_artifacts((source_writer.path, decoder_writer.path))

    assert report.passed is False
    assert any(
        "plan source differs from its session target" in error
        for error in report.errors
    )


@pytest.mark.cpu_test
def test_complete_multi_decoder_rank_trace_validates(tmp_path: Path) -> None:
    """Independent P4 and D2 sessions prove the complete request partition."""
    paths, _, _ = _write_trace(
        tmp_path,
        decoder_world_size=2,
        complete_decoder_world=True,
    )

    report = validate_localization_artifacts(paths)

    assert report.passed
    assert report.physical_pull_count == 1
    assert report.verified_pull_count == 1


@pytest.mark.cpu_test
def test_decoder_ranks_must_share_one_source_request_contract(
    tmp_path: Path,
) -> None:
    """Each D-rank partition must describe the same producer block roster."""
    paths, _, _ = _write_trace(
        tmp_path,
        decoder_world_size=2,
        complete_decoder_world=True,
        source_block_rosters={2: (12, 13), 3: (12, 13)},
    )

    report = validate_localization_artifacts(paths)

    assert not report.passed
    assert any(
        "disagree on the producer request contract" in error for error in report.errors
    )


@pytest.mark.cpu_test
def test_validator_accepts_independent_internal_request_ids(tmp_path: Path) -> None:
    paths, _, _ = _write_trace(
        tmp_path,
        producer_request_id=f"{TARGET_REQUEST_ID_BASE}-11111111",
        child_request_id=f"{TARGET_REQUEST_ID_BASE}-22222222",
    )

    report = validate_localization_artifacts(paths)

    assert report.passed
    assert report.physical_pull_count == 1
    assert report.verified_pull_count == 1


@pytest.mark.cpu_test
def test_validator_rejects_prefixed_producer_identity(tmp_path: Path) -> None:
    """Parallel choice prefixes are decoder-only lineage."""
    paths, _, _ = _write_trace(
        tmp_path,
        producer_request_id=f"4_{TARGET_REQUEST_ID_BASE}-11111111",
        child_request_id=f"{TARGET_REQUEST_ID_BASE}-22222222",
    )

    report = validate_localization_artifacts(paths)

    assert report.passed is False
    assert any(
        "source manifest request differs from its session target" in error
        for error in report.errors
    )


@pytest.mark.cpu_test
def test_reader_rejects_truncated_and_corrupted_frames(tmp_path: Path) -> None:
    """Neither truncation nor a payload mutation can resemble a clean trace."""
    paths, _, _ = _write_trace(tmp_path / "source")
    payload = paths[0].read_bytes()
    truncated = tmp_path / "truncated.msgpack"
    truncated.write_bytes(payload[:-3])
    corrupted = tmp_path / "corrupted.msgpack"
    changed = bytearray(payload)
    changed[-1] ^= 1
    corrupted.write_bytes(changed)

    with pytest.raises(LocalizationError, match="truncated"):
        read_localization_artifact(truncated)
    with pytest.raises(LocalizationError, match="checksum mismatch"):
        read_localization_artifact(corrupted)


@pytest.mark.cpu_test
def test_validator_rejects_duplicate_and_mixed_artifacts(tmp_path: Path) -> None:
    """Duplicate process files and a second run are both fail-closed."""
    paths, _, _ = _write_trace(tmp_path / "trace")
    duplicate_report = validate_localization_artifacts((*paths, paths[0]))
    assert any("duplicate" in error for error in duplicate_report.errors)

    mixed_config = _config(tmp_path / "mixed", run_id="other-run")
    mixed_writer = LocalizationArtifactWriter(
        mixed_config,
        "other-engine",
        0,
        1,
        8,
    )
    mixed_writer.close()
    mixed_report = validate_localization_artifacts((*paths, mixed_writer.path))
    assert any("mixed" in error for error in mixed_report.errors)

    mixed_target_config = _config(
        tmp_path / "mixed-target",
        target_request_ids=("chatcmpl-unrelated",),
    )
    mixed_target_writer = LocalizationArtifactWriter(
        mixed_target_config,
        "target-engine",
        0,
        1,
        8,
    )
    mixed_target_writer.close()
    mixed_target_report = validate_localization_artifacts(
        (*paths, mixed_target_writer.path)
    )
    assert any("mixed" in error for error in mixed_target_report.errors)


@pytest.mark.cpu_test
def test_validator_rejects_missing_entire_decoder_rank(tmp_path: Path) -> None:
    """A missing process artifact cannot shrink the observed decoder world."""
    paths, _, _ = _write_trace(tmp_path, decoder_world_size=2)

    report = validate_localization_artifacts(paths)

    assert not report.passed
    assert any(
        "engine decoder process ranks are incomplete" in error
        for error in report.errors
    )


@pytest.mark.cpu_test
def test_corruption_localizes_to_first_raw_staging_edge(tmp_path: Path) -> None:
    """Injected corruption names SOURCE_POST reference versus raw staging."""
    paths, _, _ = _write_trace(
        tmp_path,
        corrupt_stage=IntegrityStage.STAGING_RAW,
    )
    report = validate_localization_artifacts(paths)

    assert not report.passed
    assert len(report.errors) == 0
    assert len(report.divergences) == 1
    assert report.divergences[0].edge == "source_post_reference_vs_staging_raw"
    assert any("digest mismatch" in error for error in report.divergences[0].errors)


@pytest.mark.cpu_test
def test_fingerprint_corruption_localizes_to_post_scatter_staging(
    tmp_path: Path,
) -> None:
    """A changed compact fingerprint names the first observable P-to-D edge."""
    paths, _, _ = _write_trace(
        tmp_path,
        corrupt_stage=IntegrityStage.STAGING_POST_SCATTER,
        mode=LocalizationMode.FINGERPRINT,
    )

    report = validate_localization_artifacts(paths)

    assert report.passed is False
    assert len(report.errors) == 0
    assert len(report.divergences) == 1
    assert report.divergences[0].edge == "source_post_reference_vs_staging_post_scatter"
    assert any("digest mismatch" in error for error in report.divergences[0].errors)


@pytest.mark.cpu_test
def test_validator_rejects_noncanonical_runtime_barrier(tmp_path: Path) -> None:
    """A stale runtime ordering claim cannot validate as decoder evidence."""
    paths, _, _ = _write_trace(
        tmp_path,
        corrupt_barrier_stage=IntegrityStage.DESTINATION,
    )

    report = validate_localization_artifacts(paths)

    assert report.passed is False
    assert len(report.divergences) == 1
    assert report.divergences[0].edge == "staging_fenced_control->destination"
    assert "capture barrier label differs from stage contract" in (
        report.divergences[0].errors
    )


@pytest.mark.cpu_test
def test_missing_terminal_child_outcome_is_not_clean(tmp_path: Path) -> None:
    """A process-terminal file cannot hide a missing per-child terminal result."""
    paths, _, _ = _write_trace(tmp_path, include_event=False)
    report = validate_localization_artifacts(paths)

    assert not report.passed
    assert any("no terminal child outcome" in error for error in report.errors)


@pytest.mark.cpu_test
def test_zero_byte_outcome_is_complete_but_non_evidentiary(tmp_path: Path) -> None:
    """A full-prefix hit terminates explicitly without entering pull comparisons."""
    config = _config(tmp_path)
    manifest = _manifest(config)
    source_writer = LocalizationArtifactWriter(config, "prefill", 0, 1, 8)
    source_writer.write(
        NixlSourceManifestRecord(
            record_type=NixlSourceManifestRecord.RECORD_TYPE,
            stage=IntegrityStage.SOURCE_POST,
            manifest=manifest,
        )
    )
    source_writer.close()
    decoder_writer = LocalizationArtifactWriter(config, "decoder", 0, 1, 8)
    decoder_writer.write(
        NixlEventRecord(
            record_type=NixlEventRecord.RECORD_TYPE,
            schema_version=IntegrityIdentity.SCHEMA_VERSION,
            run_id=config.run_id,
            transport_arm=config.transport_arm,
            code="NON_EVIDENTIARY_ZERO_BYTE",
            evidentiary=False,
            producer_engine_id=manifest.producer_engine_id,
            producer_request_id=manifest.producer_request_id,
            child_request_id=f"{TARGET_REQUEST_ID_BASE}-33333333",
            observer_engine_id="decoder",
            observer_rank=0,
            detail="full-prefix hit excluded from physical-pull comparisons",
            created_ns=time.time_ns(),
        )
    )
    decoder_writer.close()

    report = validate_localization_artifacts((source_writer.path, decoder_writer.path))
    assert len(report.errors) == 0
    assert len(report.divergences) == 0
    assert report.physical_pull_count == 0
    assert report.excluded_outcome_count == 1
    assert not report.passed


@pytest.mark.cpu_test
def test_semantic_contract_detects_equal_size_region_substitution(
    tmp_path: Path,
) -> None:
    """Equal byte geometry does not make two layer-slot contracts equivalent."""
    config = _config(tmp_path)
    baseline = _region()
    substituted = _region(group_semantic_name="model.layers.60:transfer_region_0")
    baseline_digest = compute_semantic_contract_digest(
        region=baseline,
        group_index=0,
        group_token_capacity=64,
        source_plane_contract=2,
    )
    substituted_digest = compute_semantic_contract_digest(
        region=substituted,
        group_index=0,
        group_token_capacity=64,
        source_plane_contract=2,
    )

    assert baseline_digest != substituted_digest
    manifest = _manifest(config, region=substituted)
    plan = _plan(source_contract_from_manifest(manifest), local_region=baseline)
    errors = validate_localization_plan(plan)
    assert any("semantic region mismatch" in error for error in errors)


@pytest.mark.cpu_test
def test_plan_rejects_destination_row_width_different_from_source_partition(
    tmp_path: Path,
) -> None:
    """The D row must contain exactly one shard from every participating P rank."""
    config = _config(tmp_path)
    first_manifest = _manifest(config, source_rank=0)
    first_contract = source_contract_from_manifest(first_manifest)
    second_contract = msgspec.structs.replace(
        first_contract,
        source_rank=1,
        registration_generation="registration-1",
        regions=(
            msgspec.structs.replace(
                first_contract.regions[0],
                base_address=first_contract.regions[0].base_address + 0x10000,
            ),
        ),
    )
    plan = _plan(
        first_contract,
        source_contracts=(first_contract, second_contract),
        local_region=_region(base_address=0x200000, row_bytes=8),
        source_tp_size=2,
    )

    errors = validate_localization_plan(plan)

    assert any("participating source-rank geometry" in error for error in errors)


@pytest.mark.cpu_test
def test_plan_rejects_rank_local_expected_consumer_drift(tmp_path: Path) -> None:
    """Expected-consumer cardinality is one request contract across P ranks."""
    config = _config(tmp_path)
    first_manifest = _manifest(config, source_rank=0)
    first_contract = source_contract_from_manifest(first_manifest)
    second_contract = msgspec.structs.replace(
        first_contract,
        source_rank=1,
        registration_generation="registration-1",
        expected_consumers=2,
        regions=(
            msgspec.structs.replace(
                first_contract.regions[0],
                base_address=first_contract.regions[0].base_address + 0x10000,
            ),
        ),
    )
    plan = _plan(
        first_contract,
        source_contracts=(first_contract, second_contract),
        local_region=_region(base_address=0x200000, row_bytes=16),
        source_tp_size=2,
    )

    errors = validate_localization_plan(plan)

    assert "plan source ranks disagree on transfer contract" in errors


@pytest.mark.cpu_test
def test_plan_rejects_odd_trim_relative_half_mapping(tmp_path: Path) -> None:
    """Single-plane halves derive from absolute source positions after trimming."""
    config = _config(tmp_path)
    manifest = _manifest(config, blocks=(10, 11, 12), source_planes=1)
    relative_halves = (
        NixlPlanPosition(
            group_index=0,
            source_position=1,
            remote_block_id=11,
            valid_token_extent=100,
            group_token_capacity=64,
            local_block_id=100,
            destination_half=0,
        ),
        NixlPlanPosition(
            group_index=0,
            source_position=2,
            remote_block_id=12,
            valid_token_extent=100,
            group_token_capacity=64,
            local_block_id=100,
            destination_half=1,
        ),
    )
    plan = _plan(
        source_contract_from_manifest(manifest),
        selected_remote=(11, 12),
        selected_local=(100, 101),
        destination_planes=1,
        positions=relative_halves,
    )

    errors = validate_localization_plan(plan)
    assert any("ownership-filtered canonical layout" in error for error in errors)


@pytest.mark.cpu_test
def test_plan_rejects_swapped_destinations_with_unchanged_source_contract(
    tmp_path: Path,
) -> None:
    """Content equality cannot bless a source-to-destination permutation."""
    config = _config(tmp_path)
    manifest = _manifest(config)
    swapped = (
        NixlPlanPosition(
            group_index=0,
            source_position=0,
            remote_block_id=10,
            valid_token_extent=100,
            group_token_capacity=64,
            local_block_id=101,
            destination_half=-1,
        ),
        NixlPlanPosition(
            group_index=0,
            source_position=1,
            remote_block_id=11,
            valid_token_extent=100,
            group_token_capacity=64,
            local_block_id=100,
            destination_half=-1,
        ),
    )
    plan = _plan(source_contract_from_manifest(manifest), positions=swapped)

    errors = validate_localization_plan(plan)
    assert any("ownership-filtered canonical layout" in error for error in errors)


@pytest.mark.cpu_test
def test_plan_rejects_rank_slot_swap_against_topology_contract(
    tmp_path: Path,
) -> None:
    """A bijection cannot override independently recorded P/D topology."""
    paths, _, _ = _write_trace(tmp_path, rank_slots=(1, 0))

    report = validate_localization_artifacts(paths)

    assert not report.passed
    assert any(
        "rank slots differ from session-derived topology" in error
        for error in report.errors
    )


@pytest.mark.cpu_test
def test_plan_rejects_source_tp_size_different_from_producer_session(
    tmp_path: Path,
) -> None:
    """The producer session, not the plan itself, fixes source TP size."""
    paths, _, _ = _write_trace(tmp_path, recorded_source_tp_size=3)

    report = validate_localization_artifacts(paths)

    assert report.passed is False
    assert any(
        "source TP size differs from producer session world size" in error
        for error in report.errors
    )


@pytest.mark.cpu_test
def test_plan_with_skipped_group_is_non_evidentiary(tmp_path: Path) -> None:
    """Partial group coverage cannot support a localization verdict."""
    manifest = _manifest(_config(tmp_path))
    plan = msgspec.structs.replace(
        _plan(source_contract_from_manifest(manifest)),
        skipped_groups=(0,),
    )

    errors = validate_localization_plan(plan)

    assert "plan skips cache groups and is non-evidentiary" in errors


def _ownership_pruned_plan() -> NixlPlanRecord:
    """Build a two-group plan whose regions own disjoint position subsets."""
    source_regions = tuple(
        NixlRegionDescriptor(
            semantic_name=f"model.layers.{group_index}:transfer_region_0",
            group_indices=(group_index,),
            group_semantic_names=(
                (
                    group_index,
                    f"model.layers.{group_index}:transfer_region_0",
                ),
            ),
            base_address=0x300000 + group_index * 0x10000,
            registered_bytes=128 * row_bytes,
            row_bytes=row_bytes,
            shape=(128, row_bytes),
            strides=(row_bytes, 1),
            dtype="torch.uint8",
            element_size_bytes=1,
            layout="HND",
        )
        for group_index, row_bytes in enumerate((8, 4))
    )
    local_regions = tuple(
        msgspec.structs.replace(
            region,
            base_address=region.base_address + 0x100000,
        )
        for region in source_regions
    )
    contract = NixlSourceContract(
        schema_version=IntegrityIdentity.SCHEMA_VERSION,
        fingerprint_algorithm=LocalizationFingerprintAlgorithm.BLAKE2B_128,
        run_id="rank-major-layout",
        transport_arm="cuda-ipc",
        producer_engine_id="prefill",
        producer_request_id=TARGET_REQUEST_ID_BASE,
        registration_generation="registration-0",
        offer_generation=1,
        iteration=0,
        expected_consumers=1,
        source_rank=0,
        region_lengths=(8, 4),
        regions=source_regions,
        source_group_planes=(2, 1),
        valid_token_extent=100,
        group_token_capacities=(64, 64),
        block_ids=((10, 11, 12), (20, 21)),
    )
    groups = (
        GroupTransferRoster(
            group_index=0,
            source_position_start=0,
            destination_plane_count=2,
            local_block_ids=(30, 31, 32),
            remote_block_ids=(10, 11, 12),
        ),
        GroupTransferRoster(
            group_index=1,
            source_position_start=0,
            destination_plane_count=1,
            local_block_ids=(40,),
            remote_block_ids=(20, 21),
        ),
    )
    return _plan_from_layout(contract, local_regions, groups)


@pytest.mark.cpu_test
def test_rank_major_region_pruned_plan_reconstructs_exactly() -> None:
    """Region ownership controls positions, packed offsets, and staging size."""
    plan = _ownership_pruned_plan()

    assert validate_localization_plan(plan) == ()
    assert tuple(
        tuple(position.group_index for position in region.positions)
        for region in plan.region_plans
    ) == ((0, 0, 0), (1, 1))
    assert tuple(region.offset_within_rank for region in plan.region_plans) == (0, 24)
    assert plan.rank_stride_bytes == 32
    assert plan.staging_size == 32
    assert len(plan.layout_digest) == 64


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("mutation", "error_fragment"),
    (
        ("ownership", "ownership-filtered canonical layout"),
        ("runs", "runs differ"),
        ("offset", "packed offset"),
        ("rank_stride", "rank stride"),
        ("staging_size", "staging size"),
        ("digest", "layout digest"),
    ),
)
def test_rank_major_plan_rejects_noncanonical_layout_facts(
    mutation: str,
    error_fragment: str,
) -> None:
    """Every serialized layout fact is checked against reconstruction."""
    plan = _ownership_pruned_plan()
    if mutation == "ownership":
        first_region = plan.region_plans[0]
        bad_position = msgspec.structs.replace(
            first_region.positions[0],
            group_index=1,
        )
        bad_region = msgspec.structs.replace(
            first_region,
            positions=(bad_position, *first_region.positions[1:]),
        )
        plan = msgspec.structs.replace(
            plan,
            region_plans=(bad_region, *plan.region_plans[1:]),
        )
    elif mutation == "runs":
        first_region = plan.region_plans[0]
        bad_run = msgspec.structs.replace(
            first_region.runs[0],
            position_count=first_region.runs[0].position_count - 1,
        )
        bad_region = msgspec.structs.replace(first_region, runs=(bad_run,))
        plan = msgspec.structs.replace(
            plan,
            region_plans=(bad_region, *plan.region_plans[1:]),
        )
    elif mutation == "offset":
        second_region = msgspec.structs.replace(
            plan.region_plans[1],
            offset_within_rank=plan.region_plans[1].offset_within_rank + 1,
        )
        plan = msgspec.structs.replace(
            plan,
            region_plans=(plan.region_plans[0], second_region),
        )
    elif mutation == "rank_stride":
        plan = msgspec.structs.replace(
            plan,
            rank_stride_bytes=plan.rank_stride_bytes + 1,
        )
    elif mutation == "staging_size":
        plan = msgspec.structs.replace(plan, staging_size=plan.staging_size + 1)
    elif mutation == "digest":
        plan = msgspec.structs.replace(plan, layout_digest="0" * 64)
    else:
        raise AssertionError(f"unhandled mutation {mutation}")

    errors = validate_localization_plan(plan)
    assert any(error_fragment in error for error in errors)


@pytest.mark.cpu_test
def test_source_manifest_excludes_unowned_region_group_cross_product(
    tmp_path: Path,
) -> None:
    """Dead broadcast rows are not admitted into semantic integrity roots."""
    config = _config(tmp_path)
    manifest = _manifest(config)

    assert validate_source_manifest_structure(manifest) == ()
    assert all(
        leaf.group_index in manifest.regions[0].group_indices
        for leaf in manifest.leaves
    )
