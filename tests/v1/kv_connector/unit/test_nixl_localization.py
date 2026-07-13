# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for authoritative P-to-D localization artifacts."""

import time
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from queue import Queue
from unittest.mock import MagicMock

import msgspec
import pytest
import zmq

from vllm.distributed.kv_transfer.integrity import (
    IntegrityIdentity,
    IntegrityPayloadKind,
    IntegrityStage,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import (
    pull_worker as pull_worker_module,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    RemoteMeta,
    ReqMeta,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_worker import (
    NixlPullConnectorWorker,
)
from vllm.distributed.kv_transfer.nixl_localization import (
    LocalizationArtifactWriter,
    LocalizationError,
    LocalizationMode,
    ManifestStatus,
    NixlCaptureRecord,
    NixlEventRecord,
    NixlIntegrityLeaf,
    NixlLocalizationConfig,
    NixlPlanPosition,
    NixlPlanRecord,
    NixlRegionDescriptor,
    NixlSourceManifest,
    NixlSourceManifestRecord,
    NixlSourceManifestResponse,
    build_integrity_identity,
    build_integrity_leaf,
    compute_semantic_contract_digest,
    resolve_source_manifest_request,
    seal_source_manifest,
    validate_source_manifest_structure,
)
from vllm.distributed.kv_transfer.nixl_localization_validator import (
    read_localization_artifact,
    validate_localization_artifacts,
    validate_localization_plan,
)

HTTP_TARGET_REQUEST_ID = "p2d-phase2db-separated-score2-20260713-p000-s000637"
TARGET_REQUEST_ID_BASE = f"chatcmpl-{HTTP_TARGET_REQUEST_ID}"


class _ManifestTransport:
    """Factory supplying one scripted socket per manifest exchange."""

    scripts: list[list[bytes | BaseException]]
    sockets: list[MagicMock]
    exit_counts: list[int]

    def __init__(self, scripts: list[list[bytes | BaseException]]) -> None:
        """
        :param scripts: Receive scripts for successive exchanges.
        """
        self.scripts = [list(script) for script in scripts]
        self.sockets = []
        self.exit_counts = []

    def __call__(
        self,
        socket_type: int,
        address: str,
    ) -> AbstractContextManager[MagicMock]:
        """
        :param socket_type: Requested ZMQ socket type.
        :param address: Producer side-channel address.
        :returns: A fresh scripted exchange context.
        """
        assert socket_type == zmq.REQ
        assert address == "tcp://127.0.0.1:5601"
        if len(self.scripts) == 0:
            raise AssertionError("unexpected source-manifest exchange")
        socket = MagicMock()
        socket.recv.side_effect = self.scripts.pop(0)
        exchange_index = len(self.sockets)
        self.sockets.append(socket)
        self.exit_counts.append(0)
        return self._exchange(exchange_index, socket)

    @contextmanager
    def _exchange(
        self,
        exchange_index: int,
        socket: MagicMock,
    ) -> Iterator[MagicMock]:
        """Track closure of a manually retained exchange.

        :param exchange_index: Index of the exchange being entered.
        :param socket: Scripted socket yielded to the worker.
        :yields: Scripted socket for one manifest exchange.
        """
        try:
            yield socket
        finally:
            self.exit_counts[exchange_index] += 1


def _config(
    artifact_dir: Path,
    *,
    run_id: str = "run-localization",
    mode: LocalizationMode = LocalizationMode.TRACE,
    target_request_id: str = TARGET_REQUEST_ID_BASE,
) -> NixlLocalizationConfig:
    return NixlLocalizationConfig(
        mode=mode,
        run_id=run_id,
        transport_arm="tcp-shm-cuda-copy",
        target_request_id=target_request_id,
        artifact_dir=artifact_dir,
        manifest_timeout_s=30.0,
        copy_chunk_bytes=64 * 1024 * 1024,
        strict_zero_byte=False,
    )


def _region(
    *,
    base_address: int = 0x100000,
    group_semantic_name: str = "model.layers.0:transfer_region_0",
    shape: tuple[int, ...] = (16, 8),
    strides: tuple[int, ...] = (8, 1),
) -> NixlRegionDescriptor:
    return NixlRegionDescriptor(
        semantic_name=group_semantic_name,
        group_indices=(0,),
        group_semantic_names=((0, group_semantic_name),),
        base_address=base_address,
        registered_bytes=128,
        row_bytes=8,
        shape=shape,
        strides=strides,
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
) -> NixlSourceManifest:
    source_region = region if region is not None else _region()
    request_id = (
        producer_request_id
        if producer_request_id is not None
        else config.target_request_id
    )
    contract_digest = compute_semantic_contract_digest(
        region=source_region,
        group_index=0,
        group_token_capacity=64,
        source_plane_contract=2,
    )
    leaves: list[NixlIntegrityLeaf] = []
    for source_position, block_id in enumerate(blocks):
        payload = bytes((block_id + offset) % 256 for offset in range(8))
        for plane_index, payload_kind, plane_payload in (
            (-1, IntegrityPayloadKind.WIRE, payload),
            (0, IntegrityPayloadKind.COMMIT, payload[:4]),
            (1, IntegrityPayloadKind.COMMIT, payload[4:]),
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
            leaves.append(
                build_integrity_leaf(
                    identity=identity,
                    payload=plane_payload,
                    local_block_id=None,
                    destination_half=None,
                    rank_slot=None,
                )
            )
    return seal_source_manifest(
        NixlSourceManifest(
            schema_version=IntegrityIdentity.SCHEMA_VERSION,
            run_id=config.run_id,
            transport_arm=config.transport_arm,
            producer_engine_id="prefill",
            producer_request_id=request_id,
            registration_generation=registration_generation,
            offer_generation=1,
            iteration=0,
            source_rank=source_rank,
            region_lengths=(8,),
            regions=(source_region,),
            source_group_planes=(2,),
            valid_token_extent=100,
            group_token_capacities=(64,),
            block_ids=(blocks,),
            observer=True,
            copied_bytes=len(blocks) * 8,
            hashed_bytes=len(blocks) * 16,
            duration_ns=1000,
            manifest_digest=b"",
            leaves=tuple(leaves),
        )
    )


def _manifest_response(
    status: ManifestStatus,
    manifests: tuple[NixlSourceManifest, ...] = (),
) -> bytes:
    """Encode one producer response for the decoder gate.

    :param status: Producer-side manifest status.
    :param manifests: Source manifests carried by a ready response.
    :returns: Encoded side-channel payload.
    """
    return msgspec.msgpack.encode(
        NixlSourceManifestResponse(
            schema_version=IntegrityIdentity.SCHEMA_VERSION,
            status=status,
            detail=status.value,
            manifests=manifests,
        )
    )


def _source_gate_worker(
    artifact_dir: Path,
) -> tuple[NixlPullConnectorWorker, str, ReqMeta, NixlSourceManifest]:
    """Build the target-only worker state needed by the source gate.

    :param artifact_dir: Localization artifact directory.
    :returns: Worker, decoder request, transfer metadata, and source manifest.
    """
    config = _config(artifact_dir)
    producer_request_id = f"{TARGET_REQUEST_ID_BASE}-11111111"
    child_request_id = f"{TARGET_REQUEST_ID_BASE}-22222222"
    manifest = _manifest(config, producer_request_id=producer_request_id)
    worker = object.__new__(NixlPullConnectorWorker)
    worker._localization_config = config
    worker._localization_expected_by_request = {}
    worker._localization_manifest_deadlines = {}
    worker._localization_manifest_requests = {}
    worker._localization_waiting = {}
    worker._remote_registration_generations = {
        "prefill": {0: manifest.registration_generation}
    }
    worker._remote_regions = {"prefill": {0: manifest.regions}}
    metadata = ReqMeta(
        local_block_ids=(),
        local_physical_block_ids=(),
        tp_size=4,
        remote=RemoteMeta(
            block_ids=manifest.block_ids,
            host="127.0.0.1",
            port=5601,
            engine_id=manifest.producer_engine_id,
            request_id=manifest.producer_request_id,
            p2d_run_id=config.run_id,
            p2d_transport_arm=config.transport_arm,
            p2d_offer_generation=manifest.offer_generation,
            p2d_iteration=manifest.iteration,
        ),
    )
    return worker, child_request_id, metadata, manifest


def _plan(
    manifest: NixlSourceManifest,
    *,
    selected_remote: tuple[int, ...] | None = None,
    selected_local: tuple[int, ...] | None = None,
    destination_planes: int = 2,
    positions: tuple[NixlPlanPosition, ...] | None = None,
    local_region: NixlRegionDescriptor | None = None,
    child_request_id: str | None = None,
) -> NixlPlanRecord:
    remote = selected_remote if selected_remote is not None else manifest.block_ids[0]
    if selected_local is None:
        selected_local = tuple(100 + index for index in range(len(remote)))
    source_start = (
        list(manifest.block_ids[0]).index(remote[0])
        if len(remote) > 0
        else 0
    )
    if positions is None:
        positions = tuple(
            NixlPlanPosition(
                group_index=0,
                source_position=source_start + index,
                remote_block_id=block_id,
                valid_token_extent=manifest.valid_token_extent,
                group_token_capacity=manifest.group_token_capacities[0],
                local_block_id=(
                    selected_local[index]
                    if destination_planes == 2
                    else selected_local[index // 2]
                ),
                plane_index=(
                    -1
                    if destination_planes == 2
                    else (source_start + index) % 2
                ),
            )
            for index, block_id in enumerate(remote)
        )
    remote_order = [position.remote_block_id for position in positions]
    runs: list[tuple[int, int, int]] = []
    if len(remote_order) > 0:
        start = 0
        for index in range(1, len(remote_order) + 1):
            if (
                index == len(remote_order)
                or remote_order[index] != remote_order[index - 1] + 1
            ):
                runs.append((remote_order[start], index - start, start))
                start = index
    return NixlPlanRecord(
        record_type=NixlPlanRecord.RECORD_TYPE,
        schema_version=IntegrityIdentity.SCHEMA_VERSION,
        run_id=manifest.run_id,
        transport_arm=manifest.transport_arm,
        producer_engine_id=manifest.producer_engine_id,
        producer_request_id=manifest.producer_request_id,
        registration_generations=(manifest.registration_generation,),
        source_manifest_digests=(manifest.manifest_digest,),
        offer_generation=manifest.offer_generation,
        iteration=manifest.iteration,
        child_request_id=(
            child_request_id
            if child_request_id is not None
            else manifest.producer_request_id
        ),
        observer_engine_id="decoder",
        observer_rank=0,
        source_ranks=(0,),
        rank_slots=(0,),
        rank_slot_contract=((0, 0),),
        destination_group_planes=(destination_planes,),
        region_lengths=manifest.region_lengths,
        regions=(manifest.regions,),
        local_regions=(
            local_region if local_region is not None else manifest.regions[0],
        ),
        raw_remote_groups=manifest.block_ids,
        selected_remote_groups=(remote,),
        selected_local_groups=(selected_local,),
        transfer_order=positions,
        runs=tuple(runs),
        region_offsets=(0,),
        staging_offset=0,
        staging_size=len(positions) * manifest.region_lengths[0],
    )


def _mapped_wire_leaf(
    source_leaf: NixlIntegrityLeaf,
    position: NixlPlanPosition,
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
        destination_half=position.plane_index,
        rank_slot=0,
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
        (leaf.source_position, leaf.remote_block_id): leaf
        for leaf in manifest.leaves
        if leaf.payload_kind is IntegrityPayloadKind.WIRE
    }
    leaves = tuple(
        _mapped_wire_leaf(
            source_wires[(position.source_position, position.remote_block_id)],
            position,
        )
        for position in plan.transfer_order
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
    barriers = {
        IntegrityStage.STAGING_RAW: "nixl_done_without_added_device_wide_sync",
        IntegrityStage.STAGING_FENCED_CONTROL: (
            "device_synchronize_observer_control_not_gdr_flush"
        ),
        IntegrityStage.DESTINATION: (
            "post_scatter_device_synchronize_before_publication"
        ),
        IntegrityStage.PRE_READ: (
            "after_transfer_phase_entry_before_new_dma_or_forward"
        ),
    }
    return NixlCaptureRecord(
        record_type=NixlCaptureRecord.RECORD_TYPE,
        schema_version=IntegrityIdentity.SCHEMA_VERSION,
        stage=stage,
        run_id=manifest.run_id,
        transport_arm=manifest.transport_arm,
        producer_engine_id=manifest.producer_engine_id,
        producer_request_id=manifest.producer_request_id,
        registration_generation=manifest.registration_generation,
        source_manifest_digest=manifest.manifest_digest,
        offer_generation=manifest.offer_generation,
        iteration=manifest.iteration,
        child_request_id=plan.child_request_id,
        source_rank=manifest.source_rank,
        observer_engine_id=plan.observer_engine_id,
        observer_rank=plan.observer_rank,
        observer=True,
        copied_bytes=len(plan.transfer_order) * manifest.region_lengths[0],
        hashed_bytes=len(plan.transfer_order) * manifest.region_lengths[0],
        duration_ns=1000,
        barrier=barriers[stage],
        leaves=leaves,
    )


def _write_trace(
    artifact_dir: Path,
    *,
    corrupt_stage: IntegrityStage | None = None,
    include_event: bool = True,
    producer_request_id: str | None = None,
    child_request_id: str | None = None,
) -> tuple[tuple[Path, Path], NixlSourceManifest, NixlPlanRecord]:
    config = _config(artifact_dir)
    manifest = _manifest(config, producer_request_id=producer_request_id)
    plan = _plan(manifest, child_request_id=child_request_id)
    source_writer = LocalizationArtifactWriter(config, "prefill", 0)
    source_writer.write(
        NixlSourceManifestRecord(
            record_type=NixlSourceManifestRecord.RECORD_TYPE,
            stage=IntegrityStage.SOURCE_PRE,
            manifest=manifest,
        )
    )
    source_writer.write(
        NixlSourceManifestRecord(
            record_type=NixlSourceManifestRecord.RECORD_TYPE,
            stage=IntegrityStage.SOURCE_POST,
            manifest=manifest,
        )
    )
    source_writer.close()

    decoder_writer = LocalizationArtifactWriter(config, "decoder", 0)
    decoder_writer.write(plan)
    for stage in (
        IntegrityStage.STAGING_RAW,
        IntegrityStage.STAGING_FENCED_CONTROL,
        IntegrityStage.DESTINATION,
        IntegrityStage.PRE_READ,
    ):
        decoder_writer.write(
            _capture(
                manifest,
                plan,
                stage,
                corrupt=stage is corrupt_stage,
            )
        )
    if include_event:
        decoder_writer.write(
            NixlEventRecord(
                record_type=NixlEventRecord.RECORD_TYPE,
                schema_version=IntegrityIdentity.SCHEMA_VERSION,
                run_id=config.run_id,
                transport_arm=config.transport_arm,
                code="VERIFIED_PRE_READ",
                evidentiary=True,
                producer_engine_id=manifest.producer_engine_id,
                producer_request_id=manifest.producer_request_id,
                child_request_id=plan.child_request_id,
                observer_engine_id=plan.observer_engine_id,
                observer_rank=plan.observer_rank,
                detail="all required ranks and stages matched before first read",
                created_ns=time.time_ns(),
            )
        )
    decoder_writer.close()
    return (source_writer.path, decoder_writer.path), manifest, plan


@pytest.mark.cpu_test
def test_config_requires_and_selects_exact_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_NIXL_P2D_LOCALIZATION", "trace")
    monkeypatch.setenv("VLLM_NIXL_P2D_RUN_ID", "targeted-run")
    monkeypatch.setenv("VLLM_NIXL_P2D_TRANSPORT_ARM", "cuda-copy")
    monkeypatch.setenv("VLLM_NIXL_P2D_ARTIFACT_DIR", str(tmp_path))
    monkeypatch.delenv("VLLM_NIXL_P2D_TARGET_REQUEST_ID", raising=False)

    with pytest.raises(ValueError, match="TARGET_REQUEST_ID is required"):
        NixlLocalizationConfig.from_environment()

    monkeypatch.setenv(
        "VLLM_NIXL_P2D_TARGET_REQUEST_ID",
        TARGET_REQUEST_ID_BASE,
    )
    config = NixlLocalizationConfig.from_environment()

    assert config.enabled
    assert config.enabled_for(TARGET_REQUEST_ID_BASE)
    assert config.enabled_for(f"{TARGET_REQUEST_ID_BASE}-deadbeef")
    assert config.enabled_for(f"{TARGET_REQUEST_ID_BASE}-DEADBEEF") is False
    assert config.enabled_for(f"{TARGET_REQUEST_ID_BASE}-r1-deadbeef") is False
    assert config.enabled_for(f"{TARGET_REQUEST_ID_BASE}-deadbee") is False
    assert config.enabled_for(f"{TARGET_REQUEST_ID_BASE}-deadbeef0") is False
    assert config.enabled_for(HTTP_TARGET_REQUEST_ID) is False
    with pytest.raises(ValueError, match="must not include a random suffix"):
        _config(
            tmp_path,
            target_request_id=f"{TARGET_REQUEST_ID_BASE}-deadbeef",
        )


@pytest.mark.cpu_test
def test_non_target_source_gate_rejects_localization_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = _ManifestTransport([])
    monkeypatch.setattr(pull_worker_module, "zmq_ctx", transport)
    worker = object.__new__(NixlPullConnectorWorker)
    worker._localization_config = _config(tmp_path)
    worker._localization_manifest_requests = {}
    remote = RemoteMeta(
        block_ids=(),
        host="127.0.0.1",
        port=5601,
        engine_id="prefill",
        request_id="chatcmpl-unrelated",
    )
    metadata = ReqMeta(
        local_block_ids=(),
        local_physical_block_ids=(),
        tp_size=4,
        remote=remote,
    )

    assert worker._localization_source_gate("chatcmpl-unrelated", metadata, ())
    assert len(transport.sockets) == 0
    assert worker._localization_manifest_requests == {}
    remote.request_id = f"{TARGET_REQUEST_ID_BASE}-1234abcd"
    with pytest.raises(LocalizationError, match="outside the decoder target"):
        worker._localization_source_gate("chatcmpl-unrelated", metadata, ())
    remote.request_id = "chatcmpl-unrelated"
    remote.p2d_run_id = "stray-run"
    with pytest.raises(LocalizationError, match="outside the decoder target"):
        worker._localization_source_gate("chatcmpl-unrelated", metadata, ())
    assert len(transport.sockets) == 0
    assert worker._localization_manifest_requests == {}


@pytest.mark.cpu_test
def test_source_gate_preserves_slow_retry_and_accepts_queued_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker, req_id, metadata, manifest = _source_gate_worker(tmp_path)
    transport = _ManifestTransport(
        [
            [_manifest_response(ManifestStatus.PENDING)],
            [
                zmq.Again(),
                zmq.Again(),
                _manifest_response(ManifestStatus.READY, (manifest,)),
            ],
        ]
    )
    monkeypatch.setattr(pull_worker_module, "zmq_ctx", transport)

    assert worker._localization_source_gate(req_id, metadata, (0,)) is False
    deadline = worker._localization_manifest_deadlines[req_id]
    assert transport.exit_counts == [1]
    assert req_id not in worker._localization_manifest_requests

    assert worker._localization_source_gate(req_id, metadata, (0,)) is False
    request_state = worker._localization_manifest_requests[req_id]
    assert worker._localization_source_gate(req_id, metadata, (0,)) is False

    assert worker._localization_manifest_deadlines[req_id] == deadline
    assert worker._localization_manifest_requests[req_id] is request_state
    assert worker._localization_waiting[req_id] is metadata
    assert len(transport.sockets) == 2
    assert transport.exit_counts == [1, 0]
    assert transport.sockets[1].send.call_count == 1
    worker._localization_manifest_deadlines[req_id] = time.perf_counter() - 1.0
    assert worker._localization_source_gate(req_id, metadata, (0,))

    assert worker._localization_expected_by_request[req_id] == {0: manifest}
    assert req_id not in worker._localization_manifest_requests
    assert req_id not in worker._localization_manifest_deadlines
    assert req_id not in worker._localization_waiting
    assert len(transport.sockets) == 2
    assert transport.exit_counts == [1, 1]
    socket = transport.sockets[1]
    assert socket.send.call_count == 1
    assert socket.recv.call_count == 3


@pytest.mark.cpu_test
def test_source_gate_deadline_closes_exchange_and_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker, req_id, metadata, _ = _source_gate_worker(tmp_path)
    transport = _ManifestTransport([[zmq.Again(), zmq.Again()]])
    monkeypatch.setattr(pull_worker_module, "zmq_ctx", transport)

    assert worker._localization_source_gate(req_id, metadata, (0,)) is False
    worker._localization_manifest_deadlines[req_id] = time.perf_counter() - 1.0

    with pytest.raises(LocalizationError, match="source-manifest gate timed out"):
        worker._localization_source_gate(req_id, metadata, (0,))

    assert req_id not in worker._localization_manifest_requests
    assert req_id not in worker._localization_manifest_deadlines
    assert req_id not in worker._localization_waiting
    assert transport.exit_counts == [1]


@pytest.mark.cpu_test
def test_aborted_source_gate_closes_exchange(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker, req_id, metadata, _ = _source_gate_worker(tmp_path)
    transport = _ManifestTransport([[zmq.Again()]])
    monkeypatch.setattr(pull_worker_module, "zmq_ctx", transport)
    assert worker._localization_source_gate(req_id, metadata, (0,)) is False

    worker._ready_requests = Queue()
    worker._reqs_to_process = set()
    worker._reqs_to_send = {}
    monkeypatch.setattr(worker, "_begin_transfer_phase", lambda: None)
    monkeypatch.setattr(worker, "_localization_capture_pre_read", lambda _: None)
    monkeypatch.setattr(worker, "_audit_retire", lambda _: None)
    monkeypatch.setattr(
        worker,
        "_localization_capture_source_rosters",
        lambda _: None,
    )
    monkeypatch.setattr(worker, "_localization_record_event", lambda **_: None)
    monkeypatch.setattr(worker, "_send_heartbeats", lambda _: None)
    monkeypatch.setattr(worker, "_drain_transfer_phase", lambda: None)
    monkeypatch.setattr(worker, "_record_transfer_decode_boundary", lambda: None)
    connector_metadata = pull_worker_module.NixlConnectorMetadata()
    connector_metadata.reqs_not_processed.add(req_id)

    worker.start_load_kv(connector_metadata)

    assert req_id not in worker._localization_manifest_requests
    assert req_id not in worker._localization_manifest_deadlines
    assert req_id not in worker._localization_waiting
    assert transport.exit_counts == [1]


@pytest.mark.cpu_test
def test_complete_trace_validates(tmp_path: Path) -> None:
    """A complete source, plan, four-stage, and terminal trace is accepted."""
    paths, _, _ = _write_trace(tmp_path)
    report = validate_localization_artifacts(paths)

    assert report.passed
    assert report.physical_pull_count == 1
    assert report.verified_pull_count == 1


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
    mixed_writer = LocalizationArtifactWriter(mixed_config, "other-engine", 0)
    mixed_writer.close()
    mixed_report = validate_localization_artifacts((*paths, mixed_writer.path))
    assert any("mixed" in error for error in mixed_report.errors)

    mixed_target_config = _config(
        tmp_path / "mixed-target",
        target_request_id="chatcmpl-unrelated",
    )
    mixed_target_writer = LocalizationArtifactWriter(
        mixed_target_config,
        "target-engine",
        0,
    )
    mixed_target_writer.close()
    mixed_target_report = validate_localization_artifacts(
        (*paths, mixed_target_writer.path)
    )
    assert any("mixed" in error for error in mixed_target_report.errors)


@pytest.mark.cpu_test
def test_source_gate_requires_exact_rank_set_and_lineage(tmp_path: Path) -> None:
    """The producer serves no duplicate, extra, stale, or mixed rank set."""
    config = _config(tmp_path)
    producer_request_id = f"{TARGET_REQUEST_ID_BASE}-1234abcd"
    rank_zero = _manifest(config, producer_request_id=producer_request_id)
    rank_one = _manifest(
        config,
        source_rank=1,
        registration_generation="registration-2",
        producer_request_id=producer_request_id,
    )
    request = (
        b"get_source_manifest_v1",
        config.run_id,
        config.transport_arm,
        "prefill",
        producer_request_id,
        1,
        0,
        (0, 1),
    )
    store = {
        (producer_request_id, 1): {
            0: rank_zero,
            1: rank_one,
        }
    }

    ready = resolve_source_manifest_request(request, config, store)
    assert ready.status is ManifestStatus.READY
    duplicate = resolve_source_manifest_request(
        (*request[:-1], (0, 0)),
        config,
        store,
    )
    assert duplicate.status is ManifestStatus.REJECTED
    off_target = resolve_source_manifest_request(
        (*request[:4], "chatcmpl-unrelated", *request[5:]),
        config,
        store,
    )
    assert off_target.status is ManifestStatus.REJECTED
    pending = resolve_source_manifest_request(
        request,
        config,
        {(producer_request_id, 1): {0: rank_zero}},
    )
    assert pending.status is ManifestStatus.PENDING
    subset = resolve_source_manifest_request(
        (*request[:-1], (0,)),
        config,
        store,
    )
    assert subset.status is ManifestStatus.READY
    assert tuple(manifest.source_rank for manifest in subset.manifests) == (0,)
    outside_topology = resolve_source_manifest_request(
        (*request[:-1], (0, 2)),
        config,
        store,
        {0, 1},
    )
    assert outside_topology.status is ManifestStatus.REJECTED
    stale = resolve_source_manifest_request(
        (*request[:6], 1, request[7]),
        config,
        store,
    )
    assert stale.status is ManifestStatus.REJECTED
    malformed_bool_rank = resolve_source_manifest_request(
        (*request[:-1], (0, True)),
        config,
        store,
    )
    assert malformed_bool_rank.status is ManifestStatus.REJECTED


@pytest.mark.cpu_test
def test_source_gate_rejects_mixed_rank_semantics(tmp_path: Path) -> None:
    """Ranks cannot jointly serve equal-sized but semantically different rows."""
    config = _config(tmp_path)
    rank_zero = _manifest(config)
    rank_one = _manifest(
        config,
        source_rank=1,
        registration_generation="registration-2",
        region=_region(
            base_address=0x200000,
            group_semantic_name="model.layers.60:transfer_region_0",
        ),
    )
    response = resolve_source_manifest_request(
        (
            b"get_source_manifest_v1",
            config.run_id,
            config.transport_arm,
            "prefill",
            TARGET_REQUEST_ID_BASE,
            1,
            0,
            (0, 1),
        ),
        config,
        {
            (TARGET_REQUEST_ID_BASE, 1): {
                0: rank_zero,
                1: rank_one,
            }
        },
    )

    assert response.status is ManifestStatus.REJECTED
    assert "semantic" in response.detail


@pytest.mark.cpu_test
def test_corruption_localizes_to_first_raw_staging_edge(tmp_path: Path) -> None:
    """Injected payload corruption names SOURCE_PRE to STAGING_RAW first."""
    paths, _, _ = _write_trace(
        tmp_path,
        corrupt_stage=IntegrityStage.STAGING_RAW,
    )
    report = validate_localization_artifacts(paths)

    assert not report.passed
    assert len(report.errors) == 0
    assert len(report.divergences) == 1
    assert report.divergences[0].edge == "source_pre->staging_raw"
    assert any("digest mismatch" in error for error in report.divergences[0].errors)


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
    source_writer = LocalizationArtifactWriter(config, "prefill", 0)
    for stage in (IntegrityStage.SOURCE_PRE, IntegrityStage.SOURCE_POST):
        source_writer.write(
            NixlSourceManifestRecord(
                record_type=NixlSourceManifestRecord.RECORD_TYPE,
                stage=stage,
                manifest=manifest,
            )
        )
    source_writer.close()
    decoder_writer = LocalizationArtifactWriter(config, "decoder", 0)
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

    report = validate_localization_artifacts(
        (source_writer.path, decoder_writer.path)
    )
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
    plan = _plan(manifest, local_region=baseline)
    errors = validate_localization_plan(plan, (manifest,))
    assert any("semantic region mismatch" in error for error in errors)


@pytest.mark.cpu_test
def test_plan_rejects_odd_trim_relative_half_mapping(tmp_path: Path) -> None:
    """Single-plane halves derive from absolute source positions after trimming."""
    config = _config(tmp_path)
    manifest = _manifest(config, blocks=(10, 11, 12))
    relative_halves = (
        NixlPlanPosition(
            group_index=0,
            source_position=1,
            remote_block_id=11,
            valid_token_extent=100,
            group_token_capacity=64,
            local_block_id=100,
            plane_index=0,
        ),
        NixlPlanPosition(
            group_index=0,
            source_position=2,
            remote_block_id=12,
            valid_token_extent=100,
            group_token_capacity=64,
            local_block_id=100,
            plane_index=1,
        ),
    )
    plan = _plan(
        manifest,
        selected_remote=(11, 12),
        selected_local=(100,),
        destination_planes=1,
        positions=relative_halves,
    )

    errors = validate_localization_plan(plan, (manifest,))
    assert any("destination mapping mismatch" in error for error in errors)


@pytest.mark.cpu_test
def test_plan_rejects_swapped_destinations_with_unchanged_digests(
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
            plane_index=-1,
        ),
        NixlPlanPosition(
            group_index=0,
            source_position=1,
            remote_block_id=11,
            valid_token_extent=100,
            group_token_capacity=64,
            local_block_id=100,
            plane_index=-1,
        ),
    )
    plan = _plan(manifest, positions=swapped)

    errors = validate_localization_plan(plan, (manifest,))
    assert any("destination mapping mismatch" in error for error in errors)


@pytest.mark.cpu_test
def test_plan_rejects_rank_slot_swap_against_topology_contract(
    tmp_path: Path,
) -> None:
    """A bijective rank permutation is still wrong if topology did not choose it."""
    config = _config(tmp_path)
    rank_zero = _manifest(config)
    rank_one = _manifest(
        config,
        source_rank=1,
        registration_generation="registration-2",
    )
    baseline = _plan(rank_zero)
    swapped = msgspec.structs.replace(
        baseline,
        registration_generations=(
            rank_zero.registration_generation,
            rank_one.registration_generation,
        ),
        source_manifest_digests=(
            rank_zero.manifest_digest,
            rank_one.manifest_digest,
        ),
        source_ranks=(0, 1),
        rank_slots=(1, 0),
        rank_slot_contract=((0, 0), (1, 1)),
        regions=(rank_zero.regions, rank_one.regions),
        staging_size=len(baseline.transfer_order) * 2 * 8,
    )

    errors = validate_localization_plan(swapped, (rank_zero, rank_one))
    assert any("violates rank-slot contract" in error for error in errors)


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
