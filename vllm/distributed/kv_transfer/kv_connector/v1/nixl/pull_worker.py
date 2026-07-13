# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pull-specific (READ) worker-side logic for the NIXL connector."""

import os
import time
import traceback
from typing import TYPE_CHECKING

import msgspec
import numpy as np
import zmq

from vllm.distributed.kv_transfer.integrity import IntegrityIdentity
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import (
    NixlBaseConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlConnectorMetadata,
    ReqMeta,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import (
    ReadSpec,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.utils import zmq_ctx
from vllm.distributed.kv_transfer.nixl_localization import (
    GET_SOURCE_MANIFEST_MSG,
    LocalizationError,
    ManifestStatus,
    NixlEventRecord,
    NixlPlanPosition,
    NixlPlanRecord,
    NixlSourceManifestResponse,
    compute_source_manifest_digest,
    locate_subsequence,
    validate_source_manifest_structure,
)
from vllm.logger import init_logger
from vllm.utils.network_utils import make_zmq_path

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


class NixlPullConnectorWorker(NixlBaseConnectorWorker):
    """Pull-specific (READ) worker logic."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        engine_id: str,
        kv_cache_config: "KVCacheConfig",
    ) -> None:
        super().__init__(vllm_config, engine_id, kv_cache_config)
        if self._phase_separate_transfer_decode and not self.coalesce_pull:
            raise ValueError("phase_separate_transfer_decode requires coalesced pull")
        if self._phase_separate_transfer_decode and not self._no_stock_dma():
            raise ValueError(
                "phase_separate_transfer_decode requires stock DMA to be disabled"
            )

    def start_load_kv(self, metadata: NixlConnectorMetadata) -> None:
        """Start and account for receive work required by this model step.

        :param metadata: Scheduler metadata for transfers entering this step.
        """
        self._begin_transfer_phase()
        self._localization_capture_pre_read(metadata.reqs_in_batch)
        self._audit_retire(metadata)
        self._localization_capture_source_rosters(metadata.source_integrity_rosters)
        for req_id, meta in metadata.reqs_to_recv.items():
            meta.local_physical_block_ids = self._logical_to_kernel_block_ids(
                meta.local_block_ids
            )
            assert meta.remote is not None
            # Remote block IDs are kept logical here; expanded in
            # _read_blocks_for_req using the remote engine's phys ratio.
            remote_engine_id = meta.remote.engine_id
            logger.debug(
                "start_load_kv for request %s from remote engine %s. "
                "Num local_block_ids: %s. Num remote_block_ids: %s. ",
                req_id,
                remote_engine_id,
                len(meta.local_physical_block_ids),
                len(meta.remote.block_ids),
            )
            # always store metadata for failure recovery
            self._recving_metadata[req_id] = meta
            if remote_engine_id not in self._remote_agents:
                # Initiate handshake with remote engine to exchange metadata.
                with self._handshake_lock:
                    if remote_engine_id not in self._remote_agents:
                        self._background_nixl_handshake(req_id, remote_engine_id, meta)
                        continue

            # Handshake already completed, start async read xfer.
            self._read_blocks_for_req(req_id, meta)

        # Start transfers for requests whose handshakes have now finished.
        while not self._ready_requests.empty():
            self._read_blocks_for_req(*self._ready_requests.get_nowait())

        # Keep around the requests that have been part of a batch. This is
        # needed because async scheduling pushes the misalignment between the
        # moment in which requests expiration is set (P side) and the moment in
        # which blocks are read from D. As P can now more easily lag behind D
        # while processing the next batch, we make sure to only set an
        # expiration for requests that have not been read from D yet.
        for req_id in metadata.reqs_in_batch:
            self._reqs_to_process.add(req_id)

        # Remove all requests that are not to be processed (eg aborted).
        for req_id in metadata.reqs_not_processed:
            self._reqs_to_process.discard(req_id)
            waiting_meta = self._localization_waiting.get(req_id)
            waiting_remote = waiting_meta.remote if waiting_meta is not None else None
            self._localization_record_event(
                code="REQUEST_ABORTED",
                evidentiary=False,
                child_request_id=req_id,
                producer_engine_id=(
                    waiting_remote.engine_id if waiting_remote is not None else None
                ),
                producer_request_id=(
                    waiting_remote.request_id if waiting_remote is not None else None
                ),
                detail="request aborted before verified pre-read",
            )
            self._localization_waiting.pop(req_id, None)
            self._localization_manifest_deadlines.pop(req_id, None)
            self._localization_expected_by_request.pop(req_id, None)
            # We should never get an abort after setting an expiry timer
            assert req_id not in self._reqs_to_send

        self._service_localization_waiting()

        # Add to requests that are waiting to be read and track expiration.
        for req_id, expiration_time in metadata.reqs_to_send.items():
            if req_id in self._reqs_to_process:
                self._reqs_to_send[req_id] = expiration_time

        # Send heartbeats to P-side engines to keep KV blocks alive while
        # requests sit in the D scheduler WAITING queue.
        self._send_heartbeats(metadata)
        self._drain_transfer_phase()
        self._record_transfer_decode_boundary()

    def _read_blocks_for_req(self, req_id: str, meta: ReqMeta):
        assert meta.remote is not None and self.transfer_topo is not None
        localization_enabled = self._localization_config.enabled_for(req_id)
        engine_id = meta.remote.engine_id
        # Update last activity from this remote. Mind that cleanup is done on main
        # thread (this one), so we don't race on this structure.
        self._engine_last_active[engine_id] = time.perf_counter()
        plan = self.tp_mappings[engine_id]
        remote_info = self.transfer_topo.get_engine_info(engine_id)
        tp_ratio = self.transfer_topo.tp_ratio(remote_info.remote_tp_size)

        if (
            localization_enabled
            and sum(len(group) for group in meta.local_physical_block_ids) == 0
        ):
            if self._localization_writer is None:
                raise LocalizationError("enabled localization has no artifact writer")
            if req_id not in self._localization_zero_recorded:
                self._localization_writer.write(
                    NixlEventRecord(
                        record_type=NixlEventRecord.RECORD_TYPE,
                        schema_version=IntegrityIdentity.SCHEMA_VERSION,
                        run_id=self._localization_config.run_id,
                        transport_arm=self._localization_config.transport_arm,
                        code="NON_EVIDENTIARY_ZERO_BYTE",
                        evidentiary=False,
                        producer_engine_id=meta.remote.engine_id,
                        producer_request_id=meta.remote.request_id,
                        child_request_id=req_id,
                        observer_engine_id=self.engine_id,
                        observer_rank=self.tp_rank,
                        detail=(
                            "full-prefix hits require prior VERIFIED provenance and "
                            "are excluded from this localization protocol"
                        ),
                        created_ns=time.time_ns(),
                    )
                )
                self._localization_zero_recorded.add(req_id)
                self._localization_terminal_recorded.add(req_id)
            if self._localization_config.strict_zero_byte:
                raise LocalizationError(
                    f"full-prefix request {req_id} is non-evidentiary in "
                    "strict localization mode"
                )

        if (
            self._localization_source_gate(
                req_id,
                meta,
                tuple(int(rank) for rank in plan.all_source_ranks),
            )
            is False
        ):
            return

        if (
            meta.remote.request_id in self._released_rids
            and sum(len(g) for g in meta.local_physical_block_ids) > 0
        ):
            # Full-prefix-hit requests (empty local ids) read nothing
            # and are exempt from the fence.
            logger.error(
                "[release-fence] refusing pull for %s: remote request %s "
                "already released; failing (router retries with a fresh "
                "prefill).",
                req_id,
                meta.remote.request_id,
            )
            self._handle_failed_transfer(req_id, None)
            return

        meta.remote.block_ids = self._logical_to_remote_kernel_block_ids(
            meta.remote.block_ids,
            remote_info.remote_physical_blocks_per_logical,
        )
        remote_block_ids = meta.remote.block_ids
        local_block_ids = meta.local_physical_block_ids
        num_groups = len(local_block_ids)
        read_specs = [
            ReadSpec(
                remote_rank=rank,
                local_block_ids=[
                    list(local_block_ids[g])
                    if rank in plan.source_ranks_per_group[g]
                    else []
                    for g in range(num_groups)
                ],
                remote_block_ids=[
                    list(remote_block_ids[g])
                    if rank in plan.source_ranks_per_group[g]
                    else []
                    for g in range(num_groups)
                ],
            )
            for rank in plan.all_source_ranks
        ]

        # D may have to perform multiple reads from different remote ranks.
        # MLA opt: when P TP > D TP, only a single read is executed for
        # the first remote rank (cache is duplicated)..
        if self.use_mla and tp_ratio < 0:
            assert len(read_specs) == 1

        if (
            localization_enabled
            and sum(len(group) for group in local_block_ids) == 0
        ):
            result = self._coalesced_read_request(req_id, meta, read_specs)
            if result != "posted":
                raise LocalizationError(
                    f"zero-byte request {req_id} did not complete its release path"
                )
            return

        # Coalesced pull fast path (see base_worker state comment): all
        # gates must hold. A staging-pool miss parks the request FIFO until
        # completed plans free staging. Localization forbids the stock path
        # because it has no staging or destination observation points.
        if self._coalesce_gate(engine_id, tp_ratio, read_specs):
            res = self._coalesced_read_request(req_id, meta, read_specs)
            if res == "posted":
                return
            if res == "defer":
                self._coalesce_pending.append((req_id, meta, read_specs))
                logger.debug(
                    "coalesced pull: parked %s for staging (%s queued)",
                    req_id,
                    len(self._coalesce_pending),
                )
                return

        if (
            localization_enabled
            or any(self._sp_group_flags())
            or self._no_stock_dma()
        ):
            # F2b: the stock per-descriptor path cannot express the
            # single-plane K-half/2:1 mapping -- running it would write
            # a dual layout into a single-plane cache. Fail the request
            # (kv_load_failure_policy=fail) rather than corrupt.
            # Phase 21 (VLLM_GEMMA4_NIXL_NO_STOCK_DMA=1): the stock path
            # also DMAs into local KV offsets past the NIXL/UCX
            # large-offset defect threshold (block ids >= ~32768) and
            # silently corrupts; fail instead -- the router retries via
            # a fresh prefill->decode pass.
            logger.error(
                "coalesced pull unavailable for %s (sp_groups=%s "
                "no_stock_dma=%s); failing the request instead of the "
                "stock path.",
                req_id,
                any(self._sp_group_flags()),
                self._no_stock_dma(),
            )
            self._handle_failed_transfer(req_id, None)
            return

        self._stock_read_specs(req_id, meta, read_specs)

    def _service_localization_waiting(self) -> None:
        """Retry requests parked before the source-manifest gate."""
        if len(self._localization_waiting) == 0:
            return
        for req_id, meta in list(self._localization_waiting.items()):
            self._read_blocks_for_req(req_id, meta)

    def _localization_source_gate(
        self,
        req_id: str,
        meta: ReqMeta,
        required_ranks: tuple[int, ...],
    ) -> bool:
        """Require an all-rank P snapshot before posting any one-sided read.

        :param req_id: Decoder child request identifier.
        :param meta: Request transfer metadata.
        :param required_ranks: P ranks whose bytes this D worker will read.
        :returns: ``True`` only after an authoritative source manifest is ready.
        :raises LocalizationError: On timeout, rejection, or protocol mismatch.
        """
        assert meta.remote is not None
        remote = meta.remote
        remote_lineage = (
            remote.p2d_run_id,
            remote.p2d_transport_arm,
            remote.p2d_offer_generation,
            remote.p2d_iteration,
        )
        if self._localization_config.enabled_for(req_id) is False:
            if self._localization_config.enabled_for(remote.request_id) or any(
                value is not None for value in remote_lineage
            ):
                raise LocalizationError(
                    f"request {req_id} carries a target producer or localization "
                    "lineage outside the decoder target"
                )
            return True
        if (
            len(required_ranks) == 0
            or len(set(required_ranks)) != len(required_ranks)
            or any(type(rank) is not int or rank < 0 for rank in required_ranks)
        ):
            raise LocalizationError(
                f"request {req_id} has invalid required source ranks"
            )
        if req_id in self._localization_expected_by_request:
            self._localization_waiting.pop(req_id, None)
            return True
        if (
            self._localization_config.enabled_for(remote.request_id) is False
            or remote.p2d_run_id != self._localization_config.run_id
            or remote.p2d_transport_arm != self._localization_config.transport_arm
            or remote.p2d_offer_generation is None
            or remote.p2d_iteration is None
        ):
            raise LocalizationError(
                f"request {req_id} has incomplete or mismatched localization lineage"
            )

        deadline = self._localization_manifest_deadlines.setdefault(
            req_id,
            time.perf_counter() + self._localization_config.manifest_timeout_s,
        )
        path = make_zmq_path("tcp", remote.host, remote.port)
        request = (
            GET_SOURCE_MANIFEST_MSG,
            self._localization_config.run_id,
            self._localization_config.transport_arm,
            remote.engine_id,
            remote.request_id,
            remote.p2d_offer_generation,
            remote.p2d_iteration,
            required_ranks,
        )
        with zmq_ctx(zmq.REQ, path) as sock:
            sock.setsockopt(zmq.RCVTIMEO, 1000)
            sock.send(msgspec.msgpack.encode(request))
            try:
                response_bytes = sock.recv()
            except zmq.Again as error:
                raise LocalizationError(
                    f"source-manifest side channel timed out for {req_id}"
                ) from error
        try:
            response = msgspec.msgpack.decode(
                response_bytes,
                type=NixlSourceManifestResponse,
            )
        except (msgspec.DecodeError, msgspec.ValidationError) as error:
            raise LocalizationError(
                f"source-manifest response is malformed for {req_id}"
            ) from error
        if response.status is ManifestStatus.PENDING:
            if time.perf_counter() >= deadline:
                raise LocalizationError(
                    f"source-manifest gate timed out for decoder request {req_id}"
                )
            self._localization_waiting[req_id] = meta
            return False
        if response.status is not ManifestStatus.READY:
            raise LocalizationError(
                f"source-manifest gate rejected {req_id}: {response.detail}"
            )
        if len(response.manifests) != len(required_ranks):
            raise LocalizationError(
                f"source-manifest gate returned incomplete ranks for {req_id}"
            )
        if response.schema_version != IntegrityIdentity.SCHEMA_VERSION:
            raise LocalizationError(
                f"source-manifest response schema mismatch for {req_id}"
            )
        manifests = {manifest.source_rank: manifest for manifest in response.manifests}
        if len(manifests) != len(response.manifests):
            raise LocalizationError(
                f"source-manifest gate returned duplicate ranks for {req_id}"
            )
        if set(manifests) != set(required_ranks):
            raise LocalizationError(
                f"source-manifest gate returned wrong ranks for {req_id}"
            )
        for manifest in manifests.values():
            structure_errors = validate_source_manifest_structure(manifest)
            if len(structure_errors) > 0:
                raise LocalizationError(
                    f"invalid source manifest for decoder request {req_id}: "
                    f"{structure_errors[:8]}"
                )
            if (
                manifest.schema_version != IntegrityIdentity.SCHEMA_VERSION
                or manifest.run_id != self._localization_config.run_id
                or manifest.transport_arm != self._localization_config.transport_arm
                or manifest.producer_engine_id != remote.engine_id
                or manifest.producer_request_id != remote.request_id
                or manifest.offer_generation != remote.p2d_offer_generation
                or manifest.iteration != remote.p2d_iteration
                or manifest.registration_generation
                != self._remote_registration_generations[remote.engine_id][
                    manifest.source_rank
                ]
                or manifest.regions
                != self._remote_regions[remote.engine_id][manifest.source_rank]
            ):
                raise LocalizationError(
                    f"source-manifest lineage mismatch for decoder request {req_id}"
                )
            if compute_source_manifest_digest(manifest) != manifest.manifest_digest:
                raise LocalizationError(
                    f"source-manifest digest mismatch for decoder request {req_id}"
                )
        self._localization_expected_by_request[req_id] = manifests
        self._localization_manifest_deadlines.pop(req_id, None)
        self._localization_waiting.pop(req_id, None)
        logger.warning(
            "[p2d-localize] source gate READY child=%s producer=%s "
            "ranks=%s manifests=%s observer=true",
            req_id,
            remote.request_id,
            required_ranks,
            tuple(manifests[rank].manifest_digest.hex() for rank in sorted(manifests)),
        )
        return True

    def _localization_validate_plan_manifests(
        self,
        req_id: str,
        meta: ReqMeta,
        read_specs: list[ReadSpec],
        raw_remote_groups: tuple[tuple[int, ...], ...],
        region_lengths: tuple[int, ...],
    ) -> None:
        """Bind the exact unsorted transfer roster to all P-rank manifests.

        :param req_id: Decoder child request identifier.
        :param meta: Request transfer metadata.
        :param read_specs: Per-source-rank transfer specifications.
        :param raw_remote_groups: Exact pre-sort physical source groups.
        :param region_lengths: Exact source row lengths in registration order.
        :raises LocalizationError: If any manifest describes different bytes.
        """
        assert meta.remote is not None
        manifests = self._localization_expected_by_request.get(req_id)
        if manifests is None:
            raise LocalizationError(f"request {req_id} has no gated source manifest")
        if len(self._region_descriptors) != len(region_lengths):
            raise LocalizationError(
                f"request {req_id} local region descriptor cardinality differs"
            )
        destination_group_planes = tuple(
            1 if flag else 2 for flag in self._sp_group_flags()
        )
        if len(destination_group_planes) != len(raw_remote_groups):
            raise LocalizationError(
                f"request {req_id} destination plane-contract cardinality differs"
            )
        local_owned_groups = {
            group_index
            for region in self._region_descriptors
            for group_index in region.group_indices
        }
        if local_owned_groups != set(range(len(raw_remote_groups))):
            raise LocalizationError(
                f"request {req_id} local semantic regions do not cover all groups"
            )
        for spec in read_specs:
            manifest = manifests.get(spec.remote_rank)
            if manifest is None:
                raise LocalizationError(
                    f"request {req_id} lacks P-rank {spec.remote_rank} manifest"
                )
            if manifest.block_ids != raw_remote_groups:
                raise LocalizationError(
                    f"request {req_id} source roster differs on rank {spec.remote_rank}"
                )
            reference_manifest = manifests[read_specs[0].remote_rank]
            if (
                manifest.valid_token_extent != reference_manifest.valid_token_extent
                or manifest.group_token_capacities
                != reference_manifest.group_token_capacities
                or manifest.source_group_planes
                != reference_manifest.source_group_planes
                or len(manifest.group_token_capacities) != len(destination_group_planes)
            ):
                raise LocalizationError(
                    f"request {req_id} source semantic extent differs on rank "
                    f"{spec.remote_rank}"
                )
            if manifest.region_lengths != region_lengths:
                raise LocalizationError(
                    f"request {req_id} source region layout differs on rank "
                    f"{spec.remote_rank}"
                )
            if len(manifest.regions) != len(self._region_descriptors):
                raise LocalizationError(
                    f"request {req_id} source/local region count differs on rank "
                    f"{spec.remote_rank}"
                )
            for source_region, local_region in zip(
                manifest.regions,
                self._region_descriptors,
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
                    raise LocalizationError(
                        f"request {req_id} semantic region identity differs on "
                        f"rank {spec.remote_rank}"
                    )
                if local_region.row_bytes != len(read_specs) * source_region.row_bytes:
                    raise LocalizationError(
                        f"request {req_id} local/source row geometry differs on "
                        f"rank {spec.remote_rank}"
                    )
                for role, region in (
                    ("source", source_region),
                    ("local", local_region),
                ):
                    if (
                        len(region.shape) == 0
                        or len(region.shape) != len(region.strides)
                        or region.registered_bytes != region.shape[0] * region.row_bytes
                        or region.strides[0] * region.element_size_bytes
                        != region.row_bytes
                    ):
                        raise LocalizationError(
                            f"request {req_id} {role} region geometry is not "
                            f"row canonical on rank {spec.remote_rank}"
                        )

    def _no_stock_dma(self) -> bool:
        """Return whether the known-corrupt stock DMA path is disabled.

        Stock DMA is disabled by default because NIXL/UCX corrupts local KV
        offsets beyond its large-offset threshold. Explicitly setting the
        variable to ``0`` is reserved for controlled small-pool diagnosis.

        :returns: Whether requests must fail instead of using stock DMA.
        """
        return os.environ.get("VLLM_GEMMA4_NIXL_NO_STOCK_DMA", "1") == "1"

    def _stock_read_specs(
        self, req_id: str, meta: ReqMeta, read_specs: list[ReadSpec]
    ) -> None:
        """The stock per-descriptor pull for one request's read specs
        (extracted from _read_blocks_for_req so the pending-queue
        servicer can also route a request here)."""
        assert meta.remote is not None and self.transfer_topo is not None
        dst_engine_id = meta.remote.engine_id
        remote_info = self.transfer_topo.get_engine_info(dst_engine_id)
        tp_ratio = self.transfer_topo.tp_ratio(remote_info.remote_tp_size)

        for i, spec in enumerate(read_specs):
            remote_block_size = remote_info.remote_block_size
            logger.debug(
                "Remote agent %s available, calling _read_blocks"
                " on remote rank %s with remote block size %s for req %s",
                meta.remote.engine_id,
                spec.remote_rank,
                remote_block_size,
                req_id,
            )
            # Get side handles.
            if tp_ratio < 0 and not self.use_mla:
                assert remote_block_size == self.block_size
                # Remote tp_size > local tp_size: we must perform multiple
                # reads. Get the memory chunk onto which we will write to.
                local_xfer_side_handle = self.src_xfer_handles_by_tp_ratio[tp_ratio][i]
            else:
                # Single read from remote, we write to the whole memory region.
                # Also handle remote block size different from local block size.
                local_xfer_side_handle = self.src_xfer_handles_by_block_size[
                    remote_block_size
                ]

            # Destination handle: remote_engine_id -> remote_rank -> handle.
            remote_xfer_side_handle = self.dst_xfer_side_handles[meta.remote.engine_id][
                spec.remote_rank
            ]

            self._read_blocks(
                read_spec=spec,
                request_id=req_id,
                dst_engine_id=meta.remote.engine_id,
                remote_request_id=meta.remote.request_id,
                local_xfer_side_handle=local_xfer_side_handle,
                remote_xfer_side_handle=remote_xfer_side_handle,
                expected_consumers=meta.remote.expected_consumers,
            )

        if self.use_mla and tp_ratio < 0 and read_specs:
            # ..but we still need to notify the other remote ranks that we
            # have the blocks we need so they can update the request state.
            notif_id = (
                f"{meta.remote.request_id}:{self.world_size}"
                f":{meta.remote.expected_consumers}"
            ).encode()
            remote_agents = self._remote_agents[meta.remote.engine_id]
            for rank_to_notify, agent in remote_agents.items():
                if rank_to_notify != read_specs[0].remote_rank:
                    self.nixl_wrapper.send_notif(agent, notif_msg=notif_id)

    # ------------------------------------------------------------------
    # Coalesced pull
    # ------------------------------------------------------------------

    def _coalesce_gate(
        self, engine_id: str, tp_ratio: int, read_specs: list[ReadSpec]
    ) -> bool:
        """All conditions under which the coalesced path is proven
        equivalent to the stock path. Anything else -> stock."""
        if not self.coalesce_pull:
            return False
        if not (
            tp_ratio < 0
            and not self.use_mla
            and not self.use_host_buffer
            and not self._has_mamba
        ):
            return False
        if self._physical_blocks_per_logical_kv_block != 1:
            return False
        if any(self._region_is_mla) or not self._region_tensors:
            return False
        assert self.transfer_topo is not None
        remote_info = self.transfer_topo.get_engine_info(engine_id)
        if self.transfer_topo.block_size_ratio(remote_info.remote_block_size) != 1:
            return False
        layout = self._remote_layout.get(engine_id)
        if not layout or any(s.remote_rank not in layout for s in read_specs):
            return False
        # uniform shard participation: every rank reads the same lists
        # (pure SPLIT full-attention groups)
        spec0 = read_specs[0]
        for s in read_specs[1:]:
            if (
                s.local_block_ids != spec0.local_block_ids
                or s.remote_block_ids != spec0.remote_block_ids
            ):
                return False
        blens = layout[spec0.remote_rank][0]
        if len(blens) != len(self._region_tensors):
            return False
        n_ranks = len(read_specs)
        for i, blen in enumerate(blens):
            # local block row must be exactly the |tp_ratio| remote
            # shard blocks side by side, K/V halves equal
            if blen % 2 != 0 or self.block_len_per_layer[i] != n_ranks * blen:
                return False
        if not self._coalesce_region_rows():
            return False
        return self._staging_init()

    def _coalesce_service_pending(self) -> None:
        """Start transfers for requests parked on the staging pool
        (FIFO -- strict head-of-line, so a large request cannot be
        starved by smaller ones slipping past it)."""
        while self._coalesce_pending:
            req_id, meta, read_specs = self._coalesce_pending[0]
            res = self._coalesced_read_request(req_id, meta, read_specs)
            if res == "defer":
                break
            self._coalesce_pending.popleft()
            if res == "stock":
                if (
                    self._localization_config.enabled_for(req_id)
                    or self._no_stock_dma()
                ):
                    logger.error(
                        "coalesced drain: %s not expressible; failing "
                        "instead of the stock path.",
                        req_id,
                    )
                    self._handle_failed_transfer(req_id, None)
                    continue
                self._stock_read_specs(req_id, meta, read_specs)

    def _coalesced_read_request(
        self, req_id: str, meta: ReqMeta, read_specs: list[ReadSpec]
    ) -> str:
        """Post one whole-request READ per remote rank into owned staging.

        Any failure after allocation raises :class:`StagingSafetyError` and
        leaves uncertain native writers and their range owned until process
        replacement.

        :param req_id: Decoder request identifier.
        :param meta: Complete transfer metadata.
        :param read_specs: Per-source-rank transfer specifications.
        :returns: ``posted``, ``defer``, or ``stock`` before native failure.
        """
        assert meta.remote is not None and self.transfer_topo is not None
        localization_enabled = self._localization_config.enabled_for(req_id)
        engine_id = meta.remote.engine_id
        remote_info = self.transfer_topo.get_engine_info(engine_id)
        spec0 = read_specs[0]
        raw_remote_groups = (
            tuple(
                tuple(int(block_id) for block_id in group)
                for group in spec0.remote_block_ids
            )
            if localization_enabled
            else ()
        )
        local_ids, remote_ids = self._apply_prefix_caching(
            [list(g) for g in spec0.local_block_ids],
            [list(g) for g in spec0.remote_block_ids],
            remote_info.remote_physical_blocks_per_logical,
        )
        notif_id = (
            f"{meta.remote.request_id}:{self.world_size}"
            f":{meta.remote.expected_consumers}"
        ).encode()
        if len(local_ids) == 0 or sum(len(g) for g in local_ids) == 0:
            # full prefix hit: just release P's blocks on every rank
            for s in read_specs:
                agent = self._remote_agents[engine_id][s.remote_rank]
                try:
                    self.nixl_wrapper.send_notif(agent, notif_msg=notif_id)
                except Exception:
                    logger.error(
                        "full-prefix release notification failed for %s rank %s\n%s",
                        req_id,
                        s.remote_rank,
                        traceback.format_exc(),
                    )
                    self.xfer_stats.record_failed_notification()
            self._localization_expected_by_request.pop(req_id, None)
            self._localization_manifest_deadlines.pop(req_id, None)
            self._localization_waiting.pop(req_id, None)
            return "posted"

        # HMA broadcast semantics (see _compute_desc_ids): every group's
        # blocks are transferred across every region, position-ordered.
        # F2b single-plane groups (kv_planes==1): local blocks hold 2x
        # remote tokens, so remote block k of the request expands to
        # scatter destination (local_ids[k//2], half k%2); the scatter
        # keeps only the K half for those positions (sp_half >= 0).
        sp_flags = self._sp_group_flags()
        lpos_l, rpos_l, half_l = [], [], []
        group_l, source_position_l = [], []
        for gi in range(len(local_ids)):
            lg = np.asarray(local_ids[gi], dtype=np.int64)
            rg = np.asarray(remote_ids[gi], dtype=np.int64)
            if localization_enabled:
                source_start = locate_subsequence(
                    list(raw_remote_groups[gi]),
                    [int(block_id) for block_id in rg],
                )
            if sp_flags[gi]:
                if len(rg) > 2 * len(lg):
                    return "stock"
                k = np.arange(len(rg))
                lpos_l.append(lg[k // 2])
                rpos_l.append(rg)
                half_l.append((k % 2).astype(np.int8))
            else:
                if len(lg) != len(rg):
                    return "stock"
                lpos_l.append(lg)
                rpos_l.append(rg)
                half_l.append(np.full(len(lg), -1, dtype=np.int8))
            if localization_enabled:
                group_l.append(np.full(len(rg), gi, dtype=np.int32))
                source_position_l.append(
                    np.arange(source_start, source_start + len(rg), dtype=np.int64)
                )
        lpos = np.concatenate(lpos_l) if lpos_l else np.zeros(0, dtype=np.int64)
        rpos = np.concatenate(rpos_l) if rpos_l else np.zeros(0, dtype=np.int64)
        halves = np.concatenate(half_l) if half_l else np.zeros(0, dtype=np.int8)
        # Transfer order is free (the (remote, local) pairing is what
        # matters): sort by remote id so run detection harvests all the
        # adjacency the remote pool still has. The scatter index (lpos)
        # is permuted identically, so placement is unchanged.
        order = np.argsort(rpos, kind="stable")
        rpos = rpos[order]
        lpos = lpos[order]
        halves = halves[order]
        if localization_enabled:
            group_ids = np.concatenate(group_l)[order]
            source_positions = np.concatenate(source_position_l)[order]
        n_pos = len(lpos)
        n_ranks = len(read_specs)
        blens = self._remote_layout[engine_id][spec0.remote_rank][0]
        n_regions = len(blens)
        if localization_enabled:
            self._localization_validate_plan_manifests(
                req_id,
                meta,
                read_specs,
                raw_remote_groups,
                tuple(int(length) for length in blens),
            )
            manifests = self._localization_expected_by_request[req_id]
            semantic_manifest = manifests[int(read_specs[0].remote_rank)]
        else:
            semantic_manifest = None
        plan = self.tp_mappings[engine_id]
        source_ranks = tuple(int(spec.remote_rank) for spec in read_specs)
        if len(set(source_ranks)) != n_ranks:
            raise LocalizationError("coalesced plan contains duplicate source ranks")
        if any(rank not in plan.rank_to_attention_slot for rank in source_ranks):
            raise LocalizationError("coalesced plan is missing a source-rank slot")
        slots = [int(plan.rank_to_attention_slot[rank]) for rank in source_ranks]
        if sorted(slots) != list(range(n_ranks)):
            raise LocalizationError(
                "coalesced source-rank slots must be a complete bijection"
            )
        transfer_order: tuple[NixlPlanPosition, ...] = ()
        if localization_enabled:
            assert semantic_manifest is not None
            transfer_order = tuple(
                NixlPlanPosition(
                    group_index=int(group_ids[index]),
                    source_position=int(source_positions[index]),
                    remote_block_id=int(rpos[index]),
                    valid_token_extent=semantic_manifest.valid_token_extent,
                    group_token_capacity=semantic_manifest.group_token_capacities[
                        int(group_ids[index])
                    ],
                    local_block_id=int(lpos[index]),
                    plane_index=int(halves[index]),
                )
                for index in range(n_pos)
            )
            for position in transfer_order:
                if sp_flags[position.group_index]:
                    expected_half = position.source_position % 2
                    if position.plane_index != expected_half:
                        raise LocalizationError(
                            "single-plane destination half is not derived from "
                            "the absolute source position"
                        )
                elif position.plane_index != -1:
                    raise LocalizationError(
                        "dual-plane transfer position carries a destination half"
                    )
        # maximal consecutive remote-id runs: (start_id, count, pos0)
        runs: list[tuple[int, int, int]] = []
        k0 = 0
        for k in range(1, n_pos + 1):
            if k == n_pos or rpos[k] != rpos[k - 1] + 1:
                runs.append((int(rpos[k0]), k - k0, k0))
                k0 = k

        region_off = [0] * n_regions
        acc = 0
        for i in range(n_regions):
            region_off[i] = acc
            acc += n_pos * n_ranks * blens[i]
        scatter_geometry: dict[str, object] = dict(
            size=acc,
            n_pos=n_pos,
            n_ranks=n_ranks,
            blens=blens,
            region_off=region_off,
            lpos=lpos.tolist(),
            sp_half=halves.tolist(),
            slots=slots,
        )
        if localization_enabled:
            scatter_geometry.update(
                source_ranks=list(source_ranks),
                transfer_order=transfer_order,
                producer_engine_id=engine_id,
                producer_request_id=meta.remote.request_id,
                offer_generation=meta.remote.p2d_offer_generation,
                iteration=meta.remote.p2d_iteration,
            )
        if self._audit_enabled:
            tail_exclude = self._audit_tail_exclude
            audit_rows: list[int] = []
            for group_index in sorted(self._audit_groups):
                if group_index >= len(local_ids) or sp_flags[group_index]:
                    continue
                group_rows = list(local_ids[group_index])
                if len(group_rows) > tail_exclude:
                    audit_rows.extend(int(row) for row in group_rows[:-tail_exclude])
            scatter_geometry["audit_rows"] = audit_rows
        if acc > self.coalesce_staging_mb * 1024 * 1024:
            logger.warning(
                "coalesced pull: request %s needs %sMB staging (pool %sMB); stock path",
                req_id,
                acc >> 20,
                self.coalesce_staging_mb,
            )
            return "stock"
        ownership = self._create_coalesced_plan(
            req_id,
            acc,
            source_ranks,
            engine_id,
            scatter_geometry,
        )
        if ownership is None:
            return "defer"
        off = ownership.lease.offset

        if localization_enabled:
            try:
                if self._localization_writer is None:
                    raise LocalizationError(
                        "enabled localization has no artifact writer"
                    )
                self._localization_writer.write(
                    NixlPlanRecord(
                        record_type=NixlPlanRecord.RECORD_TYPE,
                        schema_version=IntegrityIdentity.SCHEMA_VERSION,
                        run_id=self._localization_config.run_id,
                        transport_arm=self._localization_config.transport_arm,
                        producer_engine_id=engine_id,
                        producer_request_id=meta.remote.request_id,
                        registration_generations=tuple(
                            self._remote_registration_generations[engine_id][
                                int(spec.remote_rank)
                            ]
                            for spec in read_specs
                        ),
                        source_manifest_digests=tuple(
                            self._localization_expected_by_request[req_id][
                                int(spec.remote_rank)
                            ].manifest_digest
                            for spec in read_specs
                        ),
                        offer_generation=int(meta.remote.p2d_offer_generation),
                        iteration=int(meta.remote.p2d_iteration),
                        child_request_id=req_id,
                        observer_engine_id=self.engine_id,
                        observer_rank=self.tp_rank,
                        source_ranks=source_ranks,
                        rank_slots=tuple(int(slot) for slot in slots),
                        rank_slot_contract=tuple(
                            sorted(
                                (
                                    int(rank),
                                    int(slot),
                                )
                                for rank, slot in (plan.rank_to_attention_slot.items())
                            )
                        ),
                        destination_group_planes=tuple(
                            1 if flag else 2 for flag in sp_flags
                        ),
                        region_lengths=tuple(int(length) for length in blens),
                        regions=tuple(
                            self._remote_regions[engine_id][int(spec.remote_rank)]
                            for spec in read_specs
                        ),
                        local_regions=self._region_descriptors,
                        raw_remote_groups=raw_remote_groups,
                        selected_remote_groups=tuple(
                            tuple(int(block_id) for block_id in group)
                            for group in remote_ids
                        ),
                        selected_local_groups=tuple(
                            tuple(int(block_id) for block_id in group)
                            for group in local_ids
                        ),
                        transfer_order=transfer_order,
                        runs=tuple(runs),
                        region_offsets=tuple(region_off),
                        staging_offset=off,
                        staging_size=acc,
                    )
                )
            except Exception as error:
                stacktrace = traceback.format_exc()
                self._fail_coalesced_plan(
                    ownership,
                    "localization plan recording failed before native posting\n"
                    + stacktrace,
                    error,
                )

        assert self._staging_buf is not None
        staging_base = self._staging_buf.data_ptr() + off
        for ridx, s in enumerate(read_specs):
            source_rank = int(s.remote_rank)
            ownership.begin_prepare(source_rank)
            try:
                blens_r, _, rdev = self._remote_layout[engine_id][s.remote_rank]
                assert blens_r == blens, "per-rank region layout mismatch"
                rbases = self.kv_caches_base_addr[engine_id][s.remote_rank]
                local_descs, remote_descs = [], []
                for i in range(n_regions):
                    base_i = staging_base + region_off[i] + ridx * n_pos * blens[i]
                    for start, cnt, p0 in runs:
                        ln = cnt * blens[i]
                        local_descs.append((base_i + p0 * blens[i], ln, self.device_id))
                        remote_descs.append((rbases[i] + start * blens[i], ln, rdev))
                ld = self.nixl_wrapper.get_xfer_descs(
                    local_descs, self.nixl_memory_type
                )
                rd = self.nixl_wrapper.get_xfer_descs(
                    remote_descs, self.nixl_memory_type
                )
                agent = self._remote_agents[engine_id][s.remote_rank]
            except Exception as error:
                stacktrace = traceback.format_exc()
                ownership.record_prepare_failure(
                    source_rank,
                    f"rank {source_rank} native preparation raised\n{stacktrace}",
                )
                self._fail_coalesced_plan(
                    ownership,
                    f"rank {source_rank} native preparation raised\n{stacktrace}",
                    error,
                )
            self._initialize_and_post_coalesced(
                ownership,
                source_rank,
                ld,
                rd,
                agent,
                notif_id,
            )
        ownership.seal_posting()
        return "posted"

    def _read_blocks(
        self,
        *,
        expected_consumers: int = 1,
        read_spec: ReadSpec,
        dst_engine_id: str,
        request_id: str,
        remote_request_id: str,
        local_xfer_side_handle: int,
        remote_xfer_side_handle: int,
    ):
        """
        Post a READ point-to-point xfer request from a single local worker to
        a single remote worker.
        """
        assert self.transfer_topo is not None
        remote_rank = read_spec.remote_rank
        local_block_ids = read_spec.local_block_ids
        remote_block_ids = read_spec.remote_block_ids

        remote_info = self.transfer_topo.get_engine_info(dst_engine_id)
        block_size_ratio = self.transfer_topo.block_size_ratio(
            remote_info.remote_block_size
        )
        if block_size_ratio > 1:
            # TODO (NickLucche) assume HMA is off. Change to handle multiple KV groups.
            assert not self._is_hma_required
            local_block_ids0 = local_block_ids[0] if local_block_ids else []
            remote_block_ids0 = remote_block_ids[0]
            local_block_ids_mapped = self.get_mapped_blocks(
                np.asarray(local_block_ids0), block_size_ratio
            ).tolist()
            if len(local_block_ids_mapped) > len(remote_block_ids0):
                # NOTE:
                # get_mapped_blocks will always expand block_ids for n times.
                # ex:
                # prefill block_ids with block_size as 4:
                # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                # Local decode block_ids with block_size as 16: [1, 2, 3]
                # expanded decode block_ids with get_mapped_blocks from [1, 2, 3] to
                # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
                # Then we clip local to align with prefill
                # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] to
                # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                local_block_ids_mapped = local_block_ids_mapped[
                    : len(remote_block_ids0)
                ]
            local_block_ids = [local_block_ids_mapped] if local_block_ids_mapped else []
            remote_block_ids = [remote_block_ids0]
        # NOTE(rob): having the staging blocks be on the READER side is
        # not going to work well (since we will have to call rearrange tensors).
        # after we detect the txn is complete (which means we cannot make the
        # read trxn async easily). If we want to make "READ" happen cleanly,
        # then we will need to have the staging blocks on the remote side.

        # NOTE(rob): according to nvidia the staging blocks are used to
        # saturate IB with heterogeneous TP sizes.

        # Number of D TP workers that will read from dst P. Propagate info
        # on notification so that dst worker can wait before freeing blocks.
        notif_id = (
            f"{remote_request_id}:{self.world_size}:{expected_consumers}"
        ).encode()

        # Full prefix cache hit: do not need to read remote blocks,
        # just notify P worker that we have the blocks we need.
        if len(local_block_ids) == 0:
            # A full prefix cache hit is indicated with an empty list.
            agent_name = self._remote_agents[dst_engine_id][remote_rank]
            try:
                self.nixl_wrapper.send_notif(agent_name, notif_msg=notif_id)
            except Exception as e:
                self._log_failure(
                    failure_type="notification_failed",
                    msg="P worker blocks will be freed after timeout. "
                    "This may indicate network issues.",
                    req_id=request_id,
                    error=e,
                    dst_engine_id=dst_engine_id,
                    remote_rank=remote_rank,
                    remote_agent_name=agent_name,
                )
                self.xfer_stats.record_failed_notification()
            return

        assert (
            len(remote_block_ids)
            == len(local_block_ids)
            == len(self.kv_cache_config.kv_cache_groups)
        )
        remote_physical_per_logical = remote_info.remote_physical_blocks_per_logical
        local_block_ids, remote_block_ids = self._apply_prefix_caching(
            local_block_ids, remote_block_ids, remote_physical_per_logical
        )

        # NOTE (nicolo) With homogeneous TP, each TP worker loads KV from
        # corresponding rank. With heterogeneous TP, fixing D>P, the D tp
        # workers will issue xfers to parts of the P worker remote kv caches.

        # Get descs ids.
        remote_block_descs_ids = self._compute_desc_ids(
            block_ids=remote_block_ids,
            dst_num_blocks=self.dst_num_blocks[dst_engine_id],
            block_size_ratio=None,
            physical_blocks_per_logical=remote_info.remote_physical_blocks_per_logical,
        )
        local_block_descs_ids = self._compute_desc_ids(
            block_ids=local_block_ids,
            dst_num_blocks=self.dst_num_blocks[self.engine_id],
            block_size_ratio=block_size_ratio,
            physical_blocks_per_logical=self._physical_blocks_per_logical_kv_block,
        )

        assert len(local_block_descs_ids) == len(remote_block_descs_ids)

        self._assert_transfer_post_allowed(coalesced=False)

        # Prepare transfer with Nixl.
        handle = None
        try:
            handle = self.nixl_wrapper.make_prepped_xfer(
                "READ",
                local_xfer_side_handle,
                local_block_descs_ids,
                remote_xfer_side_handle,
                remote_block_descs_ids,
                notif_msg=notif_id,
            )

            # Begin async xfer.
            self.nixl_wrapper.transfer(handle)

            # Use handle to check completion in future step().
            self._recving_transfers[request_id].append(handle)
        except Exception as e:
            # mark all (logical) blocks for this request as invalid
            self._log_failure(
                failure_type="transfer_setup_failed",
                req_id=request_id,
                msg="Marking blocks as invalid",
                error=e,
                dst_engine_id=dst_engine_id,
                remote_rank=remote_rank,
            )
            self._handle_failed_transfer(request_id, handle)

    def _get_new_notifs(self) -> set[str]:
        """
        Get req_ids which got a remote xfer message. When multiple consumers
        are reading from the same producer (heterogeneous TP scenario), wait
        for all consumers to be done pulling.

        Also handles heartbeat notifications ("HB:req1,req2,...") by
        extending the lease on the referenced requests.
        """
        assert self.transfer_topo is not None
        notified_req_ids: set[str] = set()
        for notifs in self.nixl_wrapper.get_new_notifs().values():
            for notif in notifs:
                msg = notif.decode("utf-8")

                # Handle heartbeat messages from D-side.
                if msg.startswith("HB:"):
                    self._handle_heartbeat(msg[3:])
                    continue

                # Producer expired a lease: fence the rid and fail any
                # parked pull for it (in-flight ones are handled at
                # completion by the release fence in get_finished).
                if msg.startswith("EXPIRED:"):
                    rid = msg[len("EXPIRED:") :]
                    logger.warning("Producer expired lease for %s; fencing.", rid)
                    self._mark_rid_released(rid)
                    still_parked = []
                    for item in self._coalesce_pending:
                        p_req_id, p_meta, p_specs = item
                        if (
                            p_meta.remote is not None
                            and p_meta.remote.request_id == rid
                        ):
                            self._handle_failed_transfer(p_req_id, None)
                        else:
                            still_parked.append(item)
                    self._coalesce_pending.clear()
                    self._coalesce_pending.extend(still_parked)
                    continue

                parts = msg.rsplit(":", 2)
                if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
                    req_id, tp_size, expected_s = parts
                    expected_consumers = int(expected_s)
                else:
                    req_id, tp_size = msg.rsplit(":", 1)
                    expected_consumers = 1
                if (
                    req_id not in self._reqs_to_send
                    and req_id not in self._reqs_to_process
                ):
                    logger.error(
                        "Potentially invalid KV blocks for "
                        "unrecognized request %s were retrieved by "
                        "a decode worker. They may have expired.",
                        req_id,
                    )
                    continue

                # NOTE: `tp_ratio` is the opposite when swapping local<>remote
                n_consumers = int(tp_size)
                tp_ratio = self.transfer_topo.tp_ratio(n_consumers)

                # Number of reads *per producer* to wait for.
                # When remote D TP > local P TP we expect `tp_ratio` reads.
                consumers_per_producer = (
                    -tp_ratio if n_consumers > self.world_size else 1
                )

                self.consumer_notification_counts_by_req[req_id] += 1
                # Wait for all consumers (D) to be done reading before
                # freeing: TP fan-out reads AND n>1 sibling pulls.
                if (
                    self.consumer_notification_counts_by_req[req_id]
                    >= consumers_per_producer * expected_consumers
                ):
                    self._localization_capture_source_post(req_id)
                    notified_req_ids.add(req_id)
                    del self.consumer_notification_counts_by_req[req_id]
                    self._reqs_to_process.remove(req_id)
                    self._reqs_to_send.pop(req_id, None)
        return notified_req_ids
