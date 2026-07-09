# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pull-specific (READ) worker-side logic for the NIXL connector."""

import time
from typing import TYPE_CHECKING

import numpy as np

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
from vllm.logger import init_logger

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
    ):
        super().__init__(vllm_config, engine_id, kv_cache_config)

    def start_load_kv(self, metadata: NixlConnectorMetadata):
        """
        Start loading by triggering non-blocking nixl_xfer.
        We check for these trnxs to complete in each step().
        """
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
            # We should never get an abort after setting an expiry timer
            assert req_id not in self._reqs_to_send

        # Add to requests that are waiting to be read and track expiration.
        for req_id, expiration_time in metadata.reqs_to_send.items():
            if req_id in self._reqs_to_process:
                self._reqs_to_send[req_id] = expiration_time

        # Send heartbeats to P-side engines to keep KV blocks alive while
        # requests sit in the D scheduler WAITING queue.
        self._send_heartbeats(metadata)

    def _read_blocks_for_req(self, req_id: str, meta: ReqMeta):
        assert meta.remote is not None and self.transfer_topo is not None
        engine_id = meta.remote.engine_id
        # Update last activity from this remote. Mind that cleanup is done on main
        # thread (this one), so we don't race on this structure.
        self._engine_last_active[engine_id] = time.perf_counter()
        plan = self.tp_mappings[engine_id]
        remote_info = self.transfer_topo.get_engine_info(engine_id)
        tp_ratio = self.transfer_topo.tp_ratio(remote_info.remote_tp_size)

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

        # Coalesced pull fast path (see base_worker state comment): all
        # gates must hold or we fall through to the stock per-descriptor
        # path, which stays authoritative. A staging-pool miss PARKS the
        # request (FIFO, serviced from get_finished) instead of taking
        # the ~20x slower stock path.
        if self._coalesce_gate(engine_id, tp_ratio, read_specs):
            res = self._coalesced_read_request(req_id, meta, read_specs)
            if res == "posted":
                return
            if res == "defer":
                self._coalesce_pending.append((req_id, meta, read_specs))
                logger.debug(
                    "coalesced pull: parked %s for staging (%s queued)",
                    req_id, len(self._coalesce_pending))
                return

        if any(self._sp_group_flags()) or self._no_stock_dma():
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
                "stock path.", req_id, any(self._sp_group_flags()),
                self._no_stock_dma())
            self._handle_failed_transfer(req_id, None)
            return

        self._stock_read_specs(req_id, meta, read_specs)

    def _no_stock_dma(self) -> bool:
        import os as _os_ns
        return _os_ns.environ.get(
            "VLLM_GEMMA4_NIXL_NO_STOCK_DMA", "0") == "1"

    def _stock_read_specs(self, req_id: str, meta: ReqMeta,
                          read_specs: list[ReadSpec]) -> None:
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
            )

        if self.use_mla and tp_ratio < 0 and read_specs:
            # ..but we still need to notify the other remote ranks that we
            # have the blocks we need so they can update the request state.
            notif_id = f"{meta.remote.request_id}:{self.world_size}".encode()
            remote_agents = self._remote_agents[meta.remote.engine_id]
            for rank_to_notify, agent in remote_agents.items():
                if rank_to_notify != read_specs[0].remote_rank:
                    self.nixl_wrapper.send_notif(agent, notif_msg=notif_id)

    # ------------------------------------------------------------------
    # Coalesced pull
    # ------------------------------------------------------------------

    def _coalesce_gate(self, engine_id: str, tp_ratio: int,
                       read_specs: list[ReadSpec]) -> bool:
        """All conditions under which the coalesced path is proven
        equivalent to the stock path. Anything else -> stock."""
        if not self.coalesce_pull:
            return False
        if not (tp_ratio < 0 and not self.use_mla
                and not self.use_host_buffer and not self._has_mamba):
            return False
        if self._physical_blocks_per_logical_kv_block != 1:
            return False
        if any(self._region_is_mla) or not self._region_tensors:
            return False
        assert self.transfer_topo is not None
        remote_info = self.transfer_topo.get_engine_info(engine_id)
        if self.transfer_topo.block_size_ratio(
                remote_info.remote_block_size) != 1:
            return False
        layout = self._remote_layout.get(engine_id)
        if not layout or any(s.remote_rank not in layout for s in read_specs):
            return False
        # uniform shard participation: every rank reads the same lists
        # (pure SPLIT full-attention groups)
        spec0 = read_specs[0]
        for s in read_specs[1:]:
            if (s.local_block_ids != spec0.local_block_ids
                    or s.remote_block_ids != spec0.remote_block_ids):
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
                if self._no_stock_dma():
                    logger.error(
                        "coalesced drain: %s not expressible; failing "
                        "instead of the stock path.", req_id)
                    self._handle_failed_transfer(req_id, None)
                    continue
                self._stock_read_specs(req_id, meta, read_specs)

    def _coalesced_read_request(self, req_id: str, meta: ReqMeta,
                                read_specs: list[ReadSpec]) -> str:
        """Whole-request coalesced pull: per remote rank, one READ xfer
        of contiguous whole-block run descriptors into staging.
        Returns "posted" (transfers in flight, or nothing to pull, or a
        mid-post failure already routed through _handle_failed_transfer),
        "defer" (staging exhausted but the request fits the pool: park
        and retry), or "stock" (nothing posted; run the stock path)."""
        assert meta.remote is not None and self.transfer_topo is not None
        engine_id = meta.remote.engine_id
        remote_info = self.transfer_topo.get_engine_info(engine_id)
        spec0 = read_specs[0]
        local_ids, remote_ids = self._apply_prefix_caching(
            [list(g) for g in spec0.local_block_ids],
            [list(g) for g in spec0.remote_block_ids],
            remote_info.remote_physical_blocks_per_logical,
        )
        notif_id = f"{meta.remote.request_id}:{self.world_size}".encode()
        if len(local_ids) == 0 or sum(len(g) for g in local_ids) == 0:
            # full prefix hit: just release P's blocks on every rank
            for s in read_specs:
                agent = self._remote_agents[engine_id][s.remote_rank]
                try:
                    self.nixl_wrapper.send_notif(agent, notif_msg=notif_id)
                except Exception:
                    self.xfer_stats.record_failed_notification()
            return "posted"

        # HMA broadcast semantics (see _compute_desc_ids): every group's
        # blocks are transferred across every region, position-ordered.
        # F2b single-plane groups (kv_planes==1): local blocks hold 2x
        # remote tokens, so remote block k of the request expands to
        # scatter destination (local_ids[k//2], half k%2); the scatter
        # keeps only the K half for those positions (sp_half >= 0).
        sp_flags = self._sp_group_flags()
        lpos_l, rpos_l, half_l = [], [], []
        for gi in range(len(local_ids)):
            lg = np.asarray(local_ids[gi], dtype=np.int64)
            rg = np.asarray(remote_ids[gi], dtype=np.int64)
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
        lpos = (np.concatenate(lpos_l) if lpos_l
                else np.zeros(0, dtype=np.int64))
        rpos = (np.concatenate(rpos_l) if rpos_l
                else np.zeros(0, dtype=np.int64))
        halves = (np.concatenate(half_l) if half_l
                  else np.zeros(0, dtype=np.int8))
        # Transfer order is free (the (remote, local) pairing is what
        # matters): sort by remote id so run detection harvests all the
        # adjacency the remote pool still has. The scatter index (lpos)
        # is permuted identically, so placement is unchanged.
        order = np.argsort(rpos, kind="stable")
        rpos = rpos[order]
        lpos = lpos[order]
        halves = halves[order]
        n_pos = len(lpos)
        n_ranks = len(read_specs)
        blens = self._remote_layout[engine_id][spec0.remote_rank][0]
        n_regions = len(blens)

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
        off = self._staging_alloc(acc)
        if off is None:
            if acc > self.coalesce_staging_mb * 1024 * 1024:
                # can never fit: the stock path is the only option
                logger.warning(
                    "coalesced pull: request %s needs %sMB staging "
                    "(pool %sMB); stock path", req_id, acc >> 20,
                    self.coalesce_staging_mb)
                return "stock"
            return "defer"

        assert self._staging_buf is not None
        staging_base = self._staging_buf.data_ptr() + off
        posted = False
        try:
            plan = self.tp_mappings[engine_id]
            for ridx, s in enumerate(read_specs):
                blens_r, _, rdev = self._remote_layout[engine_id][
                    s.remote_rank]
                assert blens_r == blens, "per-rank region layout mismatch"
                rbases = self.kv_caches_base_addr[engine_id][s.remote_rank]
                local_descs, remote_descs = [], []
                for i in range(n_regions):
                    base_i = (staging_base + region_off[i]
                              + ridx * n_pos * blens[i])
                    for (start, cnt, p0) in runs:
                        ln = cnt * blens[i]
                        local_descs.append(
                            (base_i + p0 * blens[i], ln, self.device_id))
                        remote_descs.append(
                            (rbases[i] + start * blens[i], ln, rdev))
                ld = self.nixl_wrapper.get_xfer_descs(
                    local_descs, self.nixl_memory_type)
                rd = self.nixl_wrapper.get_xfer_descs(
                    remote_descs, self.nixl_memory_type)
                agent = self._remote_agents[engine_id][s.remote_rank]
                handle = self.nixl_wrapper.initialize_xfer(
                    "READ", ld, rd, agent, notif_id)
                self.nixl_wrapper.transfer(handle)
                posted = True
                self._recving_transfers[req_id].append(handle)
        except Exception as e:
            if not posted:
                self._staging_release(off, acc)
                logger.warning(
                    "coalesced pull setup failed for %s (%s); stock path",
                    req_id, e)
                return "stock"
            self._log_failure(
                failure_type="transfer_setup_failed",
                req_id=req_id,
                msg="coalesced pull failed mid-post; marking blocks invalid",
                error=e,
                dst_engine_id=engine_id,
            )
            self._coalesce_plans[req_id] = dict(off=off, size=acc)
            self._handle_failed_transfer(req_id, None)
            return "posted"

        self._coalesce_plans[req_id] = dict(
            off=off, size=acc, n_pos=n_pos, n_ranks=n_ranks, blens=blens,
            region_off=region_off, lpos=lpos.tolist(),
            sp_half=halves.tolist(),
            slots=[plan.rank_to_attention_slot.get(s.remote_rank, 0)
                   for s in read_specs],
        )
        return "posted"

    def _read_blocks(
        self,
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
        notif_id = f"{remote_request_id}:{self.world_size}".encode()

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

                req_id, tp_size = msg.rsplit(":", 1)
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
                # Wait all consumers (D) to be done reading before freeing.
                if (
                    self.consumer_notification_counts_by_req[req_id]
                    == consumers_per_producer
                ):
                    notified_req_ids.add(req_id)
                    del self.consumer_notification_counts_by_req[req_id]
                    self._reqs_to_process.remove(req_id)
                    self._reqs_to_send.pop(req_id, None)
        return notified_req_ids
