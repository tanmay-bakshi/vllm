# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Base scheduler-side logic for the NIXL connector."""

import queue
import threading
import traceback
from typing import TYPE_CHECKING, Any

import msgspec
from zmq.constants import SocketOption, SocketType
from zmq.error import Again

from vllm import envs
from vllm.distributed.kv_transfer.kv_connector.utils import (
    BlockIds,
    EngineId,
    yield_req_data,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorHandshakeMetadata,
    KVConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    GET_META_MSG,
    PULL_OFFER_CANCELLATION_CONTROL_PREFIX,
    HeartbeatInfo,
    NixlConnectorMetadata,
    NixlHandshakePayload,
    ProducerLease,
    PullOfferCancellationAck,
    PullOfferCancellationControl,
    PullOfferCancelled,
    ReqId,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.utils import zmq_ctx
from vllm.distributed.kv_transfer.nixl_contracts import NixlSourceRoster
from vllm.distributed.kv_transfer.nixl_localization import (
    NixlLocalizationConfig,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.utils.network_utils import make_zmq_path
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    CrossAttentionSpec,
    EncoderOnlyAttentionSpec,
    FullAttentionSpec,
    MambaSpec,
    SlidingWindowSpec,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.request import Request

logger = init_logger(__name__)

_MAX_PENDING_OFFER_CANCELLATIONS = 65_536
_MAX_SPARSE_RETIRED_SOURCE_OFFERS = 65_536


class NixlBaseConnectorScheduler:
    """Base implementation of Scheduler side methods shared by pull and push."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        engine_id: str,
        kv_cache_config: "KVCacheConfig",
    ):
        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size
        self.engine_id: EngineId = engine_id
        self.kv_cache_config = kv_cache_config
        self.side_channel_host = envs.VLLM_NIXL_SIDE_CHANNEL_HOST
        self.side_channel_port = (
            envs.VLLM_NIXL_SIDE_CHANNEL_PORT
            + vllm_config.parallel_config.data_parallel_index
        )
        assert vllm_config.kv_transfer_config is not None
        self._kv_lease_duration: int = (
            vllm_config.kv_transfer_config.get_from_extra_config(
                "kv_lease_duration", 30
            )
        )
        if current_platform.device_type == "cpu":
            self.use_host_buffer = False
        else:
            self.use_host_buffer = (
                vllm_config.kv_transfer_config.kv_buffer_device == "cpu"
            )
        self._is_hma_required = (
            not vllm_config.scheduler_config.disable_hybrid_kv_cache_manager
            # Also handle unlikely SW-only model case instead of checking num_groups>1.
            and any(
                not isinstance(g.kv_cache_spec, FullAttentionSpec)
                for g in kv_cache_config.kv_cache_groups
            )
        )
        self._has_mamba = any(
            isinstance(g.kv_cache_spec, MambaSpec)
            for g in kv_cache_config.kv_cache_groups
        )

        logger.info("Initializing NIXL Scheduler %s", engine_id)
        if vllm_config.scheduler_config.disable_hybrid_kv_cache_manager:
            logger.info("Hybrid Memory Allocator is enabled with NIXL")

        # Background thread for handling new handshake requests.
        self._nixl_handshake_listener_t: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._offer_cancellation_queue: queue.Queue[PullOfferCancellationControl] = (
            queue.Queue(maxsize=_MAX_PENDING_OFFER_CANCELLATIONS)
        )
        self._localization_config = NixlLocalizationConfig.from_environment()
        self._source_rosters: dict[ReqId, NixlSourceRoster] = {}
        self._source_offer_generation = 0
        self._active_source_offer_generations: dict[ReqId, int] = {}
        self._sparse_retired_source_generations: set[int] = set()
        self._source_retired_through = 0
        if self._localization_config.enabled:
            logger.warning(
                "P-to-D localization observer enabled: run=%s arm=%s "
                "mode=%s target_count=%d. Clean results are instrumented-only "
                "evidence.",
                self._localization_config.run_id,
                self._localization_config.transport_arm,
                self._localization_config.mode.value,
                len(self._localization_config.target_request_ids),
            )

        # Requests that need to start recv/send.
        # New requests are added by update_state_after_alloc in
        # the scheduler. Used to make metadata passed to Worker.
        self._reqs_need_recv: dict[ReqId, tuple[Request, BlockIds]] = {}
        # KV-audit: rids finished since the last build_connector_meta.
        self._audit_finished_reqs: set[ReqId] = set()
        self._reqs_need_save: dict[ReqId, Request] = {}
        self._reqs_need_send: dict[ReqId, ProducerLease] = {}
        self._reqs_in_batch: set[ReqId] = set()
        # Reqs to remove from processed set because they're not to send after
        # remote prefill or aborted.
        self._reqs_not_processed: set[ReqId] = set()

        # Heartbeat tracking: requests needing periodic lease-renewal heartbeats to
        # remote P-side, stored as ready-to-send HeartbeatInfo grouped by remote engine
        self._heartbeat_by_engine: dict[EngineId, HeartbeatInfo] = {}
        # Reverse lookup: local req_id -> (engine_id, remote_req_id) for O(1) removal
        self._heartbeat_req_engine: dict[ReqId, tuple[EngineId, ReqId]] = {}
        self._heartbeat_snapshot_dirty = False

        # Gather Sliding Window sizes for each kv cache group (if any) in number of
        # blocks per KV cache group. This is used to clip the local attention window.
        sw_sizes_tokens: list[tuple[int, int]] = [
            (g.kv_cache_spec.sliding_window, g.kv_cache_spec.block_size)
            if isinstance(g.kv_cache_spec, SlidingWindowSpec)
            else (0, self.block_size)
            for g in kv_cache_config.kv_cache_groups
        ]
        # cdiv(n_tokens, block_size) gives blocks/window; add 1 to conservatively
        # account for boundary overlap eg window isn't fully aligned with blocks.
        self.blocks_per_sw = [
            cdiv(n_tokens, block_size) + 1 if n_tokens else 0
            for n_tokens, block_size in sw_sizes_tokens
        ]

        # Threshold to decide whether to compute kv cache locally
        # or pull from a remote node: minimum number of remote
        # tokens to amortize the xfer latencies
        self.kv_recompute_threshold: int = int(
            vllm_config.kv_transfer_config.get_from_extra_config(
                "kv_recompute_threshold", 64
            )
        )

        # Bi-directional KV transfer feature supports KV block
        # transfers from D node to P node
        self.is_bidirectional_kv_xfer_enabled = (
            vllm_config.kv_transfer_config.get_from_extra_config(
                "bidirectional_kv_xfer", False
            )
        )
        self.decoder_kv_blocks_ttl = (
            vllm_config.kv_transfer_config.get_from_extra_config(
                "decoder_kv_blocks_ttl", 480
            )
        )

        if self.is_bidirectional_kv_xfer_enabled and self.kv_recompute_threshold > 0:
            logger.info(
                "Bidirectional KV transfer is enabled and the kv "
                "recompute threshold is set to %d tokens."
                "KV blocks on D use a liveness deadline of %d seconds.",
                self.kv_recompute_threshold,
                self.decoder_kv_blocks_ttl,
            )

    def shutdown(self):
        self._stop_event.set()
        if self._nixl_handshake_listener_t is not None:
            self._nixl_handshake_listener_t.join()
            self._nixl_handshake_listener_t = None

    def on_new_request(self, request: "Request") -> None:
        """Track a request that may need heartbeats."""
        params = request.kv_transfer_params
        # NOTE (NickLucche) This excludes request meant for P, ie heartbeats are
        # effectively disabled for Bidirectional KV transfer.
        if params is None or not params.get("do_remote_prefill"):
            return
        # Only track if all required remote fields are present.
        remote_engine_id = params.get("remote_engine_id")
        remote_request_id = params.get("remote_request_id")
        host = params.get("remote_host")
        port = params.get("remote_port")
        tp_size = params.get("tp_size")
        if (
            remote_engine_id is None
            or remote_request_id is None
            or host is None
            or port is None
            or tp_size is None
        ):
            return
        if remote_engine_id not in self._heartbeat_by_engine:
            self._heartbeat_by_engine[remote_engine_id] = HeartbeatInfo(
                request_refcounts={},
                host=host,
                port=port,
                tp_size=tp_size,
            )
        info = self._heartbeat_by_engine[remote_engine_id]
        info.request_refcounts[remote_request_id] = (
            info.request_refcounts.get(remote_request_id, 0) + 1
        )
        self._heartbeat_snapshot_dirty = True
        self._heartbeat_req_engine[request.request_id] = (
            remote_engine_id,
            remote_request_id,
        )

    def _stop_heartbeat(self, req_id: ReqId) -> None:
        """Remove *req_id* from heartbeat tracking (if tracked)."""
        key = self._heartbeat_req_engine.pop(req_id, None)
        if key is None:
            return

        engine_id, remote_id = key
        info = self._heartbeat_by_engine.get(engine_id)
        if info is None:
            raise RuntimeError(f"Missing heartbeat owner for decoder request {req_id}")
        refcount = info.request_refcounts.get(remote_id)
        if refcount is None or refcount <= 0:
            raise RuntimeError(
                f"Missing heartbeat reference for producer request {remote_id}"
            )
        if refcount > 1:
            info.request_refcounts[remote_id] = refcount - 1
            self._heartbeat_snapshot_dirty = True
            return

        del info.request_refcounts[remote_id]
        if len(info.request_refcounts) == 0:
            del self._heartbeat_by_engine[engine_id]
        self._heartbeat_snapshot_dirty = True

    def _get_transferable_block_ids(
        self,
        block_ids: BlockIds,
        num_computed_tokens: int,
    ) -> BlockIds:
        """Return the initialized logical KV blocks for transfer.

        Producer allocation can include speculative lookahead pages beyond the
        computed token extent. Decoder self-attention block tables are positional,
        so pages wholly beyond that extent do not contain request KV. Recurrent and
        encoder-state groups are indexed by different extents and remain unchanged.

        :param block_ids: The request's allocated logical block tables.
        :param num_computed_tokens: Settled request token extent represented by the
            source cache.
        :returns: The canonical logical block tables to publish for transfer.
        """
        assert len(block_ids) == len(self.kv_cache_config.kv_cache_groups), (
            "Number of KV cache groups must match"
        )

        parallel_config = self.vllm_config.parallel_config
        context_parallel_size: int = (
            parallel_config.decode_context_parallel_size
            * parallel_config.prefill_context_parallel_size
        )
        transferable_block_ids: list[list[int]] = []
        for group, group_block_ids in zip(
            self.kv_cache_config.kv_cache_groups, block_ids
        ):
            spec = group.kv_cache_spec
            transferable_group_block_ids: list[int] = list(group_block_ids)
            if isinstance(spec, AttentionSpec) and not isinstance(
                spec, (CrossAttentionSpec, EncoderOnlyAttentionSpec)
            ):
                group_token_capacity: int = spec.block_size * context_parallel_size
                num_transferable_blocks: int = cdiv(
                    num_computed_tokens, group_token_capacity
                )
                del transferable_group_block_ids[num_transferable_blocks:]
            transferable_block_ids.append(transferable_group_block_ids)

        return self.get_sw_clipped_blocks(transferable_block_ids)

    def get_sw_clipped_blocks(self, block_ids: BlockIds) -> BlockIds:
        """
        Clip the number of blocks to the sliding window size for each kv cache group
        that employs SWA.
        This is necessary because the KV Cache manager initially allocates blocks for
        the entire sequence length, and successively cleans up blocks that are outside
        the window prior to the `request_finished_all_groups` hook.
        """
        if len(block_ids) == 0 or not self._is_hma_required:
            # No blocks to clip eg Full prefix cache hit or not a hybrid model.
            return block_ids
        # NOTE (NickLucche) This logic is currently handled at the connector level
        # because offloading connectors might want to receive the whole sequence even
        # for SWA groups. We will abstract this logic once the interface is more stable
        assert len(block_ids) == len(self.blocks_per_sw), (
            "Number of KV cache groups must match"
        )
        # For non-SWA groups, blocks_per_sw is 0 so we return all block_ids unchanged
        return tuple(
            [
                blocks[-self.blocks_per_sw[i] :]
                if self.blocks_per_sw[i] > 0
                else blocks
                for i, blocks in enumerate(block_ids)
            ]
        )

    def set_xfer_handshake_metadata(
        self, metadata: dict[int, KVConnectorHandshakeMetadata]
    ) -> None:
        """
        Set the KV connector handshake metadata for this connector.

        Args:
            metadata (dict): the handshake metadata to set.
        """
        encoded_data: dict[int, bytes] = {}
        encoder = msgspec.msgpack.Encoder()
        for tp_rank, rank_metadata in metadata.items():
            if not isinstance(rank_metadata, NixlHandshakePayload):
                raise ValueError(
                    "NixlConnectorScheduler expects NixlHandshakePayload for "
                    "handshake metadata."
                )
            encoded_data[tp_rank] = encoder.encode(rank_metadata)
            logger.debug(
                "Tp rank %d: encoded NixlHandshakePayload size: %s bytes",
                tp_rank,
                str(len(encoded_data[tp_rank])),
            )

        # Only start the listener when we have metadata to serve.
        if self._nixl_handshake_listener_t is None:
            ready_event = threading.Event()
            self._nixl_handshake_listener_t = threading.Thread(
                target=self._nixl_handshake_listener,
                args=(
                    encoded_data,
                    ready_event,
                    self._stop_event,
                    self._offer_cancellation_queue,
                    self.side_channel_host,
                    self.side_channel_port,
                ),
                daemon=True,
                name="nixl_handshake_listener",
            )
            self._nixl_handshake_listener_t.start()
            ready_event.wait()  # Wait for listener ZMQ socket to be ready.

    @staticmethod
    def _nixl_handshake_listener(
        encoded_data: dict[int, bytes],
        ready_event: threading.Event,
        stop_event: threading.Event,
        offer_cancellation_queue: queue.Queue[PullOfferCancellationControl],
        host: str,
        port: int,
    ) -> None:
        """Background thread for getting new NIXL handshakes."""
        # NOTE(rob): this is a simple implementation. We will move
        # to a better approach via HTTP endpoint soon.

        # Listen for new requests for metadata.
        path = make_zmq_path("tcp", host, port)
        logger.debug("Starting listening on path: %s", path)
        with zmq_ctx(SocketType.ROUTER, path) as sock:
            sock.setsockopt(SocketOption.RCVTIMEO, 1000)
            ready_event.set()
            while True:
                try:
                    identity, _, msg = sock.recv_multipart()
                except Again:
                    if stop_event.is_set():
                        break
                    continue
                if msg.startswith(PULL_OFFER_CANCELLATION_CONTROL_PREFIX):
                    response = NixlBaseConnectorScheduler._queue_offer_cancellation(
                        msg,
                        frozenset(encoded_data),
                        offer_cancellation_queue,
                    )
                    sock.send_multipart((identity, b"", response))
                    continue
                try:
                    request = msgspec.msgpack.decode(msg)
                except msgspec.DecodeError:
                    logger.warning("Connection listener received malformed MessagePack")
                    sock.send_multipart((identity, b"", b""))
                    continue
                if (
                    not isinstance(request, (list, tuple))
                    or len(request) != 2
                    or request[0] != GET_META_MSG
                    or type(request[1]) is not int
                    or request[1] not in encoded_data
                ):
                    logger.warning(
                        "Connection listener got invalid handshake %s", request
                    )
                    sock.send_multipart((identity, b"", b""))
                    continue
                target_tp_rank = request[1]
                logger.debug(
                    "Received handshake message for tp rank %s",
                    target_tp_rank,
                )
                sock.send_multipart((identity, b"", encoded_data[target_tp_rank]))

    @staticmethod
    def _queue_offer_cancellation(
        message: bytes,
        producer_ranks: frozenset[int],
        pending: queue.Queue[PullOfferCancellationControl],
    ) -> bytes:
        """Validate and atomically queue one side-channel cancellation.

        :param message: Prefixed typed side-channel request.
        :param producer_ranks: Producer ranks served by this scheduler.
        :param pending: Bounded cross-thread cancellation queue.
        :returns: Typed acknowledgement bytes, or empty bytes if undecodable.
        """
        try:
            control = msgspec.msgpack.decode(
                message[len(PULL_OFFER_CANCELLATION_CONTROL_PREFIX) :],
                type=PullOfferCancellationControl,
            )
        except (msgspec.DecodeError, msgspec.ValidationError):
            logger.error(
                "Rejecting malformed NIXL offer cancellation control\n%s",
                traceback.format_exc(),
            )
            return b""

        proof = control.proof
        target_ranks = control.producer_ranks
        canonical_ranks = tuple(sorted(set(target_ranks)))
        valid = (
            len(target_ranks) > 0
            and target_ranks == canonical_ranks
            and all(
                type(rank) is int and rank in producer_ranks for rank in target_ranks
            )
            and type(proof.producer_request_id) is str
            and len(proof.producer_request_id) > 0
            and type(proof.offer_generation) is int
            and proof.offer_generation > 0
            and type(proof.consumer_rank) is int
            and type(proof.consumer_tp_size) is int
            and proof.consumer_tp_size > 0
            and proof.consumer_rank >= 0
            and proof.consumer_rank < proof.consumer_tp_size
            and type(proof.expected_consumers) is int
            and proof.expected_consumers > 0
        )
        accepted = False
        if valid:
            try:
                pending.put_nowait(control)
            except queue.Full:
                logger.error(
                    "NIXL offer cancellation queue is full; producer request %s "
                    "remains pinned",
                    proof.producer_request_id,
                )
            else:
                accepted = True
        else:
            logger.error(
                "Rejecting invalid NIXL offer cancellation for producer request "
                "%s and ranks %s",
                proof.producer_request_id,
                target_ranks,
            )

        ack = PullOfferCancellationAck(
            producer_request_id=proof.producer_request_id,
            offer_generation=proof.offer_generation,
            producer_ranks=target_ranks,
            accepted=accepted,
        )
        return msgspec.msgpack.encode(ack)

    def _mamba_prefill_token_count(self, num_prompt_tokens: int) -> int:
        """D-side only. Returns N-1 for Mamba models since the decoder
        always recomputes the last token and must start from h(N-1)."""
        if self._has_mamba and num_prompt_tokens > 1:
            return num_prompt_tokens - 1
        return num_prompt_tokens

    def _truncate_mamba_request_for_prefill(self, request: "Request") -> None:
        """P-side only: drop the last prompt token so the prefiller computes
        h(N-1) instead of h(N). The decoder recomputes the last token to
        derive h(N) correctly.

        Guarded by ``_p_side_truncated`` to avoid repeated truncation if the
        request is preempted and rescheduled."""
        params = request.kv_transfer_params
        if (
            params is not None
            # Guard against repeated truncation after preemption/reschedule.
            and not params.get("_p_side_truncated")
            and request.num_prompt_tokens > 1
        ):
            if request.prompt_token_ids is not None:
                request.prompt_token_ids.pop()
            elif request.prompt_embeds is not None:
                request.prompt_embeds = request.prompt_embeds[:-1]
            else:
                return

            request._all_token_ids.pop()
            request.num_prompt_tokens -= 1
            request.max_tokens = 1
            params["_p_side_truncated"] = True

    def _build_save_meta(
        self,
        meta: NixlConnectorMetadata,
        scheduler_output: SchedulerOutput,
    ) -> None:
        # only called when use_host_buffer is True to build the save metadata

        # NOTE: For the prefill side, there might be a chance that an early added
        # request is a chunked prefill, so we need to check if new blocks are added
        for req_id, new_block_id_groups, _ in yield_req_data(scheduler_output):
            req_to_save = self._reqs_need_save.get(req_id)
            if req_to_save is None or new_block_id_groups is None:
                continue
            req = req_to_save

            assert req.kv_transfer_params is not None
            clipped_block_id_groups = self.get_sw_clipped_blocks(new_block_id_groups)
            meta.add_new_req_to_save(
                request_id=req_id,
                local_block_ids=clipped_block_id_groups,
                kv_transfer_params=req.kv_transfer_params,
            )
            assert scheduler_output.num_scheduled_tokens is not None
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
            is_partial = (
                req.num_computed_tokens + num_scheduled_tokens
            ) < req.num_prompt_tokens
            if not is_partial:
                # For non-partial prefills, once new req_meta is scheduled, it
                # can be removed from _reqs_need_save.
                # For partial prefill case, we will retain the request in
                # _reqs_need_save until all blocks are scheduled with req_meta.
                # Therefore, only pop if `not is_partial`.
                self._reqs_need_save.pop(req_id)

    def _drain_offer_cancellations(
        self,
    ) -> dict[int, tuple[PullOfferCancelled, ...]]:
        """Drain queued control proofs into worker-rank metadata.

        :returns: Deduplicated cancellation proofs keyed by producer rank.
        """
        proofs_by_rank: dict[int, set[PullOfferCancelled]] = {}
        while True:
            try:
                control = self._offer_cancellation_queue.get_nowait()
            except queue.Empty:
                break
            for producer_rank in control.producer_ranks:
                proofs_by_rank.setdefault(producer_rank, set()).add(control.proof)

        return {
            producer_rank: tuple(
                sorted(
                    proofs,
                    key=lambda proof: (
                        proof.producer_request_id,
                        proof.consumer_rank,
                        proof.consumer_tp_size,
                        proof.expected_consumers,
                    ),
                )
            )
            for producer_rank, proofs in proofs_by_rank.items()
        }

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        meta = NixlConnectorMetadata()
        meta.offer_cancellations_by_rank = self._drain_offer_cancellations()

        # Loop through scheduled reqs and convert to ReqMeta.
        for req_id, (req, block_ids) in self._reqs_need_recv.items():
            assert req.kv_transfer_params is not None
            meta.add_new_req_to_recv(
                request_id=req_id,
                local_block_ids=block_ids,
                kv_transfer_params=req.kv_transfer_params,
            )

        if self.use_host_buffer:
            self._build_save_meta(meta, scheduler_output)

        meta.reqs_to_send = self._reqs_need_send
        meta.source_rosters = self._source_rosters
        meta.source_retired_through = self._source_retired_through
        meta.scheduled_request_ids = set(scheduler_output.num_scheduled_tokens)
        meta.reqs_in_batch = self._reqs_in_batch
        meta.reqs_not_processed = self._reqs_not_processed
        meta.audit_finished = self._audit_finished_reqs
        self._audit_finished_reqs = set()

        if self._heartbeat_snapshot_dirty:
            meta.heartbeat_snapshot = {
                engine_id: HeartbeatInfo(
                    request_refcounts=dict(info.request_refcounts),
                    host=info.host,
                    port=info.port,
                    tp_size=info.tp_size,
                )
                for engine_id, info in self._heartbeat_by_engine.items()
            }
            self._heartbeat_snapshot_dirty = False

        # Clear the list once workers start the transfers
        self._reqs_need_recv.clear()
        self._reqs_in_batch = set()
        self._reqs_not_processed = set()
        self._reqs_need_send = {}
        self._source_rosters = {}

        return meta

    def update_connector_output(self, connector_output: "KVConnectorOutput") -> None:
        """Commit worker-authorized source retirements and receive terminals.

        :param connector_output: Rank-aggregated worker transfer completions.
        """
        worker_meta = connector_output.kv_connector_worker_meta
        if worker_meta is not None:
            raise RuntimeError("NIXL does not emit connector worker metadata")

        finished_sending = connector_output.finished_sending
        if finished_sending is not None:
            for req_id in finished_sending:
                self._retire_source_offer(req_id)

        completed_receive_ids = set(connector_output.finished_recving or ())
        completed_receive_ids.update(connector_output.failed_recving)
        for req_id in completed_receive_ids:
            self._stop_heartbeat(req_id)

    def _register_source_offer(self, request_id: ReqId, generation: int) -> None:
        """Bind one retained producer allocation to its gap-free generation.

        :param request_id: Producer request retaining source blocks.
        :param generation: Newly issued source-offer generation.
        """
        if type(request_id) is not str or len(request_id) == 0:
            raise ValueError("source-offer request_id must be a non-empty string")
        if type(generation) is not int or generation <= self._source_retired_through:
            raise ValueError("source-offer generation must exceed the retirement floor")
        if generation != self._source_offer_generation:
            raise RuntimeError("source-offer generation is not the current issuance")
        if request_id in self._active_source_offer_generations:
            raise RuntimeError(f"source offer {request_id} is already active")
        self._active_source_offer_generations[request_id] = generation

    def _retire_source_offer(self, request_id: ReqId) -> None:
        """Advance the contiguous producer retirement floor when possible.

        ``finished_sending`` is the exact rank-aggregated point at which the
        scheduler may return the producer blocks to the allocator. Generations
        above a gap remain explicit until every earlier allocation also retires.

        :param request_id: Producer request committed by every worker rank.
        """
        generation = self._active_source_offer_generations.pop(request_id, None)
        if generation is None:
            return
        if (
            generation <= self._source_retired_through
            or generation in self._sparse_retired_source_generations
        ):
            raise RuntimeError("source-offer generation retired more than once")
        if generation > self._source_retired_through + 1:
            if (
                len(self._sparse_retired_source_generations)
                >= _MAX_SPARSE_RETIRED_SOURCE_OFFERS
            ):
                raise RuntimeError(
                    "source-offer retirement gap exceeded its fail-stop bound"
                )
            self._sparse_retired_source_generations.add(generation)
            return

        self._source_retired_through = generation
        while (
            self._source_retired_through + 1 in self._sparse_retired_source_generations
        ):
            self._source_retired_through += 1
            self._sparse_retired_source_generations.remove(self._source_retired_through)

    def has_pending_push_work(self) -> bool:
        return False

    ############################################################
    # Abstract methods that subclasses must implement
    ############################################################

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        raise NotImplementedError

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        raise NotImplementedError

    def request_finished(
        self,
        request: "Request",
        block_ids: BlockIds,
    ) -> tuple[bool, dict[str, Any] | None]:
        raise NotImplementedError
