# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pull-specific scheduler-side logic for the NIXL connector."""

import time
from typing import TYPE_CHECKING, Any

from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_scheduler import (
    NixlBaseConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    ProducerLease,
)
from vllm.distributed.kv_transfer.nixl_localization import NixlSourceRoster
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


def _expected_consumers(params: dict[str, Any]) -> int:
    """Read and validate the producer-owned logical consumer count.

    :param params: Request KV-transfer parameters.
    :returns: Positive logical consumer count.
    :raises ValueError: If the contract is invalid.
    """
    count = params.get("expected_consumers", 1)
    if type(count) is not int or count < 1:
        raise ValueError("expected_consumers must be a positive integer")
    return count


def _consumer_tp_size(params: dict[str, Any]) -> int:
    """Read and validate the producer-owned decoder topology contract.

    :param params: Request KV-transfer parameters.
    :returns: Positive decoder tensor-parallel size.
    :raises ValueError: If the contract is invalid.
    """
    size = params.get("consumer_tp_size", 1)
    if type(size) is not int or size < 1:
        raise ValueError("consumer_tp_size must be a positive integer")
    return size


class NixlPullConnectorScheduler(NixlBaseConnectorScheduler):
    """Pull-specific scheduler logic (READ-based KV transfer)."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        engine_id: str,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config, engine_id, kv_cache_config)

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int, bool]:
        """
        For remote prefill, pull all prompt blocks from remote
        asynchronously relative to engine execution.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request
        Returns:
            * the number of tokens that can be loaded from the
              external KV cache beyond what is already computed.
            * true if the external KV cache tokens will be loaded
              asynchronously (between scheduler steps).
        """

        params = request.kv_transfer_params
        logger.debug(
            "NIXLConnector get_num_new_matched_tokens: "
            "num_computed_tokens=%s, kv_transfer_params=%s",
            num_computed_tokens,
            params,
        )

        if params is not None and params.get("do_remote_prefill"):
            # Remote prefill: get all prompt blocks from remote.
            token_ids = request.prompt_token_ids or []
            actual = self._mamba_prefill_token_count(len(token_ids))
            count = actual - num_computed_tokens
            if count > 0:
                return count, True

        if params is not None and params.get("do_remote_decode") and self._has_mamba:
            self._truncate_mamba_request_for_prefill(request)

        if (
            params is not None
            and params.get("do_remote_decode")
            and params.get("remote_block_ids")
            and all(
                p in params
                for p in (
                    "remote_engine_id",
                    "remote_request_id",
                    "remote_host",
                    "remote_port",
                )
            )
        ):
            # Decode node has kv blocks for part of prefill request, so, provide them
            # as an external token count to scheduler.
            # The tokens will be loaded if not already present
            # in the prefill node local cache
            remote_num_tokens = params.get("remote_num_tokens") or 0
            count = (
                min(remote_num_tokens, request.num_prompt_tokens) - num_computed_tokens
            )
            if count > 0:
                # Check kv_recompute_threshold: skip pull if
                # remote tokens are below the threshold.
                if (
                    self.kv_recompute_threshold > 0
                    and count < self.kv_recompute_threshold
                ):
                    logger.debug(
                        "Skipping remote pull for %s: %d remote tokens < threshold %d",
                        request.request_id,
                        count,
                        self.kv_recompute_threshold,
                    )
                    return 0, False
                if self._pull_single_flight:
                    rid = params.get("remote_request_id")
                    leader = self._pull_leaders.get(rid) if rid else None
                    if rid and leader is None:
                        self._pull_leaders[rid] = request.request_id
                        logger.info(
                            "[single-flight] %s leads pull of %s",
                            request.request_id,
                            rid,
                        )
                    elif leader is not None and leader != request.request_id:
                        # A sibling is already pulling this registration:
                        # defer (re-evaluated every step). Once the
                        # leader's blocks commit+publish, our local hit
                        # covers the prompt and count stops being > 0, so
                        # this branch is never reached again.
                        return None, False  # type: ignore[return-value]
                return count, True

        # No remote prefill for this request.
        return 0, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        params = request.kv_transfer_params
        logger.debug(
            "NIXLConnector update_state_after_alloc: "
            "num_external_tokens=%s, kv_transfer_params=%s",
            num_external_tokens,
            params,
        )

        if not params:
            return

        if params.get("do_remote_decode") or (
            params.get("do_remote_prefill") and self.is_bidirectional_kv_xfer_enabled
        ):
            self._reqs_in_batch.add(request.request_id)
        if self.use_host_buffer and params.get("do_remote_decode"):
            # NOTE: when accelerator is not directly supported by Nixl,
            # prefilled blocks need to be saved to host memory before transfer.
            self._reqs_need_save[request.request_id] = request
        elif params.get("do_remote_prefill") or (
            params.get("do_remote_decode")
            and self.is_bidirectional_kv_xfer_enabled
            and not params.get("_remote_blocks_processed")
        ):
            if params.get("remote_block_ids"):
                if all(
                    p in params
                    for p in (
                        "remote_engine_id",
                        "remote_request_id",
                        "remote_host",
                        "remote_port",
                    )
                ):
                    # If remote_blocks and num_external_tokens = 0, we have
                    # a full prefix cache hit on the local node. We need to call
                    # send_notif in _read_blocks to free the memory on the remote node.

                    unhashed_local_block_ids: BlockIds = (
                        blocks.get_unhashed_block_ids_all_groups()
                        if num_external_tokens > 0
                        else ()
                    )
                    local_block_ids = self.get_sw_clipped_blocks(
                        unhashed_local_block_ids
                    )

                    # Get unhashed blocks to pull from remote. Mind that a full prefix
                    # cache hit is indicated with an empty list.
                    self._reqs_need_recv[request.request_id] = (
                        request,
                        local_block_ids,
                    )

                else:
                    logger.warning(
                        "Got invalid KVTransferParams: %s. This "
                        "request will not utilize KVTransfer",
                        params,
                    )
            else:
                assert num_external_tokens == 0
            # Only trigger 1 KV transfer per request.
            params["do_remote_prefill"] = False
            params["_remote_blocks_processed"] = True

    def request_finished(
        self,
        request: "Request",
        block_ids: "BlockIds",
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Once a request is finished, determine whether request blocks
        should be freed now or will be sent asynchronously and freed later.
        """
        from vllm.v1.request import RequestStatus

        params = request.kv_transfer_params
        logger.debug(
            "NIXLConnector request_finished(%s), request_status=%s, "
            "kv_transfer_params=%s",
            request.request_id,
            request.status,
            params,
        )
        if not params:
            return False, None

        is_p_node = bool(params.get("do_remote_decode"))
        is_d_node = not is_p_node
        expected_consumers = _expected_consumers(params)
        consumer_tp_size = _consumer_tp_size(params)

        # Stop heartbeating for aborted requests that never reached finished_recving:
        # normal path cleans up in update_connector_output.
        self._stop_heartbeat(request.request_id)
        # KV-audit: let the worker retire this rid's audit state before
        # its blocks can be reallocated to a new pull.
        self._audit_finished_reqs.add(request.request_id)
        if self._pull_single_flight:
            _sf_rid = params.get("remote_request_id")
            if _sf_rid and self._pull_leaders.get(_sf_rid) == request.request_id:
                # Leader finished (any status). If it committed, siblings
                # are already local-hitting; if it aborted pre-commit, the
                # next sibling claims leadership and pulls.
                del self._pull_leaders[_sf_rid]

        if params.get("do_remote_prefill"):
            # If do_remote_prefill is still True when the request is finished,
            # update_state_after_alloc must not have been called (the request
            # must have been aborted before it was scheduled, e.g. via the
            # abort_immediately path used to clean up KV-transfer requests
            # rejected at the D-side serving layer).
            # To avoid stranding the prefill blocks in the prefill instance,
            # we must add empty block_ids to _reqs_need_recv so that our
            # worker side will notify and free blocks in the prefill instance.
            self._reqs_need_recv[request.request_id] = (request, [])
            params["do_remote_prefill"] = False
            return False, None

        if is_d_node and not self.is_bidirectional_kv_xfer_enabled:
            return False, None

        if request.status not in (
            RequestStatus.FINISHED_LENGTH_CAPPED,
            RequestStatus.FINISHED_STOPPED,
        ):
            # Also include the case of a P/D Prefill request with immediate
            # block free (eg abort). Stop tracking this request.
            self._reqs_not_processed.add(request.request_id)
            # Clear _reqs_need_save if a request is aborted as partial prefill.
            self._reqs_need_save.pop(request.request_id, None)
            return False, None

        block_ids = self._get_transferable_block_ids(request, block_ids)
        delay_free_blocks = any(len(group) > 0 for group in block_ids)
        remote_num_tokens = 0
        localization_params: dict[str, Any] = {}
        if delay_free_blocks:
            # Prefill request on remote. It will be read from D upon completion
            request_deadline_duration = self._kv_lease_duration
            if is_d_node:
                request_deadline_duration = self.decoder_kv_blocks_ttl
            logger.debug(
                "NIXLConnector request_finished(%s) assigned a %d-second "
                "liveness deadline",
                request.request_id,
                request_deadline_duration,
            )
            self._reqs_need_send[request.request_id] = ProducerLease(
                deadline=time.perf_counter() + request_deadline_duration,
                expected_consumers=expected_consumers,
                consumer_tp_size=consumer_tp_size,
            )
            if is_p_node and self._localization_config.enabled_for(request.request_id):
                self._localization_offer_generation += 1
                offer_generation = self._localization_offer_generation
                self._source_rosters[request.request_id] = NixlSourceRoster(
                    offer_generation=offer_generation,
                    iteration=0,
                    valid_token_extent=int(request.num_computed_tokens),
                    group_token_capacities=tuple(
                        int(group.kv_cache_spec.block_size)
                        for group in self.kv_cache_config.kv_cache_groups
                    ),
                    block_ids=tuple(
                        tuple(int(block_id) for block_id in group)
                        for group in block_ids
                    ),
                )
                localization_params = {
                    "p2d_run_id": self._localization_config.run_id,
                    "p2d_transport_arm": self._localization_config.transport_arm,
                    "p2d_offer_generation": offer_generation,
                    "p2d_iteration": 0,
                }

            remote_num_tokens = request.num_computed_tokens

        return delay_free_blocks, dict(
            do_remote_prefill=is_p_node,
            do_remote_decode=is_d_node,
            remote_block_ids=block_ids,
            remote_engine_id=self.engine_id,
            remote_request_id=request.request_id,
            remote_host=self.side_channel_host,
            remote_port=self.side_channel_port,
            tp_size=self.vllm_config.parallel_config.tensor_parallel_size,
            remote_num_tokens=remote_num_tokens,
            expected_consumers=expected_consumers,
            consumer_tp_size=consumer_tp_size,
            **localization_params,
        )
