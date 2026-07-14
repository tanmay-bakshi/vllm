# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pull-specific scheduler-side logic for the NIXL connector."""

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds, EngineId
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_scheduler import (
    NixlBaseConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    ProducerLease,
    ReqId,
)
from vllm.distributed.kv_transfer.nixl_localization import NixlSourceRoster
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass(frozen=True)
class _ProducerOfferIdentity:
    """Immutable identity of one producer-side KV offer.

    :ivar engine_id: Engine that owns the offered blocks.
    :ivar request_id: Producer request that owns the offered blocks.
    :ivar generation: Producer offer generation when one is available.
    """

    engine_id: EngineId
    request_id: ReqId
    generation: int | None


@dataclass
class _ParallelPullFlight:
    """Single-flight state for consumers of one producer offer.

    :ivar expected_consumers: Number of logical consumers in the producer lease.
    :ivar num_tokens: Exact prompt extent shared by every logical consumer.
    :ivar baseline_num_computed_tokens: APC hit observed when the leader started.
    :ivar publication_num_computed_tokens: APC hit proving full-prefix publication.
    :ivar leader_request_id: Consumer currently performing the full suffix pull.
    :ivar released_request_ids: Consumers allowed to enter normal receive setup.
    :ivar completed_request_ids: Consumers whose receive setup or abort committed.
    """

    expected_consumers: int
    num_tokens: int
    baseline_num_computed_tokens: int
    publication_num_computed_tokens: int
    leader_request_id: ReqId | None
    released_request_ids: set[ReqId]
    completed_request_ids: set[ReqId]


def _producer_offer_identity(
    params: dict[str, Any],
) -> _ProducerOfferIdentity | None:
    """Extract the immutable producer identity required for single-flight.

    :param params: Request KV-transfer parameters.
    :returns: Producer identity, or ``None`` when required identity is absent.
    :raises ValueError: If an explicit offer generation is invalid.
    """
    engine_id = params.get("remote_engine_id")
    request_id = params.get("remote_request_id")
    if (
        type(engine_id) is not str
        or len(engine_id) == 0
        or type(request_id) is not str
        or len(request_id) == 0
    ):
        return None

    generation = params.get("p2d_offer_generation")
    if generation is not None and (type(generation) is not int or generation < 0):
        raise ValueError("p2d_offer_generation must be a non-negative integer")
    return _ProducerOfferIdentity(engine_id, request_id, generation)


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

    _parallel_pull_single_flight_enabled: bool
    _parallel_pull_publication_granularity: int
    _parallel_pull_flights: dict[_ProducerOfferIdentity, _ParallelPullFlight]

    def __init__(
        self,
        vllm_config: "VllmConfig",
        engine_id: str,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config, engine_id, kv_cache_config)
        self._parallel_pull_single_flight_enabled: bool = (
            vllm_config.cache_config.enable_prefix_caching
        )
        self._parallel_pull_publication_granularity = resolve_kv_cache_block_sizes(
            kv_cache_config, vllm_config
        )[0]
        self._parallel_pull_flights: dict[
            _ProducerOfferIdentity, _ParallelPullFlight
        ] = {}

    def _remove_completed_parallel_pull(
        self,
        offer: _ProducerOfferIdentity,
        flight: _ParallelPullFlight,
    ) -> None:
        """Remove an offer after every consumer commits receive setup.

        :param offer: Producer offer owning the flight.
        :param flight: Current single-flight state for the offer.
        """
        if len(flight.completed_request_ids) < flight.expected_consumers:
            return
        if len(flight.completed_request_ids) > flight.expected_consumers:
            raise RuntimeError("producer offer completed too many consumers")
        del self._parallel_pull_flights[offer]

    def _claim_parallel_pull_leader(
        self,
        offer: _ProducerOfferIdentity,
        flight: _ParallelPullFlight,
        request_id: ReqId,
        num_computed_tokens: int,
    ) -> None:
        """Make one consumer responsible for the offer's uncached suffix.

        :param offer: Producer offer whose pull is being coordinated.
        :param flight: Current single-flight state for the offer.
        :param request_id: Consumer becoming the leader.
        :param num_computed_tokens: APC hit visible before the leader's pull.
        """
        flight.leader_request_id = request_id
        flight.baseline_num_computed_tokens = num_computed_tokens
        flight.released_request_ids.add(request_id)
        logger.info(
            "[single-flight] leader=%s offer=%s/%s generation=%s "
            "baseline_tokens=%d consumers=%d",
            request_id,
            offer.engine_id,
            offer.request_id,
            offer.generation,
            num_computed_tokens,
            flight.expected_consumers,
        )

    def _coordinate_parallel_pull(
        self,
        request: "Request",
        params: dict[str, Any],
        num_computed_tokens: int,
        num_external_tokens: int,
        max_cacheable_tokens: int,
    ) -> bool:
        """Allow one cold pull until APC exposes its published prefix.

        Followers are reconsidered from a fresh APC lookup on every scheduler
        pass. A follower is released only when the complete publishable prefix
        is present. Async external blocks enter APC only after ``finished_recving``
        promotes the leader, so that hit also proves the leader's receive reached
        its authoritative terminal state. The follower then follows the ordinary
        receive path, including its own tail read or empty-read notification, so
        the producer still receives one completion from every logical consumer.

        :param request: Candidate decoder consumer.
        :param params: Request KV-transfer parameters.
        :param num_computed_tokens: Prefix tokens found by APC for this attempt.
        :param num_external_tokens: Remaining tokens offered by the producer.
        :param max_cacheable_tokens: Largest APC hit this prompt can publish.
        :returns: Whether normal allocation and receive setup may proceed.
        """
        expected_consumers = _expected_consumers(params)
        if (
            not self._parallel_pull_single_flight_enabled
            or expected_consumers == 1
            or request.skip_reading_prefix_cache
        ):
            return True

        offer = _producer_offer_identity(params)
        if offer is None:
            return True

        flight = self._parallel_pull_flights.get(offer)
        if flight is None:
            if num_external_tokens <= 0 or max_cacheable_tokens <= num_computed_tokens:
                return True
            flight = _ParallelPullFlight(
                expected_consumers=expected_consumers,
                num_tokens=request.num_tokens,
                baseline_num_computed_tokens=num_computed_tokens,
                publication_num_computed_tokens=max_cacheable_tokens,
                leader_request_id=None,
                released_request_ids=set(),
                completed_request_ids=set(),
            )
            self._parallel_pull_flights[offer] = flight
            self._claim_parallel_pull_leader(
                offer, flight, request.request_id, num_computed_tokens
            )
            return True

        if flight.expected_consumers != expected_consumers:
            raise ValueError("expected_consumers changed for an active producer offer")
        if flight.num_tokens != request.num_tokens:
            raise ValueError("prompt length changed for an active producer offer")
        if flight.publication_num_computed_tokens != max_cacheable_tokens:
            raise ValueError(
                "APC publication target changed for an active producer offer"
            )
        if request.request_id in flight.released_request_ids:
            return True

        if num_computed_tokens >= flight.publication_num_computed_tokens:
            flight.released_request_ids.add(request.request_id)
            logger.info(
                "[single-flight] follower=%s observed offer=%s/%s "
                "generation=%s publication %d->%d tokens",
                request.request_id,
                offer.engine_id,
                offer.request_id,
                offer.generation,
                flight.baseline_num_computed_tokens,
                num_computed_tokens,
            )
            return True

        if flight.leader_request_id is None:
            self._claim_parallel_pull_leader(
                offer, flight, request.request_id, num_computed_tokens
            )
            return True

        return False

    def _complete_parallel_pull_allocation(
        self,
        request: "Request",
        params: dict[str, Any],
    ) -> None:
        """Record that a released consumer committed its receive metadata.

        :param request: Consumer whose allocation and receive setup committed.
        :param params: Request KV-transfer parameters.
        """
        offer = _producer_offer_identity(params)
        if offer is None:
            return
        flight = self._parallel_pull_flights.get(offer)
        if flight is None:
            return
        if request.request_id not in flight.released_request_ids:
            raise RuntimeError("parallel pull allocation committed before release")
        flight.completed_request_ids.add(request.request_id)
        self._remove_completed_parallel_pull(offer, flight)

    def _finish_parallel_pull_request(
        self,
        request: "Request",
        params: dict[str, Any],
    ) -> None:
        """Retire or replace a single-flight participant that has finished.

        :param request: Finished decoder consumer.
        :param params: Request KV-transfer parameters.
        """
        offer = _producer_offer_identity(params)
        if offer is None:
            return
        flight = self._parallel_pull_flights.get(offer)
        if flight is None:
            return

        flight.released_request_ids.add(request.request_id)
        flight.completed_request_ids.add(request.request_id)
        if flight.leader_request_id == request.request_id:
            flight.leader_request_id = None
        self._remove_completed_parallel_pull(offer, flight)

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
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
            max_cacheable_tokens = max(0, request.num_tokens - 1)
            max_cacheable_tokens -= (
                max_cacheable_tokens % self._parallel_pull_publication_granularity
            )
            if not self._coordinate_parallel_pull(
                request,
                params,
                num_computed_tokens,
                count,
                max_cacheable_tokens,
            ):
                return None, False
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

        was_remote_prefill = params.get("do_remote_prefill") is True
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
            if was_remote_prefill:
                self._complete_parallel_pull_allocation(request, params)

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
        self._finish_parallel_pull_request(request, params)

        if params.get("do_remote_prefill"):
            # The request finished after admission but before receive setup.
            # Keep it alive until the empty receive notifies the producer and
            # returns through finished_recving; freeing it here would make that
            # completion refer to a request the scheduler no longer owns.
            self._reqs_need_recv[request.request_id] = (request, [])
            params["do_remote_prefill"] = False
            return True, None

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

        settled_num_computed_tokens = max(
            0,
            request.num_computed_tokens - request.num_in_flight_tokens,
        )
        block_ids = self._get_transferable_block_ids(
            block_ids,
            settled_num_computed_tokens,
        )
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
                    valid_token_extent=settled_num_computed_tokens,
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

            remote_num_tokens = settled_num_computed_tokens

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
