# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import queue
import threading
from collections import deque
from typing import Any
from unittest.mock import Mock, call, patch

import msgspec
import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.distributed.kv_transfer.kv_connector.v1.base import SupportsHMA
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_scheduler import (
    NixlBaseConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import (
    NixlBaseConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    PULL_READ_COMPLETE_PREFIX,
    PullReadComplete,
    RemoteMeta,
    ReqMeta,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_worker import (
    NixlPullConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.push_worker import (
    NixlPushConnectorWorker,
)
from vllm.distributed.kv_transfer.staging_ownership import (
    StagingRangeAllocator,
    StagingSafetyError,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    SlidingWindowSpec,
)
from vllm.v1.outputs import (
    KVConnectorOutput,
    KVTransferFailure,
    KVTransferFailureReason,
    ModelRunnerOutput,
)
from vllm.v1.request import FinishReason, Request, RequestStatus

from .utils import create_request, create_scheduler, create_vllm_config

pytestmark = pytest.mark.cpu_test


class _HMAConnector(SupportsHMA):
    """Expose an asynchronous multi-group receive contract to the scheduler."""

    num_external_tokens: int
    delay_free_blocks: bool
    finished_block_ids: tuple[list[int], ...] | None

    def __init__(
        self,
        num_external_tokens: int,
        delay_free_blocks: bool = False,
    ) -> None:
        """Initialize the connector fixture.

        :param num_external_tokens: Prompt tokens supplied by the remote cache.
        :param delay_free_blocks: Whether terminal connector cleanup retains blocks.
        """
        self.num_external_tokens = num_external_tokens
        self.delay_free_blocks = delay_free_blocks
        self.finished_block_ids = None

    def on_new_request(self, request: Request) -> None:
        """Accept a newly admitted scheduler request.

        :param request: Admitted request.
        """

    def get_num_new_matched_tokens(
        self,
        request: Request,
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        """Return the asynchronous external prefix.

        :param request: Request under admission.
        :param num_computed_tokens: Locally matched prefix length.
        :returns: External prefix length and asynchronous-load marker.
        """
        return self.num_external_tokens, True

    def update_state_after_alloc(
        self,
        request: Request,
        blocks: Any,
        num_external_tokens: int,
    ) -> None:
        """Accept allocated cache blocks.

        :param request: Request under admission.
        :param blocks: Allocated cache blocks.
        :param num_external_tokens: External prefix length.
        """

    def build_connector_meta(self, scheduler_output: Any) -> None:
        """Return no worker metadata for the scheduler fixture.

        :param scheduler_output: Current scheduler output.
        """

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        """Accept a worker connector output.

        :param connector_output: Worker connector output.
        """

    def request_finished_all_groups(
        self,
        request: Request,
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, None]:
        """Capture the complete block roster released by terminal cleanup.

        :param request: Terminal request.
        :param block_ids: Per-group logical block IDs.
        :returns: Delayed-free decision and no transfer metadata.
        """
        self.finished_block_ids = block_ids
        return self.delay_free_blocks, None

    def get_kv_connector_stats(self) -> None:
        """Return no connector statistics."""

    def take_events(self) -> tuple[()]:
        """Return no cache events.

        :returns: Empty event sequence.
        """
        return ()


def _make_13_group_config(block_size: int = 16) -> KVCacheConfig:
    """Build a mixed thirteen-group HMA cache configuration.

    :param block_size: Tokens represented by each logical block.
    :returns: Cache configuration with twelve full and one sliding group.
    """
    groups = [
        KVCacheGroupSpec(
            [f"full_layer_{index}"],
            FullAttentionSpec(
                block_size=block_size,
                num_kv_heads=1,
                head_size=1,
                dtype=torch.float32,
            ),
        )
        for index in range(12)
    ]
    groups.append(
        KVCacheGroupSpec(
            ["sliding_layer"],
            SlidingWindowSpec(
                block_size=block_size,
                num_kv_heads=1,
                head_size=1,
                dtype=torch.float32,
                sliding_window=64,
            ),
        )
    )
    return KVCacheConfig(
        num_blocks=1000,
        kv_cache_tensors=[],
        kv_cache_groups=groups,
    )


def _make_13_full_group_config(block_size: int = 16) -> KVCacheConfig:
    """Build a thirteen-group all-full HMA cache configuration.

    :param block_size: Tokens represented by each logical block.
    :returns: Cache configuration with thirteen full-attention groups.
    """
    return KVCacheConfig(
        num_blocks=1000,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                [f"full_layer_{index}"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            )
            for index in range(13)
        ],
    )


def _failure(
    invalid_block_ids: set[int] | frozenset[int],
    reason: KVTransferFailureReason = KVTransferFailureReason.TRANSFER,
) -> KVTransferFailure:
    """Build a typed failure fixture.

    :param invalid_block_ids: Affected logical blocks.
    :param reason: Failure classification.
    :returns: Immutable request-scoped failure.
    """
    return KVTransferFailure(
        reason=reason,
        invalid_block_ids=frozenset(invalid_block_ids),
    )


def _model_output(connector_output: KVConnectorOutput) -> ModelRunnerOutput:
    """Build an empty model output carrying connector state.

    :param connector_output: Worker connector result.
    :returns: Model output suitable for scheduler or rank aggregation tests.
    """
    return ModelRunnerOutput(
        req_ids=[],
        req_id_to_index={},
        sampled_token_ids=[],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=None,
        kv_connector_output=connector_output,
    )


def _make_worker(
    local_block_ids: tuple[list[int], ...],
    request_id: str = "hma-failed-receive",
    worker_type: type[NixlBaseConnectorWorker] = NixlBaseConnectorWorker,
) -> NixlBaseConnectorWorker:
    """Build the smallest worker state that exercises receive completion.

    :param local_block_ids: Per-group logical destination blocks.
    :param request_id: Decoder request identifier.
    :param worker_type: Concrete worker class needed by the focused fixture.
    :returns: Worker fixture with no active transfer handles.
    """
    worker = object.__new__(worker_type)
    worker.engine_id = "decode-engine"
    worker.tp_rank = 0
    worker.transfer_topo = Mock()
    worker._phase_separate_transfer_decode = False
    worker._phase_separation_instrumented = False
    worker._recving_transfers = {}
    worker._recving_metadata = {
        request_id: ReqMeta(
            local_block_ids=local_block_ids,
            local_physical_block_ids=tuple(
                [block_id + 10_000 for block_id in group] for group in local_block_ids
            ),
            tp_size=1,
        )
    }
    worker._coalesce_plans = {}
    worker._coalesce_pending = deque()
    worker._failed_recv_outcomes = queue.Queue()
    worker._failed_recv_pending = {}
    worker._completed_failed_recv_outcomes = queue.Queue()
    worker._invalid_block_ids = queue.Queue()
    worker._released_rids = {}
    worker._rid_completion_counts = {}
    worker._reqs_to_send = {}
    worker._reqs_to_process = set()
    worker._has_mamba = False
    worker.use_host_buffer = False
    worker.use_mla = False
    worker.enable_heterogeneous_attn_post_process = False
    worker.nixl_wrapper = Mock()
    worker.xfer_stats = Mock()
    worker._service_heartbeats = Mock()
    worker._get_new_notifs = Mock(return_value=set())
    worker._audit_tick = Mock()
    worker._localization_record_event = Mock()
    worker._localization_pre_read_plans = {}
    return worker


def test_hma_worker_failure_reports_every_group_as_one_terminal() -> None:
    """A worker failure names every group without entering generic invalidation."""
    local_block_ids = tuple(
        [group_index * 10 + offset for offset in range(3)] for group_index in range(13)
    )
    request_id = "hma-failed-receive"
    worker = _make_worker(local_block_ids, request_id)

    worker._handle_failed_transfer(request_id, None)
    _, finished_recving = worker.get_finished()
    failed_recving = worker.get_failed_recving()

    expected_block_ids = {
        block_id for group_block_ids in local_block_ids for block_id in group_block_ids
    }
    assert finished_recving == {request_id}
    assert failed_recving == {request_id: _failure(expected_block_ids)}
    assert worker.get_block_ids_with_load_errors() == set()
    assert request_id not in worker._recving_metadata
    worker.xfer_stats.record_failed_transfer.assert_called_once_with()


def test_release_fence_reports_integrity_failure_for_every_group() -> None:
    """A late transfer is rejected without publishing any destination group."""
    local_block_ids = tuple(
        [group_index * 10 + offset for offset in range(3)] for group_index in range(13)
    )
    request_id = "late-decode-request"
    remote_request_id = "released-prefill-request"
    worker = _make_worker(local_block_ids, request_id)
    worker._recving_metadata[request_id].remote = RemoteMeta(
        block_ids=local_block_ids,
        host="remote-host",
        port=1234,
        engine_id="remote-engine",
        request_id=remote_request_id,
    )
    worker._recving_transfers = {request_id: []}
    worker._released_rids = {remote_request_id: 1.0}

    _, finished_recving = worker.get_finished()
    failures = worker.get_failed_recving()

    expected_block_ids = {
        block_id for group_block_ids in local_block_ids for block_id in group_block_ids
    }
    assert finished_recving == {request_id}
    assert failures == {
        request_id: _failure(
            expected_block_ids,
            reason=KVTransferFailureReason.INTEGRITY,
        )
    }
    assert worker.get_block_ids_with_load_errors() == set()


def test_failed_receive_waits_for_all_known_worker_handles() -> None:
    """A failed handle cannot terminate while a sibling handle remains active."""
    request_id = "multi-handle-receive"
    worker = _make_worker(([7],), request_id)
    worker._recving_transfers = {request_id: [22]}
    worker.nixl_wrapper.check_xfer_state.side_effect = ["PROC", "DONE"]
    worker.nixl_wrapper.get_xfer_telemetry.return_value = Mock()

    worker._handle_failed_transfer(request_id, 11)
    _, first_finished = worker.get_finished()
    first_failures = worker.get_failed_recving()

    assert first_finished == set()
    assert first_failures == {}
    assert request_id in worker._recving_metadata
    assert request_id in worker._failed_recv_pending

    _, second_finished = worker.get_finished()
    second_failures = worker.get_failed_recving()

    assert second_finished == {request_id}
    assert second_failures == {request_id: _failure({7})}
    assert worker.nixl_wrapper.release_xfer_handle.call_args_list == [
        call(11),
        call(22),
    ]


def test_failed_receive_cannot_reclaim_unquiesced_coalesced_staging() -> None:
    """Request failure preserves a range while any native writer may run."""
    allocator = StagingRangeAllocator(1024)
    plan = allocator.create_plan(
        owner_id="decode:0:0",
        request_id="coalesced-request",
        size=512,
        source_ranks=(0,),
        remote_engine_id="prefill-engine",
    )
    assert plan is not None
    plan.begin_prepare(0)
    plan.attach_handle(0, 17)
    plan.begin_post(0)
    plan.record_post_result(0, "PROC")
    plan.seal_posting()
    worker = object.__new__(NixlBaseConnectorWorker)
    worker._staging_allocator = allocator
    worker._coalesce_plans = {plan.request_id: plan}

    with pytest.raises(StagingSafetyError, match="unquiesced"):
        worker._discard_completed_coalesced_plan(plan.request_id)

    assert allocator.active == {plan.lease.generation: plan}
    assert plan.released is False


def test_failed_receive_reclaims_only_completed_coalesced_staging() -> None:
    """A successful native terminal may be discarded before publication."""
    allocator = StagingRangeAllocator(1024)
    plan = allocator.create_plan(
        owner_id="decode:0:0",
        request_id="coalesced-request",
        size=512,
        source_ranks=(0,),
        remote_engine_id="prefill-engine",
    )
    assert plan is not None
    plan.begin_prepare(0)
    plan.attach_handle(0, 17)
    plan.begin_post(0)
    plan.record_post_result(0, "DONE")
    plan.mark_native_released(0)
    plan.seal_posting()
    worker = object.__new__(NixlBaseConnectorWorker)
    worker._staging_allocator = allocator
    worker._coalesce_plans = {plan.request_id: plan}

    worker._discard_completed_coalesced_plan(plan.request_id)

    assert plan.released
    assert plan.request_id not in worker._coalesce_plans
    assert allocator.active == {}
    assert allocator.free_bytes == allocator.capacity


def test_failed_receive_reclaims_quiescent_failed_coalesced_staging() -> None:
    """A failed plan reaches terminal after all possible writers are quiescent."""
    request_id = "quiescent-failed-plan"
    worker = _make_worker(([9],), request_id)
    allocator = StagingRangeAllocator(1024)
    plan = allocator.create_plan(
        owner_id="decode:0:0",
        request_id=request_id,
        size=512,
        source_ranks=(0, 1),
        remote_engine_id="prefill-engine",
    )
    assert plan is not None
    plan.begin_prepare(0)
    plan.attach_handle(0, 17)
    plan.begin_post(0)
    plan.record_post_result(0, "DONE")
    plan.begin_prepare(1)
    plan.record_prepare_failure(1, "rank 1 preparation failed")
    plan.seal_posting()
    worker._staging_allocator = allocator
    worker._coalesce_plans = {request_id: plan}

    worker._handle_failed_transfer(request_id, None)
    _, finished_recving = worker.get_finished()

    assert finished_recving == {request_id}
    assert worker.get_failed_recving() == {request_id: _failure({9})}
    assert plan.released
    assert allocator.free_bytes == allocator.capacity
    worker.nixl_wrapper.release_xfer_handle.assert_called_once_with(17)


def test_sealed_unposted_handle_releases_without_done_evidence() -> None:
    """An unposted handle is freed without claiming a native DONE observation."""
    request_id = "sealed-unposted-plan"
    worker = _make_worker(([9],), request_id)
    allocator = StagingRangeAllocator(1024)
    plan = allocator.create_plan(
        owner_id="decode:0:0",
        request_id=request_id,
        size=512,
        source_ranks=(0,),
        remote_engine_id="prefill-engine",
    )
    assert plan is not None
    plan.begin_prepare(0)
    plan.attach_handle(0, 17)
    plan.fail("transfer phase closed before native post")
    plan.seal_posting()
    worker._staging_allocator = allocator
    worker._coalesce_plans = {request_id: plan}

    worker._release_quiescent_failed_coalesced_plan(plan)

    assert plan.released
    assert plan.slots[0].native_released is False
    assert allocator.free_bytes == allocator.capacity
    worker.nixl_wrapper.release_xfer_handle.assert_called_once_with(17)


def test_multirank_prepost_failure_discharge_is_rank_exact() -> None:
    """Only untouched P ranks receive explicit admitted-child completion."""
    request_id = "decode-request"
    producer_request_id = "prefill-request"
    producer_engine_id = "prefill-engine"
    producer_tp_size = 4
    worker = _make_worker(
        ([9],),
        request_id,
        worker_type=NixlPullConnectorWorker,
    )
    assert isinstance(worker, NixlPullConnectorWorker)
    worker.world_size = 1
    meta = worker._recving_metadata[request_id]
    meta.tp_size = producer_tp_size
    meta.remote = RemoteMeta(
        block_ids=([19],),
        host="producer-host",
        port=1234,
        engine_id=producer_engine_id,
        request_id=producer_request_id,
        expected_consumers=1,
        consumer_tp_size=1,
    )
    worker._remote_agents = {
        producer_engine_id: {
            producer_rank: f"producer-agent-{producer_rank}"
            for producer_rank in range(producer_tp_size)
        }
    }
    notification_id = worker._read_completion_notification(request_id, meta)

    allocator = StagingRangeAllocator(1024)
    plan = allocator.create_plan(
        owner_id="decode:0:0",
        request_id=request_id,
        size=512,
        source_ranks=(0, 1, 2),
        remote_engine_id=producer_engine_id,
    )
    assert plan is not None
    plan.begin_prepare(0)
    plan.attach_handle(0, 10)
    plan.begin_post(0)
    plan.record_post_result(0, "DONE")
    plan.begin_prepare(1)
    plan.attach_handle(1, 11)
    plan.fail("rank 1 failed before native post")
    worker._staging_allocator = allocator
    worker._coalesce_plans = {request_id: plan}

    worker._finish_quiescent_coalesced_failure(
        plan,
        "rank 1 failed before native post",
    )
    worker._notify_failed_coalesced_producer_ranks(
        meta,
        plan,
        notification_id,
    )

    assert plan.released
    assert worker.nixl_wrapper.release_xfer_handle.call_args_list == [
        call(10),
        call(11),
    ]
    assert worker.nixl_wrapper.send_notif.call_args_list == [
        call("producer-agent-1", notif_msg=notification_id),
        call("producer-agent-2", notif_msg=notification_id),
        call("producer-agent-3", notif_msg=notification_id),
    ]
    proof = msgspec.msgpack.decode(
        notification_id[len(PULL_READ_COMPLETE_PREFIX) :],
        type=PullReadComplete,
    )
    assert proof == PullReadComplete(
        producer_request_id=producer_request_id,
        consumer_request_id=request_id,
        consumer_index=0,
        consumer_rank=0,
        consumer_tp_size=1,
        expected_consumers=1,
    )


def test_coalesced_prepare_failure_becomes_request_terminal() -> None:
    """A pre-write native failure is reclaimed and reported without fail-stop."""
    request_id = "coalesced-prepare-failure"
    worker = _make_worker(([9],), request_id)
    allocator = StagingRangeAllocator(1024)
    plan = allocator.create_plan(
        owner_id="decode:0:0",
        request_id=request_id,
        size=512,
        source_ranks=(0,),
        remote_engine_id="prefill-engine",
    )
    assert plan is not None
    plan.begin_prepare(0)
    worker._staging_allocator = allocator
    worker._coalesce_plans = {request_id: plan}
    worker.nixl_wrapper.initialize_xfer.side_effect = RuntimeError(
        "injected preparation failure"
    )

    with patch.object(plan, "fail", wraps=plan.fail) as fail:
        posted = worker._initialize_and_post_coalesced(
            plan,
            0,
            Mock(),
            Mock(),
            "remote-agent",
            b"notification",
        )
    _, finished_recving = worker.get_finished()

    assert posted is False
    fail.assert_called_once()
    assert finished_recving == {request_id}
    assert worker.get_failed_recving() == {request_id: _failure({9})}
    assert plan.released
    assert allocator.free_bytes == allocator.capacity
    worker.nixl_wrapper.transfer.assert_not_called()


def test_failed_receive_keeps_tombstoned_coalesced_staging_pinned() -> None:
    """An uncertain native failure remains fail-stop and cannot be reused."""
    request_id = "tombstoned-failed-plan"
    worker = _make_worker(([9],), request_id)
    allocator = StagingRangeAllocator(1024)
    plan = allocator.create_plan(
        owner_id="decode:0:0",
        request_id=request_id,
        size=512,
        source_ranks=(0,),
        remote_engine_id="prefill-engine",
    )
    assert plan is not None
    plan.begin_prepare(0)
    plan.attach_handle(0, 17)
    plan.begin_post(0)
    plan.record_post_result(0, "ERR")
    plan.seal_posting()
    worker._staging_allocator = allocator
    worker._coalesce_plans = {request_id: plan}
    worker._handle_failed_transfer(request_id, None)

    with pytest.raises(StagingSafetyError, match="safety proof failed"):
        worker.get_finished()

    assert allocator.active == {plan.lease.generation: plan}
    assert plan.released is False
    worker.nixl_wrapper.release_xfer_handle.assert_not_called()


def test_failed_receive_waits_for_all_workers_and_wins_over_success() -> None:
    """Rank aggregation publishes one failed terminal only after every rank ends."""
    request_id = "request"
    failure = _failure({17, 18})
    aggregator = KVOutputAggregator(expected_finished_count=2)

    first = aggregator.aggregate(
        [
            _model_output(
                KVConnectorOutput(
                    finished_recving={request_id},
                    failed_recving={request_id: failure},
                )
            ),
            _model_output(KVConnectorOutput()),
        ]
    )

    assert first is not None and first.kv_connector_output is not None
    assert first.kv_connector_output.finished_recving is None
    assert first.kv_connector_output.failed_recving == {}

    second = aggregator.aggregate(
        [
            _model_output(
                KVConnectorOutput(
                    finished_recving={request_id},
                    failed_recving={request_id: failure},
                )
            ),
            _model_output(KVConnectorOutput(finished_recving={request_id})),
        ]
    )

    assert second is not None and second.kv_connector_output is not None
    assert second.kv_connector_output.finished_recving is None
    assert second.kv_connector_output.failed_recving == {request_id: failure}


def test_failure_without_same_worker_terminal_is_rejected() -> None:
    """Malformed worker output cannot poison a later request terminal."""
    aggregator = KVOutputAggregator(expected_finished_count=1)
    with pytest.raises(RuntimeError, match="must carry"):
        aggregator.aggregate(
            [
                _model_output(
                    KVConnectorOutput(
                        failed_recving={"request": _failure({1})},
                    )
                )
            ]
        )


def test_scheduler_stops_failed_receive_heartbeat_once() -> None:
    """The paired healthy/failed worker terminal consumes one heartbeat owner."""
    scheduler = object.__new__(NixlBaseConnectorScheduler)
    scheduler._stop_heartbeat = Mock()
    request_id = "failed-receive"

    scheduler.update_connector_output(
        KVConnectorOutput(
            finished_recving={request_id},
            failed_recving={request_id: _failure({1})},
        )
    )

    scheduler._stop_heartbeat.assert_called_once_with(request_id)


def test_hma_failure_is_not_promoted_cached_or_forwarded() -> None:
    """A typed HMA receive failure atomically terminates and frees all groups."""
    num_prompt_tokens = 64
    num_external_tokens = 48
    kv_cache_config = _make_13_group_config()
    vllm_config = create_vllm_config(
        max_num_batched_tokens=num_prompt_tokens,
        kv_load_failure_policy="recompute",
    )
    scheduler = create_scheduler(
        vllm_config,
        num_blocks=kv_cache_config.num_blocks,
        kv_cache_config=kv_cache_config,
    )
    connector = _HMAConnector(num_external_tokens)
    scheduler.connector = connector
    baseline_free_blocks = (
        scheduler.kv_cache_manager.block_pool.free_block_queue.num_free_blocks
    )
    baseline_ref_counts = [
        block.ref_cnt for block in scheduler.kv_cache_manager.block_pool.blocks
    ]

    request = create_request(num_tokens=num_prompt_tokens)
    scheduler.add_request(request)
    scheduler_output = scheduler.schedule()
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    local_block_ids = scheduler.kv_cache_manager.get_block_ids(request.request_id)
    assert len(local_block_ids) == 13
    invalid_block_ids = {
        block_id for group_block_ids in local_block_ids for block_id in group_block_ids
    }
    model_output = _model_output(
        KVConnectorOutput(
            failed_recving={request.request_id: _failure(invalid_block_ids)},
        )
    )
    cache_calls: list[tuple[str, int]] = []
    original_cache_blocks = scheduler.kv_cache_manager.cache_blocks

    def cache_blocks_spy(cached_request: Request, num_tokens: int) -> None:
        """Record any attempted publication of failed KV.

        :param cached_request: Request whose blocks would be published.
        :param num_tokens: Prefix length proposed for publication.
        """
        cache_calls.append((cached_request.request_id, num_tokens))
        original_cache_blocks(cached_request, num_tokens)

    with patch.object(
        scheduler.kv_cache_manager,
        "cache_blocks",
        side_effect=cache_blocks_spy,
    ):
        outputs = scheduler.update_from_output(scheduler_output, model_output)
        next_scheduler_output = scheduler.schedule()

    assert cache_calls == []
    assert request.status == RequestStatus.FINISHED_ERROR
    assert request.get_finished_reason() == FinishReason.ERROR
    assert request.request_id not in scheduler.requests
    assert request.request_id not in next_scheduler_output.num_scheduled_tokens
    assert connector.finished_block_ids == local_block_ids
    for manager in scheduler.kv_cache_manager.coordinator.single_type_managers:
        assert request.request_id not in manager.req_to_blocks
    assert (
        scheduler.kv_cache_manager.block_pool.free_block_queue.num_free_blocks
        == baseline_free_blocks
    )
    assert [
        block.ref_cnt for block in scheduler.kv_cache_manager.block_pool.blocks
    ] == baseline_ref_counts
    engine_outputs = next(iter(outputs.values())).outputs
    assert len(engine_outputs) == 1
    assert engine_outputs[0].request_id == request.request_id
    assert engine_outputs[0].finish_reason == FinishReason.ERROR


def test_failed_receive_releases_previously_aborted_request() -> None:
    """A late failed terminal releases only blocks retained by that receive."""
    kv_cache_config = _make_13_group_config()
    vllm_config = create_vllm_config(
        max_num_batched_tokens=64,
        kv_load_failure_policy="recompute",
    )
    scheduler = create_scheduler(
        vllm_config,
        num_blocks=kv_cache_config.num_blocks,
        kv_cache_config=kv_cache_config,
    )
    scheduler.connector = _HMAConnector(num_external_tokens=48)
    baseline_free_blocks = (
        scheduler.kv_cache_manager.block_pool.free_block_queue.num_free_blocks
    )
    request = create_request(num_tokens=64)
    scheduler.add_request(request)
    scheduler_output = scheduler.schedule()
    block_ids = scheduler.kv_cache_manager.get_block_ids(request.request_id)
    invalid_block_ids = {
        block_id for group_block_ids in block_ids for block_id in group_block_ids
    }

    scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
    assert request.request_id in scheduler.requests
    assert request.request_id in scheduler._receive_delayed_free_req_ids

    outputs = scheduler.update_from_output(
        scheduler_output,
        _model_output(
            KVConnectorOutput(
                failed_recving={request.request_id: _failure(invalid_block_ids)},
            )
        ),
    )

    assert request.status == RequestStatus.FINISHED_ABORTED
    assert request.request_id not in scheduler.requests
    assert request.request_id not in scheduler._receive_delayed_free_req_ids
    assert (
        scheduler.kv_cache_manager.block_pool.free_block_queue.num_free_blocks
        == baseline_free_blocks
    )
    assert all(len(engine_output.outputs) == 0 for engine_output in outputs.values())


def test_stale_healthy_receive_does_not_free_send_retained_request() -> None:
    """Receive completion cannot consume ownership retained for a send."""
    kv_cache_config = _make_13_group_config()
    vllm_config = create_vllm_config(max_num_batched_tokens=64)
    scheduler = create_scheduler(
        vllm_config,
        num_blocks=kv_cache_config.num_blocks,
        kv_cache_config=kv_cache_config,
    )
    request = create_request(num_tokens=64)
    scheduler.add_request(request)
    scheduler.schedule()
    assert request.status == RequestStatus.RUNNING

    scheduler.connector = _HMAConnector(
        num_external_tokens=0,
        delay_free_blocks=True,
    )
    scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
    retained_free_blocks = (
        scheduler.kv_cache_manager.block_pool.free_block_queue.num_free_blocks
    )
    assert request.request_id in scheduler.requests
    assert request.request_id not in scheduler._receive_delayed_free_req_ids

    scheduler._update_from_kv_xfer_finished(
        KVConnectorOutput(finished_recving={request.request_id})
    )

    assert request.request_id in scheduler.requests
    assert (
        scheduler.kv_cache_manager.block_pool.free_block_queue.num_free_blocks
        == retained_free_blocks
    )
    scheduler._free_blocks(request)


def test_multi_group_generic_invalidation_resets_the_complete_request() -> None:
    """Legacy invalid-block recovery handles HMA without tuple-unpack failure."""
    kv_cache_config = _make_13_full_group_config()
    vllm_config = create_vllm_config(
        max_num_batched_tokens=64,
        kv_load_failure_policy="recompute",
    )
    vllm_config.cache_config.enable_prefix_caching = False
    scheduler = create_scheduler(
        vllm_config,
        num_blocks=kv_cache_config.num_blocks,
        kv_cache_config=kv_cache_config,
    )
    scheduler.connector = _HMAConnector(num_external_tokens=48)
    request = create_request(num_tokens=64)
    scheduler.add_request(request)
    scheduler.schedule()
    block_ids = scheduler.kv_cache_manager.get_block_ids(request.request_id)
    invalid_block_id = block_ids[12][0]

    affected, affected_tokens, blocks_to_evict = (
        scheduler._update_requests_with_invalid_blocks(
            [request],
            invalid_block_ids={invalid_block_id},
            num_scheduled_tokens={},
        )
    )

    assert affected == {request.request_id}
    assert affected_tokens == 48
    assert request.num_computed_tokens == 0
    assert blocks_to_evict == {
        block_id for group_block_ids in block_ids for block_id in group_block_ids
    }


def test_push_send_failure_is_not_reported_as_failed_receive() -> None:
    """The shared polling helper preserves the distinction between send and receive."""
    worker = object.__new__(NixlPushConnectorWorker)
    worker.shutdown = Mock()
    worker._push_writer_wake = Mock()
    worker._sending_transfers_lock = threading.Lock()
    worker._sending_transfers = {"send-request": [7]}
    worker._reqs_to_send = {}
    worker._reqs_to_process = set()
    worker.consumer_notification_counts_by_req = {}
    worker._evict_finished_inbox = queue.Queue()
    worker._failed_recv_outcomes = queue.Queue()
    worker._failed_recv_pending = {}
    worker._completed_failed_recv_outcomes = queue.Queue()
    worker._invalid_block_ids = queue.Queue()
    worker.nixl_wrapper = Mock()
    worker.nixl_wrapper.check_xfer_state.return_value = "ERR"
    worker.xfer_stats = Mock()
    worker._log_failure = Mock()

    with patch.object(
        NixlBaseConnectorWorker,
        "get_finished",
        return_value=(set(), set()),
    ):
        done_sending, done_recving = worker.get_finished()

    assert done_sending == {"send-request"}
    assert done_recving == set()
    assert worker._failed_recv_outcomes.empty()
    assert worker._completed_failed_recv_outcomes.empty()
    worker.nixl_wrapper.release_xfer_handle.assert_called_once_with(7)
    worker.xfer_stats.record_failed_transfer.assert_called_once_with()
