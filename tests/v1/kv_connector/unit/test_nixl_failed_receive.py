# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import queue
import threading
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.distributed.kv_transfer.kv_connector.v1.base import SupportsHMA
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import (
    NixlBaseConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    RemoteMeta,
    ReqMeta,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_worker import (
    NixlPullConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.push_worker import (
    NixlPushConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import ReadSpec
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    SlidingWindowSpec,
)
from vllm.v1.request import FinishReason, Request, RequestStatus

from .utils import (
    create_model_runner_output,
    create_request,
    create_scheduler,
    create_vllm_config,
)

pytestmark = pytest.mark.cpu_test


class _HMAConnector(SupportsHMA):
    def __init__(
        self, num_external_tokens: int, delay_free_blocks: bool = False
    ):
        self.num_external_tokens = num_external_tokens
        self.delay_free_blocks = delay_free_blocks
        self.finished_block_ids: tuple[list[int], ...] | None = None

    def on_new_request(self, request):
        return None

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int, bool]:
        return self.num_external_tokens, True

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        return None

    def build_connector_meta(self, scheduler_output):
        return None

    def update_connector_output(self, connector_output):
        return None

    def request_finished_all_groups(self, request, block_ids):
        self.finished_block_ids = block_ids
        return self.delay_free_blocks, None

    def get_kv_connector_stats(self):
        return None

    def take_events(self):
        return ()


class _PrePostFailingNixl:
    def __init__(self):
        self.make_prepped_xfer_calls = 0
        self.transfer_calls = 0

    def make_prepped_xfer(self, *args, **kwargs):
        self.make_prepped_xfer_calls += 1
        raise RuntimeError("injected failure before handle creation")

    def transfer(self, handle):
        self.transfer_calls += 1
        raise AssertionError("a pre-post failure must not call transfer")

    def get_new_notifs(self):
        return {}

    def send_notif(self, agent, notif_msg):
        return None


def _make_13_group_config(block_size: int = 16) -> KVCacheConfig:
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


def _failure(invalid_block_ids=(), reason: str = "transfer"):
    from vllm.v1 import outputs

    reason_type = getattr(outputs, "KVTransferFailureReason", None)
    failure_type = getattr(outputs, "KVTransferFailure", None)
    if reason_type is None or failure_type is None:
        return reason
    return failure_type(
        reason=reason_type(reason),
        invalid_block_ids=frozenset(invalid_block_ids),
    )


def _attach_failed_recving(model_runner_output, failed_recving):
    assert model_runner_output.kv_connector_output is not None
    model_runner_output.kv_connector_output.failed_recving = failed_recving


def _make_worker(
    local_block_ids: tuple[list[int], ...],
    kv_cache_config: KVCacheConfig,
    request_id: str = "hma-failed-receive",
) -> tuple[NixlPullConnectorWorker, _PrePostFailingNixl]:
    wrapper = _PrePostFailingNixl()
    worker = object.__new__(NixlPullConnectorWorker)
    worker.engine_id = "local"
    worker.world_size = 1
    worker.tp_rank = 0
    worker.block_size = 16
    worker.kv_cache_config = kv_cache_config
    worker.transfer_topo = Mock()
    remote_info = SimpleNamespace(
        remote_block_size=16,
        remote_physical_blocks_per_logical=1,
    )
    worker.transfer_topo.get_engine_info.return_value = remote_info
    worker.transfer_topo.block_size_ratio.return_value = 1
    worker.num_regions = 1
    worker._has_mamba = False
    worker._is_hma_required = True
    worker._physical_blocks_per_logical_kv_block = 1
    worker._skip_pull_groups = set()
    worker._sp_group_flags = Mock(return_value=[False] * 13)
    worker.dst_num_blocks = {"local": 1000, "remote": 1000}
    worker._remote_agents = {"remote": {0: "remote-agent"}}
    worker._recving_transfers = {}
    worker._recving_metadata = {
        request_id: ReqMeta(
            local_block_ids=local_block_ids,
            local_physical_block_ids=tuple(
                [block_id + 10_000 for block_id in group]
                for group in local_block_ids
            ),
            tp_size=1,
        )
    }
    worker._invalid_block_ids = queue.Queue()
    worker._failed_recv_reqs = queue.Queue()
    worker._failed_recv_outcomes = queue.Queue()
    worker._failed_recv_pending = {}
    worker._completed_failed_recv_outcomes = queue.Queue()
    worker._coalesce_drop_plan = Mock()
    worker.nixl_wrapper = wrapper
    worker.xfer_stats = Mock()
    worker._log_failure = Mock()
    worker._get_new_notifs = Mock(return_value=set())
    worker._released_rids = {}
    worker._rid_completion_counts = {}
    worker.use_host_buffer = False
    worker.use_mla = False
    worker.enable_heterogeneous_attn_post_process = False
    worker._reqs_to_send = {}
    worker.consumer_notification_counts_by_req = {}
    worker._reqs_to_process = set()
    worker._grace_frees = {}
    worker._coalesce_pending = []
    worker._audit_tick = Mock()
    return worker, wrapper


def test_push_write_failure_is_not_reported_as_failed_receive():
    worker = object.__new__(NixlPushConnectorWorker)
    worker.shutdown = Mock()
    worker._push_writer_wake = Mock()
    worker._sending_transfers_lock = threading.Lock()
    worker._sending_transfers = {"send-request": [7]}
    worker._reqs_to_send = {}
    worker._reqs_to_process = set()
    worker.consumer_notification_counts_by_req = {}
    worker._evict_finished_inbox = queue.Queue()
    worker._failed_recv_reqs = queue.Queue()
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
    worker.nixl_wrapper.release_xfer_handle.assert_called_once_with(7)
    worker.xfer_stats.record_failed_transfer.assert_called_once_with()
    assert worker._failed_recv_reqs.empty()
    assert worker._failed_recv_outcomes.empty()
    assert worker._failed_recv_pending == {}
    assert worker._completed_failed_recv_outcomes.empty()
    assert worker._invalid_block_ids.empty()


def _run_pre_post_failure(
    worker: NixlPullConnectorWorker,
    local_block_ids: tuple[list[int], ...],
    request_id: str = "hma-failed-receive",
) -> tuple[set[str], set[int], dict[str, object]]:
    remote_block_ids = tuple(
        list(range(index * 10, index * 10 + len(group)))
        for index, group in enumerate(local_block_ids)
    )
    worker._read_blocks(
        read_spec=ReadSpec(
            remote_rank=0,
            local_block_ids=local_block_ids,
            remote_block_ids=remote_block_ids,
        ),
        dst_engine_id="remote",
        request_id=request_id,
        remote_request_id=f"prefill-{request_id}",
        local_xfer_side_handle=1,
        remote_xfer_side_handle=2,
    )
    _, finished_recving = worker.get_finished()
    invalid_block_ids = worker.get_block_ids_with_load_errors()
    get_failures = getattr(worker, "get_failed_recving", None)
    failures = get_failures() if get_failures is not None else {}
    return finished_recving, invalid_block_ids, failures


def test_hma_pre_post_failure_is_not_promoted_or_cached():
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

    worker, wrapper = _make_worker(
        local_block_ids, kv_cache_config, request.request_id
    )
    finished_recving, invalid_block_ids, failed_recving = _run_pre_post_failure(
        worker, local_block_ids, request.request_id
    )
    assert wrapper.make_prepped_xfer_calls == 1
    assert wrapper.transfer_calls == 0

    model_runner_output = create_model_runner_output(
        reqs=[],
        finished_recving=finished_recving,
        invalid_block_ids=invalid_block_ids,
    )
    _attach_failed_recving(model_runner_output, failed_recving)
    aggregator = KVOutputAggregator(expected_finished_count=1)
    model_runner_output = aggregator.aggregate([model_runner_output])
    assert model_runner_output is not None

    cache_calls = []
    original_cache_blocks = scheduler.kv_cache_manager.cache_blocks

    def cache_blocks_spy(req, num_tokens):
        cache_calls.append((req.request_id, num_tokens))
        return original_cache_blocks(req, num_tokens)

    with patch.object(
        scheduler.kv_cache_manager,
        "cache_blocks",
        side_effect=cache_blocks_spy,
    ):
        outputs = scheduler.update_from_output(
            scheduler_output, model_runner_output
        )
        next_scheduler_output = scheduler.schedule()

    assert cache_calls == [], (
        "parent promoted and cached the failed HMA receive: "
        f"cache_calls={cache_calls}, status={request.status}"
    )
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

    stale_invalid_block_ids = {
        block_id for group_block_ids in local_block_ids for block_id in group_block_ids
    }
    replacement = create_request(num_tokens=num_prompt_tokens)
    scheduler.add_request(replacement)
    replacement_scheduler_output = scheduler.schedule()
    replacement_block_ids = scheduler.kv_cache_manager.get_block_ids(
        replacement.request_id
    )
    assert any(
        block_id in stale_invalid_block_ids
        for group_block_ids in replacement_block_ids
        for block_id in group_block_ids
    )

    duplicate = create_model_runner_output(
        reqs=[],
        finished_recving=set(),
        invalid_block_ids=stale_invalid_block_ids,
    )
    _attach_failed_recving(
        duplicate,
        {request.request_id: _failure(stale_invalid_block_ids)},
    )
    duplicate_outputs = scheduler.update_from_output(
        replacement_scheduler_output, duplicate
    )
    assert all(
        len(engine_outputs.outputs) == 0
        for engine_outputs in duplicate_outputs.values()
    )
    assert replacement.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert replacement.num_computed_tokens == num_external_tokens
    assert replacement.request_id not in scheduler.failed_recving_kv_req_ids


def test_hma_worker_failure_marks_every_group_invalid():
    kv_cache_config = _make_13_group_config()
    local_block_ids = tuple(
        [index * 10 + offset for offset in range(3)] for index in range(13)
    )
    worker, wrapper = _make_worker(local_block_ids, kv_cache_config)
    meta = worker._recving_metadata["hma-failed-receive"]
    assert meta.local_physical_block_ids != meta.local_block_ids

    _, invalid_block_ids, failed_recving = _run_pre_post_failure(
        worker, local_block_ids
    )

    assert wrapper.transfer_calls == 0
    assert invalid_block_ids == {
        block_id for group in local_block_ids for block_id in group
    }


def test_release_fence_failure_retains_logical_block_ids():
    kv_cache_config = _make_13_group_config()
    local_block_ids = tuple(
        [index * 10 + offset for offset in range(3)] for index in range(13)
    )
    worker, _ = _make_worker(local_block_ids, kv_cache_config)
    request_id = "hma-failed-receive"
    remote_request_id = "released-prefill"
    worker._recving_metadata[request_id].remote = RemoteMeta(
        block_ids=local_block_ids,
        host="remote-host",
        port=1234,
        engine_id="remote",
        request_id=remote_request_id,
    )
    worker._recving_transfers = {request_id: []}
    worker._released_rids = {remote_request_id: 1.0}

    _, finished_recving = worker.get_finished()
    failed_recving = worker.get_failed_recving()

    assert finished_recving == {request_id}
    from vllm.v1.outputs import KVTransferFailureReason

    assert failed_recving[request_id].reason is KVTransferFailureReason.INTEGRITY
    assert failed_recving == {
        request_id: _failure(
            (
                block_id
                for group_block_ids in local_block_ids
                for block_id in group_block_ids
            ),
            reason="integrity",
        ),
    }
    worker._coalesce_drop_plan.assert_called_once_with(request_id)


def test_failed_receive_waits_for_all_workers_and_wins_over_success():
    request_id = "request"
    aggregator = KVOutputAggregator(expected_finished_count=2)

    failed_rank = create_model_runner_output(
        reqs=[],
        finished_recving={request_id},
        invalid_block_ids={17, 18},
    )
    _attach_failed_recving(
        failed_rank, {request_id: _failure({17, 18})}
    )
    pending_rank = create_model_runner_output(reqs=[], finished_recving=set())
    first = aggregator.aggregate([failed_rank, pending_rank])
    assert first is not None and first.kv_connector_output is not None
    assert first.kv_connector_output.finished_recving is None
    assert getattr(first.kv_connector_output, "failed_recving", {}) == {}
    assert first.kv_connector_output.invalid_block_ids == set()

    duplicate_failed_rank = create_model_runner_output(
        reqs=[],
        finished_recving={request_id},
        invalid_block_ids={17, 18},
    )
    _attach_failed_recving(
        duplicate_failed_rank, {request_id: _failure({17, 18})}
    )
    duplicate_pending_rank = create_model_runner_output(
        reqs=[], finished_recving=set()
    )
    duplicate = aggregator.aggregate(
        [duplicate_failed_rank, duplicate_pending_rank]
    )
    assert duplicate is not None and duplicate.kv_connector_output is not None
    assert duplicate.kv_connector_output.finished_recving is None
    assert duplicate.kv_connector_output.failed_recving == {}
    assert duplicate.kv_connector_output.invalid_block_ids == set()

    empty_rank = create_model_runner_output(reqs=[], finished_recving=set())
    successful_rank = create_model_runner_output(
        reqs=[], finished_recving={request_id}
    )
    second = aggregator.aggregate([empty_rank, successful_rank])
    assert second is not None and second.kv_connector_output is not None
    assert second.kv_connector_output.finished_recving is None, (
        "a rank failure was incorrectly collapsed into healthy completion"
    )
    assert getattr(second.kv_connector_output, "failed_recving", {}) == {
        request_id: _failure({17, 18}),
    }
    assert second.kv_connector_output.invalid_block_ids == set()


def test_failed_receive_releases_previously_aborted_request():
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

    scheduler.finish_requests(
        request.request_id, RequestStatus.FINISHED_ABORTED
    )
    assert request.request_id in scheduler.requests
    assert request.request_id in scheduler._receive_delayed_free_req_ids

    failure_output = create_model_runner_output(
        reqs=[],
        finished_recving=set(),
        invalid_block_ids=invalid_block_ids,
    )
    _attach_failed_recving(
        failure_output,
        {request.request_id: _failure(invalid_block_ids)},
    )
    outputs = scheduler.update_from_output(scheduler_output, failure_output)

    assert request.status == RequestStatus.FINISHED_ABORTED
    assert request.request_id not in scheduler.requests
    assert request.request_id not in scheduler.finished_recving_kv_req_ids
    assert request.request_id not in scheduler.failed_recving_kv_req_ids
    assert request.request_id not in scheduler._receive_delayed_free_req_ids
    assert (
        scheduler.kv_cache_manager.block_pool.free_block_queue.num_free_blocks
        == baseline_free_blocks
    )
    assert all(
        len(engine_outputs.outputs) == 0 for engine_outputs in outputs.values()
    )


def test_stale_healthy_receive_does_not_free_send_retained_request():
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
        num_external_tokens=0, delay_free_blocks=True
    )
    scheduler.finish_requests(
        request.request_id, RequestStatus.FINISHED_ABORTED
    )
    retained_free_blocks = (
        scheduler.kv_cache_manager.block_pool.free_block_queue.num_free_blocks
    )
    assert request.request_id in scheduler.requests
    assert request.request_id not in scheduler._receive_delayed_free_req_ids

    stale_healthy_output = create_model_runner_output(
        reqs=[], finished_recving={request.request_id}
    )
    assert stale_healthy_output.kv_connector_output is not None
    scheduler._update_from_kv_xfer_finished(
        stale_healthy_output.kv_connector_output
    )

    assert request.request_id in scheduler.requests
    assert (
        scheduler.kv_cache_manager.block_pool.free_block_queue.num_free_blocks
        == retained_free_blocks
    )

    scheduler._free_blocks(request)


def test_multi_group_invalid_blocks_do_not_tuple_unpack_crash():
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
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    block_ids = scheduler.kv_cache_manager.get_block_ids(request.request_id)
    assert len(block_ids) == 13
    invalid_block_id = block_ids[12][0]
    assert all(invalid_block_id not in group for group in block_ids[:12])

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
        block_id for group in block_ids for block_id in group
    }
