# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import queue
import threading
from collections import deque
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, call, patch

import msgspec
import pytest
import torch

from vllm.distributed.kv_transfer.coalesced_layout import (
    CoalescedTransferPlan,
    GroupTransferRoster,
    RegionOwnership,
    build_coalesced_transfer_plan,
)
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
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.packed_write_config import (
    PackedWriteConfig,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_worker import (
    NixlPullConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.push_worker import (
    NixlPushConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import ReadSpec
from vllm.distributed.kv_transfer.nixl_contracts import NixlRegionDescriptor
from vllm.distributed.kv_transfer.staging_ownership import (
    HandleState,
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


def _layout(source_rank_count: int) -> CoalescedTransferPlan:
    """Build one canonical typed layout for staging lifecycle fixtures.

    :param source_rank_count: Number of native producer-rank handles.
    :returns: Immutable rank-major transfer plan.
    """
    return build_coalesced_transfer_plan(
        source_tp_size=source_rank_count,
        source_ranks=tuple(range(source_rank_count)),
        rank_slots=tuple(range(source_rank_count)),
        groups=(
            GroupTransferRoster(
                group_index=0,
                source_position_start=0,
                destination_plane_count=2,
                local_block_ids=(0,),
                remote_block_ids=(0,),
            ),
        ),
        regions=(
            RegionOwnership(
                region_index=0,
                group_indices=(0,),
                source_row_count=4,
                destination_row_count=4,
                row_bytes=8,
            ),
        ),
    )


def _native_telemetry(total_bytes: int) -> SimpleNamespace:
    """Build one complete terminal native telemetry record.

    :param total_bytes: Bytes transferred by the native handle.
    :returns: Production-compatible telemetry fixture.
    """
    return SimpleNamespace(
        xferDuration=10,
        postDuration=2,
        totalBytes=total_bytes,
        descCount=1,
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
    worker._transfer_phase_active = False
    worker._deferred_phase_sending = set()
    worker._deferred_phase_recving = set()
    worker._transfer_phase_records = []
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
    worker._packed_write_config = PackedWriteConfig(
        enabled=False,
        chunk_bytes_per_rank=64 * 1024 * 1024,
        min_descriptors_per_rank=1,
        producer_slot_count=1,
        consumer_slot_count=1,
        alignment_bytes=256,
        warn_after_s=1.0,
        fail_after_s=2.0,
    )
    worker._packed_consumer_requests = {}
    worker._packed_producer_pending = deque()
    worker._packed_producer_operations = {}
    worker._packed_done_recving = set()
    worker._pending_coalesced_scatters = {}
    worker._failed_recv_outcomes = queue.Queue()
    worker._failed_recv_pending = {}
    worker._completed_failed_recv_outcomes = queue.Queue()
    worker._invalid_block_ids = queue.Queue()
    worker._released_remote_offers = {}
    worker._remote_offer_completion_counts = {}
    worker._remote_source_retired_through = {}
    worker._remote_source_consumption_proven = {}
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
    worker._coalesced_localization_plans = {}
    worker._localization_pre_read_plans = {}
    worker._handshake_futures = {}
    worker._handshake_active_engine_id = None
    worker._handshake_mutation_engine_id = None
    worker._handshake_fail_stop_reason = None
    worker._handshake_lock = threading.RLock()
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


@pytest.mark.parametrize("fence_mode", ["exact", "retirement-floor"])
def test_release_fences_report_integrity_failure_for_every_group(
    fence_mode: str,
) -> None:
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
        source_offer_generation=7,
    )
    worker._recving_transfers = {request_id: []}
    remote = worker._recving_metadata[request_id].remote
    assert remote is not None
    if fence_mode == "exact":
        worker._released_remote_offers = {remote.offer_key: 1.0}
    else:
        worker._remote_source_retired_through = {"remote-engine": 7}

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


@pytest.mark.parametrize("fence_mode", ["exact", "retirement-floor"])
def test_coalesced_done_race_with_source_retirement_never_scatters(
    fence_mode: str,
) -> None:
    """A source retired after native DONE cannot reach destination scatter."""
    request_id = "coalesced-retirement-race"
    worker = _make_worker(([9],), request_id)
    meta = worker._recving_metadata[request_id]
    meta.remote = RemoteMeta(
        block_ids=([19],),
        host="producer-host",
        port=1234,
        engine_id="producer-engine",
        request_id="producer-request",
        source_offer_generation=7,
    )
    remote = meta.remote

    allocator = StagingRangeAllocator(1024)
    plan = allocator.create_plan(
        owner_id="decode:0:0",
        request_id=request_id,
        layout=_layout(1),
        layout_duration_seconds=0.0,
        remote_engine_id=remote.engine_id,
    )
    assert plan is not None
    plan.begin_prepare(0)
    plan.attach_handle(0, 17)
    plan.begin_post(0)
    plan.record_post_result(0, "PROC")
    plan.seal_posting()
    worker._staging_allocator = allocator
    worker._coalesce_plans = {request_id: plan}
    worker._coalesced_scatter = Mock()
    worker.nixl_wrapper.check_xfer_state.return_value = "DONE"
    worker.nixl_wrapper.get_xfer_telemetry.return_value = _native_telemetry(
        plan.layout.staging_size_bytes
    )

    def retire_source_offer(_handle: int) -> None:
        if fence_mode == "exact":
            worker._mark_remote_offer_released(remote.offer_key)
            return
        worker._advance_remote_source_retirement_floor(
            remote.engine_id,
            remote.source_offer_generation,
        )

    worker.nixl_wrapper.release_xfer_handle.side_effect = retire_source_offer
    ready_when_failed: list[bool] = []
    original_fail = plan.fail

    def fail_after_ready(reason: str) -> None:
        ready_when_failed.append(plan.ready_to_scatter)
        original_fail(reason)

    with patch.object(plan, "fail", side_effect=fail_after_ready):
        assert worker._poll_coalesced_plans() == set()

    assert ready_when_failed == [True]
    worker._coalesced_scatter.assert_not_called()
    worker.nixl_wrapper.check_xfer_state.assert_called_once_with(17)
    worker.nixl_wrapper.release_xfer_handle.assert_called_once_with(17)
    assert plan.released
    assert request_id not in worker._coalesce_plans
    assert allocator.active == {}
    assert allocator.free_bytes == allocator.capacity
    assert worker._failed_recv_pending == {
        request_id: _failure({9}, KVTransferFailureReason.INTEGRITY)
    }

    _, finished_recving = worker.get_finished()

    assert finished_recving == {request_id}
    assert worker.get_failed_recving() == {
        request_id: _failure({9}, KVTransferFailureReason.INTEGRITY)
    }
    worker._coalesced_scatter.assert_not_called()


def test_post_done_scatter_validation_failure_still_consumes_source() -> None:
    """A decoder-local failure cannot discard its completed source read proof."""
    request_id = "coalesced-validation-failure"
    worker = _make_worker(([9],), request_id)
    meta = worker._recving_metadata[request_id]
    meta.remote = RemoteMeta(
        block_ids=([19],),
        host="producer-host",
        port=1234,
        engine_id="producer-engine",
        request_id="producer-request",
        expected_consumers=1,
        source_offer_generation=7,
    )
    remote = meta.remote

    allocator = StagingRangeAllocator(1024)
    plan = allocator.create_plan(
        owner_id="decode:0:0",
        request_id=request_id,
        layout=_layout(1),
        layout_duration_seconds=0.0,
        remote_engine_id=remote.engine_id,
    )
    assert plan is not None
    plan.begin_prepare(0)
    plan.attach_handle(0, 17)
    plan.begin_post(0)
    plan.record_post_result(0, "PROC")
    plan.seal_posting()
    worker._staging_allocator = allocator
    worker._coalesce_plans = {request_id: plan}
    worker._staging_buf = Mock()
    worker._region_rows = (Mock(),)
    worker._coalesced_scatter_stream = Mock()
    worker.nixl_wrapper.check_xfer_state.return_value = "DONE"
    worker.nixl_wrapper.get_xfer_telemetry.return_value = _native_telemetry(
        plan.layout.staging_size_bytes
    )

    validator_path = (
        "vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker."
        "validate_coalesced_scatter"
    )
    launcher_path = (
        "vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker."
        "launch_coalesced_scatter"
    )
    with (
        patch(
            validator_path, side_effect=ValueError("invalid destination geometry")
        ) as validate,
        patch(launcher_path) as launch,
    ):
        assert worker._poll_coalesced_plans() == set()

    validate.assert_called_once()
    launch.assert_not_called()
    worker.nixl_wrapper.release_xfer_handle.assert_called_once_with(17)
    assert plan.source_consumption_proven
    assert plan.released
    assert request_id not in worker._coalesce_plans
    assert allocator.free_bytes == allocator.capacity
    assert worker._remote_source_consumption_proven == {request_id: remote.offer_key}
    assert remote.offer_key not in worker._released_remote_offers

    _, finished_recving = worker.get_finished()

    assert finished_recving == {request_id}
    assert worker.get_failed_recving() == {request_id: _failure({9})}
    assert worker._remote_source_consumption_proven == {}
    assert worker._remote_offer_completion_counts == {}
    assert remote.offer_key in worker._released_remote_offers


def test_success_reobserves_source_proof_without_double_counting() -> None:
    """Natural success consumes an earlier per-child proof exactly once."""
    request_id = "coalesced-success"
    worker = _make_worker(([9],), request_id)
    meta = worker._recving_metadata[request_id]
    meta.remote = RemoteMeta(
        block_ids=([19],),
        host="producer-host",
        port=1234,
        engine_id="producer-engine",
        request_id="producer-request",
        expected_consumers=2,
        source_offer_generation=7,
    )
    remote = meta.remote
    worker._recving_transfers = {request_id: []}
    worker.enable_permute_local_kv = False
    worker.transfer_topo.get_engine_info.return_value = SimpleNamespace(
        remote_block_size=16
    )
    worker.transfer_topo.block_size_ratio.return_value = 1
    worker._record_remote_source_consumption_proven(request_id, meta)

    with patch.object(
        worker,
        "_record_remote_source_consumption_proven",
        wraps=worker._record_remote_source_consumption_proven,
    ) as record_proof:
        _, finished_recving = worker.get_finished()

    assert finished_recving == {request_id}
    record_proof.assert_called_once_with(request_id, meta)
    assert worker.get_failed_recving() == {}
    assert worker._remote_source_consumption_proven == {}
    assert worker._remote_offer_completion_counts == {remote.offer_key: 1}
    assert remote.offer_key not in worker._released_remote_offers


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
        layout=_layout(1),
        layout_duration_seconds=0.0,
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
        layout=_layout(1),
        layout_duration_seconds=0.0,
        remote_engine_id="prefill-engine",
    )
    assert plan is not None
    plan.begin_prepare(0)
    plan.attach_handle(0, 17)
    plan.begin_post(0)
    plan.record_post_result(0, "DONE")
    plan.record_native_telemetry(
        0,
        _native_telemetry(plan.layout.staging_size_bytes),
    )
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
        layout=_layout(2),
        layout_duration_seconds=0.0,
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
        layout=_layout(1),
        layout_duration_seconds=0.0,
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
        source_offer_generation=7,
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
        layout=_layout(3),
        layout_duration_seconds=0.0,
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
    assert worker._remote_source_consumption_proven == {
        request_id: meta.remote.offer_key
    }
    proof = msgspec.msgpack.decode(
        notification_id[len(PULL_READ_COMPLETE_PREFIX) :],
        type=PullReadComplete,
    )
    assert proof == PullReadComplete(
        producer_request_id=producer_request_id,
        offer_generation=7,
        consumer_request_id=request_id,
        consumer_index=0,
        consumer_rank=0,
        consumer_tp_size=1,
        expected_consumers=1,
    )

    _, finished_recving = worker.get_finished()

    assert finished_recving == {request_id}
    assert worker.get_failed_recving() == {request_id: _failure({9})}
    assert worker._remote_source_consumption_proven == {}
    assert worker._remote_offer_completion_counts == {}
    assert meta.remote.offer_key in worker._released_remote_offers


def test_coalesced_prepare_failure_becomes_request_terminal() -> None:
    """A pre-write native failure is reclaimed and reported without fail-stop."""
    request_id = "coalesced-prepare-failure"
    worker = _make_worker(([9],), request_id)
    allocator = StagingRangeAllocator(1024)
    plan = allocator.create_plan(
        owner_id="decode:0:0",
        request_id=request_id,
        layout=_layout(1),
        layout_duration_seconds=0.0,
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
        prepared = worker._prepare_coalesced_handle(
            plan,
            0,
            Mock(),
            Mock(),
            "remote-agent",
            b"notification",
        )
    _, finished_recving = worker.get_finished()

    assert prepared is False
    fail.assert_called_once()
    assert finished_recving == {request_id}
    assert worker.get_failed_recving() == {request_id: _failure({9})}
    assert plan.released
    assert allocator.free_bytes == allocator.capacity
    worker.nixl_wrapper.transfer.assert_not_called()


def test_descriptor_failure_precedes_every_rank_post_and_reclaims_plan() -> None:
    """A later-rank descriptor failure cannot race an earlier native writer."""
    request_id = "coalesced-descriptor-failure"
    producer_engine_id = "prefill-engine"
    worker = _make_worker(
        ([2],),
        request_id,
        worker_type=NixlPullConnectorWorker,
    )
    assert isinstance(worker, NixlPullConnectorWorker)
    meta = worker._recving_metadata[request_id]
    meta.tp_size = 4
    meta.remote = RemoteMeta(
        block_ids=([1],),
        host="producer-host",
        port=1234,
        engine_id=producer_engine_id,
        request_id="producer-request",
        expected_consumers=1,
        consumer_tp_size=1,
    )
    region = NixlRegionDescriptor(
        semantic_name="layer",
        group_indices=(0,),
        group_semantic_names=((0, "layer"),),
        base_address=100_000,
        registered_bytes=64,
        row_bytes=8,
        shape=(8, 8),
        strides=(8, 1),
        dtype="torch.uint8",
        element_size_bytes=1,
        layout="HND",
    )
    worker.world_size = 1
    worker.transfer_topo.get_engine_info.return_value = SimpleNamespace(
        remote_tp_size=4,
        remote_physical_blocks_per_logical=1,
    )
    worker.tp_mappings = {
        producer_engine_id: SimpleNamespace(
            rank_to_attention_slot={rank: rank for rank in range(4)}
        )
    }
    worker._remote_layout = {
        producer_engine_id: {rank: ([8], 8, 0) for rank in range(4)}
    }
    worker._remote_regions = {producer_engine_id: {0: (region,)}}
    worker._region_descriptors = (region,)
    worker.kv_caches_base_addr = {
        producer_engine_id: {rank: [200_000 + rank * 1_000] for rank in range(4)}
    }
    worker._remote_agents = {
        producer_engine_id: {rank: f"producer-agent-{rank}" for rank in range(4)}
    }
    worker._localization_config = Mock()
    worker._localization_config.enabled_for.return_value = False
    worker._sp_group_flags = Mock(return_value=[False])
    worker._apply_prefix_caching = Mock(return_value=([[2]], [[1]]))
    worker._staging_buf = Mock()
    worker._staging_buf.data_ptr.return_value = 300_000
    worker._staging_allocator = StagingRangeAllocator(1024)
    worker._coalesce_owner_sequence = 0
    worker._coalesce_warn_after_s = 30.0
    worker._coalesce_fail_after_s = 300.0
    worker.coalesce_staging_mb = 1
    worker.nixl_memory_type = "VRAM"
    worker.device_id = 0
    worker._notify_failed_coalesced_producer_ranks = Mock()
    worker._handle_failed_transfer = Mock()
    prepared_handle = object()
    worker.nixl_wrapper.get_xfer_descs.side_effect = (
        ["rank-0-local"],
        ["rank-0-remote"],
        RuntimeError("rank-1 descriptor conversion failed"),
    )
    worker.nixl_wrapper.initialize_xfer.return_value = prepared_handle
    read_specs = [
        ReadSpec(
            remote_rank=rank,
            local_block_ids=[[2]],
            remote_block_ids=[[1]],
        )
        for rank in range(4)
    ]

    result = worker._coalesced_read_request(
        request_id,
        meta,
        read_specs,
        b"notification",
    )

    assert result == "posted"
    worker.nixl_wrapper.transfer.assert_not_called()
    worker._notify_failed_coalesced_producer_ranks.assert_called_once()
    ownership = worker._notify_failed_coalesced_producer_ranks.call_args.args[1]
    assert ownership.released
    assert ownership.slots[0].state is HandleState.SEALED_UNPOSTED
    assert ownership.slots[1].state is HandleState.PREPARE_FAILED
    assert ownership.slots[2].state is HandleState.NEVER_POSTED
    assert ownership.slots[3].state is HandleState.NEVER_POSTED
    worker.nixl_wrapper.release_xfer_handle.assert_called_once_with(prepared_handle)
    assert worker._staging_allocator.free_bytes == worker._staging_allocator.capacity


def test_failed_receive_keeps_tombstoned_coalesced_staging_pinned() -> None:
    """An uncertain native failure remains fail-stop and cannot be reused."""
    request_id = "tombstoned-failed-plan"
    worker = _make_worker(([9],), request_id)
    allocator = StagingRangeAllocator(1024)
    plan = allocator.create_plan(
        owner_id="decode:0:0",
        request_id=request_id,
        layout=_layout(1),
        layout_duration_seconds=0.0,
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


def test_healthy_receive_releases_previously_aborted_request() -> None:
    """A drained receive releases destination blocks retained by an abort."""
    kv_cache_config = _make_13_group_config()
    vllm_config = create_vllm_config(max_num_batched_tokens=64)
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
    allocated_free_blocks = (
        scheduler.kv_cache_manager.block_pool.free_block_queue.num_free_blocks
    )
    assert allocated_free_blocks < baseline_free_blocks

    scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)

    assert request.request_id in scheduler.requests
    assert request.request_id in scheduler._receive_delayed_free_req_ids
    assert (
        scheduler.kv_cache_manager.block_pool.free_block_queue.num_free_blocks
        == allocated_free_blocks
    )

    outputs = scheduler.update_from_output(
        scheduler_output,
        _model_output(
            KVConnectorOutput(finished_recving={request.request_id}),
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
