"""Source-bound tests for production coalesced-staging ownership."""

import ast
import heapq
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Never
from unittest.mock import MagicMock

import pytest

from vllm.distributed.kv_transfer.staging_ownership import (
    CoalescedStagingPlan,
    HandleState,
    StagingHandleSnapshot,
    StagingOwnershipSnapshot,
    StagingRangeAllocator,
    StagingSafetyError,
    TransferQuiescenceError,
)

BASE_WORKER_PATH = (
    Path(__file__).parents[2]
    / "vllm"
    / "distributed"
    / "kv_transfer"
    / "kv_connector"
    / "v1"
    / "nixl"
    / "base_worker.py"
)
PULL_WORKER_PATH = BASE_WORKER_PATH.with_name("pull_worker.py")


def _production_worker_methods(
    *method_names: str,
    monotonic_time: float = 0.0,
    torch_module: object | None = None,
) -> type:
    """Compile exact production methods without importing the CUDA worker graph.

    :param method_names: Methods to bind to the test class.
    :param monotonic_time: Deterministic clock observation.
    :param torch_module: Fake torch surface for scatter tests.
    :returns: Class containing the exact production method bodies.
    """
    syntax = ast.parse(BASE_WORKER_PATH.read_text(), filename=str(BASE_WORKER_PATH))
    worker_node = next(
        node
        for node in syntax.body
        if isinstance(node, ast.ClassDef) and node.name == "NixlBaseConnectorWorker"
    )
    selected = [
        node
        for node in worker_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in method_names
    ]
    if {node.name for node in selected} != set(method_names):
        raise AssertionError("production staging method is missing")
    test_class = ast.ClassDef(
        name="ProductionWorkerMethods",
        bases=[],
        keywords=[],
        body=selected,
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[test_class], type_ignores=[]))
    namespace: dict[str, object] = {
        "Any": Any,
        "BaseException": BaseException,
        "CoalescedStagingPlan": CoalescedStagingPlan,
        "EngineId": str,
        "HandleState": HandleState,
        "heapq": heapq,
        "IntegrityStage": SimpleNamespace(
            STAGING_RAW="staging_raw",
            STAGING_FENCED_CONTROL="staging_fenced_control",
            DESTINATION="destination",
        ),
        "Never": Never,
        "ReqId": str,
        "StagingSafetyError": StagingSafetyError,
        "TransferQuiescenceError": TransferQuiescenceError,
        "logger": MagicMock(),
        "time": SimpleNamespace(monotonic=lambda: monotonic_time),
        "torch": MagicMock() if torch_module is None else torch_module,
        "traceback": traceback,
    }
    exec(compile(module, str(BASE_WORKER_PATH), "exec"), namespace)
    return namespace["ProductionWorkerMethods"]  # type: ignore[return-value]


def _production_pull_worker_methods(*method_names: str) -> type:
    """Compile exact pull-worker methods without importing the worker graph.

    :param method_names: Methods to bind to the test class.
    :returns: Class containing the exact production method bodies.
    """
    syntax = ast.parse(PULL_WORKER_PATH.read_text(), filename=str(PULL_WORKER_PATH))
    worker_node = next(
        node
        for node in syntax.body
        if isinstance(node, ast.ClassDef) and node.name == "NixlPullConnectorWorker"
    )
    selected = [
        node
        for node in worker_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in method_names
    ]
    if {node.name for node in selected} != set(method_names):
        raise AssertionError("production pull-worker method is missing")
    test_class = ast.ClassDef(
        name="ProductionPullWorkerMethods",
        bases=[],
        keywords=[],
        body=selected,
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[test_class], type_ignores=[]))
    namespace: dict[str, object] = {
        "ReadSpec": object,
        "ReqId": str,
        "TransferQuiescenceError": TransferQuiescenceError,
        "logger": MagicMock(),
        "np": MagicMock(),
        "traceback": traceback,
    }
    exec(compile(module, str(PULL_WORKER_PATH), "exec"), namespace)
    return namespace["ProductionPullWorkerMethods"]  # type: ignore[return-value]


def _plan(
    *,
    statuses: tuple[str, ...],
    created_at: float = 0.0,
    remote_engine_id: str = "test-producer",
    scatter: dict[str, object] | None = None,
) -> tuple[StagingRangeAllocator, CoalescedStagingPlan]:
    """Create one sealed production owner with synthetic native handles.

    :param statuses: Immediate NIXL status per source rank.
    :param created_at: Deterministic creation time.
    :param remote_engine_id: Synthetic remote engine identity.
    :param scatter: Optional scatter geometry.
    :returns: Allocator and exact active plan.
    """
    allocator = StagingRangeAllocator(1024)
    plan = allocator.create_plan(
        owner_id="test-owner",
        request_id="test-request",
        size=512,
        source_ranks=tuple(range(len(statuses))),
        remote_engine_id=remote_engine_id,
        scatter={} if scatter is None else scatter,
        created_at=created_at,
        warn_after_s=1.0,
        fail_after_s=2.0,
    )
    assert plan is not None
    for rank, status in enumerate(statuses):
        plan.begin_prepare(rank)
        plan.attach_handle(rank, object())
        plan.begin_post(rank)
        plan.record_post_result(rank, status)
    plan.seal_posting()
    return allocator, plan


def test_production_err_with_proc_fails_before_native_or_range_release() -> None:
    worker_type = _production_worker_methods(
        "_fail_coalesced_plan",
        "_poll_coalesced_plans",
    )
    allocator, plan = _plan(statuses=("PROC", "ERR"))
    worker = worker_type()
    worker._coalesce_plans = {plan.request_id: plan}
    worker.nixl_wrapper = MagicMock()
    worker.xfer_stats = MagicMock()

    with pytest.raises(StagingSafetyError, match="safety proof failed"):
        worker._poll_coalesced_plans()

    assert allocator.require_active(plan.lease.generation) is plan
    worker.nixl_wrapper.check_xfer_state.assert_not_called()
    worker.nixl_wrapper.release_xfer_handle.assert_not_called()
    assert plan.slots[0].state is HandleState.PROC
    assert plan.slots[1].native_handle is not None


def test_production_mid_post_exception_retains_handle_and_lease() -> None:
    worker_type = _production_worker_methods(
        "_fail_coalesced_plan",
        "_initialize_and_post_coalesced",
    )
    allocator = StagingRangeAllocator(1024)
    plan = allocator.create_plan(
        owner_id="post-owner",
        request_id="post-request",
        size=512,
        source_ranks=(0,),
        remote_engine_id="test-producer",
    )
    assert plan is not None
    plan.begin_prepare(0)
    native_handle = object()
    worker = worker_type()
    worker.nixl_wrapper = MagicMock()
    worker.nixl_wrapper.initialize_xfer.return_value = native_handle
    worker.nixl_wrapper.transfer.side_effect = RuntimeError("post exploded")

    with pytest.raises(StagingSafetyError, match="safety proof failed"):
        worker._initialize_and_post_coalesced(
            plan,
            0,
            object(),
            object(),
            "remote-agent",
            b"notification",
        )

    assert allocator.require_active(plan.lease.generation) is plan
    assert plan.slots[0].native_handle is native_handle
    assert plan.slots[0].state is HandleState.UNKNOWN
    assert plan.permanently_tombstoned
    worker.nixl_wrapper.release_xfer_handle.assert_not_called()


def test_production_polls_before_timeout_then_tombstones_proc() -> None:
    worker_type = _production_worker_methods(
        "_fail_coalesced_plan",
        "_poll_coalesced_plans",
        monotonic_time=3.0,
    )
    allocator, plan = _plan(statuses=("PROC",), created_at=0.0)
    worker = worker_type()
    worker._coalesce_plans = {plan.request_id: plan}
    worker.nixl_wrapper = MagicMock()
    worker.nixl_wrapper.check_xfer_state.return_value = "PROC"
    worker.xfer_stats = MagicMock()

    with pytest.raises(StagingSafetyError, match="fail-stop deadline"):
        worker._poll_coalesced_plans()

    worker.nixl_wrapper.check_xfer_state.assert_called_once()
    worker.nixl_wrapper.release_xfer_handle.assert_not_called()
    assert allocator.require_active(plan.lease.generation) is plan
    assert plan.permanently_tombstoned


def test_production_done_observed_at_deadline_is_not_failed() -> None:
    worker_type = _production_worker_methods(
        "_fail_coalesced_plan",
        "_poll_coalesced_plans",
        monotonic_time=3.0,
    )
    _, plan = _plan(statuses=("PROC",), created_at=0.0)
    worker = worker_type()
    worker._coalesce_plans = {plan.request_id: plan}
    worker.nixl_wrapper = MagicMock()
    worker.nixl_wrapper.check_xfer_state.return_value = "DONE"
    worker.nixl_wrapper.get_xfer_telemetry.return_value = object()
    worker.xfer_stats = MagicMock()

    assert worker._poll_coalesced_plans() == {plan.request_id}
    assert plan.permanently_tombstoned is False
    worker.nixl_wrapper.release_xfer_handle.assert_called_once()


def test_production_done_release_failure_retains_handle_and_generation() -> None:
    worker_type = _production_worker_methods(
        "_fail_coalesced_plan",
        "_poll_coalesced_plans",
    )
    allocator, plan = _plan(statuses=("DONE",))
    native_handle = plan.slots[0].native_handle
    worker = worker_type()
    worker._coalesce_plans = {plan.request_id: plan}
    worker.nixl_wrapper = MagicMock()
    worker.nixl_wrapper.get_xfer_telemetry.return_value = object()
    worker.nixl_wrapper.release_xfer_handle.side_effect = RuntimeError(
        "release exploded"
    )
    worker.xfer_stats = MagicMock()

    with pytest.raises(StagingSafetyError, match="safety proof failed"):
        worker._poll_coalesced_plans()

    assert allocator.require_active(plan.lease.generation) is plan
    assert plan.operation_failed
    assert plan.slots[0].state is HandleState.DONE
    assert plan.slots[0].native_handle is native_handle
    assert plan.slots[0].native_released is False


def test_failed_publication_retires_completed_plan_exactly_once() -> None:
    worker_type = _production_worker_methods(
        "_discard_quiescent_coalesced_plan",
        "_fail_coalesced_plan",
        "_poll_coalesced_plans",
        "_release_coalesced_plan",
    )
    allocator, plan = _plan(statuses=("PROC",))
    worker = worker_type()
    worker._coalesce_plans = {plan.request_id: plan}
    worker._staging_allocator = allocator
    worker.nixl_wrapper = MagicMock()
    worker.nixl_wrapper.check_xfer_state.return_value = "DONE"
    worker.nixl_wrapper.get_xfer_telemetry.return_value = object()
    worker.xfer_stats = MagicMock()

    assert worker._poll_coalesced_plans() == {plan.request_id}
    worker._discard_quiescent_coalesced_plan(plan.request_id)

    assert worker._poll_coalesced_plans() == set()
    assert plan.request_id not in worker._coalesce_plans
    assert allocator.free_bytes == allocator.capacity
    worker.nixl_wrapper.release_xfer_handle.assert_called_once()


def test_failed_receive_branch_retires_coalesced_owner_before_continue() -> None:
    syntax = ast.parse(BASE_WORKER_PATH.read_text(), filename=str(BASE_WORKER_PATH))
    worker_node = next(
        node
        for node in syntax.body
        if isinstance(node, ast.ClassDef) and node.name == "NixlBaseConnectorWorker"
    )
    get_finished = next(
        node
        for node in worker_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "_get_finished"
    )
    failed_branch = next(
        node
        for node in ast.walk(get_finished)
        if isinstance(node, ast.If)
        and any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == "_discard_quiescent_coalesced_plan"
            for child in ast.walk(node)
        )
    )

    assert any(isinstance(node, ast.Continue) for node in failed_branch.body)


def test_failed_publication_cannot_retire_live_native_handle() -> None:
    worker_type = _production_worker_methods(
        "_discard_quiescent_coalesced_plan",
        "_release_coalesced_plan",
    )
    allocator, plan = _plan(statuses=("DONE",))
    worker = worker_type()
    worker._coalesce_plans = {plan.request_id: plan}
    worker._staging_allocator = allocator

    with pytest.raises(StagingSafetyError, match="before native quiescence"):
        worker._discard_quiescent_coalesced_plan(plan.request_id)

    assert allocator.require_active(plan.lease.generation) is plan


def test_source_lease_expiry_never_authorizes_reuse() -> None:
    worker_type = _production_worker_methods(
        "_expire_source_leases", "_on_source_lease_expired"
    )
    worker = worker_type()
    worker._reqs_to_send = {"producer-request": 10.0}
    worker._source_lease_heap = [(10.0, "producer-request")]
    worker._reqs_to_process = {"producer-request"}
    worker.consumer_notification_counts_by_req = {"producer-request": 1}
    worker.xfer_stats = MagicMock()

    with pytest.raises(TransferQuiescenceError, match="process teardown"):
        worker._expire_source_leases(12.0)

    assert worker._reqs_to_send == {}
    assert worker._reqs_to_process == {"producer-request"}
    assert worker.consumer_notification_counts_by_req == {"producer-request": 1}
    worker.xfer_stats.record_kv_expired_req.assert_called_once_with()


def test_renewed_lease_cannot_hide_a_later_expired_request() -> None:
    worker_type = _production_worker_methods("_expire_source_leases")
    worker = worker_type()
    worker._reqs_to_send = {"renewed": 30.0, "expired": 20.0}
    worker._source_lease_heap = [
        (10.0, "renewed"),
        (20.0, "expired"),
        (30.0, "renewed"),
    ]
    heapq.heapify(worker._source_lease_heap)
    worker.consumer_notification_counts_by_req = {}
    worker.xfer_stats = MagicMock()
    worker._on_source_lease_expired = MagicMock()

    worker._expire_source_leases(25.0)

    assert worker._reqs_to_send == {"renewed": 30.0}
    worker._on_source_lease_expired.assert_called_once_with("expired", 0)


def test_completion_releases_pull_source_without_a_live_deadline() -> None:
    worker_type = _production_pull_worker_methods("_get_new_notifs")
    worker = worker_type()
    worker.transfer_topo = SimpleNamespace(tp_ratio=lambda remote_size: 1)
    worker.nixl_wrapper = SimpleNamespace(
        get_new_notifs=lambda: {"consumer-agent": [b"producer-request:1:2"]}
    )
    worker._reqs_to_send = {}
    worker._reqs_to_process = {"producer-request"}
    worker.consumer_notification_counts_by_req = {"producer-request": 1}
    worker.world_size = 1
    worker._localization_capture_source_post = MagicMock()

    assert worker._get_new_notifs() == {"producer-request"}
    assert worker._reqs_to_process == set()
    worker._localization_capture_source_post.assert_called_once_with("producer-request")


def test_stock_status_exception_retains_every_unproven_handle() -> None:
    worker_type = _production_worker_methods("_pop_done_transfers")
    worker = worker_type()
    worker.nixl_wrapper = MagicMock()
    worker.nixl_wrapper.check_xfer_state.side_effect = ["PROC", RuntimeError("lost")]
    worker._log_failure = MagicMock()
    worker.xfer_stats = MagicMock()
    transfers = {"request": [101, 102, 103]}

    with pytest.raises(TransferQuiescenceError, match="status query failed"):
        worker._pop_done_transfers(transfers)

    assert transfers == {"request": [101, 102, 103]}
    worker.nixl_wrapper.release_xfer_handle.assert_not_called()


def test_stock_error_state_retains_native_ownership() -> None:
    worker_type = _production_worker_methods("_pop_done_transfers")
    worker = worker_type()
    worker.nixl_wrapper = MagicMock()
    worker.nixl_wrapper.check_xfer_state.return_value = "ERR"
    worker._log_failure = MagicMock()
    worker.xfer_stats = MagicMock()
    transfers = {"request": [201]}

    with pytest.raises(TransferQuiescenceError, match="only DONE"):
        worker._pop_done_transfers(transfers)

    assert transfers == {"request": [201]}
    worker.nixl_wrapper.release_xfer_handle.assert_not_called()


def test_stock_read_post_exception_retains_native_handle() -> None:
    worker_type = _production_pull_worker_methods("_read_blocks")
    worker = worker_type()
    remote_info = SimpleNamespace(
        remote_block_size=16,
        remote_physical_blocks_per_logical=1,
    )
    worker.transfer_topo = SimpleNamespace(
        get_engine_info=lambda engine_id: remote_info,
        block_size_ratio=lambda remote_block_size: 1,
    )
    worker.kv_cache_config = SimpleNamespace(kv_cache_groups=[object()])
    worker.dst_num_blocks = {"producer": 32, "decoder": 32}
    worker.engine_id = "decoder"
    worker.world_size = 1
    worker._physical_blocks_per_logical_kv_block = 1
    worker._apply_prefix_caching = lambda local, remote, ratio: (local, remote)
    worker._compute_desc_ids = MagicMock(return_value=[0])
    worker._assert_transfer_post_allowed = MagicMock()
    worker._recving_transfers = {"request": []}
    worker.nixl_wrapper = MagicMock()
    worker.nixl_wrapper.make_prepped_xfer.return_value = 301
    worker.nixl_wrapper.transfer.side_effect = RuntimeError("post outcome lost")
    worker._log_failure = MagicMock()
    read_spec = SimpleNamespace(
        remote_rank=0,
        local_block_ids=[[1]],
        remote_block_ids=[[2]],
    )

    with pytest.raises(TransferQuiescenceError, match="ambiguous"):
        worker._read_blocks(
            expected_consumers=1,
            read_spec=read_spec,
            dst_engine_id="producer",
            request_id="request",
            remote_request_id="source-request",
            local_xfer_side_handle=11,
            remote_xfer_side_handle=22,
        )

    assert worker._recving_transfers == {"request": [301]}
    worker.nixl_wrapper.release_xfer_handle.assert_not_called()


def test_stock_prepare_failure_cannot_publish_partial_request() -> None:
    worker_type = _production_pull_worker_methods("_read_blocks")
    worker = worker_type()
    remote_info = SimpleNamespace(
        remote_block_size=16,
        remote_physical_blocks_per_logical=1,
    )
    worker.transfer_topo = SimpleNamespace(
        get_engine_info=lambda engine_id: remote_info,
        block_size_ratio=lambda remote_block_size: 1,
    )
    worker.kv_cache_config = SimpleNamespace(kv_cache_groups=[object()])
    worker.dst_num_blocks = {"producer": 32, "decoder": 32}
    worker.engine_id = "decoder"
    worker.world_size = 1
    worker._physical_blocks_per_logical_kv_block = 1
    worker._apply_prefix_caching = lambda local, remote, ratio: (local, remote)
    worker._compute_desc_ids = MagicMock(return_value=[0])
    worker._assert_transfer_post_allowed = MagicMock()
    worker._recving_transfers = {"request": [300]}
    worker.nixl_wrapper = MagicMock()
    worker.nixl_wrapper.make_prepped_xfer.side_effect = RuntimeError("prepare lost")
    worker._log_failure = MagicMock()
    read_spec = SimpleNamespace(
        remote_rank=1,
        local_block_ids=[[1]],
        remote_block_ids=[[2]],
    )

    with pytest.raises(TransferQuiescenceError, match="already posted"):
        worker._read_blocks(
            expected_consumers=1,
            read_spec=read_spec,
            dst_engine_id="producer",
            request_id="request",
            remote_request_id="source-request",
            local_xfer_side_handle=11,
            remote_xfer_side_handle=22,
        )

    assert worker._recving_transfers == {"request": [300]}
    worker.nixl_wrapper.release_xfer_handle.assert_not_called()


def test_production_sync_failure_keeps_owned_generation() -> None:
    torch_module = MagicMock()
    torch_module.cuda.synchronize.side_effect = RuntimeError("sync exploded")
    worker_type = _production_worker_methods(
        "_coalesced_scatter",
        "_fail_coalesced_plan",
        "_release_coalesced_plan",
        torch_module=torch_module,
    )
    allocator, plan = _plan(statuses=("DONE",))
    worker = worker_type()
    worker._coalesce_plans = {plan.request_id: plan}
    worker._staging_allocator = allocator
    worker._staging_buf = None
    worker._region_rows = None
    worker._localization_config = MagicMock()
    worker._localization_config.enabled_for.return_value = False
    worker._audit_enabled = False

    with pytest.raises(StagingSafetyError, match="safety proof failed"):
        worker._coalesced_scatter(plan.request_id)

    assert allocator.require_active(plan.lease.generation) is plan
    assert plan.permanently_tombstoned
    assert plan.device_quiescent is False


def test_unresolved_shutdown_and_cleanup_touch_no_native_resources() -> None:
    worker_type = _production_worker_methods("_cleanup_remote_engine", "shutdown")
    _, plan = _plan(
        statuses=("PROC",),
        remote_engine_id="producer",
        scatter={"producer_engine_id": "untrusted-geometry-value"},
    )
    worker = worker_type()
    worker._coalesce_plans = {plan.request_id: plan}
    worker._remote_agents = {"producer": {0: "agent"}}
    worker.nixl_wrapper = MagicMock()
    worker._handshake_initiation_executor = MagicMock()

    with pytest.raises(StagingSafetyError, match="live staging"):
        worker._cleanup_remote_engine("producer")
    with pytest.raises(StagingSafetyError, match="cannot quiesce"):
        worker.shutdown()

    worker.nixl_wrapper.release_xfer_handle.assert_not_called()
    worker.nixl_wrapper.release_dlist_handle.assert_not_called()
    worker.nixl_wrapper.remove_remote_agent.assert_not_called()
    worker.nixl_wrapper.deregister_memory.assert_not_called()
    worker._handshake_initiation_executor.shutdown.assert_not_called()


def test_remote_identity_is_typed_and_scatter_input_is_copied() -> None:
    allocator = StagingRangeAllocator(1024)
    scatter = {"producer_engine_id": "initial-geometry-value"}
    plan = allocator.create_plan(
        owner_id="identity-owner",
        request_id="identity-request",
        size=512,
        source_ranks=(0,),
        remote_engine_id="authoritative-producer",
        scatter=scatter,
    )
    assert plan is not None

    scatter["producer_engine_id"] = "rewritten-geometry-value"

    assert plan.remote_engine_id == "authoritative-producer"
    assert plan.scatter["producer_engine_id"] == "initial-geometry-value"


def test_ownership_snapshot_has_typed_json_compatible_evidence() -> None:
    _, plan = _plan(statuses=("DONE",), created_at=1.0)
    plan.mark_native_released(0)

    snapshot = plan.snapshot(now=3.5)
    assert isinstance(snapshot, StagingOwnershipSnapshot)
    assert snapshot.generation == plan.lease.generation
    assert snapshot.age_s == 2.5
    assert snapshot.handles == (
        StagingHandleSnapshot(
            source_rank=0,
            state=HandleState.DONE,
            native_handle_present=True,
            native_released=True,
        ),
    )
    assert snapshot.to_dict() == {
        "generation": plan.lease.generation,
        "owner_id": "test-owner",
        "request_id": "test-request",
        "remote_engine_id": "test-producer",
        "offset": 0,
        "size": 512,
        "age_s": 2.5,
        "handles": [
            {
                "source_rank": 0,
                "state": "done",
                "native_handle_present": True,
                "native_released": True,
            }
        ],
        "posting_sealed": True,
        "operation_failed": False,
        "permanently_tombstoned": False,
        "native_quiescent": True,
        "device_read_started": False,
        "device_quiescent": True,
        "reusable": True,
        "released": False,
        "failure_reason": None,
    }
