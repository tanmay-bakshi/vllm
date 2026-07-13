"""Source-bound tests for production coalesced-staging ownership."""

import ast
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Never
from unittest.mock import MagicMock

import pytest

from vllm.distributed.kv_transfer.nixl_localization import LocalizationMode
from vllm.distributed.kv_transfer.staging_ownership import (
    CoalescedStagingPlan,
    HandleState,
    StagingHandleSnapshot,
    StagingOwnershipSnapshot,
    StagingRangeAllocator,
    StagingSafetyError,
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
        "IntegrityStage": SimpleNamespace(
            STAGING_RAW="staging_raw",
            STAGING_FENCED_CONTROL="staging_fenced_control",
            STAGING_POST_SCATTER="staging_post_scatter",
            DESTINATION="destination",
        ),
        "LocalizationMode": LocalizationMode,
        "Never": Never,
        "ReqId": str,
        "StagingSafetyError": StagingSafetyError,
        "logger": MagicMock(),
        "time": SimpleNamespace(monotonic=lambda: monotonic_time),
        "torch": MagicMock() if torch_module is None else torch_module,
        "traceback": traceback,
    }
    exec(compile(module, str(BASE_WORKER_PATH), "exec"), namespace)
    return namespace["ProductionWorkerMethods"]  # type: ignore[return-value]


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


def test_fingerprint_observation_finishes_before_staging_release() -> None:
    events: list[str] = []
    torch_module = MagicMock()
    torch_module.cuda.synchronize.side_effect = lambda: events.append("sync")
    worker_type = _production_worker_methods(
        "_coalesced_scatter",
        torch_module=torch_module,
    )
    _, plan = _plan(
        statuses=("DONE",),
        scatter={
            "lpos": (),
            "n_pos": 0,
            "n_ranks": 1,
            "slots": (0,),
            "blens": (),
            "region_off": (),
        },
    )
    worker = worker_type()
    worker._coalesce_plans = {plan.request_id: plan}
    worker._staging_buf = MagicMock(device="cuda:0")
    worker._region_rows = []
    worker._audit_enabled = False
    worker._localization_pre_read_plans = {}
    worker._localization_config = MagicMock()
    worker._localization_config.enabled_for.return_value = True
    worker._localization_config.mode = LocalizationMode.FINGERPRINT

    def capture_staging(*args: object) -> None:
        assert plan.device_quiescent is False
        events.append("staging")

    def capture_destination(*args: object) -> None:
        assert plan.device_quiescent is False
        events.append("destination")

    def release(completed: CoalescedStagingPlan) -> None:
        assert completed is plan
        assert plan.device_quiescent
        events.append("release")

    worker._localization_capture_staging = capture_staging
    worker._localization_capture_destination = capture_destination
    worker._release_coalesced_plan = release

    worker._coalesced_scatter(plan.request_id)

    assert events == ["sync", "staging", "destination", "sync", "release"]
    assert worker._localization_pre_read_plans == {plan.request_id: plan.scatter}


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
