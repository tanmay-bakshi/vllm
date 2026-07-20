# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Source-bound tests for production coalesced-staging ownership."""

import ast
import traceback
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Never
from unittest.mock import MagicMock

import pytest

from vllm.distributed.kv_transfer.coalesced_layout import (
    CoalescedTransferPlan,
    GroupTransferRoster,
    RegionOwnership,
    build_coalesced_transfer_plan,
)
from vllm.distributed.kv_transfer.integrity import IntegrityStage
from vllm.distributed.kv_transfer.nixl_localization import (
    LocalizationError,
    LocalizationMode,
    localization_stage_barrier,
)
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
PULL_WORKER_PATH = BASE_WORKER_PATH.with_name("pull_worker.py")


class _ScatterEnqueueError(RuntimeError):
    """Synthetic enqueue failure carrying an optional recovery event."""

    recovery_launch: "_FakeScatterLaunch | None"

    def __init__(
        self,
        message: str,
        recovery_launch: "_FakeScatterLaunch | None",
    ) -> None:
        """Create one failure.

        :param message: Failure detail.
        :param recovery_launch: Event retaining ownership of partial work.
        """
        super().__init__(message)
        self.recovery_launch = recovery_launch


@dataclass(slots=True)
class _FakeScatterLaunch:
    """Controllable completion and timing event surface."""

    complete: bool
    gpu_milliseconds: float = 2.0
    query_error: BaseException | None = None
    timing_error: BaseException | None = None
    query_count: int = 0
    timing_count: int = 0

    def is_complete(self) -> bool:
        """Return or fail the synthetic completion query."""
        self.query_count += 1
        if self.query_error is not None:
            raise self.query_error
        return self.complete

    def gpu_duration_ms(self) -> float:
        """Return or fail the synthetic event timing query."""
        self.timing_count += 1
        if self.timing_error is not None:
            raise self.timing_error
        return self.gpu_milliseconds


class _PlanTelemetry:
    """Small production-compatible aggregate used by lifecycle tests."""

    @classmethod
    def from_native_transfers(
        cls,
        *,
        logical_bytes: int,
        layout_duration_seconds: float,
        native_transfers: tuple[object, ...],
        scatter_duration_seconds: float,
        scatter_gpu_duration_seconds: float | None,
        staging_residency_seconds: float,
    ) -> SimpleNamespace:
        """Aggregate the fields consumed by the production completion path."""
        return SimpleNamespace(
            logical_bytes=logical_bytes,
            wire_bytes=sum(int(transfer.totalBytes) for transfer in native_transfers),
            descriptor_count=sum(
                int(transfer.descCount) for transfer in native_transfers
            ),
            scatter_duration_seconds=scatter_duration_seconds,
            scatter_gpu_duration_seconds=scatter_gpu_duration_seconds,
            staging_residency_seconds=staging_residency_seconds,
            layout_duration_seconds=layout_duration_seconds,
        )


def _production_worker_methods(
    *method_names: str,
    monotonic_time: float = 0.0,
    torch_module: object | None = None,
    validate_scatter: object | None = None,
    launch_scatter: object | None = None,
) -> type:
    """Compile exact production methods without constructing a CUDA worker.

    :param method_names: Methods to bind to the test class.
    :param monotonic_time: Deterministic clock observation.
    :param torch_module: Fake torch surface for scatter tests.
    :param validate_scatter: Pre-enqueue validator implementation.
    :param launch_scatter: Asynchronous scatter launcher implementation.
    :returns: Class containing the exact production method bodies.
    """
    syntax = ast.parse(BASE_WORKER_PATH.read_text(), filename=str(BASE_WORKER_PATH))
    worker_node = next(
        node
        for node in syntax.body
        if isinstance(node, ast.ClassDef) and node.name == "NixlBaseConnectorWorker"
    )
    pending_node = next(
        node
        for node in syntax.body
        if isinstance(node, ast.ClassDef) and node.name == "_PendingCoalescedScatter"
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
    module = ast.fix_missing_locations(
        ast.Module(body=[pending_node, test_class], type_ignores=[])
    )
    namespace: dict[str, object] = {
        "Any": Any,
        "BaseException": BaseException,
        "CoalescedStagingPlan": CoalescedStagingPlan,
        "EngineId": str,
        "HandleState": HandleState,
        "IntegrityStage": IntegrityStage,
        "KVTransferFailureReason": SimpleNamespace(TRANSFER="transfer"),
        "LocalizationError": LocalizationError,
        "LocalizationMode": LocalizationMode,
        "Never": Never,
        "NixlCoalescedPlanTelemetry": _PlanTelemetry,
        "ReqId": str,
        "ScatterEnqueueError": _ScatterEnqueueError,
        "ScatterLaunch": _FakeScatterLaunch,
        "StagingSafetyError": StagingSafetyError,
        "dataclass": dataclass,
        "launch_coalesced_scatter": (
            MagicMock() if launch_scatter is None else launch_scatter
        ),
        "logger": MagicMock(),
        "localization_stage_barrier": localization_stage_barrier,
        "time": SimpleNamespace(monotonic=lambda: monotonic_time),
        "torch": MagicMock() if torch_module is None else torch_module,
        "traceback": traceback,
        "validate_coalesced_scatter": (
            MagicMock() if validate_scatter is None else validate_scatter
        ),
    }
    exec(compile(module, str(BASE_WORKER_PATH), "exec"), namespace)
    worker_type = namespace["ProductionWorkerMethods"]
    worker_type._pending_scatter_type = namespace["_PendingCoalescedScatter"]
    return worker_type  # type: ignore[return-value]


def _production_pull_worker_methods(*method_names: str) -> type:
    """Compile exact pull-worker methods without constructing a CUDA worker.

    :param method_names: Pull-worker methods to bind to the test class.
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
        "NixlConnectorMetadata": object,
        "logger": MagicMock(),
    }
    exec(compile(module, str(PULL_WORKER_PATH), "exec"), namespace)
    return namespace["ProductionPullWorkerMethods"]  # type: ignore[return-value]


def _layout(source_rank_count: int) -> CoalescedTransferPlan:
    """Build one exact canonical layout for synthetic ownership tests.

    :param source_rank_count: Number of native source-rank handles.
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
    """Return one complete synthetic NIXL telemetry record."""
    return SimpleNamespace(
        xferDuration=10,
        postDuration=2,
        totalBytes=total_bytes,
        descCount=1,
    )


def _plan(
    *,
    statuses: tuple[str, ...],
    allocator: StagingRangeAllocator | None = None,
    owner_id: str = "test-owner",
    request_id: str = "test-request",
    created_at: float = 0.0,
    remote_engine_id: str = "test-producer",
) -> tuple[StagingRangeAllocator, CoalescedStagingPlan]:
    """Create one sealed production owner with synthetic native handles.

    :param statuses: Immediate NIXL status per source rank.
    :param allocator: Shared allocator, or a new allocator.
    :param owner_id: Unique allocation owner.
    :param request_id: Unique decoder request.
    :param created_at: Deterministic creation time.
    :param remote_engine_id: Synthetic remote engine identity.
    :returns: Allocator and exact active plan.
    """
    resolved_allocator = StagingRangeAllocator(4096) if allocator is None else allocator
    plan = resolved_allocator.create_plan(
        owner_id=owner_id,
        request_id=request_id,
        layout=_layout(len(statuses)),
        layout_duration_seconds=0.001,
        remote_engine_id=remote_engine_id,
        created_at=created_at,
        warn_after_s=1.0,
        fail_after_s=2.0,
    )
    assert plan is not None
    for source_rank, status in enumerate(statuses):
        plan.begin_prepare(source_rank)
        plan.attach_handle(source_rank, object())
        plan.begin_post(source_rank)
        plan.record_post_result(source_rank, status)
    plan.seal_posting()
    return resolved_allocator, plan


def _ready_plan(
    *,
    allocator: StagingRangeAllocator | None = None,
    owner_id: str = "test-owner",
    request_id: str = "test-request",
    source_rank_count: int = 1,
) -> tuple[StagingRangeAllocator, CoalescedStagingPlan]:
    """Create a plan with complete native telemetry and released handles."""
    resolved_allocator, plan = _plan(
        statuses=("DONE",) * source_rank_count,
        allocator=allocator,
        owner_id=owner_id,
        request_id=request_id,
    )
    for source_rank in plan.layout.source_ranks:
        plan.record_native_telemetry(
            source_rank,
            _native_telemetry(plan.layout.rank_stride_bytes),
        )
        plan.mark_native_released(source_rank)
    assert plan.ready_to_scatter
    return resolved_allocator, plan


def _worker(
    worker_type: type,
    allocator: StagingRangeAllocator,
    *plans: CoalescedStagingPlan,
) -> object:
    """Initialize the production fields used by selected lifecycle methods."""
    worker = worker_type()
    worker._coalesce_plans = {plan.request_id: plan for plan in plans}
    worker._pending_coalesced_scatters = {}
    worker._staging_allocator = allocator
    worker._staging_buf = SimpleNamespace(device="cuda:0")
    worker._region_rows = (MagicMock(),)
    worker._coalesced_scatter_stream = MagicMock()
    worker._localization_config = MagicMock()
    worker._localization_config.enabled_for.return_value = False
    worker._coalesced_localization_plans = {}
    worker._localization_pre_read_plans = {}
    worker._localization_record_event = MagicMock()
    worker._recving_metadata = {
        plan.request_id: SimpleNamespace(remote=None) for plan in plans
    }
    worker._record_remote_source_consumption_proven = MagicMock()
    worker._record_failed_receive = MagicMock()
    worker._audit_enabled = False
    worker._audit_pending = []
    worker._packed_remote_engine_active = MagicMock(return_value=False)
    worker._packed_remote_engine_cleanup = MagicMock()
    worker._packed_registered_ownership_active = MagicMock(return_value=False)
    worker._packed_shutdown_cleanup = MagicMock()
    worker._clear_remote_packed_write_pools = MagicMock()
    worker.nixl_wrapper = MagicMock()
    worker.xfer_stats = MagicMock()
    return worker


_SAFE_FAILURE_METHODS = (
    "_fail_coalesced_plan",
    "_finish_quiescent_coalesced_request_failure",
    "_release_coalesced_plan",
    "_release_quiescent_failed_coalesced_plan",
)


@pytest.mark.parametrize("owner_state", ("parked", "native", "scatter"))
def test_not_processed_bookkeeping_never_cancels_owned_receive(
    owner_state: str,
) -> None:
    """Producer bookkeeping cannot become decoder receive cancellation."""
    request_id = "scheduler-aborted-request"
    worker_type = _production_pull_worker_methods("start_load_kv")
    worker = worker_type()
    parked = (request_id, object(), [object()], ())
    native_plan = object()
    pending_scatter = object()
    worker.tp_rank = 0
    worker._coalesce_pending = deque([parked] if owner_state == "parked" else ())
    worker._coalesce_plans = (
        {request_id: native_plan} if owner_state == "native" else {}
    )
    worker._pending_coalesced_scatters = (
        {request_id: pending_scatter} if owner_state == "scatter" else {}
    )
    worker._reqs_to_process = {request_id}
    worker._buffered_pull_completions = {request_id: object()}
    worker._buffered_offer_cancellations = {request_id: object()}
    worker._localization_pre_read_plans = {}
    worker._reqs_to_send = {}
    worker._pending_offer_cancellations = []
    worker._ready_requests = MagicMock()
    worker._ready_requests.empty.return_value = True
    worker._update_heartbeat_targets = MagicMock()
    worker._service_heartbeats = MagicMock()
    worker._begin_transfer_phase = MagicMock()
    worker._audit_retire = MagicMock()
    worker._capture_source_rosters = MagicMock()
    worker._record_packed_source_readiness = MagicMock()
    worker._service_packed_producer = MagicMock()
    worker._localization_record_event = MagicMock()
    worker._drain_transfer_phase = MagicMock()
    worker._localization_capture_pre_read = MagicMock()
    worker._record_transfer_decode_boundary = MagicMock()
    worker._record_failed_receive = MagicMock()
    worker._handle_failed_transfer = MagicMock()
    worker._apply_local_source_retirement = MagicMock()
    worker._apply_remote_source_retirement = MagicMock()
    metadata = SimpleNamespace(
        reqs_to_recv={},
        source_rosters={},
        source_retired_through=0,
        reqs_in_batch=set(),
        reqs_not_processed={request_id},
        reqs_to_send={},
        offer_cancellations_by_rank={0: ()},
        scheduled_request_ids=(),
    )

    worker.start_load_kv(metadata)

    assert request_id not in worker._reqs_to_process
    assert request_id not in worker._buffered_pull_completions
    assert request_id not in worker._buffered_offer_cancellations
    if owner_state == "parked":
        assert tuple(worker._coalesce_pending) == (parked,)
    elif owner_state == "native":
        assert worker._coalesce_plans == {request_id: native_plan}
    else:
        assert worker._pending_coalesced_scatters == {request_id: pending_scatter}
    worker._record_failed_receive.assert_not_called()
    worker._handle_failed_transfer.assert_not_called()


def test_parked_receive_retries_fifo_until_it_posts() -> None:
    """A parked receive remains owned and drains through its normal post path."""
    request_id = "parked-request"
    metadata = object()
    read_specs = [object()]
    source_contracts: tuple[object, ...] = ()
    parked = (request_id, metadata, read_specs, source_contracts)
    worker_type = _production_pull_worker_methods("_coalesce_service_pending")
    worker = worker_type()
    worker._coalesce_pending = deque([parked])
    worker._read_completion_notification = MagicMock(return_value=b"proof")
    worker._coalesced_read_request = MagicMock(side_effect=("defer", "posted"))
    worker._handle_failed_transfer = MagicMock()
    worker._stock_read_specs = MagicMock()

    worker._coalesce_service_pending()

    assert tuple(worker._coalesce_pending) == (parked,)
    worker._coalesced_read_request.assert_called_once_with(
        request_id,
        metadata,
        read_specs,
        b"proof",
        source_contracts,
    )

    worker._coalesce_service_pending()

    assert len(worker._coalesce_pending) == 0
    assert worker._coalesced_read_request.call_count == 2
    worker._handle_failed_transfer.assert_not_called()
    worker._stock_read_specs.assert_not_called()


def test_native_inflight_receive_drains_through_scatter_to_healthy_terminal() -> None:
    """A live receive publishes only after native and device quiescence."""
    launch = _FakeScatterLaunch(complete=False)
    validator = MagicMock()
    launcher = MagicMock(return_value=launch)
    worker_type = _production_worker_methods(
        "_coalesced_scatter",
        "_fail_coalesced_plan",
        "_poll_coalesced_plans",
        "_poll_pending_coalesced_scatters",
        "_release_coalesced_plan",
        validate_scatter=validator,
        launch_scatter=launcher,
    )
    allocator, plan = _plan(statuses=("PROC",))
    worker = _worker(worker_type, allocator, plan)
    worker.nixl_wrapper.check_xfer_state.side_effect = ("PROC", "DONE")
    worker.nixl_wrapper.get_xfer_telemetry.return_value = _native_telemetry(
        plan.layout.staging_size_bytes
    )

    assert worker._poll_coalesced_plans() == set()
    assert allocator.require_active(plan.lease.generation) is plan
    assert plan.slots[0].state is HandleState.PROC

    assert worker._poll_coalesced_plans() == set()
    assert plan.native_quiescent
    assert plan.device_read_started
    assert plan.device_quiescent is False
    assert plan.request_id in worker._pending_coalesced_scatters
    assert allocator.require_active(plan.lease.generation) is plan

    launch.complete = True
    assert worker._poll_coalesced_plans() == {plan.request_id}

    assert plan.released
    assert plan.request_id not in worker._coalesce_plans
    assert plan.request_id not in worker._pending_coalesced_scatters
    worker._record_failed_receive.assert_not_called()
    worker.xfer_stats.record_failed_transfer.assert_not_called()
    worker.xfer_stats.record_coalesced_plan.assert_called_once()


def test_production_err_with_proc_fails_before_native_or_range_release() -> None:
    """One failed native rank preserves every possibly live writer."""
    worker_type = _production_worker_methods(
        "_fail_coalesced_plan",
        "_poll_coalesced_plans",
    )
    allocator, plan = _plan(statuses=("PROC", "ERR"))
    worker = _worker(worker_type, allocator, plan)
    worker._poll_pending_coalesced_scatters = MagicMock(return_value=set())

    with pytest.raises(StagingSafetyError, match="safety proof failed"):
        worker._poll_coalesced_plans()

    assert allocator.require_active(plan.lease.generation) is plan
    worker.nixl_wrapper.check_xfer_state.assert_not_called()
    worker.nixl_wrapper.release_xfer_handle.assert_not_called()
    assert plan.slots[0].state is HandleState.PROC
    assert plan.slots[1].native_handle is not None


def test_production_mid_post_exception_retains_handle_and_lease() -> None:
    """An exception across the native post boundary permanently tombstones."""
    worker_type = _production_worker_methods(
        "_fail_coalesced_plan",
        "_prepare_coalesced_handle",
        "_post_prepared_coalesced",
    )
    allocator = StagingRangeAllocator(4096)
    plan = allocator.create_plan(
        owner_id="post-owner",
        request_id="post-request",
        layout=_layout(1),
        layout_duration_seconds=0.0,
        remote_engine_id="test-producer",
    )
    assert plan is not None
    plan.begin_prepare(0)
    native_handle = object()
    worker = _worker(worker_type, allocator, plan)
    worker._assert_transfer_post_allowed = MagicMock()
    worker.nixl_wrapper.initialize_xfer.return_value = native_handle
    worker.nixl_wrapper.transfer.side_effect = RuntimeError("post exploded")

    assert worker._prepare_coalesced_handle(
        plan,
        0,
        object(),
        object(),
        "remote-agent",
        b"notification",
    )
    with pytest.raises(StagingSafetyError, match="safety proof failed"):
        worker._post_prepared_coalesced(plan, 0)

    assert allocator.require_active(plan.lease.generation) is plan
    assert plan.slots[0].native_handle is native_handle
    assert plan.slots[0].state is HandleState.UNKNOWN
    assert plan.permanently_tombstoned
    worker.nixl_wrapper.release_xfer_handle.assert_not_called()


def test_all_rank_preparation_failure_occurs_before_any_native_post() -> None:
    """A later-rank preparation failure reclaims every unposted handle."""
    worker_type = _production_worker_methods(
        "_fail_coalesced_plan",
        "_finish_quiescent_coalesced_failure",
        "_prepare_coalesced_handle",
        "_release_coalesced_plan",
        "_release_quiescent_failed_coalesced_plan",
    )
    allocator = StagingRangeAllocator(4096)
    plan = allocator.create_plan(
        owner_id="prepare-owner",
        request_id="prepare-request",
        layout=_layout(4),
        layout_duration_seconds=0.0,
        remote_engine_id="test-producer",
    )
    assert plan is not None
    worker = _worker(worker_type, allocator, plan)
    worker._handle_failed_transfer = MagicMock()
    prepared_handle = object()
    worker.nixl_wrapper.initialize_xfer.side_effect = (
        prepared_handle,
        RuntimeError("rank one descriptor preparation failed"),
    )

    plan.begin_prepare(0)
    assert worker._prepare_coalesced_handle(
        plan,
        0,
        object(),
        object(),
        "remote-agent-0",
        b"notification",
    )
    plan.begin_prepare(1)
    assert not worker._prepare_coalesced_handle(
        plan,
        1,
        object(),
        object(),
        "remote-agent-1",
        b"notification",
    )

    assert plan.released
    assert plan.slots[0].state is HandleState.SEALED_UNPOSTED
    assert plan.slots[1].state is HandleState.PREPARE_FAILED
    assert plan.slots[2].state is HandleState.NEVER_POSTED
    assert plan.slots[3].state is HandleState.NEVER_POSTED
    worker.nixl_wrapper.transfer.assert_not_called()
    worker.nixl_wrapper.release_xfer_handle.assert_called_once_with(prepared_handle)


def test_guarded_post_failure_stops_before_this_or_later_rank_transfer() -> None:
    """A rejected post boundary raises fail-stop instead of returning to the loop."""
    worker_type = _production_worker_methods(
        "_fail_coalesced_plan",
        "_post_prepared_coalesced",
    )
    allocator = StagingRangeAllocator(4096)
    plan = allocator.create_plan(
        owner_id="guard-owner",
        request_id="guard-request",
        layout=_layout(2),
        layout_duration_seconds=0.0,
        remote_engine_id="test-producer",
    )
    assert plan is not None
    for source_rank in plan.layout.source_ranks:
        plan.begin_prepare(source_rank)
        plan.attach_handle(source_rank, object())
    worker = _worker(worker_type, allocator, plan)
    worker._assert_transfer_post_allowed = MagicMock(
        side_effect=StagingSafetyError("transfer phase closed")
    )

    with pytest.raises(StagingSafetyError, match="safety proof failed"):
        worker._post_prepared_coalesced(plan, 0)

    worker.nixl_wrapper.transfer.assert_not_called()
    assert plan.permanently_tombstoned
    assert plan.slots[0].state is HandleState.SEALED_UNPOSTED
    assert plan.slots[1].state is HandleState.SEALED_UNPOSTED
    assert allocator.require_active(plan.lease.generation) is plan


def test_coalesced_request_prepares_every_rank_before_posting() -> None:
    """The request transaction has distinct all-prepare and all-post loops."""
    syntax = ast.parse(PULL_WORKER_PATH.read_text(), filename=str(PULL_WORKER_PATH))
    method = next(
        node
        for node in ast.walk(syntax)
        if isinstance(node, ast.FunctionDef) and node.name == "_coalesced_read_request"
    )
    prepare_loop = next(
        node
        for node in ast.walk(method)
        if isinstance(node, ast.For)
        and any(
            isinstance(descendant, ast.Call)
            and isinstance(descendant.func, ast.Attribute)
            and descendant.func.attr == "_prepare_coalesced_handle"
            for descendant in ast.walk(node)
        )
    )
    post_loop = next(
        node
        for node in ast.walk(method)
        if isinstance(node, ast.For)
        and any(
            isinstance(descendant, ast.Call)
            and isinstance(descendant.func, ast.Attribute)
            and descendant.func.attr == "_post_prepared_coalesced"
            for descendant in ast.walk(node)
        )
    )

    assert prepare_loop.end_lineno is not None
    assert prepare_loop.end_lineno < post_loop.lineno
    assert any(
        isinstance(descendant, ast.Call)
        and isinstance(descendant.func, ast.Attribute)
        and descendant.func.attr == "get_xfer_descs"
        for descendant in ast.walk(prepare_loop)
    )


def test_production_polls_before_timeout_then_tombstones_proc() -> None:
    """A live native writer is queried before the fail-stop deadline wins."""
    worker_type = _production_worker_methods(
        "_fail_coalesced_plan",
        "_poll_coalesced_plans",
        monotonic_time=3.0,
    )
    allocator, plan = _plan(statuses=("PROC",), created_at=0.0)
    worker = _worker(worker_type, allocator, plan)
    worker._poll_pending_coalesced_scatters = MagicMock(return_value=set())
    worker.nixl_wrapper.check_xfer_state.return_value = "PROC"

    with pytest.raises(StagingSafetyError, match="fail-stop deadline"):
        worker._poll_coalesced_plans()

    worker.nixl_wrapper.check_xfer_state.assert_called_once()
    worker.nixl_wrapper.release_xfer_handle.assert_not_called()
    assert allocator.require_active(plan.lease.generation) is plan
    assert plan.permanently_tombstoned


def test_production_done_observed_at_deadline_is_not_failed() -> None:
    """A terminal native observation is consumed before timeout evaluation."""
    worker_type = _production_worker_methods(
        "_fail_coalesced_plan",
        "_poll_coalesced_plans",
        monotonic_time=3.0,
    )
    allocator, plan = _plan(statuses=("PROC",), created_at=0.0)
    worker = _worker(worker_type, allocator, plan)
    worker._coalesced_scatter = MagicMock()
    worker._poll_pending_coalesced_scatters = MagicMock(return_value=set())
    worker.nixl_wrapper.check_xfer_state.return_value = "DONE"
    worker.nixl_wrapper.get_xfer_telemetry.return_value = _native_telemetry(
        plan.layout.staging_size_bytes
    )

    assert worker._poll_coalesced_plans() == set()

    assert plan.permanently_tombstoned is False
    worker.nixl_wrapper.release_xfer_handle.assert_called_once()
    worker._coalesced_scatter.assert_called_once_with(plan.request_id)


def test_native_done_does_not_publish_or_release_before_scatter_event() -> None:
    """Native DONE permits enqueue, not range reclamation or publication."""
    launch = _FakeScatterLaunch(complete=False)
    validate = MagicMock()
    launcher = MagicMock(return_value=launch)
    worker_type = _production_worker_methods(
        "_coalesced_scatter",
        "_fail_coalesced_plan",
        "_poll_coalesced_plans",
        "_poll_pending_coalesced_scatters",
        "_release_coalesced_plan",
        validate_scatter=validate,
        launch_scatter=launcher,
    )
    allocator, plan = _plan(statuses=("DONE",))
    worker = _worker(worker_type, allocator, plan)
    worker.nixl_wrapper.get_xfer_telemetry.return_value = _native_telemetry(
        plan.layout.staging_size_bytes
    )

    assert worker._poll_coalesced_plans() == set()

    assert plan.ready_to_scatter
    assert plan.device_read_started
    assert plan.device_quiescent is False
    assert plan.request_id in worker._pending_coalesced_scatters
    assert allocator.require_active(plan.lease.generation) is plan
    assert plan.released is False
    validate.assert_called_once()
    launcher.assert_called_once()


def test_incomplete_scatter_event_remains_owned_across_polls() -> None:
    """An incomplete event retains the launch, tensors, and staging lease."""
    launch = _FakeScatterLaunch(complete=False)
    worker_type = _production_worker_methods(
        "_poll_pending_coalesced_scatters",
        "_release_coalesced_plan",
    )
    allocator, plan = _ready_plan()
    plan.begin_device_read()
    worker = _worker(worker_type, allocator, plan)
    worker._pending_coalesced_scatters[plan.request_id] = (
        worker_type._pending_scatter_type(
            plan=plan,
            launch=launch,
            enqueued_at=0.0,
        )
    )

    assert worker._poll_pending_coalesced_scatters() == set()
    assert worker._poll_pending_coalesced_scatters() == set()

    assert launch.query_count == 2
    assert launch.timing_count == 0
    assert plan.request_id in worker._pending_coalesced_scatters
    assert allocator.require_active(plan.lease.generation) is plan
    assert plan.device_quiescent is False


def test_successful_scatter_completion_publishes_and_releases_exactly_once() -> None:
    """One ready event produces one metric, publication, and range release."""
    launch = _FakeScatterLaunch(complete=True)
    worker_type = _production_worker_methods(
        "_poll_pending_coalesced_scatters",
        "_release_coalesced_plan",
    )
    allocator, plan = _ready_plan()
    plan.begin_device_read()
    original_release = allocator.release
    allocator.release = MagicMock(side_effect=original_release)
    worker = _worker(worker_type, allocator, plan)
    worker._pending_coalesced_scatters[plan.request_id] = (
        worker_type._pending_scatter_type(
            plan=plan,
            launch=launch,
            enqueued_at=0.0,
        )
    )

    assert worker._poll_pending_coalesced_scatters() == {plan.request_id}
    assert worker._poll_pending_coalesced_scatters() == set()

    allocator.release.assert_called_once_with(plan)
    worker.xfer_stats.record_coalesced_plan.assert_called_once()
    assert plan.released
    assert plan.request_id not in worker._coalesce_plans
    assert plan.request_id not in worker._pending_coalesced_scatters
    assert launch.query_count == 1
    assert launch.timing_count == 1
    worker._record_failed_receive.assert_not_called()
    worker.xfer_stats.record_failed_transfer.assert_not_called()


def test_scatter_event_query_failure_tombstones() -> None:
    """Loss of the completion proof forbids range reuse and publication."""
    launch = _FakeScatterLaunch(
        complete=True,
        query_error=RuntimeError("query exploded"),
    )
    worker_type = _production_worker_methods(
        "_fail_coalesced_plan",
        "_poll_pending_coalesced_scatters",
        "_release_coalesced_plan",
    )
    allocator, plan = _ready_plan()
    plan.begin_device_read()
    worker = _worker(worker_type, allocator, plan)
    worker._pending_coalesced_scatters[plan.request_id] = (
        worker_type._pending_scatter_type(
            plan=plan,
            launch=launch,
            enqueued_at=0.0,
        )
    )

    with pytest.raises(StagingSafetyError, match="event query failed"):
        worker._poll_pending_coalesced_scatters()

    assert plan.permanently_tombstoned
    assert plan.device_quiescent is False
    assert allocator.require_active(plan.lease.generation) is plan
    assert plan.request_id in worker._pending_coalesced_scatters
    worker.xfer_stats.record_coalesced_plan.assert_not_called()


def test_scatter_timing_failure_omits_only_gpu_time_after_event_quiescence() -> None:
    """Optional timing telemetry cannot invalidate or mislabel a completed event."""
    launch = _FakeScatterLaunch(
        complete=True,
        timing_error=RuntimeError("timing exploded"),
    )
    worker_type = _production_worker_methods(
        "_poll_pending_coalesced_scatters",
        "_release_coalesced_plan",
    )
    allocator, plan = _ready_plan()
    plan.begin_device_read()
    worker = _worker(worker_type, allocator, plan)
    worker._pending_coalesced_scatters[plan.request_id] = (
        worker_type._pending_scatter_type(
            plan=plan,
            launch=launch,
            enqueued_at=0.0,
        )
    )

    assert worker._poll_pending_coalesced_scatters() == {plan.request_id}

    assert plan.released
    assert plan.permanently_tombstoned is False
    assert plan.request_id not in worker._pending_coalesced_scatters
    worker.xfer_stats.record_coalesced_plan.assert_called_once()
    telemetry = worker.xfer_stats.record_coalesced_plan.call_args.args[0]
    assert telemetry.scatter_gpu_duration_seconds is None
    assert telemetry.scatter_duration_seconds == 0.0


def test_partial_enqueue_waits_for_recovery_event_then_fails_safely() -> None:
    """Partially enqueued kernels retain ownership until their recovery event."""
    recovery = _FakeScatterLaunch(complete=False)
    launcher = MagicMock(side_effect=_ScatterEnqueueError("partial enqueue", recovery))
    worker_type = _production_worker_methods(
        "_coalesced_scatter",
        "_poll_pending_coalesced_scatters",
        *_SAFE_FAILURE_METHODS,
        validate_scatter=MagicMock(),
        launch_scatter=launcher,
    )
    allocator, plan = _ready_plan()
    worker = _worker(worker_type, allocator, plan)

    worker._coalesced_scatter(plan.request_id)

    assert plan.operation_failed
    assert plan.permanently_tombstoned is False
    assert plan.device_quiescent is False
    assert worker._poll_pending_coalesced_scatters() == set()
    assert allocator.require_active(plan.lease.generation) is plan

    recovery.complete = True
    assert worker._poll_pending_coalesced_scatters() == set()

    assert plan.device_quiescent
    assert plan.released
    assert plan.request_id not in worker._coalesce_plans
    worker._record_failed_receive.assert_called_once()
    worker.xfer_stats.record_failed_transfer.assert_called_once()


def test_validation_failure_precedes_device_ownership_and_releases_safely() -> None:
    """Invalid geometry fails before any stream reader can touch staging."""
    validator = MagicMock(side_effect=ValueError("bad geometry"))
    launcher = MagicMock()
    worker_type = _production_worker_methods(
        "_coalesced_scatter",
        *_SAFE_FAILURE_METHODS,
        validate_scatter=validator,
        launch_scatter=launcher,
    )
    allocator, plan = _ready_plan()
    worker = _worker(worker_type, allocator, plan)

    worker._coalesced_scatter(plan.request_id)

    assert plan.device_read_started is False
    assert plan.device_quiescent
    assert plan.operation_failed
    assert plan.released
    launcher.assert_not_called()
    worker._record_failed_receive.assert_called_once()


def test_native_total_bytes_mismatch_never_launches_scatter() -> None:
    """Terminal native telemetry must equal the canonical packed byte count."""
    worker_type = _production_worker_methods(
        "_poll_coalesced_plans",
        *_SAFE_FAILURE_METHODS,
    )
    allocator, plan = _plan(statuses=("DONE",))
    worker = _worker(worker_type, allocator, plan)
    worker._coalesced_scatter = MagicMock()
    worker._poll_pending_coalesced_scatters = MagicMock(return_value=set())
    worker.nixl_wrapper.get_xfer_telemetry.return_value = _native_telemetry(
        plan.layout.staging_size_bytes - 1
    )

    assert worker._poll_coalesced_plans() == set()

    assert plan.operation_failed
    assert plan.released
    worker._coalesced_scatter.assert_not_called()
    worker._record_failed_receive.assert_called_once()


def test_failed_receive_while_scatter_pending_defers_range_resolution() -> None:
    """A real request failure cannot reclaim staging ahead of its event."""
    worker_type = _production_worker_methods(
        "_discard_completed_coalesced_plan",
        "_resolve_failed_coalesced_plan",
    )
    allocator, plan = _ready_plan()
    plan.begin_device_read()
    worker = _worker(worker_type, allocator, plan)
    worker._pending_coalesced_scatters[plan.request_id] = (
        worker_type._pending_scatter_type(
            plan=plan,
            launch=_FakeScatterLaunch(complete=False),
            enqueued_at=0.0,
        )
    )
    worker._release_quiescent_failed_coalesced_plan = MagicMock()

    assert worker._resolve_failed_coalesced_plan(plan.request_id) is False

    assert allocator.require_active(plan.lease.generation) is plan
    assert plan.request_id in worker._pending_coalesced_scatters
    worker._release_quiescent_failed_coalesced_plan.assert_not_called()


def test_done_native_handle_release_failure_retains_generation() -> None:
    """A failed native resource release prevents scatter and range reuse."""
    worker_type = _production_worker_methods(
        "_fail_coalesced_plan",
        "_poll_coalesced_plans",
    )
    allocator, plan = _plan(statuses=("DONE",))
    native_handle = plan.slots[0].native_handle
    worker = _worker(worker_type, allocator, plan)
    worker._poll_pending_coalesced_scatters = MagicMock(return_value=set())
    worker.nixl_wrapper.get_xfer_telemetry.return_value = _native_telemetry(
        plan.layout.staging_size_bytes
    )
    worker.nixl_wrapper.release_xfer_handle.side_effect = RuntimeError(
        "release exploded"
    )

    with pytest.raises(StagingSafetyError, match="safety proof failed"):
        worker._poll_coalesced_plans()

    assert allocator.require_active(plan.lease.generation) is plan
    assert plan.operation_failed
    assert plan.slots[0].state is HandleState.DONE
    assert plan.slots[0].native_handle is native_handle
    assert plan.slots[0].native_released is False


def test_staging_range_release_failure_never_reports_publication() -> None:
    """Allocator release failure tombstones the exact completion owner."""
    launch = _FakeScatterLaunch(complete=True)
    worker_type = _production_worker_methods(
        "_fail_coalesced_plan",
        "_poll_pending_coalesced_scatters",
        "_release_coalesced_plan",
    )
    allocator, plan = _ready_plan()
    plan.begin_device_read()
    worker = _worker(worker_type, allocator, plan)
    worker._pending_coalesced_scatters[plan.request_id] = (
        worker_type._pending_scatter_type(
            plan=plan,
            launch=launch,
            enqueued_at=0.0,
        )
    )
    allocator.release = MagicMock(
        side_effect=StagingSafetyError("allocator release exploded")
    )

    with pytest.raises(StagingSafetyError, match="safety proof failed"):
        worker._poll_pending_coalesced_scatters()

    assert allocator.require_active(plan.lease.generation) is plan
    assert plan.released is False
    assert plan.device_quiescent
    assert plan.permanently_tombstoned
    assert plan.request_id in worker._coalesce_plans
    assert plan.request_id in worker._pending_coalesced_scatters
    worker.xfer_stats.record_coalesced_plan.assert_not_called()


def test_pending_scatters_complete_in_generation_order() -> None:
    """Dictionary insertion order cannot reorder completion ownership."""
    worker_type = _production_worker_methods(
        "_poll_pending_coalesced_scatters",
        "_release_coalesced_plan",
    )
    allocator = StagingRangeAllocator(4096)
    _, first = _ready_plan(
        allocator=allocator,
        owner_id="owner-0",
        request_id="request-0",
    )
    _, second = _ready_plan(
        allocator=allocator,
        owner_id="owner-1",
        request_id="request-1",
    )
    first.begin_device_read()
    second.begin_device_read()
    release_order: list[int] = []
    original_release = allocator.release

    def release(plan: CoalescedStagingPlan) -> None:
        release_order.append(plan.lease.generation)
        original_release(plan)

    allocator.release = release  # type: ignore[method-assign]
    worker = _worker(worker_type, allocator, first, second)
    worker._pending_coalesced_scatters = {
        second.request_id: worker_type._pending_scatter_type(
            plan=second,
            launch=_FakeScatterLaunch(complete=True),
            enqueued_at=0.0,
        ),
        first.request_id: worker_type._pending_scatter_type(
            plan=first,
            launch=_FakeScatterLaunch(complete=True),
            enqueued_at=0.0,
        ),
    }

    assert worker._poll_pending_coalesced_scatters() == {
        first.request_id,
        second.request_id,
    }
    assert release_order == [first.lease.generation, second.lease.generation]


def test_unresolved_shutdown_and_cleanup_touch_no_native_resources() -> None:
    """Teardown refuses every resource mutation while staging is live."""
    worker_type = _production_worker_methods("_cleanup_remote_engine", "shutdown")
    allocator, plan = _plan(
        statuses=("PROC",),
        remote_engine_id="producer",
    )
    worker = _worker(worker_type, allocator, plan)
    worker._remote_agents = {"producer": {0: "agent"}}
    worker._handshake_initiation_executor = MagicMock()
    worker._raise_if_handshake_fail_stopped = MagicMock()

    with pytest.raises(StagingSafetyError, match="live transfer resources"):
        worker._cleanup_remote_engine("producer")
    with pytest.raises(StagingSafetyError, match="cannot quiesce"):
        worker.shutdown()

    worker.nixl_wrapper.release_xfer_handle.assert_not_called()
    worker.nixl_wrapper.release_dlist_handle.assert_not_called()
    worker.nixl_wrapper.remove_remote_agent.assert_not_called()
    worker.nixl_wrapper.deregister_memory.assert_not_called()
    worker._handshake_initiation_executor.shutdown.assert_not_called()


def test_quiescent_shutdown_closes_and_releases_every_resource_once() -> None:
    """A quiescent worker performs the complete production teardown."""
    worker_type = _production_worker_methods("_cleanup_remote_engine", "shutdown")
    worker = worker_type()
    worker._coalesce_plans = {}
    worker._coalesced_localization_plans = {}
    worker._localization_pre_read_plans = {}
    worker._packed_remote_engine_active = MagicMock(return_value=False)
    worker._packed_remote_engine_cleanup = MagicMock()
    worker._packed_registered_ownership_active = MagicMock(return_value=False)
    worker._packed_shutdown_cleanup = MagicMock()
    worker._clear_remote_packed_write_pools = MagicMock()
    worker._localization_writer = MagicMock()
    writer = worker._localization_writer
    worker._handshake_initiation_executor = MagicMock()
    worker._raise_if_handshake_fail_stopped = MagicMock()
    worker._recving_transfers = {"request": ["receive-handle"]}
    worker.src_xfer_handles_by_block_size = {8: "source-block-handle"}
    worker.src_xfer_handles_by_tp_ratio = {4: ["source-ratio-handle"]}
    worker._remote_agents = {"producer": {0: "remote-agent"}}
    worker.dst_xfer_side_handles = {"producer": {0: "destination-handle"}}
    worker.kv_caches_base_addr = {"producer": object()}
    worker.dst_num_blocks = {"producer": object()}
    worker.tp_mappings = {"producer": object()}
    worker._remote_layout = {"producer": object()}
    worker._remote_regions = {"producer": object()}
    worker._remote_registration_generations = {"producer": object()}
    worker._remote_source_semantics = {"producer": object()}
    worker._remote_rank_contracts = {"producer": object()}
    worker._remote_packed_write_producer_pools = {"producer": object()}
    worker._remote_packed_write_consumer_pools = {"producer": object()}
    worker.transfer_topo = MagicMock()
    worker._engine_last_active = {"producer": 0.0}
    worker._registered_descs = ["registered-memory"]
    worker.nixl_wrapper = MagicMock()

    worker.shutdown()

    writer.close.assert_called_once()
    assert worker._localization_writer is None
    worker._handshake_initiation_executor.shutdown.assert_called_once_with(wait=False)
    worker.nixl_wrapper.release_xfer_handle.assert_called_once_with("receive-handle")
    assert worker.nixl_wrapper.release_dlist_handle.call_count == 3
    worker.nixl_wrapper.remove_remote_agent.assert_called_once_with("remote-agent")
    worker.nixl_wrapper.deregister_memory.assert_called_once_with("registered-memory")
    worker.transfer_topo.unregister_remote_engine.assert_called_once_with("producer")
    assert "producer" not in worker._remote_rank_contracts
    assert worker._registered_descs == []


def test_plan_owns_the_exact_typed_layout_and_remote_identity() -> None:
    """Ownership retains canonical geometry rather than a mutable scatter dict."""
    allocator = StagingRangeAllocator(4096)
    layout = _layout(1)
    plan = allocator.create_plan(
        owner_id="identity-owner",
        request_id="identity-request",
        layout=layout,
        layout_duration_seconds=0.25,
        remote_engine_id="authoritative-producer",
    )
    assert plan is not None

    assert plan.remote_engine_id == "authoritative-producer"
    assert plan.layout is layout
    assert plan.layout.digest == layout.digest
    assert plan.lease.size == layout.staging_size_bytes


def test_ownership_snapshot_has_typed_json_compatible_evidence() -> None:
    """Snapshots include canonical layout and terminal telemetry evidence."""
    _, plan = _plan(statuses=("DONE",), created_at=1.0)
    plan.record_native_telemetry(
        0,
        _native_telemetry(plan.layout.staging_size_bytes),
    )
    plan.mark_native_released(0)

    snapshot = plan.snapshot(now=3.5)
    assert isinstance(snapshot, StagingOwnershipSnapshot)
    assert snapshot.generation == plan.lease.generation
    assert snapshot.age_s == 2.5
    assert snapshot.layout_digest == plan.layout.digest
    assert snapshot.handles == (
        StagingHandleSnapshot(
            source_rank=0,
            state=HandleState.DONE,
            native_handle_present=True,
            native_released=True,
            terminal_telemetry_present=True,
        ),
    )
    assert snapshot.to_dict() == {
        "generation": plan.lease.generation,
        "owner_id": "test-owner",
        "request_id": "test-request",
        "remote_engine_id": "test-producer",
        "layout_digest": plan.layout.digest,
        "offset": 0,
        "size": plan.layout.staging_size_bytes,
        "age_s": 2.5,
        "handles": [
            {
                "source_rank": 0,
                "state": "done",
                "native_handle_present": True,
                "native_released": True,
                "terminal_telemetry_present": True,
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
