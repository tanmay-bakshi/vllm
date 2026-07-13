# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Source-bound tests for transfer/decode phase separation."""

import ast
import json
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

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


def _production_method_node(
    path: Path,
    class_name: str,
    method_name: str,
) -> ast.FunctionDef | ast.AsyncFunctionDef:
    """Find one exact production method in a source file.

    :param path: Python source file.
    :param class_name: Containing class name.
    :param method_name: Method to resolve.
    :returns: Exact parsed method node.
    """
    syntax = ast.parse(path.read_text(), filename=str(path))
    class_node = next(
        node
        for node in syntax.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return next(
        node
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == method_name
    )


def _attribute_call_lines(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    attribute: str,
) -> list[int]:
    """Return source lines calling one attribute.

    :param node: Parsed production method.
    :param attribute: Called attribute name.
    :returns: Sorted source line numbers.
    """
    return sorted(
        call.lineno
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == attribute
    )


class _StagingSafetyError(RuntimeError):
    """Test substitute for the production fail-stop error."""


class _PhasePlan:
    """Minimal plan whose state is advanced by the completion substitute."""

    state: str

    def __init__(self, state: str) -> None:
        """Create a plan in one explicit native-handle state.

        :param state: Initial native-handle state.
        """
        self.state = state


def _production_worker_methods(
    *method_names: str,
    namespace_overrides: dict[str, object] | None = None,
) -> type:
    """Compile exact production worker methods without importing CUDA modules.

    :param method_names: Production methods to bind to the test class.
    :param namespace_overrides: Globals required by selected method bodies.
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
        raise AssertionError("production phase-separation method is missing")
    test_class = ast.ClassDef(
        name="ProductionWorkerMethods",
        bases=[],
        keywords=[],
        body=list(selected),
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[test_class], type_ignores=[]))
    namespace: dict[str, object] = {
        "Any": Any,
        "StagingSafetyError": _StagingSafetyError,
        "json": json,
        "logger": MagicMock(),
        "time": time,
        "torch": SimpleNamespace(accelerator=MagicMock()),
        "traceback": traceback,
    }
    if namespace_overrides is not None:
        namespace.update(namespace_overrides)
    exec(compile(module, str(BASE_WORKER_PATH), "exec"), namespace)
    return namespace["ProductionWorkerMethods"]  # type: ignore[return-value]


def test_start_load_brackets_every_possible_transfer_post() -> None:
    """The pull entrypoint must bracket all request and heartbeat processing."""
    method = _production_method_node(
        PULL_WORKER_PATH,
        "NixlPullConnectorWorker",
        "start_load_kv",
    )
    begin_lines = _attribute_call_lines(method, "_begin_transfer_phase")
    drain_lines = _attribute_call_lines(method, "_drain_transfer_phase")
    read_lines = _attribute_call_lines(method, "_read_blocks_for_req")
    heartbeat_lines = _attribute_call_lines(method, "_send_heartbeats")

    assert len(begin_lines) == 1
    assert len(drain_lines) == 1
    assert begin_lines[0] < min(read_lines)
    assert drain_lines[0] > max(read_lines)
    assert drain_lines[0] > heartbeat_lines[0]


def test_native_posts_are_guarded_before_entering_nixl() -> None:
    """Every pull transfer post must prove that transfer phase is active."""
    coalesced = _production_method_node(
        BASE_WORKER_PATH,
        "NixlBaseConnectorWorker",
        "_initialize_and_post_coalesced",
    )
    stock = _production_method_node(
        PULL_WORKER_PATH,
        "NixlPullConnectorWorker",
        "_read_blocks",
    )

    for method in (coalesced, stock):
        guard_lines = _attribute_call_lines(method, "_assert_transfer_post_allowed")
        transfer_lines = _attribute_call_lines(method, "transfer")
        assert len(guard_lines) == 1
        assert len(transfer_lines) == 1
        assert guard_lines[0] < transfer_lines[0]


def test_staging_allocation_has_no_initialization_writer() -> None:
    """Registered staging must have no asynchronous writer before NIXL."""
    method = _production_method_node(
        BASE_WORKER_PATH,
        "NixlBaseConnectorWorker",
        "_staging_init",
    )
    empty_lines = _attribute_call_lines(method, "empty")
    zero_lines = _attribute_call_lines(method, "zeros")
    register_lines = _attribute_call_lines(method, "register_memory")

    assert len(empty_lines) == 1
    assert zero_lines == []
    assert len(register_lines) == 1
    assert empty_lines[0] < register_lines[0]


@pytest.mark.parametrize("value", [False, True])
def test_phase_separation_config_accepts_exact_booleans(value: bool) -> None:
    """Boolean connector configuration must round-trip unchanged."""
    worker_type = _production_worker_methods("_parse_phase_separation_config")

    assert worker_type._parse_phase_separation_config(value) is value


@pytest.mark.parametrize("value", [None, 0, 1, "false", "true", [], {}])
def test_phase_separation_config_rejects_non_booleans(value: object) -> None:
    """Truthy lookalikes must not silently activate an evidentiary arm."""
    worker_type = _production_worker_methods("_parse_phase_separation_config")

    with pytest.raises(ValueError, match="must be a boolean"):
        worker_type._parse_phase_separation_config(value)


def test_begin_transfer_phase_resets_one_owned_epoch() -> None:
    """A new separated epoch must start from a published, inactive boundary."""
    worker_type = _production_worker_methods("_begin_transfer_phase")
    worker = worker_type()
    worker._phase_separate_transfer_decode = True
    worker._transfer_phase_active = False
    worker._transfer_phase_epoch = 4
    worker._deferred_phase_sending = set()
    worker._deferred_phase_recving = set()
    worker._coalesce_plans = {}
    worker._coalesce_pending = []
    worker._recving_transfers = {}
    worker._transfer_phase_plan_count = 11
    worker._transfer_phase_handle_count = 12
    worker._transfer_phase_byte_count = 13
    worker._transfer_phase_violation_count = 14
    worker._transfer_phase_started_ns = 0
    worker._transfer_phase_entry_sync_ns = 0

    worker._begin_transfer_phase()

    assert worker._transfer_phase_active is True
    assert worker._transfer_phase_epoch == 5
    assert worker._transfer_phase_plan_count == 0
    assert worker._transfer_phase_handle_count == 0
    assert worker._transfer_phase_byte_count == 0
    assert worker._transfer_phase_violation_count == 0
    assert worker._transfer_phase_started_ns > 0


def test_transfer_post_guard_rejects_compute_phase() -> None:
    """No native writer may start after the compute boundary opens."""
    worker_type = _production_worker_methods("_assert_transfer_post_allowed")
    worker = worker_type()
    worker._phase_separate_transfer_decode = True
    worker._transfer_phase_active = False
    worker._transfer_phase_violation_count = 0

    with pytest.raises(_StagingSafetyError, match="transfer phase"):
        worker._assert_transfer_post_allowed(coalesced=True)

    assert worker._transfer_phase_violation_count == 1


@pytest.mark.parametrize(
    ("enabled", "active", "coalesced"),
    [(False, False, False), (False, True, True), (True, True, True)],
)
def test_transfer_post_guard_allows_only_valid_boundaries(
    enabled: bool,
    active: bool,
    coalesced: bool,
) -> None:
    """Disabled mode and an active transfer phase preserve valid posts."""
    worker_type = _production_worker_methods("_assert_transfer_post_allowed")
    worker = worker_type()
    worker._phase_separate_transfer_decode = enabled
    worker._transfer_phase_active = active
    worker._transfer_phase_violation_count = 0
    worker._transfer_phase_handle_count = 0

    worker._assert_transfer_post_allowed(coalesced=coalesced)

    assert worker._transfer_phase_violation_count == 0
    assert worker._transfer_phase_handle_count == int(enabled)


def test_transfer_post_guard_rejects_stock_path() -> None:
    """Separated mode must reject posts without staging-plan ownership."""
    worker_type = _production_worker_methods("_assert_transfer_post_allowed")
    worker = worker_type()
    worker._phase_separate_transfer_decode = True
    worker._transfer_phase_active = True
    worker._transfer_phase_violation_count = 0

    with pytest.raises(_StagingSafetyError, match="owned transfer phase"):
        worker._assert_transfer_post_allowed(coalesced=False)

    assert worker._transfer_phase_violation_count == 1


def test_drain_collects_proc_to_done_before_compute_boundary() -> None:
    """PROC work must reach DONE, scatter, and synchronize before compute."""
    accelerator = MagicMock()
    worker_type = _production_worker_methods(
        "_drain_transfer_phase",
        namespace_overrides={
            "torch": SimpleNamespace(accelerator=accelerator),
        },
    )
    worker = worker_type()
    plan = _PhasePlan("PROC")
    worker._phase_separate_transfer_decode = True
    worker._transfer_phase_active = True
    worker._deferred_phase_sending = set()
    worker._deferred_phase_recving = set()
    worker._coalesce_plans = {"request": plan}
    worker._coalesce_pending = []
    worker._recving_transfers = {}
    worker._transfer_phase_violation_count = 0
    worker._transfer_phase_plan_count = 1
    worker._transfer_phase_handle_count = 1
    worker._transfer_phase_byte_count = 4096
    worker._transfer_phase_started_ns = time.monotonic_ns()
    worker._transfer_phase_entry_sync_ns = 17
    worker._transfer_phase_epoch = 3
    worker._transfer_phase_records = []
    worker.engine_id = "decoder"
    worker.tp_rank = 2
    synchronized_while_active: list[bool] = []

    def collect(*, service_pending: bool) -> tuple[set[str], set[str]]:
        """Model one authoritative completion and its synchronous scatter.

        :param service_pending: Whether pending work may be posted.
        :returns: Completion identifiers accumulated by the drain.
        """
        assert service_pending is True
        assert worker._transfer_phase_active is True
        assert plan.state == "PROC"
        plan.state = "DONE"
        worker._coalesce_plans.clear()
        return {"producer"}, {"request"}

    def synchronize() -> None:
        """Record whether compute was still closed at the exit fence."""
        synchronized_while_active.append(worker._transfer_phase_active)

    worker._get_finished = MagicMock(side_effect=collect)
    accelerator.synchronize.side_effect = synchronize

    worker._drain_transfer_phase()

    assert plan.state == "DONE"
    assert worker._deferred_phase_sending == {"producer"}
    assert worker._deferred_phase_recving == {"request"}
    assert worker._transfer_phase_active is False
    worker._get_finished.assert_called_once_with(service_pending=True)
    accelerator.synchronize.assert_called_once_with()
    assert synchronized_while_active == [True]
    assert len(worker._transfer_phase_records) == 1
    record_type, marker = worker._transfer_phase_records[0]
    assert record_type == "transfer-decode-phase"
    assert marker["event"] == "compute_boundary"
    assert marker["active_plan_count"] == 0
    assert marker["pending_request_count"] == 0
    assert marker["stock_handle_count"] == 0


def test_drain_serializes_parked_work_inside_transfer_phase() -> None:
    """A freed staging lease must launch and finish the next parked request."""
    accelerator = MagicMock()
    worker_type = _production_worker_methods(
        "_drain_transfer_phase",
        namespace_overrides={
            "torch": SimpleNamespace(accelerator=accelerator),
        },
    )
    worker = worker_type()
    plans = {
        "first": _PhasePlan("PROC"),
        "second": _PhasePlan("PENDING"),
    }
    worker._phase_separate_transfer_decode = True
    worker._transfer_phase_active = True
    worker._deferred_phase_sending = set()
    worker._deferred_phase_recving = set()
    worker._coalesce_plans = {"first": plans["first"]}
    worker._coalesce_pending = ["second"]
    worker._recving_transfers = {}
    worker._transfer_phase_violation_count = 0
    worker._transfer_phase_plan_count = 2
    worker._transfer_phase_handle_count = 2
    worker._transfer_phase_byte_count = 8192
    worker._transfer_phase_started_ns = time.monotonic_ns()
    worker._transfer_phase_entry_sync_ns = 19
    worker._transfer_phase_epoch = 4
    worker._transfer_phase_records = []
    worker.engine_id = "decoder"
    worker.tp_rank = 0
    completion_order: list[str] = []

    def collect(*, service_pending: bool) -> tuple[set[str], set[str]]:
        """Complete one owner and immediately service one parked request.

        :param service_pending: Whether pending work may be posted.
        :returns: Completion identifiers accumulated by the drain.
        """
        assert service_pending is True
        assert worker._transfer_phase_active is True
        request_id = next(iter(worker._coalesce_plans))
        plan = worker._coalesce_plans.pop(request_id)
        assert plan.state == "PROC"
        plan.state = "DONE"
        completion_order.append(request_id)
        if len(worker._coalesce_pending) > 0:
            pending_request_id = worker._coalesce_pending.pop(0)
            pending_plan = plans[pending_request_id]
            pending_plan.state = "PROC"
            worker._coalesce_plans[pending_request_id] = pending_plan
        return set(), {request_id}

    worker._get_finished = MagicMock(side_effect=collect)

    worker._drain_transfer_phase()

    assert completion_order == ["first", "second"]
    assert plans["first"].state == "DONE"
    assert plans["second"].state == "DONE"
    assert worker._deferred_phase_recving == {"first", "second"}
    assert worker._coalesce_plans == {}
    assert worker._coalesce_pending == []
    assert worker._transfer_phase_active is False
    assert [call.kwargs for call in worker._get_finished.call_args_list] == [
        {"service_pending": True},
        {"service_pending": True},
    ]
    accelerator.synchronize.assert_called_once_with()


def test_drain_exit_sync_failure_keeps_compute_closed() -> None:
    """A failed exit fence must leave the transfer phase active and fail stop."""
    accelerator = MagicMock()
    accelerator.synchronize.side_effect = RuntimeError("device unavailable")
    logger = MagicMock()
    worker_type = _production_worker_methods(
        "_drain_transfer_phase",
        namespace_overrides={
            "logger": logger,
            "torch": SimpleNamespace(accelerator=accelerator),
        },
    )
    worker = worker_type()
    worker._phase_separate_transfer_decode = True
    worker._transfer_phase_active = True
    worker._deferred_phase_sending = set()
    worker._deferred_phase_recving = set()
    worker._coalesce_plans = {}
    worker._coalesce_pending = []
    worker._recving_transfers = {}
    worker._transfer_phase_violation_count = 0
    worker._get_finished = MagicMock(return_value=(set(), set()))

    with pytest.raises(_StagingSafetyError, match="synchronization failed after"):
        worker._drain_transfer_phase()

    assert worker._transfer_phase_active is True
    assert worker._transfer_phase_violation_count == 1
    worker._get_finished.assert_called_once_with(service_pending=True)
    accelerator.synchronize.assert_called_once_with()
    logger.info.assert_not_called()


def test_drain_rejects_stock_handle_without_opening_compute() -> None:
    """Any stock receive owner must fail before collection or exit fencing."""
    accelerator = MagicMock()
    worker_type = _production_worker_methods(
        "_drain_transfer_phase",
        namespace_overrides={
            "torch": SimpleNamespace(accelerator=accelerator),
        },
    )
    worker = worker_type()
    worker._phase_separate_transfer_decode = True
    worker._transfer_phase_active = True
    worker._deferred_phase_sending = set()
    worker._deferred_phase_recving = set()
    worker._coalesce_plans = {}
    worker._coalesce_pending = []
    worker._recving_transfers = {"request": [object()]}
    worker._transfer_phase_violation_count = 0
    worker._get_finished = MagicMock(return_value=(set(), set()))

    with pytest.raises(_StagingSafetyError, match="stock receive handle"):
        worker._drain_transfer_phase()

    assert worker._transfer_phase_active is True
    assert worker._transfer_phase_violation_count == 1
    worker._get_finished.assert_not_called()
    accelerator.synchronize.assert_not_called()


def test_disabled_drain_preserves_nonblocking_boundary() -> None:
    """Disabled mode must neither collect synchronously nor fence the device."""
    accelerator = MagicMock()
    worker_type = _production_worker_methods(
        "_drain_transfer_phase",
        namespace_overrides={
            "torch": SimpleNamespace(accelerator=accelerator),
        },
    )
    worker = worker_type()
    worker._phase_separate_transfer_decode = False
    worker._get_finished = MagicMock(return_value=(set(), set()))

    worker._drain_transfer_phase()

    worker._get_finished.assert_not_called()
    accelerator.synchronize.assert_not_called()


def test_disabled_get_finished_preserves_pending_service() -> None:
    """The disabled public boundary must retain the asynchronous fast path."""
    worker_type = _production_worker_methods("get_finished")
    worker = worker_type()
    worker._phase_separate_transfer_decode = False
    worker._phase_separation_instrumented = False
    worker._get_finished = MagicMock(return_value=({"sent"}, {"received"}))

    assert worker.get_finished() == ({"sent"}, {"received"})
    worker._get_finished.assert_called_once_with(service_pending=True)


def test_separated_get_finished_publishes_deferred_completions_once() -> None:
    """The scheduler boundary must publish drained completions exactly once."""
    worker_type = _production_worker_methods("get_finished")
    worker = worker_type()
    worker._phase_separate_transfer_decode = True
    worker._phase_separation_instrumented = False
    worker._transfer_phase_active = False
    worker._deferred_phase_sending = {"sent-during-drain"}
    worker._deferred_phase_recving = {"received-during-drain"}
    worker._coalesce_plans = {}
    worker._coalesce_pending = []
    worker._recving_transfers = {}
    worker._get_finished = MagicMock(
        side_effect=[
            ({"sent-at-boundary"}, {"received-at-boundary"}),
            (set(), set()),
        ]
    )

    assert worker.get_finished() == (
        {"sent-during-drain", "sent-at-boundary"},
        {"received-during-drain", "received-at-boundary"},
    )
    assert worker._deferred_phase_sending == set()
    assert worker._deferred_phase_recving == set()
    assert worker.get_finished() == (set(), set())
    assert [item.args for item in worker._get_finished.call_args_list] == [(), ()]
    assert [item.kwargs for item in worker._get_finished.call_args_list] == [
        {"service_pending": False},
        {"service_pending": False},
    ]


def test_control_snapshot_is_buffered_until_after_model_execution() -> None:
    """Control-arm logging must not perturb the transfer/compute boundary."""
    logger = MagicMock()
    handle_state = SimpleNamespace(
        POSTING="posting",
        PROC="proc",
        UNKNOWN="unknown",
        ERR="err",
    )
    worker_type = _production_worker_methods(
        "_record_transfer_decode_boundary",
        "_flush_transfer_phase_records",
        "get_finished",
        namespace_overrides={
            "HandleState": handle_state,
            "logger": logger,
        },
    )
    worker = worker_type()
    worker._phase_separation_instrumented = True
    worker._phase_separate_transfer_decode = False
    worker._transfer_phase_records = []
    worker._coalesce_plans = {
        "request": SimpleNamespace(
            lease=SimpleNamespace(size=4096),
            slots={0: SimpleNamespace(state=handle_state.PROC)},
        )
    }
    worker._coalesce_pending = ["parked"]
    worker._recving_transfers = {}
    worker.engine_id = "decoder"
    worker.tp_rank = 0
    execution_order: list[str] = []

    def collect(*, service_pending: bool) -> tuple[set[str], set[str]]:
        """Record the post-forward collection point.

        :param service_pending: Whether pending work may be serviced.
        :returns: Empty completion sets.
        """
        assert service_pending is True
        execution_order.append("post_forward")
        return set(), set()

    def log_record(*args: object, **kwargs: object) -> None:
        """Record deferred evidence emission.

        :param args: Positional logger arguments.
        :param kwargs: Keyword logger arguments.
        """
        execution_order.append("log")

    worker._get_finished = MagicMock(side_effect=collect)
    logger.info.side_effect = log_record

    worker._record_transfer_decode_boundary()

    logger.info.assert_not_called()
    assert len(worker._transfer_phase_records) == 1
    _, record = worker._transfer_phase_records[0]
    assert record["potential_overlap"] is True
    assert record["potential_active_handle_count"] == 1
    assert worker.get_finished() == (set(), set())
    assert execution_order == ["post_forward", "log"]
    assert worker._transfer_phase_records == []
    emitted = json.loads(logger.info.call_args.args[2])
    assert emitted["event"] == "compute_boundary_snapshot"
