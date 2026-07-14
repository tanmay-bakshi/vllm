# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import queue
from collections import deque
from concurrent.futures import Future
from threading import Lock
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm.v1.engine.core import DPEngineCoreProc, EngineShutdownState
from vllm.v1.engine.core_client import (
    _COORDINATOR_CONTROL_ACK,
    AsyncMPClient,
    DPAsyncMPClient,
    InprocClient,
    SyncMPClient,
    _fail_utility_results,
)
from vllm.v1.engine.exceptions import EngineDeadError


@pytest.mark.asyncio
async def test_output_teardown_fails_every_registered_waiter() -> None:
    """Terminal output failure resolves both async and sync waiters."""
    error = RuntimeError("output path failed")
    async_waiter = asyncio.get_running_loop().create_future()
    sync_waiter: Future[object] = Future()
    utility_results = {1: async_waiter, 2: sync_waiter}

    _fail_utility_results(utility_results, Lock(), error)

    assert utility_results == {}
    with pytest.raises(RuntimeError, match="output path failed"):
        await async_waiter
    with pytest.raises(RuntimeError, match="output path failed"):
        sync_waiter.result()


@pytest.mark.asyncio
async def test_terminal_output_error_prevents_new_waiter_registration() -> None:
    """No waiter can enter the registry after output teardown begins."""
    error = RuntimeError("output path failed")
    client = object.__new__(AsyncMPClient)
    client.resources = SimpleNamespace(
        output_error=error,
        utility_results_lock=Lock(),
    )
    client.utility_results = {}
    future = asyncio.get_running_loop().create_future()

    with pytest.raises(RuntimeError, match="output path failed") as caught:
        client._register_utility_result(1, future)

    assert caught.value is error
    assert client.utility_results == {}


@pytest.mark.asyncio
async def test_completed_output_task_is_not_treated_as_live() -> None:
    """A completed output task fails immediately instead of orphaning waiters."""
    output_task = asyncio.create_task(asyncio.sleep(0))
    await output_task
    client = object.__new__(AsyncMPClient)
    client.resources = SimpleNamespace(
        output_error=None,
        output_queue_task=output_task,
        utility_results_lock=Lock(),
    )

    with pytest.raises(EngineDeadError):
        client._ensure_output_queue_task()


def test_sync_utility_send_failure_discards_waiter() -> None:
    """A failed utility send cannot leave a permanently blocked future."""
    client = object.__new__(SyncMPClient)
    client.resources = SimpleNamespace(
        output_error=None,
        utility_results_lock=Lock(),
    )
    client.utility_results = {}
    client._send_input = MagicMock(side_effect=RuntimeError("send failed"))

    with pytest.raises(RuntimeError, match="send failed"):
        client.call_utility("method")

    assert client.utility_results == {}


def test_inproc_rejection_notification_resolves_worker_future() -> None:
    """The synchronous in-process client exposes a real boolean result."""
    result_future: Future[bool] = Future()
    result_future.set_result(True)
    client = object.__new__(InprocClient)
    client.engine_core = MagicMock()
    client.engine_core.notify_kv_transfer_request_rejected.return_value = result_future

    handled = client.notify_kv_transfer_request_rejected(
        "request",
        {"remote_request_id": "producer-request"},
        "frontend rejected request",
    )

    assert handled is True


def test_dp_committed_work_stays_dormant_until_wave_start() -> None:
    """A coordinator-owned DP rank cannot step before its start signal."""
    engine_core = object.__new__(DPEngineCoreProc)
    engine_core.has_coordinator = True
    engine_core.engines_running = False
    engine_core.batch_queue = None
    engine_core.scheduler = MagicMock()
    engine_core.scheduler.has_requests.return_value = True
    engine_core.scheduler.has_maintenance_work.return_value = False
    engine_core.shutdown_state = EngineShutdownState.SHUTTING_DOWN

    assert engine_core.has_work() is True
    assert engine_core.has_step_work() is False
    assert engine_core._handle_shutdown() is True

    engine_core.batch_queue = deque([(None, None)])

    assert engine_core.has_step_work() is True


def test_dp_maintenance_work_blocks_shutdown_and_runs_without_step_fn() -> None:
    """Connector maintenance runs locally and remains teardown-owned work."""
    engine_core = object.__new__(DPEngineCoreProc)
    engine_core.has_coordinator = True
    engine_core.engines_running = False
    engine_core.batch_queue = None
    engine_core.scheduler = MagicMock()
    engine_core.scheduler.has_requests.return_value = True
    engine_core.scheduler.has_maintenance_work.return_value = True
    engine_core.shutdown_state = EngineShutdownState.SHUTTING_DOWN
    engine_core.step = MagicMock(return_value=({}, False))
    engine_core.step_fn = MagicMock()
    engine_core.output_queue = queue.Queue()
    engine_core.post_step = MagicMock()

    assert engine_core.has_step_work() is True
    assert engine_core._is_maintenance_only_iteration() is True
    assert engine_core._handle_shutdown() is True
    assert engine_core._process_engine_step(maintenance_only=True) is False
    engine_core.step.assert_called_once_with(maintenance_only=True)
    engine_core.step_fn.assert_not_called()


def test_connector_rejection_skips_data_parallel_admission_hook() -> None:
    """Rejected connector admission cannot mutate DP wave bookkeeping."""
    engine_core = object.__new__(DPEngineCoreProc)
    engine_core.scheduler = MagicMock()
    engine_core.scheduler.admit_committed_requests.return_value = False
    engine_core._request_added = MagicMock()
    prepared_request = MagicMock()

    admitted = engine_core.admit_committed_requests([(prepared_request, 3)])

    assert admitted is False
    engine_core._request_added.assert_not_called()


@pytest.mark.asyncio
async def test_completed_stats_task_is_not_treated_as_live() -> None:
    """A failed coordinator-control task is raised instead of reused."""
    error = RuntimeError("stats task failed")

    async def fail() -> None:
        raise error

    stats_update_task = asyncio.create_task(fail())
    with pytest.raises(RuntimeError, match="stats task failed"):
        await stats_update_task

    client = object.__new__(DPAsyncMPClient)
    client.resources = SimpleNamespace(
        stats_update_task=stats_update_task,
        stats_update_error=None,
        utility_results_lock=Lock(),
    )

    with pytest.raises(RuntimeError, match="stats task failed") as caught:
        client._ensure_stats_update_task()

    assert caught.value is error


@pytest.mark.asyncio
async def test_coordinator_control_send_races_stats_task_failure() -> None:
    """A blocked PAIR send cannot hide failure of its consuming task."""

    class BlockingControlSocket:
        """PAIR socket stub whose send remains blocked until cancelled."""

        def __init__(self) -> None:
            self.send_started = asyncio.Event()
            self.send_cancelled = False

        async def send(self, _message: bytes) -> None:
            self.send_started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                self.send_cancelled = True
                raise

        async def recv(self) -> bytes:
            return _COORDINATOR_CONTROL_ACK

    socket = BlockingControlSocket()
    error = RuntimeError("stats task failed")

    async def fail_after_send_starts() -> None:
        await socket.send_started.wait()
        raise error

    stats_update_task = asyncio.create_task(fail_after_send_starts())
    client = object.__new__(DPAsyncMPClient)
    client.resources = SimpleNamespace(
        stats_update_task=stats_update_task,
        stats_update_error=None,
        utility_results_lock=Lock(),
    )
    client._coordinator_control_lock = asyncio.Lock()
    client.first_req_send_socket = socket

    with pytest.raises(RuntimeError, match="stats task failed") as caught:
        await client._send_coordinator_control(b"FIRST_REQ")

    assert caught.value is error
    assert socket.send_cancelled is True


@pytest.mark.asyncio
async def test_stats_failure_preserves_best_effort_abort_eligibility() -> None:
    """Coordinator death does not falsely mark a reachable Core as dead."""
    error = RuntimeError("coordinator control failed")
    waiter = asyncio.get_running_loop().create_future()
    client = object.__new__(DPAsyncMPClient)
    client.utility_results = {1: waiter}
    client.outputs_queue = asyncio.Queue()
    client.resources = SimpleNamespace(
        engine_dead=False,
        output_error=None,
        stats_update_error=None,
        utility_results_lock=Lock(),
    )

    client._latch_stats_update_error(error)

    assert client.resources.engine_dead is False
    assert client.resources.output_error is error
    assert client.resources.stats_update_error is error
    assert await client.outputs_queue.get() is error
    with pytest.raises(RuntimeError, match="coordinator control failed"):
        await waiter
