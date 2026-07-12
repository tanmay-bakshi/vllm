# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import queue
from unittest.mock import Mock

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.nixl import base_worker
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_worker import (
    NixlPullConnectorWorker,
)

pytestmark = pytest.mark.cpu_test


class _FakeNixl:
    def __init__(self):
        self.sent_notifications: list[bytes] = []

    def get_new_notifs(self):
        return {}

    def send_notif(self, agent, notif_msg):
        self.sent_notifications.append(notif_msg)


def _make_worker() -> NixlPullConnectorWorker:
    worker = object.__new__(NixlPullConnectorWorker)
    tracker_type = getattr(base_worker, "RequestExpiryTracker", dict)
    worker._reqs_to_send = tracker_type()
    worker._lease_extension = 100
    worker._reqs_to_process = set()
    worker._recving_metadata = {}
    worker._recving_transfers = {}
    worker._failed_recv_reqs = queue.Queue()
    worker._failed_recv_outcomes = queue.Queue()
    worker._failed_recv_pending = {}
    worker._completed_failed_recv_outcomes = queue.Queue()
    worker._invalid_block_ids_emitted = set()
    worker._invalid_block_ids = queue.Queue()
    worker._ready_requests = queue.Queue()
    worker._remote_agents = {"remote": {0: "agent"}}
    worker._released_rids = {}
    worker._rid_completion_counts = {}
    worker.consumer_notification_counts_by_req = {}
    worker._grace_frees = {}
    worker._coalesce_pending = []
    worker._has_mamba = False
    worker.use_host_buffer = False
    worker.use_mla = False
    worker.enable_heterogeneous_attn_post_process = False
    worker.transfer_topo = Mock()
    worker.tp_rank = 0
    worker.world_size = 1
    worker.nixl_wrapper = _FakeNixl()
    worker.xfer_stats = Mock()
    worker._audit_retire = Mock()
    worker._audit_tick = Mock()
    worker._send_heartbeats = Mock()
    return worker


def _register(worker, deadlines):
    metadata = NixlConnectorMetadata()
    metadata.reqs_in_batch = set(deadlines)
    metadata.reqs_to_send = deadlines
    worker.start_load_kv(metadata)


def test_renewed_head_cannot_hide_expired_request(
    monkeypatch: pytest.MonkeyPatch,
):
    clock = {"now": 0.0}
    monkeypatch.setattr(
        base_worker.time,
        "perf_counter",
        lambda: clock["now"],
    )
    monkeypatch.setenv("VLLM_GEMMA4_KV_FREE_GRACE_S", "0")
    worker = _make_worker()

    _register(
        worker,
        {
            "renewed-head": 10.0,
            "expired-later": 20.0,
            "far-future": 1000.0,
        },
    )
    clock["now"] = 25.0
    worker._handle_heartbeat("renewed-head,far-future,unknown-request")

    assert worker._reqs_to_send["renewed-head"] == 125.0
    assert worker._reqs_to_send["far-future"] == 1000.0
    assert "unknown-request" not in worker._reqs_to_send
    if hasattr(worker._reqs_to_send, "_heap"):
        heap_size = len(worker._reqs_to_send._heap)
        for _ in range(100):
            worker._handle_heartbeat("far-future")
        assert len(worker._reqs_to_send._heap) == heap_size
    done_sending, _ = worker.get_finished()

    assert "expired-later" in done_sending, (
        "renewing the insertion-order head hid an already-expired later "
        f"registration: remaining={dict(worker._reqs_to_send.items())}"
    )
    assert "expired-later" not in worker._reqs_to_send
    assert "renewed-head" in worker._reqs_to_send
    assert b"EXPIRED:expired-later" in worker.nixl_wrapper.sent_notifications

    worker._reqs_to_send.pop("renewed-head")
    _register(worker, {"renewed-head": 200.0})

    clock["now"] = 125.0
    done_sending, _ = worker.get_finished()
    assert "renewed-head" not in done_sending
    assert worker._reqs_to_send["renewed-head"] == 200.0

    clock["now"] = 200.0
    done_sending, _ = worker.get_finished()
    assert "renewed-head" in done_sending
    assert "renewed-head" not in worker._reqs_to_send


def test_heartbeat_for_removed_request_is_noop(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(base_worker.time, "perf_counter", lambda: 50.0)
    worker = _make_worker()
    _register(worker, {"request": 100.0})
    worker._reqs_to_send.pop("request")

    worker._handle_heartbeat("request")

    assert "request" not in worker._reqs_to_send
