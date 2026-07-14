# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the scheduler-driven heartbeat / lease-renewal system."""

import queue
import time
from typing import Any
from unittest.mock import MagicMock, patch

import msgspec
import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_scheduler import (
    NixlBaseConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    PULL_OFFER_CANCELLATION_CONTROL_PREFIX,
    PULL_READ_COMPLETE_PREFIX,
    HeartbeatInfo,
    NixlConnectorMetadata,
    ProducerLease,
    PullOfferCancellationAck,
    PullOfferCancellationControl,
    PullOfferCancelled,
    PullReadComplete,
    RemoteMeta,
    ReqMeta,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_scheduler import (
    _consumer_tp_size,
    _expected_consumers,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_worker import (
    _consumer_ranks_for_producer,
    _parallel_consumer_index,
)
from vllm.v1.outputs import KVConnectorOutput

from .utils import create_request, make_nixl_scheduler

_ENGINE_A = "my-engine-id"


def _sched(kv_lease_duration: int = 30):
    return make_nixl_scheduler(heartbeat=True, kv_lease_duration=kv_lease_duration)


def _req(request_id: int = 1):
    return create_request(request_id=request_id, do_remote_prefill=True)


def _worker_stub():
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker import (
        NixlConnectorWorker,
    )

    w = object.__new__(NixlConnectorWorker)
    w._reqs_to_send = {}
    w._reqs_to_process = set()
    w._lease_extension = 20
    w._heartbeat_targets = {}
    w._heartbeat_interval = 5
    w._last_heartbeat_time = 0.0
    w.tp_rank = 0
    w.world_size = 1
    w._pull_completion_states = {}
    w._buffered_pull_completions = {}
    w._buffered_offer_cancellations = {}
    w._pending_offer_cancellations = []
    w._completed_pull_contracts = {}
    w._cancelled_remote_offers = set()
    w._recving_metadata = {}
    w._rid_completion_counts = {}
    w._released_rids = {}
    w._localization_source_rosters = {}
    return w


def _completion_proof(
    *,
    producer_request_id: str = "prefill-1",
    consumer_index: int,
    consumer_rank: int = 0,
    consumer_tp_size: int = 1,
    expected_consumers: int = 4,
) -> bytes:
    proof = PullReadComplete(
        producer_request_id=producer_request_id,
        consumer_request_id=(
            f"{consumer_index}_decode-request"
            if expected_consumers > 1
            else "decode-request"
        ),
        consumer_index=consumer_index,
        consumer_rank=consumer_rank,
        consumer_tp_size=consumer_tp_size,
        expected_consumers=expected_consumers,
    )
    return PULL_READ_COMPLETE_PREFIX + msgspec.msgpack.encode(proof)


def _cancellation_proof(
    *,
    producer_request_id: str = "prefill-1",
    consumer_rank: int = 0,
    consumer_tp_size: int = 1,
    expected_consumers: int = 4,
) -> PullOfferCancelled:
    return PullOfferCancelled(
        producer_request_id=producer_request_id,
        consumer_rank=consumer_rank,
        consumer_tp_size=consumer_tp_size,
        expected_consumers=expected_consumers,
    )


def _offer_params(
    *,
    producer_tp_size: int,
    consumer_tp_size: int,
    expected_consumers: int = 8,
) -> dict[str, Any]:
    return {
        "do_remote_prefill": True,
        "remote_engine_id": "producer-engine",
        "remote_request_id": "prefill-1",
        "remote_host": "producer-host",
        "remote_port": 1234,
        "tp_size": producer_tp_size,
        "expected_consumers": expected_consumers,
        "consumer_tp_size": consumer_tp_size,
    }


# ===================================================================
# Scheduler: on_new_request
# ===================================================================


def test_on_new_request_tracks_and_groups():
    """Add two reqs to same engine, one to another; verify grouping."""
    s = _sched()
    s.on_new_request(_req(1))
    s.on_new_request(_req(2))

    assert s._heartbeat_by_engine[_ENGINE_A].request_refcounts == {
        "prefill-1": 1,
        "prefill-2": 1,
    }
    info = s._heartbeat_by_engine[_ENGINE_A]
    assert (info.host, info.port, info.tp_size) == ("my-host", 1234, 1)
    assert s._heartbeat_req_engine["id-1"] == (_ENGINE_A, "prefill-1")

    # Different engine.
    r3 = _req(3)
    r3.kv_transfer_params["remote_engine_id"] = "engine-b"
    s.on_new_request(r3)
    assert len(s._heartbeat_by_engine) == 2


@pytest.mark.parametrize(
    "make_req",
    [
        lambda: create_request(request_id=2, do_remote_decode=True),
        lambda: create_request(request_id=3),  # no kv_transfer_params
    ],
    ids=["decode", "plain"],
)
def test_on_new_request_ignores_non_prefill(make_req):
    s = _sched()
    s.on_new_request(make_req())
    assert len(s._heartbeat_by_engine) == 0


# ===================================================================
# Scheduler: _stop_heartbeat
# ===================================================================


def test_stop_heartbeat_partial_and_full():
    """Stop one of two reqs on same engine, then stop the other."""
    s = _sched()
    s.on_new_request(_req(1))
    s.on_new_request(_req(2))

    s._stop_heartbeat("id-1")
    assert s._heartbeat_by_engine[_ENGINE_A].request_refcounts == {"prefill-2": 1}
    assert "id-1" not in s._heartbeat_req_engine

    s._stop_heartbeat("id-2")
    assert len(s._heartbeat_by_engine) == 0
    assert len(s._heartbeat_req_engine) == 0


def test_stop_heartbeat_keeps_shared_producer_request_alive() -> None:
    s = _sched()
    first = _req(1)
    second = _req(2)
    second.kv_transfer_params["remote_request_id"] = "prefill-1"

    s.on_new_request(first)
    s.on_new_request(second)

    info = s._heartbeat_by_engine[_ENGINE_A]
    assert info.request_refcounts == {"prefill-1": 2}

    s._stop_heartbeat("id-1")
    assert info.request_refcounts == {"prefill-1": 1}
    assert _ENGINE_A in s._heartbeat_by_engine

    s._stop_heartbeat("id-2")
    assert len(s._heartbeat_by_engine) == 0
    assert len(s._heartbeat_req_engine) == 0


# ===================================================================
# Scheduler: heartbeat ownership snapshots
# ===================================================================


def test_build_connector_meta_emits_immutable_ownership_changes() -> None:
    s = _sched()
    s.on_new_request(_req(1))

    meta1 = s.build_connector_meta(MagicMock())
    assert meta1.heartbeat_snapshot is not None
    assert meta1.heartbeat_snapshot[_ENGINE_A].request_refcounts == {"prefill-1": 1}

    meta2 = s.build_connector_meta(MagicMock())
    assert meta2.heartbeat_snapshot is None

    s.on_new_request(_req(2))
    assert meta1.heartbeat_snapshot[_ENGINE_A].request_refcounts == {"prefill-1": 1}
    meta3 = s.build_connector_meta(MagicMock())
    assert meta3.heartbeat_snapshot is not None
    assert meta3.heartbeat_snapshot[_ENGINE_A].request_refcounts == {
        "prefill-1": 1,
        "prefill-2": 1,
    }

    s._stop_heartbeat("id-1")
    s._stop_heartbeat("id-2")
    meta4 = s.build_connector_meta(MagicMock())
    assert meta4.heartbeat_snapshot == {}


# ===================================================================
# Scheduler: cleanup paths (update_connector_output / request_finished)
# ===================================================================


def test_update_connector_output_stops_heartbeat():
    s = _sched()
    s.on_new_request(_req(1))

    s.update_connector_output(
        KVConnectorOutput(
            finished_sending=None,
            finished_recving={"id-1"},
            invalid_block_ids=set(),
        )
    )

    assert len(s._heartbeat_by_engine) == 0
    assert len(s._heartbeat_req_engine) == 0


def test_request_finished_stops_heartbeat():
    s = _sched()
    r = _req(1)
    s.on_new_request(r)

    # Simulate update_state_after_alloc having consumed do_remote_prefill.
    r.kv_transfer_params["do_remote_prefill"] = False
    s.request_finished(r, block_ids=())

    assert len(s._heartbeat_by_engine) == 0
    assert len(s._heartbeat_req_engine) == 0


# ===================================================================
# Worker: _handle_heartbeat
# ===================================================================


def test_handle_heartbeat():
    w = _worker_stub()
    far_future = time.perf_counter() + 99999
    w._reqs_to_send = {"req-a": 100.0, "req-b": far_future}

    before = time.perf_counter()
    w._handle_heartbeat("req-a,req-b,req-unknown")

    # req-a: pushed forward to ~now+20.
    assert w._reqs_to_send["req-a"] >= before + 20
    # req-b: already far out, max() keeps it.
    assert w._reqs_to_send["req-b"] >= far_future
    # req-unknown: not added.
    assert "req-unknown" not in w._reqs_to_send


def test_overdue_lease_retains_producer_ownership() -> None:
    worker = _worker_stub()
    future_deadline = time.perf_counter() + 100
    worker._reqs_to_send = {
        "prefill-future": future_deadline,
        "prefill-1": time.perf_counter() - 1,
    }
    worker._reqs_to_process = {"prefill-future", "prefill-1"}
    worker._install_pull_completion_state(
        "prefill-1",
        ProducerLease(
            deadline=worker._reqs_to_send["prefill-1"],
            expected_consumers=4,
            consumer_tp_size=1,
        ),
    )
    state = worker._pull_completion_states["prefill-1"]
    state.read_completions.update({(0, 0), (1, 0), (2, 0)})
    worker.xfer_stats = MagicMock()

    worker._mark_overdue_leases(time.perf_counter())

    assert worker._reqs_to_send == {"prefill-future": future_deadline}
    assert worker._reqs_to_process == {"prefill-future", "prefill-1"}
    assert state.read_completions == {(0, 0), (1, 0), (2, 0)}
    worker.xfer_stats.record_kv_expired_req.assert_called_once_with()

    worker._mark_overdue_leases(time.perf_counter())
    worker.xfer_stats.record_kv_expired_req.assert_called_once_with()

    worker.nixl_wrapper = MagicMock()
    worker.nixl_wrapper.get_new_notifs.return_value = {
        "decoder": [_completion_proof(consumer_index=3)]
    }
    worker._localization_capture_source_post = MagicMock()

    assert worker._get_new_notifs() == {"prefill-1"}
    assert worker._reqs_to_process == {"prefill-future"}
    assert worker._pull_completion_states == {}
    worker._localization_capture_source_post.assert_called_once_with("prefill-1")


def test_completion_proofs_are_idempotent_and_contract_exact() -> None:
    worker = _worker_stub()
    worker._reqs_to_process = {"prefill-1"}
    worker._reqs_to_send = {"prefill-1": time.perf_counter() + 30}
    worker._install_pull_completion_state(
        "prefill-1",
        ProducerLease(
            deadline=worker._reqs_to_send["prefill-1"],
            expected_consumers=2,
            consumer_tp_size=2,
        ),
    )
    worker.nixl_wrapper = MagicMock()
    worker._localization_capture_source_post = MagicMock()
    duplicate = _completion_proof(
        consumer_index=0,
        consumer_rank=0,
        consumer_tp_size=2,
        expected_consumers=2,
    )
    conflicting = _completion_proof(
        consumer_index=1,
        consumer_rank=0,
        consumer_tp_size=2,
        expected_consumers=1,
    )
    worker.nixl_wrapper.get_new_notifs.return_value = {
        "decoder": [
            duplicate,
            duplicate,
            conflicting,
            _completion_proof(
                consumer_index=0,
                consumer_rank=1,
                consumer_tp_size=2,
                expected_consumers=2,
            ),
            _completion_proof(
                consumer_index=1,
                consumer_rank=0,
                consumer_tp_size=2,
                expected_consumers=2,
            ),
        ]
    }

    assert worker._get_new_notifs() == set()
    assert worker._pull_completion_states["prefill-1"].read_completions == {
        (0, 0),
        (0, 1),
        (1, 0),
    }

    worker.nixl_wrapper.get_new_notifs.return_value = {
        "decoder": [
            _completion_proof(
                consumer_index=1,
                consumer_rank=1,
                consumer_tp_size=2,
                expected_consumers=2,
            )
        ]
    }
    assert worker._get_new_notifs() == {"prefill-1"}

    worker.nixl_wrapper.get_new_notifs.return_value = {"decoder": [duplicate]}
    assert worker._get_new_notifs() == set()


def test_completion_proof_waits_for_async_producer_contract() -> None:
    worker = _worker_stub()
    worker._reqs_to_process = {"prefill-1"}
    worker._localization_capture_source_post = MagicMock()
    worker.nixl_wrapper = MagicMock()
    proof = _completion_proof(
        consumer_index=0,
        expected_consumers=1,
    )
    worker.nixl_wrapper.get_new_notifs.return_value = {"decoder": [proof]}

    assert worker._get_new_notifs() == set()
    assert worker._buffered_pull_completions == {
        "prefill-1": {
            msgspec.msgpack.decode(
                proof[len(PULL_READ_COMPLETE_PREFIX) :],
                type=PullReadComplete,
            )
        }
    }

    deadline = time.perf_counter() + 30
    worker._install_pull_completion_state(
        "prefill-1",
        ProducerLease(
            deadline=deadline,
            expected_consumers=1,
            consumer_tp_size=1,
        ),
    )
    worker._reqs_to_send["prefill-1"] = deadline
    worker.nixl_wrapper.get_new_notifs.return_value = {}

    assert worker._get_new_notifs() == {"prefill-1"}
    assert worker._buffered_pull_completions == {}
    assert worker._reqs_to_process == set()
    assert worker._reqs_to_send == {}


def test_whole_offer_cancellation_uses_rank_quorum_not_consumer_count() -> None:
    worker = _worker_stub()
    worker._reqs_to_process = {"prefill-1"}
    deadline = time.perf_counter() + 30
    worker._reqs_to_send = {"prefill-1": deadline}
    worker._localization_source_rosters = {"prefill-1": MagicMock()}
    worker._install_pull_completion_state(
        "prefill-1",
        ProducerLease(
            deadline=deadline,
            expected_consumers=8,
            consumer_tp_size=2,
        ),
    )
    worker.nixl_wrapper = MagicMock()
    worker.nixl_wrapper.get_new_notifs.return_value = {}
    worker._pending_offer_cancellations = [
        _cancellation_proof(
            consumer_rank=0,
            consumer_tp_size=2,
            expected_consumers=8,
        )
    ]

    assert worker._get_new_notifs() == set()
    state = worker._pull_completion_states["prefill-1"]
    assert state.offer_cancellations == {0}
    assert len(state.read_completions) == 0

    worker._pending_offer_cancellations = [
        _cancellation_proof(
            consumer_rank=1,
            consumer_tp_size=2,
            expected_consumers=8,
        )
    ]
    assert worker._get_new_notifs() == {"prefill-1"}
    assert worker._reqs_to_process == set()
    assert worker._reqs_to_send == {}
    assert worker._localization_source_rosters == {}
    assert (
        worker._completed_pull_contracts["prefill-1"].terminal_mode == "offer_cancelled"
    )


def test_read_and_cancellation_mixture_permanently_pins_offer() -> None:
    worker = _worker_stub()
    worker._reqs_to_process = {"prefill-1"}
    deadline = time.perf_counter() + 30
    worker._reqs_to_send = {"prefill-1": deadline}
    worker._install_pull_completion_state(
        "prefill-1",
        ProducerLease(
            deadline=deadline,
            expected_consumers=1,
            consumer_tp_size=2,
        ),
    )
    worker.nixl_wrapper = MagicMock()
    worker._localization_capture_source_post = MagicMock()
    worker._pending_offer_cancellations = [
        _cancellation_proof(
            consumer_rank=0,
            consumer_tp_size=2,
            expected_consumers=1,
        )
    ]
    worker.nixl_wrapper.get_new_notifs.return_value = {
        "decoder": [
            _completion_proof(
                consumer_index=0,
                consumer_rank=0,
                consumer_tp_size=2,
                expected_consumers=1,
            ),
        ]
    }

    assert worker._get_new_notifs() == set()
    state = worker._pull_completion_states["prefill-1"]
    assert state.has_mixed_terminal_proofs
    assert state.read_completions == {(0, 0)}
    assert state.offer_cancellations == {0}

    worker._pending_offer_cancellations = [
        _cancellation_proof(
            consumer_rank=1,
            consumer_tp_size=2,
            expected_consumers=1,
        )
    ]
    worker.nixl_wrapper.get_new_notifs.return_value = {
        "decoder": [
            _completion_proof(
                consumer_index=0,
                consumer_rank=1,
                consumer_tp_size=2,
                expected_consumers=1,
            ),
        ]
    }
    assert worker._get_new_notifs() == set()
    assert state.read_completions == state.expected_read_completions
    assert state.offer_cancellations == state.expected_offer_cancellations
    assert worker._reqs_to_process == {"prefill-1"}
    assert worker._reqs_to_send == {"prefill-1": deadline}
    worker._localization_capture_source_post.assert_not_called()


def test_offer_cancellation_buffers_only_for_owned_pre_contract_request() -> None:
    worker = _worker_stub()
    worker.nixl_wrapper = MagicMock()
    proof = _cancellation_proof(expected_consumers=1)
    worker.nixl_wrapper.get_new_notifs.return_value = {}
    worker._pending_offer_cancellations = [proof]

    assert worker._get_new_notifs() == set()
    assert worker._buffered_offer_cancellations == {}

    worker._reqs_to_process = {"prefill-1"}
    worker._pending_offer_cancellations = [proof]
    assert worker._get_new_notifs() == set()
    assert len(worker._buffered_offer_cancellations["prefill-1"]) == 1

    deadline = time.perf_counter() + 30
    worker._reqs_to_send = {"prefill-1": deadline}
    worker._install_pull_completion_state(
        "prefill-1",
        ProducerLease(
            deadline=deadline,
            expected_consumers=1,
            consumer_tp_size=1,
        ),
    )
    worker.nixl_wrapper.get_new_notifs.return_value = {}

    assert worker._get_new_notifs() == {"prefill-1"}
    assert worker._buffered_offer_cancellations == {}
    assert worker._reqs_to_process == set()


def test_side_channel_atomically_queues_typed_offer_cancellation() -> None:
    pending: queue.Queue[PullOfferCancellationControl] = queue.Queue(maxsize=1)
    control = PullOfferCancellationControl(
        producer_ranks=(0, 1),
        proof=_cancellation_proof(
            consumer_rank=0,
            consumer_tp_size=2,
            expected_consumers=8,
        ),
    )
    message = PULL_OFFER_CANCELLATION_CONTROL_PREFIX + msgspec.msgpack.encode(control)

    response = NixlBaseConnectorScheduler._queue_offer_cancellation(
        message,
        frozenset({0, 1}),
        pending,
    )

    assert msgspec.msgpack.decode(response, type=PullOfferCancellationAck) == (
        PullOfferCancellationAck(
            producer_request_id="prefill-1",
            producer_ranks=(0, 1),
            accepted=True,
        )
    )
    assert pending.get_nowait() == control


def test_side_channel_rejects_cancellation_for_unknown_producer_rank() -> None:
    pending: queue.Queue[PullOfferCancellationControl] = queue.Queue(maxsize=1)
    control = PullOfferCancellationControl(
        producer_ranks=(1,),
        proof=_cancellation_proof(expected_consumers=1),
    )
    message = PULL_OFFER_CANCELLATION_CONTROL_PREFIX + msgspec.msgpack.encode(control)

    response = NixlBaseConnectorScheduler._queue_offer_cancellation(
        message,
        frozenset({0}),
        pending,
    )

    ack = msgspec.msgpack.decode(response, type=PullOfferCancellationAck)
    assert ack.accepted is False
    assert pending.empty()


def test_scheduler_routes_and_deduplicates_cancellations_by_producer_rank() -> None:
    scheduler = _sched()
    proof = _cancellation_proof(
        consumer_rank=0,
        consumer_tp_size=2,
        expected_consumers=8,
    )
    control = PullOfferCancellationControl(
        producer_ranks=(0, 1),
        proof=proof,
    )
    scheduler._offer_cancellation_queue.put_nowait(control)
    scheduler._offer_cancellation_queue.put_nowait(control)

    scheduler_output = MagicMock()
    scheduler_output.num_scheduled_tokens = {}
    metadata = scheduler.build_connector_meta(scheduler_output)

    assert metadata.offer_cancellations_by_rank == {
        0: (proof,),
        1: (proof,),
    }
    assert scheduler._drain_offer_cancellations() == {}


def test_offer_cancellation_rejects_rank_outside_producer_mapping() -> None:
    worker = _worker_stub()
    worker.world_size = 2
    worker.tp_rank = 0
    worker._reqs_to_process = {"prefill-1"}
    deadline = time.perf_counter() + 30
    worker._reqs_to_send = {"prefill-1": deadline}
    worker._install_pull_completion_state(
        "prefill-1",
        ProducerLease(
            deadline=deadline,
            expected_consumers=1,
            consumer_tp_size=4,
        ),
    )
    worker.nixl_wrapper = MagicMock()
    worker.nixl_wrapper.get_new_notifs.return_value = {}
    worker._pending_offer_cancellations = [
        _cancellation_proof(
            consumer_rank=2,
            consumer_tp_size=4,
            expected_consumers=1,
        )
    ]

    assert worker._get_new_notifs() == set()
    assert worker._pull_completion_states["prefill-1"].offer_cancellations == set()


@pytest.mark.parametrize(
    (
        "decoder_rank",
        "decoder_tp_size",
        "producer_tp_size",
        "expected_producer_ranks",
    ),
    [
        (2, 4, 2, (1,)),
        (0, 2, 4, (0, 1)),
    ],
)
def test_decoder_sends_typed_control_to_exact_producer_ranks(
    decoder_rank: int,
    decoder_tp_size: int,
    producer_tp_size: int,
    expected_producer_ranks: tuple[int, ...],
) -> None:
    worker = _worker_stub()
    worker.tp_rank = decoder_rank
    worker.world_size = decoder_tp_size
    worker.xfer_stats = MagicMock()
    socket = MagicMock()
    socket.recv.return_value = msgspec.msgpack.encode(
        PullOfferCancellationAck(
            producer_request_id="prefill-1",
            producer_ranks=expected_producer_ranks,
            accepted=True,
        )
    )
    socket_context = MagicMock()
    socket_context.__enter__.return_value = socket

    with patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_worker.zmq_ctx",
        return_value=socket_context,
    ):
        assert worker.request_rejected_before_admission(
            "random-serving-request",
            _offer_params(
                producer_tp_size=producer_tp_size,
                consumer_tp_size=decoder_tp_size,
            ),
            "capacity rejection",
        )

    socket.send.assert_called_once()
    message = socket.send.call_args.args[0]
    assert b"random-serving-request" not in message
    assert b"capacity rejection" not in message
    control = msgspec.msgpack.decode(
        message[len(PULL_OFFER_CANCELLATION_CONTROL_PREFIX) :],
        type=PullOfferCancellationControl,
    )
    assert control == PullOfferCancellationControl(
        producer_ranks=expected_producer_ranks,
        proof=PullOfferCancelled(
            producer_request_id="prefill-1",
            consumer_rank=decoder_rank,
            consumer_tp_size=decoder_tp_size,
            expected_consumers=8,
        ),
    )
    assert set(worker._cancelled_remote_offers) == {("producer-engine", "prefill-1")}


def test_offer_cancellation_fails_closed_when_producer_rejects_control() -> None:
    worker = _worker_stub()
    worker.xfer_stats = MagicMock()
    socket = MagicMock()
    socket.recv.return_value = msgspec.msgpack.encode(
        PullOfferCancellationAck(
            producer_request_id="prefill-1",
            producer_ranks=(0,),
            accepted=False,
        )
    )
    socket_context = MagicMock()
    socket_context.__enter__.return_value = socket

    with patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_worker.zmq_ctx",
        return_value=socket_context,
    ):
        assert not worker.request_rejected_before_admission(
            "random-serving-request",
            _offer_params(producer_tp_size=1, consumer_tp_size=1),
            "capacity rejection",
        )

    assert set(worker._cancelled_remote_offers) == {("producer-engine", "prefill-1")}
    worker.xfer_stats.record_failed_notification.assert_called_once_with()


def test_offer_cancellation_refuses_existing_decoder_read_state() -> None:
    worker = _worker_stub()
    worker._recving_metadata = {
        "0_decode-request": ReqMeta(
            local_block_ids=(),
            local_physical_block_ids=(),
            tp_size=1,
            remote=RemoteMeta(
                block_ids=(),
                host="producer-host",
                port=1234,
                engine_id="producer-engine",
                request_id="prefill-1",
                expected_consumers=8,
                consumer_tp_size=1,
            ),
        )
    }
    worker._send_offer_cancellation_control = MagicMock()

    assert not worker.request_rejected_before_admission(
        "random-serving-request",
        _offer_params(producer_tp_size=1, consumer_tp_size=1),
        "capacity rejection",
    )
    worker._send_offer_cancellation_control.assert_not_called()
    assert worker._cancelled_remote_offers == set()


def test_offer_cancellation_never_evicts_an_existing_local_fence() -> None:
    worker = _worker_stub()
    worker._cancelled_remote_offers = {("older-engine", "older-offer")}
    worker._send_offer_cancellation_control = MagicMock()

    with patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_worker."
        "_MAX_CANCELLED_REMOTE_OFFERS",
        1,
    ):
        assert not worker.request_rejected_before_admission(
            "random-serving-request",
            _offer_params(producer_tp_size=1, consumer_tp_size=1),
            "capacity rejection",
        )

    assert worker._cancelled_remote_offers == {("older-engine", "older-offer")}
    worker._send_offer_cancellation_control.assert_not_called()


def test_cancelled_offer_fence_rejects_late_read_before_notification() -> None:
    worker = _worker_stub()
    assert worker._fence_remote_offer("producer-engine", "prefill-1")
    worker._handle_failed_transfer = MagicMock()
    worker._read_completion_notification = MagicMock()
    meta = ReqMeta(
        local_block_ids=(),
        local_physical_block_ids=(),
        tp_size=1,
        remote=RemoteMeta(
            block_ids=(),
            host="producer-host",
            port=1234,
            engine_id="producer-engine",
            request_id="prefill-1",
            expected_consumers=1,
            consumer_tp_size=1,
        ),
    )

    worker._read_blocks_for_req("decode-request", meta)

    worker._handle_failed_transfer.assert_called_once_with("decode-request", None)
    worker._read_completion_notification.assert_not_called()


@pytest.mark.parametrize(
    ("producer_rank", "producer_tp_size", "consumer_tp_size", "expected"),
    [
        (0, 1, 1, (0,)),
        (0, 2, 4, (0, 1)),
        (1, 2, 4, (2, 3)),
        (0, 4, 2, (0,)),
        (1, 4, 2, (0,)),
        (2, 4, 2, (1,)),
        (3, 4, 2, (1,)),
    ],
)
def test_producer_obligations_follow_tensor_parallel_mapping(
    producer_rank: int,
    producer_tp_size: int,
    consumer_tp_size: int,
    expected: tuple[int, ...],
) -> None:
    assert (
        _consumer_ranks_for_producer(
            producer_rank,
            producer_tp_size,
            consumer_tp_size,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("producer_tp_size", "consumer_tp_size"),
    [(2, 3), (3, 2)],
)
def test_producer_obligations_reject_incompatible_topologies(
    producer_tp_size: int,
    consumer_tp_size: int,
) -> None:
    with pytest.raises(ValueError, match="must be divisible"):
        _consumer_ranks_for_producer(0, producer_tp_size, consumer_tp_size)


def test_parallel_consumer_identity_is_strict() -> None:
    assert _parallel_consumer_index("0_parent-request", 2) == 0
    assert _parallel_consumer_index("1_parent-request", 2) == 1
    assert _parallel_consumer_index("parent-request", 1) == 0

    with pytest.raises(ValueError, match="lacks"):
        _parallel_consumer_index("parent-request", 2)
    with pytest.raises(ValueError, match="outside"):
        _parallel_consumer_index("2_parent-request", 2)
    with pytest.raises(ValueError, match="expects one"):
        _parallel_consumer_index("0_parent-request", 1)


@pytest.mark.parametrize("value", [False, "2", 1.5, 0, -1])
def test_producer_contract_requires_positive_integers(value: object) -> None:
    with pytest.raises(ValueError, match="expected_consumers"):
        _expected_consumers({"expected_consumers": value})
    with pytest.raises(ValueError, match="consumer_tp_size"):
        _consumer_tp_size({"consumer_tp_size": value})

    assert _expected_consumers({"expected_consumers": 2}) == 2
    assert _consumer_tp_size({"consumer_tp_size": 4}) == 4


def test_decoder_completion_proof_uses_source_owned_contract() -> None:
    worker = _worker_stub()
    worker.tp_rank = 1
    worker.world_size = 2
    meta = ReqMeta(
        local_block_ids=(),
        local_physical_block_ids=(),
        tp_size=4,
        remote=RemoteMeta(
            block_ids=(),
            host="producer",
            port=1234,
            engine_id="producer-engine",
            request_id="prefill-1",
            expected_consumers=4,
            consumer_tp_size=2,
        ),
    )

    encoded = worker._read_completion_notification("3_decode-request", meta)
    proof = msgspec.msgpack.decode(
        encoded[len(PULL_READ_COMPLETE_PREFIX) :],
        type=PullReadComplete,
    )

    assert proof == PullReadComplete(
        producer_request_id="prefill-1",
        consumer_request_id="3_decode-request",
        consumer_index=3,
        consumer_rank=1,
        consumer_tp_size=2,
        expected_consumers=4,
    )


def test_zero_byte_completion_finishes_decoder_without_release_on_send_error() -> None:
    worker = _worker_stub()
    worker._remote_agents = {"producer-engine": {0: "producer-rank-0"}}
    worker._recving_transfers = {}
    worker.nixl_wrapper = MagicMock()
    worker.nixl_wrapper.send_notif.side_effect = RuntimeError("transport failed")
    worker._log_failure = MagicMock()
    worker.xfer_stats = MagicMock()

    worker._send_zero_byte_completion(
        "decode-request",
        "producer-engine",
        0,
        b"proof",
    )

    assert worker._recving_transfers == {"decode-request": []}
    worker.xfer_stats.record_failed_notification.assert_called_once_with()
    worker._log_failure.assert_called_once()


def test_worker_retains_and_services_heartbeat_snapshot() -> None:
    worker = _worker_stub()
    worker._dispatch_heartbeat_targets = MagicMock()
    metadata = NixlConnectorMetadata()
    info = HeartbeatInfo(
        request_refcounts={"prefill-1": 2},
        host="my-host",
        port=1234,
        tp_size=1,
    )
    metadata.heartbeat_snapshot = {_ENGINE_A: info}

    worker._update_heartbeat_targets(metadata)
    worker._service_heartbeats()

    worker._dispatch_heartbeat_targets.assert_called_once_with({_ENGINE_A: info})

    unchanged = NixlConnectorMetadata()
    worker._update_heartbeat_targets(unchanged)
    assert worker._heartbeat_targets == {_ENGINE_A: info}
    worker._service_heartbeats()
    worker._dispatch_heartbeat_targets.assert_called_once()

    worker._last_heartbeat_time -= worker._heartbeat_interval
    worker._service_heartbeats()
    assert worker._dispatch_heartbeat_targets.call_count == 2

    cleared = NixlConnectorMetadata()
    cleared.heartbeat_snapshot = {}
    worker._update_heartbeat_targets(cleared)
    assert worker._heartbeat_targets == {}


def test_worker_heartbeat_payload_deduplicates_shared_request() -> None:
    worker = _worker_stub()
    worker._ensure_handshake = MagicMock(return_value=None)
    worker._remote_agents = {_ENGINE_A: {0: "producer-rank-0"}}
    worker.nixl_wrapper = MagicMock()
    targets = {
        _ENGINE_A: HeartbeatInfo(
            request_refcounts={"prefill-1": 8},
            host="my-host",
            port=1234,
            tp_size=1,
        )
    }

    worker._send_heartbeat_targets(targets)

    worker.nixl_wrapper.send_notif.assert_called_once_with(
        "producer-rank-0",
        notif_msg=b"HB:prefill-1",
    )
