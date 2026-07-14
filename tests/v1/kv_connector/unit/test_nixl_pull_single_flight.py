# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for parallel-consumer NIXL pull single-flight."""

from unittest.mock import MagicMock

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_scheduler import (
    NixlPullConnectorScheduler,
)
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request, RequestStatus

from .utils import (
    create_request,
    create_scheduler,
    create_vllm_config,
    make_nixl_scheduler,
)


def _make_parallel_consumer(
    request_id: int,
    *,
    expected_consumers: int = 3,
    engine_id: str = "producer-engine",
    producer_request_id: str = "producer-request",
    generation: int | None = 7,
    num_tokens: int = 128,
) -> Request:
    """Build one logical consumer of a shared producer offer.

    :param request_id: Unique local consumer index.
    :param expected_consumers: Producer-owned parallel-consumer count.
    :param engine_id: Producer engine identity.
    :param producer_request_id: Producer request identity.
    :param generation: Optional immutable producer offer generation.
    :param num_tokens: Prompt length shared by all consumers.
    :returns: Decoder request carrying the producer offer.
    """
    request = create_request(
        request_id=request_id,
        num_tokens=num_tokens,
        common_prefix_len=num_tokens,
        do_remote_prefill=True,
    )
    params = request.kv_transfer_params
    assert params is not None
    params.update(
        remote_engine_id=engine_id,
        remote_request_id=producer_request_id,
        remote_num_tokens=num_tokens,
        expected_consumers=expected_consumers,
    )
    if generation is None:
        params.pop("p2d_offer_generation", None)
    else:
        params["p2d_offer_generation"] = generation
    return request


def _make_inverse_pull_request(request_id: int) -> Request:
    """Build a P-side inverse-direction pull sharing a remote identity.

    :param request_id: Unique local request index.
    :returns: Request whose transfer phase is ``do_remote_decode``.
    """
    request = create_request(
        request_id=request_id,
        num_tokens=128,
        common_prefix_len=128,
        do_remote_decode=True,
    )
    params = request.kv_transfer_params
    assert params is not None
    params.update(
        remote_engine_id="decoder-engine",
        remote_request_id="decoder-request",
        remote_block_ids=([1, 2, 3],),
        remote_num_tokens=128,
        remote_host="decoder-host",
        remote_port=1234,
        expected_consumers=3,
    )
    return request


@pytest.mark.cpu_test
def test_parallel_pull_waits_for_apc_publication_and_keeps_tail_reads() -> None:
    """One child performs the cold pull and every sibling keeps a tail read."""
    scheduler: NixlPullConnectorScheduler = make_nixl_scheduler(heartbeat=True)
    leader, follower_one, follower_two = [
        _make_parallel_consumer(index) for index in range(1, 4)
    ]

    assert scheduler.get_num_new_matched_tokens(leader, 0) == (128, True)
    assert scheduler.get_num_new_matched_tokens(follower_one, 0) == (None, False)
    assert scheduler.get_num_new_matched_tokens(follower_two, 0) == (None, False)
    assert scheduler.get_num_new_matched_tokens(follower_one, 32) == (None, False)

    assert scheduler.get_num_new_matched_tokens(follower_one, 112) == (16, True)
    assert scheduler.get_num_new_matched_tokens(follower_two, 112) == (16, True)
    assert len(scheduler._parallel_pull_flights) == 1
    assert scheduler.get_num_new_matched_tokens(follower_two, 112) == (16, True)
    assert len(scheduler._parallel_pull_flights) == 1

    blocks = MagicMock()
    blocks.get_unhashed_block_ids_all_groups.return_value = ([11, 12],)
    scheduler.update_state_after_alloc(leader, blocks, 128)
    scheduler.update_state_after_alloc(follower_one, blocks, 16)
    scheduler.update_state_after_alloc(follower_two, blocks, 16)

    assert scheduler._parallel_pull_flights == {}
    assert set(scheduler._reqs_need_recv) == {
        leader.request_id,
        follower_one.request_id,
        follower_two.request_id,
    }
    assert all(
        request.kv_transfer_params is not None
        and request.kv_transfer_params["do_remote_prefill"] is False
        for request in (leader, follower_one, follower_two)
    )


@pytest.mark.cpu_test
def test_parallel_pull_apc_publication_follows_receive_completion() -> None:
    """Transferred blocks become APC-visible only after receive completion."""
    scheduler = create_scheduler(create_vllm_config(max_num_batched_tokens=64))
    leader = _make_parallel_consumer(1, expected_consumers=2)
    follower = _make_parallel_consumer(2, expected_consumers=2)

    scheduler.add_request(leader)
    scheduler.add_request(follower)
    first_output = scheduler.schedule()

    first_metadata = first_output.kv_connector_metadata
    assert isinstance(first_metadata, NixlConnectorMetadata)
    assert set(first_metadata.reqs_to_recv) == {leader.request_id}
    _, follower_hit_before_completion = scheduler.kv_cache_manager.get_computed_blocks(
        follower
    )
    assert follower_hit_before_completion == 0

    scheduler._update_from_kv_xfer_finished(
        KVConnectorOutput(
            finished_sending=None,
            finished_recving={leader.request_id},
            invalid_block_ids=set(),
        )
    )
    _, follower_hit_before_promotion = scheduler.kv_cache_manager.get_computed_blocks(
        follower
    )
    assert follower_hit_before_promotion == 0

    second_output = scheduler.schedule()

    second_metadata = second_output.kv_connector_metadata
    assert isinstance(second_metadata, NixlConnectorMetadata)
    assert set(second_metadata.reqs_to_recv) == {follower.request_id}
    assert len(second_metadata.reqs_to_recv[follower.request_id].local_block_ids) > 0


@pytest.mark.cpu_test
def test_parallel_pull_offer_identity_includes_engine_and_generation() -> None:
    """Different producer engines or generations never share a flight."""
    scheduler: NixlPullConnectorScheduler = make_nixl_scheduler()
    requests = (
        _make_parallel_consumer(1, engine_id="producer-a", generation=1),
        _make_parallel_consumer(2, engine_id="producer-b", generation=1),
        _make_parallel_consumer(3, engine_id="producer-a", generation=2),
    )

    for request in requests:
        assert scheduler.get_num_new_matched_tokens(request, 0) == (128, True)
    assert len(scheduler._parallel_pull_flights) == 3


@pytest.mark.cpu_test
def test_parallel_pull_rejects_prompt_length_drift_within_offer() -> None:
    """All children of one immutable producer offer share one prompt extent."""
    scheduler: NixlPullConnectorScheduler = make_nixl_scheduler()
    leader = _make_parallel_consumer(1, num_tokens=128)
    mismatched_follower = _make_parallel_consumer(2, num_tokens=127)

    assert scheduler.get_num_new_matched_tokens(leader, 0) == (128, True)
    with pytest.raises(ValueError, match="prompt length changed"):
        scheduler.get_num_new_matched_tokens(mismatched_follower, 0)


@pytest.mark.cpu_test
def test_parallel_pull_failure_elects_one_new_leader() -> None:
    """A failed leader cannot release all siblings from a partial APC hit."""
    scheduler: NixlPullConnectorScheduler = make_nixl_scheduler(heartbeat=True)
    leader, replacement, follower = [
        _make_parallel_consumer(index) for index in range(1, 4)
    ]

    assert scheduler.get_num_new_matched_tokens(leader, 0) == (128, True)
    assert scheduler.get_num_new_matched_tokens(replacement, 0) == (None, False)

    leader.status = RequestStatus.FINISHED_ABORTED
    params = leader.kv_transfer_params
    assert params is not None
    params["do_remote_prefill"] = False
    params["_remote_blocks_processed"] = True
    scheduler.request_finished(leader, ())

    assert scheduler.get_num_new_matched_tokens(replacement, 32) == (96, True)
    assert scheduler.get_num_new_matched_tokens(follower, 32) == (None, False)
    assert scheduler.get_num_new_matched_tokens(follower, 112) == (16, True)

    blocks = MagicMock()
    blocks.get_unhashed_block_ids_all_groups.return_value = ([21, 22],)
    scheduler.update_state_after_alloc(replacement, blocks, 96)
    scheduler.update_state_after_alloc(follower, blocks, 16)
    assert scheduler._parallel_pull_flights == {}


@pytest.mark.cpu_test
def test_inverse_direction_is_not_single_flighted() -> None:
    """The D-to-P inverse pull does not enter P-to-D offer coordination."""
    scheduler: NixlPullConnectorScheduler = make_nixl_scheduler()
    scheduler.kv_recompute_threshold = 0
    first = _make_inverse_pull_request(1)
    second = _make_inverse_pull_request(2)

    assert scheduler.get_num_new_matched_tokens(first, 0) == (128, True)
    assert scheduler.get_num_new_matched_tokens(second, 0) == (128, True)
    assert scheduler._parallel_pull_flights == {}


@pytest.mark.cpu_test
def test_parallel_pull_bypasses_single_flight_without_apc() -> None:
    """Parallel consumers remain live when prefix caching is unavailable."""
    scheduler: NixlPullConnectorScheduler = make_nixl_scheduler()
    scheduler._parallel_pull_single_flight_enabled = False
    first = _make_parallel_consumer(1)
    second = _make_parallel_consumer(2)

    assert scheduler.get_num_new_matched_tokens(first, 0) == (128, True)
    assert scheduler.get_num_new_matched_tokens(second, 0) == (128, True)
    assert scheduler._parallel_pull_flights == {}


@pytest.mark.cpu_test
def test_parallel_pull_does_not_serialize_uncacheable_tails() -> None:
    """An offer with only the mandatory tail left needs no full-pull leader."""
    scheduler: NixlPullConnectorScheduler = make_nixl_scheduler()
    first = _make_parallel_consumer(1)
    second = _make_parallel_consumer(2)

    assert scheduler.get_num_new_matched_tokens(first, 112) == (16, True)
    assert scheduler.get_num_new_matched_tokens(second, 112) == (16, True)
    assert scheduler._parallel_pull_flights == {}
