# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression coverage for window eviction while GPU batches remain in flight."""

import torch

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import ChunkedLocalAttentionSpec, SlidingWindowSpec
from vllm.v1.outputs import ModelRunnerOutput

from .utils import create_requests, create_scheduler, mock_kv

NUM_PROMPT_TOKENS = 100
BLOCK_SIZE = 16
SLIDING_WINDOW = 16
NUM_OUT_OF_WINDOW_BLOCKS = 85 // BLOCK_SIZE
CHUNK_SIZE = 32
NUM_OUT_OF_CHUNK_BLOCKS = (NUM_PROMPT_TOKENS // CHUNK_SIZE) * CHUNK_SIZE // BLOCK_SIZE


def _make_model_runner_output(
    scheduler_output: SchedulerOutput,
    token_id: int = 0,
) -> ModelRunnerOutput:
    """Build a successful model output for one scheduler batch.

    :param scheduler_output: Batch whose request IDs should be completed.
    :param token_id: Sampled token returned for each request.
    :returns: Model output matching the scheduler batch.
    """
    request_ids = list(scheduler_output.num_scheduled_tokens)
    return ModelRunnerOutput(
        req_ids=request_ids,
        req_id_to_index={
            request_id: index for index, request_id in enumerate(request_ids)
        },
        sampled_token_ids=[[token_id] for _ in request_ids],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def _create_swa_scheduler(async_scheduling: bool) -> Scheduler | AsyncScheduler:
    """Create a scheduler with one sliding-window cache group.

    :param async_scheduling: Whether two model batches may overlap.
    :returns: Scheduler configured for sliding-window attention.
    """
    return create_scheduler(
        block_size=BLOCK_SIZE,
        async_scheduling=async_scheduling,
        kv_cache_spec=SlidingWindowSpec(
            block_size=BLOCK_SIZE,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.float32,
            sliding_window=SLIDING_WINDOW,
        ),
    )


def _create_chunked_scheduler(async_scheduling: bool) -> Scheduler | AsyncScheduler:
    """Create a scheduler with one chunked-local cache group.

    :param async_scheduling: Whether two model batches may overlap.
    :returns: Scheduler configured for chunked-local attention.
    """
    return create_scheduler(
        block_size=BLOCK_SIZE,
        async_scheduling=async_scheduling,
        kv_cache_spec=ChunkedLocalAttentionSpec(
            block_size=BLOCK_SIZE,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.float32,
            attention_chunk_size=CHUNK_SIZE,
        ),
    )


def _num_null_blocks(
    scheduler: Scheduler | AsyncScheduler,
    request_id: str,
) -> int:
    """Count released logical blocks in a request's block table.

    :param scheduler: Scheduler that owns the request.
    :param request_id: Request whose block table should be inspected.
    :returns: Number of null-block entries.
    """
    manager = scheduler.kv_cache_manager.coordinator.single_type_managers[0]
    null_block = manager._null_block
    return sum(block is null_block for block in manager.req_to_blocks[request_id])


def test_num_in_flight_tokens_accounting() -> None:
    scheduler = create_scheduler(async_scheduling=True)
    request = create_requests(num_requests=1, num_tokens=NUM_PROMPT_TOKENS)[0]
    scheduler.add_request(request)

    first_output = scheduler.schedule()
    assert request.num_in_flight_tokens == NUM_PROMPT_TOKENS

    second_output = scheduler.schedule()
    assert request.num_in_flight_tokens == NUM_PROMPT_TOKENS + 1

    scheduler.update_from_output(
        first_output,
        _make_model_runner_output(first_output),
    )
    assert request.num_in_flight_tokens == 1

    scheduler.update_from_output(
        second_output,
        _make_model_runner_output(second_output),
    )
    assert request.num_in_flight_tokens == 0


def test_swa_free_waits_for_in_flight_step() -> None:
    scheduler = _create_swa_scheduler(async_scheduling=True)
    request = create_requests(
        num_requests=1,
        num_tokens=NUM_PROMPT_TOKENS,
        block_size=BLOCK_SIZE,
    )[0]
    scheduler.add_request(request)
    block_pool = scheduler.kv_cache_manager.block_pool

    prefill_output = scheduler.schedule()
    free_after_prefill = block_pool.get_num_free_blocks()

    decode_output = scheduler.schedule()
    assert _num_null_blocks(scheduler, request.request_id) == 0
    assert block_pool.get_num_free_blocks() == free_after_prefill

    scheduler.update_from_output(
        prefill_output,
        _make_model_runner_output(prefill_output),
    )
    scheduler.schedule()
    assert _num_null_blocks(scheduler, request.request_id) == NUM_OUT_OF_WINDOW_BLOCKS
    assert (
        block_pool.get_num_free_blocks()
        == free_after_prefill + NUM_OUT_OF_WINDOW_BLOCKS
    )

    scheduler.update_from_output(
        decode_output,
        _make_model_runner_output(decode_output),
    )
    scheduler.schedule()
    assert _num_null_blocks(scheduler, request.request_id) == NUM_OUT_OF_WINDOW_BLOCKS


def test_swa_free_remains_immediate_when_synchronous() -> None:
    scheduler = _create_swa_scheduler(async_scheduling=False)
    request = create_requests(
        num_requests=1,
        num_tokens=NUM_PROMPT_TOKENS,
        block_size=BLOCK_SIZE,
    )[0]
    scheduler.add_request(request)

    prefill_output = scheduler.schedule()
    scheduler.update_from_output(
        prefill_output,
        _make_model_runner_output(prefill_output),
    )
    assert request.num_in_flight_tokens == 0

    scheduler.schedule()
    assert _num_null_blocks(scheduler, request.request_id) == NUM_OUT_OF_WINDOW_BLOCKS


def test_swa_admission_cap_accounts_for_overlapping_batches() -> None:
    spec = SlidingWindowSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=1024,
    )

    assert (
        spec.max_admission_blocks_per_request(
            max_in_flight_tokens=1024,
            max_model_len=16384,
        )
        == 129
    )
    assert (
        spec.max_admission_blocks_per_request(
            max_in_flight_tokens=2048,
            max_model_len=16384,
        )
        == 193
    )


def test_chunked_local_free_waits_for_in_flight_step() -> None:
    scheduler = _create_chunked_scheduler(async_scheduling=True)
    request = create_requests(
        num_requests=1,
        num_tokens=NUM_PROMPT_TOKENS,
        block_size=BLOCK_SIZE,
    )[0]
    scheduler.add_request(request)
    block_pool = scheduler.kv_cache_manager.block_pool

    prefill_output = scheduler.schedule()
    free_after_prefill = block_pool.get_num_free_blocks()

    decode_output = scheduler.schedule()
    assert _num_null_blocks(scheduler, request.request_id) == 0
    assert block_pool.get_num_free_blocks() == free_after_prefill

    scheduler.update_from_output(
        prefill_output,
        _make_model_runner_output(prefill_output),
    )
    scheduler.schedule()
    assert _num_null_blocks(scheduler, request.request_id) == NUM_OUT_OF_CHUNK_BLOCKS
    assert (
        block_pool.get_num_free_blocks() == free_after_prefill + NUM_OUT_OF_CHUNK_BLOCKS
    )

    scheduler.update_from_output(
        decode_output,
        _make_model_runner_output(decode_output),
    )
    scheduler.schedule()
    assert _num_null_blocks(scheduler, request.request_id) == NUM_OUT_OF_CHUNK_BLOCKS


def test_chunked_local_admission_cap_accounts_for_overlapping_batches() -> None:
    spec = ChunkedLocalAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        attention_chunk_size=1024,
    )

    assert (
        spec.max_admission_blocks_per_request(
            max_in_flight_tokens=1024,
            max_model_len=16384,
        )
        == 128
    )
    assert (
        spec.max_admission_blocks_per_request(
            max_in_flight_tokens=2048,
            max_model_len=16384,
        )
        == 192
    )


def test_connector_finish_frees_on_settled_basis() -> None:
    scheduler = create_scheduler(
        block_size=BLOCK_SIZE,
        async_scheduling=True,
        use_kv_connector=mock_kv(matched_tokens=0, is_async=False),
        kv_cache_spec=SlidingWindowSpec(
            block_size=BLOCK_SIZE,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.float32,
            sliding_window=SLIDING_WINDOW,
        ),
    )
    request = create_requests(
        num_requests=1,
        num_tokens=NUM_PROMPT_TOKENS,
        block_size=BLOCK_SIZE,
    )[0]
    scheduler.add_request(request)

    prefill_output = scheduler.schedule()
    scheduler.schedule()

    scheduler._connector_finished(request)
    assert _num_null_blocks(scheduler, request.request_id) == 0

    scheduler.update_from_output(
        prefill_output,
        _make_model_runner_output(prefill_output),
    )
    scheduler._connector_finished(request)
    assert _num_null_blocks(scheduler, request.request_id) == NUM_OUT_OF_WINDOW_BLOCKS
