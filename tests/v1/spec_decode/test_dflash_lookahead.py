# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tests.v1.core.utils import create_requests
from vllm.config import (
    CacheConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
)
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

# Matches defaults from tests/v1/spec_decode/test_eagle.py
DFLASH_TARGET_DIR = "Qwen/Qwen3-8B"
DFLASH_DRAFT_DIR = "z-lab/Qwen3-8B-DFlash-b16"

BLOCK_SIZE = 16
NUM_BLOCKS = 8
NUM_SPECULATIVE_TOKENS = 3
VARIABLE_NUM_SPECULATIVE_TOKENS = 15


def _dflash_speculative_config(
    num_speculative_tokens: int,
    *,
    variable_verification: bool = False,
) -> SpeculativeConfig:
    model_config = ModelConfig(
        model=DFLASH_TARGET_DIR,
        runner="generate",
        max_model_len=100,
        trust_remote_code=True,
    )
    return SpeculativeConfig(
        target_model_config=model_config,
        target_parallel_config=ParallelConfig(),
        model=DFLASH_DRAFT_DIR,
        method="dflash",
        num_speculative_tokens=num_speculative_tokens,
        num_speculative_tokens_per_batch_size=(
            [(1, 16, num_speculative_tokens)] if variable_verification else None
        ),
    )


def _create_dflash_scheduler(
    num_speculative_tokens: int,
    *,
    variable_verification: bool = False,
    async_scheduling: bool = False,
) -> Scheduler:
    speculative_config = _dflash_speculative_config(
        num_speculative_tokens,
        variable_verification=variable_verification,
    )
    model_config = speculative_config.target_model_config
    scheduler_config = SchedulerConfig(
        max_num_seqs=16,
        max_num_batched_tokens=8192,
        max_model_len=model_config.max_model_len,
        async_scheduling=async_scheduling,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )
    cache_config = CacheConfig(
        block_size=BLOCK_SIZE,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        enable_prefix_caching=False,
    )
    vllm_config = VllmConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=ParallelConfig(),
        speculative_config=speculative_config,
    )
    # The host executor can disable async mode during config verification, while
    # this unit helper directly instantiates the requested scheduler class.
    scheduler_config.async_scheduling = async_scheduling
    num_blocks = 32 if variable_verification else NUM_BLOCKS
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=BLOCK_SIZE,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            )
        ],
    )
    cache_config.num_gpu_blocks = num_blocks
    scheduler_type = AsyncScheduler if async_scheduling else Scheduler
    return scheduler_type(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        block_size=BLOCK_SIZE,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(vllm_config),
    )


def test_dflash_prefill_reserves_lookahead_blocks():
    scheduler = _create_dflash_scheduler(NUM_SPECULATIVE_TOKENS)

    assert scheduler.num_lookahead_tokens == NUM_SPECULATIVE_TOKENS + 1

    (request,) = create_requests(
        num_requests=1,
        num_tokens=BLOCK_SIZE,
        block_size=BLOCK_SIZE,
    )
    scheduler.add_request(request)

    output = scheduler.schedule()

    assert output.num_scheduled_tokens[request.request_id] == BLOCK_SIZE
    # prefill block + one lookahead block
    assert len(output.scheduled_new_reqs[0].block_ids[0]) == 2


def test_dflash_first_prefill_query_window_fits_allocated_blocks():
    scheduler = _create_dflash_scheduler(NUM_SPECULATIVE_TOKENS)

    (request,) = create_requests(
        num_requests=1,
        num_tokens=BLOCK_SIZE,
        block_size=BLOCK_SIZE,
    )
    scheduler.add_request(request)

    output = scheduler.schedule()
    block_ids = output.scheduled_new_reqs[0].block_ids[0]
    query_positions = range(BLOCK_SIZE, BLOCK_SIZE + scheduler.num_lookahead_tokens)

    assert all(pos // BLOCK_SIZE < len(block_ids) for pos in query_positions)


def test_dflash_suppresses_proposal_outside_drafter_window() -> None:
    scheduler = _create_dflash_scheduler(NUM_SPECULATIVE_TOKENS)
    (request,) = create_requests(
        num_requests=1,
        num_tokens=97,
        block_size=BLOCK_SIZE,
    )
    scheduler.add_request(request)

    output = scheduler.schedule()

    assert scheduler.dflash_drafter_max_model_len == 100
    assert output.num_spec_tokens_to_schedule == 0


def test_dflash_drafter_window_includes_incomplete_prefill_rows() -> None:
    scheduler = SimpleNamespace(
        requests={
            "decode": SimpleNamespace(num_computed_tokens=16),
            "prefill": SimpleNamespace(num_computed_tokens=80),
        }
    )

    max_sequence_length = Scheduler._max_dflash_proposal_sequence_length(
        scheduler,
        {"decode": 4, "prefill": 17},
    )

    assert max_sequence_length == 97


def _complete_prefill(
    scheduler: Scheduler,
    output: SchedulerOutput,
) -> None:
    req_ids = list(output.num_scheduled_tokens)
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req_id: index for index, req_id in enumerate(req_ids)},
            sampled_token_ids=[[100] for _ in req_ids],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )


def _make_variable_dflash_decode_batch(
    prompt_lengths: tuple[int, ...],
    *,
    async_scheduling: bool = False,
) -> tuple[Scheduler, list[Request]]:
    scheduler = _create_dflash_scheduler(
        VARIABLE_NUM_SPECULATIVE_TOKENS,
        variable_verification=True,
        async_scheduling=async_scheduling,
    )
    requests: list[Request] = []
    for index, prompt_length in enumerate(prompt_lengths):
        (request,) = create_requests(
            num_requests=1,
            num_tokens=prompt_length,
            block_size=BLOCK_SIZE,
            req_ids=[str(index)],
        )
        requests.append(request)
        scheduler.add_request(request)
    prefill_output = scheduler.schedule()
    _complete_prefill(scheduler, prefill_output)
    return scheduler, requests


@pytest.mark.parametrize("num_draft_tokens", [3, 7, 11, 15])
def test_variable_dflash_pads_q1_rows_to_the_current_target_cohort(
    num_draft_tokens: int,
) -> None:
    scheduler, requests = _make_variable_dflash_decode_batch((16, 16))
    first, second = requests
    real_draft = list(range(num_draft_tokens))
    scheduler.update_draft_token_ids(
        DraftTokenIds(
            [first.request_id, second.request_id],
            [real_draft, []],
        )
    )

    output = scheduler.schedule()

    query_len = num_draft_tokens + 1
    assert output.dflash_verification_query_len == query_len
    assert set(output.num_scheduled_tokens.values()) == {query_len}
    assert output.scheduled_spec_decode_tokens[first.request_id] == real_draft
    assert (
        output.scheduled_spec_decode_tokens[second.request_id]
        == [-1] * num_draft_tokens
    )
    assert output.dflash_num_valid_draft_tokens == {
        first.request_id: num_draft_tokens,
        second.request_id: 0,
    }
    assert output.dflash_padded_request_ids == {second.request_id}


def test_variable_dflash_lowers_a_wider_cohort_for_a_shorter_real_prefix() -> None:
    scheduler, requests = _make_variable_dflash_decode_batch((16, 16))
    first, second = requests
    scheduler.update_draft_token_ids(
        DraftTokenIds(
            [first.request_id, second.request_id],
            [list(range(7)), [7, 8, 9]],
        )
    )

    output = scheduler.schedule()

    assert set(output.num_scheduled_tokens.values()) == {4}
    assert output.scheduled_spec_decode_tokens[first.request_id] == [0, 1, 2]
    assert output.scheduled_spec_decode_tokens[second.request_id] == [7, 8, 9]


def test_dflash_cohort_truncation_does_not_mutate_shared_async_placeholders() -> None:
    shared_placeholders = [-1] * 15
    num_scheduled_tokens = {"first": 16, "second": 16}
    scheduled_spec_decode_tokens = {
        "first": shared_placeholders,
        "second": shared_placeholders,
    }

    refunded_tokens = Scheduler._truncate_dflash_target_batch(
        8,
        num_scheduled_tokens,
        scheduled_spec_decode_tokens,
    )

    assert refunded_tokens == 16
    assert num_scheduled_tokens == {"first": 8, "second": 8}
    assert scheduled_spec_decode_tokens == {
        "first": [-1] * 7,
        "second": [-1] * 7,
    }
    assert shared_placeholders == [-1] * 15


def test_synchronous_dflash_preserves_a_proposal_across_a_scheduling_gap() -> None:
    scheduler, requests = _make_variable_dflash_decode_batch((16, 16))
    first, second = requests
    second_draft = [4, 5, 6, 7, 8, 9, 10]
    scheduler.update_draft_token_ids(
        DraftTokenIds(
            [first.request_id, second.request_id],
            [[1, 2, 3], second_draft],
        )
    )
    scheduler.max_num_scheduled_tokens = 6

    output = scheduler.schedule()

    assert output.num_scheduled_tokens == {first.request_id: 4}
    assert second.spec_token_ids == second_draft


def test_synchronous_dflash_does_not_persist_unadmitted_q1_padding() -> None:
    scheduler, requests = _make_variable_dflash_decode_batch((16, 16))
    first, second = requests
    scheduler.update_draft_token_ids(
        DraftTokenIds(
            [first.request_id, second.request_id],
            [list(range(VARIABLE_NUM_SPECULATIVE_TOKENS)), []],
        )
    )
    scheduler.max_num_scheduled_tokens = 20

    output = scheduler.schedule()

    assert output.num_scheduled_tokens == {first.request_id: 16}
    assert output.dflash_padded_request_ids == set()
    assert second.spec_token_ids == []


def test_async_dflash_reentry_is_q1_then_padded_with_explicit_provenance() -> None:
    scheduler, requests = _make_variable_dflash_decode_batch(
        (16, 16),
        async_scheduling=True,
    )
    first, second = requests
    scheduler.update_draft_token_ids(
        DraftTokenIds(
            [first.request_id, second.request_id],
            [[1, 2, 3], [4, 5, 6]],
        )
    )
    scheduler.max_num_scheduled_tokens = 4

    first_output = scheduler.schedule()

    assert first_output.num_scheduled_tokens == {first.request_id: 4}
    assert second.spec_token_ids == []

    scheduler.max_num_scheduled_tokens = 32
    reentry_output = scheduler.schedule()

    assert set(reentry_output.num_scheduled_tokens.values()) == {16}
    assert reentry_output.dflash_verification_query_len == 16
    assert reentry_output.dflash_num_valid_draft_tokens == {
        first.request_id: 15,
        second.request_id: 0,
    }
    assert reentry_output.dflash_padded_request_ids == {second.request_id}


def test_variable_dflash_never_model_limit_truncates_one_speculative_row() -> None:
    scheduler, requests = _make_variable_dflash_decode_batch((16, 97))
    scheduler.update_draft_token_ids(
        DraftTokenIds(
            [request.request_id for request in requests],
            [[1, 2, 3], [4, 5, 6]],
        )
    )

    output = scheduler.schedule()

    assert set(output.num_scheduled_tokens.values()) == {1}
    assert output.dflash_verification_query_len == 1
    assert output.scheduled_spec_decode_tokens == {}
    assert output.num_spec_tokens_to_schedule == 0


def test_synchronous_structured_dflash_preserves_supported_query_width() -> None:
    scheduler, (request,) = _make_variable_dflash_decode_batch((16,))
    request.structured_output_request = Mock()
    request.structured_output_request.grammar.validate_tokens.return_value = [1, 2, 3]
    scheduler.structured_output_manager = Mock()
    scheduler.structured_output_manager.should_advance.return_value = True

    scheduler.update_draft_token_ids(
        DraftTokenIds([request.request_id], [list(range(7))])
    )

    assert request.spec_token_ids == [1, 2, 3, -1, -1, -1, -1]
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {request.request_id: 8}
    assert output.dflash_num_valid_draft_tokens == {request.request_id: 3}


def test_async_structured_dflash_combines_grammar_validity_provenance() -> None:
    scheduler, (request,) = _make_variable_dflash_decode_batch(
        (16,),
        async_scheduling=True,
    )
    scheduler.update_draft_token_ids(
        DraftTokenIds([request.request_id], [list(range(7))])
    )
    output = scheduler.schedule()
    request.structured_output_request = Mock()
    request.structured_output_request.grammar.validate_tokens.return_value = [
        10,
        11,
        12,
    ]
    scheduler.structured_output_manager = Mock()
    scheduler.structured_output_manager.should_advance.return_value = True

    scheduler.update_draft_token_ids_in_output(
        DraftTokenIds([request.request_id], [[10, 11, 12, 13, 14, 15, 16]]),
        output,
    )

    assert output.scheduled_spec_decode_tokens[request.request_id] == [
        10,
        11,
        12,
        -1,
        -1,
        -1,
        -1,
    ]
    assert output.num_invalid_spec_tokens == {request.request_id: 4}
    assert output.dflash_num_valid_draft_tokens == {request.request_id: 3}


def test_dflash_drafter_window_reserves_bonus_token():
    # DFlash's drafter window is num_spec + 1 (the extra slot is the bonus token),
    # so max_seq_len + num_spec + 1 must stay within the draft model's max len.
    input_fits_in_drafter = GPUModelRunner._input_fits_in_drafter
    dflash_runner = SimpleNamespace(
        num_spec_tokens=NUM_SPECULATIVE_TOKENS,
        effective_drafter_max_model_len=100,
        speculative_config=_dflash_speculative_config(NUM_SPECULATIVE_TOKENS),
    )
    # window = 4, so 96 fits (96 + 4 == 100) but 97 does not (97 + 4 == 101)
    assert input_fits_in_drafter(dflash_runner, SimpleNamespace(max_seq_len=96))
    assert not input_fits_in_drafter(dflash_runner, SimpleNamespace(max_seq_len=97))
    assert not input_fits_in_drafter(dflash_runner, None)  # no metadata

    # Other drafters don't reserve the bonus token, so 97 fits (97 + 3 == 100).
    plain_runner = SimpleNamespace(
        num_spec_tokens=NUM_SPECULATIVE_TOKENS,
        effective_drafter_max_model_len=100,
        speculative_config=SimpleNamespace(use_dflash=lambda: False),
    )
    assert input_fits_in_drafter(plain_runner, SimpleNamespace(max_seq_len=97))
