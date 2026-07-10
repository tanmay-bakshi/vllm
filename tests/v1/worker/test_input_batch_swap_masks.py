"""swap_states must not alias per-request state through tensor views.

Regression for the tuple-swap aliasing bug: swapping two rows of
allowed_token_ids_mask_cpu_tensor with tuple assignment goes through
views, so the first assignment rewrites the storage the second RHS
still reads and BOTH rows end up equal to the original i2 row — after
any attention-driven batch reorder one request silently applies
another's allowed-token mask.
"""
import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch


def _req(req_id: str, allowed: list[int]) -> CachedRequestState:
    return CachedRequestState(
        req_id=req_id,
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(allowed_token_ids=allowed),
        pooling_params=None,
        mm_features=[],
        block_ids=([],),
        generator=None,
        num_computed_tokens=0,
        output_token_ids=[],
    )


def test_swap_states_allowed_token_masks_stay_distinct():
    batch = InputBatch(
        max_num_reqs=2,
        max_model_len=64,
        max_num_batched_tokens=64,
        device=torch.device("cpu"),
        vocab_size=32,
        block_sizes=[1],
        kernel_block_sizes=[1],
    )
    batch.add_request(_req("a", [1, 2]))
    batch.add_request(_req("b", [3]))
    mask = batch.allowed_token_ids_mask_cpu_tensor
    assert mask is not None
    row_a, row_b = mask[0].clone(), mask[1].clone()
    assert not torch.equal(row_a, row_b)

    batch.swap_states(0, 1)

    assert torch.equal(mask[0], row_b), "row 0 must carry b's mask"
    assert torch.equal(mask[1], row_a), "row 1 must carry a's mask"
