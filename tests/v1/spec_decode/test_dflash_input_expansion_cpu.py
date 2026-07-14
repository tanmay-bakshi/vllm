# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.utils.cpu_triton_utils import copy_and_expand_dflash_inputs_kernel


def test_dflash_input_expansion_masks_rejected_context() -> None:
    """Rejected context must not write KV or affect the next query position."""
    block_size = 4
    num_speculative_tokens = 2
    num_query_per_req = num_speculative_tokens + 1
    target_positions = torch.tensor(
        [
            8,
            9,
            10,
            11,
            20,
            21,
            22,
            23,
            32,
            33,
            34,
            35,
            44,
            45,
            46,
            47,
        ],
        dtype=torch.int64,
    )
    query_start_loc = torch.tensor([0, 4, 8, 12, 16], dtype=torch.int32)
    num_rejected_tokens = torch.tensor([4, 1, 4, 0], dtype=torch.int32)
    block_table = torch.stack(
        [
            torch.arange(100, 113, dtype=torch.int32),
            torch.arange(200, 213, dtype=torch.int32),
            torch.arange(300, 313, dtype=torch.int32),
            torch.arange(400, 413, dtype=torch.int32),
        ]
    )
    next_token_ids = torch.tensor([1000, 2000, 3000, 4000], dtype=torch.int32)

    num_context = target_positions.shape[0]
    num_query = next_token_ids.shape[0] * num_query_per_req
    output_input_ids = torch.full((num_query,), -99, dtype=torch.int32)
    output_context_positions = torch.full((num_context,), -99, dtype=torch.int64)
    output_query_positions = torch.full((num_query,), -99, dtype=torch.int64)
    output_context_slot_mapping = torch.full((num_context,), -99, dtype=torch.int64)
    output_query_slot_mapping = torch.full((num_query,), -99, dtype=torch.int64)
    output_token_indices = torch.full(
        (next_token_ids.shape[0] * num_speculative_tokens,),
        -99,
        dtype=torch.int32,
    )

    copy_and_expand_dflash_inputs_kernel[(next_token_ids.shape[0], 1)](
        next_token_ids_ptr=next_token_ids,
        target_positions_ptr=target_positions,
        out_input_ids_ptr=output_input_ids,
        out_context_positions_ptr=output_context_positions,
        out_query_positions_ptr=output_query_positions,
        out_context_slot_mapping_ptr=output_context_slot_mapping,
        out_query_slot_mapping_ptr=output_query_slot_mapping,
        out_token_indices_ptr=output_token_indices,
        block_table_ptr=block_table,
        block_table_stride=block_table.stride(0),
        query_start_loc_ptr=query_start_loc,
        num_rejected_tokens_ptr=num_rejected_tokens,
        parallel_drafting_token_id=4,
        block_size=block_size,
        num_query_per_req=num_query_per_req,
        num_speculative_tokens=num_speculative_tokens,
        total_input_tokens=num_context,
        BLOCK_SIZE=8,
        HAS_NUM_REJECTED=True,
    )

    assert torch.equal(
        output_context_positions,
        torch.tensor(
            [0, 0, 0, 0, 20, 21, 22, 0, 0, 0, 0, 0, 44, 45, 46, 47],
            dtype=torch.int64,
        ),
    )
    assert torch.equal(
        output_context_slot_mapping,
        torch.tensor(
            [
                -1,
                -1,
                -1,
                -1,
                820,
                821,
                822,
                -1,
                -1,
                -1,
                -1,
                -1,
                1644,
                1645,
                1646,
                1647,
            ],
            dtype=torch.int64,
        ),
    )
    assert torch.equal(
        output_query_positions,
        torch.tensor(
            [8, 9, 10, 23, 24, 25, 32, 33, 34, 48, 49, 50],
            dtype=torch.int64,
        ),
    )
    assert torch.equal(
        output_query_slot_mapping,
        torch.tensor(
            [408, 409, 410, 823, 824, 825, 1232, 1233, 1234, 1648, 1649, 1650],
            dtype=torch.int64,
        ),
    )
    assert torch.equal(
        output_input_ids,
        torch.tensor(
            [1000, 4, 4, 2000, 4, 4, 3000, 4, 4, 4000, 4, 4],
            dtype=torch.int32,
        ),
    )
    assert torch.equal(
        output_token_indices,
        torch.tensor([1, 2, 4, 5, 7, 8, 10, 11], dtype=torch.int32),
    )
