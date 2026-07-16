# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import cast

import pytest
import torch

from vllm.config.speculative import (
    DFlashAdaptiveVerificationConfig,
    DFlashTargetQueryLen,
    DFlashVerificationCost,
)
from vllm.v1.spec_decode.dflash import (
    make_empty_dflash_proposal,
    select_dflash_verification_prefix,
)
from vllm.v1.spec_decode.dynamic.dflash_policy import (
    DFlashAdaptiveVerificationPolicy,
)


def _make_config(
    costs_ms: dict[int, float],
    *,
    exploration_interval: int = 256,
    switch_threshold: float = 0.03,
) -> DFlashAdaptiveVerificationConfig:
    return DFlashAdaptiveVerificationConfig(
        costs=[
            DFlashVerificationCost(
                batch_size_range=(1, 32),
                sequence_length_range=(1, 1024),
                query_len=cast(DFlashTargetQueryLen, query_len),
                round_cost_ms=round_cost_ms,
            )
            for query_len, round_cost_ms in costs_ms.items()
        ],
        initial_acceptance_rates=[1.0] * 15,
        acceptance_ema_alpha=1.0,
        switch_threshold=switch_threshold,
        exploration_interval=exploration_interval,
    )


def test_policy_maximizes_expected_output_tokens_per_round_cost() -> None:
    policy = DFlashAdaptiveVerificationPolicy(
        _make_config({4: 2.0, 8: 3.0, 12: 8.0, 16: 10.0}),
        max_batch_size=32,
        max_sequence_length=1024,
    )

    assert policy.select_query_len(8, 128, 128) == 8


def test_policy_updates_prefix_acceptance_without_observing_the_tail() -> None:
    policy = DFlashAdaptiveVerificationPolicy(
        _make_config({4: 2.0, 8: 3.0, 12: 8.0, 16: 10.0}),
        max_batch_size=32,
        max_sequence_length=1024,
    )
    assert policy.select_query_len(8, 128, 128) == 8

    policy.observe(
        batch_size=8,
        min_sequence_length=128,
        max_sequence_length=128,
        query_len=8,
        num_accepted_draft_tokens=0,
    )

    assert policy.select_query_len(8, 128, 128) == 4


def test_policy_aggregates_conditional_hazards_once_per_batch() -> None:
    policy = DFlashAdaptiveVerificationPolicy(
        _make_config({4: 2.0, 8: 3.0, 12: 8.0, 16: 10.0}),
        max_batch_size=32,
        max_sequence_length=1024,
    )

    policy.observe_batch(
        batch_size=3,
        min_sequence_length=128,
        max_sequence_length=128,
        query_len=4,
        accepted_and_observed_draft_tokens=[(3, 3), (0, 3), (0, 0)],
    )

    tier = policy._get_tier(3, 128)
    assert tier is not None
    assert tier.conditional_acceptance_rates[:4] == [0.5, 1.0, 1.0, 1.0]


def test_policy_censors_unexecuted_tail_after_full_prefix_acceptance() -> None:
    policy = DFlashAdaptiveVerificationPolicy(
        _make_config({4: 2.0, 8: 3.0, 12: 8.0, 16: 10.0}),
        max_batch_size=32,
        max_sequence_length=1024,
    )

    policy.observe(
        batch_size=1,
        min_sequence_length=128,
        max_sequence_length=128,
        query_len=4,
        num_accepted_draft_tokens=3,
    )

    tier = policy._get_tier(1, 128)
    assert tier is not None
    assert tier.conditional_acceptance_rates == [1.0] * 15


def test_policy_periodically_probes_the_complete_draft_block() -> None:
    policy = DFlashAdaptiveVerificationPolicy(
        _make_config(
            {4: 2.0, 8: 3.0, 12: 8.0, 16: 10.0},
            exploration_interval=2,
        ),
        max_batch_size=32,
        max_sequence_length=1024,
    )

    assert policy.select_query_len(8, 128, 128) == 8
    assert policy.select_query_len(8, 128, 128) == 16
    assert policy.select_query_len(8, 128, 128) == 8


def test_policy_respects_a_static_query_length_ceiling() -> None:
    policy = DFlashAdaptiveVerificationPolicy(
        _make_config({4: 2.0, 8: 3.0, 12: 8.0, 16: 10.0}),
        max_batch_size=32,
        max_sequence_length=1024,
    )

    assert policy.select_query_len(8, 128, 128, max_query_len=4) == 4


def test_policy_uses_static_fallback_outside_measured_cost_surface() -> None:
    config = DFlashAdaptiveVerificationConfig(
        costs=[
            DFlashVerificationCost(
                batch_size_range=(1, 16),
                sequence_length_range=(1, 1024),
                query_len=cast(DFlashTargetQueryLen, query_len),
                round_cost_ms=float(query_len),
            )
            for query_len in (4, 8, 12, 16)
        ],
        initial_acceptance_rates=[0.5] * 15,
    )

    policy = DFlashAdaptiveVerificationPolicy(
        config,
        max_batch_size=32,
        max_sequence_length=1024,
    )

    assert policy.select_query_len(17, 128, 128) == 16
    assert policy.select_query_len(17, 128, 128, max_query_len=8) == 8


def test_policy_falls_back_and_skips_learning_across_a_tier_boundary() -> None:
    config = DFlashAdaptiveVerificationConfig(
        costs=[
            DFlashVerificationCost(
                batch_size_range=(1, 32),
                sequence_length_range=sequence_length_range,
                query_len=cast(DFlashTargetQueryLen, query_len),
                round_cost_ms=round_cost_ms,
            )
            for sequence_length_range, costs_ms in (
                ((1, 127), (2.0, 3.0, 8.0, 10.0)),
                ((128, 1024), (1.0, 4.0, 8.0, 12.0)),
            )
            for query_len, round_cost_ms in zip(
                (4, 8, 12, 16),
                costs_ms,
                strict=True,
            )
        ],
        initial_acceptance_rates=[1.0] * 15,
        acceptance_ema_alpha=1.0,
    )
    policy = DFlashAdaptiveVerificationPolicy(
        config,
        max_batch_size=32,
        max_sequence_length=1024,
    )

    assert policy.select_query_len(8, 127, 128) == 16
    policy.observe(
        batch_size=8,
        min_sequence_length=127,
        max_sequence_length=128,
        query_len=16,
        num_accepted_draft_tokens=0,
    )

    lower_tier = policy._get_tier(8, 127)
    upper_tier = policy._get_tier(8, 128)
    assert lower_tier is not None
    assert upper_tier is not None
    assert lower_tier.conditional_acceptance_rates == [1.0] * 15
    assert upper_tier.conditional_acceptance_rates == [1.0] * 15


def test_policy_rejects_overlapping_cost_tiers() -> None:
    costs = [
        DFlashVerificationCost(
            batch_size_range=batch_size_range,
            sequence_length_range=(1, 1024),
            query_len=cast(DFlashTargetQueryLen, query_len),
            round_cost_ms=float(query_len),
        )
        for batch_size_range in ((1, 16), (16, 32))
        for query_len in (4, 8, 12, 16)
    ]
    config = DFlashAdaptiveVerificationConfig(
        costs=costs,
        initial_acceptance_rates=[0.5] * 15,
    )

    with pytest.raises(ValueError, match="must not overlap"):
        DFlashAdaptiveVerificationPolicy(
            config,
            max_batch_size=32,
            max_sequence_length=1024,
        )


def test_config_requires_every_query_length_in_each_cost_tier() -> None:
    with pytest.raises(ValueError, match="must define query lengths"):
        DFlashAdaptiveVerificationConfig(
            costs=[
                DFlashVerificationCost(
                    batch_size_range=batch_size_range,
                    sequence_length_range=(1, 1024),
                    query_len=cast(DFlashTargetQueryLen, query_len),
                    round_cost_ms=float(query_len),
                )
                for batch_size_range, query_len in (
                    ((1, 16), 4),
                    ((1, 16), 8),
                    ((1, 16), 12),
                    ((17, 32), 16),
                )
            ],
            initial_acceptance_rates=[0.5] * 15,
        )


def test_config_rejects_duplicate_query_length_costs() -> None:
    costs = [
        DFlashVerificationCost(
            batch_size_range=(1, 32),
            sequence_length_range=(1, 1024),
            query_len=cast(DFlashTargetQueryLen, query_len),
            round_cost_ms=float(query_len),
        )
        for query_len in (4, 8, 12, 16, 16)
    ]

    with pytest.raises(ValueError, match="Duplicate DFlash cost"):
        DFlashAdaptiveVerificationConfig(
            costs=costs,
            initial_acceptance_rates=[0.5] * 15,
        )


def test_config_rejects_non_monotonic_acceptance_priors() -> None:
    with pytest.raises(ValueError, match="must be non-increasing"):
        DFlashAdaptiveVerificationConfig(
            costs=_make_config({4: 2.0, 8: 3.0, 12: 8.0, 16: 10.0}).costs,
            initial_acceptance_rates=[0.5, 0.6] + [0.5] * 13,
        )


@pytest.mark.parametrize("num_speculative_tokens", [3, 7, 11, 15])
def test_dflash_prefix_preserves_fixed_block_storage_stride(
    num_speculative_tokens: int,
) -> None:
    full_draft = torch.arange(30).view(2, 15)

    prefix = select_dflash_verification_prefix(full_draft, num_speculative_tokens)

    assert prefix.shape == (2, num_speculative_tokens)
    assert prefix.stride(0) == 15
    torch.testing.assert_close(prefix, full_draft[:, :num_speculative_tokens])


def test_empty_dflash_proposal_contains_no_token_guesses() -> None:
    proposal = make_empty_dflash_proposal(2, torch.device("cpu"))

    assert proposal.shape == (2, 0)
    assert proposal.dtype == torch.int32
