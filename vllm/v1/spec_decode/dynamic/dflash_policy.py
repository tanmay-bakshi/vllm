# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

from vllm.config.speculative import DFlashAdaptiveVerificationConfig
from vllm.v1.spec_decode.utils import unconditional_to_conditional_rates


@dataclass
class _VerificationTier:
    """Runtime state for one measured batch-size and sequence-length tier.

    :ivar batch_size_range: Inclusive decode batch-size range.
    :ivar sequence_length_range: Inclusive longest target-pass starting
        sequence-length range.
    :ivar costs_ms: Total measured round cost by target query length.
    :ivar conditional_acceptance_rates: Conditional acceptance estimate by draft
        position, given that the preceding draft prefix was accepted.
    :ivar current_query_len: Query length selected outside exploration rounds.
    :ivar num_decisions: Number of decisions made in this tier.
    :ivar last_full_probe: Decision index of the most recent full-prefix probe.
    """

    batch_size_range: tuple[int, int]
    sequence_length_range: tuple[int, int]
    costs_ms: dict[int, float]
    conditional_acceptance_rates: list[float]
    current_query_len: int = 16
    num_decisions: int = 0
    last_full_probe: int = 0

    def contains(self, batch_size: int, sequence_length: int) -> bool:
        """Return whether this tier owns the runtime operating point.

        :param batch_size: Number of scheduled requests.
        :param sequence_length: Longest target-pass starting sequence length.
        :returns: Whether both values are inside the tier's inclusive ranges.
        """
        return (
            self.batch_size_range[0] <= batch_size <= self.batch_size_range[1]
            and self.sequence_length_range[0]
            <= sequence_length
            <= self.sequence_length_range[1]
        )


class DFlashAdaptiveVerificationPolicy:
    """Choose a lossless DFlash target prefix from measured round utility."""

    def __init__(
        self,
        config: DFlashAdaptiveVerificationConfig,
        max_batch_size: int,
        max_sequence_length: int,
    ) -> None:
        """Initialize a policy from offline costs and acceptance priors.

        :param config: Cost-aware adaptive verification configuration.
        :param max_batch_size: Largest scheduler batch the table must cover.
        :param max_sequence_length: Largest target-pass starting sequence the
            table may cover.
        """
        self.config = config
        tiers: dict[tuple[tuple[int, int], tuple[int, int]], _VerificationTier] = {}
        for cost in config.costs:
            key = (cost.batch_size_range, cost.sequence_length_range)
            tier = tiers.get(key)
            if tier is None:
                tier = _VerificationTier(
                    batch_size_range=cost.batch_size_range,
                    sequence_length_range=cost.sequence_length_range,
                    costs_ms={},
                    conditional_acceptance_rates=unconditional_to_conditional_rates(
                        config.initial_acceptance_rates
                    ),
                )
                tiers[key] = tier
            tier.costs_ms[cost.query_len] = cost.round_cost_ms
        self._tiers = tuple(tiers.values())
        self._validate_runtime_partitions(max_batch_size, max_sequence_length)

    def _validate_runtime_partitions(
        self, max_batch_size: int, max_sequence_length: int
    ) -> None:
        batch_points = {1, max_batch_size}
        sequence_points = {1, max_sequence_length}
        for tier in self._tiers:
            for point in (tier.batch_size_range[0], tier.batch_size_range[1] + 1):
                if point <= max_batch_size:
                    batch_points.add(point)
            for point in (
                tier.sequence_length_range[0],
                tier.sequence_length_range[1] + 1,
            ):
                if point <= max_sequence_length:
                    sequence_points.add(point)

        for batch_size in batch_points:
            for sequence_length in sequence_points:
                matches = [
                    tier
                    for tier in self._tiers
                    if tier.contains(batch_size, sequence_length)
                ]
                if len(matches) > 1:
                    raise ValueError(
                        "DFlash verification cost tiers must not overlap; "
                        f"batch_size={batch_size}, sequence_length={sequence_length} "
                        f"matched {len(matches)} tiers."
                    )

    def _get_tier(
        self, batch_size: int, sequence_length: int
    ) -> _VerificationTier | None:
        matches = [
            tier for tier in self._tiers if tier.contains(batch_size, sequence_length)
        ]
        if len(matches) > 1:
            raise RuntimeError(
                "DFlash verification policy has overlapping cost tiers for "
                f"batch_size={batch_size}, sequence_length={sequence_length}."
            )
        return matches[0] if len(matches) == 1 else None

    def _get_tier_for_range(
        self,
        batch_size: int,
        min_sequence_length: int,
        max_sequence_length: int,
    ) -> _VerificationTier | None:
        """Return the one tier containing an entire runtime length interval.

        :param batch_size: Number of target-verification requests.
        :param min_sequence_length: Lower bound on the longest target-pass
            starting sequence.
        :param max_sequence_length: Upper bound on the longest target-pass
            starting sequence.
        :returns: The containing tier, or ``None`` when the interval is
            uncovered or crosses a tier boundary.
        :raises ValueError: If the sequence-length bounds are reversed.
        """
        if min_sequence_length > max_sequence_length:
            raise ValueError("DFlash sequence-length bounds must be ordered.")
        min_tier = self._get_tier(batch_size, min_sequence_length)
        max_tier = self._get_tier(batch_size, max_sequence_length)
        return min_tier if min_tier is max_tier else None

    def fallback_query_len(self, max_query_len: int) -> int:
        """Return the configured conservative query length under a ceiling.

        :param max_query_len: Static target-query ceiling.
        :returns: Largest supported fallback query length under the ceiling.
        """
        supported_query_lens = (4, 8, 12, 16)
        candidates = [
            query_len
            for query_len in supported_query_lens
            if query_len <= self.config.fallback_query_len
            and query_len <= max_query_len
        ]
        return max(candidates, default=max_query_len)

    @staticmethod
    def _utility(tier: _VerificationTier, query_len: int) -> float:
        survival_probability = 1.0
        expected_output_tokens = 1.0
        for conditional_rate in tier.conditional_acceptance_rates[: query_len - 1]:
            survival_probability *= conditional_rate
            expected_output_tokens += survival_probability
        return expected_output_tokens / tier.costs_ms[query_len]

    def select_query_len(
        self,
        batch_size: int,
        min_sequence_length: int,
        max_sequence_length: int,
        max_query_len: int = 16,
    ) -> int:
        """Select the target query length for the current verification pass.

        :param batch_size: Number of requests scheduled in this round.
        :param min_sequence_length: Lower bound on the longest target-pass
            starting sequence in the scheduled batch.
        :param max_sequence_length: Upper bound on the longest target-pass
            starting sequence in the scheduled batch.
        :param max_query_len: Optional static schedule ceiling.
        :returns: Query length chosen from the measured candidates.
        """
        tier = self._get_tier_for_range(
            batch_size,
            min_sequence_length,
            max_sequence_length,
        )
        if tier is None:
            return self.fallback_query_len(max_query_len)
        candidates = sorted(
            query_len for query_len in tier.costs_ms if query_len <= max_query_len
        )
        if len(candidates) == 0:
            return max_query_len

        tier.num_decisions += 1
        full_query_len = candidates[-1]
        if (
            tier.current_query_len != full_query_len
            and tier.num_decisions - tier.last_full_probe
            >= self.config.exploration_interval
        ):
            tier.last_full_probe = tier.num_decisions
            return full_query_len

        utilities = {
            query_len: self._utility(tier, query_len) for query_len in candidates
        }
        best_query_len = max(candidates, key=utilities.__getitem__)
        if tier.current_query_len not in utilities:
            tier.current_query_len = best_query_len
            return best_query_len

        current_utility = utilities[tier.current_query_len]
        required_utility = current_utility * (1.0 + self.config.switch_threshold)
        if utilities[best_query_len] >= required_utility:
            tier.current_query_len = best_query_len
        return tier.current_query_len

    def observe(
        self,
        batch_size: int,
        min_sequence_length: int,
        max_sequence_length: int,
        query_len: int,
        num_accepted_draft_tokens: int,
        num_observed_draft_tokens: int | None = None,
    ) -> None:
        """Update acceptance hazards from one executed prefix.

        :param batch_size: Number of requests in the executed batch.
        :param min_sequence_length: Lower bound on the longest target-pass
            starting sequence in the executed batch.
        :param max_sequence_length: Upper bound on the longest target-pass
            starting sequence in the executed batch.
        :param query_len: Executed target query length.
        :param num_accepted_draft_tokens: Accepted draft-prefix length.
        :param num_observed_draft_tokens: Number of real draft positions in the
            executed prefix. Invalid target-batch padding is excluded.
        """
        observed_draft_tokens = query_len - 1
        if num_observed_draft_tokens is not None:
            observed_draft_tokens = num_observed_draft_tokens
        self.observe_batch(
            batch_size=batch_size,
            min_sequence_length=min_sequence_length,
            max_sequence_length=max_sequence_length,
            query_len=query_len,
            accepted_and_observed_draft_tokens=[
                (num_accepted_draft_tokens, observed_draft_tokens)
            ],
        )

    def observe_batch(
        self,
        batch_size: int,
        min_sequence_length: int,
        max_sequence_length: int,
        query_len: int,
        accepted_and_observed_draft_tokens: list[tuple[int, int]],
    ) -> None:
        """Update conditional acceptance estimates once for a target batch.

        :param batch_size: Number of requests in the executed batch.
        :param min_sequence_length: Lower bound on the longest target-pass
            starting sequence in the executed batch.
        :param max_sequence_length: Upper bound on the longest target-pass
            starting sequence in the executed batch.
        :param query_len: Executed target query length.
        :param accepted_and_observed_draft_tokens: Accepted prefix length and
            real executed draft count for each non-padding row.
        """
        tier = self._get_tier_for_range(
            batch_size,
            min_sequence_length,
            max_sequence_length,
        )
        if tier is None:
            return
        if query_len not in tier.costs_ms:
            return

        num_draft_positions = query_len - 1
        successes = [0] * num_draft_positions
        at_risk = [0] * num_draft_positions
        for num_accepted, num_observed in accepted_and_observed_draft_tokens:
            if not 0 <= num_accepted <= num_observed <= num_draft_positions:
                raise ValueError(
                    "Accepted and observed DFlash draft counts must fit inside "
                    "the executed target prefix."
                )
            for position in range(num_accepted):
                successes[position] += 1
                at_risk[position] += 1
            if num_accepted < num_observed:
                at_risk[num_accepted] += 1

        alpha = self.config.acceptance_ema_alpha
        for position, position_at_risk in enumerate(at_risk):
            if position_at_risk == 0:
                continue
            observation = successes[position] / position_at_risk
            previous = tier.conditional_acceptance_rates[position]
            tier.conditional_acceptance_rates[position] = previous + alpha * (
                observation - previous
            )
        if query_len == max(tier.costs_ms):
            tier.last_full_probe = tier.num_decisions
