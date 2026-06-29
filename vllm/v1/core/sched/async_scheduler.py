# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)


class AsyncScheduler(Scheduler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # reusable read-only placeholder list for speculative decoding.
        self._spec_token_placeholders: list[int] = [-1] * self.num_spec_tokens
        self._spec_token_placeholders_by_depth: dict[int, list[int]] = {
            self.num_spec_tokens: self._spec_token_placeholders,
        }
        self.adaptive_mtp_enabled = os.environ.get(
            "VLLM_GEMMA4_ADAPTIVE_MTP", "0"
        ) == "1"
        self.adaptive_mtp_long_context_threshold = self._read_positive_int_env(
            "VLLM_GEMMA4_ADAPTIVE_MTP_LONG_CONTEXT_THRESHOLD", 4096
        )
        self.adaptive_mtp_long_context_depth = self._read_positive_int_env(
            "VLLM_GEMMA4_ADAPTIVE_MTP_LONG_CONTEXT_DEPTH", 2
        )
        self.adaptive_mtp_high_batch_threshold = self._read_positive_int_env(
            "VLLM_GEMMA4_ADAPTIVE_MTP_HIGH_BATCH_THRESHOLD", 64
        )
        self.adaptive_mtp_high_batch_depth = self._read_positive_int_env(
            "VLLM_GEMMA4_ADAPTIVE_MTP_HIGH_BATCH_DEPTH", 4
        )

    @staticmethod
    def _read_positive_int_env(name: str, default: int) -> int:
        value = os.environ.get(name)
        if value is None:
            return default
        parsed = int(value)
        if parsed < 1:
            raise ValueError(f"{name} must be >= 1, got {parsed}")
        return parsed

    def _get_async_spec_placeholder_depth(
        self, scheduler_output: SchedulerOutput
    ) -> int:
        if self.num_spec_tokens <= 0:
            return 0
        if self.adaptive_mtp_enabled is False:
            return self.num_spec_tokens

        effective_depth = self.num_spec_tokens
        max_seq_len = 0
        batch_size = 0
        scheduled_tokens_by_request = scheduler_output.num_scheduled_tokens
        for req_id, num_scheduled_tokens in scheduled_tokens_by_request.items():
            request = self.requests[req_id]
            if request.is_prefill_chunk:
                continue
            batch_size += 1
            seq_len = request.num_computed_tokens + num_scheduled_tokens
            max_seq_len = max(max_seq_len, seq_len)

        if max_seq_len >= self.adaptive_mtp_long_context_threshold:
            effective_depth = min(effective_depth, self.adaptive_mtp_long_context_depth)
        if batch_size >= self.adaptive_mtp_high_batch_threshold:
            effective_depth = min(effective_depth, self.adaptive_mtp_high_batch_depth)

        return max(1, min(self.num_spec_tokens, effective_depth))

    def _get_spec_token_placeholders(self, depth: int) -> list[int]:
        placeholders = self._spec_token_placeholders_by_depth.get(depth)
        if placeholders is None:
            placeholders = [-1] * depth
            self._spec_token_placeholders_by_depth[depth] = placeholders
        return placeholders

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        super()._update_after_schedule(scheduler_output)
        spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens
        effective_depth = self._get_async_spec_placeholder_depth(scheduler_output)
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests[req_id]
            if request.is_prefill_chunk:
                continue

            scheduler_output.pending_structured_output_tokens |= (
                request.use_structured_output and request.num_output_placeholders > 0
            )
            # The request will generate a new token plus num_spec_tokens
            # in this scheduling step.
            cur_num_spec_tokens = len(spec_decode_tokens.get(req_id, ()))
            request.num_output_placeholders += 1 + cur_num_spec_tokens
            # Add placeholders for the new draft/spec tokens.
            # We will update the actual spec token ids in the worker process.
            request.spec_token_ids = self._get_spec_token_placeholders(effective_depth)

    def _update_request_with_output(
        self, request: Request, new_token_ids: list[int]
    ) -> tuple[list[int], bool]:
        if request.async_tokens_to_discard > 0:
            # The request was force-preempted in reset_prefix_cache; drop one
            # stale in-flight async output frame per call until the counter
            # is drained.
            request.async_tokens_to_discard -= 1
            return [], False

        status_before_update = request.status
        new_token_ids, stopped = super()._update_request_with_output(
            request, new_token_ids
        )

        # Update the number of output placeholders.
        request.num_output_placeholders -= len(new_token_ids)
        assert request.num_output_placeholders >= 0

        # Cache the new tokens. Preempted requests should be skipped.
        if status_before_update == RequestStatus.RUNNING:
            self.kv_cache_manager.cache_blocks(
                request, request.num_computed_tokens - request.num_output_placeholders
            )
        return new_token_ids, stopped
