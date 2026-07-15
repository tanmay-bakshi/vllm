# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from unittest.mock import patch

import pytest

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.engine.core import EngineCore, _DFlashCalibrationDiagnostics


def _scheduler_output(
    *,
    query_len: int,
    batch_size: int,
    sequence_length: int,
    non_verification_rows: int = 0,
    padded_rows: int = 0,
) -> SchedulerOutput:
    """Build a scheduler output with explicit DFlash provenance.

    :param query_len: Target query length executed by the pass.
    :param batch_size: Number of target-verification rows.
    :param sequence_length: Longest ending verification sequence.
    :param non_verification_rows: Additional scheduled prefill rows.
    :param padded_rows: Verification rows without a real proposal.
    :returns: Synthetic scheduler output for diagnostic tests.
    """
    output = SchedulerOutput.make_empty()
    output.dflash_verification_query_len = query_len
    output.dflash_verification_batch_size = batch_size
    output.dflash_verification_max_sequence_length = sequence_length
    total_rows = batch_size + non_verification_rows
    output.num_scheduled_tokens = {str(index): query_len for index in range(total_rows)}
    output.dflash_padded_request_ids = {str(index) for index in range(padded_rows)}
    return output


def test_dflash_calibration_diagnostics_aggregate_exact_operating_points() -> None:
    """Aggregate mixed operating points and reset after each interval."""
    diagnostics = _DFlashCalibrationDiagnostics(interval=2)

    boundary = _scheduler_output(
        query_len=16,
        batch_size=8,
        sequence_length=80,
    )
    first_counted = _scheduler_output(
        query_len=8,
        batch_size=2,
        sequence_length=100,
        non_verification_rows=1,
        padded_rows=1,
    )
    second = _scheduler_output(
        query_len=4,
        batch_size=1,
        sequence_length=50,
    )

    assert diagnostics.observe(boundary, completed_ns=100) is None
    assert diagnostics.rounds == 0
    assert diagnostics.observe(first_counted, completed_ns=200) is None
    rendered = diagnostics.observe(second, completed_ns=350)
    assert rendered is not None
    assert json.loads(rendered) == {
        "elapsed_ns": 250,
        "interval_index": 1,
        "non_dflash_rounds": 0,
        "points": [
            {
                "contaminated_rounds": 0,
                "max_l": 50,
                "min_l": 50,
                "non_verification_rows": 0,
                "padded_rows": 0,
                "q": 4,
                "r": 1,
                "rounds": 1,
            },
            {
                "contaminated_rounds": 1,
                "max_l": 100,
                "min_l": 100,
                "non_verification_rows": 1,
                "padded_rows": 1,
                "q": 8,
                "r": 2,
                "rounds": 1,
            },
        ],
        "rounds": 2,
        "schema_version": 2,
    }

    diagnostics.begin_interval(400)
    assert (
        diagnostics.observe(
            _scheduler_output(
                query_len=16,
                batch_size=8,
                sequence_length=200,
            ),
            completed_ns=500,
        )
        is None
    )
    assert (
        diagnostics.observe(
            _scheduler_output(
                query_len=0,
                batch_size=0,
                sequence_length=0,
            ),
            completed_ns=550,
        )
        is None
    )
    next_rendered = diagnostics.observe(
        _scheduler_output(
            query_len=16,
            batch_size=8,
            sequence_length=240,
        ),
        completed_ns=650,
    )
    assert next_rendered is not None
    next_summary = json.loads(next_rendered)
    assert next_summary["interval_index"] == 2
    assert next_summary["rounds"] == 2
    assert next_summary["elapsed_ns"] == 250
    assert next_summary["non_dflash_rounds"] == 1
    assert next_summary["points"] == [
        {
            "contaminated_rounds": 0,
            "max_l": 240,
            "min_l": 200,
            "non_verification_rows": 0,
            "padded_rows": 0,
            "q": 16,
            "r": 8,
            "rounds": 2,
        }
    ]


def test_dflash_calibration_diagnostics_count_active_non_dflash_passes() -> None:
    """Count non-DFlash passes only while a timed interval is active."""
    diagnostics = _DFlashCalibrationDiagnostics(interval=1)

    assert (
        diagnostics.observe(
            _scheduler_output(
                query_len=0,
                batch_size=0,
                sequence_length=0,
            ),
            completed_ns=100,
        )
        is None
    )
    assert diagnostics.rounds == 0
    assert diagnostics.non_dflash_rounds == 0

    assert (
        diagnostics.observe(
            _scheduler_output(
                query_len=16,
                batch_size=8,
                sequence_length=100,
            ),
            completed_ns=200,
        )
        is None
    )
    assert (
        diagnostics.observe(
            _scheduler_output(
                query_len=0,
                batch_size=0,
                sequence_length=0,
            ),
            completed_ns=250,
        )
        is None
    )
    rendered = diagnostics.observe(
        _scheduler_output(
            query_len=16,
            batch_size=8,
            sequence_length=120,
        ),
        completed_ns=300,
    )
    assert rendered is not None
    summary = json.loads(rendered)
    assert summary["elapsed_ns"] == 100
    assert summary["non_dflash_rounds"] == 1


def test_engine_core_dflash_calibration_is_log_inert_when_disabled() -> None:
    """Avoid calibration clocks and logger calls when diagnostics are disabled."""
    engine_core = object.__new__(EngineCore)
    engine_core._dflash_calibration_diagnostics = None

    with (
        patch("vllm.v1.engine.core.logger.info") as logger_info,
        patch("vllm.v1.engine.core.time.perf_counter_ns") as perf_counter_ns,
    ):
        engine_core._record_dflash_calibration(
            _scheduler_output(
                query_len=16,
                batch_size=8,
                sequence_length=2048,
            )
        )

    logger_info.assert_not_called()
    perf_counter_ns.assert_not_called()


def test_engine_core_dflash_calibration_restarts_after_logging() -> None:
    """Restart timing after logging and count the next completed pass."""
    engine_core = object.__new__(EngineCore)
    engine_core._dflash_calibration_diagnostics = _DFlashCalibrationDiagnostics(
        interval=1
    )
    events: list[str] = []
    restart_timestamps = iter((300, 500))

    def record_log(*args: object) -> None:
        """Record one logger call.

        :param args: Logger arguments under test.
        """
        events.append("log")

    def next_restart_timestamp() -> int:
        """Return the next explicit post-log boundary.

        :returns: Injected monotonic timestamp in nanoseconds.
        """
        events.append("clock")
        return next(restart_timestamps)

    with (
        patch("vllm.v1.engine.core.logger.info", side_effect=record_log) as logger_info,
        patch(
            "vllm.v1.engine.core.time.perf_counter_ns",
            side_effect=next_restart_timestamp,
        ),
    ):
        engine_core._record_dflash_calibration(
            _scheduler_output(
                query_len=12,
                batch_size=4,
                sequence_length=8000,
            ),
            completed_ns=100,
        )
        engine_core._record_dflash_calibration(
            _scheduler_output(
                query_len=12,
                batch_size=4,
                sequence_length=8192,
            ),
            completed_ns=200,
        )
        engine_core._record_dflash_calibration(
            _scheduler_output(
                query_len=12,
                batch_size=4,
                sequence_length=8200,
            ),
            completed_ns=400,
        )

    assert events == ["log", "clock", "log", "clock"]
    assert logger_info.call_count == 2
    message, rendered = logger_info.call_args_list[0].args
    assert message == "DFlash calibration interval: %s"
    first_summary = json.loads(rendered)
    assert first_summary["elapsed_ns"] == 100
    assert first_summary["non_dflash_rounds"] == 0
    assert first_summary["points"] == [
        {
            "contaminated_rounds": 0,
            "max_l": 8192,
            "min_l": 8192,
            "non_verification_rows": 0,
            "padded_rows": 0,
            "q": 12,
            "r": 4,
            "rounds": 1,
        }
    ]
    _, next_rendered = logger_info.call_args_list[1].args
    next_summary = json.loads(next_rendered)
    assert next_summary["elapsed_ns"] == 100
    assert next_summary["points"][0]["max_l"] == 8200


def test_dflash_calibration_diagnostics_reject_non_positive_elapsed_time() -> None:
    """Reject a completed interval without positive elapsed time."""
    diagnostics = _DFlashCalibrationDiagnostics(interval=1)
    scheduler_output = _scheduler_output(
        query_len=16,
        batch_size=8,
        sequence_length=2048,
    )

    assert diagnostics.observe(scheduler_output, completed_ns=100) is None
    with pytest.raises(ValueError, match="elapsed time must be positive"):
        diagnostics.observe(scheduler_output, completed_ns=100)


def test_dflash_calibration_diagnostics_reject_non_positive_interval() -> None:
    """Reject intervals that cannot emit a bounded summary."""
    with pytest.raises(ValueError, match="must be positive"):
        _DFlashCalibrationDiagnostics(interval=0)
