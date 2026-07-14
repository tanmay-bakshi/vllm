# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from collections.abc import AsyncGenerator, Callable
from threading import Lock
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from vllm.engine.protocol import GenerationStream, StreamingInput
from vllm.outputs import RequestOutput
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.engine import EngineCoreRequest, EngineCoreRequestType
from vllm.v1.engine.async_llm import (
    AsyncLLM,
    _AsyncLLMGenerationStream,
    _await_submission_task,
)
from vllm.v1.engine.core_client import DPAsyncMPClient
from vllm.v1.engine.output_processor import RequestOutputCollector


def make_output(request_id: str, *, finished: bool) -> RequestOutput:
    """Create a minimal generation output.

    :param request_id: Request identifier copied into the output.
    :param finished: Whether this is the request's final output.
    :returns: Minimal request output for stream tests.
    """
    return RequestOutput(
        request_id=request_id,
        prompt="prompt",
        prompt_token_ids=[1],
        prompt_logprobs=None,
        outputs=[],
        finished=finished,
    )


def make_mock_engine() -> AsyncLLM:
    """Create an AsyncLLM mock suitable for stream tests.

    :returns: Engine mock with asynchronous abort support.
    """
    engine = MagicMock(spec=AsyncLLM)
    engine.log_requests = False
    engine.abort = AsyncMock()
    return engine


def make_core_request(*, n: int = 1) -> EngineCoreRequest:
    """Create an EngineCore request for admission tests.

    :param n: Number of parallel sampling children.
    :returns: Request with a stable internal and external identifier.
    """
    return EngineCoreRequest(
        request_id="internal",
        prompt_token_ids=[1],
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=1, n=n),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        external_req_id="external",
    )


def make_admission_engine() -> AsyncLLM:
    """Create an AsyncLLM mock suitable for admission-path tests.

    :returns: Engine mock with frontend and EngineCore boundaries exposed.
    """
    engine = make_mock_engine()
    engine.errored = False
    engine.vllm_config = MagicMock()
    engine.vllm_config.cache_config.kv_sharing_fast_prefill = False
    engine.input_processor = MagicMock()
    engine.output_processor = MagicMock()
    engine.output_processor.abort_requests.return_value = []
    engine.engine_core = MagicMock()
    engine.engine_core.abort_requests_async = AsyncMock()
    engine._run_output_handler = MagicMock()
    engine._add_request = AsyncMock()
    engine._submit_request_batch = AsyncMock()
    engine.add_request = AsyncLLM.add_request.__get__(engine, AsyncLLM)
    return engine


@pytest.mark.asyncio
async def test_generate_admits_request_before_returning_stream() -> None:
    engine = make_mock_engine()
    request_id = "request"
    queue = RequestOutputCollector(RequestOutputKind.DELTA, request_id)
    admission_callback = MagicMock()
    engine.add_request = AsyncMock(return_value=queue)
    engine.generate = AsyncLLM.generate.__get__(engine, AsyncLLM)

    stream = await engine.generate(
        "prompt",
        SamplingParams(max_tokens=1),
        request_id,
        on_engine_admission=admission_callback,
    )

    assert isinstance(stream, GenerationStream)
    engine.add_request.assert_awaited_once()
    assert (
        engine.add_request.await_args.kwargs["on_engine_admission"]
        is admission_callback
    )
    await stream.aclose()


@pytest.mark.asyncio
async def test_single_request_without_callback_uses_fast_submission() -> None:
    engine = make_admission_engine()

    queue = await engine.add_request(
        "external",
        make_core_request(),
        SamplingParams(max_tokens=1),
    )

    engine._add_request.assert_awaited_once()
    engine._submit_request_batch.assert_not_awaited()
    queue.close()


@pytest.mark.asyncio
async def test_single_request_with_callback_uses_authoritative_batch() -> None:
    engine = make_admission_engine()
    admission_callback = MagicMock()

    async def submit_batch(
        requests: list[EngineCoreRequest],
        callback: Callable[[], None],
    ) -> None:
        callback()

    engine._submit_request_batch.side_effect = submit_batch

    queue = await engine.add_request(
        "external",
        make_core_request(),
        SamplingParams(max_tokens=1),
        on_engine_admission=admission_callback,
    )

    engine._add_request.assert_not_awaited()
    engine._submit_request_batch.assert_awaited_once()
    assert len(engine._submit_request_batch.await_args.args[0]) == 1
    admission_callback.assert_called_once_with()
    queue.close()


@pytest.mark.asyncio
async def test_parallel_sampling_uses_one_atomic_batch() -> None:
    engine = make_admission_engine()
    admission_callback = MagicMock()

    async def submit_batch(
        requests: list[EngineCoreRequest],
        callback: Callable[[], None],
    ) -> None:
        callback()

    engine._submit_request_batch.side_effect = submit_batch

    queue = await engine.add_request(
        "external",
        make_core_request(n=3),
        SamplingParams(max_tokens=1, n=3),
        on_engine_admission=admission_callback,
    )

    engine._add_request.assert_not_awaited()
    assert engine.output_processor.add_request.call_count == 3
    engine._submit_request_batch.assert_awaited_once()
    child_requests = engine._submit_request_batch.await_args.args[0]
    assert [request.request_id for request in child_requests] == [
        "0_internal",
        "1_internal",
        "2_internal",
    ]
    admission_callback.assert_called_once_with()
    queue.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("admitted", [False, True])
async def test_batch_failure_aborts_core_only_after_commit(admitted: bool) -> None:
    engine = make_admission_engine()
    internal_ids = ["0_internal", "1_internal"]
    engine.output_processor.abort_requests.return_value = internal_ids
    admission_callback = MagicMock()

    async def fail_batch(
        requests: list[EngineCoreRequest],
        callback: Callable[[], None],
    ) -> None:
        if admitted:
            callback()
        raise RuntimeError("submission failed")

    engine._submit_request_batch.side_effect = fail_batch

    with pytest.raises(RuntimeError, match="submission failed"):
        await engine.add_request(
            "external",
            make_core_request(n=2),
            SamplingParams(max_tokens=1, n=2),
            on_engine_admission=admission_callback,
        )

    engine.output_processor.abort_requests.assert_called_once_with(
        ("internal",),
        internal=True,
    )
    if admitted:
        admission_callback.assert_called_once_with()
        engine.engine_core.abort_requests_async.assert_awaited_once_with(internal_ids)
    else:
        admission_callback.assert_not_called()
        engine.engine_core.abort_requests_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_stats_failure_after_batch_ack_aborts_core_owned_requests() -> None:
    """FIRST_REQ failure after commit still drives AsyncLLM Core cleanup."""
    client = object.__new__(DPAsyncMPClient)
    client.current_wave = 0
    client.client_index = 0
    client.engines_running = False
    client.core_engine = b"engine"
    client.utility_results = {}
    client.outputs_queue = asyncio.Queue()
    client.resources = SimpleNamespace(
        engine_dead=False,
        output_error=None,
        stats_update_error=None,
        utility_results_lock=Lock(),
    )
    client._ensure_stats_update_task = MagicMock()
    client.get_core_engine_for_requests = MagicMock(return_value=client.core_engine)
    client._submit_request_batch = AsyncMock()
    client._send_input = AsyncMock()
    control_error = RuntimeError("coordinator-control task failed")

    async def fail_first_request(_message: bytes) -> None:
        client._latch_stats_update_error(control_error)
        raise control_error

    client._send_coordinator_control = AsyncMock(side_effect=fail_first_request)

    engine = make_admission_engine()
    engine.engine_core = client
    engine._submit_request_batch = AsyncLLM._submit_request_batch.__get__(
        engine, AsyncLLM
    )
    internal_ids = ["0_internal", "1_internal"]
    engine.output_processor.abort_requests.return_value = internal_ids
    admission_callback = MagicMock()

    with pytest.raises(RuntimeError, match="coordinator-control task failed"):
        await engine.add_request(
            "external",
            make_core_request(n=2),
            SamplingParams(max_tokens=1, n=2),
            on_engine_admission=admission_callback,
        )

    admission_callback.assert_called_once_with()
    client._submit_request_batch.assert_awaited_once()
    client._send_input.assert_awaited_once_with(
        EngineCoreRequestType.ABORT,
        internal_ids,
    )
    assert client.resources.engine_dead is False


@pytest.mark.asyncio
async def test_generation_stream_close_before_iteration_aborts_request() -> None:
    engine = make_mock_engine()
    request_id = "request"
    queue = RequestOutputCollector(RequestOutputKind.DELTA, request_id)
    queue.close = MagicMock(wraps=queue.close)
    stream = _AsyncLLMGenerationStream(engine, queue, request_id)

    await stream.aclose()
    await stream.aclose()

    engine.abort.assert_awaited_once_with(request_id, internal=True)
    queue.close.assert_called_once_with()


@pytest.mark.asyncio
async def test_generation_stream_final_output_does_not_abort_request() -> None:
    engine = make_mock_engine()
    request_id = "request"
    queue = RequestOutputCollector(RequestOutputKind.DELTA, request_id)
    queue.close = MagicMock(wraps=queue.close)
    queue.put(make_output(request_id, finished=True))
    stream = _AsyncLLMGenerationStream(engine, queue, request_id)

    output = await anext(stream)
    assert output.finished
    with pytest.raises(StopAsyncIteration):
        await anext(stream)

    engine.abort.assert_not_awaited()
    queue.close.assert_called_once_with()


@pytest.mark.asyncio
async def test_generation_stream_cancellation_aborts_request() -> None:
    engine = make_mock_engine()
    request_id = "request"
    queue = RequestOutputCollector(RequestOutputKind.DELTA, request_id)
    stream = _AsyncLLMGenerationStream(engine, queue, request_id)
    consumer = asyncio.create_task(anext(stream))
    await asyncio.sleep(0)

    consumer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consumer

    engine.abort.assert_awaited_once_with(request_id, internal=True)


@pytest.mark.asyncio
async def test_submission_resolves_before_caller_cancellation_propagates() -> None:
    submission_started = asyncio.Event()
    release_submission = asyncio.Event()

    async def submit() -> None:
        submission_started.set()
        await release_submission.wait()

    submission_task = asyncio.create_task(submit())
    waiter = asyncio.create_task(_await_submission_task(submission_task))
    await submission_started.wait()

    waiter.cancel()
    await asyncio.sleep(0)
    assert waiter.done() is False
    assert submission_task.cancelled() is False

    release_submission.set()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert submission_task.done()


@pytest.mark.asyncio
async def test_admission_callback_rejected_for_streaming_input() -> None:
    engine = make_mock_engine()
    engine.errored = False
    engine.vllm_config = MagicMock()
    engine.vllm_config.cache_config.kv_sharing_fast_prefill = False
    engine.add_request = AsyncLLM.add_request.__get__(engine, AsyncLLM)
    sampling_params = SamplingParams(max_tokens=1)

    async def input_stream() -> AsyncGenerator[StreamingInput, None]:
        yield StreamingInput(prompt="prompt", sampling_params=sampling_params)

    prompt = input_stream()
    try:
        with pytest.raises(
            ValueError,
            match="on_engine_admission is not supported for streaming-input prompts",
        ):
            await engine.add_request(
                "request",
                prompt,
                sampling_params,
                on_engine_admission=lambda: None,
            )
    finally:
        await prompt.aclose()
