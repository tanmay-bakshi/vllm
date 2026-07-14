# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import logging
from collections import deque
from collections.abc import AsyncGenerator, Callable, Coroutine
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai.types.responses import ResponseErrorEvent

from vllm.engine.protocol import GenerationStream
from vllm.entrypoints.openai.chat_completion.batch_serving import (
    OpenAIServingChatBatch,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    BatchChatCompletionRequest,
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
from vllm.entrypoints.openai.engine.serving import (
    GenerationError,
    OpenAIServing,
    _KVTransferAdmission,
    _validate_kv_transfer_request_options,
)
from vllm.entrypoints.openai.responses.context import ConversationContext
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses
from vllm.entrypoints.serve.render.serving import ServingRender
from vllm.inputs import tokens_input
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.utils.async_utils import ManagedAsyncIterator


class _TestGenerationStream(GenerationStream):
    """Record deterministic closure while yielding a finite output sequence."""

    _outputs: list[RequestOutput]
    _index: int
    _closed: bool
    close_count: int

    def __init__(self, outputs: list[RequestOutput]) -> None:
        """Create a test stream.

        :param outputs: Outputs returned before the stream terminates.
        """
        self._outputs = outputs
        self._index = 0
        self._closed = False
        self.close_count = 0

    def __aiter__(self) -> "_TestGenerationStream":
        return self

    async def __anext__(self) -> RequestOutput:
        if self._index >= len(self._outputs):
            raise StopAsyncIteration
        output = self._outputs[self._index]
        self._index += 1
        return output

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.close_count += 1


class _TwoTurnContext(ConversationContext):
    """Request exactly one built-in tool turn during a generation test."""

    output_count: int

    def __init__(self) -> None:
        """Create an empty two-turn context."""
        self.output_count = 0

    def append_output(self, output: RequestOutput) -> None:
        self.output_count += 1

    def append_tool_output(self, output: Any) -> None:
        return

    async def call_tool(self) -> list[Any]:
        return []

    def need_builtin_tool_call(self) -> bool:
        return self.output_count == 1

    def render_for_completion(self) -> list[int]:
        return []

    async def init_tool_sessions(
        self,
        tool_server: Any,
        exit_stack: Any,
        request_id: str,
        mcp_tools: dict[str, Any],
    ) -> None:
        return

    async def cleanup_session(self) -> None:
        return


def _request_output(request_id: str) -> RequestOutput:
    """Create the smallest output needed by lifecycle tests.

    :param request_id: Request identifier stamped on the output.
    :returns: A finished output with no completion choices.
    """
    return RequestOutput(
        request_id=request_id,
        prompt=None,
        prompt_token_ids=[],
        prompt_logprobs=None,
        outputs=[],
        finished=True,
    )


def test_ordinary_openai_request_preserves_generate_fast_path() -> None:
    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hello"}],
    )
    serving = SimpleNamespace(has_kv_connector=True)

    admission = OpenAIServing._create_kv_transfer_admission(
        serving,
        request.kv_transfer_params,
    )

    assert admission.callback is None
    assert admission.engine_owns_offer is False


@pytest.mark.parametrize(
    ("kv_transfer_params", "stream", "n", "expected_error"),
    [
        (
            {"do_remote_decode": True},
            True,
            1,
            "KV-transfer producer requests are not supported with streaming",
        ),
        (
            {"do_remote_decode": True},
            False,
            2,
            "KV-transfer producer requests require n=1",
        ),
        ({"do_remote_prefill": True, "expected_consumers": 2}, True, 2, None),
        ({"do_remote_prefill": True}, True, 1, None),
        (
            {"do_remote_prefill": True, "expected_consumers": 2},
            True,
            1,
            "KV-transfer consumer requests require n to equal "
            "expected_consumers (2), got 1",
        ),
        (
            {"do_remote_prefill": True, "expected_consumers": False},
            False,
            1,
            "KV-transfer consumer requests require a positive integer "
            "expected_consumers",
        ),
    ],
)
def test_kv_transfer_sampling_contract_is_directional(
    kv_transfer_params: dict[str, Any],
    stream: bool,
    n: int,
    expected_error: str | None,
) -> None:
    assert (
        _validate_kv_transfer_request_options(
            kv_transfer_params,
            use_beam_search=False,
            stream=stream,
            n=n,
        )
        == expected_error
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("api", ["chat", "completion"])
@pytest.mark.parametrize("transfer_flag", ["do_remote_decode", "do_remote_prefill"])
async def test_standalone_render_rejects_kv_contract(
    api: str,
    transfer_flag: str,
) -> None:
    if api == "chat":
        request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
            kv_transfer_params={transfer_flag: True},
        )
        render = ServingRender.render_chat_request
    else:
        request = CompletionRequest(
            model="test-model",
            prompt="hello",
            kv_transfer_params={transfer_flag: True},
        )
        render = ServingRender.render_completion_request

    rejection = object()
    serving = SimpleNamespace(
        create_error_response=MagicMock(return_value=rejection),
    )

    result = await render(serving, request)

    assert result is rejection
    serving.create_error_response.assert_called_once_with(
        "kv_transfer_params must be attached to GenerateRequest after rendering"
    )


@pytest.mark.asyncio
async def test_standalone_completion_render_rejects_beam_search() -> None:
    request = CompletionRequest(
        model="test-model",
        prompt="hello",
        use_beam_search=True,
    )
    rejection = object()
    serving = SimpleNamespace(
        _check_model=AsyncMock(return_value=None),
        create_error_response=MagicMock(return_value=rejection),
    )

    result = await ServingRender.render_completion_request(serving, request)

    assert result is rejection
    serving.create_error_response.assert_called_once_with(
        "Beam search is not supported by the render endpoint"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("transfer_flag", ["do_remote_decode", "do_remote_prefill"])
async def test_completion_kv_transfer_contract_rejects_prompt_fanout(
    transfer_flag: str,
) -> None:
    request = CompletionRequest(
        model="test-model",
        prompt=["first", "second"],
        kv_transfer_params={transfer_flag: True},
    )
    rejection = object()
    serving = SimpleNamespace(
        render_completion_request=AsyncMock(
            return_value=[tokens_input([1]), tokens_input([2])]
        ),
        create_error_response=MagicMock(return_value=rejection),
    )

    result = await OpenAIServingCompletion._create_completion(
        serving,
        request,
        _KVTransferAdmission(transfer_flag == "do_remote_prefill"),
    )

    assert result is rejection
    serving.create_error_response.assert_called_once_with(
        "A KV-transfer request requires exactly one completion prompt"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("api", ["chat", "completion"])
@pytest.mark.parametrize("transfer_flag", ["do_remote_decode", "do_remote_prefill"])
async def test_kv_transfer_contract_rejects_beam_search(
    api: str,
    transfer_flag: str,
) -> None:
    if api == "chat":
        request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
            use_beam_search=True,
            kv_transfer_params={transfer_flag: True},
        )
        create = OpenAIServingChat._create_chat_completion
    else:
        request = CompletionRequest(
            model="test-model",
            prompt="hello",
            use_beam_search=True,
            kv_transfer_params={transfer_flag: True},
        )
        create = OpenAIServingCompletion._create_completion

    rejection = object()
    serving = SimpleNamespace(
        create_error_response=MagicMock(return_value=rejection),
    )

    result = await create(
        serving,
        request,
        _KVTransferAdmission(transfer_flag == "do_remote_prefill"),
    )

    assert result is rejection
    serving.create_error_response.assert_called_once_with(
        "KV-transfer requests are not supported with beam search"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("api", ["chat", "completion"])
@pytest.mark.parametrize(
    ("stream", "n", "expected_error"),
    [
        (
            True,
            1,
            "KV-transfer producer requests are not supported with streaming",
        ),
        (False, 2, "KV-transfer producer requests require n=1"),
    ],
)
async def test_direct_kv_producer_rejects_unrepresentable_response_contract(
    api: str,
    stream: bool,
    n: int,
    expected_error: str,
) -> None:
    if api == "chat":
        request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
            stream=stream,
            n=n,
            kv_transfer_params={"do_remote_decode": True},
        )
        create = OpenAIServingChat._create_chat_completion
    else:
        request = CompletionRequest(
            model="test-model",
            prompt="hello",
            stream=stream,
            n=n,
            kv_transfer_params={"do_remote_decode": True},
        )
        create = OpenAIServingCompletion._create_completion

    rejection = object()
    serving = SimpleNamespace(
        create_error_response=MagicMock(return_value=rejection),
    )

    result = await create(
        serving,
        request,
        _KVTransferAdmission(False),
    )

    assert result is rejection
    serving.create_error_response.assert_called_once_with(expected_error)


@pytest.mark.asyncio
async def test_remote_offer_transfers_only_after_awaited_engine_admission() -> None:
    admission = OpenAIServing._create_kv_transfer_admission(
        SimpleNamespace(has_kv_connector=True),
        {"do_remote_prefill": True},
    )
    generate_entered = asyncio.Event()
    admit_request = asyncio.Event()
    initial_stream = _TestGenerationStream([])

    async def transformed_stream() -> AsyncGenerator[ConversationContext, None]:
        if False:
            yield MagicMock(spec=ConversationContext)

    wrapped_stream = transformed_stream()

    async def generate(*args: Any, **kwargs: Any) -> GenerationStream:
        callback: Callable[[], None] | None = kwargs["on_engine_admission"]
        assert callback is not None
        generate_entered.set()
        await admit_request.wait()
        callback()
        return initial_stream

    serving = SimpleNamespace(
        engine_client=SimpleNamespace(generate=generate),
        _log_inputs=MagicMock(),
        _generate_with_builtin_tools=MagicMock(return_value=wrapped_stream),
    )
    task = asyncio.create_task(
        OpenAIServingResponses._start_generation_with_builtin_tools(
            serving,
            request_id="response",
            engine_input=tokens_input([1]),
            sampling_params=SamplingParams(max_tokens=1),
            context=MagicMock(spec=ConversationContext),
            on_engine_admission=admission.callback,
        )
    )

    await generate_entered.wait()
    assert task.done() is False
    assert admission.engine_owns_offer is False

    admit_request.set()
    result_stream = await task
    assert isinstance(result_stream, ManagedAsyncIterator)
    assert admission.engine_owns_offer is True
    await result_stream.aclose()
    assert initial_stream.close_count == 1


@pytest.mark.asyncio
async def test_pre_admission_error_notifies_rejection_exactly_once() -> None:
    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hello"}],
        request_id="remote-request",
        kv_transfer_params={"do_remote_prefill": True},
    )
    admission = OpenAIServing._create_kv_transfer_admission(
        SimpleNamespace(has_kv_connector=True), request.kv_transfer_params
    )
    notify_rejected = AsyncMock()
    serving = SimpleNamespace(
        has_kv_connector=True,
        _notify_kv_transfer_offer_rejected=notify_rejected,
    )

    async def fail_before_admission() -> None:
        raise ValueError("invalid request")

    with pytest.raises(ValueError, match="invalid request"):
        await OpenAIServing._with_kv_transfer_rejection_cleanup(
            serving,
            fail_before_admission(),
            request.request_id,
            request.kv_transfer_params,
            None,
            admission,
        )

    notify_rejected.assert_awaited_once_with(
        request.request_id,
        request.kv_transfer_params,
        None,
        "serving rejected request before generation ownership transfer",
    )


@pytest.mark.asyncio
async def test_post_admission_error_does_not_reject_engine_owned_offer() -> None:
    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hello"}],
        request_id="remote-request",
        kv_transfer_params={"do_remote_prefill": True},
    )
    admission = OpenAIServing._create_kv_transfer_admission(
        SimpleNamespace(has_kv_connector=True), request.kv_transfer_params
    )
    notify_rejected = AsyncMock()
    serving = SimpleNamespace(
        has_kv_connector=True,
        _notify_kv_transfer_offer_rejected=notify_rejected,
    )

    async def fail_after_admission() -> None:
        callback = admission.callback
        assert callback is not None
        callback()
        raise RuntimeError("generation failed")

    with pytest.raises(RuntimeError, match="generation failed"):
        await OpenAIServing._with_kv_transfer_rejection_cleanup(
            serving,
            fail_after_admission(),
            request.request_id,
            request.kv_transfer_params,
            None,
            admission,
        )

    assert admission.engine_owns_offer is True
    notify_rejected.assert_not_awaited()


@pytest.mark.asyncio
async def test_unstarted_response_wrapper_closes_admitted_generation_stream() -> None:
    initial_stream = _TestGenerationStream([_request_output("response_0")])

    async def response_transform() -> AsyncGenerator[ConversationContext, None]:
        if False:
            yield _TwoTurnContext()

    result_stream = ManagedAsyncIterator(
        response_transform(),
        (initial_stream,),
    )

    await result_stream.aclose()

    assert initial_stream.close_count == 1


@pytest.mark.asyncio
async def test_prestart_background_cancellation_retains_generation_owner() -> None:
    initial_stream = _TestGenerationStream([_request_output("response_0")])

    async def response_transform() -> AsyncGenerator[ConversationContext, None]:
        if False:
            yield _TwoTurnContext()

    result_stream = ManagedAsyncIterator(
        response_transform(),
        (initial_stream,),
    )
    serving = OpenAIServingResponses.__new__(OpenAIServingResponses)
    serving.background_tasks = {}
    background_started = asyncio.Event()

    async def background() -> None:
        try:
            background_started.set()
            await asyncio.Event().wait()
        finally:
            await result_stream.aclose()

    install_task = asyncio.create_task(
        serving._install_background_task(
            "response",
            result_stream,
            background(),
            background_started,
            "create_response",
        )
    )
    asyncio.get_running_loop().call_soon(
        lambda: serving.background_tasks["response"].cancel()
    )

    with pytest.raises(asyncio.CancelledError):
        await install_task

    assert serving.background_tasks == {}
    assert initial_stream.close_count == 1


@pytest.mark.asyncio
async def test_background_setup_failure_removes_provisional_state() -> None:
    initial_stream = _TestGenerationStream([_request_output("response_0")])

    async def response_transform() -> AsyncGenerator[ConversationContext, None]:
        if False:
            yield _TwoTurnContext()

    result_stream = ManagedAsyncIterator(
        response_transform(),
        (initial_stream,),
    )
    messages = [{"role": "user", "content": "hello"}]

    async def background_worker(*args: Any, **kwargs: Any) -> None:
        return

    async def fail_install(
        response_id: str,
        admitted_stream: ManagedAsyncIterator[ConversationContext],
        background_coro: Coroutine[Any, Any, None],
        background_started: asyncio.Event,
        task_name: str,
    ) -> None:
        background_coro.close()
        raise RuntimeError("background setup failed")

    serving = SimpleNamespace(
        _check_model=AsyncMock(return_value=None),
        _validate_create_responses_input=MagicMock(return_value=None),
        engine_client=SimpleNamespace(errored=False),
        enable_store=True,
        response_store={},
        response_store_lock=asyncio.Lock(),
        msg_store={},
        event_store={},
        background_tasks={},
        use_harmony=False,
        _make_request=AsyncMock(return_value=(messages, [tokens_input([1])])),
        _maybe_get_adapters=MagicMock(return_value=None),
        models=SimpleNamespace(model_name=lambda lora_request: "test-model"),
        _validate_generator_input=MagicMock(return_value=None),
        model_config=SimpleNamespace(max_model_len=16),
        default_sampling_params={},
        override_max_tokens=None,
        _extract_prompt_len=lambda engine_input: 1,
        tool_server=None,
        renderer=SimpleNamespace(get_tokenizer=lambda: MagicMock()),
        _effective_chat_template_kwargs=MagicMock(return_value={}),
        _make_response_parser=MagicMock(return_value=None),
        parser=None,
        _start_generation_with_builtin_tools=AsyncMock(return_value=result_stream),
        _run_background_request_stream=background_worker,
        _install_background_task=fail_install,
    )
    request = ResponsesRequest(
        input="hello",
        model="test-model",
        request_id="response",
        store=True,
        background=True,
        stream=True,
        max_output_tokens=1,
    )

    with pytest.raises(RuntimeError, match="background setup failed"):
        await OpenAIServingResponses._create_responses(
            serving,
            request,
            _KVTransferAdmission(False),
        )

    assert serving.response_store == {}
    assert serving.msg_store == {}
    assert serving.event_store == {}
    assert serving.background_tasks == {}
    assert initial_stream.close_count == 1


@pytest.mark.asyncio
async def test_background_task_failure_is_retrieved_and_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    initial_stream = _TestGenerationStream([_request_output("response_0")])

    async def response_transform() -> AsyncGenerator[ConversationContext, None]:
        if False:
            yield _TwoTurnContext()

    result_stream = ManagedAsyncIterator(
        response_transform(),
        (initial_stream,),
    )
    serving = OpenAIServingResponses.__new__(OpenAIServingResponses)
    serving.background_tasks = {}
    background_started = asyncio.Event()
    fail_background = asyncio.Event()

    async def background() -> None:
        try:
            background_started.set()
            await fail_background.wait()
            raise RuntimeError("background worker failed")
        finally:
            await result_stream.aclose()

    caplog.set_level(logging.ERROR)
    await serving._install_background_task(
        "response",
        result_stream,
        background(),
        background_started,
        "create_response",
    )
    task = serving.background_tasks["response"]

    fail_background.set()
    await asyncio.gather(task, return_exceptions=True)
    await asyncio.sleep(0)

    assert serving.background_tasks == {}
    assert initial_stream.close_count == 1
    assert "Background Responses task response failed" in caplog.text


@pytest.mark.asyncio
async def test_responses_stream_emits_typed_terminal_error_event() -> None:
    initial_stream = _TestGenerationStream([])

    async def response_transform() -> AsyncGenerator[ConversationContext, None]:
        if False:
            yield _TwoTurnContext()

    result_stream = ManagedAsyncIterator(
        response_transform(),
        (initial_stream,),
    )

    async def failing_processor(*args: Any, **kwargs: Any):
        raise GenerationError("generation failed")
        if False:
            yield

    serving = SimpleNamespace(
        use_harmony=False,
        _process_simple_streaming_events=failing_processor,
    )
    request = ResponsesRequest(
        input="hello",
        model="test-model",
        request_id="response",
        store=True,
        stream=True,
        max_output_tokens=1,
    )

    events = [
        event
        async for event in OpenAIServingResponses.responses_stream_generator(
            serving,
            request,
            SamplingParams(max_tokens=1),
            result_stream,
            MagicMock(spec=ConversationContext),
            "test-model",
            MagicMock(),
            MagicMock(),
        )
    ]

    assert [event.type for event in events] == [
        "response.created",
        "response.in_progress",
        "error",
    ]
    assert isinstance(events[-1], ResponseErrorEvent)
    assert events[-1].message == "generation failed"
    assert initial_stream.close_count == 1


@pytest.mark.asyncio
async def test_background_stream_error_persists_failure_and_terminates_replay() -> None:
    initial_stream = _TestGenerationStream([])

    async def response_transform() -> AsyncGenerator[ConversationContext, None]:
        if False:
            yield _TwoTurnContext()

    result_stream = ManagedAsyncIterator(
        response_transform(),
        (initial_stream,),
    )
    terminal_error = ResponseErrorEvent(
        type="error",
        code="server_error",
        message="generation failed",
        param=None,
        sequence_number=2,
    )

    async def error_stream() -> AsyncGenerator[ResponseErrorEvent, None]:
        yield terminal_error

    serving = OpenAIServingResponses.__new__(OpenAIServingResponses)
    stored_response = SimpleNamespace(status="queued")
    serving.response_store = {"response": stored_response}
    serving.response_store_lock = asyncio.Lock()
    serving.responses_stream_generator = MagicMock(return_value=error_stream())
    event_deque = deque()
    new_event_signal = asyncio.Event()
    background_done = asyncio.Event()
    background_started = asyncio.Event()

    await serving._run_background_request_stream(
        SimpleNamespace(request_id="response"),
        SamplingParams(max_tokens=1),
        result_stream,
        MagicMock(spec=ConversationContext),
        "test-model",
        MagicMock(),
        MagicMock(),
        0,
        event_deque,
        new_event_signal,
        background_done,
        background_started,
    )

    assert stored_response.status == "failed"
    assert background_done.is_set()
    assert initial_stream.close_count == 1

    serving.event_store = {"response": (event_deque, new_event_signal, background_done)}
    replayed = [
        event
        async for event in serving.responses_background_stream_generator("response")
    ]
    assert replayed == [terminal_error]


@pytest.mark.asyncio
async def test_background_stream_replay_terminates_after_cancellation() -> None:
    serving = OpenAIServingResponses.__new__(OpenAIServingResponses)
    background_done = asyncio.Event()
    background_done.set()
    serving.event_store = {
        "response": (deque(), asyncio.Event(), background_done),
    }

    replayed = [
        event
        async for event in serving.responses_background_stream_generator("response")
    ]

    assert replayed == []


@pytest.mark.asyncio
async def test_background_nonstream_exception_persists_failure() -> None:
    initial_stream = _TestGenerationStream([])

    async def response_transform() -> AsyncGenerator[ConversationContext, None]:
        if False:
            yield _TwoTurnContext()

    result_stream = ManagedAsyncIterator(
        response_transform(),
        (initial_stream,),
    )
    serving = OpenAIServingResponses.__new__(OpenAIServingResponses)
    stored_response = SimpleNamespace(status="queued")
    serving.response_store = {"response": stored_response}
    serving.response_store_lock = asyncio.Lock()
    serving.responses_full_generator = AsyncMock(
        side_effect=RuntimeError("generation failed")
    )
    background_started = asyncio.Event()

    with pytest.raises(RuntimeError, match="generation failed"):
        await serving._run_background_request(
            SimpleNamespace(request_id="response"),
            SamplingParams(max_tokens=1),
            result_stream,
            MagicMock(spec=ConversationContext),
            "test-model",
            MagicMock(),
            MagicMock(),
            0,
            background_started,
        )

    assert stored_response.status == "failed"
    assert initial_stream.close_count == 1


@pytest.mark.asyncio
async def test_responses_second_tool_turn_drops_remote_offer() -> None:
    initial_stream = _TestGenerationStream([_request_output("response_0")])
    second_stream = _TestGenerationStream([_request_output("response_1")])
    second_turn_params: list[SamplingParams] = []

    async def generate(
        prompt: Any,
        sampling_params: SamplingParams,
        request_id: str,
        **kwargs: Any,
    ) -> GenerationStream:
        second_turn_params.append(sampling_params)
        return second_stream

    serving = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=16),
        engine_client=SimpleNamespace(generate=generate),
        _log_inputs=MagicMock(),
    )
    sampling_params = SamplingParams(
        max_tokens=1,
        extra_args={
            "kv_transfer_params": {"do_remote_prefill": True},
            "preserved": "value",
        },
    )
    result_stream = OpenAIServingResponses._generate_with_builtin_tools(
        serving,
        request_id="response",
        engine_input=tokens_input([1]),
        sampling_params=sampling_params,
        context=_TwoTurnContext(),
        initial_generator=initial_stream,
    )

    async for _ in result_stream:
        pass

    assert len(second_turn_params) == 1
    assert second_turn_params[0].extra_args == {"preserved": "value"}
    assert sampling_params.extra_args == {
        "kv_transfer_params": {"do_remote_prefill": True},
        "preserved": "value",
    }
    assert initial_stream.close_count == 1
    assert second_stream.close_count == 1


@pytest.mark.asyncio
async def test_batch_chat_admits_concurrently_and_closes_on_sibling_failure() -> None:
    request = BatchChatCompletionRequest(
        model="test-model",
        messages=[
            [{"role": "user", "content": "first"}],
            [{"role": "user", "content": "second"}],
        ],
        request_id="batch",
        max_tokens=1,
    )
    all_started = asyncio.Event()
    started_request_ids: set[str] = set()
    admitted_stream = _TestGenerationStream([])

    async def generate(*args: Any, **kwargs: Any) -> GenerationStream:
        request_id: str = args[2]
        started_request_ids.add(request_id)
        if len(started_request_ids) == 2:
            all_started.set()
        await all_started.wait()
        if request_id.endswith("_1"):
            raise RuntimeError("sibling admission failed")
        return admitted_stream

    serving = SimpleNamespace(
        renderer=SimpleNamespace(tokenizer=MagicMock()),
        parser_cls=None,
        render_batch_chat_request=AsyncMock(
            return_value=(
                [
                    [{"role": "user", "content": "first"}],
                    [{"role": "user", "content": "second"}],
                ],
                [tokens_input([1]), tokens_input([2])],
            )
        ),
        _base_request_id=lambda raw_request, request_id: request_id,
        _maybe_get_adapters=lambda request, **kwargs: None,
        models=SimpleNamespace(model_name=lambda lora_request: "test-model"),
        _get_data_parallel_rank=lambda raw_request: None,
        model_config=SimpleNamespace(max_model_len=16),
        default_sampling_params={},
        override_max_tokens=None,
        _extract_prompt_len=lambda engine_input: 1,
        _log_inputs=MagicMock(),
        engine_client=SimpleNamespace(generate=generate),
        chat_completion_full_generator_batch=AsyncMock(),
    )

    with pytest.raises(RuntimeError, match="sibling admission failed"):
        await asyncio.wait_for(
            OpenAIServingChatBatch.create_batch_chat_completion(serving, request),
            timeout=1,
        )

    assert len(started_request_ids) == 2
    assert admitted_stream.close_count == 1
    serving.chat_completion_full_generator_batch.assert_not_awaited()
