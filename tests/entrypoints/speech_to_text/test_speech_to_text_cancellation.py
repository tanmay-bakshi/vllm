# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from vllm.engine.protocol import GenerationStream
from vllm.entrypoints.speech_to_text.base.serving import OpenAISpeechToText
from vllm.entrypoints.speech_to_text.transcription.protocol import TranscriptionResponse
from vllm.outputs import RequestOutput
from vllm.utils.async_utils import ManagedAsyncIterator


class _NeverFinishingGenerationStream(GenerationStream):
    """Block on generation while recording start and closure."""

    _closed: bool
    _release: asyncio.Event
    _request_id: str
    _started_request_ids: list[str] | None
    close_count: int

    def __init__(
        self,
        request_id: str,
        started_request_ids: list[str] | None = None,
    ) -> None:
        self._closed = False
        self._release = asyncio.Event()
        self._request_id = request_id
        self._started_request_ids = started_request_ids
        self.close_count = 0

    def __aiter__(self) -> "_NeverFinishingGenerationStream":
        return self

    async def __anext__(self) -> RequestOutput:
        if self._started_request_ids is not None:
            self._started_request_ids.append(self._request_id)
            self._started_request_ids = None
        await self._release.wait()
        raise StopAsyncIteration

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.close_count += 1
        self._release.set()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("engine_inputs", "expected_request_ids"),
    [
        ([{"prompt": "chunk"}], ["transcribe-outer-request"]),
        (
            [{"prompt": "chunk-0"}, {"prompt": "chunk-1"}],
            ["transcribe-outer-request-0", "transcribe-outer-request-1"],
        ),
    ],
)
async def test_non_streaming_cancel_aborts_engine_requests(
    engine_inputs, expected_request_ids
):
    streams: list[_NeverFinishingGenerationStream] = []
    started_request_ids: list[str] = []

    async def generate(*args, **kwargs):
        stream = _NeverFinishingGenerationStream(args[2], started_request_ids)
        streams.append(stream)
        return stream

    engine_client = SimpleNamespace(
        errored=False,
        generate=AsyncMock(side_effect=generate),
        abort=AsyncMock(),
        is_tracing_enabled=AsyncMock(return_value=False),
    )

    server = OpenAISpeechToText.__new__(OpenAISpeechToText)
    server.engine_client = engine_client
    server.task_type = "transcribe"
    server.models = SimpleNamespace(model_name=lambda: "audio")
    server.model_config = SimpleNamespace(max_model_len=1024)
    server.model_cls = SimpleNamespace(no_space_languages=set())
    server.default_sampling_params = {}
    server.asr_config = SimpleNamespace(max_audio_clip_s=30)
    server._check_model = AsyncMock(return_value=None)
    server._maybe_get_adapters = Mock(return_value=None)
    server._preprocess_speech_to_text = AsyncMock(return_value=(engine_inputs, 40.0))
    server._log_inputs = Mock()

    request = SimpleNamespace(
        model="audio",
        response_format="json",
        stream=False,
        use_beam_search=False,
        max_completion_tokens=None,
        language="en",
        prompt="",
        to_sampling_params=Mock(return_value=object()),
    )
    raw_request = SimpleNamespace(
        headers={"X-Request-Id": "outer-request"},
        state=SimpleNamespace(),
    )

    task = asyncio.create_task(
        server._create_speech_to_text(
            audio_data=b"audio",
            request=request,
            raw_request=raw_request,
            response_class=TranscriptionResponse,
            stream_generator_method=Mock(),
        )
    )

    async def wait_until_all_streams_start() -> None:
        while len(started_request_ids) < len(expected_request_ids):
            await asyncio.sleep(0)

    await asyncio.wait_for(wait_until_all_streams_start(), timeout=1)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    generated_request_ids = [
        call.args[2] for call in engine_client.generate.call_args_list
    ]
    assert generated_request_ids == expected_request_ids
    engine_client.abort.assert_awaited_once_with(expected_request_ids)
    assert all(stream.close_count == 1 for stream in streams)


@pytest.mark.asyncio
async def test_multi_chunk_cancellation_during_admission_closes_stream() -> None:
    admitted_stream = _NeverFinishingGenerationStream("transcribe-outer-request-0")
    first_admitted = asyncio.Event()
    hold_second_admission = asyncio.Event()

    async def generate(*args, **kwargs):
        if args[2].endswith("-0"):
            first_admitted.set()
            return admitted_stream
        await hold_second_admission.wait()
        return _NeverFinishingGenerationStream(args[2])

    engine_client = SimpleNamespace(
        errored=False,
        generate=AsyncMock(side_effect=generate),
        abort=AsyncMock(),
        is_tracing_enabled=AsyncMock(return_value=False),
    )
    server = OpenAISpeechToText.__new__(OpenAISpeechToText)
    server.engine_client = engine_client
    server.task_type = "transcribe"
    server.models = SimpleNamespace(model_name=lambda: "audio")
    server.model_config = SimpleNamespace(max_model_len=1024)
    server.model_cls = SimpleNamespace(no_space_languages=set())
    server.default_sampling_params = {}
    server.asr_config = SimpleNamespace(max_audio_clip_s=30)
    server._check_model = AsyncMock(return_value=None)
    server._maybe_get_adapters = Mock(return_value=None)
    server._preprocess_speech_to_text = AsyncMock(
        return_value=([{"prompt": "chunk-0"}, {"prompt": "chunk-1"}], 40.0)
    )
    server._log_inputs = Mock()
    request = SimpleNamespace(
        model="audio",
        response_format="json",
        stream=False,
        use_beam_search=False,
        max_completion_tokens=None,
        language="en",
        prompt="",
        to_sampling_params=Mock(return_value=object()),
    )
    raw_request = SimpleNamespace(
        headers={"X-Request-Id": "outer-request"},
        state=SimpleNamespace(),
    )
    task = asyncio.create_task(
        server._create_speech_to_text(
            audio_data=b"audio",
            request=request,
            raw_request=raw_request,
            response_class=TranscriptionResponse,
            stream_generator_method=Mock(),
        )
    )

    await asyncio.wait_for(first_admitted.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert admitted_stream.close_count == 1


@pytest.mark.asyncio
async def test_non_streaming_cancel_advances_all_chunk_generators():
    started_request_ids: list[str] = []

    async def generate(*args, **kwargs):
        return _NeverFinishingGenerationStream(args[2], started_request_ids)

    engine_client = SimpleNamespace(
        errored=False,
        generate=AsyncMock(side_effect=generate),
        abort=AsyncMock(),
        is_tracing_enabled=AsyncMock(return_value=False),
    )

    engine_inputs = [
        {"prompt": "chunk-0"},
        {"prompt": "chunk-1"},
        {"prompt": "chunk-2"},
    ]
    server = OpenAISpeechToText.__new__(OpenAISpeechToText)
    server.engine_client = engine_client
    server.task_type = "transcribe"
    server.models = SimpleNamespace(model_name=lambda: "audio")
    server.model_config = SimpleNamespace(max_model_len=1024)
    server.model_cls = SimpleNamespace(no_space_languages=set())
    server.default_sampling_params = {}
    server.asr_config = SimpleNamespace(max_audio_clip_s=30)
    server._check_model = AsyncMock(return_value=None)
    server._maybe_get_adapters = Mock(return_value=None)
    server._preprocess_speech_to_text = AsyncMock(return_value=(engine_inputs, 90.0))
    server._log_inputs = Mock()

    request = SimpleNamespace(
        model="audio",
        response_format="json",
        stream=False,
        use_beam_search=False,
        max_completion_tokens=None,
        language="en",
        prompt="",
        to_sampling_params=Mock(return_value=object()),
    )
    raw_request = SimpleNamespace(
        headers={"X-Request-Id": "outer-request"},
        state=SimpleNamespace(),
    )

    task = asyncio.create_task(
        server._create_speech_to_text(
            audio_data=b"audio",
            request=request,
            raw_request=raw_request,
            response_class=TranscriptionResponse,
            stream_generator_method=Mock(),
        )
    )
    await asyncio.sleep(0.01)

    expected_request_ids = [
        "transcribe-outer-request-0",
        "transcribe-outer-request-1",
        "transcribe-outer-request-2",
    ]
    assert set(started_request_ids) == set(expected_request_ids)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_language_detection_cancel_closes_engine_stream():
    result_stream = _NeverFinishingGenerationStream(
        "transcribe-outer-request-lang_detect"
    )
    engine_client = SimpleNamespace(
        generate=AsyncMock(return_value=result_stream),
    )

    server = OpenAISpeechToText.__new__(OpenAISpeechToText)
    server.engine_client = engine_client
    server.asr_config = SimpleNamespace()
    server.tokenizer = Mock()
    server.model_cls = SimpleNamespace(
        get_language_detection_prompt=Mock(return_value={"prompt": "detect"}),
        get_language_token_ids=Mock(return_value=[1]),
        parse_language_detection_output=Mock(),
    )

    request_id = "transcribe-outer-request-lang_detect"
    task = asyncio.create_task(server._detect_language(Mock(), request_id))
    await asyncio.sleep(0)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert result_stream.close_count == 1


@pytest.mark.asyncio
async def test_streaming_single_chunk_returns_managed_stream() -> None:
    result_stream = _NeverFinishingGenerationStream("transcribe-outer-request")
    engine_client = SimpleNamespace(
        errored=False,
        generate=AsyncMock(return_value=result_stream),
        is_tracing_enabled=AsyncMock(return_value=False),
    )
    server = OpenAISpeechToText.__new__(OpenAISpeechToText)
    server.engine_client = engine_client
    server.task_type = "transcribe"
    server.models = SimpleNamespace(model_name=lambda: "audio")
    server.model_config = SimpleNamespace(max_model_len=1024)
    server.model_cls = SimpleNamespace(no_space_languages=set())
    server.default_sampling_params = {}
    server.asr_config = SimpleNamespace(max_audio_clip_s=30)
    server._check_model = AsyncMock(return_value=None)
    server._maybe_get_adapters = Mock(return_value=None)
    server._preprocess_speech_to_text = AsyncMock(
        return_value=([{"prompt": "chunk"}], 10.0)
    )
    server._log_inputs = Mock()

    async def transform(*args, **kwargs):
        if False:
            yield ""

    request = SimpleNamespace(
        model="audio",
        response_format="json",
        stream=True,
        use_beam_search=False,
        max_completion_tokens=None,
        language="en",
        prompt="",
        to_sampling_params=Mock(return_value=object()),
    )
    raw_request = SimpleNamespace(
        headers={"X-Request-Id": "outer-request"},
        state=SimpleNamespace(),
    )

    response = await server._create_speech_to_text(
        audio_data=b"audio",
        request=request,
        raw_request=raw_request,
        response_class=TranscriptionResponse,
        stream_generator_method=transform,
    )

    assert isinstance(response, ManagedAsyncIterator)
    engine_client.generate.assert_awaited_once()
    await response.aclose()
    assert result_stream.close_count == 1
