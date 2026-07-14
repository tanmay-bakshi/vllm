# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from vllm.config.multimodal import MultiModalConfig
from vllm.engine.protocol import GenerationStream
from vllm.entrypoints.openai.engine.protocol import ErrorResponse, StreamOptions
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.serve.disagg.protocol import (
    GenerateRequest,
    GenerateResponse,
)
from vllm.entrypoints.serve.disagg.serving import ServingTokens
from vllm.logprobs import Logprob
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.renderers import renderer_from_config
from vllm.renderers.online_renderer import OnlineRenderer
from vllm.sampling_params import SamplingParams
from vllm.utils.async_utils import ManagedAsyncIterator
from vllm.v1.engine.async_llm import AsyncLLM

MODEL_NAME = "openai-community/gpt2"
BASE_MODEL_PATHS = [
    BaseModelPath(name=MODEL_NAME, model_path=MODEL_NAME),
]


@dataclass
class MockHFConfig:
    model_type: str = "any"


@dataclass
class MockModelConfig:
    task = "generate"
    runner_type = "generate"
    model = MODEL_NAME
    tokenizer = MODEL_NAME
    trust_remote_code = False
    tokenizer_mode = "auto"
    max_model_len = 100
    tokenizer_revision = None
    multimodal_config = MultiModalConfig()
    hf_config = MockHFConfig()
    hf_text_config = MockHFConfig()
    logits_processors: list[str] | None = None
    diff_sampling_param: dict | None = None
    allowed_local_media_path: str = ""
    allowed_media_domains: list[str] | None = None
    encoder_config = None
    generation_config: str = "auto"
    media_io_kwargs: dict[str, dict[str, Any]] = field(default_factory=dict)
    skip_tokenizer_init = False
    is_encoder_decoder: bool = False
    is_multimodal_model: bool = False
    renderer_num_workers: int = 1

    def get_diff_sampling_param(self):
        return self.diff_sampling_param or {}


@dataclass
class MockParallelConfig:
    _api_process_rank: int = 0


@dataclass
class MockSchedulerConfig:
    max_num_seqs: int = 128


@dataclass
class MockVllmConfig:
    model_config: MockModelConfig
    parallel_config: MockParallelConfig
    scheduler_config: MockSchedulerConfig = field(default_factory=MockSchedulerConfig)
    kv_transfer_config: object | None = None


class _RecordingGenerationStream(GenerationStream):
    """Generation stream that records deterministic closure."""

    _outputs: tuple[RequestOutput, ...]
    _index: int
    _closed: bool
    close_count: int

    def __init__(self, outputs: tuple[RequestOutput, ...]) -> None:
        """Create a generation stream.

        :param outputs: Outputs returned before the stream terminates.
        """
        self._outputs = outputs
        self._index = 0
        self._closed = False
        self.close_count = 0

    def __aiter__(self) -> "_RecordingGenerationStream":
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


def _generation_stream(*outputs: RequestOutput) -> _RecordingGenerationStream:
    """Create a closeable generation stream from outputs.

    :param outputs: Outputs returned before the stream terminates.
    :returns: Closeable test generation stream.
    """
    return _RecordingGenerationStream(outputs)


def _build_renderer(model_config: MockModelConfig):
    return renderer_from_config(
        MockVllmConfig(model_config, parallel_config=MockParallelConfig()),
    )


def _build_serving_tokens(engine: AsyncLLM, **kwargs) -> ServingTokens:
    models = OpenAIServingModels(
        engine_client=engine,
        base_model_paths=BASE_MODEL_PATHS,
    )
    online_renderer = OnlineRenderer(
        model_config=engine.model_config,
        renderer=engine.renderer,
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
    )
    serving = ServingTokens(
        engine,
        models,
        online_renderer=online_renderer,
        request_logger=None,
        **kwargs,
    )

    async def _fake_preprocess(*args, **kwargs):
        return [{"prompt_token_ids": [1, 2, 3]}]

    serving.online_renderer.preprocess_completion = AsyncMock(
        side_effect=_fake_preprocess
    )
    return serving


def _make_request_output(
    request_id: str,
    token_ids: list[int],
    finish_reason: str | None = None,
    finished: bool = False,
    prompt_token_ids: list[int] | None = None,
    logprobs: list[dict[int, Any] | None] | None = None,
    num_cached_tokens: int | None = None,
    index: int = 0,
) -> RequestOutput:
    return RequestOutput(
        request_id=request_id,
        prompt=None,
        prompt_token_ids=prompt_token_ids or [1, 2, 3],
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=index,
                text="",
                token_ids=token_ids,
                cumulative_logprob=None,
                logprobs=logprobs,
                finish_reason=finish_reason,
            )
        ],
        finished=finished,
        metrics=None,
        lora_request=None,
        encoder_prompt=None,
        encoder_prompt_token_ids=None,
        num_cached_tokens=num_cached_tokens,
    )


def _mock_engine() -> MagicMock:
    engine = MagicMock(spec=AsyncLLM)
    engine.errored = False
    engine.model_config = MockModelConfig()
    engine.vllm_config = MockVllmConfig(
        engine.model_config, parallel_config=MockParallelConfig()
    )
    engine.input_processor = MagicMock()
    engine.renderer = _build_renderer(engine.model_config)
    return engine


def _parse_sse_chunks(chunks: list[str]) -> list[Any]:
    """Parse SSE chunks into dicts (JSON) or raw strings ([DONE])."""
    parsed: list[Any] = []
    for chunk in chunks:
        assert chunk.startswith("data: ") and chunk.endswith("\n\n")
        payload = chunk[len("data: ") : -len("\n\n")]
        if payload == "[DONE]":
            parsed.append("[DONE]")
        else:
            parsed.append(json.loads(payload))
    return parsed


@pytest.mark.asyncio
async def test_serve_tokens_skips_mm_cache_for_remote_engine_execution():
    engine = _mock_engine()
    engine.generate = AsyncMock(
        return_value=_generation_stream(
            _make_request_output(
                "req-1", token_ids=[10], finish_reason="stop", finished=True
            )
        )
    )
    serving = _build_serving_tokens(engine)

    request = GenerateRequest(
        token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=1),
        model=MODEL_NAME,
        stream=False,
    )

    response = await serving.serve_tokens(request)

    assert isinstance(response, GenerateResponse)
    assert (
        serving.online_renderer.preprocess_completion.call_args.kwargs["skip_mm_cache"]
        is True
    )


@pytest.mark.asyncio
async def test_remote_offer_uses_atomic_admission_and_managed_stream() -> None:
    engine = _mock_engine()
    engine.vllm_config.kv_transfer_config = object()
    generation_stream = _generation_stream()

    async def generate(*args, **kwargs):
        on_engine_admission = kwargs["on_engine_admission"]
        assert on_engine_admission is not None
        on_engine_admission()
        return generation_stream

    engine.generate = AsyncMock(side_effect=generate)
    serving = _build_serving_tokens(engine)
    offer = {"do_remote_prefill": True, "remote_request_id": "producer"}
    sampling_params = SamplingParams(max_tokens=1)
    request = GenerateRequest(
        request_id="decoder",
        token_ids=[1, 2, 3],
        sampling_params=sampling_params,
        model=MODEL_NAME,
        stream=True,
        kv_transfer_params=offer,
    )

    response = await serving.serve_tokens(request)

    assert isinstance(response, ManagedAsyncIterator)
    submitted_params = engine.generate.await_args.args[1]
    assert submitted_params is not sampling_params
    assert submitted_params.extra_args == {"kv_transfer_params": offer}
    assert sampling_params.extra_args is None

    await response.aclose()
    assert generation_stream.close_count == 1


@pytest.mark.asyncio
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
async def test_kv_producer_rejects_unrepresentable_response_contract(
    stream: bool,
    n: int,
    expected_error: str,
) -> None:
    engine = _mock_engine()
    engine.generate = AsyncMock()
    serving = _build_serving_tokens(engine)
    request = GenerateRequest(
        token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=1, n=n),
        model=MODEL_NAME,
        stream=stream,
        kv_transfer_params={"do_remote_decode": True},
    )

    response = await serving.serve_tokens(request)

    assert isinstance(response, ErrorResponse)
    assert response.error.message == expected_error
    engine.generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_kv_consumer_allows_matching_parallel_sampling_contract() -> None:
    engine = _mock_engine()
    engine.vllm_config.kv_transfer_config = object()
    generation_stream = _generation_stream()

    async def generate(*args, **kwargs):
        on_engine_admission = kwargs["on_engine_admission"]
        assert on_engine_admission is not None
        on_engine_admission()
        return generation_stream

    engine.generate = AsyncMock(side_effect=generate)
    serving = _build_serving_tokens(engine)
    request = GenerateRequest(
        token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=1, n=2),
        model=MODEL_NAME,
        stream=True,
        kv_transfer_params={
            "do_remote_prefill": True,
            "expected_consumers": 2,
            "remote_request_id": "producer",
        },
    )

    response = await serving.serve_tokens(request)

    assert isinstance(response, ManagedAsyncIterator)
    await response.aclose()
    assert generation_stream.close_count == 1


@pytest.mark.asyncio
async def test_kv_consumer_rejects_parallel_sampling_contract_mismatch() -> None:
    engine = _mock_engine()
    engine.vllm_config.kv_transfer_config = object()
    engine.generate = AsyncMock()
    engine.notify_kv_transfer_request_rejected = AsyncMock(return_value=True)
    serving = _build_serving_tokens(engine)
    offer = {
        "do_remote_prefill": True,
        "expected_consumers": 2,
        "remote_request_id": "producer",
    }
    request = GenerateRequest(
        request_id="decoder",
        token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=1, n=1),
        model=MODEL_NAME,
        kv_transfer_params=offer,
    )

    response = await serving.serve_tokens(request)

    assert isinstance(response, ErrorResponse)
    assert "expected_consumers (2), got 1" in response.error.message
    engine.generate.assert_not_awaited()
    engine.notify_kv_transfer_request_rejected.assert_awaited_once()


@pytest.mark.asyncio
async def test_embedded_remote_offer_is_rejected() -> None:
    engine = _mock_engine()
    engine.generate = AsyncMock()
    serving = _build_serving_tokens(engine)
    request = GenerateRequest(
        token_ids=[1, 2, 3],
        sampling_params=SamplingParams(
            max_tokens=1,
            extra_args={"kv_transfer_params": {"do_remote_prefill": True}},
        ),
        model=MODEL_NAME,
    )

    response = await serving.serve_tokens(request)

    assert isinstance(response, ErrorResponse)
    assert "top-level GenerateRequest field" in response.error.message
    engine.generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_pre_admission_error_rejects_remote_offer() -> None:
    engine = _mock_engine()
    engine.vllm_config.kv_transfer_config = object()
    engine.generate = AsyncMock()
    engine.notify_kv_transfer_request_rejected = AsyncMock(return_value=True)
    serving = _build_serving_tokens(engine)
    offer = {
        "do_remote_prefill": True,
        "expected_consumers": 129,
        "remote_request_id": "producer",
    }
    request = GenerateRequest(
        request_id="decoder",
        token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=1, n=129),
        model=MODEL_NAME,
        kv_transfer_params=offer,
    )

    response = await serving.serve_tokens(request)

    assert isinstance(response, ErrorResponse)
    engine.generate.assert_not_awaited()
    engine.notify_kv_transfer_request_rejected.assert_awaited_once_with(
        "decoder",
        offer,
        "serving rejected request before generation ownership transfer",
        data_parallel_rank=None,
    )


@pytest.mark.asyncio
async def test_stream_basic():
    """Streaming returns SSE chunks with correct token_ids and ends with [DONE]."""
    engine = _mock_engine()
    engine.generate = AsyncMock(
        return_value=_generation_stream(
            _make_request_output("req-1", token_ids=[10]),
            _make_request_output("req-1", token_ids=[20, 30]),
            _make_request_output(
                "req-1", token_ids=[40], finish_reason="stop", finished=True
            ),
        )
    )
    serving = _build_serving_tokens(engine)

    request = GenerateRequest(
        token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=10),
        model=MODEL_NAME,
        stream=True,
    )

    response = await serving.serve_tokens(request)
    chunks = []
    async for chunk in response:
        chunks.append(chunk)

    parsed = _parse_sse_chunks(chunks)

    # 3 data chunks + [DONE]
    assert parsed[-1] == "[DONE]"
    data_chunks = [c for c in parsed if c != "[DONE]"]
    assert len(data_chunks) == 3

    assert data_chunks[0]["choices"][0]["token_ids"] == [10]
    assert data_chunks[1]["choices"][0]["token_ids"] == [20, 30]
    assert data_chunks[2]["choices"][0]["token_ids"] == [40]
    assert data_chunks[2]["choices"][0]["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_stream_error_mid_generation():
    """finish_reason='error' mid-stream yields error chunk then [DONE]."""
    engine = _mock_engine()
    engine.generate = AsyncMock(
        return_value=_generation_stream(
            _make_request_output("req-1", token_ids=[10]),
            _make_request_output(
                "req-1", token_ids=[20], finish_reason="error", finished=True
            ),
        )
    )
    serving = _build_serving_tokens(engine)

    request = GenerateRequest(
        token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=10),
        model=MODEL_NAME,
        stream=True,
    )

    response = await serving.serve_tokens(request)
    chunks = []
    async for chunk in response:
        chunks.append(chunk)

    assert len(chunks) >= 2
    assert any("Internal server error" in chunk for chunk in chunks), (
        f"Expected error message in chunks: {chunks}"
    )
    assert chunks[-1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_stream_error_with_empty_delta():
    """finish_reason='error' with empty delta_token_ids still raises."""
    engine = _mock_engine()
    engine.generate = AsyncMock(
        return_value=_generation_stream(
            _make_request_output("req-1", token_ids=[10]),
            _make_request_output(
                "req-1", token_ids=[], finish_reason="error", finished=True
            ),
        )
    )
    serving = _build_serving_tokens(engine)

    request = GenerateRequest(
        token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=10),
        model=MODEL_NAME,
        stream=True,
    )

    response = await serving.serve_tokens(request)
    chunks = []
    async for chunk in response:
        chunks.append(chunk)

    assert any("Internal server error" in chunk for chunk in chunks), (
        f"Expected error message in chunks: {chunks}"
    )
    assert chunks[-1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_stream_skips_empty_token_output():
    """Outputs with empty token_ids are skipped (no chunk emitted)."""
    engine = _mock_engine()
    engine.generate = AsyncMock(
        return_value=_generation_stream(
            _make_request_output("req-1", token_ids=[10]),
            _make_request_output("req-1", token_ids=[]),
            _make_request_output(
                "req-1", token_ids=[20], finish_reason="stop", finished=True
            ),
        )
    )
    serving = _build_serving_tokens(engine)

    request = GenerateRequest(
        token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=10),
        model=MODEL_NAME,
        stream=True,
    )

    response = await serving.serve_tokens(request)
    chunks = []
    async for chunk in response:
        chunks.append(chunk)

    parsed = _parse_sse_chunks(chunks)
    assert parsed[-1] == "[DONE]"
    data_chunks = [c for c in parsed if c != "[DONE]"]

    # Only 2 data chunks — the empty one is skipped
    assert len(data_chunks) == 2
    assert data_chunks[0]["choices"][0]["token_ids"] == [10]
    assert data_chunks[1]["choices"][0]["token_ids"] == [20]


@pytest.mark.asyncio
async def test_stream_include_usage():
    """stream_options.include_usage emits a final usage-only chunk."""
    engine = _mock_engine()
    engine.generate = AsyncMock(
        return_value=_generation_stream(
            _make_request_output("req-1", token_ids=[10]),
            _make_request_output(
                "req-1", token_ids=[20], finish_reason="stop", finished=True
            ),
        )
    )
    serving = _build_serving_tokens(engine)

    request = GenerateRequest(
        token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=10),
        model=MODEL_NAME,
        stream=True,
        stream_options=StreamOptions(include_usage=True),
    )

    response = await serving.serve_tokens(request)
    chunks = []
    async for chunk in response:
        chunks.append(chunk)

    parsed = _parse_sse_chunks(chunks)
    assert parsed[-1] == "[DONE]"

    # The chunk before [DONE] should be the usage-only chunk
    usage_chunk = parsed[-2]
    assert usage_chunk["choices"] == []
    assert usage_chunk["usage"]["prompt_tokens"] == 3
    assert usage_chunk["usage"]["completion_tokens"] == 2
    assert usage_chunk["usage"]["total_tokens"] == 5


@pytest.mark.asyncio
async def test_stream_continuous_usage():
    """continuous_usage_stats adds usage to every data chunk."""
    engine = _mock_engine()
    engine.generate = AsyncMock(
        return_value=_generation_stream(
            _make_request_output("req-1", token_ids=[10]),
            _make_request_output(
                "req-1", token_ids=[20], finish_reason="stop", finished=True
            ),
        )
    )
    serving = _build_serving_tokens(engine)

    request = GenerateRequest(
        token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=10),
        model=MODEL_NAME,
        stream=True,
        stream_options=StreamOptions(
            include_usage=True,
            continuous_usage_stats=True,
        ),
    )

    response = await serving.serve_tokens(request)
    chunks = []
    async for chunk in response:
        chunks.append(chunk)

    parsed = _parse_sse_chunks(chunks)
    data_chunks = [c for c in parsed if isinstance(c, dict) and c.get("choices")]

    # Every data chunk should have usage
    for i, dc in enumerate(data_chunks):
        assert dc["usage"] is not None, f"chunk {i} missing usage"
        assert dc["usage"]["prompt_tokens"] == 3

    # First chunk: 1 completion token
    assert data_chunks[0]["usage"]["completion_tokens"] == 1
    assert data_chunks[0]["usage"]["total_tokens"] == 4

    # Second chunk: 2 completion tokens (cumulative)
    assert data_chunks[1]["usage"]["completion_tokens"] == 2
    assert data_chunks[1]["usage"]["total_tokens"] == 5


@pytest.mark.asyncio
async def test_stream_with_logprobs():
    """Streaming with logprobs includes logprob data in each chunk."""
    engine = _mock_engine()
    engine.generate = AsyncMock(
        return_value=_generation_stream(
            _make_request_output(
                "req-1",
                token_ids=[10],
                logprobs=[{10: Logprob(logprob=-0.5)}],
            ),
            _make_request_output(
                "req-1",
                token_ids=[20],
                logprobs=[{20: Logprob(logprob=-1.0)}],
                finish_reason="stop",
                finished=True,
            ),
        )
    )
    serving = _build_serving_tokens(engine)

    request = GenerateRequest(
        token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=10, logprobs=1),
        model=MODEL_NAME,
        stream=True,
    )

    response = await serving.serve_tokens(request)
    chunks = []
    async for chunk in response:
        chunks.append(chunk)

    parsed = _parse_sse_chunks(chunks)
    data_chunks = [c for c in parsed if isinstance(c, dict) and c.get("choices")]

    for dc in data_chunks:
        lp = dc["choices"][0]["logprobs"]
        assert lp is not None
        assert len(lp["content"]) == 1
        assert lp["content"][0]["token"].startswith("token_id:")


@pytest.mark.asyncio
async def test_stream_prompt_tokens_details():
    """enable_prompt_tokens_details includes cached_tokens in final usage."""
    engine = _mock_engine()
    engine.generate = AsyncMock(
        return_value=_generation_stream(
            _make_request_output(
                "req-1",
                token_ids=[10],
                finish_reason="stop",
                finished=True,
                num_cached_tokens=2,
            )
        )
    )
    serving = _build_serving_tokens(engine, enable_prompt_tokens_details=True)

    request = GenerateRequest(
        token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=10),
        model=MODEL_NAME,
        stream=True,
        stream_options=StreamOptions(include_usage=True),
    )

    response = await serving.serve_tokens(request)
    chunks = []
    async for chunk in response:
        chunks.append(chunk)

    parsed = _parse_sse_chunks(chunks)
    # Usage-only chunk (before [DONE])
    usage_chunk = parsed[-2]
    assert usage_chunk["choices"] == []
    assert usage_chunk["usage"]["prompt_tokens_details"]["cached_tokens"] == 2


@pytest.mark.asyncio
async def test_stream_prompt_tokens_details_zero_cached():
    """enable_prompt_tokens_details includes cached_tokens=0 in final usage.

    Regression test for https://github.com/vllm-project/vllm/issues/44377:
    zero cached tokens must not be treated as falsy and omitted.
    """
    engine = _mock_engine()
    engine.generate = AsyncMock(
        return_value=_generation_stream(
            _make_request_output(
                "req-1",
                token_ids=[10],
                finish_reason="stop",
                finished=True,
                num_cached_tokens=0,
            )
        )
    )
    serving = _build_serving_tokens(engine, enable_prompt_tokens_details=True)

    request = GenerateRequest(
        token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=10),
        model=MODEL_NAME,
        stream=True,
        stream_options=StreamOptions(include_usage=True),
    )

    response = await serving.serve_tokens(request)
    chunks = []
    async for chunk in response:
        chunks.append(chunk)

    parsed = _parse_sse_chunks(chunks)
    # Usage-only chunk (before [DONE])
    usage_chunk = parsed[-2]
    assert usage_chunk["choices"] == []
    # Zero cached tokens must be present, not omitted
    assert usage_chunk["usage"]["prompt_tokens_details"] is not None
    assert usage_chunk["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
