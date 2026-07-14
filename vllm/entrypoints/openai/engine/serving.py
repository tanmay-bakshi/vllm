# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import time
import traceback
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import Any, ClassVar, Generic, TypeVar

from fastapi import Request
from pydantic import ConfigDict
from starlette.datastructures import Headers

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.generate.beam_search.online import BeamSearchOnlineMixin
from vllm.entrypoints.openai.engine.protocol import GenerationError
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.serve.engine.serving import BaseServing
from vllm.entrypoints.serve.engine.typing import AnyRequest
from vllm.entrypoints.serve.utils.request_logger import RequestLogger
from vllm.inputs import EngineInput
from vllm.logger import init_logger
from vllm.logprobs import Logprob, PromptLogprobs
from vllm.lora.request import LoRARequest
from vllm.tokenizers import TokenizerLike
from vllm.tracing import (
    contains_trace_headers,
    extract_trace_headers,
    log_tracing_disabled_warning,
)

logger = init_logger(__name__)

RequestT = TypeVar("RequestT", bound=AnyRequest)
_T = TypeVar("_T")


class _KVTransferAdmission:
    """Track the authoritative owner of a remote-prefill offer."""

    _tracks_remote_offer: bool
    _engine_owns_offer: bool

    def __init__(self, tracks_remote_offer: bool) -> None:
        """Create an admission boundary for one serving request.

        :param tracks_remote_offer: Whether the request carries a producer
            offer whose cleanup ownership must be transferred explicitly.
        """
        self._tracks_remote_offer = tracks_remote_offer
        self._engine_owns_offer = False

    @property
    def callback(self) -> Callable[[], None] | None:
        """Return the ownership callback only for remote-prefill requests.

        :returns: Engine admission callback, or ``None`` on the ordinary fast
            path.
        """
        if self._tracks_remote_offer is False:
            return None
        return self.transfer_to_engine

    @property
    def engine_owns_offer(self) -> bool:
        """Return whether EngineCore may own cleanup responsibility.

        :returns: Whether EngineCore committed scheduler ownership.
        """
        return self._engine_owns_offer

    def transfer_to_engine(self) -> None:
        """Transfer remote-prefill cleanup responsibility to EngineCore."""
        self._engine_owns_offer = True


def _validate_kv_transfer_request_options(
    kv_transfer_params: dict[str, Any] | None,
    *,
    use_beam_search: bool,
    stream: bool,
    n: int,
) -> str | None:
    """Validate sampling options against the KV-transfer wire contract.

    :param kv_transfer_params: Optional producer seed or consumer offer.
    :param use_beam_search: Whether the request uses frontend beam search.
    :param stream: Whether the response uses a streaming wire format.
    :param n: Effective number of sampled sequences.
    :returns: Validation error message, or ``None`` when the contract is valid.
    """
    if kv_transfer_params is None:
        return None
    if use_beam_search:
        return "KV-transfer requests are not supported with beam search"

    do_remote_decode = kv_transfer_params.get("do_remote_decode") is True
    do_remote_prefill = kv_transfer_params.get("do_remote_prefill") is True
    if do_remote_decode == do_remote_prefill:
        return (
            "KV-transfer requests require exactly one of do_remote_decode "
            "and do_remote_prefill to be true"
        )

    if do_remote_decode:
        if stream:
            return "KV-transfer producer requests are not supported with streaming"
        if n != 1:
            return "KV-transfer producer requests require n=1"
        return None

    expected_consumers = kv_transfer_params.get("expected_consumers", 1)
    if (
        isinstance(expected_consumers, bool)
        or not isinstance(expected_consumers, int)
        or expected_consumers <= 0
    ):
        return (
            "KV-transfer consumer requests require a positive integer "
            "expected_consumers"
        )
    if n != expected_consumers:
        return (
            "KV-transfer consumer requests require n to equal "
            f"expected_consumers ({expected_consumers}), got {n}"
        )
    return None


@dataclass(kw_only=True)
class ServeContext(Generic[RequestT]):
    request: RequestT
    raw_request: Request | None = None
    model_name: str
    request_id: str
    created_time: int = field(default_factory=lambda: int(time.time()))
    lora_request: LoRARequest | None = None
    engine_inputs: list[EngineInput] | None = None
    model_config = ConfigDict(arbitrary_types_allowed=True)


class OpenAIServing(BaseServing, BeamSearchOnlineMixin):
    request_id_prefix: ClassVar[str] = """
    A short string prepended to every request’s ID.
    """

    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        *,
        request_logger: RequestLogger | None,
        return_tokens_as_token_ids: bool = False,
    ):
        super().__init__(
            models=models,
            model_config=engine_client.model_config,
            request_logger=request_logger,
        )

        self.engine_client = engine_client
        self.return_tokens_as_token_ids = return_tokens_as_token_ids
        self.renderer = engine_client.renderer
        self.input_processor = engine_client.input_processor
        self.has_kv_connector = engine_client.vllm_config.kv_transfer_config is not None

        # Computed once at startup (cached by ``vllm_config`` identity) and
        # stamped on non-streaming responses. Streaming chunks deliberately
        # omit it to avoid per-chunk overhead.
        from vllm.entrypoints.serve.utils.fingerprint import get_system_fingerprint

        try:
            self.system_fingerprint: str | None = get_system_fingerprint(
                engine_client.vllm_config
            )
        except Exception:
            logger.warning(
                "Unable to compute the system fingerprint\n%s",
                traceback.format_exc(),
            )
            self.system_fingerprint = None

    def _create_kv_transfer_admission(
        self,
        kv_transfer_params: dict[str, Any] | None,
    ) -> _KVTransferAdmission:
        """Create the ownership boundary for a remote-prefill offer.

        :param kv_transfer_params: Optional remote-prefill offer metadata.
        :returns: Admission state whose callback preserves the ordinary ADD
            fast path when no offer exists.
        """
        tracks_remote_offer = (
            self.has_kv_connector
            and kv_transfer_params is not None
            and kv_transfer_params.get("do_remote_prefill") is True
        )
        return _KVTransferAdmission(tracks_remote_offer)

    def create_streaming_error_response(
        self,
        message: str | Exception,
        err_type: str = "BadRequestError",
        status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
        param: str | None = None,
    ) -> str:
        json_str = json.dumps(
            self.create_error_response(
                message=message,
                err_type=err_type,
                status_code=status_code,
                param=param,
            ).model_dump()
        )
        return json_str

    def _raise_if_error(self, finish_reason: str | None, request_id: str) -> None:
        """Raise GenerationError if finish_reason indicates an error."""
        if finish_reason == "error":
            logger.error(
                "Request %s failed with an internal error during generation",
                request_id,
            )
            raise GenerationError("Internal server error")

    def _convert_generation_error_to_streaming_response(
        self, e: GenerationError
    ) -> str:
        """Convert GenerationError to streaming error response."""
        return self.create_streaming_error_response(
            str(e),
            err_type="InternalServerError",
            status_code=e.status_code,
        )

    async def _get_trace_headers(
        self,
        headers: Headers,
    ) -> Mapping[str, str] | None:
        is_tracing_enabled = await self.engine_client.is_tracing_enabled()

        if is_tracing_enabled:
            return extract_trace_headers(headers)

        if contains_trace_headers(headers):
            log_tracing_disabled_warning()

        return None

    @staticmethod
    def _get_data_parallel_rank(raw_request: Request | None) -> int | None:
        """Pulls the data parallel rank from a header, if provided"""
        if raw_request is None:
            return None

        rank_str = raw_request.headers.get("X-data-parallel-rank")
        if rank_str is None:
            return None

        try:
            return int(rank_str)
        except ValueError:
            return None

    async def _with_kv_transfer_rejection_cleanup(
        self,
        awaitable: Awaitable[_T],
        request_id: str,
        kv_transfer_params: dict[str, Any] | None,
        raw_request: Request | None,
        admission: _KVTransferAdmission,
    ) -> _T:
        """Release a remote-prefill offer only while serving still owns it.

        :param awaitable: Endpoint coroutine that prepares and runs generation.
        :param request_id: Serving-layer request identifier.
        :param kv_transfer_params: Optional remote-prefill offer metadata.
        :param raw_request: HTTP request used for data-parallel routing.
        :param admission: Explicit serving-to-engine ownership boundary.
        :returns: Endpoint result.
        """
        if (
            self.has_kv_connector is False
            or kv_transfer_params is None
            or kv_transfer_params.get("do_remote_prefill") is not True
        ):
            return await awaitable

        try:
            return await awaitable
        finally:
            if admission.engine_owns_offer is False:
                await self._notify_kv_transfer_offer_rejected(
                    request_id,
                    kv_transfer_params,
                    raw_request,
                    "serving rejected request before generation ownership transfer",
                )

    async def _notify_kv_transfer_offer_rejected(
        self,
        request_id: str,
        kv_transfer_params: dict[str, Any],
        raw_request: Request | None,
        reason: str,
    ) -> None:
        """Notify the connector that no EngineCore request consumed an offer.

        :param request_id: Serving-layer request identifier.
        :param kv_transfer_params: Remote-prefill offer metadata.
        :param raw_request: HTTP request used for data-parallel routing.
        :param reason: Diagnostic rejection reason.
        """
        try:
            handled = await self.engine_client.notify_kv_transfer_request_rejected(
                request_id,
                kv_transfer_params,
                reason,
                data_parallel_rank=self._get_data_parallel_rank(raw_request),
            )
        except Exception:
            logger.error(
                "Failed to notify KV connector about rejected request %s\n%s",
                request_id,
                traceback.format_exc(),
            )
            return

        if handled is False:
            logger.error(
                "No KV connector accepted rejected remote-prefill request %s; "
                "producer pages remain pinned",
                request_id,
            )

    @staticmethod
    def _get_decoded_token(
        logprob: Logprob,
        token_id: int,
        tokenizer: TokenizerLike | None,
        return_as_token_id: bool = False,
    ) -> str:
        if return_as_token_id:
            return format_token_id_placeholder(token_id)

        if logprob.decoded_token is not None:
            return logprob.decoded_token

        if tokenizer is None:
            raise ValueError(
                "Unable to get tokenizer because `skip_tokenizer_init=True`"
            )

        return tokenizer.decode([token_id])


def format_token_id_placeholder(token_id: int) -> str:
    return f"token_id:{token_id}"


def resolve_token_id_placeholder(
    token: str, tokenizer: TokenizerLike
) -> tuple[str, list[int] | None]:
    """Decode a 'token_id:N' placeholder back to a token string and UTF-8 bytes.

    Returns (token, None) unchanged if token is not a placeholder.
    This is the inverse of format_token_id_placeholder / _get_decoded_token
    when return_as_token_id=True.
    """
    suffix = token.removeprefix("token_id:")
    if suffix == token:
        return token, None
    try:
        token_id = int(suffix)
    except ValueError:
        return token, None
    token_repr = tokenizer.convert_ids_to_tokens([token_id])[0]
    if token_repr is None:
        logger.warning_once(
            "resolve_token_id_placeholder: token_id %d has no vocab entry; "
            "substituting empty string",
            token_id,
        )
        return "", None
    token_str = tokenizer.convert_tokens_to_string([token_repr])
    return token_str, list(token_str.encode("utf-8", errors="replace"))


def clamp_prompt_logprobs(
    prompt_logprobs: PromptLogprobs | None,
) -> PromptLogprobs | None:
    if prompt_logprobs is None:
        return prompt_logprobs

    for logprob_dict in prompt_logprobs:
        if logprob_dict is None:
            continue
        for logprob_values in logprob_dict.values():
            if logprob_values.logprob == float("-inf"):
                logprob_values.logprob = -9999.0
    return prompt_logprobs
