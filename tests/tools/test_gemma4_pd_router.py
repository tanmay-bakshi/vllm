"""Host-only tests for the Gemma 4 prefill/decode router."""

import asyncio
import json
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest
from aiohttp import web

from tools.gemma4_pd.pd_router import PDRouter


class _CompletionRequest:
    """Minimal completions request accepted by the router handler."""

    path: str
    headers: dict[str, str]
    _body: dict

    def __init__(self, request_id: str, body: dict) -> None:
        """:param request_id: Stable request identity.
        :param body: JSON body returned to the router.
        """

        self.path = "/v1/chat/completions"
        self.headers = {"X-Request-Id": request_id}
        self._body = body

    async def json(self) -> dict:
        """:returns: Configured JSON request body."""

        return self._body


class _TraceRequest:
    """Minimal trace lookup request accepted by the router handler."""

    match_info: dict[str, str]

    def __init__(self, request_id: str) -> None:
        """:param request_id: Trace identity to retrieve."""

        self.match_info = {"rid": request_id}


def _router(
    *,
    decode_attempts: int = 3,
    trace: bool = False,
    require_prefill: bool = False,
) -> PDRouter:
    """:param decode_attempts: Maximum decode dispatch attempts.
    :param trace: Whether request tracing is enabled.
    :param require_prefill: Whether prefill failure rejects the request.
    :returns: Router configured for host-only tests.
    """

    return PDRouter(
        prefill_urls=("http://prefill",),
        decode_urls=("http://decode",),
        prefill_timeout_s=1.0,
        health_interval_s=1.0,
        min_prefill_chars=0,
        max_decode_inflight=0,
        decode_attempts=decode_attempts,
        trace=trace,
        require_prefill=require_prefill,
    )


def test_rejected_prefill_trace_is_stored_and_emitted_once(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A caught prefill exception remains inspectable after rejection."""

    router = _router(trace=True, require_prefill=True)
    router._prefill[0].healthy = True
    session = MagicMock()
    session.post.side_effect = aiohttp.ClientConnectionError(
        "prefill socket closed"
    )
    monkeypatch.setattr(router, "_session", session)
    request_id = "target-request"
    request = cast(
        web.Request,
        _CompletionRequest(
            request_id,
            {"messages": [{"role": "user", "content": "hello"}]},
        ),
    )

    response = asyncio.run(router.handle_completions(request))

    assert response.status == 503
    emitted_lines = capsys.readouterr().out.splitlines()
    assert len(emitted_lines) == 1
    emitted = json.loads(emitted_lines[0])
    stored = router._recent[request_id]
    assert emitted == stored
    assert stored["rejected"] == "prefill_unavailable"
    assert stored["p_error_type"] == "ClientConnectionError"
    assert stored["p_error_message"] == "prefill socket closed"
    assert "t_done" in stored

    lookup_request = cast(web.Request, _TraceRequest(request_id))
    lookup_response = asyncio.run(router.trace_lookup(lookup_request))
    assert lookup_response.status == 200
    assert lookup_response.body is not None
    lookup = json.loads(lookup_response.body)
    assert lookup["p_error_type"] == "ClientConnectionError"
    assert lookup["p_error_message"] == "prefill socket closed"


def test_configured_decode_attempts_reach_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The configured decode attempt count controls request dispatch."""

    router = _router(decode_attempts=1)
    run_prefill = AsyncMock(return_value=None)
    stream_decode = AsyncMock(return_value=web.Response(status=200))
    monkeypatch.setattr(router, "_run_prefill", run_prefill)
    monkeypatch.setattr(router, "_stream_decode", stream_decode)
    body = {"messages": [{"role": "user", "content": "hello"}]}
    request = cast(web.Request, _CompletionRequest("request-one", body))

    response = asyncio.run(router.handle_completions(request))

    assert response.status == 200
    stream_decode.assert_awaited_once_with(
        request,
        request.path,
        body,
        "request-one",
        attempts=1,
        tr=None,
    )


def test_decode_attempts_must_be_positive() -> None:
    """A router cannot be configured to skip decode dispatch entirely."""

    with pytest.raises(ValueError, match="decode_attempts must be at least 1"):
        _router(decode_attempts=0)
