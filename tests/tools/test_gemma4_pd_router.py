"""Host-only tests for the Gemma 4 prefill/decode router."""

import asyncio
import json
import sys
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest
from aiohttp import web

from tools.gemma4_pd.pd_router import (
    BACKEND_KEEPALIVE_TIMEOUT_S,
    PDRouter,
    parse_args,
)


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
    decode_tp_size: int = 1,
    trace: bool = False,
    require_prefill: bool = False,
) -> PDRouter:
    """:param decode_attempts: Maximum decode dispatch attempts.
    :param decode_tp_size: Tensor-parallel size of each decode backend.
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
        decode_tp_size=decode_tp_size,
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
    session.post.side_effect = aiohttp.ClientConnectionError("prefill socket closed")
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
    session.post.assert_called_once()
    assert router._stats.pd_success == 0
    assert router._stats.prefill_unavailable == 1
    assert router._prefill[0].in_flight == 0
    assert router._prefill[0].failed == 1

    lookup_request = cast(web.Request, _TraceRequest(request_id))
    lookup_response = asyncio.run(router.trace_lookup(lookup_request))
    assert lookup_response.status == 200
    assert lookup_response.body is not None
    lookup = json.loads(lookup_response.body)
    assert lookup["p_error_type"] == "ClientConnectionError"
    assert lookup["p_error_message"] == "prefill socket closed"


def test_backend_pool_expires_before_vllm_idle_connections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retire pooled sockets before vLLM's five-second idle cutoff."""

    router = _router()
    sweep_health = AsyncMock()
    monkeypatch.setattr(router, "_sweep_health", sweep_health)

    async def exercise() -> None:
        await router.start()
        session = router._session
        assert session is not None
        connector = session.connector
        assert isinstance(connector, aiohttp.TCPConnector)
        assert session.connector_owner is True
        assert connector._keepalive_timeout == BACKEND_KEEPALIVE_TIMEOUT_S
        assert BACKEND_KEEPALIVE_TIMEOUT_S == 1.0
        assert BACKEND_KEEPALIVE_TIMEOUT_S < 5.0
        await router.close()
        await asyncio.sleep(0)
        assert session.closed is True
        assert connector.closed is True

    asyncio.run(exercise())
    sweep_health.assert_awaited_once()


def test_backend_pool_reuses_bursts_but_not_idle_connections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep active and burst traffic while retiring idle backend sockets."""

    router = _router()
    monkeypatch.setattr(router, "_sweep_health", AsyncMock())
    connections: list[asyncio.BaseTransport] = []

    async def handle(request: web.Request) -> web.Response:
        transport = request.transport
        assert transport is not None
        connections.append(transport)
        await request.read()
        if request.path == "/slow":
            await asyncio.sleep(BACKEND_KEEPALIVE_TIMEOUT_S + 0.1)
        return web.Response(text="ok")

    async def exercise() -> None:
        app = web.Application()
        app.router.add_post("/{path:.*}", handle)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = runner.addresses[0][1]
        try:
            await router.start()
            session = router._session
            assert session is not None
            async with session.post(f"http://127.0.0.1:{port}/fast") as response:
                assert await response.text() == "ok"
            async with session.post(f"http://127.0.0.1:{port}/fast") as response:
                assert await response.text() == "ok"
            assert connections[0] is connections[1]

            await asyncio.sleep(BACKEND_KEEPALIVE_TIMEOUT_S + 0.1)
            async with session.post(f"http://127.0.0.1:{port}/fast") as response:
                assert await response.text() == "ok"
            assert connections[2] is not connections[1]

            async with session.post(f"http://127.0.0.1:{port}/slow") as response:
                assert await response.text() == "ok"
            assert connections[3] is connections[2]
        finally:
            await router.close()
            await runner.cleanup()

    asyncio.run(exercise())


def test_ambiguous_prefill_disconnect_is_never_retried() -> None:
    """Refuse a duplicate KV offer after the backend reads a full POST."""

    dispatch_count = 0
    router = _router(trace=True, require_prefill=True)

    async def disconnect_after_request(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        nonlocal dispatch_count
        header = await reader.readuntil(b"\r\n\r\n")
        content_length = 0
        for line in header.split(b"\r\n"):
            name, separator, value = line.partition(b":")
            if separator == b":" and name.lower() == b"content-length":
                content_length = int(value.strip())
        await reader.readexactly(content_length)
        dispatch_count += 1
        writer.close()
        await writer.wait_closed()

    async def exercise() -> web.StreamResponse:
        server = await asyncio.start_server(disconnect_after_request, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        router._prefill[0].url = f"http://127.0.0.1:{port}"
        router._prefill[0].healthy = True
        router._session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(
                keepalive_timeout=BACKEND_KEEPALIVE_TIMEOUT_S
            )
        )
        request = cast(
            web.Request,
            _CompletionRequest(
                "ambiguous-prefill",
                {"messages": [{"role": "user", "content": "hello"}]},
            ),
        )
        try:
            return await router.handle_completions(request)
        finally:
            await router._session.close()
            server.close()
            await server.wait_closed()

    response = asyncio.run(exercise())

    assert response.status == 503
    assert dispatch_count == 1
    assert router._stats.pd_success == 0
    assert router._stats.prefill_unavailable == 1
    assert router._prefill[0].completed == 1
    assert router._prefill[0].failed == 1
    assert router._prefill[0].in_flight == 0


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


def test_decode_tp_size_must_be_positive() -> None:
    """A router cannot advertise an empty decoder topology."""

    with pytest.raises(ValueError, match="decode_tp_size must be a positive integer"):
        _router(decode_tp_size=0)


def test_decode_tp_size_cli_defaults_to_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The deployment CLI defaults to a single-rank decoder."""

    monkeypatch.setattr(
        sys,
        "argv",
        ["pd_router.py", "--decode", "http://decode"],
    )

    assert parse_args().decode_tp_size == 1


def test_prefill_body_seeds_expected_consumers_before_collapsing_fanout() -> None:
    """The producer receives fan-out and topology before serving the offer."""

    router = _router(decode_tp_size=4)

    prefill_body = router._build_prefill_body(
        {
            "messages": [{"role": "user", "content": "hello"}],
            "n": 8,
        }
    )

    assert prefill_body["n"] == 1
    assert prefill_body["kv_transfer_params"]["expected_consumers"] == 8
    assert prefill_body["kv_transfer_params"]["consumer_tp_size"] == 4


@pytest.mark.parametrize(
    ("returned_params", "is_valid"),
    [
        ({"remote_engine_id": "producer", "consumer_tp_size": 4}, False),
        (
            {
                "remote_engine_id": "producer",
                "expected_consumers": 7,
                "consumer_tp_size": 4,
            },
            False,
        ),
        ({"remote_engine_id": "producer", "expected_consumers": 8}, False),
        (
            {
                "remote_engine_id": "producer",
                "expected_consumers": 8,
                "consumer_tp_size": 2,
            },
            False,
        ),
        (
            {
                "remote_engine_id": "producer",
                "expected_consumers": 8,
                "consumer_tp_size": 4,
            },
            True,
        ),
    ],
    ids=[
        "expected-consumers-absent",
        "expected-consumers-mismatch",
        "consumer-tp-size-absent",
        "consumer-tp-size-mismatch",
        "exact-contract",
    ],
)
def test_prefill_validates_producer_contract(
    returned_params: dict,
    is_valid: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A producer response must preserve the complete seeded contract."""

    router = _router(decode_tp_size=4)
    router._prefill[0].healthy = True
    response = MagicMock()
    response.status = 200
    response.json = AsyncMock(return_value={"kv_transfer_params": returned_params})
    response_context = MagicMock()
    response_context.__aenter__ = AsyncMock(return_value=response)
    response_context.__aexit__ = AsyncMock(return_value=None)
    session = MagicMock()
    session.post.return_value = response_context
    monkeypatch.setattr(router, "_session", session)
    body = {
        "messages": [{"role": "user", "content": "hello"}],
        "n": 8,
    }

    params = asyncio.run(
        router._run_prefill(
            "/v1/chat/completions",
            body,
            "request-eight",
        )
    )

    sent_body = session.post.call_args.kwargs["json"]
    assert sent_body["kv_transfer_params"]["expected_consumers"] == 8
    assert sent_body["kv_transfer_params"]["consumer_tp_size"] == 4
    if is_valid:
        assert params == returned_params
        assert router._prefill[0].failed == 0
        return
    assert params is None
    assert router._prefill[0].failed == 1
