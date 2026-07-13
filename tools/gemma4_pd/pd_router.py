#!/usr/bin/env python3
"""Route OpenAI requests across disaggregated vLLM prefill/decode pools.

Dispatch protocol (vLLM NixlConnector, matches the upstream toy proxy):
the request is first sent to a prefill backend with ``max_tokens=1`` and a
``kv_transfer_params`` seed; the populated params from its response are
attached to the original request, which is then streamed from a decode
backend that pulls the KV over NIXL.

Failure semantics: decode backends are complete vLLM instances, so any
prefill-stage failure (no healthy prefiller, timeout, HTTP error) degrades
to monolithic serving on the decode pool instead of failing the request --
unless --require-prefill is set (F2a mega-only decode: the decoders refuse
local prefill, so the router returns 503 and clients wait for P recovery;
the DR posture is deliberate, see SESSION_FINDINGS phase 18 addendum).
"""

import argparse
import asyncio
import json
import re
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass

import aiohttp
from aiohttp import web
from multidict import CIMultiDictProxy

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}

COMPLETION_PATHS = {"/v1/completions", "/v1/chat/completions"}

KV_TRANSFER_SEED = {
    "do_remote_decode": True,
    "do_remote_prefill": False,
    "remote_engine_id": None,
    "remote_block_ids": None,
    "remote_host": None,
    "remote_port": None,
}


@dataclass
class Backend:
    """Runtime state for one vLLM backend.

    :ivar url: Backend base URL.
    :ivar role: Either ``prefill`` or ``decode``.
    :ivar healthy: Result of the most recent health probe.
    :ivar in_flight: Active requests currently assigned to this backend.
    :ivar completed: Completed requests assigned to this backend.
    :ivar failed: Requests that failed while assigned to this backend.
    """

    url: str
    role: str
    healthy: bool = False
    in_flight: int = 0
    completed: int = 0
    failed: int = 0


@dataclass
class RouterStats:
    """Aggregate router counters.

    :ivar pd_success: Requests served through the full prefill->decode path.
    :ivar monolithic_fallback: Requests served decode-only after a prefill
        stage failure or an empty healthy-prefiller pool.
    :ivar decode_retries: Decode dispatches retried before first byte.
    :ivar failures: Requests that returned an error to the client.
    """

    pd_success: int = 0
    monolithic_fallback: int = 0
    prefill_unavailable: int = 0
    decode_retries: int = 0
    failures: int = 0


class PDRouter:
    """Prefill/decode disaggregation router."""

    _prefill: list[Backend]
    _decode: list[Backend]
    _stats: RouterStats
    _lock: asyncio.Lock
    _session: aiohttp.ClientSession | None
    _prefill_timeout_s: float
    _health_interval_s: float
    _min_prefill_chars: int
    _decode_attempts: int
    _health_task: asyncio.Task | None

    def __init__(
        self,
        prefill_urls: tuple[str, ...],
        decode_urls: tuple[str, ...],
        prefill_timeout_s: float,
        health_interval_s: float,
        min_prefill_chars: int,
        max_decode_inflight: int,
        decode_attempts: int,
        trace: bool = False,
        require_prefill: bool = False,
    ) -> None:
        """:param prefill_urls: Prefill backend base URLs.
        :param decode_urls: Decode backend base URLs.
        :param prefill_timeout_s: Total timeout for the prefill stage.
        :param health_interval_s: Interval between backend health sweeps.
        :param min_prefill_chars: Prompts shorter than this skip the
            prefill stage (0 disables the gate).
        :param max_decode_inflight: Admission cap on concurrently
            dispatched decode requests across the pool (0 = uncapped).
        :param decode_attempts: Maximum decode dispatch attempts.
        :param trace: Emit one JSON line per completions request with
            stage timestamps (request-in, prefill send/done, decode
            send/headers/first-chunk, done) for pipeline profiling.
        :raises ValueError: If decode_attempts is less than one.
        """

        if decode_attempts < 1:
            raise ValueError("decode_attempts must be at least 1")
        self._trace = trace
        # Recent per-request traces for GET /trace/{rid} (bounded; the
        # end-of-stream facts -- decode duration, completion tokens --
        # cannot ride in the same response's headers, which are sent
        # before the body exists).
        self._recent: OrderedDict[str, dict] = OrderedDict()
        self._prefill = [Backend(url=u, role="prefill") for u in prefill_urls]
        self._decode = [Backend(url=u, role="decode") for u in decode_urls]
        self._stats = RouterStats()
        self._lock = asyncio.Lock()
        self._session = None
        self._prefill_timeout_s = prefill_timeout_s
        self._health_interval_s = health_interval_s
        self._min_prefill_chars = min_prefill_chars
        self._decode_attempts = decode_attempts
        self._require_prefill = require_prefill
        self._decode_gate = (
            asyncio.Semaphore(max_decode_inflight)
            if max_decode_inflight > 0
            else None
        )
        self._health_task = None

    async def start(self) -> None:
        """Create the client session and start the health sweep loop."""

        # sock_read bounds how long a backend may go silent mid-request
        # (headers, first byte, or between stream chunks). Without it, a
        # killed/hung D pins the handler -- and its in_flight
        # reservation -- forever (the phase-22 stuck-counter incident).
        # 600s is far above any legitimate TTFT/generation gap here.
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=30.0,
                                        sock_read=600.0)
        self._session = aiohttp.ClientSession(timeout=timeout)
        await self._sweep_health()
        self._health_task = asyncio.create_task(self._health_loop())

    async def close(self) -> None:
        """Stop the health loop and close the client session."""

        if self._health_task is not None:
            self._health_task.cancel()
        if self._session is not None:
            await self._session.close()

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    async def _health_loop(self) -> None:
        """Sweep backend health forever."""

        while True:
            await asyncio.sleep(self._health_interval_s)
            try:
                await self._sweep_health()
            except Exception as exc:  # noqa: BLE001 - keep the loop alive
                print(f"health sweep error: {exc}", flush=True)

    async def _sweep_health(self) -> None:
        """Probe every backend health endpoint once."""

        backends = self._prefill + self._decode
        results = await asyncio.gather(
            *(self._probe_health(backend.url) for backend in backends)
        )
        for backend, healthy in zip(backends, results):
            if backend.healthy is not healthy:
                print(
                    json.dumps(
                        {
                            "event": "backend_health_change",
                            "url": backend.url,
                            "role": backend.role,
                            "healthy": healthy,
                            "ts": time.time(),
                        }
                    ),
                    flush=True,
                )
            backend.healthy = healthy

    async def _probe_health(self, url: str) -> bool:
        """Check one backend health endpoint.

        :param url: Backend base URL.
        :returns: Whether the backend reported healthy.
        """

        if self._session is None:
            return False
        timeout = aiohttp.ClientTimeout(total=5.0)
        try:
            async with self._session.get(f"{url}/health", timeout=timeout) as resp:
                return resp.status == 200
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return False

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    async def _select(self, pool: list[Backend]) -> Backend | None:
        """Reserve the least-active healthy backend from a pool.

        :param pool: Candidate backends.
        :returns: Reserved backend, or None when none are healthy.
        """

        async with self._lock:
            healthy = [b for b in pool if b.healthy is True]
            if len(healthy) == 0:
                return None
            backend = min(healthy, key=lambda b: (b.in_flight, b.completed, b.url))
            backend.in_flight += 1
            return backend

    def _release(self, backend: Backend, failed: bool) -> None:
        """Release a backend reservation.

        Synchronous on purpose: releases run in ``finally`` blocks, and
        a cancelled handler (client disconnect under
        handler_cancellation) raises CancelledError at the first await
        inside its finally -- an async lock here could skip the
        decrement and leak the reservation. Plain int ops in a
        single-threaded event loop need no lock.

        :param backend: Reserved backend.
        :param failed: Whether the request failed on this backend.
        """

        backend.in_flight -= 1
        backend.completed += 1
        if failed is True:
            backend.failed += 1
        if backend.in_flight < 0:
            print(json.dumps({
                "event": "IN_FLIGHT_UNDERFLOW",
                "url": backend.url,
                "in_flight": backend.in_flight,
                "ts": time.time(),
            }), flush=True)
            backend.in_flight = 0

    # ------------------------------------------------------------------
    # Prefill stage
    # ------------------------------------------------------------------

    def _prompt_chars(self, body: dict) -> int:
        """Approximate the prompt size of a request in characters.

        :param body: Parsed request body.
        :returns: Character count of the prompt or messages content.
        """

        prompt = body.get("prompt")
        if isinstance(prompt, str):
            return len(prompt)
        if isinstance(prompt, list):
            return sum(len(p) for p in prompt if isinstance(p, str))
        total = 0
        for message in body.get("messages", []):
            content = message.get("content", "")
            if isinstance(content, str):
                total += len(content)
        return total

    def _build_prefill_body(self, body: dict) -> dict:
        """Create the prefill-stage request body.

        :param body: Original request body.
        :returns: Prefill request body.
        """

        p_body = dict(body)
        p_body["kv_transfer_params"] = dict(KV_TRANSFER_SEED)
        p_body["stream"] = False
        p_body["max_tokens"] = 1
        # Prefill materializes prompt KV once; n>1 would create n sibling
        # leases on P for the same pages (n-1 never retrieved/heartbeated).
        p_body["n"] = 1
        p_body.pop("best_of", None)
        if "max_completion_tokens" in p_body:
            p_body["max_completion_tokens"] = 1
        p_body.pop("stream_options", None)
        p_body.pop("min_tokens", None)
        p_body.pop("min_completion_tokens", None)
        return p_body

    async def _run_prefill(
        self, path: str, body: dict, request_id: str, tr: dict | None = None
    ) -> dict | None:
        """Run the prefill stage and return populated KV transfer params.

        :param path: Completions endpoint path.
        :param body: Original request body.
        :param request_id: Shared request id for both stages.
        :param tr: Optional trace dict; stage timestamps are added.
        :returns: KV transfer params for the decode stage, or None when the
            request should fall back to monolithic decode-side serving.
        """

        if self._session is None:
            return None
        if (
            self._min_prefill_chars > 0
            and self._prompt_chars(body) < self._min_prefill_chars
        ):
            return None
        backend = await self._select(self._prefill)
        if backend is None:
            return None
        failed = True
        try:
            timeout = aiohttp.ClientTimeout(total=self._prefill_timeout_s)
            if tr is not None:
                tr["t_p_sent"] = time.time()
            async with self._session.post(
                f"{backend.url}{path}",
                json=self._build_prefill_body(body),
                headers={"X-Request-Id": request_id},
                timeout=timeout,
            ) as resp:
                if tr is not None:
                    tr["t_p_hdr"] = time.time()
                    tr["p_status"] = resp.status
                if resp.status != 200:
                    return None
                payload = await resp.json()
                if tr is not None:
                    tr["t_p_done"] = time.time()
                failed = False
                if tr is not None:
                    usage = payload.get("usage")
                    if isinstance(usage, dict):
                        tr["prompt_tokens"] = usage.get("prompt_tokens")
                params = payload.get("kv_transfer_params")
                if params and params.get("remote_block_ids"):
                    _rb = params["remote_block_ids"]
                    print(json.dumps({
                        "event": "sp_dbg_remote_lens",
                        "lens": [len(g) for g in _rb],
                        "g10_first4": (list(_rb[10][:4])
                                       if len(_rb) > 10 else []),
                    }), flush=True)
                if isinstance(params, dict) and len(params) > 0:
                    if tr is not None:
                        blocks = params.get("remote_block_ids")
                        tr["n_blocks"] = (
                            len(blocks) if isinstance(blocks, list) else 0
                        )
                    # n>1 fans out into n D-side children that each pull
                    # this rid; the producer must not free until all have
                    # read (or the lease expires).
                    try:
                        params["expected_consumers"] = max(
                            1, int(body.get("n") or 1)
                        )
                    except (TypeError, ValueError):
                        params["expected_consumers"] = 1
                    return params
                return None
        except (
            aiohttp.ClientError,
            asyncio.TimeoutError,
            json.JSONDecodeError,
        ) as exc:
            if tr is not None:
                tr["p_error_type"] = type(exc).__name__
                tr["p_error_message"] = str(exc)
            return None
        finally:
            self._release(backend, failed=failed)

    # ------------------------------------------------------------------
    # Decode stage
    # ------------------------------------------------------------------

    async def _stream_decode(
        self,
        request: web.Request,
        path: str,
        body: dict,
        request_id: str,
        attempts: int,
        tr: dict | None = None,
    ) -> web.StreamResponse:
        """Stream the decode-stage response to the client.

        Retries on a fresh decode backend when the previous dispatch fails
        before the first response byte.

        :param request: Incoming client request.
        :param path: Completions endpoint path.
        :param body: Decode request body (KV params attached when present).
        :param request_id: Shared request id for both stages.
        :param attempts: Maximum dispatch attempts.
        :param tr: Optional trace dict; stage timestamps are added.
        :returns: Streamed client response.
        """

        assert self._session is not None
        if self._decode_gate is not None:
            async with self._decode_gate:
                return await self._stream_decode_inner(
                    request, path, body, request_id, attempts, tr
                )
        return await self._stream_decode_inner(
            request, path, body, request_id, attempts, tr
        )

    async def _stream_decode_inner(
        self,
        request: web.Request,
        path: str,
        body: dict,
        request_id: str,
        attempts: int,
        tr: dict | None = None,
    ) -> web.StreamResponse:
        """Dispatch and stream one admitted decode request.

        :param request: Incoming client request.
        :param path: Completions endpoint path.
        :param body: Decode request body.
        :param request_id: Shared request id for both stages.
        :param attempts: Maximum dispatch attempts.
        :param tr: Optional trace dict; stage timestamps are added.
        :returns: Streamed client response.
        """

        assert self._session is not None
        last_error = "no healthy decode backend"
        params_consumed = False
        import os as _os_ab
        unsafe_stale = _os_ab.environ.get("PD_ROUTER_UNSAFE_STALE") == "1"
        for attempt in range(attempts):
            if body.get("kv_transfer_params") is not None and params_consumed:
                # The previous dispatch consumed the remote KV
                # registration (its children pulled and P frees at the
                # expected count). A retry MUST carry a fresh prefill's
                # params -- stale ones read freed/reused P pages.
                fresh = await self._run_prefill(
                    path,
                    body,
                    f"{request_id}-r{attempt}",
                    tr,
                )
                if fresh is None and unsafe_stale:
                    # A/B arm: pre-fix behavior, loudly.
                    print(json.dumps({
                        "event": "UNSAFE_STALE_DISPATCH",
                        "rid": request_id,
                        "attempt": attempt,
                        "reason": "fresh prefill failed",
                        "ts": time.time(),
                    }), flush=True)
                    params_consumed = False
                elif fresh is None:
                    last_error = (
                        "re-prefill failed; refusing to dispatch with "
                        "consumed kv_transfer_params"
                    )
                    self._stats.decode_retries += 1
                    continue
                else:
                    body["kv_transfer_params"] = fresh
                    params_consumed = False
            backend = await self._select(self._decode)
            if backend is None:
                break
            failed = True
            streaming_started = False
            try:
                if tr is not None:
                    tr["t_d_sent"] = time.time()
                    tr["d_backend"] = backend.url
                async with self._session.post(
                    f"{backend.url}{path}",
                    json=body,
                    headers={"X-Request-Id": request_id},
                ) as resp:
                    if tr is not None:
                        tr["t_d_hdr"] = time.time()
                        tr["d_status"] = resp.status
                    if resp.status >= 500 and attempt + 1 < attempts:
                        # Whole-request failure before any bytes streamed
                        # (e.g. decode-tier preempt-abort). The remote KV
                        # registration was consumed by the first pull, so
                        # refresh the transfer params with a new prefill
                        # pass, then retry on a fresh decode backend.
                        last_error = f"decode backend status {resp.status}"
                        self._stats.decode_retries += 1
                        params_consumed = True
                        continue
                    headers = self._response_headers(resp.headers)
                    if tr is not None:
                        # Callers take this id to GET /trace/{rid}
                        # once the response completes for the full
                        # stage timings and token counts.
                        headers["X-PD-Request-Id"] = request_id
                    response = web.StreamResponse(
                        status=resp.status,
                        reason=resp.reason,
                        headers=headers,
                    )
                    streaming_started = True
                    await response.prepare(request)
                    last_chunks = (b"", b"")
                    async for chunk in resp.content.iter_any():
                        if tr is not None and "t_d_first" not in tr:
                            tr["t_d_first"] = time.time()
                        await response.write(chunk)
                        if tr is not None:
                            # reference-only rotation: the response
                            # tail carries the usage record, parsed
                            # AFTER write_eof, off the latency path
                            last_chunks = (last_chunks[1], chunk)
                    await response.write_eof()
                    if tr is not None:
                        tr["d_tail"] = last_chunks
                    failed = resp.status >= 500
                    return response
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if streaming_started is True:
                    # Bytes already reached the client; surface the break.
                    raise
                # The dispatch may have reached the backend and pulled
                # before erroring: treat the params as consumed.
                if unsafe_stale and body.get("kv_transfer_params") is not None:
                    # A/B arm: pre-fix behavior retried with the SAME
                    # params and no fresh prefill, loudly.
                    print(json.dumps({
                        "event": "UNSAFE_STALE_DISPATCH",
                        "rid": request_id,
                        "attempt": attempt,
                        "reason": f"exception path: {last_error[:80]}",
                        "ts": time.time(),
                    }), flush=True)
                else:
                    params_consumed = True
                if attempt + 1 < attempts:
                    self._stats.decode_retries += 1
            finally:
                self._release(backend, failed=failed)
        self._stats.failures += 1
        return web.json_response(
            {"error": f"decode dispatch failed: {last_error}"}, status=502
        )

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _finalize_trace(self, tr: dict) -> None:
        """Finalize, retain, and emit one request trace.

        :param tr: Trace record accumulated while serving the request.
        """

        tr["t_done"] = time.time()
        # Decode has already completed, so parsing the retained response
        # tail cannot increase time to first byte or stream latency.
        tail = b"".join(tr.pop("d_tail", ()))[-16384:]
        hits = re.findall(rb'"completion_tokens":\s*(\d+)', tail)
        if len(hits) > 0:
            tr["completion_tokens"] = int(hits[-1])
        self._recent[tr["rid"]] = tr
        self._recent.move_to_end(tr["rid"])
        while len(self._recent) > 4096:
            self._recent.popitem(last=False)
        print(json.dumps(tr), flush=True)

    async def handle_completions(self, request: web.Request) -> web.StreamResponse:
        """Serve one completions request through the PD pipeline.

        :param request: Incoming aiohttp request.
        :returns: Streamed response from the decode stage.
        """

        try:
            body = await request.json()
        except json.JSONDecodeError:
            return web.json_response({"error": "invalid JSON body"}, status=400)
        # vLLM v1 parses but never consumes best_of; anything that could
        # fan out to != n children would break the expected_consumers
        # accounting on P. Strip it so the invariant is structural.
        body.pop("best_of", None)
        request_id = request.headers.get("X-Request-Id", str(uuid.uuid4()))
        tr = None
        if self._trace:
            tr = {
                "event": "trace",
                "rid": request_id,
                "chars": self._prompt_chars(body),
                "t_in": time.time(),
            }
        try:
            params = await self._run_prefill(request.path, body, request_id, tr)
            d_body = dict(body)
            if params is not None:
                d_body["kv_transfer_params"] = params
                self._stats.pd_success += 1
            elif self._require_prefill:
                self._stats.prefill_unavailable += 1
                if tr is not None:
                    tr["pd_path"] = False
                    tr["rejected"] = "prefill_unavailable"
                return web.json_response(
                    {"error": "prefill stage unavailable; retry later"},
                    status=503,
                    headers={"Retry-After": "5"},
                )
            else:
                self._stats.monolithic_fallback += 1
            if tr is not None:
                tr["pd_path"] = params is not None
            return await self._stream_decode(
                request,
                request.path,
                d_body,
                request_id,
                attempts=self._decode_attempts,
                tr=tr,
            )
        finally:
            if tr is not None:
                self._finalize_trace(tr)

    async def handle_passthrough(self, request: web.Request) -> web.StreamResponse:
        """Proxy a non-completions request to a healthy decode backend.

        :param request: Incoming aiohttp request.
        :returns: Proxied response.
        """

        assert self._session is not None
        backend = await self._select(self._decode)
        if backend is None:
            return web.json_response({"error": "no healthy decode backend"}, status=503)
        failed = True
        try:
            data = await request.read()
            async with self._session.request(
                request.method,
                f"{backend.url}{request.rel_url}",
                headers=self._forward_headers(request.headers),
                data=data if len(data) > 0 else None,
            ) as resp:
                payload = await resp.read()
                failed = resp.status >= 500
                return web.Response(
                    status=resp.status,
                    headers=self._response_headers(resp.headers),
                    body=payload,
                )
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            return web.json_response(
                {"error": f"{type(exc).__name__}: {exc}"}, status=502
            )
        finally:
            self._release(backend, failed=failed)

    async def health(self, _request: web.Request) -> web.Response:
        """Report router health.

        The router is healthy while at least one decode backend is healthy;
        prefill backends only affect the reported mode.

        :param _request: Incoming aiohttp request.
        :returns: Health response.
        """

        healthy_p = sum(1 for b in self._prefill if b.healthy is True)
        healthy_d = sum(1 for b in self._decode if b.healthy is True)
        ok = healthy_d > 0
        mode = "pd" if healthy_p > 0 else "monolithic-fallback"
        return web.json_response(
            {
                "status": "ok" if ok else "unavailable",
                "mode": mode,
                "healthy_prefill": healthy_p,
                "healthy_decode": healthy_d,
            },
            status=200 if ok else 503,
        )

    async def trace_lookup(self, request: web.Request) -> web.Response:
        """Return the stored trace for one request id, with derived
        stage durations. Available once the response has completed;
        entries are evicted after the most recent 4096 requests.

        :param request: Incoming aiohttp request.
        :returns: Trace response or 404.
        """

        rid = request.match_info["rid"]
        tr = self._recent.get(rid)
        if tr is None:
            return web.json_response(
                {"error": "unknown or expired request id"}, status=404
            )
        out = dict(tr)

        def ms(a: str, b: str) -> float | None:
            if a in out and b in out:
                return round((out[b] - out[a]) * 1000, 1)
            return None

        out["prefill_stage_ms"] = ms("t_p_sent", "t_p_done")
        out["decode_ttfb_ms"] = ms("t_d_sent", "t_d_first")
        out["decode_stream_ms"] = ms("t_d_first", "t_done")
        out["total_ms"] = ms("t_in", "t_done")
        return web.json_response(out)

    async def stats(self, _request: web.Request) -> web.Response:
        """Report router counters and backend state.

        :param _request: Incoming aiohttp request.
        :returns: Stats response.
        """

        def rows(pool: list[Backend]) -> list[dict]:
            return [
                {
                    "url": b.url,
                    "healthy": b.healthy,
                    "in_flight": b.in_flight,
                    "completed": b.completed,
                    "failed": b.failed,
                }
                for b in pool
            ]

        return web.json_response(
            {
                "pd_success": self._stats.pd_success,
                "monolithic_fallback": self._stats.monolithic_fallback,
                "prefill_unavailable": self._stats.prefill_unavailable,
                "decode_retries": self._stats.decode_retries,
                "failures": self._stats.failures,
                "prefill": rows(self._prefill),
                "decode": rows(self._decode),
            }
        )

    # ------------------------------------------------------------------
    # Header helpers
    # ------------------------------------------------------------------

    def _forward_headers(self, headers: CIMultiDictProxy[str]) -> dict[str, str]:
        """Create forwarded request headers.

        :param headers: Incoming request headers.
        :returns: Forwarded headers.
        """

        return {
            k: v
            for k, v in headers.items()
            if k.lower() not in HOP_BY_HOP_HEADERS
        }

    def _response_headers(self, headers: CIMultiDictProxy[str]) -> dict[str, str]:
        """Create client response headers.

        :param headers: Backend response headers.
        :returns: Response headers.
        """

        return {
            k: v
            for k, v in headers.items()
            if k.lower() not in HOP_BY_HOP_HEADERS
        }


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    :returns: Parsed arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8813)
    parser.add_argument(
        "--prefill", action="append", default=[], help="Prefill base URL. Repeatable."
    )
    parser.add_argument(
        "--decode", action="append", required=True, help="Decode base URL. Repeatable."
    )
    parser.add_argument("--prefill-timeout-s", type=float, default=120.0)
    parser.add_argument("--health-interval-s", type=float, default=5.0)
    parser.add_argument(
        "--min-prefill-chars",
        type=int,
        default=0,
        help="Prompts shorter than this skip the prefill stage (0 = off).",
    )
    parser.add_argument(
        "--max-decode-inflight",
        type=int,
        default=0,
        help="Admission cap on concurrent decode dispatches (0 = off).",
    )
    parser.add_argument(
        "--decode-attempts",
        type=int,
        default=3,
        help="Maximum decode dispatch attempts.",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Emit one JSON line per completions request with stage "
        "timestamps (pipeline profiling).",
    )
    parser.add_argument(
        "--require-prefill",
        action="store_true",
        help="Return 503 when the prefill stage is unavailable instead "
        "of falling back to monolithic decode-side serving (mega-only "
        "decoders refuse local prefill).",
    )
    return parser.parse_args()


def main() -> None:
    """Run the PD router."""

    args = parse_args()
    router = PDRouter(
        prefill_urls=tuple(args.prefill),
        decode_urls=tuple(args.decode),
        prefill_timeout_s=args.prefill_timeout_s,
        health_interval_s=args.health_interval_s,
        min_prefill_chars=args.min_prefill_chars,
        max_decode_inflight=args.max_decode_inflight,
        decode_attempts=args.decode_attempts,
        trace=args.trace,
        require_prefill=args.require_prefill,
    )
    if args.require_prefill and args.min_prefill_chars > 0:
        raise SystemExit(
            "--require-prefill conflicts with --min-prefill-chars "
            "(short prompts would be routed to decode-local prefill)")

    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.router.add_get("/health", router.health)
    app.router.add_get("/stats", router.stats)
    app.router.add_get("/trace/{rid}", router.trace_lookup)
    for path in COMPLETION_PATHS:
        app.router.add_post(path, router.handle_completions)
    app.router.add_route("*", "/{tail:.*}", router.handle_passthrough)

    async def on_startup(_app: web.Application) -> None:
        await router.start()

    async def on_cleanup(_app: web.Application) -> None:
        await router.close()

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    # Cancel handlers on client disconnect: a request whose client
    # gave up must not keep its backend reservation (in_flight)
    # alive; _release is cancellation-safe (synchronous).
    web.run_app(app, host=args.host, port=args.port,
                handler_cancellation=True)


if __name__ == "__main__":
    main()
