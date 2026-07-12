import base64
import io
import json
import threading
import traceback
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import replace
from email.message import Message
from enum import StrEnum
from pathlib import Path

import pytest

from tools.gemma4_cert.artifact import ArmDisposition, ArtifactArm
from tools.gemma4_cert.common import (
    CertificationError,
    canonical_json_bytes,
    sha256_bytes,
)
from tools.gemma4_cert.recorder import (
    CertificationRecorder,
    HttpExchange,
    RequestPlan,
    Transport,
    TransportError,
    TransportErrorKind,
    UrllibTransport,
    _NoRedirectHandler,
    load_request_plans,
)

from .conftest import certification_plan


class FakeTransport:
    def __init__(self, result: HttpExchange | TransportError) -> None:
        self.result = result
        self.requests: list[tuple[str, bytes, Mapping[str, str], float]] = []

    def send(
        self,
        endpoint: str,
        body: bytes,
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> HttpExchange:
        self.requests.append((endpoint, body, headers, timeout_seconds))
        if isinstance(self.result, TransportError):
            raise self.result
        return self.result


def test_urllib_transport_disables_proxies_and_redirects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:8080")
    transport = UrllibTransport()
    proxy_handlers = [
        handler
        for handler in transport._opener.handlers
        if isinstance(handler, urllib.request.ProxyHandler)
    ]

    assert proxy_handlers == []
    assert any(
        isinstance(handler, _NoRedirectHandler)
        for handler in transport._opener.handlers
    )


def test_urllib_transport_records_redirect_as_single_response() -> None:
    headers = Message()
    headers["Location"] = "http://other.example/redirected"
    headers["Set-Cookie"] = "first=secret"
    headers["Set-Cookie"] = "second=secret"

    class RedirectOpener:
        def open(
            self,
            request: urllib.request.Request,
            timeout: float,
        ) -> object:
            raise urllib.error.HTTPError(
                request.full_url,
                307,
                "Temporary Redirect",
                headers,
                io.BytesIO(b"redirect-body"),
            )

    transport = UrllibTransport(RedirectOpener())  # type: ignore[arg-type]

    exchange = transport.send(
        "http://router/v1/completions",
        b"{}",
        {"content-type": "application/json"},
        10.0,
    )

    assert exchange.status_code == 307
    assert exchange.header_items == (
        ("Location", "http://other.example/redirected"),
        ("Set-Cookie", "first=secret"),
        ("Set-Cookie", "second=secret"),
    )
    assert exchange.body == b"redirect-body"


def _plan(
    choice_count: int = 3,
    *,
    request_index: int = 0,
    seed: int = 17,
) -> RequestPlan:
    return RequestPlan(
        root_request_id=f"root-{request_index:03d}",
        request_index=request_index,
        body={
            "model": "model",
            "prompt": "payload",
            "n": choice_count,
            "seed": seed,
        },
        planned_choice_indices=tuple(range(choice_count)),
        headers={
            "Authorization": "Bearer secret",
            "X-Gemma4-Cert-Root-Request-Id": "untrusted-override",
        },
    )


class RecorderHarness:
    def __init__(self, arm: ArtifactArm, transport: Transport) -> None:
        self._arm = arm
        self._transport = transport

    def record(self, plan: RequestPlan) -> str:
        with self._arm.writer_session() as writer:
            return CertificationRecorder(
                writer,
                "http://router/v1/completions",
                self._transport,
                timeout_seconds=10.0,
            ).record(plan)


def _recorder(arm: ArtifactArm, transport: Transport) -> RecorderHarness:
    return RecorderHarness(arm, transport)


def _rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_recorder_preserves_raw_response_choices_tokens_and_lineage(
    arm: ArtifactArm,
) -> None:
    body = json.dumps(
        {
            "id": "response-123",
            "choices": [
                {
                    "index": 1,
                    "text": "b",
                    "logprobs": {
                        "tokens": ["b"],
                        "token_logprobs": [-0.2],
                        "top_logprobs": [{"b": -0.2}],
                        "text_offset": [0],
                    },
                },
                {
                    "index": 0,
                    "text": "a",
                    "logprobs": {
                        "content": [{"token": "a", "logprob": -0.1, "bytes": [97]}]
                    },
                },
                {"index": 2, "text": "c", "logprobs": None},
            ],
        },
        separators=(",", ":"),
    ).encode()
    transport = FakeTransport(
        HttpExchange(
            status_code=200,
            header_items=(
                ("X-Gemma4-Router-Attempt-Id", "router-7"),
                ("Set-Cookie", "session=secret"),
                ("X-Api-Key", "response-secret"),
            ),
            body=body,
        )
    )

    _recorder(arm, transport).record(_plan())

    request = _rows(arm.path / "client/requests.jsonl")[0]
    assert request["root_request_id"] == "root-000"
    assert request["router_attempt_id"] is None
    assert request["response_id"] is None
    assert request["headers"]["authorization"] == "<redacted>"
    assert request["redacted_headers"] == ["authorization"]
    assert "sensitive_header_sha256" not in request
    assert request["payload_order"] == 0
    assert request["payload_label"] == "detector"
    assert request["seed"] == 17
    assert request["request_template_sha256"] == arm.plan.payloads[0].sha256
    assert "secret" not in (arm.path / "client/requests.jsonl").read_text()
    assert transport.requests[0][2]["x-gemma4-cert-root-request-id"] == "root-000"
    response = _rows(arm.path / "client/responses.jsonl")[0]
    assert base64.b64decode(response["raw_body_base64"]) == body
    assert response["router_attempt_id"] == "router-7"
    assert response["response_id"] == "response-123"
    assert response["header_items"] == [
        {
            "name": "x-gemma4-router-attempt-id",
            "value": "router-7",
            "redacted": False,
        },
        {"name": "set-cookie", "value": "<redacted>", "redacted": True},
        {"name": "x-api-key", "value": "<redacted>", "redacted": True},
    ]
    assert "response-secret" not in (arm.path / "client/responses.jsonl").read_text()
    choices = _rows(arm.path / "client/choices.jsonl")
    assert [row["planned_choice_index"] for row in choices] == [0, 1, 2]
    assert [row["outcome"] for row in choices] == ["OBSERVED"] * 3
    assert {row["payload_order"] for row in choices} == {0}
    assert {row["seed"] for row in choices} == {17}
    assert choices[0]["raw_logprobs"][0]["content"][0]["token"] == "a"
    tokens = _rows(arm.path / "client/tokens.jsonl")
    assert [row["raw_token_record"]["token"] for row in tokens] == ["a", "b"]
    lineage = _rows(arm.path / "lineage.jsonl")
    assert [row["event_type"] for row in lineage] == [
        "CLIENT_ATTEMPT_STARTED",
        "CLIENT_ATTEMPT_COMPLETED",
    ]
    assert lineage[-1]["router_attempt_id"] == "router-7"
    assert lineage[-1]["p_registration_id"] is None
    assert {row["request_template_sha256"] for row in lineage} == {
        arm.plan.payloads[0].sha256
    }


def test_missing_duplicate_and_out_of_range_choices_remain_explicit(
    arm: ArtifactArm,
) -> None:
    body = json.dumps(
        {
            "id": "response-dup",
            "choices": [
                {"index": 0, "text": "first"},
                {"index": 0, "text": "duplicate"},
                {"index": 9, "text": "out-of-range"},
                {"text": "without-index"},
            ],
        }
    ).encode()
    transport = FakeTransport(HttpExchange(200, (), body))

    _recorder(arm, transport).record(_plan())

    choices = _rows(arm.path / "client/choices.jsonl")
    assert len(choices) == 3
    assert [row["outcome"] for row in choices] == [
        "DUPLICATE",
        "MISSING",
        "MISSING",
    ]
    assert choices[0]["occurrence_count"] == 2
    unexpected = _rows(arm.path / "client/unexpected_choices.jsonl")
    assert [row["raw_choice"].get("index") for row in unexpected] == [9, None]


@pytest.mark.parametrize(
    ("exchange", "expected_outcome"),
    [
        (HttpExchange(200, (), b"not-json"), "INVALID_RESPONSE"),
        (
            HttpExchange(
                200,
                (),
                b'{"id":"response","value":NaN,"choices":[{"index":0}]}',
            ),
            "INVALID_RESPONSE",
        ),
        (
            HttpExchange(
                200,
                (),
                b'{"id":"\\ud800","choices":[{"index":0}]}',
            ),
            "INVALID_RESPONSE",
        ),
        (
            HttpExchange(
                503,
                (),
                b'{"error":{"message":"unavailable"},"choices":[{"index":0}]}',
            ),
            "HTTP_ERROR",
        ),
    ],
)
def test_http_and_json_errors_emit_every_planned_choice(
    arm: ArtifactArm,
    exchange: HttpExchange,
    expected_outcome: str,
) -> None:
    _recorder(arm, FakeTransport(exchange)).record(_plan())

    choices = _rows(arm.path / "client/choices.jsonl")
    assert len(choices) == 3
    assert {row["outcome"] for row in choices} == {expected_outcome}
    response = _rows(arm.path / "client/responses.jsonl")[0]
    assert base64.b64decode(response["raw_body_base64"]) == exchange.body


def test_transport_error_emits_one_row_per_planned_choice(arm: ArtifactArm) -> None:
    transport = FakeTransport(TransportError(TransportErrorKind.TIMEOUT))

    _recorder(arm, transport).record(_plan())

    choices = _rows(arm.path / "client/choices.jsonl")
    assert len(choices) == 3
    assert {row["outcome"] for row in choices} == {"REQUEST_ERROR"}
    response = _rows(arm.path / "client/responses.jsonl")[0]
    assert response["raw_body_base64"] is None
    assert response["transport_error"] == {"kind": "timeout"}


def test_mutated_transport_error_kind_cannot_archive_caller_text(
    arm: ArtifactArm,
) -> None:
    secret = "transport-secret"

    class UntrustedErrorKind(StrEnum):
        LEAK = secret

    error = TransportError(TransportErrorKind.TIMEOUT)
    error.kind = UntrustedErrorKind.LEAK  # type: ignore[assignment]

    with pytest.raises(CertificationError, match="invalid classification"):
        _recorder(arm, FakeTransport(error)).record(_plan())

    for path in arm.path.rglob("*"):
        if path.is_file():
            assert secret.encode() not in path.read_bytes()


def test_unexpected_transport_exception_aborts_without_archiving_diagnostics(
    arm: ArtifactArm,
) -> None:
    secret = "Bearer transport-secret"

    class ExplodingTransport:
        def send(
            self,
            endpoint: str,
            body: bytes,
            headers: Mapping[str, str],
            timeout_seconds: float,
        ) -> HttpExchange:
            raise RuntimeError(f"invalid header value {secret!r}")

    with pytest.raises(RuntimeError):
        _recorder(arm, ExplodingTransport()).record(_plan())

    for path in arm.path.rglob("*"):
        if path.is_file():
            assert secret.encode() not in path.read_bytes()

    disposition = arm.seal(
        ArmDisposition.PASS,
        "unexpected transport failure",
        make_read_only=False,
    )

    assert disposition is ArmDisposition.INVALID


def test_malformed_sensitive_header_is_rejected_before_recording(
    arm: ArtifactArm,
) -> None:
    secret = "Bearer transport-secret\r\nX-Leak: yes"
    plan = replace(_plan(), headers={"Authorization": secret})

    with pytest.raises(CertificationError, match="CR or LF"):
        _recorder(
            arm,
            FakeTransport(HttpExchange(200, (), b"{}")),
        ).record(plan)

    assert not (arm.path / "client/requests.jsonl").exists()
    for path in arm.path.rglob("*"):
        if path.is_file():
            assert secret.encode() not in path.read_bytes()


def test_request_plan_file_requires_complete_unique_denominator(tmp_path: Path) -> None:
    path = tmp_path / "requests.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "root_request_id": "root-1",
                        "request_index": 1,
                        "planned_choice_indices": [0],
                        "body": {"n": 1, "seed": 18},
                    }
                ),
                json.dumps(
                    {
                        "root_request_id": "root-0",
                        "request_index": 0,
                        "planned_choice_indices": [0],
                        "body": {"n": 1, "seed": 17},
                    }
                ),
            ]
        )
    )

    plans = load_request_plans(path)

    assert [plan.root_request_id for plan in plans] == ["root-0", "root-1"]


def test_case_insensitive_duplicate_request_headers_are_rejected(
    arm: ArtifactArm,
) -> None:
    plan = RequestPlan(
        root_request_id="root-000",
        request_index=0,
        body={"model": "model", "prompt": "payload", "n": 3, "seed": 17},
        planned_choice_indices=(0, 1, 2),
        headers={"X-Test": "a", "x-test": "b"},
    )

    with pytest.raises(
        CertificationError, match="duplicate header after normalization"
    ):
        _recorder(arm, FakeTransport(HttpExchange(200, (), b"{}"))).record(plan)


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (
            {"model": "model", "prompt": "payload", "n": 3, "seed": 18},
            "request seed does not match",
        ),
        (
            {"model": "model", "prompt": "different", "n": 3, "seed": 17},
            "request template digest does not match",
        ),
    ],
)
def test_request_must_match_payload_and_seed_binding(
    arm: ArtifactArm,
    body: dict[str, object],
    message: str,
) -> None:
    plan = RequestPlan(
        root_request_id="root-000",
        request_index=0,
        body=body,
        planned_choice_indices=(0, 1, 2),
    )

    with pytest.raises(CertificationError, match=message):
        _recorder(arm, FakeTransport(HttpExchange(200, (), b"{}"))).record(plan)

    assert not (arm.path / "client/requests.jsonl").exists()


def test_seal_recomputes_request_template_binding(tmp_path: Path) -> None:
    arm = ArtifactArm.create(tmp_path / "cert", certification_plan())
    response = HttpExchange(
        200,
        (),
        b'{"id":"response","choices":[{"index":0},{"index":1},{"index":2}]}',
    )
    _recorder(arm, FakeTransport(response)).record(_plan())
    request = _rows(arm.path / "client/requests.jsonl")[0]
    forged_body = dict(request["parsed_body"])
    forged_body["prompt"] = "forged payload"
    forged_raw = canonical_json_bytes(forged_body)
    request["parsed_body"] = forged_body
    request["raw_body_base64"] = base64.b64encode(forged_raw).decode("ascii")
    request["raw_body_sha256"] = sha256_bytes(forged_raw)
    (arm.path / "client/requests.jsonl").write_bytes(canonical_json_bytes(request))

    disposition = arm.seal(
        ArmDisposition.PASS,
        "request row was forged",
        make_read_only=False,
    )

    assert disposition is ArmDisposition.INVALID
    summary = json.loads((arm.path / "summary.json").read_text())
    assert (
        "request line 1 template digest does not match its plan binding"
        in summary["semantic_errors"]
    )
    assert arm.verify().valid


@pytest.mark.parametrize(
    ("tamper", "expected_error"),
    [
        ("raw", "raw body SHA-256 is incorrect"),
        ("sha256", "raw body SHA-256 is incorrect"),
        ("parsed", "raw body does not match parsed_body"),
    ],
)
def test_seal_rejects_tampered_response_evidence(
    tmp_path: Path,
    tamper: str,
    expected_error: str,
) -> None:
    arm = ArtifactArm.create(tmp_path / "cert", certification_plan())
    response = HttpExchange(
        200,
        (),
        b'{"id":"response","choices":[{"index":0},{"index":1},{"index":2}]}',
    )
    _recorder(arm, FakeTransport(response)).record(_plan())
    response_row = _rows(arm.path / "client/responses.jsonl")[0]
    if tamper == "raw":
        response_row["raw_body_base64"] = base64.b64encode(b'{"id":"forged"}').decode(
            "ascii"
        )
    elif tamper == "sha256":
        response_row["raw_body_sha256"] = "0" * 64
    else:
        response_row["parsed_body"] = {"id": "forged", "choices": []}
    (arm.path / "client/responses.jsonl").write_bytes(
        canonical_json_bytes(response_row)
    )

    disposition = arm.seal(
        ArmDisposition.PASS,
        "response row was tampered",
        make_read_only=False,
    )

    assert disposition is ArmDisposition.INVALID
    summary = json.loads((arm.path / "summary.json").read_text())
    assert any(expected_error in error for error in summary["semantic_errors"])
    assert arm.verify().valid


def test_sensitive_header_name_matching_is_fail_closed(arm: ArtifactArm) -> None:
    plan = RequestPlan(
        root_request_id="root-000",
        request_index=0,
        body={"model": "model", "prompt": "payload", "n": 3, "seed": 17},
        planned_choice_indices=(0, 1, 2),
        headers={
            "X-Custom-Access-Token": "token-value",
            "X-Database-Password": "password-value",
            "X-Client-Credential": "credential-value",
            "X-Private-Key": "key-value",
            "X-Api.Key": "punctuated-key-value",
        },
    )
    transport = FakeTransport(
        HttpExchange(
            200,
            (
                ("X-Session-Identifier", "session-value"),
                ("X-Auth+Token", "punctuated-token-value"),
            ),
            b'{"id":"response","choices":[{"index":0},{"index":1},{"index":2}]}',
        )
    )

    _recorder(arm, transport).record(plan)

    archived = (arm.path / "client/requests.jsonl").read_text()
    for secret in (
        "token-value",
        "password-value",
        "credential-value",
        "key-value",
        "punctuated-key-value",
    ):
        assert secret not in archived
    response = (arm.path / "client/responses.jsonl").read_text()
    assert "session-value" not in response
    assert "punctuated-token-value" not in response


@pytest.mark.parametrize(
    ("endpoint", "message"),
    [
        ("http://user:password@router/v1/completions", "userinfo"),
        ("http://router/v1/completions?api_key=secret", "query"),
        ("http://router/v1/completions#secret", "fragment"),
    ],
)
def test_recorder_rejects_secret_bearing_endpoint_components(
    arm: ArtifactArm,
    endpoint: str,
    message: str,
) -> None:
    with pytest.raises(CertificationError, match=message):
        CertificationRecorder(
            arm,
            endpoint,
            FakeTransport(HttpExchange(200, (), b"{}")),
            timeout_seconds=10.0,
        )

    assert not (arm.path / "client/requests.jsonl").exists()


def test_recorder_requires_active_writer_session(arm: ArtifactArm) -> None:
    with pytest.raises(CertificationError, match="active artifact writer"):
        CertificationRecorder(
            arm,
            "http://router/v1/completions",
            FakeTransport(HttpExchange(200, (), b"{}")),
            timeout_seconds=10.0,
        )


def test_complete_recorder_evidence_can_seal_pass(tmp_path: Path) -> None:
    arm = ArtifactArm.create(tmp_path / "cert", certification_plan())
    response = HttpExchange(
        200,
        (("X-Gemma4-Router-Attempt-Id", "router-1"),),
        b'{"id":"response","choices":[{"index":0},{"index":1},{"index":2}]}',
    )

    _recorder(arm, FakeTransport(response)).record(_plan())
    disposition = arm.seal(
        ArmDisposition.PASS,
        "complete recorder evidence",
        make_read_only=False,
    )

    assert disposition is ArmDisposition.PASS
    assert arm.verify().valid


def test_schema_incomplete_required_attestation_forces_invalid(
    tmp_path: Path,
) -> None:
    base_plan = certification_plan()
    plan = replace(
        base_plan,
        required_artifacts=(*base_plan.required_artifacts, "attest/router.json"),
    )
    arm = ArtifactArm.create(tmp_path / "cert", plan)
    response = HttpExchange(
        200,
        (),
        b'{"id":"response","choices":[{"index":0},{"index":1},{"index":2}]}',
    )
    _recorder(arm, FakeTransport(response)).record(_plan())
    arm.write_json("attest/router.json", {})

    disposition = arm.seal(
        ArmDisposition.PASS,
        "attestation was schema-incomplete",
        make_read_only=False,
    )

    assert disposition is ArmDisposition.INVALID
    summary = json.loads((arm.path / "summary.json").read_text())
    assert any(
        "required attestation attest/router.json" in error
        for error in summary["invalid_json"]
    )
    assert arm.verify().valid


def test_partial_recorder_parent_set_forces_invalid(tmp_path: Path) -> None:
    arm = ArtifactArm.create(
        tmp_path / "cert",
        certification_plan(planned_parents=2),
    )
    response = HttpExchange(
        200,
        (),
        b'{"id":"response","choices":[{"index":0},{"index":1},{"index":2}]}',
    )

    _recorder(arm, FakeTransport(response)).record(_plan())
    disposition = arm.seal(
        ArmDisposition.PASS,
        "only one parent ran",
        make_read_only=False,
    )

    assert disposition is ArmDisposition.INVALID
    summary = json.loads((arm.path / "summary.json").read_text())
    assert "missing request row for request_index 1" in summary["semantic_errors"]
    assert arm.verify().valid


def test_double_recording_one_parent_forces_invalid(tmp_path: Path) -> None:
    arm = ArtifactArm.create(tmp_path / "cert", certification_plan())
    response = HttpExchange(
        200,
        (),
        b'{"id":"response","choices":[{"index":0},{"index":1},{"index":2}]}',
    )
    recorder = _recorder(arm, FakeTransport(response))

    recorder.record(_plan())
    recorder.record(_plan())
    disposition = arm.seal(
        ArmDisposition.PASS,
        "parent was accidentally invoked twice",
        make_read_only=False,
    )

    assert disposition is ArmDisposition.INVALID
    summary = json.loads((arm.path / "summary.json").read_text())
    assert "duplicate request row for request_index 0" in summary["semantic_errors"]
    assert arm.verify().valid


def test_partial_choice_and_missing_terminal_lineage_force_invalid(
    tmp_path: Path,
) -> None:
    arm = ArtifactArm.create(tmp_path / "cert", certification_plan())
    response = HttpExchange(
        200,
        (),
        b'{"id":"response","choices":[{"index":0},{"index":1},{"index":2}]}',
    )
    _recorder(arm, FakeTransport(response)).record(_plan())
    choice_lines = (arm.path / "client/choices.jsonl").read_text().splitlines()
    (arm.path / "client/choices.jsonl").write_text("\n".join(choice_lines[:-1]) + "\n")
    lineage_lines = (arm.path / "lineage.jsonl").read_text().splitlines()
    (arm.path / "lineage.jsonl").write_text("\n".join(lineage_lines[:-1]) + "\n")

    disposition = arm.seal(
        ArmDisposition.PASS,
        "recorder was interrupted",
        make_read_only=False,
    )

    assert disposition is ArmDisposition.INVALID
    summary = json.loads((arm.path / "summary.json").read_text())
    assert "missing choice row for key (0, 2)" in summary["semantic_errors"]
    assert "request_index 0 has 0 terminal lineage events" in summary["semantic_errors"]
    assert arm.verify().valid


def test_partial_recorder_void_arm_remains_void(tmp_path: Path) -> None:
    arm = ArtifactArm.create(
        tmp_path / "cert",
        certification_plan(planned_parents=2),
    )

    disposition = arm.seal(
        ArmDisposition.VOID,
        "state predicate was not met",
        make_read_only=False,
    )

    assert disposition is ArmDisposition.VOID
    assert arm.verify().valid


def test_writer_session_blocks_seal_between_request_and_response(
    tmp_path: Path,
) -> None:
    arm = ArtifactArm.create(tmp_path / "cert", certification_plan())
    request_started = threading.Barrier(2)
    release_response = threading.Event()
    seal_started = threading.Event()
    seal_done = threading.Event()
    dispositions: list[ArmDisposition] = []
    thread_errors: list[str] = []

    class BlockingTransport:
        def send(
            self,
            endpoint: str,
            body: bytes,
            headers: Mapping[str, str],
            timeout_seconds: float,
        ) -> HttpExchange:
            request_started.wait(timeout=2.0)
            if not release_response.wait(timeout=2.0):
                raise TimeoutError("test did not release response")
            return HttpExchange(
                200,
                (),
                b'{"id":"response","choices":[{"index":0},{"index":1},{"index":2}]}',
            )

    def record() -> None:
        try:
            with arm.writer_session() as writer:
                CertificationRecorder(
                    writer,
                    "http://router/v1/completions",
                    BlockingTransport(),
                    timeout_seconds=10.0,
                ).record(_plan())
        except Exception:
            thread_errors.append(traceback.format_exc())

    def seal() -> None:
        try:
            seal_started.set()
            dispositions.append(
                arm.seal(
                    ArmDisposition.PASS,
                    "complete batch",
                    make_read_only=False,
                )
            )
        except Exception:
            thread_errors.append(traceback.format_exc())
        finally:
            seal_done.set()

    record_thread = threading.Thread(target=record)
    record_thread.start()
    request_started.wait(timeout=2.0)
    seal_thread = threading.Thread(target=seal)
    seal_thread.start()
    assert seal_started.wait(timeout=2.0)
    assert not seal_done.wait(timeout=0.05)
    release_response.set()
    record_thread.join(timeout=2.0)
    seal_thread.join(timeout=2.0)

    assert not record_thread.is_alive()
    assert not seal_thread.is_alive()
    assert thread_errors == []
    assert dispositions == [ArmDisposition.PASS]
    assert arm.verify().valid


def test_writer_session_blocks_seal_between_parents(tmp_path: Path) -> None:
    arm = ArtifactArm.create(
        tmp_path / "cert",
        certification_plan(planned_parents=2),
    )
    between_parents = threading.Event()
    continue_batch = threading.Event()
    seal_started = threading.Event()
    seal_done = threading.Event()
    dispositions: list[ArmDisposition] = []
    thread_errors: list[str] = []
    response = HttpExchange(
        200,
        (),
        b'{"id":"response","choices":[{"index":0},{"index":1},{"index":2}]}',
    )

    def record_batch() -> None:
        try:
            with arm.writer_session() as writer:
                recorder = CertificationRecorder(
                    writer,
                    "http://router/v1/completions",
                    FakeTransport(response),
                    timeout_seconds=10.0,
                )
                recorder.record(_plan(request_index=0, seed=17))
                between_parents.set()
                if not continue_batch.wait(timeout=2.0):
                    raise TimeoutError("test did not continue batch")
                recorder.record(_plan(request_index=1, seed=18))
        except Exception:
            thread_errors.append(traceback.format_exc())

    def seal() -> None:
        try:
            seal_started.set()
            dispositions.append(
                arm.seal(
                    ArmDisposition.PASS,
                    "complete batch",
                    make_read_only=False,
                )
            )
        except Exception:
            thread_errors.append(traceback.format_exc())
        finally:
            seal_done.set()

    record_thread = threading.Thread(target=record_batch)
    record_thread.start()
    assert between_parents.wait(timeout=2.0)
    seal_thread = threading.Thread(target=seal)
    seal_thread.start()
    assert seal_started.wait(timeout=2.0)
    assert not seal_done.wait(timeout=0.05)
    continue_batch.set()
    record_thread.join(timeout=2.0)
    seal_thread.join(timeout=2.0)

    assert not record_thread.is_alive()
    assert not seal_thread.is_alive()
    assert thread_errors == []
    assert dispositions == [ArmDisposition.PASS]
    assert arm.verify().valid


def test_released_crashed_writer_session_forces_invalid(tmp_path: Path) -> None:
    arm = ArtifactArm.create(tmp_path / "cert", certification_plan())

    class CrashingTransport:
        def send(
            self,
            endpoint: str,
            body: bytes,
            headers: Mapping[str, str],
            timeout_seconds: float,
        ) -> HttpExchange:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt), arm.writer_session() as writer:
        CertificationRecorder(
            writer,
            "http://router/v1/completions",
            CrashingTransport(),
            timeout_seconds=10.0,
        ).record(_plan())

    disposition = arm.seal(
        ArmDisposition.PASS,
        "writer crashed",
        make_read_only=False,
    )

    assert disposition is ArmDisposition.INVALID
    summary = json.loads((arm.path / "summary.json").read_text())
    assert "missing response row for request_index 0" in summary["semantic_errors"]
    assert arm.verify().valid
