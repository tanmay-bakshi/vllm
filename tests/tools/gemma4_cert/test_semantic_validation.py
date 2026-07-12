import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from tools.gemma4_cert.artifact import ArmDisposition, ArtifactArm
from tools.gemma4_cert.common import CertificationError, canonical_json_bytes
from tools.gemma4_cert.recorder import (
    CertificationRecorder,
    HttpExchange,
    RequestPlan,
    TransportError,
    TransportErrorKind,
)

from .conftest import certification_plan


class StaticTransport:
    def __init__(self, result: HttpExchange | TransportError) -> None:
        self._result = result

    def send(
        self,
        endpoint: str,
        body: bytes,
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> HttpExchange:
        if isinstance(self._result, TransportError):
            raise self._result
        return self._result


def _plan(request_index: int = 0, seed: int = 17) -> RequestPlan:
    return RequestPlan(
        root_request_id=f"root-{request_index:03d}",
        request_index=request_index,
        body={
            "model": "model",
            "prompt": "payload",
            "n": 3,
            "seed": seed,
        },
        planned_choice_indices=(0, 1, 2),
    )


def _record(tmp_path: Path, result: HttpExchange | TransportError) -> ArtifactArm:
    arm = ArtifactArm.create(tmp_path / "cert", certification_plan())
    with arm.writer_session() as writer:
        recorder = CertificationRecorder(
            writer,
            "http://router/v1/completions",
            StaticTransport(result),
            timeout_seconds=10.0,
        )
        recorder.record(_plan())
    return arm


def _complex_arm(tmp_path: Path) -> ArtifactArm:
    body = canonical_json_bytes(
        {
            "id": "response-123",
            "choices": [
                {
                    "index": 0,
                    "text": "a",
                    "logprobs": {
                        "content": [
                            {"token": "a", "logprob": -0.1},
                            {"token": "b", "logprob": -0.2},
                        ]
                    },
                },
                {
                    "index": 1,
                    "text": "b",
                    "logprobs": {
                        "tokens": ["c", "d"],
                        "token_logprobs": [-0.3, -0.4],
                        "top_logprobs": [{"c": -0.3}, {"d": -0.4}],
                        "text_offset": [0, 1],
                    },
                },
                {"index": 2, "text": "c", "logprobs": None},
                {"index": 9, "text": "outside-plan"},
                {"text": "missing-index"},
            ],
        }
    )
    return _record(
        tmp_path,
        HttpExchange(
            status_code=200,
            header_items=(("X-Gemma4-Router-Attempt-Id", "router-7"),),
            body=body,
        ),
    )


def _simple_arm(tmp_path: Path) -> ArtifactArm:
    return _record(
        tmp_path,
        HttpExchange(
            status_code=200,
            header_items=(("X-Gemma4-Router-Attempt-Id", "router-7"),),
            body=canonical_json_bytes(
                {
                    "id": "response-123",
                    "choices": [
                        {"index": 0},
                        {"index": 1},
                        {"index": 2},
                    ],
                }
            ),
        ),
    )


def _rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    for sequence, row in enumerate(rows, start=1):
        row["sequence"] = sequence
    _write_rows_exact(path, rows)


def _write_rows_exact(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_bytes(b"".join(canonical_json_bytes(row) for row in rows))


def _seal_invalid(arm: ArtifactArm) -> list[str]:
    disposition = arm.seal(
        ArmDisposition.PASS,
        "semantic evidence was tampered",
        make_read_only=False,
    )
    assert disposition is ArmDisposition.INVALID
    summary = json.loads((arm.path / "summary.json").read_bytes())
    semantic_errors = summary["semantic_errors"]
    assert isinstance(semantic_errors, list)
    assert all(isinstance(error, str) for error in semantic_errors)
    return semantic_errors


def test_complete_rederived_evidence_can_seal_pass(tmp_path: Path) -> None:
    arm = _complex_arm(tmp_path)

    disposition = arm.seal(
        ArmDisposition.PASS,
        "all response-dependent evidence is internally proven",
        make_read_only=False,
    )

    assert disposition is ArmDisposition.PASS


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("outcome", "MISSING"),
        ("occurrence_count", 2),
        ("raw_choices", []),
        ("raw_logprobs", []),
        ("response_id", "forged-response"),
        ("router_attempt_id", "forged-router-attempt"),
    ],
)
def test_choice_rows_are_rederived_from_raw_response(
    tmp_path: Path,
    field: str,
    replacement: object,
) -> None:
    arm = _complex_arm(tmp_path)
    path = arm.path / "client/choices.jsonl"
    rows = _rows(path)
    rows[0][field] = replacement
    _write_rows(path, rows)

    errors = _seal_invalid(arm)

    assert any(f"choice line 1 {field}" in error for error in errors)


def test_reordered_choice_rows_are_invalid_even_after_resequencing(
    tmp_path: Path,
) -> None:
    arm = _complex_arm(tmp_path)
    path = arm.path / "client/choices.jsonl"
    rows = _rows(path)
    rows[0], rows[1] = rows[1], rows[0]
    _write_rows(path, rows)

    errors = _seal_invalid(arm)

    assert any("row order" in error for error in errors)


@pytest.mark.parametrize("mutation", ["missing-field", "extra-field"])
def test_malformed_derived_row_field_sets_force_invalid(
    tmp_path: Path,
    mutation: str,
) -> None:
    arm = _complex_arm(tmp_path)
    path = arm.path / "client/choices.jsonl"
    rows = _rows(path)
    if mutation == "missing-field":
        del rows[0]["raw_logprobs"]
    else:
        rows[0]["unproven"] = "forged"
    _write_rows(path, rows)

    errors = _seal_invalid(arm)

    assert any("malformed semantic field set" in error for error in errors)


@pytest.mark.parametrize(
    ("field", "replacement", "expected_error"),
    [
        ("response_id", "forged-response", "response_id does not match"),
        (
            "router_attempt_id",
            "forged-router-attempt",
            "router_attempt_id does not match",
        ),
        ("status_code", 503, "terminal event type does not match"),
    ],
)
def test_response_identity_and_status_drive_derived_evidence(
    tmp_path: Path,
    field: str,
    replacement: object,
    expected_error: str,
) -> None:
    arm = _complex_arm(tmp_path)
    path = arm.path / "client/responses.jsonl"
    rows = _rows(path)
    rows[0][field] = replacement
    _write_rows(path, rows)

    errors = _seal_invalid(arm)

    assert any(expected_error in error for error in errors)


@pytest.mark.parametrize(
    ("field", "replacement", "expected_error"),
    [
        (
            "event_type",
            "CLIENT_ATTEMPT_HTTP_ERROR",
            "terminal event type does not match",
        ),
        (
            "router_attempt_id",
            "forged-router-attempt",
            "terminal router_attempt_id does not match",
        ),
    ],
)
def test_terminal_lineage_is_bound_to_response_outcome(
    tmp_path: Path,
    field: str,
    replacement: object,
    expected_error: str,
) -> None:
    arm = _complex_arm(tmp_path)
    path = arm.path / "lineage.jsonl"
    rows = _rows(path)
    rows[-1][field] = replacement
    _write_rows(path, rows)

    errors = _seal_invalid(arm)

    assert any(expected_error in error for error in errors)


def test_missing_expected_token_file_forces_invalid(tmp_path: Path) -> None:
    arm = _complex_arm(tmp_path)
    (arm.path / "client/tokens.jsonl").unlink()

    errors = _seal_invalid(arm)

    assert any("client/tokens.jsonl" in error for error in errors)


@pytest.mark.parametrize("mutation", ["missing", "extra", "reordered", "corrupted"])
def test_token_rows_must_exactly_match_rederived_order(
    tmp_path: Path,
    mutation: str,
) -> None:
    arm = _complex_arm(tmp_path)
    path = arm.path / "client/tokens.jsonl"
    rows = _rows(path)
    if mutation == "missing":
        rows.pop()
    elif mutation == "extra":
        rows.append(dict(rows[-1]))
    elif mutation == "reordered":
        rows[0], rows[1] = rows[1], rows[0]
    else:
        rows[0]["raw_token_record"] = {"token": "forged"}
    _write_rows(path, rows)

    errors = _seal_invalid(arm)

    assert any("token" in error for error in errors)


def test_missing_expected_unexpected_choice_file_forces_invalid(
    tmp_path: Path,
) -> None:
    arm = _complex_arm(tmp_path)
    (arm.path / "client/unexpected_choices.jsonl").unlink()

    errors = _seal_invalid(arm)

    assert any("client/unexpected_choices.jsonl" in error for error in errors)


@pytest.mark.parametrize("mutation", ["missing", "extra", "reordered", "corrupted"])
def test_unexpected_choice_rows_must_exactly_match_rederived_order(
    tmp_path: Path,
    mutation: str,
) -> None:
    arm = _complex_arm(tmp_path)
    path = arm.path / "client/unexpected_choices.jsonl"
    rows = _rows(path)
    if mutation == "missing":
        rows.pop()
    elif mutation == "extra":
        rows.append(dict(rows[-1]))
    elif mutation == "reordered":
        rows[0], rows[1] = rows[1], rows[0]
    else:
        rows[0]["raw_choice"] = {"index": 99}
    _write_rows(path, rows)

    errors = _seal_invalid(arm)

    assert any("unexpected-choice" in error for error in errors)


@pytest.mark.parametrize(
    ("relative", "record"),
    [
        (
            "client/tokens.jsonl",
            {
                "request_index": 0,
                "planned_choice_index": 0,
                "choice_occurrence_index": 0,
                "token_index": 0,
                "raw_token_record": {"token": "forged"},
            },
        ),
        (
            "client/unexpected_choices.jsonl",
            {
                "request_index": 0,
                "unexpected_index": 0,
                "raw_choice": {"index": 99},
            },
        ),
    ],
)
def test_optional_derived_file_may_not_exist_when_no_rows_are_expected(
    tmp_path: Path,
    relative: str,
    record: dict[str, object],
) -> None:
    arm = _simple_arm(tmp_path)
    assert not (arm.path / relative).exists()
    arm.append_event(relative, record)

    errors = _seal_invalid(arm)

    assert any("cardinality mismatch" in error for error in errors)


@pytest.mark.parametrize(
    "relative",
    ["client/tokens.jsonl", "client/unexpected_choices.jsonl"],
)
def test_empty_optional_derived_file_is_invalid_when_no_rows_are_expected(
    tmp_path: Path,
    relative: str,
) -> None:
    arm = _simple_arm(tmp_path)
    path = arm.path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()

    errors = _seal_invalid(arm)

    assert any("must be absent" in error and relative in error for error in errors)


@pytest.mark.parametrize(
    ("relative", "field"),
    [
        ("client/requests.jsonl", "response_id"),
        ("client/responses.jsonl", "router_attempt_id"),
        ("lineage.jsonl", "router_attempt_id"),
    ],
)
def test_source_rows_require_nullable_fields_to_be_present(
    tmp_path: Path,
    relative: str,
    field: str,
) -> None:
    arm = _complex_arm(tmp_path)
    path = arm.path / relative
    rows = _rows(path)
    del rows[-1][field]
    _write_rows(path, rows)

    errors = _seal_invalid(arm)

    assert any("invalid semantic schema" in error for error in errors)


@pytest.mark.parametrize(
    "relative",
    [
        "client/requests.jsonl",
        "client/responses.jsonl",
        "lineage.jsonl",
    ],
)
def test_source_rows_reject_extra_semantic_fields(
    tmp_path: Path,
    relative: str,
) -> None:
    arm = _complex_arm(tmp_path)
    path = arm.path / relative
    rows = _rows(path)
    rows[0]["unproven"] = "forged"
    _write_rows(path, rows)

    errors = _seal_invalid(arm)

    assert any("invalid semantic schema" in error for error in errors)


@pytest.mark.parametrize(
    "field",
    [
        "status_code",
        "raw_body_base64",
        "raw_body_sha256",
        "parsed_body",
        "parse_error",
        "response_id",
        "router_attempt_id",
    ],
)
def test_controlled_transport_response_requires_explicit_null_fields(
    tmp_path: Path,
    field: str,
) -> None:
    arm = _record(tmp_path, TransportError(TransportErrorKind.TIMEOUT))
    path = arm.path / "client/responses.jsonl"
    rows = _rows(path)
    del rows[0][field]
    _write_rows(path, rows)

    errors = _seal_invalid(arm)

    assert any("invalid semantic schema" in error for error in errors)


@pytest.mark.parametrize("field", ["transport_message", "transport_traceback"])
def test_controlled_transport_response_rejects_legacy_diagnostics(
    tmp_path: Path,
    field: str,
) -> None:
    arm = _record(tmp_path, TransportError(TransportErrorKind.URL_ERROR))
    path = arm.path / "client/responses.jsonl"
    rows = _rows(path)
    rows[0][field] = "caller-controlled secret"
    _write_rows(path, rows)

    errors = _seal_invalid(arm)

    assert any("invalid semantic schema" in error for error in errors)


@pytest.mark.parametrize(
    "relative",
    [
        "client/requests.jsonl",
        "client/responses.jsonl",
        "client/choices.jsonl",
        "client/tokens.jsonl",
        "client/unexpected_choices.jsonl",
        "lineage.jsonl",
    ],
)
def test_every_recorder_row_requires_the_complete_event_envelope(
    tmp_path: Path,
    relative: str,
) -> None:
    arm = _complex_arm(tmp_path)
    path = arm.path / relative
    rows = _rows(path)
    del rows[0]["recorded_at_utc"]
    _write_rows_exact(path, rows)

    errors = _seal_invalid(arm)

    assert any("invalid event UTC timestamp" in error for error in errors)


@pytest.mark.parametrize(
    ("field", "replacement", "expected_error"),
    [
        ("sequence", True, "invalid event sequence"),
        ("recorded_at_utc", "", "invalid event UTC timestamp"),
        ("recorded_at_utc", "not-a-UTC-timestamp", "invalid event UTC timestamp"),
        ("recorded_monotonic_ns", True, "invalid event monotonic timestamp"),
        ("recorded_monotonic_ns", -1, "invalid event monotonic timestamp"),
    ],
)
def test_event_envelope_values_are_strictly_typed(
    tmp_path: Path,
    field: str,
    replacement: object,
    expected_error: str,
) -> None:
    arm = _complex_arm(tmp_path)
    path = arm.path / "client/requests.jsonl"
    rows = _rows(path)
    rows[0][field] = replacement
    _write_rows_exact(path, rows)

    errors = _seal_invalid(arm)

    assert any(expected_error in error for error in errors)


def test_archived_request_header_control_characters_force_invalid(
    tmp_path: Path,
) -> None:
    arm = _simple_arm(tmp_path)
    path = arm.path / "client/requests.jsonl"
    rows = _rows(path)
    headers = rows[0]["headers"]
    assert isinstance(headers, dict)
    headers["x-observer"] = "forged\r\nX-Injected: yes"
    _write_rows(path, rows)

    errors = _seal_invalid(arm)

    assert any("invalid archived headers" in error for error in errors)


def test_controlled_transport_outcome_can_seal_without_raw_body(
    tmp_path: Path,
) -> None:
    arm = _record(tmp_path, TransportError(TransportErrorKind.TIMEOUT))

    disposition = arm.seal(
        ArmDisposition.PASS,
        "controlled transport outcome is complete",
        make_read_only=False,
    )

    assert disposition is ArmDisposition.PASS


def test_transport_terminal_and_choices_are_rederived(tmp_path: Path) -> None:
    arm = _record(tmp_path, TransportError(TransportErrorKind.URL_ERROR))
    choice_path = arm.path / "client/choices.jsonl"
    choice_rows = _rows(choice_path)
    choice_rows[0]["outcome"] = "MISSING"
    _write_rows(choice_path, choice_rows)
    lineage_path = arm.path / "lineage.jsonl"
    lineage_rows = _rows(lineage_path)
    lineage_rows[-1]["event_type"] = "CLIENT_ATTEMPT_COMPLETED"
    _write_rows(lineage_path, lineage_rows)

    errors = _seal_invalid(arm)

    assert any("choice line 1 outcome" in error for error in errors)
    assert any("terminal event type does not match" in error for error in errors)


@pytest.mark.parametrize("header_name", ["bad header", "x-api-token"])
def test_router_attempt_header_must_be_safe_to_archive(
    tmp_path: Path,
    header_name: str,
) -> None:
    arm = ArtifactArm.create(tmp_path / "cert", certification_plan())

    with (
        arm.writer_session() as writer,
        pytest.raises(CertificationError, match="router attempt header"),
    ):
        CertificationRecorder(
            writer,
            "http://router/v1/completions",
            StaticTransport(HttpExchange(200, (), b"{}")),
            timeout_seconds=10.0,
            router_attempt_header=header_name,
        )


def test_reverse_parent_batch_cannot_define_its_own_canonical_order(
    tmp_path: Path,
) -> None:
    arm = ArtifactArm.create(
        tmp_path / "cert",
        certification_plan(planned_parents=2),
    )
    response = HttpExchange(
        status_code=200,
        header_items=(),
        body=canonical_json_bytes(
            {
                "id": "response",
                "choices": [{"index": 0}, {"index": 1}, {"index": 2}],
            }
        ),
    )
    with arm.writer_session() as writer:
        recorder = CertificationRecorder(
            writer,
            "http://router/v1/completions",
            StaticTransport(response),
            timeout_seconds=10.0,
        )
        recorder.record(_plan(request_index=1, seed=18))
        recorder.record(_plan(request_index=0, seed=17))

    errors = _seal_invalid(arm)

    assert "request rows are not in canonical request_index order" in errors
    assert "response rows are not in canonical request_index order" in errors
    assert "client lineage events are not in canonical request order" in errors
    assert any("row order" in error for error in errors)


def test_duplicate_router_attempt_headers_preserve_body_but_force_invalid(
    tmp_path: Path,
) -> None:
    body = canonical_json_bytes(
        {
            "id": "response",
            "choices": [{"index": 0}, {"index": 1}, {"index": 2}],
        }
    )
    arm = _record(
        tmp_path,
        HttpExchange(
            status_code=200,
            header_items=(
                ("X-Gemma4-Router-Attempt-Id", "router-a"),
                ("x-gemma4-router-attempt-id", "router-b"),
            ),
            body=body,
        ),
    )
    response = _rows(arm.path / "client/responses.jsonl")[0]

    assert response["router_attempt_id"] is None
    assert len(response["header_items"]) == 2
    assert (arm.path / "client/choices.jsonl").is_file()
    errors = _seal_invalid(arm)

    assert any(
        "ambiguous duplicate router-attempt headers" in error for error in errors
    )


def test_duplicate_non_router_headers_remain_ordered_and_certifiable(
    tmp_path: Path,
) -> None:
    arm = _record(
        tmp_path,
        HttpExchange(
            status_code=200,
            header_items=(
                ("Set-Cookie", "first=secret"),
                ("set-cookie", "second=secret"),
                ("X-Gemma4-Router-Attempt-Id", "router-7"),
            ),
            body=canonical_json_bytes(
                {
                    "id": "response",
                    "choices": [{"index": 0}, {"index": 1}, {"index": 2}],
                }
            ),
        ),
    )
    response = _rows(arm.path / "client/responses.jsonl")[0]

    assert response["header_items"] == [
        {"name": "set-cookie", "value": "<redacted>", "redacted": True},
        {"name": "set-cookie", "value": "<redacted>", "redacted": True},
        {
            "name": "x-gemma4-router-attempt-id",
            "value": "router-7",
            "redacted": False,
        },
    ]
    disposition = arm.seal(
        ArmDisposition.PASS,
        "duplicate non-router instances were preserved",
        make_read_only=False,
    )

    assert disposition is ArmDisposition.PASS


@pytest.mark.parametrize("status_code", [True, 99, 600])
def test_http_exchange_rejects_non_http_status_codes(status_code: int) -> None:
    with pytest.raises(CertificationError, match="100 to 599"):
        HttpExchange(status_code=status_code, header_items=(), body=b"{}")


@pytest.mark.parametrize(
    ("header_items", "body", "expected_error"),
    [
        ({}, b"{}", "header_items"),
        ((), "not-bytes", "body must be bytes"),
    ],
)
def test_http_exchange_rejects_malformed_transport_evidence(
    header_items: object,
    body: object,
    expected_error: str,
) -> None:
    with pytest.raises(CertificationError, match=expected_error):
        HttpExchange(  # type: ignore[arg-type]
            status_code=200,
            header_items=header_items,
            body=body,
        )


def test_seal_rejects_tampered_non_http_status_code(tmp_path: Path) -> None:
    arm = _simple_arm(tmp_path)
    path = arm.path / "client/responses.jsonl"
    rows = _rows(path)
    rows[0]["status_code"] = 99
    _write_rows(path, rows)

    errors = _seal_invalid(arm)

    assert any("invalid status_code" in error for error in errors)


@pytest.mark.parametrize(
    ("field", "replacement", "expected_error"),
    [
        (
            "endpoint",
            "http://user:password@router/v1/completions",
            "unsafe endpoint",
        ),
        ("endpoint", "http://router/v1/completions?token=secret", "unsafe endpoint"),
    ],
)
def test_seal_revalidates_archived_request_endpoint(
    tmp_path: Path,
    field: str,
    replacement: object,
    expected_error: str,
) -> None:
    arm = _simple_arm(tmp_path)
    path = arm.path / "client/requests.jsonl"
    rows = _rows(path)
    rows[0][field] = replacement
    _write_rows(path, rows)

    errors = _seal_invalid(arm)

    assert any(expected_error in error for error in errors)


def test_seal_revalidates_request_header_redaction_and_identity(
    tmp_path: Path,
) -> None:
    arm = _simple_arm(tmp_path)
    path = arm.path / "client/requests.jsonl"
    rows = _rows(path)
    headers = rows[0]["headers"]
    assert isinstance(headers, dict)
    headers["authorization"] = "Bearer plaintext-secret"
    headers["x-gemma4-cert-root-request-id"] = "forged-root"
    rows[0]["redacted_headers"] = []
    _write_rows(path, rows)

    errors = _seal_invalid(arm)

    assert any("incorrect redacted_headers provenance" in error for error in errors)
    assert any("exposes sensitive header" in error for error in errors)
    assert any("required header" in error for error in errors)
