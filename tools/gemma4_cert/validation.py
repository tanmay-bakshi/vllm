"""Cross-file semantic validation for certification recorder evidence."""

import base64
import binascii
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from tools.gemma4_cert.common import (
    CertificationError,
    canonical_json_bytes,
    is_sensitive_name,
    is_valid_http_header_name,
    sha256_bytes,
    validate_http_endpoint,
)

_RECORDER_PATHS = {
    "requests": "client/requests.jsonl",
    "responses": "client/responses.jsonl",
    "choices": "client/choices.jsonl",
    "lineage": "lineage.jsonl",
}
_OPTIONAL_RECORDER_PATHS = {
    "tokens": "client/tokens.jsonl",
    "unexpected_choices": "client/unexpected_choices.jsonl",
}
_EVENT_ENVELOPE_FIELDS = {
    "sequence",
    "recorded_at_utc",
    "recorded_monotonic_ns",
}
_SCHEMA_VERSION = 1
_BINDING_FIELDS = {
    "payload_order",
    "payload_label",
    "request_template_sha256",
    "seed",
}
_DERIVED_BASE_FIELDS = {
    "schema_version",
    "root_request_id",
    "client_attempt_id",
    "router_attempt_id",
    "response_id",
    "request_index",
    *_BINDING_FIELDS,
}
_REQUEST_FIELDS = {
    *_DERIVED_BASE_FIELDS,
    "router_attempt_header",
    "endpoint",
    "planned_choice_indices",
    "headers",
    "redacted_headers",
    "raw_body_base64",
    "raw_body_sha256",
    "parsed_body",
}
_HTTP_RESPONSE_FIELDS = {
    *_DERIVED_BASE_FIELDS,
    "router_attempt_header",
    "status_code",
    "header_items",
    "raw_body_base64",
    "raw_body_sha256",
    "parsed_body",
    "parse_error",
}
_TRANSPORT_RESPONSE_FIELDS = {
    *_HTTP_RESPONSE_FIELDS,
    "transport_error",
}
_CHOICE_FIELDS = {
    *_DERIVED_BASE_FIELDS,
    "planned_choice_index",
    "outcome",
    "occurrence_count",
    "raw_choices",
    "raw_logprobs",
}
_TOKEN_FIELDS = {
    *_DERIVED_BASE_FIELDS,
    "planned_choice_index",
    "choice_occurrence_index",
    "token_index",
    "raw_token_record",
}
_UNEXPECTED_CHOICE_FIELDS = {
    *_DERIVED_BASE_FIELDS,
    "unexpected_index",
    "raw_choice",
}
_LINEAGE_FIELDS = {
    "schema_version",
    "event_type",
    "root_request_id",
    "client_attempt_id",
    "router_attempt_id",
    "p_registration_id",
    "d_child_id",
    "physical_pull_id",
    "request_index",
    *_BINDING_FIELDS,
}
_TERMINAL_LINEAGE_EVENTS = {
    "CLIENT_ATTEMPT_COMPLETED",
    "CLIENT_ATTEMPT_HTTP_ERROR",
    "CLIENT_ATTEMPT_TRANSPORT_ERROR",
}
_GIT_COMMIT = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_UTC_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$")
_RUNTIME_EVIDENCE_FIELDS = {
    "effective_config",
    "kv_specs",
    "feature_counters",
    "semantic_invariants",
    "graph_shapes",
}


@dataclass(frozen=True, slots=True)
class ParentBinding:
    """Deterministic payload and seed assignment for one planned parent."""

    request_index: int
    payload_order: int
    payload_label: str
    request_template_sha256: str
    seed: int

    def to_mapping(self) -> dict[str, object]:
        """Return fields embedded in recorder and lineage rows."""
        return {
            "payload_order": self.payload_order,
            "payload_label": self.payload_label,
            "request_template_sha256": self.request_template_sha256,
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class DerivedChoice:
    """Choice evidence reconstructed from one archived response."""

    planned_choice_index: int
    outcome: str
    raw_choices: tuple[dict[str, object], ...]


@dataclass(frozen=True, slots=True)
class DerivedResponse:
    """Semantic response evidence reconstructed without trusting derived rows."""

    response_id: str | None
    router_attempt_id: str | None
    terminal_event_type: str
    choices: tuple[DerivedChoice, ...]
    unexpected_choices: tuple[object, ...]


def required_json_errors(
    arm_path: Path,
    required_artifacts: Sequence[str],
    protocol_id: str,
    run_id: str,
    arm_id: str,
) -> tuple[str, ...]:
    """Validate required JSON files and certification attestation schemas."""
    errors = []
    for relative in required_artifacts:
        if not relative.endswith(".json"):
            continue
        path = arm_path / relative
        if path.is_symlink() or not path.is_file():
            continue
        try:
            parsed = json.loads(path.read_bytes(), parse_constant=_reject_json_constant)
        except (json.JSONDecodeError, UnicodeError, ValueError):
            errors.append(f"required JSON is invalid: {relative}")
            continue
        if relative.startswith("attest/"):
            errors.extend(
                _attestation_errors(
                    parsed,
                    relative,
                    protocol_id,
                    run_id,
                    arm_id,
                )
            )
    return tuple(errors)


def recorder_completeness_errors(
    arm_path: Path,
    parent_bindings: Sequence[ParentBinding],
    choices_per_parent: int,
) -> tuple[str, ...]:
    """Validate recorder cardinality and re-derive all response-dependent rows."""
    errors: list[str] = []
    planned_parents = len(parent_bindings)
    bindings_by_index = {binding.request_index: binding for binding in parent_bindings}
    records = {
        name: _read_records(arm_path / relative, relative, errors)
        for name, relative in _RECORDER_PATHS.items()
    }
    requests = records["requests"]
    responses = records["responses"]
    choices = records["choices"]
    lineage = records["lineage"]
    expected_indices = set(range(planned_parents))

    if len(requests) != planned_parents:
        errors.append(
            "recorder request cardinality mismatch: "
            f"expected {planned_parents}, observed {len(requests)}"
        )
    request_by_index: dict[int, dict[str, object]] = {}
    roots: set[str] = set()
    attempts: set[str] = set()
    request_order: list[int] = []
    for line_number, request in enumerate(requests, start=1):
        _validate_semantic_schema(
            request,
            _REQUEST_FIELDS,
            "request",
            line_number,
            errors,
        )
        index = _integer_field(request, "request_index", "request", line_number, errors)
        root = _string_field(request, "root_request_id", "request", line_number, errors)
        attempt = _string_field(
            request, "client_attempt_id", "request", line_number, errors
        )
        planned_choices = request.get("planned_choice_indices")
        if not _json_values_equal(
            planned_choices,
            list(range(choices_per_parent)),
        ):
            errors.append(
                f"request line {line_number} has noncanonical planned choice indices"
            )
        if index is None or root is None or attempt is None:
            continue
        request_order.append(index)
        if index not in expected_indices:
            errors.append(
                f"request line {line_number} has unexpected request_index {index}"
            )
            continue
        _validate_binding(
            request,
            bindings_by_index[index],
            "request",
            line_number,
            errors,
        )
        _validate_request_body(request, bindings_by_index[index], line_number, errors)
        _validate_request_metadata(request, line_number, errors)
        router_attempt_header = request.get("router_attempt_header")
        if (
            not isinstance(router_attempt_header, str)
            or len(router_attempt_header) == 0
            or router_attempt_header != router_attempt_header.lower()
            or not is_valid_http_header_name(router_attempt_header)
            or is_sensitive_name(router_attempt_header)
        ):
            errors.append(
                f"request line {line_number} has invalid router_attempt_header"
            )
        if request.get("router_attempt_id") is not None:
            errors.append(f"request line {line_number} router_attempt_id must be null")
        if request.get("response_id") is not None:
            errors.append(f"request line {line_number} response_id must be null")
        if index in request_by_index:
            errors.append(f"duplicate request row for request_index {index}")
        else:
            request_by_index[index] = request
        if root in roots:
            errors.append(f"duplicate root_request_id {root!r}")
        roots.add(root)
        if attempt in attempts:
            errors.append(f"duplicate client_attempt_id {attempt!r}")
        attempts.add(attempt)
    for missing_index in sorted(expected_indices - set(request_by_index)):
        errors.append(f"missing request row for request_index {missing_index}")
    if request_order != list(range(planned_parents)):
        errors.append("request rows are not in canonical request_index order")

    if len(responses) != planned_parents:
        errors.append(
            "recorder response cardinality mismatch: "
            f"expected {planned_parents}, observed {len(responses)}"
        )
    response_by_index: dict[int, dict[str, object]] = {}
    derived_by_index: dict[int, DerivedResponse] = {}
    response_order: list[int] = []
    for line_number, response in enumerate(responses, start=1):
        response_fields = (
            _TRANSPORT_RESPONSE_FIELDS
            if response.get("status_code") is None
            else _HTTP_RESPONSE_FIELDS
        )
        _validate_semantic_schema(
            response,
            response_fields,
            "response",
            line_number,
            errors,
        )
        index = _integer_field(
            response, "request_index", "response", line_number, errors
        )
        if index is None:
            continue
        if index not in expected_indices:
            errors.append(
                f"response line {line_number} has unexpected request_index {index}"
            )
            continue
        duplicate = index in response_by_index
        if duplicate:
            errors.append(f"duplicate response row for request_index {index}")
        else:
            response_by_index[index] = response
            response_order.append(index)
        _validate_identity(
            response, request_by_index.get(index), "response", line_number, errors
        )
        _validate_binding(
            response,
            bindings_by_index[index],
            "response",
            line_number,
            errors,
        )
        derived = _derive_response(
            response,
            request_by_index.get(index),
            choices_per_parent,
            line_number,
            errors,
        )
        if not duplicate and derived is not None:
            derived_by_index[index] = derived
    for missing_index in sorted(expected_indices - set(response_by_index)):
        errors.append(f"missing response row for request_index {missing_index}")
    if response_order != list(range(planned_parents)):
        errors.append("response rows are not in canonical request_index order")

    expected_choice_keys = {
        (request_index, choice_index)
        for request_index in range(planned_parents)
        for choice_index in range(choices_per_parent)
    }
    if len(choices) != len(expected_choice_keys):
        errors.append(
            "recorder choice cardinality mismatch: "
            f"expected {len(expected_choice_keys)}, observed {len(choices)}"
        )
    observed_choice_keys: set[tuple[int, int]] = set()
    for line_number, choice in enumerate(choices, start=1):
        request_index = _integer_field(
            choice, "request_index", "choice", line_number, errors
        )
        choice_index = _integer_field(
            choice, "planned_choice_index", "choice", line_number, errors
        )
        if request_index is None or choice_index is None:
            continue
        key = (request_index, choice_index)
        if key not in expected_choice_keys:
            errors.append(f"choice line {line_number} has unexpected key {key}")
            continue
        if key in observed_choice_keys:
            errors.append(f"duplicate choice row for key {key}")
        observed_choice_keys.add(key)
        selected_request = request_by_index.get(request_index)
        _validate_identity(choice, selected_request, "choice", line_number, errors)
        _validate_binding(
            choice,
            bindings_by_index[request_index],
            "choice",
            line_number,
            errors,
        )
        selected_response = response_by_index.get(request_index)
        if selected_response is not None and not _json_values_equal(
            choice.get("response_id"),
            selected_response.get("response_id"),
        ):
            errors.append(
                f"choice line {line_number} response_id does not match its response"
            )
    for missing_key in sorted(expected_choice_keys - observed_choice_keys):
        errors.append(f"missing choice row for key {missing_key}")

    expected_choice_rows: list[dict[str, object]] = []
    expected_token_rows: list[dict[str, object]] = []
    expected_unexpected_rows: list[dict[str, object]] = []
    for request_index in range(planned_parents):
        selected_request_row = request_by_index.get(request_index)
        derived = derived_by_index.get(request_index)
        if selected_request_row is None or derived is None:
            continue
        base = _derived_row_base(
            selected_request_row,
            bindings_by_index[request_index],
            derived,
        )
        for derived_choice in derived.choices:
            raw_choices = list(derived_choice.raw_choices)
            expected_choice_rows.append(
                {
                    **base,
                    "planned_choice_index": derived_choice.planned_choice_index,
                    "outcome": derived_choice.outcome,
                    "occurrence_count": len(raw_choices),
                    "raw_choices": raw_choices,
                    "raw_logprobs": [
                        raw_choice.get("logprobs") for raw_choice in raw_choices
                    ],
                }
            )
            for occurrence_index, raw_choice in enumerate(raw_choices):
                for token_index, token in enumerate(token_entries(raw_choice)):
                    expected_token_rows.append(
                        {
                            **base,
                            "planned_choice_index": (
                                derived_choice.planned_choice_index
                            ),
                            "choice_occurrence_index": occurrence_index,
                            "token_index": token_index,
                            "raw_token_record": token,
                        }
                    )
        for unexpected_index, unexpected_choice in enumerate(
            derived.unexpected_choices
        ):
            expected_unexpected_rows.append(
                {
                    **base,
                    "unexpected_index": unexpected_index,
                    "raw_choice": unexpected_choice,
                }
            )

    _validate_derived_rows(
        choices,
        expected_choice_rows,
        _CHOICE_FIELDS,
        "choice",
        errors,
    )
    tokens = _read_optional_records(
        arm_path / _OPTIONAL_RECORDER_PATHS["tokens"],
        _OPTIONAL_RECORDER_PATHS["tokens"],
        errors,
    )
    _validate_optional_derived_rows(
        tokens,
        expected_token_rows,
        _TOKEN_FIELDS,
        "token",
        _OPTIONAL_RECORDER_PATHS["tokens"],
        errors,
    )
    unexpected_choices = _read_optional_records(
        arm_path / _OPTIONAL_RECORDER_PATHS["unexpected_choices"],
        _OPTIONAL_RECORDER_PATHS["unexpected_choices"],
        errors,
    )
    _validate_optional_derived_rows(
        unexpected_choices,
        expected_unexpected_rows,
        _UNEXPECTED_CHOICE_FIELDS,
        "unexpected-choice",
        _OPTIONAL_RECORDER_PATHS["unexpected_choices"],
        errors,
    )

    started_counts = {index: 0 for index in expected_indices}
    terminal_counts = {index: 0 for index in expected_indices}
    start_lines: dict[int, list[int]] = {index: [] for index in expected_indices}
    terminal_events: dict[int, list[tuple[int, dict[str, object]]]] = {
        index: [] for index in expected_indices
    }
    client_event_order: list[tuple[int, str]] = []
    for line_number, event in enumerate(lineage, start=1):
        _validate_semantic_schema(
            event,
            _LINEAGE_FIELDS,
            "lineage",
            line_number,
            errors,
        )
        index = _integer_field(event, "request_index", "lineage", line_number, errors)
        event_type = _string_field(event, "event_type", "lineage", line_number, errors)
        if index is None or event_type is None:
            continue
        if index not in expected_indices:
            errors.append(
                f"lineage line {line_number} has unexpected request_index {index}"
            )
            continue
        _validate_identity(
            event, request_by_index.get(index), "lineage", line_number, errors
        )
        _validate_binding(
            event,
            bindings_by_index[index],
            "lineage",
            line_number,
            errors,
        )
        if event_type == "CLIENT_ATTEMPT_STARTED":
            client_event_order.append((index, event_type))
            started_counts[index] += 1
            start_lines[index].append(line_number)
            if event.get("router_attempt_id") is not None:
                errors.append(
                    f"lineage line {line_number} start router_attempt_id must be null"
                )
        if event_type in _TERMINAL_LINEAGE_EVENTS:
            client_event_order.append((index, event_type))
            terminal_counts[index] += 1
            terminal_events[index].append((line_number, event))
        if event_type == "CLIENT_ATTEMPT_STARTED" or event_type in (
            _TERMINAL_LINEAGE_EVENTS
        ):
            for field in (
                "p_registration_id",
                "d_child_id",
                "physical_pull_id",
            ):
                if event.get(field) is not None:
                    errors.append(
                        f"lineage line {line_number} client event {field} must be null"
                    )
    for index in sorted(expected_indices):
        if started_counts[index] != 1:
            errors.append(
                f"request_index {index} has {started_counts[index]} "
                "start lineage events"
            )
        if terminal_counts[index] != 1:
            errors.append(
                f"request_index {index} has {terminal_counts[index]} "
                "terminal lineage events"
            )
        derived = derived_by_index.get(index)
        if derived is None:
            continue
        for line_number, terminal_event in terminal_events[index]:
            if terminal_event.get("event_type") != derived.terminal_event_type:
                errors.append(
                    f"lineage line {line_number} terminal event type does not "
                    "match its response outcome"
                )
            if terminal_event.get("router_attempt_id") != derived.router_attempt_id:
                errors.append(
                    f"lineage line {line_number} terminal router_attempt_id does "
                    "not match its response"
                )
        if (
            len(start_lines[index]) == 1
            and len(terminal_events[index]) == 1
            and start_lines[index][0] >= terminal_events[index][0][0]
        ):
            errors.append(
                f"request_index {index} terminal lineage event precedes its start"
            )
    if len(derived_by_index) == planned_parents:
        expected_client_event_order: list[tuple[int, str]] = []
        for index in range(planned_parents):
            expected_client_event_order.extend(
                (
                    (index, "CLIENT_ATTEMPT_STARTED"),
                    (index, derived_by_index[index].terminal_event_type),
                )
            )
        if client_event_order != expected_client_event_order:
            errors.append("client lineage events are not in canonical request order")
    return tuple(errors)


def _read_records(
    path: Path,
    relative: str,
    errors: list[str],
) -> list[dict[str, object]]:
    if path.is_symlink() or not path.is_file():
        errors.append(f"recorder artifact is missing: {relative}")
        return []
    records = []
    with path.open("rb") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                parsed = json.loads(line, parse_constant=_reject_json_constant)
            except (json.JSONDecodeError, UnicodeError, ValueError) as error:
                errors.append(
                    f"invalid recorder JSON at {relative}:{line_number}: {error}"
                )
                continue
            if not isinstance(parsed, dict):
                errors.append(
                    f"recorder row is not an object at {relative}:{line_number}"
                )
                continue
            _validate_event_envelope(parsed, relative, line_number, errors)
            records.append(parsed)
    return records


def _validate_event_envelope(
    record: dict[str, object],
    relative: str,
    line_number: int,
    errors: list[str],
) -> None:
    sequence = record.get("sequence")
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence <= 0:
        errors.append(f"invalid event sequence at {relative}:{line_number}")
    recorded_at_utc = record.get("recorded_at_utc")
    if not _is_utc_timestamp(recorded_at_utc):
        errors.append(f"invalid event UTC timestamp at {relative}:{line_number}")
    monotonic = record.get("recorded_monotonic_ns")
    if isinstance(monotonic, bool) or not isinstance(monotonic, int) or monotonic < 0:
        errors.append(f"invalid event monotonic timestamp at {relative}:{line_number}")


def _is_utc_timestamp(value: object) -> bool:
    if not isinstance(value, str) or _UTC_TIMESTAMP.fullmatch(value) is None:
        return False
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError:
        return False
    return True


def _validate_semantic_schema(
    record: dict[str, object],
    expected_fields: set[str],
    kind: str,
    line_number: int,
    errors: list[str],
) -> None:
    observed_fields = set(record) - _EVENT_ENVELOPE_FIELDS
    if observed_fields != expected_fields:
        errors.append(f"{kind} line {line_number} has an invalid semantic schema")
    if not _json_values_equal(record.get("schema_version"), _SCHEMA_VERSION):
        errors.append(f"{kind} line {line_number} has an invalid schema_version")


def _read_optional_records(
    path: Path,
    relative: str,
    errors: list[str],
) -> list[dict[str, object]] | None:
    if path.is_symlink():
        errors.append(f"optional recorder artifact is a symlink: {relative}")
        return []
    if not path.exists():
        return None
    if not path.is_file():
        errors.append(f"optional recorder artifact is not a file: {relative}")
        return []
    return _read_records(path, relative, errors)


def _validate_derived_rows(
    observed: list[dict[str, object]] | None,
    expected: list[dict[str, object]],
    expected_fields: set[str],
    kind: str,
    errors: list[str],
) -> None:
    if observed is None:
        return
    for line_number, record in enumerate(observed, start=1):
        _validate_semantic_schema(
            record,
            expected_fields,
            kind,
            line_number,
            errors,
        )
    if len(observed) != len(expected):
        errors.append(
            f"derived {kind} cardinality mismatch: "
            f"expected {len(expected)}, observed {len(observed)}"
        )
    for line_number, (actual, derived) in enumerate(
        zip(observed, expected, strict=False),
        start=1,
    ):
        semantic = {
            field: value
            for field, value in actual.items()
            if field not in _EVENT_ENVELOPE_FIELDS
        }
        if set(semantic) != set(derived):
            errors.append(
                f"{kind} line {line_number} has a malformed semantic field set"
            )
        for field in sorted(set(semantic) & set(derived)):
            if not _json_values_equal(semantic[field], derived[field]):
                errors.append(
                    f"{kind} line {line_number} {field} does not match its "
                    "archived response or row order"
                )


def _validate_optional_derived_rows(
    observed: list[dict[str, object]] | None,
    expected: list[dict[str, object]],
    expected_fields: set[str],
    kind: str,
    relative: str,
    errors: list[str],
) -> None:
    if len(expected) == 0:
        if observed is not None:
            errors.append(
                f"optional derived artifact must be absent when no rows are "
                f"expected: {relative}"
            )
            _validate_derived_rows(
                observed,
                expected,
                expected_fields,
                kind,
                errors,
            )
        return
    if observed is None:
        errors.append(
            f"derived recorder artifact is missing: {relative}; "
            f"expected {len(expected)} rows"
        )
        return
    _validate_derived_rows(
        observed,
        expected,
        expected_fields,
        kind,
        errors,
    )


def _validate_identity(
    record: dict[str, object],
    request: dict[str, object] | None,
    kind: str,
    line_number: int,
    errors: list[str],
) -> None:
    if request is None:
        return
    for field in ("root_request_id", "client_attempt_id"):
        if not _json_values_equal(record.get(field), request.get(field)):
            errors.append(
                f"{kind} line {line_number} {field} does not match its request"
            )


def _validate_binding(
    record: dict[str, object],
    binding: ParentBinding,
    kind: str,
    line_number: int,
    errors: list[str],
) -> None:
    for field, expected in binding.to_mapping().items():
        if not _json_values_equal(record.get(field), expected):
            errors.append(
                f"{kind} line {line_number} {field} does not match its plan binding"
            )


def _validate_request_body(
    record: dict[str, object],
    binding: ParentBinding,
    line_number: int,
    errors: list[str],
) -> None:
    raw = _decode_raw_body(record, "request", line_number, errors)
    parsed = record.get("parsed_body")
    if raw is None or not isinstance(parsed, dict):
        if not isinstance(parsed, dict):
            errors.append(f"request line {line_number} has invalid parsed_body")
        return
    try:
        canonical = canonical_json_bytes(parsed)
    except (TypeError, UnicodeError, ValueError):
        errors.append(f"request line {line_number} parsed_body is not canonical JSON")
        return
    if raw != canonical:
        errors.append(f"request line {line_number} raw body does not match parsed_body")
    if not _json_values_equal(parsed.get("seed"), binding.seed):
        errors.append(
            f"request line {line_number} seed does not match its plan binding"
        )
        return
    template = dict(parsed)
    del template["seed"]
    digest = sha256_bytes(canonical_json_bytes(template))
    if digest != binding.request_template_sha256:
        errors.append(
            f"request line {line_number} template digest does not match "
            "its plan binding"
        )


def _validate_request_metadata(
    record: dict[str, object],
    line_number: int,
    errors: list[str],
) -> None:
    endpoint = record.get("endpoint")
    if not isinstance(endpoint, str):
        errors.append(f"request line {line_number} has invalid endpoint")
    else:
        try:
            validate_http_endpoint(endpoint)
        except CertificationError as error:
            errors.append(f"request line {line_number} has unsafe endpoint: {error}")
    headers_value = record.get("headers")
    if not isinstance(headers_value, dict) or not all(
        isinstance(name, str)
        and name == name.lower()
        and is_valid_http_header_name(name)
        and isinstance(value, str)
        and "\r" not in value
        and "\n" not in value
        for name, value in headers_value.items()
    ):
        errors.append(f"request line {line_number} has invalid archived headers")
        return
    expected_redacted_headers = sorted(
        name for name in headers_value if is_sensitive_name(name)
    )
    if record.get("redacted_headers") != expected_redacted_headers:
        errors.append(
            f"request line {line_number} has incorrect redacted_headers provenance"
        )
    for name in expected_redacted_headers:
        if headers_value[name] != "<redacted>":
            errors.append(
                f"request line {line_number} exposes sensitive header {name!r}"
            )
    required_headers = {
        "content-type": "application/json",
        "x-gemma4-cert-root-request-id": record.get("root_request_id"),
        "x-gemma4-cert-client-attempt-id": record.get("client_attempt_id"),
    }
    for name, expected in required_headers.items():
        if not _json_values_equal(headers_value.get(name), expected):
            errors.append(
                f"request line {line_number} required header {name!r} does not "
                "match its request"
            )


def _decode_archived_header_items(
    value: object,
    line_number: int,
    errors: list[str],
) -> list[tuple[str, str, bool]]:
    if not isinstance(value, list):
        errors.append(f"response line {line_number} has invalid header_items")
        return []
    decoded = []
    for item_index, item in enumerate(value):
        if not isinstance(item, dict) or set(item) != {
            "name",
            "value",
            "redacted",
        }:
            errors.append(
                f"response line {line_number} header item {item_index} has an "
                "invalid schema"
            )
            continue
        name = item.get("name")
        header_value = item.get("value")
        redacted = item.get("redacted")
        if (
            not isinstance(name, str)
            or name != name.lower()
            or not is_valid_http_header_name(name)
            or not isinstance(header_value, str)
            or (
                isinstance(header_value, str)
                and ("\r" in header_value or "\n" in header_value)
            )
            or not isinstance(redacted, bool)
        ):
            errors.append(
                f"response line {line_number} header item {item_index} is invalid"
            )
            continue
        expected_redaction = is_sensitive_name(name)
        if redacted != expected_redaction:
            errors.append(
                f"response line {line_number} header item {item_index} has "
                "incorrect redaction provenance"
            )
        if expected_redaction and header_value != "<redacted>":
            errors.append(
                f"response line {line_number} header item {item_index} exposes "
                "a sensitive value"
            )
        decoded.append((name, header_value, redacted))
    return decoded


def _derive_response(
    record: dict[str, object],
    request: dict[str, object] | None,
    choices_per_parent: int,
    line_number: int,
    errors: list[str],
) -> DerivedResponse | None:
    configured_header = (
        request.get("router_attempt_header") if request is not None else None
    )
    archived_header = record.get("router_attempt_header")
    if (
        not isinstance(archived_header, str)
        or len(archived_header) == 0
        or archived_header != archived_header.lower()
        or not is_valid_http_header_name(archived_header)
        or is_sensitive_name(archived_header)
    ):
        errors.append(f"response line {line_number} has invalid router_attempt_header")
        return None
    if configured_header != archived_header:
        errors.append(
            f"response line {line_number} router_attempt_header does not match "
            "its request"
        )
    header_items = _decode_archived_header_items(
        record.get("header_items"),
        line_number,
        errors,
    )

    status_code = record.get("status_code")
    if status_code is None:
        transport_error = record.get("transport_error")
        if (
            not isinstance(transport_error, dict)
            or set(transport_error) != {"kind"}
            or transport_error.get("kind") not in {"timeout", "url_error"}
        ):
            errors.append(f"response line {line_number} has invalid transport_error")
        for field in (
            "raw_body_base64",
            "raw_body_sha256",
            "parsed_body",
            "parse_error",
            "response_id",
            "router_attempt_id",
        ):
            if record.get(field) is not None:
                errors.append(
                    f"response line {line_number} transport outcome has non-null "
                    f"{field}"
                )
        if len(header_items) > 0:
            errors.append(
                f"response line {line_number} transport outcome has response headers"
            )
        return DerivedResponse(
            response_id=None,
            router_attempt_id=None,
            terminal_event_type="CLIENT_ATTEMPT_TRANSPORT_ERROR",
            choices=tuple(
                DerivedChoice(index, "REQUEST_ERROR", ())
                for index in range(choices_per_parent)
            ),
            unexpected_choices=(),
        )
    if (
        isinstance(status_code, bool)
        or not isinstance(status_code, int)
        or status_code < 100
        or status_code > 599
    ):
        errors.append(f"response line {line_number} has invalid status_code")
        return None
    if "transport_error" in record:
        errors.append(f"response line {line_number} HTTP outcome has transport_error")
    raw = _decode_raw_body(record, "response", line_number, errors)
    if raw is None:
        return None
    parsed_from_raw, parse_error = _parse_response_object(raw)
    parsed = record.get("parsed_body")
    if not _json_values_equal(parsed, parsed_from_raw):
        if parsed_from_raw is None:
            errors.append(
                f"response line {line_number} parsed_body contradicts invalid raw JSON"
            )
        else:
            errors.append(
                f"response line {line_number} raw body does not match parsed_body"
            )
    if not _json_values_equal(record.get("parse_error"), parse_error):
        errors.append(
            f"response line {line_number} parse_error does not match its raw body"
        )
    response_id_value = (
        parsed_from_raw.get("id") if parsed_from_raw is not None else None
    )
    response_id = response_id_value if isinstance(response_id_value, str) else None
    if not _json_values_equal(record.get("response_id"), response_id):
        errors.append(
            f"response line {line_number} response_id does not match its raw body"
        )
    router_attempt_values = [
        value for name, value, _ in header_items if name == archived_header
    ]
    if len(router_attempt_values) > 1:
        errors.append(
            f"response line {line_number} has ambiguous duplicate "
            "router-attempt headers"
        )
    router_attempt_id = (
        router_attempt_values[0] if len(router_attempt_values) == 1 else None
    )
    if not _json_values_equal(record.get("router_attempt_id"), router_attempt_id):
        errors.append(
            f"response line {line_number} router_attempt_id does not match its "
            "archived response headers"
        )
    terminal_event_type = "CLIENT_ATTEMPT_COMPLETED"
    if status_code < 200 or status_code >= 300:
        terminal_event_type = "CLIENT_ATTEMPT_HTTP_ERROR"
    derived_choices, unexpected_choices = _derive_choices(
        status_code,
        parsed_from_raw,
        choices_per_parent,
    )
    return DerivedResponse(
        response_id=response_id,
        router_attempt_id=router_attempt_id,
        terminal_event_type=terminal_event_type,
        choices=derived_choices,
        unexpected_choices=unexpected_choices,
    )


def _parse_response_object(
    raw: bytes,
) -> tuple[dict[str, object] | None, str | None]:
    try:
        parsed = json.loads(raw, parse_constant=_reject_json_constant)
        if not isinstance(parsed, dict):
            return None, "response JSON is not an object"
        canonical_json_bytes(parsed)
    except (json.JSONDecodeError, UnicodeError, TypeError, ValueError) as error:
        return None, f"{type(error).__name__}: {error}"
    return parsed, None


def _derive_choices(
    status_code: int,
    parsed: dict[str, object] | None,
    choices_per_parent: int,
) -> tuple[tuple[DerivedChoice, ...], tuple[object, ...]]:
    planned_indices = tuple(range(choices_per_parent))
    occurrences: dict[int, list[dict[str, object]]] = {}
    unexpected: list[object] = []
    raw_choices = parsed.get("choices") if parsed is not None else None
    if isinstance(raw_choices, list):
        for raw_choice in raw_choices:
            if not isinstance(raw_choice, dict):
                unexpected.append(raw_choice)
                continue
            index = raw_choice.get("index")
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or index not in planned_indices
            ):
                unexpected.append(raw_choice)
                continue
            occurrences.setdefault(index, []).append(raw_choice)
    if status_code < 200 or status_code >= 300:
        outcome_for_missing = "HTTP_ERROR"
    elif parsed is None:
        outcome_for_missing = "INVALID_RESPONSE"
    else:
        outcome_for_missing = "MISSING"
    choices = []
    for choice_index in planned_indices:
        matches = occurrences.get(choice_index, [])
        if status_code < 200 or status_code >= 300:
            outcome = "HTTP_ERROR"
        elif len(matches) == 0:
            outcome = outcome_for_missing
        elif len(matches) == 1:
            outcome = "OBSERVED"
        else:
            outcome = "DUPLICATE"
        choices.append(
            DerivedChoice(
                planned_choice_index=choice_index,
                outcome=outcome,
                raw_choices=tuple(matches),
            )
        )
    return tuple(choices), tuple(unexpected)


def _derived_row_base(
    request: dict[str, object],
    binding: ParentBinding,
    response: DerivedResponse,
) -> dict[str, object]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "root_request_id": request.get("root_request_id"),
        "client_attempt_id": request.get("client_attempt_id"),
        "router_attempt_id": response.router_attempt_id,
        "response_id": response.response_id,
        "request_index": binding.request_index,
        **binding.to_mapping(),
    }


def token_entries(choice: Mapping[str, object]) -> list[object]:
    """Derive canonical per-token evidence from one raw response choice."""
    logprobs = choice.get("logprobs")
    if not isinstance(logprobs, dict):
        return []
    content = logprobs.get("content")
    if isinstance(content, list):
        return list(content)
    tokens = logprobs.get("tokens")
    if not isinstance(tokens, list):
        return []
    token_logprobs = _list_or_empty(logprobs.get("token_logprobs"))
    top_logprobs = _list_or_empty(logprobs.get("top_logprobs"))
    text_offset = _list_or_empty(logprobs.get("text_offset"))
    records: list[object] = []
    for index, token in enumerate(tokens):
        records.append(
            {
                "token": token,
                "logprob": _item_or_none(token_logprobs, index),
                "top_logprobs": _item_or_none(top_logprobs, index),
                "text_offset": _item_or_none(text_offset, index),
            }
        )
    return records


def _list_or_empty(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _item_or_none(values: Sequence[object], index: int) -> object:
    return values[index] if index < len(values) else None


def _json_values_equal(left: object, right: object) -> bool:
    try:
        return canonical_json_bytes(left) == canonical_json_bytes(right)
    except (TypeError, UnicodeError, ValueError):
        return False


def _decode_raw_body(
    record: dict[str, object],
    kind: str,
    line_number: int,
    errors: list[str],
) -> bytes | None:
    encoded = record.get("raw_body_base64")
    digest = record.get("raw_body_sha256")
    if not isinstance(encoded, str):
        errors.append(f"{kind} line {line_number} has invalid raw_body_base64")
        return None
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        errors.append(f"{kind} line {line_number} has invalid raw body encoding")
        return None
    if not isinstance(digest, str) or sha256_bytes(raw) != digest:
        errors.append(f"{kind} line {line_number} raw body SHA-256 is incorrect")
    return raw


def _attestation_errors(
    value: object,
    relative: str,
    protocol_id: str,
    run_id: str,
    arm_id: str,
) -> list[str]:
    prefix = f"required attestation {relative}"
    if not isinstance(value, dict):
        return [f"{prefix} is not an object"]
    errors = []
    if value.get("schema_version") != 1:
        errors.append(f"{prefix} has an invalid schema version")
    context = value.get("context")
    if not isinstance(context, dict):
        errors.append(f"{prefix} has no context object")
    else:
        expected_identity = {
            "protocol_id": protocol_id,
            "run_id": run_id,
            "arm_id": arm_id,
        }
        for field, expected in expected_identity.items():
            if context.get(field) != expected:
                errors.append(f"{prefix} context {field} does not match its arm")
        if not isinstance(context.get("role"), str) or len(context["role"]) == 0:
            errors.append(f"{prefix} context has no role")
        if (
            not isinstance(context.get("engine_boot_uuid"), str)
            or len(context["engine_boot_uuid"]) == 0
        ):
            errors.append(f"{prefix} context has no engine boot UUID")
    process = value.get("process")
    if not isinstance(process, dict):
        errors.append(f"{prefix} has no process object")
    else:
        pid = process.get("pid")
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
            errors.append(f"{prefix} has an invalid process PID")
        if not isinstance(process.get("executable"), str):
            errors.append(f"{prefix} has no process executable")
    repository = value.get("repository")
    if not isinstance(repository, dict):
        errors.append(f"{prefix} has no repository object")
    else:
        commit = repository.get("commit")
        if not isinstance(commit, str) or _GIT_COMMIT.fullmatch(commit) is None:
            errors.append(f"{prefix} has an invalid repository commit")
        for field in ("dirty_diff_sha256", "tracked_diff_sha256"):
            digest = repository.get(field)
            if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
                errors.append(f"{prefix} has an invalid repository {field}")
    for field in ("loaded_modules", "native_libraries", "named_artifacts"):
        if not isinstance(value.get(field), list):
            errors.append(f"{prefix} has no {field} array")
    software_versions = value.get("software_versions")
    if not isinstance(software_versions, dict) or not all(
        isinstance(software_versions.get(source), dict)
        for source in ("observed", "caller_supplied")
    ):
        errors.append(f"{prefix} has invalid software-version provenance")
    runtime_evidence = value.get("runtime_evidence")
    if not isinstance(runtime_evidence, dict):
        errors.append(f"{prefix} has no runtime-evidence provenance")
    else:
        for source in ("observed", "caller_supplied"):
            evidence = runtime_evidence.get(source)
            if not isinstance(evidence, dict):
                errors.append(f"{prefix} has no {source} runtime evidence")
                continue
            if set(evidence) != _RUNTIME_EVIDENCE_FIELDS:
                errors.append(f"{prefix} has incomplete {source} runtime evidence")
                continue
            for field, item in evidence.items():
                if not isinstance(item, dict) or item.get("availability") not in {
                    "available",
                    "unavailable",
                }:
                    errors.append(f"{prefix} has invalid {source} {field} provenance")
    return errors


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant: {value}")


def _integer_field(
    record: dict[str, object],
    field: str,
    kind: str,
    line_number: int,
    errors: list[str],
) -> int | None:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        errors.append(f"{kind} line {line_number} has invalid {field}")
        return None
    return value


def _string_field(
    record: dict[str, object],
    field: str,
    kind: str,
    line_number: int,
    errors: list[str],
) -> str | None:
    value = record.get(field)
    if not isinstance(value, str) or len(value) == 0:
        errors.append(f"{kind} line {line_number} has invalid {field}")
        return None
    return value
