"""Lossless request, response, choice, token, and lineage recording."""

import base64
import json
import urllib.error
import urllib.request
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from email.message import Message
from enum import StrEnum
from pathlib import Path
from typing import IO, Protocol

from tools.gemma4_cert.artifact import ArtifactWriter
from tools.gemma4_cert.common import (
    CertificationError,
    canonical_json_bytes,
    is_sensitive_name,
    is_valid_http_header_name,
    sha256_bytes,
    validate_http_endpoint,
)
from tools.gemma4_cert.validation import ParentBinding, token_entries


@dataclass(frozen=True, slots=True)
class RequestPlan:
    """One predeclared parent request and its exact expected choice indices."""

    root_request_id: str
    request_index: int
    body: Mapping[str, object]
    planned_choice_indices: tuple[int, ...]
    headers: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.root_request_id) == 0:
            raise CertificationError("root request ID cannot be empty")
        if self.request_index < 0:
            raise CertificationError("request index must be non-negative")
        if self.planned_choice_indices != tuple(
            range(len(self.planned_choice_indices))
        ):
            raise CertificationError(
                "planned choice indices must be contiguous from zero"
            )
        if len(self.planned_choice_indices) == 0:
            raise CertificationError("at least one choice must be planned")
        requested_count = self.body.get("n")
        if isinstance(requested_count, bool) or not isinstance(requested_count, int):
            raise CertificationError("request body must declare integer n")
        if requested_count != len(self.planned_choice_indices):
            raise CertificationError("request n does not match planned choice indices")
        seed = self.body.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise CertificationError("request body must declare integer seed")
        canonical_json_bytes(dict(self.body))

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "RequestPlan":
        """Parse a recorder request-plan row."""
        root_request_id = value.get("root_request_id")
        request_index = value.get("request_index")
        body = value.get("body")
        planned = value.get("planned_choice_indices")
        headers = value.get("headers", {})
        if not isinstance(root_request_id, str):
            raise CertificationError("root_request_id must be a string")
        if isinstance(request_index, bool) or not isinstance(request_index, int):
            raise CertificationError("request_index must be an integer")
        if not isinstance(body, dict):
            raise CertificationError("body must be an object")
        if not isinstance(planned, list):
            raise CertificationError("planned_choice_indices must be an array")
        if not isinstance(headers, dict) or not all(
            isinstance(key, str) and isinstance(item, str)
            for key, item in headers.items()
        ):
            raise CertificationError("headers must map strings to strings")
        indices = []
        for index in planned:
            if isinstance(index, bool) or not isinstance(index, int):
                raise CertificationError("planned choice indices must be integers")
            indices.append(index)
        return cls(
            root_request_id=root_request_id,
            request_index=request_index,
            body=body,
            planned_choice_indices=tuple(indices),
            headers=headers,
        )


@dataclass(frozen=True, slots=True)
class HttpExchange:
    """Raw HTTP response returned by a recorder transport."""

    status_code: int
    header_items: tuple[tuple[str, str], ...]
    body: bytes

    def __post_init__(self) -> None:
        _validate_http_status(self.status_code)
        if not isinstance(self.body, bytes):
            raise CertificationError("response body must be bytes")
        if not isinstance(self.header_items, tuple) or not all(
            isinstance(item, tuple)
            and len(item) == 2
            and isinstance(item[0], str)
            and isinstance(item[1], str)
            for item in self.header_items
        ):
            raise CertificationError(
                "response header_items must be a tuple of string pairs"
            )


class TransportErrorKind(StrEnum):
    """Controlled failure classes safe to retain in certification evidence."""

    TIMEOUT = "timeout"
    URL_ERROR = "url_error"


class TransportError(RuntimeError):
    """A classified request failure with no caller-controlled diagnostic text."""

    def __init__(self, kind: TransportErrorKind) -> None:
        if not isinstance(kind, TransportErrorKind):
            raise CertificationError("transport error has an invalid classification")
        super().__init__(kind.value)
        self.kind = kind


class Transport(Protocol):
    """Synchronous raw HTTP transport used by the certification recorder."""

    def send(
        self,
        endpoint: str,
        body: bytes,
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> HttpExchange:
        """Send one request and return its response without interpreting the body."""
        ...


class UrllibTransport:
    """Standard-library HTTP transport preserving raw response bytes."""

    def __init__(self, opener: urllib.request.OpenerDirector | None = None) -> None:
        self._opener = opener or urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            _NoRedirectHandler(),
        )

    def send(
        self,
        endpoint: str,
        body: bytes,
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> HttpExchange:
        """Send one POST request, retaining non-2xx responses as evidence."""
        request = urllib.request.Request(
            endpoint,
            data=body,
            headers=dict(headers),
            method="POST",
        )
        try:
            with self._opener.open(request, timeout=timeout_seconds) as response:
                return HttpExchange(
                    status_code=response.status,
                    header_items=tuple(response.headers.raw_items()),
                    body=response.read(),
                )
        except urllib.error.HTTPError as error:
            try:
                return HttpExchange(
                    status_code=error.code,
                    header_items=tuple(error.headers.raw_items()),
                    body=error.read(),
                )
            finally:
                error.close()
        except urllib.error.URLError as error:
            raise TransportError(TransportErrorKind.URL_ERROR) from error
        except TimeoutError as error:
            raise TransportError(TransportErrorKind.TIMEOUT) from error


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        request: urllib.request.Request,
        fp: IO[bytes],
        code: int,
        message: str,
        headers: Message,
        new_url: str,
    ) -> None:
        return None


class CertificationRecorder:
    """Record certification requests without denominator or lineage loss."""

    def __init__(
        self,
        arm: ArtifactWriter,
        endpoint: str,
        transport: Transport,
        *,
        timeout_seconds: float,
        router_attempt_header: str = "x-gemma4-router-attempt-id",
    ) -> None:
        if timeout_seconds <= 0:
            raise CertificationError("timeout must be positive")
        validate_http_endpoint(endpoint)
        if not isinstance(arm, ArtifactWriter):
            raise CertificationError(
                "certification recorder requires an active artifact writer session"
            )
        if not is_valid_http_header_name(router_attempt_header):
            raise CertificationError(
                "router attempt header is not a valid HTTP field name"
            )
        normalized_router_attempt_header = router_attempt_header.lower()
        if is_sensitive_name(normalized_router_attempt_header):
            raise CertificationError(
                "router attempt header cannot have a sensitive field name"
            )
        self._arm = arm
        self._endpoint = endpoint
        self._transport = transport
        self._timeout_seconds = timeout_seconds
        self._router_attempt_header = normalized_router_attempt_header

    def record(self, plan: RequestPlan) -> str:
        """Execute and durably record one parent request and every planned choice."""
        binding = self._binding(plan)
        binding_fields = binding.to_mapping()
        client_attempt_id = str(uuid.uuid4())
        body = canonical_json_bytes(dict(plan.body))
        headers = {
            **_normalize_headers(plan.headers),
            "content-type": "application/json",
            "x-gemma4-cert-root-request-id": plan.root_request_id,
            "x-gemma4-cert-client-attempt-id": client_attempt_id,
        }
        public_headers, redacted_headers = _archive_headers(headers)
        self._arm.append_event(
            "client/requests.jsonl",
            {
                "schema_version": 1,
                "root_request_id": plan.root_request_id,
                "client_attempt_id": client_attempt_id,
                "request_index": plan.request_index,
                "router_attempt_id": None,
                "router_attempt_header": self._router_attempt_header,
                "response_id": None,
                "endpoint": self._endpoint,
                "planned_choice_indices": list(plan.planned_choice_indices),
                **binding_fields,
                "headers": public_headers,
                "redacted_headers": redacted_headers,
                "raw_body_base64": base64.b64encode(body).decode("ascii"),
                "raw_body_sha256": sha256_bytes(body),
                "parsed_body": dict(plan.body),
            },
        )
        self._lineage_event(
            plan,
            binding,
            client_attempt_id,
            "CLIENT_ATTEMPT_STARTED",
            None,
        )
        try:
            exchange = self._transport.send(
                self._endpoint,
                body,
                headers,
                self._timeout_seconds,
            )
        except TransportError as error:
            self._record_transport_error(plan, binding, client_attempt_id, error)
            return client_attempt_id
        _validate_http_status(exchange.status_code)
        parsed, parse_error = _parse_json_object(exchange.body)
        normalized_response_header_items = _normalize_header_items(
            exchange.header_items
        )
        router_attempt_values = [
            value
            for name, value in normalized_response_header_items
            if name == self._router_attempt_header
        ]
        router_attempt_id = (
            router_attempt_values[0] if len(router_attempt_values) == 1 else None
        )
        archived_response_header_items = _archive_header_items(
            normalized_response_header_items
        )
        response_id = parsed.get("id") if parsed is not None else None
        if not isinstance(response_id, str):
            response_id = None
        self._arm.append_event(
            "client/responses.jsonl",
            {
                "schema_version": 1,
                "root_request_id": plan.root_request_id,
                "client_attempt_id": client_attempt_id,
                "router_attempt_id": router_attempt_id,
                "router_attempt_header": self._router_attempt_header,
                "request_index": plan.request_index,
                **binding_fields,
                "status_code": exchange.status_code,
                "header_items": archived_response_header_items,
                "raw_body_base64": base64.b64encode(exchange.body).decode("ascii"),
                "raw_body_sha256": sha256_bytes(exchange.body),
                "parsed_body": parsed,
                "parse_error": parse_error,
                "response_id": response_id,
            },
        )
        self._record_choices(
            plan,
            binding,
            client_attempt_id,
            router_attempt_id,
            response_id,
            exchange.status_code,
            parsed,
        )
        terminal = "CLIENT_ATTEMPT_COMPLETED"
        if exchange.status_code < 200 or exchange.status_code >= 300:
            terminal = "CLIENT_ATTEMPT_HTTP_ERROR"
        self._lineage_event(
            plan,
            binding,
            client_attempt_id,
            terminal,
            router_attempt_id,
        )
        return client_attempt_id

    def _record_transport_error(
        self,
        plan: RequestPlan,
        binding: ParentBinding,
        client_attempt_id: str,
        error: TransportError,
    ) -> None:
        if not isinstance(error.kind, TransportErrorKind):
            raise CertificationError("transport error has an invalid classification")
        self._arm.append_event(
            "client/responses.jsonl",
            {
                "schema_version": 1,
                "root_request_id": plan.root_request_id,
                "client_attempt_id": client_attempt_id,
                "router_attempt_id": None,
                "router_attempt_header": self._router_attempt_header,
                "request_index": plan.request_index,
                **binding.to_mapping(),
                "status_code": None,
                "header_items": [],
                "raw_body_base64": None,
                "raw_body_sha256": None,
                "parsed_body": None,
                "parse_error": None,
                "response_id": None,
                "transport_error": {"kind": error.kind.value},
            },
        )
        for choice_index in plan.planned_choice_indices:
            self._choice_event(
                plan,
                binding,
                client_attempt_id,
                None,
                None,
                choice_index,
                "REQUEST_ERROR",
                [],
            )
        self._lineage_event(
            plan,
            binding,
            client_attempt_id,
            "CLIENT_ATTEMPT_TRANSPORT_ERROR",
            None,
        )

    def _record_choices(
        self,
        plan: RequestPlan,
        binding: ParentBinding,
        client_attempt_id: str,
        router_attempt_id: str | None,
        response_id: str | None,
        status_code: int,
        parsed: Mapping[str, object] | None,
    ) -> None:
        occurrences: dict[int, list[Mapping[str, object]]] = {}
        unexpected: list[object] = []
        raw_choices = parsed.get("choices") if parsed is not None else None
        if isinstance(raw_choices, list):
            for raw_choice in raw_choices:
                if not isinstance(raw_choice, dict):
                    unexpected.append(raw_choice)
                    continue
                index = raw_choice.get("index")
                if isinstance(index, bool) or not isinstance(index, int):
                    unexpected.append(raw_choice)
                    continue
                if index not in plan.planned_choice_indices:
                    unexpected.append(raw_choice)
                    continue
                occurrences.setdefault(index, []).append(raw_choice)
        if status_code < 200 or status_code >= 300:
            outcome_for_missing = "HTTP_ERROR"
        elif parsed is None:
            outcome_for_missing = "INVALID_RESPONSE"
        else:
            outcome_for_missing = "MISSING"
        for choice_index in plan.planned_choice_indices:
            matches = occurrences.get(choice_index, [])
            if status_code < 200 or status_code >= 300:
                outcome = "HTTP_ERROR"
            elif len(matches) == 0:
                outcome = outcome_for_missing
            elif len(matches) == 1:
                outcome = "OBSERVED"
            else:
                outcome = "DUPLICATE"
            self._choice_event(
                plan,
                binding,
                client_attempt_id,
                router_attempt_id,
                response_id,
                choice_index,
                outcome,
                matches,
            )
            for occurrence_index, choice in enumerate(matches):
                self._record_tokens(
                    plan,
                    binding,
                    client_attempt_id,
                    router_attempt_id,
                    response_id,
                    choice_index,
                    occurrence_index,
                    choice,
                )
        for unexpected_index, raw_choice in enumerate(unexpected):
            self._arm.append_event(
                "client/unexpected_choices.jsonl",
                {
                    "schema_version": 1,
                    "root_request_id": plan.root_request_id,
                    "client_attempt_id": client_attempt_id,
                    "router_attempt_id": router_attempt_id,
                    "response_id": response_id,
                    "request_index": plan.request_index,
                    **binding.to_mapping(),
                    "unexpected_index": unexpected_index,
                    "raw_choice": raw_choice,
                },
            )

    def _choice_event(
        self,
        plan: RequestPlan,
        binding: ParentBinding,
        client_attempt_id: str,
        router_attempt_id: str | None,
        response_id: str | None,
        choice_index: int,
        outcome: str,
        raw_choices: Sequence[Mapping[str, object]],
    ) -> None:
        self._arm.append_event(
            "client/choices.jsonl",
            {
                "schema_version": 1,
                "root_request_id": plan.root_request_id,
                "client_attempt_id": client_attempt_id,
                "router_attempt_id": router_attempt_id,
                "response_id": response_id,
                "request_index": plan.request_index,
                **binding.to_mapping(),
                "planned_choice_index": choice_index,
                "outcome": outcome,
                "occurrence_count": len(raw_choices),
                "raw_choices": list(raw_choices),
                "raw_logprobs": [choice.get("logprobs") for choice in raw_choices],
            },
        )

    def _record_tokens(
        self,
        plan: RequestPlan,
        binding: ParentBinding,
        client_attempt_id: str,
        router_attempt_id: str | None,
        response_id: str | None,
        choice_index: int,
        occurrence_index: int,
        choice: Mapping[str, object],
    ) -> None:
        for token_index, token in enumerate(token_entries(choice)):
            self._arm.append_event(
                "client/tokens.jsonl",
                {
                    "schema_version": 1,
                    "root_request_id": plan.root_request_id,
                    "client_attempt_id": client_attempt_id,
                    "router_attempt_id": router_attempt_id,
                    "response_id": response_id,
                    "request_index": plan.request_index,
                    **binding.to_mapping(),
                    "planned_choice_index": choice_index,
                    "choice_occurrence_index": occurrence_index,
                    "token_index": token_index,
                    "raw_token_record": token,
                },
            )

    def _lineage_event(
        self,
        plan: RequestPlan,
        binding: ParentBinding,
        client_attempt_id: str,
        event_type: str,
        router_attempt_id: str | None,
    ) -> None:
        self._arm.append_event(
            "lineage.jsonl",
            {
                "schema_version": 1,
                "event_type": event_type,
                "root_request_id": plan.root_request_id,
                "client_attempt_id": client_attempt_id,
                "router_attempt_id": router_attempt_id,
                "p_registration_id": None,
                "d_child_id": None,
                "physical_pull_id": None,
                "request_index": plan.request_index,
                **binding.to_mapping(),
            },
        )

    def _binding(self, plan: RequestPlan) -> ParentBinding:
        binding = self._arm.plan.parent_binding(plan.request_index)
        if plan.body.get("seed") != binding.seed:
            raise CertificationError(
                "request seed does not match plan binding for index "
                f"{plan.request_index}"
            )
        template = dict(plan.body)
        del template["seed"]
        digest = sha256_bytes(canonical_json_bytes(template))
        if digest != binding.request_template_sha256:
            raise CertificationError(
                "request template digest does not match plan payload for "
                f"index {plan.request_index}"
            )
        return binding


def load_request_plans(path: Path) -> list[RequestPlan]:
    """Load request plans from JSONL, refusing malformed or duplicate identities."""
    plans = []
    identities: set[str] = set()
    indices: set[int] = set()
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if len(line.strip()) == 0:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as error:
                raise CertificationError(
                    f"invalid request plan line {line_number}"
                ) from error
            if not isinstance(parsed, dict):
                raise CertificationError(
                    f"request plan line {line_number} is not an object"
                )
            plan = RequestPlan.from_mapping(parsed)
            if plan.root_request_id in identities:
                raise CertificationError(
                    f"duplicate root request ID: {plan.root_request_id}"
                )
            if plan.request_index in indices:
                raise CertificationError(
                    f"duplicate request index: {plan.request_index}"
                )
            identities.add(plan.root_request_id)
            indices.add(plan.request_index)
            plans.append(plan)
    if len(plans) == 0:
        raise CertificationError("request plan is empty")
    if sorted(indices) != list(range(len(indices))):
        raise CertificationError("request indices must be contiguous from zero")
    return sorted(plans, key=lambda plan: plan.request_index)


def _archive_headers(
    headers: Mapping[str, str],
) -> tuple[dict[str, str], list[str]]:
    public = {}
    redacted = []
    for key, value in headers.items():
        normalized = key.lower()
        if is_sensitive_name(normalized):
            public[key] = "<redacted>"
            redacted.append(key)
        else:
            public[key] = value
    return public, sorted(redacted)


def _archive_header_items(
    header_items: Sequence[tuple[str, str]],
) -> list[dict[str, object]]:
    archived = []
    for name, value in header_items:
        redacted = is_sensitive_name(name)
        archived.append(
            {
                "name": name,
                "value": "<redacted>" if redacted else value,
                "redacted": redacted,
            }
        )
    return archived


def _normalize_headers(headers: Mapping[str, str]) -> dict[str, str]:
    normalized = {}
    for key, value in headers.items():
        if not is_valid_http_header_name(key):
            raise CertificationError("header name is not a valid HTTP field name")
        if "\r" in value or "\n" in value:
            raise CertificationError("header value cannot contain CR or LF")
        name = key.lower()
        if name in normalized:
            raise CertificationError(f"duplicate header after normalization: {key}")
        normalized[name] = value
    return normalized


def _normalize_header_items(
    header_items: Sequence[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    normalized = []
    for name, value in header_items:
        if not is_valid_http_header_name(name):
            raise CertificationError("header name is not a valid HTTP field name")
        if "\r" in value or "\n" in value:
            raise CertificationError("header value cannot contain CR or LF")
        normalized.append((name.lower(), value))
    return tuple(normalized)


def _validate_http_status(status_code: int) -> None:
    if (
        isinstance(status_code, bool)
        or not isinstance(status_code, int)
        or status_code < 100
        or status_code > 599
    ):
        raise CertificationError("HTTP status code must be an integer from 100 to 599")


def _parse_json_object(body: bytes) -> tuple[Mapping[str, object] | None, str | None]:
    try:
        parsed = json.loads(body, parse_constant=_reject_json_constant)
        if not isinstance(parsed, dict):
            return None, "response JSON is not an object"
        canonical_json_bytes(parsed)
    except (json.JSONDecodeError, UnicodeError, TypeError, ValueError) as error:
        return None, f"{type(error).__name__}: {error}"
    return parsed, None


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant: {value}")
