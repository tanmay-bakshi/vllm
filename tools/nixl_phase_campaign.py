# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run an accountable, cache-isolated P-to-D pathology campaign.

The choice classifier is the classifier used by the storm36 reproducer. Each
HTTP request is one parent generation. A deterministic request identity and a
distinct cache salt are attached to every parent so that no sample can reuse
another sample's prefetched prefix.
"""

import argparse
import base64
import hashlib
import json
import os
import re
import statistics
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

_TAIL_WINDOW_CHARS: int = 400
_MAX_REPEAT_PERIOD: int = 32
_MIN_DEGENERATE_TAIL_CHARS: int = 200
_TAIL_COVERAGE_THRESHOLD: float = 0.6
_ZLIB_RATIO_THRESHOLD: float = 8.0
_DISTINCT_RATIO_THRESHOLD: float = 0.12
_VERBOSE_REASONING_CHARS: int = 2500
_CAMPAIGN_ID_PATTERN: re.Pattern[str] = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


@dataclass(frozen=True)
class CampaignConfig:
    """Configuration shared by every parent in one campaign arm.

    :ivar base_url: OpenAI-compatible base URL ending in ``/v1``.
    :ivar campaign_id: Stable, arm-unique identity used in request IDs and salts.
    :ivar samples: Parent requests issued for each payload.
    :ivar concurrency: Maximum concurrently active parent requests.
    :ivar choices_per_parent: Choices requested from each parent.
    :ivar max_tokens: Optional completion-token override.
    :ivar temperature: Optional sampling-temperature override.
    :ivar seed_base: Optional first deterministic sampling seed.
    :ivar timeout_seconds: HTTP timeout for one parent request.
    :ivar capture_dir: Optional exclusive directory for pathological responses.
    :ivar capture_request_id: Optional exact request whose complete response is
        captured regardless of classification.
    """

    base_url: str
    campaign_id: str
    samples: int
    concurrency: int
    choices_per_parent: int
    max_tokens: int | None
    temperature: float | None
    seed_base: int | None
    timeout_seconds: float
    capture_dir: Path | None
    capture_request_id: str | None


@dataclass(frozen=True)
class PayloadSpec:
    """A validated request body and its position in the campaign.

    :ivar ordinal: Zero-based payload position, including duplicate paths.
    :ivar name: Human-readable payload name.
    :ivar body: Chat Completions request body.
    """

    ordinal: int
    name: str
    body: dict[str, Any]


@dataclass(frozen=True)
class ParentResult:
    """Rows and accounting observations produced by one parent request.

    :ivar rows: JSONL rows produced for the parent.
    :ivar actual_choices: Choices present in the server response.
    :ivar errors: Request or response-accounting errors.
    """

    rows: tuple[dict[str, Any], ...]
    actual_choices: int
    errors: tuple[str, ...]


@dataclass(frozen=True)
class CampaignOutcome:
    """Aggregate validity and pathology counts for a campaign arm.

    :ivar counts: Result-label and fatal-shape counts.
    :ivar parent_count: Parent requests attempted.
    :ivar expected_choices: Choices required by the campaign configuration.
    :ivar actual_choices: Choices observed in server responses.
    :ivar failed_parents: Parents with request or accounting errors.
    """

    counts: dict[str, int]
    parent_count: int
    expected_choices: int
    actual_choices: int
    failed_parents: int

    @property
    def valid(self) -> bool:
        """:returns: Whether every parent returned exactly the expected choices."""
        return self.failed_parents == 0 and self.actual_choices == self.expected_choices


def _load_request(path: Path) -> dict[str, Any]:
    """:param path: Pathology payload JSON or raw Chat Completions body.
    :returns: A validated Chat Completions request body ready to specialize.
    :raises ValueError: If the JSON does not contain an object request body.
    """
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"payload must be a JSON object: {path}")
    request = payload.get("request", payload)
    if not isinstance(request, dict):
        raise ValueError(f"payload request must be a JSON object: {path}")
    body = {key: value for key, value in request.items() if key != "extra_headers"}
    body["stream"] = False
    return body


def _load_payloads(paths: list[Path]) -> tuple[PayloadSpec, ...]:
    """:param paths: Payload paths in command-line order.
    :returns: Validated payload specifications.
    """
    return tuple(
        PayloadSpec(
            ordinal=ordinal,
            name=path.stem,
            body=_load_request(path),
        )
        for ordinal, path in enumerate(paths)
    )


def _parent_identity(
    campaign_id: str,
    payload_ordinal: int,
    sample_index: int,
) -> tuple[str, str, str]:
    """:param campaign_id: Stable identity for the campaign arm.
    :param payload_ordinal: Zero-based payload position.
    :param sample_index: Zero-based parent position within the payload.
    :returns: Parent identity, API request identity, and 256-bit cache salt.
    """
    parent_id = f"{campaign_id}:p{payload_ordinal:03d}:s{sample_index:06d}"
    request_id = f"p2d-{campaign_id}-p{payload_ordinal:03d}-s{sample_index:06d}"
    salt_material = (
        f"nixl-phase-campaign-v1\0{campaign_id}\0{payload_ordinal}\0{sample_index}"
    ).encode()
    cache_salt = base64.urlsafe_b64encode(hashlib.sha256(salt_material).digest())
    return parent_id, request_id, cache_salt.decode().rstrip("=")


def _post_chat_completion(
    base_url: str,
    body: dict[str, Any],
    request_id: str,
    timeout_seconds: float,
) -> Any:
    """:param base_url: OpenAI-compatible base URL ending in ``/v1``.
    :param body: Chat Completions request body.
    :param request_id: Exact external request identity.
    :param timeout_seconds: HTTP timeout for the request.
    :returns: Decoded response JSON.
    """
    data = json.dumps(body).encode()
    http_request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=data,
        headers={
            "Content-Type": "application/json",
            "X-Request-Id": request_id,
        },
        method="POST",
    )
    with urllib.request.urlopen(http_request, timeout=timeout_seconds) as response:
        return json.loads(response.read())


def _tail_repetition_coverage(text: str) -> float:
    """:param text: Text whose trailing window is judged for periodic repetition.
    :returns: Best fraction of the trailing window covered by one short repeating unit.
    """
    tail = text[-_TAIL_WINDOW_CHARS:]
    if len(tail) < _MIN_DEGENERATE_TAIL_CHARS:
        return 0.0
    best = 0.0
    for period in range(1, _MAX_REPEAT_PERIOD + 1):
        unit = tail[-period:]
        if unit.strip() == "":
            continue
        repeats = 0
        position = len(tail) - period
        while position >= 0 and tail[position : position + period] == unit:
            repeats += 1
            position -= period
        best = max(best, repeats * period / len(tail))
    return best


def _zlib_ratio(text: str) -> float:
    """:param text: Text whose trailing window is judged for compressibility.
    :returns: Raw-to-compressed length ratio of the trailing window.
    """
    tail = text[-_TAIL_WINDOW_CHARS * 2 :].encode()
    if len(tail) < _MIN_DEGENERATE_TAIL_CHARS:
        return 1.0
    return len(tail) / max(1, len(zlib.compress(tail, level=6)))


def _distinct_token_ratio(text: str) -> float:
    """:param text: Text whose trailing window is judged for vocabulary collapse.
    :returns: Distinct-to-total token ratio of the trailing window; 1.0 when short.
    """
    tokens = text[-_TAIL_WINDOW_CHARS:].split()
    if len(tokens) < 20:
        return 1.0
    return len(set(tokens)) / len(tokens)


def _classify_choice(choice: dict[str, Any], usage: dict[str, Any]) -> dict[str, Any]:
    """:param choice: One Chat Completions response choice.
    :param usage: Response usage block shared across choices.
    :returns: Historical storm36 metrics and pathology flags for the choice.
    """
    message = choice.get("message") or {}
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    content = message.get("content") or ""
    tool_calls = message.get("tool_calls") or []
    arguments = "".join(
        (call.get("function") or {}).get("arguments") or "" for call in tool_calls
    )
    text = reasoning + content + arguments
    coverage = _tail_repetition_coverage(text)
    compress_ratio = _zlib_ratio(text)
    distinct_ratio = _distinct_token_ratio(text)
    degenerate = (
        coverage >= _TAIL_COVERAGE_THRESHOLD
        or compress_ratio >= _ZLIB_RATIO_THRESHOLD
        or distinct_ratio <= _DISTINCT_RATIO_THRESHOLD
    )
    fatal_shape = content == "" and len(tool_calls) == 0
    finish_reason = choice.get("finish_reason")
    verbose = len(reasoning) >= _VERBOSE_REASONING_CHARS
    if degenerate:
        label = "degenerate"
    elif finish_reason == "length":
        label = "truncated"
    elif verbose:
        label = "verbose"
    else:
        label = "healthy"
    return {
        "label": label,
        "fatal_shape": fatal_shape,
        "finish_reason": finish_reason,
        "completion_tokens": usage.get("completion_tokens"),
        "reasoning_chars": len(reasoning),
        "content_chars": len(content),
        "arguments_chars": len(arguments),
        "tool_call_count": len(tool_calls),
        "tool_call_names": [
            (call.get("function") or {}).get("name") for call in tool_calls
        ],
        "tail_repetition_coverage": round(coverage, 3),
        "zlib_ratio": round(compress_ratio, 2),
        "distinct_token_ratio": round(distinct_ratio, 3),
        "tail_repr": (
            repr(text[-160:]) if degenerate or finish_reason == "length" else None
        ),
    }


def _write_json_no_replace(path: Path, value: object) -> None:
    """:param path: Exclusive destination path.
    :param value: JSON-serializable value.
    :raises FileExistsError: If the destination already exists.
    """
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _capture_selected_response(
    config: CampaignConfig,
    *,
    parent_id: str,
    request_id: str,
    cache_salt: str,
    request_seed: Any,
    response: object,
    http_status: int,
) -> Path | None:
    """Capture the complete response for the exact selected request.

    :param config: Campaign configuration containing the capture selector.
    :param parent_id: Stable campaign parent identity.
    :param request_id: External API request identity.
    :param cache_salt: Parent-specific prefix-cache salt.
    :param request_seed: Sampling seed sent for the parent.
    :param response: Decoded response body, or a lossless raw-body envelope.
    :param http_status: HTTP response status.
    :returns: Exclusive capture path when this is the selected request.
    """
    if (
        config.capture_dir is None
        or config.capture_request_id is None
        or request_id != config.capture_request_id
    ):
        return None
    capture_path = config.capture_dir / f"{request_id}-response.json"
    _write_json_no_replace(
        capture_path,
        {
            "parent_id": parent_id,
            "request_id": request_id,
            "cache_salt": cache_salt,
            "request_seed": request_seed,
            "http_status": http_status,
            "response": response,
        },
    )
    return capture_path


def _error_parent_result(
    *,
    payload: PayloadSpec,
    sample_index: int,
    parent_id: str,
    request_id: str,
    cache_salt: str,
    seed: Any,
    started_at_unix_ns: int,
    elapsed_seconds: float,
    error: str,
    selected_capture_path: Path | None = None,
) -> ParentResult:
    """:param payload: Payload attempted by the failed parent.
    :param sample_index: Parent position within the payload.
    :param parent_id: Stable campaign parent identity.
    :param request_id: External API request identity.
    :param cache_salt: Parent-specific prefix-cache salt.
    :param seed: Sampling seed sent for the parent.
    :param started_at_unix_ns: Wall-clock start timestamp.
    :param elapsed_seconds: Time spent on the failed request.
    :param error: Failure description.
    :param selected_capture_path: Complete-response capture for the selected
        request, when available.
    :returns: A single-row invalid result for the failed parent.
    """
    row = {
        "payload": payload.name,
        "payload_ordinal": payload.ordinal,
        "sample_index": sample_index,
        "parent_id": parent_id,
        "request_id": request_id,
        "response_id": None,
        "cache_salt": cache_salt,
        "choice_index": None,
        "seed": seed,
        "started_at_unix_ns": started_at_unix_ns,
        "elapsed_seconds": round(elapsed_seconds, 6),
        "label": "request_error",
        "fatal_shape": False,
        "error": error,
    }
    if selected_capture_path is not None:
        row["selected_capture_path"] = str(selected_capture_path)
    return ParentResult(rows=(row,), actual_choices=0, errors=(error,))


def _sample_parent(
    config: CampaignConfig,
    payload: PayloadSpec,
    sample_index: int,
) -> ParentResult:
    """:param config: Campaign configuration.
    :param payload: Payload sampled by this parent.
    :param sample_index: Parent position within the payload.
    :returns: Classified rows and strict response accounting.
    """
    parent_id, request_id, cache_salt = _parent_identity(
        config.campaign_id,
        payload.ordinal,
        sample_index,
    )
    body = dict(payload.body)
    body["stream"] = False
    body["n"] = config.choices_per_parent
    body["request_id"] = request_id
    body["cache_salt"] = cache_salt
    if config.max_tokens is not None:
        body["max_completion_tokens"] = config.max_tokens
    if config.temperature is not None:
        body["temperature"] = config.temperature
    if config.seed_base is not None:
        body["seed"] = config.seed_base + sample_index

    started_at_unix_ns = time.time_ns()
    started_at_monotonic = time.monotonic()
    try:
        response = _post_chat_completion(
            config.base_url,
            body,
            request_id,
            config.timeout_seconds,
        )
    except urllib.error.HTTPError as error:
        response_bytes = error.read()
        try:
            error_response: object = json.loads(response_bytes)
        except (json.JSONDecodeError, UnicodeDecodeError):
            error_response = {
                "raw_body_utf8": response_bytes.decode("utf-8", errors="replace")
            }
        selected_capture_path = _capture_selected_response(
            config,
            parent_id=parent_id,
            request_id=request_id,
            cache_salt=cache_salt,
            request_seed=body.get("seed"),
            response=error_response,
            http_status=error.code,
        )
        return _error_parent_result(
            payload=payload,
            sample_index=sample_index,
            parent_id=parent_id,
            request_id=request_id,
            cache_salt=cache_salt,
            seed=body.get("seed"),
            started_at_unix_ns=started_at_unix_ns,
            elapsed_seconds=time.monotonic() - started_at_monotonic,
            error=f"HTTPError {error.code}: {error.reason}",
            selected_capture_path=selected_capture_path,
        )
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
        return _error_parent_result(
            payload=payload,
            sample_index=sample_index,
            parent_id=parent_id,
            request_id=request_id,
            cache_salt=cache_salt,
            seed=body.get("seed"),
            started_at_unix_ns=started_at_unix_ns,
            elapsed_seconds=time.monotonic() - started_at_monotonic,
            error=f"{type(error).__name__}: {error}",
        )

    elapsed_seconds = time.monotonic() - started_at_monotonic
    if not isinstance(response, dict):
        return _error_parent_result(
            payload=payload,
            sample_index=sample_index,
            parent_id=parent_id,
            request_id=request_id,
            cache_salt=cache_salt,
            seed=body.get("seed"),
            started_at_unix_ns=started_at_unix_ns,
            elapsed_seconds=elapsed_seconds,
            error="response JSON must be an object",
        )

    selected_capture_path = _capture_selected_response(
        config,
        parent_id=parent_id,
        request_id=request_id,
        cache_salt=cache_salt,
        request_seed=body.get("seed"),
        response=response,
        http_status=200,
    )

    errors: list[str] = []
    response_id = response.get("id")
    expected_response_id = f"chatcmpl-{request_id}"
    if response_id != expected_response_id:
        errors.append(
            f"response id {response_id!r} does not match {expected_response_id!r}"
        )
    choices_value = response.get("choices")
    if not isinstance(choices_value, list):
        return _error_parent_result(
            payload=payload,
            sample_index=sample_index,
            parent_id=parent_id,
            request_id=request_id,
            cache_salt=cache_salt,
            seed=body.get("seed"),
            started_at_unix_ns=started_at_unix_ns,
            elapsed_seconds=elapsed_seconds,
            error="response choices must be a list",
        )
    choices: list[Any] = choices_value
    if len(choices) != config.choices_per_parent:
        errors.append(
            f"received {len(choices)} choices; expected {config.choices_per_parent}"
        )

    usage_value = response.get("usage")
    if usage_value is None:
        usage: dict[str, Any] = {}
    elif isinstance(usage_value, dict):
        usage = usage_value
    else:
        usage = {}
        errors.append("response usage must be an object when present")

    rows: list[dict[str, Any]] = []
    observed_indices: list[int] = []
    for response_position, choice_value in enumerate(choices):
        if not isinstance(choice_value, dict):
            errors.append(
                f"choice at response position {response_position} is not an object"
            )
            continue
        choice: dict[str, Any] = choice_value
        choice_index = choice.get("index")
        if type(choice_index) is int:
            observed_indices.append(choice_index)
        else:
            errors.append(
                f"choice at response position {response_position} has invalid index"
            )
        row = {
            "payload": payload.name,
            "payload_ordinal": payload.ordinal,
            "sample_index": sample_index,
            "parent_id": parent_id,
            "request_id": request_id,
            "response_id": response_id,
            "cache_salt": cache_salt,
            "choice_index": choice_index,
            "seed": body.get("seed"),
            "started_at_unix_ns": started_at_unix_ns,
            "elapsed_seconds": round(elapsed_seconds, 6),
            **_classify_choice(choice, usage),
        }
        if selected_capture_path is not None:
            row["selected_capture_path"] = str(selected_capture_path)
        if config.capture_dir is not None and (
            row["label"] == "degenerate" or row["fatal_shape"] is True
        ):
            capture_path = config.capture_dir / (
                f"{request_id}-r{response_position:03d}.json"
            )
            _write_json_no_replace(
                capture_path,
                {
                    "parent_id": parent_id,
                    "request_id": request_id,
                    "cache_salt": cache_salt,
                    "request_seed": body.get("seed"),
                    "choice": choice,
                },
            )
            row["capture_path"] = str(capture_path)
        rows.append(row)

    if sorted(observed_indices) != list(range(config.choices_per_parent)):
        errors.append(
            f"choice indices {observed_indices!r} do not match "
            f"{list(range(config.choices_per_parent))!r}"
        )
    if len(errors) > 0:
        for row in rows:
            row["accounting_errors"] = list(errors)
    if len(rows) == 0:
        return _error_parent_result(
            payload=payload,
            sample_index=sample_index,
            parent_id=parent_id,
            request_id=request_id,
            cache_salt=cache_salt,
            seed=body.get("seed"),
            started_at_unix_ns=started_at_unix_ns,
            elapsed_seconds=elapsed_seconds,
            error="; ".join(errors),
        )
    return ParentResult(
        rows=tuple(rows),
        actual_choices=len(choices),
        errors=tuple(errors),
    )


def _sample_payload(
    config: CampaignConfig,
    payload: PayloadSpec,
    out_handle: TextIO,
) -> tuple[dict[str, int], int, int]:
    """:param config: Campaign configuration.
    :param payload: Payload sampled by this group of parents.
    :param out_handle: Open JSONL results sink.
    :returns: Label counts, observed choices, and failed-parent count.
    """
    counts: dict[str, int] = {}
    tokens: list[int] = []
    actual_choices = 0
    failed_parents = 0

    def one(sample_index: int) -> ParentResult:
        """:param sample_index: Parent position within the payload.
        :returns: Result for one parent request.
        """
        return _sample_parent(config, payload, sample_index)

    with ThreadPoolExecutor(max_workers=config.concurrency) as pool:
        for parent_result in pool.map(one, range(config.samples)):
            actual_choices += parent_result.actual_choices
            if len(parent_result.errors) > 0:
                failed_parents += 1
            for row in parent_result.rows:
                label = row["label"]
                counts[label] = counts.get(label, 0) + 1
                if row.get("fatal_shape") is True:
                    counts["fatal_shape"] = counts.get("fatal_shape", 0) + 1
                completion_tokens = row.get("completion_tokens")
                if type(completion_tokens) is int:
                    tokens.append(completion_tokens)
                out_handle.write(json.dumps(row, sort_keys=True) + "\n")
                out_handle.flush()
                if (
                    label in ("degenerate", "request_error")
                    or row.get("fatal_shape") is True
                ):
                    print(
                        f"  HIT parent {row['parent_id']} choice "
                        f"{row.get('choice_index')}: {label}"
                    )

    token_summary = ""
    if len(tokens) > 0:
        token_summary = (
            f" completion_tokens p50={int(statistics.median(tokens))} max={max(tokens)}"
        )
    print(f"{payload.name}: {json.dumps(counts, sort_keys=True)}{token_summary}")
    return counts, actual_choices, failed_parents


def run_campaign(
    config: CampaignConfig,
    payloads: tuple[PayloadSpec, ...],
    out_handle: TextIO,
) -> CampaignOutcome:
    """:param config: Campaign configuration.
    :param payloads: Validated payloads in campaign order.
    :param out_handle: Open JSONL results sink.
    :returns: Aggregate campaign validity and pathology counts.
    :raises ValueError: If the campaign configuration is invalid.
    """
    if _CAMPAIGN_ID_PATTERN.fullmatch(config.campaign_id) is None:
        raise ValueError("campaign_id must match [A-Za-z0-9][A-Za-z0-9._-]{0,63}")
    if len(config.base_url) == 0:
        raise ValueError("base_url must not be empty")
    if config.samples <= 0:
        raise ValueError("samples must be positive")
    if config.concurrency <= 0:
        raise ValueError("concurrency must be positive")
    if config.choices_per_parent <= 0:
        raise ValueError("choices_per_parent must be positive")
    if config.timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if config.capture_request_id is not None and len(config.capture_request_id) == 0:
        raise ValueError("capture_request_id must not be empty")
    if config.capture_request_id is not None and config.capture_dir is None:
        raise ValueError("capture_request_id requires capture_dir")
    if len(payloads) == 0:
        raise ValueError("at least one payload is required")

    totals: dict[str, int] = {}
    actual_choices = 0
    failed_parents = 0
    for payload in payloads:
        counts, payload_choices, payload_failures = _sample_payload(
            config,
            payload,
            out_handle,
        )
        actual_choices += payload_choices
        failed_parents += payload_failures
        for label, count in counts.items():
            totals[label] = totals.get(label, 0) + count
    parent_count = len(payloads) * config.samples
    return CampaignOutcome(
        counts=totals,
        parent_count=parent_count,
        expected_choices=parent_count * config.choices_per_parent,
        actual_choices=actual_choices,
        failed_parents=failed_parents,
    )


def _positive_int(value: str) -> int:
    """:param value: Command-line integer text.
    :returns: Parsed positive integer.
    :raises argparse.ArgumentTypeError: If the value is not positive.
    """
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _positive_float(value: str) -> float:
    """:param value: Command-line floating-point text.
    :returns: Parsed positive float.
    :raises argparse.ArgumentTypeError: If the value is not positive.
    """
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _campaign_id(value: str) -> str:
    """:param value: Command-line campaign identity.
    :returns: Validated campaign identity.
    :raises argparse.ArgumentTypeError: If the identity is unsafe or ambiguous.
    """
    if _CAMPAIGN_ID_PATTERN.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("must match [A-Za-z0-9][A-Za-z0-9._-]{0,63}")
    return value


def _parser() -> argparse.ArgumentParser:
    """:returns: Campaign command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("payloads", nargs="+", type=Path)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--campaign-id", required=True, type=_campaign_id)
    parser.add_argument("--samples", type=_positive_int, default=32)
    parser.add_argument("--concurrency", type=_positive_int, default=4)
    parser.add_argument("--n", type=_positive_int, default=1, dest="choices_per_parent")
    parser.add_argument("--max-tokens", type=_positive_int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--seed-base", type=int, default=None)
    parser.add_argument(
        "--timeout-seconds",
        type=_positive_float,
        default=float(os.environ.get("REPRO_TIMEOUT_SECONDS", "300")),
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--capture-dir", type=Path, default=None)
    parser.add_argument("--capture-request-id", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    """:param argv: Optional command-line arguments excluding the executable.
    :returns: Zero for a valid arm, one for request/accounting failure, or two
        for setup failure.
    """
    args = _parser().parse_args(argv)
    try:
        payloads = _load_payloads(args.payloads)
        if args.out.exists():
            raise FileExistsError(f"refusing to overwrite output: {args.out}")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        if args.capture_dir is not None:
            args.capture_dir.parent.mkdir(parents=True, exist_ok=True)
            args.capture_dir.mkdir()
        config = CampaignConfig(
            base_url=args.base_url,
            campaign_id=args.campaign_id,
            samples=args.samples,
            concurrency=args.concurrency,
            choices_per_parent=args.choices_per_parent,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            seed_base=args.seed_base,
            timeout_seconds=args.timeout_seconds,
            capture_dir=args.capture_dir,
            capture_request_id=args.capture_request_id,
        )
        descriptor, temporary_name = tempfile.mkstemp(
            dir=args.out.parent,
            prefix=f".{args.out.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as out_handle:
                outcome = run_campaign(config, payloads, out_handle)
                out_handle.flush()
                os.fsync(out_handle.fileno())
            os.link(temporary_path, args.out)
        finally:
            temporary_path.unlink(missing_ok=True)
    except (OSError, ValueError) as error:
        print(
            f"campaign setup failed: {type(error).__name__}: {error}", file=sys.stderr
        )
        return 2

    print(
        "TOTAL: "
        f"{json.dumps(outcome.counts, sort_keys=True)} "
        f"parents={outcome.parent_count} "
        f"choices={outcome.actual_choices}/{outcome.expected_choices} "
        f"failed_parents={outcome.failed_parents} valid={str(outcome.valid).lower()}"
    )
    return 0 if outcome.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
