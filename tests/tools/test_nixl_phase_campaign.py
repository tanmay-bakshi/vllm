# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-only tests for the cache-isolated NIXL phase campaign driver."""

import json
import urllib.error
import urllib.request
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from tools import nixl_phase_campaign as campaign


def _config(
    *,
    samples: int = 1,
    choices_per_parent: int = 1,
    capture_dir: Path | None = None,
    capture_request_id: str | None = None,
) -> campaign.CampaignConfig:
    """:param samples: Parent requests in the synthetic campaign.
    :param choices_per_parent: Choices required from each parent.
    :param capture_dir: Optional response-capture directory.
    :param capture_request_id: Optional exact response-capture selector.
    :returns: Host-only campaign configuration.
    """
    return campaign.CampaignConfig(
        base_url="http://127.0.0.1:1/v1",
        campaign_id="control-a1",
        samples=samples,
        concurrency=2,
        choices_per_parent=choices_per_parent,
        max_tokens=None,
        temperature=None,
        seed_base=100,
        timeout_seconds=1.0,
        capture_dir=capture_dir,
        capture_request_id=capture_request_id,
    )


def _payload() -> campaign.PayloadSpec:
    """:returns: Minimal synthetic Chat Completions payload."""
    return campaign.PayloadSpec(
        ordinal=0,
        name="pathology",
        body={"model": "test", "messages": [{"role": "user", "content": "x"}]},
    )


def _choice(index: int, content: str = "healthy") -> dict[str, Any]:
    """:param index: Choice index returned by the synthetic server.
    :param content: Synthetic assistant content.
    :returns: Chat Completions choice object.
    """
    return {
        "index": index,
        "finish_reason": "stop",
        "message": {"content": content, "tool_calls": []},
    }


def test_historical_classifier_labels_and_fatal_shape_are_preserved() -> None:
    degenerate = campaign._classify_choice(
        {
            "finish_reason": "length",
            "message": {"content": " l" * 300, "tool_calls": []},
        },
        {},
    )
    truncated = campaign._classify_choice(
        {
            "finish_reason": "length",
            "message": {"content": "ordinary response", "tool_calls": []},
        },
        {},
    )
    verbose_reasoning = " ".join(f"token{index}" for index in range(500))
    verbose = campaign._classify_choice(
        {
            "finish_reason": "stop",
            "message": {
                "reasoning_content": verbose_reasoning,
                "content": "answer",
                "tool_calls": [],
            },
        },
        {},
    )
    fatal = campaign._classify_choice(
        {
            "finish_reason": "stop",
            "message": {"reasoning_content": "brief", "content": None},
        },
        {},
    )

    assert degenerate["label"] == "degenerate"
    assert truncated["label"] == "truncated"
    assert verbose["label"] == "verbose"
    assert fatal["label"] == "healthy"
    assert fatal["fatal_shape"] is True


def test_parent_identities_and_cache_salts_are_deterministic_and_unique() -> None:
    identities = [
        campaign._parent_identity("control-a1", 0, sample_index)
        for sample_index in range(16)
    ]

    assert identities[3] == campaign._parent_identity("control-a1", 0, 3)
    assert len({identity[0] for identity in identities}) == 16
    assert len({identity[1] for identity in identities}) == 16
    assert len({identity[2] for identity in identities}) == 16
    assert all(len(identity[2]) == 43 for identity in identities)
    assert identities[0] != campaign._parent_identity("control-a1", 1, 0)
    assert identities[0] != campaign._parent_identity("phase-a1", 0, 0)


def test_http_request_carries_the_same_identity_in_header_and_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = MagicMock()
    response.__enter__.return_value = response
    response.read.return_value = b'{"id": "chatcmpl-parent"}'

    def urlopen(
        request: urllib.request.Request,
        timeout: float,
    ) -> MagicMock:
        """:param request: HTTP request under test.
        :param timeout: HTTP timeout under test.
        :returns: Synthetic HTTP response context manager.
        """
        assert request.full_url == "http://127.0.0.1:1/v1/chat/completions"
        assert request.get_header("X-request-id") == "parent"
        request_body = request.data
        assert isinstance(request_body, bytes)
        assert json.loads(request_body)["request_id"] == "parent"
        assert timeout == 3.0
        return response

    monkeypatch.setattr(campaign.urllib.request, "urlopen", urlopen)

    decoded = campaign._post_chat_completion(
        "http://127.0.0.1:1/v1/",
        {"request_id": "parent"},
        "parent",
        3.0,
    )

    assert decoded == {"id": "chatcmpl-parent"}


def test_valid_campaign_salts_every_parent_and_records_exact_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_requests: list[tuple[dict[str, Any], str]] = []

    def post(
        base_url: str,
        body: dict[str, Any],
        request_id: str,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        """:param base_url: Synthetic endpoint URL.
        :param body: Request body under test.
        :param request_id: External request identity under test.
        :param timeout_seconds: Synthetic request timeout.
        :returns: A fully accounted synthetic two-choice response.
        """
        assert base_url.endswith("/v1")
        assert timeout_seconds == 1.0
        observed_requests.append((dict(body), request_id))
        return {
            "id": f"chatcmpl-{request_id}",
            "choices": [_choice(0), _choice(1)],
            "usage": {"completion_tokens": 7},
        }

    monkeypatch.setattr(campaign, "_post_chat_completion", post)
    output = StringIO()
    outcome = campaign.run_campaign(
        _config(samples=4, choices_per_parent=2),
        (_payload(),),
        output,
    )
    rows = [json.loads(line) for line in output.getvalue().splitlines()]

    assert outcome.valid
    assert outcome.parent_count == 4
    assert outcome.actual_choices == 8
    assert len(rows) == 8
    assert len(observed_requests) == 4
    assert len({body["cache_salt"] for body, _ in observed_requests}) == 4
    assert len({body["request_id"] for body, _ in observed_requests}) == 4
    assert all(
        body["request_id"] == request_id for body, request_id in observed_requests
    )
    assert all(body["n"] == 2 for body, _ in observed_requests)
    assert all(body["stream"] is False for body, _ in observed_requests)
    assert all(row["response_id"] == f"chatcmpl-{row['request_id']}" for row in rows)
    assert all(type(row["started_at_unix_ns"]) is int for row in rows)
    assert all(row["elapsed_seconds"] >= 0.0 for row in rows)
    assert len({row["parent_id"] for row in rows}) == 4
    assert len({row["cache_salt"] for row in rows}) == 4


def test_selected_response_is_captured_even_when_healthy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The selected healthy response is retained in full."""
    request_id = "p2d-control-a1-p000-s000000"
    response = {
        "id": f"chatcmpl-{request_id}",
        "choices": [_choice(0)],
        "usage": {"completion_tokens": 7},
    }

    def post(
        base_url: str,
        body: dict[str, Any],
        observed_request_id: str,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        """:returns: Complete synthetic healthy response."""
        assert len(base_url) > 0
        assert len(body) > 0
        assert observed_request_id == request_id
        assert timeout_seconds == 1.0
        return response

    capture_dir = tmp_path / "captures"
    capture_dir.mkdir()
    monkeypatch.setattr(campaign, "_post_chat_completion", post)
    output = StringIO()
    outcome = campaign.run_campaign(
        _config(
            capture_dir=capture_dir,
            capture_request_id=request_id,
        ),
        (_payload(),),
        output,
    )

    row = json.loads(output.getvalue())
    capture_path = capture_dir / f"{request_id}-response.json"
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    assert outcome.valid
    assert row["selected_capture_path"] == str(capture_path)
    assert capture["http_status"] == 200
    assert capture["response"] == response


def test_selected_http_error_body_is_captured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The selected fail-closed response body survives HTTPError handling."""
    request_id = "p2d-control-a1-p000-s000000"
    error_response = {"error": {"type": "localization_mismatch"}}

    def post(
        base_url: str,
        body: dict[str, Any],
        observed_request_id: str,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        """:raises urllib.error.HTTPError: Always, with structured evidence."""
        assert len(body) > 0
        assert observed_request_id == request_id
        assert timeout_seconds == 1.0
        raise urllib.error.HTTPError(
            url=base_url,
            code=500,
            msg="localization failed closed",
            hdrs=None,
            fp=BytesIO(json.dumps(error_response).encode()),
        )

    capture_dir = tmp_path / "captures"
    capture_dir.mkdir()
    monkeypatch.setattr(campaign, "_post_chat_completion", post)
    output = StringIO()
    outcome = campaign.run_campaign(
        _config(
            capture_dir=capture_dir,
            capture_request_id=request_id,
        ),
        (_payload(),),
        output,
    )

    row = json.loads(output.getvalue())
    capture_path = capture_dir / f"{request_id}-response.json"
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    assert outcome.valid is False
    assert row["selected_capture_path"] == str(capture_path)
    assert capture["http_status"] == 500
    assert capture["response"] == error_response


def test_request_failure_is_an_invalid_accounted_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def post(
        base_url: str,
        body: dict[str, Any],
        request_id: str,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        """:param base_url: Synthetic endpoint URL.
        :param body: Request body under test.
        :param request_id: External request identity under test.
        :param timeout_seconds: Synthetic request timeout.
        :raises OSError: Always, to model a failed request.
        """
        raise OSError("endpoint unavailable")

    monkeypatch.setattr(campaign, "_post_chat_completion", post)
    output = StringIO()
    outcome = campaign.run_campaign(_config(), (_payload(),), output)
    row = json.loads(output.getvalue())

    assert outcome.valid is False
    assert outcome.failed_parents == 1
    assert outcome.actual_choices == 0
    assert row["label"] == "request_error"
    assert row["request_id"].startswith("p2d-control-a1-")
    assert row["elapsed_seconds"] >= 0.0


def test_choice_count_or_identity_mismatch_invalidates_the_arm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def post(
        base_url: str,
        body: dict[str, Any],
        request_id: str,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        """:param base_url: Synthetic endpoint URL.
        :param body: Request body under test.
        :param request_id: External request identity under test.
        :param timeout_seconds: Synthetic request timeout.
        :returns: A deliberately under-counted response with a foreign ID.
        """
        return {"id": "chatcmpl-foreign", "choices": [_choice(0)], "usage": {}}

    monkeypatch.setattr(campaign, "_post_chat_completion", post)
    output = StringIO()
    outcome = campaign.run_campaign(
        _config(choices_per_parent=2),
        (_payload(),),
        output,
    )
    row = json.loads(output.getvalue())

    assert outcome.valid is False
    assert outcome.failed_parents == 1
    assert outcome.actual_choices == 1
    assert len(row["accounting_errors"]) == 3


def test_main_refuses_overwrite_before_issuing_requests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload_path = tmp_path / "payload.json"
    payload_path.write_text(json.dumps(_payload().body), encoding="utf-8")
    output_path = tmp_path / "results.jsonl"
    output_path.write_text("preserved\n", encoding="utf-8")

    def unexpected_post(*args: object) -> object:
        """:param args: Arguments that must never be supplied.
        :raises AssertionError: Always, because preflight must stop the arm.
        """
        raise AssertionError("request issued before overwrite refusal")

    monkeypatch.setattr(campaign, "_post_chat_completion", unexpected_post)
    result = campaign.main(
        [
            str(payload_path),
            "--base-url",
            "http://127.0.0.1:1/v1",
            "--campaign-id",
            "control-a1",
            "--samples",
            "1",
            "--out",
            str(output_path),
        ]
    )

    assert result == 2
    assert output_path.read_text(encoding="utf-8") == "preserved\n"


def test_main_atomically_publishes_invalid_evidence_and_returns_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload_path = tmp_path / "payload.json"
    payload_path.write_text(json.dumps(_payload().body), encoding="utf-8")
    output_path = tmp_path / "results.jsonl"

    def post(
        base_url: str,
        body: dict[str, Any],
        request_id: str,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        """:param base_url: Synthetic endpoint URL.
        :param body: Request body under test.
        :param request_id: External request identity under test.
        :param timeout_seconds: Synthetic request timeout.
        :returns: An empty response that cannot satisfy one parent.
        """
        return {"id": f"chatcmpl-{request_id}", "choices": [], "usage": {}}

    monkeypatch.setattr(campaign, "_post_chat_completion", post)
    result = campaign.main(
        [
            str(payload_path),
            "--base-url",
            "http://127.0.0.1:1/v1",
            "--campaign-id",
            "control-a1",
            "--samples",
            "1",
            "--out",
            str(output_path),
        ]
    )

    assert result == 1
    assert output_path.exists()
    assert json.loads(output_path.read_text(encoding="utf-8"))["label"] == (
        "request_error"
    )
    assert list(tmp_path.glob(".results.jsonl.*.tmp")) == []
