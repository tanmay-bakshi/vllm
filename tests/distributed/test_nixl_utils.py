# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.distributed.nixl_utils import canonicalize_nixl_agent_name


@pytest.mark.parametrize(
    ("native_name", "expected"),
    [
        ("decoder-agent", "decoder-agent"),
        (b"decoder-agent", "decoder-agent"),
    ],
)
def test_canonicalize_nixl_agent_name(
    native_name: str | bytes,
    expected: str,
) -> None:
    assert canonicalize_nixl_agent_name(native_name) == expected


@pytest.mark.parametrize("native_name", ["", b""])
def test_canonicalize_nixl_agent_name_rejects_empty(
    native_name: str | bytes,
) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        canonicalize_nixl_agent_name(native_name)


def test_canonicalize_nixl_agent_name_rejects_non_utf8() -> None:
    with pytest.raises(ValueError, match="valid UTF-8"):
        canonicalize_nixl_agent_name(b"\xff")


def test_canonicalize_nixl_agent_name_rejects_other_types() -> None:
    with pytest.raises(TypeError, match="bytes or text"):
        canonicalize_nixl_agent_name(1)  # type: ignore[arg-type]
