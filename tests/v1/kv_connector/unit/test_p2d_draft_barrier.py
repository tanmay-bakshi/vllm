# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Static ordering guard for the P-side asynchronous draft writer."""

import ast
from pathlib import Path

import pytest


def _contains_call(node: ast.AST, name: str) -> bool:
    """Return whether an AST node contains a call with the requested name.

    :param node: Syntax subtree.
    :param name: Function or method name.
    :returns: Whether the call exists in the subtree.
    """
    return any(
        isinstance(child, ast.Call)
        and (
            isinstance(child.func, ast.Name)
            and child.func.id == name
            or isinstance(child.func, ast.Attribute)
            and child.func.attr == name
        )
        for child in ast.walk(node)
    )


@pytest.mark.cpu_test
def test_delayed_draft_proposal_is_joined_before_every_connector_path() -> None:
    """KV transfer cannot observe a still-running proposal stream."""
    repository_root = Path(__file__).resolve().parents[4]
    runner_path = repository_root / "vllm/v1/worker/gpu_model_runner.py"
    module = ast.parse(runner_path.read_text())
    execute_model = next(
        node
        for node in ast.walk(module)
        if isinstance(node, ast.FunctionDef) and node.name == "execute_model"
    )
    barrier_index = next(
        index
        for index, statement in enumerate(execute_model.body)
        if _contains_call(statement, "wait_stream")
        and any(
            isinstance(child, ast.Attribute) and child.attr == "draft_propose_stream"
            for child in ast.walk(statement)
        )
    )
    connector_indices = [
        index
        for index, statement in enumerate(execute_model.body)
        if any(
            _contains_call(statement, call_name)
            for call_name in (
                "has_kv_transfer_group",
                "_update_states",
                "kv_connector_no_forward",
            )
        )
    ]

    assert len(connector_indices) > 0
    assert barrier_index < min(connector_indices)
    barrier = execute_model.body[barrier_index]
    assert any(
        isinstance(child, ast.Attribute) and child.attr == "_draft_propose_input_refs"
        for child in ast.walk(barrier)
    )


@pytest.mark.cpu_test
def test_pre_read_capture_follows_transfer_drain_and_precedes_model_return() -> None:
    """PRE_READ observes the final destination state before model execution."""
    repository_root = Path(__file__).resolve().parents[4]
    worker_path = repository_root / (
        "vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_worker.py"
    )
    module = ast.parse(worker_path.read_text())
    start_load = next(
        node
        for node in ast.walk(module)
        if isinstance(node, ast.FunctionDef) and node.name == "start_load_kv"
    )
    drain_index = next(
        index
        for index, statement in enumerate(start_load.body)
        if _contains_call(statement, "_drain_transfer_phase")
    )
    pre_read_index = next(
        index
        for index, statement in enumerate(start_load.body)
        if _contains_call(statement, "_localization_capture_pre_read")
    )
    boundary_index = next(
        index
        for index, statement in enumerate(start_load.body)
        if _contains_call(statement, "_record_transfer_decode_boundary")
    )
    pre_read = start_load.body[pre_read_index]

    assert drain_index < pre_read_index < boundary_index
    assert any(
        isinstance(child, ast.Attribute) and child.attr == "scheduled_request_ids"
        for child in ast.walk(pre_read)
    )


@pytest.mark.cpu_test
def test_targeted_read_path_has_no_pre_transfer_source_gate() -> None:
    """Targeted reads post from structural contracts without a source gate."""
    repository_root = Path(__file__).resolve().parents[4]
    worker_path = repository_root / (
        "vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_worker.py"
    )
    module = ast.parse(worker_path.read_text())
    read_blocks = next(
        node
        for node in ast.walk(module)
        if isinstance(node, ast.FunctionDef) and node.name == "_read_blocks_for_req"
    )
    coalesced_read = next(
        node
        for node in ast.walk(module)
        if isinstance(node, ast.FunctionDef) and node.name == "_coalesced_read_request"
    )
    contract_call = next(
        node
        for node in ast.walk(read_blocks)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_localization_build_source_contracts"
    )
    dispatch_call = next(
        node
        for node in ast.walk(read_blocks)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_coalesced_read_request"
        and len(node.args) == 4
    )

    assert not _contains_call(read_blocks, "_localization_source_gate")
    assert not any(
        isinstance(node, ast.FunctionDef) and node.name == "_localization_source_gate"
        for node in ast.walk(module)
    )
    assert contract_call.lineno < dispatch_call.lineno
    assert _contains_call(coalesced_read, "_initialize_and_post_coalesced")
