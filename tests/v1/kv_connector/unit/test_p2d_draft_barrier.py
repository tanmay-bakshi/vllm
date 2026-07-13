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
    """The source gate cannot run before a prior proposal stream is quiescent."""
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
            isinstance(child, ast.Attribute)
            and child.attr == "draft_propose_stream"
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
        isinstance(child, ast.Attribute)
        and child.attr == "_draft_propose_input_refs"
        for child in ast.walk(barrier)
    )


@pytest.mark.cpu_test
def test_pre_read_capture_follows_phase_entry_and_precedes_new_work() -> None:
    """PRE_READ follows phase entry and precedes transfer-state mutation."""
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
    phase_entry = start_load.body[1]
    pre_read_capture = start_load.body[2]

    assert isinstance(phase_entry, ast.Expr)
    assert _contains_call(phase_entry, "_begin_transfer_phase")
    assert isinstance(pre_read_capture, ast.Expr)
    assert _contains_call(pre_read_capture, "_localization_capture_pre_read")


@pytest.mark.cpu_test
def test_zero_byte_record_does_not_bypass_source_gate() -> None:
    """Default full-prefix handling records exclusion, gates, then releases P."""
    repository_root = Path(__file__).resolve().parents[4]
    worker_path = repository_root / (
        "vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_worker.py"
    )
    module = ast.parse(worker_path.read_text())
    read_blocks = next(
        node
        for node in ast.walk(module)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_read_blocks_for_req"
    )
    zero_index = next(
        index
        for index, statement in enumerate(read_blocks.body)
        if "NON_EVIDENTIARY_ZERO_BYTE" in ast.unparse(statement)
    )
    gate_index = next(
        index
        for index, statement in enumerate(read_blocks.body)
        if _contains_call(statement, "_localization_source_gate")
    )
    zero_branch = read_blocks.body[zero_index]

    assert zero_index < gate_index
    assert not any(isinstance(node, ast.Return) for node in ast.walk(zero_branch))
