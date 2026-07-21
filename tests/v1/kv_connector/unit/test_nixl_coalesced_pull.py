# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Behavioral tests for coalesced NIXL pull preparation."""

from typing import cast
from unittest.mock import MagicMock, call

import pytest

from vllm.distributed.kv_transfer.coalesced_layout import CoalescedTransferPlan
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    RemoteMeta,
    ReqMeta,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.packed_write_config import (
    PackedWriteConfig,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_worker import (
    NixlPullConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import (
    ReadSpec,
    TPMapping,
)
from vllm.distributed.kv_transfer.nixl_contracts import NixlRegionDescriptor
from vllm.distributed.kv_transfer.staging_ownership import (
    CoalescedStagingPlan,
    HandleState,
    StagingRangeAllocator,
)


def _region(base_address: int) -> NixlRegionDescriptor:
    """Build one synthetic dual-plane cache region.

    :param base_address: Registration base address.
    :returns: Complete region descriptor.
    """
    return NixlRegionDescriptor(
        semantic_name="layer0:transfer_region_0",
        group_indices=(0,),
        group_semantic_names=((0, "layer0:transfer_region_0"),),
        base_address=base_address,
        registered_bytes=64 * 8,
        row_bytes=8,
        shape=(64, 8),
        strides=(8, 1),
        dtype="uint8",
        element_size_bytes=1,
        layout="HND",
    )


def _disabled_packed_write_config() -> PackedWriteConfig:
    """Build the disabled adaptive transport configuration."""
    return PackedWriteConfig(
        enabled=False,
        chunk_bytes_per_rank=64 * 1024 * 1024,
        min_descriptors_per_rank=1,
        producer_slot_count=1,
        consumer_slot_count=1,
        alignment_bytes=256,
        warn_after_s=1.0,
        fail_after_s=2.0,
    )


def _enabled_packed_write_config(
    *,
    min_descriptors_per_rank: int,
) -> PackedWriteConfig:
    """Build an enabled adaptive transport configuration.

    :param min_descriptors_per_rank: Inclusive packed-WRITE selection threshold.
    :returns: Enabled validated configuration.
    """
    return PackedWriteConfig(
        enabled=True,
        chunk_bytes_per_rank=64 * 1024 * 1024,
        min_descriptors_per_rank=min_descriptors_per_rank,
        producer_slot_count=1,
        consumer_slot_count=1,
        alignment_bytes=256,
        warn_after_s=1.0,
        fail_after_s=2.0,
    )


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("descriptors_per_rank", "expected_mode", "expected_selected"),
    (
        (9, "direct", False),
        (10, "packed", True),
    ),
)
def test_adaptive_transport_selection_records_one_authoritative_event(
    caplog: pytest.LogCaptureFixture,
    descriptors_per_rank: int,
    expected_mode: str,
    expected_selected: bool,
) -> None:
    """The inclusive threshold drives one parseable route-selection event.

    :param caplog: Captured log records.
    :param descriptors_per_rank: Synthetic direct-plan descriptor cardinality.
    :param expected_mode: Expected structured route name.
    :param expected_selected: Whether packed WRITE should be selected.
    """
    worker = cast(
        NixlPullConnectorWorker,
        object.__new__(NixlPullConnectorWorker),
    )
    worker._packed_write_config = _enabled_packed_write_config(
        min_descriptors_per_rank=10
    )

    with caplog.at_level(
        "INFO",
        logger=("vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_worker"),
    ):
        selected = worker._select_packed_write_transport(
            direct_descriptors_per_rank=descriptors_per_rank,
            localization_enabled=False,
        )
        worker._record_transfer_plan_selection(
            request_id="consumer-request",
            use_packed_write=selected,
            direct_descriptors_per_rank=descriptors_per_rank,
            source_rank_count=4,
        )

    events = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("[transfer-plan-selected]")
    ]
    assert selected is expected_selected
    assert events == [
        "[transfer-plan-selected] request=consumer-request "
        f"mode={expected_mode} "
        f"direct_descriptors_per_rank={descriptors_per_rank} "
        "threshold_per_rank=10 "
        f"direct_descriptors_total={descriptors_per_rank * 4}"
    ]


@pytest.mark.cpu_test
def test_deferred_direct_selection_emits_once_before_prepare_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A deferred plan records its route only after successful reservation.

    :param caplog: Captured log records.
    """
    worker = cast(
        NixlPullConnectorWorker,
        object.__new__(NixlPullConnectorWorker),
    )
    remote_engine_id = "producer"
    remote_region = _region(0x200000)
    local_region = _region(0x100000)
    remote_info = MagicMock(
        remote_tp_size=1,
        remote_physical_blocks_per_logical=1,
    )
    worker.transfer_topo = MagicMock()
    worker.transfer_topo.get_engine_info.return_value = remote_info
    worker._localization_config = MagicMock()
    worker._localization_config.enabled_for.return_value = False
    worker._packed_write_config = _disabled_packed_write_config()
    worker._apply_prefix_caching = MagicMock(return_value=([[20]], [[10]]))
    worker._sp_group_flags = MagicMock(return_value=[False])
    worker._remote_layout = {remote_engine_id: {0: ([8], 64, 0)}}
    worker._remote_regions = {remote_engine_id: {0: (remote_region,)}}
    worker._region_descriptors = (local_region,)
    worker.tp_mappings = {
        remote_engine_id: TPMapping(
            source_ranks_per_group=((0,),),
            all_source_ranks=(0,),
            rank_to_attention_slot={0: 0},
            rank_offset_factor=0,
        )
    }
    worker.coalesce_staging_mb = 1
    worker._staging_buf = MagicMock()
    worker._staging_buf.data_ptr.return_value = 0x400000
    worker.device_id = 0
    worker.nixl_memory_type = "VRAM"
    worker.kv_caches_base_addr = {remote_engine_id: {0: [remote_region.base_address]}}
    worker._remote_agents = {remote_engine_id: {0: "remote-agent"}}
    worker.nixl_wrapper = MagicMock()
    worker.nixl_wrapper.get_xfer_descs.side_effect = RuntimeError(
        "descriptor construction failed"
    )
    worker._finish_quiescent_coalesced_failure = MagicMock()
    worker._notify_failed_coalesced_producer_ranks = MagicMock()
    worker._notify_non_read_producer_ranks = MagicMock()
    worker._prepare_coalesced_handle = MagicMock()

    allocator = StagingRangeAllocator(4096)
    captured_plans: list[CoalescedStagingPlan] = []
    create_attempts = 0

    def create_plan(
        req_id: str,
        engine_id: str,
        layout: CoalescedTransferPlan,
        layout_duration_seconds: float,
    ) -> CoalescedStagingPlan | None:
        nonlocal create_attempts
        create_attempts += 1
        if create_attempts == 1:
            return None
        plan = allocator.create_plan(
            owner_id="descriptor-test-owner",
            request_id=req_id,
            remote_engine_id=engine_id,
            layout=layout,
            layout_duration_seconds=layout_duration_seconds,
        )
        assert plan is not None
        captured_plans.append(plan)
        return plan

    worker._create_coalesced_plan = create_plan
    metadata = ReqMeta(
        local_block_ids=([20],),
        local_physical_block_ids=([20],),
        tp_size=1,
        remote=RemoteMeta(
            block_ids=([10],),
            host="127.0.0.1",
            port=1234,
            engine_id=remote_engine_id,
            request_id="producer-request",
        ),
    )
    read_specs = [
        ReadSpec(
            remote_rank=0,
            local_block_ids=([20],),
            remote_block_ids=([10],),
        )
    ]

    with caplog.at_level(
        "INFO",
        logger=("vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_worker"),
    ):
        deferred = worker._coalesced_read_request(
            "consumer-request",
            metadata,
            read_specs,
            b"notification",
        )
        result = worker._coalesced_read_request(
            "consumer-request",
            metadata,
            read_specs,
            b"notification",
        )

    events = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("[transfer-plan-selected]")
    ]
    assert deferred == "defer"
    assert result == "posted"
    assert events == [
        "[transfer-plan-selected] request=consumer-request mode=direct "
        "direct_descriptors_per_rank=1 threshold_per_rank=1 "
        "direct_descriptors_total=1"
    ]
    assert len(captured_plans) == 1
    plan = captured_plans[0]
    assert plan.slots[0].state is HandleState.PREPARE_FAILED
    assert plan.operation_failed
    assert plan.permanently_tombstoned is False
    assert plan.failure_reason is not None
    assert "descriptor construction failed" in plan.failure_reason
    worker._finish_quiescent_coalesced_failure.assert_called_once()
    worker._notify_failed_coalesced_producer_ranks.assert_called_once_with(
        metadata,
        plan,
        b"notification",
    )
    worker._prepare_coalesced_handle.assert_not_called()
    worker.nixl_wrapper.transfer.assert_not_called()


@pytest.mark.cpu_test
def test_descriptors_follow_rank_major_region_runs_exactly() -> None:
    """Map every canonical run to its exact rank slab and source interval."""
    worker = cast(
        NixlPullConnectorWorker,
        object.__new__(NixlPullConnectorWorker),
    )
    remote_engine_id = "producer"
    source_ranks = (3, 1, 0, 2)
    rank_slots = (2, 0, 3, 1)
    row_bytes = (8, 12)

    def region(
        *,
        region_index: int,
        group_index: int,
        base_address: int,
    ) -> NixlRegionDescriptor:
        width = row_bytes[region_index]
        return NixlRegionDescriptor(
            semantic_name=f"layer{group_index}:transfer_region_{region_index}",
            group_indices=(group_index,),
            group_semantic_names=(
                (
                    group_index,
                    f"layer{group_index}:transfer_region_{region_index}",
                ),
            ),
            base_address=base_address,
            registered_bytes=64 * width,
            row_bytes=width,
            shape=(64, width),
            strides=(width, 1),
            dtype="uint8",
            element_size_bytes=1,
            layout="HND",
        )

    remote_bases = {
        rank: (1_000_000 + rank * 100_000, 2_000_000 + rank * 100_000)
        for rank in range(4)
    }
    worker.transfer_topo = MagicMock()
    worker.transfer_topo.get_engine_info.return_value = MagicMock(
        remote_tp_size=4,
        remote_physical_blocks_per_logical=1,
    )
    worker._localization_config = MagicMock()
    worker._localization_config.enabled_for.return_value = False
    worker._packed_write_config = _disabled_packed_write_config()
    worker._apply_prefix_caching = MagicMock(
        return_value=(
            [[30, 31, 32], [40, 41]],
            [[5, 6, 8], [20, 21]],
        )
    )
    worker._sp_group_flags = MagicMock(return_value=[False, False])
    worker._remote_layout = {
        remote_engine_id: {rank: ([*row_bytes], 64, 10 + rank) for rank in range(4)}
    }
    worker._remote_regions = {
        remote_engine_id: {
            rank: (
                region(region_index=0, group_index=0, base_address=bases[0]),
                region(region_index=1, group_index=1, base_address=bases[1]),
            )
            for rank, bases in remote_bases.items()
        }
    }
    worker._region_descriptors = (
        region(region_index=0, group_index=0, base_address=3_000_000),
        region(region_index=1, group_index=1, base_address=4_000_000),
    )
    worker.tp_mappings = {
        remote_engine_id: TPMapping(
            source_ranks_per_group=(source_ranks, source_ranks),
            all_source_ranks=source_ranks,
            rank_to_attention_slot=dict(zip(source_ranks, rank_slots, strict=True)),
            rank_offset_factor=0,
        )
    }
    worker.coalesce_staging_mb = 1
    worker._staging_buf = MagicMock()
    worker._staging_buf.data_ptr.return_value = 300_000
    worker.device_id = 7
    worker.nixl_memory_type = "VRAM"
    worker.kv_caches_base_addr = {
        remote_engine_id: {rank: list(bases) for rank, bases in remote_bases.items()}
    }
    worker._remote_agents = {
        remote_engine_id: {rank: f"remote-agent-{rank}" for rank in range(4)}
    }
    worker.nixl_wrapper = MagicMock()
    worker.nixl_wrapper.get_xfer_descs.side_effect = lambda descriptors, memory_type: (
        tuple(descriptors)
    )
    worker._assert_transfer_phase_active = MagicMock()
    worker._post_prepared_coalesced = MagicMock()
    worker._notify_non_read_producer_ranks = MagicMock()

    allocator = StagingRangeAllocator(4096)
    captured_plans: list[CoalescedStagingPlan] = []

    def create_plan(
        req_id: str,
        engine_id: str,
        layout: CoalescedTransferPlan,
        layout_duration_seconds: float,
    ) -> CoalescedStagingPlan | None:
        plan = allocator.create_plan(
            owner_id="descriptor-address-owner",
            request_id=req_id,
            remote_engine_id=engine_id,
            layout=layout,
            layout_duration_seconds=layout_duration_seconds,
        )
        assert plan is not None
        captured_plans.append(plan)
        return plan

    def prepare(
        plan: CoalescedStagingPlan,
        source_rank: int,
        _local_descs: object,
        _remote_descs: object,
        _remote_agent: str,
        _notification_id: bytes,
    ) -> bool:
        plan.attach_handle(source_rank, f"handle-{source_rank}")
        return True

    worker._create_coalesced_plan = create_plan
    worker._prepare_coalesced_handle = prepare
    metadata = ReqMeta(
        local_block_ids=([30, 31, 32], [40, 41]),
        local_physical_block_ids=([30, 31, 32], [40, 41]),
        tp_size=1,
        remote=RemoteMeta(
            block_ids=([5, 6, 8], [20, 21]),
            host="127.0.0.1",
            port=1234,
            engine_id=remote_engine_id,
            request_id="producer-request",
        ),
    )
    read_specs = [
        ReadSpec(
            remote_rank=rank,
            local_block_ids=([30, 31, 32], [40, 41]),
            remote_block_ids=([5, 6, 8], [20, 21]),
        )
        for rank in source_ranks
    ]

    result = worker._coalesced_read_request(
        "consumer-request",
        metadata,
        read_specs,
        b"notification",
    )

    assert result == "posted"
    assert len(captured_plans) == 1
    layout = captured_plans[0].layout
    assert layout.rank_stride_bytes == 48
    assert layout.staging_size_bytes == 192
    expected_calls: list[tuple[list[tuple[int, int, int]], str]] = []
    for rank_index, source_rank in enumerate(source_ranks):
        staging_rank_base = 300_000 + rank_index * 48
        expected_calls.extend(
            (
                (
                    [
                        (staging_rank_base, 16, 7),
                        (staging_rank_base + 16, 8, 7),
                        (staging_rank_base + 24, 24, 7),
                    ],
                    "VRAM",
                ),
                (
                    [
                        (remote_bases[source_rank][0] + 5 * 8, 16, 10 + source_rank),
                        (remote_bases[source_rank][0] + 8 * 8, 8, 10 + source_rank),
                        (
                            remote_bases[source_rank][1] + 20 * 12,
                            24,
                            10 + source_rank,
                        ),
                    ],
                    "VRAM",
                ),
            )
        )
    actual_calls = [
        descriptor_call.args
        for descriptor_call in worker.nixl_wrapper.get_xfer_descs.call_args_list
    ]
    assert actual_calls == [
        (descriptors, memory_type) for descriptors, memory_type in expected_calls
    ]
    assert worker._post_prepared_coalesced.call_args_list == [
        call(captured_plans[0], rank) for rank in source_ranks
    ]
