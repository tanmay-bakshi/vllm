# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Focused tests for the production NIXL region handshake contract."""

import threading
from collections import defaultdict
from dataclasses import replace
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import msgspec
import pytest

from vllm.distributed.kv_transfer.kv_connector.utils import (
    EngineTransferInfo,
    TransferTopology,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import (
    NixlBaseConnectorWorker,
    NixlHandshakeFailStopError,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlAgentMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import (
    TPMapping,
    compute_tp_mapping,
)
from vllm.distributed.kv_transfer.nixl_contracts import NixlRegionDescriptor
from vllm.v1.kv_cache_interface import FullAttentionSpec

REMOTE_ENGINE_ID = "remote-engine"


class _FakeAttentionBackend:
    """Minimal backend geometry needed by :class:`TransferTopology`."""

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[int, ...]:
        """Return a K/V-first synthetic cache shape."""
        return (2, num_blocks, num_kv_heads, block_size, head_size)


def _regions(
    row_bytes: int,
    num_blocks: int,
    base_address: int,
) -> tuple[NixlRegionDescriptor, ...]:
    """Build one region that owns both synthetic cache groups."""
    return (
        NixlRegionDescriptor(
            semantic_name="layer0:transfer_region_0",
            group_indices=(0, 1),
            group_semantic_names=(
                (0, "layer0:transfer_region_0"),
                (1, "layer1:transfer_region_0"),
            ),
            base_address=base_address,
            registered_bytes=row_bytes * num_blocks,
            row_bytes=row_bytes,
            shape=(num_blocks, row_bytes),
            strides=(row_bytes, 1),
            dtype="uint8",
            element_size_bytes=1,
            layout="HND",
        ),
    )


def _worker() -> NixlBaseConnectorWorker:
    """Build only the state consumed by handshake validation."""
    worker = cast(
        NixlBaseConnectorWorker,
        object.__new__(NixlBaseConnectorWorker),
    )
    worker.engine_id = "local-engine"
    worker.tp_rank = 0
    worker.block_size = 16
    worker.num_blocks = 2
    worker.block_len_per_layer = [1024]
    worker._region_descriptors = _regions(1024, worker.num_blocks, 0x100000)
    worker._region_is_mla = [False]
    worker._has_mamba = False
    worker.use_mla = False
    worker.use_host_buffer = False
    worker.kv_cache_layout = "HND"
    worker.host_buffer_kv_cache_layout = "HND"
    worker.backend_name = "FLASH_ATTN"
    worker.enable_permute_local_kv = False
    worker.enable_heterogeneous_attn_post_process = False
    worker._is_hma_required = False
    worker._physical_blocks_per_logical_kv_block = 1
    worker._sp_flags_cache = None
    worker.kv_transfer_config = SimpleNamespace(enable_permute_local_kv=False)
    worker.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(enable_prefix_caching=False)
    )
    groups = [
        SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=16, kv_planes=2))
        for _ in range(2)
    ]
    worker.kv_cache_config = SimpleNamespace(kv_cache_groups=groups)
    worker._group_spec_types = (FullAttentionSpec, FullAttentionSpec)
    worker.transfer_topo = TransferTopology(
        tp_rank=0,
        tp_size=1,
        block_size=16,
        engine_id=worker.engine_id,
        is_mla=False,
        is_mamba=False,
        total_num_kv_heads=8,
        attn_backends=[_FakeAttentionBackend],
        tensor_shape=None,
    )
    worker.dst_num_blocks = {worker.engine_id: worker.num_blocks}
    worker._remote_agents = defaultdict(dict)
    worker._remote_regions = defaultdict(dict)
    worker._remote_layout = defaultdict(dict)
    worker._remote_source_semantics = defaultdict(dict)
    worker._remote_rank_contracts = defaultdict(dict)
    worker.tp_mappings = {}
    worker.nixl_wrapper = MagicMock()
    worker._handshake_lock = threading.RLock()
    worker._handshake_active_engine_id = None
    worker._handshake_mutation_engine_id = None
    worker._handshake_fail_stop_reason = None
    worker._handshake_futures = {}
    worker._engine_last_active = {}
    worker._engine_ttl = 0.0
    return worker


def _metadata(
    row_bytes: int = 1024,
    num_blocks: int = 3,
    base_address: int = 0x200000,
    device_id: int = 0,
) -> NixlAgentMetadata:
    """Build a complete valid remote handshake."""
    regions = _regions(row_bytes, num_blocks, base_address)
    return NixlAgentMetadata(
        engine_id=REMOTE_ENGINE_ID,
        tp_rank=device_id,
        agent_metadata=b"agent-metadata",
        kv_caches_base_addr=[base_address],
        device_id=device_id,
        num_blocks=num_blocks,
        block_lens=[row_bytes],
        kv_cache_layout="HND",
        block_size=16,
        ssm_sizes=(0, 0),
        attn_backend_name="FLASH_ATTN",
        physical_blocks_per_logical_kv_block=1,
        registration_generation=f"rank-{device_id}-generation",
        regions=regions,
        source_group_planes=(2, 2),
        physical_group_token_capacities=(16, 16),
    )


def _plan(worker: NixlBaseConnectorWorker, remote_tp_size: int) -> TPMapping:
    """Derive the production TP mapping for the synthetic worker."""
    assert worker.transfer_topo is not None
    return compute_tp_mapping(
        worker.transfer_topo,
        remote_tp_size,
        worker._group_spec_types,
    )


@pytest.mark.parametrize(
    ("mutation", "error_match"),
    [
        ("missing_regions", "region descriptor list is empty"),
        ("unordered_owners", "owners must be sorted and unique"),
        ("incomplete_coverage", "do not cover every cache group"),
        ("semantic_identity", "ordered semantic identity differs"),
        ("row_geometry", "does not match block_lens"),
        ("registration_bounds", "exceeds the uint64 address space"),
        ("plane_geometry", "plane semantics differ"),
        ("capacity_geometry", "token capacity does not scale"),
    ],
)
def test_malformed_contract_is_rejected_before_nixl_import(
    mutation: str,
    error_match: str,
) -> None:
    worker = _worker()
    metadata = _metadata()
    region = metadata.regions[0]

    if mutation == "missing_regions":
        metadata = replace(metadata, regions=())
    elif mutation == "unordered_owners":
        bad_region = msgspec.structs.replace(
            region,
            group_indices=(1, 0),
            group_semantic_names=((1, "group_1"), (0, "group_0")),
        )
        metadata = replace(metadata, regions=(bad_region,))
    elif mutation == "incomplete_coverage":
        bad_region = msgspec.structs.replace(
            region,
            group_indices=(0,),
            group_semantic_names=((0, "group_0"),),
        )
        metadata = replace(metadata, regions=(bad_region,))
    elif mutation == "semantic_identity":
        bad_region = msgspec.structs.replace(
            region,
            semantic_name="wrong_region",
        )
        metadata = replace(metadata, regions=(bad_region,))
    elif mutation == "row_geometry":
        bad_region = msgspec.structs.replace(
            region,
            row_bytes=region.row_bytes + 1,
        )
        metadata = replace(metadata, regions=(bad_region,))
    elif mutation == "registration_bounds":
        bad_region = msgspec.structs.replace(
            region,
            base_address=(1 << 64) - region.registered_bytes + 1,
        )
        metadata = replace(
            metadata,
            kv_caches_base_addr=[bad_region.base_address],
            regions=(bad_region,),
        )
    elif mutation == "plane_geometry":
        metadata = replace(metadata, source_group_planes=(1, 2))
    elif mutation == "capacity_geometry":
        metadata = replace(
            metadata,
            physical_group_token_capacities=(17, 16),
        )
    else:
        raise AssertionError(f"unhandled mutation {mutation}")

    with pytest.raises(RuntimeError, match=error_match):
        worker.add_remote_agent(metadata)

    worker.nixl_wrapper.add_remote_agent.assert_not_called()
    assert REMOTE_ENGINE_ID not in worker.dst_num_blocks
    assert REMOTE_ENGINE_ID not in worker._remote_regions
    assert worker.transfer_topo is not None
    with pytest.raises(KeyError):
        worker.transfer_topo.get_engine_info(REMOTE_ENGINE_ID)


def test_valid_contract_and_tp_mapping_are_accepted() -> None:
    worker = _worker()
    metadata = _metadata()

    contract = worker._validate_remote_agent_handshake(
        metadata,
        remote_tp_rank=0,
        remote_tp_size=1,
        plan=_plan(worker, 1),
    )
    assert contract.enable_permute_local_kv is False
    assert contract.enable_heterogeneous_attn_post_process is False


def test_metadata_rank_mismatch_is_rejected_before_nixl_import() -> None:
    worker = _worker()
    metadata = _metadata(device_id=1)

    with pytest.raises(RuntimeError, match="metadata rank does not match"):
        worker.add_remote_agent(
            metadata,
            remote_tp_rank=0,
            remote_tp_size=2,
        )

    worker.nixl_wrapper.add_remote_agent.assert_not_called()
    assert REMOTE_ENGINE_ID not in worker.dst_num_blocks
    assert REMOTE_ENGINE_ID not in worker._remote_regions


def test_failed_validation_does_not_mutate_postprocess_state() -> None:
    worker = _worker()
    worker.kv_cache_layout = "NHD"
    worker._region_descriptors = (
        msgspec.structs.replace(worker._region_descriptors[0], layout="NHD"),
    )
    worker.kv_transfer_config = SimpleNamespace(enable_permute_local_kv=True)
    metadata = replace(
        _metadata(),
        physical_group_token_capacities=(17, 16),
    )

    with pytest.raises(RuntimeError, match="token capacity does not scale"):
        worker._validate_remote_agent_handshake(
            metadata,
            remote_tp_rank=0,
            remote_tp_size=1,
            plan=_plan(worker, 1),
        )

    assert worker.enable_permute_local_kv is False
    assert worker.enable_heterogeneous_attn_post_process is False


def test_remote_engines_cannot_require_different_postprocess_contracts() -> None:
    worker = _worker()
    worker.kv_cache_layout = "NHD"
    worker._region_descriptors = (
        msgspec.structs.replace(worker._region_descriptors[0], layout="NHD"),
    )
    worker.kv_transfer_config = SimpleNamespace(enable_permute_local_kv=True)
    worker._remote_agents["existing-engine"][0] = "existing-agent"

    with pytest.raises(RuntimeError, match="incompatible destination post-processing"):
        worker._validate_remote_agent_handshake(
            _metadata(),
            remote_tp_rank=0,
            remote_tp_size=1,
            plan=_plan(worker, 1),
        )


def test_attention_rank_slots_are_validated() -> None:
    worker = _worker()
    metadata = _metadata()
    bad_plan = TPMapping(
        source_ranks_per_group=((0,), (0,)),
        all_source_ranks=(0,),
        rank_to_attention_slot={0: 1},
        rank_offset_factor=0,
    )

    with pytest.raises(RuntimeError, match="rank slots"):
        worker._validate_remote_agent_handshake(
            metadata,
            remote_tp_rank=0,
            remote_tp_size=1,
            plan=bad_plan,
        )


def test_cross_prefill_rank_geometry_must_agree() -> None:
    worker = _worker()
    remote_tp_size = 2
    rank_zero = _metadata(row_bytes=512, num_blocks=3, device_id=0)
    plan = _plan(worker, remote_tp_size)
    worker._validate_remote_agent_handshake(
        rank_zero,
        remote_tp_rank=0,
        remote_tp_size=remote_tp_size,
        plan=plan,
    )

    assert worker.transfer_topo is not None
    worker.transfer_topo.register_remote_engine(
        REMOTE_ENGINE_ID,
        EngineTransferInfo(
            remote_tp_size=remote_tp_size,
            remote_block_size=rank_zero.block_size,
            remote_block_len=rank_zero.block_lens[0],
            remote_physical_blocks_per_logical=(
                rank_zero.physical_blocks_per_logical_kv_block
            ),
        ),
    )
    worker.dst_num_blocks[REMOTE_ENGINE_ID] = rank_zero.num_blocks
    worker._remote_regions[REMOTE_ENGINE_ID][0] = rank_zero.regions
    worker._remote_layout[REMOTE_ENGINE_ID][0] = (
        list(rank_zero.block_lens),
        rank_zero.num_blocks,
        rank_zero.device_id,
    )
    worker._remote_source_semantics[REMOTE_ENGINE_ID][0] = (
        rank_zero.source_group_planes,
        rank_zero.physical_group_token_capacities,
    )

    rank_one = _metadata(row_bytes=512, num_blocks=4, device_id=1)
    with pytest.raises(RuntimeError, match="disagree on num_blocks"):
        worker._validate_remote_agent_handshake(
            rank_one,
            remote_tp_rank=1,
            remote_tp_size=remote_tp_size,
            plan=plan,
        )


@pytest.mark.parametrize(
    ("mutation", "expected_field"),
    [
        ({"attn_backend_name": "OTHER_ATTN"}, "attn_backend_name"),
        ({"ssm_sizes": (1, 0)}, "ssm_sizes"),
        (
            {"physical_blocks_per_logical_kv_block": 2},
            "physical_blocks_per_logical_kv_block",
        ),
    ],
)
def test_multi_rank_roster_disagreement_fails_before_import(
    mutation: dict[str, object],
    expected_field: str,
) -> None:
    worker = _worker()
    remote_tp_size = 2
    plan = _plan(worker, remote_tp_size)
    rank_zero = _metadata(row_bytes=512, device_id=0)
    rank_one = replace(
        _metadata(row_bytes=512, base_address=0x300000, device_id=1),
        **mutation,
    )

    with pytest.raises(RuntimeError, match=expected_field):
        worker._validate_remote_handshake_roster(
            {0: rank_zero, 1: rank_one},
            remote_tp_size,
            plan,
        )

    worker.nixl_wrapper.add_remote_agent.assert_not_called()
    assert worker.transfer_topo is not None
    with pytest.raises(KeyError):
        worker.transfer_topo.get_engine_info(REMOTE_ENGINE_ID)


def test_multi_rank_roster_rejects_per_rank_layout_differences() -> None:
    worker = _worker()
    worker.use_mla = True
    worker.kv_cache_layout = "NHD"
    worker._region_descriptors = (
        msgspec.structs.replace(worker._region_descriptors[0], layout="NHD"),
    )
    worker.kv_transfer_config = SimpleNamespace(enable_permute_local_kv=True)
    remote_tp_size = 2
    plan = _plan(worker, remote_tp_size)
    rank_zero = _metadata(row_bytes=1024, device_id=0)
    rank_one = _metadata(row_bytes=1024, base_address=0x300000, device_id=1)
    rank_one = replace(
        rank_one,
        kv_cache_layout="NHD",
        regions=tuple(
            msgspec.structs.replace(region, layout="NHD") for region in rank_one.regions
        ),
    )

    with pytest.raises(RuntimeError, match="kv_cache_layout"):
        worker._validate_remote_handshake_roster(
            {0: rank_zero, 1: rank_one},
            remote_tp_size,
            plan,
        )

    assert worker.enable_permute_local_kv is False
    worker.nixl_wrapper.add_remote_agent.assert_not_called()


def test_multi_rank_roster_rejects_per_rank_postprocess_differences() -> None:
    worker = _worker()
    worker.backend_name = "CPU_ATTN"
    remote_tp_size = 2
    plan = _plan(worker, remote_tp_size)
    rank_zero = _metadata(row_bytes=512, device_id=0)
    rank_one = replace(
        _metadata(row_bytes=512, base_address=0x300000, device_id=1),
        attn_backend_name="CPU_ATTN",
    )

    with pytest.raises(RuntimeError, match="attn_backend_name"):
        worker._validate_remote_handshake_roster(
            {0: rank_zero, 1: rank_one},
            remote_tp_size,
            plan,
        )

    assert worker.enable_heterogeneous_attn_post_process is False
    worker.nixl_wrapper.add_remote_agent.assert_not_called()


def test_rank_contract_binds_all_cross_rank_geometry() -> None:
    worker = _worker()
    reference = _metadata()
    current = replace(
        reference,
        attn_backend_name="OTHER_ATTN",
        ssm_sizes=(1, 2),
        kv_cache_layout="NHD",
        block_size=8,
        physical_blocks_per_logical_kv_block=2,
        block_lens=[512],
        num_blocks=4,
        source_group_planes=(1, 2),
        physical_group_token_capacities=(8, 16),
        regions=tuple(
            msgspec.structs.replace(
                region,
                registered_bytes=512 * 4,
                row_bytes=512,
                shape=(4, 512),
                strides=(512, 1),
                layout="NHD",
            )
            for region in reference.regions
        ),
    )

    differences = worker._rank_handshake_contract_differences(
        worker._rank_handshake_contract(reference),
        worker._rank_handshake_contract(current),
    )

    assert differences == (
        "attn_backend_name",
        "ssm_sizes",
        "kv_cache_layout",
        "block_size",
        "physical_blocks_per_logical_kv_block",
        "block_lens",
        "num_blocks",
        "source_group_planes",
        "physical_group_token_capacities",
        "region_geometry",
    )


def test_remote_import_without_handshake_owner_is_rejected_before_mutation() -> None:
    worker = _worker()

    with pytest.raises(RuntimeError, match="requires an active handshake transaction"):
        worker.add_remote_agent(_metadata())

    worker.nixl_wrapper.add_remote_agent.assert_not_called()
    assert worker.transfer_topo is not None
    with pytest.raises(KeyError):
        worker.transfer_topo.get_engine_info(REMOTE_ENGINE_ID)


def test_post_mutation_handshake_failure_permanently_forbids_retry() -> None:
    worker = _worker()
    worker._handshake_initiation_executor = MagicMock()
    metadata = _metadata()
    worker.nixl_wrapper.add_remote_agent.side_effect = RuntimeError(
        "native agent import failed"
    )
    worker._perform_nixl_handshake = MagicMock(
        side_effect=lambda *_args: worker.add_remote_agent(metadata)
    )

    with pytest.raises(
        NixlHandshakeFailStopError,
        match="partially imported remote state",
    ):
        worker._nixl_handshake("127.0.0.1", 1234, 1, REMOTE_ENGINE_ID)

    assert worker.transfer_topo is not None
    assert worker.transfer_topo.get_engine_info(REMOTE_ENGINE_ID).remote_tp_size == 1
    assert worker._handshake_mutation_engine_id == REMOTE_ENGINE_ID
    assert worker._handshake_fail_stop_reason is not None
    assert worker._remote_agents[REMOTE_ENGINE_ID] == {}

    with pytest.raises(NixlHandshakeFailStopError, match="in-process reuse"):
        worker._ensure_handshake(REMOTE_ENGINE_ID, "127.0.0.1", 1234, 1)
    with pytest.raises(NixlHandshakeFailStopError, match="in-process reuse"):
        worker._nixl_handshake("127.0.0.1", 1234, 1, REMOTE_ENGINE_ID)
    with pytest.raises(NixlHandshakeFailStopError, match="in-process reuse"):
        worker.get_finished()

    worker._handshake_initiation_executor.submit.assert_not_called()
    worker._perform_nixl_handshake.assert_called_once()


def test_later_rank_failure_after_first_import_is_also_fail_stop() -> None:
    worker = _worker()
    worker._handshake_initiation_executor = MagicMock()

    def fail_after_first_rank(*_args: object) -> dict[int, str]:
        worker._mark_handshake_mutation(REMOTE_ENGINE_ID)
        worker.dst_num_blocks[REMOTE_ENGINE_ID] = 3
        raise RuntimeError("rank one contract failed")

    worker._perform_nixl_handshake = MagicMock(side_effect=fail_after_first_rank)

    with pytest.raises(
        NixlHandshakeFailStopError,
        match="failed after its first remote-state mutation",
    ):
        worker._nixl_handshake("127.0.0.1", 1234, 2, REMOTE_ENGINE_ID)

    assert worker.dst_num_blocks[REMOTE_ENGINE_ID] == 3
    with pytest.raises(NixlHandshakeFailStopError, match="in-process reuse"):
        worker._ensure_handshake(REMOTE_ENGINE_ID, "127.0.0.1", 1234, 2)
    worker._handshake_initiation_executor.submit.assert_not_called()


def test_pre_mutation_handshake_failure_remains_retriable() -> None:
    worker = _worker()
    worker._perform_nixl_handshake = MagicMock(
        side_effect=[
            RuntimeError("remote contract validation failed"),
            {0: "remote-agent"},
        ]
    )

    with pytest.raises(RuntimeError, match="contract validation failed"):
        worker._nixl_handshake("127.0.0.1", 1234, 1, REMOTE_ENGINE_ID)

    assert worker._handshake_fail_stop_reason is None
    assert worker._handshake_active_engine_id is None
    assert worker._handshake_mutation_engine_id is None
    assert worker._nixl_handshake(
        "127.0.0.1",
        1234,
        1,
        REMOTE_ENGINE_ID,
    ) == {0: "remote-agent"}
