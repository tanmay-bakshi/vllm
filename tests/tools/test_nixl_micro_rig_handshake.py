"""Connector-v9 semantic evidence tests for the NIXL micro-rig."""

import hashlib
import json
import threading
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import pytest

from tools.gemma4_pd.nixl_micro_rig.config import ConfigError, load_config
from tools.gemma4_pd.nixl_micro_rig.handshake import (
    SEMANTIC_HANDSHAKE_CONNECTOR_VERSION,
    STATIC_SEMANTIC_EVIDENCE_SCOPE,
    load_semantic_handshake_profile,
)
from vllm.distributed.kv_transfer.kv_connector.utils import TransferTopology
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import (
    NixlBaseConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NIXL_CONNECTOR_VERSION,
    NixlAgentMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import (
    compute_tp_mapping,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec

FIXTURE_DIRECTORY = Path(__file__).parents[2] / "tools" / "gemma4_pd" / "nixl_micro_rig"
CONFIG_PATH = FIXTURE_DIRECTORY / "gemma4_tp4_to_tp1_2k.json"


class _FakeAttentionBackend:
    """Supply only the cache shape needed by :class:`TransferTopology`."""

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[int, ...]:
        """Return one synthetic K/V-first cache shape.

        :param num_blocks: Registered block count.
        :param block_size: Tokens per block.
        :param num_kv_heads: Local KV-head count.
        :param head_size: Elements per KV head.
        :returns: Synthetic cache shape.
        """
        return (2, num_blocks, num_kv_heads, block_size, head_size)


def _copy_legacy_captures(destination: Path) -> None:
    """Copy the immutable legacy evidence required by a temporary config.

    :param destination: Temporary configuration directory.
    """
    for name in ("storm37b-p-handshake.json", "storm37b-d1-handshake.json"):
        (destination / name).write_bytes((FIXTURE_DIRECTORY / name).read_bytes())


def _write_mutated_fixture(
    destination: Path,
    mutation: str,
) -> Path:
    """Write one independently authenticated semantic negative control.

    :param destination: Temporary configuration directory.
    :param mutation: Semantic field to corrupt.
    :returns: Temporary configuration path.
    """
    config_value = json.loads(CONFIG_PATH.read_text())
    fixture_path = FIXTURE_DIRECTORY / config_value["semantic_handshake_manifest"]
    fixture_value = json.loads(fixture_path.read_text())
    profile = next(
        item
        for item in fixture_value["profiles"]
        if item["name"] == config_value["semantic_handshake_profile"]
    )
    if mutation == "connector_version":
        profile["connector_version"] += 1
    elif mutation == "group_semantic_names":
        profile["group_semantic_names"][0] = "wrong_group"
    elif mutation == "source_group_planes":
        profile["source_group_planes"][0] = 1
    elif mutation == "physical_group_token_capacities":
        profile["physical_group_token_capacities"][0] += 1
    elif mutation == "region_group_indices":
        profile["region_group_indices"][0] = profile["region_group_indices"][0][1:]
    elif mutation == "runtime_capture_authenticated":
        profile["runtime_capture_authenticated"] = True
    elif mutation == "rank_identity_field":
        profile["rank_identity_field"] = "device_id"
    elif mutation == "descriptor_element_size_bytes":
        profile["descriptor_element_size_bytes"] = 3
    else:
        raise AssertionError(f"unhandled mutation {mutation}")

    fixture_payload = json.dumps(fixture_value, indent=2).encode()
    temporary_fixture = destination / fixture_path.name
    temporary_fixture.write_bytes(fixture_payload)
    config_value["semantic_handshake_sha256"] = hashlib.sha256(
        fixture_payload
    ).hexdigest()
    config_path = destination / CONFIG_PATH.name
    config_path.write_text(json.dumps(config_value))
    _copy_legacy_captures(destination)
    return config_path


def _fixture_worker() -> NixlBaseConnectorWorker:
    """Build only the local state consumed by production handshake validation.

    :returns: Minimal TP1 decoder-side worker.
    """
    config = load_config(CONFIG_PATH)
    profile = load_semantic_handshake_profile(
        FIXTURE_DIRECTORY / config.semantic_handshake_manifest,
        config.semantic_handshake_sha256,
        config.semantic_handshake_profile,
    )
    local_row_bytes = tuple(
        region.row_bytes * len(config.producer_devices) for region in config.regions
    )
    worker = cast(
        NixlBaseConnectorWorker,
        object.__new__(NixlBaseConnectorWorker),
    )
    worker.engine_id = "fixture-decoder"
    worker.tp_rank = 0
    worker.block_size = 16
    worker.num_blocks = config.source_block_count
    worker.block_len_per_layer = list(local_row_bytes)
    worker._region_descriptors = profile.region_descriptors(
        num_blocks=config.source_block_count,
        row_bytes=local_row_bytes,
        base_address=0x10000000000,
    )
    worker._region_is_mla = [False] * len(config.regions)
    worker._has_mamba = False
    worker.use_mla = False
    worker.use_host_buffer = False
    worker.kv_cache_layout = "HND"
    worker.host_buffer_kv_cache_layout = "HND"
    worker.backend_name = "FLASHINFER_GEMMA4_TRTLLM_GEN"
    worker.enable_permute_local_kv = False
    worker.enable_heterogeneous_attn_post_process = False
    worker._is_hma_required = False
    worker._physical_blocks_per_logical_kv_block = 1
    worker._sp_flags_cache = [False] * len(config.groups)
    worker.kv_transfer_config = SimpleNamespace(enable_permute_local_kv=False)
    worker.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(enable_prefix_caching=False)
    )
    worker.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(
                kv_cache_spec=SimpleNamespace(
                    block_size=group.token_capacity,
                    kv_planes=group.destination_plane_count,
                )
            )
            for group in config.groups
        ]
    )
    worker._group_spec_types = (FullAttentionSpec,) * len(config.groups)
    worker.transfer_topo = TransferTopology(
        tp_rank=0,
        tp_size=1,
        block_size=worker.block_size,
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


def test_static_fixture_is_bound_to_the_current_connector_version() -> None:
    assert SEMANTIC_HANDSHAKE_CONNECTOR_VERSION == NIXL_CONNECTOR_VERSION == 9


def test_static_profile_is_explicitly_not_a_runtime_capture() -> None:
    config = load_config(CONFIG_PATH)
    profile = load_semantic_handshake_profile(
        FIXTURE_DIRECTORY / config.semantic_handshake_manifest,
        config.semantic_handshake_sha256,
        config.semantic_handshake_profile,
    )

    assert profile.evidence_scope == STATIC_SEMANTIC_EVIDENCE_SCOPE
    assert profile.runtime_capture_authenticated is False
    assert profile.rank_identity_field == "tp_rank"


@pytest.mark.parametrize(
    "mutation",
    [
        "connector_version",
        "group_semantic_names",
        "source_group_planes",
        "physical_group_token_capacities",
        "region_group_indices",
        "runtime_capture_authenticated",
        "rank_identity_field",
        "descriptor_element_size_bytes",
    ],
)
def test_config_rejects_authenticated_semantic_fixture_drift(
    tmp_path: Path,
    mutation: str,
) -> None:
    config_path = _write_mutated_fixture(tmp_path, mutation)

    with pytest.raises(ConfigError):
        load_config(config_path)


def test_config_rejects_unauthenticated_semantic_fixture_changes(
    tmp_path: Path,
) -> None:
    config_value = json.loads(CONFIG_PATH.read_text())
    fixture_name = config_value["semantic_handshake_manifest"]
    fixture_payload = (FIXTURE_DIRECTORY / fixture_name).read_bytes() + b"\n"
    (tmp_path / fixture_name).write_bytes(fixture_payload)
    _copy_legacy_captures(tmp_path)
    config_path = tmp_path / CONFIG_PATH.name
    config_path.write_text(json.dumps(config_value))

    with pytest.raises(ConfigError, match="SHA-256"):
        load_config(config_path)


def test_static_profile_replays_through_current_v9_production_validator() -> None:
    config = load_config(CONFIG_PATH)
    profile = load_semantic_handshake_profile(
        FIXTURE_DIRECTORY / config.semantic_handshake_manifest,
        config.semantic_handshake_sha256,
        config.semantic_handshake_profile,
    )
    source_row_bytes = tuple(region.row_bytes for region in config.regions)
    worker = _fixture_worker()
    assert worker.transfer_topo is not None
    plan = compute_tp_mapping(
        worker.transfer_topo,
        len(config.producer_devices),
        worker._group_spec_types,
    )
    metadata_by_rank: dict[int, NixlAgentMetadata] = {}
    for rank in config.producer_devices:
        regions = profile.region_descriptors(
            num_blocks=config.source_block_count,
            row_bytes=source_row_bytes,
            base_address=0x20000000000 + rank * 0x10000000000,
        )
        metadata_by_rank[rank] = NixlAgentMetadata(
            engine_id="fixture-prefill",
            tp_rank=rank,
            agent_metadata=f"fixture-rank-{rank}".encode(),
            kv_caches_base_addr=[region.base_address for region in regions],
            device_id=rank,
            num_blocks=config.source_block_count,
            block_lens=list(source_row_bytes),
            kv_cache_layout="HND",
            block_size=16,
            ssm_sizes=(0, 0),
            attn_backend_name="FLASHINFER_GEMMA4_TRTLLM_GEN",
            physical_blocks_per_logical_kv_block=1,
            registration_generation=f"static-fixture-rank-{rank}",
            regions=regions,
            source_group_planes=profile.source_group_planes,
            physical_group_token_capacities=(profile.physical_group_token_capacities),
        )

    worker._validate_remote_handshake_roster(
        metadata_by_rank,
        len(config.producer_devices),
        plan,
    )

    worker.nixl_wrapper.add_remote_agent.assert_not_called()
