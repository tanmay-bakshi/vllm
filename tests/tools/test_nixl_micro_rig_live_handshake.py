"""Live connector-v11 wire capture and offline replay tests."""

import hashlib
from pathlib import Path

import msgspec
import pytest

from tools.gemma4_pd.nixl_micro_rig.config import load_config
from tools.gemma4_pd.nixl_micro_rig.handshake import (
    load_semantic_handshake_profile,
)
from tools.gemma4_pd.nixl_micro_rig.live_handshake import (
    CapturedCodeIdentity,
    CaptureWriteResult,
    LiveHandshakeCapture,
    LiveHandshakeError,
    capture_live_handshake,
    verify_live_handshake_capture,
    write_result_output,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlAgentMetadata,
    NixlHandshakePayload,
    PackedWriteConsumerPoolGeometry,
    PackedWriteProducerPoolGeometry,
)

FIXTURE_DIRECTORY = Path(__file__).parents[2] / "tools" / "gemma4_pd" / "nixl_micro_rig"
CONFIG_PATH = FIXTURE_DIRECTORY / "gemma4_tp4_to_tp1_2k.json"
CODE_IDENTITY = CapturedCodeIdentity(
    commit="a" * 40,
    tree="b" * 40,
    worktree_clean=True,
)
COMPATIBILITY_HASH = "c" * 64
MODEL_CONFIG_PAYLOAD = (
    b'{"architectures":["Gemma4ForCausalLM"],"model_type":"gemma4",'
    b'"num_key_value_heads":8}'
)
MODEL_CONFIG_SHA256 = hashlib.sha256(MODEL_CONFIG_PAYLOAD).hexdigest()


def _wire_payload(
    *,
    rank: int,
    decoder: bool,
    metadata_rank: int | None = None,
    semantic_drift: bool = False,
    block_count: int | None = None,
    decoder_row_scale: int | None = None,
    packed_pool_slot_bytes: int | None = None,
    group_token_capacities: tuple[int, ...] | None = None,
) -> bytes:
    """Build one complete synthetic live connector-v11 response.

    :param rank: Endpoint TP rank.
    :param decoder: Whether to materialize TP1 decoder geometry.
    :param metadata_rank: Optional producer-rank identity encoded in metadata.
    :param semantic_drift: Whether to corrupt one producer semantic identity.
    :param block_count: Optional advertised physical block count.
    :param decoder_row_scale: Optional decoder-to-producer row-width ratio.
    :param packed_pool_slot_bytes: Optional packed-WRITE rank stride.
    :param group_token_capacities: Optional physical group-capacity override.
    :returns: Exact outer MessagePack payload.
    """
    config = load_config(CONFIG_PATH)
    profile = load_semantic_handshake_profile(
        FIXTURE_DIRECTORY / config.semantic_handshake_manifest,
        config.semantic_handshake_sha256,
        config.semantic_handshake_profile,
    )
    if block_count is None:
        block_count = config.source_block_count
    row_scale = len(config.producer_devices) if decoder else 1
    if decoder and decoder_row_scale is not None:
        row_scale = decoder_row_scale
    row_bytes = tuple(region.row_bytes * row_scale for region in config.regions)
    regions = list(
        profile.region_descriptors(
            num_blocks=block_count,
            row_bytes=row_bytes,
            base_address=(
                0x50000000000 if decoder else 0x10000000000 + rank * 0x10000000000
            ),
        )
    )
    if semantic_drift:
        first = regions[0]
        regions[0] = msgspec.structs.replace(first, semantic_name="wrong-region")
    registration_generation = f"volatile-generation-{decoder}-{rank}"
    if packed_pool_slot_bytes is None:
        packed_pool_slot_bytes = 256 * 1024 * 1024
    if group_token_capacities is None:
        group_token_capacities = tuple(group.token_capacity for group in config.groups)
    packed_write_producer_pool = None
    packed_write_consumer_pool = None
    if decoder:
        packed_write_consumer_pool = PackedWriteConsumerPoolGeometry(
            registration_generation=registration_generation,
            base_address=0xA0000000000,
            registered_bytes=2 * 4 * packed_pool_slot_bytes,
            slot_size_bytes=4 * packed_pool_slot_bytes,
            slot_count=2,
            source_tp_size=4,
            rank_stride_bytes=packed_pool_slot_bytes,
            device_id=0,
            alignment_bytes=256,
        )
    else:
        packed_write_producer_pool = PackedWriteProducerPoolGeometry(
            registration_generation=registration_generation,
            base_address=0x90000000000 + rank * 0x10000000000,
            registered_bytes=2 * packed_pool_slot_bytes,
            slot_size_bytes=packed_pool_slot_bytes,
            slot_count=2,
            device_id=rank,
            alignment_bytes=256,
        )
    metadata = NixlAgentMetadata(
        engine_id="live-decoder" if decoder else "live-prefill",
        tp_rank=rank if metadata_rank is None else metadata_rank,
        agent_metadata=f"opaque-agent-{decoder}-{rank}".encode(),
        kv_caches_base_addr=[region.base_address for region in regions],
        device_id=0 if decoder else rank,
        num_blocks=block_count,
        block_lens=list(row_bytes),
        kv_cache_layout="HND",
        block_size=16,
        ssm_sizes=(0, 0),
        attn_backend_name="FLASHINFER_GEMMA4_TRTLLM_GEN",
        physical_blocks_per_logical_kv_block=1,
        registration_generation=registration_generation,
        regions=tuple(regions),
        source_group_planes=tuple(
            group.destination_plane_count for group in config.groups
        ),
        physical_group_token_capacities=group_token_capacities,
        packed_write_producer_pool=packed_write_producer_pool,
        packed_write_consumer_pool=packed_write_consumer_pool,
    )
    handshake = NixlHandshakePayload(
        compatibility_hash=COMPATIBILITY_HASH,
        agent_metadata_bytes=msgspec.msgpack.encode(metadata),
    )
    return msgspec.msgpack.encode(handshake)


def _payloads(
    *,
    misbound_rank: int | None = None,
    semantic_drift_rank: int | None = None,
    decoder_block_count: int | None = None,
    decoder_row_scale: int | None = None,
    packed_pool_slot_bytes: int | None = None,
    packed_pool_drift_rank: int | None = None,
    group_token_capacities: tuple[int, ...] | None = None,
) -> dict[tuple[str, int], bytes]:
    result = {
        ("producer", rank): _wire_payload(
            rank=rank,
            decoder=False,
            metadata_rank=(rank - 1) if rank == misbound_rank else rank,
            semantic_drift=rank == semantic_drift_rank,
            packed_pool_slot_bytes=(
                None
                if packed_pool_slot_bytes is None
                else packed_pool_slot_bytes
                + (2 * 1024 * 1024 if rank == packed_pool_drift_rank else 0)
            ),
            group_token_capacities=group_token_capacities,
        )
        for rank in range(4)
    }
    result[("decoder", 0)] = _wire_payload(
        rank=0,
        decoder=True,
        block_count=decoder_block_count,
        decoder_row_scale=decoder_row_scale,
        group_token_capacities=group_token_capacities,
    )
    return result


def _capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    misbound_rank: int | None = None,
    semantic_drift_rank: int | None = None,
    decoder_block_count: int | None = None,
    decoder_row_scale: int | None = None,
    packed_pool_slot_bytes: int | None = None,
    packed_pool_drift_rank: int | None = None,
    group_token_capacities: tuple[int, ...] | None = None,
) -> tuple[Path, str]:
    """Capture mocked endpoint bytes through the public command implementation.

    :param tmp_path: Fresh artifact directory.
    :param monkeypatch: Pytest patch helper.
    :param misbound_rank: Optional producer response whose identity is replayed.
    :param semantic_drift_rank: Optional corrupt producer rank.
    :param decoder_block_count: Optional decoder block-count override.
    :param decoder_row_scale: Optional decoder row-width-ratio override.
    :param packed_pool_slot_bytes: Optional producer packed-WRITE slot size.
    :param packed_pool_drift_rank: Optional producer rank with different geometry.
    :param group_token_capacities: Optional physical group-capacity override.
    :returns: Capture path and external SHA-256.
    """
    from tools.gemma4_pd.nixl_micro_rig import live_handshake

    payloads = _payloads(
        misbound_rank=misbound_rank,
        semantic_drift_rank=semantic_drift_rank,
        decoder_block_count=decoder_block_count,
        decoder_row_scale=decoder_row_scale,
        packed_pool_slot_bytes=packed_pool_slot_bytes,
        packed_pool_drift_rank=packed_pool_drift_rank,
        group_token_capacities=group_token_capacities,
    )
    monkeypatch.setattr(live_handshake, "_git_identity", lambda _: CODE_IDENTITY)
    monkeypatch.setattr(
        live_handshake,
        "_fetch_payload",
        lambda host, _port, rank, _timeout: payloads[(host, rank)],
    )
    config = load_config(CONFIG_PATH)
    model_config = tmp_path / "model-config.json"
    model_config.write_bytes(MODEL_CONFIG_PAYLOAD)
    output = tmp_path / "live-v11.json"
    result = capture_live_handshake(
        config_path=CONFIG_PATH,
        config=config,
        code_root=tmp_path,
        producer_host="producer",
        producer_port=15620,
        producer_engine_id=None,
        decoder_host="decoder",
        decoder_port=15621,
        decoder_engine_id=None,
        model_config_path=model_config,
        expected_model_config_sha256=MODEL_CONFIG_SHA256,
        output_path=output,
        timeout_seconds=1.0,
    )
    return output, result.sha256


def test_live_capture_preserves_raw_payloads_and_replays_validator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture_path, capture_sha256 = _capture(tmp_path, monkeypatch)
    capture = msgspec.json.decode(
        capture_path.read_bytes(),
        type=LiveHandshakeCapture,
    )

    assert tuple(payload.rank for payload in capture.producer.payloads) == (0, 1, 2, 3)
    assert len({payload.payload for payload in capture.producer.payloads}) == 4
    assert (
        len(
            {
                payload.normalized_contract_sha256
                for payload in capture.producer.payloads
            }
        )
        == 1
    )
    assert (
        len(
            {
                payload.registration_generation_sha256
                for payload in capture.producer.payloads
            }
        )
        == 4
    )
    assert capture_path.stat().st_mode & 0o222 == 0

    result = verify_live_handshake_capture(
        capture_path=capture_path,
        expected_sha256=capture_sha256,
        config_path=CONFIG_PATH,
        config=load_config(CONFIG_PATH),
        code_root=tmp_path,
        expected_model_config_sha256=MODEL_CONFIG_SHA256,
    )
    assert result.compatibility_hash == COMPATIBILITY_HASH
    assert result.producer_rank_count == 4
    assert result.validator_replay == "passed_without_native_mutation"


def test_live_capture_preserves_selected_producer_and_consumer_pool_geometry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slot_bytes = 256 * 1024 * 1024
    capture_path, _ = _capture(
        tmp_path,
        monkeypatch,
        packed_pool_slot_bytes=slot_bytes,
    )
    capture = msgspec.json.decode(
        capture_path.read_bytes(),
        type=LiveHandshakeCapture,
    )

    pools = []
    for payload in capture.producer.payloads:
        handshake = msgspec.msgpack.decode(payload.payload, type=NixlHandshakePayload)
        metadata = msgspec.msgpack.decode(
            handshake.agent_metadata_bytes,
            type=NixlAgentMetadata,
        )
        assert metadata.packed_write_producer_pool is not None
        assert metadata.packed_write_consumer_pool is None
        pools.append(metadata.packed_write_producer_pool)
    assert {pool.slot_size_bytes for pool in pools} == {slot_bytes}
    assert {pool.slot_count for pool in pools} == {2}
    assert len({pool.base_address for pool in pools}) == 4
    decoder_handshake = msgspec.msgpack.decode(
        capture.decoder.payloads[0].payload,
        type=NixlHandshakePayload,
    )
    decoder_metadata = msgspec.msgpack.decode(
        decoder_handshake.agent_metadata_bytes,
        type=NixlAgentMetadata,
    )
    assert decoder_metadata.packed_write_producer_pool is None
    decoder_pool = decoder_metadata.packed_write_consumer_pool
    assert decoder_pool is not None
    assert decoder_pool.rank_stride_bytes == slot_bytes
    assert decoder_pool.source_tp_size == 4
    assert decoder_pool.slot_size_bytes == 4 * slot_bytes
    assert decoder_pool.registered_bytes == 2 * 4 * slot_bytes


def test_capture_rejects_cross_rank_packed_pool_geometry_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(LiveHandshakeError, match="selected v11 contract"):
        _capture(
            tmp_path,
            monkeypatch,
            packed_pool_slot_bytes=256 * 1024 * 1024,
            packed_pool_drift_rank=2,
        )

    assert not (tmp_path / "live-v11.json").exists()


def test_capture_rejects_cross_rank_semantic_drift_before_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(LiveHandshakeError, match="different normalized contracts"):
        _capture(tmp_path, monkeypatch, semantic_drift_rank=2)

    assert not (tmp_path / "live-v11.json").exists()


def test_capture_rejects_metadata_bound_to_a_different_rank(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(
        LiveHandshakeError,
        match="metadata rank differs from the queried endpoint rank",
    ):
        _capture(tmp_path, monkeypatch, misbound_rank=2)

    assert not (tmp_path / "live-v11.json").exists()


@pytest.mark.parametrize(
    ("decoder_block_count", "decoder_row_scale", "message"),
    (
        (63_999, 4, "block count differs"),
        (64_000, 3, "row lengths differ"),
    ),
)
def test_capture_requires_exact_64k_four_way_decoder_geometry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    decoder_block_count: int,
    decoder_row_scale: int,
    message: str,
) -> None:
    with pytest.raises(LiveHandshakeError, match=message):
        _capture(
            tmp_path,
            monkeypatch,
            decoder_block_count=decoder_block_count,
            decoder_row_scale=decoder_row_scale,
        )

    assert not (tmp_path / "live-v11.json").exists()


def test_capture_reports_live_capacity_tuple_that_differs_from_rig(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stale_capacities = (32,) * 12 + (64,)

    with pytest.raises(
        LiveHandshakeError,
        match=("live metadata group token capacities .* differ from rig configuration"),
    ):
        _capture(
            tmp_path,
            monkeypatch,
            group_token_capacities=stale_capacities,
        )

    assert not (tmp_path / "live-v11.json").exists()


def test_capture_rejects_independent_engine_identity_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools.gemma4_pd.nixl_micro_rig import live_handshake

    payloads = _payloads()
    monkeypatch.setattr(live_handshake, "_git_identity", lambda _: CODE_IDENTITY)
    monkeypatch.setattr(
        live_handshake,
        "_fetch_payload",
        lambda host, _port, rank, _timeout: payloads[(host, rank)],
    )
    (tmp_path / "model-config.json").write_bytes(MODEL_CONFIG_PAYLOAD)

    with pytest.raises(
        LiveHandshakeError,
        match="captured producer engine identity differs",
    ):
        capture_live_handshake(
            config_path=CONFIG_PATH,
            config=load_config(CONFIG_PATH),
            code_root=tmp_path,
            producer_host="producer",
            producer_port=15620,
            producer_engine_id="wrong-prefill",
            decoder_host="decoder",
            decoder_port=15621,
            decoder_engine_id="live-decoder",
            model_config_path=tmp_path / "model-config.json",
            expected_model_config_sha256=MODEL_CONFIG_SHA256,
            output_path=tmp_path / "live-v11.json",
            timeout_seconds=1.0,
        )

    assert not (tmp_path / "live-v11.json").exists()


def test_capture_requires_externally_authenticated_model_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools.gemma4_pd.nixl_micro_rig import live_handshake

    model_config = tmp_path / "model-config.json"
    model_config.write_bytes(MODEL_CONFIG_PAYLOAD)
    monkeypatch.setattr(live_handshake, "_git_identity", lambda _: CODE_IDENTITY)

    with pytest.raises(LiveHandshakeError, match="model-config SHA-256"):
        capture_live_handshake(
            config_path=CONFIG_PATH,
            config=load_config(CONFIG_PATH),
            code_root=tmp_path,
            producer_host="producer",
            producer_port=15620,
            producer_engine_id=None,
            decoder_host="decoder",
            decoder_port=15621,
            decoder_engine_id=None,
            model_config_path=model_config,
            expected_model_config_sha256="0" * 64,
            output_path=tmp_path / "live-v11.json",
            timeout_seconds=1.0,
        )

    assert not (tmp_path / "live-v11.json").exists()


def test_verifier_requires_external_capture_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture_path, _ = _capture(tmp_path, monkeypatch)

    with pytest.raises(LiveHandshakeError, match="differs"):
        verify_live_handshake_capture(
            capture_path=capture_path,
            expected_sha256="0" * 64,
            config_path=CONFIG_PATH,
            config=load_config(CONFIG_PATH),
            code_root=tmp_path,
            expected_model_config_sha256=MODEL_CONFIG_SHA256,
        )


def test_verifier_detects_raw_payload_tampering_even_with_new_file_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture_path, _ = _capture(tmp_path, monkeypatch)
    capture = msgspec.json.decode(
        capture_path.read_bytes(),
        type=LiveHandshakeCapture,
    )
    rank_zero = capture.producer.payloads[0]
    corrupted_rank_zero = msgspec.structs.replace(
        rank_zero,
        payload=rank_zero.payload + b"corruption",
    )
    corrupted_producer = msgspec.structs.replace(
        capture.producer,
        payloads=(corrupted_rank_zero, *capture.producer.payloads[1:]),
    )
    corrupted_capture = msgspec.structs.replace(
        capture,
        producer=corrupted_producer,
    )
    encoded = msgspec.json.encode(corrupted_capture)
    corrupted_path = tmp_path / "corrupted.json"
    corrupted_path.write_bytes(encoded)

    with pytest.raises(LiveHandshakeError, match="wire-payload digest differs"):
        verify_live_handshake_capture(
            capture_path=corrupted_path,
            expected_sha256=hashlib.sha256(encoded).hexdigest(),
            config_path=CONFIG_PATH,
            config=load_config(CONFIG_PATH),
            code_root=tmp_path,
            expected_model_config_sha256=MODEL_CONFIG_SHA256,
        )


def test_result_output_is_deterministic_exclusive_and_read_only(tmp_path: Path) -> None:
    output = tmp_path / "capture-result.json"
    result = CaptureWriteResult(path=tmp_path / "capture.json", sha256="d" * 64)

    write_result_output(output, result)

    assert output.read_text().endswith("\n")
    assert output.stat().st_mode & 0o222 == 0
    with pytest.raises(LiveHandshakeError, match="already exists"):
        write_result_output(output, result)
