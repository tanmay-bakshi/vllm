# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for packed-WRITE pool registration and publication."""

from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.nixl import (
    base_worker as base_worker_module,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import (
    NixlBaseConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    PackedWriteConsumerPoolGeometry,
    PackedWriteProducerPoolGeometry,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.packed_write_config import (
    PackedWriteConfig,
)

_MIB = 1024 * 1024
_CHUNK_BYTES = 64 * _MIB
_SOURCE_TP_SIZE = 4
_SLOT_COUNT = 2
_ALIGNMENT_BYTES = 256
_REGISTRATION_GENERATION = "local-registration"


class _FakeTensor:
    """Minimal CUDA tensor surface consumed by pool registration."""

    def __init__(
        self,
        *,
        address: int,
        size: int,
        device: torch.device,
        dimensions: int = 1,
        dtype: torch.dtype = torch.uint8,
        contiguous: bool = True,
    ) -> None:
        self.dtype = dtype
        self.device = device
        self._address = address
        self._size = size
        self._dimensions = dimensions
        self._contiguous = contiguous

    def data_ptr(self) -> int:
        return self._address

    def dim(self) -> int:
        return self._dimensions

    def is_contiguous(self) -> bool:
        return self._contiguous

    def numel(self) -> int:
        return self._size


def _config(*, enabled: bool = True) -> PackedWriteConfig:
    return PackedWriteConfig(
        enabled=enabled,
        chunk_bytes_per_rank=_CHUNK_BYTES,
        min_descriptors_per_rank=1024,
        producer_slot_count=_SLOT_COUNT,
        consumer_slot_count=_SLOT_COUNT,
        alignment_bytes=_ALIGNMENT_BYTES,
        warn_after_s=30.0,
        fail_after_s=300.0,
    )


def _worker(
    *,
    role: str = "kv_consumer",
    enabled: bool = True,
) -> NixlBaseConnectorWorker:
    worker = cast(
        NixlBaseConnectorWorker,
        object.__new__(NixlBaseConnectorWorker),
    )
    worker._packed_write_config = _config(enabled=enabled)
    worker.kv_transfer_config = SimpleNamespace(kv_role=role)
    worker.device_type = "cuda"
    worker.device_id = 3
    worker.use_host_buffer = False
    worker._registration_generation = _REGISTRATION_GENERATION
    worker._region_tensors = [object()]
    worker._region_descriptors = (object(),)
    worker._region_rows = [
        _FakeTensor(
            address=0x20_0000,
            size=4096,
            device=torch.device("cuda", 3),
            dimensions=2,
        )
    ]
    worker._coalesce_region_rows = MagicMock(return_value=True)
    worker.nixl_wrapper = MagicMock()
    worker.nixl_wrapper.get_reg_descs.return_value = "registration"
    worker.nixl_backends = ["UCX", "CUDA_IPC"]
    worker.nixl_memory_type = "VRAM"
    worker._registered_descs = []
    worker._packed_write_producer_pool_buffer = None
    worker._packed_write_producer_pool_geometry = None
    worker._packed_write_producer_pack_stream = None
    worker._packed_write_consumer_pool_buffer = None
    worker._packed_write_consumer_pool_geometry = None
    worker._packed_write_consumer_slot_size_bytes = None
    worker._packed_write_consumer_source_tp_size = _SOURCE_TP_SIZE
    worker._packed_write_scatter_stream = None
    return worker


def _producer_geometry() -> PackedWriteProducerPoolGeometry:
    return PackedWriteProducerPoolGeometry(
        registration_generation=_REGISTRATION_GENERATION,
        base_address=0x40_0000,
        registered_bytes=_SLOT_COUNT * _CHUNK_BYTES,
        slot_size_bytes=_CHUNK_BYTES,
        slot_count=_SLOT_COUNT,
        device_id=3,
        alignment_bytes=_ALIGNMENT_BYTES,
    )


def _consumer_geometry() -> PackedWriteConsumerPoolGeometry:
    slot_size_bytes = _SOURCE_TP_SIZE * _CHUNK_BYTES
    return PackedWriteConsumerPoolGeometry(
        registration_generation=_REGISTRATION_GENERATION,
        base_address=0x80_0000,
        registered_bytes=_SLOT_COUNT * slot_size_bytes,
        slot_size_bytes=slot_size_bytes,
        slot_count=_SLOT_COUNT,
        source_tp_size=_SOURCE_TP_SIZE,
        rank_stride_bytes=_CHUNK_BYTES,
        device_id=3,
        alignment_bytes=_ALIGNMENT_BYTES,
    )


def _install_published_resources(
    worker: NixlBaseConnectorWorker,
    *,
    producer: bool,
    consumer: bool,
) -> None:
    if producer:
        worker._packed_write_producer_pool_buffer = object()
        worker._packed_write_producer_pool_geometry = _producer_geometry()
        worker._packed_write_producer_pack_stream = object()
    if consumer:
        worker._packed_write_consumer_pool_buffer = object()
        worker._packed_write_consumer_pool_geometry = _consumer_geometry()
        worker._packed_write_consumer_slot_size_bytes = _SOURCE_TP_SIZE * _CHUNK_BYTES
        worker._packed_write_scatter_stream = object()


@pytest.mark.cpu_test
@pytest.mark.parametrize("role", ["kv_producer", "kv_both"])
def test_registers_exact_bounded_producer_pool(
    monkeypatch: pytest.MonkeyPatch,
    role: str,
) -> None:
    worker = _worker(role=role)
    registered_bytes = _SLOT_COUNT * _CHUNK_BYTES
    pool = _FakeTensor(
        address=0x40_0000,
        size=registered_bytes,
        device=torch.device("cuda", 3),
    )
    pack_stream = object()
    empty = MagicMock(return_value=pool)
    stream = MagicMock(return_value=pack_stream)
    monkeypatch.setattr(base_worker_module.torch, "empty", empty)
    monkeypatch.setattr(base_worker_module.torch.cuda, "Stream", stream)

    worker._initialize_packed_write_producer_pool()

    empty.assert_called_once_with(
        registered_bytes,
        dtype=torch.uint8,
        device=torch.device("cuda", 3),
    )
    stream.assert_called_once_with(device=torch.device("cuda", 3))
    worker.nixl_wrapper.get_reg_descs.assert_called_once_with(
        [(0x40_0000, registered_bytes, 3, "")],
        "VRAM",
    )
    worker.nixl_wrapper.register_memory.assert_called_once_with(
        "registration",
        backends=["UCX", "CUDA_IPC"],
    )
    assert worker._registered_descs == ["registration"]
    assert worker._packed_write_producer_pool_buffer is pool
    assert worker._packed_write_producer_pool_geometry == _producer_geometry()
    assert worker._packed_write_producer_pack_stream is pack_stream


@pytest.mark.cpu_test
@pytest.mark.parametrize("role", ["kv_consumer", "kv_both"])
def test_registers_exact_rank_major_consumer_pool(
    monkeypatch: pytest.MonkeyPatch,
    role: str,
) -> None:
    worker = _worker(role=role)
    slot_size_bytes = _SOURCE_TP_SIZE * _CHUNK_BYTES
    registered_bytes = _SLOT_COUNT * slot_size_bytes
    pool = _FakeTensor(
        address=0x80_0000,
        size=registered_bytes,
        device=torch.device("cuda", 3),
    )
    scatter_stream = object()
    empty = MagicMock(return_value=pool)
    stream = MagicMock(return_value=scatter_stream)
    monkeypatch.setattr(base_worker_module.torch, "empty", empty)
    monkeypatch.setattr(base_worker_module.torch.cuda, "Stream", stream)

    worker._initialize_packed_write_consumer_pool()

    empty.assert_called_once_with(
        registered_bytes,
        dtype=torch.uint8,
        device=torch.device("cuda", 3),
    )
    stream.assert_called_once_with(device=torch.device("cuda", 3))
    worker.nixl_wrapper.get_reg_descs.assert_called_once_with(
        [(0x80_0000, registered_bytes, 3, "")],
        "VRAM",
    )
    worker.nixl_wrapper.register_memory.assert_called_once_with(
        "registration",
        backends=["UCX", "CUDA_IPC"],
    )
    assert worker._registered_descs == ["registration"]
    assert worker._packed_write_consumer_pool_buffer is pool
    assert worker._packed_write_consumer_pool_geometry == _consumer_geometry()
    assert worker._packed_write_consumer_slot_size_bytes == slot_size_bytes
    assert worker._packed_write_scatter_stream is scatter_stream


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("initializer", "role", "enabled"),
    [
        ("producer", "kv_producer", False),
        ("producer", "kv_consumer", True),
        ("consumer", "kv_consumer", False),
        ("consumer", "kv_producer", True),
    ],
)
def test_inactive_pool_does_not_mutate_registration(
    initializer: str,
    role: str,
    enabled: bool,
) -> None:
    worker = _worker(role=role, enabled=enabled)

    if initializer == "producer":
        worker._initialize_packed_write_producer_pool()
    else:
        worker._initialize_packed_write_consumer_pool()

    worker._coalesce_region_rows.assert_not_called()
    worker.nixl_wrapper.get_reg_descs.assert_not_called()
    worker.nixl_wrapper.register_memory.assert_not_called()
    assert worker._registered_descs == []


@pytest.mark.cpu_test
@pytest.mark.parametrize("initializer", ["producer", "consumer"])
@pytest.mark.parametrize(
    "invalid_state",
    ["cpu", "host_buffer", "missing_regions", "noncanonical_regions"],
)
def test_pool_fails_closed_before_allocation(
    monkeypatch: pytest.MonkeyPatch,
    initializer: str,
    invalid_state: str,
) -> None:
    role = "kv_producer" if initializer == "producer" else "kv_consumer"
    worker = _worker(role=role)
    if invalid_state == "cpu":
        worker.device_type = "cpu"
    elif invalid_state == "host_buffer":
        worker.use_host_buffer = True
    elif invalid_state == "missing_regions":
        worker._region_tensors = []
    else:
        worker._coalesce_region_rows.return_value = False
    empty = MagicMock()
    monkeypatch.setattr(base_worker_module.torch, "empty", empty)

    with pytest.raises(RuntimeError):
        if initializer == "producer":
            worker._initialize_packed_write_producer_pool()
        else:
            worker._initialize_packed_write_consumer_pool()

    empty.assert_not_called()
    worker.nixl_wrapper.get_reg_descs.assert_not_called()
    worker.nixl_wrapper.register_memory.assert_not_called()
    assert worker._registered_descs == []


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("initializer", "field"),
    [
        ("producer", "_packed_write_producer_pool_buffer"),
        ("producer", "_packed_write_producer_pool_geometry"),
        ("producer", "_packed_write_producer_pack_stream"),
        ("consumer", "_packed_write_consumer_pool_buffer"),
        ("consumer", "_packed_write_consumer_pool_geometry"),
        ("consumer", "_packed_write_consumer_slot_size_bytes"),
        ("consumer", "_packed_write_scatter_stream"),
    ],
)
def test_any_partial_state_rejects_duplicate_initialization(
    initializer: str,
    field: str,
) -> None:
    role = "kv_producer" if initializer == "producer" else "kv_consumer"
    worker = _worker(role=role)
    setattr(worker, field, object())

    with pytest.raises(
        RuntimeError,
        match=f"{initializer} pool is already initialized",
    ):
        if initializer == "producer":
            worker._initialize_packed_write_producer_pool()
        else:
            worker._initialize_packed_write_consumer_pool()

    worker._coalesce_region_rows.assert_not_called()
    worker.nixl_wrapper.register_memory.assert_not_called()


@pytest.mark.cpu_test
@pytest.mark.parametrize("initializer", ["producer", "consumer"])
@pytest.mark.parametrize(
    ("address", "dimensions", "contiguous"),
    [(0x40_0001, 1, True), (0x40_0000, 2, True), (0x40_0000, 1, False)],
)
def test_invalid_pool_allocation_is_never_registered(
    monkeypatch: pytest.MonkeyPatch,
    initializer: str,
    address: int,
    dimensions: int,
    contiguous: bool,
) -> None:
    role = "kv_producer" if initializer == "producer" else "kv_consumer"
    worker = _worker(role=role)
    registered_bytes = (
        _SLOT_COUNT * _CHUNK_BYTES
        if initializer == "producer"
        else _SLOT_COUNT * _SOURCE_TP_SIZE * _CHUNK_BYTES
    )
    pool = _FakeTensor(
        address=address,
        size=registered_bytes,
        device=torch.device("cuda", 3),
        dimensions=dimensions,
        contiguous=contiguous,
    )
    monkeypatch.setattr(
        base_worker_module.torch,
        "empty",
        MagicMock(return_value=pool),
    )

    with pytest.raises(RuntimeError):
        if initializer == "producer":
            worker._initialize_packed_write_producer_pool()
        else:
            worker._initialize_packed_write_consumer_pool()

    worker.nixl_wrapper.get_reg_descs.assert_not_called()
    worker.nixl_wrapper.register_memory.assert_not_called()
    assert worker._registered_descs == []


@pytest.mark.cpu_test
@pytest.mark.parametrize("initializer", ["producer", "consumer"])
def test_stream_failure_leaves_no_registered_or_published_state(
    monkeypatch: pytest.MonkeyPatch,
    initializer: str,
) -> None:
    role = "kv_producer" if initializer == "producer" else "kv_consumer"
    worker = _worker(role=role)
    registered_bytes = (
        _SLOT_COUNT * _CHUNK_BYTES
        if initializer == "producer"
        else _SLOT_COUNT * _SOURCE_TP_SIZE * _CHUNK_BYTES
    )
    pool = _FakeTensor(
        address=0x40_0000,
        size=registered_bytes,
        device=torch.device("cuda", 3),
    )
    monkeypatch.setattr(
        base_worker_module.torch,
        "empty",
        MagicMock(return_value=pool),
    )
    monkeypatch.setattr(
        base_worker_module.torch.cuda,
        "Stream",
        MagicMock(side_effect=RuntimeError("stream allocation failed")),
    )

    with pytest.raises(RuntimeError, match="stream allocation failed"):
        if initializer == "producer":
            worker._initialize_packed_write_producer_pool()
        else:
            worker._initialize_packed_write_consumer_pool()

    worker.nixl_wrapper.get_reg_descs.assert_not_called()
    worker.nixl_wrapper.register_memory.assert_not_called()
    assert worker._registered_descs == []
    assert worker._packed_write_producer_pool_geometry is None
    assert worker._packed_write_consumer_pool_geometry is None


@pytest.mark.cpu_test
@pytest.mark.parametrize("initializer", ["producer", "consumer"])
def test_native_registration_failure_is_not_published(
    monkeypatch: pytest.MonkeyPatch,
    initializer: str,
) -> None:
    role = "kv_producer" if initializer == "producer" else "kv_consumer"
    worker = _worker(role=role)
    registered_bytes = (
        _SLOT_COUNT * _CHUNK_BYTES
        if initializer == "producer"
        else _SLOT_COUNT * _SOURCE_TP_SIZE * _CHUNK_BYTES
    )
    pool = _FakeTensor(
        address=0x40_0000,
        size=registered_bytes,
        device=torch.device("cuda", 3),
    )
    monkeypatch.setattr(
        base_worker_module.torch,
        "empty",
        MagicMock(return_value=pool),
    )
    monkeypatch.setattr(
        base_worker_module.torch.cuda,
        "Stream",
        MagicMock(return_value=object()),
    )
    worker.nixl_wrapper.register_memory.side_effect = RuntimeError(
        "native registration failed"
    )

    with pytest.raises(RuntimeError, match="native registration failed"):
        if initializer == "producer":
            worker._initialize_packed_write_producer_pool()
        else:
            worker._initialize_packed_write_consumer_pool()

    assert worker._registered_descs == []
    assert worker._packed_write_producer_pool_geometry is None
    assert worker._packed_write_consumer_pool_geometry is None


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("role", "producer", "consumer"),
    [
        ("kv_producer", True, False),
        ("kv_consumer", False, True),
        ("kv_both", True, True),
    ],
)
def test_handshake_publication_is_exactly_role_scoped(
    role: str,
    producer: bool,
    consumer: bool,
) -> None:
    worker = _worker(role=role)
    _install_published_resources(worker, producer=producer, consumer=consumer)

    producer_pool, consumer_pool = worker._packed_write_handshake_pools()

    assert producer_pool == (_producer_geometry() if producer else None)
    assert consumer_pool == (_consumer_geometry() if consumer else None)


@pytest.mark.cpu_test
def test_disabled_handshake_publishes_no_pools() -> None:
    worker = _worker(role="kv_both", enabled=False)

    assert worker._packed_write_handshake_pools() == (None, None)


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("role", "producer", "consumer", "message"),
    [
        ("kv_producer", False, False, "producer registration is incomplete"),
        ("kv_consumer", False, False, "consumer registration is incomplete"),
        ("kv_both", True, False, "consumer registration is incomplete"),
        ("kv_both", False, True, "producer registration is incomplete"),
    ],
)
def test_incomplete_role_registration_is_never_published(
    role: str,
    producer: bool,
    consumer: bool,
    message: str,
) -> None:
    worker = _worker(role=role)
    _install_published_resources(worker, producer=producer, consumer=consumer)

    with pytest.raises(RuntimeError, match=message):
        worker._packed_write_handshake_pools()


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("role", "producer", "consumer", "message"),
    [
        ("kv_consumer", True, True, "inactive role owns.*producer"),
        ("kv_producer", True, True, "inactive role owns.*consumer"),
    ],
)
def test_inactive_role_resources_are_never_published(
    role: str,
    producer: bool,
    consumer: bool,
    message: str,
) -> None:
    worker = _worker(role=role)
    _install_published_resources(worker, producer=producer, consumer=consumer)

    with pytest.raises(RuntimeError, match=message):
        worker._packed_write_handshake_pools()


@pytest.mark.cpu_test
@pytest.mark.parametrize("pool", ["producer", "consumer"])
def test_stale_registration_generation_is_never_published(pool: str) -> None:
    worker = _worker(role="kv_both")
    _install_published_resources(worker, producer=True, consumer=True)
    if pool == "producer":
        worker._packed_write_producer_pool_geometry = PackedWriteProducerPoolGeometry(
            registration_generation="stale-registration",
            base_address=0x40_0000,
            registered_bytes=_SLOT_COUNT * _CHUNK_BYTES,
            slot_size_bytes=_CHUNK_BYTES,
            slot_count=_SLOT_COUNT,
            device_id=3,
            alignment_bytes=_ALIGNMENT_BYTES,
        )
    else:
        stale_consumer = _consumer_geometry()
        worker._packed_write_consumer_pool_geometry = PackedWriteConsumerPoolGeometry(
            registration_generation="stale-registration",
            base_address=stale_consumer.base_address,
            registered_bytes=stale_consumer.registered_bytes,
            slot_size_bytes=stale_consumer.slot_size_bytes,
            slot_count=stale_consumer.slot_count,
            source_tp_size=stale_consumer.source_tp_size,
            rank_stride_bytes=stale_consumer.rank_stride_bytes,
            device_id=stale_consumer.device_id,
            alignment_bytes=stale_consumer.alignment_bytes,
        )

    with pytest.raises(RuntimeError, match=f"{pool} registration is stale"):
        worker._packed_write_handshake_pools()


@pytest.mark.cpu_test
def test_noncanonical_registered_row_is_rejected_for_both_pool_roles() -> None:
    worker = _worker(role="kv_both")
    worker._region_rows = [
        _FakeTensor(
            address=0x20_0000,
            size=4096,
            device=torch.device("cpu"),
            dimensions=2,
        )
    ]

    with pytest.raises(RuntimeError, match="source region is not a canonical"):
        worker._initialize_packed_write_producer_pool()
    with pytest.raises(RuntimeError, match="destination region is not a canonical"):
        worker._initialize_packed_write_consumer_pool()

    worker.nixl_wrapper.get_reg_descs.assert_not_called()
    worker.nixl_wrapper.register_memory.assert_not_called()


@pytest.mark.cpu_test
def test_cross_layer_packed_storage_is_rejected_before_registration() -> None:
    worker = _worker()

    with pytest.raises(
        RuntimeError,
        match="incompatible with cross-layer packed KV storage",
    ):
        worker._register_packed_kv_cache(MagicMock())

    worker.nixl_wrapper.get_reg_descs.assert_not_called()
    worker.nixl_wrapper.register_memory.assert_not_called()
    assert worker._registered_descs == []
