# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import logging
import os
import threading
import time
import traceback
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import msgpack
import torch
import zmq

from vllm.config.kv_transfer import KVTransferConfig
from vllm.distributed.device_communicators.pynccl_wrapper import (
    NCCLLibrary,
    buffer_type,
    cudaStream_t,
    ncclComm_t,
    ncclDataTypeEnum,
)
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.tensor_memory_pool import (  # noqa: E501
    TensorMemoryPool,
)
from vllm.utils.network_utils import get_ip
from vllm.utils.torch_utils import current_stream

logger = logging.getLogger(__name__)

DEFAULT_MEM_POOL_SIZE_GB = 32
_MISSING = object()


@contextmanager
def set_p2p_nccl_context(num_channels: str):
    original_values: dict[str, Any] = {}
    env_vars = [
        "NCCL_MAX_NCHANNELS",
        "NCCL_MIN_NCHANNELS",
        "NCCL_CUMEM_ENABLE",
        "NCCL_BUFFSIZE",
        "NCCL_PROTO",  # LL,LL128,SIMPLE
        "NCCL_ALGO",  # RING,TREE
    ]

    for var in env_vars:
        original_values[var] = os.environ.get(var)

    logger.info("set_p2p_nccl_context, original_values: %s", original_values)

    try:
        os.environ["NCCL_MAX_NCHANNELS"] = num_channels
        os.environ["NCCL_MIN_NCHANNELS"] = num_channels
        os.environ["NCCL_CUMEM_ENABLE"] = "1"
        yield
    finally:
        for var in env_vars:
            if original_values[var] is not None:
                os.environ[var] = original_values[var]
            else:
                os.environ.pop(var, None)


@dataclass
class SendQueueItem:
    tensor_id: str
    remote_address: str
    tensor: torch.Tensor | None
    connect_only: bool = False
    enqueued_time_s: float = 0.0


class P2pNcclEngine:
    def __init__(
        self,
        local_rank: int,
        config: KVTransferConfig,
        hostname: str = "",
        port_offset: int = 0,
        library_path: str | None = None,
    ) -> None:
        self.config = config
        self.rank = port_offset
        self.local_rank = local_rank
        self.device = torch.device(f"cuda:{self.local_rank}")
        self.nccl = NCCLLibrary(library_path)

        if not hostname:
            hostname = get_ip()
        port = int(self.config.kv_port) + port_offset
        if port == 0:
            raise ValueError("Port cannot be 0")
        self._hostname = hostname
        self._port = port

        # Each card corresponds to a ZMQ address.
        self.zmq_address = f"{self._hostname}:{self._port}"

        # If `proxy_ip` or `proxy_port` is `""`,
        # then the ping thread will not be enabled.
        proxy_ip = self.config.get_from_extra_config("proxy_ip", "")
        proxy_port = self.config.get_from_extra_config("proxy_port", "")
        if proxy_ip == "" or proxy_port == "":
            self.proxy_address = ""
            self.http_address = ""
        else:
            self.proxy_address = proxy_ip + ":" + proxy_port
            # the `http_port` must be consistent with the port of OpenAI.
            http_port = self.config.get_from_extra_config("http_port", None)
            if http_port is None:
                example_cfg = {
                    "kv_connector": "P2pNcclConnector",
                    "kv_connector_extra_config": {"http_port": 8000},
                }
                example = (
                    f"--port=8000 --kv-transfer-config='{json.dumps(example_cfg)}'"
                )
                raise ValueError(
                    "kv_connector_extra_config.http_port is required. "
                    f"Example: {example}"
                )
            self.http_address = f"{self._hostname}:{http_port}"

        self.context = zmq.Context()
        self.router_socket = self.context.socket(zmq.ROUTER)
        self.router_socket.bind(f"tcp://{self.zmq_address}")

        self.poller = zmq.Poller()
        self.poller.register(self.router_socket, zmq.POLLIN)

        self.send_store_cv = threading.Condition()
        self.send_queue_cv = threading.Condition()
        self.recv_store_cv = threading.Condition()
        self.send_request_id_to_tensor_ids_lock = threading.Lock()
        self.pool_lock = threading.Lock()

        self.send_stream = torch.cuda.Stream()
        self.recv_stream = torch.cuda.Stream()

        mem_pool_size_gb = float(
            self.config.get_from_extra_config(
                "mem_pool_size_gb", DEFAULT_MEM_POOL_SIZE_GB
            )
        )
        self.pool = TensorMemoryPool(
            max_block_size=int(mem_pool_size_gb * 1024**3)
        )  # GB

        # The sending type includes tree mutually exclusive options:
        # PUT, GET, PUT_ASYNC.
        self.send_type = self.config.get_from_extra_config("send_type", "PUT_ASYNC")
        if self.send_type == "GET":
            # tensor_id: torch.Tensor
            self.send_store: dict[str, torch.Tensor] = {}
        else:
            # PUT or PUT_ASYNC
            # tensor_id: torch.Tensor
            self.send_queue: deque[SendQueueItem] = deque()
            self._active_send_count = 0
            self._prewarm_remote_addresses: set[str] = set()
            if self.send_type == "PUT_ASYNC":
                self._send_thread = threading.Thread(
                    target=self.send_async, daemon=True
                )
                self._send_thread.start()

        # tensor_id: torch.Tensor/(addr, dtype, shape)
        self.recv_store: dict[str, Any] = {}
        self.recv_request_id_to_tensor_ids: dict[str, set[str]] = {}
        self.send_request_id_to_tensor_ids: dict[str, set[str]] = {}
        self.recv_reserved_tensor_ids: set[str] = set()
        self.finished_request_ids: set[str] = set()
        self.finished_request_id_order: deque[str] = deque()
        self.socks: dict[str, Any] = {}  # remote_address: client socket
        self.comms: dict[str, Any] = {}  # remote_address: (ncclComm_t, rank)

        self.buffer_size = 0
        self.recv_inflight_bytes = 0
        self.buffer_size_threshold = float(self.config.kv_buffer_size)
        self.recv_admission_timeout_s = float(
            self.config.get_from_extra_config("recv_admission_timeout_s", "1")
        )
        self.get_store_wait_timeout_s = float(
            self.config.get_from_extra_config("get_store_wait_timeout_s", "30")
        )
        self.recv_cuda_free_margin_bytes = int(
            float(
                self.config.get_from_extra_config(
                    "recv_cuda_free_margin_bytes", str(512 * 1024**2)
                )
            )
        )
        self.recv_store_tensor_threshold = int(
            self.config.get_from_extra_config("recv_store_tensor_threshold", "0")
        )
        self.recv_store_request_threshold = int(
            self.config.get_from_extra_config("recv_store_request_threshold", "0")
        )
        self.finished_request_id_limit = int(
            self.config.get_from_extra_config("finished_request_id_limit", "4096")
        )
        self.recv_admission_rejections = 0
        self.recv_cleanup_misses = 0
        self.recv_spilled_tensors = 0
        self.recv_loaded_tensors = 0
        self.debug_timing_enabled = os.environ.get(
            "VLLM_P2P_NCCL_TIMING", "0"
        ) == "1"
        self.trace_events_enabled = os.environ.get(
            "VLLM_P2P_NCCL_TRACE", "0"
        ) == "1"
        self.request_timing_enabled = (
            os.environ.get("VLLM_P2P_NCCL_REQUEST_TIMING", "0") == "1"
            or self.trace_events_enabled
        )
        self.timing_lock = threading.Lock()
        self.timing_stats: dict[str, dict[str, float]] = {}
        self.request_timing_stats: dict[str, dict[str, dict[str, float]]] = {}

        self.nccl_num_channels = self.config.get_from_extra_config(
            "nccl_num_channels", "8"
        )

        self._listener_thread = threading.Thread(
            target=self.listen_for_requests, daemon=True
        )
        self._listener_thread.start()

        self._ping_thread = None
        if port_offset == 0 and self.proxy_address != "":
            self._ping_thread = threading.Thread(target=self.ping, daemon=True)
            self._ping_thread.start()

        logger.info(
            "💯P2pNcclEngine init, rank:%d, local_rank:%d, http_address:%s, "
            "zmq_address:%s, proxy_address:%s, send_type:%s, buffer_size_"
            "threshold:%.2f, nccl_num_channels:%s",
            self.rank,
            self.local_rank,
            self.http_address,
            self.zmq_address,
            self.proxy_address,
            self.send_type,
            self.buffer_size_threshold,
            self.nccl_num_channels,
        )

    def create_connect(self, remote_address: str | None = None):
        assert remote_address is not None
        if remote_address in self.socks and remote_address not in self.comms:
            stale_sock = self.socks.pop(remote_address)
            stale_sock.close(linger=0)
        if remote_address in self.comms and remote_address not in self.socks:
            self.comms.pop(remote_address, None)
        if remote_address in self.socks and remote_address in self.comms:
            return self.socks[remote_address], self.comms[remote_address]
        if remote_address not in self.socks:
            sock = self.context.socket(zmq.DEALER)
            try:
                sock.setsockopt_string(zmq.IDENTITY, self.zmq_address)
                sock.connect(f"tcp://{remote_address}")

                unique_id = self.nccl.ncclGetUniqueId()
                data = {"cmd": "NEW", "unique_id": bytes(unique_id.internal)}
                sock.send(msgpack.dumps(data))

                with torch.accelerator.device_index(self.device.index):
                    rank = 0
                    with set_p2p_nccl_context(self.nccl_num_channels):
                        comm: ncclComm_t = self.nccl.ncclCommInitRank(
                            2, unique_id, rank
                        )
                    self.socks[remote_address] = sock
                    self.comms[remote_address] = (comm, rank)
                    logger.info(
                        "🤝ncclCommInitRank Success, %s👉%s, MyRank:%s",
                        self.zmq_address,
                        remote_address,
                        rank,
                    )
            except Exception:
                sock.close(linger=0)
                self.socks.pop(remote_address, None)
                self.comms.pop(remote_address, None)
                raise

        if remote_address not in self.comms:
            self.socks.pop(remote_address).close(linger=0)
            raise RuntimeError(
                "P2P NCCL connection missing communicator for "
                f"remote_address:{remote_address}"
            )

        return self.socks[remote_address], self.comms[remote_address]

    def send_tensor(
        self,
        tensor_id: str,
        tensor: torch.Tensor,
        remote_address: str | None = None,
    ) -> bool:
        if tensor.is_contiguous() is False:
            contiguous_started = time.perf_counter()
            tensor = tensor.contiguous()
            self._record_timing(
                "send_tensor_contiguous",
                time.perf_counter() - contiguous_started,
                self._tensor_nbytes(tensor),
            )

        if remote_address is None:
            with self.recv_store_cv:
                self.recv_store[tensor_id] = tensor
                self.recv_store_cv.notify()
            return True

        item = SendQueueItem(
            tensor_id=tensor_id,
            remote_address=remote_address,
            tensor=tensor,
            enqueued_time_s=time.perf_counter(),
        )

        if self.send_type == "PUT":
            return self.send_sync(item)

        if self.send_type == "PUT_ASYNC":
            with self.send_queue_cv:
                queue_entries_before = len(self.send_queue)
                self.send_queue.append(item)
                self.send_queue_cv.notify()
                queue_entries_after = len(self.send_queue)
            self._emit_trace_event(
                "put_async_enqueue",
                tensor_id=tensor_id,
                remote_address=remote_address,
                nbytes=self._tensor_nbytes(tensor),
                extra={
                    "queue_entries_before": queue_entries_before,
                    "queue_entries_after": queue_entries_after,
                },
            )
            return True

        # GET
        with self.send_store_cv:
            tensor_size = tensor.element_size() * tensor.numel()
            if tensor_size > self.buffer_size_threshold:
                logger.warning(
                    "❗[GET]tensor_id:%s, tensor_size:%d, is greater than"
                    "buffer size threshold :%d, skip send to %s, rank:%d",
                    tensor_id,
                    tensor_size,
                    self.buffer_size_threshold,
                    remote_address,
                    self.rank,
                )
                return False
            while self.buffer_size + tensor_size > self.buffer_size_threshold:
                assert len(self.send_store) > 0
                oldest_tensor_id = next(iter(self.send_store))
                oldest_tensor = self.send_store.pop(oldest_tensor_id)
                self.forget_sent_tensor_id(oldest_tensor_id)
                oldest_tensor_size = (
                    oldest_tensor.element_size() * oldest_tensor.numel()
                )
                self.buffer_size -= oldest_tensor_size
                logger.debug(
                    "⛔[GET]Send to %s, tensor_id:%s, tensor_size:%d,"
                    " buffer_size:%d, oldest_tensor_size:%d, rank:%d",
                    remote_address,
                    tensor_id,
                    tensor_size,
                    self.buffer_size,
                    oldest_tensor_size,
                    self.rank,
                )
                if self.debug_timing_enabled:
                    logger.info(
                        "P2P NCCL GET evicted stored tensor, rank:%d, "
                        "evicted_tensor_id:%s, requested_tensor_id:%s, "
                        "oldest_tensor_size:%d, buffer_size:%d, entries:%d",
                        self.rank,
                        oldest_tensor_id,
                        tensor_id,
                        oldest_tensor_size,
                        self.buffer_size,
                        len(self.send_store),
                    )

            self.send_store[tensor_id] = tensor
            self.buffer_size += tensor_size
            self.have_sent_tensor_id(tensor_id)
            self.send_store_cv.notify_all()
            if self.debug_timing_enabled:
                logger.info(
                    "P2P NCCL GET stored tensor, rank:%d, tensor_id:%s, "
                    "tensor_size:%d, buffer_size:%d, entries:%d",
                    self.rank,
                    tensor_id,
                    tensor_size,
                    self.buffer_size,
                    len(self.send_store),
                )
            logger.debug(
                "🔵[GET]Send to %s, tensor_id:%s, tensor_size:%d, "
                "shape:%s, rank:%d, buffer_size:%d(%.2f%%)",
                remote_address,
                tensor_id,
                tensor_size,
                tensor.shape,
                self.rank,
                self.buffer_size,
                self.buffer_size / self.buffer_size_threshold * 100,
            )
        return True

    def prewarm_remote(self, remote_address: str) -> None:
        """Queue connection setup on the async send worker.

        :param remote_address: Remote P2P NCCL address.
        """

        if self.send_type != "PUT_ASYNC":
            return

        with self.send_queue_cv:
            if remote_address in self.socks:
                return
            if remote_address in self._prewarm_remote_addresses:
                return
            self._prewarm_remote_addresses.add(remote_address)
            self.send_queue.append(
                SendQueueItem(
                    tensor_id=f"__p2p_prewarm__#{remote_address}",
                    remote_address=remote_address,
                    tensor=None,
                    connect_only=True,
                    enqueued_time_s=time.perf_counter(),
                )
            )
            self.send_queue_cv.notify()

    @staticmethod
    def _tensor_nbytes(tensor: torch.Tensor) -> int:
        return tensor.element_size() * tensor.numel()

    @staticmethod
    def _metadata_tensor_nbytes(
        shape: list[int] | tuple[int, ...],
        dtype_name: str,
    ) -> int:
        dtype = getattr(torch, dtype_name)
        num_elements = 1
        for dim in shape:
            num_elements *= int(dim)
        return torch.empty((), dtype=dtype).element_size() * num_elements

    @staticmethod
    def _request_id_from_tensor_id(tensor_id: str) -> str:
        return tensor_id.split("#", 1)[0]

    def _record_timing(
        self,
        name: str,
        elapsed_s: float,
        nbytes: int = 0,
    ) -> None:
        if self.debug_timing_enabled is False:
            return
        with self.timing_lock:
            stats = self.timing_stats.setdefault(
                name,
                {
                    "count": 0.0,
                    "seconds": 0.0,
                    "max_seconds": 0.0,
                    "bytes": 0.0,
                },
            )
            stats["count"] += 1.0
            stats["seconds"] += elapsed_s
            stats["max_seconds"] = max(stats["max_seconds"], elapsed_s)
            stats["bytes"] += float(nbytes)

    def _record_request_timing(
        self,
        request_id: str,
        name: str,
        elapsed_s: float,
        nbytes: int = 0,
    ) -> None:
        if self.request_timing_enabled is False:
            return
        with self.timing_lock:
            request_stats = self.request_timing_stats.setdefault(request_id, {})
            stats = request_stats.setdefault(
                name,
                {
                    "count": 0.0,
                    "seconds": 0.0,
                    "max_seconds": 0.0,
                    "bytes": 0.0,
                },
            )
            stats["count"] += 1.0
            stats["seconds"] += elapsed_s
            stats["max_seconds"] = max(stats["max_seconds"], elapsed_s)
            stats["bytes"] += float(nbytes)

    def _record_request_timing_for_tensor(
        self,
        tensor_id: str,
        name: str,
        elapsed_s: float,
        nbytes: int = 0,
    ) -> None:
        self._record_request_timing(
            self._request_id_from_tensor_id(tensor_id),
            name,
            elapsed_s,
            nbytes,
        )

    def _emit_trace_event(
        self,
        event: str,
        tensor_id: str | None = None,
        remote_address: str | None = None,
        nbytes: int = 0,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if self.trace_events_enabled is False:
            return

        payload: dict[str, Any] = {
            "event": event,
            "rank": self.rank,
            "time_ns": time.time_ns(),
        }
        if tensor_id is not None:
            payload["tensor_id"] = tensor_id
            payload["request_id"] = self._request_id_from_tensor_id(tensor_id)
        if remote_address is not None:
            payload["remote_address"] = remote_address
        if nbytes > 0:
            payload["bytes"] = nbytes
        if extra is not None:
            payload.update(extra)

        logger.info("P2P NCCL trace event %s", json.dumps(payload, sort_keys=True))

    def _pop_timing_stats(self) -> dict[str, dict[str, float]]:
        if self.debug_timing_enabled is False:
            return {}
        with self.timing_lock:
            stats = self.timing_stats
            self.timing_stats = {}
        return stats

    def _pop_request_timing_stats(self) -> dict[str, dict[str, dict[str, float]]]:
        if self.request_timing_enabled is False:
            return {}
        with self.timing_lock:
            stats = self.request_timing_stats
            self.request_timing_stats = {}
        return stats

    def _log_request_timing_stats(
        self,
        reason: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        stats = self._pop_request_timing_stats()
        if len(stats) == 0:
            return
        for request_id, request_stats in sorted(stats.items()):
            payload: dict[str, Any] = {
                "event": "p2p_engine_request_timing_stats",
                "rank": self.rank,
                "reason": reason,
                "request_id": request_id,
                "send_type": self.send_type,
                "stats": request_stats,
                "time_ns": time.time_ns(),
            }
            if extra is not None:
                payload.update(extra)
            logger.info(
                "P2P NCCL engine request timing stats %s",
                json.dumps(payload, sort_keys=True),
            )

    def _mark_request_finished_locked(self, request_id: str) -> None:
        if request_id in self.finished_request_ids:
            return
        self.finished_request_ids.add(request_id)
        self.finished_request_id_order.append(request_id)
        while (
            self.finished_request_id_limit > 0
            and len(self.finished_request_id_order) > self.finished_request_id_limit
        ):
            expired_request_id = self.finished_request_id_order.popleft()
            self.finished_request_ids.discard(expired_request_id)

    def _forget_tensor_id(
        self,
        request_id_to_tensor_ids: dict[str, set[str]],
        tensor_id: str,
    ) -> None:
        request_id = self._request_id_from_tensor_id(tensor_id)
        tensor_ids = request_id_to_tensor_ids.get(request_id)
        if tensor_ids is None:
            return
        tensor_ids.discard(tensor_id)
        if len(tensor_ids) == 0:
            request_id_to_tensor_ids.pop(request_id, None)

    def _pop_recv_store_locked(self, tensor_id: str, record_miss: bool = True) -> Any:
        tensor = self.recv_store.pop(tensor_id, _MISSING)
        if tensor is _MISSING:
            if record_miss:
                self.recv_cleanup_misses += 1
            return _MISSING

        self._forget_tensor_id(self.recv_request_id_to_tensor_ids, tensor_id)
        if tensor is not None and isinstance(tensor, tuple) is False:
            self.buffer_size = max(
                0, self.buffer_size - self._tensor_nbytes(tensor)
            )
        self.recv_store_cv.notify_all()
        return tensor

    def _load_pooled_tensor(
        self,
        tensor: tuple[int, torch.dtype, torch.Size],
    ) -> torch.Tensor:
        addr, dtype, shape = tensor
        with self.pool_lock:
            loaded_tensor = self.pool.load_tensor(addr, dtype, shape, self.device)
            self.pool.free(addr)
        return loaded_tensor

    def _store_failed_recv_tensor(self, tensor_id: str) -> None:
        with self.recv_store_cv:
            if tensor_id in self.recv_store:
                self.recv_store_cv.notify_all()
                return
            request_id = self._request_id_from_tensor_id(tensor_id)
            if request_id in self.finished_request_ids:
                self.recv_store_cv.notify_all()
                return
            self.recv_store[tensor_id] = None
            self.have_received_tensor_id(tensor_id)
            self.recv_store_cv.notify_all()

    def _current_free_cuda_bytes(self) -> int:
        free_bytes, _ = torch.cuda.mem_get_info(self.device)
        return int(free_bytes)

    def _has_recv_store_capacity_locked(self, tensor_id: str) -> bool:
        if tensor_id in self.recv_store:
            return False
        if tensor_id in self.recv_reserved_tensor_ids:
            return False

        request_id = self._request_id_from_tensor_id(tensor_id)
        if request_id in self.finished_request_ids:
            return False

        if (
            self.recv_store_tensor_threshold > 0
            and len(self.recv_store) >= self.recv_store_tensor_threshold
        ):
            return False

        if self.recv_store_request_threshold <= 0:
            return True

        if request_id in self.recv_request_id_to_tensor_ids:
            return True

        return len(self.recv_request_id_to_tensor_ids) < (
            self.recv_store_request_threshold
        )

    def _reserve_put_receive(self, tensor_id: str, tensor_size: int) -> bool:
        self._emit_trace_event(
            "put_recv_reserve_start",
            tensor_id=tensor_id,
            nbytes=tensor_size,
        )
        if tensor_size > self.buffer_size_threshold:
            logger.warning(
                "🔴[PUT]Recv Tensor, tensor exceeds buffer threshold, "
                "tensor_id:%s, tensor_size:%d, threshold:%d, rank:%d",
                tensor_id,
                tensor_size,
                int(self.buffer_size_threshold),
                self.rank,
            )
            self.recv_admission_rejections += 1
            return False

        deadline = time.time() + self.recv_admission_timeout_s
        with self.recv_store_cv:
            request_id = self._request_id_from_tensor_id(tensor_id)
            if request_id in self.finished_request_ids:
                logger.warning(
                    "🔴[PUT]Recv Tensor, request already finished, "
                    "tensor_id:%s, rank:%d",
                    tensor_id,
                    self.rank,
                )
                self.recv_admission_rejections += 1
                return False
            if tensor_id in self.recv_store:
                logger.warning(
                    "🔴[PUT]Recv Tensor, duplicate tensor id, tensor_id:%s, rank:%d",
                    tensor_id,
                    self.rank,
                )
                self.recv_admission_rejections += 1
                return False
            if tensor_id in self.recv_reserved_tensor_ids:
                logger.warning(
                    "🔴[PUT]Recv Tensor, duplicate tensor id, tensor_id:%s, rank:%d",
                    tensor_id,
                    self.rank,
                )
                self.recv_admission_rejections += 1
                return False

            while True:
                has_store_capacity = self._has_recv_store_capacity_locked(tensor_id)
                has_buffer_capacity = (
                    self.buffer_size + self.recv_inflight_bytes + tensor_size
                    <= self.buffer_size_threshold
                )
                has_cuda_capacity = (
                    self._current_free_cuda_bytes()
                    >= tensor_size + self.recv_cuda_free_margin_bytes
                )
                if has_store_capacity and has_buffer_capacity and has_cuda_capacity:
                    self.recv_inflight_bytes += tensor_size
                    self.recv_reserved_tensor_ids.add(tensor_id)
                    self._emit_trace_event(
                        "put_recv_reserved",
                        tensor_id=tensor_id,
                        nbytes=tensor_size,
                        extra={
                            "resident_bytes": self.buffer_size,
                            "inflight_bytes": self.recv_inflight_bytes,
                            "recv_store_entries": len(self.recv_store),
                            "recv_store_requests": len(
                                self.recv_request_id_to_tensor_ids
                            ),
                        },
                    )
                    return True

                remaining_s = deadline - time.time()
                if remaining_s <= 0:
                    logger.warning(
                        "🔴[PUT]Recv Tensor, admission timeout, "
                        "tensor_id:%s, tensor_size:%d, buffer_size:%d, "
                        "inflight:%d, store_entries:%d, store_requests:%d, "
                        "free_cuda:%d, rank:%d",
                        tensor_id,
                        tensor_size,
                        self.buffer_size,
                        self.recv_inflight_bytes,
                        len(self.recv_store),
                        len(self.recv_request_id_to_tensor_ids),
                        self._current_free_cuda_bytes(),
                        self.rank,
                    )
                    self.recv_admission_rejections += 1
                    return False

                self.recv_store_cv.wait(timeout=min(remaining_s, 0.05))

    def _finish_put_receive(
        self,
        tensor_id: str,
        tensor: torch.Tensor | None,
        tensor_size: int,
        remote_address: bytes,
        data: dict[str, Any],
    ) -> None:
        tensor_to_store: torch.Tensor | tuple[int, torch.dtype, torch.Size] | None
        tensor_to_store = tensor

        if tensor is not None:
            with self.recv_store_cv:
                request_id = self._request_id_from_tensor_id(tensor_id)
                request_finished = request_id in self.finished_request_ids
                should_spill_to_pool = (
                    request_finished is False
                    and self.buffer_size + tensor_size > self.buffer_size_threshold
                )

            if request_finished:
                tensor_to_store = None

            if should_spill_to_pool:
                try:
                    with self.pool_lock:
                        addr = self.pool.store_tensor(tensor)
                    tensor_to_store = (addr, tensor.dtype, tensor.shape)
                    self.recv_spilled_tensors += 1
                    logger.warning(
                        "🔴[PUT]Recv Tensor, Out Of Threshold, "
                        "%s👈%s, data:%s, addr:%d",
                        self.zmq_address,
                        remote_address.decode(),
                        data,
                        addr,
                    )
                except ValueError:
                    tensor_to_store = None
                    logger.error(
                        "🔴[PUT]Recv Tensor, failed to spill to memory pool, "
                        "%s👈%s, data:%s, traceback:%s",
                        self.zmq_address,
                        remote_address.decode(),
                        data,
                        traceback.format_exc(),
                    )
                except RuntimeError:
                    tensor_to_store = None
                    logger.error(
                        "🔴[PUT]Recv Tensor, failed to spill to memory pool, "
                        "%s👈%s, data:%s, traceback:%s",
                        self.zmq_address,
                        remote_address.decode(),
                        data,
                        traceback.format_exc(),
                    )

        with self.recv_store_cv:
            self.recv_inflight_bytes = max(
                0, self.recv_inflight_bytes - tensor_size
            )
            self.recv_reserved_tensor_ids.discard(tensor_id)
            request_id = self._request_id_from_tensor_id(tensor_id)
            if request_id in self.finished_request_ids:
                self.recv_store_cv.notify_all()
                return
            is_inline_tensor = (
                tensor_to_store is not None
                and isinstance(tensor_to_store, tuple) is False
            )
            if is_inline_tensor:
                self.buffer_size += tensor_size
            self.recv_store[tensor_id] = tensor_to_store
            self.have_received_tensor_id(tensor_id)
            self._emit_trace_event(
                "put_recv_stored",
                tensor_id=tensor_id,
                nbytes=tensor_size,
                extra={
                    "is_inline_tensor": is_inline_tensor,
                    "resident_bytes": self.buffer_size,
                    "inflight_bytes": self.recv_inflight_bytes,
                    "recv_store_entries": len(self.recv_store),
                },
            )
            self.recv_store_cv.notify_all()

    def has_tensors(self, tensor_ids: list[str]) -> bool:
        if self.send_type != "PUT" and self.send_type != "PUT_ASYNC":
            return True
        with self.recv_store_cv:
            for tensor_id in tensor_ids:
                if tensor_id not in self.recv_store:
                    return False
        return True

    def recv_tensor(
        self,
        tensor_id: str,
        remote_address: str | None = None,
    ) -> torch.Tensor:
        if self.send_type == "PUT" or self.send_type == "PUT_ASYNC":
            start_time = time.perf_counter()
            self._emit_trace_event(
                "recv_tensor_wait_start",
                tensor_id=tensor_id,
                remote_address=remote_address,
            )
            with self.recv_store_cv:
                while tensor_id not in self.recv_store:
                    self.recv_store_cv.wait()
                tensor = self._pop_recv_store_locked(tensor_id)
            wait_s = time.perf_counter() - start_time
            self._record_timing("recv_tensor_wait", wait_s)
            self._record_request_timing_for_tensor(
                tensor_id,
                "recv_tensor_wait",
                wait_s,
            )
            self._emit_trace_event(
                "recv_tensor_wait_end",
                tensor_id=tensor_id,
                remote_address=remote_address,
                extra={
                    "pooled_tensor": isinstance(tensor, tuple),
                    "tensor_available": tensor is not None,
                    "wait_s": wait_s,
                },
            )

            if tensor is not None:
                if isinstance(tensor, tuple):
                    load_started = time.perf_counter()
                    tensor = self._load_pooled_tensor(tensor)
                    load_elapsed_s = time.perf_counter() - load_started
                    self._record_timing(
                        "recv_tensor_pool_load",
                        load_elapsed_s,
                        self._tensor_nbytes(tensor),
                    )
                    self._record_request_timing_for_tensor(
                        tensor_id,
                        "recv_tensor_pool_load",
                        load_elapsed_s,
                        self._tensor_nbytes(tensor),
                    )
                self.recv_loaded_tensors += 1
            else:
                duration = time.perf_counter() - start_time
                logger.warning(
                    "🔴[PUT]Recv From %s, tensor_id:%s, duration:%.3fms, rank:%d",
                    remote_address,
                    tensor_id,
                    duration * 1000,
                    self.rank,
                )
            return tensor

        # GET
        if remote_address is None:
            return None

        if remote_address not in self.socks:
            self.create_connect(remote_address)

        sock = self.socks[remote_address]
        comm, rank = self.comms[remote_address]

        data = {"cmd": "GET", "tensor_id": tensor_id}
        sock.send(msgpack.dumps(data))

        message = sock.recv()
        data = msgpack.loads(message)
        if data["ret"] != 0:
            logger.warning(
                "🔴[GET]Recv From %s, tensor_id: %s, ret: %d",
                remote_address,
                tensor_id,
                data["ret"],
            )
            return None

        with torch.cuda.stream(self.recv_stream):
            tensor = torch.empty(
                data["shape"], dtype=getattr(torch, data["dtype"]), device=self.device
            )

        self.recv(comm, tensor, rank ^ 1, self.recv_stream, tensor_id=tensor_id)

        return tensor

    def listen_for_requests(self):
        while True:
            socks = dict(self.poller.poll())
            if self.router_socket not in socks:
                continue

            remote_address, message = self.router_socket.recv_multipart()
            data = msgpack.loads(message)
            if data["cmd"] == "NEW":
                unique_id = self.nccl.unique_id_from_bytes(bytes(data["unique_id"]))
                with torch.accelerator.device_index(self.device.index):
                    rank = 1
                    with set_p2p_nccl_context(self.nccl_num_channels):
                        comm: ncclComm_t = self.nccl.ncclCommInitRank(
                            2, unique_id, rank
                        )
                    self.comms[remote_address.decode()] = (comm, rank)
                    logger.info(
                        "🤝ncclCommInitRank Success, %s👈%s, MyRank:%s",
                        self.zmq_address,
                        remote_address.decode(),
                        rank,
                    )
            elif data["cmd"] == "PUT":
                tensor_id = data["tensor_id"]
                tensor_size = self._metadata_tensor_nbytes(
                    data["shape"], data["dtype"]
                )
                self._emit_trace_event(
                    "put_metadata_received",
                    tensor_id=tensor_id,
                    remote_address=remote_address.decode(),
                    nbytes=tensor_size,
                )
                reserve_started = time.perf_counter()
                reserved = self._reserve_put_receive(tensor_id, tensor_size)
                self._record_request_timing_for_tensor(
                    tensor_id,
                    "put_recv_reserve",
                    time.perf_counter() - reserve_started,
                    tensor_size,
                )
                if reserved is False:
                    self.router_socket.send_multipart([remote_address, b"2"])
                    self._store_failed_recv_tensor(tensor_id)
                    continue

                ack_sent = False
                try:
                    alloc_started = time.perf_counter()
                    with torch.cuda.stream(self.recv_stream):
                        tensor = torch.empty(
                            data["shape"],
                            dtype=getattr(torch, data["dtype"]),
                            device=self.device,
                        )
                    self._record_request_timing_for_tensor(
                        tensor_id,
                        "put_recv_alloc",
                        time.perf_counter() - alloc_started,
                        tensor_size,
                    )
                    self.router_socket.send_multipart([remote_address, b"0"])
                    ack_sent = True
                    self._emit_trace_event(
                        "put_ack_sent",
                        tensor_id=tensor_id,
                        remote_address=remote_address.decode(),
                        nbytes=tensor_size,
                    )
                    comm, rank = self.comms[remote_address.decode()]
                    self.recv(
                        comm,
                        tensor,
                        rank ^ 1,
                        self.recv_stream,
                        tensor_id=tensor_id,
                    )
                    self._emit_trace_event(
                        "put_recv_nccl_done",
                        tensor_id=tensor_id,
                        remote_address=remote_address.decode(),
                        nbytes=tensor_size,
                    )

                except torch.cuda.OutOfMemoryError:
                    if ack_sent is False:
                        self.router_socket.send_multipart([remote_address, b"1"])
                    tensor = None
                    logger.warning(
                        "🔴[PUT]Recv Tensor, Out Of Memory, %s👈%s, data:%s",
                        self.zmq_address,
                        remote_address.decode(),
                        data,
                    )
                except RuntimeError:
                    if ack_sent is False:
                        self.router_socket.send_multipart([remote_address, b"3"])
                    tensor = None
                    logger.error(
                        "🔴[PUT]Recv Tensor, RuntimeError, %s👈%s, data:%s, "
                        "traceback:%s",
                        self.zmq_address,
                        remote_address.decode(),
                        data,
                        traceback.format_exc(),
                    )

                self._finish_put_receive(
                    tensor_id=tensor_id,
                    tensor=tensor,
                    tensor_size=tensor_size,
                    remote_address=remote_address,
                    data=data,
                )

            elif data["cmd"] == "GET":
                tensor_id = data["tensor_id"]
                with self.send_store_cv:
                    wait_started = time.perf_counter()
                    while True:
                        tensor = self.send_store.pop(tensor_id, None)
                        if tensor is not None:
                            break
                        remaining_s = (
                            self.get_store_wait_timeout_s
                            - (time.perf_counter() - wait_started)
                        )
                        if remaining_s <= 0:
                            break
                        self.send_store_cv.wait(timeout=min(remaining_s, 0.05))
                    if tensor is not None:
                        tensor_size = self._tensor_nbytes(tensor)
                        self.buffer_size = max(0, self.buffer_size - tensor_size)
                        self.forget_sent_tensor_id(tensor_id)
                        data = {
                            "ret": 0,
                            "shape": tensor.shape,
                            "dtype": str(tensor.dtype).replace("torch.", ""),
                        }
                        self._record_timing(
                            "get_store_wait",
                            time.perf_counter() - wait_started,
                            tensor_size,
                        )
                        if self.debug_timing_enabled:
                            logger.info(
                                "P2P NCCL GET served tensor, rank:%d, "
                                "tensor_id:%s, tensor_size:%d, buffer_size:%d, "
                                "send_store_entries:%d",
                                self.rank,
                                tensor_id,
                                tensor_size,
                                self.buffer_size,
                                len(self.send_store),
                            )
                    else:
                        request_id = self._request_id_from_tensor_id(tensor_id)
                        same_request_tensor_ids = [
                            stored_tensor_id
                            for stored_tensor_id in self.send_store
                            if stored_tensor_id.startswith(request_id + "#")
                        ]
                        sample_tensor_ids = list(self.send_store.keys())[:5]
                        logger.warning(
                            "P2P NCCL GET missing tensor, rank:%d, "
                            "tensor_id:%s, send_store_entries:%d, "
                            "same_request_entries:%d, sample_tensor_ids:%s",
                            self.rank,
                            tensor_id,
                            len(self.send_store),
                            len(same_request_tensor_ids),
                            sample_tensor_ids,
                        )
                        self._record_timing(
                            "get_store_timeout",
                            time.perf_counter() - wait_started,
                        )
                        data = {"ret": 1}

                self.router_socket.send_multipart([remote_address, msgpack.dumps(data)])

                if data["ret"] == 0:
                    comm, rank = self.comms[remote_address.decode()]
                    self.send(
                        comm,
                        tensor.to(self.device),
                        rank ^ 1,
                        self.send_stream,
                        tensor_id=tensor_id,
                    )
            else:
                logger.warning(
                    "🚧Unexpected, Received message from %s, data:%s",
                    remote_address,
                    data,
                )

    def have_sent_tensor_id(self, tensor_id: str):
        request_id = tensor_id.split("#")[0]
        with self.send_request_id_to_tensor_ids_lock:
            if request_id not in self.send_request_id_to_tensor_ids:
                self.send_request_id_to_tensor_ids[request_id] = set()
            self.send_request_id_to_tensor_ids[request_id].add(tensor_id)

    def forget_sent_tensor_id(self, tensor_id: str):
        with self.send_request_id_to_tensor_ids_lock:
            self._forget_tensor_id(self.send_request_id_to_tensor_ids, tensor_id)

    def have_received_tensor_id(self, tensor_id: str):
        request_id = tensor_id.split("#")[0]
        if request_id not in self.recv_request_id_to_tensor_ids:
            self.recv_request_id_to_tensor_ids[request_id] = set()
        self.recv_request_id_to_tensor_ids[request_id].add(tensor_id)

    def send_async(self):
        while True:
            with self.send_queue_cv:
                while not self.send_queue:
                    self.send_queue_cv.wait()
                item = self.send_queue.popleft()
                self._active_send_count += 1
                queue_entries_after = len(self.send_queue)
            if item.connect_only:
                try:
                    self._emit_trace_event(
                        "put_async_prewarm_start",
                        tensor_id=item.tensor_id,
                        remote_address=item.remote_address,
                    )
                    self.create_connect(item.remote_address)
                    self._emit_trace_event(
                        "put_async_prewarm_done",
                        tensor_id=item.tensor_id,
                        remote_address=item.remote_address,
                    )
                except Exception:
                    logger.error(
                        "P2P NCCL prewarm failed, remote_address:%s, "
                        "traceback:%s",
                        item.remote_address,
                        traceback.format_exc(),
                    )
                    with self.send_queue_cv:
                        self._prewarm_remote_addresses.discard(item.remote_address)
                finally:
                    with self.send_queue_cv:
                        self._active_send_count -= 1
                        if not self.send_queue and self._active_send_count == 0:
                            self.send_queue_cv.notify_all()
                continue

            assert item.tensor is not None
            self._emit_trace_event(
                "put_async_dequeue",
                tensor_id=item.tensor_id,
                remote_address=item.remote_address,
                nbytes=self._tensor_nbytes(item.tensor),
                extra={
                    "active_send_count": self._active_send_count,
                    "queue_entries_after": queue_entries_after,
                },
            )
            if item.enqueued_time_s > 0:
                self._record_request_timing_for_tensor(
                    item.tensor_id,
                    "put_async_queue_wait",
                    time.perf_counter() - item.enqueued_time_s,
                    self._tensor_nbytes(item.tensor),
                )
            try:
                self.send_sync(item)
            finally:
                with self.send_queue_cv:
                    self._active_send_count -= 1
                    if not self.send_queue and self._active_send_count == 0:
                        self.send_queue_cv.notify_all()

    def wait_for_sent(self):
        if self.send_type == "PUT_ASYNC":
            if os.environ.get("VLLM_P2P_NCCL_SKIP_WAIT_FOR_SENT", "0") == "1":
                self._record_timing("wait_for_sent_skipped", 0.0)
                return
            start_time = time.perf_counter()
            with self.send_queue_cv:
                while self.send_queue or self._active_send_count > 0:
                    self.send_queue_cv.wait()
            duration = time.perf_counter() - start_time
            self._record_timing("wait_for_sent", duration)
            self._log_request_timing_stats(
                "wait_for_sent",
                extra={"wait_for_sent_seconds": duration},
            )
            logger.debug(
                "🚧[PUT_ASYNC]It took %.3fms to wait for the send_queue"
                " to be empty, rank:%d",
                duration * 1000,
                self.rank,
            )

    def send_sync(self, item: SendQueueItem) -> bool:
        if item.remote_address is None:
            return False
        if item.remote_address not in self.socks:
            self.create_connect(item.remote_address)

        tensor = item.tensor
        assert tensor is not None

        sock = self.socks[item.remote_address]
        comm, rank = self.comms[item.remote_address]
        data = {
            "cmd": "PUT",
            "tensor_id": item.tensor_id,
            "shape": tensor.shape,
            "dtype": str(tensor.dtype).replace("torch.", ""),
        }
        sync_started = time.perf_counter()
        tensor_size = self._tensor_nbytes(tensor)
        self._emit_trace_event(
            "put_send_start",
            tensor_id=item.tensor_id,
            remote_address=item.remote_address,
            nbytes=tensor_size,
            extra={"send_type": self.send_type},
        )
        sock.send(msgpack.dumps(data))
        self._emit_trace_event(
            "put_metadata_sent",
            tensor_id=item.tensor_id,
            remote_address=item.remote_address,
            nbytes=tensor_size,
        )

        ack_started = time.perf_counter()
        response = sock.recv()
        ack_wait_s = time.perf_counter() - ack_started
        self._record_timing(
            "send_sync_ack_wait",
            ack_wait_s,
        )
        self._record_request_timing_for_tensor(
            item.tensor_id,
            "send_sync_ack_wait",
            ack_wait_s,
            tensor_size,
        )
        self._emit_trace_event(
            "put_ack_received",
            tensor_id=item.tensor_id,
            remote_address=item.remote_address,
            nbytes=tensor_size,
            extra={"ack_wait_s": ack_wait_s, "response": response.decode()},
        )
        if response != b"0":
            logger.error(
                "🔴Send Tensor, Peer Out Of Memory/Threshold, %s 👉 %s, "
                "MyRank:%s, data:%s, tensor:%s, size:%fGB, response:%s",
                self.zmq_address,
                item.remote_address,
                rank,
                data,
                tensor.shape,
                tensor.element_size() * tensor.numel() / 1024**3,
                response.decode(),
            )
            return False

        to_device_started = time.perf_counter()
        tensor_on_device = tensor.to(self.device)
        self._record_request_timing_for_tensor(
            item.tensor_id,
            "send_tensor_to_device",
            time.perf_counter() - to_device_started,
            tensor_size,
        )
        self.send(
            comm,
            tensor_on_device,
            rank ^ 1,
            self.send_stream,
            tensor_id=item.tensor_id,
        )
        total_s = time.perf_counter() - sync_started
        self._emit_trace_event(
            "put_send_done",
            tensor_id=item.tensor_id,
            remote_address=item.remote_address,
            nbytes=tensor_size,
            extra={"send_sync_total_s": total_s},
        )
        self._record_timing(
            "send_sync_total",
            total_s,
            tensor_size,
        )
        self._record_request_timing_for_tensor(
            item.tensor_id,
            "send_sync_total",
            total_s,
            tensor_size,
        )

        if self.send_type == "PUT_ASYNC":
            self.have_sent_tensor_id(item.tensor_id)

        return True

    def get_finished(
        self, finished_req_ids: set[str], no_compile_layers
    ) -> tuple[set[str] | None, set[str] | None]:
        """
        Notifies worker-side connector ids of requests that have
        finished generating tokens.

        Returns:
            ids of requests that have finished asynchronous transfer,
            tuple of (sending/saving ids, recving/loading ids).
            The finished saves/sends req ids must belong to a set provided in a
            call to this method (this call or a prior one).
        """

        recv_tensors_to_free = []
        recv_cleanup_ids: set[str] = set()
        recv_fallback_cleanup_ids: set[str] = set()
        send_cleanup_ids: set[str] = set()

        for request_id in finished_req_ids:
            request_recv_cleanup_ids: set[str] = set()
            with self.recv_store_cv:
                self._mark_request_finished_locked(request_id)
                request_recv_cleanup_ids.update(
                    self.recv_request_id_to_tensor_ids.pop(request_id, set())
                )
                recv_cleanup_ids.update(request_recv_cleanup_ids)
            if self.send_type != "GET":
                with self.send_request_id_to_tensor_ids_lock:
                    send_cleanup_ids.update(
                        self.send_request_id_to_tensor_ids.pop(request_id, set())
                    )
            if len(request_recv_cleanup_ids) == 0:
                for layer_name in no_compile_layers:
                    recv_fallback_cleanup_ids.add(request_id + "#" + layer_name)

        with self.recv_store_cv:
            for tensor_id in recv_cleanup_ids:
                tensor = self._pop_recv_store_locked(tensor_id)
                if tensor is not _MISSING and isinstance(tensor, tuple):
                    recv_tensors_to_free.append(tensor)
            for tensor_id in recv_fallback_cleanup_ids:
                tensor = self._pop_recv_store_locked(tensor_id, record_miss=False)
                if tensor is not _MISSING and isinstance(tensor, tuple):
                    recv_tensors_to_free.append(tensor)

        with self.send_store_cv:
            if self.send_type == "GET":
                for tensor_id in send_cleanup_ids:
                    tensor = self.send_store.pop(tensor_id, None)
                    if tensor is not None:
                        self.buffer_size = max(
                            0,
                            self.buffer_size - self._tensor_nbytes(tensor),
                        )
                if len(send_cleanup_ids) > 0:
                    self.send_store_cv.notify_all()

        if len(recv_tensors_to_free) > 0:
            with self.pool_lock:
                for tensor in recv_tensors_to_free:
                    addr, _, _ = tensor
                    self.pool.free(addr)

        if (
            self.recv_admission_rejections > 0
            or self.recv_cleanup_misses > 0
            or self.recv_spilled_tensors > 0
        ):
            logger.warning(
                "P2P NCCL receive counters, rank:%d, loaded:%d, "
                "admission_rejections:%d, cleanup_misses:%d, spilled:%d, "
                "resident_bytes:%d, inflight_bytes:%d, recv_store_entries:%d",
                self.rank,
                self.recv_loaded_tensors,
                self.recv_admission_rejections,
                self.recv_cleanup_misses,
                self.recv_spilled_tensors,
                self.buffer_size,
                self.recv_inflight_bytes,
                len(self.recv_store),
            )

        timing_stats = self._pop_timing_stats()
        if len(timing_stats) > 0:
            with self.recv_store_cv:
                state = {
                    "resident_bytes": self.buffer_size,
                    "inflight_bytes": self.recv_inflight_bytes,
                    "recv_store_entries": len(self.recv_store),
                    "reserved_tensor_ids": len(self.recv_reserved_tensor_ids),
                    "finished_request_ids": len(self.finished_request_ids),
                    "recv_loaded_tensors": self.recv_loaded_tensors,
                    "recv_admission_rejections": self.recv_admission_rejections,
                    "recv_cleanup_misses": self.recv_cleanup_misses,
                    "recv_spilled_tensors": self.recv_spilled_tensors,
                }
            if self.send_type == "GET":
                with self.send_store_cv:
                    state["send_store_entries"] = len(self.send_store)
                    state["send_queue_entries"] = 0
                    state["active_send_count"] = 0
            else:
                with self.send_queue_cv:
                    state["send_store_entries"] = 0
                    state["send_queue_entries"] = len(self.send_queue)
                    state["active_send_count"] = self._active_send_count
            logger.info(
                "P2P NCCL timing stats, rank:%d, state:%s, stats:%s",
                self.rank,
                json.dumps(state, sort_keys=True),
                json.dumps(timing_stats, sort_keys=True),
            )
        self._log_request_timing_stats("get_finished")

        # TODO:Retrieve requests that have already sent the KV cache.
        finished_sending: set[str] = set()

        # TODO:Retrieve requests that have already received the KV cache.
        finished_recving: set[str] = set()

        return finished_sending or None, finished_recving or None

    def ping(self):
        sock = self.context.socket(zmq.DEALER)
        sock.setsockopt_string(zmq.IDENTITY, self.zmq_address)
        logger.debug("ping start, zmq_address:%s", self.zmq_address)
        sock.connect(f"tcp://{self.proxy_address}")
        data = {
            "type": "P" if self.config.is_kv_producer else "D",
            "http_address": self.http_address,
            "zmq_address": self.zmq_address,
        }
        while True:
            sock.send(msgpack.dumps(data))
            time.sleep(3)

    def send(
        self,
        comm,
        tensor: torch.Tensor,
        dst: int,
        stream=None,
        tensor_id: str | None = None,
    ):
        assert tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}"
        )
        if stream is None:
            stream = current_stream()

        send_started = time.perf_counter()
        with torch.cuda.stream(stream):
            self.nccl.ncclSend(
                buffer_type(tensor.data_ptr()),
                tensor.numel(),
                ncclDataTypeEnum.from_torch(tensor.dtype),
                dst,
                comm,
                cudaStream_t(stream.cuda_stream),
            )
        stream.synchronize()
        send_elapsed_s = time.perf_counter() - send_started
        self._record_timing(
            "nccl_send",
            send_elapsed_s,
            self._tensor_nbytes(tensor),
        )
        if tensor_id is not None:
            self._record_request_timing_for_tensor(
                tensor_id,
                "nccl_send",
                send_elapsed_s,
                self._tensor_nbytes(tensor),
            )

    def recv(
        self,
        comm,
        tensor: torch.Tensor,
        src: int,
        stream=None,
        tensor_id: str | None = None,
    ):
        assert tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}"
        )
        if stream is None:
            stream = current_stream()

        recv_started = time.perf_counter()
        with torch.cuda.stream(stream):
            self.nccl.ncclRecv(
                buffer_type(tensor.data_ptr()),
                tensor.numel(),
                ncclDataTypeEnum.from_torch(tensor.dtype),
                src,
                comm,
                cudaStream_t(stream.cuda_stream),
            )
        stream.synchronize()
        recv_elapsed_s = time.perf_counter() - recv_started
        self._record_timing(
            "nccl_recv",
            recv_elapsed_s,
            self._tensor_nbytes(tensor),
        )
        if tensor_id is not None:
            self._record_request_timing_for_tensor(
                tensor_id,
                "nccl_recv",
                recv_elapsed_s,
                self._tensor_nbytes(tensor),
            )

    def close(self) -> None:
        self._listener_thread.join()
        if self.send_type == "PUT_ASYNC":
            self._send_thread.join()
        if self._ping_thread is not None:
            self._ping_thread.join()
