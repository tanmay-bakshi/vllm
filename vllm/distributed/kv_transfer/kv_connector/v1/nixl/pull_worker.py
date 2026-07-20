# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pull-specific (READ) worker-side logic for the NIXL connector."""

import os
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import msgspec
import numpy as np
import torch
from zmq.constants import SocketOption, SocketType
from zmq.error import ZMQError

from vllm.distributed.kv_transfer.coalesced_layout import (
    CanonicalSourceTransferPlan,
    CoalescedTransferPlan,
    GroupTransferRoster,
    PackedTransferPlan,
    RegionOwnership,
    SourceGroupTransferRoster,
    SourceRegionOwnership,
    bind_coalesced_transfer_destinations,
    build_canonical_source_plan,
    build_packed_transfer_plan,
)
from vllm.distributed.kv_transfer.integrity import IntegrityIdentity
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import (
    NixlBaseConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.coalesced_pack import (
    PackEnqueueError,
    PackLaunch,
    launch_packed_chunk_pack,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.coalesced_scatter import (
    ScatterEnqueueError,
    ScatterLaunch,
    launch_packed_chunk_scatter,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    PACKED_WRITE_ARRIVED_PREFIX,
    PACKED_WRITE_FAILED_BEFORE_WRITE_PREFIX,
    PACKED_WRITE_REQUEST_PREFIX,
    PACKED_WRITE_SOURCE_DRAINED_PREFIX,
    PULL_OFFER_CANCELLATION_CONTROL_PREFIX,
    PULL_READ_COMPLETE_PREFIX,
    NixlConnectorMetadata,
    PackedWriteArrived,
    PackedWriteChunkIdentity,
    PackedWriteCommand,
    PackedWriteConsumerEndpoint,
    PackedWriteFailedBeforeWrite,
    PackedWriteFailureCode,
    PackedWriteRequest,
    PackedWriteSourceDrained,
    PackedWriteSourceSelection,
    ProducerLease,
    PullOfferCancellationAck,
    PullOfferCancellationControl,
    PullOfferCancelled,
    PullReadComplete,
    RemoteOfferKey,
    ReqMeta,
    decode_packed_write_arrived_notification,
    decode_packed_write_failed_before_write_notification,
    decode_packed_write_request_notification,
    decode_packed_write_source_drained_notification,
    encode_packed_write_arrived_notification,
    encode_packed_write_failed_before_write_notification,
    encode_packed_write_request_notification,
    encode_packed_write_source_drained_notification,
    packed_write_destination_address,
    validate_packed_write_arrived,
    validate_packed_write_failed_before_write,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.packed_write_plan import (
    ProducerPackedWritePlan,
    reconstruct_producer_packed_write_plan,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.stats import (
    NixlPackedWriteTelemetry,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import (
    ReadSpec,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.utils import zmq_ctx
from vllm.distributed.kv_transfer.nixl_localization import (
    LocalizationError,
    NixlEventRecord,
    NixlPlanPosition,
    NixlPlanRecord,
    NixlPlanRun,
    NixlRegionPlan,
    NixlSourceContract,
    localization_producer_target,
    localization_request_target,
    validate_source_contract_structure,
)
from vllm.distributed.kv_transfer.packed_write_ownership import (
    ConsumerSlotLease,
    PackedWriteChunkKey,
    PackedWriteConsumerPool,
    PackedWriteProducerPool,
    PackedWriteRequestKey,
    PackedWriteRequestTracker,
    ProducerSlotLease,
)
from vllm.distributed.kv_transfer.staging_ownership import (
    CoalescedStagingPlan,
    HandleState,
    StagingSafetyError,
)
from vllm.distributed.nixl_utils import canonicalize_nixl_agent_name
from vllm.logger import init_logger
from vllm.utils.network_utils import make_zmq_path
from vllm.v1.outputs import KVTransferFailureReason

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)

_MAX_BUFFERED_PULL_COMPLETIONS_PER_REQUEST = 65_536
_MAX_CANCELLED_REMOTE_OFFERS = 65_536
_MAX_COMPLETED_PULL_CONTRACTS = 65_536
_MAX_PACKED_WRITE_CONSUMER_TERMINALS = 65_536
_PACKED_WRITE_CONSUMER_TERMINAL_TRIM = 16_384
_OFFER_CANCELLATION_TIMEOUT_MS = 1_000
PullContractTerminalMode = Literal["read_complete", "offer_cancelled"]
PackedDrainRecordResult = Literal["waiting", "recorded", "completed"]


@dataclass(frozen=True, slots=True)
class _AuthenticatedPackedSourceDrained:
    """One packed source-drain proof bound to its native sender."""

    drained: PackedWriteSourceDrained
    consumer_agent: str


@dataclass(frozen=True, slots=True)
class _PackedCompletionBinding:
    """Request-level packed completion identity accepted by one producer rank."""

    request_key: PackedWriteRequestKey
    consumer_endpoint: PackedWriteConsumerEndpoint
    consumer_agent: str


@dataclass
class PullCompletionState:
    """Exact terminal-proof obligations for one producer-side lease.

    :ivar expected_consumers: Logical parallel-sampling consumer count.
    :ivar consumer_tp_size: Decoder tensor-parallel size.
    :ivar expected_read_completions: Exact child/rank read proofs required locally.
    :ivar expected_offer_cancellations: Exact decoder-rank cancellation proofs.
    :ivar read_completions: Valid child/rank read proofs observed so far.
    :ivar offer_cancellations: Valid whole-offer cancellation proofs observed.
    :ivar has_mixed_terminal_proofs: Whether both terminal modes were observed.
    """

    expected_consumers: int
    consumer_tp_size: int
    expected_read_completions: frozenset[tuple[int, int]]
    expected_offer_cancellations: frozenset[int]
    offer_generation: int | None = None
    read_completions: set[tuple[int, int]] = field(default_factory=set)
    offer_cancellations: set[int] = field(default_factory=set)
    packed_completions: dict[tuple[int, int], _AuthenticatedPackedSourceDrained] = (
        field(default_factory=dict)
    )
    has_mixed_terminal_proofs: bool = False


@dataclass(frozen=True)
class CompletedPullContract:
    """A released producer contract retained for idempotency checks.

    :ivar expected_consumers: Logical parallel-sampling consumer count.
    :ivar consumer_tp_size: Decoder tensor-parallel size.
    :ivar terminal_mode: Proof mode that authorized the release.
    """

    expected_consumers: int
    consumer_tp_size: int
    offer_generation: int | None
    terminal_mode: PullContractTerminalMode
    packed_completions: frozenset[_AuthenticatedPackedSourceDrained] = frozenset()
    packed_bindings: frozenset[_PackedCompletionBinding] = frozenset()


@dataclass(slots=True)
class _ProducerPackedOperation:
    """Own one producer gather and its subsequent native WRITE.

    :ivar request: Authenticated decoder request for this producer rank.
    :ivar source_plan: Reconstructed source-only plan.
    :ivar packed_plan: Reconstructed bounded chunk plan.
    :ivar lease: Generation-scoped local producer slot ownership.
    :ivar pack_launch: CUDA gather completion owner.
    :ivar consumer_agent: Imported decoder agent and WRITE destination.
    :ivar created_at: Monotonic admission time.
    :ivar stage_started_at: Monotonic start of the current device/native stage.
    :ivar write_handle: Native WRITE handle after safe preparation.
    :ivar write_status: Latest native status after the post boundary.
    :ivar pack_failure_reason: Known quiescent pack failure, if any.
    :ivar warning_emitted: Whether the live-operation age warning was emitted.
    """

    request: PackedWriteRequest
    source_plan: CanonicalSourceTransferPlan
    packed_plan: PackedTransferPlan
    lease: ProducerSlotLease
    pack_launch: PackLaunch
    consumer_agent: str
    created_at: float
    stage_started_at: float
    write_handle: Any | None = None
    write_status: str | None = None
    pack_failure_reason: str | None = None
    warning_emitted: bool = False


@dataclass(slots=True)
class _ProducerPackedPending:
    """Park one authenticated request until its source and a slot exist.

    :ivar request: Exact decoder request.
    :ivar consumer_agent: Authenticated decoder agent.
    :ivar received_at: Monotonic request receipt time.
    :ivar warning_emitted: Whether the bounded roster-wait warning was emitted.
    :ivar reconstructed: Validated source plan cached across pool backpressure.
    """

    request: PackedWriteRequest
    consumer_agent: str
    received_at: float
    warning_emitted: bool = False
    reconstructed: ProducerPackedWritePlan | None = None


@dataclass(slots=True)
class _ConsumerPackedTerminal:
    """Retain one exact producer terminal after decoder slot retirement.

    :ivar request: Published producer-rank request.
    :ivar terminal: Exact ARRIVED or FAILED_BEFORE_WRITE proof.
    """

    request: PackedWriteRequest
    terminal: PackedWriteArrived | PackedWriteFailedBeforeWrite


@dataclass(slots=True)
class _ConsumerPackedChunk:
    """Own one decoder chunk from request publication through scatter.

    :ivar request_id: Decoder request identity.
    :ivar chunk_index: Canonical bounded-chunk ordinal.
    :ivar requests: Exact producer-rank request quorum.
    :ivar lease: Generation-scoped decoder staging lease.
    :ivar admitted_at: Monotonic decoder slot admission time.
    :ivar terminal_at: Time the complete producer terminal quorum arrived.
    :ivar scatter_launch: Event-owned scatter launch, once enqueued.
    :ivar scatter_enqueued_at: Monotonic scatter enqueue-complete time.
    :ivar scatter_enqueue_duration_seconds: CPU enqueue duration.
    :ivar failure_reason: Known quiescent scatter failure, if any.
    :ivar warning_emitted: Whether the live-operation age warning was emitted.
    """

    request_id: str
    chunk_index: int
    requests: tuple[PackedWriteRequest, ...]
    lease: ConsumerSlotLease
    admitted_at: float
    terminals_by_rank: dict[int, PackedWriteArrived | PackedWriteFailedBeforeWrite] = (
        field(default_factory=dict)
    )
    terminal_at: float | None = None
    scatter_launch: ScatterLaunch | None = None
    scatter_enqueued_at: float | None = None
    scatter_enqueue_duration_seconds: float = 0.0
    failure_reason: str | None = None
    warning_emitted: bool = False


@dataclass(slots=True)
class _ConsumerPackedRequest:
    """Gate atomic publication across all chunks of one packed write.

    :ivar request_id: Decoder request identity.
    :ivar meta: Immutable scheduler transfer metadata.
    :ivar read_specs: Canonical producer-rank transfer specifications.
    :ivar source_plan: Destination-independent selected source plan.
    :ivar destination_plan: Exact decoder placement plan.
    :ivar packed_plan: Bounded chunk sequence.
    :ivar identities_by_chunk: Rankful identities awaiting slot binding.
    :ivar tracker: Atomic request publication state.
    :ivar direct_descriptor_count: Equivalent direct descriptors across ranks.
    :ivar logical_bytes: Unpruned logical transfer bytes.
    :ivar created_at: First packed request admission time.
    :ivar failure_reason: Known request-scoped failure being retired.
    :ivar source_selections: Complete canonical source suffixes for every command.
    """

    request_id: str
    meta: ReqMeta
    read_specs: list[ReadSpec]
    source_plan: CanonicalSourceTransferPlan
    destination_plan: CoalescedTransferPlan
    packed_plan: PackedTransferPlan
    consumer_endpoint: PackedWriteConsumerEndpoint
    identities_by_chunk: tuple[tuple[PackedWriteChunkIdentity, ...], ...]
    source_selections: tuple[PackedWriteSourceSelection, ...]
    tracker: PackedWriteRequestTracker
    direct_descriptor_count: int
    logical_bytes: int
    created_at: float
    release_commands_by_producer_rank: dict[int, PackedWriteCommand] = field(
        default_factory=dict
    )
    published_chunk_count: int = 0
    pending_chunk_indices: deque[int] = field(default_factory=deque)
    active_chunks: dict[int, _ConsumerPackedChunk] = field(default_factory=dict)
    arrival_wait_seconds: float = 0.0
    scatter_enqueue_duration_seconds: float = 0.0
    scatter_wall_duration_seconds: float = 0.0
    scatter_gpu_duration_seconds: float = 0.0
    scatter_gpu_duration_available: bool = True
    warning_emitted: bool = False
    failure_reason: str | None = None


def _parallel_consumer_index(request_id: str, expected_consumers: int) -> int:
    """Derive the stable child index encoded by parallel sampling.

    :param request_id: Decoder-side request identifier.
    :param expected_consumers: Producer-owned logical consumer count.
    :returns: Parallel-sampling child index.
    :raises ValueError: If request lineage and the consumer contract disagree.
    """
    prefix, separator, _ = request_id.partition("_")
    has_child_index = len(separator) > 0 and prefix.isdigit()
    if expected_consumers == 1:
        if has_child_index:
            raise ValueError(
                f"request {request_id} is a parallel-sampling child but its "
                "producer contract expects one consumer"
            )
        return 0
    if has_child_index is False:
        raise ValueError(
            f"request {request_id} lacks a parallel-sampling child index for "
            f"{expected_consumers} consumers"
        )
    consumer_index = int(prefix)
    if consumer_index >= expected_consumers:
        raise ValueError(
            f"request {request_id} child index {consumer_index} is outside its "
            f"{expected_consumers}-consumer contract"
        )
    return consumer_index


def _consumer_ranks_for_producer(
    producer_rank: int,
    producer_tp_size: int,
    consumer_tp_size: int,
) -> tuple[int, ...]:
    """Return decoder ranks whose reads can target one producer rank.

    :param producer_rank: Local producer tensor-parallel rank.
    :param producer_tp_size: Producer tensor-parallel size.
    :param consumer_tp_size: Decoder tensor-parallel size.
    :returns: Exact decoder ranks assigned to the producer rank.
    :raises ValueError: If the tensor-parallel sizes are incompatible.
    """
    if consumer_tp_size >= producer_tp_size:
        if consumer_tp_size % producer_tp_size != 0:
            raise ValueError("consumer TP must be divisible by producer TP")
        ratio = consumer_tp_size // producer_tp_size
        start = producer_rank * ratio
        return tuple(range(start, start + ratio))
    if producer_tp_size % consumer_tp_size != 0:
        raise ValueError("producer TP must be divisible by consumer TP")
    ratio = producer_tp_size // consumer_tp_size
    return (producer_rank // ratio,)


class NixlPullConnectorWorker(NixlBaseConnectorWorker):
    """Pull-connector worker logic."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        engine_id: str,
        kv_cache_config: "KVCacheConfig",
    ) -> None:
        super().__init__(vllm_config, engine_id, kv_cache_config)
        self._pull_completion_states: dict[str, PullCompletionState] = {}
        self._buffered_pull_completions: dict[str, set[PullReadComplete]] = {}
        self._buffered_packed_source_drained: dict[
            str, set[_AuthenticatedPackedSourceDrained]
        ] = {}
        self._buffered_offer_cancellations: dict[str, set[PullOfferCancelled]] = {}
        self._pending_offer_cancellations: list[PullOfferCancelled] = []
        self._completed_pull_contracts: dict[
            tuple[str, int | None], CompletedPullContract
        ] = {}
        self._local_source_retired_through = 0
        self._cancelled_remote_offers: set[RemoteOfferKey] = set()
        self._packed_producer_pool: PackedWriteProducerPool | None = None
        self._packed_producer_pending: deque[_ProducerPackedPending] = deque()
        self._packed_producer_operations: dict[
            PackedWriteChunkIdentity, _ProducerPackedOperation
        ] = {}
        self._packed_completion_bindings: dict[
            tuple[str, int, str, int], _PackedCompletionBinding
        ] = {}
        self._packed_source_ready_events: dict[str, torch.cuda.Event] = {}
        self._packed_consumer_agents: dict[PackedWriteConsumerEndpoint, str] = {}
        self._packed_consumer_agent_last_active: dict[
            PackedWriteConsumerEndpoint, float
        ] = {}
        self._packed_consumer_pool: PackedWriteConsumerPool | None = None
        self._packed_consumer_requests: dict[str, _ConsumerPackedRequest] = {}
        self._packed_consumer_chunks: dict[
            PackedWriteChunkKey, _ConsumerPackedChunk
        ] = {}
        self._packed_terminal_notifications: deque[
            tuple[PackedWriteArrived | PackedWriteFailedBeforeWrite, str]
        ] = deque()
        self._packed_consumer_terminals: dict[
            PackedWriteChunkIdentity, _ConsumerPackedTerminal
        ] = {}
        self._packed_done_recving: set[str] = set()
        if self._phase_separate_transfer_decode and not self.coalesce_pull:
            raise ValueError("phase_separate_transfer_decode requires coalesced pull")
        if self._phase_separate_transfer_decode and not self._no_stock_dma():
            raise ValueError(
                "phase_separate_transfer_decode requires stock DMA to be disabled"
            )
        if (
            self._packed_write_config.enabled
            and self.kv_transfer_config.kv_role == "kv_consumer"
            and self._phase_separate_transfer_decode is False
        ):
            raise ValueError(
                "packed write consumers require phase_separate_transfer_decode"
            )

    def start_load_kv(self, metadata: NixlConnectorMetadata) -> None:
        """Start and account for receive work required by this model step.

        :param metadata: Scheduler metadata for transfers entering this step.
        """
        self._apply_local_source_retirement(metadata.source_retired_through)
        for meta in metadata.reqs_to_recv.values():
            if meta.remote is None:
                continue
            self._apply_remote_source_retirement(
                meta.remote.engine_id,
                meta.remote.source_retired_through,
            )
        self._update_heartbeat_targets(metadata)
        self._service_heartbeats()
        self._begin_transfer_phase()
        self._audit_retire(metadata)
        self._capture_source_rosters(metadata.source_rosters)
        self._record_packed_source_readiness(metadata.source_rosters)
        self._service_packed_producer()
        for req_id, meta in metadata.reqs_to_recv.items():
            meta.local_physical_block_ids = self._logical_to_kernel_block_ids(
                meta.local_block_ids
            )
            assert meta.remote is not None
            # Remote block IDs are kept logical here; expanded in
            # _read_blocks_for_req using the remote engine's phys ratio.
            remote_engine_id = meta.remote.engine_id
            logger.debug(
                "start_load_kv for request %s from remote engine %s. "
                "Num local_block_ids: %s. Num remote_block_ids: %s. ",
                req_id,
                remote_engine_id,
                len(meta.local_physical_block_ids),
                len(meta.remote.block_ids),
            )
            # always store metadata for failure recovery
            self._recving_metadata[req_id] = meta
            if remote_engine_id not in self._remote_agents:
                # Initiate handshake with remote engine to exchange metadata.
                with self._handshake_lock:
                    if remote_engine_id not in self._remote_agents:
                        self._background_nixl_handshake(req_id, remote_engine_id, meta)
                        continue

            # Handshake already completed, start async read xfer.
            self._read_blocks_for_req(req_id, meta)

        # Start transfers for requests whose handshakes have now finished.
        while not self._ready_requests.empty():
            self._read_blocks_for_req(*self._ready_requests.get_nowait())

        # Keep around the requests that have been part of a batch. This is
        # needed because async scheduling pushes the misalignment between the
        # moment in which requests expiration is set (P side) and the moment in
        # which blocks are read from D. As P can now more easily lag behind D
        # while processing the next batch, we make sure to only set an
        # expiration for requests that have not been read from D yet.
        for req_id in metadata.reqs_in_batch:
            self._reqs_to_process.add(req_id)

        # Remove all requests that are not to be processed (eg aborted).
        for req_id in metadata.reqs_not_processed:
            self._reqs_to_process.discard(req_id)
            self._buffered_pull_completions.pop(req_id, None)
            self._buffered_offer_cancellations.pop(req_id, None)
            pre_read_plan = self._localization_pre_read_plans.get(req_id)
            producer_engine_id: str | None = None
            producer_request_id: str | None = None
            if pre_read_plan is not None:
                contracts = pre_read_plan.source_contracts
                if len(contracts) > 0:
                    producer_engine_id = contracts[0].producer_engine_id
                    producer_request_id = contracts[0].producer_request_id
            self._localization_record_event(
                code="REQUEST_ABORTED",
                evidentiary=False,
                child_request_id=req_id,
                producer_engine_id=producer_engine_id,
                producer_request_id=producer_request_id,
                detail="request aborted before complete first-read capture",
            )
            self._localization_pre_read_plans.pop(req_id, None)
            # We should never get an abort after setting an expiry timer
            assert req_id not in self._reqs_to_send

        # Add to requests that are waiting to be read and track expiration.
        for req_id, lease in metadata.reqs_to_send.items():
            if req_id in self._reqs_to_process:
                self._install_pull_completion_state(req_id, lease)
                self._reqs_to_send[req_id] = lease.deadline

        self._pending_offer_cancellations.extend(
            metadata.offer_cancellations_by_rank.get(self.tp_rank, ())
        )

        self._drain_transfer_phase()
        self._localization_capture_pre_read(metadata.scheduled_request_ids)
        self._record_transfer_decode_boundary()

    def _install_pull_completion_state(
        self,
        request_id: str,
        lease: ProducerLease,
    ) -> None:
        """Install the immutable completion obligations for a new lease.

        :param request_id: Producer request that owns the source pages.
        :param lease: Producer-owned lifetime and consumer contract.
        """
        if lease.expected_consumers < 1 or lease.consumer_tp_size < 1:
            raise ValueError(f"request {request_id} has an invalid producer lease")
        consumer_ranks = _consumer_ranks_for_producer(
            self.tp_rank,
            self.world_size,
            lease.consumer_tp_size,
        )
        read_completions = frozenset(
            (consumer_index, consumer_rank)
            for consumer_index in range(lease.expected_consumers)
            for consumer_rank in consumer_ranks
        )
        offer_cancellations = frozenset(consumer_ranks)
        existing = self._pull_completion_states.get(request_id)
        roster = self._source_rosters.get(request_id)
        offer_generation = None if roster is None else roster.offer_generation
        if existing is not None:
            if (
                existing.expected_consumers != lease.expected_consumers
                or existing.consumer_tp_size != lease.consumer_tp_size
                or existing.expected_read_completions != read_completions
                or existing.expected_offer_cancellations != offer_cancellations
                or existing.offer_generation != offer_generation
            ):
                raise RuntimeError(
                    f"request {request_id} changed its producer completion contract"
                )
            return
        self._pull_completion_states[request_id] = PullCompletionState(
            expected_consumers=lease.expected_consumers,
            consumer_tp_size=lease.consumer_tp_size,
            expected_read_completions=read_completions,
            expected_offer_cancellations=offer_cancellations,
            offer_generation=offer_generation,
        )

    def _apply_local_source_retirement(self, retired_through: int) -> None:
        """Prune producer histories only after scheduler block retirement.

        :param retired_through: Highest gap-free local source generation returned
            to the scheduler allocator.
        """
        if type(retired_through) is not int or retired_through < 0:
            raise ValueError("local source retirement floor must be non-negative")
        if retired_through <= self._local_source_retired_through:
            return
        if any(
            state.offer_generation is not None
            and state.offer_generation <= retired_through
            for state in self._pull_completion_states.values()
        ):
            raise StagingSafetyError(
                "source retirement floor crossed an active producer contract"
            )
        self._local_source_retired_through = retired_through
        self._completed_pull_contracts = {
            key: contract
            for key, contract in self._completed_pull_contracts.items()
            if key[1] is None or key[1] > retired_through
        }

    def request_rejected_before_admission(
        self,
        request_id: str,
        kv_transfer_params: dict[str, Any],
        reason: str,
    ) -> bool:
        """Cancel a producer offer that this decoder never admitted.

        The serving request identifier and reason are local diagnostics. The
        producer receives only an exact rank proof over its immutable offer.

        :param request_id: Decoder serving request rejected before admission.
        :param kv_transfer_params: Producer-authored transfer contract.
        :param reason: Local rejection reason.
        :returns: Whether the producer control plane accepted the proof.
        """
        try:
            (
                producer_engine_id,
                producer_request_id,
                producer_host,
                producer_port,
                offer_generation,
                source_retired_through,
                expected_consumers,
                target_ranks,
            ) = self._offer_cancellation_contract(kv_transfer_params)
        except ValueError:
            logger.error(
                "Cannot cancel rejected request %s because its producer offer "
                "contract is invalid\n%s",
                request_id,
                traceback.format_exc(),
            )
            return False

        retirement_floor = self._apply_remote_source_retirement(
            producer_engine_id,
            source_retired_through,
        )
        if offer_generation <= retirement_floor:
            logger.info(
                "Skipping cancellation for producer offer %s/%s generation %d; "
                "the producer retirement floor is %d",
                producer_engine_id,
                producer_request_id,
                offer_generation,
                retirement_floor,
            )
            return True

        active_children = self._active_children_for_offer(
            producer_engine_id,
            producer_request_id,
            offer_generation,
        )
        offer_key: RemoteOfferKey = (
            producer_engine_id,
            producer_request_id,
            offer_generation,
        )
        has_completed_read = (
            offer_key in self._remote_offer_completion_counts
            or offer_key in self._released_remote_offers
        )
        if len(active_children) > 0 or has_completed_read:
            logger.error(
                "Refusing whole-offer cancellation for rejected request %s: "
                "producer offer %s/%s has decoder read state (active=%s, "
                "completed=%s)",
                request_id,
                producer_engine_id,
                producer_request_id,
                active_children,
                has_completed_read,
            )
            return False

        proof = PullOfferCancelled(
            producer_request_id=producer_request_id,
            offer_generation=offer_generation,
            consumer_rank=self.tp_rank,
            consumer_tp_size=self.world_size,
            expected_consumers=expected_consumers,
        )
        if (
            self._fence_remote_offer(
                producer_engine_id,
                producer_request_id,
                offer_generation,
            )
            is False
        ):
            return False
        logger.info(
            "Cancelling producer offer %s/%s after pre-admission rejection of "
            "decoder request %s: %s",
            producer_engine_id,
            producer_request_id,
            request_id,
            reason,
        )

        return self._send_offer_cancellation_control(
            proof,
            producer_engine_id,
            producer_host,
            producer_port,
            target_ranks,
        )

    def _offer_cancellation_contract(
        self,
        params: dict[str, Any],
    ) -> tuple[str, str, str, int, int, int, int, tuple[int, ...]]:
        """Validate and materialize one producer offer cancellation contract.

        :param params: Producer-authored transfer parameters.
        :returns: Producer address, immutable contract, and local target ranks.
        :raises ValueError: If the offer cannot be cancelled exactly.
        """
        if params.get("do_remote_prefill") is not True:
            raise ValueError("request is not an unconsumed remote-prefill offer")
        producer_engine_id = params.get("remote_engine_id")
        producer_request_id = params.get("remote_request_id")
        producer_host = params.get("remote_host")
        producer_port = params.get("remote_port")
        producer_tp_size = params.get("tp_size")
        offer_generation = params.get("source_offer_generation")
        source_retired_through = params.get("source_retired_through", 0)
        expected_consumers = params.get("expected_consumers")
        consumer_tp_size = params.get("consumer_tp_size")
        for field_name, value in (
            ("remote_engine_id", producer_engine_id),
            ("remote_request_id", producer_request_id),
            ("remote_host", producer_host),
        ):
            if type(value) is not str or len(value) == 0:
                raise ValueError(f"{field_name} must be a non-empty string")
        if type(producer_port) is not int or producer_port < 1:
            raise ValueError("remote_port must be a positive integer")
        if type(producer_tp_size) is not int or producer_tp_size < 1:
            raise ValueError("tp_size must be a positive integer")
        if type(offer_generation) is not int or offer_generation < 1:
            raise ValueError("source_offer_generation must be a positive integer")
        if (
            type(source_retired_through) is not int
            or source_retired_through < 0
            or source_retired_through >= offer_generation
        ):
            raise ValueError("source retirement floor must precede its offer")
        if type(expected_consumers) is not int or expected_consumers < 1:
            raise ValueError("expected_consumers must be a positive integer")
        if type(consumer_tp_size) is not int or consumer_tp_size < 1:
            raise ValueError("consumer_tp_size must be a positive integer")
        if consumer_tp_size != self.world_size:
            raise ValueError(
                "producer consumer_tp_size differs from the local decoder topology"
            )
        if self.tp_rank < 0 or self.tp_rank >= self.world_size:
            raise ValueError("local decoder rank is outside its topology")

        target_ranks = tuple(
            producer_rank
            for producer_rank in range(producer_tp_size)
            if self.tp_rank
            in _consumer_ranks_for_producer(
                producer_rank,
                producer_tp_size,
                self.world_size,
            )
        )
        if len(target_ranks) == 0:
            raise ValueError("decoder rank has no producer cancellation target")
        assert isinstance(producer_engine_id, str)
        assert isinstance(producer_request_id, str)
        assert isinstance(producer_host, str)
        assert isinstance(producer_port, int)
        assert isinstance(producer_tp_size, int)
        assert isinstance(offer_generation, int)
        assert isinstance(source_retired_through, int)
        assert isinstance(expected_consumers, int)
        return (
            producer_engine_id,
            producer_request_id,
            producer_host,
            producer_port,
            offer_generation,
            source_retired_through,
            expected_consumers,
            target_ranks,
        )

    def _apply_remote_source_retirement(
        self,
        producer_engine_id: str,
        retired_through: int,
    ) -> int:
        """Apply a producer floor and compact only proven-retired fences.

        :param producer_engine_id: Producer process identity.
        :param retired_through: Producer-committed contiguous generation floor.
        :returns: Monotonic local floor for the producer.
        """
        floor = self._advance_remote_source_retirement_floor(
            producer_engine_id,
            retired_through,
        )
        self._cancelled_remote_offers = {
            key
            for key in self._cancelled_remote_offers
            if key[0] != producer_engine_id or key[2] is None or key[2] > floor
        }
        return floor

    def _active_children_for_offer(
        self,
        producer_engine_id: str,
        producer_request_id: str,
        offer_generation: int,
    ) -> tuple[str, ...]:
        """Return every admitted or in-flight child for a producer offer.

        :param producer_engine_id: Producer engine identity.
        :param producer_request_id: Producer request identity.
        :param offer_generation: Producer allocation generation.
        :returns: Matching decoder child identifiers.
        """
        return tuple(
            child_request_id
            for child_request_id, meta in self._recving_metadata.items()
            if meta.remote is not None
            and meta.remote.offer_key
            == (producer_engine_id, producer_request_id, offer_generation)
        )

    def _fence_remote_offer(
        self,
        producer_engine_id: str,
        producer_request_id: str,
        offer_generation: int | None,
    ) -> bool:
        """Fence a cancelled offer against every later decoder read.

        :param producer_engine_id: Producer engine identity.
        :param producer_request_id: Producer request identity.
        :param offer_generation: Producer allocation generation.
        :returns: Whether the permanent local fence is installed.
        """
        key: RemoteOfferKey = (
            producer_engine_id,
            producer_request_id,
            offer_generation,
        )
        if self._remote_offer_retired_by_floor(key):
            return True
        if key in self._cancelled_remote_offers:
            return True
        if len(self._cancelled_remote_offers) >= _MAX_CANCELLED_REMOTE_OFFERS:
            logger.error(
                "Cannot cancel producer offer %s/%s because the permanent local "
                "fence table is full; producer source pages remain pinned",
                producer_engine_id,
                producer_request_id,
            )
            return False
        self._cancelled_remote_offers.add(key)
        return True

    def _send_offer_cancellation_control(
        self,
        proof: PullOfferCancelled,
        producer_engine_id: str,
        producer_host: str,
        producer_port: int,
        producer_ranks: tuple[int, ...],
    ) -> bool:
        """Queue a cancellation proof through the producer control plane.

        :param proof: Typed whole-offer cancellation proof.
        :param producer_engine_id: Producer engine identity.
        :param producer_host: Producer side-channel host.
        :param producer_port: Producer side-channel port.
        :param producer_ranks: Exact producer ranks covered by this proof.
        :returns: Whether the complete proof was atomically queued.
        """
        control = PullOfferCancellationControl(
            producer_ranks=producer_ranks,
            proof=proof,
        )
        message = PULL_OFFER_CANCELLATION_CONTROL_PREFIX + msgspec.msgpack.encode(
            control
        )
        path = make_zmq_path("tcp", producer_host, producer_port)
        try:
            with zmq_ctx(SocketType.REQ, path) as socket:
                socket.setsockopt(SocketOption.IMMEDIATE, 1)
                socket.setsockopt(SocketOption.SNDTIMEO, _OFFER_CANCELLATION_TIMEOUT_MS)
                socket.setsockopt(SocketOption.RCVTIMEO, _OFFER_CANCELLATION_TIMEOUT_MS)
                socket.send(message)
                response = socket.recv()
        except ZMQError as error:
            stacktrace = traceback.format_exc()
            self._log_failure(
                failure_type="offer_cancellation_control_failed",
                req_id=None,
                error=error,
                remote_engine_id=producer_engine_id,
                remote_request_id=proof.producer_request_id,
                remote_host=producer_host,
                remote_port=producer_port,
                stacktrace=stacktrace,
            )
            self.xfer_stats.record_failed_notification()
            return False

        try:
            ack = msgspec.msgpack.decode(response, type=PullOfferCancellationAck)
        except (msgspec.DecodeError, msgspec.ValidationError):
            logger.error(
                "Producer %s returned a malformed cancellation acknowledgement for "
                "offer %s; source pages remain pinned\n%s",
                producer_engine_id,
                proof.producer_request_id,
                traceback.format_exc(),
            )
            self.xfer_stats.record_failed_notification()
            return False

        if (
            ack.producer_request_id != proof.producer_request_id
            or ack.offer_generation != proof.offer_generation
            or ack.producer_ranks != producer_ranks
            or ack.accepted is False
        ):
            logger.error(
                "Producer %s rejected cancellation control for offer %s and "
                "ranks %s; source pages remain pinned",
                producer_engine_id,
                proof.producer_request_id,
                producer_ranks,
            )
            self.xfer_stats.record_failed_notification()
            return False
        return True

    def _read_blocks_for_req(self, req_id: str, meta: ReqMeta) -> None:
        assert meta.remote is not None
        remote_offer = meta.remote.offer_key
        retirement_floor = self._apply_remote_source_retirement(
            meta.remote.engine_id,
            meta.remote.source_retired_through,
        )
        if self._remote_offer_retired_by_floor(remote_offer):
            logger.error(
                "Refusing pull for %s because producer offer %s/%s generation "
                "%s is at or below retirement floor %d",
                req_id,
                meta.remote.engine_id,
                meta.remote.request_id,
                meta.remote.source_offer_generation,
                retirement_floor,
            )
            self._handle_failed_transfer(req_id, None)
            return
        if remote_offer in self._cancelled_remote_offers:
            logger.error(
                "Refusing pull for %s because producer offer %s/%s was "
                "cancelled before admission",
                req_id,
                meta.remote.engine_id,
                meta.remote.request_id,
            )
            self._handle_failed_transfer(req_id, None)
            return
        assert self.transfer_topo is not None
        localization_enabled = self._localization_config.enabled_for(req_id)
        engine_id = meta.remote.engine_id
        try:
            notification_id = self._read_completion_notification(req_id, meta)
        except ValueError as error:
            self._log_failure(
                failure_type="invalid_completion_contract",
                req_id=req_id,
                error=error,
                meta=meta,
            )
            self._handle_failed_transfer(req_id, None)
            return
        # Update last activity from this remote. Mind that cleanup is done on main
        # thread (this one), so we don't race on this structure.
        self._engine_last_active[engine_id] = time.perf_counter()
        plan = self.tp_mappings[engine_id]
        remote_info = self.transfer_topo.get_engine_info(engine_id)
        tp_ratio = self.transfer_topo.tp_ratio(remote_info.remote_tp_size)

        if (
            localization_enabled
            and sum(len(group) for group in meta.local_physical_block_ids) == 0
        ):
            if self._localization_writer is None:
                raise LocalizationError("enabled localization has no artifact writer")
            if req_id not in self._localization_zero_recorded:
                self._localization_writer.write(
                    NixlEventRecord(
                        record_type=NixlEventRecord.RECORD_TYPE,
                        schema_version=IntegrityIdentity.SCHEMA_VERSION,
                        run_id=self._localization_config.run_id,
                        transport_arm=self._localization_config.transport_arm,
                        code="NON_EVIDENTIARY_ZERO_BYTE",
                        evidentiary=False,
                        producer_engine_id=meta.remote.engine_id,
                        producer_request_id=meta.remote.request_id,
                        child_request_id=req_id,
                        observer_engine_id=self.engine_id,
                        observer_rank=self.tp_rank,
                        detail=(
                            "full-prefix hits have no transferred bytes and are "
                            "excluded from this localization protocol"
                        ),
                        created_ns=time.time_ns(),
                    )
                )
                self._localization_zero_recorded.add(req_id)
                self._localization_terminal_recorded.add(req_id)
            if self._localization_config.strict_zero_byte:
                raise LocalizationError(
                    f"full-prefix request {req_id} is non-evidentiary in "
                    "strict localization mode"
                )

        if (
            meta.remote.offer_key in self._released_remote_offers
            or self._remote_offer_retired_by_floor(meta.remote.offer_key)
        ) and sum(len(g) for g in meta.local_physical_block_ids) > 0:
            # Full-prefix-hit requests (empty local ids) read nothing
            # and are exempt from the fence.
            logger.error(
                "[release-fence] refusing pull for %s: remote request %s "
                "already released; failing (router retries with a fresh "
                "prefill).",
                req_id,
                meta.remote.request_id,
            )
            self._handle_failed_transfer(req_id, None)
            return

        meta.remote.block_ids = self._logical_to_remote_kernel_block_ids(
            meta.remote.block_ids,
            remote_info.remote_physical_blocks_per_logical,
        )
        remote_block_ids = meta.remote.block_ids
        local_block_ids = meta.local_physical_block_ids
        num_groups = len(local_block_ids)
        read_specs = [
            ReadSpec(
                remote_rank=rank,
                local_block_ids=[
                    list(local_block_ids[g])
                    if rank in plan.source_ranks_per_group[g]
                    else []
                    for g in range(num_groups)
                ],
                remote_block_ids=[
                    list(remote_block_ids[g])
                    if rank in plan.source_ranks_per_group[g]
                    else []
                    for g in range(num_groups)
                ],
            )
            for rank in plan.all_source_ranks
        ]

        # D may have to perform multiple reads from different remote ranks.
        # MLA opt: when P TP > D TP, only a single read is executed for
        # the first remote rank (cache is duplicated)..
        if self.use_mla and tp_ratio < 0:
            assert len(read_specs) == 1

        if localization_enabled and sum(len(group) for group in local_block_ids) == 0:
            result = self._coalesced_read_request(
                req_id,
                meta,
                read_specs,
                notification_id=notification_id,
            )
            if result != "posted":
                raise LocalizationError(
                    f"zero-byte request {req_id} did not complete its release path"
                )
            return

        # Coalesced pull fast path (see base_worker state comment): all
        # gates must hold. A staging-pool miss parks the request FIFO until
        # completed plans free staging. Localization forbids the stock path
        # because it has no staging or destination observation points.
        if self._coalesce_gate(engine_id, tp_ratio, read_specs):
            source_contracts: tuple[NixlSourceContract, ...] = ()
            if localization_enabled:
                spec0 = read_specs[0]
                raw_remote_groups = tuple(
                    tuple(int(block_id) for block_id in group)
                    for group in spec0.remote_block_ids
                )
                region_lengths = tuple(
                    int(length)
                    for length in self._remote_layout[engine_id][spec0.remote_rank][0]
                )
                source_contracts = self._localization_build_source_contracts(
                    req_id,
                    meta,
                    read_specs,
                    raw_remote_groups,
                    region_lengths,
                )
            res = self._coalesced_read_request(
                req_id,
                meta,
                read_specs,
                notification_id,
                source_contracts,
            )
            if res == "posted":
                return
            if res == "defer":
                self._coalesce_pending.append(
                    (req_id, meta, read_specs, source_contracts)
                )
                logger.debug(
                    "coalesced pull: parked %s for staging (%s queued)",
                    req_id,
                    len(self._coalesce_pending),
                )
                return

        if localization_enabled or any(self._sp_group_flags()) or self._no_stock_dma():
            # F2b: the stock per-descriptor path cannot express the
            # single-plane K-half/2:1 mapping -- running it would write
            # a dual layout into a single-plane cache. Fail the request
            # (kv_load_failure_policy=fail) rather than corrupt.
            # Phase 21 (VLLM_GEMMA4_NIXL_NO_STOCK_DMA=1): the stock path
            # also DMAs into local KV offsets past the NIXL/UCX
            # large-offset defect threshold (block ids >= ~32768) and
            # silently corrupts; fail instead -- the router retries via
            # a fresh prefill->decode pass.
            logger.error(
                "coalesced pull unavailable for %s (sp_groups=%s "
                "no_stock_dma=%s); failing the request instead of the "
                "stock path.",
                req_id,
                any(self._sp_group_flags()),
                self._no_stock_dma(),
            )
            self._handle_failed_transfer(req_id, None)
            return

        self._stock_read_specs(req_id, meta, read_specs, notification_id)

    def _read_completion_notification(self, req_id: str, meta: ReqMeta) -> bytes:
        """Encode one decoder-rank completion identity for a logical read.

        :param req_id: Decoder-side request identifier.
        :param meta: Remote producer metadata and immutable consumer contract.
        :returns: Typed NIXL completion notification.
        :raises ValueError: If the decoder does not match the producer contract.
        """
        if meta.remote is None:
            raise ValueError(f"request {req_id} has no remote producer metadata")
        if meta.remote.consumer_tp_size != self.world_size:
            raise ValueError(
                f"request {req_id} producer contract expects decoder TP "
                f"{meta.remote.consumer_tp_size}, local decoder TP is "
                f"{self.world_size}"
            )
        offer_generation = meta.remote.source_offer_generation
        if type(offer_generation) is not int or offer_generation < 1:
            raise ValueError(
                f"request {req_id} has no generation-scoped producer offer"
            )
        consumer_index = _parallel_consumer_index(
            req_id,
            meta.remote.expected_consumers,
        )
        proof = PullReadComplete(
            producer_request_id=meta.remote.request_id,
            offer_generation=offer_generation,
            consumer_request_id=req_id,
            consumer_index=consumer_index,
            consumer_rank=self.tp_rank,
            consumer_tp_size=self.world_size,
            expected_consumers=meta.remote.expected_consumers,
        )
        return PULL_READ_COMPLETE_PREFIX + msgspec.msgpack.encode(proof)

    def _localization_build_source_contracts(
        self,
        req_id: str,
        meta: ReqMeta,
        read_specs: list[ReadSpec],
        raw_remote_groups: tuple[tuple[int, ...], ...],
        region_lengths: tuple[int, ...],
    ) -> tuple[NixlSourceContract, ...]:
        """Build immutable source contracts from request and handshake metadata.

        :param req_id: Decoder child request identifier.
        :param meta: Request transfer metadata.
        :param read_specs: Per-source-rank transfer specifications.
        :param raw_remote_groups: Exact pre-sort physical source groups.
        :param region_lengths: Exact source row lengths in registration order.
        :returns: Contracts in the same source-rank order as ``read_specs``.
        :raises LocalizationError: If lineage, geometry, or semantics are invalid.
        """
        assert meta.remote is not None
        remote = meta.remote
        child_target = localization_request_target(
            req_id,
            self._localization_config.target_request_ids,
        )
        producer_target = localization_producer_target(
            remote.request_id,
            self._localization_config.target_request_ids,
        )
        if (
            child_target is None
            or producer_target != child_target
            or remote.p2d_run_id != self._localization_config.run_id
            or remote.p2d_transport_arm != self._localization_config.transport_arm
            or type(remote.source_offer_generation) is not int
            or remote.source_offer_generation < 1
            or type(remote.p2d_iteration) is not int
            or remote.p2d_iteration < 0
            or remote.remote_num_tokens <= 0
        ):
            raise LocalizationError(
                f"request {req_id} has incomplete or mismatched localization lineage"
            )
        source_ranks = tuple(int(spec.remote_rank) for spec in read_specs)
        if (
            len(source_ranks) == 0
            or len(set(source_ranks)) != len(source_ranks)
            or any(rank < 0 for rank in source_ranks)
        ):
            raise LocalizationError(
                f"request {req_id} has invalid required source ranks"
            )
        if len(raw_remote_groups) == 0:
            raise LocalizationError(f"request {req_id} has no source cache groups")
        if len(self._region_descriptors) != len(region_lengths):
            raise LocalizationError(
                f"request {req_id} local region descriptor cardinality differs"
            )
        destination_group_planes = tuple(
            1 if flag else 2 for flag in self._sp_group_flags()
        )
        destination_group_token_capacities = self._physical_group_token_capacities()
        if len(destination_group_planes) != len(raw_remote_groups):
            raise LocalizationError(
                f"request {req_id} destination plane-contract cardinality differs"
            )
        local_owned_groups = {
            group_index
            for region in self._region_descriptors
            for group_index in region.group_indices
        }
        if local_owned_groups != set(range(len(raw_remote_groups))):
            raise LocalizationError(
                f"request {req_id} local semantic regions do not cover all groups"
            )

        layouts = self._remote_layout.get(remote.engine_id)
        regions_by_rank = self._remote_regions.get(remote.engine_id)
        generations = self._remote_registration_generations.get(remote.engine_id)
        semantics_by_rank = self._remote_source_semantics.get(remote.engine_id)
        base_addresses_by_rank = self.kv_caches_base_addr.get(remote.engine_id)
        if (
            layouts is None
            or regions_by_rank is None
            or generations is None
            or semantics_by_rank is None
            or base_addresses_by_rank is None
        ):
            raise LocalizationError(
                f"request {req_id} has no validated producer handshake"
            )

        contracts: list[NixlSourceContract] = []
        reference_planes: tuple[int, ...] | None = None
        reference_capacities: tuple[int, ...] | None = None
        for spec, source_rank in zip(read_specs, source_ranks, strict=True):
            if (
                source_rank not in layouts
                or source_rank not in regions_by_rank
                or source_rank not in generations
                or source_rank not in semantics_by_rank
                or source_rank not in base_addresses_by_rank
            ):
                raise LocalizationError(
                    f"request {req_id} lacks validated P-rank {source_rank} metadata"
                )
            spec_remote_groups = tuple(
                tuple(int(block_id) for block_id in group)
                for group in spec.remote_block_ids
            )
            if spec_remote_groups != raw_remote_groups:
                raise LocalizationError(
                    f"request {req_id} source roster differs on rank {source_rank}"
                )
            layout = layouts[source_rank]
            source_region_lengths = tuple(int(length) for length in layout[0])
            if source_region_lengths != region_lengths:
                raise LocalizationError(
                    f"request {req_id} source region layout differs on rank "
                    f"{source_rank}"
                )
            num_blocks = int(layout[1])
            if num_blocks <= 0 or any(
                block_id < 0 or block_id >= num_blocks
                for group in raw_remote_groups
                for block_id in group
            ):
                raise LocalizationError(
                    f"request {req_id} source roster is outside rank "
                    f"{source_rank} registration"
                )

            source_group_planes, group_token_capacities = semantics_by_rank[source_rank]
            source_group_planes = tuple(int(planes) for planes in source_group_planes)
            group_token_capacities = tuple(
                int(capacity) for capacity in group_token_capacities
            )
            if (
                source_group_planes != destination_group_planes
                or group_token_capacities != destination_group_token_capacities
            ):
                raise LocalizationError(
                    f"request {req_id} source semantics differ from destination on "
                    f"rank {source_rank}"
                )
            if reference_planes is None:
                reference_planes = source_group_planes
                reference_capacities = group_token_capacities
            elif (
                source_group_planes != reference_planes
                or group_token_capacities != reference_capacities
            ):
                raise LocalizationError(
                    f"request {req_id} source semantics differ on rank {source_rank}"
                )

            regions = tuple(regions_by_rank[source_rank])
            base_addresses = tuple(
                int(address) for address in base_addresses_by_rank[source_rank]
            )
            if len(regions) != len(base_addresses) or any(
                region.base_address != base_address
                or len(region.shape) == 0
                or region.shape[0] != num_blocks
                for region, base_address in zip(
                    regions,
                    base_addresses,
                    strict=True,
                )
            ):
                raise LocalizationError(
                    f"request {req_id} source contract differs from native "
                    f"registration on rank {source_rank}"
                )
            contract = NixlSourceContract(
                schema_version=IntegrityIdentity.SCHEMA_VERSION,
                fingerprint_algorithm=self._localization_config.fingerprint_algorithm,
                run_id=self._localization_config.run_id,
                transport_arm=self._localization_config.transport_arm,
                producer_engine_id=remote.engine_id,
                producer_request_id=remote.request_id,
                registration_generation=generations[source_rank],
                offer_generation=remote.source_offer_generation,
                iteration=remote.p2d_iteration,
                expected_consumers=remote.expected_consumers,
                source_rank=source_rank,
                region_lengths=source_region_lengths,
                regions=regions,
                source_group_planes=source_group_planes,
                valid_token_extent=remote.remote_num_tokens,
                group_token_capacities=group_token_capacities,
                block_ids=raw_remote_groups,
            )
            structure_errors = validate_source_contract_structure(contract)
            if len(structure_errors) > 0:
                raise LocalizationError(
                    f"invalid source contract for decoder request {req_id}: "
                    f"{structure_errors[:8]}"
                )
            if len(regions) != len(self._region_descriptors):
                raise LocalizationError(
                    f"request {req_id} source/local region count differs on rank "
                    f"{source_rank}"
                )
            for source_region, local_region in zip(
                regions,
                self._region_descriptors,
                strict=True,
            ):
                if (
                    source_region.semantic_name != local_region.semantic_name
                    or source_region.group_indices != local_region.group_indices
                    or source_region.group_semantic_names
                    != local_region.group_semantic_names
                    or source_region.dtype != local_region.dtype
                    or source_region.element_size_bytes
                    != local_region.element_size_bytes
                    or source_region.layout != local_region.layout
                ):
                    raise LocalizationError(
                        f"request {req_id} semantic region identity differs on "
                        f"rank {source_rank}"
                    )
                if local_region.row_bytes != len(read_specs) * source_region.row_bytes:
                    raise LocalizationError(
                        f"request {req_id} local/source row geometry differs on "
                        f"rank {source_rank}"
                    )
                for role, region in (
                    ("source", source_region),
                    ("local", local_region),
                ):
                    if (
                        len(region.shape) == 0
                        or len(region.shape) != len(region.strides)
                        or region.registered_bytes != region.shape[0] * region.row_bytes
                        or region.strides[0] * region.element_size_bytes
                        != region.row_bytes
                    ):
                        raise LocalizationError(
                            f"request {req_id} {role} region geometry is not "
                            f"row canonical on rank {source_rank}"
                        )
            contracts.append(contract)
        return tuple(contracts)

    def _no_stock_dma(self) -> bool:
        """Return whether the known-corrupt stock DMA path is disabled.

        Stock DMA is disabled by default because NIXL/UCX corrupts local KV
        offsets beyond its large-offset threshold. Explicitly setting the
        variable to ``0`` is reserved for controlled small-pool diagnosis.

        :returns: Whether requests must fail instead of using stock DMA.
        """
        return os.environ.get("VLLM_GEMMA4_NIXL_NO_STOCK_DMA", "1") == "1"

    def _stock_read_specs(
        self,
        req_id: str,
        meta: ReqMeta,
        read_specs: list[ReadSpec],
        notification_id: bytes,
    ) -> None:
        """The stock per-descriptor pull for one request's read specs
        (extracted from _read_blocks_for_req so the pending-queue
        servicer can also route a request here)."""
        assert meta.remote is not None and self.transfer_topo is not None
        dst_engine_id = meta.remote.engine_id
        remote_info = self.transfer_topo.get_engine_info(dst_engine_id)
        tp_ratio = self.transfer_topo.tp_ratio(remote_info.remote_tp_size)

        for i, spec in enumerate(read_specs):
            remote_block_size = remote_info.remote_block_size
            logger.debug(
                "Remote agent %s available, calling _read_blocks"
                " on remote rank %s with remote block size %s for req %s",
                meta.remote.engine_id,
                spec.remote_rank,
                remote_block_size,
                req_id,
            )
            # Get side handles.
            if tp_ratio < 0 and not self.use_mla:
                assert remote_block_size == self.block_size
                # Remote tp_size > local tp_size: we must perform multiple
                # reads. Get the memory chunk onto which we will write to.
                local_xfer_side_handle = self.src_xfer_handles_by_tp_ratio[tp_ratio][i]
            else:
                # Single read from remote, we write to the whole memory region.
                # Also handle remote block size different from local block size.
                local_xfer_side_handle = self.src_xfer_handles_by_block_size[
                    remote_block_size
                ]

            # Destination handle: remote_engine_id -> remote_rank -> handle.
            remote_xfer_side_handle = self.dst_xfer_side_handles[meta.remote.engine_id][
                spec.remote_rank
            ]

            self._read_blocks(
                read_spec=spec,
                request_id=req_id,
                dst_engine_id=meta.remote.engine_id,
                local_xfer_side_handle=local_xfer_side_handle,
                remote_xfer_side_handle=remote_xfer_side_handle,
                notification_id=notification_id,
            )

        self._notify_non_read_producer_ranks(meta, read_specs, notification_id)

    def _notify_non_read_producer_ranks(
        self,
        meta: ReqMeta,
        read_specs: list[ReadSpec],
        notification_id: bytes,
    ) -> None:
        """Complete obligations for producer ranks this decoder does not read.

        Replicated MLA or GQA pages can be omitted from the transfer plan, but
        their producer ranks still own the lease. A no-read proof is safe as
        soon as the plan is fixed because this decoder cannot touch those pages.

        :param meta: Remote producer metadata.
        :param read_specs: Producer ranks this decoder may actually read.
        :param notification_id: Typed decoder-rank completion proof.
        """
        read_ranks = {spec.remote_rank for spec in read_specs}
        self._notify_producer_ranks_without_native_read(
            meta,
            read_ranks,
            notification_id,
        )

    def _notify_failed_coalesced_producer_ranks(
        self,
        meta: ReqMeta,
        ownership: CoalescedStagingPlan,
        notification_id: bytes,
    ) -> None:
        """Discharge only producer ranks proved untouched by a failed pull.

        This decoder child was admitted, so its terminal proof remains the
        existing :class:`PullReadComplete` carried by ``notification_id``.
        :class:`PullOfferCancelled` is a mutually exclusive whole-offer proof
        and would conflict with any rank whose native read reached DONE.

        :param meta: Immutable producer and consumer contract.
        :param ownership: Released failed plan retaining per-rank evidence.
        :param notification_id: Typed decoder-rank completion proof.
        :raises StagingSafetyError: If any source rank lacks exact terminal state.
        """
        if (
            ownership.operation_failed is False
            or ownership.released is False
            or ownership.reusable is False
        ):
            raise StagingSafetyError(
                "A failed coalesced completion lacks a released ownership proof: "
                f"{ownership.describe()}"
            )

        native_read_ranks: set[int] = set()
        for source_rank, slot in ownership.slots.items():
            if slot.state is HandleState.DONE:
                native_read_ranks.add(source_rank)
                continue
            if slot.state not in {
                HandleState.NEVER_POSTED,
                HandleState.PREPARE_FAILED,
                HandleState.SEALED_UNPOSTED,
            }:
                raise StagingSafetyError(
                    "A recoverable coalesced failure retained an ambiguous rank: "
                    f"{ownership.describe()}"
                )

        self._notify_producer_ranks_without_native_read(
            meta,
            native_read_ranks,
            notification_id,
        )
        # A notification exception is delivery-uncertain: the producer may
        # already have accepted the proof. Fence this offer locally even when
        # another producer rank remains conservatively pinned.
        self._record_remote_source_consumption_proven(
            ownership.request_id,
            meta,
        )

    def _notify_producer_ranks_without_native_read(
        self,
        meta: ReqMeta,
        native_read_ranks: set[int],
        notification_id: bytes,
    ) -> None:
        """Send one admitted-child no-read proof to each untouched rank.

        Each producer rank owns its own copy of the child/rank obligation. The
        identical typed proof must therefore reach every rank not covered by a
        native transfer, exactly once per rank.

        :param meta: Immutable producer and consumer contract.
        :param native_read_ranks: Producer ranks whose native read carries the proof.
        :param notification_id: Typed decoder-rank completion proof.
        """
        assert meta.remote is not None
        remote_agents = self._remote_agents[meta.remote.engine_id]
        obligated_producer_ranks = {
            producer_rank
            for producer_rank in remote_agents
            if self.tp_rank
            in _consumer_ranks_for_producer(
                producer_rank,
                meta.tp_size,
                self.world_size,
            )
        }
        unexpected_read_ranks = native_read_ranks - obligated_producer_ranks
        if len(unexpected_read_ranks) > 0:
            raise StagingSafetyError(
                "Native reads escaped this decoder rank's producer obligations: "
                f"{sorted(unexpected_read_ranks)}"
            )

        for producer_rank in sorted(obligated_producer_ranks - native_read_ranks):
            agent = remote_agents[producer_rank]
            try:
                self.nixl_wrapper.send_notif(agent, notif_msg=notification_id)
            except Exception:
                logger.error(
                    "Failed to prove no-read completion for producer request "
                    "%s to producer rank %d; producer pages remain pinned.\n%s",
                    meta.remote.request_id,
                    producer_rank,
                    traceback.format_exc(),
                )
                self.xfer_stats.record_failed_notification()

    # ------------------------------------------------------------------
    # Coalesced pull
    # ------------------------------------------------------------------

    def _coalesce_gate(
        self, engine_id: str, tp_ratio: int, read_specs: list[ReadSpec]
    ) -> bool:
        """All conditions under which the coalesced path is proven
        equivalent to the stock path. Anything else -> stock."""
        if not self.coalesce_pull:
            return False
        if not (
            tp_ratio < 0
            and not self.use_mla
            and not self.use_host_buffer
            and not self._has_mamba
        ):
            return False
        if self._physical_blocks_per_logical_kv_block != 1:
            return False
        if any(self._region_is_mla) or not self._region_tensors:
            return False
        assert self.transfer_topo is not None
        remote_info = self.transfer_topo.get_engine_info(engine_id)
        source_ranks = tuple(int(spec.remote_rank) for spec in read_specs)
        if (
            self.world_size != 1
            or remote_info.remote_tp_size != 4
            or tp_ratio != -4
            or len(read_specs) != 4
            or tuple(sorted(source_ranks)) != (0, 1, 2, 3)
        ):
            return False
        if self.transfer_topo.block_size_ratio(remote_info.remote_block_size) != 1:
            return False
        layout = self._remote_layout.get(engine_id)
        if not layout or any(s.remote_rank not in layout for s in read_specs):
            return False
        # uniform shard participation: every rank reads the same lists
        # (pure SPLIT full-attention groups)
        spec0 = read_specs[0]
        for s in read_specs[1:]:
            if (
                s.local_block_ids != spec0.local_block_ids
                or s.remote_block_ids != spec0.remote_block_ids
            ):
                return False
        blens = layout[spec0.remote_rank][0]
        if len(blens) != len(self._region_tensors):
            return False
        n_ranks = len(read_specs)
        for i, blen in enumerate(blens):
            # local block row must be exactly the |tp_ratio| remote
            # shard blocks side by side, K/V halves equal
            if blen % 2 != 0 or self.block_len_per_layer[i] != n_ranks * blen:
                return False
        if not self._coalesce_region_rows():
            return False
        return self._staging_init()

    def _coalesce_service_pending(self) -> None:
        """Start transfers for requests parked on the staging pool
        (FIFO -- strict head-of-line, so a large request cannot be
        starved by smaller ones slipping past it)."""
        while self._coalesce_pending:
            req_id, meta, read_specs, source_contracts = self._coalesce_pending[0]
            notification_id = self._read_completion_notification(req_id, meta)
            res = self._coalesced_read_request(
                req_id,
                meta,
                read_specs,
                notification_id,
                source_contracts,
            )
            if res == "defer":
                break
            self._coalesce_pending.popleft()
            if res == "stock":
                if (
                    self._localization_config.enabled_for(req_id)
                    or any(self._sp_group_flags())
                    or self._no_stock_dma()
                ):
                    logger.error(
                        "coalesced drain: %s not expressible; failing "
                        "instead of the stock path.",
                        req_id,
                    )
                    self._handle_failed_transfer(req_id, None)
                    continue
                self._stock_read_specs(
                    req_id,
                    meta,
                    read_specs,
                    notification_id,
                )

    def _coalesced_read_request(
        self,
        req_id: str,
        meta: ReqMeta,
        read_specs: list[ReadSpec],
        notification_id: bytes,
        source_contracts: tuple[NixlSourceContract, ...] = (),
    ) -> str:
        """Post one whole-request READ per remote rank into owned staging.

        A failure before every possible native writer is quiescent becomes a
        request-scoped terminal. Any uncertain native submission raises
        :class:`StagingSafetyError` and leaves its generation owned until
        process replacement.

        :param req_id: Decoder request identifier.
        :param meta: Complete transfer metadata.
        :param read_specs: Per-source-rank transfer specifications.
        :param notification_id: Typed decoder-rank completion proof.
        :param source_contracts: Frozen diagnostic source contracts.
        :returns: ``posted``, ``defer``, or ``stock`` before native failure.
        """
        assert meta.remote is not None and self.transfer_topo is not None
        localization_enabled = self._localization_config.enabled_for(req_id)
        engine_id = meta.remote.engine_id
        remote_info = self.transfer_topo.get_engine_info(engine_id)
        spec0 = read_specs[0]
        untrimmed_local_groups = tuple(
            tuple(int(block_id) for block_id in group)
            for group in spec0.local_block_ids
        )
        raw_remote_groups = tuple(
            tuple(int(block_id) for block_id in group)
            for group in spec0.remote_block_ids
        )
        local_ids, remote_ids = self._apply_prefix_caching(
            [list(g) for g in spec0.local_block_ids],
            [list(g) for g in spec0.remote_block_ids],
            remote_info.remote_physical_blocks_per_logical,
        )
        if len(local_ids) == 0 or sum(len(g) for g in local_ids) == 0:
            # A full prefix hit performs no native read, so its proof can be
            # sent immediately and its empty transfer can complete locally.
            for s in read_specs:
                agent = self._remote_agents[engine_id][s.remote_rank]
                try:
                    self.nixl_wrapper.send_notif(
                        agent,
                        notif_msg=notification_id,
                    )
                except Exception:
                    logger.error(
                        "full-prefix release notification failed for %s rank %s\n%s",
                        req_id,
                        s.remote_rank,
                        traceback.format_exc(),
                    )
                    self.xfer_stats.record_failed_notification()
            self._notify_non_read_producer_ranks(
                meta,
                read_specs,
                notification_id,
            )
            self._recving_transfers.setdefault(req_id, [])
            return "posted"

        sp_flags = self._sp_group_flags()
        n_ranks = len(read_specs)
        blens = self._remote_layout[engine_id][spec0.remote_rank][0]
        n_regions = len(blens)
        tp_mapping = self.tp_mappings[engine_id]
        source_ranks = tuple(int(spec.remote_rank) for spec in read_specs)
        if len(set(source_ranks)) != n_ranks:
            raise LocalizationError("coalesced plan contains duplicate source ranks")
        if any(rank not in tp_mapping.rank_to_attention_slot for rank in source_ranks):
            raise LocalizationError("coalesced plan is missing a source-rank slot")
        slots = tuple(
            int(tp_mapping.rank_to_attention_slot[rank]) for rank in source_ranks
        )
        if sorted(slots) != list(range(n_ranks)):
            raise LocalizationError(
                "coalesced source-rank slots must be a complete bijection"
            )

        group_rosters: list[GroupTransferRoster] = []
        for group_index, (raw_remote, selected_remote, selected_local) in enumerate(
            zip(raw_remote_groups, remote_ids, local_ids, strict=True)
        ):
            source_position_start = len(raw_remote) - len(selected_remote)
            if source_position_start < 0 or tuple(
                raw_remote[source_position_start:]
            ) != tuple(selected_remote):
                raise StagingSafetyError(
                    f"request {req_id} group {group_index} is not a suffix trim"
                )
            group_rosters.append(
                GroupTransferRoster(
                    group_index=group_index,
                    source_position_start=source_position_start,
                    destination_plane_count=1 if sp_flags[group_index] else 2,
                    local_block_ids=tuple(int(block_id) for block_id in selected_local),
                    remote_block_ids=tuple(
                        int(block_id) for block_id in selected_remote
                    ),
                )
            )

        remote_regions = self._remote_regions[engine_id][spec0.remote_rank]
        if (
            len(remote_regions) != n_regions
            or len(self._region_descriptors) != n_regions
        ):
            raise StagingSafetyError(
                f"request {req_id} region metadata changed after handshake"
            )
        region_ownership = tuple(
            RegionOwnership(
                region_index=region_index,
                group_indices=remote_region.group_indices,
                source_row_count=int(remote_region.shape[0]),
                destination_row_count=int(
                    self._region_descriptors[region_index].shape[0]
                ),
                row_bytes=int(blens[region_index]),
            )
            for region_index, remote_region in enumerate(remote_regions)
        )
        layout_started_ns = time.perf_counter_ns()
        source_plan = build_canonical_source_plan(
            source_tp_size=int(remote_info.remote_tp_size),
            source_ranks=source_ranks,
            groups=tuple(
                SourceGroupTransferRoster(
                    group_index=group.group_index,
                    source_position_start=group.source_position_start,
                    remote_block_ids=group.remote_block_ids,
                )
                for group in group_rosters
            ),
            regions=tuple(
                SourceRegionOwnership(
                    region_index=region.region_index,
                    group_indices=region.group_indices,
                    source_row_count=region.source_row_count,
                    row_bytes=region.row_bytes,
                )
                for region in region_ownership
            ),
        )
        transfer_layout = bind_coalesced_transfer_destinations(
            source_plan,
            rank_slots=slots,
            groups=tuple(group_rosters),
            regions=region_ownership,
        )
        layout_duration_ns = time.perf_counter_ns() - layout_started_ns
        logical_position_count = sum(
            len(group.remote_block_ids) for group in transfer_layout.groups
        )
        logical_staging_size = (
            logical_position_count * n_ranks * sum(int(blen) for blen in blens)
        )
        logger.info(
            "[coalesced-layout] request=%s digest=%s logical_bytes=%d "
            "wire_bytes=%d elided_bytes=%d regions=%d layout_ns=%d",
            req_id,
            transfer_layout.digest,
            logical_staging_size,
            transfer_layout.staging_size_bytes,
            logical_staging_size - transfer_layout.staging_size_bytes,
            n_regions,
            layout_duration_ns,
        )

        semantic_contract: NixlSourceContract | None = None
        if localization_enabled:
            if (
                len(source_contracts) != n_ranks
                or tuple(contract.source_rank for contract in source_contracts)
                != source_ranks
                or any(
                    contract.block_ids != raw_remote_groups
                    for contract in source_contracts
                )
            ):
                raise LocalizationError(
                    f"request {req_id} source contracts differ from its read plan"
                )
            semantic_contract = source_contracts[0]
        elif len(source_contracts) > 0:
            raise LocalizationError(
                f"request {req_id} carries contracts outside localization scope"
            )

        direct_descriptors_per_rank = sum(
            len(region.runs) for region in source_plan.regions
        )
        if (
            self._packed_write_config.enabled
            and localization_enabled is False
            and direct_descriptors_per_rank
            >= self._packed_write_config.min_descriptors_per_rank
        ):
            return self._packed_write_request(
                req_id=req_id,
                meta=meta,
                read_specs=read_specs,
                source_plan=source_plan,
                destination_plan=transfer_layout,
                logical_bytes=logical_staging_size,
                direct_descriptor_count=direct_descriptors_per_rank * n_ranks,
            )

        region_plans: tuple[NixlRegionPlan, ...] = ()
        if localization_enabled:
            assert semantic_contract is not None
            region_plans = tuple(
                NixlRegionPlan(
                    region_index=region.ownership.region_index,
                    offset_within_rank=region.offset_within_rank,
                    positions=tuple(
                        NixlPlanPosition(
                            group_index=position.group_index,
                            source_position=position.source_position,
                            remote_block_id=position.remote_block_id,
                            valid_token_extent=(semantic_contract.valid_token_extent),
                            group_token_capacity=(
                                semantic_contract.group_token_capacities[
                                    position.group_index
                                ]
                            ),
                            local_block_id=position.local_block_id,
                            destination_half=position.destination_half,
                        )
                        for position in region.positions
                    ),
                    runs=tuple(
                        NixlPlanRun(
                            remote_block_id=run.remote_block_id,
                            position_count=run.position_count,
                            position_start=run.position_start,
                        )
                        for run in region.runs
                    ),
                )
                for region in transfer_layout.regions
            )
            for region in region_plans:
                for position in region.positions:
                    if sp_flags[position.group_index]:
                        expected_half = position.source_position % 2
                        if position.destination_half != expected_half:
                            raise LocalizationError(
                                "single-plane destination half is not derived from "
                                "the absolute source position"
                            )
                    elif position.destination_half != -1:
                        raise LocalizationError(
                            "dual-plane transfer position carries a destination half"
                        )
        acc = transfer_layout.staging_size_bytes
        if acc > self.coalesce_staging_mb * 1024 * 1024:
            logger.warning(
                "coalesced pull: request %s needs %sMB staging (pool %sMB); stock path",
                req_id,
                acc >> 20,
                self.coalesce_staging_mb,
            )
            return "stock"
        ownership = self._create_coalesced_plan(
            req_id,
            engine_id,
            transfer_layout,
            layout_duration_ns / 1_000_000_000,
        )
        if ownership is None:
            return "defer"
        off = ownership.lease.offset

        if localization_enabled:
            try:
                if self._localization_writer is None:
                    raise LocalizationError(
                        "enabled localization has no artifact writer"
                    )
                self._localization_writer.write(
                    NixlPlanRecord(
                        record_type=NixlPlanRecord.RECORD_TYPE,
                        schema_version=IntegrityIdentity.SCHEMA_VERSION,
                        source_tp_size=transfer_layout.source_tp_size,
                        source_contracts=source_contracts,
                        child_request_id=req_id,
                        observer_engine_id=self.engine_id,
                        observer_rank=self.tp_rank,
                        rank_slots=transfer_layout.rank_slots,
                        destination_group_planes=tuple(
                            1 if flag else 2 for flag in sp_flags
                        ),
                        destination_group_token_capacities=(
                            self._physical_group_token_capacities()
                        ),
                        local_regions=self._region_descriptors,
                        untrimmed_local_groups=untrimmed_local_groups,
                        skipped_groups=tuple(sorted(self._skip_pull_groups)),
                        selected_remote_groups=tuple(
                            tuple(int(block_id) for block_id in group)
                            for group in remote_ids
                        ),
                        selected_local_groups=tuple(
                            tuple(int(block_id) for block_id in group)
                            for group in local_ids
                        ),
                        region_plans=region_plans,
                        rank_stride_bytes=transfer_layout.rank_stride_bytes,
                        layout_digest=transfer_layout.digest,
                        staging_offset=off,
                        staging_size=transfer_layout.staging_size_bytes,
                    )
                )
                self._install_coalesced_localization_plan(
                    req_id,
                    transfer_layout,
                    source_contracts,
                )
            except Exception as error:
                stacktrace = traceback.format_exc()
                self._finish_quiescent_coalesced_failure(
                    ownership,
                    "localization plan recording failed before native posting\n"
                    + stacktrace,
                    error,
                )
                self._notify_failed_coalesced_producer_ranks(
                    meta,
                    ownership,
                    notification_id,
                )
                return "posted"

        assert self._staging_buf is not None
        staging_base = self._staging_buf.data_ptr() + off
        for rank_index, s in enumerate(read_specs):
            source_rank = int(s.remote_rank)
            ownership.begin_prepare(source_rank)
            try:
                blens_r, _, rdev = self._remote_layout[engine_id][s.remote_rank]
                assert blens_r == blens, "per-rank region layout mismatch"
                rbases = self.kv_caches_base_addr[engine_id][s.remote_rank]
                local_descs, remote_descs = [], []
                for region_index, region_layout in enumerate(transfer_layout.regions):
                    row_bytes = region_layout.ownership.row_bytes
                    region_base = staging_base + transfer_layout.region_offset(
                        rank_index,
                        region_index,
                    )
                    for run in region_layout.runs:
                        length = run.position_count * row_bytes
                        local_descs.append(
                            (
                                region_base + run.position_start * row_bytes,
                                length,
                                self.device_id,
                            )
                        )
                        remote_descs.append(
                            (
                                rbases[region_index] + run.remote_block_id * row_bytes,
                                length,
                                rdev,
                            )
                        )
                if len(local_descs) == 0:
                    raise StagingSafetyError(
                        f"request {req_id} produced no native descriptors"
                    )
                ld = self.nixl_wrapper.get_xfer_descs(
                    local_descs, self.nixl_memory_type
                )
                rd = self.nixl_wrapper.get_xfer_descs(
                    remote_descs, self.nixl_memory_type
                )
                agent = self._remote_agents[engine_id][s.remote_rank]
            except Exception as error:
                stacktrace = traceback.format_exc()
                ownership.record_prepare_failure(
                    source_rank,
                    f"rank {source_rank} native preparation raised\n{stacktrace}",
                )
                self._finish_quiescent_coalesced_failure(
                    ownership,
                    f"rank {source_rank} native preparation raised\n{stacktrace}",
                    error,
                )
                self._notify_failed_coalesced_producer_ranks(
                    meta,
                    ownership,
                    notification_id,
                )
                return "posted"
            prepared = self._prepare_coalesced_handle(
                ownership,
                source_rank,
                ld,
                rd,
                agent,
                notification_id,
            )
            if prepared is False:
                self._notify_failed_coalesced_producer_ranks(
                    meta,
                    ownership,
                    notification_id,
                )
                return "posted"

        try:
            self._assert_transfer_phase_active(coalesced=True)
        except Exception as error:
            reason = "coalesced post phase rejected before native submission\n" + (
                traceback.format_exc()
            )
            ownership.fail(reason)
            self._finish_quiescent_coalesced_failure(ownership, reason, error)
            self._notify_failed_coalesced_producer_ranks(
                meta,
                ownership,
                notification_id,
            )
            return "posted"

        for source_rank in transfer_layout.source_ranks:
            self._post_prepared_coalesced(ownership, source_rank)
        self._notify_non_read_producer_ranks(meta, read_specs, notification_id)
        ownership.seal_posting()
        return "posted"

    def _record_packed_source_readiness(
        self,
        source_rosters: dict[str, Any],
    ) -> None:
        """Record CUDA ordering for newly retained producer source rosters.

        :param source_rosters: Scheduler-delivered producer roster additions.
        """
        if self._packed_write_config.enabled is False:
            return
        if self.kv_transfer_config.kv_role not in {"kv_producer", "kv_both"}:
            return
        geometry = self._packed_write_producer_pool_geometry
        buffer = self._packed_write_producer_pool_buffer
        stream = self._packed_write_producer_pack_stream
        if geometry is None or buffer is None or stream is None:
            raise StagingSafetyError("packed WRITE producer pool is unavailable")
        self._ensure_packed_producer_pool()
        current_stream = torch.cuda.current_stream(buffer.device)
        for request_id in source_rosters:
            if request_id in self._packed_source_ready_events:
                continue
            if request_id not in self._source_rosters:
                raise StagingSafetyError(
                    f"packed WRITE producer roster {request_id} was not retained"
                )
            event = torch.cuda.Event()
            event.record(current_stream)
            self._packed_source_ready_events[request_id] = event

    @staticmethod
    def _require_packed_notifying_agent(notifying_agent: str) -> None:
        """Validate one native notification sender identity.

        :param notifying_agent: Sender identity reported by NIXL.
        """
        if type(notifying_agent) is not str or len(notifying_agent) == 0:
            raise ValueError("packed WRITE sender must be a non-empty string")

    def _ensure_packed_producer_pool(self) -> PackedWriteProducerPool:
        """Return the generation-owning producer gather pool.

        :returns: Local producer pool.
        """
        if self._packed_producer_pool is not None:
            return self._packed_producer_pool
        geometry = self._packed_write_producer_pool_geometry
        if geometry is None:
            raise StagingSafetyError("packed WRITE producer geometry is unavailable")
        self._packed_producer_pool = PackedWriteProducerPool(
            geometry=geometry,
            producer_rank=self.tp_rank,
        )
        return self._packed_producer_pool

    def _bind_packed_completion(
        self,
        request: PackedWriteRequest,
        consumer_agent: str,
    ) -> _PackedCompletionBinding:
        """Bind every chunk to one immutable decoder completion identity.

        :param request: Authenticated packed WRITE command.
        :param consumer_agent: Native command sender.
        :returns: Existing or newly installed request-level binding.
        """
        identity = request.command.chunk
        if (
            identity.producer_engine_id != self.engine_id
            or identity.producer_rank != self.tp_rank
            or identity.producer_tp_size != self.world_size
        ):
            raise StagingSafetyError(
                "packed WRITE command targets a different producer endpoint"
            )
        request_key = PackedWriteRequestKey.from_chunk(
            PackedWriteChunkKey.from_identity(identity)
        )
        binding_key = (
            identity.producer_request_id,
            identity.offer_generation,
            identity.consumer_request_id,
            identity.consumer_rank,
        )
        binding = _PackedCompletionBinding(
            request_key=request_key,
            consumer_endpoint=request.consumer_endpoint,
            consumer_agent=consumer_agent,
        )
        prior = self._packed_completion_bindings.get(binding_key)
        if prior is not None:
            if prior != binding:
                raise StagingSafetyError(
                    "packed WRITE chunks changed their completion binding"
                )
            return prior
        state = self._pull_completion_states.get(identity.producer_request_id)
        if state is not None:
            completion_identity = self._packed_binding_completion_identity(
                binding,
                state,
            )
            if len(state.offer_cancellations) > 0:
                self._mark_pull_contract_conflicted(
                    identity.producer_request_id,
                    state,
                )
                raise StagingSafetyError(
                    "packed WRITE binding follows a whole-offer cancellation"
                )
            if completion_identity in state.read_completions:
                raise StagingSafetyError(
                    "packed WRITE binding follows a legacy completion proof"
                )
        roster = self._source_rosters.get(identity.producer_request_id)
        if roster is not None and roster.offer_generation != identity.offer_generation:
            raise StagingSafetyError(
                "packed WRITE command names a stale source-offer generation"
            )
        self._packed_completion_bindings[binding_key] = binding
        return binding

    def _packed_binding_completion_identity(
        self,
        binding: _PackedCompletionBinding,
        state: PullCompletionState,
    ) -> tuple[int, int]:
        """Validate a binding against the producer-owned completion contract.

        :param binding: Authenticated request-level binding.
        :param state: Active producer completion contract.
        :returns: Stable child-index and decoder-rank obligation.
        """
        key = binding.request_key
        if state.offer_generation != key.offer_generation:
            raise StagingSafetyError(
                "packed WRITE source-offer generation differs from its contract"
            )
        if key.consumer_tp_size != state.consumer_tp_size:
            raise StagingSafetyError(
                "packed WRITE decoder topology differs from the source contract"
            )
        consumer_index = _parallel_consumer_index(
            key.consumer_request_id,
            state.expected_consumers,
        )
        identity = (consumer_index, key.consumer_rank)
        if identity not in state.expected_read_completions:
            raise StagingSafetyError(
                "packed WRITE completion is outside this producer rank's obligations"
            )
        return identity

    def _packed_consumer_agent(
        self,
        endpoint: PackedWriteConsumerEndpoint,
    ) -> str:
        """Import and cache one generation-bound decoder endpoint.

        :param endpoint: Exact decoder agent and receive-pool registration.
        :returns: Native remote-agent identity.
        """
        self._evict_stale_packed_consumer_agents()
        cached = self._packed_consumer_agents.get(endpoint)
        if cached is not None:
            self._packed_consumer_agent_last_active[endpoint] = time.perf_counter()
            return cached
        agent = canonicalize_nixl_agent_name(
            self.nixl_wrapper.add_remote_agent(endpoint.agent_metadata)
        )
        self._packed_consumer_agents[endpoint] = agent
        self._packed_consumer_agent_last_active[endpoint] = time.perf_counter()
        return agent

    def _packed_consumer_agent_owned(self, consumer_agent: str) -> bool:
        """Return whether live producer state owns an imported decoder agent.

        :param consumer_agent: Native remote-agent identity.
        :returns: Whether removal would invalidate live packed work.
        """
        if any(
            pending.consumer_agent == consumer_agent
            for pending in self._packed_producer_pending
        ):
            return True
        if any(
            operation.consumer_agent == consumer_agent
            for operation in self._packed_producer_operations.values()
        ):
            return True
        if any(
            binding.consumer_agent == consumer_agent
            for binding in self._packed_completion_bindings.values()
        ):
            return True
        pool = self._packed_producer_pool
        return pool is not None and pool.owns_consumer_agent(consumer_agent)

    def _remove_packed_consumer_agent(
        self,
        endpoint: PackedWriteConsumerEndpoint,
    ) -> None:
        """Release one inactive endpoint cache entry.

        :param endpoint: Exact endpoint cache key.
        """
        agent = self._packed_consumer_agents.get(endpoint)
        if agent is None:
            self._packed_consumer_agent_last_active.pop(endpoint, None)
            return
        if self._packed_consumer_agent_owned(agent):
            raise StagingSafetyError(
                "packed WRITE endpoint removal would invalidate active ownership"
            )
        del self._packed_consumer_agents[endpoint]
        self._packed_consumer_agent_last_active.pop(endpoint, None)
        if agent in self._packed_consumer_agents.values():
            return
        if any(
            agent in rank_agents.values()
            for rank_agents in self._remote_agents.values()
        ):
            return
        self.nixl_wrapper.remove_remote_agent(agent)

    def _evict_stale_packed_consumer_agents(self) -> None:
        """Evict inactive dynamically imported decoder agents."""
        if self._engine_ttl <= 0:
            return
        now = time.perf_counter()
        for endpoint, last_active in tuple(
            self._packed_consumer_agent_last_active.items()
        ):
            if now - last_active <= self._engine_ttl:
                continue
            agent = self._packed_consumer_agents.get(endpoint)
            if agent is None:
                del self._packed_consumer_agent_last_active[endpoint]
                continue
            if self._packed_consumer_agent_owned(agent):
                self._packed_consumer_agent_last_active[endpoint] = now
                continue
            self._remove_packed_consumer_agent(endpoint)

    def _accept_packed_request_notification(
        self,
        request: PackedWriteRequest,
        notifying_agent: str,
    ) -> None:
        """Authenticate and queue one decoder-issued WRITE command.

        :param request: Decoded typed request.
        :param notifying_agent: Native sender identity reported by NIXL.
        """
        if self._packed_write_config.enabled is False:
            raise StagingSafetyError("packed WRITE request arrived while disabled")
        if self.kv_transfer_config.kv_role not in {"kv_producer", "kv_both"}:
            raise StagingSafetyError("packed WRITE request arrived on a non-producer")
        self._require_packed_notifying_agent(notifying_agent)
        consumer_agent = self._packed_consumer_agent(request.consumer_endpoint)
        if consumer_agent != notifying_agent:
            raise StagingSafetyError(
                "packed WRITE sender differs from its authenticated endpoint"
            )
        identity = request.command.chunk
        if identity.offer_generation <= self._local_source_retired_through:
            return
        completed = self._completed_pull_contracts.get(
            (identity.producer_request_id, identity.offer_generation)
        )
        if completed is not None:
            retired_binding = _PackedCompletionBinding(
                request_key=PackedWriteRequestKey.from_chunk(
                    PackedWriteChunkKey.from_identity(identity)
                ),
                consumer_endpoint=request.consumer_endpoint,
                consumer_agent=consumer_agent,
            )
            if retired_binding in completed.packed_bindings:
                return
            raise StagingSafetyError(
                "packed WRITE command conflicts with a released source contract"
            )
        self._bind_packed_completion(request, consumer_agent)
        pool = self._ensure_packed_producer_pool()
        terminal = pool.terminal_for_replay(request, consumer_agent)
        if terminal is not None:
            self._send_packed_producer_terminal(consumer_agent, terminal)
            return
        if pool.active_for_replay(request, consumer_agent):
            return
        pending = next(
            (
                candidate
                for candidate in self._packed_producer_pending
                if candidate.request.command.chunk == identity
            ),
            None,
        )
        if pending is not None:
            if pending.request != request or pending.consumer_agent != consumer_agent:
                raise StagingSafetyError(
                    "packed WRITE request replay changed authenticated fields"
                )
            return
        self._packed_producer_pending.append(
            _ProducerPackedPending(
                request=request,
                consumer_agent=consumer_agent,
                received_at=time.monotonic(),
            )
        )

    def _send_packed_producer_terminal(
        self,
        consumer_agent: str,
        terminal: PackedWriteArrived | PackedWriteFailedBeforeWrite,
    ) -> None:
        """Send one exact retained producer terminal to its decoder.

        :param consumer_agent: Authenticated decoder agent.
        :param terminal: Exact arrival or proven pre-WRITE failure.
        """
        if isinstance(terminal, PackedWriteArrived):
            encoded = encode_packed_write_arrived_notification(terminal)
        else:
            encoded = encode_packed_write_failed_before_write_notification(terminal)
        self.nixl_wrapper.send_notif(consumer_agent, notif_msg=encoded)

    def _send_unadmitted_packed_failure(
        self,
        pending: _ProducerPackedPending,
        code: PackedWriteFailureCode,
        reason: str,
    ) -> None:
        """Retain and send an exact command failure before slot admission.

        :param pending: Authenticated pending command.
        :param code: Stable pre-WRITE failure class.
        :param reason: Human-readable bounded detail.
        """
        failed = PackedWriteFailedBeforeWrite(
            command=pending.request.command,
            code=code,
            reason=reason[:1024],
        )
        pool = self._ensure_packed_producer_pool()
        pool.remember_failed_before_admission(
            pending.request,
            pending.consumer_agent,
            failed,
        )
        self.nixl_wrapper.send_notif(
            pending.consumer_agent,
            notif_msg=encode_packed_write_failed_before_write_notification(failed),
        )

    def _fail_admitted_packed_write(
        self,
        request: PackedWriteRequest,
        lease: ProducerSlotLease,
        consumer_agent: str,
        code: PackedWriteFailureCode,
        reason: str,
    ) -> None:
        """Release and notify one command proven never to have posted.

        :param request: Exact admitted producer command.
        :param lease: Generation-scoped local gather slot.
        :param consumer_agent: Authenticated decoder agent.
        :param code: Stable pre-WRITE failure class.
        :param reason: Human-readable bounded detail.
        """
        pool = self._packed_producer_pool
        if pool is None:
            raise StagingSafetyError("packed WRITE producer pool disappeared")
        failed = PackedWriteFailedBeforeWrite(
            command=request.command,
            code=code,
            reason=reason[:1024],
        )
        pool.fail_before_write(lease, failed)
        self.nixl_wrapper.send_notif(
            consumer_agent,
            notif_msg=encode_packed_write_failed_before_write_notification(failed),
        )

    def _service_packed_producer(self) -> None:
        """Advance producer gathers, WRITEs, and bounded pending commands."""
        if self._packed_write_config.enabled is False:
            return
        if self.kv_transfer_config.kv_role not in {"kv_producer", "kv_both"}:
            return
        geometry = self._packed_write_producer_pool_geometry
        buffer = self._packed_write_producer_pool_buffer
        stream = self._packed_write_producer_pack_stream
        if geometry is None or buffer is None or stream is None:
            raise StagingSafetyError("packed WRITE producer resources disappeared")
        pool = self._ensure_packed_producer_pool()

        for identity, operation in tuple(self._packed_producer_operations.items()):
            if operation.write_handle is None:
                self._advance_packed_producer_pack(operation)
            if operation.write_handle is not None:
                self._advance_packed_producer_write(operation)
            if pool.request_for_identity(identity) is None:
                self._packed_producer_operations.pop(identity, None)

        self._service_packed_producer_watchdogs()

        pending_count = len(self._packed_producer_pending)
        for _ in range(pending_count):
            pending = self._packed_producer_pending.popleft()
            request = pending.request
            identity = request.command.chunk
            retained_terminal = pool.terminal_for_replay(
                request,
                pending.consumer_agent,
            )
            if retained_terminal is not None:
                self._send_packed_producer_terminal(
                    pending.consumer_agent,
                    retained_terminal,
                )
                continue
            roster = self._source_rosters.get(identity.producer_request_id)
            source_ready_event = self._packed_source_ready_events.get(
                identity.producer_request_id
            )
            age_s = time.monotonic() - pending.received_at
            if roster is None or source_ready_event is None:
                if (
                    pending.warning_emitted is False
                    and age_s >= self._packed_write_config.warn_after_s
                ):
                    pending.warning_emitted = True
                    logger.warning(
                        "packed WRITE request %s chunk %d has waited %.3fs for "
                        "its source roster",
                        identity.producer_request_id,
                        identity.chunk_ordinal,
                        age_s,
                    )
                if age_s >= self._packed_write_config.fail_after_s:
                    self._send_unadmitted_packed_failure(
                        pending,
                        PackedWriteFailureCode.SOURCE_UNAVAILABLE,
                        "producer source roster wait timed out",
                    )
                    continue
                self._packed_producer_pending.append(pending)
                continue

            reconstructed = pending.reconstructed
            if reconstructed is None:
                try:
                    reconstructed = reconstruct_producer_packed_write_plan(
                        roster=roster,
                        regions=self._region_descriptors,
                        request=request,
                        producer_engine_id=self.engine_id,
                        producer_rank=self.tp_rank,
                        producer_tp_size=self.world_size,
                        max_chunk_bytes_per_rank=(
                            self._packed_write_config.chunk_bytes_per_rank
                        ),
                    )
                except (ValueError, RuntimeError):
                    self._send_unadmitted_packed_failure(
                        pending,
                        PackedWriteFailureCode.INVALID_REQUEST,
                        traceback.format_exc(),
                    )
                    continue
                pending.reconstructed = reconstructed

            admission = pool.admit(request, pending.consumer_agent)
            if admission.terminal is not None:
                self._send_packed_producer_terminal(
                    pending.consumer_agent,
                    admission.terminal,
                )
                continue
            if admission.newly_admitted is False:
                if admission.lease is None:
                    if age_s >= self._packed_write_config.fail_after_s:
                        self._send_unadmitted_packed_failure(
                            pending,
                            PackedWriteFailureCode.POOL_TIMEOUT,
                            "producer gather-pool wait timed out",
                        )
                    else:
                        self._packed_producer_pending.appendleft(pending)
                    break
                continue
            lease = admission.lease
            if lease is None:
                raise StagingSafetyError("new packed WRITE admission has no lease")
            sources = (
                ()
                if self._region_rows is None
                else tuple(row.reshape(-1) for row in self._region_rows)
            )
            if len(sources) != len(reconstructed.source_plan.regions):
                self._fail_admitted_packed_write(
                    request,
                    lease,
                    pending.consumer_agent,
                    PackedWriteFailureCode.SOURCE_UNAVAILABLE,
                    "canonical producer region rows are unavailable",
                )
                continue
            pack_failure_reason: str | None
            try:
                launch = launch_packed_chunk_pack(
                    buffer,
                    sources,
                    reconstructed.source_plan,
                    reconstructed.packed_plan,
                    reconstructed.chunk.chunk_index,
                    stream,
                    destination_base_offset_bytes=(
                        lease.slot_index * geometry.slot_size_bytes
                    ),
                    source_ready_event=source_ready_event,
                )
            except PackEnqueueError as error:
                if error.recovery_launch is None:
                    self._fail_admitted_packed_write(
                        request,
                        lease,
                        pending.consumer_agent,
                        PackedWriteFailureCode.PACK_FAILED,
                        str(error),
                    )
                    continue
                launch = error.recovery_launch
                pack_failure_reason = str(error)
            except Exception:
                self._fail_admitted_packed_write(
                    request,
                    lease,
                    pending.consumer_agent,
                    PackedWriteFailureCode.PACK_FAILED,
                    traceback.format_exc(),
                )
                continue
            else:
                pack_failure_reason = None
            now = time.monotonic()
            self._packed_producer_operations[identity] = _ProducerPackedOperation(
                request=request,
                source_plan=reconstructed.source_plan,
                packed_plan=reconstructed.packed_plan,
                lease=lease,
                pack_launch=launch,
                consumer_agent=pending.consumer_agent,
                created_at=now,
                stage_started_at=now,
                pack_failure_reason=pack_failure_reason,
            )

    def _advance_packed_producer_pack(
        self,
        operation: _ProducerPackedOperation,
    ) -> None:
        """Move one gather from CUDA completion into native WRITE posting.

        :param operation: Live producer operation.
        """
        pool = self._packed_producer_pool
        if pool is None:
            raise StagingSafetyError("packed WRITE producer pool disappeared")
        try:
            if operation.pack_launch.is_complete() is False:
                return
        except Exception as error:
            pool.tombstone(
                operation.lease,
                "producer pack completion event query failed",
            )
            raise StagingSafetyError(
                "producer pack completion event query failed\n" + traceback.format_exc()
            ) from error
        if operation.pack_failure_reason is not None:
            self._fail_admitted_packed_write(
                operation.request,
                operation.lease,
                operation.consumer_agent,
                PackedWriteFailureCode.PACK_FAILED,
                operation.pack_failure_reason,
            )
            return

        pool.mark_pack_complete(operation.lease)
        endpoint = operation.request.consumer_endpoint
        handle: Any | None = None
        try:
            local_descs = self.nixl_wrapper.get_xfer_descs(
                [
                    (
                        operation.lease.source_address,
                        operation.lease.payload_bytes,
                        self.device_id,
                    )
                ],
                self.nixl_memory_type,
            )
            destination_address = packed_write_destination_address(
                operation.request.command,
                endpoint.consumer_pool,
            )
            remote_descs = self.nixl_wrapper.get_xfer_descs(
                [
                    (
                        destination_address,
                        operation.lease.payload_bytes,
                        endpoint.consumer_pool.device_id,
                    )
                ],
                self.nixl_memory_type,
            )
            arrived = PackedWriteArrived(command=operation.request.command)
            handle = self.nixl_wrapper.initialize_xfer(
                "WRITE",
                local_descs,
                remote_descs,
                operation.consumer_agent,
                encode_packed_write_arrived_notification(arrived),
            )
        except Exception:
            reason = "packed WRITE native preparation failed\n" + traceback.format_exc()
            if handle is not None:
                try:
                    self.nixl_wrapper.release_xfer_handle(handle)
                except Exception as error:
                    pool.tombstone(operation.lease, reason)
                    raise StagingSafetyError(
                        reason
                        + "\nprepared handle release also failed\n"
                        + traceback.format_exc()
                    ) from error
            self._fail_admitted_packed_write(
                operation.request,
                operation.lease,
                operation.consumer_agent,
                PackedWriteFailureCode.WRITE_PREPARE_FAILED,
                reason,
            )
            return

        try:
            pool.begin_write_post(operation.lease)
        except Exception as error:
            reason = "packed WRITE post-boundary transition failed"
            try:
                self.nixl_wrapper.release_xfer_handle(handle)
            except Exception as release_error:
                pool.tombstone(operation.lease, reason)
                raise StagingSafetyError(
                    reason
                    + "\nprepared handle release also failed\n"
                    + traceback.format_exc()
                ) from release_error
            pool.tombstone(operation.lease, reason)
            raise StagingSafetyError(reason) from error
        try:
            status = self.nixl_wrapper.transfer(handle)
        except Exception as error:
            pool.tombstone(
                operation.lease,
                "native WRITE submission became uncertain",
            )
            raise StagingSafetyError(
                "packed WRITE submission became uncertain\n" + traceback.format_exc()
            ) from error
        pool.record_write_posted(operation.lease, status)
        operation.write_handle = handle
        operation.write_status = status
        operation.stage_started_at = time.monotonic()
        operation.warning_emitted = False

    def _advance_packed_producer_write(
        self,
        operation: _ProducerPackedOperation,
    ) -> None:
        """Poll one posted WRITE and release its local gather slot at DONE.

        :param operation: Live producer operation with a native handle.
        """
        pool = self._packed_producer_pool
        handle = operation.write_handle
        status = operation.write_status
        if pool is None or handle is None or status is None:
            raise StagingSafetyError("packed WRITE operation lost native ownership")
        if status == "PROC":
            try:
                status = self.nixl_wrapper.check_xfer_state(handle)
            except Exception as error:
                pool.tombstone(operation.lease, "native WRITE query failed")
                raise StagingSafetyError(
                    "packed WRITE status query failed\n" + traceback.format_exc()
                ) from error
            pool.record_write_query(operation.lease, status)
            operation.write_status = status
        if status == "PROC":
            return
        if status != "DONE":
            pool.tombstone(operation.lease, f"native WRITE became {status!r}")
            raise StagingSafetyError(f"packed WRITE became {status!r}")
        try:
            telemetry = self.nixl_wrapper.get_xfer_telemetry(handle)
        except Exception:
            telemetry = None
            logger.error(
                "packed WRITE telemetry unavailable after DONE\n%s",
                traceback.format_exc(),
            )
        if telemetry is not None:
            try:
                self.xfer_stats.record_transfer(telemetry)
            except Exception:
                logger.error(
                    "packed WRITE telemetry recording failed\n%s",
                    traceback.format_exc(),
                )
        try:
            self.nixl_wrapper.release_xfer_handle(handle)
        except Exception as error:
            pool.tombstone(operation.lease, "DONE WRITE handle release failed")
            raise StagingSafetyError(
                "packed WRITE handle release failed after DONE\n"
                + traceback.format_exc()
            ) from error
        pool.mark_write_released(operation.lease)
        arrived = pool.complete_arrived(operation.lease)
        if arrived.command != operation.request.command:
            raise StagingSafetyError("packed WRITE terminal history changed command")

    def _service_packed_producer_watchdogs(self) -> None:
        """Bound every producer device/native actor without unsafe replay."""
        now = time.monotonic()
        warn_after_s = self._packed_write_config.warn_after_s
        fail_after_s = self._packed_write_config.fail_after_s
        for operation in self._packed_producer_operations.values():
            age_s = now - operation.stage_started_at
            stage = "pack" if operation.write_handle is None else "native WRITE"
            if operation.warning_emitted is False and age_s >= warn_after_s:
                operation.warning_emitted = True
                logger.warning(
                    "packed WRITE request %s chunk %d %s has remained active for %.3fs",
                    operation.request.command.chunk.producer_request_id,
                    operation.request.command.chunk.chunk_ordinal,
                    stage,
                    age_s,
                )
            if age_s < fail_after_s:
                continue
            reason = (
                f"packed WRITE producer {stage} timeout for request "
                f"{operation.request.command.chunk.producer_request_id} chunk "
                f"{operation.request.command.chunk.chunk_ordinal} after {age_s:.3f}s"
            )
            if self._packed_producer_pool is None:
                raise StagingSafetyError(reason)
            self._packed_producer_pool.tombstone(operation.lease, reason)
            raise StagingSafetyError(reason)

    @staticmethod
    def _packed_source_selections(
        source_plan: CanonicalSourceTransferPlan,
    ) -> tuple[PackedWriteSourceSelection, ...]:
        """Encode the complete canonical selected-source suffixes.

        :param source_plan: Trusted complete selected-source plan.
        :returns: One ordered selection for every canonical cache group.
        """
        return tuple(
            PackedWriteSourceSelection(
                group_index=group.group_index,
                source_position_start=group.source_position_start,
                position_count=len(group.remote_block_ids),
            )
            for group in source_plan.groups
        )

    def _ensure_packed_consumer_pool(self) -> PackedWriteConsumerPool:
        """Return the decoder pool after validating its registered storage.

        :returns: Generation-owning decoder slot pool.
        """
        if self._packed_consumer_pool is not None:
            return self._packed_consumer_pool
        geometry = self._packed_write_consumer_pool_geometry
        buffer = self._packed_write_consumer_pool_buffer
        scatter_stream = self._packed_write_scatter_stream
        if geometry is None or buffer is None or scatter_stream is None:
            raise StagingSafetyError(
                "packed WRITE consumer registration is unavailable"
            )
        if (
            buffer.data_ptr() != geometry.base_address
            or buffer.numel() != geometry.registered_bytes
            or geometry.source_tp_size != 4
            or geometry.rank_stride_bytes
            != self._packed_write_config.chunk_bytes_per_rank
            or geometry.slot_count != self._packed_write_config.consumer_slot_count
        ):
            raise StagingSafetyError(
                "packed WRITE consumer registration differs from configuration"
            )
        self._packed_consumer_pool = PackedWriteConsumerPool(geometry)
        return self._packed_consumer_pool

    def _packed_write_request(
        self,
        *,
        req_id: str,
        meta: ReqMeta,
        read_specs: list[ReadSpec],
        source_plan: CanonicalSourceTransferPlan,
        destination_plan: CoalescedTransferPlan,
        logical_bytes: int,
        direct_descriptor_count: int,
    ) -> str:
        """Install one adaptive producer-packed WRITE request atomically.

        :param req_id: Decoder request identifier.
        :param meta: Immutable transfer metadata.
        :param read_specs: Exact producer-rank quorum.
        :param source_plan: Destination-independent selected-source plan.
        :param destination_plan: Exact decoder placement plan.
        :param logical_bytes: Unpruned logical transfer bytes.
        :param direct_descriptor_count: Equivalent direct descriptor count.
        :returns: ``posted`` after owned request installation.
        """
        if meta.remote is None:
            raise StagingSafetyError("packed WRITE request has no producer metadata")
        if req_id in self._packed_consumer_requests:
            raise StagingSafetyError(f"packed WRITE request {req_id} is duplicated")
        if source_plan.source_tp_size != 4 or self.world_size != 1:
            raise StagingSafetyError("packed WRITE requires TP4 producer to TP1 decode")
        source_ranks = tuple(int(spec.remote_rank) for spec in read_specs)
        if source_plan.source_ranks != source_ranks:
            raise StagingSafetyError("packed WRITE source-rank order changed")
        if (
            type(meta.remote.source_offer_generation) is not int
            or meta.remote.source_offer_generation < 1
            or meta.remote.remote_num_tokens <= 0
        ):
            raise StagingSafetyError("packed WRITE producer lifetime is incomplete")
        self._ensure_packed_consumer_pool()
        consumer_geometry = self._packed_write_consumer_pool_geometry
        if consumer_geometry is None:
            raise StagingSafetyError("packed WRITE consumer geometry disappeared")
        engine_id = meta.remote.engine_id
        remote_pools = self._remote_packed_write_producer_pools.get(engine_id)
        if remote_pools is None or set(remote_pools) != set(source_ranks):
            raise StagingSafetyError(
                "packed WRITE lacks an authenticated producer-pool quorum"
            )
        for source_rank in source_ranks:
            geometry = remote_pools[source_rank]
            if (
                geometry.slot_size_bytes
                != self._packed_write_config.chunk_bytes_per_rank
                or geometry.slot_count != self._packed_write_config.producer_slot_count
                or geometry.alignment_bytes != self._packed_write_config.alignment_bytes
            ):
                raise StagingSafetyError(
                    f"packed WRITE producer rank {source_rank} pool differs"
                )

        packed_plan = build_packed_transfer_plan(
            source_plan,
            max_chunk_bytes_per_rank=(self._packed_write_config.chunk_bytes_per_rank),
        )
        if len(packed_plan.chunks) == 0:
            raise StagingSafetyError("packed WRITE selected no source bytes")
        endpoint = PackedWriteConsumerEndpoint(
            consumer_engine_id=self.engine_id,
            consumer_rank=self.tp_rank,
            consumer_tp_size=self.world_size,
            registration_generation=self._registration_generation,
            agent_metadata=self.nixl_wrapper.get_agent_metadata(),
            consumer_pool=consumer_geometry,
        )
        selections = self._packed_source_selections(source_plan)
        identities_by_chunk: list[tuple[PackedWriteChunkIdentity, ...]] = []
        for chunk in packed_plan.chunks:
            identities_by_chunk.append(
                tuple(
                    PackedWriteChunkIdentity(
                        producer_engine_id=engine_id,
                        producer_request_id=meta.remote.request_id,
                        offer_generation=meta.remote.source_offer_generation,
                        producer_rank=source_rank,
                        producer_tp_size=source_plan.source_tp_size,
                        consumer_engine_id=self.engine_id,
                        consumer_request_id=req_id,
                        consumer_rank=self.tp_rank,
                        consumer_tp_size=self.world_size,
                        source_plan_digest=source_plan.digest,
                        packed_plan_digest=packed_plan.digest,
                        chunk_plan_digest=chunk.digest,
                        chunk_ordinal=chunk.chunk_index,
                        chunk_count=len(packed_plan.chunks),
                        valid_token_extent=meta.remote.remote_num_tokens,
                        exact_bytes=chunk.rank_stride_bytes,
                    )
                    for source_rank in source_ranks
                )
            )
        first_key = PackedWriteChunkKey.from_identity(identities_by_chunk[0][0])
        owner = _ConsumerPackedRequest(
            request_id=req_id,
            meta=meta,
            read_specs=read_specs,
            source_plan=source_plan,
            destination_plan=destination_plan,
            packed_plan=packed_plan,
            consumer_endpoint=endpoint,
            identities_by_chunk=tuple(identities_by_chunk),
            source_selections=selections,
            tracker=PackedWriteRequestTracker.create(first_key),
            direct_descriptor_count=direct_descriptor_count,
            logical_bytes=logical_bytes,
            created_at=time.monotonic(),
            pending_chunk_indices=deque(range(len(packed_plan.chunks))),
        )
        self._packed_consumer_requests[req_id] = owner
        if self._phase_separate_transfer_decode:
            self._transfer_phase_plan_count += 1
            self._transfer_phase_byte_count += packed_plan.transfer_size_bytes
        self._service_packed_consumer()
        logger.info(
            "[packed-write-start] request=%s source_digest=%s packed_digest=%s "
            "source_bytes=%d chunks=%d direct_descriptors=%d",
            req_id,
            source_plan.digest,
            packed_plan.digest,
            source_plan.transfer_size_bytes,
            len(packed_plan.chunks),
            direct_descriptor_count,
        )
        return "posted"

    def _admit_packed_chunks(self) -> None:
        """Fill every free decoder slot in request-creation order."""
        pool = self._ensure_packed_consumer_pool()
        while pool.free_slot_count > 0:
            owner = next(
                (
                    candidate
                    for candidate in sorted(
                        self._packed_consumer_requests.values(),
                        key=lambda value: value.created_at,
                    )
                    if candidate.failure_reason is None
                    and len(candidate.pending_chunk_indices) > 0
                ),
                None,
            )
            if owner is None:
                return
            chunk_index = owner.pending_chunk_indices.popleft()
            identities = owner.identities_by_chunk[chunk_index]
            chunk_key = PackedWriteChunkKey.from_identity(identities[0])
            lease = pool.reserve(chunk_key, identities[0].exact_bytes)
            if lease is None:
                owner.pending_chunk_indices.appendleft(chunk_index)
                return
            requests = tuple(
                PackedWriteRequest(
                    command=PackedWriteCommand(
                        chunk=identity,
                        destination=lease.binding,
                    ),
                    consumer_endpoint=owner.consumer_endpoint,
                    source_selections=owner.source_selections,
                )
                for identity in identities
            )
            if chunk_index != owner.published_chunk_count:
                raise StagingSafetyError(
                    "packed WRITE chunks were published outside canonical order"
                )
            if chunk_index == 0:
                owner.release_commands_by_producer_rank = {
                    request.command.chunk.producer_rank: request.command
                    for request in requests
                }
            else:
                for request in requests:
                    rank = request.command.chunk.producer_rank
                    anchor = owner.release_commands_by_producer_rank.get(rank)
                    if anchor is None:
                        raise StagingSafetyError(
                            "packed WRITE request lost its source-drain anchor"
                        )
                    anchor_key = PackedWriteRequestKey.from_chunk(
                        PackedWriteChunkKey.from_identity(anchor.chunk)
                    )
                    request_key = PackedWriteRequestKey.from_chunk(
                        PackedWriteChunkKey.from_identity(request.command.chunk)
                    )
                    if request_key != anchor_key:
                        raise StagingSafetyError(
                            "packed WRITE chunk changed its source-drain identity"
                        )
            pool.bind_requests(lease, requests)
            chunk = _ConsumerPackedChunk(
                request_id=owner.request_id,
                chunk_index=chunk_index,
                requests=requests,
                lease=lease,
                admitted_at=time.monotonic(),
            )
            owner.active_chunks[chunk_index] = chunk
            self._packed_consumer_chunks[chunk_key] = chunk
            try:
                for request in requests:
                    identity = request.command.chunk
                    agent = self._remote_agents[identity.producer_engine_id][
                        identity.producer_rank
                    ]
                    self.nixl_wrapper.send_notif(
                        agent,
                        notif_msg=encode_packed_write_request_notification(request),
                    )
                    pool.mark_rank_published(lease, identity.producer_rank)
                pool.seal_publication(lease)
                owner.published_chunk_count += 1
            except Exception as error:
                reason = (
                    "packed WRITE command publication became uncertain\n"
                    + traceback.format_exc()
                )
                pool.tombstone(lease, reason)
                owner.tracker.tombstone()
                raise StagingSafetyError(reason) from error

    def _require_packed_producer_sender(
        self,
        identity: PackedWriteChunkIdentity,
        notifying_agent: str,
    ) -> None:
        """Authenticate one producer-rank terminal sender.

        :param identity: Rankful packed chunk identity.
        :param notifying_agent: Native sender reported by NIXL.
        """
        rank_agents = self._remote_agents.get(identity.producer_engine_id)
        expected_agent = (
            None if rank_agents is None else rank_agents.get(identity.producer_rank)
        )
        if expected_agent is None or notifying_agent != expected_agent:
            raise StagingSafetyError(
                "packed WRITE terminal sender is not the authenticated rank agent"
            )

    @staticmethod
    def _packed_chunk_rank_request(
        chunk: _ConsumerPackedChunk,
        identity: PackedWriteChunkIdentity,
    ) -> PackedWriteRequest:
        """Resolve one exact rankful request from an active chunk.

        :param chunk: Active rank-independent decoder chunk.
        :param identity: Rankful terminal identity.
        :returns: Exact published producer-rank request.
        """
        request = next(
            (
                candidate
                for candidate in chunk.requests
                if candidate.command.chunk.producer_rank == identity.producer_rank
            ),
            None,
        )
        if request is None or request.command.chunk != identity:
            raise StagingSafetyError(
                "packed WRITE terminal differs from its active request"
            )
        return request

    def _accept_packed_arrived_notification(
        self,
        arrived: PackedWriteArrived,
        notifying_agent: str,
    ) -> None:
        """Authenticate an attached WRITE-arrival proof or exact replay.

        :param arrived: Native-attached remote-memory completion proof.
        :param notifying_agent: Native sender reported by NIXL.
        """
        identity = arrived.command.chunk
        self._require_packed_producer_sender(identity, notifying_agent)
        chunk = self._packed_consumer_chunks.get(
            PackedWriteChunkKey.from_identity(identity)
        )
        if chunk is not None:
            request = self._packed_chunk_rank_request(chunk, identity)
            try:
                validate_packed_write_arrived(request, arrived)
            except ValueError as error:
                raise StagingSafetyError(
                    "invalid packed WRITE arrival for active ownership"
                ) from error
            self._packed_terminal_notifications.append((arrived, notifying_agent))
            return
        terminal = self._packed_consumer_terminals.get(identity)
        if terminal is None:
            raise StagingSafetyError("packed WRITE arrival has no decoder owner")
        try:
            validate_packed_write_arrived(terminal.request, arrived)
        except ValueError as error:
            raise StagingSafetyError(
                "late packed WRITE arrival differs from retired ownership"
            ) from error
        if terminal.terminal != arrived:
            raise StagingSafetyError("packed WRITE rank emitted conflicting terminals")

    def _accept_packed_failed_notification(
        self,
        failed: PackedWriteFailedBeforeWrite,
        notifying_agent: str,
    ) -> None:
        """Authenticate a failed-before-WRITE proof or exact replay.

        :param failed: Exact proof that one producer rank never posted.
        :param notifying_agent: Native sender reported by NIXL.
        """
        identity = failed.command.chunk
        self._require_packed_producer_sender(identity, notifying_agent)
        chunk = self._packed_consumer_chunks.get(
            PackedWriteChunkKey.from_identity(identity)
        )
        if chunk is not None:
            request = self._packed_chunk_rank_request(chunk, identity)
            try:
                validate_packed_write_failed_before_write(request, failed)
            except ValueError as error:
                raise StagingSafetyError(
                    "invalid packed WRITE failure for active ownership"
                ) from error
            self._packed_terminal_notifications.append((failed, notifying_agent))
            return
        terminal = self._packed_consumer_terminals.get(identity)
        if terminal is None:
            raise StagingSafetyError("packed WRITE failure has no decoder owner")
        try:
            validate_packed_write_failed_before_write(terminal.request, failed)
        except ValueError as error:
            raise StagingSafetyError(
                "late packed WRITE failure differs from retired ownership"
            ) from error
        if terminal.terminal != failed:
            raise StagingSafetyError("packed WRITE rank emitted conflicting terminals")

    def _consume_packed_terminal_notifications(self) -> None:
        """Advance decoder chunks from exact rank terminals to scatter/discard."""
        pool = self._ensure_packed_consumer_pool()
        while len(self._packed_terminal_notifications) > 0:
            terminal, notifying_agent = self._packed_terminal_notifications.popleft()
            identity = terminal.command.chunk
            key = PackedWriteChunkKey.from_identity(identity)
            chunk = self._packed_consumer_chunks.get(key)
            if chunk is None:
                if isinstance(terminal, PackedWriteArrived):
                    self._accept_packed_arrived_notification(
                        terminal,
                        notifying_agent,
                    )
                else:
                    self._accept_packed_failed_notification(
                        terminal,
                        notifying_agent,
                    )
                continue
            rank = identity.producer_rank
            prior = chunk.terminals_by_rank.get(rank)
            if prior is not None:
                if prior != terminal:
                    raise StagingSafetyError("packed WRITE rank terminal conflicted")
                continue
            chunk.terminals_by_rank[rank] = terminal
            if isinstance(terminal, PackedWriteArrived):
                complete = pool.record_arrived(chunk.lease, terminal)
            else:
                complete = pool.record_failed_before_write(chunk.lease, terminal)
            if complete is False or chunk.terminal_at is not None:
                continue
            chunk.terminal_at = time.monotonic()
            owner = self._packed_consumer_requests[chunk.request_id]
            owner.arrival_wait_seconds += chunk.terminal_at - chunk.admitted_at
            if pool.has_failure(chunk.lease):
                first_failure = next(
                    value
                    for value in chunk.terminals_by_rank.values()
                    if isinstance(value, PackedWriteFailedBeforeWrite)
                )
                self._begin_packed_request_failure(
                    owner,
                    "producer rank "
                    f"{first_failure.command.chunk.producer_rank} rejected packed "
                    f"WRITE: {first_failure.code.value}: {first_failure.reason}",
                )
                self._discard_packed_chunk(owner, chunk)
                continue
            if owner.failure_reason is not None:
                self._discard_packed_chunk(owner, chunk)
                continue
            self._enqueue_packed_scatter(owner, chunk)

    def _remember_packed_chunk_terminals(
        self,
        chunk: _ConsumerPackedChunk,
    ) -> None:
        """Retain every exact rank terminal before releasing its slot.

        :param chunk: Terminal-complete decoder chunk.
        """
        if len(chunk.terminals_by_rank) != len(chunk.requests):
            raise StagingSafetyError("packed WRITE chunk terminal quorum is incomplete")
        missing = sum(
            request.command.chunk not in self._packed_consumer_terminals
            for request in chunk.requests
        )
        self._ensure_packed_consumer_terminal_capacity(missing)
        for request in chunk.requests:
            rank = request.command.chunk.producer_rank
            terminal = chunk.terminals_by_rank[rank]
            record = _ConsumerPackedTerminal(request=request, terminal=terminal)
            prior = self._packed_consumer_terminals.get(request.command.chunk)
            if prior is not None and prior != record:
                raise StagingSafetyError("packed WRITE consumer terminal conflict")
            self._packed_consumer_terminals[request.command.chunk] = record

    def _ensure_packed_consumer_terminal_capacity(self, new_count: int) -> None:
        """Preserve a bounded exact-replay window without failing normal traffic.

        Only histories from fully retired decoder requests are eligible. A late
        terminal outside this window still fails closed as ownerless instead of
        being applied to a reused slot generation.

        :param new_count: New exact rank terminals about to be retained.
        """
        if type(new_count) is not int or new_count < 0:
            raise ValueError("new_count must be a non-negative integer")
        required = (
            len(self._packed_consumer_terminals)
            + new_count
            - _MAX_PACKED_WRITE_CONSUMER_TERMINALS
        )
        if required <= 0:
            return
        active_requests = {
            PackedWriteRequestKey.from_chunk(
                PackedWriteChunkKey.from_identity(owner.identities_by_chunk[0][0])
            )
            for owner in self._packed_consumer_requests.values()
        }
        eligible = tuple(
            identity
            for identity in self._packed_consumer_terminals
            if PackedWriteRequestKey.from_chunk(
                PackedWriteChunkKey.from_identity(identity)
            )
            not in active_requests
        )
        if len(eligible) < required:
            raise StagingSafetyError(
                "packed WRITE active terminal ownership exceeds its replay window"
            )
        trim_count = min(
            len(eligible),
            max(required, _PACKED_WRITE_CONSUMER_TERMINAL_TRIM),
        )
        for identity in eligible[:trim_count]:
            del self._packed_consumer_terminals[identity]

    def _discard_packed_chunk(
        self,
        owner: _ConsumerPackedRequest,
        chunk: _ConsumerPackedChunk,
    ) -> None:
        """Release a complete rank quorum without exposing its staging data.

        :param owner: Request-wide atomic publication owner.
        :param chunk: Terminal-complete chunk to discard.
        """
        pool = self._ensure_packed_consumer_pool()
        self._remember_packed_chunk_terminals(chunk)
        completed_key = pool.discard_terminal(chunk.lease)
        if completed_key != chunk.lease.chunk:
            raise StagingSafetyError("packed WRITE discarded the wrong chunk")
        del owner.active_chunks[chunk.chunk_index]
        del self._packed_consumer_chunks[completed_key]
        self._finish_packed_failed_request(owner)

    def _enqueue_packed_scatter(
        self,
        owner: _ConsumerPackedRequest,
        chunk: _ConsumerPackedChunk,
    ) -> None:
        """Launch one event-owned scatter after an all-ARRIVED quorum.

        :param owner: Request-wide atomic publication owner.
        :param chunk: Terminal-complete all-ARRIVED chunk.
        """
        pool = self._ensure_packed_consumer_pool()
        buffer = self._packed_write_consumer_pool_buffer
        geometry = self._packed_write_consumer_pool_geometry
        scatter_stream = self._packed_write_scatter_stream
        if (
            buffer is None
            or geometry is None
            or scatter_stream is None
            or self._region_rows is None
        ):
            raise StagingSafetyError("packed WRITE scatter resources disappeared")
        pool.begin_scatter(chunk.lease)
        destinations = tuple(row.reshape(-1) for row in self._region_rows)
        enqueue_started_at = time.monotonic()
        try:
            launch = launch_packed_chunk_scatter(
                buffer,
                destinations,
                owner.destination_plan,
                owner.packed_plan,
                chunk.chunk_index,
                scatter_stream,
                staging_base_offset_bytes=(
                    chunk.lease.slot_index * geometry.slot_size_bytes
                ),
                staging_rank_stride_bytes=geometry.rank_stride_bytes,
            )
        except ScatterEnqueueError as error:
            chunk.scatter_enqueue_duration_seconds = (
                time.monotonic() - enqueue_started_at
            )
            if error.recovery_launch is None:
                reason = str(error)
                pool.tombstone(chunk.lease, reason)
                owner.tracker.tombstone()
                raise StagingSafetyError(reason) from error
            chunk.scatter_launch = error.recovery_launch
            chunk.scatter_enqueued_at = time.monotonic()
            chunk.failure_reason = str(error)
            self._begin_packed_request_failure(owner, str(error))
            return
        except Exception as error:
            reason = "packed WRITE scatter enqueue became uncertain\n" + (
                traceback.format_exc()
            )
            pool.tombstone(chunk.lease, reason)
            owner.tracker.tombstone()
            raise StagingSafetyError(reason) from error
        chunk.scatter_enqueue_duration_seconds = time.monotonic() - enqueue_started_at
        chunk.scatter_launch = launch
        chunk.scatter_enqueued_at = time.monotonic()

    def _poll_packed_scatters(self) -> None:
        """Release completed decoder slots and atomically publish requests."""
        pool = self._ensure_packed_consumer_pool()
        for owner in tuple(self._packed_consumer_requests.values()):
            for chunk in tuple(owner.active_chunks.values()):
                launch = chunk.scatter_launch
                if launch is None:
                    continue
                try:
                    if launch.is_complete() is False:
                        continue
                except Exception as error:
                    reason = "packed WRITE scatter completion query failed\n" + (
                        traceback.format_exc()
                    )
                    pool.tombstone(chunk.lease, reason)
                    owner.tracker.tombstone()
                    raise StagingSafetyError(reason) from error
                completed_at = time.monotonic()
                try:
                    gpu_duration = launch.gpu_duration_ms() / 1_000
                except Exception:
                    gpu_duration = 0.0
                    owner.scatter_gpu_duration_available = False
                    logger.error(
                        "packed WRITE scatter GPU timing unavailable\n%s",
                        traceback.format_exc(),
                    )
                self._remember_packed_chunk_terminals(chunk)
                completed_key = pool.complete_scatter(chunk.lease)
                if completed_key != chunk.lease.chunk:
                    raise StagingSafetyError(
                        "packed WRITE scatter released wrong chunk"
                    )
                if owner.failure_reason is None and chunk.failure_reason is None:
                    owner.tracker.record_chunk_complete(completed_key)
                elif owner.failure_reason is None:
                    self._begin_packed_request_failure(
                        owner,
                        chunk.failure_reason,
                    )
                owner.scatter_enqueue_duration_seconds += (
                    chunk.scatter_enqueue_duration_seconds
                )
                if chunk.scatter_enqueued_at is None:
                    raise StagingSafetyError(
                        "packed WRITE scatter completion has no start time"
                    )
                owner.scatter_wall_duration_seconds += (
                    completed_at - chunk.scatter_enqueued_at
                )
                owner.scatter_gpu_duration_seconds += gpu_duration
                del owner.active_chunks[chunk.chunk_index]
                del self._packed_consumer_chunks[completed_key]
            if owner.failure_reason is not None:
                self._finish_packed_failed_request(owner)
                continue
            if owner.tracker.publication_eligible:
                self._complete_packed_request(owner)

    def _send_final_packed_source_release(
        self,
        owner: _ConsumerPackedRequest,
    ) -> None:
        """Send the final source-page release proof to every producer rank.

        :param owner: Fully drained packed request.
        """
        if owner.meta.remote is None:
            raise StagingSafetyError("packed WRITE owner lost producer metadata")
        producer_ranks = {int(read_spec.remote_rank) for read_spec in owner.read_specs}
        if (
            owner.published_chunk_count < 1
            or set(owner.release_commands_by_producer_rank) != producer_ranks
        ):
            raise StagingSafetyError(
                "packed WRITE source release lacks its published rank anchors"
            )
        try:
            for read_spec in owner.read_specs:
                producer_rank = int(read_spec.remote_rank)
                drained = PackedWriteSourceDrained(
                    anchor=owner.release_commands_by_producer_rank[producer_rank],
                    published_chunk_count=owner.published_chunk_count,
                )
                self.nixl_wrapper.send_notif(
                    self._remote_agents[owner.meta.remote.engine_id][producer_rank],
                    notif_msg=encode_packed_write_source_drained_notification(drained),
                )
        except Exception as error:
            owner.tracker.tombstone()
            raise StagingSafetyError(
                "final packed WRITE source release became uncertain\n"
                + traceback.format_exc()
            ) from error
        self._record_remote_source_consumption_proven(
            owner.request_id,
            owner.meta,
        )

    def _complete_packed_request(self, owner: _ConsumerPackedRequest) -> None:
        """Expose one fully scattered request after final source release.

        :param owner: Successful request with no remaining chunk ownership.
        """
        if len(owner.pending_chunk_indices) > 0 or len(owner.active_chunks) > 0:
            raise StagingSafetyError(
                "packed WRITE request became eligible while active"
            )
        self._send_final_packed_source_release(owner)
        owner.tracker.mark_published()
        finished_at = time.monotonic()
        telemetry: NixlPackedWriteTelemetry | None = None
        try:
            telemetry = NixlPackedWriteTelemetry.from_completed_request(
                logical_bytes=owner.logical_bytes,
                source_bytes=owner.source_plan.transfer_size_bytes,
                direct_descriptor_count=owner.direct_descriptor_count,
                chunk_count=len(owner.packed_plan.chunks),
                source_tp_size=owner.source_plan.source_tp_size,
                arrival_wait_seconds=owner.arrival_wait_seconds,
                scatter_enqueue_duration_seconds=(
                    owner.scatter_enqueue_duration_seconds
                ),
                scatter_wall_duration_seconds=owner.scatter_wall_duration_seconds,
                scatter_gpu_duration_seconds=(
                    owner.scatter_gpu_duration_seconds
                    if owner.scatter_gpu_duration_available
                    else None
                ),
                staging_residency_seconds=finished_at - owner.created_at,
            )
            self.xfer_stats.record_packed_write(telemetry)
        except Exception:
            logger.error(
                "packed WRITE telemetry aggregation failed after request %s\n%s",
                owner.request_id,
                traceback.format_exc(),
            )
        self._packed_done_recving.add(owner.request_id)
        del self._packed_consumer_requests[owner.request_id]
        logger.info(
            "[packed-write-complete] request=%s source_bytes=%d descriptors=%d "
            "chunks=%d arrival_ms=%.3f scatter_ms=%.3f residency_ms=%.3f",
            owner.request_id,
            owner.source_plan.transfer_size_bytes,
            -1 if telemetry is None else telemetry.packed_descriptor_count,
            len(owner.packed_plan.chunks),
            owner.arrival_wait_seconds * 1_000,
            owner.scatter_wall_duration_seconds * 1_000,
            (finished_at - owner.created_at) * 1_000,
        )

    def _begin_packed_request_failure(
        self,
        owner: _ConsumerPackedRequest,
        reason: str | None,
    ) -> None:
        """Stop admitting chunks while every published command drains.

        :param owner: Request-wide atomic publication owner.
        :param reason: First exact failure detail.
        """
        if reason is None or len(reason) == 0:
            raise ValueError("packed WRITE failure reason must not be empty")
        if owner.failure_reason is None:
            owner.failure_reason = reason
            owner.tracker.fail()
            owner.pending_chunk_indices.clear()

    def _finish_packed_failed_request(
        self,
        owner: _ConsumerPackedRequest,
    ) -> None:
        """Publish failure only after every local and remote actor is drained.

        :param owner: Failed request whose active chunks may still be live.
        """
        if owner.failure_reason is None:
            return
        if len(owner.pending_chunk_indices) > 0 or len(owner.active_chunks) > 0:
            return
        self._send_final_packed_source_release(owner)
        self._record_failed_receive(
            owner.request_id,
            KVTransferFailureReason.TRANSFER,
            owner.meta,
        )
        self._packed_done_recving.add(owner.request_id)
        del self._packed_consumer_requests[owner.request_id]
        logger.warning(
            "[packed-write-failed] request=%s reason=%s",
            owner.request_id,
            owner.failure_reason,
        )

    def _service_packed_consumer_watchdogs(self) -> None:
        """Bound every decoder wait without reusing uncertain staging."""
        now = time.monotonic()
        warn_after_s = self._packed_write_config.warn_after_s
        fail_after_s = self._packed_write_config.fail_after_s
        pool = self._ensure_packed_consumer_pool()
        for owner in tuple(self._packed_consumer_requests.values()):
            if (
                owner.warning_emitted is False
                and now - owner.created_at >= warn_after_s
            ):
                owner.warning_emitted = True
                logger.warning(
                    "packed WRITE consumer request %s has remained active for %.3fs",
                    owner.request_id,
                    now - owner.created_at,
                )
            for chunk in owner.active_chunks.values():
                stage_started_at: float | None
                if chunk.scatter_launch is None:
                    stage = "WRITE arrival"
                    stage_started_at = chunk.admitted_at
                else:
                    stage = "scatter"
                    stage_started_at = chunk.scatter_enqueued_at
                if stage_started_at is None:
                    reason = f"packed WRITE {stage} has no stage-entry timestamp"
                    pool.tombstone(chunk.lease, reason)
                    owner.tracker.tombstone()
                    raise StagingSafetyError(reason)
                age_s = now - stage_started_at
                if chunk.warning_emitted is False and age_s >= warn_after_s:
                    chunk.warning_emitted = True
                    logger.warning(
                        "packed WRITE consumer request %s chunk %d %s has "
                        "remained active for %.3fs",
                        owner.request_id,
                        chunk.chunk_index,
                        stage,
                        age_s,
                    )
                if age_s < fail_after_s:
                    continue
                reason = (
                    f"packed WRITE consumer {stage} timeout for request "
                    f"{owner.request_id} chunk {chunk.chunk_index} after {age_s:.3f}s"
                )
                pool.tombstone(chunk.lease, reason)
                owner.tracker.tombstone()
                raise StagingSafetyError(reason)

    def _service_packed_consumer(self) -> None:
        """Advance every decoder-side packed WRITE without blocking."""
        if self._packed_write_config.enabled is False:
            return
        if self.kv_transfer_config.kv_role not in {"kv_consumer", "kv_both"}:
            return
        self._consume_packed_terminal_notifications()
        self._poll_packed_scatters()
        self._service_packed_consumer_watchdogs()
        self._admit_packed_chunks()

    def _packed_transfer_work_count(self) -> tuple[int, int]:
        """Return decoder work bound to the transfer/decode phase.

        Producer gather/WRITE actors are excluded. They run on an independently
        scheduled prefill engine, retain source leases, and may receive decoder
        commands while no producer transfer phase is open.

        :returns: Active decoder request count and chunks waiting for a slot.
        """
        return (
            len(self._packed_consumer_requests),
            sum(
                len(owner.pending_chunk_indices)
                for owner in self._packed_consumer_requests.values()
            ),
        )

    def _packed_registered_ownership_active(self) -> bool:
        """Return whether registered packed storage has a live generation.

        :returns: Whether shutdown must refuse memory deregistration.
        """
        producer_owned = (
            self._packed_producer_pool is not None
            and self._packed_producer_pool.has_registered_ownership
        )
        consumer_owned = (
            self._packed_consumer_pool is not None
            and self._packed_consumer_pool.has_registered_ownership
        )
        return (
            producer_owned
            or consumer_owned
            or len(self._packed_producer_pending) > 0
            or len(self._packed_producer_operations) > 0
            or len(self._packed_completion_bindings) > 0
            or len(self._buffered_packed_source_drained) > 0
            or len(self._packed_consumer_requests) > 0
        )

    def _pop_packed_done_recving(self) -> set[str]:
        """Consume decoder requests whose packed plan reached a terminal.

        :returns: Exact completed decoder request identities.
        """
        completed = set(self._packed_done_recving)
        self._packed_done_recving.clear()
        return completed

    def _packed_remote_engine_active(self, engine_id: str) -> bool:
        """Return whether packed state still owns a remote engine.

        :param engine_id: Candidate producer or decoder engine identity.
        :returns: Whether cleanup would invalidate live packed state.
        """
        if any(
            owner.meta.remote is not None and owner.meta.remote.engine_id == engine_id
            for owner in self._packed_consumer_requests.values()
        ):
            return True
        if any(
            pending.request.command.chunk.consumer_engine_id == engine_id
            for pending in self._packed_producer_pending
        ):
            return True
        if any(
            binding.request_key.consumer_engine_id == engine_id
            for binding in self._packed_completion_bindings.values()
        ):
            return True
        return any(
            operation.request.command.chunk.consumer_engine_id == engine_id
            for operation in self._packed_producer_operations.values()
        )

    def _packed_remote_engine_cleanup(self, engine_id: str) -> None:
        """Retire quiescent endpoint and terminal state for one engine.

        :param engine_id: Remote engine proven free of live packed work.
        """
        endpoints = tuple(
            endpoint
            for endpoint in self._packed_consumer_agents
            if endpoint.consumer_engine_id == engine_id
        )
        if any(
            self._packed_consumer_agent_owned(self._packed_consumer_agents[endpoint])
            for endpoint in endpoints
        ):
            raise StagingSafetyError(
                "packed WRITE engine cleanup would release active endpoint ownership"
            )
        for endpoint in endpoints:
            self._remove_packed_consumer_agent(endpoint)
        self._packed_consumer_terminals = {
            identity: terminal
            for identity, terminal in self._packed_consumer_terminals.items()
            if identity.producer_engine_id != engine_id
        }

    def _packed_shutdown_cleanup(self) -> None:
        """Release dynamically imported decoder agents after packed quiescence."""
        for endpoint in tuple(self._packed_consumer_agents):
            self._remove_packed_consumer_agent(endpoint)
        self._packed_consumer_agents.clear()
        self._packed_consumer_agent_last_active.clear()

    def _read_blocks(
        self,
        *,
        read_spec: ReadSpec,
        dst_engine_id: str,
        request_id: str,
        local_xfer_side_handle: int,
        remote_xfer_side_handle: int,
        notification_id: bytes,
    ):
        """
        Post a READ point-to-point xfer request from a single local worker to
        a single remote worker.
        """
        assert self.transfer_topo is not None
        remote_rank = read_spec.remote_rank
        local_block_ids = read_spec.local_block_ids
        remote_block_ids = read_spec.remote_block_ids

        remote_info = self.transfer_topo.get_engine_info(dst_engine_id)
        block_size_ratio = self.transfer_topo.block_size_ratio(
            remote_info.remote_block_size
        )
        if block_size_ratio > 1:
            # TODO (NickLucche) assume HMA is off. Change to handle multiple KV groups.
            assert not self._is_hma_required
            local_block_ids0 = local_block_ids[0] if local_block_ids else []
            remote_block_ids0 = remote_block_ids[0]
            local_block_ids_mapped = self.get_mapped_blocks(
                np.asarray(local_block_ids0), block_size_ratio
            ).tolist()
            if len(local_block_ids_mapped) > len(remote_block_ids0):
                # NOTE:
                # get_mapped_blocks will always expand block_ids for n times.
                # ex:
                # prefill block_ids with block_size as 4:
                # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                # Local decode block_ids with block_size as 16: [1, 2, 3]
                # expanded decode block_ids with get_mapped_blocks from [1, 2, 3] to
                # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
                # Then we clip local to align with prefill
                # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] to
                # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                local_block_ids_mapped = local_block_ids_mapped[
                    : len(remote_block_ids0)
                ]
            local_block_ids = [local_block_ids_mapped] if local_block_ids_mapped else []
            remote_block_ids = [remote_block_ids0]
        # NOTE(rob): having the staging blocks be on the READER side is
        # not going to work well (since we will have to call rearrange tensors).
        # after we detect the txn is complete (which means we cannot make the
        # read trxn async easily). If we want to make "READ" happen cleanly,
        # then we will need to have the staging blocks on the remote side.

        # NOTE(rob): according to nvidia the staging blocks are used to
        # saturate IB with heterogeneous TP sizes.

        # Full prefix cache hit: do not need to read remote blocks,
        # just notify P worker that we have the blocks we need.
        if len(local_block_ids) == 0:
            self._send_zero_byte_completion(
                request_id,
                dst_engine_id,
                remote_rank,
                notification_id,
            )
            return

        assert (
            len(remote_block_ids)
            == len(local_block_ids)
            == len(self.kv_cache_config.kv_cache_groups)
        )
        remote_physical_per_logical = remote_info.remote_physical_blocks_per_logical
        local_block_ids, remote_block_ids = self._apply_prefix_caching(
            local_block_ids, remote_block_ids, remote_physical_per_logical
        )
        if sum(len(group) for group in local_block_ids) == 0:
            self._send_zero_byte_completion(
                request_id,
                dst_engine_id,
                remote_rank,
                notification_id,
            )
            return

        # NOTE (nicolo) With homogeneous TP, each TP worker loads KV from
        # corresponding rank. With heterogeneous TP, fixing D>P, the D tp
        # workers will issue xfers to parts of the P worker remote kv caches.

        # Get descs ids.
        remote_block_descs_ids = self._compute_desc_ids(
            block_ids=remote_block_ids,
            dst_num_blocks=self.dst_num_blocks[dst_engine_id],
            block_size_ratio=None,
            physical_blocks_per_logical=remote_info.remote_physical_blocks_per_logical,
        )
        local_block_descs_ids = self._compute_desc_ids(
            block_ids=local_block_ids,
            dst_num_blocks=self.dst_num_blocks[self.engine_id],
            block_size_ratio=block_size_ratio,
            physical_blocks_per_logical=self._physical_blocks_per_logical_kv_block,
        )

        assert len(local_block_descs_ids) == len(remote_block_descs_ids)

        self._assert_transfer_post_allowed(coalesced=False)

        # Prepare transfer with Nixl.
        handle = None
        try:
            handle = self.nixl_wrapper.make_prepped_xfer(
                "READ",
                local_xfer_side_handle,
                local_block_descs_ids,
                remote_xfer_side_handle,
                remote_block_descs_ids,
                notif_msg=notification_id,
            )

            # Begin async xfer.
            self.nixl_wrapper.transfer(handle)

            # Use handle to check completion in future step().
            self._recving_transfers[request_id].append(handle)
        except Exception as e:
            # mark all (logical) blocks for this request as invalid
            self._log_failure(
                failure_type="transfer_setup_failed",
                req_id=request_id,
                msg="Marking blocks as invalid",
                error=e,
                dst_engine_id=dst_engine_id,
                remote_rank=remote_rank,
            )
            self._handle_failed_transfer(request_id, handle)

    def _send_zero_byte_completion(
        self,
        request_id: str,
        producer_engine_id: str,
        producer_rank: int,
        notification_id: bytes,
    ) -> None:
        """Report a proven no-read path and complete it on the decoder.

        :param request_id: Decoder-side request identifier.
        :param producer_engine_id: Remote producer engine.
        :param producer_rank: Remote producer tensor-parallel rank.
        :param notification_id: Typed decoder-rank completion proof.
        """
        agent_name = self._remote_agents[producer_engine_id][producer_rank]
        try:
            self.nixl_wrapper.send_notif(agent_name, notif_msg=notification_id)
        except Exception as error:
            self._log_failure(
                failure_type="notification_failed",
                msg="P worker blocks remain pinned without this completion proof",
                req_id=request_id,
                error=error,
                dst_engine_id=producer_engine_id,
                remote_rank=producer_rank,
                remote_agent_name=agent_name,
            )
            self.xfer_stats.record_failed_notification()
        self._recving_transfers.setdefault(request_id, [])

    def _get_new_notifs(self) -> set[str]:
        """Apply heartbeats and exact decoder terminal proofs.

        :returns: Producer requests whose complete immutable obligation set
            has been satisfied.
        """
        notified_req_ids: set[str] = set()
        read_proofs: list[PullReadComplete] = []
        packed_drains: list[_AuthenticatedPackedSourceDrained] = []
        cancellation_proofs: list[PullOfferCancelled] = (
            self._pending_offer_cancellations
        )
        self._pending_offer_cancellations = []

        for notifying_agent, notifs in self.nixl_wrapper.get_new_notifs().items():
            for notif in notifs:
                if notif.startswith(b"HB:"):
                    try:
                        heartbeat_payload = notif[3:].decode("utf-8")
                    except UnicodeDecodeError:
                        logger.error("Ignoring malformed NIXL heartbeat payload")
                        continue
                    self._handle_heartbeat(heartbeat_payload)
                    continue
                if notif.startswith(PULL_READ_COMPLETE_PREFIX):
                    try:
                        read_proofs.append(
                            msgspec.msgpack.decode(
                                notif[len(PULL_READ_COMPLETE_PREFIX) :],
                                type=PullReadComplete,
                            )
                        )
                    except (msgspec.DecodeError, msgspec.ValidationError):
                        logger.error(
                            "Ignoring malformed NIXL pull completion proof\n%s",
                            traceback.format_exc(),
                        )
                    continue
                if notif.startswith(PACKED_WRITE_REQUEST_PREFIX):
                    try:
                        self._accept_packed_request_notification(
                            decode_packed_write_request_notification(notif),
                            notifying_agent,
                        )
                    except (
                        ValueError,
                        msgspec.DecodeError,
                        msgspec.ValidationError,
                    ) as error:
                        raise StagingSafetyError(
                            "invalid packed write request notification\n"
                            + traceback.format_exc()
                        ) from error
                    continue
                if notif.startswith(PACKED_WRITE_ARRIVED_PREFIX):
                    try:
                        self._accept_packed_arrived_notification(
                            decode_packed_write_arrived_notification(notif),
                            notifying_agent,
                        )
                    except (
                        ValueError,
                        msgspec.DecodeError,
                        msgspec.ValidationError,
                    ) as error:
                        raise StagingSafetyError(
                            "invalid packed WRITE arrival notification\n"
                            + traceback.format_exc()
                        ) from error
                    continue
                if notif.startswith(PACKED_WRITE_FAILED_BEFORE_WRITE_PREFIX):
                    try:
                        self._accept_packed_failed_notification(
                            decode_packed_write_failed_before_write_notification(notif),
                            notifying_agent,
                        )
                    except (
                        ValueError,
                        msgspec.DecodeError,
                        msgspec.ValidationError,
                    ) as error:
                        raise StagingSafetyError(
                            "invalid packed WRITE failure notification\n"
                            + traceback.format_exc()
                        ) from error
                    continue
                if notif.startswith(PACKED_WRITE_SOURCE_DRAINED_PREFIX):
                    try:
                        packed_drains.append(
                            _AuthenticatedPackedSourceDrained(
                                drained=decode_packed_write_source_drained_notification(
                                    notif
                                ),
                                consumer_agent=notifying_agent,
                            )
                        )
                    except (
                        ValueError,
                        msgspec.DecodeError,
                        msgspec.ValidationError,
                    ) as error:
                        raise StagingSafetyError(
                            "invalid packed WRITE source-drain proof\n"
                            + traceback.format_exc()
                        ) from error
                    continue
                logger.error("Ignoring unknown NIXL pull notification")

        for req_id in tuple(self._buffered_pull_completions):
            if req_id not in self._pull_completion_states:
                continue
            read_proofs.extend(self._buffered_pull_completions.pop(req_id))
        for req_id in tuple(self._buffered_offer_cancellations):
            if req_id not in self._pull_completion_states:
                continue
            cancellation_proofs.extend(self._buffered_offer_cancellations.pop(req_id))

        read_offers = {
            (proof.producer_request_id, proof.offer_generation) for proof in read_proofs
        }
        cancellation_offers = {
            (proof.producer_request_id, proof.offer_generation)
            for proof in cancellation_proofs
        }
        for req_id, offer_generation in read_offers.intersection(cancellation_offers):
            state = self._pull_completion_states.get(req_id)
            if state is not None and state.offer_generation == offer_generation:
                self._mark_pull_contract_conflicted(req_id, state)

        for read_proof in read_proofs:
            if self._record_pull_completion(read_proof, allow_buffer=True):
                notified_req_ids.add(read_proof.producer_request_id)
        for authenticated in packed_drains:
            result = self._record_packed_source_drained(
                authenticated,
                allow_buffer=True,
            )
            if result == "completed":
                notified_req_ids.add(
                    authenticated.drained.anchor.chunk.producer_request_id
                )
        for cancellation_proof in cancellation_proofs:
            if self._record_offer_cancellation(cancellation_proof, allow_buffer=True):
                notified_req_ids.add(cancellation_proof.producer_request_id)
        self._service_packed_producer()
        for req_id, buffered in tuple(self._buffered_packed_source_drained.items()):
            for authenticated in tuple(buffered):
                result = self._record_packed_source_drained(
                    authenticated,
                    allow_buffer=False,
                )
                if result == "waiting":
                    continue
                buffered.remove(authenticated)
                if result == "completed":
                    notified_req_ids.add(req_id)
            if len(buffered) == 0:
                self._buffered_packed_source_drained.pop(req_id, None)
        notified_req_ids.update(self._complete_quiescent_pull_contracts())
        self._service_packed_consumer()
        return notified_req_ids

    @staticmethod
    def _packed_drain_binding_key(
        authenticated: _AuthenticatedPackedSourceDrained,
    ) -> tuple[str, int, str, int]:
        """Return the generation-scoped completion key for a drain proof.

        :param authenticated: Source-drain proof and native sender.
        :returns: Producer offer, decoder request, and decoder-rank identity.
        """
        identity = authenticated.drained.anchor.chunk
        return (
            identity.producer_request_id,
            identity.offer_generation,
            identity.consumer_request_id,
            identity.consumer_rank,
        )

    def _buffer_packed_source_drained(
        self,
        authenticated: _AuthenticatedPackedSourceDrained,
    ) -> None:
        """Retain one bounded exact drain proof until producer actors quiesce.

        :param authenticated: Source-drain proof and native sender.
        """
        req_id = authenticated.drained.anchor.chunk.producer_request_id
        pending = self._buffered_packed_source_drained.setdefault(req_id, set())
        if (
            authenticated not in pending
            and len(pending) >= _MAX_BUFFERED_PULL_COMPLETIONS_PER_REQUEST
        ):
            raise StagingSafetyError(
                "packed WRITE source-drain buffer exhausted for one request"
            )
        pending.add(authenticated)

    def _record_packed_source_drained(
        self,
        authenticated: _AuthenticatedPackedSourceDrained,
        *,
        allow_buffer: bool,
    ) -> PackedDrainRecordResult:
        """Authenticate and apply one packed decoder completion proof.

        :param authenticated: Source-drain proof and native sender.
        :param allow_buffer: Whether a pre-contract proof may be retained.
        :returns: Whether the proof is waiting, recorded, or completed the lease.
        """
        self._require_packed_notifying_agent(authenticated.consumer_agent)
        drained = authenticated.drained
        anchor_identity = drained.anchor.chunk
        req_id = anchor_identity.producer_request_id
        binding_key = self._packed_drain_binding_key(authenticated)
        completed = self._completed_pull_contracts.get(
            (req_id, anchor_identity.offer_generation)
        )
        if completed is not None:
            if authenticated in completed.packed_completions:
                return "recorded"
            same_identity = any(
                self._packed_drain_binding_key(prior) == binding_key
                for prior in completed.packed_completions
            )
            if same_identity:
                raise StagingSafetyError(
                    "packed WRITE source-drain replay changed exact fields"
                )
            raise StagingSafetyError(
                "packed WRITE source-drain conflicts with a released source contract"
            )
        state = self._pull_completion_states.get(req_id)
        binding = self._packed_completion_bindings.get(binding_key)
        if binding is None:
            raise StagingSafetyError(
                "packed WRITE source-drain proof has no accepted command binding"
            )
        request_key = PackedWriteRequestKey.from_chunk(
            PackedWriteChunkKey.from_identity(anchor_identity)
        )
        if request_key != binding.request_key:
            raise StagingSafetyError(
                "packed WRITE source-drain proof changed its request identity"
            )
        if binding.consumer_agent != authenticated.consumer_agent:
            raise StagingSafetyError(
                "packed WRITE source-drain sender differs from its command binding"
            )
        if state is None:
            producer_owned = (
                req_id in self._reqs_to_process or req_id in self._source_rosters
            )
            if allow_buffer and producer_owned:
                self._buffer_packed_source_drained(authenticated)
                return "waiting"
            pool = self._ensure_packed_producer_pool()
            if (
                pool.validate_source_drained(
                    drained,
                    authenticated.consumer_agent,
                )
                is False
            ):
                if self._packed_producer_request_active(req_id):
                    self._buffer_packed_source_drained(authenticated)
                    return "waiting"
                raise StagingSafetyError(
                    "ownerless packed WRITE drain lacks its terminal prefix"
                )
            pool.retire_request_key(binding.request_key)
            del self._packed_completion_bindings[binding_key]
            buffered = self._buffered_packed_source_drained.get(req_id)
            if buffered is not None:
                buffered.discard(authenticated)
                if len(buffered) == 0:
                    del self._buffered_packed_source_drained[req_id]
            return "recorded"
        roster = self._source_rosters.get(req_id)
        if (
            roster is None
            or roster.offer_generation != anchor_identity.offer_generation
            or state.offer_generation != anchor_identity.offer_generation
        ):
            raise StagingSafetyError(
                "packed WRITE source-drain names a stale source-offer generation"
            )
        completion_identity = self._packed_binding_completion_identity(binding, state)
        if len(state.offer_cancellations) > 0:
            self._mark_pull_contract_conflicted(req_id, state)
            raise StagingSafetyError(
                "packed WRITE source-drain follows a whole-offer cancellation"
            )
        if self._packed_producer_request_active(req_id):
            self._buffer_packed_source_drained(authenticated)
            return "waiting"
        pool = self._ensure_packed_producer_pool()
        if pool.validate_source_drained(drained, authenticated.consumer_agent) is False:
            raise StagingSafetyError(
                "packed WRITE source-drain proof lacks its terminal prefix"
            )
        prior = state.packed_completions.get(completion_identity)
        if prior is not None:
            if prior != authenticated:
                raise StagingSafetyError(
                    "packed WRITE source-drain replay changed exact fields"
                )
            return "recorded"
        if completion_identity in state.read_completions:
            raise StagingSafetyError(
                "packed WRITE completion identity was satisfied by another protocol"
            )
        state.packed_completions[completion_identity] = authenticated
        state.read_completions.add(completion_identity)
        if (
            state.has_mixed_terminal_proofs
            or state.read_completions != state.expected_read_completions
        ):
            return "recorded"
        if self._try_complete_pull_contract(req_id, state, "read_complete"):
            return "completed"
        return "recorded"

    def _record_pull_completion(
        self,
        proof: PullReadComplete,
        *,
        allow_buffer: bool,
    ) -> bool:
        """Apply one proof or retain it until its producer contract arrives.

        :param proof: Decoder-authored completion proof.
        :param allow_buffer: Whether an owned pre-contract request may retain
            this proof for a later worker step.
        :returns: Whether the proof completed the producer's obligation set.
        """
        req_id = proof.producer_request_id
        state = self._pull_completion_states.get(req_id)
        if state is None:
            completed_contract = self._completed_pull_contracts.get(
                (req_id, proof.offer_generation)
            )
            if completed_contract is not None:
                try:
                    identity = self._completion_identity(
                        proof,
                        completed_contract.offer_generation,
                        completed_contract.expected_consumers,
                        completed_contract.consumer_tp_size,
                    )
                except ValueError:
                    logger.error(
                        "Conflicting duplicate completion proof for released "
                        "producer request %s\n%s",
                        req_id,
                        traceback.format_exc(),
                    )
                else:
                    if any(
                        prior.drained.anchor.chunk.offer_generation
                        == proof.offer_generation
                        and prior.drained.anchor.chunk.consumer_rank == identity[1]
                        and _parallel_consumer_index(
                            prior.drained.anchor.chunk.consumer_request_id,
                            completed_contract.expected_consumers,
                        )
                        == identity[0]
                        for prior in completed_contract.packed_completions
                    ):
                        raise StagingSafetyError(
                            "legacy read completion replay targets a packed obligation"
                        )
                    if completed_contract.terminal_mode == "offer_cancelled":
                        logger.error(
                            "Read proof arrived after cancellation released producer "
                            "request %s; decoder protocol invariants were violated",
                            req_id,
                        )
                return False
            if allow_buffer and req_id in self._reqs_to_process:
                pending = self._buffered_pull_completions.setdefault(req_id, set())
                if (
                    proof not in pending
                    and len(pending) >= _MAX_BUFFERED_PULL_COMPLETIONS_PER_REQUEST
                ):
                    logger.error(
                        "Ignoring excess pre-contract completion proof for producer "
                        "request %s; source pages remain pinned",
                        req_id,
                    )
                    return False
                pending.add(proof)
                return False
            logger.error(
                "A decode worker reported a read for unowned request %s; its "
                "source pages are no longer guaranteed stable.",
                req_id,
            )
            return False

        try:
            identity = self._completion_identity(
                proof,
                state.offer_generation,
                state.expected_consumers,
                state.consumer_tp_size,
            )
        except ValueError:
            logger.error(
                "Ignoring conflicting completion proof for producer request %s\n%s",
                req_id,
                traceback.format_exc(),
            )
            return False
        if identity not in state.expected_read_completions:
            logger.error(
                "Ignoring completion proof %s outside producer request %s obligations",
                identity,
                req_id,
            )
            return False
        if any(
            binding_req_id == req_id
            and offer_generation == proof.offer_generation
            and self._packed_binding_completion_identity(binding, state) == identity
            for (
                binding_req_id,
                offer_generation,
                _consumer_request_id,
                _consumer_rank,
            ), binding in self._packed_completion_bindings.items()
        ):
            raise StagingSafetyError(
                "legacy read completion cannot satisfy a packed WRITE binding"
            )
        if len(state.offer_cancellations) > 0:
            self._mark_pull_contract_conflicted(req_id, state)
        state.read_completions.add(identity)
        if (
            state.has_mixed_terminal_proofs
            or state.read_completions != state.expected_read_completions
        ):
            return False
        return self._try_complete_pull_contract(req_id, state, "read_complete")

    def _record_offer_cancellation(
        self,
        proof: PullOfferCancelled,
        *,
        allow_buffer: bool,
    ) -> bool:
        """Apply one whole-offer proof or retain it for its contract.

        :param proof: Decoder-rank proof that no consumer was admitted.
        :param allow_buffer: Whether an owned pre-contract request may retain
            this proof for a later worker step.
        :returns: Whether the exact cancellation quorum released the offer.
        """
        req_id = proof.producer_request_id
        if any(
            binding_req_id == req_id
            and offer_generation == proof.offer_generation
            and consumer_rank == proof.consumer_rank
            for (
                binding_req_id,
                offer_generation,
                _consumer_request_id,
                consumer_rank,
            ) in self._packed_completion_bindings
        ):
            raise StagingSafetyError(
                "offer cancellation cannot satisfy a packed WRITE binding"
            )
        state = self._pull_completion_states.get(req_id)
        if state is None:
            completed_contract = self._completed_pull_contracts.get(
                (req_id, proof.offer_generation)
            )
            if completed_contract is not None:
                try:
                    consumer_rank = self._offer_cancellation_rank(
                        proof,
                        completed_contract.offer_generation,
                        completed_contract.expected_consumers,
                        completed_contract.consumer_tp_size,
                    )
                    expected_ranks = frozenset(
                        _consumer_ranks_for_producer(
                            self.tp_rank,
                            self.world_size,
                            completed_contract.consumer_tp_size,
                        )
                    )
                    if consumer_rank not in expected_ranks:
                        raise ValueError(
                            "decoder rank is outside this producer's obligations"
                        )
                except ValueError:
                    logger.error(
                        "Conflicting duplicate cancellation proof for released "
                        "producer request %s\n%s",
                        req_id,
                        traceback.format_exc(),
                    )
                else:
                    if completed_contract.terminal_mode == "read_complete":
                        logger.error(
                            "Cancellation proof arrived after reads released producer "
                            "request %s; decoder protocol invariants were violated",
                            req_id,
                        )
                return False
            if allow_buffer and req_id in self._reqs_to_process:
                pending = self._buffered_offer_cancellations.setdefault(req_id, set())
                if (
                    proof not in pending
                    and len(pending) >= _MAX_BUFFERED_PULL_COMPLETIONS_PER_REQUEST
                ):
                    logger.error(
                        "Ignoring excess pre-contract cancellation proof for producer "
                        "request %s; source pages remain pinned",
                        req_id,
                    )
                    return False
                pending.add(proof)
                return False
            logger.error(
                "A decode worker cancelled an unowned producer request %s; "
                "no source ownership was released",
                req_id,
            )
            return False

        try:
            consumer_rank = self._offer_cancellation_rank(
                proof,
                state.offer_generation,
                state.expected_consumers,
                state.consumer_tp_size,
            )
        except ValueError:
            logger.error(
                "Ignoring conflicting cancellation proof for producer request %s\n%s",
                req_id,
                traceback.format_exc(),
            )
            return False
        if consumer_rank not in state.expected_offer_cancellations:
            logger.error(
                "Ignoring cancellation proof from decoder rank %d outside producer "
                "request %s obligations",
                consumer_rank,
                req_id,
            )
            return False
        if len(state.read_completions) > 0:
            self._mark_pull_contract_conflicted(req_id, state)
        state.offer_cancellations.add(consumer_rank)
        if (
            state.has_mixed_terminal_proofs
            or state.offer_cancellations != state.expected_offer_cancellations
        ):
            return False
        return self._try_complete_pull_contract(req_id, state, "offer_cancelled")

    def _offer_cancellation_rank(
        self,
        proof: PullOfferCancelled,
        offer_generation: int | None,
        expected_consumers: int,
        consumer_tp_size: int,
    ) -> int:
        """Validate a cancellation proof against an immutable contract.

        :param proof: Decoder-authored whole-offer cancellation proof.
        :param offer_generation: Producer-owned allocation generation.
        :param expected_consumers: Producer-owned logical consumer count.
        :param consumer_tp_size: Producer-owned decoder TP size.
        :returns: Decoder rank making the cancellation assertion.
        :raises ValueError: If any consumer claim conflicts with the contract.
        """
        if proof.offer_generation != offer_generation:
            raise ValueError("decoder source-offer generation changed")
        if proof.expected_consumers != expected_consumers:
            raise ValueError("decoder logical consumer count changed")
        if proof.consumer_tp_size != consumer_tp_size:
            raise ValueError("decoder tensor-parallel size changed")
        if proof.consumer_rank < 0 or proof.consumer_rank >= consumer_tp_size:
            raise ValueError("decoder tensor-parallel rank is outside its world")
        return proof.consumer_rank

    def _mark_pull_contract_conflicted(
        self,
        req_id: str,
        state: PullCompletionState,
    ) -> None:
        """Permanently pin a contract that received both terminal modes.

        :param req_id: Producer request identifier.
        :param state: Active producer completion contract.
        """
        if state.has_mixed_terminal_proofs is False:
            logger.error(
                "Producer request %s received both read and whole-offer "
                "cancellation proofs; source pages remain pinned",
                req_id,
            )
        state.has_mixed_terminal_proofs = True

    def _packed_producer_request_active(self, producer_request_id: str) -> bool:
        """Return whether packed producer state still touches source pages.

        :param producer_request_id: Producer request whose retained source
            allocation is being considered for release.
        :returns: Whether queued work, a live gather, or an owned producer slot
            still references the request.
        """
        pending = any(
            pending.request.command.chunk.producer_request_id == producer_request_id
            for pending in self._packed_producer_pending
        )
        if pending:
            return True
        active = any(
            operation.request.command.chunk.producer_request_id == producer_request_id
            for operation in self._packed_producer_operations.values()
        )
        return active

    def _try_complete_pull_contract(
        self,
        req_id: str,
        state: PullCompletionState,
        terminal_mode: PullContractTerminalMode,
    ) -> bool:
        """Release a satisfied contract only after producer packing quiesces.

        :param req_id: Producer request identifier.
        :param state: Satisfied immutable producer contract.
        :param terminal_mode: Proof mode authorizing eventual release.
        :returns: Whether this call released the producer allocation.
        """
        if self._packed_producer_request_active(req_id):
            return False
        return self._complete_pull_contract(req_id, state, terminal_mode)

    def _complete_quiescent_pull_contracts(self) -> set[str]:
        """Finish proof-complete contracts whose producer gathers retired.

        :returns: Producer request identifiers released by this service pass.
        """
        completed: set[str] = set()
        for req_id, state in tuple(self._pull_completion_states.items()):
            if state.has_mixed_terminal_proofs:
                continue
            terminal_mode: PullContractTerminalMode | None = None
            if state.read_completions == state.expected_read_completions:
                terminal_mode = "read_complete"
            elif state.offer_cancellations == state.expected_offer_cancellations:
                terminal_mode = "offer_cancelled"
            if terminal_mode is None:
                continue
            if self._try_complete_pull_contract(req_id, state, terminal_mode):
                completed.add(req_id)
        return completed

    def _complete_pull_contract(
        self,
        req_id: str,
        state: PullCompletionState,
        terminal_mode: PullContractTerminalMode,
    ) -> bool:
        """Release a producer lease after one exact terminal proof set.

        :param req_id: Producer request identifier.
        :param state: Satisfied immutable producer contract.
        :param terminal_mode: Proof mode authorizing release.
        :returns: Always true after the release is committed.
        :raises RuntimeError: If worker ownership state is inconsistent.
        :raises StagingSafetyError: If packed producer work still owns source
            memory for the request.
        """
        if req_id not in self._reqs_to_process:
            raise RuntimeError(
                f"producer request {req_id} completed without an ownership pin"
            )
        if self._packed_producer_request_active(req_id):
            raise StagingSafetyError(
                f"producer request {req_id} cannot release source pages while "
                "packed producer work still owns them"
            )
        completed_key = (req_id, state.offer_generation)
        if (
            completed_key not in self._completed_pull_contracts
            and len(self._completed_pull_contracts) >= _MAX_COMPLETED_PULL_CONTRACTS
        ):
            raise StagingSafetyError(
                "completed producer contracts exceeded their fail-stop bound"
            )
        if terminal_mode == "read_complete":
            self._localization_capture_source_post(req_id)
        completed_bindings = frozenset(
            binding
            for (
                binding_req_id,
                offer_generation,
                _consumer_request_id,
                _consumer_rank,
            ), binding in self._packed_completion_bindings.items()
            if binding_req_id == req_id and offer_generation == state.offer_generation
        )
        if self._packed_producer_pool is not None:
            if state.offer_generation is None:
                raise StagingSafetyError(
                    "packed WRITE offer cannot retire without a source generation"
                )
            self._packed_producer_pool.retire_offer(
                req_id,
                state.offer_generation,
            )
        self._packed_completion_bindings = {
            key: binding
            for key, binding in self._packed_completion_bindings.items()
            if key[0] != req_id or key[1] != state.offer_generation
        }
        self._buffered_packed_source_drained.pop(req_id, None)
        self._source_rosters.pop(req_id, None)
        self._packed_source_ready_events.pop(req_id, None)

        del self._pull_completion_states[req_id]
        self._completed_pull_contracts[completed_key] = CompletedPullContract(
            expected_consumers=state.expected_consumers,
            consumer_tp_size=state.consumer_tp_size,
            offer_generation=state.offer_generation,
            terminal_mode=terminal_mode,
            packed_completions=frozenset(state.packed_completions.values()),
            packed_bindings=completed_bindings,
        )
        self._reqs_to_process.remove(req_id)
        self._reqs_to_send.pop(req_id, None)
        return True

    def _completion_identity(
        self,
        proof: PullReadComplete,
        offer_generation: int | None,
        expected_consumers: int,
        consumer_tp_size: int,
    ) -> tuple[int, int]:
        """Validate a wire proof against immutable producer obligations.

        :param proof: Decoder-authored completion proof.
        :param offer_generation: Producer-owned allocation generation.
        :param expected_consumers: Producer-owned logical consumer count.
        :param consumer_tp_size: Producer-owned decoder TP size.
        :returns: Stable child-index and decoder-rank identity.
        :raises ValueError: If any consumer claim conflicts with the contract.
        """
        if proof.offer_generation != offer_generation:
            raise ValueError("decoder source-offer generation changed")
        if proof.expected_consumers != expected_consumers:
            raise ValueError("decoder logical consumer count changed")
        if proof.consumer_tp_size != consumer_tp_size:
            raise ValueError("decoder tensor-parallel size changed")
        if proof.consumer_rank < 0 or proof.consumer_rank >= consumer_tp_size:
            raise ValueError("decoder tensor-parallel rank is outside its world")
        derived_index = _parallel_consumer_index(
            proof.consumer_request_id,
            expected_consumers,
        )
        if proof.consumer_index != derived_index:
            raise ValueError("decoder child index conflicts with its request lineage")
        return proof.consumer_index, proof.consumer_rank

    def _producer_completion_count(self, req_id: str) -> int:
        """Return exact pull obligations completed for an overdue lease.

        :param req_id: Producer request identifier.
        :returns: Number of unique child/rank proofs observed.
        """
        state = self._pull_completion_states.get(req_id)
        if state is None:
            return 0
        return len(state.read_completions) + len(state.offer_cancellations)
