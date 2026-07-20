# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generation-scoped ownership for bounded producer-packed WRITE staging."""

from dataclasses import dataclass, field
from enum import Enum, auto

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    PackedWriteArrived,
    PackedWriteChunkIdentity,
    PackedWriteConsumerEndpoint,
    PackedWriteConsumerPoolGeometry,
    PackedWriteFailedBeforeWrite,
    PackedWriteProducerPoolGeometry,
    PackedWriteRequest,
    PackedWriteSlotBinding,
    PackedWriteSourceDrained,
    validate_packed_write_arrived,
    validate_packed_write_failed_before_write,
    validate_packed_write_request_replay,
)
from vllm.distributed.kv_transfer.staging_ownership import StagingSafetyError

_MAX_TERMINAL_HISTORY = 65_536


@dataclass(frozen=True, slots=True)
class PackedWriteChunkKey:
    """Rank-independent identity for one bounded logical chunk."""

    producer_engine_id: str
    producer_request_id: str
    offer_generation: int
    producer_tp_size: int
    consumer_engine_id: str
    consumer_request_id: str
    consumer_rank: int
    consumer_tp_size: int
    source_plan_digest: str
    packed_plan_digest: str
    chunk_plan_digest: str
    chunk_ordinal: int
    chunk_count: int
    valid_token_extent: int
    exact_bytes: int

    @classmethod
    def from_identity(cls, identity: PackedWriteChunkIdentity) -> "PackedWriteChunkKey":
        """Drop only the producer-rank coordinate from a wire identity.

        :param identity: Rankful immutable chunk identity.
        :returns: Rank-independent chunk key.
        """
        if type(identity) is not PackedWriteChunkIdentity:
            raise ValueError("packed WRITE chunk identity must be typed")
        return cls(
            producer_engine_id=identity.producer_engine_id,
            producer_request_id=identity.producer_request_id,
            offer_generation=identity.offer_generation,
            producer_tp_size=identity.producer_tp_size,
            consumer_engine_id=identity.consumer_engine_id,
            consumer_request_id=identity.consumer_request_id,
            consumer_rank=identity.consumer_rank,
            consumer_tp_size=identity.consumer_tp_size,
            source_plan_digest=identity.source_plan_digest,
            packed_plan_digest=identity.packed_plan_digest,
            chunk_plan_digest=identity.chunk_plan_digest,
            chunk_ordinal=identity.chunk_ordinal,
            chunk_count=identity.chunk_count,
            valid_token_extent=identity.valid_token_extent,
            exact_bytes=identity.exact_bytes,
        )


@dataclass(frozen=True, slots=True)
class PackedWriteRequestKey:
    """Chunk-independent identity for one complete packed request."""

    producer_engine_id: str
    producer_request_id: str
    offer_generation: int
    producer_tp_size: int
    consumer_engine_id: str
    consumer_request_id: str
    consumer_rank: int
    consumer_tp_size: int
    source_plan_digest: str
    packed_plan_digest: str
    chunk_count: int
    valid_token_extent: int

    @classmethod
    def from_chunk(cls, chunk: PackedWriteChunkKey) -> "PackedWriteRequestKey":
        """Derive the invariant request coordinates from one chunk.

        :param chunk: Rank-independent chunk key.
        :returns: Complete-request identity.
        """
        if type(chunk) is not PackedWriteChunkKey:
            raise ValueError("packed WRITE chunk key must be typed")
        return cls(
            producer_engine_id=chunk.producer_engine_id,
            producer_request_id=chunk.producer_request_id,
            offer_generation=chunk.offer_generation,
            producer_tp_size=chunk.producer_tp_size,
            consumer_engine_id=chunk.consumer_engine_id,
            consumer_request_id=chunk.consumer_request_id,
            consumer_rank=chunk.consumer_rank,
            consumer_tp_size=chunk.consumer_tp_size,
            source_plan_digest=chunk.source_plan_digest,
            packed_plan_digest=chunk.packed_plan_digest,
            chunk_count=chunk.chunk_count,
            valid_token_extent=chunk.valid_token_extent,
        )


class ProducerSlotState(Enum):
    """Producer gather-slot lifecycle."""

    FREE = auto()
    PACKING = auto()
    PREPARING_WRITE = auto()
    POSTING_WRITE = auto()
    WRITING = auto()
    WRITE_DONE = auto()
    WRITE_RELEASED = auto()
    TOMBSTONED = auto()


@dataclass(frozen=True, slots=True)
class ProducerSlotLease:
    """Generation-scoped ownership of one local producer gather slot."""

    slot_index: int
    slot_generation: int
    source_address: int
    payload_bytes: int
    chunk: PackedWriteChunkIdentity


@dataclass(frozen=True, slots=True)
class ProducerAdmission:
    """Result of admitting or replaying one producer command."""

    lease: ProducerSlotLease | None
    terminal: PackedWriteArrived | PackedWriteFailedBeforeWrite | None
    newly_admitted: bool


@dataclass(slots=True)
class _ProducerSlot:
    """Mutable state for one producer gather slot."""

    slot_index: int
    generation: int = 0
    state: ProducerSlotState = ProducerSlotState.FREE
    request: PackedWriteRequest | None = None
    consumer_agent: str | None = None
    lease: ProducerSlotLease | None = None


@dataclass(frozen=True, slots=True)
class _ProducerTerminal:
    """Exact terminal retained for idempotent request replay."""

    request: PackedWriteRequest
    consumer_agent: str
    terminal: PackedWriteArrived | PackedWriteFailedBeforeWrite


@dataclass(slots=True)
class _ProducerRequestTerminalHistory:
    """Exact terminal prefix retained for one decoder request and producer rank."""

    consumer_endpoint: PackedWriteConsumerEndpoint
    consumer_agent: str
    terminals_by_ordinal: dict[int, _ProducerTerminal] = field(default_factory=dict)


class PackedWriteProducerPool:
    """Own producer gather slots from admission through local WRITE completion."""

    geometry: PackedWriteProducerPoolGeometry
    producer_rank: int
    _slots: list[_ProducerSlot]
    _active_by_identity: dict[PackedWriteChunkIdentity, _ProducerSlot]
    _terminal_by_identity: dict[PackedWriteChunkIdentity, _ProducerTerminal]
    _terminal_by_request: dict[PackedWriteRequestKey, _ProducerRequestTerminalHistory]

    def __init__(
        self,
        geometry: PackedWriteProducerPoolGeometry,
        producer_rank: int,
    ) -> None:
        """Create ownership for one registered producer pool.

        :param geometry: Exact local registration geometry.
        :param producer_rank: Rank that owns every slot in the pool.
        """
        if type(geometry) is not PackedWriteProducerPoolGeometry:
            raise ValueError("producer geometry must be typed")
        if type(producer_rank) is not int or producer_rank < 0:
            raise ValueError("producer_rank must be a non-negative integer")
        self.geometry = geometry
        self.producer_rank = producer_rank
        self._slots = [
            _ProducerSlot(slot_index=index) for index in range(geometry.slot_count)
        ]
        self._active_by_identity = {}
        self._terminal_by_identity = {}
        self._terminal_by_request = {}

    def admit(
        self, request: PackedWriteRequest, consumer_agent: str
    ) -> ProducerAdmission:
        """Admit one command or return its exact idempotent replay state.

        :param request: Authenticated consumer request.
        :param consumer_agent: Native agent that sent the request.
        :returns: A new lease, an existing active lease, or a terminal replay.
        """
        self._validate_request(request, consumer_agent)
        identity = request.command.chunk
        active = self._active_by_identity.get(identity)
        if active is not None:
            self._validate_owned_request(active, request, consumer_agent)
            if active.state is ProducerSlotState.TOMBSTONED:
                raise StagingSafetyError(
                    "packed WRITE replay reached a tombstoned producer slot"
                )
            return ProducerAdmission(
                lease=active.lease,
                terminal=None,
                newly_admitted=False,
            )
        completed = self._terminal_by_identity.get(identity)
        if completed is not None:
            return ProducerAdmission(
                lease=None,
                terminal=self._validate_terminal_replay(
                    completed,
                    request,
                    consumer_agent,
                ),
                newly_admitted=False,
            )
        slot = next(
            (
                candidate
                for candidate in self._slots
                if candidate.state is ProducerSlotState.FREE
            ),
            None,
        )
        if slot is None:
            return ProducerAdmission(lease=None, terminal=None, newly_admitted=False)
        slot.generation += 1
        lease = ProducerSlotLease(
            slot_index=slot.slot_index,
            slot_generation=slot.generation,
            source_address=(
                self.geometry.base_address
                + slot.slot_index * self.geometry.slot_size_bytes
            ),
            payload_bytes=identity.exact_bytes,
            chunk=identity,
        )
        slot.state = ProducerSlotState.PACKING
        slot.request = request
        slot.consumer_agent = consumer_agent
        slot.lease = lease
        self._active_by_identity[identity] = slot
        return ProducerAdmission(lease=lease, terminal=None, newly_admitted=True)

    def terminal_for_replay(
        self,
        request: PackedWriteRequest,
        consumer_agent: str,
    ) -> PackedWriteArrived | PackedWriteFailedBeforeWrite | None:
        """Return an exact retained terminal without acquiring a producer slot.

        :param request: Authenticated consumer request.
        :param consumer_agent: Native agent that sent the replay.
        :returns: Exact retained terminal, or ``None`` for a new command.
        """
        self._validate_request(request, consumer_agent)
        completed = self._terminal_by_identity.get(request.command.chunk)
        if completed is None:
            return None
        return self._validate_terminal_replay(completed, request, consumer_agent)

    def active_for_replay(
        self,
        request: PackedWriteRequest,
        consumer_agent: str,
    ) -> bool:
        """Validate whether an exact command already owns a producer slot.

        :param request: Authenticated consumer request.
        :param consumer_agent: Native agent that sent the replay.
        :returns: Whether the exact command is already active.
        """
        self._validate_request(request, consumer_agent)
        active = self._active_by_identity.get(request.command.chunk)
        if active is None:
            return False
        self._validate_owned_request(active, request, consumer_agent)
        if active.state is ProducerSlotState.TOMBSTONED:
            raise StagingSafetyError(
                "packed WRITE replay reached a tombstoned producer slot"
            )
        return True

    def retire_offer(self, producer_request_id: str, offer_generation: int) -> int:
        """Retire replay state after one exact source offer is released.

        :param producer_request_id: Producer request whose decoder quorum sent
            its final completion proofs.
        :param offer_generation: Exact source allocation generation released by
            that quorum.
        :returns: Number of exact terminal records retired.
        """
        if type(producer_request_id) is not str or len(producer_request_id) == 0:
            raise ValueError("producer_request_id must be a non-empty string")
        if type(offer_generation) is not int or offer_generation <= 0:
            raise ValueError("offer_generation must be a positive integer")
        if any(
            identity.producer_request_id == producer_request_id
            and identity.offer_generation == offer_generation
            for identity in self._active_by_identity
        ):
            raise StagingSafetyError(
                "packed WRITE offer replay state cannot retire while active"
            )
        identities = tuple(
            identity
            for identity in self._terminal_by_identity
            if identity.producer_request_id == producer_request_id
            and identity.offer_generation == offer_generation
        )
        for identity in identities:
            del self._terminal_by_identity[identity]
        request_keys = tuple(
            key
            for key in self._terminal_by_request
            if key.producer_request_id == producer_request_id
            and key.offer_generation == offer_generation
        )
        for key in request_keys:
            del self._terminal_by_request[key]
        return len(identities)

    def retire_request_key(self, request_key: PackedWriteRequestKey) -> int:
        """Retire one decoder request history with no producer source contract.

        :param request_key: Exact generation-scoped packed request identity.
        :returns: Number of exact chunk terminals retired.
        """
        if type(request_key) is not PackedWriteRequestKey:
            raise ValueError("packed WRITE request key must be typed")
        if any(
            PackedWriteRequestKey.from_chunk(
                PackedWriteChunkKey.from_identity(identity)
            )
            == request_key
            for identity in self._active_by_identity
        ):
            raise StagingSafetyError(
                "packed WRITE request history cannot retire while active"
            )
        identities = tuple(
            identity
            for identity in self._terminal_by_identity
            if PackedWriteRequestKey.from_chunk(
                PackedWriteChunkKey.from_identity(identity)
            )
            == request_key
        )
        for identity in identities:
            del self._terminal_by_identity[identity]
        self._terminal_by_request.pop(request_key, None)
        return len(identities)

    def remember_failed_before_admission(
        self,
        request: PackedWriteRequest,
        consumer_agent: str,
        failed: PackedWriteFailedBeforeWrite,
    ) -> None:
        """Retain a proven pre-WRITE failure without acquiring a slot.

        :param request: Authenticated command rejected before admission.
        :param consumer_agent: Native decoder endpoint that sent the command.
        :param failed: Exact proof that the command never crossed the post boundary.
        """
        self._validate_request(request, consumer_agent)
        try:
            validate_packed_write_failed_before_write(request, failed)
        except ValueError as error:
            raise StagingSafetyError(
                "packed WRITE pre-admission failure changed its command"
            ) from error
        active = self._active_by_identity.get(request.command.chunk)
        if active is not None:
            raise StagingSafetyError(
                "packed WRITE pre-admission failure conflicts with an active slot"
            )
        self._remember_terminal(request, consumer_agent, failed)

    def validate_source_drained(
        self,
        drained: PackedWriteSourceDrained,
        consumer_agent: str,
    ) -> bool:
        """Validate the exact terminal prefix named by a decoder drain proof.

        :param drained: Decoder-authored published-prefix drain proof.
        :param consumer_agent: Native notification sender.
        :returns: Whether every named producer command is terminal locally.
        :raises StagingSafetyError: If retained history conflicts with the proof.
        """
        if type(drained) is not PackedWriteSourceDrained:
            raise ValueError("packed WRITE source-drain proof must be typed")
        if type(consumer_agent) is not str or len(consumer_agent) == 0:
            raise ValueError("packed WRITE source-drain sender must not be empty")
        request_key = PackedWriteRequestKey.from_chunk(
            PackedWriteChunkKey.from_identity(drained.anchor.chunk)
        )
        history = self._terminal_by_request.get(request_key)
        if history is None:
            return False
        if history.consumer_agent != consumer_agent:
            raise StagingSafetyError(
                "packed WRITE source-drain sender differs from its command endpoint"
            )
        anchor = history.terminals_by_ordinal.get(0)
        if anchor is None:
            return False
        if anchor.request.command != drained.anchor:
            raise StagingSafetyError(
                "packed WRITE source-drain anchor differs from chunk zero"
            )
        expected_ordinals = set(range(drained.published_chunk_count))
        observed_ordinals = set(history.terminals_by_ordinal)
        if observed_ordinals - expected_ordinals:
            raise StagingSafetyError(
                "packed WRITE source-drain omits a published producer command"
            )
        return observed_ordinals == expected_ordinals

    def mark_pack_complete(self, lease: ProducerSlotLease) -> None:
        """Cross from completed packing into native WRITE preparation.

        :param lease: Exact active producer lease.
        """
        slot = self._require_slot(lease, ProducerSlotState.PACKING)
        slot.state = ProducerSlotState.PREPARING_WRITE

    def begin_write_post(self, lease: ProducerSlotLease) -> None:
        """Cross the boundary after which FAILED_BEFORE_WRITE is forbidden.

        :param lease: Exact active producer lease.
        """
        slot = self._require_slot(lease, ProducerSlotState.PREPARING_WRITE)
        slot.state = ProducerSlotState.POSTING_WRITE

    def record_write_posted(self, lease: ProducerSlotLease, status: str) -> None:
        """Record the native submission result without weakening uncertainty.

        :param lease: Exact active producer lease.
        :param status: Native status returned by ``transfer``.
        """
        slot = self._require_slot(lease, ProducerSlotState.POSTING_WRITE)
        if status not in {"PROC", "DONE"}:
            self.tombstone(lease, f"native WRITE returned {status!r}")
            raise StagingSafetyError(f"native packed WRITE returned {status!r}")
        slot.state = (
            ProducerSlotState.WRITE_DONE
            if status == "DONE"
            else ProducerSlotState.WRITING
        )

    def record_write_query(self, lease: ProducerSlotLease, status: str) -> None:
        """Validate one poll result for a posted WRITE.

        :param lease: Exact active producer lease.
        :param status: Native status returned by the query.
        """
        slot = self._require_slot(lease, ProducerSlotState.WRITING)
        if status not in {"PROC", "DONE"}:
            self.tombstone(lease, f"native WRITE query returned {status!r}")
            raise StagingSafetyError(f"native packed WRITE became {status!r}")
        if status == "DONE":
            slot.state = ProducerSlotState.WRITE_DONE

    def mark_write_released(self, lease: ProducerSlotLease) -> None:
        """Record successful release of one locally DONE native handle.

        :param lease: Exact active producer lease.
        """
        slot = self._require_slot(lease, ProducerSlotState.WRITE_DONE)
        slot.state = ProducerSlotState.WRITE_RELEASED

    def complete_arrived(self, lease: ProducerSlotLease) -> PackedWriteArrived:
        """Release one locally DONE WRITE and retain its exact replay terminal.

        :param lease: Exact active producer lease.
        :returns: Attached arrival proof associated with the completed WRITE.
        """
        slot = self._require_slot(lease, ProducerSlotState.WRITE_RELEASED)
        assert slot.request is not None
        assert slot.consumer_agent is not None
        arrived = PackedWriteArrived(command=slot.request.command)
        self._remember_terminal(slot.request, slot.consumer_agent, arrived)
        self._release_slot(slot)
        return arrived

    def fail_before_write(
        self,
        lease: ProducerSlotLease,
        failed: PackedWriteFailedBeforeWrite,
    ) -> None:
        """Release one command proven never to have crossed WRITE post.

        :param lease: Exact active producer lease.
        :param failed: Typed no-WRITE terminal.
        """
        slot = self._require_slot(
            lease,
            ProducerSlotState.PACKING,
            ProducerSlotState.PREPARING_WRITE,
        )
        assert slot.request is not None
        assert slot.consumer_agent is not None
        try:
            validate_packed_write_failed_before_write(slot.request, failed)
        except ValueError as error:
            raise StagingSafetyError(
                "packed WRITE pre-WRITE failure changed its command"
            ) from error
        self._remember_terminal(slot.request, slot.consumer_agent, failed)
        self._release_slot(slot)

    def tombstone(self, lease: ProducerSlotLease, reason: str) -> None:
        """Permanently prevent reuse after uncertain device or native work.

        :param lease: Exact active producer lease.
        :param reason: Diagnostic description of the uncertainty.
        """
        if type(reason) is not str or len(reason) == 0:
            raise ValueError("tombstone reason must not be empty")
        slot = self._require_slot(lease)
        slot.state = ProducerSlotState.TOMBSTONED

    def request_for_identity(
        self,
        identity: PackedWriteChunkIdentity,
    ) -> PackedWriteRequest | None:
        """Return the active request for one identity.

        :param identity: Rankful chunk identity.
        :returns: Active request, if present.
        """
        slot = self._active_by_identity.get(identity)
        return None if slot is None else slot.request

    def owns_consumer_agent(self, consumer_agent: str) -> bool:
        """Return whether live or replay state retains one imported agent.

        :param consumer_agent: Native remote-agent identity.
        :returns: Whether any active slot owns that identity.
        """
        if any(
            slot.consumer_agent == consumer_agent
            and slot.state is not ProducerSlotState.FREE
            for slot in self._slots
        ):
            return True
        return any(
            history.consumer_agent == consumer_agent
            for history in self._terminal_by_request.values()
        )

    @property
    def active_slot_count(self) -> int:
        """Return the number of live or tombstoned generations."""
        return sum(slot.state is not ProducerSlotState.FREE for slot in self._slots)

    @property
    def free_slot_count(self) -> int:
        """Return the number of immediately reusable producer slots."""
        return len(self._slots) - self.active_slot_count

    @property
    def has_registered_ownership(self) -> bool:
        """Return whether deregistration would invalidate a slot generation."""
        return self.active_slot_count > 0

    def _validate_request(
        self,
        request: PackedWriteRequest,
        consumer_agent: str,
    ) -> None:
        if type(request) is not PackedWriteRequest:
            raise ValueError("packed WRITE request must be typed")
        if type(consumer_agent) is not str or len(consumer_agent) == 0:
            raise ValueError("consumer_agent must be a non-empty string")
        identity = request.command.chunk
        if identity.producer_rank != self.producer_rank:
            raise StagingSafetyError("packed WRITE reached the wrong producer rank")
        if identity.exact_bytes > self.geometry.slot_size_bytes:
            raise StagingSafetyError("packed WRITE payload exceeds producer slot")

    @staticmethod
    def _validate_owned_request(
        slot: _ProducerSlot,
        request: PackedWriteRequest,
        consumer_agent: str,
    ) -> None:
        assert slot.request is not None
        try:
            validate_packed_write_request_replay(slot.request, request)
        except ValueError as error:
            raise StagingSafetyError(
                "packed WRITE active replay changed authenticated fields"
            ) from error
        if slot.consumer_agent != consumer_agent:
            raise StagingSafetyError(
                "packed WRITE replay changed its authenticated consumer agent"
            )

    @staticmethod
    def _validate_terminal_replay(
        completed: _ProducerTerminal,
        request: PackedWriteRequest,
        consumer_agent: str,
    ) -> PackedWriteArrived | PackedWriteFailedBeforeWrite:
        try:
            validate_packed_write_request_replay(completed.request, request)
        except ValueError as error:
            raise StagingSafetyError(
                "packed WRITE terminal replay changed authenticated fields"
            ) from error
        if completed.consumer_agent != consumer_agent:
            raise StagingSafetyError(
                "packed WRITE replay changed its authenticated consumer agent"
            )
        return completed.terminal

    def _require_slot(
        self,
        lease: ProducerSlotLease,
        *states: ProducerSlotState,
    ) -> _ProducerSlot:
        if type(lease) is not ProducerSlotLease:
            raise ValueError("producer lease must be typed")
        if lease.slot_index < 0 or lease.slot_index >= len(self._slots):
            raise StagingSafetyError("producer lease slot is outside its pool")
        slot = self._slots[lease.slot_index]
        if slot.lease != lease:
            raise StagingSafetyError("producer lease generation is stale")
        if len(states) > 0 and slot.state not in states:
            raise StagingSafetyError(
                f"producer slot is {slot.state.name}, expected "
                + "/".join(state.name for state in states)
            )
        return slot

    def _remember_terminal(
        self,
        request: PackedWriteRequest,
        consumer_agent: str,
        terminal: PackedWriteArrived | PackedWriteFailedBeforeWrite,
    ) -> None:
        identity = request.command.chunk
        prior = self._terminal_by_identity.get(identity)
        record = _ProducerTerminal(request, consumer_agent, terminal)
        if prior is not None and prior != record:
            raise StagingSafetyError("packed WRITE producer terminal conflict")
        if prior is None and len(self._terminal_by_identity) >= _MAX_TERMINAL_HISTORY:
            raise StagingSafetyError("packed WRITE producer terminal history exhausted")
        self._terminal_by_identity[identity] = record

        request_key = PackedWriteRequestKey.from_chunk(
            PackedWriteChunkKey.from_identity(identity)
        )
        history = self._terminal_by_request.get(request_key)
        if history is None:
            history = _ProducerRequestTerminalHistory(
                consumer_endpoint=request.consumer_endpoint,
                consumer_agent=consumer_agent,
            )
            self._terminal_by_request[request_key] = history
        elif (
            history.consumer_endpoint != request.consumer_endpoint
            or history.consumer_agent != consumer_agent
        ):
            raise StagingSafetyError(
                "packed WRITE request history changed its authenticated endpoint"
            )
        ordinal = identity.chunk_ordinal
        prior_ordinal = history.terminals_by_ordinal.get(ordinal)
        if prior_ordinal is not None and prior_ordinal != record:
            raise StagingSafetyError("packed WRITE request terminal prefix conflicted")
        history.terminals_by_ordinal[ordinal] = record

    def _release_slot(self, slot: _ProducerSlot) -> None:
        assert slot.request is not None
        self._active_by_identity.pop(slot.request.command.chunk)
        slot.state = ProducerSlotState.FREE
        slot.request = None
        slot.consumer_agent = None
        slot.lease = None


class ConsumerSlotState(Enum):
    """Decoder receive-slot lifecycle."""

    FREE = auto()
    PUBLISHING = auto()
    WAITING_WRITES = auto()
    SCATTERING = auto()
    TOMBSTONED = auto()


@dataclass(frozen=True, slots=True)
class ConsumerSlotLease:
    """Generation-scoped ownership of one decoder rank-major slot."""

    chunk: PackedWriteChunkKey
    binding: PackedWriteSlotBinding

    @property
    def slot_index(self) -> int:
        """Return the bound pool slot index."""
        return self.binding.slot_index


@dataclass(slots=True)
class _ConsumerSlot:
    """Mutable state for one decoder receive slot."""

    slot_index: int
    generation: int = 0
    state: ConsumerSlotState = ConsumerSlotState.FREE
    lease: ConsumerSlotLease | None = None
    requests: dict[int, PackedWriteRequest] = field(default_factory=dict)
    published_ranks: set[int] = field(default_factory=set)
    arrived: dict[int, PackedWriteArrived] = field(default_factory=dict)
    failed: dict[int, PackedWriteFailedBeforeWrite] = field(default_factory=dict)


class PackedWriteConsumerPool:
    """Own decoder rank-major slots through terminal quorum and scatter."""

    geometry: PackedWriteConsumerPoolGeometry
    _slots: list[_ConsumerSlot]
    _active_by_chunk: dict[PackedWriteChunkKey, _ConsumerSlot]

    def __init__(self, geometry: PackedWriteConsumerPoolGeometry) -> None:
        """Create ownership for one registered decoder pool.

        :param geometry: Exact local rank-major registration geometry.
        """
        if type(geometry) is not PackedWriteConsumerPoolGeometry:
            raise ValueError("consumer geometry must be typed")
        self.geometry = geometry
        self._slots = [
            _ConsumerSlot(slot_index=index) for index in range(geometry.slot_count)
        ]
        self._active_by_chunk = {}

    def reserve(
        self,
        chunk: PackedWriteChunkKey,
        payload_bytes: int,
    ) -> ConsumerSlotLease | None:
        """Reserve a slot before constructing or publishing any rank command.

        :param chunk: Rank-independent exact chunk identity.
        :param payload_bytes: Bytes each producer rank will WRITE.
        :returns: A new slot lease, or ``None`` under bounded backpressure.
        """
        if type(chunk) is not PackedWriteChunkKey:
            raise ValueError("consumer chunk key must be typed")
        if chunk.producer_tp_size != self.geometry.source_tp_size:
            raise StagingSafetyError(
                "packed WRITE chunk source topology differs from the consumer pool"
            )
        if type(payload_bytes) is not int or payload_bytes != chunk.exact_bytes:
            raise ValueError("consumer payload_bytes must equal the chunk byte count")
        if payload_bytes > self.geometry.rank_stride_bytes:
            raise ValueError("consumer payload_bytes exceeds one rank slab")
        if chunk in self._active_by_chunk:
            raise StagingSafetyError("packed WRITE chunk already owns a consumer slot")
        slot = next(
            (
                candidate
                for candidate in self._slots
                if candidate.state is ConsumerSlotState.FREE
            ),
            None,
        )
        if slot is None:
            return None
        slot.generation += 1
        binding = PackedWriteSlotBinding(
            pool_registration_generation=self.geometry.registration_generation,
            slot_index=slot.slot_index,
            slot_generation=slot.generation,
            payload_bytes=payload_bytes,
        )
        lease = ConsumerSlotLease(chunk=chunk, binding=binding)
        slot.state = ConsumerSlotState.PUBLISHING
        slot.lease = lease
        self._active_by_chunk[chunk] = slot
        return lease

    def bind_requests(
        self,
        lease: ConsumerSlotLease,
        requests: tuple[PackedWriteRequest, ...],
    ) -> None:
        """Bind the exact rank quorum before the first notification send.

        :param lease: Reserved decoder slot.
        :param requests: Complete producer-rank command quorum.
        """
        slot = self._require_slot(lease, ConsumerSlotState.PUBLISHING)
        if type(requests) is not tuple:
            raise ValueError("packed WRITE rank requests must be a tuple")
        expected_ranks = set(range(self.geometry.source_tp_size))
        request_map: dict[int, PackedWriteRequest] = {}
        for request in requests:
            if type(request) is not PackedWriteRequest:
                raise ValueError("packed WRITE rank request must be typed")
            identity = request.command.chunk
            if PackedWriteChunkKey.from_identity(identity) != lease.chunk:
                raise StagingSafetyError(
                    "packed WRITE rank quorum changed chunk identity"
                )
            if request.command.destination != lease.binding:
                raise StagingSafetyError(
                    "packed WRITE request changed destination lease"
                )
            if request.consumer_endpoint.consumer_pool != self.geometry:
                raise StagingSafetyError(
                    "packed WRITE request changed consumer geometry"
                )
            if identity.producer_rank in request_map:
                raise StagingSafetyError("packed WRITE rank quorum contains duplicates")
            request_map[identity.producer_rank] = request
        if set(request_map) != expected_ranks:
            raise StagingSafetyError("packed WRITE rank quorum is incomplete")
        if len(slot.requests) > 0:
            if slot.requests != request_map:
                raise StagingSafetyError(
                    "packed WRITE rank quorum was rebound inconsistently"
                )
            return
        slot.requests = request_map

    def mark_rank_published(self, lease: ConsumerSlotLease, producer_rank: int) -> None:
        """Record one notification send that returned successfully.

        :param lease: Exact decoder slot.
        :param producer_rank: Command rank whose send returned.
        """
        slot = self._require_slot(lease, ConsumerSlotState.PUBLISHING)
        if type(producer_rank) is not int:
            raise ValueError("producer_rank must be an integer")
        if producer_rank not in slot.requests:
            raise StagingSafetyError("published packed WRITE rank is not bound")
        if producer_rank in slot.published_ranks:
            raise StagingSafetyError("packed WRITE rank was published twice")
        slot.published_ranks.add(producer_rank)

    def seal_publication(self, lease: ConsumerSlotLease) -> None:
        """Enter terminal wait after every rank command was published.

        :param lease: Exact decoder slot.
        """
        slot = self._require_slot(lease, ConsumerSlotState.PUBLISHING)
        if slot.published_ranks != set(slot.requests):
            raise StagingSafetyError("packed WRITE publication quorum is incomplete")
        slot.state = ConsumerSlotState.WAITING_WRITES

    def record_arrived(
        self,
        lease: ConsumerSlotLease,
        arrived: PackedWriteArrived,
    ) -> bool:
        """Record one attached remote-memory-complete proof.

        :param lease: Exact decoder slot.
        :param arrived: Authenticated producer arrival.
        :returns: Whether the complete rank quorum is now terminal.
        """
        slot = self._require_slot(lease, ConsumerSlotState.WAITING_WRITES)
        rank = arrived.command.chunk.producer_rank
        request = self._request_for_rank(slot, rank)
        try:
            validate_packed_write_arrived(request, arrived)
        except ValueError as error:
            raise StagingSafetyError(
                "packed WRITE arrival changed its published command"
            ) from error
        if rank in slot.failed:
            raise StagingSafetyError("packed WRITE rank both arrived and failed")
        prior = slot.arrived.get(rank)
        if prior is not None and prior != arrived:
            raise StagingSafetyError("packed WRITE arrival replay changed command")
        slot.arrived[rank] = arrived
        return self._terminal_complete(slot)

    def record_failed_before_write(
        self,
        lease: ConsumerSlotLease,
        failed: PackedWriteFailedBeforeWrite,
    ) -> bool:
        """Record one exact proof that a producer rank never posted.

        :param lease: Exact decoder slot.
        :param failed: Authenticated producer pre-WRITE failure.
        :returns: Whether the complete rank quorum is now terminal.
        """
        slot = self._require_slot(lease, ConsumerSlotState.WAITING_WRITES)
        rank = failed.command.chunk.producer_rank
        request = self._request_for_rank(slot, rank)
        try:
            validate_packed_write_failed_before_write(request, failed)
        except ValueError as error:
            raise StagingSafetyError(
                "packed WRITE failure changed its published command"
            ) from error
        if rank in slot.arrived:
            raise StagingSafetyError("packed WRITE rank both arrived and failed")
        prior = slot.failed.get(rank)
        if prior is not None and prior != failed:
            raise StagingSafetyError("packed WRITE failure replay changed fields")
        slot.failed[rank] = failed
        return self._terminal_complete(slot)

    def begin_scatter(self, lease: ConsumerSlotLease) -> None:
        """Transfer ownership to scatter after an all-ARRIVED quorum.

        :param lease: Exact decoder slot.
        """
        slot = self._require_slot(lease, ConsumerSlotState.WAITING_WRITES)
        if set(slot.arrived) != set(slot.requests) or len(slot.failed) > 0:
            raise StagingSafetyError("scatter requires an all-ARRIVED rank quorum")
        slot.state = ConsumerSlotState.SCATTERING

    def complete_scatter(self, lease: ConsumerSlotLease) -> PackedWriteChunkKey:
        """Release a slot after its scatter completion event is terminal.

        :param lease: Exact decoder slot.
        :returns: Completed logical chunk key.
        """
        slot = self._require_slot(lease, ConsumerSlotState.SCATTERING)
        chunk = lease.chunk
        self._release_slot(slot)
        return chunk

    def discard_terminal(self, lease: ConsumerSlotLease) -> PackedWriteChunkKey:
        """Release a terminal rank quorum without scattering invalid data.

        :param lease: Exact decoder slot.
        :returns: Discarded logical chunk key.
        """
        slot = self._require_slot(lease, ConsumerSlotState.WAITING_WRITES)
        if self._terminal_complete(slot) is False:
            raise StagingSafetyError("cannot discard before every published rank ends")
        chunk = lease.chunk
        self._release_slot(slot)
        return chunk

    def tombstone(self, lease: ConsumerSlotLease, reason: str) -> None:
        """Permanently prevent reuse after uncertain publication or device work.

        :param lease: Exact decoder slot.
        :param reason: Diagnostic description of the uncertainty.
        """
        if type(reason) is not str or len(reason) == 0:
            raise ValueError("tombstone reason must not be empty")
        slot = self._require_slot(lease)
        slot.state = ConsumerSlotState.TOMBSTONED

    def request_for_rank(
        self,
        lease: ConsumerSlotLease,
        producer_rank: int,
    ) -> PackedWriteRequest:
        """Return one exact rank request owned by a slot.

        :param lease: Exact decoder slot.
        :param producer_rank: Producer rank to resolve.
        :returns: Bound request.
        """
        slot = self._require_slot(lease)
        return self._request_for_rank(slot, producer_rank)

    def terminal_complete(self, lease: ConsumerSlotLease) -> bool:
        """Return whether every published rank has one exact terminal.

        :param lease: Exact decoder slot.
        :returns: Complete terminal-quorum state.
        """
        slot = self._require_slot(lease)
        return self._terminal_complete(slot)

    def has_failure(self, lease: ConsumerSlotLease) -> bool:
        """Return whether the terminal quorum contains a pre-WRITE failure.

        :param lease: Exact decoder slot.
        :returns: Whether any rank failed before WRITE.
        """
        slot = self._require_slot(lease)
        return len(slot.failed) > 0

    @property
    def free_slot_count(self) -> int:
        """Return the number of immediately reusable decoder slots."""
        return sum(slot.state is ConsumerSlotState.FREE for slot in self._slots)

    @property
    def active_slot_count(self) -> int:
        """Return the number of live or tombstoned decoder generations."""
        return len(self._slots) - self.free_slot_count

    @property
    def has_registered_ownership(self) -> bool:
        """Return whether deregistration would invalidate a slot generation."""
        return self.active_slot_count > 0

    def _require_slot(
        self,
        lease: ConsumerSlotLease,
        *states: ConsumerSlotState,
    ) -> _ConsumerSlot:
        if type(lease) is not ConsumerSlotLease:
            raise ValueError("consumer lease must be typed")
        if lease.slot_index < 0 or lease.slot_index >= len(self._slots):
            raise StagingSafetyError("consumer lease slot is outside its pool")
        slot = self._slots[lease.slot_index]
        if slot.lease != lease:
            raise StagingSafetyError("consumer lease generation is stale")
        if len(states) > 0 and slot.state not in states:
            raise StagingSafetyError(
                f"consumer slot is {slot.state.name}, expected "
                + "/".join(state.name for state in states)
            )
        return slot

    @staticmethod
    def _request_for_rank(
        slot: _ConsumerSlot,
        producer_rank: int,
    ) -> PackedWriteRequest:
        if type(producer_rank) is not int:
            raise ValueError("producer_rank must be an integer")
        request = slot.requests.get(producer_rank)
        if request is None:
            raise StagingSafetyError("packed WRITE terminal rank is not bound")
        return request

    @staticmethod
    def _terminal_complete(slot: _ConsumerSlot) -> bool:
        terminal_ranks = set(slot.arrived) | set(slot.failed)
        return (
            len(slot.requests) > 0
            and len(set(slot.arrived) & set(slot.failed)) == 0
            and terminal_ranks == set(slot.requests)
        )

    def _release_slot(self, slot: _ConsumerSlot) -> None:
        assert slot.lease is not None
        self._active_by_chunk.pop(slot.lease.chunk)
        slot.state = ConsumerSlotState.FREE
        slot.lease = None
        slot.requests.clear()
        slot.published_ranks.clear()
        slot.arrived.clear()
        slot.failed.clear()


@dataclass(slots=True)
class PackedWriteRequestTracker:
    """Track atomic completion of every bounded chunk in one request."""

    request: PackedWriteRequestKey
    completed_chunks: dict[int, PackedWriteChunkKey] = field(default_factory=dict)
    failed: bool = False
    tombstoned: bool = False
    published: bool = False

    @classmethod
    def create(cls, first_chunk: PackedWriteChunkKey) -> "PackedWriteRequestTracker":
        """Create a tracker from the first canonical chunk.

        :param first_chunk: Chunk zero for the request.
        :returns: New request tracker.
        """
        if type(first_chunk) is not PackedWriteChunkKey:
            raise ValueError("packed WRITE tracker chunk must be typed")
        if first_chunk.chunk_ordinal != 0:
            raise ValueError("packed WRITE request tracker must start at chunk zero")
        return cls(request=PackedWriteRequestKey.from_chunk(first_chunk))

    def record_chunk_complete(self, chunk: PackedWriteChunkKey) -> bool:
        """Record one scattered chunk exactly once.

        :param chunk: Completed canonical chunk.
        :returns: Whether this completion made the request publishable.
        """
        if type(chunk) is not PackedWriteChunkKey:
            raise ValueError("packed WRITE completed chunk must be typed")
        if PackedWriteRequestKey.from_chunk(chunk) != self.request:
            raise StagingSafetyError("packed WRITE chunk changed request identity")
        if not 0 <= chunk.chunk_ordinal < self.request.chunk_count:
            raise StagingSafetyError(
                "packed WRITE chunk ordinal is outside its request"
            )
        prior = self.completed_chunks.get(chunk.chunk_ordinal)
        if prior is not None and prior != chunk:
            raise StagingSafetyError("packed WRITE chunk completion conflicted")
        if prior is not None:
            return False
        self.completed_chunks[chunk.chunk_ordinal] = chunk
        return self.publication_eligible

    def fail(self) -> None:
        """Prevent successful publication while published work drains."""
        if self.published:
            raise StagingSafetyError("published packed WRITE request cannot fail")
        self.failed = True

    def tombstone(self) -> None:
        """Permanently prevent publication after uncertain ownership."""
        if self.published:
            raise StagingSafetyError("published packed WRITE request cannot tombstone")
        self.tombstoned = True

    def mark_published(self) -> None:
        """Record the one successful request publication transition."""
        if self.publication_eligible is False or self.published:
            raise StagingSafetyError("packed WRITE request is not publishable")
        self.published = True

    @property
    def publication_eligible(self) -> bool:
        """Return whether every canonical chunk can be published atomically."""
        return (
            self.failed is False
            and self.tombstoned is False
            and self.published is False
            and len(self.completed_chunks) == self.request.chunk_count
            and set(self.completed_chunks) == set(range(self.request.chunk_count))
        )
