# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generation-scoped ownership for coalesced NIXL staging memory."""

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class StagingSafetyError(RuntimeError):
    """Signal that a coalesced transfer lost its proof of memory safety."""


class HandleState(StrEnum):
    """Safety-relevant state of one possible native staging writer."""

    NEVER_POSTED = "never_posted"
    PREPARING = "preparing"
    PREPARE_FAILED = "prepare_failed"
    PREPARED = "prepared"
    SEALED_UNPOSTED = "sealed_unposted"
    POSTING = "posting"
    PROC = "proc"
    DONE = "done"
    UNKNOWN = "unknown"
    ERR = "err"


# Releasing a loaded NIXL/UCX request may cancel or free its request object
# without proving that cancellation completed. ERR and UNKNOWN can therefore
# still write; only DONE and states that never posted prove native quiescence.
_NATIVE_QUIESCENT_STATES = {
    HandleState.NEVER_POSTED,
    HandleState.PREPARE_FAILED,
    HandleState.SEALED_UNPOSTED,
    HandleState.DONE,
}


@dataclass(frozen=True)
class StagingLease:
    """Identify one allocation generation in the registered staging buffer.

    :ivar generation: Monotonic allocation generation.
    :ivar owner_id: Unique logical plan identity.
    :ivar offset: Registration-relative byte offset.
    :ivar size: Reserved byte count.
    """

    generation: int
    owner_id: str
    offset: int
    size: int


@dataclass
class StagingHandleSlot:
    """Own one source-rank handle for the lifetime of its staging lease.

    :ivar source_rank: Producer tensor-parallel rank.
    :ivar state: Current safety state.
    :ivar native_handle: Strong reference to the NIXL handle, if prepared.
    :ivar native_released: Whether a DONE handle was released successfully.
    """

    source_rank: int
    state: HandleState = HandleState.NEVER_POSTED
    native_handle: Any | None = None
    native_released: bool = False


@dataclass(frozen=True)
class StagingHandleSnapshot:
    """Serializable evidence for one possible native writer.

    :ivar source_rank: Producer tensor-parallel rank.
    :ivar state: Safety-relevant handle state.
    :ivar native_handle_present: Whether the owner retains the opaque handle.
    :ivar native_released: Whether DONE handle resource release succeeded.
    """

    source_rank: int
    state: HandleState
    native_handle_present: bool
    native_released: bool

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable handle record.

        :returns: Rank, state, presence, and release evidence.
        """
        return {
            "source_rank": self.source_rank,
            "state": self.state.value,
            "native_handle_present": self.native_handle_present,
            "native_released": self.native_released,
        }


@dataclass(frozen=True)
class StagingOwnershipSnapshot:
    """Serializable safety evidence for one staging generation.

    :ivar generation: Monotonic allocation generation.
    :ivar owner_id: Unique logical owner identity.
    :ivar request_id: Decoder request identifier.
    :ivar remote_engine_id: Remote engine whose resources back the transfer.
    :ivar offset: Registration-relative byte offset.
    :ivar size: Reserved byte count.
    :ivar age_s: Monotonic age at snapshot time.
    :ivar handles: Ordered producer-rank handle evidence.
    :ivar posting_sealed: Whether native submission is closed.
    :ivar operation_failed: Whether publication is forbidden.
    :ivar permanently_tombstoned: Whether in-process reuse is forbidden.
    :ivar native_quiescent: Whether every possible native writer is quiescent.
    :ivar device_read_started: Whether a device reader entered staging.
    :ivar device_quiescent: Whether every device reader completed.
    :ivar reusable: Whether allocator reclamation is proved safe.
    :ivar released: Whether the allocator reclaimed the range.
    :ivar failure_reason: First logical failure reason.
    """

    generation: int
    owner_id: str
    request_id: str
    remote_engine_id: str | None
    offset: int
    size: int
    age_s: float
    handles: tuple[StagingHandleSnapshot, ...]
    posting_sealed: bool
    operation_failed: bool
    permanently_tombstoned: bool
    native_quiescent: bool
    device_read_started: bool
    device_quiescent: bool
    reusable: bool
    released: bool
    failure_reason: str | None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable evidence record.

        :returns: Ownership evidence with string-valued handle states.
        """
        return {
            "generation": self.generation,
            "owner_id": self.owner_id,
            "request_id": self.request_id,
            "remote_engine_id": self.remote_engine_id,
            "offset": self.offset,
            "size": self.size,
            "age_s": self.age_s,
            "handles": [handle.to_dict() for handle in self.handles],
            "posting_sealed": self.posting_sealed,
            "operation_failed": self.operation_failed,
            "permanently_tombstoned": self.permanently_tombstoned,
            "native_quiescent": self.native_quiescent,
            "device_read_started": self.device_read_started,
            "device_quiescent": self.device_quiescent,
            "reusable": self.reusable,
            "released": self.released,
            "failure_reason": self.failure_reason,
        }


@dataclass
class CoalescedStagingPlan:
    """Own a complete coalesced transfer and every actor touching its range.

    :ivar lease: Generation-scoped staging allocation.
    :ivar request_id: Decoder request identifier.
    :ivar remote_engine_id: Remote engine whose native resources back the plan.
    :ivar slots: Native handle slots keyed by producer rank.
    :ivar scatter: Transfer geometry retained for the plan lifetime.
    :ivar created_at: Monotonic creation time.
    :ivar warn_after_s: Age at which a structured warning becomes due.
    :ivar fail_after_s: Age at which an in-progress plan must fail closed.
    :ivar posting_sealed: Whether no further native post may begin.
    :ivar operation_failed: Whether scatter and publication are forbidden.
    :ivar failure_reason: First reason the logical operation failed.
    :ivar device_quiescent: Whether every staging reader on the device completed.
    :ivar device_read_started: Whether any device reader entered the range.
    :ivar permanently_tombstoned: Whether uncertain native/device activity
        irrevocably forbids reuse in this process.
    :ivar warning_emitted: Whether the age warning was already emitted.
    :ivar released: Whether the allocator reclaimed the lease.
    """

    lease: StagingLease
    request_id: str
    remote_engine_id: str | None
    slots: dict[int, StagingHandleSlot]
    scatter: dict[str, Any]
    created_at: float
    warn_after_s: float
    fail_after_s: float
    posting_sealed: bool = False
    operation_failed: bool = False
    failure_reason: str | None = None
    device_quiescent: bool = True
    device_read_started: bool = False
    permanently_tombstoned: bool = False
    warning_emitted: bool = False
    released: bool = False

    @classmethod
    def create(
        cls,
        *,
        lease: StagingLease,
        request_id: str,
        remote_engine_id: str | None,
        source_ranks: tuple[int, ...],
        scatter: dict[str, Any] | None = None,
        created_at: float | None = None,
        warn_after_s: float = 1.0,
        fail_after_s: float = 10.0,
    ) -> "CoalescedStagingPlan":
        """Create an owner before any native transfer operation begins.

        :param lease: Exact registered staging allocation.
        :param request_id: Decoder request identifier.
        :param remote_engine_id: Remote engine owning the source registration.
        :param source_ranks: Complete set of independently posted P ranks.
        :param scatter: Transfer geometry retained for the plan lifetime.
        :param created_at: Monotonic creation time.
        :param warn_after_s: Structured-warning threshold.
        :param fail_after_s: Fail-stop threshold for permanent progress.
        :returns: New staging owner.
        """
        if len(request_id) == 0:
            raise ValueError("request_id must not be empty")
        if remote_engine_id is not None and len(remote_engine_id) == 0:
            raise ValueError("remote_engine_id must be non-empty when present")
        if len(source_ranks) == 0 or len(set(source_ranks)) != len(source_ranks):
            raise ValueError("source_ranks must be non-empty and unique")
        if any(type(rank) is not int or rank < 0 for rank in source_ranks):
            raise ValueError("source_ranks must contain non-negative integers")
        if warn_after_s <= 0 or fail_after_s <= warn_after_s:
            raise ValueError("fail_after_s must be greater than warn_after_s")
        return cls(
            lease=lease,
            request_id=request_id,
            remote_engine_id=remote_engine_id,
            slots={rank: StagingHandleSlot(rank) for rank in source_ranks},
            scatter={} if scatter is None else dict(scatter),
            created_at=time.monotonic() if created_at is None else created_at,
            warn_after_s=warn_after_s,
            fail_after_s=fail_after_s,
        )

    def begin_prepare(self, source_rank: int) -> None:
        """Enter native preparation for one rank.

        :param source_rank: Producer rank being prepared.
        """
        slot = self._slot(source_rank)
        self._require_posting_open()
        self._require_state(slot, HandleState.NEVER_POSTED)
        slot.state = HandleState.PREPARING

    def attach_handle(self, source_rank: int, native_handle: Any) -> None:
        """Attach a prepared handle before its post can start a writer.

        :param source_rank: Producer rank owning the handle.
        :param native_handle: Opaque NIXL transfer handle.
        """
        slot = self._slot(source_rank)
        self._require_state(slot, HandleState.PREPARING)
        if native_handle is None:
            raise ValueError("native_handle must not be None")
        slot.native_handle = native_handle
        slot.state = HandleState.PREPARED

    def record_prepare_failure(self, source_rank: int, reason: str) -> None:
        """Record a preparation failure that cannot have started a DMA writer.

        :param source_rank: Producer rank whose preparation failed.
        :param reason: Failure detail.
        """
        slot = self._slot(source_rank)
        self._require_state(slot, HandleState.PREPARING)
        slot.state = HandleState.PREPARE_FAILED
        self.fail(reason)

    def begin_post(self, source_rank: int) -> None:
        """Mark the boundary after which native submission is uncertain.

        :param source_rank: Producer rank entering ``transfer``.
        """
        slot = self._slot(source_rank)
        self._require_posting_open()
        self._require_state(slot, HandleState.PREPARED)
        slot.state = HandleState.POSTING

    def record_post_result(self, source_rank: int, status: str) -> None:
        """Consume the immediate status returned by ``transfer``.

        :param source_rank: Producer rank whose post returned.
        :param status: NIXL ``DONE``, ``PROC``, or ``ERR`` status.
        """
        slot = self._slot(source_rank)
        self._require_state(slot, HandleState.POSTING)
        if status == "DONE":
            slot.state = HandleState.DONE
            return
        if status == "PROC":
            slot.state = HandleState.PROC
            return
        slot.state = HandleState.ERR if status == "ERR" else HandleState.UNKNOWN
        self.permanently_tombstoned = True
        self.fail(f"rank {source_rank} post returned {status!r}")

    def record_post_exception(self, source_rank: int, reason: str) -> None:
        """Tombstone a handle after native posting raised.

        :param source_rank: Producer rank whose post raised.
        :param reason: Failure detail.
        """
        slot = self._slot(source_rank)
        self._require_state(slot, HandleState.POSTING)
        slot.state = HandleState.UNKNOWN
        self.permanently_tombstoned = True
        self.fail(reason)

    def record_query_result(self, source_rank: int, status: str) -> None:
        """Advance a posted handle from one authoritative status observation.

        :param source_rank: Producer rank being polled.
        :param status: NIXL ``DONE``, ``PROC``, or ``ERR`` status.
        """
        slot = self._slot(source_rank)
        if slot.state not in {HandleState.PROC, HandleState.UNKNOWN}:
            raise RuntimeError(
                f"rank {source_rank} in state {slot.state} cannot be queried"
            )
        if status == "DONE":
            slot.state = HandleState.DONE
            return
        if status == "PROC":
            return
        slot.state = HandleState.ERR if status == "ERR" else HandleState.UNKNOWN
        self.permanently_tombstoned = True
        self.fail(f"rank {source_rank} query returned {status!r}")

    def record_query_exception(self, source_rank: int, reason: str) -> None:
        """Preserve uncertainty when a status query raises.

        :param source_rank: Producer rank whose query raised.
        :param reason: Failure detail.
        """
        slot = self._slot(source_rank)
        if slot.state not in {HandleState.PROC, HandleState.UNKNOWN}:
            raise RuntimeError(
                f"rank {source_rank} in state {slot.state} cannot fail a query"
            )
        slot.state = HandleState.UNKNOWN
        self.permanently_tombstoned = True
        self.fail(reason)

    def seal_posting(self) -> None:
        """Irrevocably prevent every remaining native post."""
        self._require_live()
        if self.posting_sealed:
            raise RuntimeError("staging plan posting is already sealed")
        for slot in self.slots.values():
            if slot.state is HandleState.PREPARING:
                slot.state = HandleState.PREPARE_FAILED
            elif slot.state is HandleState.PREPARED:
                slot.state = HandleState.SEALED_UNPOSTED
        self.posting_sealed = True

    def fail(self, reason: str) -> None:
        """Latch one logical failure without weakening native ownership.

        :param reason: Failure detail.
        """
        self._require_live()
        if len(reason) == 0:
            raise ValueError("failure reason must not be empty")
        self.operation_failed = True
        if self.failure_reason is None:
            self.failure_reason = reason

    def tombstone(self, reason: str) -> None:
        """Irrevocably forbid in-process reuse of this generation.

        :param reason: Failure detail.
        """
        self.permanently_tombstoned = True
        self.fail(reason)

    def begin_device_read(self) -> None:
        """Record that CUDA work may still read from staging."""
        self._require_live()
        if not self.ready_to_scatter:
            raise StagingSafetyError("only a successful terminal plan may scatter")
        self.device_read_started = True
        self.device_quiescent = False

    def mark_device_quiescent(self) -> None:
        """Record successful completion of every staging reader."""
        self._require_live()
        if self.device_read_started is False:
            raise StagingSafetyError("device quiescence has no preceding device reader")
        if self.device_quiescent:
            raise StagingSafetyError("device readers are already quiescent")
        self.device_quiescent = True

    def mark_native_released(self, source_rank: int) -> None:
        """Record resource release after independent DONE proof.

        :param source_rank: Producer rank whose DONE handle was released.
        """
        slot = self._slot(source_rank)
        self._require_state(slot, HandleState.DONE)
        slot.native_released = True

    def warning_due(self, now: float) -> bool:
        """Return whether the one-shot structured warning is due.

        :param now: Current monotonic time.
        :returns: Whether a warning should be emitted.
        """
        return (
            self.warning_emitted is False
            and self.released is False
            and now - self.created_at >= self.warn_after_s
        )

    def mark_warning_emitted(self) -> None:
        """Consume the one-shot age warning."""
        self.warning_emitted = True

    def fail_deadline_expired(self, now: float) -> bool:
        """Return whether live native writers exceeded the fail-stop deadline.

        :param now: Current monotonic time.
        :returns: Whether the decoder must fail closed.
        """
        return (
            self.released is False
            and self.ready_to_scatter is False
            and self.native_quiescent is False
            and now - self.created_at >= self.fail_after_s
        )

    @property
    def ready_to_scatter(self) -> bool:
        """Return whether every transfer succeeded and reached DONE.

        :returns: Whether scatter may begin.
        """
        return (
            self.posting_sealed
            and self.operation_failed is False
            and all(slot.state is HandleState.DONE for slot in self.slots.values())
        )

    @property
    def native_quiescent(self) -> bool:
        """Return whether no native actor can write this generation.

        :returns: Whether native quiescence is proved.
        """
        return self.posting_sealed and all(
            slot.state in _NATIVE_QUIESCENT_STATES for slot in self.slots.values()
        )

    @property
    def reusable(self) -> bool:
        """Return whether both native writers and device readers are quiescent.

        :returns: Whether the allocator may reclaim the lease.
        """
        return (
            self.permanently_tombstoned is False
            and self.native_quiescent
            and self.device_quiescent
        )

    def describe(self, now: float | None = None) -> str:
        """Return a stable diagnostic description of ownership.

        :param now: Current monotonic time.
        :returns: Plan identity, age, range, and handle states.
        """
        observed_at = time.monotonic() if now is None else now
        states = ",".join(
            f"{rank}:{slot.state.value}" for rank, slot in sorted(self.slots.items())
        )
        return (
            f"request={self.request_id} owner={self.lease.owner_id} "
            f"remote_engine={self.remote_engine_id} "
            f"generation={self.lease.generation} offset={self.lease.offset} "
            f"bytes={self.lease.size} age_s={observed_at - self.created_at:.6f} "
            f"states={states}"
        )

    def snapshot(self, now: float | None = None) -> StagingOwnershipSnapshot:
        """Capture typed ownership evidence without parsing diagnostics.

        :param now: Current monotonic time.
        :returns: Immutable safety snapshot for artifact serialization.
        """
        observed_at = time.monotonic() if now is None else now
        return StagingOwnershipSnapshot(
            generation=self.lease.generation,
            owner_id=self.lease.owner_id,
            request_id=self.request_id,
            remote_engine_id=self.remote_engine_id,
            offset=self.lease.offset,
            size=self.lease.size,
            age_s=observed_at - self.created_at,
            handles=tuple(
                StagingHandleSnapshot(
                    source_rank=rank,
                    state=slot.state,
                    native_handle_present=slot.native_handle is not None,
                    native_released=slot.native_released,
                )
                for rank, slot in sorted(self.slots.items())
            ),
            posting_sealed=self.posting_sealed,
            operation_failed=self.operation_failed,
            permanently_tombstoned=self.permanently_tombstoned,
            native_quiescent=self.native_quiescent,
            device_read_started=self.device_read_started,
            device_quiescent=self.device_quiescent,
            reusable=self.reusable,
            released=self.released,
            failure_reason=self.failure_reason,
        )

    def _slot(self, source_rank: int) -> StagingHandleSlot:
        slot = self.slots.get(source_rank)
        if slot is None:
            raise ValueError(f"source rank {source_rank} is not owned by this plan")
        return slot

    def _require_posting_open(self) -> None:
        self._require_live()
        if self.posting_sealed:
            raise RuntimeError("cannot enter native code after sealing posting")

    def _require_live(self) -> None:
        if self.released:
            raise RuntimeError("staging plan was already released")

    @staticmethod
    def _require_state(slot: StagingHandleSlot, expected: HandleState) -> None:
        if slot.state is not expected:
            raise RuntimeError(
                f"rank {slot.source_rank} state is {slot.state}, expected {expected}"
            )


@dataclass
class StagingRangeAllocator:
    """Allocate registered staging ranges without reusing live generations.

    :ivar capacity: Total registered byte capacity.
    :ivar active: Live plans keyed by allocation generation.
    """

    capacity: int
    active: dict[int, CoalescedStagingPlan] = field(default_factory=dict, init=False)
    _free: list[tuple[int, int]] = field(default_factory=list, init=False)
    _next_generation: int = field(default=0, init=False)
    _retired_generations: set[int] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("staging allocator capacity must be positive")
        if len(self._free) == 0:
            self._free = [(0, self.capacity)]

    def create_plan(
        self,
        *,
        owner_id: str,
        request_id: str,
        size: int,
        source_ranks: tuple[int, ...],
        remote_engine_id: str | None = None,
        scatter: dict[str, Any] | None = None,
        offset: int | None = None,
        generation: int | None = None,
        created_at: float | None = None,
        warn_after_s: float = 1.0,
        fail_after_s: float = 10.0,
    ) -> CoalescedStagingPlan | None:
        """Atomically reserve a range and install its typed owner.

        :param owner_id: Unique logical plan identity.
        :param request_id: Decoder request identifier.
        :param size: Required byte count.
        :param source_ranks: Complete independently posted rank set.
        :param remote_engine_id: Remote engine owning the source registration.
        :param scatter: Transfer geometry retained for the plan lifetime.
        :param offset: Exact offset for deterministic rigs, or first fit.
        :param generation: Exact generation for deterministic rigs, or monotonic.
        :param created_at: Monotonic plan creation time.
        :param warn_after_s: Structured-warning threshold.
        :param fail_after_s: Fail-stop threshold.
        :returns: Installed owner, or ``None`` when no free interval fits.
        """
        if len(owner_id) == 0:
            raise ValueError("owner_id must not be empty")
        if size <= 0 or size > self.capacity:
            raise ValueError("staging plan size lies outside allocator capacity")
        if any(plan.lease.owner_id == owner_id for plan in self.active.values()):
            raise RuntimeError(f"staging owner {owner_id!r} is already active")

        interval_index = self._find_interval(size, offset)
        if interval_index is None:
            return None
        interval_offset, interval_size = self._free[interval_index]
        chosen_offset = interval_offset if offset is None else offset
        if generation is None:
            generation = self._next_generation
        else:
            if generation < 0:
                raise ValueError("generation must be non-negative")
        if generation in self.active or generation in self._retired_generations:
            raise RuntimeError(f"staging generation {generation} is not fresh")

        lease = StagingLease(generation, owner_id, chosen_offset, size)
        plan = CoalescedStagingPlan.create(
            lease=lease,
            request_id=request_id,
            remote_engine_id=remote_engine_id,
            source_ranks=source_ranks,
            scatter=scatter,
            created_at=created_at,
            warn_after_s=warn_after_s,
            fail_after_s=fail_after_s,
        )

        before = chosen_offset - interval_offset
        after_offset = chosen_offset + size
        after = interval_offset + interval_size - after_offset
        replacement: list[tuple[int, int]] = []
        if before > 0:
            replacement.append((interval_offset, before))
        if after > 0:
            replacement.append((after_offset, after))
        self._free[interval_index : interval_index + 1] = replacement
        self._next_generation = max(self._next_generation, generation + 1)
        self.active[generation] = plan
        return plan

    def release(self, plan: CoalescedStagingPlan) -> None:
        """Reclaim exactly one proved-quiescent generation.

        :param plan: Active owner whose range should be reclaimed.
        """
        active = self.active.get(plan.lease.generation)
        if active is not plan:
            raise StagingSafetyError("staging release owner or generation mismatch")
        if plan.reusable is False:
            raise StagingSafetyError(
                f"staging range is not quiescent: {plan.describe()}"
            )
        plan.released = True
        del self.active[plan.lease.generation]
        self._retired_generations.add(plan.lease.generation)
        self._free.append((plan.lease.offset, plan.lease.size))
        self._merge_free()

    def require_active(self, generation: int) -> CoalescedStagingPlan:
        """Resolve a completion only against its live generation.

        :param generation: Allocation generation from the completion owner.
        :returns: Exact live plan.
        """
        plan = self.active.get(generation)
        if plan is None:
            raise StagingSafetyError(
                f"late completion for staging generation {generation}"
            )
        return plan

    @property
    def free_bytes(self) -> int:
        """Return currently allocatable bytes.

        :returns: Sum of free intervals.
        """
        return sum(size for _, size in self._free)

    def _find_interval(self, size: int, offset: int | None) -> int | None:
        if offset is not None and offset < 0:
            raise ValueError("staging offset must be non-negative")
        for index, (interval_offset, interval_size) in enumerate(self._free):
            if offset is None and interval_size >= size:
                return index
            if (
                offset is not None
                and interval_offset <= offset
                and offset + size <= interval_offset + interval_size
            ):
                return index
        return None

    def _merge_free(self) -> None:
        self._free.sort()
        merged: list[tuple[int, int]] = []
        for offset, size in self._free:
            if len(merged) > 0 and merged[-1][0] + merged[-1][1] == offset:
                prior_offset, prior_size = merged[-1]
                merged[-1] = (prior_offset, prior_size + size)
            else:
                merged.append((offset, size))
        self._free = merged
