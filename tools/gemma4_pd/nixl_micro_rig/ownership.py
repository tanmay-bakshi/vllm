"""Fail-stop staging ownership model for transport fault matrices."""

from dataclasses import dataclass, field
from enum import StrEnum


class HandleState(StrEnum):
    """Safety-relevant state of one independently posted rank handle."""

    UNPOSTED = "unposted"
    PROC = "proc"
    DONE = "done"
    ERR = "err"
    UNKNOWN = "unknown"
    CANCEL_ACK = "cancel_ack"


_QUIESCENT_STATES = {
    HandleState.UNPOSTED,
    HandleState.DONE,
    HandleState.CANCEL_ACK,
}


@dataclass
class StagingGeneration:
    """Track every possible writer before a staging range can be recycled.

    :ivar generation: Allocation generation attached to the staging range.
    :ivar states: Current state for every producer-rank handle.
    :ivar operation_failed: Whether any rank has failed the logical transfer.
    :ivar posting_sealed: Whether native submission is irrevocably closed.
    :ivar released: Whether ownership was irreversibly returned to the allocator.
    """

    generation: int
    states: dict[int, HandleState]
    operation_failed: bool = False
    posting_sealed: bool = False
    released: bool = False
    _posted_ranks: set[int] = field(default_factory=set)
    _native_handle_tokens: dict[int, str] = field(default_factory=dict)

    @classmethod
    def create(cls, generation: int, rank_count: int) -> "StagingGeneration":
        """Create a generation before any native calls are made.

        :param generation: Monotonic allocation generation.
        :param rank_count: Number of independently posted rank handles.
        :returns: New ownership record.
        """
        if generation < 0:
            raise ValueError("generation must be non-negative")
        if rank_count <= 0:
            raise ValueError("rank_count must be positive")
        return cls(
            generation=generation,
            states={rank: HandleState.UNPOSTED for rank in range(rank_count)},
        )

    def record_posted(
        self, rank: int, initial_state: HandleState, native_handle_token: str
    ) -> None:
        """Record a native handle before it can become an untracked writer.

        :param rank: Producer-rank handle identity.
        :param initial_state: Immediate post result, normally PROC or DONE.
        :param native_handle_token: Stable identity of the attached native handle.
        :raises RuntimeError: If ownership was already released or rank reposts.
        """
        self._require_live()
        self._require_rank(rank)
        if self.posting_sealed:
            raise RuntimeError("cannot post a handle after sealing submission")
        if rank in self._posted_ranks:
            raise RuntimeError(f"rank {rank} was already posted")
        if initial_state not in {
            HandleState.PROC,
            HandleState.DONE,
            HandleState.ERR,
            HandleState.UNKNOWN,
        }:
            raise ValueError(f"invalid initial post state: {initial_state}")
        if len(native_handle_token) == 0:
            raise ValueError("native_handle_token must not be empty")
        self._posted_ranks.add(rank)
        self._native_handle_tokens[rank] = native_handle_token
        self.states[rank] = initial_state
        if initial_state in {HandleState.ERR, HandleState.UNKNOWN}:
            self.operation_failed = True

    def record_post_exception(self, rank: int, submission_token: str) -> None:
        """Quarantine a rank whose native submission outcome is unknown.

        :param rank: Producer-rank handle identity.
        :param submission_token: Identity of the native call with unknown outcome.
        """
        self.record_posted(rank, HandleState.UNKNOWN, submission_token)

    def update(self, rank: int, state: HandleState) -> None:
        """Advance a posted handle without manufacturing quiescence.

        :param rank: Producer-rank handle identity.
        :param state: Newly observed state.
        :raises RuntimeError: If the transition is unsafe or ownership is released.
        """
        self._require_live()
        self._require_rank(rank)
        if rank not in self._posted_ranks:
            raise RuntimeError(f"rank {rank} has no posted handle")
        current = self.states[rank]
        if current in _QUIESCENT_STATES and state != current:
            raise RuntimeError(
                f"terminal rank {rank} changed from {current} to {state}"
            )
        if current is HandleState.UNKNOWN and state not in {
            HandleState.UNKNOWN,
            HandleState.DONE,
            HandleState.ERR,
            HandleState.CANCEL_ACK,
        }:
            raise RuntimeError("UNKNOWN can clear only through terminal or cancel ack")
        if current is HandleState.PROC and state not in {
            HandleState.PROC,
            HandleState.DONE,
            HandleState.ERR,
            HandleState.CANCEL_ACK,
        }:
            raise RuntimeError(f"invalid PROC transition to {state}")
        if current is HandleState.ERR and state not in {
            HandleState.ERR,
            HandleState.CANCEL_ACK,
        }:
            raise RuntimeError(
                "ERR requires explicit cancellation/release acknowledgement"
            )
        self.states[rank] = state
        if state is HandleState.ERR:
            self.operation_failed = True

    def seal_posting(self) -> None:
        """Irrevocably close native submission for this generation.

        This boundary is mandatory after either a complete post loop or the
        first failure that aborts the remaining posts. UNPOSTED ranks become
        safe only because no later code can submit them after this transition.
        """
        self._require_live()
        if self.posting_sealed:
            raise RuntimeError("staging generation posting was already sealed")
        self.posting_sealed = True

    @property
    def reusable(self) -> bool:
        """Return whether no native actor can still write the range.

        :returns: ``True`` only after every possible handle is quiescent.
        """
        return self.posting_sealed and all(
            state in _QUIESCENT_STATES for state in self.states.values()
        )

    def release(self) -> None:
        """Return ownership only after every possible writer is quiescent.

        :raises RuntimeError: If any handle remains PROC or UNKNOWN.
        """
        self._require_live()
        if not self.reusable:
            unsafe = {
                rank: state
                for rank, state in self.states.items()
                if state not in _QUIESCENT_STATES
            }
            raise RuntimeError(f"staging generation still has writers: {unsafe}")
        self.released = True

    def _require_live(self) -> None:
        if self.released:
            raise RuntimeError("staging generation was already released")

    def _require_rank(self, rank: int) -> None:
        if rank not in self.states:
            raise ValueError(f"rank {rank} is outside the staging generation")


def production_drop_plan_would_release(states: dict[int, HandleState]) -> bool:
    """Model the current coalesced failure path's immediate range drop.

    Current production calls ``_coalesce_drop_plan`` as soon as one rank
    reports ERR. It does not first prove sibling handles terminal. This adapter
    exists to keep the expected failure executable until production ownership
    is corrected.

    :param states: Rank-handle states at the first observed failure.
    :returns: Whether current production returns the staging range.
    """
    return any(state is HandleState.ERR for state in states.values())


@dataclass
class StagingRangeAllocator:
    """Bind ownership generations to non-overlapping staging byte ranges.

    :ivar capacity: Registered staging bytes managed by the allocator.
    :ivar active: Active generations keyed by allocation generation.
    """

    capacity: int
    active: dict[int, tuple[int, int, StagingGeneration]] = field(default_factory=dict)
    _released_generations: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("staging allocator capacity must be positive")

    def reserve(
        self,
        *,
        generation: int,
        offset: int,
        size: int,
        rank_count: int,
    ) -> StagingGeneration:
        """Reserve one exact range and create its ownership generation.

        :param generation: Monotonic generation identity.
        :param offset: Registration-relative byte offset.
        :param size: Reserved bytes.
        :param rank_count: Independently posted rank handles.
        :returns: New ownership generation.
        :raises RuntimeError: If the generation or range is already live.
        """
        if generation in self.active or generation in self._released_generations:
            raise RuntimeError(f"staging generation {generation} is not fresh")
        if offset < 0 or size <= 0 or offset + size > self.capacity:
            raise ValueError("staging reservation lies outside allocator capacity")
        for active_offset, active_size, _ in self.active.values():
            disjoint = (
                offset + size <= active_offset or active_offset + active_size <= offset
            )
            if not disjoint:
                raise RuntimeError("staging reservation overlaps a live generation")
        ownership = StagingGeneration.create(generation, rank_count)
        self.active[generation] = (offset, size, ownership)
        return ownership

    def release(self, generation: int) -> None:
        """Return a range after its ownership proof succeeds.

        :param generation: Active generation identity.
        """
        record = self.active.get(generation)
        if record is None:
            raise RuntimeError(f"staging generation {generation} is not active")
        ownership = record[2]
        ownership.release()
        del self.active[generation]
        self._released_generations.add(generation)

    def require_active_writer(self, generation: int, rank: int) -> None:
        """Reject late writes from released or superseded generations.

        :param generation: Generation attached to the native completion.
        :param rank: Rank whose completion is attempting to write or scatter.
        :raises RuntimeError: If the generation/rank is no longer active.
        """
        record = self.active.get(generation)
        if record is None:
            raise RuntimeError(f"late completion for staging generation {generation}")
        ownership = record[2]
        ownership._require_rank(rank)
