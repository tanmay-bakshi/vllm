"""Fixed-byte planning and buffer operations for the Target 2 transport gate."""

import enum
import re
import statistics
from dataclasses import dataclass, replace
from pathlib import Path

import torch

from tools.gemma4_pd.nixl_micro_rig.config import RigConfig, ScenarioConfig
from tools.gemma4_pd.nixl_micro_rig.data import expected_plane_bytes
from tools.gemma4_pd.nixl_micro_rig.geometry import TransferPlan, build_plan
from vllm.distributed.kv_transfer.coalesced_layout import (
    CanonicalSourceTransferPlan,
    PackedTransferChunk,
    PackedTransferPlan,
    SourceGroupTransferRoster,
    SourceRegionOwnership,
    build_canonical_source_plan,
    build_packed_transfer_plan,
)

_MIB = 1024 * 1024
_EXACT_2K_BYTES = 1_052_508_160
_EXACT_2K_POSITIONS = 803
_PACK_SLOT_COUNT = 2
_SELECTED_WRITE_CHUNK_BYTES = 256 * _MIB
_CONFORMANCE_DECODER_COUNT = 2
_CONFORMANCE_TAIL_BYTES_PER_RANK = 2 * _SELECTED_WRITE_CHUNK_BYTES + 64 * 1024
_CUDA_WRITE_HEADER = re.compile(
    r"remote memory write by ucp_put\*\(multi\).*"
    r"from cuda/GPU\d+ to cuda/(?:GPU\d+|dev\[\d+\])"
)


class GateConfigurationError(ValueError):
    """Report a Target 2 gate configuration that cannot be compared safely."""


class GateArm(enum.StrEnum):
    """Select one fixed-byte transport implementation."""

    DIRECT_READ = "direct_read"
    PACKED_READ = "packed_read"
    PACKED_WRITE = "packed_write"


class SlotState(enum.StrEnum):
    """Describe one bounded buffer slot's ownership state."""

    FREE = "free"
    DEVICE_WRITER = "device_writer"
    NATIVE_TRANSFER = "native_transfer"
    DEVICE_READER = "device_reader"
    TOMBSTONED = "tombstoned"


@dataclass(frozen=True, slots=True)
class FragmentationRegime:
    """Describe one calibrated exact-2K source-fragmentation regime.

    :ivar name: Stable evidence identity.
    :ivar run_count: Source runs distributed across the logical roster.
    :ivar expected_descriptors_per_rank: Expected owner-aware direct descriptors
        on each producer-rank handle.
    """

    name: str
    run_count: int
    expected_descriptors_per_rank: int

    def __post_init__(self) -> None:
        if len(self.name) == 0:
            raise GateConfigurationError("fragmentation name must not be empty")
        if self.run_count <= 0:
            raise GateConfigurationError("fragmentation run_count must be positive")
        if self.expected_descriptors_per_rank <= 0:
            raise GateConfigurationError(
                "expected_descriptors_per_rank must be positive"
            )


CONTIGUOUS_REGIME = FragmentationRegime(
    name="contiguous_control",
    run_count=1,
    expected_descriptors_per_rank=10,
)
C1_OBSERVED_REGIME = FragmentationRegime(
    name="observed_c1_fragmentation",
    run_count=124,
    expected_descriptors_per_rank=625,
)
C64_OBSERVED_REGIME = FragmentationRegime(
    name="observed_c64_fragmentation",
    run_count=424,
    expected_descriptors_per_rank=2120,
)
DEFAULT_FRAGMENTATION_REGIMES = (
    CONTIGUOUS_REGIME,
    C1_OBSERVED_REGIME,
    C64_OBSERVED_REGIME,
)
DEFAULT_CHUNK_MIB = (64, 128, 256, 512)
DEFAULT_IN_FLIGHT_DEPTHS = (1, 2, 4, 8)


@dataclass(frozen=True, slots=True)
class GateCase:
    """Describe one independently measured fixed-byte gate cell.

    :ivar arm: Direct or producer-packed transfer direction.
    :ivar fragmentation: Exact source descriptor regime.
    :ivar chunk_bytes: Maximum packed chunk bytes, zero for direct transfer.
    :ivar in_flight_depth: Number of logical requests offered together.
    :ivar warmup_batches: Complete unmeasured batches before evidence capture.
    :ivar measured_batches: Complete measured batches.
    """

    arm: GateArm
    fragmentation: FragmentationRegime
    chunk_bytes: int
    in_flight_depth: int
    warmup_batches: int = 1
    measured_batches: int = 3

    def __post_init__(self) -> None:
        if self.arm is GateArm.DIRECT_READ:
            if self.chunk_bytes != 0:
                raise GateConfigurationError(
                    "direct_read must not declare a packed chunk size"
                )
        elif self.chunk_bytes not in tuple(size * _MIB for size in DEFAULT_CHUNK_MIB):
            raise GateConfigurationError(
                "packed chunk_bytes must be 64, 128, 256, or 512 MiB"
            )
        if self.in_flight_depth <= 0:
            raise GateConfigurationError("in_flight_depth must be positive")
        if self.warmup_batches < 0:
            raise GateConfigurationError("warmup_batches must be non-negative")
        if self.measured_batches <= 0:
            raise GateConfigurationError("measured_batches must be positive")

    @property
    def name(self) -> str:
        """Return a stable cell identity.

        :returns: Human-readable arm, fragmentation, chunk, and depth identity.
        """
        chunk = "full" if self.chunk_bytes == 0 else f"{self.chunk_bytes // _MIB}mib"
        return (
            f"{self.arm.value}-{self.fragmentation.name}-{chunk}-"
            f"q{self.in_flight_depth}"
        )


@dataclass(frozen=True, slots=True)
class WriteConformanceCase:
    """Describe one fixed-candidate live conformance case.

    :ivar name: Stable case identity.
    :ivar rank_bytes: Exact bytes produced by each source rank per request.
    :ivar chunk_payload_bytes: Exact populated bytes per rank in every chunk.
    :ivar request_count: Requests executed by the case.
    :ivar decoder_route: Decoder index selected for each request.
    """

    name: str
    rank_bytes: int
    chunk_payload_bytes: tuple[int, ...]
    request_count: int
    decoder_route: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.name) == 0:
            raise GateConfigurationError("conformance case name must not be empty")
        if self.rank_bytes <= 0:
            raise GateConfigurationError("conformance rank bytes must be positive")
        if len(self.chunk_payload_bytes) == 0 or any(
            chunk_bytes <= 0
            or chunk_bytes > _SELECTED_WRITE_CHUNK_BYTES
            or chunk_bytes % 256 != 0
            for chunk_bytes in self.chunk_payload_bytes
        ):
            raise GateConfigurationError(
                "conformance chunks must be non-empty, bounded, and 256-byte aligned"
            )
        if sum(self.chunk_payload_bytes) != self.rank_bytes:
            raise GateConfigurationError(
                "conformance chunks do not conserve exact rank bytes"
            )
        if self.request_count <= _PACK_SLOT_COUNT:
            raise GateConfigurationError(
                "conformance must force reuse of both bounded slots"
            )
        if len(self.decoder_route) != self.request_count:
            raise GateConfigurationError(
                "conformance decoder route does not cover every request"
            )
        if set(self.decoder_route) != set(range(_CONFORMANCE_DECODER_COUNT)):
            raise GateConfigurationError(
                "conformance must route requests to both decoder ranks"
            )


def _write_conformance_schedule(
    case: WriteConformanceCase,
) -> list[dict[str, int]]:
    """Build deterministic two-pool reuse expectations for one case.

    :param case: Fixed WRITE conformance case.
    :returns: Ordered task records with producer and decoder-local slot visits.
    """
    producer_visits = [0] * _PACK_SLOT_COUNT
    consumer_visits = [
        [0] * _PACK_SLOT_COUNT for _ in range(_CONFORMANCE_DECODER_COUNT)
    ]
    consumer_task_counts = [0] * _CONFORMANCE_DECODER_COUNT
    tasks: list[dict[str, int]] = []
    task_index = 0
    for request_index, decoder_index in enumerate(case.decoder_route):
        for chunk_index, payload_bytes in enumerate(case.chunk_payload_bytes):
            producer_slot_index = task_index % _PACK_SLOT_COUNT
            producer_visits[producer_slot_index] += 1
            consumer_slot_index = consumer_task_counts[decoder_index] % _PACK_SLOT_COUNT
            consumer_task_counts[decoder_index] += 1
            consumer_visits[decoder_index][consumer_slot_index] += 1
            tasks.append(
                {
                    "task_index": task_index,
                    "request_index": request_index,
                    "decoder_index": decoder_index,
                    "chunk_index": chunk_index,
                    "payload_bytes_per_rank": payload_bytes,
                    "transfer_bytes": payload_bytes * 4,
                    "descriptor_count": 1,
                    "producer_slot_index": producer_slot_index,
                    "producer_slot_visit": producer_visits[producer_slot_index],
                    "consumer_slot_index": consumer_slot_index,
                    "consumer_slot_visit": consumer_visits[decoder_index][
                        consumer_slot_index
                    ],
                }
            )
            task_index += 1
    return tasks


def focused_write_conformance_plan(
    config: RigConfig,
    base_scenario: ScenarioConfig,
) -> dict[str, object]:
    """Build the post-selection Target 2 live conformance contract.

    This plan deliberately contains no direct or packed-READ arm. The fixed-byte
    gate already selected producer-initiated WRITE with a 256 MiB per-rank
    stride. Conformance therefore exercises only that immutable candidate.

    :param config: Exact Gemma 4 TP4-to-TP1 rig configuration.
    :param base_scenario: Scenario supplying the exact 2K model geometry.
    :returns: Machine-readable multi-decoder and multi-chunk acceptance plan.
    """
    exact_plan = build_gate_plan(config, base_scenario, C64_OBSERVED_REGIME)
    exact_packed = build_gate_packed_plan(
        exact_plan,
        _SELECTED_WRITE_CHUNK_BYTES,
    )
    exact_chunks = tuple(chunk.rank_stride_bytes for chunk in exact_packed.chunks)
    tail_row_bytes = 64 * 1024
    tail_position_count = _CONFORMANCE_TAIL_BYTES_PER_RANK // tail_row_bytes
    tail_source_plan = build_canonical_source_plan(
        source_tp_size=4,
        source_ranks=(0, 1, 2, 3),
        groups=(
            SourceGroupTransferRoster(
                group_index=0,
                source_position_start=0,
                remote_block_ids=tuple(range(tail_position_count)),
            ),
        ),
        regions=(
            SourceRegionOwnership(
                region_index=0,
                group_indices=(0,),
                source_row_count=tail_position_count,
                row_bytes=tail_row_bytes,
            ),
        ),
    )
    tail_packed = build_packed_transfer_plan(
        tail_source_plan,
        max_chunk_bytes_per_rank=_SELECTED_WRITE_CHUNK_BYTES,
    )
    tail_chunks = tuple(chunk.rank_stride_bytes for chunk in tail_packed.chunks)
    expected_tail_chunks = (
        _SELECTED_WRITE_CHUNK_BYTES,
        _SELECTED_WRITE_CHUNK_BYTES,
        tail_row_bytes,
    )
    if tail_chunks != expected_tail_chunks:
        raise GateConfigurationError(
            "production packed planner did not preserve the conformance tail shape"
        )
    cases = (
        WriteConformanceCase(
            name="exact2k-c64-two-decoder-routing",
            rank_bytes=exact_plan.transport.rank_stride_bytes,
            chunk_payload_bytes=exact_chunks,
            request_count=6,
            decoder_route=(0, 1, 0, 1, 0, 1),
        ),
        WriteConformanceCase(
            name="three-chunk-full-full-tail",
            rank_bytes=_CONFORMANCE_TAIL_BYTES_PER_RANK,
            chunk_payload_bytes=tail_chunks,
            request_count=3,
            decoder_route=(0, 1, 0),
        ),
    )
    case_records: list[dict[str, object]] = []
    for case in cases:
        schedule = _write_conformance_schedule(case)
        case_records.append(
            {
                "name": case.name,
                "rank_bytes": case.rank_bytes,
                "request_bytes": case.rank_bytes * 4,
                "chunk_payload_bytes_per_rank": list(case.chunk_payload_bytes),
                "chunk_count": len(case.chunk_payload_bytes),
                "request_count": case.request_count,
                "decoder_route": list(case.decoder_route),
                "tasks": schedule,
                "producer_slot_reuse_proved": any(
                    task["producer_slot_visit"] > 1 for task in schedule
                ),
                "consumer_slot_reuse_proved": any(
                    task["consumer_slot_visit"] > 1 for task in schedule
                ),
            }
        )
    return {
        "schema_version": 1,
        "evidence_scope": "target2_selected_packed_write_live_conformance",
        "selection_rerun": False,
        "selected_candidate": {
            "arm": GateArm.PACKED_WRITE.value,
            "chunk_bytes_per_rank": _SELECTED_WRITE_CHUNK_BYTES,
            "source_tp_size": 4,
            "decoder_count": _CONFORMANCE_DECODER_COUNT,
            "producer_slot_count": _PACK_SLOT_COUNT,
            "consumer_slot_count": _PACK_SLOT_COUNT,
            "alignment_bytes": 256,
        },
        "required_terminal_evidence": {
            "exact_byte_oracle": True,
            "one_descriptor_per_rank_chunk": True,
            "native_sender_authenticated_completion": True,
            "cuda_to_cuda_zero_copy_protocol": "cuda_ipc/cuda",
            "final_slot_states": "all_free",
            "source_preservation": True,
            "protected_gpu_baseline_unchanged": True,
        },
        "cases": case_records,
    }


def cuda_ipc_write_protocol_observations(
    log_text: str,
) -> tuple[dict[str, object], ...]:
    """Extract UCX CUDA-to-CUDA zero-copy WRITE protocol selections.

    :param log_text: One producer's complete native stdout/stderr log.
    :returns: Exact header and protocol rows with one-based line numbers.
    """
    lines = log_text.splitlines()
    observations: list[dict[str, object]] = []
    for header_index, header in enumerate(lines):
        if _CUDA_WRITE_HEADER.search(header) is None:
            continue
        for protocol_index in range(
            header_index + 1,
            min(len(lines), header_index + 12),
        ):
            protocol = lines[protocol_index]
            if (
                "0..inf" not in protocol
                or "zero-copy" not in protocol
                or "cuda_ipc/cuda" not in protocol
            ):
                continue
            observations.append(
                {
                    "header_line_number": header_index + 1,
                    "header": header.strip(),
                    "protocol_line_number": protocol_index + 1,
                    "protocol": protocol.strip(),
                }
            )
            break
    return tuple(observations)


def attest_cuda_ipc_write_protocol(
    artifact_directory: Path,
    *,
    producer_count: int,
) -> dict[str, object]:
    """Require every producer to select CUDA-IPC zero-copy for packed WRITE.

    :param artifact_directory: Completed transport-arm artifact directory.
    :param producer_count: Exact producer-rank cardinality.
    :returns: Per-rank raw UCX protocol evidence.
    :raises RuntimeError: If any producer lacks authoritative protocol output.
    """
    if producer_count <= 0:
        raise ValueError("producer_count must be positive")
    producer_records: list[dict[str, object]] = []
    for rank in range(producer_count):
        path = artifact_directory / f"producer-{rank}.log"
        try:
            log_text = path.read_text(errors="replace")
        except OSError as error:
            raise RuntimeError(
                f"Target 2 producer {rank} UCX protocol log is unavailable"
            ) from error
        observations = cuda_ipc_write_protocol_observations(log_text)
        if len(observations) == 0:
            raise RuntimeError(
                "Target 2 producer "
                f"{rank} did not prove CUDA-to-CUDA zero-copy cuda_ipc/cuda WRITE"
            )
        producer_records.append(
            {
                "rank": rank,
                "log": path.name,
                "observations": list(observations),
            }
        )
    return {
        "schema_version": 1,
        "evidence_scope": "target2_ucx_cuda_ipc_write_protocol",
        "operation": "remote_memory_write",
        "source_memory": "cuda",
        "destination_memory": "cuda",
        "protocol": "cuda_ipc/cuda",
        "selection": "zero-copy",
        "producer_count": producer_count,
        "producers": producer_records,
    }


@dataclass(slots=True)
class BufferSlotLease:
    """Fail-closed lifetime state for one reusable packed buffer slot.

    :ivar slot_index: Stable slot identity.
    :ivar state: Current exclusive accessor.
    :ivar generation: Monotonic slot generation.
    :ivar request_index: Current logical request, absent while free.
    :ivar chunk_index: Current logical chunk, absent while free.
    :ivar failure: Terminal failure evidence after tombstoning.
    """

    slot_index: int
    state: SlotState = SlotState.FREE
    generation: int = 0
    request_index: int | None = None
    chunk_index: int | None = None
    failure: str | None = None

    def begin_device_write(self, *, request_index: int, chunk_index: int) -> int:
        """Acquire a free slot for producer pack or native receive.

        :param request_index: Logical request occupying the slot.
        :param chunk_index: Logical chunk occupying the slot.
        :returns: New slot generation.
        :raises RuntimeError: If a prior accessor or failure still owns the slot.
        """
        self._require_state(SlotState.FREE)
        if request_index < 0 or chunk_index < 0:
            raise ValueError("request and chunk indices must be non-negative")
        self.generation += 1
        self.request_index = request_index
        self.chunk_index = chunk_index
        self.state = SlotState.DEVICE_WRITER
        return self.generation

    def begin_native_transfer(self) -> None:
        """Hand a device-quiescent buffer to NIXL."""
        self._require_state(SlotState.DEVICE_WRITER)
        self.state = SlotState.NATIVE_TRANSFER

    def begin_native_write(self, *, request_index: int, chunk_index: int) -> int:
        """Acquire a free receive slot for a native write.

        :param request_index: Logical request occupying the slot.
        :param chunk_index: Logical chunk occupying the slot.
        :returns: New slot generation.
        :raises RuntimeError: If a prior accessor or failure still owns the slot.
        """
        self._require_state(SlotState.FREE)
        if request_index < 0 or chunk_index < 0:
            raise ValueError("request and chunk indices must be non-negative")
        self.generation += 1
        self.request_index = request_index
        self.chunk_index = chunk_index
        self.state = SlotState.NATIVE_TRANSFER
        return self.generation

    def begin_device_read(self) -> None:
        """Hand a natively quiescent received buffer to verification/scatter."""
        self._require_state(SlotState.NATIVE_TRANSFER)
        self.state = SlotState.DEVICE_READER

    def release_after_native(self) -> None:
        """Release a producer source after native completion is proved."""
        self._require_state(SlotState.NATIVE_TRANSFER)
        self._release()

    def release_after_device(self) -> None:
        """Release a receive buffer after its device reader is quiescent."""
        self._require_state(SlotState.DEVICE_READER)
        self._release()

    def tombstone(self, reason: str) -> None:
        """Permanently prevent reuse after an ambiguous operation.

        :param reason: Non-empty failure evidence.
        """
        if len(reason) == 0:
            raise ValueError("tombstone reason must not be empty")
        self.state = SlotState.TOMBSTONED
        self.failure = reason

    def _release(self) -> None:
        self.state = SlotState.FREE
        self.request_index = None
        self.chunk_index = None

    def _require_state(self, expected: SlotState) -> None:
        if self.state is not expected:
            raise RuntimeError(
                f"slot {self.slot_index} is {self.state.value}, expected "
                f"{expected.value}"
            )


def build_gate_plan(
    config: RigConfig,
    base_scenario: ScenarioConfig,
    regime: FragmentationRegime,
    *,
    source_start_block: int = 1024,
) -> TransferPlan:
    """Build and calibrate one exact-2K fragmented request.

    :param config: Exact Gemma 4 TP4-to-TP1 rig configuration.
    :param base_scenario: Scenario supplying iteration and staging policy.
    :param regime: Required descriptor-fragmentation regime.
    :param source_start_block: First source block in the synthetic span.
    :returns: Calibrated transfer plan.
    :raises GateConfigurationError: If geometry differs from the fixed-byte gate.
    """
    if config.valid_token_extent != 2048:
        raise GateConfigurationError("Target 2 gate requires the exact 2K profile")
    source_end_block = source_start_block + _EXACT_2K_POSITIONS + regime.run_count - 2
    scenario = replace(
        base_scenario,
        name=f"target2-{regime.name}-{source_start_block}",
        source_start_block=source_start_block,
        source_end_block=source_end_block,
        run_count=regime.run_count,
        iterations=1,
        staging_offsets_mib=(0,),
        replay_manifest=None,
    )
    plan = build_plan(config, scenario)
    if plan.logical_position_count != _EXACT_2K_POSITIONS:
        raise GateConfigurationError(
            "Target 2 gate logical position count differs from exact 2K"
        )
    if plan.staging_bytes != _EXACT_2K_BYTES:
        raise GateConfigurationError(
            "Target 2 gate request bytes differ from 1,052,508,160"
        )
    if plan.descriptors_per_handle != regime.expected_descriptors_per_rank:
        raise GateConfigurationError(
            f"{regime.name} produced {plan.descriptors_per_handle} descriptors per "
            f"rank, expected {regime.expected_descriptors_per_rank}"
        )
    return plan


def build_gate_request_plans(
    config: RigConfig,
    base_scenario: ScenarioConfig,
    regime: FragmentationRegime,
    in_flight_depth: int,
) -> tuple[TransferPlan, ...]:
    """Build disjoint source spans for an offered concurrent batch.

    :param config: Exact Gemma 4 TP4-to-TP1 rig configuration.
    :param base_scenario: Scenario supplying invariant model geometry.
    :param regime: Required descriptor-fragmentation regime.
    :param in_flight_depth: Logical request count prepared together.
    :returns: Plans with non-overlapping source rows and identical wire bytes.
    :raises GateConfigurationError: If the configured registration is too small.
    """
    if in_flight_depth <= 0:
        raise GateConfigurationError("in_flight_depth must be positive")
    source_span = _EXACT_2K_POSITIONS + regime.run_count - 1
    stride = source_span + 1
    plans = tuple(
        build_gate_plan(
            config,
            base_scenario,
            regime,
            source_start_block=1024 + request_index * stride,
        )
        for request_index in range(in_flight_depth)
    )
    final_remote_block = max(
        position.remote_block_id
        for region in plans[-1].transport.regions
        for position in region.positions
    )
    if final_remote_block >= config.source_block_count:
        raise GateConfigurationError(
            f"in-flight depth {in_flight_depth} requires source row "
            f"{final_remote_block}, but registration ends at "
            f"{config.source_block_count - 1}"
        )
    source_sets = [
        {
            position.remote_block_id
            for region in plan.transport.regions
            for position in region.positions
        }
        for plan in plans
    ]
    if sum(len(blocks) for blocks in source_sets) != len(set().union(*source_sets)):
        raise AssertionError("gate request plans overlap source rows")
    return plans


def build_gate_source_plan(plan: TransferPlan) -> CanonicalSourceTransferPlan:
    """Reconstruct the production source plan represented by a rig plan.

    :param plan: Exact canonical owner-aware transfer plan.
    :returns: Destination-independent production source plan.
    :raises GateConfigurationError: If its authenticated digest differs.
    """
    transport = plan.transport
    source_plan = build_canonical_source_plan(
        source_tp_size=transport.source_tp_size,
        source_ranks=transport.source_ranks,
        groups=tuple(
            SourceGroupTransferRoster(
                group_index=group.group_index,
                source_position_start=group.source_position_start,
                remote_block_ids=group.remote_block_ids,
            )
            for group in transport.groups
        ),
        regions=tuple(
            SourceRegionOwnership(
                region_index=region.ownership.region_index,
                group_indices=region.ownership.group_indices,
                source_row_count=region.ownership.source_row_count,
                row_bytes=region.ownership.row_bytes,
            )
            for region in transport.regions
        ),
    )
    if source_plan.digest != transport.source_digest:
        raise GateConfigurationError("rig and production source digests differ")
    return source_plan


def build_gate_packed_plan(plan: TransferPlan, chunk_bytes: int) -> PackedTransferPlan:
    """Build the exact production packed plan for one gate request.

    :param plan: Exact canonical owner-aware transfer plan.
    :param chunk_bytes: Maximum bytes in each producer-rank chunk.
    :returns: Production packed transfer plan used by the serving worker.
    """
    return build_packed_transfer_plan(
        build_gate_source_plan(plan),
        max_chunk_bytes_per_rank=chunk_bytes,
    )


def packed_descriptors(
    *,
    local_base: int,
    local_device: int,
    remote_base: int,
    remote_device: int,
    chunk: PackedTransferChunk,
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    """Build the one-descriptor contiguous NIXL pair for a packed chunk.

    :param local_base: Active local slot base address.
    :param local_device: Local NIXL descriptor device.
    :param remote_base: Active remote slot base address.
    :param remote_device: Remote NIXL descriptor device.
    :param chunk: Exact populated chunk extent.
    :returns: One local and one remote raw descriptor.
    """
    if min(local_base, remote_base) <= 0:
        raise ValueError("packed descriptor bases must be positive")
    if min(local_device, remote_device) < 0:
        raise ValueError("packed descriptor devices must be non-negative")
    if chunk.rank_stride_bytes <= 0:
        raise ValueError("packed chunk must not be empty")
    return (
        [(local_base, chunk.rank_stride_bytes, local_device)],
        [(remote_base, chunk.rank_stride_bytes, remote_device)],
    )


def verify_packed_chunk(
    *,
    plan: TransferPlan,
    packed_plan: PackedTransferPlan,
    chunk_index: int,
    source_rank: int,
    iteration: int,
    packed: torch.Tensor,
    packed_base_offset_bytes: int = 0,
    batch_rows: int = 64,
) -> None:
    """Byte-compare one received chunk against the independent payload oracle.

    :param plan: Canonical owner-aware transfer plan.
    :param packed_plan: Exact production packed plan.
    :param chunk_index: Received chunk identity.
    :param source_rank: Producer rank carried by ``packed``.
    :param iteration: Payload generation written by the producer.
    :param packed: Flat CPU or CUDA receive slot.
    :param packed_base_offset_bytes: Rank-local byte offset within ``packed``.
    :param batch_rows: Maximum oracle rows constructed together.
    :raises RuntimeError: At the first exact byte mismatch.
    """
    if source_rank < 0 or source_rank >= plan.rank_count:
        raise ValueError("source_rank is outside the transfer plan")
    if iteration < 0:
        raise ValueError("iteration must be non-negative")
    if batch_rows <= 0:
        raise ValueError("batch_rows must be positive")
    if chunk_index < 0 or chunk_index >= len(packed_plan.chunks):
        raise ValueError("chunk_index is outside the packed plan")
    if packed_base_offset_bytes < 0:
        raise ValueError("packed_base_offset_bytes must be non-negative")
    if packed.dtype is not torch.uint8 or not packed.is_contiguous():
        raise ValueError("packed slot must be contiguous uint8")
    chunk = packed_plan.chunks[chunk_index]
    if packed.numel() < packed_base_offset_bytes + chunk.rank_stride_bytes:
        raise ValueError("packed slot is smaller than the populated chunk")
    for region_slice in chunk.region_slices:
        region = plan.transport.regions[region_slice.region_index]
        row_bytes = region.ownership.row_bytes
        rows = packed[
            packed_base_offset_bytes
            + region_slice.offset_within_rank : packed_base_offset_bytes
            + region_slice.offset_within_rank
            + region_slice.size_bytes
        ].view(region_slice.position_count, 2, row_bytes // 2)
        for position_start in range(0, region_slice.position_count, batch_rows):
            position_count = min(
                batch_rows, region_slice.position_count - position_start
            )
            for plane_index in (0, 1):
                expected = expected_plane_bytes(
                    plan=plan,
                    position_start=region_slice.position_start + position_start,
                    position_count=position_count,
                    source_rank=source_rank,
                    region_index=region_slice.region_index,
                    plane_index=plane_index,
                    iteration=iteration,
                    row_bytes=row_bytes,
                    device=packed.device,
                )
                actual = rows[
                    position_start : position_start + position_count,
                    plane_index,
                ]
                if not torch.equal(actual, expected):
                    raise RuntimeError(
                        "packed oracle mismatch at "
                        f"rank={source_rank} region={region_slice.region_index} "
                        f"plane={plane_index} position="
                        f"{region_slice.position_start + position_start}"
                    )


def gate_matrix(
    *,
    regimes: tuple[FragmentationRegime, ...] = DEFAULT_FRAGMENTATION_REGIMES,
    chunk_mib: tuple[int, ...] = DEFAULT_CHUNK_MIB,
    in_flight_depths: tuple[int, ...] = DEFAULT_IN_FLIGHT_DEPTHS,
    warmup_batches: int = 1,
    measured_batches: int = 3,
) -> tuple[GateCase, ...]:
    """Build the decisive direct, packed-READ, and packed-WRITE sweep.

    :param regimes: Calibrated source-fragmentation regimes.
    :param chunk_mib: Bounded packed slot capacities in MiB.
    :param in_flight_depths: Logical requests offered in each batch.
    :param warmup_batches: Unmeasured batches per fresh process cell.
    :param measured_batches: Evidence batches per fresh process cell.
    :returns: Ordered fixed-byte gate cases.
    """
    cases: list[GateCase] = []
    for regime in regimes:
        for depth in in_flight_depths:
            cases.append(
                GateCase(
                    arm=GateArm.DIRECT_READ,
                    fragmentation=regime,
                    chunk_bytes=0,
                    in_flight_depth=depth,
                    warmup_batches=warmup_batches,
                    measured_batches=measured_batches,
                )
            )
            for size_mib in chunk_mib:
                for arm in (GateArm.PACKED_READ, GateArm.PACKED_WRITE):
                    cases.append(
                        GateCase(
                            arm=arm,
                            fragmentation=regime,
                            chunk_bytes=size_mib * _MIB,
                            in_flight_depth=depth,
                            warmup_batches=warmup_batches,
                            measured_batches=measured_batches,
                        )
                    )
    return tuple(cases)


def describe_gate_case(
    config: RigConfig,
    base_scenario: ScenarioConfig,
    case: GateCase,
) -> dict[str, object]:
    """Render a machine-readable fixed-byte cell plan.

    :param config: Exact 2K rig configuration.
    :param base_scenario: Base exact-2K scenario.
    :param case: Gate cell.
    :returns: Calibrated descriptor, byte, chunk, and memory counts.
    """
    plan = build_gate_plan(config, base_scenario, case.fragmentation)
    packed_layout = (
        None
        if case.arm is GateArm.DIRECT_READ
        else build_gate_packed_plan(plan, case.chunk_bytes)
    )
    return {
        "name": case.name,
        "arm": case.arm.value,
        "fragmentation": case.fragmentation.name,
        "source_run_count": case.fragmentation.run_count,
        "in_flight_depth": case.in_flight_depth,
        "request_bytes": plan.staging_bytes,
        "rank_bytes": plan.transport.rank_stride_bytes,
        "direct_descriptors_per_rank": plan.descriptors_per_handle,
        "direct_descriptors_per_request": (
            plan.descriptors_per_handle * plan.rank_count
        ),
        "chunk_bytes": case.chunk_bytes,
        "chunk_count_per_rank": 0
        if packed_layout is None
        else len(packed_layout.chunks),
        "packed_descriptors_per_request": (
            0 if packed_layout is None else len(packed_layout.chunks) * plan.rank_count
        ),
        "producer_pack_slot_count": 0 if packed_layout is None else _PACK_SLOT_COUNT,
        "producer_pack_bytes_per_rank": (
            0 if packed_layout is None else _PACK_SLOT_COUNT * case.chunk_bytes
        ),
        "consumer_receive_bytes": (
            case.in_flight_depth * plan.staging_bytes
            if packed_layout is None
            else _PACK_SLOT_COUNT * plan.rank_count * case.chunk_bytes
        ),
        "warmup_batches": case.warmup_batches,
        "measured_batches": case.measured_batches,
    }


def summarize_gate_records(
    cases: tuple[GateCase, ...], records: list[dict[str, object]]
) -> dict[str, object]:
    """Validate and summarize one complete Target 2 arm result.

    :param cases: Ordered gate cases supplied to all five roles.
    :param records: Decoded consumer batch evidence.
    :returns: Validated per-cell medians and direct-relative comparisons.
    :raises RuntimeError: If evidence coverage, integrity, or geometry differs.
    """
    expected_record_count = sum(
        case.warmup_batches + case.measured_batches for case in cases
    )
    if len(records) != expected_record_count:
        raise RuntimeError(
            f"Target 2 batch coverage is {len(records)}, "
            f"expected {expected_record_count}"
        )

    records_by_case: dict[int, list[dict[str, object]]] = {
        case_index: [] for case_index in range(len(cases))
    }
    for record in records:
        case_index = _evidence_integer(record, "case_index")
        if case_index not in records_by_case:
            raise RuntimeError(f"Target 2 evidence has unknown case {case_index}")
        records_by_case[case_index].append(record)

    summaries: list[dict[str, object]] = []
    for case_index, case in enumerate(cases):
        case_records = records_by_case[case_index]
        case_records.sort(key=lambda record: _evidence_integer(record, "batch_index"))
        expected_batches = case.warmup_batches + case.measured_batches
        if tuple(
            _evidence_integer(record, "batch_index") for record in case_records
        ) != tuple(range(expected_batches)):
            raise RuntimeError(f"Target 2 case {case.name} batch identities differ")
        measured_elapsed: list[float] = []
        measured_goodput: list[float] = []
        measured_native_critical_ms: list[float] = []
        measured_pack_critical_ms: list[float] = []
        measured_scatter_ms: list[float] = []
        for batch_index, record in enumerate(case_records):
            _validate_gate_record(case, batch_index, record)
            if batch_index < case.warmup_batches:
                continue
            elapsed = _evidence_number(record, "elapsed_seconds")
            total_batch_bytes = _evidence_integer(record, "total_batch_bytes")
            measured_elapsed.append(elapsed)
            measured_goodput.append(total_batch_bytes / (1024**3) / elapsed)
            measured_native_critical_ms.append(_native_critical_path_ms(record, case))
            measured_pack_critical_ms.append(
                _device_critical_path_ms(record, "pack_gpu_ms")
            )
            measured_scatter_ms.append(_device_total_ms(record, "scatter_gpu_ms"))
        summaries.append(
            {
                "case_index": case_index,
                "case_name": case.name,
                "arm": case.arm.value,
                "fragmentation": case.fragmentation.name,
                "source_run_count": case.fragmentation.run_count,
                "direct_descriptors_per_rank": (
                    case.fragmentation.expected_descriptors_per_rank
                ),
                "chunk_bytes": case.chunk_bytes,
                "in_flight_depth": case.in_flight_depth,
                "measured_batches": case.measured_batches,
                "median_elapsed_seconds": statistics.median(measured_elapsed),
                "minimum_elapsed_seconds": min(measured_elapsed),
                "maximum_elapsed_seconds": max(measured_elapsed),
                "median_goodput_gib_s": statistics.median(measured_goodput),
                "median_native_critical_path_ms": statistics.median(
                    measured_native_critical_ms
                ),
                "median_pack_critical_path_ms": statistics.median(
                    measured_pack_critical_ms
                ),
                "median_scatter_total_ms": statistics.median(measured_scatter_ms),
            }
        )

    direct = {
        (summary["fragmentation"], summary["in_flight_depth"]): summary
        for summary in summaries
        if summary["arm"] == GateArm.DIRECT_READ.value
    }
    comparisons: list[dict[str, object]] = []
    for summary in summaries:
        if summary["arm"] == GateArm.DIRECT_READ.value:
            continue
        baseline = direct[(summary["fragmentation"], summary["in_flight_depth"])]
        direct_elapsed = _evidence_number(baseline, "median_elapsed_seconds")
        packed_elapsed = _evidence_number(summary, "median_elapsed_seconds")
        comparisons.append(
            {
                "fragmentation": summary["fragmentation"],
                "in_flight_depth": summary["in_flight_depth"],
                "arm": summary["arm"],
                "chunk_bytes": summary["chunk_bytes"],
                "direct_median_elapsed_seconds": direct_elapsed,
                "packed_median_elapsed_seconds": packed_elapsed,
                "latency_delta_ms": (packed_elapsed - direct_elapsed) * 1000.0,
                "direct_over_packed_speed_ratio": direct_elapsed / packed_elapsed,
                "goodput_improvement_fraction": direct_elapsed / packed_elapsed - 1.0,
            }
        )
    return {
        "schema_version": 1,
        "evidence_scope": "target2_fixed_byte_native_transport_gate",
        "record_count": len(records),
        "case_count": len(cases),
        "integrity_verdict": "exact_warmups_and_measured_wire_final_destination",
        "cells": summaries,
        "direct_relative_comparisons": comparisons,
        "target2_decision": _target2_decision(summaries),
    }


def _target2_decision(summaries: list[dict[str, object]]) -> dict[str, object]:
    """Select a bounded packed candidate from the production activation regime."""
    decision_depths = (1, 2, 4, 8)
    direct = {
        _evidence_integer(summary, "in_flight_depth"): summary
        for summary in summaries
        if _evidence_text(summary, "arm") == GateArm.DIRECT_READ.value
        and _evidence_text(summary, "fragmentation") == C64_OBSERVED_REGIME.name
        and _evidence_integer(summary, "in_flight_depth") in decision_depths
    }
    candidates: dict[tuple[str, int], dict[int, dict[str, object]]] = {}
    for summary in summaries:
        arm = _evidence_text(summary, "arm")
        depth = _evidence_integer(summary, "in_flight_depth")
        if (
            arm == GateArm.DIRECT_READ.value
            or _evidence_text(summary, "fragmentation") != C64_OBSERVED_REGIME.name
            or depth not in decision_depths
        ):
            continue
        key = (arm, _evidence_integer(summary, "chunk_bytes"))
        candidates.setdefault(key, {})[depth] = summary

    expected_candidate_keys = {
        (arm.value, chunk_mib * _MIB)
        for arm in (GateArm.PACKED_READ, GateArm.PACKED_WRITE)
        for chunk_mib in DEFAULT_CHUNK_MIB
    }
    coverage_complete = set(direct) == set(decision_depths) and all(
        set(candidates.get(key, {})) == set(decision_depths)
        for key in expected_candidate_keys
    )
    criteria = {
        "activation_fragmentation": C64_OBSERVED_REGIME.name,
        "activation_descriptors_per_rank": (
            C64_OBSERVED_REGIME.expected_descriptors_per_rank
        ),
        "production_activation_threshold_per_rank": 1024,
        "required_depths": list(decision_depths),
        "minimum_all_depth_speed_ratio": 0.98,
        "minimum_q4_q8_speed_ratio": 1.0,
        "minimum_geometric_mean_speed_ratio": 1.05,
        "near_best_tie_fraction": 0.01,
    }
    if not coverage_complete:
        return {
            "verdict": "not_evaluable_incomplete_matrix",
            "criteria": criteria,
            "candidate_scores": [],
            "selected_candidate": None,
        }

    scores: list[dict[str, object]] = []
    for arm, chunk_bytes in sorted(expected_candidate_keys):
        ratios = {
            depth: _evidence_number(direct[depth], "median_elapsed_seconds")
            / _evidence_number(
                candidates[(arm, chunk_bytes)][depth],
                "median_elapsed_seconds",
            )
            for depth in decision_depths
        }
        geometric_mean = statistics.geometric_mean(ratios.values())
        high_concurrency_minimum = min(ratios[4], ratios[8])
        eligible = (
            min(ratios.values()) >= 0.98
            and high_concurrency_minimum >= 1.0
            and geometric_mean >= 1.05
        )
        scores.append(
            {
                "arm": arm,
                "chunk_bytes": chunk_bytes,
                "speed_ratio_by_depth": {
                    str(depth): ratios[depth] for depth in decision_depths
                },
                "minimum_speed_ratio": min(ratios.values()),
                "minimum_q4_q8_speed_ratio": high_concurrency_minimum,
                "geometric_mean_speed_ratio": geometric_mean,
                "eligible": eligible,
            }
        )

    eligible_scores = [
        score for score in scores if _evidence_boolean(score, "eligible")
    ]
    if len(eligible_scores) == 0:
        return {
            "verdict": "fail_no_non_regressing_candidate",
            "criteria": criteria,
            "candidate_scores": scores,
            "selected_candidate": None,
        }
    best_geometric_mean = max(
        _evidence_number(score, "geometric_mean_speed_ratio")
        for score in eligible_scores
    )
    competitive = [
        score
        for score in eligible_scores
        if _evidence_number(score, "geometric_mean_speed_ratio")
        >= best_geometric_mean * (1.0 - 0.01)
    ]
    competitive.sort(
        key=lambda score: (
            _evidence_integer(score, "chunk_bytes"),
            0 if _evidence_text(score, "arm") == GateArm.PACKED_READ.value else 1,
            -_evidence_number(score, "geometric_mean_speed_ratio"),
        )
    )
    return {
        "verdict": "pass",
        "criteria": criteria,
        "candidate_scores": scores,
        "selected_candidate": competitive[0],
    }


def _evidence_integer(record: dict[str, object], field_name: str) -> int:
    value = record.get(field_name)
    if type(value) is not int:
        raise RuntimeError(f"Target 2 evidence {field_name} must be an integer")
    return value


def _evidence_text(record: dict[str, object], field_name: str) -> str:
    value = record.get(field_name)
    if type(value) is not str or len(value) == 0:
        raise RuntimeError(f"Target 2 evidence {field_name} must be non-empty text")
    return value


def _evidence_boolean(record: dict[str, object], field_name: str) -> bool:
    value = record.get(field_name)
    if type(value) is not bool:
        raise RuntimeError(f"Target 2 evidence {field_name} must be a boolean")
    return value


def _evidence_number(record: dict[str, object], field_name: str) -> float:
    value = record.get(field_name)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise RuntimeError(f"Target 2 evidence {field_name} must be positive")
    return float(value)


def _evidence_nonnegative_number(record: dict[str, object], field_name: str) -> float:
    value = record.get(field_name)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
        raise RuntimeError(f"Target 2 evidence {field_name} must be non-negative")
    return float(value)


def _evidence_list(record: dict[str, object], field_name: str) -> list[object]:
    value = record.get(field_name)
    if not isinstance(value, list):
        raise RuntimeError(f"Target 2 evidence {field_name} must be an array")
    return value


def _validate_gate_record(
    case: GateCase, batch_index: int, record: dict[str, object]
) -> None:
    expected_measured = batch_index >= case.warmup_batches
    rank_bytes = _EXACT_2K_BYTES // 4
    packed_chunk_count = (
        0
        if case.arm is GateArm.DIRECT_READ
        else (rank_bytes + case.chunk_bytes - 1) // case.chunk_bytes
    )
    exact_fields = {
        "case_name": case.name,
        "arm": case.arm.value,
        "fragmentation": case.fragmentation.name,
        "source_run_count": case.fragmentation.run_count,
        "direct_descriptors_per_rank": (
            case.fragmentation.expected_descriptors_per_rank
        ),
        "chunk_bytes": case.chunk_bytes,
        "in_flight_depth": case.in_flight_depth,
        "batch_index": batch_index,
        "measured": expected_measured,
        "request_bytes": _EXACT_2K_BYTES,
        "total_batch_bytes": _EXACT_2K_BYTES * case.in_flight_depth,
        "packed_window_limit": (
            0 if case.arm is GateArm.DIRECT_READ else _PACK_SLOT_COUNT
        ),
        "maximum_active_packed_tasks": (
            0
            if case.arm is GateArm.DIRECT_READ
            else min(
                _PACK_SLOT_COUNT,
                case.in_flight_depth * packed_chunk_count,
            )
        ),
    }
    for field_name, expected in exact_fields.items():
        if record.get(field_name) != expected:
            raise RuntimeError(
                f"Target 2 case {case.name} {field_name} differs: "
                f"observed={record.get(field_name)!r}, expected={expected!r}"
            )
    _evidence_number(record, "elapsed_seconds")
    iterations = _evidence_list(record, "payload_iterations")
    if len(iterations) != case.in_flight_depth or len(set(iterations)) != len(
        iterations
    ):
        raise RuntimeError(f"Target 2 case {case.name} payload identities differ")

    native_records = _evidence_list(record, "native_handles")
    expected_native_count = case.in_flight_depth * 4
    if case.arm is not GateArm.DIRECT_READ:
        chunk_indices = {
            _evidence_integer(native, "chunk_index")
            for native in native_records
            if isinstance(native, dict)
        }
        if chunk_indices != set(range(packed_chunk_count)):
            raise RuntimeError(f"Target 2 case {case.name} packed chunks differ")
        expected_native_count *= packed_chunk_count
    if len(native_records) != expected_native_count:
        raise RuntimeError(f"Target 2 case {case.name} native handle count differs")
    native_identities: set[tuple[int, int, int]] = set()
    bytes_by_request_rank: dict[tuple[int, int], int] = {}
    for native in native_records:
        if not isinstance(native, dict):
            raise RuntimeError("Target 2 native telemetry must be an object")
        request_index = _evidence_integer(native, "request_index")
        rank = _evidence_integer(native, "rank")
        chunk_index = (
            0
            if case.arm is GateArm.DIRECT_READ
            else _evidence_integer(native, "chunk_index")
        )
        if request_index >= case.in_flight_depth or rank >= 4:
            raise RuntimeError(f"Target 2 case {case.name} native identity differs")
        native_identity = (request_index, chunk_index, rank)
        if native_identity in native_identities:
            raise RuntimeError(f"Target 2 case {case.name} duplicates a native handle")
        native_identities.add(native_identity)
        if native.get("backend") != "UCX":
            raise RuntimeError(f"Target 2 case {case.name} selected non-UCX")
        expected_descriptors = (
            case.fragmentation.expected_descriptors_per_rank
            if case.arm is GateArm.DIRECT_READ
            else 1
        )
        if native.get("descriptor_count") != expected_descriptors:
            raise RuntimeError(
                f"Target 2 case {case.name} native descriptor count differs"
            )
        if case.arm is GateArm.DIRECT_READ:
            expected_bytes = _EXACT_2K_BYTES // 4
        else:
            total_bytes = native.get("total_bytes")
            if type(total_bytes) is not int or not 0 < total_bytes <= case.chunk_bytes:
                raise RuntimeError(
                    f"Target 2 case {case.name} packed byte count differs"
                )
            expected_bytes = total_bytes
        if native.get("total_bytes") != expected_bytes:
            raise RuntimeError(f"Target 2 case {case.name} native bytes differ")
        bytes_by_request_rank[(request_index, rank)] = (
            bytes_by_request_rank.get((request_index, rank), 0) + expected_bytes
        )
    expected_request_rank_bytes = {
        (request_index, rank): _EXACT_2K_BYTES // 4
        for request_index in range(case.in_flight_depth)
        for rank in range(4)
    }
    if bytes_by_request_rank != expected_request_rank_bytes:
        raise RuntimeError(f"Target 2 case {case.name} wire coverage differs")

    expected_integrity_mode = {
        (GateArm.DIRECT_READ, False): "every_request",
        (GateArm.DIRECT_READ, True): "all_staging_and_final_scatter",
        (GateArm.PACKED_READ, False): "every_chunk_and_request",
        (GateArm.PACKED_READ, True): "final_request",
        (GateArm.PACKED_WRITE, False): "every_chunk_and_request",
        (GateArm.PACKED_WRITE, True): "final_request",
    }[(case.arm, expected_measured)]
    if record.get("integrity_mode") != expected_integrity_mode:
        raise RuntimeError(f"Target 2 case {case.name} integrity mode differs")
    final_states = _evidence_list(record, "final_slot_states")
    expected_slot_count = (
        case.in_flight_depth
        if case.arm is GateArm.DIRECT_READ
        else 4 * _PACK_SLOT_COUNT
    )
    if len(final_states) != expected_slot_count or any(
        state != SlotState.FREE.value for state in final_states
    ):
        raise RuntimeError(f"Target 2 case {case.name} retained a live slot")


def _native_critical_path_ms(record: dict[str, object], case: GateCase) -> float:
    grouped: dict[tuple[int, int], list[float]] = {}
    for value in _evidence_list(record, "native_handles"):
        if not isinstance(value, dict):
            raise RuntimeError("Target 2 native telemetry must be an object")
        request_index = _evidence_integer(value, "request_index")
        chunk_index = (
            0
            if case.arm is GateArm.DIRECT_READ
            else _evidence_integer(value, "chunk_index")
        )
        duration = _evidence_number(value, "transfer_duration_us") / 1000.0
        grouped.setdefault((request_index, chunk_index), []).append(duration)
    return sum(max(durations) for durations in grouped.values())


def _device_critical_path_ms(record: dict[str, object], field_name: str) -> float:
    grouped: dict[tuple[int, int], list[float]] = {}
    for value in _evidence_list(record, field_name):
        if not isinstance(value, dict):
            raise RuntimeError(f"Target 2 {field_name} entry must be an object")
        identity = (
            _evidence_integer(value, "request_index"),
            _evidence_integer(value, "chunk_index"),
        )
        duration = _evidence_nonnegative_number(value, "duration_ms")
        grouped.setdefault(identity, []).append(duration)
    return sum(max(durations) for durations in grouped.values())


def _device_total_ms(record: dict[str, object], field_name: str) -> float:
    total = 0.0
    for value in _evidence_list(record, field_name):
        if isinstance(value, dict):
            total += _evidence_nonnegative_number(value, "duration_ms")
            continue
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
            raise RuntimeError(f"Target 2 {field_name} duration must be non-negative")
        total += float(value)
    return total
