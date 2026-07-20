"""GPU runtime roles for the Target 2 fixed-byte transport gate."""

import base64
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from nixl._api import nixl_agent, nixl_xfer_handle

from tools.gemma4_pd.nixl_micro_rig.attestation import (
    collect_process_attestation,
    write_process_maps,
)
from tools.gemma4_pd.nixl_micro_rig.config import RigConfig, load_config
from tools.gemma4_pd.nixl_micro_rig.data import (
    fill_destination_canary_rows,
    fill_source_rows,
    verify_destination_rows,
    verify_source_rows,
    verify_staging_rows,
)
from tools.gemma4_pd.nixl_micro_rig.geometry import TransferPlan
from tools.gemma4_pd.nixl_micro_rig.protocol import (
    JsonChannel,
    StopPayload,
    StoppedPayload,
)
from tools.gemma4_pd.nixl_micro_rig.roles import (
    _UCX_BACKEND,
    _agent,
    _assert_role_visibility,
    _connect,
    _control_server,
    _fill_staging_guards,
    _raw_descriptors,
    _register_regions,
    _scatter,
    _staging_guard_ranges,
    _telemetry,
    _verify_staging_guards,
)
from tools.gemma4_pd.nixl_micro_rig.target2_gate import (
    BufferSlotLease,
    GateArm,
    GateCase,
    SlotState,
    build_gate_packed_plan,
    build_gate_request_plans,
    build_gate_source_plan,
    packed_descriptors,
    verify_packed_chunk,
)
from tools.gemma4_pd.nixl_micro_rig.target2_gate_protocol import (
    GateBatchCompletePayload,
    GateBatchPostPayload,
    GateBatchPreparePayload,
    GateBatchReadyPayload,
    GateConsumerHelloPayload,
    GatePackCommandPayload,
    GatePackReadyPayload,
    GateProducerHelloPayload,
)
from vllm.distributed.kv_transfer.coalesced_layout import (
    PackedTransferChunk,
    rank_major_slot_base,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.coalesced_pack import (
    launch_packed_chunk_pack,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.coalesced_scatter import (
    launch_packed_chunk_scatter,
)

_PACK_SLOT_COUNT = 2
_GUARD_BYTES = 1024 * 1024
_POLL_SECONDS = 0.0001


@dataclass(frozen=True, slots=True)
class _PackedTask:
    """Bind one logical request to one bounded packed chunk.

    :ivar index: Monotonic task identity within the batch.
    :ivar request_index: Logical request identity.
    :ivar plan: Exact request-specific source plan.
    :ivar chunk: Rank-local packed chunk geometry.
    """

    index: int
    request_index: int
    plan: TransferPlan
    chunk: PackedTransferChunk

    @property
    def slot_index(self) -> int:
        """Return the alternating bounded-pool slot.

        :returns: Slot index for this task.
        """
        return self.index % _PACK_SLOT_COUNT


@dataclass(frozen=True, slots=True)
class _ActivePackedTask:
    """Own one ready packed task until native and scatter completion."""

    task: _PackedTask
    commands: tuple[GatePackCommandPayload, ...]
    ready: tuple[GatePackReadyPayload, ...]
    handles: tuple[tuple[int, nixl_xfer_handle, str], ...]


@dataclass(frozen=True, slots=True)
class _ProducerPackedTask:
    """Own one producer pack and optional WRITE until readiness publication."""

    command: GatePackCommandPayload
    chunk: PackedTransferChunk
    generation: int
    pack_gpu_ms: float
    notification_id: bytes
    write_handle: nixl_xfer_handle | None
    write_status: str | None


@dataclass(slots=True)
class _NotificationInbox:
    """Retain unmatched NIXL notifications while waiting for exact identities."""

    agent: nixl_agent
    timeout_seconds: int
    pending: set[bytes]

    def wait(self, expected: bytes) -> None:
        """Consume one exact notification without dropping unrelated messages.

        :param expected: Unique notification identity.
        :raises RuntimeError: If the notification does not arrive by the deadline.
        """
        deadline = time.monotonic() + self.timeout_seconds
        while expected not in self.pending:
            notifications = self.agent.get_new_notifs(backends=[_UCX_BACKEND])
            self.pending.update(
                notification
                for batch in notifications.values()
                for notification in batch
            )
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"NIXL notification deadline expired for {expected!r}"
                )
            if expected not in self.pending:
                time.sleep(0.001)
        self.pending.remove(expected)


def _notification_id(
    *,
    run_id: str,
    case_index: int,
    batch_index: int,
    request_index: int,
    rank: int,
    chunk_index: int | None,
) -> bytes:
    chunk = "direct" if chunk_index is None else str(chunk_index)
    return (
        f"target2:{run_id}:{case_index}:{batch_index}:{request_index}:{rank}:{chunk}"
    ).encode()


def _payload_iterations(
    *, case_index: int, batch_index: int, in_flight_depth: int
) -> tuple[int, ...]:
    if batch_index >= 1000 or in_flight_depth >= 1000:
        raise ValueError("gate batch and in-flight indices must remain below 1000")
    start = case_index * 1_000_000 + batch_index * 1000
    return tuple(start + request_index for request_index in range(in_flight_depth))


def _batch_payload(
    *, case: GateCase, case_index: int, batch_index: int
) -> GateBatchPreparePayload:
    return GateBatchPreparePayload(
        case_name=case.name,
        arm=case.arm.value,
        run_count=case.fragmentation.run_count,
        expected_descriptors_per_rank=(
            case.fragmentation.expected_descriptors_per_rank
        ),
        chunk_bytes=case.chunk_bytes,
        in_flight_depth=case.in_flight_depth,
        batch_index=batch_index,
        measured=batch_index >= case.warmup_batches,
        payload_iterations=_payload_iterations(
            case_index=case_index,
            batch_index=batch_index,
            in_flight_depth=case.in_flight_depth,
        ),
    )


def _validate_batch_payload(
    expected: GateBatchPreparePayload,
    actual: GateBatchPreparePayload,
) -> None:
    if actual != expected:
        raise RuntimeError(
            f"consumer gate batch differs: observed={actual}, expected={expected}"
        )


def _wait_handle(
    *,
    agent: nixl_agent,
    handle: nixl_xfer_handle,
    initial_status: str,
    timeout_seconds: int,
) -> None:
    if initial_status == "DONE":
        return
    if initial_status != "PROC":
        raise RuntimeError(f"NIXL post returned terminal failure {initial_status!r}")
    deadline = time.monotonic() + timeout_seconds
    while True:
        status = agent.check_xfer_state(handle)
        if status == "DONE":
            return
        if status != "PROC":
            raise RuntimeError(f"NIXL transfer returned terminal failure {status!r}")
        if time.monotonic() >= deadline:
            raise RuntimeError("NIXL transfer deadline expired")
        time.sleep(_POLL_SECONDS)


def _pack_slot_backings(
    *, device: torch.device, slot_bytes: int
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    backings = tuple(
        torch.full(
            (slot_bytes + 2 * _GUARD_BYTES,),
            0x6D + slot_index,
            dtype=torch.uint8,
            device=device,
        )
        for slot_index in range(_PACK_SLOT_COUNT)
    )
    slots = tuple(
        backing[_GUARD_BYTES : _GUARD_BYTES + slot_bytes] for backing in backings
    )
    return backings, slots


def _verify_slot_guards(backings: tuple[torch.Tensor, ...], slot_bytes: int) -> None:
    for slot_index, backing in enumerate(backings):
        expected = 0x6D + slot_index
        if not torch.all(backing[:_GUARD_BYTES] == expected).item():
            raise RuntimeError(f"packed slot {slot_index} leading guard changed")
        if not torch.all(backing[_GUARD_BYTES + slot_bytes :] == expected).item():
            raise RuntimeError(f"packed slot {slot_index} trailing guard changed")


def _case_plans(config: RigConfig, case: GateCase) -> tuple[TransferPlan, ...]:
    return build_gate_request_plans(
        config,
        config.scenarios[0],
        case.fragmentation,
        case.in_flight_depth,
    )


def _validate_native_telemetry(
    record: dict[str, object], *, expected_bytes: int, expected_descriptors: int
) -> None:
    """Require terminal UCX telemetry to match the exact transfer geometry.

    :param record: NIXL telemetry record.
    :param expected_bytes: Exact bytes represented by the handle.
    :param expected_descriptors: Exact descriptors represented by the handle.
    :raises RuntimeError: If backend selection or transfer geometry differs.
    """
    expected = (_UCX_BACKEND, expected_bytes, expected_descriptors)
    observed = (
        record.get("backend"),
        record.get("total_bytes"),
        record.get("descriptor_count"),
    )
    if observed != expected:
        raise RuntimeError(
            "native transfer telemetry differs: "
            f"observed={observed}, expected={expected}"
        )


def run_target2_gate_producer(
    *,
    config_path: Path,
    transport_arm_name: str,
    run_id: str,
    rank: int,
    cases: tuple[GateCase, ...],
    artifact_directory: Path,
    expected_cuda_visibility: str,
) -> None:
    """Run one producer rank for every fixed-byte gate cell.

    :param config_path: Immutable exact-2K rig configuration.
    :param transport_arm_name: Fresh-process UCX transport arm.
    :param run_id: Campaign UUID.
    :param rank: Producer TP rank.
    :param cases: Ordered gate matrix.
    :param artifact_directory: Fresh arm artifact directory.
    :param expected_cuda_visibility: Preflighted producer GPU UUID roster.
    """
    config = load_config(config_path)
    transport_arm = config.transport_arm(transport_arm_name)
    _assert_role_visibility(expected_cuda_visibility)
    if rank < 0 or rank >= len(config.producer_devices):
        raise ValueError("producer rank is outside the configured TP roster")
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    agent = _agent(f"target2-p-{run_id}-r{rank}", config)
    source_regions = tuple(
        torch.empty(
            config.source_block_count * region.row_bytes,
            dtype=torch.uint8,
            device=device,
        )
        for region in config.regions
    )
    maximum_chunk_bytes = max(
        (case.chunk_bytes for case in cases if case.arm is not GateArm.DIRECT_READ),
        default=64 * 1024 * 1024,
    )
    pack_backings, pack_slots = _pack_slot_backings(
        device=device, slot_bytes=maximum_chunk_bytes
    )
    source_registration = _register_regions(agent, source_regions)
    pack_registration = _register_regions(agent, pack_backings)
    pack_stream = torch.cuda.Stream(device=device)
    attestation = collect_process_attestation(
        agent,
        role=f"target2-producer:{rank}",
        physical_device=config.producer_devices[rank],
        logical_device=rank,
    )
    artifact_directory.mkdir(parents=True, exist_ok=True)
    (artifact_directory / f"producer-{rank}-attestation.json").write_text(
        json.dumps(attestation, indent=2, sort_keys=True)
    )
    write_process_maps(artifact_directory / f"producer-{rank}-proc-maps.txt")
    server = _control_server(config, rank)
    completed_cleanly = False
    try:
        connection, _ = server.accept()
        with JsonChannel(
            connection,
            run_id=run_id,
            config_fingerprint=config.fingerprint,
            transport_arm=transport_arm.name,
            local_role="producer",
            local_rank=rank,
            remote_role="consumer",
            remote_rank=0,
            timeout_seconds=float(config.transfer_timeout_seconds),
        ) as channel:
            channel.send(
                GateProducerHelloPayload(
                    agent_metadata=base64.b64encode(
                        agent.get_agent_metadata()
                    ).decode(),
                    source_base_addresses=tuple(
                        region.data_ptr() for region in source_regions
                    ),
                    source_region_bytes=tuple(
                        region.numel() for region in source_regions
                    ),
                    pack_slot_base_addresses=tuple(
                        slot.data_ptr() for slot in pack_slots
                    ),
                    pack_slot_bytes=maximum_chunk_bytes,
                    logical_device=rank,
                ),
                iteration=-1,
            )
            consumer_hello = channel.receive(GateConsumerHelloPayload, iteration=-1)
            if consumer_hello.receive_slot_bytes != maximum_chunk_bytes:
                raise RuntimeError("consumer packed receive capacity differs")
            if consumer_hello.logical_device != 0:
                raise RuntimeError("consumer packed receive device differs")
            remote_consumer = agent.add_remote_agent(
                base64.b64decode(consumer_hello.agent_metadata)
            )
            notification_inbox = _NotificationInbox(
                agent=agent,
                timeout_seconds=config.transfer_timeout_seconds,
                pending=set(),
            )
            global_batch = 0
            for case_index, case in enumerate(cases):
                plans = _case_plans(config, case)
                source_plans = tuple(build_gate_source_plan(plan) for plan in plans)
                packed_plans = (
                    None
                    if case.arm is GateArm.DIRECT_READ
                    else tuple(
                        build_gate_packed_plan(plan, case.chunk_bytes) for plan in plans
                    )
                )
                layout = None if packed_plans is None else packed_plans[0]
                batch_count = case.warmup_batches + case.measured_batches
                for batch_index in range(batch_count):
                    expected_batch = _batch_payload(
                        case=case,
                        case_index=case_index,
                        batch_index=batch_index,
                    )
                    actual_batch = channel.receive(
                        GateBatchPreparePayload, iteration=global_batch
                    )
                    _validate_batch_payload(expected_batch, actual_batch)
                    for request_index, plan in enumerate(plans):
                        fill_source_rows(
                            config=config,
                            plan=plan,
                            source_rank=rank,
                            iteration=actual_batch.payload_iterations[request_index],
                            regions=source_regions,
                        )
                        verify_source_rows(
                            config=config,
                            plan=plan,
                            source_rank=rank,
                            iteration=actual_batch.payload_iterations[request_index],
                            regions=source_regions,
                        )
                    source_ready_event = torch.cuda.Event()
                    source_ready_event.record(torch.cuda.current_stream(device))
                    channel.send(
                        GateBatchReadyPayload(
                            source_plan_digests=tuple(
                                plan.transport.source_digest for plan in plans
                            )
                        ),
                        iteration=global_batch,
                    )

                    slots = tuple(
                        BufferSlotLease(slot_index=slot_index)
                        for slot_index in range(_PACK_SLOT_COUNT)
                    )
                    notifications_seen = 0
                    pending_slot_notifications: dict[int, bytes] = {}
                    if case.arm is not GateArm.DIRECT_READ:
                        assert layout is not None
                        assert packed_plans is not None
                        command_count = case.in_flight_depth * len(layout.chunks)
                        command_ordinal = 0
                        while command_ordinal < command_count:
                            wave_size = min(
                                _PACK_SLOT_COUNT,
                                command_count - command_ordinal,
                            )
                            wave: list[_ProducerPackedTask] = []
                            for wave_index in range(wave_size):
                                ordinal = command_ordinal + wave_index
                                command = channel.receive(
                                    GatePackCommandPayload,
                                    iteration=global_batch,
                                )
                                expected_request_index, expected_chunk_index = divmod(
                                    ordinal, len(layout.chunks)
                                )
                                expected_slot_index = ordinal % _PACK_SLOT_COUNT
                                expected_notification = _notification_id(
                                    run_id=run_id,
                                    case_index=case_index,
                                    batch_index=batch_index,
                                    request_index=expected_request_index,
                                    rank=rank,
                                    chunk_index=expected_chunk_index,
                                )
                                observed_command = (
                                    command.request_index,
                                    command.chunk_index,
                                    command.slot_index,
                                    base64.b64decode(command.notification_id),
                                )
                                expected_command = (
                                    expected_request_index,
                                    expected_chunk_index,
                                    expected_slot_index,
                                    expected_notification,
                                )
                                if observed_command != expected_command:
                                    raise RuntimeError(
                                        "consumer pack command order or identity "
                                        "differs: "
                                        f"observed={observed_command}, "
                                        f"expected={expected_command}"
                                    )
                                prior_notification = pending_slot_notifications.pop(
                                    command.slot_index, None
                                )
                                if prior_notification is not None:
                                    notification_inbox.wait(prior_notification)
                                    notifications_seen += 1
                                    slots[command.slot_index].release_after_native()
                                lease = slots[command.slot_index]
                                generation = lease.begin_device_write(
                                    request_index=command.request_index,
                                    chunk_index=command.chunk_index,
                                )
                                chunk = layout.chunks[command.chunk_index]
                                pack_launch = launch_packed_chunk_pack(
                                    pack_slots[command.slot_index],
                                    source_regions,
                                    source_plans[command.request_index],
                                    packed_plans[command.request_index],
                                    command.chunk_index,
                                    pack_stream,
                                    destination_base_offset_bytes=0,
                                    source_ready_event=source_ready_event,
                                )
                                pack_launch.completion_event.synchronize()
                                lease.begin_native_transfer()
                                notification_id = base64.b64decode(
                                    command.notification_id
                                )
                                write_handle: nixl_xfer_handle | None = None
                                write_status: str | None = None
                                if case.arm is GateArm.PACKED_READ:
                                    pending_slot_notifications[command.slot_index] = (
                                        notification_id
                                    )
                                else:
                                    local_raw, remote_raw = packed_descriptors(
                                        local_base=pack_slots[
                                            command.slot_index
                                        ].data_ptr(),
                                        local_device=rank,
                                        remote_base=(
                                            rank_major_slot_base(
                                                consumer_hello.receive_slot_base_addresses[
                                                    command.slot_index
                                                ],
                                                rank,
                                                case.chunk_bytes,
                                            )
                                        ),
                                        remote_device=consumer_hello.logical_device,
                                        chunk=chunk,
                                    )
                                    write_handle = agent.initialize_xfer(
                                        "WRITE",
                                        agent.get_xfer_descs(local_raw, "VRAM"),
                                        agent.get_xfer_descs(remote_raw, "VRAM"),
                                        remote_consumer,
                                        notification_id,
                                        backends=[_UCX_BACKEND],
                                    )
                                    write_status = agent.transfer(write_handle)
                                wave.append(
                                    _ProducerPackedTask(
                                        command=command,
                                        chunk=chunk,
                                        generation=generation,
                                        pack_gpu_ms=pack_launch.gpu_duration_ms(),
                                        notification_id=notification_id,
                                        write_handle=write_handle,
                                        write_status=write_status,
                                    )
                                )
                            for prepared in wave:
                                native_telemetry: dict[str, object] = {}
                                native_done = False
                                if prepared.write_handle is not None:
                                    assert prepared.write_status is not None
                                    _wait_handle(
                                        agent=agent,
                                        handle=prepared.write_handle,
                                        initial_status=prepared.write_status,
                                        timeout_seconds=(
                                            config.transfer_timeout_seconds
                                        ),
                                    )
                                    native_telemetry = _telemetry(
                                        agent, prepared.write_handle
                                    )
                                    agent.release_xfer_handle(prepared.write_handle)
                                    slots[
                                        prepared.command.slot_index
                                    ].release_after_native()
                                    native_done = True
                                channel.send(
                                    GatePackReadyPayload(
                                        request_index=(prepared.command.request_index),
                                        chunk_index=prepared.command.chunk_index,
                                        slot_index=prepared.command.slot_index,
                                        slot_generation=prepared.generation,
                                        payload_bytes=(
                                            prepared.chunk.rank_stride_bytes
                                        ),
                                        pack_gpu_ms=prepared.pack_gpu_ms,
                                        native_done=native_done,
                                        native_telemetry=native_telemetry,
                                    ),
                                    iteration=global_batch,
                                )
                            command_ordinal += wave_size

                    complete = channel.receive(
                        GateBatchCompletePayload, iteration=global_batch
                    )
                    if not complete.success:
                        raise RuntimeError("consumer reported Target 2 batch failure")
                    if case.arm is GateArm.DIRECT_READ:
                        for request_index in range(case.in_flight_depth):
                            notification_inbox.wait(
                                _notification_id(
                                    run_id=run_id,
                                    case_index=case_index,
                                    batch_index=batch_index,
                                    request_index=request_index,
                                    rank=rank,
                                    chunk_index=None,
                                )
                            )
                            notifications_seen += 1
                    else:
                        for slot_index, notification_id in tuple(
                            pending_slot_notifications.items()
                        ):
                            notification_inbox.wait(notification_id)
                            notifications_seen += 1
                            slots[slot_index].release_after_native()
                            del pending_slot_notifications[slot_index]
                    for request_index, plan in enumerate(plans):
                        verify_source_rows(
                            config=config,
                            plan=plan,
                            source_rank=rank,
                            iteration=actual_batch.payload_iterations[request_index],
                            regions=source_regions,
                        )
                    _verify_slot_guards(pack_backings, maximum_chunk_bytes)
                    channel.send(
                        GateBatchPostPayload(
                            sources_verified=True,
                            notifications_seen=notifications_seen,
                            final_slot_states=tuple(slot.state.value for slot in slots),
                        ),
                        iteration=global_batch,
                    )
                    global_batch += 1
            channel.receive(StopPayload, iteration=-1)
            channel.send(StoppedPayload(), iteration=-1)
            completed_cleanly = True
    finally:
        server.close()
        if completed_cleanly:
            agent.deregister_memory(pack_registration, backends=[_UCX_BACKEND])
            agent.deregister_memory(source_registration, backends=[_UCX_BACKEND])


def _validate_producer_hello(
    config: RigConfig,
    rank: int,
    hello: GateProducerHelloPayload,
    maximum_chunk_bytes: int,
) -> None:
    expected_source_bytes = tuple(
        config.source_block_count * region.row_bytes for region in config.regions
    )
    if hello.source_region_bytes != expected_source_bytes:
        raise RuntimeError(f"producer rank {rank} source registration differs")
    if len(hello.source_base_addresses) != len(config.regions):
        raise RuntimeError(f"producer rank {rank} source region count differs")
    if hello.logical_device != rank:
        raise RuntimeError(f"producer rank {rank} logical device differs")
    if hello.pack_slot_bytes != maximum_chunk_bytes:
        raise RuntimeError(f"producer rank {rank} pack slot capacity differs")


def _ready_digests(plans: tuple[TransferPlan, ...]) -> tuple[str, ...]:
    return tuple(plan.transport.source_digest for plan in plans)


def _run_direct_batch(
    *,
    config: RigConfig,
    case: GateCase,
    case_index: int,
    batch_index: int,
    payload: GateBatchPreparePayload,
    plans: tuple[TransferPlan, ...],
    agent: nixl_agent,
    remote_agents: list[str],
    remote_bases: list[list[int]],
    remote_devices: list[int],
    staging: torch.Tensor,
    destinations: tuple[torch.Tensor, ...],
    scatter_stream: torch.cuda.Stream,
    run_id: str,
) -> dict[str, object]:
    active_bytes = case.in_flight_depth * plans[0].staging_bytes
    if active_bytes > staging.numel():
        raise RuntimeError(
            f"direct q{case.in_flight_depth} needs {active_bytes} staging bytes, "
            f"capacity is {staging.numel()}"
        )
    guard_ranges = _staging_guard_ranges(
        capacity=staging.numel(), offset=0, size=active_bytes
    )
    _fill_staging_guards(staging, guard_ranges, 0xD7)
    staging[:active_bytes].fill_(0xA5)
    torch.cuda.synchronize(staging.device)
    leases = tuple(
        BufferSlotLease(slot_index=request_index)
        for request_index in range(case.in_flight_depth)
    )
    handles: list[tuple[int, int, nixl_xfer_handle, str]] = []
    started = time.perf_counter()
    for request_index, plan in enumerate(plans):
        leases[request_index].begin_native_write(
            request_index=request_index, chunk_index=0
        )
        staging_offset = request_index * plan.staging_bytes
        for rank in range(plan.rank_count):
            local_raw, remote_raw = _raw_descriptors(
                config=config,
                plan=plan,
                staging=staging,
                staging_offset=staging_offset,
                rank=rank,
                remote_bases=remote_bases[rank],
                remote_device=remote_devices[rank],
            )
            handle = agent.initialize_xfer(
                "READ",
                agent.get_xfer_descs(local_raw, "VRAM"),
                agent.get_xfer_descs(remote_raw, "VRAM"),
                remote_agents[rank],
                _notification_id(
                    run_id=run_id,
                    case_index=case_index,
                    batch_index=batch_index,
                    request_index=request_index,
                    rank=rank,
                    chunk_index=None,
                ),
                backends=[_UCX_BACKEND],
            )
            handles.append((request_index, rank, handle, agent.transfer(handle)))
    native_records: list[dict[str, object]] = []
    for request_index, rank, handle, initial_status in handles:
        _wait_handle(
            agent=agent,
            handle=handle,
            initial_status=initial_status,
            timeout_seconds=config.transfer_timeout_seconds,
        )
        record = _telemetry(agent, handle)
        _validate_native_telemetry(
            record,
            expected_bytes=plans[request_index].transport.rank_stride_bytes,
            expected_descriptors=case.fragmentation.expected_descriptors_per_rank,
        )
        record.update({"request_index": request_index, "rank": rank})
        native_records.append(record)
        agent.release_xfer_handle(handle)
    scatter_gpu_ms: list[float] = []
    for request_index, plan in enumerate(plans):
        leases[request_index].begin_device_read()
        if not payload.measured:
            fill_destination_canary_rows(
                config=config,
                plan=plan,
                destinations=destinations,
                value=(0x31 + request_index) & 0xFF,
            )
        scatter_gpu_ms.append(
            _scatter(
                plan=plan,
                staging=staging,
                staging_offset=request_index * plan.staging_bytes,
                destinations=destinations,
                stream=scatter_stream,
            )
        )
        if not payload.measured:
            verify_destination_rows(
                config=config,
                plan=plan,
                iteration=payload.payload_iterations[request_index],
                destinations=destinations,
            )
        leases[request_index].release_after_device()
    elapsed = time.perf_counter() - started

    _verify_staging_guards(staging, guard_ranges, 0xD7)
    for request_index, plan in enumerate(plans):
        verify_staging_rows(
            config=config,
            plan=plan,
            iteration=payload.payload_iterations[request_index],
            staging=staging,
            staging_offset=request_index * plan.staging_bytes,
        )
    verify_destination_rows(
        config=config,
        plan=plans[-1],
        iteration=payload.payload_iterations[-1],
        destinations=destinations,
    )
    return {
        "elapsed_seconds": elapsed,
        "native_handles": native_records,
        "pack_gpu_ms": [],
        "scatter_gpu_ms": scatter_gpu_ms,
        "packed_window_limit": 0,
        "maximum_active_packed_tasks": 0,
        "integrity_mode": (
            "every_request" if not payload.measured else "all_staging_and_final_scatter"
        ),
        "final_slot_states": tuple(lease.state.value for lease in leases),
    }


def _run_packed_batch(
    *,
    config: RigConfig,
    case: GateCase,
    case_index: int,
    batch_index: int,
    payload: GateBatchPreparePayload,
    plans: tuple[TransferPlan, ...],
    agent: nixl_agent,
    channels: list[JsonChannel],
    remote_agents: list[str],
    remote_pack_bases: list[tuple[int, ...]],
    remote_devices: list[int],
    receive_slots: tuple[torch.Tensor, ...],
    receive_backings: tuple[torch.Tensor, ...],
    destinations: tuple[torch.Tensor, ...],
    scatter_stream: torch.cuda.Stream,
    notification_inbox: _NotificationInbox,
    run_id: str,
    global_batch: int,
) -> dict[str, object]:
    packed_plans = tuple(
        build_gate_packed_plan(plan, case.chunk_bytes) for plan in plans
    )
    chunk_count = len(packed_plans[0].chunks)
    if any(len(packed_plan.chunks) != chunk_count for packed_plan in packed_plans):
        raise RuntimeError("request-specific packed chunk counts differ")
    tasks = tuple(
        _PackedTask(
            index=request_index * chunk_count + chunk.chunk_index,
            request_index=request_index,
            plan=plan,
            chunk=chunk,
        )
        for request_index, (plan, packed_plan) in enumerate(
            zip(plans, packed_plans, strict=True)
        )
        for chunk in packed_plan.chunks
    )
    leases = tuple(
        tuple(
            BufferSlotLease(slot_index=rank * _PACK_SLOT_COUNT + slot_index)
            for slot_index in range(_PACK_SLOT_COUNT)
        )
        for rank in range(plans[0].rank_count)
    )
    native_records: list[dict[str, object]] = []
    pack_gpu_ms: list[dict[str, object]] = []
    scatter_gpu_ms: list[dict[str, object]] = []
    commands_by_task: dict[int, tuple[GatePackCommandPayload, ...]] = {}

    def send_task(task: _PackedTask) -> None:
        if task.index in commands_by_task:
            raise RuntimeError(f"packed task {task.index} was sent more than once")
        commands: list[GatePackCommandPayload] = []
        for rank, channel in enumerate(channels):
            notification_id = _notification_id(
                run_id=run_id,
                case_index=case_index,
                batch_index=batch_index,
                request_index=task.request_index,
                rank=rank,
                chunk_index=task.chunk.chunk_index,
            )
            command = GatePackCommandPayload(
                request_index=task.request_index,
                chunk_index=task.chunk.chunk_index,
                slot_index=task.slot_index,
                notification_id=base64.b64encode(notification_id).decode(),
            )
            leases[rank][task.slot_index].begin_native_write(
                request_index=task.request_index,
                chunk_index=task.chunk.chunk_index,
            )
            channel.send(command, iteration=global_batch)
            commands.append(command)
        commands_by_task[task.index] = tuple(commands)

    def receive_ready(task: _PackedTask) -> tuple[GatePackReadyPayload, ...]:
        ready_payloads = tuple(
            channel.receive(GatePackReadyPayload, iteration=global_batch)
            for channel in channels
        )
        for rank, ready in enumerate(ready_payloads):
            expected = (
                task.request_index,
                task.chunk.chunk_index,
                task.slot_index,
                leases[rank][task.slot_index].generation,
                task.chunk.rank_stride_bytes,
            )
            observed = (
                ready.request_index,
                ready.chunk_index,
                ready.slot_index,
                ready.slot_generation,
                ready.payload_bytes,
            )
            if observed != expected:
                raise RuntimeError(
                    f"producer rank {rank} packed readiness differs: "
                    f"observed={observed}, expected={expected}"
                )
            if case.arm is GateArm.PACKED_READ:
                if ready.native_done or len(ready.native_telemetry) > 0:
                    raise RuntimeError(
                        f"producer rank {rank} READ readiness claims native work"
                    )
            else:
                if not ready.native_done:
                    raise RuntimeError(
                        f"producer rank {rank} WRITE lacks native completion"
                    )
                _validate_native_telemetry(
                    ready.native_telemetry,
                    expected_bytes=task.chunk.rank_stride_bytes,
                    expected_descriptors=1,
                )
            pack_gpu_ms.append(
                {
                    "request_index": task.request_index,
                    "chunk_index": task.chunk.chunk_index,
                    "rank": rank,
                    "duration_ms": ready.pack_gpu_ms,
                }
            )
        return ready_payloads

    def activate_task(task: _PackedTask) -> _ActivePackedTask:
        ready_payloads = receive_ready(task)
        command_payloads = commands_by_task[task.index]
        active_handles: list[tuple[int, nixl_xfer_handle, str]] = []
        if case.arm is GateArm.PACKED_READ:
            slot = receive_slots[task.slot_index]
            for rank in range(task.plan.rank_count):
                rank_base = rank_major_slot_base(
                    slot.data_ptr(), rank, case.chunk_bytes
                )
                local_raw, remote_raw = packed_descriptors(
                    local_base=rank_base,
                    local_device=slot.get_device(),
                    remote_base=remote_pack_bases[rank][task.slot_index],
                    remote_device=remote_devices[rank],
                    chunk=task.chunk,
                )
                handle = agent.initialize_xfer(
                    "READ",
                    agent.get_xfer_descs(local_raw, "VRAM"),
                    agent.get_xfer_descs(remote_raw, "VRAM"),
                    remote_agents[rank],
                    base64.b64decode(command_payloads[rank].notification_id),
                    backends=[_UCX_BACKEND],
                )
                active_handles.append((rank, handle, agent.transfer(handle)))
        return _ActivePackedTask(
            task=task,
            commands=command_payloads,
            ready=ready_payloads,
            handles=tuple(active_handles),
        )

    def finish_native(active: _ActivePackedTask) -> None:
        task = active.task
        if case.arm is GateArm.PACKED_READ:
            for rank, handle, initial_status in active.handles:
                _wait_handle(
                    agent=agent,
                    handle=handle,
                    initial_status=initial_status,
                    timeout_seconds=config.transfer_timeout_seconds,
                )
                record = _telemetry(agent, handle)
                _validate_native_telemetry(
                    record,
                    expected_bytes=task.chunk.rank_stride_bytes,
                    expected_descriptors=1,
                )
                record.update(
                    {
                        "request_index": task.request_index,
                        "chunk_index": task.chunk.chunk_index,
                        "rank": rank,
                    }
                )
                native_records.append(record)
                agent.release_xfer_handle(handle)
            return
        for rank, ready in enumerate(active.ready):
            notification_inbox.wait(
                base64.b64decode(active.commands[rank].notification_id)
            )
            record = dict(ready.native_telemetry)
            record.update(
                {
                    "request_index": task.request_index,
                    "chunk_index": task.chunk.chunk_index,
                    "rank": rank,
                }
            )
            native_records.append(record)

    if not payload.measured:
        fill_destination_canary_rows(
            config=config,
            plan=plans[0],
            destinations=destinations,
            value=0x41,
        )
    started = time.perf_counter()
    pending_index = 0
    maximum_active_tasks = 0
    while pending_index < len(tasks):
        wave_end = min(pending_index + _PACK_SLOT_COUNT, len(tasks))
        wave_tasks = tasks[pending_index:wave_end]
        for task in wave_tasks:
            send_task(task)
        active_tasks = tuple(activate_task(task) for task in wave_tasks)
        maximum_active_tasks = max(maximum_active_tasks, len(active_tasks))
        for active in active_tasks:
            task = active.task
            finish_native(active)
            slot = receive_slots[task.slot_index]
            for rank in range(task.plan.rank_count):
                leases[rank][task.slot_index].begin_device_read()
                if not payload.measured:
                    verify_packed_chunk(
                        plan=task.plan,
                        packed_plan=packed_plans[task.request_index],
                        chunk_index=task.chunk.chunk_index,
                        source_rank=rank,
                        iteration=payload.payload_iterations[task.request_index],
                        packed=slot,
                        packed_base_offset_bytes=rank * case.chunk_bytes,
                    )
            scatter_launch = launch_packed_chunk_scatter(
                slot,
                destinations,
                task.plan.transport,
                packed_plans[task.request_index],
                task.chunk.chunk_index,
                scatter_stream,
                staging_base_offset_bytes=0,
                staging_rank_stride_bytes=case.chunk_bytes,
            )
            scatter_launch.completion_event.synchronize()
            scatter_gpu_ms.append(
                {
                    "request_index": task.request_index,
                    "chunk_index": task.chunk.chunk_index,
                    "duration_ms": scatter_launch.gpu_duration_ms(),
                }
            )
            for rank in range(task.plan.rank_count):
                leases[rank][task.slot_index].release_after_device()

            request_complete = task.chunk.chunk_index + 1 == chunk_count
            if request_complete and not payload.measured:
                verify_destination_rows(
                    config=config,
                    plan=task.plan,
                    iteration=payload.payload_iterations[task.request_index],
                    destinations=destinations,
                )
                next_index = task.index + 1
                if next_index < len(tasks):
                    next_task = tasks[next_index]
                    fill_destination_canary_rows(
                        config=config,
                        plan=next_task.plan,
                        destinations=destinations,
                        value=(0x41 + next_task.request_index) & 0xFF,
                    )
        pending_index = wave_end
    elapsed = time.perf_counter() - started

    verify_destination_rows(
        config=config,
        plan=plans[-1],
        iteration=payload.payload_iterations[-1],
        destinations=destinations,
    )
    _verify_slot_guards(receive_backings, receive_slots[0].numel())
    return {
        "elapsed_seconds": elapsed,
        "native_handles": native_records,
        "pack_gpu_ms": pack_gpu_ms,
        "scatter_gpu_ms": scatter_gpu_ms,
        "packed_window_limit": _PACK_SLOT_COUNT,
        "maximum_active_packed_tasks": maximum_active_tasks,
        "integrity_mode": (
            "every_chunk_and_request" if not payload.measured else "final_request"
        ),
        "final_slot_states": tuple(
            lease.state.value for rank_leases in leases for lease in rank_leases
        ),
    }


def run_target2_gate_consumer(
    *,
    config_path: Path,
    transport_arm_name: str,
    run_id: str,
    cases: tuple[GateCase, ...],
    artifact_directory: Path,
    expected_cuda_visibility: str,
) -> None:
    """Run the TP1 decoder consumer for every fixed-byte gate cell.

    :param config_path: Immutable exact-2K rig configuration.
    :param transport_arm_name: Fresh-process UCX transport arm.
    :param run_id: Campaign UUID.
    :param cases: Ordered gate matrix.
    :param artifact_directory: Fresh arm artifact directory.
    :param expected_cuda_visibility: Preflighted consumer GPU UUID.
    """
    config = load_config(config_path)
    transport_arm = config.transport_arm(transport_arm_name)
    _assert_role_visibility(expected_cuda_visibility)
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    agent = _agent(f"target2-d-{run_id}", config)
    rank_count = len(config.producer_devices)
    maximum_chunk_bytes = max(
        case.chunk_bytes for case in cases if case.arm is not GateArm.DIRECT_READ
    )
    destinations = tuple(
        torch.empty(
            config.source_block_count * rank_count * region.row_bytes,
            dtype=torch.uint8,
            device=device,
        )
        for region in config.regions
    )
    destination_registration = _register_regions(agent, destinations)
    staging = torch.empty(
        config.staging_capacity_mib * 1024 * 1024,
        dtype=torch.uint8,
        device=device,
    )
    staging_registration = _register_regions(agent, (staging,))
    receive_slot_bytes = rank_count * maximum_chunk_bytes
    receive_backings, receive_slots = _pack_slot_backings(
        device=device,
        slot_bytes=receive_slot_bytes,
    )
    receive_registration = _register_regions(agent, receive_backings)

    channels: list[JsonChannel] = []
    remote_agents: list[str] = []
    remote_bases: list[list[int]] = []
    remote_pack_bases: list[tuple[int, ...]] = []
    remote_devices: list[int] = []
    for rank in range(rank_count):
        channel = JsonChannel(
            _connect(config, rank),
            run_id=run_id,
            config_fingerprint=config.fingerprint,
            transport_arm=transport_arm.name,
            local_role="consumer",
            local_rank=0,
            remote_role="producer",
            remote_rank=rank,
            timeout_seconds=float(config.transfer_timeout_seconds),
        )
        hello = channel.receive(GateProducerHelloPayload, iteration=-1)
        _validate_producer_hello(config, rank, hello, maximum_chunk_bytes)
        remote_agents.append(
            agent.add_remote_agent(base64.b64decode(hello.agent_metadata))
        )
        remote_bases.append(list(hello.source_base_addresses))
        remote_pack_bases.append(hello.pack_slot_base_addresses)
        remote_devices.append(hello.logical_device)
        channel.send(
            GateConsumerHelloPayload(
                agent_metadata=base64.b64encode(agent.get_agent_metadata()).decode(),
                receive_slot_base_addresses=tuple(
                    slot.data_ptr() for slot in receive_slots
                ),
                receive_slot_bytes=maximum_chunk_bytes,
                logical_device=0,
            ),
            iteration=-1,
        )
        channels.append(channel)

    scatter_stream = torch.cuda.Stream(device=device)
    notification_inbox = _NotificationInbox(
        agent=agent,
        timeout_seconds=config.transfer_timeout_seconds,
        pending=set(),
    )
    attestation = collect_process_attestation(
        agent,
        role="target2-consumer:0",
        physical_device=config.consumer_device,
        logical_device=0,
    )
    artifact_directory.mkdir(parents=True, exist_ok=True)
    (artifact_directory / "consumer-attestation.json").write_text(
        json.dumps(attestation, indent=2, sort_keys=True)
    )
    write_process_maps(artifact_directory / "consumer-proc-maps.txt")
    records: list[dict[str, object]] = []
    completed_cleanly = False
    global_batch = 0
    try:
        for case_index, case in enumerate(cases):
            plans = _case_plans(config, case)
            expected_digests = _ready_digests(plans)
            batch_count = case.warmup_batches + case.measured_batches
            for batch_index in range(batch_count):
                payload = _batch_payload(
                    case=case,
                    case_index=case_index,
                    batch_index=batch_index,
                )
                for channel in channels:
                    channel.send(payload, iteration=global_batch)
                for rank, channel in enumerate(channels):
                    ready = channel.receive(
                        GateBatchReadyPayload, iteration=global_batch
                    )
                    if ready.source_plan_digests != expected_digests:
                        raise RuntimeError(
                            f"producer rank {rank} source plan digests differ"
                        )
                if case.arm is GateArm.DIRECT_READ:
                    result = _run_direct_batch(
                        config=config,
                        case=case,
                        case_index=case_index,
                        batch_index=batch_index,
                        payload=payload,
                        plans=plans,
                        agent=agent,
                        remote_agents=remote_agents,
                        remote_bases=remote_bases,
                        remote_devices=remote_devices,
                        staging=staging,
                        destinations=destinations,
                        scatter_stream=scatter_stream,
                        run_id=run_id,
                    )
                else:
                    result = _run_packed_batch(
                        config=config,
                        case=case,
                        case_index=case_index,
                        batch_index=batch_index,
                        payload=payload,
                        plans=plans,
                        agent=agent,
                        channels=channels,
                        remote_agents=remote_agents,
                        remote_pack_bases=remote_pack_bases,
                        remote_devices=remote_devices,
                        receive_slots=receive_slots,
                        receive_backings=receive_backings,
                        destinations=destinations,
                        scatter_stream=scatter_stream,
                        notification_inbox=notification_inbox,
                        run_id=run_id,
                        global_batch=global_batch,
                    )
                for channel in channels:
                    channel.send(
                        GateBatchCompletePayload(success=True),
                        iteration=global_batch,
                    )
                post_records = [
                    channel.receive(GateBatchPostPayload, iteration=global_batch)
                    for channel in channels
                ]
                expected_notifications = (
                    case.in_flight_depth
                    if case.arm is GateArm.DIRECT_READ
                    else (
                        case.in_flight_depth
                        * len(build_gate_packed_plan(plans[0], case.chunk_bytes).chunks)
                        if case.arm is GateArm.PACKED_READ
                        else 0
                    )
                )
                for rank, post in enumerate(post_records):
                    if not post.sources_verified:
                        raise RuntimeError(
                            f"producer rank {rank} did not verify its sources"
                        )
                    if post.notifications_seen != expected_notifications:
                        raise RuntimeError(
                            f"producer rank {rank} notification count differs"
                        )
                    if any(
                        state != SlotState.FREE.value
                        for state in post.final_slot_states
                    ):
                        raise RuntimeError(
                            f"producer rank {rank} retained a live pack slot"
                        )
                records.append(
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
                        "batch_index": batch_index,
                        "measured": payload.measured,
                        "payload_iterations": payload.payload_iterations,
                        "request_bytes": plans[0].staging_bytes,
                        "total_batch_bytes": (
                            plans[0].staging_bytes * case.in_flight_depth
                        ),
                        **result,
                    }
                )
                global_batch += 1
        for channel in channels:
            channel.send(StopPayload(), iteration=-1)
        for channel in channels:
            channel.receive(StoppedPayload, iteration=-1)
        completed_cleanly = True
    finally:
        (artifact_directory / "target2-gate-batches.json").write_text(
            json.dumps(records, indent=2, sort_keys=True)
        )
        if completed_cleanly:
            agent.deregister_memory(receive_registration, backends=[_UCX_BACKEND])
            agent.deregister_memory(staging_registration, backends=[_UCX_BACKEND])
            agent.deregister_memory(destination_registration, backends=[_UCX_BACKEND])
        for channel in channels:
            channel.close()
