"""Independent producer and consumer runtime roles for the NIXL micro-rig."""

import base64
import hashlib
import json
import os
import socket
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from nixl._api import (
    nixl_agent,
    nixl_agent_config,
    nixl_prepped_dlist_handle,
    nixl_xfer_handle,
)

from tools.gemma4_pd.nixl_micro_rig.attestation import (
    collect_process_attestation,
    write_process_maps,
)
from tools.gemma4_pd.nixl_micro_rig.config import RigConfig, load_config
from tools.gemma4_pd.nixl_micro_rig.data import (
    ObservationContext,
    compact_digests,
    destination_observations,
    fill_destination_canary_rows,
    fill_source_rows,
    source_observations,
    staging_observations,
    verify_destination_canary_rows,
    verify_destination_rows,
    verify_source_rows,
    verify_staging_rows,
    write_observations,
)
from tools.gemma4_pd.nixl_micro_rig.geometry import (
    TransferPlan,
    build_configured_plan,
)
from tools.gemma4_pd.nixl_micro_rig.protocol import (
    CompletePayload,
    DigestRow,
    HelloPayload,
    JsonChannel,
    PreparedPayload,
    PreparePayload,
    SourcePostPayload,
    StopPayload,
    StoppedPayload,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.coalesced_scatter import (
    launch_coalesced_scatter,
    scatter_coalesced_reference,
)
from vllm.distributed.kv_transfer.staging_ownership import (
    HandleState,
    StagingRangeAllocator,
    StagingSafetyError,
)
from vllm.distributed.nixl_utils import canonicalize_nixl_agent_name

_UCX_BACKEND = "UCX"
_PREPARED_DESCRIPTOR_PLANES = 2
_STAGING_GUARD_BYTES = 1024 * 1024


def _agent(name: str, config: RigConfig) -> nixl_agent:
    """Create the exact NIXL agent configuration used by the server.

    :param name: Unique agent identity.
    :param config: Complete rig configuration.
    :returns: Initialized NIXL agent with progress thread enabled.
    """
    agent_config = nixl_agent_config(
        enable_prog_thread=True,
        enable_listen_thread=False,
        capture_telemetry=True,
        num_threads=config.nixl_num_threads,
        backends=[_UCX_BACKEND],
    )
    return nixl_agent(name, agent_config)


def _plan(config_path: Path, config: RigConfig, scenario_name: str) -> TransferPlan:
    """Resolve one exact synthetic or replay-backed transfer plan.

    :param config_path: Configuration path used to resolve replay inputs.
    :param config: Complete rig configuration.
    :param scenario_name: Configured scenario identity.
    :returns: Exact transfer plan shared by producer and consumer roles.
    """
    scenario = config.scenario(scenario_name)
    return build_configured_plan(config_path, config, scenario)


def _control_server(config: RigConfig, rank: int) -> socket.socket:
    """Bind one producer-rank control listener.

    :param config: Complete rig configuration.
    :param rank: Producer TP rank.
    :returns: Listening TCP socket with the campaign timeout applied.
    """
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.settimeout(config.transfer_timeout_seconds)
    server.bind((config.control_host, config.control_port + rank))
    server.listen(1)
    return server


def _connect(config: RigConfig, rank: int) -> socket.socket:
    """Connect the consumer to one producer control listener.

    :param config: Complete rig configuration.
    :param rank: Producer TP rank.
    :returns: Connected TCP socket.
    :raises RuntimeError: If the endpoint remains unavailable until timeout.
    """
    deadline = time.monotonic() + config.transfer_timeout_seconds
    while True:
        connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        connection.settimeout(config.transfer_timeout_seconds)
        try:
            connection.connect((config.control_host, config.control_port + rank))
        except (ConnectionRefusedError, TimeoutError) as error:
            connection.close()
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"producer rank {rank} control endpoint timed out"
                ) from error
            time.sleep(0.1)
            continue
        return connection


def _assert_role_visibility(expected: str) -> None:
    """Require the exact CUDA visibility established by the launcher.

    :param expected: Required ``CUDA_VISIBLE_DEVICES`` text.
    :raises RuntimeError: If inherited process state differs.
    """
    actual = os.environ.get("CUDA_VISIBLE_DEVICES")
    if actual != expected:
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES is {actual!r}, expected exact role mapping "
            f"{expected!r}"
        )


def _register_regions(agent: nixl_agent, regions: tuple[torch.Tensor, ...]) -> object:
    """Register complete CUDA tensors with the UCX backend.

    :param agent: Owning NIXL agent.
    :param regions: CUDA tensors retained for the registration lifetime.
    :returns: NIXL registration descriptor required for deregistration.
    """
    descriptors = [
        (region.data_ptr(), region.numel(), region.get_device(), "")
        for region in regions
    ]
    return agent.register_memory(descriptors, "VRAM", backends=[_UCX_BACKEND])


def _validate_producer_hello(
    config: RigConfig,
    rank: int,
    hello: HelloPayload,
    expected_cuda_visibility: str,
) -> None:
    """Require a producer advertisement to match its configured role exactly.

    :param config: Authenticated production-topology configuration.
    :param rank: Expected producer TP rank.
    :param hello: Typed producer advertisement.
    :param expected_cuda_visibility: Preflighted producer UUID roster.
    :raises RuntimeError: If process wiring or registration geometry differs.
    """
    expected_region_bytes = tuple(
        config.source_block_count * region.row_bytes for region in config.regions
    )
    if hello.region_bytes != expected_region_bytes:
        raise RuntimeError(
            f"producer rank {rank} registration lengths differ from configuration"
        )
    if len(hello.base_addresses) != len(config.regions):
        raise RuntimeError(f"producer rank {rank} registration count differs")
    if any(address <= 0 for address in hello.base_addresses):
        raise RuntimeError(f"producer rank {rank} advertised a null address")
    if hello.logical_device != rank:
        raise RuntimeError(
            f"producer rank {rank} advertised logical device {hello.logical_device}"
        )
    expected_attestation = {
        "role": f"producer:{rank}",
        "physical_device": config.producer_devices[rank],
        "logical_device": rank,
    }
    for field, expected in expected_attestation.items():
        actual = hello.attestation.get(field)
        if type(actual) is not type(expected) or actual != expected:
            raise RuntimeError(
                f"producer rank {rank} attestation {field} is {actual!r}, "
                f"expected {expected!r}"
            )
    environment = hello.attestation.get("environment")
    if not isinstance(environment, dict):
        raise RuntimeError(f"producer rank {rank} attestation lacks environment")
    if environment.get("CUDA_VISIBLE_DEVICES") != expected_cuda_visibility:
        raise RuntimeError(f"producer rank {rank} CUDA visibility differs")


def _observation_context(
    *,
    run_id: str,
    arm_name: str,
    producer_engine_id: str,
    iteration: int,
    producer_request_id: str,
    child_request_id: str,
) -> ObservationContext:
    """Create stage-independent evidence lineage for one iteration.

    :param run_id: Campaign UUID.
    :param arm_name: Fresh-process transport arm.
    :param producer_engine_id: Shared producer engine identity.
    :param iteration: Global campaign iteration.
    :param producer_request_id: Producer request identity.
    :param child_request_id: Consumer child request identity.
    :returns: Canonical observation context.
    """
    return ObservationContext(
        run_id=run_id,
        transport_arm=arm_name,
        producer_engine_id=producer_engine_id,
        producer_request_id=producer_request_id,
        offer_generation=iteration,
        iteration=iteration,
        child_request_id=child_request_id,
        consumer_engine_id=f"micro-d-{run_id}",
    )


def _await_notification(
    agent: nixl_agent, notification_id: bytes, timeout_seconds: int
) -> bool:
    """Wait for one exact NIXL notification.

    :param agent: Producer NIXL agent.
    :param notification_id: Exact notification payload.
    :param timeout_seconds: Maximum wait duration.
    :returns: Whether the notification arrived before the deadline.
    """
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        notifications = agent.get_new_notifs(backends=[_UCX_BACKEND])
        if any(
            message == notification_id
            for messages in notifications.values()
            for message in messages
        ):
            return True
        time.sleep(0.001)
    return False


def run_producer(
    *,
    config_path: Path,
    arm_name: str,
    run_id: str,
    rank: int,
    artifact_directory: Path,
    expected_cuda_visibility: str,
) -> None:
    """Run one independent P-rank NIXL agent and source registration.

    :param config_path: Authenticated rig configuration.
    :param arm_name: Fresh-process transport arm.
    :param run_id: Campaign UUID.
    :param rank: P tensor-parallel rank.
    :param artifact_directory: Immutable run artifact directory.
    :param expected_cuda_visibility: Preflighted producer UUID roster.
    """
    config = load_config(config_path)
    arm = config.transport_arm(arm_name)
    _assert_role_visibility(expected_cuda_visibility)
    if rank < 0 or rank >= len(config.producer_devices):
        raise ValueError(f"producer rank is out of range: {rank}")
    logical_device = rank
    torch.cuda.set_device(logical_device)
    agent = _agent(f"micro-p-{run_id}-r{rank}", config)
    producer_engine_id = f"micro-p-{run_id}"
    region_tensors = tuple(
        torch.empty(
            config.source_block_count * region.row_bytes,
            dtype=torch.uint8,
            device=f"cuda:{logical_device}",
        )
        for region in config.regions
    )
    registration = _register_regions(agent, region_tensors)
    attestation = collect_process_attestation(
        agent,
        role=f"producer:{rank}",
        physical_device=config.producer_devices[rank],
        logical_device=logical_device,
    )
    artifact_directory.mkdir(parents=True, exist_ok=True)
    (artifact_directory / f"producer-{rank}-attestation.json").write_text(
        json.dumps(attestation, indent=2, sort_keys=True)
    )
    write_process_maps(artifact_directory / f"producer-{rank}-proc-maps.txt")
    hello_payload = HelloPayload(
        agent_metadata=base64.b64encode(agent.get_agent_metadata()).decode(),
        base_addresses=tuple(tensor.data_ptr() for tensor in region_tensors),
        logical_device=logical_device,
        region_bytes=tuple(tensor.numel() for tensor in region_tensors),
        attestation=attestation,
    )
    server = _control_server(config, rank)
    completed_cleanly = False
    try:
        connection, _ = server.accept()
        with JsonChannel(
            connection,
            run_id=run_id,
            config_fingerprint=config.fingerprint,
            transport_arm=arm.name,
            local_role="producer",
            local_rank=rank,
            remote_role="consumer",
            remote_rank=0,
            timeout_seconds=float(config.transfer_timeout_seconds),
        ) as channel:
            channel.send(hello_payload, iteration=-1)
            global_iteration = 0
            for scenario in config.scenarios:
                plan = _plan(config_path, config, scenario.name)
                for scenario_iteration in range(scenario.iterations):
                    prepare = channel.receive(
                        PreparePayload, iteration=global_iteration
                    )
                    if prepare.scenario != scenario.name:
                        raise RuntimeError(
                            "consumer scenario order differs from producer"
                        )
                    if prepare.scenario_iteration != scenario_iteration:
                        raise RuntimeError("consumer scenario iteration is stale")
                    producer_request_id = prepare.producer_request_id
                    child_request_id = prepare.child_request_id
                    notification_id = base64.b64decode(prepare.notification_id)
                    context = _observation_context(
                        run_id=run_id,
                        arm_name=arm.name,
                        producer_engine_id=producer_engine_id,
                        iteration=global_iteration,
                        producer_request_id=producer_request_id,
                        child_request_id=child_request_id,
                    )
                    if plan.logical_position_count > 0:
                        fill_source_rows(
                            config=config,
                            plan=plan,
                            source_rank=rank,
                            iteration=global_iteration,
                            regions=region_tensors,
                        )
                        verify_source_rows(
                            config=config,
                            plan=plan,
                            source_rank=rank,
                            iteration=global_iteration,
                            regions=region_tensors,
                        )
                        before = source_observations(
                            config=config,
                            plan=plan,
                            context=context,
                            source_rank=rank,
                            regions=region_tensors,
                        )
                    else:
                        before = []
                    write_observations(
                        artifact_directory / f"producer-{rank}-source.jsonl", before
                    )
                    digest_rows = tuple(
                        DigestRow.from_json(row) for row in compact_digests(before)
                    )
                    channel.send(
                        PreparedPayload(
                            digests=digest_rows,
                            observation_count=len(digest_rows),
                        ),
                        iteration=global_iteration,
                    )
                    completed = channel.receive(
                        CompletePayload, iteration=global_iteration
                    )
                    if not completed.success:
                        raise RuntimeError("consumer reported transfer failure")
                    if completed.notification_id != prepare.notification_id:
                        raise RuntimeError("completion notification identity changed")
                    notification_seen = _await_notification(
                        agent, notification_id, config.transfer_timeout_seconds
                    )
                    if plan.logical_position_count > 0:
                        verify_source_rows(
                            config=config,
                            plan=plan,
                            source_rank=rank,
                            iteration=global_iteration,
                            regions=region_tensors,
                        )
                        after = source_observations(
                            config=config,
                            plan=plan,
                            context=context,
                            source_rank=rank,
                            regions=region_tensors,
                        )
                    else:
                        after = []
                    matches_before = compact_digests(before) == compact_digests(after)
                    channel.send(
                        SourcePostPayload(
                            matches_pre=matches_before,
                            notification_seen=notification_seen,
                        ),
                        iteration=global_iteration,
                    )
                    if not matches_before or not notification_seen:
                        raise RuntimeError("producer post-transfer verification failed")
                    global_iteration += 1
            channel.receive(StopPayload, iteration=-1)
            channel.send(StoppedPayload(), iteration=-1)
            completed_cleanly = True
    finally:
        server.close()
        if completed_cleanly:
            agent.deregister_memory(registration, backends=[_UCX_BACKEND])


def _prepared_descriptors(
    *,
    base_addresses: list[int],
    row_bytes: list[int],
    block_count: int,
    rank_count: int,
    rank: int,
    device_id: int,
    destination: bool,
) -> np.ndarray:
    """Construct the production-shaped stock prepared descriptor list.

    :param base_addresses: Region registration base addresses.
    :param row_bytes: Producer row bytes for each region.
    :param block_count: Physical rows per region.
    :param rank_count: Producer TP rank count.
    :param rank: Producer rank represented by this list.
    :param device_id: Logical CUDA descriptor device.
    :param destination: Whether to construct TP1 destination descriptors.
    :returns: Packed address, length, and device descriptor array.
    """
    descriptor_count = len(base_addresses) * _PREPARED_DESCRIPTOR_PLANES * block_count
    descriptors: np.ndarray = np.empty((descriptor_count, 3), dtype=np.uint64)
    cursor = 0
    block_indices: np.ndarray = np.arange(block_count, dtype=np.uint64)
    for base_address, remote_row_bytes in zip(base_addresses, row_bytes, strict=True):
        chunk = remote_row_bytes // 2
        if destination:
            local_row_bytes = remote_row_bytes * rank_count
            for plane in (0, 1):
                plane_offset = plane * rank_count * chunk + rank * chunk
                addresses: np.ndarray = (
                    base_address + block_indices * local_row_bytes + plane_offset
                )
                descriptors[cursor : cursor + block_count, 0] = addresses
                descriptors[cursor : cursor + block_count, 1] = chunk
                descriptors[cursor : cursor + block_count, 2] = device_id
                cursor += block_count
        else:
            for plane in (0, 1):
                addresses = (
                    base_address + block_indices * remote_row_bytes + plane * chunk
                )
                descriptors[cursor : cursor + block_count, 0] = addresses
                descriptors[cursor : cursor + block_count, 1] = chunk
                descriptors[cursor : cursor + block_count, 2] = device_id
                cursor += block_count
    return descriptors


def _prepare_handshake_side_state(
    *,
    agent: nixl_agent,
    config: RigConfig,
    destinations: tuple[torch.Tensor, ...],
    remote_agents: list[str],
    remote_bases: list[list[int]],
    remote_devices: list[int],
) -> tuple[tuple[nixl_prepped_dlist_handle, nixl_prepped_dlist_handle], ...]:
    """Create and retain the stock prepared dlists bypassed by raw coalescing."""
    handles: list[tuple[nixl_prepped_dlist_handle, nixl_prepped_dlist_handle]] = []
    destination_bases = [tensor.data_ptr() for tensor in destinations]
    row_bytes = [region.row_bytes for region in config.regions]
    rank_count = len(config.producer_devices)
    for rank in range(rank_count):
        local_descriptors = _prepared_descriptors(
            base_addresses=destination_bases,
            row_bytes=row_bytes,
            block_count=config.source_block_count,
            rank_count=rank_count,
            rank=rank,
            device_id=destinations[0].get_device(),
            destination=True,
        )
        remote_descriptors = _prepared_descriptors(
            base_addresses=remote_bases[rank],
            row_bytes=row_bytes,
            block_count=config.source_block_count,
            rank_count=rank_count,
            rank=rank,
            device_id=remote_devices[rank],
            destination=False,
        )
        local = agent.prep_xfer_dlist(
            "NIXL_INIT_AGENT", local_descriptors, "VRAM", backends=[_UCX_BACKEND]
        )
        remote = agent.prep_xfer_dlist(
            remote_agents[rank],
            remote_descriptors,
            "VRAM",
            backends=[_UCX_BACKEND],
        )
        handles.append((local, remote))
    return tuple(handles)


class VictimCanary:
    """Run deterministic GEMM while guarding live inputs, output, and redzones."""

    def __init__(self, config: RigConfig, device: torch.device) -> None:
        """Allocate deterministic compute inputs, output, and redzones.

        :param config: Complete rig configuration.
        :param device: Consumer CUDA device.
        """
        order = config.victim.matrix_order
        generator = torch.Generator(device=device)
        generator.manual_seed(0x5EED37B)
        self._a = torch.rand((order, order), generator=generator, device=device)
        self._b = torch.rand((order, order), generator=generator, device=device)
        self._a_reference = self._a.clone()
        self._b_reference = self._b.clone()
        self._output = torch.empty_like(self._a)
        self._expected = torch.mm(self._a, self._b)
        self._redzones = tuple(
            torch.full(
                (config.victim.redzone_bytes,),
                0x5A + index,
                dtype=torch.uint8,
                device=device,
            )
            for index in range(5)
        )
        self._rounds = config.victim.rounds
        self._stream = torch.cuda.Stream(device=device)
        self._start = torch.cuda.Event(enable_timing=True)
        self._end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(device)

    def launch(self) -> None:
        """Queue the deterministic victim workload on its private stream."""
        with torch.cuda.stream(self._stream):
            self._start.record(self._stream)
            for _ in range(self._rounds):
                torch.mm(self._a, self._b, out=self._output)
            self._end.record(self._stream)
        self._start.synchronize()
        if self._end.query():
            raise RuntimeError("victim completed before transport could be posted")

    @property
    def incomplete(self) -> bool:
        """Return whether the victim end event is still pending.

        :returns: ``True`` while deterministic compute remains in flight.
        """
        return not self._end.query()

    def finish_and_verify(self) -> float:
        """Wait for victim completion and verify every deterministic canary.

        :returns: Victim CUDA duration in milliseconds.
        """
        self._end.synchronize()
        if not torch.equal(self._a, self._a_reference):
            raise RuntimeError("victim input A changed during transport")
        if not torch.equal(self._b, self._b_reference):
            raise RuntimeError("victim input B changed during transport")
        if not torch.equal(self._output, self._expected):
            raise RuntimeError("victim GEMM output differs from baseline")
        for index, redzone in enumerate(self._redzones):
            if not torch.all(redzone == 0x5A + index).item():
                raise RuntimeError(f"victim redzone {index} changed during transport")
        return self._start.elapsed_time(self._end)


def _raw_descriptors(
    *,
    config: RigConfig,
    plan: TransferPlan,
    staging: torch.Tensor,
    staging_offset: int,
    rank: int,
    remote_bases: list[int],
    remote_device: int,
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    """Construct coalesced staging and source descriptors for one rank.

    :param config: Complete rig configuration.
    :param plan: Exact transfer plan.
    :param staging: Registered consumer staging tensor.
    :param staging_offset: Active allocation-generation byte offset.
    :param rank: Producer rank represented by the handle.
    :param remote_bases: Producer source registration bases.
    :param remote_device: Producer logical CUDA descriptor device.
    :returns: Local and remote raw descriptor lists.
    """
    local: list[tuple[int, int, int]] = []
    remote: list[tuple[int, int, int]] = []
    for region_index, region in enumerate(config.regions):
        region_layout = plan.transport.regions[region_index]
        local_region = (
            staging.data_ptr()
            + staging_offset
            + plan.transport.region_offset(rank, region_index)
        )
        for run in region_layout.runs:
            byte_count = run.position_count * region.row_bytes
            local.append(
                (
                    local_region + run.position_start * region.row_bytes,
                    byte_count,
                    staging.get_device(),
                )
            )
            remote.append(
                (
                    remote_bases[region_index] + run.remote_block_id * region.row_bytes,
                    byte_count,
                    remote_device,
                )
            )
    return local, remote


def _staging_guard_ranges(
    *, capacity: int, offset: int, size: int
) -> tuple[tuple[int, int], ...]:
    """Return adjacent registered ranges that native writes must not touch.

    :param capacity: Total registered staging bytes.
    :param offset: Active allocation offset.
    :param size: Active allocation bytes.
    :returns: Non-empty half-open guard ranges before and after the allocation.
    """
    before = (max(0, offset - _STAGING_GUARD_BYTES), offset)
    end = offset + size
    after = (end, min(capacity, end + _STAGING_GUARD_BYTES))
    return tuple((start, stop) for start, stop in (before, after) if stop > start)


def _fill_staging_guards(
    staging: torch.Tensor, ranges: tuple[tuple[int, int], ...], value: int
) -> None:
    """Seed adjacent staging guard ranges.

    :param staging: Full registered staging tensor.
    :param ranges: Half-open guard ranges.
    :param value: Byte sentinel for this allocation generation.
    """
    for start, stop in ranges:
        staging[start:stop].fill_(value)


def _verify_staging_guards(
    staging: torch.Tensor, ranges: tuple[tuple[int, int], ...], value: int
) -> None:
    """Verify adjacent staging guard ranges after native completion.

    :param staging: Full registered staging tensor.
    :param ranges: Half-open guard ranges.
    :param value: Expected byte sentinel.
    :raises RuntimeError: If any guard byte changed.
    """
    for start, stop in ranges:
        if not torch.all(staging[start:stop] == value).item():
            raise RuntimeError(f"staging guard changed outside [{start}, {stop})")


def _telemetry(agent: nixl_agent, handle: nixl_xfer_handle) -> dict[str, object]:
    """Capture backend selection and terminal transfer telemetry.

    :param agent: Consumer NIXL agent.
    :param handle: Terminal transfer handle.
    :returns: JSON-serializable backend and timing record.
    """
    telemetry = agent.get_xfer_telemetry(handle)
    return {
        "backend": agent.query_xfer_backend(handle),
        "start_time_us": telemetry.startTime,
        "post_duration_us": telemetry.postDuration,
        "transfer_duration_us": telemetry.xferDuration,
        "total_bytes": telemetry.totalBytes,
        "descriptor_count": telemetry.descCount,
    }


def _scatter(
    *,
    plan: TransferPlan,
    staging: torch.Tensor,
    staging_offset: int,
    destinations: tuple[torch.Tensor, ...],
    stream: torch.cuda.Stream | None = None,
) -> float:
    """Execute the production scatter implementation against rig tensors.

    :param plan: Exact canonical region-owned transfer plan.
    :param staging: Registered raw transport destination.
    :param staging_offset: Active allocation-generation byte offset.
    :param destinations: Registered TP1 destination regions.
    :param stream: Persistent CUDA stream used by the production scatter path.
    :returns: CUDA event duration in milliseconds, or zero for the CPU reference.
    """
    if staging.device.type == "cpu":
        scatter_coalesced_reference(
            staging,
            destinations,
            plan.transport,
            staging_base_offset_bytes=staging_offset,
        )
        return 0.0
    if staging.device.type != "cuda":
        raise ValueError("the micro-rig scatter requires CPU or CUDA tensors")
    if stream is None:
        raise ValueError("the CUDA micro-rig scatter requires a persistent stream")

    stream.wait_stream(torch.cuda.current_stream(staging.device))
    launch = launch_coalesced_scatter(
        staging,
        destinations,
        plan.transport,
        stream,
        staging_base_offset_bytes=staging_offset,
    )
    launch.completion_event.synchronize()
    return launch.gpu_duration_ms()


def _compare_compact(
    expected_rows: list[list[int | str]], actual_rows: list[list[int | str]]
) -> None:
    """Compare compact source and consumer leaves by exact identity.

    :param expected_rows: Source digest rows from all producer agents.
    :param actual_rows: Staging or destination digest rows.
    :raises RuntimeError: If identities duplicate, disappear, or change bytes.
    """
    expected = {tuple(row[:-1]): row[-1] for row in expected_rows}
    actual = {tuple(row[:-1]): row[-1] for row in actual_rows}
    if len(expected) != len(expected_rows) or len(actual) != len(actual_rows):
        raise RuntimeError("duplicate integrity identity in compact observations")
    if expected != actual:
        missing = list(expected.keys() - actual.keys())[:4]
        extra = list(actual.keys() - expected.keys())[:4]
        mismatched = [
            key
            for key in expected.keys() & actual.keys()
            if expected[key] != actual[key]
        ][:4]
        raise RuntimeError(
            f"integrity mismatch missing={missing} extra={extra} "
            f"mismatched={mismatched}"
        )


def run_consumer(
    *,
    config_path: Path,
    arm_name: str,
    run_id: str,
    artifact_directory: Path,
    expected_cuda_visibility: str,
    producer_cuda_visibility: str,
) -> None:
    """Run the TP1 D agent with exact live registrations and coalesced pulls.

    :param config_path: Immutable run-local configuration.
    :param arm_name: Fresh-process transport arm.
    :param run_id: Campaign UUID.
    :param artifact_directory: Fresh arm artifact directory.
    :param expected_cuda_visibility: Preflighted consumer GPU UUID.
    :param producer_cuda_visibility: Preflighted producer UUID roster.
    """
    config = load_config(config_path)
    arm = config.transport_arm(arm_name)
    _assert_role_visibility(expected_cuda_visibility)
    logical_device = 0
    torch.cuda.set_device(logical_device)
    agent = _agent(f"micro-d-{run_id}", config)
    rank_count = len(config.producer_devices)

    destinations = tuple(
        torch.empty(
            config.source_block_count * rank_count * region.row_bytes,
            dtype=torch.uint8,
            device="cuda:0",
        )
        for region in config.regions
    )
    destination_registration = _register_regions(agent, destinations)

    channels: list[JsonChannel] = []
    remote_agents: list[str] = []
    remote_bases: list[list[int]] = []
    remote_devices: list[int] = []
    remote_metadata_sha256: list[str] = []
    producer_attestations: list[dict[str, object]] = []
    for rank in range(rank_count):
        channel = JsonChannel(
            _connect(config, rank),
            run_id=run_id,
            config_fingerprint=config.fingerprint,
            transport_arm=arm.name,
            local_role="consumer",
            local_rank=0,
            remote_role="producer",
            remote_rank=rank,
            timeout_seconds=float(config.transfer_timeout_seconds),
        )
        hello = channel.receive(HelloPayload, iteration=-1)
        _validate_producer_hello(config, rank, hello, producer_cuda_visibility)
        metadata = base64.b64decode(hello.agent_metadata)
        remote_agents.append(
            canonicalize_nixl_agent_name(agent.add_remote_agent(metadata))
        )
        remote_bases.append(list(hello.base_addresses))
        remote_devices.append(hello.logical_device)
        remote_metadata_sha256.append(hashlib.sha256(metadata).hexdigest())
        producer_attestations.append(hello.attestation)
        channels.append(channel)

    prepared_handles = _prepare_handshake_side_state(
        agent=agent,
        config=config,
        destinations=destinations,
        remote_agents=remote_agents,
        remote_bases=remote_bases,
        remote_devices=remote_devices,
    )
    staging = torch.empty(
        config.staging_capacity_mib * 1024 * 1024,
        dtype=torch.uint8,
        device="cuda:0",
    )
    staging_registration = _register_regions(agent, (staging,))
    allocator = StagingRangeAllocator(capacity=staging.numel())
    victim = VictimCanary(config, torch.device("cuda:0"))
    scatter_stream = torch.cuda.Stream(device=staging.device)
    attestation = collect_process_attestation(
        agent,
        role="consumer:0",
        physical_device=config.consumer_device,
        logical_device=logical_device,
    )
    artifact_directory.mkdir(parents=True, exist_ok=True)
    (artifact_directory / "consumer-attestation.json").write_text(
        json.dumps(
            {
                "consumer": attestation,
                "producers": producer_attestations,
                "prepared_handle_pairs": len(prepared_handles),
            },
            indent=2,
            sort_keys=True,
        )
    )
    (artifact_directory / "runtime-registrations.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "transport_arm": arm.name,
                "config_fingerprint": config.fingerprint,
                "destination": [
                    {
                        "region_index": region_index,
                        "base_address": destination.data_ptr(),
                        "byte_length": destination.numel(),
                        "logical_device": destination.get_device(),
                    }
                    for region_index, destination in enumerate(destinations)
                ],
                "staging": {
                    "base_address": staging.data_ptr(),
                    "byte_length": staging.numel(),
                    "logical_device": staging.get_device(),
                },
                "producers": [
                    {
                        "rank": rank,
                        "remote_agent": remote_agents[rank],
                        "agent_metadata_sha256": remote_metadata_sha256[rank],
                        "base_addresses": remote_bases[rank],
                        "region_bytes": [
                            config.source_block_count * region.row_bytes
                            for region in config.regions
                        ],
                        "logical_device": remote_devices[rank],
                    }
                    for rank in range(rank_count)
                ],
                "prepared_handle_pairs": len(prepared_handles),
            },
            indent=2,
            sort_keys=True,
        )
    )
    write_process_maps(artifact_directory / "consumer-proc-maps.txt")

    iteration_records: list[dict[str, object]] = []
    global_iteration = 0
    completed_cleanly = False
    try:
        for scenario in config.scenarios:
            plan = _plan(config_path, config, scenario.name)
            for region in plan.transport.regions:
                scatter_targets = tuple(
                    (position.local_block_id, position.destination_half)
                    for position in region.positions
                )
                if len(set(scatter_targets)) != len(scatter_targets):
                    raise RuntimeError(
                        "production lane forbids duplicate local scatter targets "
                        f"within region {region.ownership.region_index}"
                    )
            for scenario_iteration in range(scenario.iterations):
                producer_request_id = f"p-{run_id}-{global_iteration}"
                child_request_id = f"d-{run_id}-{global_iteration}"
                notification_id = (
                    f"micro-rig:{run_id}:{arm.name}:{global_iteration}"
                ).encode()
                prepare_payload = PreparePayload(
                    scenario=scenario.name,
                    scenario_iteration=scenario_iteration,
                    producer_request_id=producer_request_id,
                    child_request_id=child_request_id,
                    notification_id=base64.b64encode(notification_id).decode(),
                )
                for channel in channels:
                    channel.send(prepare_payload, iteration=global_iteration)
                source_rows: list[list[int | str]] = []
                for channel in channels:
                    prepared = channel.receive(
                        PreparedPayload, iteration=global_iteration
                    )
                    source_rows.extend(row.to_json() for row in prepared.digests)

                if plan.logical_position_count == 0:
                    for remote_agent in remote_agents:
                        agent.send_notif(
                            remote_agent, notification_id, backend=_UCX_BACKEND
                        )
                    for channel in channels:
                        channel.send(
                            CompletePayload(
                                success=True,
                                notification_id=prepare_payload.notification_id,
                            ),
                            iteration=global_iteration,
                        )
                    iteration_records.append(
                        {
                            "iteration": global_iteration,
                            "scenario": scenario.name,
                            "scenario_iteration": scenario_iteration,
                            "producer_request_id": producer_request_id,
                            "child_request_id": child_request_id,
                            "notification_id": prepare_payload.notification_id,
                            "has_content_evidence": False,
                            "evidence_status": "non_evidentiary_zero_byte",
                        }
                    )
                else:
                    staging_offset = (
                        scenario.staging_offsets_mib[
                            scenario_iteration % len(scenario.staging_offsets_mib)
                        ]
                        * 1024
                        * 1024
                    )
                    ownership = allocator.create_plan(
                        owner_id=f"rig:{global_iteration}",
                        request_id=child_request_id,
                        generation=global_iteration,
                        offset=staging_offset,
                        layout=plan.transport,
                        layout_duration_seconds=plan.layout_duration_seconds,
                        remote_engine_id=f"micro-p-{run_id}",
                    )
                    if ownership is None:
                        raise StagingSafetyError(
                            "configured staging generation overlaps a live owner"
                        )
                    staging[staging_offset : staging_offset + plan.staging_bytes].fill_(
                        (0xA5 + global_iteration) & 0xFF
                    )
                    guard_ranges = _staging_guard_ranges(
                        capacity=staging.numel(),
                        offset=staging_offset,
                        size=plan.staging_bytes,
                    )
                    guard_value = (0xD0 + global_iteration) & 0xFF
                    _fill_staging_guards(staging, guard_ranges, guard_value)
                    destination_canary = (0x3C + global_iteration) & 0xFF
                    fill_destination_canary_rows(
                        config=config,
                        plan=plan,
                        destinations=destinations,
                        value=destination_canary,
                    )
                    handles: list[nixl_xfer_handle] = []
                    handle_records: list[dict[str, object]] = []
                    victim.launch()
                    transfer_start = torch.cuda.Event(enable_timing=True)
                    transfer_done = torch.cuda.Event(enable_timing=True)
                    transfer_start.record()
                    transfer_start.synchronize()
                    saw_proc_during_victim = False
                    current_rank: int | None = None
                    try:
                        for rank in range(rank_count):
                            current_rank = rank
                            ownership.begin_prepare(rank)
                            local_raw, remote_raw = _raw_descriptors(
                                config=config,
                                plan=plan,
                                staging=staging,
                                staging_offset=staging_offset,
                                rank=rank,
                                remote_bases=remote_bases[rank],
                                remote_device=remote_devices[rank],
                            )
                            local = agent.get_xfer_descs(local_raw, "VRAM")
                            remote = agent.get_xfer_descs(remote_raw, "VRAM")
                            handle = agent.initialize_xfer(
                                "READ",
                                local,
                                remote,
                                remote_agents[rank],
                                notification_id,
                                backends=[_UCX_BACKEND],
                            )
                            handles.append(handle)
                            ownership.attach_handle(rank, handle)
                            ownership.begin_post(rank)
                            status = agent.transfer(handle)
                            ownership.record_post_result(rank, status)
                            state = ownership.slots[rank].state
                            if state is HandleState.PROC and victim.incomplete:
                                saw_proc_during_victim = True
                            if ownership.operation_failed:
                                ownership.seal_posting()
                                raise StagingSafetyError(
                                    "NIXL post failed; staging remains tombstoned: "
                                    f"{ownership.describe()}"
                                )
                    except Exception as error:
                        stacktrace = traceback.format_exc()
                        if current_rank is not None:
                            state = ownership.slots[current_rank].state
                            if state is HandleState.PREPARING:
                                ownership.record_prepare_failure(
                                    current_rank,
                                    f"rank {current_rank} preparation raised",
                                )
                            elif state is HandleState.POSTING:
                                ownership.record_post_exception(
                                    current_rank,
                                    f"rank {current_rank} post raised",
                                )
                        if ownership.posting_sealed is False:
                            ownership.seal_posting()
                        if isinstance(error, StagingSafetyError):
                            raise
                        raise StagingSafetyError(
                            "NIXL submission failed; staging remains owned\n"
                            f"{stacktrace}"
                        ) from error
                    ownership.seal_posting()

                    deadline = time.monotonic() + config.transfer_timeout_seconds
                    while any(
                        slot.state is HandleState.PROC
                        for slot in ownership.slots.values()
                    ):
                        for rank, handle in enumerate(handles):
                            if ownership.slots[rank].state is not HandleState.PROC:
                                continue
                            try:
                                status = agent.check_xfer_state(handle)
                            except Exception as error:
                                ownership.record_query_exception(
                                    rank,
                                    f"rank {rank} status query raised",
                                )
                                raise StagingSafetyError(
                                    "NIXL status query raised; staging remains "
                                    f"tombstoned\n{traceback.format_exc()}"
                                ) from error
                            ownership.record_query_result(rank, status)
                            state = ownership.slots[rank].state
                            if state is HandleState.PROC and victim.incomplete:
                                saw_proc_during_victim = True
                            if ownership.operation_failed:
                                raise StagingSafetyError(
                                    "NIXL status failed; staging remains tombstoned: "
                                    f"{ownership.describe()}"
                                )
                        if time.monotonic() >= deadline and any(
                            slot.state is HandleState.PROC
                            for slot in ownership.slots.values()
                        ):
                            ownership.tombstone("NIXL transfer deadline expired")
                            raise StagingSafetyError(
                                "NIXL transfer deadline expired; staging remains "
                                "tombstoned"
                            )
                        time.sleep(0.0001)

                    for rank, handle in enumerate(handles):
                        if ownership.slots[rank].state is not HandleState.DONE:
                            raise StagingSafetyError(
                                "non-DONE native handle reached cleanup; staging "
                                "remains tombstoned"
                            )
                        handle_records.append(_telemetry(agent, handle))
                        ownership.record_native_telemetry(
                            rank,
                            handle_records[-1],
                        )
                        if handle_records[-1]["backend"] != _UCX_BACKEND:
                            raise RuntimeError("NIXL selected a non-UCX backend")
                        try:
                            agent.release_xfer_handle(handle)
                        except Exception as error:
                            raise StagingSafetyError(
                                "DONE handle resource cleanup failed; the rig process "
                                f"must exit\n{traceback.format_exc()}"
                            ) from error
                        ownership.mark_native_released(rank)

                    transfer_done.record()
                    transfer_done.synchronize()
                    victim_duration_ms = victim.finish_and_verify()
                    _verify_staging_guards(staging, guard_ranges, guard_value)
                    verify_destination_canary_rows(
                        config=config,
                        plan=plan,
                        destinations=destinations,
                        value=destination_canary,
                    )
                    if not saw_proc_during_victim:
                        raise RuntimeError(
                            "INVALID arm: no PROC handle was observed while victim ran"
                        )
                    ownership.begin_device_read()
                    scatter_gpu_duration_ms = _scatter(
                        plan=plan,
                        staging=staging,
                        staging_offset=staging_offset,
                        destinations=destinations,
                        stream=scatter_stream,
                    )
                    verify_staging_rows(
                        config=config,
                        plan=plan,
                        iteration=global_iteration,
                        staging=staging,
                        staging_offset=staging_offset,
                    )
                    verify_destination_rows(
                        config=config,
                        plan=plan,
                        iteration=global_iteration,
                        destinations=destinations,
                    )
                    context = _observation_context(
                        run_id=run_id,
                        arm_name=arm.name,
                        producer_engine_id=f"micro-p-{run_id}",
                        iteration=global_iteration,
                        producer_request_id=producer_request_id,
                        child_request_id=child_request_id,
                    )
                    staged = staging_observations(
                        config=config,
                        plan=plan,
                        context=context,
                        staging=staging,
                        staging_offset=staging_offset,
                    )
                    committed = destination_observations(
                        config=config,
                        plan=plan,
                        context=context,
                        destinations=destinations,
                    )
                    staged_compact = compact_digests(staged)
                    committed_compact = compact_digests(committed)
                    _compare_compact(source_rows, staged_compact)
                    _compare_compact(source_rows, committed_compact)
                    write_observations(
                        artifact_directory / "consumer-staging.jsonl", staged
                    )
                    write_observations(
                        artifact_directory / "consumer-destination.jsonl", committed
                    )
                    torch.cuda.synchronize()
                    ownership.mark_device_quiescent()
                    allocator.release(ownership)
                    ownership_snapshot = ownership.snapshot().to_dict()
                    for channel in channels:
                        channel.send(
                            CompletePayload(
                                success=True,
                                notification_id=prepare_payload.notification_id,
                            ),
                            iteration=global_iteration,
                        )
                    iteration_records.append(
                        {
                            "iteration": global_iteration,
                            "scenario": scenario.name,
                            "scenario_iteration": scenario_iteration,
                            "producer_request_id": producer_request_id,
                            "child_request_id": child_request_id,
                            "notification_id": prepare_payload.notification_id,
                            "has_content_evidence": True,
                            "staging_offset": staging_offset,
                            "staging_guard_ranges": guard_ranges,
                            "staging_guard_value": guard_value,
                            "destination_canary_value": destination_canary,
                            "source_observations": len(source_rows),
                            "staging_observations": len(staged),
                            "destination_observations": len(committed),
                            "victim_duration_ms": victim_duration_ms,
                            "scatter_gpu_duration_ms": scatter_gpu_duration_ms,
                            "transfer_bracket_ms": transfer_start.elapsed_time(
                                transfer_done
                            ),
                            "saw_proc_during_victim": saw_proc_during_victim,
                            "handles": handle_records,
                            "staging_ownership": ownership_snapshot,
                            "evidence_status": "content_digest",
                        }
                    )

                for channel in channels:
                    source_post = channel.receive(
                        SourcePostPayload, iteration=global_iteration
                    )
                    if not source_post.matches_pre or not source_post.notification_seen:
                        raise RuntimeError("producer post-transfer evidence failed")
                global_iteration += 1
        for channel in channels:
            channel.send(StopPayload(), iteration=-1)
        for channel in channels:
            channel.receive(StoppedPayload, iteration=-1)
        completed_cleanly = True
    finally:
        (artifact_directory / "iterations.json").write_text(
            json.dumps(iteration_records, indent=2, sort_keys=True)
        )
        if completed_cleanly:
            for local_handle, remote_handle in prepared_handles:
                agent.release_dlist_handle(local_handle)
                agent.release_dlist_handle(remote_handle)
            agent.deregister_memory(staging_registration, backends=[_UCX_BACKEND])
            agent.deregister_memory(destination_registration, backends=[_UCX_BACKEND])
        for channel in channels:
            channel.close()
