import errno
import fcntl
import hashlib
import json
import math
import os
import re
import signal
import stat
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType

from tools.gemma4_pd.lane_lifecycle.schema import (
    SCHEMA_VERSION,
    GpuProcess,
    LaneSnapshot,
    MetricRecord,
    ProcessRecord,
    ProtectedGroup,
    ServiceRecord,
    SnapshotValidationError,
    SocketRecord,
    write_json_file,
)
from tools.gemma4_pd.lane_lifecycle.system import (
    Host,
    HostInspectionError,
    LinuxHost,
    LiveProcess,
    ProcessGroupNotFoundError,
    ProcessNotFoundError,
    decode_bytes,
    discover_launcher_files,
    encode_bytes,
    endpoint_port,
    environment_subset,
    listener_owner,
    materialize_process,
    record_instance_identity,
    record_launch_identity,
    sha256_file,
)

AUTHORITATIVE_STORM37B_SHA256 = (
    "e4f4e7ff7bd1bcee0de3c8413d2915cf5f3b9b0caafc7b65fe4053e5069d732a"
)
DEFAULT_STORM37B_ARCHIVE = Path("/data/colleague/evidence/storm37b-20260712.tar.zst")
PRIVATE_ARTIFACT_ROOT = Path("/data/colleague/private/lane-lifecycle")
EXPERIMENT_LANE_LOCK = Path("/data/colleague/locks/gemma4-experiment-lane.lock")
_METRIC_PATTERN = re.compile(
    rb"^(vllm:num_requests_(?:running|waiting))(?:\{[^}]*\})?\s+([^\s]+)(?:\s+\d+)?$"
)


class LifecycleError(RuntimeError):
    """Indicate that a lifecycle safety precondition was not satisfied."""


class ExperimentLaneLock:
    """Hold the one host-wide experiment-lane ownership lock."""

    _descriptor: int | None

    def __init__(self) -> None:
        """Initialize an unheld lock handle."""
        self._descriptor = None

    def __enter__(self) -> "ExperimentLaneLock":
        """Acquire the host-wide lock without waiting.

        :returns: Held lock handle.
        :raises LifecycleError: If another lifecycle or campaign owns the lane.
        """
        directory = EXPERIMENT_LANE_LOCK.parent
        _reject_symlink_components(directory.absolute())
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        if directory.is_symlink() or not directory.is_dir():
            raise LifecycleError(f"unsafe experiment lock directory {directory}")
        try:
            descriptor = os.open(
                EXPERIMENT_LANE_LOCK,
                os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW,
                0o600,
            )
        except OSError as error:
            raise LifecycleError(
                f"cannot open experiment-lane lock: {error}"
            ) from error
        try:
            status = os.fstat(descriptor)
            if (
                not stat.S_ISREG(status.st_mode)
                or stat.S_IMODE(status.st_mode) != 0o600
            ):
                raise LifecycleError("experiment-lane lock must be a mode-0600 file")
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.ftruncate(descriptor, 0)
            _write_all(
                descriptor,
                f"pid={os.getpid()} acquired_at={_utc_now()}\n".encode(),
            )
            os.fsync(descriptor)
        except OSError as error:
            os.close(descriptor)
            if error.errno in {errno.EACCES, errno.EAGAIN}:
                raise LifecycleError(
                    f"experiment lane is already locked at {EXPERIMENT_LANE_LOCK}"
                ) from error
            raise LifecycleError(
                f"cannot acquire experiment-lane lock: {error}"
            ) from error
        except LifecycleError:
            os.close(descriptor)
            raise
        self._descriptor = descriptor
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release the host-wide lock.

        :param exception_type: Active exception type, when any.
        :param exception: Active exception, when any.
        :param traceback: Active traceback, when any.
        """
        del exception_type, exception, traceback
        if self._descriptor is None:
            return
        fcntl.flock(self._descriptor, fcntl.LOCK_UN)
        os.close(self._descriptor)
        self._descriptor = None


@dataclass(frozen=True, slots=True)
class ServiceSpec:
    """Define one canonical experiment-lane endpoint.

    :ivar role: Canonical role name.
    :ivar port: Exact listener port.
    :ivar health_path: Readiness endpoint, or ``None`` for listener readiness.
    :ivar gpu_indices: Exact GPUs owned by the service group.
    """

    role: str
    port: int
    health_path: str | None
    gpu_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RuntimeServiceState:
    """Classify one lane role against its captured launch contract.

    :ivar role: Canonical role name.
    :ivar status: ``captured``, ``equivalent``, or ``absent``.
    :ivar root: Current listener root when running.
    :ivar group_members: Stable current group when running.
    """

    role: str
    status: str
    root: LiveProcess | None
    group_members: tuple[LiveProcess, ...]


SERVICE_SPECS = (
    ServiceSpec("p", 8810, "/health", (0, 1, 2, 3)),
    ServiceSpec("d1", 8811, "/health", (4,)),
    ServiceSpec("proxy", 8812, None, ()),
    ServiceSpec("router", 8813, "/health", ()),
    ServiceSpec("d2", 8815, "/health", (5,)),
)
RESTORE_ORDER = ("p", "d1", "d2", "router", "proxy")
STOP_PHASES = (("router", "proxy"), ("d1", "d2"), ("p",))


class OperationJournal:
    """Persist lifecycle decisions before and after every mutation."""

    _descriptor: int
    directory: Path

    def __init__(self, artifact_directory: Path, operation: str) -> None:
        """Create a private operation journal.

        :param artifact_directory: Loaded snapshot directory.
        :param operation: Operation name.
        """
        operations = artifact_directory / "operations"
        operations.mkdir(mode=0o700, exist_ok=True)
        _require_directory_mode(operations, 0o700)
        timestamp = _utc_now().replace(":", "").replace("-", "")
        self.directory = operations / f"{timestamp}-{operation}-{os.getpid()}"
        self.directory.mkdir(mode=0o700)
        self._descriptor = os.open(
            self.directory / "events.jsonl",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
            0o600,
        )
        _fsync_directory(self.directory)
        _fsync_directory(operations)

    def record(self, event: str, **details: object) -> None:
        """Append and durably commit one operation event.

        :param event: Stable event name.
        :param details: JSON-compatible event fields.
        """
        payload = {
            "at_utc": _utc_now(),
            "event": event,
            "details": details,
        }
        line = (
            json.dumps(payload, sort_keys=True, ensure_ascii=True, allow_nan=False)
            + "\n"
        ).encode()
        _write_all(self._descriptor, line)
        os.fsync(self._descriptor)

    def close(self) -> None:
        """Close and durably publish the operation journal."""
        os.fsync(self._descriptor)
        os.close(self._descriptor)
        _fsync_directory(self.directory)


def parse_quiescence_metrics(body: bytes) -> tuple[float, float]:
    """Parse exact vLLM running and waiting request gauges.

    :param body: Prometheus exposition body.
    :returns: Running and waiting series sums.
    :raises LifecycleError: If either metric is absent or unsafe.
    """
    values: dict[bytes, list[float]] = {
        b"vllm:num_requests_running": [],
        b"vllm:num_requests_waiting": [],
    }
    for line in body.splitlines():
        match = _METRIC_PATTERN.fullmatch(line.strip())
        if match is None:
            continue
        try:
            value = float(match.group(2))
        except ValueError as error:
            raise LifecycleError(f"invalid Prometheus metric value {line!r}") from error
        if not math.isfinite(value) or value < 0.0:
            raise LifecycleError(f"unsafe Prometheus metric value {line!r}")
        values[match.group(1)].append(value)
    if any(len(series) == 0 for series in values.values()):
        raise LifecycleError("running/waiting request metrics are missing")
    return sum(values[b"vllm:num_requests_running"]), sum(
        values[b"vllm:num_requests_waiting"]
    )


def capture(
    artifact_directory: Path,
    *,
    storm_archive: Path = DEFAULT_STORM37B_ARCHIVE,
    host: Host | None = None,
) -> LaneSnapshot:
    """Capture and seal the lane while holding its host-wide lock.

    :param artifact_directory: New private snapshot directory.
    :param storm_archive: Authoritative storm37b archive.
    :param host: Host provider used by tests.
    :returns: Sealed lifecycle snapshot.
    """
    with ExperimentLaneLock():
        return _capture_locked(
            artifact_directory,
            storm_archive=storm_archive,
            host=host,
        )


def _capture_locked(
    artifact_directory: Path,
    *,
    storm_archive: Path,
    host: Host | None,
) -> LaneSnapshot:
    """Capture and seal a restorable experiment-lane snapshot.

    :param artifact_directory: New private snapshot directory.
    :param storm_archive: Authoritative storm37b archive.
    :param host: Host provider used by tests.
    :returns: Sealed lifecycle snapshot.
    :raises LifecycleError: If any state is ambiguous or unsafe.
    """
    selected_host = LinuxHost() if host is None else host
    artifact_directory = _create_private_artifact_directory(artifact_directory)
    for relative in ("executables", "metrics"):
        directory = artifact_directory / relative
        directory.mkdir(mode=0o700)
        _fsync_directory(artifact_directory)

    tool_identity = selected_host.tool_identity()

    listeners_before = selected_host.listeners()
    roots: dict[str, LiveProcess] = {}
    groups: dict[str, tuple[LiveProcess, ...]] = {}
    for spec in SERVICE_SPECS:
        root = selected_host.process(listener_owner(listeners_before, spec.port))
        if root.pid != root.pgid or root.pid != root.sid:
            raise LifecycleError(
                f"{spec.role} listener PID {root.pid} is not a session/group leader"
            )
        roots[spec.role] = root
        groups[spec.role] = _stable_group_snapshot(selected_host, root.pgid)
    if len({root.pid for root in roots.values()}) != len(SERVICE_SPECS):
        raise LifecycleError("experiment service listener PIDs overlap")
    if len({root.pgid for root in roots.values()}) != len(SERVICE_SPECS):
        raise LifecycleError("experiment service process groups overlap")

    gpu_devices, gpu_processes = selected_host.gpu_state()
    gpu_pgids: dict[int, int] = {}
    for process in gpu_processes:
        gpu_pgids[process.pid] = selected_host.process(process.pid).pgid
    _validate_live_gpu_ownership(groups, gpu_processes, gpu_pgids)

    protected_groups_live: dict[int, tuple[LiveProcess, ...]] = {}
    protected_gpus_by_pgid: dict[int, set[int]] = {}
    for process in gpu_processes:
        if process.gpu_index < 6:
            continue
        pgid = gpu_pgids[process.pid]
        protected_gpus_by_pgid.setdefault(pgid, set()).add(process.gpu_index)
    production_proxy = selected_host.process(listener_owner(listeners_before, 8000))
    if production_proxy.pgid in {root.pgid for root in roots.values()}:
        raise LifecycleError("production proxy :8000 overlaps an experiment group")
    protected_gpus_by_pgid.setdefault(production_proxy.pgid, set())
    for pgid in protected_gpus_by_pgid:
        protected_groups_live[pgid] = _stable_group_snapshot(selected_host, pgid)

    materialized: dict[tuple[int, int], ProcessRecord] = {}

    def materialize(process: LiveProcess) -> ProcessRecord:
        """Materialize one process once per exact instance.

        :param process: Live process state.
        :returns: Sealed process record.
        """
        key = (process.pid, process.start_time_ticks)
        if key not in materialized:
            materialized[key] = materialize_process(
                selected_host, process, artifact_directory
            )
        return materialized[key]

    services: list[ServiceRecord] = []
    for spec in SERVICE_SPECS:
        root = roots[spec.role]
        _require_reopenable_executable(selected_host, root)
        service_gpu_indices = tuple(
            sorted(
                process.gpu_index
                for process in gpu_processes
                if gpu_pgids[process.pid] == root.pgid
            )
        )
        services.append(
            ServiceRecord(
                role=spec.role,
                port=spec.port,
                health_path=spec.health_path,
                root=materialize(root),
                group_members=tuple(materialize(item) for item in groups[spec.role]),
                gpu_indices=service_gpu_indices,
                ucx_environment_b64=tuple(
                    encode_bytes(item)
                    for item in environment_subset(
                        root.environment, (b"UCX_", b"NIXL_")
                    )
                ),
                launcher_files=discover_launcher_files(selected_host, root),
            )
        )

    protected: list[ProtectedGroup] = []
    for pgid, members in sorted(protected_groups_live.items()):
        member_pids = {member.pid for member in members}
        protected_listener_ports = tuple(
            sorted(
                endpoint_port(listener.local)
                for listener in listeners_before
                if len(set(listener.process_ids) & member_pids) > 0
            )
        )
        protected.append(
            ProtectedGroup(
                pgid=pgid,
                gpu_indices=tuple(sorted(protected_gpus_by_pgid[pgid])),
                group_members=tuple(materialize(item) for item in members),
                listener_ports=protected_listener_ports,
            )
        )

    metrics: list[MetricRecord] = []
    for role in ("p", "d1", "d2"):
        service = next(item for item in services if item.role == role)
        url = f"http://127.0.0.1:{service.port}/metrics"
        status, body = selected_host.http_get(url, 10.0)
        if status != 200:
            raise LifecycleError(f"{role} metrics returned HTTP {status}")
        running, waiting = parse_quiescence_metrics(body)
        if running != 0.0 or waiting != 0.0:
            raise LifecycleError(
                f"{role} is not idle: running={running}, waiting={waiting}"
            )
        relative = f"metrics/{role}.prom"
        _write_private_bytes(artifact_directory / relative, body)
        metrics.append(
            MetricRecord(
                role=role,
                url=url,
                running=running,
                waiting=waiting,
                body_sha256=hashlib.sha256(body).hexdigest(),
                body_artifact=relative,
            )
        )

    storm = selected_host.file_digest(os.fsencode(storm_archive.resolve()))
    if storm.sha256 != AUTHORITATIVE_STORM37B_SHA256:
        raise LifecycleError(
            f"storm37b archive digest is {storm.sha256}, not authoritative"
        )
    connections = selected_host.connections()
    listeners_after = selected_host.listeners()
    _require_same_listener_owners(listeners_before, listeners_after)
    for role, root in roots.items():
        current = selected_host.process(root.pid)
        if current.instance_identity() != root.instance_identity():
            raise LifecycleError(f"{role} changed during capture")
        current_group = _stable_group_snapshot(selected_host, root.pgid)
        if _live_group_identity(current_group) != _live_group_identity(groups[role]):
            raise LifecycleError(f"{role} group membership changed during capture")
    for pgid, members in protected_groups_live.items():
        current_group = _stable_group_snapshot(selected_host, pgid)
        if _live_group_identity(current_group) != _live_group_identity(members):
            raise LifecycleError(
                f"protected group {pgid} membership changed during capture"
            )
    if selected_host.tool_identity() != tool_identity:
        raise LifecycleError("lifecycle tool identity changed during capture")

    snapshot = LaneSnapshot(
        schema_version=SCHEMA_VERSION,
        captured_at_utc=_utc_now(),
        hostname=selected_host.hostname(),
        boot_id=selected_host.boot_id(),
        kernel_release=selected_host.kernel_release(),
        tool_identity=tool_identity,
        storm_archive=storm,
        services=tuple(services),
        protected_groups=tuple(protected),
        gpu_devices=gpu_devices,
        gpu_processes=gpu_processes,
        listeners=listeners_after,
        connections=connections,
        metrics=tuple(metrics),
    )
    snapshot = LaneSnapshot.from_dict(snapshot.to_dict())
    write_json_file(artifact_directory / "snapshot.json", snapshot.to_dict())
    _write_seal(artifact_directory)
    _fsync_directory(artifact_directory)
    return snapshot


def load_snapshot(artifact_directory: Path) -> LaneSnapshot:
    """Load and verify a private sealed lifecycle snapshot.

    :param artifact_directory: Snapshot directory.
    :returns: Validated snapshot.
    :raises LifecycleError: If permissions, paths, or hashes are unsafe.
    """
    _reject_symlink_components(artifact_directory.absolute())
    resolved = artifact_directory.resolve(strict=True)
    _reject_public_evidence_path(resolved)
    _enforce_data_private_root(resolved)
    _require_directory_mode(resolved, 0o700)
    seal_path = resolved / "SHA256SUMS"
    snapshot_path = resolved / "snapshot.json"
    _require_regular_private_file(seal_path)
    _require_regular_private_file(snapshot_path)
    _verify_operations_tree(resolved)
    sealed = _read_seal(seal_path)
    actual_files = _sealed_files(resolved)
    if set(sealed) != set(actual_files):
        raise LifecycleError(
            "sealed artifact set mismatch: "
            f"seal={sorted(sealed)}, actual={sorted(actual_files)}"
        )
    for relative, expected_digest in sealed.items():
        path = _confined_artifact_path(resolved, relative)
        _require_regular_private_file(path)
        actual_digest = sha256_file(path)
        if actual_digest != expected_digest:
            raise LifecycleError(f"artifact digest mismatch for {relative}")
    try:
        snapshot = LaneSnapshot.from_json_bytes(snapshot_path.read_bytes())
    except SnapshotValidationError as error:
        raise LifecycleError(f"snapshot schema is invalid: {error}") from error
    referenced = {"snapshot.json"}
    for service in snapshot.services:
        for process in service.group_members:
            referenced.add(process.executable_artifact)
        for metric in snapshot.metrics:
            referenced.add(metric.body_artifact)
    for group in snapshot.protected_groups:
        for process in group.group_members:
            referenced.add(process.executable_artifact)
    if not referenced.issubset(sealed):
        raise LifecycleError(
            "snapshot references unsealed artifacts: "
            f"{sorted(referenced - set(sealed))}"
        )
    return snapshot


def stop(
    artifact_directory: Path,
    *,
    dry_run: bool,
    term_timeout_seconds: float = 30.0,
    kill_timeout_seconds: float = 15.0,
    host: Host | None = None,
) -> dict[str, object]:
    """Quiesce the lane while holding its host-wide ownership lock.

    :param artifact_directory: Private sealed snapshot directory.
    :param dry_run: Validate and report without signaling any process.
    :param term_timeout_seconds: Graceful-stop wait per phase.
    :param kill_timeout_seconds: Forced-stop wait per phase.
    :param host: Host provider used by tests.
    :returns: Structured operation result.
    """
    with ExperimentLaneLock():
        return _stop_locked(
            artifact_directory,
            dry_run=dry_run,
            term_timeout_seconds=term_timeout_seconds,
            kill_timeout_seconds=kill_timeout_seconds,
            host=host,
        )


def _stop_locked(
    artifact_directory: Path,
    *,
    dry_run: bool,
    term_timeout_seconds: float,
    kill_timeout_seconds: float,
    host: Host | None,
) -> dict[str, object]:
    """Quiesce the experiment lane in a protected fail-closed order.

    :param artifact_directory: Private sealed snapshot directory.
    :param dry_run: Validate and report without signaling any process.
    :param term_timeout_seconds: Graceful-stop wait per phase.
    :param kill_timeout_seconds: Forced-stop wait per phase.
    :param host: Host provider used by tests.
    :returns: Structured operation result.
    """
    if term_timeout_seconds <= 0.0 or kill_timeout_seconds <= 0.0:
        raise LifecycleError("stop timeouts must be positive")
    selected_host = LinuxHost() if host is None else host
    snapshot = load_snapshot(artifact_directory)
    _attest_static(selected_host, snapshot)
    _attest_protected(selected_host, snapshot)
    _attest_services(selected_host, snapshot, captured_instances=True)
    _require_idle(selected_host, snapshot, samples=2)
    plan = [
        {
            "phase": index,
            "roles": list(roles),
            "signals": [
                {
                    "role": role,
                    "pgid": snapshot.service(role).root.pgid,
                    "signal": "SIGTERM",
                }
                for role in roles
            ],
        }
        for index, roles in enumerate(STOP_PHASES, start=1)
    ]
    if dry_run:
        return {"status": "DRY_RUN", "operation": "stop", "plan": plan}

    journal = OperationJournal(artifact_directory, "stop")
    journal.record(
        "preconditions_passed",
        plan=plan,
        tool_identity=snapshot.tool_identity.to_dict(),
    )
    mutated = False
    try:
        mutated = True
        _stop_phase(
            selected_host,
            snapshot,
            STOP_PHASES[0],
            term_timeout_seconds,
            kill_timeout_seconds,
            journal,
        )
        _require_idle(selected_host, snapshot, samples=2)
        _wait_no_engine_connections(selected_host, term_timeout_seconds)
        _stop_phase(
            selected_host,
            snapshot,
            STOP_PHASES[1],
            term_timeout_seconds,
            kill_timeout_seconds,
            journal,
        )
        _stop_phase(
            selected_host,
            snapshot,
            STOP_PHASES[2],
            term_timeout_seconds,
            kill_timeout_seconds,
            journal,
        )
        _attest_stopped(selected_host, snapshot)
        journal.record("stop_complete")
    except (LifecycleError, HostInspectionError) as error:
        journal.record("stop_failed", error=str(error), mutated=mutated)
        if not mutated:
            raise
        try:
            recovered = _restore_missing(
                selected_host,
                snapshot,
                journal,
                readiness_timeout_seconds=1800.0,
                log_prefix="stop-recovery",
            )
        except (LifecycleError, HostInspectionError) as recovery_error:
            journal.record("stop_recovery_failed", error=str(recovery_error))
            raise LifecycleError(
                f"stop failed ({error}); automatic recovery also failed "
                f"({recovery_error})"
            ) from recovery_error
        journal.record("stop_recovered", launched_roles=recovered)
        raise LifecycleError(
            f"stop failed ({error}); the lane was restored to a "
            "running-equivalent state; "
            "capture a fresh snapshot before retrying"
        ) from error
    finally:
        journal.close()
    return {"status": "PASS", "operation": "stop", "plan": plan}


def restore(
    artifact_directory: Path,
    *,
    dry_run: bool,
    readiness_timeout_seconds: float = 1800.0,
    rollback_timeout_seconds: float = 30.0,
    host: Host | None = None,
) -> dict[str, object]:
    """Restore the lane while holding its host-wide ownership lock.

    :param artifact_directory: Private sealed snapshot directory.
    :param dry_run: Validate and report without launching any process.
    :param readiness_timeout_seconds: Readiness wait per service.
    :param rollback_timeout_seconds: Cleanup wait after a failed restore.
    :param host: Host provider used by tests.
    :returns: Structured operation result.
    """
    with ExperimentLaneLock():
        return _restore_locked(
            artifact_directory,
            dry_run=dry_run,
            readiness_timeout_seconds=readiness_timeout_seconds,
            rollback_timeout_seconds=rollback_timeout_seconds,
            host=host,
        )


def _restore_locked(
    artifact_directory: Path,
    *,
    dry_run: bool,
    readiness_timeout_seconds: float,
    rollback_timeout_seconds: float,
    host: Host | None,
) -> dict[str, object]:
    """Restore the exact lane launch images in dependency order.

    :param artifact_directory: Private sealed snapshot directory.
    :param dry_run: Validate and report without launching any process.
    :param readiness_timeout_seconds: Readiness wait per service.
    :param rollback_timeout_seconds: Cleanup wait after a failed restore.
    :param host: Host provider used by tests.
    :returns: Structured operation result.
    """
    if readiness_timeout_seconds <= 0.0 or rollback_timeout_seconds <= 0.0:
        raise LifecycleError("restore timeouts must be positive")
    selected_host = LinuxHost() if host is None else host
    snapshot = load_snapshot(artifact_directory)
    _attest_static(selected_host, snapshot)
    _attest_protected(selected_host, snapshot)
    states = _classify_runtime(selected_host, snapshot)
    _validate_partial_dependencies(states)
    plan = [
        {
            "role": role,
            "port": snapshot.service(role).port,
            "executable_sha256": snapshot.service(role).root.executable_sha256,
            "launch_contract_sha256": _launch_contract_digest(
                snapshot.service(role).root
            ),
            "observed_state": states[role].status,
            "action": "retain" if states[role].status != "absent" else "launch",
        }
        for role in RESTORE_ORDER
    ]
    if dry_run:
        return {"status": "DRY_RUN", "operation": "restore", "plan": plan}

    journal = OperationJournal(artifact_directory, "restore")
    journal.record(
        "preconditions_passed",
        plan=plan,
        tool_identity=snapshot.tool_identity.to_dict(),
    )
    try:
        launched_roles = _restore_missing(
            selected_host,
            snapshot,
            journal,
            readiness_timeout_seconds=readiness_timeout_seconds,
            rollback_timeout_seconds=rollback_timeout_seconds,
            log_prefix="restore",
        )
        journal.record("restore_complete", launched_roles=launched_roles)
    finally:
        journal.close()
    return {"status": "PASS", "operation": "restore", "plan": plan}


def verify(
    artifact_directory: Path,
    *,
    expected_state: str,
    host: Host | None = None,
) -> dict[str, object]:
    """Verify the lane while holding its host-wide ownership lock.

    :param artifact_directory: Private sealed snapshot directory.
    :param expected_state: One of ``captured``, ``running``, or ``stopped``.
    :param host: Host provider used by tests.
    :returns: Structured verification result.
    """
    with ExperimentLaneLock():
        return _verify_locked(
            artifact_directory,
            expected_state=expected_state,
            host=host,
        )


def _verify_locked(
    artifact_directory: Path,
    *,
    expected_state: str,
    host: Host | None,
) -> dict[str, object]:
    """Verify a captured, restored, or stopped lane against its snapshot.

    :param artifact_directory: Private sealed snapshot directory.
    :param expected_state: One of ``captured``, ``running``, or ``stopped``.
    :param host: Host provider used by tests.
    :returns: Structured verification result.
    """
    selected_host = LinuxHost() if host is None else host
    snapshot = load_snapshot(artifact_directory)
    _attest_static(selected_host, snapshot)
    _attest_protected(selected_host, snapshot)
    if expected_state == "stopped":
        _attest_stopped(selected_host, snapshot)
    elif expected_state in {"captured", "running"}:
        _attest_services(
            selected_host,
            snapshot,
            captured_instances=expected_state == "captured",
        )
        _require_idle(selected_host, snapshot, samples=2)
    else:
        raise LifecycleError(f"unsupported expected state {expected_state!r}")
    ucx = {
        service.role: [
            decode_bytes(item).decode(errors="surrogateescape")
            for item in service.ucx_environment_b64
        ]
        for service in snapshot.services
    }
    return {
        "status": "PASS",
        "operation": "verify",
        "expected_state": expected_state,
        "protected_gpu_indices": [6, 7],
        "ucx_environment": ucx,
        "storm37b_sha256": snapshot.storm_archive.sha256,
    }


def _classify_runtime(
    host: Host, snapshot: LaneSnapshot
) -> dict[str, RuntimeServiceState]:
    """Classify every role as captured, equivalent, or absent.

    :param host: Host state provider.
    :param snapshot: Sealed lane snapshot.
    :returns: Runtime state by role.
    """
    listeners = host.listeners()
    states: dict[str, RuntimeServiceState] = {}
    for service in snapshot.services:
        matches = [
            listener
            for listener in listeners
            if endpoint_port(listener.local) == service.port
        ]
        if len(matches) == 0:
            _require_captured_service_absent(host, service)
            states[service.role] = RuntimeServiceState(
                role=service.role,
                status="absent",
                root=None,
                group_members=(),
            )
            continue
        if len(matches) != 1 or len(matches[0].process_ids) != 1:
            raise LifecycleError(
                f"{service.role} listener identity is ambiguous during recovery"
            )
        root = host.process(matches[0].process_ids[0])
        group = _stable_group_snapshot(host, root.pgid)
        captured_root = root.instance_identity() == record_instance_identity(
            service.root
        )
        if captured_root:
            expected_group = tuple(
                sorted(
                    (
                        record_instance_identity(member)
                        for member in service.group_members
                    ),
                    key=str,
                )
            )
            if _live_group_identity(group) != expected_group:
                raise LifecycleError(
                    f"captured {service.role} process group has drifted"
                )
            status = "captured"
        else:
            if root.pid != root.pgid or root.pid != root.sid:
                raise LifecycleError(
                    f"{service.role} listener is not an owned session/group leader"
                )
            if root.launch_identity() != record_launch_identity(service.root):
                raise LifecycleError(
                    f"{service.role} listener does not match the captured launch image"
                )
            status = "equivalent"
        states[service.role] = RuntimeServiceState(
            role=service.role,
            status=status,
            root=root,
            group_members=group,
        )
    _validate_partial_gpu_ownership(host, snapshot, states)
    return states


def _launch_contract_digest(process: ProcessRecord) -> str:
    """Hash private launch fields for non-secret operation reporting.

    :param process: Captured root process.
    :returns: SHA-256 digest of its exact launch contract.
    """
    payload = {
        "argv_b64": list(process.argv_b64),
        "environment_b64": list(process.environment_b64),
        "cwd_b64": process.cwd_b64,
        "executable_sha256": process.executable_sha256,
        "resource_limits": [item.to_dict() for item in process.resource_limits],
        "stdin_target_b64": process.stdin_target_b64,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _require_captured_service_absent(host: Host, service: ServiceRecord) -> None:
    """Prove that every captured instance for an absent role is gone.

    :param host: Host state provider.
    :param service: Captured service record.
    """
    if _group_exists(host, service.root.pgid):
        raise LifecycleError(
            f"{service.role} listener is absent but captured group "
            f"{service.root.pgid} remains"
        )
    for member in service.group_members:
        try:
            current = host.process(member.pid)
        except ProcessNotFoundError:
            continue
        if current.start_time_ticks == member.start_time_ticks:
            raise LifecycleError(
                f"{service.role} listener is absent but captured PID "
                f"{member.pid} remains"
            )


def _validate_partial_gpu_ownership(
    host: Host,
    snapshot: LaneSnapshot,
    states: dict[str, RuntimeServiceState],
) -> None:
    """Join current GPUs 0-5 to the classified service groups.

    :param host: Host state provider.
    :param snapshot: Sealed lane snapshot.
    :param states: Classified runtime states.
    """
    _, gpu_processes = host.gpu_state()
    observed: dict[str, set[int]] = {
        service.role: set() for service in snapshot.services
    }
    running_pgids = {
        role: state.root.pgid
        for role, state in states.items()
        if state.root is not None
    }
    for process in gpu_processes:
        if process.gpu_index >= 6:
            continue
        pgid = host.process(process.pid).pgid
        owners = [
            role for role, owner_pgid in running_pgids.items() if pgid == owner_pgid
        ]
        if len(owners) != 1:
            raise LifecycleError(
                f"GPU {process.gpu_index} process {process.pid} is not owned "
                "by one running role"
            )
        observed[owners[0]].add(process.gpu_index)
    for service in snapshot.services:
        expected = (
            set(service.gpu_indices)
            if states[service.role].status != "absent"
            else set()
        )
        if observed[service.role] != expected:
            raise LifecycleError(
                f"{service.role} partial-state GPU ownership is "
                f"{sorted(observed[service.role])}, expected {sorted(expected)}"
            )


def _validate_partial_dependencies(states: dict[str, RuntimeServiceState]) -> None:
    """Reject partial topologies that invert service dependencies.

    :param states: Classified runtime states.
    """
    running = {role for role, state in states.items() if state.status != "absent"}
    if "p" not in running and len(running & {"d1", "d2"}) > 0:
        raise LifecycleError("cannot restore P beneath a still-running decode service")
    if len({"p", "d1", "d2"} - running) > 0 and len(running & {"router", "proxy"}) > 0:
        raise LifecycleError(
            "ingress remains live while an engine dependency is absent"
        )


def _restore_missing(
    host: Host,
    snapshot: LaneSnapshot,
    journal: OperationJournal,
    *,
    readiness_timeout_seconds: float,
    rollback_timeout_seconds: float = 30.0,
    log_prefix: str,
) -> list[str]:
    """Retain valid roles and transactionally launch only absent roles.

    :param host: Host state provider.
    :param snapshot: Sealed lane snapshot.
    :param journal: Durable operation journal.
    :param readiness_timeout_seconds: Readiness deadline per service.
    :param rollback_timeout_seconds: Cleanup deadline after a failed launch.
    :param log_prefix: Private restore-log prefix.
    :returns: Roles launched by this invocation.
    """
    _attest_protected(host, snapshot)
    states = _classify_runtime(host, snapshot)
    _validate_partial_dependencies(states)
    executables = {
        role: Path(
            os.fsdecode(_reopenable_executable(host, snapshot.service(role).root))
        )
        for role in RESTORE_ORDER
        if states[role].status == "absent"
    }
    launched: list[tuple[str, int]] = []
    try:
        for role in RESTORE_ORDER:
            if states[role].status != "absent":
                journal.record("service_retained", role=role, state=states[role].status)
                continue
            service = snapshot.service(role)
            stdout_path = journal.directory / f"{log_prefix}-{role}.stdout.log"
            stderr_path = journal.directory / f"{log_prefix}-{role}.stderr.log"
            pid = host.launch_exact(
                service.root,
                executables[role],
                stdout_path,
                stderr_path,
            )
            launched.append((role, pid))
            journal.record("service_launched", role=role, pid=pid)
            _wait_service_ready(
                host,
                snapshot,
                role,
                pid,
                readiness_timeout_seconds,
            )
            journal.record("service_ready", role=role, pid=pid)
        _attest_services(host, snapshot, captured_instances=False)
        _require_idle(host, snapshot, samples=2)
        _attest_protected(host, snapshot)
    except (LifecycleError, HostInspectionError) as error:
        journal.record("restore_failed", error=str(error))
        try:
            _rollback_launched(
                host,
                snapshot,
                launched,
                rollback_timeout_seconds,
                journal,
            )
        except LifecycleError as rollback_error:
            raise LifecycleError(
                f"restore failed ({error}); rollback also failed ({rollback_error})"
            ) from rollback_error
        raise
    return [role for role, _ in launched]


def _stable_group_snapshot(host: Host, pgid: int) -> tuple[LiveProcess, ...]:
    """Require two identical full reads of one process group.

    :param host: Host state provider.
    :param pgid: Process-group identifier.
    :returns: Stable process-group membership.
    """
    first = host.group_members(pgid)
    second = host.group_members(pgid)
    if _live_group_identity(first) != _live_group_identity(second):
        raise LifecycleError(
            f"process group {pgid} membership or identity is not stable"
        )
    return first


def _live_group_identity(
    group: tuple[LiveProcess, ...],
) -> tuple[tuple[object, ...], ...]:
    """Normalize a live process group for exact comparison.

    :param group: Live process-group membership.
    :returns: Sorted exact instance identities.
    """
    return tuple(sorted((member.instance_identity() for member in group), key=str))


def _validate_live_gpu_ownership(
    groups: dict[str, tuple[LiveProcess, ...]],
    gpu_processes: tuple[GpuProcess, ...],
    gpu_pgids: dict[int, int],
) -> None:
    """Validate captured experiment and protected GPU ownership.

    :param groups: Experiment process groups by role.
    :param gpu_processes: NVIDIA compute-process inventory.
    :param gpu_pgids: Compute-process PGIDs by PID.
    """
    expected = {
        "p": {0, 1, 2, 3},
        "d1": {4},
        "d2": {5},
        "router": set(),
        "proxy": set(),
    }
    group_pgids = {role: members[0].pgid for role, members in groups.items()}
    observed: dict[str, set[int]] = {role: set() for role in expected}
    protected_indices: set[int] = set()
    for process in gpu_processes:
        pid = process.pid
        index = process.gpu_index
        if index >= 6:
            protected_indices.add(index)
            if gpu_pgids[pid] in set(group_pgids.values()):
                raise LifecycleError(
                    f"experiment group owns protected GPU {index} process {pid}"
                )
            continue
        owners = [role for role, pgid in group_pgids.items() if gpu_pgids[pid] == pgid]
        if len(owners) != 1:
            raise LifecycleError(
                f"GPU {index} process {pid} does not have one experiment owner"
            )
        observed[owners[0]].add(index)
    if observed != expected:
        raise LifecycleError(f"experiment GPU ownership mismatch: {observed}")
    if protected_indices != {6, 7}:
        raise LifecycleError(
            "production GPU protection requires exact indices 6 and 7, "
            f"got {protected_indices}"
        )


def _require_same_listener_owners(
    before: tuple[SocketRecord, ...], after: tuple[SocketRecord, ...]
) -> None:
    """Require every lane listener owner to remain stable.

    :param before: Initial listener inventory.
    :param after: Final listener inventory.
    """
    for spec in SERVICE_SPECS:
        if listener_owner(before, spec.port) != listener_owner(after, spec.port):
            raise LifecycleError(f"{spec.role} listener changed during capture")


def _require_reopenable_executable(host: Host, process: LiveProcess) -> bytes:
    """Resolve a live process's exact reopenable executable path.

    :param host: Host state provider.
    :param process: Live process state.
    :returns: Raw verified executable path.
    """
    return _reopenable_executable_from_values(
        host,
        process.argv,
        process.cwd,
        process.executable_sha256,
        process.executable_size,
    )


def _reopenable_executable(host: Host, process: ProcessRecord) -> bytes:
    """Resolve a captured process's exact reopenable executable path.

    :param host: Host state provider.
    :param process: Captured process record.
    :returns: Raw verified executable path.
    """
    return _reopenable_executable_from_values(
        host,
        tuple(decode_bytes(item) for item in process.argv_b64),
        decode_bytes(process.cwd_b64),
        process.executable_sha256,
        process.executable_size,
    )


def _reopenable_executable_from_values(
    host: Host,
    argv: tuple[bytes, ...],
    cwd: bytes,
    expected_sha256: str,
    expected_size: int,
) -> bytes:
    """Verify argv zero against the executable inode captured in use.

    :param host: Host state provider.
    :param argv: Raw argument vector.
    :param cwd: Raw working directory.
    :param expected_sha256: Captured executable digest.
    :param expected_size: Captured executable size.
    :returns: Raw verified executable path.
    """
    if len(argv) == 0:
        raise LifecycleError("process argument vector is empty")
    candidate = argv[0] if os.path.isabs(argv[0]) else os.path.join(cwd, argv[0])
    candidate = os.path.abspath(candidate)
    try:
        digest = host.file_digest(candidate)
    except HostInspectionError as error:
        raise LifecycleError(
            f"original executable path {candidate!r} cannot be reopened"
        ) from error
    if digest.sha256 != expected_sha256 or digest.size != expected_size:
        raise LifecycleError(
            f"original executable path {candidate!r} no longer matches its loaded inode"
        )
    return candidate


def _attest_static(host: Host, snapshot: LaneSnapshot) -> None:
    """Verify immutable host, storm, launcher, and tool identity.

    :param host: Host state provider.
    :param snapshot: Sealed lane snapshot.
    """
    if host.boot_id() != snapshot.boot_id:
        raise LifecycleError("host boot ID changed after capture")
    if host.hostname() != snapshot.hostname:
        raise LifecycleError("hostname changed after capture")
    if host.tool_identity() != snapshot.tool_identity:
        raise LifecycleError("lifecycle tool Git or file identity changed")
    storm = host.file_digest(decode_bytes(snapshot.storm_archive.path_b64))
    if storm != snapshot.storm_archive:
        raise LifecycleError("storm37b archive path, symlink, size, or digest changed")
    for service in snapshot.services:
        _reopenable_executable(host, service.root)
        for expected in service.launcher_files:
            current = host.file_digest(decode_bytes(expected.path_b64))
            if current != expected:
                raise LifecycleError(f"{service.role} launcher input changed")


def _attest_protected(host: Host, snapshot: LaneSnapshot) -> None:
    """Verify production proxy, listeners, process groups, and GPUs 6-7.

    :param host: Host state provider.
    :param snapshot: Sealed lane snapshot.
    """
    devices, gpu_processes = host.gpu_state()
    if devices != snapshot.gpu_devices:
        raise LifecycleError("physical GPU inventory changed")
    expected_gpu = sorted(
        (
            process.gpu_index,
            process.gpu_uuid,
            process.pid,
            process.process_name,
        )
        for process in snapshot.gpu_processes
        if process.gpu_index >= 6
    )
    current_gpu = sorted(
        (
            process.gpu_index,
            process.gpu_uuid,
            process.pid,
            process.process_name,
        )
        for process in gpu_processes
        if process.gpu_index >= 6
    )
    if current_gpu != expected_gpu:
        raise LifecycleError("production GPU 6/7 process identity changed")
    listeners = host.listeners()
    for group in snapshot.protected_groups:
        current_members = _stable_group_snapshot(host, group.pgid)
        expected_members = tuple(
            sorted(
                (record_instance_identity(member) for member in group.group_members),
                key=str,
            )
        )
        if _live_group_identity(current_members) != expected_members:
            raise LifecycleError(f"protected process group {group.pgid} changed")
        current_pids = {member.pid for member in current_members}
        current_ports = tuple(
            sorted(
                endpoint_port(listener.local)
                for listener in listeners
                if len(set(listener.process_ids) & current_pids) > 0
            )
        )
        if current_ports != group.listener_ports:
            raise LifecycleError(
                f"protected process group {group.pgid} listener identity changed"
            )


def _attest_services(
    host: Host, snapshot: LaneSnapshot, *, captured_instances: bool
) -> None:
    """Verify all running lane services and joined GPU ownership.

    :param host: Host state provider.
    :param snapshot: Sealed lane snapshot.
    :param captured_instances: Require original PID/start-time identities.
    """
    listeners = host.listeners()
    root_pgids: dict[str, int] = {}
    for service in snapshot.services:
        pid = listener_owner(listeners, service.port)
        current = host.process(pid)
        root_pgids[service.role] = current.pgid
        if captured_instances:
            if current.instance_identity() != record_instance_identity(service.root):
                raise LifecycleError(
                    f"{service.role} captured process identity changed"
                )
            current_group = _stable_group_snapshot(host, service.root.pgid)
            expected_group = tuple(
                sorted(
                    (
                        record_instance_identity(member)
                        for member in service.group_members
                    ),
                    key=str,
                )
            )
            if _live_group_identity(current_group) != expected_group:
                raise LifecycleError(f"{service.role} captured process group changed")
            continue
        if current.pid != current.pgid or current.pid != current.sid:
            raise LifecycleError(
                f"restored {service.role} is not a session/group leader"
            )
        if current.launch_identity() != record_launch_identity(service.root):
            raise LifecycleError(f"restored {service.role} launch image changed")
    _attest_live_gpu_mapping(host, snapshot, root_pgids)
    for service in snapshot.services:
        if service.health_path is None:
            continue
        status, _ = host.http_get(
            f"http://127.0.0.1:{service.port}{service.health_path}", 10.0
        )
        if status != 200:
            raise LifecycleError(f"{service.role} health returned HTTP {status}")


def _attest_live_gpu_mapping(
    host: Host, snapshot: LaneSnapshot, root_pgids: dict[str, int]
) -> None:
    """Join every experiment GPU process to one running service group.

    :param host: Host state provider.
    :param snapshot: Sealed lane snapshot.
    :param root_pgids: Current service PGIDs by role.
    """
    _, gpu_processes = host.gpu_state()
    observed: dict[str, set[int]] = {
        service.role: set() for service in snapshot.services
    }
    for process in gpu_processes:
        if process.gpu_index >= 6:
            continue
        pgid = host.process(process.pid).pgid
        owners = [role for role, root_pgid in root_pgids.items() if root_pgid == pgid]
        if len(owners) != 1:
            raise LifecycleError(
                f"GPU {process.gpu_index} process {process.pid} has "
                "ambiguous lane ownership"
            )
        observed[owners[0]].add(process.gpu_index)
    expected = {service.role: set(service.gpu_indices) for service in snapshot.services}
    if observed != expected:
        raise LifecycleError(f"live experiment GPU ownership mismatch: {observed}")


def _require_idle(host: Host, snapshot: LaneSnapshot, *, samples: int) -> None:
    """Require repeated zero running and waiting request metrics.

    :param host: Host state provider.
    :param snapshot: Sealed lane snapshot.
    :param samples: Number of stable samples.
    """
    for sample in range(samples):
        for role in ("p", "d1", "d2"):
            service = snapshot.service(role)
            status, body = host.http_get(
                f"http://127.0.0.1:{service.port}/metrics", 10.0
            )
            if status != 200:
                raise LifecycleError(f"{role} metrics returned HTTP {status}")
            running, waiting = parse_quiescence_metrics(body)
            if running != 0.0 or waiting != 0.0:
                raise LifecycleError(
                    f"{role} is not idle: running={running}, waiting={waiting}"
                )
        if sample + 1 < samples:
            host.sleep(1.0)


def _stop_phase(
    host: Host,
    snapshot: LaneSnapshot,
    roles: tuple[str, ...],
    term_timeout_seconds: float,
    kill_timeout_seconds: float,
    journal: OperationJournal,
) -> None:
    """Stop one dependency phase with protected identity checks.

    :param host: Host state provider.
    :param snapshot: Sealed lane snapshot.
    :param roles: Roles stopped together in this phase.
    :param term_timeout_seconds: Graceful-stop deadline.
    :param kill_timeout_seconds: Forced-stop deadline.
    :param journal: Durable operation journal.
    """
    _attest_protected(host, snapshot)
    for role in roles:
        _attest_service_instance(host, snapshot.service(role))
        _require_group_not_protected(host, snapshot, snapshot.service(role).root.pgid)
    for role in roles:
        pgid = snapshot.service(role).root.pgid
        journal.record("signal", role=role, pgid=pgid, signal="SIGTERM")
        host.kill_group(pgid, signal.SIGTERM)
    remaining = _wait_groups(host, snapshot, roles, term_timeout_seconds)
    if len(remaining) == 0:
        _attest_protected(host, snapshot)
        return
    _attest_protected(host, snapshot)
    for role in remaining:
        service = snapshot.service(role)
        _require_remaining_group_subset(host, service)
        _require_group_not_protected(host, snapshot, service.root.pgid)
        journal.record("signal", role=role, pgid=service.root.pgid, signal="SIGKILL")
        host.kill_group(service.root.pgid, signal.SIGKILL)
    remaining = _wait_groups(host, snapshot, tuple(remaining), kill_timeout_seconds)
    if len(remaining) > 0:
        raise LifecycleError(f"process groups did not stop: {remaining}")
    _attest_protected(host, snapshot)


def _attest_service_instance(host: Host, service: ServiceRecord) -> None:
    """Revalidate one captured service immediately before a signal.

    :param host: Host state provider.
    :param service: Captured service record.
    """
    listeners = host.listeners()
    pid = listener_owner(listeners, service.port)
    current = host.process(pid)
    if current.instance_identity() != record_instance_identity(service.root):
        raise LifecycleError(f"{service.role} process identity changed before signal")
    current_group = _stable_group_snapshot(host, service.root.pgid)
    expected = tuple(
        sorted(
            (record_instance_identity(member) for member in service.group_members),
            key=str,
        )
    )
    if _live_group_identity(current_group) != expected:
        raise LifecycleError(f"{service.role} process group changed before signal")


def _require_group_not_protected(
    host: Host, snapshot: LaneSnapshot, target_pgid: int
) -> None:
    """Prove that a signal target owns no protected process.

    :param host: Host state provider.
    :param snapshot: Sealed lane snapshot.
    :param target_pgid: Candidate signal target.
    """
    _, gpu_processes = host.gpu_state()
    protected_pgids = {
        host.process(process.pid).pgid
        for process in gpu_processes
        if process.gpu_index >= 6
    }
    protected_pgids.update(group.pgid for group in snapshot.protected_groups)
    if target_pgid in protected_pgids:
        raise LifecycleError(
            f"refusing to signal protected production process group {target_pgid}"
        )


def _require_remaining_group_subset(host: Host, service: ServiceRecord) -> None:
    """Allow SIGKILL only for surviving captured group members.

    :param host: Host state provider.
    :param service: Captured service record.
    """
    try:
        current = host.group_members(service.root.pgid)
    except ProcessGroupNotFoundError:
        return
    expected = {record_instance_identity(member) for member in service.group_members}
    unexpected = [
        member.pid for member in current if member.instance_identity() not in expected
    ]
    if len(unexpected) > 0:
        raise LifecycleError(
            f"{service.role} acquired unexpected members before SIGKILL: {unexpected}"
        )


def _wait_groups(
    host: Host,
    snapshot: LaneSnapshot,
    roles: tuple[str, ...],
    timeout_seconds: float,
) -> list[str]:
    """Wait for selected captured groups to disappear.

    :param host: Host state provider.
    :param snapshot: Sealed lane snapshot.
    :param roles: Roles whose groups must disappear.
    :param timeout_seconds: Wait deadline.
    :returns: Roles still present at the deadline.
    """
    deadline = host.monotonic() + timeout_seconds
    while True:
        remaining = [
            role
            for role in roles
            if _group_exists(host, snapshot.service(role).root.pgid)
        ]
        if len(remaining) == 0 or host.monotonic() >= deadline:
            return remaining
        host.sleep(0.25)


def _group_exists(host: Host, pgid: int) -> bool:
    """Distinguish exact group absence from inspection failure.

    :param host: Host state provider.
    :param pgid: Process-group identifier.
    :returns: Whether the group exists.
    """
    try:
        host.group_members(pgid)
    except ProcessGroupNotFoundError:
        return False
    return True


def _wait_no_engine_connections(host: Host, timeout_seconds: float) -> None:
    """Wait for ingress-owned engine API connections to drain.

    :param host: Host state provider.
    :param timeout_seconds: Drain deadline.
    """
    deadline = host.monotonic() + timeout_seconds
    while True:
        unsafe = _unsafe_engine_connections(host)
        if len(unsafe) == 0:
            return
        if host.monotonic() >= deadline:
            raise LifecycleError(
                f"engine API connections remain after ingress stop: {unsafe}"
            )
        host.sleep(0.25)


def _unsafe_engine_connections(host: Host) -> list[str]:
    """List synchronized connections that still touch an engine API.

    :param host: Host state provider.
    :returns: Unsafe raw socket rows.
    """
    engine_ports = {8810, 8811, 8815}
    active_states = {
        "ESTAB",
        "SYN-SENT",
        "SYN-RECV",
        "FIN-WAIT-1",
        "FIN-WAIT-2",
        "CLOSE-WAIT",
        "LAST-ACK",
        "CLOSING",
    }
    return [
        record.raw
        for record in host.connections()
        if record.state in active_states
        and (
            endpoint_port(record.local) in engine_ports
            or endpoint_port(record.peer) in engine_ports
        )
    ]


def _attest_stopped(host: Host, snapshot: LaneSnapshot) -> None:
    """Prove the lane is absent, GPUs 0-5 are clear, and production is intact.

    :param host: Host state provider.
    :param snapshot: Sealed lane snapshot.
    """
    _attest_protected(host, snapshot)
    lane_ports = {service.port for service in snapshot.services}
    open_ports = sorted(
        endpoint_port(listener.local)
        for listener in host.listeners()
        if endpoint_port(listener.local) in lane_ports
    )
    if len(open_ports) > 0:
        raise LifecycleError(f"experiment listeners remain: {open_ports}")
    for service in snapshot.services:
        if _group_exists(host, service.root.pgid):
            raise LifecycleError(
                f"captured {service.role} process group {service.root.pgid} remains"
            )
        for member in service.group_members:
            try:
                current = host.process(member.pid)
            except ProcessNotFoundError:
                continue
            if current.start_time_ticks == member.start_time_ticks:
                raise LifecycleError(
                    f"captured {service.role} process {member.pid} remains"
                )
    _, gpu_processes = host.gpu_state()
    lane_gpu_processes = [
        (process.gpu_index, process.pid)
        for process in gpu_processes
        if process.gpu_index <= 5
    ]
    if len(lane_gpu_processes) > 0:
        raise LifecycleError(f"GPUs 0-5 are not clear after stop: {lane_gpu_processes}")


def _wait_service_ready(
    host: Host,
    snapshot: LaneSnapshot,
    role: str,
    launched_pid: int,
    timeout_seconds: float,
) -> None:
    """Wait for one launched service without masking an exited root.

    :param host: Host state provider.
    :param snapshot: Sealed lane snapshot.
    :param role: Launched role.
    :param launched_pid: Direct child PID.
    :param timeout_seconds: Readiness deadline.
    """
    service = snapshot.service(role)
    deadline = host.monotonic() + timeout_seconds
    last_error = "listener absent"
    while host.monotonic() < deadline:
        exit_status = host.child_exit_status(launched_pid)
        if exit_status is not None:
            raise LifecycleError(
                f"{role} root PID {launched_pid} exited during boot with "
                f"status {exit_status}"
            )
        try:
            current = host.process(launched_pid)
            if current.launch_identity() != record_launch_identity(service.root):
                raise LifecycleError(f"restored {role} launch image changed")
            owner = listener_owner(host.listeners(), service.port)
            if owner != launched_pid:
                raise LifecycleError(
                    f"restored {role} port is owned by PID {owner}, not {launched_pid}"
                )
            if service.health_path is not None:
                status, _ = host.http_get(
                    f"http://127.0.0.1:{service.port}{service.health_path}", 10.0
                )
                if status != 200:
                    last_error = f"health HTTP {status}"
                    host.sleep(1.0)
                    continue
            _require_role_gpu_ready(host, snapshot, role, current.pgid)
            _attest_protected(host, snapshot)
            return
        except HostInspectionError as error:
            last_error = str(error)
        host.sleep(1.0)
    raise LifecycleError(f"{role} did not become ready: {last_error}")


def _require_role_gpu_ready(
    host: Host, snapshot: LaneSnapshot, role: str, pgid: int
) -> None:
    """Require one launched role to own its exact GPU set.

    :param host: Host state provider.
    :param snapshot: Sealed lane snapshot.
    :param role: Launched role.
    :param pgid: Current service PGID.
    """
    service = snapshot.service(role)
    _, gpu_processes = host.gpu_state()
    owned = {
        process.gpu_index
        for process in gpu_processes
        if process.gpu_index <= 5 and host.process(process.pid).pgid == pgid
    }
    if owned != set(service.gpu_indices):
        raise HostInspectionError(
            f"{role} GPU ownership is {sorted(owned)}, expected {service.gpu_indices}"
        )


def _rollback_launched(
    host: Host,
    snapshot: LaneSnapshot,
    launched: list[tuple[str, int]],
    timeout_seconds: float,
    journal: OperationJournal,
) -> None:
    """Attempt identity-safe cleanup for every process launched here.

    :param host: Host state provider.
    :param snapshot: Sealed lane snapshot.
    :param launched: Roles and root PIDs launched by this invocation.
    :param timeout_seconds: Graceful and forced cleanup deadline.
    :param journal: Durable operation journal.
    """
    errors: list[str] = []
    owned_groups: dict[int, tuple[tuple[object, ...], ...]] = {}
    for role, pid in reversed(launched):
        try:
            process = host.process(pid)
        except ProcessNotFoundError:
            try:
                if _group_exists(host, pid):
                    errors.append(
                        f"{role}: launched root {pid} vanished while its group remains"
                    )
            except HostInspectionError as error:
                errors.append(f"{role}: cannot inspect launched group {pid}: {error}")
            continue
        try:
            if process.pid != process.pgid or process.pid != process.sid:
                raise LifecycleError(f"PID {pid} is not its session/group leader")
            if process.launch_identity() != record_launch_identity(
                snapshot.service(role).root
            ):
                raise LifecycleError("launch identity changed")
            group = _stable_group_snapshot(host, process.pgid)
            if any(member.sid != pid or member.pgid != pid for member in group):
                raise LifecycleError("launched session contains a foreign member")
            owned_groups[pid] = tuple(member.instance_identity() for member in group)
            _attest_protected(host, snapshot)
            _require_group_not_protected(host, snapshot, process.pgid)
            journal.record(
                "rollback_signal", role=role, pgid=process.pgid, signal="SIGTERM"
            )
            host.kill_group(process.pgid, signal.SIGTERM)
        except (LifecycleError, HostInspectionError, OSError) as error:
            errors.append(f"{role}: SIGTERM rollback failed: {error}")
    deadline = host.monotonic() + timeout_seconds
    while host.monotonic() < deadline:
        try:
            remaining = [pid for pid in owned_groups if _group_exists(host, pid)]
        except HostInspectionError as error:
            errors.append(f"rollback wait inspection failed: {error}")
            break
        if len(remaining) == 0:
            break
        host.sleep(0.25)
    for role, pid in reversed(launched):
        if pid not in owned_groups:
            continue
        try:
            if not _group_exists(host, pid):
                continue
            current = _stable_group_snapshot(host, pid)
            allowed = set(owned_groups[pid])
            if any(member.instance_identity() not in allowed for member in current):
                raise LifecycleError("group identity changed before SIGKILL")
            if any(member.sid != pid or member.pgid != pid for member in current):
                raise LifecycleError("group acquired a foreign session member")
            _attest_protected(host, snapshot)
            _require_group_not_protected(host, snapshot, pid)
            journal.record("rollback_signal", role=role, pgid=pid, signal="SIGKILL")
            host.kill_group(pid, signal.SIGKILL)
        except (LifecycleError, HostInspectionError, OSError) as error:
            errors.append(f"{role}: SIGKILL rollback failed: {error}")
    kill_deadline = host.monotonic() + timeout_seconds
    while host.monotonic() < kill_deadline:
        try:
            remaining_after_kill = [
                pid for pid in owned_groups if _group_exists(host, pid)
            ]
        except HostInspectionError as error:
            errors.append(f"SIGKILL rollback wait inspection failed: {error}")
            break
        if len(remaining_after_kill) == 0:
            break
        host.sleep(0.25)
    for role, pid in reversed(launched):
        if pid not in owned_groups:
            continue
        try:
            if _group_exists(host, pid):
                errors.append(f"{role}: launched group {pid} remains after rollback")
        except HostInspectionError as error:
            errors.append(f"{role}: final rollback inspection failed: {error}")
    try:
        _attest_protected(host, snapshot)
    except (LifecycleError, HostInspectionError) as error:
        errors.append(f"protected identity check failed after rollback: {error}")
    if len(errors) > 0:
        raise LifecycleError("; ".join(errors))


def _create_private_artifact_directory(path: Path) -> Path:
    """Create a symlink-free private snapshot directory.

    :param path: Requested artifact directory.
    :returns: Resolved new directory.
    """
    absolute = path.absolute()
    _reject_public_evidence_path(absolute)
    _enforce_data_private_root(absolute)
    if absolute.parent == absolute:
        raise LifecycleError("snapshot directory cannot be the filesystem root")
    _reject_symlink_components(absolute.parent)
    absolute.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    _reject_symlink_components(absolute.parent)
    resolved_parent = absolute.parent.resolve(strict=True)
    resolved = resolved_parent / absolute.name
    _reject_public_evidence_path(resolved)
    _enforce_data_private_root(resolved)
    resolved.mkdir(mode=0o700)
    _require_directory_mode(resolved, 0o700)
    _fsync_directory(resolved.parent)
    return resolved


def _reject_public_evidence_path(path: Path) -> None:
    """Reject lifecycle secrets under public evidence locations.

    :param path: Candidate artifact path.
    """
    forbidden = {
        Path("/data/colleague/evidence"),
        Path("/data/tmp/dva_handoff"),
    }
    for root in forbidden:
        try:
            path.relative_to(root)
        except ValueError:
            continue
        raise LifecycleError(
            "private lifecycle state cannot be stored under public evidence "
            f"root {root}"
        )


def _enforce_data_private_root(path: Path) -> None:
    """Confine box-side lifecycle state to its private root.

    :param path: Candidate artifact path.
    """
    data_root = Path("/data")
    try:
        path.relative_to(data_root)
    except ValueError:
        return
    try:
        path.relative_to(PRIVATE_ARTIFACT_ROOT)
    except ValueError as error:
        raise LifecycleError(
            "lifecycle artifacts under /data must be confined to "
            f"{PRIVATE_ARTIFACT_ROOT}"
        ) from error


def _reject_symlink_components(path: Path) -> None:
    """Reject every existing symlink component in a private path.

    :param path: Candidate private path.
    """
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        if not current.exists() and not current.is_symlink():
            continue
        if current.is_symlink():
            raise LifecycleError(f"private lifecycle path traverses symlink: {current}")


def _write_private_bytes(path: Path, payload: bytes) -> None:
    """Create, fsync, and publish one mode-0600 payload.

    :param path: New private file.
    :param payload: Exact file content.
    """
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
        0o600,
    )
    try:
        _write_all(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _write_seal(artifact_directory: Path) -> None:
    """Seal every immutable snapshot payload.

    :param artifact_directory: Snapshot root.
    """
    files = _sealed_files(artifact_directory)
    lines = [
        f"{sha256_file(artifact_directory / relative)}  {relative}\n"
        for relative in files
    ]
    _write_private_bytes(artifact_directory / "SHA256SUMS", "".join(lines).encode())


def _read_seal(path: Path) -> dict[str, str]:
    """Parse a canonical sorted SHA256SUMS file.

    :param path: Seal file.
    :returns: Digest by confined relative path.
    """
    result: dict[str, str] = {}
    lines = path.read_text(encoding="ascii").splitlines()
    for line_number, line in enumerate(lines, start=1):
        digest, separator, relative = line.partition("  ")
        if separator != "  " or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise LifecycleError(f"malformed SHA256SUMS line {line_number}")
        _validate_relative_path(relative)
        if relative in result:
            raise LifecycleError(f"duplicate sealed path {relative}")
        result[relative] = digest
    if list(result) != sorted(result):
        raise LifecycleError("SHA256SUMS paths are not sorted")
    return result


def _sealed_files(artifact_directory: Path) -> list[str]:
    """Enumerate and permission-check immutable snapshot payloads.

    :param artifact_directory: Snapshot root.
    :returns: Sorted relative file paths.
    """
    result: list[str] = []
    for path in artifact_directory.rglob("*"):
        relative = path.relative_to(artifact_directory)
        if len(relative.parts) > 0 and relative.parts[0] == "operations":
            continue
        if relative.as_posix() == "SHA256SUMS":
            continue
        if path.is_symlink():
            raise LifecycleError(f"snapshot artifact cannot be a symlink: {relative}")
        if path.is_dir():
            _require_directory_mode(path, 0o700)
            continue
        if not path.is_file():
            raise LifecycleError(f"snapshot artifact is not a regular file: {relative}")
        _require_regular_private_file(path)
        result.append(relative.as_posix())
    return sorted(result)


def _verify_operations_tree(artifact_directory: Path) -> None:
    """Validate private operation journals excluded from the capture seal.

    :param artifact_directory: Snapshot root.
    """
    operations = artifact_directory / "operations"
    if not operations.exists() and not operations.is_symlink():
        return
    if operations.is_symlink():
        raise LifecycleError("operations directory cannot be a symlink")
    _require_directory_mode(operations, 0o700)
    for path in operations.rglob("*"):
        if path.is_symlink():
            raise LifecycleError(f"operation artifact cannot be a symlink: {path}")
        if path.is_dir():
            _require_directory_mode(path, 0o700)
            continue
        _require_regular_private_file(path)


def _confined_artifact_path(root: Path, relative: str) -> Path:
    """Resolve a sealed relative path without symlink traversal.

    :param root: Snapshot root.
    :param relative: Canonical relative path.
    :returns: Confined resolved path.
    """
    _validate_relative_path(relative)
    candidate = root.joinpath(*relative.split("/"))
    current = root
    for part in relative.split("/"):
        current = current / part
        if current.is_symlink():
            raise LifecycleError(f"artifact path traverses symlink: {relative}")
    try:
        candidate.resolve(strict=True).relative_to(root)
    except (OSError, ValueError) as error:
        raise LifecycleError(f"artifact path escapes snapshot: {relative}") from error
    return candidate


def _validate_relative_path(relative: str) -> None:
    """Require a canonical traversal-free relative path.

    :param relative: Candidate relative path.
    """
    path = Path(relative)
    if path.is_absolute() or len(path.parts) == 0:
        raise LifecycleError(f"invalid snapshot-relative path {relative!r}")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise LifecycleError(f"unsafe snapshot-relative path {relative!r}")
    if path.as_posix() != relative:
        raise LifecycleError(f"non-canonical snapshot-relative path {relative!r}")


def _require_regular_private_file(path: Path) -> None:
    """Require a regular mode-0600 private file.

    :param path: Candidate file.
    """
    status = path.lstat()
    if not stat.S_ISREG(status.st_mode):
        raise LifecycleError(f"required artifact is not a regular file: {path}")
    if stat.S_IMODE(status.st_mode) != 0o600:
        raise LifecycleError(f"private artifact mode must be 0600: {path}")


def _require_directory_mode(path: Path, expected: int) -> None:
    """Require one exact private directory mode.

    :param path: Candidate directory.
    :param expected: Exact permission bits.
    """
    status = path.lstat()
    if not stat.S_ISDIR(status.st_mode) or stat.S_IMODE(status.st_mode) != expected:
        raise LifecycleError(f"directory mode must be {expected:04o}: {path}")


def _fsync_directory(path: Path) -> None:
    """Durably publish directory entry changes.

    :param path: Directory to fsync.
    """
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_all(descriptor: int, payload: bytes) -> None:
    """Write all bytes while detecting a zero-progress write.

    :param descriptor: Open file descriptor.
    :param payload: Exact bytes to write.
    """
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise LifecycleError("private artifact write made no progress")
        offset += written


def _utc_now() -> str:
    """Return a canonical UTC timestamp.

    :returns: ISO-8601 timestamp ending in ``Z``.
    """
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
