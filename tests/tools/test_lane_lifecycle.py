import hashlib
import os
import signal
from pathlib import Path

import pytest

from tools.gemma4_pd.lane_lifecycle import lifecycle as lifecycle_module
from tools.gemma4_pd.lane_lifecycle.cli import main
from tools.gemma4_pd.lane_lifecycle.lifecycle import (
    AUTHORITATIVE_STORM37B_SHA256,
    ExperimentLaneLock,
    LifecycleError,
    _stable_group_snapshot,
    _wait_service_ready,
    capture,
    load_snapshot,
    parse_quiescence_metrics,
    restore,
    stop,
)
from tools.gemma4_pd.lane_lifecycle.schema import (
    SCHEMA_VERSION,
    FileDigest,
    GpuDevice,
    GpuProcess,
    LaneSnapshot,
    MetricRecord,
    ProcessRecord,
    ProtectedGroup,
    ResourceLimit,
    ServiceRecord,
    SnapshotValidationError,
    SocketRecord,
    ToolIdentity,
    write_json_file,
)
from tools.gemma4_pd.lane_lifecycle.system import (
    HostInspectionError,
    LiveProcess,
    ProcessGroupNotFoundError,
    ProcessNotFoundError,
    decode_bytes,
    encode_bytes,
    parse_ss_output,
    record_launch_identity,
)

_METRICS = b"""# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0"} 0.0
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{engine="0"} 0.0
"""
_EXECUTABLE_BYTES = b"fake-executable"
_EXECUTABLE_SHA256 = hashlib.sha256(_EXECUTABLE_BYTES).hexdigest()
_LIMITS = (ResourceLimit(name="RLIMIT_NOFILE", soft=65535, hard=1048576),)
_ROLE_PORTS = {"p": 8810, "d1": 8811, "proxy": 8812, "router": 8813, "d2": 8815}
_ROLE_GPUS = {"p": (0, 1, 2, 3), "d1": (4,), "d2": (5,), "router": (), "proxy": ()}


def _process(
    pid: int,
    pgid: int,
    role: str,
    *,
    root: bool,
) -> ProcessRecord:
    """Build one deterministic process record.

    :param pid: Process identifier.
    :param pgid: Process-group identifier.
    :param role: Synthetic role name.
    :param root: Whether this process is the group root.
    :returns: Synthetic process record.
    """
    argv = (b"/bin/fake", b"--role", role.encode())
    environment = (b"PATH=/bin", b"UCX_TLS=tcp,shm,cuda_copy")
    return ProcessRecord(
        pid=pid,
        ppid=1 if root else pgid,
        pgid=pgid,
        sid=pgid,
        start_time_ticks=pid * 10,
        state="S",
        argv_b64=tuple(encode_bytes(item) for item in argv),
        environment_b64=tuple(encode_bytes(item) for item in environment),
        cwd_b64=encode_bytes(b"/tmp"),
        executable_link_b64=encode_bytes(b"/bin/fake"),
        executable_sha256=_EXECUTABLE_SHA256,
        executable_size=len(_EXECUTABLE_BYTES),
        executable_artifact=f"executables/{_EXECUTABLE_SHA256}",
        resource_limits=_LIMITS,
        stdin_target_b64=encode_bytes(b"/dev/null"),
        stdout_target_b64=encode_bytes(b"/tmp/original.log"),
        stderr_target_b64=encode_bytes(b"/tmp/original.log"),
    )


def _live(
    record: ProcessRecord, *, pid: int | None = None, pgid: int | None = None
) -> LiveProcess:
    """Convert a process record to fake live state.

    :param record: Captured process record.
    :param pid: Optional replacement PID.
    :param pgid: Optional replacement PGID.
    :returns: Fake live process.
    """
    selected_pid = record.pid if pid is None else pid
    selected_pgid = record.pgid if pgid is None else pgid
    return LiveProcess(
        pid=selected_pid,
        ppid=record.ppid if pid is None else 1,
        pgid=selected_pgid,
        sid=selected_pgid,
        start_time_ticks=(
            record.start_time_ticks if pid is None else selected_pid * 10
        ),
        state="S",
        argv=tuple(decode_bytes(item) for item in record.argv_b64),
        environment=tuple(decode_bytes(item) for item in record.environment_b64),
        cwd=decode_bytes(record.cwd_b64),
        executable_link=decode_bytes(record.executable_link_b64),
        executable_sha256=record.executable_sha256,
        executable_size=record.executable_size,
        resource_limits=record.resource_limits,
        stdin_target=b"/dev/null",
        stdout_target=b"/tmp/restore.log",
        stderr_target=b"/tmp/restore.log",
    )


def _listener(port: int, pid: int) -> SocketRecord:
    """Build one fake listener record.

    :param port: Listener port.
    :param pid: Owner PID.
    :returns: Synthetic socket record.
    """
    raw = f'LISTEN 0 2048 127.0.0.1:{port} 0.0.0.0:* users:(("x",pid={pid},fd=1))'
    return SocketRecord(
        state="LISTEN",
        receive_queue=0,
        send_queue=2048,
        local=f"127.0.0.1:{port}",
        peer="0.0.0.0:*",
        process_ids=(pid,),
        raw=raw,
    )


def _build_snapshot() -> LaneSnapshot:
    """Build a complete internally joined lifecycle snapshot.

    :returns: Validated synthetic snapshot.
    """
    services: list[ServiceRecord] = []
    gpu_processes: list[GpuProcess] = []
    listeners: list[SocketRecord] = []
    next_pid = 100
    for role in ("p", "d1", "proxy", "router", "d2"):
        root = _process(next_pid, next_pid, role, root=True)
        members = [root]
        for gpu_index in _ROLE_GPUS[role]:
            worker = _process(next_pid + gpu_index + 1, next_pid, role, root=False)
            members.append(worker)
            gpu_processes.append(
                GpuProcess(
                    gpu_index=gpu_index,
                    gpu_uuid=f"GPU-{gpu_index}",
                    pid=worker.pid,
                    process_name="worker",
                    used_memory_mib=100,
                )
            )
        services.append(
            ServiceRecord(
                role=role,
                port=_ROLE_PORTS[role],
                health_path=None if role == "proxy" else "/health",
                root=root,
                group_members=tuple(members),
                gpu_indices=_ROLE_GPUS[role],
                ucx_environment_b64=(encode_bytes(b"UCX_TLS=tcp,shm,cuda_copy"),),
                launcher_files=(),
            )
        )
        listeners.append(_listener(_ROLE_PORTS[role], root.pid))
        next_pid += 100

    protected: list[ProtectedGroup] = []
    for pgid, gpu_index, port in ((700, 6, 8820), (800, 7, 8821)):
        root = _process(pgid, pgid, f"prod{gpu_index}", root=True)
        worker = _process(pgid + 1, pgid, f"prod{gpu_index}", root=False)
        protected.append(
            ProtectedGroup(
                pgid=pgid,
                gpu_indices=(gpu_index,),
                group_members=(root, worker),
                listener_ports=(port,),
            )
        )
        listeners.append(_listener(port, root.pid))
        gpu_processes.append(
            GpuProcess(
                gpu_index=gpu_index,
                gpu_uuid=f"GPU-{gpu_index}",
                pid=worker.pid,
                process_name="production",
                used_memory_mib=200,
            )
        )
    proxy_root = _process(900, 900, "prod-proxy", root=True)
    protected.append(
        ProtectedGroup(
            pgid=900,
            gpu_indices=(),
            group_members=(proxy_root,),
            listener_ports=(8000,),
        )
    )
    listeners.append(_listener(8000, proxy_root.pid))

    executable = FileDigest(
        path_b64=encode_bytes(b"/repo/tools/gemma4_pd/lane_lifecycle/lifecycle.py"),
        size=4,
        sha256=hashlib.sha256(b"tool").hexdigest(),
        symlink_target_b64=None,
    )
    snapshot = LaneSnapshot(
        schema_version=SCHEMA_VERSION,
        captured_at_utc="2026-07-13T00:00:00Z",
        hostname="fake-host",
        boot_id="fake-boot",
        kernel_release="fake-kernel",
        tool_identity=ToolIdentity(
            git_head="1" * 40,
            git_tree="2" * 40,
            tool_files=(executable,),
        ),
        storm_archive=FileDigest(
            path_b64=encode_bytes(b"/sealed/storm37b.tar.zst"),
            size=123,
            sha256=AUTHORITATIVE_STORM37B_SHA256,
            symlink_target_b64=None,
        ),
        services=tuple(services),
        protected_groups=tuple(protected),
        gpu_devices=tuple(
            GpuDevice(
                index=index,
                uuid=f"GPU-{index}",
                pci_bus_id=f"0000:{index:02x}:00.0",
                name="B300",
            )
            for index in range(8)
        ),
        gpu_processes=tuple(sorted(gpu_processes, key=lambda item: item.gpu_index)),
        listeners=tuple(listeners),
        connections=(),
        metrics=tuple(
            MetricRecord(
                role=role,
                url=f"http://127.0.0.1:{_ROLE_PORTS[role]}/metrics",
                running=0.0,
                waiting=0.0,
                body_sha256=hashlib.sha256(_METRICS).hexdigest(),
                body_artifact=f"metrics/{role}.prom",
            )
            for role in ("p", "d1", "d2")
        ),
    )
    return LaneSnapshot.from_dict(snapshot.to_dict())


class FakeHost:
    """Model process, port, and GPU ownership without host mutation."""

    snapshot: LaneSnapshot
    groups: dict[int, tuple[LiveProcess, ...]]
    roots: dict[str, LiveProcess]
    signals: list[tuple[int, int]]
    launches: list[tuple[str, tuple[object, ...], Path]]
    now: float
    next_pid: int

    def __init__(self, snapshot: LaneSnapshot, *, running: bool) -> None:
        """Initialize fake host state.

        :param snapshot: Synthetic lane snapshot.
        :param running: Populate the experiment services when true.
        """
        self.snapshot = snapshot
        self.groups = {}
        self.roots = {}
        self.signals = []
        self.launches = []
        self.now = 0.0
        self.next_pid = 2000
        if running:
            for service in snapshot.services:
                group = tuple(_live(member) for member in service.group_members)
                self.groups[service.root.pgid] = group
                self.roots[service.role] = group[0]
        for group_record in snapshot.protected_groups:
            self.groups[group_record.pgid] = tuple(
                _live(member) for member in group_record.group_members
            )

    def boot_id(self) -> str:
        """Return the fake boot ID.

        :returns: Boot identifier.
        """
        return self.snapshot.boot_id

    def hostname(self) -> str:
        """Return the fake hostname.

        :returns: Hostname.
        """
        return self.snapshot.hostname

    def kernel_release(self) -> str:
        """Return the fake kernel release.

        :returns: Kernel release.
        """
        return self.snapshot.kernel_release

    def tool_identity(self) -> ToolIdentity:
        """Return the fake lifecycle-tool identity.

        :returns: Tool identity.
        """
        return self.snapshot.tool_identity

    def process(self, pid: int) -> LiveProcess:
        """Read one fake process.

        :param pid: Process identifier.
        :returns: Fake live process.
        """
        for group in self.groups.values():
            for process in group:
                if process.pid == pid:
                    return process
        raise ProcessNotFoundError(f"process {pid} is absent")

    def group_members(self, pgid: int) -> tuple[LiveProcess, ...]:
        """Read one fake process group.

        :param pgid: Process-group identifier.
        :returns: Fake group members.
        """
        if pgid not in self.groups:
            raise ProcessGroupNotFoundError(f"group {pgid} is absent")
        return self.groups[pgid]

    def listeners(self) -> tuple[SocketRecord, ...]:
        """Return current fake listeners.

        :returns: Listener inventory.
        """
        records = [
            _listener(_ROLE_PORTS[role], root.pid) for role, root in self.roots.items()
        ]
        for group in self.snapshot.protected_groups:
            if group.pgid not in self.groups:
                continue
            root_pid = self.groups[group.pgid][0].pid
            records.extend(_listener(port, root_pid) for port in group.listener_ports)
        return tuple(records)

    def connections(self) -> tuple[SocketRecord, ...]:
        """Return an empty fake connection inventory.

        :returns: Empty connection tuple.
        """
        return ()

    def gpu_state(self) -> tuple[tuple[GpuDevice, ...], tuple[GpuProcess, ...]]:
        """Join fake compute processes to running groups.

        :returns: Device and compute-process inventories.
        """
        processes: list[GpuProcess] = []
        for role, root in self.roots.items():
            members = self.groups[root.pgid]
            for gpu_index, member in zip(_ROLE_GPUS[role], members[1:], strict=True):
                processes.append(
                    GpuProcess(
                        gpu_index=gpu_index,
                        gpu_uuid=f"GPU-{gpu_index}",
                        pid=member.pid,
                        process_name="worker",
                        used_memory_mib=100,
                    )
                )
        processes.extend(
            process for process in self.snapshot.gpu_processes if process.gpu_index >= 6
        )
        return self.snapshot.gpu_devices, tuple(
            sorted(processes, key=lambda item: item.gpu_index)
        )

    def http_get(self, url: str, timeout_seconds: float) -> tuple[int, bytes]:
        """Serve fake health and metric endpoints.

        :param url: Local URL.
        :param timeout_seconds: Ignored request timeout.
        :returns: HTTP status and body.
        """
        del timeout_seconds
        port = int(url.split(":")[2].split("/")[0])
        if not any(_ROLE_PORTS[role] == port for role in self.roots):
            raise HostInspectionError(f"port {port} is absent")
        return 200, _METRICS if url.endswith("/metrics") else b""

    def copy_process_executable(self, pid: int, destination: Path) -> FileDigest:
        """Materialize the shared fake executable.

        :param pid: Source process identifier.
        :param destination: New evidence path.
        :returns: Copied executable digest.
        """
        del pid
        descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
            0o600,
        )
        try:
            os.write(descriptor, _EXECUTABLE_BYTES)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return FileDigest(
            path_b64=encode_bytes(os.fsencode(destination.resolve())),
            size=len(_EXECUTABLE_BYTES),
            sha256=_EXECUTABLE_SHA256,
            symlink_target_b64=None,
        )

    def file_digest(self, path: bytes) -> FileDigest:
        """Return one fake external file identity.

        :param path: Raw absolute path.
        :returns: Synthetic file digest.
        """
        if path == b"/bin/fake":
            return FileDigest(
                path_b64=encode_bytes(path),
                size=len(_EXECUTABLE_BYTES),
                sha256=_EXECUTABLE_SHA256,
                symlink_target_b64=None,
            )
        if path == b"/sealed/storm37b.tar.zst":
            return self.snapshot.storm_archive
        raise HostInspectionError(f"unknown file {path!r}")

    def kill_group(self, pgid: int, signal_number: int) -> None:
        """Remove one unprotected fake process group.

        :param pgid: Process-group identifier.
        :param signal_number: Recorded POSIX signal.
        """
        protected_pgids = {group.pgid for group in self.snapshot.protected_groups}
        assert pgid not in protected_pgids
        self.signals.append((pgid, signal_number))
        self.groups.pop(pgid)
        for role, root in list(self.roots.items()):
            if root.pgid == pgid:
                del self.roots[role]

    def launch_exact(
        self,
        process: ProcessRecord,
        executable: Path,
        stdout_path: Path,
        stderr_path: Path,
    ) -> int:
        """Launch one fake service from its exact process record.

        :param process: Captured root process.
        :param executable: Verified executable path.
        :param stdout_path: Private stdout path.
        :param stderr_path: Private stderr path.
        :returns: New root PID.
        """
        del stdout_path, stderr_path
        role = decode_bytes(process.argv_b64[2]).decode()
        pid = self.next_pid
        self.next_pid += 100
        root = _live(process, pid=pid, pgid=pid)
        members = [root]
        for gpu_index in _ROLE_GPUS[role]:
            worker_record = _process(pid + gpu_index + 1, pid, role, root=False)
            members.append(_live(worker_record))
        self.groups[pid] = tuple(members)
        self.roots[role] = root
        self.launches.append((role, record_launch_identity(process), executable))
        return pid

    def child_exit_status(self, pid: int) -> int | None:
        """Report a live fake child.

        :param pid: Child PID.
        :returns: ``None`` while present.
        """
        self.process(pid)
        return None

    def sleep(self, seconds: float) -> None:
        """Advance fake monotonic time.

        :param seconds: Time increment.
        """
        self.now += seconds

    def monotonic(self) -> float:
        """Return fake monotonic time.

        :returns: Fake seconds.
        """
        return self.now


def _seal_snapshot(path: Path, snapshot: LaneSnapshot) -> None:
    """Write a private sealed synthetic artifact.

    :param path: New artifact directory.
    :param snapshot: Synthetic snapshot.
    """
    path.mkdir(mode=0o700)
    executables = path / "executables"
    metrics = path / "metrics"
    executables.mkdir(mode=0o700)
    metrics.mkdir(mode=0o700)
    executable = executables / _EXECUTABLE_SHA256
    executable.write_bytes(_EXECUTABLE_BYTES)
    executable.chmod(0o600)
    for role in ("p", "d1", "d2"):
        metric = metrics / f"{role}.prom"
        metric.write_bytes(_METRICS)
        metric.chmod(0o600)
    write_json_file(path / "snapshot.json", snapshot.to_dict())
    files = sorted(
        item for item in path.rglob("*") if item.is_file() and item.name != "SHA256SUMS"
    )
    seal = "".join(
        f"{hashlib.sha256(item.read_bytes()).hexdigest()}  "
        f"{item.relative_to(path).as_posix()}\n"
        for item in files
    )
    seal_path = path / "SHA256SUMS"
    seal_path.write_text(seal, encoding="ascii")
    seal_path.chmod(0o600)


@pytest.fixture(autouse=True)
def _private_experiment_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Route each test to an isolated host-wide lock path.

    :param tmp_path: Per-test temporary directory.
    :param monkeypatch: Pytest monkeypatch fixture.
    """
    monkeypatch.setattr(
        lifecycle_module,
        "EXPERIMENT_LANE_LOCK",
        tmp_path / "locks" / "gemma4-experiment-lane.lock",
    )


def test_schema_rejects_duplicate_keys_and_non_finite_numbers() -> None:
    """Snapshot parsing rejects ambiguous and non-finite JSON."""
    with pytest.raises(SnapshotValidationError, match="duplicate JSON key"):
        LaneSnapshot.from_json_bytes(b'{"schema_version": 1, "schema_version": 1}')
    with pytest.raises(SnapshotValidationError, match="non-finite"):
        LaneSnapshot.from_json_bytes(b'{"value": NaN}')


def test_mutating_cli_requires_explicit_mode() -> None:
    """Stop and restore cannot execute through an omitted mode flag."""
    with pytest.raises(SystemExit):
        main(["stop", "--artifact-dir", "/private/snapshot"])
    with pytest.raises(SystemExit):
        main(["restore", "--artifact-dir", "/private/snapshot"])


def test_capture_seals_private_full_environment_with_fake_host(
    tmp_path: Path,
) -> None:
    """Capture preserves full environment data only in private sealed files.

    :param tmp_path: Per-test temporary directory.
    """
    template = _build_snapshot()
    host = FakeHost(template, running=True)
    artifact = tmp_path / "captured"
    snapshot = capture(
        artifact,
        storm_archive=Path("/sealed/storm37b.tar.zst"),
        host=host,
    )

    assert (
        snapshot.service("p").root.environment_b64
        == template.service("p").root.environment_b64
    )
    assert load_snapshot(artifact) == snapshot
    assert artifact.stat().st_mode & 0o777 == 0o700
    for path in artifact.rglob("*"):
        if path.is_file():
            assert path.stat().st_mode & 0o777 == 0o600


def test_metrics_parser_requires_exact_finite_quiescence_gauges() -> None:
    """Quiescence parsing rejects missing and non-finite gauges."""
    assert parse_quiescence_metrics(_METRICS) == (0.0, 0.0)
    with pytest.raises(LifecycleError, match="missing"):
        parse_quiescence_metrics(b"vllm:num_requests_running 0\n")
    with pytest.raises(LifecycleError, match="unsafe"):
        parse_quiescence_metrics(
            b"vllm:num_requests_running NaN\nvllm:num_requests_waiting 0\n"
        )


def test_ss_parser_preserves_listener_owner() -> None:
    """Socket parsing preserves exact listener PID ownership."""
    records = parse_ss_output(
        'LISTEN 0 2048 127.0.0.1:8810 0.0.0.0:* users:(("python",pid=42,fd=7))\n'
    )
    assert records[0].process_ids == (42,)
    assert records[0].local == "127.0.0.1:8810"


def test_snapshot_loader_rejects_mode_drift_and_symlinks(tmp_path: Path) -> None:
    """Snapshot loading rejects public modes and operation symlinks.

    :param tmp_path: Per-test temporary directory.
    """
    snapshot = _build_snapshot()
    artifact = tmp_path / "snapshot"
    _seal_snapshot(artifact, snapshot)
    assert load_snapshot(artifact) == snapshot
    (artifact / "snapshot.json").chmod(0o644)
    with pytest.raises(LifecycleError, match="0600"):
        load_snapshot(artifact)

    (artifact / "snapshot.json").chmod(0o600)
    operations = artifact / "operations"
    operations.mkdir(mode=0o700)
    (operations / "escape").symlink_to(tmp_path)
    with pytest.raises(LifecycleError, match="symlink"):
        load_snapshot(artifact)


def test_stop_dry_run_never_signals_and_execute_uses_safe_order(
    tmp_path: Path,
) -> None:
    """Stop dry-run is inert and execution respects dependency order.

    :param tmp_path: Per-test temporary directory.
    """
    snapshot = _build_snapshot()
    artifact = tmp_path / "snapshot"
    _seal_snapshot(artifact, snapshot)
    host = FakeHost(snapshot, running=True)

    result = stop(artifact, dry_run=True, host=host)
    assert result["status"] == "DRY_RUN"
    assert host.signals == []

    result = stop(artifact, dry_run=False, host=host)
    assert result["status"] == "PASS"
    expected_pgids = [
        snapshot.service(role).root.pgid
        for role in ("router", "proxy", "d1", "d2", "p")
    ]
    assert host.signals == [(pgid, signal.SIGTERM) for pgid in expected_pgids]
    assert all(group.pgid in host.groups for group in snapshot.protected_groups)


def test_restore_is_state_aware_and_exact(tmp_path: Path) -> None:
    """Restore retains valid roles and launches missing roles exactly.

    :param tmp_path: Per-test temporary directory.
    """
    snapshot = _build_snapshot()
    artifact = tmp_path / "snapshot"
    _seal_snapshot(artifact, snapshot)
    host = FakeHost(snapshot, running=True)
    host.kill_group(snapshot.service("router").root.pgid, signal.SIGTERM)
    host.kill_group(snapshot.service("proxy").root.pgid, signal.SIGTERM)
    host.kill_group(snapshot.service("d1").root.pgid, signal.SIGTERM)

    dry_run = restore(artifact, dry_run=True, host=host)
    actions = {item["role"]: item["action"] for item in dry_run["plan"]}
    assert all("environment_b64" not in item for item in dry_run["plan"])
    assert actions == {
        "p": "retain",
        "d1": "launch",
        "d2": "retain",
        "router": "launch",
        "proxy": "launch",
    }

    result = restore(artifact, dry_run=False, host=host)
    assert result["status"] == "PASS"
    assert [item[0] for item in host.launches] == ["d1", "router", "proxy"]
    assert all(item[2] == Path("/bin/fake") for item in host.launches)
    assert all(group.pgid in host.groups for group in snapshot.protected_groups)


def test_partial_stop_failure_restores_only_missing_roles(tmp_path: Path) -> None:
    """A partial stop failure transactionally recovers missing roles.

    :param tmp_path: Per-test temporary directory.
    """
    snapshot = _build_snapshot()
    artifact = tmp_path / "snapshot"
    _seal_snapshot(artifact, snapshot)
    host = FakeHost(snapshot, running=True)
    original_kill = host.kill_group
    failed = False

    def fail_second_decode(pgid: int, signal_number: int) -> None:
        """Inject one D2 signal failure.

        :param pgid: Process-group identifier.
        :param signal_number: POSIX signal number.
        """
        nonlocal failed
        if pgid == snapshot.service("d2").root.pgid and not failed:
            failed = True
            raise HostInspectionError("injected D2 signal failure")
        original_kill(pgid, signal_number)

    host.kill_group = fail_second_decode  # type: ignore[method-assign]
    with pytest.raises(LifecycleError, match="running-equivalent state"):
        stop(artifact, dry_run=False, host=host)
    assert set(host.roots) == {"p", "d1", "d2", "router", "proxy"}
    assert [item[0] for item in host.launches] == ["d1", "router", "proxy"]
    assert all(group.pgid in host.groups for group in snapshot.protected_groups)


def test_restore_rejects_ingress_above_missing_engine(tmp_path: Path) -> None:
    """Restore rejects ingress that remains above an absent engine.

    :param tmp_path: Per-test temporary directory.
    """
    snapshot = _build_snapshot()
    artifact = tmp_path / "snapshot"
    _seal_snapshot(artifact, snapshot)
    host = FakeHost(snapshot, running=True)
    host.kill_group(snapshot.service("d1").root.pgid, signal.SIGTERM)

    with pytest.raises(LifecycleError, match="ingress remains live"):
        restore(artifact, dry_run=True, host=host)
    assert host.launches == []


def test_production_proxy_identity_drift_blocks_every_signal(tmp_path: Path) -> None:
    """Production proxy drift prevents every experiment signal.

    :param tmp_path: Per-test temporary directory.
    """
    snapshot = _build_snapshot()
    artifact = tmp_path / "snapshot"
    _seal_snapshot(artifact, snapshot)
    host = FakeHost(snapshot, running=True)
    proxy_group = next(
        group for group in snapshot.protected_groups if 8000 in group.listener_ports
    )
    del host.groups[proxy_group.pgid]

    with pytest.raises(ProcessGroupNotFoundError):
        stop(artifact, dry_run=False, host=host)
    assert host.signals == []


def test_group_snapshot_does_not_treat_inspection_failure_as_absence(
    tmp_path: Path,
) -> None:
    """Group inspection errors remain distinct from proven absence.

    :param tmp_path: Per-test temporary directory.
    """
    del tmp_path
    snapshot = _build_snapshot()
    host = FakeHost(snapshot, running=True)

    def fail_group(pgid: int) -> tuple[LiveProcess, ...]:
        """Inject an ambiguous group inspection failure.

        :param pgid: Process-group identifier.
        """
        raise HostInspectionError(f"cannot inspect {pgid}")

    host.group_members = fail_group  # type: ignore[method-assign]
    with pytest.raises(HostInspectionError, match="cannot inspect"):
        _stable_group_snapshot(host, snapshot.service("p").root.pgid)


def test_experiment_lane_lock_rejects_nonblocking_contention() -> None:
    """The shared experiment-lane lock rejects concurrent ownership."""
    with (
        ExperimentLaneLock(),
        pytest.raises(LifecycleError, match="already locked"),
        ExperimentLaneLock(),
    ):
        raise AssertionError("contended lock unexpectedly acquired")


def test_boot_wait_fails_immediately_when_launched_root_exits() -> None:
    """A terminated root fails readiness without consuming its timeout."""
    snapshot = _build_snapshot()
    host = FakeHost(snapshot, running=True)
    pid = snapshot.service("p").root.pid

    def exited(child_pid: int) -> int | None:
        """Report an injected child exit status.

        :param child_pid: Polled child PID.
        :returns: Injected exit code.
        """
        assert child_pid == pid
        return 42

    host.child_exit_status = exited  # type: ignore[method-assign]
    with pytest.raises(LifecycleError, match="status 42"):
        _wait_service_ready(host, snapshot, "p", pid, 1800.0)
    assert host.now == 0.0
