import base64
import binascii
import csv
import hashlib
import os
import re
import resource
import select
import shutil
import signal
import socket
import subprocess
import time
import traceback
import urllib.error
import urllib.request
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from tools.gemma4_pd.lane_lifecycle.schema import (
    FileDigest,
    GpuDevice,
    GpuProcess,
    ProcessRecord,
    ResourceLimit,
    SocketRecord,
    ToolIdentity,
)

_PID_PATTERN = re.compile(r"pid=(\d+)")
_RESOURCE_LIMIT_NAMES = tuple(
    name
    for name in (
        "RLIMIT_AS",
        "RLIMIT_CORE",
        "RLIMIT_CPU",
        "RLIMIT_DATA",
        "RLIMIT_FSIZE",
        "RLIMIT_LOCKS",
        "RLIMIT_MEMLOCK",
        "RLIMIT_MSGQUEUE",
        "RLIMIT_NICE",
        "RLIMIT_NOFILE",
        "RLIMIT_NPROC",
        "RLIMIT_RSS",
        "RLIMIT_RTPRIO",
        "RLIMIT_RTTIME",
        "RLIMIT_SIGPENDING",
        "RLIMIT_STACK",
    )
    if hasattr(resource, name)
)


class HostInspectionError(RuntimeError):
    """Indicate that host state could not be inspected unambiguously."""


class ProcessNotFoundError(HostInspectionError):
    """Indicate that an exact process identifier is absent."""


class ProcessGroupNotFoundError(HostInspectionError):
    """Indicate that an exact process group is absent."""


@dataclass(frozen=True, slots=True)
class LiveProcess:
    """Contain raw process state read from procfs.

    :ivar pid: Process identifier.
    :ivar ppid: Parent process identifier.
    :ivar pgid: Process-group identifier.
    :ivar sid: Session identifier.
    :ivar start_time_ticks: Kernel start time.
    :ivar state: Kernel process state.
    :ivar argv: Raw argument vector.
    :ivar environment: Raw environment entries.
    :ivar cwd: Raw working directory.
    :ivar executable_link: Raw executable-link target.
    :ivar executable_sha256: Digest of the executable inode in use.
    :ivar executable_size: Executable size in bytes.
    :ivar resource_limits: Exact POSIX resource-limit vector.
    :ivar stdin_target: Raw fd 0 target, when readable.
    :ivar stdout_target: Raw fd 1 target, when readable.
    :ivar stderr_target: Raw fd 2 target, when readable.
    """

    pid: int
    ppid: int
    pgid: int
    sid: int
    start_time_ticks: int
    state: str
    argv: tuple[bytes, ...]
    environment: tuple[bytes, ...]
    cwd: bytes
    executable_link: bytes
    executable_sha256: str
    executable_size: int
    resource_limits: tuple[ResourceLimit, ...]
    stdin_target: bytes | None
    stdout_target: bytes | None
    stderr_target: bytes | None

    def instance_identity(self) -> tuple[object, ...]:
        """Return fields that identify the exact process instance.

        :returns: Comparable process identity.
        """
        return (
            self.pid,
            self.ppid,
            self.pgid,
            self.sid,
            self.start_time_ticks,
            self.argv,
            self.environment,
            self.cwd,
            self.executable_sha256,
            self.resource_limits,
            self.stdin_target,
        )

    def launch_identity(self) -> tuple[object, ...]:
        """Return fields that must survive an exact restart.

        :returns: Comparable launch identity.
        """
        return (
            self.argv,
            self.environment,
            self.cwd,
            self.executable_sha256,
            self.executable_size,
            self.resource_limits,
            self.stdin_target,
        )


class Host(Protocol):
    """Define the host operations used by lifecycle orchestration."""

    def boot_id(self) -> str:
        """Return the current Linux boot identifier.

        :returns: Boot identifier.
        """
        ...

    def hostname(self) -> str:
        """Return the current hostname.

        :returns: Hostname.
        """
        ...

    def kernel_release(self) -> str:
        """Return the current kernel release.

        :returns: Kernel release.
        """
        ...

    def process(self, pid: int) -> LiveProcess:
        """Read one exact process.

        :param pid: Process identifier.
        :returns: Live process state.
        """
        ...

    def group_members(self, pgid: int) -> tuple[LiveProcess, ...]:
        """Read a complete process group.

        :param pgid: Process-group identifier.
        :returns: Sorted process group.
        """
        ...

    def listeners(self) -> tuple[SocketRecord, ...]:
        """Read all TCP listeners.

        :returns: Parsed listener records.
        """
        ...

    def connections(self) -> tuple[SocketRecord, ...]:
        """Read all non-listening TCP sockets.

        :returns: Parsed connection records.
        """
        ...

    def gpu_state(self) -> tuple[tuple[GpuDevice, ...], tuple[GpuProcess, ...]]:
        """Read physical devices and compute processes.

        :returns: Device and process inventories.
        """
        ...

    def http_get(self, url: str, timeout_seconds: float) -> tuple[int, bytes]:
        """Fetch one local HTTP endpoint.

        :param url: Endpoint URL.
        :param timeout_seconds: Request timeout.
        :returns: HTTP status and body.
        """
        ...

    def tool_identity(self) -> ToolIdentity:
        """Return the clean repository and lifecycle-tool identity.

        :returns: Tool identity.
        """
        ...

    def copy_process_executable(self, pid: int, destination: Path) -> FileDigest:
        """Copy the executable inode currently held by one process.

        :param pid: Process identifier.
        :param destination: New snapshot file.
        :returns: Digest of the copied executable.
        """
        ...

    def file_digest(self, path: bytes) -> FileDigest:
        """Read one file's content and path identity.

        :param path: Raw absolute path.
        :returns: File digest.
        """
        ...

    def kill_group(self, pgid: int, signal_number: int) -> None:
        """Signal one process group.

        :param pgid: Process-group identifier.
        :param signal_number: POSIX signal number.
        """
        ...

    def launch_exact(
        self,
        process: ProcessRecord,
        executable: Path,
        stdout_path: Path,
        stderr_path: Path,
    ) -> int:
        """Launch one exact process image in a new session.

        :param process: Captured process image.
        :param executable: Verified executable path.
        :param stdout_path: New stdout log.
        :param stderr_path: New stderr log.
        :returns: Child process identifier.
        """
        ...

    def child_exit_status(self, pid: int) -> int | None:
        """Poll one directly launched child without blocking.

        :param pid: Child process identifier.
        :returns: Exit code when terminated, otherwise ``None``.
        """
        ...

    def sleep(self, seconds: float) -> None:
        """Wait for host state to advance.

        :param seconds: Wait duration.
        """
        ...

    def monotonic(self) -> float:
        """Return a monotonic timestamp.

        :returns: Seconds on a monotonic clock.
        """
        ...


def encode_bytes(value: bytes) -> str:
    """Encode raw Linux bytes for JSON.

    :param value: Raw bytes.
    :returns: ASCII Base64 text.
    """
    return base64.b64encode(value).decode("ascii")


def decode_bytes(value: str) -> bytes:
    """Decode raw Linux bytes from JSON.

    :param value: ASCII Base64 text.
    :returns: Raw bytes.
    :raises HostInspectionError: If the encoding is invalid.
    """
    try:
        return base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as error:
        raise HostInspectionError(
            "invalid Base64 data in lifecycle snapshot"
        ) from error


def environment_mapping(entries: tuple[bytes, ...]) -> dict[bytes, bytes]:
    """Convert exact procfs environment entries into an execve mapping.

    :param entries: Raw ``KEY=VALUE`` entries in captured order.
    :returns: Byte-preserving environment mapping.
    :raises HostInspectionError: If an entry is malformed or duplicates a key.
    """
    result: dict[bytes, bytes] = {}
    for entry in entries:
        key, separator, value = entry.partition(b"=")
        if separator != b"=" or len(key) == 0 or b"\x00" in key or b"\x00" in value:
            raise HostInspectionError("process environment contains an invalid entry")
        if key in result:
            raise HostInspectionError(
                f"process environment contains duplicate key {key!r}"
            )
        result[key] = value
    return result


def record_from_live(process: LiveProcess, executable_artifact: str) -> ProcessRecord:
    """Convert raw live state to a serializable process record.

    :param process: Live process state.
    :param executable_artifact: Snapshot-relative executable copy.
    :returns: Serializable process record.
    """
    environment_mapping(process.environment)
    return ProcessRecord(
        pid=process.pid,
        ppid=process.ppid,
        pgid=process.pgid,
        sid=process.sid,
        start_time_ticks=process.start_time_ticks,
        state=process.state,
        argv_b64=tuple(encode_bytes(item) for item in process.argv),
        environment_b64=tuple(encode_bytes(item) for item in process.environment),
        cwd_b64=encode_bytes(process.cwd),
        executable_link_b64=encode_bytes(process.executable_link),
        executable_sha256=process.executable_sha256,
        executable_size=process.executable_size,
        executable_artifact=executable_artifact,
        resource_limits=process.resource_limits,
        stdin_target_b64=(
            None if process.stdin_target is None else encode_bytes(process.stdin_target)
        ),
        stdout_target_b64=(
            None
            if process.stdout_target is None
            else encode_bytes(process.stdout_target)
        ),
        stderr_target_b64=(
            None
            if process.stderr_target is None
            else encode_bytes(process.stderr_target)
        ),
    )


def record_instance_identity(record: ProcessRecord) -> tuple[object, ...]:
    """Return exact-instance fields from a snapshot process record.

    :param record: Snapshot process record.
    :returns: Comparable process identity.
    """
    return (
        record.pid,
        record.ppid,
        record.pgid,
        record.sid,
        record.start_time_ticks,
        tuple(decode_bytes(item) for item in record.argv_b64),
        tuple(decode_bytes(item) for item in record.environment_b64),
        decode_bytes(record.cwd_b64),
        record.executable_sha256,
        record.resource_limits,
        None
        if record.stdin_target_b64 is None
        else decode_bytes(record.stdin_target_b64),
    )


def record_launch_identity(record: ProcessRecord) -> tuple[object, ...]:
    """Return exact-launch fields from a snapshot process record.

    :param record: Snapshot process record.
    :returns: Comparable launch identity.
    """
    return (
        tuple(decode_bytes(item) for item in record.argv_b64),
        tuple(decode_bytes(item) for item in record.environment_b64),
        decode_bytes(record.cwd_b64),
        record.executable_sha256,
        record.executable_size,
        record.resource_limits,
        None
        if record.stdin_target_b64 is None
        else decode_bytes(record.stdin_target_b64),
    )


def parse_ss_output(output: str) -> tuple[SocketRecord, ...]:
    """Parse stable one-line ``ss -H -O`` output.

    :param output: Complete command output.
    :returns: Parsed socket records.
    :raises HostInspectionError: If any row is malformed.
    """
    records: list[SocketRecord] = []
    for line_number, raw in enumerate(output.splitlines(), start=1):
        if len(raw.strip()) == 0:
            continue
        fields = raw.split(maxsplit=5)
        if len(fields) < 5:
            raise HostInspectionError(f"malformed ss row {line_number}: {raw!r}")
        try:
            receive_queue = int(fields[1])
            send_queue = int(fields[2])
        except ValueError as error:
            raise HostInspectionError(
                f"invalid queue size in ss row {line_number}: {raw!r}"
            ) from error
        process_text = "" if len(fields) == 5 else fields[5]
        records.append(
            SocketRecord(
                state=fields[0],
                receive_queue=receive_queue,
                send_queue=send_queue,
                local=fields[3],
                peer=fields[4],
                process_ids=tuple(
                    sorted({int(match) for match in _PID_PATTERN.findall(process_text)})
                ),
                raw=raw,
            )
        )
    return tuple(records)


def endpoint_port(endpoint: str) -> int:
    """Extract a numeric port from an ``ss`` endpoint.

    :param endpoint: Raw endpoint text.
    :returns: Numeric port.
    :raises HostInspectionError: If the endpoint is not numeric.
    """
    _, separator, port_text = endpoint.rpartition(":")
    if separator != ":":
        raise HostInspectionError(f"socket endpoint has no port: {endpoint!r}")
    try:
        return int(port_text)
    except ValueError as error:
        raise HostInspectionError(
            f"socket endpoint has non-numeric port: {endpoint!r}"
        ) from error


def listener_owner(listeners: tuple[SocketRecord, ...], port: int) -> int:
    """Resolve one exact listener process by port.

    :param listeners: Complete listener inventory.
    :param port: Required TCP port.
    :returns: Sole owning process identifier.
    :raises HostInspectionError: If listener identity is absent or ambiguous.
    """
    matches = [record for record in listeners if endpoint_port(record.local) == port]
    if len(matches) != 1:
        raise HostInspectionError(
            f"port {port} must have exactly one TCP listener; found {len(matches)}"
        )
    if len(matches[0].process_ids) != 1:
        raise HostInspectionError(
            f"port {port} listener must expose exactly one owner PID"
        )
    return matches[0].process_ids[0]


def environment_subset(
    entries: tuple[bytes, ...], prefixes: tuple[bytes, ...]
) -> tuple[bytes, ...]:
    """Select environment entries by key prefix.

    :param entries: Raw environment entries.
    :param prefixes: Required key prefixes.
    :returns: Selected entries in original order.
    """
    selected: list[bytes] = []
    for entry in entries:
        key, separator, _ = entry.partition(b"=")
        if separator != b"=":
            raise HostInspectionError("process environment contains an invalid entry")
        if any(key.startswith(prefix) for prefix in prefixes):
            selected.append(entry)
    return tuple(selected)


class LinuxHost:
    """Inspect and operate the Linux host without shell interpretation."""

    _proc_root: Path

    def __init__(self, proc_root: Path = Path("/proc")) -> None:
        """Initialize Linux host access.

        :param proc_root: Procfs mount point.
        """
        self._proc_root = proc_root

    def boot_id(self) -> str:
        """Return the current Linux boot identifier.

        :returns: Boot identifier.
        """
        return (self._proc_root / "sys/kernel/random/boot_id").read_text().strip()

    def hostname(self) -> str:
        """Return the current hostname.

        :returns: Hostname.
        """
        return socket.gethostname()

    def kernel_release(self) -> str:
        """Return the current kernel release.

        :returns: Kernel release.
        """
        return os.uname().release

    def _proc_path(self, pid: int, name: str) -> bytes:
        """Build a raw procfs path.

        :param pid: Process identifier.
        :param name: Process-relative procfs name.
        :returns: Raw filesystem path.
        """
        return os.fsencode(self._proc_root / str(pid) / name)

    def _readlink(self, pid: int, name: str) -> bytes | None:
        """Read an optional raw procfs symlink.

        :param pid: Process identifier.
        :param name: Process-relative procfs name.
        :returns: Raw target when readable.
        """
        try:
            return os.readlink(self._proc_path(pid, name))
        except OSError:
            return None

    def process(self, pid: int) -> LiveProcess:
        """Read one exact process.

        :param pid: Process identifier.
        :returns: Live process state.
        :raises HostInspectionError: If procfs state is incomplete or unstable.
        """
        proc_dir = self._proc_root / str(pid)
        try:
            stat = (proc_dir / "stat").read_bytes()
            marker = stat.rfind(b") ")
            if marker < 0:
                raise HostInspectionError(f"malformed /proc/{pid}/stat")
            fields = stat[marker + 2 :].split()
            if len(fields) < 20:
                raise HostInspectionError(f"short /proc/{pid}/stat")
            state = fields[0].decode("ascii")
            ppid = int(fields[1])
            pgid = int(fields[2])
            sid = int(fields[3])
            start_time_ticks = int(fields[19])
            argv = _split_nul((proc_dir / "cmdline").read_bytes())
            environment = _split_nul((proc_dir / "environ").read_bytes())
            cwd = os.readlink(self._proc_path(pid, "cwd"))
            executable_link = os.readlink(self._proc_path(pid, "exe"))
            executable_sha256, executable_size = _sha256_file(
                _path_from_bytes(self._proc_path(pid, "exe"))
            )
            resource_limits = _read_resource_limits(pid)
        except FileNotFoundError as error:
            raise ProcessNotFoundError(f"process {pid} is absent") from error
        except (OSError, ValueError, UnicodeDecodeError) as error:
            raise HostInspectionError(
                f"cannot inspect process {pid}: {error}"
            ) from error
        if len(argv) == 0:
            raise HostInspectionError(f"process {pid} has an empty argument vector")
        environment_mapping(environment)
        return LiveProcess(
            pid=pid,
            ppid=ppid,
            pgid=pgid,
            sid=sid,
            start_time_ticks=start_time_ticks,
            state=state,
            argv=argv,
            environment=environment,
            cwd=cwd,
            executable_link=executable_link,
            executable_sha256=executable_sha256,
            executable_size=executable_size,
            resource_limits=resource_limits,
            stdin_target=self._readlink(pid, "fd/0"),
            stdout_target=self._readlink(pid, "fd/1"),
            stderr_target=self._readlink(pid, "fd/2"),
        )

    def group_members(self, pgid: int) -> tuple[LiveProcess, ...]:
        """Read a complete process group.

        :param pgid: Process-group identifier.
        :returns: Sorted process group.
        """
        members: list[LiveProcess] = []
        for entry in self._proc_root.iterdir():
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            try:
                candidate_pgid = self._process_group_id(pid)
            except ProcessNotFoundError:
                continue
            if candidate_pgid != pgid:
                continue
            process = self.process(pid)
            if process.pgid == pgid:
                members.append(process)
        if len(members) == 0:
            try:
                os.killpg(pgid, 0)
            except ProcessLookupError as error:
                raise ProcessGroupNotFoundError(
                    f"process group {pgid} is absent"
                ) from error
            except OSError as error:
                raise HostInspectionError(
                    f"cannot test process group {pgid}: {error}"
                ) from error
            raise HostInspectionError(
                f"process group {pgid} exists but cannot be enumerated"
            )
        return tuple(sorted(members, key=lambda process: process.pid))

    def _process_group_id(self, pid: int) -> int:
        """Read only the inexpensive PGID field from procfs.

        :param pid: Process identifier.
        :returns: Process-group identifier.
        """
        try:
            stat_value = (self._proc_root / str(pid) / "stat").read_bytes()
        except FileNotFoundError as error:
            raise ProcessNotFoundError(f"process {pid} is absent") from error
        except OSError as error:
            raise HostInspectionError(
                f"cannot inspect process {pid}: {error}"
            ) from error
        marker = stat_value.rfind(b") ")
        if marker < 0:
            raise HostInspectionError(f"malformed /proc/{pid}/stat")
        fields = stat_value[marker + 2 :].split()
        if len(fields) < 3:
            raise HostInspectionError(f"short /proc/{pid}/stat")
        try:
            return int(fields[2])
        except ValueError as error:
            raise HostInspectionError(f"invalid /proc/{pid}/stat") from error

    def _run(self, command: list[str]) -> str:
        """Run one inspection command without shell interpretation.

        :param command: Exact argument vector.
        :returns: Standard output.
        """
        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            raise HostInspectionError(
                f"command failed: {command!r}: {error}"
            ) from error
        return result.stdout

    def listeners(self) -> tuple[SocketRecord, ...]:
        """Read all TCP listeners.

        :returns: Parsed listener records.
        """
        return parse_ss_output(self._run(["ss", "-H", "-O", "-ltnp"]))

    def connections(self) -> tuple[SocketRecord, ...]:
        """Read all non-listening TCP sockets.

        :returns: Parsed connection records.
        """
        return parse_ss_output(self._run(["ss", "-H", "-O", "-tnp"]))

    def gpu_state(self) -> tuple[tuple[GpuDevice, ...], tuple[GpuProcess, ...]]:
        """Read physical devices and compute processes.

        :returns: Device and process inventories.
        """
        device_output = self._run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,pci.bus_id,name",
                "--format=csv,noheader,nounits",
            ]
        )
        process_output = self._run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ]
        )
        devices: list[GpuDevice] = []
        uuid_to_index: dict[str, int] = {}
        for row in csv.reader(device_output.splitlines(), skipinitialspace=True):
            if len(row) != 4:
                raise HostInspectionError(f"malformed NVIDIA device row: {row!r}")
            index = int(row[0].strip())
            uuid = row[1].strip()
            devices.append(
                GpuDevice(
                    index=index,
                    uuid=uuid,
                    pci_bus_id=row[2].strip(),
                    name=row[3].strip(),
                )
            )
            uuid_to_index[uuid] = index
        processes: list[GpuProcess] = []
        for row in csv.reader(process_output.splitlines(), skipinitialspace=True):
            if len(row) == 0:
                continue
            if len(row) != 4:
                raise HostInspectionError(f"malformed NVIDIA process row: {row!r}")
            uuid = row[0].strip()
            if uuid not in uuid_to_index:
                raise HostInspectionError(f"unknown GPU UUID in process row: {uuid}")
            processes.append(
                GpuProcess(
                    gpu_index=uuid_to_index[uuid],
                    gpu_uuid=uuid,
                    pid=int(row[1].strip()),
                    process_name=row[2].strip(),
                    used_memory_mib=int(row[3].strip()),
                )
            )
        return (
            tuple(sorted(devices, key=lambda device: device.index)),
            tuple(sorted(processes, key=lambda item: (item.gpu_index, item.pid))),
        )

    def http_get(self, url: str, timeout_seconds: float) -> tuple[int, bytes]:
        """Fetch one local HTTP endpoint.

        :param url: Endpoint URL.
        :param timeout_seconds: Request timeout.
        :returns: HTTP status and body.
        :raises HostInspectionError: If the endpoint cannot be fetched.
        """
        request = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                return response.status, response.read()
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise HostInspectionError(f"cannot fetch {url}: {error}") from error

    def tool_identity(self) -> ToolIdentity:
        """Return the clean repository and lifecycle-tool identity.

        :returns: Tool identity.
        :raises HostInspectionError: If the worktree is dirty or inconsistent.
        """
        package_directory = Path(__file__).resolve().parent
        repository_text = self._run(
            ["git", "-C", str(package_directory), "rev-parse", "--show-toplevel"]
        ).strip()
        repository = Path(repository_text).resolve()
        try:
            package_relative = package_directory.relative_to(repository)
        except ValueError as error:
            raise HostInspectionError(
                "lifecycle package is outside its reported Git repository"
            ) from error
        if package_relative.as_posix() != "tools/gemma4_pd/lane_lifecycle":
            raise HostInspectionError(
                f"unexpected lifecycle package location {package_relative}"
            )
        status = self._run(
            [
                "git",
                "-C",
                str(repository),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ]
        )
        if len(status) > 0:
            raise HostInspectionError(
                "lifecycle operations require a completely clean Git worktree"
            )
        git_head = self._run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"]
        ).strip()
        git_tree = self._run(
            ["git", "-C", str(repository), "rev-parse", "HEAD^{tree}"]
        ).strip()
        tracked_output = self._run(
            [
                "git",
                "-C",
                str(repository),
                "ls-tree",
                "-r",
                "--name-only",
                git_head,
                "--",
                package_relative.as_posix(),
            ]
        )
        tracked = sorted(line for line in tracked_output.splitlines() if len(line) > 0)
        if len(tracked) == 0:
            raise HostInspectionError("lifecycle tool has no tracked files")
        files = tuple(
            self.file_digest(os.fsencode(repository / relative)) for relative in tracked
        )
        return ToolIdentity(
            git_head=git_head,
            git_tree=git_tree,
            tool_files=files,
        )

    def copy_process_executable(self, pid: int, destination: Path) -> FileDigest:
        """Copy the executable inode currently held by one process.

        :param pid: Process identifier.
        :param destination: New snapshot file.
        :returns: Digest of the copied executable.
        """
        source = _path_from_bytes(self._proc_path(pid, "exe"))
        try:
            with source.open("rb") as source_file:
                raw_target = os.open(
                    destination,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
                    0o600,
                )
                target_file = os.fdopen(raw_target, "wb")
                try:
                    shutil.copyfileobj(source_file, target_file, length=1024 * 1024)
                    target_file.flush()
                    os.fsync(target_file.fileno())
                finally:
                    target_file.close()
        except OSError as error:
            raise HostInspectionError(
                f"cannot copy executable for process {pid}: {error}"
            ) from error
        digest, size = _sha256_file(destination)
        return FileDigest(
            path_b64=encode_bytes(os.fsencode(destination.resolve())),
            size=size,
            sha256=digest,
            symlink_target_b64=None,
        )

    def file_digest(self, path: bytes) -> FileDigest:
        """Read one file's content and path identity.

        :param path: Raw absolute path.
        :returns: File digest.
        """
        candidate = _path_from_bytes(path)
        try:
            digest, size = _sha256_file(candidate)
            symlink_target = os.readlink(path) if os.path.islink(path) else None
            absolute = os.path.abspath(path)
        except OSError as error:
            raise HostInspectionError(
                f"cannot attest file {path!r}: {error}"
            ) from error
        return FileDigest(
            path_b64=encode_bytes(absolute),
            size=size,
            sha256=digest,
            symlink_target_b64=(
                None if symlink_target is None else encode_bytes(symlink_target)
            ),
        )

    def kill_group(self, pgid: int, signal_number: int) -> None:
        """Signal one process group.

        :param pgid: Process-group identifier.
        :param signal_number: POSIX signal number.
        """
        try:
            os.killpg(pgid, signal_number)
        except OSError as error:
            raise HostInspectionError(
                f"cannot signal process group {pgid} with {signal_number}: {error}"
            ) from error

    def launch_exact(
        self,
        process: ProcessRecord,
        executable: Path,
        stdout_path: Path,
        stderr_path: Path,
    ) -> int:
        """Launch one exact process image in a new session.

        :param process: Captured process image.
        :param executable: Verified executable path.
        :param stdout_path: New stdout log.
        :param stderr_path: New stderr log.
        :returns: Child process identifier.
        :raises HostInspectionError: If the child cannot reach ``execve``.
        """
        argv = tuple(decode_bytes(item) for item in process.argv_b64)
        environment_entries = tuple(
            decode_bytes(item) for item in process.environment_b64
        )
        environment = environment_mapping(environment_entries)
        cwd = decode_bytes(process.cwd_b64)
        stdin_target = (
            None
            if process.stdin_target_b64 is None
            else decode_bytes(process.stdin_target_b64)
        )
        if stdin_target != b"/dev/null":
            raise HostInspectionError("restored service stdin must be /dev/null")
        if len(argv) == 0:
            raise HostInspectionError("cannot restore an empty argument vector")
        read_fd, write_fd = os.pipe2(os.O_CLOEXEC)
        try:
            pid = os.fork()
        except OSError as error:
            os.close(read_fd)
            os.close(write_fd)
            raise HostInspectionError(
                f"cannot fork restore process: {error}"
            ) from error
        if pid == 0:
            try:
                os.close(read_fd)
                os.setsid()
                os.chdir(cwd)
                stdin_fd = os.open(b"/dev/null", os.O_RDONLY | os.O_CLOEXEC)
                stdout_fd = os.open(
                    os.fsencode(stdout_path),
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
                stderr_fd = os.open(
                    os.fsencode(stderr_path),
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
                for limit in process.resource_limits:
                    resource_number = getattr(resource, limit.name)
                    resource.setrlimit(resource_number, (limit.soft, limit.hard))
                os.dup2(stdin_fd, 0)
                os.dup2(stdout_fd, 1)
                os.dup2(stderr_fd, 2)
                os.close(stdin_fd)
                os.close(stdout_fd)
                os.close(stderr_fd)
                os.execve(os.fsencode(executable), argv, environment)
            except BaseException:
                message = traceback.format_exc().encode(errors="backslashreplace")
                with suppress(BaseException):
                    os.write(write_fd, message)
            finally:
                os._exit(127)
        os.close(write_fd)
        try:
            ready, _, _ = select.select([read_fd], [], [], 10.0)
            if len(ready) == 0:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
                raise HostInspectionError(
                    f"restore child {pid} did not complete execve handshake"
                )
            error_text = os.read(read_fd, 65536)
        finally:
            os.close(read_fd)
        if len(error_text) > 0:
            os.waitpid(pid, 0)
            raise HostInspectionError(error_text.decode(errors="replace").strip())
        return pid

    def child_exit_status(self, pid: int) -> int | None:
        """Poll one directly launched child without blocking.

        :param pid: Child process identifier.
        :returns: Exit code when terminated, otherwise ``None``.
        :raises HostInspectionError: If the PID is not a child of this process.
        """
        try:
            waited_pid, status = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError as error:
            raise HostInspectionError(
                f"restore PID {pid} is not a child of the lifecycle process"
            ) from error
        if waited_pid == 0:
            return None
        return os.waitstatus_to_exitcode(status)

    def sleep(self, seconds: float) -> None:
        """Wait for host state to advance.

        :param seconds: Wait duration.
        """
        time.sleep(seconds)

    def monotonic(self) -> float:
        """Return a monotonic timestamp.

        :returns: Seconds on a monotonic clock.
        """
        return time.monotonic()


def discover_launcher_files(host: Host, process: LiveProcess) -> tuple[FileDigest, ...]:
    """Attest every regular file named by a process argument.

    :param host: Host inspection provider.
    :param process: Root process image.
    :returns: Sorted unique file attestations.
    """
    candidates: dict[bytes, FileDigest] = {}
    for argument in process.argv:
        if len(argument) == 0 or argument.startswith(b"-"):
            continue
        path = (
            argument if os.path.isabs(argument) else os.path.join(process.cwd, argument)
        )
        absolute = os.path.abspath(path)
        if not os.path.isfile(absolute):
            continue
        candidates[absolute] = host.file_digest(absolute)
    return tuple(candidates[path] for path in sorted(candidates))


def materialize_process(
    host: Host, process: LiveProcess, artifact_directory: Path
) -> ProcessRecord:
    """Copy and bind a process executable into the snapshot.

    :param host: Host inspection provider.
    :param process: Live process state.
    :param artifact_directory: Snapshot root.
    :returns: Stable process record.
    :raises HostInspectionError: If the process changes during capture.
    """
    relative = f"executables/{process.executable_sha256}"
    destination = artifact_directory / relative
    if not destination.exists():
        digest = host.copy_process_executable(process.pid, destination)
        if digest.sha256 != process.executable_sha256:
            raise HostInspectionError(
                f"process {process.pid} executable changed while being copied"
            )
    else:
        digest, size = _sha256_file(destination)
        if digest != process.executable_sha256 or size != process.executable_size:
            raise HostInspectionError(
                f"shared executable artifact {destination} has inconsistent content"
            )
    current = host.process(process.pid)
    if current.instance_identity() != process.instance_identity():
        raise HostInspectionError(f"process {process.pid} changed during capture")
    return record_from_live(process, relative)


def _split_nul(value: bytes) -> tuple[bytes, ...]:
    """Split one NUL-terminated procfs byte vector.

    :param value: Raw procfs data.
    :returns: Byte entries without the terminal delimiter.
    """
    parts = value.split(b"\x00")
    if len(parts) > 0 and parts[-1] == b"":
        parts.pop()
    if any(b"\x00" in part for part in parts):
        raise HostInspectionError("procfs record contains an embedded NUL")
    return tuple(parts)


def _sha256_file(path: Path) -> tuple[str, int]:
    """Hash one regular file in bounded chunks.

    :param path: File to hash.
    :returns: SHA-256 digest and byte size.
    """
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as descriptor:
            while True:
                chunk = descriptor.read(1024 * 1024)
                if len(chunk) == 0:
                    break
                digest.update(chunk)
                size += len(chunk)
    except OSError as error:
        raise HostInspectionError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest(), size


def _read_resource_limits(pid: int) -> tuple[ResourceLimit, ...]:
    """Read the complete supported resource-limit vector for one process.

    :param pid: Process identifier.
    :returns: Stable named resource limits.
    """
    limits: list[ResourceLimit] = []
    for name in _RESOURCE_LIMIT_NAMES:
        resource_number = getattr(resource, name)
        try:
            soft, hard = resource.prlimit(pid, resource_number)
        except ProcessLookupError as error:
            raise ProcessNotFoundError(f"process {pid} is absent") from error
        except OSError as error:
            raise HostInspectionError(
                f"cannot read {name} for process {pid}: {error}"
            ) from error
        limits.append(ResourceLimit(name=name, soft=soft, hard=hard))
    return tuple(limits)


def _path_from_bytes(value: bytes) -> Path:
    """Convert raw filesystem bytes through surrogateescape.

    :param value: Raw path bytes.
    :returns: Round-trippable path object.
    """
    return Path(os.fsdecode(value))


def sha256_file(path: Path) -> str:
    """Return a file's SHA-256 digest.

    :param path: File to hash.
    :returns: Lowercase SHA-256 digest.
    """
    return _sha256_file(path)[0]
