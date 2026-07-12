import base64
import binascii
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

SCHEMA_VERSION = 1
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_GIT_OID_PATTERN = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")


class SnapshotValidationError(ValueError):
    """Indicate that a lifecycle snapshot does not satisfy its schema."""


def _object(value: object, location: str) -> dict[str, object]:
    """Require a JSON object.

    :param value: Candidate value.
    :param location: Schema location.
    :returns: Validated object.
    """
    if not isinstance(value, dict):
        raise SnapshotValidationError(f"{location} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise SnapshotValidationError(f"{location} keys must be strings")
    return value


def _keys(value: dict[str, object], expected: set[str], location: str) -> None:
    """Require one exact object field set.

    :param value: Candidate object.
    :param expected: Exact required field names.
    :param location: Schema location.
    """
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise SnapshotValidationError(
            f"{location} has wrong fields: missing={missing}, extra={extra}"
        )


def _string(value: object, location: str) -> str:
    """Require a JSON string.

    :param value: Candidate value.
    :param location: Schema location.
    :returns: Validated string.
    """
    if not isinstance(value, str):
        raise SnapshotValidationError(f"{location} must be a string")
    return value


def _optional_string(value: object, location: str) -> str | None:
    """Require a JSON string or null.

    :param value: Candidate value.
    :param location: Schema location.
    :returns: Validated optional string.
    """
    if value is None:
        return None
    return _string(value, location)


def _integer(value: object, location: str) -> int:
    """Require a JSON integer distinct from Boolean values.

    :param value: Candidate value.
    :param location: Schema location.
    :returns: Validated integer.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise SnapshotValidationError(f"{location} must be an integer")
    return value


def _number(value: object, location: str) -> float:
    """Require a finite JSON number.

    :param value: Candidate value.
    :param location: Schema location.
    :returns: Validated finite number.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SnapshotValidationError(f"{location} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise SnapshotValidationError(f"{location} must be finite")
    return result


def _sha256(value: object, location: str) -> str:
    """Require a lowercase SHA-256 digest.

    :param value: Candidate value.
    :param location: Schema location.
    :returns: Validated digest.
    """
    result = _string(value, location)
    if _SHA256_PATTERN.fullmatch(result) is None:
        raise SnapshotValidationError(f"{location} must be a lowercase SHA-256 digest")
    return result


def _base64(value: object, location: str, *, absolute_path: bool = False) -> str:
    """Require canonical Base64 data without NUL bytes.

    :param value: Candidate value.
    :param location: Schema location.
    :param absolute_path: Require decoded data to name an absolute path.
    :returns: Validated Base64 text.
    """
    result = _string(value, location)
    try:
        decoded = base64.b64decode(result, validate=True)
    except (ValueError, binascii.Error) as error:
        raise SnapshotValidationError(f"{location} must be valid Base64") from error
    if b"\x00" in decoded:
        raise SnapshotValidationError(f"{location} cannot contain NUL")
    if absolute_path and not decoded.startswith(b"/"):
        raise SnapshotValidationError(f"{location} must encode an absolute path")
    return result


def _relative_artifact(value: object, location: str) -> str:
    """Require a confined snapshot-relative artifact path.

    :param value: Candidate value.
    :param location: Schema location.
    :returns: Validated relative path.
    """
    result = _string(value, location)
    path = PurePosixPath(result)
    if path.is_absolute() or len(path.parts) == 0:
        raise SnapshotValidationError(f"{location} must be a relative artifact path")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise SnapshotValidationError(f"{location} escapes the snapshot directory")
    return result


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Build a JSON object while rejecting duplicate keys.

    :param pairs: Parsed object pairs in source order.
    :returns: Unique-key object.
    """
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise SnapshotValidationError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    """Reject non-finite JSON constants.

    :param value: Parsed constant spelling.
    :raises SnapshotValidationError: Unconditionally.
    """
    raise SnapshotValidationError(f"non-finite JSON constant {value!r} is forbidden")


def _list(value: object, location: str) -> list[object]:
    """Require a JSON array.

    :param value: Candidate value.
    :param location: Schema location.
    :returns: Validated array.
    """
    if not isinstance(value, list):
        raise SnapshotValidationError(f"{location} must be an array")
    return value


def _strings(value: object, location: str) -> tuple[str, ...]:
    """Require an array of strings.

    :param value: Candidate value.
    :param location: Schema location.
    :returns: Validated strings.
    """
    return tuple(
        _string(item, f"{location}[{index}]")
        for index, item in enumerate(_list(value, location))
    )


def _integers(value: object, location: str) -> tuple[int, ...]:
    """Require an array of integers.

    :param value: Candidate value.
    :param location: Schema location.
    :returns: Validated integers.
    """
    return tuple(
        _integer(item, f"{location}[{index}]")
        for index, item in enumerate(_list(value, location))
    )


@dataclass(frozen=True, slots=True)
class FileDigest:
    """Describe immutable file content and its path identity.

    :ivar path_b64: Base64-encoded raw absolute path.
    :ivar size: File size in bytes.
    :ivar sha256: Lowercase SHA-256 digest.
    :ivar symlink_target_b64: Base64-encoded raw symlink target, when applicable.
    """

    path_b64: str
    size: int
    sha256: str
    symlink_target_b64: str | None

    def to_dict(self) -> dict[str, object]:
        """Serialize the file digest.

        :returns: JSON-compatible object.
        """
        return {
            "path_b64": self.path_b64,
            "size": self.size,
            "sha256": self.sha256,
            "symlink_target_b64": self.symlink_target_b64,
        }

    @classmethod
    def from_dict(cls, value: object, location: str) -> "FileDigest":
        """Parse a file digest.

        :param value: Candidate JSON value.
        :param location: Human-readable schema location.
        :returns: Validated file digest.
        """
        data = _object(value, location)
        _keys(data, {"path_b64", "size", "sha256", "symlink_target_b64"}, location)
        return cls(
            path_b64=_base64(
                data["path_b64"], f"{location}.path_b64", absolute_path=True
            ),
            size=_integer(data["size"], f"{location}.size"),
            sha256=_sha256(data["sha256"], f"{location}.sha256"),
            symlink_target_b64=(
                None
                if data["symlink_target_b64"] is None
                else _base64(
                    data["symlink_target_b64"], f"{location}.symlink_target_b64"
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class ResourceLimit:
    """Describe one exact POSIX resource limit.

    :ivar name: Stable ``resource.RLIMIT_*`` name.
    :ivar soft: Soft limit, with ``-1`` representing infinity.
    :ivar hard: Hard limit, with ``-1`` representing infinity.
    """

    name: str
    soft: int
    hard: int

    def to_dict(self) -> dict[str, object]:
        """Serialize the resource limit.

        :returns: JSON-compatible object.
        """
        return {"name": self.name, "soft": self.soft, "hard": self.hard}

    @classmethod
    def from_dict(cls, value: object, location: str) -> "ResourceLimit":
        """Parse a resource limit.

        :param value: Candidate JSON value.
        :param location: Human-readable schema location.
        :returns: Validated resource limit.
        """
        data = _object(value, location)
        _keys(data, {"name", "soft", "hard"}, location)
        name = _string(data["name"], f"{location}.name")
        if re.fullmatch(r"RLIMIT_[A-Z]+", name) is None:
            raise SnapshotValidationError(f"{location}.name is not an RLIMIT name")
        soft = _integer(data["soft"], f"{location}.soft")
        hard = _integer(data["hard"], f"{location}.hard")
        if soft < -1 or hard < -1:
            raise SnapshotValidationError(f"{location} has an invalid negative limit")
        if hard != -1 and (soft == -1 or soft > hard):
            raise SnapshotValidationError(f"{location} soft limit exceeds hard limit")
        return cls(name=name, soft=soft, hard=hard)


@dataclass(frozen=True, slots=True)
class ProcessRecord:
    """Describe one exact Linux process instance and launch image.

    :ivar pid: Process identifier at capture time.
    :ivar ppid: Parent process identifier at capture time.
    :ivar pgid: Process-group identifier at capture time.
    :ivar sid: Session identifier at capture time.
    :ivar start_time_ticks: Kernel start time from ``/proc/PID/stat``.
    :ivar state: Kernel process state.
    :ivar argv_b64: Base64-encoded raw argument vector entries.
    :ivar environment_b64: Base64-encoded raw ``KEY=VALUE`` environment entries.
    :ivar cwd_b64: Base64-encoded raw working directory.
    :ivar executable_link_b64: Base64-encoded raw ``/proc/PID/exe`` target.
    :ivar executable_sha256: Digest of the executable inode in use.
    :ivar executable_size: Executable size in bytes.
    :ivar executable_artifact: Snapshot-relative executable copy.
    :ivar resource_limits: Exact POSIX root resource-limit vector.
    :ivar stdin_target_b64: Base64-encoded fd 0 target, when readable.
    :ivar stdout_target_b64: Base64-encoded fd 1 target, when readable.
    :ivar stderr_target_b64: Base64-encoded fd 2 target, when readable.
    """

    pid: int
    ppid: int
    pgid: int
    sid: int
    start_time_ticks: int
    state: str
    argv_b64: tuple[str, ...]
    environment_b64: tuple[str, ...]
    cwd_b64: str
    executable_link_b64: str
    executable_sha256: str
    executable_size: int
    executable_artifact: str
    resource_limits: tuple[ResourceLimit, ...]
    stdin_target_b64: str | None
    stdout_target_b64: str | None
    stderr_target_b64: str | None

    def to_dict(self) -> dict[str, object]:
        """Serialize the process record.

        :returns: JSON-compatible object.
        """
        return {
            "pid": self.pid,
            "ppid": self.ppid,
            "pgid": self.pgid,
            "sid": self.sid,
            "start_time_ticks": self.start_time_ticks,
            "state": self.state,
            "argv_b64": list(self.argv_b64),
            "environment_b64": list(self.environment_b64),
            "cwd_b64": self.cwd_b64,
            "executable_link_b64": self.executable_link_b64,
            "executable_sha256": self.executable_sha256,
            "executable_size": self.executable_size,
            "executable_artifact": self.executable_artifact,
            "resource_limits": [item.to_dict() for item in self.resource_limits],
            "stdin_target_b64": self.stdin_target_b64,
            "stdout_target_b64": self.stdout_target_b64,
            "stderr_target_b64": self.stderr_target_b64,
        }

    @classmethod
    def from_dict(cls, value: object, location: str) -> "ProcessRecord":
        """Parse a process record.

        :param value: Candidate JSON value.
        :param location: Human-readable schema location.
        :returns: Validated process record.
        """
        data = _object(value, location)
        expected = {
            "pid",
            "ppid",
            "pgid",
            "sid",
            "start_time_ticks",
            "state",
            "argv_b64",
            "environment_b64",
            "cwd_b64",
            "executable_link_b64",
            "executable_sha256",
            "executable_size",
            "executable_artifact",
            "resource_limits",
            "stdin_target_b64",
            "stdout_target_b64",
            "stderr_target_b64",
        }
        _keys(data, expected, location)
        return cls(
            pid=_integer(data["pid"], f"{location}.pid"),
            ppid=_integer(data["ppid"], f"{location}.ppid"),
            pgid=_integer(data["pgid"], f"{location}.pgid"),
            sid=_integer(data["sid"], f"{location}.sid"),
            start_time_ticks=_integer(
                data["start_time_ticks"], f"{location}.start_time_ticks"
            ),
            state=_string(data["state"], f"{location}.state"),
            argv_b64=tuple(
                _base64(item, f"{location}.argv_b64[{index}]")
                for index, item in enumerate(
                    _list(data["argv_b64"], f"{location}.argv_b64")
                )
            ),
            environment_b64=tuple(
                _base64(item, f"{location}.environment_b64[{index}]")
                for index, item in enumerate(
                    _list(data["environment_b64"], f"{location}.environment_b64")
                )
            ),
            cwd_b64=_base64(data["cwd_b64"], f"{location}.cwd_b64", absolute_path=True),
            executable_link_b64=_base64(
                data["executable_link_b64"], f"{location}.executable_link_b64"
            ),
            executable_sha256=_sha256(
                data["executable_sha256"], f"{location}.executable_sha256"
            ),
            executable_size=_integer(
                data["executable_size"], f"{location}.executable_size"
            ),
            executable_artifact=_relative_artifact(
                data["executable_artifact"], f"{location}.executable_artifact"
            ),
            resource_limits=tuple(
                ResourceLimit.from_dict(item, f"{location}.resource_limits[{index}]")
                for index, item in enumerate(
                    _list(data["resource_limits"], f"{location}.resource_limits")
                )
            ),
            stdin_target_b64=(
                None
                if data["stdin_target_b64"] is None
                else _base64(data["stdin_target_b64"], f"{location}.stdin_target_b64")
            ),
            stdout_target_b64=(
                None
                if data["stdout_target_b64"] is None
                else _base64(data["stdout_target_b64"], f"{location}.stdout_target_b64")
            ),
            stderr_target_b64=(
                None
                if data["stderr_target_b64"] is None
                else _base64(data["stderr_target_b64"], f"{location}.stderr_target_b64")
            ),
        )


@dataclass(frozen=True, slots=True)
class SocketRecord:
    """Describe a TCP socket reported by ``ss``.

    :ivar state: TCP state.
    :ivar receive_queue: Receive-queue byte count.
    :ivar send_queue: Send-queue byte count.
    :ivar local: Raw local endpoint text.
    :ivar peer: Raw peer endpoint text.
    :ivar process_ids: Owning process identifiers visible to ``ss``.
    :ivar raw: Complete one-line ``ss`` record.
    """

    state: str
    receive_queue: int
    send_queue: int
    local: str
    peer: str
    process_ids: tuple[int, ...]
    raw: str

    def to_dict(self) -> dict[str, object]:
        """Serialize the socket record.

        :returns: JSON-compatible object.
        """
        return {
            "state": self.state,
            "receive_queue": self.receive_queue,
            "send_queue": self.send_queue,
            "local": self.local,
            "peer": self.peer,
            "process_ids": list(self.process_ids),
            "raw": self.raw,
        }

    @classmethod
    def from_dict(cls, value: object, location: str) -> "SocketRecord":
        """Parse a socket record.

        :param value: Candidate JSON value.
        :param location: Human-readable schema location.
        :returns: Validated socket record.
        """
        data = _object(value, location)
        _keys(
            data,
            {
                "state",
                "receive_queue",
                "send_queue",
                "local",
                "peer",
                "process_ids",
                "raw",
            },
            location,
        )
        return cls(
            state=_string(data["state"], f"{location}.state"),
            receive_queue=_integer(data["receive_queue"], f"{location}.receive_queue"),
            send_queue=_integer(data["send_queue"], f"{location}.send_queue"),
            local=_string(data["local"], f"{location}.local"),
            peer=_string(data["peer"], f"{location}.peer"),
            process_ids=_integers(data["process_ids"], f"{location}.process_ids"),
            raw=_string(data["raw"], f"{location}.raw"),
        )


@dataclass(frozen=True, slots=True)
class GpuDevice:
    """Describe one physical GPU.

    :ivar index: Stable index used by the experiment.
    :ivar uuid: NVIDIA GPU UUID.
    :ivar pci_bus_id: PCI bus identifier.
    :ivar name: Device model.
    """

    index: int
    uuid: str
    pci_bus_id: str
    name: str

    def to_dict(self) -> dict[str, object]:
        """Serialize the GPU device.

        :returns: JSON-compatible object.
        """
        return {
            "index": self.index,
            "uuid": self.uuid,
            "pci_bus_id": self.pci_bus_id,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, value: object, location: str) -> "GpuDevice":
        """Parse a GPU device.

        :param value: Candidate JSON value.
        :param location: Human-readable schema location.
        :returns: Validated GPU device.
        """
        data = _object(value, location)
        _keys(data, {"index", "uuid", "pci_bus_id", "name"}, location)
        return cls(
            index=_integer(data["index"], f"{location}.index"),
            uuid=_string(data["uuid"], f"{location}.uuid"),
            pci_bus_id=_string(data["pci_bus_id"], f"{location}.pci_bus_id"),
            name=_string(data["name"], f"{location}.name"),
        )


@dataclass(frozen=True, slots=True)
class GpuProcess:
    """Describe one NVIDIA compute process.

    :ivar gpu_index: Physical GPU index.
    :ivar gpu_uuid: Physical GPU UUID.
    :ivar pid: Compute-process identifier.
    :ivar process_name: NVIDIA-reported process name.
    :ivar used_memory_mib: Reported GPU memory allocation in MiB.
    """

    gpu_index: int
    gpu_uuid: str
    pid: int
    process_name: str
    used_memory_mib: int

    def to_dict(self) -> dict[str, object]:
        """Serialize the GPU process.

        :returns: JSON-compatible object.
        """
        return {
            "gpu_index": self.gpu_index,
            "gpu_uuid": self.gpu_uuid,
            "pid": self.pid,
            "process_name": self.process_name,
            "used_memory_mib": self.used_memory_mib,
        }

    @classmethod
    def from_dict(cls, value: object, location: str) -> "GpuProcess":
        """Parse a GPU process.

        :param value: Candidate JSON value.
        :param location: Human-readable schema location.
        :returns: Validated GPU process.
        """
        data = _object(value, location)
        _keys(
            data,
            {"gpu_index", "gpu_uuid", "pid", "process_name", "used_memory_mib"},
            location,
        )
        return cls(
            gpu_index=_integer(data["gpu_index"], f"{location}.gpu_index"),
            gpu_uuid=_string(data["gpu_uuid"], f"{location}.gpu_uuid"),
            pid=_integer(data["pid"], f"{location}.pid"),
            process_name=_string(data["process_name"], f"{location}.process_name"),
            used_memory_mib=_integer(
                data["used_memory_mib"], f"{location}.used_memory_mib"
            ),
        )


@dataclass(frozen=True, slots=True)
class MetricRecord:
    """Describe the quiescence metrics for one model server.

    :ivar role: Canonical service role.
    :ivar url: Metrics endpoint.
    :ivar running: Sum of all running-request series.
    :ivar waiting: Sum of all waiting-request series.
    :ivar body_sha256: Digest of the captured Prometheus body.
    :ivar body_artifact: Snapshot-relative Prometheus body.
    """

    role: str
    url: str
    running: float
    waiting: float
    body_sha256: str
    body_artifact: str

    def to_dict(self) -> dict[str, object]:
        """Serialize the metric record.

        :returns: JSON-compatible object.
        """
        return {
            "role": self.role,
            "url": self.url,
            "running": self.running,
            "waiting": self.waiting,
            "body_sha256": self.body_sha256,
            "body_artifact": self.body_artifact,
        }

    @classmethod
    def from_dict(cls, value: object, location: str) -> "MetricRecord":
        """Parse a metric record.

        :param value: Candidate JSON value.
        :param location: Human-readable schema location.
        :returns: Validated metric record.
        """
        data = _object(value, location)
        _keys(
            data,
            {"role", "url", "running", "waiting", "body_sha256", "body_artifact"},
            location,
        )
        return cls(
            role=_string(data["role"], f"{location}.role"),
            url=_string(data["url"], f"{location}.url"),
            running=_number(data["running"], f"{location}.running"),
            waiting=_number(data["waiting"], f"{location}.waiting"),
            body_sha256=_sha256(data["body_sha256"], f"{location}.body_sha256"),
            body_artifact=_relative_artifact(
                data["body_artifact"], f"{location}.body_artifact"
            ),
        )


@dataclass(frozen=True, slots=True)
class ServiceRecord:
    """Describe one experiment-lane service and its ownership group.

    :ivar role: Canonical service role.
    :ivar port: Exact TCP listener port.
    :ivar health_path: HTTP health path, or ``None`` for listener-only readiness.
    :ivar root: Listener process used for exact restoration.
    :ivar group_members: Complete captured process-group membership.
    :ivar gpu_indices: GPUs owned by this process group.
    :ivar ucx_environment_b64: Raw UCX and NIXL environment entries.
    :ivar launcher_files: Files named by the root argument vector.
    """

    role: str
    port: int
    health_path: str | None
    root: ProcessRecord
    group_members: tuple[ProcessRecord, ...]
    gpu_indices: tuple[int, ...]
    ucx_environment_b64: tuple[str, ...]
    launcher_files: tuple[FileDigest, ...]

    def to_dict(self) -> dict[str, object]:
        """Serialize the service record.

        :returns: JSON-compatible object.
        """
        return {
            "role": self.role,
            "port": self.port,
            "health_path": self.health_path,
            "root": self.root.to_dict(),
            "group_members": [member.to_dict() for member in self.group_members],
            "gpu_indices": list(self.gpu_indices),
            "ucx_environment_b64": list(self.ucx_environment_b64),
            "launcher_files": [item.to_dict() for item in self.launcher_files],
        }

    @classmethod
    def from_dict(cls, value: object, location: str) -> "ServiceRecord":
        """Parse a service record.

        :param value: Candidate JSON value.
        :param location: Human-readable schema location.
        :returns: Validated service record.
        """
        data = _object(value, location)
        _keys(
            data,
            {
                "role",
                "port",
                "health_path",
                "root",
                "group_members",
                "gpu_indices",
                "ucx_environment_b64",
                "launcher_files",
            },
            location,
        )
        members = tuple(
            ProcessRecord.from_dict(item, f"{location}.group_members[{index}]")
            for index, item in enumerate(
                _list(data["group_members"], f"{location}.group_members")
            )
        )
        launchers = tuple(
            FileDigest.from_dict(item, f"{location}.launcher_files[{index}]")
            for index, item in enumerate(
                _list(data["launcher_files"], f"{location}.launcher_files")
            )
        )
        return cls(
            role=_string(data["role"], f"{location}.role"),
            port=_integer(data["port"], f"{location}.port"),
            health_path=_optional_string(
                data["health_path"], f"{location}.health_path"
            ),
            root=ProcessRecord.from_dict(data["root"], f"{location}.root"),
            group_members=members,
            gpu_indices=_integers(data["gpu_indices"], f"{location}.gpu_indices"),
            ucx_environment_b64=_strings(
                data["ucx_environment_b64"],
                f"{location}.ucx_environment_b64",
            ),
            launcher_files=launchers,
        )


@dataclass(frozen=True, slots=True)
class ProtectedGroup:
    """Describe a process group that owns production GPUs 6 or 7.

    :ivar pgid: Protected process-group identifier.
    :ivar gpu_indices: Protected GPUs used by the group.
    :ivar group_members: Complete captured membership and launch identity.
    :ivar listener_ports: Listener ports owned by members of the group.
    """

    pgid: int
    gpu_indices: tuple[int, ...]
    group_members: tuple[ProcessRecord, ...]
    listener_ports: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        """Serialize the protected group.

        :returns: JSON-compatible object.
        """
        return {
            "pgid": self.pgid,
            "gpu_indices": list(self.gpu_indices),
            "group_members": [member.to_dict() for member in self.group_members],
            "listener_ports": list(self.listener_ports),
        }

    @classmethod
    def from_dict(cls, value: object, location: str) -> "ProtectedGroup":
        """Parse a protected group.

        :param value: Candidate JSON value.
        :param location: Human-readable schema location.
        :returns: Validated protected group.
        """
        data = _object(value, location)
        _keys(
            data,
            {"pgid", "gpu_indices", "group_members", "listener_ports"},
            location,
        )
        members = tuple(
            ProcessRecord.from_dict(item, f"{location}.group_members[{index}]")
            for index, item in enumerate(
                _list(data["group_members"], f"{location}.group_members")
            )
        )
        return cls(
            pgid=_integer(data["pgid"], f"{location}.pgid"),
            gpu_indices=_integers(data["gpu_indices"], f"{location}.gpu_indices"),
            group_members=members,
            listener_ports=_integers(
                data["listener_ports"], f"{location}.listener_ports"
            ),
        )


@dataclass(frozen=True, slots=True)
class ToolIdentity:
    """Bind lifecycle mutations to one clean repository and tool image.

    :ivar git_head: Clean Git commit identifier.
    :ivar git_tree: Tree identifier committed by ``git_head``.
    :ivar tool_files: Content identities for every tracked lifecycle tool file.
    """

    git_head: str
    git_tree: str
    tool_files: tuple[FileDigest, ...]

    def to_dict(self) -> dict[str, object]:
        """Serialize the tool identity.

        :returns: JSON-compatible object.
        """
        return {
            "git_head": self.git_head,
            "git_tree": self.git_tree,
            "tool_files": [item.to_dict() for item in self.tool_files],
        }

    @classmethod
    def from_dict(cls, value: object, location: str) -> "ToolIdentity":
        """Parse a tool identity.

        :param value: Candidate JSON value.
        :param location: Human-readable schema location.
        :returns: Validated tool identity.
        """
        data = _object(value, location)
        _keys(data, {"git_head", "git_tree", "tool_files"}, location)
        git_head = _string(data["git_head"], f"{location}.git_head")
        git_tree = _string(data["git_tree"], f"{location}.git_tree")
        if _GIT_OID_PATTERN.fullmatch(git_head) is None:
            raise SnapshotValidationError(f"{location}.git_head is not a Git OID")
        if _GIT_OID_PATTERN.fullmatch(git_tree) is None:
            raise SnapshotValidationError(f"{location}.git_tree is not a Git OID")
        tool_files = tuple(
            FileDigest.from_dict(item, f"{location}.tool_files[{index}]")
            for index, item in enumerate(
                _list(data["tool_files"], f"{location}.tool_files")
            )
        )
        if len(tool_files) == 0:
            raise SnapshotValidationError("tool identity cannot have an empty file set")
        if len({item.path_b64 for item in tool_files}) != len(tool_files):
            raise SnapshotValidationError("tool identity contains duplicate paths")
        return cls(git_head=git_head, git_tree=git_tree, tool_files=tool_files)


@dataclass(frozen=True, slots=True)
class LaneSnapshot:
    """Describe a restorable experiment lane and protected production identity.

    :ivar schema_version: Exact lifecycle schema version.
    :ivar captured_at_utc: ISO-8601 capture time.
    :ivar hostname: Captured hostname.
    :ivar boot_id: Linux boot identifier binding all start times.
    :ivar kernel_release: Captured kernel release.
    :ivar tool_identity: Clean Git and lifecycle-tool content identity.
    :ivar storm_archive: Sealed storm37b archive identity.
    :ivar services: Five exact experiment-lane services.
    :ivar protected_groups: Process groups owning production GPUs 6 and 7.
    :ivar gpu_devices: Complete physical GPU inventory.
    :ivar gpu_processes: Complete compute-process inventory.
    :ivar listeners: Complete captured TCP listener inventory.
    :ivar connections: Complete captured TCP connection inventory.
    :ivar metrics: Zero running/waiting evidence for P, D1, and D2.
    """

    schema_version: int
    captured_at_utc: str
    hostname: str
    boot_id: str
    kernel_release: str
    tool_identity: ToolIdentity
    storm_archive: FileDigest
    services: tuple[ServiceRecord, ...]
    protected_groups: tuple[ProtectedGroup, ...]
    gpu_devices: tuple[GpuDevice, ...]
    gpu_processes: tuple[GpuProcess, ...]
    listeners: tuple[SocketRecord, ...]
    connections: tuple[SocketRecord, ...]
    metrics: tuple[MetricRecord, ...]

    def service(self, role: str) -> ServiceRecord:
        """Return one canonical service.

        :param role: Canonical role name.
        :returns: Matching service record.
        :raises SnapshotValidationError: If the role is absent or duplicated.
        """
        matches = [service for service in self.services if service.role == role]
        if len(matches) != 1:
            raise SnapshotValidationError(
                f"snapshot must contain exactly one {role!r} service"
            )
        return matches[0]

    def to_dict(self) -> dict[str, object]:
        """Serialize the snapshot.

        :returns: JSON-compatible object.
        """
        return {
            "schema_version": self.schema_version,
            "captured_at_utc": self.captured_at_utc,
            "hostname": self.hostname,
            "boot_id": self.boot_id,
            "kernel_release": self.kernel_release,
            "tool_identity": self.tool_identity.to_dict(),
            "storm_archive": self.storm_archive.to_dict(),
            "services": [service.to_dict() for service in self.services],
            "protected_groups": [group.to_dict() for group in self.protected_groups],
            "gpu_devices": [device.to_dict() for device in self.gpu_devices],
            "gpu_processes": [process.to_dict() for process in self.gpu_processes],
            "listeners": [socket.to_dict() for socket in self.listeners],
            "connections": [socket.to_dict() for socket in self.connections],
            "metrics": [metric.to_dict() for metric in self.metrics],
        }

    @classmethod
    def from_dict(cls, value: object) -> "LaneSnapshot":
        """Parse and validate a lifecycle snapshot.

        :param value: Candidate JSON value.
        :returns: Validated snapshot.
        """
        data = _object(value, "snapshot")
        expected = {
            "schema_version",
            "captured_at_utc",
            "hostname",
            "boot_id",
            "kernel_release",
            "tool_identity",
            "storm_archive",
            "services",
            "protected_groups",
            "gpu_devices",
            "gpu_processes",
            "listeners",
            "connections",
            "metrics",
        }
        _keys(data, expected, "snapshot")
        schema_version = _integer(data["schema_version"], "snapshot.schema_version")
        if schema_version != SCHEMA_VERSION:
            raise SnapshotValidationError(
                f"unsupported schema version {schema_version}; "
                f"expected {SCHEMA_VERSION}"
            )
        snapshot = cls(
            schema_version=schema_version,
            captured_at_utc=_string(
                data["captured_at_utc"], "snapshot.captured_at_utc"
            ),
            hostname=_string(data["hostname"], "snapshot.hostname"),
            boot_id=_string(data["boot_id"], "snapshot.boot_id"),
            kernel_release=_string(data["kernel_release"], "snapshot.kernel_release"),
            tool_identity=ToolIdentity.from_dict(
                data["tool_identity"], "snapshot.tool_identity"
            ),
            storm_archive=FileDigest.from_dict(
                data["storm_archive"], "snapshot.storm_archive"
            ),
            services=tuple(
                ServiceRecord.from_dict(item, f"snapshot.services[{index}]")
                for index, item in enumerate(
                    _list(data["services"], "snapshot.services")
                )
            ),
            protected_groups=tuple(
                ProtectedGroup.from_dict(item, f"snapshot.protected_groups[{index}]")
                for index, item in enumerate(
                    _list(data["protected_groups"], "snapshot.protected_groups")
                )
            ),
            gpu_devices=tuple(
                GpuDevice.from_dict(item, f"snapshot.gpu_devices[{index}]")
                for index, item in enumerate(
                    _list(data["gpu_devices"], "snapshot.gpu_devices")
                )
            ),
            gpu_processes=tuple(
                GpuProcess.from_dict(item, f"snapshot.gpu_processes[{index}]")
                for index, item in enumerate(
                    _list(data["gpu_processes"], "snapshot.gpu_processes")
                )
            ),
            listeners=tuple(
                SocketRecord.from_dict(item, f"snapshot.listeners[{index}]")
                for index, item in enumerate(
                    _list(data["listeners"], "snapshot.listeners")
                )
            ),
            connections=tuple(
                SocketRecord.from_dict(item, f"snapshot.connections[{index}]")
                for index, item in enumerate(
                    _list(data["connections"], "snapshot.connections")
                )
            ),
            metrics=tuple(
                MetricRecord.from_dict(item, f"snapshot.metrics[{index}]")
                for index, item in enumerate(_list(data["metrics"], "snapshot.metrics"))
            ),
        )
        snapshot.validate_contract()
        return snapshot

    @classmethod
    def from_json_bytes(cls, value: bytes) -> "LaneSnapshot":
        """Parse a snapshot from UTF-8 JSON.

        :param value: Serialized snapshot.
        :returns: Validated snapshot.
        :raises SnapshotValidationError: If JSON or schema validation fails.
        """
        try:
            decoded = json.loads(
                value,
                object_pairs_hook=_strict_object,
                parse_constant=_reject_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise SnapshotValidationError(f"invalid snapshot JSON: {error}") from error
        return cls.from_dict(decoded)

    def to_json_bytes(self) -> bytes:
        """Serialize a deterministic snapshot document.

        :returns: UTF-8 JSON ending with a newline.
        """
        return (
            json.dumps(
                self.to_dict(),
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode()

    def validate_contract(self) -> None:
        """Validate cross-record lifecycle invariants.

        :raises SnapshotValidationError: If any invariant is violated.
        """
        expected_roles = {"p", "d1", "d2", "router", "proxy"}
        roles = [service.role for service in self.services]
        if set(roles) != expected_roles or len(roles) != len(expected_roles):
            raise SnapshotValidationError(
                f"service roles must be exactly {sorted(expected_roles)}"
            )
        expected_ports = {
            "p": 8810,
            "d1": 8811,
            "proxy": 8812,
            "router": 8813,
            "d2": 8815,
        }
        expected_gpus = {
            "p": (0, 1, 2, 3),
            "d1": (4,),
            "d2": (5,),
            "router": (),
            "proxy": (),
        }
        for service in self.services:
            if service.port != expected_ports[service.role]:
                raise SnapshotValidationError(
                    f"{service.role} must own port {expected_ports[service.role]}"
                )
            if tuple(sorted(service.gpu_indices)) != expected_gpus[service.role]:
                raise SnapshotValidationError(
                    f"{service.role} GPU ownership must be "
                    f"{expected_gpus[service.role]}"
                )
            if (
                service.root.pid != service.root.pgid
                or service.root.pid != service.root.sid
            ):
                raise SnapshotValidationError(
                    f"{service.role} root must be its process-group and session leader"
                )
            if service.root.stdin_target_b64 != base64.b64encode(b"/dev/null").decode():
                raise SnapshotValidationError(
                    f"{service.role} stdin must be bound to /dev/null"
                )
            member_pids = {member.pid for member in service.group_members}
            if service.root.pid not in member_pids:
                raise SnapshotValidationError(
                    f"{service.role} root is absent from its process group"
                )
            if any(
                member.pgid != service.root.pgid for member in service.group_members
            ):
                raise SnapshotValidationError(
                    f"{service.role} contains a foreign process-group member"
                )
            for process in service.group_members:
                _validate_process_encoding(process, f"service {service.role}")
        protected_gpus = sorted(
            gpu for group in self.protected_groups for gpu in group.gpu_indices
        )
        if protected_gpus != [6, 7]:
            raise SnapshotValidationError(
                "protected process groups must own exactly GPUs 6 and 7"
            )
        protected_listener_ports = {
            port for group in self.protected_groups for port in group.listener_ports
        }
        if not {8000, 8820, 8821}.issubset(protected_listener_ports):
            raise SnapshotValidationError(
                "protected groups must include production listeners 8000, "
                "8820, and 8821"
            )
        protected_pgids = {group.pgid for group in self.protected_groups}
        lane_pgids = {service.root.pgid for service in self.services}
        if len(protected_pgids & lane_pgids) > 0:
            raise SnapshotValidationError(
                "experiment and protected production process groups overlap"
            )
        for group in self.protected_groups:
            if len(group.group_members) == 0:
                raise SnapshotValidationError("protected groups cannot be empty")
            if any(member.pgid != group.pgid for member in group.group_members):
                raise SnapshotValidationError(
                    f"protected group {group.pgid} contains a foreign member"
                )
            for process in group.group_members:
                _validate_process_encoding(process, f"protected group {group.pgid}")
        device_indices = sorted(device.index for device in self.gpu_devices)
        if device_indices != list(range(8)):
            raise SnapshotValidationError(
                "GPU inventory must contain exact indices 0-7"
            )
        devices_by_uuid = {device.uuid: device for device in self.gpu_devices}
        if len(devices_by_uuid) != len(self.gpu_devices):
            raise SnapshotValidationError("GPU inventory contains duplicate UUIDs")
        service_members: dict[int, str] = {}
        for service in self.services:
            for member in service.group_members:
                if member.pid in service_members:
                    raise SnapshotValidationError(
                        f"process {member.pid} appears in multiple service groups"
                    )
                service_members[member.pid] = service.role
        protected_members: dict[int, ProtectedGroup] = {}
        for group in self.protected_groups:
            for member in group.group_members:
                if member.pid in protected_members or member.pid in service_members:
                    raise SnapshotValidationError(
                        f"process {member.pid} appears in overlapping ownership groups"
                    )
                protected_members[member.pid] = group
        expected_role_by_gpu = {0: "p", 1: "p", 2: "p", 3: "p", 4: "d1", 5: "d2"}
        observed_gpu_indices: set[int] = set()
        for process in self.gpu_processes:
            device = devices_by_uuid.get(process.gpu_uuid)
            if device is None or device.index != process.gpu_index:
                raise SnapshotValidationError(
                    f"GPU process {process.pid} has an inconsistent UUID/index pair"
                )
            observed_gpu_indices.add(process.gpu_index)
            if process.gpu_index <= 5:
                expected_role = expected_role_by_gpu[process.gpu_index]
                if service_members.get(process.pid) != expected_role:
                    raise SnapshotValidationError(
                        f"GPU {process.gpu_index} process {process.pid} is not "
                        f"owned by {expected_role}"
                    )
                continue
            protected_group = protected_members.get(process.pid)
            if (
                protected_group is None
                or process.gpu_index not in protected_group.gpu_indices
            ):
                raise SnapshotValidationError(
                    f"GPU {process.gpu_index} process {process.pid} is not protected"
                )
        if observed_gpu_indices != set(range(8)):
            raise SnapshotValidationError(
                "every GPU index 0-7 must have a joined compute-process owner"
            )
        metric_roles = sorted(metric.role for metric in self.metrics)
        if metric_roles != ["d1", "d2", "p"]:
            raise SnapshotValidationError("metrics must cover exactly P, D1, and D2")
        if any(
            metric.running != 0.0 or metric.waiting != 0.0 for metric in self.metrics
        ):
            raise SnapshotValidationError(
                "captured running and waiting request metrics must both be zero"
            )
        if self.storm_archive.sha256 != (
            "e4f4e7ff7bd1bcee0de3c8413d2915cf5f3b9b0caafc7b65fe4053e5069d732a"
        ):
            raise SnapshotValidationError(
                "storm37b archive digest is not authoritative"
            )


def _validate_process_encoding(process: ProcessRecord, location: str) -> None:
    """Validate raw byte fields and resource-limit uniqueness.

    :param process: Process record.
    :param location: Human-readable ownership location.
    """
    if len(process.argv_b64) == 0:
        raise SnapshotValidationError(
            f"{location} process {process.pid} has empty argv"
        )
    argv = [base64.b64decode(item, validate=True) for item in process.argv_b64]
    if any(len(item) == 0 or b"\x00" in item for item in argv):
        raise SnapshotValidationError(
            f"{location} process {process.pid} has invalid argv"
        )
    environment_keys: set[bytes] = set()
    for item in process.environment_b64:
        entry = base64.b64decode(item, validate=True)
        key, separator, value = entry.partition(b"=")
        if separator != b"=" or len(key) == 0 or b"\x00" in key or b"\x00" in value:
            raise SnapshotValidationError(
                f"{location} process {process.pid} has invalid environment"
            )
        if key in environment_keys:
            raise SnapshotValidationError(
                f"{location} process {process.pid} has duplicate environment keys"
            )
        environment_keys.add(key)
    _base64(process.cwd_b64, f"{location}.cwd_b64", absolute_path=True)
    _base64(process.executable_link_b64, f"{location}.executable_link_b64")
    _sha256(process.executable_sha256, f"{location}.executable_sha256")
    _relative_artifact(process.executable_artifact, f"{location}.executable_artifact")
    limit_names = [item.name for item in process.resource_limits]
    if len(limit_names) != len(set(limit_names)):
        raise SnapshotValidationError(
            f"{location} process {process.pid} has duplicate resource limits"
        )
    if "RLIMIT_NOFILE" not in limit_names:
        raise SnapshotValidationError(
            f"{location} process {process.pid} has no RLIMIT_NOFILE record"
        )


def write_json_file(path: Path, value: dict[str, object]) -> None:
    """Write deterministic JSON with no non-finite values.

    :param path: Destination path.
    :param value: JSON-compatible object.
    """
    payload = (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode()
    raw_descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
        0o600,
    )
    descriptor = os.fdopen(raw_descriptor, "wb")
    try:
        descriptor.write(payload)
        descriptor.flush()
        os.fsync(descriptor.fileno())
    finally:
        descriptor.close()
    directory_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
