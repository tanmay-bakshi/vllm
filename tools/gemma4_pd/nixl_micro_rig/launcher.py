"""Spawn-safe campaign launcher with immutable GPU safety boundaries."""

import fcntl
import hashlib
import importlib.util
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import time
import traceback
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from importlib import metadata
from pathlib import Path

from tools.gemma4_pd.nixl_micro_rig.campaign_validator import (
    CampaignValidation,
    validate_campaign,
    validate_campaign_evidence,
)
from tools.gemma4_pd.nixl_micro_rig.config import (
    RigConfig,
    load_config,
    parse_config_bytes,
)
from tools.gemma4_pd.nixl_micro_rig.geometry import (
    build_configured_plan,
    describe_plan,
)
from tools.gemma4_pd.nixl_micro_rig.selection import RunSelection

_HOST_ALLOWED_GPUS = frozenset(range(6))
_HOST_DENIED_GPUS = frozenset({6, 7})
_LOCK_PATH = Path("/data/colleague/locks/gemma4-experiment-lane.lock")


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"terminal JSON duplicates key {key!r}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"terminal JSON contains non-finite constant {value}")


class RunStatus(StrEnum):
    """Terminal disposition of one published campaign."""

    PASS = "PASS"
    FAIL = "FAIL"
    INVALID = "INVALID"


@dataclass(frozen=True)
class CampaignResult:
    """Published campaign path and terminal disposition.

    :ivar run_directory: Atomically published, sealed artifact directory.
    :ivar status: PASS, FAIL, or INVALID disposition.
    """

    run_directory: Path
    status: RunStatus


class ArmExecutionError(RuntimeError):
    """Report a supervised native role failure in a validly started arm."""


class ArmCorrectnessFailure(ArmExecutionError):
    """Report an atomically recorded consumer correctness failure."""


class ArmSupervisorTimeout(ArmExecutionError):
    """Report a nonterminal native arm that exceeded its overall deadline."""


def _sha256(path: Path) -> str:
    """Hash one file.

    :param path: File to hash.
    :returns: Lowercase SHA-256 digest.
    """
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _write_json(path: Path, value: object) -> None:
    """Write one deterministic JSON artifact.

    :param path: Fresh or replaceable artifact path.
    :param value: Strict JSON-compatible value.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _nvidia_smi(*fields: str, compute_apps: bool = False) -> list[list[str]]:
    """Query machine-readable NVIDIA state without initializing CUDA.

    :param fields: ``nvidia-smi`` query fields.
    :param compute_apps: Query compute applications instead of GPU inventory.
    :returns: Stripped CSV rows.
    :raises RuntimeError: If ``nvidia-smi`` fails.
    """
    query = "--query-compute-apps=" if compute_apps else "--query-gpu="
    command = [
        "nvidia-smi",
        query + ",".join(fields),
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"GPU preflight failed: {command}") from error
    if len(result.stdout.strip()) == 0:
        return []
    return [
        [field.strip() for field in line.split(",")]
        for line in result.stdout.splitlines()
        if len(line.strip()) > 0
    ]


def _gpu_inventory() -> tuple[dict[int, dict[str, object]], dict[str, int]]:
    """Read exact GPU index, UUID, and free-memory inventory.

    :returns: Inventory by index and reverse UUID mapping.
    :raises RuntimeError: If rows are malformed or identities duplicate.
    """
    inventory: dict[int, dict[str, object]] = {}
    uuid_to_index: dict[str, int] = {}
    for row in _nvidia_smi("index", "uuid", "memory.free"):
        if len(row) != 3:
            raise RuntimeError(f"malformed GPU inventory row: {row}")
        index_text, gpu_uuid, free_mib_text = row
        index = int(index_text)
        if index in inventory or gpu_uuid in uuid_to_index:
            raise RuntimeError("GPU inventory contains duplicate identity")
        inventory[index] = {"uuid": gpu_uuid, "free_mib": int(free_mib_text)}
        uuid_to_index[gpu_uuid] = index
    return inventory, uuid_to_index


def _required_free_bytes(config: RigConfig) -> dict[int, int]:
    """Compute fail-closed free-memory requirements for every selected GPU.

    :param config: Authenticated production-topology configuration.
    :returns: Required free bytes keyed by physical GPU index.
    """
    source_bytes = config.source_block_count * sum(
        region.row_bytes for region in config.regions
    )
    rank_count = len(config.producer_devices)
    destination_bytes = source_bytes * rank_count
    staging_bytes = config.staging_capacity_mib * 1024 * 1024
    matrix_bytes = config.victim.matrix_order**2 * 4
    victim_bytes = matrix_bytes * 6 + config.victim.redzone_bytes * 5
    safety_margin = 4 * 1024**3
    required = {
        device: source_bytes + safety_margin for device in config.producer_devices
    }
    required[config.consumer_device] = (
        destination_bytes + staging_bytes + victim_bytes + safety_margin
    )
    return required


def preflight(config: RigConfig) -> dict[str, object]:
    """Fail closed on device drift, foreign PIDs, or insufficient memory.

    :param config: Authenticated production-topology configuration.
    :returns: GPU inventory recorded into the run artifact.
    """
    selected = set(config.producer_devices) | {config.consumer_device}
    if not selected <= _HOST_ALLOWED_GPUS or len(selected & _HOST_DENIED_GPUS) > 0:
        raise RuntimeError("selected GPUs violate the immutable host boundary")

    inventory, uuid_to_index = _gpu_inventory()
    if not selected <= inventory.keys():
        raise RuntimeError("configured GPUs are absent from nvidia-smi inventory")

    foreign: list[dict[str, object]] = []
    for row in _nvidia_smi("gpu_uuid", "pid", "process_name", compute_apps=True):
        if len(row) != 3:
            raise RuntimeError(f"malformed compute-process row: {row}")
        gpu_uuid, pid_text, process_name = row
        index = uuid_to_index.get(gpu_uuid)
        if index in selected:
            foreign.append({"gpu": index, "pid": int(pid_text), "name": process_name})
    if len(foreign) > 0:
        raise RuntimeError(f"selected GPUs have foreign compute processes: {foreign}")

    required = _required_free_bytes(config)
    insufficient = {
        device: {
            "required_bytes": required_bytes,
            "free_bytes": int(inventory[device]["free_mib"]) * 1024 * 1024,
        }
        for device, required_bytes in required.items()
        if int(inventory[device]["free_mib"]) * 1024 * 1024 < required_bytes
    }
    if len(insufficient) > 0:
        raise RuntimeError(f"selected GPUs lack required free memory: {insufficient}")
    return {
        "selected_devices": sorted(selected),
        "selected_device_uuids": {
            device: inventory[device]["uuid"] for device in sorted(selected)
        },
        "immutable_allowed_devices": sorted(_HOST_ALLOWED_GPUS),
        "immutable_denied_devices": sorted(_HOST_DENIED_GPUS),
        "inventory": inventory,
        "required_free_bytes": required,
    }


def _selected_gpu_postflight(
    config: RigConfig,
    *,
    timeout_seconds: float = 10.0,
) -> dict[str, object]:
    """Prove that no rig process remains attached to a selected GPU.

    :param config: Complete rig configuration.
    :param timeout_seconds: Maximum NVIDIA process-table convergence interval.
    :returns: Empty selected-device process roster and observed UUIDs.
    :raises RuntimeError: If a process remains or GPU identity changes.
    """
    selected = set(config.producer_devices) | {config.consumer_device}
    deadline = time.monotonic() + timeout_seconds
    remaining: list[dict[str, object]] = []
    device_uuids: dict[int, str] = {}
    while time.monotonic() < deadline:
        inventory, uuid_to_index = _gpu_inventory()
        if not selected <= inventory.keys():
            raise RuntimeError("selected GPU disappeared during postflight")
        device_uuids = {
            device: str(inventory[device]["uuid"]) for device in sorted(selected)
        }
        remaining = []
        for row in _nvidia_smi("gpu_uuid", "pid", "process_name", compute_apps=True):
            if len(row) != 3:
                raise RuntimeError(f"malformed compute-process row: {row}")
            gpu_uuid, pid_text, process_name = row
            device = uuid_to_index.get(gpu_uuid)
            if device in selected:
                remaining.append(
                    {
                        "device": device,
                        "gpu_uuid": gpu_uuid,
                        "pid": int(pid_text),
                        "process_name": process_name,
                    }
                )
        if len(remaining) == 0:
            return {
                "selected_devices": sorted(selected),
                "selected_device_uuids": device_uuids,
                "processes": [],
            }
        time.sleep(0.1)
    raise RuntimeError(f"selected GPUs retain role processes: {remaining}")


def _read_process_identity(pid: int) -> dict[str, object]:
    """Capture PID-reuse-safe identity for one protected GPU process.

    :param pid: Linux process identifier.
    :returns: Start time, process-group/session identity, argv, and executable.
    :raises RuntimeError: If the process cannot be identified atomically enough.
    """
    proc = Path("/proc") / str(pid)
    try:
        stat_text = (proc / "stat").read_text()
        command_end = stat_text.rfind(")")
        if command_end < 0:
            raise RuntimeError(f"process {pid} stat has no command terminator")
        fields = stat_text[command_end + 2 :].split()
        if len(fields) <= 19:
            raise RuntimeError(f"process {pid} stat is truncated")
        starttime_ticks = int(fields[19])
        cmdline = (proc / "cmdline").read_bytes()
        if len(cmdline) == 0:
            raise RuntimeError(f"process {pid} has an empty argv")
        argv = [
            os.fsdecode(argument) for argument in cmdline.rstrip(b"\0").split(b"\0")
        ]
        executable = os.readlink(proc / "exe")
        verify_text = (proc / "stat").read_text()
    except OSError as error:
        raise RuntimeError(f"cannot identify protected GPU process {pid}") from error
    verify_end = verify_text.rfind(")")
    verify_fields = verify_text[verify_end + 2 :].split()
    if verify_end < 0 or len(verify_fields) <= 19:
        raise RuntimeError(f"process {pid} verification stat is malformed")
    if int(verify_fields[19]) != starttime_ticks:
        raise RuntimeError(f"protected GPU process {pid} changed during capture")
    return {
        "pid": pid,
        "starttime_ticks": starttime_ticks,
        "process_group": int(fields[2]),
        "session_id": int(fields[3]),
        "argv_raw_hex": cmdline.hex(),
        "argv": argv,
        "executable": executable,
    }


def _listener_socket_inodes(ports: frozenset[int]) -> dict[int, frozenset[int]]:
    """Resolve exact TCP listener socket inodes for protected ports.

    :param ports: TCP ports whose process owners must remain stable.
    :returns: Listener socket inodes keyed by port.
    :raises RuntimeError: If a port is missing or the kernel table is malformed.
    """
    inodes: dict[int, set[int]] = {port: set() for port in ports}
    for table_path in (Path("/proc/net/tcp"), Path("/proc/net/tcp6")):
        try:
            lines = table_path.read_text().splitlines()[1:]
        except OSError as error:
            raise RuntimeError(
                f"cannot inspect TCP listeners in {table_path}"
            ) from error
        for line in lines:
            fields = line.split()
            if len(fields) < 10 or fields[3] != "0A":
                continue
            try:
                port = int(fields[1].rsplit(":", maxsplit=1)[1], 16)
                inode = int(fields[9])
            except (IndexError, ValueError) as error:
                raise RuntimeError(f"malformed TCP listener row: {line}") from error
            if port in ports:
                inodes[port].add(inode)
    missing = [port for port, values in inodes.items() if len(values) == 0]
    if len(missing) > 0:
        raise RuntimeError(f"protected TCP listeners are absent: {missing}")
    return {port: frozenset(values) for port, values in inodes.items()}


def _listener_process_owners(
    listener_inodes: dict[int, frozenset[int]],
) -> dict[int, tuple[dict[str, object], ...]]:
    """Capture every process retaining each protected listener socket.

    :param listener_inodes: Kernel listener inodes keyed by protected port.
    :returns: PID-reuse-safe owner identities keyed by port.
    :raises RuntimeError: If any socket lacks an observable process owner.
    """
    inode_ports = {
        inode: port for port, inodes in listener_inodes.items() for inode in inodes
    }
    owner_pids: dict[int, set[int]] = {port: set() for port in listener_inodes}
    for proc in Path("/proc").iterdir():
        if not proc.name.isdecimal():
            continue
        try:
            descriptors = tuple((proc / "fd").iterdir())
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        for descriptor in descriptors:
            try:
                target = os.readlink(descriptor)
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
            if not target.startswith("socket:[") or not target.endswith("]"):
                continue
            try:
                inode = int(target[8:-1])
            except ValueError:
                continue
            port = inode_ports.get(inode)
            if port is not None:
                owner_pids[port].add(int(proc.name))
    owners: dict[int, tuple[dict[str, object], ...]] = {}
    for port, pids in owner_pids.items():
        if len(pids) == 0:
            raise RuntimeError(f"protected TCP port {port} has no observable owner")
        owners[port] = tuple(_read_process_identity(pid) for pid in sorted(pids))
    return owners


def _protected_listener_snapshot() -> list[dict[str, object]]:
    """Capture race-checked process ownership for production listener ports.

    :returns: Canonical listener inode and process-owner records.
    :raises RuntimeError: If socket identity changes during capture.
    """
    ports = frozenset({8000, 8820, 8821})
    first_inodes = _listener_socket_inodes(ports)
    first_owners = _listener_process_owners(first_inodes)
    second_inodes = _listener_socket_inodes(ports)
    second_owners = _listener_process_owners(second_inodes)
    third_inodes = _listener_socket_inodes(ports)
    if (
        first_inodes != second_inodes
        or second_inodes != third_inodes
        or first_owners != second_owners
    ):
        raise RuntimeError("protected TCP listener ownership changed during capture")
    return [
        {
            "port": port,
            "socket_inodes": sorted(first_inodes[port]),
            "owners": list(first_owners[port]),
        }
        for port in sorted(ports)
    ]


def _protected_compute_rows(
    protected_devices: frozenset[int],
) -> tuple[dict[int, str], tuple[tuple[int, int, str], ...]]:
    inventory, uuid_to_index = _gpu_inventory()
    missing = protected_devices - inventory.keys()
    if len(missing) > 0:
        raise RuntimeError(
            f"protected GPUs are absent from inventory: {sorted(missing)}"
        )
    identities = {
        device: str(inventory[device]["uuid"]) for device in protected_devices
    }
    rows: list[tuple[int, int, str]] = []
    for row in _nvidia_smi("gpu_uuid", "pid", "process_name", compute_apps=True):
        if len(row) != 3:
            raise RuntimeError(f"malformed compute-process row: {row}")
        gpu_uuid, pid_text, process_name = row
        device = uuid_to_index.get(gpu_uuid)
        if device in protected_devices:
            rows.append((device, int(pid_text), process_name))
    return identities, tuple(sorted(rows))


def protected_gpu_process_snapshot(config: RigConfig) -> dict[str, object]:
    """Capture a race-checked GPU6/7 process identity roster.

    :param config: Configuration carrying the immutable protected-device set.
    :returns: JSON-serializable protected-device and process identities.
    :raises RuntimeError: If NVIDIA or ``/proc`` identities move during capture.
    """
    protected = frozenset(config.protected_devices) & _HOST_DENIED_GPUS
    if protected != _HOST_DENIED_GPUS:
        raise RuntimeError("protected snapshot must cover exactly GPUs 6 and 7")
    device_uuids, first_rows = _protected_compute_rows(protected)
    by_pid: dict[int, dict[str, object]] = {}
    process_names: dict[int, str] = {}
    process_devices: dict[int, set[int]] = {}
    for device, pid, process_name in first_rows:
        if pid in process_names and process_names[pid] != process_name:
            raise RuntimeError(f"protected process {pid} has conflicting names")
        process_names[pid] = process_name
        process_devices.setdefault(pid, set()).add(device)
    for pid in sorted(process_names):
        record = _read_process_identity(pid)
        record["process_name"] = process_names[pid]
        record["devices"] = sorted(process_devices[pid])
        by_pid[pid] = record
    verify_uuids, second_rows = _protected_compute_rows(protected)
    if device_uuids != verify_uuids or first_rows != second_rows:
        raise RuntimeError("protected GPU process roster changed during capture")
    return {
        "protected_devices": sorted(protected),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "device_uuids": {device: device_uuids[device] for device in sorted(protected)},
        "processes": [by_pid[pid] for pid in sorted(by_pid)],
        "listeners": _protected_listener_snapshot(),
    }


@contextmanager
def _exclusive_lock() -> Iterator[None]:
    """Hold the host-wide micro-rig lock for a complete campaign.

    :yields: Control while the exclusive lock is held.
    """
    lock_directory = _LOCK_PATH.parent
    lock_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    directory_stat = lock_directory.lstat()
    if (
        not stat.S_ISDIR(directory_stat.st_mode)
        or stat.S_IMODE(directory_stat.st_mode) != 0o700
    ):
        raise RuntimeError("experiment-lane lock directory is not private")
    flags = os.O_CREAT | os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor = os.open(_LOCK_PATH, flags, 0o600)
    try:
        directory_descriptor = os.open(
            lock_directory,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        lock_stat = os.fstat(descriptor)
        if (
            not stat.S_ISREG(lock_stat.st_mode)
            or stat.S_IMODE(lock_stat.st_mode) != 0o600
        ):
            raise RuntimeError("experiment-lane lock file identity differs")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"micro-rig lock is already held: {_LOCK_PATH}"
            ) from error
        os.ftruncate(descriptor, 0)
        os.lseek(descriptor, 0, os.SEEK_SET)
        os.write(descriptor, f"pid={os.getpid()}\n".encode())
        os.fsync(descriptor)
        yield
    finally:
        os.close(descriptor)


def _cuda_visibility(config: RigConfig, role: str, device_uuids: dict[int, str]) -> str:
    """Build an authoritative UUID visibility roster for one role.

    :param config: Complete rig configuration.
    :param role: Producer or consumer process role.
    :param device_uuids: Preflighted UUIDs keyed by nvidia-smi index.
    :returns: Exact ``CUDA_VISIBLE_DEVICES`` value.
    :raises RuntimeError: If a selected physical device lacks a UUID.
    """
    if role == "producer":
        devices = config.producer_devices
    elif role == "consumer":
        devices = (config.consumer_device,)
    else:
        raise ValueError(f"unknown role: {role}")
    missing = [device for device in devices if device not in device_uuids]
    if len(missing) > 0:
        raise RuntimeError(f"preflight lacks selected GPU UUIDs: {missing}")
    return ",".join(device_uuids[device] for device in devices)


def _role_environment(
    config: RigConfig,
    selection: RunSelection,
    role: str,
    device_uuids: dict[int, str],
) -> dict[str, str]:
    """Build exact child state before its Python interpreter starts.

    :param config: Complete rig configuration.
    :param selection: Exact scenario and fresh-process transport arm.
    :param role: Producer or consumer process role.
    :param device_uuids: Preflighted UUIDs keyed by nvidia-smi index.
    :returns: Complete subprocess environment with no inherited UCX drift.
    """
    selection.validate(config)
    arm = config.transport_arm(selection.transport_arm_name)
    visibility = _cuda_visibility(config, role, device_uuids)
    environment = dict(os.environ)
    for name in tuple(environment):
        if name.startswith("UCX_") or name.startswith("NIXL_"):
            del environment[name]
    environment["CUDA_VISIBLE_DEVICES"] = visibility
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment["UCX_TLS"] = arm.ucx_tls
    environment["UCX_PROTO_INFO"] = "y"
    environment["UCX_RNDV_SCHEME"] = arm.ucx_rndv_scheme
    environment["UCX_NET_DEVICES"] = arm.ucx_net_devices
    if arm.ucx_memtype_cache is None:
        environment.pop("UCX_MEMTYPE_CACHE", None)
    else:
        environment["UCX_MEMTYPE_CACHE"] = arm.ucx_memtype_cache
    return environment


def _copy_run_input(source_root: Path, input_directory: Path, name: str) -> Path:
    """Copy one relative authenticated input into the unpublished run tree.

    :param source_root: Directory containing the source campaign configuration.
    :param input_directory: Fresh run-local input directory.
    :param name: Relative input path recorded in the configuration.
    :returns: Copied input path.
    :raises RuntimeError: If the path escapes either input tree or is a symlink.
    """
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"run input must be a confined relative path: {name}")
    source = source_root / relative
    cursor = source_root
    if source_root.is_symlink():
        raise RuntimeError(f"run input root must not be a symlink: {source_root}")
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise RuntimeError(f"run input path must not traverse a symlink: {source}")
    resolved_root = source_root.resolve()
    resolved_source = source.resolve()
    if not resolved_source.is_relative_to(resolved_root) or not source.is_file():
        raise RuntimeError(f"run input must be a regular non-symlink file: {source}")
    destination = input_directory / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def _git_output(repository: Path, *arguments: str) -> bytes:
    """Run one read-only Git query.

    :param repository: Worktree root.
    :param arguments: Git arguments.
    :returns: Raw standard output.
    :raises RuntimeError: If Git rejects the query.
    """
    command = ["git", "-C", str(repository), *arguments]
    try:
        return subprocess.run(command, check=True, capture_output=True).stdout
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"source provenance command failed: {command}") from error


def _source_repository() -> Path:
    candidate = Path(__file__).resolve()
    root = (
        _git_output(candidate.parent, "rev-parse", "--show-toplevel").decode().strip()
    )
    repository = Path(root).resolve()
    if candidate != repository / "tools/gemma4_pd/nixl_micro_rig/launcher.py":
        raise RuntimeError("launcher is not executing from the attested worktree")
    return repository


def _package_spec(name: str) -> dict[str, object]:
    spec = importlib.util.find_spec(name)
    if spec is None:
        raise RuntimeError(f"cannot resolve required Python package {name}")
    return {
        "name": name,
        "origin": spec.origin,
        "search_locations": (
            list(spec.submodule_search_locations)
            if spec.submodule_search_locations is not None
            else []
        ),
    }


def _collect_source_provenance(run_directory: Path) -> dict[str, object]:
    repository = _source_repository()
    status = _git_output(
        repository,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    diff = _git_output(repository, "diff", "--no-ext-diff", "--binary", "HEAD")
    if len(status) > 0 or len(diff) > 0:
        raise RuntimeError("micro-rig source worktree must be clean")
    provenance_directory = run_directory / "provenance"
    source_tree = provenance_directory / "source-tree"
    source_tree.mkdir(parents=True)
    tracked_output = _git_output(
        repository,
        "ls-files",
        "-z",
        "--",
        "tools/gemma4_pd/nixl_micro_rig",
        "vllm/distributed/kv_transfer/integrity.py",
        "vllm/distributed/kv_transfer/staging_ownership.py",
    )
    tracked = [
        item.decode()
        for item in tracked_output.rstrip(b"\0").split(b"\0")
        if len(item) > 0
    ]
    records: list[dict[str, str]] = []
    for relative_text in tracked:
        relative = Path(relative_text)
        source = repository / relative
        destination = source_tree / relative
        if source.is_symlink() or not source.is_file():
            raise RuntimeError(f"tracked source is not a regular file: {relative}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        digest = _sha256(source)
        if _sha256(destination) != digest:
            raise RuntimeError(f"source provenance copy differs: {relative}")
        records.append({"path": relative.as_posix(), "sha256": digest})
    archive_path = provenance_directory / "source.tar"
    archive_command = [
        "git",
        "-C",
        str(repository),
        "archive",
        "--format=tar",
        f"--output={archive_path}",
        "HEAD",
        "--",
        "tools/gemma4_pd/nixl_micro_rig",
        "vllm/distributed/kv_transfer/integrity.py",
        "vllm/distributed/kv_transfer/staging_ownership.py",
    ]
    try:
        subprocess.run(archive_command, check=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise RuntimeError("cannot archive executed micro-rig source") from error
    status_path = provenance_directory / "git-status.txt"
    diff_path = provenance_directory / "git-diff.patch"
    status_path.write_bytes(status)
    diff_path.write_bytes(diff)
    python_path = Path(sys.executable).resolve()
    nixl_distribution = metadata.distribution("nixl")
    nixl_cuda_distribution = metadata.distribution("nixl-cu13")
    record: dict[str, object] = {
        "repository": str(repository),
        "head": _git_output(repository, "rev-parse", "HEAD").decode().strip(),
        "tree": _git_output(repository, "rev-parse", "HEAD^{tree}").decode().strip(),
        "branch": _git_output(repository, "rev-parse", "--abbrev-ref", "HEAD")
        .decode()
        .strip(),
        "worktree_clean": True,
        "git_status_sha256": hashlib.sha256(status).hexdigest(),
        "git_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "files": records,
        "git_archive": {
            "path": archive_path.relative_to(run_directory).as_posix(),
            "sha256": _sha256(archive_path),
        },
        "runtime_paths": {
            "python": {
                "path": str(python_path),
                "sha256": _sha256(python_path),
                "version": sys.version,
            },
            "nixl": {
                "version": nixl_distribution.version,
                "distribution_root": str(nixl_distribution.locate_file("")),
                "spec": _package_spec("nixl"),
            },
            "nixl-cu13": {
                "version": nixl_cuda_distribution.version,
                "distribution_root": str(nixl_cuda_distribution.locate_file("")),
            },
            "vllm": _package_spec("vllm"),
        },
    }
    _write_json(provenance_directory / "source.json", record)
    return record


def _prepare_run_inputs(
    *,
    source_config_path: Path,
    run_directory: Path,
    selection: RunSelection,
) -> tuple[Path, RigConfig, dict[str, object]]:
    if source_config_path.is_symlink() or not source_config_path.is_file():
        raise RuntimeError("source configuration must be a regular non-symlink file")
    input_directory = run_directory / "inputs"
    input_directory.mkdir()
    source_copy = input_directory / "config.json"
    shutil.copy2(source_config_path, source_copy)
    source_config = parse_config_bytes(
        source_copy.read_bytes(),
        source=str(source_copy),
    )
    selection.validate(source_config)
    selection_path = input_directory / "selection.json"
    _write_json(selection_path, selection.to_json())
    input_names = {
        source_config.source_handshake_manifest,
        source_config.destination_handshake_manifest,
    }
    input_names.update(
        scenario.replay_manifest
        for scenario in source_config.scenarios
        if scenario.replay_manifest is not None
    )
    reserved = {"config.json", "selection.json"}
    if len(input_names & reserved) > 0:
        raise RuntimeError("referenced input collides with reserved run input name")
    copied = [
        _copy_run_input(source_config_path.parent, input_directory, name)
        for name in sorted(input_names)
    ]
    config = load_config(source_copy)
    if config != source_config:
        raise RuntimeError("staged configuration differs after input resolution")
    files = [source_copy, selection_path, *copied]
    records = [
        {
            "path": path.relative_to(run_directory).as_posix(),
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(files)
    ]
    bundle_payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    bundle_fingerprint = hashlib.sha256(bundle_payload).hexdigest()
    provenance = {
        "source_config_path": str(source_config_path),
        "config_fingerprint": config.fingerprint,
        "selection": selection.to_json(),
        "selection_fingerprint": selection.fingerprint,
        "input_bundle_fingerprint": bundle_fingerprint,
        "files": records,
    }
    _write_json(run_directory / "provenance" / "inputs.json", provenance)
    for path in files:
        path.chmod(0o444)
    return source_copy, config, provenance


def _signal_owned_process_group(
    process: subprocess.Popen[bytes],
    expected_starttime_ticks: int,
    signal_number: int,
) -> None:
    """Signal only the still-identical process group created by the launcher.

    :param process: Owned subprocess whose PID is also its session PGID.
    :param expected_starttime_ticks: PID-reuse-safe launch identity.
    :param signal_number: Signal sent to the complete owned process group.
    :raises RuntimeError: If PID identity or process-group ownership differs.
    """
    if process.poll() is not None:
        members = _process_group_members(process.pid)
        for member in members:
            if member["session_id"] != process.pid:
                raise RuntimeError(
                    f"orphaned role PGID {process.pid} has a foreign session member"
                )
            if member["starttime_ticks"] < expected_starttime_ticks:
                raise RuntimeError(
                    f"orphaned role PGID {process.pid} predates its leader"
                )
            _signal_exact_process(
                int(member["pid"]),
                int(member["starttime_ticks"]),
                signal_number,
            )
        return
    identity = _read_process_identity(process.pid)
    if identity["starttime_ticks"] != expected_starttime_ticks:
        raise RuntimeError(f"owned role PID {process.pid} was reused before cleanup")
    try:
        process_group = os.getpgid(process.pid)
    except ProcessLookupError:
        return
    if process_group != process.pid:
        raise RuntimeError(
            f"owned role PID {process.pid} is in unexpected PGID {process_group}"
        )
    try:
        os.killpg(process_group, signal_number)
    except ProcessLookupError:
        return


def _process_group_members(process_group: int) -> list[dict[str, int]]:
    """Capture surviving members of one Linux process group.

    :param process_group: Process-group identifier.
    :returns: PID, start-time, group, and session identities.
    :raises RuntimeError: If a candidate member changes during capture.
    """
    members: list[dict[str, int]] = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdecimal():
            continue
        try:
            first = (proc / "stat").read_text()
            command_end = first.rfind(")")
            fields = first[command_end + 2 :].split()
            if command_end < 0 or len(fields) <= 19:
                continue
            if int(fields[2]) != process_group:
                continue
            identity = {
                "pid": int(proc.name),
                "process_group": int(fields[2]),
                "session_id": int(fields[3]),
                "starttime_ticks": int(fields[19]),
            }
            second = (proc / "stat").read_text()
        except (FileNotFoundError, ProcessLookupError):
            continue
        except OSError as error:
            raise RuntimeError(
                f"cannot inspect process group {process_group}"
            ) from error
        verify_end = second.rfind(")")
        verify_fields = second[verify_end + 2 :].split()
        if verify_end < 0 or len(verify_fields) <= 19:
            raise RuntimeError(
                f"process group {process_group} member identity is malformed"
            )
        if (
            int(verify_fields[2]) != identity["process_group"]
            or int(verify_fields[3]) != identity["session_id"]
            or int(verify_fields[19]) != identity["starttime_ticks"]
        ):
            raise RuntimeError(
                f"process group {process_group} member changed during capture"
            )
        members.append(identity)
    return sorted(members, key=lambda member: member["pid"])


def _signal_exact_process(
    pid: int,
    expected_starttime_ticks: int,
    signal_number: int,
) -> None:
    """Signal one PID-reuse-safe orphaned process-group member.

    :param pid: Linux process identifier.
    :param expected_starttime_ticks: Captured process identity.
    :param signal_number: Signal to deliver through a pidfd.
    :raises RuntimeError: If the identity changes before signaling.
    """
    try:
        descriptor = os.pidfd_open(pid)
    except ProcessLookupError:
        return
    try:
        identity = _read_process_identity(pid)
        if identity["starttime_ticks"] != expected_starttime_ticks:
            raise RuntimeError(f"orphaned role PID {pid} was reused before cleanup")
        signal.pidfd_send_signal(descriptor, signal_number)
    except ProcessLookupError:
        return
    finally:
        os.close(descriptor)


def _owned_process_group_active(process: subprocess.Popen[bytes]) -> bool:
    """Return whether a role leader or any member of its session remains.

    :param process: Owned role subprocess and process-group leader.
    :returns: Whether cleanup still has a live target.
    """
    return process.poll() is None or len(_process_group_members(process.pid)) > 0


def _active_owned_process_groups(
    processes: dict[str, subprocess.Popen[bytes]],
    cleanup_errors: list[str],
) -> set[str]:
    """Find role process groups still requiring cleanup without aborting audit.

    :param processes: Owned role processes by stable name.
    :param cleanup_errors: Cleanup evidence accumulator.
    :returns: Names whose leader or process-group members remain.
    """
    active: set[str] = set()
    for name, process in processes.items():
        try:
            group_active = _owned_process_group_active(process)
        except RuntimeError as error:
            message = f"{name} process-group inspection failed: {error}"
            if message not in cleanup_errors:
                cleanup_errors.append(message)
            group_active = process.poll() is None
        if group_active:
            active.add(name)
    return active


def _capture_launched_process_starttime(
    process: subprocess.Popen[bytes],
    *,
    timeout_seconds: float = 2.0,
    clock: Callable[[], float] = time.monotonic,
    pause: Callable[[float], None] = time.sleep,
) -> int:
    """Capture a fresh child start time under a finite retry window.

    :param process: Newly started role subprocess.
    :param timeout_seconds: Maximum ``/proc`` stabilization interval.
    :param clock: Monotonic clock dependency.
    :param pause: Bounded retry delay dependency.
    :returns: Linux process start time in clock ticks.
    :raises RuntimeError: If the child exits or cannot be identified in time.
    """
    deadline = clock() + timeout_seconds
    last_error: RuntimeError | None = None
    while clock() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"role process {process.pid} exited before identity capture"
            )
        try:
            identity = _read_process_identity(process.pid)
        except RuntimeError as error:
            last_error = error
            pause(0.01)
            continue
        return int(identity["starttime_ticks"])
    raise RuntimeError(
        f"role process {process.pid} identity did not stabilize"
    ) from last_error


def _arm_supervisor_timeout_seconds(
    config: RigConfig,
    selection: RunSelection,
) -> int:
    """Derive a finite whole-arm deadline from selected iteration bounds.

    :param config: Full campaign configuration.
    :param selection: Exact selected scenario.
    :returns: Overall supervisor timeout in seconds.
    """
    scenario = config.scenario(selection.scenario_name)
    return (scenario.iterations + 1) * config.transfer_timeout_seconds + 300


def _wait_for_processes(
    processes: dict[str, subprocess.Popen[bytes]],
    *,
    deadline: float,
    timeout_seconds: int,
    clock: Callable[[], float] = time.monotonic,
    pause: Callable[[float], None] = time.sleep,
) -> dict[str, int]:
    """Wait for role terminals under one finite supervisor deadline.

    :param processes: Owned role subprocesses by stable role name.
    :param deadline: Absolute monotonic deadline.
    :param timeout_seconds: Human-readable configured duration.
    :param clock: Monotonic clock dependency.
    :param pause: Bounded polling delay dependency.
    :returns: Nonzero role exit codes.
    :raises ArmSupervisorTimeout: If any role remains nonterminal at deadline.
    """
    active = set(processes)
    failures: dict[str, int] = {}
    while len(active) > 0 and len(failures) == 0:
        if clock() >= deadline:
            raise ArmSupervisorTimeout(
                "native arm exceeded its nonterminal supervisor deadline of "
                f"{timeout_seconds} seconds"
            )
        for name in tuple(active):
            returncode = processes[name].poll()
            if returncode is None:
                continue
            active.remove(name)
            if returncode != 0:
                failures[name] = returncode
        if len(active) > 0 and len(failures) == 0:
            pause(0.1)
    return failures


def _consumer_reported_correctness_failure(
    *,
    artifact_directory: Path,
    run_id: str,
    config_fingerprint: str,
    input_bundle_fingerprint: str,
    selection: RunSelection,
) -> bool:
    """Recognize only an exact, authoritative consumer FAIL terminal.

    :param artifact_directory: Selected arm artifact directory.
    :param run_id: Campaign UUID.
    :param config_fingerprint: Full configuration identity.
    :param input_bundle_fingerprint: Immutable input-bundle identity.
    :param selection: Exact scenario and transport arm.
    :returns: Whether a typed correctness mismatch was durably recorded.
    """
    path = artifact_directory / "consumer-terminal.json"
    try:
        terminal = json.loads(
            path.read_text(),
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, json.JSONDecodeError, ValueError):
        return False
    if not isinstance(terminal, dict):
        return False
    expected = {
        "schema_version": 1,
        "run_id": run_id,
        "config_fingerprint": config_fingerprint,
        "selection": selection.to_json(),
        "input_bundle_fingerprint": input_bundle_fingerprint,
        "role": "consumer",
        "rank": 0,
        "status": "FAIL",
    }
    return all(
        type(terminal.get(field)) is type(value) and terminal.get(field) == value
        for field, value in expected.items()
    )


def _run_arm(
    *,
    config_path: Path,
    config: RigConfig,
    selection: RunSelection,
    input_bundle_fingerprint: str,
    run_id: str,
    artifact_directory: Path,
    device_uuids: dict[int, str],
) -> None:
    """Run and supervise one fresh-interpreter 4P+1D transport arm.

    :param config_path: Byte-identical, immutable full input configuration.
    :param config: Parsed full configuration.
    :param selection: Exact scenario and transport arm.
    :param input_bundle_fingerprint: SHA-256 identity of every run input.
    :param run_id: Campaign UUID.
    :param artifact_directory: Fresh per-arm artifact directory.
    :param device_uuids: Preflighted physical GPU UUIDs.
    :raises ArmExecutionError: If any role exits unsuccessfully.
    """
    producer_visibility = _cuda_visibility(config, "producer", device_uuids)
    role_specs = [
        ("producer", rank, f"producer-{rank}")
        for rank in range(len(config.producer_devices))
    ] + [("consumer", 0, "consumer")]
    processes: dict[str, subprocess.Popen[bytes]] = {}
    process_starttimes: dict[str, int] = {}
    launch_records: list[dict[str, object]] = []
    failures: dict[str, int] = {}
    repository = _source_repository()
    supervisor_seconds = _arm_supervisor_timeout_seconds(config, selection)
    supervisor_deadline = time.monotonic() + supervisor_seconds
    try:
        for role, rank, name in role_specs:
            environment = _role_environment(config, selection, role, device_uuids)
            visibility = environment["CUDA_VISIBLE_DEVICES"]
            command = [
                sys.executable,
                "-m",
                "tools.gemma4_pd.nixl_micro_rig.role_entry",
                "--config",
                str(config_path),
                "--scenario",
                selection.scenario_name,
                "--transport-arm",
                selection.transport_arm_name,
                "--input-bundle-fingerprint",
                input_bundle_fingerprint,
                "--run-id",
                run_id,
                "--role",
                role,
                "--rank",
                str(rank),
                "--artifact-directory",
                str(artifact_directory),
                "--expected-cuda-visibility",
                visibility,
                "--producer-cuda-visibility",
                producer_visibility,
            ]
            log_path = artifact_directory / f"{name}.log"
            descriptor = os.open(
                log_path,
                os.O_CREAT | os.O_WRONLY | os.O_EXCL,
                0o644,
            )
            try:
                process = subprocess.Popen(
                    command,
                    cwd=repository,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=descriptor,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            finally:
                os.close(descriptor)
            processes[name] = process
            process_starttimes[name] = _capture_launched_process_starttime(process)
            launch_records.append(
                {
                    "run_id": run_id,
                    "config_fingerprint": config.fingerprint,
                    "input_bundle_fingerprint": input_bundle_fingerprint,
                    "selection": selection.to_json(),
                    "name": name,
                    "role": role,
                    "rank": rank,
                    "pid": process.pid,
                    "starttime_ticks": process_starttimes[name],
                    "argv": command,
                    "environment": {
                        key: environment[key]
                        for key in sorted(environment)
                        if key.startswith("UCX_")
                        or key in {"CUDA_DEVICE_ORDER", "CUDA_VISIBLE_DEVICES"}
                    },
                }
            )
        _write_json(artifact_directory / "process-launches.json", launch_records)
        failures = _wait_for_processes(
            processes,
            deadline=supervisor_deadline,
            timeout_seconds=supervisor_seconds,
        )
        if len(failures) > 0:
            if _consumer_reported_correctness_failure(
                artifact_directory=artifact_directory,
                run_id=run_id,
                config_fingerprint=config.fingerprint,
                input_bundle_fingerprint=input_bundle_fingerprint,
                selection=selection,
            ):
                raise ArmCorrectnessFailure(
                    f"micro-rig observed a correctness failure: {failures}"
                )
            raise ArmExecutionError(f"micro-rig arm processes failed: {failures}")
    finally:
        cleanup_errors: list[str] = []
        active_groups = _active_owned_process_groups(processes, cleanup_errors)
        for name, process in processes.items():
            if name in active_groups:
                try:
                    starttime = process_starttimes.get(name)
                    if starttime is None:
                        cleanup_errors.append(
                            f"{name} lacks PID start-time proof for group cleanup"
                        )
                        process.terminate()
                    else:
                        _signal_owned_process_group(
                            process,
                            starttime,
                            signal.SIGTERM,
                        )
                except (OSError, RuntimeError) as error:
                    cleanup_errors.append(f"{name} TERM: {error}")
        terminate_deadline = time.monotonic() + 5.0
        active_groups = _active_owned_process_groups(processes, cleanup_errors)
        while len(active_groups) > 0 and time.monotonic() < terminate_deadline:
            time.sleep(0.05)
            active_groups = _active_owned_process_groups(processes, cleanup_errors)
        for name, process in processes.items():
            if name in active_groups:
                try:
                    starttime = process_starttimes.get(name)
                    if starttime is None:
                        cleanup_errors.append(
                            f"{name} lacks PID start-time proof for KILL cleanup"
                        )
                        process.kill()
                    else:
                        _signal_owned_process_group(
                            process,
                            starttime,
                            signal.SIGKILL,
                        )
                except (OSError, RuntimeError) as error:
                    cleanup_errors.append(f"{name} KILL: {error}")
        kill_deadline = time.monotonic() + 5.0
        active_groups = _active_owned_process_groups(processes, cleanup_errors)
        while len(active_groups) > 0 and time.monotonic() < kill_deadline:
            time.sleep(0.05)
            active_groups = _active_owned_process_groups(processes, cleanup_errors)
        for process in processes.values():
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired as error:
                cleanup_errors.append(f"PID {process.pid} did not reap: {error}")
            try:
                survivors = _process_group_members(process.pid)
            except RuntimeError as error:
                cleanup_errors.append(f"PID {process.pid} group audit failed: {error}")
            else:
                if len(survivors) > 0:
                    cleanup_errors.append(
                        f"PID {process.pid} process group survived cleanup: {survivors}"
                    )
        _write_json(
            artifact_directory / "process-exits.json",
            {
                "run_id": run_id,
                "config_fingerprint": config.fingerprint,
                "input_bundle_fingerprint": input_bundle_fingerprint,
                "selection": selection.to_json(),
                "exit_codes": {
                    name: process.returncode
                    for name, process in sorted(processes.items())
                },
            },
        )
        _write_json(
            artifact_directory / "cleanup.json",
            {
                "run_id": run_id,
                "config_fingerprint": config.fingerprint,
                "input_bundle_fingerprint": input_bundle_fingerprint,
                "selection": selection.to_json(),
                "errors": cleanup_errors,
            },
        )
        if len(cleanup_errors) > 0 and sys.exc_info()[0] is None:
            raise RuntimeError(f"role cleanup failed: {cleanup_errors}")


def _write_internal_checksums(run_directory: Path) -> None:
    checksum_path = run_directory / "SHA256SUMS"
    records: list[str] = []
    for path in sorted(run_directory.rglob("*")):
        if not path.is_file() or path == checksum_path:
            continue
        relative = path.relative_to(run_directory).as_posix()
        if "\n" in relative or "\r" in relative:
            raise RuntimeError("artifact path cannot be represented in SHA256SUMS")
        records.append(f"{_sha256(path)}  {relative}\n")
    checksum_path.write_text("".join(records))
    for line in checksum_path.read_text().splitlines():
        expected, relative = line.split("  ", maxsplit=1)
        if _sha256(run_directory / relative) != expected:
            raise RuntimeError(f"internal checksum verification failed: {relative}")


def _seal_tree(run_directory: Path) -> None:
    for path in sorted(run_directory.rglob("*"), reverse=True):
        if path.is_symlink():
            raise RuntimeError(f"run artifact cannot contain a symlink: {path}")
        path.chmod(0o555 if path.is_dir() else 0o444)
    run_directory.chmod(0o555)


def _fsync_tree(run_directory: Path) -> None:
    """Flush every artifact and directory before atomic publication.

    :param run_directory: Complete unpublished run tree.
    """
    for path in sorted(run_directory.rglob("*")):
        flags = os.O_RDONLY | (os.O_DIRECTORY if path.is_dir() else 0)
        descriptor = os.open(path, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    descriptor = os.open(run_directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_run(
    *,
    build_directory: Path,
    artifact_root: Path,
    run_id: str,
    status: RunStatus,
) -> Path:
    suffix = "" if status is RunStatus.PASS else f".{status.value}"
    destination = artifact_root / f"{run_id}{suffix}"
    if destination.exists():
        raise RuntimeError(f"campaign publication already exists: {destination}")
    _write_internal_checksums(build_directory)
    _fsync_tree(build_directory)
    _seal_tree(build_directory)
    _fsync_tree(build_directory)
    complete_validation = validate_campaign(
        build_directory,
        disposition=status.value,
        expected_published_name=destination.name,
    )
    if complete_validation.passed is False:
        raise RuntimeError(
            "sealed campaign candidate failed final validation: "
            f"{complete_validation.errors}"
        )
    os.replace(build_directory, destination)
    descriptor = os.open(artifact_root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return destination


def _run_campaign_locked(
    config_path: Path,
    artifact_root: Path,
    *,
    selection: RunSelection,
) -> CampaignResult:
    """Run one selected scenario and UCX arm, then seal all evidence.

    :param config_path: Authenticated production-topology configuration.
    :param artifact_root: Parent directory for atomic run publication.
    :param selection: Exact configured scenario and transport arm to execute.
    :returns: Published run path and PASS, FAIL, or INVALID status.
    """
    artifact_root.mkdir(parents=True, exist_ok=True)
    run_id = str(uuid.uuid4())
    build_directory = artifact_root / f".{run_id}.partial.{os.getpid()}"
    build_directory.mkdir(parents=False, exist_ok=False)
    started_ns = time.time_ns()
    status = RunStatus.INVALID
    detail = "campaign did not reach native execution"
    failure_trace = ""
    arm_completed = False
    correctness_failure = False
    protected_match = False
    final_evidence_validation = CampaignValidation(
        disposition=RunStatus.INVALID.value,
        passed=False,
        errors=("campaign validation did not run",),
        counts={},
    )
    attempted_result_validation: CampaignValidation | None = None
    config: RigConfig | None = None
    input_record: dict[str, object] | None = None
    try:
        source_record = _collect_source_provenance(build_directory)
        run_config_path, config, input_record = _prepare_run_inputs(
            source_config_path=config_path.absolute(),
            run_directory=build_directory,
            selection=selection,
        )
        scenario = config.scenario(selection.scenario_name)
        plan = build_configured_plan(run_config_path, config, scenario)
        _write_json(
            build_directory / "plan.json",
            {
                "run_id": run_id,
                "config_fingerprint": config.fingerprint,
                "input_bundle_fingerprint": input_record["input_bundle_fingerprint"],
                "selection": selection.to_json(),
                "plan": describe_plan(config, scenario, plan),
            },
        )
        _write_json(
            build_directory / "campaign-start.json",
            {
                "schema_version": 1,
                "run_id": run_id,
                "started_ns": started_ns,
                "selection": selection.to_json(),
                "selection_fingerprint": selection.fingerprint,
                "config_fingerprint": config.fingerprint,
                "source_head": source_record["head"],
                "input_bundle_fingerprint": input_record["input_bundle_fingerprint"],
            },
        )
        execution_failure: str | None = None
        execution_trace = ""
        before: dict[str, object] | None = None
        after: dict[str, object] | None = None
        protected_identity = {
            "run_id": run_id,
            "config_fingerprint": config.fingerprint,
            "input_bundle_fingerprint": input_record["input_bundle_fingerprint"],
            "selection": selection.to_json(),
        }
        before = {
            **protected_gpu_process_snapshot(config),
            **protected_identity,
        }
        _write_json(build_directory / "protected-gpus-before.json", before)
        try:
            preflight_record = {
                **preflight(config),
                "run_id": run_id,
                "config_fingerprint": config.fingerprint,
                "input_bundle_fingerprint": input_record["input_bundle_fingerprint"],
                "selection": selection.to_json(),
            }
            _write_json(build_directory / "preflight.json", preflight_record)
            raw_device_uuids = preflight_record["selected_device_uuids"]
            if not isinstance(raw_device_uuids, dict) or not all(
                type(device) is int and type(gpu_uuid) is str
                for device, gpu_uuid in raw_device_uuids.items()
            ):
                raise RuntimeError("preflight GPU UUID map is malformed")
            arm_directory = (
                build_directory
                / "arms"
                / selection.transport_arm_name
                / selection.scenario_name
            )
            arm_directory.mkdir(parents=True)
            _run_arm(
                config_path=run_config_path,
                config=config,
                selection=selection,
                input_bundle_fingerprint=str(input_record["input_bundle_fingerprint"]),
                run_id=run_id,
                artifact_directory=arm_directory,
                device_uuids=dict(raw_device_uuids),
            )
            arm_completed = True
        except ArmCorrectnessFailure as error:
            correctness_failure = True
            execution_failure = str(error)
            execution_trace = traceback.format_exc()
        except ArmExecutionError as error:
            execution_failure = str(error)
            execution_trace = traceback.format_exc()
        finally:
            selected_postflight = {
                **_selected_gpu_postflight(config),
                **protected_identity,
            }
            _write_json(
                build_directory / "selected-gpus-after.json",
                selected_postflight,
            )
            after = {
                **protected_gpu_process_snapshot(config),
                **protected_identity,
            }
            _write_json(build_directory / "protected-gpus-after.json", after)
        protected_match = before == after
        if protected_match is False:
            detail = "protected GPU process identities changed"
        elif execution_failure is not None:
            if correctness_failure:
                status = RunStatus.FAIL
            detail = execution_failure
            failure_trace = execution_trace
        elif arm_completed is False:
            detail = "native arm did not report a terminal result"
        else:
            status = RunStatus.PASS
            detail = "native arm and all evidence invariants passed"
        attempted_result_validation = validate_campaign_evidence(
            build_directory,
            disposition=status.value,
        )
        final_evidence_validation = attempted_result_validation
        if (
            status in {RunStatus.PASS, RunStatus.FAIL}
            and attempted_result_validation.passed is False
        ):
            status = RunStatus.INVALID
            detail = "cross-artifact validation failed"
            final_evidence_validation = validate_campaign_evidence(
                build_directory,
                disposition=status.value,
            )
    except (Exception, KeyboardInterrupt) as error:
        status = RunStatus.INVALID
        detail = str(error)
        failure_trace = traceback.format_exc()
        final_evidence_validation = validate_campaign_evidence(
            build_directory,
            disposition=RunStatus.INVALID.value,
        )
    final_input_fingerprint = (
        str(input_record["input_bundle_fingerprint"])
        if input_record is not None
        else None
    )
    _write_json(
        build_directory / "validation.json",
        {
            "schema_version": 1,
            "run_id": run_id,
            "config_fingerprint": config.fingerprint if config is not None else None,
            "input_bundle_fingerprint": final_input_fingerprint,
            "selection": selection.to_json(),
            "attempted_result_validation": (
                attempted_result_validation.to_json()
                if attempted_result_validation is not None
                else None
            ),
            "final_evidence_validation": final_evidence_validation.to_json(),
        },
    )
    _write_json(
        build_directory / "run-status.json",
        {
            "schema_version": 1,
            "run_id": run_id,
            "status": status.value,
            "detail": detail,
            "traceback": failure_trace,
            "started_ns": started_ns,
            "completed_ns": time.time_ns(),
            "selection": selection.to_json(),
            "selection_fingerprint": selection.fingerprint,
            "input_bundle_fingerprint": final_input_fingerprint,
            "arm_completed": arm_completed,
            "protected_gpu_identity_match": protected_match,
            "evidence_validation_passed": final_evidence_validation.passed,
            "attempted_result_validation_passed": (
                attempted_result_validation.passed
                if attempted_result_validation is not None
                else None
            ),
            "config_fingerprint": config.fingerprint if config is not None else None,
        },
    )
    published = _publish_run(
        build_directory=build_directory,
        artifact_root=artifact_root,
        run_id=run_id,
        status=status,
    )
    return CampaignResult(run_directory=published, status=status)


def run_campaign(
    config_path: Path,
    artifact_root: Path,
    *,
    selection: RunSelection,
) -> CampaignResult:
    """Run and publish one campaign under exclusive experiment-lane ownership.

    :param config_path: Authenticated production-topology configuration.
    :param artifact_root: Parent directory for atomic run publication.
    :param selection: Exact configured scenario and transport arm to execute.
    :returns: Published run path and PASS, FAIL, or INVALID status.
    """
    with _exclusive_lock():
        return _run_campaign_locked(
            config_path,
            artifact_root,
            selection=selection,
        )
