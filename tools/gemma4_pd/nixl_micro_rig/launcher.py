"""Spawn-safe campaign launcher with immutable GPU safety boundaries."""

import fcntl
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import TypedDict

from tools.gemma4_pd.nixl_micro_rig.config import RigConfig, load_config

_HOST_ALLOWED_GPUS = frozenset(range(6))
_HOST_DENIED_GPUS = frozenset({6, 7})
_LOCK_PATH = Path("/data/colleague/locks/gemma4-nixl-micro-rig.lock")
_TARGET2_PACK_SLOT_COUNT = 2
_TARGET2_GUARD_BYTES = 1024 * 1024
_ARM_ENVIRONMENT_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    "UCX_TLS",
    "UCX_PROTO_INFO",
    "UCX_MEMTYPE_CACHE",
    "UCX_RNDV_SCHEME",
)
_TARGET2_SOURCE_PATHS = (
    "tools/gemma4_pd/nixl_micro_rig/__main__.py",
    "tools/gemma4_pd/nixl_micro_rig/attestation.py",
    "tools/gemma4_pd/nixl_micro_rig/config.py",
    "tools/gemma4_pd/nixl_micro_rig/data.py",
    "tools/gemma4_pd/nixl_micro_rig/geometry.py",
    "tools/gemma4_pd/nixl_micro_rig/handshake.py",
    "tools/gemma4_pd/nixl_micro_rig/launcher.py",
    "tools/gemma4_pd/nixl_micro_rig/protocol.py",
    "tools/gemma4_pd/nixl_micro_rig/roles.py",
    "tools/gemma4_pd/nixl_micro_rig/target2_gate.py",
    "tools/gemma4_pd/nixl_micro_rig/target2_gate_protocol.py",
    "tools/gemma4_pd/nixl_micro_rig/target2_gate_roles.py",
    "vllm/distributed/kv_transfer/coalesced_layout.py",
    "vllm/distributed/kv_transfer/integrity.py",
    "vllm/distributed/kv_transfer/kv_connector/v1/nixl/coalesced_pack.py",
    "vllm/distributed/kv_transfer/kv_connector/v1/nixl/coalesced_scatter.py",
    "vllm/distributed/kv_transfer/nixl_contracts.py",
    "vllm/distributed/kv_transfer/staging_ownership.py",
    "vllm/distributed/nixl_utils.py",
)

_GateCaseSpec = tuple[str, str, int, int, int, int, int, int]


class _GpuInventoryRow(TypedDict):
    """Describe one preflighted GPU inventory row."""

    uuid: str
    free_mib: int


def _target2_source_identity() -> dict[str, object]:
    """Hash every Python source file that defines the native Target 2 gate."""
    source_root = Path(__file__).resolve().parents[3]
    records: list[dict[str, object]] = []
    for relative_path in _TARGET2_SOURCE_PATHS:
        path = source_root / relative_path
        before = path.stat()
        payload = path.read_bytes()
        after = path.stat()
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        if before_identity != after_identity or len(payload) != after.st_size:
            raise RuntimeError(f"Target 2 source changed while hashing: {path}")
        records.append(
            {
                "path": relative_path,
                "resolved_path": str(path.resolve()),
                "device": after.st_dev,
                "inode": after.st_ino,
                "size": after.st_size,
                "mtime_ns": after.st_mtime_ns,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    aggregate_payload = json.dumps(
        records,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return {
        "schema_version": 1,
        "source_root": str(source_root),
        "aggregate_sha256": hashlib.sha256(aggregate_payload).hexdigest(),
        "files": records,
    }


def _target2_runtime_distribution_identity() -> dict[str, str]:
    """Require and describe the GPU vLLM distribution used by the gate.

    :returns: Installed distribution version and metadata location.
    :raises RuntimeError: If local package metadata selects a CPU build.
    """
    distribution = importlib.metadata.distribution("vllm")
    version = distribution.version
    if "cpu" in version.lower():
        raise RuntimeError(
            "Target 2 gate resolved a CPU vLLM distribution; stage source "
            "without local egg-info or dist-info metadata"
        )
    return {
        "vllm_distribution_root": str(distribution.locate_file("")),
        "vllm_version": version,
    }


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

    gpu_rows = _nvidia_smi("index", "uuid", "memory.free")
    inventory: dict[int, _GpuInventoryRow] = {}
    uuid_to_index: dict[str, int] = {}
    for index_text, gpu_uuid, free_mib_text in gpu_rows:
        inventory_index = int(index_text)
        inventory[inventory_index] = {
            "uuid": gpu_uuid,
            "free_mib": int(free_mib_text),
        }
        uuid_to_index[gpu_uuid] = inventory_index
    if not selected <= inventory.keys():
        raise RuntimeError("configured GPUs are absent from nvidia-smi inventory")

    compute_processes: dict[int, list[dict[str, object]]] = {
        device: [] for device in inventory
    }
    foreign: list[dict[str, object]] = []
    for gpu_uuid, pid_text, process_name in _nvidia_smi(
        "gpu_uuid", "pid", "process_name", compute_apps=True
    ):
        selected_index = uuid_to_index.get(gpu_uuid)
        if selected_index is None:
            raise RuntimeError(f"compute process reported unknown GPU UUID {gpu_uuid}")
        process = {
            "gpu": selected_index,
            "gpu_uuid": gpu_uuid,
            "pid": int(pid_text),
            "name": process_name,
        }
        compute_processes[selected_index].append(process)
        if selected_index in selected:
            foreign.append(process)
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
        "protected_compute_processes": {
            device: compute_processes[device]
            for device in sorted(config.protected_devices)
            if device in compute_processes
        },
    }


def _target2_postflight(
    config: RigConfig, preflight_record: dict[str, object]
) -> dict[str, object]:
    """Prove selected-GPU cleanup and an unchanged protected process baseline."""
    raw_inventory = preflight_record.get("inventory")
    raw_protected = preflight_record.get("protected_compute_processes")
    if not isinstance(raw_inventory, dict) or not isinstance(raw_protected, dict):
        raise AssertionError("Target 2 preflight process evidence is malformed")
    uuid_to_index: dict[str, int] = {}
    for device, raw_row in raw_inventory.items():
        if type(device) is not int or not isinstance(raw_row, dict):
            raise AssertionError("Target 2 preflight GPU inventory is malformed")
        gpu_uuid = raw_row.get("uuid")
        if type(gpu_uuid) is not str:
            raise AssertionError("Target 2 preflight GPU UUID is malformed")
        uuid_to_index[gpu_uuid] = device

    selected = set(config.producer_devices) | {config.consumer_device}
    deadline = time.monotonic() + 60.0
    while True:
        current: dict[int, list[dict[str, object]]] = {
            device: [] for device in raw_inventory if type(device) is int
        }
        for gpu_uuid, pid_text, process_name in _nvidia_smi(
            "gpu_uuid", "pid", "process_name", compute_apps=True
        ):
            device = uuid_to_index.get(gpu_uuid)
            if device is None:
                raise RuntimeError(f"postflight reported unknown GPU UUID {gpu_uuid}")
            current[device].append(
                {
                    "gpu": device,
                    "gpu_uuid": gpu_uuid,
                    "pid": int(pid_text),
                    "name": process_name,
                }
            )
        occupied = {
            device: current[device]
            for device in sorted(selected)
            if len(current[device]) > 0
        }
        if len(occupied) == 0:
            break
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Target 2 selected GPUs remain occupied: {occupied}")
        time.sleep(0.2)
    observed_protected = {
        device: current[device]
        for device in sorted(config.protected_devices)
        if device in current
    }
    if observed_protected != raw_protected:
        raise RuntimeError(
            "Target 2 protected compute baseline changed: "
            f"observed={observed_protected}, expected={raw_protected}"
        )
    return {
        "schema_version": 1,
        "selected_devices_clear": sorted(selected),
        "protected_compute_processes": observed_protected,
    }


@contextmanager
def _exclusive_lock() -> Iterator[None]:
    """Hold the host-wide micro-rig lock for a complete campaign.

    :yields: Control while the exclusive lock is held.
    """
    _LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK_PATH.open("a+") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"micro-rig lock is already held: {_LOCK_PATH}"
            ) from error
        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(f"pid={os.getpid()}\n")
        lock_file.flush()
        yield


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


def _configure_arm_environment(
    config: RigConfig,
    arm_name: str,
    role: str,
    device_uuids: dict[int, str],
) -> str:
    """Set CUDA and UCX state before importing torch or NIXL.

    :param config: Complete rig configuration.
    :param arm_name: Fresh-process transport arm.
    :param role: Producer or consumer process role.
    :param device_uuids: Preflighted UUIDs keyed by nvidia-smi index.
    :returns: Exact UUID-based CUDA visibility assigned to the role.
    """
    arm = config.transport_arm(arm_name)
    visibility = _cuda_visibility(config, role, device_uuids)
    os.environ["CUDA_VISIBLE_DEVICES"] = visibility
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["UCX_TLS"] = arm.ucx_tls
    os.environ["UCX_PROTO_INFO"] = "y"
    if arm.ucx_memtype_cache is None:
        os.environ.pop("UCX_MEMTYPE_CACHE", None)
    else:
        os.environ["UCX_MEMTYPE_CACHE"] = arm.ucx_memtype_cache
    if arm.ucx_rndv_scheme is None:
        os.environ.pop("UCX_RNDV_SCHEME", None)
    else:
        os.environ["UCX_RNDV_SCHEME"] = arm.ucx_rndv_scheme
    return visibility


@contextmanager
def _arm_spawn_environment(
    config: RigConfig,
    arm_name: str,
    role: str,
    device_uuids: dict[int, str],
) -> Iterator[str]:
    """Bind a spawned interpreter before it imports any GPU runtime module.

    Python's spawn bootstrap imports the main module before invoking the target
    function. The parent must therefore supply the role environment at
    :meth:`multiprocessing.Process.start`, even though the role validates and
    reapplies the same contract after entry.

    :param config: Complete rig configuration.
    :param arm_name: Fresh-process transport arm.
    :param role: Producer or consumer process role.
    :param device_uuids: Preflighted UUIDs keyed by nvidia-smi index.
    :yields: Exact UUID-based CUDA visibility inherited by the child.
    """
    prior = {key: os.environ.get(key) for key in _ARM_ENVIRONMENT_KEYS}
    visibility = _configure_arm_environment(
        config,
        arm_name,
        role,
        device_uuids,
    )
    try:
        yield visibility
    finally:
        for key, value in prior.items():
            if value is None:
                os.environ.pop(key, None)
                continue
            os.environ[key] = value


def _redirect_process_output(path: Path) -> None:
    """Attach process stdout and stderr to one exclusive native log.

    :param path: Fresh per-role log path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_EXCL, 0o644)
    os.dup2(descriptor, sys.stdout.fileno())
    os.dup2(descriptor, sys.stderr.fileno())
    os.close(descriptor)


def _copy_run_input(source_root: Path, run_directory: Path, name: str) -> None:
    """Copy one relative authenticated input into the immutable run tree.

    :param source_root: Directory containing the campaign configuration.
    :param run_directory: Fresh immutable artifact directory.
    :param name: Relative input path recorded in the configuration.
    :raises RuntimeError: If the path escapes either input tree.
    """
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"run input must be a confined relative path: {name}")
    source = source_root / relative
    destination = run_directory / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _role_entry(
    config_path_text: str,
    arm_name: str,
    run_id: str,
    role: str,
    rank: int,
    artifact_directory_text: str,
    log_path_text: str,
    device_uuids: dict[int, str],
) -> None:
    """Spawn one role after establishing its CUDA and UCX environment.

    :param config_path_text: Immutable run-local configuration path.
    :param arm_name: Fresh-process transport arm identity.
    :param run_id: Campaign UUID.
    :param role: Producer or consumer role.
    :param rank: Role-local rank.
    :param artifact_directory_text: Fresh arm artifact directory path.
    :param log_path_text: Exclusive native output log path.
    :param device_uuids: Preflighted UUIDs keyed by nvidia-smi index.
    """
    config_path = Path(config_path_text)
    artifact_directory = Path(artifact_directory_text)
    _redirect_process_output(Path(log_path_text))
    try:
        config = load_config(config_path)
        visibility = _configure_arm_environment(config, arm_name, role, device_uuids)
        producer_visibility = _cuda_visibility(config, "producer", device_uuids)
        if role == "producer":
            from tools.gemma4_pd.nixl_micro_rig.roles import run_producer

            run_producer(
                config_path=config_path,
                arm_name=arm_name,
                run_id=run_id,
                rank=rank,
                artifact_directory=artifact_directory,
                expected_cuda_visibility=visibility,
            )
        else:
            from tools.gemma4_pd.nixl_micro_rig.roles import run_consumer

            run_consumer(
                config_path=config_path,
                arm_name=arm_name,
                run_id=run_id,
                artifact_directory=artifact_directory,
                expected_cuda_visibility=visibility,
                producer_cuda_visibility=producer_visibility,
            )
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)


def _target2_role_entry(
    config_path_text: str,
    arm_name: str,
    run_id: str,
    role: str,
    rank: int,
    case_specs: tuple[_GateCaseSpec, ...],
    artifact_directory_text: str,
    log_path_text: str,
    device_uuids: dict[int, str],
) -> None:
    """Spawn one Target 2 role after establishing CUDA and UCX state.

    Primitive case specifications keep Torch out of the spawned interpreter
    until GPU visibility has been bound to the preflighted UUID roster.

    :param config_path_text: Immutable run-local configuration path.
    :param arm_name: Fresh-process transport arm identity.
    :param run_id: Campaign UUID.
    :param role: Producer or consumer role.
    :param rank: Role-local rank.
    :param case_specs: Ordered primitive fixed-byte gate cases.
    :param artifact_directory_text: Fresh arm artifact directory path.
    :param log_path_text: Exclusive native output log path.
    :param device_uuids: Preflighted UUIDs keyed by nvidia-smi index.
    """
    config_path = Path(config_path_text)
    artifact_directory = Path(artifact_directory_text)
    _redirect_process_output(Path(log_path_text))
    try:
        config = load_config(config_path)
        visibility = _configure_arm_environment(config, arm_name, role, device_uuids)
        from tools.gemma4_pd.nixl_micro_rig.target2_gate import (
            FragmentationRegime,
            GateArm,
            GateCase,
        )
        from tools.gemma4_pd.nixl_micro_rig.target2_gate_roles import (
            run_target2_gate_consumer,
            run_target2_gate_producer,
        )

        cases = tuple(
            GateCase(
                arm=GateArm(spec[0]),
                fragmentation=FragmentationRegime(
                    name=spec[1],
                    run_count=spec[2],
                    expected_descriptors_per_rank=spec[3],
                ),
                chunk_bytes=spec[4],
                in_flight_depth=spec[5],
                warmup_batches=spec[6],
                measured_batches=spec[7],
            )
            for spec in case_specs
        )
        if role == "producer":
            run_target2_gate_producer(
                config_path=config_path,
                transport_arm_name=arm_name,
                run_id=run_id,
                rank=rank,
                cases=cases,
                artifact_directory=artifact_directory,
                expected_cuda_visibility=visibility,
            )
            return
        run_target2_gate_consumer(
            config_path=config_path,
            transport_arm_name=arm_name,
            run_id=run_id,
            cases=cases,
            artifact_directory=artifact_directory,
            expected_cuda_visibility=visibility,
        )
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)


def _run_arm(
    *,
    config_path: Path,
    config: RigConfig,
    arm_name: str,
    run_id: str,
    artifact_directory: Path,
    device_uuids: dict[int, str],
) -> None:
    """Run and supervise one fresh 4P+1D transport arm.

    :param config_path: Immutable run-local configuration.
    :param config: Parsed configuration matching ``config_path``.
    :param arm_name: Fresh-process transport arm identity.
    :param run_id: Campaign UUID.
    :param artifact_directory: Fresh arm artifact directory.
    :param device_uuids: Preflighted UUIDs keyed by nvidia-smi index.
    :raises RuntimeError: If any role exits unsuccessfully.
    """
    context = multiprocessing.get_context("spawn")
    processes: list[BaseProcess] = []
    try:
        for rank in range(len(config.producer_devices)):
            producer_process = context.Process(
                target=_role_entry,
                name=f"micro-rig-p{rank}",
                args=(
                    str(config_path),
                    arm_name,
                    run_id,
                    "producer",
                    rank,
                    str(artifact_directory),
                    str(artifact_directory / f"producer-{rank}.log"),
                    device_uuids,
                ),
            )
            with _arm_spawn_environment(
                config,
                arm_name,
                "producer",
                device_uuids,
            ):
                producer_process.start()
            processes.append(producer_process)
        consumer_process = context.Process(
            target=_role_entry,
            name="micro-rig-d0",
            args=(
                str(config_path),
                arm_name,
                run_id,
                "consumer",
                0,
                str(artifact_directory),
                str(artifact_directory / "consumer.log"),
                device_uuids,
            ),
        )
        with _arm_spawn_environment(
            config,
            arm_name,
            "consumer",
            device_uuids,
        ):
            consumer_process.start()
        processes.append(consumer_process)
        active: set[BaseProcess] = set(processes)
        failures: dict[str, int | None] = {}
        while len(active) > 0 and len(failures) == 0:
            for active_process in tuple(active):
                active_process.join(timeout=0.1)
                if active_process.exitcode is None:
                    continue
                active.remove(active_process)
                if active_process.exitcode != 0:
                    failures[active_process.name] = active_process.exitcode
        if len(failures) > 0:
            raise RuntimeError(f"micro-rig arm processes failed: {failures}")
    finally:
        for managed_process in processes:
            if managed_process.is_alive():
                managed_process.terminate()
        for managed_process in processes:
            managed_process.join(timeout=5.0)
        for managed_process in processes:
            if managed_process.is_alive():
                managed_process.kill()
        for managed_process in processes:
            managed_process.join()


def _run_target2_arm(
    *,
    config_path: Path,
    config: RigConfig,
    arm_name: str,
    run_id: str,
    case_specs: tuple[_GateCaseSpec, ...],
    artifact_directory: Path,
    device_uuids: dict[int, str],
) -> None:
    """Run and supervise one fresh 4P+1D Target 2 transport arm.

    :param config_path: Immutable run-local configuration.
    :param config: Parsed configuration matching ``config_path``.
    :param arm_name: Fresh-process transport arm identity.
    :param run_id: Campaign UUID.
    :param case_specs: Ordered primitive fixed-byte gate cases.
    :param artifact_directory: Fresh arm artifact directory.
    :param device_uuids: Preflighted UUIDs keyed by nvidia-smi index.
    :raises RuntimeError: If any role exits unsuccessfully.
    """
    context = multiprocessing.get_context("spawn")
    processes: list[BaseProcess] = []
    try:
        for rank in range(len(config.producer_devices)):
            producer_process = context.Process(
                target=_target2_role_entry,
                name=f"target2-gate-p{rank}",
                args=(
                    str(config_path),
                    arm_name,
                    run_id,
                    "producer",
                    rank,
                    case_specs,
                    str(artifact_directory),
                    str(artifact_directory / f"producer-{rank}.log"),
                    device_uuids,
                ),
            )
            with _arm_spawn_environment(
                config,
                arm_name,
                "producer",
                device_uuids,
            ):
                producer_process.start()
            processes.append(producer_process)
        consumer_process = context.Process(
            target=_target2_role_entry,
            name="target2-gate-d0",
            args=(
                str(config_path),
                arm_name,
                run_id,
                "consumer",
                0,
                case_specs,
                str(artifact_directory),
                str(artifact_directory / "consumer.log"),
                device_uuids,
            ),
        )
        with _arm_spawn_environment(
            config,
            arm_name,
            "consumer",
            device_uuids,
        ):
            consumer_process.start()
        processes.append(consumer_process)
        active: set[BaseProcess] = set(processes)
        failures: dict[str, int | None] = {}
        while len(active) > 0 and len(failures) == 0:
            for active_process in tuple(active):
                active_process.join(timeout=0.1)
                if active_process.exitcode is None:
                    continue
                active.remove(active_process)
                if active_process.exitcode != 0:
                    failures[active_process.name] = active_process.exitcode
        if len(failures) > 0:
            raise RuntimeError(f"Target 2 gate processes failed: {failures}")
    finally:
        for managed_process in processes:
            if managed_process.is_alive():
                managed_process.terminate()
        for managed_process in processes:
            managed_process.join(timeout=5.0)
        for managed_process in processes:
            if managed_process.is_alive():
                managed_process.kill()
        for managed_process in processes:
            managed_process.join()


def _target2_additional_memory(
    config: RigConfig, maximum_chunk_bytes: int
) -> dict[int, int]:
    """Return guarded bounded-pool bytes beyond the base campaign preflight.

    :param config: Exact TP4-to-TP1 rig configuration.
    :param maximum_chunk_bytes: Largest selected per-rank packed chunk.
    :returns: Additional required bytes keyed by physical GPU index.
    """
    guarded_slot_bytes = maximum_chunk_bytes + 2 * _TARGET2_GUARD_BYTES
    producer_pool_bytes = _TARGET2_PACK_SLOT_COUNT * guarded_slot_bytes
    consumer_slot_bytes = (
        len(config.producer_devices) * maximum_chunk_bytes + 2 * _TARGET2_GUARD_BYTES
    )
    consumer_pool_bytes = _TARGET2_PACK_SLOT_COUNT * consumer_slot_bytes
    additional = {device: producer_pool_bytes for device in config.producer_devices}
    additional[config.consumer_device] = consumer_pool_bytes
    return additional


def _validate_target2_memory(
    config: RigConfig,
    preflight_record: dict[str, object],
    maximum_chunk_bytes: int,
) -> dict[int, int]:
    """Require free memory for the base rig plus guarded Target 2 pools.

    :param config: Exact TP4-to-TP1 rig configuration.
    :param preflight_record: Successful base preflight evidence.
    :param maximum_chunk_bytes: Largest selected per-rank packed chunk.
    :returns: Total required bytes keyed by physical GPU index.
    :raises RuntimeError: If inventory evidence is malformed or insufficient.
    """
    raw_inventory = preflight_record.get("inventory")
    raw_required = preflight_record.get("required_free_bytes")
    if not isinstance(raw_inventory, dict) or not isinstance(raw_required, dict):
        raise AssertionError("preflight memory evidence is malformed")
    additional = _target2_additional_memory(config, maximum_chunk_bytes)
    required = {
        device: int(raw_required[device]) + additional[device] for device in additional
    }
    insufficient: dict[int, dict[str, int]] = {}
    for device, required_bytes in required.items():
        inventory_row = raw_inventory.get(device)
        if not isinstance(inventory_row, dict):
            raise AssertionError(f"preflight inventory lacks GPU {device}")
        free_mib = inventory_row.get("free_mib")
        if type(free_mib) is not int:
            raise AssertionError(f"preflight GPU {device} free memory is malformed")
        free_bytes = free_mib * 1024 * 1024
        if free_bytes < required_bytes:
            insufficient[device] = {
                "required_bytes": required_bytes,
                "free_bytes": free_bytes,
            }
    if len(insufficient) > 0:
        raise RuntimeError(
            f"selected GPUs lack Target 2 bounded-pool memory: {insufficient}"
        )
    return required


def run_campaign(config_path: Path, artifact_root: Path) -> Path:
    """Run every UCX arm in fresh 4P+1D process groups.

    :param config_path: Authenticated production-topology configuration.
    :param artifact_root: Parent directory for the immutable run directory.
    :returns: Created run artifact directory.
    """
    from tools.gemma4_pd.nixl_micro_rig.geometry import (
        build_configured_plan,
        describe_plan,
    )

    source_config = load_config(config_path)
    source_plan_records = [
        describe_plan(
            source_config,
            scenario,
            build_configured_plan(config_path, source_config, scenario),
        )
        for scenario in source_config.scenarios
    ]
    run_id = str(uuid.uuid4())
    run_directory = artifact_root / run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, run_directory / config_path.name)
    input_names = {
        source_config.legacy_source_handshake_manifest,
        source_config.legacy_destination_handshake_manifest,
        source_config.semantic_handshake_manifest,
    } | {
        scenario.replay_manifest
        for scenario in source_config.scenarios
        if scenario.replay_manifest is not None
    }
    for input_name in input_names:
        _copy_run_input(config_path.parent, run_directory, input_name)
    run_config_path = run_directory / config_path.name
    config = load_config(run_config_path)
    plan_records = [
        describe_plan(
            config,
            scenario,
            build_configured_plan(run_config_path, config, scenario),
        )
        for scenario in config.scenarios
    ]
    if plan_records != source_plan_records:
        raise RuntimeError("run-local canonical plans differ from their source inputs")
    (run_directory / "plans.json").write_text(
        json.dumps(plan_records, indent=2, sort_keys=True)
    )
    (run_directory / "campaign-start.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "config_fingerprint": config.fingerprint,
                "config_path": config_path.name,
                "transport_arms": [arm.name for arm in config.transport_arms],
                "plan_count": len(plan_records),
            },
            indent=2,
            sort_keys=True,
        )
    )
    with _exclusive_lock():
        preflight_record = preflight(config)
        (run_directory / "preflight.json").write_text(
            json.dumps(preflight_record, indent=2, sort_keys=True)
        )
        raw_device_uuids = preflight_record["selected_device_uuids"]
        if not isinstance(raw_device_uuids, dict) or not all(
            type(device) is int and type(gpu_uuid) is str
            for device, gpu_uuid in raw_device_uuids.items()
        ):
            raise AssertionError("preflight GPU UUID map is malformed")
        device_uuids = dict(raw_device_uuids)
        for arm in config.transport_arms:
            arm_directory = run_directory / arm.name
            arm_directory.mkdir()
            _run_arm(
                config_path=run_config_path,
                config=config,
                arm_name=arm.name,
                run_id=run_id,
                artifact_directory=arm_directory,
                device_uuids=device_uuids,
            )
    (run_directory / "campaign-complete.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "config_fingerprint": config.fingerprint,
                "completed_arms": [arm.name for arm in config.transport_arms],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return run_directory


def run_target2_gate(
    config_path: Path,
    artifact_root: Path,
    *,
    transport_arm_names: tuple[str, ...],
    chunk_mib: tuple[int, ...],
    in_flight_depths: tuple[int, ...],
    warmup_batches: int,
    measured_batches: int,
) -> Path:
    """Run the fixed-byte direct, packed-READ, and packed-WRITE gate.

    :param config_path: Authenticated exact-2K configuration.
    :param artifact_root: Parent directory for the immutable run directory.
    :param transport_arm_names: Explicit fresh-process UCX arms.
    :param chunk_mib: Selected bounded per-rank chunk capacities.
    :param in_flight_depths: Selected offered request depths.
    :param warmup_batches: Unmeasured batches per cell.
    :param measured_batches: Measured batches per cell.
    :returns: Created immutable result directory.
    """
    if len(transport_arm_names) == 0 or len(set(transport_arm_names)) != len(
        transport_arm_names
    ):
        raise ValueError("Target 2 transport arms must be non-empty and unique")
    source_config = load_config(config_path)
    for arm_name in transport_arm_names:
        source_config.transport_arm(arm_name)

    from tools.gemma4_pd.nixl_micro_rig.target2_gate import (
        GateArm,
        attest_cuda_ipc_write_protocol,
        describe_gate_case,
        gate_matrix,
        summarize_gate_records,
    )

    cases = gate_matrix(
        chunk_mib=chunk_mib,
        in_flight_depths=in_flight_depths,
        warmup_batches=warmup_batches,
        measured_batches=measured_batches,
    )
    source_plan_records = [
        describe_gate_case(source_config, source_config.scenarios[0], case)
        for case in cases
    ]
    staging_capacity = source_config.staging_capacity_mib * 1024 * 1024
    for record in source_plan_records:
        if record.get("arm") != "direct_read":
            continue
        consumer_receive_bytes = record.get("consumer_receive_bytes")
        if type(consumer_receive_bytes) is not int:
            raise AssertionError("Target 2 planned consumer bytes are malformed")
        if consumer_receive_bytes > staging_capacity:
            raise RuntimeError("Target 2 direct concurrency exceeds staging capacity")
    case_specs: tuple[_GateCaseSpec, ...] = tuple(
        (
            case.arm.value,
            case.fragmentation.name,
            case.fragmentation.run_count,
            case.fragmentation.expected_descriptors_per_rank,
            case.chunk_bytes,
            case.in_flight_depth,
            case.warmup_batches,
            case.measured_batches,
        )
        for case in cases
    )
    maximum_chunk_bytes = max(
        (case.chunk_bytes for case in cases if case.chunk_bytes > 0),
        default=64 * 1024 * 1024,
    )
    source_identity = _target2_source_identity()
    source_identity["runtime_distribution"] = _target2_runtime_distribution_identity()

    run_id = str(uuid.uuid4())
    run_directory = artifact_root / run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, run_directory / config_path.name)
    input_names = {
        source_config.legacy_source_handshake_manifest,
        source_config.legacy_destination_handshake_manifest,
        source_config.semantic_handshake_manifest,
    }
    for input_name in input_names:
        _copy_run_input(config_path.parent, run_directory, input_name)
    run_config_path = run_directory / config_path.name
    config = load_config(run_config_path)
    plan_records = [
        describe_gate_case(config, config.scenarios[0], case) for case in cases
    ]
    if plan_records != source_plan_records:
        raise RuntimeError("run-local Target 2 plans differ from source inputs")
    (run_directory / "target2-gate-plan.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "evidence_scope": "target2_fixed_byte_native_transport_gate",
                "run_id": run_id,
                "config_fingerprint": config.fingerprint,
                "request_bytes": 1_052_508_160,
                "transport_arms": list(transport_arm_names),
                "cases": plan_records,
            },
            indent=2,
            sort_keys=True,
        )
    )
    (run_directory / "target2-code-identity.json").write_text(
        json.dumps(source_identity, indent=2, sort_keys=True)
    )
    (run_directory / "campaign-start.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": run_id,
                "config_fingerprint": config.fingerprint,
                "config_path": config_path.name,
                "transport_arms": list(transport_arm_names),
                "case_count": len(cases),
                "warmup_batches_per_case": warmup_batches,
                "measured_batches_per_case": measured_batches,
                "source_aggregate_sha256": source_identity["aggregate_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    with _exclusive_lock():
        preflight_record = preflight(config)
        target2_required = _validate_target2_memory(
            config,
            preflight_record,
            maximum_chunk_bytes,
        )
        preflight_record["target2_total_required_free_bytes"] = target2_required
        (run_directory / "preflight.json").write_text(
            json.dumps(preflight_record, indent=2, sort_keys=True)
        )
        raw_device_uuids = preflight_record["selected_device_uuids"]
        if not isinstance(raw_device_uuids, dict) or not all(
            type(device) is int and type(gpu_uuid) is str
            for device, gpu_uuid in raw_device_uuids.items()
        ):
            raise AssertionError("preflight GPU UUID map is malformed")
        device_uuids = dict(raw_device_uuids)
        for arm_name in transport_arm_names:
            arm_directory = run_directory / arm_name
            arm_directory.mkdir()
            _run_target2_arm(
                config_path=run_config_path,
                config=config,
                arm_name=arm_name,
                run_id=run_id,
                case_specs=case_specs,
                artifact_directory=arm_directory,
                device_uuids=device_uuids,
            )
            protocol_evidence: dict[str, object] | None = None
            arm = config.transport_arm(arm_name)
            has_packed_write = any(case.arm is GateArm.PACKED_WRITE for case in cases)
            if "cuda_ipc" in arm.ucx_tls.split(",") and has_packed_write:
                protocol_evidence = attest_cuda_ipc_write_protocol(
                    arm_directory,
                    producer_count=len(config.producer_devices),
                )
                (arm_directory / "target2-cuda-ipc-write-protocol.json").write_text(
                    json.dumps(protocol_evidence, indent=2, sort_keys=True)
                )
            raw_records = json.loads(
                (arm_directory / "target2-gate-batches.json").read_text()
            )
            if not isinstance(raw_records, list) or not all(
                isinstance(record, dict) for record in raw_records
            ):
                raise RuntimeError("Target 2 batch artifact is malformed")
            summary = summarize_gate_records(cases, raw_records)
            summary["run_id"] = run_id
            summary["config_fingerprint"] = config.fingerprint
            summary["transport_arm"] = arm_name
            summary["cuda_ipc_write_protocol"] = protocol_evidence
            (arm_directory / "target2-gate-summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True)
            )
        postflight_record = _target2_postflight(config, preflight_record)
        (run_directory / "postflight.json").write_text(
            json.dumps(postflight_record, indent=2, sort_keys=True)
        )
    final_source_identity = _target2_source_identity()
    final_source_identity["runtime_distribution"] = (
        _target2_runtime_distribution_identity()
    )
    if final_source_identity != source_identity:
        raise RuntimeError("Target 2 source identity changed during the campaign")
    (run_directory / "campaign-complete.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": run_id,
                "config_fingerprint": config.fingerprint,
                "completed_arms": list(transport_arm_names),
                "case_count": len(cases),
                "verdict": "complete_exact_integrity",
                "source_aggregate_sha256": source_identity["aggregate_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return run_directory
