"""Spawn-safe campaign launcher with immutable GPU safety boundaries."""

import fcntl
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import traceback
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import TypedDict

from tools.gemma4_pd.nixl_micro_rig.config import RigConfig, load_config
from tools.gemma4_pd.nixl_micro_rig.geometry import (
    build_configured_plan,
    describe_plan,
)

_HOST_ALLOWED_GPUS = frozenset(range(6))
_HOST_DENIED_GPUS = frozenset({6, 7})
_LOCK_PATH = Path("/data/colleague/locks/gemma4-nixl-micro-rig.lock")


class _GpuInventoryRow(TypedDict):
    """Describe one preflighted GPU inventory row."""

    uuid: str
    free_mib: int


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

    foreign: list[dict[str, object]] = []
    for gpu_uuid, pid_text, process_name in _nvidia_smi(
        "gpu_uuid", "pid", "process_name", compute_apps=True
    ):
        selected_index = uuid_to_index.get(gpu_uuid)
        if selected_index is not None and selected_index in selected:
            foreign.append(
                {
                    "gpu": selected_index,
                    "pid": int(pid_text),
                    "name": process_name,
                }
            )
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


def run_campaign(config_path: Path, artifact_root: Path) -> Path:
    """Run every UCX arm in fresh 4P+1D process groups.

    :param config_path: Authenticated production-topology configuration.
    :param artifact_root: Parent directory for the immutable run directory.
    :returns: Created run artifact directory.
    """
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
