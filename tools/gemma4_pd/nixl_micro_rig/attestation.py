"""Loaded-process and NIXL backend attestation for the micro-rig."""

import ctypes
import hashlib
import importlib.util
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import torch
from nixl import _api as nixl_api
from nixl import _bindings as nixl_bindings
from nixl._api import nixl_agent

from tools.gemma4_pd.nixl_micro_rig.selection import RunSelection


@dataclass(frozen=True)
class ProcessAttestation:
    """One coherent process record and the maps bytes it authenticates.

    :ivar record: JSON-serializable process and native-library identity.
    :ivar proc_maps: Exact ``/proc/self/maps`` bytes hashed by ``record``.
    :ivar proc_limits: Exact ``/proc/self/limits`` bytes hashed by ``record``.
    """

    record: dict[str, object]
    proc_maps: bytes
    proc_limits: bytes


def _sha256(path: Path) -> str:
    """Hash one loaded artifact.

    :param path: Artifact path.
    :returns: Lowercase SHA-256 digest.
    """
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _loaded_native_libraries(maps_payload: bytes) -> tuple[Path, ...]:
    """Return loaded NIXL and UCX native library paths.

    :param maps_payload: One coherent ``/proc/self/maps`` snapshot.
    :returns: Sorted, unique existing paths from that snapshot.
    """
    paths: set[Path] = set()
    for line in maps_payload.decode(errors="surrogateescape").splitlines():
        fields = line.split()
        if len(fields) < 6:
            continue
        path_text = fields[-1]
        name = Path(path_text).name
        if not any(
            marker in name
            for marker in (
                "libnixl",
                "libplugin_UCX",
                "libucp",
                "libuct",
                "libucs",
                "libucm",
            )
        ):
            continue
        path = Path(path_text)
        if path.is_file():
            paths.add(path.resolve())
    return tuple(sorted(paths))


def _observed_gpu_uuid(physical_device: int) -> str:
    """Query an in-role NVIDIA UUID independently of the launcher claim.

    :param physical_device: Host NVIDIA device index.
    :returns: Observed GPU UUID.
    :raises RuntimeError: If the inventory is unavailable or ambiguous.
    """
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise RuntimeError("cannot observe in-role NVIDIA GPU UUID") from error
    matches: list[str] = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            raise RuntimeError(f"malformed in-role NVIDIA inventory row: {line}")
        if int(fields[0]) == physical_device:
            matches.append(fields[1])
    if len(matches) != 1:
        raise RuntimeError(
            f"physical GPU {physical_device} has {len(matches)} UUID observations"
        )
    return matches[0]


def _loaded_python_modules() -> list[dict[str, str]]:
    """Hash every loaded micro-rig and shared integrity Python module.

    :returns: Canonically ordered module, path, and SHA-256 records.
    """
    records: list[dict[str, str]] = []
    for name, module in sorted(sys.modules.items()):
        path_text = getattr(module, "__file__", None)
        if type(path_text) is not str:
            continue
        path = Path(path_text).resolve()
        if path.suffix == ".pyc":
            path = Path(importlib.util.source_from_cache(str(path))).resolve()
        is_rig_module = "tools/gemma4_pd/nixl_micro_rig" in path.as_posix()
        if is_rig_module is False and name not in {
            "vllm.distributed.kv_transfer.integrity",
            "vllm.distributed.kv_transfer.staging_ownership",
        }:
            continue
        if not path.is_file():
            raise RuntimeError(f"loaded Python module has no source file: {name}")
        records.append({"module": name, "path": str(path), "sha256": _sha256(path)})
    return records


def _ucx_version(libraries: tuple[Path, ...]) -> str:
    """Query the exact loaded UCP library version.

    :param libraries: Loaded native library paths.
    :returns: Version string reported by ``ucp_get_version_string``.
    :raises RuntimeError: If no loaded UCP library can be identified.
    """
    ucp_paths = [path for path in libraries if path.name.startswith("libucp")]
    if len(ucp_paths) != 1:
        raise RuntimeError(f"expected one loaded libucp, found {ucp_paths}")
    library = ctypes.CDLL(str(ucp_paths[0]))
    function = library.ucp_get_version_string
    function.argtypes = []
    function.restype = ctypes.c_char_p
    result = function()
    if result is None:
        raise RuntimeError("ucp_get_version_string returned NULL")
    return result.decode()


def _archive_native_libraries(
    libraries: tuple[Path, ...],
    *,
    run_directory: Path,
    archive_directory: Path,
) -> list[dict[str, str]]:
    """Archive exact bytes for every mapped NIXL and UCX library.

    :param libraries: Paths derived from one coherent process-maps snapshot.
    :param run_directory: Unpublished campaign root.
    :param archive_directory: Fresh role-specific library archive.
    :returns: Live path, digest, and confined archived-path records.
    """
    archive_directory.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, str]] = []
    for path in libraries:
        digest = _sha256(path)
        destination = archive_directory / f"{digest}-{path.name}"
        shutil.copyfile(path, destination)
        if _sha256(destination) != digest:
            raise RuntimeError(f"archived native library changed during copy: {path}")
        records.append(
            {
                "path": str(path),
                "sha256": digest,
                "archived_path": destination.relative_to(run_directory).as_posix(),
            }
        )
    return records


def collect_process_attestation(
    agent: nixl_agent,
    *,
    run_id: str,
    config_fingerprint: str,
    input_bundle_fingerprint: str,
    selection: RunSelection,
    role: str,
    physical_device: int,
    logical_device: int,
    configured_gpu_uuid: str,
    run_directory: Path,
    native_archive_directory: Path,
) -> ProcessAttestation:
    """Capture exact loaded artifacts and backend configuration.

    :param agent: Initialized NIXL agent in the attested process.
    :param run_id: Shared campaign UUID.
    :param config_fingerprint: SHA-256 identity of the untouched full config.
    :param input_bundle_fingerprint: SHA-256 identity of every immutable input.
    :param selection: Exact scenario and transport arm.
    :param role: Process role and rank identity.
    :param physical_device: Physical GPU selected by the role.
    :param logical_device: CUDA-visible ordinal used by NIXL descriptors.
    :param configured_gpu_uuid: Preflighted UUID selected for this logical device.
    :param run_directory: Unpublished campaign root.
    :param native_archive_directory: Fresh role-specific loaded-library archive.
    :returns: Coherent record and process-maps snapshot.
    """
    maps_payload = Path("/proc/self/maps").read_bytes()
    limits_payload = Path("/proc/self/limits").read_bytes()
    libraries = _loaded_native_libraries(maps_payload)
    library_records = _archive_native_libraries(
        libraries,
        run_directory=run_directory,
        archive_directory=native_archive_directory,
    )
    api_path = Path(nixl_api.__file__).resolve()
    bindings_path = Path(nixl_bindings.__file__).resolve()
    environment_names = {
        name
        for name in os.environ
        if name.startswith("NIXL_") or name.startswith("UCX_")
    } | {
        "CUDA_VISIBLE_DEVICES",
        "CUDA_DEVICE_ORDER",
        "LD_LIBRARY_PATH",
        "PYTHONPATH",
    }
    observed_gpu_uuid = _observed_gpu_uuid(physical_device)
    if observed_gpu_uuid != configured_gpu_uuid:
        raise RuntimeError(
            "in-role NVIDIA GPU UUID differs from the preflighted identity"
        )
    stat_text = Path("/proc/self/stat").read_text()
    command_end = stat_text.rfind(")")
    stat_fields = stat_text[command_end + 2 :].split()
    if command_end < 0 or len(stat_fields) <= 19:
        raise RuntimeError("process stat is malformed")
    record = {
        "run_id": run_id,
        "config_fingerprint": config_fingerprint,
        "input_bundle_fingerprint": input_bundle_fingerprint,
        "selection": selection.to_json(),
        "selection_fingerprint": selection.fingerprint,
        "role": role,
        "pid": os.getpid(),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "starttime_ticks": int(stat_fields[19]),
        "python_argv": list(sys.argv),
        "argv_raw_hex": Path("/proc/self/cmdline").read_bytes().hex(),
        "executable": os.readlink("/proc/self/exe"),
        "cwd": os.readlink("/proc/self/cwd"),
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "nixl_version": metadata.version("nixl"),
        "nixl_cuda_version": metadata.version("nixl-cu13"),
        "nixl_api": {"path": str(api_path), "sha256": _sha256(api_path)},
        "nixl_bindings": {
            "path": str(bindings_path),
            "sha256": _sha256(bindings_path),
        },
        "loaded_python_modules": _loaded_python_modules(),
        "loaded_native_libraries": library_records,
        "proc_maps_sha256": hashlib.sha256(maps_payload).hexdigest(),
        "proc_limits_sha256": hashlib.sha256(limits_payload).hexdigest(),
        "rlimit_nofile": list(resource.getrlimit(resource.RLIMIT_NOFILE)),
        "ucx_version": _ucx_version(libraries),
        "nixl_plugin_list": agent.get_plugin_list(),
        "nixl_ucx_plugin_params": agent.get_plugin_params("UCX"),
        "nixl_ucx_backend_params": agent.get_backend_params("UCX"),
        "nixl_ucx_memory_types": agent.get_backend_mem_types("UCX"),
        "physical_device": physical_device,
        "logical_device": logical_device,
        "configured_gpu_uuid": configured_gpu_uuid,
        "observed_gpu_uuid": observed_gpu_uuid,
        "cuda_device_name": torch.cuda.get_device_name(logical_device),
        "environment": {
            name: os.environ.get(name) for name in sorted(environment_names)
        },
    }
    normalized = json.loads(json.dumps(record, allow_nan=False))
    if not isinstance(normalized, dict):
        raise AssertionError("attestation did not normalize to a JSON object")
    return ProcessAttestation(
        record=normalized,
        proc_maps=maps_payload,
        proc_limits=limits_payload,
    )


def write_process_maps(path: Path, payload: bytes) -> None:
    """Preserve the full loaded-object map used by attestation.

    :param path: Immutable run artifact path.
    :param payload: Exact maps bytes hashed by the process attestation.
    """
    path.write_bytes(payload)


def write_process_limits(path: Path, payload: bytes) -> None:
    """Preserve the kernel limits snapshot used by attestation.

    :param path: Immutable run artifact path.
    :param payload: Exact limits bytes hashed by the process attestation.
    """
    path.write_bytes(payload)
