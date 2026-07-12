"""Loaded-process and NIXL backend attestation for the micro-rig."""

import ctypes
import hashlib
import os
import platform
import sys
from importlib import metadata
from pathlib import Path

import torch
from nixl import _api as nixl_api
from nixl import _bindings as nixl_bindings
from nixl._api import nixl_agent


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


def _loaded_native_libraries() -> tuple[Path, ...]:
    """Return loaded NIXL and UCX native library paths.

    :returns: Sorted, unique existing paths from ``/proc/self/maps``.
    """
    maps_path = Path("/proc/self/maps")
    paths: set[Path] = set()
    for line in maps_path.read_text().splitlines():
        fields = line.split()
        if len(fields) < 6:
            continue
        path_text = fields[-1]
        name = Path(path_text).name
        if not any(
            marker in name
            for marker in ("libnixl", "libucp", "libuct", "libucs", "libucm")
        ):
            continue
        path = Path(path_text)
        if path.is_file():
            paths.add(path.resolve())
    return tuple(sorted(paths))


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


def collect_process_attestation(
    agent: nixl_agent,
    *,
    role: str,
    physical_device: int,
    logical_device: int,
) -> dict[str, object]:
    """Capture exact loaded artifacts and backend configuration.

    :param agent: Initialized NIXL agent in the attested process.
    :param role: Process role and rank identity.
    :param physical_device: Physical GPU selected by the role.
    :param logical_device: CUDA-visible ordinal used by NIXL descriptors.
    :returns: JSON-serializable attestation.
    """
    libraries = _loaded_native_libraries()
    library_records = [
        {"path": str(path), "sha256": _sha256(path)} for path in libraries
    ]
    api_path = Path(nixl_api.__file__).resolve()
    bindings_path = Path(nixl_bindings.__file__).resolve()
    environment_names = (
        "CUDA_VISIBLE_DEVICES",
        "UCX_TLS",
        "UCX_MEMTYPE_CACHE",
        "UCX_RNDV_SCHEME",
        "UCX_PROTO_INFO",
    )
    return {
        "role": role,
        "pid": os.getpid(),
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
        "loaded_native_libraries": library_records,
        "ucx_version": _ucx_version(libraries),
        "nixl_plugin_list": agent.get_plugin_list(),
        "nixl_ucx_plugin_params": agent.get_plugin_params("UCX"),
        "nixl_ucx_backend_params": agent.get_backend_params("UCX"),
        "nixl_ucx_memory_types": agent.get_backend_mem_types("UCX"),
        "physical_device": physical_device,
        "logical_device": logical_device,
        "cuda_device_name": torch.cuda.get_device_name(logical_device),
        "environment": {name: os.environ.get(name) for name in environment_names},
    }
