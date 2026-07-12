"""Loaded-process and offline artifact attestation."""

import base64
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

from tools.gemma4_cert.artifact import ArtifactWriter
from tools.gemma4_cert.common import (
    CertificationError,
    canonical_json_bytes,
    is_sensitive_name,
    sha256_file,
    utc_now,
)

_BUILD_ID = re.compile(r"Build ID:\s*([0-9a-fA-F]+)")
_ALLOWLISTED_ENVIRONMENT = {
    "VLLM_ATTENTION_BACKEND",
    "VLLM_KV_TRANSFER_CONFIG",
    "VLLM_USE_V1",
}


@dataclass(frozen=True, slots=True)
class AttestationContext:
    """Certification and process identity attached to an attestation record."""

    protocol_id: str
    run_id: str
    arm_id: str
    role: str
    engine_boot_uuid: str
    rank: int | None
    gpu_uuid: str | None


@dataclass(frozen=True, slots=True)
class NamedArtifact:
    """A file whose content participates in the deployed system identity."""

    kind: str
    path: Path


@dataclass(frozen=True, slots=True)
class RuntimeEvidence:
    """Runtime facts grouped under an explicit evidence provenance."""

    effective_config: Mapping[str, object] | None
    kv_specs: Mapping[str, object] | None
    feature_counters: Mapping[str, int] | None
    semantic_invariants: Mapping[str, object] | None
    graph_shapes: Sequence[object] | None


@dataclass(frozen=True, slots=True)
class AttestationInputs:
    """Observed and caller-supplied runtime evidence kept strictly separate."""

    observed: RuntimeEvidence
    caller_supplied: RuntimeEvidence
    software_versions: Mapping[str, str]


def build_attestation(
    context: AttestationContext,
    repository: Path,
    inputs: AttestationInputs,
    *,
    module_names: Iterable[str] | None = None,
    native_libraries: Iterable[Path] = (),
    named_artifacts: Iterable[NamedArtifact] = (),
    include_mapped_native_libraries: bool = False,
) -> dict[str, object]:
    """Build a machine-readable attestation for the calling process."""
    process = _process_identity()
    modules = collect_loaded_modules(module_names)
    native_paths = set(path.resolve() for path in native_libraries)
    if include_mapped_native_libraries:
        native_paths.update(collect_mapped_native_paths())
    native_records = [native_artifact(path) for path in sorted(native_paths)]
    artifacts = [
        {"kind": artifact.kind, **file_artifact(artifact.path)}
        for artifact in named_artifacts
    ]
    record = {
        "schema_version": 1,
        "emitted_at_utc": utc_now(),
        "context": {
            "protocol_id": context.protocol_id,
            "run_id": context.run_id,
            "arm_id": context.arm_id,
            "role": context.role,
            "engine_boot_uuid": context.engine_boot_uuid,
            "rank": context.rank,
            "gpu_uuid": context.gpu_uuid,
        },
        "process": process,
        "allowlisted_environment": collect_allowlisted_environment(),
        "repository": repository_state(repository),
        "loaded_modules": modules,
        "native_libraries": native_records,
        "named_artifacts": artifacts,
        "software_versions": {
            "observed": default_software_versions(),
            "caller_supplied": _redact_mapping(inputs.software_versions),
        },
        "runtime_evidence": {
            "observed": _runtime_evidence_mapping(inputs.observed),
            "caller_supplied": _runtime_evidence_mapping(inputs.caller_supplied),
        },
    }
    canonical_json_bytes(record)
    return record


def write_attestation(
    writer: ArtifactWriter,
    relative: str,
    context: AttestationContext,
    repository: Path,
    inputs: AttestationInputs,
    *,
    module_names: Iterable[str] | None = None,
    native_libraries: Iterable[Path] = (),
    named_artifacts: Iterable[NamedArtifact] = (),
    include_mapped_native_libraries: bool = False,
) -> dict[str, object]:
    """Build and exclusively write an attestation through an active arm writer."""
    if not isinstance(writer, ArtifactWriter):
        raise CertificationError(
            "cert-bound attestation requires an active artifact writer session"
        )
    record = build_attestation(
        context,
        repository,
        inputs,
        module_names=module_names,
        native_libraries=native_libraries,
        named_artifacts=named_artifacts,
        include_mapped_native_libraries=include_mapped_native_libraries,
    )
    writer.write_json(relative, record)
    return record


def repository_state(repository: Path) -> dict[str, object]:
    """Attest a Git commit and all tracked and untracked working-tree changes."""
    root = repository.resolve()
    commit = _git(root, "rev-parse", "HEAD").decode("ascii").strip()
    diff = _git(root, "diff", "--binary", "--no-ext-diff", "HEAD", "--")
    untracked_raw = _git(root, "ls-files", "--others", "--exclude-standard", "-z")
    untracked_paths = sorted(
        os.fsdecode(value) for value in untracked_raw.split(b"\0") if len(value) > 0
    )
    digest = hashlib.sha256()
    digest.update(b"gemma4-cert-dirty-v1\0")
    digest.update(len(diff).to_bytes(8, "big"))
    digest.update(diff)
    untracked: list[dict[str, object]] = []
    for relative in untracked_paths:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise CertificationError(
                f"untracked path is not a regular file: {relative}"
            )
        file_digest = sha256_file(path)
        stat = path.stat()
        encoded_path = os.fsencode(relative)
        digest.update(len(encoded_path).to_bytes(8, "big"))
        digest.update(encoded_path)
        digest.update(stat.st_mode.to_bytes(8, "big"))
        digest.update(stat.st_size.to_bytes(8, "big"))
        digest.update(bytes.fromhex(file_digest))
        untracked.append(
            {
                "path": relative,
                "mode": stat.st_mode,
                "size": stat.st_size,
                "sha256": file_digest,
            }
        )
    status = _git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    return {
        "path": str(root),
        "commit": commit,
        "dirty": len(status) > 0,
        "dirty_diff_sha256": digest.hexdigest(),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "status_porcelain_v1_z_base64": base64.b64encode(status).decode("ascii"),
        "untracked_files": untracked,
    }


def collect_loaded_modules(
    module_names: Iterable[str] | None,
) -> list[dict[str, object]]:
    """Hash loaded Python module files by their actual import paths."""
    if module_names is None:
        selected = sorted(sys.modules.items())
    else:
        selected = []
        for name in sorted(set(module_names)):
            module = sys.modules.get(name)
            if module is None:
                raise CertificationError(f"module is not loaded: {name}")
            selected.append((name, module))
    records = []
    for name, module in selected:
        path = _module_path(module)
        if path is None:
            continue
        records.append({"module": name, **file_artifact(path)})
    return records


def collect_mapped_native_paths() -> set[Path]:
    """Return native shared objects mapped into the calling process."""
    maps_path = Path(f"/proc/{os.getpid()}/maps")
    if not maps_path.is_file():
        raise CertificationError("mapped native library discovery requires /proc")
    paths: set[Path] = set()
    with maps_path.open("r", encoding="utf-8") as source:
        for line in source:
            fields = line.rstrip("\n").split(maxsplit=5)
            if len(fields) != 6 or not fields[5].startswith("/"):
                continue
            candidate = Path(fields[5])
            if ".so" in candidate.name and candidate.is_file():
                paths.add(candidate.resolve())
    return paths


def file_artifact(path: Path) -> dict[str, object]:
    """Describe and hash one regular artifact without following a final symlink."""
    resolved = path.resolve(strict=True)
    if path.is_symlink() or not resolved.is_file():
        raise CertificationError(f"attested path is not a regular file: {path}")
    stat = resolved.stat()
    return {
        "path": str(path),
        "resolved_path": str(resolved),
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(resolved),
    }


def native_artifact(path: Path) -> dict[str, object]:
    """Describe an ELF binary and require its GNU build ID."""
    record = file_artifact(path)
    readelf = shutil.which("readelf")
    if readelf is None:
        raise CertificationError("readelf is required to attest native build IDs")
    result = subprocess.run(
        [readelf, "-n", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise CertificationError(f"readelf failed for {path}: {result.stderr.strip()}")
    match = _BUILD_ID.search(result.stdout)
    if match is None:
        raise CertificationError(f"native artifact has no GNU build ID: {path}")
    return {**record, "gnu_build_id": match.group(1).lower()}


def default_software_versions() -> dict[str, str]:
    """Collect stable host and installed-package version identifiers."""
    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for distribution in (
        "vllm",
        "torch",
        "nixl",
        "ucx-py",
        "cuda-python",
        "nvidia-cuda-runtime-cu12",
        "nvidia-cuda-runtime-cu13",
        "nvidia-nccl-cu12",
        "nvidia-nccl-cu13",
    ):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    for variable in ("CUDA_VERSION", "NVIDIA_DRIVER_CAPABILITIES"):
        value = os.environ.get(variable)
        if value is not None:
            versions[f"env:{variable}"] = value
    driver_version = Path("/proc/driver/nvidia/version")
    if driver_version.is_file():
        versions["nvidia_driver"] = driver_version.read_text(encoding="utf-8").strip()
    return versions


def collect_allowlisted_environment() -> dict[str, object]:
    """Record only declared role/configuration flags, redacting sensitive names."""
    result: dict[str, object] = {}
    for name, value in sorted(os.environ.items()):
        if name not in _ALLOWLISTED_ENVIRONMENT and not name.startswith("VLLM_GEMMA4_"):
            continue
        result[name] = _environment_value(name, value)
    return result


def _process_identity() -> dict[str, object]:
    pid = os.getpid()
    start_time = _linux_process_start_time(pid)
    boot_id_path = Path("/proc/sys/kernel/random/boot_id")
    boot_id = (
        boot_id_path.read_text(encoding="ascii").strip()
        if boot_id_path.is_file()
        else None
    )
    return {
        "pid": pid,
        "start_time_utc": start_time,
        "os_boot_uuid": boot_id,
        "executable": sys.executable,
        "argv_recorded": False,
    }


def _linux_process_start_time(pid: int) -> str | None:
    stat_path = Path(f"/proc/{pid}/stat")
    system_stat = Path("/proc/stat")
    if not stat_path.is_file() or not system_stat.is_file():
        return None
    stat_text = stat_path.read_text(encoding="ascii")
    closing_parenthesis = stat_text.rfind(")")
    if closing_parenthesis < 0:
        raise CertificationError("malformed process stat record")
    fields = stat_text[closing_parenthesis + 2 :].split()
    if len(fields) <= 19:
        raise CertificationError("process stat record lacks start time")
    start_ticks = int(fields[19])
    boot_seconds = None
    with system_stat.open("r", encoding="ascii") as source:
        for line in source:
            if line.startswith("btime "):
                boot_seconds = int(line.split()[1])
                break
    if boot_seconds is None:
        raise CertificationError("system stat record lacks boot time")
    ticks_per_second = os.sysconf("SC_CLK_TCK")
    start_seconds = boot_seconds + start_ticks / ticks_per_second
    return (
        datetime.fromtimestamp(start_seconds, UTC)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _module_path(module: ModuleType) -> Path | None:
    value = getattr(module, "__file__", None)
    if not isinstance(value, str):
        return None
    path = Path(value)
    if path.is_symlink() or not path.is_file():
        return None
    return path


def _runtime_evidence_mapping(evidence: RuntimeEvidence) -> dict[str, object]:
    return {
        "effective_config": _available_mapping(evidence.effective_config),
        "kv_specs": _available_mapping(evidence.kv_specs),
        "feature_counters": _available_mapping(evidence.feature_counters),
        "semantic_invariants": _available_mapping(evidence.semantic_invariants),
        "graph_shapes": _available_sequence(evidence.graph_shapes),
    }


def _available_mapping(value: Mapping[str, object] | None) -> dict[str, object]:
    if value is None:
        return {"availability": "unavailable", "value": None}
    return {"availability": "available", "value": _redact_mapping(value)}


def _available_sequence(value: Sequence[object] | None) -> dict[str, object]:
    if value is None:
        return {"availability": "unavailable", "value": None}
    return {"availability": "available", "value": _redact_value(list(value), None)}


def _redact_mapping(value: Mapping[str, object]) -> dict[str, object]:
    lowered = {key.lower(): item for key, item in value.items()}
    declared_name = lowered.get("name")
    declared_key = lowered.get("key")
    redacts_value = (
        isinstance(declared_name, str) and is_sensitive_name(declared_name)
    ) or (isinstance(declared_key, str) and is_sensitive_name(declared_key))
    return {
        key: "<redacted>"
        if key.lower() == "value" and redacts_value
        else _redact_value(item, key)
        for key, item in value.items()
    }


def _redact_value(value: object, key: str | None) -> object:
    if key is not None and is_sensitive_name(key):
        return "<redacted>"
    if isinstance(value, Mapping):
        return _redact_mapping(value)
    if isinstance(value, list):
        return [_redact_value(item, None) for item in value]
    if isinstance(value, tuple):
        return [_redact_value(item, None) for item in value]
    if isinstance(value, str) and value.lstrip().lower().startswith("bearer "):
        return "<redacted>"
    return value


def _environment_value(name: str, value: str) -> dict[str, object]:
    if is_sensitive_name(name):
        return {"redacted": True, "value": "<redacted>"}
    parsed: object = value
    if value.lstrip().startswith(("{", "[")):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = value
    return {"redacted": False, "value": _redact_value(parsed, None)}


def _git(repository: Path, *arguments: str) -> bytes:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as error:
        stderr = error.stderr.decode("utf-8", errors="replace").strip()
        raise CertificationError(
            f"git {' '.join(arguments)} failed: {stderr}"
        ) from error
    return result.stdout
