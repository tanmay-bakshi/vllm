"""Cross-artifact validation for sealed NIXL micro-rig campaigns."""

import base64
import hashlib
import json
import math
import os
import tarfile
from dataclasses import dataclass
from pathlib import Path

from tools.gemma4_pd.nixl_micro_rig.config import RigConfig, load_config
from tools.gemma4_pd.nixl_micro_rig.geometry import (
    TransferPlan,
    build_configured_plan,
    describe_plan,
)
from tools.gemma4_pd.nixl_micro_rig.selection import RunSelection
from tools.gemma4_pd.nixl_micro_rig.semantic_contract import (
    compute_rig_semantic_contract_digest,
)

_DISPOSITIONS = frozenset({"PASS", "FAIL", "INVALID"})
_STAGING_GUARD_BYTES = 1024 * 1024


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"artifact JSON duplicates key {key!r}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"artifact JSON contains non-finite constant {value}")


def _strict_json_loads(payload: str) -> object:
    return json.loads(
        payload,
        object_pairs_hook=_strict_json_object,
        parse_constant=_reject_json_constant,
    )


@dataclass(frozen=True)
class CampaignValidation:
    """Describe whether one campaign artifact is internally coherent.

    :ivar disposition: Terminal result whose evidence contract was validated.
    :ivar passed: Whether every required invariant passed.
    :ivar errors: Ordered validation failures.
    :ivar counts: Validated evidence cardinalities.
    """

    disposition: str
    passed: bool
    errors: tuple[str, ...]
    counts: dict[str, int]

    def to_json(self) -> dict[str, object]:
        """Return a strict JSON representation.

        :returns: JSON-serializable validation record.
        """
        return {
            "disposition": self.disposition,
            "passed": self.passed,
            "errors": list(self.errors),
            "counts": dict(self.counts),
        }


@dataclass(frozen=True)
class CampaignIdentity:
    """Shared identity required in every native artifact.

    :ivar run_id: Campaign UUID.
    :ivar config_fingerprint: SHA-256 identity of the full input config.
    :ivar input_bundle_fingerprint: SHA-256 identity of every immutable input.
    :ivar selection: Exact scenario and transport arm.
    """

    run_id: str
    config_fingerprint: str
    input_bundle_fingerprint: str
    selection: RunSelection


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _read_object(path: Path) -> dict[str, object]:
    value = _strict_json_loads(path.read_text())
    if not isinstance(value, dict) or not all(type(key) is str for key in value):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _read_array(path: Path) -> list[object]:
    value = _strict_json_loads(path.read_text())
    if not isinstance(value, list):
        raise ValueError(f"{path} must contain a JSON array")
    return value


def _exact(record: dict[str, object], field: str, expected: object) -> bool:
    actual = record.get(field)
    return type(actual) is type(expected) and actual == expected


def _confined_file(root: Path, relative_text: str) -> Path:
    relative = Path(relative_text)
    if relative.is_absolute() or ".." in relative.parts or relative == Path("."):
        raise ValueError(f"artifact path is not confined: {relative_text}")
    path = root / relative
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"artifact path traverses a symlink: {relative_text}")
    resolved_root = root.resolve()
    if not path.resolve().is_relative_to(resolved_root) or not path.is_file():
        raise ValueError(f"artifact is not a regular confined file: {relative_text}")
    return path


def validate_internal_checksums(
    run_directory: Path,
    *,
    required: bool = False,
) -> tuple[str, ...]:
    """Verify canonical SHA256SUMS membership and content.

    :param run_directory: Sealed or unpublished run directory.
    :param required: Whether a missing manifest is a validation error.
    :returns: Ordered checksum errors. An unpublished tree without a manifest
        has no checksum error yet.
    """
    checksum_path = run_directory / "SHA256SUMS"
    if not checksum_path.exists():
        return ("published campaign lacks SHA256SUMS",) if required else ()
    errors: list[str] = []
    symlinks = [
        path.relative_to(run_directory).as_posix()
        for path in run_directory.rglob("*")
        if path.is_symlink()
    ]
    if len(symlinks) > 0:
        errors.append(f"run tree contains symlinks: {sorted(symlinks)}")
    recorded: list[str] = []
    seen: set[str] = set()
    for line_number, line in enumerate(checksum_path.read_text().splitlines(), start=1):
        fields = line.split("  ", maxsplit=1)
        if len(fields) != 2:
            errors.append(f"SHA256SUMS:{line_number} is malformed")
            continue
        expected, relative = fields
        if len(expected) != 64 or any(
            character not in "0123456789abcdef" for character in expected
        ):
            errors.append(f"SHA256SUMS:{line_number} has invalid digest")
            continue
        if relative in seen:
            errors.append(f"SHA256SUMS duplicates {relative}")
            continue
        seen.add(relative)
        recorded.append(relative)
        try:
            path = _confined_file(run_directory, relative)
        except ValueError as error:
            errors.append(str(error))
            continue
        if _sha256(path) != expected:
            errors.append(f"SHA256SUMS digest differs: {relative}")
    actual = {
        path.relative_to(run_directory).as_posix()
        for path in run_directory.rglob("*")
        if path.is_file() and path != checksum_path and not path.is_symlink()
    }
    if set(recorded) != actual:
        errors.append("SHA256SUMS membership differs from run tree")
    if recorded != sorted(recorded):
        errors.append("SHA256SUMS paths are not canonically ordered")
    return tuple(errors)


def _read_selection(path: Path) -> RunSelection:
    value = _read_object(path)
    if frozenset(value) != frozenset({"schema_version", "scenario", "transport_arm"}):
        raise ValueError("selection keys differ")
    if not _exact(value, "schema_version", RunSelection.SCHEMA_VERSION):
        raise ValueError("selection schema version differs")
    scenario = value.get("scenario")
    arm = value.get("transport_arm")
    if type(scenario) is not str or type(arm) is not str:
        raise ValueError("selection names must be strings")
    return RunSelection(scenario_name=scenario, transport_arm_name=arm)


def _validate_input_provenance(
    *,
    run_directory: Path,
    config: RigConfig,
    selection: RunSelection,
    provenance: dict[str, object],
    errors: list[str],
) -> str:
    records = provenance.get("files")
    if not isinstance(records, list):
        errors.append("input provenance files must be an array")
        return ""
    expected_paths = {
        "inputs/config.json",
        "inputs/selection.json",
        f"inputs/{config.source_handshake_manifest}",
        f"inputs/{config.destination_handshake_manifest}",
    }
    expected_paths.update(
        f"inputs/{scenario.replay_manifest}"
        for scenario in config.scenarios
        if scenario.replay_manifest is not None
    )
    normalized: list[dict[str, object]] = []
    seen: set[str] = set()
    for raw_record in records:
        if not isinstance(raw_record, dict) or frozenset(raw_record) != frozenset(
            {"path", "size", "sha256"}
        ):
            errors.append("input provenance record schema differs")
            continue
        relative = raw_record.get("path")
        size = raw_record.get("size")
        expected_digest = raw_record.get("sha256")
        if type(relative) is not str or type(size) is not int:
            errors.append("input provenance record types differ")
            continue
        if type(expected_digest) is not str or len(expected_digest) != 64:
            errors.append(f"input provenance digest is invalid: {relative}")
            continue
        if relative in seen:
            errors.append(f"input provenance duplicates {relative}")
            continue
        seen.add(relative)
        try:
            path = _confined_file(run_directory, relative)
        except ValueError as error:
            errors.append(str(error))
            continue
        if path.stat().st_size != size:
            errors.append(f"input artifact size differs: {relative}")
        if _sha256(path) != expected_digest:
            errors.append(f"input artifact digest differs: {relative}")
        normalized.append({"path": relative, "size": size, "sha256": expected_digest})
    if seen != expected_paths:
        missing = sorted(expected_paths - seen)
        extra = sorted(seen - expected_paths)
        errors.append(
            f"input provenance membership differs: missing={missing} extra={extra}"
        )
    if normalized != sorted(normalized, key=lambda item: str(item["path"])):
        errors.append("input provenance records are not canonically ordered")
    bundle_payload = json.dumps(
        normalized, sort_keys=True, separators=(",", ":")
    ).encode()
    bundle_fingerprint = hashlib.sha256(bundle_payload).hexdigest()
    if provenance.get("input_bundle_fingerprint") != bundle_fingerprint:
        errors.append("input bundle fingerprint differs")
    if provenance.get("config_fingerprint") != config.fingerprint:
        errors.append("input config fingerprint differs")
    if provenance.get("selection") != selection.to_json():
        errors.append("input selection differs")
    if provenance.get("selection_fingerprint") != selection.fingerprint:
        errors.append("input selection fingerprint differs")
    return bundle_fingerprint


def _validate_source_provenance(
    run_directory: Path,
    errors: list[str],
) -> tuple[Path | None, dict[str, str]]:
    provenance = _read_object(run_directory / "provenance" / "source.json")
    if provenance.get("worktree_clean") is not True:
        errors.append("executed source worktree was not clean")
    empty_digest = hashlib.sha256(b"").hexdigest()
    if provenance.get("git_status_sha256") != empty_digest:
        errors.append("executed source Git status was not empty")
    if provenance.get("git_diff_sha256") != empty_digest:
        errors.append("executed source Git diff was not empty")
    files = provenance.get("files")
    if not isinstance(files, list) or len(files) == 0:
        errors.append("source provenance has no files")
        return None, {}
    source_tree = run_directory / "provenance" / "source-tree"
    recorded: set[str] = set()
    source_hashes: dict[str, str] = {}
    for raw_record in files:
        if not isinstance(raw_record, dict):
            errors.append("source provenance contains a non-object record")
            continue
        relative = raw_record.get("path")
        expected = raw_record.get("sha256")
        if type(relative) is not str or type(expected) is not str:
            errors.append("source provenance record has invalid fields")
            continue
        if relative in recorded:
            errors.append(f"source provenance duplicates {relative}")
            continue
        recorded.add(relative)
        source_hashes[relative] = expected
        try:
            path = _confined_file(source_tree, relative)
        except ValueError as error:
            errors.append(str(error))
            continue
        if _sha256(path) != expected:
            errors.append(f"source artifact digest differs: {relative}")
    actual = {
        path.relative_to(source_tree).as_posix()
        for path in source_tree.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    if actual != recorded:
        errors.append("source-tree membership differs from source provenance")
    archive = provenance.get("git_archive")
    if not isinstance(archive, dict):
        errors.append("source provenance lacks git archive identity")
    else:
        archive_path = archive.get("path")
        archive_sha256 = archive.get("sha256")
        if type(archive_path) is not str or type(archive_sha256) is not str:
            errors.append("git archive provenance has invalid fields")
        else:
            try:
                resolved = _confined_file(run_directory, archive_path)
            except ValueError as error:
                errors.append(str(error))
            else:
                if _sha256(resolved) != archive_sha256:
                    errors.append("git archive identity differs")
                try:
                    with tarfile.open(resolved, mode="r:") as source_archive:
                        archive_hashes: dict[str, str] = {}
                        for member in source_archive.getmembers():
                            if not member.isfile():
                                continue
                            relative = Path(member.name)
                            if relative.is_absolute() or ".." in relative.parts:
                                errors.append("git archive contains an unsafe path")
                                continue
                            extracted = source_archive.extractfile(member)
                            if extracted is None:
                                errors.append(
                                    f"git archive cannot read source file {member.name}"
                                )
                                continue
                            archive_hashes[relative.as_posix()] = hashlib.sha256(
                                extracted.read()
                            ).hexdigest()
                except (tarfile.TarError, OSError) as error:
                    errors.append(f"git archive is malformed: {error}")
                else:
                    if archive_hashes != source_hashes:
                        errors.append("git archive and copied source tree differ")
    runtime_paths = provenance.get("runtime_paths")
    if not isinstance(runtime_paths, dict):
        errors.append("source provenance lacks runtime package paths")
    else:
        if frozenset(runtime_paths) != frozenset(
            {"python", "nixl", "nixl-cu13", "vllm"}
        ):
            errors.append("runtime package path membership differs")
    repository = provenance.get("repository")
    if type(repository) is not str:
        errors.append("source provenance lacks repository path")
        return None, source_hashes
    return Path(repository), source_hashes


def _validate_protected_snapshot(
    snapshot: dict[str, object],
    label: str,
    errors: list[str],
) -> None:
    if snapshot.get("protected_devices") != [6, 7]:
        errors.append(f"{label} protected-device roster differs")
    boot_id = snapshot.get("boot_id")
    if type(boot_id) is not str or len(boot_id) == 0:
        errors.append(f"{label} lacks boot identity")
    device_uuids = snapshot.get("device_uuids")
    if not isinstance(device_uuids, dict) or frozenset(device_uuids) != frozenset(
        {"6", "7"}
    ):
        errors.append(f"{label} protected GPU UUID roster differs")
    processes = snapshot.get("processes")
    if not isinstance(processes, list):
        errors.append(f"{label} process roster is not an array")
        return
    seen: set[tuple[int, int]] = set()
    covered_devices: set[int] = set()
    for index, process in enumerate(processes):
        if not isinstance(process, dict):
            errors.append(f"{label} process {index} is not an object")
            continue
        required = {
            "pid": int,
            "starttime_ticks": int,
            "process_group": int,
            "session_id": int,
            "argv_raw_hex": str,
            "argv": list,
            "executable": str,
            "process_name": str,
            "devices": list,
        }
        for field, expected_type in required.items():
            if type(process.get(field)) is not expected_type:
                errors.append(f"{label} process {index} field {field} differs")
        pid = process.get("pid")
        starttime = process.get("starttime_ticks")
        if type(pid) is int and type(starttime) is int:
            identity = (pid, starttime)
            if identity in seen:
                errors.append(f"{label} duplicates process identity {identity}")
            seen.add(identity)
        devices = process.get("devices")
        if isinstance(devices, list) and any(
            type(device) is not int or device not in {6, 7} for device in devices
        ):
            errors.append(f"{label} process {index} device roster differs")
        elif isinstance(devices, list):
            covered_devices.update(devices)
        raw_hex = process.get("argv_raw_hex")
        if type(raw_hex) is str:
            try:
                raw_argv = bytes.fromhex(raw_hex)
            except ValueError:
                errors.append(f"{label} process {index} raw argv is not hex")
            else:
                if len(raw_argv) == 0:
                    errors.append(f"{label} process {index} raw argv is empty")
                elif process.get("argv") != [
                    os.fsdecode(argument)
                    for argument in raw_argv.rstrip(b"\0").split(b"\0")
                ]:
                    errors.append(f"{label} process {index} argv decoding differs")
    if covered_devices != {6, 7}:
        errors.append(f"{label} process coverage does not include GPUs 6 and 7")
    listeners = snapshot.get("listeners")
    if not isinstance(listeners, list) or len(listeners) != 3:
        errors.append(f"{label} protected listener roster differs")
        return
    listener_ports: list[int] = []
    for listener_index, listener in enumerate(listeners):
        if not isinstance(listener, dict) or set(listener) != {
            "port",
            "socket_inodes",
            "owners",
        }:
            errors.append(f"{label} listener {listener_index} schema differs")
            continue
        port = listener.get("port")
        if type(port) is not int:
            errors.append(f"{label} listener {listener_index} port differs")
            continue
        listener_ports.append(port)
        socket_inodes = listener.get("socket_inodes")
        if (
            not isinstance(socket_inodes, list)
            or len(socket_inodes) == 0
            or len(set(socket_inodes)) != len(socket_inodes)
            or any(type(inode) is not int or inode <= 0 for inode in socket_inodes)
        ):
            errors.append(f"{label} listener {port} socket identity differs")
        owners = listener.get("owners")
        if not isinstance(owners, list) or len(owners) == 0:
            errors.append(f"{label} listener {port} has no process owners")
            continue
        seen_owner_identities: set[tuple[int, int]] = set()
        for owner_index, owner in enumerate(owners):
            required_owner_fields = {
                "pid": int,
                "starttime_ticks": int,
                "process_group": int,
                "session_id": int,
                "argv_raw_hex": str,
                "argv": list,
                "executable": str,
            }
            if not isinstance(owner, dict) or set(owner) != set(required_owner_fields):
                errors.append(
                    f"{label} listener {port} owner {owner_index} schema differs"
                )
                continue
            for field, expected_type in required_owner_fields.items():
                if type(owner.get(field)) is not expected_type:
                    errors.append(
                        f"{label} listener {port} owner {owner_index} "
                        f"field {field} differs"
                    )
            pid = owner.get("pid")
            starttime = owner.get("starttime_ticks")
            if type(pid) is int and type(starttime) is int:
                owner_identity = (pid, starttime)
                if owner_identity in seen_owner_identities:
                    errors.append(f"{label} listener {port} duplicates an owner")
                seen_owner_identities.add(owner_identity)
            raw_hex = owner.get("argv_raw_hex")
            if type(raw_hex) is str:
                try:
                    raw_argv = bytes.fromhex(raw_hex)
                except ValueError:
                    errors.append(
                        f"{label} listener {port} owner {owner_index} argv is not hex"
                    )
                else:
                    if len(raw_argv) == 0:
                        errors.append(
                            f"{label} listener {port} owner {owner_index} argv is empty"
                        )
                    elif owner.get("argv") != [
                        os.fsdecode(argument)
                        for argument in raw_argv.rstrip(b"\0").split(b"\0")
                    ]:
                        errors.append(
                            f"{label} listener {port} owner {owner_index} "
                            "argv decoding differs"
                        )
    if listener_ports != [8000, 8820, 8821]:
        errors.append(f"{label} protected listener ports differ")


def _expected_environment(
    config: RigConfig,
    selection: RunSelection,
    visibility: str,
) -> dict[str, str]:
    arm = config.transport_arm(selection.transport_arm_name)
    environment = {
        "UCX_NET_DEVICES": arm.ucx_net_devices,
        "UCX_PROTO_INFO": "y",
        "UCX_RNDV_SCHEME": arm.ucx_rndv_scheme,
        "UCX_TLS": arm.ucx_tls,
    }
    if arm.ucx_memtype_cache is not None:
        environment["UCX_MEMTYPE_CACHE"] = arm.ucx_memtype_cache
    return environment


def _validate_process_attestation(
    *,
    record: object,
    identity: CampaignIdentity,
    expected_role: str,
    expected_physical_device: int,
    expected_logical_device: int,
    expected_gpu_uuid: str,
    expected_visibility: str,
    expected_ucx: dict[str, str],
    run_directory: Path,
    source_repository: Path | None,
    source_hashes: dict[str, str],
    maps_path: Path,
    limits_path: Path,
    errors: list[str],
) -> None:
    if not isinstance(record, dict):
        errors.append(f"{expected_role} attestation is not an object")
        return
    expected_fields = {
        "run_id": identity.run_id,
        "config_fingerprint": identity.config_fingerprint,
        "input_bundle_fingerprint": identity.input_bundle_fingerprint,
        "selection": identity.selection.to_json(),
        "selection_fingerprint": identity.selection.fingerprint,
        "role": expected_role,
        "physical_device": expected_physical_device,
        "logical_device": expected_logical_device,
        "configured_gpu_uuid": expected_gpu_uuid,
        "observed_gpu_uuid": expected_gpu_uuid,
    }
    for field, expected in expected_fields.items():
        if not _exact(record, field, expected):
            errors.append(f"{expected_role} attestation {field} differs")
    for field, expected_type in (
        ("pid", int),
        ("starttime_ticks", int),
        ("boot_id", str),
        ("argv_raw_hex", str),
        ("python_argv", list),
        ("executable", str),
        ("cwd", str),
    ):
        if type(record.get(field)) is not expected_type:
            errors.append(f"{expected_role} attestation {field} type differs")
    environment = record.get("environment")
    if not isinstance(environment, dict):
        errors.append(f"{expected_role} attestation lacks environment")
    else:
        if environment.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
            errors.append(f"{expected_role} CUDA device order differs")
        if environment.get("CUDA_VISIBLE_DEVICES") != expected_visibility:
            errors.append(f"{expected_role} CUDA visibility differs")
        actual_ucx = {
            name: value
            for name, value in environment.items()
            if name.startswith("UCX_") or name.startswith("NIXL_")
        }
        if actual_ucx != expected_ucx:
            errors.append(f"{expected_role} UCX environment differs")
    for field in ("nixl_version", "nixl_cuda_version", "ucx_version"):
        value = record.get(field)
        if type(value) is not str or len(value) == 0:
            errors.append(f"{expected_role} attestation lacks {field}")
    plugin_list = record.get("nixl_plugin_list")
    if not isinstance(plugin_list, list) or "UCX" not in plugin_list:
        errors.append(f"{expected_role} attestation lacks the UCX plugin")
    for field in ("nixl_ucx_plugin_params", "nixl_ucx_backend_params"):
        if not isinstance(record.get(field), dict):
            errors.append(f"{expected_role} attestation {field} differs")
    memory_types = record.get("nixl_ucx_memory_types")
    if not isinstance(memory_types, list) or "VRAM" not in memory_types:
        errors.append(f"{expected_role} attestation UCX memory types differ")
    api_records: list[tuple[str, Path]] = []
    for field in ("nixl_api", "nixl_bindings"):
        artifact = record.get(field)
        if not isinstance(artifact, dict) or set(artifact) != {"path", "sha256"}:
            errors.append(f"{expected_role} attestation {field} schema differs")
            continue
        path_text = artifact.get("path")
        digest = artifact.get("sha256")
        if type(path_text) is not str or type(digest) is not str:
            errors.append(f"{expected_role} attestation {field} types differ")
            continue
        path = Path(path_text).resolve()
        if not path.is_file() or _sha256(path) != digest:
            errors.append(f"{expected_role} attestation {field} identity differs")
            continue
        api_records.append((field, path))
    libraries = record.get("loaded_native_libraries")
    if not isinstance(libraries, list):
        errors.append(f"{expected_role} attestation lacks loaded libraries")
    else:
        if not maps_path.is_file():
            errors.append(f"{expected_role} lacks process maps")
            mapped_paths: set[Path] = set()
        else:
            mapped_paths = {
                Path(fields[-1]).resolve()
                for line in maps_path.read_text().splitlines()
                if len(fields := line.split()) >= 6 and fields[-1].startswith("/")
            }
        api_record = record.get("nixl_api")
        api_path_text = api_record.get("path") if isinstance(api_record, dict) else None
        site_packages: Path | None = None
        if type(api_path_text) is str:
            api_path = Path(api_path_text).resolve()
            site_packages = next(
                (
                    parent
                    for parent in api_path.parents
                    if parent.name == "site-packages"
                ),
                None,
            )
        if site_packages is not None:
            for field, path in api_records:
                if not path.is_relative_to(site_packages):
                    errors.append(
                        f"{expected_role} attestation {field} is outside site-packages"
                    )
        seen_library_paths: set[Path] = set()
        names: set[str] = set()
        for index, item in enumerate(libraries):
            if not isinstance(item, dict) or frozenset(item) != frozenset(
                {"path", "sha256", "archived_path"}
            ):
                errors.append(f"{expected_role} library {index} schema differs")
                continue
            path_text = item.get("path")
            digest = item.get("sha256")
            archived_path = item.get("archived_path")
            if (
                type(path_text) is not str
                or type(digest) is not str
                or type(archived_path) is not str
            ):
                errors.append(f"{expected_role} library {index} types differ")
                continue
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                errors.append(f"{expected_role} library {index} digest differs")
                continue
            path = Path(path_text).resolve()
            if path in seen_library_paths:
                errors.append(f"{expected_role} duplicates loaded library {path}")
            seen_library_paths.add(path)
            names.add(path.name)
            if path not in mapped_paths:
                errors.append(f"{expected_role} library is absent from archived maps")
            if site_packages is None or not path.is_relative_to(site_packages):
                errors.append(
                    f"{expected_role} library is outside bundled site-packages"
                )
            try:
                archived = _confined_file(run_directory, archived_path)
            except ValueError as error:
                errors.append(str(error))
            else:
                if _sha256(archived) != digest:
                    errors.append(f"{expected_role} archived library identity differs")
        for prefix in ("libnixl", "libplugin_UCX", "libucp"):
            if not any(name.startswith(prefix) for name in names):
                errors.append(f"{expected_role} lacks loaded {prefix}")
    modules = record.get("loaded_python_modules")
    if not isinstance(modules, list) or len(modules) == 0:
        errors.append(f"{expected_role} lacks loaded Python module identities")
    elif source_repository is not None:
        seen_modules: set[str] = set()
        seen_module_paths: set[str] = set()
        for module_record in modules:
            if not isinstance(module_record, dict):
                errors.append(f"{expected_role} has malformed Python module identity")
                continue
            module_name = module_record.get("module")
            path_text = module_record.get("path")
            digest = module_record.get("sha256")
            if (
                type(module_name) is not str
                or type(path_text) is not str
                or type(digest) is not str
            ):
                errors.append(f"{expected_role} has invalid Python module fields")
                continue
            if module_name in seen_modules:
                errors.append(f"{expected_role} duplicates module {module_name}")
            seen_modules.add(module_name)
            path = Path(path_text)
            try:
                relative = path.relative_to(source_repository).as_posix()
            except ValueError:
                errors.append(f"{expected_role} module is outside executed source")
                continue
            seen_module_paths.add(relative)
            if source_hashes.get(relative) != digest:
                errors.append(f"{expected_role} module {module_name} source differs")
        required_module_paths = {
            "tools/gemma4_pd/nixl_micro_rig/attestation.py",
            "tools/gemma4_pd/nixl_micro_rig/config.py",
            "tools/gemma4_pd/nixl_micro_rig/data.py",
            "tools/gemma4_pd/nixl_micro_rig/geometry.py",
            "tools/gemma4_pd/nixl_micro_rig/outcome.py",
            "tools/gemma4_pd/nixl_micro_rig/protocol.py",
            "tools/gemma4_pd/nixl_micro_rig/role_entry.py",
            "tools/gemma4_pd/nixl_micro_rig/roles.py",
            "tools/gemma4_pd/nixl_micro_rig/selection.py",
            "tools/gemma4_pd/nixl_micro_rig/semantic_contract.py",
            "vllm/distributed/kv_transfer/integrity.py",
            "vllm/distributed/kv_transfer/staging_ownership.py",
        }
        missing_modules = required_module_paths - seen_module_paths
        if len(missing_modules) > 0:
            errors.append(
                f"{expected_role} lacks executed modules {sorted(missing_modules)}"
            )
    maps_digest = record.get("proc_maps_sha256")
    if type(maps_digest) is not str or not maps_path.is_file():
        errors.append(f"{expected_role} lacks process maps")
    elif _sha256(maps_path) != maps_digest:
        errors.append(f"{expected_role} process maps digest differs")
    limits_digest = record.get("proc_limits_sha256")
    if type(limits_digest) is not str or not limits_path.is_file():
        errors.append(f"{expected_role} lacks process limits")
    elif _sha256(limits_path) != limits_digest:
        errors.append(f"{expected_role} process limits digest differs")
    if record.get("rlimit_nofile") != [65_535, 1_048_576]:
        errors.append(f"{expected_role} RLIMIT_NOFILE differs")


def _observation_map(
    *,
    path: Path,
    expected_stage: str,
    identity: CampaignIdentity,
    config: RigConfig,
    plan: TransferPlan,
    errors: list[str],
) -> dict[str, str]:
    observations: dict[str, str] = {}
    pairings = {
        (
            pairing.group_index,
            pairing.group_position,
            pairing.remote_block_id,
        ): pairing
        for pairing in plan.sorted_pairings
    }
    source_stage = expected_stage in {"source_pre", "source_post"}
    with path.open() as source:
        for line_number, line in enumerate(source, start=1):
            try:
                record = _strict_json_loads(line)
            except (json.JSONDecodeError, ValueError):
                errors.append(f"{path.name}:{line_number} is invalid JSON")
                continue
            if not isinstance(record, dict):
                errors.append(f"{path.name}:{line_number} is not an object")
                continue
            expected_record_keys = {
                "stage",
                "identity",
                "digest",
                "child_request_id",
                "observer_engine_id",
                "observer_rank",
                "local_block_id",
                "config_fingerprint",
                "input_bundle_fingerprint",
                "scenario",
                "evidence_status",
            }
            if set(record) != expected_record_keys:
                errors.append(f"{path.name}:{line_number} record schema differs")
            expected_outer = {
                "stage": expected_stage,
                "config_fingerprint": identity.config_fingerprint,
                "input_bundle_fingerprint": identity.input_bundle_fingerprint,
                "scenario": identity.selection.scenario_name,
                "evidence_status": "content_digest",
            }
            for field, expected in expected_outer.items():
                if not _exact(record, field, expected):
                    errors.append(f"{path.name}:{line_number} field {field} differs")
            integrity_identity = record.get("identity")
            digest = record.get("digest")
            if not isinstance(integrity_identity, dict) or type(digest) is not str:
                errors.append(f"{path.name}:{line_number} lacks identity or digest")
                continue
            if len(digest) != 32 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                errors.append(f"{path.name}:{line_number} digest is not BLAKE2b-128")
            expected_identity_keys = {
                "run_id",
                "transport_arm",
                "producer_engine_id",
                "producer_request_id",
                "registration_generation",
                "semantic_contract_digest",
                "offer_generation",
                "iteration",
                "source_rank",
                "region_index",
                "group_index",
                "plane_index",
                "source_position",
                "remote_block_id",
                "valid_token_extent",
                "group_token_capacity",
                "payload_kind",
                "byte_length",
            }
            if set(integrity_identity) != expected_identity_keys:
                errors.append(f"{path.name}:{line_number} identity schema differs")
                continue
            iteration = integrity_identity.get("iteration")
            source_rank = integrity_identity.get("source_rank")
            region_index = integrity_identity.get("region_index")
            group_index = integrity_identity.get("group_index")
            source_position = integrity_identity.get("source_position")
            remote_block_id = integrity_identity.get("remote_block_id")
            integer_fields = (
                iteration,
                source_rank,
                region_index,
                group_index,
                source_position,
                remote_block_id,
            )
            if any(type(value) is not int for value in integer_fields):
                errors.append(f"{path.name}:{line_number} identity types differ")
                continue
            if (
                not 0
                <= iteration
                < config.scenario(identity.selection.scenario_name).iterations
            ):
                errors.append(f"{path.name}:{line_number} iteration differs")
                continue
            if not 0 <= source_rank < plan.rank_count:
                errors.append(f"{path.name}:{line_number} source rank differs")
                continue
            if not 0 <= region_index < len(config.regions):
                errors.append(f"{path.name}:{line_number} region differs")
                continue
            pairing = pairings.get((group_index, source_position, remote_block_id))
            if pairing is None:
                errors.append(f"{path.name}:{line_number} source pairing differs")
                continue
            group = config.groups[group_index]
            region = config.regions[region_index]
            expected_identity = {
                "run_id": identity.run_id,
                "transport_arm": identity.selection.transport_arm_name,
                "producer_engine_id": f"micro-p-{identity.run_id}",
                "producer_request_id": f"p-{identity.run_id}-{iteration}",
                "registration_generation": (
                    f"{identity.run_id}:{identity.selection.transport_arm_name}:"
                    f"producer-rank-{source_rank}:registration-1"
                ),
                "semantic_contract_digest": compute_rig_semantic_contract_digest(
                    config,
                    group_index,
                    region_index,
                ).hex(),
                "offer_generation": iteration,
                "iteration": iteration,
                "source_rank": source_rank,
                "region_index": region_index,
                "group_index": group_index,
                "plane_index": -1,
                "source_position": source_position,
                "remote_block_id": remote_block_id,
                "valid_token_extent": config.valid_token_extent,
                "group_token_capacity": group.token_capacity,
                "payload_kind": "wire",
                "byte_length": region.row_bytes,
            }
            if integrity_identity != expected_identity:
                errors.append(f"{path.name}:{line_number} identity domain differs")
            expected_observation = {
                "child_request_id": (
                    None if source_stage else f"d-{identity.run_id}-{iteration}"
                ),
                "observer_engine_id": (
                    f"micro-p-{identity.run_id}"
                    if source_stage
                    else f"micro-d-{identity.run_id}"
                ),
                "observer_rank": source_rank if source_stage else 0,
                "local_block_id": None if source_stage else pairing.local_block_id,
            }
            for field, expected in expected_observation.items():
                if not _exact(record, field, expected):
                    errors.append(
                        f"{path.name}:{line_number} observation field {field} differs"
                    )
            key = json.dumps(integrity_identity, sort_keys=True, separators=(",", ":"))
            if key in observations:
                errors.append(f"{path.name}:{line_number} duplicates an identity")
                continue
            observations[key] = digest
    return observations


def _validate_role_terminals(
    *,
    arm_directory: Path,
    config: RigConfig,
    identity: CampaignIdentity,
    disposition: str,
    errors: list[str],
) -> None:
    if disposition == "PASS":
        terminal_specs = [
            (arm_directory / f"producer-{rank}-terminal.json", "producer", rank)
            for rank in range(len(config.producer_devices))
        ] + [(arm_directory / "consumer-terminal.json", "consumer", 0)]
    else:
        terminal_specs = [(arm_directory / "consumer-terminal.json", "consumer", 0)]
    statuses: list[str] = []
    for path, role, rank in terminal_specs:
        terminal = _read_object(path)
        expected = {
            "run_id": identity.run_id,
            "config_fingerprint": identity.config_fingerprint,
            "selection": identity.selection.to_json(),
            "input_bundle_fingerprint": identity.input_bundle_fingerprint,
            "role": role,
            "rank": rank,
        }
        for field, value in expected.items():
            if not _exact(terminal, field, value):
                errors.append(f"{path.name} field {field} differs")
        if set(terminal) != {
            *expected,
            "schema_version",
            "status",
            "detail",
            "traceback",
            "completed_ns",
        }:
            errors.append(f"{path.name} schema differs")
        if not _exact(terminal, "schema_version", 1):
            errors.append(f"{path.name} schema version differs")
        if type(terminal.get("detail")) is not str:
            errors.append(f"{path.name} detail differs")
        if type(terminal.get("traceback")) is not str:
            errors.append(f"{path.name} traceback differs")
        if type(terminal.get("completed_ns")) is not int:
            errors.append(f"{path.name} completion time differs")
        status = terminal.get("status")
        if type(status) is not str or status not in _DISPOSITIONS:
            errors.append(f"{path.name} has invalid status")
        else:
            statuses.append(status)
    if disposition == "PASS" and statuses != ["PASS"] * len(terminal_specs):
        errors.append("PASS campaign has a non-PASS role terminal")
    if disposition == "FAIL" and statuses != ["FAIL"]:
        errors.append("FAIL campaign lacks an authoritative consumer FAIL terminal")


def _validate_pass_staging_ownership(
    *,
    record: object,
    identity: CampaignIdentity,
    iteration: int,
    child_request_id: str,
    staging_offset: int,
    staging_bytes: int,
    rank_count: int,
    errors: list[str],
) -> None:
    """Validate a released, proved-quiescent staging generation.

    :param record: Serialized ownership snapshot.
    :param identity: Shared campaign identity.
    :param iteration: Exact allocation generation.
    :param child_request_id: Consumer request owning the range.
    :param staging_offset: Registration-relative allocation offset.
    :param staging_bytes: Exact allocation size.
    :param rank_count: Complete native writer count.
    :param errors: Validation error accumulator.
    """
    if not isinstance(record, dict):
        errors.append(f"iteration {iteration} lacks staging ownership evidence")
        return
    expected_keys = {
        "generation",
        "owner_id",
        "request_id",
        "remote_engine_id",
        "offset",
        "size",
        "age_s",
        "handles",
        "posting_sealed",
        "operation_failed",
        "permanently_tombstoned",
        "native_quiescent",
        "device_read_started",
        "device_quiescent",
        "reusable",
        "released",
        "failure_reason",
    }
    if set(record) != expected_keys:
        errors.append(f"iteration {iteration} staging ownership schema differs")
    expected_fields = {
        "generation": iteration,
        "owner_id": f"rig:{iteration}",
        "request_id": child_request_id,
        "remote_engine_id": f"micro-p-{identity.run_id}",
        "offset": staging_offset,
        "size": staging_bytes,
        "posting_sealed": True,
        "operation_failed": False,
        "permanently_tombstoned": False,
        "native_quiescent": True,
        "device_read_started": True,
        "device_quiescent": True,
        "reusable": True,
        "released": True,
        "failure_reason": None,
    }
    for field, expected in expected_fields.items():
        if not _exact(record, field, expected):
            errors.append(
                f"iteration {iteration} staging ownership field {field} differs"
            )
    age_s = record.get("age_s")
    if type(age_s) is not float or not math.isfinite(age_s) or age_s < 0:
        errors.append(f"iteration {iteration} staging ownership age differs")
    handles = record.get("handles")
    if not isinstance(handles, list) or len(handles) != rank_count:
        errors.append(f"iteration {iteration} staging ownership handles differ")
        return
    expected_handle_keys = {
        "source_rank",
        "state",
        "native_handle_present",
        "native_released",
    }
    for rank, handle in enumerate(handles):
        expected_handle = {
            "source_rank": rank,
            "state": "done",
            "native_handle_present": True,
            "native_released": True,
        }
        if (
            not isinstance(handle, dict)
            or set(handle) != expected_handle_keys
            or handle != expected_handle
        ):
            errors.append(
                f"iteration {iteration} staging ownership handle {rank} differs"
            )


def _validate_fail_arm(
    *,
    run_directory: Path,
    config: RigConfig,
    identity: CampaignIdentity,
    preflight: dict[str, object],
    errors: list[str],
) -> None:
    """Validate common supervision evidence for a typed correctness failure.

    :param run_directory: Unpublished campaign root.
    :param config: Complete rig configuration.
    :param identity: Shared campaign identity.
    :param preflight: Selected-device preflight record.
    :param errors: Validation error accumulator.
    """
    selection = identity.selection
    arm_directory = (
        run_directory / "arms" / selection.transport_arm_name / selection.scenario_name
    )
    _validate_role_terminals(
        arm_directory=arm_directory,
        config=config,
        identity=identity,
        disposition="FAIL",
        errors=errors,
    )
    expected_names = {
        *(f"producer-{rank}" for rank in range(len(config.producer_devices))),
        "consumer",
    }
    shared = {
        "run_id": identity.run_id,
        "config_fingerprint": identity.config_fingerprint,
        "input_bundle_fingerprint": identity.input_bundle_fingerprint,
        "selection": selection.to_json(),
    }
    launches = _read_array(arm_directory / "process-launches.json")
    launch_names: set[str] = set()
    for index, launch in enumerate(launches):
        if not isinstance(launch, dict):
            errors.append(f"FAIL process launch {index} is not an object")
            continue
        for field, expected in shared.items():
            if not _exact(launch, field, expected):
                errors.append(f"FAIL process launch {index} field {field} differs")
        name = launch.get("name")
        if type(name) is not str or name in launch_names:
            errors.append(f"FAIL process launch {index} name differs")
        else:
            launch_names.add(name)
        if (
            type(launch.get("pid")) is not int
            or type(launch.get("starttime_ticks")) is not int
            or not isinstance(launch.get("argv"), list)
            or not isinstance(launch.get("environment"), dict)
        ):
            errors.append(f"FAIL process launch {index} identity differs")
    if launch_names != expected_names:
        errors.append("FAIL process launch membership differs")
    exits = _read_object(arm_directory / "process-exits.json")
    cleanup = _read_object(arm_directory / "cleanup.json")
    for label, record in (("exits", exits), ("cleanup", cleanup)):
        for field, expected in shared.items():
            if not _exact(record, field, expected):
                errors.append(f"FAIL process {label} field {field} differs")
    exit_codes = exits.get("exit_codes")
    if (
        not isinstance(exit_codes, dict)
        or set(exit_codes) != expected_names
        or any(type(value) is not int for value in exit_codes.values())
        or exit_codes.get("consumer") == 0
    ):
        errors.append("FAIL process exit evidence differs")
    if cleanup.get("errors") != []:
        errors.append("FAIL campaign has cleanup errors")
    expected_selected = sorted((*config.producer_devices, config.consumer_device))
    raw_uuids = preflight.get("selected_device_uuids")
    postflight = _read_object(run_directory / "selected-gpus-after.json")
    for field, expected in shared.items():
        if not _exact(preflight, field, expected):
            errors.append(f"FAIL preflight field {field} differs")
        if not _exact(postflight, field, expected):
            errors.append(f"FAIL selected GPU postflight field {field} differs")
    if preflight.get("selected_devices") != expected_selected:
        errors.append("FAIL preflight selected-device roster differs")
    if postflight.get("selected_devices") != expected_selected:
        errors.append("FAIL selected GPU postflight device roster differs")
    if postflight.get("selected_device_uuids") != raw_uuids:
        errors.append("FAIL selected GPU postflight UUID identity differs")
    if postflight.get("processes") != []:
        errors.append("FAIL selected GPU postflight retains role processes")


def _validate_pass_arm(
    *,
    run_directory: Path,
    config_path: Path,
    config: RigConfig,
    identity: CampaignIdentity,
    preflight: dict[str, object],
    source_repository: Path | None,
    source_hashes: dict[str, str],
    errors: list[str],
    counts: dict[str, int],
) -> None:
    selection = identity.selection
    scenario = config.scenario(selection.scenario_name)
    plan = build_configured_plan(config_path, config, scenario)
    expected_plan = describe_plan(config, scenario, plan)
    expected_plan_artifact = {
        "run_id": identity.run_id,
        "config_fingerprint": identity.config_fingerprint,
        "input_bundle_fingerprint": identity.input_bundle_fingerprint,
        "selection": selection.to_json(),
        "plan": expected_plan,
    }
    if _read_object(run_directory / "plan.json") != expected_plan_artifact:
        errors.append("published plan differs from selected configuration")
    arm_directory = (
        run_directory / "arms" / selection.transport_arm_name / selection.scenario_name
    )
    _validate_role_terminals(
        arm_directory=arm_directory,
        config=config,
        identity=identity,
        disposition="PASS",
        errors=errors,
    )
    launch_records = _read_array(arm_directory / "process-launches.json")
    expected_process_count = len(config.producer_devices) + 1
    if len(launch_records) != expected_process_count:
        errors.append("process launch count differs")
    launch_by_name: dict[str, dict[str, object]] = {}
    for index, record in enumerate(launch_records):
        if not isinstance(record, dict):
            errors.append(f"process launch {index} is not an object")
            continue
        launch_identity = {
            "run_id": identity.run_id,
            "config_fingerprint": identity.config_fingerprint,
            "input_bundle_fingerprint": identity.input_bundle_fingerprint,
            "selection": selection.to_json(),
        }
        for field, expected in launch_identity.items():
            if not _exact(record, field, expected):
                errors.append(f"process launch {index} field {field} differs")
        if set(record) != {
            *launch_identity,
            "name",
            "role",
            "rank",
            "pid",
            "starttime_ticks",
            "argv",
            "environment",
        }:
            errors.append(f"process launch {index} schema differs")
        if (
            type(record.get("pid")) is not int
            or type(record.get("starttime_ticks")) is not int
            or not isinstance(record.get("argv"), list)
            or not isinstance(record.get("environment"), dict)
        ):
            errors.append(f"process launch {index} process identity differs")
        name = record.get("name")
        if type(name) is not str or name in launch_by_name:
            errors.append(f"process launch {index} name differs")
        else:
            launch_by_name[name] = record
    expected_launch_names = {
        f"producer-{rank}" for rank in range(len(config.producer_devices))
    }
    expected_launch_names.add("consumer")
    if set(launch_by_name) != expected_launch_names:
        errors.append("process launch name membership differs")
    exits = _read_object(arm_directory / "process-exits.json")
    exit_identity = {
        "run_id": identity.run_id,
        "config_fingerprint": identity.config_fingerprint,
        "input_bundle_fingerprint": identity.input_bundle_fingerprint,
        "selection": selection.to_json(),
    }
    for field, expected in exit_identity.items():
        if not _exact(exits, field, expected):
            errors.append(f"process exits field {field} differs")
    exit_codes = exits.get("exit_codes")
    if not isinstance(exit_codes, dict) or set(exit_codes) != expected_launch_names:
        errors.append("process exit-code membership differs")
    elif any(type(code) is not int or code != 0 for code in exit_codes.values()):
        errors.append("PASS campaign has nonzero process exit")
    cleanup = _read_object(arm_directory / "cleanup.json")
    for field, expected in exit_identity.items():
        if not _exact(cleanup, field, expected):
            errors.append(f"cleanup field {field} differs")
    if set(cleanup) != {*exit_identity, "errors"}:
        errors.append("cleanup schema differs")
    if cleanup.get("errors") != []:
        errors.append("PASS campaign has cleanup errors")
    expected_selected = sorted((*config.producer_devices, config.consumer_device))
    preflight_identity = {
        "run_id": identity.run_id,
        "config_fingerprint": identity.config_fingerprint,
        "input_bundle_fingerprint": identity.input_bundle_fingerprint,
        "selection": selection.to_json(),
    }
    for field, expected in preflight_identity.items():
        if not _exact(preflight, field, expected):
            errors.append(f"preflight field {field} differs")
    if preflight.get("selected_devices") != expected_selected:
        errors.append("preflight selected-device roster differs")
    if preflight.get("immutable_denied_devices") != [6, 7]:
        errors.append("preflight protected-device roster differs")
    raw_uuids = preflight.get("selected_device_uuids")
    if not isinstance(raw_uuids, dict):
        errors.append("preflight lacks selected device UUIDs")
        return
    device_uuids = {int(device): str(value) for device, value in raw_uuids.items()}
    if set(device_uuids) != set(expected_selected):
        errors.append("preflight selected GPU UUID membership differs")
        return
    postflight = _read_object(run_directory / "selected-gpus-after.json")
    for field, expected in preflight_identity.items():
        if not _exact(postflight, field, expected):
            errors.append(f"selected GPU postflight field {field} differs")
    if set(postflight) != {
        *preflight_identity,
        "selected_devices",
        "selected_device_uuids",
        "processes",
    }:
        errors.append("selected GPU postflight schema differs")
    if postflight.get("selected_devices") != expected_selected:
        errors.append("selected GPU postflight device roster differs")
    if postflight.get("selected_device_uuids") != raw_uuids:
        errors.append("selected GPU postflight UUID identity differs")
    if postflight.get("processes") != []:
        errors.append("selected GPU postflight retains role processes")
    producer_visibility = ",".join(
        device_uuids[device] for device in config.producer_devices
    )
    producer_records: list[dict[str, object]] = []
    expected_ucx = _expected_environment(config, selection, producer_visibility)
    for rank, physical_device in enumerate(config.producer_devices):
        path = arm_directory / f"producer-{rank}-attestation.json"
        record = _read_object(path)
        producer_records.append(record)
        _validate_process_attestation(
            record=record,
            identity=identity,
            expected_role=f"producer:{rank}",
            expected_physical_device=physical_device,
            expected_logical_device=rank,
            expected_gpu_uuid=device_uuids[physical_device],
            expected_visibility=producer_visibility,
            expected_ucx=expected_ucx,
            run_directory=run_directory,
            source_repository=source_repository,
            source_hashes=source_hashes,
            maps_path=arm_directory / f"producer-{rank}-proc-maps.txt",
            limits_path=arm_directory / f"producer-{rank}-proc-limits.txt",
            errors=errors,
        )
        launch = launch_by_name.get(f"producer-{rank}")
        if launch is not None:
            expected_launch_environment = {
                **expected_ucx,
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": producer_visibility,
            }
            if launch.get("environment") != expected_launch_environment:
                errors.append(f"producer:{rank} launch environment differs")
            if launch.get("role") != "producer" or launch.get("rank") != rank:
                errors.append(f"producer:{rank} launch role identity differs")
            launch_argv = launch.get("argv")
            expected_raw = (
                b"\0".join(
                    str(argument).encode(errors="surrogateescape")
                    for argument in launch_argv
                )
                + b"\0"
                if isinstance(launch_argv, list)
                else b""
            )
            if (
                record.get("pid") != launch.get("pid")
                or record.get("starttime_ticks") != launch.get("starttime_ticks")
                or record.get("argv_raw_hex") != expected_raw.hex()
            ):
                errors.append(f"producer:{rank} launch and process identity differ")
    consumer_root = _read_object(arm_directory / "consumer-attestation.json")
    consumer_record = consumer_root.get("consumer")
    consumer_visibility = device_uuids[config.consumer_device]
    _validate_process_attestation(
        record=consumer_record,
        identity=identity,
        expected_role="consumer:0",
        expected_physical_device=config.consumer_device,
        expected_logical_device=0,
        expected_gpu_uuid=consumer_visibility,
        expected_visibility=consumer_visibility,
        expected_ucx=_expected_environment(config, selection, consumer_visibility),
        run_directory=run_directory,
        source_repository=source_repository,
        source_hashes=source_hashes,
        maps_path=arm_directory / "consumer-proc-maps.txt",
        limits_path=arm_directory / "consumer-proc-limits.txt",
        errors=errors,
    )
    consumer_launch = launch_by_name.get("consumer")
    if isinstance(consumer_record, dict) and consumer_launch is not None:
        expected_consumer_environment = {
            **_expected_environment(config, selection, consumer_visibility),
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": consumer_visibility,
        }
        if consumer_launch.get("environment") != expected_consumer_environment:
            errors.append("consumer launch environment differs")
        if (
            consumer_launch.get("role") != "consumer"
            or consumer_launch.get("rank") != 0
        ):
            errors.append("consumer launch role identity differs")
        launch_argv = consumer_launch.get("argv")
        expected_raw = (
            b"\0".join(
                str(argument).encode(errors="surrogateescape")
                for argument in launch_argv
            )
            + b"\0"
            if isinstance(launch_argv, list)
            else b""
        )
        if (
            consumer_record.get("pid") != consumer_launch.get("pid")
            or consumer_record.get("starttime_ticks")
            != consumer_launch.get("starttime_ticks")
            or consumer_record.get("argv_raw_hex") != expected_raw.hex()
        ):
            errors.append("consumer launch and process identity differ")
    if consumer_root.get("producers") != producer_records:
        errors.append("consumer and producer attestations differ")
    registrations = _read_object(arm_directory / "runtime-registrations.json")
    registration_identity = {
        "run_id": identity.run_id,
        "transport_arm": selection.transport_arm_name,
        "scenario": selection.scenario_name,
        "selection_fingerprint": selection.fingerprint,
        "config_fingerprint": identity.config_fingerprint,
        "input_bundle_fingerprint": identity.input_bundle_fingerprint,
    }
    for field, expected in registration_identity.items():
        if not _exact(registrations, field, expected):
            errors.append(f"runtime registrations field {field} differs")
    destinations = registrations.get("destination")
    producers = registrations.get("producers")
    if not isinstance(destinations, list) or len(destinations) != len(config.regions):
        errors.append("runtime destination registration count differs")
    if not isinstance(producers, list) or len(producers) != plan.rank_count:
        errors.append("runtime producer registration count differs")
    staging = registrations.get("staging")
    if not isinstance(staging, dict) or staging.get("byte_length") != (
        config.staging_capacity_mib * 1024 * 1024
    ):
        errors.append("runtime staging registration differs")

    iterations = _read_array(arm_directory / "iterations.json")
    if len(iterations) != scenario.iterations:
        errors.append(
            f"iteration count is {len(iterations)}, expected {scenario.iterations}"
        )
    expected_snapshot_observations = plan.position_count * len(config.regions)
    expected_consumer_observations = expected_snapshot_observations * plan.rank_count
    expected_bytes = plan.position_count * sum(
        region.row_bytes for region in config.regions
    )
    for iteration, raw_record in enumerate(iterations):
        if not isinstance(raw_record, dict):
            errors.append(f"iteration {iteration} is not an object")
            continue
        producer_request_id = f"p-{identity.run_id}-{iteration}"
        child_request_id = f"d-{identity.run_id}-{iteration}"
        notification_id = base64.b64encode(
            (
                f"micro-rig:{identity.run_id}:"
                f"{selection.transport_arm_name}:{iteration}"
            ).encode()
        ).decode()
        staging_offset = (
            scenario.staging_offsets_mib[iteration % len(scenario.staging_offsets_mib)]
            * 1024
            * 1024
        )
        expected_fields = {
            "run_id": identity.run_id,
            "config_fingerprint": identity.config_fingerprint,
            "input_bundle_fingerprint": identity.input_bundle_fingerprint,
            "selection_fingerprint": selection.fingerprint,
            "transport_arm": selection.transport_arm_name,
            "iteration": iteration,
            "scenario": selection.scenario_name,
            "scenario_iteration": iteration,
            "producer_request_id": producer_request_id,
            "child_request_id": child_request_id,
            "notification_id": notification_id,
            "has_content_evidence": True,
            "staging_offset": staging_offset,
            "staging_guard_value": (0xD0 + iteration) & 0xFF,
            "destination_canary_value": (0x3C + iteration) & 0xFF,
            "source_observations": expected_consumer_observations,
            "staging_observations": expected_consumer_observations,
            "destination_observations": expected_consumer_observations,
            "saw_proc_during_victim": True,
            "evidence_status": "content_digest",
        }
        expected_iteration_keys = set(expected_fields) | {
            "staging_guard_ranges",
            "victim_duration_ms",
            "transfer_bracket_ms",
            "handles",
            "staging_ownership",
        }
        if set(raw_record) != expected_iteration_keys:
            errors.append(f"iteration {iteration} schema differs")
        for field, expected in expected_fields.items():
            if not _exact(raw_record, field, expected):
                errors.append(f"iteration {iteration} field {field} differs")
        for timing_field in ("victim_duration_ms", "transfer_bracket_ms"):
            timing = raw_record.get(timing_field)
            if (
                type(timing) not in {int, float}
                or not math.isfinite(timing)
                or timing < 0
            ):
                errors.append(f"iteration {iteration} {timing_field} differs")
        handles = raw_record.get("handles")
        if not isinstance(handles, list) or len(handles) != plan.rank_count:
            errors.append(f"iteration {iteration} handle count differs")
            continue
        for rank, handle in enumerate(handles):
            if not isinstance(handle, dict):
                errors.append(f"iteration {iteration} handle {rank} is invalid")
                continue
            if set(handle) != {
                "backend",
                "start_time_us",
                "post_duration_us",
                "transfer_duration_us",
                "total_bytes",
                "descriptor_count",
            }:
                errors.append(f"iteration {iteration} handle {rank} schema differs")
            if handle.get("backend") != "UCX":
                errors.append(f"iteration {iteration} handle {rank} backend differs")
            if (
                type(handle.get("descriptor_count")) is not int
                or handle.get("descriptor_count") != plan.descriptors_per_handle
            ):
                errors.append(
                    f"iteration {iteration} handle {rank} descriptor count differs"
                )
            if (
                type(handle.get("total_bytes")) is not int
                or handle.get("total_bytes") != expected_bytes
            ):
                errors.append(f"iteration {iteration} handle {rank} bytes differ")
            for timing_field in (
                "start_time_us",
                "post_duration_us",
                "transfer_duration_us",
            ):
                timing = handle.get(timing_field)
                if (
                    type(timing) not in {int, float}
                    or not math.isfinite(timing)
                    or timing < 0
                ):
                    errors.append(
                        f"iteration {iteration} handle {rank} {timing_field} differs"
                    )
        capacity = config.staging_capacity_mib * 1024 * 1024
        staging_end = staging_offset + plan.staging_bytes
        expected_guard_ranges = [
            [max(0, staging_offset - _STAGING_GUARD_BYTES), staging_offset],
            [staging_end, min(capacity, staging_end + _STAGING_GUARD_BYTES)],
        ]
        expected_guard_ranges = [
            bounds for bounds in expected_guard_ranges if bounds[1] > bounds[0]
        ]
        if raw_record.get("staging_guard_ranges") != expected_guard_ranges:
            errors.append(f"iteration {iteration} staging guard ranges differ")
        _validate_pass_staging_ownership(
            record=raw_record.get("staging_ownership"),
            identity=identity,
            iteration=iteration,
            child_request_id=child_request_id,
            staging_offset=staging_offset,
            staging_bytes=plan.staging_bytes,
            rank_count=plan.rank_count,
            errors=errors,
        )

    source_pre: dict[str, str] = {}
    source_post: dict[str, str] = {}
    for rank in range(plan.rank_count):
        pre = _observation_map(
            path=arm_directory / f"producer-{rank}-source-pre.jsonl",
            expected_stage="source_pre",
            identity=identity,
            config=config,
            plan=plan,
            errors=errors,
        )
        post = _observation_map(
            path=arm_directory / f"producer-{rank}-source-post.jsonl",
            expected_stage="source_post",
            identity=identity,
            config=config,
            plan=plan,
            errors=errors,
        )
        if len(source_pre.keys() & pre.keys()) > 0:
            errors.append(f"producer {rank} duplicates source-pre identities")
        if len(source_post.keys() & post.keys()) > 0:
            errors.append(f"producer {rank} duplicates source-post identities")
        source_pre.update(pre)
        source_post.update(post)
    staging_observations = _observation_map(
        path=arm_directory / "consumer-staging.jsonl",
        expected_stage="staging_raw",
        identity=identity,
        config=config,
        plan=plan,
        errors=errors,
    )
    destination_observations = _observation_map(
        path=arm_directory / "consumer-destination.jsonl",
        expected_stage="destination",
        identity=identity,
        config=config,
        plan=plan,
        errors=errors,
    )
    counts.update(
        {
            "iterations": len(iterations),
            "source_pre_observations": len(source_pre),
            "source_post_observations": len(source_post),
            "staging_observations": len(staging_observations),
            "destination_observations": len(destination_observations),
        }
    )
    expected_total = expected_consumer_observations * scenario.iterations
    for label, observations in (
        ("source-pre", source_pre),
        ("source-post", source_post),
        ("staging", staging_observations),
        ("destination", destination_observations),
    ):
        if len(observations) != expected_total:
            errors.append(f"{label} observation count differs")
    if source_pre != source_post:
        errors.append("source-pre and source-post identities differ")
    if source_pre != staging_observations:
        errors.append("source and staging identities differ")
    if source_pre != destination_observations:
        errors.append("source and destination identities differ")


def validate_campaign_evidence(
    run_directory: Path,
    *,
    disposition: str,
) -> CampaignValidation:
    """Validate campaign evidence before its terminal envelope is written.

    :param run_directory: Published or unpublished campaign directory.
    :param disposition: Expected PASS, FAIL, or INVALID status.
    :returns: Cross-artifact validation result.
    """
    errors: list[str] = []
    counts: dict[str, int] = {}
    if disposition not in _DISPOSITIONS:
        errors.append(f"unsupported campaign disposition: {disposition}")
        disposition = "INVALID"
    try:
        config_path = run_directory / "inputs" / "config.json"
        config = load_config(config_path)
        selection = _read_selection(run_directory / "inputs" / "selection.json")
        selection.validate(config)
        input_provenance = _read_object(run_directory / "provenance" / "inputs.json")
        bundle_fingerprint = _validate_input_provenance(
            run_directory=run_directory,
            config=config,
            selection=selection,
            provenance=input_provenance,
            errors=errors,
        )
        start = _read_object(run_directory / "campaign-start.json")
        run_id = start.get("run_id")
        if type(run_id) is not str:
            raise ValueError("campaign start lacks run ID")
        identity = CampaignIdentity(
            run_id=run_id,
            config_fingerprint=config.fingerprint,
            input_bundle_fingerprint=bundle_fingerprint,
            selection=selection,
        )
        expected_start = {
            "schema_version": 1,
            "run_id": identity.run_id,
            "selection": selection.to_json(),
            "selection_fingerprint": selection.fingerprint,
            "config_fingerprint": config.fingerprint,
            "input_bundle_fingerprint": bundle_fingerprint,
        }
        for field, expected in expected_start.items():
            if not _exact(start, field, expected):
                errors.append(f"campaign start field {field} differs")
        source_repository, source_hashes = _validate_source_provenance(
            run_directory, errors
        )
        before = _read_object(run_directory / "protected-gpus-before.json")
        after = _read_object(run_directory / "protected-gpus-after.json")
        _validate_protected_snapshot(before, "before", errors)
        _validate_protected_snapshot(after, "after", errors)
        protected_identity = {
            "run_id": identity.run_id,
            "config_fingerprint": identity.config_fingerprint,
            "input_bundle_fingerprint": identity.input_bundle_fingerprint,
            "selection": selection.to_json(),
        }
        for label, snapshot in (("before", before), ("after", after)):
            for field, expected in protected_identity.items():
                if not _exact(snapshot, field, expected):
                    errors.append(f"{label} protected snapshot field {field} differs")
        if before != after:
            errors.append("protected GPU process identities changed")
        preflight = _read_object(run_directory / "preflight.json")
        if disposition == "PASS":
            _validate_pass_arm(
                run_directory=run_directory,
                config_path=config_path,
                config=config,
                identity=identity,
                preflight=preflight,
                source_repository=source_repository,
                source_hashes=source_hashes,
                errors=errors,
                counts=counts,
            )
        elif disposition == "FAIL":
            _validate_fail_arm(
                run_directory=run_directory,
                config=config,
                identity=identity,
                preflight=preflight,
                errors=errors,
            )
    except (KeyError, IndexError, OSError, TypeError, ValueError) as error:
        errors.append(f"campaign artifact is incomplete or malformed: {error}")
    return CampaignValidation(
        disposition=disposition,
        passed=len(errors) == 0,
        errors=tuple(errors),
        counts=counts,
    )


def _validate_publication_envelope(
    *,
    run_directory: Path,
    disposition: str,
    evidence: CampaignValidation,
    expected_published_name: str | None,
    errors: list[str],
) -> None:
    """Bind terminal status and saved validation to the sealed evidence tree.

    :param run_directory: Published campaign directory.
    :param disposition: Status selected from the terminal record.
    :param evidence: Fresh validation of the final disposition.
    :param expected_published_name: Final atomic-publication basename.
    :param errors: Validation error accumulator.
    """
    status = _read_object(run_directory / "run-status.json")
    validation = _read_object(run_directory / "validation.json")
    expected_status_keys = {
        "schema_version",
        "run_id",
        "status",
        "detail",
        "traceback",
        "started_ns",
        "completed_ns",
        "selection",
        "selection_fingerprint",
        "input_bundle_fingerprint",
        "arm_completed",
        "protected_gpu_identity_match",
        "evidence_validation_passed",
        "attempted_result_validation_passed",
        "config_fingerprint",
    }
    if set(status) != expected_status_keys:
        errors.append("run status schema differs")
    if not _exact(status, "schema_version", 1):
        errors.append("run status schema version differs")
    if not _exact(status, "status", disposition):
        errors.append("run status disposition differs")
    run_id = status.get("run_id")
    if type(run_id) is not str:
        errors.append("run status lacks run ID")
        run_id = ""
    expected_name = run_id if disposition == "PASS" else f"{run_id}.{disposition}"
    observed_name = (
        run_directory.name
        if expected_published_name is None
        else expected_published_name
    )
    if observed_name != expected_name:
        errors.append("published directory name differs from run status")
    started_ns = status.get("started_ns")
    completed_ns = status.get("completed_ns")
    if (
        type(started_ns) is not int
        or type(completed_ns) is not int
        or completed_ns < started_ns
    ):
        errors.append("run status clock interval differs")
    for field in ("detail", "traceback"):
        if type(status.get(field)) is not str:
            errors.append(f"run status field {field} differs")
    expected_validation_keys = {
        "schema_version",
        "run_id",
        "config_fingerprint",
        "input_bundle_fingerprint",
        "selection",
        "attempted_result_validation",
        "final_evidence_validation",
    }
    if set(validation) != expected_validation_keys:
        errors.append("validation envelope schema differs")
    if not _exact(validation, "schema_version", 1):
        errors.append("validation envelope schema version differs")
    for field in (
        "run_id",
        "config_fingerprint",
        "input_bundle_fingerprint",
        "selection",
    ):
        if not _exact(validation, field, status.get(field)):
            errors.append(f"validation envelope field {field} differs")
    if validation.get("final_evidence_validation") != evidence.to_json():
        errors.append("saved final evidence validation differs")
    if not _exact(status, "evidence_validation_passed", evidence.passed):
        errors.append("run status evidence-validation result differs")
    attempted = validation.get("attempted_result_validation")
    if attempted is None:
        if status.get("attempted_result_validation_passed") is not None:
            errors.append("run status has an unbacked attempted result")
    elif not isinstance(attempted, dict):
        errors.append("attempted result validation is malformed")
    else:
        attempted_disposition = attempted.get("disposition")
        if type(attempted_disposition) is not str:
            errors.append("attempted result validation lacks a disposition")
        else:
            fresh_attempt = validate_campaign_evidence(
                run_directory,
                disposition=attempted_disposition,
            )
            if attempted != fresh_attempt.to_json():
                errors.append("saved attempted result validation differs")
            if not _exact(
                status,
                "attempted_result_validation_passed",
                fresh_attempt.passed,
            ):
                errors.append("run status attempted-validation result differs")
    for path in (run_directory, *run_directory.rglob("*")):
        if path.is_symlink():
            continue
        if path.stat().st_mode & 0o222 != 0:
            errors.append(
                f"published artifact remains writable: "
                f"{path.relative_to(run_directory) if path != run_directory else '.'}"
            )


def validate_campaign(
    run_directory: Path,
    *,
    disposition: str | None = None,
    expected_published_name: str | None = None,
) -> CampaignValidation:
    """Validate a complete, published campaign artifact.

    :param run_directory: Published campaign directory.
    :param disposition: Optional expected terminal disposition.
    :param expected_published_name: Final basename for a hidden candidate.
    :returns: Complete evidence, checksum, seal, and terminal validation.
    """
    errors: list[str] = []
    try:
        status_record = _read_object(run_directory / "run-status.json")
        recorded_disposition = status_record.get("status")
        if type(recorded_disposition) is not str:
            recorded_disposition = "INVALID"
    except (OSError, ValueError):
        recorded_disposition = disposition if disposition is not None else "INVALID"
        errors.append("campaign lacks a valid run-status.json")
    if disposition is not None and disposition != recorded_disposition:
        errors.append("requested and recorded campaign dispositions differ")
    evidence = validate_campaign_evidence(
        run_directory,
        disposition=recorded_disposition,
    )
    if recorded_disposition in {"PASS", "FAIL"}:
        errors.extend(evidence.errors)
    errors.extend(validate_internal_checksums(run_directory, required=True))
    try:
        _validate_publication_envelope(
            run_directory=run_directory,
            disposition=recorded_disposition,
            evidence=evidence,
            expected_published_name=expected_published_name,
            errors=errors,
        )
    except (KeyError, OSError, TypeError, ValueError) as error:
        errors.append(f"publication envelope is incomplete or malformed: {error}")
    return CampaignValidation(
        disposition=recorded_disposition,
        passed=len(errors) == 0,
        errors=tuple(errors),
        counts=evidence.counts,
    )
