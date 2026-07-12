"""Construct a complete minimal PASS artifact for offline validator tests."""

import base64
import hashlib
import json
import sys
import tarfile
from pathlib import Path

from tools.gemma4_pd.nixl_micro_rig import launcher
from tools.gemma4_pd.nixl_micro_rig.campaign_validator import (
    validate_campaign_evidence,
)
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

_RUN_ID = "12345678-1234-5678-1234-567812345678"
_SOURCE_MODULES = (
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
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_inputs(
    run_directory: Path,
) -> tuple[RigConfig, RunSelection, str]:
    input_directory = run_directory / "inputs"
    input_directory.mkdir(parents=True)
    source_manifest = {
        "ranks": [
            {
                "rank": 0,
                "num_blocks": 2,
                "block_lens": [4],
                "device_id": 0,
                "block_size": 16,
                "physical_blocks_per_logical_kv_block": 1,
                "engine_id": "fixture-p",
                "compatibility_hash": "fixture-compatible",
                "kv_cache_layout": "HND",
                "attn_backend_name": "FLASHINFER_GEMMA4_TRTLLM_GEN",
            }
        ]
    }
    destination_manifest = {
        "ranks": [
            {
                "rank": 0,
                "num_blocks": 2,
                "block_lens": [4],
                "device_id": 0,
                "block_size": 16,
                "physical_blocks_per_logical_kv_block": 1,
                "compatibility_hash": "fixture-compatible",
                "kv_cache_layout": "HND",
                "attn_backend_name": "FLASHINFER_GEMMA4_TRTLLM_GEN",
            }
        ]
    }
    source_path = input_directory / "source-handshake.json"
    destination_path = input_directory / "destination-handshake.json"
    _write_json(source_path, source_manifest)
    _write_json(destination_path, destination_manifest)
    config_value = {
        "schema_version": 1,
        "source_handshake_manifest": source_path.name,
        "destination_handshake_manifest": destination_path.name,
        "source_handshake_sha256": _sha256(source_path),
        "destination_handshake_sha256": _sha256(destination_path),
        "producer_devices": [0],
        "consumer_device": 1,
        "protected_devices": [6, 7],
        "source_block_count": 2,
        "valid_token_extent": 16,
        "staging_capacity_mib": 1,
        "nixl_num_threads": 1,
        "pattern_chunk_bytes": 4,
        "transfer_timeout_seconds": 1,
        "control_host": "127.0.0.1",
        "control_port": 19000,
        "regions": [{"name": "region", "row_bytes": 4}],
        "groups": [
            {
                "index": 0,
                "name": "group",
                "token_capacity": 16,
                "remote_position_count": 1,
                "local_position_count": 1,
                "owned_region_indices": [0],
            }
        ],
        "scenarios": [
            {
                "name": "scenario",
                "source_start_block": 0,
                "source_end_block": 0,
                "run_count": 1,
                "iterations": 1,
                "staging_offsets_mib": [0],
            }
        ],
        "transport_arms": [
            {
                "name": "arm",
                "ucx_tls": "tcp,shm,cuda_copy",
                "ucx_rndv_scheme": "get_zcopy",
                "ucx_net_devices": "all",
            }
        ],
        "victim": {"matrix_order": 1, "rounds": 1, "redzone_bytes": 4},
    }
    config_path = input_directory / "config.json"
    _write_json(config_path, config_value)
    selection = RunSelection("scenario", "arm")
    selection_path = input_directory / "selection.json"
    _write_json(selection_path, selection.to_json())
    config = load_config(config_path)
    paths = sorted((config_path, selection_path, source_path, destination_path))
    records = [
        {
            "path": path.relative_to(run_directory).as_posix(),
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in paths
    ]
    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    bundle_fingerprint = hashlib.sha256(payload).hexdigest()
    _write_json(
        run_directory / "provenance" / "inputs.json",
        {
            "source_config_path": str(config_path),
            "config_fingerprint": config.fingerprint,
            "selection": selection.to_json(),
            "selection_fingerprint": selection.fingerprint,
            "input_bundle_fingerprint": bundle_fingerprint,
            "files": records,
        },
    )
    return config, selection, bundle_fingerprint


def _write_source_provenance(run_directory: Path) -> tuple[Path, list[dict[str, str]]]:
    repository = run_directory.parent / "executed-source"
    source_tree = run_directory / "provenance" / "source-tree"
    records: list[dict[str, str]] = []
    module_records: list[dict[str, str]] = []
    for index, relative_text in enumerate(_SOURCE_MODULES):
        relative = Path(relative_text)
        payload = f"# fixture source {index}\n".encode()
        executed_path = repository / relative
        copied_path = source_tree / relative
        executed_path.parent.mkdir(parents=True, exist_ok=True)
        copied_path.parent.mkdir(parents=True, exist_ok=True)
        executed_path.write_bytes(payload)
        copied_path.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        records.append({"path": relative_text, "sha256": digest})
        module_records.append(
            {
                "module": f"fixture_module_{index}",
                "path": str(executed_path),
                "sha256": digest,
            }
        )
    archive_path = run_directory / "provenance" / "source.tar"
    with tarfile.open(archive_path, mode="w") as archive:
        for relative_text in _SOURCE_MODULES:
            archive.add(source_tree / relative_text, arcname=relative_text)
    empty_digest = hashlib.sha256(b"").hexdigest()
    _write_json(
        run_directory / "provenance" / "source.json",
        {
            "repository": str(repository),
            "head": "1" * 40,
            "tree": "2" * 40,
            "branch": "fixture",
            "worktree_clean": True,
            "git_status_sha256": empty_digest,
            "git_diff_sha256": empty_digest,
            "files": records,
            "git_archive": {
                "path": archive_path.relative_to(run_directory).as_posix(),
                "sha256": _sha256(archive_path),
            },
            "runtime_paths": {
                "python": {},
                "nixl": {},
                "nixl-cu13": {},
                "vllm": {},
            },
        },
    )
    (run_directory / "provenance" / "git-status.txt").write_bytes(b"")
    (run_directory / "provenance" / "git-diff.patch").write_bytes(b"")
    return repository, module_records


def _native_runtime(
    run_directory: Path,
) -> tuple[
    dict[str, dict[str, str]],
    list[dict[str, str]],
    bytes,
    bytes,
]:
    site_packages = run_directory.parent / "runtime" / "site-packages"
    nixl_directory = site_packages / "nixl"
    nixl_directory.mkdir(parents=True)
    api_path = nixl_directory / "_api.py"
    bindings_path = nixl_directory / "_bindings.so"
    api_path.write_bytes(b"fixture api")
    bindings_path.write_bytes(b"fixture bindings")
    api_records = {
        "nixl_api": {"path": str(api_path), "sha256": _sha256(api_path)},
        "nixl_bindings": {
            "path": str(bindings_path),
            "sha256": _sha256(bindings_path),
        },
    }
    libraries: list[dict[str, str]] = []
    map_lines: list[str] = []
    archive_directory = run_directory / "provenance" / "native-libraries"
    archive_directory.mkdir()
    for name in ("libnixl.so", "libplugin_UCX.so", "libucp.so"):
        path = site_packages / name
        path.write_bytes(name.encode())
        digest = _sha256(path)
        archived = archive_directory / f"{digest}-{name}"
        archived.write_bytes(path.read_bytes())
        libraries.append(
            {
                "path": str(path),
                "sha256": digest,
                "archived_path": archived.relative_to(run_directory).as_posix(),
            }
        )
        map_lines.append(f"0-1 r-xp 00000000 00:00 0 {path}\n")
    return api_records, libraries, "".join(map_lines).encode(), b"fixture limits\n"


def _process_attestation(
    *,
    role: str,
    physical_device: int,
    gpu_uuid: str,
    visibility: str,
    pid: int,
    starttime_ticks: int,
    argv: list[str],
    selection: RunSelection,
    config: RigConfig,
    bundle_fingerprint: str,
    module_records: list[dict[str, str]],
    api_records: dict[str, dict[str, str]],
    libraries: list[dict[str, str]],
    maps_payload: bytes,
    limits_payload: bytes,
) -> dict[str, object]:
    environment = {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": visibility,
        "UCX_NET_DEVICES": "all",
        "UCX_PROTO_INFO": "y",
        "UCX_RNDV_SCHEME": "get_zcopy",
        "UCX_TLS": "tcp,shm,cuda_copy",
    }
    return {
        "run_id": _RUN_ID,
        "config_fingerprint": config.fingerprint,
        "input_bundle_fingerprint": bundle_fingerprint,
        "selection": selection.to_json(),
        "selection_fingerprint": selection.fingerprint,
        "role": role,
        "pid": pid,
        "starttime_ticks": starttime_ticks,
        "boot_id": "fixture-boot",
        "argv_raw_hex": (
            b"\0".join(argument.encode() for argument in argv) + b"\0"
        ).hex(),
        "python_argv": argv[1:],
        "executable": "/usr/bin/python",
        "cwd": "/fixture",
        "configured_gpu_uuid": gpu_uuid,
        "observed_gpu_uuid": gpu_uuid,
        "physical_device": physical_device,
        "logical_device": 0,
        "nixl_version": "1.3.0",
        "nixl_cuda_version": "1.3.0",
        "ucx_version": "1.21.0",
        "nixl_plugin_list": ["UCX"],
        "nixl_ucx_plugin_params": {},
        "nixl_ucx_backend_params": {},
        "nixl_ucx_memory_types": ["VRAM"],
        **api_records,
        "loaded_python_modules": module_records,
        "loaded_native_libraries": libraries,
        "proc_maps_sha256": hashlib.sha256(maps_payload).hexdigest(),
        "proc_limits_sha256": hashlib.sha256(limits_payload).hexdigest(),
        "rlimit_nofile": [65_535, 1_048_576],
        "environment": environment,
    }


def _protected_snapshot(
    *,
    config: RigConfig,
    selection: RunSelection,
    bundle_fingerprint: str,
) -> dict[str, object]:
    argv = ["python", "serve"]
    raw_hex = b"python\0serve\0".hex()
    process = {
        "pid": 600,
        "starttime_ticks": 60,
        "process_group": 600,
        "session_id": 600,
        "argv_raw_hex": raw_hex,
        "argv": argv,
        "executable": "/usr/bin/python",
        "process_name": "server",
        "devices": [6, 7],
    }
    listeners = [
        {
            "port": port,
            "socket_inodes": [port],
            "owners": [
                {
                    "pid": port,
                    "starttime_ticks": port,
                    "process_group": port,
                    "session_id": port,
                    "argv_raw_hex": raw_hex,
                    "argv": argv,
                    "executable": "/usr/bin/python",
                }
            ],
        }
        for port in (8000, 8820, 8821)
    ]
    return {
        "protected_devices": [6, 7],
        "boot_id": "fixture-boot",
        "device_uuids": {6: "GPU-6", 7: "GPU-7"},
        "processes": [process],
        "listeners": listeners,
        "run_id": _RUN_ID,
        "config_fingerprint": config.fingerprint,
        "input_bundle_fingerprint": bundle_fingerprint,
        "selection": selection.to_json(),
    }


def _observation(
    *,
    stage: str,
    config: RigConfig,
    selection: RunSelection,
    plan: TransferPlan,
    bundle_fingerprint: str,
) -> dict[str, object]:
    pairing = plan.sorted_pairings[0]
    source_stage = stage in {"source_pre", "source_post"}
    identity = {
        "run_id": _RUN_ID,
        "transport_arm": selection.transport_arm_name,
        "producer_engine_id": f"micro-p-{_RUN_ID}",
        "producer_request_id": f"p-{_RUN_ID}-0",
        "registration_generation": f"{_RUN_ID}:arm:producer-rank-0:registration-1",
        "semantic_contract_digest": compute_rig_semantic_contract_digest(
            config, 0, 0
        ).hex(),
        "offer_generation": 0,
        "iteration": 0,
        "source_rank": 0,
        "region_index": 0,
        "group_index": 0,
        "plane_index": -1,
        "source_position": pairing.group_position,
        "remote_block_id": pairing.remote_block_id,
        "valid_token_extent": config.valid_token_extent,
        "group_token_capacity": config.groups[0].token_capacity,
        "payload_kind": "wire",
        "byte_length": config.regions[0].row_bytes,
    }
    return {
        "stage": stage,
        "identity": identity,
        "digest": "a" * 32,
        "child_request_id": None if source_stage else f"d-{_RUN_ID}-0",
        "observer_engine_id": (
            f"micro-p-{_RUN_ID}" if source_stage else f"micro-d-{_RUN_ID}"
        ),
        "observer_rank": 0,
        "local_block_id": None if source_stage else pairing.local_block_id,
        "config_fingerprint": config.fingerprint,
        "input_bundle_fingerprint": bundle_fingerprint,
        "scenario": selection.scenario_name,
        "evidence_status": "content_digest",
    }


def build_published_pass_campaign(artifact_root: Path) -> Path:
    """Build, validate, seal, and publish one coherent minimal PASS campaign.

    :param artifact_root: Fresh publication parent.
    :returns: Published PASS directory.
    """
    artifact_root.mkdir()
    build_directory = artifact_root / ".fixture.partial"
    build_directory.mkdir()
    config, selection, bundle_fingerprint = _write_inputs(build_directory)
    _, module_records = _write_source_provenance(build_directory)
    api_records, libraries, maps_payload, limits_payload = _native_runtime(
        build_directory
    )
    scenario = config.scenario(selection.scenario_name)
    config_path = build_directory / "inputs" / "config.json"
    plan = build_configured_plan(config_path, config, scenario)
    shared = {
        "run_id": _RUN_ID,
        "config_fingerprint": config.fingerprint,
        "input_bundle_fingerprint": bundle_fingerprint,
        "selection": selection.to_json(),
    }
    _write_json(
        build_directory / "campaign-start.json",
        {
            "schema_version": 1,
            **shared,
            "selection_fingerprint": selection.fingerprint,
            "started_ns": 1,
            "source_head": "1" * 40,
        },
    )
    _write_json(
        build_directory / "plan.json",
        {**shared, "plan": describe_plan(config, scenario, plan)},
    )
    protected = _protected_snapshot(
        config=config,
        selection=selection,
        bundle_fingerprint=bundle_fingerprint,
    )
    _write_json(build_directory / "protected-gpus-before.json", protected)
    _write_json(build_directory / "protected-gpus-after.json", protected)
    uuids = {0: "GPU-0", 1: "GPU-1"}
    _write_json(
        build_directory / "preflight.json",
        {
            **shared,
            "selected_devices": [0, 1],
            "selected_device_uuids": uuids,
            "immutable_allowed_devices": list(range(6)),
            "immutable_denied_devices": [6, 7],
            "inventory": {},
            "required_free_bytes": {},
        },
    )
    _write_json(
        build_directory / "selected-gpus-after.json",
        {
            **shared,
            "selected_devices": [0, 1],
            "selected_device_uuids": uuids,
            "processes": [],
        },
    )
    arm_directory = build_directory / "arms" / "arm" / "scenario"
    arm_directory.mkdir(parents=True)
    producer_argv = [sys.executable, "-m", "fixture", "producer"]
    consumer_argv = [sys.executable, "-m", "fixture", "consumer"]
    producer_environment = {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": "GPU-0",
        "UCX_NET_DEVICES": "all",
        "UCX_PROTO_INFO": "y",
        "UCX_RNDV_SCHEME": "get_zcopy",
        "UCX_TLS": "tcp,shm,cuda_copy",
    }
    consumer_environment = {
        **producer_environment,
        "CUDA_VISIBLE_DEVICES": "GPU-1",
    }
    launches = [
        {
            **shared,
            "name": "producer-0",
            "role": "producer",
            "rank": 0,
            "pid": 100,
            "starttime_ticks": 1000,
            "argv": producer_argv,
            "environment": producer_environment,
        },
        {
            **shared,
            "name": "consumer",
            "role": "consumer",
            "rank": 0,
            "pid": 200,
            "starttime_ticks": 2000,
            "argv": consumer_argv,
            "environment": consumer_environment,
        },
    ]
    _write_json(arm_directory / "process-launches.json", launches)
    _write_json(
        arm_directory / "process-exits.json",
        {**shared, "exit_codes": {"producer-0": 0, "consumer": 0}},
    )
    _write_json(arm_directory / "cleanup.json", {**shared, "errors": []})
    producer_attestation = _process_attestation(
        role="producer:0",
        physical_device=0,
        gpu_uuid="GPU-0",
        visibility="GPU-0",
        pid=100,
        starttime_ticks=1000,
        argv=producer_argv,
        selection=selection,
        config=config,
        bundle_fingerprint=bundle_fingerprint,
        module_records=module_records,
        api_records=api_records,
        libraries=libraries,
        maps_payload=maps_payload,
        limits_payload=limits_payload,
    )
    consumer_attestation = _process_attestation(
        role="consumer:0",
        physical_device=1,
        gpu_uuid="GPU-1",
        visibility="GPU-1",
        pid=200,
        starttime_ticks=2000,
        argv=consumer_argv,
        selection=selection,
        config=config,
        bundle_fingerprint=bundle_fingerprint,
        module_records=module_records,
        api_records=api_records,
        libraries=libraries,
        maps_payload=maps_payload,
        limits_payload=limits_payload,
    )
    _write_json(arm_directory / "producer-0-attestation.json", producer_attestation)
    _write_json(
        arm_directory / "consumer-attestation.json",
        {"consumer": consumer_attestation, "producers": [producer_attestation]},
    )
    for role_name in ("producer-0", "consumer"):
        (arm_directory / f"{role_name}-proc-maps.txt").write_bytes(maps_payload)
        (arm_directory / f"{role_name}-proc-limits.txt").write_bytes(limits_payload)
    for role, path_name in (("producer", "producer-0"), ("consumer", "consumer")):
        _write_json(
            arm_directory / f"{path_name}-terminal.json",
            {
                "schema_version": 1,
                **shared,
                "role": role,
                "rank": 0,
                "status": "PASS",
                "detail": "role completed cleanly",
                "traceback": "",
                "completed_ns": 2,
            },
        )
    _write_json(
        arm_directory / "runtime-registrations.json",
        {
            "run_id": _RUN_ID,
            "transport_arm": "arm",
            "scenario": "scenario",
            "selection_fingerprint": selection.fingerprint,
            "config_fingerprint": config.fingerprint,
            "input_bundle_fingerprint": bundle_fingerprint,
            "destination": [{}],
            "producers": [{}],
            "staging": {"byte_length": 1024 * 1024},
        },
    )
    iteration = {
        "run_id": _RUN_ID,
        "config_fingerprint": config.fingerprint,
        "input_bundle_fingerprint": bundle_fingerprint,
        "selection_fingerprint": selection.fingerprint,
        "transport_arm": "arm",
        "iteration": 0,
        "scenario": "scenario",
        "scenario_iteration": 0,
        "producer_request_id": f"p-{_RUN_ID}-0",
        "child_request_id": f"d-{_RUN_ID}-0",
        "notification_id": base64.b64encode(
            f"micro-rig:{_RUN_ID}:arm:0".encode()
        ).decode(),
        "has_content_evidence": True,
        "staging_offset": 0,
        "staging_guard_ranges": [[4, 1024 * 1024]],
        "staging_guard_value": 0xD0,
        "destination_canary_value": 0x3C,
        "source_observations": 1,
        "staging_observations": 1,
        "destination_observations": 1,
        "victim_duration_ms": 1.0,
        "transfer_bracket_ms": 1.0,
        "saw_proc_during_victim": True,
        "evidence_status": "content_digest",
        "handles": [
            {
                "backend": "UCX",
                "start_time_us": 1,
                "post_duration_us": 1,
                "transfer_duration_us": 1,
                "total_bytes": 4,
                "descriptor_count": 1,
            }
        ],
        "staging_ownership": {
            "generation": 0,
            "owner_id": "rig:0",
            "request_id": f"d-{_RUN_ID}-0",
            "remote_engine_id": f"micro-p-{_RUN_ID}",
            "offset": 0,
            "size": 4,
            "age_s": 1.0,
            "handles": [
                {
                    "source_rank": 0,
                    "state": "done",
                    "native_handle_present": True,
                    "native_released": True,
                }
            ],
            "posting_sealed": True,
            "operation_failed": False,
            "permanently_tombstoned": False,
            "native_quiescent": True,
            "device_read_started": True,
            "device_quiescent": True,
            "reusable": True,
            "released": True,
            "failure_reason": None,
        },
    }
    _write_json(arm_directory / "iterations.json", [iteration])
    for stage, name in (
        ("source_pre", "producer-0-source-pre.jsonl"),
        ("source_post", "producer-0-source-post.jsonl"),
        ("staging_raw", "consumer-staging.jsonl"),
        ("destination", "consumer-destination.jsonl"),
    ):
        observation = _observation(
            stage=stage,
            config=config,
            selection=selection,
            plan=plan,
            bundle_fingerprint=bundle_fingerprint,
        )
        (arm_directory / name).write_text(
            json.dumps(observation, sort_keys=True) + "\n"
        )
    evidence = validate_campaign_evidence(build_directory, disposition="PASS")
    if evidence.passed is False:
        raise AssertionError(f"fixture evidence is invalid: {evidence.errors}")
    _write_json(
        build_directory / "validation.json",
        {
            "schema_version": 1,
            **shared,
            "attempted_result_validation": evidence.to_json(),
            "final_evidence_validation": evidence.to_json(),
        },
    )
    _write_json(
        build_directory / "run-status.json",
        {
            "schema_version": 1,
            **shared,
            "status": "PASS",
            "detail": "fixture PASS",
            "traceback": "",
            "started_ns": 1,
            "completed_ns": 2,
            "selection_fingerprint": selection.fingerprint,
            "arm_completed": True,
            "protected_gpu_identity_match": True,
            "evidence_validation_passed": True,
            "attempted_result_validation_passed": True,
        },
    )
    return launcher._publish_run(
        build_directory=build_directory,
        artifact_root=artifact_root,
        run_id=_RUN_ID,
        status=launcher.RunStatus.PASS,
    )


def make_tree_writable(root: Path) -> None:
    """Restore fixture permissions for temporary-directory cleanup.

    :param root: Sealed fixture root.
    """
    root.chmod(0o755)
    for path in root.rglob("*"):
        if path.is_dir():
            path.chmod(0o755)
        else:
            path.chmod(0o644)
