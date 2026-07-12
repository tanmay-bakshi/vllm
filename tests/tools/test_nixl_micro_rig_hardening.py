"""Host-only tests for native campaign safety and artifact publication."""

import json
import os
import signal
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from tests.tools.nixl_campaign_pass_fixture import (
    build_published_pass_campaign,
    make_tree_writable,
)
from tools.gemma4_pd.nixl_micro_rig import campaign_validator, launcher, role_entry
from tools.gemma4_pd.nixl_micro_rig.campaign_validator import (
    validate_campaign,
    validate_internal_checksums,
)
from tools.gemma4_pd.nixl_micro_rig.config import (
    ConfigError,
    load_config,
    parse_config_bytes,
)
from tools.gemma4_pd.nixl_micro_rig.protocol import (
    JsonChannel,
    PreparePayload,
    ProtocolError,
)
from tools.gemma4_pd.nixl_micro_rig.selection import RunSelection

CONFIG_PATH = (
    Path(__file__).parents[2]
    / "tools"
    / "gemma4_pd"
    / "nixl_micro_rig"
    / "gemma4_tp4_to_tp1.json"
)


def _selection() -> RunSelection:
    return RunSelection(
        scenario_name="observed_ceiling_102_runs",
        transport_arm_name="cuda_copy",
    )


def test_run_selection_preserves_full_config_identity() -> None:
    config = load_config(CONFIG_PATH)
    selection = _selection()

    selection.validate(config)

    assert len(config.scenarios) == 9
    assert len(config.transport_arms) == 2
    assert selection.fingerprint == selection.fingerprint
    assert len(selection.fingerprint) == 64


def test_run_selection_rejects_unknown_identity() -> None:
    config = load_config(CONFIG_PATH)

    with pytest.raises(ConfigError, match="unknown scenario"):
        RunSelection("missing", "cuda_copy").validate(config)
    with pytest.raises(ConfigError, match="unknown transport arm"):
        RunSelection("observed_ceiling_102_runs", "missing").validate(config)


def test_config_parser_rejects_duplicate_keys_and_nonfinite_values() -> None:
    duplicate = b'{"schema_version": 1, "schema_version": 1}'
    nonfinite = b'{"schema_version": NaN}'

    with pytest.raises(ConfigError, match="duplicates key"):
        parse_config_bytes(duplicate, source="duplicate")
    with pytest.raises(ConfigError, match="non-finite"):
        parse_config_bytes(nonfinite, source="nonfinite")


def test_config_rejects_artifact_path_names() -> None:
    value = json.loads(CONFIG_PATH.read_text())
    value["scenarios"][0]["name"] = "../escape"

    with pytest.raises(ConfigError, match="artifact-safe"):
        parse_config_bytes(json.dumps(value).encode(), source="unsafe-name")


def test_role_environment_scrubs_inherited_ucx_before_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(CONFIG_PATH)
    monkeypatch.setenv("UCX_LOG_LEVEL", "trace")
    monkeypatch.setenv("UCX_NET_DEVICES", "wrong")
    monkeypatch.setenv("UCX_TLS", "wrong")
    monkeypatch.setenv("NIXL_LOG_LEVEL", "trace")
    original = dict(os.environ)
    uuids = {device: f"GPU-{device}" for device in range(8)}

    environment = launcher._role_environment(
        config,
        _selection(),
        "consumer",
        uuids,
    )

    assert dict(os.environ) == original
    assert {
        name: value
        for name, value in environment.items()
        if name.startswith("UCX_") or name.startswith("NIXL_")
    } == {
        "UCX_NET_DEVICES": "all",
        "UCX_PROTO_INFO": "y",
        "UCX_RNDV_SCHEME": "get_zcopy",
        "UCX_TLS": "tcp,shm,cuda_copy",
    }
    assert environment["CUDA_VISIBLE_DEVICES"] == "GPU-4"


def test_role_entry_imports_no_accelerator_library() -> None:
    script = (
        "import sys; "
        "import tools.gemma4_pd.nixl_micro_rig.role_entry; "
        "assert 'torch' not in sys.modules; "
        "assert 'nixl' not in sys.modules"
    )

    subprocess.run([sys.executable, "-c", script], check=True)


def test_role_entry_binds_production_nofile_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    applied: list[tuple[int, tuple[int, int]]] = []
    monkeypatch.setattr(
        role_entry.resource,
        "getrlimit",
        lambda resource_id: (1024, 1_048_576),
    )
    monkeypatch.setattr(
        role_entry.resource,
        "setrlimit",
        lambda resource_id, limits: applied.append((resource_id, limits)),
    )

    role_entry._bind_role_nofile_limit()

    assert applied == [(role_entry.resource.RLIMIT_NOFILE, (65_535, 1_048_576))]


def test_campaign_lock_conflicts_with_experiment_lane_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path.chmod(0o700)
    monkeypatch.setattr(launcher, "_LOCK_PATH", tmp_path / "experiment-lane.lock")

    with (
        launcher._exclusive_lock(),
        pytest.raises(RuntimeError, match="already held"),
        launcher._exclusive_lock(),
    ):
        pass


def test_protected_listener_snapshot_rejects_absent_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    header = "sl local_address rem_address st tx rx tr tm retr uid timeout inode"
    rows = [
        "0: 00000000:2274 00000000:0000 0A 0:0 00:0 0 1000 0 8820",
        "1: 00000000:2275 00000000:0000 0A 0:0 00:0 0 1000 0 8821",
    ]

    def fake_read_text(path: Path) -> str:
        return "\n".join([header, *rows]) if path.name == "tcp" else header

    monkeypatch.setattr(launcher.Path, "read_text", fake_read_text)

    with pytest.raises(RuntimeError, match="8000"):
        launcher._listener_socket_inodes(frozenset({8000, 8820, 8821}))


def test_protected_listener_snapshot_rejects_owner_handoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inodes = {
        8000: frozenset({80}),
        8820: frozenset({820}),
        8821: frozenset({821}),
    }
    first_owner = {
        port: (
            {
                "pid": port,
                "starttime_ticks": 1,
                "process_group": port,
                "session_id": port,
                "argv_raw_hex": b"server\0".hex(),
                "argv": ["server"],
                "executable": "/server",
            },
        )
        for port in inodes
    }
    second_owner = {
        **first_owner,
        8000: (
            {
                **first_owner[8000][0],
                "pid": 9000,
                "starttime_ticks": 2,
            },
        ),
    }
    owners = iter((first_owner, second_owner))
    monkeypatch.setattr(launcher, "_listener_socket_inodes", lambda ports: inodes)
    monkeypatch.setattr(
        launcher,
        "_listener_process_owners",
        lambda observed_inodes: next(owners),
    )

    with pytest.raises(RuntimeError, match="ownership changed"):
        launcher._protected_listener_snapshot()


def test_protected_gpu_snapshot_binds_pid_starttime_and_raw_argv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(CONFIG_PATH)
    rows = ({6: "GPU-6", 7: "GPU-7"}, ((6, 123, "server"), (7, 123, "server")))
    monkeypatch.setattr(launcher, "_protected_compute_rows", lambda devices: rows)
    monkeypatch.setattr(
        launcher,
        "_read_process_identity",
        lambda pid: {
            "pid": pid,
            "starttime_ticks": 456,
            "process_group": 123,
            "session_id": 123,
            "argv_raw_hex": b"python\0serve\0".hex(),
            "argv": ["python", "serve"],
            "executable": "/usr/bin/python",
        },
    )
    listeners = [
        {
            "port": port,
            "socket_inodes": [port],
            "owners": [
                {
                    "pid": port,
                    "starttime_ticks": 456,
                    "process_group": port,
                    "session_id": port,
                    "argv_raw_hex": b"python\0serve\0".hex(),
                    "argv": ["python", "serve"],
                    "executable": "/usr/bin/python",
                }
            ],
        }
        for port in (8000, 8820, 8821)
    ]
    monkeypatch.setattr(launcher, "_protected_listener_snapshot", lambda: listeners)

    snapshot = launcher.protected_gpu_process_snapshot(config)

    assert snapshot["protected_devices"] == [6, 7]
    assert snapshot["processes"] == [
        {
            "pid": 123,
            "starttime_ticks": 456,
            "process_group": 123,
            "session_id": 123,
            "argv_raw_hex": b"python\0serve\0".hex(),
            "argv": ["python", "serve"],
            "executable": "/usr/bin/python",
            "process_name": "server",
            "devices": [6, 7],
        }
    ]
    assert snapshot["listeners"] == listeners


def test_protected_gpu_snapshot_rejects_membership_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(CONFIG_PATH)
    samples = iter(
        (
            ({6: "GPU-6", 7: "GPU-7"}, ((6, 123, "server"),)),
            ({6: "GPU-6", 7: "GPU-7"}, ()),
        )
    )
    monkeypatch.setattr(
        launcher, "_protected_compute_rows", lambda devices: next(samples)
    )
    monkeypatch.setattr(
        launcher,
        "_read_process_identity",
        lambda pid: {
            "pid": pid,
            "starttime_ticks": 456,
            "process_group": 123,
            "session_id": 123,
            "argv_raw_hex": "00",
            "argv": ["server"],
            "executable": "/server",
        },
    )
    monkeypatch.setattr(launcher, "_protected_listener_snapshot", lambda: [])

    with pytest.raises(RuntimeError, match="changed during capture"):
        launcher.protected_gpu_process_snapshot(config)


def test_supervisor_deadline_rejects_nonterminal_process() -> None:
    class FakeProcess:
        def poll(self) -> None:
            return None

    times = iter((0.0, 2.0))

    with pytest.raises(launcher.ArmSupervisorTimeout, match="nonterminal"):
        launcher._wait_for_processes(
            {"producer-0": FakeProcess()},  # type: ignore[dict-item]
            deadline=1.0,
            timeout_seconds=1,
            clock=lambda: next(times),
            pause=lambda seconds: None,
        )


def test_cleanup_signals_only_identical_owned_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProcess:
        pid = 123

        def poll(self) -> None:
            return None

    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(
        launcher,
        "_read_process_identity",
        lambda pid: {"starttime_ticks": 456},
    )
    monkeypatch.setattr(launcher.os, "getpgid", lambda pid: 123)
    monkeypatch.setattr(
        launcher.os,
        "killpg",
        lambda process_group, signal_number: signals.append(
            (process_group, signal_number)
        ),
    )

    launcher._signal_owned_process_group(
        FakeProcess(),  # type: ignore[arg-type]
        456,
        signal.SIGTERM,
    )

    assert signals == [(123, signal.SIGTERM)]


def test_cleanup_rejects_reused_process_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProcess:
        pid = 123

        def poll(self) -> None:
            return None

    monkeypatch.setattr(
        launcher,
        "_read_process_identity",
        lambda pid: {"starttime_ticks": 999},
    )

    with pytest.raises(RuntimeError, match="reused"):
        launcher._signal_owned_process_group(
            FakeProcess(),  # type: ignore[arg-type]
            456,
            signal.SIGTERM,
        )


def test_cleanup_signals_orphaned_process_group_members_by_pidfd_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ExitedProcess:
        pid = 123

        def poll(self) -> int:
            return 1

    signals: list[tuple[int, int, int]] = []
    monkeypatch.setattr(
        launcher,
        "_process_group_members",
        lambda process_group: [
            {
                "pid": 124,
                "process_group": 123,
                "session_id": 123,
                "starttime_ticks": 457,
            }
        ],
    )
    monkeypatch.setattr(
        launcher,
        "_signal_exact_process",
        lambda pid, starttime, signal_number: signals.append(
            (pid, starttime, signal_number)
        ),
    )

    launcher._signal_owned_process_group(
        ExitedProcess(),  # type: ignore[arg-type]
        456,
        signal.SIGKILL,
    )

    assert signals == [(124, 457, signal.SIGKILL)]


def test_protocol_rejects_selection_mismatch() -> None:
    first, second = socket.socketpair()
    common = {
        "run_id": "12345678-1234-5678-1234-567812345678",
        "config_fingerprint": "a" * 64,
        "input_bundle_fingerprint": "b" * 64,
        "transport_arm": "cuda_copy",
        "timeout_seconds": 1.0,
    }
    with (
        JsonChannel(
            first,
            scenario="scenario-a",
            local_role="producer",
            local_rank=0,
            remote_role="consumer",
            remote_rank=0,
            **common,
        ) as sender,
        JsonChannel(
            second,
            scenario="scenario-b",
            local_role="consumer",
            local_rank=0,
            remote_role="producer",
            remote_rank=0,
            **common,
        ) as receiver,
    ):
        sender.send(
            PreparePayload(
                scenario="scenario-a",
                scenario_iteration=0,
                producer_request_id="producer",
                child_request_id="consumer",
                notification_id="bm90aWZpY2F0aW9u",
            ),
            iteration=0,
        )
        with pytest.raises(ProtocolError, match="scenario"):
            receiver.receive(PreparePayload, iteration=0)


def test_run_inputs_are_byte_identical_complete_and_read_only(tmp_path: Path) -> None:
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    (run_directory / "provenance").mkdir()

    copied_path, config, provenance = launcher._prepare_run_inputs(
        source_config_path=CONFIG_PATH,
        run_directory=run_directory,
        selection=_selection(),
    )

    assert copied_path.read_bytes() == CONFIG_PATH.read_bytes()
    assert len(config.scenarios) == 9
    assert len(config.transport_arms) == 2
    assert len(provenance["input_bundle_fingerprint"]) == 64
    assert copied_path.stat().st_mode & 0o222 == 0
    recorded = {record["path"] for record in provenance["files"]}
    assert recorded == {
        "inputs/config.json",
        "inputs/selection.json",
        "inputs/storm37b-p-handshake.json",
        "inputs/storm37b-d1-handshake.json",
    }


def test_run_input_rejects_symlink_components(tmp_path: Path) -> None:
    source = tmp_path / "source"
    outside = tmp_path / "outside"
    destination = tmp_path / "destination"
    source.mkdir()
    outside.mkdir()
    destination.mkdir()
    (outside / "manifest.json").write_text("{}")
    (source / "linked").symlink_to(outside, target_is_directory=True)

    with pytest.raises(RuntimeError, match="symlink"):
        launcher._copy_run_input(
            source,
            destination,
            "linked/manifest.json",
        )


@pytest.mark.parametrize(
    ("status", "suffix"),
    [
        (launcher.RunStatus.PASS, ""),
        (launcher.RunStatus.FAIL, ".FAIL"),
        (launcher.RunStatus.INVALID, ".INVALID"),
    ],
)
def test_run_publication_is_atomic_hashed_and_sealed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: launcher.RunStatus,
    suffix: str,
) -> None:
    monkeypatch.setattr(
        launcher,
        "validate_campaign",
        lambda *args, **kwargs: campaign_validator.CampaignValidation(
            disposition=status.value,
            passed=True,
            errors=(),
            counts={},
        ),
    )
    run_id = "12345678-1234-5678-1234-567812345678"
    build = tmp_path / f".{run_id}.partial.1"
    build.mkdir()
    (build / "artifact.txt").write_text("evidence")

    published = launcher._publish_run(
        build_directory=build,
        artifact_root=tmp_path,
        run_id=run_id,
        status=status,
    )

    assert published == tmp_path / f"{run_id}{suffix}"
    assert not build.exists()
    assert (published / "SHA256SUMS").read_text().endswith("  artifact.txt\n")
    assert published.stat().st_mode & 0o222 == 0
    assert (published / "artifact.txt").stat().st_mode & 0o222 == 0
    published.chmod(0o755)


def test_incomplete_campaign_is_explicitly_invalid(tmp_path: Path) -> None:
    (tmp_path / "run-status.json").write_text(json.dumps({"status": "INVALID"}))

    validation = validate_campaign(tmp_path)

    assert validation.disposition == "INVALID"
    assert validation.passed is False
    assert len(validation.errors) > 0


def test_complete_published_pass_campaign_validates_end_to_end(
    tmp_path: Path,
) -> None:
    published = build_published_pass_campaign(tmp_path / "runs")
    try:
        validation = validate_campaign(published)

        assert validation.passed is True
        assert validation.disposition == "PASS"
        assert validation.counts == {
            "iterations": 1,
            "source_pre_observations": 1,
            "source_post_observations": 1,
            "staging_observations": 1,
            "destination_observations": 1,
        }
    finally:
        make_tree_writable(published)


def test_complete_campaign_rejects_protected_listener_owner_drift(
    tmp_path: Path,
) -> None:
    published = build_published_pass_campaign(tmp_path / "runs")
    make_tree_writable(published)
    after_path = published / "protected-gpus-after.json"
    after = json.loads(after_path.read_text())
    after["listeners"][0]["owners"][0]["starttime_ticks"] += 1
    after_path.write_text(json.dumps(after, indent=2, sort_keys=True) + "\n")

    validation = validate_campaign(published)

    assert validation.passed is False
    assert any(
        "protected GPU process identities changed" in error
        for error in validation.errors
    )


@pytest.mark.parametrize(
    "payload",
    [
        '{"status":"INVALID","status":"PASS"}',
        '{"status":NaN}',
    ],
)
def test_validator_rejects_noncanonical_json(tmp_path: Path, payload: str) -> None:
    path = tmp_path / "artifact.json"
    path.write_text(payload)

    with pytest.raises(ValueError):
        campaign_validator._read_object(path)


def test_validator_rejects_empty_protected_process_coverage() -> None:
    errors: list[str] = []
    campaign_validator._validate_protected_snapshot(
        {
            "protected_devices": [6, 7],
            "boot_id": "boot",
            "device_uuids": {"6": "GPU-6", "7": "GPU-7"},
            "processes": [],
        },
        "before",
        errors,
    )

    assert any("coverage" in error for error in errors)


def test_internal_checksums_detect_content_and_membership_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        launcher,
        "validate_campaign",
        lambda *args, **kwargs: campaign_validator.CampaignValidation(
            disposition="PASS",
            passed=True,
            errors=(),
            counts={},
        ),
    )
    run_id = "12345678-1234-5678-1234-567812345678"
    build = tmp_path / ".partial"
    build.mkdir()
    (build / "artifact.txt").write_text("evidence")
    published = launcher._publish_run(
        build_directory=build,
        artifact_root=tmp_path,
        run_id=run_id,
        status=launcher.RunStatus.PASS,
    )
    assert validate_internal_checksums(published) == ()
    published.chmod(0o755)
    artifact = published / "artifact.txt"
    artifact.chmod(0o644)
    artifact.write_text("tampered")
    (published / "unlisted.txt").write_text("extra")

    errors = validate_internal_checksums(published)

    assert any("digest differs" in error for error in errors)
    assert any("membership differs" in error for error in errors)
