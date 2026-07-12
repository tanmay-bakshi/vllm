import subprocess
import threading
import traceback
from dataclasses import replace
from pathlib import Path

import pytest

from tools.gemma4_cert.artifact import ArmDisposition, ArtifactArm
from tools.gemma4_cert.attestation import (
    AttestationContext,
    AttestationInputs,
    NamedArtifact,
    RuntimeEvidence,
    build_attestation,
    native_artifact,
    repository_state,
    write_attestation,
)
from tools.gemma4_cert.common import CertificationError
from tools.gemma4_cert.validation import required_json_errors

from .conftest import certification_plan


def _git(repository: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
    )


def _repository(path: Path) -> Path:
    path.mkdir()
    _git(path, "init", "-q")
    (path / "tracked.txt").write_text("original\n")
    _git(path, "add", "tracked.txt")
    _git(
        path,
        "-c",
        "user.name=Certification Test",
        "-c",
        "user.email=certification@example.invalid",
        "commit",
        "-q",
        "-m",
        "Initial",
    )
    return path


def _evidence(**overrides: object) -> RuntimeEvidence:
    values = {
        "effective_config": {},
        "kv_specs": {},
        "feature_counters": {},
        "semantic_invariants": {},
        "graph_shapes": [],
        **overrides,
    }
    return RuntimeEvidence(**values)


def test_repository_hash_includes_tracked_diff_and_untracked_content(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path / "repository")
    clean = repository_state(repository)
    (repository / "tracked.txt").write_text("modified\n")
    (repository / "untracked.bin").write_bytes(b"first")

    first = repository_state(repository)
    (repository / "untracked.bin").write_bytes(b"second")
    second = repository_state(repository)

    assert not clean["dirty"]
    assert first["dirty"]
    assert first["dirty_diff_sha256"] != clean["dirty_diff_sha256"]
    assert first["dirty_diff_sha256"] != second["dirty_diff_sha256"]
    assert first["untracked_files"][0]["path"] == "untracked.bin"


def test_attestation_keeps_observed_and_supplied_evidence_separate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository(tmp_path / "repository")
    launcher = tmp_path / "launcher.sh"
    launcher.write_text("#!/bin/sh\n")
    monkeypatch.setenv("UNRELATED_SECRET", "must-not-appear")
    monkeypatch.setenv("VLLM_GEMMA4_API_KEY", "must-be-redacted")
    monkeypatch.setenv("VLLM_GEMMA4_COALESCED", "1")
    monkeypatch.setenv(
        "VLLM_KV_TRANSFER_CONFIG",
        '{"headers":[{"name":"authorization","value":"Bearer hidden"}]}',
    )
    context = AttestationContext(
        protocol_id="protocol-a",
        run_id="run-a",
        arm_id="arm-a",
        role="producer",
        engine_boot_uuid="engine-boot-a",
        rank=0,
        gpu_uuid="GPU-a",
    )
    inputs = AttestationInputs(
        observed=_evidence(
            effective_config={"role": "producer"},
            kv_specs={"groups": 13},
            feature_counters={"pulls": 7},
            semantic_invariants={"source_quiesced": True},
            graph_shapes=[[1, 1]],
        ),
        caller_supplied=_evidence(
            effective_config={
                "expected_role": "producer",
                "nested": {"access_token": "must-be-redacted"},
                "api.key": "punctuated-secret",
                "headers": [
                    {"name": "authorization", "value": "Bearer must-be-redacted"},
                    {"key": "private_key", "value": "must-be-redacted"},
                    {"name": "layout", "value": "allowed"},
                ],
            },
            kv_specs={"expected_groups": 13},
        ),
        software_versions={"nixl": "1.3.0-declared"},
    )

    record = build_attestation(
        context,
        repository,
        inputs,
        module_names=["tools.gemma4_cert.attestation"],
        named_artifacts=[NamedArtifact("launcher", launcher)],
    )

    observed = record["runtime_evidence"]["observed"]
    supplied = record["runtime_evidence"]["caller_supplied"]
    assert observed["effective_config"] == {
        "availability": "available",
        "value": {"role": "producer"},
    }
    assert observed["feature_counters"]["value"] == {"pulls": 7}
    assert supplied["effective_config"]["value"] == {
        "expected_role": "producer",
        "nested": {"access_token": "<redacted>"},
        "api.key": "<redacted>",
        "headers": [
            {"name": "authorization", "value": "<redacted>"},
            {"key": "private_key", "value": "<redacted>"},
            {"name": "layout", "value": "allowed"},
        ],
    }
    assert record["software_versions"]["caller_supplied"] == {"nixl": "1.3.0-declared"}
    assert record["loaded_modules"][0]["module"] == ("tools.gemma4_cert.attestation")
    assert record["named_artifacts"][0]["kind"] == "launcher"
    assert record["process"]["pid"] > 0
    assert record["process"]["argv_recorded"] is False
    assert "UNRELATED_SECRET" not in record["allowlisted_environment"]
    assert record["allowlisted_environment"]["VLLM_GEMMA4_API_KEY"] == {
        "redacted": True,
        "value": "<redacted>",
    }
    assert record["allowlisted_environment"]["VLLM_GEMMA4_COALESCED"] == {
        "redacted": False,
        "value": "1",
    }
    assert record["allowlisted_environment"]["VLLM_KV_TRANSFER_CONFIG"]["value"] == {
        "headers": [{"name": "authorization", "value": "<redacted>"}]
    }


def test_unavailable_runtime_evidence_is_explicit(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "repository")
    context = AttestationContext(
        protocol_id="protocol-a",
        run_id="run-a",
        arm_id="arm-a",
        role="offline-probe",
        engine_boot_uuid="probe-boot",
        rank=None,
        gpu_uuid=None,
    )
    unavailable = RuntimeEvidence(None, None, None, None, None)

    record = build_attestation(
        context,
        repository,
        AttestationInputs(unavailable, unavailable, {}),
        module_names=["tools.gemma4_cert.attestation"],
    )

    observed = record["runtime_evidence"]["observed"]
    assert observed["kv_specs"] == {
        "availability": "unavailable",
        "value": None,
    }
    assert observed["feature_counters"] == {
        "availability": "unavailable",
        "value": None,
    }


def test_arm_attestation_refuses_to_overwrite_evidence(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "repository")
    arm = ArtifactArm.create(tmp_path / "cert", certification_plan())
    context = AttestationContext(
        protocol_id="protocol-a",
        run_id="run-a",
        arm_id="arm-a",
        role="router",
        engine_boot_uuid="engine-boot-a",
        rank=None,
        gpu_uuid=None,
    )
    inputs = AttestationInputs(
        observed=_evidence(),
        caller_supplied=_evidence(),
        software_versions={},
    )

    with pytest.raises(CertificationError, match="active artifact writer"):
        write_attestation(
            arm,
            "attest/router.json",
            context,
            repository,
            inputs,
            module_names=["tools.gemma4_cert.attestation"],
        )

    with arm.writer_session() as writer:
        write_attestation(
            writer,
            "attest/router.json",
            context,
            repository,
            inputs,
            module_names=["tools.gemma4_cert.attestation"],
        )
        with pytest.raises(FileExistsError):
            write_attestation(
                writer,
                "attest/router.json",
                context,
                repository,
                inputs,
                module_names=["tools.gemma4_cert.attestation"],
            )


def test_native_artifact_records_gnu_build_id() -> None:
    readelf = Path("/usr/bin/readelf")
    executable = Path("/bin/ls")
    if not readelf.is_file() or not executable.is_file():
        pytest.skip("ELF build-ID tools are not available")

    record = native_artifact(executable)

    assert isinstance(record["gnu_build_id"], str)
    assert len(record["gnu_build_id"]) > 0


def test_cert_bound_attestation_writer_blocks_seal(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "repository")
    arm = ArtifactArm.create(tmp_path / "cert", certification_plan())
    context = AttestationContext(
        protocol_id=arm.plan.protocol_id,
        run_id=arm.plan.run_id,
        arm_id=arm.plan.arm_id,
        role="offline-probe",
        engine_boot_uuid="probe-boot",
        rank=None,
        gpu_uuid=None,
    )
    unavailable = RuntimeEvidence(None, None, None, None, None)
    inputs = AttestationInputs(unavailable, unavailable, {})
    writer_ready = threading.Event()
    release_writer = threading.Event()
    seal_started = threading.Event()
    seal_done = threading.Event()
    dispositions: list[ArmDisposition] = []
    thread_errors: list[str] = []

    def attest() -> None:
        try:
            with arm.writer_session() as writer:
                writer_ready.set()
                if not release_writer.wait(timeout=2.0):
                    raise TimeoutError("test did not release attestation writer")
                write_attestation(
                    writer,
                    "attest/offline-probe.json",
                    context,
                    repository,
                    inputs,
                    module_names=["tools.gemma4_cert.attestation"],
                )
        except Exception:
            thread_errors.append(traceback.format_exc())

    def seal() -> None:
        try:
            seal_started.set()
            dispositions.append(
                arm.seal(
                    ArmDisposition.VOID,
                    "offline attestation only",
                    make_read_only=False,
                )
            )
        except Exception:
            thread_errors.append(traceback.format_exc())
        finally:
            seal_done.set()

    writer_thread = threading.Thread(target=attest)
    writer_thread.start()
    assert writer_ready.wait(timeout=2.0)
    seal_thread = threading.Thread(target=seal)
    seal_thread.start()
    assert seal_started.wait(timeout=2.0)
    assert not seal_done.wait(timeout=0.05)
    release_writer.set()
    writer_thread.join(timeout=2.0)
    seal_thread.join(timeout=2.0)

    assert not writer_thread.is_alive()
    assert not seal_thread.is_alive()
    assert thread_errors == []
    assert dispositions == [ArmDisposition.VOID]
    assert (arm.path / "attest/offline-probe.json").is_file()
    assert "attest/offline-probe.json" in (arm.path / "SHA256SUMS").read_text()
    assert arm.verify().valid


def test_generated_attestation_satisfies_required_schema(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "repository")
    base_plan = certification_plan()
    plan = replace(
        base_plan,
        required_artifacts=(*base_plan.required_artifacts, "attest/router.json"),
    )
    arm = ArtifactArm.create(tmp_path / "cert", plan)
    context = AttestationContext(
        protocol_id=plan.protocol_id,
        run_id=plan.run_id,
        arm_id=plan.arm_id,
        role="router",
        engine_boot_uuid="router-boot",
        rank=None,
        gpu_uuid=None,
    )
    unavailable = RuntimeEvidence(None, None, None, None, None)

    with arm.writer_session() as writer:
        write_attestation(
            writer,
            "attest/router.json",
            context,
            repository,
            AttestationInputs(unavailable, unavailable, {}),
            module_names=["tools.gemma4_cert.attestation"],
        )

    assert (
        required_json_errors(
            arm.path,
            plan.required_artifacts,
            plan.protocol_id,
            plan.run_id,
            plan.arm_id,
        )
        == ()
    )
