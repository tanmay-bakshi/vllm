import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from tools.gemma4_cert.artifact import ArmDisposition, ArtifactArm, PayloadSpec
from tools.gemma4_cert.common import CertificationError, sha256_file

from .conftest import certification_plan


def test_create_hashes_plan_and_refuses_arm_reuse(tmp_path: Path) -> None:
    plan = certification_plan()
    arm = ArtifactArm.create(tmp_path / "cert", plan)

    plan_bytes = (arm.path / "plan.json").read_bytes()
    assert (arm.path / "plan.sha256").read_text() == (
        f"{hashlib.sha256(plan_bytes).hexdigest()}  plan.json\n"
    )
    assert (arm.path / "plan.json").stat().st_mode & 0o222 == 0
    assert (arm.path / "plan.sha256").stat().st_mode & 0o222 == 0

    with pytest.raises(FileExistsError):
        ArtifactArm.create(tmp_path / "cert", plan)


def test_plan_parent_count_must_equal_payload_seed_product() -> None:
    with pytest.raises(CertificationError, match="payload count multiplied"):
        replace(certification_plan(), planned_parents=2)


def test_parent_bindings_are_payload_major() -> None:
    base = certification_plan()
    second_payload = PayloadSpec(
        order=1,
        sha256="f" * 64,
        label="second",
    )
    plan = replace(
        base,
        payloads=(*base.payloads, second_payload),
        seeds=(17, 18),
        planned_parents=4,
    )

    bindings = plan.parent_bindings()

    assert [binding.payload_order for binding in bindings] == [0, 0, 1, 1]
    assert [binding.seed for binding in bindings] == [17, 18, 17, 18]


def test_mutated_plan_blocks_all_further_writes(tmp_path: Path) -> None:
    arm = ArtifactArm.create(tmp_path / "cert", certification_plan())
    plan = json.loads((arm.path / "plan.json").read_text())
    plan["hypothesis"] = "changed after execution began"
    (arm.path / "plan.json").chmod(0o644)
    (arm.path / "plan.json").write_text(json.dumps(plan))

    with pytest.raises(CertificationError, match="pre-execution hash"):
        arm.append_event("events/router.jsonl", {"event": "should-not-land"})


def test_jsonl_events_are_contiguously_sequenced(arm: ArtifactArm) -> None:
    first = arm.append_event("events/router.jsonl", {"event": "first"})
    second = arm.append_event("events/router.jsonl", {"event": "second"})

    assert first["sequence"] == 1
    assert second["sequence"] == 2
    rows = [
        json.loads(line)
        for line in (arm.path / "events/router.jsonl").read_text().splitlines()
    ]
    assert [row["sequence"] for row in rows] == [1, 2]


def test_missing_required_artifact_forces_invalid_and_manifest_verifies(
    tmp_path: Path,
) -> None:
    arm = ArtifactArm.create(
        tmp_path / "cert",
        certification_plan(required_artifacts=("client/requests.jsonl",)),
    )

    effective = arm.seal(
        ArmDisposition.PASS,
        "nominal execution ended",
        make_read_only=False,
    )

    assert effective is ArmDisposition.INVALID
    summary = json.loads((arm.path / "summary.json").read_text())
    assert summary["missing_required_artifacts"] == ["client/requests.jsonl"]
    manifest = (arm.path / "SHA256SUMS").read_text()
    assert "SHA256SUMS" not in manifest
    assert arm.verify().valid


def test_manifest_detects_tampering_and_unmanifested_files(tmp_path: Path) -> None:
    arm = ArtifactArm.create(tmp_path / "cert", certification_plan())
    arm.write_json("state/before.json", {"state": "fresh"})
    arm.seal(ArmDisposition.PASS, "complete", make_read_only=False)

    (arm.path / "state/before.json").write_text("tampered\n")
    (arm.path / "events/late.jsonl").write_text("{}\n")
    result = arm.verify()

    assert not result.valid
    assert "digest mismatch: state/before.json" in result.errors
    assert "unmanifested artifact: events/late.jsonl" in result.errors


def test_void_arm_is_preserved_and_cannot_be_reopened_for_writes(
    tmp_path: Path,
) -> None:
    plan = certification_plan()
    arm = ArtifactArm.create(tmp_path / "cert", plan)

    disposition = arm.seal(
        ArmDisposition.VOID,
        "state predicate was not met",
        make_read_only=False,
    )

    assert disposition is ArmDisposition.VOID
    with pytest.raises(CertificationError, match="sealed"):
        arm.append_event("events/router.jsonl", {"event": "late"})
    with pytest.raises(FileExistsError):
        ArtifactArm.create(tmp_path / "cert", plan)


def test_incomplete_recorder_forces_invalid_despite_present_declared_artifact(
    tmp_path: Path,
) -> None:
    arm = ArtifactArm.create(
        tmp_path / "cert",
        certification_plan(required_artifacts=("client/requests.jsonl",)),
    )
    arm.append_event("client/requests.jsonl", {"request": 0})

    effective = arm.seal(
        ArmDisposition.FAIL,
        "injected fault was detected",
        make_read_only=False,
    )

    assert effective is ArmDisposition.INVALID
    assert arm.verify().valid


def test_artifact_paths_cannot_escape_the_arm(arm: ArtifactArm) -> None:
    with pytest.raises(CertificationError, match="confined"):
        arm.write_json("../outside.json", {})


def test_reserved_event_fields_cannot_replace_recorder_metadata(
    arm: ArtifactArm,
) -> None:
    with pytest.raises(CertificationError, match="reserved event fields"):
        arm.append_event("events/router.jsonl", {"sequence": 99})


@pytest.mark.parametrize(
    "relative",
    [".arm.lock", "plan.json", "plan.sha256", "summary.json", "SHA256SUMS"],
)
def test_generic_writes_cannot_precreate_root_reserved_files(
    arm: ArtifactArm,
    relative: str,
) -> None:
    with pytest.raises(CertificationError, match="reserved"):
        arm.write_bytes(relative, b"hostile")
    with pytest.raises(CertificationError, match="reserved"):
        arm.append_event(relative, {"hostile": True})


def test_precreated_summary_prevents_sealing(arm: ArtifactArm) -> None:
    (arm.path / "summary.json").write_text("{}\n")

    with pytest.raises(FileExistsError):
        arm.seal(ArmDisposition.VOID, "void arm", make_read_only=False)


def test_verify_rejects_manifest_without_summary(arm: ArtifactArm) -> None:
    entries = sorted(
        path
        for path in arm.path.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    manifest = b"".join(
        f"{sha256_file(path)}  {path.relative_to(arm.path).as_posix()}\n".encode()
        for path in entries
    )
    (arm.path / "SHA256SUMS").write_bytes(manifest)

    result = arm.verify()

    assert not result.valid
    assert "summary.json is missing or not a regular file" in result.errors


def test_required_json_must_parse(tmp_path: Path) -> None:
    base_plan = certification_plan()
    plan = replace(
        base_plan,
        required_artifacts=(*base_plan.required_artifacts, "state/before.json"),
    )
    arm = ArtifactArm.create(tmp_path / "cert", plan)
    arm.write_bytes("state/before.json", b"not-json")

    disposition = arm.seal(
        ArmDisposition.PASS,
        "required JSON was malformed",
        make_read_only=False,
    )

    assert disposition is ArmDisposition.INVALID
    summary = json.loads((arm.path / "summary.json").read_text())
    assert "required JSON is invalid: state/before.json" in summary["invalid_json"]
    assert arm.verify().valid


def test_data_tmp_is_forbidden() -> None:
    with pytest.raises(CertificationError, match="/data/tmp"):
        ArtifactArm.create(Path("/data/tmp/cert"), certification_plan())
    with pytest.raises(CertificationError, match="/data/tmp"):
        ArtifactArm.open(Path("/data/tmp/protocol/run/arm"))


def test_open_rejects_noncanonical_arm_suffix(tmp_path: Path) -> None:
    arm = ArtifactArm.create(tmp_path / "cert", certification_plan())
    wrong_path = arm.path.parent / "wrong-arm"
    arm.path.rename(wrong_path)

    with pytest.raises(CertificationError, match="canonical protocol/run/arm"):
        ArtifactArm.open(wrong_path)
