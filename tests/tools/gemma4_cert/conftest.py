from pathlib import Path

import pytest

from tools.gemma4_cert.artifact import ArtifactArm, CertificationPlan, PayloadSpec
from tools.gemma4_cert.common import canonical_json_bytes, sha256_bytes


def certification_plan(
    *,
    required_artifacts: tuple[str, ...] = (
        "client/requests.jsonl",
        "client/responses.jsonl",
        "client/choices.jsonl",
        "lineage.jsonl",
    ),
    planned_parents: int = 1,
    choices_per_parent: int = 3,
    arm_id: str = "arm-a",
) -> CertificationPlan:
    return CertificationPlan(
        protocol_id="protocol-a",
        protocol_version="1",
        run_id="run-a",
        arm_id=arm_id,
        created_at_utc="2026-07-12T00:00:00Z",
        hypothesis="the mechanism is correct",
        intervention="exercise the mechanism",
        state_predicate={"producer": "fresh"},
        payloads=(
            PayloadSpec(
                order=0,
                sha256=sha256_bytes(
                    canonical_json_bytes(
                        {
                            "model": "model",
                            "prompt": "payload",
                            "n": choices_per_parent,
                        }
                    )
                ),
                label="detector",
            ),
        ),
        seeds=tuple(17 + index for index in range(planned_parents)),
        planned_parents=planned_parents,
        choices_per_parent=choices_per_parent,
        stopping_rule={"parents": planned_parents},
        acceptance_criteria=("all rows are present",),
        required_artifacts=required_artifacts,
    )


@pytest.fixture
def arm(tmp_path: Path) -> ArtifactArm:
    return ArtifactArm.create(tmp_path / "cert", certification_plan())
