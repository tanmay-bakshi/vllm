"""Command-line interface for certification evidence collection."""

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

from tools.gemma4_cert.artifact import ArmDisposition, ArtifactArm, load_plan
from tools.gemma4_cert.attestation import (
    AttestationContext,
    AttestationInputs,
    NamedArtifact,
    RuntimeEvidence,
    write_attestation,
)
from tools.gemma4_cert.common import CertificationError, read_json_file
from tools.gemma4_cert.recorder import (
    CertificationRecorder,
    UrllibTransport,
    load_request_plans,
)


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the Gemma 4 certification command line."""
    parser = _parser()
    parsed = parser.parse_args(arguments)
    try:
        return parsed.handler(parsed)
    except CertificationError as error:
        parser.error(str(error))
    return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m tools.gemma4_cert")
    commands = parser.add_subparsers(dest="command", required=True)

    initialize = commands.add_parser("init", help="create a new arm from a frozen plan")
    initialize.add_argument("--cert-root", type=Path, required=True)
    initialize.add_argument("--plan", type=Path, required=True)
    initialize.set_defaults(handler=_initialize)

    append = commands.add_parser("append", help="append one sequenced JSONL event")
    append.add_argument("--arm", type=Path, required=True)
    append.add_argument("--path", required=True)
    append.add_argument("--record", type=Path, required=True)
    append.set_defaults(handler=_append)

    seal = commands.add_parser("seal", help="seal an arm and write SHA256SUMS")
    seal.add_argument("--arm", type=Path, required=True)
    seal.add_argument(
        "--disposition", choices=[item.value for item in ArmDisposition], required=True
    )
    seal.add_argument("--reason", required=True)
    seal.add_argument("--details", type=Path)
    seal.set_defaults(handler=_seal)

    verify = commands.add_parser("verify", help="verify a sealed arm manifest")
    verify.add_argument("--arm", type=Path, required=True)
    verify.set_defaults(handler=_verify)

    attest = commands.add_parser("attest", help="emit an offline process attestation")
    attest.add_argument("--arm", type=Path, required=True)
    attest.add_argument("--path", required=True)
    attest.add_argument("--repository", type=Path, required=True)
    attest.add_argument("--role", required=True)
    attest.add_argument("--engine-boot-uuid", required=True)
    attest.add_argument("--rank", type=int)
    attest.add_argument("--gpu-uuid")
    attest.add_argument("--module", action="append")
    attest.add_argument("--native", action="append", type=Path, default=[])
    attest.add_argument("--artifact", action="append", default=[])
    attest.add_argument("--include-mapped-native-libraries", action="store_true")
    _evidence_arguments(attest, "supplied")
    attest.add_argument("--software-versions", type=Path)
    attest.set_defaults(handler=_attest)

    record = commands.add_parser("record", help="execute a predeclared request JSONL")
    record.add_argument("--arm", type=Path, required=True)
    record.add_argument("--endpoint", required=True)
    record.add_argument("--requests", type=Path, required=True)
    record.add_argument("--timeout-seconds", type=float, default=300.0)
    record.add_argument("--header", action="append", default=[])
    record.add_argument(
        "--router-attempt-header",
        default="x-gemma4-router-attempt-id",
    )
    record.set_defaults(handler=_record)
    return parser


def _evidence_arguments(parser: argparse.ArgumentParser, prefix: str) -> None:
    parser.add_argument(f"--{prefix}-effective-config", type=Path)
    parser.add_argument(f"--{prefix}-kv-specs", type=Path)
    parser.add_argument(f"--{prefix}-feature-counters", type=Path)
    parser.add_argument(f"--{prefix}-semantic-invariants", type=Path)
    parser.add_argument(f"--{prefix}-graph-shapes", type=Path)


def _initialize(arguments: argparse.Namespace) -> int:
    arm = ArtifactArm.create(arguments.cert_root, load_plan(arguments.plan))
    _print_json({"arm": str(arm.path), "plan_sha256": _plan_digest(arm.path)})
    return 0


def _append(arguments: argparse.Namespace) -> int:
    record = _json_mapping(arguments.record)
    enriched = ArtifactArm.open(arguments.arm).append_event(arguments.path, record)
    _print_json(enriched)
    return 0


def _seal(arguments: argparse.Namespace) -> int:
    details = _json_mapping(arguments.details) if arguments.details is not None else {}
    arm = ArtifactArm.open(arguments.arm)
    disposition = arm.seal(
        ArmDisposition(arguments.disposition),
        arguments.reason,
        details,
    )
    result = arm.verify()
    _print_json(
        {
            "disposition": disposition.value,
            "manifest_sha256": result.manifest_sha256,
            "verified": result.valid,
            "errors": list(result.errors),
        }
    )
    return 0 if result.valid else 1


def _verify(arguments: argparse.Namespace) -> int:
    result = ArtifactArm.open(arguments.arm).verify()
    _print_json(
        {
            "valid": result.valid,
            "manifest_sha256": result.manifest_sha256,
            "errors": list(result.errors),
        }
    )
    return 0 if result.valid else 1


def _attest(arguments: argparse.Namespace) -> int:
    arm = ArtifactArm.open(arguments.arm)
    context = AttestationContext(
        protocol_id=arm.plan.protocol_id,
        run_id=arm.plan.run_id,
        arm_id=arm.plan.arm_id,
        role=arguments.role,
        engine_boot_uuid=arguments.engine_boot_uuid,
        rank=arguments.rank,
        gpu_uuid=arguments.gpu_uuid,
    )
    inputs = AttestationInputs(
        observed=RuntimeEvidence(None, None, None, None, None),
        caller_supplied=_evidence(arguments, "supplied"),
        software_versions=_optional_string_mapping(arguments.software_versions),
    )
    artifacts = [_named_artifact(value) for value in arguments.artifact]
    with arm.writer_session() as writer:
        record = write_attestation(
            writer,
            arguments.path,
            context,
            arguments.repository,
            inputs,
            module_names=arguments.module,
            native_libraries=arguments.native,
            named_artifacts=artifacts,
            include_mapped_native_libraries=arguments.include_mapped_native_libraries,
        )
    repository_record = record.get("repository")
    if not isinstance(repository_record, dict):
        raise CertificationError("attestation repository record is malformed")
    _print_json(
        {
            "arm": str(arm.path),
            "path": arguments.path,
            "commit": repository_record.get("commit"),
            "dirty_diff_sha256": repository_record.get("dirty_diff_sha256"),
        }
    )
    return 0


def _record(arguments: argparse.Namespace) -> int:
    arm = ArtifactArm.open(arguments.arm)
    common_headers = _key_value_pairs(arguments.header)
    plans = [
        replace(plan, headers={**common_headers, **dict(plan.headers)})
        for plan in load_request_plans(arguments.requests)
    ]
    if len(plans) != arm.plan.planned_parents:
        raise CertificationError(
            "request-plan parent count does not match the frozen arm plan"
        )
    if any(
        len(plan.planned_choice_indices) != arm.plan.choices_per_parent
        for plan in plans
    ):
        raise CertificationError(
            "request-plan choice count does not match the frozen arm plan"
        )
    with arm.writer_session() as writer:
        recorder = CertificationRecorder(
            writer,
            arguments.endpoint,
            UrllibTransport(),
            timeout_seconds=arguments.timeout_seconds,
            router_attempt_header=arguments.router_attempt_header,
        )
        for plan in plans:
            recorder.record(plan)
    _print_json({"recorded_parents": len(plans), "arm": str(arm.path)})
    return 0


def _evidence(arguments: argparse.Namespace, prefix: str) -> RuntimeEvidence:
    return RuntimeEvidence(
        effective_config=_optional_mapping(
            getattr(arguments, f"{prefix}_effective_config")
        ),
        kv_specs=_optional_mapping(getattr(arguments, f"{prefix}_kv_specs")),
        feature_counters=_optional_integer_mapping(
            _optional_mapping(getattr(arguments, f"{prefix}_feature_counters")),
            f"{prefix} feature counters",
        ),
        semantic_invariants=_optional_mapping(
            getattr(arguments, f"{prefix}_semantic_invariants")
        ),
        graph_shapes=_optional_sequence(getattr(arguments, f"{prefix}_graph_shapes")),
    )


def _named_artifact(value: str) -> NamedArtifact:
    if "=" not in value:
        raise CertificationError("artifacts must use KIND=PATH")
    kind, raw_path = value.split("=", maxsplit=1)
    if len(kind) == 0 or len(raw_path) == 0:
        raise CertificationError("artifacts must use non-empty KIND=PATH")
    return NamedArtifact(kind=kind, path=Path(raw_path))


def _key_value_pairs(values: Sequence[str]) -> dict[str, str]:
    result = {}
    for value in values:
        if "=" not in value:
            raise CertificationError("headers must use NAME=VALUE")
        key, item = value.split("=", maxsplit=1)
        if len(key) == 0:
            raise CertificationError("header names cannot be empty")
        result[key] = item
    return result


def _json_mapping(path: Path) -> Mapping[str, object]:
    parsed = read_json_file(path)
    if not isinstance(parsed, dict):
        raise CertificationError(f"{path} must contain an object")
    return parsed


def _optional_mapping(path: Path | None) -> Mapping[str, object] | None:
    return None if path is None else _json_mapping(path)


def _optional_string_mapping(path: Path | None) -> Mapping[str, str]:
    values = _optional_mapping(path)
    if values is None:
        return {}
    if not all(
        isinstance(key, str) and isinstance(value, str) for key, value in values.items()
    ):
        raise CertificationError("software versions must map strings to strings")
    return {key: value for key, value in values.items() if isinstance(value, str)}


def _optional_integer_mapping(
    values: Mapping[str, object] | None,
    label: str,
) -> Mapping[str, int] | None:
    if values is None:
        return None
    if not all(
        isinstance(key, str) and isinstance(value, int) and not isinstance(value, bool)
        for key, value in values.items()
    ):
        raise CertificationError(f"{label} must map strings to integers")
    return {
        key: value
        for key, value in values.items()
        if isinstance(value, int) and not isinstance(value, bool)
    }


def _optional_sequence(path: Path | None) -> Sequence[object] | None:
    if path is None:
        return None
    parsed = read_json_file(path)
    if not isinstance(parsed, list):
        raise CertificationError(f"{path} must contain an array")
    return parsed


def _plan_digest(path: Path) -> str:
    line = (path / "plan.sha256").read_text(encoding="ascii")
    return line.split("  ", maxsplit=1)[0]


def _print_json(value: object) -> None:
    json.dump(value, sys.stdout, allow_nan=False, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
