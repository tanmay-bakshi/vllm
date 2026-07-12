"""Immutable, content-addressed certification arm artifacts."""

import fcntl
import json
import os
import re
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from tools.gemma4_cert.common import (
    CertificationError,
    append_sequenced_json,
    atomic_write_exclusive,
    canonical_json_bytes,
    read_json_file,
    safe_relative_path,
    sha256_bytes,
    sha256_file,
    utc_now,
    validate_sequenced_jsonl,
)
from tools.gemma4_cert.validation import (
    ParentBinding,
    recorder_completeness_errors,
    required_json_errors,
)

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RESERVED_FILES = {
    ".arm.lock",
    "plan.json",
    "plan.sha256",
    "summary.json",
    "SHA256SUMS",
}
_SUMMARY_FIELDS = {
    "schema_version",
    "requested_disposition",
    "disposition",
    "reason",
    "missing_required_artifacts",
    "invalid_json",
    "invalid_jsonl",
    "semantic_errors",
    "details",
    "sealed_at_utc",
}
_PROCESS_LOCKS: dict[Path, threading.RLock] = {}
_PROCESS_LOCKS_GUARD = threading.Lock()
_WRITER_CONSTRUCTION_TOKEN = object()


class ArmDisposition(StrEnum):
    """Terminal classification of a certification arm."""

    PASS = "PASS"
    FAIL = "FAIL"
    INVALID = "INVALID"
    VOID = "VOID"


@dataclass(frozen=True, slots=True)
class PayloadSpec:
    """One request template digest in its predeclared execution order."""

    order: int
    sha256: str
    label: str

    def __post_init__(self) -> None:
        if self.order < 0:
            raise CertificationError("payload order must be non-negative")
        if _SHA256.fullmatch(self.sha256) is None:
            raise CertificationError("payload SHA-256 must be lowercase hexadecimal")
        if len(self.label) == 0:
            raise CertificationError("payload label cannot be empty")

    def to_mapping(self) -> dict[str, object]:
        """Return the canonical JSON representation."""
        return {"order": self.order, "sha256": self.sha256, "label": self.label}


@dataclass(frozen=True, slots=True)
class CertificationPlan:
    """Predeclared inputs and acceptance contract for one certification arm."""

    protocol_id: str
    protocol_version: str
    run_id: str
    arm_id: str
    created_at_utc: str
    hypothesis: str
    intervention: str
    state_predicate: Mapping[str, object]
    payloads: tuple[PayloadSpec, ...]
    seeds: tuple[int, ...]
    planned_parents: int
    choices_per_parent: int
    stopping_rule: Mapping[str, object]
    acceptance_criteria: tuple[str, ...]
    required_artifacts: tuple[str, ...]

    def __post_init__(self) -> None:
        for label, value in (
            ("protocol_id", self.protocol_id),
            ("protocol_version", self.protocol_version),
            ("run_id", self.run_id),
            ("arm_id", self.arm_id),
        ):
            if _IDENTIFIER.fullmatch(value) is None:
                raise CertificationError(f"invalid {label}: {value!r}")
        for label, value in (
            ("created_at_utc", self.created_at_utc),
            ("hypothesis", self.hypothesis),
            ("intervention", self.intervention),
        ):
            if len(value) == 0:
                raise CertificationError(f"{label} cannot be empty")
        if self.planned_parents <= 0 or self.choices_per_parent <= 0:
            raise CertificationError(
                "planned request and choice counts must be positive"
            )
        orders = [payload.order for payload in self.payloads]
        if len(orders) == 0:
            raise CertificationError("at least one payload must be declared")
        if orders != list(range(len(orders))):
            raise CertificationError("payload order must be contiguous and canonical")
        labels = [payload.label for payload in self.payloads]
        if len(set(labels)) != len(labels):
            raise CertificationError("payload labels must be unique")
        if len(set(self.seeds)) != len(self.seeds):
            raise CertificationError("seeds must be unique")
        if len(self.seeds) == 0:
            raise CertificationError("at least one seed must be declared")
        expected_parents = len(self.payloads) * len(self.seeds)
        if self.planned_parents != expected_parents:
            raise CertificationError(
                "planned parents must equal payload count multiplied by seed count"
            )
        if len(self.acceptance_criteria) == 0:
            raise CertificationError("acceptance criteria cannot be empty")
        if len(self.state_predicate) == 0:
            raise CertificationError("state predicate cannot be empty")
        if len(self.stopping_rule) == 0:
            raise CertificationError("stopping rule cannot be empty")
        if len(self.required_artifacts) == 0:
            raise CertificationError("required artifacts cannot be empty")
        for relative in self.required_artifacts:
            path = safe_relative_path(relative)
            if path.as_posix() in _RESERVED_FILES:
                raise CertificationError(
                    f"reserved artifact cannot be required: {relative}"
                )
        if len(set(self.required_artifacts)) != len(self.required_artifacts):
            raise CertificationError("required artifact paths must be unique")
        canonical_json_bytes(self.to_mapping())

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "CertificationPlan":
        """Parse and type-check a plan mapping."""
        payload_values = _mapping_sequence(value, "payloads")
        payloads = tuple(
            PayloadSpec(
                order=_integer(payload, "order"),
                sha256=_string(payload, "sha256"),
                label=_string(payload, "label"),
            )
            for payload in payload_values
        )
        return cls(
            protocol_id=_string(value, "protocol_id"),
            protocol_version=_string(value, "protocol_version"),
            run_id=_string(value, "run_id"),
            arm_id=_string(value, "arm_id"),
            created_at_utc=_string(value, "created_at_utc"),
            hypothesis=_string(value, "hypothesis"),
            intervention=_string(value, "intervention"),
            state_predicate=_mapping(value, "state_predicate"),
            payloads=payloads,
            seeds=tuple(
                _integer_item(item, "seeds") for item in _sequence(value, "seeds")
            ),
            planned_parents=_integer(value, "planned_parents"),
            choices_per_parent=_integer(value, "choices_per_parent"),
            stopping_rule=_mapping(value, "stopping_rule"),
            acceptance_criteria=tuple(
                _string_item(item, "acceptance_criteria")
                for item in _sequence(value, "acceptance_criteria")
            ),
            required_artifacts=tuple(
                _string_item(item, "required_artifacts")
                for item in _sequence(value, "required_artifacts")
            ),
        )

    def to_mapping(self) -> dict[str, object]:
        """Return the canonical JSON representation."""
        return {
            "protocol_id": self.protocol_id,
            "protocol_version": self.protocol_version,
            "run_id": self.run_id,
            "arm_id": self.arm_id,
            "created_at_utc": self.created_at_utc,
            "hypothesis": self.hypothesis,
            "intervention": self.intervention,
            "state_predicate": dict(self.state_predicate),
            "payloads": [payload.to_mapping() for payload in self.payloads],
            "seeds": list(self.seeds),
            "planned_parents": self.planned_parents,
            "choices_per_parent": self.choices_per_parent,
            "stopping_rule": dict(self.stopping_rule),
            "acceptance_criteria": list(self.acceptance_criteria),
            "required_artifacts": list(self.required_artifacts),
        }

    def parent_binding(self, request_index: int) -> ParentBinding:
        """Return the payload-major payload and seed assignment for a parent."""
        if request_index < 0 or request_index >= self.planned_parents:
            raise CertificationError(
                f"request index is outside the plan: {request_index}"
            )
        seed_count = len(self.seeds)
        payload_order = request_index // seed_count
        seed = self.seeds[request_index % seed_count]
        payload = self.payloads[payload_order]
        return ParentBinding(
            request_index=request_index,
            payload_order=payload.order,
            payload_label=payload.label,
            request_template_sha256=payload.sha256,
            seed=seed,
        )

    def parent_bindings(self) -> tuple[ParentBinding, ...]:
        """Return every parent binding in payload-major execution order."""
        return tuple(
            self.parent_binding(request_index)
            for request_index in range(self.planned_parents)
        )


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """Result of checking a sealed arm against its manifest."""

    valid: bool
    errors: tuple[str, ...]
    manifest_sha256: str | None


class ArtifactWriter:
    """Active writer bound to an arm's shared lifecycle lock."""

    def __init__(self, arm: "ArtifactArm", token: object) -> None:
        if token is not _WRITER_CONSTRUCTION_TOKEN:
            raise CertificationError("artifact writers are created by writer_session()")
        self._arm = arm
        self._active = True

    @property
    def path(self) -> Path:
        """Return the bound arm path."""
        return self._arm.path

    @property
    def plan(self) -> CertificationPlan:
        """Return the frozen plan for the bound arm."""
        return self._arm.plan

    def write_json(self, relative: str, value: object) -> Path:
        """Write one exclusive JSON artifact inside the locked arm."""
        self._assert_active()
        return self._arm._write_json_unlocked(relative, value)

    def write_bytes(self, relative: str, value: bytes) -> Path:
        """Write one exclusive opaque artifact inside the locked arm."""
        self._assert_active()
        return self._arm._write_bytes_unlocked(relative, value)

    def append_event(
        self, relative: str, record: Mapping[str, object]
    ) -> dict[str, object]:
        """Append one sequenced JSONL event inside the locked arm."""
        self._assert_active()
        return self._arm._append_event_unlocked(relative, record)

    def _close(self) -> None:
        self._active = False

    def _assert_active(self) -> None:
        if not self._active:
            raise CertificationError("artifact writer session is no longer active")


class ArtifactArm:
    """Owns one append-only certification arm directory."""

    _SUBDIRECTORIES = ("attest", "state", "client", "events", "integrity", "metrics")

    def __init__(self, path: Path, plan: CertificationPlan) -> None:
        self.path = path
        self.plan = plan

    @classmethod
    def create(cls, cert_root: Path, plan: CertificationPlan) -> "ArtifactArm":
        """Create a never-before-used arm and freeze its plan hash."""
        resolved_root = cert_root.resolve()
        _validate_storage_root(resolved_root)
        path = resolved_root / plan.protocol_id / plan.run_id / plan.arm_id
        path.mkdir(parents=True, exist_ok=False)
        atomic_write_exclusive(path / ".arm.lock", b"")
        (path / ".arm.lock").chmod(0o444)
        for relative in cls._SUBDIRECTORIES:
            (path / relative).mkdir()
        plan_bytes = canonical_json_bytes(plan.to_mapping())
        atomic_write_exclusive(path / "plan.json", plan_bytes)
        atomic_write_exclusive(
            path / "plan.sha256", f"{sha256_bytes(plan_bytes)}  plan.json\n".encode()
        )
        (path / "plan.json").chmod(0o444)
        (path / "plan.sha256").chmod(0o444)
        return cls(path=path, plan=plan)

    @classmethod
    def open(cls, path: Path) -> "ArtifactArm":
        """Open an existing arm after validating its frozen plan."""
        resolved_path = path.resolve()
        _validate_storage_root(resolved_path)
        parsed = read_json_file(resolved_path / "plan.json")
        if not isinstance(parsed, dict):
            raise CertificationError("plan.json must contain an object")
        plan = CertificationPlan.from_mapping(parsed)
        expected_suffix = (plan.protocol_id, plan.run_id, plan.arm_id)
        if tuple(resolved_path.parts[-3:]) != expected_suffix:
            raise CertificationError(
                "arm path does not match its canonical protocol/run/arm identity"
            )
        arm = cls(path=resolved_path, plan=plan)
        arm._verify_plan_hash()
        return arm

    @property
    def sealed(self) -> bool:
        """Return whether this arm has a terminal manifest."""
        return (self.path / "SHA256SUMS").exists()

    @contextmanager
    def writer_session(self) -> Iterator[ArtifactWriter]:
        """Hold the shared lifecycle lock across a complete evidence-producing job."""
        with self._arm_lock(exclusive=False):
            self._assert_writable()
            writer = ArtifactWriter(self, _WRITER_CONSTRUCTION_TOKEN)
            try:
                yield writer
            finally:
                writer._close()

    def write_json(self, relative: str, value: object) -> Path:
        """Write one JSON artifact without replacing prior evidence."""
        with self._arm_lock(exclusive=False):
            return self._write_json_unlocked(relative, value)

    def _write_json_unlocked(self, relative: str, value: object) -> Path:
        destination = self._writable_destination(relative)
        atomic_write_exclusive(destination, canonical_json_bytes(value))
        return destination

    def write_bytes(self, relative: str, value: bytes) -> Path:
        """Write one opaque artifact without replacing prior evidence."""
        with self._arm_lock(exclusive=False):
            return self._write_bytes_unlocked(relative, value)

    def _write_bytes_unlocked(self, relative: str, value: bytes) -> Path:
        destination = self._writable_destination(relative)
        atomic_write_exclusive(destination, value)
        return destination

    def append_event(
        self, relative: str, record: Mapping[str, object]
    ) -> dict[str, object]:
        """Durably append one timestamped and monotonically sequenced JSONL record."""
        with self._arm_lock(exclusive=False):
            return self._append_event_unlocked(relative, record)

    def _append_event_unlocked(
        self, relative: str, record: Mapping[str, object]
    ) -> dict[str, object]:
        destination = self._writable_destination(relative)
        return append_sequenced_json(destination, record)

    def missing_required_artifacts(self) -> tuple[str, ...]:
        """Return predeclared artifacts that are absent or not regular files."""
        with self._arm_lock(exclusive=False):
            return self._missing_required_artifacts()

    def _missing_required_artifacts(self) -> tuple[str, ...]:
        missing = []
        for relative in self.plan.required_artifacts:
            candidate = self.path / safe_relative_path(relative)
            if candidate.is_symlink() or not candidate.is_file():
                missing.append(relative)
        return tuple(missing)

    def seal(
        self,
        disposition: ArmDisposition,
        reason: str,
        details: Mapping[str, object] | None = None,
        *,
        make_read_only: bool = True,
    ) -> ArmDisposition:
        """Classify an arm, write its summary, and freeze a complete manifest."""
        with self._arm_lock(exclusive=True):
            self._assert_writable()
            if len(reason) == 0:
                raise CertificationError("terminal reason cannot be empty")
            missing = self._missing_required_artifacts()
            json_errors = self._required_json_errors()
            jsonl_errors = self._jsonl_errors()
            semantic_errors = self._semantic_errors()
            effective = disposition
            if disposition is not ArmDisposition.VOID and (
                len(missing) > 0
                or len(json_errors) > 0
                or len(jsonl_errors) > 0
                or len(semantic_errors) > 0
            ):
                effective = ArmDisposition.INVALID
            summary = {
                "schema_version": 1,
                "requested_disposition": disposition.value,
                "disposition": effective.value,
                "reason": reason,
                "missing_required_artifacts": list(missing),
                "invalid_json": list(json_errors),
                "invalid_jsonl": list(jsonl_errors),
                "semantic_errors": list(semantic_errors),
                "details": dict(details or {}),
                "sealed_at_utc": utc_now(),
            }
            atomic_write_exclusive(
                self.path / "summary.json", canonical_json_bytes(summary)
            )
            entries = self._artifact_entries()
            manifest = b"".join(
                f"{sha256_file(path)}  {relative}\n".encode()
                for relative, path in entries
            )
            atomic_write_exclusive(self.path / "SHA256SUMS", manifest)
            if make_read_only:
                self._make_read_only()
            return effective

    def verify(self) -> VerificationResult:
        """Verify the frozen plan, manifest membership, and every file digest."""
        with self._arm_lock(exclusive=False):
            return self._verify_unlocked()

    def _verify_unlocked(self) -> VerificationResult:
        errors: list[str] = []
        try:
            self._verify_plan_hash()
        except CertificationError as error:
            errors.append(str(error))
        manifest_path = self.path / "SHA256SUMS"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            errors.append("SHA256SUMS is missing or not a regular file")
            return VerificationResult(False, tuple(errors), None)
        manifest_bytes = manifest_path.read_bytes()
        expected: dict[str, str] = {}
        try:
            manifest_text = manifest_bytes.decode("utf-8")
        except UnicodeDecodeError:
            errors.append("SHA256SUMS is not valid UTF-8")
            manifest_text = ""
        for line_number, line in enumerate(manifest_text.splitlines(), start=1):
            parts = line.split("  ", maxsplit=1)
            if len(parts) != 2 or _SHA256.fullmatch(parts[0]) is None:
                errors.append(f"invalid manifest line {line_number}")
                continue
            relative = parts[1]
            try:
                safe_relative_path(relative)
            except CertificationError as error:
                errors.append(str(error))
                continue
            if relative in expected:
                errors.append(f"duplicate manifest entry: {relative}")
            expected[relative] = parts[0]
        actual = {relative: path for relative, path in self._artifact_entries()}
        canonical_manifest = b"".join(
            f"{sha256_file(path)}  {relative}\n".encode()
            for relative, path in sorted(actual.items())
        )
        if manifest_bytes != canonical_manifest:
            errors.append(
                "SHA256SUMS is not canonical or does not cover every artifact"
            )
        for relative in sorted(set(expected) - set(actual)):
            errors.append(f"manifest file is missing: {relative}")
        for relative in sorted(set(actual) - set(expected)):
            errors.append(f"unmanifested artifact: {relative}")
        for relative in sorted(set(expected) & set(actual)):
            try:
                digest = sha256_file(actual[relative])
            except CertificationError as error:
                errors.append(str(error))
                continue
            if digest != expected[relative]:
                errors.append(f"digest mismatch: {relative}")
        errors.extend(self._summary_errors())
        return VerificationResult(
            valid=len(errors) == 0,
            errors=tuple(errors),
            manifest_sha256=sha256_bytes(manifest_bytes),
        )

    def _writable_destination(self, relative: str) -> Path:
        self._assert_writable()
        path = safe_relative_path(relative)
        if len(path.parts) == 1 and path.name in _RESERVED_FILES:
            raise CertificationError(f"artifact path is reserved: {relative}")
        destination = self.path / path
        current = self.path
        for part in path.parts[:-1]:
            current /= part
            if current.is_symlink():
                raise CertificationError(f"artifact parent is a symlink: {current}")
        if destination.is_symlink():
            raise CertificationError(f"artifact is a symlink: {destination}")
        return destination

    def _assert_writable(self) -> None:
        self._verify_plan_hash()
        if self.sealed:
            raise CertificationError("sealed arms cannot be modified")

    def _verify_plan_hash(self) -> None:
        plan_path = self.path / "plan.json"
        hash_path = self.path / "plan.sha256"
        if plan_path.is_symlink() or hash_path.is_symlink():
            raise CertificationError("plan files cannot be symlinks")
        try:
            hash_line = hash_path.read_text(encoding="ascii")
        except FileNotFoundError as error:
            raise CertificationError("plan.sha256 is missing") from error
        parts = hash_line.rstrip("\n").split("  ", maxsplit=1)
        if (
            len(parts) != 2
            or parts[1] != "plan.json"
            or _SHA256.fullmatch(parts[0]) is None
        ):
            raise CertificationError("plan.sha256 is malformed")
        if sha256_file(plan_path) != parts[0]:
            raise CertificationError(
                "plan.json no longer matches its pre-execution hash"
            )

    def _artifact_entries(self) -> list[tuple[str, Path]]:
        entries = []
        for directory, directory_names, file_names in os.walk(self.path):
            base = Path(directory)
            for directory_name in directory_names:
                candidate = base / directory_name
                if candidate.is_symlink():
                    raise CertificationError(
                        f"artifact directory is a symlink: {candidate}"
                    )
            for file_name in file_names:
                candidate = base / file_name
                relative = candidate.relative_to(self.path).as_posix()
                if relative == "SHA256SUMS":
                    continue
                if candidate.is_symlink() or not candidate.is_file():
                    raise CertificationError(f"invalid artifact file: {candidate}")
                entries.append((relative, candidate))
        return sorted(entries)

    def _make_read_only(self) -> None:
        for _, path in self._artifact_entries():
            path.chmod(0o444)
        (self.path / "SHA256SUMS").chmod(0o444)
        for directory, directory_names, _ in os.walk(self.path, topdown=False):
            for directory_name in directory_names:
                (Path(directory) / directory_name).chmod(0o555)
        self.path.chmod(0o555)

    def _jsonl_errors(self) -> tuple[str, ...]:
        errors = []
        for _, path in self._artifact_entries():
            if path.suffix != ".jsonl":
                continue
            try:
                validate_sequenced_jsonl(path)
            except CertificationError as error:
                errors.append(str(error))
        return tuple(errors)

    def _semantic_errors(self) -> tuple[str, ...]:
        return recorder_completeness_errors(
            self.path,
            self.plan.parent_bindings(),
            self.plan.choices_per_parent,
        )

    def _required_json_errors(self) -> tuple[str, ...]:
        return required_json_errors(
            self.path,
            self.plan.required_artifacts,
            self.plan.protocol_id,
            self.plan.run_id,
            self.plan.arm_id,
        )

    def _summary_errors(self) -> tuple[str, ...]:
        path = self.path / "summary.json"
        if path.is_symlink() or not path.is_file():
            return ("summary.json is missing or not a regular file",)
        raw = path.read_bytes()
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return ("summary.json is not valid JSON",)
        if not isinstance(parsed, dict):
            return ("summary.json must contain an object",)
        errors = []
        if set(parsed) != _SUMMARY_FIELDS:
            errors.append("summary.json has a noncanonical field set")
        try:
            canonical_summary = canonical_json_bytes(parsed)
        except (TypeError, UnicodeError, ValueError):
            errors.append("summary.json contains unsupported JSON values")
            canonical_summary = None
        if canonical_summary != raw:
            errors.append("summary.json is not canonically serialized")
        requested_value = parsed.get("requested_disposition")
        try:
            if not isinstance(requested_value, str):
                raise TypeError
            requested = ArmDisposition(requested_value)
        except (TypeError, ValueError):
            errors.append("summary.json has an invalid requested disposition")
            requested = None
        actual_missing = list(self._missing_required_artifacts())
        actual_json_errors = list(self._required_json_errors())
        actual_jsonl_errors = list(self._jsonl_errors())
        actual_semantic_errors = list(self._semantic_errors())
        if parsed.get("missing_required_artifacts") != actual_missing:
            errors.append("summary.json required-artifact result is incorrect")
        if parsed.get("invalid_json") != actual_json_errors:
            errors.append("summary.json JSON validation result is incorrect")
        if parsed.get("invalid_jsonl") != actual_jsonl_errors:
            errors.append("summary.json JSONL validation result is incorrect")
        if parsed.get("semantic_errors") != actual_semantic_errors:
            errors.append("summary.json semantic validation result is incorrect")
        if requested is not None:
            expected = requested
            if requested is not ArmDisposition.VOID and (
                len(actual_missing) > 0
                or len(actual_json_errors) > 0
                or len(actual_jsonl_errors) > 0
                or len(actual_semantic_errors) > 0
            ):
                expected = ArmDisposition.INVALID
            if parsed.get("disposition") != expected.value:
                errors.append("summary.json has an incorrect effective disposition")
        if parsed.get("schema_version") != 1:
            errors.append("summary.json has an invalid schema version")
        if not isinstance(parsed.get("reason"), str) or len(parsed["reason"]) == 0:
            errors.append("summary.json has an invalid reason")
        if not isinstance(parsed.get("details"), dict):
            errors.append("summary.json has invalid details")
        timestamp = parsed.get("sealed_at_utc")
        if not isinstance(timestamp, str) or not timestamp.endswith("Z"):
            errors.append("summary.json has an invalid seal timestamp")
        return tuple(errors)

    @contextmanager
    def _arm_lock(self, *, exclusive: bool) -> Iterator[None]:
        lock_path = self.path / ".arm.lock"
        if lock_path.is_symlink() or not lock_path.is_file():
            raise CertificationError("arm lock is missing or invalid")
        process_lock = _process_lock(lock_path)
        with process_lock:
            descriptor = os.open(lock_path, os.O_RDONLY)
            operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
            try:
                fcntl.flock(descriptor, operation)
                yield
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)


def _process_lock(path: Path) -> threading.RLock:
    resolved = path.resolve()
    with _PROCESS_LOCKS_GUARD:
        lock = _PROCESS_LOCKS.get(resolved)
        if lock is None:
            lock = threading.RLock()
            _PROCESS_LOCKS[resolved] = lock
        return lock


def _validate_storage_root(path: Path) -> None:
    forbidden_root = Path("/data/tmp")
    if path == forbidden_root or forbidden_root in path.parents:
        raise CertificationError("certification artifacts cannot live under /data/tmp")


def load_plan(path: Path) -> CertificationPlan:
    """Load and validate a plan from a JSON file."""
    parsed = read_json_file(path)
    if not isinstance(parsed, dict):
        raise CertificationError("plan input must contain an object")
    return CertificationPlan.from_mapping(parsed)


def _mapping(value: Mapping[str, object], name: str) -> Mapping[str, object]:
    item = value.get(name)
    if not isinstance(item, dict):
        raise CertificationError(f"{name} must be an object")
    return item


def _mapping_sequence(
    value: Mapping[str, object], name: str
) -> tuple[Mapping[str, object], ...]:
    return tuple(_mapping_item(item, name) for item in _sequence(value, name))


def _mapping_item(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise CertificationError(f"each {name} item must be an object")
    return value


def _sequence(value: Mapping[str, object], name: str) -> Sequence[object]:
    item = value.get(name)
    if not isinstance(item, list):
        raise CertificationError(f"{name} must be an array")
    return item


def _string(value: Mapping[str, object], name: str) -> str:
    return _string_item(value.get(name), name)


def _string_item(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise CertificationError(f"{name} must contain strings")
    return value


def _integer(value: Mapping[str, object], name: str) -> int:
    return _integer_item(value.get(name), name)


def _integer_item(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CertificationError(f"{name} must contain integers")
    return value
