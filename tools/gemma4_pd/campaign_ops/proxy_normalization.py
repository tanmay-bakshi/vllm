#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Prove an ephemeral experiment-lane proxy candidate without mutating service."""

import argparse
import ctypes
import hashlib
import http.client
import json
import math
import os
import resource
import signal
import socket
import stat
import subprocess
import sys
import time
import traceback
import urllib.parse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from types import FrameType, TracebackType
from typing import Protocol, cast
from uuid import UUID

from tools.gemma4_pd.experiment_lane_lock import (
    EXPERIMENT_LANE_LOCK,
    ExperimentLaneLock,
    ExperimentLaneLockError,
)

SCHEMA_VERSION = 2
CANONICAL_HOST = "127.0.0.1"
CANONICAL_PORT = 8812
DEFAULT_CANDIDATE_PORT = 18812
DEFAULT_PROTECTED_PORTS = (8000, 8810, 8811, 8813, 8815, 8820, 8821)
DEFAULT_IDLE_METRICS_URLS = (
    "http://127.0.0.1:8810/metrics",
    "http://127.0.0.1:8811/metrics",
    "http://127.0.0.1:8815/metrics",
)
GPU_CONFIRMATION = "SEND_ONE_DETERMINISTIC_P_TO_D_GPU_GATE"
_PR_SET_PDEATHSIG = 1
_PRCTL: Callable[[int, int, int, int, int], int] | None = None
if sys.platform == "linux":
    _PRCTL = cast(
        Callable[[int, int, int, int, int], int],
        ctypes.CDLL(None, use_errno=True).prctl,
    )
_STATE_FIELDS = frozenset(
    {
        "before_sha256",
        "candidate_identity",
        "config_sha256",
        "evidence",
        "phase",
        "schema_version",
    }
)
_CONFIG_FIELDS = frozenset(
    {
        "candidate_port",
        "canonical_port",
        "decoder_host",
        "decoder_port",
        "idle_metrics_urls",
        "legacy_pid",
        "model",
        "prefiller_host",
        "prefiller_port",
        "protected_ports",
        "proxy_script",
        "proxy_script_sha256",
        "python",
        "python_entrypoint_chain",
        "python_executable",
        "python_sha256",
        "pyvenv_config",
        "pyvenv_config_sha256",
        "readiness_timeout_seconds",
        "schema_version",
        "workdir",
    }
)
_IDENTITY_FIELDS = frozenset(
    {"boot_id", "pid", "process_group_id", "session_id", "start_time_ticks"}
)
_EVIDENCE_FIELDS = {
    "preflight.json": frozenset(
        {
            "application_requests_sent",
            "candidate_port_absent",
            "legacy_identity",
            "protected",
            "schema_version",
        }
    ),
    "semantic-launch.json": frozenset(
        {
            "candidate_application_requests_before_semantic",
            "candidate_identity",
            "candidate_listener",
            "candidate_process",
            "idle",
            "legacy_identity",
            "legacy_listener",
            "protected",
            "schema_version",
        }
    ),
    "semantic-gate.json": frozenset(
        {
            "candidate_identity",
            "candidate_listener_after",
            "candidate_logs",
            "candidate_process_after",
            "cleanup",
            "idle_after",
            "idle_before",
            "legacy_listener_after",
            "legacy_listener_before",
            "protected_after",
            "protected_before",
            "schema_version",
            "semantic",
        }
    ),
    "semantic-gate-failure.json": frozenset(
        {
            "candidate_identity",
            "candidate_logs",
            "cleanup",
            "failure_traceback",
            "schema_version",
        }
    ),
    "semantic-preparation-failure.json": frozenset(
        {
            "candidate_identity",
            "candidate_logs",
            "cleanup",
            "failure_traceback",
            "schema_version",
        }
    ),
    "semantic-cleanup-failure.json": frozenset(
        {
            "candidate_identity",
            "cleanup_traceback",
            "failure_traceback",
            "schema_version",
        }
    ),
}
_RESOURCE_NAMES = tuple(
    name
    for name in (
        "RLIMIT_AS",
        "RLIMIT_CORE",
        "RLIMIT_CPU",
        "RLIMIT_DATA",
        "RLIMIT_FSIZE",
        "RLIMIT_LOCKS",
        "RLIMIT_MEMLOCK",
        "RLIMIT_MSGQUEUE",
        "RLIMIT_NICE",
        "RLIMIT_NOFILE",
        "RLIMIT_NPROC",
        "RLIMIT_RSS",
        "RLIMIT_RTPRIO",
        "RLIMIT_RTTIME",
        "RLIMIT_SIGPENDING",
        "RLIMIT_STACK",
    )
    if hasattr(resource, name)
)


class NormalizationError(RuntimeError):
    """Report an unsafe or inconclusive proxy-normalization operation."""


class NormalizationInterrupted(NormalizationError):
    """Report the first operator signal delivered during a semantic action."""

    signal_number: int

    def __init__(self, signal_number: int) -> None:
        """Create one signal interruption.

        :param signal_number: First delivered signal number.
        """
        self.signal_number = signal_number
        super().__init__(f"semantic action interrupted by signal {signal_number}")


class FirstSignalGuard:
    """Convert the first operator signal into one deferred cleanup exception."""

    _exception_deferred: bool
    _previous: dict[int, object]
    _signal_number: int | None

    def __init__(self) -> None:
        """Create an inactive first-signal boundary."""
        self._exception_deferred = False
        self._previous = {}
        self._signal_number = None

    def _handle(self, signal_number: int, frame: FrameType | None) -> None:
        """Latch the first signal and raise unless interruption is deferred.

        :param signal_number: Delivered signal number.
        :param frame: Interrupted Python frame.
        :raises NormalizationInterrupted: For the first non-deferred signal.
        """
        del frame
        if self._signal_number is not None:
            return
        self._signal_number = signal_number
        if self._exception_deferred:
            return
        raise NormalizationInterrupted(signal_number)

    def __enter__(self) -> "FirstSignalGuard":
        """Install handlers for HUP, INT, and TERM.

        :returns: Active signal boundary.
        """
        if len(self._previous) > 0:
            raise NormalizationError("first-signal guard is already active")
        for managed_signal in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
            self._previous[managed_signal] = signal.signal(managed_signal, self._handle)
        return self

    def defer_exception(self) -> None:
        """Record the first signal without interrupting the active operation."""
        if len(self._previous) == 0:
            raise NormalizationError("first-signal guard is not active")
        self._exception_deferred = True

    def raise_pending(self) -> None:
        """Raise a signal recorded during a deferred operation.

        :raises NormalizationInterrupted: If an operator signal is pending.
        """
        self._exception_deferred = False
        if self._signal_number is not None:
            raise NormalizationInterrupted(self._signal_number)

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Restore the process signal dispositions.

        :param exception_type: Active exception type, when any.
        :param exception: Active exception, when any.
        :param traceback: Active traceback, when any.
        """
        del exception_type, exception, traceback
        for managed_signal, previous in self._previous.items():
            signal.signal(managed_signal, previous)
        self._previous.clear()
        self._exception_deferred = False


class Phase(str, Enum):
    """Name the durable proxy-normalization transaction states."""

    BEFORE_SEALED = "before_sealed"
    PREPARED = "prepared"
    SEMANTIC_RUNNING = "semantic_running"
    SEMANTIC_PROVEN = "semantic_proven"
    SEMANTIC_FAILED = "semantic_failed"
    FAILED = "failed"


_PHASE_EVIDENCE = {
    Phase.BEFORE_SEALED: (),
    Phase.PREPARED: ("preflight.json",),
    Phase.SEMANTIC_RUNNING: ("preflight.json", "semantic-launch.json"),
    Phase.SEMANTIC_PROVEN: (
        "preflight.json",
        "semantic-launch.json",
        "semantic-gate.json",
    ),
    Phase.SEMANTIC_FAILED: (
        "preflight.json",
        "semantic-launch.json",
        "semantic-gate-failure.json",
    ),
}


@dataclass(frozen=True, slots=True)
class ProcessIdentity:
    """Identify one process incarnation without trusting a reusable PID.

    :ivar boot_id: Canonical Linux boot UUID.
    :ivar pid: Process identifier.
    :ivar process_group_id: Process-group identifier.
    :ivar session_id: Session identifier.
    :ivar start_time_ticks: Process start time after boot in clock ticks.
    """

    boot_id: str
    pid: int
    process_group_id: int
    session_id: int
    start_time_ticks: int


@dataclass(frozen=True, slots=True)
class HttpResponse:
    """Contain one bounded local HTTP response.

    :ivar status: HTTP status code.
    :ivar headers: Canonical lowercase response headers.
    :ivar body: Complete bounded response body.
    """

    status: int
    headers: tuple[tuple[str, str], ...]
    body: bytes


@dataclass(frozen=True, slots=True)
class ObservedLegacyLaunch:
    """Record the exact launch identity observed on the live legacy proxy.

    :ivar argv: Complete observed argument vector.
    :ivar workdir: Exact observed working-directory spelling.
    :ivar proxy_script: Current resolved legacy proxy source path.
    :ivar proxy_script_sha256: Current legacy proxy source digest.
    """

    argv: tuple[bytes, ...]
    workdir: bytes
    proxy_script: Path
    proxy_script_sha256: str

    def to_record(self) -> dict[str, object]:
        """Serialize the exact observed legacy launch identity.

        :returns: Exact JSON legacy-launch record.
        """
        return {
            "argv_hex": [value.hex() for value in self.argv],
            "proxy_script_hex": os.fsencode(self.proxy_script).hex(),
            "proxy_script_sha256": self.proxy_script_sha256,
            "workdir_hex": self.workdir.hex(),
        }


@dataclass(frozen=True, slots=True)
class LaunchContract:
    """Define the normalized proxy process independent of its listening port.

    :ivar python: Attested virtual-environment Python entrypoint.
    :ivar python_executable: Resolved current Python executable.
    :ivar proxy_script: Canonical proxy source file.
    :ivar workdir: Canonical physical working directory.
    :ivar prefiller_host: Prefill service host passed to the proxy.
    :ivar prefiller_port: Prefill service port passed to the proxy.
    :ivar decoder_host: Decode service host passed to the proxy.
    :ivar decoder_port: Decode service port passed to the proxy.
    :ivar environment: Exact legacy environment replayed by the normalized proxy.
    :ivar umask: Exact legacy process umask.
    :ivar resource_limits: Exact legacy resource-limit vector.
    """

    python: Path
    python_executable: Path
    proxy_script: Path
    workdir: Path
    prefiller_host: str
    prefiller_port: int
    decoder_host: str
    decoder_port: int
    environment: tuple[tuple[bytes, bytes], ...]
    umask: int
    resource_limits: tuple[tuple[str, int, int], ...]

    def argv(self, port: int) -> tuple[bytes, ...]:
        """Build the exact normalized argument vector for one port.

        :param port: Loopback listening port.
        :returns: Exact byte argument vector.
        """
        return (
            os.fsencode(self.python),
            os.fsencode(self.proxy_script),
            b"--host",
            CANONICAL_HOST.encode(),
            b"--port",
            str(port).encode(),
            b"--prefiller-hosts",
            self.prefiller_host.encode(),
            b"--prefiller-ports",
            str(self.prefiller_port).encode(),
            b"--decoder-hosts",
            self.decoder_host.encode(),
            b"--decoder-ports",
            str(self.decoder_port).encode(),
        )

    def to_record(self, candidate_port: int) -> dict[str, object]:
        """Serialize the exact candidate launch contract.

        :param candidate_port: Alternate loopback candidate port.
        :returns: Exact JSON launch-contract record.
        """
        return {
            "argv_hex": [value.hex() for value in self.argv(candidate_port)],
            "environment": [
                {"key_hex": key.hex(), "value_hex": value.hex()}
                for key, value in self.environment
            ],
            "resource_limits": [
                {"hard": hard, "name": name, "soft": soft}
                for name, soft, hard in self.resource_limits
            ],
            "umask": self.umask,
        }


@dataclass(frozen=True, slots=True)
class NormalizationConfig:
    """Bind every operator-selected normalization input.

    :ivar legacy_pid: Expected legacy proxy PID.
    :ivar candidate_port: Alternate loopback candidate port.
    :ivar python: Attested virtual-environment Python entrypoint.
    :ivar python_executable: Resolved current Python executable.
    :ivar python_entrypoint_chain: Exact symlink chain ending at the executable.
    :ivar pyvenv_config: Canonical virtual-environment configuration.
    :ivar pyvenv_config_sha256: Virtual-environment configuration digest.
    :ivar proxy_script: Canonical proxy source file.
    :ivar workdir: Canonical proxy working directory.
    :ivar prefiller_host: Prefill service host.
    :ivar prefiller_port: Prefill service port.
    :ivar decoder_host: Decode service host.
    :ivar decoder_port: Decode service port.
    :ivar model: Served model name used only by the explicit semantic gate.
    :ivar protected_ports: Listener identities that must never drift.
    :ivar idle_metrics_urls: Engine metrics endpoints that must remain idle.
    :ivar readiness_timeout_seconds: Bounded local readiness timeout.
    :ivar python_sha256: Current Python executable digest.
    :ivar proxy_script_sha256: Proxy source digest.
    """

    legacy_pid: int
    candidate_port: int
    python: Path
    python_executable: Path
    python_entrypoint_chain: tuple[tuple[str, str, int, int], ...]
    pyvenv_config: Path
    pyvenv_config_sha256: str
    proxy_script: Path
    workdir: Path
    prefiller_host: str
    prefiller_port: int
    decoder_host: str
    decoder_port: int
    model: str
    protected_ports: tuple[int, ...]
    idle_metrics_urls: tuple[str, ...]
    readiness_timeout_seconds: float
    python_sha256: str
    proxy_script_sha256: str

    def to_record(self) -> dict[str, object]:
        """Serialize the immutable public configuration.

        :returns: Strict configuration record.
        """
        return {
            "candidate_port": self.candidate_port,
            "canonical_port": CANONICAL_PORT,
            "decoder_host": self.decoder_host,
            "decoder_port": self.decoder_port,
            "idle_metrics_urls": list(self.idle_metrics_urls),
            "legacy_pid": self.legacy_pid,
            "model": self.model,
            "prefiller_host": self.prefiller_host,
            "prefiller_port": self.prefiller_port,
            "protected_ports": list(self.protected_ports),
            "proxy_script": str(self.proxy_script),
            "proxy_script_sha256": self.proxy_script_sha256,
            "python": str(self.python),
            "python_entrypoint_chain": [
                {
                    "device": device,
                    "inode": inode,
                    "path": path,
                    "target_hex": target_hex,
                }
                for path, target_hex, device, inode in self.python_entrypoint_chain
            ],
            "python_executable": str(self.python_executable),
            "python_sha256": self.python_sha256,
            "pyvenv_config": str(self.pyvenv_config),
            "pyvenv_config_sha256": self.pyvenv_config_sha256,
            "readiness_timeout_seconds": self.readiness_timeout_seconds,
            "schema_version": SCHEMA_VERSION,
            "workdir": str(self.workdir),
        }


@dataclass(frozen=True, slots=True)
class TransactionState:
    """Contain the mutable head of one append-only evidence transaction.

    :ivar phase: Current durable state-machine phase.
    :ivar config_sha256: Immutable configuration digest.
    :ivar before_sha256: Immutable pre-mutation evidence digest.
    :ivar candidate_identity: Active ephemeral candidate identity, when any.
    :ivar evidence: Ordered immutable evidence names and digests.
    """

    phase: Phase
    config_sha256: str
    before_sha256: str
    candidate_identity: ProcessIdentity | None
    evidence: tuple[tuple[str, str], ...]

    def to_record(self) -> dict[str, object]:
        """Serialize the exact mutable state schema.

        :returns: Strict state record.
        """
        identity = (
            None if self.candidate_identity is None else asdict(self.candidate_identity)
        )
        return {
            "before_sha256": self.before_sha256,
            "candidate_identity": identity,
            "config_sha256": self.config_sha256,
            "evidence": [
                {"name": name, "sha256": digest} for name, digest in self.evidence
            ],
            "phase": self.phase.value,
            "schema_version": SCHEMA_VERSION,
        }


class NormalizationSystem(Protocol):
    """Define host operations needed by the normalization state machine."""

    def capture_process(self, pid: int) -> dict[str, object]:
        """Capture complete process, mapping, FD, and listener evidence."""
        ...

    def identity(self, pid: int) -> ProcessIdentity:
        """Capture one PID-reuse-safe process identity."""
        ...

    def require_identity(self, expected: ProcessIdentity) -> None:
        """Require one process incarnation to remain exact."""
        ...

    def process_group_identities(self, pgid: int) -> tuple[ProcessIdentity, ...]:
        """Capture the exact members of one process group."""
        ...

    def listener_snapshot(self, port: int) -> dict[str, object]:
        """Capture one exact TCP listener and all owning process identities."""
        ...

    def protected_snapshot(self, ports: tuple[int, ...]) -> dict[str, object]:
        """Capture immutable listener identities for protected ports."""
        ...

    def require_port_absent(self, port: int) -> None:
        """Require no TCP listener on one alternate port."""
        ...

    def idle_snapshot(self, urls: tuple[str, ...]) -> dict[str, object]:
        """Require zero running and waiting requests on every engine."""
        ...

    def launch(
        self,
        contract: LaunchContract,
        port: int,
        stdout_path: Path,
        stderr_path: Path,
    ) -> ProcessIdentity:
        """Launch one normalized detached proxy process."""
        ...

    def wait_for_listener_gate(
        self,
        contract: LaunchContract,
        identity: ProcessIdentity,
        port: int,
        timeout_seconds: float,
    ) -> dict[str, object]:
        """Require launch identity and exact loopback listener ownership."""
        ...

    def require_launch_contract(
        self,
        contract: LaunchContract,
        identity: ProcessIdentity,
        port: int,
    ) -> dict[str, object]:
        """Capture and require one exact running launch contract."""
        ...

    def stop_group(
        self,
        identities: tuple[ProcessIdentity, ...],
        timeout_seconds: float,
    ) -> None:
        """Terminate only one exact process group and prove its disappearance."""
        ...

    def semantic_gate(
        self,
        port: int,
        model: str,
        timeout_seconds: float,
    ) -> dict[str, object]:
        """Send deterministic non-streaming and streaming P-to-D requests."""
        ...


def _canonical_json(value: object) -> bytes:
    """Encode one strict canonical JSON value.

    :param value: JSON-compatible value.
    :returns: Canonical UTF-8 payload ending in one newline.
    :raises NormalizationError: If the value is not strict JSON.
    """
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode()
    except (TypeError, ValueError) as error:
        raise NormalizationError("evidence is not strict JSON") from error


def _sha256_bytes(payload: bytes) -> str:
    """Hash one byte payload.

    :param payload: Bytes to hash.
    :returns: Lowercase SHA-256 digest.
    """
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    """Hash one regular file through a no-follow descriptor.

    :param path: File to hash.
    :returns: Lowercase SHA-256 digest.
    :raises NormalizationError: If the path is unsafe or unreadable.
    """
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as error:
        raise NormalizationError(f"cannot open file for hashing: {path}") from error
    digest = hashlib.sha256()
    try:
        file_stat = os.fstat(descriptor)
        if stat.S_ISREG(file_stat.st_mode) is False:
            raise NormalizationError(f"hash input is not a regular file: {path}")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if len(chunk) == 0:
                break
            digest.update(chunk)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _require_digest(value: object, *, label: str) -> str:
    """Require a lowercase SHA-256 identity.

    :param value: Candidate JSON value.
    :param label: Diagnostic field name.
    :returns: Validated digest.
    :raises NormalizationError: If the digest is malformed.
    """
    if type(value) is not str or len(value) != 64:
        raise NormalizationError(f"{label} is not a SHA-256 digest")
    if any(character not in "0123456789abcdef" for character in value):
        raise NormalizationError(f"{label} is not lowercase hexadecimal")
    return value


def _require_object(
    value: object,
    *,
    label: str,
    fields: frozenset[str] | None = None,
) -> dict[str, object]:
    """Require one strict JSON object and optional exact field set.

    :param value: Candidate value.
    :param label: Diagnostic field name.
    :param fields: Exact required field set.
    :returns: Validated object.
    :raises NormalizationError: If the object or schema differs.
    """
    if type(value) is not dict:
        raise NormalizationError(f"{label} is not an object")
    record = cast(dict[str, object], value)
    if fields is not None and set(record) != fields:
        raise NormalizationError(f"{label} field set differs")
    return record


def _require_private_directory(path: Path, *, create: bool) -> Path:
    """Require one normalized private physical directory.

    :param path: Candidate directory.
    :param create: Whether to create the final directory exclusively.
    :returns: Validated directory.
    :raises NormalizationError: If the directory boundary is unsafe.
    """
    if path.is_absolute() is False or path != Path(os.path.normpath(path)):
        raise NormalizationError("artifact directory must be absolute and normalized")
    if create:
        parent = path.parent
        try:
            parent_resolved = parent.resolve(strict=True)
            parent_stat = parent.lstat()
        except OSError as error:
            raise NormalizationError("artifact parent is unavailable") from error
        if (
            parent_resolved != parent
            or stat.S_ISDIR(parent_stat.st_mode) is False
            or stat.S_IMODE(parent_stat.st_mode) != 0o700
            or parent_stat.st_uid != os.geteuid()
        ):
            raise NormalizationError(
                "artifact parent must be a private euid-owned physical directory"
            )
        try:
            path.mkdir(mode=0o700)
        except OSError as error:
            raise NormalizationError("cannot create artifact directory") from error
        _fsync_directory(parent)
    try:
        resolved = path.resolve(strict=True)
        path_stat = path.lstat()
    except OSError as error:
        raise NormalizationError("artifact directory is unavailable") from error
    if (
        resolved != path
        or stat.S_ISDIR(path_stat.st_mode) is False
        or stat.S_IMODE(path_stat.st_mode) != 0o700
        or path_stat.st_uid != os.geteuid()
    ):
        raise NormalizationError(
            "artifact directory is not private, euid-owned, and physical"
        )
    return path


def _fsync_directory(path: Path) -> None:
    """Durably synchronize one directory.

    :param path: Directory to synchronize.
    :raises NormalizationError: If synchronization fails.
    """
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as error:
        raise NormalizationError(f"cannot synchronize directory: {path}") from error


def _write_all(descriptor: int, payload: bytes) -> None:
    """Write one complete payload.

    :param descriptor: Destination file descriptor.
    :param payload: Complete payload.
    :raises NormalizationError: If a write fails or stalls.
    """
    offset = 0
    try:
        while offset < len(payload):
            count = os.write(descriptor, payload[offset:])
            if count <= 0:
                raise NormalizationError("evidence write made no progress")
            offset += count
    except OSError as error:
        raise NormalizationError("evidence write failed") from error


class ArtifactStore:
    """Persist immutable evidence and one strictly validated state head."""

    _root: Path
    _root_device: int
    _root_inode: int

    def __init__(self, root: Path, *, create: bool) -> None:
        """Open or exclusively create one private transaction directory.

        :param root: Transaction artifact directory.
        :param create: Whether this is a new transaction.
        """
        self._root = _require_private_directory(root, create=create)
        root_stat = self._root.lstat()
        self._root_device = root_stat.st_dev
        self._root_inode = root_stat.st_ino

    def _require_stable_root(self) -> None:
        """Require the artifact path to retain its original directory identity.

        :raises NormalizationError: If ownership, mode, or inode changed.
        """
        _require_private_directory(self._root, create=False)
        try:
            root_stat = self._root.lstat()
        except OSError as error:
            raise NormalizationError(
                "artifact directory identity is unavailable"
            ) from error
        if (root_stat.st_dev, root_stat.st_ino) != (
            self._root_device,
            self._root_inode,
        ):
            raise NormalizationError("artifact directory identity changed")

    @property
    def root(self) -> Path:
        """Return the private artifact directory.

        :returns: Transaction artifact directory.
        """
        return self._root

    def write_immutable(self, name: str, value: object) -> str:
        """Create one immutable canonical evidence file.

        :param name: Confined evidence filename.
        :param value: Strict JSON evidence.
        :returns: Evidence payload SHA-256 digest.
        :raises NormalizationError: If the name or write is unsafe.
        """
        self._require_stable_root()
        if Path(name).name != name or name in {"config.json", "state.json"}:
            raise NormalizationError("immutable evidence name is unsafe")
        payload = _canonical_json(value)
        path = self._root / name
        try:
            descriptor = os.open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
                0o400,
            )
        except OSError as error:
            raise NormalizationError(
                f"cannot create immutable evidence: {name}"
            ) from error
        try:
            _write_all(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _fsync_directory(self._root)
        return _sha256_bytes(payload)

    def write_config(self, config: NormalizationConfig) -> str:
        """Create the immutable transaction configuration.

        :param config: Validated normalization configuration.
        :returns: Configuration SHA-256 digest.
        """
        self._require_stable_root()
        payload = _canonical_json(config.to_record())
        path = self._root / "config.json"
        try:
            descriptor = os.open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
                0o400,
            )
        except OSError as error:
            raise NormalizationError("cannot create immutable configuration") from error
        try:
            _write_all(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _fsync_directory(self._root)
        return _sha256_bytes(payload)

    def write_state(self, state: TransactionState) -> None:
        """Atomically publish one new transaction state.

        :param state: New strictly serialized state.
        """
        self._require_stable_root()
        payload = _canonical_json(state.to_record())
        temporary = self._root / f".state-{os.getpid()}-{time.monotonic_ns()}"
        try:
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
                0o600,
            )
        except OSError as error:
            raise NormalizationError("cannot create transaction state") from error
        try:
            _write_all(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            os.replace(temporary, self._root / "state.json")
        except OSError as error:
            temporary.unlink(missing_ok=True)
            raise NormalizationError("cannot publish transaction state") from error
        _fsync_directory(self._root)

    def load_json(self, name: str, *, expected_mode: int) -> object:
        """Load one confined regular JSON file with exact mode.

        :param name: Confined filename.
        :param expected_mode: Exact required permissions.
        :returns: Parsed strict JSON value.
        :raises NormalizationError: If identity, mode, or JSON differs.
        """
        self._require_stable_root()
        if Path(name).name != name:
            raise NormalizationError("artifact filename is unsafe")
        path = self._root / name
        try:
            descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        except OSError as error:
            raise NormalizationError(f"cannot open artifact: {name}") from error
        try:
            file_stat = os.fstat(descriptor)
            if (
                stat.S_ISREG(file_stat.st_mode) is False
                or stat.S_IMODE(file_stat.st_mode) != expected_mode
            ):
                raise NormalizationError(f"artifact mode or type differs: {name}")
            payload = bytearray()
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if len(chunk) == 0:
                    break
                payload.extend(chunk)
                if len(payload) > 128 * 1024 * 1024:
                    raise NormalizationError(f"artifact is too large: {name}")
        finally:
            os.close(descriptor)
        try:
            value = json.loads(bytes(payload))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise NormalizationError(f"artifact is not valid JSON: {name}") from error
        if _canonical_json(value) != bytes(payload):
            raise NormalizationError(f"artifact is not canonical JSON: {name}")
        return value

    def load_config_record(self) -> dict[str, object]:
        """Load and validate the immutable public configuration schema.

        :returns: Strict configuration record.
        """
        return _require_object(
            self.load_json("config.json", expected_mode=0o400),
            label="config",
            fields=_CONFIG_FIELDS,
        )

    def load_state(self) -> TransactionState:
        """Load, validate, and authenticate the current transaction state.

        :returns: Authenticated transaction state.
        :raises NormalizationError: If any immutable evidence has drifted.
        """
        record = _require_object(
            self.load_json("state.json", expected_mode=0o600),
            label="state",
            fields=_STATE_FIELDS,
        )
        if record.get("schema_version") != SCHEMA_VERSION:
            raise NormalizationError("state schema version differs")
        try:
            phase = Phase(record.get("phase"))
        except (TypeError, ValueError) as error:
            raise NormalizationError("state phase is invalid") from error
        config_sha256 = _require_digest(
            record.get("config_sha256"), label="state.config_sha256"
        )
        before_sha256 = _require_digest(
            record.get("before_sha256"), label="state.before_sha256"
        )
        if _sha256_file(self._root / "config.json") != config_sha256:
            raise NormalizationError("immutable configuration digest changed")
        if _sha256_file(self._root / "before.json") != before_sha256:
            raise NormalizationError("immutable before evidence digest changed")

        identity_value = record.get("candidate_identity")
        identity: ProcessIdentity | None = None
        if identity_value is not None:
            identity_record = _require_object(
                identity_value,
                label="state.candidate_identity",
                fields=_IDENTITY_FIELDS,
            )
            identity = _identity_from_record(identity_record)

        raw_evidence = record.get("evidence")
        if type(raw_evidence) is not list:
            raise NormalizationError("state.evidence is not a list")
        evidence: list[tuple[str, str]] = []
        seen: set[str] = set()
        for index, raw_item in enumerate(raw_evidence):
            item = _require_object(
                raw_item,
                label=f"state.evidence[{index}]",
                fields=frozenset({"name", "sha256"}),
            )
            name = item.get("name")
            if type(name) is not str or Path(name).name != name or name in seen:
                raise NormalizationError("state evidence name is unsafe or repeated")
            digest = _require_digest(
                item.get("sha256"), label=f"state.evidence[{index}].sha256"
            )
            if _sha256_file(self._root / name) != digest:
                raise NormalizationError(f"immutable evidence digest changed: {name}")
            seen.add(name)
            evidence.append((name, digest))
        state = TransactionState(
            phase=phase,
            config_sha256=config_sha256,
            before_sha256=before_sha256,
            candidate_identity=identity,
            evidence=tuple(evidence),
        )
        _validate_state_evidence(self, state)
        return state


def _identity_from_record(record: Mapping[str, object]) -> ProcessIdentity:
    """Parse one strict process identity record.

    :param record: Exact-schema identity record.
    :returns: Validated process identity.
    :raises NormalizationError: If any identity field is malformed.
    """
    boot_id = record.get("boot_id")
    if type(boot_id) is not str:
        raise NormalizationError("process boot ID is not a string")
    try:
        parsed_boot_id = UUID(boot_id)
    except ValueError as error:
        raise NormalizationError("process boot ID is malformed") from error
    if str(parsed_boot_id) != boot_id:
        raise NormalizationError("process boot ID is not canonical")
    values: dict[str, int] = {}
    for name in ("pid", "process_group_id", "session_id", "start_time_ticks"):
        value = record.get(name)
        if type(value) is not int or value <= 0:
            raise NormalizationError(f"process {name} is not a positive integer")
        values[name] = value
    return ProcessIdentity(
        boot_id=boot_id,
        pid=values["pid"],
        process_group_id=values["process_group_id"],
        session_id=values["session_id"],
        start_time_ticks=values["start_time_ticks"],
    )


def _validate_state_evidence(store: ArtifactStore, state: TransactionState) -> None:
    """Validate phase-specific evidence names, schemas, and identity links.

    :param store: Stable artifact store.
    :param state: Parsed transaction state.
    :raises NormalizationError: If the state cannot be derived from its evidence.
    """
    names = tuple(name for name, _ in state.evidence)
    if state.phase == Phase.FAILED:
        allowed_names = {
            ("preflight.json", "semantic-preparation-failure.json"),
            (
                "preflight.json",
                "semantic-launch.json",
                "semantic-cleanup-failure.json",
            ),
            ("preflight.json", "semantic-cleanup-failure.json"),
        }
        sequence_valid = names in allowed_names
    else:
        sequence_valid = names == _PHASE_EVIDENCE[state.phase]
    if sequence_valid is False:
        raise NormalizationError("state evidence sequence does not derive its phase")
    identity_required = state.phase == Phase.SEMANTIC_RUNNING
    identity_forbidden = state.phase in {
        Phase.BEFORE_SEALED,
        Phase.PREPARED,
        Phase.SEMANTIC_PROVEN,
        Phase.SEMANTIC_FAILED,
    }
    if identity_required and state.candidate_identity is None:
        raise NormalizationError("running semantic state lacks candidate identity")
    if identity_forbidden and state.candidate_identity is not None:
        raise NormalizationError(
            "inactive transaction state records a candidate identity"
        )

    records: dict[str, dict[str, object]] = {}
    for name in names:
        fields = _EVIDENCE_FIELDS.get(name)
        if fields is None:
            raise NormalizationError(f"state references unknown evidence: {name}")
        record = _require_object(
            store.load_json(name, expected_mode=0o400),
            label=name,
            fields=fields,
        )
        if record.get("schema_version") != SCHEMA_VERSION:
            raise NormalizationError(f"evidence schema version differs: {name}")
        records[name] = record

    preflight = records.get("preflight.json")
    if preflight is not None:
        if (
            preflight.get("application_requests_sent") != 0
            or preflight.get("candidate_port_absent") is not True
        ):
            raise NormalizationError("preflight evidence is not application-free")
        _identity_from_record(
            _require_object(
                preflight.get("legacy_identity"),
                label="preflight.legacy_identity",
                fields=_IDENTITY_FIELDS,
            )
        )

    launch = records.get("semantic-launch.json")
    launch_identity: ProcessIdentity | None = None
    if launch is not None:
        if launch.get("candidate_application_requests_before_semantic") != 0:
            raise NormalizationError(
                "semantic launch sent an application request early"
            )
        launch_identity = _identity_from_record(
            _require_object(
                launch.get("candidate_identity"),
                label="semantic-launch.candidate_identity",
                fields=_IDENTITY_FIELDS,
            )
        )
        if (
            state.phase == Phase.SEMANTIC_RUNNING
            and state.candidate_identity != launch_identity
        ):
            raise NormalizationError(
                "running candidate identity differs from launch evidence"
            )

    terminal_name = names[-1] if len(names) > 0 else None
    if terminal_name == "semantic-gate.json":
        terminal = records[terminal_name]
        if launch_identity is None:
            raise NormalizationError("semantic proof lacks launch identity")
        terminal_identity = _identity_from_record(
            _require_object(
                terminal.get("candidate_identity"),
                label="semantic-gate.candidate_identity",
                fields=_IDENTITY_FIELDS,
            )
        )
        cleanup = _require_object(
            terminal.get("cleanup"),
            label="semantic-gate.cleanup",
            fields=frozenset({"candidate_group_absent", "candidate_port_absent"}),
        )
        semantic = _require_object(
            terminal.get("semantic"), label="semantic-gate.semantic"
        )
        if (
            terminal_identity != launch_identity
            or cleanup
            != {"candidate_group_absent": True, "candidate_port_absent": True}
            or semantic.get("texts_equal") is not True
        ):
            raise NormalizationError(
                "semantic proof identity, cleanup, or result differs"
            )
    elif terminal_name == "semantic-gate-failure.json":
        terminal = records[terminal_name]
        failure_identity = _identity_from_record(
            _require_object(
                terminal.get("candidate_identity"),
                label="semantic-gate-failure.candidate_identity",
                fields=_IDENTITY_FIELDS,
            )
        )
        cleanup = _require_object(
            terminal.get("cleanup"),
            label="semantic-gate-failure.cleanup",
            fields=frozenset({"candidate_group_absent", "candidate_port_absent"}),
        )
        if failure_identity != launch_identity or cleanup != {
            "candidate_group_absent": True,
            "candidate_port_absent": True,
        }:
            raise NormalizationError(
                "semantic failure does not prove candidate cleanup"
            )
    elif terminal_name == "semantic-preparation-failure.json":
        terminal = records[terminal_name]
        raw_failure_identity = terminal.get("candidate_identity")
        if raw_failure_identity is not None:
            failure_identity = _identity_from_record(
                _require_object(
                    raw_failure_identity,
                    label="semantic-preparation-failure.candidate_identity",
                    fields=_IDENTITY_FIELDS,
                )
            )
            if (
                failure_identity.pid != failure_identity.process_group_id
                or failure_identity.pid != failure_identity.session_id
            ):
                raise NormalizationError(
                    "semantic preparation failure identity is not a detached leader"
                )
        cleanup = _require_object(
            terminal.get("cleanup"),
            label="semantic-preparation-failure.cleanup",
            fields=frozenset({"candidate_group_absent", "candidate_port_absent"}),
        )
        if cleanup != {
            "candidate_group_absent": True,
            "candidate_port_absent": True,
        }:
            raise NormalizationError("semantic preparation failure lacks cleanup proof")
        if state.candidate_identity is not None:
            raise NormalizationError(
                "inactive semantic preparation failure records a candidate identity"
            )
    elif terminal_name == "semantic-cleanup-failure.json":
        terminal = records[terminal_name]
        raw_cleanup_identity = terminal.get("candidate_identity")
        if raw_cleanup_identity is None:
            if state.candidate_identity is not None:
                raise NormalizationError("cleanup failure identity differs from state")
        else:
            cleanup_identity = _identity_from_record(
                _require_object(
                    raw_cleanup_identity,
                    label="semantic-cleanup-failure.candidate_identity",
                    fields=_IDENTITY_FIELDS,
                )
            )
            if state.candidate_identity != cleanup_identity:
                raise NormalizationError("cleanup failure identity differs from state")


def _parse_process_stat(payload: bytes, pid: int, boot_id: str) -> ProcessIdentity:
    """Parse identity fields from Linux ``/proc/PID/stat``.

    :param payload: Raw proc-stat payload.
    :param pid: Expected process identifier.
    :param boot_id: Current host boot identifier.
    :returns: PID-reuse-safe process identity.
    :raises NormalizationError: If procfs returned a malformed record.
    """
    if payload.endswith(b"\n"):
        payload = payload[:-1]
    prefix = f"{pid} (".encode()
    command_end = payload.rfind(b")")
    if payload.startswith(prefix) is False or command_end < len(prefix):
        raise NormalizationError("proc stat command field is malformed")
    suffix = payload[command_end + 1 :]
    if suffix.startswith(b" ") is False:
        raise NormalizationError("proc stat separator is malformed")
    fields = suffix[1:].split(b" ")
    if len(fields) < 20 or any(len(field) == 0 for field in fields):
        raise NormalizationError("proc stat field sequence is malformed")

    parsed: list[int] = []
    for index, label in ((2, "process group"), (3, "session"), (19, "start time")):
        field = fields[index]
        if field.isascii() is False or field.isdigit() is False:
            raise NormalizationError(f"proc stat {label} is malformed")
        value = int(field)
        if value <= 0 or str(value).encode() != field:
            raise NormalizationError(f"proc stat {label} is not canonical")
        parsed.append(value)
    return ProcessIdentity(
        boot_id=boot_id,
        pid=pid,
        process_group_id=parsed[0],
        session_id=parsed[1],
        start_time_ticks=parsed[2],
    )


def _decode_environment(entries: Sequence[bytes]) -> tuple[tuple[bytes, bytes], ...]:
    """Decode an exact procfs environment into an exec-compatible mapping.

    :param entries: Raw nonempty environment entries.
    :returns: Canonically sorted environment pairs.
    :raises NormalizationError: If an entry or key is malformed or repeated.
    """
    environment: dict[bytes, bytes] = {}
    for entry in entries:
        if b"=" not in entry:
            raise NormalizationError("legacy environment entry lacks an equals sign")
        key, value = entry.split(b"=", maxsplit=1)
        if len(key) == 0 or key in environment or b"\x00" in key or b"\x00" in value:
            raise NormalizationError("legacy environment key is empty or repeated")
        environment[key] = value
    return tuple(sorted(environment.items()))


def _split_nul_payload(payload: bytes, *, label: str) -> tuple[bytes, ...]:
    """Parse a NUL-terminated procfs byte-vector.

    :param payload: Raw cmdline or environment payload.
    :param label: Diagnostic vector name.
    :returns: Exact nonempty entries.
    :raises NormalizationError: If termination or entries are malformed.
    """
    if len(payload) == 0 or payload.endswith(b"\x00") is False:
        raise NormalizationError(f"{label} is empty or not NUL terminated")
    entries = tuple(payload[:-1].split(b"\x00"))
    if any(len(entry) == 0 for entry in entries):
        raise NormalizationError(f"{label} contains an empty entry")
    return entries


def _read_fd_all(descriptor: int, *, limit: int) -> bytes:
    """Read one descriptor to EOF under a hard byte limit.

    :param descriptor: Readable descriptor.
    :param limit: Maximum accepted byte count.
    :returns: Complete payload.
    :raises NormalizationError: If the payload exceeds the limit.
    """
    result = bytearray()
    while True:
        try:
            chunk = os.read(descriptor, min(1024 * 1024, limit + 1 - len(result)))
        except OSError as error:
            raise NormalizationError("descriptor read failed") from error
        if len(chunk) == 0:
            break
        result.extend(chunk)
        if len(result) > limit:
            raise NormalizationError("descriptor payload exceeds its evidence limit")
    return bytes(result)


def _hash_descriptor(descriptor: int) -> tuple[str, int]:
    """Hash a seekable regular-file descriptor from offset zero.

    :param descriptor: Open regular-file descriptor.
    :returns: Digest and byte count.
    :raises NormalizationError: If hashing fails.
    """
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if len(chunk) == 0:
                break
            digest.update(chunk)
            size += len(chunk)
        return digest.hexdigest(), size
    except OSError as error:
        raise NormalizationError("cannot hash process-backed file") from error


def _parse_umask(status_payload: bytes) -> int:
    """Parse the exact process umask from Linux status evidence.

    :param status_payload: Raw ``/proc/PID/status`` bytes.
    :returns: Numeric umask.
    :raises NormalizationError: If the field is absent or malformed.
    """
    matches = [
        line for line in status_payload.splitlines() if line.startswith(b"Umask:")
    ]
    if len(matches) != 1:
        raise NormalizationError("process status does not contain one umask")
    value = matches[0].split(b":", maxsplit=1)[1].strip()
    if len(value) != 4 or any(character not in b"01234567" for character in value):
        raise NormalizationError("process umask is malformed")
    return int(value, 8)


def _record_stat(file_stat: os.stat_result) -> dict[str, object]:
    """Serialize stable inode identity and file metadata.

    :param file_stat: Captured stat result.
    :returns: Strict stat record.
    """
    return {
        "device": file_stat.st_dev,
        "gid": file_stat.st_gid,
        "inode": file_stat.st_ino,
        "mode": file_stat.st_mode,
        "size": file_stat.st_size,
        "uid": file_stat.st_uid,
    }


class LinuxNormalizationSystem:
    """Perform normalization against one local Linux procfs and network stack."""

    _launched_processes: dict[ProcessIdentity, subprocess.Popen[bytes]]
    _proc_root: Path

    def __init__(self, proc_root: Path = Path("/proc")) -> None:
        """Bind host inspection to the Linux proc filesystem.

        :param proc_root: Proc filesystem root.
        """
        if sys.platform != "linux":
            raise NormalizationError("proxy normalization requires Linux")
        self._launched_processes = {}
        self._proc_root = proc_root

    def _boot_id(self) -> str:
        """Read the canonical current boot UUID.

        :returns: Canonical boot identifier.
        """
        try:
            value = (self._proc_root / "sys/kernel/random/boot_id").read_text().strip()
            parsed = UUID(value)
        except (OSError, ValueError) as error:
            raise NormalizationError("Linux boot ID is unavailable") from error
        if str(parsed) != value:
            raise NormalizationError("Linux boot ID is not canonical")
        return value

    def identity(self, pid: int) -> ProcessIdentity:
        """Capture one PID-reuse-safe process identity.

        :param pid: Process identifier.
        :returns: Exact process identity.
        """
        try:
            payload = (self._proc_root / str(pid) / "stat").read_bytes()
        except OSError as error:
            raise NormalizationError(f"process {pid} is unavailable") from error
        return _parse_process_stat(payload, pid, self._boot_id())

    def require_identity(self, expected: ProcessIdentity) -> None:
        """Require one process incarnation to remain exact.

        :param expected: Expected process identity.
        :raises NormalizationError: If the process exited or its PID was reused.
        """
        if self.identity(expected.pid) != expected:
            raise NormalizationError(f"process identity drifted for PID {expected.pid}")

    def _process_group_identities_allow_absent(
        self, pgid: int
    ) -> tuple[ProcessIdentity, ...]:
        """Capture every visible current member of one process group.

        :param pgid: Process-group identifier.
        :returns: Sorted exact group roster, possibly empty.
        :raises NormalizationError: If inspection races.
        """
        identities: list[ProcessIdentity] = []
        try:
            children = tuple(self._proc_root.iterdir())
        except OSError as error:
            raise NormalizationError("cannot enumerate procfs") from error
        for child in children:
            if child.name.isdecimal() is False:
                continue
            pid = int(child.name)
            try:
                identity = self.identity(pid)
            except NormalizationError:
                continue
            if identity.process_group_id == pgid:
                identities.append(identity)
        identities.sort(key=lambda item: item.pid)
        if len(identities) == 0:
            return ()
        first = tuple(identities)
        second: list[ProcessIdentity] = []
        for identity in first:
            try:
                self.require_identity(identity)
            except NormalizationError as error:
                raise NormalizationError(
                    "process group changed during capture"
                ) from error
            second.append(identity)
        if tuple(second) != first:
            raise NormalizationError("process group changed during capture")
        return first

    def process_group_identities(self, pgid: int) -> tuple[ProcessIdentity, ...]:
        """Capture every current member of one process group.

        :param pgid: Process-group identifier.
        :returns: Sorted exact nonempty group roster.
        :raises NormalizationError: If the group is absent or inspection races.
        """
        identities = self._process_group_identities_allow_absent(pgid)
        if len(identities) == 0:
            raise NormalizationError(f"process group {pgid} is absent")
        return identities

    def _resource_limits(self, pid: int) -> tuple[tuple[str, int, int], ...]:
        """Capture the exact supported resource-limit vector.

        :param pid: Process identifier.
        :returns: Sorted limit name, soft limit, and hard limit records.
        """
        limits: list[tuple[str, int, int]] = []
        for name in _RESOURCE_NAMES:
            limit_id = cast(int, getattr(resource, name))
            try:
                soft, hard = resource.prlimit(pid, limit_id)
            except OSError as error:
                raise NormalizationError(
                    f"cannot capture resource limit {name} for PID {pid}"
                ) from error
            limits.append((name, soft, hard))
        return tuple(limits)

    def _capture_executable(self, pid_root: Path) -> dict[str, object]:
        """Capture the in-use executable inode, including deleted links.

        :param pid_root: Process procfs directory.
        :returns: Executable evidence record.
        """
        path = pid_root / "exe"
        try:
            link = os.readlink(path)
            descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
        except OSError as error:
            raise NormalizationError(
                "cannot open the in-use process executable"
            ) from error
        try:
            file_stat = os.fstat(descriptor)
            if stat.S_ISREG(file_stat.st_mode) is False:
                raise NormalizationError("in-use process executable is not regular")
            digest, size = _hash_descriptor(descriptor)
        finally:
            os.close(descriptor)
        if size != file_stat.st_size:
            raise NormalizationError("executable size changed while hashing")
        return {
            "link_hex": os.fsencode(link).hex(),
            "sha256": digest,
            "stat": _record_stat(file_stat),
        }

    def _capture_cwd(self, pid_root: Path) -> dict[str, object]:
        """Capture the process working-directory link and inode.

        :param pid_root: Process procfs directory.
        :returns: Working-directory evidence record.
        """
        path = pid_root / "cwd"
        try:
            link = os.readlink(path)
            descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        except OSError as error:
            raise NormalizationError(
                "cannot capture process working directory"
            ) from error
        try:
            directory_stat = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        return {
            "link_hex": os.fsencode(link).hex(),
            "stat": _record_stat(directory_stat),
        }

    def _capture_mappings(
        self, pid_root: Path, maps_payload: bytes
    ) -> list[dict[str, object]]:
        """Capture every proc map and hash map-files where the kernel permits it.

        :param pid_root: Process procfs directory.
        :param maps_payload: Raw proc maps payload.
        :returns: Ordered mapping evidence.
        :raises NormalizationError: If the maps grammar is malformed.
        """
        records: list[dict[str, object]] = []
        hash_cache: dict[tuple[int, int], dict[str, object]] = {}
        for line in maps_payload.splitlines():
            fields = line.split(maxsplit=5)
            if len(fields) < 5:
                raise NormalizationError("process maps line is malformed")
            address, permissions, offset, device, inode = fields[:5]
            path = fields[5] if len(fields) == 6 else b""
            address_parts = address.split(b"-", maxsplit=1)
            device_parts = device.split(b":", maxsplit=1)
            if len(address_parts) != 2 or len(device_parts) != 2:
                raise NormalizationError("process maps identity is malformed")
            try:
                start = int(address_parts[0], 16)
                end = int(address_parts[1], 16)
                parsed_offset = int(offset, 16)
                major = int(device_parts[0], 16)
                minor = int(device_parts[1], 16)
                parsed_inode = int(inode)
            except ValueError as error:
                raise NormalizationError(
                    "process maps numeric field is malformed"
                ) from error
            if end <= start or len(permissions) != 4:
                raise NormalizationError(
                    "process maps range or permissions are malformed"
                )
            map_file: dict[str, object] | None = None
            if parsed_inode > 0 and len(path) > 0 and path.startswith(b"[") is False:
                key = (os.makedev(major, minor), parsed_inode)
                cached = hash_cache.get(key)
                if cached is not None:
                    map_file = cached
                else:
                    magic_path = pid_root / "map_files" / address.decode("ascii")
                    try:
                        link = os.readlink(magic_path)
                        descriptor = os.open(magic_path, os.O_RDONLY | os.O_CLOEXEC)
                    except OSError as error:
                        map_file = {
                            "errno": error.errno,
                            "status": "inaccessible",
                        }
                    else:
                        try:
                            mapped_stat = os.fstat(descriptor)
                            digest, size = _hash_descriptor(descriptor)
                        finally:
                            os.close(descriptor)
                        if size != mapped_stat.st_size:
                            raise NormalizationError(
                                "mapped file size changed while hashing"
                            )
                        map_file = {
                            "link_hex": os.fsencode(link).hex(),
                            "sha256": digest,
                            "stat": _record_stat(mapped_stat),
                            "status": "readable",
                        }
                    hash_cache[key] = map_file
            records.append(
                {
                    "device_major": major,
                    "device_minor": minor,
                    "end": end,
                    "inode": parsed_inode,
                    "map_file": map_file,
                    "offset": parsed_offset,
                    "path_hex": path.hex(),
                    "permissions": permissions.decode("ascii"),
                    "start": start,
                }
            )
        return records

    def _capture_fds(self, pid_root: Path) -> list[dict[str, object]]:
        """Capture every process descriptor target, inode, and fdinfo payload.

        :param pid_root: Process procfs directory.
        :returns: Canonically sorted FD records.
        """
        fd_root = pid_root / "fd"
        try:
            names = sorted(
                (entry.name for entry in fd_root.iterdir() if entry.name.isdecimal()),
                key=int,
            )
        except OSError as error:
            raise NormalizationError("cannot enumerate process descriptors") from error
        records: list[dict[str, object]] = []
        for name in names:
            fd_path = fd_root / name
            try:
                target = os.readlink(fd_path)
                file_stat = fd_path.stat()
                fdinfo = (pid_root / "fdinfo" / name).read_bytes()
            except FileNotFoundError as error:
                raise NormalizationError(
                    "process descriptor changed during capture"
                ) from error
            except OSError as error:
                raise NormalizationError("cannot capture process descriptor") from error
            records.append(
                {
                    "fd": int(name),
                    "fdinfo_hex": fdinfo.hex(),
                    "fdinfo_sha256": _sha256_bytes(fdinfo),
                    "stat": _record_stat(file_stat),
                    "target_hex": os.fsencode(target).hex(),
                }
            )
        return records

    def _tcp_listeners(self) -> tuple[dict[str, object], ...]:
        """Parse all local IPv4 and IPv6 TCP listeners from procfs.

        :returns: Canonically sorted listener records.
        """
        records: list[dict[str, object]] = []
        for filename, family in (("tcp", socket.AF_INET), ("tcp6", socket.AF_INET6)):
            path = self._proc_root / "net" / filename
            try:
                lines = path.read_text(encoding="ascii").splitlines()
            except OSError as error:
                raise NormalizationError(f"cannot read /proc/net/{filename}") from error
            for line in lines[1:]:
                fields = line.split()
                if len(fields) < 10 or fields[3] != "0A":
                    continue
                local = fields[1].split(":", maxsplit=1)
                if len(local) != 2:
                    raise NormalizationError("TCP listener address is malformed")
                try:
                    packed = bytes.fromhex(local[0])
                    port = int(local[1], 16)
                    inode = int(fields[9])
                except ValueError as error:
                    raise NormalizationError(
                        "TCP listener numeric field is malformed"
                    ) from error
                if family == socket.AF_INET:
                    if len(packed) != 4:
                        raise NormalizationError(
                            "IPv4 listener address has the wrong size"
                        )
                    host = socket.inet_ntop(family, packed[::-1])
                    family_name = "ipv4"
                else:
                    if len(packed) != 16:
                        raise NormalizationError(
                            "IPv6 listener address has the wrong size"
                        )
                    normalized = b"".join(
                        packed[index : index + 4][::-1]
                        for index in range(0, len(packed), 4)
                    )
                    host = socket.inet_ntop(family, normalized)
                    family_name = "ipv6"
                records.append(
                    {
                        "family": family_name,
                        "host": host,
                        "inode": inode,
                        "port": port,
                    }
                )
        records.sort(
            key=lambda item: (
                cast(int, item["port"]),
                cast(str, item["family"]),
                cast(str, item["host"]),
                cast(int, item["inode"]),
            )
        )
        return tuple(records)

    def _socket_owners(self, inode: int) -> tuple[ProcessIdentity, ...]:
        """Find all processes holding one socket inode.

        :param inode: Socket inode.
        :returns: Sorted exact owner identities.
        """
        target = f"socket:[{inode}]"
        owners: list[ProcessIdentity] = []
        try:
            process_roots = tuple(self._proc_root.iterdir())
        except OSError as error:
            raise NormalizationError("cannot enumerate socket owners") from error
        for process_root in process_roots:
            if process_root.name.isdecimal() is False:
                continue
            try:
                descriptors = tuple((process_root / "fd").iterdir())
            except (FileNotFoundError, PermissionError):
                continue
            except OSError as error:
                raise NormalizationError(
                    "cannot enumerate process socket descriptors"
                ) from error
            owns_socket = False
            for descriptor in descriptors:
                try:
                    owns_socket = os.readlink(descriptor) == target
                except FileNotFoundError:
                    continue
                except PermissionError:
                    break
                if owns_socket:
                    break
            if owns_socket is False:
                continue
            try:
                owners.append(self.identity(int(process_root.name)))
            except NormalizationError as error:
                raise NormalizationError(
                    "socket owner changed during capture"
                ) from error
        owners.sort(key=lambda item: item.pid)
        if len(owners) == 0:
            raise NormalizationError(f"listener socket {inode} has no visible owner")
        return tuple(owners)

    def listener_snapshot(self, port: int) -> dict[str, object]:
        """Capture one exact TCP listener and all owning process identities.

        :param port: Exact listener port.
        :returns: Listener and owner evidence.
        :raises NormalizationError: If the listener is absent or ambiguous.
        """
        matches = [record for record in self._tcp_listeners() if record["port"] == port]
        if len(matches) != 1:
            raise NormalizationError(
                f"port {port} does not have exactly one TCP listener"
            )
        listener = matches[0]
        owners = self._socket_owners(cast(int, listener["inode"]))
        return {"listener": listener, "owners": [asdict(owner) for owner in owners]}

    def protected_snapshot(self, ports: tuple[int, ...]) -> dict[str, object]:
        """Capture immutable listener identities for protected ports.

        :param ports: Canonically sorted protected ports.
        :returns: Exact listener roster keyed by decimal port.
        """
        return {str(port): self.listener_snapshot(port) for port in ports}

    def require_port_absent(self, port: int) -> None:
        """Require no TCP listener on one alternate port.

        :param port: Port that must be unbound.
        :raises NormalizationError: If any listener owns the port.
        """
        if any(record["port"] == port for record in self._tcp_listeners()):
            raise NormalizationError(f"port {port} already has a TCP listener")

    def _request(
        self,
        method: str,
        url: str,
        body: bytes | None,
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> HttpResponse:
        """Issue one bounded local HTTP request and retain error responses.

        :param method: HTTP method.
        :param url: Loopback URL.
        :param body: Optional request body.
        :param headers: Explicit request headers.
        :param timeout_seconds: Per-request timeout.
        :returns: Complete bounded response.
        :raises NormalizationError: If the request transport fails.
        """
        parsed = urllib.parse.urlsplit(url)
        if (
            parsed.scheme != "http"
            or parsed.hostname != CANONICAL_HOST
            or parsed.username is not None
            or parsed.password is not None
            or len(parsed.fragment) > 0
        ):
            raise NormalizationError("local HTTP URL is not direct loopback HTTP")
        try:
            port = parsed.port
        except ValueError as error:
            raise NormalizationError("local HTTP URL port is malformed") from error
        if port is None or port <= 0 or port > 65535:
            raise NormalizationError("local HTTP URL lacks a valid port")
        target = parsed.path if len(parsed.path) > 0 else "/"
        if len(parsed.query) > 0:
            target = f"{target}?{parsed.query}"
        deadline = time.monotonic() + timeout_seconds
        connection = http.client.HTTPConnection(
            CANONICAL_HOST,
            port,
            timeout=timeout_seconds,
        )
        try:
            connection.request(method, target, body=body, headers=dict(headers))
            opened = connection.getresponse()
            payload = bytearray()
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise NormalizationError("local HTTP request exceeded its deadline")
                if connection.sock is not None:
                    connection.sock.settimeout(remaining)
                chunk = opened.read1(
                    min(1024 * 1024, 16 * 1024 * 1024 + 1 - len(payload))
                )
                if len(chunk) == 0:
                    break
                payload.extend(chunk)
                if len(payload) > 16 * 1024 * 1024:
                    raise NormalizationError("local HTTP response exceeded 16 MiB")
            response_headers = tuple(
                sorted(
                    (name.lower(), value.strip()) for name, value in opened.getheaders()
                )
            )
            return HttpResponse(opened.status, response_headers, bytes(payload))
        except (OSError, TimeoutError, http.client.HTTPException) as error:
            raise NormalizationError(f"local HTTP request failed: {url}") from error
        finally:
            connection.close()

    def idle_snapshot(self, urls: tuple[str, ...]) -> dict[str, object]:
        """Require zero running and waiting requests on every engine.

        :param urls: Exact local metrics endpoints.
        :returns: Parsed idle metrics evidence.
        :raises NormalizationError: If either metric is absent or nonzero.
        """
        result: dict[str, object] = {}
        for url in urls:
            response = self._request("GET", url, None, {}, 5.0)
            if response.status != 200:
                raise NormalizationError(
                    f"metrics endpoint returned {response.status}: {url}"
                )
            series: dict[bytes, list[float]] = {
                b"vllm:num_requests_running": [],
                b"vllm:num_requests_waiting": [],
            }
            for line in response.body.splitlines():
                stripped = line.strip()
                for metric in series:
                    if stripped.startswith(metric) is False:
                        continue
                    name_and_value = stripped.split()
                    if len(name_and_value) < 2:
                        raise NormalizationError("metrics exposition line is malformed")
                    name = name_and_value[0].split(b"{", maxsplit=1)[0]
                    if name != metric:
                        continue
                    try:
                        series[metric].append(float(name_and_value[1]))
                    except ValueError as error:
                        raise NormalizationError(
                            "metrics value is malformed"
                        ) from error
            if any(len(values) == 0 for values in series.values()):
                raise NormalizationError(f"idle metrics are missing: {url}")
            running = sum(series[b"vllm:num_requests_running"])
            waiting = sum(series[b"vllm:num_requests_waiting"])
            if running != 0.0 or waiting != 0.0:
                raise NormalizationError(f"engine is not idle: {url}")
            result[url] = {
                "body_sha256": _sha256_bytes(response.body),
                "running": running,
                "status": response.status,
                "waiting": waiting,
            }
        return result

    def capture_process(self, pid: int) -> dict[str, object]:
        """Capture complete process, mapping, FD, and listener evidence.

        :param pid: Process identifier.
        :returns: Strict process snapshot.
        """
        before_identity = self.identity(pid)
        pid_root = self._proc_root / str(pid)
        try:
            cmdline = (pid_root / "cmdline").read_bytes()
            environ = (pid_root / "environ").read_bytes()
            maps_payload = (pid_root / "maps").read_bytes()
            status_payload = (pid_root / "status").read_bytes()
        except OSError as error:
            raise NormalizationError(f"cannot capture process {pid}") from error
        cmdline_entries = _split_nul_payload(cmdline, label="process cmdline")
        environment_entries = _split_nul_payload(environ, label="process environment")
        resource_limits = self._resource_limits(pid)
        record = {
            "cmdline_hex": cmdline.hex(),
            "cmdline_sha256": _sha256_bytes(cmdline),
            "cwd": self._capture_cwd(pid_root),
            "environment_entries_hex": [entry.hex() for entry in environment_entries],
            "environment_sha256": _sha256_bytes(environ),
            "executable": self._capture_executable(pid_root),
            "fds": self._capture_fds(pid_root),
            "group": [
                asdict(identity)
                for identity in self.process_group_identities(
                    before_identity.process_group_id
                )
            ],
            "identity": asdict(before_identity),
            "maps": self._capture_mappings(pid_root, maps_payload),
            "maps_hex": maps_payload.hex(),
            "maps_sha256": _sha256_bytes(maps_payload),
            "resource_limits": [
                {"hard": hard, "name": name, "soft": soft}
                for name, soft, hard in resource_limits
            ],
            "status_hex": status_payload.hex(),
            "status_sha256": _sha256_bytes(status_payload),
            "umask": _parse_umask(status_payload),
        }
        self.require_identity(before_identity)
        try:
            final_cmdline = (pid_root / "cmdline").read_bytes()
            final_environ = (pid_root / "environ").read_bytes()
            final_maps = (pid_root / "maps").read_bytes()
        except OSError as error:
            raise NormalizationError(
                "process changed during evidence capture"
            ) from error
        if (
            final_cmdline != cmdline
            or final_environ != environ
            or final_maps != maps_payload
        ):
            raise NormalizationError("process launch inputs changed during capture")
        if (
            _split_nul_payload(final_cmdline, label="process cmdline")
            != cmdline_entries
        ):
            raise NormalizationError("process cmdline changed during capture")
        return record

    @staticmethod
    def _child_setup(contract: LaunchContract, expected_parent_pid: int) -> None:
        """Apply the normalized process creation contract after fork.

        :param contract: Exact launch contract.
        :param expected_parent_pid: Parent identity protected by ``PDEATHSIG``.
        """
        if _PRCTL is None:
            raise RuntimeError("Linux prctl is unavailable")
        if _PRCTL(_PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0) != 0:
            error_number = ctypes.get_errno()
            raise OSError(error_number, os.strerror(error_number))
        if os.getppid() != expected_parent_pid:
            os._exit(125)
        os.umask(contract.umask)
        for name, soft, hard in contract.resource_limits:
            resource.setrlimit(cast(int, getattr(resource, name)), (soft, hard))
        signal.pthread_sigmask(signal.SIG_SETMASK, set())
        for signal_number in signal.valid_signals():
            if signal_number in {signal.SIGKILL, signal.SIGSTOP}:
                continue
            signal.signal(signal_number, signal.SIG_DFL)

    def launch(
        self,
        contract: LaunchContract,
        port: int,
        stdout_path: Path,
        stderr_path: Path,
    ) -> ProcessIdentity:
        """Launch one normalized detached proxy process.

        :param contract: Exact normalized launch contract.
        :param port: Loopback listening port.
        :param stdout_path: Exclusive private stdout log.
        :param stderr_path: Exclusive private stderr log.
        :returns: Detached process identity.
        """
        if port <= 0 or port > 65535:
            raise NormalizationError("launch port is outside the TCP range")
        try:
            stdout_descriptor = os.open(
                stdout_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
                0o600,
            )
            stderr_descriptor = os.open(
                stderr_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
                0o600,
            )
        except OSError as error:
            if "stdout_descriptor" in locals():
                os.close(stdout_descriptor)
            raise NormalizationError("cannot create normalized proxy logs") from error
        try:
            parent_pid = os.getpid()
            process = subprocess.Popen(
                contract.argv(port),
                executable=os.fsencode(contract.python),
                cwd=contract.workdir,
                env=dict(contract.environment),
                stdin=subprocess.DEVNULL,
                stdout=stdout_descriptor,
                stderr=stderr_descriptor,
                close_fds=True,
                start_new_session=True,
                restore_signals=False,
                preexec_fn=lambda: self._child_setup(contract, parent_pid),
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise NormalizationError("cannot launch normalized proxy") from error
        finally:
            os.close(stdout_descriptor)
            os.close(stderr_descriptor)
        try:
            identity = self.identity(process.pid)
            if (
                identity.process_group_id != identity.pid
                or identity.session_id != identity.pid
            ):
                raise NormalizationError(
                    "normalized proxy is not its session and group leader"
                )
            self._launched_processes[identity] = process
            return identity
        except (NormalizationError, OSError):
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5.0)
            raise

    def require_launch_contract(
        self,
        contract: LaunchContract,
        identity: ProcessIdentity,
        port: int,
    ) -> dict[str, object]:
        """Require one running process to match the normalized launch contract.

        :param contract: Expected launch contract.
        :param identity: Expected process incarnation.
        :param port: Expected loopback port.
        :returns: Complete process snapshot.
        """
        self.require_identity(identity)
        if (
            identity.process_group_id != identity.pid
            or identity.session_id != identity.pid
        ):
            raise NormalizationError("normalized proxy lost detached ownership")
        group = self.process_group_identities(identity.process_group_id)
        if group != (identity,):
            raise NormalizationError(
                "normalized proxy process group is not a singleton"
            )
        pid_root = self._proc_root / str(identity.pid)
        try:
            cmdline = (pid_root / "cmdline").read_bytes()
            environ = (pid_root / "environ").read_bytes()
            cwd_link = os.fsencode(os.readlink(pid_root / "cwd"))
        except OSError as error:
            raise NormalizationError(
                "cannot validate normalized proxy launch inputs"
            ) from error
        if cmdline != b"\x00".join(contract.argv(port)) + b"\x00":
            raise NormalizationError("normalized proxy argument vector differs")
        environment_entries = _split_nul_payload(
            environ, label="normalized environment"
        )
        if _decode_environment(environment_entries) != contract.environment:
            raise NormalizationError("normalized proxy environment differs")
        if cwd_link != os.fsencode(contract.workdir):
            raise NormalizationError("normalized proxy working directory differs")
        snapshot = self.capture_process(identity.pid)
        executable = _require_object(snapshot["executable"], label="process.executable")
        if executable.get("sha256") != _sha256_file(contract.python_executable):
            raise NormalizationError("normalized proxy executable digest differs")
        if snapshot.get("umask") != contract.umask:
            raise NormalizationError("normalized proxy umask differs")
        expected_limits = [
            {"hard": hard, "name": name, "soft": soft}
            for name, soft, hard in contract.resource_limits
        ]
        if snapshot.get("resource_limits") != expected_limits:
            raise NormalizationError("normalized proxy resource limits differ")
        try:
            lock_stat = EXPERIMENT_LANE_LOCK.lstat()
        except OSError as error:
            raise NormalizationError(
                "experiment-lane lock identity is absent"
            ) from error
        raw_fds = snapshot.get("fds")
        if type(raw_fds) is not list:
            raise NormalizationError(
                "normalized proxy descriptor evidence is malformed"
            )
        for raw_fd in raw_fds:
            fd_record = _require_object(raw_fd, label="normalized proxy descriptor")
            fd_stat = _require_object(
                fd_record.get("stat"), label="normalized proxy descriptor stat"
            )
            if (
                fd_stat.get("device") == lock_stat.st_dev
                and fd_stat.get("inode") == lock_stat.st_ino
            ):
                raise NormalizationError("normalized proxy inherited the lane lease")
        return snapshot

    def wait_for_listener_gate(
        self,
        contract: LaunchContract,
        identity: ProcessIdentity,
        port: int,
        timeout_seconds: float,
    ) -> dict[str, object]:
        """Require identity, launch contract, and exact loopback listener ownership.

        :param contract: Expected normalized launch contract.
        :param identity: Expected process identity.
        :param port: Expected alternate or canonical port.
        :param timeout_seconds: Bounded readiness timeout.
        :returns: Listener and process evidence without an application request.
        """
        deadline = time.monotonic() + timeout_seconds
        last_error = "candidate did not bind its exact listener"
        while time.monotonic() < deadline:
            self.require_identity(identity)
            try:
                listener = self.listener_snapshot(port)
                listener_record = _require_object(
                    listener["listener"], label="listener"
                )
                owners = listener.get("owners")
                if (
                    listener_record.get("family") != "ipv4"
                    or listener_record.get("host") != CANONICAL_HOST
                    or owners != [asdict(identity)]
                ):
                    raise NormalizationError(
                        "normalized proxy listener identity differs"
                    )
                process = self.require_launch_contract(contract, identity, port)
                return {
                    "application_requests_sent": 0,
                    "listener": self.listener_snapshot(port),
                    "process": process,
                }
            except NormalizationError as error:
                last_error = str(error)
                time.sleep(0.1)
        raise NormalizationError(last_error)

    def stop_group(
        self,
        identities: tuple[ProcessIdentity, ...],
        timeout_seconds: float,
    ) -> None:
        """Terminate one exact session group and prove complete disappearance.

        :param identities: Sealed detached process-group roster.
        :param timeout_seconds: Bounded graceful-termination timeout.
        :raises NormalizationError: If ownership drifts or shutdown times out.
        """
        if (
            len(identities) == 0
            or tuple(sorted(identities, key=lambda item: item.pid)) != identities
        ):
            raise NormalizationError("proxy process group roster is empty or unsorted")
        identity = identities[0]
        if (
            identity.pid != identity.process_group_id
            or identity.pid != identity.session_id
        ):
            raise NormalizationError("proxy is not a detached session and group leader")
        if any(
            member.boot_id != identity.boot_id
            or member.process_group_id != identity.process_group_id
            or member.session_id != identity.session_id
            for member in identities
        ):
            raise NormalizationError("proxy process group identity is inconsistent")

        term_deadline = time.monotonic() + timeout_seconds
        kill_deadline = term_deadline + 5.0
        while True:
            launched_process = self._launched_processes.get(identity)
            launched_process_alive = False
            if launched_process is not None:
                if launched_process.poll() is None:
                    launched_process_alive = True
                else:
                    del self._launched_processes[identity]
            current = self._process_group_identities_allow_absent(
                identity.process_group_id
            )
            if len(current) == 0:
                if launched_process_alive:
                    raise NormalizationError(
                        "launched proxy remains alive outside its process group scan"
                    )
                self._launched_processes.pop(identity, None)
                return
            for member in current:
                if (
                    member.boot_id != identity.boot_id
                    or member.process_group_id != identity.process_group_id
                    or member.session_id != identity.session_id
                    or (member.pid == identity.pid and member != identity)
                ):
                    raise NormalizationError(
                        "proxy process group ownership drifted during termination"
                    )
            now = time.monotonic()
            if now >= kill_deadline:
                raise NormalizationError("proxy process group survived SIGKILL")
            signal_number = signal.SIGTERM if now < term_deadline else signal.SIGKILL
            for member in current:
                try:
                    pidfd = os.pidfd_open(member.pid, 0)
                except ProcessLookupError:
                    continue
                except OSError as error:
                    raise NormalizationError(
                        "cannot open an exact proxy pidfd"
                    ) from error
                try:
                    try:
                        self.require_identity(member)
                    except NormalizationError:
                        continue
                    try:
                        signal.pidfd_send_signal(pidfd, signal_number)
                    except ProcessLookupError:
                        continue
                finally:
                    os.close(pidfd)
            time.sleep(0.05)

    @staticmethod
    def _chat_completion_text(
        response: HttpResponse,
    ) -> tuple[str, dict[str, object]]:
        """Extract one non-streaming chat-completion text.

        :param response: Complete completion response.
        :returns: Completion text and parsed response object.
        :raises NormalizationError: If the response is not one successful completion.
        """
        if response.status != 200:
            raise NormalizationError(
                f"non-streaming chat gate returned HTTP {response.status}"
            )
        try:
            value = json.loads(response.body)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise NormalizationError(
                "non-streaming chat response is not JSON"
            ) from error
        record = _require_object(value, label="non-streaming chat response")
        choices = record.get("choices")
        if type(choices) is not list or len(choices) != 1:
            raise NormalizationError("non-streaming chat response choice count differs")
        choice = _require_object(choices[0], label="non-streaming chat response choice")
        message = _require_object(
            choice.get("message"), label="non-streaming chat response message"
        )
        text = message.get("content")
        if type(text) is not str or len(text) == 0:
            raise NormalizationError("non-streaming chat response text is empty")
        return text, record

    @staticmethod
    def _stream_text(response: HttpResponse) -> tuple[str, list[dict[str, object]]]:
        """Assemble one OpenAI chat-completion SSE stream.

        :param response: Complete bounded streaming response.
        :returns: Concatenated text and parsed event sequence.
        :raises NormalizationError: If the stream grammar or choices differ.
        """
        if response.status != 200:
            raise NormalizationError(
                f"streaming semantic gate returned HTTP {response.status}"
            )
        events: list[dict[str, object]] = []
        pieces: list[str] = []
        done = False
        for line in response.body.splitlines():
            if len(line) == 0:
                continue
            if line.startswith(b"data: ") is False:
                raise NormalizationError("streaming semantic response is not SSE")
            payload = line[6:]
            if payload == b"[DONE]":
                if done:
                    raise NormalizationError("streaming semantic response repeats DONE")
                done = True
                continue
            if done:
                raise NormalizationError(
                    "streaming semantic response continued after DONE"
                )
            try:
                value = json.loads(payload)
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise NormalizationError(
                    "streaming semantic event is not JSON"
                ) from error
            record = _require_object(value, label="streaming semantic event")
            choices = record.get("choices")
            if type(choices) is not list or len(choices) != 1:
                raise NormalizationError(
                    "streaming semantic event choice count differs"
                )
            choice = _require_object(
                choices[0], label="streaming semantic event choice"
            )
            delta = _require_object(
                choice.get("delta"), label="streaming chat event delta"
            )
            text = delta.get("content")
            if text is not None and type(text) is not str:
                raise NormalizationError("streaming chat event text is not a string")
            if type(text) is str:
                pieces.append(text)
            events.append(record)
        if done is False or len(events) == 0:
            raise NormalizationError("streaming semantic response is incomplete")
        combined = "".join(pieces)
        if len(combined) == 0:
            raise NormalizationError("streaming semantic response text is empty")
        return combined, events

    def semantic_gate(
        self,
        port: int,
        model: str,
        timeout_seconds: float,
    ) -> dict[str, object]:
        """Send deterministic non-streaming and streaming P-to-D requests.

        :param port: Alternate candidate port.
        :param model: Exact served model name.
        :param timeout_seconds: Per-request inference timeout.
        :returns: Exact request and response evidence.
        :raises NormalizationError: If either path fails or their text differs.
        """
        prompt = "Write the integers one through five in order, separated by commas."
        common = {
            "chat_template_kwargs": {"enable_thinking": False},
            "max_tokens": 32,
            "messages": [{"content": prompt, "role": "user"}],
            "model": model,
            "seed": 0,
            "temperature": 0.0,
        }
        nonstreaming_request = {**common, "stream": False}
        streaming_request = {**common, "stream": True}
        url = f"http://{CANONICAL_HOST}:{port}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        nonstreaming_payload = _canonical_json(nonstreaming_request)[:-1]
        streaming_payload = _canonical_json(streaming_request)[:-1]
        nonstreaming = self._request(
            "POST", url, nonstreaming_payload, headers, timeout_seconds
        )
        nonstreaming_text, nonstreaming_record = self._chat_completion_text(
            nonstreaming
        )
        streaming = self._request(
            "POST", url, streaming_payload, headers, timeout_seconds
        )
        streaming_text, streaming_events = self._stream_text(streaming)
        if streaming_text != nonstreaming_text:
            raise NormalizationError(
                "deterministic streaming and non-streaming proxy outputs differ"
            )
        expected_compact_text = "1,2,3,4,5"
        if expected_compact_text not in "".join(nonstreaming_text.split()):
            raise NormalizationError(
                "deterministic proxy output does not contain the requested "
                "integer sequence"
            )
        return {
            "expected_compact_text": expected_compact_text,
            "nonstreaming": {
                "body_hex": nonstreaming.body.hex(),
                "body_sha256": _sha256_bytes(nonstreaming.body),
                "headers": [list(item) for item in nonstreaming.headers],
                "parsed": nonstreaming_record,
                "request": nonstreaming_request,
                "request_sha256": _sha256_bytes(nonstreaming_payload),
                "status": nonstreaming.status,
                "text_sha256": _sha256_bytes(nonstreaming_text.encode()),
            },
            "streaming": {
                "body_hex": streaming.body.hex(),
                "body_sha256": _sha256_bytes(streaming.body),
                "events": streaming_events,
                "headers": [list(item) for item in streaming.headers],
                "request": streaming_request,
                "request_sha256": _sha256_bytes(streaming_payload),
                "status": streaming.status,
                "text_sha256": _sha256_bytes(streaming_text.encode()),
            },
            "texts_equal": True,
        }


def _physical_file(path: Path, *, label: str) -> Path:
    """Require one canonical physical regular file.

    :param path: Candidate file.
    :param label: Diagnostic input name.
    :returns: Canonical physical path.
    :raises NormalizationError: If the file is indirect or absent.
    """
    if path.is_absolute() is False:
        raise NormalizationError(f"{label} must be absolute")
    try:
        resolved = path.resolve(strict=True)
        path_stat = path.lstat()
    except OSError as error:
        raise NormalizationError(f"{label} is unavailable") from error
    if resolved != path or stat.S_ISREG(path_stat.st_mode) is False:
        raise NormalizationError(f"{label} must be a canonical physical file")
    return path


def _python_entrypoint(
    path: Path,
) -> tuple[Path, tuple[tuple[str, str, int, int], ...], Path, str]:
    """Attest a virtual-environment entrypoint and its complete symlink chain.

    :param path: Absolute virtual-environment Python entrypoint.
    :returns: Resolved executable, symlink chain, pyvenv path, and pyvenv digest.
    :raises NormalizationError: If the entrypoint or virtual environment is unsafe.
    """
    if (
        path.is_absolute() is False
        or path != Path(os.path.normpath(path))
        or path.parent.name != "bin"
    ):
        raise NormalizationError(
            "python must be a normalized absolute virtual-environment entrypoint"
        )
    venv_root = path.parent.parent
    try:
        venv_stat = venv_root.lstat()
        venv_resolved = venv_root.resolve(strict=True)
    except OSError as error:
        raise NormalizationError("python virtual environment is unavailable") from error
    if (
        venv_resolved != venv_root
        or stat.S_ISDIR(venv_stat.st_mode) is False
        or venv_stat.st_uid != os.geteuid()
    ):
        raise NormalizationError(
            "python virtual environment must be physical and euid-owned"
        )
    pyvenv_config = _physical_file(
        venv_root / "pyvenv.cfg", label="python virtual-environment configuration"
    )

    current = path
    seen: set[tuple[int, int]] = set()
    chain: list[tuple[str, str, int, int]] = []
    for _ in range(40):
        try:
            current_stat = current.lstat()
        except OSError as error:
            raise NormalizationError(
                "python entrypoint chain is unavailable"
            ) from error
        identity = (current_stat.st_dev, current_stat.st_ino)
        if identity in seen:
            raise NormalizationError("python entrypoint chain contains a cycle")
        seen.add(identity)
        if stat.S_ISLNK(current_stat.st_mode):
            try:
                target = os.readlink(current)
            except OSError as error:
                raise NormalizationError(
                    "python entrypoint symlink is unreadable"
                ) from error
            chain.append(
                (
                    str(current),
                    os.fsencode(target).hex(),
                    current_stat.st_dev,
                    current_stat.st_ino,
                )
            )
            target_path = Path(target)
            if target_path.is_absolute() is False:
                target_path = current.parent / target_path
            current = Path(os.path.normpath(target_path))
            continue
        if stat.S_ISREG(current_stat.st_mode) is False:
            raise NormalizationError(
                "python entrypoint does not resolve to a regular executable"
            )
        chain.append((str(current), "", current_stat.st_dev, current_stat.st_ino))
        resolved = path.resolve(strict=True)
        if current.resolve(strict=True) != resolved:
            raise NormalizationError("python entrypoint chain resolution differs")
        return (
            resolved,
            tuple(chain),
            pyvenv_config,
            _sha256_file(pyvenv_config),
        )
    raise NormalizationError("python entrypoint chain is too deep")


def _physical_workdir(path: Path) -> Path:
    """Require one canonical physical working directory.

    :param path: Candidate working directory.
    :returns: Canonical physical directory.
    :raises NormalizationError: If the directory is indirect or absent.
    """
    if path.is_absolute() is False:
        raise NormalizationError("workdir must be absolute")
    try:
        resolved = path.resolve(strict=True)
        path_stat = path.lstat()
    except OSError as error:
        raise NormalizationError("workdir is unavailable") from error
    if resolved != path or stat.S_ISDIR(path_stat.st_mode) is False:
        raise NormalizationError("workdir must be a canonical physical directory")
    return path


def build_config(
    *,
    legacy_pid: int,
    candidate_port: int,
    python: Path,
    proxy_script: Path,
    workdir: Path,
    model: str,
    readiness_timeout_seconds: float,
) -> NormalizationConfig:
    """Validate and bind a new normalization transaction configuration.

    :param legacy_pid: Expected legacy proxy PID.
    :param candidate_port: Alternate loopback candidate port.
    :param python: Canonical current Python executable.
    :param proxy_script: Canonical proxy source file.
    :param workdir: Canonical proxy working directory.
    :param model: Served model name for the explicit semantic gate.
    :param readiness_timeout_seconds: Bounded readiness timeout.
    :returns: Immutable normalization configuration.
    :raises NormalizationError: If any input is unsafe.
    """
    if type(legacy_pid) is not int or legacy_pid <= 0:
        raise NormalizationError("legacy PID must be positive")
    if (
        type(candidate_port) is not int
        or candidate_port <= 0
        or candidate_port > 65535
        or candidate_port == CANONICAL_PORT
        or candidate_port in DEFAULT_PROTECTED_PORTS
    ):
        raise NormalizationError("candidate port is unsafe")
    if (
        type(model) is not str
        or len(model) == 0
        or "\x00" in model
        or "\n" in model
        or "\r" in model
    ):
        raise NormalizationError("model name is empty or unsafe")
    if (
        type(readiness_timeout_seconds) not in {int, float}
        or math.isfinite(readiness_timeout_seconds) is False
        or readiness_timeout_seconds <= 0.0
        or readiness_timeout_seconds > 300.0
    ):
        raise NormalizationError("readiness timeout must be in (0, 300]")
    (
        python_executable,
        python_entrypoint_chain,
        pyvenv_config,
        pyvenv_config_sha256,
    ) = _python_entrypoint(python)
    proxy_script = _physical_file(proxy_script, label="proxy script")
    workdir = _physical_workdir(workdir)
    return NormalizationConfig(
        legacy_pid=legacy_pid,
        candidate_port=candidate_port,
        python=python,
        python_executable=python_executable,
        python_entrypoint_chain=python_entrypoint_chain,
        pyvenv_config=pyvenv_config,
        pyvenv_config_sha256=pyvenv_config_sha256,
        proxy_script=proxy_script,
        workdir=workdir,
        prefiller_host="localhost",
        prefiller_port=8810,
        decoder_host="localhost",
        decoder_port=8811,
        model=model,
        protected_ports=DEFAULT_PROTECTED_PORTS,
        idle_metrics_urls=DEFAULT_IDLE_METRICS_URLS,
        readiness_timeout_seconds=float(readiness_timeout_seconds),
        python_sha256=_sha256_file(python_executable),
        proxy_script_sha256=_sha256_file(proxy_script),
    )


def _load_config(store: ArtifactStore) -> NormalizationConfig:
    """Load and revalidate the immutable transaction configuration.

    :param store: Open transaction artifact store.
    :returns: Current authenticated configuration.
    :raises NormalizationError: If an input or its current digest drifted.
    """
    record = store.load_config_record()
    if record.get("schema_version") != SCHEMA_VERSION:
        raise NormalizationError("configuration schema version differs")
    if record.get("canonical_port") != CANONICAL_PORT:
        raise NormalizationError("configuration canonical port differs")
    integer_names = (
        "legacy_pid",
        "candidate_port",
        "prefiller_port",
        "decoder_port",
    )
    integers: dict[str, int] = {}
    for name in integer_names:
        value = record.get(name)
        if type(value) is not int:
            raise NormalizationError(f"configuration {name} is not an integer")
        integers[name] = value
    string_names = (
        "python",
        "python_executable",
        "pyvenv_config",
        "proxy_script",
        "workdir",
        "model",
    )
    strings: dict[str, str] = {}
    for name in string_names:
        value = record.get(name)
        if type(value) is not str:
            raise NormalizationError(f"configuration {name} is not a string")
        strings[name] = value
    prefiller_host = record.get("prefiller_host")
    decoder_host = record.get("decoder_host")
    if prefiller_host != "localhost" or decoder_host != "localhost":
        raise NormalizationError("configuration upstream hosts differ")
    protected_ports = record.get("protected_ports")
    idle_metrics_urls = record.get("idle_metrics_urls")
    if protected_ports != list(DEFAULT_PROTECTED_PORTS):
        raise NormalizationError("configuration protected ports differ")
    if idle_metrics_urls != list(DEFAULT_IDLE_METRICS_URLS):
        raise NormalizationError("configuration idle metrics endpoints differ")
    timeout = record.get("readiness_timeout_seconds")
    if type(timeout) not in {int, float}:
        raise NormalizationError("configuration readiness timeout is not numeric")
    config = build_config(
        legacy_pid=integers["legacy_pid"],
        candidate_port=integers["candidate_port"],
        python=Path(strings["python"]),
        proxy_script=Path(strings["proxy_script"]),
        workdir=Path(strings["workdir"]),
        model=strings["model"],
        readiness_timeout_seconds=float(timeout),
    )
    if record != config.to_record():
        raise NormalizationError("configuration launch input identity drifted")
    return config


def _legacy_options(argv: tuple[bytes, ...]) -> dict[bytes, bytes]:
    """Parse the complete accepted legacy proxy option grammar.

    :param argv: Argument vector after Python and proxy source.
    :returns: Exact option-to-value mapping with no unconsumed arguments.
    :raises NormalizationError: If an option is unknown, repeated, or malformed.
    """
    allowed = frozenset(
        {
            b"--host",
            b"--port",
            b"--prefiller-hosts",
            b"--prefiller-ports",
            b"--decoder-hosts",
            b"--decoder-ports",
        }
    )
    options: dict[bytes, bytes] = {}
    index = 0
    while index < len(argv):
        name = argv[index]
        if name not in allowed or name in options:
            raise NormalizationError("legacy proxy argv contains an unknown option")
        if index + 1 >= len(argv):
            raise NormalizationError("legacy proxy option lacks a value")
        value = argv[index + 1]
        if len(value) == 0 or value.startswith(b"--"):
            raise NormalizationError("legacy proxy option value is malformed")
        options[name] = value
        index += 2
    required = allowed - {b"--host"}
    option_names = set(options)
    if option_names != set(required) and option_names != set(allowed):
        raise NormalizationError("legacy proxy option roster differs")
    return options


def _launch_contracts(
    config: NormalizationConfig,
    legacy: Mapping[str, object],
) -> tuple[ObservedLegacyLaunch, LaunchContract]:
    """Validate the legacy launch and derive the selected candidate contract.

    :param config: Immutable normalization configuration.
    :param legacy: Sealed legacy process snapshot.
    :returns: Exact observed legacy identity and selected candidate contract.
    :raises NormalizationError: If the legacy process is not the expected proxy.
    """
    identity_record = _require_object(
        legacy.get("identity"), label="legacy.identity", fields=_IDENTITY_FIELDS
    )
    identity = _identity_from_record(identity_record)
    if identity.pid != config.legacy_pid:
        raise NormalizationError("legacy snapshot PID differs")
    group = legacy.get("group")
    if group != [asdict(identity)]:
        raise NormalizationError("legacy proxy group is not a singleton")
    if identity.pid != identity.process_group_id or identity.pid != identity.session_id:
        raise NormalizationError("legacy proxy is not its detached group leader")

    cmdline_hex = legacy.get("cmdline_hex")
    if type(cmdline_hex) is not str:
        raise NormalizationError("legacy cmdline evidence is malformed")
    try:
        cmdline = bytes.fromhex(cmdline_hex)
    except ValueError as error:
        raise NormalizationError(
            "legacy cmdline evidence is not hexadecimal"
        ) from error
    argv = _split_nul_payload(cmdline, label="legacy cmdline")
    if len(argv) < 2 or Path(os.fsdecode(argv[1])).name != "toy_proxy_server.py":
        raise NormalizationError("legacy command is not the expected toy proxy")
    if argv[0] != os.fsencode(config.python):
        raise NormalizationError("legacy Python entrypoint differs")
    options = _legacy_options(argv[2:])
    if options[b"--port"] != str(CANONICAL_PORT).encode():
        raise NormalizationError("legacy proxy port argument differs")
    expected_options = {
        b"--prefiller-hosts": b"localhost",
        b"--prefiller-ports": b"8810",
        b"--decoder-hosts": b"localhost",
        b"--decoder-ports": b"8811",
    }
    for name, expected in expected_options.items():
        if options[name] != expected:
            raise NormalizationError("legacy proxy upstream argument differs")
    if options.get(b"--host", CANONICAL_HOST.encode()) != CANONICAL_HOST.encode():
        raise NormalizationError("legacy proxy host argument differs")

    cwd = _require_object(legacy.get("cwd"), label="legacy.cwd")
    cwd_link = cwd.get("link_hex")
    if type(cwd_link) is not str:
        raise NormalizationError("legacy working-directory evidence is malformed")
    try:
        cwd_bytes = bytes.fromhex(cwd_link)
    except ValueError as error:
        raise NormalizationError(
            "legacy working-directory link is not hexadecimal"
        ) from error
    legacy_workdir = Path(os.fsdecode(cwd_bytes))
    if legacy_workdir.is_absolute() is False:
        raise NormalizationError("legacy proxy working directory is not absolute")
    try:
        if legacy_workdir.is_dir() is False:
            raise NormalizationError(
                "legacy proxy working directory is not a directory"
            )
    except OSError as error:
        raise NormalizationError(
            "legacy proxy working directory is unavailable"
        ) from error

    legacy_script = Path(os.fsdecode(argv[1]))
    if legacy_script.is_absolute() is False:
        legacy_script = legacy_workdir / legacy_script
    try:
        legacy_script = legacy_script.resolve(strict=True)
    except OSError as error:
        raise NormalizationError("legacy proxy source is unavailable") from error
    legacy_script_sha256 = _sha256_file(legacy_script)
    if legacy_script_sha256 != config.proxy_script_sha256:
        raise NormalizationError("legacy and candidate proxy source digests differ")

    executable = _require_object(legacy.get("executable"), label="legacy.executable")
    executable_link = executable.get("link_hex")
    if type(executable_link) is not str:
        raise NormalizationError("legacy executable link is malformed")
    try:
        executable_link_bytes = bytes.fromhex(executable_link)
    except ValueError as error:
        raise NormalizationError("legacy executable link is not hexadecimal") from error
    if executable_link_bytes.endswith(b" (deleted)") is False:
        raise NormalizationError("legacy executable is not the known deleted runtime")
    mappings = legacy.get("maps")
    if type(mappings) is not list:
        raise NormalizationError("legacy mappings are malformed")
    deleted_mapping = False
    for raw_mapping in mappings:
        mapping = _require_object(raw_mapping, label="legacy mapping")
        path_hex = mapping.get("path_hex")
        if type(path_hex) is not str:
            raise NormalizationError("legacy mapping path is malformed")
        try:
            path_bytes = bytes.fromhex(path_hex)
        except ValueError as error:
            raise NormalizationError(
                "legacy mapping path is not hexadecimal"
            ) from error
        deleted_mapping = deleted_mapping or path_bytes.endswith(b" (deleted)")
    if deleted_mapping is False:
        raise NormalizationError("legacy process lacks the known deleted mappings")

    environment_hex = legacy.get("environment_entries_hex")
    if type(environment_hex) is not list:
        raise NormalizationError("legacy environment evidence is malformed")
    environment_entries: list[bytes] = []
    for value in environment_hex:
        if type(value) is not str:
            raise NormalizationError("legacy environment entry is malformed")
        try:
            environment_entries.append(bytes.fromhex(value))
        except ValueError as error:
            raise NormalizationError(
                "legacy environment entry is not hexadecimal"
            ) from error
    umask = legacy.get("umask")
    if type(umask) is not int or umask < 0 or umask > 0o777:
        raise NormalizationError("legacy umask is malformed")
    raw_limits = legacy.get("resource_limits")
    if type(raw_limits) is not list:
        raise NormalizationError("legacy resource limits are malformed")
    limits: list[tuple[str, int, int]] = []
    for raw_limit in raw_limits:
        limit = _require_object(
            raw_limit,
            label="legacy resource limit",
            fields=frozenset({"hard", "name", "soft"}),
        )
        name = limit.get("name")
        soft = limit.get("soft")
        hard = limit.get("hard")
        if (
            type(name) is not str
            or name not in _RESOURCE_NAMES
            or type(soft) is not int
            or type(hard) is not int
        ):
            raise NormalizationError("legacy resource limit value is malformed")
        limits.append((name, soft, hard))
    if tuple(name for name, _, _ in limits) != _RESOURCE_NAMES:
        raise NormalizationError("legacy resource-limit roster differs")
    observed = ObservedLegacyLaunch(
        argv=argv,
        workdir=cwd_bytes,
        proxy_script=legacy_script,
        proxy_script_sha256=legacy_script_sha256,
    )
    candidate = LaunchContract(
        python=config.python,
        python_executable=config.python_executable,
        proxy_script=config.proxy_script,
        workdir=config.workdir,
        prefiller_host=config.prefiller_host,
        prefiller_port=config.prefiller_port,
        decoder_host=config.decoder_host,
        decoder_port=config.decoder_port,
        environment=_decode_environment(environment_entries),
        umask=umask,
        resource_limits=tuple(limits),
    )
    return observed, candidate


def _before_record(
    config: NormalizationConfig,
    system: NormalizationSystem,
) -> dict[str, object]:
    """Capture and validate all immutable pre-mutation evidence.

    :param config: New transaction configuration.
    :param system: Host implementation.
    :returns: Sealed pre-mutation evidence.
    """
    system.require_port_absent(config.candidate_port)
    legacy = system.capture_process(config.legacy_pid)
    observed, contract = _launch_contracts(config, legacy)
    legacy_identity = _identity_from_record(
        _require_object(legacy["identity"], label="legacy.identity")
    )
    listener = system.listener_snapshot(CANONICAL_PORT)
    listener_record = _require_object(listener.get("listener"), label="legacy listener")
    if (
        listener_record.get("family") != "ipv4"
        or listener_record.get("host") != CANONICAL_HOST
        or listener.get("owners") != [asdict(legacy_identity)]
    ):
        raise NormalizationError("legacy 8812 listener identity differs")
    protected = system.protected_snapshot(config.protected_ports)
    return {
        "captured_at_unix_ns": time.time_ns(),
        "legacy_launch_contract": observed.to_record(),
        "launch_contract": contract.to_record(config.candidate_port),
        "legacy": legacy,
        "legacy_listener": listener,
        "protected": protected,
        "schema_version": SCHEMA_VERSION,
    }


def _load_before(
    store: ArtifactStore,
    config: NormalizationConfig,
) -> tuple[dict[str, object], LaunchContract, ProcessIdentity]:
    """Load and validate sealed pre-mutation evidence.

    :param store: Authenticated transaction artifact store.
    :param config: Current immutable configuration.
    :returns: Before record, normalized launch contract, and legacy identity.
    :raises NormalizationError: If the before evidence schema differs.
    """
    before = _require_object(
        store.load_json("before.json", expected_mode=0o400),
        label="before",
        fields=frozenset(
            {
                "captured_at_unix_ns",
                "legacy_launch_contract",
                "launch_contract",
                "legacy",
                "legacy_listener",
                "protected",
                "schema_version",
            }
        ),
    )
    if before.get("schema_version") != SCHEMA_VERSION:
        raise NormalizationError("before evidence schema version differs")
    legacy = _require_object(before.get("legacy"), label="before.legacy")
    observed, contract = _launch_contracts(config, legacy)
    serialized_legacy = _require_object(
        before.get("legacy_launch_contract"),
        label="before.legacy_launch_contract",
        fields=frozenset(
            {"argv_hex", "proxy_script_hex", "proxy_script_sha256", "workdir_hex"}
        ),
    )
    if serialized_legacy != observed.to_record():
        raise NormalizationError(
            "serialized legacy launch contract differs from process evidence"
        )
    serialized_contract = _require_object(
        before.get("launch_contract"),
        label="before.launch_contract",
        fields=frozenset({"argv_hex", "environment", "resource_limits", "umask"}),
    )
    if serialized_contract != contract.to_record(config.candidate_port):
        raise NormalizationError(
            "serialized candidate launch contract differs from legacy evidence"
        )
    identity = _identity_from_record(
        _require_object(
            legacy.get("identity"),
            label="before.legacy.identity",
            fields=_IDENTITY_FIELDS,
        )
    )
    return before, contract, identity


def _transition(
    store: ArtifactStore,
    state: TransactionState,
    *,
    phase: Phase,
    candidate_identity: ProcessIdentity | None,
    evidence_name: str | None = None,
    evidence_value: object | None = None,
) -> TransactionState:
    """Seal optional evidence and atomically advance the transaction state.

    :param store: Transaction artifact store.
    :param state: Authenticated current state.
    :param phase: New durable phase.
    :param candidate_identity: Active normalized process identity, when any.
    :param evidence_name: New immutable evidence filename.
    :param evidence_value: New immutable evidence value.
    :returns: Published new state.
    :raises NormalizationError: If evidence arguments are inconsistent.
    """
    if (evidence_name is None) != (evidence_value is None):
        raise NormalizationError("transition evidence name and value are inconsistent")
    evidence = list(state.evidence)
    if evidence_name is not None:
        digest = store.write_immutable(evidence_name, evidence_value)
        evidence.append((evidence_name, digest))
    updated = TransactionState(
        phase=phase,
        config_sha256=state.config_sha256,
        before_sha256=state.before_sha256,
        candidate_identity=candidate_identity,
        evidence=tuple(evidence),
    )
    store.write_state(updated)
    return updated


def _require_phase(state: TransactionState, expected: Phase) -> None:
    """Require one exact state-machine phase.

    :param state: Current transaction state.
    :param expected: Only accepted phase.
    :raises NormalizationError: If the state machine is elsewhere.
    """
    if state.phase != expected:
        raise NormalizationError(
            f"transaction phase is {state.phase.value}, expected {expected.value}"
        )


def _require_candidate_identity(state: TransactionState) -> ProcessIdentity:
    """Require an active candidate identity in the current state.

    :param state: Current transaction state.
    :returns: Active candidate identity.
    :raises NormalizationError: If no active identity is recorded.
    """
    if state.candidate_identity is None:
        raise NormalizationError("transaction does not record an active candidate")
    return state.candidate_identity


def _same_protected(
    system: NormalizationSystem,
    config: NormalizationConfig,
    before: Mapping[str, object],
) -> dict[str, object]:
    """Require every protected listener identity to remain byte-for-byte exact.

    :param system: Host implementation.
    :param config: Transaction configuration.
    :param before: Sealed pre-mutation evidence.
    :returns: Current protected snapshot.
    :raises NormalizationError: If any listener or owner drifted.
    """
    current = system.protected_snapshot(config.protected_ports)
    if current != before.get("protected"):
        raise NormalizationError("protected listener identities drifted")
    return current


def _exact_group(
    system: NormalizationSystem,
    identity: ProcessIdentity,
) -> tuple[ProcessIdentity, ...]:
    """Require one exact singleton detached proxy group.

    :param system: Host implementation.
    :param identity: Expected root identity.
    :returns: Exact singleton group.
    """
    system.require_identity(identity)
    group = system.process_group_identities(identity.process_group_id)
    if group != (identity,):
        raise NormalizationError("proxy group roster differs from sealed identity")
    return group


def _require_legacy_listener(
    system: NormalizationSystem,
    before: Mapping[str, object],
    legacy_identity: ProcessIdentity,
) -> dict[str, object]:
    """Require the exact sealed legacy identity to retain port 8812.

    :param system: Host implementation.
    :param before: Sealed preflight evidence.
    :param legacy_identity: Exact legacy process incarnation.
    :returns: Current listener snapshot.
    :raises NormalizationError: If process or listener ownership drifted.
    """
    system.require_identity(legacy_identity)
    listener = system.listener_snapshot(CANONICAL_PORT)
    if listener != before.get("legacy_listener"):
        raise NormalizationError("legacy listener identity drifted")
    return listener


def _cleanup_candidate(
    system: NormalizationSystem,
    config: NormalizationConfig,
    candidate: ProcessIdentity | None,
) -> dict[str, object]:
    """Terminate a candidate session and prove its group and listener absent.

    :param system: Host implementation.
    :param config: Immutable transaction configuration.
    :param candidate: Candidate identity when launch returned one.
    :returns: Exact cleanup proof.
    :raises NormalizationError: If termination or disappearance is inconclusive.
    """
    if candidate is not None:
        system.stop_group((candidate,), config.readiness_timeout_seconds)
    system.require_port_absent(config.candidate_port)
    return {"candidate_group_absent": True, "candidate_port_absent": True}


def prepare_candidate(
    artifact_directory: Path,
    config: NormalizationConfig,
    *,
    lease_fd: int | None,
    system: NormalizationSystem,
) -> TransactionState:
    """Seal and revalidate the complete application-free normalization preflight.

    This operation launches no process and issues no HTTP or application request.

    :param artifact_directory: New private transaction directory.
    :param config: Validated normalization configuration.
    :param lease_fd: Borrowed experiment-lane lease, or ``None`` for direct ownership.
    :param system: Host implementation.
    :returns: Durable prepared state with no alternate listener.
    """
    with ExperimentLaneLock(lease_fd):
        store = ArtifactStore(artifact_directory, create=True)
        config_sha256 = store.write_config(config)
        before = _before_record(config, system)
        before_sha256 = store.write_immutable("before.json", before)
        state = TransactionState(
            phase=Phase.BEFORE_SEALED,
            config_sha256=config_sha256,
            before_sha256=before_sha256,
            candidate_identity=None,
            evidence=(),
        )
        store.write_state(state)
        loaded_config = _load_config(store)
        if loaded_config != config:
            raise NormalizationError("prepared configuration changed after sealing")
        _, _, legacy_identity = _load_before(store, config)
        system.require_port_absent(config.candidate_port)
        _require_legacy_listener(system, before, legacy_identity)
        protected = _same_protected(system, config, before)
        system.require_port_absent(config.candidate_port)
        return _transition(
            store,
            state,
            phase=Phase.PREPARED,
            candidate_identity=None,
            evidence_name="preflight.json",
            evidence_value={
                "application_requests_sent": 0,
                "candidate_port_absent": True,
                "legacy_identity": asdict(legacy_identity),
                "protected": protected,
                "schema_version": SCHEMA_VERSION,
            },
        )


def run_semantic_gate(
    artifact_directory: Path,
    *,
    confirmation: str,
    lease_fd: int | None,
    system: NormalizationSystem,
    inference_timeout_seconds: float,
) -> TransactionState:
    """Run the first explicit deterministic GPU semantic gate against the candidate.

    :param artifact_directory: Existing transaction directory.
    :param confirmation: Exact GPU-request acknowledgement token.
    :param lease_fd: Borrowed experiment-lane lease, or ``None`` for direct ownership.
    :param system: Host implementation.
    :param inference_timeout_seconds: Per-request inference timeout.
    :returns: Durable semantic-proven state.
    :raises NormalizationError: If confirmation, state, or evidence differs.
    """
    if confirmation != GPU_CONFIRMATION:
        raise NormalizationError(
            "semantic-gate is the first GPU request; pass the exact confirmation token"
        )
    if (
        type(inference_timeout_seconds) not in {int, float}
        or math.isfinite(inference_timeout_seconds) is False
        or inference_timeout_seconds <= 0.0
        or inference_timeout_seconds > 900.0
    ):
        raise NormalizationError("inference timeout must be in (0, 900]")
    with ExperimentLaneLock(lease_fd):
        store = ArtifactStore(artifact_directory, create=False)
        config = _load_config(store)
        state = store.load_state()
        _require_phase(state, Phase.PREPARED)
        before, contract, legacy_identity = _load_before(store, config)
        system.require_port_absent(config.candidate_port)
        protected_before = _same_protected(system, config, before)
        idle_before = system.idle_snapshot(config.idle_metrics_urls)
        legacy_listener_before = _require_legacy_listener(
            system, before, legacy_identity
        )
        candidate: ProcessIdentity | None = None
        running_state: TransactionState | None = None
        semantic: dict[str, object] | None = None
        candidate_listener_after: dict[str, object] | None = None
        candidate_process_after: dict[str, object] | None = None
        idle_after: dict[str, object] | None = None
        protected_after: dict[str, object] | None = None
        legacy_listener_after: dict[str, object] | None = None
        cleanup: dict[str, object] | None = None
        with FirstSignalGuard() as signal_guard:
            try:
                signal_guard.defer_exception()
                candidate = system.launch(
                    contract,
                    config.candidate_port,
                    store.root / "candidate.stdout.log",
                    store.root / "candidate.stderr.log",
                )
                signal_guard.raise_pending()
                listener_gate = system.wait_for_listener_gate(
                    contract,
                    candidate,
                    config.candidate_port,
                    config.readiness_timeout_seconds,
                )
                _exact_group(system, candidate)
                protected_at_launch = _same_protected(system, config, before)
                idle_at_launch = system.idle_snapshot(config.idle_metrics_urls)
                legacy_listener_at_launch = _require_legacy_listener(
                    system, before, legacy_identity
                )
                running_state = _transition(
                    store,
                    state,
                    phase=Phase.SEMANTIC_RUNNING,
                    candidate_identity=candidate,
                    evidence_name="semantic-launch.json",
                    evidence_value={
                        "candidate_application_requests_before_semantic": 0,
                        "candidate_identity": asdict(candidate),
                        "candidate_listener": listener_gate["listener"],
                        "candidate_process": listener_gate["process"],
                        "idle": idle_at_launch,
                        "legacy_identity": asdict(legacy_identity),
                        "legacy_listener": legacy_listener_at_launch,
                        "protected": protected_at_launch,
                        "schema_version": SCHEMA_VERSION,
                    },
                )
                semantic = system.semantic_gate(
                    config.candidate_port,
                    config.model,
                    float(inference_timeout_seconds),
                )
                idle_after = system.idle_snapshot(config.idle_metrics_urls)
                protected_after = _same_protected(system, config, before)
                legacy_listener_after = _require_legacy_listener(
                    system, before, legacy_identity
                )
                candidate_listener_after = system.listener_snapshot(
                    config.candidate_port
                )
                candidate_process_after = system.require_launch_contract(
                    contract, candidate, config.candidate_port
                )
                if _load_config(store) != config:
                    raise NormalizationError(
                        "candidate launch inputs drifted during semantic proof"
                    )
                if (
                    candidate is None
                    or running_state is None
                    or semantic is None
                    or candidate_listener_after is None
                    or candidate_process_after is None
                    or idle_after is None
                    or protected_after is None
                    or legacy_listener_after is None
                ):
                    raise NormalizationError("semantic proof is incomplete")
                signal_guard.defer_exception()
            except BaseException:
                failure_traceback = traceback.format_exc()
                signal_guard.defer_exception()
                try:
                    cleanup = _cleanup_candidate(system, config, candidate)
                except BaseException as cleanup_error:
                    cleanup_traceback = traceback.format_exc()
                    failure_state = (
                        running_state if running_state is not None else state
                    )
                    _transition(
                        store,
                        failure_state,
                        phase=Phase.FAILED,
                        candidate_identity=candidate,
                        evidence_name="semantic-cleanup-failure.json",
                        evidence_value={
                            "candidate_identity": (
                                None if candidate is None else asdict(candidate)
                            ),
                            "cleanup_traceback": cleanup_traceback,
                            "failure_traceback": failure_traceback,
                            "schema_version": SCHEMA_VERSION,
                        },
                    )
                    signal_guard.raise_pending()
                    raise NormalizationError(
                        "candidate cleanup failed after semantic action failure"
                    ) from cleanup_error
                candidate_logs = _failure_log_evidence(store.root)
                failure_state = running_state if running_state is not None else state
                evidence_name = (
                    "semantic-gate-failure.json"
                    if running_state is not None
                    else "semantic-preparation-failure.json"
                )
                failure_phase = (
                    Phase.SEMANTIC_FAILED if running_state is not None else Phase.FAILED
                )
                _transition(
                    store,
                    failure_state,
                    phase=failure_phase,
                    candidate_identity=None,
                    evidence_name=evidence_name,
                    evidence_value={
                        "candidate_identity": (
                            None if candidate is None else asdict(candidate)
                        ),
                        "candidate_logs": candidate_logs,
                        "cleanup": cleanup,
                        "failure_traceback": failure_traceback,
                        "schema_version": SCHEMA_VERSION,
                    },
                )
                signal_guard.raise_pending()
                raise
            try:
                cleanup = _cleanup_candidate(system, config, candidate)
            except BaseException as cleanup_error:
                cleanup_traceback = traceback.format_exc()
                if running_state is None:
                    raise NormalizationError(
                        "semantic candidate lacks a durable running state"
                    ) from cleanup_error
                _transition(
                    store,
                    running_state,
                    phase=Phase.FAILED,
                    candidate_identity=candidate,
                    evidence_name="semantic-cleanup-failure.json",
                    evidence_value={
                        "candidate_identity": (
                            None if candidate is None else asdict(candidate)
                        ),
                        "cleanup_traceback": cleanup_traceback,
                        "failure_traceback": None,
                        "schema_version": SCHEMA_VERSION,
                    },
                )
                signal_guard.raise_pending()
                raise NormalizationError(
                    "candidate cleanup failed after semantic proof"
                ) from cleanup_error
            if cleanup is None:
                raise NormalizationError("semantic proof is incomplete after cleanup")
            try:
                candidate_logs = {
                    "stderr": _log_evidence(store.root / "candidate.stderr.log"),
                    "stdout": _log_evidence(store.root / "candidate.stdout.log"),
                }
                terminal_state = _transition(
                    store,
                    running_state,
                    phase=Phase.SEMANTIC_PROVEN,
                    candidate_identity=None,
                    evidence_name="semantic-gate.json",
                    evidence_value={
                        "candidate_identity": asdict(candidate),
                        "candidate_listener_after": candidate_listener_after,
                        "candidate_logs": candidate_logs,
                        "candidate_process_after": candidate_process_after,
                        "cleanup": cleanup,
                        "idle_after": idle_after,
                        "idle_before": idle_before,
                        "legacy_listener_after": legacy_listener_after,
                        "legacy_listener_before": legacy_listener_before,
                        "protected_after": protected_after,
                        "protected_before": protected_before,
                        "schema_version": SCHEMA_VERSION,
                        "semantic": semantic,
                    },
                )
            except BaseException:
                failure_traceback = traceback.format_exc()
                candidate_logs = _failure_log_evidence(store.root)
                _transition(
                    store,
                    running_state,
                    phase=Phase.SEMANTIC_FAILED,
                    candidate_identity=None,
                    evidence_name="semantic-gate-failure.json",
                    evidence_value={
                        "candidate_identity": asdict(candidate),
                        "candidate_logs": candidate_logs,
                        "cleanup": cleanup,
                        "failure_traceback": failure_traceback,
                        "schema_version": SCHEMA_VERSION,
                    },
                )
                signal_guard.raise_pending()
                raise
            signal_guard.raise_pending()
            return terminal_state


def _log_evidence(path: Path) -> dict[str, object]:
    """Capture one bounded private process log after its writer has exited.

    :param path: Exact private log path.
    :returns: Complete log evidence.
    :raises NormalizationError: If the log is unsafe or exceeds 64 MiB.
    """
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as error:
        raise NormalizationError(f"cannot open proxy log: {path.name}") from error
    try:
        file_stat = os.fstat(descriptor)
        if (
            stat.S_ISREG(file_stat.st_mode) is False
            or stat.S_IMODE(file_stat.st_mode) != 0o600
        ):
            raise NormalizationError(f"proxy log mode or type differs: {path.name}")
        payload = _read_fd_all(descriptor, limit=64 * 1024 * 1024)
    finally:
        os.close(descriptor)
    return {
        "body_hex": payload.hex(),
        "sha256": _sha256_bytes(payload),
        "size": len(payload),
    }


def _failure_log_evidence(root: Path) -> dict[str, object]:
    """Capture each candidate log that exists after proven cleanup.

    :param root: Transaction artifact directory.
    :returns: Per-stream captured evidence or an explicit unavailable result.
    """
    logs: dict[str, object] = {}
    for stream in ("stderr", "stdout"):
        try:
            evidence = _log_evidence(root / f"candidate.{stream}.log")
        except NormalizationError as error:
            logs[stream] = {"reason": str(error), "status": "unavailable"}
            continue
        logs[stream] = {"evidence": evidence, "status": "captured"}
    return logs


def inspect_transaction(artifact_directory: Path) -> dict[str, object]:
    """Authenticate and summarize a transaction without mutating the lane.

    :param artifact_directory: Existing transaction directory.
    :returns: Strict state plus the next authorized action.
    """
    store = ArtifactStore(artifact_directory, create=False)
    _load_config(store)
    state = store.load_state()
    next_actions = {
        Phase.BEFORE_SEALED: (
            "preflight sealing was interrupted; start a new artifact directory"
        ),
        Phase.PREPARED: (
            "PAUSE: semantic-gate is the first command that sends GPU inference"
        ),
        Phase.SEMANTIC_RUNNING: (
            "candidate ownership is unresolved; do not mutate the lane"
        ),
        Phase.SEMANTIC_PROVEN: (
            "semantic proof is complete; canonical cutover is not implemented"
        ),
        Phase.SEMANTIC_FAILED: "retain evidence and start a new transaction",
        Phase.FAILED: "retain evidence and resolve any recorded identity manually",
    }
    return {
        "next_action": next_actions[state.phase],
        "state": state.to_record(),
    }


def _positive_integer(value: str) -> int:
    """Parse one canonical positive decimal integer for argparse.

    :param value: Candidate argument spelling.
    :returns: Parsed positive integer.
    :raises argparse.ArgumentTypeError: If the spelling is not canonical.
    """
    if value.isascii() is False or value.isdecimal() is False:
        raise argparse.ArgumentTypeError("must be a positive decimal integer")
    result = int(value)
    if result <= 0 or str(result) != value:
        raise argparse.ArgumentTypeError("must be a canonical positive integer")
    return result


def _positive_float(value: str) -> float:
    """Parse one finite positive decimal timeout for argparse.

    :param value: Candidate timeout spelling.
    :returns: Parsed positive timeout.
    :raises argparse.ArgumentTypeError: If the timeout is unsafe.
    """
    try:
        result = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive number") from error
    if result <= 0.0 or result == float("inf") or result != result:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return result


def build_parser() -> argparse.ArgumentParser:
    """Build the explicit normalization state-machine CLI.

    :returns: Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Prove an ephemeral replacement for the degraded experiment proxy. "
            "No command implicitly advances into the next safety boundary."
        )
    )
    subparsers = parser.add_subparsers(dest="action", required=True)

    prepare = subparsers.add_parser(
        "prepare",
        help="seal an application-free preflight without launching a candidate",
    )
    prepare.add_argument("--artifact-directory", type=Path, required=True)
    prepare.add_argument("--legacy-pid", type=_positive_integer, required=True)
    prepare.add_argument(
        "--candidate-port", type=_positive_integer, default=DEFAULT_CANDIDATE_PORT
    )
    prepare.add_argument("--python", type=Path, required=True)
    prepare.add_argument("--proxy-script", type=Path, required=True)
    prepare.add_argument("--workdir", type=Path, required=True)
    prepare.add_argument("--model", required=True)
    prepare.add_argument(
        "--readiness-timeout-seconds", type=_positive_float, default=30.0
    )
    prepare.add_argument("--lease-fd", type=_positive_integer)

    semantic = subparsers.add_parser(
        "semantic-gate",
        help="FIRST GPU REQUEST: deterministic streaming and non-streaming P-to-D gate",
    )
    semantic.add_argument("--artifact-directory", type=Path, required=True)
    semantic.add_argument(
        "--confirm-gpu-request",
        required=True,
        help=f"must equal {GPU_CONFIRMATION!r}",
    )
    semantic.add_argument(
        "--inference-timeout-seconds", type=_positive_float, default=300.0
    )
    semantic.add_argument("--lease-fd", type=_positive_integer)

    inspect = subparsers.add_parser(
        "inspect",
        help="authenticate artifacts and print the current safety boundary",
    )
    inspect.add_argument("--artifact-directory", type=Path, required=True)
    return parser


def _print_summary(state: TransactionState) -> None:
    """Print one exact successful state transition summary.

    :param state: Published transaction state.
    """
    summary = {
        "candidate_identity": (
            None
            if state.candidate_identity is None
            else asdict(state.candidate_identity)
        ),
        "evidence_count": len(state.evidence),
        "phase": state.phase.value,
        "schema_version": SCHEMA_VERSION,
    }
    print(_canonical_json(summary).decode(), end="")


def main(arguments: list[str] | None = None) -> int:
    """Execute one explicit normalization state transition.

    :param arguments: Optional argument vector excluding the program name.
    :returns: Process exit status.
    """
    parser = build_parser()
    parsed = parser.parse_args(arguments)
    action = cast(str, parsed.action)
    try:
        if action == "inspect":
            print(
                _canonical_json(
                    inspect_transaction(cast(Path, parsed.artifact_directory))
                ).decode(),
                end="",
            )
            return 0
        system = LinuxNormalizationSystem()
        if action == "prepare":
            config = build_config(
                legacy_pid=cast(int, parsed.legacy_pid),
                candidate_port=cast(int, parsed.candidate_port),
                python=cast(Path, parsed.python),
                proxy_script=cast(Path, parsed.proxy_script),
                workdir=cast(Path, parsed.workdir),
                model=cast(str, parsed.model),
                readiness_timeout_seconds=cast(float, parsed.readiness_timeout_seconds),
            )
            state = prepare_candidate(
                cast(Path, parsed.artifact_directory),
                config,
                lease_fd=cast(int | None, parsed.lease_fd),
                system=system,
            )
        elif action == "semantic-gate":
            state = run_semantic_gate(
                cast(Path, parsed.artifact_directory),
                confirmation=cast(str, parsed.confirm_gpu_request),
                lease_fd=cast(int | None, parsed.lease_fd),
                system=system,
                inference_timeout_seconds=cast(float, parsed.inference_timeout_seconds),
            )
        else:
            raise NormalizationError(f"unknown normalization action: {action}")
    except (ExperimentLaneLockError, NormalizationError) as error:
        print(f"proxy normalization failed: {error}", file=sys.stderr)
        return 1
    _print_summary(state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
