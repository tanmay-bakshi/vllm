"""Shared serialization and file-integrity primitives."""

import fcntl
import hashlib
import json
import os
import re
import urllib.parse
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from time import monotonic_ns
from typing import BinaryIO, TypeAlias

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]

_HEADER_NAME = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")
_SENSITIVE_NAME_SEPARATOR = re.compile(r"[^a-z0-9]+")
_SENSITIVE_NAME_FRAGMENTS = (
    "accesskey",
    "apikey",
    "auth",
    "bearer",
    "cookie",
    "credential",
    "password",
    "privatekey",
    "secret",
    "session",
    "signature",
    "token",
)


class CertificationError(RuntimeError):
    """Base error for invalid certification artifacts or operations."""


def utc_now() -> str:
    """Return a normalized UTC timestamp."""
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def canonical_json_bytes(value: object) -> bytes:
    """Serialize JSON deterministically without permitting non-finite numbers."""
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def sha256_bytes(data: bytes) -> str:
    """Return the hexadecimal SHA-256 digest of bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a regular file without following a final symlink."""
    if path.is_symlink() or not path.is_file():
        raise CertificationError(f"artifact is not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_valid_http_header_name(name: str) -> bool:
    """Return whether a name is an RFC-compatible HTTP field name."""
    return _HEADER_NAME.fullmatch(name) is not None


def is_sensitive_name(name: str) -> bool:
    """Return whether a field name requires value redaction."""
    compact = _SENSITIVE_NAME_SEPARATOR.sub("", name.lower())
    return any(fragment in compact for fragment in _SENSITIVE_NAME_FRAGMENTS)


def validate_http_endpoint(endpoint: str) -> None:
    """Validate a credential-free absolute HTTP endpoint."""
    try:
        parsed = urllib.parse.urlsplit(endpoint)
        hostname = parsed.hostname
    except ValueError as error:
        raise CertificationError("recorder endpoint is not a valid URL") from error
    if parsed.scheme not in {"http", "https"} or hostname is None:
        raise CertificationError("recorder endpoint must be an absolute HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise CertificationError("recorder endpoint cannot contain userinfo")
    if len(parsed.query) > 0 or "?" in endpoint:
        raise CertificationError("recorder endpoint cannot contain a query")
    if len(parsed.fragment) > 0 or "#" in endpoint:
        raise CertificationError("recorder endpoint cannot contain a fragment")


def safe_relative_path(value: str) -> PurePosixPath:
    """Validate a portable artifact path confined to an arm directory."""
    path = PurePosixPath(value)
    if path.is_absolute() or len(path.parts) == 0:
        raise CertificationError(f"artifact path must be relative: {value!r}")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise CertificationError(f"artifact path is not confined: {value!r}")
    return path


def atomic_write_exclusive(path: Path, data: bytes) -> None:
    """Create and durably write a file, refusing to replace existing evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as destination:
            destination.write(data)
            destination.flush()
            os.fsync(destination.fileno())
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def append_sequenced_json(
    path: Path, record: Mapping[str, object]
) -> dict[str, object]:
    """Append one locked, sequenced, crash-durable JSONL event."""
    reserved = {"sequence", "recorded_at_utc", "recorded_monotonic_ns"}
    collisions = sorted(reserved & set(record))
    if len(collisions) > 0:
        raise CertificationError(
            f"record uses reserved event fields: {', '.join(collisions)}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        with os.fdopen(descriptor, "r+b", closefd=False) as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            sequence = _last_sequence(stream) + 1
            enriched = {
                **record,
                "sequence": sequence,
                "recorded_at_utc": utc_now(),
                "recorded_monotonic_ns": monotonic_ns(),
            }
            stream.seek(0, os.SEEK_END)
            stream.write(canonical_json_bytes(enriched))
            stream.flush()
            os.fsync(stream.fileno())
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)
    return enriched


def read_json_file(path: Path) -> object:
    """Read one UTF-8 JSON file."""
    with path.open("r", encoding="utf-8") as source:
        return json.load(source)


def validate_sequenced_jsonl(path: Path) -> None:
    """Validate that a JSONL artifact has complete contiguous event numbers."""
    expected = 1
    with path.open("rb") as source:
        for line_number, raw_line in enumerate(source, start=1):
            try:
                parsed = json.loads(raw_line, parse_constant=_reject_json_constant)
            except (json.JSONDecodeError, UnicodeError, ValueError) as error:
                raise CertificationError(
                    f"invalid JSONL record at {path}:{line_number}"
                ) from error
            if not isinstance(parsed, dict):
                raise CertificationError(
                    f"JSONL record is not an object at {path}:{line_number}"
                )
            if canonical_json_bytes(parsed) != raw_line:
                raise CertificationError(
                    f"JSONL record is not canonical at {path}:{line_number}"
                )
            sequence = parsed.get("sequence")
            if (
                isinstance(sequence, bool)
                or not isinstance(sequence, int)
                or sequence != expected
            ):
                raise CertificationError(
                    f"non-contiguous JSONL sequence at {path}:{line_number}"
                )
            expected += 1


def _last_sequence(stream: BinaryIO) -> int:
    stream.seek(0, os.SEEK_END)
    position = stream.tell()
    if position == 0:
        return 0
    stream.seek(-1, os.SEEK_END)
    if stream.read(1) != b"\n":
        raise CertificationError("JSONL does not end with a complete record")
    suffix = b""
    while position > 0:
        chunk_size = min(position, 4096)
        position -= chunk_size
        stream.seek(position)
        suffix = stream.read(chunk_size) + suffix
        content = suffix.rstrip(b"\r\n")
        line_start = content.rfind(b"\n")
        if line_start >= 0 or position == 0:
            raw_line = content[line_start + 1 :]
            break
    try:
        parsed = json.loads(raw_line, parse_constant=_reject_json_constant)
        if canonical_json_bytes(parsed) != raw_line + b"\n":
            raise CertificationError("JSONL ends in a noncanonical record")
    except (json.JSONDecodeError, UnicodeError, TypeError, ValueError) as error:
        raise CertificationError("JSONL ends in an invalid record") from error
    if not isinstance(parsed, dict):
        raise CertificationError("JSONL record is not an object")
    sequence = parsed.get("sequence")
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence <= 0:
        raise CertificationError("JSONL sequence is invalid")
    return sequence


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant: {value}")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
