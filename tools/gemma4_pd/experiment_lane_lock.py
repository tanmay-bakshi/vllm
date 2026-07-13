# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Canonical ownership contract for the host-wide experiment lane."""

import errno
import fcntl
import os
import stat
import sys
from pathlib import Path
from types import TracebackType

EXPERIMENT_LANE_LOCK = Path("/data/colleague/locks/gemma4-experiment-lane.lock")


class ExperimentLaneLockError(RuntimeError):
    """Report an unsafe or unavailable experiment-lane ownership boundary."""


def _validate_lock_directory(*, create: bool) -> Path:
    """Validate the private physical parent of the canonical lane lock.

    :param create: Whether a direct owner may create the directory.
    :returns: Validated canonical lock directory.
    :raises ExperimentLaneLockError: If the directory boundary is unsafe.
    """
    lock_directory = EXPERIMENT_LANE_LOCK.parent
    if create:
        try:
            lock_directory.mkdir(parents=False, exist_ok=True, mode=0o700)
        except OSError as error:
            raise ExperimentLaneLockError(
                "experiment-lane lock directory cannot be created"
            ) from error
    try:
        directory_stat = lock_directory.lstat()
        resolved_directory = lock_directory.resolve(strict=True)
    except OSError as error:
        raise ExperimentLaneLockError(
            "experiment-lane lock directory is absent"
        ) from error
    if (
        lock_directory.is_absolute() is False
        or resolved_directory != lock_directory
        or stat.S_ISDIR(directory_stat.st_mode) is False
        or stat.S_IMODE(directory_stat.st_mode) != 0o700
        or directory_stat.st_uid != os.geteuid()
    ):
        raise ExperimentLaneLockError(
            "experiment-lane lock directory is not private and euid-owned"
        )
    return lock_directory


def _validate_lock_file(descriptor: int) -> os.stat_result:
    """Bind one read-write descriptor to the canonical physical lock file.

    :param descriptor: Open candidate lock descriptor.
    :returns: Stable canonical descriptor identity.
    :raises ExperimentLaneLockError: If path, mode, inode, or access differ.
    """
    try:
        descriptor_stat = os.fstat(descriptor)
        path_stat = EXPERIMENT_LANE_LOCK.lstat()
    except OSError as error:
        raise ExperimentLaneLockError(
            "experiment-lane lock file identity is unavailable"
        ) from error
    if (
        stat.S_ISREG(descriptor_stat.st_mode) is False
        or stat.S_IMODE(descriptor_stat.st_mode) != 0o600
        or stat.S_ISREG(path_stat.st_mode) is False
        or stat.S_IMODE(path_stat.st_mode) != 0o600
        or descriptor_stat.st_uid != os.geteuid()
        or path_stat.st_uid != os.geteuid()
        or (descriptor_stat.st_dev, descriptor_stat.st_ino)
        != (path_stat.st_dev, path_stat.st_ino)
    ):
        raise ExperimentLaneLockError("experiment-lane lock file identity differs")
    try:
        status_flags = fcntl.fcntl(descriptor, fcntl.F_GETFL)
    except OSError as error:
        raise ExperimentLaneLockError(
            "experiment-lane lock access mode is unavailable"
        ) from error
    if status_flags & os.O_ACCMODE != os.O_RDWR:
        raise ExperimentLaneLockError(
            "experiment-lane lock descriptor is not read-write"
        )
    return descriptor_stat


def _require_borrowed_write_flock(
    descriptor: int,
    descriptor_stat: os.stat_result,
) -> None:
    """Prove Linux reports a write FLOCK on this exact inherited descriptor.

    :param descriptor: Borrowed canonical lock descriptor.
    :param descriptor_stat: Validated canonical descriptor identity.
    :raises ExperimentLaneLockError: If descriptor-local lock proof is absent.
    """
    if sys.platform != "linux":
        raise ExperimentLaneLockError(
            "borrowed experiment-lane leases require Linux fdinfo"
        )
    try:
        fdinfo = (Path("/proc/self/fdinfo") / str(descriptor)).read_text()
    except OSError as error:
        raise ExperimentLaneLockError(
            "experiment-lane lease FD lock evidence is absent"
        ) from error
    expected_device = (
        os.major(descriptor_stat.st_dev),
        os.minor(descriptor_stat.st_dev),
    )
    for line in fdinfo.splitlines():
        fields = line.split()
        if len(fields) < 7 or fields[0] != "lock:":
            continue
        if fields[2:5] != ["FLOCK", "ADVISORY", "WRITE"]:
            continue
        identity = fields[6].rsplit(":", maxsplit=2)
        if len(identity) != 3:
            continue
        try:
            device = (int(identity[0], 16), int(identity[1], 16))
            inode = int(identity[2])
        except ValueError:
            continue
        if device == expected_device and inode == descriptor_stat.st_ino:
            return
    raise ExperimentLaneLockError(
        "experiment-lane lease FD does not hold a write FLOCK"
    )


def _set_close_on_exec(descriptor: int) -> None:
    """Set and verify the borrowed descriptor's child-local CLOEXEC flag.

    :param descriptor: Borrowed canonical lock descriptor.
    :raises ExperimentLaneLockError: If the descriptor flags cannot be secured.
    """
    try:
        descriptor_flags = fcntl.fcntl(descriptor, fcntl.F_GETFD)
        fcntl.fcntl(
            descriptor,
            fcntl.F_SETFD,
            descriptor_flags | fcntl.FD_CLOEXEC,
        )
        verified_flags = fcntl.fcntl(descriptor, fcntl.F_GETFD)
    except OSError as error:
        raise ExperimentLaneLockError(
            "experiment-lane lease FD flags are unavailable"
        ) from error
    if verified_flags & fcntl.FD_CLOEXEC == 0:
        raise ExperimentLaneLockError("experiment-lane lease FD is not close-on-exec")


def _write_all(descriptor: int, payload: bytes) -> None:
    """Write one complete lock-owner record.

    :param descriptor: Owned canonical lock descriptor.
    :param payload: Complete owner record.
    :raises ExperimentLaneLockError: If the record cannot be written.
    """
    written = 0
    try:
        while written < len(payload):
            count = os.write(descriptor, payload[written:])
            if count <= 0:
                raise ExperimentLaneLockError(
                    "experiment-lane owner record write made no progress"
                )
            written += count
    except OSError as error:
        raise ExperimentLaneLockError(
            "cannot write experiment-lane owner record"
        ) from error


def _acquire_owned_lock() -> int:
    """Create or open and exclusively acquire the canonical lane lock.

    :returns: Owned close-on-exec lock descriptor.
    :raises ExperimentLaneLockError: If the lane cannot be safely acquired.
    """
    lock_directory = _validate_lock_directory(create=True)
    try:
        directory_descriptor = os.open(
            lock_directory,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
    except OSError as error:
        raise ExperimentLaneLockError(
            "cannot open experiment-lane lock directory"
        ) from error
    descriptor: int | None = None
    lock_file_open = False
    try:
        directory_stat = os.fstat(directory_descriptor)
        path_stat = lock_directory.lstat()
        if (directory_stat.st_dev, directory_stat.st_ino) != (
            path_stat.st_dev,
            path_stat.st_ino,
        ) or directory_stat.st_uid != os.geteuid():
            raise ExperimentLaneLockError(
                "experiment-lane lock directory identity differs"
            )
        flags = os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW
        try:
            descriptor = os.open(
                EXPERIMENT_LANE_LOCK.name,
                flags | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=directory_descriptor,
            )
        except OSError as error:
            if error.errno != errno.EEXIST:
                raise ExperimentLaneLockError(
                    "cannot create experiment-lane lock"
                ) from error
            try:
                descriptor = os.open(
                    EXPERIMENT_LANE_LOCK.name,
                    flags,
                    dir_fd=directory_descriptor,
                )
            except OSError as open_error:
                raise ExperimentLaneLockError(
                    "cannot open experiment-lane lock"
                ) from open_error
        os.fsync(directory_descriptor)
        lock_file_open = True
    finally:
        os.close(directory_descriptor)
        if lock_file_open is False and descriptor is not None:
            os.close(descriptor)
    if descriptor is None:
        raise ExperimentLaneLockError("experiment-lane lock descriptor is absent")
    keep_open = False
    try:
        _validate_lock_file(descriptor)
        try:
            fcntl.flock(
                descriptor,
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except OSError as error:
            if error.errno in {errno.EACCES, errno.EAGAIN}:
                raise ExperimentLaneLockError(
                    f"experiment lane is already held at {EXPERIMENT_LANE_LOCK}"
                ) from error
            raise ExperimentLaneLockError(
                "cannot acquire experiment-lane lock"
            ) from error
        os.ftruncate(descriptor, 0)
        os.lseek(descriptor, 0, os.SEEK_SET)
        _write_all(descriptor, f"pid={os.getpid()}\n".encode())
        os.fsync(descriptor)
        keep_open = True
        return descriptor
    finally:
        if keep_open is False:
            os.close(descriptor)


def _require_unused_descriptor(descriptor: int) -> None:
    """Require one nonreserved descriptor number that is not already open.

    :param descriptor: Fixed descriptor requested by the transaction owner.
    :raises ExperimentLaneLockError: If the number is invalid or already open.
    """
    if type(descriptor) is not int or descriptor < 3:
        raise ExperimentLaneLockError(
            "experiment-lane lease descriptor is invalid or reserved"
        )
    try:
        os.fstat(descriptor)
    except OSError as error:
        if error.errno == errno.EBADF:
            return
        raise ExperimentLaneLockError(
            "cannot inspect experiment-lane lease descriptor"
        ) from error
    raise ExperimentLaneLockError("experiment-lane lease descriptor is already open")


def acquire_experiment_lane_lease(descriptor: int) -> int:
    """Acquire the canonical lease at one fixed inheritable descriptor.

    The returned descriptor shares the exact locked open-file description
    created by the canonical acquisition path. The caller owns the descriptor
    and must keep it open for the complete transaction.

    :param descriptor: Unused descriptor number inherited by the transaction.
    :returns: The requested inheritable descriptor.
    :raises ExperimentLaneLockError: If acquisition or fixed-FD placement fails.
    """
    _require_unused_descriptor(descriptor)
    owned_descriptor = _acquire_owned_lock()
    active_descriptor = owned_descriptor
    lease_ready = False
    try:
        if owned_descriptor != descriptor:
            try:
                duplicated_descriptor = fcntl.fcntl(
                    owned_descriptor,
                    fcntl.F_DUPFD,
                    descriptor,
                )
            except OSError as error:
                raise ExperimentLaneLockError(
                    "cannot place experiment-lane lease at fixed descriptor"
                ) from error
            if duplicated_descriptor != descriptor:
                os.close(duplicated_descriptor)
                raise ExperimentLaneLockError(
                    "experiment-lane lease descriptor became occupied"
                )
            active_descriptor = descriptor
            os.close(owned_descriptor)
        try:
            os.set_inheritable(active_descriptor, True)
        except OSError as error:
            raise ExperimentLaneLockError(
                "cannot make experiment-lane lease inheritable"
            ) from error
        _validate_lock_file(active_descriptor)
        if os.get_inheritable(active_descriptor) is False:
            raise ExperimentLaneLockError(
                "experiment-lane lease descriptor is not inheritable"
            )
        lease_ready = True
        return active_descriptor
    finally:
        if lease_ready is False:
            os.close(active_descriptor)


def _validate_borrowed_lock(descriptor: int) -> None:
    """Validate one already-held canonical lock OFD without taking ownership.

    :param descriptor: Inherited transaction lock descriptor.
    :raises ExperimentLaneLockError: If exact OFD ownership cannot be proven.
    """
    _validate_lock_directory(create=False)
    descriptor_stat = _validate_lock_file(descriptor)
    _set_close_on_exec(descriptor)
    _require_borrowed_write_flock(descriptor, descriptor_stat)

    try:
        competitor = os.open(
            EXPERIMENT_LANE_LOCK,
            os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
    except OSError as error:
        raise ExperimentLaneLockError(
            "cannot open experiment-lane lock competitor"
        ) from error
    try:
        _validate_lock_file(competitor)
        try:
            fcntl.flock(
                competitor,
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except OSError as error:
            if error.errno not in {errno.EACCES, errno.EAGAIN}:
                raise ExperimentLaneLockError(
                    "cannot probe the experiment-lane lease owner"
                ) from error
        else:
            fcntl.flock(competitor, fcntl.LOCK_UN)
            raise ExperimentLaneLockError("experiment-lane lease is not already locked")
        try:
            fcntl.flock(
                descriptor,
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except OSError as error:
            if error.errno in {errno.EACCES, errno.EAGAIN}:
                raise ExperimentLaneLockError(
                    "experiment-lane lease FD does not own the active lock"
                ) from error
            raise ExperimentLaneLockError(
                "cannot validate experiment-lane lease FD"
            ) from error
    finally:
        os.close(competitor)


class ExperimentLaneLock:
    """Own or borrow the canonical host-wide experiment-lane lock."""

    _borrowed: bool
    _descriptor: int | None
    _entered: bool

    def __init__(self, lease_fd: int | None = None) -> None:
        """Configure direct ownership or an inherited transaction lease.

        :param lease_fd: Existing locked OFD, or None to acquire directly.
        :raises ExperimentLaneLockError: If a borrowed descriptor is malformed.
        """
        if lease_fd is not None and (type(lease_fd) is not int or lease_fd < 3):
            raise ExperimentLaneLockError("experiment-lane lease FD is invalid")
        self._borrowed = lease_fd is not None
        self._descriptor = lease_fd
        self._entered = False

    def __enter__(self) -> "ExperimentLaneLock":
        """Establish the configured lane-ownership boundary.

        :returns: Active lock boundary.
        :raises ExperimentLaneLockError: If ownership cannot be proven.
        """
        if self._entered:
            raise ExperimentLaneLockError(
                "experiment-lane lock context is already active"
            )
        if self._borrowed:
            if self._descriptor is None:
                raise ExperimentLaneLockError(
                    "borrowed experiment-lane descriptor is absent"
                )
            _validate_borrowed_lock(self._descriptor)
        else:
            self._descriptor = _acquire_owned_lock()
        self._entered = True
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """End direct ownership while leaving borrowed ownership untouched.

        :param exception_type: Active exception type, when any.
        :param exception: Active exception, when any.
        :param traceback: Active traceback, when any.
        """
        del exception_type, exception, traceback
        if self._entered is False:
            return
        self._entered = False
        if self._borrowed:
            return
        descriptor = self._descriptor
        self._descriptor = None
        if descriptor is None:
            raise ExperimentLaneLockError("owned experiment-lane descriptor is absent")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        except OSError as error:
            raise ExperimentLaneLockError(
                "cannot release experiment-lane lock"
            ) from error
        finally:
            os.close(descriptor)
