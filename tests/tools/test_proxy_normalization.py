# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Host-only tests for ephemeral experiment-proxy candidate proof."""

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import TracebackType
from unittest import mock

from tools.gemma4_pd.campaign_ops import proxy_normalization

_BOOT_ID = "11111111-1111-1111-1111-111111111111"
_LEGACY_IDENTITY = proxy_normalization.ProcessIdentity(
    boot_id=_BOOT_ID,
    pid=600494,
    process_group_id=600494,
    session_id=600494,
    start_time_ticks=100,
)
_CANDIDATE_IDENTITY = proxy_normalization.ProcessIdentity(
    boot_id=_BOOT_ID,
    pid=700001,
    process_group_id=700001,
    session_id=700001,
    start_time_ticks=200,
)
_CANDIDATE_CHILD_IDENTITY = proxy_normalization.ProcessIdentity(
    boot_id=_BOOT_ID,
    pid=700003,
    process_group_id=_CANDIDATE_IDENTITY.process_group_id,
    session_id=_CANDIDATE_IDENTITY.session_id,
    start_time_ticks=201,
)


class FakeLaneLock:
    """Provide a no-op lane boundary for host-only state-machine tests."""

    active_depth = 0

    def __init__(self, lease_fd: int | None = None) -> None:
        """Accept the same lock-construction contract.

        :param lease_fd: Ignored fake inherited descriptor.
        """
        self.lease_fd = lease_fd

    def __enter__(self) -> "FakeLaneLock":
        """Enter the fake ownership boundary.

        :returns: This fake lock.
        """
        type(self).active_depth += 1
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Leave the fake ownership boundary.

        :param exception_type: Active exception type, when any.
        :param exception: Active exception, when any.
        :param traceback: Active traceback, when any.
        """
        del exception_type, exception, traceback
        type(self).active_depth -= 1


class FakeSystem:
    """Model exact proxy identities while performing no process or network IO."""

    _config: proxy_normalization.NormalizationConfig
    active: dict[int, proxy_normalization.ProcessIdentity]
    candidate_port: int
    fail_candidate_stops: int
    fail_semantic: bool
    gpu_requests: int
    legacy_proxy_script: Path
    legacy_workdir: Path
    operations: list[str]
    protected_generation: int
    signal_during_launch: bool
    signal_during_stop: bool

    def __init__(
        self,
        config: proxy_normalization.NormalizationConfig,
        legacy_proxy_script: Path,
        legacy_workdir: Path,
    ) -> None:
        """Create one healthy degraded lane fixture.

        :param config: Transaction configuration used by legacy evidence.
        :param legacy_proxy_script: Distinct live proxy source path.
        :param legacy_workdir: Distinct live proxy working directory.
        """
        self._config = config
        self.active = {_LEGACY_IDENTITY.pid: _LEGACY_IDENTITY}
        self.candidate_port = config.candidate_port
        self.fail_candidate_stops = 0
        self.fail_semantic = False
        self.gpu_requests = 0
        self.legacy_proxy_script = legacy_proxy_script
        self.legacy_workdir = legacy_workdir
        self.operations = []
        self.protected_generation = 0
        self.signal_during_launch = False
        self.signal_during_stop = False

    def _legacy_cmdline(self) -> bytes:
        """Build the exact degraded proxy argument vector.

        :returns: NUL-terminated legacy cmdline.
        """
        argv = (
            os.fsencode(self._config.python),
            os.fsencode(self.legacy_proxy_script),
            b"--port",
            b"8812",
            b"--prefiller-hosts",
            b"localhost",
            b"--prefiller-ports",
            b"8810",
            b"--decoder-hosts",
            b"localhost",
            b"--decoder-ports",
            b"8811",
        )
        return b"\x00".join(argv) + b"\x00"

    def capture_process(self, pid: int) -> dict[str, object]:
        """Return sealed legacy evidence with deleted runtime mappings.

        :param pid: Expected legacy PID.
        :returns: Minimal complete evidence consumed by the orchestrator.
        """
        if pid != _LEGACY_IDENTITY.pid or pid not in self.active:
            raise proxy_normalization.NormalizationError("fake process is absent")
        cmdline = self._legacy_cmdline()
        environment_entries = (
            b"PATH=/usr/bin",
            b"PWD=" + os.fsencode(self.legacy_workdir),
        )
        environment = b"\x00".join(environment_entries) + b"\x00"
        return {
            "cmdline_hex": cmdline.hex(),
            "cmdline_sha256": proxy_normalization._sha256_bytes(cmdline),
            "cwd": {
                "link_hex": os.fsencode(self.legacy_workdir).hex(),
                "stat": {},
            },
            "environment_entries_hex": [entry.hex() for entry in environment_entries],
            "environment_sha256": proxy_normalization._sha256_bytes(environment),
            "executable": {
                "link_hex": b"/usr/bin/python3.12 (deleted)".hex(),
                "sha256": "1" * 64,
                "stat": {},
            },
            "fds": [],
            "group": [proxy_normalization.asdict(_LEGACY_IDENTITY)],
            "identity": proxy_normalization.asdict(_LEGACY_IDENTITY),
            "maps": [
                {"path_hex": b"/usr/lib/python3.12/lib-dynload/_ssl.so (deleted)".hex()}
            ],
            "maps_hex": "",
            "maps_sha256": proxy_normalization._sha256_bytes(b""),
            "resource_limits": [
                {"hard": -1, "name": name, "soft": -1}
                for name in proxy_normalization._RESOURCE_NAMES
            ],
            "status_hex": "",
            "status_sha256": proxy_normalization._sha256_bytes(b""),
            "umask": 0o002,
        }

    def identity(self, pid: int) -> proxy_normalization.ProcessIdentity:
        """Return one active fake process identity.

        :param pid: Process identifier.
        :returns: Active identity.
        """
        identity = self.active.get(pid)
        if identity is None:
            raise proxy_normalization.NormalizationError("fake process is absent")
        return identity

    def require_identity(self, expected: proxy_normalization.ProcessIdentity) -> None:
        """Require one exact active identity.

        :param expected: Expected identity.
        """
        if self.active.get(expected.pid) != expected:
            raise proxy_normalization.NormalizationError("fake identity drifted")

    def process_group_identities(
        self, pgid: int
    ) -> tuple[proxy_normalization.ProcessIdentity, ...]:
        """Return the exact fake group roster.

        :param pgid: Process-group identifier.
        :returns: Sorted active group identities.
        """
        group = tuple(
            sorted(
                (
                    identity
                    for identity in self.active.values()
                    if identity.process_group_id == pgid
                ),
                key=lambda item: item.pid,
            )
        )
        if len(group) == 0:
            raise proxy_normalization.NormalizationError("fake group is absent")
        return group

    def listener_snapshot(self, port: int) -> dict[str, object]:
        """Return one active fake listener.

        :param port: Listener port.
        :returns: Listener and exact owner identity.
        """
        identity: proxy_normalization.ProcessIdentity | None = None
        if port == proxy_normalization.CANONICAL_PORT:
            if _LEGACY_IDENTITY.pid in self.active:
                identity = _LEGACY_IDENTITY
        elif port == self.candidate_port and _CANDIDATE_IDENTITY.pid in self.active:
            identity = _CANDIDATE_IDENTITY
        if identity is None:
            raise proxy_normalization.NormalizationError("fake listener is absent")
        return {
            "listener": {
                "family": "ipv4",
                "host": proxy_normalization.CANONICAL_HOST,
                "inode": port * 10,
                "port": port,
            },
            "owners": [proxy_normalization.asdict(identity)],
        }

    def protected_snapshot(self, ports: tuple[int, ...]) -> dict[str, object]:
        """Return stable protected identities unrelated to fake proxy PIDs.

        :param ports: Exact protected port roster.
        :returns: Stable protected snapshot.
        """
        return {
            str(port): {
                "generation": self.protected_generation,
                "port": port,
            }
            for port in ports
        }

    def require_port_absent(self, port: int) -> None:
        """Require the fake candidate or canonical listener to be absent.

        :param port: Port that must be unbound.
        """
        try:
            self.listener_snapshot(port)
        except proxy_normalization.NormalizationError:
            return
        raise proxy_normalization.NormalizationError("fake listener is present")

    def idle_snapshot(self, urls: tuple[str, ...]) -> dict[str, object]:
        """Return exact zero-request engine evidence.

        :param urls: Metrics endpoint roster.
        :returns: Stable idle snapshot.
        """
        return {url: {"running": 0.0, "status": 200, "waiting": 0.0} for url in urls}

    @staticmethod
    def _write_log(path: Path, payload: bytes) -> None:
        """Create one private fake process log.

        :param path: Exclusive log path.
        :param payload: Log payload.
        """
        path.write_bytes(payload)
        path.chmod(0o600)

    def launch(
        self,
        contract: proxy_normalization.LaunchContract,
        port: int,
        stdout_path: Path,
        stderr_path: Path,
    ) -> proxy_normalization.ProcessIdentity:
        """Activate one alternate or canonical fake process.

        :param contract: Normalized launch contract.
        :param port: Requested listener port.
        :param stdout_path: Exclusive stdout log.
        :param stderr_path: Exclusive stderr log.
        :returns: New fake identity.
        """
        del contract
        if FakeLaneLock.active_depth != 1:
            raise AssertionError("candidate launch escaped the lane lease")
        if self.signal_during_launch:
            os.kill(os.getpid(), proxy_normalization.signal.SIGTERM)
        state_path = stdout_path.parent / "state.json"
        if port == self.candidate_port:
            state = json.loads(state_path.read_bytes())
            if state["phase"] != proxy_normalization.Phase.PREPARED.value:
                raise AssertionError(
                    "candidate launched before immutable evidence was sealed"
                )
            identity = _CANDIDATE_IDENTITY
            self.operations.append("launch-candidate")
        else:
            raise proxy_normalization.NormalizationError("unexpected fake launch port")
        self._write_log(stdout_path, b"fake stdout\n")
        self._write_log(stderr_path, b"")
        self.active[identity.pid] = identity
        return identity

    def wait_for_listener_gate(
        self,
        contract: proxy_normalization.LaunchContract,
        identity: proxy_normalization.ProcessIdentity,
        port: int,
        timeout_seconds: float,
    ) -> dict[str, object]:
        """Return successful request-free listener evidence.

        :param contract: Expected launch contract.
        :param identity: Expected active identity.
        :param port: Expected listener port.
        :param timeout_seconds: Bounded timeout.
        :returns: Fake gate evidence.
        """
        del timeout_seconds
        self.require_identity(identity)
        self.listener_snapshot(port)
        return {
            "application_requests_sent": 0,
            "listener": self.listener_snapshot(port),
            "process": self.require_launch_contract(contract, identity, port),
        }

    def require_launch_contract(
        self,
        contract: proxy_normalization.LaunchContract,
        identity: proxy_normalization.ProcessIdentity,
        port: int,
    ) -> dict[str, object]:
        """Require one exact active fake launch.

        :param contract: Expected launch contract.
        :param identity: Expected active identity.
        :param port: Expected listener port.
        :returns: Minimal fake process evidence.
        """
        del contract
        self.require_identity(identity)
        self.listener_snapshot(port)
        return {"identity": proxy_normalization.asdict(identity), "port": port}

    def stop_group(
        self,
        identities: tuple[proxy_normalization.ProcessIdentity, ...],
        timeout_seconds: float,
    ) -> None:
        """Remove one exact singleton fake process.

        :param identities: Exact group roster.
        :param timeout_seconds: Ignored fake timeout.
        """
        del timeout_seconds
        if FakeLaneLock.active_depth != 1:
            raise AssertionError("candidate cleanup escaped the lane lease")
        if len(identities) != 1:
            raise proxy_normalization.NormalizationError("fake group is not singleton")
        identity = identities[0]
        self.require_identity(identity)
        self.operations.append(f"stop-{identity.pid}")
        if identity == _CANDIDATE_IDENTITY and self.fail_candidate_stops > 0:
            self.fail_candidate_stops -= 1
            raise proxy_normalization.NormalizationError(
                "injected candidate termination interruption"
            )
        if self.signal_during_stop:
            self.signal_during_stop = False
            os.kill(os.getpid(), proxy_normalization.signal.SIGTERM)
        del self.active[identity.pid]

    def semantic_gate(
        self,
        port: int,
        model: str,
        timeout_seconds: float,
    ) -> dict[str, object]:
        """Record the only fake operation representing a GPU request.

        :param port: Expected candidate port.
        :param model: Bound served model.
        :param timeout_seconds: Bounded inference timeout.
        :returns: Deterministic fake semantic evidence.
        """
        del model, timeout_seconds
        if FakeLaneLock.active_depth != 1:
            raise AssertionError("semantic request escaped the lane lease")
        self.listener_snapshot(port)
        self.gpu_requests += 1
        self.operations.append("semantic-gate")
        if self.fail_semantic:
            raise proxy_normalization.NormalizationError("injected semantic failure")
        return {"texts_equal": True}


class StopGroupSystem(proxy_normalization.LinuxNormalizationSystem):
    """Drive the Linux group-termination algorithm with deterministic rosters."""

    _rosters: list[tuple[proxy_normalization.ProcessIdentity, ...]]

    def __init__(
        self, rosters: list[tuple[proxy_normalization.ProcessIdentity, ...]]
    ) -> None:
        """Create one deterministic process-group sequence.

        :param rosters: Consecutive group snapshots returned to the algorithm.
        """
        self._rosters = list(rosters)

    def _process_group_identities_allow_absent(
        self, pgid: int
    ) -> tuple[proxy_normalization.ProcessIdentity, ...]:
        """Return the next deterministic group snapshot.

        :param pgid: Expected process-group identifier.
        :returns: Next exact group snapshot.
        """
        if pgid != _CANDIDATE_IDENTITY.process_group_id:
            raise AssertionError("unexpected process group")
        if len(self._rosters) > 1:
            return self._rosters.pop(0)
        return self._rosters[0]

    def require_identity(self, expected: proxy_normalization.ProcessIdentity) -> None:
        """Accept identities already supplied by the deterministic roster.

        :param expected: Expected process identity.
        """
        del expected


class ProxyNormalizationTest(unittest.TestCase):
    """Verify durable safety boundaries for ephemeral candidate proof."""

    def setUp(self) -> None:
        """Replace the real host-wide lock with a test-local context."""
        FakeLaneLock.active_depth = 0
        patcher = mock.patch.object(
            proxy_normalization,
            "ExperimentLaneLock",
            FakeLaneLock,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def _rewrite_json(path: Path, value: object, *, mode: int) -> bytes:
        """Replace one test artifact with canonical JSON.

        :param path: Artifact path.
        :param value: Replacement JSON value.
        :param mode: Final artifact mode.
        :returns: Canonical replacement payload.
        """
        payload = proxy_normalization._canonical_json(value)
        path.chmod(0o600)
        path.write_bytes(payload)
        path.chmod(mode)
        return payload

    @classmethod
    def _rewrite_evidence(
        cls,
        artifacts: Path,
        name: str,
        value: object,
    ) -> None:
        """Replace evidence and bind its new digest into the test state.

        :param artifacts: Transaction artifact directory.
        :param name: Evidence filename.
        :param value: Replacement evidence value.
        """
        payload = cls._rewrite_json(artifacts / name, value, mode=0o400)
        state_path = artifacts / "state.json"
        state = json.loads(state_path.read_bytes())
        found = False
        for item in state["evidence"]:
            if item["name"] != name:
                continue
            item["sha256"] = proxy_normalization._sha256_bytes(payload)
            found = True
            break
        if found is False:
            raise AssertionError("test state does not reference replacement evidence")
        cls._rewrite_json(state_path, state, mode=0o600)

    def _fixture(
        self,
    ) -> tuple[
        tempfile.TemporaryDirectory[str],
        Path,
        proxy_normalization.NormalizationConfig,
        FakeSystem,
    ]:
        """Create canonical physical files and one unused artifact path.

        :returns: Temporary owner, artifact path, config, and fake host.
        """
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name).resolve()
        root.chmod(0o700)
        legacy_workdir = root / "primary"
        legacy_workdir.mkdir()
        legacy_proxy_script = legacy_workdir / "toy_proxy_server.py"
        legacy_proxy_script.write_text("print('proxy fixture')\n")
        workdir = root / "clean-clone"
        workdir.mkdir()
        proxy_script = workdir / "toy_proxy_server.py"
        proxy_script.write_bytes(legacy_proxy_script.read_bytes())
        venv = root / "venv"
        venv_bin = venv / "bin"
        venv_bin.mkdir(parents=True)
        python = venv_bin / "python3"
        python.symlink_to(Path(sys.executable).resolve(strict=True))
        (venv / "pyvenv.cfg").write_text(
            "home = "
            f"{Path(sys.executable).resolve(strict=True).parent}\n"
            "include-system-site-packages = true\n"
            f"version = {sys.version_info.major}.{sys.version_info.minor}\n"
        )
        config = proxy_normalization.build_config(
            legacy_pid=_LEGACY_IDENTITY.pid,
            candidate_port=18812,
            python=python,
            proxy_script=proxy_script,
            workdir=workdir,
            model="gemma-4-31B-it",
            readiness_timeout_seconds=5.0,
        )
        return (
            temporary,
            root / "artifacts",
            config,
            FakeSystem(config, legacy_proxy_script, legacy_workdir),
        )

    def test_prepare_is_application_free_and_leaves_no_candidate(self) -> None:
        """Seal preflight evidence without launching or requesting through a proxy."""
        temporary, artifacts, config, system = self._fixture()
        with temporary:
            state = proxy_normalization.prepare_candidate(
                artifacts,
                config,
                lease_fd=64,
                system=system,
            )
            summary = proxy_normalization.inspect_transaction(artifacts)

        self.assertEqual(state.phase, proxy_normalization.Phase.PREPARED)
        self.assertIsNone(state.candidate_identity)
        self.assertEqual(system.gpu_requests, 0)
        self.assertEqual(system.operations, [])
        self.assertIn(_LEGACY_IDENTITY.pid, system.active)
        self.assertNotIn(_CANDIDATE_IDENTITY.pid, system.active)
        self.assertEqual(FakeLaneLock.active_depth, 0)
        self.assertIn("PAUSE", summary["next_action"])

    def test_prepare_separates_observed_and_selected_launch_contracts(self) -> None:
        """Bind distinct digest-equal source paths without conflating their identity."""
        temporary, artifacts, config, system = self._fixture()
        with temporary:
            proxy_normalization.prepare_candidate(
                artifacts, config, lease_fd=64, system=system
            )
            store = proxy_normalization.ArtifactStore(artifacts, create=False)
            before = store.load_json("before.json", expected_mode=0o400)
            observed, candidate = proxy_normalization._launch_contracts(
                config,
                system.capture_process(config.legacy_pid),
            )

        self.assertNotEqual(observed.proxy_script, candidate.proxy_script)
        self.assertNotEqual(observed.workdir, os.fsencode(candidate.workdir))
        self.assertEqual(observed.proxy_script_sha256, config.proxy_script_sha256)
        self.assertEqual(
            dict(candidate.environment)[b"PWD"],
            os.fsencode(system.legacy_workdir),
        )
        self.assertEqual(before["legacy_launch_contract"], observed.to_record())
        self.assertEqual(
            before["launch_contract"],
            candidate.to_record(config.candidate_port),
        )
        self.assertEqual(
            before["launch_contract"]["argv_hex"],
            [value.hex() for value in candidate.argv(config.candidate_port)],
        )

    def test_legacy_source_drift_blocks_candidate_launch(self) -> None:
        """Recheck current observed source bytes before any candidate or GPU action."""
        temporary, artifacts, config, system = self._fixture()
        with temporary:
            proxy_normalization.prepare_candidate(
                artifacts, config, lease_fd=64, system=system
            )
            system.legacy_proxy_script.write_text("print('drifted primary')\n")
            with self.assertRaisesRegex(
                proxy_normalization.NormalizationError,
                "source digests differ",
            ):
                proxy_normalization.run_semantic_gate(
                    artifacts,
                    confirmation=proxy_normalization.GPU_CONFIRMATION,
                    lease_fd=64,
                    system=system,
                    inference_timeout_seconds=10.0,
                )

        self.assertEqual(system.operations, [])
        self.assertEqual(system.gpu_requests, 0)
        self.assertNotIn(_CANDIDATE_IDENTITY.pid, system.active)

    def test_semantic_gate_requires_exact_confirmation_before_launch(self) -> None:
        """Refuse the first GPU action before a candidate process exists."""
        temporary, artifacts, config, system = self._fixture()
        with temporary:
            proxy_normalization.prepare_candidate(
                artifacts, config, lease_fd=64, system=system
            )
            with self.assertRaisesRegex(
                proxy_normalization.NormalizationError,
                "first GPU request",
            ):
                proxy_normalization.run_semantic_gate(
                    artifacts,
                    confirmation="yes",
                    lease_fd=64,
                    system=system,
                    inference_timeout_seconds=10.0,
                )

        self.assertEqual(system.gpu_requests, 0)
        self.assertEqual(system.operations, [])
        self.assertNotIn(_CANDIDATE_IDENTITY.pid, system.active)

    def test_semantic_gate_owns_launch_request_and_cleanup_under_one_lease(
        self,
    ) -> None:
        """Publish proof only after the ephemeral candidate is completely absent."""
        temporary, artifacts, config, system = self._fixture()
        with temporary:
            proxy_normalization.prepare_candidate(
                artifacts, config, lease_fd=64, system=system
            )
            state = proxy_normalization.run_semantic_gate(
                artifacts,
                confirmation=proxy_normalization.GPU_CONFIRMATION,
                lease_fd=64,
                system=system,
                inference_timeout_seconds=10.0,
            )
            authenticated = proxy_normalization.ArtifactStore(
                artifacts, create=False
            ).load_state()

        self.assertEqual(state.phase, proxy_normalization.Phase.SEMANTIC_PROVEN)
        self.assertEqual(authenticated, state)
        self.assertIsNone(state.candidate_identity)
        self.assertEqual(system.gpu_requests, 1)
        self.assertIn(_LEGACY_IDENTITY.pid, system.active)
        self.assertNotIn(_CANDIDATE_IDENTITY.pid, system.active)
        self.assertEqual(
            system.operations,
            [
                "launch-candidate",
                "semantic-gate",
                f"stop-{_CANDIDATE_IDENTITY.pid}",
            ],
        )
        self.assertEqual(FakeLaneLock.active_depth, 0)

    def test_signal_during_launch_is_deferred_until_candidate_cleanup(self) -> None:
        """Close the fork-to-identity signal window without orphaning a listener."""
        temporary, artifacts, config, system = self._fixture()
        system.signal_during_launch = True
        with temporary:
            proxy_normalization.prepare_candidate(
                artifacts, config, lease_fd=64, system=system
            )
            with self.assertRaises(proxy_normalization.NormalizationInterrupted):
                proxy_normalization.run_semantic_gate(
                    artifacts,
                    confirmation=proxy_normalization.GPU_CONFIRMATION,
                    lease_fd=64,
                    system=system,
                    inference_timeout_seconds=10.0,
                )
            state = proxy_normalization.ArtifactStore(
                artifacts, create=False
            ).load_state()

        self.assertEqual(state.phase, proxy_normalization.Phase.FAILED)
        self.assertIsNone(state.candidate_identity)
        self.assertNotIn(_CANDIDATE_IDENTITY.pid, system.active)
        self.assertEqual(system.gpu_requests, 0)
        self.assertEqual(FakeLaneLock.active_depth, 0)

    def test_signal_during_cleanup_is_raised_after_terminal_publication(self) -> None:
        """Latch cleanup-time interruption until semantic proof is durable."""
        temporary, artifacts, config, system = self._fixture()
        system.signal_during_stop = True
        with temporary:
            proxy_normalization.prepare_candidate(
                artifacts, config, lease_fd=64, system=system
            )
            with self.assertRaises(proxy_normalization.NormalizationInterrupted):
                proxy_normalization.run_semantic_gate(
                    artifacts,
                    confirmation=proxy_normalization.GPU_CONFIRMATION,
                    lease_fd=64,
                    system=system,
                    inference_timeout_seconds=10.0,
                )
            state = proxy_normalization.ArtifactStore(
                artifacts, create=False
            ).load_state()

        self.assertEqual(state.phase, proxy_normalization.Phase.SEMANTIC_PROVEN)
        self.assertIsNone(state.candidate_identity)
        self.assertNotIn(_CANDIDATE_IDENTITY.pid, system.active)

    def test_signal_during_log_capture_is_raised_after_terminal_publication(
        self,
    ) -> None:
        """Latch log-capture interruption until semantic proof is durable."""
        temporary, artifacts, config, system = self._fixture()
        original_log_evidence = proxy_normalization._log_evidence
        signal_sent = False

        def log_evidence(path: Path) -> dict[str, object]:
            """Deliver one signal before capturing the first candidate log.

            :param path: Candidate log path.
            :returns: Complete candidate log evidence.
            """
            nonlocal signal_sent
            if signal_sent is False:
                signal_sent = True
                os.kill(os.getpid(), proxy_normalization.signal.SIGTERM)
            return original_log_evidence(path)

        with temporary:
            proxy_normalization.prepare_candidate(
                artifacts, config, lease_fd=64, system=system
            )
            with (
                mock.patch.object(
                    proxy_normalization,
                    "_log_evidence",
                    side_effect=log_evidence,
                ),
                self.assertRaises(proxy_normalization.NormalizationInterrupted),
            ):
                proxy_normalization.run_semantic_gate(
                    artifacts,
                    confirmation=proxy_normalization.GPU_CONFIRMATION,
                    lease_fd=64,
                    system=system,
                    inference_timeout_seconds=10.0,
                )
            state = proxy_normalization.ArtifactStore(
                artifacts, create=False
            ).load_state()

        self.assertTrue(signal_sent)
        self.assertEqual(state.phase, proxy_normalization.Phase.SEMANTIC_PROVEN)
        self.assertIsNone(state.candidate_identity)
        self.assertNotIn(_CANDIDATE_IDENTITY.pid, system.active)

    def test_log_capture_failure_publishes_inactive_terminal_failure(self) -> None:
        """Replace a dead-candidate running head when terminal logs are unavailable."""
        temporary, artifacts, config, system = self._fixture()
        with temporary:
            proxy_normalization.prepare_candidate(
                artifacts, config, lease_fd=64, system=system
            )
            with (
                mock.patch.object(
                    proxy_normalization,
                    "_log_evidence",
                    side_effect=proxy_normalization.NormalizationError(
                        "injected log capture failure"
                    ),
                ),
                self.assertRaisesRegex(
                    proxy_normalization.NormalizationError,
                    "injected log capture failure",
                ),
            ):
                proxy_normalization.run_semantic_gate(
                    artifacts,
                    confirmation=proxy_normalization.GPU_CONFIRMATION,
                    lease_fd=64,
                    system=system,
                    inference_timeout_seconds=10.0,
                )
            state = proxy_normalization.ArtifactStore(
                artifacts, create=False
            ).load_state()

        self.assertEqual(state.phase, proxy_normalization.Phase.SEMANTIC_FAILED)
        self.assertIsNone(state.candidate_identity)
        self.assertNotIn(_CANDIDATE_IDENTITY.pid, system.active)

    def test_signal_during_state_write_is_raised_after_terminal_publication(
        self,
    ) -> None:
        """Latch publication-time interruption until the terminal state is durable."""
        temporary, artifacts, config, system = self._fixture()
        original_write_state = proxy_normalization.ArtifactStore.write_state
        signal_sent = False

        def write_state(
            store: proxy_normalization.ArtifactStore,
            state: proxy_normalization.TransactionState,
        ) -> None:
            """Deliver one signal while publishing semantic proof.

            :param store: Transaction artifact store.
            :param state: State being published.
            """
            nonlocal signal_sent
            if (
                signal_sent is False
                and state.phase == proxy_normalization.Phase.SEMANTIC_PROVEN
            ):
                signal_sent = True
                os.kill(os.getpid(), proxy_normalization.signal.SIGTERM)
            original_write_state(store, state)

        with temporary:
            proxy_normalization.prepare_candidate(
                artifacts, config, lease_fd=64, system=system
            )
            with (
                mock.patch.object(
                    proxy_normalization.ArtifactStore,
                    "write_state",
                    new=write_state,
                ),
                self.assertRaises(proxy_normalization.NormalizationInterrupted),
            ):
                proxy_normalization.run_semantic_gate(
                    artifacts,
                    confirmation=proxy_normalization.GPU_CONFIRMATION,
                    lease_fd=64,
                    system=system,
                    inference_timeout_seconds=10.0,
                )
            state = proxy_normalization.ArtifactStore(
                artifacts, create=False
            ).load_state()

        self.assertTrue(signal_sent)
        self.assertEqual(state.phase, proxy_normalization.Phase.SEMANTIC_PROVEN)
        self.assertIsNone(state.candidate_identity)
        self.assertNotIn(_CANDIDATE_IDENTITY.pid, system.active)

    def test_semantic_failure_cleans_candidate_before_failure_publication(
        self,
    ) -> None:
        """Retain failed semantic evidence only after candidate disappearance."""
        temporary, artifacts, config, system = self._fixture()
        system.fail_semantic = True
        with temporary:
            proxy_normalization.prepare_candidate(
                artifacts, config, lease_fd=64, system=system
            )
            with self.assertRaisesRegex(
                proxy_normalization.NormalizationError,
                "injected semantic failure",
            ):
                proxy_normalization.run_semantic_gate(
                    artifacts,
                    confirmation=proxy_normalization.GPU_CONFIRMATION,
                    lease_fd=64,
                    system=system,
                    inference_timeout_seconds=10.0,
                )
            state = proxy_normalization.ArtifactStore(
                artifacts, create=False
            ).load_state()

        self.assertEqual(state.phase, proxy_normalization.Phase.SEMANTIC_FAILED)
        self.assertIsNone(state.candidate_identity)
        self.assertNotIn(_CANDIDATE_IDENTITY.pid, system.active)
        self.assertIn(_LEGACY_IDENTITY.pid, system.active)
        self.assertEqual(FakeLaneLock.active_depth, 0)

    def test_cleanup_failure_preserves_exact_candidate_identity(self) -> None:
        """Fail closed with the live identity when complete cleanup is unproved."""
        temporary, artifacts, config, system = self._fixture()
        system.fail_candidate_stops = 1
        with temporary:
            proxy_normalization.prepare_candidate(
                artifacts, config, lease_fd=64, system=system
            )
            with self.assertRaisesRegex(
                proxy_normalization.NormalizationError,
                "candidate cleanup failed",
            ):
                proxy_normalization.run_semantic_gate(
                    artifacts,
                    confirmation=proxy_normalization.GPU_CONFIRMATION,
                    lease_fd=64,
                    system=system,
                    inference_timeout_seconds=10.0,
                )
            state = proxy_normalization.ArtifactStore(
                artifacts, create=False
            ).load_state()

        self.assertEqual(state.phase, proxy_normalization.Phase.FAILED)
        self.assertEqual(state.candidate_identity, _CANDIDATE_IDENTITY)
        self.assertIn(_CANDIDATE_IDENTITY.pid, system.active)

    def test_group_cleanup_signals_members_created_after_launch(self) -> None:
        """Terminate a same-session child discovered only during cleanup."""
        system = StopGroupSystem(
            [
                (_CANDIDATE_IDENTITY, _CANDIDATE_CHILD_IDENTITY),
                (),
            ]
        )
        with (
            mock.patch.object(
                proxy_normalization.os,
                "pidfd_open",
                side_effect=lambda pid, flags: pid + flags + 1000,
                create=True,
            ),
            mock.patch.object(proxy_normalization.os, "close"),
            mock.patch.object(
                proxy_normalization.signal,
                "pidfd_send_signal",
                create=True,
            ) as send_signal,
            mock.patch.object(proxy_normalization.time, "sleep"),
        ):
            system.stop_group((_CANDIDATE_IDENTITY,), 5.0)

        signaled_pidfds = {call.args[0] for call in send_signal.call_args_list}
        self.assertEqual(
            signaled_pidfds,
            {
                _CANDIDATE_IDENTITY.pid + 1000,
                _CANDIDATE_CHILD_IDENTITY.pid + 1000,
            },
        )

    def test_group_cleanup_rejects_leader_pid_reuse(self) -> None:
        """Never signal a new process incarnation that reused the leader PID."""
        reused = proxy_normalization.ProcessIdentity(
            boot_id=_BOOT_ID,
            pid=_CANDIDATE_IDENTITY.pid,
            process_group_id=_CANDIDATE_IDENTITY.process_group_id,
            session_id=_CANDIDATE_IDENTITY.session_id,
            start_time_ticks=_CANDIDATE_IDENTITY.start_time_ticks + 1,
        )
        system = StopGroupSystem([(reused,)])
        with self.assertRaisesRegex(
            proxy_normalization.NormalizationError,
            "ownership drifted",
        ):
            system.stop_group((_CANDIDATE_IDENTITY,), 5.0)

    def test_state_phase_cannot_claim_semantic_proof_without_evidence(self) -> None:
        """Reject an owner-written phase advance lacking its exact evidence chain."""
        temporary, artifacts, config, system = self._fixture()
        with temporary:
            proxy_normalization.prepare_candidate(
                artifacts, config, lease_fd=64, system=system
            )
            state_path = artifacts / "state.json"
            record = json.loads(state_path.read_bytes())
            record["phase"] = proxy_normalization.Phase.SEMANTIC_PROVEN.value
            state_path.write_bytes(proxy_normalization._canonical_json(record))
            with self.assertRaisesRegex(
                proxy_normalization.NormalizationError,
                "evidence sequence",
            ):
                proxy_normalization.ArtifactStore(artifacts, create=False).load_state()

    def test_serialized_candidate_contract_cannot_differ_from_recomputation(
        self,
    ) -> None:
        """Reject digest-authentic candidate argv that differs from legacy inputs."""
        temporary, artifacts, config, system = self._fixture()
        with temporary:
            proxy_normalization.prepare_candidate(
                artifacts, config, lease_fd=64, system=system
            )
            before_path = artifacts / "before.json"
            before = json.loads(before_path.read_bytes())
            before["launch_contract"]["argv_hex"][5] = b"18813".hex()
            self._rewrite_json(before_path, before, mode=0o400)
            with self.assertRaisesRegex(
                proxy_normalization.NormalizationError,
                "serialized candidate launch contract differs",
            ):
                proxy_normalization._load_before(
                    proxy_normalization.ArtifactStore(artifacts, create=False),
                    config,
                )

    def test_serialized_candidate_contract_rejects_extra_fields(self) -> None:
        """Reject candidate launch evidence outside its exact schema."""
        temporary, artifacts, config, system = self._fixture()
        with temporary:
            proxy_normalization.prepare_candidate(
                artifacts, config, lease_fd=64, system=system
            )
            before_path = artifacts / "before.json"
            before = json.loads(before_path.read_bytes())
            before["launch_contract"]["unexpected"] = True
            self._rewrite_json(before_path, before, mode=0o400)
            with self.assertRaisesRegex(
                proxy_normalization.NormalizationError,
                "before.launch_contract field set differs",
            ):
                proxy_normalization._load_before(
                    proxy_normalization.ArtifactStore(artifacts, create=False),
                    config,
                )

    def test_preparation_failure_rejects_malformed_historical_identity(self) -> None:
        """Validate the optional historical candidate identity at exact schema."""
        temporary, artifacts, config, system = self._fixture()
        system.signal_during_launch = True
        with temporary:
            proxy_normalization.prepare_candidate(
                artifacts, config, lease_fd=64, system=system
            )
            with self.assertRaises(proxy_normalization.NormalizationInterrupted):
                proxy_normalization.run_semantic_gate(
                    artifacts,
                    confirmation=proxy_normalization.GPU_CONFIRMATION,
                    lease_fd=64,
                    system=system,
                    inference_timeout_seconds=10.0,
                )
            name = "semantic-preparation-failure.json"
            evidence = json.loads((artifacts / name).read_bytes())
            evidence["candidate_identity"]["unexpected"] = True
            self._rewrite_evidence(artifacts, name, evidence)
            with self.assertRaisesRegex(
                proxy_normalization.NormalizationError,
                "candidate_identity field set differs",
            ):
                proxy_normalization.ArtifactStore(artifacts, create=False).load_state()

    def test_preparation_failure_cleanup_requires_inactive_state(self) -> None:
        """Reject an active FAILED head when preparation evidence proves cleanup."""
        temporary, artifacts, config, system = self._fixture()
        system.signal_during_launch = True
        with temporary:
            proxy_normalization.prepare_candidate(
                artifacts, config, lease_fd=64, system=system
            )
            with self.assertRaises(proxy_normalization.NormalizationInterrupted):
                proxy_normalization.run_semantic_gate(
                    artifacts,
                    confirmation=proxy_normalization.GPU_CONFIRMATION,
                    lease_fd=64,
                    system=system,
                    inference_timeout_seconds=10.0,
                )
            state_path = artifacts / "state.json"
            state = json.loads(state_path.read_bytes())
            state["candidate_identity"] = proxy_normalization.asdict(
                _CANDIDATE_IDENTITY
            )
            self._rewrite_json(state_path, state, mode=0o600)
            with self.assertRaisesRegex(
                proxy_normalization.NormalizationError,
                "inactive semantic preparation failure",
            ):
                proxy_normalization.ArtifactStore(artifacts, create=False).load_state()

    def test_legacy_argv_rejects_every_unconsumed_argument(self) -> None:
        """Reject unknown legacy behavior instead of silently normalizing it away."""
        temporary, _, config, system = self._fixture()
        with temporary:
            legacy = system.capture_process(config.legacy_pid)
            cmdline = bytes.fromhex(str(legacy["cmdline_hex"]))
            entries = proxy_normalization._split_nul_payload(
                cmdline, label="legacy cmdline"
            )
            changed = (*entries, b"--reload", b"true")
            legacy["cmdline_hex"] = (b"\x00".join(changed) + b"\x00").hex()
            with self.assertRaisesRegex(
                proxy_normalization.NormalizationError,
                "unknown option",
            ):
                proxy_normalization._launch_contracts(config, legacy)

    def test_python_entrypoint_retains_venv_only_package_semantics(self) -> None:
        """Bind argv0 to the venv entrypoint rather than its resolved executable."""
        temporary, _, config, system = self._fixture()
        with temporary:
            site_packages = (
                config.python.parent.parent
                / "lib"
                / f"python{sys.version_info.major}.{sys.version_info.minor}"
                / "site-packages"
            )
            site_packages.mkdir(parents=True)
            module_name = "_proxy_normalization_venv_only"
            (site_packages / f"{module_name}.py").write_text("VALUE = 73\n")
            environment = dict(os.environ)
            environment.pop("PYTHONPATH", None)
            through_entrypoint = subprocess.run(
                [
                    config.python,
                    "-I",
                    "-c",
                    f"import {module_name}; print({module_name}.VALUE)",
                ],
                cwd="/",
                env=environment,
                capture_output=True,
                check=False,
                text=True,
            )
            through_executable = subprocess.run(
                [
                    config.python_executable,
                    "-I",
                    "-c",
                    f"import {module_name}",
                ],
                cwd="/",
                env=environment,
                capture_output=True,
                check=False,
                text=True,
            )
            _, contract = proxy_normalization._launch_contracts(
                config, system.capture_process(config.legacy_pid)
            )

        self.assertNotEqual(config.python, config.python_executable)
        self.assertEqual(through_entrypoint.returncode, 0)
        self.assertEqual(through_entrypoint.stdout.strip(), "73")
        self.assertNotEqual(through_executable.returncode, 0)
        self.assertEqual(
            contract.argv(config.candidate_port)[0], os.fsencode(config.python)
        )

    def test_cutover_is_absent_from_the_executable_surface(self) -> None:
        """Expose no canonical mutation action before replay binding exists."""
        parser = proxy_normalization.build_parser()
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["cutover"])

    def test_immutable_evidence_tamper_is_detected(self) -> None:
        """Reject a byte change in the sealed pre-mutation snapshot."""
        temporary, artifacts, config, system = self._fixture()
        with temporary:
            proxy_normalization.prepare_candidate(
                artifacts, config, lease_fd=64, system=system
            )
            before = artifacts / "before.json"
            before.chmod(0o600)
            before.write_bytes(before.read_bytes() + b" ")
            before.chmod(0o400)
            with self.assertRaisesRegex(
                proxy_normalization.NormalizationError,
                "before evidence digest changed",
            ):
                proxy_normalization.ArtifactStore(artifacts, create=False).load_state()

    def test_stream_parser_requires_one_done_terminated_nonempty_choice(self) -> None:
        """Assemble strict SSE text and reject missing completion termination."""
        response = proxy_normalization.HttpResponse(
            status=200,
            headers=(),
            body=(
                b'data: {"choices":[{"text":"one"}]}\n\n'
                b'data: {"choices":[{"text":", two"}]}\n\n'
                b"data: [DONE]\n\n"
            ),
        )
        text, events = proxy_normalization.LinuxNormalizationSystem._stream_text(
            response
        )
        self.assertEqual(text, "one, two")
        self.assertEqual(len(events), 2)
        with self.assertRaisesRegex(
            proxy_normalization.NormalizationError,
            "incomplete",
        ):
            proxy_normalization.LinuxNormalizationSystem._stream_text(
                proxy_normalization.HttpResponse(
                    status=200,
                    headers=(),
                    body=b'data: {"choices":[{"text":"one"}]}\n\n',
                )
            )

    def test_artifact_parent_must_be_private(self) -> None:
        """Refuse transaction evidence beneath a group-readable parent."""
        temporary, artifacts, config, system = self._fixture()
        with temporary:
            artifacts.parent.chmod(0o755)
            with self.assertRaisesRegex(
                proxy_normalization.NormalizationError,
                "private euid-owned physical directory",
            ):
                proxy_normalization.prepare_candidate(
                    artifacts, config, lease_fd=64, system=system
                )


if __name__ == "__main__":
    unittest.main()
