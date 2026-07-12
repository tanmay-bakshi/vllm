import argparse
import json
import sys
from pathlib import Path

from tools.gemma4_pd.lane_lifecycle.lifecycle import (
    DEFAULT_STORM37B_ARCHIVE,
    LifecycleError,
    capture,
    restore,
    stop,
    verify,
)
from tools.gemma4_pd.lane_lifecycle.schema import SnapshotValidationError
from tools.gemma4_pd.lane_lifecycle.system import HostInspectionError


def _parser() -> argparse.ArgumentParser:
    """Build the lifecycle command-line parser.

    :returns: Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Capture, quiesce, restore, and verify the Gemma 4 experiment lane "
            "while treating GPUs 6 and 7 as immutable production state."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture_parser = subparsers.add_parser("capture")
    capture_parser.add_argument("--artifact-dir", required=True, type=Path)
    capture_parser.add_argument(
        "--storm-archive", type=Path, default=DEFAULT_STORM37B_ARCHIVE
    )

    stop_parser = subparsers.add_parser("stop")
    stop_parser.add_argument("--artifact-dir", required=True, type=Path)
    stop_mode = stop_parser.add_mutually_exclusive_group(required=True)
    stop_mode.add_argument("--dry-run", action="store_true")
    stop_mode.add_argument("--execute", action="store_true")
    stop_parser.add_argument("--term-timeout-seconds", type=float, default=30.0)
    stop_parser.add_argument("--kill-timeout-seconds", type=float, default=15.0)

    restore_parser = subparsers.add_parser("restore")
    restore_parser.add_argument("--artifact-dir", required=True, type=Path)
    restore_mode = restore_parser.add_mutually_exclusive_group(required=True)
    restore_mode.add_argument("--dry-run", action="store_true")
    restore_mode.add_argument("--execute", action="store_true")
    restore_parser.add_argument(
        "--readiness-timeout-seconds", type=float, default=1800.0
    )
    restore_parser.add_argument("--rollback-timeout-seconds", type=float, default=30.0)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--artifact-dir", required=True, type=Path)
    verify_parser.add_argument(
        "--expected-state",
        choices=("captured", "running", "stopped"),
        required=True,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the lane lifecycle command.

    :param argv: Optional command-line arguments.
    :returns: Process exit status.
    """
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "capture":
            snapshot = capture(
                arguments.artifact_dir,
                storm_archive=arguments.storm_archive,
            )
            result: dict[str, object] = {
                "status": "PASS",
                "operation": "capture",
                "artifact_directory": str(arguments.artifact_dir.absolute()),
                "captured_at_utc": snapshot.captured_at_utc,
                "storm37b_sha256": snapshot.storm_archive.sha256,
            }
        elif arguments.command == "stop":
            result = stop(
                arguments.artifact_dir,
                dry_run=arguments.dry_run,
                term_timeout_seconds=arguments.term_timeout_seconds,
                kill_timeout_seconds=arguments.kill_timeout_seconds,
            )
        elif arguments.command == "restore":
            result = restore(
                arguments.artifact_dir,
                dry_run=arguments.dry_run,
                readiness_timeout_seconds=arguments.readiness_timeout_seconds,
                rollback_timeout_seconds=arguments.rollback_timeout_seconds,
            )
        else:
            result = verify(
                arguments.artifact_dir,
                expected_state=arguments.expected_state,
            )
    except (LifecycleError, HostInspectionError, SnapshotValidationError) as error:
        print(
            json.dumps(
                {"status": "FAIL", "error": str(error)},
                sort_keys=True,
                ensure_ascii=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True))
    return 0
