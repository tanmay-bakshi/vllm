"""Fresh-interpreter entrypoint for one native micro-rig role."""

import argparse
import json
import os
import resource
import time
import traceback
from pathlib import Path

from tools.gemma4_pd.nixl_micro_rig.config import load_config
from tools.gemma4_pd.nixl_micro_rig.outcome import RigCorrectnessFailure
from tools.gemma4_pd.nixl_micro_rig.selection import RunSelection

_ROLE_NOFILE_LIMIT = 65_535


def _parser() -> argparse.ArgumentParser:
    """Build the exact internal role CLI.

    :returns: Configured argument parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--transport-arm", required=True)
    parser.add_argument("--input-bundle-fingerprint", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--role", choices=("producer", "consumer"), required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--artifact-directory", type=Path, required=True)
    parser.add_argument("--expected-cuda-visibility", required=True)
    parser.add_argument("--producer-cuda-visibility", required=True)
    return parser


def _write_terminal(path: Path, terminal: dict[str, object]) -> None:
    """Atomically persist and flush one authoritative role terminal.

    :param path: Final role-terminal artifact path.
    :param terminal: Strict terminal record.
    """
    payload = (
        json.dumps(terminal, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _bind_role_nofile_limit() -> None:
    """Match the production P/D file-descriptor limit before native imports."""
    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    if hard_limit != 1_048_576:
        raise RuntimeError(f"role RLIMIT_NOFILE hard limit differs: {hard_limit}")
    if soft_limit != _ROLE_NOFILE_LIMIT:
        resource.setrlimit(
            resource.RLIMIT_NOFILE,
            (_ROLE_NOFILE_LIMIT, hard_limit),
        )


def main() -> None:
    """Import and execute the selected role after startup environment binding."""
    arguments = _parser().parse_args()
    _bind_role_nofile_limit()
    selection = RunSelection(
        scenario_name=arguments.scenario,
        transport_arm_name=arguments.transport_arm,
    )
    config = load_config(arguments.config)
    selection.validate(config)
    terminal_path = arguments.artifact_directory / (
        f"producer-{arguments.rank}-terminal.json"
        if arguments.role == "producer"
        else "consumer-terminal.json"
    )
    terminal: dict[str, object]
    exit_error: Exception | None = None
    try:
        from tools.gemma4_pd.nixl_micro_rig.roles import run_consumer, run_producer

        if arguments.role == "producer":
            run_producer(
                config_path=arguments.config,
                selection=selection,
                input_bundle_fingerprint=arguments.input_bundle_fingerprint,
                run_id=arguments.run_id,
                rank=arguments.rank,
                artifact_directory=arguments.artifact_directory,
                expected_cuda_visibility=arguments.expected_cuda_visibility,
            )
        else:
            run_consumer(
                config_path=arguments.config,
                selection=selection,
                input_bundle_fingerprint=arguments.input_bundle_fingerprint,
                run_id=arguments.run_id,
                artifact_directory=arguments.artifact_directory,
                expected_cuda_visibility=arguments.expected_cuda_visibility,
                producer_cuda_visibility=arguments.producer_cuda_visibility,
            )
        terminal = {
            "schema_version": 1,
            "run_id": arguments.run_id,
            "config_fingerprint": config.fingerprint,
            "selection": selection.to_json(),
            "input_bundle_fingerprint": arguments.input_bundle_fingerprint,
            "role": arguments.role,
            "rank": arguments.rank,
            "status": "PASS",
            "detail": "role completed cleanly",
            "traceback": "",
            "completed_ns": time.time_ns(),
        }
    except RigCorrectnessFailure as error:
        exit_error = error
        terminal = {
            "schema_version": 1,
            "run_id": arguments.run_id,
            "config_fingerprint": config.fingerprint,
            "selection": selection.to_json(),
            "input_bundle_fingerprint": arguments.input_bundle_fingerprint,
            "role": arguments.role,
            "rank": arguments.rank,
            "status": "FAIL",
            "detail": str(error),
            "traceback": traceback.format_exc(),
            "completed_ns": time.time_ns(),
        }
    except Exception as error:
        exit_error = error
        terminal = {
            "schema_version": 1,
            "run_id": arguments.run_id,
            "config_fingerprint": config.fingerprint,
            "selection": selection.to_json(),
            "input_bundle_fingerprint": arguments.input_bundle_fingerprint,
            "role": arguments.role,
            "rank": arguments.rank,
            "status": "INVALID",
            "detail": str(error),
            "traceback": traceback.format_exc(),
            "completed_ns": time.time_ns(),
        }
    _write_terminal(terminal_path, terminal)
    if exit_error is not None:
        raise exit_error


if __name__ == "__main__":
    main()
