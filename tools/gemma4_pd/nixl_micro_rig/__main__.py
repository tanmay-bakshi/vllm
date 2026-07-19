"""Command-line entrypoint for the Gemma 4 NIXL transport micro-rig."""

import argparse
import json
from pathlib import Path

from tools.gemma4_pd.nixl_micro_rig.config import load_config
from tools.gemma4_pd.nixl_micro_rig.geometry import (
    build_configured_plan,
    describe_plan,
)
from tools.gemma4_pd.nixl_micro_rig.integrity_check import (
    integrity_contract_self_test,
)


def _parser() -> argparse.ArgumentParser:
    """Construct the explicit host-only and GPU campaign CLI.

    :returns: Configured command-line parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "self-test"):
        command = subparsers.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--artifact-root", type=Path, required=True)
    capture = subparsers.add_parser("capture-handshake")
    capture.add_argument("--config", type=Path, required=True)
    capture.add_argument("--code-root", type=Path, required=True)
    capture.add_argument("--producer-host", required=True)
    capture.add_argument("--producer-port", type=int, required=True)
    capture.add_argument("--producer-engine-id")
    capture.add_argument("--decoder-host", required=True)
    capture.add_argument("--decoder-port", type=int, required=True)
    capture.add_argument("--decoder-engine-id")
    capture.add_argument("--model-config", type=Path, required=True)
    capture.add_argument("--expected-model-config-sha256", required=True)
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--result-output", type=Path, required=True)
    capture.add_argument("--timeout-seconds", type=float, default=5.0)
    verify = subparsers.add_parser("verify-handshake")
    verify.add_argument("--config", type=Path, required=True)
    verify.add_argument("--code-root", type=Path, required=True)
    verify.add_argument("--capture", type=Path, required=True)
    verify.add_argument("--expected-sha256", required=True)
    verify.add_argument("--expected-model-config-sha256", required=True)
    verify.add_argument("--result-output", type=Path, required=True)
    return parser


def main() -> None:
    """Validate, describe, or explicitly execute the transport campaign."""
    arguments = _parser().parse_args()
    config = load_config(arguments.config)
    if arguments.command == "plan":
        print(
            json.dumps(
                [
                    describe_plan(
                        config,
                        scenario,
                        build_configured_plan(arguments.config, config, scenario),
                    )
                    for scenario in config.scenarios
                ],
                indent=2,
                sort_keys=True,
            )
        )
        return
    if arguments.command == "self-test":
        print(json.dumps(integrity_contract_self_test(), indent=2, sort_keys=True))
        return
    if arguments.command in {"capture-handshake", "verify-handshake"}:
        from tools.gemma4_pd.nixl_micro_rig.live_handshake import (
            CaptureVerificationResult,
            CaptureWriteResult,
            capture_live_handshake,
            verify_live_handshake_capture,
            write_result_output,
        )

        result: CaptureWriteResult | CaptureVerificationResult
        if arguments.command == "capture-handshake":
            result = capture_live_handshake(
                config_path=arguments.config,
                config=config,
                code_root=arguments.code_root,
                producer_host=arguments.producer_host,
                producer_port=arguments.producer_port,
                producer_engine_id=arguments.producer_engine_id,
                decoder_host=arguments.decoder_host,
                decoder_port=arguments.decoder_port,
                decoder_engine_id=arguments.decoder_engine_id,
                model_config_path=arguments.model_config,
                expected_model_config_sha256=(arguments.expected_model_config_sha256),
                output_path=arguments.output,
                timeout_seconds=arguments.timeout_seconds,
            )
        else:
            result = verify_live_handshake_capture(
                capture_path=arguments.capture,
                expected_sha256=arguments.expected_sha256,
                config_path=arguments.config,
                config=config,
                code_root=arguments.code_root,
                expected_model_config_sha256=(arguments.expected_model_config_sha256),
            )
        write_result_output(arguments.result_output, result)
        print(arguments.result_output)
        return

    from tools.gemma4_pd.nixl_micro_rig.launcher import run_campaign

    run_directory = run_campaign(arguments.config, arguments.artifact_root)
    print(run_directory)


if __name__ == "__main__":
    main()
