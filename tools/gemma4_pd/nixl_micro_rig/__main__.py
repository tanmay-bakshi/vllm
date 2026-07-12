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

    from tools.gemma4_pd.nixl_micro_rig.launcher import run_campaign

    run_directory = run_campaign(arguments.config, arguments.artifact_root)
    print(run_directory)


if __name__ == "__main__":
    main()
