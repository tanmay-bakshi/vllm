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
    plan = subparsers.add_parser("plan")
    plan.add_argument("--config", type=Path, required=True)
    plan.add_argument("--scenario")
    self_test = subparsers.add_parser("self-test")
    self_test.add_argument("--config", type=Path, required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--artifact-root", type=Path, required=True)
    run.add_argument("--scenario", required=True)
    run.add_argument("--transport-arm", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--run-directory", type=Path, required=True)
    return parser


def main() -> None:
    """Validate, describe, or explicitly execute the transport campaign."""
    arguments = _parser().parse_args()
    if arguments.command == "validate":
        from tools.gemma4_pd.nixl_micro_rig.campaign_validator import (
            validate_campaign,
        )

        validation = validate_campaign(arguments.run_directory)
        print(json.dumps(validation.to_json(), indent=2, sort_keys=True))
        if validation.passed is False:
            raise SystemExit(1)
        return

    config = load_config(arguments.config)
    if arguments.command == "plan":
        scenarios = (
            config.scenarios
            if arguments.scenario is None
            else (config.scenario(arguments.scenario),)
        )
        print(
            json.dumps(
                [
                    describe_plan(
                        config,
                        scenario,
                        build_configured_plan(arguments.config, config, scenario),
                    )
                    for scenario in scenarios
                ],
                indent=2,
                sort_keys=True,
            )
        )
        return
    if arguments.command == "self-test":
        print(json.dumps(integrity_contract_self_test(), indent=2, sort_keys=True))
        return

    from tools.gemma4_pd.nixl_micro_rig.launcher import RunStatus, run_campaign
    from tools.gemma4_pd.nixl_micro_rig.selection import RunSelection

    selection = RunSelection(
        scenario_name=arguments.scenario,
        transport_arm_name=arguments.transport_arm,
    )
    selection.validate(config)
    result = run_campaign(
        arguments.config,
        arguments.artifact_root,
        selection=selection,
    )
    print(
        json.dumps(
            {
                "run_directory": str(result.run_directory),
                "status": result.status.value,
            },
            sort_keys=True,
        )
    )
    if result.status is RunStatus.FAIL:
        raise SystemExit(2)
    if result.status is RunStatus.INVALID:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
