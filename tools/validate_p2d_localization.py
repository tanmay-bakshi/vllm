# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate a complete set of authoritative P-to-D localization artifacts."""

import argparse
import json
from pathlib import Path

from vllm.distributed.kv_transfer.nixl_localization_validator import (
    validate_localization_artifacts,
)


def parse_args() -> argparse.Namespace:
    """Parse validator command-line arguments.

    :returns: Parsed artifact paths.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "artifacts",
        nargs="+",
        type=Path,
        help="Every P and D worker .p2d.msgpack artifact from one run",
    )
    return parser.parse_args()


def main() -> int:
    """Validate artifacts and print one machine-readable verdict.

    :returns: Zero for a complete clean trace, one for invalid artifacts, or
        two when the trace is complete enough to localize a divergence.
    """
    args = parse_args()
    report = validate_localization_artifacts(tuple(args.artifacts))
    print(
        json.dumps(
            {
                "passed": report.passed,
                "artifact_count": report.artifact_count,
                "physical_pull_count": report.physical_pull_count,
                "verified_pull_count": report.verified_pull_count,
                "excluded_outcome_count": report.excluded_outcome_count,
                "terminal_outcome_count": report.terminal_outcome_count,
                "divergences": [
                    {
                        "child_request_id": divergence.child_request_id,
                        "observer_engine_id": divergence.observer_engine_id,
                        "observer_rank": divergence.observer_rank,
                        "source_rank": divergence.source_rank,
                        "edge": divergence.edge,
                        "errors": list(divergence.errors),
                    }
                    for divergence in report.divergences
                ],
                "errors": list(report.errors),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if len(report.errors) > 0:
        return 1
    if len(report.divergences) > 0:
        return 2
    return 0 if report.passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
