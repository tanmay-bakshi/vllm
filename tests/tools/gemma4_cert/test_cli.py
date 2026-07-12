from pathlib import Path

import pytest

from tools.gemma4_cert.cli import _parser


def test_offline_attestation_rejects_observed_evidence_options(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit):
        _parser().parse_args(
            [
                "attest",
                "--arm",
                str(tmp_path / "arm"),
                "--path",
                "attest/router.json",
                "--repository",
                str(tmp_path),
                "--role",
                "router",
                "--engine-boot-uuid",
                "boot",
                "--observed-effective-config",
                str(tmp_path / "config.json"),
            ]
        )
