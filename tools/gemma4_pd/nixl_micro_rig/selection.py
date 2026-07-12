"""Typed campaign selection shared by launcher and native roles."""

import hashlib
import json
from dataclasses import dataclass
from typing import ClassVar

from tools.gemma4_pd.nixl_micro_rig.config import RigConfig


@dataclass(frozen=True, slots=True)
class RunSelection:
    """Select exactly one configured scenario and transport arm.

    :ivar scenario_name: Exact scenario identity from the full input config.
    :ivar transport_arm_name: Exact transport-arm identity from the full config.
    """

    SCHEMA_VERSION: ClassVar[int] = 1

    scenario_name: str
    transport_arm_name: str

    def __post_init__(self) -> None:
        """Reject non-string or empty selection identities."""
        for field, value in (
            ("scenario_name", self.scenario_name),
            ("transport_arm_name", self.transport_arm_name),
        ):
            if type(value) is not str:
                raise TypeError(f"{field} must be an exact string")
            if len(value) == 0:
                raise ValueError(f"{field} must not be empty")

    def validate(self, config: RigConfig) -> None:
        """Require both selected identities to exist in the full config.

        :param config: Untouched full campaign configuration.
        """
        config.scenario(self.scenario_name)
        config.transport_arm(self.transport_arm_name)

    def to_json(self) -> dict[str, object]:
        """Return the exact JSON selection identity.

        :returns: Strict JSON-compatible identity.
        """
        return {
            "schema_version": self.SCHEMA_VERSION,
            "scenario": self.scenario_name,
            "transport_arm": self.transport_arm_name,
        }

    @property
    def fingerprint(self) -> str:
        """Return the canonical SHA-256 selection fingerprint.

        :returns: Lowercase SHA-256 digest.
        """
        payload = json.dumps(
            self.to_json(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(payload).hexdigest()
