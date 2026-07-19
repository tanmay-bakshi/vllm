"""Typed configuration for the Gemma 4 NIXL transport micro-rig."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from tools.gemma4_pd.nixl_micro_rig.handshake import (
    SEMANTIC_HANDSHAKE_CONNECTOR_VERSION,
    SemanticHandshakeError,
    SemanticHandshakeProfile,
    load_semantic_handshake_profile,
)


class ConfigError(ValueError):
    """Report an invalid or ambiguous micro-rig configuration."""


def _require_keys(
    value: dict[str, object],
    *,
    required: set[str],
    optional: set[str],
    context: str,
) -> None:
    """Validate an object's exact configuration vocabulary.

    :param value: Object to validate.
    :param required: Keys that must be present.
    :param optional: Keys that may be present.
    :param context: Human-readable object name for diagnostics.
    :raises ConfigError: If a key is missing or unknown.
    """
    missing = required - value.keys()
    unknown = value.keys() - required - optional
    if len(missing) > 0:
        raise ConfigError(f"{context} is missing keys: {sorted(missing)}")
    if len(unknown) > 0:
        raise ConfigError(f"{context} has unknown keys: {sorted(unknown)}")


def _as_dict(value: object, context: str) -> dict[str, object]:
    """Require a JSON object.

    :param value: Parsed JSON value.
    :param context: Human-readable object name for diagnostics.
    :returns: The validated object.
    :raises ConfigError: If the value is not an object.
    """
    if not isinstance(value, dict):
        raise ConfigError(f"{context} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise ConfigError(f"{context} keys must be strings")
    return value


def _as_int(value: object, context: str) -> int:
    """Require a non-boolean integer.

    :param value: Parsed JSON value.
    :param context: Human-readable field name for diagnostics.
    :returns: The validated integer.
    :raises ConfigError: If the value is not an integer.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{context} must be an integer")
    return value


def _as_str(value: object, context: str) -> str:
    """Require a non-empty string.

    :param value: Parsed JSON value.
    :param context: Human-readable field name for diagnostics.
    :returns: The validated string.
    :raises ConfigError: If the value is not a non-empty string.
    """
    if not isinstance(value, str) or len(value) == 0:
        raise ConfigError(f"{context} must be a non-empty string")
    return value


def _as_relative_path(value: object, context: str) -> str:
    """Require a confined, non-empty relative path.

    :param value: Parsed JSON value.
    :param context: Human-readable field name for diagnostics.
    :returns: Validated relative path text.
    :raises ConfigError: If the path is absolute or can escape its root.
    """
    text = _as_str(value, context)
    path = Path(text)
    if path.is_absolute() or ".." in path.parts:
        raise ConfigError(f"{context} must be a confined relative path")
    return text


def _as_list(value: object, context: str) -> list[object]:
    """Require a JSON array.

    :param value: Parsed JSON value.
    :param context: Human-readable field name for diagnostics.
    :returns: The validated array.
    :raises ConfigError: If the value is not an array.
    """
    if not isinstance(value, list):
        raise ConfigError(f"{context} must be an array")
    return value


@dataclass(frozen=True)
class RegionConfig:
    """Describe one P-side physical registration region.

    :ivar name: Stable region identity.
    :ivar row_bytes: Bytes occupied by one P-rank block row.
    """

    name: str
    row_bytes: int

    def __post_init__(self) -> None:
        if len(self.name) == 0:
            raise ConfigError("region name must not be empty")
        if self.row_bytes <= 0 or self.row_bytes % 4 != 0:
            raise ConfigError("region row_bytes must be positive and 4-byte aligned")

    @classmethod
    def from_json(cls, value: object, index: int) -> "RegionConfig":
        """Parse one region.

        :param value: Parsed region object.
        :param index: Region index used in diagnostics.
        :returns: Parsed region configuration.
        """
        context = f"regions[{index}]"
        obj = _as_dict(value, context)
        _require_keys(
            obj,
            required={"name", "row_bytes"},
            optional=set(),
            context=context,
        )
        return cls(
            name=_as_str(obj["name"], f"{context}.name"),
            row_bytes=_as_int(obj["row_bytes"], f"{context}.row_bytes"),
        )


@dataclass(frozen=True)
class GroupConfig:
    """Describe one group-wise prefix-trim input.

    :ivar index: Stable KV-cache group index.
    :ivar name: Human-readable group identity.
    :ivar token_capacity: Exact number of request tokens represented by one
        physical source block for this group.
    :ivar destination_plane_count: One for packed single-plane destinations or
        two for standard K/V destinations.
    :ivar remote_position_count: P positions before prefix trimming.
    :ivar local_position_count: D positions remaining after a prefix hit.
    :ivar owned_region_indices: Authoritative physical regions for the group.
    """

    index: int
    name: str
    token_capacity: int
    destination_plane_count: int
    remote_position_count: int
    local_position_count: int
    owned_region_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ConfigError("group index must be non-negative")
        if len(self.name) == 0:
            raise ConfigError("group name must not be empty")
        if self.token_capacity <= 0:
            raise ConfigError("group token_capacity must be positive")
        if self.destination_plane_count not in (1, 2):
            raise ConfigError("group destination_plane_count must be one or two")
        if self.remote_position_count <= 0:
            raise ConfigError("group remote_position_count must be positive")
        if self.local_position_count < 0:
            raise ConfigError("group local_position_count must be non-negative")
        remote_positions_per_local = 2 if self.destination_plane_count == 1 else 1
        local_capacity = (
            self.remote_position_count + remote_positions_per_local - 1
        ) // remote_positions_per_local
        if self.local_position_count > local_capacity:
            raise ConfigError(
                "group local_position_count exceeds its destination-plane capacity"
            )
        if len(self.owned_region_indices) == 0:
            raise ConfigError("group owned_region_indices must not be empty")
        if min(self.owned_region_indices) < 0:
            raise ConfigError("group owned_region_indices must be non-negative")
        if len(set(self.owned_region_indices)) != len(self.owned_region_indices):
            raise ConfigError("group owned_region_indices must be unique")

    @classmethod
    def from_json(cls, value: object, array_index: int) -> "GroupConfig":
        """Parse one group.

        :param value: Parsed group object.
        :param array_index: Array index used in diagnostics.
        :returns: Parsed group configuration.
        """
        context = f"groups[{array_index}]"
        obj = _as_dict(value, context)
        _require_keys(
            obj,
            required={
                "index",
                "name",
                "token_capacity",
                "destination_plane_count",
                "remote_position_count",
                "local_position_count",
                "owned_region_indices",
            },
            optional=set(),
            context=context,
        )
        return cls(
            index=_as_int(obj["index"], f"{context}.index"),
            name=_as_str(obj["name"], f"{context}.name"),
            token_capacity=_as_int(obj["token_capacity"], f"{context}.token_capacity"),
            destination_plane_count=_as_int(
                obj["destination_plane_count"],
                f"{context}.destination_plane_count",
            ),
            remote_position_count=_as_int(
                obj["remote_position_count"],
                f"{context}.remote_position_count",
            ),
            local_position_count=_as_int(
                obj["local_position_count"],
                f"{context}.local_position_count",
            ),
            owned_region_indices=tuple(
                _as_int(item, f"{context}.owned_region_indices[{region_index}]")
                for region_index, item in enumerate(
                    _as_list(
                        obj["owned_region_indices"],
                        f"{context}.owned_region_indices",
                    )
                )
            ),
        )


@dataclass(frozen=True)
class ScenarioConfig:
    """Describe one source-offset and run-fragmentation arm.

    :ivar name: Stable scenario identity.
    :ivar source_start_block: First physical P block ID.
    :ivar source_end_block: Last selected physical P block ID.
    :ivar run_count: Number of consecutive source-ID runs.
    :ivar iterations: Number of transfers performed in this scenario.
    :ivar staging_offsets_mib: Registration-relative staging offsets, cycled
        across iterations to exercise allocation and reuse generations.
    :ivar replay_manifest: Optional captured group-wise remote/local pairing.
    """

    name: str
    source_start_block: int
    source_end_block: int
    run_count: int
    iterations: int
    staging_offsets_mib: tuple[int, ...]
    replay_manifest: str | None

    def __post_init__(self) -> None:
        if len(self.name) == 0:
            raise ConfigError("scenario name must not be empty")
        if self.source_start_block < 0:
            raise ConfigError("source_start_block must be non-negative")
        if self.source_end_block < self.source_start_block:
            raise ConfigError("source_end_block must not precede source_start_block")
        if self.run_count < 0:
            raise ConfigError("run_count must be non-negative")
        if self.iterations <= 0:
            raise ConfigError("iterations must be positive")
        if len(self.staging_offsets_mib) == 0:
            raise ConfigError("staging_offsets_mib must not be empty")
        if min(self.staging_offsets_mib) < 0:
            raise ConfigError("staging_offsets_mib must be non-negative")
        if self.replay_manifest is not None:
            replay_path = Path(self.replay_manifest)
            if replay_path.is_absolute() or ".." in replay_path.parts:
                raise ConfigError("replay_manifest must be a confined relative path")

    @classmethod
    def from_json(cls, value: object, index: int) -> "ScenarioConfig":
        """Parse one scenario.

        :param value: Parsed scenario object.
        :param index: Scenario index used in diagnostics.
        :returns: Parsed scenario configuration.
        """
        context = f"scenarios[{index}]"
        obj = _as_dict(value, context)
        required = {
            "name",
            "source_start_block",
            "source_end_block",
            "run_count",
            "iterations",
            "staging_offsets_mib",
        }
        _require_keys(
            obj,
            required=required,
            optional={"replay_manifest"},
            context=context,
        )
        offset_values = _as_list(
            obj["staging_offsets_mib"], f"{context}.staging_offsets_mib"
        )
        replay_manifest = obj.get("replay_manifest")
        if replay_manifest is not None:
            replay_manifest = _as_relative_path(
                replay_manifest, f"{context}.replay_manifest"
            )
        return cls(
            name=_as_str(obj["name"], f"{context}.name"),
            source_start_block=_as_int(
                obj["source_start_block"], f"{context}.source_start_block"
            ),
            source_end_block=_as_int(
                obj["source_end_block"], f"{context}.source_end_block"
            ),
            run_count=_as_int(obj["run_count"], f"{context}.run_count"),
            iterations=_as_int(obj["iterations"], f"{context}.iterations"),
            staging_offsets_mib=tuple(
                _as_int(item, f"{context}.staging_offsets_mib[{offset_index}]")
                for offset_index, item in enumerate(offset_values)
            ),
            replay_manifest=replay_manifest,
        )


@dataclass(frozen=True)
class TransportArmConfig:
    """Describe one fresh-process UCX transport arm.

    :ivar name: Stable arm identity.
    :ivar ucx_tls: Exact UCX transport selection.
    :ivar ucx_memtype_cache: Optional UCX memory-type cache setting.
    :ivar ucx_rndv_scheme: Optional UCX rendezvous scheme setting.
    """

    name: str
    ucx_tls: str
    ucx_memtype_cache: str | None
    ucx_rndv_scheme: str | None

    def __post_init__(self) -> None:
        if len(self.name) == 0:
            raise ConfigError("transport arm name must not be empty")
        if len(self.ucx_tls) == 0:
            raise ConfigError("transport arm ucx_tls must not be empty")

    @classmethod
    def from_json(cls, value: object, index: int) -> "TransportArmConfig":
        """Parse one transport arm.

        :param value: Parsed arm object.
        :param index: Arm index used in diagnostics.
        :returns: Parsed transport arm.
        """
        context = f"transport_arms[{index}]"
        obj = _as_dict(value, context)
        _require_keys(
            obj,
            required={"name", "ucx_tls"},
            optional={"ucx_memtype_cache", "ucx_rndv_scheme"},
            context=context,
        )
        memtype = obj.get("ucx_memtype_cache")
        rndv = obj.get("ucx_rndv_scheme")
        if memtype is not None:
            memtype = _as_str(memtype, f"{context}.ucx_memtype_cache")
        if rndv is not None:
            rndv = _as_str(rndv, f"{context}.ucx_rndv_scheme")
        return cls(
            name=_as_str(obj["name"], f"{context}.name"),
            ucx_tls=_as_str(obj["ucx_tls"], f"{context}.ucx_tls"),
            ucx_memtype_cache=memtype,
            ucx_rndv_scheme=rndv,
        )


@dataclass(frozen=True)
class VictimConfig:
    """Configure deterministic concurrent GEMM and redzone canaries.

    :ivar matrix_order: Order of each square FP32 GEMM matrix.
    :ivar rounds: Number of GEMMs queued per transfer.
    :ivar redzone_bytes: Bytes in every canary redzone.
    """

    matrix_order: int
    rounds: int
    redzone_bytes: int

    def __post_init__(self) -> None:
        if self.matrix_order <= 0:
            raise ConfigError("victim matrix_order must be positive")
        if self.rounds <= 0:
            raise ConfigError("victim rounds must be positive")
        if self.redzone_bytes <= 0 or self.redzone_bytes % 4 != 0:
            raise ConfigError("victim redzone_bytes must be positive and aligned")

    @classmethod
    def from_json(cls, value: object) -> "VictimConfig":
        """Parse the victim configuration.

        :param value: Parsed victim object.
        :returns: Parsed victim configuration.
        """
        obj = _as_dict(value, "victim")
        _require_keys(
            obj,
            required={"matrix_order", "rounds", "redzone_bytes"},
            optional=set(),
            context="victim",
        )
        return cls(
            matrix_order=_as_int(obj["matrix_order"], "victim.matrix_order"),
            rounds=_as_int(obj["rounds"], "victim.rounds"),
            redzone_bytes=_as_int(obj["redzone_bytes"], "victim.redzone_bytes"),
        )


@dataclass(frozen=True)
class RigConfig:
    """Describe one complete transport micro-rig campaign.

    :ivar schema_version: Configuration schema version.
    :ivar legacy_source_handshake_manifest: Captured P physical-geometry manifest.
    :ivar legacy_destination_handshake_manifest: Captured D physical-geometry
        manifest.
    :ivar legacy_source_handshake_sha256: Expected source-manifest identity.
    :ivar legacy_destination_handshake_sha256: Expected destination-manifest
        identity.
    :ivar semantic_handshake_manifest: Static connector-v9 semantic fixture.
    :ivar semantic_handshake_sha256: Expected semantic-fixture identity.
    :ivar semantic_handshake_profile: Selected semantic projection identity.
    :ivar producer_devices: Physical GPUs used to emulate P TP ranks.
    :ivar consumer_device: Physical GPU used for D staging, scatter, and victim.
    :ivar protected_devices: Physical GPUs the rig must never select.
    :ivar source_block_count: Blocks in every P registration region.
    :ivar valid_token_extent: Exact synthetic request-token extent represented
        by every configured transfer plan.
    :ivar staging_capacity_mib: D staging registration capacity.
    :ivar nixl_num_threads: NIXL UCX worker thread count.
    :ivar pattern_chunk_bytes: Maximum verification pattern chunk.
    :ivar transfer_timeout_seconds: Per-transfer terminal-state deadline.
    :ivar control_host: TCP control endpoint host.
    :ivar control_port: TCP control endpoint port.
    :ivar regions: Ordered physical registration regions.
    :ivar groups: Ordered KV-cache groups before prefix trimming.
    :ivar scenarios: Source geometry scenarios.
    :ivar transport_arms: Fresh-process UCX configurations.
    :ivar victim: Concurrent-compute canary configuration.
    """

    schema_version: int
    legacy_source_handshake_manifest: str
    legacy_destination_handshake_manifest: str
    legacy_source_handshake_sha256: str
    legacy_destination_handshake_sha256: str
    semantic_handshake_manifest: str
    semantic_handshake_sha256: str
    semantic_handshake_profile: str
    producer_devices: tuple[int, ...]
    consumer_device: int
    protected_devices: tuple[int, ...]
    source_block_count: int
    valid_token_extent: int
    staging_capacity_mib: int
    nixl_num_threads: int
    pattern_chunk_bytes: int
    transfer_timeout_seconds: int
    control_host: str
    control_port: int
    regions: tuple[RegionConfig, ...]
    groups: tuple[GroupConfig, ...]
    scenarios: tuple[ScenarioConfig, ...]
    transport_arms: tuple[TransportArmConfig, ...]
    victim: VictimConfig

    def __post_init__(self) -> None:
        immutable_allowed_devices = frozenset(range(6))
        immutable_denied_devices = frozenset({6, 7})
        if self.schema_version != 2:
            raise ConfigError(f"unsupported schema_version: {self.schema_version}")
        for context, path_text in (
            (
                "legacy_source_handshake_manifest",
                self.legacy_source_handshake_manifest,
            ),
            (
                "legacy_destination_handshake_manifest",
                self.legacy_destination_handshake_manifest,
            ),
            ("semantic_handshake_manifest", self.semantic_handshake_manifest),
        ):
            if len(path_text) == 0:
                raise ConfigError(f"{context} must not be empty")
            path = Path(path_text)
            if path.is_absolute() or ".." in path.parts:
                raise ConfigError(f"{context} must be a confined relative path")
        for context, digest in (
            (
                "legacy_source_handshake_sha256",
                self.legacy_source_handshake_sha256,
            ),
            (
                "legacy_destination_handshake_sha256",
                self.legacy_destination_handshake_sha256,
            ),
            ("semantic_handshake_sha256", self.semantic_handshake_sha256),
        ):
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ConfigError(f"{context} must contain lowercase SHA-256 hex")
        if len(self.semantic_handshake_profile) == 0:
            raise ConfigError("semantic_handshake_profile must not be empty")
        if len(self.producer_devices) == 0:
            raise ConfigError("producer_devices must not be empty")
        if len(set(self.producer_devices)) != len(self.producer_devices):
            raise ConfigError("producer_devices must be unique")
        if self.consumer_device in self.producer_devices:
            raise ConfigError("consumer_device must not be a producer device")
        selected = set(self.producer_devices) | {self.consumer_device}
        if not selected <= immutable_allowed_devices:
            raise ConfigError(
                "this host rig is hard-limited to physical GPUs 0 through 5"
            )
        if len(selected & immutable_denied_devices) > 0:
            raise ConfigError("physical GPUs 6 and 7 are immutable denylist entries")
        if not immutable_denied_devices <= set(self.protected_devices):
            raise ConfigError("protected_devices must include immutable GPUs 6 and 7")
        forbidden = selected & set(self.protected_devices)
        if len(forbidden) > 0:
            raise ConfigError(f"selected protected devices: {sorted(forbidden)}")
        if min(selected) < 0:
            raise ConfigError("device IDs must be non-negative")
        if self.source_block_count <= 0:
            raise ConfigError("source_block_count must be positive")
        if self.valid_token_extent <= 0:
            raise ConfigError("valid_token_extent must be positive")
        if self.staging_capacity_mib <= 0:
            raise ConfigError("staging_capacity_mib must be positive")
        if self.nixl_num_threads < 0:
            raise ConfigError("nixl_num_threads must be non-negative")
        if self.pattern_chunk_bytes <= 0 or self.pattern_chunk_bytes % 4 != 0:
            raise ConfigError("pattern_chunk_bytes must be positive and aligned")
        if self.transfer_timeout_seconds <= 0:
            raise ConfigError("transfer_timeout_seconds must be positive")
        if len(self.control_host) == 0:
            raise ConfigError("control_host must not be empty")
        if self.control_port <= 0 or self.control_port > 65535:
            raise ConfigError("control_port must be in [1, 65535]")
        if self.control_port + len(self.producer_devices) - 1 > 65535:
            raise ConfigError("producer control ports exceed 65535")
        if len(self.regions) == 0:
            raise ConfigError("regions must not be empty")
        if len({region.name for region in self.regions}) != len(self.regions):
            raise ConfigError("region names must be unique")
        if len(self.groups) == 0:
            raise ConfigError("groups must not be empty")
        if tuple(group.index for group in self.groups) != tuple(
            range(len(self.groups))
        ):
            raise ConfigError("group indices must be contiguous and ordered from zero")
        if len({group.name for group in self.groups}) != len(self.groups):
            raise ConfigError("group names must be unique")
        for group in self.groups:
            if max(group.owned_region_indices) >= len(self.regions):
                raise ConfigError(f"group {group.index} owns an out-of-range region")
        for region_index in range(len(self.regions)):
            plane_counts = {
                group.destination_plane_count
                for group in self.groups
                if region_index in group.owned_region_indices
            }
            if len(plane_counts) == 0:
                raise ConfigError(f"region {region_index} has no owning group")
            if len(plane_counts) != 1:
                raise ConfigError(
                    f"region {region_index} mixes destination plane layouts"
                )
        if len(self.scenarios) == 0:
            raise ConfigError("scenarios must not be empty")
        if len({scenario.name for scenario in self.scenarios}) != len(self.scenarios):
            raise ConfigError("scenario names must be unique")
        if len(self.transport_arms) == 0:
            raise ConfigError("transport_arms must not be empty")
        if len({arm.name for arm in self.transport_arms}) != len(self.transport_arms):
            raise ConfigError("transport arm names must be unique")

        position_count = sum(
            group.remote_position_count
            - (2 if group.destination_plane_count == 1 else 1)
            * (
                (
                    group.remote_position_count
                    + (2 if group.destination_plane_count == 1 else 1)
                    - 1
                )
                // (2 if group.destination_plane_count == 1 else 1)
                - group.local_position_count
            )
            for group in self.groups
        )
        for scenario in self.scenarios:
            if position_count == 0:
                if scenario.run_count != 0:
                    raise ConfigError("zero-byte plans require run_count=0")
                continue
            span = scenario.source_end_block - scenario.source_start_block + 1
            if span < position_count:
                raise ConfigError(
                    f"scenario {scenario.name!r} has a {span}-block span for "
                    f"{position_count} selected positions"
                )
            gap_count = span - position_count
            if scenario.run_count > position_count:
                raise ConfigError(
                    f"scenario {scenario.name!r} has more runs than positions"
                )
            if scenario.run_count == 1 and gap_count != 0:
                raise ConfigError(
                    f"scenario {scenario.name!r} requests one run with gaps"
                )
            if scenario.run_count > 1 and gap_count < scenario.run_count - 1:
                raise ConfigError(
                    f"scenario {scenario.name!r} cannot separate every run"
                )
            if scenario.source_end_block >= self.source_block_count:
                raise ConfigError(
                    f"scenario {scenario.name!r} reaches block "
                    f"{scenario.source_end_block}, "
                    f"outside source_block_count={self.source_block_count}"
                )

    @classmethod
    def from_json(cls, value: object) -> "RigConfig":
        """Parse and validate a complete configuration.

        :param value: Parsed root JSON object.
        :returns: Parsed micro-rig configuration.
        """
        obj = _as_dict(value, "root")
        required = {
            "schema_version",
            "legacy_source_handshake_manifest",
            "legacy_destination_handshake_manifest",
            "legacy_source_handshake_sha256",
            "legacy_destination_handshake_sha256",
            "semantic_handshake_manifest",
            "semantic_handshake_sha256",
            "semantic_handshake_profile",
            "producer_devices",
            "consumer_device",
            "protected_devices",
            "source_block_count",
            "valid_token_extent",
            "staging_capacity_mib",
            "nixl_num_threads",
            "pattern_chunk_bytes",
            "transfer_timeout_seconds",
            "control_host",
            "control_port",
            "regions",
            "groups",
            "scenarios",
            "transport_arms",
            "victim",
        }
        _require_keys(obj, required=required, optional=set(), context="root")

        producer_values = _as_list(obj["producer_devices"], "producer_devices")
        protected_values = _as_list(obj["protected_devices"], "protected_devices")
        region_values = _as_list(obj["regions"], "regions")
        group_values = _as_list(obj["groups"], "groups")
        scenario_values = _as_list(obj["scenarios"], "scenarios")
        arm_values = _as_list(obj["transport_arms"], "transport_arms")

        return cls(
            schema_version=_as_int(obj["schema_version"], "schema_version"),
            legacy_source_handshake_manifest=_as_relative_path(
                obj["legacy_source_handshake_manifest"],
                "legacy_source_handshake_manifest",
            ),
            legacy_destination_handshake_manifest=_as_relative_path(
                obj["legacy_destination_handshake_manifest"],
                "legacy_destination_handshake_manifest",
            ),
            legacy_source_handshake_sha256=_as_str(
                obj["legacy_source_handshake_sha256"],
                "legacy_source_handshake_sha256",
            ),
            legacy_destination_handshake_sha256=_as_str(
                obj["legacy_destination_handshake_sha256"],
                "legacy_destination_handshake_sha256",
            ),
            semantic_handshake_manifest=_as_relative_path(
                obj["semantic_handshake_manifest"],
                "semantic_handshake_manifest",
            ),
            semantic_handshake_sha256=_as_str(
                obj["semantic_handshake_sha256"],
                "semantic_handshake_sha256",
            ),
            semantic_handshake_profile=_as_str(
                obj["semantic_handshake_profile"],
                "semantic_handshake_profile",
            ),
            producer_devices=tuple(
                _as_int(item, f"producer_devices[{index}]")
                for index, item in enumerate(producer_values)
            ),
            consumer_device=_as_int(obj["consumer_device"], "consumer_device"),
            protected_devices=tuple(
                _as_int(item, f"protected_devices[{index}]")
                for index, item in enumerate(protected_values)
            ),
            source_block_count=_as_int(obj["source_block_count"], "source_block_count"),
            valid_token_extent=_as_int(obj["valid_token_extent"], "valid_token_extent"),
            staging_capacity_mib=_as_int(
                obj["staging_capacity_mib"], "staging_capacity_mib"
            ),
            nixl_num_threads=_as_int(obj["nixl_num_threads"], "nixl_num_threads"),
            pattern_chunk_bytes=_as_int(
                obj["pattern_chunk_bytes"], "pattern_chunk_bytes"
            ),
            transfer_timeout_seconds=_as_int(
                obj["transfer_timeout_seconds"], "transfer_timeout_seconds"
            ),
            control_host=_as_str(obj["control_host"], "control_host"),
            control_port=_as_int(obj["control_port"], "control_port"),
            regions=tuple(
                RegionConfig.from_json(item, index)
                for index, item in enumerate(region_values)
            ),
            groups=tuple(
                GroupConfig.from_json(item, index)
                for index, item in enumerate(group_values)
            ),
            scenarios=tuple(
                ScenarioConfig.from_json(item, index)
                for index, item in enumerate(scenario_values)
            ),
            transport_arms=tuple(
                TransportArmConfig.from_json(item, index)
                for index, item in enumerate(arm_values)
            ),
            victim=VictimConfig.from_json(obj["victim"]),
        )

    @property
    def fingerprint(self) -> str:
        """Return a stable SHA-256 identity for the full configuration.

        :returns: Lowercase hexadecimal SHA-256 digest.
        """
        encoded = json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    def transport_arm(self, name: str) -> TransportArmConfig:
        """Resolve one transport arm by name.

        :param name: Arm identity.
        :returns: Matching arm.
        :raises ConfigError: If the arm does not exist.
        """
        for arm in self.transport_arms:
            if arm.name == name:
                return arm
        raise ConfigError(f"unknown transport arm: {name}")

    def scenario(self, name: str) -> ScenarioConfig:
        """Resolve one scenario by name.

        :param name: Scenario identity.
        :returns: Matching scenario.
        :raises ConfigError: If the scenario does not exist.
        """
        for scenario in self.scenarios:
            if scenario.name == name:
                return scenario
        raise ConfigError(f"unknown scenario: {name}")

    def region_destination_plane_count(self, region_index: int) -> int:
        """Return the unique destination plane layout for one region.

        :param region_index: Canonical physical region index.
        :returns: One for packed single-plane or two for standard K/V.
        """
        plane_counts = {
            group.destination_plane_count
            for group in self.groups
            if region_index in group.owned_region_indices
        }
        if len(plane_counts) != 1:
            raise ConfigError(
                f"region {region_index} lacks one destination plane layout"
            )
        return next(iter(plane_counts))


def _manifest_ranks(path: Path, expected_sha256: str) -> list[dict[str, object]]:
    """Load and authenticate one captured handshake manifest.

    :param path: Captured manifest path.
    :param expected_sha256: Required lowercase file digest.
    :returns: Rank records from the manifest.
    :raises ConfigError: If the file identity or shape is invalid.
    """
    payload = path.read_bytes()
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ConfigError(
            f"handshake manifest {path} has SHA-256 {actual_sha256}, "
            f"expected {expected_sha256}"
        )
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ConfigError(f"invalid JSON in {path}: {error}") from error
    root = _as_dict(value, str(path))
    ranks = _as_list(root.get("ranks"), f"{path}.ranks")
    return [
        _as_dict(rank, f"{path}.ranks[{index}]") for index, rank in enumerate(ranks)
    ]


def _manifest_int_list(value: object, context: str) -> tuple[int, ...]:
    """Parse an integer array from captured handshake data.

    :param value: Parsed JSON value.
    :param context: Human-readable field name.
    :returns: Parsed integers.
    """
    return tuple(
        _as_int(item, f"{context}[{index}]")
        for index, item in enumerate(_as_list(value, context))
    )


def _validate_legacy_handshake_manifests(
    config: RigConfig,
    directory: Path,
) -> None:
    """Require physical geometry to match the legacy storm37b captures.

    :param config: Parsed rig configuration.
    :param directory: Directory used to resolve manifest paths.
    :raises ConfigError: If any transcribed geometry differs from capture.
    """
    source_path = directory / config.legacy_source_handshake_manifest
    destination_path = directory / config.legacy_destination_handshake_manifest
    source_ranks = _manifest_ranks(
        source_path,
        config.legacy_source_handshake_sha256,
    )
    destination_ranks = _manifest_ranks(
        destination_path,
        config.legacy_destination_handshake_sha256,
    )
    if len(source_ranks) != len(config.producer_devices):
        raise ConfigError(
            f"source handshake has {len(source_ranks)} ranks, expected "
            f"{len(config.producer_devices)}"
        )
    source_block_lens = tuple(region.row_bytes for region in config.regions)
    source_engine_ids: set[str] = set()
    compatibility_hashes: set[str] = set()
    for rank_index, rank in enumerate(source_ranks):
        captured_rank = _as_int(rank.get("rank"), f"source rank[{rank_index}].rank")
        captured_blocks = _as_int(
            rank.get("num_blocks"), f"source rank[{rank_index}].num_blocks"
        )
        captured_lens = _manifest_int_list(
            rank.get("block_lens"), f"source rank[{rank_index}].block_lens"
        )
        captured_device = _as_int(
            rank.get("device_id"), f"source rank[{rank_index}].device_id"
        )
        captured_block_size = _as_int(
            rank.get("block_size"), f"source rank[{rank_index}].block_size"
        )
        captured_ratio = _as_int(
            rank.get("physical_blocks_per_logical_kv_block"),
            f"source rank[{rank_index}].physical_blocks_per_logical_kv_block",
        )
        engine_id = _as_str(
            rank.get("engine_id"), f"source rank[{rank_index}].engine_id"
        )
        compatibility_hash = _as_str(
            rank.get("compatibility_hash"),
            f"source rank[{rank_index}].compatibility_hash",
        )
        source_engine_ids.add(engine_id)
        compatibility_hashes.add(compatibility_hash)
        if captured_rank != rank_index:
            raise ConfigError("source handshake ranks must be ordered and contiguous")
        if captured_blocks != config.source_block_count:
            raise ConfigError(
                f"source rank {rank_index} captured {captured_blocks} blocks, "
                f"configured {config.source_block_count}"
            )
        if captured_lens != source_block_lens:
            raise ConfigError(
                f"source rank {rank_index} block_lens {captured_lens} differ "
                f"from configured regions {source_block_lens}"
            )
        if captured_device != rank_index:
            raise ConfigError("source handshake device IDs must equal TP ranks")
        if captured_block_size != 16 or captured_ratio != 1:
            raise ConfigError("source handshake block mapping differs from ship path")
        if rank.get("kv_cache_layout") != "HND":
            raise ConfigError("source handshake layout must be HND")
        if rank.get("attn_backend_name") != "FLASHINFER_GEMMA4_TRTLLM_GEN":
            raise ConfigError("source handshake attention backend differs")
    if len(source_engine_ids) != 1 or len(compatibility_hashes) != 1:
        raise ConfigError("source ranks disagree on engine or compatibility identity")

    if len(destination_ranks) != 1:
        raise ConfigError("destination handshake must contain exactly one TP1 rank")
    destination = destination_ranks[0]
    destination_blocks = _as_int(
        destination.get("num_blocks"), "destination rank.num_blocks"
    )
    destination_lens = _manifest_int_list(
        destination.get("block_lens"), "destination rank.block_lens"
    )
    destination_block_size = _as_int(
        destination.get("block_size"), "destination rank.block_size"
    )
    destination_ratio = _as_int(
        destination.get("physical_blocks_per_logical_kv_block"),
        "destination rank.physical_blocks_per_logical_kv_block",
    )
    expected_destination_lens = tuple(
        row_bytes * len(config.producer_devices) for row_bytes in source_block_lens
    )
    if destination_blocks != config.source_block_count:
        raise ConfigError(
            f"destination captured {destination_blocks} blocks, configured "
            f"{config.source_block_count}"
        )
    if destination_lens != expected_destination_lens:
        raise ConfigError(
            f"destination block_lens {destination_lens} differ from expected "
            f"TP4 scatter rows {expected_destination_lens}"
        )
    if _as_int(destination.get("rank"), "destination rank.rank") != 0:
        raise ConfigError("destination handshake must be TP rank zero")
    if _as_int(destination.get("device_id"), "destination rank.device_id") != 0:
        raise ConfigError(
            "destination capture must use isolated logical CUDA device zero"
        )
    if destination_block_size != 16 or destination_ratio != 1:
        raise ConfigError("destination handshake block mapping differs from ship path")
    if destination.get("kv_cache_layout") != "HND":
        raise ConfigError("destination handshake layout must be HND")
    if destination.get("attn_backend_name") != "FLASHINFER_GEMMA4_TRTLLM_GEN":
        raise ConfigError("destination handshake attention backend differs")
    if destination.get("compatibility_hash") not in compatibility_hashes:
        raise ConfigError("P and D handshake compatibility hashes differ")


def load_configured_semantic_handshake_profile(
    config: RigConfig,
    directory: Path,
) -> SemanticHandshakeProfile:
    """Load and bind the configured static semantic fixture.

    :param config: Parsed rig configuration.
    :param directory: Directory used to resolve the fixture path.
    :returns: Authenticated semantic profile.
    :raises ConfigError: If fixture identity or rig semantics differ.
    """
    path = directory / config.semantic_handshake_manifest
    try:
        profile = load_semantic_handshake_profile(
            path,
            config.semantic_handshake_sha256,
            config.semantic_handshake_profile,
        )
        region_group_indices = tuple(
            tuple(
                group.index
                for group in config.groups
                if region_index in group.owned_region_indices
            )
            for region_index in range(len(config.regions))
        )
        profile.validate_rig_contract(
            connector_version=SEMANTIC_HANDSHAKE_CONNECTOR_VERSION,
            group_semantic_names=tuple(group.name for group in config.groups),
            source_group_planes=tuple(
                group.destination_plane_count for group in config.groups
            ),
            physical_group_token_capacities=tuple(
                group.token_capacity for group in config.groups
            ),
            region_semantic_names=tuple(region.name for region in config.regions),
            region_group_indices=region_group_indices,
        )
        source_row_bytes = tuple(region.row_bytes for region in config.regions)
        profile.region_descriptors(
            num_blocks=config.source_block_count,
            row_bytes=source_row_bytes,
            base_address=0x10000000000,
        )
        profile.region_descriptors(
            num_blocks=config.source_block_count,
            row_bytes=tuple(
                row_bytes * len(config.producer_devices)
                for row_bytes in source_row_bytes
            ),
            base_address=0x20000000000,
        )
    except SemanticHandshakeError as error:
        raise ConfigError(str(error)) from error
    return profile


def load_config(path: Path) -> RigConfig:
    """Load a rig configuration from disk.

    :param path: JSON configuration path.
    :returns: Parsed and validated configuration.
    :raises ConfigError: If JSON parsing or validation fails.
    """
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ConfigError(f"invalid JSON in {path}: {error}") from error
    config = RigConfig.from_json(value)
    _validate_legacy_handshake_manifests(config, path.parent)
    load_configured_semantic_handshake_profile(config, path.parent)
    return config
