"""Typed connector-v9 semantic fixtures for the NIXL micro-rig."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from vllm.distributed.kv_transfer.nixl_contracts import NixlRegionDescriptor

SEMANTIC_HANDSHAKE_SCHEMA_VERSION = 2
SEMANTIC_HANDSHAKE_CONNECTOR_VERSION = 9
STATIC_SEMANTIC_EVIDENCE_SCOPE = "static_model_free_connector_v9_contract_fixture"


class SemanticHandshakeError(ValueError):
    """Report an invalid or falsely labelled semantic fixture."""


def _object(value: object, context: str) -> dict[str, object]:
    """Require a JSON object with string keys.

    :param value: Parsed JSON value.
    :param context: Human-readable value identity.
    :returns: Validated object.
    :raises SemanticHandshakeError: If the value is not an object.
    """
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise SemanticHandshakeError(f"{context} must be an object")
    return value


def _exact_keys(
    value: dict[str, object],
    expected: set[str],
    context: str,
) -> None:
    """Require one exact fixture vocabulary.

    :param value: Object whose keys are validated.
    :param expected: Complete expected key set.
    :param context: Human-readable value identity.
    :raises SemanticHandshakeError: If keys are missing or unknown.
    """
    if set(value) != expected:
        raise SemanticHandshakeError(
            f"{context} keys differ: observed={sorted(value)}, "
            f"expected={sorted(expected)}"
        )


def _string(value: object, context: str) -> str:
    """Require a non-empty string.

    :param value: Parsed JSON value.
    :param context: Human-readable value identity.
    :returns: Validated string.
    :raises SemanticHandshakeError: If the value is not a non-empty string.
    """
    if not isinstance(value, str) or len(value) == 0:
        raise SemanticHandshakeError(f"{context} must be a non-empty string")
    return value


def _integer(value: object, context: str) -> int:
    """Require a non-boolean integer.

    :param value: Parsed JSON value.
    :param context: Human-readable value identity.
    :returns: Validated integer.
    :raises SemanticHandshakeError: If the value is not an integer.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise SemanticHandshakeError(f"{context} must be an integer")
    return value


def _array(value: object, context: str) -> list[object]:
    """Require a JSON array.

    :param value: Parsed JSON value.
    :param context: Human-readable value identity.
    :returns: Validated array.
    :raises SemanticHandshakeError: If the value is not an array.
    """
    if not isinstance(value, list):
        raise SemanticHandshakeError(f"{context} must be an array")
    return value


def _string_tuple(value: object, context: str) -> tuple[str, ...]:
    """Parse an ordered string array.

    :param value: Parsed JSON value.
    :param context: Human-readable value identity.
    :returns: Validated tuple.
    """
    return tuple(
        _string(item, f"{context}[{index}]")
        for index, item in enumerate(_array(value, context))
    )


def _integer_tuple(value: object, context: str) -> tuple[int, ...]:
    """Parse an ordered integer array.

    :param value: Parsed JSON value.
    :param context: Human-readable value identity.
    :returns: Validated tuple.
    """
    return tuple(
        _integer(item, f"{context}[{index}]")
        for index, item in enumerate(_array(value, context))
    )


def _nested_integer_tuple(
    value: object,
    context: str,
) -> tuple[tuple[int, ...], ...]:
    """Parse an ordered array of integer arrays.

    :param value: Parsed JSON value.
    :param context: Human-readable value identity.
    :returns: Validated nested tuple.
    """
    return tuple(
        _integer_tuple(item, f"{context}[{index}]")
        for index, item in enumerate(_array(value, context))
    )


@dataclass(frozen=True, slots=True)
class SemanticHandshakeProfile:
    """Describe one static, model-free connector-v9 semantic projection.

    :ivar name: Stable profile identity selected by rig configuration.
    :ivar connector_version: Production wire-contract version under test.
    :ivar evidence_scope: Explicit non-runtime evidence classification.
    :ivar runtime_capture_authenticated: Whether a live v9 payload is preserved.
    :ivar rank_identity_field: Wire field binding metadata to its producer rank.
    :ivar group_semantic_names: Ordered synthetic group identities.
    :ivar source_group_planes: Ordered source plane semantics.
    :ivar physical_group_token_capacities: Ordered physical token capacities.
    :ivar region_semantic_names: Ordered synthetic region identities.
    :ivar region_group_indices: Ordered region ownership sets.
    :ivar descriptor_dtype: Synthetic packed descriptor dtype identity.
    :ivar descriptor_element_size_bytes: Bytes per synthetic descriptor element.
    :ivar descriptor_layout: Synthetic descriptor layout identity.
    """

    name: str
    connector_version: int
    evidence_scope: str
    runtime_capture_authenticated: bool
    rank_identity_field: str
    group_semantic_names: tuple[str, ...]
    source_group_planes: tuple[int, ...]
    physical_group_token_capacities: tuple[int, ...]
    region_semantic_names: tuple[str, ...]
    region_group_indices: tuple[tuple[int, ...], ...]
    descriptor_dtype: str
    descriptor_element_size_bytes: int
    descriptor_layout: str

    def __post_init__(self) -> None:
        if len(self.name) == 0:
            raise SemanticHandshakeError("semantic profile name must not be empty")
        if self.evidence_scope != STATIC_SEMANTIC_EVIDENCE_SCOPE:
            raise SemanticHandshakeError(
                "semantic fixture must declare the model-free fixture evidence scope"
            )
        if self.runtime_capture_authenticated:
            raise SemanticHandshakeError(
                "static semantic fixtures cannot claim runtime authentication"
            )
        if self.rank_identity_field != "tp_rank":
            raise SemanticHandshakeError(
                "connector-v9 metadata rank identity must be bound by tp_rank"
            )
        group_count = len(self.group_semantic_names)
        if group_count == 0:
            raise SemanticHandshakeError("semantic fixture has no cache groups")
        if len(set(self.group_semantic_names)) != group_count:
            raise SemanticHandshakeError("semantic group names must be unique")
        if len(self.source_group_planes) != group_count:
            raise SemanticHandshakeError(
                "source_group_planes cardinality differs from semantic groups"
            )
        if any(plane not in (1, 2) for plane in self.source_group_planes):
            raise SemanticHandshakeError(
                "source_group_planes must contain only one or two"
            )
        if len(self.physical_group_token_capacities) != group_count:
            raise SemanticHandshakeError(
                "physical_group_token_capacities cardinality differs from "
                "semantic groups"
            )
        if any(capacity <= 0 for capacity in self.physical_group_token_capacities):
            raise SemanticHandshakeError(
                "physical_group_token_capacities must be positive"
            )
        region_count = len(self.region_semantic_names)
        if region_count == 0:
            raise SemanticHandshakeError("semantic fixture has no regions")
        if len(set(self.region_semantic_names)) != region_count:
            raise SemanticHandshakeError("semantic region names must be unique")
        if len(self.region_group_indices) != region_count:
            raise SemanticHandshakeError(
                "region ownership cardinality differs from semantic regions"
            )
        covered_groups: set[int] = set()
        for region_index, owners in enumerate(self.region_group_indices):
            if len(owners) == 0:
                raise SemanticHandshakeError(
                    f"semantic region {region_index} has no owner"
                )
            if owners != tuple(sorted(set(owners))):
                raise SemanticHandshakeError(
                    f"semantic region {region_index} owners are not sorted and unique"
                )
            if any(owner < 0 or owner >= group_count for owner in owners):
                raise SemanticHandshakeError(
                    f"semantic region {region_index} owner is out of range"
                )
            covered_groups.update(owners)
        if covered_groups != set(range(group_count)):
            raise SemanticHandshakeError(
                "semantic regions do not cover every cache group"
            )
        if len(self.descriptor_dtype) == 0:
            raise SemanticHandshakeError("descriptor_dtype must not be empty")
        if self.descriptor_element_size_bytes <= 0:
            raise SemanticHandshakeError(
                "descriptor_element_size_bytes must be positive"
            )
        if self.descriptor_layout != "packed":
            raise SemanticHandshakeError(
                "model-free semantic fixtures must use packed descriptors"
            )

    @classmethod
    def from_json(cls, value: object, index: int) -> "SemanticHandshakeProfile":
        """Parse one strict fixture profile.

        :param value: Parsed profile object.
        :param index: Profile array index used in diagnostics.
        :returns: Typed semantic profile.
        """
        context = f"profiles[{index}]"
        obj = _object(value, context)
        expected = {
            "name",
            "connector_version",
            "evidence_scope",
            "runtime_capture_authenticated",
            "rank_identity_field",
            "group_semantic_names",
            "source_group_planes",
            "physical_group_token_capacities",
            "region_semantic_names",
            "region_group_indices",
            "descriptor_dtype",
            "descriptor_element_size_bytes",
            "descriptor_layout",
        }
        _exact_keys(obj, expected, context)
        runtime_capture_authenticated = obj["runtime_capture_authenticated"]
        if not isinstance(runtime_capture_authenticated, bool):
            raise SemanticHandshakeError(
                f"{context}.runtime_capture_authenticated must be a boolean"
            )
        return cls(
            name=_string(obj["name"], f"{context}.name"),
            connector_version=_integer(
                obj["connector_version"], f"{context}.connector_version"
            ),
            evidence_scope=_string(obj["evidence_scope"], f"{context}.evidence_scope"),
            runtime_capture_authenticated=runtime_capture_authenticated,
            rank_identity_field=_string(
                obj["rank_identity_field"], f"{context}.rank_identity_field"
            ),
            group_semantic_names=_string_tuple(
                obj["group_semantic_names"], f"{context}.group_semantic_names"
            ),
            source_group_planes=_integer_tuple(
                obj["source_group_planes"], f"{context}.source_group_planes"
            ),
            physical_group_token_capacities=_integer_tuple(
                obj["physical_group_token_capacities"],
                f"{context}.physical_group_token_capacities",
            ),
            region_semantic_names=_string_tuple(
                obj["region_semantic_names"], f"{context}.region_semantic_names"
            ),
            region_group_indices=_nested_integer_tuple(
                obj["region_group_indices"], f"{context}.region_group_indices"
            ),
            descriptor_dtype=_string(
                obj["descriptor_dtype"], f"{context}.descriptor_dtype"
            ),
            descriptor_element_size_bytes=_integer(
                obj["descriptor_element_size_bytes"],
                f"{context}.descriptor_element_size_bytes",
            ),
            descriptor_layout=_string(
                obj["descriptor_layout"], f"{context}.descriptor_layout"
            ),
        )

    def validate_rig_contract(
        self,
        *,
        connector_version: int,
        group_semantic_names: tuple[str, ...],
        source_group_planes: tuple[int, ...],
        physical_group_token_capacities: tuple[int, ...],
        region_semantic_names: tuple[str, ...],
        region_group_indices: tuple[tuple[int, ...], ...],
    ) -> None:
        """Bind this independent fixture to one typed rig configuration.

        :param connector_version: Current production connector version.
        :param group_semantic_names: Ordered configured group names.
        :param source_group_planes: Ordered configured source planes.
        :param physical_group_token_capacities: Ordered configured capacities.
        :param region_semantic_names: Ordered configured region names.
        :param region_group_indices: Ordered configured region owners.
        :raises SemanticHandshakeError: If the fixture differs from the rig.
        """
        observed = (
            self.connector_version,
            self.group_semantic_names,
            self.source_group_planes,
            self.physical_group_token_capacities,
            self.region_semantic_names,
            self.region_group_indices,
        )
        expected = (
            connector_version,
            group_semantic_names,
            source_group_planes,
            physical_group_token_capacities,
            region_semantic_names,
            region_group_indices,
        )
        if observed != expected:
            raise SemanticHandshakeError(
                "connector-v9 semantic fixture differs from the typed rig contract: "
                f"observed={observed}, expected={expected}"
            )

    def region_descriptors(
        self,
        *,
        num_blocks: int,
        row_bytes: tuple[int, ...],
        base_address: int,
    ) -> tuple[NixlRegionDescriptor, ...]:
        """Materialize complete packed v9 descriptors for validator replay.

        :param num_blocks: Physical rows in each registration.
        :param row_bytes: Ordered bytes per physical row.
        :param base_address: First positive synthetic registration address.
        :returns: Complete non-overlapping semantic region descriptors.
        :raises SemanticHandshakeError: If physical geometry is invalid.
        """
        if num_blocks <= 0 or base_address <= 0:
            raise SemanticHandshakeError(
                "descriptor num_blocks and base_address must be positive"
            )
        if len(row_bytes) != len(self.region_semantic_names):
            raise SemanticHandshakeError(
                "descriptor row cardinality differs from semantic regions"
            )
        descriptors: list[NixlRegionDescriptor] = []
        next_base = base_address
        for region_index, bytes_per_row in enumerate(row_bytes):
            if (
                bytes_per_row <= 0
                or bytes_per_row % self.descriptor_element_size_bytes != 0
            ):
                raise SemanticHandshakeError(
                    f"region {region_index} row bytes cannot represent the dtype"
                )
            row_elements = bytes_per_row // self.descriptor_element_size_bytes
            registered_bytes = num_blocks * bytes_per_row
            owners = self.region_group_indices[region_index]
            descriptors.append(
                NixlRegionDescriptor(
                    semantic_name=self.region_semantic_names[region_index],
                    group_indices=owners,
                    group_semantic_names=tuple(
                        (owner, self.group_semantic_names[owner]) for owner in owners
                    ),
                    base_address=next_base,
                    registered_bytes=registered_bytes,
                    row_bytes=bytes_per_row,
                    shape=(num_blocks, row_elements),
                    strides=(row_elements, 1),
                    dtype=self.descriptor_dtype,
                    element_size_bytes=self.descriptor_element_size_bytes,
                    layout=self.descriptor_layout,
                )
            )
            next_base += registered_bytes + 4096
        return tuple(descriptors)


def load_semantic_handshake_profile(
    path: Path,
    expected_sha256: str,
    profile_name: str,
) -> SemanticHandshakeProfile:
    """Authenticate and load one semantic profile by exact identity.

    :param path: Checked-in semantic fixture manifest.
    :param expected_sha256: Required lowercase file digest.
    :param profile_name: Exact profile identity.
    :returns: Selected typed semantic profile.
    :raises SemanticHandshakeError: If authentication or parsing fails.
    """
    payload = path.read_bytes()
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != expected_sha256:
        raise SemanticHandshakeError(
            f"semantic handshake manifest {path} has SHA-256 {actual_sha256}, "
            f"expected {expected_sha256}"
        )
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as error:
        raise SemanticHandshakeError(f"invalid JSON in {path}: {error}") from error
    root = _object(value, str(path))
    _exact_keys(root, {"schema_version", "profiles"}, str(path))
    schema_version = _integer(root["schema_version"], f"{path}.schema_version")
    if schema_version != SEMANTIC_HANDSHAKE_SCHEMA_VERSION:
        raise SemanticHandshakeError(
            f"unsupported semantic handshake schema_version {schema_version}"
        )
    profiles = tuple(
        SemanticHandshakeProfile.from_json(profile, index)
        for index, profile in enumerate(_array(root["profiles"], f"{path}.profiles"))
    )
    if len({profile.name for profile in profiles}) != len(profiles):
        raise SemanticHandshakeError("semantic profile names must be unique")
    for profile in profiles:
        if profile.name == profile_name:
            return profile
    raise SemanticHandshakeError(f"unknown semantic handshake profile {profile_name}")
