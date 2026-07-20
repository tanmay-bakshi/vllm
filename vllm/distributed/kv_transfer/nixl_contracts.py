# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wire-level contracts shared by the NIXL connector and diagnostics."""

import msgspec


class NixlRegionDescriptor(msgspec.Struct, array_like=True, frozen=True):
    """Semantic and physical identity of one registered KV region.

    :ivar semantic_name: Stable ordered layer identity.
    :ivar group_indices: Cache groups backed by the registration.
    :ivar group_semantic_names: Stable semantic identity within each group.
    :ivar base_address: Registered memory base address.
    :ivar registered_bytes: Exact registration length.
    :ivar row_bytes: Bytes in one physical cache row.
    :ivar shape: Tensor view shape.
    :ivar strides: Tensor view strides in elements.
    :ivar dtype: Canonical tensor dtype text.
    :ivar element_size_bytes: Bytes in one tensor element.
    :ivar layout: Physical KV-cache layout.
    """

    semantic_name: str
    group_indices: tuple[int, ...]
    group_semantic_names: tuple[tuple[int, str], ...]
    base_address: int
    registered_bytes: int
    row_bytes: int
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    dtype: str
    element_size_bytes: int
    layout: str


class NixlSourceRoster(msgspec.Struct, array_like=True, frozen=True):
    """Exact producer block roster retained through remote consumption.

    :ivar offer_generation: Monotonic producer allocation generation.
    :ivar iteration: Producer content generation within the allocation.
    :ivar expected_consumers: Logical decoder consumers sharing the offer.
    :ivar valid_token_extent: Number of settled source tokens.
    :ivar group_token_capacities: Physical tokens represented by each group row.
    :ivar block_ids: Exact physical producer rows in group order.
    """

    offer_generation: int
    iteration: int
    expected_consumers: int
    valid_token_extent: int
    group_token_capacities: tuple[int, ...]
    block_ids: tuple[tuple[int, ...], ...]
