"""Pure semantic identity for micro-rig integrity leaves."""

import hashlib
import json
from functools import cache

from tools.gemma4_pd.nixl_micro_rig.config import RigConfig
from vllm.distributed.kv_transfer.integrity import IntegrityIdentity


@cache
def compute_rig_semantic_contract_digest(
    config: RigConfig,
    group_index: int,
    region_index: int,
) -> bytes:
    """Hash one exact synthetic group-to-region interpretation.

    The model-free rig has no scheduler or model tensors, so its semantic
    authority is the checked-in, typed geometry. The digest distinguishes
    semantically owned rows from transport-only broadcast rows.

    :param config: Complete typed rig configuration.
    :param group_index: KV-cache group owning the transfer position.
    :param region_index: Physical registration region carrying the row.
    :returns: Thirty-two-byte BLAKE2b semantic contract digest.
    """
    group = config.groups[group_index]
    region = config.regions[region_index]
    source_shape = (config.source_block_count, 2, region.row_bytes // 2)
    destination_shape = (
        config.source_block_count,
        2,
        len(config.producer_devices),
        region.row_bytes // 2,
    )
    contract = {
        "schema_version": IntegrityIdentity.SCHEMA_VERSION,
        "contract": "gemma4-nixl-micro-rig-synthetic-v1",
        "group_index": group.index,
        "group_name": group.name,
        "group_token_capacity": group.token_capacity,
        "owned_region_indices": group.owned_region_indices,
        "region_index": region_index,
        "region_name": region.name,
        "row_bytes": region.row_bytes,
        "semantically_owned": region_index in group.owned_region_indices,
        "source_dtype": "uint8",
        "source_layout": "block,plane,row-byte",
        "source_shape": source_shape,
        "source_strides": (region.row_bytes, region.row_bytes // 2, 1),
        "source_plane_contract": 2,
        "destination_dtype": "uint8",
        "destination_layout": "block,plane,rank-slot,row-byte",
        "destination_shape": destination_shape,
        "destination_strides": (
            region.row_bytes * len(config.producer_devices),
            region.row_bytes * len(config.producer_devices) // 2,
            region.row_bytes // 2,
            1,
        ),
    }
    payload = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.blake2b(
        payload,
        digest_size=32,
        person=b"p2d-rig-sem-v1",
    ).digest()
