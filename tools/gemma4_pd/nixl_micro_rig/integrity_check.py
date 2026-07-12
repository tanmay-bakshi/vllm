"""Runtime contract check for the shared P-to-D integrity primitive."""

from vllm.distributed.kv_transfer.integrity import (
    IntegrityIdentity,
    IntegrityPayloadKind,
    compute_integrity_digest,
)


def integrity_contract_self_test() -> dict[str, str]:
    """Prove the loaded digest rejects historical linear-sum blind spots.

    :returns: Hex digests for the baseline and three negative controls.
    :raises RuntimeError: If any corruption control collides with the baseline.
    """
    identity = IntegrityIdentity(
        run_id="micro-rig-self-test",
        transport_arm="host",
        producer_engine_id="producer",
        producer_request_id="request",
        registration_generation="self-test-registration-1",
        semantic_contract_digest=bytes(range(32)),
        offer_generation=1,
        iteration=0,
        source_rank=0,
        region_index=0,
        group_index=0,
        plane_index=-1,
        source_position=0,
        remote_block_id=32768,
        valid_token_extent=64,
        group_token_capacity=64,
        payload_kind=IntegrityPayloadKind.WIRE,
        byte_length=64,
    )
    baseline = bytes(range(64))
    bit_flip = bytearray(baseline)
    bit_flip[17] ^= 0x01
    permutation = baseline[32:] + baseline[:32]
    compensating = bytearray(baseline)
    compensating[11] += 1
    compensating[29] -= 1
    if sum(compensating) != sum(baseline):
        raise AssertionError("compensating mutation did not preserve the old sum")

    payloads = {
        "baseline": baseline,
        "bit_flip": bit_flip,
        "permutation": permutation,
        "compensating": compensating,
    }
    digests = {
        name: compute_integrity_digest(identity, payload).hex()
        for name, payload in payloads.items()
    }
    if len(set(digests.values())) != len(digests):
        raise RuntimeError(f"integrity negative control collision: {digests}")
    return digests
