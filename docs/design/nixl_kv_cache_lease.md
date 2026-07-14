# NIXL KV Source Ownership and Heartbeats

In disaggregated prefill/decode deployments, a producer must keep every offered
KV page immutable until all consumers that can address it have finished their
transfer. A timer cannot establish that condition. Network silence is
indistinguishable from a queued consumer, a delayed model step, or an in-flight
RDMA operation.

The NIXL connector therefore uses completion notifications as the authority for
page release. Heartbeats and deadlines are liveness signals that distinguish a
healthy queued consumer from a stalled transfer; they never authorize page
reuse.

## Safety invariant

A producer request remains in `_reqs_to_process` while any consumer may still
read its block roster. The scheduler releases the request's pages only after the
worker reports it in `finished_sending`, which requires the entire exact
completion-proof set for that producer rank.

The producer owns the completion contract. It records both the number of
logical decoder children (the original request's `n`) and the decoder
tensor-parallel size before offering any blocks. Each producer rank derives its
immutable proof set as the Cartesian product of:

- every logical child index; and
- every decoder rank that can address that producer rank under the configured
  producer/decoder tensor-parallel mapping.

When decoder TP is at least producer TP, each producer rank owns its contiguous
group of decoder ranks. When producer TP is larger, each producer rank owns the
single decoder rank to which it maps. The TP sizes must divide evenly in either
direction.

The producer lifecycle is:

1. **Owned with a deadline.** The request is present in `_reqs_to_process` and
   `_reqs_to_send`. Heartbeats may extend its deadline.
2. **Overdue but owned.** The deadline elapsed, so `_reqs_to_send` is removed
   and the overdue metric is recorded once. `_reqs_to_process`, the partial
   proof set, source rosters, and KV pages remain intact.
3. **Completed.** Every exact child/rank obligation has a valid proof. The
   worker removes `_reqs_to_process`, reports `finished_sending`, and the
   scheduler may reuse the pages.

This state machine fails closed. A dead or unreachable consumer can retain
capacity, but it cannot turn a stale block address into apparently successful
inference.

## Decoder ownership tracking

Heartbeat ownership begins in the decoder scheduler as soon as a remote-prefill
request arrives, including while the request is waiting. Ownership is grouped by
producer engine and reference-counted by producer request ID. Reference counting
is required because one HTTP request with `n > 1` creates multiple decoder
requests that share one producer request ID.

The scheduler sends complete ownership snapshots to the worker only when that
state changes:

- `None` means the worker's retained snapshot is unchanged;
- a nonempty mapping replaces the retained targets; and
- an empty mapping clears all targets.

Snapshots are copied before they enter asynchronous worker metadata, so later
scheduler mutations cannot alter queued state.

## Worker heartbeat delivery

The worker retains the latest ownership snapshot and owns the transmission
cadence. It services heartbeats at the beginning of `start_load_kv()` and at the
beginning of every `_get_finished()` poll. Servicing `_get_finished()` is
important for phase-separated transfers because the transfer drain calls it
repeatedly while a large staged read is in progress.

The default heartbeat interval is `kv_lease_duration // 6`, with a minimum of
one second. Each accepted heartbeat extends a matching producer deadline to at
least `now + kv_lease_duration * 2 // 3`.

Pull-mode heartbeats run on the engine worker thread alongside other pull NIXL
operations. Push mode dispatches heartbeat batches through the existing
`nixl-push-writer` thread, which owns push-side notification operations.

A long model operation or delayed first snapshot can still miss a deadline.
Correctness does not depend on heartbeat punctuality: the producer transitions
to overdue retention and waits for completion proof.

## Completion and release

The producer returns its logical-child count and decoder TP size with the block
offer. A decoder must match that contract to its local TP world and derive its
stable child index from the decoder request lineage before it can read.

Each completion notification is a typed proof containing the producer and
decoder request IDs, child index, decoder rank, decoder TP size, and logical
child count. For a native NIXL read, the proof is attached to the transfer and
is delivered only after that read succeeds. A decoder that needs no bytes
because of a complete prefix-cache hit sends the same proof directly. A
producer rank omitted from a fixed replicated MLA or GQA transfer plan receives
a no-read proof because that decoder cannot address its pages.

Each producer rank validates the proof against its immutable contract and the
decoder request lineage. It stores the resulting `(child_index, decoder_rank)`
identity in a set. Duplicate proofs are idempotent no-ops. Malformed proofs,
contract mismatches, impossible identities, and identities assigned to another
producer rank are ignored, so the source pages remain pinned. The request is
released only when the observed set exactly equals the producer rank's derived
obligation set.

Consumer-local `_released_rids` fencing remains in place after all logical
decoder children complete locally. It prevents an extra or delayed local pull
from committing against a producer request already known to be complete, but it
is not the producer's release authority.

There is no timeout-based invalidation message. Deadlines are diagnostic only
and never authorize page reuse. A safe finite reclamation protocol would
require a request-scoped revoke followed by an acknowledgement that queued and
in-flight reads are quiescent. Freeing after an unacknowledged notification or
a fixed grace period does not satisfy the ownership invariant.

## Failure behavior

If a decoder, router, or network path disappears before completion, producer
pages stay pinned. Operators should treat a rising overdue counter as a failed
consumer path and recycle or repair the affected service instance. Restarting an
instance destroys its allocator and transport session together, so no surviving
consumer can retain a valid address into that old allocation.

Bidirectional transfer follows the same rule. `decoder_kv_blocks_ttl` supplies a
liveness deadline for decoder-resident pages, but an elapsed deadline alone does
not authorize reuse while a later prefiller may still hold their block roster.

## Configuration

The ownership mechanism is configured through `kv_connector_extra_config`:

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `kv_lease_duration` | 30s | Initial producer deadline and basis for heartbeat cadence and extension. |
| `decoder_kv_blocks_ttl` | 480s | Liveness deadline for decoder-resident pages in bidirectional mode. |

```bash
vllm serve <MODEL> \
  --kv-transfer-config '{
    "kv_connector": "NixlConnector",
    "kv_role": "kv_producer",
    "kv_connector_extra_config": {"kv_lease_duration": 60}
  }'
```

`vllm:nixl_num_kv_expired_reqs` counts requests whose liveness deadline
elapsed before authoritative completion. The name refers to the deadline, not
to page release.

For full connector configuration, see the
[NixlConnector Usage Guide](../features/nixl_connector_usage.md).
