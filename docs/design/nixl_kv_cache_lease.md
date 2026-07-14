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
terminal-proof set for that producer rank: either all child/rank read
completions or the decoder-rank cancellation quorum.

The producer owns the completion contract. Under the required serving
contract, the producer leg has an effective `n` of one. Before offering blocks,
the coordinator supplies a positive `expected_consumers` count for the logical
decoder children and the decoder tensor-parallel size. Each producer rank
derives its immutable proof set as the Cartesian product of:

- every logical child index; and
- every decoder rank that can address that producer rank under the configured
  producer/decoder tensor-parallel mapping.

When decoder TP is at least producer TP, each producer rank owns its contiguous
group of decoder ranks. When producer TP is larger, each producer rank owns the
single decoder rank to which it maps. The TP sizes must divide evenly in either
direction.

Producer source ownership and decoder admission ownership are linked but
concurrent state machines. The producer owns its source pages before the offer
is returned to serving. Decoder serving ownership determines which component
may choose the offer's terminal path; it does not own the producer pages.

## Producer source lifecycle

1. **Owned with a deadline.** The request is present in `_reqs_to_process` and
   `_reqs_to_send` before the offer leaves the producer. Heartbeats may extend
   its deadline.
2. **Overdue but owned.** The deadline elapsed, so `_reqs_to_send` is removed
   and the overdue metric is recorded once. `_reqs_to_process`, partial
   terminal proofs, source rosters, and KV pages remain intact.
3. **Read-completed.** Every exact child/rank read obligation has a valid
   completion proof.
4. **Cancelled before decoder admission.** Every expected decoder rank has
   supplied one typed whole-offer cancellation proof and no rank has reported
   read ownership.
5. **Released.** Exactly one valid terminal proof set is complete:
   all child/rank read completions or the decoder-rank cancellation quorum.
   The worker removes `_reqs_to_process`, reports `finished_sending`, and the
   scheduler may reuse the pages.

Mixed, missing, or conflicting terminal modes retain producer ownership. A
dead or unreachable consumer can retain capacity, but it cannot turn a stale
block address into apparently successful inference.

## Decoder admission ownership

1. **Serving-owned offer.** The coordinator attaches one producer offer after
   rendering. EngineCore has not acknowledged scheduler commit, so serving
   owns the decision to admit or reject it.
2. **Pre-admission rejection.** Every D worker verifies that it has no local
   read state and queues a typed whole-offer cancellation proof to the exact
   producer rank. The synchronous acknowledgement proves the cancellation was
   queued, not that producer pages were already released.
3. **EngineCore-owned offer.** EngineCore validates and atomically commits
   every decoder child, then acknowledges admission. Scheduler and connector
   cleanup are EngineCore-owned from this point. Serving must not submit
   pre-admission cleanup after acknowledgement.

Producer workers consume queued cancellation proofs during connector
maintenance and release asynchronously only after observing the exact
decoder-rank quorum. Once any decoder read state exists, cancellation is
refused and only the exact child/rank read-completion set can release the
source.

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

The producer returns `expected_consumers` and decoder TP size with the block
offer. A decoder must match that contract to its local TP world, require its
effective `n` to equal `expected_consumers`, and derive each stable child index
from decoder request lineage before it can read.

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

## Pull request-shape boundary

The intended pull contract is:

| Leg | Required contract |
| --- | --- |
| Producer with `do_remote_decode` | effective `n=1` and non-streaming |
| Consumer with `do_remote_prefill` | effective `n` equals positive `expected_consumers`; streaming is allowed |
| Any KV request with beam search | rejected before ownership transfer |
| Completion KV request with prompt fanout | rejected before ownership transfer |
| Standalone render endpoint with KV metadata | rejected; the coordinator attaches one contract after singleton rendering |

Chat Completions, Completions, and the disaggregated Generate API enforce the
directional sampling rules above. Standalone render rejects direct KV metadata
and beam search. Enforcement is not complete in the current dirty candidate:

- the Responses API does not call the directional validator;
- a KV dictionary is not required to contain exactly one of
  `do_remote_decode` and `do_remote_prefill`;
- completion derender warns and drops mismatched response contracts instead of
  failing closed.

These are release blockers recorded in the active Gemma 4 handoff. Until they
are fixed, the table is the required contract rather than a claim that every
frontend enforces it. Multi-prompt KV accounting is unsupported.

## Failure behavior

If a decoder, router, or network path disappears before completion, producer
pages stay pinned. Operators should treat a rising overdue counter as a failed
consumer path and recycle or repair the affected service instance. Restarting an
instance destroys its allocator and transport session together, so no surviving
consumer can retain a valid address into that old allocation.

Bidirectional transfer follows the same no-timeout-release invariant.
`decoder_kv_blocks_ttl` supplies a liveness deadline for decoder-resident pages,
but an elapsed deadline alone does not authorize reuse while a later prefiller
may still hold their block roster.

The exact typed child/rank completion and cancellation contract in this
document describes pull mode. The push connector has a separate wire and
accounting protocol. Push is outside the Gemma F9 evidence boundary until it is
upgraded and tested against the same ownership guarantees.

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
