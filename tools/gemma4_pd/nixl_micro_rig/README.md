# Gemma 4 NIXL transport micro-rig

## Current status

The production-topology rig is implemented, and its host-only planning,
self-test, integrity, lifecycle, semantic-handshake, and staging-generation
checks pass. No authoritative GPU transport result for the current connector-v11
implementation is preserved in the workspace. Native GPU qualification and its
immutable result tree are required before release; the rig, launcher, and static
fixtures alone do not constitute that result.

This rig isolates the same-host TP4-to-TP1 NIXL/UCX data plane from model
startup. Its production-topology lane uses four independent producer OS
processes and NIXL agents, plus one consumer process and agent.

The checked-in physical geometry is authenticated against the captured
storm37b handshakes:

- four P ranks, each with ten 64,000-row registrations and 65,536-byte rows;
- one D rank with ten 64,000-row registrations and 262,144-byte rows;
- HND layout, block size 16, physical-to-logical ratio 1;
- one 12,288 MiB staging registration allocated after destination registration
  and stock prepared-dlist construction;
- raw coalesced READ handles built from the production canonical transport
  planner, with one outer slab per P rank and owned-region offsets within it;
- scatter rows laid out as `[K rank slots][V rank slots]` by the same
  event-owned Triton implementation used by the inference server.

Those captures contain the legacy physical handshake fields only. They do not
contain connector-v11 `registration_generation`, region descriptors, source
group planes, physical group token capacities, or model-derived semantic
names. Their SHA-256 identity therefore authenticates the listed physical
geometry, never the connector-v11 semantic contract.

`connector-v11-semantic-fixtures.json` is a separate, SHA-authenticated static
fixture. Each configuration selects one typed profile whose ordered group
names, plane counts, token capacities, region names, and ownership must match
the rig exactly. Host tests materialize complete packed source and destination
region descriptors from that profile and pass the full four-rank TP4-to-TP1
roster through the production connector-v11 handshake validator. This proves
that the declared rig contract satisfies the current validator, including
TP row scaling and cross-rank equality. The profile explicitly records
`runtime_capture_authenticated: false`: its synthetic names, tensor views,
addresses, and registration generations are not evidence of a live Gemma 4
registration. Runtime semantic authentication requires a preserved live
connector-v11 payload from the deployed stack.

The 2,672-position roster is synthetic and is labeled as such in every plan.
Authoritative ownership expands it to 13,360 transferred region positions per
P rank: groups 0 through 11 own regions 0 through 4, while group 12 owns
regions 5 through 9. The resulting request occupies 3,340 MiB instead of the
old 6,680 MiB all-position/all-region cross-product. Every plan records the
canonical SHA-256 layout digest, per-region positions and runs, rank stride,
and region offsets within each rank slab.
The legacy capture proves registration geometry, not one request's exact group
roster. A versioned replay manifest can supply actual group-wise remote and
local block IDs through `ScenarioConfig.replay_manifest`. A synthetic result is
diagnostic evidence, never production-plan certification.

The separately selectable `gemma4_tp4_to_tp1_2k.json` profile reproduces the
exact 2K production request geometry used by native qualification:

- groups 0 through 9 contain 64 dual-plane positions each;
- groups 10 and 11 contain 65 dual-plane positions each;
- group 12 contains 33 dual-plane positions;
- groups 0 through 11 own regions 0 through 4, while group 12 owns regions 5
  through 9;
- every P row is 65,536 bytes, producing 4,015 region positions per rank and
  an exact TP4 staging allocation of 1,052,508,160 bytes (1003.75 MiB).

The production profile is all dual-plane. Separate host cases and the CUDA
scatter equivalence test cover the optional single-plane global-group mode,
including odd suffixes and absolute source-position half selection. The native
NIXL micro-rig campaigns remain faithful to the deployed all-dual-plane DFlash
geometry rather than presenting synthetic single-plane transport as production
evidence.

Fragmentation scenarios are calibrated after ownership pruning. The largest
ownership partition spans five regions, so 203 region-local runs produce 1,020
descriptors per handle and 204 produce 1,025, the first attainable count above
NIXL's 1,024-descriptor split boundary for this topology.

## Host-only checks

Run these without reserving or initializing a GPU:

```bash
VENV=/data/gemma4-2026-06-optimization-effort/vllm/.venv
CONFIG=tools/gemma4_pd/nixl_micro_rig/gemma4_tp4_to_tp1.json

$VENV/bin/python -m tools.gemma4_pd.nixl_micro_rig plan --config "$CONFIG"
$VENV/bin/python -m tools.gemma4_pd.nixl_micro_rig self-test --config "$CONFIG"
$VENV/bin/python -m pytest -q \
  tests/tools/test_nixl_micro_rig.py \
  tests/v1/kv_connector/unit/test_nixl_integrity.py
```

The staging-generation regression calls the source-bound production ownership
path. A generation with any posted native operation remains live until every
possible writer is terminal and device quiescence is established. `ERR`,
`UNKNOWN`, nonterminal timeout, or failed handle release tombstones the
generation and forbids range reuse. The historical fail-first result proved
the former defect; the current passing test proves the replacement host state
machine, not native GPU behavior.

## GPU campaign

The GPU command is intentionally explicit:

```bash
$VENV/bin/python -m tools.gemma4_pd.nixl_micro_rig run \
  --config "$CONFIG" \
  --artifact-root /data/colleague/micro-rig-runs
```

### Target 2 fixed-byte transport gate

The Target 2 gate compares the existing owner-aware READ against producer
packing followed by contiguous READ and contiguous WRITE. Every cell moves the
same exact 1,052,508,160-byte request. It covers the contiguous control plus
the calibrated C1 and C64 fragmentation regimes, 64/128/256/512 MiB bounded
chunks, and configurable in-flight request depth in one 4P+1D process group.
It calls the serving worker's production `PackedTransferPlan`, Triton pack, and
Triton TP4 scatter primitives directly. Two guarded slots per producer rank and
two guarded rank-major TP4 receive slots on the decoder execute identical
two-task waves for packed READ and packed WRITE without unbounded memory. The
decoder advertises each complete receive-slot base, and both directions derive
rank slabs with the active candidate's chunk stride. A smaller candidate never
inherits the largest swept candidate's rank spacing.

Planning is host-only:

```bash
CONFIG=tools/gemma4_pd/nixl_micro_rig/gemma4_tp4_to_tp1_2k.json

$VENV/bin/python -m tools.gemma4_pd.nixl_micro_rig target2-plan \
  --config "$CONFIG" \
  --chunk-mib 64 128 256 512 \
  --in-flight-depth 1 2 4 8 \
  --warmup-batches 1 \
  --measured-batches 3
```

After the lifecycle owner has cleanly retired every process on GPUs 0 through
5, run the native CUDA-IPC/UCX gate from the checked-out repository:

```bash
$VENV/bin/python -m tools.gemma4_pd.nixl_micro_rig target2-gate \
  --config "$CONFIG" \
  --artifact-root /data/colleague/gemma4-target2-fixed-byte-gate \
  --transport-arm cuda_ipc \
  --chunk-mib 64 128 256 512 \
  --in-flight-depth 1 2 4 8 \
  --warmup-batches 1 \
  --measured-batches 3
```

The environment must provide the repository's Python dependencies, CUDA-enabled
Torch, the NIXL Python package and UCX plugin used by the serving stack,
`nvidia-smi`, and writable access to `/data/colleague/locks` plus the selected
artifact root. The launcher refuses occupied or insufficiently free GPUs 0
through 5 and never admits GPUs 6 or 7.

The immutable run directory contains `target2-gate-plan.json`,
`target2-code-identity.json`, `campaign-start.json`, `preflight.json`, and terminal
`postflight.json` and `campaign-complete.json`. Postflight waits for all selected
GPU registrations to disappear and requires the exact protected-process roster
captured before the run to remain unchanged. Each transport-arm directory
contains five process attestations and logs, raw `target2-gate-batches.json`,
and validated `target2-gate-summary.json`. CUDA-IPC arms also contain
`target2-cuda-ipc-write-protocol.json`; publication fails unless every producer
log proves a CUDA-to-CUDA `remote memory write` selected UCX zero-copy
`cuda_ipc/cuda`. Host/control `tcp/lo` protocol rows do not satisfy that check.
The summary schema is
`target2_fixed_byte_native_transport_gate` version 1. It reports per-cell
median elapsed time, GiB/s, native critical-path time, producer-pack time,
decoder-scatter time, and direct-relative latency/goodput deltas. A summary is
written only after every warmup verifies every received chunk and completed
request, while every measured batch proves exact source preservation, wire-byte
coverage, final-request destination content, guards, notifications, descriptor
geometry, backend selection, and slot lifetime. This keeps byte-oracle work out
of the measured packed critical path without calling the resulting evidence
bit-exact for unretained intermediate measured requests.

The machine decision is `target2_decision` in the arm summary. It evaluates
only the calibrated C64 fragmentation regime (2,120 descriptors per rank),
which is above the production packed-WRITE activation threshold of 1,024.
Every q1/q2/q4/q8 median speed ratio must be at least 0.98, q4 and q8 must each
be at least 1.0, and their geometric mean must be at least 1.05. Candidates
within one percent of the best geometric mean are tied in favor of the smaller
registered pool and then READ. `pass` authorizes the selected arm and chunk;
`fail_no_non_regressing_candidate` is a hard architecture stop, not permission
to weaken the gate. The serving conformance below is intentionally narrower:
its already-selected candidate is connector-v11 producer-initiated WRITE256,
while direct and packed READ remain selection-evidence controls only.

Once the gate has selected WRITE, serving qualification must not spend another
campaign reselecting it. The focused host-only plan fixes the winner at a
256 MiB per-rank stride and defines two live cases: six exact-2K C64 requests
alternating across two decoders, and three requests whose per-rank payload is
two full chunks plus a 64 KiB tail. Both cases force reuse of the two producer
and decoder slots. Their live result must preserve exact source and destination
bytes, one descriptor per rank chunk, authenticated native-sender completion,
the CUDA-IPC protocol evidence above, all-free final pools, and the protected
GPU baseline.

```bash
$VENV/bin/python -m tools.gemma4_pd.nixl_micro_rig \
  target2-conformance-plan --config "$CONFIG"
```

The code identity covers the gate, canonical layout, and exact production pack
and scatter sources. The launcher re-hashes them after all cells and refuses to
publish `campaign-complete.json` if any inode, timestamp, size, or digest
changes during the run.

Select the exact 2K production-semantic profile with:

```bash
CONFIG=tools/gemma4_pd/nixl_micro_rig/gemma4_tp4_to_tp1_2k.json

$VENV/bin/python -m tools.gemma4_pd.nixl_micro_rig run \
  --config "$CONFIG" \
  --artifact-root /data/colleague/micro-rig-runs
```

The launcher fails closed unless all of these conditions hold:

- physical GPUs 0 through 5 are the only permitted devices;
- GPUs 6 and 7 remain on an immutable denylist;
- every selected GPU has no foreign compute process;
- every selected GPU has enough free memory for its exact registration role;
- the host-wide micro-rig lock is exclusively held;
- every UCX transport arm starts in fresh spawned processes.

The launcher copies configuration, both authenticated evidence manifests, and
any request-plan replay into the immutable run directory, then every role
loads that copy. A child failure terminates its sibling process group immediately;
it does not strand four peers behind the 900-second diagnostic timeout.

Producer processes receive the preflighted UUID roster for physical GPUs
0,1,2,3 and select their logical TP-rank ordinal. The consumer receives only
the preflighted UUID for physical GPU 4, as logical device 0. UUID binding
prevents CUDA enumeration order from weakening the GPU 6/7 denylist. CUDA
visibility and UCX configuration are set before importing Torch or NIXL.

## Live connector-v11 capture

The production candidate must preserve one live wire capture for each TP1
decoder after all candidate side-channel listeners are READY and before the
first request. Do not run this command against a connector-v8 deployment. The
capture sends only the ordinary read-only `GET_META_MSG`; it does not import a
remote agent, register memory, or post a native operation.

The lifecycle authority must prove that ports 15620, 15621, and 15622 belong
to the candidate process groups immediately before and after capture. Engine
IDs are runtime-volatile, so the command derives them from those authenticated
endpoints. All four P ranks must advertise one exact engine identity, and the
P and D identities must differ. Optional `--producer-engine-id` and
`--decoder-engine-id` arguments add independent equality assertions when such
an authority is available.

The model's total KV-head count is included in the compatibility hash but is
not recoverable from the handshake payload. The capture therefore accepts the
exact deployed `config.json`, requires its independently preserved SHA-256 from
the already authenticated model-source manifest, embeds its raw bytes, and
derives `num_key_value_heads` itself. A caller-supplied integer is not accepted
as model evidence.

```bash
CONFIG=tools/gemma4_pd/nixl_micro_rig/gemma4_tp4_to_tp1_2k.json
CANDIDATE=/data/colleague/tanmay-gemma4-owner-aware-transfer-<commit12>-v1
EVIDENCE=/data/colleague/gemma4-production-fixed-q16-owner-aware-gate-evidence-<utc>-v1
MODEL_CONFIG=/data/gemma4/models/gemma-4-31B-it-NVFP4/config.json
MODEL_CONFIG_SHA256=aa03a6a490fb743b8186f09c60ec39a30fdbaf7c3a18dfef8e52a62c2beb9ae4

$VENV/bin/python -m tools.gemma4_pd.nixl_micro_rig capture-handshake \
  --config "$CONFIG" \
  --code-root "$CANDIDATE" \
  --producer-host 127.0.0.1 \
  --producer-port 15620 \
  --decoder-host 127.0.0.1 \
  --decoder-port 15621 \
  --model-config "$MODEL_CONFIG" \
  --expected-model-config-sha256 "$MODEL_CONFIG_SHA256" \
  --output "$EVIDENCE/live-connector-v11-d1.json" \
  --result-output "$EVIDENCE/live-connector-v11-d1.capture-result.json"
```

Repeat the command with decoder port 15622 and distinct capture/result outputs.
The gate evidence manifest must preserve each capture digest from its result
artifact independently, then replay it before traffic:

```bash
$VENV/bin/python -m tools.gemma4_pd.nixl_micro_rig verify-handshake \
  --config "$CONFIG" \
  --code-root "$CANDIDATE" \
  --capture "$EVIDENCE/live-connector-v11-d1.json" \
  --expected-sha256 "$CAPTURE_SHA256" \
  --expected-model-config-sha256 "$MODEL_CONFIG_SHA256" \
  --result-output "$EVIDENCE/live-connector-v11-d1.verify-result.json"
```

Verification authenticates the external capture digest, clean Git commit and
tree, typed rig fingerprint, every exact outer and inner MessagePack payload,
the shared P/D compatibility hash, all four P ranks, and the exact captured D
registration contract. It then invokes the production all-rank handshake
validator without native mutation. The terminal verdict is
`passed_without_native_mutation`.

vLLM imports may emit process logs before the CLI's own stdout. Automation
must consume the exclusive, read-only `--result-output` JSON artifact and
authenticate that file independently; it must not parse stdout as a complete
JSON document.

Raw payloads preserve opaque NIXL metadata, registration addresses, device
ordinals, and registration generations exactly. The separate normalized
cross-rank digest omits only those rank- or process-volatile values and binds
the fields the production validator requires to agree across P ranks: engine,
backend, cache and block geometry, source planes, token capacities, and
address-independent region semantics. The raw payload hashes and generation
hashes ensure normalization cannot hide capture tampering.

## Evidence contract

Each nonzero transfer records BLAKE2b-128 leaves with canonical producer,
child, rank, region, group, source-position, remote/local block, stage, and
generation lineage. Source, staging, and destination leaves are compared. An
independently generated rank/region/physical-block/KV/iteration byte oracle
also verifies each stage exactly. The payload is deliberately a function of
physical source address, so replay plans with the same remote block in multiple
groups remain deterministic while their group and source-position lineage stays
distinct in the integrity identity.

Each transfer also seeds adjacent staging guard bands and every destination
row that the raw transport must leave untouched. Both are verified after NIXL
completion and before scatter, so a clean destination digest cannot hide an
earlier out-of-range write that scatter subsequently overwrote.

The CUDA lane launches the production asynchronous scatter on one persistent
stream, waits on its completion event, records exact CUDA-event duration, and
then verifies the independent staging and destination oracles. The CPU lane
uses the canonical synchronous reference implementation.

Zero-byte full-prefix operations retain notification and lineage evidence but
emit `NON_EVIDENTIARY_ZERO_BYTE`. They never enter a clean content denominator.

An overlap arm is valid only when at least one NIXL handle is observed in
`PROC` while the deterministic victim CUDA event remains incomplete. Victim
inputs, output, and redzones are verified after transfer completion.

Every process records:

- argv and relevant CUDA, UCX, and NIXL environment;
- NIXL plugin and instantiated UCX backend parameters;
- queried backend and per-handle telemetry;
- mapped NIXL, UCX, and `libplugin_UCX` paths and SHA-256 digests;
- exact UCX version from the loaded `libucp`;
- full `/proc/self/maps` plus its SHA-256 digest;
- full integrity observations and iteration summaries.

No clean micro-rig result exonerates the inference server. A firing result
localizes the defect below model execution. Full-model storms remain the final
integration and certification layer.
