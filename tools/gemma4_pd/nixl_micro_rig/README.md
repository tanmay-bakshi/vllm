# Gemma 4 NIXL transport micro-rig

## Current status

The production-topology rig is implemented, and its host-only planning,
self-test, integrity, lifecycle, semantic-handshake, and staging-generation
checks pass. No authoritative GPU transport result for the current connector-v9
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
contain connector-v9 `registration_generation`, region descriptors, source
group planes, physical group token capacities, or model-derived semantic
names. Their SHA-256 identity therefore authenticates the listed physical
geometry, never the connector-v9 semantic contract.

`connector-v9-semantic-fixtures.json` is a separate, SHA-authenticated static
fixture. Each configuration selects one typed profile whose ordered group
names, plane counts, token capacities, region names, and ownership must match
the rig exactly. Host tests materialize complete packed source and destination
region descriptors from that profile and pass the full four-rank TP4-to-TP1
roster through the production connector-v9 handshake validator. This proves
that the declared rig contract satisfies the current validator, including
TP row scaling and cross-rank equality. The profile explicitly records
`runtime_capture_authenticated: false`: its synthetic names, tensor views,
addresses, and registration generations are not evidence of a live Gemma 4
registration. Runtime semantic authentication requires a preserved live
connector-v9 payload from the deployed stack.

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

## Live connector-v9 capture

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
  --output "$EVIDENCE/live-connector-v9-d1.json" \
  --result-output "$EVIDENCE/live-connector-v9-d1.capture-result.json"
```

Repeat the command with decoder port 15622 and distinct capture/result outputs.
The gate evidence manifest must preserve each capture digest from its result
artifact independently, then replay it before traffic:

```bash
$VENV/bin/python -m tools.gemma4_pd.nixl_micro_rig verify-handshake \
  --config "$CONFIG" \
  --code-root "$CANDIDATE" \
  --capture "$EVIDENCE/live-connector-v9-d1.json" \
  --expected-sha256 "$CAPTURE_SHA256" \
  --expected-model-config-sha256 "$MODEL_CONFIG_SHA256" \
  --result-output "$EVIDENCE/live-connector-v9-d1.verify-result.json"
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
