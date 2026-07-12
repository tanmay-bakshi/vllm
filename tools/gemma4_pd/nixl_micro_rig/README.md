# Gemma 4 NIXL transport micro-rig

This rig isolates the same-host TP4-to-TP1 NIXL/UCX data plane from model
startup. Its production-topology lane uses four independent producer OS
processes and NIXL agents, plus one consumer process and agent.

The checked-in geometry is authenticated against the captured storm37b
handshakes:

- four P ranks, each with ten 64,000-row registrations and 65,536-byte rows;
- one D rank with ten 64,000-row registrations and 262,144-byte rows;
- HND layout, block size 16, physical-to-logical ratio 1;
- one 12,288 MiB staging registration allocated after destination registration
  and stock prepared-dlist construction;
- raw coalesced READ handles with the server's region/rank/run staging layout;
- scatter rows laid out as `[K rank slots][V rank slots]`.

The 2,672-position roster is synthetic and is labeled as such in every plan.
Handshake capture proves registration geometry, not one request's exact group
roster. A versioned replay manifest can supply actual group-wise remote and
local block IDs through `ScenarioConfig.replay_manifest`. A synthetic result is
diagnostic evidence, never production-plan certification.

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

The strict XFAIL calls the source-bound production `_coalesce_drop_plan`
implementation. It demonstrates the known staging lifetime defect when one
rank is still `PROC` and a sibling reports `ERR`. The safe ownership model
requires a sealed post phase and terminal or cancellation-acknowledged state
for every possible writer before range reuse.

## GPU campaign

The GPU command is intentionally explicit:

```bash
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

The launcher copies configuration, authenticated handshake manifests, and any
request-plan replay into the immutable run directory, then every role loads
that copy. A child failure terminates its sibling process group immediately;
it does not strand four peers behind the 900-second diagnostic timeout.

Producer processes receive the preflighted UUID roster for physical GPUs
0,1,2,3 and select their logical TP-rank ordinal. The consumer receives only
the preflighted UUID for physical GPU 4, as logical device 0. UUID binding
prevents CUDA enumeration order from weakening the GPU 6/7 denylist. CUDA
visibility and UCX configuration are set before importing Torch or NIXL.

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
