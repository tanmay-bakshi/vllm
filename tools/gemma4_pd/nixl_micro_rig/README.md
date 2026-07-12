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
  tests/tools/test_nixl_micro_rig_hardening.py \
  tests/tools/test_nixl_staging_safety.py \
  tests/v1/kv_connector/unit/test_nixl_integrity.py \
  tests/v1/kv_connector/unit/test_nixl_localization.py \
  tests/v1/kv_connector/unit/test_p2d_draft_barrier.py
```

The source-bound staging tests call the production implementation directly.
The safe ownership model requires a sealed post phase, authoritative `DONE`
or a state that never posted for every possible writer, and completed device
reads before range reuse. `ERR` and `UNKNOWN` never prove native quiescence.

## GPU campaign

The GPU command is intentionally explicit:

```bash
$VENV/bin/python -m tools.gemma4_pd.nixl_micro_rig run \
  --config "$CONFIG" \
  --scenario observed_ceiling_102_runs \
  --transport-arm cuda_copy \
  --artifact-root /data/colleague/micro-rig-runs
```

One invocation runs exactly one scenario and one transport arm. The complete
configuration remains byte-identical; `inputs/selection.json` binds the chosen
pair without constructing a reduced configuration.

The launcher fails closed unless all of these conditions hold:

- physical GPUs 0 through 5 are the only permitted devices;
- GPUs 6 and 7 remain on an immutable denylist;
- every selected GPU has no foreign compute process;
- every selected GPU has enough free memory for its exact registration role;
- the shared experiment-lane lock is held from source capture through atomic
  publication, excluding lifecycle restore and other campaigns;
- PID, start time, process group, raw argv, and executable identities on
  production GPUs 6/7 remain identical;
- listener ownership for ports 8000, 8820, and 8821 remains identical;
- every role starts in a fresh Python interpreter with a finite supervisor
  deadline, and the selected GPUs are empty again after cleanup.

The launcher copies configuration, authenticated handshake manifests, and any
request-plan replay into a hidden run directory, then every role loads that
copy. A child failure terminates and, if needed, kills every owned process
group under bounded waits. Publication happens only after cross-artifact,
checksum, read-only seal, final-status, and directory-identity validation.

Producer processes receive the preflighted UUID roster for physical GPUs
0,1,2,3 and select their logical TP-rank ordinal. The consumer receives only
the preflighted UUID for physical GPU 4, as logical device 0. UUID binding
prevents CUDA enumeration order from weakening the GPU 6/7 denylist. CUDA
visibility and UCX configuration are set before importing Torch or NIXL.
Inherited `UCX_*` and `NIXL_*` variables are removed, the selected UCX
vocabulary is installed exactly, and role `RLIMIT_NOFILE` is bound to the
production soft/hard pair of 65,535/1,048,576.

## Evidence contract

Each nonzero transfer records separate `SOURCE_PRE` and `SOURCE_POST` files
plus BLAKE2b-128 leaves with canonical producer,
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
- mapped NIXL, UCX, and `libplugin_UCX` paths, SHA-256 digests, and archived
  loaded bytes;
- exact UCX version from the loaded `libucp`;
- full `/proc/self/maps` plus its SHA-256 digest;
- full `/proc/self/limits`, its SHA-256 digest, and exact file-descriptor limit;
- full integrity observations and iteration summaries.

Each published directory has an explicit `PASS`, `FAIL`, or `INVALID`
disposition. `FAIL` is reserved for an atomically recorded, typed correctness
mismatch. Setup drift, crashes, timeouts, missing evidence, cleanup uncertainty,
and protected-production changes are `INVALID`; they are never inferred as a
correctness failure from a nonzero process exit. Validate an artifact offline
with:

```bash
$VENV/bin/python -m tools.gemma4_pd.nixl_micro_rig validate \
  --run-directory /data/colleague/micro-rig-runs/<run-id>
```

No clean micro-rig result exonerates the inference server. A firing result
localizes the defect below model execution. Full-model storms remain the final
integration and certification layer.
