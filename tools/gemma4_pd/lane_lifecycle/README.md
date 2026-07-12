# Experiment-lane lifecycle

This package captures, quiesces, restores, and verifies the Gemma 4
experiment lane while treating the production stack as immutable. The exact
lane is:

| Role | Port | GPU ownership |
| --- | ---: | --- |
| P | 8810 | 0, 1, 2, 3 |
| D1 | 8811 | 4 |
| legacy proxy | 8812 | none |
| router | 8813 | none |
| D2 | 8815 | 5 |

Production listeners `8000`, `8820`, and `8821`, their complete process
groups, and every compute process on GPUs 6 and 7 are captured as protected
identity. Every mutation re-attests them. A target process group is rejected
if it overlaps any protected group.

Every command holds a nonblocking exclusive lock at
`/data/colleague/locks/gemma4-experiment-lane.lock` for its full inspection
and mutation window. Native campaigns must hold that same lock from preflight
through artifact sealing and cleanup, so a campaign and restore cannot both
observe empty GPUs and launch concurrently.

## Safety contract

`capture` is read-only with respect to servers and GPUs. It discovers owners
from the exact listener ports rather than stored PIDs. It seals:

- boot ID, PID/start time/PGID/SID, raw argv, full raw environment, cwd, the
  executable inode hash and an evidence copy;
- fd 0 identity, which must be `/dev/null`, plus fd 1 and fd 2 targets;
- the complete typed POSIX resource-limit vector, including
  `RLIMIT_NOFILE`;
- stable, double-read process-group membership;
- GPU inventory and joined compute-process ownership;
- listeners, connections, and zero running/waiting request metrics;
- launcher inputs, the authoritative storm37b archive, and the clean Git
  HEAD/tree plus every tracked lifecycle-tool file.

The snapshot directory is `0700`. Every payload, executable evidence copy,
metric body, seal, operation journal, and restore log is `0600`. Box-side
artifacts under `/data` are confined to
`/data/colleague/private/lane-lifecycle`; the public evidence and handoff
trees are rejected. Snapshot-relative paths cannot traverse symlinks or
escape the snapshot. `SHA256SUMS` binds every sealed payload.

`stop` rechecks the complete snapshot and samples zero request metrics twice.
It sends signals by process group in three phases:

1. router and legacy proxy;
2. D1 and D2, after engine API connections drain;
3. P.

Each signal is durably journaled before it is sent. `SIGKILL` is permitted
only after remaining group members are proven to be a subset of the captured
instances. Completion requires all five ports closed, every captured
PID/start-time instance and group absent, GPUs 0 through 5 clear, and
production unchanged.

If stop fails after a mutation, recovery classifies each role as the exact
captured instance, a launch-equivalent restored instance, fully absent, or
ambiguous. Existing valid services are retained and only absent services are
launched in `P -> D1 -> D2 -> router -> proxy` order. Ambiguous state is never
patched around. A successful automatic recovery requires a fresh capture
before another stop.

`restore` uses that same state-aware transaction. It rejects live ingress
when an engine dependency is absent and rejects a live D beneath an absent P.
It executes the hash-verified original executable path with the exact raw
argv, environment, cwd, `/dev/null` stdin, and resource limits. Evidence
copies are never executed because relocating CPython can alter prefix and
module discovery. New stdout/stderr logs are exclusive private files; old
logs are never truncated. On failure, every process launched by that
invocation is independently identity-checked and rolled back, with all
cleanup errors aggregated.

## Commands

Use a new private directory for capture. The following commands are examples;
they have deliberately not been run by this change.

```bash
python -m tools.gemma4_pd.lane_lifecycle capture \
  --artifact-dir /data/colleague/private/lane-lifecycle/pre-native-campaign

python -m tools.gemma4_pd.lane_lifecycle stop \
  --artifact-dir /data/colleague/private/lane-lifecycle/pre-native-campaign \
  --dry-run

python -m tools.gemma4_pd.lane_lifecycle stop \
  --artifact-dir /data/colleague/private/lane-lifecycle/pre-native-campaign \
  --execute

python -m tools.gemma4_pd.lane_lifecycle verify \
  --artifact-dir /data/colleague/private/lane-lifecycle/pre-native-campaign \
  --expected-state stopped

python -m tools.gemma4_pd.lane_lifecycle restore \
  --artifact-dir /data/colleague/private/lane-lifecycle/pre-native-campaign \
  --dry-run

python -m tools.gemma4_pd.lane_lifecycle restore \
  --artifact-dir /data/colleague/private/lane-lifecycle/pre-native-campaign \
  --execute

python -m tools.gemma4_pd.lane_lifecycle verify \
  --artifact-dir /data/colleague/private/lane-lifecycle/pre-native-campaign \
  --expected-state running
```

`captured` verification requires the original PID/start-time identities.
`running` accepts either those identities or an exact restored launch image.
`stopped` requires the lane and GPUs 0 through 5 to be empty. Every command
returns structured JSON and exits nonzero on the first unproved invariant.
