# Gemma 4 certification tools

This package records certification evidence without inferring success from an
incomplete response or from a mutable directory. It is deliberately independent
of the vLLM runtime so the recorder, offline attestation probe, and artifact
contract can run in a CPU-only environment.

## Artifact lifecycle

An arm plan is a JSON object with these fields:

```json
{
  "protocol_id": "gemma4-pd",
  "protocol_version": "1",
  "run_id": "run-20260712",
  "arm_id": "fresh-control",
  "created_at_utc": "2026-07-12T00:00:00Z",
  "hypothesis": "declared hypothesis",
  "intervention": "declared intervention",
  "state_predicate": {"producer_state": "fresh"},
  "payloads": [
    {
      "order": 0,
      "label": "low-cap-detector",
      "sha256": "151683767b7c33b56b60adfe9d1972741df47d8e9ca3e7a2cbb235cbe2183166"
    }
  ],
  "seeds": [17],
  "planned_parents": 1,
  "choices_per_parent": 8,
  "stopping_rule": {"parents": 1, "sequential_looks": 1},
  "acceptance_criteria": ["all mechanism gates pass"],
  "required_artifacts": [
    "client/requests.jsonl",
    "client/responses.jsonl",
    "client/choices.jsonl",
    "lineage.jsonl"
  ]
}
```

Initialize the arm before execution:

```bash
.venv/bin/python -m tools.gemma4_cert init \
  --cert-root /data/colleague/cert --plan arm-plan.json
```

This exclusively creates
`cert/<protocol_id>/<run_id>/<arm_id>/`, writes `plan.json`, fsyncs it,
and writes `plan.sha256`. Reusing an arm ID is an error, including when the old
arm is void. Every append revalidates the frozen plan. Sealing writes
`summary.json` and a sorted `SHA256SUMS` that excludes itself. Missing
predeclared artifacts force `INVALID`; an explicitly void arm remains `VOID`.
The manifest digest printed by `seal` must be copied to an external append-only
collector. A manifest on the same mutable host is detection, not an integrity
anchor.

Sealing also validates recorder completeness across files. Every planned parent
must have one request, one response, the exact planned choice rows, consistent
root/client-attempt identities, and exactly one start and terminal lineage event.
Recorder rows use versioned exact schemas and complete typed event envelopes.
Request, response, and response-derived rows must follow frozen payload-major
request-index order. Optional token and unexpected-choice files must be absent
when their independently derived row set is empty. Partial, repeated, reordered,
or malformed execution forces `INVALID`; an explicitly void arm remains `VOID`.
Reserved root files (`plan.json`, `plan.sha256`, `.arm.lock`,
`summary.json`, and `SHA256SUMS`) are owned exclusively by the artifact library.
Every required JSON artifact must parse. Required `attest/*.json` files must also
match the arm identity and the attestation process, repository-hash, artifact-array,
software-provenance, and runtime-evidence schema before an arm can pass.
Long-running writers use `ArtifactArm.writer_session()`, which holds the shared
lifecycle lock for the entire job. The recorder CLI holds one session around the
full parent batch, including network waits, so sealing cannot split a request or
commit between parents.

## Recorder input

The recorder consumes JSONL with one parent per line:

```json
{"root_request_id":"root-0000","request_index":0,"planned_choice_indices":[0,1,2,3,4,5,6,7],"body":{"model":"model","messages":[{"role":"user","content":"prompt"}],"n":8,"logprobs":true,"seed":17},"headers":{}}
```

Run it only against an initialized arm whose parent and choice counts match:

```bash
.venv/bin/python -m tools.gemma4_cert record \
  --arm /data/colleague/cert/gemma4-pd/run-20260712/fresh-control \
  --endpoint http://router/v1/chat/completions \
  --requests requests.jsonl
```

For every parent, the recorder first persists the canonical request bytes and
then the complete raw response or transport error. It emits exactly one row for
each planned choice. Missing, duplicate, malformed, non-2xx, and transport-error
outcomes remain explicit; out-of-range choices are preserved separately.
`lineage.jsonl` always contains root and client attempt IDs plus explicit nullable
router-attempt, producer-registration, D-child, and physical-pull fields. Runtime
components can append later lineage events using the same artifact API.
Authorization-like headers are replaced with a `<redacted>` marker and never
written in plaintext. Sensitive header values are discarded without retaining a digest.
Response header instances are archived in wire order without collapsing duplicate
names. Duplicate non-router fields remain certifiable; duplicate configured
router-attempt fields are retained as evidence but make the router identity
ambiguous and force `INVALID`.
Expected transport failures retain only a closed failure class, with no exception
message or traceback. Unexpected transport exceptions abort the writer session and
force an invalid arm rather than archiving caller-controlled diagnostics.
Endpoints containing userinfo, query strings, or fragments are rejected. The
HTTP transport ignores environment proxy settings and never follows redirects;
a 30x response and its raw body are evidence for the sole attempted request.

The plan size is exact: `planned_parents == len(payloads) * len(seeds)`.
Assignment is payload-major, so parent `i` uses payload
`i // len(seeds)` and seed `seeds[i % len(seeds)]`. A payload SHA-256 hashes the
canonical transmitted JSON body after removing only its top-level `seed` field.
Canonical JSON uses sorted keys, compact separators, UTF-8, and one trailing
newline. Every request, response, choice, token, and lineage row carries the
resolved payload order, label, template digest, and seed; sealing verifies those
bindings against the frozen plan.

## Attestation

The attestation module is suitable for runtime import. The standalone `attest`
command is an offline probe. It records the commit, tracked binary diff, untracked
file identities, loaded Python files, explicitly selected native GNU build IDs,
named launcher/model/tokenizer/adapter artifacts, process identity, and software
versions. It never calls `nvidia-smi` or initializes a GPU.

Runtime-observed config, KV specs, counters, invariants, and graph shapes are
stored separately from caller-supplied JSON. The offline probe should normally
use the `--supplied-*` options. Omitted runtime-only evidence is labeled
`unavailable`; it is not replaced with an empty invented observation. Process
arguments and the general environment are not archived. Only a small role/config
environment allowlist is recorded, with sensitive-looking names redacted.

```bash
.venv/bin/python -m tools.gemma4_cert attest \
  --arm /data/colleague/cert/gemma4-pd/run-20260712/fresh-control \
  --path attest/p-rank0.json \
  --repository . \
  --role producer --engine-boot-uuid BOOT --rank 0 --gpu-uuid GPU-UUID \
  --module tools.gemma4_cert.attestation \
  --artifact launcher=/path/to/launcher.sh \
  --supplied-effective-config effective-config.json
```

Native files passed with `--native` must be ELF artifacts with GNU build IDs.
Use `--include-mapped-native-libraries` only inside the process being attested;
an offline probe's maps describe the probe, not a server.

The offline CLI accepts only `--supplied-*` evidence. Runtime-observed config,
KV specs, counters, invariants, and graph shapes can be populated only through
the in-process `AttestationInputs.observed` API. The CLI writes through the arm's
writer session; it cannot create an unbound output beside the certificate.

A locally sealed `PASS` is not yet a deployment certificate. Release still
requires role-specific in-process G0 validation, complete router-to-P-to-D
runtime lineage, and receipt of the final manifest by the external append-only
collector. Those integrations are intentionally outside this offline tool.
