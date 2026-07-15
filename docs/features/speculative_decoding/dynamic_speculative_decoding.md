# Dynamic Speculative Decoding

## Why is Dynamic SD needed?

SD methods need to verify K tokens for each sequence during decoding. As BS increases, the effective BS becomes BS\*K which increases the compute requirement during verification. When this BS\*K goes beyond a critical BS then SD negatively impacts the decode speed (TPOT). DSD helps by tuning the K to an optimal value such that we continue to reap the benefits from SD.

## Use cases

* Variable concurrency workload using same deployment. K would decrease as concurrency increases.
* During RL rollout where we start off with high BS but then end up with small BS due to very few long tail request which end up generating a lot of tokens stalling the progress of the current rollout. Here K would go up during the end of rollout.

## `--speculative-config` schema

To use Dynamic SD, add `num_speculative_tokens_per_batch_size` to the config of an SD method which is a list of list. Here, an entry is `[start_bs, end_bs, optimal_K]` which means when the concurrency is within range `[start_bs, end_bs]` then `optimal_K` number of draft tokens are used. For e.g.,

```bash
--speculative-config '{
    "method": "eagle",
    "model": "yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
    "num_speculative_tokens": 3,
    "num_speculative_tokens_per_batch_size": [
      [1, 64, 3],
      [65, 128, 1],
      [129, 512, 0]
    ]
  }'
```

implies that:

* K=3 will be used when the concurrency is in range [1, 64]
* K=1 will be used when the concurrency is in range [65, 128]
* K=0 will be used when the concurrency is in range [129, 512], i.e., no draft tokens will be produced.

## Online Examples

### Dynamic SD Eagle Drafter

```bash
VLLM_USE_V2_MODEL_RUNNER=0 vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --speculative-config '{
    "method": "eagle",
    "model": "yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
    "num_speculative_tokens": 3,
    "num_speculative_tokens_per_batch_size": [
      [1, 64, 3],
      [65, 128, 1],
      [129, 512, 0]
    ]
  }'
```

### Dynamic SD Eagle3 Drafter

```bash
VLLM_USE_V2_MODEL_RUNNER=0 vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --speculative-config '{
    "method": "eagle3",
    "model": "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
    "num_speculative_tokens": 3,
    "num_speculative_tokens_per_batch_size": [
      [1, 16, 5],
      [17, 32, 4],
      [33, 64, 3],
      [65, 128, 1],
      [129, 512, 0]
    ]
  }'

```

### Cost-aware DFlash target verification

DFlash adaptive verification does not shorten the draft-model input. The
drafter always executes its complete trained block of 16 queries and produces
15 draft tokens. The scheduler exposes only a prefix of 3, 7, 11, or 15 draft
tokens to the next target pass, producing target query lengths 4, 8, 12, or 16.

`dflash_adaptive_verification.costs` is an offline measured table. Each
calibrated batch-size and sequence-length rectangle must contain one total round
cost for every target query length, and calibrated rectangles must not overlap.
An operating point outside those rectangles uses `fallback_query_len` (16 by
default), capped by the active static schedule. This leaves unmeasured or
unreachable regions conservative without inventing costs. Incomplete prefill
chunks do not contribute to the decode batch size. A round cost includes target
verification, the fixed 16-query draft pass, and sampling. At runtime, the
policy updates conditional per-position acceptance hazards once per executed
batch, with rejected tails treated as censored rather than failed observations.
Invalid target padding is excluded. The policy chooses the prefix with the
highest expected accepted output tokens per millisecond. A switching threshold
prevents oscillation, and periodic full-block probes keep acceptance estimates
for the tail current.

Every DFlash target verification cohort uses one supported query length. A row
without a proposal can join an existing cohort through explicit invalid
padding, which is excluded from acceptance statistics. DFlash's padded proposal
pass still retains every scheduled row, including incomplete prefills, so its
context-window guard uses the longest row in the complete padded drafter batch.

The ordinary one-query decode graph remains available when the fixed DFlash
window no longer fits near the drafter model's sequence-length limit. In that
case no proposal is emitted. Invalid target positions remain `-1` in rejection
metadata; their clamped embedding input is a non-observable implementation
detail rather than a draft-token guess.

```bash
VLLM_USE_V2_MODEL_RUNNER=0 vllm serve TARGET_MODEL \
  --speculative-config '{
    "method": "dflash",
    "model": "DFLASH_MODEL",
    "num_speculative_tokens": 15,
    "dflash_adaptive_verification": {
      "costs": [
        {"batch_size_range":[1,32],"sequence_length_range":[1,131072],"query_len":4,"round_cost_ms":3.0},
        {"batch_size_range":[1,32],"sequence_length_range":[1,131072],"query_len":8,"round_cost_ms":4.0},
        {"batch_size_range":[1,32],"sequence_length_range":[1,131072],"query_len":12,"round_cost_ms":5.0},
        {"batch_size_range":[1,32],"sequence_length_range":[1,131072],"query_len":16,"round_cost_ms":6.0}
      ],
      "fallback_query_len": 16,
      "initial_acceptance_rates": [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.15,0.1,0.08,0.06,0.04,0.02,0.01]
    }
  }'
```

The numbers above demonstrate the schema only. Production costs and acceptance
priors must come from the deployment's matched workload and hardware.

For low-overhead calibration, set
`VLLM_DFLASH_CALIBRATION_LOG_INTERVAL` to a positive number of completed target
passes, such as 32. The EngineCore then emits one JSON summary per interval,
with points keyed by the query length actually executed (`q`) and verification
batch size (`r`), the minimum and maximum ending sequence lengths, and counts
of non-verification and invalid-padding rows. The default value is zero, which
does not invoke the calibration logger. Timed calibration should use this
aggregate instead of per-iteration logging, since writing one line per pass
perturbs the cadence being measured.

## Limitations

* only usable with Model Runner V1
* Eagle and Eagle-3 dynamic drafting use piecewise CUDA graphs
* DFlash target-only adaptive verification supports full decode CUDA graphs for
  query lengths 4, 8, 12, and 16
