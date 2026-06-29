# Gemma 4 NVFP4 Serving

This branch contains the Gemma 4 changes used by the B300 deployment:

- Gemma 4 tool-call and reasoning parsing fixes.
- A deployable Gemma 4 chat template at `examples/gemma4/tool_chat_template_gemma4.jinja`.
- Gemma 4 text-attention routing to the FlashInfer TRTLLM-GEN path when the checkpoint and runtime support it.
- Gemma 4 MTP/speculative decoding fixes and optional adaptive MTP depth.
- Disaggregated KV-transfer changes used during the P/D experiments.

The measured primary deployment shape was multiple independent TP1 replicas of the
NVFP4 `google/gemma-4-31B-it` checkpoint, each paired with the Gemma 4 assistant
checkpoint for MTP.

## Runtime Requirements

- NVIDIA Blackwell-class GPU support with CUDA and FlashInfer built with TRTLLM attention support.
- A Gemma 4 NVFP4 checkpoint whose config declares ModelOpt FP8 KV cache.
- The matching Gemma 4 assistant checkpoint for MTP.
- `VLLM_KV_CACHE_LAYOUT=HND`, which the Gemma 4 TRTLLM-GEN path also sets during config verification.
- `VLLM_USE_V2_MODEL_RUNNER=0`.
- `VLLM_BATCH_INVARIANT=0`.

## Launch Shape

Replace the checkpoint paths, cache paths, and GPU id with paths appropriate for
the target machine.

```bash
export CUDA_VISIBLE_DEVICES=0
export VLLM_KV_CACHE_LAYOUT=HND
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_BATCH_INVARIANT=0

vllm serve /path/to/gemma-4-31B-it-NVFP4 \
  --served-model-name google/gemma-4-31B-it \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.9027 \
  --limit-mm-per-prompt '{"image":4}' \
  --reasoning-parser gemma4 \
  --tool-call-parser gemma4 \
  --enable-auto-tool-choice \
  --chat-template examples/gemma4/tool_chat_template_gemma4.jinja \
  --default-chat-template-kwargs '{"enable_thinking": true}' \
  --async-scheduling \
  --speculative-config '{"method":"mtp","model":"/path/to/gemma-4-31B-it-assistant","num_speculative_tokens":6}' \
  --stream-interval 20
```

The production shape used one process per GPU and placed a load balancer in
front of the replicas.

## Optional Diagnostics

These settings are off in the primary deployment unless explicitly enabled:

```bash
export VLLM_GEMMA4_REQUEST_TRACE=1
export GEMMA4_SCHEDULER_TRACE=1
export VLLM_P2P_NCCL_TRACE=1
export VLLM_P2P_NCCL_REQUEST_TIMING=1
```

Adaptive MTP is also optional. The primary deployment currently leaves it off.

```bash
export VLLM_GEMMA4_ADAPTIVE_MTP=1
export VLLM_GEMMA4_ADAPTIVE_MTP_LONG_CONTEXT_THRESHOLD=4096
export VLLM_GEMMA4_ADAPTIVE_MTP_LONG_CONTEXT_DEPTH=2
export VLLM_GEMMA4_ADAPTIVE_MTP_HIGH_BATCH_THRESHOLD=64
export VLLM_GEMMA4_ADAPTIVE_MTP_HIGH_BATCH_DEPTH=4
```
