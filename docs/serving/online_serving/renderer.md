# Renderer APIs

The renderer API disaggregates preprocessing and enables a token-in/token-out
API server.

- GPU-less deployment of frontend: Allow preprocessing (tokenization, MM input processing) and postprocessing (detokenization, tool call parsing, reasoning parsing) to run without GPU.
- Disaggregated tokenization: Support use cases such as llm-d, Dynamo, and custom frontends that need to leverage vLLM's preprocessing logic without running the full inference engine.
- Tokens-in / tokens-out engine: Make the engine a pure token-in / token-out service, decoupled from request preprocessing.

The standalone render endpoints are transport-pure. They reject
`kv_transfer_params` and beam search rather than creating or transferring a
remote KV offer during CPU-only preprocessing. A disaggregation coordinator
renders first, requires a singleton GenerateRequest, then attaches one
top-level KV contract before sending it to the inference engine.

Completion derender currently warns and drops `kv_transfer_params` when
GenerateResponses disagree. That behavior is not a fail-closed transport
contract and is a release blocker for the active Gemma 4 candidate. Supported
KV flows must reject disagreement rather than silently removing ownership
metadata.

## API Reference

- [Completions Render API](renderer.md) (`/v1/completions/render`)
    - Render completion requests
- [Chat Completions Render API](renderer.md) (`/v1/chat/completions/render`)
    - Render chat completions
