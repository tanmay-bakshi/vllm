# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma4 FlashInfer TRTLLM-GEN attention backend."""

import os

import torch

from vllm import envs
from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.platforms.interface import DeviceCapability
from vllm.utils.flashinfer import can_use_trtllm_attention
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.flashinfer import (
    FP8_DTYPE,
    FIDecode,
    FIPrefill,
    FlashInferBackend,
    FlashInferImpl,
    FlashInferMetadata,
    FlashInferMetadataBuilder,
    TRTLLMDecode,
    TRTLLMPrefill,
    _get_trtllm_gen_workspace_buffer,
)
from vllm.v1.kv_cache_interface import AttentionSpec, UniformTypeKVCacheSpecs

logger = init_logger(__name__)

_FP8_CACHE_DTYPES = {"fp8", "fp8_e4m3"}


def _is_fp8_cache_dtype(kv_cache_dtype: CacheDType | str | None) -> bool:
    """Return whether a cache dtype is an E4M3 FP8 cache.

    :param kv_cache_dtype: KV-cache dtype value from configuration or metadata.
    :returns: ``True`` when the dtype is compatible with the Gemma4 TRTLLM path.
    """

    return kv_cache_dtype in _FP8_CACHE_DTYPES


def _has_nonempty_mm_prefix_range(attn_metadata: object) -> bool:
    """Return whether metadata carries real multimodal prefix ranges.

    :param attn_metadata: Attention metadata object.
    :returns: ``True`` when at least one request has a non-empty MM prefix range.
    """

    mm_prefix_range = getattr(attn_metadata, "mm_prefix_range", None)
    if mm_prefix_range is None:
        return False
    if isinstance(mm_prefix_range, dict) is False:
        raise TypeError(
            "Gemma4 TRTLLM-GEN expected mm_prefix_range to be a dictionary."
        )
    for ranges in mm_prefix_range.values():
        if len(ranges) > 0:
            return True
    return False


class Gemma4FlashInferTRTLLMGenBackend(FlashInferBackend):
    """Gemma4-specific FlashInfer backend that requires TRTLLM-GEN kernels."""

    @staticmethod
    def get_name() -> str:
        """Return the backend registry name.

        :returns: Attention backend enum name.
        """

        return "FLASHINFER_GEMMA4_TRTLLM_GEN"

    @staticmethod
    def get_impl_cls() -> type["Gemma4FlashInferTRTLLMGenImpl"]:
        """Return the attention implementation class.

        :returns: Gemma4 TRTLLM-GEN attention implementation class.
        """

        return Gemma4FlashInferTRTLLMGenImpl

    @staticmethod
    def get_builder_cls() -> type["Gemma4FlashInferTRTLLMGenMetadataBuilder"]:
        """Return the metadata builder class.

        :returns: Gemma4 TRTLLM-GEN metadata builder class.
        """

        return Gemma4FlashInferTRTLLMGenMetadataBuilder

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        """Allow Gemma4 text-only requests on multimodal-prefix models.

        :returns: ``True`` so model-level Gemma4 MM-prefix validation can pass.
        """

        return True

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        """Return an incompatibility reason for non-Gemma4 fast-path inputs.

        :param head_size: Per-head attention dimension.
        :param dtype: Model activation dtype.
        :param kv_cache_dtype: Logical KV-cache dtype.
        :param block_size: User-selected block size, when provided.
        :param use_mla: Whether the selector is looking for an MLA backend.
        :param has_sink: Whether attention sinks are configured.
        :param use_sparse: Whether the selector is looking for sparse attention.
        :param use_mm_prefix: Whether the selector requires multimodal-prefix support.
        :param device_capability: CUDA device capability.
        :returns: ``None`` when the combination is supported, otherwise a reason.
        """

        reason = super().supports_combination(
            head_size=head_size,
            dtype=dtype,
            kv_cache_dtype=kv_cache_dtype,
            block_size=block_size,
            use_mla=use_mla,
            has_sink=has_sink,
            use_sparse=use_sparse,
            use_mm_prefix=use_mm_prefix,
            device_capability=device_capability,
        )
        if reason is not None:
            return reason
        if device_capability.major != 10:
            return "Gemma4 TRTLLM-GEN requires a Blackwell SM100-family GPU"
        if _is_fp8_cache_dtype(kv_cache_dtype) is False and kv_cache_dtype != "bfloat16":
            return "Gemma4 TRTLLM-GEN requires an FP8 E4M3 or explicit bf16 KV cache"
        return None


class Gemma4FlashInferTRTLLMGenMetadataBuilder(FlashInferMetadataBuilder):
    """Metadata builder that admits only all-TRTLLM Gemma4 attention batches."""

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        """Initialize the all-TRTLLM metadata builder.

        :param kv_cache_spec: Attention KV-cache specification for this group.
        :param layer_names: Attention layer names sharing this builder.
        :param vllm_config: Active vLLM configuration.
        :param device: Device used for persistent metadata buffers.
        """

        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._validate_group_contract()
        if self.device.type == "cuda":
            workspace = _get_trtllm_gen_workspace_buffer()
            logger.info_once(
                "Gemma4 TRTLLM-GEN workspace is allocated with %d bytes.",
                workspace.numel() * workspace.element_size(),
            )
        logger.info_once(
            "Gemma4 TRTLLM-GEN attention group: layers=%d, head_dim=%d, "
            "query_heads=%d, kv_heads=%d, window_left=%d, page_size=%d, "
            "cache_dtype=%s, query_dtype=%s.",
            len(layer_names),
            self.head_dim,
            self.num_qo_heads,
            self.num_kv_heads,
            self.window_left,
            self.page_size,
            self.cache_dtype,
            self.q_data_type,
        )

    @classmethod
    def get_cudagraph_support(
        cls: type["Gemma4FlashInferTRTLLMGenMetadataBuilder"],
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        """Return CUDA graph support for all-TRTLLM metadata.

        :param vllm_config: Active vLLM configuration.
        :param kv_cache_spec: Attention KV-cache specification.
        :returns: Uniform batch CUDA graph support when every spec is compatible.
        """

        return super().get_cudagraph_support(vllm_config, kv_cache_spec)

    def _iter_attention_specs(self) -> list[AttentionSpec]:
        """Return all attention specs represented by this builder.

        :returns: Concrete attention specs.
        """

        if isinstance(self.kv_cache_spec, UniformTypeKVCacheSpecs):
            specs = [
                spec
                for spec in self.kv_cache_spec.kv_cache_specs.values()
                if isinstance(spec, AttentionSpec)
            ]
            if len(specs) == 0:
                raise ValueError("Gemma4 TRTLLM-GEN requires attention KV specs.")
            return specs
        if isinstance(self.kv_cache_spec, AttentionSpec):
            return [self.kv_cache_spec]
        raise TypeError(
            "Gemma4 TRTLLM-GEN requires AttentionSpec, got "
            f"{type(self.kv_cache_spec).__name__}."
        )

    def _validate_group_contract(self) -> None:
        """Validate the source-level assumptions for the optimized path."""

        if envs.VLLM_BATCH_INVARIANT:
            raise ValueError("Gemma4 TRTLLM-GEN does not support batch-invariant mode.")
        if self.use_dcp:
            raise ValueError("Gemma4 TRTLLM-GEN does not support DCP.")
        if _is_fp8_cache_dtype(self.cache_dtype):
            # ModelOpt FP8 KV serving: queries are FP8-quantized upstream.
            if self.q_data_type != FP8_DTYPE:
                raise ValueError(
                    "Gemma4 TRTLLM-GEN requires FP8 query quantization. "
                    "Set disable_flashinfer_q_quantization=False."
                )
        elif self.cache_dtype == "auto":
            # Explicit bf16 KV serving (the parent builder normalizes
            # non-quantized cache dtypes to "auto" with the spec dtype
            # equal to the model dtype); queries stay in the model dtype.
            if self.q_data_type != self.kv_cache_spec.dtype:
                raise ValueError(
                    "Gemma4 TRTLLM-GEN bf16 KV serving requires model-dtype "
                    f"queries, got {self.q_data_type}."
                )
        else:
            raise ValueError(
                "Gemma4 TRTLLM-GEN requires an FP8 E4M3 or bf16 KV cache, got "
                f"{self.cache_dtype}."
            )
        self._expected_q_dtype = (
            FP8_DTYPE
            if _is_fp8_cache_dtype(self.cache_dtype)
            else self.kv_cache_spec.dtype
        )
        if can_use_trtllm_attention(self.num_qo_heads, self.num_kv_heads) is False:
            raise ValueError(
                "Gemma4 TRTLLM-GEN requires TRTLLM attention support and "
                "query heads divisible by KV heads."
            )
        for spec in self._iter_attention_specs():
            if spec.num_kv_heads != self.num_kv_heads:
                raise ValueError(
                    "Gemma4 TRTLLM-GEN metadata group has inconsistent KV heads: "
                    f"{spec.num_kv_heads} != {self.num_kv_heads}."
                )
            if spec.head_size != self.head_dim:
                raise ValueError(
                    "Gemma4 TRTLLM-GEN metadata group has inconsistent head dims: "
                    f"{spec.head_size} != {self.head_dim}."
                )

    def _validate_metadata(self, metadata: FlashInferMetadata) -> FlashInferMetadata:
        """Reject metadata that would leave the TRTLLM-GEN fast path.

        :param metadata: Built FlashInfer metadata.
        :returns: The validated metadata.
        """

        if metadata.use_cascade:
            raise ValueError("Gemma4 TRTLLM-GEN does not support cascade metadata.")
        if isinstance(metadata.prefill, (FIPrefill, FIDecode)):
            raise TypeError("Gemma4 TRTLLM-GEN rejected native FlashInfer metadata.")
        if isinstance(metadata.decode, (FIPrefill, FIDecode)):
            raise TypeError("Gemma4 TRTLLM-GEN rejected native FlashInfer metadata.")
        if metadata.prefill is not None and not isinstance(
            metadata.prefill, TRTLLMPrefill
        ):
            raise TypeError(
                "Gemma4 TRTLLM-GEN prefill metadata must be TRTLLMPrefill."
            )
        if metadata.decode is not None and not isinstance(metadata.decode, TRTLLMDecode):
            raise TypeError("Gemma4 TRTLLM-GEN decode metadata must be TRTLLMDecode.")
        if metadata.q_data_type != self._expected_q_dtype:
            raise TypeError(
                "Gemma4 TRTLLM-GEN metadata must use "
                f"{self._expected_q_dtype} queries, got {metadata.q_data_type}."
            )
        return metadata

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlashInferMetadata:
        """Build all-TRTLLM metadata for a Gemma4 attention group.

        :param common_prefix_len: Common prefix length for cascade attention.
        :param common_attn_metadata: Common scheduler attention metadata.
        :param fast_build: Whether to favor metadata build speed.
        :returns: Validated FlashInfer metadata.
        """

        metadata = super().build(
            common_prefix_len=common_prefix_len,
            common_attn_metadata=common_attn_metadata,
            fast_build=fast_build,
        )
        return self._validate_metadata(metadata)

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> FlashInferMetadata:
        """Build capture metadata after the TRTLLM workspace exists.

        :param common_attn_metadata: Common scheduler attention metadata.
        :returns: Validated FlashInfer metadata.
        """

        if self.device.type == "cuda":
            _get_trtllm_gen_workspace_buffer()
        metadata = super().build_for_cudagraph_capture(common_attn_metadata)
        return self._validate_metadata(metadata)


class Gemma4FlashInferTRTLLMGenImpl(FlashInferImpl):
    """FlashInfer implementation wrapper that rejects non-TRTLLM fallbacks."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        *args: object,
        **kwargs: object,
    ) -> None:
        """Initialize the implementation and validate per-layer GQA.

        :param num_heads: Per-rank query head count.
        :param head_size: Attention head dimension.
        :param scale: Attention scale.
        :param num_kv_heads: Per-rank KV head count.
        :param args: Remaining ``FlashInferImpl`` constructor arguments.
        :param kwargs: Remaining ``FlashInferImpl`` constructor keyword arguments.
        """

        if num_heads % num_kv_heads != 0:
            raise ValueError(
                "Gemma4 TRTLLM-GEN requires query heads divisible by KV heads, "
                f"got {num_heads=} and {num_kv_heads=}."
            )
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            *args,
            **kwargs,
        )
        if self.support_trtllm_attn is False:
            raise ValueError(
                "Gemma4 TRTLLM-GEN requires FlashInfer TRTLLM attention support."
            )
        # FP8 KV serving quantizes queries upstream; bf16 KV serving keeps
        # queries in the model dtype (no quant-query support needed).
        self._quantized_kv = is_quantized_kv_cache(self.kv_cache_dtype)
        if self._quantized_kv and self.supports_quant_query_input is False:
            raise ValueError(
                "Gemma4 TRTLLM-GEN requires quantized query input support."
            )
        self._metadata_trace_enabled: bool = (
            os.environ.get("VLLM_GEMMA4_TRTLLM_GEN_METADATA_TRACE", "0") == "1"
        )
        self._metadata_trace_prefill_logged: bool = False
        self._metadata_trace_decode_logged: bool = False

    def _log_metadata_trace(self, attn_metadata: FlashInferMetadata) -> None:
        """Log one-shot metadata proof for debug runs.

        :param attn_metadata: Validated FlashInfer attention metadata.
        """

        if self._metadata_trace_enabled is False:
            return
        if (
            self._metadata_trace_prefill_logged is False
            and attn_metadata.prefill is not None
        ):
            self._metadata_trace_prefill_logged = True
            logger.info(
                "Gemma4 TRTLLM-GEN metadata trace prefill=%s decode=%s "
                "q_data_type=%s use_cascade=%s.",
                type(attn_metadata.prefill).__name__,
                type(attn_metadata.decode).__name__
                if attn_metadata.decode is not None
                else "None",
                attn_metadata.q_data_type,
                attn_metadata.use_cascade,
            )
        if (
            self._metadata_trace_decode_logged is False
            and attn_metadata.decode is not None
        ):
            self._metadata_trace_decode_logged = True
            logger.info(
                "Gemma4 TRTLLM-GEN metadata trace prefill=%s decode=%s "
                "q_data_type=%s use_cascade=%s.",
                type(attn_metadata.prefill).__name__
                if attn_metadata.prefill is not None
                else "None",
                type(attn_metadata.decode).__name__,
                attn_metadata.q_data_type,
                attn_metadata.use_cascade,
            )

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashInferMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run attention after verifying that metadata stays on TRTLLM-GEN.

        :param layer: vLLM attention layer wrapper.
        :param query: Query tensor.
        :param key: Key tensor.
        :param value: Value tensor.
        :param kv_cache: KV-cache tensor.
        :param attn_metadata: FlashInfer metadata.
        :param output: Output tensor.
        :param output_scale: Optional fused output quantization scale.
        :param output_block_scale: Optional fused output block scale.
        :returns: Attention output tensor.
        """

        if kv_cache.dim() == 5 and kv_cache.size(1) == 1:
            # F2b single-plane global cache: the stock path can neither
            # read nor write it (no V plane exists). Only boot-time
            # dummy / capture-warmup traffic may land here -- router
            # admission plus the mega chain own every real decode, and
            # a rogue prefill would only leave its OWN pages unwritten
            # (garbage for that request; no cross-request corruption).
            # Zero the output, count loudly, touch nothing.
            if attn_metadata is not None:
                self._sp_stock_hits = getattr(self, "_sp_stock_hits", 0) + 1
                n = self._sp_stock_hits
                if n <= 8 or n % 1000 == 0:
                    logger.warning(
                        "Gemma4 single-plane layer served by STOCK "
                        "attention (#%d): expected only for boot "
                        "dummies; real traffic here means broken "
                        "admission.",
                        n,
                    )
            output.zero_()
            return output

        if attn_metadata is not None:
            if attn_metadata.use_cascade:
                raise ValueError(
                    "Gemma4 TRTLLM-GEN rejected cascade attention metadata."
                )
            if attn_metadata.prefill is not None and not isinstance(
                attn_metadata.prefill, TRTLLMPrefill
            ):
                raise TypeError(
                    "Gemma4 TRTLLM-GEN prefill metadata must be TRTLLMPrefill."
                )
            if attn_metadata.decode is not None and not isinstance(
                attn_metadata.decode, TRTLLMDecode
            ):
                raise TypeError(
                    "Gemma4 TRTLLM-GEN decode metadata must be TRTLLMDecode."
                )
            if self._quantized_kv:
                if attn_metadata.q_data_type != FP8_DTYPE:
                    raise TypeError("Gemma4 TRTLLM-GEN requires FP8 query metadata.")
            elif attn_metadata.q_data_type == FP8_DTYPE:
                # bf16 KV serving: the parent forward asserts the metadata
                # q dtype matches the (model-dtype) query tensor.
                raise TypeError(
                    "Gemma4 TRTLLM-GEN bf16 KV serving must not quantize queries."
                )
            if _has_nonempty_mm_prefix_range(attn_metadata):
                raise NotImplementedError(
                    "Gemma4 TRTLLM-GEN currently supports text-only requests. "
                    "Multimodal prefix ranges require a FlashInfer masking path."
                )
            self._log_metadata_trace(attn_metadata)
        return super().forward(
            layer=layer,
            query=query,
            key=key,
            value=value,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            output=output,
            output_scale=output_scale,
            output_block_scale=output_block_scale,
        )
