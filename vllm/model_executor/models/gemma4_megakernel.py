"""Env-gated megakernel decode path for Gemma4 (GPU-local, TP1).

Enable with VLLM_GEMMA4_MEGAKERNEL=1 (hook at the bottom of gemma4.py;
with the flag unset that hook is a single env check at import and this
module is never imported). Kernels + weight-prep recipes are imported
from the megakernel workspace (VLLM_GEMMA4_MEGAKERNEL_PATH).

Knobs: MK_MSET "q:r,..." = (tokens-per-request, requests) batch shapes
to compile+capture (default "1:1,16:1"; each pair costs ~35-70s prep);
MK_PDL=1 enables PDL launches (measured ~nil under graphs, default 0);
MK_NO_CAPTURE=1 forces eager chains (debug).

Validated (SESSION_FINDINGS B4/B5): logprob parity at the stock
reproducibility envelope, DFlash acceptance through graph replays,
TPOT C1 TP1 1.449ms w/ DFlash (3.05x stock same-workload); M in
{1..64} kernel checks; C>1 via per-row block tables.

install() wraps Gemma4Model.__call__ (the support_torch_compile
dispatch) with a gated runner; it coexists with every
CompilationMode, including VLLM_COMPILE + FULL_AND_PIECEWISE, so the
stock path above the mega set keeps its inductor fusions and
piecewise graphs (q=1 bootstrap steps included). Pure-decode
uniform-batch steps of a prepared shape run every layer as
  B3 (post-ff norm+residual+ls -> input norm -> qkv FP8 -> head norms
      + RoPE -> q fp8 + paged KV append)
  -> trtllm-gen decode core (same call/metadata as the deployed backend)
  -> B2 (attn-out fp8 -> o_proj FP8 -> post-attn norm+residual -> pre-ff
      norm + fp4 quant -> full NVFP4 MLP)
= 3 launches/layer; anything else falls through to the stock path.

Numerics = the B4-validated contract: plain-w norms, layer_scalar in
scales[3] (layer i carries ls_{i-1}; the tail applies ls_59), q/k/v
scales 1.0, checkpoint-exact fp4 bytes + static input_scale alphas,
attn projections online per-channel FP8. Weight prep reads the raw
safetensors via mkbench.real_weights.load_layer (proven recipes).

Deliberate Phase-1 compromises (flagged for Phase 2):
  - eager only (no cudagraph/PDL), python launch overhead per kernel;
  - out->mo copy per layer bridges B2's padded-N2 rows into B3's
    contiguous mo reader (~2us/layer; stride param later);
  - m_real compiled for {1} (plain decode); DFlash adds {16} next.
"""
import os
import sys

import torch

MK_ROOT = os.environ.get(
    "VLLM_GEMMA4_MEGAKERNEL_PATH",
    "/data/gemma4-2026-06-optimization-effort/megakernel")
if MK_ROOT not in sys.path:
    sys.path.insert(0, MK_ROOT)

HIDDEN, INTER = 5376, 21504
N2 = 5632                      # down/o_proj padded N (tile 128 x cga 4)
MAXM = 128
EPS = 1e-6
# compile-time KV extent: a fixed upper bound instead of the live
# cache's numel. It is only a flat layout extent (addressing is
# page-driven, no masking against it), so the bound makes kernel
# compiles independent of boot state: they can start during memory
# profiling and survive KV rebinds without recompiling.
KV_NUMEL_BOUND = 1 << 38


def _log(msg):
    print(f"[mk_gemma4] {msg}", flush=True)


def _install_dsl_compile_lock():
    """Serialize ALL cute-DSL compiles in this process. The DSL
    front-end is thread-hostile (two concurrent compiles abort with
    nanobind 'No current Location'), and flashinfer's NVFP4 kernels
    compile through the same singleton, so the async prep thread must
    never overlap them. BaseDSL._func is the shared entry for both
    cute.compile and @cute.jit direct calls; it re-enters itself
    during tracing, hence the RLock."""
    import threading
    from cutlass.base_dsl.dsl import BaseDSL
    if getattr(BaseDSL, "_mk_compile_lock_installed", False):
        return
    lock = threading.RLock()
    orig = BaseDSL._func

    def locked(self, *args, **kwargs):
        with lock:
            return orig(self, *args, **kwargs)

    BaseDSL._func = locked
    BaseDSL._mk_compile_lock_installed = True
    _log("cute-DSL compile lock installed (process-wide)")


def _compile_shapes(m_set, facts, pdl, tag):
    """Compile B3+B2 for every (q, r) x flavor. facts: flavor -> dict
    of ints (n_qkv, heads, KV, hd, q_size, ps, bt_len, cs_numel); all
    boot-invariant, so async-compiled kernels stay valid across KV
    rebinds. Every cute.compile goes through the process-wide lock."""
    import time
    from mk_fused.fused_preattn_sm100 import Sm100PreAttnKernel
    from mk_fused.fused_postattn_sm100 import Sm100PostAttnKernel
    from mkbench.cutedsl_driver import (compile_fused_preattn,
                                        compile_fused_postattn)
    t0 = time.time()
    k3, k2, k2_by_m = {}, {}, {}
    for (q, r) in m_set:
        M = q * r
        mp = max(M, 8)
        for f, c in facts.items():
            if f == "global" and "MK_KSPLIT_Q" not in os.environ:
                os.environ["MK_KSPLIT_Q"] = "2"
            k = Sm100PreAttnKernel(
                mp, M, c["n_qkv"], HIDDEN, c["heads"], c["KV"],
                c["hd"], kv_numel=KV_NUMEL_BOUND,
                cs_numel=c["cs_numel"], bt_len=c["bt_len"],
                page_size=c["ps"], bt_stride=c["bt_len"], q_len=q,
                mo_stride=N2,
                cluster_shape_mn=(1, 4), enable_pdl=pdl,
                kv_bf16=c.get("kv_bf16", False))
            k3[(f, q, r)] = compile_fused_preattn(k)
            if f == "global":
                os.environ.pop("MK_KSPLIT_Q", None)
            if (f, M) not in k2_by_m:
                k = Sm100PostAttnKernel(
                    mp, M, 2 * INTER, HIDDEN, N2, INTER,
                    c["q_size"], mma_tiler_mn=(128, 128),
                    cluster_shape_mn=(1, 4), enable_pdl=pdl)
                k2_by_m[(f, M)] = compile_fused_postattn(k)
            k2[(f, q, r)] = k2_by_m[(f, M)]
            _log(f"[{tag}] compiled B3+B2 {f} q={q} r={r} M={M} "
                 f"({time.time() - t0:.0f}s)")
    return k3, k2


def _kv5(kvc: torch.Tensor, KV: int, hd: int) -> torch.Tensor:
    """Canonical contiguous HND view (nb, 2, KV, ps, hd) of a layer's
    registered KV cache, whatever view shape it was registered with.
    Physical storage is contiguous with pages outermost (layout=HND
    asserted by the deployed backend); sorting dims by stride recovers
    the contiguous base for a clean reshape. bf16 caches (all-bf16 KV
    serving) are element-typed already; fp8 registers as fp8/uint8."""
    v = kvc if kvc.dtype == torch.bfloat16 else kvc.view(torch.float8_e4m3fn)
    order = sorted(range(v.dim()), key=lambda d: -v.stride(d))
    base = v.permute(order)
    assert base.is_contiguous(), (kvc.shape, kvc.stride())
    ps = v.stride(0) // (2 * KV * hd)
    nb = v.numel() // (2 * KV * ps * hd)
    return base.reshape(nb, 2, KV, ps, hd)


class _LP:
    """Per-layer prepared state (buffers referenced by raw pointers in
    the arg lists MUST be kept alive here)."""
    __slots__ = ("flavor", "lname", "b3_args", "b2_args", "bt_idx",
                 "fl3_idx", "fl2_idx",
                 "kvc_hnd", "window_left", "hd", "q_size", "keep")


class MegaRunner:
    def __init__(self):
        self.enabled = False
        self.prepared = False
        self.prep_failed = False
        self.m_set = ((1, 1), (16, 1))   # (q_len, R) pairs
        self.steps = 0
        self.kv_map = None         # layer_name -> live cache tensor
        self.fb = {"prefill": 0, "multi_seq": 0, "m": 0}  # fallbacks
        # cudagraph state: one graph per m, captured after 2 eager warm
        # runs; metadata-pointer guards (per distinct KV group) trigger
        # eager fallback + recapture if vLLM rebinds its buffers
        self.graphs = {}
        self.g_guards = {}
        self.eager_runs = {}
        self.graph_steps = 0
        self.eager_steps = 0
        self.no_capture = os.environ.get("MK_NO_CAPTURE", "0") == "1"
        # native mode: vLLM's FULL_DECODE_ONLY cudagraphs capture the
        # chain; the runner never builds its own graphs or guards. Run
        # the engine WITHOUT enforce-eager and with
        # -cc {"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"}.
        self.native = os.environ.get("MK_NATIVE", "0") == "1"
        self.native_captured = set()
        # bisect knob: in native mode, capture-pass sizes with q*r >
        # bake_max are baked STOCK while their warmups still run prep +
        # the eager chain (plus one stock pass so the stock kernels for
        # that size stay warmed). Separates "prep/eager ran at big M"
        # from "a mega graph was CAPTURED at big M". 0 = off.
        self.bake_max = int(os.environ.get("MK_BAKE_MAX", "0"))
        # log allocator deltas for the chain inside each capture region
        self.cap_debug = os.environ.get("MK_CAP_DEBUG", "0") == "1"
        # forensics: redzone guards between flags tensors + checksum log
        # after prep and after each eager chain (first 80 eager steps)
        self.flag_log = os.environ.get("MK_FLAG_LOG", "0") == "1"
        # overlap kernel compiles with engine boot: the profiling-phase
        # gated step starts them on a background thread (cross-boot
        # file caching is structurally unavailable: the DSL front-end
        # is ~80% of compile time and cute.compile hard-disables the
        # jit cache; see SESSION_FINDINGS phase 8)
        self.async_compile = os.environ.get("MK_ASYNC_COMPILE", "1") == "1"
        self._compile_thread = None
        self._compile_err = False
        self._pre = None
        self._last_mds = None
        self._last_m = 0
        ms = os.environ.get("MK_MSET")
        if ms:
            self.m_set = tuple(
                (int(p.split(":")[0]), int(p.split(":")[1]))
                for p in ms.split(","))
        self.pdl = os.environ.get("MK_PDL", "0") == "1"
        # periodic counter log for serve-mode observability (0 = off)
        self.log_every = int(os.environ.get("MK_LOG_EVERY", "0"))

    # ---------- one-shot KV tensor harvest ----------
    # Direct attribute reads at gate time. The old harvest fired
    # forward_pre_hooks during one sacrificial stock step; under
    # VLLM_COMPILE the stock step runs guard-dropped compiled code
    # where dynamically-added module hooks never fire, so hooks are
    # structurally unusable here. attn.kv_cache[ve] is bound by
    # bind_kv_cache before any decode step, and validity is enforced
    # by _kv_ok on every gated step exactly as before (stub detection
    # + data_ptr movement), so a stale or early read self-heals: stay
    # stock this step, re-read on the next.
    def _harvest_now(self, model):
        kv_map = {}
        for layer in model.layers:
            attn = layer.self_attn.attn
            kvc = attn.kv_cache[0] if isinstance(
                attn.kv_cache, (list, tuple)) else attn.kv_cache
            if kvc is None or kvc.numel() == 0:
                return False   # not bound yet; stay stock
            kv_map[attn.layer_name] = kvc
        self.kv_map = kv_map
        kvc = kv_map[model.layers[0].self_attn.attn.layer_name]
        _log(f"kv harvest (direct): {len(kv_map)} layers, layer0 "
             f"{tuple(kvc.shape)} stride={tuple(kvc.stride())}")
        return True

    # ---------- async kernel compiles ----------
    def _compile_facts(self, model, md_all):
        """Flavor -> compile-relevant integers. Everything here is
        boot-invariant (the KV extent is KV_NUMEL_BOUND), so facts read
        from the memory-profiling phase are valid for the real engine."""
        facts = {}
        for layer in model.layers:
            f = "global" if layer.is_full_attention else "local"
            if f in facts:
                continue
            a = layer.self_attn
            kvc = self.kv_map[a.attn.layer_name]
            KV = a.attn.impl.num_kv_heads
            hd = a.attn.impl.head_size
            facts[f] = dict(
                KV=KV, hd=hd, ps=_kv5(kvc, KV, hd).shape[3],
                heads=a.attn.impl.num_heads,
                q_size=a.attn.impl.num_heads * hd,
                n_qkv=(a.attn.impl.num_heads + 2 * KV) * hd,
                bt_len=int(md_all[a.attn.layer_name].decode
                           .block_tables.shape[1]),
                cs_numel=a.rotary_emb.cos_sin_cache.numel(),
                kv_bf16=(kvc.dtype == torch.bfloat16),
            )
        return facts

    def _start_async_compiles(self, model, md_all):
        if (not self.async_compile or self._compile_thread is not None
                or self.prepared or self.kv_map is None):
            return
        import threading
        try:
            _install_dsl_compile_lock()
            facts = self._compile_facts(model, md_all)
        except Exception:
            import traceback
            traceback.print_exc()
            return
        m_set, pdl = self.m_set, self.pdl

        def work():
            try:
                # Bind the CUDA primary context to this fresh thread
                # BEFORE any DSL work: the occupancy probe
                # (flashinfer get_max_active_clusters) uses the driver
                # API, and on a context-less thread it silently falls
                # back to sm_count -- baking an oversubscribed
                # persistent grid whose progressive-release gates
                # deadlock the first launch (100% GPU spin at the
                # first mega warmup; found on the first mode-3 boot,
                # where this trigger path first ran compiles off the
                # main thread).
                torch.cuda.set_device(0)
                torch.cuda.current_stream().synchronize()
                self._pre = _compile_shapes(m_set, facts, pdl, "async")
                _log("async compiles done")
            except Exception:
                self._compile_err = True
                import traceback
                traceback.print_exc()

        self._compile_thread = threading.Thread(
            target=work, daemon=True, name="mk-compile")
        self._compile_thread.start()
        _log(f"async kernel compiles started ({len(m_set)} shapes x "
             f"{len(facts)} flavors)")

    def _join_async_compiles(self):
        t = self._compile_thread
        if t is None:
            return None
        if t.is_alive():
            _log("waiting for async compiles...")
        t.join()
        return None if self._compile_err else self._pre

    # ---------- KV-binding guards ----------
    # vLLM's cudagraph memory profiling (default since v0.21) runs
    # warmup+capture dummy passes against a TEMPORARY profiling KV
    # cache before the real one exists, and it profiles the two
    # LARGEST capture sizes. Any m_set covering those shapes made the
    # runner harvest/prep there: baked kv pointers (and the
    # compile-time kv_numel) referenced a cache freed right after
    # profiling, so every mega launch afterwards read and wrote
    # through dangling pointers. That was the whole "M>=96 capture
    # poison". Guard on both sides: never prep on a stub, and reset
    # if the live binding ever moves out from under an existing prep
    # (sleep/wake, elastic re-init, future ordering changes).
    def _kv_ok(self, model, md_all):
        a = model.layers[0].self_attn.attn
        kvc = self.kv_map.get(a.layer_name)
        if kvc is None:
            return False
        live = a.kv_cache[0] if isinstance(
            a.kv_cache, (list, tuple)) else a.kv_cache
        if live.data_ptr() != kvc.data_ptr():
            _log("kv binding moved from under the harvest/prep; "
                 "resetting")
            return False
        if not self.prepared:
            # stub check (once, pre-prep): a real cache must hold at
            # least one max-len request (pages >= block-table length)
            nb = _kv5(kvc, a.impl.num_kv_heads,
                      a.impl.head_size).shape[0]
            bt_len = int(md_all[a.layer_name].decode
                         .block_tables.shape[1])
            if nb < bt_len:
                _log(f"kv cache is a profiling stub ({nb} pages < "
                     f"bt_len {bt_len}); staying stock until the "
                     "real cache is bound")
                return False
        return True

    def _reset_prep(self):
        self.kv_map = None
        if self.prepared:
            self.prepared = False
            self.native_captured.clear()
            self.graphs.clear()
            self.g_guards.clear()
            self.eager_runs.clear()
            for attr in ("lps", "k3", "k2", "cs_keep", "ws", "flags3",
                         "flags2", "flag_guards", "aux_buf"):
                if hasattr(self, attr):
                    delattr(self, attr)
            torch.cuda.empty_cache()
            _log("released stale prep state; will re-harvest + re-prep")

    # ---------- gate ----------
    def _gate(self, model, intermediate_tensors):
        """(md_all, (q, r)) for a pure uniform-batch decode step of ANY
        shape, else None. m_set eligibility is step()'s check, after
        the KV guards: the async-compile trigger must fire on stub-KV
        steps of any uniform shape (with capture-size lists whose
        largest sizes exceed the mega set, the memory-profiling
        dummies are the only pre-real-KV uniform steps we ever see)."""
        if intermediate_tensors is not None:
            return None
        from vllm.forward_context import get_forward_context
        ctx = get_forward_context()
        md_all = ctx.attn_metadata
        if not isinstance(md_all, dict):
            return None
        lname0 = model.layers[0].self_attn.attn.layer_name
        md0 = md_all.get(lname0)
        if md0 is None or getattr(md0, "num_prefill_tokens", 1) != 0:
            self.fb["prefill"] += 1
            return None
        t = int(md0.num_decode_tokens)
        r = int(getattr(md0, "num_decodes", 0))
        if r < 1 or t % max(r, 1) != 0:
            self.fb["multi_seq"] += 1
            return None
        return md_all, (t // r, r)

    # ---------- per-step decision tree ----------
    def step(self, model, input_ids, positions, inputs_embeds, md_all,
             key):
        """One gated uniform-decode step. Returns the model output if
        the mega chain ran, None to fall through to the stock path
        (which under VLLM_COMPILE is the compiled forward; the runner
        never touches it)."""
        capturing = torch.cuda.is_current_stream_capturing()
        if not self.prepared and capturing:
            # never harvest/prep/compile inside a capture; vLLM's
            # warmup passes for each size run first and prep there
            return None
        if self.kv_map is None and not self._harvest_now(model):
            return None
        if not self._kv_ok(model, md_all):
            # profiling stub, or the engine re-bound its KV cache:
            # start the kernel compiles in the background (compile
            # facts from this metadata are boot-invariant), then
            # forget the harvest and serve stock; a later step
            # re-reads the live tensors
            self._start_async_compiles(model, md_all)
            self._reset_prep()
            return None
        if key not in self.m_set:
            self.fb["m"] += 1
            return None
        over = (self.native and self.bake_max
                and key[0] * key[1] > self.bake_max)
        if over and capturing:
            # bisect knob: bake this size STOCK; its warmups below
            # still ran prep + the eager chain
            return None
        if not self.prepared:
            self.prepare(model)
        out = self.run(model, input_ids, positions, inputs_embeds,
                       md_all, key)
        # bake_max warmups keep the stock kernels for this size warmed
        # too: discard the chain's output and let the caller run stock
        return None if over else out

    # ---------- weight/kernel prep (once, lazy) ----------
    def prepare(self, model):
        import time
        t0 = time.time()
        from mkbench.real_weights import load_layer, _interleave_rows
        from impls.fused_postattn import block_b
        from vllm import _custom_ops as ops
        from vllm.model_executor.layers.quantization.utils.nvfp4_utils \
            import swizzle_blockscale
        from vllm.forward_context import get_forward_context

        dev = "cuda"
        layers = list(model.layers)
        assert layers[0].per_layer_input_gate is None, "PLE unsupported"
        md_all = get_forward_context().attn_metadata

        # ---- flavor constants from live modules ----
        def flavor_of(layer):
            return "global" if layer.is_full_attention else "local"

        fl_first = {}
        for i, l in enumerate(layers):
            fl_first.setdefault(flavor_of(l), i)
        fc = {}
        for fl, i0 in fl_first.items():
            a = layers[i0].self_attn
            kvc = self.kv_map[a.attn.layer_name]
            KV = a.attn.impl.num_kv_heads
            hd = a.attn.impl.head_size
            v = _kv5(kvc, KV, hd)
            ps = v.shape[3]
            _log(f"kv[{a.attn.layer_name}]: reg {tuple(kvc.shape)} "
                 f"-> hnd {tuple(v.shape)}")
            md = md_all[a.attn.layer_name]
            fc[fl] = dict(
                KV=KV, hd=hd, ps=ps, heads=a.attn.impl.num_heads,
                q_size=a.attn.impl.num_heads * hd,
                n_qkv=(a.attn.impl.num_heads + 2 * KV) * hd,
                kv_numel=kvc.numel(),
                bt_len=int(md.decode.block_tables.shape[1]),
                window_left=int(a.attn.impl.window_left),
                cs=a.rotary_emb.cos_sin_cache.float().contiguous(),
                kv_bf16=(kvc.dtype == torch.bfloat16),
            )
        _log(f"flavors: { {k: {kk: vv for kk, vv in v.items() if kk != 'cs'} for k, v in fc.items()} }")

        # ---- shared scratch (sized for MAXM rows / max shapes) ----
        NQmax = max(v["n_qkv"] for v in fc.values())
        s = self
        s.mo = torch.zeros(MAXM, HIDDEN, dtype=torch.bfloat16, device=dev)
        s.r2 = torch.zeros(MAXM, HIDDEN, dtype=torch.bfloat16, device=dev)
        s.h = torch.zeros(MAXM, HIDDEN, dtype=torch.bfloat16, device=dev)
        s.xq8 = torch.zeros(MAXM, HIDDEN, dtype=torch.uint8, device=dev)
        s.xs = torch.ones(MAXM, dtype=torch.float32, device=dev)
        s.pos_i32 = torch.zeros(MAXM, dtype=torch.int32, device=dev)
        s.qkvacc = {f: torch.zeros(MAXM, fc[f]["n_qkv"],
                                   dtype=torch.bfloat16, device=dev)
                    for f in fc}
        # all-bf16 KV serving stores q/k/v bf16 (2B/elem); the trtllm
        # core then runs its bf16 path. Both flavors share the engine's
        # cache dtype.
        s.kv_bf16 = all(fc[f]["kv_bf16"] for f in fc)
        assert s.kv_bf16 or not any(fc[f]["kv_bf16"] for f in fc), fc
        s.qdt = torch.bfloat16 if s.kv_bf16 else torch.float8_e4m3fn
        s.q8 = {f: torch.zeros(MAXM, fc[f]["q_size"],
                               dtype=torch.bfloat16 if s.kv_bf16
                               else torch.uint8,
                               device=dev) for f in fc}
        s.ao = {f: torch.zeros(MAXM, fc[f]["q_size"], dtype=torch.bfloat16,
                               device=dev) for f in fc}
        s.aq = {f: torch.zeros(MAXM, fc[f]["q_size"], dtype=torch.uint8,
                               device=dev) for f in fc}
        s.as_buf = torch.ones(MAXM, dtype=torch.float32, device=dev)
        s.oacc = torch.zeros(MAXM, N2, dtype=torch.bfloat16, device=dev)
        s.out = torch.zeros(MAXM, N2, dtype=torch.bfloat16, device=dev)
        sf_k1 = (HIDDEN // 16 + 3) // 4
        sf_k2 = (INTER // 16 + 3) // 4
        s.xq = torch.zeros(MAXM, HIDDEN // 2, dtype=torch.uint8, device=dev)
        s.xsf = torch.zeros(32 * 4 * 1 * 4 * sf_k1, dtype=torch.uint8,
                            device=dev)
        s.iq = torch.zeros(MAXM, INTER // 2, dtype=torch.uint8, device=dev)
        s.isf = torch.zeros(32 * 4 * 1 * 4 * sf_k2, dtype=torch.uint8,
                            device=dev)
        ks_o = int(os.environ.get("MK_KSPLIT_O", "3"))
        ks_d = int(os.environ.get("MK_KSPLIT", "2"))
        t2_slots = (N2 // 128) * ks_d
        # The kernels' progressive-release gates poll persistent epoch
        # counters whose arithmetic assumes every launch on a counter set
        # has the same m_real. Sharing a set across compiled shapes
        # deadlocks the first larger-shape launch after smaller-shape
        # traffic (poll target runs ahead of the cumulative release) and
        # opens gates early in the other direction. One set per shape.
        s.flag_guards = []

        def _fl(n):
            t = torch.zeros(n, dtype=torch.int32, device=dev)
            # redzone right after each flags tensor: same segment in the
            # caching allocator with high probability, so an OOB writer
            # runs into it and the flaglog checksum exposes it
            s.flag_guards.append(
                torch.zeros(512, dtype=torch.int32, device=dev))
            return t

        s.flags3 = {(f, q, r): _fl(2048)
                    for f in fc for (q, r) in s.m_set}
        s.flags2 = {(f, q * r): _fl(
                        2 * INTER // 128 + t2_slots + 2 + 2048)
                    for f in fc for (q, r) in s.m_set}
        # trtllm-gen workspace is allocated AFTER cap_max_seq is known
        # (see below): its demand scales with the baked max_seq_len, and
        # an undersized buffer corrupts long-context attention SILENTLY
        # before it eventually IMAs at higher concurrency.

        # ---- kernels compile at the END of prepare: weight prep runs
        # first so it overlaps the async compile thread's tail
        for f in fc:
            fc[f]["cs_numel"] = fc[f]["cs"].numel()

        # ---- per-layer weights + arg lists ----
        s.lps = []
        FP8 = torch.float8_e4m3fn
        for i, layer in enumerate(layers):
            lw = load_layer(i)
            f = lw["ltype"]
            assert f == flavor_of(layer), (i, f)
            c = fc[f]
            lp = _LP()
            lp.flavor, lp.hd, lp.q_size = f, c["hd"], c["q_size"]
            lp.lname = layer.self_attn.attn.layer_name
            lp.window_left = c["window_left"]
            kvc = _kv5(self.kv_map[lp.lname], c["KV"], c["hd"])
            lp.kvc_hnd = kvc

            wqkv = torch.cat([lw["wq"], lw["wk"], lw["wv"]], 0).contiguous()
            wq8, wqs = ops.scaled_fp8_quant(wqkv,
                                            use_per_token_if_dynamic=True)
            wq8 = block_b(wq8.view(torch.uint8), c["n_qkv"]).view(FP8)
            wqs = wqs.flatten().float().contiguous()

            wo_pad = torch.zeros(N2, c["q_size"], dtype=torch.bfloat16,
                                 device=dev)
            wo_pad[:HIDDEN] = lw["wo"]
            wo8, wos = ops.scaled_fp8_quant(wo_pad,
                                            use_per_token_if_dynamic=True)
            wos = wos.flatten().float().contiguous()
            wos[HIDDEN:] = 0.0
            wo8 = block_b(wo8.view(torch.uint8), N2).view(FP8)

            gu_q = _interleave_rows(
                torch.cat([lw["g_q"].view(torch.uint8),
                           lw["u_q"].view(torch.uint8)], 0), INTER)
            gu_q = block_b(gu_q, 2 * INTER)
            gu_s = swizzle_blockscale(_interleave_rows(
                torch.cat([lw["g_s"].view(FP8), lw["u_s"].view(FP8)], 0),
                INTER)).contiguous()
            d_q = torch.zeros(N2, INTER // 2, dtype=torch.uint8, device=dev)
            d_q[:HIDDEN] = lw["d_q"].view(torch.uint8)
            d_q = block_b(d_q, N2)
            d_s = torch.zeros(N2, INTER // 16, dtype=FP8, device=dev)
            d_s[:HIDDEN] = lw["d_s"].view(FP8)
            d_s = swizzle_blockscale(d_s).contiguous()

            alpha = torch.tensor(
                [lw["g_ins"] * lw["g_ws2"], lw["d_ins"] * lw["d_ws2"],
                 1.0 / lw["d_ins"], 1.0 / lw["g_ins"]],
                dtype=torch.float32, device=dev)
            ls_prev = 1.0 if i == 0 else float(
                layers[i - 1].layer_scalar.float().item())
            scales = torch.tensor([1.0, 1.0, 1.0, ls_prev],
                                  dtype=torch.float32, device=dev)
            wpf2_prev = (lw["ln_pff"] if i == 0
                         else layers[i - 1].post_feedforward_layernorm
                         .weight.data)
            norms = dict(win=lw["ln_in"].contiguous(),
                         qn=lw["qn"].contiguous(),
                         kn=lw["kn"].contiguous(),
                         wpa=lw["ln_pa"].contiguous(),
                         wpf=lw["ln_pf"].contiguous(),
                         wpf2_prev=wpf2_prev.contiguous())

            lp.b3_args = [
                s.out.data_ptr(), s.r2.data_ptr(),
                norms["wpf2_prev"].data_ptr(), norms["win"].data_ptr(),
                s.h.data_ptr(), s.xq8.data_ptr(), s.xs.data_ptr(),
                wq8.data_ptr(), wqs.data_ptr(),
                s.qkvacc[f].data_ptr(),
                norms["qn"].data_ptr(), norms["kn"].data_ptr(),
                c["cs"].data_ptr(), s.pos_i32.data_ptr(),
                0,  # block table ptr, patched per step (bt_idx)
                s.q8[f].data_ptr(), kvc.data_ptr(),
                scales, 0,  # flags3 ptr, patched per shape (fl3_idx)
            ]
            lp.bt_idx = 14
            lp.fl3_idx = 18
            lp.b2_args = [
                s.ao[f].data_ptr(), s.aq[f].data_ptr(),
                s.as_buf.data_ptr(),
                wo8.data_ptr(), wos.data_ptr(), s.oacc.data_ptr(),
                s.h.data_ptr(), norms["wpa"].data_ptr(), s.r2.data_ptr(),
                norms["wpf"].data_ptr(),
                s.xq.data_ptr(), s.xsf.data_ptr(),
                gu_q.data_ptr(), gu_s.data_ptr(),
                s.iq.data_ptr(), s.isf.data_ptr(),
                d_q.data_ptr(), d_s.data_ptr(),
                s.out.data_ptr(), alpha,
                0,  # flags2 ptr, patched per shape (fl2_idx)
            ]
            lp.fl2_idx = 20
            lp.keep = [wq8, wqs, wo8, wos, gu_q, gu_s, d_q, d_s, alpha,
                       scales, norms]
            s.lps.append(lp)
            del lw, wqkv, wo_pad
            if i % 10 == 9:
                torch.cuda.empty_cache()
                _log(f"prepped layer {i + 1}/60 ({time.time() - t0:.0f}s)")

        last = layers[-1]
        s.w_pff_last = last.post_feedforward_layernorm.weight.data
        s.ls_last = float(last.layer_scalar.float().item())
        s.norm_w_final = model.norm.weight.data
        s.norm_eps_final = float(model.norm.variance_epsilon)

        # graph-mode persistents + DFlash aux tap buffers
        from flashinfer.decode import trtllm_batch_decode_with_kv_cache
        s._decode = trtllm_batch_decode_with_kv_cache
        s._ops = ops
        s.xin = torch.zeros(MAXM, HIDDEN, dtype=torch.bfloat16, device=dev)
        s.xt = torch.zeros(MAXM, HIDDEN, dtype=torch.bfloat16, device=dev)
        s.hid = torch.zeros(MAXM, HIDDEN, dtype=torch.bfloat16, device=dev)
        s.taps = sorted(set(
            getattr(model, "aux_hidden_state_layers", ()) or ()))
        s.tapset = set(s.taps)
        s.aux_buf = {k: torch.zeros(MAXM, HIDDEN, dtype=torch.bfloat16,
                                    device=dev) for k in s.taps}
        s.cap_max_seq = max(c["bt_len"] * c["ps"] for c in fc.values())
        # trtllm-gen workspace: 512MB per 8192 of cap_max_seq, floor
        # 1024MB (deployed reference 413MB@8192; 1024MB at max_seq 65536
        # produced silent long-context garbage then an IMA at C4).
        # MK_WS_MB > 0 overrides.
        ws_mb = int(os.environ.get("MK_WS_MB", "0")) or max(
            1024, s.cap_max_seq // 8192 * 512)
        s.ws = torch.zeros(ws_mb * 1024 * 1024, dtype=torch.uint8,
                           device=dev)
        _log(f"trtllm workspace {ws_mb}MB (cap_max_seq {s.cap_max_seq})")
        # KEEPALIVE for the cos_sin caches: their data_ptrs are baked
        # into every B3 arg list, but fc is a local — without this ref
        # they get freed to the allocator cache and the first
        # torch.cuda.graph.__enter__ (which calls empty_cache) unmaps
        # them out from under the baked pointers. (7-round bisect.)
        s.cs_keep = {f: fc[f]["cs"] for f in fc}
        _log(f"aux taps: {s.taps}; cap_max_seq {s.cap_max_seq}")
        pre = self._join_async_compiles()
        if pre is not None:
            s.k3, s.k2 = pre
            _log(f"kernels: adopted async-compiled set "
                 f"({time.time() - t0:.0f}s into prepare)")
        else:
            _install_dsl_compile_lock()
            s.k3, s.k2 = _compile_shapes(self.m_set, fc, self.pdl, "prep")
        self.prepared = True
        if s.flag_log:
            s._flagdump("post-prep")
        _log(f"prepare done in {time.time() - t0:.0f}s")

    def _flagdump(self, tag):
        s = self
        gsum = sum(int(t.abs().sum()) for t in s.flag_guards)
        f3 = {f"{k[0][0]}{k[1]}:{k[2]}": int(v.abs().sum())
              for k, v in s.flags3.items()}
        f2 = {f"{k[0][0]}{k[1]}": int(v.abs().sum())
              for k, v in s.flags2.items()}
        _log(f"flaglog {tag}: guards={gsum} f3={f3} f2={f2}")

    # ---------- the decode step ----------
    def _chain(self, q, r, mds, max_seq):
        """The capturable decode step: reads xin/pos_i32 + baked
        pointers, writes hid + aux_buf. Allocation- and sync-free.
        q = tokens per request, r = requests, T = q*r total rows."""
        s = self
        m = q * r
        # B3 reads B2's padded out directly (mo_stride=N2); zeroed rows
        # are the layer-0 "mo = 0" entry
        s.out[:m].zero_()
        s.r2[:m].copy_(s.xin[:m])
        if 0 in s.tapset:
            s.aux_buf[0][:m].copy_(s.xin[:m])
        for i, (lp, md) in enumerate(zip(s.lps, mds)):
            args = lp.b3_args
            args[lp.bt_idx] = md.decode.block_tables.data_ptr()
            args[lp.fl3_idx] = s.flags3[(lp.flavor, q, r)].data_ptr()
            s.k3[(lp.flavor, q, r)](*args)
            if i in s.tapset and i > 0:
                s.aux_buf[i][:m].copy_(s.h[:m])
            s._decode(
                query=s.q8[lp.flavor][:m].view(s.qdt)
                .view(m, -1, lp.hd),
                kv_cache=lp.kvc_hnd, workspace_buffer=s.ws,
                block_tables=md.decode.block_tables,
                seq_lens=md.decode.seq_lens,
                max_seq_len=max_seq,
                bmm1_scale=1.0, bmm2_scale=1.0,
                window_left=lp.window_left,
                out=s.ao[lp.flavor][:m].view(m, -1, lp.hd),
                q_len_per_req=q,
                enable_pdl=s.pdl,
            )
            lp.b2_args[lp.fl2_idx] = (
                s.flags2[(lp.flavor, m)].data_ptr())
            s.k2[(lp.flavor, q, r)](*lp.b2_args)
        # tail: close layer 59 + the model's final norm. One contiguous
        # staging copy for rms_norm's input contract (was per-layer).
        s.mo[:m].copy_(s.out[:m, :HIDDEN])
        s._ops.rms_norm(s.xt[:m], s.mo[:m], s.w_pff_last, EPS)
        s.xt[:m].add_(s.r2[:m])
        s.xt[:m].mul_(s.ls_last)
        if 60 in s.tapset:
            s.aux_buf[60][:m].copy_(s.xt[:m])
        s._ops.rms_norm(s.hid[:m], s.xt[:m], s.norm_w_final,
                        s.norm_eps_final)

    def _build_guards(self, mds):
        seen, guards = set(), []
        for lp, md in zip(self.lps, mds):
            bp = md.decode.block_tables.data_ptr()
            if bp not in seen:
                seen.add(bp)
                guards.append((lp.lname, bp,
                               md.decode.seq_lens.data_ptr()))
        return guards

    def _guards_ok(self, key, md_all):
        for lname, bp, sp in self.g_guards[key]:
            d = md_all[lname].decode
            if (d.block_tables.data_ptr() != bp
                    or d.seq_lens.data_ptr() != sp):
                return False
        return True

    def run(self, model, input_ids, positions, inputs_embeds, md_all,
            key):
        s = self
        q, r = key
        m = q * r
        x = (inputs_embeds if inputs_embeds is not None
             else model.embed_input_ids(input_ids))
        s.xin[:m].copy_(x)
        s.pos_i32[:m].copy_(positions[:m].to(torch.int32))
        mds = [md_all[lp.lname] for lp in s.lps]
        s._last_mds, s._last_m = mds, key
        if s.native:
            cap = torch.cuda.is_current_stream_capturing()
            st0 = torch.cuda.memory_stats() if (s.cap_debug and cap) else None
            self._chain(q, r, mds, s.cap_max_seq)
            s.steps += 1
            if cap:
                if st0 is not None:
                    st1 = torch.cuda.memory_stats()
                    _log(f"capdbg {key}: x_ptr={x.data_ptr():#x} "
                         f"chain_allocs="
                         f"{st1['allocation.all.current'] - st0['allocation.all.current']} "
                         f"chain_bytes="
                         f"{st1['allocated_bytes.all.current'] - st0['allocated_bytes.all.current']}")
                if key not in s.native_captured:
                    s.native_captured.add(key)
                    _log(f"native: chain baked into vLLM graph {key}")
            else:
                s.eager_steps += 1
                if s.flag_log and s.eager_steps <= 80:
                    s._flagdump(f"after-eager {key}")
            hidden = s.hid[:m]
            if s.taps:
                return hidden, [s.aux_buf[k][:m] for k in s.taps]
            return hidden
        if s.no_capture:
            self._chain(q, r, mds, s.cap_max_seq)
            s.eager_steps += 1
            s.steps += 1
            hidden = s.hid[:m]
            if s.taps:
                return hidden, [s.aux_buf[k][:m] for k in s.taps]
            return hidden

        g = s.graphs.get(key)
        if g is not None and not self._guards_ok(key, md_all):
            _log(f"graph {key}: metadata pointers moved; recapturing")
            del s.graphs[key]
            s.g_guards.pop(key, None)
            s.eager_runs[key] = 1
            g = None
        if g is not None:
            g.replay()
            s.graph_steps += 1
        elif s.eager_runs.get(key, 0) >= 2:
            gg = torch.cuda.CUDAGraph()
            # vLLM's own capture recipe, all three ingredients (bisected
            # over 6 rounds; the killer was Python GC running mid-capture
            # and cudaFree-ing collected tensors into the recording):
            #   1. freeze/disable GC across the capture window
            #   2. parallel_state.graph_capture: capture-safe comm state
            #      + a dedicated non-default stream set as current
            #   3. thread_local error mode (engine threads stay active)
            import gc
            from vllm.distributed.parallel_state import graph_capture
            gc.collect()
            gc.freeze()
            gc.disable()
            try:
                with graph_capture(device=s.xin.device):
                    with torch.cuda.graph(
                            gg, stream=torch.cuda.current_stream(),
                            capture_error_mode="thread_local"):
                        self._chain(q, r, mds, s.cap_max_seq)
            finally:
                gc.enable()
                gc.unfreeze()
            s.graphs[key] = gg
            s.g_guards[key] = self._build_guards(mds)
            gg.replay()
            s.graph_steps += 1
            _log(f"captured cudagraph {key} "
                 f"({len(s.g_guards[key])} md guards)")
        else:
            # eager warmups use the SAME max_seq_len the capture will
            # bake: trtllm lazily initializes per-configuration state on
            # first use, and that must never happen inside a capture
            self._chain(q, r, mds, s.cap_max_seq)
            s.eager_runs[key] = s.eager_runs.get(key, 0) + 1
            s.eager_steps += 1

        s.steps += 1
        if s.log_every and s.steps % s.log_every == 0:
            _log(f"steps={s.steps} graph={s.graph_steps} "
                 f"eager={s.eager_steps} fb={s.fb} "
                 f"graphs={sorted(s.graphs)}")
        hidden = s.hid[:m]
        if s.taps:
            return hidden, [s.aux_buf[k][:m] for k in s.taps]
        return hidden


RUNNER = MegaRunner()
_ORIG_CALL = None


def install():
    """Gate Gemma4Model.__call__ — NOT forward. Under VLLM_COMPILE the
    support_torch_compile machinery owns forward: torch.compile binds
    self.forward at model init, and the bytecode-hook dispatch swaps
    Gemma4Model.forward.__code__ per call, so a forward wrapper would
    be dynamo-traced (raw-pointer kernel launches, forward-context
    reads: untraceable) and would poison original_code_object(). The
    __call__ wrapper sits above all of that: dynamo never sees it,
    compiled dispatch is delegated untouched, and vLLM's FULL
    cudagraph capture — which enters through model.__call__ — records
    whatever the gate picks: the mega chain for m_set shapes, the
    compiled stock forward for everything else. At CompilationMode
    NONE the decorator's __call__ routes straight to forward, so the
    same wrapper covers the uncompiled config too."""
    global _ORIG_CALL
    from vllm.model_executor.models import gemma4 as g4
    if _ORIG_CALL is not None:
        return
    _ORIG_CALL = g4.Gemma4Model.__call__

    def gated_call(self, input_ids, positions, intermediate_tensors=None,
                   inputs_embeds=None, per_layer_inputs=None, **kwargs):
        r = RUNNER
        if (r.enabled and not r.prep_failed
                and not torch.compiler.is_compiling()):
            g = r._gate(self, intermediate_tensors)
            if g is not None:
                md_all, key = g
                try:
                    out = r.step(self, input_ids, positions,
                                 inputs_embeds, md_all, key)
                except Exception:
                    r.prep_failed = True
                    import traceback
                    traceback.print_exc()
                    raise
                if out is not None:
                    return out
        return _ORIG_CALL(self, input_ids, positions,
                          intermediate_tensors, inputs_embeds,
                          per_layer_inputs, **kwargs)

    g4.Gemma4Model.__call__ = gated_call
    _log("Gemma4Model.__call__ gated (RUNNER.enabled="
         f"{RUNNER.enabled})")


def enable_megakernel():
    """Import-time entry (gemma4.py, VLLM_GEMMA4_MEGAKERNEL=1): gate
    Gemma4Model.__call__ and arm the runner. Weight prep + kernel
    compiles stay lazy (first gated decode step, ~80-150s once;
    async-compiled during boot when profiling-stub steps are seen)."""
    install()
    RUNNER.enabled = True
    _log("megakernel decode path ENABLED "
         f"(m_set={RUNNER.m_set}, pdl={RUNNER.pdl}, path={MK_ROOT})")
