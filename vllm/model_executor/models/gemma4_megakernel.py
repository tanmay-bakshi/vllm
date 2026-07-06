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

install() wraps Gemma4Model.forward with a gated runner: pure-decode
single-sequence steps of a prepared batch size run every layer as
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
MAXM = 64
EPS = 1e-6


def _log(msg):
    print(f"[mk_gemma4] {msg}", flush=True)


def _kv5(kvc: torch.Tensor, KV: int, hd: int) -> torch.Tensor:
    """Canonical contiguous HND view (nb, 2, KV, ps, hd) of a layer's
    registered KV cache, whatever view shape it was registered with.
    Physical storage is contiguous with pages outermost (layout=HND
    asserted by the deployed backend); sorting dims by stride recovers
    the contiguous base for a clean reshape."""
    v = kvc.view(torch.float8_e4m3fn)
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
    # layer.kv_cache[0] read at model-forward start is a one-block stub
    # for at least some layers; the live tensors are observably bound by
    # the time each Attention.forward runs (probe-verified). Harvest them
    # with self-removing pre-hooks during one sacrificial stock step.
    def _harvest(self, model):
        self.kv_map = {}
        handles = []

        def mk_hook(attn, lname, idx):
            def hook(module, args):
                kvc = attn.kv_cache[0] if isinstance(
                    attn.kv_cache, (list, tuple)) else attn.kv_cache
                self.kv_map[lname] = kvc
                if idx < 6:
                    _log(f"harvest[{lname}]: {tuple(kvc.shape)} "
                         f"stride={tuple(kvc.stride())} "
                         f"n={len(attn.kv_cache) if isinstance(attn.kv_cache, (list, tuple)) else 1}")
                for h in handles_for[lname]:
                    h.remove()
                return None
            return hook

        handles_for = {}
        for i, layer in enumerate(model.layers):
            attn = layer.self_attn.attn
            lname = attn.layer_name
            h = attn.register_forward_pre_hook(mk_hook(attn, lname, i))
            handles_for[lname] = [h]
            handles.append(h)
        _log("kv harvest hooks installed (one stock step)")

    # ---------- gate ----------
    def _gate(self, model, intermediate_tensors):
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
        q = t // r
        if (q, r) not in self.m_set:
            self.fb["m"] += 1
            return None
        return md_all, (q, r)

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
        s.q8 = {f: torch.zeros(MAXM, fc[f]["q_size"], dtype=torch.uint8,
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
        s.flags3 = {f: torch.zeros(2048, dtype=torch.int32, device=dev)
                    for f in fc}
        s.flags2 = {f: torch.zeros(2 * INTER // 128 + t2_slots + 2 + 2048,
                                   dtype=torch.int32, device=dev)
                    for f in fc}
        # trtllm-gen workspace: internal split scheduling scales with
        # max_seq_len (graphs capture at the 8192 upper bound); deployed
        # allocates 413MB for this config -- match it with headroom
        ws_mb = int(os.environ.get("MK_WS_MB", "512"))
        s.ws = torch.zeros(ws_mb * 1024 * 1024, dtype=torch.uint8, device=dev)

        # ---- compile the four kernels (m_real per self.m_set) ----
        from mk_fused.fused_preattn_sm100 import Sm100PreAttnKernel
        from mk_fused.fused_postattn_sm100 import Sm100PostAttnKernel
        from mkbench.cutedsl_driver import (compile_fused_preattn,
                                            compile_fused_postattn)
        s.k3, s.k2 = {}, {}
        k2_by_m = {}
        for (q, r) in self.m_set:
            M = q * r
            mp = max(M, 8)
            for f in fc:
                c = fc[f]
                if f == "global" and "MK_KSPLIT_Q" not in os.environ:
                    os.environ["MK_KSPLIT_Q"] = "2"
                k = Sm100PreAttnKernel(
                    mp, M, c["n_qkv"], HIDDEN, c["heads"], c["KV"],
                    c["hd"], kv_numel=c["kv_numel"],
                    cs_numel=c["cs"].numel(), bt_len=c["bt_len"],
                    page_size=c["ps"], bt_stride=c["bt_len"], q_len=q,
                    cluster_shape_mn=(1, 4), enable_pdl=self.pdl)
                s.k3[(f, q, r)] = compile_fused_preattn(k)
                if f == "global":
                    os.environ.pop("MK_KSPLIT_Q", None)
                if (f, M) not in k2_by_m:
                    k = Sm100PostAttnKernel(
                        mp, M, 2 * INTER, HIDDEN, N2, INTER,
                        c["q_size"], mma_tiler_mn=(128, 128),
                        cluster_shape_mn=(1, 4), enable_pdl=self.pdl)
                    k2_by_m[(f, M)] = compile_fused_postattn(k)
                s.k2[(f, q, r)] = k2_by_m[(f, M)]
                _log(f"compiled B3+B2 {f} q={q} r={r} M={M} "
                     f"({time.time() - t0:.0f}s)")

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
                s.mo.data_ptr(), s.r2.data_ptr(),
                norms["wpf2_prev"].data_ptr(), norms["win"].data_ptr(),
                s.h.data_ptr(), s.xq8.data_ptr(), s.xs.data_ptr(),
                wq8.data_ptr(), wqs.data_ptr(),
                s.qkvacc[f].data_ptr(),
                norms["qn"].data_ptr(), norms["kn"].data_ptr(),
                c["cs"].data_ptr(), s.pos_i32.data_ptr(),
                0,  # block table ptr, patched per step (bt_idx)
                s.q8[f].data_ptr(), kvc.data_ptr(),
                scales, s.flags3[f].data_ptr(),
            ]
            lp.bt_idx = 14
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
                s.out.data_ptr(), alpha, s.flags2[f].data_ptr(),
            ]
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
        # KEEPALIVE for the cos_sin caches: their data_ptrs are baked
        # into every B3 arg list, but fc is a local — without this ref
        # they get freed to the allocator cache and the first
        # torch.cuda.graph.__enter__ (which calls empty_cache) unmaps
        # them out from under the baked pointers. (7-round bisect.)
        s.cs_keep = {f: fc[f]["cs"] for f in fc}
        _log(f"aux taps: {s.taps}; cap_max_seq {s.cap_max_seq}")
        self.prepared = True
        _log(f"prepare done in {time.time() - t0:.0f}s")

    # ---------- the decode step ----------
    def _chain(self, q, r, mds, max_seq):
        """The capturable decode step: reads xin/pos_i32 + baked
        pointers, writes hid + aux_buf. Allocation- and sync-free.
        q = tokens per request, r = requests, T = q*r total rows."""
        s = self
        m = q * r
        s.mo[:m].zero_()
        s.r2[:m].copy_(s.xin[:m])
        if 0 in s.tapset:
            s.aux_buf[0][:m].copy_(s.xin[:m])
        for i, (lp, md) in enumerate(zip(s.lps, mds)):
            args = lp.b3_args
            args[lp.bt_idx] = md.decode.block_tables.data_ptr()
            s.k3[(lp.flavor, q, r)](*args)
            if i in s.tapset and i > 0:
                s.aux_buf[i][:m].copy_(s.h[:m])
            s._decode(
                query=s.q8[lp.flavor][:m].view(torch.float8_e4m3fn)
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
            s.k2[(lp.flavor, q, r)](*lp.b2_args)
            s.mo[:m].copy_(s.out[:m, :HIDDEN])
        # tail: close layer 59 + the model's final norm
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
_ORIG = None


def install():
    global _ORIG
    from vllm.model_executor.models import gemma4 as g4
    if _ORIG is not None:
        return
    _ORIG = g4.Gemma4Model.forward

    def fwd(self, input_ids, positions, intermediate_tensors,
            inputs_embeds=None, per_layer_inputs=None, **kwargs):
        r = RUNNER
        if r.enabled and not r.prep_failed:
            g = r._gate(self, intermediate_tensors)
            if g is not None:
                md_all, m = g
                if r.kv_map is None:
                    # sacrificial stock step harvests live KV tensors
                    r._harvest(self)
                elif len(r.kv_map) < len(list(self.layers)):
                    pass  # hooks still pending; stay on stock
                else:
                    if not r.prepared:
                        try:
                            r.prepare(self)
                        except Exception:
                            r.prep_failed = True
                            import traceback
                            traceback.print_exc()
                            raise
                    return r.run(self, input_ids, positions,
                                 inputs_embeds, md_all, m)  # m=(q,r)
        return _ORIG(self, input_ids, positions, intermediate_tensors,
                     inputs_embeds, per_layer_inputs, **kwargs)

    g4.Gemma4Model.forward = fwd
    _log("Gemma4Model.forward patched (RUNNER.enabled="
         f"{RUNNER.enabled})")


def enable_megakernel():
    """Import-time entry (gemma4.py, VLLM_GEMMA4_MEGAKERNEL=1): patch
    Gemma4Model.forward and arm the runner. Weight prep + kernel
    compiles stay lazy (first gated decode step, ~80-150s once)."""
    install()
    RUNNER.enabled = True
    _log("megakernel decode path ENABLED "
         f"(m_set={RUNNER.m_set}, pdl={RUNNER.pdl}, path={MK_ROOT})")
