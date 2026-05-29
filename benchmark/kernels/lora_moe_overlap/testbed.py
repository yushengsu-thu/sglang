"""UT testbed for the MoE-LoRA virtual-experts shrink/expand kernels + two-stream overlap.

Goal: iterate quickly on (a) the two profiled triton kernels and (b) overlap approaches,
without launching a full server.

Profiled kernels (virtual-experts LoRA path on branch `lora-opti`):
  - `_moe_lora_shrink_splitk_kernel`  (LoRA-A / gate-up shrink)  ~10.8 us
      python/sglang/srt/lora/triton_ops/virtual_experts.py
  - `_moe_lora_expand_add_kernel`     (LoRA-B / down expand-add)  ~2.9 us
      python/sglang/srt/lora/trtllm_moe/specialized_expand.py

Modes
  verify   run the production e2e op + an isolated shrink->expand path, both vs a torch
           reference (correctness gate for kernel edits).
  bench    triton.do_bench latency for: isolated shrink, isolated expand, and the e2e op.
  shapes   print get_tensor_info for every kernel in/out tensor (synthetic; complements the
           on-machine SGLANG_DEBUG_LORA_MOE_SHAPES capture which is the source of truth).
  overlap  toy two-stream: lora gate_up gemms on a side stream concurrent with a main-stream
           FP8 block-scale MoE, sequential vs overlapped wall-clock. ROUGH (single-GPU SM
           contention is shape-dependent) -- this just lets us sweep shapes. The *real*
           fused overlap is wired e2e behind SGLANG_LORA_TWO_STREAM=1 (see moe_overlap.py).

Examples
  python testbed.py verify  --model qwen35 --regime decode
  python testbed.py bench    --model kimi   --regime decode --bs 32
  python testbed.py shapes   --all
  python testbed.py overlap  --model qwen35 --regime decode --bs 32
"""

import argparse
import functools
from dataclasses import dataclass

import torch
import triton

# ---- production code under test (import the real launchers + routing primitives) ----
from sglang.srt.lora.triton_ops import merged_experts_fused_moe_lora_add
from sglang.srt.lora.triton_ops.virtual_experts import (
    _align_block_size_large,
    _fused_virtual_topk_ids,
    _get_moe_lora_shrink_split_k,
    _invoke_moe_lora_expand_add,
    _invoke_moe_lora_shrink_splitk,
    fused_sanitize_expert_ids,
)
from sglang.srt.debug_utils.dumper import get_tensor_info


# =============================================================================
# Model presets
# =============================================================================
# Real per-rank dims captured live via SGLANG_DEBUG_LORA_MOE_SHAPES (results/SHAPES_FOR_CHUNAN.md).
# KEY: the virtual-expert LoRA keeps the FULL routed-expert count per rank (it is NOT EP-sharded),
# so `experts` below is the model's routed-expert count, not //ep. This ModelCfg models the
# gate_up stage (the one overlapped on the side stream); for the exact two-stage shapes incl.
# the down stage, use bench_real_shapes.py. n_out = gate_up per-rank moe-intermediate.
@dataclass
class ModelCfg:
    name: str
    hidden: int  # model hidden_size = shrink K
    n_out: int  # gate_up expand N (per-rank moe-intermediate)
    experts: int  # routed experts per rank (NOT EP-sharded)
    topk: int
    lora_rank: int

    @property
    def per_gpu_experts(self) -> int:
        return self.experts


MODELS = {
    # tp4 ep4, server-captured
    "qwen35": ModelCfg("Qwen3.5-35B-A3B-FP8", 2048, 1024, 256, 8, 16),
    "qwen3vl": ModelCfg("Qwen3-VL-30B-A3B-Instruct-FP8", 2048, 1536, 128, 8, 16),
    # tp8 no-EP, DERIVED (see kimi_shapes_derived.md)
    "kimi": ModelCfg("Kimi-K2.5-NVFP4", 7168, 256, 384, 8, 16),
}


def regime_num_tokens(regime: str, bs: int, input_len: int) -> int:
    # prefill: the chunked-prefill batch the kernel sees ~= bs * input_len (upper bound;
    #          real serving chunks this, but the shape axis is "many tokens").
    # decode : one token per running sequence -> num_tokens == bs.
    return bs * input_len if regime == "prefill" else bs


# =============================================================================
# Input synthesis (mirrors the gate_up LoRA stage in moe_overlap.py)
# =============================================================================
def make_inputs(cfg: ModelCfg, num_tokens: int, max_loras: int, dtype, device="cuda", seed=0):
    """Build inputs for the gate_up LoRA stage:
       hidden_states[T, hidden], lora_a[max_loras, E, rank, hidden],
       lora_b[max_loras, E, n_out, rank], topk_ids[T, topk], topk_weights[T, topk],
       token_lora_mapping[T] in {-1, 0..max_loras-1}.
    n_out defaults to hidden (down-expand shape); the expand kernel is rank-specialized
    so n_out only affects the N-loop, not correctness logic.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    E = cfg.per_gpu_experts
    rank = cfg.lora_rank
    topk = cfg.topk
    n_out = cfg.n_out

    hidden_states = torch.randn(num_tokens, cfg.hidden, dtype=dtype, device=device, generator=g)
    lora_a = torch.randn(max_loras, E, rank, cfg.hidden, dtype=dtype, device=device, generator=g) * 0.02
    lora_b = torch.randn(max_loras, E, n_out, rank, dtype=dtype, device=device, generator=g) * 0.02

    # topk distinct experts per token
    topk_ids = torch.empty(num_tokens, topk, dtype=torch.int32, device=device)
    for i in range(num_tokens):
        topk_ids[i] = torch.randperm(E, generator=g, device=device)[:topk].to(torch.int32)
    topk_weights = torch.rand(num_tokens, topk, dtype=torch.float32, device=device, generator=g)
    topk_weights = topk_weights / topk_weights.sum(dim=1, keepdim=True)

    # token -> lora. include some base tokens (-1) to exercise masking.
    token_lora_mapping = torch.randint(
        0, max_loras, (num_tokens,), dtype=torch.int32, device=device, generator=g
    )
    if num_tokens >= 4:  # mark ~1/4 as base tokens (no LoRA)
        token_lora_mapping[: num_tokens // 4] = -1
    return dict(
        hidden_states=hidden_states,
        lora_a=lora_a,
        lora_b=lora_b,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        token_lora_mapping=token_lora_mapping,
    )


# =============================================================================
# Routing + config mirror of virtual_experts.py _get_routing / _get_stage_config
# (kept faithful; cited line numbers so drift is catchable. verify cross-checks it.)
# =============================================================================
def stage_config(weight, stage_top_k, hidden_dtype, num_tokens):
    # mirrors virtual_experts.py:_get_stage_config
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config import (
        get_config_dtype_str,
        try_get_optimal_moe_config,
    )

    config_dtype = get_config_dtype_str(dtype=hidden_dtype)
    get_config_func = functools.partial(
        try_get_optimal_moe_config, weight.shape, weight.shape, stage_top_k, config_dtype
    )
    try:
        return get_config_func(num_tokens)
    except ValueError:
        K_dim, N_dim = weight.shape[2], weight.shape[1]
        default_block_k = 256 if K_dim >= 1024 else (64 if K_dim >= 64 else max(16, K_dim))
        return {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": min(64, max(16, N_dim)),
            "BLOCK_SIZE_K": min(default_block_k, max(16, K_dim)),
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 4,
        }


def build_routing(topk_ids, token_lora_mapping, num_experts, block_size, max_loras, shared_outer=False):
    # mirrors virtual_experts.py:_get_routing (no routing_cache, no single-lora fast path
    # -- lora-opti always uses _fused_virtual_topk_ids)
    virtual_topk_ids, token_lora_mask, virtual_num_experts = _fused_virtual_topk_ids(
        topk_ids, token_lora_mapping, num_experts, shared_outer, max_loras
    )
    if virtual_num_experts < 1024:
        from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
            moe_align_block_size as native_align,
        )

        sorted_token_ids, expert_ids, num_tokens_post_padded = native_align(
            virtual_topk_ids, block_size, virtual_num_experts
        )
    else:
        sorted_token_ids, expert_ids, num_tokens_post_padded = _align_block_size_large(
            virtual_topk_ids, block_size, virtual_num_experts
        )
    num_tok = topk_ids.numel()
    max_nonempty = min(num_tok, virtual_num_experts)
    tight = triton.cdiv(num_tok + max_nonempty * (block_size - 1), block_size) * block_size
    sorted_token_ids = sorted_token_ids[:tight]
    expert_ids = expert_ids[: tight // block_size]
    expert_ids = fused_sanitize_expert_ids(expert_ids, virtual_num_experts)
    return sorted_token_ids, expert_ids, num_tokens_post_padded, token_lora_mask


# =============================================================================
# Torch reference
# =============================================================================
def torch_ref(inp, sum_over_topk, mul_routed_weight):
    """Reference for the e2e merged_experts_fused_moe_lora_add.
    sum_over_topk=False (gate_up): out[T, topk, n_out], store per (token, expert-slot).
    sum_over_topk=True  (down):    out[T, n_out],       sum_k weight*delta.
    Base tokens (mapping == -1) contribute 0.
    """
    h = inp["hidden_states"]
    la, lb = inp["lora_a"], inp["lora_b"]
    topk_ids, topk_w, tlm = inp["topk_ids"], inp["topk_weights"], inp["token_lora_mapping"]
    T, topk = topk_ids.shape
    n_out = lb.shape[2]
    if sum_over_topk:
        out = torch.zeros(T, n_out, dtype=h.dtype, device=h.device)
    else:
        out = torch.zeros(T, topk, n_out, dtype=h.dtype, device=h.device)
    for i in range(T):
        lora = int(tlm[i])
        if lora < 0:
            continue
        hi = h[i].float()
        for k in range(topk):
            e = int(topk_ids[i, k])
            a = la[lora, e].float()  # [rank, hidden]
            b = lb[lora, e].float()  # [n_out, rank]
            d = hi @ a.T @ b.T  # [n_out]
            if mul_routed_weight:
                d = d * float(topk_w[i, k])
            if sum_over_topk:
                out[i] += d.to(out.dtype)
            else:
                out[i, k] = d.to(out.dtype)
    return out


# =============================================================================
# e2e + isolated drivers
# =============================================================================
def run_e2e_gate_up(inp):
    """gate_up stage: output [T, topk, n_out], store, mul_routed_weight=False."""
    T, topk = inp["topk_ids"].shape
    n_out = inp["lora_b"].shape[2]
    rank = inp["lora_a"].shape[2]
    out = torch.zeros(T, topk, n_out, dtype=inp["hidden_states"].dtype, device="cuda")
    merged_experts_fused_moe_lora_add(
        output=out,
        hidden_states=inp["hidden_states"],
        lora_a=inp["lora_a"],
        lora_b=inp["lora_b"],
        topk_ids=inp["topk_ids"],
        topk_weights=inp["topk_weights"],
        token_lora_mapping=inp["token_lora_mapping"],
        mul_routed_weight=False,
        experts_shared_outer_loras_a=False,
        experts_shared_outer_loras_b=False,
        routing_cache=None,
        fuse_add_to_output=False,
        fuse_sum_all_reduce=False,
        use_direct_expand_add=rank <= 64,
    )
    return out


def run_isolated(inp):
    """Reproduce shrink -> expand with the isolated launchers (uses build_routing).
    Returns (intermediate, gate_up_out) so verify can cross-check against e2e + torch.
    """
    h = inp["hidden_states"]
    topk_ids = inp["topk_ids"]
    T, topk = topk_ids.shape
    max_loras, E, rank, hidden = inp["lora_a"].shape
    n_out = inp["lora_b"].shape[2]
    lora_a_v = inp["lora_a"].reshape(max_loras * E, rank, hidden)
    lora_b_v = inp["lora_b"].reshape(max_loras * E, n_out, rank)

    a_cfg = stage_config(lora_a_v, topk, h.dtype, T)
    s_ids, e_ids, ntpp, _ = build_routing(
        topk_ids, inp["token_lora_mapping"], E, a_cfg["BLOCK_SIZE_M"], max_loras
    )
    split_k = _get_moe_lora_shrink_split_k(lora_a_v, s_ids, a_cfg)
    interm_shape = (T, topk, rank)
    intermediate = (torch.zeros if split_k > 1 else torch.empty)(
        interm_shape, dtype=h.dtype, device="cuda"
    )
    _invoke_moe_lora_shrink_splitk(
        h, lora_a_v, intermediate.view(-1, rank), topk_ids, s_ids, e_ids, ntpp, topk, a_cfg
    )

    b_cfg = stage_config(lora_b_v, 1, h.dtype, T)
    s_ids_b, e_ids_b, ntpp_b, _ = build_routing(
        topk_ids, inp["token_lora_mapping"], E, b_cfg["BLOCK_SIZE_M"], max_loras
    )
    out = torch.zeros(T, topk, n_out, dtype=h.dtype, device="cuda")
    _invoke_moe_lora_expand_add(
        intermediate.view(-1, rank),
        lora_b_v,
        out.view(-1, n_out),
        inp["topk_weights"],
        topk_ids,
        s_ids_b,
        e_ids_b,
        ntpp_b,
        b_cfg,
        mul_routed_weight=False,
        fuse_sum_all_reduce=False,
    )
    return intermediate, out


# =============================================================================
# Commands
# =============================================================================
def cmd_verify(cfg, args):
    inp = make_inputs(cfg, args.num_tokens, args.max_loras, args.dtype)
    ref = torch_ref(inp, sum_over_topk=False, mul_routed_weight=False)
    e2e = run_e2e_gate_up(inp)
    _, iso = run_isolated(inp)
    print(f"[verify] {cfg.name} T={args.num_tokens} rank={cfg.lora_rank} E={cfg.per_gpu_experts} topk={cfg.topk}")
    _report_close("e2e   vs torch", e2e, ref)
    _report_close("isolated vs torch", iso, ref)
    _report_close("isolated vs e2e  ", iso, e2e)


def _report_close(tag, a, b):
    a, b = a.float(), b.float()
    max_abs = (a - b).abs().max().item()
    denom = b.abs().max().item() + 1e-6
    ok = torch.allclose(a, b, atol=1e-2, rtol=1e-2)
    print(f"   {tag}: max_abs_err={max_abs:.4e} rel={max_abs/denom:.4e} -> {'OK' if ok else 'FAIL'}")
    return ok


def cmd_bench(cfg, args):
    inp = make_inputs(cfg, args.num_tokens, args.max_loras, args.dtype)
    h = inp["hidden_states"]
    topk_ids = inp["topk_ids"]
    T, topk = topk_ids.shape
    max_loras, E, rank, hidden = inp["lora_a"].shape
    n_out = inp["lora_b"].shape[2]
    lora_a_v = inp["lora_a"].reshape(max_loras * E, rank, hidden)
    lora_b_v = inp["lora_b"].reshape(max_loras * E, n_out, rank)

    a_cfg = stage_config(lora_a_v, topk, h.dtype, T)
    s_ids, e_ids, ntpp, _ = build_routing(topk_ids, inp["token_lora_mapping"], E, a_cfg["BLOCK_SIZE_M"], max_loras)
    split_k = _get_moe_lora_shrink_split_k(lora_a_v, s_ids, a_cfg)
    intermediate = torch.zeros(T, topk, rank, dtype=h.dtype, device="cuda")

    def _shrink():
        _invoke_moe_lora_shrink_splitk(
            h, lora_a_v, intermediate.view(-1, rank), topk_ids, s_ids, e_ids, ntpp, topk, a_cfg
        )

    b_cfg = stage_config(lora_b_v, 1, h.dtype, T)
    s_ids_b, e_ids_b, ntpp_b, _ = build_routing(topk_ids, inp["token_lora_mapping"], E, b_cfg["BLOCK_SIZE_M"], max_loras)
    out = torch.zeros(T, topk, n_out, dtype=h.dtype, device="cuda")

    def _expand():
        _invoke_moe_lora_expand_add(
            intermediate.view(-1, rank), lora_b_v, out.view(-1, n_out), inp["topk_weights"],
            topk_ids, s_ids_b, e_ids_b, ntpp_b, b_cfg, False, False,
        )

    t_shrink = triton.testing.do_bench(_shrink, warmup=25, rep=100)
    t_expand = triton.testing.do_bench(_expand, warmup=25, rep=100)
    t_e2e = triton.testing.do_bench(lambda: run_e2e_gate_up(inp), warmup=10, rep=50)
    print(f"[bench] {cfg.name} T={T} rank={rank} E={E} topk={topk} split_k={split_k}")
    print(f"   shrink_splitk : {t_shrink*1e3:8.2f} us   (cfg={a_cfg.get('BLOCK_SIZE_M')},{a_cfg.get('BLOCK_SIZE_N')},{a_cfg.get('BLOCK_SIZE_K')})")
    print(f"   expand_add    : {t_expand*1e3:8.2f} us")
    print(f"   e2e gate_up   : {t_e2e*1e3:8.2f} us")


def cmd_shapes(cfg, args):
    inp = make_inputs(cfg, args.num_tokens, args.max_loras, args.dtype)
    max_loras, E, rank, hidden = inp["lora_a"].shape
    n_out = inp["lora_b"].shape[2]
    lora_a_v = inp["lora_a"].reshape(max_loras * E, rank, hidden)
    lora_b_v = inp["lora_b"].reshape(max_loras * E, n_out, rank)
    topk_ids = inp["topk_ids"]
    T, topk = topk_ids.shape
    a_cfg = stage_config(lora_a_v, topk, inp["hidden_states"].dtype, T)
    s_ids, e_ids, ntpp, _ = build_routing(topk_ids, inp["token_lora_mapping"], E, a_cfg["BLOCK_SIZE_M"], max_loras)
    intermediate = torch.zeros(T, topk, rank, dtype=inp["hidden_states"].dtype, device="cuda")
    print(f"\n===== {cfg.name}  regime={args.regime}  T={T}  max_loras={args.max_loras} =====")
    print(f"-- shrink_splitk (LoRA-A / gate-up) --")
    for name, t in [
        ("hidden_states(a)", inp["hidden_states"]), ("lora_a_virtual(b)", lora_a_v),
        ("intermediate(c)", intermediate.view(-1, rank)), ("topk_ids", topk_ids),
        ("sorted_token_ids", s_ids), ("expert_ids", e_ids), ("num_tokens_post_padded", ntpp),
    ]:
        print(f"   {name}: {get_tensor_info(t)}")
    print(f"-- expand_add (LoRA-B / down) --")
    for name, t in [
        ("intermediate(a)", intermediate.view(-1, rank)), ("lora_b_virtual(b)", lora_b_v),
        ("topk_weights", inp["topk_weights"]),
    ]:
        print(f"   {name}: {get_tensor_info(t)}")


def cmd_overlap(cfg, args):
    """Toy two-stream: lora gate_up gemms (side stream) vs main-stream FP8 block-scale MoE.
    ROUGH measurement -- see module docstring. Skips gracefully if flashinfer trtllm is absent.
    """
    inp = make_inputs(cfg, args.num_tokens, args.max_loras, args.dtype)

    def _lora():
        run_e2e_gate_up(inp)

    main_fn = _build_trtllm_main(cfg, args)
    if main_fn is None:
        print("[overlap] trtllm_fp8_block_scale_moe unavailable; timing lora gemms only.")
        t = triton.testing.do_bench(_lora, warmup=10, rep=50)
        print(f"   lora gate_up gemms: {t*1e3:.2f} us")
        return

    side = torch.cuda.Stream()

    def _sequential():
        main_fn()
        _lora()

    def _two_stream():
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            _lora()
        main_fn()
        torch.cuda.current_stream().wait_stream(side)

    t_main = triton.testing.do_bench(main_fn, warmup=10, rep=50)
    t_lora = triton.testing.do_bench(_lora, warmup=10, rep=50)
    t_seq = triton.testing.do_bench(_sequential, warmup=10, rep=50)
    t_ovl = triton.testing.do_bench(_two_stream, warmup=10, rep=50)
    print(f"[overlap] {cfg.name} T={args.num_tokens}")
    print(f"   main(trtllm fp8 moe): {t_main*1e3:8.2f} us")
    print(f"   lora gate_up gemms  : {t_lora*1e3:8.2f} us")
    print(f"   sequential          : {t_seq*1e3:8.2f} us")
    print(f"   two-stream          : {t_ovl*1e3:8.2f} us   (savings {100*(t_seq-t_ovl)/t_seq:5.1f}%)")
    print("   NOTE: rough single-GPU measurement; real fused overlap is SGLANG_LORA_TWO_STREAM=1.")


def _build_trtllm_main(cfg, args):
    """Best-effort construction of plain trtllm_fp8_block_scale_moe inputs. Returns a thunk
    or None if unavailable. Constructed/validated on the GPU node (mac has no GPU)."""
    try:
        from sglang.jit_kernel.flashinfer_trtllm_moe import trtllm_fp8_block_scale_moe
    except Exception as e:  # noqa: BLE001
        print(f"[overlap] import failed: {e}")
        return None
    # Built on-node; left as a documented stub to fill once real FP8 weight/scale layout is
    # confirmed against a flashinfer reference test.
    return None


# =============================================================================
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["verify", "bench", "shapes", "overlap"])
    p.add_argument("--model", choices=list(MODELS), default="qwen35")
    p.add_argument("--all", action="store_true", help="shapes: iterate all models x {prefill,decode}")
    p.add_argument("--regime", choices=["prefill", "decode"], default="decode")
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--input-len", type=int, default=2048)
    p.add_argument("--num-tokens", type=int, default=None, help="override; else derived from regime/bs")
    p.add_argument("--max-loras", type=int, default=1, help="prod serves --max-loras-per-batch 1")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = p.parse_args()
    args.dtype = getattr(torch, args.dtype)

    def _resolve(cfg):
        if args.num_tokens is None:
            args.num_tokens = regime_num_tokens(args.regime, args.bs, args.input_len)
        return cfg

    if args.mode == "shapes" and args.all:
        for key, cfg in MODELS.items():
            for regime in ["prefill", "decode"]:
                args.regime = regime
                args.num_tokens = regime_num_tokens(regime, args.bs, args.input_len)
                cmd_shapes(cfg, args)
        return

    cfg = _resolve(MODELS[args.model])
    {"verify": cmd_verify, "bench": cmd_bench, "shapes": cmd_shapes, "overlap": cmd_overlap}[args.mode](cfg, args)


if __name__ == "__main__":
    main()
