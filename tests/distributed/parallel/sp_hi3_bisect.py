"""HI3 Ulysses SP bf16 divergence bisection — per-layer, with determinism check.

The fp32 parity test (``sp_hi3_parity.py``, relerr ~1e-6) passes, yet e2e (bf16)
shows the SP train-forward diverging from non-SP: image GRPO ratio_mean 1.00000
(no-SP) -> 0.995 (SP), std 1e-5 -> 4e-3. This harness reproduces + localizes that
under the real variable the parity test never exercised: **bf16** (+ depth).

For each (world, dtype): build a shrunken real HI3 decoder (LAYERS layers), hook
every decoder layer, run the forward TWICE (determinism), and save per-layer
hidden (gathered to full seq under SP). Run world=1 first (reference), then
world=N: per-layer max|Δ| vs the world=1 reference localizes the first/growing
divergence.

  HI3_MODEL_PATH=<dir> DTYPE=bf16 LAYERS=8 SEQLEN=66 \
  torchrun --nproc_per_node=1 --master_port=29580 sp_hi3_bisect.py   # ref
  torchrun --nproc_per_node=8 --master_port=29581 sp_hi3_bisect.py   # compare
"""
import os

import torch
import torch.distributed as dist

from unirl.train.backend.veomni import _compat

_compat.ensure_installed()
from veomni.distributed.parallel_state import get_parallel_state, init_parallel_state  # noqa: E402
from unirl.train.backend.veomni.sp import apply_sequence_parallelism  # noqa: E402

if os.environ.get("TRITON_LINEAR"):
    # Route every nn.Linear through a fixed-tiling Triton matmul -> M-invariant,
    # so the SP (M=L/sp) and non-SP (M=L) forwards use the SAME K-reduction order.
    import triton
    import triton.language as tl
    import torch.nn.functional as _F

    @triton.jit
    def _mmk(A, B, C, M, N, K, sam, sak, sbk, sbn, scm, scn,
             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        pm = tl.program_id(0)
        pn = tl.program_id(1)
        om = pm * BM + tl.arange(0, BM)
        on = pn * BN + tl.arange(0, BN)
        ok = tl.arange(0, BK)
        ap = A + om[:, None] * sam + ok[None, :] * sak
        bp = B + ok[:, None] * sbk + on[None, :] * sbn
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k in range(0, K, BK):
            a = tl.load(ap, mask=(om[:, None] < M) & (ok[None, :] < K - k), other=0.0)
            b = tl.load(bp, mask=(ok[:, None] < K - k) & (on[None, :] < N), other=0.0)
            acc += tl.dot(a, b)
            ap += BK * sak
            bp += BK * sbk
        tl.store(C + om[:, None] * scm + on[None, :] * scn, acc.to(C.dtype.element_ty),
                 mask=(om[:, None] < M) & (on[None, :] < N))

    def _triton_linear(x, weight, bias=None):
        shp = x.shape
        a = x.reshape(-1, shp[-1]).contiguous()
        b = weight.t().contiguous()
        M, K = a.shape
        N = b.shape[1]
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)
        BM, BN, BK = 128, 128, 32
        grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
        _mmk[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                   c.stride(0), c.stride(1), BM, BN, BK)
        c = c.reshape(*shp[:-1], N)
        return c + bias if bias is not None else c

    _F.linear = _triton_linear

if os.environ.get("BATCH_INVARIANT") and int(os.environ.get("WORLD_SIZE", "1")) > 1:
    # Only the sp>1 run goes batch-invariant; sp=1 stays cuBLAS (the reference the
    # rollout matches). FINAL=0 then means VeOmni's batch-invariant matmul reproduces
    # cuBLAS-at-M=full on the *real* activations (not just random data).
    from veomni.ops.batch_invariant_ops import enable_batch_invariant_mode

    enable_batch_invariant_mode()

HI3_DIR = os.environ.get(
    "HI3_MODEL_PATH", "/apdcephfs_fsgm3/share_305110755/hunyuan/linyuwu/HunyuanImage-3-Instruct"
)
DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[os.environ.get("DTYPE", "bf16")]
LAYERS = int(os.environ.get("LAYERS", "8"))
L = int(os.environ.get("SEQLEN", "66"))
MODE = os.environ.get("MODE", "gen_text")
TAG = f"L{LAYERS}_{os.environ.get('DTYPE','bf16')}_S{L}_{MODE}"
SD_PATH = f"/tmp/hi3_bisect_sd_{TAG}.pt"
REF_PATH = f"/tmp/hi3_bisect_ref_{TAG}.pt"


def build_model(dev):
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    CfgCls = get_class_from_dynamic_module("configuration_hunyuan_image_3.HunyuanImage3Config", HI3_DIR)
    ModelCls = get_class_from_dynamic_module("modeling_hunyuan_image_3.HunyuanImage3Model", HI3_DIR)
    cfg = CfgCls.from_pretrained(HI3_DIR)
    cfg.num_hidden_layers = LAYERS
    real = bool(os.environ.get("REAL_WEIGHTS"))
    if not real:
        cfg.num_experts = 8
        if isinstance(getattr(cfg, "moe_topk", None), list):
            cfg.moe_topk = [min(x, 4) for x in cfg.moe_topk]
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(0)
    m = ModelCls(cfg).to(dev).to(DTYPE)
    if real:
        _load_real_layer_weights(m, LAYERS, dev, DTYPE)
    return m, cfg


def _load_real_layer_weights(m, n_layers, dev, dtype):
    """Load the REAL checkpoint weights for layers 0..n_layers-1 (+ non-layer decoder
    params) so the parity test runs on the real bf16-rounding behavior the random-weight
    bisect misses. Keeps full config (64 experts)."""
    import glob

    from safetensors.torch import safe_open

    keep = set(range(n_layers))
    sd = {}
    for f in sorted(glob.glob(os.path.join(HI3_DIR, "*.safetensors"))):
        with safe_open(f, framework="pt", device=str(dev)) as fh:
            for k in fh.keys():
                if not k.startswith("model."):
                    continue
                rest = k[len("model."):]
                if rest.startswith("layers."):
                    if int(rest.split(".")[1]) not in keep:
                        continue
                sd[rest] = fh.get_tensor(k).to(dtype)
    res = m.load_state_dict(sd, strict=False)
    miss = [x for x in res.missing_keys if not x.startswith(("embed", "wte", "tok"))]
    print(f"[REAL_WEIGHTS] loaded {len(sd)} tensors; missing(non-embed)={len(miss)} unexpected={len(res.unexpected_keys)}", flush=True)


def make_mask(b, l, dev):
    allow = torch.ones(l, l, dtype=torch.bool, device=dev).tril()
    a, c = l // 4, (3 * l) // 4
    allow[a:c, a:c] = True
    m = torch.zeros(l, l, dtype=DTYPE, device=dev)
    m.masked_fill_(~allow, torch.finfo(DTYPE).min)
    return m.view(1, 1, l, l).expand(b, 1, l, l).contiguous()


def make_rope(b, l, d, dev):
    pos = torch.arange(l, device=dev).float()
    i = torch.arange(0, d, 2, device=dev).float()
    ang = torch.outer(pos, 1.0 / (10000 ** (i / d)))
    emb = torch.cat([ang, ang], -1)
    return emb.cos()[None].expand(b, l, d).to(DTYPE).contiguous(), emb.sin()[None].expand(b, l, d).to(DTYPE).contiguous()


def main():
    if not os.path.isdir(HI3_DIR):
        print(f"SKIP: HI3 dir not found {HI3_DIR}", flush=True)
        return
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    lr = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(lr)
    dev = torch.device("cuda", lr)
    init_parallel_state(dp_size=1, ulysses_size=world, dp_mode="fsdp2", device_type="cuda")

    m, cfg = build_model(dev)
    if rank == 0 and not os.path.exists(SD_PATH):
        torch.save(m.state_dict(), SD_PATH)
    dist.barrier()
    m.load_state_dict(torch.load(SD_PATH, map_location=dev))
    if world > 1 and os.environ.get("GROUPED_MOE"):
        # sp>1 only: restructure experts to stacked params so the SP adapter routes
        # them through veomni's M-invariant grouped-GEMM. sp=1 stays per-expert cuBLAS
        # (the reference), so FINAL=0 means grouped(M=shard) == cuBLAS(M=full) on real
        # activations.
        from unirl.train.backend.veomni.sp.hi3_moe import restructure_hi3_experts

        restructure_hi3_experts(m)
    if world > 1:
        apply_sequence_parallelism(m, world)
    m.eval()

    # Dim-aware capture: captured[key] = (tensor, seq_dim). seq is dim=1 for most
    # activations, dim=2 for the per-head q after qk-norm ([B, H, L, d]).
    captured = {}

    def cap(key, out, dim=1):
        captured[key] = (out[0] if isinstance(out, (tuple, list)) else out, dim)

    # accumulation: every layer's output, plus attn & mlp sub-steps for ALL layers
    for i, layer in enumerate(m.layers):
        layer.register_forward_hook(lambda mod, inp, out, i=i: cap(i, out, 1))
        layer.self_attn.register_forward_hook(lambda mod, inp, out, i=i: cap(f"Y{i:02d}a_attn", out, 1))
        layer.mlp.register_forward_hook(lambda mod, inp, out, i=i: cap(f"Y{i:02d}b_mlp", out, 1))

    # fine op-by-op trajectory through layer 0
    L0 = m.layers[0]
    L0.register_forward_pre_hook(lambda mod, args: cap("X00_layer_input", (args[0],), 1))
    if hasattr(L0, "input_layernorm"):
        L0.input_layernorm.register_forward_hook(lambda mod, inp, out: cap("X01_input_ln", out, 1))
    L0.self_attn.qkv_proj.register_forward_hook(lambda mod, inp, out: cap("X02_qkv_proj", out, 1))
    if hasattr(L0.self_attn, "query_layernorm"):
        L0.self_attn.query_layernorm.register_forward_hook(lambda mod, inp, out: cap("X03_q_after_qknorm", out, 2))
    L0.self_attn.o_proj.register_forward_hook(lambda mod, inp, out: cap("X04_o_proj", out, 1))
    L0.self_attn.register_forward_hook(lambda mod, inp, out: cap("X05_attn_out", out, 1))
    if hasattr(L0, "post_attention_layernorm"):
        L0.post_attention_layernorm.register_forward_hook(lambda mod, inp, out: cap("X06_post_attn_ln", out, 1))
    L0.mlp.register_forward_hook(lambda mod, inp, out: cap("X07_mlp_out", out, 1))

    b, d = 1, cfg.head_dim
    torch.manual_seed(123)
    inputs_embeds = torch.randn(b, L, cfg.hidden_size, device=dev, dtype=DTYPE)
    mask = make_mask(b, L, dev)
    cos, sin = make_rope(b, L, d, dev)
    pos = torch.arange(L, device=dev).unsqueeze(0).expand(b, L).contiguous()

    def run_once():
        captured.clear()
        with torch.no_grad():
            out = m(inputs_embeds=inputs_embeds, attention_mask=mask, position_ids=pos,
                    custom_pos_emb=(cos, sin), mode=MODE, use_cache=False, return_dict=True)
        sp = get_parallel_state().sp_group if world > 1 else None

        def full(h, dim):  # gather seq shards along `dim` -> trim to true L
            if world == 1:
                return h.detach().float().cpu()
            g = [torch.empty_like(h) for _ in range(world)]
            dist.all_gather(g, h.contiguous(), group=sp)
            o = torch.cat(g, dim=dim)
            sl = [slice(None)] * o.dim()
            sl[dim] = slice(0, L)
            return o[tuple(sl)].detach().float().cpu()

        detail = {k: full(*captured[k]) for k in captured if isinstance(k, str)}
        layers = [full(*captured[i]) for i in range(LAYERS)]
        fin = full(out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0], 1)
        return layers, detail, fin

    layersA, detailA, finalA = run_once()
    layersB, detailB, finalB = run_once()

    if rank == 0:
        det = max((a - b_).abs().max().item() for a, b_ in zip(layersA, layersB))
        print(f"[{TAG}] world={world} DETERMINISM (2 passes) max|Δ|= {det:.3e}", flush=True)

        def stat(a, b):
            d = (a - b).abs()
            return d.max().item(), d.mean().item(), (d > 0).float().mean().item() * 100.0

        def line(label, a, b):
            mx, mn, pc = stat(a, b)
            rel = mx / (a.abs().max().item() + 1e-9)
            print(f"   {label:16s}: max={mx:.3e} rel={rel:.2e} mean={mn:.3e} %elems_differ={pc:6.3f}", flush=True)

        if world == 1:
            torch.save({"layers": layersA, "detail": detailA, "final": finalA}, REF_PATH)
            print(f"[{TAG}] saved reference (world=1).", flush=True)
        elif os.path.exists(REF_PATH):
            ref = torch.load(REF_PATH)
            print(f"[{TAG}] ========== sp=1 vs sp={world} ==========", flush=True)
            print("LAYER-0 TRAJECTORY (op by op):", flush=True)
            for k in sorted(k for k in detailA if k.startswith("X")):
                line(k[3:], ref["detail"][k], detailA[k])
            print("PER-LAYER SUB-STEPS (attn / mlp, every layer):", flush=True)
            for k in sorted(k for k in detailA if k.startswith("Y")):
                line(k[1:], ref["detail"][k], detailA[k])
            print("ACCUMULATION (residual stream, layer output):", flush=True)
            for i in range(LAYERS):
                line(f"layer{i:02d}", ref["layers"][i], layersA[i])
            line("FINAL", ref["final"], finalA)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
