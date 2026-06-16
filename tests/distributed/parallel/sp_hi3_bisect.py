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
    cfg.num_experts = 8
    if isinstance(getattr(cfg, "moe_topk", None), list):
        cfg.moe_topk = [min(x, 4) for x in cfg.moe_topk]
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(0)
    return ModelCls(cfg).to(dev).to(DTYPE), cfg


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
    if world > 1:
        apply_sequence_parallelism(m, world)
    m.eval()

    # per-layer capture hooks + sub-module (attn / mlp) hooks for the first layers
    captured = {}
    for i, layer in enumerate(m.layers):
        layer.register_forward_hook(lambda mod, inp, out, i=i: captured.__setitem__(i, out[0] if isinstance(out, (tuple, list)) else out))
    # Fine-grained op-by-op trajectory through layer 0: see identity (=0) at the
    # input, where the first non-zero appears (the qkv_proj GEMM), and how the
    # attention propagates it. All these tensors carry seq on dim=1 (gatherable).
    def t(out):
        return out[0] if isinstance(out, (tuple, list)) else out

    L0 = m.layers[0]
    L0.register_forward_pre_hook(lambda mod, args: captured.__setitem__("00_layer_input", args[0]))
    for nm in ("input_layernorm", "ln1", "attention_layernorm"):
        if hasattr(L0, nm):
            getattr(L0, nm).register_forward_hook(lambda mod, inp, out: captured.__setitem__("01_input_ln", t(out)))
            break
    L0.self_attn.qkv_proj.register_forward_hook(lambda mod, inp, out: captured.__setitem__("02_qkv_proj", t(out)))
    L0.self_attn.o_proj.register_forward_hook(lambda mod, inp, out: captured.__setitem__("03_o_proj", t(out)))
    L0.self_attn.register_forward_hook(lambda mod, inp, out: captured.__setitem__("04_attn_out", t(out)))
    for nm in ("post_attention_layernorm", "ln2", "post_attn_layernorm"):
        if hasattr(L0, nm):
            getattr(L0, nm).register_forward_hook(lambda mod, inp, out: captured.__setitem__("05_post_attn_ln", t(out)))
            break
    L0.mlp.register_forward_hook(lambda mod, inp, out: captured.__setitem__("06_mlp_out", t(out)))

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

        def full(h):  # gather seq shards -> [B, L', D] -> trim to L
            if world == 1:
                return h.detach().float().cpu()
            g = [torch.empty_like(h) for _ in range(world)]
            dist.all_gather(g, h.contiguous(), group=sp)
            return torch.cat(g, dim=1)[:, :L, :].detach().float().cpu()

        subs = {k: full(captured[k]) for k in captured if isinstance(k, str)}
        return [full(captured[i]) for i in range(LAYERS)], subs, full(out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0])

    layersA, subsA, finalA = run_once()
    layersB, subsB, finalB = run_once()

    if rank == 0:
        det = max((a - b_).abs().max().item() for a, b_ in zip(layersA, layersB))
        print(f"[{TAG}] world={world} DETERMINISM max|Δ| over 2 passes (per-layer)= {det:.3e}", flush=True)
        if world == 1:
            torch.save({"layers": layersA, "subs": subsA, "final": finalA}, REF_PATH)
            print(f"[{TAG}] saved reference (world=1).", flush=True)
        elif os.path.exists(REF_PATH):
            ref = torch.load(REF_PATH)
            print(f"[{TAG}] TRAJECTORY through layer 0 (op by op), sp=1 vs sp={world}:", flush=True)
            for k in sorted(subsA):
                a, bb = ref["subs"][k], subsA[k]
                diff = (a - bb).abs().max().item()
                rel = diff / (a.abs().max().item() + 1e-9)
                print(f"   {k:16s}: max|Δ|={diff:.3e}  rel={rel:.3e}", flush=True)
            print(f"[{TAG}] ACCUMULATION across layers (layer output), sp=1 vs sp={world}:", flush=True)
            for i in range(LAYERS):
                a, bb = ref["layers"][i], layersA[i]
                diff = (a - bb).abs().max().item()
                rel = diff / (a.abs().max().item() + 1e-9)
                print(f"   layer{i:2d}: max|Δ|={diff:.3e}  rel={rel:.3e}", flush=True)
            fa, fb = ref["final"], finalA
            fdiff = (fa - fb).abs().max().item()
            frel = fdiff / (fa.abs().max().item() + 1e-9)
            print(f"   FINAL : max|Δ|={fdiff:.3e}  rel={frel:.3e}  {'PASS' if frel < 1e-3 else 'DIVERGENT'}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
