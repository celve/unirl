"""At sp=1, is our SP adapter equivalent to the ORIGINAL HI3 attention?

Runs on ONE rank. First does a forward with the *unpatched* HI3 model (the
reference). Then forces our SP code path ON at world=1 — the Ulysses all-to-alls
become 1-rank identities and slice/gather become no-ops, so the ONLY thing that
can differ from the original is our rewrite itself (the extra reshapes / contiguous
/ op order around SDPA). Same M (=L) on both sides, so this isolates "is the
adapter a faithful identity?" from the M-dependence that bites at sp>1.

  HI3_MODEL_PATH=<dir> DTYPE=bf16 LAYERS=8 SEQLEN=64 \
  torchrun --nproc_per_node=1 --master_port=29590 sp_hi3_identity.py
"""
import os

import torch
import torch.distributed as dist

from unirl.train.backend.veomni import _compat

_compat.ensure_installed()
import veomni.distributed.parallel_state as PS  # noqa: E402
from veomni.distributed.parallel_state import init_parallel_state  # noqa: E402
from unirl.train.backend.veomni.sp import apply_sequence_parallelism  # noqa: E402

HI3_DIR = os.environ.get("HI3_MODEL_PATH", "/dockerdata/HunyuanImage-3-Instruct")
DTYPE = {"bf16": torch.bfloat16, "fp32": torch.float32}[os.environ.get("DTYPE", "bf16")]
LAYERS = int(os.environ.get("LAYERS", "8"))
L = int(os.environ.get("SEQLEN", "64"))


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


class _FakePS:
    """Force the SP code path ON with a 1-rank (identity) group."""

    ulysses_enabled = True
    ulysses_group = None  # set to WORLD at runtime
    sp_group = None
    ulysses_size = 1
    sp_size = 1


def main():
    if not os.path.isdir(HI3_DIR):
        print(f"SKIP: HI3 dir not found {HI3_DIR}", flush=True)
        return
    dist.init_process_group("nccl")
    torch.cuda.set_device(0)
    dev = torch.device("cuda", 0)
    init_parallel_state(dp_size=1, ulysses_size=1, dp_mode="fsdp2", device_type="cuda")

    m, cfg = build_model(dev)
    m.eval()

    captured = {}

    def grab(out):
        h = out[0] if isinstance(out, (tuple, list)) else out
        return h.detach().float().cpu()

    for i, layer in enumerate(m.layers):
        layer.register_forward_hook(lambda mod, inp, out, i=i: captured.__setitem__(i, grab(out)))
    m.layers[0].self_attn.qkv_proj.register_forward_hook(lambda mod, inp, out: captured.__setitem__("qkv", grab(out)))
    m.layers[0].self_attn.register_forward_hook(lambda mod, inp, out: captured.__setitem__("attn", grab(out)))

    torch.manual_seed(123)
    inputs_embeds = torch.randn(1, L, cfg.hidden_size, device=dev, dtype=DTYPE)
    mask = make_mask(1, L, dev)
    cos, sin = make_rope(1, L, cfg.head_dim, dev)
    pos = torch.arange(L, device=dev).unsqueeze(0).contiguous()

    def fwd():
        captured.clear()
        with torch.no_grad():
            o = m(inputs_embeds=inputs_embeds, attention_mask=mask, position_ids=pos,
                  custom_pos_emb=(cos, sin), mode="gen_text", use_cache=False, return_dict=True)
        snap = {k: v for k, v in captured.items()}
        snap["final"] = grab(o.last_hidden_state if hasattr(o, "last_hidden_state") else o[0])
        return snap

    # 1) ORIGINAL HI3 (no adapter installed)
    ref = fwd()

    # 2) force our SP path ON at world=1, then install the adapter
    _FakePS.ulysses_group = dist.group.WORLD
    _FakePS.sp_group = dist.group.WORLD
    PS.get_parallel_state = lambda: _FakePS()
    apply_sequence_parallelism(m, 1)
    sp = fwd()

    dt = os.environ.get("DTYPE", "bf16")
    print(f"=== sp=1 SP-path vs ORIGINAL HI3  ({dt}, {LAYERS} layers, L={L}) ===", flush=True)

    def show(label, a, b):
        d = (a - b).abs()
        mx = d.max().item()
        pc = (d > 0).float().mean().item() * 100.0
        tag = "IDENTICAL" if mx == 0.0 else ("~1ULP" if mx < 0.1 else "DIFFERENT")
        print(f"  {label:10s}: max|Δ|={mx:.3e}  %elems_differ={pc:6.3f}  {tag}", flush=True)

    show("L0.qkv", ref["qkv"], sp["qkv"])
    show("L0.attn", ref["attn"], sp["attn"])
    for i in range(LAYERS):
        show(f"layer{i:02d}", ref[i], sp[i])
    show("FINAL", ref["final"], sp["final"])

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
