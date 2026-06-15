"""HunyuanImage3 (HI3) Ulysses SP parity: sp=1 reference vs sp=N.

Validates the real integration — the HI3 adapter
(``unirl.train.backend.veomni.sp.hunyuan_image3``): the all-to-all monkey-patched
into ``HunyuanImage3SDPAAttention`` + the decoder slice/gather wrapper — on a
shrunken real HI3 decoder (``HunyuanImage3Model``, trust_remote_code, 2 layers /
8 experts, random weights, real head config 32 Q / 8 KV / head_dim 128).

Both runs use identical weights (saved state dict). At sp=1 ``apply_sequence_
parallelism`` is a no-op (the original attention + a 4D causal-image mask); at
sp=N the adapter shards the sequence. Run sp=1 first (saves weights + ref), then
sp=N (``L=66`` -> sp=2 is divisible, sp=4 exercises the wrapper's auto-pad):

  HI3_MODEL_PATH=<dir with modeling_hunyuan_image_3.py + config> \
  torchrun --nproc_per_node=1 --master_port=29570 sp_hi3_parity.py   # ref
  torchrun --nproc_per_node=2 --master_port=29571 sp_hi3_parity.py   # compare (no pad)
  torchrun --nproc_per_node=4 --master_port=29572 sp_hi3_parity.py   # compare (auto-pad)

fp32; a real SP bug shows a large systematic diff (expect ~1e-6).
"""
import os
import sys

import torch
import torch.distributed as dist

from unirl.train.backend.veomni import _compat
_compat.ensure_installed()
from veomni.distributed.parallel_state import init_parallel_state, get_parallel_state  # noqa: E402
from unirl.train.backend.veomni.sp import apply_sequence_parallelism  # noqa: E402

HI3_DIR = os.environ.get(
    "HI3_MODEL_PATH",
    "/apdcephfs_fsgm3/share_305110755/hunyuan/linyuwu/HunyuanImage-3-Instruct",
)
SD_PATH = "/tmp/hi3_sd.pt"


def build_model(dev):
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    CfgCls = get_class_from_dynamic_module(
        "configuration_hunyuan_image_3.HunyuanImage3Config", HI3_DIR
    )
    ModelCls = get_class_from_dynamic_module(
        "modeling_hunyuan_image_3.HunyuanImage3Model", HI3_DIR
    )
    cfg = CfgCls.from_pretrained(HI3_DIR)
    cfg.num_hidden_layers = 2
    cfg.num_experts = 8
    if isinstance(getattr(cfg, "moe_topk", None), list):
        cfg.moe_topk = [min(x, 4) for x in cfg.moe_topk]
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(0)
    return ModelCls(cfg).to(dev).to(torch.float32), cfg


def make_mask(b, l, dev):
    m = torch.ones(l, l, dtype=torch.bool, device=dev).tril()
    a, c = l // 4, (3 * l) // 4
    m[a:c, a:c] = True  # image block = bidirectional (non-causal)
    return m.view(1, 1, l, l).expand(b, 1, l, l).contiguous()


def make_rope(b, l, d, dev, dtype):
    pos = torch.arange(l, device=dev).float()
    i = torch.arange(0, d, 2, device=dev).float()
    ang = torch.outer(pos, 1.0 / (10000 ** (i / d)))
    emb = torch.cat([ang, ang], -1)
    return (
        emb.cos()[None].expand(b, l, d).to(dtype).contiguous(),
        emb.sin()[None].expand(b, l, d).to(dtype).contiguous(),
    )


def main():
    if not os.path.isdir(HI3_DIR):
        print(f"SKIP: HI3 modeling dir not found at {HI3_DIR} (set HI3_MODEL_PATH)", flush=True)
        return

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    lr = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(lr)
    dev = torch.device("cuda", lr)
    init_parallel_state(dp_size=1, ulysses_size=world, dp_mode="fsdp2", device_type="cuda")

    m, cfg = build_model(dev)
    if os.path.exists(SD_PATH):
        m.load_state_dict(torch.load(SD_PATH, map_location=dev))
    elif rank == 0:
        torch.save(m.state_dict(), SD_PATH)
    dist.barrier()

    if world > 1:
        apply_sequence_parallelism(m, world)  # installs the HI3 attn patch + decoder wrapper
    m.eval()

    b, l, d = 1, 66, cfg.head_dim
    torch.manual_seed(123)
    inputs_embeds = torch.randn(b, l, cfg.hidden_size, device=dev, dtype=torch.float32)
    mask = make_mask(b, l, dev)
    cos, sin = make_rope(b, l, d, dev, torch.float32)
    pos = torch.arange(l, device=dev).unsqueeze(0).expand(b, l).contiguous()

    with torch.no_grad():
        out = m(
            inputs_embeds=inputs_embeds,
            attention_mask=mask,
            position_ids=pos,
            custom_pos_emb=(cos, sin),
            mode="gen_text",
            use_cache=False,
            return_dict=True,
        )
        hidden = out.last_hidden_state  # [B, L, hidden]

    if rank == 0:
        ps = get_parallel_state()
        print(
            f"world={world} sp_size={ps.sp_size} ulysses={ps.ulysses_enabled} "
            f"hidden={tuple(hidden.shape)}",
            flush=True,
        )
        torch.save(hidden.float().cpu(), f"/tmp/hi3_hidden_w{world}.pt")
        ref_path = "/tmp/hi3_hidden_w1.pt"
        if world > 1 and os.path.exists(ref_path):
            a, b_ = torch.load(ref_path), hidden.float().cpu()
            diff = (a - b_).abs().max().item()
            relerr = diff / (a.abs().max().item() + 1e-9)
            print(
                f"PARITY sp=1 vs sp={world}: max|Δhidden|={diff:.3e} relerr={relerr:.3e} "
                f"{'PASS' if relerr < 1e-3 else 'CHECK'}",
                flush=True,
            )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
