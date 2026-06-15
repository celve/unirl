"""REAL Qwen3-4B replay-path parity — reproduces the e2e DRPO ratio≈0.11 gap.

The e2e (VeOmni backend, sp=2) shows ratio = exp(new_logp - old_logp) ≈ 0.11
stable, i.e. the train-side replay logprob is ~2.2 nats/token BELOW SGLang's,
while the FSDP baseline has ratio=1.0000. All isolated SP tests pass on a 2-layer
toy; this probe runs the EXACT replay path (`_replay_aware_forward` chunked head +
the SP decoder wrapper) on the REAL Qwen3-4B-Base over a realistic B=1 left-padded
prompt+response, to localize the 2.2-nat gap to a layer:

  world=1            -> plain HF (no FSDP, no SP): the ground-truth reference.
  world=1 SP_FSDP=1  -> FSDP2 + sp=1 (no SP attn/wrapper): isolates FSDP+replay.
  world=2            -> FSDP2 + sp=2 (full e2e replay path): the suspect.

  ref == fsdp1 == sp2 -> replay path is fine; gap is elsewhere (weight sync).
  ref == fsdp1 != sp2 -> SP attention/wrapper bug.
  ref != fsdp1        -> FSDP+_replay_aware bug (not SP).

  MODEL=/dockerdata/Qwen3-4B-Base torchrun --nproc_per_node=1 --master_port=29800 \
      tests/distributed/parallel/sp_ar_replay_real.py                       # ref
  MODEL=... SP_FSDP=1 torchrun --nproc_per_node=1 --master_port=29801 ...   # fsdp1
  MODEL=... torchrun --nproc_per_node=2 --master_port=29802 ...             # sp2
"""
import os
from types import MethodType

import torch
import torch.distributed as dist

from unirl.train.backend.veomni import _compat
_compat.ensure_installed()
_compat.ensure_attention_patch_installed()
from veomni.distributed.parallel_state import init_parallel_state, get_parallel_state
from unirl.train.backend.veomni.sp import apply_sequence_parallelism
from unirl.train.backend.veomni.sp.ar import SP_ATTN_IMPL
from unirl.models.qwen3.ar import _replay_aware_forward
from torch.distributed.fsdp import fully_shard

MODEL = os.environ.get("MODEL", "/dockerdata/Qwen3-4B-Base")
SP_FSDP = int(os.environ.get("SP_FSDP", "0"))   # world=1 only: wrap FSDP at sp=1
PROMPT = int(os.environ.get("PROMPT", "400"))    # real prompt tokens (left-padded)
PADL = int(os.environ.get("PADL", "48"))         # left-pad amount
RESP = int(os.environ.get("RESP", "300"))        # response tokens
REF = "/tmp/ar_replay_real_ref.pt"


def set_attn(m):
    m.config._attn_implementation = SP_ATTN_IMPL
    for mod in m.modules():
        c = getattr(mod, "config", None)
        if c is not None and hasattr(c, "_attn_implementation"):
            c._attn_implementation = SP_ATTN_IMPL


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    lr = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(lr)
    dev = torch.device("cuda", lr)
    init_parallel_state(dp_size=1, ulysses_size=world, dp_mode="fsdp2", device_type="cuda")

    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(dev)
    m.forward = MethodType(_replay_aware_forward, m)   # the dual-mode replay forward

    use_sp = world > 1
    if use_sp:
        set_attn(m)
        for layer in m.model.layers:
            fully_shard(layer, mesh=get_parallel_state().fsdp_mesh)
        fully_shard(m, mesh=get_parallel_state().fsdp_mesh)
        apply_sequence_parallelism(m, world)
    elif SP_FSDP:
        set_attn(m)                                    # same kernel, sp disabled (ulysses_size=1)
        for layer in m.model.layers:
            fully_shard(layer, mesh=get_parallel_state().fsdp_mesh)
        fully_shard(m, mesh=get_parallel_state().fsdp_mesh)
    m.eval()

    # B=1 left-padded prompt + right-attached response (the train micro geometry).
    V = int(m.config.vocab_size)
    torch.manual_seed(7)
    prompt_real = torch.randint(1, V, (1, PROMPT), device=dev)
    prompt_ids = torch.cat([torch.zeros(1, PADL, device=dev, dtype=torch.long), prompt_real], dim=1)
    prompt_mask = torch.cat([torch.zeros(1, PADL, device=dev, dtype=torch.long),
                             torch.ones(1, PROMPT, device=dev, dtype=torch.long)], dim=1)
    prompt_len = prompt_ids.shape[1]
    resp = torch.randint(1, V, (1, RESP), device=dev)
    resp_mask = torch.ones(1, RESP, device=dev, dtype=torch.long)

    full_ids = torch.cat([prompt_ids, resp], dim=1)
    full_mask = torch.cat([prompt_mask, resp_mask], dim=1)
    position_ids = (full_mask.long().cumsum(-1) - 1).clamp(min=0)

    with torch.no_grad():
        per_token = m(
            input_ids=full_ids, attention_mask=full_mask, position_ids=position_ids,
            response_tokens=resp, prompt_len=prompt_len, temperature=1.0,
            autocast_dtype=torch.bfloat16,
        )  # [1, RESP] fp32

    tag = f"sp{world}" if use_sp else ("fsdp1" if SP_FSDP else "ref")
    if rank == 0:
        lp = per_token[0].float().cpu()
        print(f"[{tag}] logp shape={tuple(lp.shape)} mean={lp.mean():.4f} sum={lp.sum():.2f}", flush=True)
        if tag == "ref":
            torch.save(lp, REF)
            print("saved ref", flush=True)
        elif os.path.exists(REF):
            ref = torch.load(REF)
            d = (lp - ref)
            print(f"[{tag} vs ref] mean(Δlogp)={d.mean():.4f} mean|Δ|={d.abs().mean():.4f} "
                  f"max|Δ|={d.abs().max():.4f} ratio=exp(meanΔ)={float(torch.exp(d.mean())):.4f} "
                  f"{'PASS' if d.abs().mean() < 0.05 else 'FAIL'}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
