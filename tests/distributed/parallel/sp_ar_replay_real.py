"""REAL Qwen3-4B replay-path parity — reproduces the e2e DRPO ratio≈0.11 gap.

The e2e (VeOmni backend, sp=2) shows ratio = exp(new_logp - old_logp) ≈ 0.11
stable, i.e. the train-side replay logprob is ~2.2 nats/token BELOW SGLang's,
while the FSDP baseline has ratio=1.0000. All isolated SP tests pass on a 2-layer
toy; this probe runs the EXACT replay path (`_replay_aware_forward` chunked head +
the SP decoder wrapper) on the REAL Qwen3-4B-Base over a B=1 left-padded prompt +
**greedy-generated (high-confidence)** response — random tokens (logp≈-13, near the
uniform floor) are INSENSITIVE to attention corruption and hide the bug; the model's
own greedy picks (logp≈-0.5) expose any distribution flattening.

  world=1            -> plain HF: greedy-gen the response, save it + the ref logp.
  world=1 SP_FSDP=1  -> FSDP2 + sp=1 (no SP attn/wrapper): isolates FSDP+replay.
  world=2            -> FSDP2 + sp=2 (full e2e replay path): the suspect.

  ref == fsdp1 == sp2 -> replay path is fine; gap is elsewhere (weight sync).
  ref == fsdp1 != sp2 -> SP attention/wrapper bug.

  MODEL=/dockerdata/Qwen3-4B-Base torchrun --nproc_per_node=1 --master_port=29810 \
      tests/distributed/parallel/sp_ar_replay_real.py                       # ref+gen
  MODEL=... SP_FSDP=1 torchrun --nproc_per_node=1 --master_port=29811 ...   # fsdp1
  MODEL=... torchrun --nproc_per_node=2 --master_port=29812 ...             # sp2
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
SP_FSDP = int(os.environ.get("SP_FSDP", "0"))
PROMPT = int(os.environ.get("PROMPT", "200"))    # real prompt tokens
PADL = int(os.environ.get("PADL", "48"))         # left-pad amount
RESP = int(os.environ.get("RESP", "128"))        # greedy response tokens
IO = "/tmp/ar_replay_real_io.pt"
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

    is_ref = (world == 1 and not SP_FSDP)
    if is_ref:
        # Greedy-generate a high-confidence response from a fixed random prompt,
        # BEFORE patching forward. Save prompt+response for the other configs.
        V = int(m.config.vocab_size)
        torch.manual_seed(7)
        prompt_real = torch.randint(1, V, (1, PROMPT), device=dev)
        with torch.no_grad():
            gen = m.generate(prompt_real, max_new_tokens=RESP, do_sample=False,
                             use_cache=True, pad_token_id=0)
        resp = gen[:, PROMPT:PROMPT + RESP].contiguous()
        torch.save({"prompt_real": prompt_real.cpu(), "resp": resp.cpu()}, IO)
    else:
        io = torch.load(IO, map_location=dev)
        prompt_real, resp = io["prompt_real"].to(dev), io["resp"].to(dev)

    m.forward = MethodType(_replay_aware_forward, m)   # dual-mode replay forward

    use_sp = world > 1
    if use_sp:
        set_attn(m)
        for layer in m.model.layers:
            fully_shard(layer, mesh=get_parallel_state().fsdp_mesh)
        fully_shard(m, mesh=get_parallel_state().fsdp_mesh)
        apply_sequence_parallelism(m, world)
    elif SP_FSDP:
        set_attn(m)
        for layer in m.model.layers:
            fully_shard(layer, mesh=get_parallel_state().fsdp_mesh)
        fully_shard(m, mesh=get_parallel_state().fsdp_mesh)
    m.eval()

    P = prompt_real.shape[1]
    RES = resp.shape[1]
    prompt_ids = torch.cat([torch.zeros(1, PADL, device=dev, dtype=torch.long), prompt_real], dim=1)
    prompt_mask = torch.cat([torch.zeros(1, PADL, device=dev, dtype=torch.long),
                             torch.ones(1, P, device=dev, dtype=torch.long)], dim=1)
    prompt_len = prompt_ids.shape[1]
    resp_mask = torch.ones(1, RES, device=dev, dtype=torch.long)

    full_ids = torch.cat([prompt_ids, resp], dim=1)
    full_mask = torch.cat([prompt_mask, resp_mask], dim=1)
    position_ids = (full_mask.long().cumsum(-1) - 1).clamp(min=0)

    with torch.no_grad():
        per_token = m(
            input_ids=full_ids, attention_mask=full_mask, position_ids=position_ids,
            response_tokens=resp, prompt_len=prompt_len, temperature=1.0,
            autocast_dtype=torch.bfloat16,
        )  # [1, RES] fp32

    tag = f"sp{world}" if use_sp else ("fsdp1" if SP_FSDP else "ref")
    if rank == 0:
        lp = per_token[0].float().cpu()
        print(f"[{tag}] logp shape={tuple(lp.shape)} mean={lp.mean():.4f} (greedy=confident)", flush=True)
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
