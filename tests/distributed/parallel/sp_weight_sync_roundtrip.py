"""VeOmni weight-sync EXTRACTION round-trip — does sglang get the train model?

The e2e shows ratio = exp(new_logp - old_logp) ≈ 0.05-0.11 (FSDP baseline = 1.0)
at BOTH sp=1 and sp=2, while the SP replay forward is provably exact. That means
old_logp (SGLang) ≠ new_logp (train replay) at the SAME initial weights — i.e.
the weights SGLang receives via TensorWeightSync don't reproduce the train model.

This probe tests exactly that, with NO checkpoint and NO SGLang: veomni_parallelize
a model, compute replay logp on the SHARDED train model (== new_logp), then extract
weights via the sync's EXACT path (raw_state_dict -> _to_full_tensor), load them into
a FRESH plain model, and recompute logp (== what SGLang would compute from old_logp's
weights). Faithful extraction => identical logp (ratio 1). A corrupt/scaled param =>
ratio != 1, and per-param scale diff localizes it.

  MODEL=/dockerdata/Qwen3-4B-Base torchrun --nproc_per_node=2 --master_port=29820 \
      tests/distributed/parallel/sp_weight_sync_roundtrip.py        # dp=2, sp=1 (the bug)
"""
import os
from types import MethodType

import torch
import torch.distributed as dist

from unirl.train.backend.veomni import _compat
_compat.ensure_installed()
from veomni.distributed.parallel_state import init_parallel_state, get_parallel_state
from unirl.train.backend.veomni.wrap import veomni_parallelize
from unirl.models.qwen3.ar import _replay_aware_forward
from unirl.utils.peft_merge import raw_state_dict

MODEL = os.environ.get("MODEL", "/dockerdata/Qwen3-4B-Base")
SP = int(os.environ.get("SP", "1"))           # ulysses degree (1 = reproduce the no-SP bug)
LAYERS = int(os.environ.get("LAYERS", "0"))    # 0 = full model; else truncate for speed


def build_logp(model, ids, mask, pos, resp, prompt_len):
    with torch.no_grad():
        return model(input_ids=ids, attention_mask=mask, position_ids=pos,
                     response_tokens=resp, prompt_len=prompt_len, temperature=1.0,
                     autocast_dtype=torch.bfloat16)[0].float().cpu()


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    lr = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(lr)
    dev = torch.device("cuda", lr)
    init_parallel_state(dp_size=world // SP, ulysses_size=SP, dp_mode="fsdp2", device_type="cuda")

    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(MODEL)
    if LAYERS:
        cfg.num_hidden_layers = LAYERS
    cfg.use_cache = False

    # Meta-init + veomni_parallelize (the e2e backend path); to_empty materializes
    # random weights — fine, we test extraction faithfulness, not absolute values.
    with torch.device("meta"):
        m = AutoModelForCausalLM.from_config(cfg)
    veomni_parallelize(m, block_class_names=("Qwen3DecoderLayer",),
                       param_dtype="bf16", master_dtype="fp32",
                       reshard_after_forward=True, activation_checkpointing=False)
    m.forward = MethodType(_replay_aware_forward, m)
    m.eval()

    # Confident tokens: greedy from a fixed random prompt on the sharded model.
    V = int(cfg.vocab_size)
    torch.manual_seed(3)
    P, R = 64, 48
    prompt = torch.randint(1, V, (1, P), device=dev)
    ids = torch.cat([prompt, torch.zeros(1, R, device=dev, dtype=torch.long)], dim=1)
    # cheap greedy fill via the chunked replay is awkward; just use random resp —
    # extraction corruption (a scaled/summed param) shows on ANY fixed tokens.
    resp = torch.randint(1, V, (1, R), device=dev)
    full_ids = torch.cat([prompt, resp], dim=1)
    full_mask = torch.ones(1, P + R, device=dev, dtype=torch.long)
    pos = (full_mask.long().cumsum(-1) - 1).clamp(min=0)

    logp_train = build_logp(m, full_ids, full_mask, pos, resp, P)  # the train model's view

    # Extract via the sync's EXACT path; gather full tensors to rank 0.
    extracted = {name: t.detach().float().cpu() for name, t in raw_state_dict(m)}

    if rank == 0:
        fresh = AutoModelForCausalLM.from_config(cfg).to(dev).to(torch.bfloat16)
        missing, unexpected = fresh.load_state_dict(
            {k: v.to(torch.bfloat16) for k, v in extracted.items()}, strict=False)
        fresh.forward = MethodType(_replay_aware_forward, fresh)
        fresh.eval()
        logp_fresh = build_logp(fresh, full_ids, full_mask, pos, resp, P)
        d = logp_fresh - logp_train
        print(f"[roundtrip sp={SP} dp={world // SP}] n_params_extracted={len(extracted)} "
              f"missing={len(missing)} unexpected={len(unexpected)}", flush=True)
        print(f"[roundtrip] logp_train.mean={logp_train.mean():.4f} logp_fresh.mean={logp_fresh.mean():.4f} "
              f"mean(Δ)={d.mean():.4f} mean|Δ|={d.abs().mean():.4f} max|Δ|={d.abs().max():.4f} "
              f"ratio=exp(meanΔ)={float(torch.exp(d.mean())):.4f} "
              f"{'PASS(sync faithful)' if d.abs().mean() < 0.02 else 'FAIL(sync corrupts)'}", flush=True)
        if missing:
            print("  MISSING (sglang keeps its own):", missing[:8], flush=True)
        if unexpected:
            print("  UNEXPECTED (sglang ignores):", unexpected[:8], flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
