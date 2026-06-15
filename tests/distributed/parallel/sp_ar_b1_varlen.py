"""AR Ulysses SP parity for B=1 LEFT-PADDED varlen — the train micro geometry.

This is the case the e2e actually runs (TrainStack micro_batch_size=1 -> B=1) and
the one the B=2 varlen test (`sp_ar_varlen.py`) MISSED. With a single padded
sample, cumsum position_ids carries repeated zeros; VeOmni's SP attention gathers
q/k/v to the full sequence but forwards the *sliced* position_ids to flash-attn,
whose varlen inference reads the resets as bogus sequence boundaries and corrupts
the logprobs (was relerr ~0.25). `sp/ar.py` fixes it by stripping the pad to a
dense, monotonic-position_ids sequence and re-padding the hidden states.

Params via env: SP_B (default 1), SP_HEADS ('big'=32q/8kv/d128 [Qwen3-4B] or
'small'), SP_PAD (default 1 = left-padded varlen; 0 = dense).

  torchrun --nproc_per_node=1 --master_port=29590 tests/distributed/parallel/sp_ar_b1_varlen.py
  torchrun --nproc_per_node=2 --master_port=29591 tests/distributed/parallel/sp_ar_b1_varlen.py
"""
import os
import torch
import torch.distributed as dist

from unirl.train.backend.veomni import _compat
_compat.ensure_installed()
_compat.ensure_attention_patch_installed()
from veomni.distributed.parallel_state import init_parallel_state, get_parallel_state
from unirl.train.backend.veomni.sp import apply_sequence_parallelism
from unirl.train.backend.veomni.sp.ar import SP_ATTN_IMPL

B = int(os.environ.get("SP_B", "1"))
HEADS = os.environ.get("SP_HEADS", "big")
PAD = int(os.environ.get("SP_PAD", "1"))
SD_PATH = f"/tmp/ar_b1v_sd_{HEADS}_{B}.pt"


def build_model(dev):
    from transformers import AutoModelForCausalLM, Qwen3Config
    if HEADS == "big":  # Qwen3-4B head geometry
        cfg = Qwen3Config(vocab_size=1000, hidden_size=512, intermediate_size=1024,
                          num_hidden_layers=2, num_attention_heads=32, num_key_value_heads=8,
                          max_position_embeddings=1024, head_dim=128)
    else:
        cfg = Qwen3Config(vocab_size=1000, hidden_size=256, intermediate_size=512,
                          num_hidden_layers=2, num_attention_heads=8, num_key_value_heads=4,
                          max_position_embeddings=1024, head_dim=32)
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(cfg).to(dev).to(torch.bfloat16)


def set_attn(m):
    m.config._attn_implementation = SP_ATTN_IMPL
    for mod in m.modules():
        cfg = getattr(mod, "config", None)
        if cfg is not None and hasattr(cfg, "_attn_implementation"):
            cfg._attn_implementation = SP_ATTN_IMPL


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    lr = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(lr)
    dev = torch.device("cuda", lr)
    init_parallel_state(dp_size=1, ulysses_size=world, dp_mode="fsdp2", device_type="cuda")

    m = build_model(dev)
    if os.path.exists(SD_PATH):
        m.load_state_dict(torch.load(SD_PATH, map_location=dev))
    elif rank == 0:
        torch.save(m.state_dict(), SD_PATH)
    dist.barrier()
    set_attn(m)
    if world > 1:
        apply_sequence_parallelism(m, world)
    m.eval()

    L = 418
    # Default real_len is ODD (317): the stripped span is not divisible by sp,
    # so the pad-strip path must round it up itself with monotonic position_ids
    # (an even default would mask the real e2e crash). Override with SP_REAL.
    real_lens = ([int(os.environ.get("SP_REAL", "317"))] if B == 1 else [318, 380][:B]) if PAD else [L] * B
    pad_id = 0
    torch.manual_seed(123)
    ids = torch.full((B, L), pad_id, device=dev, dtype=torch.long)
    mask = torch.zeros((B, L), device=dev, dtype=torch.long)
    for b, n in enumerate(real_lens):
        ids[b, L - n:] = torch.randint(1, 1000, (n,), device=dev)
        mask[b, L - n:] = 1
    pos = (mask.long().cumsum(-1) - 1).clamp(min=0)
    mask_arg = mask if PAD else None

    with torch.no_grad():
        hidden = m.model(input_ids=ids, attention_mask=mask_arg, position_ids=pos,
                         use_cache=False, return_dict=True).last_hidden_state
        logits = m.lm_head(hidden).float()
        chosen = logits.log_softmax(-1)[:, :-1].gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    valid = (mask[:, :-1] * mask[:, 1:]).bool()

    tag = f"{HEADS}_B{B}_pad{PAD}"
    if rank == 0:
        torch.save({"chosen": chosen.cpu(), "valid": valid.cpu()}, f"/tmp/ar_b1v_{tag}_w{world}.pt")
        ref = f"/tmp/ar_b1v_{tag}_w1.pt"
        if world > 1 and os.path.exists(ref):
            r = torch.load(ref)
            a = r["chosen"][r["valid"]]; b = chosen.cpu()[valid.cpu()]
            diff = (a - b).abs().max().item(); relerr = diff / (a.abs().max().item() + 1e-9)
            print(f"AR B1 VARLEN {tag} sp=1 vs sp={world}: max|d|={diff:.3e} relerr={relerr:.3e} "
                  f"{'PASS' if relerr < 3e-2 else 'FAIL'}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
