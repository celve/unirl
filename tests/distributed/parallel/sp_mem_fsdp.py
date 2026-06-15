"""SP-vs-FSDP peak-memory bench: same per-DP-group work, FSDP(sp=1) vs FSDP+SP(sp=2).

Real Qwen3-4B config (hidden 2560, 36 layers, 32q/8kv, head_dim 128), FSDP2 over
the folded dp_shard_sp mesh (params sharded across all ranks in BOTH cases — the
only difference is SP shards the SEQUENCE/activations across sp ranks). Measures
torch.cuda.max_memory_allocated for a forward+backward.

  SP=1 torchrun --nproc_per_node=2 --master_port=2961X sp_mem_fsdp.py   # FSDP baseline
  SP=2 torchrun --nproc_per_node=2 --master_port=2961Y sp_mem_fsdp.py   # FSDP + Ulysses
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
from torch.distributed.fsdp import fully_shard

SP = int(os.environ.get("SP", "1"))
L = int(os.environ.get("L", "8192"))
LAYERS = int(os.environ.get("LAYERS", "36"))
AC = int(os.environ.get("AC", "0"))


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
    init_parallel_state(dp_size=world // SP, ulysses_size=SP, dp_mode="fsdp2", device_type="cuda")
    ps = get_parallel_state()

    from transformers import AutoModelForCausalLM, Qwen3Config
    cfg = Qwen3Config(vocab_size=151936, hidden_size=2560, intermediate_size=9728,
                      num_hidden_layers=LAYERS, num_attention_heads=32, num_key_value_heads=8,
                      head_dim=128, max_position_embeddings=131072)
    torch.manual_seed(0)
    m = AutoModelForCausalLM.from_config(cfg).to(dev).to(torch.bfloat16)
    set_attn(m)
    if AC:
        m.gradient_checkpointing_enable()
    mesh = ps.fsdp_mesh
    for layer in m.model.layers:
        fully_shard(layer, mesh=mesh)
    # NB: shard only the decoder layers (where params+activations dominate), not
    # the root — root-sharding makes the embedding weight a DTensor while
    # input_ids stays plain -> aten.embedding mixed Tensor/DTensor error. The
    # unsharded embed is a small constant equal in sp=1 and sp=2, so the SP
    # activation-memory comparison is unaffected.
    if SP > 1:
        apply_sequence_parallelism(m, SP)
    m.train()

    B = 1
    ids = torch.randint(0, 1000, (B, L), device=dev)
    pos = torch.arange(L, device=dev).unsqueeze(0)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    ok, peak = True, 0.0
    try:
        out = m.model(input_ids=ids, position_ids=pos, use_cache=False, return_dict=True).last_hidden_state
        out.float().pow(2).mean().backward()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 1e9
    except torch.cuda.OutOfMemoryError:
        ok = False
    if rank == 0:
        print(f"MEMBENCH sp={SP} AC={AC} world={world} L={L} layers={LAYERS} "
              f"per_rank_L={L // SP} peak={'OOM' if not ok else f'{peak:.2f}GB'}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
