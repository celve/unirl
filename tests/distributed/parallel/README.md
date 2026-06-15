# Ulysses sequence-parallelism (SP) parity tests

Multi-GPU torchrun scripts validating the VeOmni-backend Ulysses SP
(`unirl/train/backend/veomni/sp/`). Each compares **sp=1** (reference, SP a
no-op) against **sp=N** (sequence sliced across N ranks) — they must match to
fp tolerance, since the all-to-all is mathematically exact and the folded
FSDP+SP mesh needs no manual `sp_size` gradient compensation.

Run the reference (`--nproc_per_node=1`) first, then the SP run (`=2`):

```bash
cd /root/unirl && source .venv/bin/activate
torchrun --nproc_per_node=1 --master_port=29570 tests/distributed/parallel/<test>.py   # ref
torchrun --nproc_per_node=2 --master_port=29571 tests/distributed/parallel/<test>.py   # compare
```

| Script | Validates | Result |
|--------|-----------|--------|
| `sp_parity.py` | VeOmni Ulysses primitives (slice/gather + all-to-all), fwd + grad (manual SUM/MEAN combine) | fwd exact; MEAN-grad relerr ~2e-7 |
| `sp_fsdp.py` | **real FSDP2** over the folded `dp_shard_sp` mesh, grad parity (no sp_size comp) | relerr ~2e-7 |
| `sp_ar_parity.py` | AR (Qwen3) registered SP attn + decoder slice/gather, logprob parity | exact (0.0) |
| `sp_ar_varlen.py` | AR left-padded varlen logprob parity (B=2) | exact (0.0) |
| `sp_ar_b1_varlen.py` | AR B=1 (train micro geometry) left-padded varlen — pad-strip path | relerr ~9e-4 |
| `sp_qwenimage_parity.py` | diffusion qwen-image (dispatch-patch), fwd | relerr ~2e-7 |
| `sp_sd3_parity.py` | diffusion SD3 (processor-injection), fwd | relerr ~2e-7 |
| `sp_wan_parity.py` | diffusion Wan (dispatch + image-self/text-cross attn, 5D, Wan RoPE), fwd | relerr ~2e-7 |
| `sp_flux2_parity.py` | diffusion flux2 (dispatch + dual→single blocks + text-strip; model-level slice/gather), fwd | relerr ~2e-7 |
| `sp_diffusion_backward.py` | diffusion qwen-image **backward** grad parity (MEAN-combine) | worst relerr ~6e-7 |

Notes:
- `sp=1` is the no-op path in every script (the reference), so these also guard
  the "SP disabled" regression.
- v1 uses sequence lengths divisible by sp_size (no padding); full attention
  can't tolerate unmasked padding. AR handles left-padding (HF flash unpadding).
- Requires the `[veomni]` extra + flash-attn in the env; needs >= N GPUs.

## Memory benchmarks (USP advantage vs FSDP baseline)

`torch.cuda.max_memory_allocated` for a forward+backward, FSDP2-sharded blocks
(params sharded identically in both cases — the only difference is SP sharding
the sequence/activations across sp ranks). Per-rank peak, sp=1 (FSDP baseline)
vs sp=2 (FSDP + Ulysses). The saving is activation memory, so it grows with
sequence length.

`sd3_mem_fsdp.py` — SD3.5-medium transformer (measured, H20). world=4 so FSDP
shards params ÷4 in EVERY column; only the Ulysses degree (sequence sharding)
changes, isolating the activation saving. Reduction = sp=1 / sp:

| latent (joint seq) | sp=1 (FSDP) | sp=2     | sp=4     | sp=4 reduction |
|--------------------|-------------|----------|----------|----------------|
| 128 (4352)         | 12.24 GB    | 8.82 GB  | 7.14 GB  | 1.71×          |
| 192 (9472)         | 20.45 GB    | 12.97 GB | 9.22 GB  | 2.22×          |
| 256 (16640)        | 31.51 GB    | 18.58 GB | 12.13 GB | 2.60×          |

Both the per-sp reduction and the sp=4-vs-sp=2 gain grow with sequence length
(activation memory ∝ seq/sp; fixed costs — sharded params, embeds, pos_embed —
don't shard with the sequence, so returns are sub-linear but widen with length).

`sp_mem_fsdp.py` — Qwen3-4B AR (needs the cu130 `.venv-sglang` env: the
HF flash-attn kernel; de-risk on real Qwen3 fwd+bwd at L=16384 measured
14.45 GB (sp=1) vs 8.19 GB (sp=2) = 1.76×).
