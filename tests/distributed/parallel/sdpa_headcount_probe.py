"""Is bf16 SDPA head-count-dependent? (HI3 SP divergence root-cause probe.)

Ulysses gives each rank H/sp heads; the all-to-all is exact, so the only way the
SP attention can differ from the full-head attention is if SDPA itself yields a
different bf16 result for the SAME per-head q/k/v depending on how many heads are
batched. Compare the first 4 heads of a 32-head SDPA against a standalone 4-head
SDPA (identical inputs), under each backend, in bf16 and fp32.
"""
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

torch.manual_seed(0)
B, H, L, D, KV = 1, 32, 64, 128, 8  # HI3: 32 Q heads, 8 KV heads, head_dim 128
dev = "cuda"


def probe(dtype):
    q = torch.randn(B, H, L, D, device=dev, dtype=dtype)
    k = torch.randn(B, KV, L, D, device=dev, dtype=dtype)
    v = torch.randn(B, KV, L, D, device=dev, dtype=dtype)
    kr = k.repeat_interleave(H // KV, dim=1)  # GQA expand -> 32
    vr = v.repeat_interleave(H // KV, dim=1)
    mask = torch.zeros(B, 1, L, L, device=dev, dtype=dtype).tril_().bool()
    mask = torch.where(mask, torch.zeros((), device=dev, dtype=dtype), torch.full((), torch.finfo(dtype).min, device=dev, dtype=dtype))

    def run(backend):
        ctx = sdpa_kernel(backend) if backend else torch.autograd.grad_mode.no_grad()
        with torch.no_grad():
            if backend:
                with sdpa_kernel(backend):
                    o32 = F.scaled_dot_product_attention(q, kr, vr, attn_mask=mask)
                    o4 = F.scaled_dot_product_attention(q[:, :4], kr[:, :4], vr[:, :4], attn_mask=mask)
            else:
                o32 = F.scaled_dot_product_attention(q, kr, vr, attn_mask=mask)
                o4 = F.scaled_dot_product_attention(q[:, :4], kr[:, :4], vr[:, :4], attn_mask=mask)
        return (o32[:, :4].float() - o4.float()).abs().max().item()

    print(f"== dtype={dtype} ==")
    print(f"  default(auto-select): head32[:4] vs head4  max|Δ| = {run(None):.3e}")
    for name, b in [("MATH", SDPBackend.MATH), ("EFFICIENT", SDPBackend.EFFICIENT_ATTENTION), ("FLASH", SDPBackend.FLASH_ATTENTION)]:
        try:
            print(f"  {name:10s}: {run([b]):.3e}")
        except Exception as e:
            print(f"  {name:10s}: ERR {str(e)[:70]}")


for dt in (torch.bfloat16, torch.float32):
    probe(dt)
