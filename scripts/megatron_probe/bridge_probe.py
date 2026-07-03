"""mbridge TE-free probe (Phase 0): can mbridge build + convert Qwen3 with LOCAL spec
(no TransformerEngine) on cu130 + mcore 0.13.1? If yes → the bridge works on our pod
and no cu12 image is needed. Validates forward vs HF + export round-trip.

  cd /root/unirl && source .venv-sglang/bin/activate
  MODEL=/root/unirl/models/local/Qwen3-0.6B python scripts/megatron_probe/bridge_probe.py
"""
import os
import torch

def log(*a): print(*a, flush=True)

MODEL = os.environ.get("MODEL", "/root/unirl/models/local/Qwen3-0.6B")

# --- monkeypatch: force the local (TE-free) layer spec ---
import mbridge.core.llm_bridge as _llm
_orig_spec = _llm.get_gpt_decoder_block_spec
def _force_local(config, use_transformer_engine=True, **kw):
    return _orig_spec(config, use_transformer_engine=False, **kw)
_llm.get_gpt_decoder_block_spec = _force_local
log("[patch] forced use_transformer_engine=False (local spec)")

# --- dist + mpu ---
os.environ.setdefault("MASTER_ADDR", "127.0.0.1"); os.environ.setdefault("MASTER_PORT", "29517")
os.environ.setdefault("RANK", "0"); os.environ.setdefault("WORLD_SIZE", "1"); os.environ.setdefault("LOCAL_RANK", "0")
torch.cuda.set_device(0)
import torch.distributed as dist
if not dist.is_initialized(): dist.init_process_group("nccl")
from megatron.core import parallel_state as mpu
if not mpu.model_parallel_is_initialized(): mpu.initialize_model_parallel(1, 1)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
model_parallel_cuda_manual_seed(1234)
log("[stage0] dist + mpu ok")

from mbridge import AutoBridge
bridge = AutoBridge.from_pretrained(MODEL)
log("[stage1] bridge =", type(bridge).__name__)

# build + load
try:
    model = bridge.get_model(weight_path=MODEL)
    log("[stage2] get_model(weight_path=MODEL) ok")
except Exception as e:
    log("[stage2] get_model(weight_path=) failed:", repr(e)[:180])
    model = bridge.get_model()
    ms = model if isinstance(model, list) else [model]
    bridge.load_weights(ms, MODEL)
    log("[stage2b] get_model() + load_weights ok")

ms = model if isinstance(model, list) else [model]
m0 = ms[0]
try: m0 = m0.cuda()
except Exception: pass
try: m0 = m0.bfloat16()
except Exception: pass
m0.eval()
log("[stage2] model params:", sum(1 for _ in m0.named_parameters()))

# forward vs HF oracle
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL)
ids = tok("The capital of France is", return_tensors="pt").input_ids.cuda()
S = ids.shape[1]
with torch.no_grad():
    hf = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda().eval()
    hf_logits = hf(ids).logits
    pos = torch.arange(S, device=ids.device).unsqueeze(0)
    amask = torch.triu(torch.ones(1, 1, S, S, device=ids.device, dtype=torch.bool), diagonal=1)
    out = m0(input_ids=ids, position_ids=pos, attention_mask=amask)
    mc = out.transpose(0, 1) if out.shape[0] == S else out
agree = (hf_logits[0].argmax(-1) == mc[0, :, : hf_logits.shape[-1]].argmax(-1)).float().mean().item()
log(f"[stage3] mbridge-vs-HF argmax agreement = {agree:.3f} (want ~1.0)")

# export round-trip
try:
    exported = {name: t for name, t in bridge.export_weights(ms)}
    log(f"[stage4] export_weights -> {len(exported)} tensors; sample names: {list(exported)[:3]}")
except Exception as e:
    log("[stage4] export_weights failed:", repr(e)[:180])

log("[done] mbridge TE-FREE WORKS" if agree > 0.9 else "[done] mbridge forward MISMATCH")
