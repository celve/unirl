"""GPU golden-diff: AR replay with STORED vs DERIVED prompt conditions (LIN-446 C+I).

Proves on the REAL Qwen3 model that deriving the prompt condition from an input Part
(``Qwen3ARConditions.from_input_segment``) instead of storing it at rollout yields
identical ``padding_replay`` logprobs — i.e. conditions can be derived, not stored,
with zero behavioral change to training replay.

Run on a GPU pod inside tmux:
    QWEN3_PATH=/mnt/gz/models/Qwen3-4B-Base /root/unirl/.venv/bin/python scripts/golden_diff_ar_conditions.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from unirl.models.qwen3.bundle import Qwen3Bundle  # noqa: E402
from unirl.models.qwen3.conditions import Qwen3ARConditions  # noqa: E402
from unirl.models.qwen3.pipeline import Qwen3Pipeline  # noqa: E402
from unirl.types.conditions import TextTokenCondition  # noqa: E402
from unirl.types.primitives import Texts  # noqa: E402
from unirl.types.rollout_resp import RolloutTrack  # noqa: E402
from unirl.types.segments.text import TextSegment  # noqa: E402

PATH = os.environ.get("QWEN3_PATH", "/mnt/gz/models/Qwen3-4B-Base")


def main():
    dev = torch.device("cuda")
    print(f"[load] tokenizer+model from {PATH} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        PATH, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(dev).eval()
    bundle = Qwen3Bundle(transformer=model, tokenizer=tok, dtype=torch.bfloat16, device=dev, pretrained_path=PATH)
    pipe = Qwen3Pipeline.from_bundle(bundle)
    print("[load] done", flush=True)

    texts = Texts(texts=["What is 2+2? Answer:", "Name a primary color. Answer:"])
    conds = pipe.chat_template.embed(texts)  # STORED Qwen3ARConditions (chat-templated)
    ids, mask = conds.prompt.input_ids, conds.prompt.attention_mask
    width = int(ids.shape[1])
    print(f"[stored] prompt input_ids shape={tuple(ids.shape)} real_lens={mask.sum(-1).tolist()}", flush=True)

    # Fabricated short response segment (varlen). replay computes logp from the model.
    resp = [torch.tensor([100, 200, 300], dtype=torch.long), torch.tensor([400, 500], dtype=torch.long)]
    segment = TextSegment.pack(tokens=resp)

    # Input Part holds per-sample REAL prompt tokens; derive the condition from it.
    rows = [ids[i][mask[i].bool()].cpu() for i in range(ids.shape[0])]
    inp = RolloutTrack(
        sample_ids=[f"p{i}" for i in range(len(rows))], stage="input", segment=TextSegment.pack(tokens=rows)
    )
    derived = Qwen3ARConditions.from_input_segment(inp.segment)
    d_ids, d_mask = derived.prompt.input_ids, derived.prompt.attention_mask
    if int(d_ids.shape[1]) < width:  # pad to the stored width for a like-for-like shape
        pad = width - int(d_ids.shape[1])
        d_ids = torch.nn.functional.pad(d_ids, (0, pad), value=0)
        d_mask = torch.nn.functional.pad(d_mask, (0, pad), value=0)
    derived = Qwen3ARConditions(prompt=TextTokenCondition(input_ids=d_ids, attention_mask=d_mask))
    print(f"[derived] prompt input_ids shape={tuple(d_ids.shape)} real_lens={d_mask.sum(-1).tolist()}", flush=True)

    with torch.no_grad():
        logp_old = pipe.ar.padding_replay(conds, segment=segment)
        logp_new = pipe.ar.padding_replay(derived, segment=segment)

    maxdiff = (logp_old.float() - logp_new.float()).abs().max().item()
    equal = bool(torch.equal(logp_old, logp_new))
    close = bool(torch.allclose(logp_old, logp_new, atol=1e-4, rtol=1e-4))
    print(f"[logp] old={logp_old.float().tolist()}", flush=True)
    print(f"[logp] new={logp_new.float().tolist()}", flush=True)
    print(f"[result] max_abs_diff={maxdiff:.3e} torch.equal={equal} allclose={close}", flush=True)
    print("GOLDEN_DIFF_PASS" if close else "GOLDEN_DIFF_FAIL", flush=True)


if __name__ == "__main__":
    main()
