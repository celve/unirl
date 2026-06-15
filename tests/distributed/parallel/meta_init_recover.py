"""Meta-init non-persistent buffer recover (RoPE inv_freq) — the VeOmni AR DRPO fix.

Guards the root-cause fix for VeOmni AR DRPO not learning: ``meta_init_transformer``
builds the model on meta, the backend's ``to_empty`` clobbers the **non-persistent**
RoPE ``inv_freq`` / ``original_inv_freq`` buffers (absent from the checkpoint, never
restored by the weight load), and the train model then runs with garbage rotary
frequencies -> garbage replay log-probs -> the DRPO rollout/replay ratio collapses
(~0.05) and nothing learns. ``capture_init_state`` (carried on the bundle) +
``restore_init_state`` (replayed after the weight load) recover the real init values.

CPU-only, no network (uses a tiny Qwen3 config):

  python tests/distributed/parallel/meta_init_recover.py
"""
import torch
from accelerate import init_empty_weights
from transformers import AutoModelForCausalLM, Qwen3Config

from unirl.models.types.meta_init import capture_init_state, restore_init_state

TARGET = "model.rotary_emb.inv_freq"


def main() -> None:
    cfg = Qwen3Config(
        vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=128,
    )
    # Reference: an eager build computes inv_freq correctly in __init__.
    ref = dict(AutoModelForCausalLM.from_config(cfg).named_buffers())[TARGET].clone()

    # Meta build (parameters on meta, buffers real on CPU) — the bundle's path.
    with init_empty_weights(include_buffers=False):
        model = AutoModelForCausalLM.from_config(cfg)
    captured = capture_init_state(model)
    assert TARGET in captured["buffers"], f"{TARGET} not captured: {list(captured['buffers'])}"
    assert torch.allclose(captured["buffers"][TARGET].float(), ref.float(), atol=1e-4), "capture wrong value"

    # The backend's to_empty clobbers non-persistent buffers (the bug).
    model.to_empty(device="cpu")
    garbage = dict(model.named_buffers())[TARGET]
    assert not torch.allclose(garbage.float(), ref.float(), atol=1e-3), "to_empty did not clobber — test moot"

    # The fix: recover from the carried capture.
    n = restore_init_state(model, captured)
    got = dict(model.named_buffers())[TARGET]
    ok = torch.allclose(got.float(), ref.float(), atol=1e-4)
    print(f"captured={len(captured['buffers'])}buf+{len(captured['attrs'])}attr recovered={n} matches_ref={ok}")
    assert n >= 1 and ok, "meta-init recover FAILED"

    # restore_init_state(None) is a safe no-op (eager bundles).
    assert restore_init_state(model, None) == 0
    print("PASS: meta-init RoPE inv_freq recover")


if __name__ == "__main__":
    main()
