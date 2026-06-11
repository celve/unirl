#!/usr/bin/env python3
"""Bagel AR stage CPU checks — no GPU, no checkpoint, no flash_attn.

Drives ``BagelARStage`` / ``rl_ops`` against a FAKE navit model that honors the
vendored ``forward_inference`` kwargs contract with exactly-representable fp32
arithmetic (weights are multiples of 1/8, causal state = cumulative-mean mixing),
so the per-token rollout and the one-shot teacher-forced replay are numerically
identical by construction — any drift is a bookkeeping bug, not kernel noise.

Checks:
  1. flash-free import       — ``import unirl.models.bagel`` with flash_attn poisoned
  2. _pack_text_ids parity   — byte-equivalence with ``prepare_prompts`` bookkeeping (bs=1)
  3. BagelARConditions       — for_sample / from_dict / to_dict round trip + errors
  4. rollout == replay       — sampled (T=0.7) and greedy (T=0) per-token logps match
  5. replay grads            — backward through replay reaches embed/mix/lm_head
  6. eval-mode guard         — .train() raises (navit forward dispatch contract)
  7. task routing            — BagelPipeline._resolve_task inference table

Run:  python3 scripts/bagel_ar_cpu_check.py
"""
from __future__ import annotations

import sys
from types import SimpleNamespace

# Poison BEFORE importing unirl: the package import must never touch flash_attn.
sys.modules.setdefault("flash_attn", None)

import torch
from torch import nn

from unirl.models.bagel import BagelARConditions, BagelARStage
from unirl.models.bagel import rl_ops
from unirl.models.bagel.pipeline import BagelPipeline
from unirl.types.primitives import Image, Images, Texts
from unirl.types.rollout_req import RolloutReq
from unirl.types.sampling import ARSamplingParams, DiffusionSamplingParams

HIDDEN, VOCAB = 32, 64
BOS, EOS = 1, 2


def log(msg: str) -> None:
    print(f"[bagel-ar-cpu] {msg}", flush=True)


def _q(t: torch.Tensor) -> torch.Tensor:
    """Quantize to multiples of 1/8 so fp32 sums are exact (order-independent)."""
    return torch.round(t * 8.0) / 8.0


class NaiveCache:  # resolved by rl_ops.init_und_context via sys.modules lookup
    def __init__(self, num_layers: int) -> None:
        self.embs: torch.Tensor | None = None  # [kv, H] accumulated query embeddings


class _Out:
    def __init__(self, h: torch.Tensor, past: NaiveCache) -> None:
        self.packed_query_sequence = h
        self.past_key_values = past


class FakeInner(nn.Module):  # plays Qwen2Model
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB, HIDDEN)


class FakeLM(nn.Module):  # plays Qwen2ForCausalLM
    def __init__(self) -> None:
        super().__init__()
        self.model = FakeInner()
        self.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)
        self.mix = nn.Linear(HIDDEN, HIDDEN, bias=False)
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(_q(p))

    def forward_inference(
        self,
        *,
        packed_query_sequence,
        query_lens,
        packed_query_position_ids,
        packed_query_indexes,
        past_key_values,
        key_values_lens,
        packed_key_value_indexes,
        update_past_key_values=True,
        is_causal=True,
        mode="und",
    ) -> _Out:
        q = packed_query_sequence  # [n, H]
        stored = past_key_values.embs
        kv = 0 if stored is None else int(stored.shape[0])
        assert int(key_values_lens[0]) == kv, f"kv_len mismatch: ctx says {int(key_values_lens[0])}, cache has {kv}"
        assert int(packed_query_indexes[0]) == kv, "query index must start at kv_len"
        all_emb = q if stored is None else torch.cat([stored, q], dim=0)  # [kv+n, H]
        # Causal cumulative mean: row j sees rows 0..j — identical whether fed
        # one token at a time (rollout) or in one pass (replay): exact sums.
        csum = torch.cumsum(all_emb, dim=0)
        counts = torch.arange(1, all_emb.shape[0] + 1, dtype=torch.float32).unsqueeze(1)
        h_full = all_emb + self.mix(csum / counts)
        if update_past_key_values:
            past_key_values.embs = all_emb
        return _Out(h_full[kv:], past_key_values)


class FakeBagel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = FakeLM()
        self.config = SimpleNamespace(llm_config=SimpleNamespace(num_hidden_layers=2, freeze_und=False))

    # No @torch.no_grad (rl_ops._raw falls back to the plain function).
    def forward_cache_update_text(
        self,
        past_key_values,
        packed_text_ids,
        packed_text_position_ids,
        text_token_lens,
        packed_text_indexes,
        packed_key_value_indexes,
        key_values_lens,
    ):
        emb = self.language_model.model.embed_tokens(packed_text_ids)
        out = self.language_model.forward_inference(
            packed_query_sequence=emb,
            query_lens=text_token_lens,
            packed_query_position_ids=packed_text_position_ids,
            packed_query_indexes=packed_text_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=True,
            is_causal=True,
            mode="und",
        )
        return out.past_key_values


class FakeBundle:
    def __init__(self, model: FakeBagel) -> None:
        self.model = model
        self.transformer = model.language_model
        self.new_token_ids = {"bos_token_id": BOS, "eos_token_id": EOS}
        self.device = torch.device("cpu")


def make_conditions() -> BagelARConditions:
    s0 = [{"kind": "text", "ids": torch.tensor([BOS, 10, 11, 12, EOS], dtype=torch.long)}]
    s1 = [{"kind": "text", "ids": torch.tensor([BOS, 20, 21, EOS], dtype=torch.long)}]
    return BagelARConditions(prompt_splits=[s0, s1])


def check_pack_text_ids() -> None:
    ids = torch.tensor([BOS, 5, 6, EOS], dtype=torch.long)
    gi = rl_ops._pack_text_ids(ids, kv_len=7, rope_start=3)
    # prepare_prompts bookkeeping at bs=1 (vendor bagel.py:232-264).
    assert torch.equal(gi["text_token_lens"], torch.tensor([4], dtype=torch.int))
    assert torch.equal(gi["packed_text_ids"], ids)
    assert torch.equal(gi["packed_text_position_ids"], torch.arange(3, 7))
    assert torch.equal(gi["packed_text_indexes"], torch.arange(7, 11))
    assert torch.equal(gi["packed_key_value_indexes"], torch.arange(7))
    assert torch.equal(gi["key_values_lens"], torch.tensor([7], dtype=torch.int))
    log("PASS  _pack_text_ids parity")


def check_conditions() -> None:
    c = make_conditions()
    assert c.batch_size == 2
    rt = BagelARConditions.from_dict(c.to_dict())
    assert rt is c
    one = BagelARConditions.for_sample(splits=[{"kind": "text", "ids": torch.tensor([BOS, EOS])}])
    assert one.batch_size == 1
    for bad in ({"kind": "audio"}, "not-a-dict"):
        try:
            BagelARConditions.for_sample(splits=[bad])  # type: ignore[list-item]
            raise AssertionError("for_sample accepted a bad split")
        except ValueError:
            pass
    try:
        BagelARConditions.from_dict({"bagel": object()})
        raise AssertionError("from_dict accepted wrong key")
    except ValueError:
        pass
    log("PASS  BagelARConditions round-trip + errors")


def check_rollout_replay(stage: BagelARStage, temperature: float, replay_temperature: float) -> None:
    conditions = make_conditions()
    sp = ARSamplingParams(temperature=temperature, max_new_tokens=12, top_p=0.9, top_k=8)
    torch.manual_seed(7)
    segment = stage.autoregress(conditions, sampling_params=sp)
    lengths = [int(n) for n in segment.lengths.tolist()]
    assert len(lengths) == 2 and all(n >= 1 for n in lengths), lengths
    with torch.no_grad():
        new_logp = stage.replay(conditions, segment=segment, temperature=replay_temperature)
    old_logp = segment.log_probs
    assert new_logp.shape == old_logp.shape, (new_logp.shape, old_logp.shape)
    delta = (new_logp - old_logp).abs().max().item()
    assert delta < 1e-5, f"rollout vs replay |dlogp| = {delta}"
    log(f"PASS  rollout==replay (T={temperature}, replay_T={replay_temperature}, "
        f"lens={lengths}, max|dlogp|={delta:.2e})")


def check_replay_grads(stage: BagelARStage) -> None:
    conditions = make_conditions()
    sp = ARSamplingParams(temperature=0.7, max_new_tokens=8, top_p=1.0, top_k=0)
    torch.manual_seed(11)
    segment = stage.autoregress(conditions, sampling_params=sp)
    lm = stage.model.transformer
    lm.zero_grad(set_to_none=True)
    with torch.enable_grad():
        new_logp = stage.replay(conditions, segment=segment, temperature=0.7)
        new_logp.sum().backward()
    for name in ("model.embed_tokens.weight", "mix.weight", "lm_head.weight"):
        g = dict(lm.named_parameters())[name].grad
        assert g is not None and g.abs().sum() > 0, f"no grad on {name}"
    log("PASS  replay grads reach embed/mix/lm_head (prompt prefill carries grad)")


def check_eval_guard(stage: BagelARStage) -> None:
    stage.model.transformer.train()
    try:
        stage.autoregress(make_conditions(), sampling_params=ARSamplingParams(max_new_tokens=2))
        raise AssertionError("autoregress ran in train() mode")
    except RuntimeError:
        pass
    finally:
        stage.model.transformer.eval()
    log("PASS  eval-mode guard")


def check_task_routing() -> None:
    img = Images.from_list([Image(pixels=torch.zeros(3, 8, 8))])
    ar, diff = ARSamplingParams(), DiffusionSamplingParams()
    cases = [
        ({"text": Texts(texts=["x"])}, diff, None, "t2i"),
        ({"text": Texts(texts=["x"]), "image": img}, diff, None, "it2i"),
        ({"text": Texts(texts=["x"])}, ar, None, "t2t"),
        ({"text": Texts(texts=["x"]), "image": img}, ar, None, "it2t"),
        ({"image": img}, ar, "i2t", "i2t"),  # pure i2t must be explicit
        ({"text": Texts(texts=["x"])}, diff, "it2t", "it2t"),  # explicit wins
    ]
    for prims, sp, explicit, expect in cases:
        req = RolloutReq(
            sample_ids=["s0"],
            primitives=prims,
            sampling_params=sp,
            stage_config={"task": explicit} if explicit else {},
        )
        got = BagelPipeline._resolve_task(req)
        assert got == expect, f"routing: expected {expect}, got {got}"
    log("PASS  task routing")


def main() -> int:
    import unirl.models.bagel  # noqa: F401  (flash poisoned at the top — import must survive)

    log("PASS  flash-free import")
    check_pack_text_ids()
    check_conditions()

    torch.manual_seed(0)
    bundle = FakeBundle(FakeBagel())
    bundle.model.eval()
    stage = BagelARStage(model=bundle, autocast_precision="fp32", logprob_precision="fp32")

    check_rollout_replay(stage, temperature=0.7, replay_temperature=0.7)
    check_rollout_replay(stage, temperature=0.0, replay_temperature=1.0)
    check_replay_grads(stage)
    check_eval_guard(stage)
    check_task_routing()
    log("ALL CHECKS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
