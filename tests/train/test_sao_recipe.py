from __future__ import annotations

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from unirl.train.backend.base import LrSchedulerConfig
from unirl.train.optim import build_lr_scheduler

ROOT = Path(__file__).resolve().parents[2]


def test_constant_with_warmup_reaches_base_lr_and_stays_there() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=5.0)
    scheduler = build_lr_scheduler(
        LrSchedulerConfig(type="constant_with_warmup", warmup_steps=2, total_steps=20),
        optimizer=optimizer,
    )
    assert scheduler is not None
    assert scheduler.get_last_lr() == pytest.approx([0.0])

    observed = []
    for _ in range(5):
        optimizer.step()
        scheduler.step()
        observed.append(scheduler.get_last_lr()[0])
    assert observed == pytest.approx([2.5, 5.0, 5.0, 5.0, 5.0])


@pytest.mark.parametrize(
    ("name", "random_value_head", "model_name", "batch_size"),
    [
        ("deep_research_sao_4b_smoke.yaml", True, "Qwen3DecoderLayer", 4),
        ("deep_research_sao_30b_math.yaml", False, "Qwen3MoeDecoderLayer", 128),
    ],
)
def test_sao_recipes_resolve_and_preserve_paper_contract(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    random_value_head: bool,
    model_name: str,
    batch_size: int,
) -> None:
    monkeypatch.setenv("DATA_PATH", "/tmp/tir.jsonl")
    monkeypatch.setenv("QWEN3_INSTRUCT_PATH", "/tmp/actor")
    monkeypatch.setenv("VALUE_MODEL_PATH", "/tmp/critic")
    cfg = OmegaConf.load(ROOT / "examples" / "deep_research" / name)
    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(resolved, dict)

    assert resolved["batch_size"] == batch_size
    assert resolved["sampling"]["samples_per_prompt"] == 1
    assert resolved["rollout"]["config"]["episode_sampling"]["samples_per_prompt"] == 1
    assert resolved["actor"]["algorithm"]["eps_low"] == 0.3
    assert resolved["actor"]["algorithm"]["eps_high"] == 5.0
    assert resolved["actor"]["backend"]["block_class_names"] == [model_name]
    assert resolved["critic"]["backend"]["block_class_names"] == [model_name]
    critic_config = resolved["critic"]["bundle"]["config"]
    assert critic_config["allow_random_value_init"] is random_value_head
    assert critic_config["pretrained_value_ckpt_path"] == "/tmp/critic"
    assert resolved["critic"]["backend"]["optimizer_cfg"]["learning_rate"] == 5.0e-6
    assert resolved["stack"]["critic_updates_per_actor"] == 2
    assert resolved["stack"]["gae_alpha"] == 1.5
    assert resolved["stack"]["critic_lambda"] == 1.0
    if not random_value_head:
        assert resolved["actor"]["backend"]["fsdp_cfg"]["checkpoint_format"] == "dcp"
        assert resolved["critic"]["backend"]["fsdp_cfg"]["checkpoint_format"] == "dcp"
