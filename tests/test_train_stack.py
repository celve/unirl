"""Tests for ``diffusionrl.training.stack.TrainStack``.

Covers the cfg-backed read path after Step 2: ``TrainStack`` no longer
takes ``max_grad_norm`` as a dataclass field or ``mini_batch_size`` /
``micro_batch_size`` as per-call kwargs — they come out of the cfg node
stored on ``self.cfg``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from omegaconf import OmegaConf

from diffusionrl.training.stack import TrainStack


def _make_cfg(*, mini: int = 2, micro: int = 1, clip: float = 2.5) -> OmegaConf:
    return OmegaConf.create(
        {
            "training": {
                "plan": {"local_mini_batch_size": mini, "micro_batch_size": micro},
                "execution": {"max_grad_norm": clip},
            }
        }
    )


def _make_batch(batch_size: int = 2):
    batch = MagicMock(name="TrainingBatch")
    batch.batch_size = batch_size

    # slice(start, end) returns a smaller mock that behaves the same way; the
    # child batch_size is the slice width so nested micro slicing terminates.
    def _slice(start: int, end: int):
        child = MagicMock(name=f"BatchSlice[{start}:{end}]")
        child.batch_size = end - start
        child.slice = _slice
        return child

    batch.slice = _slice
    return batch


def _make_stack(cfg, *, clipped_value: float = 0.5) -> TrainStack:
    backend = MagicMock(name="backend")
    backend.clip_grad_norm = MagicMock(return_value=clipped_value)
    backend.model = MagicMock(name="model")
    optimizer = MagicMock(name="optimizer")
    optimizer.param_groups = [{"lr": 0.001}]
    scheduler = MagicMock(name="scheduler")
    scheduler.get_last_lr = MagicMock(return_value=[0.001])
    algorithm = MagicMock(name="algorithm")
    algorithm.compute_loss_and_backward = MagicMock(return_value=(1.0, {}, 1, True))
    algorithm.resolve_training_timesteps = MagicMock(return_value=[0])
    ema_manager = MagicMock(name="ema")
    return TrainStack(
        backend=backend,
        optimizer=optimizer,
        scheduler=scheduler,
        algorithm=algorithm,
        ema_manager=ema_manager,
        cfg=cfg,
    )


def test_train_batch_reads_sizing_from_cfg():
    """train_batch / train_minibatch slice widths come from cfg.training.plan."""
    cfg = _make_cfg(mini=2, micro=1)
    stack = _make_stack(cfg)
    batch = _make_batch(batch_size=4)

    stack.train_batch(batch=batch, rollout_step=0)

    # mini slicing: batch_size=4, mini=2 → 2 mini-batches, each slicing (0,2)
    # and (2,4). Then each mini recurses with micro=1, slicing (0,1) and (1,2).
    # compute_loss_and_backward fires once per micro slice.
    assert stack.algorithm.compute_loss_and_backward.call_count == 4


def test_train_minibatch_reads_max_grad_norm_from_cfg():
    """clip_grad_norm is called with cfg.training.execution.max_grad_norm."""
    cfg = _make_cfg(clip=3.14)
    stack = _make_stack(cfg)
    batch = _make_batch(batch_size=1)

    stack.train_batch(batch=batch, rollout_step=0)

    stack.backend.clip_grad_norm.assert_called_with(3.14)
