import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

from omegaconf import DictConfig

from diffusionrl.distributed.group.device_pool import DevicePool

if TYPE_CHECKING:
    from diffusionrl.train.stack import TrainStepResult

logger = logging.getLogger(__name__)


class BaseTrainer:
    """Owns a DevicePool. Subclasses use ``placement(self.pool, ...)`` to
    instantiate their ``Remote`` roles inside ``__init__`` / ``setup``.

    Also owns the optional (rank-0/driver) Weights & Biases logger shared by
    every v2 trainer. Subclasses call :meth:`_init_wandb` once at the top of
    ``train``, :meth:`_log_rollout` after each ``train_step`` (a no-op when
    reporting is off), and :meth:`_finish_wandb` in a ``finally``.
    """

    def __init__(
        self,
        *,
        num_devices: int,
        devices_per_node: int = 8,
        workers_per_device: int = 2,
        logging_cfg: Optional[DictConfig] = None,
    ) -> None:
        self.num_devices = num_devices
        self.pool = DevicePool(
            num_devices=num_devices,
            devices_per_node=devices_per_node,
            workers_per_device=workers_per_device,
        )
        self.pool.setup()

        # Optional wandb logging (driver/rank-0). Stays ``None`` until
        # _init_wandb runs, and remains ``None`` when the ``logging`` block is
        # absent or ``report_to_wandb`` is off — every _log_rollout then no-ops.
        self.logging_cfg = logging_cfg
        self.wandb_logger = None
        self._optimizer_step = 0

    # ---- wandb logging (shared by all v2 trainers) -------------------------

    def _init_wandb(self, *, num_rollouts: int, extra: Optional[Dict[str, Any]] = None) -> None:
        """Open the (rank-0/driver) wandb run from the optional ``logging`` block.

        No-op when the block is absent or ``report_to_wandb`` is off — every
        subsequent :meth:`_log_rollout` then short-circuits. The whole ``train``
        loop runs on the driver, so ``rank=0``.
        """
        cfg = self.logging_cfg
        if cfg is None:
            return
        if not bool(cfg.get("report_to_wandb", False)) or not cfg.get("project_name"):
            return

        from diffusionrl.utils.wandb_logger import init_logger

        raw_tags = cfg.get("tags")
        tags = [str(t) for t in raw_tags] if raw_tags else None
        sampling_params = getattr(self, "sampling_params", None)
        run_config: Dict[str, Any] = {
            "num_devices": self.num_devices,
            "batch_size": getattr(self, "batch_size", None),
            "num_rollouts": num_rollouts,
            "samples_per_prompt": getattr(sampling_params, "samples_per_prompt", None),
        }
        if extra:
            run_config.update(extra)

        project = cfg.get("project_name")
        self.wandb_logger = init_logger(
            project=str(project),
            run_name=cfg.get("run_name"),
            config=run_config,
            rank=0,
            tags=tags,
            entity=cfg.get("entity") or None,
        )
        if self.wandb_logger.initialized:
            logger.info("WandB initialized: project=%s run=%s", project, cfg.get("run_name"))

    def _log_rollout(
        self,
        rollout_id: int,
        results: Union["TrainStepResult", Dict[str, "TrainStepResult"]],
        resp: Any,
        *,
        step_time_s: Optional[float] = None,
    ) -> None:
        """Log one rollout's metrics to wandb. No-op when reporting is off.

        ``rollout/*`` carries reward/advantage distribution stats (single-track
        keys unprefixed; multi-track auto-prefixed by track name). ``train/*``
        carries optimizer scalars + algorithm metrics — per-track namespaced
        (``<track>/<key>``) when ``results`` is a ``{track: TrainStepResult}``
        dict, as for the PE trainer. ``perf/rollout_time_s`` is the optional
        wall-clock for the step.
        """
        wb = self.wandb_logger
        if wb is None or not wb.initialized:
            return

        from diffusionrl.utils.wandb_logger import aggregate_stage_results
        from diffusionrl.utils.wandb_metrics import compute_rollout_resp_metrics

        step = rollout_id + 1
        wb.log_rollout(step, compute_rollout_resp_metrics(resp=resp))

        if isinstance(results, dict):
            train_metrics: Dict[str, Any] = {
                f"{name}/{key}": value
                for name, result in results.items()
                for key, value in aggregate_stage_results([result]).items()
            }
            has_backward = any(bool(r.has_backward) for r in results.values())
        else:
            train_metrics = dict(aggregate_stage_results([results]))
            has_backward = bool(results.has_backward)

        if has_backward:
            self._optimizer_step += 1
            wb.log_step(self._optimizer_step, train_metrics)

        if step_time_s is not None:
            wb.log_perf(step, {"rollout_time_s": float(step_time_s)})

    def _finish_wandb(self) -> None:
        """Close the wandb run if one is open."""
        if self.wandb_logger is not None:
            self.wandb_logger.finish()
