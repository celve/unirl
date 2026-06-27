"""AgentLoop — a synchronous, config-driven multi-turn rollout driver (LIN-492).

See ``docs/agent-loop-design.md``. One generic loop runs N turns over the existing
synchronous rollout engine, threading state through the ``Sample``/``Part`` model: each turn
forks a generation shell, the engine fills it, and (optionally) an environment returns an
observation that re-enters the chain as a mask-0 input Part. Behaviour is fully determined by
``(plan, environment, max_turns)`` — diffusion / AR / unified (Composed) / agentic loops are
all *configurations* of this one class, not subclasses.

Prototype scope: synchronous only (the engine's ``generate`` is a blocking call).
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Optional, Tuple, Union

from unirl.rollout.loop.engine_port import RolloutEnginePort
from unirl.rollout.loop.environment import Environment
from unirl.types.sample import Sample
from unirl.types.sampling import BaseSamplingParams

# One turn of the generation plan: how many samples to fork, and with which params. The
# params TYPE selects the modality/engine (ARSamplingParams vs DiffusionSamplingParams),
# matching the existing ``Sample.gen_part(params_type)`` convention.
Turn = Tuple[int, BaseSamplingParams]


class AgentLoop:
    """One generic, SYNCHRONOUS multi-turn rollout loop.

    Owns control flow and trajectory assembly only — not generation (engine), world dynamics
    (environment), scoring, or training. Behaviour = ``(plan, environment, max_turns)``:

    - ``plan`` — a fixed ``list`` of :data:`Turn` (run exactly those turns; e.g. the Composed
      AR→diffusion flow), or a single :data:`Turn` repeated until the environment signals
      ``done`` or ``max_turns`` is reached (the agentic case).
    - ``environment`` — optional. ``None`` => pure generation (single-turn or fixed plan).
    - ``max_turns`` — hard upper bound on turns (a safety bound for repeated plans).
    """

    def __init__(
        self,
        plan: Union[List[Turn], Turn],
        environment: Optional[Environment] = None,
        max_turns: int = 8,
    ) -> None:
        self.plan = plan
        self.environment = environment
        self.max_turns = max_turns

    def run(self, engine: RolloutEnginePort, request: Sample) -> Sample:
        """Drive the episode: ``fork → generate → (env.step → observe)`` per turn."""
        sample = self.environment.reset(request) if self.environment is not None else request
        for turn, (branch, params) in enumerate(self._turns()):
            if turn >= self.max_turns:
                break
            sample = engine.generate(sample.fork(branch, sampling_params=params))  # the agent acts
            if self.environment is None:
                continue
            observation, done, _info = self.environment.step(sample)  # the world responds
            if done:
                break
            if observation is not None:
                sample = sample.observe(observation)  # mask-0 input Part
        return sample

    def _turns(self) -> Iterable[Turn]:
        # fixed plan -> the list; agentic plan -> repeat one turn (bounded by env.done / max_turns)
        return self.plan if isinstance(self.plan, list) else itertools.repeat(self.plan)


__all__ = ["AgentLoop", "Turn"]
