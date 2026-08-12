"""AReaL-compatible controller for the Tongyi deep-research task."""

from __future__ import annotations

import dataclasses
import logging
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Mapping, Optional, Sequence

from unirl.rollout.env.areal_deep_research import extract_first_answer
from unirl.rollout.harness.protocol import BaseHarnessConfig, HarnessContext, HarnessOutcome
from unirl.types.primitives import Texts
from unirl.types.sample import _part_with_field
from unirl.types.sampling import ARSamplingParams

if TYPE_CHECKING:
    from unirl.rollout.env.protocol import Environment
    from unirl.types.sample import Sample

logger = logging.getLogger(__name__)

_PROTOCOL = "areal_deep_research/v1"
_TOKEN_LIMIT_NUDGE = (
    "You have now reached the maximum context length you can handle. "
    "You should stop making tool calls and, based on all the information above, "
    "think again and provide what you consider the most likely answer in the following format:"
    "<think>your final thinking</think>\n<answer>your answer</answer>"
)
_CALL_LIMIT_NUDGE = (
    "Sorry, the number of llm calls exceeds the limit. You should stop making tool calls and, "
    "based on all the information above, think again and provide what you consider the most likely answer "
    "in the following format:<think>your final thinking</think>\n<answer>your answer</answer>"
)


class _InfrastructureFailure(RuntimeError):
    pass


class _Suspended(RuntimeError):
    pass


@dataclass
class _ControllerState:
    phase: Literal["ordinary", "token_rescue", "call_rescue"] = "ordinary"
    policy_call_count: int = 0
    retry_count: int = 0
    elapsed_seconds: float = 0.0
    prompt_tokens_at_rescue: Optional[int] = None
    rescue_prompt_tokens: Optional[int] = None
    tool_calls: int = 0
    search_calls: int = 0
    visit_calls: int = 0
    controller_error_type: Optional[str] = None
    controller_error_message: Optional[str] = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "_ControllerState":
        fields = {field.name for field in dataclasses.fields(cls)}
        return cls(**{key: value[key] for key in fields if key in value})


@dataclass(frozen=True)
class ARealDeepResearchHarnessConfig(BaseHarnessConfig):
    """Recipe-owned limits for :class:`ARealDeepResearchHarness`."""

    max_policy_calls: int = 100
    max_new_tokens_per_call: int = 8192
    soft_trajectory_tokens: int = 27648
    answer_now_prompt_tokens: int = 22118
    hard_context_tokens: int = 32768
    max_generation_attempts: int = 3
    retry_backoff_seconds: Sequence[float] = (0.5, 1.0)
    wall_timeout_seconds: float = 9000.0

    def __post_init__(self) -> None:
        backoffs = tuple(float(value) for value in self.retry_backoff_seconds)
        object.__setattr__(self, "retry_backoff_seconds", backoffs)
        if self.max_policy_calls < 2:
            raise ValueError("AReaL deep research requires at least two policy calls")
        if self.max_new_tokens_per_call <= 0:
            raise ValueError("max_new_tokens_per_call must be positive")
        if self.soft_trajectory_tokens <= 0 or self.answer_now_prompt_tokens != (4 * self.soft_trajectory_tokens) // 5:
            raise ValueError("answer_now_prompt_tokens must equal floor(0.8 * soft_trajectory_tokens)")
        if not self.answer_now_prompt_tokens < self.soft_trajectory_tokens < self.hard_context_tokens:
            raise ValueError("AReaL deep-research token limits are invalid")
        if self.max_new_tokens_per_call > self.hard_context_tokens:
            raise ValueError("max_new_tokens_per_call cannot exceed hard_context_tokens")
        if self.max_generation_attempts < 1:
            raise ValueError("max_generation_attempts must be positive")
        if len(backoffs) != self.max_generation_attempts - 1 or any(value < 0 for value in backoffs):
            raise ValueError("retry_backoff_seconds must contain one non-negative delay between each attempt")
        if self.wall_timeout_seconds <= 0:
            raise ValueError("wall_timeout_seconds must be positive")

    def make_harness(self, *, env: "Environment", sampling: Any) -> "ARealDeepResearchHarness":
        if not isinstance(sampling, ARSamplingParams):
            raise TypeError("AReaL deep research requires ARSamplingParams")
        if int(sampling.max_new_tokens) != self.max_new_tokens_per_call:
            raise ValueError(
                "AReaL deep-research sampling.max_new_tokens must equal "
                f"max_new_tokens_per_call ({self.max_new_tokens_per_call})"
            )
        return ARealDeepResearchHarness(env=env, sampling=sampling, config=self)


class ARealDeepResearchHarness:
    """Run the corrected 100-call AReaL deep-research state machine."""

    ENGINE = "policy"

    def __init__(
        self,
        *,
        env: "Environment",
        sampling: ARSamplingParams,
        config: ARealDeepResearchHarnessConfig,
    ) -> None:
        self.env = env
        self.sampling = sampling
        self.config = config

    def run(self, request: "Sample", context: HarnessContext) -> HarnessOutcome:
        sample = request
        state = _ControllerState()
        started_at = time.monotonic()
        try:
            state = self._restore_state(request)
            started_at -= state.elapsed_seconds
            if state.policy_call_count == 0 and not sample.gen_parts():
                sample = self.env.reset(request)
            elif state.policy_call_count != len(sample.gen_parts()):
                raise _InfrastructureFailure("invalid_resume_state")

            if state.phase != "ordinary":
                return self._run_rescue(sample, state, context, started_at)

            ordinary_limit = self.config.max_policy_calls - 1
            while state.policy_call_count < ordinary_limit:
                self._check_boundary(context, state, started_at)
                sample = self._generate(sample, self.sampling, state, context, started_at)
                state.policy_call_count += 1

                observation, done, info = self.env.step(sample)
                self._record_tool_call(state, info)
                if observation is not None:
                    sample = sample.observe(observation, role="user")
                if done:
                    return self._outcome(
                        sample,
                        state,
                        started_at,
                        status="completed",
                        termination_reason="answer",
                        prediction=info.get("prediction") or "",
                        answer_tag_found=bool(info.get("answer_tag_found")),
                    )

                next_request = sample.fork(1, sampling_params=self.sampling)
                prompt_tokens = context.count_prompt_tokens(self.ENGINE, next_request)
                if prompt_tokens > self.config.answer_now_prompt_tokens:
                    state.phase = "token_rescue"
                    state.prompt_tokens_at_rescue = prompt_tokens
                    sample = sample.observe(Texts(texts=[_TOKEN_LIMIT_NUDGE]), role="user")
                    return self._run_rescue(sample, state, context, started_at)

            state.phase = "call_rescue"
            sample = sample.observe(Texts(texts=[_CALL_LIMIT_NUDGE]), role="user")
            return self._run_rescue(sample, state, context, started_at)
        except _Suspended:
            return self._outcome(
                sample,
                state,
                started_at,
                status="suspended",
                termination_reason="suspended",
                controller_state=dataclasses.asdict(state),
            )
        except _InfrastructureFailure as exc:
            return self._outcome(
                sample,
                state,
                started_at,
                status="failed",
                termination_reason=str(exc),
            )
        except Exception as exc:  # noqa: BLE001 - isolate controller failure
            logger.warning("AReaL deep-research controller failed (%s)", type(exc).__name__, exc_info=False)
            state.controller_error_type = type(exc).__name__
            state.controller_error_message = _safe_error_message(exc)
            return self._outcome(
                sample,
                state,
                started_at,
                status="failed",
                termination_reason="controller_error",
            )
        finally:
            close = getattr(self.env, "close", None)
            if close is not None:
                try:
                    close(sample)
                except Exception:  # noqa: BLE001 - preserve trajectory
                    logger.warning("AReaL deep-research environment teardown failed", exc_info=False)

    def _run_rescue(
        self,
        sample: "Sample",
        state: _ControllerState,
        context: HarnessContext,
        started_at: float,
    ) -> HarnessOutcome:
        self._check_boundary(context, state, started_at)
        if state.policy_call_count >= self.config.max_policy_calls:
            raise _InfrastructureFailure("policy_call_limit_exhausted")

        provisional = sample.fork(1, sampling_params=self.sampling)
        prompt_tokens = context.count_prompt_tokens(self.ENGINE, provisional)
        state.rescue_prompt_tokens = prompt_tokens
        headroom = self.config.hard_context_tokens - prompt_tokens
        if headroom <= 0:
            raise _InfrastructureFailure("context_exhausted")

        rescue_sampling = dataclasses.replace(
            self.sampling,
            max_new_tokens=min(self.config.max_new_tokens_per_call, headroom),
        )
        sample = self._generate(sample, rescue_sampling, state, context, started_at)
        state.policy_call_count += 1

        completion = self._latest_text(sample)
        tagged = extract_first_answer(completion)
        prediction = tagged if tagged is not None else completion
        prefix = "token_limit_rescue" if state.phase == "token_rescue" else "call_limit_rescue"
        reason = prefix if tagged is not None else f"{prefix}_format_error"
        return self._outcome(
            sample,
            state,
            started_at,
            status="completed",
            termination_reason=reason,
            prediction=prediction,
            answer_tag_found=tagged is not None,
        )

    def _generate(
        self,
        sample: "Sample",
        sampling: ARSamplingParams,
        state: _ControllerState,
        context: HarnessContext,
        started_at: float,
    ) -> "Sample":
        generation_request = self._with_sampling_seed(sample, state).fork(1, sampling_params=sampling)
        for attempt in range(self.config.max_generation_attempts):
            self._check_wall(state, started_at)
            try:
                return context.generate(self.ENGINE, generation_request)
            except Exception as exc:  # noqa: BLE001 - redact retry failures
                state.retry_count += 1
                logger.warning(
                    "AReaL deep-research policy attempt %d/%d failed (%s)",
                    attempt + 1,
                    self.config.max_generation_attempts,
                    type(exc).__name__,
                    exc_info=False,
                )
                if attempt + 1 >= self.config.max_generation_attempts:
                    raise _InfrastructureFailure("generation_retry_exhausted") from None
                delay = self.config.retry_backoff_seconds[attempt]
                if self._elapsed(started_at) + delay >= self.config.wall_timeout_seconds:
                    raise _InfrastructureFailure("wall_timeout") from None
                time.sleep(delay)
        raise _InfrastructureFailure("generation_retry_exhausted")

    @staticmethod
    def _with_sampling_seed(sample: "Sample", state: _ControllerState) -> "Sample":
        root = sample.parts[0]
        control = dict(root.control)
        ar_control = dict(control.get("ar") or {})
        seed_base = ar_control.get("sampling_seed_base")
        if seed_base is None:
            return sample
        ar_control["sampling_seed"] = (int(seed_base) + int(state.policy_call_count)) % ((1 << 63) - 1)
        control["ar"] = ar_control
        seeded_root = _part_with_field(root, "control", control)
        return sample.with_parts([seeded_root, *sample.parts[1:]])

    def _check_boundary(self, context: HarnessContext, state: _ControllerState, started_at: float) -> None:
        self._check_wall(state, started_at)
        if context.suspend_requested():
            raise _Suspended

    def _check_wall(self, state: _ControllerState, started_at: float) -> None:
        state.elapsed_seconds = self._elapsed(started_at)
        if state.elapsed_seconds >= self.config.wall_timeout_seconds:
            raise _InfrastructureFailure("wall_timeout")

    @staticmethod
    def _elapsed(started_at: float) -> float:
        return max(0.0, time.monotonic() - started_at)

    @staticmethod
    def _record_tool_call(state: _ControllerState, info: Mapping[str, Any]) -> None:
        call = info.get("tool_call")
        if not isinstance(call, dict):
            return
        state.tool_calls += 1
        if call.get("name") == "search":
            state.search_calls += 1
        elif call.get("name") == "visit":
            state.visit_calls += 1

    @staticmethod
    def _latest_text(sample: "Sample") -> str:
        frontier = sample.gen_parts()[-1].primitives.get("text")
        if not isinstance(frontier, Texts) or len(frontier.texts) != 1:
            raise _InfrastructureFailure("missing_policy_completion")
        return frontier.texts[0] or ""

    @staticmethod
    def _restore_state(sample: "Sample") -> _ControllerState:
        if not sample.parts:
            return _ControllerState()
        rows = sample.parts[-1].metadata
        harness = (rows[0] or {}).get("harness") if rows else None
        if not isinstance(harness, Mapping) or harness.get("protocol") != _PROTOCOL:
            if sample.gen_parts():
                raise _InfrastructureFailure("missing_resume_state")
            return _ControllerState()
        raw_state = harness.get("controller_state")
        if not isinstance(raw_state, Mapping):
            raise _InfrastructureFailure("missing_resume_state")
        return _ControllerState.from_mapping(raw_state)

    def _outcome(
        self,
        sample: "Sample",
        state: _ControllerState,
        started_at: float,
        *,
        status: Literal["completed", "suspended", "failed"],
        termination_reason: str,
        prediction: Optional[str] = None,
        answer_tag_found: bool = False,
        controller_state: Optional[Mapping[str, Any]] = None,
    ) -> HarnessOutcome:
        state.elapsed_seconds = self._elapsed(started_at)
        metadata = {
            "protocol": _PROTOCOL,
            "termination_reason": termination_reason,
            "policy_call_count": state.policy_call_count,
            "prompt_tokens_at_rescue": state.prompt_tokens_at_rescue,
            "rescue_prompt_tokens": state.rescue_prompt_tokens,
            "answer_tag_found": bool(answer_tag_found),
            "tool_calls": state.tool_calls,
            "search_calls": state.search_calls,
            "visit_calls": state.visit_calls,
            "retry_count": state.retry_count,
            "elapsed_seconds": state.elapsed_seconds,
            "controller_error_type": state.controller_error_type,
            "controller_error_message": state.controller_error_message,
        }
        if prediction is not None:
            metadata["prediction"] = prediction
        if controller_state is not None:
            metadata["controller_state"] = dict(controller_state)
        return HarnessOutcome(sample=sample, status=status, metadata=metadata)


def _safe_error_message(exc: BaseException) -> str:
    message = " ".join(str(exc).split())[:500]
    message = re.sub(r"Bearer\s+[^\s,;]+", "Bearer <redacted>", message, flags=re.IGNORECASE)
    return re.sub(r"(?i)(api[_-]?key|app[_-]?key)(\s*[=:]\s*)[^\s,;]+", r"\1\2<redacted>", message)


__all__ = ["ARealDeepResearchHarness", "ARealDeepResearchHarnessConfig"]
