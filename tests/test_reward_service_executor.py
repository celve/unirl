"""Tests for RewardServiceExecutor.

Covers:
- Request conversion (DiffusionRL → RewardService wire format)
- Response conversion (RewardService → DiffusionRL)
- Sub-metric reduction strategies
- Weighted aggregation across rewards
- Error handling and partial failures
- Health check / is_available
- Config validation (reward_service backend)
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Direct-import helpers: bypass heavy __init__.py chains
# ---------------------------------------------------------------------------

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)


def _load_module_directly(dotpath: str, filepath: str):
    """Import a single .py file without triggering its package __init__.py."""
    parts = dotpath.split(".")
    for i in range(1, len(parts)):
        parent = ".".join(parts[:i])
        if parent not in sys.modules:
            pkg = types.ModuleType(parent)
            pkg.__path__ = [os.path.join(_PROJECT_ROOT, *parts[:i])]
            pkg.__package__ = parent
            sys.modules[parent] = pkg

    spec = importlib.util.spec_from_file_location(dotpath, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotpath] = mod
    spec.loader.exec_module(mod)
    return mod


# Load only the leaf modules we need (avoids ray, heavy ML deps, etc.)
_load_module_directly(
    "diffusionrl.utils.batched",
    os.path.join(_PROJECT_ROOT, "diffusionrl", "utils", "batched.py"),
)
_load_module_directly(
    "diffusionrl.types.reward",
    os.path.join(_PROJECT_ROOT, "diffusionrl", "types", "reward.py"),
)
_load_module_directly(
    "diffusionrl.reward.base",
    os.path.join(_PROJECT_ROOT, "diffusionrl", "reward", "base.py"),
)
_rse_mod = _load_module_directly(
    "diffusionrl.reward.reward_service_executor",
    os.path.join(_PROJECT_ROOT, "diffusionrl", "reward", "reward_service_executor.py"),
)

RewardServiceExecutor = _rse_mod.RewardServiceExecutor
_encode_image_b64 = _rse_mod._encode_image_b64
_pil_from_tensor = _rse_mod._pil_from_tensor

from diffusionrl.types.reward import RewardRequest  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_pil_image(width: int = 64, height: int = 64) -> Image.Image:
    return Image.new("RGB", (width, height), color=(128, 64, 32))


def _make_tensor_image(channels: int = 3, height: int = 64, width: int = 64) -> torch.Tensor:
    return torch.rand(channels, height, width)


def _make_executor(**overrides) -> RewardServiceExecutor:
    defaults = dict(
        base_url="http://localhost:8080",
        required_rewards=["hpsv2", "clip"],
        reward_weights={"hpsv2": 0.6, "clip": 0.4},
        model_name="reward_service",
        timeout=10.0,
        max_retries=1,
        retry_delay=0.0,
    )
    defaults.update(overrides)
    return RewardServiceExecutor(**defaults)


def _make_reward_request(n: int = 3) -> RewardRequest:
    return RewardRequest(
        images=[_make_pil_image() for _ in range(n)],
        prompts=[f"prompt_{i}" for i in range(n)],
    )


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------


class TestEncodeImageB64:
    def test_pil_image_roundtrip(self):
        img = _make_pil_image()
        b64 = _encode_image_b64(img, image_format="PNG")
        assert isinstance(b64, str) and len(b64) > 0

    def test_tensor_image_roundtrip(self):
        tensor = _make_tensor_image()
        b64 = _encode_image_b64(tensor, image_format="JPEG", quality=90)
        assert isinstance(b64, str) and len(b64) > 0

    def test_rgba_converted_to_rgb(self):
        rgba = Image.new("RGBA", (32, 32), color=(128, 64, 32, 255))
        b64 = _encode_image_b64(rgba, image_format="JPEG")
        assert isinstance(b64, str)


class TestPilFromTensor:
    def test_float_tensor(self):
        t = torch.rand(3, 16, 16)
        img = _pil_from_tensor(t)
        assert isinstance(img, Image.Image) and img.size == (16, 16)

    def test_byte_tensor(self):
        t = torch.randint(0, 256, (3, 16, 16), dtype=torch.uint8)
        img = _pil_from_tensor(t)
        assert isinstance(img, Image.Image)


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


class TestExecutorInit:
    def test_empty_required_rewards_raises(self):
        with pytest.raises(ValueError, match="required_rewards"):
            RewardServiceExecutor(base_url="http://localhost:8080", required_rewards=[])

    def test_bad_reduce_strategy_raises(self):
        with pytest.raises(ValueError, match="sub_metric_reduce"):
            RewardServiceExecutor(
                base_url="http://localhost:8080",
                required_rewards=["hpsv2"],
                sub_metric_reduce="invalid",
            )

    def test_default_construction(self):
        ex = _make_executor()
        assert ex.required_rewards == ["hpsv2", "clip"]
        assert ex.reward_weights == {"hpsv2": 0.6, "clip": 0.4}

    def test_zero_max_retries_raises(self):
        with pytest.raises(ValueError, match="max_retries"):
            RewardServiceExecutor(
                base_url="http://localhost:8080",
                required_rewards=["hpsv2"],
                max_retries=0,
            )

    def test_negative_retry_delay_raises(self):
        with pytest.raises(ValueError, match="retry_delay"):
            RewardServiceExecutor(
                base_url="http://localhost:8080",
                required_rewards=["hpsv2"],
                retry_delay=-1.0,
            )


# ---------------------------------------------------------------------------
# Request conversion
# ---------------------------------------------------------------------------


class TestBuildScorePayload:
    def test_basic_conversion(self):
        ex = _make_executor()
        payload = ex._build_score_payload(_make_reward_request(n=2))
        assert len(payload["requests"]) == 2
        for entry in payload["requests"]:
            assert len(entry["history"]) == 1
            assert "text" in entry["history"][0]
            assert "image_b64" in entry["history"][0]
            assert entry["required_rewards"] == ["hpsv2", "clip"]

    def test_prompts_aligned_with_images(self):
        ex = _make_executor()
        payload = ex._build_score_payload(_make_reward_request(n=3))
        for i, entry in enumerate(payload["requests"]):
            assert entry["history"][0]["text"] == f"prompt_{i}"

    def test_tensor_images_encoded(self):
        ex = _make_executor()
        request = RewardRequest(
            images=[_make_tensor_image(), _make_tensor_image()],
            prompts=["a", "b"],
        )
        payload = ex._build_score_payload(request)
        assert len(payload["requests"]) == 2

    def test_empty_request(self):
        ex = _make_executor()
        payload = ex._build_score_payload(RewardRequest(images=[], prompts=[]))
        assert payload == {"requests": []}


# ---------------------------------------------------------------------------
# Response conversion
# ---------------------------------------------------------------------------


class TestParseScoreResponse:
    def test_normal_response(self):
        ex = _make_executor(
            required_rewards=["hpsv2", "clip"],
            reward_weights={"hpsv2": 0.6, "clip": 0.4},
        )
        raw = {
            "results": [
                {"hpsv2": {"score": 1.0}, "clip": {"similarity": 0.5}},
                {"hpsv2": {"score": 0.8}, "clip": {"similarity": 0.6}},
            ],
            "errors": [{}, {}],
        }
        resp = ex._parse_score_response(raw, batch_size=2, compute_time=1.0)
        assert len(resp.rewards) == 2
        assert resp.component_rewards["hpsv2"] == [1.0, 0.8]
        assert resp.component_rewards["clip"] == [0.5, 0.6]
        assert abs(resp.rewards[0] - 0.8) < 1e-6  # (1.0*0.6 + 0.5*0.4) / 1.0
        assert abs(resp.rewards[1] - 0.72) < 1e-6  # (0.8*0.6 + 0.6*0.4) / 1.0
        assert resp.successes == [True, True]

    def test_partial_reward_failure(self):
        ex = _make_executor(required_rewards=["hpsv2", "clip"])
        raw = {
            "results": [{"hpsv2": {"score": 0.9}}],
            "errors": [{"clip": "TimeoutError"}],
        }
        resp = ex._parse_score_response(raw, batch_size=1, compute_time=0.5)
        assert resp.rewards[0] > 0.0
        assert resp.component_rewards["clip"] == [0.0]
        assert resp.successes == [False]
        assert "clip" in resp.errors[0]

    def test_all_rewards_fail(self):
        ex = _make_executor(required_rewards=["hpsv2"])
        raw = {"results": [{}], "errors": [{"hpsv2": "OOM"}]}
        resp = ex._parse_score_response(raw, batch_size=1, compute_time=0.1)
        assert resp.rewards == [0.0]
        assert resp.successes == [False]

    def test_padding_short_response(self):
        ex = _make_executor(required_rewards=["hpsv2"])
        raw = {"results": [{"hpsv2": {"score": 0.7}}], "errors": [{}]}
        resp = ex._parse_score_response(raw, batch_size=3, compute_time=0.1)
        assert len(resp.rewards) == 3
        assert resp.rewards[0] == pytest.approx(0.7)
        assert resp.rewards[1] == 0.0 and resp.rewards[2] == 0.0


# ---------------------------------------------------------------------------
# Sub-metric reduction
# ---------------------------------------------------------------------------


class TestReduceSubMetrics:
    def test_first(self):
        assert _make_executor(sub_metric_reduce="first")._reduce_sub_metrics({"a": 0.5, "b": 0.9}) == 0.5

    def test_mean(self):
        assert _make_executor(sub_metric_reduce="mean")._reduce_sub_metrics({"a": 0.4, "b": 0.6}) == pytest.approx(0.5)

    def test_max(self):
        assert _make_executor(sub_metric_reduce="max")._reduce_sub_metrics({"a": 0.3, "b": 0.9}) == 0.9

    def test_empty(self):
        assert _make_executor()._reduce_sub_metrics({}) == 0.0


# ---------------------------------------------------------------------------
# End-to-end compute_rewards (mocked HTTP)
# ---------------------------------------------------------------------------


class TestComputeRewardsMocked:
    def test_success(self):
        ex = _make_executor(required_rewards=["hpsv2"], reward_weights={"hpsv2": 1.0})
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "results": [{"hpsv2": {"score": 0.85}}, {"hpsv2": {"score": 0.72}}],
            "errors": [{}, {}],
        }
        with patch.object(ex._session, "post", return_value=mock_resp):
            resp = ex.compute_rewards(_make_reward_request(n=2))
        assert resp.rewards[0] == pytest.approx(0.85)
        assert resp.rewards[1] == pytest.approx(0.72)
        assert all(resp.successes)

    def test_http_failure_raises_by_default(self):
        """Default raise_on_failure=True: exceptions propagate."""
        import requests as http_requests

        ex = _make_executor(max_retries=1, retry_delay=0.0)
        with patch.object(
            ex._session,
            "post",
            side_effect=http_requests.exceptions.ConnectionError("Connection refused"),
        ):
            with pytest.raises(RuntimeError, match="failed after 1 retries"):
                ex.compute_rewards(_make_reward_request(n=2))

    def test_http_failure_degraded_mode(self):
        """raise_on_failure=False: returns zeroed rewards instead of raising."""
        import requests as http_requests

        ex = _make_executor(max_retries=1, retry_delay=0.0, raise_on_failure=False)
        with patch.object(
            ex._session,
            "post",
            side_effect=http_requests.exceptions.ConnectionError("Connection refused"),
        ):
            resp = ex.compute_rewards(_make_reward_request(n=2))
        assert all(r == 0.0 for r in resp.rewards)
        assert all(not s for s in resp.successes)


# ---------------------------------------------------------------------------
# is_available
# ---------------------------------------------------------------------------


class TestIsAvailable:
    @staticmethod
    def _mock_health_resp(status_code: int, json_payload):
        m = MagicMock(status_code=status_code)
        if isinstance(json_payload, Exception):
            m.json.side_effect = json_payload
        else:
            m.json.return_value = json_payload
        return m

    def test_healthy_all_required_rewards_present(self):
        """Server roster is a superset of required_rewards -> True, log once."""
        ex = _make_executor(required_rewards=["hpsv2", "clip"])
        payload = {
            "status": "ok",
            "rewards": {
                "hpsv2": ["hpsv2:ready"],
                "clip": ["clip:ready", "clip:ready"],
                "pickscore": ["pickscore:ready"],
            },
        }
        with patch.object(ex._session, "get", return_value=self._mock_health_resp(200, payload)):
            assert ex.is_available() is True

    def test_unreachable(self):
        import requests as http_requests

        ex = _make_executor()
        with patch.object(ex._session, "get", side_effect=http_requests.exceptions.ConnectionError("refused")):
            assert ex.is_available() is False

    def test_non_200_status_returns_false(self):
        ex = _make_executor(required_rewards=["hpsv2"])
        with patch.object(ex._session, "get", return_value=self._mock_health_resp(503, {})):
            assert ex.is_available() is False

    def test_missing_reward_raises(self):
        """The exact 'unifiedreward' typo case the user hit -- caught at /health."""
        ex = _make_executor(required_rewards=["hpsv2", "unifiedreward"])
        payload = {
            "status": "ok",
            "rewards": {
                "hpsv2": ["hpsv2:ready"],
                "unified_reward": ["unified_reward:ready"],
                "pickscore": ["pickscore:ready"],
            },
        }
        with patch.object(ex._session, "get", return_value=self._mock_health_resp(200, payload)):
            with pytest.raises(ValueError, match="unifiedreward.*not served") as excinfo:
                ex.is_available()
        # The message must list the available roster so the user can spot the typo.
        assert "unified_reward" in str(excinfo.value)
        assert ex.base_url in str(excinfo.value)

    def test_bad_body_shape_raises(self):
        """/health body without a 'rewards' dict -> ValueError, not silent True."""
        ex = _make_executor(required_rewards=["hpsv2"])
        with patch.object(ex._session, "get", return_value=self._mock_health_resp(200, {"status": "ok"})):
            with pytest.raises(ValueError, match="unexpected shape"):
                ex.is_available()

    def test_non_json_body_raises(self):
        ex = _make_executor(required_rewards=["hpsv2"])
        with patch.object(
            ex._session,
            "get",
            return_value=self._mock_health_resp(200, ValueError("not json")),
        ):
            with pytest.raises(ValueError, match="non-JSON body"):
                ex.is_available()

    def test_validation_runs_only_once(self):
        """Repeated is_available() calls do NOT re-raise (or re-log) on second hit."""
        ex = _make_executor(required_rewards=["hpsv2"])
        payload = {"status": "ok", "rewards": {"hpsv2": ["hpsv2:ready"]}}
        with patch.object(ex._session, "get", return_value=self._mock_health_resp(200, payload)) as mock_get:
            assert ex.is_available() is True
            assert ex.is_available() is True
            # Both pings hit the network (operational signal still cheap), but the
            # validation parsed the body once: the flag short-circuits subsequent
            # calls so the user only sees one INFO log line.
            assert mock_get.call_count == 2
            assert ex._remote_rewards_validated is True


# ---------------------------------------------------------------------------
# Config validation (cmdline/schema.py RewardConfig)
# ---------------------------------------------------------------------------


class TestRewardConfigValidation:
    """Test RewardConfig.validate() for the reward_service backend."""

    @staticmethod
    def _load_reward_config_class():
        class _StubModule(types.ModuleType):
            def __init__(self, name):
                super().__init__(name)
                self.__path__ = []  # Package-like so sub-imports resolve

            def __getattr__(self, name):
                if name == "__path__":
                    return []
                return lambda *a, **kw: None

        stubs = [
            "diffusionrl.cmdline.validation",
            "diffusionrl.algorithms",
            "diffusionrl.algorithms.base",
            "diffusionrl.algorithms.construction",
            "diffusionrl.algorithms.registry",
            "diffusionrl.algorithms.grpo",
            "diffusionrl.cmdline.algorithms",
            "diffusionrl.config.validation",
        ]
        originals = {}
        for name in stubs:
            originals[name] = sys.modules.get(name)
            sys.modules[name] = _StubModule(name)
        try:
            spec = importlib.util.spec_from_file_location(
                "diffusionrl.cmdline.schema_isolated",
                os.path.join(_PROJECT_ROOT, "diffusionrl", "cmdline", "schema.py"),
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.RewardConfig
        finally:
            for name in stubs:
                if originals[name] is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = originals[name]

    def test_reward_service_requires_url(self):
        RC = self._load_reward_config_class()
        cfg = RC(reward_backend="reward_service", reward_service_url=None, reward_components=["hpsv2"])
        with pytest.raises(ValueError, match="reward_service_url"):
            cfg.validate()

    def test_reward_service_requires_components(self):
        RC = self._load_reward_config_class()
        cfg = RC(reward_backend="reward_service", reward_service_url="http://localhost", reward_components=None)
        with pytest.raises(ValueError, match="reward_components"):
            cfg.validate()

    def test_reward_service_valid(self):
        RC = self._load_reward_config_class()
        cfg = RC(
            reward_backend="reward_service",
            reward_service_url="http://localhost:8080",
            reward_components=["hpsv2", "clip"],
            reward_weights=[0.6, 0.4],
        )
        cfg.validate()  # Should not raise

    def test_reward_service_rejects_concat_aggregation(self):
        # 'concat' is a cross-executor flatten op in reward.aggregation.AGGREGATORS;
        # the reward_service backend multiplexes inside a single executor and
        # cannot honour it. We want this caught at args-parse time, well before
        # Ray spins up and surfaces the late check inside RewardServiceExecutor.
        RC = self._load_reward_config_class()
        cfg = RC(
            reward_backend="reward_service",
            reward_service_url="http://localhost:8080",
            reward_components=["hpsv2", "clip"],
            reward_weights=[0.6, 0.4],
            reward_aggregation_method="concat",
        )
        with pytest.raises(ValueError, match="concat.*not supported.*reward_service"):
            cfg.validate()


# ---------------------------------------------------------------------------
# Aggregation methods
# ---------------------------------------------------------------------------


class TestAggregationMethods:
    """Test _aggregate_scores with different aggregation_method settings."""

    def test_weighted_sum(self):
        ex = _make_executor(aggregation_method="weighted_sum", reward_weights={"hpsv2": 0.6, "clip": 0.4})
        raw = {
            "results": [{"hpsv2": {"score": 1.0}, "clip": {"score": 0.5}}],
            "errors": [{}],
        }
        resp = ex._parse_score_response(raw, batch_size=1, compute_time=0.1)
        # (1.0*0.6 + 0.5*0.4) / (0.6+0.4) = 0.8
        assert resp.rewards[0] == pytest.approx(0.8)

    def test_mean(self):
        ex = _make_executor(aggregation_method="mean")
        raw = {
            "results": [{"hpsv2": {"score": 0.8}, "clip": {"score": 0.4}}],
            "errors": [{}],
        }
        resp = ex._parse_score_response(raw, batch_size=1, compute_time=0.1)
        # (0.8 + 0.4) / 2 = 0.6
        assert resp.rewards[0] == pytest.approx(0.6)

    def test_min(self):
        ex = _make_executor(aggregation_method="min")
        raw = {
            "results": [{"hpsv2": {"score": 0.9}, "clip": {"score": 0.3}}],
            "errors": [{}],
        }
        resp = ex._parse_score_response(raw, batch_size=1, compute_time=0.1)
        assert resp.rewards[0] == pytest.approx(0.3)

    def test_max(self):
        ex = _make_executor(aggregation_method="max")
        raw = {
            "results": [{"hpsv2": {"score": 0.2}, "clip": {"score": 0.7}}],
            "errors": [{}],
        }
        resp = ex._parse_score_response(raw, batch_size=1, compute_time=0.1)
        assert resp.rewards[0] == pytest.approx(0.7)

    def test_invalid_aggregation_method_raises(self):
        with pytest.raises(ValueError, match="aggregation_method"):
            _make_executor(aggregation_method="invalid")


# ---------------------------------------------------------------------------
# _pil_from_tensor robustness
# ---------------------------------------------------------------------------


class TestPilFromTensorRobust:
    def test_float_0_to_255_range(self):
        """Float tensor in 0–255 range should be normalized to 0–1."""
        t = torch.tensor([[[128.0, 64.0], [32.0, 255.0]]] * 3)  # 3xHxW
        img = _pil_from_tensor(t)
        assert isinstance(img, Image.Image)

    def test_gpu_tensor_moved_to_cpu(self):
        """Tensor should be detached and moved to CPU (no error even on CPU-only)."""
        t = torch.rand(3, 8, 8, requires_grad=True)
        img = _pil_from_tensor(t)
        assert isinstance(img, Image.Image)

    def test_clamping_out_of_range(self):
        """Values > 255 or < 0 should be clamped."""
        t = torch.tensor([[[300.0, -10.0], [128.0, 0.0]]] * 3)
        img = _pil_from_tensor(t)
        assert isinstance(img, Image.Image)
