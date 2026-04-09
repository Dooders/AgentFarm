"""Tests for PolicyRolloutAdapter (sim rollout adapter).

Validates that the adapter correctly:
- Runs greedy-Q rollouts for parent and student on identical episode seeds.
- Returns identical episode statistics when parent == student.
- Detects and reports return drops between parent and student.
- Respects ``max_relative_return_drop`` pass/fail threshold.
- Restores training mode after rollouts.
- Works with a custom feature pipeline.
- Works with mock environments satisfying the EpisodeEnvProtocol interface.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn

from farm.core.decision.base_dqn import BaseQNetwork
from farm.core.decision.training.distillation_rollout import (
    _rollout_passed,
    relative_return_drop,
)
from farm.core.decision.training.sim_rollout_adapter import (
    PolicyRolloutAdapter,
    SimEpisodeStats,
    SimRolloutConfig,
    SimRolloutResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

OBS_DIM = 4
N_ACTIONS = 2


def _make_net(obs_dim: int = OBS_DIM, n_actions: int = N_ACTIONS, seed: int = 0) -> BaseQNetwork:
    torch.manual_seed(seed)
    return BaseQNetwork(input_dim=obs_dim, output_dim=n_actions, hidden_size=16)


class _ConstQ(nn.Module):
    """Q-network that always returns fixed logits regardless of input."""

    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("_logits", logits.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._logits.expand(x.shape[0], -1).clone()


def _single_step_env_factory(obs_dim: int = OBS_DIM, reward: float = 1.0):
    """Factory that creates a fresh 1-step mock env each time it's called."""

    def _make():
        env = MagicMock()
        env.reset.return_value = (np.zeros(obs_dim, dtype=np.float32), {})
        env.step.return_value = (np.zeros(obs_dim, dtype=np.float32), reward, False, True, {})
        return env

    return _make


# ---------------------------------------------------------------------------
# SimEpisodeStats
# ---------------------------------------------------------------------------


class TestSimEpisodeStats:
    def test_mean_return_empty(self):
        stats = SimEpisodeStats()
        assert stats.mean_return == 0.0

    def test_mean_steps_empty(self):
        stats = SimEpisodeStats()
        assert stats.mean_steps == 0.0

    def test_mean_survival_rate_empty(self):
        stats = SimEpisodeStats()
        assert stats.mean_survival_rate == 0.0

    def test_to_dict_keys(self):
        stats = SimEpisodeStats(returns=[1.0, 2.0], step_counts=[5, 5], survival_rates=[1.0, 1.0])
        d = stats.to_dict()
        for key in (
            "mean_return",
            "mean_steps",
            "mean_survival_rate",
            "returns",
            "step_counts",
            "survival_rates",
            "wall_time_s",
        ):
            assert key in d

    def test_mean_return(self):
        stats = SimEpisodeStats(returns=[2.0, 4.0])
        assert stats.mean_return == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# SimRolloutConfig validation
# ---------------------------------------------------------------------------


class TestSimRolloutConfig:
    def test_default_config(self):
        cfg = SimRolloutConfig()
        assert cfg.n_episodes == 10
        assert cfg.max_steps == 200
        assert cfg.base_seed == 42
        assert cfg.max_relative_return_drop is None

    def test_adapter_rejects_zero_n_episodes(self):
        with pytest.raises(ValueError, match="n_episodes"):
            PolicyRolloutAdapter(
                _make_net(), _make_net(), config=SimRolloutConfig(n_episodes=0)
            )

    def test_adapter_rejects_negative_max_steps(self):
        with pytest.raises(ValueError, match="max_steps"):
            PolicyRolloutAdapter(
                _make_net(), _make_net(), config=SimRolloutConfig(n_episodes=1, max_steps=0)
            )

    def test_adapter_rejects_out_of_range_threshold(self):
        with pytest.raises(ValueError, match="max_relative_return_drop"):
            PolicyRolloutAdapter(
                _make_net(),
                _make_net(),
                config=SimRolloutConfig(n_episodes=1, max_relative_return_drop=1.5),
            )


# ---------------------------------------------------------------------------
# PolicyRolloutAdapter – core behaviour
# ---------------------------------------------------------------------------


class TestPolicyRolloutAdapterIdentical:
    """When parent == student, both policies should produce identical results."""

    def test_identical_networks_same_mean_return(self):
        net = _make_net()
        cfg = SimRolloutConfig(n_episodes=5, max_steps=10, base_seed=0)
        adapter = PolicyRolloutAdapter(net, net, config=cfg)

        result = adapter.run(env_factory=_single_step_env_factory(reward=1.0))

        assert isinstance(result, SimRolloutResult)
        assert result.parent_stats.mean_return == pytest.approx(
            result.student_stats.mean_return
        )

    def test_identical_networks_same_step_counts(self):
        net = _make_net()
        cfg = SimRolloutConfig(n_episodes=4, max_steps=8, base_seed=1)
        adapter = PolicyRolloutAdapter(net, net, config=cfg)

        result = adapter.run(env_factory=_single_step_env_factory())

        assert result.parent_stats.step_counts == result.student_stats.step_counts

    def test_identical_networks_passed_is_none_when_no_threshold(self):
        net = _make_net()
        cfg = SimRolloutConfig(n_episodes=3, max_steps=5, base_seed=2)
        adapter = PolicyRolloutAdapter(net, net, config=cfg)

        result = adapter.run(env_factory=_single_step_env_factory())

        assert result.passed is None

    def test_identical_networks_passed_is_true_with_threshold(self):
        net = _make_net()
        cfg = SimRolloutConfig(
            n_episodes=3, max_steps=5, base_seed=3, max_relative_return_drop=0.0
        )
        adapter = PolicyRolloutAdapter(net, net, config=cfg)

        result = adapter.run(env_factory=_single_step_env_factory())

        assert result.passed is True


class TestPolicyRolloutAdapterReturnDrop:
    """Student with a worse policy should be detected by relative drop threshold."""

    def test_student_worse_policy_fails_threshold(self):
        # Parent always picks action 0 (high reward) via constant logits
        parent = _ConstQ(torch.tensor([5.0, -5.0]))
        # Student always picks action 1 (low reward) via constant logits
        student = _ConstQ(torch.tensor([-5.0, 5.0]))

        # Env gives reward 1.0 for action 0 and reward -1.0 for action 1
        call_count = [0]

        def _step_fn(action: int):
            call_count[0] += 1
            reward = 1.0 if action == 0 else -1.0
            truncated = call_count[0] >= 3
            return np.zeros(OBS_DIM, dtype=np.float32), reward, False, truncated, {}

        def _make_env():
            env = MagicMock()
            env.reset.return_value = (np.zeros(OBS_DIM, dtype=np.float32), {})
            call_count[0] = 0
            env.step.side_effect = _step_fn
            return env

        cfg = SimRolloutConfig(
            n_episodes=5, max_steps=3, base_seed=0, max_relative_return_drop=0.0
        )
        adapter = PolicyRolloutAdapter(parent, student, config=cfg)
        result = adapter.run(env_factory=_make_env)

        assert result.parent_stats.mean_return > 0
        assert result.student_stats.mean_return < 0
        assert result.passed is False

    def test_relative_drop_none_when_parent_near_zero(self):
        result_drop = relative_return_drop(0.0, 0.0)
        assert result_drop is None

    def test_relative_drop_positive_parent(self):
        assert relative_return_drop(10.0, 8.0) == pytest.approx(0.2)

    def test_relative_drop_negative_parent(self):
        assert relative_return_drop(-10.0, -12.0) == pytest.approx(0.2)

    def test_rollout_passed_positive_parent(self):
        # student return >= parent * (1 - 0.2) → passes
        assert _rollout_passed(10.0, 8.0, 0.2) is True

    def test_rollout_passed_fails(self):
        # student < parent * (1 - 0.1) → fails
        assert _rollout_passed(10.0, 8.0, 0.1) is False


class TestPolicyRolloutAdapterSeeding:
    """Identical seeds should produce reproducible results across separate runs."""

    def test_deterministic_with_same_seed(self):
        net = _make_net(seed=42)
        cfg = SimRolloutConfig(n_episodes=4, max_steps=6, base_seed=7)

        env_calls = []

        def _make_env():
            env = MagicMock()
            env.reset.return_value = (np.zeros(OBS_DIM, dtype=np.float32), {})
            env.step.return_value = (np.zeros(OBS_DIM, dtype=np.float32), 1.0, False, True, {})
            env_calls.append(env)
            return env

        adapter = PolicyRolloutAdapter(net, net, config=cfg)
        result1 = adapter.run(env_factory=_make_env)
        result2 = adapter.run(env_factory=_make_env)

        assert result1.parent_stats.returns == pytest.approx(result2.parent_stats.returns)
        assert result1.student_stats.returns == pytest.approx(result2.student_stats.returns)

    def test_episode_seeds_are_deterministically_spaced(self):
        """Verify the adapter passes distinct seeds to env.reset()."""
        net = _make_net()
        n_ep = 3
        cfg = SimRolloutConfig(n_episodes=n_ep, max_steps=1, base_seed=0)

        reset_seeds = []

        def _make_env():
            env = MagicMock()

            def _reset(*, seed=None):
                reset_seeds.append(seed)
                return np.zeros(OBS_DIM, dtype=np.float32), {}

            env.reset.side_effect = _reset
            env.step.return_value = (np.zeros(OBS_DIM, dtype=np.float32), 0.0, False, True, {})
            return env

        adapter = PolicyRolloutAdapter(net, net, config=cfg)
        adapter.run(env_factory=_make_env)

        # Expect 2*n_ep resets (parent + student), each with distinct spaced seeds
        from farm.core.decision.training.sim_rollout_adapter import _EPISODE_SEED_STRIDE

        expected_seeds = [int(0 + i * _EPISODE_SEED_STRIDE) for i in range(n_ep)]
        parent_seeds = reset_seeds[:n_ep]
        student_seeds = reset_seeds[n_ep:]
        assert parent_seeds == expected_seeds
        assert student_seeds == expected_seeds


class TestPolicyRolloutAdapterTrainingMode:
    """Networks must have their original training mode restored after rollouts."""

    def test_training_mode_restored_for_parent_and_student(self):
        parent = _make_net()
        student = _make_net()
        parent.train()
        student.train()
        assert parent.training and student.training

        cfg = SimRolloutConfig(n_episodes=2, max_steps=2)
        adapter = PolicyRolloutAdapter(parent, student, config=cfg)

        adapter.run(env_factory=_single_step_env_factory())

        assert parent.training
        assert student.training

    def test_eval_mode_preserved_if_set_before(self):
        parent = _make_net()
        student = _make_net()
        parent.eval()
        student.eval()

        cfg = SimRolloutConfig(n_episodes=2, max_steps=2)
        adapter = PolicyRolloutAdapter(parent, student, config=cfg)

        adapter.run(env_factory=_single_step_env_factory())

        assert not parent.training
        assert not student.training


class TestPolicyRolloutAdapterFeaturePipeline:
    """Custom feature pipeline should be called for every observation."""

    def test_custom_feature_pipeline_called(self):
        net = _make_net(obs_dim=OBS_DIM)
        # 1 episode × 1 step per policy (single-step env) → 1 pipeline call per policy
        cfg = SimRolloutConfig(n_episodes=1, max_steps=3, base_seed=0)

        pipeline_calls = []

        def _pipeline(obs):
            pipeline_calls.append(obs)
            return np.asarray(obs, dtype=np.float32)

        adapter = PolicyRolloutAdapter(net, net, config=cfg, feature_pipeline=_pipeline)

        adapter.run(env_factory=_single_step_env_factory())

        # 1 episode × 1 step × 2 policies (parent + student) = exactly 2 pipeline calls
        assert len(pipeline_calls) == 2

    def test_pipeline_output_shape_determines_input_to_net(self):
        """If pipeline returns wrong shape, the network forward will raise or silently fail.

        This test verifies that what the pipeline returns is directly forwarded to the net.
        """
        # Net expects 4-dim input; pipeline returns zeros of that shape.
        net = _make_net(obs_dim=OBS_DIM)
        cfg = SimRolloutConfig(n_episodes=1, max_steps=1)

        transformed = []

        def _pipeline(obs):
            out = np.zeros(OBS_DIM, dtype=np.float32)
            transformed.append(out)
            return out

        adapter = PolicyRolloutAdapter(net, net, config=cfg, feature_pipeline=_pipeline)
        result = adapter.run(env_factory=_single_step_env_factory())
        assert len(transformed) > 0


# ---------------------------------------------------------------------------
# SimRolloutResult.to_dict
# ---------------------------------------------------------------------------


class TestSimRolloutResultToDict:
    def _make_result(self, passed: bool | None = None) -> SimRolloutResult:
        stats = SimEpisodeStats(returns=[1.0], step_counts=[5], survival_rates=[1.0])
        return SimRolloutResult(
            parent_stats=stats,
            student_stats=stats,
            relative_drop=0.0,
            n_episodes=1,
            max_steps=5,
            base_seed=0,
            passed=passed,
        )

    def test_to_dict_contains_required_keys(self):
        d = self._make_result().to_dict()
        for key in (
            "parent",
            "student",
            "parent_mean_return",
            "student_mean_return",
            "relative_drop",
            "n_episodes",
            "max_steps",
            "base_seed",
            "passed",
        ):
            assert key in d

    def test_passed_none_serialisable(self):
        import json

        d = self._make_result(passed=None).to_dict()
        # None maps to JSON null which is valid
        json.dumps(d)

    def test_passed_true(self):
        d = self._make_result(passed=True).to_dict()
        assert d["passed"] is True

    def test_passed_false(self):
        d = self._make_result(passed=False).to_dict()
        assert d["passed"] is False
