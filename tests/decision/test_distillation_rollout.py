"""Tests for synthetic MDP rollout comparison (distillation issue #597)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from farm.core.decision.base_dqn import BaseQNetwork
from farm.core.decision.training.distillation_rollout import (
    SeededLinearMDP,
    compare_parent_student_rollouts,
    relative_return_drop,
)


class _ConstQ(nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("_logits", logits.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._logits.expand(x.shape[0], -1).clone()


def test_seeded_linear_mdp_reset_deterministic():
    env = SeededLinearMDP(4, 2, base_seed=999, max_steps=10)
    a = env.reset(12345)
    b = env.reset(12345)
    np.testing.assert_array_almost_equal(a, b)


def test_seeded_linear_mdp_step_truncates():
    env = SeededLinearMDP(3, 2, base_seed=1, max_steps=2)
    env.reset(0)
    _, _, _, trunc = env.step(0)
    assert not trunc
    _, _, _, trunc = env.step(0)
    assert trunc


def test_relative_return_drop_positive_parent():
    assert relative_return_drop(100.0, 90.0) == pytest.approx(0.1)


def test_relative_return_drop_negative_parent():
    assert relative_return_drop(-100.0, -110.0) == pytest.approx(0.1)


def test_same_module_identical_rollout_means():
    net = BaseQNetwork(input_dim=4, output_dim=2, hidden_size=16)
    torch.manual_seed(0)
    result = compare_parent_student_rollouts(
        net,
        net,
        obs_dim=4,
        n_actions=2,
        base_seed=42,
        n_episodes=8,
        max_steps=15,
        device=torch.device("cpu"),
        max_relative_return_drop=None,
    )
    assert result.parent_mean_return == pytest.approx(result.student_mean_return)
    assert result.passed is None


def test_rollout_threshold_can_fail():
    parent = _ConstQ(torch.tensor([2.0, 0.0, 0.0, 0.0]))
    student = _ConstQ(torch.tensor([0.0, 2.0, 0.0, 0.0]))
    result = compare_parent_student_rollouts(
        parent,
        student,
        obs_dim=4,
        n_actions=4,
        base_seed=7,
        n_episodes=24,
        max_steps=20,
        device=torch.device("cpu"),
        max_relative_return_drop=0.0,
    )
    assert result.passed is False


def test_rollout_threshold_passes_for_identical_policies():
    logits = torch.tensor([1.0, 0.5, 0.0, 0.0])
    policy = _ConstQ(logits)
    result = compare_parent_student_rollouts(
        policy,
        policy,
        obs_dim=4,
        n_actions=4,
        base_seed=3,
        n_episodes=10,
        max_steps=12,
        device=torch.device("cpu"),
        max_relative_return_drop=0.0,
    )
    assert result.passed is True


def test_compare_requires_positive_n_episodes():
    net = BaseQNetwork(input_dim=2, output_dim=2, hidden_size=4)
    with pytest.raises(ValueError, match="n_episodes"):
        compare_parent_student_rollouts(
            net,
            net,
            obs_dim=2,
            n_actions=2,
            base_seed=1,
            n_episodes=0,
            max_steps=5,
            device=torch.device("cpu"),
        )
