"""Tests for farm/core/decision/training/collector.py – ExperienceCollector."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from farm.core.decision.training.collector import ExperienceCollector


def _make_agent_and_env(action_name="move", reward=1.0, terminated=False, truncated=False):
    """Build minimal mock agent and environment for ExperienceCollector."""
    agent = MagicMock()
    # decide_action returns an object with a .name attribute
    mock_action = MagicMock()
    mock_action.name = action_name
    agent.decide_action.return_value = mock_action

    env = MagicMock()
    env.reset.return_value = None
    # step returns (next_state, reward, terminated, truncated, info)
    env.step.return_value = (np.zeros(4), reward, terminated, truncated, {})

    return agent, env


class TestExperienceCollector:
    def test_init_creates_feature_engineer(self):
        from farm.core.decision.feature_engineering import FeatureEngineer
        collector = ExperienceCollector()
        assert isinstance(collector.feature_engineer, FeatureEngineer)

    def test_collect_episode_returns_list_of_tuples(self):
        collector = ExperienceCollector()
        agent, env = _make_agent_and_env(terminated=False, truncated=False)

        # Mock feature extraction to return a fixed vector
        collector.feature_engineer.extract_features = MagicMock(return_value=np.zeros(9))

        with patch("farm.core.decision.training.collector.action_name_to_index", return_value=0):
            data = collector.collect_episode(agent, env, max_steps=3)

        assert len(data) == 3
        for state, action, reward in data:
            assert isinstance(state, np.ndarray)
            assert isinstance(action, int)
            assert isinstance(reward, float)

    def test_collect_episode_stops_on_terminated(self):
        collector = ExperienceCollector()
        agent, env = _make_agent_and_env(terminated=True)
        collector.feature_engineer.extract_features = MagicMock(return_value=np.zeros(9))

        with patch("farm.core.decision.training.collector.action_name_to_index", return_value=1):
            data = collector.collect_episode(agent, env, max_steps=100)

        # terminated=True on first step → only 1 experience collected
        assert len(data) == 1

    def test_collect_episode_stops_on_truncated(self):
        collector = ExperienceCollector()
        agent, env = _make_agent_and_env(truncated=True)
        collector.feature_engineer.extract_features = MagicMock(return_value=np.zeros(9))

        with patch("farm.core.decision.training.collector.action_name_to_index", return_value=2):
            data = collector.collect_episode(agent, env, max_steps=100)

        assert len(data) == 1

    def test_collect_episode_calls_env_reset(self):
        collector = ExperienceCollector()
        agent, env = _make_agent_and_env()
        collector.feature_engineer.extract_features = MagicMock(return_value=np.zeros(9))

        with patch("farm.core.decision.training.collector.action_name_to_index", return_value=0):
            collector.collect_episode(agent, env, max_steps=2)

        env.reset.assert_called_once()

    def test_collect_episode_calls_decide_action(self):
        collector = ExperienceCollector()
        agent, env = _make_agent_and_env()
        collector.feature_engineer.extract_features = MagicMock(return_value=np.zeros(9))

        with patch("farm.core.decision.training.collector.action_name_to_index", return_value=0):
            data = collector.collect_episode(agent, env, max_steps=5)

        assert agent.decide_action.call_count == 5

    def test_collect_episode_passes_action_index_to_env_step(self):
        collector = ExperienceCollector()
        agent, env = _make_agent_and_env()
        collector.feature_engineer.extract_features = MagicMock(return_value=np.zeros(9))

        with patch("farm.core.decision.training.collector.action_name_to_index", return_value=3) as mock_idx:
            data = collector.collect_episode(agent, env, max_steps=2)

        env.step.assert_called_with(3)

    def test_collect_episode_empty_when_max_steps_zero(self):
        collector = ExperienceCollector()
        agent, env = _make_agent_and_env()
        collector.feature_engineer.extract_features = MagicMock(return_value=np.zeros(9))

        with patch("farm.core.decision.training.collector.action_name_to_index", return_value=0):
            data = collector.collect_episode(agent, env, max_steps=0)

        assert data == []
