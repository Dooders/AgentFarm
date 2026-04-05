"""Tests for farm/core/decision/feature_engineering.py – FeatureEngineer."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from farm.core.decision.feature_engineering import FeatureEngineer


def _make_agent(
    position=(5.0, 10.0),
    resource_level=3.0,
    has_resource_comp=True,
    has_reproduction_comp=True,
    has_combat_comp=True,
    has_perception_comp=True,
    is_defending=False,
    starvation_counter=2,
):
    """Build a minimal mock AgentCore suitable for FeatureEngineer."""
    agent = MagicMock()
    agent.position = position
    agent.resource_level = resource_level

    # Resource component
    resource_comp = MagicMock() if has_resource_comp else None
    if resource_comp is not None:
        resource_comp.starvation_counter = starvation_counter
        resource_comp.config.starvation_threshold = 10

    # Reproduction component
    repro_comp = MagicMock() if has_reproduction_comp else None
    if repro_comp is not None:
        repro_comp.config.offspring_cost = 4.0

    # Combat component
    combat_comp = MagicMock() if has_combat_comp else None
    if combat_comp is not None:
        combat_comp.health = 80.0
        combat_comp.config.starting_health = 100.0
        combat_comp.is_defending = is_defending

    # Perception component
    perception_comp = MagicMock() if has_perception_comp else None
    if perception_comp is not None:
        perception_comp.config.perception_radius = 20

    def get_component(name):
        return {
            "resource": resource_comp,
            "reproduction": repro_comp,
            "combat": combat_comp,
            "perception": perception_comp,
        }.get(name)

    agent.get_component = get_component
    return agent


def _make_environment(width=100, height=100, time=50, nearby_resources=None, nearby_agents=None):
    """Build a minimal mock Environment."""
    env = MagicMock()
    env.width = width
    env.height = height
    env.time = time
    env.resources = [object() for _ in range(10)]
    env.agents = [object() for _ in range(5)]

    if nearby_resources is None:
        nearby_resources = [object(), object()]
    if nearby_agents is None:
        nearby_agents = [object()]

    env.get_nearby_resources.return_value = nearby_resources
    env.get_nearby_agents.return_value = nearby_agents
    return env


class TestFeatureEngineerExtractFeatures:
    def test_returns_numpy_array(self):
        fe = FeatureEngineer()
        agent = _make_agent()
        env = _make_environment()
        features = fe.extract_features(agent, env)
        assert isinstance(features, np.ndarray)

    def test_feature_count(self):
        """Expect 9 features total: health, resource, x, y, resource_density,
        starvation_ratio, agent_density, is_defending, time."""
        fe = FeatureEngineer()
        agent = _make_agent()
        env = _make_environment()
        features = fe.extract_features(agent, env)
        assert features.shape == (9,)

    def test_health_ratio_normalized(self):
        fe = FeatureEngineer()
        agent = _make_agent()  # health=80, starting=100 → ratio=0.8
        env = _make_environment()
        features = fe.extract_features(agent, env)
        assert features[0] == pytest.approx(0.8)

    def test_position_normalized(self):
        fe = FeatureEngineer()
        agent = _make_agent(position=(25.0, 50.0))
        env = _make_environment(width=100, height=100)
        features = fe.extract_features(agent, env)
        assert features[2] == pytest.approx(0.25)
        assert features[3] == pytest.approx(0.50)

    def test_time_feature_bounded(self):
        fe = FeatureEngineer()
        agent = _make_agent()
        env = _make_environment(time=1500)  # 1500 % 1000 = 500 → 0.5
        features = fe.extract_features(agent, env)
        assert features[8] == pytest.approx(0.5)

    def test_is_defending_flag(self):
        fe = FeatureEngineer()
        agent_def = _make_agent(is_defending=True)
        agent_off = _make_agent(is_defending=False)
        env = _make_environment()
        assert fe.extract_features(agent_def, env)[7] == 1.0
        assert fe.extract_features(agent_off, env)[7] == 0.0

    def test_missing_combat_component(self):
        """When combat component is absent, health defaults to 0."""
        fe = FeatureEngineer()
        agent = _make_agent(has_combat_comp=False)
        env = _make_environment()
        features = fe.extract_features(agent, env)
        # health/starting_health → 0/100 = 0.0
        assert features[0] == pytest.approx(0.0)

    def test_missing_resource_component(self):
        fe = FeatureEngineer()
        agent = _make_agent(has_resource_comp=False)
        env = _make_environment()
        features = fe.extract_features(agent, env)
        assert features.shape == (9,)

    def test_missing_reproduction_component_uses_default_max(self):
        fe = FeatureEngineer()
        agent = _make_agent(has_reproduction_comp=False, resource_level=12.0)
        env = _make_environment()
        features = fe.extract_features(agent, env)
        # max_resources = 24.0 (default), resource_level=12 → ratio=0.5
        assert features[1] == pytest.approx(0.5)

    def test_missing_perception_component_uses_defaults(self):
        fe = FeatureEngineer()
        agent = _make_agent(has_perception_comp=False)
        env = _make_environment()
        # Should not raise
        features = fe.extract_features(agent, env)
        assert features.shape == (9,)

    def test_get_nearby_resources_raises_fallback(self):
        """If get_nearby_resources raises, resource_density falls back to ~0."""
        fe = FeatureEngineer()
        agent = _make_agent()
        env = _make_environment()
        env.get_nearby_resources.side_effect = AttributeError("boom")
        features = fe.extract_features(agent, env)
        # resource_density = 0 / 10 = 0.0
        assert features[4] == pytest.approx(0.0)

    def test_get_nearby_agents_raises_fallback(self):
        fe = FeatureEngineer()
        agent = _make_agent()
        env = _make_environment()
        env.get_nearby_agents.side_effect = TypeError("boom")
        features = fe.extract_features(agent, env)
        # agent_density = 0 / 5 = 0.0
        assert features[6] == pytest.approx(0.0)

    def test_no_get_nearby_resources_method(self):
        """If environment lacks get_nearby_resources, fallback to empty list."""
        fe = FeatureEngineer()
        agent = _make_agent()
        env = _make_environment()
        del env.get_nearby_resources  # Remove the attribute entirely
        # hasattr check in feature engineering should handle this
        features = fe.extract_features(agent, env)
        assert features.shape == (9,)
