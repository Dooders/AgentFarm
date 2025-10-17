"""Tests for feature engineering with component-based agent architecture."""

import numpy as np
import pytest

from farm.config import SimulationConfig
from farm.core.agent.config.agent_config import AgentConfig, CombatConfig, ResourceConfig
from farm.core.agent.factory import AgentFactory
from farm.core.decision.feature_engineering import FeatureEngineer
from farm.core.environment import Environment
from farm.core.services.implementations import SpatialIndexAdapter


class TestFeatureEngineering:
    """Test feature engineering with component-based agents."""

    def setup_method(self):
        """Set up test environment and factory."""
        self.env = Environment(
            width=100,
            height=100,
            resource_distribution={"density": 0.1, "amount_range": (1, 5)},
            config=SimulationConfig(),
        )

        spatial_service = SpatialIndexAdapter(self.env.spatial_index)
        self.factory = AgentFactory(spatial_service=spatial_service, default_config=AgentConfig())

        self.feature_engineer = FeatureEngineer()

    def test_default_agent_features(self):
        """Test feature extraction from default agent."""
        agent = self.factory.create_default_agent(agent_id="test_agent", position=(50, 50), initial_resources=100)
        self.env.add_agent(agent)

        features = self.feature_engineer.extract_features(agent, self.env)

        # Should extract 9 features
        assert len(features) == 9
        assert isinstance(features, np.ndarray)

        # Health should be normalized (1.0 for full health)
        assert features[0] == 1.0

        # Resource level should be normalized
        assert features[1] > 0

        # Position should be normalized (0.5 for center)
        assert features[2] == 0.5  # x position
        assert features[3] == 0.5  # y position

    def test_agent_with_custom_config(self):
        """Test feature extraction with custom agent configuration."""
        custom_config = AgentConfig(
            resource=ResourceConfig(starvation_threshold=50, base_consumption_rate=2),
            combat=CombatConfig(starting_health=150.0),
        )

        agent = self.factory.create_default_agent(
            agent_id="custom_agent", position=(25, 75), initial_resources=25, config=custom_config
        )
        self.env.add_agent(agent)

        features = self.feature_engineer.extract_features(agent, self.env)

        # Should extract 9 features
        assert len(features) == 9

        # Health should be normalized (1.0 for full health)
        assert features[0] == 1.0

        # Resource level should be normalized
        assert features[1] > 0

    def test_starvation_features(self):
        """Test feature extraction with starvation scenario."""
        agent = self.factory.create_default_agent(agent_id="starving_agent", position=(10, 10), initial_resources=0)
        self.env.add_agent(agent)

        # Simulate starvation
        resource_comp = agent.get_component("resource")
        if resource_comp:
            resource_comp._starvation_counter = 10

        features = self.feature_engineer.extract_features(agent, self.env)

        # Should extract 9 features
        assert len(features) == 9

        # Resource level should be 0
        assert features[1] == 0.0

        # Starvation ratio should be > 0 (10/100 = 0.1)
        assert features[5] > 0

    def test_defending_agent_features(self):
        """Test feature extraction with defending agent."""
        agent = self.factory.create_default_agent(agent_id="defending_agent", position=(90, 90), initial_resources=75)
        self.env.add_agent(agent)

        # Set agent to defending
        combat_comp = agent.get_component("combat")
        if combat_comp:
            combat_comp.start_defense(3)

        features = self.feature_engineer.extract_features(agent, self.env)

        # Should extract 9 features
        assert len(features) == 9

        # Defending flag should be 1.0
        assert features[7] == 1.0

    def test_component_access_properties(self):
        """Test that agent properties correctly delegate to components."""
        agent = self.factory.create_default_agent(agent_id="test_agent", position=(50, 50), initial_resources=100)

        # Test that components are accessible directly
        assert agent.get_component("combat").health == 100.0  # From combat component
        assert agent.get_component("combat").max_health == 100.0  # From combat component
        assert agent.get_component("resource").level == 100  # From resource component
        assert agent.get_component("resource").starvation_steps == 0  # From resource component
        assert agent.get_component("resource")._config.starvation_threshold == 100  # From resource component config
        assert not agent.get_component("combat").is_defending  # From combat component

    def test_feature_engineering_robustness(self):
        """Test that feature engineering handles missing components gracefully."""
        # Create agent without some components
        agent = self.factory.create_default_agent(agent_id="minimal_agent", position=(50, 50), initial_resources=50)

        # Remove a component to test robustness
        agent.remove_component("combat")

        # Should still work with default values
        features = self.feature_engineer.extract_features(agent, self.env)

        # Should extract 9 features
        assert len(features) == 9

        # Health should be 0 (no combat component)
        assert features[0] == 0.0
