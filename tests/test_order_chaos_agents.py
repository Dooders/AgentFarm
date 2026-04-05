"""Tests for Order and Chaos agent types.

Validates that Order and Chaos agents are correctly recognized, configured,
and created with the expected behavioral parameters.
"""

import pytest
from unittest.mock import Mock

from farm.core.agent import AgentFactory, AgentServices
from farm.config.config import PopulationConfig, SimulationConfig


class TestOrderChaosAgentTypes:
    """Test Order and Chaos agent type recognition and configuration."""

    @pytest.fixture
    def mock_services(self):
        """Create mock services for testing."""
        return AgentServices(
            spatial_service=Mock(),
            time_service=Mock(current_time=Mock(return_value=0)),
            metrics_service=Mock(),
            logging_service=Mock(),
            validation_service=Mock(is_valid_position=Mock(return_value=True)),
            lifecycle_service=Mock(),
        )

    def test_order_agent_creation(self, mock_services):
        """Test that an order agent can be created with correct type."""
        factory = AgentFactory(mock_services)
        agent = factory.create_default_agent(
            agent_id="order_001",
            position=(10.0, 10.0),
            initial_resources=100.0,
            agent_type="order",
        )
        assert agent.agent_type == "order"
        assert agent.alive is True

    def test_chaos_agent_creation(self, mock_services):
        """Test that a chaos agent can be created with correct type."""
        factory = AgentFactory(mock_services)
        agent = factory.create_default_agent(
            agent_id="chaos_001",
            position=(20.0, 20.0),
            initial_resources=100.0,
            agent_type="chaos",
        )
        assert agent.agent_type == "chaos"
        assert agent.alive is True

    def test_order_agent_action_weights(self, mock_services):
        """Order agents should have very low attack weight per config."""
        config = SimulationConfig()
        order_params = config.agent_parameters["OrderAgent"]
        assert order_params["attack_weight"] < order_params["share_weight"]

    def test_chaos_agent_action_weights(self, mock_services):
        """Chaos agents should have very high attack weight per config."""
        config = SimulationConfig()
        chaos_params = config.agent_parameters["ChaosAgent"]
        assert chaos_params["attack_weight"] > chaos_params["share_weight"]

    def test_all_agent_types_have_components(self, mock_services):
        """All new agent types should have the standard set of components."""
        factory = AgentFactory(mock_services)
        for agent_type in ("order", "chaos"):
            agent = factory.create_default_agent(
                agent_id=f"{agent_type}_comp_test",
                position=(0.0, 0.0),
                agent_type=agent_type,
            )
            assert agent.get_component("movement") is not None
            assert agent.get_component("resource") is not None
            assert agent.get_component("combat") is not None


class TestPopulationConfig:
    """Test PopulationConfig includes order and chaos agents."""

    def test_default_order_chaos_agents_zero(self):
        """Order and Chaos agents default to 0 for backward compatibility."""
        config = PopulationConfig()
        assert config.order_agents == 0
        assert config.chaos_agents == 0

    def test_order_chaos_agents_can_be_set(self):
        """Order and Chaos agent counts can be configured."""
        config = PopulationConfig(order_agents=5, chaos_agents=3)
        assert config.order_agents == 5
        assert config.chaos_agents == 3

    def test_agent_type_ratios_include_new_types(self):
        """agent_type_ratios should include OrderAgent and ChaosAgent entries."""
        config = PopulationConfig()
        assert "OrderAgent" in config.agent_type_ratios
        assert "ChaosAgent" in config.agent_type_ratios

    def test_existing_agent_ratios_unchanged(self):
        """Existing agent type ratios should not be changed by adding new types."""
        config = PopulationConfig()
        assert config.agent_type_ratios["SystemAgent"] == pytest.approx(0.33)
        assert config.agent_type_ratios["IndependentAgent"] == pytest.approx(0.33)
        assert config.agent_type_ratios["ControlAgent"] == pytest.approx(0.34)


class TestSimulationConfigAgentParameters:
    """Test SimulationConfig includes order and chaos agent parameters."""

    def test_order_agent_parameters_exist(self):
        """OrderAgent parameters must be defined in SimulationConfig."""
        config = SimulationConfig()
        assert "OrderAgent" in config.agent_parameters
        params = config.agent_parameters["OrderAgent"]
        assert "gather_efficiency_multiplier" in params
        assert "share_weight" in params
        assert "attack_weight" in params

    def test_chaos_agent_parameters_exist(self):
        """ChaosAgent parameters must be defined in SimulationConfig."""
        config = SimulationConfig()
        assert "ChaosAgent" in config.agent_parameters
        params = config.agent_parameters["ChaosAgent"]
        assert "gather_efficiency_multiplier" in params
        assert "share_weight" in params
        assert "attack_weight" in params

    def test_order_agent_low_attack(self):
        """Order agents should have a lower attack weight than chaos agents."""
        config = SimulationConfig()
        order_attack = config.agent_parameters["OrderAgent"]["attack_weight"]
        chaos_attack = config.agent_parameters["ChaosAgent"]["attack_weight"]
        assert order_attack < chaos_attack

    def test_chaos_agent_low_share(self):
        """Chaos agents should have a lower share weight than order agents."""
        config = SimulationConfig()
        order_share = config.agent_parameters["OrderAgent"]["share_weight"]
        chaos_share = config.agent_parameters["ChaosAgent"]["share_weight"]
        assert chaos_share < order_share

    def test_order_agent_high_resource_threshold(self):
        """Order agents should maintain higher resource reserves than chaos agents."""
        config = SimulationConfig()
        order_threshold = config.agent_parameters["OrderAgent"]["min_resource_threshold"]
        chaos_threshold = config.agent_parameters["ChaosAgent"]["min_resource_threshold"]
        assert order_threshold > chaos_threshold

    def test_visualization_colors_include_new_types(self):
        """Visualization colors should include OrderAgent and ChaosAgent."""
        config = SimulationConfig()
        assert "OrderAgent" in config.visualization.agent_colors
        assert "ChaosAgent" in config.visualization.agent_colors

    def test_metric_colors_include_new_types(self):
        """Metric colors should include order_agents and chaos_agents."""
        config = SimulationConfig()
        assert "order_agents" in config.visualization.metric_colors
        assert "chaos_agents" in config.visualization.metric_colors
