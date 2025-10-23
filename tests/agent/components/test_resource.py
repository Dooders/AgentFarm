"""
Comprehensive unit tests for ResourceComponent.

Tests all functionality including resource management, consumption,
starvation mechanics, and service integration.
"""

import pytest
from unittest.mock import Mock

from farm.core.agent.components.resource import ResourceComponent
from farm.core.agent.config.component_configs import ResourceConfig
from farm.core.agent.services import AgentServices


class TestResourceComponentInitialization:
    """Test component initialization and configuration."""
    
    def test_init_with_default_config(self):
        """Test initialization with default configuration."""
        services = Mock(spec=AgentServices)
        config = ResourceConfig()
        component = ResourceComponent(services, config)
        
        assert component.config == config
        assert component.level == 0.0
        assert component.starvation_counter == 0
        assert component.name == "ResourceComponent"
        assert component.services == services
    
    def test_init_with_custom_config(self):
        """Test initialization with custom configuration."""
        services = Mock(spec=AgentServices)
        config = ResourceConfig(
            base_consumption_rate=2.0,
            starvation_threshold=50,
            offspring_initial_resources=15.0,
            offspring_cost=8.0
        )
        component = ResourceComponent(services, config)
        
        assert component.config.base_consumption_rate == 2.0
        assert component.config.starvation_threshold == 50
        assert component.config.offspring_initial_resources == 15.0
        assert component.config.offspring_cost == 8.0
    
    def test_attach_to_core(self):
        """Test attaching component to agent core."""
        services = Mock(spec=AgentServices)
        config = ResourceConfig()
        component = ResourceComponent(services, config)
        
        core = Mock()
        component.attach(core)
        
        assert component.core == core
    
    def test_lifecycle_hooks(self):
        """Test that lifecycle hooks are callable."""
        services = Mock(spec=AgentServices)
        config = ResourceConfig()
        component = ResourceComponent(services, config)
        
        # These should not raise exceptions
        component.on_step_start()
        component.on_step_end()
        component.on_terminate()


class TestOnStepStart:
    """Test resource consumption on step start."""
    
    @pytest.fixture
    def component(self):
        """Create a resource component for testing."""
        services = Mock(spec=AgentServices)
        config = ResourceConfig(base_consumption_rate=1.5)
        return ResourceComponent(services, config)
    
    def test_consumption_with_positive_resources(self, component):
        """Test consumption when resources are positive."""
        component.level = 10.0
        component.on_step_start()
        
        assert component.level == 8.5  # 10.0 - 1.5
    
    def test_consumption_with_zero_resources(self, component):
        """Test consumption when resources are zero."""
        component.level = 0.0
        component.on_step_start()
        
        assert component.level == 0.0  # Clamped to 0.0, cannot go negative
    
    def test_consumption_with_negative_resources(self, component):
        """Test consumption when resources are already negative."""
        component.level = -5.0
        component.on_step_start()
        
        assert component.level == 0.0  # Clamped to 0.0, cannot go negative
    
    def test_consumption_exact_amount(self, component):
        """Test consumption of exact resource amount."""
        component.level = 1.5
        component.on_step_start()
        
        assert component.level == 0.0
    
    def test_consumption_zero_rate(self):
        """Test consumption with zero consumption rate."""
        services = Mock(spec=AgentServices)
        config = ResourceConfig(base_consumption_rate=0.0)
        component = ResourceComponent(services, config)
        
        component.level = 10.0
        component.on_step_start()
        
        assert component.level == 10.0  # Unchanged
    
    def test_consumption_negative_rate(self):
        """Test consumption with negative consumption rate (gains resources)."""
        services = Mock(spec=AgentServices)
        config = ResourceConfig(base_consumption_rate=-2.0)
        component = ResourceComponent(services, config)
        
        component.level = 5.0
        component.on_step_start()
        
        assert component.level == 7.0  # 5.0 - (-2.0) = 5.0 + 2.0


class TestStarvationCheck:
    """Test starvation checking and counter management."""
    
    @pytest.fixture
    def component(self):
        """Create a resource component for testing."""
        services = Mock(spec=AgentServices)
        config = ResourceConfig(starvation_threshold=3)
        return ResourceComponent(services, config)
    
    def test_starvation_check_positive_resources(self, component):
        """Test starvation check with positive resources."""
        component.level = 5.0
        component.starvation_counter = 2
        
        result = component._check_starvation()
        
        assert result is False
        assert component.starvation_counter == 0  # Reset
    
    def test_starvation_check_zero_resources(self, component):
        """Test starvation check with zero resources."""
        component.level = 0.0
        component.starvation_counter = 1
        
        result = component._check_starvation()
        
        assert result is False
        assert component.starvation_counter == 2  # Incremented
    
    def test_starvation_check_negative_resources(self, component):
        """Test starvation check with negative resources."""
        component.level = -2.0
        component.starvation_counter = 0
        
        result = component._check_starvation()
        
        assert result is False
        assert component.starvation_counter == 1  # Incremented
    
    def test_starvation_death(self, component):
        """Test starvation death when threshold reached."""
        component.level = -1.0
        component.starvation_counter = 2  # One less than threshold
        
        result = component._check_starvation()
        
        assert result is True
        assert component.starvation_counter == 3  # Incremented to threshold
    
    def test_starvation_death_with_core(self, component):
        """Test starvation death triggers core termination."""
        core = Mock()
        component.attach(core)
        
        component.level = -1.0
        component.starvation_counter = 2
        
        result = component._check_starvation()
        
        assert result is True
        core.terminate.assert_called_once()
    
    def test_starvation_death_no_core(self, component):
        """Test starvation death without core attached."""
        component.level = -1.0
        component.starvation_counter = 2
        
        result = component._check_starvation()
        
        assert result is True
        # No core to terminate
    
    def test_starvation_counter_reset(self, component):
        """Test starvation counter reset when resources become positive."""
        component.level = -1.0
        component.starvation_counter = 1  # Start with 1, so increment to 2 (below threshold of 3)
        
        # First check (still starving, but not at threshold yet)
        result1 = component._check_starvation()
        assert result1 is False
        assert component.starvation_counter == 2
        
        # Add resources
        component.level = 5.0
        
        # Second check (no longer starving)
        result2 = component._check_starvation()
        assert result2 is False
        assert component.starvation_counter == 0  # Reset


class TestAddResources:
    """Test adding resources to the component."""
    
    @pytest.fixture
    def component(self):
        """Create a resource component for testing."""
        services = Mock(spec=AgentServices)
        config = ResourceConfig()
        return ResourceComponent(services, config)
    
    def test_add_positive_amount(self, component):
        """Test adding positive amount of resources."""
        component.level = 5.0
        component.add(3.0)
        
        assert component.level == 8.0
    
    def test_add_to_zero_resources(self, component):
        """Test adding resources when level is zero."""
        component.level = 0.0
        component.add(10.0)
        
        assert component.level == 10.0
    
    def test_add_to_negative_resources(self, component):
        """Test adding resources when level is negative."""
        component.level = -5.0
        component.add(8.0)
        
        assert component.level == 3.0
    
    def test_add_zero_amount(self, component):
        """Test adding zero amount of resources."""
        component.level = 5.0
        component.add(0.0)
        
        assert component.level == 5.0  # Unchanged
    
    def test_add_negative_amount(self, component):
        """Test adding negative amount (removing resources)."""
        component.level = 10.0
        component.add(-3.0)
        
        assert component.level == 7.0
    
    def test_add_large_amount(self, component):
        """Test adding large amount of resources."""
        component.level = 1.0
        component.add(1000.0)
        
        assert component.level == 1001.0


class TestRemoveResources:
    """Test removing resources from the component."""
    
    @pytest.fixture
    def component(self):
        """Create a resource component for testing."""
        services = Mock(spec=AgentServices)
        config = ResourceConfig()
        return ResourceComponent(services, config)
    
    def test_remove_sufficient_resources(self, component):
        """Test removing resources when sufficient available."""
        component.level = 10.0
        result = component.remove(3.0)
        
        assert result is True
        assert component.level == 7.0
    
    def test_remove_exact_amount(self, component):
        """Test removing exact amount of resources."""
        component.level = 5.0
        result = component.remove(5.0)
        
        assert result is True
        assert component.level == 0.0
    
    def test_remove_insufficient_resources(self, component):
        """Test removing resources when insufficient available."""
        component.level = 3.0
        result = component.remove(5.0)
        
        assert result is False
        assert component.level == 3.0  # Unchanged
    
    def test_remove_from_zero_resources(self, component):
        """Test removing resources when level is zero."""
        component.level = 0.0
        result = component.remove(1.0)
        
        assert result is False
        assert component.level == 0.0  # Unchanged
    
    def test_remove_from_negative_resources(self, component):
        """Test removing resources when level is negative."""
        component.level = -2.0
        result = component.remove(1.0)
        
        assert result is False
        assert component.level == -2.0  # Unchanged
    
    def test_remove_zero_amount(self, component):
        """Test removing zero amount of resources."""
        component.level = 5.0
        result = component.remove(0.0)
        
        assert result is True
        assert component.level == 5.0  # Unchanged
    
    def test_remove_negative_amount(self, component):
        """Test removing negative amount (adding resources)."""
        component.level = 5.0
        result = component.remove(-2.0)
        
        assert result is True
        assert component.level == 7.0  # 5.0 - (-2.0) = 7.0


class TestProperties:
    """Test component properties."""
    
    @pytest.fixture
    def component(self):
        """Create a resource component for testing."""
        services = Mock(spec=AgentServices)
        config = ResourceConfig()
        return ResourceComponent(services, config)
    
    def test_has_resources_positive(self, component):
        """Test has_resources with positive resources."""
        component.level = 5.0
        assert component.has_resources is True
    
    def test_has_resources_zero(self, component):
        """Test has_resources with zero resources."""
        component.level = 0.0
        assert component.has_resources is False
    
    def test_has_resources_negative(self, component):
        """Test has_resources with negative resources."""
        component.level = -2.0
        assert component.has_resources is False
    
    def test_is_starving_positive_counter(self, component):
        """Test is_starving with positive starvation counter."""
        component.starvation_counter = 2
        assert component.is_starving is True
    
    def test_is_starving_zero_counter(self, component):
        """Test is_starving with zero starvation counter."""
        component.starvation_counter = 0
        assert component.is_starving is False
    
    def test_is_starving_negative_counter(self, component):
        """Test is_starving with negative starvation counter."""
        component.starvation_counter = -1
        assert component.is_starving is False
    
    def test_turns_until_starvation_positive_resources(self, component):
        """Test turns_until_starvation with positive resources."""
        component.level = 10.0
        component.starvation_counter = 0
        assert component.turns_until_starvation == 100  # starvation_threshold
    
    def test_turns_until_starvation_zero_resources(self, component):
        """Test turns_until_starvation with zero resources."""
        component.level = 0.0
        component.starvation_counter = 3
        assert component.turns_until_starvation == 97  # 100 - 3
    
    def test_turns_until_starvation_negative_resources(self, component):
        """Test turns_until_starvation with negative resources."""
        component.level = -5.0
        component.starvation_counter = 5
        assert component.turns_until_starvation == 95  # 100 - 5
    
    def test_turns_until_starvation_at_threshold(self, component):
        """Test turns_until_starvation at starvation threshold."""
        component.level = -1.0
        component.starvation_counter = 100  # At threshold
        assert component.turns_until_starvation == 0
    
    def test_turns_until_starvation_over_threshold(self, component):
        """Test turns_until_starvation over starvation threshold."""
        component.level = -1.0
        component.starvation_counter = 105  # Over threshold
        assert component.turns_until_starvation == 0  # Clamped to 0


class TestEdgeCases:
    """Test edge cases and error scenarios."""
    
    def test_very_small_consumption_rate(self):
        """Test with very small consumption rate."""
        services = Mock(spec=AgentServices)
        config = ResourceConfig(base_consumption_rate=0.001)
        component = ResourceComponent(services, config)
        
        component.level = 1.0
        component.on_step_start()
        
        assert component.level == 0.999
    
    def test_very_large_consumption_rate(self):
        """Test with very large consumption rate."""
        services = Mock(spec=AgentServices)
        config = ResourceConfig(base_consumption_rate=1000.0)
        component = ResourceComponent(services, config)
        
        component.level = 500.0
        component.on_step_start()
        
        assert component.level == 0.0  # Clamped to 0.0, cannot go negative
    
    def test_starvation_threshold_zero(self):
        """Test with zero starvation threshold."""
        services = Mock(spec=AgentServices)
        config = ResourceConfig(starvation_threshold=0)
        component = ResourceComponent(services, config)
        
        component.level = -1.0
        component.starvation_counter = 0
        
        result = component._check_starvation()
        
        assert result is True  # Dies immediately
    
    def test_starvation_threshold_negative(self):
        """Test with negative starvation threshold."""
        services = Mock(spec=AgentServices)
        config = ResourceConfig(starvation_threshold=-1)
        component = ResourceComponent(services, config)
        
        component.level = -1.0
        component.starvation_counter = 0
        
        result = component._check_starvation()
        
        assert result is True  # Dies immediately (counter 0 >= -1)
    
    def test_very_large_resource_amounts(self):
        """Test with very large resource amounts."""
        services = Mock(spec=AgentServices)
        config = ResourceConfig()
        component = ResourceComponent(services, config)
        
        component.level = 1e6
        component.add(1e6)
        assert component.level == 2e6
        
        result = component.remove(1e6)
        assert result is True
        assert component.level == 1e6
    
    def test_floating_point_precision(self):
        """Test floating point precision issues."""
        services = Mock(spec=AgentServices)
        config = ResourceConfig(base_consumption_rate=0.1)
        component = ResourceComponent(services, config)
        
        component.level = 1.0
        
        # Multiple small consumptions
        for _ in range(10):
            component.on_step_start()
        
        # Should be close to 0.0 (1.0 - 10 * 0.1)
        assert abs(component.level) < 1e-10
    
    def test_starvation_counter_overflow(self):
        """Test starvation counter with very large values."""
        services = Mock(spec=AgentServices)
        config = ResourceConfig(starvation_threshold=100)
        component = ResourceComponent(services, config)
        
        component.level = -1.0
        component.starvation_counter = 1000  # Very large counter
        
        result = component._check_starvation()
        
        assert result is True  # Still dies (counter >= threshold)
        assert component.turns_until_starvation == 0  # Clamped to 0


class TestIntegrationScenarios:
    """Test complex integration scenarios."""
    
    def test_complete_resource_workflow(self):
        """Test a complete resource management workflow."""
        # Setup services
        services = Mock(spec=AgentServices)
        config = ResourceConfig(
            base_consumption_rate=2.0,
            starvation_threshold=3
        )
        component = ResourceComponent(services, config)
        
        # Test initial state
        assert component.level == 0.0
        assert component.starvation_counter == 0
        assert component.has_resources is False
        assert component.is_starving is False
        assert component.turns_until_starvation == 3
        
        # Add initial resources
        component.add(10.0)
        assert component.level == 10.0
        assert component.has_resources is True
        assert component.is_starving is False
        
        # Consume resources over multiple steps
        component.on_step_start()  # 10.0 - 2.0 = 8.0
        assert component.level == 8.0
        assert component.starvation_counter == 0
        
        component.on_step_start()  # 8.0 - 2.0 = 6.0
        assert component.level == 6.0
        
        # Remove some resources
        result = component.remove(4.0)
        assert result is True
        assert component.level == 2.0
        
        # Continue consuming until starvation
        component.on_step_start()  # 2.0 - 2.0 = 0.0
        assert component.level == 0.0
        assert component.starvation_counter == 1  # Incremented because level <= 0
        
        component.on_step_start()  # 0.0 - 2.0 = -2.0, but clamped to 0.0, counter = 2
        assert component.level == 0.0  # Clamped to 0.0
        assert component.starvation_counter == 2
        assert component.is_starving is True
        assert component.turns_until_starvation == 1
        
        component.on_step_start()  # 0.0 - 2.0 = -2.0, but clamped to 0.0, counter = 3 (death)
        assert component.level == 0.0  # Clamped to 0.0
        assert component.starvation_counter == 3
        assert component.turns_until_starvation == 0
        
        # Death on next step
        core = Mock()
        component.attach(core)
        component.on_step_start()  # 0.0 - 2.0 = -2.0, but clamped to 0.0, counter = 4 (continues incrementing)
        assert component.level == 0.0  # Clamped to 0.0
        assert component.starvation_counter == 4
        core.terminate.assert_called_once()
    
    def test_starvation_recovery_scenario(self):
        """Test recovery from starvation."""
        services = Mock(spec=AgentServices)
        config = ResourceConfig(
            base_consumption_rate=1.0,
            starvation_threshold=5
        )
        component = ResourceComponent(services, config)
        
        # Start starving
        component.level = -1.0
        component.on_step_start()  # -1.0 - 1.0 = -2.0, but clamped to 0.0, counter = 1
        assert component.level == 0.0  # Clamped to 0.0
        assert component.starvation_counter == 1
        assert component.is_starving is True
        
        component.on_step_start()  # 0.0 - 1.0 = -1.0, but clamped to 0.0, counter = 2
        assert component.level == 0.0  # Clamped to 0.0
        assert component.starvation_counter == 2
        
        # Add resources to recover
        component.add(10.0)  # 0.0 + 10.0 = 10.0
        assert component.level == 10.0
        
        # Next step should reset starvation counter
        component.on_step_start()  # 10.0 - 1.0 = 9.0
        assert component.level == 9.0
        assert component.starvation_counter == 0  # Reset
        assert component.is_starving is False
        assert component.has_resources is True
    
    def test_error_recovery_scenarios(self):
        """Test error recovery in various scenarios."""
        services = Mock(spec=AgentServices)
        config = ResourceConfig()
        component = ResourceComponent(services, config)
        
        # Test that component continues to work despite various edge cases
        # Zero consumption
        component.config = ResourceConfig(base_consumption_rate=0.0)
        component.level = 5.0
        component.on_step_start()
        assert component.level == 5.0
        
        # Negative consumption (gains resources)
        component.config = ResourceConfig(base_consumption_rate=-1.0)
        component.on_step_start()
        assert component.level == 6.0
        
        # Zero starvation threshold
        component.config = ResourceConfig(starvation_threshold=0)
        component.level = -1.0
        component.starvation_counter = 0
        result = component._check_starvation()
        assert result is True
        
        # Test lifecycle hooks still work
        component.on_step_start()
        component.on_step_end()
        component.on_terminate()
