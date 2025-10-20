"""
Comprehensive unit tests for ReproductionComponent.

Tests all functionality including reproduction eligibility, cost management,
offspring creation, logging integration, and service interaction.
"""

import pytest
from unittest.mock import Mock, patch

from farm.core.agent.components.reproduction import ReproductionComponent
from farm.core.agent.config.component_configs import ReproductionConfig
from farm.core.agent.services import AgentServices


class TestReproductionComponentInitialization:
    """Test component initialization and configuration."""
    
    def test_init_with_default_config(self):
        """Test initialization with default configuration."""
        services = Mock(spec=AgentServices)
        config = ReproductionConfig()
        component = ReproductionComponent(services, config)
        
        assert component.config == config
        assert component.offspring_created == 0
        assert component.name == "ReproductionComponent"
        assert component.services == services
    
    def test_init_with_custom_config(self):
        """Test initialization with custom configuration."""
        services = Mock(spec=AgentServices)
        config = ReproductionConfig(
            offspring_initial_resources=20.0,
            offspring_cost=10.0
        )
        component = ReproductionComponent(services, config)
        
        assert component.config.offspring_initial_resources == 20.0
        assert component.config.offspring_cost == 10.0
    
    def test_attach_to_core(self):
        """Test attaching component to agent core."""
        services = Mock(spec=AgentServices)
        config = ReproductionConfig()
        component = ReproductionComponent(services, config)
        
        core = Mock()
        component.attach(core)
        
        assert component.core == core
    
    def test_lifecycle_hooks(self):
        """Test that lifecycle hooks are callable."""
        services = Mock(spec=AgentServices)
        config = ReproductionConfig()
        component = ReproductionComponent(services, config)
        
        # These should not raise exceptions
        component.on_step_start()
        component.on_step_end()
        component.on_terminate()


class TestCanReproduce:
    """Test reproduction eligibility checking."""
    
    @pytest.fixture
    def component(self):
        """Create a reproduction component for testing."""
        services = Mock(spec=AgentServices)
        config = ReproductionConfig(offspring_cost=5.0)
        return ReproductionComponent(services, config)
    
    def test_can_reproduce_no_core(self, component):
        """Test can_reproduce when no core is attached."""
        result = component.can_reproduce()
        
        assert result is False
    
    def test_can_reproduce_no_resource_component(self, component):
        """Test can_reproduce when no resource component is available."""
        core = Mock()
        core.get_component.return_value = None
        component.attach(core)
        
        result = component.can_reproduce()
        
        assert result is False
        core.get_component.assert_called_once_with("resource")
    
    def test_can_reproduce_insufficient_resources(self, component):
        """Test can_reproduce when resources are insufficient."""
        core = Mock()
        resource_component = Mock()
        resource_component.level = 3.0  # Less than offspring_cost (5.0)
        core.get_component.return_value = resource_component
        component.attach(core)
        
        result = component.can_reproduce()
        
        assert result is False
    
    def test_can_reproduce_exact_resources(self, component):
        """Test can_reproduce when resources exactly match cost."""
        core = Mock()
        resource_component = Mock()
        resource_component.level = 5.0  # Exactly offspring_cost
        core.get_component.return_value = resource_component
        component.attach(core)
        
        result = component.can_reproduce()
        
        assert result is True
    
    def test_can_reproduce_sufficient_resources(self, component):
        """Test can_reproduce when resources exceed cost."""
        core = Mock()
        resource_component = Mock()
        resource_component.level = 10.0  # More than offspring_cost (5.0)
        core.get_component.return_value = resource_component
        component.attach(core)
        
        result = component.can_reproduce()
        
        assert result is True
    
    def test_can_reproduce_zero_cost(self):
        """Test can_reproduce with zero offspring cost."""
        services = Mock(spec=AgentServices)
        config = ReproductionConfig(offspring_cost=0.0)
        component = ReproductionComponent(services, config)
        
        core = Mock()
        resource_component = Mock()
        resource_component.level = 0.0  # Zero resources
        core.get_component.return_value = resource_component
        component.attach(core)
        
        result = component.can_reproduce()
        
        assert result is True  # Can reproduce with zero cost even with zero resources
    
    def test_can_reproduce_negative_resources(self, component):
        """Test can_reproduce with negative resources."""
        core = Mock()
        resource_component = Mock()
        resource_component.level = -2.0  # Negative resources
        core.get_component.return_value = resource_component
        component.attach(core)
        
        result = component.can_reproduce()
        
        assert result is False  # Cannot reproduce with negative resources


class TestReproduce:
    """Test reproduction functionality."""
    
    @pytest.fixture
    def component_with_core(self):
        """Create a reproduction component with attached core."""
        services = Mock(spec=AgentServices)
        config = ReproductionConfig(offspring_cost=5.0)
        component = ReproductionComponent(services, config)
        
        core = Mock()
        core.agent_id = "parent_agent"
        core.position = (10.0, 20.0)
        core.state = Mock()
        core.state.generation = 5
        component.attach(core)
        
        return component
    
    def test_reproduce_success(self, component_with_core):
        """Test successful reproduction."""
        # Setup resource component with sufficient resources
        resource_component = Mock()
        resource_component.level = 10.0
        resource_component.remove.return_value = True
        component_with_core.core.get_component.return_value = resource_component
        
        result = component_with_core.reproduce()
        
        # Should return None (actual offspring creation handled by factory)
        assert result is None
        
        # Should increment offspring count
        assert component_with_core.offspring_created == 1
        
        # Should deduct cost from resources
        resource_component.remove.assert_called_once_with(5.0)
    
    def test_reproduce_insufficient_resources(self, component_with_core):
        """Test reproduction failure due to insufficient resources."""
        # Setup resource component with insufficient resources
        resource_component = Mock()
        resource_component.level = 3.0  # Less than offspring_cost
        component_with_core.core.get_component.return_value = resource_component
        
        result = component_with_core.reproduce()
        
        # Should return None (reproduction failed)
        assert result is None
        
        # Should not increment offspring count
        assert component_with_core.offspring_created == 0
        
        # Should not deduct cost from resources
        resource_component.remove.assert_not_called()
    
    def test_reproduce_no_core(self):
        """Test reproduction when no core is attached."""
        services = Mock(spec=AgentServices)
        config = ReproductionConfig()
        component = ReproductionComponent(services, config)
        
        result = component.reproduce()
        
        assert result is None
        assert component.offspring_created == 0
    
    def test_reproduce_no_resource_component(self, component_with_core):
        """Test reproduction when no resource component is available."""
        component_with_core.core.get_component.return_value = None
        
        result = component_with_core.reproduce()
        
        assert result is None
        assert component_with_core.offspring_created == 0
    
    def test_reproduce_resource_removal_failure(self, component_with_core):
        """Test reproduction when resource removal fails."""
        # Setup resource component that fails to remove resources
        resource_component = Mock()
        resource_component.level = 10.0
        resource_component.remove.return_value = False  # Removal fails
        component_with_core.core.get_component.return_value = resource_component
        
        result = component_with_core.reproduce()
        
        # Should still return None (reproduction failed)
        assert result is None
        
        # Should still increment offspring count (implementation doesn't check removal success)
        assert component_with_core.offspring_created == 1
        
        # Should attempt to deduct cost
        resource_component.remove.assert_called_once_with(5.0)
    
    def test_reproduce_multiple_offspring(self, component_with_core):
        """Test multiple successful reproductions."""
        # Setup resource component with sufficient resources
        resource_component = Mock()
        resource_component.level = 20.0  # Enough for multiple reproductions
        resource_component.remove.return_value = True
        component_with_core.core.get_component.return_value = resource_component
        
        # First reproduction
        result1 = component_with_core.reproduce()
        assert result1 is None
        assert component_with_core.offspring_created == 1
        
        # Second reproduction
        result2 = component_with_core.reproduce()
        assert result2 is None
        assert component_with_core.offspring_created == 2
        
        # Third reproduction
        result3 = component_with_core.reproduce()
        assert result3 is None
        assert component_with_core.offspring_created == 3
        
        # Should have called remove 3 times
        assert resource_component.remove.call_count == 3


class TestReproductionLogging:
    """Test reproduction logging functionality."""
    
    @pytest.fixture
    def component_with_logging(self):
        """Create a reproduction component with logging service."""
        services = Mock(spec=AgentServices)
        services.logging_service = Mock()
        config = ReproductionConfig(offspring_cost=5.0, offspring_initial_resources=10.0)
        component = ReproductionComponent(services, config)
        
        core = Mock()
        core.agent_id = "parent_agent"
        core.position = (15.0, 25.0)
        core.state = Mock()
        core.state.generation = 3
        component.attach(core)
        
        return component
    
    def test_reproduce_logging_success(self, component_with_logging):
        """Test logging on successful reproduction."""
        # Setup resource component that actually modifies level when remove is called
        resource_component = Mock()
        resource_component.level = 10.0
        
        def mock_remove(amount):
            resource_component.level -= amount
            return True
        
        resource_component.remove.side_effect = mock_remove
        component_with_logging.core.get_component.return_value = resource_component
        
        # Mock current time
        component_with_logging.services.get_current_time.return_value = 42
        
        component_with_logging.reproduce()
        
        # Verify logging call
        component_with_logging.services.logging_service.log_reproduction_event.assert_called_once()
        call_args = component_with_logging.services.logging_service.log_reproduction_event.call_args[1]
        
        assert call_args['step_number'] == 42
        assert call_args['parent_id'] == "parent_agent"
        assert call_args['offspring_id'] == ""  # Will be assigned by factory
        assert call_args['success'] is True
        assert call_args['parent_resources_before'] == 10.0  # 5.0 + 5.0 (current level + cost)
        assert call_args['parent_resources_after'] == 5.0
        assert call_args['offspring_initial_resources'] == 10.0
        assert call_args['failure_reason'] == ""
        assert call_args['parent_generation'] == 3
        assert call_args['offspring_generation'] == 0
        assert call_args['parent_position'] == (15.0, 25.0)
    
    def test_reproduce_logging_error(self, component_with_logging):
        """Test logging error handling."""
        # Setup resource component
        resource_component = Mock()
        resource_component.level = 10.0
        resource_component.remove.return_value = True
        component_with_logging.core.get_component.return_value = resource_component
        
        # Make logging service raise exception
        component_with_logging.services.logging_service.log_reproduction_event.side_effect = Exception("Logging error")
        
        # Should not raise exception
        result = component_with_logging.reproduce()
        
        assert result is None
        assert component_with_logging.offspring_created == 1  # Still increments
    
    def test_reproduce_no_logging_service(self):
        """Test reproduction without logging service."""
        services = Mock(spec=AgentServices)
        services.logging_service = None
        config = ReproductionConfig()
        component = ReproductionComponent(services, config)
        
        core = Mock()
        core.agent_id = "parent_agent"
        core.position = (10.0, 20.0)
        component.attach(core)
        
        resource_component = Mock()
        resource_component.level = 10.0
        resource_component.remove.return_value = True
        core.get_component.return_value = resource_component
        
        # Should not raise exception
        result = component.reproduce()
        
        assert result is None
        assert component.offspring_created == 1
    
    def test_reproduce_logging_without_state(self, component_with_logging):
        """Test logging when core has no state attribute."""
        # Remove state attribute
        del component_with_logging.core.state
        
        resource_component = Mock()
        resource_component.level = 10.0
        resource_component.remove.return_value = True
        component_with_logging.core.get_component.return_value = resource_component
        
        # Should not raise exception
        result = component_with_logging.reproduce()
        
        assert result is None
        assert component_with_logging.offspring_created == 1
        
        # Verify logging call with default generation
        call_args = component_with_logging.services.logging_service.log_reproduction_event.call_args[1]
        assert call_args['parent_generation'] == 0  # Default when no state


class TestTotalOffspring:
    """Test total offspring tracking."""
    
    @pytest.fixture
    def component(self):
        """Create a reproduction component for testing."""
        services = Mock(spec=AgentServices)
        config = ReproductionConfig()
        return ReproductionComponent(services, config)
    
    def test_total_offspring_initial(self, component):
        """Test initial offspring count."""
        assert component.total_offspring == 0
    
    def test_total_offspring_after_reproduction(self, component):
        """Test offspring count after reproduction."""
        # Manually increment (simulating successful reproduction)
        component.offspring_created = 3
        assert component.total_offspring == 3
    
    def test_total_offspring_property_consistency(self, component):
        """Test that total_offspring property matches offspring_created."""
        component.offspring_created = 7
        assert component.total_offspring == component.offspring_created


class TestEdgeCases:
    """Test edge cases and error scenarios."""
    
    def test_reproduce_exact_cost(self):
        """Test reproduction with exact cost amount."""
        services = Mock(spec=AgentServices)
        config = ReproductionConfig(offspring_cost=3.0)
        component = ReproductionComponent(services, config)
        
        core = Mock()
        core.agent_id = "parent_agent"
        core.position = (0.0, 0.0)
        component.attach(core)
        
        resource_component = Mock()
        resource_component.level = 3.0  # Exactly the cost
        resource_component.remove.return_value = True
        core.get_component.return_value = resource_component
        
        result = component.reproduce()
        
        assert result is None
        assert component.offspring_created == 1
        resource_component.remove.assert_called_once_with(3.0)
    
    def test_reproduce_just_below_cost(self):
        """Test reproduction when resources are just below cost."""
        services = Mock(spec=AgentServices)
        config = ReproductionConfig(offspring_cost=5.0)
        component = ReproductionComponent(services, config)
        
        core = Mock()
        core.agent_id = "parent_agent"
        component.attach(core)
        
        resource_component = Mock()
        resource_component.level = 4.999  # Just below cost
        core.get_component.return_value = resource_component
        
        result = component.reproduce()
        
        assert result is None
        assert component.offspring_created == 0
        resource_component.remove.assert_not_called()
    
    def test_reproduce_very_large_cost(self):
        """Test reproduction with very large cost."""
        services = Mock(spec=AgentServices)
        config = ReproductionConfig(offspring_cost=1e6)
        component = ReproductionComponent(services, config)
        
        core = Mock()
        core.agent_id = "parent_agent"
        component.attach(core)
        
        resource_component = Mock()
        resource_component.level = 1e6 + 1  # Just above cost
        resource_component.remove.return_value = True
        core.get_component.return_value = resource_component
        
        result = component.reproduce()
        
        assert result is None
        assert component.offspring_created == 1
        resource_component.remove.assert_called_once_with(1e6)
    
    def test_reproduce_very_small_cost(self):
        """Test reproduction with very small cost."""
        services = Mock(spec=AgentServices)
        config = ReproductionConfig(offspring_cost=0.001)
        component = ReproductionComponent(services, config)
        
        core = Mock()
        core.agent_id = "parent_agent"
        component.attach(core)
        
        resource_component = Mock()
        resource_component.level = 0.001  # Exactly the cost
        resource_component.remove.return_value = True
        core.get_component.return_value = resource_component
        
        result = component.reproduce()
        
        assert result is None
        assert component.offspring_created == 1
        resource_component.remove.assert_called_once_with(0.001)
    
    def test_reproduce_negative_cost(self):
        """Test reproduction with negative cost (should not happen but test edge case)."""
        services = Mock(spec=AgentServices)
        config = ReproductionConfig(offspring_cost=-5.0)
        component = ReproductionComponent(services, config)
        
        core = Mock()
        core.agent_id = "parent_agent"
        component.attach(core)
        
        resource_component = Mock()
        resource_component.level = 0.0  # Zero resources
        resource_component.remove.return_value = True
        core.get_component.return_value = resource_component
        
        result = component.reproduce()
        
        assert result is None
        assert component.offspring_created == 1
        resource_component.remove.assert_called_once_with(-5.0)  # Would add resources
    
    def test_reproduce_negative_resources(self):
        """Test reproduction with negative resources."""
        services = Mock(spec=AgentServices)
        config = ReproductionConfig(offspring_cost=5.0)
        component = ReproductionComponent(services, config)
        
        core = Mock()
        core.agent_id = "parent_agent"
        component.attach(core)
        
        resource_component = Mock()
        resource_component.level = -2.0  # Negative resources
        core.get_component.return_value = resource_component
        
        result = component.reproduce()
        
        assert result is None
        assert component.offspring_created == 0
        resource_component.remove.assert_not_called()


class TestIntegrationScenarios:
    """Test complex integration scenarios."""
    
    def test_complete_reproduction_workflow(self):
        """Test a complete reproduction workflow."""
        # Setup services
        services = Mock(spec=AgentServices)
        services.logging_service = Mock()
        services.get_current_time.return_value = 100
        
        config = ReproductionConfig(
            offspring_cost=8.0,
            offspring_initial_resources=15.0
        )
        component = ReproductionComponent(services, config)
        
        # Attach core
        core = Mock()
        core.agent_id = "parent_123"
        core.position = (50.0, 75.0)
        core.state = Mock()
        core.state.generation = 10
        component.attach(core)
        
        # Setup resource component that actually modifies level when remove is called
        resource_component = Mock()
        resource_component.level = 25.0  # Sufficient resources
        
        def mock_remove(amount):
            resource_component.level -= amount
            return True
        
        resource_component.remove.side_effect = mock_remove
        core.get_component.return_value = resource_component
        
        # Test initial state
        assert component.offspring_created == 0
        assert component.total_offspring == 0
        assert component.can_reproduce() is True
        
        # First reproduction
        result1 = component.reproduce()
        assert result1 is None
        assert component.offspring_created == 1
        assert component.total_offspring == 1
        resource_component.remove.assert_called_with(8.0)
        
        # Verify logging
        services.logging_service.log_reproduction_event.assert_called_once()
        call_args = services.logging_service.log_reproduction_event.call_args[1]
        assert call_args['step_number'] == 100
        assert call_args['parent_id'] == "parent_123"
        assert call_args['success'] is True
        assert call_args['parent_resources_before'] == 25.0  # 17.0 + 8.0 (current level + cost)
        assert call_args['parent_resources_after'] == 17.0
        assert call_args['offspring_initial_resources'] == 15.0
        assert call_args['parent_generation'] == 10
        
        # Second reproduction
        result2 = component.reproduce()
        assert result2 is None
        assert component.offspring_created == 2
        assert component.total_offspring == 2
        
        # Third reproduction (should still work)
        result3 = component.reproduce()
        assert result3 is None
        assert component.offspring_created == 3
        assert component.total_offspring == 3
        
        # Verify total resource removal
        assert resource_component.remove.call_count == 3
    
    def test_reproduction_with_insufficient_resources_after_success(self):
        """Test reproduction workflow when resources become insufficient."""
        services = Mock(spec=AgentServices)
        config = ReproductionConfig(offspring_cost=10.0)
        component = ReproductionComponent(services, config)
        
        core = Mock()
        core.agent_id = "parent_agent"
        component.attach(core)
        
        resource_component = Mock()
        resource_component.level = 15.0  # Enough for one reproduction
        resource_component.remove.return_value = True
        core.get_component.return_value = resource_component
        
        # First reproduction succeeds
        result1 = component.reproduce()
        assert result1 is None
        assert component.offspring_created == 1
        
        # Simulate resource consumption (level now 5.0, insufficient for second reproduction)
        resource_component.level = 5.0
        
        # Second reproduction fails
        result2 = component.reproduce()
        assert result2 is None
        assert component.offspring_created == 1  # Still 1, not incremented
        
        # Verify only one removal call
        assert resource_component.remove.call_count == 1
    
    def test_error_recovery_scenarios(self):
        """Test error recovery in various scenarios."""
        services = Mock(spec=AgentServices)
        services.logging_service = Mock()
        services.logging_service.log_reproduction_event.side_effect = Exception("Logging error")
        
        config = ReproductionConfig()
        component = ReproductionComponent(services, config)
        
        core = Mock()
        core.agent_id = "parent_agent"
        core.position = (0.0, 0.0)
        component.attach(core)
        
        resource_component = Mock()
        resource_component.level = 10.0
        resource_component.remove.return_value = True
        core.get_component.return_value = resource_component
        
        # Test that component continues to work despite logging errors
        result = component.reproduce()
        assert result is None
        assert component.offspring_created == 1
        
        # Test lifecycle hooks still work
        component.on_step_start()
        component.on_step_end()
        component.on_terminate()
        
        # Test can_reproduce still works
        assert component.can_reproduce() is True
