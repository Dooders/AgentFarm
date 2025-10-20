"""
Comprehensive unit tests for MovementComponent.

Tests all functionality including position management, movement validation,
spatial queries, error handling, and service integration.
"""

import math
import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import Tuple

from farm.core.agent.components.movement import MovementComponent
from farm.core.agent.config.component_configs import MovementConfig
from farm.core.agent.services import AgentServices


class TestMovementComponentInitialization:
    """Test component initialization and configuration."""
    
    def test_init_with_default_config(self):
        """Test initialization with default configuration."""
        services = Mock(spec=AgentServices)
        config = MovementConfig()
        component = MovementComponent(services, config)
        
        assert component.config == config
        assert component.position == (0.0, 0.0)
        assert component.name == "MovementComponent"
        assert component.services == services
    
    def test_init_with_custom_config(self):
        """Test initialization with custom configuration."""
        services = Mock(spec=AgentServices)
        config = MovementConfig(max_movement=10.0, perception_radius=8)
        component = MovementComponent(services, config)
        
        assert component.config.max_movement == 10.0
        assert component.config.perception_radius == 8
    
    def test_attach_to_core(self):
        """Test attaching component to agent core."""
        services = Mock(spec=AgentServices)
        config = MovementConfig()
        component = MovementComponent(services, config)
        
        core = Mock()
        component.attach(core)
        
        assert component.core == core
    
    def test_lifecycle_hooks(self):
        """Test that lifecycle hooks are callable."""
        services = Mock(spec=AgentServices)
        config = MovementConfig()
        component = MovementComponent(services, config)
        
        # These should not raise exceptions
        component.on_step_start()
        component.on_step_end()
        component.on_terminate()


class TestPositionValidation:
    """Test position validation and error handling."""
    
    @pytest.fixture
    def component(self):
        """Create a movement component for testing."""
        services = Mock(spec=AgentServices)
        services.spatial_service = None  # Explicitly set to None
        services.validation_service = None
        config = MovementConfig()
        return MovementComponent(services, config)
    
    def test_set_position_valid(self, component):
        """Test setting a valid position."""
        new_position = (5.0, 10.0)
        result = component.set_position(new_position)
        
        assert result is True
        assert component.position == new_position
    
    def test_set_position_same_position(self, component):
        """Test setting the same position (no change)."""
        original_position = (0.0, 0.0)
        result = component.set_position(original_position)
        
        assert result is True
        assert component.position == original_position
    
    def test_set_position_invalid_type(self, component):
        """Test setting position with invalid type."""
        with pytest.raises(ValueError, match="Position must be a tuple/list of 2 numbers"):
            component.set_position("invalid")
    
    def test_set_position_wrong_length(self, component):
        """Test setting position with wrong number of coordinates."""
        with pytest.raises(ValueError, match="Position must be a tuple/list of 2 numbers"):
            component.set_position((1.0,))
        
        with pytest.raises(ValueError, match="Position must be a tuple/list of 2 numbers"):
            component.set_position((1.0, 2.0, 3.0))
    
    def test_set_position_non_numeric_coordinates(self, component):
        """Test setting position with non-numeric coordinates."""
        with pytest.raises(ValueError, match="Position coordinates must be numbers"):
            component.set_position(("x", "y"))
    
    def test_set_position_nan_coordinates(self, component):
        """Test setting position with NaN coordinates."""
        with pytest.raises(ValueError, match="Position coordinates cannot be NaN or infinity"):
            component.set_position((float('nan'), 1.0))
        
        with pytest.raises(ValueError, match="Position coordinates cannot be NaN or infinity"):
            component.set_position((1.0, float('nan')))
    
    def test_set_position_infinity_coordinates(self, component):
        """Test setting position with infinity coordinates."""
        with pytest.raises(ValueError, match="Position coordinates cannot be NaN or infinity"):
            component.set_position((float('inf'), 1.0))
        
        with pytest.raises(ValueError, match="Position coordinates cannot be NaN or infinity"):
            component.set_position((1.0, float('-inf')))
    
    def test_set_position_with_validation_service_success(self, component):
        """Test position validation with validation service."""
        validation_service = Mock()
        validation_service.is_valid_position.return_value = True
        component.services.validation_service = validation_service
        
        result = component.set_position((5.0, 10.0))
        
        assert result is True
        validation_service.is_valid_position.assert_called_once_with((5.0, 10.0))
    
    def test_set_position_with_validation_service_failure(self, component):
        """Test position validation failure with validation service."""
        validation_service = Mock()
        validation_service.is_valid_position.return_value = False
        component.services.validation_service = validation_service
        
        result = component.set_position((5.0, 10.0))
        
        assert result is False
        assert component.position == (0.0, 0.0)  # Position should not change
        validation_service.is_valid_position.assert_called_once_with((5.0, 10.0))
    
    def test_set_position_validation_service_error(self, component):
        """Test handling of validation service errors."""
        validation_service = Mock()
        validation_service.is_valid_position.side_effect = Exception("Service error")
        component.services.validation_service = validation_service
        
        # Should not raise exception, should continue without validation
        result = component.set_position((5.0, 10.0))
        
        assert result is True
        assert component.position == (5.0, 10.0)


class TestMovementOperations:
    """Test movement operations and distance validation."""
    
    @pytest.fixture
    def component(self):
        """Create a movement component for testing."""
        services = Mock(spec=AgentServices)
        services.spatial_service = None  # Explicitly set to None
        services.validation_service = None
        config = MovementConfig(max_movement=5.0)
        return MovementComponent(services, config)
    
    def test_move_to_valid_distance(self, component):
        """Test moving to a position within max distance."""
        target_position = (3.0, 4.0)  # Distance = 5.0
        result = component.move_to(target_position)
        
        assert result is True
        assert component.position == target_position
    
    def test_move_to_exact_max_distance(self, component):
        """Test moving to a position at exactly max distance."""
        target_position = (5.0, 0.0)  # Distance = 5.0
        result = component.move_to(target_position)
        
        assert result is True
        assert component.position == target_position
    
    def test_move_to_exceeds_max_distance(self, component):
        """Test moving to a position that exceeds max distance."""
        target_position = (6.0, 0.0)  # Distance = 6.0 > 5.0
        result = component.move_to(target_position)
        
        assert result is False
        assert component.position == (0.0, 0.0)  # Position should not change
    
    def test_move_to_diagonal_distance(self, component):
        """Test moving to a diagonal position."""
        # Distance = sqrt(3^2 + 4^2) = 5.0
        target_position = (3.0, 4.0)
        result = component.move_to(target_position)
        
        assert result is True
        assert component.position == target_position
    
    def test_move_to_invalid_position_format(self, component):
        """Test moving to an invalid position format."""
        with pytest.raises(ValueError, match="Invalid position format"):
            component.move_to("invalid")
    
    def test_move_to_with_position_validation_failure(self, component):
        """Test movement when position validation fails."""
        validation_service = Mock()
        validation_service.is_valid_position.return_value = False
        component.services.validation_service = validation_service
        
        target_position = (2.0, 0.0)  # Valid distance
        result = component.move_to(target_position)
        
        assert result is False
        assert component.position == (0.0, 0.0)  # Position should not change
    
    def test_move_to_with_spatial_service_error(self, component):
        """Test movement when spatial service fails."""
        spatial_service = Mock()
        spatial_service.mark_positions_dirty.side_effect = Exception("Spatial error")
        component.services.spatial_service = spatial_service
        
        target_position = (2.0, 0.0)
        result = component.move_to(target_position)
        
        # Movement should still succeed despite spatial service error
        assert result is True
        assert component.position == target_position


class TestSpatialQueries:
    """Test spatial queries and nearby position functionality."""
    
    @pytest.fixture
    def component(self):
        """Create a movement component for testing."""
        services = Mock(spec=AgentServices)
        services.spatial_service = None  # Explicitly set to None
        services.validation_service = None
        config = MovementConfig()
        component = MovementComponent(services, config)
        component.position = (10.0, 10.0)  # Set initial position
        return component
    
    def test_get_nearby_positions_no_spatial_service(self, component):
        """Test getting nearby positions when no spatial service is available."""
        component.services.spatial_service = None
        
        result = component.get_nearby_positions(5)
        
        assert result == []
    
    def test_get_nearby_positions_invalid_radius_type(self, component):
        """Test getting nearby positions with invalid radius type."""
        with pytest.raises(ValueError, match="Radius must be a number"):
            component.get_nearby_positions("invalid")
    
    def test_get_nearby_positions_negative_radius(self, component):
        """Test getting nearby positions with negative radius."""
        with pytest.raises(ValueError, match="Radius must be non-negative"):
            component.get_nearby_positions(-1)
    
    def test_get_nearby_positions_zero_radius(self, component):
        """Test getting nearby positions with zero radius."""
        spatial_service = Mock()
        spatial_service.get_nearby.return_value = []
        component.services.spatial_service = spatial_service
        
        result = component.get_nearby_positions(0)
        
        assert result == []
        spatial_service.get_nearby.assert_called_once_with((10.0, 10.0), 0, [])
    
    def test_get_nearby_positions_dict_result(self, component):
        """Test getting nearby positions with dictionary result from spatial service."""
        # Mock entities with position attributes
        entity1 = Mock()
        entity1.position = (12.0, 10.0)
        entity2 = Mock()
        entity2.position = (8.0, 12.0)
        entity3 = Mock()  # No position attribute
        del entity3.position  # Remove position attribute
        entity4 = Mock()
        entity4.position = None  # None position
        
        spatial_service = Mock()
        spatial_service.get_nearby.return_value = {
            "agents": [entity1, entity2, entity3, entity4]
        }
        component.services.spatial_service = spatial_service
        
        result = component.get_nearby_positions(5)
        
        # Only entities with valid positions should be included
        expected_positions = [(12.0, 10.0), (8.0, 12.0)]
        assert len(result) == 2
        assert (12.0, 10.0) in result
        assert (8.0, 12.0) in result
    
    def test_get_nearby_positions_list_result(self, component):
        """Test getting nearby positions with list result from spatial service."""
        entity1 = Mock()
        entity1.position = (12.0, 10.0)
        entity2 = Mock()
        entity2.position = (8.0, 12.0)
        
        spatial_service = Mock()
        spatial_service.get_nearby.return_value = [entity1, entity2]
        component.services.spatial_service = spatial_service
        
        result = component.get_nearby_positions(5)
        
        expected_positions = [(12.0, 10.0), (8.0, 12.0)]
        assert result == expected_positions
    
    def test_get_nearby_positions_unexpected_result_type(self, component):
        """Test getting nearby positions with unexpected result type."""
        spatial_service = Mock()
        spatial_service.get_nearby.return_value = "unexpected"
        component.services.spatial_service = spatial_service
        
        result = component.get_nearby_positions(5)
        
        assert result == []
    
    def test_get_nearby_positions_spatial_service_error(self, component):
        """Test getting nearby positions when spatial service raises exception."""
        spatial_service = Mock()
        spatial_service.get_nearby.side_effect = Exception("Spatial service error")
        component.services.spatial_service = spatial_service
        
        result = component.get_nearby_positions(5)
        
        assert result == []


class TestServiceIntegration:
    """Test integration with validation and spatial services."""
    
    @pytest.fixture
    def component_with_services(self):
        """Create a movement component with all services."""
        services = Mock(spec=AgentServices)
        services.validation_service = Mock()
        services.spatial_service = Mock()
        services.logging_service = Mock()
        services.metrics_service = Mock()
        services.time_service = Mock()
        services.lifecycle_service = Mock()
        
        config = MovementConfig()
        return MovementComponent(services, config)
    
    def test_spatial_service_mark_dirty_on_position_change(self, component_with_services):
        """Test that spatial service is marked dirty when position changes."""
        component_with_services.set_position((5.0, 10.0))
        
        component_with_services.services.spatial_service.mark_positions_dirty.assert_called_once()
    
    def test_spatial_service_not_marked_dirty_on_same_position(self, component_with_services):
        """Test that spatial service is not marked dirty when position doesn't change."""
        component_with_services.set_position((0.0, 0.0))  # Same as initial position
        
        component_with_services.services.spatial_service.mark_positions_dirty.assert_not_called()
    
    def test_spatial_service_error_handling(self, component_with_services):
        """Test handling of spatial service errors."""
        component_with_services.services.spatial_service.mark_positions_dirty.side_effect = Exception("Spatial error")
        
        # Should not raise exception
        result = component_with_services.set_position((5.0, 10.0))
        
        assert result is True
        assert component_with_services.position == (5.0, 10.0)
    
    def test_validation_service_integration(self, component_with_services):
        """Test integration with validation service."""
        component_with_services.services.validation_service.is_valid_position.return_value = True
        
        result = component_with_services.set_position((5.0, 10.0))
        
        assert result is True
        component_with_services.services.validation_service.is_valid_position.assert_called_once_with((5.0, 10.0))


class TestProperties:
    """Test component properties."""
    
    @pytest.fixture
    def component(self):
        """Create a movement component for testing."""
        services = Mock(spec=AgentServices)
        config = MovementConfig(perception_radius=7)
        component = MovementComponent(services, config)
        component.position = (3.14, 2.71)
        return component
    
    def test_x_property(self, component):
        """Test X coordinate property."""
        assert component.x == 3.14
    
    def test_y_property(self, component):
        """Test Y coordinate property."""
        assert component.y == 2.71
    
    def test_perception_radius_property(self, component):
        """Test perception radius property."""
        assert component.perception_radius == 7


class TestEdgeCases:
    """Test edge cases and error scenarios."""
    
    @pytest.fixture
    def component(self):
        """Create a movement component for testing."""
        services = Mock(spec=AgentServices)
        services.spatial_service = None  # Explicitly set to None
        services.validation_service = None
        config = MovementConfig()
        return MovementComponent(services, config)
    
    def test_move_to_very_small_distance(self, component):
        """Test moving a very small distance."""
        target_position = (0.001, 0.0)
        result = component.move_to(target_position)
        
        assert result is True
        assert component.position == target_position
    
    def test_move_to_very_large_distance(self, component):
        """Test moving a very large distance."""
        target_position = (1000.0, 0.0)
        result = component.move_to(target_position)
        
        assert result is False
        assert component.position == (0.0, 0.0)
    
    def test_set_position_with_float_and_int(self, component):
        """Test setting position with mixed float and int coordinates."""
        target_position = (5, 10.0)  # int and float
        result = component.set_position(target_position)
        
        assert result is True
        assert component.position == target_position
    
    def test_get_nearby_positions_with_float_radius(self, component):
        """Test getting nearby positions with float radius."""
        spatial_service = Mock()
        spatial_service.get_nearby.return_value = []
        component.services.spatial_service = spatial_service
        
        result = component.get_nearby_positions(5.5)
        
        assert result == []
        spatial_service.get_nearby.assert_called_once_with((0.0, 0.0), 5.5, [])
    
    def test_position_with_negative_coordinates(self, component):
        """Test setting position with negative coordinates."""
        target_position = (-5.0, -10.0)
        result = component.set_position(target_position)
        
        assert result is True
        assert component.position == target_position
    
    def test_move_to_negative_coordinates(self, component):
        """Test moving to negative coordinates."""
        target_position = (-3.0, -4.0)  # Distance = 5.0
        result = component.move_to(target_position)
        
        assert result is True
        assert component.position == target_position


class TestLogging:
    """Test logging functionality."""
    
    @pytest.fixture
    def component(self):
        """Create a movement component for testing."""
        services = Mock(spec=AgentServices)
        services.spatial_service = None  # Explicitly set to None
        services.validation_service = None
        config = MovementConfig()
        return MovementComponent(services, config)
    
    @patch('farm.core.agent.components.movement.logger')
    def test_logging_on_position_update(self, mock_logger, component):
        """Test that position updates are logged."""
        component.set_position((5.0, 10.0))
        
        # Should log the position update
        mock_logger.debug.assert_called()
        log_calls = [call[0][0] for call in mock_logger.debug.call_args_list]
        assert any("Position updated from" in call for call in log_calls)
    
    @patch('farm.core.agent.components.movement.logger')
    def test_logging_on_validation_failure(self, mock_logger, component):
        """Test that validation failures are logged."""
        validation_service = Mock()
        validation_service.is_valid_position.return_value = False
        component.services.validation_service = validation_service
        
        component.set_position((5.0, 10.0))
        
        # Should log the validation failure
        mock_logger.debug.assert_called()
        log_calls = [call[0][0] for call in mock_logger.debug.call_args_list]
        assert any("Position validation failed" in call for call in log_calls)
    
    @patch('farm.core.agent.components.movement.logger')
    def test_logging_on_movement_distance_exceeded(self, mock_logger, component):
        """Test that movement distance violations are logged."""
        component.move_to((10.0, 0.0))  # Exceeds max_movement of 8.0
        
        # Should log the distance violation
        mock_logger.debug.assert_called()
        log_calls = [call[0][0] for call in mock_logger.debug.call_args_list]
        assert any("Movement distance" in call and "exceeds max" in call for call in log_calls)
    
    @patch('farm.core.agent.components.movement.logger')
    def test_logging_on_spatial_service_error(self, mock_logger, component):
        """Test that spatial service errors are logged."""
        spatial_service = Mock()
        spatial_service.mark_positions_dirty.side_effect = Exception("Spatial error")
        component.services.spatial_service = spatial_service
        
        component.set_position((5.0, 10.0))
        
        # Should log the spatial service error
        mock_logger.warning.assert_called()
        log_calls = [call[0][0] for call in mock_logger.warning.call_args_list]
        assert any("Failed to mark spatial positions dirty" in call for call in log_calls)


class TestIntegrationScenarios:
    """Test complex integration scenarios."""
    
    def test_complete_movement_workflow(self):
        """Test a complete movement workflow with all services."""
        # Setup services
        validation_service = Mock()
        validation_service.is_valid_position.return_value = True
        
        spatial_service = Mock()
        spatial_service.mark_positions_dirty.return_value = None
        spatial_service.get_nearby.return_value = []
        
        services = Mock(spec=AgentServices)
        services.validation_service = validation_service
        services.spatial_service = spatial_service
        
        # Create component
        config = MovementConfig(max_movement=10.0, perception_radius=5)
        component = MovementComponent(services, config)
        
        # Test workflow
        # 1. Set initial position
        assert component.set_position((0.0, 0.0)) is True
        assert component.position == (0.0, 0.0)
        
        # 2. Move to new position
        assert component.move_to((3.0, 4.0)) is True
        assert component.position == (3.0, 4.0)
        
        # 3. Try to move too far
        assert component.move_to((20.0, 0.0)) is False
        assert component.position == (3.0, 4.0)  # Should not change
        
        # 4. Query nearby positions
        nearby = component.get_nearby_positions(5)
        assert nearby == []
        
        # 5. Verify service calls
        assert validation_service.is_valid_position.call_count == 2  # Two successful moves
        assert spatial_service.mark_positions_dirty.call_count == 1  # One position change (initial position doesn't change)
        assert spatial_service.get_nearby.call_count == 1  # One nearby query
    
    def test_error_recovery_scenarios(self):
        """Test error recovery in various scenarios."""
        # Setup services with errors
        validation_service = Mock()
        validation_service.is_valid_position.side_effect = Exception("Validation error")
        
        spatial_service = Mock()
        spatial_service.mark_positions_dirty.side_effect = Exception("Spatial error")
        spatial_service.get_nearby.side_effect = Exception("Query error")
        
        services = Mock(spec=AgentServices)
        services.validation_service = validation_service
        services.spatial_service = spatial_service
        
        config = MovementConfig()
        component = MovementComponent(services, config)
        
        # Test that component continues to work despite service errors
        assert component.set_position((5.0, 10.0)) is True
        assert component.position == (5.0, 10.0)
        
        assert component.move_to((2.0, 3.0)) is True
        assert component.position == (2.0, 3.0)
        
        nearby = component.get_nearby_positions(5)
        assert nearby == []  # Should return empty list on error
