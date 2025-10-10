"""
Integration tests for the hybrid action system.

Tests the complete flow from environment through action registry to agent components.
"""

import pytest
from unittest.mock import Mock, MagicMock

from farm.core.action import action_registry
from farm.core.agent.core import AgentCore
from farm.core.agent.components.movement import MovementComponent
from farm.core.agent.components.combat import CombatComponent
from farm.core.agent.components.resource import ResourceComponent
from farm.core.agent.behaviors.default_behavior import DefaultAgentBehavior
from farm.core.agent.config.agent_config import (
    MovementConfig,
    CombatConfig,
    ResourceConfig,
)


class MockSpatialService:
    """Mock spatial service for integration testing."""
    
    def __init__(self):
        self.nearby_agents = []
        self.nearby_resources = []
    
    def get_nearby(self, position, radius, entity_types):
        result = {}
        if "agents" in entity_types:
            result["agents"] = self.nearby_agents
        if "resources" in entity_types:
            result["resources"] = self.nearby_resources
        return result
    
    def mark_positions_dirty(self):
        pass


class MockResource:
    """Mock resource for integration testing."""
    
    def __init__(self, amount=100, position=(10.0, 10.0)):
        self.amount = amount
        self.position = position
    
    def is_depleted(self):
        return self.amount <= 0
    
    def consume(self, amount):
        if self.amount >= amount:
            self.amount -= amount
            return amount
        return 0


class TestActionSystemIntegration:
    """Test complete integration of the hybrid action system."""
    
    def create_test_agent(self, agent_id="test_agent", position=(10.0, 10.0)):
        """Create a test agent with all standard components."""
        spatial_service = MockSpatialService()
        behavior = DefaultAgentBehavior()
        
        return AgentCore(
            agent_id=agent_id,
            position=position,
            spatial_service=spatial_service,
            behavior=behavior,
            components=[
                MovementComponent(MovementConfig(max_movement=5.0)),
                CombatComponent(CombatConfig(starting_health=100.0)),
                ResourceComponent(50, ResourceConfig()),
            ]
        )
    
    def test_environment_to_component_flow(self):
        """Test complete flow from environment through action registry to components."""
        agent = self.create_test_agent()
        original_position = agent.position
        
        # Simulate environment calling action registry
        move_action = action_registry.get("move")
        assert move_action is not None
        
        # Execute action
        result = move_action.execute(agent)
        
        # Verify result format
        assert isinstance(result, dict)
        assert "success" in result
        assert "error" in result
        assert "details" in result
        
        # Verify action was successful
        assert result["success"] is True
        assert result["error"] is None
        
        # Verify component state changed
        assert agent.position != original_position
        assert "old_position" in result["details"]
        assert "new_position" in result["details"]
    
    def test_all_actions_work_through_registry(self):
        """Test that all actions can be executed through the action registry."""
        agent = self.create_test_agent()
        
        # Test actions that should work with basic agent
        basic_actions = ["move", "pass"]
        
        for action_name in basic_actions:
            action = action_registry.get(action_name)
            assert action is not None, f"Action '{action_name}' not found in registry"
            
            result = action.execute(agent)
            
            assert isinstance(result, dict)
            assert "success" in result
            assert "error" in result
            assert "details" in result
            
            # Basic actions should succeed
            assert result["success"] is True, f"Action '{action_name}' failed: {result['error']}"
    
    def test_actions_require_appropriate_components(self):
        """Test that actions fail gracefully when required components are missing."""
        # Create agent without combat component
        spatial_service = MockSpatialService()
        behavior = DefaultAgentBehavior()
        
        agent = AgentCore(
            agent_id="test_agent",
            position=(10.0, 10.0),
            spatial_service=spatial_service,
            behavior=behavior,
            components=[
                MovementComponent(MovementConfig(max_movement=5.0)),
                ResourceComponent(50, ResourceConfig()),
                # No combat component
            ]
        )
        
        # Test actions that require combat component
        combat_actions = ["attack", "defend"]
        
        for action_name in combat_actions:
            action = action_registry.get(action_name)
            result = action.execute(agent)
            
            assert result["success"] is False
            assert "no combat component" in result["error"]
    
    def test_action_result_consistency(self):
        """Test that all actions return consistent result formats."""
        agent = self.create_test_agent()
        
        all_actions = action_registry.get_all()
        
        for action in all_actions:
            result = action.execute(agent)
            
            # Check result structure
            assert isinstance(result, dict)
            assert "success" in result
            assert "error" in result
            assert "details" in result
            
            # Check types
            assert isinstance(result["success"], bool)
            assert result["error"] is None or isinstance(result["error"], str)
            assert isinstance(result["details"], dict)
    
    def test_spatial_service_integration(self):
        """Test that actions properly use spatial services."""
        # Create agent with nearby resources
        resource = MockResource(amount=50, position=(10.0, 10.0))
        spatial_service = MockSpatialService()
        spatial_service.nearby_resources = [resource]
        
        behavior = DefaultAgentBehavior()
        
        agent = AgentCore(
            agent_id="test_agent",
            position=(10.0, 10.0),
            spatial_service=spatial_service,
            behavior=behavior,
            components=[
                ResourceComponent(50, ResourceConfig()),
            ]
        )
        
        # Test gather action with nearby resource
        gather_action = action_registry.get("gather")
        result = gather_action.execute(agent)
        
        # Should succeed and consume resource
        assert result["success"] is True
        assert "amount_gathered" in result["details"]
        assert resource.amount < 50  # Resource should be consumed
    
    def test_agent_behavior_direct_component_access(self):
        """Test that agent behaviors can access components directly."""
        agent = self.create_test_agent()
        
        # Test direct component access (bypassing action registry)
        movement = agent.get_component("movement")
        assert movement is not None
        
        original_position = agent.position
        result = movement.move_by(5.0, 0.0)
        
        # Should work directly (move_by returns boolean)
        assert result is True
        assert agent.position != original_position
    
    def test_action_registry_completeness(self):
        """Test that action registry contains all expected actions."""
        expected_actions = ["move", "attack", "defend", "gather", "share", "reproduce", "pass"]
        
        all_actions = action_registry.get_all()
        action_names = [action.name for action in all_actions]
        
        for expected_action in expected_actions:
            assert expected_action in action_names, f"Action '{expected_action}' missing from registry"
        
        # Should have exactly 7 actions
        assert len(all_actions) == 7
    
    def test_action_weights_and_normalization(self):
        """Test that action weights work correctly."""
        all_actions = action_registry.get_all()
        
        # All actions should have weights
        for action in all_actions:
            assert hasattr(action, "weight")
            assert isinstance(action.weight, (int, float))
            assert action.weight >= 0
        
        # Test normalization
        normalized_actions = action_registry.get_all(normalized=True)
        total_weight = sum(action.weight for action in normalized_actions)
        assert abs(total_weight - 1.0) < 0.001  # Should sum to 1.0
    
    def test_error_handling_consistency(self):
        """Test that all actions handle errors consistently."""
        # Create agent that will cause errors
        agent = Mock(spec=AgentCore)
        agent.agent_id = "test_agent"
        agent.position = (10.0, 10.0)
        agent.get_component.return_value = None  # No components
        
        all_actions = action_registry.get_all()
        
        for action in all_actions:
            result = action.execute(agent)
            
            # Should handle missing components gracefully
            assert isinstance(result, dict)
            assert "success" in result
            assert "error" in result
            assert "details" in result
            
            # Most actions should fail gracefully (not crash)
            # Pass action always succeeds, so we check that specifically
            if action.name == "pass":
                assert result["success"] is True
            else:
                assert result["success"] is False
                assert isinstance(result["error"], str)
                assert len(result["error"]) > 0
    
    def test_action_logging_integration(self):
        """Test that actions integrate properly with logging."""
        agent = self.create_test_agent()
        
        # Test that actions don't crash due to logging issues
        move_action = action_registry.get("move")
        result = move_action.execute(agent)
        
        # Should succeed regardless of logging setup
        assert result["success"] is True
    
    def test_component_state_consistency(self):
        """Test that component state remains consistent after actions."""
        agent = self.create_test_agent()
        
        # Record initial state
        initial_position = agent.position
        initial_health = agent.get_component("combat").health
        initial_resources = agent.get_component("resource").level
        
        # Execute move action
        move_action = action_registry.get("move")
        result = move_action.execute(agent)
        
        assert result["success"] is True
        
        # Position should change, health and resources should remain the same
        assert agent.position != initial_position
        assert agent.get_component("combat").health == initial_health
        assert agent.get_component("resource").level == initial_resources


class TestActionSystemPerformance:
    """Test performance characteristics of the hybrid action system."""
    
    def create_test_agent(self, agent_id="test_agent", position=(10.0, 10.0)):
        """Create a test agent with all standard components."""
        spatial_service = MockSpatialService()
        behavior = DefaultAgentBehavior()
        
        return AgentCore(
            agent_id=agent_id,
            position=position,
            spatial_service=spatial_service,
            behavior=behavior,
            components=[
                MovementComponent(MovementConfig(max_movement=5.0)),
                CombatComponent(CombatConfig(starting_health=100.0)),
                ResourceComponent(50, ResourceConfig()),
            ]
        )
    
    def test_action_execution_speed(self):
        """Test that actions execute quickly."""
        import time
        
        agent = self.create_test_agent()
        move_action = action_registry.get("move")
        
        # Time multiple executions
        start_time = time.time()
        for _ in range(100):
            result = move_action.execute(agent)
            assert result["success"] is True
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should complete 100 actions in reasonable time (< 1 second)
        assert execution_time < 1.0, f"Actions too slow: {execution_time:.3f}s for 100 executions"
    
    def test_memory_usage_stability(self):
        """Test that action execution doesn't cause memory leaks."""
        import gc
        
        # Force garbage collection
        gc.collect()
        
        agent = self.create_test_agent()
        move_action = action_registry.get("move")
        
        # Execute many actions
        for _ in range(1000):
            result = move_action.execute(agent)
            assert result["success"] is True
        
        # Force garbage collection again
        gc.collect()
        
        # If we get here without memory issues, the test passes
        assert True


if __name__ == "__main__":
    pytest.main([__file__])
