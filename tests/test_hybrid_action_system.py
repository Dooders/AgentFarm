"""
Unit tests for the hybrid action system.

Tests the refactored action functions that delegate to agent components
while maintaining compatibility with the environment's action registry.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any

from farm.core.action import (
    action_registry,
    ActionType,
    move_action,
    attack_action,
    defend_action,
    gather_action,
    share_action,
    reproduce_action,
    pass_action,
    validate_agent_config,
    calculate_euclidean_distance,
    find_closest_entity,
)
from farm.core.agent.core import AgentCore
from farm.core.agent.components.movement import MovementComponent
from farm.core.agent.components.combat import CombatComponent
from farm.core.agent.components.resource import ResourceComponent
from farm.core.agent.components.reproduction import ReproductionComponent
from farm.core.agent.behaviors.default_behavior import DefaultAgentBehavior
from farm.core.agent.config.agent_config import (
    MovementConfig,
    CombatConfig,
    ResourceConfig,
    ReproductionConfig,
)


class MockSpatialService:
    """Mock spatial service for testing."""
    
    def __init__(self, nearby_agents=None, nearby_resources=None):
        self.nearby_agents = nearby_agents or []
        self.nearby_resources = nearby_resources or []
    
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
    """Mock resource for testing."""
    
    def __init__(self, amount=100, resource_id="test_resource", position=(10.0, 10.0)):
        self.amount = amount
        self.resource_id = resource_id
        self.position = position
    
    def is_depleted(self):
        return self.amount <= 0
    
    def consume(self, amount):
        if self.amount >= amount:
            self.amount -= amount
            return amount
        return 0


class TestActionRegistryIntegration:
    """Test that action registry works with refactored actions."""
    
    def test_action_registry_contains_all_actions(self):
        """Test that all actions are registered in the action registry."""
        expected_actions = ["move", "attack", "defend", "gather", "share", "reproduce", "pass"]
        
        for action_name in expected_actions:
            action = action_registry.get(action_name)
            assert action is not None, f"Action '{action_name}' not found in registry"
            assert action.name == action_name
    
    def test_action_registry_execution_works(self):
        """Test that actions can be executed through the registry."""
        # Create a minimal agent
        spatial_service = MockSpatialService()
        behavior = DefaultAgentBehavior()
        
        agent = AgentCore(
            agent_id="test_agent",
            position=(10.0, 10.0),
            spatial_service=spatial_service,
            behavior=behavior,
            components=[
                MovementComponent(MovementConfig(max_movement=5.0)),
                CombatComponent(CombatConfig(starting_health=100.0)),
                ResourceComponent(50, ResourceConfig()),
            ]
        )
        
        # Test that actions can be executed
        move_action = action_registry.get("move")
        result = move_action.execute(agent)
        
        assert isinstance(result, dict)
        assert "success" in result
        assert "error" in result
        assert "details" in result


class TestActionFunctionRefactoring:
    """Test that action functions properly delegate to components."""
    
    def create_test_agent(self, components=None):
        """Create a test agent with specified components."""
        if components is None:
            components = [
                MovementComponent(MovementConfig(max_movement=5.0)),
                CombatComponent(CombatConfig(starting_health=100.0)),
                ResourceComponent(50, ResourceConfig()),
            ]
        
        spatial_service = MockSpatialService()
        behavior = DefaultAgentBehavior()
        
        return AgentCore(
            agent_id="test_agent",
            position=(10.0, 10.0),
            spatial_service=spatial_service,
            behavior=behavior,
            components=components,
        )
    
    def test_move_action_delegates_to_movement_component(self):
        """Test that move_action delegates to MovementComponent."""
        agent = self.create_test_agent()
        original_position = agent.position
        
        result = move_action(agent)
        
        assert result["success"] is True
        assert result["error"] is None
        assert "old_position" in result["details"]
        assert "new_position" in result["details"]
        assert result["details"]["old_position"] == original_position
        # Position should have changed
        assert agent.position != original_position
    
    def test_move_action_fails_without_movement_component(self):
        """Test that move_action fails when agent has no movement component."""
        agent = self.create_test_agent(components=[
            CombatComponent(CombatConfig(starting_health=100.0)),
            ResourceComponent(50, ResourceConfig()),
        ])
        
        result = move_action(agent)
        
        assert result["success"] is False
        assert "no movement component" in result["error"]
    
    def test_attack_action_delegates_to_combat_component(self):
        """Test that attack_action delegates to CombatComponent."""
        # Create target agent
        target = self.create_test_agent()
        target.agent_id = "target_agent"
        
        # Create attacker with target nearby
        spatial_service = MockSpatialService(nearby_agents=[target])
        behavior = DefaultAgentBehavior()
        
        attacker = AgentCore(
            agent_id="attacker",
            position=(10.0, 10.0),
            spatial_service=spatial_service,
            behavior=behavior,
            components=[
                CombatComponent(CombatConfig(starting_health=100.0)),
            ]
        )
        
        result = attack_action(attacker)
        
        assert result["success"] is True
        assert "damage_dealt" in result["details"]
        assert "target_id" in result["details"]
        assert result["details"]["target_id"] == "target_agent"
    
    def test_attack_action_fails_without_combat_component(self):
        """Test that attack_action fails when agent has no combat component."""
        agent = self.create_test_agent(components=[
            MovementComponent(MovementConfig(max_movement=5.0)),
            ResourceComponent(50, ResourceConfig()),
        ])
        
        result = attack_action(agent)
        
        assert result["success"] is False
        assert "no combat component" in result["error"]
    
    def test_attack_action_fails_with_no_targets(self):
        """Test that attack_action fails when no targets are nearby."""
        agent = self.create_test_agent()
        
        result = attack_action(agent)
        
        assert result["success"] is False
        assert "No valid targets found" in result["error"]
    
    def test_defend_action_delegates_to_combat_component(self):
        """Test that defend_action delegates to CombatComponent."""
        agent = self.create_test_agent()
        combat = agent.get_component("combat")
        
        # Initially not defending
        assert not combat.is_defending
        
        result = defend_action(agent)
        
        assert result["success"] is True
        assert "duration" in result["details"]
        assert "cost" in result["details"]
        # Should now be defending
        assert combat.is_defending
    
    def test_defend_action_fails_without_combat_component(self):
        """Test that defend_action fails when agent has no combat component."""
        agent = self.create_test_agent(components=[
            MovementComponent(MovementConfig(max_movement=5.0)),
            ResourceComponent(50, ResourceConfig()),
        ])
        
        result = defend_action(agent)
        
        assert result["success"] is False
        assert "no combat component" in result["error"]
    
    def test_gather_action_delegates_to_resource_component(self):
        """Test that gather_action delegates to ResourceComponent."""
        # Create resource nearby
        resource = MockResource(amount=50)
        spatial_service = MockSpatialService(nearby_resources=[resource])
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
        
        original_resources = agent.resource_level
        original_resource_amount = resource.amount
        
        result = gather_action(agent)
        
        assert result["success"] is True
        assert "amount_gathered" in result["details"]
        assert "agent_resources_before" in result["details"]
        assert "agent_resources_after" in result["details"]
        
        # Agent should have more resources
        assert agent.resource_level > original_resources
        # Resource should have less amount
        assert resource.amount < original_resource_amount
    
    def test_gather_action_fails_without_resource_component(self):
        """Test that gather_action fails when agent has no resource component."""
        agent = self.create_test_agent(components=[
            MovementComponent(MovementConfig(max_movement=5.0)),
            CombatComponent(CombatConfig(starting_health=100.0)),
        ])
        
        result = gather_action(agent)
        
        assert result["success"] is False
        assert "no resource component" in result["error"]
    
    def test_gather_action_fails_with_no_resources(self):
        """Test that gather_action fails when no resources are nearby."""
        agent = self.create_test_agent()
        
        result = gather_action(agent)
        
        assert result["success"] is False
        assert "No available resources found" in result["error"]
    
    def test_share_action_delegates_to_resource_components(self):
        """Test that share_action delegates to ResourceComponent methods."""
        # Create target agent
        target = self.create_test_agent()
        target.agent_id = "target_agent"
        
        # Create sharer with target nearby
        spatial_service = MockSpatialService(nearby_agents=[target])
        behavior = DefaultAgentBehavior()
        
        sharer = AgentCore(
            agent_id="sharer",
            position=(10.0, 10.0),
            spatial_service=spatial_service,
            behavior=behavior,
            components=[
                ResourceComponent(50, ResourceConfig()),
            ]
        )
        
        original_sharer_resources = sharer.resource_level
        original_target_resources = target.resource_level
        
        result = share_action(sharer)
        
        assert result["success"] is True
        assert "amount_shared" in result["details"]
        assert "agent_resources_before" in result["details"]
        assert "agent_resources_after" in result["details"]
        assert "target_resources_before" in result["details"]
        assert "target_resources_after" in result["details"]
        
        # Sharer should have fewer resources
        assert sharer.resource_level < original_sharer_resources
        # Target should have more resources
        assert target.resource_level > original_target_resources
    
    def test_share_action_fails_without_resource_component(self):
        """Test that share_action fails when agent has no resource component."""
        agent = self.create_test_agent(components=[
            MovementComponent(MovementConfig(max_movement=5.0)),
            CombatComponent(CombatConfig(starting_health=100.0)),
        ])
        
        result = share_action(agent)
        
        assert result["success"] is False
        assert "no resource component" in result["error"]
    
    def test_share_action_fails_with_no_targets(self):
        """Test that share_action fails when no targets are nearby."""
        agent = self.create_test_agent()
        
        result = share_action(agent)
        
        assert result["success"] is False
        assert "No valid targets found" in result["error"]
    
    def test_reproduce_action_delegates_to_reproduction_component(self):
        """Test that reproduce_action delegates to ReproductionComponent."""
        # Mock lifecycle service
        lifecycle_service = Mock()
        lifecycle_service.get_next_agent_id.return_value = "offspring_001"
        lifecycle_service.add_agent = Mock()
        
        # Create agent with reproduction component
        spatial_service = MockSpatialService()
        behavior = DefaultAgentBehavior()
        
        agent = AgentCore(
            agent_id="parent",
            position=(10.0, 10.0),
            spatial_service=spatial_service,
            behavior=behavior,
            components=[
                ResourceComponent(100, ResourceConfig()),  # Enough resources
                ReproductionComponent(
                    ReproductionConfig(
                        reproduction_threshold=50,
                        offspring_cost=20,
                        offspring_initial_resources=30,
                    ),
                    lifecycle_service=lifecycle_service,
                ),
            ],
            lifecycle_service=lifecycle_service,
        )
        
        original_resources = agent.resource_level
        
        result = reproduce_action(agent)
        
        # Should succeed (assuming reproduction chance roll passes)
        if result["success"]:
            assert "offspring_id" in result["details"]
            assert "cost" in result["details"]
            # Parent should have fewer resources
            assert agent.resource_level < original_resources
        else:
            # Might fail due to reproduction chance
            assert "Reproduction chance not met" in result["error"] or "Insufficient resources" in result["error"]
    
    def test_reproduce_action_fails_without_reproduction_component(self):
        """Test that reproduce_action fails when agent has no reproduction component."""
        agent = self.create_test_agent()
        
        result = reproduce_action(agent)
        
        assert result["success"] is False
        assert "no reproduction component" in result["error"]
    
    def test_pass_action_works_without_components(self):
        """Test that pass_action works without requiring specific components."""
        agent = self.create_test_agent()
        
        result = pass_action(agent)
        
        assert result["success"] is True
        assert "reason" in result["details"]
        assert result["details"]["reason"] == "strategic_inaction"


class TestActionHelperFunctions:
    """Test helper functions used by actions."""
    
    def test_validate_agent_config_always_returns_true_for_agentcore(self):
        """Test that validate_agent_config always returns True for AgentCore."""
        agent = Mock(spec=AgentCore)
        agent.agent_id = "test_agent"
        
        result = validate_agent_config(agent, "test_action")
        
        assert result is True
    
    def test_calculate_euclidean_distance(self):
        """Test Euclidean distance calculation."""
        pos1 = (0.0, 0.0)
        pos2 = (3.0, 4.0)
        
        distance = calculate_euclidean_distance(pos1, pos2)
        
        assert distance == 5.0
    
    def test_find_closest_entity(self):
        """Test finding closest entity."""
        agent = Mock()
        agent.position = (1.0, 1.0)
        
        entity1 = Mock()
        entity1.position = (2.0, 1.0)  # Distance: 1.0
        
        entity2 = Mock()
        entity2.position = (5.0, 5.0)  # Distance: 5.66
        
        closest, distance = find_closest_entity(agent, [entity1, entity2], "test")
        
        assert closest is entity1
        assert distance == pytest.approx(1.0)
    
    def test_find_closest_entity_empty_list(self):
        """Test finding closest entity with empty list."""
        agent = Mock()
        agent.position = (1.0, 1.0)
        
        closest, distance = find_closest_entity(agent, [], "test")
        
        assert closest is None
        assert distance == float("inf")


class TestActionErrorHandling:
    """Test error handling in action functions."""
    
    def test_action_functions_handle_exceptions_gracefully(self):
        """Test that action functions handle exceptions gracefully."""
        # Create agent with components that will raise exceptions
        movement = Mock(spec=MovementComponent)
        movement.max_movement = 5.0
        movement.move_by.side_effect = Exception("Movement failed")
        
        agent = Mock(spec=AgentCore)
        agent.agent_id = "test_agent"
        agent.position = (10.0, 10.0)
        agent.get_component.return_value = movement
        
        result = move_action(agent)
        
        assert result["success"] is False
        assert "exception" in result["error"].lower()
        assert "exception_type" in result["details"]
    
    def test_action_functions_return_proper_result_format(self):
        """Test that all action functions return the expected result format."""
        agent = Mock(spec=AgentCore)
        agent.agent_id = "test_agent"
        agent.position = (10.0, 10.0)
        agent.get_component.return_value = None  # No components
        
        actions = [move_action, attack_action, defend_action, gather_action, share_action, reproduce_action, pass_action]
        
        for action_func in actions:
            result = action_func(agent)
            
            # Check result format
            assert isinstance(result, dict)
            assert "success" in result
            assert "error" in result
            assert "details" in result
            
            assert isinstance(result["success"], bool)
            assert result["error"] is None or isinstance(result["error"], str)
            assert isinstance(result["details"], dict)


class TestActionRegistryCompatibility:
    """Test that the action registry maintains compatibility with existing code."""
    
    def test_action_registry_returns_action_objects(self):
        """Test that action registry returns proper Action objects."""
        move_action_obj = action_registry.get("move")
        
        assert move_action_obj is not None
        assert hasattr(move_action_obj, "name")
        assert hasattr(move_action_obj, "weight")
        assert hasattr(move_action_obj, "function")
        assert hasattr(move_action_obj, "execute")
        
        assert move_action_obj.name == "move"
    
    def test_action_registry_get_all_returns_all_actions(self):
        """Test that get_all returns all registered actions."""
        all_actions = action_registry.get_all()
        
        assert len(all_actions) == 7  # All 7 actions should be registered
        
        action_names = [action.name for action in all_actions]
        expected_names = ["move", "attack", "defend", "gather", "share", "reproduce", "pass"]
        
        for expected_name in expected_names:
            assert expected_name in action_names
    
    def test_action_registry_normalization(self):
        """Test that action registry can normalize weights."""
        all_actions = action_registry.get_all(normalized=True)
        
        total_weight = sum(action.weight for action in all_actions)
        assert abs(total_weight - 1.0) < 0.001  # Should sum to 1.0 when normalized


class TestActionIntegrationWithEnvironment:
    """Test integration between actions and environment systems."""
    
    def test_actions_work_with_mock_environment_services(self):
        """Test that actions work with mock environment services."""
        # Create a more realistic mock environment setup
        spatial_service = MockSpatialService()
        behavior = DefaultAgentBehavior()
        
        agent = AgentCore(
            agent_id="test_agent",
            position=(10.0, 10.0),
            spatial_service=spatial_service,
            behavior=behavior,
            components=[
                MovementComponent(MovementConfig(max_movement=5.0)),
                CombatComponent(CombatConfig(starting_health=100.0)),
                ResourceComponent(50, ResourceConfig()),
            ]
        )
        
        # Test that all actions can be executed
        actions_to_test = [
            ("move", move_action),
            ("pass", pass_action),
        ]
        
        for action_name, action_func in actions_to_test:
            result = action_func(agent)
            assert result["success"] is True, f"{action_name} action failed: {result['error']}"
    
    def test_actions_maintain_logging_compatibility(self):
        """Test that actions maintain logging compatibility."""
        # This test ensures that the logging calls in actions don't break
        # even if the logging infrastructure is not fully set up
        
        agent = Mock(spec=AgentCore)
        agent.agent_id = "test_agent"
        agent.position = (10.0, 10.0)
        agent.get_component.return_value = None
        
        # Should not raise exceptions even with minimal setup
        result = pass_action(agent)
        assert isinstance(result, dict)


if __name__ == "__main__":
    pytest.main([__file__])
