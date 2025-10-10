"""
Unit tests for CombatComponent.

Tests verify:
- Health tracking
- Attack mechanics
- Damage calculation
- Defense mechanics
- Death triggering
"""

import pytest
from unittest.mock import Mock
from farm.core.agent.config.agent_config import CombatConfig
from farm.core.agent.components.combat import CombatComponent


@pytest.fixture
def mock_agent():
    """Create a mock agent."""
    agent = Mock()
    agent.agent_id = "test_agent"
    agent.alive = True
    agent.terminate = Mock()
    agent.get_component = Mock(return_value=None)
    return agent


@pytest.fixture
def combat_component(mock_agent):
    """Create a CombatComponent attached to mock agent."""
    config = CombatConfig(
        starting_health=100.0,
        base_attack_strength=10.0,
        defense_reduction=0.5
    )
    component = CombatComponent(config)
    component.attach(mock_agent)
    return component


class TestCombatComponent:
    """Tests for CombatComponent."""
    
    def test_component_name(self, combat_component):
        """Test component has correct name."""
        assert combat_component.name == "combat"
    
    def test_initial_health(self, combat_component):
        """Test initial health equals starting health."""
        assert combat_component.health == 100.0
        assert combat_component.max_health == 100.0
    
    def test_health_ratio_full(self, combat_component):
        """Test health ratio when at full health."""
        assert combat_component.health_ratio == 1.0
    
    def test_health_ratio_half(self, combat_component):
        """Test health ratio when at half health."""
        combat_component.set_health(50.0)
        assert combat_component.health_ratio == 0.5
    
    def test_is_alive_true(self, combat_component):
        """Test is_alive when agent has health."""
        assert combat_component.is_alive is True
    
    def test_is_alive_false(self, combat_component):
        """Test is_alive when health depleted."""
        combat_component.set_health(0.0)
        assert combat_component.is_alive is False
    
    def test_take_damage_normal(self, combat_component):
        """Test taking damage."""
        damage_taken = combat_component.take_damage(25.0)
        
        assert damage_taken == 25.0
        assert combat_component.health == 75.0
    
    def test_take_damage_with_defense(self, combat_component):
        """Test damage reduction when defending."""
        combat_component.start_defense()
        damage_taken = combat_component.take_damage(40.0)
        
        # Should take 50% damage (40 * 0.5 = 20)
        assert damage_taken == 20.0
        assert combat_component.health == 80.0
    
    def test_take_damage_death(self, combat_component, mock_agent):
        """Test that fatal damage triggers death."""
        combat_component.take_damage(150.0)
        
        assert combat_component.health == 0.0
        mock_agent.terminate.assert_called_once()
    
    def test_heal(self, combat_component):
        """Test healing."""
        combat_component.set_health(50.0)
        healed = combat_component.heal(30.0)
        
        assert healed == 30.0
        assert combat_component.health == 80.0
    
    def test_heal_capped_at_max(self, combat_component):
        """Test healing is capped at max health."""
        combat_component.set_health(90.0)
        healed = combat_component.heal(50.0)
        
        assert healed == 10.0  # Only healed to max
        assert combat_component.health == 100.0
    
    def test_set_health(self, combat_component):
        """Test setting health directly."""
        combat_component.set_health(75.0)
        assert combat_component.health == 75.0
    
    def test_set_health_triggers_death(self, combat_component, mock_agent):
        """Test setting health to 0 triggers death."""
        combat_component.set_health(0.0)
        
        assert combat_component.health == 0.0
        mock_agent.terminate.assert_called_once()
    
    def test_defense_not_active_initially(self, combat_component):
        """Test agent not defending initially."""
        assert combat_component.is_defending is False
        assert combat_component.defense_turns_remaining == 0
    
    def test_start_defense(self, combat_component):
        """Test starting defensive stance."""
        combat_component.start_defense(3)
        
        assert combat_component.is_defending is True
        assert combat_component.defense_turns_remaining == 3
    
    def test_defense_timer_countdown(self, combat_component):
        """Test defense timer decrements."""
        combat_component.start_defense(3)
        
        combat_component.on_step_end()
        assert combat_component.defense_turns_remaining == 2
        assert combat_component.is_defending is True
        
        combat_component.on_step_end()
        assert combat_component.defense_turns_remaining == 1
        assert combat_component.is_defending is True
        
        combat_component.on_step_end()
        assert combat_component.defense_turns_remaining == 0
        assert combat_component.is_defending is False
    
    def test_end_defense(self, combat_component):
        """Test ending defense early."""
        combat_component.start_defense(5)
        combat_component.end_defense()
        
        assert combat_component.is_defending is False
        assert combat_component.defense_turns_remaining == 0
    
    def test_get_defense_strength_not_defending(self, combat_component):
        """Test defense strength when not defending."""
        assert combat_component.get_defense_strength() == 0.0
    
    def test_get_defense_strength_defending(self, combat_component):
        """Test defense strength when defending."""
        combat_component.start_defense()
        # Default base_defense_strength is 5.0
        assert combat_component.get_defense_strength() == 5.0
    
    def test_attack_successful(self, combat_component, mock_agent):
        """Test attacking another agent."""
        # Create target agent with combat component
        target = Mock()
        target.agent_id = "target_agent"
        
        target_combat = CombatComponent(CombatConfig())
        target_combat.attach(Mock())
        
        mock_agent.get_component = Mock(return_value=combat_component)
        target.get_component = Mock(return_value=target_combat)
        
        result = combat_component.attack(target)
        
        assert result["success"] is True
        assert result["damage_dealt"] == 10.0  # Base attack strength
        assert result["target_health"] == 90.0
    
    def test_attack_damaged_attacker(self, combat_component, mock_agent):
        """Test attack damage scales with attacker health."""
        # Reduce attacker health to 50%
        combat_component.set_health(50.0)
        
        # Create target
        target = Mock()
        target_combat = CombatComponent(CombatConfig())
        target_combat.attach(Mock())
        target.get_component = Mock(return_value=target_combat)
        
        result = combat_component.attack(target)
        
        # Damage should be scaled by health ratio (10.0 * 0.5 = 5.0)
        assert result["damage_dealt"] == 5.0
    
    def test_attack_no_target_combat(self, combat_component, mock_agent):
        """Test attacking agent without combat component."""
        target = Mock()
        target.get_component = Mock(return_value=None)
        
        result = combat_component.attack(target)
        
        assert result["success"] is False
        assert "no combat component" in result["error"].lower()
    
    def test_get_state(self, combat_component):
        """Test state serialization."""
        combat_component.set_health(75.0)
        combat_component.start_defense(2)
        
        state = combat_component.get_state()
        
        assert state["health"] == 75.0
        assert state["is_defending"] is True
        assert state["defense_timer"] == 2
    
    def test_load_state(self, combat_component):
        """Test state deserialization."""
        state = {
            "health": 60.0,
            "is_defending": True,
            "defense_timer": 3
        }
        
        combat_component.load_state(state)
        
        assert combat_component.health == 60.0
        assert combat_component.is_defending is True
        assert combat_component.defense_turns_remaining == 3
    
    def test_round_trip_serialization(self, combat_component):
        """Test save/load preserves state."""
        combat_component.set_health(85.0)
        combat_component.start_defense(4)
        
        state = combat_component.get_state()
        
        new_component = CombatComponent(CombatConfig())
        new_component.load_state(state)
        
        assert new_component.health == 85.0
        assert new_component.is_defending is True
        assert new_component.defense_turns_remaining == 4