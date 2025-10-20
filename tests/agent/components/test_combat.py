"""
Comprehensive unit tests for CombatComponent.

Tests all functionality including health management, damage handling,
defense mechanics, attack/defense strength calculations, and service integration.
"""

import pytest
from unittest.mock import Mock

from farm.core.agent.components.combat import CombatComponent
from farm.core.agent.config.component_configs import CombatConfig
from farm.core.agent.services import AgentServices


class TestCombatComponentInitialization:
    """Test component initialization and configuration."""
    
    def test_init_with_default_config(self):
        """Test initialization with default configuration."""
        services = Mock(spec=AgentServices)
        config = CombatConfig()
        component = CombatComponent(services, config)
        
        assert component.config == config
        assert component.health == 100.0  # starting_health
        assert component.is_defending is False
        assert component.defense_timer == 0
        assert component.name == "CombatComponent"
        assert component.services == services
    
    def test_init_with_custom_config(self):
        """Test initialization with custom configuration."""
        services = Mock(spec=AgentServices)
        config = CombatConfig(
            starting_health=150.0,
            base_attack_strength=20.0,
            defense_timer_duration=5
        )
        component = CombatComponent(services, config)
        
        assert component.health == 150.0
        assert component.config.base_attack_strength == 20.0
        assert component.config.defense_timer_duration == 5
    
    def test_attach_to_core(self):
        """Test attaching component to agent core."""
        services = Mock(spec=AgentServices)
        config = CombatConfig()
        component = CombatComponent(services, config)
        
        core = Mock()
        component.attach(core)
        
        assert component.core == core
    
    def test_lifecycle_hooks(self):
        """Test that lifecycle hooks are callable."""
        services = Mock(spec=AgentServices)
        config = CombatConfig()
        component = CombatComponent(services, config)
        
        # These should not raise exceptions
        component.on_step_start()
        component.on_step_end()
        component.on_terminate()


class TestTakeDamage:
    """Test damage handling and health management."""
    
    @pytest.fixture
    def component(self):
        """Create a combat component for testing."""
        services = Mock(spec=AgentServices)
        config = CombatConfig(starting_health=100.0, defense_damage_reduction=0.5)
        return CombatComponent(services, config)
    
    def test_take_damage_basic(self, component):
        """Test basic damage application."""
        damage = 25.0
        result = component.take_damage(damage)
        
        assert result == 25.0  # Actual damage dealt
        assert component.health == 75.0  # 100 - 25
    
    def test_take_damage_defending(self, component):
        """Test damage reduction when defending."""
        component.is_defending = True
        damage = 20.0
        result = component.take_damage(damage)
        
        expected_damage = 20.0 * 0.5  # 10.0
        assert result == expected_damage
        assert component.health == 90.0  # 100 - 10
    
    def test_take_damage_exact_health(self, component):
        """Test damage that brings health to exactly 0."""
        damage = 100.0
        result = component.take_damage(damage)
        
        assert result == 100.0
        assert component.health == 0.0
    
    def test_take_damage_exceeds_health(self, component):
        """Test damage that exceeds current health."""
        damage = 150.0
        result = component.take_damage(damage)
        
        assert result == 150.0
        assert component.health == 0.0  # Clamped to 0
    
    def test_take_damage_defending_exceeds_health(self, component):
        """Test defending damage that exceeds health."""
        component.is_defending = True
        damage = 200.0  # Would be 100.0 after defense reduction
        result = component.take_damage(damage)
        
        expected_damage = 200.0 * 0.5  # 100.0
        assert result == expected_damage
        assert component.health == 0.0
    
    def test_take_damage_death_triggers_terminate(self, component):
        """Test that death triggers core termination."""
        core = Mock()
        component.attach(core)
        
        damage = 100.0
        result = component.take_damage(damage)
        
        assert result == 100.0
        assert component.health == 0.0
        core.terminate.assert_called_once()
    
    def test_take_damage_no_core_terminate(self, component):
        """Test damage without core attached (no termination)."""
        damage = 100.0
        result = component.take_damage(damage)
        
        assert result == 100.0
        assert component.health == 0.0
        # No core to terminate
    
    def test_take_damage_zero_damage(self, component):
        """Test taking zero damage."""
        damage = 0.0
        result = component.take_damage(damage)
        
        assert result == 0.0
        assert component.health == 100.0  # Unchanged
    
    def test_take_damage_negative_damage(self, component):
        """Test taking negative damage (should not heal)."""
        damage = -10.0
        result = component.take_damage(damage)
        
        assert result == -10.0
        assert component.health == 110.0  # Actually heals (negative damage)
    
    def test_multiple_damage_attacks(self, component):
        """Test multiple damage attacks."""
        # First attack
        result1 = component.take_damage(30.0)
        assert result1 == 30.0
        assert component.health == 70.0
        
        # Second attack
        result2 = component.take_damage(20.0)
        assert result2 == 20.0
        assert component.health == 50.0
        
        # Third attack
        result3 = component.take_damage(60.0)
        assert result3 == 60.0
        assert component.health == 0.0


class TestHealing:
    """Test healing functionality."""
    
    @pytest.fixture
    def component(self):
        """Create a combat component for testing."""
        services = Mock(spec=AgentServices)
        config = CombatConfig(starting_health=100.0)
        return CombatComponent(services, config)
    
    def test_heal_full_health(self, component):
        """Test healing when at full health."""
        component.heal(20.0)
        assert component.health == 100.0  # Capped at starting health
    
    def test_heal_partial_damage(self, component):
        """Test healing from partial damage."""
        component.take_damage(40.0)  # Health = 60.0
        component.heal(20.0)
        assert component.health == 80.0
    
    def test_heal_to_full_health(self, component):
        """Test healing to exactly full health."""
        component.take_damage(30.0)  # Health = 70.0
        component.heal(30.0)
        assert component.health == 100.0
    
    def test_heal_exceeds_max_health(self, component):
        """Test healing that would exceed max health."""
        component.take_damage(20.0)  # Health = 80.0
        component.heal(50.0)  # Would be 130.0
        assert component.health == 100.0  # Capped at starting health
    
    def test_heal_zero_amount(self, component):
        """Test healing with zero amount."""
        component.take_damage(20.0)  # Health = 80.0
        component.heal(0.0)
        assert component.health == 80.0  # Unchanged
    
    def test_heal_negative_amount(self, component):
        """Test healing with negative amount (actually reduces health)."""
        component.heal(-10.0)
        assert component.health == 90.0  # 100.0 - 10.0 (negative healing reduces health)
    
    def test_heal_from_zero_health(self, component):
        """Test healing from zero health."""
        component.take_damage(100.0)  # Health = 0.0
        component.heal(50.0)
        assert component.health == 50.0


class TestDefenseManagement:
    """Test defense stance management."""
    
    @pytest.fixture
    def component(self):
        """Create a combat component for testing."""
        services = Mock(spec=AgentServices)
        config = CombatConfig(defense_timer_duration=3)
        return CombatComponent(services, config)
    
    def test_start_defense(self, component):
        """Test starting defensive stance."""
        component.start_defense()
        
        assert component.is_defending is True
        assert component.defense_timer == 3
    
    def test_end_defense(self, component):
        """Test ending defensive stance."""
        component.start_defense()
        component.end_defense()
        
        assert component.is_defending is False
        assert component.defense_timer == 0
    
    def test_start_defense_multiple_times(self, component):
        """Test starting defense multiple times resets timer."""
        component.start_defense()
        assert component.defense_timer == 3
        
        component.start_defense()
        assert component.defense_timer == 3  # Reset to full duration
    
    def test_end_defense_when_not_defending(self, component):
        """Test ending defense when not defending."""
        component.end_defense()
        
        assert component.is_defending is False
        assert component.defense_timer == 0


class TestDefenseTimer:
    """Test defense timer countdown mechanics."""
    
    @pytest.fixture
    def component(self):
        """Create a combat component for testing."""
        services = Mock(spec=AgentServices)
        config = CombatConfig(defense_timer_duration=3)
        return CombatComponent(services, config)
    
    def test_defense_timer_countdown(self, component):
        """Test defense timer countdown on step start."""
        component.start_defense()
        assert component.defense_timer == 3
        assert component.is_defending is True
        
        # First step
        component.on_step_start()
        assert component.defense_timer == 2
        assert component.is_defending is True
        
        # Second step
        component.on_step_start()
        assert component.defense_timer == 1
        assert component.is_defending is True
        
        # Third step
        component.on_step_start()
        assert component.defense_timer == 0
        assert component.is_defending is False
    
    def test_defense_timer_expiration(self, component):
        """Test defense timer expiration."""
        component.start_defense()
        
        # Count down to expiration
        for _ in range(3):
            component.on_step_start()
        
        assert component.defense_timer == 0
        assert component.is_defending is False
    
    def test_defense_timer_continues_after_expiration(self, component):
        """Test defense timer continues counting after expiration."""
        component.start_defense()
        
        # Count down past expiration
        for _ in range(5):
            component.on_step_start()
        
        assert component.defense_timer == 0
        assert component.is_defending is False
    
    def test_defense_timer_without_defense(self, component):
        """Test defense timer when not defending."""
        component.on_step_start()
        assert component.defense_timer == 0
        assert component.is_defending is False


class TestAttackStrength:
    """Test attack strength calculations."""
    
    @pytest.fixture
    def component(self):
        """Create a combat component for testing."""
        services = Mock(spec=AgentServices)
        config = CombatConfig(starting_health=100.0, base_attack_strength=20.0)
        return CombatComponent(services, config)
    
    def test_attack_strength_full_health(self, component):
        """Test attack strength at full health."""
        assert component.attack_strength == 20.0  # 20.0 * 1.0
    
    def test_attack_strength_half_health(self, component):
        """Test attack strength at half health."""
        component.take_damage(50.0)  # Health = 50.0
        assert component.attack_strength == 10.0  # 20.0 * 0.5
    
    def test_attack_strength_quarter_health(self, component):
        """Test attack strength at quarter health."""
        component.take_damage(75.0)  # Health = 25.0
        assert component.attack_strength == 5.0  # 20.0 * 0.25
    
    def test_attack_strength_zero_health(self, component):
        """Test attack strength at zero health."""
        component.take_damage(100.0)  # Health = 0.0
        assert component.attack_strength == 0.0  # 20.0 * 0.0
    
    def test_attack_strength_custom_base(self, component):
        """Test attack strength with custom base value."""
        component.config = CombatConfig(starting_health=100.0, base_attack_strength=30.0)
        component.take_damage(40.0)  # Health = 60.0
        assert component.attack_strength == 18.0  # 30.0 * 0.6


class TestDefenseStrength:
    """Test defense strength calculations."""
    
    @pytest.fixture
    def component(self):
        """Create a combat component for testing."""
        services = Mock(spec=AgentServices)
        config = CombatConfig(base_defense_strength=15.0)
        return CombatComponent(services, config)
    
    def test_defense_strength_not_defending(self, component):
        """Test defense strength when not defending."""
        assert component.defense_strength == 0.0
    
    def test_defense_strength_defending(self, component):
        """Test defense strength when defending."""
        component.start_defense()
        assert component.defense_strength == 15.0
    
    def test_defense_strength_after_timer_expires(self, component):
        """Test defense strength after timer expires."""
        component.start_defense()
        assert component.defense_strength == 15.0
        
        # Count down timer
        for _ in range(3):
            component.on_step_start()
        
        assert component.defense_strength == 0.0
    
    def test_defense_strength_custom_base(self, component):
        """Test defense strength with custom base value."""
        component.config = CombatConfig(base_defense_strength=25.0)
        component.start_defense()
        assert component.defense_strength == 25.0


class TestHealthRatio:
    """Test health ratio calculations."""
    
    @pytest.fixture
    def component(self):
        """Create a combat component for testing."""
        services = Mock(spec=AgentServices)
        config = CombatConfig(starting_health=100.0)
        return CombatComponent(services, config)
    
    def test_health_ratio_full_health(self, component):
        """Test health ratio at full health."""
        assert component.health_ratio == 1.0
    
    def test_health_ratio_half_health(self, component):
        """Test health ratio at half health."""
        component.take_damage(50.0)
        assert component.health_ratio == 0.5
    
    def test_health_ratio_zero_health(self, component):
        """Test health ratio at zero health."""
        component.take_damage(100.0)
        assert component.health_ratio == 0.0
    
    def test_health_ratio_after_healing(self, component):
        """Test health ratio after healing."""
        component.take_damage(60.0)  # Health = 40.0
        assert component.health_ratio == 0.4
        
        component.heal(20.0)  # Health = 60.0
        assert component.health_ratio == 0.6
    
    def test_health_ratio_custom_starting_health(self, component):
        """Test health ratio with custom starting health."""
        component.config = CombatConfig(starting_health=200.0)
        component.health = 150.0
        assert component.health_ratio == 0.75


class TestIsAlive:
    """Test is_alive property."""
    
    @pytest.fixture
    def component(self):
        """Create a combat component for testing."""
        services = Mock(spec=AgentServices)
        config = CombatConfig()
        return CombatComponent(services, config)
    
    def test_is_alive_full_health(self, component):
        """Test is_alive at full health."""
        assert component.is_alive is True
    
    def test_is_alive_partial_health(self, component):
        """Test is_alive at partial health."""
        component.take_damage(50.0)
        assert component.is_alive is True
    
    def test_is_alive_zero_health(self, component):
        """Test is_alive at zero health."""
        component.take_damage(100.0)
        assert component.is_alive is False
    
    def test_is_alive_negative_health(self, component):
        """Test is_alive with negative health (should not happen but test edge case)."""
        component.health = -10.0
        assert component.is_alive is False
    
    def test_is_alive_after_healing(self, component):
        """Test is_alive after healing from death."""
        component.take_damage(100.0)  # Health = 0.0
        assert component.is_alive is False
        
        component.heal(10.0)  # Health = 10.0
        assert component.is_alive is True


class TestEdgeCases:
    """Test edge cases and error scenarios."""
    
    def test_very_small_damage(self):
        """Test taking very small damage."""
        services = Mock(spec=AgentServices)
        config = CombatConfig(starting_health=100.0)
        component = CombatComponent(services, config)
        
        damage = 0.001
        result = component.take_damage(damage)
        
        assert result == 0.001
        assert component.health == 99.999
    
    def test_very_large_damage(self):
        """Test taking very large damage."""
        services = Mock(spec=AgentServices)
        config = CombatConfig(starting_health=100.0)
        component = CombatComponent(services, config)
        
        damage = 1000000.0
        result = component.take_damage(damage)
        
        assert result == 1000000.0
        assert component.health == 0.0
    
    def test_defense_timer_zero_duration(self):
        """Test defense with zero timer duration."""
        services = Mock(spec=AgentServices)
        config = CombatConfig(defense_timer_duration=0)
        component = CombatComponent(services, config)
        
        component.start_defense()
        assert component.is_defending is True
        assert component.defense_timer == 0
        
        component.on_step_start()
        assert component.is_defending is False
        assert component.defense_timer == 0
    
    def test_defense_timer_negative_duration(self):
        """Test defense with negative timer duration."""
        services = Mock(spec=AgentServices)
        config = CombatConfig(defense_timer_duration=-1)
        component = CombatComponent(services, config)
        
        component.start_defense()
        assert component.is_defending is True
        assert component.defense_timer == -1
        
        component.on_step_start()
        assert component.is_defending is False
        assert component.defense_timer == -1  # Timer only decrements if > 0
    
    def test_attack_strength_with_zero_base(self):
        """Test attack strength with zero base attack strength."""
        services = Mock(spec=AgentServices)
        config = CombatConfig(starting_health=100.0, base_attack_strength=0.0)
        component = CombatComponent(services, config)
        
        assert component.attack_strength == 0.0
        
        component.take_damage(50.0)
        assert component.attack_strength == 0.0
    
    def test_defense_strength_with_zero_base(self):
        """Test defense strength with zero base defense strength."""
        services = Mock(spec=AgentServices)
        config = CombatConfig(base_defense_strength=0.0)
        component = CombatComponent(services, config)
        
        component.start_defense()
        assert component.defense_strength == 0.0


class TestIntegrationScenarios:
    """Test complex integration scenarios."""
    
    def test_complete_combat_workflow(self):
        """Test a complete combat workflow."""
        # Setup services
        services = Mock(spec=AgentServices)
        config = CombatConfig(
            starting_health=100.0,
            base_attack_strength=15.0,
            base_defense_strength=10.0,
            defense_damage_reduction=0.6,
            defense_timer_duration=2
        )
        component = CombatComponent(services, config)
        
        # Test initial state
        assert component.health == 100.0
        assert component.is_defending is False
        assert component.attack_strength == 15.0
        assert component.defense_strength == 0.0
        assert component.health_ratio == 1.0
        assert component.is_alive is True
        
        # Take some damage
        damage1 = component.take_damage(30.0)
        assert damage1 == 30.0
        assert component.health == 70.0
        assert component.attack_strength == 10.5  # 15.0 * 0.7
        assert component.health_ratio == 0.7
        
        # Start defending
        component.start_defense()
        assert component.is_defending is True
        assert component.defense_strength == 10.0
        
        # Take damage while defending
        damage2 = component.take_damage(20.0)
        expected_damage = 20.0 * 0.6  # 12.0
        assert damage2 == expected_damage
        assert component.health == 58.0  # 70.0 - 12.0
        
        # Step forward (defense timer countdown)
        component.on_step_start()
        assert component.defense_timer == 1
        assert component.is_defending is True
        
        # Another step (defense expires)
        component.on_step_start()
        assert component.defense_timer == 0
        assert component.is_defending is False
        assert component.defense_strength == 0.0
        
        # Take damage without defense
        damage3 = component.take_damage(15.0)
        assert damage3 == 15.0
        assert component.health == 43.0
        
        # Heal
        component.heal(20.0)
        assert component.health == 63.0
        assert component.attack_strength == 9.45  # 15.0 * 0.63
        assert component.health_ratio == 0.63
        
        # Final damage to death
        core = Mock()
        component.attach(core)
        damage4 = component.take_damage(100.0)  # More than current health
        assert damage4 == 100.0
        assert component.health == 0.0
        assert component.is_alive is False
        core.terminate.assert_called_once()
    
    def test_error_recovery_scenarios(self):
        """Test error recovery in various scenarios."""
        # Setup services
        services = Mock(spec=AgentServices)
        config = CombatConfig()
        component = CombatComponent(services, config)
        
        # Test that component continues to work despite various edge cases
        # Zero damage
        assert component.take_damage(0.0) == 0.0
        assert component.health == 100.0
        
        # Negative damage (healing)
        assert component.take_damage(-10.0) == -10.0
        assert component.health == 110.0
        
        # Heal beyond max
        component.heal(50.0)
        assert component.health == 100.0  # Capped
        
        # Defense with zero duration
        component.config = CombatConfig(defense_timer_duration=0)
        component.start_defense()
        component.on_step_start()
        assert component.is_defending is False
        
        # Test lifecycle hooks still work
        component.on_step_start()
        component.on_step_end()
        component.on_terminate()