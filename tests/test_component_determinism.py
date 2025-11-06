#!/usr/bin/env python3
"""
Component-level deterministic testing for AgentFarm simulation.

This module tests deterministic behavior of individual agent components that use
randomness, focusing on high-risk components: Movement, Resource, Reproduction,
and DefaultAgentBehavior.
"""

import pytest
from unittest.mock import patch, Mock

from farm.core.agent import AgentCore, DefaultAgentBehavior
from farm.core.agent.components import (
    CombatComponent,
    MovementComponent,
    PerceptionComponent,
    ReproductionComponent,
    ResourceComponent,
)
from farm.core.agent.config.component_configs import (
    AgentComponentConfig,
    CombatConfig,
    MovementConfig,
    PerceptionConfig,
    ReproductionConfig,
    ResourceConfig as ComponentResourceConfig,
)
from farm.core.environment import Environment
from farm.config import SimulationConfig
from tests.conftest import seed_all_rngs, capture_component_state


@pytest.mark.determinism
class TestDefaultAgentBehaviorDeterminism:
    """Test deterministic behavior of DefaultAgentBehavior."""
    
    def test_action_selection_determinism(self, deterministic_seed):
        """Test that weighted random action selection is deterministic with same seed."""
        from farm.core.action import Action
        from unittest.mock import Mock
        
        # Create Action objects with weights (DefaultAgentBehavior now uses weighted selection)
        actions = [
            Action("defend", 0.25, Mock()),
            Action("attack", 0.1, Mock()),
            Action("gather", 0.3, Mock()),
            Action("share", 0.2, Mock()),
            Action("move", 0.4, Mock()),
            Action("reproduce", 0.15, Mock()),
            Action("pass", 0.05, Mock()),
        ]
        
        # Create mock core with actions
        mock_core = Mock()
        mock_core.actions = actions
        
        # Test with same seed multiple times
        seed_all_rngs(deterministic_seed)
        behavior1 = DefaultAgentBehavior()
        
        seed_all_rngs(deterministic_seed)
        behavior2 = DefaultAgentBehavior()
        
        # Generate action sequences
        actions1 = []
        actions2 = []
        
        for _ in range(20):
            seed_all_rngs(deterministic_seed)
            action1 = behavior1.decide_action(mock_core, None, None)
            actions1.append(action1.name)
            
            seed_all_rngs(deterministic_seed)
            action2 = behavior2.decide_action(mock_core, None, None)
            actions2.append(action2.name)
        
        # Action sequences should be identical (weighted selection is deterministic with same seed)
        assert actions1 == actions2, "DefaultAgentBehavior action selection is not deterministic"
    
    def test_action_history_consistency(self, deterministic_seed):
        """Test that action history is consistent across runs."""
        from farm.core.action import Action
        from unittest.mock import Mock
        
        # Create Action objects with weights (matching ActionType names)
        actions = [
            Action("move", 0.4, Mock()),
            Action("gather", 0.3, Mock()),
            Action("attack", 0.3, Mock()),
        ]
        
        # Create two behaviors with same seed
        seed_all_rngs(deterministic_seed)
        behavior1 = DefaultAgentBehavior()
        
        seed_all_rngs(deterministic_seed)
        behavior2 = DefaultAgentBehavior()
        
        # Execute same sequence of actions
        for _ in range(10):
            seed_all_rngs(deterministic_seed)
            action1 = behavior1.decide_action(None, None, actions)
            
            seed_all_rngs(deterministic_seed)
            action2 = behavior2.decide_action(None, None, actions)
            
            assert action1 == action2, "Actions should be identical with same seed"
        
        # Action histories should be identical
        assert behavior1.action_history == behavior2.action_history, "Action histories should be identical"


@pytest.mark.determinism
class TestResourceComponentDeterminism:
    """Test deterministic behavior of ResourceComponent."""
    
    def create_test_agent(self, seed: int) -> AgentCore:
        """Create a test agent with ResourceComponent."""
        # Create mock environment
        from farm.config.config import EnvironmentConfig
        config = SimulationConfig(
            environment=EnvironmentConfig(width=50, height=50),
            seed=seed
        )
        with patch("farm.database.utilities.setup_db") as mock_setup_db:
            mock_setup_db.return_value = Mock()
            env = Environment(width=50, height=50, resource_distribution={"amount": 10}, config=config)
            
            # Create services from environment
            from farm.core.simulation import create_services_from_environment
            services = create_services_from_environment(env)
            
            # Create resource component
            resource_config = ComponentResourceConfig()
            resource_component = ResourceComponent(services, resource_config)
            
            # Create agent
            agent = AgentCore(
                agent_id="test_agent",
                position=(25, 25),
                services=services,
                behavior=DefaultAgentBehavior(),
                components=[resource_component],
                config=AgentComponentConfig(),
                environment=env,
                initial_resources=10.0,
            )
            
            # Attach component
            resource_component.attach(agent)
            
            # Initialize resource level
            resource_component.level = 10.0
            
            return agent
    
    def test_resource_consumption_determinism(self, deterministic_seed):
        """Test that resource consumption is deterministic."""
        # Create two agents with same seed
        seed_all_rngs(deterministic_seed)
        agent1 = self.create_test_agent(deterministic_seed)
        
        seed_all_rngs(deterministic_seed)
        agent2 = self.create_test_agent(deterministic_seed)
        
        # Execute same number of steps
        for _ in range(10):
            seed_all_rngs(deterministic_seed)
            agent1.step()
            
            seed_all_rngs(deterministic_seed)
            agent2.step()
        
        # Resource levels should be identical
        resource_comp1 = agent1.get_component("resource")
        resource_comp2 = agent2.get_component("resource")
        
        assert resource_comp1.level == resource_comp2.level, "Resource levels should be identical"
        assert resource_comp1.starvation_counter == resource_comp2.starvation_counter, "Starvation counters should be identical"
    
    def test_starvation_counter_determinism(self, deterministic_seed):
        """Test that starvation counter updates are deterministic."""
        # Create agents with low initial resources
        seed_all_rngs(deterministic_seed)
        agent1 = self.create_test_agent(deterministic_seed)
        agent1.get_component("resource").level = 1.0  # Low resources
        
        seed_all_rngs(deterministic_seed)
        agent2 = self.create_test_agent(deterministic_seed)
        agent2.get_component("resource").level = 1.0  # Low resources
        
        # Execute steps and track starvation counter
        starvation_counts1 = []
        starvation_counts2 = []
        
        for _ in range(15):
            seed_all_rngs(deterministic_seed)
            agent1.step()
            starvation_counts1.append(agent1.get_component("resource").starvation_counter)
            
            seed_all_rngs(deterministic_seed)
            agent2.step()
            starvation_counts2.append(agent2.get_component("resource").starvation_counter)
        
        # Starvation counters should be identical
        assert starvation_counts1 == starvation_counts2, "Starvation counter updates should be deterministic"


@pytest.mark.determinism
class TestReproductionComponentDeterminism:
    """Test deterministic behavior of ReproductionComponent."""
    
    def create_test_agent(self, seed: int) -> AgentCore:
        """Create a test agent with ReproductionComponent."""
        # Create mock environment
        from farm.config.config import EnvironmentConfig
        config = SimulationConfig(
            environment=EnvironmentConfig(width=50, height=50),
            seed=seed
        )
        with patch("farm.database.utilities.setup_db") as mock_setup_db:
            mock_setup_db.return_value = Mock()
            env = Environment(width=50, height=50, resource_distribution={"amount": 10}, config=config)
            
            # Create services from environment
            from farm.core.simulation import create_services_from_environment
            services = create_services_from_environment(env)
            
            # Create reproduction component
            reproduction_config = ReproductionConfig()
            reproduction_component = ReproductionComponent(services, reproduction_config)
            
            # Create agent
            agent = AgentCore(
                agent_id="test_agent",
                position=(25, 25),
                services=services,
                behavior=DefaultAgentBehavior(),
                components=[reproduction_component],
                config=AgentComponentConfig(),
                environment=env,
                initial_resources=20.0,  # High resources for reproduction
            )
            
            # Attach component
            reproduction_component.attach(agent)
            
            return agent
    
    def test_offspring_creation_determinism(self, deterministic_seed):
        """Test that offspring creation is deterministic."""
        # Create two agents with same seed
        seed_all_rngs(deterministic_seed)
        agent1 = self.create_test_agent(deterministic_seed)
        
        seed_all_rngs(deterministic_seed)
        agent2 = self.create_test_agent(deterministic_seed)
        
        # Execute same number of steps
        offspring_counts1 = []
        offspring_counts2 = []
        
        for _ in range(20):
            seed_all_rngs(deterministic_seed)
            agent1.step()
            offspring_counts1.append(agent1.get_component("reproduction").offspring_created)
            
            seed_all_rngs(deterministic_seed)
            agent2.step()
            offspring_counts2.append(agent2.get_component("reproduction").offspring_created)
        
        # Offspring counts should be identical
        assert offspring_counts1 == offspring_counts2, "Offspring creation should be deterministic"
    
    def test_reproduction_cooldown_determinism(self, deterministic_seed):
        """Test that reproduction cooldown is deterministic."""
        # Create agents
        seed_all_rngs(deterministic_seed)
        agent1 = self.create_test_agent(deterministic_seed)
        
        seed_all_rngs(deterministic_seed)
        agent2 = self.create_test_agent(deterministic_seed)
        
        # Track cooldown values
        cooldowns1 = []
        cooldowns2 = []
        
        for _ in range(15):
            seed_all_rngs(deterministic_seed)
            agent1.step()
            cooldowns1.append(getattr(agent1.get_component("reproduction"), "reproduction_cooldown", 0))
            
            seed_all_rngs(deterministic_seed)
            agent2.step()
            cooldowns2.append(getattr(agent2.get_component("reproduction"), "reproduction_cooldown", 0))
        
        # Cooldown values should be identical
        assert cooldowns1 == cooldowns2, "Reproduction cooldown should be deterministic"


@pytest.mark.determinism
class TestMovementComponentDeterminism:
    """Test deterministic behavior of MovementComponent."""
    
    def create_test_agent(self, seed: int) -> AgentCore:
        """Create a test agent with MovementComponent."""
        # Create mock environment
        from farm.config.config import EnvironmentConfig
        config = SimulationConfig(
            environment=EnvironmentConfig(width=50, height=50),
            seed=seed
        )
        with patch("farm.database.utilities.setup_db") as mock_setup_db:
            mock_setup_db.return_value = Mock()
            env = Environment(width=50, height=50, resource_distribution={"amount": 10}, config=config)
            
            # Create services from environment
            from farm.core.simulation import create_services_from_environment
            services = create_services_from_environment(env)
            
            # Create movement component
            movement_config = MovementConfig()
            movement_component = MovementComponent(services, movement_config)
            
            # Create agent
            agent = AgentCore(
                agent_id="test_agent",
                position=(25, 25),
                services=services,
                behavior=DefaultAgentBehavior(),
                components=[movement_component],
                config=AgentComponentConfig(),
                environment=env,
                initial_resources=10.0,
            )
            
            # Attach component
            movement_component.attach(agent)
            
            return agent
    
    def test_movement_target_selection_determinism(self, deterministic_seed):
        """Test that movement target selection is deterministic."""
        # Create two agents with same seed
        seed_all_rngs(deterministic_seed)
        agent1 = self.create_test_agent(deterministic_seed)
        
        seed_all_rngs(deterministic_seed)
        agent2 = self.create_test_agent(deterministic_seed)
        
        # Track positions over time
        positions1 = []
        positions2 = []
        
        for _ in range(15):
            seed_all_rngs(deterministic_seed)
            agent1.step()
            positions1.append(agent1.position)
            
            seed_all_rngs(deterministic_seed)
            agent2.step()
            positions2.append(agent2.position)
        
        # Positions should be identical
        assert positions1 == positions2, "Movement should be deterministic"
    
    def test_movement_component_state_determinism(self, deterministic_seed):
        """Test that movement component internal state is deterministic."""
        # Create agents
        seed_all_rngs(deterministic_seed)
        agent1 = self.create_test_agent(deterministic_seed)
        
        seed_all_rngs(deterministic_seed)
        agent2 = self.create_test_agent(deterministic_seed)
        
        # Execute steps and capture component states
        states1 = []
        states2 = []
        
        for _ in range(10):
            seed_all_rngs(deterministic_seed)
            agent1.step()
            states1.append(capture_component_state(agent1.get_component("movement")))
            
            seed_all_rngs(deterministic_seed)
            agent2.step()
            states2.append(capture_component_state(agent2.get_component("movement")))
        
        # Component states should be identical
        assert states1 == states2, "Movement component states should be deterministic"


@pytest.mark.determinism
class TestComponentIntegrationDeterminism:
    """Test deterministic behavior when multiple components interact."""
    
    def create_full_agent(self, seed: int) -> AgentCore:
        """Create a test agent with all components."""
        # Create mock environment
        from farm.config.config import EnvironmentConfig
        config = SimulationConfig(
            environment=EnvironmentConfig(width=50, height=50),
            seed=seed
        )
        with patch("farm.database.utilities.setup_db") as mock_setup_db:
            mock_setup_db.return_value = Mock()
            env = Environment(width=50, height=50, resource_distribution={"amount": 10}, config=config)
            
            # Create services from environment
            from farm.core.simulation import create_services_from_environment
            services = create_services_from_environment(env)
            
            # Create all components
            components = [
                MovementComponent(services, MovementConfig()),
                ResourceComponent(services, ComponentResourceConfig()),
                CombatComponent(services, CombatConfig()),
                PerceptionComponent(services, PerceptionConfig()),
                ReproductionComponent(services, ReproductionConfig()),
            ]
            
            # Create agent
            agent = AgentCore(
                agent_id="test_agent",
                position=(25, 25),
                services=services,
                behavior=DefaultAgentBehavior(),
                components=components,
                config=AgentComponentConfig(),
                environment=env,
                initial_resources=15.0,
            )
            
            # Attach all components
            for component in components:
                component.attach(agent)
            
            return agent
    
    def test_multi_component_determinism(self, deterministic_seed):
        """Test that multiple components work deterministically together."""
        # Create two agents with same seed
        seed_all_rngs(deterministic_seed)
        agent1 = self.create_full_agent(deterministic_seed)
        
        seed_all_rngs(deterministic_seed)
        agent2 = self.create_full_agent(deterministic_seed)
        
        # Track comprehensive state over time
        states1 = []
        states2 = []
        
        for step in range(20):
            seed_all_rngs(deterministic_seed)
            agent1.step()
            
            # Capture comprehensive state
            state1 = {
                "position": agent1.position,
                "resource_level": agent1.resource_level,
                "alive": agent1.alive,
                "components": {}
            }
            
            for name, component in agent1._components.items():
                state1["components"][name] = capture_component_state(component)
            
            states1.append(state1)
            
            seed_all_rngs(deterministic_seed)
            agent2.step()
            
            # Capture comprehensive state
            state2 = {
                "position": agent2.position,
                "resource_level": agent2.resource_level,
                "alive": agent2.alive,
                "components": {}
            }
            
            for name, component in agent2._components.items():
                state2["components"][name] = capture_component_state(component)
            
            states2.append(state2)
        
        # States should be identical
        assert states1 == states2, "Multi-component integration should be deterministic"


if __name__ == "__main__":
    # Allow running as standalone script for debugging
    pytest.main([__file__, "-v", "-m", "determinism"])
