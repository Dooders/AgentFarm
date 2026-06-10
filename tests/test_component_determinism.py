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


def create_mock_db():
    """
    Create a mock simulation database whose genome-id lookups report "not found".

    A bare ``Mock()`` returns truthy Mock objects from ``_execute_in_transaction``,
    which sends ``Identity.genome_id``'s uniqueness loop into infinite counter
    probing (and unbounded mock-call recording) once an agent successfully
    reproduces.
    """
    mock_db = Mock()
    mock_db._execute_in_transaction.return_value = False
    return mock_db


def run_seeded_trajectory(seed, create_agent, steps, record, prepare=None):
    """
    Seed all RNGs once, create an agent, and step it while recording observations.

    Seeding only once per trajectory (instead of before every step) ensures the
    test exercises sequential RNG consumption - the way randomness is actually
    used in a simulation run.

    Parameters
    ----------
    seed : int
        Seed applied once before agent creation
    create_agent : callable
        Factory receiving the seed and returning the agent under test
    steps : int
        Number of steps to execute
    record : callable
        Function mapping the agent to the per-step observation to record
    prepare : callable, optional
        Hook invoked with the agent after creation, before stepping

    Returns
    -------
    list
        Per-step observations
    """
    seed_all_rngs(seed)
    agent = create_agent(seed)
    if prepare is not None:
        prepare(agent)
    observations = []
    for _ in range(steps):
        agent.step()
        observations.append(record(agent))
    return observations


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
        
        # Seed once per run so sequential RNG consumption is actually exercised;
        # re-seeding before every draw would just repeat the same draw 20 times.
        seed_all_rngs(deterministic_seed)
        behavior1 = DefaultAgentBehavior()
        actions1 = [behavior1.decide_action(mock_core, None, None).name for _ in range(20)]

        seed_all_rngs(deterministic_seed)
        behavior2 = DefaultAgentBehavior()
        actions2 = [behavior2.decide_action(mock_core, None, None).name for _ in range(20)]

        # Action sequences should be identical (weighted selection is deterministic with same seed)
        assert actions1 == actions2, "DefaultAgentBehavior action selection is not deterministic"
        # Sanity check: the sequence should contain multiple distinct actions, otherwise
        # this test would pass trivially for a constant-action behavior.
        assert len(set(actions1)) > 1, "Expected a varied action sequence from weighted selection"
    
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
        
        # Seed once per run so sequential RNG consumption is actually exercised.
        seed_all_rngs(deterministic_seed)
        behavior1 = DefaultAgentBehavior()
        for _ in range(10):
            behavior1.decide_action(None, None, actions)

        seed_all_rngs(deterministic_seed)
        behavior2 = DefaultAgentBehavior()
        for _ in range(10):
            behavior2.decide_action(None, None, actions)

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
            mock_setup_db.return_value = create_mock_db()
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
        def record(agent):
            resource = agent.get_component("resource")
            return (resource.level, resource.starvation_counter)

        series1 = run_seeded_trajectory(deterministic_seed, self.create_test_agent, 10, record)
        series2 = run_seeded_trajectory(deterministic_seed, self.create_test_agent, 10, record)

        assert series1 == series2, "Resource consumption should be deterministic"

    def test_starvation_counter_determinism(self, deterministic_seed):
        """Test that starvation counter updates are deterministic."""
        def set_low_resources(agent):
            agent.get_component("resource").level = 1.0

        def record(agent):
            return agent.get_component("resource").starvation_counter

        series1 = run_seeded_trajectory(
            deterministic_seed, self.create_test_agent, 15, record, prepare=set_low_resources
        )
        series2 = run_seeded_trajectory(
            deterministic_seed, self.create_test_agent, 15, record, prepare=set_low_resources
        )

        assert series1 == series2, "Starvation counter updates should be deterministic"


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
            mock_setup_db.return_value = create_mock_db()
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
        def record(agent):
            return agent.get_component("reproduction").offspring_created

        series1 = run_seeded_trajectory(deterministic_seed, self.create_test_agent, 20, record)
        series2 = run_seeded_trajectory(deterministic_seed, self.create_test_agent, 20, record)

        assert series1 == series2, "Offspring creation should be deterministic"

    def test_reproduction_cooldown_determinism(self, deterministic_seed):
        """Test that reproduction cooldown is deterministic."""
        def record(agent):
            return getattr(agent.get_component("reproduction"), "reproduction_cooldown", 0)

        series1 = run_seeded_trajectory(deterministic_seed, self.create_test_agent, 15, record)
        series2 = run_seeded_trajectory(deterministic_seed, self.create_test_agent, 15, record)

        assert series1 == series2, "Reproduction cooldown should be deterministic"


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
            mock_setup_db.return_value = create_mock_db()
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
        positions1 = run_seeded_trajectory(deterministic_seed, self.create_test_agent, 15, lambda a: a.position)
        positions2 = run_seeded_trajectory(deterministic_seed, self.create_test_agent, 15, lambda a: a.position)

        assert positions1 == positions2, "Movement should be deterministic"

    def test_movement_component_state_determinism(self, deterministic_seed):
        """Test that movement component internal state is deterministic."""
        def record(agent):
            return capture_component_state(agent.get_component("movement"))

        states1 = run_seeded_trajectory(deterministic_seed, self.create_test_agent, 10, record)
        states2 = run_seeded_trajectory(deterministic_seed, self.create_test_agent, 10, record)

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
            mock_setup_db.return_value = create_mock_db()
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
        def record(agent):
            return {
                "position": agent.position,
                "resource_level": agent.resource_level,
                "alive": agent.alive,
                "components": {
                    name: capture_component_state(component)
                    for name, component in agent._components.items()
                },
            }

        states1 = run_seeded_trajectory(deterministic_seed, self.create_full_agent, 20, record)
        states2 = run_seeded_trajectory(deterministic_seed, self.create_full_agent, 20, record)

        assert states1 == states2, "Multi-component integration should be deterministic"


if __name__ == "__main__":
    # Allow running as standalone script for debugging
    pytest.main([__file__, "-v", "-m", "determinism"])
