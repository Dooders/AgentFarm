#!/usr/bin/env python3
"""
Pytest-compatible deterministic testing suite for AgentFarm simulation.

This module provides comprehensive deterministic testing using pytest framework
with proper fixtures, markers, and parametrized tests for multi-seed validation.
"""

import os
import tempfile
from typing import Dict, List, Optional

import pytest

from farm.config import SimulationConfig
from farm.core.simulation import run_simulation
from tests.test_deterministic import get_simulation_state_hash, run_determinism_test


@pytest.mark.determinism
def test_determinism_single_seed(deterministic_seed):
    """Test basic determinism with a single seed."""
    result = run_determinism_test(
        environment="testing",
        num_steps=50,
        seed=deterministic_seed
    )
    assert result, f"Determinism test failed with seed {deterministic_seed}"


@pytest.mark.determinism
@pytest.mark.parametrize("seed", [42, 123, 456, 789, 999])
def test_determinism_multiple_seeds(seed):
    """Test determinism across multiple seeds."""
    result = run_determinism_test(
        environment="testing",
        num_steps=50,
        seed=seed
    )
    assert result, f"Determinism test failed with seed {seed}"


@pytest.mark.determinism
def test_determinism_step_by_step(deterministic_seed):
    """Test determinism at intermediate steps."""
    snapshot_steps = [10, 25, 50]
    result = run_determinism_test(
        environment="testing",
        num_steps=50,
        seed=deterministic_seed,
        use_snapshot_steps=snapshot_steps
    )
    assert result, f"Step-by-step determinism test failed with seed {deterministic_seed}"


@pytest.mark.determinism
@pytest.mark.parametrize("agent_count", [5, 10, 15])
def test_determinism_different_agent_counts(agent_count, deterministic_seed):
    """Test determinism with varying agent counts."""
    # Create custom config with specific agent count
    config = SimulationConfig.from_centralized_config(environment="testing")
    config.population.system_agents = agent_count // 3
    config.population.independent_agents = agent_count // 3
    config.population.control_agents = agent_count - 2 * (agent_count // 3)
    config.seed = deterministic_seed
    config.use_in_memory_db = True
    config.persist_db_on_completion = False
    
    # Run deterministic test with custom config
    result = run_deterministic_simulation_pair(config, deterministic_seed, 30)
    assert result, f"Determinism test failed with {agent_count} agents"


@pytest.mark.determinism
@pytest.mark.parametrize("world_size", [(50, 50), (75, 75), (100, 100)])
def test_determinism_different_world_sizes(world_size, deterministic_seed):
    """Test determinism with varying world sizes."""
    width, height = world_size
    
    # Create custom config with specific world size
    config = SimulationConfig.from_centralized_config(environment="testing")
    config.environment.width = width
    config.environment.height = height
    config.seed = deterministic_seed
    config.use_in_memory_db = True
    config.persist_db_on_completion = False
    
    # Run deterministic test with custom config
    result = run_deterministic_simulation_pair(config, deterministic_seed, 30)
    assert result, f"Determinism test failed with world size {world_size}"


@pytest.mark.determinism_regression
def test_determinism_regression_critical(deterministic_seed):
    """Critical regression test for determinism - must always pass."""
    result = run_determinism_test(
        environment="testing",
        num_steps=100,
        seed=deterministic_seed
    )
    assert result, f"Critical determinism regression test failed with seed {deterministic_seed}"


@pytest.mark.determinism
@pytest.mark.slow
def test_determinism_long_simulation(deterministic_seed):
    """Test determinism over longer simulation runs."""
    result = run_determinism_test(
        environment="testing",
        num_steps=200,
        seed=deterministic_seed
    )
    assert result, f"Long simulation determinism test failed with seed {deterministic_seed}"


def run_deterministic_simulation_pair(config: SimulationConfig, seed: int, steps: int) -> bool:
    """
    Run two identical simulations with the same seed and compare their results.
    
    Parameters
    ----------
    config : SimulationConfig
        Configuration for the simulation
    seed : int
        Seed value to use for both simulations
    steps : int
        Number of simulation steps to run
        
    Returns
    -------
    bool
        True if the simulations were deterministic, False otherwise
    """
    from tests.conftest import seed_all_rngs
    
    # Set fixed seeds for all random generators
    seed_all_rngs(seed)
    
    # Create temporary directories for simulations
    with tempfile.TemporaryDirectory() as temp_dir:
        sim_dir_1 = os.path.join(temp_dir, "sim_1")
        sim_dir_2 = os.path.join(temp_dir, "sim_2")
        os.makedirs(sim_dir_1, exist_ok=True)
        os.makedirs(sim_dir_2, exist_ok=True)
        
        # Reset seeds again to ensure same initial state
        seed_all_rngs(seed)
        
        # Run first simulation
        env1 = run_simulation(
            num_steps=steps,
            config=config,
            path=sim_dir_1,
            save_config=True,
            seed=seed
        )
        
        # Reset seeds again
        seed_all_rngs(seed)
        
        # Run second simulation
        env2 = run_simulation(
            num_steps=steps,
            config=config,
            path=sim_dir_2,
            save_config=True,
            seed=seed
        )
        
        # Compare final states
        hash1 = get_simulation_state_hash(env1, debug=False)
        hash2 = get_simulation_state_hash(env2, debug=False)
        
        return hash1 == hash2


@pytest.mark.determinism
def test_state_hash_consistency(deterministic_seed):
    """Test that state hash generation is consistent."""
    from tests.conftest import seed_all_rngs
    
    # Set seed
    seed_all_rngs(deterministic_seed)
    
    # Create config
    config = SimulationConfig.from_centralized_config(environment="testing")
    config.seed = deterministic_seed
    config.use_in_memory_db = True
    config.persist_db_on_completion = False
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Run simulation
        env = run_simulation(
            num_steps=30,
            config=config,
            path=temp_dir,
            save_config=True,
            seed=deterministic_seed
        )
        
        # Generate hash multiple times - should be identical
        hash1 = get_simulation_state_hash(env, debug=False)
        hash2 = get_simulation_state_hash(env, debug=False)
        hash3 = get_simulation_state_hash(env, debug=False)
        
        assert hash1 == hash2 == hash3, "State hash generation is not consistent"


@pytest.mark.determinism
def test_determinism_with_different_environments(deterministic_seed):
    """Test determinism across different configuration environments."""
    environments = ["testing", "development"]
    
    for env_name in environments:
        result = run_determinism_test(
            environment=env_name,
            num_steps=30,
            seed=deterministic_seed
        )
        assert result, f"Determinism test failed with environment {env_name}"


if __name__ == "__main__":
    # Allow running as standalone script for debugging
    pytest.main([__file__, "-v", "-m", "determinism"])
