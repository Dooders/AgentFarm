#!/usr/bin/env python3
"""
Script to test determinism in the AgentFarm simulation.
This will run two identical simulations and compare their outcomes to validate determinism.
"""

import argparse
import hashlib
import json
import numpy as np
import os
import random
import torch
from datetime import datetime

from farm.config import SimulationConfig
from farm.core.simulation import run_simulation


def get_simulation_state_hash(environment, debug=False):
    """
    Generate a hash of the current simulation state to check for equality.
    
    Parameters
    ----------
    environment : Environment
        The simulation environment to hash
    debug : bool, optional
        Whether to print detailed debug information, by default False
        
    Returns
    -------
    str
        Hash of the state as a hex string
    """
    # Dictionary to hold all relevant state
    state = {
        "time": environment.time,
        "agents": [],
        "resources": []
    }
    
    # Sort agents by ID for consistency
    for agent in sorted(environment._agent_objects.values(), key=lambda a: a.agent_id):
        agent_state = {
            "id": agent.agent_id,
            "type": agent.__class__.__name__,
            "position": agent.position,
            "resource_level": agent.get_component("resource").level,
            "alive": agent.alive,
            "generation": agent.state_manager.generation,
        }
        state["agents"].append(agent_state)
    
    # Sort resources by ID for consistency
    for resource in sorted(environment.resources, key=lambda r: r.resource_id):
        resource_state = {
            "id": resource.resource_id,
            "position": resource.position,
            "amount": resource.amount
        }
        state["resources"].append(resource_state)
    
    # Convert to JSON and hash it
    state_json = json.dumps(state, sort_keys=True)
    state_hash = hashlib.sha256(state_json.encode()).hexdigest()
    
    if debug:
        print(f"Time: {state['time']}")
        print(f"Number of agents: {len(state['agents'])}")
        print(f"Number of resources: {len(state['resources'])}")
        # Print the first agent and resource for debugging
        if state['agents']:
            print(f"First agent: {state['agents'][0]}")
        if state['resources']:
            print(f"First resource: {state['resources'][0]}")
    
    return state_hash, state if debug else state_hash


def compare_states(state1, state2):
    """
    Compare two simulation states and output the differences.
    
    Parameters
    ----------
    state1 : dict
        First simulation state
    state2 : dict
        Second simulation state
    """
    print("\nComparing simulation states:")
    
    # Check if time steps match
    if state1["time"] != state2["time"]:
        print(f"Time mismatch: {state1['time']} vs {state2['time']}")
    
    # Check agent counts
    if len(state1["agents"]) != len(state2["agents"]):
        print(f"Agent count mismatch: {len(state1['agents'])} vs {len(state2['agents'])}")
    
    # Check resource counts
    if len(state1["resources"]) != len(state2["resources"]):
        print(f"Resource count mismatch: {len(state1['resources'])} vs {len(state2['resources'])}")
    
    # Compare agents
    min_agents = min(len(state1["agents"]), len(state2["agents"]))
    for i in range(min_agents):
        agent1 = state1["agents"][i]
        agent2 = state2["agents"][i]
        
        if agent1 != agent2:
            print(f"Agent {i} mismatch:")
            # Find which fields differ
            for key in agent1:
                if key in agent2 and agent1[key] != agent2[key]:
                    print(f"  - {key}: {agent1[key]} vs {agent2[key]}")
    
    # Compare resources
    min_resources = min(len(state1["resources"]), len(state2["resources"]))
    resource_diffs = 0
    for i in range(min_resources):
        resource1 = state1["resources"][i]
        resource2 = state2["resources"][i]
        
        if resource1 != resource2:
            resource_diffs += 1
            if resource_diffs <= 3:  # Only show first few differences to avoid spam
                print(f"Resource {i} mismatch:")
                # Find which fields differ
                for key in resource1:
                    if key in resource2 and resource1[key] != resource2[key]:
                        print(f"  - {key}: {resource1[key]} vs {resource2[key]}")
    
    if resource_diffs > 3:
        print(f"... and {resource_diffs - 3} more resource differences")


def run_determinism_test(environment, num_steps, seed=42, use_snapshot_steps=None):
    """
    Run two identical simulations with the same seed and compare their results.
    
    Parameters
    ----------
    environment : str
        Configuration environment name to load
    num_steps : int
        Number of simulation steps to run
    seed : int
        Seed value to use for both simulations
    use_snapshot_steps : list, optional
        List of step numbers to take snapshots at for comparison
    
    Returns
    -------
    bool
        True if the simulations were deterministic, False otherwise
    """
    print(f"Starting determinism test with seed {seed}")
    
    # Set fixed seeds for all random generators
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Note: PyTorch CuDNN deterministic mode can be very slow and may not be necessary
    # for current simulation determinism. Uncomment if needed for debugging:
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Load configuration
    config = SimulationConfig.from_centralized_config(environment=environment)
    
    # Override any config parameters that might affect determinism
    config.seed = seed
    
    # Use in-memory database to avoid external state issues but allow persistence testing
    config.use_in_memory_db = True
    config.persist_db_on_completion = False
    
    # Snapshot steps for comparison
    if use_snapshot_steps is None:
        # Take snapshots at the start, middle, and end of the simulation
        snapshot_steps = [1, num_steps // 2, num_steps]
    else:
        snapshot_steps = use_snapshot_steps
    
    # Run first simulation
    print("Running first simulation...")
    sim_dir_1 = f"simulations/determinism_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}_1"
    os.makedirs(sim_dir_1, exist_ok=True)
    
    # Reset seeds again to ensure same initial state
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Keep CuDNN settings consistent
    
    # Run first simulation and capture state snapshots
    env1 = run_simulation(
        num_steps=num_steps,
        config=config,
        path=sim_dir_1,
        save_config=True,
        seed=seed
    )
    
    # Run second simulation
    print("Running second simulation...")
    sim_dir_2 = f"simulations/determinism_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}_2"
    os.makedirs(sim_dir_2, exist_ok=True)
    
    # Reset seeds again
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Keep CuDNN settings consistent
    
    # Run second simulation
    env2 = run_simulation(
        num_steps=num_steps,
        config=config,
        path=sim_dir_2,
        save_config=True,
        seed=seed
    )
    
    # Compare final states
    hash1, state1 = get_simulation_state_hash(env1, debug=True)
    hash2, state2 = get_simulation_state_hash(env2, debug=True)
    
    is_deterministic = hash1 == hash2
    
    print("\nDeterminism Test Results:")
    print(f"  - First simulation state hash: {hash1}")
    print(f"  - Second simulation state hash: {hash2}")
    print(f"  - Deterministic: {is_deterministic}")
    
    if not is_deterministic:
        print("\nDetailed comparison:")
        compare_states(state1, state2)
        
        print("\nThe simulation has non-deterministic elements.")
        print("This could be due to:")
        print("  - Uncontrolled random number generation")
        print("  - Floating point instability")
        print("  - Parallelism/threading issues")
        print("  - External state affecting the simulation")
    else:
        print("\nThe simulation is fully deterministic with the given seed.")
    
    return is_deterministic


def main():
    """Main entry point for the determinism test script."""
    parser = argparse.ArgumentParser(description="Test simulation determinism")
    parser.add_argument(
        "--environment",
        type=str,
        default="testing",
        choices=["development", "production", "testing"],
        help="Configuration environment"
    )
    parser.add_argument(
        "--steps", 
        type=int, 
        default=100, 
        help="Number of simulation steps to run"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="Seed value for reproducibility"
    )
    args = parser.parse_args()
    
    # Run determinism test
    is_deterministic = run_determinism_test(
        environment=args.environment,
        num_steps=args.steps,
        seed=args.seed
    )
    
    # Exit with status code based on determinism
    exit(0 if is_deterministic else 1)


if __name__ == "__main__":
    main() 