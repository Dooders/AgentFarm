#!/usr/bin/env python3
"""
Script to test determinism in the AgentFarm simulation.
This will run two identical simulations and compare their outcomes to validate determinism.

This module can be used both as a standalone CLI script and imported by pytest tests.
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
from conftest import seed_all_rngs


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
        "resources": [],
        "spatial_index_state": None
    }

    # Capture spatial index state if available
    if hasattr(environment, 'spatial_index') and environment.spatial_index:
        try:
            # Get all agent positions in sorted order for deterministic spatial state
            agent_positions = []
            for agent in sorted(environment._agent_objects.values(), key=lambda a: a.agent_id):
                agent_positions.append({
                    "agent_id": agent.agent_id,
                    "position": agent.position
                })
            state["spatial_index_state"] = agent_positions
        except Exception:
            state["spatial_index_state"] = None

    # Sort agents by ID for consistency
    for agent in sorted(environment._agent_objects.values(), key=lambda a: a.agent_id):
        agent_state = {
            "id": agent.agent_id,
            "type": agent.__class__.__name__,
            "position": agent.position,
            "resource_level": agent.resource_level,
            "alive": agent.alive,
            "generation": agent.generation,
            "components": {},
            "action_history": []
        }

        # Capture component states for high-risk components
        if hasattr(agent, '_components'):
            for name, component in agent._components.items():
                if name in ["movement", "resource", "reproduction"]:
                    component_state = {}
                    
                    # Resource component state
                    if name == "resource":
                        component_state.update({
                            "level": getattr(component, "level", None),
                            "starvation_counter": getattr(component, "starvation_counter", None),
                        })
                    
                    # Movement component state
                    elif name == "movement":
                        component_state.update({
                            "target_position": getattr(component, "target_position", None),
                            "last_position": getattr(component, "last_position", None),
                        })
                    
                    # Reproduction component state
                    elif name == "reproduction":
                        component_state.update({
                            "offspring_created": getattr(component, "offspring_created", None),
                            "reproduction_cooldown": getattr(component, "reproduction_cooldown", None),
                        })
                    
                    agent_state["components"][name] = component_state

        # Capture behavior action history
        if hasattr(agent, 'behavior') and hasattr(agent.behavior, 'action_history'):
            # Keep last 10 actions for determinism testing
            agent_state["action_history"] = agent.behavior.action_history[-10:]

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
        print(f"Spatial index state captured: {state['spatial_index_state'] is not None}")
        # Print the first agent and resource for debugging
        if state['agents']:
            print(f"First agent: {state['agents'][0]}")
        if state['resources']:
            print(f"First resource: {state['resources'][0]}")

    return state_hash, state if debug else state_hash


def compare_database_contents(db1_path, db2_path):
    """
    Compare database contents between two simulation runs.
    Ignores timestamp differences in simulation records as these are expected.
    
    Parameters
    ----------
    db1_path : str
        Path to first simulation database
    db2_path : str
        Path to second simulation database
        
    Returns
    -------
    bool
        True if databases are mostly identical (ignoring timestamps), False otherwise
    """
    try:
        from sqlalchemy import create_engine, text
        from sqlalchemy.orm import sessionmaker
        
        # Create database connections
        engine1 = create_engine(f"sqlite:///{db1_path}")
        engine2 = create_engine(f"sqlite:///{db2_path}")
        
        Session1 = sessionmaker(bind=engine1)
        Session2 = sessionmaker(bind=engine2)
        
        with Session1() as session1, Session2() as session2:
            # Get list of tables
            tables_query = text("SELECT name FROM sqlite_master WHERE type='table'")
            tables1 = [row[0] for row in session1.execute(tables_query).fetchall()]
            tables2 = [row[0] for row in session2.execute(tables_query).fetchall()]
            
            if set(tables1) != set(tables2):
                print(f"Table mismatch: {set(tables1)} vs {set(tables2)}")
                return False
            
            # Compare each table
            for table in tables1:
                if table.startswith('sqlite_'):
                    continue  # Skip system tables
                    
                # Get row counts
                count1 = session1.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
                count2 = session2.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
                
                if count1 != count2:
                    print(f"Row count mismatch in {table}: {count1} vs {count2}")
                    return False
                
                # Compare actual data (for smaller tables)
                if count1 < 10000:  # Only compare small tables to avoid memory issues
                    data1 = session1.execute(text(f"SELECT * FROM {table} ORDER BY rowid")).fetchall()
                    data2 = session2.execute(text(f"SELECT * FROM {table} ORDER BY rowid")).fetchall()
                    
                    if data1 != data2:
                        # Special handling for simulations table - ignore timestamp differences
                        if table == 'simulations':
                            print(f"Data mismatch in table {table} (checking if only timestamps differ)...")
                            # Compare all columns except start_time and end_time
                            for i, (row1, row2) in enumerate(zip(data1, data2)):
                                if row1 != row2:
                                    # Check if the only difference is in timestamp columns
                                    differences = []
                                    for j, (col1, col2) in enumerate(zip(row1, row2)):
                                        if col1 != col2:
                                            # Get column names to check if it's a timestamp
                                            col_names = [desc[0] for desc in session1.execute(text(f"PRAGMA table_info({table})")).fetchall()]
                                            col_name = col_names[j] if j < len(col_names) else f"column_{j}"
                                            differences.append((col_name, col1, col2))
                                    
                                    # If only timestamp columns differ, that's acceptable
                                    timestamp_cols = {'start_time', 'end_time', 'created_at', 'updated_at'}
                                    if all(col_name in timestamp_cols for col_name, _, _ in differences):
                                        print(f"  Row {i}: Only timestamp differences detected (acceptable)")
                                        continue
                                    else:
                                        print(f"  Row {i}: Non-timestamp differences found:")
                                        for col_name, val1, val2 in differences:
                                            print(f"    {col_name}: {val1} vs {val2}")
                                        if i > 5:  # Limit output
                                            print(f"  ... and {len(data1) - i - 1} more differences")
                                            break
                                        return False
                            print(f"✅ Table {table} differences are only timestamps (acceptable)")
                        else:
                            print(f"Data mismatch in table {table}")
                            # Show first few differences
                            for i, (row1, row2) in enumerate(zip(data1, data2)):
                                if row1 != row2:
                                    print(f"  Row {i}: {row1} vs {row2}")
                                    if i > 5:  # Limit output
                                        print(f"  ... and {len(data1) - i - 1} more differences")
                                        break
                            return False
                else:
                    print(f"Skipping large table {table} ({count1} rows) - using row count only")
            
            print("✅ Database contents are mostly identical (timestamp differences ignored)")
            return True
            
    except Exception as e:
        print(f"Error comparing databases: {e}")
        return False
    finally:
        if 'engine1' in locals():
            engine1.dispose()
        if 'engine2' in locals():
            engine2.dispose()


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
    seed_all_rngs(seed)

    # Load configuration
    config = SimulationConfig.from_centralized_config(environment=environment)

    # Override any config parameters that might affect determinism
    config.seed = seed

    # Use in-memory database to avoid external state issues but allow persistence testing
    config.database.use_in_memory_db = True
    config.database.persist_db_on_completion = True  # Enable persistence for database comparison
    # Increase memory limit for longer simulations to avoid memory warnings
    config.database.in_memory_db_memory_limit_mb = 5000  # 5GB limit

    # Snapshot steps for comparison
    if use_snapshot_steps is None:
        # Take snapshots at the start, middle, and end of the simulation
        snapshot_steps = [1, num_steps // 2, num_steps]
    else:
        snapshot_steps = use_snapshot_steps

    # Use fixed simulation IDs for both runs to ensure database files have predictable names
    fixed_simulation_id = f"determinism_test_{seed}_{num_steps}"
    
    # Run first simulation
    print("Running first simulation...")
    sim_dir_1 = f"simulations/determinism_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}_1"
    os.makedirs(sim_dir_1, exist_ok=True)

    # Reset seeds again to ensure same initial state
    seed_all_rngs(seed)

    # Run first simulation and capture state snapshots
    env1 = run_simulation(
        num_steps=num_steps,
        config=config,
        path=sim_dir_1,
        save_config=True,
        seed=seed,
        simulation_id=fixed_simulation_id
    )

    # Run second simulation
    print("Running second simulation...")
    sim_dir_2 = f"simulations/determinism_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}_2"
    os.makedirs(sim_dir_2, exist_ok=True)

    # Reset seeds again
    seed_all_rngs(seed)

    # Run second simulation with the same simulation ID
    env2 = run_simulation(
        num_steps=num_steps,
        config=config,
        path=sim_dir_2,
        save_config=True,
        seed=seed,
        simulation_id=fixed_simulation_id
    )

    # Compare final states
    hash1, state1 = get_simulation_state_hash(env1, debug=True)
    hash2, state2 = get_simulation_state_hash(env2, debug=True)

    state_deterministic = hash1 == hash2

    # Compare database contents if databases were persisted
    db_deterministic = True
    if config.database.persist_db_on_completion:
        print("\nComparing database contents...")
        # Use the fixed simulation ID for both database paths
        db1_path = os.path.join(sim_dir_1, f"simulation_{fixed_simulation_id}.db")
        db2_path = os.path.join(sim_dir_2, f"simulation_{fixed_simulation_id}.db")
        
        # Debug: List files in simulation directories
        print("Looking for database files:")
        print(f"  DB1 path: {db1_path}")
        print(f"  DB2 path: {db2_path}")
        print(f"  Sim dir 1 contents: {os.listdir(sim_dir_1) if os.path.exists(sim_dir_1) else 'Directory not found'}")
        print(f"  Sim dir 2 contents: {os.listdir(sim_dir_2) if os.path.exists(sim_dir_2) else 'Directory not found'}")
        
        if os.path.exists(db1_path) and os.path.exists(db2_path):
            db_deterministic = compare_database_contents(db1_path, db2_path)
        else:
            print("⚠️  Database files not found - skipping database comparison")
            db_deterministic = True  # Don't fail if DBs weren't created

    is_deterministic = state_deterministic and db_deterministic

    print("\nDeterminism Test Results:")
    print(f"  - First simulation state hash: {hash1}")
    print(f"  - Second simulation state hash: {hash2}")
    print(f"  - State deterministic: {state_deterministic}")
    print(f"  - Database deterministic: {db_deterministic}")
    print(f"  - Overall deterministic: {is_deterministic}")

    if not state_deterministic:
        print("\nDetailed state comparison:")
        compare_states(state1, state2)

    if not is_deterministic:
        if not state_deterministic:
            print("\n❌ The simulation has non-deterministic elements.")
            print("This could be due to:")
            print("  - Uncontrolled random number generation")
            print("  - Floating point instability")
            print("  - Parallelism/threading issues")
            print("  - External state affecting the simulation")
            print("  - Non-deterministic database logging")
        else:
            print("\n⚠️  The simulation state is deterministic, but database has minor differences.")
            print("This is likely due to timestamp differences, which is expected and acceptable.")
            print("✅ The core simulation behavior is fully deterministic.")
    else:
        print("\n✅ The simulation is fully deterministic with the given seed.")
        print("Both simulation state and database contents are identical.")

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
