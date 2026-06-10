#!/usr/bin/env python3
"""
Script to test determinism in the AgentFarm simulation.
This will run two identical simulations and compare their outcomes to validate determinism.

This module can be used both as a standalone CLI script and imported by pytest tests.

Note: this harness deliberately does NOT pre-seed any RNGs itself. Reproducibility
must come entirely from ``run_simulation(seed=...)`` so that a regression in the
production seeding path (``init_random_seeds`` / ``Environment`` seeding) is caught
here instead of being masked by test-side seeding.
"""

import argparse
import hashlib
import json
import os
from datetime import datetime

from farm.config import SimulationConfig
from farm.core.simulation import run_simulation

# Columns whose values are allowed to differ between two otherwise identical runs.
# module_id: object references can vary; simulation_db_path: absolute path to this run's DB file.
ACCEPTABLE_DIFFERENCE_COLUMNS = frozenset(
    {"start_time", "end_time", "created_at", "updated_at", "timestamp", "module_id", "simulation_db_path"}
)

# Number of rows fetched per chunk when comparing database tables.
ROW_CHUNK_SIZE = 5000


def capture_simulation_state(environment):
    """
    Capture a deterministic, JSON-serializable snapshot of the simulation state.

    Parameters
    ----------
    environment : Environment
        The simulation environment to snapshot

    Returns
    -------
    dict
        Serializable state snapshot
    """
    state = {
        "time": environment.time,
        "agents": [],
        "resources": [],
        "spatial_index_state": None,
    }

    # Capture spatial index state if available
    if hasattr(environment, "spatial_index") and environment.spatial_index:
        try:
            # Get all agent positions in sorted order for deterministic spatial state
            agent_positions = []
            for agent in sorted(environment._agent_objects.values(), key=lambda a: a.agent_id):
                agent_positions.append(
                    {
                        "agent_id": agent.agent_id,
                        "position": agent.position,
                    }
                )
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
            "action_history": [],
        }

        # Capture component states for high-risk components
        if hasattr(agent, "_components"):
            for name, component in agent._components.items():
                if name == "resource":
                    agent_state["components"][name] = {
                        "level": getattr(component, "level", None),
                        "starvation_counter": getattr(component, "starvation_counter", None),
                    }
                elif name == "movement":
                    agent_state["components"][name] = {
                        "target_position": getattr(component, "target_position", None),
                        "last_position": getattr(component, "last_position", None),
                    }
                elif name == "reproduction":
                    agent_state["components"][name] = {
                        "offspring_created": getattr(component, "offspring_created", None),
                        "reproduction_cooldown": getattr(component, "reproduction_cooldown", None),
                    }

        # Capture behavior action history (last 10 actions)
        if hasattr(agent, "behavior") and hasattr(agent.behavior, "action_history"):
            agent_state["action_history"] = agent.behavior.action_history[-10:]

        state["agents"].append(agent_state)

    # Sort resources by ID for consistency
    for resource in sorted(environment.resources, key=lambda r: r.resource_id):
        state["resources"].append(
            {
                "id": resource.resource_id,
                "position": resource.position,
                "amount": resource.amount,
            }
        )

    return state


def hash_simulation_state(state):
    """Hash a state snapshot produced by :func:`capture_simulation_state`."""
    state_json = json.dumps(state, sort_keys=True)
    return hashlib.sha256(state_json.encode()).hexdigest()


def get_simulation_state_hash(environment):
    """
    Generate a hash of the current simulation state to check for equality.

    Parameters
    ----------
    environment : Environment
        The simulation environment to hash

    Returns
    -------
    str
        SHA-256 hash of the state as a hex string
    """
    return hash_simulation_state(capture_simulation_state(environment))


def _print_state_summary(state):
    """Print a short summary of a captured state snapshot."""
    print(f"Time: {state['time']}")
    print(f"Number of agents: {len(state['agents'])}")
    print(f"Number of resources: {len(state['resources'])}")
    print(f"Spatial index state captured: {state['spatial_index_state'] is not None}")
    if state["agents"]:
        print(f"First agent: {state['agents'][0]}")
    if state["resources"]:
        print(f"First resource: {state['resources'][0]}")


def _rows_match(row1, row2, col_names):
    """
    Compare two table rows, tolerating differences in acceptable columns.

    Returns
    -------
    tuple
        (match, differences) where differences lists (col_name, val1, val2)
        for non-acceptable mismatches.
    """
    differences = []
    for j, (col1, col2) in enumerate(zip(row1, row2)):
        if col1 != col2:
            col_name = col_names[j] if j < len(col_names) else f"column_{j}"
            if col_name not in ACCEPTABLE_DIFFERENCE_COLUMNS:
                differences.append((col_name, col1, col2))
    return not differences, differences


def _compare_table(session1, session2, table):
    """
    Compare one table between two databases, chunk by chunk.

    Returns True if the tables match (ignoring acceptable columns), False otherwise.
    """
    from sqlalchemy import text

    count1 = session1.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
    count2 = session2.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()

    if count1 != count2:
        print(f"Row count mismatch in {table}: {count1} vs {count2}")
        return False

    col_info = session1.execute(text(f"PRAGMA table_info({table})")).fetchall()
    col_names = [row[1] for row in col_info]  # Column name is in index 1

    mismatches_reported = 0
    table_matches = True
    for offset in range(0, count1, ROW_CHUNK_SIZE):
        chunk_query = text(f"SELECT * FROM {table} ORDER BY rowid LIMIT :limit OFFSET :offset")
        params = {"limit": ROW_CHUNK_SIZE, "offset": offset}
        data1 = session1.execute(chunk_query, params).fetchall()
        data2 = session2.execute(chunk_query, params).fetchall()

        if data1 == data2:
            continue

        for i, (row1, row2) in enumerate(zip(data1, data2)):
            if row1 == row2:
                continue
            match, differences = _rows_match(row1, row2, col_names)
            if not match:
                table_matches = False
                mismatches_reported += 1
                if mismatches_reported <= 5:
                    print(f"  Row {offset + i}: Non-acceptable differences found:")
                    for col_name, val1, val2 in differences:
                        print(f"    {col_name}: {val1} vs {val2}")
                else:
                    print("  ... further differences suppressed")
                    return False

    return table_matches


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
        True if databases are identical (ignoring acceptable columns), False otherwise
    """
    try:
        from sqlalchemy import create_engine, text
        from sqlalchemy.orm import sessionmaker

        engine1 = create_engine(f"sqlite:///{db1_path}")
        engine2 = create_engine(f"sqlite:///{db2_path}")

        Session1 = sessionmaker(bind=engine1)
        Session2 = sessionmaker(bind=engine2)

        with Session1() as session1, Session2() as session2:
            tables_query = text("SELECT name FROM sqlite_master WHERE type='table'")
            tables1 = [row[0] for row in session1.execute(tables_query).fetchall()]
            tables2 = [row[0] for row in session2.execute(tables_query).fetchall()]

            if set(tables1) != set(tables2):
                print(f"Table mismatch: {set(tables1)} vs {set(tables2)}")
                return False

            for table in tables1:
                if table.startswith("sqlite_"):
                    continue  # Skip system tables
                if not _compare_table(session1, session2, table):
                    print(f"❌ Table {table} differs between runs")
                    return False

            print("✅ Database contents are identical (acceptable column differences ignored)")
            return True

    except Exception as e:
        print(f"Error comparing databases: {e}")
        return False
    finally:
        if "engine1" in locals():
            engine1.dispose()
        if "engine2" in locals():
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


def _make_snapshot_recorder(snapshot_steps, snapshots):
    """
    Build an ``on_step_end`` hook that records state hashes at the given step numbers.

    Parameters
    ----------
    snapshot_steps : list of int
        1-based step numbers at which to record a state hash
    snapshots : dict
        Output mapping of step number -> state hash, filled in during the run
    """
    target_steps = set(snapshot_steps)

    def on_step_end(environment, step_index):
        step_number = step_index + 1
        if step_number in target_steps:
            snapshots[step_number] = get_simulation_state_hash(environment)

    return on_step_end


def run_determinism_test(environment, num_steps, seed=42, snapshot_steps=None):
    """
    Run two identical simulations with the same seed and compare their results.

    Reproducibility relies solely on ``run_simulation(seed=...)``; this harness
    intentionally performs no RNG seeding of its own.

    Parameters
    ----------
    environment : str
        Configuration environment name to load
    num_steps : int
        Number of simulation steps to run
    seed : int
        Seed value to use for both simulations
    snapshot_steps : list, optional
        1-based step numbers at which to compare intermediate state hashes.
        Defaults to the start, middle, and end of the simulation.

    Returns
    -------
    bool
        True if the simulations were deterministic, False otherwise
    """
    print(f"Starting determinism test with seed {seed}")

    # Load configuration
    config = SimulationConfig.from_centralized_config(environment=environment)

    # Override any config parameters that might affect determinism
    config.seed = seed

    # Use in-memory database to avoid external state issues but allow persistence testing
    config.database.use_in_memory_db = True
    config.database.persist_db_on_completion = True  # Enable persistence for database comparison
    # Increase memory limit for longer simulations to avoid memory warnings
    config.database.in_memory_db_memory_limit_mb = 5000  # 5GB limit

    if snapshot_steps is None:
        # Take snapshots at the start, middle, and end of the simulation
        snapshot_steps = [1, num_steps // 2, num_steps]
    snapshot_steps = sorted({step for step in snapshot_steps if 1 <= step <= num_steps})

    # Use fixed simulation IDs for both runs to ensure database files have predictable names
    fixed_simulation_id = f"determinism_test_{seed}_{num_steps}"

    # Run first simulation
    print("Running first simulation...")
    sim_dir_1 = f"simulations/determinism_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}_1"
    os.makedirs(sim_dir_1, exist_ok=True)

    snapshots1 = {}
    env1 = run_simulation(
        num_steps=num_steps,
        config=config,
        path=sim_dir_1,
        save_config=True,
        seed=seed,
        simulation_id=fixed_simulation_id,
        on_step_end=_make_snapshot_recorder(snapshot_steps, snapshots1),
    )

    # Run second simulation
    print("Running second simulation...")
    sim_dir_2 = f"simulations/determinism_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}_2"
    os.makedirs(sim_dir_2, exist_ok=True)

    snapshots2 = {}
    env2 = run_simulation(
        num_steps=num_steps,
        config=config,
        path=sim_dir_2,
        save_config=True,
        seed=seed,
        simulation_id=fixed_simulation_id,
        on_step_end=_make_snapshot_recorder(snapshot_steps, snapshots2),
    )

    # Compare final states
    state1 = capture_simulation_state(env1)
    state2 = capture_simulation_state(env2)
    _print_state_summary(state1)
    _print_state_summary(state2)
    hash1 = hash_simulation_state(state1)
    hash2 = hash_simulation_state(state2)

    state_deterministic = hash1 == hash2

    # Compare intermediate snapshots (require every requested step to be captured)
    snapshots_deterministic = all(
        step in snapshots1 and step in snapshots2 and snapshots1[step] == snapshots2[step]
        for step in snapshot_steps
    )
    if not snapshots_deterministic:
        print("\n❌ Intermediate state snapshots diverge:")
        for step in snapshot_steps:
            h1 = snapshots1.get(step)
            h2 = snapshots2.get(step)
            marker = "==" if h1 == h2 else "!="
            print(f"  Step {step}: {h1} {marker} {h2}")

    # Compare database contents (persistence is enabled above, so missing files are a failure)
    print("\nComparing database contents...")
    db1_path = os.path.join(sim_dir_1, f"simulation_{fixed_simulation_id}.db")
    db2_path = os.path.join(sim_dir_2, f"simulation_{fixed_simulation_id}.db")

    print("Looking for database files:")
    print(f"  DB1 path: {db1_path}")
    print(f"  DB2 path: {db2_path}")

    if os.path.exists(db1_path) and os.path.exists(db2_path):
        db_deterministic = compare_database_contents(db1_path, db2_path)
    else:
        print("❌ Database files not found - persistence failed, treating as non-deterministic")
        print(f"  Sim dir 1 contents: {os.listdir(sim_dir_1) if os.path.exists(sim_dir_1) else 'Directory not found'}")
        print(f"  Sim dir 2 contents: {os.listdir(sim_dir_2) if os.path.exists(sim_dir_2) else 'Directory not found'}")
        db_deterministic = False

    is_deterministic = state_deterministic and snapshots_deterministic and db_deterministic

    print("\nDeterminism Test Results:")
    print(f"  - First simulation state hash: {hash1}")
    print(f"  - Second simulation state hash: {hash2}")
    print(f"  - State deterministic: {state_deterministic}")
    print(f"  - Intermediate snapshots deterministic: {snapshots_deterministic} (steps {snapshot_steps})")
    print(f"  - Database deterministic: {db_deterministic}")
    print(f"  - Overall deterministic: {is_deterministic}")

    if not state_deterministic:
        print("\nDetailed state comparison:")
        compare_states(state1, state2)

    if not is_deterministic:
        print("\n❌ The simulation has non-deterministic elements.")
        print("This could be due to:")
        print("  - Uncontrolled random number generation")
        print("  - Floating point instability")
        print("  - Parallelism/threading issues")
        print("  - External state affecting the simulation")
        print("  - Non-deterministic database logging")
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
