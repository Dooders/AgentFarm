#!/usr/bin/env python3
"""
Example: Using Centralized Storage for Multiple Simulations

This example demonstrates how to store multiple simulations in a single
database file using ExperimentDatabase instead of creating separate
database files for each simulation.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from farm.config import SimulationConfig
from farm.database.experiment_database import ExperimentDatabase
from farm.database.data_logging import DataLoggingConfig


def example_basic_usage():
    """Example 1: Basic usage of ExperimentDatabase."""
    print("\n" + "="*70)
    print("Example 1: Basic Usage")
    print("="*70)
    
    # Create a centralized database
    db_path = "experiments/example_experiment.db"
    os.makedirs("experiments", exist_ok=True)
    
    config = SimulationConfig.from_centralized_config()
    
    experiment_db = ExperimentDatabase(
        db_path=db_path,
        experiment_id="example_exp_001",
        config=config
    )
    
    print(f"Created experiment database: {db_path}")
    
    # Run 3 simulations
    for i in range(3):
        sim_id = f"sim_{i:03d}"
        print(f"\nRunning simulation: {sim_id}")
        
        # Create context for this simulation
        sim_context = experiment_db.create_simulation_context(
            simulation_id=sim_id,
            parameters={"run": i, "seed": i * 100}
        )
        
        # Simulate some data logging
        # In real usage, your simulation would call these methods
        
        # Log a few steps
        for step in range(5):
            agent_states = [
                (f"agent_{j}", float(j), float(j), 100.0, 100.0, 100.0, 0, False, 0.0, step)
                for j in range(10)
            ]
            resource_states = [
                (f"resource_{k}", 50.0, float(k*10), float(k*10))
                for k in range(5)
            ]
            metrics = {
                "total_agents": len(agent_states),
                "total_resources": sum(r[1] for r in resource_states),
                "average_agent_health": 100.0,
                "average_agent_resources": 100.0,
                "births": 0,
                "deaths": 0,
                "resources_consumed": 0.0
            }
            
            sim_context.logger.log_step(step, agent_states, resource_states, metrics)
        
        # Flush data
        sim_context.flush_all_buffers()
        
        # Update status
        experiment_db.update_simulation_status(
            simulation_id=sim_id,
            status="completed",
            results_summary={"steps": 5, "final_agents": 10}
        )
        
        print(f"  ✓ Completed {sim_id}")
    
    # Close database
    experiment_db.close()
    
    print(f"\n✓ All simulations stored in: {db_path}")
    print("\nQuery with: python scripts/query_experiment.py experiments/example_experiment.db --command list")


def example_with_logging_config():
    """Example 2: Using custom logging configuration."""
    print("\n" + "="*70)
    print("Example 2: Custom Logging Configuration")
    print("="*70)
    
    db_path = "experiments/example_with_config.db"
    os.makedirs("experiments", exist_ok=True)
    
    config = SimulationConfig.from_centralized_config()
    
    experiment_db = ExperimentDatabase(
        db_path=db_path,
        experiment_id="example_exp_002",
        config=config
    )
    
    print(f"Created experiment database: {db_path}")
    
    # Create custom logging configuration
    logging_config = DataLoggingConfig(
        buffer_size=100,  # Smaller buffer for more frequent writes
        commit_interval=10  # Commit every 10 seconds
    )
    
    sim_context = experiment_db.create_simulation_context(
        simulation_id="sim_custom",
        parameters={"note": "Using custom logging config"},
        logging_config=logging_config
    )
    
    print(f"Created simulation with custom logging config:")
    print(f"  - Buffer size: {logging_config.buffer_size}")
    print(f"  - Commit interval: {logging_config.commit_interval}s")
    
    # Log some data
    for step in range(3):
        agent_states = [
            (f"agent_{j}", 0.0, 0.0, 50.0, 50.0, 100.0, 0, False, 0.0, step)
            for j in range(5)
        ]
        metrics = {
            "total_agents": len(agent_states),
            "total_resources": 100.0,
            "average_agent_health": 50.0,
            "average_agent_resources": 50.0,
            "births": 0,
            "deaths": 0,
            "resources_consumed": 0.0
        }
        
        sim_context.logger.log_step(step, agent_states, [], metrics)
    
    sim_context.flush_all_buffers()
    experiment_db.update_simulation_status("sim_custom", "completed")
    experiment_db.close()
    
    print(f"\n✓ Simulation stored in: {db_path}")


def example_query_results():
    """Example 3: Querying stored simulation data."""
    print("\n" + "="*70)
    print("Example 3: Querying Stored Data")
    print("="*70)
    
    db_path = "experiments/example_experiment.db"
    
    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}")
        print("Run example_basic_usage() first!")
        return
    
    # Connect to database
    config = SimulationConfig.from_centralized_config()
    experiment_db = ExperimentDatabase(
        db_path=db_path,
        experiment_id="example_exp_001",
        config=config
    )
    
    # Query simulation IDs
    sim_ids = experiment_db.get_simulation_ids()
    print(f"\nFound {len(sim_ids)} simulations:")
    for sim_id in sim_ids:
        print(f"  - {sim_id}")
    
    # Query using SQLAlchemy
    from farm.database.models import SimulationStepModel
    
    session = experiment_db.Session()
    
    # Get step counts per simulation
    print("\nStep counts per simulation:")
    for sim_id in sim_ids:
        step_count = session.query(SimulationStepModel).filter(
            SimulationStepModel.simulation_id == sim_id
        ).count()
        print(f"  {sim_id}: {step_count} steps")
    
    session.close()
    experiment_db.close()
    
    print("\n✓ Query completed")


def example_comparison():
    """Example 4: Comparing multiple simulations."""
    print("\n" + "="*70)
    print("Example 4: Comparing Simulations")
    print("="*70)
    
    db_path = "experiments/comparison_experiment.db"
    os.makedirs("experiments", exist_ok=True)
    
    config = SimulationConfig.from_centralized_config()
    
    # Create experiment with different parameters
    experiment_db = ExperimentDatabase(
        db_path=db_path,
        experiment_id="comparison_exp",
        config=config
    )
    
    print(f"Running parameter sweep experiment...")
    
    # Test different initial population sizes
    for pop_size in [5, 10, 20]:
        sim_id = f"sim_pop{pop_size}"
        
        sim_context = experiment_db.create_simulation_context(
            simulation_id=sim_id,
            parameters={"initial_population": pop_size}
        )
        
        # Simulate population growth
        for step in range(10):
            # Simple growth model
            current_pop = pop_size + step
            
            agent_states = [
                (f"agent_{j}", 0.0, 0.0, 100.0, 100.0, 100.0, 0, False, 0.0, step)
                for j in range(current_pop)
            ]
            metrics = {
                "total_agents": current_pop,
                "total_resources": 1000.0,
                "average_agent_health": 100.0,
                "average_agent_resources": 100.0,
                "births": 1 if step > 0 else pop_size,
                "deaths": 0,
                "resources_consumed": 0.0
            }
            
            sim_context.logger.log_step(step, agent_states, [], metrics)
        
        sim_context.flush_all_buffers()
        experiment_db.update_simulation_status(
            simulation_id=sim_id,
            status="completed",
            results_summary={"initial_pop": pop_size, "final_pop": pop_size + 9}
        )
        
        print(f"  ✓ Completed {sim_id}")
    
    experiment_db.close()
    
    print(f"\n✓ Comparison experiment stored in: {db_path}")
    print("\nCompare with: python scripts/query_experiment.py experiments/comparison_experiment.db --command compare --metric total_agents")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("CENTRALIZED STORAGE EXAMPLES")
    print("="*70)
    
    try:
        # Run examples
        example_basic_usage()
        example_with_logging_config()
        example_query_results()
        example_comparison()
        
        print("\n" + "="*70)
        print("ALL EXAMPLES COMPLETED")
        print("="*70)
        print("\nNext steps:")
        print("1. Explore the databases with: python scripts/query_experiment.py")
        print("2. Read the guide: docs/CENTRALIZED_STORAGE_GUIDE.md")
        print("3. Integrate with your simulation code")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
