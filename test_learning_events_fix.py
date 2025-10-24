#!/usr/bin/env python3
"""
Test script to verify that learning events are now properly inserted into the database.
This script creates a simple simulation and checks if learning events are being inserted.
"""

import tempfile
import os
import sys
from pathlib import Path

# Add the farm module to the path
sys.path.insert(0, str(Path(__file__).parent))

from farm.core.simulation import run_simulation
from farm.config import SimulationConfig
from farm.database.models import LearningExperienceModel

def test_learning_event_insertion_fix():
    """Test that learning events are properly inserted after the fix."""
    
    # Create a temporary directory for the simulation
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Testing learning event insertion fix in: {temp_dir}")
        
        # Create a minimal simulation configuration
        config = SimulationConfig.from_centralized_config(environment="testing")
        
        # Enable debug logging
        from farm.utils.logging import configure_logging
        configure_logging(
            environment="testing",
            log_level="DEBUG",
            disable_console=False
        )
        
        # Reduce agent counts for faster testing
        config.population.system_agents = 2
        config.population.independent_agents = 2
        config.population.control_agents = 1
        
        # Reduce simulation steps
        config.max_steps = 10
        
        # Enable learning for agents
        config.agent_behavior.enable_learning = True
        
        # Use file-based database instead of in-memory for testing
        config.database.use_in_memory_db = False
        
        print("Running simulation...")
        
        # Run the simulation
        environment = run_simulation(
            num_steps=10,
            config=config,
            path=temp_dir,
            save_config=True,
            disable_console_logging=True
        )
        
        print("Simulation completed. Checking database...")
        
        # Check if learning events were inserted
        if environment.db is not None:
            # Use the session manager to get a session
            with environment.db.session_manager.session_scope() as session:
                experiences = session.query(LearningExperienceModel).filter(
                    LearningExperienceModel.simulation_id == environment.simulation_id
                ).all()
                
                print(f"Found {len(experiences)} learning experiences in database")
                
                # Debug: Check if agents have decision modules
                print(f"Number of agents: {len(environment.agents)}")
                for i, agent_id in enumerate(environment.agents[:3]):  # Check first 3 agents
                    agent = environment._agent_objects[agent_id]
                    print(f"Agent {i+1} ({agent_id}):")
                    print(f"  - Has behavior: {hasattr(agent, 'behavior')}")
                    if hasattr(agent, 'behavior'):
                        print(f"  - Behavior type: {type(agent.behavior).__name__}")
                        if hasattr(agent.behavior, 'decision_module'):
                            print(f"  - Has decision module: {agent.behavior.decision_module is not None}")
                            if agent.behavior.decision_module:
                                print(f"  - Algorithm type: {agent.behavior.decision_module.config.algorithm_type}")
                                print(f"  - Has database logger: {agent.behavior.decision_module._has_database_logger()}")
                
                if experiences:
                    print("✅ SUCCESS: Learning events are being inserted into the database!")
                    for i, exp in enumerate(experiences[:5]):  # Show first 5
                        print(f"  {i+1}. Agent: {exp.agent_id}, Step: {exp.step_number}, "
                              f"Module: {exp.module_type}, Action: {exp.action_taken_mapped}, "
                              f"Reward: {exp.reward}")
                    if len(experiences) > 5:
                        print(f"  ... and {len(experiences) - 5} more")
                    return True
                else:
                    print("❌ FAILURE: No learning events found in database")
                    return False
        else:
            print("❌ FAILURE: No database available")
            return False

if __name__ == "__main__":
    success = test_learning_event_insertion_fix()
    sys.exit(0 if success else 1)