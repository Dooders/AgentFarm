#!/usr/bin/env python3
"""
Test script to verify learning event database insertion.
This script creates a simple simulation and checks if learning events are being inserted.
"""

import tempfile
import os
import sys
from pathlib import Path

# Add the farm module to the path
sys.path.insert(0, str(Path(__file__).parent))

from farm.database.database import SimulationDatabase
from farm.database.data_logging import DataLogger
from farm.database.models import LearningExperienceModel
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

def test_learning_event_insertion():
    """Test that learning events are properly inserted into the database."""
    
    # Create a temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
        db_path = tmp_file.name
    
    try:
        # Create database
        simulation_id = "test_sim_001"
        db = SimulationDatabase(db_path, simulation_id=simulation_id)
        
        # Create data logger
        logger = DataLogger(db, simulation_id)
        
        print(f"Testing learning event insertion with simulation_id: {simulation_id}")
        print(f"Database path: {db_path}")
        
        # Create required parent records first
        print("\n=== Setting up required parent records ===")
        
        # Create simulation record
        from farm.database.models import Simulation
        with db.get_session() as session:
            simulation = Simulation(
                simulation_id=simulation_id,
                status="running",
                parameters={"test": "parameters"},
                simulation_db_path=db_path
            )
            session.add(simulation)
            session.commit()
        
        # Create agent record
        from farm.database.models import AgentModel
        with db.get_session() as session:
            agent = AgentModel(
                simulation_id=simulation_id,
                agent_id="test_agent_001",
                birth_time=0,
                agent_type="test",
                position_x=0.0,
                position_y=0.0,
                initial_resources=100.0,
                starting_health=100.0,
                starvation_counter=0,
                genome_id="test_genome",
                generation=1
            )
            session.add(agent)
            session.commit()
        
        print("Parent records created successfully")
        
        # Test 1: Direct learning experience logging
        print("\n=== Test 1: Direct learning experience logging ===")
        logger.log_learning_experience(
            step_number=1,
            agent_id="test_agent_001",
            module_type="test_module",
            module_id=123,
            action_taken=0,
            action_taken_mapped="test_action",
            reward=1.5
        )
        
        # Flush the buffer to ensure data is written
        logger.flush_learning_buffer()
        
        # Check if data was inserted
        with db.get_session() as session:
            experiences = session.query(LearningExperienceModel).filter(
                LearningExperienceModel.simulation_id == simulation_id
            ).all()
            
            print(f"Found {len(experiences)} learning experiences in database")
            for exp in experiences:
                print(f"  - Agent: {exp.agent_id}, Step: {exp.step_number}, Reward: {exp.reward}")
        
        # Create additional agents for multiple learning experiences
        print("\n=== Creating additional agents ===")
        with db.get_session() as session:
            for i in range(5):
                agent = AgentModel(
                    simulation_id=simulation_id,
                    agent_id=f"test_agent_{i:03d}",
                    birth_time=0,
                    agent_type="test",
                    position_x=float(i),
                    position_y=float(i),
                    initial_resources=100.0,
                    starting_health=100.0,
                    starvation_counter=0,
                    genome_id=f"test_genome_{i}",
                    generation=1
                )
                session.add(agent)
            session.commit()
        
        # Test 2: Multiple learning experiences
        print("\n=== Test 2: Multiple learning experiences ===")
        for i in range(5):
            logger.log_learning_experience(
                step_number=i + 2,
                agent_id=f"test_agent_{i:03d}",
                module_type="test_module",
                module_id=123 + i,
                action_taken=i,
                action_taken_mapped=f"test_action_{i}",
                reward=float(i) * 0.5
            )
        
        # Flush all buffers
        logger.flush_all_buffers()
        
        # Check final count
        with db.get_session() as session:
            experiences = session.query(LearningExperienceModel).filter(
                LearningExperienceModel.simulation_id == simulation_id
            ).all()
            
            print(f"Found {len(experiences)} total learning experiences in database")
            
            # Print all experiences
            for exp in experiences:
                print(f"  - Agent: {exp.agent_id}, Step: {exp.step_number}, "
                      f"Module: {exp.module_type}, Action: {exp.action_taken_mapped}, "
                      f"Reward: {exp.reward}")
        
        # Test 3: Check database schema
        print("\n=== Test 3: Database schema verification ===")
        engine = create_engine(f"sqlite:///{db_path}")
        with engine.connect() as conn:
            result = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='learning_experiences'")
            table_exists = result.fetchone() is not None
            print(f"learning_experiences table exists: {table_exists}")
            
            if table_exists:
                result = conn.execute("PRAGMA table_info(learning_experiences)")
                columns = result.fetchall()
                print("Table columns:")
                for col in columns:
                    print(f"  - {col[1]} ({col[2]})")
        
        print("\n=== Test Results ===")
        if len(experiences) == 6:  # 1 + 5
            print("✅ SUCCESS: All learning events were properly inserted into the database")
            return True
        else:
            print(f"❌ FAILURE: Expected 6 learning events, found {len(experiences)}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        if os.path.exists(db_path):
            os.unlink(db_path)

if __name__ == "__main__":
    success = test_learning_event_insertion()
    sys.exit(0 if success else 1)