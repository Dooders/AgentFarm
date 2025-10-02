#!/usr/bin/env python3
"""Line profile database logging"""
import sys
import os
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from farm.database.database import SimulationDatabase

try:
    from line_profiler import profile
except ImportError:
    def profile(func):
        return func

# Patch database logger methods
from farm.database.logger import DatabaseLogger
original_log_action = DatabaseLogger.log_agent_action

@profile
def profiled_log_action(self, step_number, agent_id, action_type, resources_before, 
                        resources_after, reward, details=None):
    return original_log_action(self, step_number, agent_id, action_type, 
                               resources_before, resources_after, reward, details)

DatabaseLogger.log_agent_action = profiled_log_action

def main():
    """Profile database logging operations."""
    # Create temp database
    temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    db_path = temp_db.name
    temp_db.close()
    
    try:
        db = SimulationDatabase(db_path=db_path, simulation_id="test_sim")
        
        print("Profiling database logging...")
        # Log many actions
        for i in range(1000):
            db.logger.log_agent_action(
                step_number=i,
                agent_id=f"agent_{i % 50}",
                action_type="move",
                resources_before=10.0,
                resources_after=9.5,
                reward=0.1,
                details={},
            )
        
        db.logger.flush_all_buffers()
        db.close()
        print("Profiling complete!")
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)

if __name__ == "__main__":
    main()
