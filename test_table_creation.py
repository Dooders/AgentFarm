#!/usr/bin/env python3
"""
Test script to check if the learning_experiences table is being created.
"""

import tempfile
import os
import sys
from pathlib import Path

# Add the farm module to the path
sys.path.insert(0, str(Path(__file__).parent))

from farm.database.database import SimulationDatabase
from farm.database.models import Base
from sqlalchemy import create_engine, text

def test_table_creation():
    """Test that all tables including learning_experiences are created."""
    
    # Create a temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
        db_path = tmp_file.name
    
    try:
        print(f"Testing table creation with database: {db_path}")
        
        # Create database
        simulation_id = "test_sim_001"
        db = SimulationDatabase(db_path, simulation_id=simulation_id)
        
        # Check what tables were created
        with db.session_manager.session_scope() as session:
            # Get list of tables
            result = session.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
            tables = [row[0] for row in result.fetchall()]
            
            print(f"Created tables: {tables}")
            
            # Check if learning_experiences table exists
            if 'learning_experiences' in tables:
                print("✅ SUCCESS: learning_experiences table was created!")
                
                # Check table structure
                result = session.execute(text("PRAGMA table_info(learning_experiences)"))
                columns = result.fetchall()
                print("Table columns:")
                for col in columns:
                    print(f"  - {col[1]} ({col[2]})")
                
                return True
            else:
                print("❌ FAILURE: learning_experiences table was NOT created!")
                print("Available tables:", tables)
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
    success = test_table_creation()
    sys.exit(0 if success else 1)