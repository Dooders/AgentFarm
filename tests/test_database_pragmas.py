"""Tests for SQLite pragma optimizations in the database module."""

import os
import tempfile
import unittest
from pathlib import Path

from farm.core.config_hydra_bridge import HydraSimulationConfig
from farm.database import (
    SimulationDatabase,
    InMemorySimulationDatabase,
    get_pragma_profile,
    PRAGMA_PROFILES,
)


class TestDatabasePragmas(unittest.TestCase):
    """Test SQLite pragma optimizations for different database configurations."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_db.sqlite"
        self.config = HydraSimulationConfig()
        self.db = None

    def tearDown(self):
        """Clean up test environment."""
        if self.db:
            try:
                self.db.close()
            except Exception:
                pass
            self.db = None
            
        # Give the system a moment to release file handles
        import time
        time.sleep(0.5)
        
        try:
            self.temp_dir.cleanup()
        except Exception as e:
            print(f"Warning: Could not clean up temporary directory: {e}")

    def test_default_pragmas(self):
        """Test default pragma settings."""
        self.db = SimulationDatabase(self.db_path.as_posix(), simulation_id="test_simulation")
        pragmas = self.db.get_current_pragmas()
        
        # Check default settings
        self.assertEqual(pragmas.get("journal_mode", "").upper(), "WAL")
        self.assertEqual(int(pragmas.get("synchronous", -1)), 1)  # NORMAL
        
        # Handle temp_store as integer (0=DEFAULT, 1=FILE, 2=MEMORY)
        temp_store = pragmas.get("temp_store", 0)
        self.assertEqual(int(temp_store), 2)  # MEMORY
        
        self.assertEqual(pragmas.get("foreign_keys", 0), 1)
        
        self.db.close()
        self.db = None

    def test_performance_profile(self):
        """Test performance profile pragma settings."""
        self.config.db_pragma_profile = "performance"
        self.db = SimulationDatabase(self.db_path.as_posix(), self.config, simulation_id="test_simulation")
        pragmas = self.db.get_current_pragmas()
        
        # Check performance settings
        self.assertEqual(int(pragmas.get("synchronous", -1)), 0)  # OFF
        self.assertEqual(pragmas.get("journal_mode", "").upper(), "MEMORY")
        self.assertEqual(int(pragmas.get("page_size", -1)), 8192)
        self.assertEqual(pragmas.get("automatic_index", 1), 0)  # OFF
        
        self.db.close()
        self.db = None

    def test_safety_profile(self):
        """Test safety profile pragma settings."""
        self.config.db_pragma_profile = "safety"
        self.db = SimulationDatabase(self.db_path.as_posix(), self.config, simulation_id="test_simulation")
        pragmas = self.db.get_current_pragmas()
        
        # Check safety settings
        self.assertEqual(int(pragmas.get("synchronous", -1)), 2)  # FULL
        self.assertEqual(pragmas.get("journal_mode", "").upper(), "WAL")
        self.assertEqual(int(pragmas.get("page_size", -1)), 4096)
        
        self.db.close()
        self.db = None

    def test_memory_profile(self):
        """Test memory-optimized profile pragma settings."""
        self.config.db_pragma_profile = "memory"
        self.db = SimulationDatabase(self.db_path.as_posix(), self.config, simulation_id="test_simulation")
        pragmas = self.db.get_current_pragmas()
        
        # Check memory-optimized settings
        self.assertEqual(int(pragmas.get("synchronous", -1)), 1)  # NORMAL
        self.assertEqual(pragmas.get("journal_mode", "").upper(), "MEMORY")
        
        # Check cache size is limited
        cache_size = int(pragmas.get("cache_size", 0))
        self.assertLessEqual(abs(cache_size), 51200)  # 50MB or less
        
        self.db.close()
        self.db = None

    def test_custom_pragmas(self):
        """Test custom pragma overrides."""
        self.config.db_pragma_profile = "balanced"
        self.config.db_custom_pragmas = {
            "synchronous": "OFF",
            "cache_size": "-524288",  # 512MB
            "mmap_size": "536870912",  # 512MB
        }
        
        self.db = SimulationDatabase(self.db_path.as_posix(), self.config, simulation_id="test_simulation")
        pragmas = self.db.get_current_pragmas()
        
        # Check custom overrides
        self.assertEqual(int(pragmas.get("synchronous", -1)), 0)  # OFF
        self.assertEqual(pragmas.get("journal_mode", "").upper(), "WAL")  # Not overridden
        
        # Check cache size is set to custom value
        cache_size = int(pragmas.get("cache_size", 0))
        self.assertLessEqual(abs(cache_size), 524288)  # 512MB
        
        # Check mmap_size is set
        mmap_size = int(pragmas.get("mmap_size", 0))
        self.assertEqual(mmap_size, 536870912)  # 512MB
        
        self.db.close()
        self.db = None

    def test_in_memory_database_pragmas(self):
        """Test in-memory database pragma settings."""
        self.config.db_pragma_profile = "balanced"
        self.db = InMemorySimulationDatabase(config=self.config, simulation_id="test_simulation")
        pragmas = self.db.get_current_pragmas()
    
        # Check in-memory database uses the specified profile
        self.assertEqual(int(pragmas.get("synchronous", -1)), 1)  # NORMAL
    
        # NOTE: SQLite in-memory databases always use journal_mode=MEMORY
        # regardless of the setting, so we don't test for WAL mode here
        self.assertEqual(pragmas.get("journal_mode", "").upper(), "MEMORY")
    
        # Check other settings are applied correctly
        self.assertEqual(int(pragmas.get("page_size", -1)), 4096)  # Default for balanced
        self.assertEqual(int(pragmas.get("busy_timeout", -1)), 30000)  # 30 seconds for balanced
    
        self.db.close()
        self.db = None
    
        # Test with performance profile
        self.config.db_pragma_profile = "performance"
        self.db = InMemorySimulationDatabase(config=self.config, simulation_id="test_simulation")
        pragmas = self.db.get_current_pragmas()
    
        # Check performance settings
        # NOTE: In-memory databases now correctly apply the performance profile synchronous=OFF setting
        self.assertEqual(int(pragmas.get("synchronous", -1)), 0)  # OFF
        self.assertEqual(pragmas.get("journal_mode", "").upper(), "MEMORY")
        self.assertEqual(int(pragmas.get("page_size", -1)), 8192)  # Larger for performance
        self.assertEqual(int(pragmas.get("busy_timeout", -1)), 60000)  # 60 seconds for performance
    
        self.db.close()
        self.db = None

    def test_workload_adjustment(self):
        """Test runtime pragma adjustment for different workloads."""
        self.db = SimulationDatabase(self.db_path.as_posix(), simulation_id="test_simulation")
        
        # Check initial settings
        initial_pragmas = self.db.get_current_pragmas()
        self.assertEqual(int(initial_pragmas.get("synchronous", -1)), 1)  # NORMAL
        self.assertEqual(initial_pragmas.get("journal_mode", "").upper(), "WAL")
        
        # Adjust for write-heavy workload
        self.db.adjust_pragmas_for_workload("write_heavy")
        write_pragmas = self.db.get_current_pragmas()
        self.assertEqual(int(write_pragmas.get("synchronous", -1)), 0)  # OFF
        self.assertEqual(write_pragmas.get("journal_mode", "").upper(), "MEMORY")
        
        # Adjust for read-heavy workload
        self.db.adjust_pragmas_for_workload("read_heavy")
        read_pragmas = self.db.get_current_pragmas()
        self.assertEqual(int(read_pragmas.get("synchronous", -1)), 1)  # NORMAL
        self.assertEqual(read_pragmas.get("journal_mode", "").upper(), "WAL")
        self.assertGreater(int(read_pragmas.get("mmap_size", 0)), 0)  # mmap enabled
        
        self.db.close()
        self.db = None

    def test_pragma_profile_helper(self):
        """Test pragma profile helper functions."""
        # Test getting performance profile
        perf_profile = get_pragma_profile("performance")
        self.assertEqual(perf_profile["synchronous"], "OFF")
        self.assertEqual(perf_profile["journal_mode"], "MEMORY")
        
        # Test getting balanced profile
        balanced_profile = get_pragma_profile("balanced")
        self.assertEqual(balanced_profile["synchronous"], "NORMAL")
        self.assertEqual(balanced_profile["journal_mode"], "WAL")
        
        # Test getting unknown profile (should return balanced)
        unknown_profile = get_pragma_profile("unknown")
        self.assertEqual(unknown_profile, PRAGMA_PROFILES["balanced"])


if __name__ == "__main__":
    unittest.main() 