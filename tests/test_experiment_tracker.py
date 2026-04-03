"""Tests for farm/core/experiment_tracker.py (ExperimentTracker class)."""
import json
import os
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from farm.core.experiment_tracker import ExperimentTracker


def _make_tracker(tmp_dir):
    """Create an ExperimentTracker using a temporary directory."""
    return ExperimentTracker(experiments_dir=tmp_dir)


def _make_sqlite_db(path: str) -> None:
    """Create a minimal SQLite database with a SimulationMetrics table."""
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE SimulationMetrics (id INTEGER PRIMARY KEY, value REAL)"
        )
        for i in range(5):
            conn.execute("INSERT INTO SimulationMetrics (value) VALUES (?)", (float(i),))
        conn.commit()


class TestExperimentTrackerInit(unittest.TestCase):
    def test_creates_directory(self):
        with tempfile.TemporaryDirectory() as parent:
            tracker_dir = os.path.join(parent, "experiments")
            ExperimentTracker(experiments_dir=tracker_dir)
            self.assertTrue(os.path.isdir(tracker_dir))

    def test_creates_metadata_file(self):
        with tempfile.TemporaryDirectory() as parent:
            tracker_dir = os.path.join(parent, "experiments")
            ExperimentTracker(experiments_dir=tracker_dir)
            metadata_file = os.path.join(tracker_dir, "metadata.json")
            self.assertTrue(os.path.exists(metadata_file))

    def test_loads_existing_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            tracker_dir = os.path.join(tmp, "experiments")
            os.makedirs(tracker_dir, exist_ok=True)
            metadata_file = os.path.join(tracker_dir, "metadata.json")
            with open(metadata_file, "w") as f:
                json.dump({"experiments": {"exp_1": {"name": "test"}}}, f)
            tracker = ExperimentTracker(experiments_dir=tracker_dir)
            self.assertIn("exp_1", tracker.metadata["experiments"])


class TestRegisterExperiment(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.tracker = _make_tracker(self.tmp)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_returns_experiment_id(self):
        exp_id = self.tracker.register_experiment("test_run", {"steps": 100}, "data.db")
        self.assertIsInstance(exp_id, str)
        self.assertGreater(len(exp_id), 0)

    def test_stores_metadata(self):
        exp_id = self.tracker.register_experiment("my_exp", {"steps": 50}, "sim.db")
        self.assertIn(exp_id, self.tracker.metadata["experiments"])
        exp_data = self.tracker.metadata["experiments"][exp_id]
        self.assertEqual(exp_data["name"], "my_exp")
        self.assertEqual(exp_data["config"], {"steps": 50})
        self.assertEqual(exp_data["db_path"], "sim.db")
        self.assertEqual(exp_data["status"], "registered")

    def test_empty_name_raises(self):
        with self.assertRaises(ValueError):
            self.tracker.register_experiment("", {}, "db.sqlite")

    def test_whitespace_name_raises(self):
        with self.assertRaises(ValueError):
            self.tracker.register_experiment("   ", {}, "db.sqlite")

    def test_empty_db_path_raises(self):
        with self.assertRaises(ValueError):
            self.tracker.register_experiment("valid_name", {}, "")

    def test_name_is_stripped(self):
        exp_id = self.tracker.register_experiment("  run_1  ", {}, "db.db")
        exp_data = self.tracker.metadata["experiments"][exp_id]
        self.assertEqual(exp_data["name"], "run_1")

    def test_metadata_persisted_to_file(self):
        exp_id = self.tracker.register_experiment("persistent", {}, "p.db")
        with open(self.tracker.metadata_file) as f:
            saved = json.load(f)
        self.assertIn(exp_id, saved["experiments"])

    def test_unique_ids_for_multiple_registrations(self):
        id1 = self.tracker.register_experiment("run1", {}, "db1.db")
        id2 = self.tracker.register_experiment("run2", {}, "db2.db")
        self.assertNotEqual(id1, id2)


class TestExportExperimentData(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.tracker = _make_tracker(self.tmp)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_export_to_csv(self):
        db_path = os.path.join(self.tmp, "sim.db")
        _make_sqlite_db(db_path)
        exp_id = self.tracker.register_experiment("export_test", {}, db_path)

        output_path = os.path.join(self.tmp, "output.csv")
        self.tracker.export_experiment_data(exp_id, output_path)
        self.assertTrue(os.path.exists(output_path))

        with open(output_path) as f:
            lines = f.readlines()
        # Header + 5 data rows
        self.assertEqual(len(lines), 6)

    def test_export_missing_experiment_raises(self):
        with self.assertRaises(ValueError):
            self.tracker.export_experiment_data("nonexistent_id", "/tmp/out.csv")


class TestCleanupOldExperiments(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.tracker = _make_tracker(self.tmp)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_cleanup_removes_old_experiments(self):
        db_path = os.path.join(self.tmp, "old.db")
        _make_sqlite_db(db_path)
        exp_id = self.tracker.register_experiment("old_run", {}, db_path)

        # Backdate the experiment timestamp
        old_time = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        self.tracker.metadata["experiments"][exp_id]["timestamp"] = old_time
        self.tracker._save_metadata()

        self.tracker.cleanup_old_experiments(days_old=30)
        self.assertNotIn(exp_id, self.tracker.metadata["experiments"])

    def test_cleanup_keeps_recent_experiments(self):
        db_path = os.path.join(self.tmp, "recent.db")
        _make_sqlite_db(db_path)
        exp_id = self.tracker.register_experiment("recent_run", {}, db_path)

        # Timestamp is just set (recent), should not be removed
        self.tracker.cleanup_old_experiments(days_old=30)
        self.assertIn(exp_id, self.tracker.metadata["experiments"])

    def test_cleanup_removes_db_file(self):
        db_path = os.path.join(self.tmp, "to_delete.db")
        _make_sqlite_db(db_path)
        exp_id = self.tracker.register_experiment("old_run2", {}, db_path)

        old_time = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        self.tracker.metadata["experiments"][exp_id]["timestamp"] = old_time
        self.tracker._save_metadata()

        self.tracker.cleanup_old_experiments(days_old=30)
        self.assertFalse(os.path.exists(db_path))

    def test_cleanup_missing_db_file_does_not_raise(self):
        exp_id = self.tracker.register_experiment("ghost", {}, "/nonexistent/path.db")
        old_time = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        self.tracker.metadata["experiments"][exp_id]["timestamp"] = old_time
        self.tracker._save_metadata()

        # Should not raise even if db file is missing
        self.tracker.cleanup_old_experiments(days_old=30)
        self.assertNotIn(exp_id, self.tracker.metadata["experiments"])


class TestSaveMetadataBackup(unittest.TestCase):
    def test_backup_created_and_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            tracker = _make_tracker(tmp)
            tracker.register_experiment("run", {}, "db.db")
            # After save, there should be no leftover .bak file
            backup_path = tracker.metadata_file + ".bak"
            self.assertFalse(os.path.exists(backup_path))
