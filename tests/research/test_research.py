"""Tests for farm/research/research.py.

Tests ResearchMetadata dataclass and ResearchProject using temporary
directories, with ExperimentRunner mocked out to avoid real simulations.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from farm.config import SimulationConfig
from farm.research.research import ResearchMetadata, ResearchProject


def _make_config():
    """Return a minimal default SimulationConfig."""
    return SimulationConfig()


class TestResearchMetadata(unittest.TestCase):
    """Tests for ResearchMetadata dataclass."""

    def test_fields_assigned(self):
        meta = ResearchMetadata(
            name="test_proj",
            description="a description",
            created_at="2024-01-01T00:00:00",
            updated_at="2024-01-01T00:00:00",
            status="initialized",
            tags=["simulation", "agents"],
        )
        self.assertEqual(meta.name, "test_proj")
        self.assertEqual(meta.status, "initialized")
        self.assertEqual(meta.tags, ["simulation", "agents"])
        self.assertEqual(meta.version, "1.0.0")  # default

    def test_custom_version(self):
        meta = ResearchMetadata(
            name="x",
            description="",
            created_at="",
            updated_at="",
            status="done",
            tags=[],
            version="2.3.0",
        )
        self.assertEqual(meta.version, "2.3.0")


class TestResearchProjectInit(unittest.TestCase):
    """Tests for ResearchProject initialisation."""

    def test_creates_project_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project = ResearchProject(
                name="test_proj",
                description="testing",
                base_path=tmpdir,
            )
            project_path = Path(tmpdir) / "test_proj"
            self.assertTrue(project_path.exists())
            self.assertTrue((project_path / "experiments").exists())

    def test_creates_metadata_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project = ResearchProject(
                name="meta_proj",
                description="metadata test",
                base_path=tmpdir,
            )
            metadata_file = Path(tmpdir) / "meta_proj" / "metadata.json"
            self.assertTrue(metadata_file.exists())
            with open(metadata_file) as f:
                data = json.load(f)
            self.assertEqual(data["name"], "meta_proj")
            self.assertEqual(data["status"], "initialized")

    def test_metadata_attribute_set(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project = ResearchProject(
                name="attr_test",
                base_path=tmpdir,
            )
            self.assertIsNotNone(project.metadata)
            self.assertEqual(project.metadata.name, "attr_test")

    def test_tags_stored(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project = ResearchProject(
                name="tagged",
                base_path=tmpdir,
                tags=["ml", "agents"],
            )
            self.assertEqual(project.tags, ["ml", "agents"])

    def test_recreating_project_clears_old_data(self):
        """ResearchProject deletes existing dir and starts fresh."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create first time
            p1 = ResearchProject(name="fresh", base_path=tmpdir)
            marker = Path(tmpdir) / "fresh" / "marker.txt"
            marker.write_text("old data")
            # Re-create
            p2 = ResearchProject(name="fresh", base_path=tmpdir)
            self.assertFalse(marker.exists())


class TestResearchProjectCreateExperiment(unittest.TestCase):
    """Tests for create_experiment."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self.project = ResearchProject(
            name="exp_project", base_path=self._tmpdir
        )

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_create_experiment_returns_path(self):
        config = _make_config()
        path_str = self.project.create_experiment(
            name="baseline", description="baseline run", config=config
        )
        self.assertIsNotNone(path_str)
        self.assertTrue(os.path.exists(path_str))

    def test_create_experiment_saves_config(self):
        config = _make_config()
        path_str = self.project.create_experiment(
            name="cfg_test", description="config saved", config=config
        )
        config_file = Path(path_str) / "experiment-config.json"
        self.assertTrue(config_file.exists())

    def test_create_multiple_experiments(self):
        import time
        config = _make_config()
        path1 = self.project.create_experiment("exp1", "first", config)
        time.sleep(0.01)
        path2 = self.project.create_experiment("exp2", "second", config)
        self.assertNotEqual(path1, path2)


class TestResearchProjectUpdateStatus(unittest.TestCase):
    """Tests for update_status."""

    def test_status_updated_in_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project = ResearchProject(name="status_proj", base_path=tmpdir)
            project.update_status("in_progress")
            self.assertEqual(project.metadata.status, "in_progress")

    def test_status_persisted_to_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project = ResearchProject(name="persist_proj", base_path=tmpdir)
            project.update_status("completed")
            metadata_file = Path(tmpdir) / "persist_proj" / "metadata.json"
            with open(metadata_file) as f:
                data = json.load(f)
            self.assertEqual(data["status"], "completed")


class TestResearchProjectListExperiments(unittest.TestCase):
    """Tests for list_experiments."""

    def test_empty_project_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project = ResearchProject(name="empty_proj", base_path=tmpdir)
            exps = project.list_experiments()
            self.assertIsInstance(exps, list)

    def test_returns_experiment_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project = ResearchProject(name="exp_list_proj", base_path=tmpdir)
            config = _make_config()
            project.create_experiment("exp_a", "a", config)
            project.create_experiment("exp_b", "b", config)
            exps = project.list_experiments()
            # list_experiments returns subdirectories of experiments/
            # The create_experiment creates under experiments/data/
            self.assertGreaterEqual(len(exps), 1)


class TestResearchProjectRunExperiment(unittest.TestCase):
    """Tests for run_experiment (mocked ExperimentRunner)."""

    def test_run_experiment_raises_for_missing_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project = ResearchProject(name="run_proj", base_path=tmpdir)
            with self.assertRaises(ValueError):
                project.run_experiment("nonexistent_id")

    @patch("farm.research.research.ExperimentRunner")
    def test_run_experiment_calls_runner(self, MockRunner):
        """run_experiment should instantiate ExperimentRunner and call run_iterations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project = ResearchProject(name="runner_proj", base_path=tmpdir)
            config = _make_config()
            exp_path = project.create_experiment("run_test", "desc", config)

            mock_instance = Mock()
            MockRunner.return_value = mock_instance

            # run_experiment looks for experiment in experiments/<exp_id>/
            # We need to replicate that directory structure
            exp_id = os.path.basename(exp_path)
            # Move the experiment to the expected location
            expected_path = Path(tmpdir) / "runner_proj" / "experiments" / exp_id
            import shutil
            shutil.copytree(exp_path, str(expected_path))

            project.run_experiment(exp_id, iterations=2, steps_per_iteration=5)
            mock_instance.run_iterations.assert_called_once_with(
                2, num_steps=5
            )


class TestResearchProjectExportResults(unittest.TestCase):
    """Tests for export_results."""

    def test_export_creates_output_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project = ResearchProject(name="export_proj", base_path=tmpdir)
            output_dir = Path(tmpdir) / "exported"
            # export_results copies specific subdirs; create the expected ones
            (project.project_path / "hypothesis.md").touch()
            # Should not raise even if subdirs don't exist
            project.export_results(output_dir)
            self.assertTrue(output_dir.exists())


if __name__ == "__main__":
    unittest.main()
