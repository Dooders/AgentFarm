"""Tests for config watcher module (ConfigWatcher and ReloadableConfig)."""

import os
import tempfile
import threading
import time
import unittest

import yaml

from farm.config import SimulationConfig
from farm.config.watcher import (
    ConfigWatcher,
    ReloadableConfig,
    create_reloadable_config,
    get_global_watcher,
    watch_config_file,
)


def _write_config(filepath: str, simulation_steps: int = 10) -> None:
    """Write a minimal SimulationConfig YAML to a file."""
    config = SimulationConfig(simulation_steps=simulation_steps)
    config.to_yaml(filepath)


class TestConfigWatcher(unittest.TestCase):
    """Tests for ConfigWatcher class."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # Use a very short interval for tests
        self.watcher = ConfigWatcher(watch_interval=0.05)

    def tearDown(self):
        if self.watcher.running:
            self.watcher.stop()
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _config_path(self, name: str = "config.yaml") -> str:
        return os.path.join(self.tmpdir, name)

    def test_get_file_hash_existing_file(self):
        """_get_file_hash returns a non-empty string for an existing file."""
        p = self._config_path()
        _write_config(p)
        h = self.watcher._get_file_hash(p)
        self.assertIsInstance(h, str)
        self.assertGreater(len(h), 0)

    def test_get_file_hash_missing_file_raises(self):
        """_get_file_hash raises FileNotFoundError for missing file."""
        with self.assertRaises(FileNotFoundError):
            self.watcher._get_file_hash(os.path.join(self.tmpdir, "no_such.yaml"))

    def test_watch_file_registers_callback(self):
        """watch_file registers a callback for the file."""
        p = self._config_path()
        _write_config(p)
        cb = lambda config: None
        self.watcher.watch_file(p, cb)
        watched = self.watcher.get_watched_files()
        self.assertIn(os.path.realpath(p), watched)

    def test_watch_file_missing_file(self):
        """watch_file on a non-existent file stores empty hash."""
        p = self._config_path("missing.yaml")
        cb = lambda config: None
        self.watcher.watch_file(p, cb)
        watched = self.watcher.get_watched_files()
        self.assertIn(os.path.realpath(p), watched)

    def test_unwatch_file_removes_callback(self):
        """unwatch_file removes the callback and the file entry."""
        p = self._config_path()
        _write_config(p)
        cb = lambda config: None
        self.watcher.watch_file(p, cb)
        self.watcher.unwatch_file(p, cb)
        watched = self.watcher.get_watched_files()
        self.assertNotIn(os.path.realpath(p), watched)

    def test_unwatch_file_all_callbacks(self):
        """unwatch_file with no callback removes all callbacks."""
        p = self._config_path()
        _write_config(p)
        self.watcher.watch_file(p, lambda config: None)
        self.watcher.watch_file(p, lambda config: None)
        self.watcher.unwatch_file(p)
        watched = self.watcher.get_watched_files()
        self.assertNotIn(os.path.realpath(p), watched)

    def test_start_and_stop(self):
        """Watcher thread starts and stops cleanly."""
        self.watcher.start()
        self.assertTrue(self.watcher.running)
        self.assertIsNotNone(self.watcher.thread)
        self.watcher.stop()
        self.assertFalse(self.watcher.running)

    def test_start_idempotent(self):
        """Calling start() twice does not create a second thread."""
        self.watcher.start()
        thread_id = id(self.watcher.thread)
        self.watcher.start()
        self.assertEqual(id(self.watcher.thread), thread_id)
        self.watcher.stop()

    def test_get_watched_files_returns_copy(self):
        """get_watched_files returns a copy, not internal state."""
        p = self._config_path()
        _write_config(p)
        self.watcher.watch_file(p, lambda config: None)
        watched = self.watcher.get_watched_files()
        watched["fake_key"] = "fake_value"
        # Original should not be modified
        self.assertNotIn("fake_key", self.watcher.get_watched_files())

    def test_callback_triggered_on_file_change(self):
        """Callback is triggered when file content changes."""
        p = self._config_path()
        _write_config(p, simulation_steps=10)

        received = []
        event = threading.Event()

        def cb(config):
            received.append(config.simulation_steps)
            event.set()

        self.watcher.watch_file(p, cb)
        self.watcher.start()

        # Modify the file
        time.sleep(0.1)
        _write_config(p, simulation_steps=99)

        # Wait for callback with timeout
        triggered = event.wait(timeout=2.0)
        self.watcher.stop()

        self.assertTrue(triggered, "Callback was not triggered after file change")
        self.assertIn(99, received)

    def test_no_callback_on_unchanged_file(self):
        """Callback is NOT triggered when file does not change."""
        p = self._config_path()
        _write_config(p)

        called = []
        self.watcher.watch_file(p, lambda config: called.append(True))
        self.watcher.start()

        time.sleep(0.3)
        self.watcher.stop()

        self.assertEqual(called, [], "Callback was incorrectly triggered without file change")


class TestReloadableConfig(unittest.TestCase):
    """Tests for ReloadableConfig class."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _config_path(self, name: str = "config.yaml") -> str:
        return os.path.join(self.tmpdir, name)

    def test_config_property_returns_config(self):
        """config property returns the underlying SimulationConfig."""
        config = SimulationConfig(simulation_steps=5)
        rc = ReloadableConfig(config)
        self.assertIs(rc.config, config)

    def test_attribute_delegation(self):
        """Attribute access is delegated to the underlying config."""
        config = SimulationConfig(simulation_steps=42)
        rc = ReloadableConfig(config)
        self.assertEqual(rc.simulation_steps, 42)

    def test_repr(self):
        """repr() returns a useful string."""
        config = SimulationConfig()
        rc = ReloadableConfig(config)
        self.assertIn("ReloadableConfig", repr(rc))

    def test_reload_from_file(self):
        """Manual reload updates config from file."""
        p = self._config_path()
        _write_config(p, simulation_steps=10)

        config = SimulationConfig.from_yaml(p)
        rc = ReloadableConfig(config)

        _write_config(p, simulation_steps=77)
        rc.reload_from_file(p)

        self.assertEqual(rc.config.simulation_steps, 77)

    def test_change_callback_called_on_reload(self):
        """Change callbacks are called when reload_from_file is used."""
        p = self._config_path()
        _write_config(p, simulation_steps=10)

        config = SimulationConfig.from_yaml(p)
        rc = ReloadableConfig(config)

        received = []
        rc.add_change_callback(lambda config: received.append(config.simulation_steps))

        _write_config(p, simulation_steps=55)
        rc.reload_from_file(p)

        self.assertIn(55, received)

    def test_remove_change_callback(self):
        """Removed callbacks are not called."""
        config = SimulationConfig()
        p = self._config_path()
        _write_config(p)
        rc = ReloadableConfig(config)

        called = []
        cb = lambda config: called.append(True)
        rc.add_change_callback(cb)
        rc.remove_change_callback(cb)

        rc.reload_from_file(p)
        self.assertEqual(called, [])

    def test_stop_watching(self):
        """stop_watching stops the watcher."""
        p = self._config_path()
        _write_config(p)

        # Use a dedicated watcher for this test
        watcher = ConfigWatcher(watch_interval=0.05)
        config = SimulationConfig.from_yaml(p)
        rc = ReloadableConfig(config, watcher=watcher)
        rc.watch_file(p)
        self.assertTrue(rc._watcher_started)

        rc.stop_watching()
        self.assertFalse(rc._watcher_started)


class TestCreateReloadableConfig(unittest.TestCase):
    """Tests for create_reloadable_config factory function."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        # Stop global watcher if started
        from farm.config.watcher import _global_watcher
        if _global_watcher.running:
            _global_watcher.stop()

    def test_create_from_config_object(self):
        """create_reloadable_config accepts a SimulationConfig."""
        config = SimulationConfig(simulation_steps=10)
        rc = create_reloadable_config(config)
        self.assertIsInstance(rc, ReloadableConfig)
        self.assertEqual(rc.config.simulation_steps, 10)

    def test_create_from_path(self):
        """create_reloadable_config accepts a file path."""
        p = os.path.join(self.tmpdir, "cfg.yaml")
        _write_config(p, simulation_steps=25)
        rc = create_reloadable_config(p)
        self.assertIsInstance(rc, ReloadableConfig)
        self.assertEqual(rc.config.simulation_steps, 25)


class TestGetGlobalWatcher(unittest.TestCase):
    """Tests for get_global_watcher."""

    def test_returns_watcher_instance(self):
        watcher = get_global_watcher()
        self.assertIsInstance(watcher, ConfigWatcher)

    def test_returns_same_instance(self):
        w1 = get_global_watcher()
        w2 = get_global_watcher()
        self.assertIs(w1, w2)


if __name__ == "__main__":
    unittest.main()
