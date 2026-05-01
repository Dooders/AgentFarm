"""Unit tests for ``EnvironmentalGridManager`` and ``TemporalGridManager``."""

import os
import tempfile
import unittest

import numpy as np

from farm.config.config import MemmapConfig
from farm.core.environment_grids import (
    ENVIRONMENTAL_LAYER_NAMES,
    EnvironmentalGridManager,
)
from farm.core.temporal_grids import (
    DEFAULT_TEMPORAL_CHANNEL_SPECS,
    TemporalGridManager,
)


def _ram_config() -> MemmapConfig:
    return MemmapConfig()  # all toggles off


def _memmap_config(tmpdir: str, *, env: bool = False, temp: bool = False) -> MemmapConfig:
    return MemmapConfig(
        directory=tmpdir,
        dtype="float32",
        mode="w+",
        use_for_environmental=env,
        use_for_temporal=temp,
    )


class TestEnvironmentalGridManagerRam(unittest.TestCase):
    def setUp(self):
        self.mgr = EnvironmentalGridManager(
            height=10, width=8, memmap_config=_ram_config()
        )

    def tearDown(self):
        self.mgr.close()

    def test_default_layers_registered(self):
        self.assertEqual(set(self.mgr.names()), set(ENVIRONMENTAL_LAYER_NAMES))

    def test_set_get_window(self):
        grid = np.zeros((10, 8), dtype=np.float32)
        grid[5, 3] = 0.7
        self.mgr.set("OBSTACLES", grid)
        win = self.mgr.get_window("OBSTACLES", 4, 7, 2, 5)
        self.assertEqual(win.shape, (3, 3))
        self.assertAlmostEqual(float(win[1, 1]), 0.7)

    def test_set_wrong_shape_raises(self):
        with self.assertRaises(ValueError):
            self.mgr.set("OBSTACLES", np.zeros((4, 4), dtype=np.float32))

    def test_get_window_pads_outside_world(self):
        win = self.mgr.get_window("OBSTACLES", -2, 2, -2, 2)
        self.assertEqual(win.shape, (4, 4))
        self.assertTrue(np.all(win == 0))

    def test_unknown_layer_raises(self):
        with self.assertRaises(KeyError):
            self.mgr.get("MYSTERY")

    def test_has_memmap_false(self):
        self.assertFalse(self.mgr.has_memmap)


class TestEnvironmentalGridManagerMemmap(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="env_grid_test_")
        self.mgr = EnvironmentalGridManager(
            height=8,
            width=8,
            memmap_config=_memmap_config(self.tmpdir, env=True),
            simulation_id="env_test",
        )

    def tearDown(self):
        self.mgr.close(delete_files=True)
        # remove leftover files just in case
        for fname in os.listdir(self.tmpdir):
            try:
                os.remove(os.path.join(self.tmpdir, fname))
            except OSError:
                pass
        os.rmdir(self.tmpdir)

    def test_has_memmap_true(self):
        self.assertTrue(self.mgr.has_memmap)

    def test_round_trip_via_window(self):
        grid = np.arange(64, dtype=np.float32).reshape(8, 8) / 64.0
        self.mgr.set("TERRAIN_COST", grid)
        win = self.mgr.get_window("TERRAIN_COST", 0, 8, 0, 8)
        self.assertTrue(np.allclose(win, grid))

    def test_total_size_bytes_matches_disk(self):
        # 3 layers * 8 * 8 * 4 bytes
        self.assertEqual(self.mgr.total_size_bytes(), 3 * 8 * 8 * 4)


class TestTemporalGridManagerRam(unittest.TestCase):
    def setUp(self):
        self.mgr = TemporalGridManager(height=10, width=10, memmap_config=_ram_config())

    def tearDown(self):
        self.mgr.close()

    def test_default_channels_registered(self):
        names = set(self.mgr.channel_names())
        for spec in DEFAULT_TEMPORAL_CHANNEL_SPECS:
            self.assertIn(spec.name, names)

    def test_deposit_and_window(self):
        self.mgr.deposit("DAMAGE_HEAT", [(5, 5, 0.4), (7, 1, 0.2)])
        win = self.mgr.get_window("DAMAGE_HEAT", 4, 7, 4, 7)
        self.assertEqual(win.shape, (3, 3))
        self.assertAlmostEqual(float(win[1, 1]), 0.4)

    def test_deposit_out_of_bounds_ignored(self):
        self.mgr.deposit("TRAILS", [(100, 100, 1.0)])
        self.assertTrue(np.all(self.mgr.get("TRAILS") == 0))

    def test_apply_decay_uses_default_gamma(self):
        self.mgr.deposit("DAMAGE_HEAT", [(2, 2, 1.0)])
        self.mgr.apply_decay("DAMAGE_HEAT")
        spec = next(s for s in DEFAULT_TEMPORAL_CHANNEL_SPECS if s.name == "DAMAGE_HEAT")
        self.assertAlmostEqual(float(self.mgr.get("DAMAGE_HEAT")[2, 2]), spec.default_gamma, places=5)

    def test_apply_decay_all_channels(self):
        for name in self.mgr.channel_names():
            self.mgr.deposit(name, [(0, 0, 1.0)])
        self.mgr.apply_decay()
        for spec in DEFAULT_TEMPORAL_CHANNEL_SPECS:
            self.assertAlmostEqual(
                float(self.mgr.get(spec.name)[0, 0]), spec.default_gamma, places=5
            )

    def test_clip_max(self):
        self.mgr.deposit("DAMAGE_HEAT", [(3, 3, 5.0)])
        self.assertEqual(float(self.mgr.get("DAMAGE_HEAT")[3, 3]), 1.0)

    def test_overwrite_mode(self):
        self.mgr.deposit("DAMAGE_HEAT", [(3, 3, 0.4)])
        self.mgr.deposit("DAMAGE_HEAT", [(3, 3, 0.2)], accumulate=False)
        self.assertAlmostEqual(float(self.mgr.get("DAMAGE_HEAT")[3, 3]), 0.2)


class TestTemporalGridManagerMemmap(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="temporal_grid_test_")
        self.mgr = TemporalGridManager(
            height=6,
            width=6,
            memmap_config=_memmap_config(self.tmpdir, temp=True),
            simulation_id="temp_test",
        )

    def tearDown(self):
        self.mgr.close(delete_files=True)
        for fname in os.listdir(self.tmpdir):
            try:
                os.remove(os.path.join(self.tmpdir, fname))
            except OSError:
                pass
        os.rmdir(self.tmpdir)

    def test_has_memmap_true(self):
        self.assertTrue(self.mgr.has_memmap)

    def test_filenames_include_temporal_namespace(self):
        # one .dat per channel
        names = os.listdir(self.tmpdir)
        for spec in DEFAULT_TEMPORAL_CHANNEL_SPECS:
            self.assertTrue(
                any(spec.storage_name in n for n in names),
                msg=f"expected file containing '{spec.storage_name}', got {names}",
            )
            self.assertTrue(any(n.startswith("temporal_") for n in names))

    def test_decay_persists_across_flush(self):
        self.mgr.deposit("TRAILS", [(2, 2, 1.0)])
        spec = next(s for s in DEFAULT_TEMPORAL_CHANNEL_SPECS if s.name == "TRAILS")
        self.mgr.apply_decay("TRAILS")
        self.mgr.flush()
        # Underlying file should reflect decayed value (re-read directly)
        path = None
        for fname in os.listdir(self.tmpdir):
            if "trails" in fname:
                path = os.path.join(self.tmpdir, fname)
                break
        self.assertIsNotNone(path)
        # Open as memmap with same shape
        arr = np.memmap(path, dtype="float32", mode="r", shape=(6, 6))
        self.assertAlmostEqual(float(arr[2, 2]), spec.default_gamma, places=5)


if __name__ == "__main__":
    unittest.main()
