"""Unit tests for ``farm.core.memmap_manager.MemmapManager``."""

import os
import tempfile
import unittest

import numpy as np

from farm.core.memmap_manager import MemmapManager, sanitize_for_filename


class TestSanitizeFilename(unittest.TestCase):
    def test_replaces_unsafe_chars(self):
        self.assertEqual(sanitize_for_filename("a/b c:d"), "a-b-c-d")

    def test_keeps_alnum_dash_underscore(self):
        self.assertEqual(sanitize_for_filename("Sim_42-A"), "Sim_42-A")

    def test_truncates(self):
        big = "x" * 200
        self.assertEqual(len(sanitize_for_filename(big, max_length=64)), 64)


class TestMemmapManager(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="memmap_mgr_test_")
        self.mgr = MemmapManager(
            directory=self.tmpdir,
            simulation_id="sim-001",
            namespace="ns",
            default_dtype="float32",
        )

    def tearDown(self):
        self.mgr.close_all(delete_files=True)
        # tmpdir may still contain leftover files from failed tests; remove
        # them to avoid pollution between cases.
        for fname in os.listdir(self.tmpdir):
            try:
                os.remove(os.path.join(self.tmpdir, fname))
            except FileNotFoundError:
                # Best-effort cleanup: file may already have been removed.
                continue
            except OSError as exc:
                # Ignore only "already removed" race conditions; surface
                # unexpected filesystem errors.
                if exc.errno != getattr(os, "ENOENT", 2):
                    raise
        os.rmdir(self.tmpdir)

    def test_create_initializes_zeros_and_registers(self):
        arr = self.mgr.create("alpha", (4, 5))
        self.assertEqual(arr.shape, (4, 5))
        self.assertTrue(self.mgr.has("alpha"))
        self.assertTrue(np.all(arr == 0))

    def test_create_duplicate_raises(self):
        self.mgr.create("alpha", (2, 2))
        with self.assertRaises(ValueError):
            self.mgr.create("alpha", (2, 2))

    def test_filename_includes_pid_simulation_id_namespace_and_shape(self):
        self.mgr.create("alpha", (3, 7))
        info = self.mgr.info("alpha")
        base = os.path.basename(info.path)
        self.assertIn("ns_", base)
        self.assertIn("alpha", base)
        self.assertIn("sim-001", base)
        self.assertIn(f"p{os.getpid()}", base)
        self.assertTrue(base.endswith("3x7.dat"))

    def test_get_window_zero_pads_outside_bounds(self):
        arr = self.mgr.create("g", (4, 4))
        arr[:] = 3.0
        arr.flush()
        win = self.mgr.get_window("g", -1, 3, -1, 3)
        # Top-left padded with zeros, rest filled with 3.0
        self.assertEqual(win.shape, (4, 4))
        self.assertEqual(win.dtype, np.float32)
        self.assertTrue(np.all(win[0, :] == 0))
        self.assertTrue(np.all(win[:, 0] == 0))
        self.assertTrue(np.all(win[1:, 1:] == 3.0))

    def test_get_window_normalize(self):
        arr = self.mgr.create("g", (2, 2))
        arr[:] = 5.0
        arr.flush()
        win = self.mgr.get_window("g", 0, 2, 0, 2, normalize_by=10.0)
        self.assertTrue(np.allclose(win, 0.5))
        # Clipped at 1.0
        arr[:] = 30.0
        arr.flush()
        win = self.mgr.get_window("g", 0, 2, 0, 2, normalize_by=10.0)
        self.assertTrue(np.all(win == 1.0))

    def test_window_returns_independent_copy(self):
        arr = self.mgr.create("g", (3, 3))
        arr[1, 1] = 9.0
        arr.flush()
        win = self.mgr.get_window("g", 0, 3, 0, 3)
        win[0, 0] = -1.0  # mutate copy
        # Underlying array is untouched.
        self.assertEqual(float(self.mgr.get("g")[0, 0]), 0.0)

    def test_scale_and_add_at(self):
        arr = self.mgr.create("g", (4, 4))
        self.mgr.add_at("g", 2, 2, 0.7)
        self.mgr.add_at("g", 2, 2, 0.5, clip_max=1.0)
        self.assertAlmostEqual(float(arr[2, 2]), 1.0)
        self.mgr.add_at("g", 999, 999, 1.0)  # OOB silently ignored
        self.mgr.scale("g", 0.5)
        self.assertAlmostEqual(float(arr[2, 2]), 0.5)

    def test_close_removes_file_when_requested(self):
        self.mgr.create("g", (2, 2))
        path = self.mgr.info("g").path
        self.assertTrue(os.path.exists(path))
        self.mgr.close("g", delete_file=True)
        self.assertFalse(os.path.exists(path))
        self.assertFalse(self.mgr.has("g"))

    def test_close_keeps_file_by_default(self):
        self.mgr.create("g", (2, 2))
        path = self.mgr.info("g").path
        self.mgr.close("g")
        self.assertTrue(os.path.exists(path))
        # Manually clean up.
        os.remove(path)

    def test_total_size_bytes(self):
        self.mgr.create("a", (4, 4))
        self.mgr.create("b", (2, 8))
        # Both float32 -> 16*4 + 16*4 == 128 bytes
        self.assertEqual(self.mgr.total_size_bytes(), 128)


if __name__ == "__main__":
    unittest.main()
