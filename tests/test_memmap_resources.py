import os
import re
import tempfile
import unittest

import numpy as np

from farm.core.config import SimulationConfig
from farm.core.environment import Environment


class TestMemmapResources(unittest.TestCase):
    def setUp(self):
        self.tmpdir = os.getenv("TMPDIR", tempfile.gettempdir())

        self.cfg = SimulationConfig(
            width=60,
            height=60,
            initial_resources=200,
            max_resource_amount=30,
            use_memmap_resources=True,
            memmap_dir=self.tmpdir,
            memmap_dtype="float32",
            memmap_mode="w+",
            seed=12345,
        )
        self.env = Environment(
            width=self.cfg.width,
            height=self.cfg.height,
            resource_distribution={},
            config=self.cfg,
            db_path=":memory:",
        )

    def tearDown(self):
        if hasattr(self, "env") and self.env:
            # ensure memmap file is flushed but retained
            self.env.close()

    def test_memmap_created_and_window_shape(self):
        rm = self.env.resource_manager
        self.assertTrue(rm.has_memmap)
        path = getattr(rm, "_memmap_path", None)
        self.assertIsNotNone(path)
        self.assertTrue(os.path.exists(path))

        R = self.env.observation_config.R
        ay, ax = self.cfg.height // 2, self.cfg.width // 2
        win = rm.get_resource_window(ay - R, ay + R + 1, ax - R, ax + R + 1)
        self.assertEqual(win.shape, (2 * R + 1, 2 * R + 1))
        self.assertTrue(np.all(win >= 0.0))
        self.assertTrue(np.all(win <= 1.0))

    def test_observation_uses_memmap_without_spatial_query(self):
        # Make spatial_index.get_nearby raise if called; memmap path should not call it
        def boom(*args, **kwargs):
            raise AssertionError("spatial_index.get_nearby should not be called when memmap is enabled")

        self.env.spatial_index.get_nearby = boom  # type: ignore

        # Add an agent and request observation
        from farm.core.agent import BaseAgent

        agent = BaseAgent(
            agent_id=self.env.get_next_agent_id(),
            position=(int(self.cfg.width // 2), int(self.cfg.height // 2)),
            resource_level=5,
            environment=self.env,
            spatial_service=self.env.spatial_service,
        )
        self.env.add_agent(agent)

        obs = self.env.observe(agent.agent_id)
        obs_space = self.env.observation_space(agent.agent_id)
        self.assertIsInstance(obs, np.ndarray)
        self.assertEqual(obs.shape, obs_space.shape)
        self.assertEqual(obs.dtype, obs_space.dtype)

    def test_close_flushes_but_does_not_delete(self):
        rm = self.env.resource_manager
        path = getattr(rm, "_memmap_path", None)
        self.assertIsNotNone(path)
        self.env.close()
        # File should still exist by default (delete=False)
        self.assertTrue(path and os.path.exists(path))
        # Now delete explicitly
        rm.cleanup_memmap(delete=True)
        self.assertFalse(os.path.exists(path))

    def test_filename_contains_pid_and_simulation_id(self):
        rm = self.env.resource_manager
        path = getattr(rm, "_memmap_path", "")
        base = os.path.basename(path)
        # must contain _p<digits>
        self.assertRegex(base, r"_p\d+")
        # simulation id string should appear in name
        sim_id = self.env.simulation_id
        self.assertIn(sim_id, base)


if __name__ == "__main__":
    unittest.main()

