import os
import re
import tempfile
import unittest

import numpy as np

from farm.config import SimulationConfig
from farm.core.environment import Environment


class TestMemmapResources(unittest.TestCase):
    def setUp(self):
        self.tmpdir = os.getenv("TMPDIR", tempfile.gettempdir())

        from farm.config.config import EnvironmentConfig, ResourceConfig

        self.cfg = SimulationConfig(
            environment=EnvironmentConfig(width=60, height=60),
            resources=ResourceConfig(initial_resources=200, max_resource_amount=30),
            seed=12345,
        )

        # Add memmap parameters dynamically (accessed via getattr in ResourceManager)
        self.cfg.use_memmap_resources = True
        self.cfg.memmap_dir = self.tmpdir
        self.cfg.memmap_dtype = "float32"
        self.cfg.memmap_mode = "w+"
        self.env = Environment(
            width=self.cfg.environment.width,
            height=self.cfg.environment.height,
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
        ay, ax = self.cfg.environment.height // 2, self.cfg.environment.width // 2
        win = rm.get_resource_window(ay - R, ay + R + 1, ax - R, ax + R + 1)
        self.assertEqual(win.shape, (2 * R + 1, 2 * R + 1))
        self.assertTrue(np.all(win >= 0.0))
        self.assertTrue(np.all(win <= 1.0))

    def test_observation_uses_memmap_without_spatial_query(self):
        # Track calls to spatial_index.get_nearby to ensure resources don't use spatial queries
        original_get_nearby = self.env.spatial_index.get_nearby
        calls = []

        def track_calls(*args, **kwargs):
            calls.append(args)
            return original_get_nearby(*args, **kwargs)

        self.env.spatial_index.get_nearby = track_calls

        # Add an agent and request observation
        from farm.core.agent import AgentFactory

        factory = AgentFactory(spatial_service=self.env.spatial_service)
        agent = factory.create_default_agent(
            agent_id=self.env.get_next_agent_id(),
            position=(int(self.cfg.environment.width // 2), int(self.cfg.environment.height // 2)),
            initial_resources=5,
        )
        self.env.add_agent(agent)

        obs = self.env.observe(agent.agent_id)
        obs_space = self.env.observation_space(agent.agent_id)
        self.assertIsInstance(obs, np.ndarray)
        self.assertEqual(obs.shape, obs_space.shape)
        self.assertEqual(obs.dtype, obs_space.dtype)

        # Verify that spatial queries were made for agents but NOT for resources
        resource_queries = [call for call in calls if len(call) >= 3 and "resources" in call[2]]
        self.assertEqual(len(resource_queries), 0, "Resource observation should not use spatial queries when memmap is enabled")

    def test_close_flushes_but_does_not_delete(self):
        rm = self.env.resource_manager
        path = getattr(rm, "_memmap_path", None)
        self.assertIsNotNone(path)
        self.env.close()
        # File should still exist by default (delete=False)
        self.assertTrue(path and os.path.exists(path))
        # Now delete explicitly
        rm.cleanup_memmap(delete_file=True)
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

