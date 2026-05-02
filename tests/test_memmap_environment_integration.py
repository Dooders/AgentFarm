"""Integration tests verifying ``Environment`` uses the memmap-backed grids.

These tests exercise the full environment integration of the new
:class:`EnvironmentalGridManager` and :class:`TemporalGridManager` introduced
to extend memory-mapped arrays beyond the resource grid (issue #426).
"""

import os
import tempfile
import unittest

import numpy as np

from farm.config import SimulationConfig
from farm.config.config import EnvironmentConfig, MemmapConfig, ResourceConfig
from farm.core.environment import Environment


class TestEnvironmentMemmapIntegration(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="env_memmap_int_")
        self.cfg = SimulationConfig(
            environment=EnvironmentConfig(width=32, height=32),
            resources=ResourceConfig(initial_resources=20, max_resource_amount=10),
            memmap=MemmapConfig(
                directory=self.tmpdir,
                dtype="float32",
                mode="w+",
                use_for_resources=True,
                use_for_environmental=True,
                use_for_temporal=True,
                delete_on_close=True,
            ),
            seed=42,
        )
        self.env = Environment(
            width=self.cfg.environment.width,
            height=self.cfg.environment.height,
            resource_distribution={},
            config=self.cfg,
            db_path=":memory:",
        )

    def tearDown(self):
        try:
            self.env.close()
        finally:
            for fname in os.listdir(self.tmpdir):
                try:
                    os.remove(os.path.join(self.tmpdir, fname))
                except OSError:
                    # Best-effort teardown cleanup: files may already be gone or
                    # temporarily unavailable on some platforms.
                    pass
            os.rmdir(self.tmpdir)

    def test_environment_provisions_memmap_grids(self):
        self.assertTrue(self.env.environmental_grids.has_memmap)
        self.assertTrue(self.env.temporal_grids.has_memmap)
        self.assertTrue(self.env.resource_manager.has_memmap)

    def test_set_environmental_layer_round_trip(self):
        grid = np.zeros((32, 32), dtype=np.float32)
        grid[10, 12] = 1.0
        grid[15, 20] = 0.5
        self.env.set_environmental_layer("OBSTACLES", grid)
        win = self.env.environmental_grids.get_window("OBSTACLES", 8, 17, 10, 22)
        self.assertEqual(win.shape, (9, 12))
        self.assertAlmostEqual(float(win[2, 2]), 1.0)
        self.assertAlmostEqual(float(win[7, 10]), 0.5)

    def test_deposit_temporal_event_visible_in_observation_pipeline(self):
        # Place a synthetic damage event near (16, 16) which we'll observe
        # from an agent positioned at (16, 16).
        self.env.deposit_temporal_events(
            "DAMAGE_HEAT", [(16, 16, 0.9), (18, 14, 0.4)]
        )

        from farm.core.agent import AgentFactory, AgentServices

        services = AgentServices(
            spatial_service=self.env.spatial_service,
            time_service=getattr(self.env, "time_service", None),
            metrics_service=getattr(self.env, "metrics_service", None),
            logging_service=getattr(self.env, "logging_service", None),
            validation_service=getattr(self.env, "validation_service", None),
            lifecycle_service=getattr(self.env, "lifecycle_service", None),
        )
        factory = AgentFactory(services)
        agent = factory.create_default_agent(
            agent_id=self.env.get_next_agent_id(),
            position=(16, 16),
            initial_resources=5,
            environment=self.env,
        )
        self.env.add_agent(agent)

        obs = self.env.observe(agent.agent_id)
        # Channel 8 is DAMAGE_HEAT in the dynamic registry; assert the
        # agent center (R, R) has the deposited value reflected.
        from farm.core.channels import Channel

        R = self.env.observation_config.R
        self.assertAlmostEqual(
            float(obs[Channel.DAMAGE_HEAT, R, R]), 0.9, places=4
        )

    def test_temporal_grid_decays_on_environment_update(self):
        from farm.core.temporal_grids import DEFAULT_TEMPORAL_CHANNEL_SPECS

        spec = next(
            s for s in DEFAULT_TEMPORAL_CHANNEL_SPECS if s.name == "DAMAGE_HEAT"
        )
        self.env.deposit_temporal_events("DAMAGE_HEAT", [(5, 5, 1.0)])
        before = float(self.env.temporal_grids.get("DAMAGE_HEAT")[5, 5])
        self.env.update()
        after = float(self.env.temporal_grids.get("DAMAGE_HEAT")[5, 5])
        self.assertAlmostEqual(after, before * spec.default_gamma, places=5)

    def test_reset_clears_temporal_activity_flags(self):
        self.env.deposit_temporal_events("ALLY_SIGNAL", [(6, 6, 0.7)])
        self.assertTrue(self.env.temporal_grids.has_any_data("ALLY_SIGNAL"))
        self.env.reset()
        self.assertFalse(self.env.temporal_grids.has_any_data("ALLY_SIGNAL"))

    def test_close_deletes_files_when_configured(self):
        env_paths = []
        if self.env.environmental_grids.has_memmap:
            env_paths.extend(
                self.env.environmental_grids._manager.info(name).path  # noqa: SLF001
                for name in self.env.environmental_grids.names()
            )
        for p in env_paths:
            self.assertTrue(os.path.exists(p))
        self.env.close()
        for p in env_paths:
            self.assertFalse(os.path.exists(p))

    def test_close_cleans_grid_memmaps_even_without_resource_manager(self):
        env_paths = []
        if self.env.environmental_grids.has_memmap:
            env_paths.extend(
                self.env.environmental_grids._manager.info(name).path  # noqa: SLF001
                for name in self.env.environmental_grids.names()
            )
        for p in env_paths:
            self.assertTrue(os.path.exists(p))
        # Simulate partial-init/failure state where resource manager is unavailable.
        self.env.resource_manager = None
        self.env.close()
        for p in env_paths:
            self.assertFalse(os.path.exists(p))


class TestEnvironmentMemmapDisabled(unittest.TestCase):
    """Environment should still function correctly with memmap disabled."""

    def setUp(self):
        self.cfg = SimulationConfig(
            environment=EnvironmentConfig(width=24, height=24),
            resources=ResourceConfig(initial_resources=5, max_resource_amount=10),
            memmap=MemmapConfig(),  # everything off
            seed=7,
        )
        self.env = Environment(
            width=24, height=24, resource_distribution={}, config=self.cfg, db_path=":memory:"
        )

    def tearDown(self):
        self.env.close()

    def test_grids_exist_in_ram_mode(self):
        self.assertFalse(self.env.environmental_grids.has_memmap)
        self.assertFalse(self.env.temporal_grids.has_memmap)
        self.assertFalse(self.env.resource_manager.has_memmap)

    def test_deposit_and_observe_works_without_memmap(self):
        self.env.deposit_temporal_events("ALLY_SIGNAL", [(12, 12, 0.6)])
        win = self.env.temporal_grids.get_window("ALLY_SIGNAL", 10, 15, 10, 15)
        self.assertEqual(win.shape, (5, 5))
        self.assertAlmostEqual(float(win[2, 2]), 0.6)


if __name__ == "__main__":
    unittest.main()
