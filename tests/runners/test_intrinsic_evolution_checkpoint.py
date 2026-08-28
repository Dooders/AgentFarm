"""Tests for intrinsic-evolution mid-run checkpointing."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from farm.runners.intrinsic_evolution_checkpoint import (
    CHECKPOINT_VERSION,
    CheckpointMeta,
    clear_checkpoint,
    has_resumable_checkpoint,
    read_checkpoint_meta,
    restore_environment_from_checkpoint,
    save_checkpoint,
)
from scripts.run_stable_profile_seed_sweep import _prepare_run_dir_for_resume


class TestCheckpointMeta(unittest.TestCase):
    def test_save_and_detect_resumable(self):
        with TemporaryDirectory() as tmp:
            payload = {
                "version": CHECKPOINT_VERSION,
                "logical_step": 250,
                "num_steps_configured": 1000,
                "total_sim_steps": 1200,
                "effective_warmup": 200,
                "agents": [],
                "resources": [],
            }
            meta = save_checkpoint(tmp, payload)
            self.assertEqual(meta.logical_step, 250)
            self.assertTrue(has_resumable_checkpoint(tmp, num_steps=1000))
            loaded = read_checkpoint_meta(tmp)
            assert loaded is not None
            self.assertEqual(loaded.logical_step, 250)

    def test_complete_relative_to_warmup_not_resumable(self):
        with TemporaryDirectory() as tmp:
            payload = {
                "version": CHECKPOINT_VERSION,
                "logical_step": 1200,
                "num_steps_configured": 1000,
                "total_sim_steps": 1200,
                "effective_warmup": 200,
                "agents": [],
                "resources": [],
            }
            save_checkpoint(tmp, payload)
            # post-warmup completed == 1000 → not resumable as partial
            self.assertFalse(has_resumable_checkpoint(tmp, num_steps=1000))

    def test_clear_checkpoint(self):
        with TemporaryDirectory() as tmp:
            save_checkpoint(
                tmp,
                {
                    "version": CHECKPOINT_VERSION,
                    "logical_step": 10,
                    "num_steps_configured": 100,
                    "total_sim_steps": 100,
                    "effective_warmup": 0,
                    "agents": [],
                    "resources": [],
                },
            )
            clear_checkpoint(tmp)
            self.assertIsNone(read_checkpoint_meta(tmp))
            self.assertFalse(has_resumable_checkpoint(tmp, num_steps=100))


class TestPrepareRunDirKeepsCheckpoint(unittest.TestCase):
    def test_keeps_incomplete_with_checkpoint(self):
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "seed_7"
            run_dir.mkdir()
            (run_dir / "partial.db").write_text("x", encoding="utf-8")
            save_checkpoint(
                str(run_dir),
                {
                    "version": CHECKPOINT_VERSION,
                    "logical_step": 40,
                    "num_steps_configured": 100,
                    "total_sim_steps": 100,
                    "effective_warmup": 0,
                    "agents": [],
                    "resources": [],
                },
            )
            _prepare_run_dir_for_resume(run_dir, num_steps=100, resume=True)
            self.assertTrue(run_dir.exists())
            self.assertTrue((run_dir / "partial.db").exists())

    def test_still_removes_incomplete_without_checkpoint(self):
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "seed_9"
            run_dir.mkdir()
            (run_dir / "partial.db").write_text("x", encoding="utf-8")
            _prepare_run_dir_for_resume(run_dir, num_steps=100, resume=True)
            self.assertFalse(run_dir.exists())

    def test_keeps_complete_metadata(self):
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "seed_42"
            run_dir.mkdir()
            meta = {"num_steps_completed": 100, "final_population": 12}
            (run_dir / "intrinsic_evolution_metadata.json").write_text(
                json.dumps(meta), encoding="utf-8"
            )
            _prepare_run_dir_for_resume(run_dir, num_steps=100, resume=True)
            self.assertTrue(run_dir.exists())


class TestCheckpointMetaRoundTrip(unittest.TestCase):
    def test_from_dict(self):
        meta = CheckpointMeta.from_dict(
            {
                "version": 1,
                "logical_step": 5,
                "num_steps_configured": 50,
                "total_sim_steps": 50,
                "effective_warmup": 0,
            }
        )
        self.assertEqual(meta.to_dict()["logical_step"], 5)


class TestRestoreEnvironmentPostconditions(unittest.TestCase):
    """Regression tests for restore_environment_from_checkpoint postconditions.

    Each test verifies a specific bug-fix: the original implementation left
    ``cached_total_resources`` at zero, kept ``identity._agent_counter`` at
    zero (enabling ID collision on the next birth), dropped ``genome_id``/
    ``parent_ids``, and did not apply the chromosome to the live decision
    module before loading saved model state.
    """

    def _make_payload(self, *, resource_amount: float = 25.0, genome_id: str = "gid-abc"):
        """Build a minimal checkpoint payload with one resource and one agent."""
        from farm.core.hyperparameter_chromosome import default_hyperparameter_chromosome

        chromosome = default_hyperparameter_chromosome()
        return {
            "version": CHECKPOINT_VERSION,
            "logical_step": 50,
            "num_steps_configured": 100,
            "total_sim_steps": 100,
            "effective_warmup": 0,
            "simulation_id": "test-resume",
            "environment_time": 50,
            "resources": [
                {
                    "resource_id": 0,
                    "position": (1, 1),
                    "amount": resource_amount,
                    "max_amount": resource_amount,
                    "regeneration_rate": 0.1,
                }
            ],
            "agents": [
                {
                    "agent_id": "agent-1",
                    "agent_type": "AgentCore",
                    "position": (2, 2),
                    "resources": 5.0,
                    "health": 100.0,
                    "generation": 3,
                    "genome_id": genome_id,
                    "parent_ids": ["parent-0"],
                    "alive": True,
                    "chromosome": chromosome.to_dict(),
                    "model_state": None,
                    "decision_is_trained": False,
                }
            ],
            "rng": {},
            "inheritance_telemetry": {},
        }

    def _make_mock_agent(self):
        """Return a mock agent with minimal attributes needed by the restore path."""
        inner_state = SimpleNamespace(genome_id="", parent_ids=[])

        def model_copy(*, update):
            for k, v in update.items():
                setattr(inner_state, k, v)
            return inner_state

        inner_state.model_copy = model_copy

        agent = MagicMock()
        agent.agent_id = "agent-1"
        agent.genome_id = ""
        agent.state._state = inner_state
        agent.state._state.model_copy = model_copy
        agent.behavior.decision_module = None
        agent.get_component.return_value = None
        return agent

    def _build_mock_env(self, identity_counter: int = 0):
        env = MagicMock()
        env.resources = []
        env.resource_manager = MagicMock()
        env.resource_manager.resources = []
        env.cached_total_resources = 0.0
        env.db = None
        env.spatial_index = None
        env._agent_objects = {}
        env._alive_agents = set()
        env.agents = []
        env.time = 0
        identity = MagicMock()
        identity._agent_counter = identity_counter
        env.identity = identity
        return env

    @patch("farm.core.simulation.create_services_from_environment")
    @patch("farm.core.agent.factory.AgentFactory")
    @patch("farm.core.environment.Environment")
    def test_cached_total_resources_set_after_restore(
        self, mock_env_cls, mock_factory_cls, mock_services
    ):
        """cached_total_resources must equal sum of restored resource amounts."""
        resource_amount = 37.5
        payload = self._make_payload(resource_amount=resource_amount)
        env = self._build_mock_env()
        mock_env_cls.return_value = env

        mock_agent = self._make_mock_agent()
        mock_factory_cls.return_value.create_learning_agent.return_value = mock_agent

        def add_agent(agent, **kwargs):
            env._agent_objects[agent.agent_id] = agent

        env.add_agent.side_effect = add_agent

        from farm.config import SimulationConfig
        from farm.runners.intrinsic_evolution_experiment import IntrinsicEvolutionPolicy

        cfg = SimulationConfig()
        policy = IntrinsicEvolutionPolicy()
        import random

        restore_environment_from_checkpoint(
            payload,
            config=cfg,
            path=None,
            policy=policy,
            policy_rng=random.Random(0),
        )

        self.assertAlmostEqual(env.cached_total_resources, resource_amount, places=6)

    @patch("farm.core.simulation.create_services_from_environment")
    @patch("farm.core.agent.factory.AgentFactory")
    @patch("farm.core.environment.Environment")
    def test_identity_counter_advanced_after_restore(
        self, mock_env_cls, mock_factory_cls, mock_services
    ):
        """identity._agent_counter must be >= number of restored agents."""
        payload = self._make_payload()
        env = self._build_mock_env(identity_counter=0)
        mock_env_cls.return_value = env

        mock_agent = self._make_mock_agent()
        mock_factory_cls.return_value.create_learning_agent.return_value = mock_agent

        def add_agent(agent, **kwargs):
            env._agent_objects[agent.agent_id] = agent

        env.add_agent.side_effect = add_agent

        import random

        from farm.config import SimulationConfig
        from farm.runners.intrinsic_evolution_experiment import IntrinsicEvolutionPolicy

        restore_environment_from_checkpoint(
            payload,
            config=SimulationConfig(),
            path=None,
            policy=IntrinsicEvolutionPolicy(),
            policy_rng=random.Random(0),
        )

        # There is 1 restored agent; counter must be >= 1 to avoid collision.
        self.assertGreaterEqual(env.identity._agent_counter, 1)

    @patch("farm.core.simulation.create_services_from_environment")
    @patch("farm.core.agent.factory.AgentFactory")
    @patch("farm.core.environment.Environment")
    def test_genome_id_and_parent_ids_restored(
        self, mock_env_cls, mock_factory_cls, mock_services
    ):
        """genome_id and parent_ids must be written into agent.state._state."""
        genome_id = "gid-xyz-restored"
        payload = self._make_payload(genome_id=genome_id)
        env = self._build_mock_env()
        mock_env_cls.return_value = env

        mock_agent = self._make_mock_agent()
        captured_state: list = []

        def add_agent(agent, **kwargs):
            env._agent_objects[agent.agent_id] = agent
            captured_state.append(agent.state._state)

        env.add_agent.side_effect = add_agent
        mock_factory_cls.return_value.create_learning_agent.return_value = mock_agent

        import random

        from farm.config import SimulationConfig
        from farm.runners.intrinsic_evolution_experiment import IntrinsicEvolutionPolicy

        restore_environment_from_checkpoint(
            payload,
            config=SimulationConfig(),
            path=None,
            policy=IntrinsicEvolutionPolicy(),
            policy_rng=random.Random(0),
        )

        self.assertTrue(captured_state, "add_agent was not called")
        state = captured_state[0]
        self.assertEqual(state.genome_id, genome_id)
        self.assertEqual(state.parent_ids, ["parent-0"])

    @patch("farm.core.initial_diversity._apply_chromosome_to_agent")
    @patch("farm.core.simulation.create_services_from_environment")
    @patch("farm.core.agent.factory.AgentFactory")
    @patch("farm.core.environment.Environment")
    def test_chromosome_applied_to_agent_before_model_load(
        self, mock_env_cls, mock_factory_cls, mock_services, mock_apply
    ):
        """_apply_chromosome_to_agent must be called for agents with chromosomes."""
        payload = self._make_payload()
        env = self._build_mock_env()
        mock_env_cls.return_value = env

        mock_agent = self._make_mock_agent()
        mock_factory_cls.return_value.create_learning_agent.return_value = mock_agent

        def add_agent(agent, **kwargs):
            env._agent_objects[agent.agent_id] = agent

        env.add_agent.side_effect = add_agent

        import random

        from farm.config import SimulationConfig
        from farm.runners.intrinsic_evolution_experiment import IntrinsicEvolutionPolicy

        restore_environment_from_checkpoint(
            payload,
            config=SimulationConfig(),
            path=None,
            policy=IntrinsicEvolutionPolicy(),
            policy_rng=random.Random(0),
        )

        mock_apply.assert_called_once()
        _, chromosome_arg = mock_apply.call_args.args
        from farm.core.hyperparameter_chromosome import HyperparameterChromosome
        self.assertIsInstance(chromosome_arg, HyperparameterChromosome)


if __name__ == "__main__":
    unittest.main()
