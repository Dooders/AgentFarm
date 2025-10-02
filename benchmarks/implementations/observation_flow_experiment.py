from __future__ import annotations

"""
ObservationFlow experiment using the new Experiment API.
"""

import time as _time
from typing import Any, Dict

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None

from benchmarks.core.experiments import Experiment, ExperimentContext
from benchmarks.core.registry import register_experiment
from farm.config import EnvironmentConfig, SimulationConfig
from farm.config.config import DatabaseConfig
from farm.core.agent import BaseAgent
from farm.core.environment import Environment
from farm.core.observations import ObservationConfig
from farm.core.services.implementations import SpatialIndexAdapter


@register_experiment(slug="observation_flow", summary="Measure Environment.observe() throughput and latency", tags=["perception", "scalability"])
class ObservationFlowExperiment(Experiment):
    param_schema = {
        "type": "object",
        "properties": {
            "width": {"type": "integer", "default": 200},
            "height": {"type": "integer", "default": 200},
            "num_agents": {"type": "integer", "default": 200},
            "steps": {"type": "integer", "default": 100},
            "radius": {"type": "integer", "default": 6},
            "fov_radius": {"type": "integer", "default": 6},
            "device": {"type": "string", "default": "cpu"},
        },
        "required": [],
    }

    def __init__(self, params=None) -> None:
        super().__init__(params)
        self._env: Environment | None = None
        self._agent_ids: list[str] = []

    def _make_env(self) -> Environment:
        width = int(self.params.get("width", 200))
        height = int(self.params.get("height", 200))
        radius = int(self.params.get("radius", 6))
        fov = int(self.params.get("fov_radius", 6))
        device = str(self.params.get("device", "cpu"))

        obs_cfg = ObservationConfig(R=radius, fov_radius=fov, device=device, dtype="float32", initialization="zeros")
        sim_cfg = SimulationConfig(
            seed=42,
            observation=obs_cfg,
            environment=EnvironmentConfig(width=width, height=height),
            database=DatabaseConfig(use_in_memory_db=True, persist_db_on_completion=False),
        )
        env = Environment(
            width=width,
            height=height,
            resource_distribution={"amount": 10},
            db_path=":memory:",
            config=sim_cfg,
        )
        return env

    def _spawn_agents(self, env: Environment) -> None:
        width = int(self.params.get("width", 200))
        height = int(self.params.get("height", 200))
        num_agents = int(self.params.get("num_agents", 200))

        if np is not None:
            rng = np.random.default_rng(123)
            def rand_int(low: int, high: int) -> int:
                return int(rng.integers(low, high))
        else:
            import random
            random.seed(123)
            def rand_int(low: int, high: int) -> int:
                return random.randint(low, high - 1)

        self._agent_ids.clear()
        for i in range(num_agents):
            x = float(rand_int(0, width))
            y = float(rand_int(0, height))
            agent = BaseAgent(
                agent_id=f"A{i}",
                position=(x, y),
                resource_level=0,
                spatial_service=env.spatial_service,
                environment=env,
                config=env.config,
            )
            env.add_agent(agent)
            self._agent_ids.append(agent.agent_id)

        env.spatial_index.set_references(list(env._agent_objects.values()), env.resources)
        env.spatial_index.update()

    def setup(self, context: ExperimentContext) -> None:
        env = self._make_env()
        self._env = env
        self._spawn_agents(env)

    def execute_once(self, context: ExperimentContext) -> Dict[str, Any]:
        assert self._env is not None
        env = self._env
        steps = int(self.params.get("steps", 100))

        # Warm up one observe per agent to populate caches
        for agent_id in self._agent_ids:
            _ = env.observe(agent_id)

        total_observes = 0
        t0 = _time.perf_counter()
        per_step_times: list[float] = []
        for _ in range(steps):
            s0 = _time.perf_counter()
            for agent_id in self._agent_ids:
                _ = env.observe(agent_id)
                total_observes += 1
            per_step_times.append(_time.perf_counter() - s0)
        total_time = _time.perf_counter() - t0
        observes_per_sec = total_observes / total_time if total_time > 0 else 0.0

        if np is not None and per_step_times:
            mean_step = float(np.mean(per_step_times))
            p95_step = float(np.percentile(per_step_times, 95))
        else:
            mean_step = (sum(per_step_times) / len(per_step_times)) if per_step_times else 0.0
            sorted_times = sorted(per_step_times)
            p95_step = sorted_times[int(0.95 * (len(sorted_times) - 1))] if sorted_times else 0.0

        metrics: Dict[str, Any] = {
            "total_observes": total_observes,
            "total_time_s": total_time,
            "observes_per_sec": observes_per_sec,
            "mean_step_time_s": mean_step,
            "p95_step_time_s": p95_step,
            "steps": steps,
            "num_agents": len(self._agent_ids),
        }
        return metrics

    def teardown(self, context: ExperimentContext) -> None:
        try:
            if self._env is not None:
                self._env.close()
        finally:
            self._env = None
            self._agent_ids.clear()

