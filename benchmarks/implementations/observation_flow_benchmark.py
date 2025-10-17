from __future__ import annotations

import time as _time
from typing import Any, Dict, Optional

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None

from benchmarks.core.experiments import Experiment, ExperimentContext
from benchmarks.core.registry import register_experiment
from farm.config import EnvironmentConfig, SimulationConfig
from farm.config.config import DatabaseConfig
from farm.core.agent import AgentCore, AgentFactory
from farm.core.environment import Environment
from farm.core.observations import ObservationConfig
from farm.core.services.implementations import SpatialIndexAdapter


@register_experiment("observation_flow")
class ObservationFlowBenchmark(Experiment):
    """
    Benchmark observation generation throughput and latency.

    This benchmark measures the performance of the AgentFarm observation system
    by testing observation generation across multiple agents and simulation steps.
    It provides comprehensive metrics on observation throughput, latency, and
    memory efficiency.

    Key Metrics Measured:
    - observe() calls per second (overall throughput)
    - per-step observation latency (mean and p95)
    - Observation memory usage and efficiency
    - Cache hit rates and rebuild statistics
    - Estimated computational load (GFLOPs)

    Parameters
    ----------
    width : int, default=200
        Environment width in grid cells
    height : int, default=200
        Environment height in grid cells
    num_agents : int, default=200
        Number of agents to spawn for testing
    steps : int, default=100
        Number of simulation steps to run
    radius : int, default=6
        Observation radius for each agent
    fov_radius : int, default=6
        Field of view radius for observations
    device : str, default="cpu"
        Device for observation computation ("cpu" or "cuda")
    """

    def __init__(
        self,
        *,
        width: int = 200,
        height: int = 200,
        num_agents: int = 200,
        steps: int = 100,
        radius: int = 6,
        fov_radius: int = 6,
        device: str = "cpu",
    ) -> None:
        params = {
            "width": width,
            "height": height,
            "num_agents": num_agents,
            "steps": steps,
            "radius": radius,
            "fov_radius": fov_radius,
            "device": device,
        }
        super().__init__(params)
        self._env: Optional[Environment] = None
        self._agents: list[AgentCore] = []
        self._agent_ids: list[str] = []
        self._spatial: Optional[SpatialIndexAdapter] = None
        self._steps = steps
        self._num_agents = num_agents
        self._width = width
        self._height = height
        self._radius = radius
        self._fov = fov_radius
        self._device = device

    def _make_env(self) -> Environment:
        obs_cfg = ObservationConfig(
            R=self._radius,
            fov_radius=self._fov,
            device=self._device,
            dtype="float32",
            initialization="zeros",
        )
        sim_cfg = SimulationConfig(
            seed=42,
            observation=obs_cfg,
            environment=EnvironmentConfig(width=self._width, height=self._height),
            database=DatabaseConfig(use_in_memory_db=True, persist_db_on_completion=False),
        )
        env = Environment(
            width=self._width,
            height=self._height,
            resource_distribution={"amount": 10},
            db_path=":memory:",
            config=sim_cfg,
        )
        return env

    def _spawn_agents(self, env: Environment) -> None:
        """Spawn agents with uniform random placement across the environment.
        
        Creates the specified number of agents with random positions within the environment
        bounds. Uses a fixed seed (123) for reproducible results. Each agent is created
        with zero initial resources and added to the environment's spatial index.
        """
        # Use numpy for better random number generation if available, fallback to standard random
        if np is not None:
            rng = np.random.default_rng(123)  # Fixed seed for reproducibility

            def rand_int(low: int, high: int) -> int:
                return int(rng.integers(low, high))
        else:
            import random

            random.seed(123)  # Fixed seed for reproducibility

            def rand_int(low: int, high: int) -> int:
                return random.randint(low, high - 1)

        # Clear any existing agents
        self._agents.clear()
        self._agent_ids.clear()

        # Create agents with random positions
        factory = AgentFactory(spatial_service=env.spatial_service)
        for i in range(self._num_agents):
            x = float(rand_int(0, self._width))
            y = float(rand_int(0, self._height))
            agent = factory.create_default_agent(
                agent_id=f"A{i}",
                position=(x, y),
                initial_resources=0,
            )
            env.add_agent(agent)
            self._agents.append(agent)
            self._agent_ids.append(agent.agent_id)

        # Update spatial index to include all new agents
        env.spatial_index.set_references(list(env._agent_objects.values()), env.resources)
        env.spatial_index.update()

    def setup(self, context: ExperimentContext) -> None:
        env = self._make_env()
        self._env = env
        self._spatial = env.spatial_service
        self._spawn_agents(env)

    def execute_once(self, context: ExperimentContext) -> Dict[str, Any]:
        assert self._env is not None
        env = self._env

        # Warmup: perform one observation per agent to populate caches and initialize systems
        for agent_id in self._agent_ids:
            _ = env.observe(agent_id)

        # Measure observation performance across multiple simulation steps
        total_observes = 0
        t0 = _time.perf_counter()
        per_step_times: list[float] = []

        for _ in range(self._steps):
            step_start = _time.perf_counter()
            # Observe all agents in this step
            for agent_id in self._agent_ids:
                _ = env.observe(agent_id)
                total_observes += 1
            per_step_times.append(_time.perf_counter() - step_start)

        total_time = _time.perf_counter() - t0
        obs_per_sec = total_observes / total_time if total_time > 0 else 0.0
        if np is not None:
            mean_step = float(np.mean(per_step_times)) if per_step_times else 0.0
            p95_step = float(np.percentile(per_step_times, 95)) if per_step_times else 0.0
        else:
            # Simple approximations without numpy
            mean_step = sum(per_step_times) / len(per_step_times) if per_step_times else 0.0
            sorted_times = sorted(per_step_times)
            if sorted_times:
                k = int(0.95 * (len(sorted_times) - 1))
                p95_step = sorted_times[k]
            else:
                p95_step = 0.0

        # Collect detailed observation metrics from a sample agent
        obs_metrics = {}
        try:
            if self._agent_ids:
                sample_id = self._agent_ids[0]
                obs = env.agent_observations.get(sample_id)
                if obs is not None and hasattr(obs, "get_metrics"):
                    obs_metrics = obs.get_metrics()
        except Exception:
            obs_metrics = {}

        # Collect perception system profiling data
        perc_profile = {}
        try:
            if hasattr(env, "get_perception_profile"):
                perc_profile = env.get_perception_profile(reset=True)
        except Exception:
            perc_profile = {}

        # Estimate computational load (GFLOPs) for observation tensor construction
        # This provides insight into the computational complexity of observations
        S = 2 * self._radius + 1  # Observation grid size (e.g., 13x13 for radius=6)
        C = 13  # Number of observation channels (SELF_HP, ALLIES_HP, ENEMIES_HP, RESOURCES, OBSTACLES, TERRAIN_COST, VISIBILITY, KNOWN_EMPTY, DAMAGE_HEAT, TRAILS, ALLY_SIGNAL, GOAL, LANDMARKS)
        k_ops_per_cell = 2.0  # Estimated operations per cell during dense reconstruction
        total_cells = float(C * S * S)
        dense_rebuilds = float(obs_metrics.get("dense_rebuilds", 0))
        gflops_est = (total_cells * k_ops_per_cell * dense_rebuilds) / 1e9 / max(total_time, 1e-9)

        return {
            "total_observes": total_observes,
            "total_time_s": total_time,
            "observes_per_sec": obs_per_sec,
            "mean_step_time_s": mean_step,
            "p95_step_time_s": p95_step,
            "steps": self._steps,
            "num_agents": self._num_agents,
            "obs_dense_bytes": int(obs_metrics.get("dense_bytes", 0)),
            "obs_sparse_bytes": int(obs_metrics.get("sparse_logical_bytes", 0)),
            "obs_memory_reduction_percent": float(obs_metrics.get("memory_reduction_percent", 0.0)),
            "obs_cache_hit_rate": float(obs_metrics.get("cache_hit_rate", 1.0)),
            "obs_dense_rebuilds": int(obs_metrics.get("dense_rebuilds", 0)),
            "obs_dense_rebuild_time_s_total": float(obs_metrics.get("dense_rebuild_time_s_total", 0.0)),
            "gflops_observation_est": float(max(0.0, gflops_est)),
            "perception_profile": perc_profile,
        }

    def teardown(self, context: ExperimentContext) -> None:
        try:
            if self._env is not None:
                self._env.close()
        finally:
            self._env = None
            self._agents.clear()
            self._agent_ids.clear()
