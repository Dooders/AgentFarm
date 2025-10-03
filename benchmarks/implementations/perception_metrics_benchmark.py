from __future__ import annotations

import time as _time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from benchmarks.core.experiments import Experiment, ExperimentContext
from benchmarks.core.registry import register_experiment
from farm.config import EnvironmentConfig, SimulationConfig
from farm.config.config import DatabaseConfig
from farm.core.agent import BaseAgent
from farm.core.environment import Environment
from farm.core.observations import ObservationConfig

DEFAULT_MIN_ENV_SIZE = 200
DEFAULT_ENV_SCALE = 2.5
DEFAULT_CHANNEL_COUNT = 13
OPERATIONS_PER_CELL_ESTIMATE = 2.0


@register_experiment("perception_metrics")
class PerceptionMetricsBenchmark(Experiment):
    """Benchmark perception metrics across agent counts, radii, and storage modes.

    Measures:
      - Per-agent observation latency and throughput
      - Observation memory (dense vs sparse logical)
      - Cache hit rates for lazy dense construction
      - Bilinear vs nearest timing from environment profile
      - Estimated GFLOPs for dense reconstruction
    """

    def __init__(
        self,
        agent_counts: Optional[Union[int, List[int]]] = None,
        radii: Optional[Union[int, List[int]]] = None,
        storage_modes: Optional[Union[str, List[str]]] = None,
        use_bilinear_list: Optional[Union[bool, List[bool]]] = None,
        steps: int = 10,
        device: str = "cpu",
    ) -> None:
        # Handle both single values (from sweep) and lists (from baseline)
        def ensure_list(param, default):
            if param is None:
                return default
            elif isinstance(param, list):
                return param
            else:
                # Single value from sweep - wrap in list
                return [param]

        default_agent_counts = [100, 1000, 10000]
        default_radii = [5, 8, 10]
        default_storage_modes = ["hybrid", "dense"]
        default_use_bilinear_list = [True, False]

        agent_counts = ensure_list(agent_counts, default_agent_counts)
        radii = ensure_list(radii, default_radii)
        storage_modes = ensure_list(storage_modes, default_storage_modes)
        use_bilinear_list = ensure_list(use_bilinear_list, default_use_bilinear_list)

        super().__init__({
            "agent_counts": agent_counts,
            "radii": radii,
            "storage_modes": storage_modes,
            "use_bilinear_list": use_bilinear_list,
            "steps": steps,
            "device": device,
        })
        self._agent_counts = agent_counts
        self._radii = radii
        self._storage_modes = storage_modes
        self._use_bilinear_list = use_bilinear_list
        self._steps = steps
        self._device = device

        self._env: Environment | None = None
        self._agent_ids: List[str] = []

    def _make_env(
        self, width: int, height: int, R: int, storage_mode: str, use_bilinear: bool
    ) -> Environment:
        obs_cfg = ObservationConfig(
            R=R,
            fov_radius=R,
            device=self._device,
            dtype="float32",
            initialization="zeros",
            storage_mode=storage_mode,
            enable_metrics=True,
        )
        sim_cfg = SimulationConfig(
            observation=obs_cfg,
            environment=EnvironmentConfig(
                width=width,
                height=height,
                use_bilinear_interpolation=use_bilinear,
            ),
            database=DatabaseConfig(
                use_in_memory_db=True,
                persist_db_on_completion=False,
            ),
        )
        env = Environment(
            width=width,
            height=height,
            resource_distribution={"amount": 10},
            db_path=":memory:",
            config=sim_cfg,
        )
        return env

    def _spawn_agents(self, env: Environment, num_agents: int) -> List[str]:
        rng = np.random.default_rng(123)
        agent_ids: List[str] = []
        for i in range(num_agents):
            x = float(rng.integers(0, env.width))
            y = float(rng.integers(0, env.height))
            agent = BaseAgent(
                agent_id=f"A{i}",
                position=(x, y),
                resource_level=0,
                spatial_service=env.spatial_service,
                environment=env,
                config=env.config,
            )
            env.add_agent(agent)
            agent_ids.append(agent.agent_id)
        env.spatial_index.set_references(
            list(env._agent_objects.values()), env.resources
        )
        env.spatial_index.update()
        return agent_ids

    def setup(self, context: ExperimentContext) -> None:
        # Environment is created per configuration inside run()
        pass

    def _run_once(
        self, num_agents: int, R: int, storage_mode: str, use_bilinear: bool
    ) -> Dict[str, Any]:
        width = max(
            DEFAULT_MIN_ENV_SIZE, int(DEFAULT_ENV_SCALE * R * (num_agents**0.5))
        )
        height = width
        env = self._make_env(width, height, R, storage_mode, use_bilinear)
        agent_ids = self._spawn_agents(env, num_agents)

        # Warmup
        for aid in agent_ids:
            _ = env.observe(aid)

        # Measure
        total_observes = 0
        per_step_times: List[float] = []
        t0 = _time.perf_counter()
        for _ in range(self._steps):
            step_start = _time.perf_counter()
            for aid in agent_ids:
                _ = env.observe(aid)
                total_observes += 1
            per_step_times.append(_time.perf_counter() - step_start)
        total_time = _time.perf_counter() - t0

        mean_step = float(np.mean(per_step_times)) if per_step_times else 0.0
        p95_step = float(np.percentile(per_step_times, 95)) if per_step_times else 0.0

        # Sample metrics
        obs_metrics = {}
        if agent_ids:
            sample_obs = env.agent_observations.get(agent_ids[0])
            if sample_obs is not None and hasattr(sample_obs, "get_metrics"):
                obs_metrics = sample_obs.get_metrics()
        perc_profile = getattr(env, "get_perception_profile", lambda reset=False: {})(
            reset=True
        )

        # GFLOPs est
        C = DEFAULT_CHANNEL_COUNT
        S = 2 * R + 1
        k_ops_per_cell = OPERATIONS_PER_CELL_ESTIMATE
        dense_rebuilds = float(obs_metrics.get("dense_rebuilds", 0))
        total_cells = float(C * S * S)
        gflops_est = (
            (total_cells * k_ops_per_cell * dense_rebuilds)
            / 1e9
            / max(total_time, 1e-9)
        )

        env.close()

        return {
            "agents": num_agents,
            "R": R,
            "storage_mode": storage_mode,
            "use_bilinear": use_bilinear,
            "total_observes": total_observes,
            "total_time_s": total_time,
            "observes_per_sec": (
                (total_observes / total_time) if total_time > 0 else 0.0
            ),
            "mean_step_time_s": mean_step,
            "p95_step_time_s": p95_step,
            "obs_dense_bytes": int(obs_metrics.get("dense_bytes", 0)),
            "obs_sparse_bytes": int(obs_metrics.get("sparse_logical_bytes", 0)),
            "obs_memory_reduction_percent": float(
                obs_metrics.get("memory_reduction_percent", 0.0)
            ),
            "obs_cache_hit_rate": float(obs_metrics.get("cache_hit_rate", 1.0)),
            "obs_dense_rebuilds": int(obs_metrics.get("dense_rebuilds", 0)),
            "obs_dense_rebuild_time_s_total": float(
                obs_metrics.get("dense_rebuild_time_s_total", 0.0)
            ),
            "gflops_observation_est": float(max(0.0, gflops_est)),
            "perception_profile": perc_profile,
        }

    def execute_once(self, context: ExperimentContext) -> Dict[str, Any]:
        results: List[Dict[str, Any]] = []
        for n in self._agent_counts:
            for r in self._radii:
                for mode in self._storage_modes:
                    for bilinear in self._use_bilinear_list:
                        results.append(self._run_once(n, r, mode, bilinear))
        return {"runs": results}

    def teardown(self, context: ExperimentContext) -> None:
        if self._env is not None:
            self._env.close()
        self._env = None
