#!/usr/bin/env python3
"""
Observation Generation Profiler - Phase 2 Component-Level Profiling

Profiles observation generation to identify bottlenecks:
- Multi-channel observation build time
- Bilinear interpolation vs. nearest-neighbor
- Memmap resource window vs. spatial queries
- Perception system overhead
- Memory allocations per observation
- Scaling with observation radius

Usage:
    python -m benchmarks.implementations.profiling.observation_profiler
"""

import time
from typing import List

import numpy as np

from farm.config import SimulationConfig
from farm.core.agent import AgentFactory, AgentServices
from farm.core.environment import Environment
from farm.core.observations import ObservationConfig


class ObservationProfiler:
    """Profile observation generation operations."""

    def __init__(self):
        self.results = {}

    def create_test_environment(
        self,
        num_agents: int,
        num_resources: int,
        obs_radius: int = 5,
        use_memmap: bool = False,
    ) -> Environment:
        """Create a test environment for profiling."""
        config = SimulationConfig.from_centralized_config(environment="development")

        # Configure observation settings
        config.observation = ObservationConfig(R=obs_radius)

        # Configure resource settings
        config.resources.use_memmap_resources = use_memmap

        # Create environment
        env = Environment(
            width=100,
            height=100,
            resource_distribution={"type": "random", "amount": num_resources},
            config=config,
        )

        # Create services for agent factory
        services = AgentServices(
            spatial_service=env.spatial_service,
            time_service=getattr(env, "time_service", None),
            metrics_service=getattr(env, "metrics_service", None),
            logging_service=getattr(env, "logging_service", None),
            validation_service=getattr(env, "validation_service", None),
            lifecycle_service=getattr(env, "lifecycle_service", None),
        )

        # Create agent factory
        factory = AgentFactory(services)

        # Create agents
        for i in range(num_agents):
            agent = factory.create_default_agent(
                agent_id=env.get_next_agent_id(),
                position=(
                    np.random.uniform(0, env.width),
                    np.random.uniform(0, env.height),
                ),
                initial_resources=10.0,
                environment=env,
                agent_type="profiler",
            )
            env.add_agent(agent)

        # Update spatial index
        env.spatial_index.update()

        return env

    def profile_observation_generation(self, agent_counts: List[int], obs_radius: int = 5):
        """Profile observation generation time for different agent counts."""
        print("\n" + "=" * 60)
        print(f"Profiling Observation Generation (radius={obs_radius})")
        print("=" * 60 + "\n")

        results = {}

        for num_agents in agent_counts:
            print(f"Testing with {num_agents} agents...")

            # Create environment
            env = self.create_test_environment(
                num_agents=num_agents,
                num_resources=num_agents * 2,
                obs_radius=obs_radius,
            )

            # Profile observation generation for all agents
            start = time.perf_counter()
            for agent_id in env.agents:
                _ = env._get_observation(agent_id)
            total_time = time.perf_counter() - start

            results[num_agents] = {
                "total_time": total_time,
                "time_per_observation": total_time / num_agents,
                "observations_per_second": (num_agents / total_time if total_time > 0 else 0),
            }

            per_obs_ms = (total_time * 1000 / num_agents) if num_agents > 0 else 0
            rate = (num_agents / total_time) if total_time > 0 else 0
            print(f"  Total: {total_time * 1000:.2f}ms, Per obs: {per_obs_ms:.2f}ms, Rate: {rate:.0f} obs/s")

            # Clean up
            env.cleanup()

        self.results[f"obs_generation_r{obs_radius}"] = results
        return results

    def profile_observation_radius(self, num_agents: int, radii: List[int]):
        """Profile observation generation with different radii."""
        print("\n" + "=" * 60)
        print(f"Profiling Observation Radius Impact ({num_agents} agents)")
        print("=" * 60 + "\n")

        results = {}

        for radius in radii:
            print(f"Testing radius {radius}...")

            # Create environment
            env = self.create_test_environment(
                num_agents=num_agents,
                num_resources=num_agents * 2,
                obs_radius=radius,
            )

            # Profile observation generation
            start = time.perf_counter()
            for agent_id in env.agents:
                _ = env._get_observation(agent_id)
            total_time = time.perf_counter() - start

            obs_size = 2 * radius + 1

            results[radius] = {
                "total_time": total_time,
                "time_per_observation": total_time / num_agents,
                "observation_size": obs_size,
                "observation_cells": obs_size * obs_size,
            }

            per_obs_ms = (total_time * 1000 / num_agents) if num_agents > 0 else 0
            print(
                f"  Size: {obs_size}x{obs_size} ({obs_size * obs_size} cells), "
                f"Time: {total_time * 1000:.2f}ms, "
                f"Per obs: {per_obs_ms:.2f}ms"
            )

            # Clean up
            env.cleanup()

        self.results["radius_impact"] = results
        return results

    def profile_memmap_vs_spatial(self, num_agents: int, num_resources: int):
        """Compare memmap resource window vs spatial queries."""
        print("\n" + "=" * 60)
        print("Comparing Memmap vs Spatial Queries")
        print(f"({num_agents} agents, {num_resources} resources)")
        print("=" * 60 + "\n")

        results = {}

        # Test with spatial queries
        print("Testing with spatial queries...")
        env_spatial = self.create_test_environment(
            num_agents=num_agents,
            num_resources=num_resources,
            obs_radius=5,
            use_memmap=False,
        )

        start = time.perf_counter()
        for agent_id in env_spatial.agents:
            _ = env_spatial._get_observation(agent_id)
        spatial_time = time.perf_counter() - start

        results["spatial_queries"] = {
            "total_time": spatial_time,
            "time_per_observation": spatial_time / num_agents,
        }

        per_obs_ms = (spatial_time * 1000 / num_agents) if num_agents > 0 else 0
        print(f"  Time: {spatial_time * 1000:.2f}ms, Per obs: {per_obs_ms:.2f}ms")

        env_spatial.cleanup()

        # Test with memmap
        print("\nTesting with memmap...")
        env_memmap = self.create_test_environment(
            num_agents=num_agents,
            num_resources=num_resources,
            obs_radius=5,
            use_memmap=True,
        )

        start = time.perf_counter()
        for agent_id in env_memmap.agents:
            _ = env_memmap._get_observation(agent_id)
        memmap_time = time.perf_counter() - start

        results["memmap"] = {
            "total_time": memmap_time,
            "time_per_observation": memmap_time / num_agents,
        }

        per_obs_ms = (memmap_time * 1000 / num_agents) if num_agents > 0 else 0
        print(f"  Time: {memmap_time * 1000:.2f}ms, Per obs: {per_obs_ms:.2f}ms")

        # Calculate speedup
        speedup = spatial_time / memmap_time if memmap_time > 0 else 0
        results["speedup"] = speedup

        print(f"\n  Speedup: {speedup:.2f}x ({'memmap faster' if speedup > 1 else 'spatial faster'})")

        env_memmap.cleanup()

        self.results["memmap_comparison"] = results
        return results

    def profile_perception_overhead(self, num_agents: int):
        """Profile perception system overhead."""
        print("\n" + "=" * 60)
        print(f"Profiling Perception System Components ({num_agents} agents)")
        print("=" * 60 + "\n")

        env = self.create_test_environment(
            num_agents=num_agents,
            num_resources=num_agents * 2,
            obs_radius=5,
        )

        # Reset perception profile (accumulated during observation generation) in a robust way
        if hasattr(env, "reset_perception_profile") and callable(getattr(env, "reset_perception_profile")):
            env.reset_perception_profile()
        elif hasattr(env, "_perception_profile"):
            env._perception_profile = {
                "spatial_query_time_s": 0.0,
                "bilinear_time_s": 0.0,
                "nearest_time_s": 0.0,
                "bilinear_points": 0,
                "nearest_points": 0,
            }
        else:
            print(
                "Warning: Environment does not support perception profile reset. Accumulated timing data may include results from previous profiling runs."
            )

        # Generate observations (accumulates profile data)
        total_start = time.perf_counter()
        for agent_id in env.agents:
            _ = env._get_observation(agent_id)
        total_time = time.perf_counter() - total_start

        # Get accumulated profile
        if hasattr(env, "get_perception_profile") and callable(getattr(env, "get_perception_profile")):
            profile = env.get_perception_profile(reset=False)
        elif hasattr(env, "_perception_profile"):
            profile = env._perception_profile
        else:
            profile = {}

        results = {
            "total_time": total_time,
            "spatial_query_time": profile.get("spatial_query_time_s", 0),
            "bilinear_time": profile.get("bilinear_time_s", 0),
            "nearest_time": profile.get("nearest_time_s", 0),
            "bilinear_points": profile.get("bilinear_points", 0),
            "nearest_points": profile.get("nearest_points", 0),
        }

        # Calculate percentages
        if total_time > 0:
            results["spatial_query_pct"] = (profile.get("spatial_query_time_s", 0) / total_time) * 100
            results["bilinear_pct"] = (profile.get("bilinear_time_s", 0) / total_time) * 100
            results["nearest_pct"] = (profile.get("nearest_time_s", 0) / total_time) * 100

        print(f"Total observation time: {total_time * 1000:.2f}ms")
        print("\nBreakdown:")
        print(
            f"  Spatial queries: {results.get('spatial_query_time', 0) * 1000:.2f}ms "
            f"({results.get('spatial_query_pct', 0):.1f}%)"
        )
        print(
            f"  Bilinear interp: {results.get('bilinear_time', 0) * 1000:.2f}ms "
            f"({results.get('bilinear_pct', 0):.1f}%) "
            f"[{results['bilinear_points']} points]"
        )
        print(
            f"  Nearest neighbor: {results.get('nearest_time', 0) * 1000:.2f}ms "
            f"({results.get('nearest_pct', 0):.1f}%) "
            f"[{results['nearest_points']} points]"
        )

        env.cleanup()

        self.results["perception_overhead"] = results
        return results

    def generate_report(self):
        """Generate a summary report of profiling results."""
        print("\n" + "=" * 60)
        print("Observation Generation Profiling Report")
        print("=" * 60 + "\n")

        # Observation generation scaling
        for key in self.results:
            if key.startswith("obs_generation_r"):
                radius = key.split("_r")[1]
                print(f"## Observation Generation (radius={radius})\n")
                for num_agents, data in sorted(self.results[key].items()):
                    print(
                        f"  {num_agents:>4} agents: {data['time_per_observation'] * 1000:.2f}ms per obs, "
                        f"{data['observations_per_second']:.0f} obs/s"
                    )
                print()

        # Radius impact
        if "radius_impact" in self.results:
            print("## Observation Radius Impact\n")
            for radius, data in sorted(self.results["radius_impact"].items()):
                print(
                    f"  Radius {radius:>2} ({data['observation_size']:>2}x{data['observation_size']:>2}): "
                    f"{data['time_per_observation'] * 1000:.2f}ms per obs"
                )
            print()

        # Memmap comparison
        if "memmap_comparison" in self.results:
            print("## Memmap vs Spatial Queries\n")
            data = self.results["memmap_comparison"]
            print(f"  Spatial queries: {data['spatial_queries']['time_per_observation'] * 1000:.2f}ms per obs")
            print(f"  Memmap:          {data['memmap']['time_per_observation'] * 1000:.2f}ms per obs")
            print(f"  Speedup:         {data['speedup']:.2f}x")
            print()

        # Perception overhead
        if "perception_overhead" in self.results:
            print("## Perception System Breakdown\n")
            data = self.results["perception_overhead"]
            print(f"  Spatial queries: {data.get('spatial_query_pct', 0):.1f}%")
            print(f"  Bilinear interp: {data.get('bilinear_pct', 0):.1f}%")
            print(f"  Nearest neighbor: {data.get('nearest_pct', 0):.1f}%")
            print()

        print("=" * 60 + "\n")


def main():
    """Run observation generation profiling suite."""
    profiler = ObservationProfiler()

    print("=" * 60)
    print("Observation Generation Profiler - Phase 2")
    print("=" * 60)

    # Profile observation generation scaling
    profiler.profile_observation_generation(agent_counts=[10, 50, 100, 200], obs_radius=5)

    # Profile radius impact
    profiler.profile_observation_radius(num_agents=50, radii=[3, 5, 10, 15, 20])

    # Compare memmap vs spatial queries
    profiler.profile_memmap_vs_spatial(num_agents=50, num_resources=100)

    # Profile perception overhead
    profiler.profile_perception_overhead(num_agents=50)

    # Generate report
    profiler.generate_report()

    print("Observation generation profiling complete!")
    print("  Results saved in profiler.results")


if __name__ == "__main__":
    main()
