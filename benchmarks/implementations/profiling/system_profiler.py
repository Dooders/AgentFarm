#!/usr/bin/env python3
"""
System-Level Profiler - Phase 4

Profiles system-wide performance and resource usage:
- CPU usage over time
- Memory usage over time
- Disk I/O patterns
- Scaling with agent count
- Scaling with simulation steps
- Scaling with environment size
- Thread/process metrics

Usage:
    python -m benchmarks.implementations.profiling.system_profiler
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import psutil

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from farm.config import SimulationConfig
from farm.core.simulation import run_simulation


class SystemProfiler:
    """Profile system-level resource usage during simulation."""

    def __init__(self):
        self.results = {}
        self.process = psutil.Process()

    def sample_system_metrics(self) -> Dict:
        """Sample current system metrics."""
        cpu_percent = self.process.cpu_percent(interval=0.1)
        memory_info = self.process.memory_info()
        
        try:
            io_counters = self.process.io_counters()
            disk_read = io_counters.read_bytes
            disk_write = io_counters.write_bytes
        except (AttributeError, PermissionError):
            disk_read = 0
            disk_write = 0
        
        return {
            "timestamp": time.time(),
            "cpu_percent": cpu_percent,
            "memory_rss_mb": memory_info.rss / 1024**2,
            "memory_vms_mb": memory_info.vms / 1024**2,
            "disk_read_mb": disk_read / 1024**2,
            "disk_write_mb": disk_write / 1024**2,
            "num_threads": self.process.num_threads(),
        }

    def profile_scaling_agents(self, agent_counts: List[int], steps: int = 100):
        """Profile how performance scales with agent count."""
        print("\n" + "="*60)
        print(f"Profiling Agent Count Scaling ({steps} steps)")
        print("="*60 + "\n")

        results = {}
        
        for num_agents in agent_counts:
            print(f"Testing with {num_agents} agents...")
            
            # Create config
            config = SimulationConfig.from_centralized_config(environment="development")
            config.population.system_agents = num_agents
            config.population.independent_agents = 0
            config.population.control_agents = 0
            config.database.use_in_memory_db = True
            
            # Sample metrics before
            before_metrics = self.sample_system_metrics()
            
            # Run simulation
            start_time = time.time()
            try:
                env = run_simulation(
                    num_steps=steps,
                    config=config,
                    path=None,
                    save_config=False,
                )
                duration = time.time() - start_time
                
                # Sample metrics after
                after_metrics = self.sample_system_metrics()
                
                # Calculate statistics
                steps_per_second = steps / duration if duration > 0 else 0
                memory_growth = after_metrics["memory_rss_mb"] - before_metrics["memory_rss_mb"]
                
                results[num_agents] = {
                    "duration": duration,
                    "steps_per_second": steps_per_second,
                    "avg_step_time": duration / steps if steps > 0 else 0,
                    "memory_before_mb": before_metrics["memory_rss_mb"],
                    "memory_after_mb": after_metrics["memory_rss_mb"],
                    "memory_growth_mb": memory_growth,
                    "memory_per_agent_kb": (memory_growth * 1024) / num_agents if num_agents > 0 else 0,
                    "cpu_percent": after_metrics["cpu_percent"],
                    "final_agent_count": len(env.agents),
                }
                
                print(f"  Duration: {duration:.2f}s ({steps_per_second:.1f} steps/s)")
                print(f"  Memory: {memory_growth:+.1f} MB ({memory_growth*1024/num_agents:.1f} KB/agent)")
                print(f"  Final agents: {len(env.agents)}/{num_agents}")
                
                # Clean up
                env.cleanup()
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
                results[num_agents] = {"error": str(e)}
            
            time.sleep(2)  # Cool down between runs
        
        self.results["scaling_agents"] = results
        return results

    def profile_scaling_steps(self, step_counts: List[int], num_agents: int = 50):
        """Profile how performance scales with simulation steps."""
        print("\n" + "="*60)
        print(f"Profiling Step Count Scaling ({num_agents} agents)")
        print("="*60 + "\n")

        results = {}
        
        for num_steps in step_counts:
            print(f"Testing with {num_steps} steps...")
            
            # Create config
            config = SimulationConfig.from_centralized_config(environment="development")
            config.population.system_agents = num_agents
            config.population.independent_agents = 0
            config.population.control_agents = 0
            config.database.use_in_memory_db = True
            
            # Sample metrics before
            before_metrics = self.sample_system_metrics()
            
            # Run simulation
            start_time = time.time()
            try:
                env = run_simulation(
                    num_steps=num_steps,
                    config=config,
                    path=None,
                    save_config=False,
                )
                duration = time.time() - start_time
                
                # Sample metrics after
                after_metrics = self.sample_system_metrics()
                
                steps_per_second = num_steps / duration if duration > 0 else 0
                
                results[num_steps] = {
                    "duration": duration,
                    "steps_per_second": steps_per_second,
                    "avg_step_time": duration / num_steps if num_steps > 0 else 0,
                    "memory_after_mb": after_metrics["memory_rss_mb"],
                    "cpu_percent": after_metrics["cpu_percent"],
                }
                
                print(f"  Duration: {duration:.2f}s ({steps_per_second:.1f} steps/s)")
                print(f"  Avg step: {duration*1000/num_steps:.2f}ms")
                
                # Clean up
                env.cleanup()
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
                results[num_steps] = {"error": str(e)}
            
            time.sleep(2)
        
        self.results["scaling_steps"] = results
        return results

    def profile_scaling_environment(self, env_sizes: List[Tuple[int, int]], 
                                   num_agents: int = 50, steps: int = 100):
        """Profile how performance scales with environment size."""
        print("\n" + "="*60)
        print(f"Profiling Environment Size Scaling ({num_agents} agents, {steps} steps)")
        print("="*60 + "\n")

        results = {}
        
        for width, height in env_sizes:
            size_key = f"{width}x{height}"
            print(f"Testing {size_key} environment...")
            
            # Create config
            config = SimulationConfig.from_centralized_config(environment="development")
            config.environment.width = width
            config.environment.height = height
            config.population.system_agents = num_agents
            config.population.independent_agents = 0
            config.population.control_agents = 0
            config.database.use_in_memory_db = True
            
            # Run simulation
            start_time = time.time()
            try:
                env = run_simulation(
                    num_steps=steps,
                    config=config,
                    path=None,
                    save_config=False,
                )
                duration = time.time() - start_time
                
                steps_per_second = steps / duration if duration > 0 else 0
                
                results[size_key] = {
                    "width": width,
                    "height": height,
                    "area": width * height,
                    "duration": duration,
                    "steps_per_second": steps_per_second,
                    "avg_step_time": duration / steps if steps > 0 else 0,
                }
                
                print(f"  Duration: {duration:.2f}s ({steps_per_second:.1f} steps/s)")
                
                # Clean up
                env.cleanup()
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
                results[size_key] = {"error": str(e)}
            
            time.sleep(2)
        
        self.results["scaling_environment"] = results
        return results

    def profile_memory_over_time(self, num_agents: int = 50, steps: int = 500, 
                                sample_interval: int = 10):
        """Profile memory usage over simulation time."""
        print("\n" + "="*60)
        print(f"Profiling Memory Over Time ({num_agents} agents, {steps} steps)")
        print("="*60 + "\n")

        # Create config
        config = SimulationConfig.from_centralized_config(environment="development")
        config.population.system_agents = num_agents
        config.population.independent_agents = 0
        config.population.control_agents = 0
        config.database.use_in_memory_db = True
        
        # Will need to instrument the simulation loop
        # For now, just sample at start and end
        print("Sampling memory usage...")
        
        before_metrics = self.sample_system_metrics()
        
        start_time = time.time()
        try:
            env = run_simulation(
                num_steps=steps,
                config=config,
                path=None,
                save_config=False,
            )
            duration = time.time() - start_time
            
            after_metrics = self.sample_system_metrics()
            
            results = {
                "num_agents": num_agents,
                "steps": steps,
                "duration": duration,
                "memory_start_mb": before_metrics["memory_rss_mb"],
                "memory_end_mb": after_metrics["memory_rss_mb"],
                "memory_growth_mb": after_metrics["memory_rss_mb"] - before_metrics["memory_rss_mb"],
                "memory_growth_per_step_kb": 
                    (after_metrics["memory_rss_mb"] - before_metrics["memory_rss_mb"]) * 1024 / steps,
            }
            
            print(f"  Duration: {duration:.2f}s")
            print(f"  Memory: {before_metrics['memory_rss_mb']:.1f} MB → "
                  f"{after_metrics['memory_rss_mb']:.1f} MB "
                  f"({results['memory_growth_mb']:+.1f} MB)")
            print(f"  Growth: {results['memory_growth_per_step_kb']:.2f} KB/step")
            
            # Clean up
            env.cleanup()
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results = {"error": str(e)}
        
        self.results["memory_over_time"] = results
        return results

    def profile_cpu_utilization(self, num_agents: int = 100, steps: int = 200):
        """Profile CPU utilization during simulation."""
        print("\n" + "="*60)
        print(f"Profiling CPU Utilization ({num_agents} agents, {steps} steps)")
        print("="*60 + "\n")

        # Create config
        config = SimulationConfig.from_centralized_config(environment="development")
        config.population.system_agents = num_agents
        config.population.independent_agents = 0
        config.population.control_agents = 0
        config.database.use_in_memory_db = True
        
        print("Measuring CPU usage...")
        
        # Sample CPU before
        cpu_before = psutil.cpu_percent(interval=1, percpu=True)
        
        start_time = time.time()
        try:
            env = run_simulation(
                num_steps=steps,
                config=config,
                path=None,
                save_config=False,
            )
            duration = time.time() - start_time
            
            # Sample CPU after
            cpu_after = psutil.cpu_percent(interval=1, percpu=True)
            
            results = {
                "num_agents": num_agents,
                "steps": steps,
                "duration": duration,
                "cpu_count": psutil.cpu_count(),
                "cpu_before": cpu_before,
                "cpu_after": cpu_after,
                "avg_cpu_usage": sum(cpu_after) / len(cpu_after),
                "max_cpu_usage": max(cpu_after),
            }
            
            print(f"  Duration: {duration:.2f}s")
            print(f"  CPU cores: {results['cpu_count']}")
            print(f"  Avg CPU usage: {results['avg_cpu_usage']:.1f}%")
            print(f"  Max CPU usage: {results['max_cpu_usage']:.1f}%")
            
            # Clean up
            env.cleanup()
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results = {"error": str(e)}
        
        self.results["cpu_utilization"] = results
        return results

    def generate_report(self):
        """Generate a summary report of system profiling results."""
        print("\n" + "="*60)
        print("System-Level Profiling Report")
        print("="*60 + "\n")

        # Agent scaling
        if "scaling_agents" in self.results:
            print("## Agent Count Scaling\n")
            for num_agents, data in sorted(self.results["scaling_agents"].items()):
                if "error" not in data:
                    print(f"  {num_agents:>4} agents: {data['steps_per_second']:>6.1f} steps/s, "
                          f"{data['memory_per_agent_kb']:>6.1f} KB/agent")
            
            # Analyze scaling
            scaling_data = [(k, v) for k, v in self.results["scaling_agents"].items() 
                          if "error" not in v]
            if len(scaling_data) >= 2:
                first = scaling_data[0][1]
                last = scaling_data[-1][1]
                ratio = scaling_data[-1][0] / scaling_data[0][0]
                time_ratio = last["duration"] / first["duration"]
                
                if time_ratio < ratio * 1.5:
                    scaling_type = "Near-linear (good!)"
                elif time_ratio < ratio * 2.5:
                    scaling_type = "Sub-quadratic (acceptable)"
                else:
                    scaling_type = "Poor scaling (investigate!)"
                
                print(f"\n  Scaling: {scaling_type}")
                print(f"  {ratio:.0f}x agents → {time_ratio:.1f}x time")
            print()

        # Step scaling
        if "scaling_steps" in self.results:
            print("## Step Count Scaling\n")
            for num_steps, data in sorted(self.results["scaling_steps"].items()):
                if "error" not in data:
                    print(f"  {num_steps:>5} steps: {data['steps_per_second']:>6.1f} steps/s, "
                          f"{data['avg_step_time']*1000:>6.2f} ms/step")
            print()

        # Environment size scaling
        if "scaling_environment" in self.results:
            print("## Environment Size Scaling\n")
            for size_key, data in sorted(self.results["scaling_environment"].items()):
                if "error" not in data:
                    print(f"  {size_key:>10}: {data['steps_per_second']:>6.1f} steps/s")
            print()

        # Memory over time
        if "memory_over_time" in self.results:
            data = self.results["memory_over_time"]
            if "error" not in data:
                print("## Memory Growth\n")
                print(f"  Start: {data['memory_start_mb']:.1f} MB")
                print(f"  End: {data['memory_end_mb']:.1f} MB")
                print(f"  Growth: {data['memory_growth_mb']:+.1f} MB")
                print(f"  Rate: {data['memory_growth_per_step_kb']:.2f} KB/step")
                print()

        # CPU utilization
        if "cpu_utilization" in self.results:
            data = self.results["cpu_utilization"]
            if "error" not in data:
                print("## CPU Utilization\n")
                print(f"  Cores: {data['cpu_count']}")
                print(f"  Avg usage: {data['avg_cpu_usage']:.1f}%")
                print(f"  Max usage: {data['max_cpu_usage']:.1f}%")
                print()

        print("="*60 + "\n")


def main():
    """Run system-level profiling suite."""
    profiler = SystemProfiler()
    
    print("="*60)
    print("System-Level Profiler - Phase 4")
    print("="*60)
    
    # Profile agent count scaling
    profiler.profile_scaling_agents(agent_counts=[10, 25, 50, 100, 200])
    
    # Profile step count scaling
    profiler.profile_scaling_steps(step_counts=[50, 100, 250, 500])
    
    # Profile environment size scaling
    profiler.profile_scaling_environment(
        env_sizes=[(50, 50), (100, 100), (200, 200), (500, 500)],
        num_agents=50,
        steps=50
    )
    
    # Profile memory growth
    profiler.profile_memory_over_time(num_agents=50, steps=500)
    
    # Profile CPU utilization
    profiler.profile_cpu_utilization(num_agents=100, steps=200)
    
    # Generate report
    profiler.generate_report()
    
    print("✓ System-level profiling complete!")
    print("  Results saved in profiler.results")


if __name__ == "__main__":
    main()
