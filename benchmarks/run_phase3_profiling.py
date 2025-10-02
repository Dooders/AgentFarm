#!/usr/bin/env python3
"""
Phase 3 Profiling Script: Micro-Level (Line-by-Line) Profiling

This script runs line-by-line profiling on hot functions identified in Phase 1 & 2:
1. Line profiler on specific functions
2. Memory profiler on memory-intensive functions
3. Generates detailed line-level reports

Usage:
    python benchmarks/run_phase3_profiling.py
    python benchmarks/run_phase3_profiling.py --function observe
    python benchmarks/run_phase3_profiling.py --memory
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class Phase3Profiler:
    """Orchestrates Phase 3 line-level profiling."""

    def __init__(self, output_dir: str = "profiling_results/phase3"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "profiling_runs": [],
            "analysis": {},
        }

    def create_profiled_script(self, function_name: str, script_type: str = "line") -> Path:
        """Create a temporary script with profiling decorators."""
        
        if function_name == "observe":
            script_content = '''#!/usr/bin/env python3
"""Line profile observation generation"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from farm.config import SimulationConfig
from farm.core.environment import Environment

# Import line_profiler
try:
    from line_profiler import profile
except ImportError:
    # Fallback decorator if line_profiler not available
    def profile(func):
        return func

# Patch the method with profiling decorator
original_get_observation = Environment._get_observation

@profile
def profiled_get_observation(self, agent_id):
    return original_get_observation(self, agent_id)

Environment._get_observation = profiled_get_observation

def main():
    """Run simulation with profiled observation generation."""
    config = SimulationConfig.from_centralized_config(environment="development")
    
    # Create environment
    env = Environment(
        width=100,
        height=100,
        resource_distribution={"type": "random", "amount": 100},
        config=config,
    )
    
    # Create test agents
    from farm.core.agent import BaseAgent
    for i in range(20):
        agent = BaseAgent(
            agent_id=env.get_next_agent_id(),
            position=(50.0, 50.0),
            resource_level=10,
            spatial_service=env.spatial_service,
            environment=env,
        )
        env.add_agent(agent)
    
    env.spatial_index.update()
    
    # Generate observations to profile
    print("Generating observations for profiling...")
    for _ in range(5):
        for agent_id in env.agents:
            _ = env._get_observation(agent_id)
    
    env.cleanup()
    print("Profiling complete!")

if __name__ == "__main__":
    main()
'''
        
        elif function_name == "agent_act":
            script_content = '''#!/usr/bin/env python3
"""Line profile agent action execution"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from farm.config import SimulationConfig
from farm.core.environment import Environment
from farm.core.agent import BaseAgent

try:
    from line_profiler import profile
except ImportError:
    def profile(func):
        return func

# Patch agent.act with profiling
original_act = BaseAgent.act

@profile
def profiled_act(self):
    return original_act(self)

BaseAgent.act = profiled_act

def main():
    """Run simulation with profiled agent actions."""
    config = SimulationConfig.from_centralized_config(environment="development")
    
    env = Environment(
        width=100,
        height=100,
        resource_distribution={"type": "random", "amount": 100},
        config=config,
    )
    
    # Create agents
    for i in range(10):
        agent = BaseAgent(
            agent_id=env.get_next_agent_id(),
            position=(50.0 + i, 50.0),
            resource_level=50,
            spatial_service=env.spatial_service,
            environment=env,
        )
        env.add_agent(agent)
    
    env.spatial_index.update()
    
    # Run agent actions
    print("Executing agent actions for profiling...")
    for _ in range(10):
        for agent in env.agent_objects:
            if agent.alive:
                agent.act()
        env.update()
    
    env.cleanup()
    print("Profiling complete!")

if __name__ == "__main__":
    main()
'''
        
        elif function_name == "spatial_update":
            script_content = '''#!/usr/bin/env python3
"""Line profile spatial index updates"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from farm.core.spatial import SpatialIndex

try:
    from line_profiler import profile
except ImportError:
    def profile(func):
        return func

# Patch spatial index update
original_update = SpatialIndex.update

@profile
def profiled_update(self):
    return original_update(self)

SpatialIndex.update = profiled_update

def main():
    """Profile spatial index updates."""
    # Create test entities
    class MockAgent:
        def __init__(self, agent_id, position):
            self.agent_id = agent_id
            self.position = position
            self.alive = True
    
    agents = [MockAgent(f"agent_{i}", (i*10, i*10)) for i in range(100)]
    
    # Create spatial index
    spatial_index = SpatialIndex(1000, 1000, enable_batch_updates=False)
    spatial_index.set_references(agents, [])
    
    print("Profiling spatial index updates...")
    # Profile multiple updates with position changes
    for iteration in range(10):
        # Move some agents
        for i in range(len(agents) // 2):
            agents[i].position = (i*10 + iteration, i*10 + iteration)
        
        spatial_index.mark_positions_dirty()
        spatial_index.update()
    
    print("Profiling complete!")

if __name__ == "__main__":
    main()
'''
        
        elif function_name == "database_log":
            script_content = '''#!/usr/bin/env python3
"""Line profile database logging"""
import sys
import os
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from farm.database.database import SimulationDatabase

try:
    from line_profiler import profile
except ImportError:
    def profile(func):
        return func

# Patch database logger methods
from farm.database.logger import DatabaseLogger
original_log_action = DatabaseLogger.log_agent_action

@profile
def profiled_log_action(self, step_number, agent_id, action_type, resources_before, 
                        resources_after, reward, details=None):
    return original_log_action(self, step_number, agent_id, action_type, 
                               resources_before, resources_after, reward, details)

DatabaseLogger.log_agent_action = profiled_log_action

def main():
    """Profile database logging operations."""
    # Create temp database
    temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    db_path = temp_db.name
    temp_db.close()
    
    try:
        db = SimulationDatabase(db_path=db_path, simulation_id="test_sim")
        
        print("Profiling database logging...")
        # Log many actions
        for i in range(1000):
            db.logger.log_agent_action(
                step_number=i,
                agent_id=f"agent_{i % 50}",
                action_type="move",
                resources_before=10.0,
                resources_after=9.5,
                reward=0.1,
                details={},
            )
        
        db.logger.flush_all_buffers()
        db.close()
        print("Profiling complete!")
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)

if __name__ == "__main__":
    main()
'''
        
        else:
            raise ValueError(f"Unknown function: {function_name}")
        
        # Write script to temp file
        script_file = self.output_dir / f"profile_{function_name}.py"
        with open(script_file, "w") as f:
            f.write(script_content)
        
        return script_file

    def run_line_profile(self, function_name: str):
        """Run line profiler on a specific function."""
        print(f"\n{'='*60}")
        print(f"Line Profiling: {function_name}")
        print(f"{'='*60}\n")

        # Create profiled script
        script_file = self.create_profiled_script(function_name, "line")
        output_file = self.output_dir / f"line_profile_{function_name}.txt"
        
        # Run with kernprof
        cmd = [
            "python3", "-m", "kernprof",
            "-l", "-v",
            str(script_file),
        ]

        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(Path(__file__).parent.parent),
            )
            duration = time.time() - start_time
            
            # Save output
            with open(output_file, "w") as f:
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"Duration: {duration:.2f}s\n")
                f.write(f"Exit Code: {result.returncode}\n\n")
                f.write("=== STDOUT ===\n")
                f.write(result.stdout)
                f.write("\n=== STDERR ===\n")
                f.write(result.stderr)
            
            # Also print to console
            if result.stdout:
                print(result.stdout)
            
            self.results["profiling_runs"].append({
                "type": "line_profile",
                "function": function_name,
                "duration": duration,
                "success": result.returncode == 0,
                "output_file": str(output_file),
            })

            print(f"✓ Completed in {duration:.2f}s")
            print(f"  Output: {output_file}")
            return result.returncode == 0

        except subprocess.TimeoutExpired:
            print(f"✗ Timeout after 5 minutes")
            return False
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run_memory_profile(self, function_name: str):
        """Run memory profiler on a specific function."""
        print(f"\n{'='*60}")
        print(f"Memory Profiling: {function_name}")
        print(f"{'='*60}\n")

        # Create profiled script (similar but for memory)
        script_file = self.create_profiled_script(function_name, "memory")
        output_file = self.output_dir / f"memory_profile_{function_name}.txt"
        
        # Run with memory profiler
        cmd = [
            "python3", "-m", "memory_profiler",
            str(script_file),
        ]

        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(Path(__file__).parent.parent),
            )
            duration = time.time() - start_time
            
            # Save output
            with open(output_file, "w") as f:
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"Duration: {duration:.2f}s\n")
                f.write(f"Exit Code: {result.returncode}\n\n")
                f.write("=== STDOUT ===\n")
                f.write(result.stdout)
                f.write("\n=== STDERR ===\n")
                f.write(result.stderr)
            
            if result.stdout:
                print(result.stdout)
            
            self.results["profiling_runs"].append({
                "type": "memory_profile",
                "function": function_name,
                "duration": duration,
                "success": result.returncode == 0,
                "output_file": str(output_file),
            })

            print(f"✓ Completed in {duration:.2f}s")
            return result.returncode == 0

        except subprocess.TimeoutExpired:
            print(f"✗ Timeout after 5 minutes")
            return False
        except Exception as e:
            print(f"✗ Error: {e}")
            return False

    def run_all_profiles(self, specific_function: str = None, memory_only: bool = False):
        """Run all line-level profiles."""
        print("\n" + "="*60)
        print("PHASE 3 PROFILING: Micro-Level (Line-by-Line)")
        print("="*60)

        # Target functions based on expected Phase 1/2 bottlenecks
        functions = ["observe", "agent_act", "spatial_update", "database_log"]
        
        if specific_function:
            if specific_function not in functions:
                print(f"✗ Unknown function: {specific_function}")
                print(f"Available: {', '.join(functions)}")
                return
            functions = [specific_function]

        # Run line profiles
        if not memory_only:
            for func in functions:
                success = self.run_line_profile(func)
                if not success:
                    print(f"⚠️  Warning: Line profile for {func} failed")
                time.sleep(2)

        # Run memory profiles (optional, takes longer)
        if memory_only:
            for func in functions:
                success = self.run_memory_profile(func)
                if not success:
                    print(f"⚠️  Warning: Memory profile for {func} failed")
                time.sleep(2)

    def analyze_results(self):
        """Analyze profiling results and generate insights."""
        print("\n" + "="*60)
        print("Analyzing Line-Level Profiling Results")
        print("="*60 + "\n")

        analysis = {
            "total_runs": len(self.results["profiling_runs"]),
            "successful_runs": sum(1 for r in self.results["profiling_runs"] if r["success"]),
            "failed_runs": sum(1 for r in self.results["profiling_runs"] if not r["success"]),
            "total_profiling_time": sum(r["duration"] for r in self.results["profiling_runs"]),
        }

        self.results["analysis"] = analysis

        print(f"Total Function Profiles: {analysis['total_runs']}")
        print(f"  ✓ Successful: {analysis['successful_runs']}")
        print(f"  ✗ Failed: {analysis['failed_runs']}")
        print(f"Total Time: {analysis['total_profiling_time']:.1f}s")

    def save_results(self):
        """Save profiling results summary."""
        output_file = self.output_dir / "phase3_summary.json"
        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n📄 Results saved to: {output_file}")

    def generate_report(self):
        """Generate human-readable profiling report."""
        report_file = self.output_dir / "PHASE3_REPORT.md"
        
        with open(report_file, "w") as f:
            f.write("# Phase 3 Line-Level Profiling Report\n\n")
            f.write(f"**Generated:** {self.results['timestamp']}\n\n")
            
            f.write("## Summary\n\n")
            analysis = self.results["analysis"]
            f.write(f"- **Total Profiles:** {analysis['total_runs']}\n")
            f.write(f"- **Successful:** {analysis['successful_runs']}\n")
            f.write(f"- **Failed:** {analysis['failed_runs']}\n")
            f.write(f"- **Total Time:** {analysis['total_profiling_time']:.1f}s\n\n")
            
            f.write("## Function Profiles\n\n")
            for run in self.results["profiling_runs"]:
                status = "✓" if run["success"] else "✗"
                f.write(f"### {status} {run['function']} ({run['type']})\n\n")
                f.write(f"- **Duration:** {run['duration']:.2f}s\n")
                f.write(f"- **Output:** `{run['output_file']}`\n\n")
                
                # Try to extract key lines from output
                try:
                    with open(run['output_file']) as log:
                        content = log.read()
                        # Extract line profile data if present
                        if "Line #" in content and "Hits" in content:
                            # Find the profile table
                            lines = content.split('\n')
                            in_profile = False
                            profile_lines = []
                            for line in lines:
                                if "Line #" in line and "Hits" in line:
                                    in_profile = True
                                if in_profile:
                                    profile_lines.append(line)
                                    if len(profile_lines) > 30:  # Limit output
                                        break
                            
                            if profile_lines:
                                f.write("**Profile Data (top 30 lines):**\n```\n")
                                f.write('\n'.join(profile_lines))
                                f.write("\n```\n\n")
                except:
                    pass

            f.write("## How to Interpret Results\n\n")
            f.write("### Line Profiler Output\n\n")
            f.write("- **Line #**: Line number in source code\n")
            f.write("- **Hits**: Number of times line was executed\n")
            f.write("- **Time**: Total time spent on that line (microseconds)\n")
            f.write("- **Per Hit**: Average time per execution\n")
            f.write("- **% Time**: Percentage of total function time\n\n")
            f.write("**Focus on:**\n")
            f.write("- Lines with high % Time (major contributors)\n")
            f.write("- Lines with high Hits and significant time (optimization opportunity)\n")
            f.write("- Unexpected slow lines (algorithmic issues)\n\n")
            
            f.write("### Memory Profiler Output\n\n")
            f.write("- **Mem usage**: Memory used at that point\n")
            f.write("- **Increment**: Memory added/removed by that line\n\n")
            f.write("**Focus on:**\n")
            f.write("- Large increments (memory allocations)\n")
            f.write("- Lines with repeated allocations\n")
            f.write("- Memory leaks (increasing without decreasing)\n\n")
            
            f.write("## Next Steps\n\n")
            f.write("1. Review line-by-line profiles for each function\n")
            f.write("2. Identify specific lines consuming most time\n")
            f.write("3. Analyze why those lines are slow:\n")
            f.write("   - Algorithmic complexity?\n")
            f.write("   - Unnecessary computations?\n")
            f.write("   - Inefficient data structures?\n")
            f.write("   - Too many allocations?\n")
            f.write("4. Plan specific optimizations for hot lines\n")
            f.write("5. Implement and validate improvements with benchmarks\n")

        print(f"📋 Report saved to: {report_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run Phase 3 line-level profiling on hot functions"
    )
    parser.add_argument(
        "--function",
        choices=["observe", "agent_act", "spatial_update", "database_log"],
        help="Profile a specific function only",
    )
    parser.add_argument(
        "--memory",
        action="store_true",
        help="Run memory profiler instead of line profiler",
    )
    parser.add_argument(
        "--output-dir",
        default="profiling_results/phase3",
        help="Output directory for profiling results",
    )
    
    args = parser.parse_args()

    profiler = Phase3Profiler(output_dir=args.output_dir)
    
    try:
        profiler.run_all_profiles(
            specific_function=args.function,
            memory_only=args.memory
        )
        profiler.analyze_results()
        profiler.save_results()
        profiler.generate_report()
        
        print("\n" + "="*60)
        print("✓ Phase 3 Profiling Complete!")
        print("="*60)
        print(f"\nResults directory: {profiler.output_dir}")
        print("\nNext: Review line-level profiles and implement optimizations")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Profiling interrupted by user")
        profiler.save_results()
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Profiling failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
