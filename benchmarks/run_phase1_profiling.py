#!/usr/bin/env python3
"""
Phase 1 Profiling Script: Macro-Level Bottleneck Identification

This script runs comprehensive Phase 1 profiling to identify major bottlenecks:
1. cProfile baseline with different agent counts
2. py-spy sampling profiles  
3. Generates analysis reports and recommendations

Usage:
    python benchmarks/run_phase1_profiling.py
    python benchmarks/run_phase1_profiling.py --quick  # Faster, fewer iterations
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class Phase1Profiler:
    """Orchestrates Phase 1 profiling runs."""

    def __init__(self, output_dir: str = "profiling_results/phase1", quick_mode: bool = False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.quick_mode = quick_mode
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "profiling_runs": [],
            "analysis": {},
        }

    def run_cprofile_baseline(self, steps: int, agents: int, name: str):
        """Run cProfile baseline profiling."""
        print(f"\n{'='*60}")
        print(f"Running cProfile: {name}")
        print(f"Steps: {steps}, Agents: {agents}")
        print(f"{'='*60}\n")

        output_base = self.output_dir / f"cprofile_{name}"
        log_file = output_base.with_suffix(".log")
        
        cmd = [
            sys.executable,
            "run_simulation.py",
            "--steps", str(steps),
            "--perf-profile",
            "--no-snakeviz",
            "--environment", "development",
        ]

        # Override agent count via environment variable if needed
        # For now, using config defaults
        
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
            )
            duration = time.time() - start_time
            
            # Save output
            with open(log_file, "w") as f:
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"Duration: {duration:.2f}s\n")
                f.write(f"Exit Code: {result.returncode}\n\n")
                f.write("=== STDOUT ===\n")
                f.write(result.stdout)
                f.write("\n=== STDERR ===\n")
                f.write(result.stderr)

            # Parse profiling output if available
            profile_stats_path = Path("simulations/profile_stats.txt")
            if profile_stats_path.exists():
                # Copy to our results directory
                dest = output_base.with_suffix(".profile.txt")
                with open(profile_stats_path) as src, open(dest, "w") as dst:
                    dst.write(src.read())

            profile_binary_path = Path("simulations/profile_stats.prof")
            if profile_binary_path.exists():
                # Copy binary profile
                dest = output_base.with_suffix(".prof")
                import shutil
                shutil.copy(profile_binary_path, dest)

            self.results["profiling_runs"].append({
                "type": "cprofile",
                "name": name,
                "steps": steps,
                "agents": agents,
                "duration": duration,
                "success": result.returncode == 0,
                "output_file": str(log_file),
            })

            print(f"✓ Completed in {duration:.2f}s")
            return result.returncode == 0

        except subprocess.TimeoutExpired:
            print(f"✗ Timeout after 10 minutes")
            return False
        except Exception as e:
            print(f"✗ Error: {e}")
            return False

    def run_pyspy_profile(self, steps: int, agents: int, name: str, duration: int = 60):
        """Run py-spy sampling profiler."""
        print(f"\n{'='*60}")
        print(f"Running py-spy: {name}")
        print(f"Steps: {steps}, Duration: {duration}s")
        print(f"{'='*60}\n")

        output_base = self.output_dir / f"pyspy_{name}"
        svg_file = output_base.with_suffix(".svg")
        speedscope_file = output_base.with_suffix(".speedscope.json")

        # Check if py-spy is available
        try:
            subprocess.run(["py-spy", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("✗ py-spy not found in PATH")
            return False

        # Run simulation with py-spy
        cmd = [
            "py-spy",
            "record",
            "-o", str(svg_file),
            "-f", "speedscope",
            "-o", str(speedscope_file),
            "--",
            sys.executable,
            "run_simulation.py",
            "--steps", str(steps),
            "--environment", "development",
        ]

        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=max(duration * 2, 300),  # At least 5 minutes
            )
            elapsed = time.time() - start_time

            self.results["profiling_runs"].append({
                "type": "pyspy",
                "name": name,
                "steps": steps,
                "agents": agents,
                "duration": elapsed,
                "success": result.returncode == 0,
                "svg_file": str(svg_file) if svg_file.exists() else None,
                "speedscope_file": str(speedscope_file) if speedscope_file.exists() else None,
            })

            if result.returncode == 0:
                print(f"✓ Completed in {elapsed:.2f}s")
                print(f"  Flame graph: {svg_file}")
                print(f"  Speedscope: {speedscope_file}")
            else:
                print(f"✗ Failed with exit code {result.returncode}")
                print(f"  Error: {result.stderr}")

            return result.returncode == 0

        except subprocess.TimeoutExpired:
            print(f"✗ Timeout")
            return False
        except Exception as e:
            print(f"✗ Error: {e}")
            return False

    def run_all_profiles(self):
        """Run all Phase 1 profiling configurations."""
        print("\n" + "="*60)
        print("PHASE 1 PROFILING: Macro-Level Bottleneck Identification")
        print("="*60)

        if self.quick_mode:
            print("\n🚀 Quick Mode: Running reduced profiling suite")
            configs = [
                ("baseline_small", 500, 50),
            ]
        else:
            print("\n📊 Full Mode: Running comprehensive profiling suite")
            configs = [
                ("baseline_small", 500, 50),
                ("baseline_medium", 1000, 100),
                ("baseline_large", 1000, 200),
            ]

        # Run cProfile baselines
        print("\n" + "="*60)
        print("Step 1: cProfile Baselines")
        print("="*60)
        
        for name, steps, agents in configs:
            success = self.run_cprofile_baseline(steps, agents, name)
            if not success and not self.quick_mode:
                print(f"⚠️  Warning: {name} failed, continuing anyway...")
            time.sleep(2)  # Brief pause between runs

        # Run py-spy profiles (if not quick mode)
        if not self.quick_mode:
            print("\n" + "="*60)
            print("Step 2: py-spy Sampling Profiles")
            print("="*60)

            pyspy_configs = [
                ("sampling_medium", 2000, 100, 90),
            ]

            for name, steps, agents, duration in pyspy_configs:
                success = self.run_pyspy_profile(steps, agents, name, duration)
                if not success:
                    print(f"⚠️  Warning: py-spy {name} failed, continuing anyway...")
                time.sleep(2)

    def analyze_results(self):
        """Analyze profiling results and generate insights."""
        print("\n" + "="*60)
        print("Analyzing Profiling Results")
        print("="*60 + "\n")

        analysis = {
            "total_runs": len(self.results["profiling_runs"]),
            "successful_runs": sum(1 for r in self.results["profiling_runs"] if r["success"]),
            "failed_runs": sum(1 for r in self.results["profiling_runs"] if not r["success"]),
            "total_profiling_time": sum(r["duration"] for r in self.results["profiling_runs"]),
        }

        self.results["analysis"] = analysis

        print(f"Total Profiling Runs: {analysis['total_runs']}")
        print(f"  ✓ Successful: {analysis['successful_runs']}")
        print(f"  ✗ Failed: {analysis['failed_runs']}")
        print(f"Total Time: {analysis['total_profiling_time']:.1f}s")

    def save_results(self):
        """Save profiling results summary."""
        output_file = self.output_dir / "phase1_summary.json"
        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n📄 Results saved to: {output_file}")

    def generate_report(self):
        """Generate human-readable profiling report."""
        report_file = self.output_dir / "PHASE1_REPORT.md"
        
        with open(report_file, "w") as f:
            f.write("# Phase 1 Profiling Report\n\n")
            f.write(f"**Generated:** {self.results['timestamp']}\n\n")
            
            f.write("## Summary\n\n")
            analysis = self.results["analysis"]
            f.write(f"- **Total Runs:** {analysis['total_runs']}\n")
            f.write(f"- **Successful:** {analysis['successful_runs']}\n")
            f.write(f"- **Failed:** {analysis['failed_runs']}\n")
            f.write(f"- **Total Time:** {analysis['total_profiling_time']:.1f}s\n\n")
            
            f.write("## Profiling Runs\n\n")
            for run in self.results["profiling_runs"]:
                status = "✓" if run["success"] else "✗"
                f.write(f"### {status} {run['name']} ({run['type']})\n\n")
                f.write(f"- **Steps:** {run['steps']}\n")
                f.write(f"- **Agents:** {run.get('agents', 'N/A')}\n")
                f.write(f"- **Duration:** {run['duration']:.2f}s\n")
                
                if run['type'] == 'cprofile' and 'output_file' in run:
                    f.write(f"- **Log:** `{run['output_file']}`\n")
                elif run['type'] == 'pyspy':
                    if run.get('svg_file'):
                        f.write(f"- **Flame Graph:** `{run['svg_file']}`\n")
                    if run.get('speedscope_file'):
                        f.write(f"- **Speedscope:** `{run['speedscope_file']}`\n")
                f.write("\n")

            f.write("## Next Steps\n\n")
            f.write("1. Review cProfile output files to identify top time-consuming functions\n")
            f.write("2. Open py-spy flame graphs in browser to visualize call stacks\n")
            f.write("3. Import speedscope files at https://www.speedscope.app/ for interactive analysis\n")
            f.write("4. Document top 10 bottlenecks in profiling plan\n")
            f.write("5. Proceed to Phase 2: Component-Level Profiling\n")

        print(f"📋 Report saved to: {report_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run Phase 1 profiling to identify macro-level bottlenecks"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: Run reduced profiling suite for faster results",
    )
    parser.add_argument(
        "--output-dir",
        default="profiling_results/phase1",
        help="Output directory for profiling results",
    )
    
    args = parser.parse_args()

    profiler = Phase1Profiler(output_dir=args.output_dir, quick_mode=args.quick)
    
    try:
        profiler.run_all_profiles()
        profiler.analyze_results()
        profiler.save_results()
        profiler.generate_report()
        
        print("\n" + "="*60)
        print("✓ Phase 1 Profiling Complete!")
        print("="*60)
        print(f"\nResults directory: {profiler.output_dir}")
        print("\nNext: Review the generated reports and proceed to Phase 2")
        
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
