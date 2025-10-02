#!/usr/bin/env python3
"""
Phase 2 Profiling Script: Component-Level Profiling

This script runs comprehensive Phase 2 profiling to identify component bottlenecks:
1. Spatial index operations
2. Observation generation
3. Database logging
4. Decision module (optional)
5. Resource management (optional)

Usage:
    python benchmarks/run_phase2_profiling.py
    python benchmarks/run_phase2_profiling.py --quick  # Faster, core components only
    python benchmarks/run_phase2_profiling.py --component spatial  # Single component
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


class Phase2Profiler:
    """Orchestrates Phase 2 component-level profiling."""

    def __init__(self, output_dir: str = "profiling_results/phase2", quick_mode: bool = False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.quick_mode = quick_mode
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "profiling_runs": [],
            "analysis": {},
        }

    def run_component_profiler(self, component: str, script_path: str):
        """Run a component profiler script."""
        print(f"\n{'='*60}")
        print(f"Profiling Component: {component}")
        print(f"{'='*60}\n")

        output_file = self.output_dir / f"{component}_profile.log"
        
        cmd = [
            sys.executable,
            "-m",
            script_path.replace("/", ".").replace(".py", ""),
        ]

        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
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

            self.results["profiling_runs"].append({
                "component": component,
                "duration": duration,
                "success": result.returncode == 0,
                "output_file": str(output_file),
            })

            print(f"✓ Completed in {duration:.2f}s")
            return result.returncode == 0

        except subprocess.TimeoutExpired:
            print(f"✗ Timeout after 10 minutes")
            return False
        except Exception as e:
            print(f"✗ Error: {e}")
            return False

    def run_all_profilers(self, specific_component: str = None):
        """Run all component profilers (or a specific one)."""
        print("\n" + "="*60)
        print("PHASE 2 PROFILING: Component-Level Analysis")
        print("="*60)

        components = {
            "spatial": "benchmarks/implementations/profiling/spatial_index_profiler.py",
            "observation": "benchmarks/implementations/profiling/observation_profiler.py",
            "database": "benchmarks/implementations/profiling/database_profiler.py",
        }

        if self.quick_mode:
            print("\n🚀 Quick Mode: Core components only")
            # Only run the most critical components
            components = {
                "spatial": components["spatial"],
                "observation": components["observation"],
            }
        
        if specific_component:
            if specific_component not in components:
                print(f"✗ Unknown component: {specific_component}")
                print(f"Available: {', '.join(components.keys())}")
                return
            components = {specific_component: components[specific_component]}

        # Run each component profiler
        for component, script_path in components.items():
            success = self.run_component_profiler(component, script_path)
            if not success:
                print(f"⚠️  Warning: {component} profiler failed")
            time.sleep(2)  # Brief pause between profilers

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

        print(f"Total Component Profiles: {analysis['total_runs']}")
        print(f"  ✓ Successful: {analysis['successful_runs']}")
        print(f"  ✗ Failed: {analysis['failed_runs']}")
        print(f"Total Time: {analysis['total_profiling_time']:.1f}s")

    def save_results(self):
        """Save profiling results summary."""
        output_file = self.output_dir / "phase2_summary.json"
        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n📄 Results saved to: {output_file}")

    def generate_report(self):
        """Generate human-readable profiling report."""
        report_file = self.output_dir / "PHASE2_REPORT.md"
        
        with open(report_file, "w") as f:
            f.write("# Phase 2 Component-Level Profiling Report\n\n")
            f.write(f"**Generated:** {self.results['timestamp']}\n\n")
            
            f.write("## Summary\n\n")
            analysis = self.results["analysis"]
            f.write(f"- **Total Components:** {analysis['total_runs']}\n")
            f.write(f"- **Successful:** {analysis['successful_runs']}\n")
            f.write(f"- **Failed:** {analysis['failed_runs']}\n")
            f.write(f"- **Total Time:** {analysis['total_profiling_time']:.1f}s\n\n")
            
            f.write("## Component Profiles\n\n")
            for run in self.results["profiling_runs"]:
                status = "✓" if run["success"] else "✗"
                f.write(f"### {status} {run['component'].title()}\n\n")
                f.write(f"- **Duration:** {run['duration']:.2f}s\n")
                f.write(f"- **Log:** `{run['output_file']}`\n\n")
                
                # Try to extract key findings from log
                try:
                    with open(run['output_file']) as log:
                        content = log.read()
                        # Extract report sections if present
                        if "Profiling Report" in content:
                            report_start = content.find("Profiling Report")
                            report_section = content[report_start:report_start+2000]
                            f.write("**Key Findings:**\n```\n")
                            f.write(report_section[:1000])  # First 1000 chars
                            f.write("\n```\n\n")
                except:
                    pass

            f.write("## Next Steps\n\n")
            f.write("1. Review component-specific logs for detailed findings\n")
            f.write("2. Identify top bottlenecks within each component\n")
            f.write("3. Compare results with Phase 1 macro-level findings\n")
            f.write("4. Select specific functions for Phase 3 line-level profiling\n")
            f.write("5. Plan optimization strategies for highest-impact bottlenecks\n")

        print(f"📋 Report saved to: {report_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run Phase 2 component-level profiling"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: Profile core components only",
    )
    parser.add_argument(
        "--component",
        choices=["spatial", "observation", "database"],
        help="Profile a specific component only",
    )
    parser.add_argument(
        "--output-dir",
        default="profiling_results/phase2",
        help="Output directory for profiling results",
    )
    
    args = parser.parse_args()

    profiler = Phase2Profiler(output_dir=args.output_dir, quick_mode=args.quick)
    
    try:
        profiler.run_all_profilers(specific_component=args.component)
        profiler.analyze_results()
        profiler.save_results()
        profiler.generate_report()
        
        print("\n" + "="*60)
        print("✓ Phase 2 Profiling Complete!")
        print("="*60)
        print(f"\nResults directory: {profiler.output_dir}")
        print("\nNext: Review component reports and proceed to Phase 3 for line-level profiling")
        
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
