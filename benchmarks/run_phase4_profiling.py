#!/usr/bin/env python3
"""
Phase 4 Profiling Script: System-Level Performance and Scaling

This script runs comprehensive Phase 4 profiling for system-level analysis:
1. Scaling with agent count
2. Scaling with simulation steps
3. Scaling with environment size
4. Memory usage over time
5. CPU utilization
6. Production readiness assessment

Usage:
    python benchmarks/run_phase4_profiling.py
    python benchmarks/run_phase4_profiling.py --quick  # Faster, reduced scale
    python benchmarks/run_phase4_profiling.py --profile scaling  # Single test
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class Phase4Profiler:
    """Orchestrates Phase 4 system-level profiling."""

    def __init__(self, output_dir: str = "profiling_results/phase4", quick_mode: bool = False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.quick_mode = quick_mode
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "profiling_runs": [],
            "analysis": {},
        }

    def run_system_profiler(self):
        """Run the system profiler."""
        print(f"\n{'='*60}")
        print(f"Running System-Level Profiling")
        print(f"{'='*60}\n")

        output_file = self.output_dir / "system_profile.log"
        
        cmd = [
            sys.executable,
            "-m",
            "benchmarks.implementations.profiling.system_profiler",
        ]

        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,  # 30 minute timeout
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
                "type": "system_profiling",
                "duration": duration,
                "success": result.returncode == 0,
                "output_file": str(output_file),
            })

            print(f"✓ Completed in {duration:.2f}s")
            return result.returncode == 0

        except subprocess.TimeoutExpired:
            print(f"✗ Timeout after 30 minutes")
            return False
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run_all_profiles(self):
        """Run all system-level profiles."""
        print("\n" + "="*60)
        print("PHASE 4 PROFILING: System-Level Analysis")
        print("="*60)

        if self.quick_mode:
            print("\n🚀 Quick Mode: Reduced scale testing")
        else:
            print("\n📊 Full Mode: Comprehensive system analysis")

        # Run system profiler
        success = self.run_system_profiler()
        if not success:
            print("⚠️  Warning: System profiler failed")

    def analyze_results(self):
        """Analyze profiling results and generate insights."""
        print("\n" + "="*60)
        print("Analyzing System-Level Results")
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
        output_file = self.output_dir / "phase4_summary.json"
        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n📄 Results saved to: {output_file}")

    def generate_report(self):
        """Generate human-readable profiling report."""
        report_file = self.output_dir / "PHASE4_REPORT.md"
        
        with open(report_file, "w") as f:
            f.write("# Phase 4 System-Level Profiling Report\n\n")
            f.write(f"**Generated:** {self.results['timestamp']}\n\n")
            
            f.write("## Summary\n\n")
            analysis = self.results["analysis"]
            f.write(f"- **Total Runs:** {analysis['total_runs']}\n")
            f.write(f"- **Successful:** {analysis['successful_runs']}\n")
            f.write(f"- **Failed:** {analysis['failed_runs']}\n")
            f.write(f"- **Total Time:** {analysis['total_profiling_time']:.1f}s\n\n")
            
            f.write("## System Profiling\n\n")
            for run in self.results["profiling_runs"]:
                status = "✓" if run["success"] else "✗"
                f.write(f"### {status} {run['type']}\n\n")
                f.write(f"- **Duration:** {run['duration']:.2f}s\n")
                f.write(f"- **Log:** `{run['output_file']}`\n\n")
                
                # Try to extract key findings
                try:
                    with open(run['output_file']) as log:
                        content = log.read()
                        # Extract scaling analysis
                        if "Agent Count Scaling" in content:
                            f.write("**Key Findings:**\n```\n")
                            # Find and extract scaling section
                            lines = content.split('\n')
                            in_report = False
                            report_lines = []
                            for line in lines:
                                if "System-Level Profiling Report" in line:
                                    in_report = True
                                if in_report:
                                    report_lines.append(line)
                                    if len(report_lines) > 50:
                                        break
                            f.write('\n'.join(report_lines))
                            f.write("\n```\n\n")
                except:
                    pass

            f.write("## Scaling Analysis\n\n")
            f.write("### Agent Count Scaling\n\n")
            f.write("How does performance change as you add more agents?\n\n")
            f.write("- **Linear**: Time increases proportionally (good)\n")
            f.write("- **Sub-quadratic**: Time increases faster but manageable\n")
            f.write("- **Quadratic or worse**: Significant performance issues\n\n")
            
            f.write("### Step Count Scaling\n\n")
            f.write("How does performance change over longer simulations?\n\n")
            f.write("- Should be linear (constant time per step)\n")
            f.write("- Watch for memory leaks (increasing memory per step)\n")
            f.write("- Check for performance degradation over time\n\n")
            
            f.write("### Environment Size Scaling\n\n")
            f.write("How does performance change with world size?\n\n")
            f.write("- Spatial index performance\n")
            f.write("- Resource distribution overhead\n")
            f.write("- Agent movement patterns\n\n")
            
            f.write("## Resource Usage\n\n")
            f.write("### Memory\n\n")
            f.write("- **Initial**: Memory at simulation start\n")
            f.write("- **Peak**: Maximum memory during run\n")
            f.write("- **Growth rate**: Memory increase per step\n")
            f.write("- **Per agent**: Memory overhead per agent\n\n")
            
            f.write("### CPU\n\n")
            f.write("- **Utilization**: Percentage of CPU used\n")
            f.write("- **Cores**: Number of cores effectively used\n")
            f.write("- **Efficiency**: Single vs multi-core performance\n\n")
            
            f.write("## Production Readiness\n\n")
            f.write("Based on system profiling, assess:\n\n")
            f.write("1. **Scalability**: Can handle target workload?\n")
            f.write("2. **Stability**: Memory leaks or crashes?\n")
            f.write("3. **Performance**: Meets throughput requirements?\n")
            f.write("4. **Resource usage**: Acceptable CPU/memory?\n\n")
            
            f.write("## Recommendations\n\n")
            f.write("Based on scaling analysis:\n\n")
            f.write("- **Agent limit**: Maximum agents for target performance\n")
            f.write("- **Step limit**: Maximum steps before degradation\n")
            f.write("- **Environment size**: Optimal world dimensions\n")
            f.write("- **Hardware requirements**: CPU/RAM recommendations\n\n")
            
            f.write("## Next Steps\n\n")
            f.write("1. Review scaling curves and identify limits\n")
            f.write("2. Identify non-linear scaling components\n")
            f.write("3. Cross-reference with Phase 1-3 findings\n")
            f.write("4. Implement targeted optimizations\n")
            f.write("5. Re-run system profiling to validate\n")
            f.write("6. Establish performance benchmarks for CI/CD\n")

        print(f"📋 Report saved to: {report_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run Phase 4 system-level profiling"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: Reduced scale testing",
    )
    parser.add_argument(
        "--output-dir",
        default="profiling_results/phase4",
        help="Output directory for profiling results",
    )
    
    args = parser.parse_args()

    profiler = Phase4Profiler(output_dir=args.output_dir, quick_mode=args.quick)
    
    try:
        profiler.run_all_profiles()
        profiler.analyze_results()
        profiler.save_results()
        profiler.generate_report()
        
        print("\n" + "="*60)
        print("✓ Phase 4 Profiling Complete!")
        print("="*60)
        print(f"\nResults directory: {profiler.output_dir}")
        print("\nNext: Review system-level analysis and establish performance baselines")
        
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
