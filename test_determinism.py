#!/usr/bin/env python3
"""
Script to test simulation determinism by running two identical simulations
with the same seed and comparing their results.
"""

import os
import sys
import subprocess
import json
import time
from datetime import datetime
from pathlib import Path

# Set PYTHONHASHSEED for deterministic hash operations
os.environ['PYTHONHASHSEED'] = '0'

def run_simulation(seed=42, steps=1000, run_id=1):
    """
    Run a single simulation with specified parameters.
    
    Returns:
        dict: Dictionary containing simulation results and metadata
    """
    print(f"Running simulation {run_id} with seed={seed}, steps={steps}")
    
    # Create output directory for this run
    output_dir = f"simulations/determinism_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{run_id}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Command to run simulation
    cmd = [
        sys.executable, "run_simulation.py",
        "--environment", "development",
        "--profile", "simulation", 
        "--steps", str(steps),
        "--seed", str(seed),
        "--in-memory",
        "--no-persist",
        "--log-level", "INFO"
    ]
    
    # Set environment variables for determinism
    env = os.environ.copy()
    env['PYTHONHASHSEED'] = '0'
    env['CUDA_VISIBLE_DEVICES'] = ''  # Use CPU for maximum reproducibility
    
    print(f"Command: {' '.join(cmd)}")
    
    # Run simulation and capture output
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minute timeout
            env=env  # Pass environment with PYTHONHASHSEED
        )
        end_time = time.time()
        
        # Parse output for key metrics
        output_lines = result.stdout.split('\n')
        
        # Extract key information from output
        final_agent_count = None
        simulation_time = None
        early_termination = False
        
        for line in output_lines:
            if "Final agent count:" in line:
                try:
                    final_agent_count = int(line.split("Final agent count:")[1].strip().split()[0])
                except:
                    pass
            elif "Simulation completed in" in line:
                try:
                    simulation_time = float(line.split("Simulation completed in")[1].strip().split()[0])
                except:
                    pass
            elif "Simulation stopped early" in line:
                early_termination = True
        
        return {
            "run_id": run_id,
            "seed": seed,
            "steps": steps,
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "final_agent_count": final_agent_count,
            "simulation_time": simulation_time,
            "early_termination": early_termination,
            "wall_time": end_time - start_time,
            "success": result.returncode == 0,
            "output_dir": output_dir
        }
        
    except subprocess.TimeoutExpired:
        print(f"Simulation {run_id} timed out after 30 minutes")
        return {
            "run_id": run_id,
            "seed": seed,
            "steps": steps,
            "return_code": -1,
            "stdout": "",
            "stderr": "Timeout after 30 minutes",
            "final_agent_count": None,
            "simulation_time": None,
            "early_termination": False,
            "wall_time": 1800,
            "success": False,
            "output_dir": output_dir
        }
    except Exception as e:
        print(f"Error running simulation {run_id}: {e}")
        return {
            "run_id": run_id,
            "seed": seed,
            "steps": steps,
            "return_code": -1,
            "stdout": "",
            "stderr": str(e),
            "final_agent_count": None,
            "simulation_time": None,
            "early_termination": False,
            "wall_time": 0,
            "success": False,
            "output_dir": output_dir
        }

def compare_results(result1, result2):
    """
    Compare two simulation results to check for determinism.
    
    Returns:
        dict: Comparison results
    """
    comparison = {
        "deterministic": True,
        "differences": [],
        "summary": {}
    }
    
    # Check return codes
    if result1["return_code"] != result2["return_code"]:
        comparison["deterministic"] = False
        comparison["differences"].append(f"Return codes differ: {result1['return_code']} vs {result2['return_code']}")
    
    # Check final agent counts
    if result1["final_agent_count"] != result2["final_agent_count"]:
        comparison["deterministic"] = False
        comparison["differences"].append(f"Final agent counts differ: {result1['final_agent_count']} vs {result2['final_agent_count']}")
    
    # Check early termination
    if result1["early_termination"] != result2["early_termination"]:
        comparison["deterministic"] = False
        comparison["differences"].append(f"Early termination differs: {result1['early_termination']} vs {result2['early_termination']}")
    
    # Check simulation times (allow small differences due to timing variations)
    if result1["simulation_time"] is not None and result2["simulation_time"] is not None:
        time_diff = abs(result1["simulation_time"] - result2["simulation_time"])
        if time_diff > 1.0:  # Allow 1 second difference
            comparison["differences"].append(f"Simulation times differ significantly: {result1['simulation_time']:.2f}s vs {result2['simulation_time']:.2f}s (diff: {time_diff:.2f}s)")
            # Note: Time differences don't necessarily indicate non-determinism
    
    # Check success status
    if result1["success"] != result2["success"]:
        comparison["deterministic"] = False
        comparison["differences"].append(f"Success status differs: {result1['success']} vs {result2['success']}")
    
    # Summary
    comparison["summary"] = {
        "run1_final_agents": result1["final_agent_count"],
        "run2_final_agents": result2["final_agent_count"],
        "run1_time": result1["simulation_time"],
        "run2_time": result2["simulation_time"],
        "run1_success": result1["success"],
        "run2_success": result2["success"],
        "run1_wall_time": result1["wall_time"],
        "run2_wall_time": result2["wall_time"]
    }
    
    return comparison

def main():
    """Main function to run determinism test."""
    print("=" * 60)
    print("SIMULATION DETERMINISM TEST")
    print("=" * 60)
    print(f"Test started at: {datetime.now()}")
    print(f"Running 2 simulations with 1,000 steps each, seed=42")
    print()
    
    # Run first simulation
    print("Starting first simulation...")
    result1 = run_simulation(seed=42, steps=1000, run_id=1)
    
    print(f"First simulation completed:")
    print(f"  Success: {result1['success']}")
    print(f"  Final agent count: {result1['final_agent_count']}")
    print(f"  Simulation time: {result1['simulation_time']:.2f}s")
    print(f"  Wall time: {result1['wall_time']:.2f}s")
    print(f"  Early termination: {result1['early_termination']}")
    print()
    
    # Run second simulation
    print("Starting second simulation...")
    result2 = run_simulation(seed=42, steps=1000, run_id=2)
    
    print(f"Second simulation completed:")
    print(f"  Success: {result2['success']}")
    print(f"  Final agent count: {result2['final_agent_count']}")
    print(f"  Simulation time: {result2['simulation_time']:.2f}s")
    print(f"  Wall time: {result2['wall_time']:.2f}s")
    print(f"  Early termination: {result2['early_termination']}")
    print()
    
    # Compare results
    print("Comparing results...")
    comparison = compare_results(result1, result2)
    
    print("=" * 60)
    print("DETERMINISM TEST RESULTS")
    print("=" * 60)
    
    if comparison["deterministic"]:
        print("✅ SIMULATIONS ARE DETERMINISTIC")
        print("Both simulations produced identical results.")
    else:
        print("❌ SIMULATIONS ARE NOT DETERMINISTIC")
        print("Differences found:")
        for diff in comparison["differences"]:
            print(f"  - {diff}")
    
    print()
    print("Summary:")
    print(f"  Run 1 final agents: {comparison['summary']['run1_final_agents']}")
    print(f"  Run 2 final agents: {comparison['summary']['run2_final_agents']}")
    print(f"  Run 1 simulation time: {comparison['summary']['run1_time']:.2f}s")
    print(f"  Run 2 simulation time: {comparison['summary']['run2_time']:.2f}s")
    print(f"  Run 1 wall time: {comparison['summary']['run1_wall_time']:.2f}s")
    print(f"  Run 2 wall time: {comparison['summary']['run2_wall_time']:.2f}s")
    
    # Save detailed results
    results_file = f"determinism_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    detailed_results = {
        "test_info": {
            "timestamp": datetime.now().isoformat(),
            "seed": 42,
            "steps": 1000,
            "deterministic": comparison["deterministic"]
        },
        "run1": result1,
        "run2": result2,
        "comparison": comparison
    }
    
    with open(results_file, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")
    print(f"Test completed at: {datetime.now()}")
    
    return comparison["deterministic"]

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
