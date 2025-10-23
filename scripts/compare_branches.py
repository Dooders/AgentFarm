#!/usr/bin/env python3
"""
Compare simulation results between branches to verify behavioral parity.

This script helps identify differences in simulation outcomes between
the main and dev branches after the agent refactoring.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_simulation(branch: str, seed: int, steps: int) -> dict:
    """
    Run a simulation on a specific branch and return results.
    
    Args:
        branch: Git branch name
        seed: Random seed for determinism
        steps: Number of simulation steps
        
    Returns:
        Dictionary with simulation results
    """
    print(f"Running simulation on {branch} branch (seed={seed}, steps={steps})...")
    
    # Checkout branch
    result = subprocess.run(
        ["git", "checkout", branch],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"Error checking out {branch}: {result.stderr}")
        return None
    
    # Run simulation (don't use --skip-validation as it's not in main branch)
    result = subprocess.run(
        [
            sys.executable, 
            "run_simulation.py",
            "--steps", str(steps),
            "--seed", str(seed)
        ],
        capture_output=True,
        text=True,
        timeout=300
    )
    
    if result.returncode != 0:
        print(f"Simulation failed on {branch}: {result.stderr}")
        return None
    
    # Parse output
    output = result.stdout
    agent_count = None
    runtime = None
    
    for line in output.split('\n'):
        if "Final agent count:" in line:
            agent_count = int(line.split(':')[1].strip())
        if "Simulation completed in" in line:
            runtime = float(line.split()[3])
    
    return {
        "branch": branch,
        "seed": seed,
        "steps": steps,
        "agent_count": agent_count,
        "runtime": runtime,
        "output": output
    }


def compare_results(main_result: dict, dev_result: dict) -> None:
    """Compare and report differences between branch results."""
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    
    print(f"\nSeed: {main_result['seed']}, Steps: {main_result['steps']}")
    print(f"\nMain Branch:")
    print(f"  - Final Agents: {main_result['agent_count']}")
    print(f"  - Runtime: {main_result['runtime']:.2f}s")
    
    print(f"\nDev Branch:")
    print(f"  - Final Agents: {dev_result['agent_count']}")
    print(f"  - Runtime: {dev_result['runtime']:.2f}s")
    
    print(f"\nDifferences:")
    agent_diff = dev_result['agent_count'] - main_result['agent_count']
    runtime_diff = dev_result['runtime'] - main_result['runtime']
    
    if agent_diff == 0:
        print(f"  ✅ Agent Count: IDENTICAL")
    else:
        print(f"  ❌ Agent Count: {agent_diff:+d} agents ({dev_result['agent_count']} vs {main_result['agent_count']})")
    
    print(f"  ⏱️  Runtime: {runtime_diff:+.2f}s ({runtime_diff/main_result['runtime']*100:+.1f}%)")
    
    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Compare simulation results between branches"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44],
        help="Random seeds to test"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of simulation steps"
    )
    parser.add_argument(
        "--main-branch",
        type=str,
        default="main",
        help="Main branch name"
    )
    parser.add_argument(
        "--dev-branch", 
        type=str,
        default="dev",
        help="Development branch name"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file for results"
    )
    
    args = parser.parse_args()
    
    # Store current branch
    result = subprocess.run(
        ["git", "branch", "--show-current"],
        capture_output=True,
        text=True
    )
    original_branch = result.stdout.strip()
    
    try:
        all_results = []
        
        for seed in args.seeds:
            # Run on main
            main_result = run_simulation(args.main_branch, seed, args.steps)
            if not main_result:
                continue
            
            # Run on dev
            dev_result = run_simulation(args.dev_branch, seed, args.steps)
            if not dev_result:
                continue
            
            # Compare
            compare_results(main_result, dev_result)
            
            all_results.append({
                "seed": seed,
                "main": main_result,
                "dev": dev_result
            })
        
        # Save results if output specified
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"\n📊 Results saved to: {args.output}")
    
    finally:
        # Restore original branch
        subprocess.run(["git", "checkout", original_branch], capture_output=True)
        print(f"\n✅ Restored to {original_branch} branch")


if __name__ == "__main__":
    main()
