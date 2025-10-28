#!/usr/bin/env python3
"""
Test script to verify deterministic simulation behavior.
"""

import sys
import sqlite3
from pathlib import Path

def compare_simulations(db1_path, db2_path):
    """Compare two simulation databases for determinism."""
    
    print(f"\n=== Comparing Simulations ===")
    print(f"DB1: {db1_path}")
    print(f"DB2: {db2_path}")
    
    conn1 = sqlite3.connect(db1_path)
    conn2 = sqlite3.connect(db2_path)
    
    # Compare initial agents
    print("\n1. Comparing initial agents...")
    agents1 = conn1.execute("""
        SELECT agent_id, agent_type, position_x, position_y, initial_resources, starting_health
        FROM agents ORDER BY agent_id
    """).fetchall()
    
    agents2 = conn2.execute("""
        SELECT agent_id, agent_type, position_x, position_y, initial_resources, starting_health
        FROM agents ORDER BY agent_id
    """).fetchall()
    
    if agents1 == agents2:
        print(f"   ✓ Initial agents match ({len(agents1)} agents)")
    else:
        print(f"   ✗ Initial agents differ!")
        print(f"     DB1 has {len(agents1)} agents")
        print(f"     DB2 has {len(agents2)} agents")
        
        # Find differences
        for i, (a1, a2) in enumerate(zip(agents1, agents2)):
            if a1 != a2:
                print(f"     First difference at index {i}:")
                print(f"       DB1: {a1}")
                print(f"       DB2: {a2}")
                break
    
    # Compare resource states at step 0
    print("\n2. Comparing initial resources...")
    resources1 = conn1.execute("""
        SELECT resource_id, amount, position_x, position_y
        FROM resource_states WHERE step_number = 0
        ORDER BY resource_id
    """).fetchall()
    
    resources2 = conn2.execute("""
        SELECT resource_id, amount, position_x, position_y
        FROM resource_states WHERE step_number = 0
        ORDER BY resource_id
    """).fetchall()
    
    if resources1 == resources2:
        print(f"   ✓ Initial resources match ({len(resources1)} resources)")
    else:
        print(f"   ✗ Initial resources differ!")
        print(f"     DB1 has {len(resources1)} resources")
        print(f"     DB2 has {len(resources2)} resources")
        
        # Find first difference
        for i, (r1, r2) in enumerate(zip(resources1, resources2)):
            if r1 != r2:
                print(f"     First difference at index {i}:")
                print(f"       DB1: {r1}")
                print(f"       DB2: {r2}")
                break
    
    # Compare final states
    print("\n3. Comparing final states...")
    final_step1 = conn1.execute("SELECT MAX(step_number) FROM simulation_steps").fetchone()[0]
    final_step2 = conn2.execute("SELECT MAX(step_number) FROM simulation_steps").fetchone()[0]
    
    print(f"   DB1 final step: {final_step1}")
    print(f"   DB2 final step: {final_step2}")
    
    if final_step1 == final_step2:
        print(f"   ✓ Both simulations ran for {final_step1} steps")
    else:
        print(f"   ✗ Simulations ran for different number of steps!")
    
    # Compare agent counts
    print("\n4. Comparing agent counts...")
    count1 = len(agents1)
    count2 = len(agents2)
    
    if count1 == count2:
        print(f"   ✓ Both have {count1} total agents")
    else:
        print(f"   ✗ Different agent counts: DB1={count1}, DB2={count2}")
    
    conn1.close()
    conn2.close()
    
    # Determine if deterministic
    is_deterministic = (agents1 == agents2 and resources1 == resources2 and final_step1 == final_step2)
    
    print(f"\n=== Result: {'DETERMINISTIC ✓' if is_deterministic else 'NON-DETERMINISTIC ✗'} ===\n")
    
    return is_deterministic

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python test_determinism.py <db1_path> <db2_path>")
        sys.exit(1)
    
    db1 = sys.argv[1]
    db2 = sys.argv[2]
    
    if not Path(db1).exists():
        print(f"Error: {db1} does not exist")
        sys.exit(1)
    
    if not Path(db2).exists():
        print(f"Error: {db2} does not exist")
        sys.exit(1)
    
    is_det = compare_simulations(db1, db2)
    sys.exit(0 if is_det else 1)

