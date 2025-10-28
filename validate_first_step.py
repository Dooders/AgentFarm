#!/usr/bin/env python3
"""
Validate that the first step between two simulations is exactly the same.
"""

import sys
import sqlite3
from pathlib import Path

def compare_first_step(db1_path, db2_path):
    """Compare the first step of two simulation databases."""

    print(f"🔍 Comparing First Steps Between:")
    print(f"   DB1: {db1_path}")
    print(f"   DB2: {db2_path}")
    print()

    conn1 = sqlite3.connect(db1_path)
    conn2 = sqlite3.connect(db2_path)

    try:
        # 1. Compare initial conditions
        print("1. INITIAL CONDITIONS COMPARISON")
        print("=" * 50)

        # Compare agents at step 0
        agents1 = conn1.execute("""
            SELECT agent_id, agent_type, position_x, position_y, initial_resources, starting_health
            FROM agents ORDER BY agent_id
        """).fetchall()

        agents2 = conn2.execute("""
            SELECT agent_id, agent_type, position_x, position_y, initial_resources, starting_health
            FROM agents ORDER BY agent_id
        """).fetchall()

        if agents1 == agents2:
            print(f"✅ Initial agents identical: {len(agents1)} agents")
        else:
            print(f"❌ Initial agents differ!")
            return False

        # Compare initial resources
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
            print(f"✅ Initial resources identical: {len(resources1)} resources")
        else:
            print(f"❌ Initial resources differ!")
            return False

        print()

        # 2. Compare first step actions
        print("2. FIRST STEP ACTIONS COMPARISON")
        print("=" * 50)

        actions1 = conn1.execute("""
            SELECT agent_id, step_number, action_type, action_target_id, reward, details
            FROM agent_actions WHERE step_number = 1
            ORDER BY agent_id
        """).fetchall()

        actions2 = conn2.execute("""
            SELECT agent_id, step_number, action_type, action_target_id, reward, details
            FROM agent_actions WHERE step_number = 1
            ORDER BY agent_id
        """).fetchall()

        if actions1 == actions2:
            print(f"✅ First step actions identical: {len(actions1)} actions")
        else:
            print(f"❌ First step actions differ!")
            print(f"   DB1: {len(actions1)} actions")
            print(f"   DB2: {len(actions2)} actions")
            return False

        print()

        # 3. Compare agent states after first step
        print("3. AGENT STATES AFTER FIRST STEP")
        print("=" * 50)

        states1 = conn1.execute("""
            SELECT agent_id, step_number, position_x, position_y, resource_level, current_health, total_reward
            FROM agent_states WHERE step_number = 1
            ORDER BY agent_id
        """).fetchall()

        states2 = conn2.execute("""
            SELECT agent_id, step_number, position_x, position_y, resource_level, current_health, total_reward
            FROM agent_states WHERE step_number = 1
            ORDER BY agent_id
        """).fetchall()

        if states1 == states2:
            print(f"✅ Agent states after step 1 identical: {len(states1)} states")
        else:
            print(f"❌ Agent states after step 1 differ!")
            print(f"   DB1: {len(states1)} states")
            print(f"   DB2: {len(states2)} states")
            return False

        print()

        # 4. Compare resource states after first step
        print("4. RESOURCE STATES AFTER FIRST STEP")
        print("=" * 50)

        res_states1 = conn1.execute("""
            SELECT resource_id, step_number, amount, position_x, position_y
            FROM resource_states WHERE step_number = 1
            ORDER BY resource_id
        """).fetchall()

        res_states2 = conn2.execute("""
            SELECT resource_id, step_number, amount, position_x, position_y
            FROM resource_states WHERE step_number = 1
            ORDER BY resource_id
        """).fetchall()

        if res_states1 == res_states2:
            print(f"✅ Resource states after step 1 identical: {len(res_states1)} resources")
        else:
            print(f"❌ Resource states after step 1 differ!")
            print(f"   DB1: {len(res_states1)} resources")
            print(f"   DB2: {len(res_states2)} resources")
            # Show first difference
            for i, (r1, r2) in enumerate(zip(res_states1, res_states2)):
                if r1 != r2:
                    print(f"   First difference at resource {i}:")
                    print(f"     DB1: {r1}")
                    print(f"     DB2: {r2}")
                    break
            return False

        print()

        # 5. Compare simulation metadata (ignore timestamps and simulation IDs)
        print("5. SIMULATION METADATA")
        print("=" * 50)

        meta1 = conn1.execute("SELECT end_time, status, parameters FROM simulations").fetchone()
        meta2 = conn2.execute("SELECT end_time, status, parameters FROM simulations").fetchone()

        if meta1 == meta2:
            print("✅ Simulation metadata identical (excluding timestamps and IDs)")
        else:
            print("❌ Simulation metadata differs")
            return False

        print()

        # 6. Final validation
        print("🎉 FIRST STEP VALIDATION: PASSED")
        print("=" * 50)
        print("✅ Initial conditions identical")
        print("✅ First step actions identical")
        print("✅ Agent states after step 1 identical")
        print("✅ Resource states after step 1 identical")
        print("✅ Simulation metadata identical")
        print()
        print("🎯 CONCLUSION: The first step between both simulations is EXACTLY the same!")

        return True

    finally:
        conn1.close()
        conn2.close()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python validate_first_step.py <db1_path> <db2_path>")
        sys.exit(1)

    db1 = sys.argv[1]
    db2 = sys.argv[2]

    if not Path(db1).exists():
        print(f"Error: {db1} does not exist")
        sys.exit(1)

    if not Path(db2).exists():
        print(f"Error: {db2} does not exist")
        sys.exit(1)

    success = compare_first_step(db1, db2)
    sys.exit(0 if success else 1)
