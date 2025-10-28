#!/usr/bin/env python3
"""
Validate that the tenth step between two simulations is exactly the same.
"""

import sys
import sqlite3
from pathlib import Path

def compare_tenth_step(db1_path, db2_path):
    """Compare the tenth step of two simulation databases."""

    print(f"🔍 Comparing Tenth Steps Between:")
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
            print(f"   DB1: {len(agents1)} agents")
            print(f"   DB2: {len(agents2)} agents")
            
            # Show first difference
            for i, (a1, a2) in enumerate(zip(agents1, agents2)):
                if a1 != a2:
                    print(f"   First difference at agent {i}:")
                    print(f"     DB1: {a1}")
                    print(f"     DB2: {a2}")
                    break
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

        # 2. Compare step-by-step progression
        print("2. STEP-BY-STEP PROGRESSION")
        print("=" * 50)

        for step in range(1, 11):  # Steps 1-10
            print(f"\n--- STEP {step} ---")
            
            # Agent states
            states1 = conn1.execute("""
                SELECT agent_id, position_x, position_y, resource_level, current_health, total_reward
                FROM agent_states WHERE step_number = ?
                ORDER BY agent_id
            """, (step,)).fetchall()

            states2 = conn2.execute("""
                SELECT agent_id, position_x, position_y, resource_level, current_health, total_reward
                FROM agent_states WHERE step_number = ?
                ORDER BY agent_id
            """, (step,)).fetchall()

            if states1 == states2:
                print(f"✅ Agent states identical: {len(states1)} states")
            else:
                print(f"❌ Agent states differ at step {step}!")
                print(f"   DB1: {len(states1)} states")
                print(f"   DB2: {len(states2)} states")
                
                # Show first difference
                for i, (s1, s2) in enumerate(zip(states1, states2)):
                    if s1 != s2:
                        print(f"   First difference at agent {i}:")
                        print(f"     DB1: {s1}")
                        print(f"     DB2: {s2}")
                        break
                return False

            # Resource states
            res_states1 = conn1.execute("""
                SELECT resource_id, amount, position_x, position_y
                FROM resource_states WHERE step_number = ?
                ORDER BY resource_id
            """, (step,)).fetchall()

            res_states2 = conn2.execute("""
                SELECT resource_id, amount, position_x, position_y
                FROM resource_states WHERE step_number = ?
                ORDER BY resource_id
            """, (step,)).fetchall()

            if res_states1 == res_states2:
                print(f"✅ Resource states identical: {len(res_states1)} resources")
            else:
                print(f"❌ Resource states differ at step {step}!")
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

            # Actions for this step
            actions1 = conn1.execute("""
                SELECT agent_id, action_type, action_target_id, reward, details
                FROM agent_actions WHERE step_number = ?
                ORDER BY agent_id
            """, (step,)).fetchall()

            actions2 = conn2.execute("""
                SELECT agent_id, action_type, action_target_id, reward, details
                FROM agent_actions WHERE step_number = ?
                ORDER BY agent_id
            """, (step,)).fetchall()

            if actions1 == actions2:
                print(f"✅ Actions identical: {len(actions1)} actions")
            else:
                print(f"❌ Actions differ at step {step}!")
                print(f"   DB1: {len(actions1)} actions")
                print(f"   DB2: {len(actions2)} actions")
                
                # Show first difference
                for i, (a1, a2) in enumerate(zip(actions1, actions2)):
                    if a1 != a2:
                        print(f"   First difference at action {i}:")
                        print(f"     DB1: {a1}")
                        print(f"     DB2: {a2}")
                        break
                return False

        print()

        # 3. Final validation
        print("🎉 TENTH STEP VALIDATION: PASSED")
        print("=" * 50)
        print("✅ Initial conditions identical")
        print("✅ All steps 1-10 identical")
        print("✅ Agent states identical at each step")
        print("✅ Resource states identical at each step")
        print("✅ Actions identical at each step")
        print()
        print("🎯 CONCLUSION: All steps up to and including step 10 are EXACTLY the same!")

        return True

    finally:
        conn1.close()
        conn2.close()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python validate_tenth_step.py <db1_path> <db2_path>")
        sys.exit(1)

    db1 = sys.argv[1]
    db2 = sys.argv[2]

    if not Path(db1).exists():
        print(f"Error: {db1} does not exist")
        sys.exit(1)

    if not Path(db2).exists():
        print(f"Error: {db2} does not exist")
        sys.exit(1)

    success = compare_tenth_step(db1, db2)
    sys.exit(0 if success else 1)
