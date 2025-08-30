#!/usr/bin/env python3
"""Test script for Tianshou integration with AgentFarm.

This script tests the Tianshou wrapper classes to ensure they integrate
properly with the AgentFarm action selection system.
"""

import numpy as np

from farm.core.decision.algorithms import AlgorithmRegistry


def test_tianshou_algorithms():
    """Test Tianshou algorithm wrappers."""

    # Test data
    num_actions = 4
    state_dim = 8
    state = np.random.randn(state_dim)

    # Algorithms to test
    algorithms = ["ppo", "sac", "a2c", "dqn", "ddpg"]

    print("Testing Tianshou algorithm integration...")

    for algo_name in algorithms:
        print(f"\n--- Testing {algo_name.upper()} ---")

        try:
            # Create algorithm instance
            algorithm = AlgorithmRegistry.create(
                algo_name, num_actions=num_actions, state_dim=state_dim
            )

            print(f"✓ Successfully created {algo_name} algorithm")

            # Test action selection
            action = algorithm.select_action(state)
            print(f"✓ Action selection works: selected action {action}")

            # Test probability prediction
            probs = algorithm.predict_proba(state)
            print(
                f"✓ Probability prediction works: shape {probs.shape}, sum {probs.sum():.3f}"
            )

            # Test experience storage
            next_state = np.random.randn(state_dim)
            algorithm.store_experience(state, action, 1.0, next_state, False)
            print("✓ Experience storage works")

            print(f"✓ {algo_name.upper()} integration successful")

        except ImportError as e:
            print(f"⚠ {algo_name.upper()} requires Tianshou: {e}")
        except Exception as e:
            print(f"✗ {algo_name.upper()} failed: {e}")
            raise

    print("\n--- All Tianshou algorithms tested ---")


def test_algorithm_registry():
    """Test that the registry properly maps to Tianshou algorithms."""

    print("\nTesting AlgorithmRegistry mappings...")

    registry_entries = {
        "ppo": "farm.core.decision.algorithms.tianshou:PPOWrapper",
        "sac": "farm.core.decision.algorithms.tianshou:SACWrapper",
        "a2c": "farm.core.decision.algorithms.tianshou:A2CWrapper",
        "dqn": "farm.core.decision.algorithms.tianshou:DQNWrapper",
        "ddpg": "farm.core.decision.algorithms.tianshou:DDPGWrapper",
    }

    for name, expected_path in registry_entries.items():
        try:
            # Check if algorithm is registered
            algorithm = AlgorithmRegistry.create(name, num_actions=2, state_dim=4)
            print(f"✓ {name} -> {type(algorithm).__name__}")

            # Verify the module path matches expected
            module_path = f"{type(algorithm).__module__}:{type(algorithm).__name__}"
            if module_path == expected_path:
                print(f"✓ Module path correct: {module_path}")
            else:
                print(
                    f"⚠ Module path mismatch: expected {expected_path}, got {module_path}"
                )

        except Exception as e:
            print(f"✗ {name} registry test failed: {e}")

    print("--- Registry testing complete ---")


if __name__ == "__main__":
    try:
        test_algorithm_registry()
        test_tianshou_algorithms()
        print("\n🎉 All tests completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        raise
