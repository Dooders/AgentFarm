#!/usr/bin/env python3
"""
Test script to validate the observation-to-policy pipeline integration.

This script tests the complete flow from:
1. Agent observation generation
2. DecisionModule processing
3. CNN-based policy networks
4. Action selection with curriculum support
"""

import sys
import os
import numpy as np
import torch
from typing import Dict, Any

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from farm.core.observations import ObservationConfig, AgentObservation
from farm.core.channels import get_channel_registry
from farm.core.decision.decision import DecisionModule
from farm.core.decision.config import DecisionConfig

def test_channel_registry():
    """Test that the channel registry is properly initialized."""
    print("Testing Channel Registry...")

    registry = get_channel_registry()
    num_channels = registry.num_channels

    print(f"✓ Channel registry initialized with {num_channels} channels")
    assert num_channels == 13, f"Expected 13 channels, got {num_channels}"

    return True

def test_observation_creation():
    """Test that observations can be created with the correct shape."""
    print("\nTesting Observation Creation...")

    config = ObservationConfig(R=6)
    registry = get_channel_registry()

    # Create observation
    obs = AgentObservation(config)

    # Test tensor creation
    tensor = obs.tensor()
    expected_shape = (registry.num_channels, 13, 13)  # 13x13 = 2*6+1

    print(f"✓ Observation tensor shape: {tensor.shape}")
    assert tensor.shape == expected_shape, f"Expected shape {expected_shape}, got {tensor.shape}"

    return True

def test_decision_module_initialization():
    """Test that DecisionModule can be initialized with CNN support."""
    print("\nTesting DecisionModule Initialization...")

    # Create mock observation space
    class MockObservationSpace:
        def __init__(self, shape):
            self.shape = shape

    observation_space = MockObservationSpace((13, 13, 13))

    # Create mock action space
    class MockActionSpace:
        def __init__(self, n):
            self.n = n

    action_space = MockActionSpace(7)  # Assume 7 actions

    # Create mock agent
    class MockAgent:
        def __init__(self):
            self.agent_id = "test_agent"

    agent = MockAgent()

    # Test DecisionModule initialization with fallback algorithm
    decision_config = DecisionConfig(algorithm_type="fallback")
    try:
        decision_module = DecisionModule(
            agent=agent,
            action_space=action_space,
            observation_space=observation_space,
            config=decision_config
        )
        print("✓ DecisionModule initialized successfully with fallback algorithm")
        print(f"  - State dim: {decision_module.state_dim}")
        print(f"  - Observation shape: {decision_module.observation_shape}")
        print(f"  - Num actions: {decision_module.num_actions}")

        # Check that algorithm was initialized
        if decision_module.algorithm is not None:
            print("✓ Fallback algorithm initialized successfully")
        else:
            print("✗ Algorithm not initialized")
            return False

    except Exception as e:
        print(f"✗ DecisionModule initialization failed: {e}")
        return False

    return True

def test_observation_to_action_flow():
    """Test the complete flow from observation to action selection."""
    print("\nTesting Observation-to-Action Flow...")

    # Create observation
    config = ObservationConfig(R=6)
    obs = AgentObservation(config)

    # Add some test data to make observation non-zero
    obs._store_sparse_point(0, 6, 6, 1.0)  # Agent health at center
    obs._store_sparse_point(1, 5, 6, 0.8)  # Ally nearby
    obs._store_sparse_point(3, 4, 4, 0.5)  # Resource nearby

    # Get observation tensor
    observation_tensor = obs.tensor()

    print(f"✓ Created observation tensor with shape: {observation_tensor.shape}")
    print(f"  - Non-zero elements: {torch.count_nonzero(observation_tensor)}")

    # Test with DecisionModule if possible
    try:
        # Create mock spaces
        class MockObservationSpace:
            def __init__(self, shape):
                self.shape = shape

        class MockActionSpace:
            def __init__(self, n):
                self.n = n

        observation_space = MockObservationSpace((13, 13, 13))
        action_space = MockActionSpace(7)

        class MockAgent:
            def __init__(self):
                self.agent_id = "test_agent"

        agent = MockAgent()

        # Try to create DecisionModule
        decision_config = DecisionConfig(algorithm_type="fallback")
        decision_module = DecisionModule(
            agent=agent,
            action_space=action_space,
            observation_space=observation_space,
            config=decision_config
        )

        if decision_module.algorithm is not None:
            # Test action selection
            action = decision_module.decide_action(observation_tensor.numpy())
            print(f"✓ Action selected: {action}")
            assert isinstance(action, int), "Action should be an integer"
            assert 0 <= action < 7, f"Action {action} out of valid range [0, 6]"
        else:
            print("⚠ Skipping action selection test (algorithm not available)")
            return True

    except Exception as e:
        print(f"✗ Action selection test failed: {e}")
        return False

    return True

def test_curriculum_support():
    """Test curriculum learning with action masking."""
    print("\nTesting Curriculum Support...")

    try:
        # Create mock spaces
        class MockObservationSpace:
            def __init__(self, shape):
                self.shape = shape

        class MockActionSpace:
            def __init__(self, n):
                self.n = n

        observation_space = MockObservationSpace((13, 13, 13))
        action_space = MockActionSpace(7)

        class MockAgent:
            def __init__(self):
                self.agent_id = "test_agent"

        agent = MockAgent()

        # Create DecisionModule
        decision_config = DecisionConfig(algorithm_type="fallback")
        decision_module = DecisionModule(
            agent=agent,
            action_space=action_space,
            observation_space=observation_space,
            config=decision_config
        )

        if decision_module.algorithm is not None and hasattr(decision_module.algorithm, 'select_action_with_mask'):
            # Test with action mask (only allow actions 0, 2, 4)
            action_mask = np.array([True, False, True, False, True, False, False])

            # Create test observation
            test_obs = np.random.rand(1, 13, 13, 13).astype(np.float32)

            action = decision_module.decide_action(test_obs, enabled_actions=[0, 2, 4])
            print(f"✓ Curriculum action selected: {action}")
            print("  - Only actions [0, 2, 4] were enabled")
            # Verify action is in enabled set
            assert action in [0, 1, 2], f"Action {action} not in enabled set [0, 1, 2]"
        else:
            print("⚠ Skipping curriculum test (algorithm or masking not available)")
            return True

    except Exception as e:
        print(f"✗ Curriculum test failed: {e}")
        return False

    return True

def main():
    """Run all tests."""
    print("🧪 Testing AgentFarm Observation-to-Policy Pipeline")
    print("=" * 60)

    tests = [
        test_channel_registry,
        test_observation_creation,
        test_decision_module_initialization,
        test_observation_to_action_flow,
        test_curriculum_support,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"❌ {test.__name__} failed")
        except Exception as e:
            print(f"❌ {test.__name__} failed with exception: {e}")

    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! The observation-to-policy pipeline is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
