#!/usr/bin/env python3
"""
Simple test runner for decision module tests to validate they work correctly.
"""

import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_imports():
    """Test that all decision test modules can be imported."""
    try:
        import tests.decision.test_decision_module

        print("✓ Decision module tests imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import decision module tests: {e}")
        return False

    try:
        import tests.decision.test_decision_config

        print("✓ Decision config tests imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import decision config tests: {e}")
        return False

    try:
        import tests.decision.test_decision_integration

        print("✓ Decision integration tests imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import decision integration tests: {e}")
        return False

    return True


def test_basic_functionality():
    """Test basic functionality of the decision module."""
    try:
        from unittest.mock import Mock

        from farm.core.decision.config import DecisionConfig
        from farm.core.decision.decision import DecisionModule

        print("✓ Core decision module imports successful")

        # Test config creation
        config = DecisionConfig()
        print(f"✓ DecisionConfig created: algorithm_type={config.algorithm_type}")

        # Test basic mock setup
        mock_agent = Mock()
        mock_agent.agent_id = "test_agent"
        mock_env = Mock()
        mock_env.action_space = Mock()
        mock_env.action_space.n = 4
        mock_agent.environment = mock_env

        # Test module creation (should work even without SB3)
        module = DecisionModule(mock_agent, config)
        print(f"✓ DecisionModule created: agent_id={module.agent_id}")

        return True

    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False


if __name__ == "__main__":
    print("Testing decision module test files...\n")

    success = True

    print("1. Testing imports...")
    if not test_imports():
        success = False

    print("\n2. Testing basic functionality...")
    if not test_basic_functionality():
        success = False

    print(f"\n{'='*50}")
    if success:
        print("✓ All tests passed! Decision module tests are ready.")
    else:
        print("✗ Some tests failed. Please check the errors above.")
        sys.exit(1)
