#!/usr/bin/env python3
"""
Test suite for Pydantic validation in Hydra configuration system.
"""

import sys
import os
sys.path.append('/workspace')

from farm.core.config_hydra_simple import create_simple_hydra_config_manager
from farm.core.config_hydra_models import (
    HydraSimulationConfig,
    HydraEnvironmentConfig,
    HydraAgentConfig,
    validate_config_dict,
    validate_environment_config,
    validate_agent_config
)
from pydantic import ValidationError


def test_pydantic_validation():
    """Test Pydantic validation functionality."""
    print("Testing Pydantic validation...")
    
    # Test 1: Valid configuration
    print("\n1. Testing valid configuration...")
    config_manager = create_simple_hydra_config_manager(
        config_dir="/workspace/config_hydra/conf",
        environment="development",
        agent="system_agent"
    )
    
    try:
        validated_config = config_manager.get_validated_config()
        print(f"   ✅ Valid configuration passed validation")
        print(f"   Simulation ID: {validated_config.simulation_id}")
        print(f"   Environment: {validated_config.width}x{validated_config.height}")
        print(f"   Max steps: {validated_config.max_steps}")
    except ValidationError as e:
        print(f"   ❌ Valid configuration failed validation: {e}")
        return False
    
    # Test 2: Configuration validation method
    print("\n2. Testing configuration validation method...")
    errors = config_manager.validate_configuration()
    if not errors:
        print("   ✅ Configuration validation passed")
    else:
        print(f"   ❌ Configuration validation failed: {errors}")
        return False
    
    # Test 3: Environment validation
    print("\n3. Testing environment validation...")
    env_errors = config_manager.validate_environment_config()
    if not env_errors:
        print("   ✅ Environment validation passed")
    else:
        print(f"   ❌ Environment validation failed: {env_errors}")
        return False
    
    # Test 4: Agent validation
    print("\n4. Testing agent validation...")
    agent_errors = config_manager.validate_agent_config()
    if not agent_errors:
        print("   ✅ Agent validation passed")
    else:
        print(f"   ❌ Agent validation failed: {agent_errors}")
        return False
    
    return True


def test_pydantic_models_directly():
    """Test Pydantic models directly."""
    print("\nTesting Pydantic models directly...")
    
    # Test 1: Valid HydraSimulationConfig
    print("\n1. Testing valid HydraSimulationConfig...")
    try:
        config = HydraSimulationConfig(
            width=100,
            height=100,
            max_steps=1000,
            system_agents=10,
            independent_agents=10,
            control_agents=10
        )
        print(f"   ✅ Valid HydraSimulationConfig created")
        print(f"   Environment: {config.width}x{config.height}")
        print(f"   Max steps: {config.max_steps}")
    except ValidationError as e:
        print(f"   ❌ Valid HydraSimulationConfig failed: {e}")
        return False
    
    # Test 2: Invalid configuration (negative values)
    print("\n2. Testing invalid configuration (negative values)...")
    try:
        invalid_config = HydraSimulationConfig(
            width=-100,  # Invalid: negative width
            height=100,
            max_steps=1000
        )
        print("   ❌ Invalid configuration should have failed")
        return False
    except ValidationError as e:
        print(f"   ✅ Invalid configuration correctly rejected: {len(e.errors())} errors")
        for error in e.errors():
            print(f"     - {error['loc']}: {error['msg']}")
    
    # Test 3: Invalid agent ratios
    print("\n3. Testing invalid agent ratios...")
    try:
        invalid_config = HydraSimulationConfig(
            width=100,
            height=100,
            max_steps=1000,
            agent_type_ratios={
                "SystemAgent": 0.5,
                "IndependentAgent": 0.5,
                "ControlAgent": 0.5  # Total > 1.0
            }
        )
        print("   ❌ Invalid agent ratios should have failed")
        return False
    except ValidationError as e:
        print(f"   ✅ Invalid agent ratios correctly rejected: {len(e.errors())} errors")
        for error in e.errors():
            print(f"     - {error['loc']}: {error['msg']}")
    
    # Test 4: Valid environment config
    print("\n4. Testing valid environment config...")
    try:
        env_config = HydraEnvironmentConfig(
            debug=True,
            max_steps=500,
            learning_rate=0.01
        )
        print(f"   ✅ Valid environment config created")
        print(f"   Debug: {env_config.debug}")
        print(f"   Max steps: {env_config.max_steps}")
    except ValidationError as e:
        print(f"   ❌ Valid environment config failed: {e}")
        return False
    
    # Test 5: Valid agent config
    print("\n5. Testing valid agent config...")
    try:
        from farm.core.config_hydra_models import AgentParameters
        agent_config = HydraAgentConfig(
            agent_parameters={
                "SystemAgent": AgentParameters(
                    share_weight=0.4,
                    attack_weight=0.1
                )
            }
        )
        print(f"   ✅ Valid agent config created")
        print(f"   SystemAgent share_weight: {agent_config.agent_parameters['SystemAgent'].share_weight}")
    except ValidationError as e:
        print(f"   ❌ Valid agent config failed: {e}")
        return False
    
    return True


def test_validation_functions():
    """Test validation functions."""
    print("\nTesting validation functions...")
    
    # Test 1: Valid config dict
    print("\n1. Testing validate_config_dict with valid data...")
    valid_config = {
        "width": 100,
        "height": 100,
        "max_steps": 1000,
        "system_agents": 10,
        "independent_agents": 10,
        "control_agents": 10,
        "simulation_id": "test"
    }
    
    try:
        validated = validate_config_dict(valid_config)
        print(f"   ✅ Valid config dict passed validation")
        print(f"   Environment: {validated.width}x{validated.height}")
    except ValidationError as e:
        print(f"   ❌ Valid config dict failed: {e}")
        return False
    
    # Test 2: Invalid config dict
    print("\n2. Testing validate_config_dict with invalid data...")
    invalid_config = {
        "width": -100,  # Invalid
        "height": 100,
        "max_steps": 1000
    }
    
    try:
        validated = validate_config_dict(invalid_config)
        print("   ❌ Invalid config dict should have failed")
        return False
    except ValidationError as e:
        print(f"   ✅ Invalid config dict correctly rejected: {len(e.errors())} errors")
        for error in e.errors():
            print(f"     - {error['loc']}: {error['msg']}")
    
    return True


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print("\nTesting edge cases...")
    
    # Test 1: Minimum valid values
    print("\n1. Testing minimum valid values...")
    try:
        config = HydraSimulationConfig(
            width=10,  # Minimum
            height=10,  # Minimum
            max_steps=1,  # Minimum
            system_agents=0,  # Minimum
            independent_agents=0,  # Minimum
            control_agents=0  # Minimum
        )
        print(f"   ✅ Minimum values accepted")
        print(f"   Environment: {config.width}x{config.height}")
    except ValidationError as e:
        print(f"   ❌ Minimum values rejected: {e}")
        return False
    
    # Test 2: Maximum valid values (within population limits)
    print("\n2. Testing maximum valid values...")
    try:
        config = HydraSimulationConfig(
            width=10000,  # Maximum
            height=10000,  # Maximum
            max_steps=1000000,  # Maximum
            max_population=3000,  # Set high enough for agents
            system_agents=1000,  # Maximum
            independent_agents=1000,  # Maximum
            control_agents=1000  # Maximum
        )
        print(f"   ✅ Maximum values accepted")
        print(f"   Environment: {config.width}x{config.height}")
    except ValidationError as e:
        print(f"   ❌ Maximum values rejected: {e}")
        return False
    
    # Test 3: Boundary violations
    print("\n3. Testing boundary violations...")
    try:
        config = HydraSimulationConfig(
            width=10001,  # Exceeds maximum
            height=100,
            max_steps=1000
        )
        print("   ❌ Boundary violation should have failed")
        return False
    except ValidationError as e:
        print(f"   ✅ Boundary violation correctly rejected: {len(e.errors())} errors")
        for error in e.errors():
            print(f"     - {error['loc']}: {error['msg']}")
    
    return True


def main():
    """Run all Pydantic validation tests."""
    print("Pydantic Validation Test Suite")
    print("=" * 50)
    
    tests = [
        ("Pydantic Validation Integration", test_pydantic_validation),
        ("Pydantic Models Direct Testing", test_pydantic_models_directly),
        ("Validation Functions", test_validation_functions),
        ("Edge Cases and Boundaries", test_edge_cases),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} passed")
            else:
                failed += 1
                print(f"❌ {test_name} failed")
        except Exception as e:
            failed += 1
            print(f"❌ {test_name} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print("Pydantic Validation Test Results")
    print("=" * 50)
    print(f"Tests passed: {passed}")
    print(f"Tests failed: {failed}")
    print(f"Total tests: {passed + failed}")
    
    if failed == 0:
        print("\n🎉 All Pydantic validation tests passed!")
        print("The Pydantic validation system is working correctly.")
    else:
        print(f"\n⚠️  {failed} test(s) failed.")
        print("Please check the error messages above.")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())