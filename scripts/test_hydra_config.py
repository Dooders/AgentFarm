#!/usr/bin/env python3
"""
Test script to verify Hydra configuration loading.

This script tests the Hydra config loader and compatibility layer
to ensure configurations load correctly from the Hydra config system.
"""

import sys
from pathlib import Path

# Add workspace root to path
workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(workspace_root))

def test_hydra_loader_direct():
    """Test HydraConfigLoader directly."""
    print("=" * 60)
    print("Test 1: Direct HydraConfigLoader usage")
    print("=" * 60)
    
    try:
        from farm.config.hydra_loader import HydraConfigLoader
        
        loader = HydraConfigLoader()
        config = loader.load_config(environment="development")
        
        print(f"? Config loaded successfully")
        print(f"   - simulation_steps: {config.simulation_steps}")
        print(f"   - environment.width: {config.environment.width}")
        print(f"   - environment.height: {config.environment.height}")
        print(f"   - population.system_agents: {config.population.system_agents}")
        return True
    except Exception as e:
        print(f"? Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hydra_loader_with_profile():
    """Test HydraConfigLoader with profile."""
    print("\n" + "=" * 60)
    print("Test 2: HydraConfigLoader with profile")
    print("=" * 60)
    
    try:
        from farm.config.hydra_loader import HydraConfigLoader
        
        loader = HydraConfigLoader()
        config = loader.load_config(
            environment="production",
            profile="benchmark"
        )
        
        print(f"? Config loaded successfully with profile")
        print(f"   - simulation_steps: {config.simulation_steps}")
        print(f"   - environment.width: {config.environment.width}")
        print(f"   - population.system_agents: {config.population.system_agents}")
        print(f"   - debug: {config.logging.debug}")
        return True
    except Exception as e:
        print(f"? Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hydra_loader_with_overrides():
    """Test HydraConfigLoader with overrides."""
    print("\n" + "=" * 60)
    print("Test 3: HydraConfigLoader with overrides")
    print("=" * 60)
    
    try:
        from farm.config.hydra_loader import HydraConfigLoader
        
        loader = HydraConfigLoader()
        config = loader.load_config(
            environment="development",
            overrides=["simulation_steps=200", "population.system_agents=50"]
        )
        
        print(f"? Config loaded successfully with overrides")
        print(f"   - simulation_steps: {config.simulation_steps} (should be 200)")
        print(f"   - population.system_agents: {config.population.system_agents} (should be 50)")
        
        # Verify overrides worked
        assert config.simulation_steps == 200, f"Expected 200, got {config.simulation_steps}"
        assert config.population.system_agents == 50, f"Expected 50, got {config.population.system_agents}"
        print(f"   ? Overrides verified")
        return True
    except Exception as e:
        print(f"? Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_from_hydra_method():
    """Test SimulationConfig.from_hydra() method."""
    print("\n" + "=" * 60)
    print("Test 4: SimulationConfig.from_hydra() method")
    print("=" * 60)
    
    try:
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from farm.config import SimulationConfig
        
        # Initialize Hydra
        GlobalHydra.instance().clear()
        config_dir = workspace_root / "conf"
        initialize_config_dir(
            config_dir=str(config_dir.absolute()),
            config_name="config",
            version_base=None,
        )
        
        # Compose config
        cfg = compose(config_name="config", overrides=["environment=development"])
        
        # Convert using from_hydra
        config = SimulationConfig.from_hydra(cfg)
        
        print(f"? Config converted successfully")
        print(f"   - simulation_steps: {config.simulation_steps}")
        print(f"   - environment.width: {config.environment.width}")
        return True
    except Exception as e:
        print(f"? Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_compatibility_layer():
    """Test compatibility layer (load_config with use_hydra=True)."""
    print("\n" + "=" * 60)
    print("Test 5: Compatibility layer (use_hydra=True)")
    print("=" * 60)
    
    try:
        from farm.config import load_config
        
        config = load_config(
            environment="development",
            profile=None,
            use_hydra=True
        )
        
        print(f"? Config loaded via compatibility layer")
        print(f"   - simulation_steps: {config.simulation_steps}")
        print(f"   - environment.width: {config.environment.width}")
        return True
    except Exception as e:
        print(f"? Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_environment_configs():
    """Test all environment configs."""
    print("\n" + "=" * 60)
    print("Test 6: All environment configs")
    print("=" * 60)
    
    environments = ["development", "production", "testing"]
    results = []
    
    try:
        from farm.config.hydra_loader import HydraConfigLoader
        
        loader = HydraConfigLoader()
        
        for env in environments:
            config = loader.load_config(environment=env)
            print(f"? {env}: width={config.environment.width}, steps={config.simulation_steps}")
            results.append(True)
        
        return all(results)
    except Exception as e:
        print(f"? Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_profile_configs():
    """Test all profile configs."""
    print("\n" + "=" * 60)
    print("Test 7: All profile configs")
    print("=" * 60)
    
    profiles = ["benchmark", "simulation", "research"]
    results = []
    
    try:
        from farm.config.hydra_loader import HydraConfigLoader
        
        loader = HydraConfigLoader()
        
        for profile in profiles:
            config = loader.load_config(environment="development", profile=profile)
            print(f"? {profile}: steps={config.simulation_steps}, width={config.environment.width}")
            results.append(True)
        
        return all(results)
    except Exception as e:
        print(f"? Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Hydra Configuration Loading Tests")
    print("=" * 60)
    print()
    
    # Check if Hydra is installed
    try:
        import hydra
        print(f"? Hydra is installed (version: {hydra.__version__})")
    except ImportError:
        print("? Hydra is not installed. Install with: pip install hydra-core>=1.3.0")
        return 1
    
    # Check if config directory exists
    config_dir = workspace_root / "conf"
    if not config_dir.exists():
        print(f"? Config directory not found: {config_dir}")
        print("   Make sure Phase 1 of Hydra migration is complete.")
        return 1
    
    print(f"? Config directory found: {config_dir}")
    print()
    
    # Run tests
    tests = [
        test_hydra_loader_direct,
        test_hydra_loader_with_profile,
        test_hydra_loader_with_overrides,
        test_from_hydra_method,
        test_compatibility_layer,
        test_environment_configs,
        test_profile_configs,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"? Test {test.__name__} raised exception: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("? All tests passed!")
        return 0
    else:
        print(f"? {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
