#!/usr/bin/env python3

import sys
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

print("Testing Alien Invasion Implementation...")
print("=" * 50)

# Test 1: Configuration
try:
    from farm.core.config import SimulationConfig
    config = SimulationConfig.from_yaml("config_alien_invasion.yaml")
    print("✓ Configuration loaded successfully")
    print(f"  Aliens: {config.alien_agents}, Humans: {config.human_agents}")
except Exception as e:
    print(f"✗ Configuration failed: {e}")
    sys.exit(1)

# Test 2: Agent imports
try:
    from farm.agents.alien_agent import AlienAgent
    from farm.agents.human_agent import HumanAgent
    print("✓ Agent classes imported successfully")
except Exception as e:
    print(f"✗ Agent import failed: {e}")
    sys.exit(1)

# Test 3: Environment import
try:
    from farm.environments.alien_invasion_environment import create_alien_invasion_simulation
    print("✓ Environment imported successfully")
except Exception as e:
    print(f"✗ Environment import failed: {e}")
    sys.exit(1)

print("\n🎉 All basic imports successful!")
print("Alien invasion simulation is ready to run.")
print("\nTo run the full simulation:")
print("  python run_alien_invasion.py")
print("\nTo run with analysis:")
print("  python run_alien_invasion.py --analyze")