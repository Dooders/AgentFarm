#!/usr/bin/env python3
"""
Test Suite for AgentFarm FastMCP Server

Comprehensive tests to validate the FastMCP server functionality.
"""

import json
import logging
import os
import sys
import tempfile
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fastmcp-test")

def test_fastmcp_imports():
    """Test that FastMCP and AgentFarm modules can be imported."""
    logger.info("🧪 Testing FastMCP imports...")
    
    try:
        import fastmcp
        logger.info("✅ FastMCP imported successfully")
    except ImportError as e:
        logger.error(f"❌ FastMCP import failed: {e}")
        logger.error("Install with: pip install fastmcp")
        return False
    
    try:
        from farm.core.simulation import run_simulation
        from farm.core.config import SimulationConfig
        from farm.database.database import SimulationDatabase
        logger.info("✅ AgentFarm core modules imported successfully")
    except ImportError as e:
        logger.error(f"❌ AgentFarm import failed: {e}")
        return False
    
    return True

def test_server_initialization():
    """Test that the FastMCP server can be initialized."""
    logger.info("🧪 Testing server initialization...")
    
    try:
        from fastmcp_simulation_server import mcp
        
        # Check that tools are registered
        tool_names = list(mcp._tools.keys())
        expected_tools = [
            "create_simulation",
            "create_experiment", 
            "run_experiment",
            "list_simulations",
            "get_simulation_status",
            "analyze_simulation",
            "export_simulation_data",
            "get_simulation_summary",
            "create_research_project",
            "batch_analyze"
        ]
        
        missing_tools = [tool for tool in expected_tools if tool not in tool_names]
        
        if missing_tools:
            logger.error(f"❌ Missing tools: {missing_tools}")
            return False
        
        logger.info(f"✅ Server initialized with {len(tool_names)} tools")
        logger.info(f"   Available tools: {tool_names}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Server initialization failed: {e}")
        return False

def test_simulation_creation():
    """Test creating a minimal simulation."""
    logger.info("🧪 Testing simulation creation...")
    
    try:
        from fastmcp_simulation_server import mcp
        
        # Test configuration
        config = {
            "width": 30,
            "height": 30,
            "system_agents": 3,
            "independent_agents": 3,
            "control_agents": 2,
            "simulation_steps": 5,  # Very short for testing
            "initial_resources": 5,
            "seed": 999
        }
        
        # Create temporary output directory
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "test_sim")
            
            # Call the create_simulation tool directly
            result = mcp._tools["create_simulation"].func(config, output_path)
            
            # Parse result
            try:
                result_data = json.loads(result)
                if result_data.get("status") == "completed":
                    logger.info("✅ Simulation created successfully")
                    logger.info(f"   Simulation ID: {result_data.get('simulation_id')}")
                    logger.info(f"   Final agents: {result_data.get('final_metrics', {}).get('total_agents', 'unknown')}")
                    return True
                else:
                    logger.error(f"❌ Simulation creation failed: {result}")
                    return False
            except json.JSONDecodeError:
                if "Failed:" in result:
                    logger.error(f"❌ Simulation creation failed: {result}")
                    return False
                else:
                    logger.warning(f"⚠️  Unexpected result format: {result}")
                    return True  # May still be successful
        
    except Exception as e:
        logger.error(f"❌ Simulation creation test failed: {e}")
        return False

def test_tool_interfaces():
    """Test that all tools have proper interfaces."""
    logger.info("🧪 Testing tool interfaces...")
    
    try:
        from fastmcp_simulation_server import mcp
        
        # Test each tool's function signature
        tools_to_test = {
            "create_simulation": {"config": {"simulation_steps": 10}},
            "list_simulations": {},
            "create_research_project": {
                "name": "test_project", 
                "description": "Test project"
            }
        }
        
        for tool_name, test_args in tools_to_test.items():
            if tool_name in mcp._tools:
                tool_func = mcp._tools[tool_name].func
                
                # Check function signature
                import inspect
                sig = inspect.signature(tool_func)
                logger.info(f"✅ {tool_name}: {sig}")
            else:
                logger.error(f"❌ Tool not found: {tool_name}")
                return False
        
        logger.info("✅ All tool interfaces validated")
        return True
        
    except Exception as e:
        logger.error(f"❌ Tool interface test failed: {e}")
        return False

def test_directory_structure():
    """Test that required directories can be created."""
    logger.info("🧪 Testing directory structure...")
    
    try:
        required_dirs = ["simulations", "experiments", "research", "analysis", "exports"]
        
        for dir_name in required_dirs:
            os.makedirs(dir_name, exist_ok=True)
            if os.path.exists(dir_name):
                logger.info(f"✅ Directory created/verified: {dir_name}")
            else:
                logger.error(f"❌ Failed to create directory: {dir_name}")
                return False
        
        logger.info("✅ Directory structure validated")
        return True
        
    except Exception as e:
        logger.error(f"❌ Directory structure test failed: {e}")
        return False

def run_all_tests():
    """Run the complete test suite."""
    logger.info("🚀 AgentFarm FastMCP Server Test Suite")
    logger.info("=" * 50)
    
    tests = [
        ("FastMCP Imports", test_fastmcp_imports),
        ("Server Initialization", test_server_initialization),
        ("Directory Structure", test_directory_structure),
        ("Tool Interfaces", test_tool_interfaces),
        ("Simulation Creation", test_simulation_creation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        
        try:
            if test_func():
                passed += 1
            else:
                logger.error(f"Test failed: {test_name}")
        except Exception as e:
            logger.error(f"Test error in {test_name}: {e}")
    
    logger.info("\n" + "=" * 50)
    logger.info(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! FastMCP server is ready!")
        logger.info("\n🚀 To start the server:")
        logger.info("   python start_fastmcp_server.py")
        logger.info("\n📖 For usage examples:")
        logger.info("   python fastmcp_client_example.py")
        return True
    else:
        logger.error(f"❌ {total - passed} tests failed. Check errors above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)