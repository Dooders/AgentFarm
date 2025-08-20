#!/usr/bin/env python3
"""
AgentFarm MCP Server Launcher

This script properly configures the environment and starts the MCP server
for the AgentFarm simulation system.
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("mcp-launcher")

def setup_environment():
    """Setup the environment for running the MCP server."""
    
    # Get the current directory (where this script is located)
    current_dir = Path(__file__).parent.absolute()
    
    # Add the current directory to Python path so 'farm' module can be imported
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
        logger.info(f"Added {current_dir} to Python path")
    
    # Verify required directories exist
    required_dirs = ["simulations", "experiments", "research", "analysis", "exports"]
    for dir_name in required_dirs:
        dir_path = current_dir / dir_name
        dir_path.mkdir(exist_ok=True)
        logger.info(f"Ensured directory exists: {dir_path}")
    
    # Check if farm module can be imported
    try:
        import farm
        logger.info("Successfully imported farm module")
    except ImportError as e:
        logger.error(f"Failed to import farm module: {e}")
        logger.error("Make sure you're running this from the AgentFarm root directory")
        sys.exit(1)
    
    return current_dir

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        "mcp",
        "pandas", 
        "numpy",
        "sqlalchemy",
        "torch",
        "pyyaml"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✓ {package}")
        except ImportError:
            missing_packages.append(package)
            logger.warning(f"✗ {package} (missing)")
    
    if missing_packages:
        logger.error(f"Missing required packages: {missing_packages}")
        logger.error("Install them with: pip install -r mcp_requirements.txt")
        sys.exit(1)
    
    logger.info("All dependencies satisfied")

def main():
    """Main launcher function."""
    logger.info("AgentFarm MCP Server Launcher")
    logger.info("=" * 40)
    
    # Setup environment
    current_dir = setup_environment()
    
    # Check dependencies
    check_dependencies()
    
    # Import and run the MCP server
    try:
        logger.info("Starting MCP server...")
        
        # Import the server module
        import mcp_simulation_server
        
        # Run the server
        import asyncio
        asyncio.run(mcp_simulation_server.main())
        
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()