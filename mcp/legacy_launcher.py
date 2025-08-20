#!/usr/bin/env python3
"""
FastMCP Server Launcher for AgentFarm

Simple launcher that sets up the environment and starts the FastMCP server.
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fastmcp-launcher")

def setup_environment():
    """Setup environment for the FastMCP server."""
    current_dir = Path(__file__).parent.absolute()
    
    # Add to Python path
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
        logger.info(f"Added {current_dir} to Python path")
    
    # Create required directories
    dirs = ["simulations", "experiments", "research", "analysis", "exports"]
    for dir_name in dirs:
        (current_dir / dir_name).mkdir(exist_ok=True)
    
    # Test imports
    try:
        import fastmcp
        import farm
        logger.info("✓ All required modules available")
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Install with: pip install -r fastmcp_requirements.txt")
        sys.exit(1)

def main():
    """Launch the FastMCP server."""
    logger.info("AgentFarm FastMCP Server Launcher")
    logger.info("=" * 40)
    
    setup_environment()
    
    try:
        # Import and start the server
        from fastmcp_simulation_server import mcp
        
        logger.info("Starting FastMCP server for AgentFarm...")
        logger.info("Server ready for LLM agent connections!")
        
        # Run the server
        mcp.run()
        
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()