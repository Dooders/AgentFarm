#!/usr/bin/env python3
"""
Production FastMCP Server for AgentFarm Simulation System

Main entry point for the production MCP server.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path for farm imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from .server import AgentFarmMCPServer
from .config import MCPServerConfig

def setup_argument_parser() -> argparse.ArgumentParser:
    """Setup command line argument parser."""
    parser = argparse.ArgumentParser(
        description="AgentFarm FastMCP Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m mcp.main                          # Start with default settings
  python -m mcp.main --log-level DEBUG        # Enable debug logging
  python -m mcp.main --base-dir /path/data     # Use custom data directory
  python -m mcp.main --transport http --port 8000  # Run as HTTP server
        """
    )
    
    # Server configuration
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path.cwd(),
        help="Base directory for data storage (default: current directory)"
    )
    
    parser.add_argument(
        "--server-name",
        default="AgentFarm Simulation Server",
        help="Name of the MCP server"
    )
    
    # Transport options
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="Transport method (default: stdio)"
    )
    
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for HTTP transport (default: 127.0.0.1)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for HTTP transport (default: 8000)"
    )
    
    # Performance options
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=5,
        help="Maximum concurrent simulations (default: 5)"
    )
    
    parser.add_argument(
        "--memory-limit",
        type=int,
        help="Memory limit in MB (optional)"
    )
    
    # Development options
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Enable development mode with enhanced logging"
    )
    
    return parser

def create_server_config(args: argparse.Namespace) -> MCPServerConfig:
    """Create server configuration from command line arguments."""
    config = MCPServerConfig()
    
    # Apply command line overrides
    config.server_name = args.server_name
    config.log_level = args.log_level
    config.base_dir = args.base_dir
    config.max_concurrent_simulations = args.max_concurrent
    
    if args.memory_limit:
        config.memory_limit_mb = args.memory_limit
    
    # Development mode adjustments
    if args.dev:
        config.log_level = "DEBUG"
        config.cleanup_temp_files = False
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
    
    return config

def validate_environment() -> bool:
    """Validate that the environment is ready for the server."""
    logger = logging.getLogger(__name__)
    
    # Check Python version
    if sys.version_info < (3, 8):
        logger.error("Python 3.8 or higher is required")
        return False
    
    # Check required packages
    required_packages = [
        "fastmcp",
        "pandas", 
        "numpy",
        "sqlalchemy",
        "torch"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {missing_packages}")
        logger.error("Install with: pip install -r mcp/requirements.txt")
        return False
    
    # Check farm package
    try:
        import farm
        logger.info("✓ AgentFarm package available")
    except ImportError:
        logger.error("AgentFarm package not found. Ensure you're in the correct directory.")
        return False
    
    return True

def main():
    """Main entry point for the MCP server."""
    # Parse command line arguments
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Create configuration
    config = create_server_config(args)
    
    # Setup initial logging
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    # Validate environment
    if not validate_environment():
        sys.exit(1)
    
    # Create and start server
    try:
        logger.info("Initializing AgentFarm MCP Server...")
        server = AgentFarmMCPServer(config)
        
        # Print server information
        server_info = server.get_server_info()
        logger.info(f"Server: {server_info['server_name']}")
        logger.info(f"Version: {server_info['version']}")
        logger.info(f"Base directory: {config.base_dir}")
        logger.info(f"Available tools: {server_info['tools_available']}")
        
        # Prepare transport options
        transport_kwargs = {}
        if args.transport == "http":
            transport_kwargs = {
                "transport": "http",
                "host": args.host,
                "port": args.port
            }
            logger.info(f"Starting HTTP server on {args.host}:{args.port}")
        else:
            logger.info("Starting stdio server for MCP clients")
        
        # Start the server
        server.run(**transport_kwargs)
        
    except KeyboardInterrupt:
        logger.info("Server shutdown requested by user")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("Server shutdown complete")

if __name__ == "__main__":
    main()