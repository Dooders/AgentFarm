#!/usr/bin/env python3
"""
Production launcher for AgentFarm MCP Server.

This script provides a robust production environment for running the MCP server
with proper error handling, logging, and monitoring.
"""

import logging
import os
import signal
import sys
import time
from pathlib import Path

# Add parent directory for farm imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from .main import main as mcp_main
from .config import MCPServerConfig
from .utils import get_system_info

# Configure logging for launcher
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mcp-launcher")

class MCPServerLauncher:
    """Production launcher for the MCP server with monitoring and recovery."""
    
    def __init__(self):
        self.server_process = None
        self.shutdown_requested = False
        self.start_time = None
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True
    
    def pre_flight_check(self) -> bool:
        """Perform comprehensive pre-flight checks."""
        logger.info("🚀 AgentFarm MCP Server - Production Launcher")
        logger.info("=" * 60)
        
        # System information
        logger.info("📊 System Information:")
        sys_info = get_system_info()
        logger.info(f"   Platform: {sys_info['platform']}")
        logger.info(f"   Python: {sys_info['python_version']}")
        logger.info(f"   CPU cores: {sys_info['cpu_count']}")
        logger.info(f"   Memory: {sys_info['memory_available'] / (1024**3):.1f}GB available")
        logger.info(f"   Disk: {sys_info['disk_usage'] / (1024**3):.1f}GB free")
        
        # Check Python version
        if sys.version_info < (3, 8):
            logger.error("❌ Python 3.8 or higher required")
            return False
        
        logger.info("✅ Python version compatible")
        
        # Check dependencies
        logger.info("📦 Checking dependencies...")
        required_packages = [
            "fastmcp",
            "pandas",
            "numpy", 
            "sqlalchemy",
            "torch",
            "farm"
        ]
        
        missing = []
        for package in required_packages:
            try:
                __import__(package)
                logger.info(f"   ✅ {package}")
            except ImportError:
                logger.error(f"   ❌ {package}")
                missing.append(package)
        
        if missing:
            logger.error(f"Missing packages: {missing}")
            logger.error("Install with: pip install -r mcp/requirements.txt")
            return False
        
        # Check disk space (minimum 1GB free)
        if sys_info['disk_usage'] < 1024**3:
            logger.warning("⚠️  Low disk space (< 1GB free)")
        
        # Check memory (minimum 2GB available)
        if sys_info['memory_available'] < 2 * 1024**3:
            logger.warning("⚠️  Low memory (< 2GB available)")
        
        logger.info("✅ Pre-flight checks completed")
        return True
    
    def start_server(self, args: list = None) -> None:
        """Start the MCP server with monitoring."""
        if not self.pre_flight_check():
            logger.error("❌ Pre-flight checks failed")
            sys.exit(1)
        
        logger.info("🎯 Starting AgentFarm MCP Server...")
        self.start_time = time.time()
        
        try:
            # Override sys.argv if args provided
            if args:
                original_argv = sys.argv
                sys.argv = ["mcp_server"] + args
            else:
                original_argv = None
            
            # Start the main server
            mcp_main()
            
        except KeyboardInterrupt:
            logger.info("🛑 Shutdown requested by user")
        except Exception as e:
            logger.error(f"💥 Server error: {e}", exc_info=True)
            raise
        finally:
            # Restore original argv
            if original_argv:
                sys.argv = original_argv
            
            # Log runtime statistics
            if self.start_time:
                runtime = time.time() - self.start_time
                logger.info(f"📊 Server runtime: {runtime:.1f} seconds")
            
            logger.info("🏁 Server shutdown complete")
    
    def health_check(self) -> bool:
        """Perform server health check."""
        try:
            # Check if required directories exist
            config = MCPServerConfig()
            
            required_dirs = ["simulations", "experiments", "research", "analysis", "exports"]
            for dir_name in required_dirs:
                dir_path = config.get_path(dir_name.rstrip('s'))  # Remove 's' for config method
                if not dir_path.exists():
                    logger.error(f"Required directory missing: {dir_path}")
                    return False
            
            # Check system resources
            sys_info = get_system_info()
            
            # Memory check (warn if < 1GB available)
            if sys_info['memory_available'] < 1024**3:
                logger.warning("Low memory available")
                return False
            
            # Disk space check (warn if < 500MB free)
            if sys_info['disk_usage'] < 500 * 1024**2:
                logger.warning("Low disk space available")
                return False
            
            logger.info("✅ Health check passed")
            return True
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

def main():
    """Main launcher entry point."""
    launcher = MCPServerLauncher()
    
    # Handle command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "health-check":
        # Run health check only
        if launcher.health_check():
            print("✅ Server health check passed")
            sys.exit(0)
        else:
            print("❌ Server health check failed")
            sys.exit(1)
    
    # Start the server
    try:
        launcher.start_server()
    except Exception as e:
        logger.error(f"Launcher failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()