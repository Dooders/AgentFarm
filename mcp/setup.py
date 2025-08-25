#!/usr/bin/env python3
"""
Setup script for AgentFarm MCP Server

Handles installation, configuration, and deployment of the production MCP server.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp-setup")

def install_dependencies():
    """Install required dependencies for the MCP server."""
    logger.info("📦 Installing MCP server dependencies...")
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
        logger.info("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install dependencies: {e}")
        return False

def verify_agentfarm():
    """Verify that AgentFarm is properly available."""
    logger.info("🔍 Verifying AgentFarm availability...")
    
    try:
        # Add parent directory to path
        parent_dir = Path(__file__).parent.parent
        sys.path.insert(0, str(parent_dir))
        
        import farm
        from farm.core.simulation import run_simulation
        from farm.core.config import SimulationConfig
        
        logger.info("✅ AgentFarm modules available")
        return True
    except ImportError as e:
        logger.error(f"❌ AgentFarm not available: {e}")
        logger.error("Ensure you're running from the AgentFarm root directory")
        return False

def create_production_config():
    """Create production configuration file."""
    logger.info("⚙️  Creating production configuration...")
    
    config_content = """# AgentFarm MCP Server Production Configuration

# Server settings
MCP_LOG_LEVEL=INFO
MCP_BASE_DIR=./data
MCP_MAX_CONCURRENT=5

# Performance settings  
MCP_MEMORY_LIMIT_MB=4096
MCP_CLEANUP_TEMP=true

# Security settings
MCP_VALIDATE_PATHS=true
MCP_RESTRICT_BASE_DIR=true
"""
    
    config_file = Path(__file__).parent / "production.env"
    
    try:
        with open(config_file, "w") as f:
            f.write(config_content)
        
        logger.info(f"✅ Production config created: {config_file}")
        logger.info("   Load with: source mcp/production.env")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to create config: {e}")
        return False

def run_health_check():
    """Run comprehensive health check."""
    logger.info("🏥 Running health check...")
    
    try:
        from .launcher import MCPServerLauncher
        
        launcher = MCPServerLauncher()
        if launcher.health_check():
            logger.info("✅ Health check passed")
            return True
        else:
            logger.error("❌ Health check failed")
            return False
    except Exception as e:
        logger.error(f"❌ Health check error: {e}")
        return False

def create_startup_script():
    """Create production startup script."""
    logger.info("🚀 Creating startup script...")
    
    startup_script = """#!/bin/bash
# AgentFarm MCP Server Production Startup Script

set -e

# Load configuration
if [ -f "mcp/production.env" ]; then
    source mcp/production.env
    echo "✅ Loaded production configuration"
fi

# Set Python path
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Start server
echo "🚀 Starting AgentFarm MCP Server..."
python -m mcp.main "$@"
"""
    
    script_file = Path(__file__).parent / "start_production.sh"
    
    try:
        with open(script_file, "w") as f:
            f.write(startup_script)
        
        # Make executable
        os.chmod(script_file, 0o755)
        
        logger.info(f"✅ Startup script created: {script_file}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to create startup script: {e}")
        return False

def main():
    """Main setup function."""
    logger.info("🔧 AgentFarm MCP Server Setup")
    logger.info("=" * 40)
    
    setup_steps = [
        ("Install Dependencies", install_dependencies),
        ("Verify AgentFarm", verify_agentfarm),
        ("Create Production Config", create_production_config),
        ("Create Startup Script", create_startup_script),
        ("Run Health Check", run_health_check)
    ]
    
    failed_steps = []
    
    for step_name, step_func in setup_steps:
        logger.info(f"\n--- {step_name} ---")
        
        try:
            if not step_func():
                failed_steps.append(step_name)
        except Exception as e:
            logger.error(f"Error in {step_name}: {e}")
            failed_steps.append(step_name)
    
    # Summary
    logger.info("\n" + "=" * 40)
    if not failed_steps:
        logger.info("🎉 Setup completed successfully!")
        logger.info("\n🚀 To start the server:")
        logger.info("   ./mcp/start_production.sh")
        logger.info("\n📖 Or use Python module:")
        logger.info("   python -m mcp.main")
        logger.info("\n🔧 For development:")
        logger.info("   python -m mcp.main --dev")
        return True
    else:
        logger.error(f"❌ Setup failed. Failed steps: {failed_steps}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)