"""
AgentFarm MCP Server Package

Production-ready FastMCP server for AgentFarm simulation system.
"""

__version__ = "1.0.0"
__author__ = "AgentFarm Team"

from .server import AgentFarmMCPServer
from .config import MCPServerConfig

__all__ = ["AgentFarmMCPServer", "MCPServerConfig"]