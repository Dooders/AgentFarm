"""
Configuration management for AgentFarm MCP Server.
"""

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

@dataclass
class MCPServerConfig:
    """Configuration for the AgentFarm MCP Server."""
    
    # Server settings
    server_name: str = "AgentFarm Simulation Server"
    log_level: str = "INFO"
    
    # Directory settings
    base_dir: Path = field(default_factory=lambda: Path.cwd())
    simulations_dir: str = "simulations"
    experiments_dir: str = "experiments"
    research_dir: str = "research"
    analysis_dir: str = "analysis"
    exports_dir: str = "exports"
    temp_dir: str = "temp"
    
    # Simulation defaults
    default_simulation_steps: int = 1000
    default_environment_size: int = 100
    default_agents_per_type: int = 10
    default_initial_resources: int = 20
    
    # Analysis settings
    default_analysis_types: List[str] = field(
        default_factory=lambda: ["dominance", "advantage"]
    )
    save_analysis_to_db: bool = True
    
    # Export settings
    default_export_format: str = "csv"
    default_export_types: List[str] = field(
        default_factory=lambda: ["all"]
    )
    
    # Performance settings
    max_concurrent_simulations: int = 5
    memory_limit_mb: Optional[int] = None
    cleanup_temp_files: bool = True
    
    def __post_init__(self):
        """Initialize configuration after creation."""
        # Convert string paths to Path objects
        if isinstance(self.base_dir, str):
            self.base_dir = Path(self.base_dir)
        
        # Create required directories
        self.ensure_directories()
        
        # Setup logging
        self.setup_logging()
    
    def ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        directories = [
            self.simulations_dir,
            self.experiments_dir, 
            self.research_dir,
            self.analysis_dir,
            self.exports_dir,
            self.temp_dir
        ]
        
        for directory in directories:
            dir_path = self.base_dir / directory
            dir_path.mkdir(exist_ok=True, parents=True)
    
    def setup_logging(self) -> None:
        """Configure logging for the server."""
        log_level = getattr(logging, self.log_level.upper(), logging.INFO)
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.base_dir / "mcp_server.log")
            ]
        )
    
    def get_path(self, directory_type: str) -> Path:
        """Get the full path for a directory type."""
        dir_map = {
            "simulations": self.simulations_dir,
            "experiments": self.experiments_dir,
            "research": self.research_dir,
            "analysis": self.analysis_dir,
            "exports": self.exports_dir,
            "temp": self.temp_dir
        }
        
        if directory_type not in dir_map:
            raise ValueError(f"Unknown directory type: {directory_type}")
        
        return self.base_dir / dir_map[directory_type]
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return {
            "server_name": self.server_name,
            "log_level": self.log_level,
            "base_dir": str(self.base_dir),
            "directories": {
                "simulations": self.simulations_dir,
                "experiments": self.experiments_dir,
                "research": self.research_dir,
                "analysis": self.analysis_dir,
                "exports": self.exports_dir,
                "temp": self.temp_dir
            },
            "defaults": {
                "simulation_steps": self.default_simulation_steps,
                "environment_size": self.default_environment_size,
                "agents_per_type": self.default_agents_per_type,
                "initial_resources": self.default_initial_resources
            }
        }
    
    @classmethod
    def from_env(cls) -> "MCPServerConfig":
        """Create configuration from environment variables."""
        config = cls()
        
        # Override with environment variables if present
        if "MCP_LOG_LEVEL" in os.environ:
            config.log_level = os.environ["MCP_LOG_LEVEL"]
        
        if "MCP_BASE_DIR" in os.environ:
            config.base_dir = Path(os.environ["MCP_BASE_DIR"])
        
        if "MCP_MAX_CONCURRENT" in os.environ:
            config.max_concurrent_simulations = int(os.environ["MCP_MAX_CONCURRENT"])
        
        return config