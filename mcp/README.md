# AgentFarm Production MCP Server

A production-ready **FastMCP** server providing LLM agents with comprehensive access to the AgentFarm simulation system.

## 🏗 **Production Architecture**

```
mcp/
├── __init__.py           # Package initialization
├── main.py              # Main server entry point
├── server.py            # Core server implementation
├── config.py            # Configuration management
├── utils.py             # Utility functions
├── launcher.py          # Production launcher with monitoring
├── tests.py             # Comprehensive test suite
├── client_example.py    # Production client examples
├── requirements.txt     # Dependencies
└── README.md           # This documentation
```

## 🚀 **Quick Start**

### 1. Install Dependencies
```bash
pip install -r mcp/requirements.txt
```

### 2. Run Tests
```bash
python -m mcp.tests
```

### 3. Start Production Server
```bash
python -m mcp.main
```

Or with custom options:
```bash
python -m mcp.main --log-level DEBUG --base-dir /data/simulations
```

### 4. Health Check
```bash
python -m mcp.launcher health-check
```

## 🛠 **Available Tools**

| Tool | Purpose | Key Parameters |
|------|---------|---------------|
| `create_simulation` | Run single simulation | `config`, `output_path` |
| `create_experiment` | Setup multi-iteration experiment | `name`, `base_config`, `variations` |
| `run_experiment` | Execute experiment | `experiment_id`, `run_analysis` |
| `list_simulations` | List available simulations | `search_path` |
| `get_simulation_status` | Get simulation metrics | `simulation_id` |
| `get_simulation_summary` | Detailed simulation summary | `simulation_path` |
| `analyze_simulation` | Run analysis modules | `simulation_path`, `analysis_types` |
| `export_simulation_data` | Export data | `simulation_path`, `format_type` |
| `batch_analyze` | Analyze multiple simulations | `experiment_path`, `analysis_modules` |
| `create_research_project` | Setup research structure | `name`, `description` |

## 🎯 **Production Features**

### **Robust Configuration Management**
- Environment variable overrides
- Configurable directories and defaults
- Validation and error checking
- Production logging setup

### **Comprehensive Error Handling**
- Standardized error responses
- Detailed logging with context
- Graceful degradation on failures
- Input validation and sanitization

### **Performance Optimization**
- Concurrent simulation limits
- Memory usage monitoring
- Temporary file cleanup
- Efficient data structures

### **Production Monitoring**
- Health checks and diagnostics
- System resource monitoring
- Runtime statistics
- Structured logging

## 🔧 **Configuration Options**

### **Environment Variables**
```bash
export MCP_LOG_LEVEL=DEBUG
export MCP_BASE_DIR=/data/agentfarm
export MCP_MAX_CONCURRENT=10
```

### **Command Line Options**
```bash
python -m mcp.main \
  --log-level INFO \
  --base-dir /data/agentfarm \
  --max-concurrent 5 \
  --transport http \
  --port 8000
```

### **Programmatic Configuration**
```python
from mcp.config import MCPServerConfig
from mcp.server import AgentFarmMCPServer

config = MCPServerConfig()
config.base_dir = Path("/data/agentfarm")
config.max_concurrent_simulations = 10

server = AgentFarmMCPServer(config)
server.run()
```

## 📊 **Usage Examples**

### **Basic Simulation**
```python
# Create simulation
result = create_simulation({
    "config": {
        "width": 100,
        "height": 100,
        "system_agents": 15,
        "independent_agents": 10,
        "control_agents": 5,
        "simulation_steps": 1000,
        "seed": 42
    }
})

# Get results
summary = get_simulation_summary({
    "simulation_path": "/path/to/simulation.db"
})
```

### **Research Experiment**
```python
# Create research project
project = create_research_project({
    "name": "cooperation_study",
    "description": "Investigating cooperative behavior emergence"
})

# Design experiment
experiment = create_experiment({
    "name": "agent_composition_effects",
    "base_config": {"simulation_steps": 2000},
    "variations": [
        {"system_agents": 20, "independent_agents": 5},
        {"system_agents": 10, "independent_agents": 15},
        {"system_agents": 5, "independent_agents": 20}
    ]
})

# Run and analyze
run_result = run_experiment({"experiment_id": exp_id})
analysis = batch_analyze({"experiment_path": experiment_path})
```

### **Data Export and Analysis**
```python
# Export comprehensive data
export_simulation_data({
    "simulation_path": simulation_db,
    "output_path": "research_data/study_001",
    "format_type": "csv",
    "data_types": ["all"]
})

# Run comprehensive analysis
analyze_simulation({
    "simulation_path": experiment_directory,
    "analysis_types": ["dominance", "advantage"],
    "output_path": "analysis/study_001"
})
```

## 🔍 **Monitoring and Diagnostics**

### **Server Health Check**
```bash
python -m mcp.launcher health-check
```

### **System Information**
```python
from mcp.utils import get_system_info
info = get_system_info()
print(f"Memory: {info['memory_available'] / (1024**3):.1f}GB")
print(f"Disk: {info['disk_usage'] / (1024**3):.1f}GB")
```

### **Server Statistics**
```python
server_info = server.get_server_info()
print(f"Active simulations: {server_info['active_simulations']}")
print(f"Active experiments: {server_info['active_experiments']}")
```

## 🧪 **Testing**

### **Run Full Test Suite**
```bash
python -m mcp.tests
```

### **Unit Tests Only**
```bash
python -m unittest mcp.tests.TestMCPServerConfig
```

### **Integration Tests**
```bash
python -m mcp.tests run_integration_tests
```

## 🚀 **Deployment Options**

### **Stdio Transport (Default)**
```bash
python -m mcp.main
```
Perfect for direct LLM integration.

### **HTTP Transport**
```bash
python -m mcp.main --transport http --host 0.0.0.0 --port 8000
```
For networked access and web-based clients.

### **Docker Deployment**
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY mcp/ ./mcp/
COPY farm/ ./farm/
COPY requirements.txt ./

RUN pip install -r requirements.txt
RUN pip install -r mcp/requirements.txt

EXPOSE 8000
CMD ["python", "-m", "mcp.main", "--transport", "http", "--host", "0.0.0.0"]
```

### **Systemd Service**
```ini
[Unit]
Description=AgentFarm MCP Server
After=network.target

[Service]
Type=simple
User=agentfarm
WorkingDirectory=/opt/agentfarm
ExecStart=/opt/agentfarm/venv/bin/python -m mcp.main
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

## 🔐 **Security Considerations**

### **File System Access**
- Server validates all file paths
- Prevents directory traversal attacks
- Restricts operations to configured base directory

### **Resource Limits**
- Configurable memory limits
- Maximum concurrent simulation limits
- Automatic cleanup of temporary files

### **Input Validation**
- All tool inputs are validated
- Configuration parameters are sanitized
- Error messages don't leak sensitive information

## 📈 **Performance Tuning**

### **Memory Management**
```python
config.memory_limit_mb = 4096  # 4GB limit
config.cleanup_temp_files = True
config.max_concurrent_simulations = 3
```

### **Database Optimization**
- Uses WAL mode for SQLite
- Optimized pragma settings
- Efficient bulk operations

### **Analysis Performance**
- Parallel analysis module execution
- Caching of intermediate results
- Streaming for large datasets

## 🐛 **Troubleshooting**

### **Common Issues**

1. **Import Errors**
   ```bash
   # Ensure proper Python path
   export PYTHONPATH="/path/to/AgentFarm:$PYTHONPATH"
   ```

2. **Memory Issues**
   ```bash
   # Reduce simulation size
   python -m mcp.main --memory-limit 2048
   ```

3. **Permission Errors**
   ```bash
   # Check directory permissions
   chmod 755 /data/agentfarm
   ```

### **Debug Mode**
```bash
python -m mcp.main --dev --log-level DEBUG
```

### **Logs Location**
- Default: `mcp_server.log` in base directory
- Custom: Set via configuration

## 🔄 **Integration Patterns**

### **Claude Desktop**
```json
{
  "mcpServers": {
    "agentfarm": {
      "command": "python",
      "args": ["-m", "mcp.main"],
      "cwd": "/path/to/AgentFarm"
    }
  }
}
```

### **Custom LLM Client**
```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

server_params = StdioServerParameters(
    command="python",
    args=["-m", "mcp.main"],
    cwd="/path/to/AgentFarm"
)

async with stdio_client(server_params) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()
        
        # Use tools
        result = await session.call_tool("create_simulation", {
            "config": {"simulation_steps": 500}
        })
```

## 📝 **API Documentation**

### **Response Format**
All tools return JSON strings with consistent structure:

```json
{
  "success": true,
  "tool": "create_simulation",
  "data": {
    "simulation_id": "sim_20241215_143022_abc123",
    "status": "completed"
  },
  "timestamp": "2024-12-15T14:30:22"
}
```

### **Error Format**
```json
{
  "error": true,
  "tool": "create_simulation", 
  "message": "Configuration validation failed",
  "timestamp": "2024-12-15T14:30:22"
}
```

## 🎓 **Best Practices**

1. **Always validate results** - Check for error responses
2. **Use meaningful names** - For experiments and research projects  
3. **Set seeds** - For reproducible results
4. **Monitor resources** - Check system capacity before large experiments
5. **Organize output** - Use structured directory hierarchies
6. **Test configurations** - Start with small simulations

This production MCP server provides enterprise-grade reliability and features for LLM-driven simulation research! 🎯