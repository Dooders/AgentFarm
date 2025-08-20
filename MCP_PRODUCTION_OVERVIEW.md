# 🚀 AgentFarm Production MCP Server

## ✅ **Clean Production Implementation**

All legacy code removed - only production-ready FastMCP implementation remains.

## 📁 **Production Directory Structure**

```
mcp/
├── __init__.py              # Package initialization  
├── server.py               # Core FastMCP server implementation
├── config.py               # Production configuration management
├── main.py                 # Main entry point with CLI options
├── utils.py                # Utility functions and helpers
├── launcher.py             # Production launcher with monitoring
├── tests.py                # Comprehensive test suite
├── setup.py                # Installation and setup script
├── demo.py                 # Production demo workflows
├── client_example.py       # Production client examples
├── requirements.txt        # Production dependencies
├── example_configs.yaml    # Example configurations
└── README.md               # Complete documentation
```

## 🎯 **Key Production Features**

### **Enterprise-Grade Server**
- **FastMCP Framework**: Clean `@mcp.tool()` decorators
- **Robust Error Handling**: Comprehensive exception management  
- **Configuration Management**: Environment variables + CLI options
- **Performance Monitoring**: Resource usage tracking
- **Security**: Input validation and path restrictions

### **Comprehensive Tool Suite**
- **`create_simulation`** - Single simulation execution
- **`create_experiment`** - Multi-iteration experiments
- **`run_experiment`** - Execute experiments  
- **`analyze_simulation`** - Comprehensive analysis
- **`batch_analyze`** - Multi-simulation analysis
- **`export_simulation_data`** - Data export
- **`list_simulations`** - Discovery and inventory
- **`get_simulation_status`** - Status monitoring
- **`get_simulation_summary`** - Detailed summaries
- **`create_research_project`** - Research organization

### **Production Operations**
- **Health Checks**: `python -m mcp.launcher health-check`
- **Monitoring**: Resource usage and performance tracking
- **Logging**: Structured logging with multiple levels
- **Testing**: Full test suite with integration tests
- **Setup**: Automated installation and configuration

## 🚀 **Quick Start**

### **1. Setup**
```bash
cd mcp
python setup.py
```

### **2. Test**
```bash
python -m mcp.tests
```

### **3. Start Server**
```bash
python -m mcp.main
```

### **4. Run Demo**
```bash
python demo.py
```

## 🎛 **Command Line Interface**

```bash
python -m mcp.main [OPTIONS]

Options:
  --log-level {DEBUG,INFO,WARNING,ERROR}  # Logging level
  --base-dir PATH                         # Data directory
  --transport {stdio,http}                # Transport method
  --host HOST                             # HTTP host (default: 127.0.0.1)
  --port PORT                            # HTTP port (default: 8000)  
  --max-concurrent N                     # Max concurrent simulations
  --memory-limit MB                      # Memory limit
  --dev                                  # Development mode
```

## 📊 **Usage Examples**

### **LLM Agent Integration**
```python
# Create simulation studying cooperation
create_simulation({
    "config": {
        "system_agents": 20,
        "independent_agents": 10,
        "simulation_steps": 1500,
        "seed": 42
    }
})

# Analyze results
analyze_simulation({
    "simulation_path": "simulations/sim_xyz/simulation.db",
    "analysis_types": ["dominance", "social_behavior"]
})

# Export findings
export_simulation_data({
    "simulation_path": "simulations/sim_xyz/simulation.db", 
    "output_path": "research_data",
    "format_type": "csv"
})
```

### **Research Workflow**
```python
# 1. Create research project
create_research_project({
    "name": "cooperation_emergence",
    "description": "Study of cooperative behavior emergence"
})

# 2. Design experiment
create_experiment({
    "name": "cooperation_conditions",
    "base_config": {"simulation_steps": 2000},
    "variations": [
        {"system_agents": 25, "independent_agents": 5},
        {"system_agents": 15, "independent_agents": 15}, 
        {"system_agents": 5, "independent_agents": 25}
    ]
})

# 3. Execute and analyze
run_experiment({"experiment_id": exp_id})
batch_analyze({"experiment_path": experiment_dir})
```

## 🔧 **Configuration**

### **Environment Variables**
```bash
export MCP_LOG_LEVEL=INFO
export MCP_BASE_DIR=/data/agentfarm
export MCP_MAX_CONCURRENT=10
```

### **Production Config**
```python
from mcp.config import MCPServerConfig

config = MCPServerConfig()
config.log_level = "INFO"
config.max_concurrent_simulations = 5
config.memory_limit_mb = 4096
```

## 🎉 **Ready for Production Use**

The FastMCP server is now **production-ready** with:

- ✅ **Clean Architecture**: No legacy code, modular design
- ✅ **Comprehensive Testing**: Full test suite with integration tests
- ✅ **Robust Configuration**: Environment variables, CLI options, validation
- ✅ **Error Handling**: Graceful error management and logging
- ✅ **Performance**: Resource monitoring and optimization
- ✅ **Security**: Input validation and path restrictions
- ✅ **Documentation**: Complete guides and examples

**Start the server**: `python -m mcp.main`

**Connect your LLM agent** and begin conducting sophisticated agent-based simulation research! 🚀