# AgentFarm FastMCP Server

A **FastMCP** server that provides LLM agents with programmatic access to the AgentFarm simulation system. This server uses the [FastMCP framework](https://github.com/jlowin/fastmcp) for simplified MCP server development.

## 🎯 **What This Provides**

An LLM agent can use this server to:

- **🔬 Create Simulations** - Configure and run agent-based simulations
- **🧪 Run Experiments** - Execute multi-iteration parameter studies  
- **📊 Analyze Results** - Perform comprehensive analysis on outcomes
- **📁 Export Data** - Extract simulation data in various formats
- **🔍 Compare Studies** - Analyze differences between simulation runs

## 🚀 **Quick Start**

### 1. Install Dependencies
```bash
pip install -r fastmcp_requirements.txt
```

### 2. Start the Server
```bash
python start_fastmcp_server.py
```

### 3. Connect Your LLM Agent
The server runs using FastMCP's stdio transport, ready for LLM connections.

## 🛠 **Available Tools**

### Core Simulation Tools

#### `create_simulation`
**Purpose**: Create and run a single simulation
**Parameters**:
```json
{
  "config": {
    "width": 100,
    "height": 100,
    "system_agents": 15,
    "independent_agents": 10,
    "control_agents": 5,
    "simulation_steps": 1000,
    "initial_resources": 25,
    "seed": 42
  },
  "output_path": "my_simulation"
}
```

#### `create_experiment`
**Purpose**: Set up multi-iteration experiments with parameter variations
**Parameters**:
```json
{
  "name": "population_study",
  "base_config": {"simulation_steps": 1000},
  "variations": [
    {"system_agents": 20, "independent_agents": 5},
    {"system_agents": 10, "independent_agents": 15}
  ],
  "num_iterations": 2
}
```

#### `run_experiment`
**Purpose**: Execute a created experiment
**Parameters**:
```json
{
  "experiment_id": "exp_20241215_143022_123456",
  "run_analysis": true
}
```

### Analysis Tools

#### `analyze_simulation`
**Purpose**: Run comprehensive analysis on results
**Parameters**:
```json
{
  "simulation_path": "experiments/my_study/",
  "analysis_types": ["dominance", "advantage"],
  "output_path": "analysis/my_study_results"
}
```

#### `batch_analyze`
**Purpose**: Analyze multiple simulations across an experiment
**Parameters**:
```json
{
  "experiment_path": "experiments/population_study/",
  "analysis_modules": ["dominance", "advantage"],
  "save_to_db": true
}
```

### Data Management Tools

#### `export_simulation_data`
**Purpose**: Export simulation data in various formats
**Parameters**:
```json
{
  "simulation_path": "simulations/sim_001/simulation.db",
  "output_path": "exports/sim_001_data",
  "format_type": "csv",
  "data_types": ["agents", "actions", "steps"]
}
```

#### `list_simulations`
**Purpose**: List all available simulations
**Parameters**:
```json
{
  "search_path": "simulations"
}
```

#### `get_simulation_status`
**Purpose**: Get status and metrics of a simulation
**Parameters**:
```json
{
  "simulation_id": "20241215_143022"
}
```

#### `get_simulation_summary`
**Purpose**: Get detailed summary with statistics
**Parameters**:
```json
{
  "simulation_path": "simulations/sim_001/simulation.db",
  "include_charts": false
}
```

### Research Tools

#### `create_research_project`
**Purpose**: Create structured research project
**Parameters**:
```json
{
  "name": "cooperation_emergence",
  "description": "Study of cooperative behavior emergence",
  "tags": ["cooperation", "emergence", "social_behavior"]
}
```

## 📋 **Example Usage in LLM Agent**

```python
# The LLM agent would make these calls through the MCP protocol:

# 1. Create a simulation to study agent cooperation
create_simulation({
  "config": {
    "system_agents": 20,
    "independent_agents": 10,
    "simulation_steps": 1500,
    "seed": 123
  }
})

# 2. Analyze the results
analyze_simulation({
  "simulation_path": "simulations/sim_20241215_143022/simulation.db",
  "analysis_types": ["social_behavior", "dominance"]
})

# 3. Export data for further analysis
export_simulation_data({
  "simulation_path": "simulations/sim_20241215_143022/simulation.db",
  "output_path": "research_data",
  "format_type": "csv"
})
```

## 🎯 **Research Workflow Example**

```python
# 1. Create research project
create_research_project({
  "name": "agent_cooperation_study",
  "description": "Investigating emergence of cooperative behaviors"
})

# 2. Design experiment with parameter variations
create_experiment({
  "name": "cooperation_conditions",
  "base_config": {"simulation_steps": 2000},
  "variations": [
    {"system_agents": 25, "independent_agents": 5},  # High cooperation
    {"system_agents": 5, "independent_agents": 25},   # High competition
    {"system_agents": 15, "independent_agents": 15}   # Balanced
  ]
})

# 3. Run experiment
run_experiment({
  "experiment_id": "exp_20241215_143022_123456",
  "run_analysis": true
})

# 4. Batch analysis across all conditions
batch_analyze({
  "experiment_path": "experiments/cooperation_conditions_20241215_143022/",
  "analysis_modules": ["dominance", "advantage", "social_behavior"]
})
```

## 🔧 **Key Features of FastMCP Implementation**

### **Simplified Tool Definition**
- Clean `@mcp.tool()` decorators
- Automatic schema generation from type hints
- Simple function-based approach

### **Built-in Error Handling**
- Comprehensive exception catching
- Detailed error messages with context
- Graceful degradation on failures

### **JSON Response Format**
- Structured JSON responses for easy parsing
- Consistent error messaging
- Rich metadata in results

### **Global State Management**
- Tracks active simulations and experiments
- Enables status checking and result retrieval
- Memory-efficient simulation tracking

## 📁 **File Structure**

```
AgentFarm/
├── fastmcp_simulation_server.py    # Main FastMCP server
├── start_fastmcp_server.py        # Server launcher
├── fastmcp_client_example.py      # Usage examples
├── fastmcp_requirements.txt       # Dependencies
├── README_FASTMCP.md              # This documentation
├── simulations/                   # Created simulations
├── experiments/                   # Multi-iteration experiments
├── research/                      # Research projects
├── analysis/                      # Analysis outputs
└── exports/                       # Exported data
```

## 🧪 **Testing the Server**

Run the test script to verify everything works:

```bash
python test_mcp_server.py
```

This will validate:
- Core imports and dependencies
- Simulation creation and execution
- Database access and querying
- Configuration handling

## 🔗 **Integration with LLM Clients**

### **Claude Desktop Integration**
Add to your Claude Desktop config:

```json
{
  "mcpServers": {
    "agentfarm": {
      "command": "python",
      "args": ["path/to/fastmcp_simulation_server.py"],
      "cwd": "path/to/AgentFarm"
    }
  }
}
```

### **Other MCP Clients**
The server uses standard stdio transport and is compatible with any MCP-enabled client.

## 🎓 **Analysis Capabilities**

The server provides access to AgentFarm's full analysis suite:

- **Dominance Analysis**: Population dominance patterns and switching dynamics
- **Advantage Analysis**: Resource acquisition and reproduction advantages  
- **Genesis Analysis**: Initial condition effects and outcome prediction
- **Social Behavior Analysis**: Cooperation, competition, and spatial patterns

## 🚨 **Important Notes**

1. **Dependencies**: Ensure PyTorch and other AgentFarm dependencies are installed
2. **Python Path**: The server automatically adds the current directory to Python path
3. **File Permissions**: Ensure write permissions for simulation and analysis directories
4. **Memory Usage**: Large experiments may require significant memory and disk space

## 🆚 **FastMCP vs Standard MCP**

**Advantages of FastMCP**:
- ✅ Simpler tool definition with decorators
- ✅ Automatic schema generation
- ✅ Less boilerplate code
- ✅ Built-in error handling
- ✅ Function-based approach

**Key Differences**:
- Uses `@mcp.tool()` decorators instead of complex schema definitions
- Returns simple strings instead of MCP content objects
- Automatic JSON serialization handling
- Simplified server setup with `mcp.run()`

The FastMCP implementation is much cleaner and easier to maintain while providing the same functionality!