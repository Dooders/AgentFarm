# FastMCP AgentFarm Server - Usage Guide

## 🎯 **Overview**

This guide shows how to use the FastMCP AgentFarm server for conducting agent-based simulation research with LLM agents.

## 🚀 **Quick Setup**

### 1. Install Dependencies
```bash
pip install -r fastmcp_requirements.txt
```

### 2. Test the Server
```bash
python test_fastmcp_server.py
```

### 3. Start the Server
```bash
python start_fastmcp_server.py
```

## 📚 **Basic Usage Patterns**

### **Single Simulation Study**

```python
# 1. Create and run simulation
result = create_simulation({
    "config": {
        "width": 100,
        "height": 100,
        "system_agents": 15,
        "independent_agents": 10, 
        "control_agents": 5,
        "simulation_steps": 1000,
        "seed": 42
    },
    "output_path": "studies/cooperation_basic"
})

# 2. Get detailed summary
summary = get_simulation_summary({
    "simulation_path": "studies/cooperation_basic/simulation.db"
})

# 3. Analyze results
analysis = analyze_simulation({
    "simulation_path": "studies/cooperation_basic/simulation.db",
    "analysis_types": ["dominance", "social_behavior"]
})
```

### **Multi-Condition Experiment**

```python
# 1. Create experiment with parameter variations
experiment = create_experiment({
    "name": "agent_ratio_study", 
    "base_config": {
        "width": 100,
        "height": 100,
        "simulation_steps": 1500,
        "initial_resources": 25
    },
    "variations": [
        {"system_agents": 20, "independent_agents": 5, "control_agents": 5},
        {"system_agents": 10, "independent_agents": 15, "control_agents": 5}, 
        {"system_agents": 5, "independent_agents": 5, "control_agents": 20}
    ],
    "description": "Testing how agent type ratios affect population dynamics"
})

# 2. Run the experiment
results = run_experiment({
    "experiment_id": "exp_20241215_143022_123456",
    "run_analysis": true
})

# 3. Batch analyze all conditions
batch_analysis = batch_analyze({
    "experiment_path": "experiments/agent_ratio_study_20241215_143022/",
    "analysis_modules": ["dominance", "advantage", "social_behavior"]
})
```

### **Research Project Workflow**

```python
# 1. Create research project structure
project = create_research_project({
    "name": "emergence_of_cooperation",
    "description": "Investigating how cooperative behaviors emerge in competitive environments",
    "tags": ["cooperation", "emergence", "social_behavior", "multi_agent"]
})

# 2. Design multiple experiments within the project
experiments = [
    # Experiment 1: Resource scarcity effects
    create_experiment({
        "name": "resource_scarcity_cooperation",
        "base_config": {"simulation_steps": 2000},
        "variations": [
            {"initial_resources": 5, "resource_regen_rate": 0.02},
            {"initial_resources": 15, "resource_regen_rate": 0.1}, 
            {"initial_resources": 30, "resource_regen_rate": 0.2}
        ]
    }),
    
    # Experiment 2: Population density effects  
    create_experiment({
        "name": "density_cooperation",
        "base_config": {"simulation_steps": 2000},
        "variations": [
            {"system_agents": 5, "independent_agents": 5},
            {"system_agents": 15, "independent_agents": 15},
            {"system_agents": 30, "independent_agents": 30}
        ]
    })
]

# 3. Run all experiments and analyze
for exp_data in experiments:
    exp_id = extract_experiment_id(exp_data)
    run_experiment({"experiment_id": exp_id})
```

## 🎛 **Common Configuration Patterns**

### **High Competition Environment**
```json
{
  "width": 80,
  "height": 80,
  "system_agents": 5,
  "independent_agents": 25,
  "control_agents": 5,
  "initial_resources": 10,
  "resource_regen_rate": 0.05,
  "attack_base_damage": 15.0
}
```

### **Cooperation-Friendly Environment**
```json
{
  "width": 120,
  "height": 120,
  "system_agents": 25,
  "independent_agents": 5,
  "control_agents": 5,
  "initial_resources": 30,
  "resource_regen_rate": 0.15,
  "share_range": 40.0,
  "cooperation_memory": 200
}
```

### **Resource Scarcity Study**
```json
{
  "width": 100,
  "height": 100,
  "system_agents": 15,
  "independent_agents": 15,
  "control_agents": 15,
  "initial_resources": 8,
  "resource_regen_rate": 0.03,
  "max_resource_amount": 12,
  "starvation_threshold": 2
}
```

## 📊 **Analysis Output Examples**

### **Dominance Analysis Results**
```json
{
  "analysis_completed": true,
  "results": {
    "dominance": {
      "status": "completed",
      "simulations_analyzed": 5,
      "key_findings": {
        "dominant_type": "SystemAgent",
        "dominance_switches": 3,
        "stability_score": 0.75
      }
    }
  }
}
```

### **Advantage Analysis Results**
```json
{
  "analysis_completed": true,
  "results": {
    "advantage": {
      "status": "completed", 
      "simulations_analyzed": 5,
      "resource_acquisition_advantage": {
        "system_vs_independent": 0.23,
        "system_vs_control": 0.15
      }
    }
  }
}
```

## 🔧 **Troubleshooting**

### **Common Issues**

1. **"Module farm not found"**
   - Solution: Ensure you're running from the AgentFarm root directory
   - The server adds the current directory to Python path automatically

2. **"Simulation failed: [error]"**
   - Check the configuration parameters are valid
   - Ensure sufficient disk space for database files
   - Verify environment dimensions are reasonable

3. **"Analysis failed: [error]"**
   - Ensure simulation completed successfully
   - Check that analysis modules are available
   - Verify input paths exist

4. **Memory issues with large experiments**
   - Reduce simulation_steps for testing
   - Use fewer agents for initial experiments
   - Monitor system memory usage

### **Debug Mode**

Enable debug logging for detailed information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### **Validation Checks**

Before running experiments:

```python
# Test with minimal simulation first
test_config = {
    "width": 20,
    "height": 20, 
    "system_agents": 2,
    "independent_agents": 2,
    "control_agents": 1,
    "simulation_steps": 10
}

result = create_simulation({"config": test_config})
```

## 🎓 **Advanced Usage**

### **Custom Analysis Pipeline**
```python
# 1. Run experiment
run_experiment({"experiment_id": exp_id})

# 2. Comprehensive analysis
analyze_simulation({
    "simulation_path": experiment_path,
    "analysis_types": ["dominance", "advantage", "genesis", "social_behavior"]
})

# 3. Export for external tools
export_simulation_data({
    "simulation_path": db_path,
    "format_type": "parquet",  # Efficient format
    "data_types": ["all"]
})

# 4. Compare with baseline
compare_simulations({
    "simulation_paths": [baseline_path, experiment_path],
    "metrics": ["population_dynamics", "cooperation_rates"]
})
```

### **Systematic Parameter Exploration**
```python
# Create systematic parameter grid
parameter_grid = []
for agents in [10, 20, 30]:
    for resources in [15, 25, 35]:
        for steps in [1000, 2000]:
            parameter_grid.append({
                "system_agents": agents,
                "initial_resources": resources,
                "simulation_steps": steps
            })

# Run experiment with full grid
experiment = create_experiment({
    "name": "systematic_exploration",
    "base_config": {"width": 100, "height": 100},
    "variations": parameter_grid,
    "num_iterations": len(parameter_grid)
})
```

## 💡 **Tips for LLM Agents**

1. **Start Small**: Begin with short simulations (100-500 steps) to test configurations

2. **Use Seeds**: Always set a seed for reproducible results

3. **Monitor Progress**: Check simulation status periodically for long-running experiments

4. **Validate Configs**: Test configurations with minimal parameters first

5. **Organize Output**: Use descriptive output paths and naming conventions

6. **Incremental Analysis**: Run basic analysis first, then add more complex modules

This FastMCP server provides a powerful interface for LLM agents to conduct sophisticated agent-based modeling research! 🚀