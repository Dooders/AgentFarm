# Ant Colony Simulation Preset

This preset provides a biologically accurate simulation of ant colony behavior with realistic metabolism, resource management, and social dynamics.

## 🐜 Key Features

### **Realistic Metabolism**
- **Base consumption rate**: 0.03 per step (vs 0.15 default)
- **Starvation threshold**: 100 steps (~3-4 days without food)
- **Initial resources**: 30 units (~10 days of energy storage)
- **Resource regeneration**: Slower, more realistic rates

### **Ant-Specific Behavior**
- **Smaller movement range**: 5 units (vs 8 default)
- **Limited perception**: 3 unit radius (vs 2 default)
- **Shorter social interactions**: 15 unit range
- **Higher reproduction costs**: 15 units (vs 3 default)
- **Lower health**: 50 HP (vs 100 default)

### **Colony Dynamics**
- **Population ratio**: 60% system agents, 40% independent
- **Larger colony size**: 50 total agents (vs 30 default)
- **Resource scarcity**: Limited resources encourage cooperation
- **Gradual learning**: Curriculum phases for realistic development

## 🚀 Usage

### **Quick Start**
```bash
python run_ant_colony.py
```

### **Using the Template System**
```python
from farm.api.config_templates import ConfigTemplateManager

# Get the ant colony template
manager = ConfigTemplateManager()
template = manager.get_template("ant_colony")

# Create a configuration
config = manager.create_config_from_template("ant_colony", {
    "steps": 1000,
    "agents": {"system_agents": 20, "independent_agents": 15}
})
```

### **Custom Configuration**
```python
from farm.config.config import SimulationConfig

# Load the preset
with open("ant_colony_preset.json", "r") as f:
    config_data = json.load(f)

config = SimulationConfig.from_dict(config_data)

# Modify parameters
config.agent_behavior.base_consumption_rate = 0.02  # Even more realistic
config.population.system_agents = 40  # Larger colony
```

## 📊 Expected Behavior

### **Resource Management**
- Agents will need to actively gather resources to survive
- Starvation events should occur more frequently than default
- Resource sharing becomes critical for colony survival
- Higher reproduction costs create natural population limits

### **Social Dynamics**
- System agents (cooperative) should thrive in resource-scarce environments
- Independent agents (competitive) may struggle without cooperation
- Resource sharing becomes essential for colony survival
- Combat is less frequent but more meaningful

### **Learning Patterns**
- Initial phase: Basic movement and gathering
- Intermediate phase: Social interactions and sharing
- Advanced phase: Complex strategies including reproduction

## 🔬 Scientific Accuracy

This preset is based on research into ant metabolism and behavior:

- **Metabolic rates**: Based on fire ant (*Solenopsis invicta*) studies
- **Starvation survival**: 3-4 days without food (realistic for worker ants)
- **Resource storage**: 10+ days of energy reserves (realistic fat body storage)
- **Social behavior**: Cooperative resource sharing and colony maintenance
- **Reproduction costs**: Higher costs reflect the energy investment in offspring

## 📈 Monitoring

The simulation tracks key metrics:
- **Starvation deaths**: Should be higher than default simulations
- **Resource levels**: Should show more variation and scarcity
- **Social interactions**: Increased sharing and cooperation
- **Population dynamics**: More realistic growth and decline patterns

## 🎯 Research Applications

This preset is ideal for studying:
- **Collective behavior** in resource-scarce environments
- **Cooperation vs. competition** strategies
- **Metabolic constraints** on social evolution
- **Colony survival** under different resource conditions
- **Learning dynamics** in social insects

## ⚙️ Configuration Files

- `ant_colony_preset.json`: Complete configuration file
- `run_ant_colony.py`: Simple script to run the simulation
- `farm/api/config_templates.py`: Template system integration

## 🔧 Customization

To modify the preset for specific research questions:

1. **Resource scarcity**: Reduce `resources.initial_resources`
2. **Colony size**: Adjust `population.system_agents` and `population.independent_agents`
3. **Metabolism**: Modify `agent_behavior.base_consumption_rate`
4. **Social behavior**: Change `agent_behavior.share_weight` and `agent_behavior.attack_weight`
5. **Learning**: Adjust `learning.epsilon_start` and `learning.epsilon_decay`

## 📚 References

- Fire ant metabolism studies (Solenopsis invicta)
- Ant colony resource management research
- Social insect cooperation and competition studies
- Metabolic scaling in social insects
