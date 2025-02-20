"""Example showing how to use context managers in a simulation."""

from contextlib import ExitStack
from farm.core.config import SimulationConfig
from farm.controllers.simulation_controller import SimulationController
from farm.agents import ControlAgent, SystemAgent, IndependentAgent

def run_basic_simulation():
    """Run a basic simulation with different agent types."""
    
    # Load config
    config = SimulationConfig.from_yaml("config.yaml")
    
    # Use simulation controller as context manager
    with SimulationController(config, "simulations/test.db") as sim:
        # Initialize simulation first
        sim.initialize_simulation()
        
        # Create agents using ExitStack to manage multiple contexts
        with ExitStack() as stack:
            # Create different types of agents
            agents = [
                # Control agents
                stack.enter_context(
                    ControlAgent("control_1", (0,0), 10, sim.environment)
                ),
                stack.enter_context(
                    ControlAgent("control_2", (1,1), 10, sim.environment)
                ),
                
                # System agents
                stack.enter_context(
                    SystemAgent("system_1", (2,2), 10, sim.environment)
                ),
                
                # Independent agents
                stack.enter_context(
                    IndependentAgent("indie_1", (3,3), 10, sim.environment)
                )
            ]
            
            # Create parent-child relationship between agents
            agents[0].create_child_context(agents[1])  # control_1 is parent of control_2
            
            # Start simulation
            sim.start()
            
            # Run until complete
            while sim.is_running:
                # Each agent acts in turn
                for agent in agents:
                    if agent.alive:
                        agent.act()
                        
                # Optional: Add delay or other logic between steps
                
            # Agents and simulation cleaned up automatically when contexts exit

def run_experiment():
    """Run multiple simulation iterations as an experiment."""
    from farm.controllers.experiment_controller import ExperimentController
    
    config = SimulationConfig.from_yaml("config.yaml")
    
    # Use experiment controller as context manager
    with ExperimentController("test_exp", "Testing agents", config) as exp:
        # Define variations to test
        variations = [
            {"control_agents": 2, "system_agents": 0},
            {"control_agents": 0, "system_agents": 2},
        ]
        
        # Run experiment with variations
        exp.run_experiment(
            num_iterations=3,
            variations=variations,
            num_steps=1000
        )
        
        # Analyze results
        exp.analyze_results()
        # Results and analysis saved automatically when context exits

if __name__ == "__main__":
    # Run basic simulation
    run_basic_simulation()
    
    # Run experiment
    run_experiment() 