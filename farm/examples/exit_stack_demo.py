"""Demonstration of ExitStack usage in simulation contexts."""

from contextlib import ExitStack
from farm.core.config import SimulationConfig
from farm.controllers.simulation_controller import SimulationController
from farm.agents import ControlAgent, SystemAgent, IndependentAgent

def demonstrate_exit_stack():
    """Show different ways to use ExitStack with agents."""
    
    config = SimulationConfig.from_yaml("config.yaml")
    
    # ExitStack manages multiple context managers
    with SimulationController(config, "simulations/exit_stack_demo.db") as sim:
        sim.initialize_simulation()
        
        # Basic ExitStack usage
        with ExitStack() as stack:
            # Create different types of agents
            agents = []
            
            # Each enter_context() call registers a new context manager
            # They will be cleaned up in reverse order when ExitStack exits
            agents.append(
                stack.enter_context(
                    ControlAgent("control", (0,0), 10, sim.environment)
                )
            )
            
            agents.append(
                stack.enter_context(
                    SystemAgent("system", (1,1), 10, sim.environment)
                )
            )
            
            agents.append(
                stack.enter_context(
                    IndependentAgent("indie", (2,2), 10, sim.environment)
                )
            )
            
            # All agents are now active and will be cleaned up automatically

def demonstrate_dynamic_contexts():
    """Show how to dynamically add/remove contexts."""
    
    config = SimulationConfig.from_yaml("config.yaml")
    
    with SimulationController(config, "simulations/dynamic.db") as sim:
        sim.initialize_simulation()
        
        with ExitStack() as stack:
            agents = []
            
            # Dynamically create agents based on condition
            for i in range(10):
                if i % 2 == 0:
                    # Even numbered agents are ControlAgents
                    agent = stack.enter_context(
                        ControlAgent(f"control_{i}", (i,0), 10, sim.environment)
                    )
                else:
                    # Odd numbered agents are SystemAgents
                    agent = stack.enter_context(
                        SystemAgent(f"system_{i}", (i,0), 10, sim.environment)
                    )
                agents.append(agent)

def demonstrate_nested_stacks():
    """Show how to use nested ExitStacks for complex hierarchies."""
    
    config = SimulationConfig.from_yaml("config.yaml")
    
    with SimulationController(config, "simulations/nested.db") as sim:
        sim.initialize_simulation()
        
        # Outer stack for parent agents
        with ExitStack() as parent_stack:
            parents = []
            
            # Create parent agents
            for i in range(3):
                parent = parent_stack.enter_context(
                    ControlAgent(f"parent_{i}", (i,0), 20, sim.environment)
                )
                parents.append(parent)
                
                # Inner stack for each parent's children
                with ExitStack() as child_stack:
                    children = []
                    
                    # Create children for this parent
                    for j in range(2):
                        child = child_stack.enter_context(
                            ControlAgent(
                                f"child_{i}_{j}", 
                                (i,j+1), 
                                10, 
                                sim.environment
                            )
                        )
                        children.append(child)
                        parent.create_child_context(child)
                    
                    # Children will be cleaned up when child_stack exits
                
            # Parents will be cleaned up when parent_stack exits

def demonstrate_callback_stack():
    """Show how to use ExitStack with callbacks."""
    
    config = SimulationConfig.from_yaml("config.yaml")
    
    with SimulationController(config, "simulations/callback.db") as sim:
        sim.initialize_simulation()
        
        with ExitStack() as stack:
            # Register cleanup callbacks
            def cleanup_callback(agent):
                print(f"Cleaning up {agent.agent_id}")
                if agent.alive:
                    agent.die()
            
            agents = []
            for i in range(5):
                agent = ControlAgent(f"agent_{i}", (i,0), 10, sim.environment)
                # Push callback that will be called when stack exits
                stack.callback(cleanup_callback, agent)
                agents.append(agent)
                
            # Callbacks will be executed in reverse order when stack exits

def demonstrate_error_handling():
    """Show how ExitStack handles errors."""
    
    config = SimulationConfig.from_yaml("config.yaml")
    
    try:
        with SimulationController(config, "simulations/error.db") as sim:
            sim.initialize_simulation()
            
            with ExitStack() as stack:
                agents = []
                
                # Create some agents
                for i in range(5):
                    agent = stack.enter_context(
                        ControlAgent(f"agent_{i}", (i,0), 10, sim.environment)
                    )
                    agents.append(agent)
                
                # Simulate an error
                raise RuntimeError("Simulation error!")
                
    except RuntimeError as e:
        print(f"Error occurred: {e}")
        print("All agents were properly cleaned up by ExitStack")

if __name__ == "__main__":
    demonstrate_exit_stack()
    demonstrate_dynamic_contexts()
    demonstrate_nested_stacks()
    demonstrate_callback_stack()
    demonstrate_error_handling() 