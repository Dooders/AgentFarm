"""Example demonstrating agent termination with context managers."""

from contextlib import ExitStack
from farm.core.config import SimulationConfig
from farm.controllers.simulation_controller import SimulationController
from farm.agents import ControlAgent, SystemAgent

def demonstrate_agent_termination():
    """Show different ways to terminate agents safely."""
    
    config = SimulationConfig.from_yaml("config.yaml")
    
    with SimulationController(config, "simulations/termination.db") as sim:
        sim.initialize_simulation()
        
        # Method 1: Let context manager handle cleanup
        with ExitStack() as stack:
            agents = []
            
            # Create some agents
            for i in range(5):
                agent = stack.enter_context(
                    ControlAgent(f"agent_{i}", (i,0), 10, sim.environment)
                )
                agents.append(agent)
                
            # When we exit the with block, agents are automatically terminated
            
        # Method 2: Early termination within context
        with ExitStack() as stack:
            agents = []
            
            # Create agents
            for i in range(5):
                agent = stack.enter_context(
                    ControlAgent(f"agent_{i}", (i,0), 10, sim.environment)
                )
                agents.append(agent)
            
            # Run simulation until condition
            for step in range(100):
                for agent in agents[:]:  # Copy list to allow modification
                    if agent.alive:
                        # Check termination condition
                        if agent.resource_level < 5:
                            # Proper way to terminate agent early:
                            # 1. Call die() to update state
                            agent.die()
                            # 2. Agent's __exit__ will handle cleanup when context exits
                            agents.remove(agent)
                        else:
                            agent.act()

        # Method 3: Selective termination with nested contexts
        with ExitStack() as outer_stack:
            permanent_agents = []
            
            # Create permanent agents
            for i in range(3):
                agent = outer_stack.enter_context(
                    ControlAgent(f"permanent_{i}", (i,0), 20, sim.environment)
                )
                permanent_agents.append(agent)
            
            # Create temporary agents that we'll terminate early
            for step in range(10):
                with ExitStack() as temp_stack:
                    temp_agents = []
                    
                    # Create temporary agents
                    for i in range(2):
                        agent = temp_stack.enter_context(
                            SystemAgent(f"temp_{step}_{i}", (i,5), 5, sim.environment)
                        )
                        temp_agents.append(agent)
                    
                    # Run temporary agents for a while
                    for _ in range(10):
                        for agent in temp_agents:
                            if agent.alive:
                                agent.act()
                    
                    # Temp agents automatically terminated when inner context exits

            # Permanent agents continue until outer context exits

def demonstrate_parent_child_termination():
    """Show how to handle parent-child termination."""
    
    config = SimulationConfig.from_yaml("config.yaml")
    
    with SimulationController(config, "simulations/family_termination.db") as sim:
        sim.initialize_simulation()
        
        with ExitStack() as stack:
            # Create parent
            parent = stack.enter_context(
                ControlAgent("parent", (0,0), 20, sim.environment)
            )
            
            # Create children with their own stack
            with ExitStack() as child_stack:
                children = []
                for i in range(3):
                    child = child_stack.enter_context(
                        ControlAgent(f"child_{i}", (i,1), 10, sim.environment)
                    )
                    children.append(child)
                    parent.create_child_context(child)
                
                # Run until parent dies
                while parent.alive:
                    if parent.resource_level < 5:
                        # When parent dies:
                        # 1. Parent dies
                        parent.die()
                        # 2. Children are automatically cleaned up due to parent-child relationship
                        break
                    
                    # Normal simulation step
                    parent.act()
                    for child in children:
                        if child.alive:
                            child.act()

def demonstrate_graceful_shutdown():
    """Show how to gracefully shut down agents."""
    
    config = SimulationConfig.from_yaml("config.yaml")
    
    with SimulationController(config, "simulations/shutdown.db") as sim:
        sim.initialize_simulation()
        
        with ExitStack() as stack:
            agents = []
            
            # Create agents
            for i in range(5):
                agent = stack.enter_context(
                    ControlAgent(f"agent_{i}", (i,0), 10, sim.environment)
                )
                agents.append(agent)
            
            try:
                # Run simulation
                for step in range(100):
                    # Simulate emergency shutdown condition
                    if step == 50:
                        raise KeyboardInterrupt("Emergency shutdown")
                        
                    for agent in agents:
                        if agent.alive:
                            agent.act()
                            
            except KeyboardInterrupt:
                # Graceful shutdown:
                # 1. Let agents finish current actions
                for agent in agents:
                    if agent.alive:
                        agent.die()  # Proper state cleanup
                # 2. Context managers handle the rest
                print("Gracefully shut down")

if __name__ == "__main__":
    demonstrate_agent_termination()
    demonstrate_parent_child_termination()
    demonstrate_graceful_shutdown() 