"""Example demonstrating relationship cleanup with context managers."""

from contextlib import ExitStack
from farm.core.config import SimulationConfig
from farm.controllers.simulation_controller import SimulationController
from farm.agents import ControlAgent

def demonstrate_relationship_cleanup():
    """Show how relationship cleanup works."""
    
    config = SimulationConfig.from_yaml("config.yaml")
    
    with SimulationController(config, "simulations/relationships.db") as sim:
        sim.initialize_simulation()
        
        with ExitStack() as stack:
            # Create parent
            parent = stack.enter_context(
                ControlAgent("parent", (0,0), 20, sim.environment)
            )
            
            # Create children
            children = [
                stack.enter_context(
                    ControlAgent(f"child_{i}", (i,1), 10, sim.environment)
                )
                for i in range(3)
            ]
            
            # Establish relationships
            for child in children:
                parent.create_child_context(child)
                
            # When parent dies:
            if parent.resource_level < 5:
                parent.die()
                # This happens automatically in __exit__:
                # 1. Parent's _child_contexts is cleared
                # 2. Each child's _parent_context is set to None
                # 3. Children continue to exist and function independently
                
                # Children are still alive and active
                for child in children:
                    print(f"Child {child.agent_id} is still alive: {child.alive}")
                    print(f"Child {child.agent_id} parent context: {child._parent_context}")
                    child.act()  # Children can still act

def demonstrate_relationship_transfer():
    """Show how relationships can be transferred."""
    
    config = SimulationConfig.from_yaml("config.yaml")
    
    with SimulationController(config, "simulations/transfer.db") as sim:
        sim.initialize_simulation()
        
        with ExitStack() as stack:
            # Create original parent
            parent1 = stack.enter_context(
                ControlAgent("parent1", (0,0), 20, sim.environment)
            )
            
            # Create child
            child = stack.enter_context(
                ControlAgent("child", (1,0), 10, sim.environment)
            )
            
            # Create potential new parent
            parent2 = stack.enter_context(
                ControlAgent("parent2", (2,0), 20, sim.environment)
            )
            
            # Establish initial relationship
            parent1.create_child_context(child)
            
            # When original parent dies
            if parent1.resource_level < 5:
                parent1.die()
                # Relationship is automatically cleaned up
                
                # Can establish new relationship
                parent2.create_child_context(child)
                
                print(f"Child's new parent: {child._parent_context.agent_id}")

if __name__ == "__main__":
    demonstrate_relationship_cleanup()
    demonstrate_relationship_transfer() 