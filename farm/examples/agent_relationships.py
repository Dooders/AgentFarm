"""Example demonstrating parent-child relationships between agents."""

from contextlib import ExitStack
import math
from farm.core.config import SimulationConfig
from farm.controllers.simulation_controller import SimulationController
from farm.agents import ControlAgent, SystemAgent

def demonstrate_agent_relationships():
    """Show different ways to use agent parent-child relationships."""
    
    config = SimulationConfig.from_yaml("config.yaml")
    
    with SimulationController(config, "simulations/relationships.db") as sim:
        sim.initialize_simulation()
        
        with ExitStack() as stack:
            # Create a family structure
            parent = stack.enter_context(
                ControlAgent("parent", (0,0), 20, sim.environment)
            )
            
            # Create children
            children = [
                stack.enter_context(
                    ControlAgent(f"child_{i}", (i,i), 10, sim.environment)
                )
                for i in range(3)
            ]
            
            # Create grandchildren
            grandchildren = [
                stack.enter_context(
                    ControlAgent(f"grandchild_{i}", (i+5,i+5), 5, sim.environment)
                )
                for i in range(2)
            ]
            
            # Establish relationships
            for child in children:
                # Make each child a child of parent
                parent.create_child_context(child)
            
            # Make grandchildren children of first child
            for grandchild in grandchildren:
                children[0].create_child_context(grandchild)

            # Example behaviors using relationships:
            
            # 1. Resource sharing within family
            def share_resources_with_family():
                """Parent shares resources with children when they're low."""
                for child in children:
                    if child.resource_level < 5 and parent.resource_level > 10:
                        transfer = min(5, parent.resource_level - 10)
                        parent.resource_level -= transfer
                        child.resource_level += transfer
            
            # 2. Defensive formation
            def form_defensive_group():
                """Children move closer to parent when threatened."""
                parent_pos = parent.position
                for i, child in enumerate(children):
                    # Form circle around parent
                    angle = (i / len(children)) * 2 * 3.14159
                    child.position = (
                        parent_pos[0] + math.cos(angle) * 2,
                        parent_pos[1] + math.sin(angle) * 2
                    )
            
            # 3. Coordinated gathering
            def coordinate_gathering():
                """Family members coordinate to gather resources efficiently."""
                if parent.alive:
                    # Parent scouts for resources
                    resources = parent.environment.get_nearby_resources(parent.position, 10)
                    if resources:
                        # Assign resources to children
                        for child, resource in zip(children, resources):
                            if child.alive:
                                child.position = resource.position
                                child.gather_module.gather(resource)

            # 4. Inheritance on death
            def distribute_inheritance():
                """When parent dies, resources are distributed to children."""
                if not parent.alive and parent.resource_level > 0:
                    share = parent.resource_level / len(children)
                    for child in children:
                        if child.alive:
                            child.resource_level += share
                    parent.resource_level = 0

            # Run simulation with family behaviors
            sim.start()
            
            while sim.is_running:
                if parent.alive:
                    share_resources_with_family()
                    
                if any(child.current_health < child.starting_health * 0.5 
                      for child in children):
                    form_defensive_group()
                else:
                    coordinate_gathering()
                    
                distribute_inheritance()
                
                # Normal agent actions
                if parent.alive:
                    parent.act()
                for child in children:
                    if child.alive:
                        child.act()
                for grandchild in grandchildren:
                    if grandchild.alive:
                        grandchild.act()

def demonstrate_agent_evolution():
    """Show how parent-child relationships can be used for evolution."""
    
    config = SimulationConfig.from_yaml("config.yaml")
    
    with SimulationController(config, "simulations/evolution.db") as sim:
        sim.initialize_simulation()
        
        with ExitStack() as stack:
            # Create initial generation
            parents = [
                stack.enter_context(
                    SystemAgent(f"gen0_{i}", (i,0), 10, sim.environment)
                )
                for i in range(5)
            ]
            
            generation = 0
            while generation < 10:  # Run for 10 generations
                # Run current generation
                while any(parent.alive for parent in parents):
                    for parent in parents:
                        if parent.alive:
                            parent.act()
                
                # Create next generation from most successful parents
                successful_parents = sorted(
                    parents, 
                    key=lambda p: p.total_reward,
                    reverse=True
                )[:2]
                
                # Create children with inherited traits
                new_generation = []
                for i, parent in enumerate(successful_parents):
                    children = [
                        stack.enter_context(
                            SystemAgent(
                                f"gen{generation+1}_{i}_{j}", 
                                (j,generation+1), 
                                10, 
                                sim.environment,
                                parent_ids=[parent.agent_id],
                                generation=generation+1
                            )
                        )
                        for j in range(3)
                    ]
                    
                    # Establish parent-child relationship
                    for child in children:
                        parent.create_child_context(child)
                    
                    new_generation.extend(children)
                
                parents = new_generation
                generation += 1

if __name__ == "__main__":
    demonstrate_agent_relationships()
    demonstrate_agent_evolution() 