import json
import os
from pathlib import Path
from typing import List, Optional

import networkx as nx
from graphviz import Digraph
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from farm.database.models import AgentModel, AgentStateModel, ReproductionEventModel


def get_database_session() -> Session:
    """Create and return a database session."""
    engine = create_engine(
        "sqlite:///simulations/simulation.db"
    )  # Adjust connection string as needed
    Session = sessionmaker(bind=engine)
    return Session()


def generate_family_tree(
    session: Session,
    output_path: str = "family_tree",
    max_generations: Optional[int] = None,
    include_agent_details: bool = True,
    color_by: str = "offspring",  # 'offspring', 'age', or 'resources'
) -> Digraph:
    """
    Generate and save a family tree visualization using reproduction events.
    Nodes can be colored based on offspring count, age, or total resources consumed.

    Parameters
    ----------
    session : Session
        SQLAlchemy session for database queries
    output_path : str
        Path where the visualization should be saved (without extension)
    max_generations : Optional[int]
        Maximum number of generations to include in visualization
    include_agent_details : bool
        Whether to include additional agent details in node labels
    color_by : str
        'offspring' to color by number of offspring
        'age' to color by agent age
        'resources' to color by total resources consumed

    Returns
    -------
    dot : graphviz.Digraph
        A Graphviz Digraph object representing the family tree
    """
    dot = Digraph(comment="Agent Family Tree", engine="dot")
    dot.attr(rankdir="LR")
    dot.attr("node", shape="box", style="rounded")

    # Query reproduction events
    query = session.query(ReproductionEventModel).filter(
        ReproductionEventModel.success == True
    )
    if max_generations:
        query = query.filter(ReproductionEventModel.parent_generation < max_generations)
    reproduction_events = query.all()

    # Create sets to track agents and count offspring
    agents = set()
    offspring_count = {}
    for event in reproduction_events:
        agents.add(event.parent_id)
        if event.offspring_id:
            agents.add(event.offspring_id)
            offspring_count[event.parent_id] = (
                offspring_count.get(event.parent_id, 0) + 1
            )

    # Query agent details and states
    agent_details = {}
    resources_consumed = {}
    if include_agent_details:
        agent_models = (
            session.query(AgentModel).filter(AgentModel.agent_id.in_(agents)).all()
        )
        agent_details = {agent.agent_id: agent for agent in agent_models}

        # Calculate total resources consumed for each agent
        for agent_id in agents:
            states = (
                session.query(AgentStateModel)
                .filter(AgentStateModel.agent_id == agent_id)
                .order_by(AgentStateModel.step_number)
                .all()
            )

            if states:
                # Calculate total resources consumed from changes in resource levels
                total_consumed = 0
                for i in range(1, len(states)):
                    resource_diff = (
                        states[i - 1].resource_level - states[i].resource_level
                    )
                    if resource_diff > 0:  # Only count decreases as consumption
                        total_consumed += resource_diff
                resources_consumed[agent_id] = total_consumed

    def get_node_color(agent_id):
        if color_by == "offspring":
            count = offspring_count.get(agent_id, 0)
            if count == 0:
                return "#ffffff"  # White in hex format
            max_count = max(offspring_count.values())
            intensity = count / max_count

        elif color_by == "age":
            if agent_id not in agent_details:
                return "#ffffff"
            agent = agent_details[agent_id]
            death_time = agent.death_time or max(
                state.step_number for state in agent.states
            )
            age = death_time - agent.birth_time
            max_age = max(
                (a.death_time or max(s.step_number for s in a.states)) - a.birth_time
                for a in agent_details.values()
            )
            intensity = age / max_age if max_age > 0 else 0

        else:  # color_by == "resources"
            consumed = resources_consumed.get(agent_id, 0)
            if consumed == 0:
                return "#ffffff"
            max_consumed = max(resources_consumed.values())
            intensity = consumed / max_consumed

        # Convert to a hex color from light blue to dark blue
        # Ensure rgb_value is between 0 and 255
        rgb_value = max(
            0, min(255, int(255 * (1 - intensity * 0.3)))
        )  # 70% minimum brightness
        hex_value = f"#{rgb_value:02x}{rgb_value:02x}ff"
        return hex_value

    # Add nodes with agent information
    for agent_id in agents:
        label = f"Agent {agent_id}"
        if include_agent_details and agent_id in agent_details:
            agent = agent_details[agent_id]
            label = (
                f"Agent {agent_id}\n"
                f"Gen: {agent.generation}\n"
                f"Type: {agent.agent_type}\n"
                f"Birth: {agent.birth_time}"
            )
            if color_by == "offspring":
                label += f"\nOffspring: {offspring_count.get(agent_id, 0)}"
            elif color_by == "resources":
                label += (
                    f"\nResources consumed: {resources_consumed.get(agent_id, 0):.1f}"
                )

        dot.node(
            agent_id,
            label=label,
            fillcolor=get_node_color(agent_id),
            style="filled,rounded",
        )

    # Add edges from parent to offspring
    for event in reproduction_events:
        if event.offspring_id:
            dot.edge(
                event.parent_id, event.offspring_id, label=f"Step {event.step_number}"
            )

    # Save both PDF and interactive versions
    dot.render(output_path + "_pdf", view=False, cleanup=True)  # Added "_pdf" suffix
    return dot


def generate_interactive_tree(
    session: Session,
    output_path: str = "family_tree",
    max_generations: Optional[int] = None,
    color_by: str = "offspring",
) -> None:
    """
    Generate an interactive web-based family tree visualization.

    Creates an HTML file with a responsive visualization that supports:
    - Pan and zoom
    - Node selection and highlighting
    - Node details on hover/click
    - Search functionality
    - Filtering by generation
    """
    # Query reproduction events
    query = session.query(ReproductionEventModel).filter(
        ReproductionEventModel.success == True
    )
    if max_generations:
        query = query.filter(ReproductionEventModel.parent_generation < max_generations)
    reproduction_events = query.all()

    # Create sets to track agents and count offspring
    agents = set()
    offspring_count = {}
    for event in reproduction_events:
        agents.add(event.parent_id)
        if event.offspring_id:
            agents.add(event.offspring_id)
            offspring_count[event.parent_id] = (
                offspring_count.get(event.parent_id, 0) + 1
            )

    # Query agent details and calculate resources
    agent_details = {}
    resources_consumed = {}
    agent_models = (
        session.query(AgentModel).filter(AgentModel.agent_id.in_(agents)).all()
    )
    agent_details = {agent.agent_id: agent for agent in agent_models}

    # Calculate resources consumed
    for agent_id in agents:
        states = (
            session.query(AgentStateModel)
            .filter(AgentStateModel.agent_id == agent_id)
            .order_by(AgentStateModel.step_number)
            .all()
        )

        if states:
            total_consumed = 0
            for i in range(1, len(states)):
                resource_diff = states[i - 1].resource_level - states[i].resource_level
                if resource_diff > 0:
                    total_consumed += resource_diff
            resources_consumed[agent_id] = total_consumed

    def get_node_color(agent_id):
        if color_by == "offspring":
            count = offspring_count.get(agent_id, 0)
            if count == 0:
                return "#ffffff"  # White in hex format
            max_count = max(offspring_count.values())
            intensity = count / max_count
        else:  # color_by == "resources"
            consumed = resources_consumed.get(agent_id, 0)
            if consumed == 0:
                return "#ffffff"
            max_consumed = max(resources_consumed.values())
            intensity = consumed / max_consumed

        # Convert to a hex color from light blue to dark blue
        # Ensure rgb_value is between 0 and 255
        rgb_value = max(
            0, min(255, int(255 * (1 - intensity * 0.3)))
        )  # 70% minimum brightness
        hex_value = f"#{rgb_value:02x}{rgb_value:02x}ff"
        return hex_value

    # Convert to network structure
    G = nx.DiGraph()

    # Add nodes with their attributes
    for agent_id in agents:
        if agent_id in agent_details:
            agent = agent_details[agent_id]
            G.add_node(
                agent_id,
                generation=agent.generation,
                agent_type=agent.agent_type,
                birth_time=agent.birth_time,
                offspring_count=offspring_count.get(agent_id, 0),
                resources_consumed=resources_consumed.get(agent_id, 0),
                color=get_node_color(agent_id),
            )

    # Add edges
    for event in reproduction_events:
        if event.offspring_id:
            G.add_edge(event.parent_id, event.offspring_id, step=event.step_number)

    # Convert to JSON format for visualization
    graph_data = {
        "nodes": [
            {
                "id": node,
                "label": (f"Agent {node}"),
                "details": (
                    f"Generation: {G.nodes[node]['generation']}\n"
                    f"Type: {G.nodes[node]['agent_type']}\n"
                    f"Birth: {G.nodes[node]['birth_time']}\n"
                    + (
                        f"Offspring: {G.nodes[node]['offspring_count']}"
                        if color_by == "offspring"
                        else f"Resources: {G.nodes[node]['resources_consumed']:.1f}"
                    )
                )
                .replace("Generation:", "<b>Generation:</b>")
                .replace("Type:", "<b>Type:</b>")
                .replace("Birth:", "<b>Birth:</b>")
                .replace("Offspring:", "<b>Offspring:</b>")
                .replace("Resources:", "<b>Resources:</b>"),
                **G.nodes[node],
            }
            for node in G.nodes()
        ],
        "edges": [
            {"source": u, "target": v, "step": G.edges[u, v]["step"]}
            for u, v in G.edges()
        ],
    }

    # Add debug logging
    print(f"Number of nodes: {len(graph_data['nodes'])}")
    print(f"Number of edges: {len(graph_data['edges'])}")
    if len(graph_data["nodes"]) == 0:
        print("No nodes found in graph!")
        print(f"Number of agents: {len(agents)}")
        print(f"Number of agent_details: {len(agent_details)}")

    # Create output directory if it doesn't exist
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write data file
    with open(f"{output_path}.json", "w") as f:
        json.dump(graph_data, f)

    # Generate HTML file with visualization that references an external JS file
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Interactive Family Tree</title>
    <link rel="icon" href="data:,">  <!-- Empty favicon to prevent 404 -->
    <script src="https://unpkg.com/cytoscape@3.23.0/dist/cytoscape.min.js"></script>
    <script src="https://unpkg.com/dagre@0.8.5/dist/dagre.min.js"></script>
    <script src="https://unpkg.com/cytoscape-dagre@2.5.0/cytoscape-dagre.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/lodash.js/4.17.21/lodash.min.js"></script>
    <style>
        body { margin: 0; background-color: #000; }
        #cy { 
            width: 100vw; 
            height: 100vh; 
            position: absolute;
            font-family: Arial, sans-serif;
            background-color: #000;
        }
        #controls {
            position: fixed;
            top: 10px;
            right: 10px;
            z-index: 1000;
            display: flex;
            flex-direction: column;
            gap: 10px;
            background: rgba(0, 0, 0, 0.7);
            padding: 10px;
            border-radius: 5px;
            color: white;
            font-family: monospace;
        }
        .control-row {
            display: flex;
            gap: 10px;
            align-items: center;
        }
        .control-button {
            background: rgba(255, 255, 255, 0.8);
            border: none;
            padding: 5px 10px;
            border-radius: 3px;
            cursor: pointer;
        }
        .info-text {
            font-size: 12px;
            white-space: nowrap;
        }
        #hud-overlay {
            position: fixed;
            top: 20px;
            left: 20px;
            min-width: 200px;
            min-height: 100px;
            max-width: 300px;
            background-color: #001428;
            border: 2px solid #00ff00;
            z-index: 9999;
            border-radius: 5px;
            box-shadow: 0 0 10px rgba(0, 128, 255, 0.5);
            padding: 15px;
            color: white;
            font-family: monospace;
            pointer-events: none;
            opacity: 0;
            display: none;
            transition: opacity 0.3s ease-in-out;
        }
        #hud-overlay h3 {
            margin: 0 0 10px 0;
            color: #4af;
            font-size: 14px;
        }
        #hud-overlay p {
            margin: 5px 0;
            font-size: 12px;
            line-height: 1.4;
        }
        #hud-overlay strong {
            color: #8cf;
        }
        #hud-overlay-2, #hud-overlay-3 {
            pointer-events: none;
        }
        select.control-button {
            background: rgba(255, 255, 255, 0.8);
            border: none;
            padding: 5px 10px;
            border-radius: 3px;
            cursor: pointer;
            color: #000;
            font-family: monospace;
            min-width: 150px;
        }
        
        select.control-button:hover {
            background: rgba(255, 255, 255, 0.9);
        }
        
        select.control-button:focus {
            outline: none;
            box-shadow: 0 0 0 2px rgba(0, 128, 255, 0.5);
        }
    </style>
</head>
<body>
    <div id="hud-overlay"></div>
    <div id="controls">
        <div class="control-row">
            <div id="zoom-level" class="info-text">Zoom: 1.00</div>
            <button class="control-button" id="fit-button">Fit to Screen</button>
        </div>
        <div class="control-row">
            <select id="color-by" class="control-button">
                <option value="offspring">Color by Offspring</option>
                <option value="resources">Color by Resources</option>
                <option value="age">Color by Age</option>
            </select>
        </div>
        <div id="root-position" class="info-text">Root: (0, 0)</div>
    </div>
    <div id="cy"></div>
    <script src="/static/js/family_tree.js"></script>
</body>
</html>"""

    with open(f"{output_path}.html", "w") as f:
        f.write(html_content)

    print(f"Interactive visualization created at {output_path}.html")


if __name__ == "__main__":
    session = get_database_session()
    try:
        # Generate both visualizations
        generate_family_tree(
            session, output_path="family_tree", max_generations=5, color_by="offspring"
        )

        generate_interactive_tree(
            session,
            output_path="family_tree",
            max_generations=5,
            color_by="offspring",
        )
        print(
            "Open family_tree.html in your web browser to view the interactive visualization"
        )
        print("Check family_tree_pdf.pdf for the static visualization")
    finally:
        session.close()
