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
                return "white"
            max_count = max(offspring_count.values())
            intensity = count / max_count

        elif color_by == "age":
            if agent_id not in agent_details:
                return "white"
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
                return "white"
            max_consumed = max(resources_consumed.values())
            intensity = consumed / max_consumed

        # Convert to a hex color from light blue to dark blue
        rgb_value = int(255 * (1 - intensity * 0.3))  # 70% minimum brightness
        return f"#{rgb_value:02x}{rgb_value:02x}ff"

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
                return "white"
            max_count = max(offspring_count.values())
            intensity = count / max_count
        else:  # color_by == "resources"
            consumed = resources_consumed.get(agent_id, 0)
            if consumed == 0:
                return "white"
            max_consumed = max(resources_consumed.values())
            intensity = consumed / max_consumed

        rgb_value = int(255 * (1 - intensity * 0.3))
        return f"#{rgb_value:02x}{rgb_value:02x}ff"

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

    # Generate HTML file with visualization
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
                width: 150px;
                height: 150px;
                background: rgba(0, 128, 255, 0.8);
                border: 1px solid rgba(0, 128, 255, 1);
                z-index: 1000;
                border-radius: 5px;
                box-shadow: 0 0 10px rgba(0, 128, 255, 0.5);
                pointer-events: none;  /* Allow clicking through the HUD */
            }
            #hud-overlay-2, #hud-overlay-3 {
                pointer-events: none;  /* Allow clicking through the HUD */
            }
        </style>
    </head>
    <body>
        <div id="hud-overlay"></div>
        <div id="hud-overlay-2"></div>
        <div id="hud-overlay-3"></div>
        <div id="controls">
            <div class="control-row">
                <div id="zoom-level" class="info-text">Zoom: 1.00</div>
                <button class="control-button" id="fit-button">Fit to Screen</button>
            </div>
            <div id="root-position" class="info-text">Root: (0, 0)</div>
        </div>
        <div id="cy"></div>
        <script>
            fetch('/static/family_tree.json')
                .then(response => response.json())
                .then(data => {
                    const HUD_WIDTH = 200;
                    const HUD_HEIGHT = 200;
                    let isPushing = false;

                    const cy = cytoscape({
                        container: document.getElementById('cy'),
                        elements: {
                            nodes: data.nodes.map(node => ({
                                data: { ...node },
                                grabbable: false
                            })),
                            edges: data.edges.map(edge => ({
                                data: { source: edge.source, target: edge.target, label: `Step ${edge.step}` }
                            }))
                        },
                        userPanningEnabled: true,
                        userZoomingEnabled: true,
                        boxSelectionEnabled: false,
                        autounselectify: true,
                        wheelSensitivity: 0.2,
                        minZoom: 0.01,
                        maxZoom: 3,
                        style: [
                            {
                                selector: 'node',
                                style: {
                                    'label': function(ele) {
                                        return ele.data('label') + '\\n\\n' + ele.data('details');
                                    },
                                    'text-valign': 'center',
                                    'text-halign': 'center',
                                    'text-wrap': 'wrap',
                                    'background-color': function(ele) {
                                        const zoom = ele.cy().zoom();
                                        const baseColor = ele.data('color');
                                        if (baseColor === 'white') return baseColor;
                                        
                                        // Parse the original RGB values
                                        const r = parseInt(baseColor.slice(1,3), 16);
                                        const g = parseInt(baseColor.slice(3,5), 16);
                                        const b = parseInt(baseColor.slice(5,7), 16);
                                        
                                        // More aggressive darkening based on zoom
                                        // Will make distant nodes much brighter against black
                                        const darknessFactor = Math.max(0.2, Math.min(1, Math.pow(zoom, 2)));
                                        
                                        // Adjust RGB values, keeping blue at 255
                                        const newR = Math.round(r / darknessFactor);
                                        const newG = Math.round(g / darknessFactor);
                                        
                                        return `#${newR.toString(16).padStart(2,'0')}${newG.toString(16).padStart(2,'0')}ff`;
                                    },
                                    'text-valign': 'center',
                                    'text-halign': 'center',
                                    'text-wrap': 'wrap',
                                    'shape': 'rectangle',
                                    'width': '300px',
                                    'height': '180px',
                                    'padding': '20px',
                                    'font-size': '20px',
                                    'text-max-width': '280px',
                                    'border-width': '2px',
                                    'border-color': '#666',
                                    'border-style': 'solid',
                                    'text-margin-y': '8px',
                                    'text-background-color': 'white',
                                    'text-background-opacity': '0.9',
                                    'text-background-padding': '8px',
                                    'text-wrap': 'wrap',
                                    'text-transform': 'none'
                                }
                            },
                            {
                                selector: 'edge',
                                style: {
                                    'curve-style': 'unbundled-bezier',
                                    'control-point-distances': [50],
                                    'control-point-weights': [0.5],
                                    'target-arrow-shape': 'triangle',
                                    'label': 'data(label)',
                                    'font-size': '12px',
                                    'text-rotation': 'autorotate',
                                    'text-margin-y': '-10px',
                                    'line-color': '#fff',
                                    'target-arrow-color': '#fff',
                                    'width': 2,
                                    'color': '#fff'
                                }
                            }
                        ],
                        panningEnabled: true,
                        userPanningEnabled: true,
                        minZoom: 0.01,
                        maxZoom: 3,
                    });

                    // Soft boundary pan logic
                    let lastPan = { x: 0, y: 0 };
                    cy.on('pan', _.throttle(() => {
                        const zoom = cy.zoom();
                        const pan = cy.pan();
                        const panDelta = {
                            x: pan.x - lastPan.x,
                            y: pan.y - lastPan.y
                        };
                        
                        // Check if we're in the HUD zone
                        const inHudZone = {
                            x: -pan.x / zoom < HUD_WIDTH,
                            y: -pan.y / zoom < HUD_HEIGHT
                        };

                        // If actively pushing against HUD (large pan delta), allow breakthrough
                        if (Math.abs(panDelta.x) > 20 || Math.abs(panDelta.y) > 20) {
                            isPushing = true;
                            lastPan = pan;
                            return;  // Allow the pan
                        }

                        // If not pushing and in HUD zone, apply soft resistance
                        if (!isPushing && (inHudZone.x || inHudZone.y)) {
                            const newPan = { ...pan };
                            if (inHudZone.x) {
                                newPan.x = -HUD_WIDTH * zoom;
                            }
                            if (inHudZone.y) {
                                newPan.y = -HUD_HEIGHT * zoom;
                            }
                            cy.pan(newPan);
                        }

                        lastPan = pan;
                    }, 16));

                    // Reset pushing state when pan ends
                    cy.on('panend', () => {
                        isPushing = false;
                    });

                    // Run initial layout centered in available space
                    cy.layout({
                        name: 'dagre',
                        rankDir: 'LR',
                        nodeSep: 50,
                        rankSep: 200,
                        padding: 50,
                        animate: false,
                        spacingFactor: 1.0
                    }).run();

                    // Center the graph in the available space (excluding HUD)
                    const centerGraph = () => {
                        const bb = cy.elements().boundingBox();
                        const availableWidth = cy.width() - HUD_WIDTH;
                        const availableHeight = cy.height() - HUD_HEIGHT;
                        
                        // Calculate center position in available space
                        const centerX = HUD_WIDTH + (availableWidth / 2);
                        const centerY = HUD_HEIGHT + (availableHeight / 2);
                        
                        // Calculate required zoom to fit
                        const widthRatio = availableWidth / bb.w;
                        const heightRatio = availableHeight / bb.h;
                        const newZoom = Math.min(widthRatio, heightRatio) * 0.8; // 80% of available space
                        
                        // Set zoom and pan to center
                        cy.zoom(newZoom);
                        cy.center();
                        
                        // Adjust pan to account for HUD
                        const currentPan = cy.pan();
                        cy.pan({
                            x: currentPan.x + (HUD_WIDTH / 2),
                            y: currentPan.y + (HUD_HEIGHT / 2)
                        });

                        // Force position update after centering
                        setTimeout(updateRootPosition, 50);
                    };

                    // Initial centering
                    centerGraph();

                    // Update fit button to use centering
                    document.getElementById('fit-button').addEventListener('click', centerGraph);

                    // Update zoom level display
                    const zoomDisplay = document.getElementById('zoom-level');
                    cy.on('zoom', _.throttle(() => {
                        zoomDisplay.textContent = `Zoom: ${cy.zoom().toFixed(2)}`;
                    }, 100));

                    // Function to update root node position display
                    const updateRootPosition = _.throttle(() => {
                        const rootNode = cy.nodes()[0];
                        if (rootNode) {
                            const renderedPosition = rootNode.renderedPosition();
                            const zoom = cy.zoom();
                            
                            // Convert rendered position back to model coordinates
                            const modelX = Math.round(renderedPosition.x / zoom);
                            const modelY = Math.round(renderedPosition.y / zoom);
                            
                            document.getElementById('root-position').textContent = 
                                `Root: (${modelX}, ${modelY})`;
                        }
                    }, 16);  // 60fps

                    // Add event listeners
                    cy.on('render', updateRootPosition);
                    cy.on('pan', updateRootPosition);
                    cy.on('zoom', updateRootPosition);
                    cy.on('position', updateRootPosition);
                    cy.on('layoutstop', updateRootPosition);

                    // Also update on viewport changes
                    cy.on('viewport', updateRootPosition);

                    // Make the entire updateRootPosition function available to the cy object
                    cy.updateRootPosition = updateRootPosition;
                });
        </script>
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
