#!/usr/bin/env python3
"""
animate_simulation.py

This script creates an animation of the simulation steps, showing how agents and resources
move and interact over time. The output is an MP4 video that can be easily served in a web app.

Usage:
    python animate_simulation.py [iteration_number]
"""

import json
import multiprocessing as mp
import os
import sqlite3
import sys
import tempfile
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle
from moviepy.editor import ImageSequenceClip
from tqdm import tqdm  # Import tqdm for better progress tracking if available


def get_state_at_step(conn, step_number):
    """
    Extract positions of agents and resources at a specific step.
    """
    # Get agents at this step
    agents_query = """
    SELECT a.agent_id, ag.agent_type, a.position_x, a.position_y, a.resource_level as resources
    FROM agent_states a
    JOIN agents ag ON a.agent_id = ag.agent_id
    WHERE a.step_number = ?
    """
    agents = pd.read_sql_query(agents_query, conn, params=(step_number,))

    # Get resources at this step
    resources_query = """
    SELECT resource_id, position_x, position_y, amount
    FROM resource_states
    WHERE step_number = ?
    """
    resources = pd.read_sql_query(resources_query, conn, params=(step_number,))

    return agents, resources


def create_frame(
    agents,
    resources,
    config,
    step_number,
    width=100,
    height=100,
    show_gathering_range=False,
):
    """
    Create a single frame of the animation.

    Args:
        agents: DataFrame containing agent data
        resources: DataFrame containing resource data
        config: Simulation configuration
        step_number: Current simulation step
        width: Width of the simulation area
        height: Height of the simulation area
        show_gathering_range: Whether to show gathering range circles around agents (default: False)
    """
    plt.clf()  # Clear any existing figures

    # Create figure with fixed size and DPI for stability
    fig = plt.figure(figsize=(10, 10), dpi=100)

    # Add padding to prevent cutoff (5% padding on each side)
    padding = 5
    ax = fig.add_subplot(111)

    # Plot resources
    scatter = ax.scatter(
        resources["position_x"],
        resources["position_y"],
        s=resources["amount"] * 5,
        alpha=0.5,
        c="green",
        label="Resources",
    )

    # Plot agents with different colors by type
    agent_colors = {
        "SystemAgent": "blue",
        "IndependentAgent": "red",
        "ControlAgent": "orange",
    }

    # Track when agents first appear
    agent_first_seen = {}

    # Scale for agent size based on resources and birth status
    def get_agent_size(resource_level, agent_id, current_step):
        min_size = 50  # Increased minimum size for better visibility
        max_size = 600  # Maximum size

        # For initial generation (step 0), show normal size
        if current_step == 0:
            normalized_level = max(0, resource_level) / 100
            scaled_size = min_size + (max_size - min_size) * (normalized_level**0.7)
            return scaled_size

        # Track when we first see this agent
        if agent_id not in agent_first_seen:
            agent_first_seen[agent_id] = current_step

        # For newly born agents, scale size based on how many steps they've been alive
        steps_alive = current_step - agent_first_seen[agent_id]
        if steps_alive < 10:  # Growth period of 10 steps
            growth_factor = (steps_alive / 10) ** 0.5  # Smoother growth curve
            normalized_level = max(0, resource_level) / 100
            base_size = min_size + (max_size - min_size) * (normalized_level**0.7)
            start_size = min_size * 1.5  # Start at 1.5x minimum size
            return start_size + (base_size - start_size) * growth_factor

        # Normal size calculation for established agents
        normalized_level = max(0, resource_level) / 100
        scaled_size = min_size + (max_size - min_size) * (normalized_level**0.7)
        return scaled_size

    for agent_type, color in agent_colors.items():
        agent_data = agents[agents["agent_type"] == agent_type]
        if not agent_data.empty:
            # Scale marker size based on resource level and birth status
            sizes = agent_data.apply(
                lambda x: get_agent_size(x["resources"], x["agent_id"], step_number),
                axis=1,
            )

            ax.scatter(
                agent_data["position_x"],
                agent_data["position_y"],
                s=sizes,
                c=color,
                edgecolors="black",
                label=agent_type,
            )

            # Add agent ID labels and resource amounts
            for _, agent in agent_data.iterrows():
                ax.annotate(
                    f"{agent['agent_id'][-4:]}\n({agent['resources']:.1f})",
                    (agent["position_x"], agent["position_y"]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                )

    # Draw gathering range circles around agents if enabled
    if show_gathering_range:
        gathering_range = config.get("gathering_range", 30)
        for _, agent in agents.iterrows():
            circle = Circle(
                (agent["position_x"], agent["position_y"]),
                gathering_range,
                fill=False,
                linestyle="--",
                color=agent_colors[agent["agent_type"]],
                alpha=0.3,
            )
            ax.add_patch(circle)

    # Set plot limits with padding
    ax.set_xlim(-padding, width + padding)
    ax.set_ylim(-padding, height + padding)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")

    # Add step number to title
    ax.set_title(f"Simulation Step {step_number}")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0))

    # Use fixed layout to prevent sliding
    plt.tight_layout(pad=1.5, rect=[0, 0, 1, 1])
    return fig


def create_frame_for_step(
    step, conn_path, config, temp_dir, show_gathering_range=False
):
    """
    Create a frame for a specific step (for parallel processing).
    """
    # Connect to the database (each process needs its own connection)
    conn = sqlite3.connect(conn_path)

    # Get state for this step
    agents, resources = get_state_at_step(conn, step)

    # Create the frame
    fig = create_frame(
        agents,
        resources,
        config,
        step,
        show_gathering_range=show_gathering_range,
    )

    # Save frame
    frame_path = os.path.join(temp_dir, f"frame_{step:04d}.png")
    fig.savefig(frame_path, dpi=100)
    plt.close(fig)

    # Close connection
    conn.close()

    return frame_path


def create_simulation_video(
    iteration,
    experiment_path,
    output_dir="simulation_videos",
    fps=10,
    max_steps=None,
    show_gathering_range=False,
    parallel=True,
    num_processes=None,
    skip_frames=1,  # Process every nth frame
):
    """
    Create a video of the simulation for a specific iteration.

    Args:
        iteration: Iteration number to animate
        experiment_path: Path to experiment directory
        output_dir: Directory to save the video
        fps: Frames per second for the video
        max_steps: Maximum number of steps to animate (None for all steps)
        show_gathering_range: Whether to show gathering range circles around agents (default: False)
        parallel: Whether to use parallel processing (default: True)
        num_processes: Number of processes to use (default: number of CPU cores)
        skip_frames: Process every nth frame to reduce workload (default: 1, process all frames)
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    folder = os.path.join(experiment_path, f"iteration_{iteration}")
    db_path = os.path.join(folder, "simulation.db")
    config_path = os.path.join(folder, "config.json")

    if not os.path.exists(db_path) or not os.path.exists(config_path):
        print(f"Missing files for iteration {iteration}")
        return None

    # Load config
    with open(config_path, "r") as f:
        config = json.load(f)

    # Create temporary directory for frames
    with tempfile.TemporaryDirectory() as temp_dir:
        # Connect to database to get max step
        conn = sqlite3.connect(db_path)

        # Get max step number
        max_step = pd.read_sql_query(
            "SELECT MAX(step_number) as max_step FROM agent_states", conn
        ).iloc[0]["max_step"]

        conn.close()  # Close this connection as we'll use separate ones for parallel processing

        # Limit steps if specified
        if max_steps is not None:
            max_step = min(max_step, max_steps - 1)

        # Apply frame skipping if specified
        steps_to_process = list(range(0, max_step + 1, skip_frames))

        print(f"Creating {len(steps_to_process)} frames...")

        if parallel and len(steps_to_process) > 1:
            # Use multiprocessing to create frames in parallel
            if num_processes is None:
                num_processes = max(1, mp.cpu_count() - 1)  # Leave one CPU free

            print(f"Using {num_processes} processes for parallel frame creation")

            # Create a partial function with fixed arguments
            process_frame = partial(
                create_frame_for_step,
                conn_path=db_path,
                config=config,
                temp_dir=temp_dir,
                show_gathering_range=show_gathering_range,
            )

            # Use a process pool to process frames in parallel
            with mp.Pool(processes=num_processes) as pool:
                # Use tqdm if available for better progress tracking
                try:
                    frame_files = list(
                        tqdm(
                            pool.imap(process_frame, steps_to_process),
                            total=len(steps_to_process),
                        )
                    )
                except NameError:
                    # If tqdm is not available
                    frame_files = pool.map(process_frame, steps_to_process)
        else:
            # Sequential processing
            conn = sqlite3.connect(db_path)
            frame_files = []

            for step in steps_to_process:
                if step % 10 == 0:  # Progress update every 10 steps
                    print(f"Processing step {step}/{max_step}")

                agents, resources = get_state_at_step(conn, step)
                fig = create_frame(
                    agents,
                    resources,
                    config,
                    step,
                    show_gathering_range=show_gathering_range,
                )

                # Save frame
                frame_path = os.path.join(temp_dir, f"frame_{step:04d}.png")
                fig.savefig(frame_path, dpi=100)
                frame_files.append(frame_path)
                plt.close(fig)

            conn.close()

        # Create video from frames
        print("Creating video...")
        frame_list = sorted(frame_files)  # Ensure frames are in order
        clip = ImageSequenceClip([str(f) for f in frame_list], fps=fps)

        # Save as MP4
        output_path = os.path.join(output_dir, f"simulation_{iteration}.mp4")
        clip.write_videofile(
            output_path, codec="libx264", fps=fps, threads=4, verbose=False, logger=None
        )
        print(f"Video saved to: {output_path}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python animate_simulation.py <iteration_number>")
        sys.exit(1)

    iteration = int(sys.argv[1])
    experiment_path = (
        "results/one_of_a_kind/experiments/data/one_of_a_kind_20250302_193353"
    )

    # Create video with optimized parameters
    create_simulation_video(
        iteration,
        experiment_path,
        max_steps=2000,
        show_gathering_range=False,
        parallel=True,  # Enable parallel processing
        num_processes=None,  # Auto-detect CPU cores
        skip_frames=1,  # Process every frame
        fps=15,  # Increase FPS to compensate for skipped frames
    )


if __name__ == "__main__":
    main()
