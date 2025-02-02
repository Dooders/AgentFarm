import glob
import os
import tkinter as tk
from tkinter import ttk
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sqlalchemy import func
from PIL import Image, ImageDraw, ImageFont
import io

from farm.database.database import SimulationDatabase
from farm.database.models import Simulation, SimulationStepModel


def find_simulation_databases(base_path: str) -> List[str]:
    """
    Find all SQLite database files in the specified directory.

    Parameters
    ----------
    base_path : str
        Base directory to search for simulation databases

    Returns
    -------
    List[str]
        List of paths to found database files
    """
    # Create directory if it doesn't exist
    os.makedirs(base_path, exist_ok=True)

    # Find all .db files in the directory
    db_pattern = os.path.join(base_path, "*.db")
    db_files = glob.glob(db_pattern)

    if not db_files:
        print(f"Warning: No database files found in {base_path}")

    return sorted(db_files)


def gather_simulation_metadata(db_path: str) -> Dict[str, Any]:
    """
    Gather high-level metadata for a single simulation, including duration,
    configuration, and a summary of final-step metrics.
    """
    # Get a readable name from the database path
    sim_name = os.path.splitext(os.path.basename(db_path))[0]

    db = SimulationDatabase(db_path)
    session = db.Session()

    try:
        # Debug print to check table structure
        step = session.query(SimulationStepModel).first()
        if step:
            print("Available columns:", [column.key for column in SimulationStepModel.__table__.columns])

        # 1) Fetch simulation record
        sim = (
            session.query(Simulation).order_by(Simulation.simulation_id.desc()).first()
        )

        # 2) Calculate simulation duration in steps
        final_step_row = session.query(func.max(SimulationStepModel.step_number)).one()
        final_step = final_step_row[0] if final_step_row[0] is not None else 0

        # 3) Get step-wise population data
        population_data = (
            session.query(SimulationStepModel.total_agents)
            .order_by(SimulationStepModel.step_number)
            .all()
        )
        population_series = pd.Series([p[0] for p in population_data])

        # Calculate population statistics
        median_population = population_series.median()
        mode_population = population_series.mode().iloc[0]
        mean_population = population_series.mean()

        # 4) Calculate across-time averages
        avg_metrics = session.query(
            func.avg(SimulationStepModel.total_agents).label("avg_total_agents"),
            func.avg(SimulationStepModel.births).label("avg_births"),
            func.avg(SimulationStepModel.deaths).label("avg_deaths"),
            func.avg(SimulationStepModel.total_resources).label("avg_total_resources"),
            func.avg(SimulationStepModel.average_agent_health).label("avg_health"),
            func.avg(SimulationStepModel.average_reward).label("avg_reward"),
        ).first()

        # 5) Gather second-to-last step metrics
        final_step_row = session.query(func.max(SimulationStepModel.step_number)).one()
        final_step = final_step_row[0] if final_step_row[0] is not None else 0
        second_to_last_step = final_step - 1 if final_step > 0 else 0

        final_metrics = (
            session.query(SimulationStepModel)
            .filter(SimulationStepModel.step_number == second_to_last_step)
            .first()
        )

        if final_metrics:
            final_snapshot = {
                "final_total_agents": final_metrics.total_agents,
                "final_total_resources": final_metrics.total_resources,
                "final_average_health": final_metrics.average_agent_health,
                "final_average_reward": final_metrics.average_reward,
                "final_births": final_metrics.births,
                "final_deaths": final_metrics.deaths,
            }
        else:
            final_snapshot = {
                "final_total_agents": None,
                "final_total_resources": None,
                "final_average_health": None,
                "final_average_reward": None,
                "final_births": None,
                "final_deaths": None,
            }

        # 6) Extract config parameters
        config = sim.parameters if sim and sim.parameters else {}

        # Add average age calculation
        avg_age_result = session.query(
            func.avg(SimulationStepModel.average_agent_age)
        ).scalar()

        # 7) Build complete metadata dictionary
        return {
            "simulation_name": sim_name,
            "db_path": db_path,
            "simulation_id": sim.simulation_id if sim else None,
            "start_time": str(sim.start_time) if sim else None,
            "end_time": str(sim.end_time) if sim and sim.end_time else None,
            "status": sim.status if sim else None,
            "duration_steps": final_step,
            "config": config,
            # Population statistics
            "median_population": float(median_population),
            "mode_population": float(mode_population),
            "mean_population": float(mean_population),
            # Across-time averages
            "avg_total_agents": float(avg_metrics[0]) if avg_metrics[0] else 0,
            "avg_births": float(avg_metrics[1]) if avg_metrics[1] else 0,
            "avg_deaths": float(avg_metrics[2]) if avg_metrics[2] else 0,
            "avg_total_resources": float(avg_metrics[3]) if avg_metrics[3] else 0,
            "avg_health": float(avg_metrics[4]) if avg_metrics[4] else 0,
            "avg_reward": float(avg_metrics[5]) if avg_metrics[5] else 0,
            "avg_agent_age": float(avg_age_result) if avg_age_result else 0,
            **final_snapshot,
        }
    finally:
        session.close()
        db.close()


def plot_population_dynamics(df: pd.DataFrame, output_dir: str):
    """Plot population dynamics across simulations."""
    plt.figure(figsize=(12, 6))
    x = range(len(df))

    plt.plot(x, df["median_population"], "b--", label="Median Population")
    plt.plot(x, df["mode_population"], "g--", label="Mode Population")
    plt.plot(x, df["final_total_agents"], "r-", label="Final Population")
    plt.title("Population Comparison Across Simulations")
    plt.xlabel("Simulation Index")
    plt.ylabel("Number of Agents")
    plt.legend()

    # Save the figure without the note first
    temp_path = os.path.join(output_dir, 'temp_population_comparison.png')
    plt.savefig(temp_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Add the note using Pillow
    note_text = ('Note: This plot compares different population metrics across simulations. '
                'The median and mode show typical population levels, while the final population '
                'indicates how simulations ended.')
    
    final_path = os.path.join(output_dir, 'population_comparison.png')
    add_note_to_image(temp_path, note_text)
    
    # Clean up temporary file
    os.remove(temp_path)


def plot_resource_utilization(df: pd.DataFrame, output_dir: str):
    """Plot resource utilization across simulations."""
    plt.figure(figsize=(12, 6))
    x = range(len(df))
    plt.plot(x, df["avg_total_resources"], "g-", label="Average Resources")
    plt.plot(x, df["final_total_resources"], "m--", label="Final Resources")
    plt.title("Resource Utilization Across Simulations")
    plt.xlabel("Simulation Index")
    plt.ylabel("Resource Level")
    plt.legend()

    # Save the figure without the note first
    temp_path = os.path.join(output_dir, 'temp_resource_comparison.png')
    plt.savefig(temp_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Add the note using Pillow
    note_text = ('Note: This visualization tracks resource levels across simulations. '
                'Average resources show typical availability, while final resources '
                'indicate end-state resource levels.')
    
    final_path = os.path.join(output_dir, 'resource_comparison.png')
    add_note_to_image(temp_path, note_text)
    
    # Clean up temporary file
    os.remove(temp_path)


def plot_health_reward_distribution(df: pd.DataFrame, output_dir: str):
    """Plot health and reward distributions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    sns.boxplot(data=df[["avg_health", "final_average_health"]], ax=ax1)
    ax1.set_title("Health Distribution")
    ax1.set_ylabel("Health Level")

    sns.boxplot(data=df[["avg_reward", "final_average_reward"]], ax=ax2)
    ax2.set_title("Reward Distribution")
    ax2.set_ylabel("Reward Level")

    plt.tight_layout()
    
    # Save the figure without the note first
    temp_path = os.path.join(output_dir, 'temp_health_reward_distribution.png')
    plt.savefig(temp_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Add the note using Pillow
    note_text = ('Note: These box plots show the distribution of health and reward metrics. '
                'The boxes show the middle 50% of values, while whiskers extend to the full range '
                'excluding outliers.')
    
    final_path = os.path.join(output_dir, 'health_reward_distribution.png')
    add_note_to_image(temp_path, note_text)
    
    # Clean up temporary file
    os.remove(temp_path)


def plot_population_vs_cycles(df: pd.DataFrame, output_dir: str):
    """Plot population vs simulation duration."""
    plt.figure(figsize=(12, 6))
    plt.scatter(
        df["avg_total_agents"], df["duration_steps"], alpha=0.6, c="blue", s=100
    )

    plt.title("Average Population vs Simulation Duration")
    plt.xlabel("Average Population")
    plt.ylabel("Number of Cycles (Steps)")

    z = np.polyfit(df["avg_total_agents"], df["duration_steps"], 1)
    p = np.poly1d(z)
    plt.plot(
        df["avg_total_agents"],
        p(df["avg_total_agents"]),
        "r--",
        alpha=0.8,
        label=f"Trend line (slope: {z[0]:.2f})",
    )

    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save the figure without the note first
    temp_path = os.path.join(output_dir, 'temp_population_vs_cycles.png')
    plt.savefig(temp_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Add the note using Pillow
    note_text = ('Note: This scatter plot reveals the relationship between population size and '
                'simulation duration. The trend line shows the general correlation between '
                'these variables.')
    
    final_path = os.path.join(output_dir, 'population_vs_cycles.png')
    add_note_to_image(temp_path, note_text)
    
    # Clean up temporary file
    os.remove(temp_path)


def plot_population_vs_age(df: pd.DataFrame, output_dir: str):
    """Plot population vs agent age with birth rate color gradient."""
    plt.figure(figsize=(12, 6))

    scatter = plt.scatter(
        df["avg_total_agents"],
        df["avg_agent_age"],
        c=df["avg_births"],
        s=100,
        alpha=0.6,
        cmap="viridis",
    )

    cbar = plt.colorbar(scatter)
    cbar.set_label("Average Birth Rate", rotation=270, labelpad=15)

    correlation = df["avg_total_agents"].corr(df["avg_agent_age"])
    plt.title(f"Average Population vs Average Agent Age (r = {correlation:.2f})")
    plt.xlabel("Average Population")
    plt.ylabel("Average Agent Age")

    z = np.polyfit(df["avg_total_agents"], df["avg_agent_age"], 1)
    p = np.poly1d(z)
    plt.plot(
        df["avg_total_agents"],
        p(df["avg_total_agents"]),
        "r--",
        alpha=0.8,
        label=f"Trend line (slope: {z[0]:.2f})",
    )

    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save the figure without the note first
    temp_path = os.path.join(output_dir, 'temp_population_vs_age.png')
    plt.savefig(temp_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Add the note using Pillow
    note_text = ('Note: This scatter plot shows how population size relates to average agent age. '
                'The color gradient indicates birth rates, while the trend line shows the overall '
                'relationship between population and age.')
    
    final_path = os.path.join(output_dir, 'population_vs_age.png')
    add_note_to_image(temp_path, note_text)
    
    # Clean up temporary file
    os.remove(temp_path)


def plot_population_trends_across_simulations(df: pd.DataFrame, output_dir: str):
    """
    Create a visualization showing population trends across all simulations.
    """
    # Create figure with two subplots sharing x axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), height_ratios=[1, 2])
    fig.suptitle('Population Trends Across All Simulations', fontsize=14, y=0.95)
    
    # Get all simulation databases and collect step-wise data
    all_populations = []
    max_steps = 0
    
    for db_path in df['db_path']:
        db = SimulationDatabase(db_path)
        session = db.Session()
        
        try:
            population_data = (
                session.query(SimulationStepModel.step_number, SimulationStepModel.total_agents)
                .order_by(SimulationStepModel.step_number)
                .all()
            )
            
            steps = np.array([p[0] for p in population_data])
            pop = np.array([p[1] for p in population_data])
            
            max_steps = max(max_steps, len(steps))
            all_populations.append(pop)
            
        finally:
            session.close()
            db.close()
    
    # Pad shorter simulations with NaN values
    padded_populations = []
    for pop in all_populations:
        if len(pop) < max_steps:
            padded = np.pad(
                pop,
                (0, max_steps - len(pop)),
                mode='constant',
                constant_values=np.nan
            )
        else:
            padded = pop
        padded_populations.append(padded)
    
    population_array = np.array(padded_populations)
    
    # Calculate statistics
    mean_pop = np.nanmean(population_array, axis=0)
    median_pop = np.nanmedian(population_array, axis=0)
    std_pop = np.nanstd(population_array, axis=0)
    confidence_interval = 1.96 * std_pop / np.sqrt(len(all_populations))
    
    steps = np.arange(max_steps)
    
    # Plot full range in top subplot (linear scale)
    ax1.plot(steps, mean_pop, 'b-', label='Mean Population', linewidth=2)
    ax1.plot(steps, median_pop, 'g--', label='Median Population', linewidth=2)
    ax1.fill_between(
        steps,
        mean_pop - confidence_interval,
        mean_pop + confidence_interval,
        color='b',
        alpha=0.2,
        label='95% Confidence Interval'
    )
    
    # Customize top subplot
    ax1.set_ylabel('Number of Agents', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Add text to explain the views
    ax1.text(0.02, 0.98, 'Full Range View', 
             transform=ax1.transAxes, fontsize=10, 
             verticalalignment='top')
    
    # Plot steady-state detail in bottom subplot (linear scale)
    # Find where the initial spike settles (simple heuristic)
    settling_point = 55  # Skip first 100 steps to avoid initial spike
    
    ax2.plot(steps[settling_point:], mean_pop[settling_point:], 'b-', 
            label='Mean Population', linewidth=2)
    ax2.plot(steps[settling_point:], median_pop[settling_point:], 'g--', 
            label='Median Population', linewidth=2)
    ax2.fill_between(
        steps[settling_point:],
        mean_pop[settling_point:] - confidence_interval[settling_point:],
        mean_pop[settling_point:] + confidence_interval[settling_point:],
        color='b',
        alpha=0.2,
        label='95% Confidence Interval'
    )
    
    # Customize bottom subplot
    steady_state_min = np.nanmin(mean_pop[settling_point:] - confidence_interval[settling_point:])
    steady_state_max = np.nanmax(mean_pop[settling_point:] + confidence_interval[settling_point:])
    padding = (steady_state_max - steady_state_min) * 0.1
    ax2.set_ylim(steady_state_min - padding, steady_state_max + padding)
    ax2.set_xlabel('Simulation Step', fontsize=12)
    ax2.set_ylabel('Number of Agents', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    # Add text to explain the views
    ax2.text(0.02, 0.98, 'Steady State Detail', 
             transform=ax2.transAxes, fontsize=10, 
             verticalalignment='top')
    
    # Save the figure without the note first
    temp_path = os.path.join(output_dir, 'temp_population_trends.png')
    plt.savefig(temp_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Add the note using Pillow
    note_text = ('Note: This visualization shows population changes over time across all simulations. '
                'The top panel displays the full range including initial population spikes, while '
                'the bottom panel focuses on the steady-state behavior after initial fluctuations settle.')
    
    final_path = os.path.join(output_dir, 'population_trends.png')
    add_note_to_image(temp_path, note_text)
    
    # Clean up temporary file
    os.remove(temp_path)


def plot_population_trends_by_type_across_simulations(df: pd.DataFrame, output_dir: str):
    """
    Create a visualization showing population trends by agent type across all simulations.
    """
    print(f"Starting plot_population_trends_by_type_across_simulations with output_dir: {output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with four subplots: one large at top, three below
    fig = plt.figure(figsize=(15, 20))
    gs = plt.GridSpec(4, 1, height_ratios=[1, 1, 1, 1], hspace=0.3)
    
    ax_full = fig.add_subplot(gs[0])  # Top subplot for full view
    ax1 = fig.add_subplot(gs[1])      # System agents
    ax2 = fig.add_subplot(gs[2])      # Independent agents
    ax3 = fig.add_subplot(gs[3])      # Control agents
    
    fig.suptitle('Population Trends by Agent Type Across All Simulations', fontsize=14, y=0.95)
    
    # Get all simulation databases and collect step-wise data
    agent_types = {
        'System Agents': [],
        'Independent Agents': [],
        'Control Agents': []
    }
    max_steps = 0
    
    for db_path in df['db_path']:
        db = SimulationDatabase(db_path)
        session = db.Session()
        
        try:
            population_data = (
                session.query(
                    SimulationStepModel.step_number,
                    SimulationStepModel.system_agents,
                    SimulationStepModel.independent_agents,
                    SimulationStepModel.control_agents
                )
                .order_by(SimulationStepModel.step_number)
                .all()
            )
            
            steps = np.array([p[0] for p in population_data])
            system_pop = np.array([p[1] if p[1] is not None else 0 for p in population_data])
            independent_pop = np.array([p[2] if p[2] is not None else 0 for p in population_data])
            control_pop = np.array([p[3] if p[3] is not None else 0 for p in population_data])
            
            max_steps = max(max_steps, len(steps))
            agent_types['System Agents'].append(system_pop)
            agent_types['Independent Agents'].append(independent_pop)
            agent_types['Control Agents'].append(control_pop)
            
        finally:
            session.close()
            db.close()
    
    # Pad shorter simulations with NaN values for each agent type
    padded_populations = {agent_type: [] for agent_type in agent_types.keys()}
    for agent_type, populations in agent_types.items():
        for pop in populations:
            if len(pop) < max_steps:
                padded = np.pad(
                    pop,
                    (0, max_steps - len(pop)),
                    mode='constant',
                    constant_values=np.nan
                )
            else:
                padded = pop
            padded_populations[agent_type].append(padded)
    
    # Convert to numpy arrays and calculate statistics
    population_arrays = {
        agent_type: np.array(pops) 
        for agent_type, pops in padded_populations.items()
    }
    
    steps = np.arange(max_steps)
    colors = {
        'System Agents': 'blue',
        'Independent Agents': 'green',
        'Control Agents': 'red'
    }
    axes = {
        'System Agents': ax1,
        'Independent Agents': ax2,
        'Control Agents': ax3
    }
    
    # Plot full range view at the top
    for agent_type, pop_array in population_arrays.items():
        color = colors[agent_type]
        mean_pop = np.nanmean(pop_array, axis=0)
        median_pop = np.nanmedian(pop_array, axis=0)
        std_pop = np.nanstd(pop_array, axis=0)
        confidence_interval = 1.96 * std_pop / np.sqrt(len(pop_array))
        
        # Plot mean line for full view
        ax_full.plot(steps, mean_pop, color=color, linestyle='-', 
                    label=f'{agent_type} (Mean)', linewidth=2)
        ax_full.fill_between(
            steps,
            mean_pop - confidence_interval,
            mean_pop + confidence_interval,
            color=color,
            alpha=0.2
        )
    
    # Customize full view subplot
    ax_full.set_ylabel('Number of Agents', fontsize=12)
    ax_full.grid(True, alpha=0.3)
    ax_full.legend(fontsize=10)
    ax_full.set_title('Full Range View - All Agent Types', fontsize=12, pad=10)
    
    # Plot stable view for each agent type
    settling_point = 55  # Skip first 55 steps to avoid initial spike
    axes = {
        'System Agents': ax1,
        'Independent Agents': ax2,
        'Control Agents': ax3
    }
    
    for agent_type, pop_array in population_arrays.items():
        ax = axes[agent_type]
        color = colors[agent_type]
        
        mean_pop = np.nanmean(pop_array, axis=0)
        median_pop = np.nanmedian(pop_array, axis=0)
        std_pop = np.nanstd(pop_array, axis=0)
        confidence_interval = 1.96 * std_pop / np.sqrt(len(pop_array))
        
        # Plot stable period only
        ax.plot(steps[settling_point:], mean_pop[settling_point:], 
               color=color, linestyle='-', label='Mean Population', linewidth=2)
        ax.plot(steps[settling_point:], median_pop[settling_point:], 
               color=color, linestyle='--', label='Median Population', linewidth=2)
        
        # Add confidence interval
        ax.fill_between(
            steps[settling_point:],
            mean_pop[settling_point:] - confidence_interval[settling_point:],
            mean_pop[settling_point:] + confidence_interval[settling_point:],
            color=color,
            alpha=0.2,
            label='95% Confidence Interval'
        )
        
        # Customize subplot
        ax.set_ylabel('Number of Agents', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        ax.set_title(f'{agent_type} Population Trends (Stable Period)', fontsize=12, pad=10)
    
    # Set common x-axis label for bottom subplot only
    ax3.set_xlabel('Simulation Step', fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure without the note first
    temp_path = os.path.join(output_dir, 'temp_population_trends_by_type.png')
    final_path = os.path.join(output_dir, 'population_trends_by_type.png')
    
    # Ensure we're using forward slashes for consistency
    temp_path = os.path.normpath(temp_path)
    final_path = os.path.normpath(final_path)
    
    print(f"Saving temporary file to: {temp_path}")
    try:
        plt.savefig(temp_path, dpi=300, bbox_inches='tight')
        print(f"Temporary file saved successfully")
        print(f"File size: {os.path.getsize(temp_path)} bytes")
    except Exception as e:
        print(f"Error saving temporary file: {str(e)}")
        return
    finally:
        plt.close()
    
    # Add the note using Pillow
    note_text = ('Note: This visualization shows population changes over time for each agent type '
                'across all simulations. The top panel shows the full range for all agent types, '
                'while the lower panels show detailed stable-period trends for each type. '
                'The lines show mean and median populations, while the shaded areas represent '
                '95% confidence intervals.')
    
    print(f"Adding note and saving final file to: {final_path}")
    add_note_to_image(temp_path, note_text)
    
    # Clean up temporary file
    try:
        if os.path.exists(temp_path):
            os.remove(temp_path)
            print(f"Temporary file removed: {temp_path}")
    except Exception as e:
        print(f"Error removing temporary file: {str(e)}")
    
    if os.path.exists(final_path):
        print(f"Final image saved successfully to: {final_path}")
        print(f"File size: {os.path.getsize(final_path)} bytes")
    else:
        print(f"Warning: Final image not found at: {final_path}")


def add_note_to_image(figure_path, note_text):
    """
    Add a note below the matplotlib figure using Pillow.
    """
    try:
        print(f"Opening image from: {figure_path}")
        # Open the original figure
        with Image.open(figure_path) as img:
            print(f"Original image size: {img.size}")
            
            # Try to load Arial font, fall back to default if not available
            try:
                font = ImageFont.truetype("arial.ttf", 14)
                print("Using Arial font")
            except OSError:
                font = ImageFont.load_default()
                print("Using default font")
            
            # Calculate wrapped text dimensions
            margin = 120
            max_text_width = img.width - (2 * margin)
            
            # Wrap text and calculate required height
            words = note_text.split()
            lines = []
            current_line = []
            current_width = 0
            
            for word in words:
                word_width = font.getlength(word + " ")
                if current_width + word_width <= max_text_width:
                    current_line.append(word)
                    current_width += word_width
                else:
                    lines.append(" ".join(current_line))
                    current_line = [word]
                    current_width = word_width
            
            if current_line:
                lines.append(" ".join(current_line))
            
            # Calculate required height for text
            line_height = 24
            text_height = len(lines) * line_height
            note_height = text_height + (3 * margin)
            
            print(f"Creating new image with dimensions: {img.width}x{img.height + note_height}")
            # Create new image with space for text
            new_img = Image.new('RGB', (img.width, img.height + note_height), 'white')
            new_img.paste(img, (0, 0))
            
            # Draw a light gray line to separate the note from the plot
            draw = ImageDraw.Draw(new_img)
            line_y = img.height + margin
            draw.line([(margin, line_y), (img.width - margin, line_y)], 
                     fill='#CCCCCC', width=2)
            
            # Add the text lines
            y = img.height + (margin * 1.5)
            
            for line in lines:
                # Center each line
                text_bbox = draw.textbbox((0, 0), line, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                x = (img.width - text_width) // 2
                
                draw.text((x, y), line, fill='black', font=font)
                y += line_height
            
            # Save the combined image to a new file to avoid any file locking issues
            final_path = figure_path.replace('temp_', '')
            print(f"Saving annotated image to: {final_path}")
            new_img.save(final_path, format='PNG')
            print(f"Image saved successfully")
            
            # Verify the file was created
            if os.path.exists(final_path):
                print(f"Verified: File exists at {final_path}")
                print(f"File size: {os.path.getsize(final_path)} bytes")
            else:
                print(f"Warning: File not found at {final_path}")
                
    except Exception as e:
        print(f"Error in add_note_to_image: {str(e)}")
        import traceback
        traceback.print_exc()


def plot_population_candlestick(df: pd.DataFrame, output_dir: str):
    """
    Create a candlestick-style visualization showing population trends across simulations.
    """
    # Create figure with two subplots sharing x axis
    fig = plt.figure(figsize=(15, 12))
    gs = plt.GridSpec(3, 2, figure=fig)
    
    # Get all simulation databases and collect step-wise data
    all_populations = []
    max_steps = 0
    
    for db_path in df['db_path']:
        db = SimulationDatabase(db_path)
        session = db.Session()
        
        try:
            population_data = (
                session.query(SimulationStepModel.step_number, SimulationStepModel.total_agents)
                .order_by(SimulationStepModel.step_number)
                .all()
            )
            
            steps = np.array([p[0] for p in population_data])
            pop = np.array([p[1] for p in population_data])
            
            max_steps = max(max_steps, len(steps))
            all_populations.append(pop)
            
        finally:
            session.close()
            db.close()
    
    # Pad shorter simulations with NaN values
    padded_populations = []
    for pop in all_populations:
        if len(pop) < max_steps:
            padded = np.pad(
                pop,
                (0, max_steps - len(pop)),
                mode='constant',
                constant_values=np.nan
            )
        else:
            padded = pop
        padded_populations.append(padded)
    
    population_array = np.array(padded_populations)
    
    # Calculate statistics for each time step
    steps = np.arange(max_steps)
    percentile_25 = np.nanpercentile(population_array, 25, axis=0)
    percentile_75 = np.nanpercentile(population_array, 75, axis=0)
    median_pop = np.nanmedian(population_array, axis=0)
    min_pop = np.nanmin(population_array, axis=0)
    max_pop = np.nanmax(population_array, axis=0)
    
    # Plot full range in top subplot
    # Plot min-max range
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.fill_between(steps, min_pop, max_pop, alpha=0.2, color='gray', label='Min-Max Range')
    # Plot interquartile range
    ax1.fill_between(steps, percentile_25, percentile_75, alpha=0.4, color='blue', label='Interquartile Range')
    # Plot median line
    ax1.plot(steps, median_pop, 'r-', label='Median', linewidth=1.5)
    
    # Customize top subplot
    ax1.set_ylabel('Number of Agents', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.text(0.02, 0.98, 'Full Range Distribution', 
             transform=ax1.transAxes, fontsize=10, 
             verticalalignment='top')
    
    # Plot steady-state detail in bottom subplot
    settling_point = 55
    
    # Plot min-max range for steady state
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.fill_between(steps[settling_point:], 
                    min_pop[settling_point:], 
                    max_pop[settling_point:], 
                    alpha=0.2, color='gray', label='Min-Max Range')
    # Plot interquartile range
    ax2.fill_between(steps[settling_point:], 
                    percentile_25[settling_point:], 
                    percentile_75[settling_point:], 
                    alpha=0.4, color='blue', label='Interquartile Range')
    # Plot median line
    ax2.plot(steps[settling_point:], median_pop[settling_point:], 
             'r-', label='Median', linewidth=1.5)
    
    # Customize bottom subplot
    ax2.set_ylim(np.nanmin(min_pop[settling_point:]), np.nanmax(max_pop[settling_point:]))
    ax2.set_xlabel('Simulation Step', fontsize=12)
    ax2.set_ylabel('Number of Agents', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.text(0.02, 0.98, 'Steady State Distribution', 
             transform=ax2.transAxes, fontsize=10, 
             verticalalignment='top')
    
    # Save the figure without the note first
    temp_path = os.path.join(output_dir, 'temp_population_distribution.png')
    plt.savefig(temp_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Add the note using Pillow
    note_text = ('Note: This chart shows the spread of population sizes across simulations. '
                'The gray area shows the full range (min to max), while the blue area shows where '
                'the middle 50% of populations fall. The red line tracks the median population.')
    
    final_path = os.path.join(output_dir, 'population_distribution.png')
    add_note_to_image(temp_path, note_text)
    
    # Clean up temporary file
    os.remove(temp_path)


def plot_comparative_metrics(df: pd.DataFrame, output_dir: str = "simulations/analysis"):
    """Generate comparative visualizations of simulation metrics."""
    print(f"Creating output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    plot_population_dynamics(df, output_dir)
    plot_resource_utilization(df, output_dir)
    plot_health_reward_distribution(df, output_dir)
    plot_population_vs_cycles(df, output_dir)
    plot_population_vs_age(df, output_dir)
    plot_population_trends_across_simulations(df, output_dir)
    plot_population_trends_by_type_across_simulations(df, output_dir)
    plot_population_candlestick(df, output_dir)


def compare_simulations(db_paths: List[str]) -> pd.DataFrame:
    """Compare N simulations by collecting key metadata and stats into a pandas DataFrame."""
    records = []
    for path in db_paths:
        meta = gather_simulation_metadata(path)
        records.append(meta)

    df = pd.DataFrame(records)

    # Generate summary statistics
    print("\nComparative Simulation Summary:")
    print("\nBasic Statistics:")
    print(df[["duration_steps", "avg_total_agents", "avg_total_resources"]].describe())

    print("\nConfiguration Differences:")
    if "config" in df.columns:
        configs = pd.json_normalize(df["config"].apply(lambda x: x or {}))
        if not configs.empty:
            print(configs.describe())

    # Generate plots
    plot_comparative_metrics(df)

    # Launch interactive analysis window
    # create_interactive_analysis_window(df)

    return df


def create_dashboard_summary(
    df: pd.DataFrame, output_dir: str = "simulations/analysis"
):
    """
    Create a dashboard-style summary image highlighting key metrics across simulations.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing simulation metrics
    output_dir : str
        Directory to save the dashboard image
    """
    # Use a built-in style instead of 'seaborn'
    plt.style.use("fivethirtyeight")  # Alternative: 'ggplot'

    # Create figure with subplots in a 3x2 grid
    fig = plt.figure(figsize=(20, 15))
    gs = plt.GridSpec(3, 2, figure=fig)

    # 1. Population Metrics (Top Left)
    ax1 = fig.add_subplot(gs[0, 0])
    metrics = {
        "Avg Population": df["avg_total_agents"].mean(),
        "Peak Population": df["final_total_agents"].max(),
        "Avg Birth Rate": df["avg_births"].mean(),
        "Avg Death Rate": df["avg_deaths"].mean(),
    }

    colors = ["#2ecc71", "#3498db", "#e74c3c", "#f1c40f"]
    ax1.bar(range(len(metrics)), metrics.values(), color=colors)
    ax1.set_xticks(range(len(metrics)))
    ax1.set_xticklabels(metrics.keys(), rotation=45)
    ax1.set_title("Population Metrics", fontsize=12, pad=20)

    # Add value labels on bars
    for i, v in enumerate(metrics.values()):
        ax1.text(i, v, f"{v:.1f}", ha="center", va="bottom")

    # 2. Resource Metrics (Top Right)
    ax2 = fig.add_subplot(gs[0, 1])
    resource_data = {
        "Avg Resources": df["avg_total_resources"].mean(),
        "Final Resources": df["final_total_resources"].mean(),
    }

    ax2.pie(
        resource_data.values(),
        labels=resource_data.keys(),
        autopct="%1.1f%%",
        colors=["#2ecc71", "#e74c3c"],
    )
    ax2.set_title("Resource Distribution", fontsize=12, pad=20)

    # 3. Health & Reward Trends (Middle)
    ax3 = fig.add_subplot(gs[1, :])

    # Use simulation names for x-axis if available
    if "simulation_name" in df.columns:
        x = df["simulation_name"]
    else:
        x = range(len(df))

    ax3.plot(
        x, df["final_average_health"], "r-", label="Pre-Final Agent Health", linewidth=2
    )
    ax3.plot(x, df["avg_health"], "g-", label="Average Agent Health", linewidth=2)

    ax3.set_title(
        "Health Metrics Across Simulations (Pre-Final Step)", fontsize=12, pad=20
    )
    ax3.set_xlabel("Simulation Name")
    ax3.set_ylabel("Health")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Rotate x-axis labels if they're text
    if "simulation_name" in df.columns:
        plt.xticks(rotation=45, ha="right")

    # 4. Key Statistics Table (Bottom)
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis("tight")
    ax4.axis("off")

    stats_data = [
        ["Metric", "Mean", "Min", "Max", "Std Dev"],
        [
            "Duration (steps)",
            f"{df['duration_steps'].mean():.1f}",
            f"{df['duration_steps'].min():.1f}",
            f"{df['duration_steps'].max():.1f}",
            f"{df['duration_steps'].std():.1f}",
        ],
        [
            "Total Agents",
            f"{df['avg_total_agents'].mean():.1f}",
            f"{df['avg_total_agents'].min():.1f}",
            f"{df['avg_total_agents'].max():.1f}",
            f"{df['avg_total_agents'].std():.1f}",
        ],
        [
            "Final Health",
            f"{df['final_average_health'].mean():.2f}",
            f"{df['final_average_health'].min():.2f}",
            f"{df['final_average_health'].max():.2f}",
            f"{df['final_average_health'].std():.2f}",
        ],
        [
            "Average Health",
            f"{df['avg_health'].mean():.2f}",
            f"{df['avg_health'].min():.2f}",
            f"{df['avg_health'].max():.2f}",
            f"{df['avg_health'].std():.2f}",
        ],
    ]

    table = ax4.table(cellText=stats_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)

    # Style header row
    for i in range(len(stats_data[0])):
        table[(0, i)].set_facecolor("#3498db")
        table[(0, i)].set_text_props(color="white")

    plt.suptitle("Simulation Analysis Dashboard", fontsize=16, y=0.95)
    plt.tight_layout()

    # Save the dashboard
    plt.savefig(
        os.path.join(output_dir, "simulation_dashboard.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def create_interactive_analysis_window(df: pd.DataFrame):
    """Create an interactive window for population vs age analysis."""
    import tkinter as tk
    from tkinter import ttk

    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    # Create the main window
    root = tk.Tk()
    root.title("Interactive Population Analysis")
    root.geometry("1000x800")

    # Create frame for controls
    control_frame = ttk.Frame(root, padding="5")
    control_frame.pack(fill=tk.X)

    # Create frame for the plot
    plot_frame = ttk.Frame(root, padding="5")
    plot_frame.pack(fill=tk.BOTH, expand=True)

    # Variables for plot customization
    color_var = tk.StringVar(value="avg_births")
    x_var = tk.StringVar(value="avg_total_agents")
    y_var = tk.StringVar(value="avg_agent_age")

    available_metrics = [
        "avg_births",
        "avg_deaths",
        "avg_health",
        "avg_reward",
        "avg_total_resources",
        "median_population",
        "mode_population",
        "avg_total_agents",
        "avg_agent_age",
        "duration_steps",
    ]

    # Create the figure and canvas
    fig, ax = plt.subplots(figsize=(10, 6))
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_plot():
        """Update the plot based on current settings."""
        # Clear the entire figure
        fig.clear()
        ax = fig.add_subplot(111)

        # Create scatter plot with selected metrics
        scatter = ax.scatter(
            df[x_var.get()],
            df[y_var.get()],
            c=df[color_var.get()],
            s=100,
            alpha=0.6,
            cmap="viridis",
        )

        # Add colorbar
        fig.colorbar(scatter, ax=ax, label=color_var.get())

        # Calculate correlation
        correlation = df[x_var.get()].corr(df[y_var.get()])

        # Add trend line
        z = np.polyfit(df[x_var.get()], df[y_var.get()], 1)
        p = np.poly1d(z)
        ax.plot(
            df[x_var.get()],
            p(df[x_var.get()]),
            "r--",
            alpha=0.8,
            label=f"Trend line (slope: {z[0]:.2f})",
        )

        # Set labels and title
        ax.set_title(f"{x_var.get()} vs {y_var.get()} (r = {correlation:.2f})")
        ax.set_xlabel(x_var.get().replace("_", " ").title())
        ax.set_ylabel(y_var.get().replace("_", " ").title())
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Refresh canvas
        fig.tight_layout()
        canvas.draw()

    # Create controls
    ttk.Label(control_frame, text="X Axis:").pack(side=tk.LEFT, padx=5)
    x_menu = ttk.Combobox(
        control_frame, textvariable=x_var, values=available_metrics, width=20
    )
    x_menu.pack(side=tk.LEFT, padx=5)

    ttk.Label(control_frame, text="Y Axis:").pack(side=tk.LEFT, padx=5)
    y_menu = ttk.Combobox(
        control_frame, textvariable=y_var, values=available_metrics, width=20
    )
    y_menu.pack(side=tk.LEFT, padx=5)

    ttk.Label(control_frame, text="Color by:").pack(side=tk.LEFT, padx=5)
    color_menu = ttk.Combobox(
        control_frame, textvariable=color_var, values=available_metrics, width=20
    )
    color_menu.pack(side=tk.LEFT, padx=5)

    # Bind events
    x_menu.bind("<<ComboboxSelected>>", lambda e: update_plot())
    y_menu.bind("<<ComboboxSelected>>", lambda e: update_plot())
    color_menu.bind("<<ComboboxSelected>>", lambda e: update_plot())

    # Initial plot
    update_plot()

    # Start the GUI event loop
    root.mainloop()


def main(path: str):
    # Find all simulation databases
    db_paths = find_simulation_databases(path)

    if not db_paths:
        print(f"No simulation databases found in path: {path}")
        return

    print(f"Found {len(db_paths)} database files")
    
    # Compare and analyze simulations
    results_df = compare_simulations(db_paths)

    # Save results to CSV
    output_dir = os.path.join("simulations", "analysis")
    print(f"Using output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    results_df.to_csv(os.path.join(output_dir, "simulation_comparison.csv"), index=False)
    print(f"Results saved to {output_dir}")

    # Create dashboard summary
    create_dashboard_summary(results_df, output_dir)

    # Create interactive analysis window
    # create_interactive_analysis_window(results_df)


if __name__ == "__main__":
    main("experiments/initial_experiments/databases")
