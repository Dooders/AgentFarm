import matplotlib.pyplot as plt
import pandas as pd
from sqlalchemy import create_engine
from farm.charts.chart_analyzer import ChartAnalyzer
from farm.charts.chart_actions import *
from farm.charts.chart_agents import *
from farm.charts.chart_simulation import *

# Define charts list at module level
AVAILABLE_CHARTS = [
    # Simulation charts
    "population_dynamics", "births_and_deaths", "births_and_deaths_by_type",
    "resource_efficiency", "agent_health_and_age", "combat_metrics",
    "resource_sharing", "evolutionary_metrics", "resource_distribution_entropy",
    "rewards", "average_resources", "agent_lifespan_histogram",
    "agent_type_comparison",
    
    # Action charts
    "action_type_distribution", "rewards_by_action_type", "resource_changes",
    "action_frequency_over_time", "rewards_over_time", "action_target_distribution",
    
    # Agent charts
    "lifespan_distribution", "lineage_size", "agent_types_over_time",
    "reproduction_success_rate"
]

def validate_chart(chart_name: str, show_plot: bool = True):
    """
    Validate a specific chart's analysis by showing both the analysis text and the plot.
    
    Args:
        chart_name: Name of the chart to validate
        show_plot: Whether to display the plot (default True)
    """
    # Initialize analyzer without saving files
    analyzer = ChartAnalyzer(save_charts=False)
    
    # Connect to database
    engine = create_engine("sqlite:///simulations/simulation.db")
    simulation_df = pd.read_sql("SELECT * FROM simulation_steps", engine)
    actions_df = pd.read_sql("SELECT * FROM agent_actions", engine)
    agents_df = pd.read_sql("SELECT * FROM Agents", engine)

    # Dictionary mapping chart names to their plotting functions
    chart_functions = {
        # Simulation charts
        "population_dynamics": lambda: plot_population_dynamics(simulation_df),
        "births_and_deaths": lambda: plot_births_and_deaths(simulation_df),
        "births_and_deaths_by_type": lambda: plot_births_and_deaths_by_type(simulation_df, "sqlite:///simulations/simulation.db"),
        "resource_efficiency": lambda: plot_resource_efficiency(simulation_df),
        "agent_health_and_age": lambda: plot_agent_health_and_age(simulation_df),
        "combat_metrics": lambda: plot_combat_metrics(simulation_df),
        "resource_sharing": lambda: plot_resource_sharing(simulation_df),
        "evolutionary_metrics": lambda: plot_evolutionary_metrics(simulation_df),
        "resource_distribution_entropy": lambda: plot_resource_distribution_entropy(simulation_df),
        "rewards": lambda: plot_rewards(simulation_df),
        "average_resources": lambda: plot_average_resources(simulation_df),
        "agent_lifespan_histogram": lambda: plot_agent_lifespan_histogram(simulation_df, "sqlite:///simulations/simulation.db"),
        "agent_type_comparison": lambda: plot_agent_type_comparison(simulation_df, "sqlite:///simulations/simulation.db"),
        
        # Action charts
        "action_type_distribution": lambda: plot_action_type_distribution(actions_df),
        "rewards_by_action_type": lambda: plot_rewards_by_action_type(actions_df),
        "resource_changes": lambda: plot_resource_changes(actions_df),
        "action_frequency_over_time": lambda: plot_action_frequency_over_time(actions_df),
        "rewards_over_time": lambda: plot_rewards_over_time(actions_df),
        "action_target_distribution": lambda: plot_action_target_distribution(actions_df),
        
        # Agent charts
        "lifespan_distribution": lambda: plot_lifespan_distribution(agents_df),
        "lineage_size": lambda: plot_lineage_size(agents_df),
        "agent_types_over_time": lambda: plot_agent_types_over_time(agents_df),
        "reproduction_success_rate": lambda: plot_reproduction_success_rate(simulation_df, "sqlite:///simulations/simulation.db"),
    }

    if chart_name not in chart_functions:
        print(f"Error: Chart '{chart_name}' not found!")
        print("Available charts:")
        for name in chart_functions.keys():
            print(f"- {name}")
        return

    print(f"\n{'='*50}")
    print(f"Validating: {chart_name}")
    print(f"{'='*50}\n")

    # Get and print analysis
    print("ANALYSIS:")
    print("-" * 20)
    analysis = analyzer._analyze_simulation_chart(chart_name, simulation_df)
    print(analysis)
    print("\n")

    # Show plot if requested
    if show_plot:
        print("Generating plot...")
        try:
            plt = chart_functions[chart_name]()
            if plt is not None:
                plt.show()
            else:
                print("Warning: Plot function returned None")
        except Exception as e:
            print(f"Error generating plot: {e}")

def list_available_charts():
    """Print all available charts that can be validated."""
    print("\nAvailable charts to validate:")
    print("-" * 30)
    for i, chart in enumerate(AVAILABLE_CHARTS, 1):
        print(f"{i}. {chart}")

def validate_all_charts(pause_between: bool = True):
    """
    Validate all charts sequentially.
    
    Args:
        pause_between: Whether to pause between charts for user review (default True)
    """
    total_charts = len(AVAILABLE_CHARTS)
    
    for i, chart_name in enumerate(AVAILABLE_CHARTS, 1):
        print(f"\nProcessing chart {i}/{total_charts}: {chart_name}")
        print("=" * 50)
        
        validate_chart(chart_name, show_plot=True)
        
        if pause_between and i < total_charts:
            input("\nPress Enter to continue to next chart...")
            plt.close('all')  # Close all open plots before continuing

def main():
    """Main function to handle script execution."""
    import sys
    
    if len(sys.argv) == 1:
        # No arguments provided - run all charts by default
        print("No arguments provided - running all charts with pauses...")
        validate_all_charts(pause_between=True)
    elif sys.argv[1] == "--list":
        list_available_charts()
    elif sys.argv[1] == "--all":
        pause_between = "--no-pause" not in sys.argv
        validate_all_charts(pause_between=pause_between)
    else:
        validate_chart(sys.argv[1])

if __name__ == "__main__":
    main() 