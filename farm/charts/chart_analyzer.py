import os
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
from sqlalchemy import create_engine

from .chart_actions import (
    plot_action_frequency_over_time,
    plot_action_target_distribution,
    plot_action_type_distribution,
    plot_position_changes,
    plot_resource_changes,
    plot_rewards_by_action_type,
    plot_rewards_over_time,
)
from .chart_agents import (
    plot_agent_types_over_time,
    plot_health_vs_resources,
    plot_lifespan_distribution,
    plot_lineage_size,
    plot_resources_by_generation,
    plot_spatial_distribution,
    plot_starvation_thresholds,
)
from .chart_simulation import (
    plot_agent_health_and_age,
    plot_agent_lifespan_histogram,
    plot_agent_type_comparison,
    plot_average_resources,
    plot_births_and_deaths,
    plot_births_and_deaths_by_type,
    plot_combat_metrics,
    plot_evolutionary_metrics,
    plot_generational_analysis,
    plot_population_dynamics,
    plot_reproduction_failure_reasons,
    plot_reproduction_resources,
    plot_reproduction_success_rate,
    plot_resource_distribution_entropy,
    plot_resource_efficiency,
    plot_resource_sharing,
    plot_rewards,
)
from .llm_client import LLMClient


def save_plot(plt, chart_name, save_to_file=True):
    """Helper function to save plot to file and return path."""
    if save_to_file:
        output_dir = "chart_analysis"
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"{chart_name}.png")
        plt.savefig(file_path)
        plt.close()
        return file_path
    else:
        # Return the current figure for in-memory use
        return plt.gcf()


class ChartAnalyzer:
    def __init__(self, output_dir: str = "example_output", save_charts: bool = True):
        """
        Initialize the chart analyzer.

        Args:
            output_dir: Directory to save charts and analyses
            save_charts: Whether to save charts to files or keep in memory
        """
        self.output_dir = output_dir
        self.save_charts = save_charts
        self.llm_client = LLMClient()
        if save_charts:
            os.makedirs(output_dir, exist_ok=True)

    def analyze_all_charts(self, actions_df=None, agents_df=None) -> Dict[str, str]:
        """Generate and analyze all charts, returning a dictionary of analyses."""
        analyses = {}

        # Load simulation steps data
        try:
            engine = create_engine("sqlite:///simulations/simulation_results.db")
            simulation_df = pd.read_sql("SELECT * FROM simulation_steps", engine)

            # Simulation charts
            simulation_chart_functions = {
                "population_dynamics": plot_population_dynamics,
                "births_and_deaths": plot_births_and_deaths,
                "births_and_deaths_by_type": lambda df: plot_births_and_deaths_by_type(
                    df, "sqlite:///simulations/simulation_results.db"
                ),
                "resource_efficiency": plot_resource_efficiency,
                "agent_health_and_age": plot_agent_health_and_age,
                "combat_metrics": plot_combat_metrics,
                "resource_sharing": plot_resource_sharing,
                "evolutionary_metrics": plot_evolutionary_metrics,
                "resource_distribution_entropy": plot_resource_distribution_entropy,
                "rewards": plot_rewards,
                "average_resources": plot_average_resources,
                "agent_lifespan_histogram": lambda df: plot_agent_lifespan_histogram(
                    df, "sqlite:///simulations/simulation_results.db"
                ),
                "agent_type_comparison": lambda df: plot_agent_type_comparison(
                    df, "sqlite:///simulations/simulation_results.db"
                ),
                "reproduction_success_rate": lambda df: plot_reproduction_success_rate(
                    df, "sqlite:///simulations/simulation_results.db"
                ),
                "reproduction_resources": lambda df: plot_reproduction_resources(
                    df, "sqlite:///simulations/simulation_results.db"
                ),
                "generational_analysis": lambda df: plot_generational_analysis(
                    df, "sqlite:///simulations/simulation_results.db"
                ),
                "reproduction_failure_reasons": lambda df: plot_reproduction_failure_reasons(
                    df, "sqlite:///simulations/simulation_results.db"
                ),
            }

            for chart_name, chart_func in simulation_chart_functions.items():
                try:
                    print(f"Generating {chart_name} chart...")
                    plt.figure()
                    chart_func(simulation_df)
                    if self.save_charts:
                        image_path = save_plot(plt, chart_name)
                    else:
                        plt.close()  # Clean up the figure after analysis

                    analysis = self._analyze_simulation_chart(chart_name, simulation_df)
                    analyses[chart_name] = analysis
                except Exception as e:
                    print(f"Error generating {chart_name} chart: {e}")
                    analyses[chart_name] = f"Analysis failed: {str(e)}"

            # Process action charts if actions_df is provided
            if actions_df is not None:
                self.llm_client.set_data(actions_df, data_type="actions")
                action_chart_functions = {
                    "action_type_distribution": plot_action_type_distribution,
                    "rewards_by_action_type": plot_rewards_by_action_type,
                    "resource_changes": plot_resource_changes,
                    "action_frequency_over_time": plot_action_frequency_over_time,
                    "rewards_over_time": plot_rewards_over_time,
                    "action_target_distribution": plot_action_target_distribution,
                }

                for chart_name, chart_func in action_chart_functions.items():
                    try:
                        if self.save_charts:
                            image_path = chart_func(actions_df)
                            analysis = self.llm_client.analyze_chart(image_path)
                        else:
                            fig = chart_func(actions_df, save_to_file=False)
                            analysis = self.llm_client.analyze_figure(fig)
                            plt.close(fig)
                        analyses[chart_name] = analysis
                    except Exception as e:
                        print(f"Error analyzing {chart_name}: {str(e)}")
                        analyses[chart_name] = f"Analysis failed: {str(e)}"

            # Process agent charts if agents_df is provided
            if agents_df is not None:
                self.llm_client.set_data(agents_df, data_type="agents")
                agent_chart_functions = {
                    "lifespan_distribution": plot_lifespan_distribution,
                    "spatial_distribution": plot_spatial_distribution,
                    "resources_by_generation": plot_resources_by_generation,
                    "starvation_thresholds": plot_starvation_thresholds,
                    "lineage_size": plot_lineage_size,
                    "health_vs_resources": plot_health_vs_resources,
                    "agent_types_over_time": plot_agent_types_over_time,
                }

                for chart_name, chart_func in agent_chart_functions.items():
                    try:
                        image_path = chart_func(agents_df)
                        analysis = self.llm_client.analyze_chart(image_path)
                        analyses[chart_name] = analysis
                    except Exception as e:
                        print(f"Error analyzing {chart_name}: {str(e)}")
                        analyses[chart_name] = f"Analysis failed: {str(e)}"

            # Save analyses to text file if saving is enabled
            if self.save_charts:
                text_path = os.path.join(self.output_dir, "chart_analyses.txt")
                with open(text_path, "w") as f:
                    f.write("SIMULATION ANALYSIS SUMMARY\n\n")
                    for chart_name, analysis in analyses.items():
                        f.write(f"\n{'='*30}\n")
                        f.write(f"{chart_name} Analysis\n\n")
                        f.write(f"{analysis.strip()}")
                        f.write(f"\n{'='*30}\n")

            return analyses

        except Exception as e:
            print(f"Error loading simulation data: {e}")
            return analyses

    def _analyze_simulation_chart(self, chart_name: str, df: pd.DataFrame) -> str:
        """Analyze simulation charts based on their type."""
        try:
            if chart_name == "population_dynamics":
                return self._analyze_population_dynamics(df)
            elif chart_name == "births_and_deaths":
                return self._analyze_births_and_deaths(df)
            elif chart_name == "births_and_deaths_by_type":
                return self._analyze_births_and_deaths_by_type(df)
            elif chart_name == "resource_efficiency":
                return self._analyze_resource_efficiency(df)
            elif chart_name == "agent_health_and_age":
                return self._analyze_agent_health_and_age(df)
            elif chart_name == "combat_metrics":
                return self._analyze_combat_metrics(df)
            elif chart_name == "resource_sharing":
                return self._analyze_resource_sharing(df)
            elif chart_name == "evolutionary_metrics":
                return self._analyze_evolutionary_metrics(df)
            elif chart_name == "resource_distribution_entropy":
                return self._analyze_resource_distribution_entropy(df)
            elif chart_name == "rewards":
                return self._analyze_rewards(df)
            elif chart_name == "average_resources":
                return self._analyze_average_resources(df)
            elif chart_name == "agent_lifespan_histogram":
                return self._analyze_agent_lifespan_histogram(df)
            elif chart_name == "agent_type_comparison":
                return self._analyze_agent_type_comparison(df)
            elif chart_name == "reproduction_success_rate":
                return self._analyze_reproduction_success_rate(df)
            elif chart_name == "reproduction_resources":
                return self._analyze_reproduction_resources(df)
            elif chart_name == "generational_analysis":
                return self._analyze_generational_analysis(df)
            elif chart_name == "reproduction_failure_reasons":
                return self._analyze_reproduction_failure_reasons(df)
            else:
                return f"Analysis not implemented for {chart_name}"
        except Exception as e:
            return f"Error analyzing {chart_name}: {str(e)}"

    def _analyze_population_dynamics(self, df: pd.DataFrame) -> str:
        """Analyze population dynamics chart."""
        latest = df.iloc[-1]
        trend = df["total_agents"].diff().mean()

        return f"""
Population Dynamics Analysis:
- Current population: {latest["total_agents"]} total agents
- Composition: {latest["system_agents"]} system, {latest["independent_agents"]} independent, {latest["control_agents"]} control
- Population trend: {"Growing" if trend > 0 else "Declining"} ({abs(trend):.2f} agents per step)
- Dominant type: {max(("System", latest["system_agents"]), ("Independent", latest["independent_agents"]), ("Control", latest["control_agents"]), key=lambda x: x[1])[0]}
"""

    def _analyze_births_and_deaths(self, df: pd.DataFrame) -> str:
        """Analyze births and deaths chart."""
        total_births = df["births"].sum()
        total_deaths = df["deaths"].sum()
        net_growth = total_births - total_deaths

        return f"""
Population Change Analysis:
- Total births: {total_births}
- Total deaths: {total_deaths}
- Net population growth: {net_growth}
- Birth rate: {total_births/len(df):.2f} per step
- Death rate: {total_deaths/len(df):.2f} per step
- Population sustainability: {"Sustainable" if net_growth >= 0 else "Declining"}
"""

    def _analyze_resource_efficiency(self, df: pd.DataFrame) -> str:
        """Analyze resource efficiency chart."""
        avg_efficiency = df["resource_efficiency"].mean()
        efficiency_trend = df["resource_efficiency"].diff().mean()

        return f"""
Resource Efficiency Analysis:
- Average efficiency: {avg_efficiency:.2f}
- Efficiency trend: {"Improving" if efficiency_trend > 0 else "Declining"} ({abs(efficiency_trend):.3f} per step)
- Total resources: {df["total_resources"].iloc[-1]:.0f}
- Resource stability: {"Stable" if df["total_resources"].std()/df["total_resources"].mean() < 0.1 else "Variable"}
"""

    def _analyze_combat_metrics(self, df: pd.DataFrame) -> str:
        """Analyze combat metrics chart."""
        total_encounters = df["combat_encounters"].sum()
        total_successes = df["successful_attacks"].sum()
        success_rate = (
            (total_successes / total_encounters * 100) if total_encounters > 0 else 0
        )

        return f"""
Combat Analysis:
- Total combat encounters: {total_encounters}
- Successful attacks: {total_successes}
- Success rate: {success_rate:.1f}%
- Combat frequency: {total_encounters/len(df):.2f} encounters per step
"""

    def _analyze_resource_sharing(self, df: pd.DataFrame) -> str:
        """Analyze resource sharing chart."""
        total_shared = df["resources_shared"].sum()
        avg_shared = df["resources_shared"].mean()
        sharing_trend = df["resources_shared"].diff().mean()

        return f"""
Resource Sharing Analysis:
- Total resources shared: {total_shared:.0f}
- Average per step: {avg_shared:.2f}
- Sharing trend: {"Increasing" if sharing_trend > 0 else "Decreasing"}
- Rate of change: {abs(sharing_trend):.3f} per step
"""

    def _analyze_evolutionary_metrics(self, df: pd.DataFrame) -> str:
        """Analyze evolutionary metrics chart."""
        avg_diversity = df["genetic_diversity"].mean()
        diversity_trend = df["genetic_diversity"].diff().mean()
        avg_dominance = df["dominant_genome_ratio"].mean()

        return f"""
Evolutionary Analysis:
- Average genetic diversity: {avg_diversity:.2f}
- Diversity trend: {"Increasing" if diversity_trend > 0 else "Decreasing"}
- Average genome dominance: {avg_dominance:.1f}%
- Population stability: {"Stable" if abs(diversity_trend) < 0.01 else "Evolving"}
"""

    def _analyze_resource_distribution_entropy(self, df: pd.DataFrame) -> str:
        """Analyze resource distribution entropy chart."""
        avg_entropy = df["resource_distribution_entropy"].mean()
        entropy_trend = df["resource_distribution_entropy"].diff().mean()

        return f"""
Resource Distribution Analysis:
- Average entropy: {avg_entropy:.2f}
- Distribution trend: {"More even" if entropy_trend > 0 else "More concentrated"}
- Rate of change: {abs(entropy_trend):.3f} per step
- Distribution type: {"Even" if avg_entropy > 0.7 else "Concentrated"}
"""

    def _analyze_rewards(self, df: pd.DataFrame) -> str:
        """Analyze rewards chart."""
        avg_reward = df["average_reward"].mean()
        reward_trend = df["average_reward"].diff().mean()
        total_reward = df["average_reward"].sum()

        return f"""
Reward Analysis:
- Average reward: {avg_reward:.2f}
- Reward trend: {"Improving" if reward_trend > 0 else "Declining"}
- Total accumulated: {total_reward:.0f}
- Performance: {"Effective" if avg_reward > 0 else "Needs improvement"}
"""

    def _analyze_average_resources(self, df: pd.DataFrame) -> str:
        """Analyze average resources chart."""
        current_resources = df["average_agent_resources"].iloc[-1]
        avg_resources = df["average_agent_resources"].mean()
        resource_trend = df["average_agent_resources"].diff().mean()

        return f"""
Average Resources Analysis:
- Current average: {current_resources:.2f}
- Overall average: {avg_resources:.2f}
- Resource trend: {"Increasing" if resource_trend > 0 else "Decreasing"}
- Rate of change: {abs(resource_trend):.3f} per step
"""

    def _analyze_births_and_deaths_by_type(self, df: pd.DataFrame) -> str:
        """Analyze births and deaths by agent type."""
        return f"""
Population Changes by Type Analysis:
- System agents: {df["system_agents"].iloc[-1]} current ({df["births"].sum()} births, {df["deaths"].sum()} deaths)
- Independent agents: {df["independent_agents"].iloc[-1]} current
- Control agents: {df["control_agents"].iloc[-1]} current
- Most active type: {max(("System", df["system_agents"].mean()), ("Independent", df["independent_agents"].mean()), ("Control", df["control_agents"].mean()), key=lambda x: x[1])[0]}
"""

    def _analyze_agent_health_and_age(self, df: pd.DataFrame) -> str:
        """Analyze agent health and age metrics."""
        return f"""
Health and Age Analysis:
- Average health: {df["average_agent_health"].mean():.2f}
- Health trend: {"Improving" if df["average_agent_health"].diff().mean() > 0 else "Declining"}
- Average age: {df["average_agent_age"].mean():.1f} steps
- Age distribution: {"Young" if df["average_agent_age"].mean() < 50 else "Mature"} population
"""

    def _analyze_agent_lifespan_histogram(self, df: pd.DataFrame) -> str:
        """Analyze agent lifespan distribution."""
        return f"""
Lifespan Distribution Analysis:
- Average lifespan: {df["average_agent_age"].mean():.1f} steps
- Maximum recorded: {df["average_agent_age"].max():.1f} steps
- Population turnover: {"High" if df["deaths"].mean() > 1 else "Low"}
- Survival rate: {"High" if df["deaths"].mean() < df["births"].mean() else "Low"}
"""

    def _analyze_agent_type_comparison(self, df: pd.DataFrame) -> str:
        """Analyze agent type comparison metrics."""
        return f"""
Agent Type Comparison:
- Dominant type: {max(("System", df["system_agents"].mean()), ("Independent", df["independent_agents"].mean()), ("Control", df["control_agents"].mean()), key=lambda x: x[1])[0]}
- Resource efficiency: {df["resource_efficiency"].mean():.2f}
- Health levels: {df["average_agent_health"].mean():.2f}
- Performance metrics: {"Balanced" if df["resource_efficiency"].std() < 0.1 else "Varied"} across types
"""

    def _analyze_reproduction_success_rate(self, df: pd.DataFrame) -> str:
        """Analyze reproduction success rates."""
        return f"""
Reproduction Success Analysis:
- Birth rate: {df["births"].mean():.2f} per step
- Success trend: {"Improving" if df["births"].diff().mean() > 0 else "Declining"}
- Population growth: {"Positive" if df["births"].sum() > df["deaths"].sum() else "Negative"}
- Sustainability: {"Sustainable" if df["births"].mean() > df["deaths"].mean() else "Unsustainable"}
"""

    def _analyze_reproduction_resources(self, df: pd.DataFrame) -> str:
        """Analyze reproduction resource distribution."""
        return f"""
Reproduction Resource Analysis:
- Average resources: {df["average_agent_resources"].mean():.2f}
- Resource efficiency: {df["resource_efficiency"].mean():.2f}
- Distribution: {"Even" if df["resource_distribution_entropy"].mean() > 0.7 else "Uneven"}
- Sustainability: {"Efficient" if df["resource_efficiency"].mean() > 0.5 else "Inefficient"}
"""

    def _analyze_generational_analysis(self, df: pd.DataFrame) -> str:
        """Analyze generational metrics."""
        return f"""
Generational Analysis:
- Current generation: {df["current_max_generation"].max()}
- Average lifespan: {df["average_agent_age"].mean():.1f} steps
- Genetic diversity: {df["genetic_diversity"].mean():.2f}
- Evolution rate: {"Rapid" if df["genetic_diversity"].diff().mean() > 0.01 else "Stable"}
"""

    def _analyze_reproduction_failure_reasons(self, df: pd.DataFrame) -> str:
        """Analyze reproduction failure patterns."""
        return f"""
Reproduction Failure Analysis:
- Success rate: {(df["births"].sum() / (df["births"].sum() + df["deaths"].sum()) * 100):.1f}%
- Main limitation: {"Resources" if df["resource_efficiency"].mean() < 0.5 else "Other factors"}
- Trend: {"Improving" if df["births"].diff().mean() > 0 else "Declining"}
- Sustainability: {"Sustainable" if df["births"].mean() > df["deaths"].mean() else "Unsustainable"}
"""


def main(actions_df=None, agents_df=None):
    """Main function to run chart analysis."""
    analyzer = ChartAnalyzer()
    analyses = analyzer.analyze_all_charts(actions_df, agents_df)

    # Print analyses
    for chart_name, analysis in analyses.items():
        print(f"\n=== {chart_name} Analysis ===")
        print(analysis)
        print("=" * 50)


if __name__ == "__main__":
    import pandas as pd
    from sqlalchemy import create_engine

    connection_string = "sqlite:///simulations/simulation_results.db"
    engine = create_engine(connection_string)
    actions_df = pd.read_sql("SELECT * FROM agent_actions", engine)
    agents_df = pd.read_sql("SELECT * FROM Agents", engine)
    main(actions_df, agents_df)
