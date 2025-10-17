import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
from sqlalchemy import create_engine

from farm.core.interfaces import DatabaseProtocol
from farm.utils.logging import get_logger

logger = get_logger(__name__)

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
    plot_lifespan_distribution,
    plot_lineage_size,
    plot_reproduction_success_rate,
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
    plot_resource_distribution_entropy,
    plot_resource_efficiency,
    plot_resource_sharing,
    plot_rewards,
)
from .chart_utils import save_plot  # Import from utilities module
from .llm_client import LLMClient
from farm.core.services import EnvConfigService


class ChartAnalyzer:
    def __init__(
        self,
        database: DatabaseProtocol,
        output_dir: Optional[Path] = None,
        save_charts: bool = True,
    ):
        """
        Initialize the chart analyzer.

        Args:
            database: Database instance implementing DatabaseProtocol for data access
            output_dir: Optional directory to save charts and analyses
            save_charts: Whether to save charts to files or keep in memory
        """
        self.db = database
        self.output_dir = output_dir if output_dir else Path("example_output")
        self.save_charts = save_charts
        # Inject config service explicitly (no implicit fallbacks)
        cfg = EnvConfigService()
        self.llm_client = LLMClient(api_key=None, config_service=cfg)

    def analyze_all_charts(self, output_path: Optional[Path] = None, database: Optional[DatabaseProtocol] = None) -> Dict[str, str]:
        """Generate and analyze all charts, returning a dictionary of analyses."""
        analyses = {}

        if output_path:
            self.output_dir = output_path

        # Use provided database or fall back to instance database
        db_to_use = database if database is not None else self.db

        try:
            # Use the database connection directly instead of creating a new one
            simulation_df = pd.read_sql(
                "SELECT * FROM simulation_steps", db_to_use.engine
            )
            actions_df = pd.read_sql("SELECT * FROM agent_actions", db_to_use.engine)
            agents_df = pd.read_sql("SELECT * FROM agents", db_to_use.engine)

            # Get connection string from engine
            connection_string = str(db_to_use.engine.url)

            # Simulation charts
            simulation_chart_functions = {
                "population_dynamics": plot_population_dynamics,
                "births_and_deaths": plot_births_and_deaths,
                "births_and_deaths_by_type": plot_births_and_deaths_by_type,
                "resource_efficiency": plot_resource_efficiency,
                "agent_health_and_age": plot_agent_health_and_age,
                "combat_metrics": plot_combat_metrics,
                "resource_sharing": plot_resource_sharing,
                "evolutionary_metrics": plot_evolutionary_metrics,
                "resource_distribution_entropy": plot_resource_distribution_entropy,
                "rewards": plot_rewards,
                "average_resources": plot_average_resources,
                "agent_lifespan_histogram": plot_agent_lifespan_histogram,
                "agent_type_comparison": plot_agent_type_comparison,
                "reproduction_success_rate": plot_reproduction_success_rate,
                "reproduction_resources": plot_reproduction_resources,
                "generational_analysis": plot_generational_analysis,
                "reproduction_failure_reasons": plot_reproduction_failure_reasons,
            }

            # Update chart functions to use the database connection
            for chart_name, chart_func in simulation_chart_functions.items():
                try:
                    print(f"Generating {chart_name} chart...")
                    # Pass connection string to chart functions that need it
                    if chart_name in [
                        "births_and_deaths_by_type",
                        "agent_lifespan_histogram",
                        "agent_type_comparison",
                        "reproduction_success_rate",
                        "reproduction_resources",
                        "generational_analysis",
                        "reproduction_failure_reasons",
                    ]:
                        plt = chart_func(simulation_df, connection_string)
                    else:
                        plt = chart_func(simulation_df)

                    if plt is not None:
                        if self.save_charts:
                            image_path = save_plot(
                                plt, chart_name, self.output_dir.as_posix()
                            )
                            analysis = self._analyze_simulation_chart(
                                chart_name, simulation_df
                            )
                        else:
                            analysis = self._analyze_simulation_chart(
                                chart_name, simulation_df
                            )
                            plt.close()
                        analyses[chart_name] = analysis
                except Exception as e:
                    logger.error(
                        "chart_generation_error",
                        chart_name=chart_name,
                        error_type=type(e).__name__,
                        error_message=str(e),
                        exc_info=True,
                    )
                    analyses[chart_name] = f"Analysis failed: {str(e)}"

            # Process action charts
            if not actions_df.empty:
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
                        plt = chart_func(actions_df)
                        if plt is not None:
                            if self.save_charts:
                                image_path = save_plot(
                                    plt, chart_name, self.output_dir.as_posix()
                                )
                                analysis = self._analyze_simulation_chart(
                                    chart_name, actions_df
                                )
                            else:
                                analysis = self._analyze_simulation_chart(
                                    chart_name, actions_df
                                )
                                plt.close()
                            analyses[chart_name] = analysis
                    except Exception as e:
                        logger.error(
                            "action_chart_analysis_error",
                            chart_name=chart_name,
                            error_type=type(e).__name__,
                            error_message=str(e),
                            exc_info=True,
                        )
                        analyses[chart_name] = f"Analysis failed: {str(e)}"

            # Process agent charts
            if not agents_df.empty:
                self.llm_client.set_data(agents_df, data_type="agents")
                agent_chart_functions = {
                    "lifespan_distribution": plot_lifespan_distribution,
                    "lineage_size": plot_lineage_size,
                    "agent_types_over_time": plot_agent_types_over_time,
                }

                for chart_name, chart_func in agent_chart_functions.items():
                    try:
                        plt = chart_func(agents_df)
                        if plt is not None:
                            if self.save_charts:
                                image_path = save_plot(
                                    plt, chart_name, self.output_dir.as_posix()
                                )
                                analysis = self._analyze_simulation_chart(
                                    chart_name, agents_df
                                )
                            else:
                                analysis = self._analyze_simulation_chart(
                                    chart_name, agents_df
                                )
                                plt.close()
                            analyses[chart_name] = analysis
                    except Exception as e:
                        logger.error(
                            "agent_chart_analysis_error",
                            chart_name=chart_name,
                            error_type=type(e).__name__,
                            error_message=str(e),
                            exc_info=True,
                        )
                        analyses[chart_name] = f"Analysis failed: {str(e)}"

            # Save analyses to text file if saving is enabled
            if self.save_charts:
                text_path = self.output_dir / "chart_analysis.txt"
                with open(text_path, "w", encoding="utf-8") as f:
                    f.write("SIMULATION ANALYSIS SUMMARY\n\n")
                    for chart_name, analysis in analyses.items():
                        f.write(f"\n{'='*30}\n")
                        f.write(f"{chart_name} Analysis\n\n")
                        f.write(f"{analysis.strip()}")
                        f.write(f"\n{'='*30}\n")

            return analyses

        except Exception as e:
            logger.error(
                "simulation_data_loading_error",
                error_type=type(e).__name__,
                error_message=str(e),
                exc_info=True,
            )
            return analyses

    def _analyze_simulation_chart(self, chart_name: str, df: pd.DataFrame) -> str:
        """Analyze simulation charts based on their type."""
        try:
            if chart_name == "lifespan_distribution":
                return self._analyze_lifespan_distribution(df)
            elif chart_name == "lineage_size":
                return self._analyze_lineage_size(df)
            elif chart_name == "agent_types_over_time":
                return self._analyze_agent_types_over_time(df)
            elif chart_name == "action_type_distribution":
                return self._analyze_action_type_distribution(df)
            elif chart_name == "rewards_by_action_type":
                return self._analyze_rewards_by_action_type(df)
            elif chart_name == "resource_changes":
                return self._analyze_resource_changes(df)
            elif chart_name == "action_frequency_over_time":
                return self._analyze_action_frequency_over_time(df)
            elif chart_name == "rewards_over_time":
                return self._analyze_rewards_over_time(df)
            elif chart_name == "action_target_distribution":
                return self._analyze_action_target_distribution(df)
            elif chart_name == "population_dynamics":
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
            logger.error(
                "chart_analysis_method_error",
                chart_name=chart_name,
                error_type=type(e).__name__,
                error_message=str(e),
                exc_info=True,
            )
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
        """Analyze reproduction success rate metrics."""
        try:
            # Connect to database to get reproduction events data
            engine = create_engine("sqlite:///simulations/simulation.db")
            repro_query = """
            SELECT
                COUNT(*) as total_attempts,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_attempts
            FROM reproduction_events
            """
            repro_data = pd.read_sql(repro_query, engine)

            total_attempts = repro_data["total_attempts"].iloc[0]
            successful_attempts = repro_data["successful_attempts"].iloc[0]
            success_rate = (
                (successful_attempts / total_attempts * 100)
                if total_attempts > 0
                else 0
            )

            return f"""
Reproduction Success Analysis:
- Total reproduction attempts: {total_attempts}
- Successful reproductions: {successful_attempts}
- Success rate: {success_rate:.1f}%
- Population growth: {"Positive" if successful_attempts > 0 else "None"}
"""
        except Exception as e:
            return f"Error analyzing reproduction success rate: {str(e)}"

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

    def _analyze_action_type_distribution(self, df: pd.DataFrame) -> str:
        """Analyze action type distribution."""
        try:
            action_counts = df["action_type"].value_counts()
            total_actions = len(df)
            most_common = action_counts.index[0]

            # Convert the first 3 items to a list before joining
            top_3_actions = [
                f"{action}: {count}" for action, count in action_counts.head(3).items()
            ]

            return f"""
Action Type Analysis:
- Total actions: {total_actions}
- Most common action: {most_common} ({action_counts[most_common]} occurrences)
- Action diversity: {len(action_counts)} different types
- Distribution: {', '.join(top_3_actions)}...
"""
        except Exception as e:
            return f"Error analyzing action type distribution: {str(e)}"

    def _analyze_rewards_by_action_type(self, df: pd.DataFrame) -> str:
        """Analyze rewards by action type."""
        try:
            avg_rewards = df.groupby("action_type")["reward"].agg(["mean", "count"])
            best_action = avg_rewards["mean"].idxmax()

            return f"""
Action Rewards Analysis:
- Most rewarding action: {best_action} (avg: {avg_rewards.loc[best_action, 'mean']:.2f})
- Total actions analyzed: {avg_rewards['count'].sum()}
- Action effectiveness: {len(avg_rewards[avg_rewards['mean'] > 0])} profitable types
"""
        except Exception as e:
            return f"Error analyzing rewards by action type: {str(e)}"

    def _analyze_resource_changes(self, df: pd.DataFrame) -> str:
        """Analyze resource changes from actions."""
        try:
            df["resource_change"] = df["resources_after"] - df["resources_before"]
            avg_change = df["resource_change"].mean()
            positive_changes = (df["resource_change"] > 0).sum()

            return f"""
Resource Change Analysis:
- Average resource change: {avg_change:.2f}
- Positive changes: {positive_changes} ({(positive_changes/len(df)*100):.1f}%)
- Resource efficiency: {"Positive" if avg_change > 0 else "Negative"} net change
"""
        except Exception as e:
            return f"Error analyzing resource changes: {str(e)}"

    def _analyze_action_frequency_over_time(self, df: pd.DataFrame) -> str:
        """Analyze action frequency patterns."""
        try:
            actions_per_step = df.groupby("step_number").size()
            avg_actions = actions_per_step.mean()
            trend = actions_per_step.diff().mean()

            return f"""
Action Frequency Analysis:
- Average actions per step: {avg_actions:.2f}
- Action trend: {"Increasing" if trend > 0 else "Decreasing"}
- Activity level: {"High" if avg_actions > 10 else "Moderate" if avg_actions > 5 else "Low"}
"""
        except Exception as e:
            return f"Error analyzing action frequency: {str(e)}"

    def _analyze_rewards_over_time(self, df: pd.DataFrame) -> str:
        """Analyze reward patterns over time."""
        try:
            rewards_per_step = df.groupby("step_number")["reward"].sum()
            avg_reward = rewards_per_step.mean()
            trend = rewards_per_step.diff().mean()

            return f"""
Rewards Over Time Analysis:
- Average reward per step: {avg_reward:.2f}
- Reward trend: {"Improving" if trend > 0 else "Declining"}
- Performance: {"Effective" if avg_reward > 0 else "Needs improvement"}
"""
        except Exception as e:
            return f"Error analyzing rewards over time: {str(e)}"

    def _analyze_action_target_distribution(self, df: pd.DataFrame) -> str:
        """Analyze action target patterns."""
        try:
            targets = []
            for _, row in df.iterrows():
                if pd.notnull(row["details"]):
                    try:
                        details = json.loads(row["details"])
                        if "target_id" in details:
                            targets.append(details["target_id"])
                        elif "target_position" in details:
                            targets.append("position")
                        else:
                            targets.append("no_target")
                    except json.JSONDecodeError:
                        targets.append("invalid_details")

            target_counts = pd.Series(targets).value_counts()

            return f"""
Action Target Analysis:
- Most common target: {target_counts.index[0]} ({target_counts.iloc[0]} times)
- Target diversity: {len(target_counts)} different targets
- Targeting pattern: {"Focused" if len(target_counts) < 5 else "Diverse"}
"""
        except Exception as e:
            return f"Error analyzing action targets: {str(e)}"

    def _analyze_lifespan_distribution(self, df: pd.DataFrame) -> str:
        """Analyze the distribution of agent lifespans."""
        try:
            df["lifespan"] = df["death_time"] - df["birth_time"]
            avg_lifespan = df["lifespan"].mean()
            max_lifespan = df["lifespan"].max()
            survival_rate = (df["lifespan"] > avg_lifespan).mean() * 100

            return f"""
Lifespan Distribution Analysis:
- Average lifespan: {avg_lifespan:.1f} steps
- Maximum lifespan: {max_lifespan:.1f} steps
- Survival rate: {survival_rate:.1f}% exceed average
- Population health: {"Robust" if survival_rate > 50 else "Struggling"}
"""
        except Exception as e:
            return f"Error analyzing lifespan distribution: {str(e)}"

    def _analyze_lineage_size(self, df: pd.DataFrame) -> str:
        """Analyze the distribution of lineage sizes."""
        try:
            lineage_sizes = df["genome_id"].value_counts()
            avg_lineage = float(lineage_sizes.mean())
            max_lineage = lineage_sizes.max()
            successful_lineages = lineage_sizes.gt(avg_lineage).sum()

            return f"""
Lineage Size Analysis:
- Average lineage size: {avg_lineage:.1f} agents
- Largest lineage: {max_lineage} agents
- Successful lineages: {successful_lineages} above average
- Genetic diversity: {"High" if len(lineage_sizes) > len(df)/10.0 else "Low"}
"""
        except Exception as e:
            return f"Error analyzing lineage sizes: {str(e)}"

    def _analyze_agent_types_over_time(self, df: pd.DataFrame) -> str:
        """Analyze the evolution of agent types over time."""
        try:
            type_counts = (
                df.groupby(["agent_type", "birth_time"]).size().unstack(fill_value=0)
            )
            dominant_type = df["agent_type"].mode().iloc[0]
            type_diversity = len(df["agent_type"].unique())

            return f"""
Agent Type Evolution Analysis:
- Dominant type: {dominant_type}
- Type diversity: {type_diversity} different types
- Population composition: {', '.join(f'{t}: {c}' for t, c in df["agent_type"].value_counts().items())}
- Evolution pattern: {"Diverse" if type_diversity > 2 else "Specialized"}
"""
        except Exception as e:
            return f"Error analyzing agent types over time: {str(e)}"


def main(actions_df=None, agents_df=None):
    """Main function to run chart analysis."""
    # Create a mock database for the analyzer
    from farm.database.database import SimulationDatabase

    db = SimulationDatabase("sqlite:///simulations/simulation.db")
    analyzer = ChartAnalyzer(db)
    analyses = analyzer.analyze_all_charts()

    # Print analyses
    for chart_name, analysis in analyses.items():
        print(f"\n=== {chart_name} Analysis ===")
        print(analysis)
        print("=" * 50)


if __name__ == "__main__":
    import pandas as pd
    from sqlalchemy import create_engine

    connection_string = "sqlite:///simulations/simulation.db"
    engine = create_engine(connection_string)
    actions_df = pd.read_sql("SELECT * FROM agent_actions", engine)
    agents_df = pd.read_sql("SELECT * FROM Agents", engine)
    main(actions_df, agents_df)
