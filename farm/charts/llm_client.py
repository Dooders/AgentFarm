import base64
import json
import os
from typing import Optional

import numpy as np
import pandas as pd
from openai import OpenAI

from farm.core.services import EnvConfigService, IConfigService
from farm.utils.logging import get_logger

logger = get_logger(__name__)


class LLMClient:
    def __init__(
        self,
        api_key: Optional[str],
        config_service: IConfigService,
    ):
        """Initialize OpenAI client with API key from injected config service."""
        self._config = config_service
        self.api_key = api_key or self._config.get_openai_api_key()
        if not self.api_key:
            raise ValueError(
                "OpenAI API key must be provided via IConfigService or explicitly"
            )
        self.client = OpenAI(api_key=self.api_key)
        self.analyses = {}
        self.actions_df = None
        self.agents_df = None

    def set_data(self, df: pd.DataFrame, data_type: str = "actions"):
        """Set the DataFrame for analysis.

        Args:
            df: DataFrame to analyze
            data_type: Either 'actions' or 'agents'
        """
        if data_type == "actions":
            self.actions_df = df
        elif data_type == "agents":
            self.agents_df = df
        else:
            raise ValueError("data_type must be either 'actions' or 'agents'")

    def analyze_chart(self, image_path: str) -> str:
        """Analyze a chart image and return a detailed description."""
        try:
            chart_name = os.path.basename(image_path).replace(".png", "")

            # Action-based analyses
            action_analyses = {
                "action_type_distribution": self._analyze_action_distribution,
                "rewards_by_action_type": self._analyze_rewards_by_action,
                "resource_changes": self._analyze_resource_changes,
                "action_frequency_over_time": self._analyze_temporal_patterns,
                "rewards_over_time": self._analyze_reward_progression,
                "action_target_distribution": self._analyze_target_distribution,
            }

            # Agent-based analyses
            agent_analyses = {
                "lifespan_distribution": self._analyze_lifespan_distribution,
                "spatial_distribution": self._analyze_spatial_distribution,
                "resources_by_generation": self._analyze_resources_by_generation,
                "starvation_counters": self._analyze_starvation_counters,
                "lineage_size": self._analyze_lineage_size,
                "health_vs_resources": self._analyze_health_vs_resources,
                "agent_types_over_time": self._analyze_agent_types_over_time,
                "reproduction_success_rate": self._analyze_reproduction_success_rate,
            }

            # Determine which analysis to use
            if chart_name in action_analyses:
                if self.actions_df is None:
                    return "Error: Actions data not available"
                analysis = action_analyses[chart_name]()
            elif chart_name in agent_analyses:
                if self.agents_df is None:
                    return "Error: Agents data not available"
                analysis = agent_analyses[chart_name]()
            else:
                return f"Analysis not available for {chart_name}"

            self.analyses[chart_name] = analysis
            output_dir = os.path.dirname(image_path)
            self._save_analyses(output_dir)

            return analysis
        except Exception as e:
            logger.error(
                "chart_analysis_failed",
                error_type=type(e).__name__,
                error_message=str(e),
            )
            return f"Error analyzing chart: {e}"

    def _analyze_action_distribution(self) -> str:
        if self.actions_df is None:
            return "Error: Actions data not available"
        action_counts = self.actions_df["action_type"].value_counts()
        most_common = action_counts.index[0]
        least_common = action_counts.index[-1]

        return f"""
Action Distribution Analysis:
- Most frequent action: {most_common} ({action_counts[most_common]} times, {action_counts[most_common]/len(self.actions_df)*100:.1f}% of total)
- Least frequent action: {least_common} ({action_counts[least_common]} times)
- Action diversity: {len(action_counts)} different types
- Distribution pattern: {'Balanced' if action_counts.std()/action_counts.mean() < 0.5 else 'Highly varied'}"""

    def _analyze_rewards_by_action(self) -> str:
        if self.actions_df is None:
            return "Error: Actions data not available"
        reward_stats = self.actions_df.groupby("action_type")["reward"].agg(
            ["mean", "std", "count"]
        )
        best_action = reward_stats["mean"].idxmax()
        worst_action = reward_stats["mean"].idxmin()

        best_reward = reward_stats.loc[best_action, "mean"]
        worst_reward = reward_stats.loc[worst_action, "mean"]
        effectiveness_ratio = (
            best_reward / worst_reward if worst_reward != 0 else float("inf")  # type: ignore
        )

        return f"""
Reward Analysis by Action:
- Most rewarding action: {best_action} (avg {best_reward:.2f})
- Least rewarding action: {worst_action} (avg {worst_reward:.2f})
- Reward consistency: {'High' if reward_stats['std'].mean() < 1.0 else 'Variable'}
- Key insight: {best_action} is {effectiveness_ratio:.1f}x more effective than {worst_action}"""

    def _analyze_resource_changes(self) -> str:
        if self.actions_df is None:
            return "Error: Actions data not available"
        self.actions_df["resource_change"] = (
            self.actions_df["resources_after"] - self.actions_df["resources_before"]
        )
        total_change = self.actions_df["resource_change"].sum()
        avg_change = self.actions_df["resource_change"].mean()

        return f"""
Resource Impact Analysis:
- Net resource change: {total_change:+.0f} units
- Average change per action: {avg_change:+.2f} units
- Resource efficiency: {'Positive' if total_change > 0 else 'Negative'} net outcome
- Trend: {'Accumulating' if total_change > 0 else 'Depleting'} resources over time"""

    def _analyze_temporal_patterns(self) -> str:
        if self.actions_df is None:
            return "Error: Actions data not available"
        early_actions = self.actions_df[
            self.actions_df["step_number"] <= self.actions_df["step_number"].median()
        ]
        late_actions = self.actions_df[
            self.actions_df["step_number"] > self.actions_df["step_number"].median()
        ]

        early_common = early_actions["action_type"].mode()[0]
        late_common = late_actions["action_type"].mode()[0]

        return f"""
Temporal Pattern Analysis:
- Early phase dominant action: {early_common}
- Late phase dominant action: {late_common}
- Behavior change: {'Significant' if early_common != late_common else 'Consistent'} over time
- Evolution: {'Adapted strategy' if early_common != late_common else 'Maintained strategy'}"""

    def _analyze_reward_progression(self) -> str:
        if self.actions_df is None:
            return "Error: Actions data not available"
        rewards = self.actions_df.groupby("step_number")["reward"].sum()
        trend = np.polyfit(range(len(rewards)), rewards, 1)[0]

        return f"""
Reward Progression Analysis:
- Overall trend: {'Improving' if trend > 0 else 'Declining'}
- Rate of change: {abs(trend):.2f} per step
- Total rewards: {rewards.sum():.0f}
- Performance: {'Learning effective' if trend > 0 else 'Needs optimization'}"""

    def _analyze_target_distribution(self) -> str:
        if self.actions_df is None:
            return "Error: Actions data not available"
        targets = self.actions_df["action_target_id"].value_counts()

        return f"""
Target Selection Analysis:
- Most targeted: ID {targets.index[0]} ({targets.iloc[0]} times)
- Target spread: {len(targets)} unique targets
- Selection pattern: {'Focused' if len(targets) < 5 else 'Diverse'} targeting
- Preference strength: {'Strong' if targets.iloc[0]/targets.sum() > 0.5 else 'Balanced'}"""

    def _analyze_lifespan_distribution(self) -> str:
        if self.agents_df is None:
            return "Error: Agents data not available"
        lifespan = self.agents_df["death_time"] - self.agents_df["birth_time"]
        avg_lifespan = lifespan.mean()
        max_lifespan = lifespan.max()

        skew_value = lifespan.skew()
        skew_direction = "Right-skewed" if skew_value > 0 else "Left-skewed"  # type: ignore
        return f"""
Lifespan Distribution Analysis:
- Average lifespan: {avg_lifespan:.1f} time units
- Maximum lifespan: {max_lifespan:.1f} time units
- Survival rate: {(lifespan > avg_lifespan).mean()*100:.1f}% exceed average
- Distribution: {skew_direction}"""

    def _analyze_spatial_distribution(self) -> str:
        if self.agents_df is None:
            return "Error: Agents data not available"
        x_spread = self.agents_df["position_x"].std()
        y_spread = self.agents_df["position_y"].std()

        return f"""
Spatial Distribution Analysis:
- X-axis spread: {x_spread:.2f} units
- Y-axis spread: {y_spread:.2f} units
- Distribution pattern: {'Clustered' if x_spread + y_spread < 10 else 'Dispersed'}
- Spatial bias: {'Balanced' if abs(x_spread - y_spread) < 1 else 'Directional'}"""

    def _analyze_resources_by_generation(self) -> str:
        try:
            if self.agents_df is None:
                return "Error: Agents data not available"
            gen_stats = self.agents_df.groupby("generation")["initial_resources"].agg(
                ["mean", "std"]
            )

            # Handle potential numpy warnings by checking for valid data
            if len(gen_stats) > 1:  # Need at least 2 generations for trend
                x = np.array(gen_stats.index, dtype=float)
                y = np.array(gen_stats["mean"], dtype=float)
                mask = ~np.isnan(y)  # Remove NaN values
                if np.sum(mask, out=None) > 1:  # Need at least 2 valid points for trend
                    trend = np.polyfit(x[mask], y[mask], 1)[0]
                else:
                    trend = 0
            else:
                trend = 0

            return f"""
Resource Evolution Analysis:
- Resource trend: {'Increasing' if trend > 0 else 'Decreasing'} over generations
- Rate of change: {abs(trend):.2f} per generation
- Resource stability: {'Stable' if gen_stats['std'].mean() < gen_stats['mean'].mean()*0.5 else 'Variable'}
- Generations analyzed: {len(gen_stats)}
- Average resources: {gen_stats['mean'].mean():.1f} units"""
        except Exception as e:
            return f"Error analyzing resource evolution: {str(e)}"

    def _analyze_starvation_counters(self) -> str:
        if self.agents_df is None:
            return "Error: Agents data not available"
        # Check if starvation_counter column exists (it may come from agent_states, not agents table)
        if "starvation_counter" not in self.agents_df.columns:
            return """
Starvation Counter Analysis:
- Data not available from agents table
- Starvation counter tracking has been moved to agent_states table
- Recommendation: Query agent_states table for starvation counter data"""
        
        thresholds = self.agents_df.groupby("agent_type")["starvation_counter"].agg(
            ["mean", "count"]
        )
        if thresholds["mean"].isna().all() or (thresholds["mean"] == 0).all():
            return f"""
Starvation Counter Analysis:
- No meaningful counter variation detected
- Agent types present: {len(thresholds)} types
- Data quality note: All counters are zero or missing
- Recommendation: Check starvation_counter data collection"""

        return f"""
Starvation Counter Analysis:
- Highest counter: {thresholds['mean'].max():.1f} ({thresholds['mean'].idxmax()})
- Lowest counter: {thresholds['mean'].min():.1f} ({thresholds['mean'].idxmin()})
- Counter range: {thresholds['mean'].max() - thresholds['mean'].min():.1f}
- Type distribution: {', '.join(f"{idx}: {val:,.0f}" for idx, val in thresholds['count'].items())}"""

    def _analyze_health_vs_resources(self) -> str:
        try:
            # Check if we have valid data to analyze
            if self.agents_df is None:
                return "Error: Agents data not available"
            health_range = (
                self.agents_df["starting_health"].max()
                - self.agents_df["starting_health"].min()
            )
            resource_range = (
                self.agents_df["initial_resources"].max()
                - self.agents_df["initial_resources"].min()
            )

            # Handle cases where all values are the same
            if health_range == 0 or resource_range == 0:
                return f"""
Health-Resource Relationship:
- Unable to calculate correlation: {'constant health values' if health_range == 0 else 'constant resource values'}
- Health range: {health_range:.1f}
- Resource range: {resource_range:.1f}
- Data quality note: One or more metrics show no variation"""

            # Calculate correlation only if we have valid ranges
            correlation = self.agents_df["starting_health"].corr(
                self.agents_df["initial_resources"]
            )
            if pd.isna(correlation):
                return f"""
Health-Resource Relationship:
- Unable to calculate correlation
- Possible data issues: Missing values or constant values
- Health range: {health_range:.1f}
- Resource range: {resource_range:.1f}
- Data quality note: Check for missing or invalid values"""

            return f"""
Health-Resource Relationship:
- Correlation: {correlation:.2f}
- Relationship: {'Strong' if abs(correlation) > 0.7 else 'Weak'} {'positive' if correlation > 0 else 'negative'}
- Health range: {health_range:.1f}
- Resource range: {resource_range:.1f}
- Sample size: {len(self.agents_df)} agents"""
        except Exception as e:
            return f"Error analyzing health-resource relationship: {str(e)}"

    def _analyze_agent_types_over_time(self) -> str:
        try:
            if self.agents_df is None:
                return "Error: Agents data not available"
            type_counts = self.agents_df.groupby(["agent_type", "birth_time"]).size()
            type_totals = self.agents_df["agent_type"].value_counts()
            most_common = type_totals.index[0]
            type_ratio = type_totals[most_common] / len(self.agents_df) * 100

            return f"""
Population Evolution Analysis:
- Dominant type: {most_common} ({type_ratio:.1f}% of population)
- Type diversity: {len(type_totals)} types
- Population composition: {', '.join(f"{idx}: {val:,.0f}" for idx, val in type_totals.items())}
- Type stability: {'Dynamic' if len(type_counts) > len(self.agents_df['agent_type'].unique()) else 'Static'}"""
        except Exception as e:
            return f"Error analyzing agent types: {str(e)}"

    def _analyze_lineage_size(self) -> str:
        try:
            if self.agents_df is None:
                return "Error: Agents data not available"
            
            # Extract base genome_id (without counter) for lineage grouping
            # Format: parent1:parent2[:counter] -> group by parent1:parent2
            def get_base_genome_id(genome_id: str) -> str:
                """Extract base genome_id without counter."""
                if pd.isna(genome_id) or genome_id == "":
                    return "::"
                parts = str(genome_id).split(":")
                # If has counter (3 parts), return first 2 parts joined
                if len(parts) == 3 and parts[2].isdigit():
                    return f"{parts[0]}:{parts[1]}"
                # Otherwise return as-is (already base or malformed)
                return ":".join(parts[:2]) if len(parts) >= 2 else str(genome_id)
            
            self.agents_df["base_genome_id"] = self.agents_df["genome_id"].apply(get_base_genome_id)
            lineage_sizes = self.agents_df["base_genome_id"].value_counts()
            successful_lineages = (lineage_sizes > 1).sum()

            return f"""
Lineage Analysis:
- Largest lineage: {lineage_sizes.max()} descendants
- Average lineage: {lineage_sizes.mean():.1f} descendants
- Successful genomes: {successful_lineages} ({successful_lineages/len(lineage_sizes)*100:.1f}% of total)
- Lineage distribution: Single: {(lineage_sizes == 1).sum()}, 2-5: {((lineage_sizes > 1) & (lineage_sizes <= 5)).sum()}, >5: {(lineage_sizes > 5).sum()}"""
        except Exception as e:
            return f"Error analyzing lineages: {str(e)}"

    def _analyze_reproduction_success_rate(self) -> str:
        if self.agents_df is None:
            return "Error: Agents data not available"
        if "success" in self.agents_df.columns:
            success_rate = self.agents_df["success"].mean() * 100
            trend = (
                self.agents_df.groupby("step_number")["success"].mean().diff().mean()
            )

            return f"""
Reproduction Success Analysis:
- Overall success rate: {success_rate:.1f}%
- Success trend: {'Improving' if trend > 0 else 'Declining'}
- Rate of change: {abs(trend)*100:.2f}% per step
- Reproduction attempts: {len(self.agents_df)}"""
        return "Reproduction data not available"

    def _save_analyses(self, output_dir: str):
        """Save all analyses to both JSON and text files."""
        json_path = os.path.join(output_dir, "chart_analyses.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.analyses, f, indent=4)

        text_path = os.path.join(output_dir, "chart_analyses.txt")
        with open(text_path, "w", encoding="utf-8") as f:
            f.write("SIMULATION ANALYSIS SUMMARY\n\n")
            for chart_name, analysis in self.analyses.items():
                f.write(f"\n{'='*30}\n")
                f.write(analysis.strip())
                f.write(f"\n{'='*30}\n")
