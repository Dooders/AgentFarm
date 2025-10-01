"""Module for identifying and analyzing significant events in simulation history.

This module provides tools to analyze simulation data and identify key events
that may explain major changes in population dynamics, resource distribution,
and other important metrics.

Example:
    >>> from significant_events import SignificantEventAnalyzer
    >>> analyzer = SignificantEventAnalyzer(simulation_db)
    >>> events = analyzer.get_significant_events(start_step=100, end_step=200)
    >>> print(events)
    [
        SignificantEvent(
            step_number=100,
            event_type="population_shift",
            description="Significant increase in system agents",
            metrics={"population_ratio": 0.5, "ratio_change": 0.1},
            severity=0.8,
        ),

        SignificantEvent(
            step_number=101,
            event_type="resource_crisis",
            description="Resource efficiency drops sharply",
            metrics={"resource_efficiency": 0.45},
            severity=0.7,
        ),
    ]
"""

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text

from farm.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class SignificantEvent:
    """Represents a significant event detected in the simulation.

    Attributes
    ----------
    step_number : int
        Simulation step when event occurred
    event_type : str
        Category of event (e.g. "population_shift", "resource_crisis")
    description : str
        Human-readable description of the event
    metrics : Dict
        Relevant metrics and their values at time of event
    severity : float
        Relative importance score (0-1)
    """

    step_number: int
    event_type: str
    description: str
    metrics: Dict
    severity: float


class SignificantEventAnalyzer:
    """Analyzes simulation history to identify significant events.

    This class processes simulation metrics and agent data to detect notable
    events that may explain major changes in simulation outcomes.

    Attributes
    ----------
    db : SimulationDatabase
        Database connection to query simulation history
    high_scores_path : str
        Path to JSON file storing high scores
    high_scores : dict
        Current high scores data

    Methods
    -------
    get_significant_events(start_step, end_step, min_severity)
        Returns all significant events in chronological order
    get_event_summary(events)
        Generates human-readable summary of events
    """

    def __init__(
        self, simulation_db, high_scores_path="utils/simulation_high_scores.json"
    ):
        """Initialize analyzer with database connection.

        Parameters
        ----------
        simulation_db : SimulationDatabase
            Database connection to query simulation history
        high_scores_path : str, optional
            Path to high scores JSON file
        """
        self.db = simulation_db
        self.high_scores_path = high_scores_path
        self.high_scores = self._load_high_scores()

    def _load_high_scores(self) -> dict:
        """Load high scores from JSON file or create default if not exists."""
        default_high_scores = {
            "population": {
                "max_total_agents": {"value": 0, "step": 0, "date": None},
                "max_system_agents": {"value": 0, "step": 0, "date": None},
                "max_independent_agents": {"value": 0, "step": 0, "date": None},
                "max_generation": {"value": 0, "step": 0, "date": None},
            },
            "resources": {
                "max_total_resources": {"value": 0, "step": 0, "date": None},
                "max_agent_resources": {"value": 0, "step": 0, "date": None},
                "max_resource_efficiency": {"value": 0, "step": 0, "date": None},
            },
            "combat": {
                "most_combat_encounters": {"value": 0, "step": 0, "date": None},
                "highest_success_rate": {"value": 0, "step": 0, "date": None},
            },
            "learning": {
                "highest_reward": {"value": 0, "step": 0, "date": None},
                "most_learning_agents": {"value": 0, "step": 0, "date": None},
            },
            "reproduction": {
                "most_births": {"value": 0, "step": 0, "date": None},
                "highest_success_rate": {"value": 0, "step": 0, "date": None},
            },
        }

        if os.path.exists(self.high_scores_path):
            try:
                with open(self.high_scores_path, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning("Corrupted high scores file, creating new one")
                return default_high_scores
        return default_high_scores

    def _save_high_scores(self):
        """Save high scores to JSON file."""
        with open(self.high_scores_path, "w") as f:
            json.dump(self.high_scores, f, indent=2)

    def _check_and_update_high_score(
        self, category: str, metric: str, value: float, step: int
    ) -> bool:
        """Check if value breaks record and update if so."""
        if category not in self.high_scores:
            return False

        if metric not in self.high_scores[category]:
            return False

        current_record = self.high_scores[category][metric]["value"]
        if value > current_record:
            self.high_scores[category][metric].update(
                {"value": value, "step": step, "date": datetime.now().isoformat()}
            )
            self._save_high_scores()
            return True
        return False

    def get_significant_events(
        self,
        start_step: Optional[int] = None,
        end_step: Optional[int] = None,
        min_severity: float = 0.3,
    ) -> List[SignificantEvent]:
        """Identify all significant events in the given time range.

        Parameters
        ----------
        start_step : int, optional
            First step to analyze
        end_step : int, optional
            Last step to analyze
        min_severity : float, optional
            Minimum severity threshold (0-1) for events

        Returns
        -------
        List[SignificantEvent]
            Chronological list of significant events
        """
        # Get simulation metrics for the time range
        metrics_df = pd.read_sql(
            f"""
            SELECT * FROM simulation_steps 
            WHERE step_number >= COALESCE(:start_step, 0)
            AND step_number <= COALESCE(:end_step, step_number)
            ORDER BY step_number
            """,
            self.db.engine,
            params={
                k: v
                for k, v in {"start_step": start_step, "end_step": end_step}.items()
                if v is not None
            },
        )

        events = []

        # Analyze population dynamics
        events.extend(self._analyze_population_shifts(metrics_df))

        # Analyze resource distribution
        events.extend(self._analyze_resource_events(metrics_df))

        # Analyze combat patterns
        events.extend(self._analyze_combat_events(metrics_df))

        # Add reproduction analysis
        events.extend(self._analyze_reproduction_events(start_step, end_step))

        # Add health incident analysis
        events.extend(self._analyze_health_incidents(start_step, end_step))

        # Add learning experience analysis
        events.extend(self._analyze_learning_experiences(start_step, end_step))

        # Add action pattern analysis
        events.extend(self._analyze_action_patterns(start_step, end_step))

        # Check for high scores in metrics
        for _, row in metrics_df.iterrows():
            step = row["step_number"]

            # Population records
            if self._check_and_update_high_score(
                "population", "max_total_agents", row["total_agents"], step
            ):
                events.append(
                    SignificantEvent(
                        step_number=step,
                        event_type="high_score",
                        description=f"New record: Highest total population ({row['total_agents']} agents)",
                        metrics={"total_agents": row["total_agents"]},
                        severity=0.8,
                    )
                )

            if self._check_and_update_high_score(
                "population", "max_generation", row["current_max_generation"], step
            ):
                events.append(
                    SignificantEvent(
                        step_number=step,
                        event_type="high_score",
                        description=f"New record: Highest generation reached ({row['current_max_generation']})",
                        metrics={"max_generation": row["current_max_generation"]},
                        severity=0.7,
                    )
                )

            # Resource records
            if self._check_and_update_high_score(
                "resources", "max_total_resources", row["total_resources"], step
            ):
                events.append(
                    SignificantEvent(
                        step_number=step,
                        event_type="high_score",
                        description=f"New record: Most total resources ({row['total_resources']:.1f})",
                        metrics={"total_resources": row["total_resources"]},
                        severity=0.7,
                    )
                )

            if self._check_and_update_high_score(
                "resources", "max_resource_efficiency", row["resource_efficiency"], step
            ):
                events.append(
                    SignificantEvent(
                        step_number=step,
                        event_type="high_score",
                        description=f"New record: Highest resource efficiency ({row['resource_efficiency']:.2f})",
                        metrics={"resource_efficiency": row["resource_efficiency"]},
                        severity=0.6,
                    )
                )

            # Combat records
            if (
                row["combat_encounters"] > 0
            ):  # Only check combat records if there was combat
                if self._check_and_update_high_score(
                    "combat", "most_combat_encounters", row["combat_encounters"], step
                ):
                    events.append(
                        SignificantEvent(
                            step_number=step,
                            event_type="high_score",
                            description=f"New record: Most combat encounters ({row['combat_encounters']})",
                            metrics={"combat_encounters": row["combat_encounters"]},
                            severity=0.7,
                        )
                    )

                success_rate = row["successful_attacks"] / row["combat_encounters"]
                if self._check_and_update_high_score(
                    "combat", "highest_success_rate", success_rate, step
                ):
                    events.append(
                        SignificantEvent(
                            step_number=step,
                            event_type="high_score",
                            description=f"New record: Highest combat success rate ({success_rate:.1%})",
                            metrics={"success_rate": success_rate},
                            severity=0.6,
                        )
                    )

        # Filter by severity and sort chronologically
        significant_events = [
            event for event in events if event.severity >= min_severity
        ]
        significant_events.sort(key=lambda x: x.step_number)

        return significant_events

    def _analyze_population_shifts(
        self, metrics_df: pd.DataFrame
    ) -> List[SignificantEvent]:
        """Detect significant changes in population composition."""
        events = []

        # Calculate population ratios and their changes
        metrics_df["total_pop"] = metrics_df["total_agents"]
        for agent_type in ["system_agents", "independent_agents", "control_agents"]:
            metrics_df[f"{agent_type}_ratio"] = (
                metrics_df[agent_type] / metrics_df["total_pop"]
            )
            metrics_df[f"{agent_type}_ratio_change"] = metrics_df[
                f"{agent_type}_ratio"
            ].diff()

        # Detect rapid changes in population ratios
        for agent_type in ["system", "independent", "control"]:
            ratio_col = f"{agent_type}_agents_ratio"
            change_col = f"{agent_type}_agents_ratio_change"

            # Find significant changes (using standard deviation as threshold)
            threshold = metrics_df[change_col].std() * 2
            significant_changes = metrics_df[abs(metrics_df[change_col]) > threshold]

            for _, row in significant_changes.iterrows():
                direction = "increase" if row[change_col] > 0 else "decline"
                severity = min(abs(row[change_col]) / threshold, 1.0)

                events.append(
                    SignificantEvent(
                        step_number=row["step_number"],
                        event_type="population_shift",
                        description=f"Significant {direction} in {agent_type} agent population",
                        metrics={
                            "population_ratio": row[ratio_col],
                            "ratio_change": row[change_col],
                            "total_population": row["total_pop"],
                            "agent_count": row[f"{agent_type}_agents"],
                        },
                        severity=severity,
                    )
                )

        return events

    def _analyze_resource_events(
        self, metrics_df: pd.DataFrame
    ) -> List[SignificantEvent]:
        """Detect significant resource-related events."""
        events = []

        # Calculate resource metrics
        metrics_df["resource_per_agent"] = (
            metrics_df["total_resources"] / metrics_df["total_agents"]
        )
        metrics_df["resource_efficiency_change"] = metrics_df[
            "resource_efficiency"
        ].diff()

        # Detect resource crises
        resource_threshold = metrics_df["resource_per_agent"].mean() * 0.5
        crisis_periods = metrics_df[
            metrics_df["resource_per_agent"] < resource_threshold
        ]

        for _, row in crisis_periods.iterrows():
            severity = 1 - (
                row["resource_per_agent"] / metrics_df["resource_per_agent"].mean()
            )

            events.append(
                SignificantEvent(
                    step_number=row["step_number"],
                    event_type="resource_crisis",
                    description="Severe resource shortage affecting population",
                    metrics={
                        "resources_per_agent": row["resource_per_agent"],
                        "total_resources": row["total_resources"],
                        "resource_efficiency": row["resource_efficiency"],
                        "resource_distribution_entropy": row[
                            "resource_distribution_entropy"
                        ],
                    },
                    severity=severity,
                )
            )

        return events

    def _analyze_combat_events(
        self, metrics_df: pd.DataFrame
    ) -> List[SignificantEvent]:
        """Detect significant combat-related events."""
        events = []

        # Skip analysis if no combat data
        if metrics_df["combat_encounters"].sum() == 0:
            logger.debug("No combat encounters found in metrics data")
            return events

        # Calculate combat intensity metrics with better handling of edge cases
        metrics_df["combat_success_rate"] = metrics_df[
            "successful_attacks"
        ] / metrics_df["combat_encounters"].replace(0, 1)

        # Calculate rolling averages for smoother detection
        window_size = 5
        metrics_df["combat_encounters_avg"] = (
            metrics_df["combat_encounters"]
            .rolling(window=window_size, min_periods=1)
            .mean()
        )

        # More sensitive threshold for combat detection
        base_threshold = max(
            metrics_df["combat_encounters_avg"].mean() * 1.5,  # Lower multiplier
            1.0,  # Minimum threshold - any combat is notable if it's rare
        )

        # Detect periods of combat activity
        combat_periods = metrics_df[metrics_df["combat_encounters"] > base_threshold]

        logger.debug(
            f"Found {len(combat_periods)} steps with combat above threshold {base_threshold:.2f}"
        )

        for _, row in combat_periods.iterrows():
            # Calculate severity based on how much above threshold
            severity = max(
                min(row["combat_encounters"] / base_threshold, 1.0),
                0.4,  # Minimum severity to ensure visibility
            )

            # Determine if this was a particularly successful period
            high_success = row["combat_success_rate"] > 0.6

            description = "Period of intense combat activity" + (
                " with high success rate" if high_success else ""
            )

            events.append(
                SignificantEvent(
                    step_number=row["step_number"],
                    event_type="intense_combat",
                    description=description,
                    metrics={
                        "combat_encounters": row["combat_encounters"],
                        "successful_attacks": row["successful_attacks"],
                        "combat_success_rate": row["combat_success_rate"],
                        "average_encounters": row["combat_encounters_avg"],
                    },
                    severity=severity,
                )
            )

        # Also detect significant changes in combat patterns
        metrics_df["combat_change"] = metrics_df["combat_encounters"].diff()
        significant_changes = metrics_df[
            abs(metrics_df["combat_change"]) > base_threshold
        ]

        for _, row in significant_changes.iterrows():
            direction = "increase" if row["combat_change"] > 0 else "decrease"
            severity = max(min(abs(row["combat_change"]) / base_threshold, 1.0), 0.4)

            events.append(
                SignificantEvent(
                    step_number=row["step_number"],
                    event_type="combat_pattern_change",
                    description=f"Sudden {direction} in combat activity",
                    metrics={
                        "combat_change": row["combat_change"],
                        "combat_encounters": row["combat_encounters"],
                        "previous_encounters": row["combat_encounters"]
                        - row["combat_change"],
                        "combat_success_rate": row["combat_success_rate"],
                    },
                    severity=severity,
                )
            )

        return events

    def _analyze_reproduction_events(
        self, start_step: Optional[int] = None, end_step: Optional[int] = None
    ) -> List[SignificantEvent]:
        """Detect significant reproduction-related events."""
        events = []

        # Query reproduction events data with debug logging
        query = f"""
        SELECT 
            step_number,
            COUNT(*) as total_attempts,
            SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_attempts,
            AVG(parent_resources_before) as avg_parent_resources,
            AVG(CASE WHEN success = 1 THEN offspring_initial_resources ELSE NULL END) 
                as avg_offspring_resources,
            GROUP_CONCAT(failure_reason) as failure_reasons
        FROM reproduction_events
        WHERE step_number >= COALESCE(:start_step, 0)
        AND step_number <= COALESCE(:end_step, step_number)
        GROUP BY step_number
        ORDER BY step_number
        """

        logger.debug(
            f"Executing reproduction query with params: start={start_step}, end={end_step}"
        )
        reproduction_df = pd.read_sql(
            query,
            self.db.engine,
            params={
                k: v
                for k, v in {"start_step": start_step, "end_step": end_step}.items()
                if v is not None
            },
        )

        if reproduction_df.empty:
            logger.info("No reproduction events found in the specified time range")
            return events

        # Calculate success rates with more lenient thresholds
        reproduction_df["success_rate"] = (
            reproduction_df["successful_attempts"] / reproduction_df["total_attempts"]
        ).fillna(
            0
        )  # Handle division by zero
        reproduction_df["success_rate_change"] = reproduction_df["success_rate"].diff()

        # Lower threshold for detecting significant changes
        success_rate_threshold = max(
            reproduction_df["success_rate_change"].std() * 1.5,  # Lower multiplier
            0.1,  # Minimum threshold
        )

        significant_changes = reproduction_df[
            abs(reproduction_df["success_rate_change"]) > success_rate_threshold
        ]

        logger.debug(
            f"Found {len(significant_changes)} significant reproduction rate changes"
        )

        for _, row in significant_changes.iterrows():
            direction = "increase" if row["success_rate_change"] > 0 else "decline"
            # Increase minimum severity to ensure events aren't filtered out
            severity = max(
                min(abs(row["success_rate_change"]) / success_rate_threshold, 1.0),
                0.4,  # Minimum severity
            )

            events.append(
                SignificantEvent(
                    step_number=row["step_number"],
                    event_type="reproduction_shift",
                    description=f"Significant {direction} in reproduction success rate",
                    metrics={
                        "success_rate": row["success_rate"],
                        "rate_change": row["success_rate_change"],
                        "total_attempts": row["total_attempts"],
                        "successful_attempts": row["successful_attempts"],
                        "avg_parent_resources": row["avg_parent_resources"],
                        "avg_offspring_resources": row["avg_offspring_resources"],
                    },
                    severity=severity,
                )
            )

        # More lenient resource crisis detection
        resource_threshold = (
            reproduction_df["avg_parent_resources"].mean() * 0.6
        )  # Increased threshold
        resource_crisis = reproduction_df[
            reproduction_df["avg_parent_resources"] < resource_threshold
        ]

        logger.debug(f"Found {len(resource_crisis)} reproduction resource crises")

        for _, row in resource_crisis.iterrows():
            severity = max(
                1
                - (
                    row["avg_parent_resources"]
                    / reproduction_df["avg_parent_resources"].mean()
                ),
                0.4,  # Minimum severity
            )

            events.append(
                SignificantEvent(
                    step_number=row["step_number"],
                    event_type="reproduction_resource_crisis",
                    description="Critical shortage of resources for reproduction",
                    metrics={
                        "avg_parent_resources": row["avg_parent_resources"],
                        "success_rate": row["success_rate"],
                        "total_attempts": row["total_attempts"],
                        "successful_attempts": row["successful_attempts"],
                    },
                    severity=severity,
                )
            )

        # Adjust failure pattern analysis
        failure_df = pd.read_sql(
            f"""
            SELECT 
                step_number,
                failure_reason,
                COUNT(*) as failure_count
            FROM reproduction_events
            WHERE success = 0
            AND step_number >= COALESCE(:start_step, 0)
            AND step_number <= COALESCE(:end_step, step_number)
            GROUP BY step_number, failure_reason
            HAVING failure_count > 3  -- Lowered threshold
            ORDER BY step_number, failure_count DESC
            """,
            self.db.engine,
            params={
                k: v
                for k, v in {"start_step": start_step, "end_step": end_step}.items()
                if v is not None
            },
        )

        logger.debug(f"Found {len(failure_df)} reproduction failure patterns")

        for _, row in failure_df.iterrows():
            severity = max(
                min(row["failure_count"] / 5, 1.0),  # Adjusted normalization
                0.4,  # Minimum severity
            )

            events.append(
                SignificantEvent(
                    step_number=row["step_number"],
                    event_type="reproduction_failure_pattern",
                    description=f"Frequent reproduction failures: {row['failure_reason']}",
                    metrics={
                        "failure_reason": row["failure_reason"],
                        "failure_count": row["failure_count"],
                    },
                    severity=severity,
                )
            )

        logger.info(f"Total reproduction events detected: {len(events)}")
        return events

    def _analyze_health_incidents(
        self, start_step: Optional[int] = None, end_step: Optional[int] = None
    ) -> List[SignificantEvent]:
        """Detect significant health-related events and patterns.

        Analyzes health incidents to identify:
        - Sudden spikes in health incidents
        - Mass health decline events
        - Patterns of specific health incident causes
        """
        events = []

        # Query health incidents with aggregation by step
        query = """
            SELECT 
                step_number,
                COUNT(*) as incident_count,
                AVG(health_before - health_after) as avg_health_loss,
                GROUP_CONCAT(DISTINCT cause) as causes,
                COUNT(DISTINCT agent_id) as affected_agents,
                MIN(health_after) as min_health_after,
                AVG(health_after) as avg_health_after
            FROM health_incidents
            WHERE step_number >= COALESCE(:start_step, 0)
            AND step_number <= COALESCE(:end_step, step_number)
            GROUP BY step_number
            ORDER BY step_number
        """

        health_df = pd.read_sql(
            query,
            self.db.engine,
            params={
                k: v
                for k, v in {"start_step": start_step, "end_step": end_step}.items()
                if v is not None
            },
        )

        if health_df.empty:
            logger.debug("No health incidents found in the specified time range")
            return events

        # Calculate rolling averages for smoother detection
        window_size = 5
        health_df["incident_count_avg"] = (
            health_df["incident_count"]
            .rolling(window=window_size, min_periods=1)
            .mean()
        )

        # Calculate incident spikes
        incident_std = health_df["incident_count"].std()
        base_threshold = max(
            health_df["incident_count_avg"].mean() + (incident_std * 1.5),
            3.0,  # Minimum threshold for significance
        )

        # Detect significant spikes in health incidents
        spike_periods = health_df[health_df["incident_count"] > base_threshold]

        for _, row in spike_periods.iterrows():
            # Calculate severity based on deviation from normal
            severity = min(row["incident_count"] / base_threshold, 1.0)
            severity = max(severity, 0.4)  # Minimum severity threshold

            # Analyze causes
            causes = row["causes"].split(",") if row["causes"] else []
            primary_cause = max(set(causes), key=causes.count) if causes else "unknown"

            # Create description based on patterns
            if row["avg_health_loss"] > 0.5:  # Significant average health loss
                description = f"Mass health decline event: {row['affected_agents']} agents affected"
            else:
                description = (
                    f"Spike in health incidents: {row['incident_count']} incidents"
                )

            if primary_cause != "unknown":
                description += f" (primary cause: {primary_cause})"

            events.append(
                SignificantEvent(
                    step_number=row["step_number"],
                    event_type="health_crisis",
                    description=description,
                    metrics={
                        "incident_count": row["incident_count"],
                        "affected_agents": row["affected_agents"],
                        "average_health_loss": row["avg_health_loss"],
                        "minimum_health": row["min_health_after"],
                        "average_health": row["avg_health_after"],
                        "primary_cause": primary_cause,
                    },
                    severity=severity,
                )
            )

        # Analyze patterns of sustained health decline
        health_df["health_trend"] = (
            health_df["avg_health_after"]
            .rolling(window=window_size, min_periods=1)
            .mean()
            .diff()
        )

        # Detect periods of sustained health decline
        decline_threshold = health_df["health_trend"].std() * -1.5
        decline_periods = health_df[health_df["health_trend"] < decline_threshold]

        for _, row in decline_periods.iterrows():
            severity = min(abs(row["health_trend"] / decline_threshold), 1.0)
            severity = max(severity, 0.4)

            events.append(
                SignificantEvent(
                    step_number=row["step_number"],
                    event_type="health_decline_trend",
                    description="Sustained decline in population health",
                    metrics={
                        "health_decline_rate": row["health_trend"],
                        "average_health": row["avg_health_after"],
                        "affected_agents": row["affected_agents"],
                        "incident_count": row["incident_count"],
                    },
                    severity=severity,
                )
            )

        logger.info(f"Detected {len(events)} significant health-related events")
        return events

    def _analyze_learning_experiences(
        self, start_step: Optional[int] = None, end_step: Optional[int] = None
    ) -> List[SignificantEvent]:
        """Analyze learning experiences to detect significant patterns and anomalies."""
        events = []

        # Query learning experiences with aggregation
        query = """
            SELECT 
                step_number,
                module_type,
                COUNT(*) as experience_count,
                COUNT(DISTINCT agent_id) as learning_agents,
                AVG(reward) as avg_reward,
                MIN(reward) as min_reward,
                MAX(reward) as max_reward,
                COUNT(DISTINCT module_id) as unique_modules,
                GROUP_CONCAT(DISTINCT action_taken_mapped) as actions_taken
            FROM learning_experiences
            WHERE step_number >= COALESCE(:start_step, 0)
            AND step_number <= COALESCE(:end_step, step_number)
            GROUP BY step_number, module_type
            ORDER BY step_number, module_type
        """

        learning_df = pd.read_sql(
            query,
            self.db.engine,
            params={
                k: v
                for k, v in {"start_step": start_step, "end_step": end_step}.items()
                if v is not None
            },
        )

        if learning_df.empty:
            logger.debug("No learning experiences found in the specified time range")
            return events

        # Calculate rolling statistics for each module type
        for module_type in learning_df["module_type"].unique():
            module_data = learning_df[learning_df["module_type"] == module_type].copy()

            # Skip if insufficient data
            if len(module_data) < 3:
                continue

            # Calculate rolling averages
            window_size = 5
            module_data["experience_count_avg"] = (
                module_data["experience_count"]
                .rolling(window=window_size, min_periods=1)
                .mean()
            )

            module_data["reward_avg"] = (
                module_data["avg_reward"]
                .rolling(window=window_size, min_periods=1)
                .mean()
            )

            # Detect learning spikes
            count_std = module_data["experience_count"].std()
            count_threshold = max(
                module_data["experience_count_avg"].mean() + (count_std * 1.5),
                5.0,  # Minimum threshold
            )

            spike_periods = module_data[
                module_data["experience_count"] > count_threshold
            ]

            for _, row in spike_periods.iterrows():
                severity = min(row["experience_count"] / count_threshold, 1.0)
                severity = max(severity, 0.4)

                # Analyze actions taken
                actions = (
                    row["actions_taken"].split(",") if row["actions_taken"] else []
                )
                primary_action = (
                    max(set(actions), key=actions.count) if actions else "unknown"
                )

                events.append(
                    SignificantEvent(
                        step_number=row["step_number"],
                        event_type="learning_spike",
                        description=(
                            f"Surge in learning activity for {module_type} module: "
                            f"{row['learning_agents']} agents, "
                            f"{row['experience_count']} experiences"
                        ),
                        metrics={
                            "module_type": module_type,
                            "experience_count": row["experience_count"],
                            "learning_agents": row["learning_agents"],
                            "average_reward": row["avg_reward"],
                            "reward_range": row["max_reward"] - row["min_reward"],
                            "unique_modules": row["unique_modules"],
                            "primary_action": primary_action,
                        },
                        severity=severity,
                    )
                )

            # Detect reward anomalies
            module_data["reward_change"] = module_data["avg_reward"].diff()
            reward_std = module_data["reward_change"].std()
            reward_threshold = reward_std * 2

            reward_anomalies = module_data[
                abs(module_data["reward_change"]) > reward_threshold
            ]

            for _, row in reward_anomalies.iterrows():
                direction = "increase" if row["reward_change"] > 0 else "decrease"
                severity = min(abs(row["reward_change"]) / reward_threshold, 1.0)
                severity = max(severity, 0.4)

                events.append(
                    SignificantEvent(
                        step_number=row["step_number"],
                        event_type="learning_reward_shift",
                        description=(
                            f"Significant {direction} in learning rewards for {module_type} module"
                        ),
                        metrics={
                            "module_type": module_type,
                            "reward_change": row["reward_change"],
                            "average_reward": row["avg_reward"],
                            "experience_count": row["experience_count"],
                            "learning_agents": row["learning_agents"],
                        },
                        severity=severity,
                    )
                )

        # Analyze collective learning trends
        total_data = (
            learning_df.groupby("step_number")
            .agg(
                {
                    "experience_count": "sum",
                    "learning_agents": "sum",
                    "avg_reward": "mean",
                    "unique_modules": "sum",
                }
            )
            .reset_index()
        )

        total_data["learning_efficiency"] = (
            total_data["avg_reward"]
            * total_data["learning_agents"]
            / total_data["experience_count"].replace(0, 1)
        )

        # Detect collective learning breakthroughs
        total_data["efficiency_change"] = total_data["learning_efficiency"].diff()
        efficiency_std = total_data["efficiency_change"].std()
        breakthrough_threshold = efficiency_std * 2

        breakthroughs = total_data[
            total_data["efficiency_change"] > breakthrough_threshold
        ]

        for _, row in breakthroughs.iterrows():
            severity = min(row["efficiency_change"] / breakthrough_threshold, 1.0)
            severity = max(severity, 0.4)

            events.append(
                SignificantEvent(
                    step_number=row["step_number"],
                    event_type="learning_breakthrough",
                    description="Collective learning breakthrough detected",
                    metrics={
                        "learning_efficiency": row["learning_efficiency"],
                        "efficiency_change": row["efficiency_change"],
                        "total_experiences": row["experience_count"],
                        "total_learning_agents": row["learning_agents"],
                        "average_reward": row["avg_reward"],
                        "active_modules": row["unique_modules"],
                    },
                    severity=severity,
                )
            )

        # Add agent-level analysis
        agent_query = """
            SELECT 
                step_number,
                agent_id,
                module_type,
                COUNT(*) as experience_count,
                AVG(reward) as avg_reward,
                SUM(reward) as total_reward,
                MIN(reward) as min_reward,
                MAX(reward) as max_reward,
                GROUP_CONCAT(DISTINCT action_taken_mapped) as actions_taken
            FROM learning_experiences
            WHERE step_number >= COALESCE(:start_step, 0)
            AND step_number <= COALESCE(:end_step, step_number)
            GROUP BY step_number, agent_id, module_type
            ORDER BY step_number, agent_id, module_type
        """

        agent_learning_df = pd.read_sql(
            agent_query,
            self.db.engine,
            params={
                k: v
                for k, v in {"start_step": start_step, "end_step": end_step}.items()
                if v is not None
            },
        )

        if not agent_learning_df.empty:
            # Calculate agent performance metrics
            for agent_id in agent_learning_df["agent_id"].unique():
                agent_data = agent_learning_df[
                    agent_learning_df["agent_id"] == agent_id
                ].copy()

                # Skip if insufficient data
                if len(agent_data) < 3:
                    continue

                # Calculate rolling metrics
                window_size = 3  # Smaller window for individual analysis
                agent_data["reward_avg"] = (
                    agent_data["avg_reward"]
                    .rolling(window=window_size, min_periods=1)
                    .mean()
                )
                agent_data["reward_trend"] = agent_data["reward_avg"].diff()

                # Detect rapid learning progress
                reward_std = agent_data["reward_trend"].std()
                progress_threshold = reward_std * 2

                progress_spikes = agent_data[
                    agent_data["reward_trend"] > progress_threshold
                ]

                for _, row in progress_spikes.iterrows():
                    severity = min(row["reward_trend"] / progress_threshold, 1.0)
                    severity = max(severity, 0.4)

                    # Get module-specific details
                    module_info = (
                        f"in {row['module_type']}"
                        if row["module_type"]
                        else "across modules"
                    )

                    events.append(
                        SignificantEvent(
                            step_number=row["step_number"],
                            event_type="agent_learning_breakthrough",
                            description=(
                                f"Agent {agent_id} shows rapid learning progress {module_info}"
                            ),
                            metrics={
                                "agent_id": agent_id,
                                "module_type": row["module_type"],
                                "reward_improvement": row["reward_trend"],
                                "current_reward_level": row["avg_reward"],
                                "experience_count": row["experience_count"],
                                "reward_range": row["max_reward"] - row["min_reward"],
                            },
                            severity=severity,
                        )
                    )

                # Detect consistent high performers
                performance_window = 5
                agent_data["performance_score"] = (
                    agent_data["avg_reward"] * agent_data["experience_count"]
                )
                agent_data["high_performer"] = (
                    agent_data["performance_score"]
                    > agent_data["performance_score"].mean()
                    + agent_data["performance_score"].std()
                )

                # Find sustained high performance periods
                high_performance_streaks = (
                    agent_data["high_performer"]
                    .astype(float)
                    .rolling(performance_window)
                    .sum()
                    >= performance_window
                    * 0.8  # 80% of window needs to be high performance
                )

                streak_starts = agent_data[
                    high_performance_streaks
                    & (
                        ~high_performance_streaks.shift(1).fillna(0).astype(bool)
                    )  # Explicit type conversion
                ]

                for _, row in streak_starts.iterrows():
                    events.append(
                        SignificantEvent(
                            step_number=row["step_number"],
                            event_type="agent_sustained_performance",
                            description=(
                                f"Agent {agent_id} demonstrates sustained high performance "
                                f"in {row['module_type']}"
                            ),
                            metrics={
                                "agent_id": agent_id,
                                "module_type": row["module_type"],
                                "average_reward": row["avg_reward"],
                                "total_reward": row["total_reward"],
                                "experience_count": row["experience_count"],
                                "performance_score": row["performance_score"],
                            },
                            severity=0.6,  # Fixed severity for sustained performance
                        )
                    )

        logger.info(f"Detected {len(events)} significant learning events")
        return events

    def _analyze_action_patterns(
        self, start_step: Optional[int] = None, end_step: Optional[int] = None
    ) -> List[SignificantEvent]:
        """Analyze patterns and shifts in non-combat agent actions.

        Detects significant changes in:
        - Resource sharing behavior
        - Movement patterns
        - Defensive stance adoption
        - Action type distributions

        Parameters
        ----------
        start_step : int, optional
            First step to analyze
        end_step : int, optional
            Last step to analyze

        Returns
        -------
        List[SignificantEvent]
            Detected action pattern events
        """
        events = []

        # Query action data with aggregation by step and type
        query = """
            SELECT 
                step_number,
                action_type,
                COUNT(*) as action_count,
                COUNT(DISTINCT agent_id) as unique_agents,
                AVG(CASE 
                    WHEN resources_after IS NOT NULL 
                    AND resources_before IS NOT NULL 
                    THEN resources_after - resources_before 
                    ELSE 0 
                END) as avg_resource_change,
                SUM(CASE 
                    WHEN reward IS NOT NULL 
                    THEN reward 
                    ELSE 0 
                END) as total_reward
            FROM agent_actions
            WHERE step_number >= COALESCE(:start_step, 0)
            AND step_number <= COALESCE(:end_step, step_number)
            AND action_type NOT IN ('attack', 'defend_from_attack')
            GROUP BY step_number, action_type
            ORDER BY step_number, action_type
        """

        action_df = pd.read_sql(
            query,
            self.db.engine,
            params={
                k: v
                for k, v in {"start_step": start_step, "end_step": end_step}.items()
                if v is not None
            },
        )

        if action_df.empty:
            logger.debug("No non-combat actions found in the specified time range")
            return events

        # Calculate rolling statistics for each action type
        window_size = 5
        for action_type in action_df["action_type"].unique():
            type_data = action_df[action_df["action_type"] == action_type].copy()

            # Skip if insufficient data
            if len(type_data) < window_size:
                continue

            # Calculate rolling averages and changes
            type_data["action_count_avg"] = (
                type_data["action_count"]
                .rolling(window=window_size, min_periods=1)
                .mean()
            )

            type_data["action_count_change"] = type_data["action_count"].diff()
            type_data["reward_per_action"] = type_data["total_reward"] / type_data[
                "action_count"
            ].replace(0, 1)

            # Detect significant changes in action frequency
            count_std = type_data["action_count_change"].std()
            threshold = max(count_std * 1.5, 3.0)  # At least 3 actions difference

            significant_changes = type_data[
                abs(type_data["action_count_change"]) > threshold
            ]

            for _, row in significant_changes.iterrows():
                direction = "increase" if row["action_count_change"] > 0 else "decrease"
                severity = min(abs(row["action_count_change"]) / threshold, 1.0)
                severity = max(severity, 0.4)  # Minimum severity threshold

                events.append(
                    SignificantEvent(
                        step_number=row["step_number"],
                        event_type="action_pattern_shift",
                        description=(
                            f"Significant {direction} in {action_type} actions: "
                            f"{row['action_count']} actions by {row['unique_agents']} agents"
                        ),
                        metrics={
                            "action_type": action_type,
                            "action_count": row["action_count"],
                            "action_count_change": row["action_count_change"],
                            "unique_agents": row["unique_agents"],
                            "avg_resource_change": row["avg_resource_change"],
                            "reward_per_action": row["reward_per_action"],
                        },
                        severity=severity,
                    )
                )

        # Analyze resource sharing patterns specifically
        sharing_data = action_df[action_df["action_type"] == "share_resources"].copy()
        if not sharing_data.empty:
            sharing_data["resource_flow"] = (
                sharing_data["action_count"] * sharing_data["avg_resource_change"]
            )
            sharing_data["resource_flow_change"] = sharing_data["resource_flow"].diff()

            # Detect significant changes in resource sharing
            flow_std = sharing_data["resource_flow_change"].std()
            flow_threshold = flow_std * 2

            sharing_changes = sharing_data[
                abs(sharing_data["resource_flow_change"]) > flow_threshold
            ]

            for _, row in sharing_changes.iterrows():
                direction = (
                    "increase" if row["resource_flow_change"] > 0 else "decrease"
                )
                severity = min(abs(row["resource_flow_change"]) / flow_threshold, 1.0)
                severity = max(severity, 0.4)

                events.append(
                    SignificantEvent(
                        step_number=row["step_number"],
                        event_type="resource_sharing_shift",
                        description=(
                            f"Significant {direction} in resource sharing activity: "
                            f"{row['unique_agents']} agents involved"
                        ),
                        metrics={
                            "resource_flow": row["resource_flow"],
                            "flow_change": row["resource_flow_change"],
                            "sharing_actions": row["action_count"],
                            "unique_agents": row["unique_agents"],
                            "avg_transfer_amount": row["avg_resource_change"],
                        },
                        severity=severity,
                    )
                )

        # Analyze defensive behavior patterns
        defensive_query = """
            SELECT 
                s.step_number,  -- Explicitly use s.step_number
                COUNT(DISTINCT s.agent_id) as defending_agents,
                COUNT(DISTINCT a.agent_id) as total_agents,
                AVG(s.resource_level) as avg_defender_resources
            FROM agent_states s
            JOIN (
                SELECT DISTINCT step_number, agent_id  -- Added DISTINCT for efficiency
                FROM agent_states
            ) a ON s.step_number = a.step_number
            WHERE s.is_defending = 1
            AND s.step_number >= COALESCE(:start_step, 0)
            AND s.step_number <= COALESCE(:end_step, s.step_number)  -- Use s.step_number
            GROUP BY s.step_number  -- Explicitly use s.step_number
            ORDER BY s.step_number  -- Explicitly use s.step_number
        """

        defense_df = pd.read_sql(
            defensive_query,
            self.db.engine,
            params={
                k: v
                for k, v in {"start_step": start_step, "end_step": end_step}.items()
                if v is not None
            },
        )

        if not defense_df.empty:
            defense_df["defense_ratio"] = (
                defense_df["defending_agents"] / defense_df["total_agents"]
            )
            defense_df["ratio_change"] = defense_df["defense_ratio"].diff()

            # Detect significant changes in defensive behavior
            ratio_std = defense_df["ratio_change"].std()
            ratio_threshold = max(ratio_std * 2, 0.1)  # At least 10% change

            defensive_shifts = defense_df[
                abs(defense_df["ratio_change"]) > ratio_threshold
            ]

            for _, row in defensive_shifts.iterrows():
                direction = "increase" if row["ratio_change"] > 0 else "decrease"
                severity = min(abs(row["ratio_change"]) / ratio_threshold, 1.0)
                severity = max(severity, 0.4)

                events.append(
                    SignificantEvent(
                        step_number=row["step_number"],
                        event_type="defensive_behavior_shift",
                        description=(
                            f"Significant {direction} in defensive behavior: "
                            f"{row['defending_agents']} agents ({row['defense_ratio']:.1%} of population)"
                        ),
                        metrics={
                            "defending_agents": row["defending_agents"],
                            "total_agents": row["total_agents"],
                            "defense_ratio": row["defense_ratio"],
                            "ratio_change": row["ratio_change"],
                            "avg_defender_resources": row["avg_defender_resources"],
                        },
                        severity=severity,
                    )
                )

        return events

    def get_event_summary(self, events: List[SignificantEvent]) -> str:
        """Generate a human-readable summary of significant events."""
        if not events:
            return "No significant events detected in the specified time range."

        summary = ["Significant Events Analysis\n"]
        summary.append("=" * 30 + "\n")

        # Define event type ordering and grouping
        event_type_order = [
            ("population_shift", "Population Dynamics"),
            ("resource_crisis", "Resource Events"),
            ("intense_combat", "Combat Events"),
            ("reproduction_shift", "Reproduction Trends"),
            ("reproduction_resource_crisis", "Reproduction Resources"),
            ("reproduction_failure_pattern", "Reproduction Failures"),
            ("health_crisis", "Health Incidents"),
            ("health_decline_trend", "Health Decline Trends"),
            ("learning_spike", "Learning Activity"),
            ("learning_reward_shift", "Learning Rewards"),
            ("learning_breakthrough", "Learning Breakthroughs"),
            ("agent_learning_breakthrough", "Individual Learning Progress"),
            ("agent_sustained_performance", "Agent Performance Highlights"),
            ("action_pattern_shift", "Action Patterns"),
            ("resource_sharing_shift", "Resource Sharing"),
            ("defensive_behavior_shift", "Defensive Behavior"),
            ("high_score", "New Records"),
        ]

        # Group events by type
        event_types = {}
        for event in events:
            if event.event_type not in event_types:
                event_types[event.event_type] = []
            event_types[event.event_type].append(event)

        # Summarize each type of event in order
        for event_type, section_title in event_type_order:
            if event_type in event_types:
                summary.append(f"\n{section_title}:")
                summary.append("-" * 20)

                # Sort events by severity and take top 3
                type_events = sorted(
                    event_types[event_type], key=lambda x: x.severity, reverse=True
                )[:3]

                for event in type_events:
                    summary.append(
                        f"\nStep {event.step_number} (Severity: {event.severity:.2f}):"
                    )
                    summary.append(f"  {event.description}")

                    # Format metrics based on event type
                    for metric, value in event.metrics.items():
                        if isinstance(value, float):
                            if "rate" in metric:
                                summary.append(f"  - {metric}: {value:.1%}")
                            else:
                                summary.append(f"  - {metric}: {value:.2f}")
                        else:
                            summary.append(f"  - {metric}: {value}")

        return "\n".join(summary)

    def analyze_simulation(
        self,
        start_step: Optional[int] = None,
        end_step: Optional[int] = None,
        min_severity: float = 0.3,
        path: Optional[str] = None,
    ):
        """Analyze simulation and return both events and summary.

        Parameters
        ----------
        start_step : int, optional
            First step to analyze
        end_step : int, optional
            Last step to analyze
        min_severity : float, optional
            Minimum severity threshold (0-1) for events

        Returns
        -------
        tuple
            (List[SignificantEvent], str) containing events and summary
        """
        events = self.get_significant_events(
            start_step=start_step, end_step=end_step, min_severity=min_severity
        )
        summary = self.get_event_summary(events)

        if path:
            with open(os.path.join(path, "significant_events.txt"), "w") as f:
                f.write(summary)

        return events, summary

    def get_event_types(self, events):
        """Get counts of each event type.

        Parameters
        ----------
        events : List[SignificantEvent]
            Events to analyze

        Returns
        -------
        dict
            Mapping of event types to counts
        """
        event_counts = {}
        for event in events:
            event_counts[event.event_type] = event_counts.get(event.event_type, 0) + 1
        return event_counts

    def get_events_by_type(self, events, event_type):
        """Filter events by type.

        Parameters
        ----------
        events : List[SignificantEvent]
            Events to filter
        event_type : str
            Event type to filter for

        Returns
        -------
        List[SignificantEvent]
            Events matching the specified type
        """
        return [e for e in events if e.event_type == event_type]

    def get_events_in_range(self, events, start_step, end_step):
        """Filter events by step range.

        Parameters
        ----------
        events : List[SignificantEvent]
            Events to filter
        start_step : int
            First step to include
        end_step : int
            Last step to include

        Returns
        -------
        List[SignificantEvent]
            Events within the step range
        """
        return [
            e
            for e in events
            if (start_step is None or e.step_number >= start_step)
            and (end_step is None or e.step_number <= end_step)
        ]
