"""Module for identifying and analyzing significant events in simulation history.

This module provides tools to analyze simulation data and identify key events 
that may explain major changes in population dynamics, resource distribution,
and other important metrics.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import logging
from sqlalchemy import text

logger = logging.getLogger(__name__)


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

    Methods
    -------
    analyze_population_dynamics(start_step, end_step)
        Identifies significant population changes and their likely causes
    analyze_resource_distribution(start_step, end_step)
        Detects resource-related events that impacted populations
    get_significant_events(start_step, end_step)
        Returns all significant events in chronological order
    """

    def __init__(self, simulation_db):
        """Initialize analyzer with database connection.

        Parameters
        ----------
        simulation_db : SimulationDatabase
            Database connection to query simulation history
        """
        self.db = simulation_db

    def get_significant_events(
        self, start_step: int = None, end_step: int = None, min_severity: float = 0.3
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
            params={"start_step": start_step, "end_step": end_step},
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
        metrics_df["combat_success_rate"] = (
            metrics_df["successful_attacks"] / 
            metrics_df["combat_encounters"].replace(0, 1)
        )

        # Calculate rolling averages for smoother detection
        window_size = 5
        metrics_df["combat_encounters_avg"] = metrics_df["combat_encounters"].rolling(
            window=window_size, min_periods=1
        ).mean()

        # More sensitive threshold for combat detection
        base_threshold = max(
            metrics_df["combat_encounters_avg"].mean() * 1.5,  # Lower multiplier
            1.0  # Minimum threshold - any combat is notable if it's rare
        )

        # Detect periods of combat activity
        combat_periods = metrics_df[
            metrics_df["combat_encounters"] > base_threshold
        ]

        logger.debug(
            f"Found {len(combat_periods)} steps with combat above threshold {base_threshold:.2f}"
        )

        for _, row in combat_periods.iterrows():
            # Calculate severity based on how much above threshold
            severity = max(
                min(row["combat_encounters"] / base_threshold, 1.0),
                0.4  # Minimum severity to ensure visibility
            )

            # Determine if this was a particularly successful period
            high_success = row["combat_success_rate"] > 0.6

            description = (
                "Period of intense combat activity" +
                (" with high success rate" if high_success else "")
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
            severity = max(
                min(abs(row["combat_change"]) / base_threshold, 1.0),
                0.4
            )

            events.append(
                SignificantEvent(
                    step_number=row["step_number"],
                    event_type="combat_pattern_change",
                    description=f"Sudden {direction} in combat activity",
                    metrics={
                        "combat_change": row["combat_change"],
                        "combat_encounters": row["combat_encounters"],
                        "previous_encounters": row["combat_encounters"] - row["combat_change"],
                        "combat_success_rate": row["combat_success_rate"],
                    },
                    severity=severity,
                )
            )

        return events

    def _analyze_reproduction_events(
        self, start_step: int = None, end_step: int = None
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
        
        logger.debug(f"Executing reproduction query with params: start={start_step}, end={end_step}")
        reproduction_df = pd.read_sql(
            query,
            self.db.engine,
            params={"start_step": start_step, "end_step": end_step},
        )

        if reproduction_df.empty:
            logger.info("No reproduction events found in the specified time range")
            return events

        # Calculate success rates with more lenient thresholds
        reproduction_df['success_rate'] = (
            reproduction_df['successful_attempts'] / reproduction_df['total_attempts']
        ).fillna(0)  # Handle division by zero
        reproduction_df['success_rate_change'] = reproduction_df['success_rate'].diff()

        # Lower threshold for detecting significant changes
        success_rate_threshold = max(
            reproduction_df['success_rate_change'].std() * 1.5,  # Lower multiplier
            0.1  # Minimum threshold
        )
        
        significant_changes = reproduction_df[
            abs(reproduction_df['success_rate_change']) > success_rate_threshold
        ]

        logger.debug(f"Found {len(significant_changes)} significant reproduction rate changes")

        for _, row in significant_changes.iterrows():
            direction = "increase" if row['success_rate_change'] > 0 else "decline"
            # Increase minimum severity to ensure events aren't filtered out
            severity = max(
                min(abs(row['success_rate_change']) / success_rate_threshold, 1.0),
                0.4  # Minimum severity
            )

            events.append(
                SignificantEvent(
                    step_number=row['step_number'],
                    event_type="reproduction_shift",
                    description=f"Significant {direction} in reproduction success rate",
                    metrics={
                        "success_rate": row['success_rate'],
                        "rate_change": row['success_rate_change'],
                        "total_attempts": row['total_attempts'],
                        "successful_attempts": row['successful_attempts'],
                        "avg_parent_resources": row['avg_parent_resources'],
                        "avg_offspring_resources": row['avg_offspring_resources'],
                    },
                    severity=severity,
                )
            )

        # More lenient resource crisis detection
        resource_threshold = reproduction_df['avg_parent_resources'].mean() * 0.6  # Increased threshold
        resource_crisis = reproduction_df[
            reproduction_df['avg_parent_resources'] < resource_threshold
        ]

        logger.debug(f"Found {len(resource_crisis)} reproduction resource crises")

        for _, row in resource_crisis.iterrows():
            severity = max(
                1 - (row['avg_parent_resources'] / reproduction_df['avg_parent_resources'].mean()),
                0.4  # Minimum severity
            )

            events.append(
                SignificantEvent(
                    step_number=row['step_number'],
                    event_type="reproduction_resource_crisis",
                    description="Critical shortage of resources for reproduction",
                    metrics={
                        "avg_parent_resources": row['avg_parent_resources'],
                        "success_rate": row['success_rate'],
                        "total_attempts": row['total_attempts'],
                        "successful_attempts": row['successful_attempts'],
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
            params={"start_step": start_step, "end_step": end_step},
        )

        logger.debug(f"Found {len(failure_df)} reproduction failure patterns")

        for _, row in failure_df.iterrows():
            severity = max(
                min(row['failure_count'] / 5, 1.0),  # Adjusted normalization
                0.4  # Minimum severity
            )

            events.append(
                SignificantEvent(
                    step_number=row['step_number'],
                    event_type="reproduction_failure_pattern",
                    description=f"Frequent reproduction failures: {row['failure_reason']}",
                    metrics={
                        "failure_reason": row['failure_reason'],
                        "failure_count": row['failure_count'],
                    },
                    severity=severity,
                )
            )

        logger.info(f"Total reproduction events detected: {len(events)}")
        return events

    def _analyze_health_incidents(
        self, start_step: int = None, end_step: int = None
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
            params={"start_step": start_step, "end_step": end_step},
        )
        
        if health_df.empty:
            logger.debug("No health incidents found in the specified time range")
            return events
        
        # Calculate rolling averages for smoother detection
        window_size = 5
        health_df['incident_count_avg'] = health_df['incident_count'].rolling(
            window=window_size, min_periods=1
        ).mean()
        
        # Calculate incident spikes
        incident_std = health_df['incident_count'].std()
        base_threshold = max(
            health_df['incident_count_avg'].mean() + (incident_std * 1.5),
            3.0  # Minimum threshold for significance
        )
        
        # Detect significant spikes in health incidents
        spike_periods = health_df[health_df['incident_count'] > base_threshold]
        
        for _, row in spike_periods.iterrows():
            # Calculate severity based on deviation from normal
            severity = min(
                row['incident_count'] / base_threshold,
                1.0
            )
            severity = max(severity, 0.4)  # Minimum severity threshold
            
            # Analyze causes
            causes = row['causes'].split(',') if row['causes'] else []
            primary_cause = max(set(causes), key=causes.count) if causes else "unknown"
            
            # Create description based on patterns
            if row['avg_health_loss'] > 0.5:  # Significant average health loss
                description = f"Mass health decline event: {row['affected_agents']} agents affected"
            else:
                description = f"Spike in health incidents: {row['incident_count']} incidents"
            
            if primary_cause != "unknown":
                description += f" (primary cause: {primary_cause})"
            
            events.append(
                SignificantEvent(
                    step_number=row['step_number'],
                    event_type="health_crisis",
                    description=description,
                    metrics={
                        "incident_count": row['incident_count'],
                        "affected_agents": row['affected_agents'],
                        "average_health_loss": row['avg_health_loss'],
                        "minimum_health": row['min_health_after'],
                        "average_health": row['avg_health_after'],
                        "primary_cause": primary_cause,
                    },
                    severity=severity,
                )
            )
        
        # Analyze patterns of sustained health decline
        health_df['health_trend'] = health_df['avg_health_after'].rolling(
            window=window_size, min_periods=1
        ).mean().diff()
        
        # Detect periods of sustained health decline
        decline_threshold = health_df['health_trend'].std() * -1.5
        decline_periods = health_df[health_df['health_trend'] < decline_threshold]
        
        for _, row in decline_periods.iterrows():
            severity = min(
                abs(row['health_trend'] / decline_threshold),
                1.0
            )
            severity = max(severity, 0.4)
            
            events.append(
                SignificantEvent(
                    step_number=row['step_number'],
                    event_type="health_decline_trend",
                    description="Sustained decline in population health",
                    metrics={
                        "health_decline_rate": row['health_trend'],
                        "average_health": row['avg_health_after'],
                        "affected_agents": row['affected_agents'],
                        "incident_count": row['incident_count'],
                    },
                    severity=severity,
                )
            )
        
        logger.info(f"Detected {len(events)} significant health-related events")
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
            ("health_decline_trend", "Health Decline Trends")
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
                    event_types[event_type], 
                    key=lambda x: x.severity, 
                    reverse=True
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


def main():
    """Run significant event analysis from command line."""
    import argparse
    import logging

    from farm.database.database import SimulationDatabase

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Analyze significant events in simulation history"
    )
    parser.add_argument("--start-step", type=int, help="First step to analyze")
    parser.add_argument("--end-step", type=int, help="Last step to analyze")
    parser.add_argument(
        "--min-severity",
        type=float,
        default=0.3,
        help="Minimum severity threshold (0-1) for events",
    )
    parser.add_argument("--output", help="Path to save analysis results (optional)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Connect to database
        db_path = "simulations/simulation_results.db"
        print(f"Connecting to database: {db_path}")
        db = SimulationDatabase(db_path)

        # Check tables and data
        with db.engine.connect() as conn:
            # Check if reproduction_events table exists
            table_exists = conn.execute(
                text("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='reproduction_events'
                """)
            ).fetchone()
            
            if not table_exists:
                logger.warning("reproduction_events table does not exist!")
            else:
                # Check if table has data
                count = conn.execute(
                    text("SELECT COUNT(*) FROM reproduction_events")
                ).scalar()
                logger.info(f"Found {count} reproduction events in database")

            # Check combat data
            try:
                combat_data = conn.execute(
                    text("""
                        SELECT 
                            COUNT(*) as count, 
                            COALESCE(SUM(combat_encounters), 0) as total_encounters,
                            COALESCE(AVG(combat_encounters), 0) as avg_encounters
                        FROM simulation_steps 
                        WHERE combat_encounters > 0
                    """)
                ).fetchone()
                
                if combat_data and combat_data[0] > 0:
                    logger.info(
                        f"Combat stats: {combat_data[0]} steps with combat, "
                        f"total encounters: {combat_data[1]}, "
                        f"avg per step: {combat_data[2]:.2f}"
                    )
                else:
                    logger.info("No combat events found in database")
            except Exception as e:
                logger.warning(f"Error checking combat data: {e}")

        # Create analyzer
        analyzer = SignificantEventAnalyzer(db)

        # Get events
        print("Analyzing significant events...")
        events = analyzer.get_significant_events(
            start_step=args.start_step,
            end_step=args.end_step,
            min_severity=args.min_severity,
        )

        # Log event type counts
        event_counts = {}
        for event in events:
            event_counts[event.event_type] = event_counts.get(event.event_type, 0) + 1
        
        if event_counts:
            logger.info("Event counts by type: %s", event_counts)
        else:
            logger.info("No significant events detected")

        # Generate summary
        summary = analyzer.get_event_summary(events)

        # Output results
        if args.output:
            with open(args.output, "w") as f:
                f.write(summary)
            print(f"\nAnalysis saved to: {args.output}")
        else:
            print("\n" + summary)

    except Exception as e:
        logger.exception("Error during analysis")
        raise
    finally:
        if "db" in locals():
            db.close()


if __name__ == "__main__":
    main()
