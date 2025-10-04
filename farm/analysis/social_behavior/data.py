"""
Social Behavior Data Processing

This module handles data processing for social behavior analysis from simulation databases.
"""

from pathlib import Path
from typing import Optional
import pandas as pd

from farm.database.database import SimulationDatabase
from farm.database.repositories.action_repository import ActionRepository
from farm.database.repositories.agent_repository import AgentRepository
from farm.analysis.social_behavior.compute import compute_all_social_metrics


def process_social_behavior_data(
    experiment_path: Path,
    use_database: bool = True,
    **kwargs
) -> pd.DataFrame:
    """Process social behavior data from experiment.

    Args:
        experiment_path: Path to experiment directory
        use_database: Whether to use database or direct file access
        **kwargs: Additional processing options

    Returns:
        DataFrame with processed social behavior metrics
    """
    # Find simulation database
    db_path = experiment_path / "simulation.db"
    if not db_path.exists():
        # Try alternative locations
        db_path = experiment_path / "data" / "simulation.db"

    if not db_path.exists():
        raise FileNotFoundError(f"No simulation database found in {experiment_path}")

    # Connect to database
    db = SimulationDatabase(f"sqlite:///{db_path}")
    session = db.session_manager.get_session()

    try:
        # Compute all social metrics
        metrics = compute_all_social_metrics(session)

        # Convert to DataFrame for compatibility with analysis framework
        # Since social behavior analysis is complex and multi-dimensional,
        # we return a summary DataFrame
        summary_data = {
            'metric_type': [],
            'value': [],
            'description': []
        }

        # Extract key summary metrics
        if "summary" in metrics:
            summary = metrics["summary"]

            # Overall cooperation-competition ratio
            if "overall_coop_comp_ratio" in summary:
                ratio = summary["overall_coop_comp_ratio"]
                summary_data['metric_type'].append('overall_coop_comp_ratio')
                summary_data['value'].append(float(ratio) if ratio != float('inf') else 999)
                ratio_value = float(ratio) if ratio != float('inf') else 999
                summary_data['description'].append(f"Overall cooperation-competition ratio: {ratio_value:.3f}")

            # Total social interactions
            if "total_social_interactions" in summary:
                summary_data['metric_type'].append('total_social_interactions')
                summary_data['value'].append(summary["total_social_interactions"])
                summary_data['description'].append(f"Total social interactions: {summary['total_social_interactions']}")

            # Clustering ratio
            if "clustering_ratio" in summary:
                summary_data['metric_type'].append('clustering_ratio')
                summary_data['value'].append(summary["clustering_ratio"])
                summary_data['description'].append(f"Clustering ratio: {summary['clustering_ratio']:.3f}")

        # Add agent type summaries
        if "agent_type_summary" in summary:
            for agent_type, type_summary in summary["agent_type_summary"].items():
                for metric, value in type_summary.items():
                    summary_data['metric_type'].append(f"{agent_type}_{metric}")
                    summary_data['value'].append(float(value))
                    summary_data['description'].append(f"{agent_type} {metric.replace('_', ' ')}: {value:.3f}")

        df = pd.DataFrame(summary_data)
        return df

    finally:
        session.close()
        db.session_manager.close()


def load_social_behavior_data_from_db(db_path: str) -> dict:
    """Load raw social behavior data directly from database.

    Args:
        db_path: Path to database file

    Returns:
        Dictionary with raw social behavior data
    """
    db = SimulationDatabase(f"sqlite:///{db_path}")
    session = db.session_manager.get_session()

    try:
        # Compute comprehensive social metrics
        return compute_all_social_metrics(session)
    finally:
        session.close()
        db.session_manager.close()
