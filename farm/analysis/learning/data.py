"""
Learning data processing for analysis.
"""

import json
from pathlib import Path
from typing import Optional
import pandas as pd

from farm.database.session_manager import SessionManager
from farm.database.repositories.learning_repository import LearningRepository
from farm.database.database import SimulationDatabase
from farm.utils.logging import get_logger

logger = get_logger(__name__)


def process_learning_data(experiment_path: Path, use_database: bool = True, **kwargs) -> pd.DataFrame:
    """Process learning data from experiment.

    Args:
        experiment_path: Path to experiment directory
        use_database: Whether to use database or direct file access
        **kwargs: Additional options

    Returns:
        DataFrame with learning metrics over time
    """
    # Try to load from database first
    df = None
    try:
        # Find simulation database
        db_path = experiment_path / "simulation.db"
        if not db_path.exists():
            # Try alternative locations
            db_path = experiment_path / "data" / "simulation.db"

        if db_path.exists():
            # Load data using existing infrastructure
            db_uri = f"sqlite:///{db_path}"
            session_manager = SessionManager(db_uri)
            repository = LearningRepository(session_manager)

            # Get learning experiences (now ActionModel instances with module_type)
            experiences = repository.get_learning_experiences(session_manager.create_session(), scope="simulation")

            if not experiences:
                # Return empty DataFrame with correct structure
                df = pd.DataFrame(
                    columns=[
                        "step",
                        "agent_id",
                        "module_type",
                        "reward",
                        "action_taken",
                        "action_taken_mapped",
                        "state_delta",
                        "loss",
                    ]
                )
            else:
                # Convert ActionModel instances to dict with proper column mapping
                rows = []
                for exp in experiences:
                    row = {
                        "step": exp.step_number,
                        "agent_id": exp.agent_id,
                        "module_type": exp.module_type,
                        "module_id": exp.module_id,
                        "reward": exp.reward,
                        "action_type": exp.action_type,
                        "action_taken": None,
                        "action_taken_mapped": None,
                    }
                    # Extract action_taken and action_taken_mapped from details JSON
                    if exp.details:
                        try:
                            details = json.loads(exp.details) if isinstance(exp.details, str) else exp.details
                            row["action_taken"] = details.get("action_taken")
                            row["action_taken_mapped"] = details.get("action_taken_mapped")
                        except (json.JSONDecodeError, TypeError) as json_error:
                            logger.warning(
                                "learning_data_json_parse_failed",
                                agent_id=exp.agent_id,
                                step=exp.step_number,
                                details=exp.details,
                                error_type=type(json_error).__name__,
                                error_message=str(json_error),
                            )
                    rows.append(row)
                
                df = pd.DataFrame(rows)

                # Ensure required columns exist
                required_cols = ["step", "agent_id", "module_type", "reward", "action_taken"]
                for col in required_cols:
                    if col not in df.columns:
                        df[col] = None

                # Add derived columns for analysis
                if "reward" in df.columns:
                    df["reward_ma"] = df.groupby("agent_id")["reward"].transform(
                        lambda x: x.rolling(window=10, min_periods=1).mean()
                    )

    except Exception as e:
        # If database loading fails, try to load from CSV files
        logger.warning(
            "learning_data_database_load_failed",
            experiment_path=str(experiment_path),
            error_type=type(e).__name__,
            error_message=str(e),
        )

    if df is None or df.empty:
        # For learning analysis, we don't have CSV fallback files available
        # Return an empty DataFrame with correct structure
        df = pd.DataFrame(
            columns=[
                "step",
                "agent_id",
                "module_type",
                "reward",
                "action_taken",
                "action_taken_mapped",
                "state_delta",
                "loss",
            ]
        )

    return df


def process_learning_progress_data(experiment_path: Path) -> pd.DataFrame:
    """Process learning progress data for time series analysis.

    Args:
        experiment_path: Path to experiment directory

    Returns:
        DataFrame with learning progress metrics
    """
    # Try to load from database first
    df = None
    try:
        db_path = experiment_path / "simulation.db"
        if not db_path.exists():
            db_path = experiment_path / "data" / "simulation.db"

        if db_path.exists():
            db = SimulationDatabase(f"sqlite:///{db_path}")
            repository = LearningRepository(db.session_manager)

            try:
                # Get learning progress
                progress_list = repository.get_learning_progress(
                    repository.session_manager.create_session(), scope="simulation"
                )

                if not progress_list:
                    df = pd.DataFrame(columns=["step", "reward", "action_count", "unique_actions"])
                else:
                    # Convert to DataFrame
                    df = pd.DataFrame(
                        [
                            {
                                "step": p.step,
                                "reward": p.reward,
                                "action_count": p.action_count,
                                "unique_actions": p.unique_actions,
                            }
                            for p in progress_list
                        ]
                    )

            finally:
                db.close()
    except Exception as e:
        # If database loading fails, try to load from CSV files
        logger.warning(
            "aggregate_learning_data_database_load_failed",
            experiment_path=str(experiment_path),
            error_type=type(e).__name__,
            error_message=str(e),
        )

    if df is None:
        # Return empty DataFrame with correct structure
        df = pd.DataFrame(columns=["step", "reward", "action_count", "unique_actions"])

    return df
