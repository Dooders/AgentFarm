"""
Combat data processing for analysis.
"""

from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd

from farm.database.session_manager import SessionManager
from farm.database.repositories.action_repository import ActionRepository
from farm.analysis.common.utils import find_database_path, is_successful_attack
from farm.utils.logging import get_logger

logger = get_logger(__name__)


def process_combat_data(experiment_path: Path, use_database: bool = True, **kwargs) -> pd.DataFrame:
    """Process combat data from experiment.

    Args:
        experiment_path: Path to experiment directory
        use_database: Whether to use database or direct file access
        **kwargs: Additional options

    Returns:
        DataFrame with combat metrics over time
    """
    # Try to load from database first, with robust fallback and logging
    df: Optional[pd.DataFrame] = None
    if use_database:
        db = None
        try:
            db_path = find_database_path(experiment_path, "simulation.db")
            logger.info(f"Loading combat actions from database: {db_path}")
            db_uri = f"sqlite:///{db_path}"
            session_manager = SessionManager(db_uri)
            action_repo = ActionRepository(session_manager)

            # Filter attack actions at database level
            attack_actions = action_repo.get_actions_by_scope("simulation", action_type="attack")

            combat_actions: List[Dict[str, Any]] = []
            for action in attack_actions:
                combat_actions.append(
                    {
                        "step": getattr(action, "step_number", 0),
                        "agent_id": getattr(action, "agent_id", None),
                        "action_type": getattr(action, "action_type", ""),
                        "reward": getattr(action, "reward", 0.0) or 0.0,
                        "details": getattr(action, "details", {}),
                    }
                )

            if combat_actions:
                df = pd.DataFrame(combat_actions)
            else:
                df = pd.DataFrame(columns=["step", "agent_id", "action_type", "reward", "details"])

        except Exception as e:
            logger.exception(f"Failed loading combat actions from database. Falling back to CSV. Error: {e}")
            df = None

    # Fallback to CSV files, if DB failed or returned empty
    if df is None or df.empty:
        data_dir = experiment_path / "data"
        csv_path = data_dir / "actions.csv"
        if csv_path.exists():
            logger.info(f"Loading actions from CSV fallback: {csv_path}")
            raw_df = pd.read_csv(csv_path)
            # Filter for combat actions
            if "action_type" in raw_df.columns:
                df = raw_df[raw_df["action_type"] == "attack"].copy()
            else:
                # No action type information; return empty combat data
                df = pd.DataFrame(columns=["step", "agent_id", "action_type", "reward", "details"])
        else:
            raise FileNotFoundError(f"No simulation database or actions.csv found in {experiment_path}")

    # Extract combat-specific details efficiently
    if "details" in df.columns:
        extracted = df["details"].apply(
            lambda x: (
                x.get("damage_dealt", 0.0) if isinstance(x, dict) else 0.0,
                x.get("target_id", None) if isinstance(x, dict) else None,
                x.get("attack_range", 0.0) if isinstance(x, dict) else 0.0,
            )
        )
        if not extracted.empty:
            df[["damage_dealt", "target_id", "attack_range"]] = pd.DataFrame(extracted.tolist(), index=df.index)
        else:
            df["damage_dealt"] = 0.0
            df["target_id"] = None
            df["attack_range"] = 0.0
        df.drop(columns=["details"], inplace=True, errors="ignore")
    else:
        # Ensure expected columns exist
        if "damage_dealt" not in df.columns:
            df["damage_dealt"] = 0.0
        if "target_id" not in df.columns:
            df["target_id"] = None
        if "attack_range" not in df.columns:
            df["attack_range"] = 0.0

    # Standardize column names used elsewhere
    if "step_number" in df.columns and "step" not in df.columns:
        df.rename(columns={"step_number": "step"}, inplace=True)

    return df


def process_combat_metrics_data(experiment_path: Path, use_database: bool = True, **kwargs) -> pd.DataFrame:
    """Process combat metrics by deriving from actions table.

    Args:
        experiment_path: Path to experiment directory
        use_database: Whether to use database or direct file access
        **kwargs: Additional options

    Returns:
        DataFrame with combat metrics over time (derived from actions table)
    """
    # Try DB first with safe fallback
    df: Optional[pd.DataFrame] = None
    if use_database:
        try:
            db_path = find_database_path(experiment_path, "simulation.db")
            logger.info(f"Loading combat metrics from database (derived from actions): {db_path}")
            db_uri = f"sqlite:///{db_path}"
            session_manager = SessionManager(db_uri)
            action_repo = ActionRepository(session_manager)

            # Get attack actions filtered at database level
            attack_actions = action_repo.get_actions_by_scope("simulation", action_type="attack")
            
            if attack_actions:
                # Convert to DataFrame for easier aggregation
                action_data = [
                    {
                        "step": getattr(a, "step_number", 0),
                        "reward": getattr(a, "reward", 0.0) or 0.0,
                    }
                    for a in attack_actions
                ]
                action_df = pd.DataFrame(action_data)
                
                # Aggregate by step
                grouped = action_df.groupby("step").agg(
                    combat_encounters=("step", "count"),  # Count of attacks = combat_encounters
                    successful_attacks=("reward", lambda x: sum(1 for r in x if is_successful_attack(r)) if len(x) > 0 else 0),  # Count successful attacks
                )
                
                # Add per-step columns (same as cumulative for now since we're aggregating per step)
                grouped["combat_encounters_this_step"] = grouped["combat_encounters"]
                grouped["successful_attacks_this_step"] = grouped["successful_attacks"]
                
                # Reset index to make step a column
                grouped.reset_index(inplace=True)
                df = grouped
            else:
                df = pd.DataFrame(columns=["step", "combat_encounters", "successful_attacks", 
                                         "combat_encounters_this_step", "successful_attacks_this_step"])
                
            session_manager.close()
        except Exception as e:
            logger.exception(f"Failed loading combat metrics from database. Falling back to CSV. Error: {e}")
            df = None

    if df is None or df.empty:
        data_dir = experiment_path / "data"
        # Try common metrics CSV filenames
        candidates = [
            data_dir / "step_metrics.csv",
            data_dir / "metrics.csv",
        ]
        csv_path = next((p for p in candidates if p.exists()), None)
        if csv_path is None:
            # No metrics fallback available; return empty frame with expected columns
            return pd.DataFrame(columns=["step", "combat_encounters", "successful_attacks", 
                                       "combat_encounters_this_step", "successful_attacks_this_step"])

        logger.info(f"Loading combat metrics from CSV fallback: {csv_path}")
        raw = pd.read_csv(csv_path)
        cols = {}
        # Map flexible column names if present
        cols["step"] = (
            "step"
            if "step" in raw.columns
            else next((c for c in raw.columns if c.lower() in {"timestep", "time", "iteration"}), None)
        )
        cols["combat_encounters"] = "combat_encounters" if "combat_encounters" in raw.columns else None
        cols["successful_attacks"] = "successful_attacks" if "successful_attacks" in raw.columns else None
        cols["combat_encounters_this_step"] = (
            "combat_encounters_this_step" if "combat_encounters_this_step" in raw.columns else None
        )
        cols["successful_attacks_this_step"] = (
            "successful_attacks_this_step" if "successful_attacks_this_step" in raw.columns else None
        )

        # Build the output using available columns
        out = pd.DataFrame()
        if cols["step"] is not None:
            out["step"] = raw[cols["step"]]
        else:
            out["step"] = range(len(raw))
        for k in [
            "combat_encounters",
            "successful_attacks",
            "combat_encounters_this_step",
            "successful_attacks_this_step",
        ]:
            if cols[k] is not None:
                out[k] = raw[cols[k]]
            else:
                out[k] = 0
        df = out

    return df


def process_agent_combat_stats(experiment_path: Path, agent_ids: Optional[List[int]] = None, **kwargs) -> pd.DataFrame:
    """Process agent-specific combat statistics.

    Args:
        experiment_path: Path to experiment directory
        agent_ids: Specific agent IDs to analyze (None for all)
        **kwargs: Additional options

    Returns:
        DataFrame with agent combat statistics
    """
    combat_df = process_combat_data(experiment_path)

    if combat_df.empty:
        return pd.DataFrame(columns=["agent_id", "total_attacks", "successful_attacks", "total_damage", "avg_damage"])

    # Group by agent and calculate stats
    agent_stats = []

    grouped = combat_df.groupby("agent_id")
    for agent_id, group in grouped:
        if agent_ids is None or agent_id in agent_ids:
            total_attacks = len(group)
            successful_attacks = (group["damage_dealt"] > 0).sum()
            total_damage = group["damage_dealt"].sum()
            avg_damage = group["damage_dealt"].mean()
            avg_reward = group["reward"].mean()

            agent_stats.append(
                {
                    "agent_id": agent_id,
                    "total_attacks": total_attacks,
                    "successful_attacks": successful_attacks,
                    "success_rate": successful_attacks / total_attacks if total_attacks > 0 else 0.0,
                    "total_damage": total_damage,
                    "avg_damage": avg_damage,
                    "avg_reward": avg_reward,
                }
            )

    return pd.DataFrame(agent_stats)
