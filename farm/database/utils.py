"""Utility functions for database operations."""

import json
from typing import Dict, Optional

import pandas as pd


def extract_agent_counts_from_json(agent_type_counts) -> Dict[str, int]:
    """Extract agent type counts from JSON column.
    
    Parameters
    ----------
    agent_type_counts : str, dict, or None
        JSON string or dict containing agent type counts
        
    Returns
    -------
    Dict[str, int]
        Dictionary mapping agent types to counts
    """
    if agent_type_counts is None:
        return {}
    
    if isinstance(agent_type_counts, dict):
        return agent_type_counts
    
    if isinstance(agent_type_counts, str):
        try:
            return json.loads(agent_type_counts)
        except (json.JSONDecodeError, TypeError):
            return {}
    
    return {}


def normalize_simulation_steps_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize simulation_steps DataFrame to extract agent counts from JSON.
    
    This function adds backwards-compatible columns (system_agents, independent_agents, control_agents)
    by extracting them from the agent_type_counts JSON column.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from simulation_steps table
        
    Returns
    -------
    pd.DataFrame
        DataFrame with agent type counts extracted to separate columns
    """
    if df.empty:
        return df
    
    # Check if agent_type_counts column exists
    if "agent_type_counts" not in df.columns:
        # If old columns exist, keep them (for backwards compatibility during migration)
        if all(col in df.columns for col in ["system_agents", "independent_agents", "control_agents"]):
            return df
        # Otherwise, add empty columns
        df["system_agents"] = 0
        df["independent_agents"] = 0
        df["control_agents"] = 0
        return df
    
    # Extract counts from JSON column
    def extract_counts(row):
        counts = extract_agent_counts_from_json(row.get("agent_type_counts"))
        return pd.Series({
            "system_agents": counts.get("system", 0),
            "independent_agents": counts.get("independent", 0),
            "control_agents": counts.get("control", 0),
        })
    
    # Extract agent counts
    counts_df = df.apply(extract_counts, axis=1)
    
    # Add columns, overwriting if they already exist
    for col in ["system_agents", "independent_agents", "control_agents"]:
        df[col] = counts_df[col]
    
    return df

