"""
Significant events statistical computations.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from pathlib import Path

from farm.analysis.common.utils import calculate_statistics


def detect_significant_events(db_connection, start_step: int = 0, end_step: Optional[int] = None,
                            min_severity: float = 0.3) -> List[Dict[str, Any]]:
    """Detect significant events from simulation database.

    Args:
        db_connection: Database connection to simulation data
        start_step: Starting step for analysis
        end_step: Ending step for analysis (optional)
        min_severity: Minimum severity threshold for events

    Returns:
        List of detected significant events
    """
    # This is a placeholder implementation
    # In a real implementation, this would query the database for:
    # - Agent deaths, births, major state changes
    # - Resource depletion events
    # - Population crashes/booms
    # - Unusual action patterns
    # - Environmental changes

    events = []

    # Mock some example events for now
    # TODO: Implement actual database queries
    # Example event data structure for future implementation
    # Each event should be a dictionary with at least the following keys:
    # - 'type': str, one of ['agent_death', 'agent_birth', 'resource_depletion', 'population_crash', 'environmental_change']
    # - 'step': int, simulation step when the event occurred
    # - 'impact_scale': float, quantifies the impact (e.g., 0.0–1.0)
    # - 'details': dict, additional event-specific information
    events = [
        {
            'type': 'agent_death',
            'step': 120,
            'impact_scale': 0.7,
            'details': {'agent_id': 42, 'cause': 'starvation'}
        },
        {
            'type': 'resource_depletion',
            'step': 135,
            'impact_scale': 0.9,
            'details': {'resource': 'water', 'remaining': 0}
        },
        {
            'type': 'population_crash',
            'step': 150,
            'impact_scale': 1.0,
            'details': {'population_before': 200, 'population_after': 50}
        },
        {
            'type': 'environmental_change',
            'step': 160,
            'impact_scale': 0.6,
            'details': {'change': 'temperature_drop', 'delta': -5}
        },
        {
            'type': 'agent_birth',
            'step': 170,
            'impact_scale': 0.3,
            'details': {'agent_id': 99, 'parent_id': 42}
        }
    ]
    # TODO: Implement actual database queries to replace this example data

    return events


def compute_event_severity(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compute severity scores for events.

    Args:
        events: List of event dictionaries

    Returns:
        Events with severity scores added
    """
    for event in events:
        # Calculate severity based on event type and impact
        base_severity = {
            'agent_death': 0.5,
            'agent_birth': 0.3,
            'resource_depletion': 0.8,
            'population_crash': 0.9,
            'environmental_change': 0.6,
        }.get(event.get('type', 'unknown'), 0.1)

        # Modify by scale/impact
        impact_multiplier = event.get('impact_scale', 1.0)
        severity = min(1.0, base_severity * impact_multiplier)

        event['severity'] = severity
        event['severity_category'] = 'high' if severity > 0.7 else 'medium' if severity > 0.4 else 'low'

    return events


def compute_event_patterns(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute patterns in event sequences.

    Args:
        events: List of events with timestamps

    Returns:
        Dictionary of pattern statistics
    """
    if not events:
        return {}

    patterns = {}

    # Convert to DataFrame for analysis
    df = pd.DataFrame(events)

    if 'step' in df.columns:
        # Event frequency over time
        event_counts = df.groupby('step').size()
        patterns['event_frequency'] = calculate_statistics(event_counts.values)

        # Time between events
        if len(df) > 1:
            time_diffs = np.diff(sorted(df['step'].values))
            patterns['inter_event_times'] = calculate_statistics(time_diffs)

    # Event type distribution
    if 'type' in df.columns:
        type_counts = df['type'].value_counts()
        patterns['event_types'] = type_counts.to_dict()

    # Severity distribution
    if 'severity' in df.columns:
        severity_values = df['severity'].values
        patterns['severity_distribution'] = calculate_statistics(severity_values)

    return patterns


def compute_event_impact(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute the impact of events on simulation metrics.

    Args:
        events: List of events

    Returns:
        Dictionary of impact analysis results
    """
    impact = {}

    if not events:
        return impact

    df = pd.DataFrame(events)

    # Group by event type and compute average impact
    if 'type' in df.columns and 'impact_scale' in df.columns:
        impact_by_type = df.groupby('type')['impact_scale'].agg(['mean', 'std', 'count'])
        impact['impact_by_type'] = impact_by_type.to_dict('index')

    # Overall impact statistics
    if 'impact_scale' in df.columns:
        impact_scales = df['impact_scale'].values
        impact['overall_impact'] = calculate_statistics(impact_scales)

    return impact
