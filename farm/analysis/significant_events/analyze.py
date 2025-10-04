"""
Significant events analysis functions.
"""

import json
from pathlib import Path
from typing import Optional

from farm.analysis.common.context import AnalysisContext
from farm.analysis.significant_events.compute import (
    detect_significant_events,
    compute_event_severity,
    compute_event_patterns,
    compute_event_impact,
)


def analyze_significant_events(ctx: AnalysisContext, **kwargs) -> None:
    """Analyze significant events and save results.

    Args:
        ctx: Analysis context
        **kwargs: Additional parameters including db_connection, start_step, end_step, min_severity
    """
    db_connection = kwargs.get('db_connection')
    start_step = kwargs.get('start_step', 0)
    end_step = kwargs.get('end_step')
    min_severity = kwargs.get('min_severity', 0.3)

    ctx.logger.info(f"Analyzing significant events (severity >= {min_severity})...")

    # Detect events
    events = detect_significant_events(db_connection, start_step, end_step, min_severity)

    # Compute severity scores
    events_with_severity = compute_event_severity(events)

    # Filter by minimum severity
    significant_events = [e for e in events_with_severity if e.get('severity', 0) >= min_severity]

    # Save results
    results = {
        'total_events_detected': len(events),
        'significant_events': len(significant_events),
        'min_severity_threshold': min_severity,
        'events': significant_events,
    }

    output_file = ctx.get_output_file("significant_events.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    ctx.logger.info(f"Saved significant events to {output_file}")
    ctx.report_progress("Significant events analysis complete", 0.5)


def analyze_event_patterns(ctx: AnalysisContext, **kwargs) -> None:
    """Analyze patterns in significant events.

    Args:
        ctx: Analysis context
        **kwargs: Additional parameters including db_connection, start_step, end_step, min_severity
    """
    db_connection = kwargs.get('db_connection')
    start_step = kwargs.get('start_step', 0)
    end_step = kwargs.get('end_step')
    min_severity = kwargs.get('min_severity', 0.3)

    ctx.logger.info("Analyzing event patterns...")

    # Detect and score events
    events = detect_significant_events(db_connection, start_step, end_step, min_severity)
    events_with_severity = compute_event_severity(events)
    significant_events = [e for e in events_with_severity if e.get('severity', 0) >= min_severity]

    # Compute patterns
    patterns = compute_event_patterns(significant_events)

    output_file = ctx.get_output_file("event_patterns.json")
    with open(output_file, 'w') as f:
        json.dump(patterns, f, indent=2, default=str)
    ctx.logger.info(f"Saved event patterns to {output_file}")
    ctx.report_progress("Event patterns analysis complete", 0.5)


def analyze_event_impact(ctx: AnalysisContext, **kwargs) -> None:
    """Analyze the impact of significant events.

    Args:
        ctx: Analysis context
        **kwargs: Additional parameters including db_connection, start_step, end_step, min_severity
    """
    db_connection = kwargs.get('db_connection')
    start_step = kwargs.get('start_step', 0)
    end_step = kwargs.get('end_step')
    min_severity = kwargs.get('min_severity', 0.3)

    ctx.logger.info("Analyzing event impact...")

    # Detect and score events
    events = detect_significant_events(db_connection, start_step, end_step, min_severity)
    events_with_severity = compute_event_severity(events)
    significant_events = [e for e in events_with_severity if e.get('severity', 0) >= min_severity]

    # Compute impact
    impact = compute_event_impact(significant_events)

    output_file = ctx.get_output_file("event_impact.json")
    with open(output_file, 'w') as f:
        json.dump(impact, f, indent=2, default=str)
    ctx.logger.info(f"Saved event impact analysis to {output_file}")
    ctx.report_progress("Event impact analysis complete", 0.5)
