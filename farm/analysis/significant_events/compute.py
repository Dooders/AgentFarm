"""
Significant events statistical computations.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from pathlib import Path
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_

from farm.analysis.common.utils import calculate_statistics
from farm.database.models import (
    AgentModel,
    AgentStateModel,
    SimulationStepModel,
    ResourceModel,
    HealthIncident,
    ReproductionEventModel,
    ActionModel,
)


def detect_significant_events(
    db_connection, start_step: int = 0, end_step: Optional[int] = None, min_severity: float = 0.3
) -> List[Dict[str, Any]]:
    """Detect significant events from simulation database.

    Args:
        db_connection: Database connection to simulation data (SessionManager or Session)
        start_step: Starting step for analysis
        end_step: Ending step for analysis (optional)
        min_severity: Minimum severity threshold for events

    Returns:
        List of detected significant events with fields:
        - type: Event type (agent_death, agent_birth, resource_depletion, population_crash, 
                health_critical, mass_combat, resource_boom)
        - step: Simulation step when event occurred
        - impact_scale: Quantified impact (0.0-1.0)
        - details: Additional event-specific information
    """
    events = []
    
    # Check if db_connection has execute_with_retry (SessionManager) or is a Session
    if hasattr(db_connection, 'execute_with_retry'):
        session_manager = db_connection
        def query_func(func):
            return session_manager.execute_with_retry(func)
    else:
        # Assume it's a Session object
        session = db_connection
        def query_func(func):
            return func(session)
    
    # Detect agent deaths
    events.extend(_detect_agent_deaths(query_func, start_step, end_step))
    
    # Detect agent births
    events.extend(_detect_agent_births(query_func, start_step, end_step))
    
    # Detect population crashes and booms
    events.extend(_detect_population_changes(query_func, start_step, end_step))
    
    # Detect critical health incidents
    events.extend(_detect_critical_health_incidents(query_func, start_step, end_step))
    
    # Detect mass combat events
    events.extend(_detect_mass_combat_events(query_func, start_step, end_step))
    
    # Detect resource depletion events
    events.extend(_detect_resource_depletion(query_func, start_step, end_step))

    return events


def _detect_agent_deaths(query_func, start_step: int, end_step: Optional[int]) -> List[Dict[str, Any]]:
    """Detect agent death events."""
    def query(session: Session) -> List[Dict[str, Any]]:
        q = session.query(
            AgentModel.agent_id,
            AgentModel.death_time,
            AgentModel.agent_type,
            AgentModel.generation,
        ).filter(
            AgentModel.death_time.isnot(None),
            AgentModel.death_time >= start_step
        )
        
        if end_step is not None:
            q = q.filter(AgentModel.death_time <= end_step)
        
        results = q.all()
        
        events = []
        for agent_id, death_time, agent_type, generation in results:
            # Calculate impact based on agent type and generation
            # Higher generation and certain types have higher impact
            impact_scale = 0.4  # Base impact
            if agent_type == 'system':
                impact_scale += 0.2
            if generation and generation > 5:
                impact_scale += min(0.3, generation * 0.03)
            
            impact_scale = min(1.0, impact_scale)
            
            events.append({
                'type': 'agent_death',
                'step': death_time,
                'impact_scale': impact_scale,
                'details': {
                    'agent_id': agent_id,
                    'agent_type': agent_type,
                    'generation': generation,
                }
            })
        
        return events
    
    return query_func(query)


def _detect_agent_births(query_func, start_step: int, end_step: Optional[int]) -> List[Dict[str, Any]]:
    """Detect agent birth events."""
    def query(session: Session) -> List[Dict[str, Any]]:
        q = session.query(
            ReproductionEventModel.step_number,
            ReproductionEventModel.parent_id,
            ReproductionEventModel.offspring_id,
            ReproductionEventModel.success,
            ReproductionEventModel.offspring_generation,
        ).filter(
            ReproductionEventModel.success == True,
            ReproductionEventModel.step_number >= start_step
        )
        
        if end_step is not None:
            q = q.filter(ReproductionEventModel.step_number <= end_step)
        
        results = q.all()
        
        events = []
        for step_number, parent_id, offspring_id, success, generation in results:
            # Calculate impact based on generation
            impact_scale = 0.2  # Base impact for births
            if generation and generation > 5:
                impact_scale += min(0.3, generation * 0.02)
            
            impact_scale = min(1.0, impact_scale)
            
            events.append({
                'type': 'agent_birth',
                'step': step_number,
                'impact_scale': impact_scale,
                'details': {
                    'offspring_id': offspring_id,
                    'parent_id': parent_id,
                    'generation': generation,
                }
            })
        
        return events
    
    return query_func(query)


def _detect_population_changes(query_func, start_step: int, end_step: Optional[int]) -> List[Dict[str, Any]]:
    """Detect population crashes and booms."""
    def query(session: Session) -> List[Dict[str, Any]]:
        q = session.query(
            SimulationStepModel.step_number,
            SimulationStepModel.total_agents,
            SimulationStepModel.births,
            SimulationStepModel.deaths,
        ).filter(
            SimulationStepModel.step_number >= start_step
        )
        
        if end_step is not None:
            q = q.filter(SimulationStepModel.step_number <= end_step)
        
        q = q.order_by(SimulationStepModel.step_number)
        results = q.all()
        
        events = []
        prev_population = None
        
        for step_number, total_agents, births, deaths in results:
            if prev_population is not None and total_agents > 0:
                # Calculate population change rate
                change_rate = abs(total_agents - prev_population) / prev_population
                
                # Detect crashes (>30% decrease)
                if total_agents < prev_population and change_rate > 0.3:
                    impact_scale = min(1.0, change_rate)
                    events.append({
                        'type': 'population_crash',
                        'step': step_number,
                        'impact_scale': impact_scale,
                        'details': {
                            'population_before': prev_population,
                            'population_after': total_agents,
                            'change_rate': change_rate,
                            'deaths': deaths,
                        }
                    })
                
                # Detect booms (>40% increase)
                elif total_agents > prev_population and change_rate > 0.4:
                    impact_scale = min(1.0, change_rate * 0.7)  # Booms slightly less impactful
                    events.append({
                        'type': 'population_boom',
                        'step': step_number,
                        'impact_scale': impact_scale,
                        'details': {
                            'population_before': prev_population,
                            'population_after': total_agents,
                            'change_rate': change_rate,
                            'births': births,
                        }
                    })
            
            prev_population = total_agents if total_agents else prev_population
        
        return events
    
    return query_func(query)


def _detect_critical_health_incidents(query_func, start_step: int, end_step: Optional[int]) -> List[Dict[str, Any]]:
    """Detect critical health incidents."""
    def query(session: Session) -> List[Dict[str, Any]]:
        q = session.query(
            HealthIncident.step_number,
            HealthIncident.agent_id,
            HealthIncident.health_before,
            HealthIncident.health_after,
            HealthIncident.cause,
        ).filter(
            HealthIncident.step_number >= start_step
        )
        
        if end_step is not None:
            q = q.filter(HealthIncident.step_number <= end_step)
        
        results = q.all()
        
        events = []
        for step_number, agent_id, health_before, health_after, cause in results:
            # Detect critical health drops (>50% health loss or drops below 20%)
            if health_before and health_after is not None:
                health_drop = health_before - health_after
                drop_rate = health_drop / health_before if health_before > 0 else 0
                
                is_critical = (drop_rate > 0.5) or (health_after < 20)
                
                if is_critical:
                    impact_scale = min(1.0, 0.4 + drop_rate * 0.5)
                    
                    events.append({
                        'type': 'health_critical',
                        'step': step_number,
                        'impact_scale': impact_scale,
                        'details': {
                            'agent_id': agent_id,
                            'health_before': health_before,
                            'health_after': health_after,
                            'cause': cause,
                            'drop_rate': drop_rate,
                        }
                    })
        
        return events
    
    return query_func(query)


def _detect_mass_combat_events(query_func, start_step: int, end_step: Optional[int]) -> List[Dict[str, Any]]:
    """Detect mass combat events."""
    def query(session: Session) -> List[Dict[str, Any]]:
        q = session.query(
            SimulationStepModel.step_number,
            SimulationStepModel.combat_encounters_this_step,
            SimulationStepModel.successful_attacks_this_step,
            SimulationStepModel.total_agents,
        ).filter(
            SimulationStepModel.step_number >= start_step,
            SimulationStepModel.combat_encounters_this_step > 0
        )
        
        if end_step is not None:
            q = q.filter(SimulationStepModel.step_number <= end_step)
        
        results = q.all()
        
        events = []
        for step_number, combat_encounters, successful_attacks, total_agents in results:
            # Detect mass combat (>20% of population involved or >10 encounters)
            if total_agents and total_agents > 0:
                combat_rate = combat_encounters / total_agents
                
                is_mass_combat = (combat_rate > 0.2) or (combat_encounters > 10)
                
                if is_mass_combat:
                    impact_scale = min(1.0, 0.5 + combat_rate * 0.5)
                    
                    events.append({
                        'type': 'mass_combat',
                        'step': step_number,
                        'impact_scale': impact_scale,
                        'details': {
                            'combat_encounters': combat_encounters,
                            'successful_attacks': successful_attacks,
                            'total_agents': total_agents,
                            'combat_rate': combat_rate,
                        }
                    })
        
        return events
    
    return query_func(query)


def _detect_resource_depletion(query_func, start_step: int, end_step: Optional[int]) -> List[Dict[str, Any]]:
    """Detect resource depletion events."""
    def query(session: Session) -> List[Dict[str, Any]]:
        # Get resource states and detect when resources drop to very low levels
        q = session.query(
            SimulationStepModel.step_number,
            SimulationStepModel.total_resources,
            SimulationStepModel.average_agent_resources,
            SimulationStepModel.total_agents,
        ).filter(
            SimulationStepModel.step_number >= start_step
        )
        
        if end_step is not None:
            q = q.filter(SimulationStepModel.step_number <= end_step)
        
        q = q.order_by(SimulationStepModel.step_number)
        results = q.all()
        
        events = []
        prev_resources = None
        
        for step_number, total_resources, avg_resources, total_agents in results:
            # Detect severe resource drops or critically low resources
            if prev_resources is not None and prev_resources > 0:
                drop_rate = (prev_resources - total_resources) / prev_resources
                
                # Critical resource depletion (>60% drop or avg < 5 per agent)
                is_critical = (drop_rate > 0.6) or (avg_resources is not None and avg_resources < 5)
                
                if is_critical and total_resources is not None:
                    impact_scale = 0.6
                    if drop_rate > 0:
                        impact_scale = min(1.0, 0.5 + drop_rate * 0.5)
                    elif avg_resources is not None and avg_resources < 5:
                        impact_scale = min(1.0, 0.7 + (5 - avg_resources) * 0.1)
                    
                    events.append({
                        'type': 'resource_depletion',
                        'step': step_number,
                        'impact_scale': impact_scale,
                        'details': {
                            'total_resources_before': prev_resources,
                            'total_resources_after': total_resources,
                            'average_per_agent': avg_resources,
                            'drop_rate': drop_rate if drop_rate > 0 else 0,
                        }
                    })
            
            prev_resources = total_resources if total_resources is not None else prev_resources
        
        return events
    
    return query_func(query)


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
            "agent_death": 0.5,
            "agent_birth": 0.3,
            "resource_depletion": 0.8,
            "population_crash": 0.9,
            "population_boom": 0.6,
            "health_critical": 0.7,
            "mass_combat": 0.8,
            "environmental_change": 0.6,
        }.get(event.get("type", "unknown"), 0.1)

        # Modify by scale/impact
        impact_multiplier = event.get("impact_scale", 1.0)
        # Ensure impact_multiplier is non-negative to avoid negative severity
        impact_multiplier = max(0.0, impact_multiplier)
        severity = min(1.0, base_severity * impact_multiplier)

        event["severity"] = severity
        event["severity_category"] = "high" if severity > 0.7 else "medium" if severity > 0.4 else "low"

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

    if "step" in df.columns:
        # Event frequency over time
        event_counts = df.groupby("step").size()
        patterns["event_frequency"] = calculate_statistics(event_counts.values)

        # Time between events
        if len(df) > 1:
            time_diffs = np.diff(sorted(df["step"].values))
            patterns["inter_event_times"] = calculate_statistics(time_diffs)

    # Event type distribution
    if "type" in df.columns:
        type_counts = df["type"].value_counts()
        patterns["event_types"] = type_counts.to_dict()

    # Severity distribution
    if "severity" in df.columns:
        severity_values = df["severity"].values
        patterns["severity_distribution"] = calculate_statistics(severity_values)

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
    if "type" in df.columns and "impact_scale" in df.columns:
        impact_by_type = df.groupby("type")["impact_scale"].agg(["mean", "std", "count"])
        impact["impact_by_type"] = impact_by_type.to_dict("index")

    # Overall impact statistics
    if "impact_scale" in df.columns:
        impact_scales = df["impact_scale"].values
        impact["overall_impact"] = calculate_statistics(impact_scales)

    return impact
