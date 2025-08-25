"""
Genesis Analysis Module

This module provides functions to analyze how initial states and conditions impact
simulation outcomes, identifying patterns and relationships between starting configurations
and eventual dominance.
"""

import glob
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from farm.analysis.genesis.compute import (
    compute_critical_period_metrics,
    compute_genesis_impact_scores,
    compute_initial_state_metrics,
    extract_features_from_metrics,
    transform_metrics_for_plotting,
)

logger = logging.getLogger(__name__)


def analyze_genesis_factors(session) -> Dict[str, Any]:
    """
    Analyze how initial conditions impact simulation outcomes for a single simulation.

    Parameters
    ----------
    session : SQLAlchemy session
        Database session for the simulation

    Returns
    -------
    Dict[str, Any]
        Dictionary containing analysis results
    """
    results = {}

    try:
        # Check database schema version
        inspector = inspect(session.bind)
        agents_columns = [col["name"] for col in inspector.get_columns("agents")]
        has_action_weights = "action_weights" in agents_columns

        # Add database schema info to results
        results["database_schema"] = {"has_action_weights_column": has_action_weights}

        # Log database schema information
        if has_action_weights:
            logger.info("Database schema has action_weights column")
        else:
            logger.warning(
                "Database schema does not have action_weights column. Some analysis features may be limited."
            )

        # Compute initial state metrics
        try:
            initial_metrics = compute_initial_state_metrics(session)
            # Transform metrics into plotting-friendly format
            initial_metrics = transform_metrics_for_plotting(initial_metrics)
            results["initial_metrics"] = initial_metrics
            logger.info(
                f"Initial metrics computed and transformed: {list(initial_metrics.keys())}"
            )
        except Exception as e:
            logger.error(f"Error computing initial state metrics: {e}")
            results["initial_metrics_error"] = str(e)
            results["initial_metrics"] = {}

        # Compute critical period metrics
        try:
            critical_period_metrics = compute_critical_period_metrics(session)
            results["critical_period"] = critical_period_metrics
        except Exception as e:
            logger.error(f"Error computing critical period metrics: {e}")
            results["critical_period_error"] = str(e)
            results["critical_period"] = {}

        # Compute genesis impact scores
        try:
            impact_scores = compute_genesis_impact_scores(session)
            results["impact_scores"] = impact_scores
        except Exception as e:
            logger.error(f"Error computing impact scores: {e}")
            results["impact_scores_error"] = str(e)
            results["impact_scores"] = {}

        # Extract key insights
        try:
            insights = extract_genesis_insights(
                results.get("initial_metrics", {}),
                results.get("critical_period", {}),
                results.get("impact_scores", {}),
            )
            results["insights"] = insights
        except Exception as e:
            logger.error(f"Error extracting insights: {e}")
            results["insights_error"] = str(e)
            results["insights"] = {}

    except Exception as e:
        logger.error(f"Unhandled error in genesis analysis: {e}")
        results["error"] = str(e)

    return results


def extract_genesis_insights(
    initial_metrics: Dict[str, Any],
    critical_period_metrics: Dict[str, Any],
    impact_scores: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Extract key insights from genesis analysis results.

    Parameters
    ----------
    initial_metrics : Dict[str, Any]
        Initial state metrics
    critical_period_metrics : Dict[str, Any]
        Critical period metrics
    impact_scores : Dict[str, Any]
        Genesis impact scores

    Returns
    -------
    Dict[str, Any]
        Dictionary containing key insights
    """
    insights = {}

    # Extract key initial advantages
    initial_advantages = extract_initial_advantages(initial_metrics)
    insights["initial_advantages"] = initial_advantages

    # Identify critical early events
    critical_events = identify_critical_events(critical_period_metrics)
    insights["critical_events"] = critical_events

    # Determine most impactful initial factors
    if "overall_impact_scores" in impact_scores:
        # Get top 5 most impactful factors
        impact_factors = impact_scores["overall_impact_scores"]
        sorted_factors = sorted(
            impact_factors.items(), key=lambda x: x[1], reverse=True
        )[:5]
        insights["top_impact_factors"] = dict(sorted_factors)

    # Determine if initial conditions were deterministic
    if "critical_period_dominant" in critical_period_metrics:
        critical_dominant = critical_period_metrics.get("critical_period_dominant")

        # Check if the agent type with initial advantages became dominant in critical period
        if initial_advantages and "dominant_agent_type" in initial_advantages:
            initial_dominant = initial_advantages["dominant_agent_type"]

            insights["initial_advantage_realized"] = (
                critical_dominant == initial_dominant
            )
            insights["determinism_level"] = (
                "high" if insights["initial_advantage_realized"] else "low"
            )

    return insights


def extract_initial_advantages(initial_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract key initial advantages from initial state metrics.

    Parameters
    ----------
    initial_metrics : Dict[str, Any]
        Initial state metrics

    Returns
    -------
    Dict[str, Any]
        Dictionary containing initial advantage information
    """
    advantages = {}

    # Extract resource proximity advantages
    if "initial_relative_advantages" in initial_metrics:
        relative_advantages = initial_metrics["initial_relative_advantages"]

        if "resource_proximity_advantage" in relative_advantages:
            proximity_advantages = relative_advantages["resource_proximity_advantage"]

            # Calculate net advantage for each agent type
            agent_types = set()
            for pair in proximity_advantages.keys():
                agent_types.update(pair.split("_vs_"))

            net_advantages = {agent_type: 0 for agent_type in agent_types}

            for pair, metrics in proximity_advantages.items():
                type1, type2 = pair.split("_vs_")

                # Resource count advantage
                if "resource_count_advantage" in metrics:
                    count_advantage = metrics["resource_count_advantage"]
                    if count_advantage > 0:
                        net_advantages[type1] += count_advantage
                    else:
                        net_advantages[type2] += abs(count_advantage)

                # Resource amount advantage
                if "resource_amount_advantage" in metrics:
                    amount_advantage = metrics["resource_amount_advantage"]
                    if amount_advantage > 0:
                        net_advantages[type1] += amount_advantage
                    else:
                        net_advantages[type2] += abs(amount_advantage)

            # Determine dominant agent type based on resource proximity
            if net_advantages:
                advantages["resource_proximity_net_advantages"] = net_advantages
                advantages["dominant_agent_type"] = max(
                    net_advantages, key=lambda x: net_advantages[x]
                )

    # Extract attribute advantages
    if "agent_starting_attributes" in initial_metrics:
        starting_attributes = initial_metrics["agent_starting_attributes"]

        # Compare initial resources
        initial_resources = {}
        for agent_type, attributes in starting_attributes.items():
            if "initial_resources" in attributes:
                initial_resources[agent_type] = attributes["initial_resources"].get(
                    "mean", 0
                )

        if initial_resources:
            advantages["initial_resources"] = initial_resources
            advantages["resource_advantage_type"] = max(
                initial_resources, key=lambda x: initial_resources[x]
            )

    return advantages


def identify_critical_events(
    critical_period_metrics: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Identify key events during the critical period that impact simulation outcomes.

    Parameters
    ----------
    critical_period_metrics : Dict[str, Any]
        Critical period metrics

    Returns
    -------
    List[Dict[str, Any]]
        List of critical events
    """
    critical_events = []

    # First reproduction events
    if "first_reproduction_events" in critical_period_metrics:
        first_reproductions = critical_period_metrics["first_reproduction_events"]

        if first_reproductions:
            # Find the agent type that reproduced first
            first_reproducer = min(first_reproductions.items(), key=lambda x: x[1])

            critical_events.append(
                {
                    "event_type": "first_reproduction",
                    "agent_type": first_reproducer[0],
                    "step": first_reproducer[1],
                    "description": f"{first_reproducer[0]} was first to reproduce at step {first_reproducer[1]}",
                }
            )

    # Early deaths
    if "early_deaths" in critical_period_metrics:
        early_deaths = critical_period_metrics["early_deaths"]

        if early_deaths:
            # Find the agent type with most early deaths
            most_deaths = max(early_deaths.items(), key=lambda x: x[1])

            if most_deaths[1] > 0:
                critical_events.append(
                    {
                        "event_type": "early_deaths",
                        "agent_type": most_deaths[0],
                        "count": most_deaths[1],
                        "description": f"{most_deaths[0]} suffered {most_deaths[1]} early deaths",
                    }
                )

    # Population growth rates
    growth_rates = {}
    for key, value in critical_period_metrics.items():
        if key.endswith("_growth_rate"):
            agent_type = key.replace("_growth_rate", "")
            growth_rates[agent_type] = value

    if growth_rates:
        # Find the agent type with highest growth rate
        highest_growth = max(growth_rates.items(), key=lambda x: x[1])

        if highest_growth[1] > 0:
            critical_events.append(
                {
                    "event_type": "rapid_growth",
                    "agent_type": highest_growth[0],
                    "growth_rate": highest_growth[1],
                    "description": f"{highest_growth[0]} achieved {highest_growth[1]:.2f}x growth in critical period",
                }
            )

    return critical_events


def analyze_genesis_across_simulations(experiment_path: str) -> Dict[str, Any]:
    """
    Analyze initial conditions and their impact across multiple simulations.

    Parameters
    ----------
    experiment_path : str
        Path to the experiment directory containing simulation data

    Returns
    -------
    Dict[str, Any]
        Dictionary containing cross-simulation analysis results
    """
    results = {
        "simulations": [],
        "predictive_models": {},
        "cross_simulation_patterns": {},
        "determinism_analysis": {},
    }

    # Find all simulation folders
    sim_folders = glob.glob(os.path.join(experiment_path, "iteration_*"))

    if not sim_folders:
        return {"error": f"No simulation folders found in {experiment_path}"}

    # Collect data from each simulation
    simulation_data = []

    for folder in sim_folders:
        db_path = os.path.join(folder, "simulation.db")

        if not os.path.exists(db_path):
            continue

        # Create database connection
        engine = create_engine(f"sqlite:///{db_path}")
        Session = sessionmaker(bind=engine)
        session = Session()

        try:
            # Analyze this simulation
            sim_results = analyze_genesis_factors(session)

            # Extract iteration number from folder name
            iteration = os.path.basename(folder).split("_")[1]

            # Store results
            simulation_data.append({"iteration": iteration, "results": sim_results})

        except Exception as e:
            logger.error(f"Error analyzing simulation in {folder}: {e}")
        finally:
            session.close()

    results["simulations"] = simulation_data

    # Perform cross-simulation analysis if we have enough data
    if len(simulation_data) >= 5:
        # Build predictive models
        predictive_models = build_predictive_models(simulation_data)
        results["predictive_models"] = predictive_models

        # Analyze patterns across simulations
        cross_sim_patterns = analyze_cross_simulation_patterns(simulation_data)
        results["cross_simulation_patterns"] = cross_sim_patterns

        # Analyze determinism of initial conditions
        determinism = analyze_determinism(simulation_data)
        results["determinism_analysis"] = determinism

    return results


def build_predictive_models(simulation_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build predictive models to forecast simulation outcomes based on initial conditions.

    Parameters
    ----------
    simulation_data : List[Dict[str, Any]]
        List of simulation results

    Returns
    -------
    Dict[str, Any]
        Dictionary containing predictive models and their performance
    """
    model_results = {}

    # Extract features and targets
    features = []
    dominance_targets = []
    survival_targets = []

    for sim in simulation_data:
        results = sim["results"]

        # Skip if missing required data
        if "initial_metrics" not in results or "insights" not in results:
            continue

        # Extract features from initial metrics
        sim_features = extract_features_from_metrics(results["initial_metrics"])

        # Extract dominance target
        if "insights" in results and "initial_advantages" in results["insights"]:
            advantages = results["insights"]["initial_advantages"]
            if "dominant_agent_type" in advantages:
                features.append(sim_features)
                dominance_targets.append(advantages["dominant_agent_type"])

        # Extract survival target from critical period
        if "critical_period" in results:
            critical_period = results["critical_period"]
            if "early_deaths" in critical_period:
                early_deaths = sum(critical_period["early_deaths"].values())
                survival_targets.append(early_deaths)

    # Build dominance prediction model if we have enough data
    if len(features) >= 5 and len(dominance_targets) >= 5:
        try:
            # Convert features to DataFrame
            feature_df = pd.DataFrame(features)

            # Handle missing values
            # First, check if we have any columns that are all NaN
            all_nan_cols = feature_df.columns[feature_df.isna().all()].tolist()
            if all_nan_cols:
                logger.warning(f"Dropping columns that are all NaN: {all_nan_cols}")
                feature_df = feature_df.drop(columns=all_nan_cols)

            # For remaining NaN values, use mean imputation
            feature_df = feature_df.fillna(feature_df.mean())

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                feature_df, dominance_targets, test_size=0.3, random_state=42
            )

            # Train model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # Evaluate model
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            # Get feature importance
            importance = model.feature_importances_
            feature_importance = {
                feature: float(imp)
                for feature, imp in zip(feature_df.columns, importance)
            }

            # Sort by importance
            sorted_importance = dict(
                sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[
                    :10
                ]
            )

            model_results["dominance_prediction"] = {
                "accuracy": accuracy,
                "feature_importance": sorted_importance,
            }
        except Exception as e:
            logger.error(f"Error building dominance prediction model: {e}")

    # Build survival prediction model if we have enough data
    if len(features) >= 5 and len(survival_targets) >= 5:
        try:
            # Convert features to DataFrame
            feature_df = pd.DataFrame(features)

            # Handle missing values
            # First, check if we have any columns that are all NaN
            all_nan_cols = feature_df.columns[feature_df.isna().all()].tolist()
            if all_nan_cols:
                logger.warning(f"Dropping columns that are all NaN: {all_nan_cols}")
                feature_df = feature_df.drop(columns=all_nan_cols)

            # For remaining NaN values, use mean imputation
            feature_df = feature_df.fillna(feature_df.mean())

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                feature_df, survival_targets, test_size=0.3, random_state=42
            )

            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # Evaluate model
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)

            # Get feature importance
            importance = model.feature_importances_
            feature_importance = {
                feature: float(imp)
                for feature, imp in zip(feature_df.columns, importance)
            }

            # Sort by importance
            sorted_importance = dict(
                sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[
                    :10
                ]
            )

            model_results["survival_prediction"] = {
                "r2_score": r2,
                "feature_importance": sorted_importance,
            }
        except Exception as e:
            logger.error(f"Error building survival prediction model: {e}")

    return model_results


def analyze_cross_simulation_patterns(
    simulation_data: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Analyze patterns in initial conditions and outcomes across simulations.

    Parameters
    ----------
    simulation_data : List[Dict[str, Any]]
        List of simulation results

    Returns
    -------
    Dict[str, Any]
        Dictionary containing cross-simulation patterns
    """
    patterns = {}

    # Extract initial advantage data
    advantage_data = []

    for sim in simulation_data:
        results = sim["results"]

        # Skip if missing required data
        if "insights" not in results or "initial_advantages" not in results["insights"]:
            continue

        advantages = results["insights"]["initial_advantages"]

        if (
            "dominant_agent_type" in advantages
            and "resource_proximity_net_advantages" in advantages
        ):
            advantage_data.append(
                {
                    "iteration": sim["iteration"],
                    "dominant_type": advantages["dominant_agent_type"],
                    "net_advantages": advantages["resource_proximity_net_advantages"],
                }
            )

    # Analyze advantage consistency
    if advantage_data:
        # Count occurrences of each dominant type
        dominant_counts = {}
        for data in advantage_data:
            dominant_type = data["dominant_type"]
            dominant_counts[dominant_type] = dominant_counts.get(dominant_type, 0) + 1

        # Calculate consistency percentage
        total_sims = len(advantage_data)
        consistency = {
            agent_type: count / total_sims
            for agent_type, count in dominant_counts.items()
        }

        patterns["advantage_consistency"] = consistency

        # Determine most consistently advantaged type
        if consistency:
            patterns["most_consistent_advantage"] = max(
                consistency.items(), key=lambda x: x[1]
            )

    # Analyze critical events consistency
    critical_events = []

    for sim in simulation_data:
        results = sim["results"]

        # Skip if missing required data
        if "insights" not in results or "critical_events" not in results["insights"]:
            continue

        events = results["insights"]["critical_events"]

        if events:
            critical_events.append({"iteration": sim["iteration"], "events": events})

    # Analyze first reproduction consistency
    if critical_events:
        first_reproduction = {}

        for data in critical_events:
            for event in data["events"]:
                if event["event_type"] == "first_reproduction":
                    agent_type = event["agent_type"]
                    first_reproduction[agent_type] = (
                        first_reproduction.get(agent_type, 0) + 1
                    )

        if first_reproduction:
            total_sims = len(critical_events)
            first_reproduction_consistency = {
                agent_type: count / total_sims
                for agent_type, count in first_reproduction.items()
            }

            patterns["first_reproduction_consistency"] = first_reproduction_consistency

    return patterns


def analyze_determinism(simulation_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze how deterministic initial conditions are in predicting simulation outcomes.

    Parameters
    ----------
    simulation_data : List[Dict[str, Any]]
        List of simulation results

    Returns
    -------
    Dict[str, Any]
        Dictionary containing determinism analysis
    """
    determinism = {}

    # Count how often initial advantage translates to critical period dominance
    advantage_realized_count = 0
    total_count = 0

    for sim in simulation_data:
        results = sim["results"]

        # Skip if missing required data
        if (
            "insights" not in results
            or "initial_advantage_realized" not in results["insights"]
        ):
            continue

        if results["insights"]["initial_advantage_realized"]:
            advantage_realized_count += 1

        total_count += 1

    # Calculate determinism percentage
    if total_count > 0:
        determinism["initial_advantage_realization_rate"] = (
            advantage_realized_count / total_count
        )

        # Classify determinism level
        rate = determinism["initial_advantage_realization_rate"]
        if rate >= 0.8:
            determinism["determinism_level"] = "very high"
        elif rate >= 0.6:
            determinism["determinism_level"] = "high"
        elif rate >= 0.4:
            determinism["determinism_level"] = "moderate"
        else:
            determinism["determinism_level"] = "low"

    return determinism


def analyze_critical_period(session, critical_period_end: int = 100) -> Dict[str, Any]:
    """
    Perform detailed analysis of the critical early period of a simulation.

    Parameters
    ----------
    session : SQLAlchemy session
        Database session for the simulation
    critical_period_end : int, optional
        The step number that marks the end of the critical period, by default 100

    Returns
    -------
    Dict[str, Any]
        Dictionary containing critical period analysis
    """
    results = {}

    # Compute critical period metrics
    metrics = compute_critical_period_metrics(session, critical_period_end)

    if "error" in metrics:
        return metrics

    results["metrics"] = metrics

    # Analyze population dynamics during critical period
    population_dynamics = analyze_critical_population_dynamics(metrics)
    results["population_dynamics"] = population_dynamics

    # Analyze resource acquisition during critical period
    resource_dynamics = analyze_critical_resource_dynamics(metrics)
    results["resource_dynamics"] = resource_dynamics

    # Analyze key events and their impact
    key_events = analyze_critical_events(metrics)
    results["key_events"] = key_events

    # Determine if critical period outcome predicts final outcome
    predictive_power = analyze_critical_period_predictive_power(
        session, metrics, critical_period_end
    )
    results["predictive_power"] = predictive_power

    return results


def analyze_critical_population_dynamics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze population dynamics during the critical period.

    Parameters
    ----------
    metrics : Dict[str, Any]
        Critical period metrics

    Returns
    -------
    Dict[str, Any]
        Dictionary containing population dynamics analysis
    """
    dynamics = {}

    # Extract growth rates
    growth_rates = {}
    for key, value in metrics.items():
        if key.endswith("_growth_rate"):
            agent_type = key.replace("_growth_rate", "")
            growth_rates[agent_type] = value

    if growth_rates:
        dynamics["growth_rates"] = growth_rates

        # Determine fastest growing agent type
        fastest_growing = max(growth_rates.items(), key=lambda x: x[1])
        dynamics["fastest_growing"] = {
            "agent_type": fastest_growing[0],
            "growth_rate": fastest_growing[1],
        }

    # Determine dominant agent type at end of critical period
    if "critical_period_dominant" in metrics:
        dynamics["dominant_type"] = metrics["critical_period_dominant"]

    return dynamics


def analyze_critical_resource_dynamics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze resource acquisition dynamics during the critical period.

    Parameters
    ----------
    metrics : Dict[str, Any]
        Critical period metrics

    Returns
    -------
    Dict[str, Any]
        Dictionary containing resource dynamics analysis
    """
    dynamics = {}

    # Extract average resources by agent type
    avg_resources = {}
    for key, value in metrics.items():
        if key.endswith("_avg_resources"):
            agent_type = key.replace("_avg_resources", "")
            avg_resources[agent_type] = value

    if avg_resources:
        dynamics["avg_resources"] = avg_resources

        # Determine agent type with most resources
        most_resources = max(avg_resources.items(), key=lambda x: x[1])
        dynamics["most_resources"] = {
            "agent_type": most_resources[0],
            "avg_resources": most_resources[1],
        }

    return dynamics


def analyze_critical_events(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze key events during the critical period.

    Parameters
    ----------
    metrics : Dict[str, Any]
        Critical period metrics

    Returns
    -------
    Dict[str, Any]
        Dictionary containing key events analysis
    """
    events = {}

    # Analyze first reproduction events
    if "first_reproduction_events" in metrics:
        first_reproductions = metrics["first_reproduction_events"]

        if first_reproductions:
            events["first_reproductions"] = first_reproductions

            # Find the agent type that reproduced first
            first_reproducer = min(first_reproductions.items(), key=lambda x: x[1])
            events["first_reproducer"] = {
                "agent_type": first_reproducer[0],
                "step": first_reproducer[1],
            }

    # Analyze early deaths
    if "early_deaths" in metrics:
        early_deaths = metrics["early_deaths"]

        if early_deaths:
            events["early_deaths"] = early_deaths

            # Find the agent type with most early deaths
            most_deaths = max(early_deaths.items(), key=lambda x: x[1])
            events["most_deaths"] = {
                "agent_type": most_deaths[0],
                "count": most_deaths[1],
            }

    # Analyze early social interactions
    if "early_social_interactions" in metrics:
        social_interactions = metrics["early_social_interactions"]

        if social_interactions:
            events["social_interactions"] = social_interactions

            # Calculate total interactions by agent type
            total_interactions = {}
            for agent_type, interactions in social_interactions.items():
                total_interactions[agent_type] = sum(interactions.values())

            if total_interactions:
                # Find the most social agent type
                most_social = max(total_interactions.items(), key=lambda x: x[1])
                events["most_social"] = {
                    "agent_type": most_social[0],
                    "interaction_count": most_social[1],
                }

    return events


def analyze_critical_period_predictive_power(
    session, metrics: Dict[str, Any], critical_period_end: int
) -> Dict[str, Any]:
    """
    Analyze how well the critical period predicts the final simulation outcome.

    Parameters
    ----------
    session : SQLAlchemy session
        Database session for the simulation
    metrics : Dict[str, Any]
        Critical period metrics
    critical_period_end : int
        The step number that marks the end of the critical period

    Returns
    -------
    Dict[str, Any]
        Dictionary containing predictive power analysis
    """
    from farm.database.models import SimulationStepModel

    predictive_power = {}

    # Get the critical period dominant type
    critical_dominant = metrics.get("critical_period_dominant")

    if not critical_dominant:
        return {"error": "No critical period dominant type found"}

    # Get the final simulation step
    final_step = (
        session.query(SimulationStepModel)
        .order_by(SimulationStepModel.step_number.desc())
        .first()
    )

    if not final_step:
        return {"error": "No final step found"}

    # Determine final dominant type
    agent_counts = {
        "SystemAgent": final_step.system_agents,
        "IndependentAgent": final_step.independent_agents,
        "ControlAgent": final_step.control_agents,
    }

    if any(agent_counts.values()):
        final_dominant = max(agent_counts, key=lambda x: agent_counts[x])

        # Check if critical period dominant matches final dominant
        prediction_correct = critical_dominant == final_dominant

        predictive_power["critical_dominant"] = critical_dominant
        predictive_power["final_dominant"] = final_dominant
        predictive_power["prediction_correct"] = prediction_correct

        # Calculate predictive accuracy
        predictive_power["predictive_accuracy"] = 1.0 if prediction_correct else 0.0

    return predictive_power
