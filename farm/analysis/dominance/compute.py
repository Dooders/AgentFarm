from farm.database.models import AgentModel, SimulationStepModel


def compute_population_dominance(sim_session):
    """
    Compute the dominant agent type by final population.
    Query the final simulation step and choose the type with the highest count.
    """
    final_step = (
        sim_session.query(SimulationStepModel)
        .order_by(SimulationStepModel.step_number.desc())
        .first()
    )
    if final_step is None:
        return None
    # Create a dictionary of agent counts
    counts = {
        "system": final_step.system_agents,
        "independent": final_step.independent_agents,
        "control": final_step.control_agents,
    }
    # Return the key with the maximum count
    return max(counts, key=counts.get)


def compute_survival_dominance(sim_session):
    """
    Compute the dominant agent type by average survival time.
    For each agent, compute survival time as (death_time - birth_time) if the agent has died.
    (For agents still alive, use the final step as a proxy)
    Then, for each agent type, compute the average survival time.
    Return the type with the highest average.
    """
    agents = sim_session.query(AgentModel).all()

    # Get the final step number for calculating survival of still-alive agents
    final_step = (
        sim_session.query(SimulationStepModel)
        .order_by(SimulationStepModel.step_number.desc())
        .first()
    )
    final_step_number = final_step.step_number if final_step else 0

    survival_by_type = {}
    count_by_type = {}
    for agent in agents:
        # For alive agents, use the final step as the death time
        if agent.death_time is not None:
            survival = agent.death_time - agent.birth_time
        else:
            survival = final_step_number - agent.birth_time

        survival_by_type.setdefault(agent.agent_type, 0)
        count_by_type.setdefault(agent.agent_type, 0)
        survival_by_type[agent.agent_type] += survival
        count_by_type[agent.agent_type] += 1

    avg_survival = {
        agent_type: (survival_by_type[agent_type] / count_by_type[agent_type])
        for agent_type in survival_by_type
        if count_by_type[agent_type] > 0
    }
    if not avg_survival:
        return None
    return max(avg_survival, key=avg_survival.get)


def compute_dominance_switches(sim_session):
    """
    Analyze how often agent types switch dominance during a simulation.

    This function examines the entire simulation history to identify:
    1. Total number of dominance switches
    2. Average duration of dominance periods for each agent type
    3. Volatility of dominance (frequency of switches in different phases)
    4. Transition matrix showing which agent types tend to take over from others

    Returns a dictionary with dominance switching statistics.
    """
    # Query all simulation steps ordered by step number
    sim_steps = (
        sim_session.query(SimulationStepModel)
        .order_by(SimulationStepModel.step_number.asc())
        .all()
    )

    if not sim_steps:
        return None

    # Initialize tracking variables
    agent_types = ["system", "independent", "control"]
    current_dominant = None
    previous_dominant = None
    dominance_periods = {agent_type: [] for agent_type in agent_types}
    switches = []
    transition_matrix = {
        from_type: {to_type: 0 for to_type in agent_types} for from_type in agent_types
    }

    # Track the current dominance period
    period_start_step = 0
    total_steps = len(sim_steps)

    # Process each simulation step
    for step_idx, step in enumerate(sim_steps):
        # Determine which type is dominant in this step
        counts = {
            "system": step.system_agents,
            "independent": step.independent_agents,
            "control": step.control_agents,
        }

        # Skip steps with no agents
        if sum(counts.values()) == 0:
            continue

        current_dominant = max(counts, key=counts.get)

        # If this is the first step with agents, initialize
        if previous_dominant is None:
            previous_dominant = current_dominant
            period_start_step = step_idx
            continue

        # Check if dominance has switched
        if current_dominant != previous_dominant:
            # Record the switch
            switches.append(
                {
                    "step": step.step_number,
                    "from": previous_dominant,
                    "to": current_dominant,
                    "phase": (
                        "early"
                        if step_idx < total_steps / 3
                        else "middle" if step_idx < 2 * total_steps / 3 else "late"
                    ),
                }
            )

            # Update transition matrix
            transition_matrix[previous_dominant][current_dominant] += 1

            # Record the duration of the completed dominance period
            period_duration = step_idx - period_start_step
            dominance_periods[previous_dominant].append(period_duration)

            # Reset for the new period
            period_start_step = step_idx
            previous_dominant = current_dominant

    # Record the final dominance period
    if previous_dominant is not None:
        final_period_duration = total_steps - period_start_step
        dominance_periods[previous_dominant].append(final_period_duration)

    # Calculate average dominance period durations
    avg_dominance_periods = {}
    for agent_type in agent_types:
        periods = dominance_periods[agent_type]
        avg_dominance_periods[agent_type] = (
            sum(periods) / len(periods) if periods else 0
        )

    # Calculate phase-specific switch counts
    phase_switches = {
        "early": sum(1 for s in switches if s["phase"] == "early"),
        "middle": sum(1 for s in switches if s["phase"] == "middle"),
        "late": sum(1 for s in switches if s["phase"] == "late"),
    }

    # Calculate normalized transition probabilities
    transition_probabilities = {from_type: {} for from_type in agent_types}

    for from_type in agent_types:
        total_transitions = sum(transition_matrix[from_type].values())
        for to_type in agent_types:
            transition_probabilities[from_type][to_type] = (
                transition_matrix[from_type][to_type] / total_transitions
                if total_transitions > 0
                else 0
            )

    # Return comprehensive results
    return {
        "total_switches": len(switches),
        "switches_per_step": len(switches) / total_steps if total_steps > 0 else 0,
        "switches_detail": switches,
        "avg_dominance_periods": avg_dominance_periods,
        "phase_switches": phase_switches,
        "transition_matrix": transition_matrix,
        "transition_probabilities": transition_probabilities,
    }


def compute_comprehensive_dominance(sim_session):
    """
    Compute a comprehensive dominance score that considers the entire simulation history.

    This function uses multiple metrics to determine dominance:
    1. Area Under the Curve (AUC): Total agent-steps throughout the simulation
    2. Recency-weighted AUC: Gives more weight to later steps in the simulation
    3. Dominance duration: How many steps each agent type was dominant
    4. Growth trend: Positive growth trends in the latter half of simulation
    5. Final population ratio: The proportion of agents at the end of simulation

    Returns a dictionary with dominance scores for each agent type and the overall dominant type.
    """
    # Query all simulation steps ordered by step number
    sim_steps = (
        sim_session.query(SimulationStepModel)
        .order_by(SimulationStepModel.step_number.asc())
        .all()
    )

    if not sim_steps:
        return None

    # Initialize metrics
    agent_types = ["system", "independent", "control"]
    total_steps = len(sim_steps)

    # Calculate Area Under the Curve (agent-steps)
    auc = {agent_type: 0 for agent_type in agent_types}

    # Recency-weighted AUC (recent steps count more)
    recency_weighted_auc = {agent_type: 0 for agent_type in agent_types}

    # Count how many steps each type was dominant
    dominance_duration = {agent_type: 0 for agent_type in agent_types}

    # Track agent counts for calculating trends
    agent_counts = {agent_type: [] for agent_type in agent_types}

    # Process each simulation step
    for i, step in enumerate(sim_steps):
        # Calculate recency weight (later steps count more)
        recency_weight = 1 + (i / total_steps)

        # Update metrics for each agent type
        for agent_type in agent_types:
            agent_count = getattr(step, f"{agent_type}_agents")

            # Basic AUC - sum of agent counts across all steps
            auc[agent_type] += agent_count

            # Recency-weighted AUC
            recency_weighted_auc[agent_type] += agent_count * recency_weight

            # Track agent counts for trend analysis
            agent_counts[agent_type].append(agent_count)

        # Determine which type was dominant in this step
        counts = {
            "system": step.system_agents,
            "independent": step.independent_agents,
            "control": step.control_agents,
        }
        dominant_type = max(counts, key=counts.get) if any(counts.values()) else None
        if dominant_type:
            dominance_duration[dominant_type] += 1

    # Calculate growth trends in the latter half
    growth_trends = {}
    for agent_type in agent_types:
        counts = agent_counts[agent_type]
        if len(counts) >= 4:  # Need at least a few points for meaningful trend
            # Focus on the latter half of the simulation
            latter_half = counts[len(counts) // 2 :]

            if all(x == 0 for x in latter_half):
                growth_trends[agent_type] = 0
            else:
                # Simple trend calculation: last value compared to average of latter half
                latter_half_avg = sum(latter_half) / len(latter_half)
                if latter_half_avg == 0:
                    growth_trends[agent_type] = 0
                else:
                    growth_trends[agent_type] = (
                        latter_half[-1] - latter_half_avg
                    ) / latter_half_avg
        else:
            growth_trends[agent_type] = 0

    # Calculate final population ratios
    final_step = sim_steps[-1]
    total_final_agents = final_step.total_agents
    final_ratios = {}

    if total_final_agents > 0:
        for agent_type in agent_types:
            agent_count = getattr(final_step, f"{agent_type}_agents")
            final_ratios[agent_type] = agent_count / total_final_agents
    else:
        final_ratios = {agent_type: 0 for agent_type in agent_types}

    # Normalize metrics to [0,1] scale
    normalized_metrics = {}

    for metric_name, metric_values in [
        ("auc", auc),
        ("recency_weighted_auc", recency_weighted_auc),
        ("dominance_duration", dominance_duration),
    ]:
        total = sum(metric_values.values())
        if total > 0:
            normalized_metrics[metric_name] = {
                agent_type: value / total for agent_type, value in metric_values.items()
            }
        else:
            normalized_metrics[metric_name] = {
                agent_type: 0 for agent_type in agent_types
            }

    # Calculate final composite score with weights for different metrics
    weights = {
        "auc": 0.2,  # Basic population persistence
        "recency_weighted_auc": 0.3,  # Emphasize later simulation stages
        "dominance_duration": 0.2,  # Reward consistent dominance
        "growth_trend": 0.1,  # Reward positive growth in latter half
        "final_ratio": 0.2,  # Reward final state
    }

    composite_scores = {agent_type: 0 for agent_type in agent_types}

    for agent_type in agent_types:
        composite_scores[agent_type] = (
            weights["auc"] * normalized_metrics["auc"][agent_type]
            + weights["recency_weighted_auc"]
            * normalized_metrics["recency_weighted_auc"][agent_type]
            + weights["dominance_duration"]
            * normalized_metrics["dominance_duration"][agent_type]
            + weights["growth_trend"]
            * (max(0, growth_trends[agent_type]))  # Only count positive growth
            + weights["final_ratio"] * final_ratios[agent_type]
        )

    # Determine overall dominant type
    dominant_type = (
        max(composite_scores, key=composite_scores.get)
        if any(composite_scores.values())
        else None
    )

    # Return comprehensive results
    return {
        "dominant_type": dominant_type,
        "scores": composite_scores,
        "metrics": {
            "auc": auc,
            "recency_weighted_auc": recency_weighted_auc,
            "dominance_duration": dominance_duration,
            "growth_trends": growth_trends,
            "final_ratios": final_ratios,
        },
        "normalized_metrics": normalized_metrics,
    }


def compute_dominance_switch_factors(df):
    """
    Calculate factors that correlate with dominance switching patterns.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with simulation analysis results

    Returns
    -------
    dict
        Dictionary with analysis results
    """
    import logging

    import pandas as pd

    if df.empty or "total_switches" not in df.columns:
        logging.warning("No dominance switch data available for analysis")
        return None

    results = {}

    # 1. Correlation between initial conditions and switching frequency
    initial_condition_cols = [
        col
        for col in df.columns
        if any(x in col for x in ["initial_", "resource_", "proximity"])
    ]

    if initial_condition_cols and len(df) > 5:
        # Calculate correlations with total switches
        corr_with_switches = (
            df[initial_condition_cols + ["total_switches"]]
            .corr()["total_switches"]
            .drop("total_switches")
        )

        # Get top positive and negative correlations
        top_positive = corr_with_switches.sort_values(ascending=False).head(5)
        top_negative = corr_with_switches.sort_values().head(5)

        results["top_positive_correlations"] = top_positive.to_dict()
        results["top_negative_correlations"] = top_negative.to_dict()

        logging.info("\nFactors associated with MORE dominance switching:")
        for factor, corr in top_positive.items():
            if abs(corr) > 0.1:  # Only report meaningful correlations
                logging.info(f"  {factor}: {corr:.3f}")

        logging.info("\nFactors associated with LESS dominance switching:")
        for factor, corr in top_negative.items():
            if abs(corr) > 0.1:  # Only report meaningful correlations
                logging.info(f"  {factor}: {corr:.3f}")

    # 2. Relationship between switching and final dominance
    if "comprehensive_dominance" in df.columns:
        # Average switches by dominant type
        switches_by_dominant = df.groupby("comprehensive_dominance")[
            "total_switches"
        ].mean()
        results["switches_by_dominant_type"] = switches_by_dominant.to_dict()

        logging.info("\nAverage dominance switches by final dominant type:")
        for agent_type, avg_switches in switches_by_dominant.items():
            logging.info(f"  {agent_type}: {avg_switches:.2f}")

    # 3. Relationship between switching and reproduction metrics
    reproduction_cols = [col for col in df.columns if "reproduction" in col]
    if reproduction_cols and len(df) > 5:
        # Calculate correlations with total switches
        repro_corr = (
            df[reproduction_cols + ["total_switches"]]
            .corr()["total_switches"]
            .drop("total_switches")
        )

        # Get top correlations (absolute value)
        top_repro_corr = repro_corr.abs().sort_values(ascending=False).head(5)
        top_repro_factors = repro_corr[top_repro_corr.index]

        results["reproduction_correlations"] = top_repro_factors.to_dict()

        logging.info("\nReproduction factors most associated with dominance switching:")
        for factor, corr in top_repro_factors.items():
            if abs(corr) > 0.1:  # Only report meaningful correlations
                direction = "more" if corr > 0 else "fewer"
                logging.info(f"  {factor}: {corr:.3f} ({direction} switches)")

    return results


def aggregate_reproduction_analysis_results(df, numeric_repro_cols):
    """
    Aggregate results from multiple reproduction analysis functions.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with simulation analysis results
    numeric_repro_cols : list
        List of numeric reproduction column names

    Returns
    -------
    dict
        Dictionary with aggregated reproduction analysis results
    """
    import logging

    import pandas as pd

    from farm.analysis.dominance.analyze import (
        analyze_by_agent_type,
        analyze_high_vs_low_switching,
        analyze_reproduction_advantage,
        analyze_reproduction_efficiency,
        analyze_reproduction_timing,
    )

    # Check if df is a DataFrame
    if not isinstance(df, pd.DataFrame):
        logging.warning(
            "Input to aggregate_reproduction_analysis_results is not a DataFrame"
        )
        return {}

    results = {}

    # Analyze high vs low switching groups
    high_low_results = analyze_high_vs_low_switching(df, numeric_repro_cols)
    if isinstance(high_low_results, dict) and high_low_results:
        results.update(high_low_results)

    # Analyze first reproduction timing
    timing_results = analyze_reproduction_timing(df, numeric_repro_cols)
    if isinstance(timing_results, dict) and timing_results:
        results.update(timing_results)

    # Analyze reproduction efficiency
    efficiency_results = analyze_reproduction_efficiency(df, numeric_repro_cols)
    if isinstance(efficiency_results, dict) and efficiency_results:
        results.update(efficiency_results)

    # Analyze reproduction advantage
    advantage_results = analyze_reproduction_advantage(df, numeric_repro_cols)
    if isinstance(advantage_results, dict) and advantage_results:
        results.update(advantage_results)

    # Analyze by agent type
    agent_type_results = analyze_by_agent_type(df, numeric_repro_cols)
    if isinstance(agent_type_results, dict) and agent_type_results:
        results.update(agent_type_results)

    return results
