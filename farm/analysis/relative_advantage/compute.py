"""
Relative Advantage Computation Module

This module provides functions to compute various relative advantages between agent types
throughout the simulation history. It analyzes multiple dimensions of advantage including
resource acquisition, reproduction, survival, population growth, and combat.
"""

import logging
from typing import Dict, Any, List, Tuple, Optional
import time

from sqlalchemy import func


def compute_relative_advantages(sim_session, focus_agent_type=None):
    """Calculate comprehensive relative advantages between agent types.
    
    This function analyzes various dimensions of relative advantage between agent types
    and returns a detailed breakdown of advantages in different categories.
    
    Parameters
    ----------
    sim_session : SQLAlchemy session
        Database session for the simulation
    focus_agent_type : str, optional
        If provided, calculate advantages relative to this specific agent type
        
    Returns
    -------
    dict
        Dictionary containing relative advantage metrics
    """
    start_time = time.time()
    logging.debug("Starting compute_relative_advantages")
    
    from farm.database.models import (
        AgentModel, 
        AgentStateModel, 
        SimulationStepModel,
        ResourceModel,
        ActionModel,
        ReproductionEventModel
    )
    
    results = {}
    agent_types = ["system", "independent", "control"]
    
    # If focus type is provided, only compare against that type
    if focus_agent_type and focus_agent_type in agent_types:
        comparison_pairs = [(t, focus_agent_type) for t in agent_types if t != focus_agent_type]
    else:
        # All pairwise comparisons
        comparison_pairs = [(t1, t2) 
                           for i, t1 in enumerate(agent_types) 
                           for t2 in agent_types[i+1:]]
    
    logging.debug(f"Calculating advantages for {len(comparison_pairs)} agent pairs")
    
    # Get data needed for analysis
    logging.debug("Getting simulation step data")
    max_step = sim_session.query(func.max(SimulationStepModel.step_number)).scalar() or 0
    early_phase_end = max_step // 3
    mid_phase_end = 2 * max_step // 3
    logging.debug(f"Simulation phases: early (1-{early_phase_end}), mid ({early_phase_end+1}-{mid_phase_end}), late ({mid_phase_end+1}-{max_step})")

    # 1. Resource Acquisition Advantage
    # ---------------------------------
    logging.debug("Calculating resource acquisition advantages")
    category_start_time = time.time()
    results["resource_acquisition"] = {}
    
    # Calculate average resource level over time for each agent type
    for agent_type in agent_types:
        # Average resource level in early phase
        early_avg_resource = sim_session.query(
            func.avg(AgentStateModel.resource_level)
        ).join(
            AgentModel, AgentStateModel.agent_id == AgentModel.agent_id
        ).filter(
            AgentModel.agent_type == agent_type,
            AgentStateModel.step_number <= early_phase_end
        ).scalar() or 0
        
        # Average resource level in middle phase
        mid_avg_resource = sim_session.query(
            func.avg(AgentStateModel.resource_level)
        ).join(
            AgentModel, AgentStateModel.agent_id == AgentModel.agent_id
        ).filter(
            AgentModel.agent_type == agent_type,
            AgentStateModel.step_number > early_phase_end,
            AgentStateModel.step_number <= mid_phase_end
        ).scalar() or 0
        
        # Late phase average resources
        late_avg_resource = sim_session.query(
            func.avg(AgentStateModel.resource_level)
        ).join(
            AgentModel, AgentStateModel.agent_id == AgentModel.agent_id
        ).filter(
            AgentModel.agent_type == agent_type,
            AgentStateModel.step_number > mid_phase_end
        ).scalar() or 0
        
        results["resource_acquisition"][agent_type] = {
            "early_phase": early_avg_resource,
            "mid_phase": mid_avg_resource,
            "late_phase": late_avg_resource
        }
    
    # Calculate resource acquisition advantages
    for type1, type2 in comparison_pairs:
        # Early phase advantage
        early_adv = (results["resource_acquisition"][type1]["early_phase"] - 
                     results["resource_acquisition"][type2]["early_phase"])
        
        # Mid phase advantage
        mid_adv = (results["resource_acquisition"][type1]["mid_phase"] - 
                  results["resource_acquisition"][type2]["mid_phase"])
        
        # Late phase advantage
        late_adv = (results["resource_acquisition"][type1]["late_phase"] - 
                   results["resource_acquisition"][type2]["late_phase"])
        
        # Overall advantage trajectory (positive means advantage is increasing)
        advantage_trajectory = late_adv - early_adv
        
        key = f"{type1}_vs_{type2}"
        results["resource_acquisition"][key] = {
            "early_phase_advantage": early_adv,
            "mid_phase_advantage": mid_adv,
            "late_phase_advantage": late_adv,
            "advantage_trajectory": advantage_trajectory
        }
    
    category_duration = time.time() - category_start_time
    logging.debug(f"Completed resource acquisition advantages in {category_duration:.2f}s")
    
    # 2. Reproduction Advantage
    # ------------------------
    logging.debug("Calculating reproduction advantages")
    category_start_time = time.time()
    results["reproduction"] = {}
    
    # Calculate reproduction metrics for each agent type
    for agent_type in agent_types:
        # Count successful reproductions
        reproductions = sim_session.query(
            func.count(ReproductionEventModel.event_id)
        ).join(
            AgentModel, ReproductionEventModel.parent_id == AgentModel.agent_id
        ).filter(
            AgentModel.agent_type == agent_type,
            ReproductionEventModel.success == True
        ).scalar() or 0
        
        # Count total reproduction attempts
        attempts = sim_session.query(
            func.count(ReproductionEventModel.event_id)
        ).join(
            AgentModel, ReproductionEventModel.parent_id == AgentModel.agent_id
        ).filter(
            AgentModel.agent_type == agent_type
        ).scalar() or 0
        
        # Calculate reproduction success rate
        success_rate = reproductions / max(attempts, 1)
        
        # Calculate reproduction efficiency (offspring per resource spent)
        resource_spent = sim_session.query(
            func.sum(ReproductionEventModel.parent_resources_before - 
                    ReproductionEventModel.parent_resources_after)
        ).join(
            AgentModel, ReproductionEventModel.parent_id == AgentModel.agent_id
        ).filter(
            AgentModel.agent_type == agent_type,
            ReproductionEventModel.success == True
        ).scalar() or 0
        
        reproduction_efficiency = reproductions / max(resource_spent, 1)
        
        # Calculate average time to first reproduction
        first_repro_time = sim_session.query(
            func.min(ReproductionEventModel.step_number)
        ).join(
            AgentModel, ReproductionEventModel.parent_id == AgentModel.agent_id
        ).filter(
            AgentModel.agent_type == agent_type,
            ReproductionEventModel.success == True
        ).scalar()
        
        results["reproduction"][agent_type] = {
            "success_rate": success_rate,
            "total_offspring": reproductions,
            "reproduction_efficiency": reproduction_efficiency,
            "first_reproduction_time": first_repro_time or float('inf')
        }
    
    # Calculate reproduction advantages
    for type1, type2 in comparison_pairs:
        # Success rate advantage
        rate_adv = (results["reproduction"][type1]["success_rate"] - 
                    results["reproduction"][type2]["success_rate"])
        
        # Efficiency advantage
        efficiency_adv = (results["reproduction"][type1]["reproduction_efficiency"] - 
                         results["reproduction"][type2]["reproduction_efficiency"])
        
        # First reproduction time advantage (negative is better - reproduced earlier)
        time1 = results["reproduction"][type1]["first_reproduction_time"]
        time2 = results["reproduction"][type2]["first_reproduction_time"]
        
        if time1 != float('inf') and time2 != float('inf'):
            time_adv = time2 - time1  # Positive means type1 reproduced earlier
        elif time1 != float('inf'):
            time_adv = float('inf')  # Type1 reproduced but type2 didn't
        elif time2 != float('inf'):
            time_adv = float('-inf')  # Type2 reproduced but type1 didn't
        else:
            time_adv = 0  # Neither reproduced
        
        key = f"{type1}_vs_{type2}"
        results["reproduction"][key] = {
            "success_rate_advantage": rate_adv,
            "efficiency_advantage": efficiency_adv,
            "first_reproduction_advantage": time_adv
        }
    
    category_duration = time.time() - category_start_time
    logging.debug(f"Completed reproduction advantages in {category_duration:.2f}s")
    
    # 3. Survival Advantage
    # --------------------
    logging.debug("Calculating survival advantages")
    category_start_time = time.time()
    results["survival"] = {}
    
    # Calculate survival metrics for each agent type
    for agent_type in agent_types:
        # Average lifespan of dead agents
        avg_lifespan = sim_session.query(
            func.avg(AgentModel.death_time - AgentModel.birth_time)
        ).filter(
            AgentModel.agent_type == agent_type,
            AgentModel.death_time != None
        ).scalar() or 0
        
        # Survival rate (percentage of agents still alive)
        total_agents = sim_session.query(
            func.count(AgentModel.agent_id)
        ).filter(
            AgentModel.agent_type == agent_type
        ).scalar() or 0
        
        alive_agents = sim_session.query(
            func.count(AgentModel.agent_id)
        ).filter(
            AgentModel.agent_type == agent_type,
            AgentModel.death_time == None
        ).scalar() or 0
        
        survival_rate = alive_agents / max(total_agents, 1)
        
        results["survival"][agent_type] = {
            "average_lifespan": avg_lifespan,
            "survival_rate": survival_rate
        }
    
    # Calculate survival advantages
    for type1, type2 in comparison_pairs:
        # Lifespan advantage
        lifespan_adv = (results["survival"][type1]["average_lifespan"] - 
                       results["survival"][type2]["average_lifespan"])
        
        # Survival rate advantage
        rate_adv = (results["survival"][type1]["survival_rate"] - 
                   results["survival"][type2]["survival_rate"])
        
        key = f"{type1}_vs_{type2}"
        results["survival"][key] = {
            "lifespan_advantage": lifespan_adv,
            "survival_rate_advantage": rate_adv
        }
    
    category_duration = time.time() - category_start_time
    logging.debug(f"Completed survival advantages in {category_duration:.2f}s")
    
    # 4. Population Growth Advantage
    # -----------------------------
    logging.debug("Calculating population growth advantages")
    category_start_time = time.time()
    results["population_growth"] = {}
    
    # Get population counts over time by phase
    for agent_type in agent_types:
        # Early phase average population
        early_avg_pop = sim_session.query(
            func.avg(getattr(SimulationStepModel, f"{agent_type}_agents"))
        ).filter(
            SimulationStepModel.step_number <= early_phase_end
        ).scalar() or 0
        
        # Mid phase average population
        mid_avg_pop = sim_session.query(
            func.avg(getattr(SimulationStepModel, f"{agent_type}_agents"))
        ).filter(
            SimulationStepModel.step_number > early_phase_end,
            SimulationStepModel.step_number <= mid_phase_end
        ).scalar() or 0
        
        # Late phase average population
        late_avg_pop = sim_session.query(
            func.avg(getattr(SimulationStepModel, f"{agent_type}_agents"))
        ).filter(
            SimulationStepModel.step_number > mid_phase_end
        ).scalar() or 0
        
        # Calculate growth rates between phases
        early_to_mid_growth = (mid_avg_pop - early_avg_pop) / max(early_avg_pop, 1)
        mid_to_late_growth = (late_avg_pop - mid_avg_pop) / max(mid_avg_pop, 1)
        overall_growth = (late_avg_pop - early_avg_pop) / max(early_avg_pop, 1)
        
        results["population_growth"][agent_type] = {
            "early_phase_population": early_avg_pop,
            "mid_phase_population": mid_avg_pop,
            "late_phase_population": late_avg_pop,
            "early_to_mid_growth": early_to_mid_growth,
            "mid_to_late_growth": mid_to_late_growth,
            "overall_growth_rate": overall_growth
        }
    
    # Calculate population growth advantages
    for type1, type2 in comparison_pairs:
        # Early phase population advantage
        early_adv = (results["population_growth"][type1]["early_phase_population"] - 
                     results["population_growth"][type2]["early_phase_population"])
        
        # Mid phase population advantage
        mid_adv = (results["population_growth"][type1]["mid_phase_population"] - 
                   results["population_growth"][type2]["mid_phase_population"])
        
        # Late phase population advantage
        late_adv = (results["population_growth"][type1]["late_phase_population"] - 
                    results["population_growth"][type2]["late_phase_population"])
        
        # Growth rate advantage
        growth_adv = (results["population_growth"][type1]["overall_growth_rate"] - 
                      results["population_growth"][type2]["overall_growth_rate"])
        
        key = f"{type1}_vs_{type2}"
        results["population_growth"][key] = {
            "early_phase_advantage": early_adv,
            "mid_phase_advantage": mid_adv,
            "late_phase_advantage": late_adv,
            "growth_rate_advantage": growth_adv,
            "advantage_trajectory": late_adv - early_adv
        }
    
    category_duration = time.time() - category_start_time
    logging.debug(f"Completed population growth advantages in {category_duration:.2f}s")
    
    # 5. Combat and Competition Advantage
    # ----------------------------------
    if hasattr(ActionModel, 'action_type'):  # Make sure the schema has this field
        logging.debug("Calculating combat advantages")
        category_start_time = time.time()
        results["combat"] = {}
        
        # Calculate combat metrics for each agent type
        for agent_type in agent_types:
            # Attack success rate
            attack_attempts = sim_session.query(
                func.count(ActionModel.action_id)
            ).join(
                AgentModel, ActionModel.agent_id == AgentModel.agent_id
            ).filter(
                AgentModel.agent_type == agent_type,
                ActionModel.action_type == 'attack'
            ).scalar() or 0
            
            successful_attacks = sim_session.query(
                func.count(ActionModel.action_id)
            ).join(
                AgentModel, ActionModel.agent_id == AgentModel.agent_id
            ).filter(
                AgentModel.agent_type == agent_type,
                ActionModel.action_type == 'attack',
                ActionModel.reward > 0  # Assuming positive reward means successful attack
            ).scalar() or 0
            
            attack_success_rate = successful_attacks / max(attack_attempts, 1)
            
            # Defense success rate (survival when targeted)
            times_targeted = sim_session.query(
                func.count(ActionModel.action_id)
            ).join(
                AgentModel, ActionModel.action_target_id == AgentModel.agent_id
            ).filter(
                AgentModel.agent_type == agent_type,
                ActionModel.action_type == 'attack'
            ).scalar() or 0
            
            successful_defenses = sim_session.query(
                func.count(ActionModel.action_id)
            ).join(
                AgentModel, ActionModel.action_target_id == AgentModel.agent_id
            ).filter(
                AgentModel.agent_type == agent_type,
                ActionModel.action_type == 'attack',
                ActionModel.reward <= 0  # Assuming non-positive reward means defense succeeded
            ).scalar() or 0
            
            defense_success_rate = successful_defenses / max(times_targeted, 1)
            
            results["combat"][agent_type] = {
                "attack_success_rate": attack_success_rate,
                "defense_success_rate": defense_success_rate,
                "attack_attempts": attack_attempts,
                "times_targeted": times_targeted
            }
        
        # Calculate combat advantages
        for type1, type2 in comparison_pairs:
            # Attack success advantage
            attack_adv = (results["combat"][type1]["attack_success_rate"] - 
                         results["combat"][type2]["attack_success_rate"])
            
            # Defense success advantage
            defense_adv = (results["combat"][type1]["defense_success_rate"] - 
                          results["combat"][type2]["defense_success_rate"])
            
            # Direct combat advantage (attack success against this specific opponent)
            # This is a simplification - in a real implementation, you'd need to join more carefully
            try:
                t1_vs_t2_attacks = sim_session.query(
                    func.count(ActionModel.action_id)
                ).join(
                    AgentModel, ActionModel.agent_id == AgentModel.agent_id
                ).filter(
                    AgentModel.agent_type == type1,
                    ActionModel.action_type == 'attack',
                    ActionModel.action_target_id.in_(
                        sim_session.query(AgentModel.agent_id).filter(
                            AgentModel.agent_type == type2
                        )
                    )
                ).scalar() or 0
                
                t1_vs_t2_success = sim_session.query(
                    func.count(ActionModel.action_id)
                ).join(
                    AgentModel, ActionModel.agent_id == AgentModel.agent_id
                ).filter(
                    AgentModel.agent_type == type1,
                    ActionModel.action_type == 'attack',
                    ActionModel.action_target_id.in_(
                        sim_session.query(AgentModel.agent_id).filter(
                            AgentModel.agent_type == type2
                        )
                    ),
                    ActionModel.reward > 0
                ).scalar() or 0
                
                direct_combat_success_rate = t1_vs_t2_success / max(t1_vs_t2_attacks, 1)
            except Exception as e:
                logging.warning(f"Error calculating direct combat stats: {e}")
                direct_combat_success_rate = 0.5  # Neutral value
            
            key = f"{type1}_vs_{type2}"
            results["combat"][key] = {
                "attack_success_advantage": attack_adv,
                "defense_success_advantage": defense_adv,
                "direct_combat_success_rate": direct_combat_success_rate
            }
        
        category_duration = time.time() - category_start_time
        logging.debug(f"Completed combat advantages in {category_duration:.2f}s")
    else:
        logging.debug("Skipping combat advantages (action_type field not found)")
    
    # 6. Initial Positioning Advantage
    # -------------------------------
    # This is a special case - we need to get data from the beginning of the simulation
    logging.debug("Calculating initial positioning advantages")
    category_start_time = time.time()
    results["initial_positioning"] = {}
    
    # Get the first step in the simulation
    first_step = sim_session.query(SimulationStepModel).order_by(SimulationStepModel.step_number.asc()).first()
    if first_step:
        # For each agent type, calculate starting positions
        for type1, type2 in comparison_pairs:
            key = f"{type1}_vs_{type2}"
            results["initial_positioning"][key] = {}
            
            # We need to look for specific metrics that were precomputed during simulation
            # These would typically be calculated from initial positions and saved
            # This is a simplification - in a real implementation, you'd compute these from raw data
            adv_fields = [
                (f"{type1}_vs_{type2}_nearest_resource_advantage", "resource_proximity_advantage"),
                (f"{type1}_vs_{type2}_resources_in_range_advantage", "resources_in_range_advantage"),
                (f"{type1}_vs_{type2}_resource_amount_advantage", "resource_amount_advantage")
            ]
            
            # In a real implementation, these would come from initial position calculations
            # Here we're just providing a placeholder
            for db_field, result_field in adv_fields:
                results["initial_positioning"][key][result_field] = 0
    
    category_duration = time.time() - category_start_time
    logging.debug(f"Completed initial positioning advantages in {category_duration:.2f}s")
    
    # 7. Composite Relative Advantage Score
    # ------------------------------------
    logging.debug("Calculating composite advantage scores")
    category_start_time = time.time()
    
    # Define weights for different advantage categories
    weights = {
        "resource_acquisition": 0.25,
        "reproduction": 0.25,
        "survival": 0.2,
        "population_growth": 0.2,
        "combat": 0.1 if "combat" in results else 0,
        "initial_positioning": 0 if "initial_positioning" not in results else 0.15
    }
    
    # Normalize weights to sum to 1
    weight_sum = sum(weights.values())
    weights = {k: v/weight_sum for k, v in weights.items()}
    
    results["composite_advantage"] = {}
    
    # Calculate composite advantage scores for each pair
    for type1, type2 in comparison_pairs:
        key = f"{type1}_vs_{type2}"
        composite_score = 0
        advantage_components = {}
        
        # Resource acquisition component
        if "resource_acquisition" in results and key in results["resource_acquisition"]:
            resource_adv = (
                results["resource_acquisition"][key]["early_phase_advantage"] * 0.3 +
                results["resource_acquisition"][key]["mid_phase_advantage"] * 0.3 +
                results["resource_acquisition"][key]["late_phase_advantage"] * 0.4
            )
            composite_score += weights["resource_acquisition"] * resource_adv
            advantage_components["resource_acquisition"] = resource_adv
        
        # Reproduction component
        if "reproduction" in results and key in results["reproduction"]:
            repro_adv = (
                results["reproduction"][key]["success_rate_advantage"] * 0.4 +
                results["reproduction"][key]["efficiency_advantage"] * 0.3 +
                (1 if results["reproduction"][key]["first_reproduction_advantage"] > 0 else 
                 0 if results["reproduction"][key]["first_reproduction_advantage"] == 0 else -1) * 0.3
            )
            composite_score += weights["reproduction"] * repro_adv
            advantage_components["reproduction"] = repro_adv
        
        # Survival component
        if "survival" in results and key in results["survival"]:
            survival_adv = (
                results["survival"][key]["lifespan_advantage"] * 0.5 +
                results["survival"][key]["survival_rate_advantage"] * 0.5
            )
            # Normalize extremely large values
            if abs(survival_adv) > 1000:
                survival_adv = 1000 * (1 if survival_adv > 0 else -1)
            
            normalized_survival_adv = survival_adv / 1000  # Scale to roughly -1 to 1
            composite_score += weights["survival"] * normalized_survival_adv
            advantage_components["survival"] = normalized_survival_adv
        
        # Population growth component
        if "population_growth" in results and key in results["population_growth"]:
            growth_adv = (
                results["population_growth"][key]["early_phase_advantage"] * 0.2 +
                results["population_growth"][key]["mid_phase_advantage"] * 0.3 +
                results["population_growth"][key]["late_phase_advantage"] * 0.3 +
                results["population_growth"][key]["growth_rate_advantage"] * 0.2
            )
            # Normalize by dividing by the maximum population to get a -1 to 1 scale
            max_pop = max(
                max([results["population_growth"][t]["late_phase_population"] for t in agent_types]),
                1
            )
            normalized_growth_adv = growth_adv / max_pop
            composite_score += weights["population_growth"] * normalized_growth_adv
            advantage_components["population_growth"] = normalized_growth_adv
        
        # Combat component
        if "combat" in results and key in results["combat"]:
            combat_adv = (
                results["combat"][key]["attack_success_advantage"] * 0.3 +
                results["combat"][key]["defense_success_advantage"] * 0.3 +
                (results["combat"][key]["direct_combat_success_rate"] - 0.5) * 2 * 0.4
            )
            composite_score += weights["combat"] * combat_adv
            advantage_components["combat"] = combat_adv
        
        # Initial positioning component
        if "initial_positioning" in results and key in results["initial_positioning"]:
            # This would be derived from the actual initial positioning metrics
            # Here we're just using a placeholder
            pos_adv = 0
            for advantage_key in results["initial_positioning"][key]:
                pos_adv += results["initial_positioning"][key][advantage_key] * 0.33
            
            composite_score += weights["initial_positioning"] * pos_adv
            advantage_components["initial_positioning"] = pos_adv
        
        results["composite_advantage"][key] = {
            "score": composite_score,
            "components": advantage_components
        }
    
    category_duration = time.time() - category_start_time
    logging.debug(f"Completed composite advantage calculation in {category_duration:.2f}s")
    
    total_duration = time.time() - start_time
    logging.debug(f"Completed compute_relative_advantages in {total_duration:.2f}s")
    
    return results


def compute_advantage_dominance_correlation(sim_session):
    """
    Compute correlation between relative advantages and ultimate dominance.
    
    This function analyzes how strongly different types of relative advantage
    correlate with which agent type achieves dominance by the end of the simulation.
    
    Parameters
    ----------
    sim_session : SQLAlchemy session
        Database session for the simulation
    
    Returns
    -------
    dict
        Dictionary containing correlation metrics between advantages and dominance
    """
    import time
    
    start_time = time.time()
    logging.debug("Starting compute_advantage_dominance_correlation")
    
    from farm.database.models import SimulationStepModel
    from farm.analysis.dominance.compute import compute_comprehensive_dominance
    
    # First, calculate comprehensive dominance
    logging.debug("Computing comprehensive dominance")
    dominance_result = compute_comprehensive_dominance(sim_session)
    if not dominance_result or "dominant_type" not in dominance_result:
        logging.warning("No dominant type found, skipping advantage-dominance correlation")
        return None
    
    dominant_type = dominance_result["dominant_type"]
    logging.debug(f"Dominant type: {dominant_type}")
    
    # Calculate relative advantages
    logging.debug("Computing relative advantages for correlation analysis")
    advantages = compute_relative_advantages(sim_session, focus_agent_type=dominant_type)
    
    # Initialize results
    results = {
        "dominant_type": dominant_type,
        "advantage_correlations": {}
    }
    
    # For each advantage category, determine if the dominant type had an advantage
    # and how strong that advantage was
    logging.debug("Analyzing advantage correlations with dominance")
    for category in advantages:
        if category == "composite_advantage":
            continue  # Skip the composite score
            
        category_results = {}
        
        # For each agent type pair involving the dominant type
        for key in advantages[category]:
            if key.startswith(f"{dominant_type}_vs_") or key.endswith(f"_vs_{dominant_type}"):
                # For each specific advantage metric in this category
                for metric, value in advantages[category][key].items():
                    # Skip non-advantage metrics (e.g., raw values)
                    if "advantage" not in metric and "trajectory" not in metric:
                        continue
                        
                    # Determine if advantage favors dominant type
                    favors_dominant = (
                        (key.startswith(f"{dominant_type}_vs_") and value > 0) or
                        (key.endswith(f"_vs_{dominant_type}") and value < 0)
                    )
                    
                    # Record both the raw advantage and whether it favors dominant type
                    metric_key = f"{key}_{metric}"
                    category_results[metric_key] = {
                        "value": value,
                        "favors_dominant": favors_dominant
                    }
        
        results["advantage_correlations"][category] = category_results
    
    # Calculate summary statistics
    logging.debug("Calculating advantage summary statistics")
    summary = {
        "total_advantages": 0,
        "advantages_favoring_dominant": 0,
        "advantage_ratio": 0
    }
    
    for category, metrics in results["advantage_correlations"].items():
        for metric, data in metrics.items():
            summary["total_advantages"] += 1
            if data["favors_dominant"]:
                summary["advantages_favoring_dominant"] += 1
    
    if summary["total_advantages"] > 0:
        summary["advantage_ratio"] = (
            summary["advantages_favoring_dominant"] / summary["total_advantages"]
        )
    
    results["summary"] = summary
    
    total_duration = time.time() - start_time
    logging.debug(f"Completed advantage-dominance correlation in {total_duration:.2f}s")
    
    return results 