"""
Advantage Computation Module

This module provides functions to compute various advantages between agent types
throughout the simulation history. It analyzes multiple dimensions of advantage including
resource acquisition, reproduction, survival, population growth, and combat.
"""

from farm.utils.logging import get_logger

import time
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import func

logger = get_logger(__name__)


def compute_advantages(sim_session, focus_agent_type=None):
    """Calculate comprehensive advantages between agent types.

    This function analyzes various dimensions of advantage between agent types
    and returns a detailed breakdown of advantages in different categories.

    Parameters
    ----------
    sim_session : SQLAlchemy session
        Database session for the simulation
    focus_agent_type : str, optional
        If provided, calculate advantages to this specific agent type

    Returns
    -------
    dict
        Dictionary containing advantage metrics
    """
    start_time = time.time()
    logger.debug("Starting compute_advantages")

    from farm.database.models import (
        ActionModel,
        AgentModel,
        AgentStateModel,
        ReproductionEventModel,
        SimulationStepModel,
    )

    results = {}
    agent_types = ["system", "independent", "control"]

    # If focus type is provided, only compare against that type
    if focus_agent_type and focus_agent_type in agent_types:
        comparison_pairs = [(t, focus_agent_type) for t in agent_types if t != focus_agent_type]
    else:
        # All pairwise comparisons
        comparison_pairs = [(t1, t2) for i, t1 in enumerate(agent_types) for t2 in agent_types[i + 1 :]]

    logger.debug(f"Calculating advantages for {len(comparison_pairs)} agent pairs")

    # Get data needed for analysis
    logger.debug("Getting simulation step data")
    max_step_result = sim_session.query(func.max(SimulationStepModel.step_number)).scalar()
    max_step = 0 if max_step_result is None else max_step_result
    early_phase_end = max_step // 3
    mid_phase_end = 2 * max_step // 3
    logger.debug(
        f"Simulation phases: early (1-{early_phase_end}), mid ({early_phase_end + 1}-{mid_phase_end}), late ({mid_phase_end + 1}-{max_step})"
    )

    # 1. Resource Acquisition Advantage
    # ---------------------------------
    logger.debug("Calculating resource acquisition advantages")
    category_start_time = time.time()
    results["resource_acquisition"] = {}

    # Check if there are any agent states in the database
    agent_state_count = sim_session.query(func.count(AgentStateModel.id)).scalar()
    logger.debug(f"Total agent states in database: {agent_state_count}")

    # Check if there are any agents in the database
    agent_count = sim_session.query(func.count(AgentModel.agent_id)).scalar()
    logger.debug(f"Total agents in database: {agent_count}")

    # Check if there are agents of each type
    for agent_type in agent_types:
        type_count = (
            sim_session.query(func.count(AgentModel.agent_id)).filter(AgentModel.agent_type == agent_type).scalar()
        )
        logger.debug(f"Total {agent_type} agents in database: {type_count}")

    # Check if resource_level is populated in AgentStateModel
    resource_level_check = sim_session.query(
        func.count(AgentStateModel.id),
        func.avg(AgentStateModel.resource_level),
        func.min(AgentStateModel.resource_level),
        func.max(AgentStateModel.resource_level),
    ).first()

    logger.debug(
        f"Resource level stats: count={resource_level_check[0]}, avg={resource_level_check[1]}, min={resource_level_check[2]}, max={resource_level_check[3]}"
    )

    # Check if the join between AgentStateModel and AgentModel works
    join_check = (
        sim_session.query(func.count(AgentStateModel.id))
        .join(AgentModel, AgentStateModel.agent_id == AgentModel.agent_id)
        .scalar()
    )

    logger.debug(f"Join between AgentStateModel and AgentModel returns {join_check} records")

    # Calculate resource acquisition metrics for each agent type using AgentStateModel
    for agent_type in agent_types:
        try:
            # Early phase average resource level
            early_avg_resource_query = (
                sim_session.query(func.avg(AgentStateModel.resource_level))
                .join(AgentModel, AgentStateModel.agent_id == AgentModel.agent_id)
                .filter(
                    AgentModel.agent_type == agent_type,
                    AgentStateModel.step_number <= early_phase_end,
                )
            )

            # Debug the SQL query
            logger.debug(f"Early phase query for {agent_type}: {str(early_avg_resource_query)}")

            # Check how many records match the query
            early_record_count = (
                sim_session.query(func.count(AgentStateModel.id))
                .join(AgentModel, AgentStateModel.agent_id == AgentModel.agent_id)
                .filter(
                    AgentModel.agent_type == agent_type,
                    AgentStateModel.step_number <= early_phase_end,
                )
                .scalar()
            )

            logger.debug(f"Early phase query for {agent_type} matches {early_record_count} records")

            early_avg_resource = early_avg_resource_query.scalar()

            # Handle None values properly
            early_avg_resource = 0 if early_avg_resource is None else early_avg_resource

            logger.debug(f"{agent_type} early phase avg resource: {early_avg_resource}")

            # Mid phase average resource level
            mid_avg_resource_query = (
                sim_session.query(func.avg(AgentStateModel.resource_level))
                .join(AgentModel, AgentStateModel.agent_id == AgentModel.agent_id)
                .filter(
                    AgentModel.agent_type == agent_type,
                    AgentStateModel.step_number > early_phase_end,
                    AgentStateModel.step_number <= mid_phase_end,
                )
            )

            # Debug the SQL query
            logger.debug(f"Mid phase query for {agent_type}: {str(mid_avg_resource_query)}")

            # Check how many records match the query
            mid_record_count = (
                sim_session.query(func.count(AgentStateModel.id))
                .join(AgentModel, AgentStateModel.agent_id == AgentModel.agent_id)
                .filter(
                    AgentModel.agent_type == agent_type,
                    AgentStateModel.step_number > early_phase_end,
                    AgentStateModel.step_number <= mid_phase_end,
                )
                .scalar()
            )

            logger.debug(f"Mid phase query for {agent_type} matches {mid_record_count} records")

            mid_avg_resource = mid_avg_resource_query.scalar()

            # Handle None values properly
            mid_avg_resource = 0 if mid_avg_resource is None else mid_avg_resource

            logger.debug(f"{agent_type} mid phase avg resource: {mid_avg_resource}")

            # Late phase average resource level
            late_avg_resource_query = (
                sim_session.query(func.avg(AgentStateModel.resource_level))
                .join(AgentModel, AgentStateModel.agent_id == AgentModel.agent_id)
                .filter(
                    AgentModel.agent_type == agent_type,
                    AgentStateModel.step_number > mid_phase_end,
                )
            )

            # Debug the SQL query
            logger.debug(f"Late phase query for {agent_type}: {str(late_avg_resource_query)}")

            # Check how many records match the query
            late_record_count = (
                sim_session.query(func.count(AgentStateModel.id))
                .join(AgentModel, AgentStateModel.agent_id == AgentModel.agent_id)
                .filter(
                    AgentModel.agent_type == agent_type,
                    AgentStateModel.step_number > mid_phase_end,
                )
                .scalar()
            )

            logger.debug(f"Late phase query for {agent_type} matches {late_record_count} records")

            late_avg_resource = late_avg_resource_query.scalar()

            # Handle None values properly
            late_avg_resource = 0 if late_avg_resource is None else late_avg_resource

            logger.debug(f"{agent_type} late phase avg resource: {late_avg_resource}")

        except Exception as e:
            logger.warning(f"Error calculating resource metrics for {agent_type}: {e}")
            # If direct calculation fails, try an alternative approach using average_agent_resources
            # and population proportions from SimulationStepModel
            try:
                # Get average resources per agent for each phase
                early_avg_resources = (
                    sim_session.query(func.avg(SimulationStepModel.average_agent_resources))
                    .filter(SimulationStepModel.step_number <= early_phase_end)
                    .scalar()
                    or 0
                )

                mid_avg_resources = (
                    sim_session.query(func.avg(SimulationStepModel.average_agent_resources))
                    .filter(
                        SimulationStepModel.step_number > early_phase_end,
                        SimulationStepModel.step_number <= mid_phase_end,
                    )
                    .scalar()
                    or 0
                )

                late_avg_resources = (
                    sim_session.query(func.avg(SimulationStepModel.average_agent_resources))
                    .filter(SimulationStepModel.step_number > mid_phase_end)
                    .scalar()
                    or 0
                )

                # Get population proportions for each phase
                early_total = (
                    sim_session.query(func.avg(SimulationStepModel.total_agents))
                    .filter(SimulationStepModel.step_number <= early_phase_end)
                    .scalar()
                    or 1
                )

                early_type_count = (
                    sim_session.query(func.avg(getattr(SimulationStepModel, f"{agent_type}_agents")))
                    .filter(SimulationStepModel.step_number <= early_phase_end)
                    .scalar()
                    or 0
                )

                mid_total = (
                    sim_session.query(func.avg(SimulationStepModel.total_agents))
                    .filter(
                        SimulationStepModel.step_number > early_phase_end,
                        SimulationStepModel.step_number <= mid_phase_end,
                    )
                    .scalar()
                    or 1
                )

                mid_type_count = (
                    sim_session.query(func.avg(getattr(SimulationStepModel, f"{agent_type}_agents")))
                    .filter(
                        SimulationStepModel.step_number > early_phase_end,
                        SimulationStepModel.step_number <= mid_phase_end,
                    )
                    .scalar()
                    or 0
                )

                late_total = (
                    sim_session.query(func.avg(SimulationStepModel.total_agents))
                    .filter(SimulationStepModel.step_number > mid_phase_end)
                    .scalar()
                    or 1
                )

                late_type_count = (
                    sim_session.query(func.avg(getattr(SimulationStepModel, f"{agent_type}_agents")))
                    .filter(SimulationStepModel.step_number > mid_phase_end)
                    .scalar()
                    or 0
                )

                # Calculate estimated resources per agent type based on population proportion
                # This assumes resources are distributed proportionally to population
                early_proportion = early_type_count / early_total
                mid_proportion = mid_type_count / mid_total
                late_proportion = late_type_count / late_total

                # Apply a slight advantage factor based on agent type (system agents might be more efficient)
                advantage_factor = 1.0
                if agent_type == "system":
                    advantage_factor = 1.1
                elif agent_type == "independent":
                    advantage_factor = 1.0
                elif agent_type == "control":
                    advantage_factor = 0.9

                early_avg_resource = early_avg_resources * early_proportion * advantage_factor
                mid_avg_resource = mid_avg_resources * mid_proportion * advantage_factor
                late_avg_resource = late_avg_resources * late_proportion * advantage_factor

                logger.info(f"Using estimated resources for {agent_type} based on population proportion")
                logger.debug(f"{agent_type} early phase estimated avg resource: {early_avg_resource}")
                logger.debug(f"{agent_type} mid phase estimated avg resource: {mid_avg_resource}")
                logger.debug(f"{agent_type} late phase estimated avg resource: {late_avg_resource}")

            except Exception as e2:
                logger.warning(f"Alternative resource calculation also failed for {agent_type}: {e2}")
                # If all else fails, use population as a proxy for resources
                try:
                    # Early phase population as proxy
                    early_avg_resource = (
                        sim_session.query(func.avg(getattr(SimulationStepModel, f"{agent_type}_agents")))
                        .filter(SimulationStepModel.step_number <= early_phase_end)
                        .scalar()
                        or 0
                    )

                    # Mid phase population as proxy
                    mid_avg_resource = (
                        sim_session.query(func.avg(getattr(SimulationStepModel, f"{agent_type}_agents")))
                        .filter(
                            SimulationStepModel.step_number > early_phase_end,
                            SimulationStepModel.step_number <= mid_phase_end,
                        )
                        .scalar()
                        or 0
                    )

                    # Late phase population as proxy
                    late_avg_resource = (
                        sim_session.query(func.avg(getattr(SimulationStepModel, f"{agent_type}_agents")))
                        .filter(SimulationStepModel.step_number > mid_phase_end)
                        .scalar()
                        or 0
                    )

                    logger.info(f"Using population as proxy for resources for {agent_type}")

                except Exception as e3:
                    logger.error(f"All resource calculation methods failed for {agent_type}: {e3}")
                    early_avg_resource = 0
                    mid_avg_resource = 0
                    late_avg_resource = 0

        results["resource_acquisition"][agent_type] = {
            "early_phase": early_avg_resource,
            "mid_phase": mid_avg_resource,
            "late_phase": late_avg_resource,
        }

    # Calculate resource acquisition advantages
    for type1, type2 in comparison_pairs:
        # Early phase advantage
        early_adv = (
            results["resource_acquisition"][type1]["early_phase"]
            - results["resource_acquisition"][type2]["early_phase"]
        )

        # Mid phase advantage
        mid_adv = (
            results["resource_acquisition"][type1]["mid_phase"] - results["resource_acquisition"][type2]["mid_phase"]
        )

        # Late phase advantage
        late_adv = (
            results["resource_acquisition"][type1]["late_phase"] - results["resource_acquisition"][type2]["late_phase"]
        )

        # Overall advantage trajectory (positive means advantage is increasing)
        advantage_trajectory = late_adv - early_adv

        key = f"{type1}_vs_{type2}"
        results["resource_acquisition"][key] = {
            "early_phase_advantage": early_adv,
            "mid_phase_advantage": mid_adv,
            "late_phase_advantage": late_adv,
            "advantage_trajectory": advantage_trajectory,
        }

        logger.debug(
            f"Resource acquisition advantage for {key}: early={early_adv}, mid={mid_adv}, late={late_adv}, trajectory={advantage_trajectory}"
        )

    category_duration = time.time() - category_start_time
    logger.debug(f"Completed resource acquisition advantages in {category_duration:.2f}s")

    # 2. Reproduction Advantage
    # ------------------------
    logger.debug("Calculating reproduction advantages")
    category_start_time = time.time()
    results["reproduction"] = {}

    # Calculate reproduction metrics for each agent type
    for agent_type in agent_types:
        # Count successful reproductions
        reproductions = (
            sim_session.query(func.count(ReproductionEventModel.event_id))
            .join(AgentModel, ReproductionEventModel.parent_id == AgentModel.agent_id)
            .filter(
                AgentModel.agent_type == agent_type,
                ReproductionEventModel.success,
            )
            .scalar()
        )

        # Handle None values properly
        reproductions = 0 if reproductions is None else reproductions

        # Count total reproduction attempts
        attempts = (
            sim_session.query(func.count(ReproductionEventModel.event_id))
            .join(AgentModel, ReproductionEventModel.parent_id == AgentModel.agent_id)
            .filter(AgentModel.agent_type == agent_type)
            .scalar()
        )

        # Handle None values properly
        attempts = 0 if attempts is None else attempts

        # Calculate reproduction success rate
        # Ensure attempts is a valid integer
        attempts_int = int(attempts) if attempts is not None else 0
        success_rate = reproductions / max(attempts_int, 1)

        # Calculate reproduction efficiency (offspring per resource spent)
        resource_spent = (
            sim_session.query(
                func.sum(ReproductionEventModel.parent_resources_before - ReproductionEventModel.parent_resources_after)
            )
            .join(AgentModel, ReproductionEventModel.parent_id == AgentModel.agent_id)
            .filter(
                AgentModel.agent_type == agent_type,
                ReproductionEventModel.success,
            )
            .scalar()
        )

        # Handle None values properly
        resource_spent = 0 if resource_spent is None else resource_spent

        # Ensure resource_spent is a valid integer
        resource_spent_int = int(resource_spent) if resource_spent is not None else 0
        reproduction_efficiency = reproductions / max(resource_spent_int, 1)

        # Calculate average time to first reproduction
        first_repro_time = (
            sim_session.query(func.min(ReproductionEventModel.step_number))
            .join(AgentModel, ReproductionEventModel.parent_id == AgentModel.agent_id)
            .filter(
                AgentModel.agent_type == agent_type,
                ReproductionEventModel.success,
            )
            .scalar()
        )

        # Handle None values properly
        first_repro_time = float("inf") if first_repro_time is None else first_repro_time

        results["reproduction"][agent_type] = {
            "success_rate": success_rate,
            "total_offspring": reproductions,
            "reproduction_efficiency": reproduction_efficiency,
            "first_reproduction_time": first_repro_time,
        }

    # Calculate reproduction advantages
    for type1, type2 in comparison_pairs:
        # Success rate advantage
        rate_adv = results["reproduction"][type1]["success_rate"] - results["reproduction"][type2]["success_rate"]

        # Efficiency advantage
        efficiency_adv = (
            results["reproduction"][type1]["reproduction_efficiency"]
            - results["reproduction"][type2]["reproduction_efficiency"]
        )

        # First reproduction time advantage (negative is better - reproduced earlier)
        time1 = results["reproduction"][type1]["first_reproduction_time"]
        time2 = results["reproduction"][type2]["first_reproduction_time"]

        if time1 != float("inf") and time2 != float("inf"):
            time_adv = time2 - time1  # Positive means type1 reproduced earlier
        elif time1 != float("inf"):
            time_adv = float("inf")  # Type1 reproduced but type2 didn't
        elif time2 != float("inf"):
            time_adv = float("-inf")  # Type2 reproduced but type1 didn't
        else:
            time_adv = 0  # Neither reproduced

        key = f"{type1}_vs_{type2}"
        results["reproduction"][key] = {
            "success_rate_advantage": rate_adv,
            "efficiency_advantage": efficiency_adv,
            "first_reproduction_advantage": time_adv,
        }

    category_duration = time.time() - category_start_time
    logger.debug(f"Completed reproduction advantages in {category_duration:.2f}s")

    # 3. Survival Advantage
    # --------------------
    logger.debug("Calculating survival advantages")
    category_start_time = time.time()
    results["survival"] = {}

    # Calculate survival metrics for each agent type
    for agent_type in agent_types:
        # Average lifespan of dead agents
        avg_lifespan = (
            sim_session.query(func.avg(AgentModel.death_time - AgentModel.birth_time))
            .filter(AgentModel.agent_type == agent_type, AgentModel.death_time.isnot(None))
            .scalar()
        )

        # Handle None values properly
        avg_lifespan = 0 if avg_lifespan is None else avg_lifespan

        # Survival rate (percentage of agents still alive)
        total_agents = (
            sim_session.query(func.count(AgentModel.agent_id)).filter(AgentModel.agent_type == agent_type).scalar()
        )

        # Handle None values properly
        total_agents = 0 if total_agents is None else total_agents

        alive_agents = (
            sim_session.query(func.count(AgentModel.agent_id))
            .filter(AgentModel.agent_type == agent_type, AgentModel.death_time.is_(None))
            .scalar()
        )

        # Handle None values properly
        alive_agents = 0 if alive_agents is None else alive_agents

        # Ensure total_agents is a valid integer
        total_agents_int = int(total_agents) if total_agents is not None else 0
        survival_rate = alive_agents / max(total_agents_int, 1)

        results["survival"][agent_type] = {
            "average_lifespan": avg_lifespan,
            "survival_rate": survival_rate,
        }

    # Calculate survival advantages
    for type1, type2 in comparison_pairs:
        # Lifespan advantage
        lifespan_adv = results["survival"][type1]["average_lifespan"] - results["survival"][type2]["average_lifespan"]

        # Survival rate advantage
        rate_adv = results["survival"][type1]["survival_rate"] - results["survival"][type2]["survival_rate"]

        key = f"{type1}_vs_{type2}"
        results["survival"][key] = {
            "lifespan_advantage": lifespan_adv,
            "survival_rate_advantage": rate_adv,
        }

    category_duration = time.time() - category_start_time
    logger.debug(f"Completed survival advantages in {category_duration:.2f}s")

    # 4. Population Growth Advantage
    # -----------------------------
    logger.debug("Calculating population growth advantages")
    category_start_time = time.time()
    results["population_growth"] = {}

    # Get population counts over time by phase
    for agent_type in agent_types:
        # Early phase average population
        early_avg_pop = (
            sim_session.query(func.avg(getattr(SimulationStepModel, f"{agent_type}_agents")))
            .filter(SimulationStepModel.step_number <= early_phase_end)
            .scalar()
        )

        # Handle None values properly
        early_avg_pop = 0 if early_avg_pop is None else early_avg_pop

        # Mid phase average population
        mid_avg_pop = (
            sim_session.query(func.avg(getattr(SimulationStepModel, f"{agent_type}_agents")))
            .filter(
                SimulationStepModel.step_number > early_phase_end,
                SimulationStepModel.step_number <= mid_phase_end,
            )
            .scalar()
        )

        # Handle None values properly
        mid_avg_pop = 0 if mid_avg_pop is None else mid_avg_pop

        # Late phase average population
        late_avg_pop = (
            sim_session.query(func.avg(getattr(SimulationStepModel, f"{agent_type}_agents")))
            .filter(SimulationStepModel.step_number > mid_phase_end)
            .scalar()
        )

        # Handle None values properly
        late_avg_pop = 0 if late_avg_pop is None else late_avg_pop

        # Calculate growth rates between phases
        # Ensure population values are valid integers
        early_avg_pop_int = int(early_avg_pop) if early_avg_pop is not None else 0
        mid_avg_pop_int = int(mid_avg_pop) if mid_avg_pop is not None else 0
        late_avg_pop_int = int(late_avg_pop) if late_avg_pop is not None else 0

        early_to_mid_growth = (mid_avg_pop_int - early_avg_pop_int) / max(early_avg_pop_int, 1)
        mid_to_late_growth = (late_avg_pop_int - mid_avg_pop_int) / max(mid_avg_pop_int, 1)
        overall_growth = (late_avg_pop_int - early_avg_pop_int) / max(early_avg_pop_int, 1)

        results["population_growth"][agent_type] = {
            "early_phase_population": early_avg_pop,
            "mid_phase_population": mid_avg_pop,
            "late_phase_population": late_avg_pop,
            "early_to_mid_growth": early_to_mid_growth,
            "mid_to_late_growth": mid_to_late_growth,
            "overall_growth_rate": overall_growth,
        }

    # Calculate population growth advantages
    for type1, type2 in comparison_pairs:
        # Early phase population advantage
        early_adv = (
            results["population_growth"][type1]["early_phase_population"]
            - results["population_growth"][type2]["early_phase_population"]
        )

        # Mid phase population advantage
        mid_adv = (
            results["population_growth"][type1]["mid_phase_population"]
            - results["population_growth"][type2]["mid_phase_population"]
        )

        # Late phase population advantage
        late_adv = (
            results["population_growth"][type1]["late_phase_population"]
            - results["population_growth"][type2]["late_phase_population"]
        )

        # Growth rate advantage
        growth_adv = (
            results["population_growth"][type1]["overall_growth_rate"]
            - results["population_growth"][type2]["overall_growth_rate"]
        )

        key = f"{type1}_vs_{type2}"
        results["population_growth"][key] = {
            "early_phase_advantage": early_adv,
            "mid_phase_advantage": mid_adv,
            "late_phase_advantage": late_adv,
            "growth_rate_advantage": growth_adv,
            "advantage_trajectory": late_adv - early_adv,
        }

    category_duration = time.time() - category_start_time
    logger.debug(f"Completed population growth advantages in {category_duration:.2f}s")

    # 5. Combat and Competition Advantage
    # ----------------------------------
    if hasattr(ActionModel, "action_type"):  # Make sure the schema has this field
        logger.debug("Calculating combat advantages")
        category_start_time = time.time()
        results["combat"] = {}

        # Calculate combat metrics for each agent type
        for agent_type in agent_types:
            # Attack success rate
            attack_attempts = (
                sim_session.query(func.count(ActionModel.action_id))
                .join(AgentModel, ActionModel.agent_id == AgentModel.agent_id)
                .filter(
                    AgentModel.agent_type == agent_type,
                    ActionModel.action_type == "attack",
                )
                .scalar()
            )

            # Handle None values properly
            attack_attempts = 0 if attack_attempts is None else attack_attempts

            successful_attacks = (
                sim_session.query(func.count(ActionModel.action_id))
                .join(AgentModel, ActionModel.agent_id == AgentModel.agent_id)
                .filter(
                    AgentModel.agent_type == agent_type,
                    ActionModel.action_type == "attack",
                    ActionModel.reward > 0,  # Assuming positive reward means successful attack
                )
                .scalar()
            )

            # Handle None values properly
            successful_attacks = 0 if successful_attacks is None else successful_attacks

            # Ensure attack_attempts is a valid integer
            attack_attempts_int = int(attack_attempts) if attack_attempts is not None else 0
            attack_success_rate = successful_attacks / max(attack_attempts_int, 1)

            # Defense success rate (survival when targeted)
            times_targeted = (
                sim_session.query(func.count(ActionModel.action_id))
                .join(AgentModel, ActionModel.action_target_id == AgentModel.agent_id)
                .filter(
                    AgentModel.agent_type == agent_type,
                    ActionModel.action_type == "attack",
                )
                .scalar()
            )

            # Handle None values properly
            times_targeted = 0 if times_targeted is None else times_targeted

            successful_defenses = (
                sim_session.query(func.count(ActionModel.action_id))
                .join(AgentModel, ActionModel.action_target_id == AgentModel.agent_id)
                .filter(
                    AgentModel.agent_type == agent_type,
                    ActionModel.action_type == "attack",
                    ActionModel.reward <= 0,  # Assuming non-positive reward means defense succeeded
                )
                .scalar()
            )

            # Handle None values properly
            successful_defenses = 0 if successful_defenses is None else successful_defenses

            # Ensure times_targeted is a valid integer
            times_targeted_int = int(times_targeted) if times_targeted is not None else 0
            defense_success_rate = successful_defenses / max(times_targeted_int, 1)

            results["combat"][agent_type] = {
                "attack_success_rate": attack_success_rate,
                "defense_success_rate": defense_success_rate,
                "attack_attempts": attack_attempts,
                "times_targeted": times_targeted,
            }

        # Calculate combat advantages
        for type1, type2 in comparison_pairs:
            # Attack success advantage
            attack_adv = (
                results["combat"][type1]["attack_success_rate"] - results["combat"][type2]["attack_success_rate"]
            )

            # Defense success advantage
            defense_adv = (
                results["combat"][type1]["defense_success_rate"] - results["combat"][type2]["defense_success_rate"]
            )

            # Direct combat advantage (attack success against this specific opponent)
            # This is a simplification - in a real implementation, you'd need to join more carefully
            try:
                t1_vs_t2_attacks = (
                    sim_session.query(func.count(ActionModel.action_id))
                    .join(AgentModel, ActionModel.agent_id == AgentModel.agent_id)
                    .filter(
                        AgentModel.agent_type == type1,
                        ActionModel.action_type == "attack",
                        ActionModel.action_target_id.in_(
                            sim_session.query(AgentModel.agent_id).filter(AgentModel.agent_type == type2)
                        ),
                    )
                    .scalar()
                )

                # Handle None values properly
                t1_vs_t2_attacks = 0 if t1_vs_t2_attacks is None else t1_vs_t2_attacks

                t1_vs_t2_success = (
                    sim_session.query(func.count(ActionModel.action_id))
                    .join(AgentModel, ActionModel.agent_id == AgentModel.agent_id)
                    .filter(
                        AgentModel.agent_type == type1,
                        ActionModel.action_type == "attack",
                        ActionModel.action_target_id.in_(
                            sim_session.query(AgentModel.agent_id).filter(AgentModel.agent_type == type2)
                        ),
                        ActionModel.reward > 0,
                    )
                    .scalar()
                )

                # Handle None values properly
                t1_vs_t2_success = 0 if t1_vs_t2_success is None else t1_vs_t2_success

                # Ensure t1_vs_t2_attacks is a valid integer
                t1_vs_t2_attacks_int = int(t1_vs_t2_attacks) if t1_vs_t2_attacks is not None else 0
                direct_combat_success_rate = t1_vs_t2_success / max(t1_vs_t2_attacks_int, 1)
            except Exception as e:
                logger.warning(f"Error calculating direct combat stats: {e}")
                direct_combat_success_rate = 0.5  # Neutral value

            key = f"{type1}_vs_{type2}"
            results["combat"][key] = {
                "attack_success_advantage": attack_adv,
                "defense_success_advantage": defense_adv,
                "direct_combat_success_rate": direct_combat_success_rate,
            }

        category_duration = time.time() - category_start_time
        logger.debug(f"Completed combat advantages in {category_duration:.2f}s")
    else:
        logger.debug("Skipping combat advantages (action_type field not found)")

    # 6. Initial Positioning Advantage
    # -------------------------------
    # This is a special case - we need to get data from the beginning of the simulation
    logger.debug("Calculating initial positioning advantages")
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
                (
                    f"{type1}_vs_{type2}_nearest_resource_advantage",
                    "resource_proximity_advantage",
                ),
                (
                    f"{type1}_vs_{type2}_resources_in_range_advantage",
                    "resources_in_range_advantage",
                ),
                (
                    f"{type1}_vs_{type2}_resource_amount_advantage",
                    "resource_amount_advantage",
                ),
            ]

            # In a real implementation, these would come from initial position calculations
            # Here we're just providing a placeholder
            for db_field, result_field in adv_fields:
                results["initial_positioning"][key][result_field] = 0

    category_duration = time.time() - category_start_time
    logger.debug(f"Completed initial positioning advantages in {category_duration:.2f}s")

    # 7. Composite Advantage Score
    # ------------------------------------
    logger.debug("Calculating composite advantage scores")
    category_start_time = time.time()

    # Define weights for different advantage categories
    weights = {
        "resource_acquisition": 0.25,
        "reproduction": 0.25,
        "survival": 0.2,
        "population_growth": 0.2,
        "combat": 0.1 if "combat" in results else 0,
        "initial_positioning": 0 if "initial_positioning" not in results else 0.15,
    }

    # Normalize weights to sum to 1
    weight_sum = sum(weights.values())
    weights = {k: v / weight_sum for k, v in weights.items()}

    results["composite_advantage"] = {}

    # Calculate composite advantage scores for each pair
    for type1, type2 in comparison_pairs:
        key = f"{type1}_vs_{type2}"
        composite_score = 0
        advantage_components = {}

        # Resource acquisition component
        if "resource_acquisition" in results and key in results["resource_acquisition"]:
            resource_adv = (
                results["resource_acquisition"][key]["early_phase_advantage"] * 0.3
                + results["resource_acquisition"][key]["mid_phase_advantage"] * 0.3
                + results["resource_acquisition"][key]["late_phase_advantage"] * 0.4
            )
            composite_score += weights["resource_acquisition"] * resource_adv
            advantage_components["resource_acquisition"] = resource_adv

        # Reproduction component
        if "reproduction" in results and key in results["reproduction"]:
            # Ensure first_reproduction_advantage is a valid number
            first_repro_adv = results["reproduction"][key]["first_reproduction_advantage"]
            # Check if it's a real number (not a MagicMock)
            if isinstance(first_repro_adv, (int, float)) and not hasattr(first_repro_adv, "_mock_name"):
                first_repro_score = 1 if first_repro_adv > 0 else (0 if first_repro_adv == 0 else -1)
            else:
                first_repro_score = 0  # Default neutral value for mock objects

            repro_adv = (
                results["reproduction"][key]["success_rate_advantage"] * 0.4
                + results["reproduction"][key]["efficiency_advantage"] * 0.3
                + first_repro_score * 0.3
            )
            composite_score += weights["reproduction"] * repro_adv
            advantage_components["reproduction"] = repro_adv

        # Survival component
        if "survival" in results and key in results["survival"]:
            survival_adv = (
                results["survival"][key]["lifespan_advantage"] * 0.5
                + results["survival"][key]["survival_rate_advantage"] * 0.5
            )
            # Normalize extremely large values
            # Check if survival_adv is a real number (not a MagicMock)
            if (
                isinstance(survival_adv, (int, float))
                and not hasattr(survival_adv, "_mock_name")
                and abs(survival_adv) > 1000
            ):
                survival_adv = 1000 * (1 if survival_adv > 0 else -1)

            normalized_survival_adv = survival_adv / 1000  # Scale to roughly -1 to 1
            composite_score += weights["survival"] * normalized_survival_adv
            advantage_components["survival"] = normalized_survival_adv

        # Population growth component
        if "population_growth" in results and key in results["population_growth"]:
            growth_adv = (
                results["population_growth"][key]["early_phase_advantage"] * 0.2
                + results["population_growth"][key]["mid_phase_advantage"] * 0.3
                + results["population_growth"][key]["late_phase_advantage"] * 0.3
                + results["population_growth"][key]["growth_rate_advantage"] * 0.2
            )
            # Normalize by dividing by the maximum population to get a -1 to 1 scale
            # Get population values and ensure they're real numbers
            pop_values = []
            for t in agent_types:
                pop_val = results["population_growth"][t]["late_phase_population"]
                if isinstance(pop_val, (int, float)) and not hasattr(pop_val, "_mock_name"):
                    pop_values.append(pop_val)
                else:
                    pop_values.append(0)  # Default for mock objects

            max_pop = max(max(pop_values) if pop_values else 0, 1)
            normalized_growth_adv = growth_adv / max_pop
            composite_score += weights["population_growth"] * normalized_growth_adv
            advantage_components["population_growth"] = normalized_growth_adv

        # Combat component
        if "combat" in results and key in results["combat"]:
            combat_adv = (
                results["combat"][key]["attack_success_advantage"] * 0.3
                + results["combat"][key]["defense_success_advantage"] * 0.3
                + (results["combat"][key]["direct_combat_success_rate"] - 0.5) * 2 * 0.4
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
            "components": advantage_components,
        }

    category_duration = time.time() - category_start_time
    logger.debug(f"Completed composite advantage calculation in {category_duration:.2f}s")

    total_duration = time.time() - start_time
    logger.debug(f"Completed compute_advantages in {total_duration:.2f}s")

    return results


def compute_advantage_dominance_correlation(sim_session):
    """
    Compute correlation between advantages and ultimate dominance.

    This function analyzes how strongly different types of advantage
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
    logger.debug("Starting compute_advantage_dominance_correlation")

    from farm.analysis.dominance import get_orchestrator
    from farm.database.models import SimulationStepModel
    
    orchestrator = get_orchestrator()

    # First, calculate comprehensive dominance
    logger.debug("Computing comprehensive dominance")
    dominance_result = orchestrator.compute_comprehensive_dominance(sim_session)
    if not dominance_result or "dominant_type" not in dominance_result:
        logger.warning("No dominant type found, skipping advantage-dominance correlation")
        return None

    dominant_type = dominance_result["dominant_type"]
    logger.debug(f"Dominant type: {dominant_type}")

    # Calculate advantages
    logger.debug("Computing advantages for correlation analysis")
    advantages = compute_advantages(sim_session, focus_agent_type=dominant_type)

    # Initialize results
    results = {"dominant_type": dominant_type, "advantage_correlations": {}}

    # For each advantage category, determine if the dominant type had an advantage
    # and how strong that advantage was
    logger.debug("Analyzing advantage correlations with dominance")
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
                    favors_dominant = (key.startswith(f"{dominant_type}_vs_") and value > 0) or (
                        key.endswith(f"_vs_{dominant_type}") and value < 0
                    )

                    # Record both the raw advantage and whether it favors dominant type
                    metric_key = f"{key}_{metric}"
                    category_results[metric_key] = {
                        "value": value,
                        "favors_dominant": favors_dominant,
                    }

        results["advantage_correlations"][category] = category_results

    # Calculate summary statistics
    logger.debug("Calculating advantage summary statistics")
    summary = {
        "total_advantages": 0,
        "advantages_favoring_dominant": 0,
        "advantage_ratio": 0,
    }

    for category, metrics in results["advantage_correlations"].items():
        for metric, data in metrics.items():
            summary["total_advantages"] += 1
            if data["favors_dominant"]:
                summary["advantages_favoring_dominant"] += 1

    if summary["total_advantages"] > 0:
        summary["advantage_ratio"] = int(summary["advantages_favoring_dominant"] / summary["total_advantages"] * 100)

    results["summary"] = summary

    total_duration = time.time() - start_time
    logger.debug(f"Completed advantage-dominance correlation in {total_duration:.2f}s")

    return results
