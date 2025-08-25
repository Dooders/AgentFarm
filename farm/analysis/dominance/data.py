import logging

import sqlalchemy
from scipy.spatial.distance import euclidean

from farm.database.models import (
    AgentModel,
    ReproductionEventModel,
    ResourceModel,
    SimulationStepModel,
)


def get_final_population_counts(sim_session):
    """
    Get the final population counts for each agent type.
    """
    final_step = (
        sim_session.query(SimulationStepModel)
        .order_by(SimulationStepModel.step_number.desc())
        .first()
    )
    if final_step is None:
        return None

    return {
        "system_agents": final_step.system_agents,
        "independent_agents": final_step.independent_agents,
        "control_agents": final_step.control_agents,
        "total_agents": final_step.total_agents,
        "final_step": final_step.step_number,
    }


def get_agent_survival_stats(sim_session):
    """
    Get detailed survival statistics for each agent type.
    """
    try:
        logging.info("Calculating agent survival statistics...")
        agents = sim_session.query(AgentModel).all()
        logging.info(f"Found {len(agents)} agents")

        # Initialize counters and accumulators
        stats = {
            "system": {"count": 0, "alive": 0, "dead": 0, "total_survival": 0},
            "independent": {"count": 0, "alive": 0, "dead": 0, "total_survival": 0},
            "control": {"count": 0, "alive": 0, "dead": 0, "total_survival": 0},
        }

        # Map from database agent types to our internal types
        agent_type_map = {
            "SystemAgent": "system",
            "IndependentAgent": "independent",
            "ControlAgent": "control",
        }

        # Get the final step number for calculating survival of still-alive agents
        final_step = (
            sim_session.query(SimulationStepModel)
            .order_by(SimulationStepModel.step_number.desc())
            .first()
        )
        final_step_number = final_step.step_number if final_step else 0
        logging.info(f"Final step number: {final_step_number}")

        for agent in agents:
            # Map the agent type from the database to our internal type
            if agent.agent_type in agent_type_map:
                agent_type = agent_type_map[agent.agent_type]
            else:
                agent_type = agent.agent_type.lower()

            if agent_type not in stats:
                logging.warning(f"Unknown agent type: {agent.agent_type}")
                continue

            stats[agent_type]["count"] += 1

            if agent.death_time is not None:
                stats[agent_type]["dead"] += 1
                survival = agent.death_time - agent.birth_time
            else:
                stats[agent_type]["alive"] += 1
                # For alive agents, use the final step as the death time
                survival = final_step_number - agent.birth_time

            stats[agent_type]["total_survival"] += survival

        # Calculate averages
        result = {}
        for agent_type, data in stats.items():
            if data["count"] > 0:
                result[f"{agent_type}_count"] = data["count"]
                result[f"{agent_type}_alive"] = data["alive"]
                result[f"{agent_type}_dead"] = data["dead"]
                result[f"{agent_type}_avg_survival"] = (
                    data["total_survival"] / data["count"]
                )
                if data["dead"] > 0:
                    result[f"{agent_type}_dead_ratio"] = data["dead"] / data["count"]
                else:
                    result[f"{agent_type}_dead_ratio"] = 0

        logging.info(f"Agent survival statistics: {result}")
        return result
    except Exception as e:
        logging.error(f"Error calculating agent survival statistics: {e}")
        return {}


def get_initial_positions_and_resources(sim_session, config):
    """
    Get the initial positions of agents and resources.
    Calculate distances between agents and resources.
    """
    # Get the initial agents (birth_time = 0)
    initial_agents = (
        sim_session.query(AgentModel).filter(AgentModel.birth_time == 0).all()
    )

    # Get the initial resources (step_number = 0)
    initial_resources = (
        sim_session.query(ResourceModel).filter(ResourceModel.step_number == 0).all()
    )

    if not initial_agents or not initial_resources:
        return {}

    # Count initial agents by type
    system_count = sum(
        1 for agent in initial_agents if agent.agent_type == "SystemAgent"
    )
    independent_count = sum(
        1 for agent in initial_agents if agent.agent_type == "IndependentAgent"
    )
    control_count = sum(
        1 for agent in initial_agents if agent.agent_type == "ControlAgent"
    )

    # Count resources and total resource amount
    resource_count = len(initial_resources)
    resource_amount = sum(resource.amount for resource in initial_resources)

    # Extract positions
    agent_positions = {
        agent.agent_type.lower(): (agent.position_x, agent.position_y)
        for agent in initial_agents
    }

    resource_positions = [
        (resource.position_x, resource.position_y, resource.amount)
        for resource in initial_resources
    ]

    # Set initial count values
    result: dict[str, int | float] = {
        "initial_system_count": system_count,
        "initial_independent_count": independent_count,
        "initial_control_count": control_count,
        "initial_resource_count": resource_count,
        "initial_resource_amount": resource_amount,
    }

    # Calculate distances to resources
    for agent_type, pos in agent_positions.items():
        # Calculate distances to all resources
        distances = [
            euclidean(pos, (r_pos[0], r_pos[1])) for r_pos in resource_positions
        ]

        # Calculate distance to nearest resource
        if distances:
            result[f"{agent_type}_nearest_resource_dist"] = float(min(distances))

            # Calculate average distance to all resources
            result[f"{agent_type}_avg_resource_dist"] = float(
                sum(distances) / len(distances)
            )

            # Calculate weighted distance (by resource amount)
            weighted_distances = [
                distances[i]
                * (
                    1 / (resource_positions[i][2] + 1)
                )  # Add 1 to avoid division by zero
                for i in range(len(distances))
            ]
            result[f"{agent_type}_weighted_resource_dist"] = float(
                sum(weighted_distances) / len(weighted_distances)
            )

            # Count resources within gathering range
            gathering_range = config.get("gathering_range", 30)
            resources_in_range = sum(1 for d in distances if d <= gathering_range)
            result[f"{agent_type}_resources_in_range"] = resources_in_range

            # Calculate total resource amount within gathering range
            resource_amount_in_range = sum(
                resource_positions[i][2]
                for i in range(len(distances))
                if distances[i] <= gathering_range
            )
            result[f"{agent_type}_resource_amount_in_range"] = resource_amount_in_range

    # Calculate relative advantages (differences between agent types)
    agent_types = ["system", "independent", "control"]
    for i, type1 in enumerate(agent_types):
        for type2 in agent_types[i + 1 :]:
            # Difference in nearest resource distance
            key1 = f"{type1}_nearest_resource_dist"
            key2 = f"{type2}_nearest_resource_dist"
            if key1 in result and key2 in result:
                result[f"{type1}_vs_{type2}_nearest_resource_advantage"] = (
                    result[key2] - result[key1]
                )

            # Difference in resources in range
            key1 = f"{type1}_resources_in_range"
            key2 = f"{type2}_resources_in_range"
            if key1 in result and key2 in result:
                result[f"{type1}_vs_{type2}_resources_in_range_advantage"] = (
                    result[key1] - result[key2]
                )

            # Difference in resource amount in range
            key1 = f"{type1}_resource_amount_in_range"
            key2 = f"{type2}_resource_amount_in_range"
            if key1 in result and key2 in result:
                result[f"{type1}_vs_{type2}_resource_amount_advantage"] = (
                    result[key1] - result[key2]
                )

    return result


def get_reproduction_stats(sim_session):
    """
    Analyze reproduction patterns for each agent type.
    """
    try:
        # Check if ReproductionEventModel table exists
        inspector = sqlalchemy.inspect(sim_session.bind)
        if "reproduction_events" not in inspector.get_table_names():
            logging.warning("No reproduction_events table found in database")
            return {}

        # Query reproduction events
        try:
            reproduction_events = sim_session.query(ReproductionEventModel).all()
            logging.info(
                f"Found {len(reproduction_events)} reproduction events in database"
            )
        except Exception as e:
            logging.error(f"Error querying reproduction events: {e}")
            return {}

        if not reproduction_events:
            logging.warning("No reproduction events found in the database")
            return {}

        # Get all agents to determine their types
        try:
            # Create a mapping of agent IDs to normalized agent types
            agents_raw = {
                agent.agent_id: agent.agent_type
                for agent in sim_session.query(AgentModel).all()
            }

            # Normalize agent types (handle different case formats)
            agents = {}
            for agent_id, agent_type in agents_raw.items():
                # Convert to lowercase for comparison
                agent_type_lower = agent_type.lower()

                # Map to standard types based on substring matching
                if "system" in agent_type_lower:
                    normalized_type = "system"
                elif "independent" in agent_type_lower:
                    normalized_type = "independent"
                elif "control" in agent_type_lower:
                    normalized_type = "control"
                else:
                    normalized_type = "unknown"

                agents[agent_id] = normalized_type

            logging.info(f"Found {len(agents)} agents in database")

            # Log the agent type mapping for debugging
            agent_types_found = set(agents_raw.values())
            normalized_types = set(agents.values())
            logging.info(
                f"Original agent types in database: {', '.join(agent_types_found)}"
            )
            logging.info(f"Normalized to: {', '.join(normalized_types)}")

        except Exception as e:
            logging.error(f"Error querying agents: {e}")
            return {}

        # Initialize counters
        stats = {
            "system": {
                "attempts": 0,
                "successes": 0,
                "failures": 0,
                "first_reproduction_time": float("inf"),
                "resources_spent": 0,
                "offspring_resources": 0,
            },
            "independent": {
                "attempts": 0,
                "successes": 0,
                "failures": 0,
                "first_reproduction_time": float("inf"),
                "resources_spent": 0,
                "offspring_resources": 0,
            },
            "control": {
                "attempts": 0,
                "successes": 0,
                "failures": 0,
                "first_reproduction_time": float("inf"),
                "resources_spent": 0,
                "offspring_resources": 0,
            },
        }

        # Process reproduction events
        unknown_agent_types = set()
        missing_resource_data = 0

        for event in reproduction_events:
            try:
                parent_id = event.parent_id
                parent_type = agents.get(parent_id, "unknown")

                if parent_type not in stats:
                    if parent_type not in unknown_agent_types:
                        unknown_agent_types.add(parent_type)
                        logging.warning(
                            f"Unknown agent type: {parent_type} for agent {parent_id}"
                        )
                    continue

                stats[parent_type]["attempts"] += 1

                # Calculate resources spent on reproduction
                try:
                    resources_spent = (
                        event.parent_resources_before - event.parent_resources_after
                    )
                    stats[parent_type]["resources_spent"] += resources_spent
                except (TypeError, AttributeError):
                    missing_resource_data += 1
                    # Skip resource calculation if data is missing

                if event.success:
                    stats[parent_type]["successes"] += 1
                    # Track first successful reproduction time
                    stats[parent_type]["first_reproduction_time"] = min(
                        stats[parent_type]["first_reproduction_time"], event.step_number
                    )
                    # Track resources given to offspring
                    if (
                        hasattr(event, "offspring_initial_resources")
                        and event.offspring_initial_resources is not None
                    ):
                        stats[parent_type][
                            "offspring_resources"
                        ] += event.offspring_initial_resources
                else:
                    stats[parent_type]["failures"] += 1
            except Exception as e:
                logging.error(f"Error processing reproduction event: {e}")
                continue

        if missing_resource_data > 0:
            logging.warning(
                f"Missing resource data for {missing_resource_data} reproduction events"
            )

        if unknown_agent_types:
            logging.warning(
                f"Found unknown agent types: {', '.join(unknown_agent_types)}"
            )

        # Calculate derived metrics
        result = {}
        for agent_type, data in stats.items():
            if data["attempts"] > 0:
                result[f"{agent_type}_reproduction_attempts"] = data["attempts"]
                result[f"{agent_type}_reproduction_successes"] = data["successes"]
                result[f"{agent_type}_reproduction_failures"] = data["failures"]

                # Calculate success rate
                result[f"{agent_type}_reproduction_success_rate"] = (
                    data["successes"] / data["attempts"]
                )

                # Calculate resource metrics if we have resource data
                if data["resources_spent"] > 0:
                    result[f"{agent_type}_avg_resources_per_reproduction"] = (
                        data["resources_spent"] / data["attempts"]
                    )

                    if data["successes"] > 0 and data["offspring_resources"] > 0:
                        result[f"{agent_type}_avg_offspring_resources"] = (
                            data["offspring_resources"] / data["successes"]
                        )
                        result[f"{agent_type}_reproduction_efficiency"] = (
                            data["offspring_resources"] / data["resources_spent"]
                        )

                # First reproduction time
                if data["first_reproduction_time"] != float("inf"):
                    result[f"{agent_type}_first_reproduction_time"] = data[
                        "first_reproduction_time"
                    ]
                else:
                    result[f"{agent_type}_first_reproduction_time"] = (
                        -1
                    )  # No successful reproduction

        # Calculate relative advantages (differences between agent types)
        agent_types = ["system", "independent", "control"]
        for i, type1 in enumerate(agent_types):
            for type2 in agent_types[i + 1 :]:
                # Difference in reproduction success rate
                key1 = f"{type1}_reproduction_success_rate"
                key2 = f"{type2}_reproduction_success_rate"
                if key1 in result and key2 in result:
                    result[f"{type1}_vs_{type2}_reproduction_rate_advantage"] = (
                        result[key1] - result[key2]
                    )

                # Difference in first reproduction time (negative is better - earlier reproduction)
                key1 = f"{type1}_first_reproduction_time"
                key2 = f"{type2}_first_reproduction_time"
                if (
                    key1 in result
                    and key2 in result
                    and result[key1] > 0
                    and result[key2] > 0
                ):
                    result[f"{type1}_vs_{type2}_first_reproduction_advantage"] = (
                        result[key2] - result[key1]
                    )

                # Difference in reproduction efficiency
                key1 = f"{type1}_reproduction_efficiency"
                key2 = f"{type2}_reproduction_efficiency"
                if key1 in result and key2 in result:
                    result[f"{type1}_vs_{type2}_reproduction_efficiency_advantage"] = (
                        result[key1] - result[key2]
                    )

        # Log summary of reproduction stats
        if result:
            logging.info("\nReproduction statistics summary:")
            for agent_type in agent_types:
                attempts_key = f"{agent_type}_reproduction_attempts"
                success_rate_key = f"{agent_type}_reproduction_success_rate"
                if attempts_key in result:
                    attempts = result[attempts_key]
                    success_rate = result.get(success_rate_key, 0) * 100
                    logging.info(
                        f"  {agent_type}: {attempts} attempts, {success_rate:.1f}% success rate"
                    )

            # Log all calculated metrics for debugging
            logging.info(f"Calculated {len(result)} reproduction metrics:")
            for key in sorted(result.keys())[:10]:  # Show first 10
                logging.info(f"  {key}: {result[key]}")
        else:
            logging.warning("No reproduction statistics could be calculated")

        return result

    except Exception as e:
        logging.error(f"Error in get_reproduction_stats: {e}")
        import traceback

        logging.error(traceback.format_exc())
        return {}
