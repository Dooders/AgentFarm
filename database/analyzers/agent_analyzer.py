from database.data_types import BasicAgentInfo
from database.repositories.agent_repository import AgentRepository


class AgentAnalyzer:
    #! maybe add this to repository as a method

    def __init__(self, repository: AgentRepository):
        """Initialize the AgentStatsAnalyzer.

        Args:
            repository (AgentRepository): Repository interface for accessing agent data.
        """
        self.repository = repository

    def analyze(self, agent_id: str) -> BasicAgentInfo:
        """Analyze comprehensive statistics for a specific agent.

        Compiles and returns various statistical metrics about an agent including:
        1. Basic agent stats (ID, type, timestamps)
        2. Position data (last known coordinates)
        3. Genealogy info (parent, generation, genome)
        4. Performance metrics (actions, health, learning, targeting)

        Args:
            agent_id (int): The unique identifier of the agent to analyze.

        Returns:
            BasicAgentStats: A data object containing all analyzed statistics.
                Includes fields for basic info, position, genealogy, and performance metrics.

        Note:
            Performance metrics are calculated based on relationship counts:
            - total_actions: Number of actions taken by the agent
            - total_health_incidents: Number of health-related events
            - learning_experiences_count: Number of learning events
            - times_targeted: Number of times this agent was targeted by others
        """
        agent = self.repository.get_agent_by_id(agent_id)
        return BasicAgentInfo(
            agent_id=agent.agent_id,
            agent_type=agent.agent_type,
            birth_time=agent.birth_time,
            death_time=agent.death_time,
            lifespan=(
                (agent.death_time - agent.birth_time) if agent.death_time else None
            ),
            initial_resources=agent.initial_resources,
            starting_health=agent.starting_health,
            starvation_threshold=agent.starvation_threshold,
            # Genealogy
            generation=agent.generation,
            genome_id=agent.genome_id,
        )
