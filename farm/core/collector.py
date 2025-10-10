import numpy as np
import pandas as pd


class DataCollector:
    def __init__(self, config):
        self.config = config
        self.data = []
        self.births_this_cycle = 0
        self.deaths_this_cycle = 0
        self.competitive_interactions = 0
        self.agent_resource_history = {}

    def collect(self, environment, step):
        alive_agents = [agent for agent in environment.agent_objects if agent.alive]

        for agent in alive_agents:
            if agent.agent_id not in self.agent_resource_history:
                self.agent_resource_history[agent.agent_id] = []
            self.agent_resource_history[agent.agent_id].append(agent.get_component("resource").level)

        data_point = {
            "step": step,
            "total_agent_count": len(alive_agents),
            "total_resources": sum(
                resource.amount for resource in environment.resources
            ),
            "total_consumption": sum(agent.get_component("resource").level for agent in alive_agents),
            "average_resource_per_agent": (
                sum(agent.get_component("resource").level for agent in alive_agents) / len(alive_agents)
                if alive_agents
                else 0
            ),
            "births": self.births_this_cycle,
            "deaths": self.deaths_this_cycle,
            "average_lifespan": self._calculate_average_lifespan(environment),
            "resource_efficiency": self._calculate_resource_efficiency(alive_agents),
            "agent_territory": self._calculate_territory_control(
                alive_agents, environment
            ),
            "resource_density": self._calculate_resource_density(environment),
            "population_stability": self._calculate_population_stability(),
            "survival_rate": len(alive_agents)
            / max(environment.get_initial_agent_count(), 1),
            "competitive_interactions": self.competitive_interactions,
            "avg_resource_accumulation": self._calculate_avg_resource_accumulation(),
            "resource_inequality": self._calculate_resource_inequality(alive_agents),
        }
        self.data.append(data_point)

        self.births_this_cycle = 0
        self.deaths_this_cycle = 0
        self.competitive_interactions = 0

    def _calculate_average_lifespan(self, environment):
        alive_agents = [agent for agent in environment.agent_objects if agent.alive]
        return (
            sum(
                environment.time - getattr(agent, "birth_time", 0)
                for agent in alive_agents
            )
            / len(alive_agents)
            if alive_agents
            else 0
        )

    def _calculate_resource_efficiency(self, agents):
        if not agents:
            return 0
        return sum(
            0.0 / max(agent.get_component("resource").level, 1) for agent in agents  # total_reward not tracked
        ) / len(agents)

    def _calculate_territory_control(self, agents, environment):
        if not agents:
            return 0
        total_area = environment.width * environment.height
        territory_size = sum(
            self._estimate_agent_territory(agent, environment) for agent in agents
        )
        return territory_size / total_area

    def _calculate_resource_density(self, environment):
        total_area = environment.width * environment.height
        total_resources = sum(resource.amount for resource in environment.resources)
        return total_resources / total_area

    def calculate_average_resources(self, environment):
        alive_agents = [agent for agent in environment.agent_objects if agent.alive]
        if not alive_agents:
            return 0
        return sum(agent.get_component("resource").level for agent in alive_agents) / len(alive_agents)

    def _estimate_agent_territory(self, agent, environment):
        nearest_distance = float("inf")
        for other in environment.agents:
            if other != agent and other.alive:
                dist = np.sqrt(
                    (agent.position[0] - other.position[0]) ** 2
                    + (agent.position[1] - other.position[1]) ** 2
                )
                nearest_distance = min(nearest_distance, dist)
        return min(
            np.pi * (nearest_distance / 2) ** 2, environment.width * environment.height
        )

    def _calculate_population_stability(self):
        if len(self.data) < 10:
            return 1.0

        recent_population = [d["total_agent_count"] for d in self.data[-10:]]
        return 1.0 - np.std(recent_population) / max(
            float(np.mean(recent_population)), 1.0
        )

    def _calculate_avg_resource_accumulation(self):
        if not self.agent_resource_history:
            return 0

        accumulation_rates = []
        for agent_resources in self.agent_resource_history.values():
            if len(agent_resources) >= 2:
                window_size = min(10, len(agent_resources))
                recent_resources = agent_resources[-window_size:]
                rate = (recent_resources[-1] - recent_resources[0]) / window_size
                accumulation_rates.append(rate)

        return np.mean(accumulation_rates) if accumulation_rates else 0

    def _calculate_resource_inequality(self, agents):
        if not agents:
            return 0

        resources = np.array([agent.get_component("resource").level for agent in agents])
        if np.all(resources == 0):
            return 0

        resources = np.sort(resources)
        n = len(resources)
        index = np.arange(1, n + 1)
        return ((2 * index - n - 1) * resources).sum() / (n * resources.sum())

    def to_dataframe(self):
        return pd.DataFrame(self.data)
