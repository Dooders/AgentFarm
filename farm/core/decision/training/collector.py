from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

import numpy as np

from farm.core.action import action_name_to_index
from farm.core.decision.feature_engineering import FeatureEngineer

if TYPE_CHECKING:
    from farm.core.agent import AgentCore
    from farm.core.environment import Environment


class ExperienceCollector:
    """Collect training data for ML algorithms.

    Produces a list of (state_features, action, reward) tuples from an episode.
    """

    def __init__(self) -> None:
        self.feature_engineer = FeatureEngineer()

    def collect_episode(
        self, agent: "AgentCore", environment: "Environment", max_steps: int = 200
    ) -> List[Tuple[np.ndarray, int, float]]:
        data: List[Tuple[np.ndarray, int, float]] = []

        # Reset environment to start a new episode
        environment.reset()
        done = False
        steps = 0

        while not done and steps < max_steps:
            # Extract features from current state
            state_features = self.feature_engineer.extract_features(agent, environment)

            # Agent selects an action based on the current state
            selected_action = agent.decide_action()

            # Map action name to environment action index using centralized mapping
            action_index = action_name_to_index(selected_action.name)

            # Apply the action to the environment and get reward
            next_state, reward, terminated, truncated, info = environment.step(
                action_index
            )

            # Check if episode is done (terminated or truncated)
            done = terminated or truncated

            # Record the experience tuple
            data.append((state_features, action_index, reward))

            steps += 1

        return data
