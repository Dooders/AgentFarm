from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

import numpy as np

from farm.actions.feature_engineering import FeatureEngineer

if TYPE_CHECKING:
    from farm.agents.base_agent import BaseAgent
    from farm.core.environment import Environment


class ExperienceCollector:
    """Collect training data for ML algorithms.

    Produces a list of (state_features, action, reward) tuples from an episode.
    """

    def __init__(self) -> None:
        self.feature_engineer = FeatureEngineer()

    def collect_episode(
        self, agent: "BaseAgent", environment: "Environment", max_steps: int = 200
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

            # Map action name to environment action index
            action_name_to_index = {
                "defend": 0,
                "attack": 1,
                "gather": 2,
                "share": 3,
                "move": 4,
                "reproduce": 5,
            }
            action_index = action_name_to_index.get(selected_action.name.lower(), 0)

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
