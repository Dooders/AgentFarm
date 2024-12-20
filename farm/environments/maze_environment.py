import numpy as np
from .base_environment import BaseEnvironment
from gymnasium import spaces


class MazeEnv(BaseEnvironment):
    def __init__(self, maze, start, end, **kwargs):
        super().__init__(width=maze.shape[1], height=maze.shape[0], **kwargs)
        self.maze = maze
        self.start = start
        self.end = end
        self.state = start
        self.action_space = spaces.Discrete(4)  # Up, Down, Left, Right
        self.observation_space = spaces.Box(
            low=0, high=max(self.width, self.height) - 1, shape=(2,), dtype=np.int32
        )

    def reset(self):
        """Reset the environment to its initial state."""
        self.state = self.start
        return np.array(self.state, dtype=np.int32)

    def step(self, action):
        """Apply an action and return the environment's new state."""
        old_state = self.state
        x, y = old_state
        old_distance = np.linalg.norm(np.array(self.end) - np.array(self.state))

        # Update position based on action
        new_x, new_y = x, y
        if action == 0 and y > 0 and self.maze[y - 1, x] == 0:  # Up
            new_y -= 1
        elif action == 1 and y < self.height - 1 and self.maze[y + 1, x] == 0:  # Down
            new_y += 1
        elif action == 2 and x > 0 and self.maze[y, x - 1] == 0:  # Left
            new_x -= 1
        elif action == 3 and x < self.width - 1 and self.maze[y, x + 1] == 0:  # Right
            new_x += 1

        self.state = (new_x, new_y)
        new_distance = np.linalg.norm(np.array(self.end) - np.array(self.state))

        # Reward structure
        if self.state == self.end:
            reward = 100
            done = True
        else:
            reward = 10 * (old_distance - new_distance)
            reward -= 0.01  # Time penalty
            if self.state == old_state:
                reward -= 0.1  # Penalty for hitting a wall
            done = False

        return np.array(self.state, dtype=np.int32), reward, done, {}

    def render(self, mode="human"):
        """Render the current state of the maze."""
        maze_copy = self.maze.copy()
        x, y = self.state
        maze_copy[y, x] = 2  # Mark the agent's current position
        print(maze_copy)
