import gym
from gym import spaces
import numpy as np


class MushroomEnvironment(gym.Env):
    def __init__(
        self,
        grid_size=300,
        mushroom_ratio=0.01,
        max_steps=10000,
        num_patches=10,
        patch_size_range=(5, 20),
        num_agents=4,
    ):
        super(MushroomEnvironment, self).__init__()

        self.grid_size = grid_size
        self.mushroom_ratio = mushroom_ratio
        self.max_steps = max_steps
        self.num_patches = num_patches
        self.patch_size_range = patch_size_range
        self.num_agents = num_agents

        # Action space: continuous angle in radians
        self.action_space = spaces.Box(
            low=0, high=2 * np.pi, shape=(1,), dtype=np.float32
        )

        # Observation space: [x, y, steps_since_last_pickup] for each agent
        self.observation_space = spaces.Box(
            low=0,
            high=max(grid_size, max_steps),
            shape=(3 * num_agents,),
            dtype=np.float32,
        )

        self.reset()

    def reset(self):
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self._generate_mushroom_patches()
        self.agent_positions = np.random.rand(self.num_agents, 2) * self.grid_size
        self.steps_since_pickup = np.zeros(self.num_agents, dtype=np.int32)
        self.total_steps = 0
        self.mushrooms_collected = np.zeros(self.num_agents, dtype=np.int32)
        return self._get_obs()

    def _generate_mushroom_patches(self):
        total_mushrooms = int(self.grid_size * self.grid_size * self.mushroom_ratio)
        mushrooms_placed = 0

        for _ in range(self.num_patches):
            if mushrooms_placed >= total_mushrooms:
                break

            patch_size = np.random.randint(
                self.patch_size_range[0], self.patch_size_range[1] + 1
            )
            patch_center = np.random.randint(0, self.grid_size, size=2)

            for i in range(
                patch_center[0] - patch_size // 2, patch_center[0] + patch_size // 2
            ):
                for j in range(
                    patch_center[1] - patch_size // 2, patch_center[1] + patch_size // 2
                ):
                    if 0 <= i < self.grid_size and 0 <= j < self.grid_size:
                        if (
                            np.random.random() < 0.7
                        ):  # 70% chance of mushroom within patch
                            if (
                                self.grid[i, j] == 0
                                and mushrooms_placed < total_mushrooms
                            ):
                                self.grid[i, j] = 1
                                mushrooms_placed += 1

        # If we haven't placed enough mushrooms, fill in randomly until we reach the desired ratio
        while mushrooms_placed < total_mushrooms:
            i, j = np.random.randint(0, self.grid_size, size=2)
            if self.grid[i, j] == 0:
                self.grid[i, j] = 1
                mushrooms_placed += 1

    def step(self, actions):
        self.total_steps += 1
        self.steps_since_pickup += 1

        rewards = np.zeros(self.num_agents)

        for i, action in enumerate(actions):
            # Move agent
            dx = np.cos(action)
            dy = np.sin(action)
            self.agent_positions[i] += np.array([dx, dy])

            # Apply toroidal geometry
            self.agent_positions[i] = self.agent_positions[i] % self.grid_size

            # Check for mushroom
            grid_x, grid_y = np.floor(self.agent_positions[i]).astype(int)
            if self.grid[grid_x, grid_y] == 1:
                rewards[i] = 1
                self.mushrooms_collected[i] += 1
                self.grid[grid_x, grid_y] = 0
                self.steps_since_pickup[i] = 0

        done = self.total_steps >= self.max_steps
        return self._get_obs(), rewards, done, {}

    def _get_obs(self):
        return np.concatenate(
            [
                np.array(
                    [*self.agent_positions[i], self.steps_since_pickup[i]],
                    dtype=np.float32,
                )
                for i in range(self.num_agents)
            ]
        )
