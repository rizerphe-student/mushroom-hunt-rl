import gym
from gym import spaces
import numpy as np


class MushroomEnvironment(gym.Env):
    def __init__(
        self,
        grid_size=300,
        mushroom_ratio=0.01,
        max_steps=10000,
        num_patches=50,
        mushrooms_in_patch=50,
        patch_radius=10,
        num_agents=4,
        pickup_chance=0.7,
    ):
        super(MushroomEnvironment, self).__init__()

        self.grid_size = grid_size
        self.mushroom_ratio = mushroom_ratio
        self.max_steps = max_steps
        self.num_patches = num_patches
        self.patch_radius = patch_radius
        self.num_mushrooms_per_center = mushrooms_in_patch
        self.num_agents = num_agents
        self.pickup_chance = pickup_chance

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
        centers = np.random.randint(0, self.grid_size, size=(self.num_patches, 2))
        for center in centers:
            mushrooms_placed = 0
            while mushrooms_placed < self.num_mushrooms_per_center:
                x = np.random.randint(
                    max(0, center[0] - self.patch_radius),
                    min(self.grid_size, center[0] + self.patch_radius + 1),
                )
                y = np.random.randint(
                    max(0, center[1] - self.patch_radius),
                    min(self.grid_size, center[1] + self.patch_radius + 1),
                )
                if (
                    self.grid[x, y] == 0
                    and np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
                    <= self.patch_radius
                ):
                    self.grid[x, y] = 1
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
            if self.agent_positions[i][0] < 0:
                self.agent_positions[i][0] += self.grid_size
            if self.agent_positions[i][1] < 0:
                self.agent_positions[i][1] += self.grid_size

            # Check for mushroom
            grid_x, grid_y = np.floor(self.agent_positions[i]).astype(int)

            try:
                value = self.grid[grid_x, grid_y]
            except IndexError:
                print("Error getting grid value at:")
                print(f"{grid_x, grid_y = }")
                print(f"Agent positions at {i} are:")
                print(self.agent_positions[i])
                print("All agent positions:")
                print(self.agent_positions)
                grid_x, grid_y = self.grid_size//2, self.grid_size//2
                value = self.grid[grid_x, grid_y]

            if (
                np.random.random() < self.pickup_chance
                and value == 1
            ):
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
