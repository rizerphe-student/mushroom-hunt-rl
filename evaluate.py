import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from copy import deepcopy

from agent import Agent
from environment import MushroomEnvironment
from visualize import MushroomVisualizer


def evaluate(
    max_steps: int,
    num_agents: int,
    agent: Agent,
    visualize: bool = True,
    visualization_steps: int = 1,
):
    env = MushroomEnvironment(num_patches=70, num_agents=num_agents)

    if visualize:
        vis = MushroomVisualizer(env.grid_size)

    state = env.reset()

    prev_grid = deepcopy(env.grid)

    mushrooms_eaten = 0
    total_mushrooms = np.sum(prev_grid)

    for step in tqdm(range(max_steps)):
        agent_states = [state[i * 3 : (i + 1) * 3] for i in range(num_agents)]
        actions = agent.act_eval(agent_states)
        next_state, rewards, done, _ = env.step(actions)
        state = next_state

        mushrooms_eaten += np.sum(prev_grid - env.grid)
        prev_grid = deepcopy(env.grid)

        if visualize and step % visualization_steps == 0:
            vis.draw(env.grid, env.agent_positions)
            if vis.check_quit():
                return mushrooms_eaten, total_mushrooms

        if done:
            break

    if visualize:
        vis.close()

    return mushrooms_eaten, total_mushrooms


if __name__ == "__main__":
    max_steps = 1000
    num_agents = 40
    visualize = True
    visualization_steps = 10
    agent = Agent(
        state_size=3, action_size=1, hidden_state_size=8, num_agents=num_agents
    )
    agent.load("agent_ep0.pth")

    mushrooms_eaten, total_mushrooms = evaluate(
        max_steps,
        num_agents,
        agent,
        visualize,
        visualization_steps,
    )
    print(f"Mushrooms eaten: {mushrooms_eaten} - {mushrooms_eaten/total_mushrooms:.2f}")
