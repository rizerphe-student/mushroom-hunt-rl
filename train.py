import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from agent import Agent
from environment import MushroomEnvironment
from visualize import MushroomVisualizer


def train(
    episodes,
    max_steps,
    num_agents,
    agent,
    visualize=False,
    visualization_interval=1,
    visualization_steps=1,
):
    env = MushroomEnvironment(num_patches=70, num_agents=num_agents)

    if visualize:
        vis = MushroomVisualizer(env.grid_size)

    scores: list[float] = []

    for episode in range(episodes):
        state = env.reset()
        episode_scores = np.zeros(num_agents)

        for step in tqdm(range(max_steps)):
            agent_states = [state[i * 3 : (i + 1) * 3] for i in range(num_agents)]
            actions = agent.act(agent_states)
            next_state, rewards, done, _ = env.step(actions)
            agent.learn(
                agent_states,
                actions,
                rewards,
                [next_state[i * 3 : (i + 1) * 3] for i in range(num_agents)],
                done,
            )
            state = next_state
            episode_scores += rewards

            if (
                visualize
                and episode % visualization_interval == 0
                and step % visualization_steps == 0
            ):
                vis.draw(env.grid, env.agent_positions)
                if vis.check_quit():
                    return scores

            if done:
                break

        mean_score = sum(episode_scores) / len(episode_scores)

        scores.append(mean_score)

        print(
            f"Episode: {episode}, Mean score: {mean_score} Epsilon: {agent.epsilon:.2f}, Scores: {episode_scores}"
        )

        agent.save(f"agent_ep{episode}.pth")

    if visualize:
        vis.close()

    return scores


# The rest of the file remains unchanged


def plot_results(scores):
    scores = np.array(scores)
    plt.figure(figsize=(10, 5))
    plt.plot(scores)
    plt.title("Training Progress")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    episodes = 1000
    max_steps = 5000
    num_agents = 40
    visualize = True
    visualization_interval = 1
    visualization_steps = 100
    agent = Agent(
        state_size=3, action_size=1, hidden_state_size=8, num_agents=num_agents
    )

    scores = train(
        episodes,
        max_steps,
        num_agents,
        agent,
        visualize,
        visualization_interval,
        visualization_steps,
    )
    plot_results(scores)
    print("Saving...", end=" ")
    agent.save("agent.pth")
    print("done")
