import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class MushroomAgent(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, state_size):
        super(MushroomAgent, self).__init__()
        self.fc1 = nn.Linear(input_size + state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.state_size = state_size
        self.state_decoder = nn.Linear(hidden_size, state_size)

    def forward(self, x, hidden_state=None):
        if hidden_state is None:
            x = torch.cat((x, torch.zeros(x.size(0), self.state_size)), dim=1)
        else:
            x = torch.cat((x, hidden_state), dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x), self.state_decoder(x)


class Agent:
    def __init__(
        self,
        state_size,
        action_size,
        hidden_state_size,
        num_agents,
        hidden_size=8,
        lr=1e-3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.hidden_size = hidden_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MushroomAgent(
            state_size, hidden_size, action_size, hidden_state_size
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.hidden_states = [None for _ in range(num_agents)]
        self.criterion = nn.MSELoss()

    def act(self, states):
        actions = []
        for i, state in enumerate(states):
            if np.random.rand() <= self.epsilon:
                actions.append(np.random.choice(self.action_size))
            else:
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action_values, new_hidden_state = self.model(
                        state, self.hidden_states[i]
                    )
                actions.append(torch.argmax(action_values).item())
                self.hidden_states[i] = new_hidden_state
        return actions

    def learn(self, states, actions, rewards, next_states, done):
        for i in range(self.num_agents):
            state = torch.FloatTensor(states[i]).unsqueeze(0).to(self.device)
            next_state = torch.FloatTensor(next_states[i]).unsqueeze(0).to(self.device)
            reward = torch.FloatTensor([rewards[i]]).to(self.device)

            q_values, _state = self.model(state, self.hidden_states[i])
            next_q_values = self.model(next_state, _state)[0]

            target = q_values.clone()
            target[0][actions[i]] = reward + self.gamma * torch.max(next_q_values) * (
                1 - done
            )

            loss = self.criterion(q_values, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filename):
        torch.save(self.model.state_dict(), filename)

    def load(self, filename):
        self.model.load_state_dict(torch.load(filename))
