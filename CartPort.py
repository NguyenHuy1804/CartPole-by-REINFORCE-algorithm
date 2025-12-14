# CartPort by REINFORCE algorithm

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Policy Network

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)



# REINFORCE Agent

class REINFORCEAgent:
    def __init__(self, obs_dim, action_dim, lr=1e-3, gamma=0.99):
        self.policy = PolicyNetwork(obs_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma

        self.log_probs = []
        self.rewards = []

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        probs = self.policy(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        self.log_probs.append(dist.log_prob(action))
        return action.item()

    def update(self):
        # ----- Compute Returns -----
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # ----- Compute Loss -----
        loss = 0
        for log_prob, G in zip(self.log_probs, returns):
            loss += -log_prob * G

        # ----- Gradient Update -----
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Clear memory
        self.log_probs = []
        self.rewards = []


# TRAINING

env = gym.make("CartPole-v1")
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = REINFORCEAgent(obs_dim, action_dim)

num_episodes = 200
episode_rewards = []

for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.rewards.append(reward)
        state = next_state
        total_reward += reward

    agent.update()
    episode_rewards.append(total_reward)

    if episode % 50 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}")

env.close()

# PLOT LEARNING CURVE

def moving_average(x, window=20):
    return np.convolve(x, np.ones(window)/window, mode='valid')

plt.figure()
plt.plot(episode_rewards, label="Raw Reward")
plt.plot(moving_average(episode_rewards), label="Smoothed Reward")
plt.xlabel("Episodes")
plt.ylabel("Total Reward per Episode")
plt.title("REINFORCE on CartPole-v1")
plt.legend()
plt.show()

# RENDER

env = gym.make("CartPole-v1", render_mode="human")

for _ in range(3):  # demo 3 episodes
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32)
        probs = agent.policy(state_tensor)
        action = torch.argmax(probs).item()  # greedy action

        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

    print("Demo episode reward:", total_reward)

env.close()
