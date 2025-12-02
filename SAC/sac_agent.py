import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from collections import deque
import random
import matplotlib.pyplot as plt
from datetime import datetime

from gym_pybullet_drones.envs.FlyThruGateAvitary import FlyThruGateAvitary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================================
# Replay Buffer
# ============================================================================
class ReplayBuffer:

    def __init__(self, capacity=1000000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # Nếu state / next_state là tuple (obs, info, ...) thì chỉ lấy obs
        if isinstance(state, (tuple, list)):
            state = state[0]
        if isinstance(next_state, (tuple, list)):
            next_state = next_state[0]

        # Ép về vector 1D float32 (cùng shape cho mọi mẫu)
        state = np.array(state, dtype=np.float32).reshape(-1)
        next_state = np.array(next_state, dtype=np.float32).reshape(-1)
        action = np.array(action, dtype=np.float32).reshape(-1)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (np.array(state), np.array(action),
                np.array(reward, dtype=np.float32), np.array(next_state),
                np.array(done, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)

# ============================================================================
# Neural Network Architectures
# ============================================================================
class QNetwork(nn.Module):
    """Critic Network (Q-function)"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(QNetwork, self).__init__()

        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value  = self.fc3(x)
        return q_value


class GaussianPolicy(nn.Module):
    """Actor Network (Policy)"""
    def __init__(self, state_dim, action_dim, hidden_dim=256,
                 action_scale=1.0, action_bias=0.0, log_std_min=-20, log_std_max=2):
        super(GaussianPolicy, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

        self.action_scale = action_scale
        self.action_bias = action_bias
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)

        # Reparameterization trick
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        # Compute log probability
        log_prob = normal.log_prob(x_t)
        # Enforcing action bounds
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)

        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob, mean

# ============================================================================
# SAC Agent
# ============================================================================
class SACAgent:

    def __init__(self,
                 state_dim,
                 action_dim,
                 hidden_dim=256,
                 lr=3e-4,
                 gamma=0.99,
                 tau=0.005,
                 alpha=0.2,
                 automatic_entropy_tuning=True):

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.automatic_entropy_tuning = automatic_entropy_tuning

        # Actor
        self.policy = GaussianPolicy(state_dim, action_dim,
                                     hidden_dim).to(device)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(),
                                                 lr=lr)

        # Critics
        self.q1 = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=lr)

        # Target Critics
        self.q1_target = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.q2_target = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Automatic entropy tuning
        if self.automatic_entropy_tuning:
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

    def select_action(self, state, evaluate=False):

        # nếu state là tuple/list (vd: (obs, info) hoặc (obs, r, term, trunc, info))
        if isinstance(state, (tuple, list)):
            state = state[0]  # chỉ lấy quan sát obs

        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        if evaluate:
            _, _, action = self.policy.sample(state)
        else:
            action, _, _ = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update(self, replay_buffer, batch_size=256):
        # Sample batch
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = \
            replay_buffer.sample(batch_size)

        state_batch = torch.FloatTensor(state_batch).to(device)
        action_batch = torch.FloatTensor(action_batch).to(device)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1).to(device).view(-1, 1)
        next_state_batch = torch.FloatTensor(next_state_batch).to(device)
        done_batch = torch.FloatTensor(done_batch).unsqueeze(1).to(device).view(-1, 1)

        with torch.no_grad():
            next_action, next_log_prob, _ = self.policy.sample(
                next_state_batch)
            q1_next_target = self.q1_target(next_state_batch, next_action)
            q2_next_target = self.q2_target(next_state_batch, next_action)
            min_q_next_target = torch.min(
                q1_next_target, q2_next_target) - self.alpha * next_log_prob
            next_q_value = reward_batch + (
                1 - done_batch) * self.gamma * min_q_next_target

        # Update Q-functions
        q1_value = self.q1(state_batch, action_batch)
        q2_value = self.q2(state_batch, action_batch)
        q1_loss = F.mse_loss(q1_value, next_q_value)
        q2_loss = F.mse_loss(q2_value, next_q_value)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # Update Policy
        new_action, log_prob, _ = self.policy.sample(state_batch)
        q1_new = self.q1(state_batch, new_action)
        q2_new = self.q2(state_batch, new_action)
        min_q_new = torch.min(q1_new, q2_new)

        policy_loss = ((self.alpha * log_prob) - min_q_new).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Update alpha (temperature parameter)
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha *
                           (log_prob + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp()

        # Soft update target networks
        self.soft_update(self.q1, self.q1_target)
        self.soft_update(self.q2, self.q2_target)

        return q1_loss.item(), q2_loss.item(), policy_loss.item()

    # Soft update target network parameters
    def soft_update(self, source, target):
        for target_param, param in zip(target.parameters(),
                                       source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) +
                                    param.data * self.tau)

    def save(self, filepath):
        torch.save(
            {
                'policy_state_dict': self.policy.state_dict(),
                'q1_state_dict': self.q1.state_dict(),
                'q2_state_dict': self.q2.state_dict(),
                'q1_target_state_dict': self.q1_target.state_dict(),
                'q2_target_state_dict': self.q2_target.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'q1_optimizer': self.q1_optimizer.state_dict(),
                'q2_optimizer': self.q2_optimizer.state_dict(),
            }, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.q1.load_state_dict(checkpoint['q1_state_dict'])
        self.q2.load_state_dict(checkpoint['q2_state_dict'])
        self.q1_target.load_state_dict(checkpoint['q1_target_state_dict'])
        self.q2_target.load_state_dict(checkpoint['q2_target_state_dict'])
        print(f"Model loaded from {filepath}")