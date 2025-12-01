import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import random
from torch.distributions import Normal

class MLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, ff_dim):
        super(MLPEncoder, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.zeros_(self.embedding.bias)
        self.block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, hidden_dim)
        )
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.block(x)
        return x #  [batch_size, hidden_dim] 

# Implmentation: If racing state_dim = 36, thrugate state_dim = 12, action_dim = 4 for both
class Critic(nn.Module):
    def __init__(self, state_dim=12, action_dim=4):
        """
        :param state_dim: Dimension of input state (int) - 12 - thrugate, 36 - racing
        :param action_dim: Dimension of input action (int) - 4
        :return:
        """
        super(Critic, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.state_encoder = MLPEncoder(state_dim, 128, 256) # state encoder output 128-dim

        self.fc2 = nn.Linear(128 + action_dim, 256)
        nn.init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain('tanh'))

        self.fc_out = nn.Linear(256, 1)
        nn.init.uniform_(self.fc_out.weight, -0.003, +0.003)

        self.activation = nn.Tanh()

    def forward(self, state, action):
        """
        Returns Q-value for given state and action
        :param state: input state (tensor) - [batch_size, state_dim]
        :param action: input action (tensor) - [batch_size, action_dim]
        :return: Q-value (tensor) - [batch_size, 1]
        """
        s = self.state_encoder(state) # [batch_size, 128]
        x = torch.cat([s, action], dim=1) # [batch_size, 128 + action_dim]
        x = self.activation(self.fc2(x)) # [batch_size, 256]
        q_value = self.fc_out(x) # [batch_size, 1]
        return q_value
    

class Actor(nn.Module):
    def __init__(self, state_dim=12, action_dim=4, stochastic=False):
        """
        :param state_dim: Dimension of input state (int) - 12 - thrugate, 36 - racing
        :param action_dim: Dimension of output action (int) - 4
        :param max_action: Maximum action value (float)
        :param stochastic: Whether to use stochastic policy (bool)
        :return:
        """
        super(Actor, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.stochastic = stochastic

        self.state_encoder = MLPEncoder(self.state_dim, 128, 256)

        self.fc = nn.Linear(128, action_dim, bias=False)
        nn.init.uniform_(self.fc.weight, -0.003, +0.003)

        if self.stochastic:
            self.log_std = nn.Linear(128, action_dim, bias=False)
            nn.init.uniform_(self.log_std.weight, -0.003, +0.003)
        
        self.tanh = nn.Tanh()

    def forward(self, state, explore=False):
        """
        Returns action for given state
        :param state: input state (tensor) - [batch_size, state_dim]
        :return: action (tensor) - [batch_size, action_dim]
        """
        s = self.state_encoder(state) # [batch_size, 128]
        if self.stochastic:
            means = self.fc(s) # [batch_size, action_dim]
            log_stds = self.log_std(s) # [batch_size, action_dim]
            log_stds = torch.clamp(log_stds, min=-10, max=2) # to avoid numerical issues
            stds = log_stds.exp() # torch.exp(log_stds) # [batch_size, action_dim]
            dists = Normal(means, stds) # distribution Gaussian

            if explore:
                x = dists.rsample()  # sampling with reparameterization trick
            else:
                x = means  # for evaluation, use mean
            actions = self.tanh(x)  # squash to [-1, 1]
            log_probs = dists.log_prob(x) - torch.log(1-actions.pow(2) + 1e-6)
            entropies = -log_probs.sum(dim=1, keepdim=True)
            return actions, entropies
        else:
            actions = self.tanh(self.fc(s)) # [batch_size, action_dim]
            return actions