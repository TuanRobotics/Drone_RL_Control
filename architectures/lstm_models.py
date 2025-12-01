import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import random
from torch.distributions import Normal

# https://github.com/vy007vikas/PyTorch-ActorCriticRL

EPS = 0.003

class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, batch_first=True):
        super(MyLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size=hidden_size, batch_first=batch_first, bidirectional=False, num_layers=1, dropout=0)
        self.lstm.bias_hh_l0.data.fill_(-0.2) # force lstm to output to depend more on last state at the initialization.

    def forward(self, x):
        x, (_, _) = self.lstm(x)
        x = x[:, -1]
        return x # dim: [batch_size, hidden_size] - 128

class Critic(nn.Module):

    def __init__(self, state_dim=12, action_dim=4): # 12 for thrugate, 36 for racing
        """
        :param state_dim: Dimension of input state (int)
        :param action_dim: Dimension of input action (int)
        :return:
        """
        super(Critic, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.state_encoder = MyLSTM(input_size=self.state_dim, hidden_size=128, batch_first=True)

        self.fc2 = nn.Linear(128 + self.action_dim, 256)
        nn.init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain('tanh'))
        
        self.fc_out = nn.Linear(256, 1, bias=False)
        nn.init.uniform_(self.fc_out.weight, -0.003,+0.003)

        self.act = nn.Tanh()

    def forward(self, state, action):
        """
        returns Value function Q(s,a) obtained from critic network
        :param state: Input state (Torch Variable : [n,state_dim] )
        :param action: Input Action (Torch Variable : [n,action_dim] )
        :return: Value function : Q(S,a) (Torch Variable : [n,1] )
        """
        s = self.state_encoder(state) # [n,128]
        x = torch.cat((s,action),dim=1) # [n,128+action_dim]
        x = self.act(self.fc2(x)) # [n,256]
        x = self.fc_out(x)*10 # [n,1]
        return x


class Actor(nn.Module):

    def __init__(self, state_dim=12, action_dim=4, stochastic=False): # 12 for thrugate, 36 for racing
        """
        :param state_dim: Dimension of input state (int)
        :param action_dim: Dimension of output action (int)
        :param action_lim: Used to limit action in [-action_lim,action_lim]
        :return:
        """
        super(Actor, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.stochastic = stochastic

        self.state_encoder = MyLSTM(input_size=self.state_dim, hidden_size=128, batch_first=True)
        self.fc = nn.Linear(128, action_dim, bias=False)
        nn.init.uniform_(self.fc.weight, -0.003,+0.003)

        if self.stochastic:
            self.log_std = nn.Linear(128, action_dim, bias=False)
            nn.init.uniform_(self.log_std.weight, -0.003,+0.003)

        self.tanh = nn.Tanh()

    def forward(self, state, explore=True):
        """
        returns either:
        - deterministic policy function mu(s) as policy action.
        - stochastic action sampled from tanh-gaussian policy, with its entropy value.
        this function returns actions lying in (-1,1) 
        :param state: Input state (Torch Variable : [n,state_dim] )
        :return: Output action (Torch Variable: [n,action_dim] )
        """
        s = self.state_encoder(state) # [n,128]
        if self.stochastic:
            means = self.fc(s) # [n,action_dim]
            log_stds = self.log_std(s) # [n,action_dim]
            log_stds = torch.clamp(log_stds, min=-10.0, max=2.0) # to avoid numerical issues
            stds = log_stds.exp() # [n,action_dim]
            #print(stds)
            dists = Normal(means, stds) # distribution Gaussian
            if explore:
                x = dists.rsample() # for reparameterization trick (mean + std * N(0,1))
            else:
                x = means            
            actions = self.tanh(x) # squash to [-1,1]
            log_probs = dists.log_prob(x) - torch.log(1-actions.pow(2) + 1e-6) # change of variables formula
            entropies = -log_probs.sum(dim=1, keepdim=True) # [n,1]
            return actions, entropies # return both action and entropy

        else:
            actions = self.tanh(self.fc(s)) # [n,action_dim]
            return actions

