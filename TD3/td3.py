import numpy as np
import random
import torch
from collections import namedtuple, deque
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from TD3.noise import DecayingOrnsteinUhlenbeckNoise, GaussianNoise
import os 
from itertools import chain


class ReplayBuffer:
    """Simle experience replay buffer for deep reinforcement algorithms."""
    def __init__(self, action_size, buffer_size, batch_size, device):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.device = device
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None], axis=0)).float().to(self.device)
        actions = torch.from_numpy(np.stack([e.action for e in experiences if e is not None], axis=0)).float().to(self.device)
        rewards = torch.from_numpy(np.stack([e.reward for e in experiences if e is not None], axis=0)).float().unsqueeze(-1).to(self.device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None], axis=0)).float().to(self.device)
        dones = torch.from_numpy(np.stack([e.done for e in experiences if e is not None], axis=0).astype(np.uint8)).float().unsqueeze(-1).to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)
    

class MLPEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, ff_size):
        super(MLPEncoder, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.zeros_(self.embedding.bias)
        self.block = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, ff_size), nn.GELU(), nn.Linear(ff_size, hidden_size))

    def forward(self, x):
        x = self.embedding(x)
        x = self.block(x)
        return x


class Critic(nn.Module):

    def __init__(self, state_dim=12, action_dim=4):
        """
        :param state_dim: Dimension of input state (int)
        :param action_dim: Dimension of input action (int)
        :return:
        """
        super(Critic, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.state_encoder = MLPEncoder(self.state_dim, 128, 128)

        self.fc2 = nn.Linear(128 + self.action_dim, 128)
        nn.init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain('tanh'))
        
        self.fc_out = nn.Linear(128, 1, bias=False)
        nn.init.uniform_(self.fc_out.weight, -0.003,+0.003)

        self.act = nn.Tanh()

    def forward(self, state, action):
        """
        returns Value function Q(s,a) obtained from critic network
        :param state: Input state (Torch Variable : [n,state_dim] )
        :param action: Input Action (Torch Variable : [n,action_dim] )
        :return: Value function : Q(S,a) (Torch Variable : [n,1] )
        """
        # Debug states shape
        # print(f"State shape in Critic forward: {state.shape}")
        s = self.state_encoder(state)
        x = torch.cat((s,action),dim=1) # concatenate along the second dimension
        x = self.act(self.fc2(x))
        x = self.fc_out(x)*10
        return x


class Actor(nn.Module):

    def __init__(self, state_dim=12, action_dim=4, stochastic=False):
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

        self.state_encoder = MLPEncoder(self.state_dim, 128, 128)

        self.fc = nn.Linear(128, action_dim, bias=False)
        nn.init.uniform_(self.fc.weight, -0.003, +0.003)
        #nn.init.zeros_(self.fc.bias)

        if self.stochastic:
            self.log_std = nn.Linear(128, action_dim, bias=False)
            nn.init.uniform_(self.log_std.weight, -0.003, +0.003)
            #nn.init.zeros_(self.log_std.bias)  

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
        s = self.state_encoder(state)
        if self.stochastic:
            means = self.fc(s)
            log_stds = self.log_std(s)
            log_stds = torch.clamp(log_stds, min=-10.0, max=2.0)
            stds = log_stds.exp()
            dists = Normal(means, stds)
            if explore:
                x = dists.rsample()
            else:
                x = means            
            actions = self.tanh(x)
            log_probs = dists.log_prob(x) - torch.log(1-actions.pow(2) + 1e-6)
            entropies = -log_probs.sum(dim=1, keepdim=True)
            return actions, entropies

        else:
            actions = self.tanh(self.fc(s))
            return actions
        
class TD3Agent:
    def __init__(self, Actor, Critic, clip_low,
                 clip_high, state_size=12, action_size=4, 
                 update_freq=int(2),
                 lr=4e-4, weight_decay=0, 
                 gamma=0.98, tau=0.01, batch_size=64, 
                 buffer_size=int(500000), device=None):
        self.state_size = state_size
        self.action_size = action_size
        self.update_freq = update_freq
        
        self.learn_call = int(0)

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        

        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        else:
            self.device = torch.device(device)

        self.clip_low = torch.tensor(clip_low).to(self.device)
        self.clip_high = torch.tensor(clip_high).to(self.device)

        self.train_actor = Actor().to(self.device)
        self.target_actor= Actor().to(self.device).eval()
        self.hard_update(self.train_actor, self.target_actor) # hard update at the beginning
        self.actor_optim = torch.optim.AdamW(self.train_actor.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)
        print(f'Number of paramters of Actor Net: {sum(p.numel() for p in self.train_actor.parameters())}')
        
        self.train_critic_1 = Critic().to(self.device)
        self.target_critic_1 = Critic().to(self.device).eval()
        self.hard_update(self.train_critic_1, self.target_critic_1) # hard update at the beginning
        self.critic_1_optim = torch.optim.AdamW(self.train_critic_1.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)

        self.train_critic_2 = Critic().to(self.device)
        self.target_critic_2 = Critic().to(self.device).eval()
        self.hard_update(self.train_critic_2, self.target_critic_2) # hard update at the beginning
        self.critic_2_optim = torch.optim.AdamW(self.train_critic_2.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)
        print(f'Number of paramters of Single Critic Net: {sum(p.numel() for p in self.train_critic_2.parameters())}')

        self.noise_generator = DecayingOrnsteinUhlenbeckNoise(mu=np.zeros(action_size), theta=4.0, sigma=1.2, dt=0.04, sigma_decay=0.9995)
        self.target_noise = GaussianNoise(mu=np.zeros(action_size), sigma=0.2, clip=0.4)
        
        self.memory= ReplayBuffer(action_size= action_size, buffer_size= buffer_size, \
            batch_size= self.batch_size, device=self.device)
        
        self.mse_loss = torch.nn.MSELoss()

    def learn_with_batches(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.learn_one_step()

    def learn_one_step(self):
        if(len(self.memory)>self.batch_size):
            exp=self.memory.sample()
            self.learn(exp) 
    def learn(self, exp):
        self.learn_call+=1
        states, actions, rewards, next_states, done = exp
        # Debug shapes
        # print(f"States shape: {states.shape}, Actions shape: {actions.shape}")
        # torch.Size([64, 1, 12]), Actions shape: torch.Size([64, 1, 4])
        # convert to 2D tensors if necessary
        if len(states.shape) > 2:
            states = states.view(states.size(0), -1)
        if len(actions.shape) > 2:
            actions = actions.view(actions.size(0), -1)
        if len(next_states.shape) > 2:
            next_states = next_states.view(next_states.size(0), -1)
        # Debug again
        # print(f"After reshape - States shape: {states.shape}, Actions shape: {actions.shape}, Next states shape: {next_states.shape}")

        #update critic
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            noise = torch.from_numpy(self.target_noise()).float().to(self.device)
            next_actions = next_actions + noise
            next_actions = torch.clamp(next_actions, self.clip_low, self.clip_high)
            # Error here
            Q_targets_next_1 = self.target_critic_1(next_states, next_actions)
            Q_targets_next_2 = self.target_critic_2(next_states, next_actions)
            Q_targets_next = torch.min(Q_targets_next_1, Q_targets_next_2).detach()
            Q_targets = rewards + (self.gamma * Q_targets_next * (1-done))
            #Q_targets = rewards + (self.gamma * Q_targets_next) 

        Q_expected_1 = self.train_critic_1(states, actions)
        critic_1_loss = self.mse_loss(Q_expected_1, Q_targets)
        #critic_1_loss = torch.nn.SmoothL1Loss()(Q_expected_1, Q_targets)

        self.critic_1_optim.zero_grad(set_to_none=True)
        critic_1_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.train_critic_1.parameters(), 1)
        self.critic_1_optim.step()

        Q_expected_2 = self.train_critic_2(states, actions)   
        critic_2_loss = self.mse_loss(Q_expected_2, Q_targets)
        #critic_2_loss = torch.nn.SmoothL1Loss()(Q_expected_2, Q_targets)
        
        self.critic_2_optim.zero_grad(set_to_none=True)
        critic_2_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.train_critic_2.parameters(), 1)
        self.critic_2_optim.step()

        if self.learn_call % self.update_freq == 0: # update_freq = 2
            self.learn_call = 0
            #update actor
            actions_pred = self.train_actor(states)
            actor_loss = -self.train_critic_1(states, actions_pred).mean()
            
            self.actor_optim.zero_grad(set_to_none=True)
            actor_loss.backward()
            #torch.nn.utils.clip_grad_norm_(self.train_actor.parameters(), 1)
            self.actor_optim.step()
        
            #using soft upates
            self.soft_update(self.train_actor, self.target_actor)
            self.soft_update(self.train_critic_1, self.target_critic_1)
            self.soft_update(self.train_critic_2, self.target_critic_2)
        
    @torch.no_grad()        
    def get_action(self, state, explore=False):
        #self.train_actor.eval()
        # print(f"State shape in get_action: {state.shape}")
        state = torch.from_numpy(state).unsqueeze(0).float().to(self.device)
        #with torch.no_grad():
        action = self.train_actor(state).cpu().data.numpy()[0]
        #self.train_actor.train()

        if explore:
            noise = self.noise_generator()
            #print(noise)
            action += noise
        return action
    
    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
    
    def hard_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)
    
    def save_ckpt(self, episode, prefix='last'):
        # Create folder 
        td3_dir = os.path.join("log_dir", "td3_thrugate")
        # Create log directory if it doesn't exist
        if not os.path.exists(td3_dir):
            os.makedirs(td3_dir)
        actor_dir = os.path.join(td3_dir, "actor")
        critics_dir = os.path.join(td3_dir, "critics")
        if not os.path.exists(actor_dir):
            os.makedirs(actor_dir)
        if not os.path.exists(critics_dir):
            os.makedirs(critics_dir)
        
        actor_file = os.path.join(actor_dir, "_".join([prefix, episode, "actor.pth"]))
        critic_1_file = os.path.join(critics_dir, "_".join([prefix, episode, "critic_1.pth"]))
        critic_2_file = os.path.join(critics_dir, "_".join([prefix, episode, "critic_2.pth"]))
        
        # Lưu checkpoints
        torch.save(self.train_actor.state_dict(), actor_file)
        torch.save(self.train_critic_1.state_dict(), critic_1_file)
        torch.save(self.train_critic_2.state_dict(), critic_2_file)
        print(f"Saved checkpoints at epoch {episode}")
    
    def load_ckpt(self, actor_path, critic_path1, critic_path2):
        # Load checkpoints
        try:
            self.train_actor.load_state_dict(torch.load(actor_path, map_location=self.device))
            print(f"Loaded actor from {actor_path}")
        except:
            print("Actor checkpoint cannot be loaded.")
        
        try:
            self.train_critic_1.load_state_dict(torch.load(critic_path1, map_location=self.device))
            self.train_critic_2.load_state_dict(torch.load(critic_path2, map_location=self.device))
            print(f"Loaded critics from {critic_path1} and {critic_path2}")
        except:
            print("Critic checkpoints cannot be loaded.")

    def train_mode(self):
        self.train_actor.train()
        self.train_critic_1.train()
        self.train_critic_2.train()

    def eval_mode(self):
        self.train_actor.eval()
        self.train_critic_1.eval()
        self.train_critic_2.eval()

    def freeze_networks(self):
        for p in chain(self.train_actor.parameters(), self.train_critic_1.parameters(), self.train_critic_2.parameters()):
            p.requires_grad = False

    def step_end(self):
        self.noise_generator.step_end()

    def episode_end(self):
        self.noise_generator.episode_end()  