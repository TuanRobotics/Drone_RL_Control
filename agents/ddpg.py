import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random

# Thiết lập device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================================
# Replay Buffer
# ============================================================================
class ReplayBuffer:
    def __init__(self, capacity=1000000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """Add experience to replay buffer"""
        # Nếu state / next_state là tuple (obs, info, ...) thì chỉ lấy obs
        if isinstance(state, (tuple, list)):
            state = state[0]
        if isinstance(next_state, (tuple, list)):
            next_state = next_state[0]
        
        # Ép về vector 1D float32 (cùng shape cho mỗi mẫu)
        state = np.array(state, dtype=np.float32).reshape(-1)
        next_state = np.array(next_state, dtype=np.float32).reshape(-1)
        action = np.array(action, dtype=np.float32).reshape(-1)
        
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample a batch of experiences"""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            np.array(state),
            np.array(action),
            np.array(reward, dtype=np.float32),
            np.array(next_state),
            np.array(done, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)


# ============================================================================
# Ornstein-Uhlenbeck Noise (for DDPG exploration)
# ============================================================================
class OUNoise:
    """Ornstein-Uhlenbeck process for temporal correlated noise"""
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu
        self.reset()
    
    def reset(self):
        """Reset the internal state to mean"""
        self.state = np.ones(self.action_dim) * self.mu
    
    def sample(self):
        """Generate noise sample"""
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state


# ============================================================================
# Actor Network
# ============================================================================
class Actor(nn.Module):
    """Deterministic policy network"""
    def __init__(self, state_dim, action_dim, hidden_dim=256, max_action=1.0):
        super(Actor, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        self.max_action = max_action
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        
        nn.init.constant_(self.fc1.bias, 0.1)
        nn.init.constant_(self.fc2.bias, 0.1)
        nn.init.uniform_(self.fc3.bias, -3e-3, 3e-3)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x * self.max_action


# ============================================================================
# Critic Network (Single Q-network for DDPG)
# ============================================================================
class Critic(nn.Module):
    """Q-function network for DDPG (single critic)"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        
        # Q architecture
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        
        nn.init.constant_(self.fc1.bias, 0.1)
        nn.init.constant_(self.fc2.bias, 0.1)
        nn.init.uniform_(self.fc3.bias, -3e-3, 3e-3)
    
    def forward(self, state, action):
        """Return Q-value"""
        sa = torch.cat([state, action], dim=1)
        
        q = F.relu(self.fc1(sa))
        q = F.relu(self.fc2(q))
        q = self.fc3(q)
        
        return q


# ============================================================================
# DDPG Agent
# ============================================================================
class DDPGAgent:
    def __init__(self,
                 actor_class,
                 critic_class,
                 state_size,
                 action_size,
                 clip_low=-1,
                 clip_high=1,
                 hidden_dim=256,
                 lr_actor=1e-4,
                 lr_critic=1e-3,
                 gamma=0.99,
                 tau=0.001,
                 batch_size=64,
                 buffer_size=1000000,
                 exploration_noise=0.1,
                 noise_type='gaussian'):
        """
        DDPG Agent initialization
        
        Args:
            actor_class: Actor network class
            critic_class: Critic network class
            state_size: Dimension of state space
            action_size: Dimension of action space
            clip_low: Lower bound for action clipping
            clip_high: Upper bound for action clipping
            hidden_dim: Hidden layer dimension
            lr_actor: Learning rate for actor
            lr_critic: Learning rate for critic
            gamma: Discount factor
            tau: Soft update parameter
            batch_size: Batch size for training
            buffer_size: Replay buffer capacity
            exploration_noise: Std of exploration noise
            noise_type: Type of noise ('gaussian' or 'ou')
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.exploration_noise = exploration_noise
        self.clip_low = clip_low
        self.clip_high = clip_high
        self.noise_type = noise_type
        
        self.max_action = float(clip_high)
        
        # Actor networks
        self.actor = actor_class(state_size, action_size, hidden_dim, 
                                self.max_action).to(device)
        self.actor_target = actor_class(state_size, action_size, hidden_dim,
                                       self.max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), 
                                               lr=lr_actor)
        
        # Critic networks (single Q-network for DDPG)
        self.critic = critic_class(state_size, action_size, hidden_dim).to(device)
        self.critic_target = critic_class(state_size, action_size, 
                                         hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                lr=lr_critic)
        
        # Replay buffer
        self.memory = ReplayBuffer(buffer_size)
        
        # Exploration noise
        if noise_type == 'ou':
            self.noise = OUNoise(action_size, sigma=exploration_noise)
        else:
            self.noise = None
        
        # Training step counter
        self.total_it = 0
        
        # Mode flag
        self.is_training = True
    
    def get_action(self, state, explore=True):
        """
        Select action based on current policy
        
        Args:
            state: Current state
            explore: Whether to add exploration noise
        """
        # Nếu state là tuple/list (obs, info) thì chỉ lấy obs
        if isinstance(state, (tuple, list)):
            state = state[0]
        
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()[0]
        
        # Add exploration noise
        if explore and self.is_training:
            if self.noise_type == 'ou' and self.noise is not None:
                # Ornstein-Uhlenbeck noise
                noise = self.noise.sample()
            else:
                # Gaussian noise
                noise = np.random.normal(0, self.exploration_noise, 
                                        size=self.action_size)
            
            action = action + noise
            action = np.clip(action, self.clip_low, self.clip_high)
        
        return action
    
    def learn_one_step(self):
        """
        Update actor and critic networks (one training step)
        DDPG updates both networks every step (no delayed update)
        """
        if len(self.memory) < self.batch_size:
            return None, None
        
        self.total_it += 1
        
        # Sample batch from replay buffer
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = \
            self.memory.sample(self.batch_size)
        
        state = torch.FloatTensor(state_batch).to(device)
        action = torch.FloatTensor(action_batch).to(device)
        reward = torch.FloatTensor(reward_batch).unsqueeze(1).to(device)
        next_state = torch.FloatTensor(next_state_batch).to(device)
        done = torch.FloatTensor(done_batch).unsqueeze(1).to(device)
        
        # ========== Update Critic ==========
        with torch.no_grad():
            # Select next action from target actor (no noise added in DDPG target)
            next_action = self.actor_target(next_state)
            
            # Compute target Q-value
            target_q = self.critic_target(next_state, next_action)
            target_q = reward + (1 - done) * self.gamma * target_q
        
        # Get current Q estimate
        current_q = self.critic(state, action)
        
        # Compute critic loss
        critic_loss = F.mse_loss(current_q, target_q)
        
        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()
        
        # ========== Update Actor (every step in DDPG) ==========
        # Compute actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()
        
        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
        
        # ========== Soft update target networks ==========
        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)
        
        return critic_loss.item(), actor_loss.item()
    
    def soft_update(self, source, target):
        """Soft update model parameters: θ' ← τθ + (1-τ)θ'"""
        for target_param, param in zip(target.parameters(), 
                                      source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau)
    
    def reset_noise(self):
        """Reset OU noise (if using)"""
        if self.noise_type == 'ou' and self.noise is not None:
            self.noise.reset()
    
    def set_noise_std(self, noise_std):
        """Update exploration noise standard deviation"""
        self.exploration_noise = noise_std
        if self.noise_type == 'ou' and self.noise is not None:
            self.noise.sigma = noise_std
    
    def train_mode(self):
        """Set agent to training mode"""
        self.is_training = True
        self.actor.train()
        self.critic.train()
    
    def eval_mode(self):
        """Set agent to evaluation mode"""
        self.is_training = False
        self.actor.eval()
        self.critic.eval()
    
    def step_end(self):
        """Called at the end of each step (for compatibility)"""
        pass
    
    def episode_end(self):
        """Called at the end of each episode (for compatibility)"""
        pass
    
    def save(self, filepath):
        """
        Save model checkpoint
        
        Args:
            filepath: Path to save checkpoint
        """
        torch.save({
            'total_it': self.total_it,
            'actor_state_dict': self.actor.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'exploration_noise': self.exploration_noise,
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """
        Load model checkpoint
        
        Args:
            filepath: Path to checkpoint file
        """
        checkpoint = torch.load(filepath, map_location=device)
        
        self.total_it = checkpoint['total_it']
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        
        if 'exploration_noise' in checkpoint:
            self.exploration_noise = checkpoint['exploration_noise']
        
        print(f"Model loaded from {filepath}")
        print(f"Resuming from iteration {self.total_it}")