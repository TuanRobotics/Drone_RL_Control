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
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x * self.max_action


# ============================================================================
# Critic Network
# ============================================================================
class Critic(nn.Module):
    """Q-function network (Twin critics for TD3)"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        
        # Q1 architecture
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        # Q2 architecture
        self.fc4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, 1)
    
    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        
        # Q1
        q1 = F.relu(self.fc1(sa))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)
        
        # Q2
        q2 = F.relu(self.fc4(sa))
        q2 = F.relu(self.fc5(q2))
        q2 = self.fc6(q2)
        
        return q1, q2
    
    def Q1(self, state, action):
        """Return only Q1 value"""
        sa = torch.cat([state, action], dim=1)
        q1 = F.relu(self.fc1(sa))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)
        return q1


# ============================================================================
# TD3 Agent
# ============================================================================
class TD3Agent:
    def __init__(self,
                 actor_class,
                 critic_class,
                 state_size,
                 action_size,
                 clip_low=-1,
                 clip_high=1,
                 hidden_dim=64,
                 lr_actor=3e-4,
                 lr_critic=3e-4,
                 gamma=0.99,
                 tau=0.005,
                 policy_noise=0.2,
                 noise_clip=0.5,
                 policy_freq=2,
                 batch_size=256,
                 buffer_size=1000000,
                 exploration_noise=0.1):
        """
        TD3 Agent initialization
        
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
            policy_noise: Noise added to target policy
            noise_clip: Range to clip target policy noise
            policy_freq: Frequency of delayed policy updates
            batch_size: Batch size for training
            buffer_size: Replay buffer capacity
            exploration_noise: Std of Gaussian exploration noise
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.batch_size = batch_size
        self.exploration_noise = exploration_noise
        self.clip_low = clip_low
        self.clip_high = clip_high
        
        self.max_action = float(clip_high)
        
        # Actor networks
        self.actor = actor_class(state_size, action_size, hidden_dim, 
                                self.max_action).to(device)
        self.actor_target = actor_class(state_size, action_size, hidden_dim,
                                       self.max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), 
                                               lr=lr_actor)
        
        # Critic networks
        self.critic = critic_class(state_size, action_size, hidden_dim).to(device)
        self.critic_target = critic_class(state_size, action_size, 
                                         hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                lr=lr_critic)
        
        # Replay buffer
        self.memory = ReplayBuffer(buffer_size)
        
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
            noise = np.random.normal(0, self.exploration_noise, 
                                    size=self.action_size)
            action = action + noise
            action = np.clip(action, self.clip_low, self.clip_high)
        
        return action
    
    def learn_one_step(self):
        """
        Update actor and critic networks (one training step)
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
            # Select action according to target policy with added noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip)
            
            next_action = (self.actor_target(next_state) + noise).clamp(
                self.clip_low, self.clip_high)
            
            # Compute target Q-values
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (1 - done) * self.gamma * target_q
        
        # Get current Q estimates
        current_q1, current_q2 = self.critic(state, action)
        
        # Compute critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + \
                     F.mse_loss(current_q2, target_q)
        
        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # ========== Delayed Policy Update ==========
        actor_loss_value = None
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            actor_loss_value = actor_loss.item()
            
            # Optimize actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Soft update target networks
            self.soft_update(self.actor, self.actor_target)
            self.soft_update(self.critic, self.critic_target)
        
        return critic_loss.item(), actor_loss_value
    
    def soft_update(self, source, target):
        """Soft update model parameters"""
        for target_param, param in zip(target.parameters(), 
                                      source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau)
    
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

        torch.save({
            'total_it': self.total_it,
            'actor_state_dict': self.actor.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
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
        
        print(f"Model loaded from {filepath}")
        print(f"Resuming from iteration {self.total_it}")


# ============================================================================
# Convenience function to create TD3 agent
# ============================================================================
def create_td3_agent(state_dim, action_dim, **kwargs):
    """
    Convenience function to create TD3 agent with default parameters
    
    Args:
        state_dim: State space dimension
        action_dim: Action space dimension
        **kwargs: Additional parameters to override defaults
    
    Returns:
        TD3Agent instance
    """
    return TD3Agent(
        Actor,
        Critic,
        state_size=state_dim,
        action_size=action_dim,
        **kwargs
    )

# # ============================================================================
# # Example usage
# # ============================================================================
# if __name__ == "__main__":
#     print("Testing TD3 Agent...")
    
#     # Create agent
#     agent = TD3Agent(
#         Actor,
#         Critic,
#         state_size=12,
#         action_size=4,
#         clip_low=-1,
#         clip_high=1
#     )
    
#     print(f"\nAgent created successfully!")
#     print(f"State size: {agent.state_size}")
#     print(f"Action size: {agent.action_size}")
#     print(f"Device: {device}")
    
#     # Test action selection
#     dummy_state = np.random.randn(12)
#     action = agent.get_action(dummy_state, explore=True)
#     print(f"\nTest action (with exploration): {action}")
#     print(f"Action shape: {action.shape}")
    
#     action_eval = agent.get_action(dummy_state, explore=False)
#     print(f"Test action (without exploration): {action_eval}")
    
#     # Test memory operations
#     agent.memory.add(
#         dummy_state,
#         action,
#         1.0,
#         np.random.randn(12),
#         False
#     )
#     print(f"\nMemory size: {len(agent.memory)}")
    
#     print("\n✓ All tests passed!")
