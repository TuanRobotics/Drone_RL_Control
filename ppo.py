import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal
import gym

class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = self.network(state)
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        # Clamp log_std to prevent numerical instability
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        return mean, std
    
    def get_action(self, state, deterministic=False):
            """Sample action from policy"""
            mean, std = self.forward(state)
            
            if deterministic:
                return mean, None, None
            
            # Create Gaussian distribution
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
            
            return action, log_prob, dist.entropy().sum(dim=-1, keepdim=True)
    
    def evaluate_actions(self, state, action):
            """Evaluate log probability and entropy of given actions"""
            mean, std = self.forward(state)
            dist = Normal(mean, std)
            
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
            entropy = dist.entropy().sum(dim=-1, keepdim=True)
            
            return log_prob, entropy

# Critic 
class Critic(nn.Module):
    """Value network that estimates state value"""
    def __init__(self, state_dim, hidden_dim=256):
        super(Critic, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state):
        return self.network(state)

# Replay bufer 
class PPOMemory:
    """Buffer to store trajectories"""
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        
    def store(self, state, action, reward, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        
    def get_tensors(self, device):
        """Convert stored data to tensors"""
        states = torch.FloatTensor(np.array(self.states)).to(device)
        actions = torch.FloatTensor(np.array(self.actions)).to(device)
        rewards = torch.FloatTensor(np.array(self.rewards)).unsqueeze(1).to(device)
        dones = torch.FloatTensor(np.array(self.dones)).unsqueeze(1).to(device)
        log_probs = torch.FloatTensor(np.array(self.log_probs)).to(device)
        values = torch.FloatTensor(np.array(self.values)).to(device)
        
        return states, actions, rewards, dones, log_probs, values

class PPO:
    """Proximal Policy Optimization agent"""
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=256,
        lr_actor=3e-4,
        lr_critic=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        epochs=10,
        batch_size=64,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.epochs = epochs
        self.batch_size = batch_size
        
        # Initialize networks
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.critic = Critic(state_dim, hidden_dim).to(device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Memory
        self.memory = PPOMemory()
        
    def select_action(self, state, deterministic=False):
        """Select action from policy"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob, _ = self.actor.get_action(state, deterministic)
            value = self.critic(state)
            
        return action.cpu().numpy()[0], log_prob.cpu().item() if log_prob is not None else None, value.cpu().item()
    
    def store_transition(self, state, action, reward, done, log_prob, value):
        """Store transition in memory"""
        self.memory.store(state, action, reward, done, log_prob, value)
    
    def compute_gae(self, rewards, values, dones, next_value):
        """Compute Generalized Advantage Estimation"""
        advantages = []
        gae = 0
        
        # Convert to numpy for easier manipulation
        rewards = rewards.cpu().numpy().flatten()
        values = values.cpu().numpy().flatten()
        dones = dones.cpu().numpy().flatten()
        
        # Append next_value for bootstrap
        values = np.append(values, next_value)
        
        # Compute GAE backward from last step
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value_t = next_value
            else:
                next_value_t = values[t + 1]
                
            # TD error: delta = r + gamma * V(s') - V(s)
            delta = rewards[t] + self.gamma * next_value_t * (1 - dones[t]) - values[t]
            
            # GAE: A = delta + gamma * lambda * A_next
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        advantages = torch.FloatTensor(advantages).unsqueeze(1).to(self.device)
        returns = advantages + torch.FloatTensor(values[:-1]).unsqueeze(1).to(self.device)
        
        return advantages, returns
    
    def update(self, next_state):
        """Update policy using PPO algorithm"""
        # Get data from memory
        states, actions, rewards, dones, old_log_probs, values = self.memory.get_tensors(self.device)
        
        # Compute next value for GAE
        with torch.no_grad():
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            next_value = self.critic(next_state_tensor).cpu().item()
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert old_log_probs to tensor if needed
        old_log_probs = old_log_probs.detach()
        
        # Training statistics
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        
        # Multiple epochs of optimization
        dataset_size = states.shape[0]
        for epoch in range(self.epochs):
            # Generate random indices for mini-batches
            indices = np.random.permutation(dataset_size)
            
            # Mini-batch training
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                # Get mini-batch
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Evaluate actions
                new_log_probs, entropy = self.actor.evaluate_actions(batch_states, batch_actions)
                new_values = self.critic(batch_states)
                
                # Compute ratio for PPO
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Clipped surrogate objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.MSELoss()(new_values, batch_returns)
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = actor_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Update actor
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                
                # Accumulate statistics
                total_actor_loss += actor_loss.item()
                total_critic_loss += value_loss.item()
                total_entropy += entropy.mean().item()
        
        # Clear memory
        self.memory.clear()
        
        # Return average losses
        num_updates = self.epochs * (dataset_size // self.batch_size)
        return {
            'actor_loss': total_actor_loss / num_updates,
            'critic_loss': total_critic_loss / num_updates,
            'entropy': total_entropy / num_updates
        }
    
    def save(self, filepath):
        """Save model"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        print(f"Model loaded from {filepath}")