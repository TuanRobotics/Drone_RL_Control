#!/usr/bin/env python3

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

from FlyThruGateAvitary import FlyThruGateAvitary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

# Thiết lập device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================================
# Replay Buffer
# ============================================================================
class ReplayBuffer:
    def __init__(self, capacity=1000000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
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
        x = self.fc3(x)
        return x


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
    def __init__(self, state_dim, action_dim, hidden_dim=256,
                 lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2, 
                 automatic_entropy_tuning=True):
        
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.automatic_entropy_tuning = automatic_entropy_tuning
        
        # Actor
        self.policy = GaussianPolicy(state_dim, action_dim, hidden_dim).to(device)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        
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
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1).to(device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(device)
        done_batch = torch.FloatTensor(done_batch).unsqueeze(1).to(device)
        
        with torch.no_grad():
            next_action, next_log_prob, _ = self.policy.sample(next_state_batch)
            q1_next_target = self.q1_target(next_state_batch, next_action)
            q2_next_target = self.q2_target(next_state_batch, next_action)
            min_q_next_target = torch.min(q1_next_target, q2_next_target) - self.alpha * next_log_prob
            next_q_value = reward_batch + (1 - done_batch) * self.gamma * min_q_next_target
        
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
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp()
        
        # Soft update target networks
        self.soft_update(self.q1, self.q1_target)
        self.soft_update(self.q2, self.q2_target)
        
        return q1_loss.item(), q2_loss.item(), policy_loss.item()
    
    def soft_update(self, source, target):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
    
    def save(self, filepath):
        torch.save({
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


# ============================================================================
# Training Function
# ============================================================================
def train_sac(
    num_episodes=5000,
    max_steps=1000,
    batch_size=256,
    buffer_size=1000000,
    start_steps=10000,
    update_after=1000,
    update_every=50,
    save_every=100,
    eval_every=50,
    num_eval_episodes=5,
    gui=False
):
    # Create environment
    env = FlyThruGateAvitary(
        drone_model=DroneModel.CF2X,
        initial_xyzs=np.array([[0, 0, 0.5]]),
        physics=Physics.PYB,
        pyb_freq=240,
        ctrl_freq=30,
        gui=gui,
        obs=ObservationType.KIN,
        act=ActionType.RPM
    )
    
    # Get dimensions
    obs = env.reset()
    state_dim = obs.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Action space: {env.action_space}")
    
    # Create agent and replay buffer
    agent = SACAgent(state_dim, action_dim)
    replay_buffer = ReplayBuffer(buffer_size)
    
    # Metrics
    episode_rewards = []
    episode_lengths = []
    eval_rewards = []
    q1_losses = []
    q2_losses = []
    policy_losses = []
    
    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"sac_drone_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    total_steps = 0
    
    print("\n" + "="*60)
    print("Starting SAC Training for Drone Gate Navigation")
    print("="*60 + "\n")
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(max_steps):
            # Select action
            if total_steps < start_steps:
                action = env.action_space.sample()  # Random action
            else:
                action = agent.select_action(state)
            
            # Execute action
            next_state, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            total_steps += 1
            
            # Store transition
            replay_buffer.push(state, action, reward, next_state, done or truncated)
            
            state = next_state
            
            # Update agent
            if total_steps >= update_after and total_steps % update_every == 0:
                for _ in range(update_every):
                    q1_loss, q2_loss, policy_loss = agent.update(replay_buffer, batch_size)
                    q1_losses.append(q1_loss)
                    q2_losses.append(q2_loss)
                    policy_losses.append(policy_loss)
            
            if done or truncated:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Logging
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_length = np.mean(episode_lengths[-10:])
            print(f"Episode {episode+1}/{num_episodes} | "
                  f"Steps: {total_steps} | "
                  f"Reward: {episode_reward:.2f} | "
                  f"Avg(10): {avg_reward:.2f} | "
                  f"Length: {episode_length}")
        
        # Evaluation
        if (episode + 1) % eval_every == 0:
            eval_reward = evaluate_policy(env, agent, num_eval_episodes)
            eval_rewards.append(eval_reward)
            print(f"\n{'='*60}")
            print(f"Evaluation at Episode {episode+1}: Avg Reward = {eval_reward:.2f}")
            print(f"{'='*60}\n")
        
        # Save model
        if (episode + 1) % save_every == 0:
            agent.save(os.path.join(save_dir, f"sac_model_ep{episode+1}.pt"))
            plot_training_curves(episode_rewards, eval_rewards, q1_losses, 
                               policy_losses, save_dir, episode+1)
    
    env.close()
    
    # Final save
    agent.save(os.path.join(save_dir, "sac_model_final.pt"))
    plot_training_curves(episode_rewards, eval_rewards, q1_losses, 
                        policy_losses, save_dir, "final")
    
    print(f"\nTraining completed! Models saved in {save_dir}")
    
    return agent, episode_rewards, eval_rewards


def evaluate_policy(env, agent, num_episodes=5):
    """Evaluate the policy without exploration noise"""
    eval_rewards = []
    
    for _ in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state, evaluate=True)
            next_state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            state = next_state
            
            if done or truncated:
                break
        
        eval_rewards.append(episode_reward)
    
    return np.mean(eval_rewards)


def plot_training_curves(episode_rewards, eval_rewards, q1_losses, 
                         policy_losses, save_dir, episode):
    """Plot and save training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Episode rewards
    axes[0, 0].plot(episode_rewards, alpha=0.3, label='Episode Reward')
    if len(episode_rewards) >= 50:
        smoothed = np.convolve(episode_rewards, np.ones(50)/50, mode='valid')
        axes[0, 0].plot(range(49, len(episode_rewards)), smoothed, 
                       label='Smoothed (50 episodes)', linewidth=2)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Training Rewards')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Evaluation rewards
    if eval_rewards:
        axes[0, 1].plot(eval_rewards, marker='o')
        axes[0, 1].set_xlabel('Evaluation Step')
        axes[0, 1].set_ylabel('Average Reward')
        axes[0, 1].set_title('Evaluation Rewards')
        axes[0, 1].grid(True)
    
    # Q losses
    if q1_losses:
        axes[1, 0].plot(q1_losses, alpha=0.3)
        if len(q1_losses) >= 100:
            smoothed = np.convolve(q1_losses, np.ones(100)/100, mode='valid')
            axes[1, 0].plot(range(99, len(q1_losses)), smoothed, linewidth=2)
        axes[1, 0].set_xlabel('Update Step')
        axes[1, 0].set_ylabel('Q1 Loss')
        axes[1, 0].set_title('Critic Loss')
        axes[1, 0].grid(True)
    
    # Policy losses
    if policy_losses:
        axes[1, 1].plot(policy_losses, alpha=0.3)
        if len(policy_losses) >= 100:
            smoothed = np.convolve(policy_losses, np.ones(100)/100, mode='valid')
            axes[1, 1].plot(range(99, len(policy_losses)), smoothed, linewidth=2)
        axes[1, 1].set_xlabel('Update Step')
        axes[1, 1].set_ylabel('Policy Loss')
        axes[1, 1].set_title('Actor Loss')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'training_curves_ep{episode}.png'))
    plt.close()


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    # Training configuration
    agent, episode_rewards, eval_rewards = train_sac(
        num_episodes=5000,
        max_steps=1000,
        batch_size=256,
        buffer_size=1000000,
        start_steps=10000,      # Random exploration steps
        update_after=1000,      # Start training after this many steps
        update_every=50,        # Update frequency
        save_every=100,         # Save model every N episodes
        eval_every=50,          # Evaluate every N episodes
        num_eval_episodes=5,    # Number of episodes for evaluation
        gui=False               # Set True to see visualization
    )
    
    print("\nTraining Summary:")
    print(f"Final average reward (last 100 episodes): {np.mean(episode_rewards[-100:]):.2f}")
    print(f"Best evaluation reward: {max(eval_rewards):.2f}")
