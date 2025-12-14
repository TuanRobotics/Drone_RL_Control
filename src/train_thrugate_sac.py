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

from agents.sac_agent import ReplayBuffer, SACAgent
from gym_pybullet_drones.envs.FlyThruGateAvitary import FlyThruGateAvitary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


# ============================================================================
# Training Function
# ============================================================================
def train_sac(num_episodes=5000,
              max_steps=500,
              batch_size=256,
              buffer_size=1000000,
              start_steps=10000,
              update_after=1000,
              update_every=50,
              save_every=100,
              eval_every=50,
              num_eval_episodes=5,
              gui=False):
    # Create environment
    env = FlyThruGateAvitary(drone_model=DroneModel.CF2X,
                             initial_xyzs=np.array([[0, 0, 0.5]]),
                             physics=Physics.PYB,
                             pyb_freq=240,
                             ctrl_freq=30,
                             gui=gui,
                             obs=ObservationType.KIN,
                             act=ActionType.RPM)

    # Get dimensions
    obs, _ = env.reset()
    state_dim = obs.shape[1]  # 12 - number of state
    action_dim = env.action_space.shape[1]  # 4 - number of action

    # Create agent and replay buffer
    agent = SACAgent(state_dim, action_dim, hidden_dim=256)
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
    save_dir = f"/home/tuan/Desktop/drone_rl_control/log_dir/sac_training_thrugate/sac_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    total_steps = 0

    print("\n" + "=" * 60)
    print("Starting SAC Training for Drone Gate Navigation")
    print("=" * 60 + "\n")

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0

        for step in range(env.EPISODE_LEN_SEC * env.CTRL_FREQ):
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
            replay_buffer.push(state, action, reward, next_state, done
                               or truncated)

            state = next_state

            # Update agent
            if total_steps >= update_after and total_steps % update_every == 0:
                for _ in range(update_every):
                    q1_loss, q2_loss, policy_loss = agent.update(
                        replay_buffer, batch_size)
                    q1_losses.append(q1_loss)
                    q2_losses.append(q2_loss)
                    policy_losses.append(policy_loss)

            if done or truncated:
                break

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        # Logging
        if (episode + 1) % 10 == 0:
            avg_reward_10 = np.mean(episode_rewards[-10:])
            print(f"Episode {episode+1}/{num_episodes} | "
                  f"Steps: {total_steps} | "
                  f"Reward: {episode_reward:.2f} | "
                  f"Avg(10): {avg_reward_10:.2f} | "
                  f"Length: {episode_length}")
        
        avg_reward_100 = np.mean(episode_rewards[-100:])
        # Evaluation
        if (episode + 1) % eval_every == 0:
            eval_reward = evaluate_policy(env, agent, num_eval_episodes)
            eval_rewards.append(eval_reward)
            print(f"\n{'='*60}")
            print(
                f"Avg Reward over 100 episodes: {avg_reward_100:.2f}\n"
                f"Evaluation at Episode {episode+1}: Avg Evaluated Reward = {eval_reward:.2f}"
            )
            print(f"{'='*60}\n")

        # Early stopping if solved
        # if avg_reward_100 >= 120:
        #     print(f"\nEnvironment solved in {episode+1} episodes!")
        #     print(f"Average Reward: {avg_reward_100:.2f}")
        #     print(f"{'='*60}\n")
        #     break

        # # Save model
        # if (episode + 1) % save_every == 0:
        #     # agent.save(os.path.join(save_dir, f"sac_model_ep{episode+1}.pt"))
        #     plot_training_curves(episode_rewards, eval_rewards, q1_losses,
        #                          policy_losses, save_dir, episode + 1)

    env.close()

    # Final save + plot
    print(f"{'='*60}\n")
    agent.save(os.path.join(save_dir, "sac_model_final.pt"))
    plot_training_curves(episode_rewards, eval_rewards, q1_losses,
                         policy_losses, save_dir, "final")

    print(f"\nTraining completed! Final model saved in {save_dir}")

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
        num_episodes=10000,
        batch_size=64,
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