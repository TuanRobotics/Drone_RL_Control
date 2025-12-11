#!/usr/bin/env python3

import os
import time
import argparse
from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt

from PPO.ppo_agent import PPO, device
from gym_pybullet_drones.envs.FlyThruGateAvitary import FlyThruGateAvitary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.utils.utils import sync


def make_env(gui=False, record=False):
    return FlyThruGateAvitary(gui=gui,
                              record=record,
                              obs=ObservationType('kin'),
                              act=ActionType('rpm'))


def evaluate_policy(agent, episodes, max_steps, gui=False, seed=1234):
    """Run evaluation episodes without adding to the rollout buffer."""
    eval_env = make_env(gui=gui, record=False)
    rewards = []

    for ep in range(episodes):
        state, info = eval_env.reset(seed=seed + ep, options={})
        state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
        ep_reward = 0.0

        for _ in range(max_steps):
            state_tensor = torch.FloatTensor(state).to(device)
            with torch.no_grad():
                action_mean = agent.policy_old.actor(state_tensor)
            action = action_mean.cpu().numpy().flatten()
            action = np.expand_dims(action, axis=0)
            action = np.clip(action, eval_env.action_space.low,
                             eval_env.action_space.high)

            state, reward, terminated, truncated, _ = eval_env.step(action)
            state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
            ep_reward += reward

            if terminated or truncated:
                break

        rewards.append(ep_reward)

    eval_env.close()
    return float(np.mean(rewards)), rewards


def plot_training_curves(episode_rewards, eval_rewards, policy_losses,
                         value_losses, entropies, save_dir, label):
    """Plot and save basic PPO training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(episode_rewards, alpha=0.35, label='Episode reward')
    if len(episode_rewards) >= 50:
        smooth = np.convolve(episode_rewards, np.ones(50) / 50, mode='valid')
        axes[0, 0].plot(range(49, len(episode_rewards)),
                        smooth,
                        linewidth=2,
                        label='Smoothed (50 ep)')
    axes[0, 0].set_title('Training rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True)
    axes[0, 0].legend()

    if eval_rewards:
        axes[0, 1].plot(eval_rewards, marker='o')
        axes[0, 1].set_title('Evaluation rewards')
        axes[0, 1].set_xlabel('Eval step')
        axes[0, 1].set_ylabel('Average reward')
        axes[0, 1].grid(True)
    else:
        axes[0, 1].axis('off')

    if policy_losses:
        axes[1, 0].plot(policy_losses, alpha=0.6, label='Policy loss')
        axes[1, 0].set_title('Policy loss')
        axes[1, 0].set_xlabel('Update step')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True)
        axes[1, 0].legend()
    else:
        axes[1, 0].axis('off')

    if value_losses or entropies:
        axes[1, 1].set_title('Value loss / Entropy')
        axes[1, 1].set_xlabel('Update step')
        axes[1, 1].grid(True)
        if value_losses:
            axes[1, 1].plot(value_losses, color='tab:blue', label='Value loss')
        if entropies:
            ax2 = axes[1, 1].twinx()
            ax2.plot(entropies,
                     color='tab:orange',
                     alpha=0.6,
                     label='Entropy')
            ax2.set_ylabel('Entropy', color='tab:orange')
            ax2.tick_params(axis='y', labelcolor='tab:orange')
        axes[1, 1].legend(loc='upper left')
    else:
        axes[1, 1].axis('off')

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'training_curves_{label}.png'))
    plt.close()


def train(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = make_env(gui=args.gui, record=args.record)
    state, info = env.reset(seed=args.seed, options={})
    state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
    state_dim = state.shape[1] if len(state.shape) > 1 else state.shape[0]
    action_dim = (env.action_space.shape[1] if len(env.action_space.shape) > 1
                  else env.action_space.shape[0])

    agent = PPO(state_dim, action_dim, args.lr_actor, args.lr_critic,
                args.gamma, args.k_epochs, args.eps_clip, args.action_std)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("ppo_training", f"ppo_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    episode_rewards = []
    episode_lengths = []
    eval_rewards = []
    policy_losses = []
    value_losses = []
    entropies = []
    total_timesteps = 0

    start_time = datetime.now().replace(microsecond=0)
    print(f"Started PPO training at: {start_time}")

    for ep in range(args.episodes):
        state, info = env.reset(seed=args.seed + ep, options={})
        state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
        ep_reward = 0.0
        ep_len = 0
        frame_start = time.time()

        for step in range(args.max_steps):
            action = agent.select_action(state)
            action = np.expand_dims(action, axis=0)
            action = np.clip(action, env.action_space.low,
                             env.action_space.high)

            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = np.nan_to_num(next_state, nan=0.0, posinf=0.0, neginf=0.0)
            done = terminated or truncated

            agent.buffer.rewards.append(reward)
            agent.buffer.is_terminals.append(done)

            state = next_state
            ep_reward += reward
            ep_len += 1
            total_timesteps += 1

            if args.gui:
                env.render()
                sync(step, frame_start, env.CTRL_TIMESTEP)

            if (total_timesteps >= args.update_after
                    and total_timesteps % args.update_timestep == 0):
                pol_loss, val_loss, entropy = agent.update()
                policy_losses.append(pol_loss)
                value_losses.append(val_loss)
                entropies.append(entropy)

            if (args.std_decay_freq > 0
                    and total_timesteps % args.std_decay_freq == 0):
                agent.decay_action_std(args.std_decay_rate,
                                       args.min_action_std)

            if done:
                break

        if total_timesteps >= args.update_after and agent.buffer.rewards:
            pol_loss, val_loss, entropy = agent.update()
            policy_losses.append(pol_loss)
            value_losses.append(val_loss)
            entropies.append(entropy)

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_len)

        if (ep + 1) % args.log_every == 0:
            window = min(len(episode_rewards), args.log_every)
            avg_reward = np.mean(episode_rewards[-window:])
            print(f"Ep {ep+1}/{args.episodes} | Steps {total_timesteps} | "
                  f"Reward {ep_reward:.2f} | Avg({window}) {avg_reward:.2f} | "
                  f"Len {ep_len}")

        if args.eval_every > 0 and (ep + 1) % args.eval_every == 0:
            mean_eval, _ = evaluate_policy(agent,
                                           args.eval_episodes,
                                           args.max_steps,
                                           gui=False,
                                           seed=10_000 + ep)
            eval_rewards.append(mean_eval)
            print(f"[Eval] Episode {ep+1}: avg reward {mean_eval:.2f}")

    env.close()

    agent.save(os.path.join(save_dir, "ppo_model_final.pt"))
    plot_training_curves(episode_rewards, eval_rewards, policy_losses,
                         value_losses, entropies, save_dir, "final")

    end_time = datetime.now().replace(microsecond=0)
    print(f"{'='*60}")
    print(f"Training finished at: {end_time}")
    print(f"Total time: {end_time - start_time}")
    print(f"Model and plots saved to: {save_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PPO agent to fly through a gate.")
    parser.add_argument('--episodes', type=int, default=3000,
                        help='Number of training episodes.')
    parser.add_argument('--max_steps', type=int, default=1000,
                        help='Max steps per episode.')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor.')
    parser.add_argument('--lr_actor', type=float, default=3e-4,
                        help='Actor learning rate.')
    parser.add_argument('--lr_critic', type=float, default=1e-3,
                        help='Critic learning rate.')
    parser.add_argument('--k_epochs', type=int, default=80,
                        help='Number of epochs per PPO update.')
    parser.add_argument('--eps_clip', type=float, default=0.2,
                        help='Clipping epsilon for PPO.')
    parser.add_argument('--action_std', type=float, default=0.6,
                        help='Initial action std for exploration.')
    parser.add_argument('--min_action_std', type=float, default=0.1,
                        help='Minimum action std after decay.')
    parser.add_argument('--std_decay_rate', type=float, default=0.05,
                        help='Linear decay rate for action std.')
    parser.add_argument('--std_decay_freq', type=int, default=int(2.5e5),
                        help='Timesteps between std decays.')
    parser.add_argument('--update_after', type=int, default=2000,
                        help='Collect steps before the first update.')
    parser.add_argument('--update_timestep', type=int, default=50,
                        help='Timesteps between PPO updates.')
    parser.add_argument('--eval_every', type=int, default=0,
                        help='Evaluate every N episodes (0 to disable).')
    parser.add_argument('--eval_episodes', type=int, default=5,
                        help='Evaluation episodes per eval run.')
    parser.add_argument('--log_every', type=int, default=10,
                        help='Print training stats every N episodes.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility.')
    parser.add_argument('--gui', action='store_true',
                        help='Enable PyBullet GUI.')
    parser.add_argument('--record', action='store_true',
                        help='Record simulation video.')
    return parser.parse_args()


if __name__ == '__main__':
    train(parse_args())
