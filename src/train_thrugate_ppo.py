#!/usr/bin/env python3
import os
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

from agents.ppo_agent import PPO
from gym_pybullet_drones.envs.FlyThruGateAvitary import FlyThruGateAvitary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


def _reshape_action_for_env(action, action_space):
    """Ensure the action matches the environment's expected shape."""
    action = np.asarray(action, dtype=np.float32)
    if action.ndim == 1:
        action = np.expand_dims(action, axis=0)
    try:
        action = action.reshape(action_space.shape)
    except Exception:
        pass
    return action


# ============================================================================
# Training
# ============================================================================

def train_ppo():
    
    """Train PPO agent, mirroring SAC logging/plotting workflow."""
    env = FlyThruGateAvitary(drone_model=DroneModel.CF2X,
                             initial_xyzs=np.array([[0, 0, 0.5]]),
                             physics=Physics.PYB,
                             pyb_freq=240,
                             ctrl_freq=30,
                             gui=False,
                             obs=ObservationType.KIN,
                             act=ActionType.RPM)

    obs, _ = env.reset()
    state_dim = 12
    action_dim = 4

    # Hyperparameters PPO 
    K_epochs=80
    eps_clip=0.2
    gamma=0.99
    lr_actor=3e-4
    lr_critic=1e-3
    action_std=0.6
    action_std_decay_rate=0.05
    min_action_std=0.1
    action_std_decay_freq=int(2.5e5)
    eval_every=50
    num_eval_episodes=5

    agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs,
                eps_clip, action_std)

    # Metrics
    episode_rewards = []
    eval_rewards = []
    critic_losses = []
    actor_losses = []
    entropy_hist = []

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path("/home/tuan/Desktop/drone_rl_control/log_dir")
    save_dir = base_dir / "ppo_training_thrugate" / f"ppo_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_prefix = base_dir / f"ppo_metrics_thrugate{timestamp}"
    log_f_name = save_dir / 'PPO_log.csv'
    log_f = open(log_f_name, "w+")
    log_f.write('episode,timestep,reward, averagate_reward_100\n')

    print("\n" + "=" * 60)
    print("Starting PPO Training for Drone Gate Navigation")
    print("=" * 60 + "\n")

    total_timesteps = 0
    update_timestep = env.EPISODE_LEN_SEC*env.CTRL_FREQ * 4
    log_freq =  env.EPISODE_LEN_SEC*env.CTRL_FREQ * 2
    max_training_timesteps = int(3e6)
    total_timesteps = 0
    i_episode = 0
    log_running_reward = 0
    losses = []
    loss_steps = []

    while total_timesteps <= max_training_timesteps:
        state, _ = env.reset(seed=42, options={})
        episode_reward = 0

        for i in range(env.EPISODE_LEN_SEC * env.CTRL_FREQ):
            action = agent.select_action(state)
            action_env = _reshape_action_for_env(action, env.action_space)

            next_state, reward, terminated, truncated, _ = env.step(action_env)
            done = terminated or truncated 

            agent.buffer.rewards.append(reward)
            agent.buffer.is_terminals.append(done)

            total_timesteps += 1
            episode_reward += reward
            state = next_state

            if total_timesteps % update_timestep == 0:
                loss_obj = agent.update()
                losses.append(loss_obj)
                loss_steps.append(total_timesteps)

            if total_timesteps % action_std_decay_freq == 0:
                agent.decay_action_std(action_std_decay_rate, min_action_std)

            if total_timesteps % log_freq == 0:

                # log average reward till last episode
                log_avg_reward = np.mean(episode_rewards) if episode_rewards else 0
                log_avg_reward = round(log_avg_reward, 4)
                log_avg_reward_100 = np.mean(avg_reward_100) if episode_rewards else 0
                log_avg_reward_100 = round(log_avg_reward_100, 4)

                log_f.write('{}, {}, {}, {}\n'.format(i_episode, total_timesteps, log_avg_reward, log_avg_reward_100))
                log_f.flush()

            if done:
                break
        
        episode_rewards.append(episode_reward)
        avg_reward_100 = np.mean(episode_rewards[-100:])
        i_episode += 1

        if (i_episode) % eval_every == 0:
            eval_reward = evaluate_policy(env, agent, num_eval_episodes)
            eval_rewards.append(eval_reward)
            print(f"\n{'='*60}")
            print(
                f"Episode {i_episode} | "
                f"Avg Reward over 100 episodes: {avg_reward_100:.2f} | "
                f"Evaluation at Episode {i_episode}: Avg Evaluated Reward = {eval_reward:.2f}"
            )

            print(f"{'='*60}\n")
    env.close()

    print(f"{'='*60}\n")
    agent.save(save_dir / "ppo_model_final.pt")

    # Plot losses 
    plt.figure()
    plt.plot(loss_steps, losses)
    plt.xlabel("Timesteps")
    plt.ylabel("PPO Loss")
    plt.title("PPO Loss over Time")
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_dir / "ppo_loss.png")
    plt.close()

    print(f"\nTraining completed! Final model saved in {save_dir}")


def evaluate_policy(env, agent, num_episodes=5):
    """Evaluate the policy without modifying PPO buffers."""
    eval_rewards = []
    policy_device = next(agent.policy_old.parameters()).device

    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            state_tensor = torch.as_tensor(state,
                                           dtype=torch.float32,
                                           device=policy_device)
            with torch.no_grad():
                action_mean = agent.policy_old.actor(state_tensor)
            action = action_mean.cpu().numpy()
            action_env = _reshape_action_for_env(action, env.action_space)

            next_state, reward, terminated, truncated, _ = env.step(action_env)
            episode_reward += reward
            state = next_state
            done = terminated or truncated

        eval_rewards.append(episode_reward)

    return float(np.mean(eval_rewards))

# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    train_ppo()
