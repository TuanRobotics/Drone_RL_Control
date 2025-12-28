#!/usr/bin/env python3
import os
from collections import deque
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

from agents.ppo_agent import PPO
from gym_pybullet_drones.envs.FlyThruGateAvitary import FlyThruGateAvitary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


def _reshape_action_for_env(action, action_space):
    """Ensure actions always match the env's expected (NUM_DRONES, 4) shape."""
    action = np.asarray(action, dtype=np.float32)
    if action.ndim == 1:
        action = np.expand_dims(action, axis=0)
    try:
        action = action.reshape(action_space.shape)
    except Exception:
        pass
    return action


def _update_curriculum_on_rate(env, success_history, success_rate_threshold,
                               success_window):
    """Increase curriculum when success rate over last window exceeds threshold."""
    if env.curriculum_level >= env.max_curriculum_level:
        return  # Already at max level

    if len(success_history) < success_window:
        return

    success_rate = sum(success_history) / float(len(success_history))
    if success_rate > success_rate_threshold:
        env.curriculum_level += 1
        print(f"[Curriculum] Level advanced to {env.curriculum_level} "
              f"(success_rate={success_rate:.2f} over last {len(success_history)} episodes)")
        success_history.clear()


def _plot_rewards(episode_rewards, save_dir, tag):
    """Plot episode rewards and a smoothed curve."""
    if not episode_rewards:
        return
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, alpha=0.35, label="Episode reward")
    if len(episode_rewards) >= 50:
        smoothed = np.convolve(episode_rewards, np.ones(50) / 50, mode='valid')
        plt.plot(range(49, len(episode_rewards)), smoothed,
                 label="Smoothed (50 episodes)", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("PPO training rewards")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"ppo_rewards_{tag}.png"))
    plt.close()


# ============================================================================
# Training Function
# ============================================================================
def train_ppo_curriculum(num_episodes=10000,
                         gamma=0.99,
                         lr_actor=3e-4,
                         lr_critic=1e-3,
                         K_epochs=80,
                         eps_clip=0.2,
                         action_std=0.6,
                         action_std_decay=0.0,
                         min_action_std=0.1,
                         action_std_decay_freq=50,
                         curriculum_start_level=0,
                         max_curriculum_level=5,
                         success_window=30,
                         success_rate_threshold=0.5,
                         gui=False):
    """Train PPO agent with curriculum-enabled FlyThruGateAvitary.

    During training, only rewards are logged/saved (no critic/actor loss curves).
    """
    env = FlyThruGateAvitary(drone_model=DroneModel.CF2X,
                             initial_xyzs=np.array([[0, 0, 0.5]]),
                             physics=Physics.PYB,
                             pyb_freq=240,
                             ctrl_freq=30,
                             gui=gui,
                             obs=ObservationType.KIN,
                             act=ActionType.RPM,
                             use_curriculum=True,
                             curriculum_level=curriculum_start_level,
                             max_curriculum_level=max_curriculum_level)

    obs, _ = env.reset()
    state_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma,
                K_epochs, eps_clip, action_std_init=action_std)
    
    # agent.load("/home/tuan/Desktop/drone_rl_control/log_dir/ppo_training_thrugate_curriculum/ppo_20251217_183828/ppo_model_final.pt")
    # print("Loaded pretrained PPO model.")
    episode_rewards = []
    curriculum_levels = []

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"/home/tuan/Desktop/drone_rl_control/log_dir/ppo_training_thrugate_curriculum/ppo_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    reward_csv_path = os.path.join(save_dir, "episode_rewards.csv")

    with open(reward_csv_path, "w", newline="") as f:
        f.write("episode,reward,curriculum_level\n")

    success_history = deque(maxlen=success_window)

    print("\n" + "=" * 60)
    print("Starting PPO Training with Curriculum for Drone Gate Navigation")
    print("=" * 60 + "\n")

    for episode in range(num_episodes):
        env.success_passed = False
        state, _ = env.reset()
        episode_reward = 0.0

        for _ in range(env.EPISODE_LEN_SEC * env.CTRL_FREQ):
            action = agent.select_action(state)
            action_env = _reshape_action_for_env(action, env.action_space)

            next_state, reward, terminated, truncated, _ = env.step(action_env)
            done = terminated or truncated

            agent.buffer.rewards.append(reward)
            agent.buffer.is_terminals.append(done)

            episode_reward += reward
            state = next_state

            if done:
                break

        agent.update()

        # Optional action std decay (by episode)
        if action_std_decay > 0 and (episode + 1) % action_std_decay_freq == 0:
            agent.decay_action_std(action_std_decay, min_action_std)

        episode_rewards.append(episode_reward)

        success = getattr(env, "center_gate_passed", False)
        success_history.append(1 if success else 0)
        _update_curriculum_on_rate(env, success_history,
                                   success_rate_threshold, success_window)
        curriculum_levels.append(env.curriculum_level)

        with open(reward_csv_path, "a", newline="") as f:
            f.write(f"{episode + 1},{episode_reward},{env.curriculum_level}\n")

        if (episode + 1) % 10 == 0:
            avg_reward_10 = np.mean(episode_rewards[-10:])
            recent_rate = float(np.mean(success_history)) if success_history else 0.0
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Reward: {episode_reward:.2f} | "
                  f"Avg(10): {avg_reward_10:.2f} | "
                  f"Curriculum: L{env.curriculum_level}/{env.max_curriculum_level} | "
                  f"Success rate (last {len(success_history)}): {recent_rate:.2f}")
        if episode % 2000 == 0 and episode > 0:
            agent.save(os.path.join(save_dir, f"ppo_model_ep{episode}.pt"))
            _plot_rewards(episode_rewards, save_dir, f"ep{episode}")

    env.close()

    agent.save(os.path.join(save_dir, "ppo_model_final.pt"))
    _plot_rewards(episode_rewards, save_dir, "final")

    print(f"\nTraining completed! Final model and reward logs saved to {save_dir}")

    return agent, episode_rewards, curriculum_levels


if __name__ == "__main__":
    train_ppo_curriculum(
        num_episodes=30000,
        gamma=0.99,
        lr_actor=3e-4,
        lr_critic=1e-3,
        K_epochs=80,
        eps_clip=0.2,
        action_std=0.6,
        action_std_decay=0.7,
        min_action_std=0.1,
        action_std_decay_freq=5000,
        curriculum_start_level=0,
        max_curriculum_level=5,
        success_window=30,
        success_rate_threshold=0.75,
        gui=False
    )
