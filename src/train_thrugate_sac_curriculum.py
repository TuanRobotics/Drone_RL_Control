#!/usr/bin/env python3
import os
from collections import deque
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

from agents.sac_agent import ReplayBuffer, SACAgent
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
        return # Already at max level

    if len(success_history) < success_window:
        return

    success_rate = sum(success_history) / float(len(success_history))
    if success_rate > success_rate_threshold:
        env.curriculum_level += 1
        print(f"[Curriculum] Level advanced to {env.curriculum_level} "
              f"(success_rate={success_rate:.2f} over last {len(success_history)} episodes)")
        success_history.clear()


# ============================================================================
# Training Function
# ============================================================================
def train_sac_curriculum(num_episodes=5000,
                         batch_size=256,
                         buffer_size=1_000_000,
                         start_steps=10000,
                         update_after=1000,
                         update_every=50,
                         save_every=100,
                         eval_every=50,
                         num_eval_episodes=5,
                         curriculum_start_level=0,
                         max_curriculum_level=5,
                         success_window=30,
                         success_rate_threshold=0.5,
                         gui=False):
    """Train SAC agent with curriculum-enabled FlyThruGateAvitary."""
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

    agent = SACAgent(state_dim, action_dim=action_dim, hidden_dim=256)
    replay_buffer = ReplayBuffer(buffer_size)

    # Metrics
    episode_rewards = []
    episode_lengths = []
    eval_rewards = []
    q1_losses = []
    q2_losses = []
    policy_losses = []
    curriculum_levels = []

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"/home/tuan/Desktop/drone_rl_control/log_dir/sac_training_thrugate_curriculum/sac_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    curriculum_csv_path = os.path.join(save_dir, "curriculum_levels.csv")

    total_steps = 0
    success_history = deque(maxlen=success_window)

    print("\n" + "=" * 60)
    print("Starting SAC Training with Curriculum for Drone Gate Navigation")
    print("=" * 60 + "\n")

    for episode in range(num_episodes):
        env.success_passed = False
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0

        for step in range(env.EPISODE_LEN_SEC * env.CTRL_FREQ):
            if total_steps < start_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)

            action_env = _reshape_action_for_env(action, env.action_space)

            next_state, reward, terminated, truncated, _ = env.step(action_env)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1
            total_steps += 1

            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state

            if total_steps >= update_after and total_steps % update_every == 0:
                for _ in range(update_every):
                    q1_loss, q2_loss, policy_loss = agent.update(
                        replay_buffer, batch_size)
                    q1_losses.append(q1_loss)
                    q2_losses.append(q2_loss)
                    policy_losses.append(policy_loss)

            if done:
                break

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        success = getattr(env, "center_gate_passed", False)
        success_history.append(1 if success else 0)
        _update_curriculum_on_rate(env, success_history,
                                   success_rate_threshold, success_window)
        curriculum_levels.append(env.curriculum_level)

        if episode % 1000 == 0 and episode > 0:
            agent.save(os.path.join(save_dir, f"sac_model_ep{episode}.pt"))
            plot_training_curves(episode_rewards, eval_rewards, q1_losses,
                                 policy_losses, save_dir, episode)
            plot_curriculum_levels(curriculum_levels, save_dir, f"ep{episode}")

        if (episode + 1) % 10 == 0:
            avg_reward_10 = np.mean(episode_rewards[-10:])
            recent_rate = float(np.mean(success_history)) if success_history else 0.0
            print(f"Episode {episode+1}/{num_episodes} | "
                  f"Steps: {total_steps} | "
                  f"Reward: {episode_reward:.2f} | "
                  f"Avg(10): {avg_reward_10:.2f} | "
                  f"Len: {episode_length} | "
                  f"Curriculum: L{env.curriculum_level}/{env.max_curriculum_level} | "
                  f"Success rate (last {len(success_history)}): {recent_rate:.2f}")

        avg_reward_100 = np.mean(episode_rewards[-100:])
        if (episode + 1) % eval_every == 0:
            eval_reward = evaluate_policy(env, agent, num_eval_episodes)
            eval_rewards.append(eval_reward)
            print(f"\n{'='*60}")
            print(
                f"Avg Reward over 100 episodes: {avg_reward_100:.2f}\n"
                f"Evaluation at Episode {episode+1}: Avg Evaluated Reward = {eval_reward:.2f}\n"
                f"Curriculum level: {env.curriculum_level}"
            )
            print(f"{'='*60}\n")

    env.close()

    print(f"{'='*60}\n")
    agent.save(os.path.join(save_dir, "sac_model_final.pt"))
    plot_training_curves(episode_rewards, eval_rewards, q1_losses,
                         policy_losses, save_dir, "final")
    plot_curriculum_levels(curriculum_levels, save_dir, "final")
    save_curriculum_levels(curriculum_levels, curriculum_csv_path)

    print(f"\nTraining completed! Final model saved in {save_dir}")

    return agent, episode_rewards, eval_rewards


def evaluate_policy(env, agent, num_episodes=5):
    """Evaluate the policy without exploration noise."""
    eval_rewards = []

    for _ in range(num_episodes):
        env.success_passed = False
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.select_action(state, evaluate=True)
            action_env = _reshape_action_for_env(action, env.action_space)
            next_state, reward, terminated, truncated, _ = env.step(action_env)
            episode_reward += reward
            state = next_state
            done = terminated or truncated

        eval_rewards.append(episode_reward)

    return float(np.mean(eval_rewards))


def plot_training_curves(episode_rewards, eval_rewards, q1_losses,
                         policy_losses, save_dir, episode):
    """Plot and save training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

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

    if eval_rewards:
        axes[0, 1].plot(eval_rewards, marker='o')
        axes[0, 1].set_xlabel('Evaluation Step')
        axes[0, 1].set_ylabel('Average Reward')
        axes[0, 1].set_title('Evaluation Rewards')
        axes[0, 1].grid(True)

    if q1_losses:
        axes[1, 0].plot(q1_losses, alpha=0.3)
        if len(q1_losses) >= 100:
            smoothed = np.convolve(q1_losses, np.ones(100)/100, mode='valid')
            axes[1, 0].plot(range(99, len(q1_losses)), smoothed, linewidth=2)
        axes[1, 0].set_xlabel('Update Step')
        axes[1, 0].set_ylabel('Q1 Loss')
        axes[1, 0].set_title('Critic Loss')
        axes[1, 0].grid(True)

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


def plot_curriculum_levels(curriculum_levels, save_dir, tag):
    """Plot curriculum level progression."""
    if not curriculum_levels:
        return
    plt.figure(figsize=(8, 4))
    plt.plot(curriculum_levels, marker='o', alpha=0.8)
    plt.xlabel('Episode')
    plt.ylabel('Curriculum Level')
    plt.title('Curriculum progression')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'curriculum_levels_{tag}.png'))
    plt.close()


def save_curriculum_levels(curriculum_levels, csv_path):
    """Save curriculum level per episode to CSV."""
    with open(csv_path, "w", newline="") as f:
        for idx, level in enumerate(curriculum_levels, start=1):
            f.write(f"{idx},{level}\n")


if __name__ == "__main__":
    agent, episode_rewards, eval_rewards = train_sac_curriculum(
        num_episodes=10000,
        batch_size=128,
        buffer_size=1_000_000,
        start_steps=10000,
        update_after=1000,
        update_every=50,
        save_every=100,
        eval_every=50,
        num_eval_episodes=5,
        curriculum_start_level=0,
        max_curriculum_level=5,
        success_window=30,
        success_rate_threshold=0.7,
        gui=False
    )

    print("\nTraining Summary:")
    print(f"Final average reward (last 100 episodes): {np.mean(episode_rewards[-100:]):.2f}")
    if eval_rewards:
        print(f"Best evaluation reward: {max(eval_rewards):.2f}")
