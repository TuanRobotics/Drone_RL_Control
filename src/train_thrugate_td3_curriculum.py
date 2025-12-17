import os
from collections import deque
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from agents.td3_agent import TD3Agent, Actor, Critic
from gym_pybullet_drones.envs.FlyThruGateAvitary import FlyThruGateAvitary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType


def _reshape_action_for_env(action, action_space):
    """Ensure action matches env action_space shape."""
    action = np.asarray(action, dtype=np.float32)
    try:
        action = action.reshape(action_space.shape)
    except Exception:
        action = np.expand_dims(action, axis=0)
    return action


def _update_curriculum_on_rate(env, success_history, success_rate_threshold,
                               success_window):
    """Increase curriculum when success rate over last window exceeds threshold."""
    if env.curriculum_level >= env.max_curriculum_level:
        return

    if len(success_history) < success_window:
        return

    success_rate = sum(success_history) / float(len(success_history))
    if success_rate > success_rate_threshold:
        env.curriculum_level += 1
        print(f"[Curriculum] Level advanced to {env.curriculum_level} "
              f"(success_rate={success_rate:.2f} over last {len(success_history)} episodes)")
        success_history.clear()


def train_td3_curriculum(num_episodes=20000,
                         learn_every=1,
                         warmup_steps=10000,
                         eval_every=100,
                         num_eval_episodes=5,
                         curriculum_start_level=0,
                         max_curriculum_level=5,
                         success_window=30,
                         success_rate_threshold=0.8,
                         gui=False):
    """Train TD3 agent with curriculum-enabled FlyThruGateAvitary."""
    env = FlyThruGateAvitary(obs=ObservationType.KIN,
                             act=ActionType.RPM,
                             gui=gui,
                             use_curriculum=True,
                             curriculum_level=curriculum_start_level,
                             max_curriculum_level=max_curriculum_level)

    obs, _ = env.reset()
    state_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    clip_low = float(np.min(env.action_space.low))
    clip_high = float(np.max(env.action_space.high))

    agent = TD3Agent(Actor,
                     Critic,
                     hidden_dim=256,
                     clip_low=clip_low,
                     clip_high=clip_high,
                     state_size=state_dim,
                     action_size=action_dim)
    agent.train_mode()

    episode_rewards = []
    eval_rewards = []
    critic_losses = []
    actor_losses = []
    curriculum_levels = []

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = "/home/tuan/Desktop/drone_rl_control/log_dir"
    save_dir = Path(base_dir) / "td3_training_thrugate_curriculum" / f"td3_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_prefix = Path(base_dir) / f"td3_curriculum_metrics_{timestamp}"
    curriculum_csv_path = f"{csv_prefix}_curriculum_levels.csv"

    print("\n" + "=" * 60)
    print("Starting TD3 Training with Curriculum for Drone Gate Navigation")
    print("=" * 60 + "\n")

    total_steps = 0
    success_history = deque(maxlen=success_window)

    for episode in range(num_episodes):
        env.success_passed = False
        state, _ = env.reset()
        episode_reward = 0

        for step in range(env.EPISODE_LEN_SEC * env.CTRL_FREQ):
            if total_steps < warmup_steps:
                action = env.action_space.sample()
            else:
                action = agent.get_action(state, explore=True)

            action = _reshape_action_for_env(action, env.action_space)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.memory.add(state, action, reward, next_state, done)
            state = next_state

            total_steps += 1
            episode_reward += reward

            if total_steps >= warmup_steps and total_steps % learn_every == 0:
                critic_loss, actor_loss = agent.learn_one_step()
                if critic_loss is not None:
                    critic_losses.append(critic_loss)
                if actor_loss is not None:
                    actor_losses.append(actor_loss)

            if done:
                break

        episode_rewards.append(episode_reward)
        avg_reward_100 = np.mean(episode_rewards[-100:])

        success = getattr(env, "center_gate_passed", False)

        success_history.append(1 if success else 0)
        _update_curriculum_on_rate(env, success_history,
                                   success_rate_threshold, success_window)
        curriculum_levels.append(env.curriculum_level)

        if episode % 1000 == 0 and episode > 0:
            agent.save(os.path.join(save_dir, f"td3_model_ep{episode}.pt"))
            plot_training_curves(episode_rewards, eval_rewards, critic_losses,
                                 actor_losses, save_dir, episode)
            plot_curriculum_levels(curriculum_levels, save_dir, f"ep{episode}")

        if (episode + 1) % 10 == 0:
            avg_reward_10 = np.mean(episode_rewards[-10:])
            recent_rate = float(np.mean(success_history)) if success_history else 0.0
            print(f"Episode {episode+1}/{num_episodes} | "
                  f"Steps: {total_steps} | "
                  f"Reward: {episode_reward:.2f} | "
                  f"Avg(10): {avg_reward_10:.2f} | "
                  f"Curriculum: L{env.curriculum_level}/{env.max_curriculum_level} | "
                  f"Success rate (last {len(success_history)}): {recent_rate:.2f}")

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
    agent.save(os.path.join(save_dir, "td3_model_final.pt"))
    plot_training_curves(episode_rewards, eval_rewards, critic_losses,
                         actor_losses, save_dir, "final")
    save_metrics_to_csv(episode_rewards, eval_rewards, critic_losses,
                        actor_losses, curriculum_levels, csv_prefix,
                        curriculum_csv_path)
    plot_curriculum_levels(curriculum_levels, save_dir, "final")

    print(f"\nTraining completed! Final model saved in {save_dir}")


def evaluate_policy(env, agent, num_episodes=5):
    """Evaluate TD3 policy without exploration noise."""
    eval_rewards = []

    for _ in range(num_episodes):
        env.success_passed = False
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.get_action(state, explore=False)
            action = _reshape_action_for_env(action, env.action_space)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            state = next_state
            done = terminated or truncated

        eval_rewards.append(episode_reward)

    return float(np.mean(eval_rewards))


def save_metrics_to_csv(episode_rewards, eval_rewards, critic_losses,
                        actor_losses, curriculum_levels, csv_prefix,
                        curriculum_csv_path):
    """Persist rewards, losses, and curriculum levels to CSV files inside src/."""
    training_path = f"{csv_prefix}_training_rewards.csv"
    eval_path = f"{csv_prefix}_evaluation_rewards.csv"
    loss_path = f"{csv_prefix}_losses.csv"

    with open(training_path, "w", newline="") as f:
        for idx, reward in enumerate(episode_rewards, start=1):
            f.write(f"{idx},{reward}\n")

    with open(eval_path, "w", newline="") as f:
        for idx, reward in enumerate(eval_rewards, start=1):
            f.write(f"{idx},{reward}\n")

    with open(loss_path, "w", newline="") as f:
        max_len = max(len(critic_losses), len(actor_losses))
        for idx in range(max_len):
            critic = critic_losses[idx] if idx < len(critic_losses) else ""
            actor = actor_losses[idx] if idx < len(actor_losses) else ""
            f.write(f"{idx + 1},{critic},{actor}\n")

    with open(curriculum_csv_path, "w", newline="") as f:
        for idx, level in enumerate(curriculum_levels, start=1):
            f.write(f"{idx},{level}\n")


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


if __name__ == '__main__':
    train_td3_curriculum(num_episodes=10000,
                         warmup_steps=1000,
                         eval_every=50,
                         num_eval_episodes=5,
                         curriculum_start_level=0,
                         max_curriculum_level=5,
                         success_window=30,
                         success_rate_threshold=0.85,
                         gui=False)
