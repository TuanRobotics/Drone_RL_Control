import os
import csv
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from agents.td3_agent import TD3Agent, Actor, Critic
from gym_pybullet_drones.envs.FlyThruGateAvitary import FlyThruGateAvitary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType


def save_metrics_to_csv(episode_rewards, eval_rewards, critic_losses,
                        actor_losses, csv_prefix):
    """Persist rewards and losses to CSV files inside src/."""
    training_path = f"{csv_prefix}_training_rewards.csv"
    eval_path = f"{csv_prefix}_evaluation_rewards.csv"
    loss_path = f"{csv_prefix}_losses.csv"

    with open(training_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward"])
        for idx, reward in enumerate(episode_rewards, start=1):
            writer.writerow([idx, reward])

    with open(eval_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["evaluation_step", "average_reward"])
        for idx, reward in enumerate(eval_rewards, start=1):
            writer.writerow([idx, reward])

    with open(loss_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["update_step", "critic_loss", "actor_loss"])
        max_len = max(len(critic_losses), len(actor_losses))
        for idx in range(max_len):
            critic = critic_losses[idx] if idx < len(critic_losses) else ""
            actor = actor_losses[idx] if idx < len(actor_losses) else ""
            writer.writerow([idx + 1, critic, actor])


def _reshape_action_for_env(action, action_space):
    """Ensure action matches env action_space shape."""
    action = np.asarray(action, dtype=np.float32)
    try:
        action = action.reshape(action_space.shape)
    except Exception:
        action = np.expand_dims(action, axis=0)
    return action

def train_td3(num_episodes=20000,
              max_steps=500,
              learn_every=1,
              warmup_steps=50,
              eval_every=50,
              num_eval_episodes=5,
              gui=False):
    """Train TD3 agent with SAC-like logging and CSV outputs."""
    env = FlyThruGateAvitary(obs=ObservationType.KIN,
                             act=ActionType.RPM,
                             gui=gui)

    obs, _ = env.reset()
    state_dim = obs.shape[1]
    action_dim = env.action_space.shape[1]
    clip_low = float(np.min(env.action_space.low))
    clip_high = float(np.max(env.action_space.high))

    agent = TD3Agent(Actor,
                     Critic,
                     clip_low=clip_low,
                     clip_high=clip_high,
                     state_size=state_dim,
                     action_size=action_dim)
    agent.train_mode()

    # Metrics
    episode_rewards = []
    eval_rewards = []
    critic_losses = []
    actor_losses = []

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = "/home/tuan/Desktop/drone_rl_control/log_dir/td3_training_thrugate"
    save_dir = Path(base_dir) / "td3_training" / f"td3_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_prefix = Path(base_dir) / f"td3_metrics_{timestamp}"

    print("\n" + "=" * 60)
    print("Starting TD3 Training for Drone Gate Navigation")
    print("=" * 60 + "\n")

    total_steps = 0

    for episode in range(num_episodes):
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

            # Learn every step after warmup
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

        if (episode + 1) % 10 == 0:
            avg_reward_10 = np.mean(episode_rewards[-10:])
            print(f"Episode {episode+1}/{num_episodes} | "
                  f"Steps: {total_steps} | "
                  f"Reward: {episode_reward:.2f} | "
                  f"Avg(10): {avg_reward_10:.2f}")

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

        # Early stopping
        # if avg_reward_100 >= 120:
        #     print(f"\nEnvironment solved in {episode+1} episodes!")
        #     print(f"Average Reward: {avg_reward_100:.2f}")
        #     print(f"{'='*60}\n")
        #     break

    env.close()

    # Final save + plot + CSV
    print(f"{'='*60}\n")
    agent.save(os.path.join(save_dir, "td3_model_final.pt"))
    plot_training_curves(episode_rewards, eval_rewards, critic_losses,
                         actor_losses, save_dir, "final")
    save_metrics_to_csv(episode_rewards, eval_rewards, critic_losses,
                        actor_losses, csv_prefix)

    print(f"\nTraining completed! Final model saved in {save_dir}")


def evaluate_policy(env, agent, num_episodes=5):
    """Evaluate TD3 policy without exploration noise."""
    eval_rewards = []

    for _ in range(num_episodes):
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


if __name__ == '__main__':
    train_td3(num_episodes=20000)

