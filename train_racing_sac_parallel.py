#!/usr/bin/env python3
"""SAC training loop for drone racing with distributed initialization and parallel sampling.

The implementation follows the ideas from the paper snippets:
- Distributed initialization: start each rollout from hover poses sampled around
  gate centers and segment midpoints to expose the policy to all gates early.
  After the agent learns to fly, the initializer also reuses states from past
  trajectories to keep curriculum pressure near difficult regions.
- Parallel sampling: run multiple PyBullet environments in parallel Python
  processes to speed up data collection and diversify rollouts.

Note: this script keeps the environment GUI disabled for parallel safety.
"""

import os
import multiprocessing as mp
import random
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Deque, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from SAC.sac_agent import ReplayBuffer, SACAgent
from gym_pybullet_drones.envs.DroneRacingAvitary import DroneRacingAviary
from gym_pybullet_drones.utils.enums import (
    ActionType,
    DroneModel,
    ObservationType,
    Physics,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Distributed initialization strategy
# ---------------------------------------------------------------------------
@dataclass
class InitConfig:
    hover_sigma: float = 0.2          # Position noise (m) around anchor points
    min_z: float = 0.3                # Keep above ground
    max_z: float = 2.0
    yaw_range: Tuple[float, float] = (-np.pi, np.pi)
    use_traj_after: int = 2000        # Start sampling from past trajectories after N stored states
    traj_prob: float = 0.5            # Probability of using a trajectory state once unlocked
    traj_buffer: int = 50000          # Max stored positions from past rollouts


class DistributedInitializer:
    """Samples diverse start poses for racing rollouts."""

    def __init__(self, racing_setup: dict, cfg: InitConfig | None = None):
        self.cfg = cfg or InitConfig()
        # Build anchor points: gate centers + midpoints between gates
        gate_centers = [np.array(v[0], dtype=np.float32) for v in racing_setup.values()]
        segment_midpoints = [
            0.5 * (gate_centers[i] + gate_centers[i + 1])
            for i in range(len(gate_centers) - 1)
        ]
        self.anchor_points = gate_centers + segment_midpoints
        self.trajectory_positions: Deque[np.ndarray] = deque(
            maxlen=self.cfg.traj_buffer
        )
        ys = [p[1] for p in gate_centers]
        self.y_min = min(ys) - 0.5
        self.y_max = max(ys) + 0.5
        self.x_limit = 1.5  # match track limit ~1.7

    def record_state_vector(self, state_vec: np.ndarray) -> None:
        """Store position from a full state vector for future resets."""
        if state_vec is None or len(state_vec) < 3:
            return
        self.trajectory_positions.append(np.array(state_vec[:3], dtype=np.float32))

    def _sample_anchor(self) -> np.ndarray:
        base = random.choice(self.anchor_points)
        noise = np.random.normal(scale=self.cfg.hover_sigma, size=3)
        pos = base + noise
        pos[2] = np.clip(pos[2], self.cfg.min_z, self.cfg.max_z)
        pos[1] = np.clip(pos[1], self.y_min, self.y_max)
        pos[0] = np.clip(pos[0], -self.x_limit, self.x_limit)
        return pos.astype(np.float32)

    def _sample_from_trajectory(self) -> np.ndarray:
        pos = random.choice(self.trajectory_positions)
        noise = np.random.normal(scale=0.05, size=3)
        pos = pos + noise
        pos[2] = np.clip(pos[2], self.cfg.min_z, self.cfg.max_z)
        pos[1] = np.clip(pos[1], self.y_min, self.y_max)
        pos[0] = np.clip(pos[0], -self.x_limit, self.x_limit)
        return pos.astype(np.float32)

    # Function for 
    def sample_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (init_xyzs, init_rpys) for a single-drone env."""
        use_traj = (
            len(self.trajectory_positions) > self.cfg.use_traj_after
            and random.random() < self.cfg.traj_prob
        )
        pos = (
            self._sample_from_trajectory()
            if use_traj
            else self._sample_anchor()
        )
        yaw = random.uniform(*self.cfg.yaw_range)
        init_xyzs = pos.reshape(1, 3)
        init_rpys = np.array([[0.0, 0.0, yaw]], dtype=np.float32)
        return init_xyzs, init_rpys

# ---------------------------------------------------------------------------
# Environment worker (for parallel sampling)
# ---------------------------------------------------------------------------
def _extract_reward_done(reward):
    """Return (reward_value, extra_done_flag) from env reward output."""
    extra_done = False
    if isinstance(reward, (tuple, list)):
        reward_val = reward[0]
        if len(reward) > 1 and isinstance(reward[1], (bool, np.bool_)):
            extra_done = bool(reward[1])
    else:
        reward_val = reward
    reward_val = float(np.asarray(reward_val).reshape(-1)[0])
    return reward_val, extra_done


def _env_worker(
    env_fn: Callable[[], DroneRacingAviary],
    pipe: mp.connection.Connection,
    initializer: DistributedInitializer,
):
    """Run one environment in a subprocess."""
    env = env_fn()
    try:
        while True:
            cmd, data = pipe.recv()
            if cmd == "reset":
                init_xyzs, init_rpys = initializer.sample_pose()
                env.INIT_XYZS = init_xyzs
                env.INIT_RPYS = init_rpys
                obs, info = env.reset()
                state_vec = env._getDroneStateVector(0)
                initializer.record_state_vector(state_vec)
                pipe.send((obs.reshape(-1), info, state_vec))
            elif cmd == "step":
                action = data
                obs, reward, terminated, truncated, info = env.step(
                    action.reshape(1, -1)
                )
                reward_val, reward_done = _extract_reward_done(reward)
                done = bool(terminated or truncated or reward_done)
                state_vec = env._getDroneStateVector(0)
                initializer.record_state_vector(state_vec)
                pipe.send((obs.reshape(-1), reward_val, done, info, state_vec))
            elif cmd == "close":
                env.close()
                pipe.close()
                break
            else:
                raise RuntimeError(f"Unknown command to worker: {cmd}")
    except KeyboardInterrupt:
        env.close()
        pipe.close()


class ParallelRacingSampler:
    """Lightweight parallel sampler using multiprocessing Pipes."""

    def __init__(
        self,
        num_envs: int,
        env_fn: Callable[[], DroneRacingAviary],
        initializer: DistributedInitializer,
    ):
        self.num_envs = num_envs
        self.parent_conns: List[mp.connection.Connection] = []
        self.processes: List[mp.Process] = []
        for _ in range(num_envs):
            parent_conn, child_conn = mp.Pipe()
            proc = mp.Process(
                target=_env_worker,
                args=(env_fn, child_conn, initializer),
                daemon=True,
            )
            proc.start()
            child_conn.close()
            self.parent_conns.append(parent_conn)
            self.processes.append(proc)
        # Init action space using a temp env
        tmp_env = env_fn()
        self.action_space = tmp_env.action_space
        tmp_env.close()

    def reset(self) -> List[np.ndarray]:
        for conn in self.parent_conns:
            conn.send(("reset", None))
        results = [conn.recv() for conn in self.parent_conns]
        obs, _, _ = zip(*results)
        return list(obs)

    def reset_indices(self, indices: List[int]) -> List[np.ndarray]:
        for idx in indices:
            self.parent_conns[idx].send(("reset", None))
        results = [self.parent_conns[idx].recv() for idx in indices]
        obs, _, _ = zip(*results)
        return list(obs)

    def step(self, actions: List[np.ndarray]):
        assert len(actions) == self.num_envs
        for conn, act in zip(self.parent_conns, actions):
            conn.send(("step", act))
        return [conn.recv() for conn in self.parent_conns]

    def close(self):
        for conn in self.parent_conns:
            conn.send(("close", None))
        for proc in self.processes:
            proc.join(timeout=1.0)


# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------
def _smooth_curve(values: List[float], window: int = 50) -> np.ndarray:
    if len(values) < window or window <= 1:
        return np.array(values, dtype=np.float32)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def _plot_training_curves(
    rewards: List[float],
    lengths: List[int],
    q1_losses: List[float],
    q2_losses: List[float],
    policy_losses: List[float],
    save_path: str,
):
    if len(rewards) == 0:
        return
    plt.figure(figsize=(14, 10))
    plt.suptitle("SAC Drone Racing - Training Progress", fontsize=16, weight="bold")

    # Rewards
    plt.subplot(2, 2, 1)
    plt.plot(rewards, alpha=0.25, label="Episode Reward")
    smooth_r = _smooth_curve(rewards, window=max(20, len(rewards) // 20))
    plt.plot(range(len(rewards) - len(smooth_r), len(rewards)), smooth_r, label="Smoothed")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()

    # Episode length
    plt.subplot(2, 2, 2)
    plt.plot(lengths, color="#ff7f0e", alpha=0.5)
    plt.xlabel("Episodes")
    plt.ylabel("Length (steps)")
    plt.grid(True, linestyle="--", alpha=0.4)

    # Critic losses
    plt.subplot(2, 2, 3)
    if q1_losses:
        plt.plot(q1_losses, alpha=0.25, label="Q1 Loss")
        smooth_q1 = _smooth_curve(q1_losses, window=max(50, len(q1_losses) // 30))
        plt.plot(range(len(q1_losses) - len(smooth_q1), len(q1_losses)), smooth_q1, label="Q1 Smoothed")
    if q2_losses:
        plt.plot(q2_losses, alpha=0.25, label="Q2 Loss", color="#2ca02c")
        smooth_q2 = _smooth_curve(q2_losses, window=max(50, len(q2_losses) // 30))
        plt.plot(range(len(q2_losses) - len(smooth_q2), len(q2_losses)), smooth_q2, label="Q2 Smoothed", color="#2ca02c")
    plt.xlabel("Updates")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()

    # Policy loss
    plt.subplot(2, 2, 4)
    if policy_losses:
        plt.plot(policy_losses, alpha=0.3, label="Policy Loss", color="#d62728")
        smooth_p = _smooth_curve(policy_losses, window=max(50, len(policy_losses) // 30))
        plt.plot(range(len(policy_losses) - len(smooth_p), len(policy_losses)), smooth_p, label="Smoothed", color="#d62728")
    plt.xlabel("Updates")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=200)
    plt.close()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_sac_racing_parallel(
    num_envs: int = 8,
    total_steps: int = 300_000,
    start_steps: int = 10_000,
    batch_size: int = 256,
    update_after: int = 5_000,
    update_every: int = 50,
    updates_per_step: int = 1,
    log_interval: int = 1_000,
    plot_interval_episodes: int = 200,
    save_every_episodes: int = 500,
    save_path: str | None = None,
):
    """Train SAC on DroneRacingAviary with distributed init + parallel sampling."""

    def make_env():
        return DroneRacingAviary(
            drone_model=DroneModel.CF2X,
            physics=Physics.PYB,
            pyb_freq=240,
            ctrl_freq=48,
            gui=False,
            obs=ObservationType.KIN,
            act=ActionType.RPM,
        )

    # Dummy env to fetch track layout
    dummy_env = make_env()
    initializer = DistributedInitializer(dummy_env.racing_setup)
    sample_obs, _ = dummy_env.reset()
    state_dim = sample_obs.reshape(-1).shape[0]
    action_dim = dummy_env.action_space.shape[1]
    dummy_env.close()

    sampler = ParallelRacingSampler(num_envs, make_env, initializer)
    agent = SACAgent(state_dim, action_dim)
    replay_buffer = ReplayBuffer()

    # Logging setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if save_path:
        log_dir = os.path.dirname(save_path) if os.path.splitext(save_path)[1] else save_path
    else:
        log_dir = os.path.join("log_dir", f"sac_racing_parallel_{timestamp}")
        save_path = os.path.join(log_dir, "sac_model.pt")
    os.makedirs(log_dir, exist_ok=True)
    curves_path = os.path.join(log_dir, "training_curves.png")
    csv_path = os.path.join(log_dir, "episode_log.csv")
    csv_file = open(csv_path, "w")
    csv_file.write("episode,total_steps,episode_reward,episode_length\n")

    states = sampler.reset()
    episode_rewards = [0.0 for _ in range(num_envs)]
    episode_lengths = [0 for _ in range(num_envs)]
    total_env_steps = 0
    total_episodes = 0

    reward_history: List[float] = []
    length_history: List[int] = []
    q1_loss_history: List[float] = []
    q2_loss_history: List[float] = []
    policy_loss_history: List[float] = []

    print(f"Starting SAC racing training on {num_envs} envs, device={device}")
    start_time = time.time()

    while total_env_steps < total_steps:
        actions = []
        for idx, state in enumerate(states):
            if total_env_steps < start_steps:
                act = sampler.action_space.sample()[0]
            else:
                act = agent.select_action(state, evaluate=False)
            actions.append(act)

        # Step all envs in parallel
        results = sampler.step(actions)
        next_states = []
        for i, (obs, reward, done, info, _) in enumerate(results):
            replay_buffer.push(states[i], actions[i], reward, obs, done)
            episode_rewards[i] += reward
            episode_lengths[i] += 1
            next_states.append(obs if not done else None)
            if done:
                total_episodes += 1
                reward_history.append(episode_rewards[i])
                length_history.append(episode_lengths[i])
                csv_file.write(f"{total_episodes},{total_env_steps},{episode_rewards[i]},{episode_lengths[i]}\n")
                if total_episodes % 50 == 0:
                    last50 = reward_history[-50:]
                    print(
                        f"[Ep {total_episodes}] recent avg reward: {np.mean(last50):.2f} "
                        f"(len {np.mean(length_history[-50:]):.1f})"
                    )
                else:
                    print(
                        f"[Ep {total_episodes}] reward {episode_rewards[i]:.2f} "
                        f"len {episode_lengths[i]} steps={total_env_steps}"
                    )
                episode_rewards[i] = 0.0
                episode_lengths[i] = 0

        total_env_steps += num_envs

        # Reset finished envs with distributed init
        if any(s is None for s in next_states):
            reset_indices = [i for i, s in enumerate(next_states) if s is None]
            reset_obs = sampler.reset_indices(reset_indices)
            for idx, obs in zip(reset_indices, reset_obs):
                next_states[idx] = obs
                episode_rewards[idx] = 0.0
                episode_lengths[idx] = 0

        states = next_states

        # SAC updates
        if total_env_steps >= update_after and total_env_steps % update_every == 0:
            for _ in range(update_every * updates_per_step):
                if len(replay_buffer) < batch_size:
                    break
                q1_loss, q2_loss, policy_loss = agent.update(replay_buffer, batch_size)
                q1_loss_history.append(q1_loss)
                q2_loss_history.append(q2_loss)
                policy_loss_history.append(policy_loss)

        # Logging
        if total_env_steps % log_interval == 0:
            avg_reward = np.mean(episode_rewards)
            valid_lengths = [l for l in episode_lengths if l > 0]
            avg_length = np.mean(valid_lengths) if valid_lengths else 0.0
            elapsed = time.time() - start_time
            print(
                f"Steps {total_env_steps}/{total_steps} | "
                f"AvgEpReward {avg_reward:.2f} | "
                f"AvgEpLen {avg_length:.1f} | "
                f"Elapsed {elapsed/60:.1f} min"
            )

        # Plot curves periodically based on completed episodes
        if total_episodes > 0 and total_episodes % plot_interval_episodes == 0:
            _plot_training_curves(
                reward_history,
                length_history,
                q1_loss_history,
                q2_loss_history,
                policy_loss_history,
                curves_path,
            )

        # Save checkpoints periodically by episode count
        if total_episodes > 0 and total_episodes % save_every_episodes == 0:
            agent.save(os.path.join(log_dir, f"sac_model_ep{total_episodes}.pt"))
            print(f"Checkpoint saved at episode {total_episodes}")

    sampler.close()
    if save_path:
        agent.save(save_path)
        _plot_training_curves(
            reward_history,
            length_history,
            q1_loss_history,
            q2_loss_history,
            policy_loss_history,
            curves_path,
        )
        print(f"Saved final SAC model and curves to {log_dir}")

    csv_file.close()
    print("Training finished.")
    return agent


if __name__ == "__main__":
    train_sac_racing_parallel(
        num_envs=4,
        total_steps=500000,
        save_every_episodes=50000,
        plot_interval_episodes=200,
    )
