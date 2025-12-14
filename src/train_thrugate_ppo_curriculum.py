#!/usr/bin/env python3
"""
Train PPO on the curriculum-based fly-through-gate environment.

The script is structured for reproducibility and reporting:
- automatic run folder with timestamp
- config snapshot to JSON
- CSV logs for training and evaluation
- best/last checkpoints
"""

from __future__ import annotations

import json
import os
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Tuple

import numpy as np
import torch

from agents.ppo_agent import PPO
from gym_pybullet_drones.envs.FlyThruGateCurriculumAvitary import FlyThruGateCurriculumAvitary
from gym_pybullet_drones.utils.enums import ActionType, ObservationType


@dataclass
class PPOConfig:
    max_training_timesteps: int = int(3e6)
    action_std: float = 0.6
    action_std_decay_rate: float = 0.05
    min_action_std: float = 0.1
    action_std_decay_freq: int = int(2.5e5)
    K_epochs: int = 80
    eps_clip: float = 0.2
    gamma: float = 0.99
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3


@dataclass
class TrainConfig:
    seed: int = 42
    gui: bool = False
    record_video: bool = False
    eval_every_episodes: int = 25
    eval_episodes: int = 5
    log_every_episode: int = 1
    print_every_episodes: int = 5
    save_every_timesteps: int = int(1e5)
    run_name: str = "ppo_thrugate_curriculum"
    log_root: str = "log_dir/ppo_training_thrugate"
    start_level: int = 0
    success_window: int = 30
    success_rate_threshold: float = 0.7


class SuccessCurriculum:
    """Tracks recent successes and bumps the curriculum level for the next episode."""

    def __init__(self, num_levels: int, window_size: int, threshold: float, start_level: int = 0):
        self.num_levels = num_levels
        self.window_size = window_size
        self.threshold = threshold
        self.level = max(0, min(start_level, num_levels - 1))
        self.success_history = deque(maxlen=window_size)

    @property
    def success_rate(self) -> float:
        return float(np.mean(self.success_history)) if self.success_history else 0.0

    def record(self, success: bool) -> int:
        self.success_history.append(1 if success else 0)
        if len(self.success_history) < self.window_size:
            return self.level
        if self.success_rate >= self.threshold and self.level < self.num_levels - 1:
            self.level += 1
            self.success_history.clear()
        return self.level


def make_run_dirs(cfg: TrainConfig) -> Tuple[str, str, str, str]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(cfg.log_root, cfg.run_name, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    train_csv = os.path.join(run_dir, "train_log.csv")
    eval_csv = os.path.join(run_dir, "eval_log.csv")
    return run_dir, ckpt_dir, train_csv, eval_csv


def save_config(run_dir: str, train_cfg: TrainConfig, ppo_cfg: PPOConfig, env: FlyThruGateCurriculumAvitary):
    cfg_path = os.path.join(run_dir, "config.json")
    payload = {
        "train": asdict(train_cfg),
        "ppo": asdict(ppo_cfg),
        "env": {
            "episode_len_sec": env.EPISODE_LEN_SEC,
            "ctrl_freq": env.CTRL_FREQ,
            "gate_pos": list(env.GATE_POS),
            "spawn_ranges": env.spawner.spawn_ranges,
            "num_levels": env.num_levels,
        },
    }
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def flatten_obs(obs: np.ndarray) -> np.ndarray:
    """Flatten observation to 1D for the PPO networks."""
    return np.asarray(obs, dtype=np.float32).reshape(-1)


def select_deterministic_action(agent: PPO, obs: np.ndarray) -> np.ndarray:
    """Use the actor mean for evaluation (no exploration noise)."""
    with torch.no_grad():
        state = torch.as_tensor(obs, dtype=torch.float32)
        action = agent.policy_old.actor(state)
    return action.cpu().numpy().flatten()


def evaluate_policy(env: FlyThruGateCurriculumAvitary, agent: PPO, episodes: int) -> Tuple[float, float]:
    rewards = []
    successes = []
    for _ in range(episodes):
        obs, info = env.reset()
        obs = flatten_obs(obs)
        ep_reward = 0.0
        for _ in range(env.EPISODE_LEN_SEC * env.CTRL_FREQ):
            action = select_deterministic_action(agent, obs)
            action = np.expand_dims(action, axis=0)
            obs, reward, terminated, truncated, info = env.step(action)
            obs = flatten_obs(obs)
            ep_reward += reward
            done = terminated or truncated
            if done:
                successes.append(1 if terminated else 0)
                break
        rewards.append(ep_reward)
    mean_reward = float(np.mean(rewards)) if rewards else 0.0
    mean_success = float(np.mean(successes)) if successes else 0.0
    return mean_reward, mean_success


def train(train_cfg: TrainConfig, ppo_cfg: PPOConfig):
    np.random.seed(train_cfg.seed)
    torch.manual_seed(train_cfg.seed)

    env = FlyThruGateCurriculumAvitary(
        obs=ObservationType('kin'),
        act=ActionType('rpm'),
        gui=train_cfg.gui,
        record=train_cfg.record_video,
        start_level=train_cfg.start_level,
    )
    eval_env = FlyThruGateCurriculumAvitary(
        obs=ObservationType('kin'),
        act=ActionType('rpm'),
        gui=False,
        record=False,
        start_level=train_cfg.start_level,
    )

    initial_obs, _ = env.reset(seed=train_cfg.seed, options={})
    initial_obs = flatten_obs(initial_obs)
    
    state_dim = initial_obs.shape[0]
    action_dim = env.action_space.shape[-1]

    agent = PPO(state_dim, action_dim, ppo_cfg.lr_actor, ppo_cfg.lr_critic, ppo_cfg.gamma,
                ppo_cfg.K_epochs, ppo_cfg.eps_clip, ppo_cfg.action_std)

    steps_per_episode = env.EPISODE_LEN_SEC * env.CTRL_FREQ
    update_timestep = steps_per_episode * 4  # update after 4 episodes worth of steps

    run_dir, ckpt_dir, train_csv, eval_csv = make_run_dirs(train_cfg)
    save_config(run_dir, train_cfg, ppo_cfg, env)

    with open(train_csv, "w", encoding="utf-8") as f:
        f.write("episode,timestep,ep_reward,ep_len,success,level,success_rate,range_min,range_max,action_std\n")
    with open(eval_csv, "w", encoding="utf-8") as f:
        f.write("episode,timestep,eval_reward,eval_success_rate,level_at_eval\n")

    print(f"Logging to: {run_dir}")
    print(f"Episode length (steps): {steps_per_episode}")

    time_step = 0
    episode_idx = 0
    best_eval_reward = -np.inf
    best_eval_path = os.path.join(ckpt_dir, "best_model.pth")
    last_ckpt_path = os.path.join(ckpt_dir, "last_model.pth")
    curriculum = SuccessCurriculum(
        num_levels=env.num_levels,
        window_size=train_cfg.success_window,
        threshold=train_cfg.success_rate_threshold,
        start_level=train_cfg.start_level,
    )

    while time_step <= ppo_cfg.max_training_timesteps:
        obs, info = env.reset(seed=train_cfg.seed + episode_idx, options={})
        obs = flatten_obs(obs)
        current_range = info.get("spawn_range", env.spawner.range_for_level(env.curriculum_level))
        ep_reward = 0.0
        ep_len = 0
        success_flag = 0

        for _ in range(steps_per_episode):
            action = agent.select_action(obs)
            action = np.expand_dims(action, axis=0)
            obs, reward, terminated, truncated, info = env.step(action)
            obs = flatten_obs(obs)
            done = terminated or truncated

            agent.buffer.rewards.append(reward)
            agent.buffer.is_terminals.append(done)

            time_step += 1
            ep_len += 1
            ep_reward += reward
            current_range = info.get("spawn_range", current_range)

            if time_step % update_timestep == 0:
                agent.update()

            if time_step % ppo_cfg.action_std_decay_freq == 0:
                agent.decay_action_std(ppo_cfg.action_std_decay_rate, ppo_cfg.min_action_std)

            if time_step % train_cfg.save_every_timesteps == 0:
                ckpt_path = os.path.join(ckpt_dir, f"step_{time_step}_ppo.pth")
                agent.save(ckpt_path)

            if done:
                success_flag = 1 if terminated else 0
                break

        env.curriculum_level = curriculum.record(success_flag == 1)
        level = env.curriculum_level
        success_rate = curriculum.success_rate
        current_range = env.spawner.range_for_level(level)

        with open(train_csv, "a", encoding="utf-8") as f:
            f.write(f"{episode_idx},{time_step},{ep_reward:.3f},{ep_len},{success_flag},{level},{success_rate:.3f},{current_range[0]:.3f},{current_range[1]:.3f},{agent.action_std:.3f}\n")

        if (episode_idx + 1) % train_cfg.print_every_episodes == 0:
            print(f"Ep {episode_idx+1:05d} | steps {time_step:07d} | reward {ep_reward:7.2f} | level {level} | success_rate {success_rate:.3f} | range {current_range}")

        if (episode_idx + 1) % train_cfg.eval_every_episodes == 0:
            eval_env.curriculum_level = env.curriculum_level
            eval_reward, eval_success = evaluate_policy(eval_env, agent, train_cfg.eval_episodes)
            with open(eval_csv, "a", encoding="utf-8") as f:
                f.write(f"{episode_idx},{time_step},{eval_reward:.3f},{eval_success:.3f},{eval_env.curriculum_level}\n")
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                agent.save(best_eval_path)
                print(f"[Eval] New best avg reward {best_eval_reward:.2f} at episode {episode_idx+1}, saved to {best_eval_path}")

        episode_idx += 1

    agent.save(last_ckpt_path)
    print(f"Training finished. Best model: {best_eval_path} | Last model: {last_ckpt_path}")


if __name__ == "__main__":
    train_cfg = TrainConfig()
    ppo_cfg = PPOConfig()
    train(train_cfg, ppo_cfg)
