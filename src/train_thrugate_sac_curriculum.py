#!/usr/bin/env python3
"""Train SAC on the curriculum-based FlyThruGate environment."""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from agents.sac_agent import ReplayBuffer, SACAgent
from gym_pybullet_drones.envs.FlyThruGateCurriculumAvitary import FlyThruGateCurriculumAvitary
from gym_pybullet_drones.utils.enums import ActionType, ObservationType


@dataclass
class TrainConfig:
    num_episodes: int = 20000
    batch_size: int = 128
    buffer_size: int = 1_000_000
    start_steps: int = 5000
    update_after: int = 1000
    update_every: int = 50
    eval_every: int = 50
    num_eval_episodes: int = 5
    gui: bool = False
    start_level: int = 0
    success_window: int = 30
    success_rate_threshold: float = 0.7
    run_name: str = "sac_thrugate_curriculum"
    log_root: str = "/home/tuan/Desktop/drone_rl_control/log_dir/sac_training_thrugate_curriculum"


class SuccessCurriculum:
    """Tracks success rate over a window and bumps level for the next episode."""

    def __init__(self, num_levels: int, window_size: int, threshold: float, start_level: int = 0):
        self.num_levels = num_levels
        self.window_size = window_size
        self.threshold = threshold
        self.level = max(0, min(start_level, num_levels - 1))
        self.history = []

    @property
    def success_rate(self) -> float:
        return float(np.mean(self.history)) if self.history else 0.0

    def record(self, success: bool) -> int:
        self.history.append(1 if success else 0)
        if len(self.history) > self.window_size:
            self.history.pop(0)
        if len(self.history) == self.window_size and self.success_rate >= self.threshold:
            if self.level < self.num_levels - 1:
                self.level += 1
                self.history.clear()
        return self.level


def make_dirs(cfg: TrainConfig) -> Tuple[Path, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg.log_root) / cfg.run_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, ckpt_dir


def flatten_obs(obs) -> np.ndarray:
    return np.asarray(obs, dtype=np.float32).reshape(-1)


def reshape_action(action: np.ndarray, action_space) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32)
    return action.reshape(action_space.shape)


def evaluate_policy(env: FlyThruGateCurriculumAvitary, agent: SACAgent, episodes: int) -> float:
    rewards = []
    for _ in range(episodes):
        obs, _ = env.reset()
        state = flatten_obs(obs)
        ep_reward = 0.0
        for _ in range(env.EPISODE_LEN_SEC * env.CTRL_FREQ):
            action = agent.select_action(state, evaluate=True)
            action = reshape_action(action, env.action_space)
            obs, reward, terminated, truncated, _ = env.step(action)
            state = flatten_obs(obs)
            ep_reward += reward
            if terminated or truncated:
                break
        rewards.append(ep_reward)
    return float(np.mean(rewards)) if rewards else 0.0


def train(cfg: TrainConfig):
    # Seed
    np.random.seed(42)
    torch.manual_seed(42)

    env = FlyThruGateCurriculumAvitary(
        obs=ObservationType("kin"),
        act=ActionType("rpm"),
        gui=cfg.gui,
        record=False,
        start_level=cfg.start_level,
    )
    eval_env = FlyThruGateCurriculumAvitary(
        obs=ObservationType("kin"),
        act=ActionType("rpm"),
        gui=False,
        record=False,
        start_level=cfg.start_level,
    )

    init_obs, _ = env.reset()
    state_dim = flatten_obs(init_obs).shape[0]
    action_dim = env.action_space.shape[-1]

    agent = SACAgent(state_dim, action_dim)
    buffer = ReplayBuffer(cfg.buffer_size)

    run_dir, ckpt_dir = make_dirs(cfg)

    curriculum = SuccessCurriculum(
        num_levels=env.num_levels,
        window_size=cfg.success_window,
        threshold=cfg.success_rate_threshold,
        start_level=cfg.start_level,
    )

    steps_per_episode = env.EPISODE_LEN_SEC * env.CTRL_FREQ
    total_steps = 0
    episode_rewards = []
    eval_rewards = []

    print(f"Logging to: {run_dir}")
    for episode in range(cfg.num_episodes):
        obs, info = env.reset()
        state = flatten_obs(obs)
        current_range = info.get("spawn_range", env.spawner.range_for_level(env.curriculum_level))
        ep_reward = 0.0
        success_flag = 0

        for _ in range(steps_per_episode):
            if total_steps < cfg.start_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)
                action = reshape_action(action, env.action_space)

            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = flatten_obs(next_obs)
            done = terminated or truncated

            buffer.push(state, action, reward, next_state, done)
            state = next_state
            current_range = info.get("spawn_range", current_range)

            total_steps += 1
            ep_reward += reward

            if total_steps >= cfg.update_after and total_steps % cfg.update_every == 0:
                agent.update(buffer, cfg.batch_size)

            if done:
                success_flag = 1 if terminated else 0
                break

        episode_rewards.append(ep_reward)
        env.curriculum_level = curriculum.record(success_flag == 1)

        if (episode + 1) % cfg.eval_every == 0:
            eval_env.curriculum_level = env.curriculum_level
            eval_reward = evaluate_policy(eval_env, agent, cfg.num_eval_episodes)
            eval_rewards.append(eval_reward)
            print(
                f"Ep {episode+1:05d} | steps {total_steps:07d} | reward {ep_reward:7.2f} | "
                f"level {env.curriculum_level} | success_rate {curriculum.success_rate:.3f} | "
                f"range {current_range} | eval {eval_reward:.2f}"
            )

        # Save checkpoints occasionally
        if (episode + 1) % (cfg.eval_every * 4) == 0:
            ckpt_path = ckpt_dir / f"step_{total_steps}_sac.pth"
            agent.save(str(ckpt_path))

    agent.save(str(ckpt_dir / "final_sac.pth"))
    print("Training finished.")
    return {
        "episode_rewards": episode_rewards,
        "eval_rewards": eval_rewards,
        "config": asdict(cfg),
        "run_dir": str(run_dir),
    }


if __name__ == "__main__":
    train(TrainConfig())
