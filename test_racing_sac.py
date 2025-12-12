#!/usr/bin/env python3
"""Evaluate SAC racing policy and record videos."""

import argparse
import os
import time
from datetime import datetime

import numpy as np
from gym_pybullet_drones.envs.DroneRacingAviary import DroneRacingAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType
from gym_pybullet_drones.utils.utils import sync

from SAC.sac_agent import SACAgent


def test(model_path: str, episodes: int = 10, record: bool = True, gui: bool = True):
    print("=" * 88)
    print("SAC Drone Racing - Testing")
    print("=" * 88)

    output_dir = os.path.join("results_sac_racing", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(output_dir, exist_ok=True)

    env = DroneRacingAviary(
        drone_model=DroneModel.CF2X,
        physics=Physics.PYB,
        pyb_freq=240,
        ctrl_freq=48,
        gui=gui,
        record=record,
        obs=ObservationType.KIN,
        act=ActionType.RPM,
    )

    obs, _ = env.reset()
    state_dim = obs.reshape(-1).shape[0]
    action_dim = env.action_space.shape[1]
    print(f"State dim: {state_dim} | Action dim: {action_dim}")

    agent = SACAgent(state_dim, action_dim)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    agent.load(model_path)

    rewards, lengths, successes = [], [], 0

    for ep in range(episodes):
        obs, info = env.reset(seed=42 + ep, options={})
        ep_reward, ep_len = 0.0, 0
        start = time.time()
        max_steps = env.EPISODE_LEN_SEC * env.CTRL_FREQ + 200

        for step in range(max_steps):
            action = agent.select_action(obs, evaluate=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_len += 1

            if gui:
                env.render()
                sync(step, start, env.CTRL_TIMESTEP)

            if terminated or truncated:
                pass_flags = info.get("passing_flag", [])
                if len(pass_flags) > 0 and pass_flags[-1]:
                    successes += 1
                break

        rewards.append(ep_reward)
        lengths.append(ep_len)
        print(
            f"Episode {ep+1}/{episodes} | Reward {ep_reward:7.2f} | "
            f"Len {ep_len:4d} | Success {'✓' if (len(info.get('passing_flag', []))>0 and info['passing_flag'][-1]) else '✗'}"
        )

    env.close()

    print("\n" + "=" * 88)
    print("Test Summary")
    print("=" * 88)
    print(f"Episodes:       {episodes}")
    print(f"Average Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Average Length: {np.mean(lengths):.2f}")
    print(f"Successes:      {successes}/{episodes}")

    # Save stats
    results_file = os.path.join(output_dir, "test_results.txt")
    with open(results_file, "w") as f:
        f.write("SAC Drone Racing - Test Results\n")
        f.write("=" * 88 + "\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Episodes: {episodes}\n")
        f.write(f"Average Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}\n")
        f.write(f"Average Length: {np.mean(lengths):.2f}\n")
        f.write(f"Successes: {successes}/{episodes}\n\n")
        f.write("Episode Details:\n")
        f.write("-" * 88 + "\n")
        for i, (r, l) in enumerate(zip(rewards, lengths)):
            f.write(f"Episode {i+1}: Reward={r:.2f}, Length={l}\n")

    print(f"\nSaved test results to {results_file}")
    print("Videos/frames will be under the environment OUTPUT_FOLDER (PyBullet default).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test SAC racing agent and record videos")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/tuan/Desktop/drone_rl_control/log_dir/results_racing_sac_single/sac_semicircle_final.pt",
        help="Path to trained SAC checkpoint",
    )
    parser.add_argument("--episodes", type=int, default=10, help="Number of test rollouts")
    parser.add_argument("--record", action="store_true", default=True, help="Enable video recording")
    parser.add_argument("--no-record", dest="record", action="store_false", help="Disable video recording")
    parser.add_argument("--gui", action="store_true", default=True, help="Enable GUI")
    parser.add_argument("--no-gui", dest="gui", action="store_false", help="Disable GUI")

    args = parser.parse_args()

    print(f"\nTest config:\n  model: {args.model_path}\n  episodes: {args.episodes}\n  record: {args.record}\n  gui: {args.gui}\n")
    test(args.model_path, args.episodes, args.record, args.gui)
