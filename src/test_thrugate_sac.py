#!/usr/bin/env python3

import os
import time
import numpy as np
import torch
from datetime import datetime
import argparse

from gym_pybullet_drones.envs.FlyThruGateAvitary import FlyThruGateAvitary
from agents.sac_agent import QNetwork,GaussianPolicy, ReplayBuffer, SACAgent

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

# Import SAC Agent từ file training
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

def test(args):
    print("=" * 60)
    print("SAC Drone Gate Navigation - Testing")
    print("=" * 60)

    ################## Configuration ##################
    DEFAULT_GUI = True
    DEFAULT_RECORD_VIDEO = True
    DEFAULT_OBS = ObservationType('kin')
    DEFAULT_ACT = ActionType('rpm')
    DEFAULT_OUTPUT_FOLDER = './results/results_thrugate_sac'
    if args.curriculum: 
        DEFAULT_OUTPUT_FOLDER = './results/results_thrugate_sac_curriculum'

    total_test_episodes = args.episodes
    hidden_dim = 256  # Hidden layer dimension for networks
    #####################################################

    # Initialize environment
    print("\nInitializing environment...")
    env = FlyThruGateAvitary(
        drone_model=DroneModel.CF2X,
        initial_xyzs=np.array([[0, 0, 0.5]]),
        physics=Physics.PYB,
        pyb_freq=240,
        ctrl_freq=30,
        gui=DEFAULT_GUI,
        record=DEFAULT_RECORD_VIDEO,
        obs=DEFAULT_OBS,
        act=DEFAULT_ACT,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        use_curriculum=args.curriculum,
        curriculum_level=5
    )

    # Get state and action dimensions
    obs, _ = env.reset()
    state_dim = int(np.prod(obs.shape))                # số chiều state
    action_dim = int(np.prod(env.action_space.shape))  # số chiều action

    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")

    # Initialize SAC agent
    print("\nInitializing SAC agent...")
    sac_agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim
    )

    # Load pretrained model
    checkpoint_path = args.model_path
    if not os.path.exists(checkpoint_path):
        print(f"Error: Model file not found at {checkpoint_path}")
        return

    print(f"Loading network from: {checkpoint_path}")
    sac_agent.load(checkpoint_path)
    print("=" * 60)

    # Testing loop
    test_rewards = []
    test_lengths = []
    success_count = 0
    success_times = []
    center_times = []

    print(f"\nStarting test for {total_test_episodes} episodes...")
    print("=" * 60)

    for episode in range(total_test_episodes):
        obs, info = env.reset(seed=42 + episode, options={})
        ep_reward = 0
        ep_length = 0
        start = time.time()
        center_time = None

        # Run episode
        max_steps = (env.EPISODE_LEN_SEC + 30) * env.CTRL_FREQ
        for i in range(max_steps):
            # Select action (deterministic for testing)
            action = sac_agent.select_action(obs, evaluate=True)
            action = _reshape_action_for_env(action, env.action_space)
            # Execute action
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_length += 1
            if center_time is None and getattr(env, "center_gate_passed", False):
                center_time = ep_length / env.CTRL_FREQ

            # Render if GUI is enabled
            if DEFAULT_GUI:
                env.render()
                sync(i, start, env.CTRL_TIMESTEP)

            # Check if episode is done
            if terminated or truncated:
                # Success is when the drone passes the gate center
                if center_time is not None:
                    success_count += 1
                    success_times.append(center_time)
                    print(f"Episode {episode + 1} passed gate center in {center_time:.2f} seconds.")
                break
        # Store episode statistics
        test_rewards.append(ep_reward)
        test_lengths.append(ep_length)
        center_times.append(center_time)

        # Print episode results
        center_str = f"{center_time:.2f}s" if center_time is not None else "-"
        print(f'Episode {episode + 1}/{total_test_episodes} | '
              f'Reward: {ep_reward:7.2f} | '
              f'Length: {ep_length:4d} | '
              f'Center: {center_str} | '
              f'Success: {"✓" if center_time is not None else "✗"}')

    env.close()

    # Print summary statistics
    print("\n" + "=" * 88)
    print("Test Summary")
    print("=" * 88)
    print(f"Total Episodes:        {total_test_episodes}")
    print(f"Average Reward:        {np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}")
    print(f"Max Reward:            {np.max(test_rewards):.2f}")
    print(f"Min Reward:            {np.min(test_rewards):.2f}")
    print(f"Average Length:        {np.mean(test_lengths):.2f}")
    print(f"Success Rate:          {success_count}/{total_test_episodes} "
          f"({100 * success_count / total_test_episodes:.1f}%)")
    if success_times:
        print(f"Avg center-pass time:  {np.mean(success_times):.2f}s ± {np.std(success_times):.2f}s")
        print(f"Best/Worst center:     {np.min(success_times):.2f}s / {np.max(success_times):.2f}s")
    print("=" * 88)

    # Save summary
    out_dir = os.path.join(DEFAULT_OUTPUT_FOLDER, "test_summary")
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "test_results_sac.txt")
    with open(summary_path, "w") as f:
        f.write("SAC FlyThruGateAvitary Test Results\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Episodes: {total_test_episodes}\n")
        f.write(f"Avg reward: {np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}\n")
        f.write(f"Success rate: {success_count}/{total_test_episodes} ({100*success_count/total_test_episodes:.1f}%)\n")
        if success_times:
            f.write(f"Avg center-pass time: {np.mean(success_times):.2f}s ± {np.std(success_times):.2f}s\n")
            f.write(f"Best center: {np.min(success_times):.2f}s | Worst center: {np.max(success_times):.2f}s\n")
        f.write("\nEpisode details:\n")
        for idx, (r, l) in enumerate(zip(test_rewards, test_lengths)):
            f.write(f"Ep {idx+1}: reward={r:.2f}, len={l}")
            if idx < len(center_times) and center_times[idx] is not None:
                f.write(f", center_time={center_times[idx]:.2f}s")
            f.write("\n")
        if success_times:
            f.write("\nCenter pass times (s): " + ", ".join(f"{t:.2f}" for t in success_times) + "\n")



# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test SAC agent for drone gate navigation')
    parser.add_argument('--model_path', type=str,
                        # /home/tuan/Desktop/drone_rl_control/log_dir/sac_training_thrugate_curriculum/sac_20251217_200922/sac_model_ep5000.pt
                       default='/home/tuan/Desktop/drone_rl_control/log_dir/sac_training_thrugate_curriculum/sac_20251217_200922/sac_model_ep5000.pt',
                       help='Path to the trained model checkpoint')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of test episodes')
    parser.add_argument('--record', type=bool, default=True,
                       help='Record video of the test')
    parser.add_argument('--gui', type=bool, default=True,
                       help='Enable GUI visualization') 
    parser.add_argument('--curriculum', type=bool, default=True,
                       help='Use curriculum learning during testing')

    args = parser.parse_args()

    print(f"\nTest Configuration:")
    print(f"  Model Path: {args.model_path}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Record Video: {args.record}")
    print(f"  GUI: {args.gui}")
    print()

    test(args)
