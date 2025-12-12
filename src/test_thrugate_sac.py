#!/usr/bin/env python3

import os
import time
import numpy as np
import torch
from datetime import datetime
import argparse

from gym_pybullet_drones.envs.FlyThruGateAvitary import FlyThruGateAvitary
from SAC.sac_agent import QNetwork,GaussianPolicy, ReplayBuffer, SACAgent

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

# Import SAC Agent từ file training
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test(args):
    print("=" * 88)
    print("SAC Drone Gate Navigation - Testing")
    print("=" * 88)

    ################## Configuration ##################
    DEFAULT_GUI = True
    DEFAULT_RECORD_VIDEO = args.record
    DEFAULT_OUTPUT_FOLDER = 'log_dir/results_sac_thrugate/'
    if os.path.exists(DEFAULT_OUTPUT_FOLDER) is False:
        os.makedirs(DEFAULT_OUTPUT_FOLDER)

    DEFAULT_OBS = ObservationType('kin')
    DEFAULT_ACT = ActionType('rpm')

    total_test_episodes = args.episodes
    hidden_dim = 256
    #####################################################

    # Create output folder
    filename = os.path.join(DEFAULT_OUTPUT_FOLDER,
                           'recording_' + datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    if not os.path.exists(filename):
        os.makedirs(filename + '/')

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
        act=DEFAULT_ACT
    )

    # Get state and action dimensions
    obs, _ = env.reset()
    state_dim = obs.shape[1]                # số chiều state
    action_dim = env.action_space.shape[1]  # số chiều action

    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")

    # Initialize SAC agent
    print("\nInitializing SAC agent...")
    sac_agent = SACAgent(state_dim, action_dim, hidden_dim=256,
                    lr=3e-4, gamma=0.99, tau=0.005)

    # Load pretrained model
    checkpoint_path = args.model_path
    if not os.path.exists(checkpoint_path):
        print(f"Error: Model file not found at {checkpoint_path}")
        return

    print(f"Loading network from: {checkpoint_path}")
    sac_agent.load(checkpoint_path)
    print("-" * 88)

    # Testing loop
    test_rewards = []
    test_lengths = []
    success_count = 0
    success_times = []

    print(f"\nStarting test for {total_test_episodes} episodes...")
    print("-" * 88)

    for episode in range(total_test_episodes):
        obs, info = env.reset(seed=42 + episode, options={})
        ep_reward = 0
        ep_length = 0
        start = time.time()

        # Run episode
        max_steps = (env.EPISODE_LEN_SEC + 30) * env.CTRL_FREQ
        for i in range(max_steps):
            # Select action (deterministic for testing)
            action = sac_agent.select_action(obs, evaluate=True)

            # Execute action
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_length += 1

            # Render if GUI is enabled
            if DEFAULT_GUI:
                env.render()
                sync(i, start, env.CTRL_TIMESTEP)

            # Check if episode is done
            if terminated or truncated:
                # Check if drone passed the gate (reward > 5 means successful)
                if ep_reward > 5:
                    success_count += 1
                    success_times.append(ep_length / env.CTRL_FREQ)
                break

        # Store episode statistics
        test_rewards.append(ep_reward)
        test_lengths.append(ep_length)

        # Print episode results
        print(f'Episode {episode + 1}/{total_test_episodes} | '
              f'Reward: {ep_reward:7.2f} | '
              f'Length: {ep_length:4d} | '
              f'Success: {"✓" if ep_reward > 5 else "✗"}')

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
        print(f"Average Success Time:  {np.mean(success_times):.2f}s ± {np.std(success_times):.2f}s")
        print(f"Best Success Time:     {np.min(success_times):.2f}s")
        print(f"Worst Success Time:    {np.max(success_times):.2f}s")
    print("=" * 88)

    # Save results to file
    results_file = os.path.join(filename, 'test_results.txt')
    with open(results_file, 'w') as f:
        f.write("SAC Drone Gate Navigation - Test Results\n")
        f.write("=" * 88 + "\n\n")
        f.write(f"Model: {checkpoint_path}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Total Episodes: {total_test_episodes}\n")
        f.write(f"Average Reward: {np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}\n")
        f.write(f"Max Reward: {np.max(test_rewards):.2f}\n")
        f.write(f"Min Reward: {np.min(test_rewards):.2f}\n")
        f.write(f"Average Length: {np.mean(test_lengths):.2f}\n")
        f.write(f"Success Rate: {success_count}/{total_test_episodes} "
                f"({100 * success_count / total_test_episodes:.1f}%)\n\n")
        f.write("Episode Details:\n")
        f.write("-" * 88 + "\n")
        for i, (reward, length) in enumerate(zip(test_rewards, test_lengths)):
            f.write(f"Episode {i+1}: Reward = {reward:.2f}, Length = {length}")
            if reward > 5:
                f.write(f", Success time = {length/env.CTRL_FREQ:.2f}s")
            f.write("\n")
        f.write("\nSuccess times (s): " + ", ".join([f"{t:.2f}" for t in success_times]) + "\n")

    print(f"\nResults saved to: {results_file}")


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test SAC agent for drone gate navigation')
    parser.add_argument('--model_path', type=str,
                       default='log_dir/sac_thrugate_20251208_122448/sac_model_final.pt',
                       help='Path to the trained model checkpoint')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of test episodes')
    parser.add_argument('--record', type=bool, default=True,
                       help='Record video of the test')
    parser.add_argument('--gui', type=bool, default=True,
                       help='Enable GUI visualization')

    args = parser.parse_args()

    print(f"\nTest Configuration:")
    print(f"  Model Path: {args.model_path}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Record Video: {args.record}")
    print(f"  GUI: {args.gui}")
    print()

    test(args)
