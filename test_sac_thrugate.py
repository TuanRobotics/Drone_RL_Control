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
    DEFAULT_OBS = ObservationType('kin')
    DEFAULT_ACT = ActionType('rpm')

    total_test_episodes = args.episodes
    hidden_dim = 256
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
    print("-" * 88)

    # Testing loop
    test_rewards = []
    test_lengths = []
    success_count = 0

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
    print("=" * 88)


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test SAC agent for drone gate navigation')
    parser.add_argument('--model_path', type=str,
                       default='sac_drone_20241201_120000/sac_model_final.pt',
                       help='Path to the trained model checkpoint')
    parser.add_argument('--episodes', type=int, default=1,
                       help='Number of test episodes')
    parser.add_argument('--record', type=bool, default=False,
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