#!/usr/bin/env python3

import os
import time
from datetime import datetime
import argparse
import gymnasium as gym
import numpy as np
import torch
from agents.ppo_agent import PPO

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from gym_pybullet_drones.envs.FlyThruGateAvitary import FlyThruGateAvitary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

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

#################################### Testing ###################################
def test(args):
    print("============================================================================================")

    ################## hyperparameters ##################

    max_ep_len = 1000           # max timesteps in one episode
    action_std = 0.1            # set same std for action distribution which was used while saving

    render = True              # render environment on screen
    frame_delay = 0             # if required; add delay b/w frames

    total_test_episodes = 10    # total num of testing episodes

    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor

    lr_actor = 0.0003           # learning rate for actor
    lr_critic = 0.001           # learning rate for critic

    #####################################################
    DEFAULT_GUI = True
    DEFAULT_RECORD_VIDEO = True
    DEFAULT_OUTPUT_FOLDER = './results/results_thrugate_ppo'
    if not os.path.exists(DEFAULT_OUTPUT_FOLDER):
        os.makedirs(DEFAULT_OUTPUT_FOLDER)
    DEFAULT_COLAB = False

    DEFAULT_OBS = ObservationType('kin') # 'kin' or 'rgb'
    DEFAULT_ACT = ActionType('rpm') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'
    filename = os.path.join(DEFAULT_OUTPUT_FOLDER, 'recording_'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    if not os.path.exists(filename):
        print(filename)
        os.makedirs(filename+'/')
    
    env = FlyThruGateAvitary(gui=DEFAULT_GUI,
                         obs=DEFAULT_OBS,
                         act=DEFAULT_ACT,
                         record=DEFAULT_RECORD_VIDEO,
                         output_folder=DEFAULT_OUTPUT_FOLDER
                         )

    # state space dimension
    state_dim = 16
    # action space dimension
    action_dim = 4

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std)

    # preTrained weights directory

    random_seed = 0             #### set this to load a particular checkpoint trained on random seed
    run_num_pretrained = 0      #### set this to load a particular checkpoint num

    # checkpoint_path = "log_dir/thrugate_ppo/19980_ppo_drone.pth"
    
    checkpoint_path = args.model_path
    

    print("Loading network from : " + checkpoint_path)

    ppo_agent.load(checkpoint_path)

    print("=" * 60)

    rewards = []
    lengths = []
    successes = 0
    success_times = []

    for ep in range(total_test_episodes):
        obs, info = env.reset(seed=42 + ep, options={})
        ep_reward = 0
        ep_len = 0
        start = time.time()
        for i in range((env.EPISODE_LEN_SEC+20)*env.CTRL_FREQ):
            action = ppo_agent.select_action(obs)
            action = _reshape_action_for_env(action, env.action_space)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_len += 1
            if render:
                env.render()
                sync(i, start, env.CTRL_TIMESTEP)
            if terminated or truncated:
                if ep_reward > 10:
                    successes += 1
                    success_times.append(ep_len / env.CTRL_FREQ)
                    print(f"Episode {ep+1} succeeded in {ep_len/env.CTRL_FREQ:.2f} seconds.")
                break
        rewards.append(ep_reward)
        lengths.append(ep_len)
        ppo_agent.buffer.clear()
        print(f"Episode {ep+1}/{total_test_episodes} | "
              f"Reward {ep_reward:.2f} | "
              f"Len {ep_len} | "
              f"Success {'✓' if ep_reward>5 else '✗'}")

    env.close()
    # Summary
    print("--------------------------------------------------------------------------------------------")
    print(f"Avg reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Success rate: {successes}/{total_test_episodes} ({100*successes/total_test_episodes:.1f}%)")
    if success_times:
        print(f"Avg success time: {np.mean(success_times):.2f}s ± {np.std(success_times):.2f}s")
        print(f"Best: {np.min(success_times):.2f}s | Worst: {np.max(success_times):.2f}s")

    # Save summary
    out_dir = os.path.join(DEFAULT_OUTPUT_FOLDER, "test_summary")
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "ppo_test_results.txt")
    with open(summary_path, "w") as f:
        f.write("PPO FlyThruGateAvitary Test Results\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Episodes: {total_test_episodes}\n")
        f.write(f"Avg reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}\n")
        f.write(f"Success rate: {successes}/{total_test_episodes} ({100*successes/total_test_episodes:.1f}%)\n")
        if success_times:
            f.write(f"Avg success time: {np.mean(success_times):.2f}s ± {np.std(success_times):.2f}s\n")
            f.write(f"Best: {np.min(success_times):.2f}s | Worst: {np.max(success_times):.2f}s\n")
        f.write("\nEpisode details:\n")
        for idx, (r, l) in enumerate(zip(rewards, lengths)):
            f.write(f"Ep {idx+1}: reward={r:.2f}, len={l}")
            if r > 5:
                f.write(f", success_time={l/env.CTRL_FREQ:.2f}s")
            f.write("\n")
        if success_times:
            f.write("\nSuccess times (s): " + ", ".join(f"{t:.2f}" for t in success_times) + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test PPO agent for drone gate navigation')
    parser.add_argument('--model_path', type=str,
                       default='/home/tuan/Desktop/drone_rl_control/log_dir/ppo_training_thrugate/ppo_20251217_100841/ppo_model_ep23481.pt',
                       help='Path to the trained model checkpoint')

    args = parser.parse_args()

    test(args)