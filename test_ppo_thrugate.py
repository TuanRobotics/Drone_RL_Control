#!/usr/bin/env python3
import os
import time
from datetime import datetime
import argparse
import gymnasium as gym
import numpy as np
import torch
from PPO.ppo_agent import PPO

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from gym_pybullet_drones.envs.FlyThruGateAvitary import FlyThruGateAvitary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType


def test():
    print("============================================================================================")

    ################## hyperparameters ##################
    hidden_dim = 256
    lr_actor = 3e-4
    lr_critic = 1e-3
    gamma = 0.99
    gae_lambda = 0.95
    clip_epsilon = 0.2
    entropy_coef = 0.01
    value_coef = 0.5
    max_grad_norm = 0.5
    K_epochs = 80
    batch_size = 64

    render = True
    total_test_episodes = 10    # Số episode để test
    #####################################################
    
    DEFAULT_GUI = True
    DEFAULT_RECORD_VIDEO = True
    DEFAULT_OUTPUT_FOLDER = 'results_ppo_thrugate/'
    DEFAULT_OBS = ObservationType('kin')
    DEFAULT_ACT = ActionType('rpm')
    
    filename = os.path.join(DEFAULT_OUTPUT_FOLDER, 'recording_'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    if not os.path.exists(filename):
        os.makedirs(filename+'/')

    # Initialize environment
    env = FlyThruGateAvitary(gui=DEFAULT_GUI,
                           obs=DEFAULT_OBS,
                           act=DEFAULT_ACT,
                           record=DEFAULT_RECORD_VIDEO)

    # State and action dimensions
    state_dim = 12
    action_dim = 4
    
    # Initialize PPO agent
    ppo_agent = PPO(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_epsilon=clip_epsilon,
        entropy_coef=entropy_coef,
        value_coef=value_coef,
        max_grad_norm=max_grad_norm,
        epochs=K_epochs,
        batch_size=batch_size
    )
    
    # Load pretrained model
    checkpoint_path = "log_dir/thrugate/ep_456120_ts_2980000_ppo_gate.pth"
    print("Loading network from: " + checkpoint_path)
    ppo_agent.load(checkpoint_path)
    print("--------------------------------------------------------------------------------------------")

    test_running_reward = 0

    obs, info = env.reset(seed=42, options={})
    ep_reward = 0
    start_time = datetime.now().replace(microsecond=0)
    start = time.time()
    for i in range((env.EPISODE_LEN_SEC+30)*env.CTRL_FREQ):
        action, _, _= ppo_agent.select_action(obs)
        action = np.expand_dims(action, axis=0)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward
        env.render()
        sync(i, start, env.CTRL_TIMESTEP)
        if terminated:
            break

    # clear buffer
    ppo_agent.memory.clear()

    test_running_reward +=  ep_reward
    print('Episode: {} \t\t Reward: {}'.format(0, round(ep_reward, 2)))
    ep_reward = 0

    env.close()
    
    print("\n============================================================================================")
    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print(f"Average test reward: {avg_test_reward}")
    print("============================================================================================")


if __name__ == '__main__':
    test()