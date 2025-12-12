#!/usr/bin/env python3

import os
import time
from datetime import datetime
import argparse
import gymnasium as gym
import numpy as np
import torch
import csv
from TD3.td3 import TD3Agent
from TD3.td3 import Actor, Critic,  ReplayBuffer
import numpy as np

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.FlyThruGateAvitary import FlyThruGateAvitary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from collections import deque

############################### Training ####################################
## Test function 
def test(env, agent, render=True, max_t_step=1000, explore=False, n_times=1):
    sum_scores = 0
    for i in range(n_times):
        state, info = env.reset()
        score = 0
        done=False
        t = int(0)
        while not done and t < max_t_step:
            t += int(1)
            action = agent.get_action(state, explore=explore)
            action = action.clip(min=env.action_space.low, max=env.action_space.high)
            # print(action) # Debug: print action

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = next_state
            score += reward
            if render:
                env.render()
        sum_scores += score
    mean_score = sum_scores/n_times
    print('\rTest Episodes\tMean Score: {:.2f}'.format(mean_score))
    return mean_score


def train(env, agent, n_episodes=5000, score_limit=300.0, explore_episode=50, 
          test_f=200, max_t_step=1000, csv_filename="training_td3_data.csv", 
          learn_every=1, warmup_steps=50):
    """
    Improved training function with CSV logging
    
    Args:
        csv_filename: Tên file CSV để lưu dữ liệu
        learn_every: Số steps giữa mỗi lần học (TD3 standard = 1)
        warmup_steps: Số steps random trước khi bắt đầu học
    """
    scores_deque = deque(maxlen=100)
    scores = []
    test_scores = []
    max_score = -np.inf
    total_steps = 0
    
    # Create output folder if it doesn't exist
    os.makedirs(DEFAULT_OUTPUT_FOLDER, exist_ok=True)
    csv_path = os.path.join(DEFAULT_OUTPUT_FOLDER, csv_filename)
    
    # Open CSV file and write header
    with open(csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['episode', 'timestep', 'reward', 'cumulative_reward', 'done'])
    
    print(f"Logging training data to: {csv_path}")

    for i_episode in range(1, n_episodes+1):
        state, info = env.reset()
        score = 0
        done = False
        agent.train_mode()
        t = int(0)
        episode_data = []  # save data for this episode
        
        while not done and t < max_t_step:    
            t += int(1)
            total_steps += 1
            
            # Exploration: random action in period of warmup steps
            action = agent.get_action(state, explore=True)
            
            if i_episode == 1 and t == 1:
                print(f"Action type: {type(action)}")
                print(f"Action shape: {action.shape}")
                print(f"Action: {action}")
                
            action = action.clip(min=env.action_space.low, max=env.action_space.high)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Use done instead of info["dead"] for consistency
            agent.memory.add(state, action, reward, next_state, done)
            
            state = next_state
            score += reward
            # Save data for CSV
            # reward - in this step and cumulative reward till now 
            episode_data.append([i_episode, total_steps, reward, score, done])
            
            # IMPORTANT: Learn after every step (TD3 standard practice)
            if total_steps >= warmup_steps and total_steps % learn_every == 0:
                if len(agent.memory) > agent.batch_size:
                    agent.learn_one_step()
            
            agent.step_end()

        # Save data to CSV after episode ends (to reduce I/O operations)
        with open(csv_path, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerows(episode_data)
        
        # Episode ended
        if i_episode > explore_episode:
            agent.episode_end()

        scores_deque.append(score)
        avg_score_100 = np.mean(scores_deque)
        scores.append((i_episode, score, avg_score_100))
        
        if i_episode % 5 == 0 or i_episode == 1:
            print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}\tSteps: {}'.format(
                i_episode, avg_score_100, score, total_steps), end="")

        # Testing and saving
        if i_episode % test_f == 0 or avg_score_100 > score_limit:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            agent.eval_mode()
            test_score = test(env, agent, render=False, n_times=20)
            test_scores.append((i_episode, test_score))
            agent.save_ckpt('ep'+str(i_episode), prefix="mlp_model")
            
            # Save test scores to a separate file
            test_csv_path = os.path.join(DEFAULT_OUTPUT_FOLDER, "test_scores.csv")
            with open(test_csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if i_episode == test_f:  # First time
                    writer.writerow(['episode', 'test_score'])
                writer.writerow([i_episode, test_score])
            
            if avg_score_100 > score_limit:
                print(f"\n Solved! Average score {avg_score_100:.2f} > {score_limit}")
                break
            agent.train_mode()

    print(f"\n✅ Training completed! Data saved to {csv_path}")
    return np.array(scores).transpose(), np.array(test_scores).transpose()


if __name__ == '__main__':
    DEFAULT_GUI = False
    DEFAULT_RECORD_VIDEO = False
    DEFAULT_OUTPUT_FOLDER = 'log_dir/td3_thrugate'
    DEFAULT_COLAB = False

    DEFAULT_OBS = ObservationType('kin') # 'kin' or 'rgb' , KIN - (1,12) , RGB - (3,64,64)
    DEFAULT_ACT = ActionType('rpm') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'

    # Tạo timestamp cho run này
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"training_data_{timestamp}.csv"

    env = FlyThruGateAvitary(obs=DEFAULT_OBS, act=DEFAULT_ACT, gui=DEFAULT_GUI, record=DEFAULT_RECORD_VIDEO)
    agent = TD3Agent(Actor, Critic, clip_low=-1, clip_high=1, state_size=12, action_size=4)
    
    print("Starting TD3 Drone Training...")
    scores, test_scores = train(
        env, agent, 
        n_episodes=100000,
        csv_filename=csv_filename,
        learn_every=1,  # Learn after every step
        warmup_steps=50  # 50 steps random to fill replay buffer
    )
