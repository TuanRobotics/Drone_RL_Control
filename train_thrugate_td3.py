import os
import time
from datetime import datetime
import argparse
import gymnasium as gym
import numpy as np
import torch
import csv
from TD3.td3_agent import TD3Agent
from TD3.td3_agent import Actor, Critic,  ReplayBuffer
import numpy as np
import matplotlib.pyplot as plt

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
    scores_deque = []
    scores = []
    test_scores = []
    max_score = -np.inf
    total_steps = 0
    
    # # Create output folder if it doesn't exist
    # os.makedirs(DEFAULT_OUTPUT_FOLDER, exist_ok=True)
    # csv_path = os.path.join(DEFAULT_OUTPUT_FOLDER, csv_filename)
    
    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"td3_training\\td3_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    # # Open CSV file and write header
    # with open(csv_path, 'w', newline='') as csvfile:
    #     csv_writer = csv.writer(csvfile)
    #     csv_writer.writerow(['episode', 'timestep', 'reward', 'cumulative_reward', 'done'])
    
    # print(f"Logging training data to: {csv_path}")

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

        # # Save data to CSV after episode ends (to reduce I/O operations)
        # with open(csv_path, 'a', newline='') as csvfile:
        #     csv_writer = csv.writer(csvfile)
        #     csv_writer.writerows(episode_data)
        
        # Episode ended
        if i_episode > explore_episode:
            agent.episode_end()

        scores_deque.append(score)
        # print(scores_deque)
        avg_score_100 = np.mean(scores_deque)
        scores.append((i_episode, score, avg_score_100))
        
        if i_episode % 10 == 0:          
            print(f"Episode {i_episode}/{n_episodes} | "
                f"Steps: {total_steps} | "
                f"Reward: {score:.2f} | "
                f"Avg(10): {avg_score_100:.2f}")

        # # Testing and saving
        # if i_episode % test_f == 0 or avg_score_100 > score_limit:
        #     print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        #     agent.eval_mode()
        #     test_score = test(env, agent, render=False, n_times=20)
        #     test_scores.append((i_episode, test_score))
        #     agent.save_ckpt('ep'+str(i_episode), prefix="mlp_model")
            
        #     # Save test scores to a separate file
        #     test_csv_path = os.path.join(DEFAULT_OUTPUT_FOLDER, "test_scores.csv")
        #     with open(test_csv_path, 'a', newline='') as f:
        #         writer = csv.writer(f)
        #         if i_episode == test_f:  # First time
        #             writer.writerow(['episode', 'test_score'])
        #         writer.writerow([i_episode, test_score])
            
        #     if avg_score_100 > score_limit:
        #         print(f"\n Solved! Average score {avg_score_100:.2f} > {score_limit}")
        #         break
        #     agent.train_mode()

    # Final save + plot
    print(f"{'='*60}\n")
    agent.save(os.path.join(save_dir, "td3_model_final.pt"))
    plot_training_curves(scores_deque, [], [],
                         [], save_dir, "final")

    print(f"\nTraining completed! Final model saved in {save_dir}")

    # print(f"\n✅ Training completed! Data saved to {csv_path}")
    return np.array(scores).transpose(), np.array(test_scores).transpose()

def plot_training_curves(episode_rewards, eval_rewards, q1_losses,
                         policy_losses, save_dir, episode):
    """Plot and save training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Episode rewards
    axes[0, 0].plot(episode_rewards, alpha=0.3, label='Episode Reward')
    if len(episode_rewards) >= 50:
        smoothed = np.convolve(episode_rewards, np.ones(50)/50, mode='valid')
        axes[0, 0].plot(range(49, len(episode_rewards)), smoothed,
                       label='Smoothed (50 episodes)', linewidth=2)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Training Rewards')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Evaluation rewards
    if eval_rewards:
        axes[0, 1].plot(eval_rewards, marker='o')
        axes[0, 1].set_xlabel('Evaluation Step')
        axes[0, 1].set_ylabel('Average Reward')
        axes[0, 1].set_title('Evaluation Rewards')
        axes[0, 1].grid(True)

    # Q losses
    if q1_losses:
        axes[1, 0].plot(q1_losses, alpha=0.3)
        if len(q1_losses) >= 100:
            smoothed = np.convolve(q1_losses, np.ones(100)/100, mode='valid')
            axes[1, 0].plot(range(99, len(q1_losses)), smoothed, linewidth=2)
        axes[1, 0].set_xlabel('Update Step')
        axes[1, 0].set_ylabel('Q1 Loss')
        axes[1, 0].set_title('Critic Loss')
        axes[1, 0].grid(True)

    # Policy losses
    if policy_losses:
        axes[1, 1].plot(policy_losses, alpha=0.3)
        if len(policy_losses) >= 100:
            smoothed = np.convolve(policy_losses, np.ones(100)/100, mode='valid')
            axes[1, 1].plot(range(99, len(policy_losses)), smoothed, linewidth=2)
        axes[1, 1].set_xlabel('Update Step')
        axes[1, 1].set_ylabel('Policy Loss')
        axes[1, 1].set_title('Actor Loss')
        axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'training_curves_ep{episode}.png'))
    plt.close()


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
        n_episodes=3000,
        csv_filename=csv_filename,
        learn_every=1,  # Learn after every step
        warmup_steps=50  # 50 steps random to fill replay buffer
    )
