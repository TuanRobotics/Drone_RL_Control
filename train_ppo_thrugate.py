#!/usr/bin/env python3
import os
import time
from datetime import datetime
import numpy as np
import torch
from PPO.ppo_agent import PPO

from gym_pybullet_drones.envs.FlyThruGateAvitary import FlyThruGateAvitary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType


def train():
    # Environment settings
    DEFAULT_OBS = ObservationType('kin')  # 'kin' or 'rgb'
    DEFAULT_ACT = ActionType('rpm')  # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'
    
    # Initialize environment
    env = FlyThruGateAvitary(
        obs=DEFAULT_OBS,
        act=DEFAULT_ACT, 
        gui = False
    )
    
    # Environment info
    state_dim = 12
    action_dim = 4
    
    # PPO hyperparameters
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
    
    # Training settings
    max_training_timesteps = int(3e6)
    update_timestep = env.EPISODE_LEN_SEC * env.CTRL_FREQ * 4  # Update every 4 episodes
    print_freq = env.EPISODE_LEN_SEC * env.CTRL_FREQ * 10
    log_freq = env.EPISODE_LEN_SEC * env.CTRL_FREQ * 2
    save_model_freq = int(50000)
    
    # Logging setup
    log_dir = "log_dir/"
    run_num = "thrugate"
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(os.path.join(log_dir, str(run_num))):
        os.makedirs(os.path.join(log_dir, str(run_num)))
    
    log_f_name = log_dir + 'PPO_log_' + str(run_num) + ".csv"
    
    print("=" * 80)
    print("Current logging run: ", run_num)
    print("Logging at: " + log_f_name)
    print("Episode length (sec): ", env.EPISODE_LEN_SEC)
    print("Control frequency: ", env.CTRL_FREQ)
    print("Steps per episode: ", env.EPISODE_LEN_SEC * env.CTRL_FREQ)
    print("Update every: ", update_timestep, " timesteps")
    print("=" * 80)
    
    log_f = open(log_f_name, "w+")
    log_f.write('episode,timestep,reward\n')
    
    # Tracking variables
    print_running_reward = 0
    print_running_episodes = 0
    log_running_reward = 0
    log_running_episodes = 0
    
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
    
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT): ", start_time)
    print("=" * 80)
    
    # Training loop
    time_step = 0
    i_episode = 0
    
    while time_step <= max_training_timesteps:
        obs, info = env.reset(seed=42, options={})
        current_ep_reward = 0
        
        for step in range(env.EPISODE_LEN_SEC * env.CTRL_FREQ):
            # Select action
            action, log_prob, value = ppo_agent.select_action(obs, deterministic=False)

            # if step % 1000 == 0:
            #     print(f"Selected Action: {action}")
            #     print(f"Shape of actions: {action.shape}")

            # Expand action dimensions for environment
            action_expanded = np.expand_dims(action, axis=0)
            
            # Take step in environment
            next_obs, reward, terminated, truncated, info = env.step(action_expanded)
            done = terminated or truncated
            
            # Store transition
            ppo_agent.store_transition(obs, action, reward, done, log_prob, value)
            
            # Update observation
            obs = next_obs
            
            time_step += 1
            current_ep_reward += reward
            
            # Update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update(next_obs)
            
            # Logging
            # if time_step % 100 == 0:
            #     print(f"Episode: {i_episode} \t Timestep: {time_step} \t Reward: {round(current_ep_reward, 2)}")
            if time_step % log_freq == 0 and log_running_episodes > 0:
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)
                
                log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                log_f.flush()
                
                log_running_reward = 0
                log_running_episodes = 0
            
            # Print statistics

            if time_step % print_freq == 0:
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)
                
                print("Episode: {} \t Timestep: {} \t Average Reward: {}".format(
                    i_episode, time_step, print_avg_reward))
                
                print_running_reward = 0
                print_running_episodes = 0
            
            # Save model
            if time_step % save_model_freq == 0:
                print("-" * 80)
                checkpoint_path = os.path.join(log_dir, str(run_num), 
                                             f"ep_{i_episode}_ts_{time_step}_ppo_gate.pth")
                print("Saving model at: " + checkpoint_path)
                ppo_agent.save(checkpoint_path)
                print("Elapsed Time: ", datetime.now().replace(microsecond=0) - start_time)
                print("-" * 80)
            
            # Break if episode is done
            if done:
                break
        
        # Update running rewards
        print_running_reward += current_ep_reward
        print_running_episodes += 1
        
        log_running_reward += current_ep_reward
        log_running_episodes += 1
        
        i_episode += 1
    
    # Cleanup
    log_f.close()
    env.close()
    
    # Print total training time
    print("=" * 80)
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT): ", start_time)
    print("Finished training at (GMT): ", end_time)
    print("Total training time: ", end_time - start_time)
    print("=" * 80)


if __name__ == '__main__':
    train()