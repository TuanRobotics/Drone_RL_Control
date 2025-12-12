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
from gym_pybullet_drones.envs.FlyThruGateAvitary import FlyThruGateAvitary
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

############################### Training ####################################

# init environment
def train():
    DEFAULT_GUI = True
    DEFAULT_RECORD_VIDEO = False
    DEFAULT_OUTPUT_FOLDER = 'results'
    DEFAULT_COLAB = False

    DEFAULT_OBS = ObservationType('kin') # 'kin' or 'rgb' , KIN - (1,12) , RGB - (3,64,64)
    DEFAULT_ACT = ActionType('rpm') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'


    env = FlyThruGateAvitary(obs=DEFAULT_OBS, act=DEFAULT_ACT)
    # init agent
    state_dim = 12
    action_dim = 4
    action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)
    #####################################################

    ################ PPO hyperparameters ################
    #update_timestep = max_ep_len * 4      # update policy every n timesteps
    max_training_timesteps = int(3e6)   # break training loop if timeteps > max_training_timesteps
    K_epochs = 80               # update policy for K epochs in one PPO update

    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor
    lr_actor = 0.0003       # learning rate for actor network
    lr_critic = 0.001       # learning rate for critic network

    random_seed = 0         # set random seed if required (0 = no random seed)
    log_dir = "log_dir/"

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    run_num = "thrugate_ppo"
    log_f_name = log_dir + 'PPO_log_' + str(run_num) + ".csv"
    if not os.path.exists(os.path.join(log_dir, str(run_num))):
        os.mkdir(os.path.join(log_dir, str(run_num)))
    checkpoint_path = log_dir + "ppo_drone.pth"
    plot_dir = os.path.join(log_dir, "plots_ppo")
    os.makedirs(plot_dir, exist_ok=True)

    print("current logging run number for " + " gym pybulet drone : ", run_num)
    print("logging at : " + log_f_name)
    log_f = open(log_f_name,"w+")
    log_f.write('episode,timestep,reward\n')
    # log avg reward in the interval (in num timesteps)
    print(env.EPISODE_LEN_SEC)
    print(env.CTRL_FREQ)
    print("step per episode", env.EPISODE_LEN_SEC*env.CTRL_FREQ)
    update_timestep = env.EPISODE_LEN_SEC*env.CTRL_FREQ * 4
    print_freq = env.EPISODE_LEN_SEC*env.CTRL_FREQ  * 10        # print avg reward in the interval (in num timesteps)
    log_freq =  env.EPISODE_LEN_SEC*env.CTRL_FREQ * 2
    save_model_freq = int(1e5)          # save model frequency (in num timesteps)
    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0
    policy_loss_hist = []
    value_loss_hist = []
    entropy_hist = []
    reward_hist = []

    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std)

    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    time_step = 0
    i_episode = 0
    while time_step <= max_training_timesteps:
        obs, info = env.reset(seed=42, options={})

        current_ep_reward = 0
        for i in range((env.EPISODE_LEN_SEC)*env.CTRL_FREQ):
            action = ppo_agent.select_action(obs)
            action = np.expand_dims(action, axis=0)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            #print("Obs:", obs, "\tAction", action, "\tReward:", reward, "\tTerminated:", terminated, "\tTruncated:", truncated)
             # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step += 1
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                pol_loss, val_loss, entropy = ppo_agent.update()
                policy_loss_hist.append(pol_loss)
                value_loss_hist.append(val_loss)
                entropy_hist.append(entropy)

            if time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            if time_step % log_freq == 0:

                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                log_f.write('{}, {}, {}\n'.format(i_episode, time_step, log_avg_reward))
                log_f.flush()

                log_running_reward = 0
                log_running_episodes = 0

            if time_step % print_freq == 0:

                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))


                print_running_reward = 0
                print_running_episodes = 0


            # save model weights
            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                checkpoint_path = os.path.join(log_dir, str(run_num), str(i_episode) +"_ppo_drone.pth")
                print("saving model at : " + checkpoint_path)
                ppo_agent.save(checkpoint_path)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            # break; if the episode is over
            if done:
                break


        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1
        reward_hist.append(current_ep_reward)

        i_episode += 1

    log_f.close()
    env.close()
    # Plot reward and losses
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(reward_hist)
        plt.xlabel("Episode")
        plt.ylabel("Episode Reward")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "episode_rewards.png"))
        plt.close()

        if policy_loss_hist:
            plt.figure()
            plt.plot(policy_loss_hist, label="Policy loss")
            plt.plot(value_loss_hist, label="Value loss")
            plt.xlabel("Update idx")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "losses.png"))
            plt.close()

            plt.figure()
            plt.plot(entropy_hist)
            plt.xlabel("Update idx")
            plt.ylabel("Entropy")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "entropy.png"))
            plt.close()
    except Exception as e:
        print(f"Plotting failed: {e}")

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")

if __name__ == '__main__':
    train()
