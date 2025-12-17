# #!/usr/bin/env python3
# import os
# import csv
# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# from datetime import datetime
# from pathlib import Path

# from agents.ppo_agent import PPO
# from gym_pybullet_drones.envs.FlyThruGateAvitary import FlyThruGateAvitary
# from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


# def _reshape_action_for_env(action, action_space):
#     """Ensure the action matches the environment's expected shape."""
#     action = np.asarray(action, dtype=np.float32)
#     if action.ndim == 1:
#         action = np.expand_dims(action, axis=0)
#     try:
#         action = action.reshape(action_space.shape)
#     except Exception:
#         pass
#     return action


# # ============================================================================
# # Training
# # ============================================================================

# def train_ppo():
    
#     """Train PPO agent, mirroring SAC logging/plotting workflow."""
#     env = FlyThruGateAvitary(drone_model=DroneModel.CF2X,
#                              initial_xyzs=np.array([[0, 0, 0.5]]),
#                              physics=Physics.PYB,
#                              pyb_freq=240,
#                              ctrl_freq=30,
#                              gui=False,
#                              use_curriculum=False,
#                              obs=ObservationType.KIN,
#                              act=ActionType.RPM)
    
#     env_eval = FlyThruGateAvitary(drone_model=DroneModel.CF2X,
#                                  initial_xyzs=np.array([[0, 0, 0.5]]),
#                                  physics=Physics.PYB,
#                                  pyb_freq=240,
#                                  ctrl_freq=30,
#                                  gui=False,
#                                  use_curriculum=False,
#                                  obs=ObservationType.KIN,
#                                  act=ActionType.RPM)

#     obs, _ = env.reset()
#     state_dim = int(np.prod(env.observation_space.shape))
#     action_dim = int(np.prod(env.action_space.shape))

#     # Hyperparameters PPO 
#     K_epochs=80
#     eps_clip=0.2
#     gamma=0.99
#     lr_actor=3e-4
#     lr_critic=1e-3
#     action_std=0.7
#     action_std_decay_rate=0.05
#     min_action_std=0.1
#     action_std_decay_freq=int(2.5e5)
#     eval_every=100
#     num_eval_episodes=5

#     agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs,
#                 eps_clip, action_std)

#     # Metrics
#     episode_rewards = []
#     eval_rewards = []
#     critic_losses = []
#     actor_losses = []
#     entropy_hist = []

#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     base_dir = Path("/home/tuan/Desktop/drone_rl_control/log_dir")
#     save_dir = base_dir / "ppo_training_thrugate" / f"ppo_{timestamp}"
#     save_dir.mkdir(parents=True, exist_ok=True)
#     csv_prefix = base_dir / f"ppo_metrics_thrugate{timestamp}"
#     log_f_name = save_dir / 'PPO_log.csv'
#     log_f = open(log_f_name, "w+")
#     log_f.write('episode,timestep,reward, averagate_reward_100\n')

#     print("\n" + "=" * 60)
#     print("Starting PPO Training for Drone Gate Navigation")
#     print("=" * 60 + "\n")

#     total_timesteps = 0
#     update_timestep = env.EPISODE_LEN_SEC*env.CTRL_FREQ * 2
#     log_freq =  env.EPISODE_LEN_SEC*env.CTRL_FREQ * 4
#     max_training_timesteps = int(5e6) 
#     total_timesteps = 0
#     i_episode = 0
#     log_running_reward = 0
#     losses = []
#     loss_steps = []

#     while total_timesteps <= max_training_timesteps:
#         state, _ = env.reset()
#         episode_reward = 0

#         for i in range(env.EPISODE_LEN_SEC * env.CTRL_FREQ):
#             action = agent.select_action(state)
#             action_env = _reshape_action_for_env(action, env.action_space)
#             if i_episode == 1:
#                 print("Sampled action:", action_env)

#             next_state, reward, terminated, truncated, _ = env.step(action_env)
#             done = terminated or truncated 

#             agent.buffer.rewards.append(reward)
#             agent.buffer.is_terminals.append(done)

#             total_timesteps += 1
#             episode_reward += reward
#             state = next_state

#             if total_timesteps % update_timestep == 0: 
#                 loss_obj = agent.update()
#                 losses.append(loss_obj)
#                 loss_steps.append(total_timesteps) 

#             if total_timesteps % action_std_decay_freq == 0:
#                 agent.decay_action_std(action_std_decay_rate, min_action_std)

#             if total_timesteps % log_freq == 0:

#                 # log average reward till last episode
#                 log_avg_reward = np.mean(episode_rewards) if episode_rewards else 0
#                 log_avg_reward = round(log_avg_reward, 4)
#                 log_avg_reward_100 = np.mean(avg_reward_100) if episode_rewards else 0
#                 log_avg_reward_100 = round(log_avg_reward_100, 4)

#                 log_f.write('{}, {}, {}, {}\n'.format(i_episode, total_timesteps, log_avg_reward, log_avg_reward_100))
#                 log_f.flush()

#             if done:
#                 break
        
#         episode_rewards.append(episode_reward)
#         avg_reward_100 = np.mean(episode_rewards[-100:])
#         i_episode += 1

#         if i_episode % 2400 == 0:
#             agent.save(save_dir / f"ppo_model_ep{i_episode}.pt")
#             # Plot training curves
#             plt.figure()
#             plt.plot(episode_rewards, alpha=0.3, label="Episode Reward")
#             if len(episode_rewards) >= 50:
#                 smoothed = np.convolve(episode_rewards, np.ones(50) / 50, mode='valid')
#                 plt.plot(range(49, len(episode_rewards)), smoothed,
#                          label='Smoothed (50 episodes)', linewidth=2)
#             plt.xlabel("Episodes")
#             plt.ylabel("Rewards")
#             plt.title("PPO Training Rewards over Episodes")
#             plt.legend()
#             plt.grid()
#             plt.tight_layout()
#             plt.savefig(save_dir / f"ppo_rewards_ep{i_episode}.png")
#             plt.close()
            

#         if (i_episode) % eval_every == 0:
#             eval_reward = evaluate_policy(env_eval, agent, num_eval_episodes)
#             eval_rewards.append(eval_reward)
#             print(f"\n{'='*60}")
#             print(
#                 f"Episode {i_episode} | "
#                 f"Steps: {total_timesteps} | "
#                 f"Reward: {episode_reward:.2f} | "
#                 f"Avg Reward over 100 episodes: {avg_reward_100:.2f} | "
#                 f"Evaluation at Episode {i_episode}: Avg Evaluated Reward = {eval_reward:.2f}"
#             )

#             print(f"{'='*60}\n")
#     env.close()
#     env_eval.close()
#     log_f.close()

#     print(f"{'='*60}\n")
#     agent.save(save_dir / "ppo_model_final.pt")

#     # Plot losses 
#     plt.figure()
#     plt.plot(loss_steps, losses)
#     plt.xlabel("Timesteps")
#     plt.ylabel("PPO Loss")
#     plt.title("PPO Loss over Time")
#     plt.grid()
#     plt.tight_layout()
#     plt.savefig(save_dir / "ppo_loss.png")
#     plt.close()

#     print(f"\nTraining completed! Final model saved in {save_dir}")


# def evaluate_policy(env, agent, num_episodes=5):
#     """Evaluate the policy without modifying PPO buffers."""
#     eval_rewards = []
#     policy_device = next(agent.policy_old.parameters()).device

#     for _ in range(num_episodes):
#         state, _ = env.reset(seed=42, options={})
#         episode_reward = 0
#         done = False

#         while not done:
#             state_tensor = torch.as_tensor(state,
#                                            dtype=torch.float32,
#                                            device=policy_device)
#             with torch.no_grad():
#                 action_mean = agent.policy_old.actor(state_tensor)
#             action = action_mean.cpu().numpy()
#             action_env = _reshape_action_for_env(action, env.action_space)

#             next_state, reward, terminated, truncated, _ = env.step(action_env)
#             episode_reward += reward
#             state = next_state
#             done = terminated or truncated

#         eval_rewards.append(episode_reward)

#     return float(np.mean(eval_rewards))

# # ============================================================================
# # Main
# # ============================================================================
# if __name__ == "__main__":
#     train_ppo()

import os
import time
from datetime import datetime
import argparse
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
from agents.ppo_agent import PPO
from pathlib import Path

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.FlyThruGateAvitary import FlyThruGateAvitary
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
# init environment
def train():
    DEFAULT_GUI = True
    DEFAULT_RECORD_VIDEO = False
    DEFAULT_OUTPUT_FOLDER = 'results'
    DEFAULT_COLAB = False

    DEFAULT_OBS = ObservationType('kin') # 'kin' or 'rgb'
    DEFAULT_ACT = ActionType('rpm') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'


    env = FlyThruGateAvitary(obs=DEFAULT_OBS, act=DEFAULT_ACT)
    # init agent
    state_dim = 16
    action_dim = 4
    action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05       # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(3e5)  # action_std decay frequency (in num timesteps)
    #####################################################

    ################ PPO hyperparameters ################
    #update_timestep = max_ep_len * 4      # update policy every n timesteps
    max_training_timesteps = int(5e6)   # break training loop if timeteps > max_training_timesteps
    K_epochs = 80               # update policy for K epochs in one PPO update

    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor
    lr_actor = 3e-4         # learning rate for actor network
    lr_critic = 1e-3       # learning rate for critic network


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path("/home/tuan/Desktop/drone_rl_control/log_dir")
    save_dir = base_dir / "ppo_training_thrugate" / f"ppo_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = save_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    csv_prefix = base_dir / f"ppo_metrics_thrugate{timestamp}"
    log_f_name = save_dir / 'PPO_log.csv'
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
    episode_rewards = []

    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std)

    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    time_step = 0
    i_episode = 0

    def save_reward_plot(episode_count: int) -> None:
        """Plot episode rewards and persist to the run directory."""
        plt.figure(figsize=(10, 5))
        plt.plot(episode_rewards, alpha=0.35, label="Episode reward")
        if len(episode_rewards) >= 50:
            smoothed = np.convolve(episode_rewards, np.ones(50)/50, mode='valid')
            plt.plot(range(49, len(episode_rewards)), smoothed,
                     label='Smoothed (50 episodes)', linewidth=2)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title(f"PPO training rewards up to episode {episode_count}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plots_dir / f"ppo_rewards_ep{episode_count}.png")
        plt.close()

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
                ppo_agent.update()

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
                ppo_agent.save(save_dir / f"ppo_model_ep{i_episode}.pt")
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
        episode_rewards.append(current_ep_reward)

        i_episode += 1
        if i_episode % 2400 == 0:
            save_reward_plot(i_episode)



    log_f.close()
    env.close()

    ppo_agent.save(save_dir / f"ppo_model_final.pt")

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")
    
    if episode_rewards:
        save_reward_plot(i_episode)


if __name__ == '__main__':
    train()
