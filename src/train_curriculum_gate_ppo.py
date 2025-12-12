import os
from datetime import datetime
import numpy as np

from agents.ppo_agent import PPO
from gym_pybullet_drones.envs.FlyThruGateCurriculumAvitary import FlyThruGateCurriculumAvitary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType


def train():
    # Advance curriculum when success rate over the last N episodes exceeds threshold.
    SUCCESS_WINDOW = 30
    SUCCESS_RATE_THRESHOLD = 0.7

    DEFAULT_GUI = True
    DEFAULT_RECORD_VIDEO = False
    DEFAULT_OUTPUT_FOLDER = 'results'
    DEFAULT_COLAB = False

    DEFAULT_OBS = ObservationType('kin')  # 'kin' or 'rgb'
    DEFAULT_ACT = ActionType('rpm')       # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'

    env = FlyThruGateCurriculumAvitary(
        obs=DEFAULT_OBS,
        act=DEFAULT_ACT,
        gui=DEFAULT_GUI,
        record=DEFAULT_RECORD_VIDEO,
        success_window=SUCCESS_WINDOW,
        success_rate_threshold=SUCCESS_RATE_THRESHOLD,
    )

    state_dim = env.observation_space.shape[-1]
    action_dim = env.action_space.shape[-1]
    action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)

    ################ PPO hyperparameters ################
    max_training_timesteps = int(3e6)   # break training loop if timeteps > max_training_timesteps
    K_epochs = 80                       # update policy for K epochs in one PPO update

    eps_clip = 0.2                      # clip parameter for PPO
    gamma = 0.99                        # discount factor
    lr_actor = 0.0003                   # learning rate for actor network
    lr_critic = 0.001                   # learning rate for critic network

    log_dir = "log_dir/"
    run_num = "curriculum_gate_ppo/"

    os.makedirs(os.path.join(log_dir, str(run_num)), exist_ok=True)
    log_f_name = os.path.join(log_dir, run_num, 'PPO_log.csv')
    checkpoint_path = os.path.join(log_dir, run_num, "ppo_drone.pth")

    print("current logging run number for gym pybullet drone :", run_num)
    print("logging at :", log_f_name)
    log_f = open(log_f_name, "w+")
    log_f.write('episode,timestep,reward,level,success_rate\n')

    update_timestep = env.EPISODE_LEN_SEC * env.CTRL_FREQ * 4
    print_freq = env.EPISODE_LEN_SEC * env.CTRL_FREQ * 10
    log_freq = env.EPISODE_LEN_SEC * env.CTRL_FREQ * 2
    save_model_freq = int(1e5)

    print_running_reward = 0
    print_running_episodes = 0
    log_running_reward = 0
    log_running_episodes = 0

    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std)

    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) :", start_time)
    print(f"Episode length (steps): {env.EPISODE_LEN_SEC * env.CTRL_FREQ}")

    time_step = 0
    i_episode = 0
    while time_step <= max_training_timesteps:
        obs, info = env.reset(seed=42, options={})
        current_level = info.get("curriculum_level", 0)
        current_rate = info.get("recent_success_rate", 0.0)

        current_ep_reward = 0
        for _ in range(env.EPISODE_LEN_SEC * env.CTRL_FREQ):
            action = ppo_agent.select_action(obs)
            action = np.expand_dims(action, axis=0)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            current_level = info.get("curriculum_level", current_level)
            current_rate = info.get("recent_success_rate", current_rate)

            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step += 1
            current_ep_reward += reward

            if time_step % update_timestep == 0:
                ppo_agent.update()

            if time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            if time_step % log_freq == 0 and log_running_episodes > 0:
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)
                log_level = info.get("curriculum_level", current_level)
                log_rate = info.get("recent_success_rate", current_rate)
                log_f.write('{}, {}, {}, {}, {:.3f}\n'.format(i_episode, time_step, log_avg_reward, log_level, log_rate))
                log_f.flush()
                log_running_reward = 0
                log_running_episodes = 0

            if time_step % print_freq == 0 and print_running_episodes > 0:
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)
                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {} \t\t Level : {} \t\t Success rate: {:.2f}".format(
                    i_episode, time_step, print_avg_reward, info.get("curriculum_level", current_level), info.get("recent_success_rate", current_rate)))
                print_running_reward = 0
                print_running_episodes = 0

            if time_step % save_model_freq == 0:
                ckpt = os.path.join(log_dir, str(run_num), f"{i_episode}_ppo_drone.pth")
                print("--------------------------------------------------------------------------------------------")
                print("saving model at :", ckpt)
                ppo_agent.save(ckpt)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            if done:
                break

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1

    log_f.close()
    env.close()

    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) :", start_time)
    print("Finished training at (GMT) :", end_time)
    print("Total training time  :", end_time - start_time)
    print("============================================================================================")


if __name__ == '__main__':
    train()
