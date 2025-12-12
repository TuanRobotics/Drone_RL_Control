import os
from datetime import datetime
import numpy as np

from SAC.sac_agent import SACAgent, ReplayBuffer
from gym_pybullet_drones.envs.DroneRacingAviary import DroneRacingAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType


def train():
    """
    """
    # Environment defaults (mirrors PPO)
    DEFAULT_GUI = False
    DEFAULT_RECORD_VIDEO = False
    DEFAULT_OBS = ObservationType('kin')
    DEFAULT_ACT = ActionType('rpm')

    env = DroneRacingAviary(
        gui=DEFAULT_GUI,
        record=DEFAULT_RECORD_VIDEO,
        obs=DEFAULT_OBS,
        act=DEFAULT_ACT,
    )

    # Derive dimensions from the first reset
    obs, _ = env.reset(seed=42, options={})
    state_dim = int(np.prod(np.asarray(obs).shape))
    action_dim = env.action_space.shape[-1]
    max_ep_len = env.EPISODE_LEN_SEC * env.CTRL_FREQ

    # SAC hyperparameters
    max_training_timesteps = int(3e6)
    batch_size = 256
    replay_capacity = 1_000_000
    start_steps = 10_000       # pure random actions to bootstrap the buffer
    start_update = 5_000       # start learning after this many steps
    update_every = 50
    updates_per_iter = 2
    lr = 3e-4
    gamma = 0.99
    tau = 0.005

    # Logging/saving setup (kept similar to PPO)
    log_dir = "log_dir/"
    run_num = "results_racing_sac_single/"
    log_f_name = log_dir + run_num + 'SAC_log.csv'
    os.makedirs(os.path.join(log_dir, str(run_num)), exist_ok=True)

    print("current logging run number for  gym pybulet drone : ", run_num)
    print("logging at : " + log_f_name)
    log_f = open(log_f_name, "w+")
    log_f.write('episode,timestep,reward\n')

    update_timestep = max_ep_len * 4
    print_freq = max_ep_len * 10
    log_freq = max_ep_len * 2
    save_model_freq = int(1e5)

    print("EPISODE_LEN_SEC:", env.EPISODE_LEN_SEC)
    print("CTRL_FREQ:", env.CTRL_FREQ)
    print("step per episode", max_ep_len)

    print_running_reward = 0
    print_running_episodes = 0
    log_running_reward = 0
    log_running_episodes = 0

    sac_agent = SACAgent(state_dim, action_dim, hidden_dim=256, lr=lr, gamma=gamma, tau=tau)
    replay_buffer = ReplayBuffer(capacity=replay_capacity)

    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    time_step = 0
    i_episode = 0

    while time_step <= max_training_timesteps:
        obs, info = env.reset(seed=42, options={})
        current_ep_reward = 0

        for i in range((env.EPISODE_LEN_SEC) * env.CTRL_FREQ):
            if time_step < start_steps:
                action = env.action_space.sample()
            else:
                action = sac_agent.select_action(obs, evaluate=False)
            if time_step==1:
                print("Sampled action:", action)
            if time_step == start_steps:
                print(f"Sampled action at time step {time_step} from policy:", action)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            replay_buffer.push(obs, action, reward, next_obs, done)
            current_ep_reward += reward
            time_step += 1

            # update SAC agent (mirrors PPO timing but keeps SAC updates)
            if (time_step % update_timestep == 0 and
                    len(replay_buffer) >= batch_size and
                    time_step >= start_update):
                for _ in range(updates_per_iter):
                    sac_agent.update(replay_buffer, batch_size=batch_size)

            # Log to CSV
            if time_step % log_freq == 0:
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)
                log_f.write('{}, {}, {}\n'.format(i_episode, time_step, log_avg_reward))
                log_f.flush()
                log_running_reward = 0
                log_running_episodes = 0

            # Console print
            if time_step % print_freq == 0:
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)
                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(
                    i_episode, time_step, print_avg_reward))
                print_running_reward = 0
                print_running_episodes = 0

            # Save model weights
            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                checkpoint_path = os.path.join(log_dir, str(run_num), f"{i_episode}_sac_drone.pth")
                print("saving model at : " + checkpoint_path)
                sac_agent.save(checkpoint_path)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            if done or time_step > max_training_timesteps:
                break

            obs = next_obs

        print_running_reward += current_ep_reward
        print_running_episodes += 1
        log_running_reward += current_ep_reward
        log_running_episodes += 1
        i_episode += 1

    final_path = os.path.join(log_dir, str(run_num), "sac_drone_final.pth")
    sac_agent.save(final_path)
    log_f.close()
    env.close()

    end_time = datetime.now().replace(microsecond=0)
    print("============================================================================================")
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("Model saved at       : ", final_path)
    print("============================================================================================")


if __name__ == '__main__':
    train()
