import numpy as np
import torch
from gym_pybullet_drones.envs.FlyThruGateAvitary import FlyThruGateAvitary
from TD3.td3 import TD3Agent, Actor, Critic
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
import os
from datetime import datetime


def test_agent(env, agent, episodes=5, max_steps=200, render=False):
    """
    Test TD3 agent trong môi trường FlyThruGateAvitary.
    
    Args:
        env: enviroment  (FlyThruGateAvitary)
        agent: TD3Agent trained
        episodes: episode test
        max_steps: max step per episode
        render: whether to render GUI (pybullet GUI/on-screen info)
    """

    agent.eval_mode()       
    print("========== START TESTING ==========")

    all_rewards = []

    for ep in range(episodes):
        obs, _ = env.reset()
        ep_reward = 0

        for step in range(max_steps):

            # Lấy action từ agent
            action = agent.get_action(obs, explore=False)   # explore=False khi test
            
            # Env step
            next_obs, reward, terminated, truncated, info = env.step(action)

            ep_reward += reward
            obs = next_obs

            if render:
                pos = info["position"]
                print(f"[EP {ep+1} | Step {step}] Pos: {pos}, Reward: {reward:.3f}")

            # Nếu kết thúc episode
            if terminated or truncated:
                break

        print(f"[Episode {ep+1}] Total Reward = {ep_reward:.2f}")
        all_rewards.append(ep_reward)

    print("=========== TEST FINISHED ===========")
    print(f"Average Reward over {episodes} episodes: {np.mean(all_rewards):.2f}")

    return all_rewards


if __name__ == "__main__":

    DEFAULT_GUI = True
    DEFAULT_RECORD_VIDEO = False
    DEFAULT_OUTPUT_FOLDER = 'results_td3_thrugate'
    if not os.path.exists(DEFAULT_OUTPUT_FOLDER):
        os.makedirs(DEFAULT_OUTPUT_FOLDER)
    DEFAULT_COLAB = False

    DEFAULT_OBS = ObservationType('kin') # 'kin' or 'rgb'
    DEFAULT_ACT = ActionType('rpm') 
    filename = os.path.join(DEFAULT_OUTPUT_FOLDER, 'recording_'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    env = FlyThruGateAvitary(gui=DEFAULT_GUI,
                           obs=DEFAULT_OBS,
                           act=DEFAULT_ACT,
                           record=DEFAULT_RECORD_VIDEO)

    agent = TD3Agent(
        Actor, Critic,
        clip_low=-1.0, clip_high=1.0,
        state_size=12,
        action_size=4,
        gamma=0.98,
        tau=0.01,
        lr=5e-4,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Load checkpoint nếu có
    agent.load_ckpt(
        actor_path="/home/tuan/Desktop/drone_rl_control/log_dir/td3_thrugate/actor/mlp_model_ep5000_actor.pth",
        critic_path1="/home/tuan/Desktop/drone_rl_control/log_dir/td3_thrugate/critics/mlp_model_ep5000_critic_1.pth",
        critic_path2="/home/tuan/Desktop/drone_rl_control/log_dir/td3_thrugate/critics/mlp_model_ep5000_critic_2.pth",
    )

    # Test
    test_agent(env, agent, episodes=10, render=True)
    env.close()