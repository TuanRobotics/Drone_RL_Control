import numpy as np
import torch
from gym_pybullet_drones.envs.FlyThruGateAvitary import FlyThruGateAvitary
from TD3.td3 import TD3Agent, Actor, Critic
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
import os
from datetime import datetime
import time
from gym_pybullet_drones.utils.utils import sync


def test_agent():
    """
    Test TD3 agent trong môi trường FlyThruGateAvitary.
    
    Args:
        env: enviroment  (FlyThruGateAvitary)
        agent: TD3Agent trained
        episodes: episode test
        max_steps: max step per episode
        render: whether to render GUI (pybullet GUI/on-screen info)
    """

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
        actor_path="log_dir/td3_thrugate/actor/mlp_model_ep10000_actor.pth",
        critic_path1="log_dir/td3_thrugate/critics/mlp_model_ep10000_critic_1.pth",
        critic_path2="log_dir/td3_thrugate/critics/mlp_model_ep10000_critic_2.pth",
    )

    agent.eval_mode()       
    print("========== START TESTING ==========")

    obs, info = env.reset(seed=42, options={})
    ep_reward = 0
    start_time = datetime.now().replace(microsecond=0)
    start = time.time()

    for i in range((env.EPISODE_LEN_SEC+20)*env.CTRL_FREQ):
        action = agent.get_action(obs, explore=False)
        action = np.expand_dims(action, axis=0)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward
        env.render()
        sync(i, start, env.CTRL_TIMESTEP)
        if terminated:
            break
    print(f"Test Episode Reward: {ep_reward}")

    test_running_reward +=  ep_reward
    print('Episode: {} \t\t Reward: {}'.format(0, round(ep_reward, 2)))
    ep_reward = 0

    print("========== TESTING COMPLETED ==========")
    env.close()

if __name__ == "__main__":
    # Test
    test_agent()