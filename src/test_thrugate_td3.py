import numpy as np
import torch
import argparse

from gym_pybullet_drones.envs.FlyThruGateAvitary import FlyThruGateAvitary
from agents.td3_agent import TD3Agent, Actor, Critic
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
import os
from datetime import datetime
import time
from gym_pybullet_drones.utils.utils import sync

def _reshape_action_for_env(action, action_space):
    """Ensure actions always match the env's expected (NUM_DRONES, 4) shape."""
    action = np.asarray(action, dtype=np.float32)
    if action.ndim == 1:
        action = np.expand_dims(action, axis=0)
    try:
        action = action.reshape(action_space.shape)
    except Exception:
        pass
    return action

def test_agent(args):
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
    DEFAULT_RECORD_VIDEO = True
    DEFAULT_OUTPUT_FOLDER = './results/results_test_td3_thrugate'
    if args.curriculum:
        DEFAULT_OUTPUT_FOLDER = './results/results_test_td3_thrugate_curriculum'
    if not os.path.exists(DEFAULT_OUTPUT_FOLDER):
        os.makedirs(DEFAULT_OUTPUT_FOLDER)
    DEFAULT_COLAB = False

    DEFAULT_OBS = ObservationType('kin') # 'kin' or 'rgb'
    DEFAULT_ACT = ActionType('rpm')
    # filename = os.path.join(DEFAULT_OUTPUT_FOLDER, 'recording_'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    
    print("\nInitializing environment...") 
    env = FlyThruGateAvitary(gui=DEFAULT_GUI,
                           obs=DEFAULT_OBS,
                           act=DEFAULT_ACT,
                           record=DEFAULT_RECORD_VIDEO,
                           output_folder=DEFAULT_OUTPUT_FOLDER,
                           use_curriculum=args.curriculum,
                           curriculum_level=5)

    obs, info = env.reset()
    state_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")

    print("\nInitializing agent...")
    agent = TD3Agent(
        Actor, Critic,
        hidden_dim=256,
        state_size=state_dim,
        action_size=action_dim,
    )

    print("Agent initialized.")
    print("=" * 60)

    
    # Load pretrained model
    checkpoint_path = args.model_path
    if not os.path.exists(checkpoint_path):
        print(f"Error: Model file not found at {checkpoint_path}")
        return
    
    print(f"Loading network from: {checkpoint_path}")
    agent.load(checkpoint_path)
    print("=" * 60)

    agent.eval_mode()

    total_test_episodes = args.episodes

    print(f"\nStarting test for {total_test_episodes} episodes...")
    print("=" * 60)

    test_rewards = []
    test_lengths = []
    success_count = 0
    success_times = []
    center_times = []

    for ep in range(total_test_episodes):
        obs, info = env.reset(seed=42 + ep, options={})
        ep_reward = 0
        ep_len = 0
        start = time.time()
        env.success_passed = False
        center_time = None

        for i in range((env.EPISODE_LEN_SEC+20)*env.CTRL_FREQ):
            action = agent.get_action(obs, explore=False)
            action = _reshape_action_for_env(action, env.action_space)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_len += 1
            if center_time is None and getattr(env, "center_gate_passed", False):
                center_time = ep_len / env.CTRL_FREQ
            env.render()
            sync(i, start, env.CTRL_TIMESTEP)
            if terminated or truncated:
                # Success is when the drone passes the gate center
                if center_time is not None:
                    success_count += 1
                    success_times.append(center_time)
                break
        test_rewards.append(ep_reward)
        test_lengths.append(ep_len)
        center_times.append(center_time)
        center_str = f"{center_time:.2f}s" if center_time is not None else "-"
        print(f"Episode {ep+1}/{total_test_episodes} | Reward {ep_reward:.2f} | Len {ep_len} | Center {center_str} | Success {'✓' if center_time is not None else '✗'}")

    print(f"Avg reward: {np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}")
    print(f"Success rate: {success_count}/{total_test_episodes} ({100*success_count/total_test_episodes:.1f}%)")
    if success_times:
        print(f"Avg center-pass time: {np.mean(success_times):.2f}s ± {np.std(success_times):.2f}s")
        print(f"Best center: {np.min(success_times):.2f}s | Worst center: {np.max(success_times):.2f}s")

    print("========== TESTING COMPLETED ==========")
    env.close()

    # Print summary statistics
    print("\n" + "=" * 88)
    print("Test Summary")
    print("=" * 88)
    print(f"Total Episodes:        {total_test_episodes}")
    print(f"Average Reward:        {np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}")
    print(f"Max Reward:            {np.max(test_rewards):.2f}")
    print(f"Min Reward:            {np.min(test_rewards):.2f}")
    print(f"Average Length:        {np.mean(test_lengths):.2f}")
    print(f"Success Rate:          {success_count}/{total_test_episodes} "
          f"({100 * success_count / total_test_episodes:.1f}%)")
    if success_times:
        print(f"Avg center-pass time:  {np.mean(success_times):.2f}s ± {np.std(success_times):.2f}s")
        print(f"Best/Worst center:     {np.min(success_times):.2f}s / {np.max(success_times):.2f}s")
    print("=" * 88)

    # Save summary
    out_dir = os.path.join(DEFAULT_OUTPUT_FOLDER, "test_summary")
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "test_results_td3.txt")
    with open(summary_path, "w") as f:
        f.write("TD3 FlyThruGateAvitary Test Results\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Episodes: {total_test_episodes}\n")
        f.write(f"Avg reward: {np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}\n")
        f.write(f"Success rate: {success_count}/{total_test_episodes} ({100*success_count/total_test_episodes:.1f}%)\n")
        if success_times:
            f.write(f"Avg center-pass time: {np.mean(success_times):.2f}s ± {np.std(success_times):.2f}s\n")
            f.write(f"Best center: {np.min(success_times):.2f}s | Worst center: {np.max(success_times):.2f}s\n")
        f.write("\nEpisode details:\n")
        for idx, (r, l) in enumerate(zip(test_rewards, test_lengths)):
            f.write(f"Ep {idx+1}: reward={r:.2f}, len={l}")
            if idx < len(center_times) and center_times[idx] is not None:
                f.write(f", center_time={center_times[idx]:.2f}s")
            f.write("\n")
        if success_times:
            f.write("\nCenter pass times (s): " + ", ".join(f"{t:.2f}" for t in success_times) + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test TD3 agent for drone gate navigation')
    parser.add_argument('--model_path', type=str,
                       default='/home/tuan/Desktop/drone_rl_control/log_dir/td3_training_thrugate_curriculum/td3_20251217_161813/td3_model_ep5000.pt',
                       help='Path to the trained model checkpoint')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of test episodes')
    parser.add_argument('--record', type=bool, default=True,
                       help='Record video of the test')
    parser.add_argument('--gui', type=bool, default=True,
                       help='Enable GUI visualization')
    parser.add_argument('--curriculum', type=bool, default=False,
                       help='Use curriculum learning during testing')

    args = parser.parse_args()

    print(f"\nTest Configuration:")
    print(f"  Model Path: {args.model_path}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Record Video: {args.record}")
    print(f"  GUI: {args.gui}")
    print()

    test_agent(args)
    
