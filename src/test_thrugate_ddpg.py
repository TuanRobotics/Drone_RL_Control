#!/usr/bin/env python3

import os
import time
import argparse
import numpy as np

from gym_pybullet_drones.envs.FlyThruGateAvitary import FlyThruGateAvitary
from agents.ddpg import DDPGAgent, Actor, Critic
from gym_pybullet_drones.utils.enums import (DroneModel, Physics, ActionType,
                                             ObservationType)
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
    """Run evaluation episodes for a trained DDPG agent."""
    DEFAULT_GUI = True
    DEFAULT_RECORD_VIDEO = True
    DEFAULT_OUTPUT_FOLDER = './results/results_test_ddpg_thrugate'
    os.makedirs(DEFAULT_OUTPUT_FOLDER, exist_ok=True)

    DEFAULT_OBS = ObservationType('kin')
    DEFAULT_ACT = ActionType('rpm')

    print("\nInitializing environment...")
    env = FlyThruGateAvitary(drone_model=DroneModel.CF2X,
                             initial_xyzs=np.array([[0, 0, 0.5]]),
                             physics=Physics.PYB,
                             pyb_freq=240,
                             ctrl_freq=30,
                             gui=DEFAULT_GUI,
                             obs=DEFAULT_OBS,
                             act=DEFAULT_ACT,
                             record=DEFAULT_RECORD_VIDEO,
                             output_folder=DEFAULT_OUTPUT_FOLDER)

    obs, _ = env.reset()
    state_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    clip_low = float(np.min(env.action_space.low))
    clip_high = float(np.max(env.action_space.high))

    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")

    print("\nInitializing agent...")
    agent = DDPGAgent(
        Actor,
        Critic,
        hidden_dim=256,
        clip_low=clip_low,
        clip_high=clip_high,
        state_size=state_dim,
        action_size=action_dim,
        exploration_noise=0.1,
        noise_type='ou'
    )
    agent.eval_mode()

    checkpoint_path = args.model_path
    if not os.path.exists(checkpoint_path):
        print(f"Error: Model file not found at {checkpoint_path}")
        return

    print(f"Loading network from: {checkpoint_path}")
    agent.load(checkpoint_path)
    print("=" * 60)

    total_test_episodes = args.episodes

    print(f"\nStarting test for {total_test_episodes} episodes...")
    print("=" * 60)

    test_rewards = []
    test_lengths = []
    success_count = 0
    success_times = []
    success_flags = []
    success_durations = []

    for ep in range(total_test_episodes):
        obs, info = env.reset(seed=42 + ep, options={})
        agent.reset_noise()
        env.success_passed = False
        ep_reward = 0
        ep_len = 0
        start = time.time()
        success = False

        for i in range((env.EPISODE_LEN_SEC + 20) * env.CTRL_FREQ):
            action = agent.get_action(obs, explore=False)
            action = _reshape_action_for_env(action, env.action_space)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_len += 1

            if DEFAULT_GUI:
                env.render()
                sync(i, start, env.CTRL_TIMESTEP)

            if terminated or truncated:
                success = getattr(env, "success_passed", False)
                if success:
                    success_count += 1
                    duration = ep_len / env.CTRL_FREQ
                    success_times.append(duration)
                break

        test_rewards.append(ep_reward)
        test_lengths.append(ep_len)
        success_flags.append(success)
        success_durations.append(ep_len / env.CTRL_FREQ if success else None)
        print(f"Episode {ep+1}/{total_test_episodes} | Reward {ep_reward:.2f} | "
              f"Len {ep_len} | Success {'✓' if success else '✗'}")

    print(f"Avg reward: {np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}")
    print(f"Success rate: {success_count}/{total_test_episodes} ({100*success_count/total_test_episodes:.1f}%)")
    if success_times:
        print(f"Avg success time: {np.mean(success_times):.2f}s ± {np.std(success_times):.2f}s")
        print(f"Best: {np.min(success_times):.2f}s | Worst: {np.max(success_times):.2f}s")

    print("========== TESTING COMPLETED ==========")
    env.close()

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
    print("=" * 88)

    out_dir = os.path.join(DEFAULT_OUTPUT_FOLDER, "test_summary")
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "test_results_ddpg.txt")
    with open(summary_path, "w") as f:
        f.write("DDPG FlyThruGateAvitary Test Results\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Episodes: {total_test_episodes}\n")
        f.write(f"Avg reward: {np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}\n")
        f.write(f"Success rate: {success_count}/{total_test_episodes} ({100*success_count/total_test_episodes:.1f}%)\n")
        if success_times:
            f.write(f"Avg success time: {np.mean(success_times):.2f}s ± {np.std(success_times):.2f}s\n")
            f.write(f"Best: {np.min(success_times):.2f}s | Worst: {np.max(success_times):.2f}s\n")
        f.write("\nEpisode details:\n")
        for idx, (r, l, success_flag, duration) in enumerate(
                zip(test_rewards, test_lengths, success_flags,
                    success_durations)):
            f.write(f"Ep {idx+1}: reward={r:.2f}, len={l}")
            if success_flag and duration is not None:
                f.write(f", success_time={duration:.2f}s")
            f.write("\n")
        if success_times:
            f.write("\nSuccess times (s): " + ", ".join(f"{t:.2f}" for t in success_times) + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test DDPG agent for drone gate navigation')
    parser.add_argument('--model_path', type=str,
                        default='/home/tuan/Desktop/drone_rl_control/log_dir/ddpg_training_thrugate/ddpg_20251216_221229/ddpg_model_final.pt',
                        help='Path to the trained model checkpoint')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of test episodes')
    parser.add_argument('--record', type=bool, default=True,
                        help='Record video of the test')
    parser.add_argument('--gui', type=bool, default=True,
                        help='Enable GUI visualization')

    args = parser.parse_args()

    print(f"\nTest Configuration:")
    print(f"  Model Path: {args.model_path}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Record Video: {args.record}")
    print(f"  GUI: {args.gui}")
    print()

    test_agent(args)
