"""
Evaluate a trained SAC policy on SemiCircleRacingAviary.

Loads the saved model (state_dict) from train_sac_semicircle_full.py, runs a few
episodes, prints rewards and gates passed. Optionally records video (GUI must be
False for offscreen recording in this codebase).
"""

import argparse
import os
import time
import numpy as np

from SAC.sac_agent import SACAgent
from gym_pybullet_drones.envs import SemiCircleRacingAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType


def load_agent(model_path: str, env_sample) -> SACAgent:
    state_dim = env_sample.observation_space.shape[0]
    action_dim = env_sample.action_space.shape[0]
    agent = SACAgent(state_dim=state_dim, action_dim=action_dim, hidden_dim=128,
                    lr=3e-4, gamma=0.99, tau=0.005)
    agent.load(model_path)
    return agent


def run_eval(
    model_path: str,
    episodes: int = 5,
    gui: bool = False,
    record: bool = False,
    spawn_mode: str = "random_track",
    sleep: float = 0.02,
    explore: bool = False,
    random_actions: bool = False,
):
    env = SemiCircleRacingAviary(
        gui=gui,
        record=record,
        obs=ObservationType.KIN,
        act=ActionType.RPM,
    )
    agent = load_agent(model_path, env)

    ep_rewards = []
    for ep in range(episodes):
        obs, info = env.reset(options={"spawn": spawn_mode})
        done = False
        total_reward = 0.0
        steps = 0
        while not done:
            if random_actions:
                action = env.action_space.sample()
            else:
                action = agent.select_action(obs, evaluate=not explore)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            total_reward += reward
            steps += 1
            if sleep > 0.0:
                time.sleep(sleep)  # slow down so the trajectory is visible
        ep_rewards.append(total_reward)
        passed = info.get("passing_flag", [])
        print(f"Episode {ep+1}: reward={total_reward:.2f}, steps={steps}, gates_passed={sum(passed) if passed else 'n/a'}")

    print(f"\nAvg reward over {episodes} eps: {np.mean(ep_rewards):.2f}")
    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="results_sac_semicircle/sac_semicircle.pt", help="Path to saved model")
    parser.add_argument("--episodes", type=int, default=5, help="Number of eval episodes")
    parser.add_argument("--gui", action="store_true", help="Enable PyBullet GUI (default: on)")
    parser.add_argument("--no-gui", action="store_true", help="Disable GUI")
    parser.add_argument("--record", action="store_true", help="Enable video recording (GUI can be False for offscreen)")
    parser.add_argument("--spawn", type=str, default="random_track", choices=["random_track", "success_replay", "default"], help="Spawn strategy for eval")
    parser.add_argument("--sleep", type=float, default=0.02, help="Seconds to sleep each step for visualization (set 0 to run fast)")
    parser.add_argument("--explore", action="store_true", help="Use stochastic actions (sample) instead of deterministic mean")
    parser.add_argument("--random-actions", action="store_true", help="Ignore policy and use random actions (sanity check env)")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found at {args.model}")

    gui = True
    if args.no_gui:
        gui = False
    elif args.gui:
        gui = True

    run_eval(
        model_path=args.model,
        episodes=args.episodes,
        gui=gui,
        record=args.record,
        spawn_mode=args.spawn,
        sleep=args.sleep,
        explore=args.explore,
        random_actions=args.random_actions,
    )


if __name__ == "__main__":
    main()
