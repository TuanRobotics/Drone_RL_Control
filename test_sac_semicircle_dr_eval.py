#!/usr/bin/env python3
"""
Evaluate a SAC policy trained with train_sac_semicircle_dr.py.

Loads a checkpoint, runs a few episodes (with or without domain randomization),
and prints rewards plus gates passed. Useful to sanity-check training output.
"""

import argparse
import os
import time
import numpy as np

from SAC.sac_agent import SACAgent
from randomized_semicircle_env import RandomizedSemiCircleAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

# Default randomization matches train_sac_semicircle_dr.py
DEFAULT_RAND_CFG = {
    "mass_scale_range": (0.9, 1.1),
    "inertia_scale_range": (0.9, 1.1),
    "kf_scale_range": (0.9, 1.1),
    "km_scale_range": (0.9, 1.1),
    "drag_scale_range": (0.8, 1.2),
    "wind_mag_range": (0.0, 1.0),
}

# Disable DR by collapsing ranges to 1.0 and wind to 0
NO_RAND_CFG = {
    "mass_scale_range": (1.0, 1.0),
    "inertia_scale_range": (1.0, 1.0),
    "kf_scale_range": (1.0, 1.0),
    "km_scale_range": (1.0, 1.0),
    "drag_scale_range": (1.0, 1.0),
    "wind_mag_range": (0.0, 0.0),
}

# Track/layout reset options
DEFAULT_RESET_OPTS = {
    "random_track_layout": True,
    "radius_range": (3.0, 6.0),
    "center_jitter": 1.0,
    "yaw_jitter": 0.4,
}


def load_agent(model_path: str, env_sample) -> SACAgent:
    state_dim = env_sample.observation_space.shape[0]
    action_dim = int(np.prod(env_sample.action_space.shape))
    agent = SACAgent(state_dim=state_dim, action_dim=action_dim, hidden_dim=256,
                     lr=3e-4, gamma=0.99, tau=0.005)
    agent.load(model_path, load_optimizer=False)
    return agent


def run_eval(
    model_path: str,
    episodes: int = 5,
    gui: bool = False,
    record: bool = False,
    spawn_mode: str = "default",
    sleep: float = 0.05,
    explore: bool = False,
    random_actions: bool = False,
    use_domain_rand: bool = True,
    reset_opts: dict | None = None,
    seed: int | None = None,
    require_all_gates: bool = False,
):
    reset_opts = reset_opts or DEFAULT_RESET_OPTS
    rand_cfg = DEFAULT_RAND_CFG if use_domain_rand else NO_RAND_CFG

    env = RandomizedSemiCircleAviary(
        gui=gui,
        record=record,
        obs=ObservationType.KIN,
        act=ActionType.RPM,
        rand_cfg=rand_cfg,
    )
    if seed is not None:
        env._rng = np.random.default_rng(seed)  # keep spawn deterministic if desired

    agent = load_agent(model_path, env)

    ep_rewards = []
    ep_success = 0
    for ep in range(episodes):
        obs, info = env.reset(options={"spawn": spawn_mode, **reset_opts})
        done = False
        total_reward = 0.0
        steps = 0
        last_info = {}
        while not done:
            if random_actions:
                action = env.action_space.sample()
            else:
                action = agent.select_action(obs, evaluate=not explore)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            total_reward += reward
            steps += 1
            last_info = info
            if sleep > 0.0:
                time.sleep(sleep)
        ep_rewards.append(total_reward)
        passed = last_info.get("passing_flag", [])
        gate_total = len(passed) if passed is not None else 0
        gate_count = int(np.sum(passed)) if gate_total > 0 else 0
        success_all_gates = gate_total > 0 and gate_count == gate_total
        ep_success += int(success_all_gates)
        print(
            f"Episode {ep+1}: reward={total_reward:.2f}, steps={steps}, "
            f"gates_passed={gate_count}, spawn={spawn_mode}, "
            f"domain_rand={'on' if use_domain_rand else 'off'}, "
            f"all_gates={'yes' if success_all_gates else 'no'}"
        )
        if require_all_gates and not success_all_gates:
            print("  -> Missed at least one gate.")

    print(f"\nAvg reward over {episodes} eps: {np.mean(ep_rewards):.2f}")
    print(f"All-gates success: {ep_success}/{episodes}")
    if require_all_gates and ep_success < episodes:
        raise SystemExit("Some episodes missed gates (require_all_gates enabled).")
    env.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        default="log_dir/results_sac_semicircle_dr/sac_semicircle_dr_final.pt",
                        help="Path to saved SAC model checkpoint")
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--gui", action="store_true", help="Enable PyBullet GUI")
    parser.add_argument("--no-gui", action="store_true", help="Force disable GUI")
    parser.add_argument("--record", action="store_true", help="Record video (offscreen when GUI is False)")
    parser.add_argument("--spawn", type=str, default="default",
                        choices=["default", "random_track", "success_replay"],
                        help="Spawn strategy for evaluation")
    parser.add_argument("--sleep", type=float, default=0.05, help="Seconds to sleep each step for visualization")
    parser.add_argument("--explore", action="store_true",
                        help="Use stochastic actions (sample) instead of deterministic mean")
    parser.add_argument("--random-actions", action="store_true",
                        help="Ignore policy and sample random actions (env sanity check)")
    parser.add_argument("--no-dr", action="store_true",
                        help="Disable domain randomization during eval (use clean dynamics)")
    parser.add_argument("--seed", type=int, default=None, help="Seed for env RNG")
    parser.add_argument("--require-all-gates", action="store_true",
                        help="Highlight failures if any gate is missed")
    return parser.parse_args()


def main():
    args = parse_args()

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
        use_domain_rand=False,
        reset_opts=DEFAULT_RESET_OPTS,
        seed=args.seed,
        require_all_gates=args.require_all_gates,
    )


if __name__ == "__main__":
    main()
