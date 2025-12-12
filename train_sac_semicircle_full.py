"""
Full SAC training on SemiCircleRacingAviary with parallel sampling and
distributed initialization.

- Uses SyncVectorEnv to avoid shared-memory permission issues.
- Curriculum spawn: start at easy gate-0 spawn, then random_track, then success_replay.
- Logs episode rewards and losses; saves plots and model checkpoints.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from gymnasium.vector import SyncVectorEnv

from SAC.sac_agent import SACAgent, ReplayBuffer, device
from gym_pybullet_drones.envs import SemiCircleRacingAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType


def make_env(seed: int, spawn_mode: str = "random_track"):
    rng = np.random.default_rng(seed)

    def _init():
        env = SemiCircleRacingAviary(
            gui=False,
            record=False,
            obs=ObservationType.KIN,
            act=ActionType.RPM,
        )
        env._rng = rng
        env._spawn_mode = spawn_mode
        return env

    return _init


def reset_envs(vec_env, spawn_mode: str):
    options = {"spawn": spawn_mode}
    return vec_env.reset(options=options)


def train(
    num_envs: int = 4,
    total_steps: int = 1200000,
    batch_size: int = 128,
    start_update: int = 5_000, # start updates after this many steps
    update_every: int = 50,
    updates_per_iter: int = 1,
    replay_capacity: int = 500_000,
    log_interval: int = 1_000,
    save_dir: str = "log_dir/results_sac_semicircle_training/step02",
    success_replay_start: int = 300_000, # switch to success_replay after this many steps
    default_phase_steps: int = 100_000,   # stay at easy start (gate 0) for first N steps
    save_every_episodes: int = 50_000,
    resume_path: str = "log_dir/sac_thrugate_20251208_122448/sac_model_final.pt", # optional path to resume from
    resume_if_exists: bool = False, # if True, try to auto-resume from final checkpoint in save_dir
):
    os.makedirs(save_dir, exist_ok=True)

    vec_env = SyncVectorEnv([make_env(i) for i in range(num_envs)])
    spawn_mode = "default"
    obs, info = reset_envs(vec_env, spawn_mode)

    # Dimensions
    state_dim = obs.shape[1] if obs.ndim == 2 else obs.reshape(num_envs, -1).shape[1] # 36
    action_dim = vec_env.single_action_space.shape[1] # 4

    # agent = SACAgent(state_dim=state_dim, action_dim=action_dim, hidden_dim=256,
    #                 lr=3e-4, gamma=0.99, tau=0.005)
    agent = SACAgent(state_dim, action_dim, hidden_dim=256,
                    lr=3e-4, gamma=0.99, tau=0.005)
    buffer = ReplayBuffer(capacity=replay_capacity)

    # Resume from a saved checkpoint if requested
    ckpt_to_load = resume_path
    # if ckpt_to_load is None and resume_if_exists:
    #     ckpt_to_load = os.path.join(save_dir, "sac_semicircle.pt")
    if ckpt_to_load:
        if os.path.isfile(ckpt_to_load):
            agent.load(ckpt_to_load, load_optimizer=True)
            print(f"Resumed training from checkpoint: {ckpt_to_load}")
        else:
            print(f"Resume requested but checkpoint not found at {ckpt_to_load}, starting fresh.")

    episode_rewards = np.zeros(num_envs, dtype=np.float32)
    episode_lengths = np.zeros(num_envs, dtype=np.int32)
    reward_history = []
    actor_loss_hist = []
    critic_loss_hist = []
    gate_history = []

    def _save_plot(values, ylabel, filename, xlabel="Update step"):
        if not values:
            return
        plt.figure()
        plt.plot(values)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, filename))
        plt.close()

    start_time = time.time()

    for step in range(1, total_steps + 1):
        # Curriculum: default (easy) -> random_track -> success_replay
        if step >= success_replay_start:
            if spawn_mode != "success_replay":
                print(f"[Curriculum] Switching spawn -> success_replay at step {step}")
            spawn_mode = "success_replay"
        elif step >= default_phase_steps:
            if spawn_mode != "random_track":
                print(f"[Curriculum] Switching spawn -> random_track at step {step}")
            spawn_mode = "random_track"

        # Select actions
        actions = np.zeros((num_envs, action_dim), dtype=np.float32)
        for i in range(num_envs):
            actions[i] = agent.select_action(obs[i], evaluate=False)

        # Step vector env
        next_obs, rewards, term, trunc, infos = vec_env.step(actions)
        done = np.logical_or(term, trunc)

        # Store transitions
        for i in range(num_envs):
            buffer.push(
                obs[i],
                actions[i],
                rewards[i],
                next_obs[i],
                float(done[i]),
            )
            episode_rewards[i] += rewards[i]
            episode_lengths[i] += 1

            if done[i]:
                reward_history.append(episode_rewards[i])
                # gates passed this episode
                gates = 0
                if isinstance(infos, dict) and "passing_flag" in infos:
                    pf = infos["passing_flag"][i]
                    gates = int(np.sum(pf)) if pf is not None else 0
                elif isinstance(infos, (list, tuple)) and i < len(infos) and isinstance(infos[i], dict):
                    pf = infos[i].get("passing_flag")
                    gates = int(np.sum(pf)) if pf is not None else 0
                gate_history.append(gates)
                episode_rewards[i] = 0.0
                episode_lengths[i] = 0

                total_episodes = len(reward_history)
                if save_every_episodes and total_episodes % save_every_episodes == 0:
                    ckpt_path = os.path.join(save_dir, f"sac_semicircle_ep{total_episodes}.pt")
                    agent.save(ckpt_path)
                    print(f"Checkpoint saved at episode {total_episodes}: {ckpt_path}")

        # Reset finished envs with chosen spawn mode (SyncVectorEnv has no reset_done)
        if np.any(done):
            for i, d in enumerate(done):
                if d:
                    obsi, infoi = vec_env.envs[i].reset(options={"spawn": spawn_mode})
                    obs[i] = obsi
                else:
                    obs[i] = next_obs[i]
        else:
            obs = next_obs

        # Updates
        if len(buffer) > start_update and step % update_every == 0:
            for _ in range(updates_per_iter):
                q1_loss, q2_loss, policy_loss = agent.update(buffer, batch_size=batch_size)
                critic_loss_hist.append(0.5 * (q1_loss + q2_loss))
                actor_loss_hist.append(policy_loss)

        # Logging
        if step % log_interval == 0:
            elapsed = time.time() - start_time
            mean_ep_rew = np.mean(reward_history[-100:]) if reward_history else 0.0
            mean_gates = np.mean(gate_history[-100:]) if gate_history else 0.0
            print(
                f"Step {step}/{total_steps} | Buffer {len(buffer)} | "
                f"MeanEpRew(100) {mean_ep_rew:.2f} | MeanGates(100) {mean_gates:.2f} | "
                f"FPS {step/elapsed:.1f} | spawn {spawn_mode}"
            )

    # Save model
    model_path = os.path.join(save_dir, "sac_semicircle.pt")
    agent.save(model_path)
    print(f"Model saved to {model_path}")

    # Plots
    _save_plot(reward_history, "Episode reward", "rewards.png", xlabel="Episode")
    _save_plot(actor_loss_hist, "Actor loss", "actor_loss.png")
    _save_plot(critic_loss_hist, "Critic loss", "critic_loss.png")
    vec_env.close()


if __name__ == "__main__":
   train(
    num_envs=16,
    total_steps=1200000,
    batch_size=256,        # tăng batch
    start_update=10_000,   # học sớm hơn chút
    update_every=50,       # update thường xuyên hơn
    updates_per_iter=2,    # mỗi lần update 2 gradient step
    replay_capacity=500_000,
    log_interval=1_000,
    success_replay_start=500_000,
    save_every_episodes=50_000,
    resume_path="", # để trống để train lại từ đầu
)
