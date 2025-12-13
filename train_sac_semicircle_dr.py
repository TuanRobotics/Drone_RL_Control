"""
SAC training on SemiCircleRacingAviary with domain randomization.

- Uses RandomizedSemiCircleAviary (mass/inertia/motor/drag/wind jitter).
- Parallel sampling via SyncVectorEnv.
- Curriculum: start with random_track, switch to success_replay later.
- Logs rewards, gate counts, losses; saves checkpoints and plots.
"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from gymnasium.vector import SyncVectorEnv

from agents.sac_agent import SACAgent, ReplayBuffer
from randomized_semicircle_env import make_randomized_env
from gym_pybullet_drones.utils.enums import ObservationType, ActionType


def reset_envs(vec_env, spawn_mode: str, reset_opts: dict | None = None):
    options = {"spawn": spawn_mode}
    if reset_opts:
        options.update(reset_opts)
    return vec_env.reset(options=options)


def train(
    num_envs: int = 8,
    total_steps: int = 1_000_000,
    batch_size: int = 256,
    start_update: int = 10_000,
    update_every: int = 50,
    updates_per_iter: int = 2,
    replay_capacity: int = 1_000_000,
    log_interval: int = 1_000,
    save_dir: str = "log_dir/results_sac_semicircle_dr",
    success_replay_start: int = 500_000,
    save_every_episodes: int = 100_000,
    rand_cfg: dict | None = None,
    reset_opts: dict | None = None,
):
    """
    rand_cfg: domain randomization ranges passed to env (see randomized_semicircle_env.py)
    reset_opts: track randomization options merged into reset options
    """
    os.makedirs(save_dir, exist_ok=True)

    rand_cfg = rand_cfg or {
        "mass_scale_range": (0.9, 1.1),
        "inertia_scale_range": (0.9, 1.1),
        "kf_scale_range": (0.9, 1.1),
        "km_scale_range": (0.9, 1.1),
        "drag_scale_range": (0.8, 1.2),
        "wind_mag_range": (0.0, 1.0),
    }
    reset_opts = reset_opts or {
        "random_track_layout": True,
        "radius_range": (3.0, 6.0),
        "center_jitter": 1.0,
        "yaw_jitter": 0.4,
    }

    vec_env = SyncVectorEnv([make_randomized_env(seed=i, rand_cfg=rand_cfg) for i in range(num_envs)])
    spawn_mode = "random_track"
    obs, info = reset_envs(vec_env, spawn_mode, reset_opts)

    # Dimensions
    state_dim = obs.shape[1] if obs.ndim == 2 else obs.reshape(num_envs, -1).shape[1]
    action_dim = vec_env.single_action_space.shape[1]

    agent = SACAgent(state_dim=state_dim, action_dim=action_dim, hidden_dim=256, lr=3e-4, gamma=0.99, tau=0.005)
    buffer = ReplayBuffer(capacity=replay_capacity)

    episode_rewards = np.zeros(num_envs, dtype=np.float32)
    reward_history = []
    gate_history = []
    actor_loss_hist = []
    critic_loss_hist = []

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
        if step >= success_replay_start:
            spawn_mode = "success_replay"

        actions = np.zeros((num_envs, action_dim), dtype=np.float32)
        for i in range(num_envs):
            actions[i] = agent.select_action(obs[i], evaluate=False)

        next_obs, rewards, term, trunc, infos = vec_env.step(actions)
        done = np.logical_or(term, trunc)

        for i in range(num_envs):
            buffer.push(obs[i], actions[i], rewards[i], next_obs[i], float(done[i]))
            episode_rewards[i] += rewards[i]

            if done[i]:
                reward_history.append(episode_rewards[i])
                gates = 0
                if isinstance(infos, dict) and "passing_flag" in infos:
                    pf = infos["passing_flag"][i]
                    gates = int(np.sum(pf)) if pf is not None else 0
                elif isinstance(infos, (list, tuple)) and i < len(infos) and isinstance(infos[i], dict):
                    pf = infos[i].get("passing_flag")
                    gates = int(np.sum(pf)) if pf is not None else 0
                gate_history.append(gates)
                episode_rewards[i] = 0.0

                total_episodes = len(reward_history)
                if save_every_episodes and total_episodes % save_every_episodes == 0:
                    ckpt_path = os.path.join(save_dir, f"sac_dr_ep{total_episodes}.pt")
                    agent.save(ckpt_path)
                    _save_plot(reward_history, "Episode reward", f"rewards_ep{total_episodes}.png", xlabel="Episode")
                    _save_plot(actor_loss_hist, "Actor loss", f"actor_loss_ep{total_episodes}.png")
                    _save_plot(critic_loss_hist, "Critic loss", f"critic_loss_ep{total_episodes}.png")
                    print(f"Checkpoint saved at episode {total_episodes}: {ckpt_path}")

                obsi, infoi = vec_env.envs[i].reset(options={"spawn": spawn_mode, **reset_opts})
                next_obs[i] = obsi

        obs = next_obs

        if len(buffer) > start_update and step % update_every == 0:
            for _ in range(updates_per_iter):
                q1_loss, q2_loss, policy_loss = agent.update(buffer, batch_size=batch_size)
                critic_loss_hist.append(0.5 * (q1_loss + q2_loss))
                actor_loss_hist.append(policy_loss)

        if step % log_interval == 0:
            elapsed = time.time() - start_time
            mean_ep_rew = np.mean(reward_history[-100:]) if reward_history else 0.0
            mean_gates = np.mean(gate_history[-100:]) if gate_history else 0.0
            print(
                f"Step {step}/{total_steps} | Buffer {len(buffer)} | "
                f"MeanEpRew(100) {mean_ep_rew:.2f} | MeanGates(100) {mean_gates:.2f} | "
                f"FPS {step/elapsed:.1f} | spawn {spawn_mode}"
            )

    final_model = os.path.join(save_dir, "sac_semicircle_dr_final.pt")
    agent.save(final_model)
    print(f"Model saved to {final_model}")

    _save_plot(reward_history, "Episode reward", "rewards.png", xlabel="Episode")
    _save_plot(actor_loss_hist, "Actor loss", "actor_loss.png")
    _save_plot(critic_loss_hist, "Critic loss", "critic_loss.png")
    vec_env.close()


if __name__ == "__main__":
    train()
