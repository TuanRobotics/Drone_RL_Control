#!/usr/bin/env python3
import copy
import random
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym  # hoặc gymnasium, tùy bạn dùng cái nào


# ============================================================
# Utils
# ============================================================

def fanin_init(layer):
    if isinstance(layer, nn.Linear):
        lim = 1. / np.sqrt(layer.weight.data.size(0))
        nn.init.uniform_(layer.weight.data, -lim, lim)
        nn.init.uniform_(layer.bias.data, -lim, lim)


# ============================================================
# Networks
# ============================================================

class Actor(nn.Module):
    """
    Deterministic policy: state -> action in [-1, 1]^action_dim
    Scale về range của env sẽ làm ở TD3Agent.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.apply(fanin_init)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # output trong [-1, 1]
        return torch.tanh(self.net(state))


class Critic(nn.Module):
    """
    Q-network: (state, action) -> Q-value
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.apply(fanin_init)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


# ============================================================
# Replay Buffer
# ============================================================

class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, capacity: int = 1_000_00):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        idx = self.ptr
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.next_states[idxs],
            self.dones[idxs],
        )

    def __len__(self):
        return self.size


# ============================================================
# TD3 Agent
# ============================================================

@dataclass
class TD3Config:
    gamma: float = 0.99
    tau: float = 0.005
    policy_noise: float = 0.2       # noise cho target policy (smoothing)
    noise_clip: float = 0.5
    expl_noise: float = 0.1         # exploration noise khi tương tác env
    policy_delay: int = 2
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    batch_size: int = 256
    buffer_capacity: int = 500_000
    hidden_dim: int = 256


class TD3Agent:
    def __init__(self, state_dim: int, action_space, device: str = "cpu",
                 config: TD3Config = TD3Config()):
        self.device = torch.device(device)
        self.cfg = config

        self.state_dim = state_dim
        self.action_dim = action_space.shape[0]

        # Action range
        self.action_low = torch.as_tensor(action_space.low, dtype=torch.float32, device=self.device)
        self.action_high = torch.as_tensor(action_space.high, dtype=torch.float32, device=self.device)
        self.action_scale = (self.action_high - self.action_low) / 2.0
        self.action_bias = (self.action_high + self.action_low) / 2.0

        # Networks
        self.actor = Actor(state_dim, self.action_dim, self.cfg.hidden_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)

        self.critic1 = Critic(state_dim, self.action_dim, self.cfg.hidden_dim).to(self.device)
        self.critic2 = Critic(state_dim, self.action_dim, self.cfg.hidden_dim).to(self.device)
        self.critic1_target = copy.deepcopy(self.critic1).to(self.device)
        self.critic2_target = copy.deepcopy(self.critic2).to(self.device)

        # Optimizers
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=self.cfg.actor_lr)
        self.critic1_opt = optim.Adam(self.critic1.parameters(), lr=self.cfg.critic_lr)
        self.critic2_opt = optim.Adam(self.critic2.parameters(), lr=self.cfg.critic_lr)

        # Replay buffer
        self.replay = ReplayBuffer(state_dim, self.action_dim, capacity=self.cfg.buffer_capacity)

        self.total_it = 0

    # -----------------------------
    # Utilities
    # -----------------------------
    def _to_tensor(self, x):
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    def _scale_action(self, action_normalized: torch.Tensor) -> torch.Tensor:
        # action_normalized trong [-1,1] -> scale về range env
        return action_normalized * self.action_scale + self.action_bias

    # -----------------------------
    # Interaction
    # -----------------------------
    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> np.ndarray:
        """
        state: np.array (state_dim,)
        return: np.array (action_dim,) trong range env
        """
        state_t = self._to_tensor(state).unsqueeze(0)  # (1, state_dim)
        with torch.no_grad():
            action_n = self.actor(state_t)  # [-1,1]
            action = self._scale_action(action_n).cpu().numpy().flatten()

        if not eval_mode:
            noise = np.random.normal(0, self.cfg.expl_noise, size=self.action_dim)
            action = action + noise
        # clip theo env
        action = np.clip(action, self.action_low.cpu().numpy(), self.action_high.cpu().numpy())
        return action

    def store_transition(self, state, action, reward, next_state, done):
        self.replay.add(state, action, reward, next_state, done)

    # -----------------------------
    # Training step
    # -----------------------------
    def train_step(self):
        if len(self.replay) < self.cfg.batch_size:
            return

        self.total_it += 1

        states, actions, rewards, next_states, dones = self.replay.sample(self.cfg.batch_size)

        states = self._to_tensor(states)
        actions = self._to_tensor(actions)
        rewards = self._to_tensor(rewards)
        next_states = self._to_tensor(next_states)
        dones = self._to_tensor(dones)

        # ---------- Critic update ----------
        with torch.no_grad():
            # Target policy smoothing
            next_action_n = self.actor_target(next_states)  # [-1,1]
            next_action = self._scale_action(next_action_n)

            noise = (torch.randn_like(next_action) * self.cfg.policy_noise).clamp(
                -self.cfg.noise_clip, self.cfg.noise_clip
            )
            next_action = (next_action + noise).clamp(self.action_low, self.action_high)

            # Clipped double Q
            target_Q1 = self.critic1_target(next_states, next_action)
            target_Q2 = self.critic2_target(next_states, next_action)
            target_Q = torch.min(target_Q1, target_Q2)

            target = rewards + (1.0 - dones) * self.cfg.gamma * target_Q

        current_Q1 = self.critic1(states, actions)
        current_Q2 = self.critic2(states, actions)

        critic1_loss = nn.MSELoss()(current_Q1, target)
        critic2_loss = nn.MSELoss()(current_Q2, target)

        self.critic1_opt.zero_grad()
        critic1_loss.backward()
        self.critic1_opt.step()

        self.critic2_opt.zero_grad()
        critic2_loss.backward()
        self.critic2_opt.step()

        # ---------- Delayed actor & target update ----------
        if self.total_it % self.cfg.policy_delay == 0:
            # Actor loss: maximize Q1(s, π(s)) -> minimize -Q1
            pi_n = self.actor(states)
            pi = self._scale_action(pi_n)
            actor_loss = -self.critic1(states, pi).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            # Soft update targets
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic1, self.critic1_target)
            self._soft_update(self.critic2, self.critic2_target)

    def _soft_update(self, net: nn.Module, target_net: nn.Module):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(self.cfg.tau * param.data + (1 - self.cfg.tau) * target_param.data)


# ============================================================
# Training loop example cho DRONE
# ============================================================
from gym_pybullet_drones.envs.FlyThruGateAvitary import FlyThruGateAvitary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
def make_env() -> gym.Env:
    """
    Chỗ này bạn sửa theo env của bạn.
    Ví dụ với gym-pybullet-drones (single agent):

    from gym_pybullet_drones.envs.single_agent_rl import HoverAviary
    from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

    env = HoverAviary(
        drone_model=DroneModel.CF2X,
        obs=ObservationType.KIN,
        act=ActionType.RPM,
        gui=False,
        record=False
    )
    return env
    """
    # Demo: dùng một env continuous bất kỳ (thay bằng FlyThruGateAviary của bạn)
    # Ví dụ: LunarLanderContinuous-v2
    # env = gym.make("LunarLanderContinuous-v2")
    # return env
    DEFAULT_GUI = True
    DEFAULT_RECORD_VIDEO = False
    DEFAULT_OUTPUT_FOLDER = 'results'
    DEFAULT_COLAB = False

    DEFAULT_OBS = ObservationType('kin') # 'kin' or 'rgb' , KIN - (1,12) , RGB - (3,64,64)
    DEFAULT_ACT = ActionType('rpm') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'


    env = FlyThruGateAvitary(obs=DEFAULT_OBS, act=DEFAULT_ACT)
    return env


def train_td3_drone(
    num_episodes: int = 500,
    max_steps: int = 1000,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    env = make_env()
    state_dim = env.observation_space.shape[0]
    action_space = env.action_space

    agent = TD3Agent(state_dim, action_space, device=device)

    for ep in range(num_episodes):
        # Gym < 0.26: state = env.reset()
        # Gymnasium / mới: state, info = env.reset()
        out = env.reset()
        if isinstance(out, tuple):
            state, _ = out
        else:
            state = out

        ep_reward = 0.0

        for t in range(max_steps):
            action = agent.select_action(state, eval_mode=False)
            step_out = env.step(action)

            # Hỗ trợ cả gym cũ (4 giá trị) lẫn mới (5 giá trị)
            if len(step_out) == 4:
                next_state, reward, done, info = step_out
                truncated = False
            else:
                next_state, reward, terminated, truncated, info = step_out
                done = terminated or truncated

            agent.store_transition(state, action, reward, next_state, done)
            agent.train_step()

            state = next_state
            ep_reward += reward

            if done:
                break

        print(f"Episode {ep+1}/{num_episodes} | Return: {ep_reward:.2f}")

    env.close()


if __name__ == "__main__":
    train_td3_drone()
