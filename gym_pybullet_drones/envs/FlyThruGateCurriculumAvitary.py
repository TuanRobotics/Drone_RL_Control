#!/usr/bin/env python3

import numpy as np
import pybullet as p
import pkg_resources
from dataclasses import dataclass
from typing import List, Sequence
from collections import deque

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


@dataclass
class CurriculumLevel:
    """Configuration for a single curriculum stage."""
    forward_offset: float  # spawn distance in +y from the gate plane
    xy_noise: float        # lateral noise for spawn sampling
    z_noise: float         # vertical noise for spawn sampling
    success_radius: float  # distance threshold to count as passing the gate
    forward_limit: float   # allowed distance in front of the gate
    back_limit: float      # allowed distance behind the gate
    x_limit: float         # allowed lateral displacement


class FlyThruGateCurriculumAvitary(BaseRLAviary):
    """Single-gate environment with a 5-level curriculum that starts near the gate."""

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 num_drones: int = 1,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui: bool = False,
                 record: bool = False,
                 obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.RPM,
                 curriculum_levels: Sequence[CurriculumLevel] | None = None,
                 success_window: int = 30,
                 success_rate_threshold: float = 0.7):
        """
        A curriculum-driven version of the fly-through-gate task.

        The drone starts close to the gate and progressively spawns farther away
        as it collects successful episodes.
        """
        self.EPISODE_LEN_SEC = 8
        self._rng = np.random.default_rng()
        self.GATE_POS = np.array([0.0, -2.0, 0.75])
        self.FINAL_TARGET = self.GATE_POS.copy()

        self.curriculum_levels: List[CurriculumLevel] = list(curriculum_levels) if curriculum_levels else self._default_curriculum()
        if len(self.curriculum_levels) < 5:
            raise ValueError("FlyThruGateCurriculumAvitary requires at least 5 curriculum levels.")

        self.curriculum_level = 0
        self.success_window = success_window
        self.success_rate_threshold = success_rate_threshold
        self.recent_results: deque[bool] = deque(maxlen=success_window)
        self.level_successes = 0
        self.episode_counter = 0
        self._last_episode_success: bool | None = None
        self._recent_success_rate = 0.0

        spawn = self._spawn_for_level(self.curriculum_level)
        init_xyzs = np.vstack([spawn for _ in range(num_drones)]) if initial_xyzs is None else initial_xyzs
        default_rpys = np.zeros((num_drones, 3))
        default_rpys[:, 2] = -np.pi / 2  # Face the gate (negative y direction)
        init_rpys = default_rpys if initial_rpys is None else initial_rpys

        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         initial_xyzs=init_xyzs,
                         initial_rpys=init_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act)

    ##############################################################################
    # Load URDF to create the gate
    def _addObstacles(self):
        super()._addObstacles()
        p.loadURDF(pkg_resources.resource_filename('gym_pybullet_drones', 'assets/gate.urdf'),
                   [0, -1, 0],
                   p.getQuaternionFromEuler([0, 0, 1.5]),
                   physicsClientId=self.CLIENT
                   )

    def _default_curriculum(self) -> List[CurriculumLevel]:
        """Five stages from near-gate spawn to farther, noisier starts."""
        return [
            CurriculumLevel(forward_offset=0.35, xy_noise=0.03, z_noise=0.02, success_radius=0.35,
                            forward_limit=1.0, back_limit=1.0, x_limit=1.0),
            CurriculumLevel(forward_offset=0.7, xy_noise=0.06, z_noise=0.03, success_radius=0.30,
                            forward_limit=1.3, back_limit=1.2, x_limit=1.2),
            CurriculumLevel(forward_offset=1.1, xy_noise=0.09, z_noise=0.05, success_radius=0.25,
                            forward_limit=1.7, back_limit=1.5, x_limit=1.5),
            CurriculumLevel(forward_offset=1.6, xy_noise=0.13, z_noise=0.07, success_radius=0.22,
                            forward_limit=2.1, back_limit=1.7, x_limit=1.8),
            CurriculumLevel(forward_offset=2.1, xy_noise=0.18, z_noise=0.09, success_radius=0.20,
                            forward_limit=2.6, back_limit=2.0, x_limit=2.2),
        ]

    def _spawn_for_level(self, level_idx: int) -> np.ndarray:
        """Sample an initial spawn in front of the gate for the current level."""
        level = self.curriculum_levels[level_idx]
        x = self.GATE_POS[0] + self._rng.normal(0.0, level.xy_noise)
        y = self.GATE_POS[1] + level.forward_offset + self._rng.normal(0.0, level.xy_noise)
        z = max(0.1, self.GATE_POS[2] + self._rng.normal(0.0, level.z_noise))
        return np.array([x, y, z], dtype=float)

    def _bounds_for_level(self, level: CurriculumLevel):
        y_max = self.GATE_POS[1] + level.forward_limit
        y_min = self.GATE_POS[1] - level.back_limit
        z_max = 2.0
        return level.x_limit, y_min, y_max, z_max

    def _update_curriculum_from_previous_episode(self):
        """Advance curriculum when success rate over the window is high enough."""
        if self._last_episode_success is None:
            return

        self.episode_counter += 1
        self.recent_results.append(self._last_episode_success)
        self.level_successes = sum(self.recent_results)
        self._recent_success_rate = self.level_successes / len(self.recent_results)

        if (len(self.recent_results) >= self.success_window
                and self._recent_success_rate >= self.success_rate_threshold
                and self.curriculum_level < len(self.curriculum_levels) - 1):
            self.curriculum_level += 1
            self.recent_results.clear()
            self.level_successes = 0
            print(f"[Curriculum] Moved to level {self.curriculum_level + 1}/{len(self.curriculum_levels)} "
                  f"(rate={self._recent_success_rate:.2f} over last {self.success_window} episodes)")
        self._last_episode_success = None

    def reset(self, seed: int = None, options: dict | None = None):
        """Reset the environment and resample spawn based on the curriculum level."""
        self._update_curriculum_from_previous_episode()
        spawn = self._spawn_for_level(self.curriculum_level)
        for i in range(self.NUM_DRONES):
            self.INIT_XYZS[i, :] = spawn
            self.INIT_RPYS[i, :] = np.array([0.0, 0.0, -np.pi / 2])

        obs, info = super().reset(seed=seed, options=options)
        info["curriculum_level"] = self.curriculum_level
        info["level_successes"] = self.level_successes
        info["success_window"] = self.success_window
        info["recent_success_rate"] = self._recent_success_rate
        info["success_rate_threshold"] = self.success_rate_threshold
        return obs, info

    def _has_passed_gate(self, position: np.ndarray, level: CurriculumLevel) -> bool:
        """Success when the drone crosses the gate plane and stays within radius."""
        passed_plane = position[1] <= self.GATE_POS[1]
        return passed_plane and np.linalg.norm(position - self.GATE_POS) <= level.success_radius

    def _is_out_of_bounds(self, position: np.ndarray, level: CurriculumLevel) -> bool:
        """Check workspace bounds around the gate based on the curriculum level."""
        x_lim, y_min, y_max, z_max = self._bounds_for_level(level)
        return (
            abs(position[0]) > x_lim
            or position[1] < y_min
            or position[1] > y_max
            or position[2] < 0.05
            or position[2] > z_max
        )

    def _computeReward(self):
        """Reward that encourages moving toward and through the gate."""
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        vel = state[10:13]
        level = self.curriculum_levels[self.curriculum_level]

        distance = np.linalg.norm(pos - self.GATE_POS)
        shaping_scale = 0.25 + 0.05 * self.curriculum_level
        distance_reward = np.exp(-distance / shaping_scale)

        progress_reward = max(0.0, -vel[1])  # positive when moving toward -y
        smoothness_penalty = 0.05 * np.linalg.norm(vel)

        if self._has_passed_gate(pos, level):
            return 15.0

        if self._is_out_of_bounds(pos, level):
            return -10.0

        reward = 2.0 * distance_reward + 1.0 * progress_reward - smoothness_penalty
        return float(reward)

    def _computeTruncated(self):
        """Stop the episode when the drone leaves the workspace or time expires."""
        state = self._getDroneStateVector(0)
        level = self.curriculum_levels[self.curriculum_level]
        pos = state[0:3]
        if self._is_out_of_bounds(pos, level) or abs(state[7]) > 0.5 or abs(state[8]) > 0.5:
            self._last_episode_success = False
            return True

        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            self._last_episode_success = False
            return True
        return False

    def _computeTerminated(self):
        """Terminate only when the gate is successfully passed."""
        pos = self._getDroneStateVector(0)[0:3]
        level = self.curriculum_levels[self.curriculum_level]
        if self._has_passed_gate(pos, level):
            self._last_episode_success = True
            return True
        return False

    def _computeInfo(self):
        state = self._getDroneStateVector(0)
        info = {
            "position": state[0:3],
            "quaternion": state[3:7],
            "rpy": state[7:10],
            "lin_vel": state[10:13],
            "ang_vel": state[13:16],
            "curriculum_level": self.curriculum_level,
            "level_successes": self.level_successes,
            "success_window": self.success_window,
            "recent_success_rate": self._recent_success_rate,
            "success_rate_threshold": self.success_rate_threshold,
        }
        return info

    def _clipAndNormalizeState(self, state):
        """Normalize state to [-1, 1] using curriculum-aware bounds."""
        max_x = max(level.x_limit for level in self.curriculum_levels)
        max_y_forward = max(abs(self.GATE_POS[1]) + level.forward_limit for level in self.curriculum_levels)
        max_y_back = max(abs(self.GATE_POS[1]) + level.back_limit for level in self.curriculum_levels)
        max_xy = max(max_x, max_y_forward, max_y_back)
        max_z = 2.0
        max_lin_vel_xy = 3
        max_lin_vel_z = 1.5
        max_pitch_roll = np.pi

        clipped_pos_xy = np.clip(state[0:2], -max_xy, max_xy)
        clipped_pos_z = np.clip(state[2], 0, max_z)
        clipped_rp = np.clip(state[7:9], -max_pitch_roll, max_pitch_roll)
        clipped_vel_xy = np.clip(state[10:12], -max_lin_vel_xy, max_lin_vel_xy)
        clipped_vel_z = np.clip(state[12], -max_lin_vel_z, max_lin_vel_z)

        if self.GUI:
            self._clipAndNormalizeStateWarning(state,
                                               clipped_pos_xy,
                                               clipped_pos_z,
                                               clipped_rp,
                                               clipped_vel_xy,
                                               clipped_vel_z)

        normalized_pos_xy = clipped_pos_xy / max_xy
        normalized_pos_z = clipped_pos_z / max_z
        normalized_rp = clipped_rp / max_pitch_roll
        normalized_yaw = state[9] / np.pi
        normalized_vel_xy = clipped_vel_xy / max_lin_vel_xy
        normalized_vel_z = clipped_vel_z / max_lin_vel_z
        ang_vel_norm = np.linalg.norm(state[13:16])
        normalized_ang_vel = state[13:16] / ang_vel_norm if ang_vel_norm != 0 else state[13:16]

        norm_and_clipped = np.hstack([normalized_pos_xy,
                                      normalized_pos_z,
                                      state[3:7],
                                      normalized_rp,
                                      normalized_yaw,
                                      normalized_vel_xy,
                                      normalized_vel_z,
                                      normalized_ang_vel,
                                      state[16:20]]).reshape(20,)
        return norm_and_clipped

    def _clipAndNormalizeStateWarning(self,
                                      state,
                                      clipped_pos_xy,
                                      clipped_pos_z,
                                      clipped_rp,
                                      clipped_vel_xy,
                                      clipped_vel_z):
        """Print a warning when the state needed clipping (debug only)."""
        if not (clipped_pos_xy == np.array(state[0:2])).all():
            print("[WARNING] it", self.step_counter, "in FlyThruGateCurriculumAvitary._clipAndNormalizeState(), clipped xy position [{:.2f} {:.2f}]".format(state[0], state[1]))
        if not (clipped_pos_z == np.array(state[2])).all():
            print("[WARNING] it", self.step_counter, "in FlyThruGateCurriculumAvitary._clipAndNormalizeState(), clipped z position [{:.2f}]".format(state[2]))
        if not (clipped_rp == np.array(state[7:9])).all():
            print("[WARNING] it", self.step_counter, "in FlyThruGateCurriculumAvitary._clipAndNormalizeState(), clipped roll/pitch [{:.2f} {:.2f}]".format(state[7], state[8]))
        if not (clipped_vel_xy == np.array(state[10:12])).all():
            print("[WARNING] it", self.step_counter, "in FlyThruGateCurriculumAvitary._clipAndNormalizeState(), clipped xy velocity [{:.2f} {:.2f}]".format(state[10], state[11]))
        if not (clipped_vel_z == np.array(state[12])).all():
            print("[WARNING] it", self.step_counter, "in FlyThruGateCurriculumAvitary._clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(state[12]))
