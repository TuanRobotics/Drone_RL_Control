#!/usr/bin/env python3

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import pkg_resources
import pybullet as p

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import ActionType, DroneModel, ObservationType, Physics


@dataclass
class CurriculumLevel:
    """Configuration for one curriculum stage."""

    forward_offset: float  # spawn distance in +y from the gate plane
    xy_noise: float  # lateral noise for spawn sampling
    z_noise: float  # vertical noise for spawn sampling
    success_radius: float  # distance threshold to count as passing the gate
    forward_limit: float  # allowed distance in front of the gate
    back_limit: float  # allowed distance behind the gate
    x_limit: float  # allowed lateral displacement


class Curriculum:
    """Minimal holder for curriculum levels so training code can drive the progression."""

    def __init__(self, levels: Sequence[CurriculumLevel], start_level: int = 0):
        if not levels:
            raise ValueError("Curriculum must contain at least one level.")
        self.levels: List[CurriculumLevel] = list(levels)
        self.level = self._clamp(start_level)

    @property
    def num_levels(self) -> int:
        return len(self.levels)

    def _clamp(self, value: int) -> int:
        return max(0, min(int(value), self.num_levels - 1))

    def set_level(self, value: int) -> None:
        self.level = self._clamp(value)

    def current_level(self) -> CurriculumLevel:
        return self.levels[self.level]


class SpawnSampler:
    """Samples initial positions in front of the gate for a given curriculum level."""

    def __init__(self, levels: Sequence[CurriculumLevel], rng: np.random.Generator | None = None):
        self.levels = list(levels)
        self.spawn_ranges = self._build_spawn_ranges(self.levels)
        self.rng = rng if rng is not None else np.random.default_rng()

    def range_for_level(self, level_idx: int) -> Tuple[float, float]:
        return self.spawn_ranges[level_idx]

    def sample(self, gate_pos: np.ndarray, level_idx: int) -> np.ndarray:
        """Sample an initial spawn in front of the gate."""
        level = self.levels[level_idx]
        forward_min, forward_max = self.spawn_ranges[level_idx]
        forward_offset = self.rng.uniform(forward_min, forward_max) # it is the y coordinate offset from the gate position
        x = gate_pos[0] + self.rng.normal(0.0, level.xy_noise)
        y = gate_pos[1] + forward_offset + self.rng.normal(0.0, level.xy_noise)
        z = max(0.1, gate_pos[2] + self.rng.normal(0.0, level.z_noise))
        return np.array([x, y, z], dtype=float)

    @staticmethod
    def _build_spawn_ranges(levels: Sequence[CurriculumLevel]) -> List[Tuple[float, float]]:
        """Construct spawn ranges without exceeding workspace limits."""
        ranges: List[Tuple[float, float]] = []
        for level in levels:
            spread = max(0.15, 0.25 * level.forward_offset)
            forward_min = max(0.05, level.forward_offset - spread) # it is for avoiding spawn behind the gate
            forward_max = min(level.forward_limit - 0.05, level.forward_offset + spread) # avoid spawning too far
            if forward_max <= forward_min:
                forward_max = forward_min + 0.05 # It is to ensure there is always some range to sample from
            ranges.append((forward_min, forward_max))
        return ranges # it is a list of tuples (min, max) for each level


class FlyThruGateCurriculumAvitary(BaseRLAviary):
    """Single-gate environment with curriculum-aware spawn ranges."""

    def __init__(
        self,
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
        start_level: int = 0,
    ):
        """
        A curriculum-driven version of the fly-through-gate task.

        Success tracking and level progression are expected to be handled in the
        training loop; the environment only exposes spawn sampling per level.
        """
        self.EPISODE_LEN_SEC = 12
        self._rng = np.random.default_rng()
        self.GATE_POS = np.array([0.0, -2.0, 0.3])
        self.FINAL_TARGET = self.GATE_POS.copy()

        levels = list(curriculum_levels) if curriculum_levels else self._default_curriculum()
        self.curriculum = Curriculum(levels, start_level=start_level)
        self.spawner = SpawnSampler(self.curriculum.levels, rng=self._rng)

        spawn = self.spawner.sample(self.GATE_POS, self.curriculum_level)
        init_xyzs = np.vstack([spawn for _ in range(num_drones)]) if initial_xyzs is None else initial_xyzs
        default_rpys = np.zeros((num_drones, 3))
        default_rpys[:, 2] = -np.pi / 2  # Face the gate (negative y direction)
        init_rpys = default_rpys if initial_rpys is None else initial_rpys

        super().__init__(
            drone_model=drone_model,
            num_drones=num_drones,
            initial_xyzs=init_xyzs,
            initial_rpys=init_rpys,
            physics=physics,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=record,
            obs=obs,
            act=act,
        )

    # ------------------------------------------------------------------------- #
    # Properties
    @property
    def curriculum_level(self) -> int:
        return self.curriculum.level

    @curriculum_level.setter
    def curriculum_level(self, value: int):
        self.curriculum.set_level(value)

    @property
    def num_levels(self) -> int:
        return self.curriculum.num_levels

    # ------------------------------------------------------------------------- #
    # Obstacles and scene setup
    def _addObstacles(self):
        super()._addObstacles()
        box_id = p.loadURDF(
            pkg_resources.resource_filename("gym_pybullet_drones", "assets/gate.urdf"),
            [0, -1, 0],
            p.getQuaternionFromEuler([0, 0, 1.5]),
            physicsClientId=self.CLIENT,
        )
        gate_pos, _ = p.getBasePositionAndOrientation(box_id)
        self.GATE_POS = np.array(gate_pos, dtype=float)

    # ------------------------------------------------------------------------- #
    # Curriculum helpers
    def _default_curriculum(self) -> List[CurriculumLevel]:
        """Five stages from near-gate spawn to farther, noisier starts."""
        return [
            CurriculumLevel(
                forward_offset=0.1,
                xy_noise=0.03,
                z_noise=0.02,
                success_radius=0.15,
                forward_limit=1.0,
                back_limit=1.0,
                x_limit=1.0,
            ),
            CurriculumLevel(
                forward_offset=0.3,
                xy_noise=0.06,
                z_noise=0.03,
                success_radius=0.15,
                forward_limit=0.7,
                back_limit=1.2,
                x_limit=1.2,
            ),
            CurriculumLevel(
                forward_offset=1.1,
                xy_noise=0.09,
                z_noise=0.05,
                success_radius=0.1,
                forward_limit=1.7,
                back_limit=1.5,
                x_limit=1.5,
            ),
            CurriculumLevel(
                forward_offset=1.5,
                xy_noise=0.13,
                z_noise=0.07,
                success_radius=0.1,
                forward_limit=2.1,
                back_limit=1.7,
                x_limit=1.8,
            ),
            CurriculumLevel(
                forward_offset=2.0,
                xy_noise=0.18,
                z_noise=0.09,
                success_radius=0.05,
                forward_limit=2.6,
                back_limit=2.0,
                x_limit=2.2,
            ),
        ]

    def _bounds_for_level(self, level: CurriculumLevel):
        y_max = self.GATE_POS[1] + level.forward_limit
        y_min = self.GATE_POS[1] - level.back_limit
        z_max = 2.0
        return level.x_limit, y_min, y_max, z_max

    # ------------------------------------------------------------------------- #
    # Environment overrides
    def reset(self, seed: int = None, options: dict | None = None):
        """
        Reset the environment and resample spawn based on the externally-controlled level.

        The call to ``super().reset`` computes the observation as usual.
        """
        spawn = self.spawner.sample(self.GATE_POS, self.curriculum_level)
        for i in range(self.NUM_DRONES):
            self.INIT_XYZS[i, :] = spawn
            self.INIT_RPYS[i, :] = np.array([0.0, 0.0, -np.pi / 2])

        obs, info = super().reset(seed=seed, options=options)
        info.update(
            {
                "curriculum_level": self.curriculum_level,
                "spawn_range": self.spawner.range_for_level(self.curriculum_level),
            }
        )
        return super().reset(seed=seed, options=options)

    def _has_passed_gate(self, position: np.ndarray, level: CurriculumLevel) -> bool:
        """Success when the drone crosses the gate plane and stays within radius."""
        passed_plane = position[1] <= self.FINAL_TARGET[1]
        return passed_plane and np.linalg.norm(position - self.FINAL_TARGET) <= level.success_radius

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
        level = self.curriculum.current_level()

        gate_direction = np.array([0.0, -1.0, 0.0])  # Toward negative y (through the gate)
        way_point = self.GATE_POS + np.array([0.0, -1.0, 0.25])
        self.FINAL_TARGET = way_point

        distance = np.linalg.norm(pos - way_point)
        distance_reward = np.exp(-2.0 * distance)

        progress = np.dot(vel, gate_direction)
        progress_reward = max(0.0, progress)

        if self._has_passed_gate(pos, level):
            return 10.0

        if self._is_out_of_bounds(pos, level):
            return -10.0

        reward = 2.0 * distance_reward + progress_reward - 0.1
        return float(reward)

    def _computeTruncated(self):
        """Stop the episode when the drone leaves the workspace or time expires."""
        state = self._getDroneStateVector(0)
        level = self.curriculum.current_level()
        pos = state[0:3]
        if self._is_out_of_bounds(pos, level) or abs(state[7]) > 0.5 or abs(state[8]) > 0.5:
            return True

        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        return False

    def _computeTerminated(self):
        """Terminate only when the gate is successfully passed."""
        pos = self._getDroneStateVector(0)[0:3]
        level = self.curriculum.current_level()
        if self._has_passed_gate(pos, level):
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
            "spawn_range": self.spawner.range_for_level(self.curriculum_level),
        }
        return info

    def _clipAndNormalizeState(self, state):
        """Normalize state to [-1, 1] using curriculum-aware bounds."""
        max_x = max(level.x_limit for level in self.curriculum.levels)
        max_y_forward = max(abs(self.GATE_POS[1]) + level.forward_limit for level in self.curriculum.levels)
        max_y_back = max(abs(self.GATE_POS[1]) + level.back_limit for level in self.curriculum.levels)
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
            self._clipAndNormalizeStateWarning(
                state,
                clipped_pos_xy,
                clipped_pos_z,
                clipped_rp,
                clipped_vel_xy,
                clipped_vel_z,
            )

        normalized_pos_xy = clipped_pos_xy / max_xy
        normalized_pos_z = clipped_pos_z / max_z
        normalized_rp = clipped_rp / max_pitch_roll
        normalized_yaw = state[9] / np.pi
        normalized_vel_xy = clipped_vel_xy / max_lin_vel_xy
        normalized_vel_z = clipped_vel_z / max_lin_vel_z
        ang_vel_norm = np.linalg.norm(state[13:16])
        normalized_ang_vel = state[13:16] / ang_vel_norm if ang_vel_norm != 0 else state[13:16]

        norm_and_clipped = np.hstack(
            [
                normalized_pos_xy,
                normalized_pos_z,
                state[3:7],
                normalized_rp,
                normalized_yaw,
                normalized_vel_xy,
                normalized_vel_z,
                normalized_ang_vel,
                state[16:20],
            ]
        ).reshape(20,)
        return norm_and_clipped

    def _clipAndNormalizeStateWarning(
        self,
        state,
        clipped_pos_xy,
        clipped_pos_z,
        clipped_rp,
        clipped_vel_xy,
        clipped_vel_z,
    ):
        """Print a warning when the state needed clipping (debug only)."""
        if not (clipped_pos_xy == np.array(state[0:2])).all():
            print(
                "[WARNING] it",
                self.step_counter,
                "in FlyThruGateCurriculumAvitary._clipAndNormalizeState(), clipped xy position [{:.2f} {:.2f}]".format(
                    state[0], state[1]
                ),
            )
        if not (clipped_pos_z == np.array(state[2])).all():
            print(
                "[WARNING] it",
                self.step_counter,
                "in FlyThruGateCurriculumAvitary._clipAndNormalizeState(), clipped z position [{:.2f}]".format(
                    state[2]
                ),
            )
        if not (clipped_rp == np.array(state[7:9])).all():
            print(
                "[WARNING] it",
                self.step_counter,
                "in FlyThruGateCurriculumAvitary._clipAndNormalizeState(), clipped roll/pitch [{:.2f} {:.2f}]".format(
                    state[7], state[8]
                ),
            )
        if not (clipped_vel_xy == np.array(state[10:12])).all():
            print(
                "[WARNING] it",
                self.step_counter,
                "in FlyThruGateCurriculumAvitary._clipAndNormalizeState(), clipped xy velocity [{:.2f} {:.2f}]".format(
                    state[10], state[11]
                ),
            )
        if not (clipped_vel_z == np.array(state[12])).all():
            print(
                "[WARNING] it",
                self.step_counter,
                "in FlyThruGateCurriculumAvitary._clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(
                    state[12]
                ),
            )
