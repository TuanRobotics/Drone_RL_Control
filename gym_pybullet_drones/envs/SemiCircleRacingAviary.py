import numpy as np
import pybullet as p
import pkg_resources

from gym_pybullet_drones.envs.BaseRacingRLAviary import BaseRacingRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, ObservationType, ActionType, Physics


class SemiCircleRacingAviary(BaseRacingRLAviary):
    """Racing environment with five gates arranged on a semicircular track."""

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 48,
                 gui: bool = False,
                 record: bool = False,
                 obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.RPM,
                 radius: float = 4.0,
                 center_xy: tuple = (0.0, 0.0)):
        self.EPISODE_LEN_SEC = 12
        self.collided = False
        self._radius = radius
        self._center_xy = np.array(center_xy, dtype=float)
        self._num_gates = 5
        self._gate_span = np.pi  # Half circle
        self._gate_half_depth = 0.6

        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act)
        # Reset progress tracking based on the actual initial state
        self.prev_pos = self._getDroneStateVector(0)[:3]

    def _addObstacles(self):
        """Place five gates on a semicircle and store their metadata."""
        super()._addObstacles()
        self.racing_setup, self.gate_yaws = self._build_semicircle_setup()
        self.passing_flag = [False for _ in range(len(self.racing_setup))]
        gate_ids = []
        for idx in sorted(self.racing_setup.keys()):
            gate_center = self.racing_setup[idx][0]
            gate_yaw = self.gate_yaws[idx]
            gate_ids.append(
                p.loadURDF(
                    pkg_resources.resource_filename('gym_pybullet_drones', 'assets/gate.urdf'),
                    gate_center,
                    p.getQuaternionFromEuler([0, 0, gate_yaw]),
                    physicsClientId=self.CLIENT
                )
            )
        self.GATE_IDs = np.array(gate_ids)

    def _build_semicircle_setup(self):
        """Return gate centers/corners and yaw angles for a semicircular track."""
        angles = np.linspace(-self._gate_span / 2, self._gate_span / 2, self._num_gates)
        setup = {}
        yaws = {}
        half_w = self.w / 2
        half_h = self.h / 2
        for idx, ang in enumerate(angles):
            cx = self._center_xy[0] + self._radius * np.cos(ang)
            cy = self._center_xy[1] + self._radius * np.sin(ang)
            cz = self.h / 2 + self.offset
            yaw = ang + np.pi / 2  # Make each gate face along the tangent
            rot = np.array([
                [np.cos(yaw), -np.sin(yaw), 0.0],
                [np.sin(yaw), np.cos(yaw), 0.0],
                [0.0, 0.0, 1.0],
            ])
            offsets = [
                np.array([0.0, 0.0, 0.0]),  # Center
                np.array([-half_w, 0.0, half_h]),
                np.array([half_w, 0.0, half_h]),
                np.array([-half_w, 0.0, -half_h]),
                np.array([half_w, 0.0, -half_h]),
            ]
            setup[idx] = [list(np.array([cx, cy, cz]) + rot.dot(off)) for off in offsets]
            yaws[idx] = yaw
        return setup, yaws

    def _computeReward(self):
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        ang_vel = state[13:16]
        next_gate = None
        for i in sorted(self.racing_setup.keys()):
            if not self.passing_flag[i]:
                next_gate = i
                break
        if next_gate is None:
            return 0.0

        cur_dist = np.linalg.norm(np.array(self.racing_setup[next_gate][0]) - pos)
        prev_dist = np.linalg.norm(np.array(self.racing_setup[next_gate][0]) - self.prev_pos)
        self.prev_pos = pos.copy()

        progress_reward = prev_dist - cur_dist - 0.001 * np.linalg.norm(ang_vel)
        passing = False
        if self._is_inside_gate(pos, next_gate):
            self.passing_flag[next_gate] = True
            passing = True

        self.collided = False
        for gate_id in self.GATE_IDs:
            if len(p.getContactPoints(bodyA=self.DRONE_IDS[0], bodyB=gate_id, physicsClientId=self.CLIENT)) != 0:
                self.collided = True
                break

        if passing:
            reward = 10.0
        elif self.collided:
            reward = -10.0
        else:
            reward = progress_reward
        return reward

    def _is_inside_gate(self, position, gate_idx):
        """Check if position is within rotated gate bounds."""
        center = np.array(self.racing_setup[gate_idx][0])
        yaw = self.gate_yaws[gate_idx]
        rel = position - center
        rot = np.array([
            [np.cos(-yaw), -np.sin(-yaw), 0.0],
            [np.sin(-yaw), np.cos(-yaw), 0.0],
            [0.0, 0.0, 1.0],
        ])
        local = rot.dot(rel)
        half_w = self.w / 2
        half_h = self.h / 2
        return (
            -half_w < local[0] < half_w and
            -self._gate_half_depth < local[1] < self._gate_half_depth and
            -half_h < local[2] < half_h
        )

    def _computeTruncated(self):
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        if np.linalg.norm(pos[0:2] - self._center_xy) > self._radius + 3:
            return True
        if pos[2] < 0.0 or pos[2] > self.h + 2 * self.offset:
            return True
        if abs(state[7]) > 0.8 or abs(state[8]) > 0.8:
            return True
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        return False

    def _computeTerminated(self):
        if all(self.passing_flag) or self.collided:
            return True
        return False

    def _computeInfo(self):
        return {"passing_flag": self.passing_flag}

    def _clipAndNormalizeState(self, state):
        MAX_LIN_VEL_XY = 3
        MAX_LIN_VEL_Z = 1

        MAX_XY = MAX_LIN_VEL_XY * self.EPISODE_LEN_SEC
        MAX_Z = MAX_LIN_VEL_Z * self.EPISODE_LEN_SEC
        MAX_PITCH_ROLL = np.pi

        clipped_pos_xy = np.clip(state[0:2], -MAX_XY, MAX_XY)
        clipped_pos_z = np.clip(state[2], 0, MAX_Z)
        clipped_rp = np.clip(state[7:9], -MAX_PITCH_ROLL, MAX_PITCH_ROLL)
        clipped_vel_xy = np.clip(state[10:12], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
        clipped_vel_z = np.clip(state[12], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)

        if self.GUI:
            self._clipAndNormalizeStateWarning(state,
                                               clipped_pos_xy,
                                               clipped_pos_z,
                                               clipped_rp,
                                               clipped_vel_xy,
                                               clipped_vel_z)

        normalized_pos_xy = clipped_pos_xy / MAX_XY
        normalized_pos_z = clipped_pos_z / MAX_Z
        normalized_rp = clipped_rp / MAX_PITCH_ROLL
        normalized_y = state[9] / np.pi
        normalized_vel_xy = clipped_vel_xy / MAX_LIN_VEL_XY
        normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_XY
        normalized_ang_vel = state[13:16] / np.linalg.norm(state[13:16]) if np.linalg.norm(state[13:16]) != 0 else state[13:16]

        norm_and_clipped = np.hstack([normalized_pos_xy,
                                      normalized_pos_z,
                                      state[3:7],
                                      normalized_rp,
                                      normalized_y,
                                      normalized_vel_xy,
                                      normalized_vel_z,
                                      normalized_ang_vel,
                                      state[16:20]
                                      ]).reshape(20,)
        return norm_and_clipped
