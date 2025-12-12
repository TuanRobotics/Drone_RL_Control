import numpy as np
import pybullet as p
import pkg_resources
from gymnasium import spaces
from collections import deque

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
        # RNG for spawn sampling
        self._rng = np.random.default_rng() # Initialize with default seed
        self.EPISODE_LEN_SEC = 12
        self.collided = False
        self._radius = radius
        self._center_xy = np.array(center_xy, dtype=float)
        self._global_yaw_offset = 0.0
        self._num_gates = 7
        self._gate_span = np.pi  # Half circle (180 degrees)
        self._gate_half_depth = 0.6 # Half depth of each gate along the track direction
        self.success_buffer = deque(maxlen=10000)  # store successful pass poses

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

        self.racing_setup, self.gate_yaws = self._build_semicircle_setup() # Build semicircular gate setup
        preset_flags = getattr(self, "_preset_passing_flags", None)
        self.passing_flag = preset_flags if preset_flags is not None else [False for _ in range(len(self.racing_setup))]
        self._preset_passing_flags = None
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
            yaw = ang + np.pi / 2 + self._global_yaw_offset  # Make each gate face along the tangent
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

    def _observationSpace(self):
        """Use a fixed 36-dim kinematic racing observation to match _computeObs."""
        if self.OBS_TYPE == ObservationType.RGB:
            return super()._observationSpace()
        lo = -np.inf * np.ones(36)
        hi = np.inf * np.ones(36)
        return spaces.Box(low=lo, high=hi, dtype=np.float32)

    def reset(self, seed: int = None, options: dict | None = None):
        """Reset env with optional distributed track spawn for curriculum."""
        if options is None:
            options = {}
        spawn_mode = options.get("spawn", "default")  # "default" | "random_track" | "success_replay"
        # Optional track randomization per episode
        if options.get("random_track_layout", False):
            rmin, rmax = options.get("radius_range", (3.0, 6.0))
            self._radius = float(self._rng.uniform(rmin, rmax))
            jitter = float(options.get("center_jitter", 1.0))
            self._center_xy = np.array([self._rng.uniform(-jitter, jitter), self._rng.uniform(-jitter, jitter)])
            yaw_jitter = float(options.get("yaw_jitter", 0.4))
            self._global_yaw_offset = float(self._rng.uniform(-yaw_jitter, yaw_jitter))
        if spawn_mode == "success_replay" and len(self.success_buffer) > 0:
            xyz, yaw, passed_flags = self._sample_success_spawn(options)
            self.INIT_XYZS[0, :] = xyz
            self.INIT_RPYS[0, :] = np.array([0.0, 0.0, yaw])
            self.prev_pos = xyz
            self._preset_passing_flags = passed_flags
        elif spawn_mode == "random_track":
            xyz, yaw, passed_flags = self._sample_track_spawn(options)
            self.INIT_XYZS[0, :] = xyz
            self.INIT_RPYS[0, :] = np.array([0.0, 0.0, yaw])
            self.prev_pos = xyz
            self._preset_passing_flags = passed_flags
        elif spawn_mode == "default":
            # Place just before gate 0, facing gate 0
            gate_center = np.array(self.racing_setup[0][0])
            gate_yaw = self.gate_yaws[0]
            back_offset = float(options.get("back_offset", 1.0))
            tangent = np.array([np.cos(gate_yaw), np.sin(gate_yaw)])
            pos_xy = gate_center[:2] - back_offset * tangent
            pos_z = gate_center[2]
            xyz = np.array([pos_xy[0], pos_xy[1], pos_z])
            self.INIT_XYZS[0, :] = xyz
            self.INIT_RPYS[0, :] = np.array([0.0, 0.0, gate_yaw])
            self.prev_pos = xyz
            self._preset_passing_flags = [False for _ in range(self._num_gates)]
        return super().reset(seed=seed, options=options)

    def _computeObs(self):
        if self.OBS_TYPE == ObservationType.RGB:
            return super()._computeObs()
        obs = super()._computeObs()
        return np.asarray(obs, dtype=np.float32).reshape(-1)

    def _computeReward(self):
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        ang_vel = state[13:16]

        # Xác định gate hiện tại & gate trước
        next_gate = None
        for i in sorted(self.racing_setup.keys()):
            if not self.passing_flag[i]:
                next_gate = i
                break
        if next_gate is None:
            return 0.0  # đã hoàn thành

        prev_gate = max(next_gate - 1, 0)  # tùy bạn định nghĩa lap/start

        g1 = np.array(self.racing_setup[prev_gate][0])   # center gate trước
        g2 = np.array(self.racing_setup[next_gate][0])   # center gate sau
        seg = g2 - g1
        seg_len = np.linalg.norm(seg) + 1e-6

        # project vị trí hiện tại & trước đó lên segment
        def s_on_segment(p):
            return np.dot(p - g1, seg) / seg_len

        s_cur = s_on_segment(pos)
        s_prev = s_on_segment(self.prev_pos)

        path_progress = s_cur - s_prev     # giống rp(t) trong paper

        # nhỏ thôi, giống -b ||ω||^2
        rate_penalty = 0.001 * np.linalg.norm(ang_vel) ** 2

        # (optional) safety reward như paper – có thể thêm sau
        safety_reward = 0.0  # TODO: tính theo khoảng cách tới mặt phẳng gate nếu muốn

        # Check pass gate / collision
        passing = False
        if self._crossed_gate(self.prev_pos, pos, next_gate):
            self.passing_flag[next_gate] = True
            passing = True

        self.collided = any(
            len(p.getContactPoints(bodyA=self.DRONE_IDS[0],
                                bodyB=gate_id,
                                physicsClientId=self.CLIENT)) != 0
            for gate_id in self.GATE_IDs
        )

        # Terminal penalty khi crash (kiểu r_T)
        if self.collided:
            gate_center = g2
            dg = np.linalg.norm(pos - gate_center)
            wg = self.w  # hoặc kích thước thực của gate
            crash_penalty = -min((dg / wg) ** 2, 20.0)
            reward = crash_penalty
        else:
            reward = path_progress + safety_reward - rate_penalty

        # (tuỳ) nếu muốn vẫn giữ success_buffer
        if passing and not self.collided:
            self.success_buffer.append((pos.copy(), self.gate_yaws[next_gate], next_gate))

        self.prev_pos = pos.copy()
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

    def _crossed_gate(self, prev_position, cur_position, gate_idx):
        """Detect whether the drone actually flew through the gate volume (back -> front)."""
        center = np.array(self.racing_setup[gate_idx][0])
        yaw = self.gate_yaws[gate_idx]
        rel_prev = prev_position - center
        rel_cur = cur_position - center
        rot = np.array([
            [np.cos(-yaw), -np.sin(-yaw), 0.0],
            [np.sin(-yaw), np.cos(-yaw), 0.0],
            [0.0, 0.0, 1.0],
        ])
        local_prev = rot.dot(rel_prev)
        local_cur = rot.dot(rel_cur)

        half_w = self.w / 2
        half_h = self.h / 2
        half_d = self._gate_half_depth
        tol = 0.02  # tighter margin to avoid false positives

        within_prev = (
            -half_w - tol < local_prev[0] < half_w + tol and
            -half_h - tol < local_prev[2] < half_h + tol
        )
        within_cur = (
            -half_w - tol < local_cur[0] < half_w + tol and
            -half_h - tol < local_cur[2] < half_h + tol
        )

        # Require a full traverse through the gate depth, not just touching the plane.
        crossed_plane = (local_prev[1] < -half_d) and (local_cur[1] > half_d)
        forward_delta = (local_cur[1] - local_prev[1]) > 0.1  # ensure forward motion
        return within_prev and within_cur and crossed_plane and forward_delta

    def _computeTruncated(self):
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        if np.linalg.norm(pos[0:2] - self._center_xy) > self._radius + 3:
            return True
        if pos[2] < 0.2 or pos[2] > self.h + 2 * self.offset:
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

    def _sample_track_spawn(self, options: dict):
        """Sample a hover pose near a random gate, oriented toward the next gate."""
        gate_idx = int(self._rng.integers(0, self._num_gates))
        center = np.array(self.racing_setup[gate_idx][0])
        yaw = self.gate_yaws[gate_idx]
        back_offset = float(options.get("back_offset", 0.8))
        noise_xy = float(options.get("noise_xy", 0.2))
        noise_z = float(options.get("noise_z", 0.05))

        # Move slightly upstream along the track tangent and add small noise
        tangent = np.array([np.cos(yaw), np.sin(yaw)])
        pos_xy = center[:2] - back_offset * tangent + self._rng.normal(0, noise_xy, size=2)
        pos_z = center[2] + self._rng.normal(0, noise_z)
        xyz = np.array([pos_xy[0], pos_xy[1], pos_z])

        # Mark previous gates as already passed so the next target is gate_idx
        passed_flags = [i < gate_idx for i in range(self._num_gates)] # Gates with index less than gate_idx are considered passed
        return xyz, yaw, passed_flags

    def _sample_success_spawn(self, options: dict):
        """Sample a pose from previously successful gate passes."""
        idx = int(self._rng.integers(0, len(self.success_buffer)))
        pos, yaw, gate_idx = self.success_buffer[idx]
        noise_xy = float(options.get("noise_xy", 0.05))
        noise_z = float(options.get("noise_z", 0.05))
        xyz = np.array(pos) + self._rng.normal(0, [noise_xy, noise_xy, noise_z])
        # Gates up to and including gate_idx are treated as passed; next target is gate_idx+1
        passed_flags = [i <= gate_idx for i in range(self._num_gates)]
        return xyz, yaw, passed_flags
