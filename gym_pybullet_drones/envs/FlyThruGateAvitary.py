#!/usr/bin/env python3

import os
from matplotlib.pylab import seed
import numpy as np
import pybullet as p
import pkg_resources
import gymnasium as gym
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class FlyThruGateAvitary(BaseRLAviary):
    def __init__(self,
                drone_model: DroneModel = DroneModel.CF2X,
                initial_xyzs=None,
                initial_rpys=None,
                physics: Physics = Physics.PYB,
                num_drones: int = 1,
                pyb_freq: int=240,
                ctrl_freq: int=30,
                gui: bool = False,
                curriculum_level: int = 0,
                max_curriculum_level: int = 5,
                record=False,
                use_curriculum: bool = False,
                obs: ObservationType = ObservationType.KIN,
                act: ActionType = ActionType.RPM,
                output_folder="results"
                ):
        """
        Initialization of a single agent RL environment.
        Using the generic single agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        freq : int, optional
            The frequency (Hz) at which the physics engine steps.
        aggregate_phy_steps : int, optional
            The number of physics steps within one call to `BaseAviary.step()`.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation in folder `files/videos/`.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)
        """
        self.EPISODE_LEN_SEC = 8  # Episode length in seconds
        self.use_curriculum = use_curriculum # Set True to training with curriculum learning
        self.curriculum_level = curriculum_level
        self.max_curriculum_level = max_curriculum_level

        self.GATE_POS = np.array([0, 0.0, 0.0])  # Center of gate
        self.GATE_SCALE = 0.8  # Slightly larger but still narrow
        self.WAYPOINT = np.array([0.0, 0.0, 0.0])
        self.CENTER_GATE = np.array([0.0, 0.0, 0.0])
        self.passed_gate = False  # Flag for track if gate is passed
        self.GATE_ORN = None
        self.gate_normal = None

        # self.success_passed = False
        self.center_gate_passed = False
        self.way_point_success = True 
        self.time_passed_gate = 0.0
        self.threshold_success = 0.05  # Distance threshold to consider gate passed

        # Setup for checking collision with gate
        self.GATE_ID = None # Will be set when loading the gate URDF

        super().__init__(
            drone_model=drone_model,
            num_drones=num_drones,
            initial_xyzs=initial_xyzs,
            initial_rpys=initial_rpys,
            physics=physics,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=record,
            obs=obs,
            act=act,
            output_folder=output_folder
        )
        
    ##############################################################################
    # Load urdf to create the gate 
    def _addObstacles(self): 
        pos = [0.0, -1.0, 0.2]
        tilt = np.deg2rad(0.0)  # 30°
        orn = p.getQuaternionFromEuler([tilt, 0.0, 1.57])
        super()._addObstacles()
        self.GATE_ID = p.loadURDF(
            pkg_resources.resource_filename('gym_pybullet_drones', 'assets/gate.urdf'),
            pos,
            orn,
            physicsClientId=self.CLIENT,
            globalScaling=self.GATE_SCALE,
        )
        self.GATE_POS, self.GATE_ORN = p.getBasePositionAndOrientation(self.GATE_ID, physicsClientId=self.CLIENT)
        rot = np.array(p.getMatrixFromQuaternion(self.GATE_ORN)).reshape(3, 3)
        center_offset = np.array([0.0, 0.0, 0.06 * self.GATE_SCALE])
        self.CENTER_GATE = self.GATE_POS + rot @ center_offset
        self.gate_normal = rot @ np.array([1.0, 0.0, 0.0])  # local +X -> world 
        self.WAYPOINT = self.CENTER_GATE - 0.2 * self.gate_normal

        # Check size of gate length/width/height
        # aabb_min, aabb_max = p.getAABB(self.GATE_ID, physicsClientId=self.CLIENT)
        # size = np.array(aabb_max) - np.array(aabb_min)  # [x, y, z] theo trục thế giới
        # print(f"Gate AABB size (m): {size}")

        # # Test
        # print(f"Gate normal: {self.gate_normal}")
        # print(f"Center gate: {self.CENTER_GATE}")
        # x, y, z = self.WAYPOINT
        # print(f"Waypoint: [{x:.2f}, {y:.2f}, {z:.2f}]") 

        
    def _observationSpace(self): 
        """Returns the observation space of the environment.

        Returns
        -------
        gym.spaces.Box
            The observation space.

        """
        # If not curriculum learning, always use dim = 16
        if self.use_curriculum:
            dim = 19
        else:
            dim = 19
        low = np.full((dim,), -np.inf, dtype=np.float32)
        high = np.full((dim,), np.inf, dtype=np.float32)
        return spaces.Box(low=low, high=high, dtype=np.float32)
        
    def _computeObs(self):
        """Return concatenated state, reference, and tracking error.

        Observation layout (float32):
        [0:3]   -> position (x, y, z) (m)
        [3:7]   -> orientation (quaternion x, y, z, w)
        [7:10]  -> roll, pitch, yaw (rad)   
        [10:13] -> linear velocity (vx, vy, vz) (m/s)
        [13:16] -> angular velocity (wx, wy, wz) (rad/s)
        [16:19] -> gate position (x, y, z) (m)
        [19:23] -> gate orientation (quaternion x, y, z, w)
        """

        state = self._getDroneStateVector(0)
        pos = state[0:3] 
        rpy = state[7:10]
        vel = state[10:13]
        ang_vel = state[13:16]

        rel_pos_to_gate = self.CENTER_GATE - pos

        if self.use_curriculum:
            obs = np.concatenate(
                [
                    pos, # 3
                    vel, # 3
                    rpy, # 3
                    ang_vel,   # 3
                    rel_pos_to_gate,  # 3
                    self.GATE_ORN,  # 4
                ]
            ).astype(np.float32)
        else:
            obs = np.concatenate(
                [
                    pos, # 3
                    vel, # 3
                    rpy, # 3
                    ang_vel,   # 3
                    rel_pos_to_gate,
                    self.GATE_ORN,  # 4
                ]
            ).astype(np.float32)
        return obs

    def _computeReward(self):

        # """Reward for gate navigation task"""
        state = self._getDroneStateVector(0) 
        pos = state[0:3]
        vel = state[10:13]  # [vx, vy, vz]\

        distance = np.linalg.norm(pos - self.CENTER_GATE)
        distance_reward = np.exp(-2.0 * distance)

        progress = np.dot(vel, self.gate_normal)  # Velocity component 
        progress_reward = max(0, progress) 

        self.center_gate_passed = (pos[1] < self.CENTER_GATE[1]) and (distance < 0.25)

        if self.center_gate_passed and self._check_collision_with_gate(0)==False:
            reward = 10.0 
            return reward

        if self._check_collision_with_gate(0):
            print(f"Collision with gate detected at position: {state[0:3]}")
            return -10.0  # Collision penalty

        return distance_reward + progress_reward

    def _check_collision_with_gate(self, drone_idx=0) -> bool:
        drone_id = self.DRONE_IDS[drone_idx]
        
        cps = p.getContactPoints(
            bodyA=drone_id,
            bodyB=self.GATE_ID,
            physicsClientId=self.CLIENT
        )
        
        actual_collisions = [cp for cp in cps if cp[8] <= 0.0]  

        # body = self.DRONE_IDS[drone_idx]         
        # aabb_min, aabb_max = p.getAABB(body, linkIndex=-1, physicsClientId=self.CLIENT)
        # size = np.array(aabb_max) - np.array(aabb_min)  # [x, y, z] (m)
        # print("Drone size (AABB):", size)

    
        return len(actual_collisions) > 0

    def _computeTruncated(self):
        """Compute the truncated flag for the current step.

        Returns:
            bool: The truncated flag.
        """
        state = self._getDroneStateVector(0)
        if self._check_collision_with_gate(drone_idx=0):
            return True
        
        # Out of bounds termination
        if np.linalg.norm(state[0:2] - self.CENTER_GATE[0:2]) > 5.0 or state[2] > 5.0:  # Too far from gate position
            return True
        
        # Excessive pitch or roll
        if abs(state[7]) > np.pi/3 or abs(state[8]) > np.pi/3:  
            return True
        
        # Ground collision 
        if abs(state[2]) < 0.15:
            return True
        
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False
        
    def _computeTerminated(self):
        """Compute the terminated flag for the current step.

        Returns:
            bool: The terminated flag.
        """
        # Success condition: passed through the gate and close to final target
        state = self._getDroneStateVector(0)

        # Go to waypoint and through center gate 
        if self.center_gate_passed and self._check_collision_with_gate(drone_idx=0): 
            print(f"Success go through narrow space !")
            return True

        # Out of time termination
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False
    
    # For curriculum learning: adjust initial position based on curriculum level
    def reset(self, seed=None, options=None):
        
        self.center_gate_passed = False
        self.way_point_success = False
        INIT_X = np.array([0.0, 0.0, 0.0])
        # Make randomization for initial state 

        # Randomization of initial position around the gate
        # Random initial position near the gate
        x = np.random.uniform(-1.5, 1.5)
        y = np.random.uniform(-0.5, 1.5)
        z = np.random.uniform(0.3, 2.0)

        self.INIT_XYZS[0, :] = INIT_X + np.array([x, y, z])
        self.INIT_RPYS[0, 2] = 0.2 * (np.random.rand() - 0.5)
        self.INIT_RPYS[0, 0:2] = np.array([0.0, 0.0])
        
        if seed is not None:
            super().reset(seed=seed)

        if self.use_curriculum:
            lvl = min(self.curriculum_level, self.max_curriculum_level)

            # Clear curriculum structure
            if lvl == 0:
                # Level 0: Near the gate, no noise
                self.threshold_success = 0.17
                forward_offset = 0.2
                lateral_noise = 0.01 * (np.random.rand() - 0.5)
                vertical_noise = 0.01 * (np.random.rand() - 0.5)
                self.INIT_RPYS[0, 2] = 0.0
            elif lvl == 1:
                # Level 1: Near the gate, small noise
                forward_offset = 0.3
                lateral_noise = 0.05 * (np.random.rand() - 0.5)
                vertical_noise = 0.05 * (np.random.rand() - 0.5)
                self.INIT_RPYS[0, 2] = 0.0
                self.threshold_success = 0.15
            elif lvl == 2:
                forward_offset = 0.5
                lateral_noise = 0.05 * (np.random.rand() - 0.5)
                vertical_noise = 0.05 * (np.random.rand() - 0.5)
                self.INIT_RPYS[0, 2] = 0.0
                self.threshold_success = 0.13
            elif lvl == 3:
                forward_offset = 0.8
                lateral_noise = 0.1 * (np.random.rand() - 0.5)
                vertical_noise = 0.1 * (np.random.rand() - 0.5)
                self.INIT_RPYS[0, 2] = 0.0
                self.threshold_success = 0.11
            elif lvl == 4:
                forward_offset = 1.0
                lateral_noise = 0.1 * (np.random.rand() - 0.5)
                vertical_noise = 0.1 * (np.random.rand() - 0.5)
                self.INIT_RPYS[0, 2] = 0.0
                self.threshold_success = 0.1
            else:  # lvl >= 5 (max) - comment for testing without curriculum
                forward_offset = 1.2
                lateral_noise = 0.1 # 0.5 * (np.random.rand() - 0.5)
                vertical_noise = 0.1 #0.5 * (np.random.rand() - 0.5)
                self.threshold_success = 0.08
                # Can add random yaw orientation
                self.INIT_RPYS[0, 2] = 0.2 # 0.2 * (np.random.rand() - 0.5)
            
            self.INIT_XYZS[0, :] = self.CENTER_GATE + np.array([
                lateral_noise, 
                forward_offset, 
                vertical_noise
            ])
            self.INIT_RPYS[0, 0:2] = np.array([0.0, 0.0])

        return super().reset(seed=seed, options=options)
    
    def _computeInfo(self):
        state = self._getDroneStateVector(0)
        info = {
            "position": state[0:3],
            "quaternion": state[3:7],
            "rpy": state[7:10],
            "lin_vel": state[10:13],
            "ang_vel": state[13:16],
            "time_passed_gate": self.time_passed_gate,
        }
        return info
    
    def _clipAndNormalizeState(self,
                               state
                               ):
        """Normalizes a drone's state to the [-1,1] range.

        Parameters
        ----------
        state : ndarray
            (20,)-shaped array of floats containing the non-normalized state of a single drone.

        Returns
        -------
        ndarray
            (20,)-shaped array of floats containing the normalized state of a single drone.

        """
        MAX_LIN_VEL_XY = 3 
        MAX_LIN_VEL_Z = 1

        MAX_XY = MAX_LIN_VEL_XY*self.EPISODE_LEN_SEC
        MAX_Z = MAX_LIN_VEL_Z*self.EPISODE_LEN_SEC

        MAX_PITCH_ROLL = np.pi # Full range

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
                                               clipped_vel_z
                                               )

        normalized_pos_xy = clipped_pos_xy / MAX_XY
        normalized_pos_z = clipped_pos_z / MAX_Z
        normalized_rp = clipped_rp / MAX_PITCH_ROLL
        normalized_y = state[9] / np.pi # No reason to clip
        normalized_vel_xy = clipped_vel_xy / MAX_LIN_VEL_XY
        normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_XY
        normalized_ang_vel = state[13:16]/np.linalg.norm(state[13:16]) if np.linalg.norm(state[13:16]) != 0 else state[13:16]

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
    
    def _clipAndNormalizeStateWarning(self,
                                      state,
                                      clipped_pos_xy,
                                      clipped_pos_z,
                                      clipped_rp,
                                      clipped_vel_xy,
                                      clipped_vel_z,
                                      ):
        """Debugging printouts associated to `_clipAndNormalizeState`.

        Print a warning if values in a state vector is out of the clipping range.

        """
        if not(clipped_pos_xy == np.array(state[0:2])).all():
            print("[WARNING] it", self.step_counter, "in FlyThruGateAviary._clipAndNormalizeState(), clipped xy position [{:.2f} {:.2f}]".format(state[0], state[1]))
        if not(clipped_pos_z == np.array(state[2])).all():
            print("[WARNING] it", self.step_counter, "in FlyThruGateAviary._clipAndNormalizeState(), clipped z position [{:.2f}]".format(state[2]))
        if not(clipped_rp == np.array(state[7:9])).all():
            print("[WARNING] it", self.step_counter, "in FlyThruGateAviary._clipAndNormalizeState(), clipped roll/pitch [{:.2f} {:.2f}]".format(state[7], state[8]))
        if not(clipped_vel_xy == np.array(state[10:12])).all():
            print("[WARNING] it", self.step_counter, "in FlyThruGateAviary._clipAndNormalizeState(), clipped xy velocity [{:.2f} {:.2f}]".format(state[10], state[11]))
        if not(clipped_vel_z == np.array(state[12])).all():
            print("[WARNING] it", self.step_counter, "in FlyThruGateAviary._clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(state[12]))
