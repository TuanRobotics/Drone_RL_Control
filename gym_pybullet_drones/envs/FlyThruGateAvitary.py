#!/usr/bin/env python3

import os
import numpy as np
import pybullet as p
import pkg_resources

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
                record=False,
                obs: ObservationType = ObservationType.KIN,
                act: ActionType = ActionType.RPM
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
        self.EPISODE_LEN_SEC = 16  # Episode length in seconds
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
            act=act
        )

        self.GATE_POS = np.array([0, -2, 0.3])  # Center of gate
        self.FINAL_TARGET = np.array([0, -3, 0.5])
        self.passed_gate = False  # Flag for track if gate is passed
        self.GATE_ORN = None

        self.success_passed = False

    ##############################################################################
    # Load urdf to create the gate 
    def _addObstacles(self):

        super()._addObstacles()
        boxId = p.loadURDF(pkg_resources.resource_filename('gym_pybullet_drones', 'assets/gate.urdf'),
                   [0.0, -1.5, 0.3],
                   p.getQuaternionFromEuler([0, 0, 1.5]),
                   physicsClientId=self.CLIENT
                   )
        self.GATE_POS, self.GATE_ORN = p.getBasePositionAndOrientation(boxId)
    ##############################################################################
    # def _computeReward(self):
        # """Reward for gate navigation task"""
        # state = self._getDroneStateVector(0)
        # norm_ep_time = (self.step_counter/self.PYB_FREQ) / self.EPISODE_LEN_SEC
        # return max (0, 1 - np.linalg.norm(np.array([0, -3*norm_ep_time, 0.75])-state[0:3]))
    def _computeReward(self): 

        """Reward for gate navigation task"""
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        vel = state[10:13]  # [vx, vy, vz]
        
        # Position of gate and its normal vector
        gate_pos = self.GATE_POS  # x, y, z của gate 
        # print("Gate position:", gate_pos)
        gate_normal = np.array([0.0, -1.0, 0.0])  # Direction 
        
        # 1. Distance reward
        way_point = gate_pos  + np.array([0.0, -0.5, 0.25])  # Waypoint at the center of the gate
        self.FINAL_TARGET = way_point
        # print(f"Way point:", way_point)
        distance = np.linalg.norm(pos - way_point)  # Euclidean distance to waypoint allway > 0 
        distance_reward = np.exp(-2.0 * distance) # max : 1 when distance = 0
        
        # 2. Progress reward - reward for moving towards the gate : reward shaping
        progress = np.dot(vel, gate_normal)  # Velocity component 
        progress_reward = max(0, progress)   # max 0 if moving away from gate
        
        #3. Alignment reward - reward for going straight through the gate
        vel_norm = np.linalg.norm(vel)
        if vel_norm > 0.1:
            alignment = np.dot(vel / vel_norm, gate_normal) # max 1 when fully aligned
            alignment_reward = max(0, alignment)
        else:
            alignment_reward = 0
        
        # 4. Gate passing bonus - reward for passing through the gate
        passed_gate = np.linalg.norm(pos - way_point) < 0.1 and pos[1] <= way_point[1]  # Passed y of gate and close to gate
        if passed_gate:
            reward = 10.0 
            self.success_passed = True
            return reward
        
        # 5. Penalty for collision or going out of bounds
        out_of_bounds = abs(pos[0]) > 2.0 or abs(pos[1]) > 3.0 or pos[2] < 0.1 or pos[2] > 2.0
        if out_of_bounds:
            reward = -10.0
            return reward
        
        # Total reward (can tune coefficients)
        reward = (
            2.0 * distance_reward +      
            1.0 * progress_reward + 0.1*alignment_reward       
        )
        return reward
    
    def _computeTruncated(self):
        """Compute the truncated flag for the current step.

        Returns:
            bool: The truncated flag.
        """
        state = self._getDroneStateVector(0)
        if (abs(state[0]) > 2.0 or abs(state[1]) > 3.0 or state[2] > 2.0 # Truncate when the drone is too far away
             or abs(state[7]) > .4 or abs(state[8]) > .4 # Truncate when the drone is too tilted
        ):
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
        distance = np.linalg.norm(state[0:3] - self.FINAL_TARGET)
        # Success condition: close to final target
        if distance < 0.2:
            self.success_passed = True
            return True
        
        # Out of time termination
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False
    
    def _computeInfo(self):
        state = self._getDroneStateVector(0)
        info = {
            "position": state[0:3],
            "quaternion": state[3:7],
            "rpy": state[7:10],
            "lin_vel": state[10:13],
            "ang_vel": state[13:16],
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