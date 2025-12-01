# Env drone racing through gates
# Reward: progress towards next gate + bonus for passing through gate
# Termination: collision or timeout
import os
import numpy as np
import pybullet as p
import pkg_resources
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, ObservationType, ActionType, Physics

class DroneRacingAvitary(BaseRLAviary):
    """Drone Racing environment class for multiple drones racing through gates in a PyBullet simulation.

    The environment is initialized with a specified number of drones, observation type, action type, and physics engine.
    The drones are rewarded based on their progress towards the next gate and receive a bonus for passing through gates.
    The episode terminates upon collision or timeout.

    Attributes:
        NUM_DRONES (int): Number of drones in the environment.
        OBS_TYPE (ObservationType): Type of observation (kinematic or RGB).
        ACT_TYPE (ActionType): Type of action (RPM, PID, etc.).
        PHYSICS (Physics): Physics engine used for simulation.
    """

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 48,
                 gui=False,
                 record= False,
                 obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.RPM):
        
        super().__init__(
            drone_model=drone_model,
            num_drones=1,
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
        self.EPISODE_LEN_SEC = 10  # Episode length in seconds
        # Additional initialization code specific to drone racing can be added here
        self.collided = False

    def _addObstacles(self):
        super()._addObstacles()
        g1 = p.loadURDF(pkg_resources.resource_filename('gym_pybullet_drones', 'assets/gate.urdf'),
                   [0, 1, 0],
                   p.getQuaternionFromEuler([0, 0, np.pi/2]),
                   physicsClientId=self.CLIENT
                   )
        g2 = p.loadURDF(pkg_resources.resource_filename('gym_pybullet_drones', 'assets/gate.urdf'),
                   [0.5, 3, 0],
                   p.getQuaternionFromEuler([0, 0, np.pi/2]),
                   physicsClientId=self.CLIENT
                   )
        g3 = p.loadURDF(pkg_resources.resource_filename('gym_pybullet_drones', 'assets/gate.urdf'),
                   [-0.5, 5, 0],
                   p.getQuaternionFromEuler([0, 0, np.pi/2]),
                   physicsClientId=self.CLIENT
                   )
        g4 = p.loadURDF(pkg_resources.resource_filename('gym_pybullet_drones', 'assets/gate.urdf'),
                   [0.5, 7, 0],
                   p.getQuaternionFromEuler([0, 0, np.pi/2]),
                   physicsClientId=self.CLIENT
                   )
        g5 = p.loadURDF(pkg_resources.resource_filename('gym_pybullet_drones', 'assets/gate.urdf'),
                   [1.0, 9, 0],
                   p.getQuaternionFromEuler([0, 0, np.pi/2]),
                   physicsClientId=self.CLIENT
                   )

        self.GATE_IDs = np.array([g1, g2, g3, g4, g5])

    def _computeReward(self):

        state = self._getDroneStateVector(0)
        pos = state[0:3]
        rpy = state[7:10]
        ang_vel=state[13:16]

        found_gate = False # Check if found gate to compute progress
        for i, key in enumerate(self.racing_setup.keys()):
            if self.passing_flag[i]:
                continue
            cur_dist = np.linalg.norm(np.array(self.racing_setup[key][0]) - pos)
            prev_dist = np.linalg.norm(np.array(self.racing_setup[key][0]) - self.prev_pos)
            found_gate = True
            break
        if not found_gate:
            # all gates passed
            return 0.0, True
        
        self.prev_pos = state[:3]
        on_way_reward = prev_dist - cur_dist - 0.001 * np.linalg.norm(ang_vel)
        passing = False 

        for i, key in enumerate(self.racing_setup.keys()):
            if self.passing_flag[i]:
                continue
            # Check x y z within gate bounds 
            """
            x_left = self.racing_setup[key][1][0]
            x_right = self.racing_setup[key][2][0]
            y_bottom = self.racing_setup[key][1][1] - 0.5
            y_top = self.racing_setup[key][2][1] + 0.5
            z_lower = self.racing_setup[key][4][2]
            z_upper = self.racing_setup[key][1][2]
            if x_left < pos[0] < x_right and y_bottom < pos[1] < y_top and z_lower < pos[2] < z_upper:
            """
            if self.racing_setup[key][1][0] < pos[0] < self.racing_setup[key][2][0] and \
               self.racing_setup[key][1][1] - 0.5 < pos[1] < self.racing_setup[key][2][1] + 0.5 and \
               self.racing_setup[key][4][2] < pos[2] < self.racing_setup[key][1][2]:
                self.passing_flag[i] = True
                passing = True
                break

        self.collide = False
        for i in range(4):
            contact_points = p.getContactPoints(bodyA=self.DRONE_IDS[0],
                       bodyB=self.GATE_IDs[i],
                       physicsClientId=self.CLIENT
                       )
            if len(contact_points) != 0:
                self.collide = True
                break
        if passing:
            print("passing")
            print(state[:3])
            print(self.passing_flag)
            reward = 10
        elif self.collide:
            print("collision")
            reward = -10
        else:
            reward = on_way_reward
        
        return reward 


    def _computeTruncated(self):

        state = self._getDroneStateVector(0)
        # Truncate when the drone is too far away
        if (abs(state[0]) > self.w/2 + 0.5 + 2*self.offset or state[1] > 6 or state[1] < -2 * self.offset \
        or state[3] > self.h + 2 * self.offset
        ):
            if self.passing_flag[0]:
                print('trucated')
                print(state[:3])
            return True
        if abs(state[7]) > 0.78 or abs(state[8]) > 0.78: # Truncate when the drone is too tilted
            if self.passing_flag[0]:
                print('trucated by tilted')
                print(state[:3])
            return True
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False
    
    def _computeTerminated(self):

        if  self.passing_flag[4] == True or self.collide == True:
            return True
        else:
            return False
        
    def _computeInfo(self):
        info = {}
        info['passing_flag'] = self.passing_flag
        return info
    
    def _clipAndNormalizeState(self,state):
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

    ################################################################################

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

        
    
   