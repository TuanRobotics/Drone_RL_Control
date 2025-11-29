# Env drone racing through gates
# Reward: progress towards next gate + bonus for passing through gate
# Termination: collision or timeout
import os
import numpy as np
import pybullet as p
import pkg_resources
from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, ObservationType, ActionType, Physics

class DronceRacingAvitary(BaseAviary):
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
        # Additional initialization code specific to drone racing can be added here

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
                   [0.5, 6, 0],
                   p.getQuaternionFromEuler([0, 0, np.pi/2]),
                   physicsClientId=self.CLIENT
                   )
        g5 = p.loadURDF(pkg_resources.resource_filename('gym_pybullet_drones', 'assets/gate.urdf'),
                   [1.0, 7, 0],
                   p.getQuaternionFromEuler([0, 0, np.pi/2]),
                   physicsClientId=self.CLIENT
                   )

        self.GATE_IDs = np.array([g1, g2, g3, g4, g5])

    def _computeReward(self):
        pass 

    def _computeTruncated(self):
        pass
    def _computeInfo(self):
        pass
    def _clipAndNormalizeState(self,tate):
        pass 
    
        
    
   