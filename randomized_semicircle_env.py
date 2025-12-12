"""
Domain-randomized wrapper for SemiCircleRacingAviary.

Randomizes physical parameters each reset to improve sim2real robustness:
- Drone mass and inertia scaling
- Motor thrust/torque coefficients
- Drag coefficients
- External wind
- Track layout randomization (optional pass-through)
"""

import numpy as np
import pybullet as p

from gym_pybullet_drones.envs import SemiCircleRacingAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

# class for randomized env:
"""
Randomized SemiCircle Racing Aviary environment with domain randomization.
we can specify ranges for various physical parameters to be randomized at each reset.
And we can also randomize the track layout by passing options during reset.
"""
class RandomizedSemiCircleAviary(SemiCircleRacingAviary):
    def __init__(self,
                 gui: bool = False,
                 record: bool = False,
                 obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.RPM,
                 rand_cfg: dict | None = None,
                 **kwargs):
        self.rand_cfg = rand_cfg or {}
        super().__init__(gui=gui, record=record, obs=obs, act=act, **kwargs)

    def reset(self, seed=None, options=None):
        options = options or {}
        self._apply_domain_randomization()
        return super().reset(seed=seed, options=options)

    def _apply_domain_randomization(self):
        cfg = self.rand_cfg
        c = lambda key, default: cfg.get(key, default)
        # Mass/inertia scaling
        mass_scale = float(np.random.uniform(*c("mass_scale_range", (0.9, 1.1))))
        inertia_scale = float(np.random.uniform(*c("inertia_scale_range", (0.9, 1.1))))
        # Motor coeffs
        kf_scale = float(np.random.uniform(*c("kf_scale_range", (0.9, 1.1))))
        km_scale = float(np.random.uniform(*c("km_scale_range", (0.9, 1.1))))
        # Drag
        drag_scale = float(np.random.uniform(*c("drag_scale_range", (0.8, 1.2))))
        # Wind
        wind_mag = float(np.random.uniform(*c("wind_mag_range", (0.0, 1.0))))
        wind_yaw = float(np.random.uniform(-np.pi, np.pi))
        wind = np.array([np.cos(wind_yaw), np.sin(wind_yaw), 0.0]) * wind_mag

        # Apply to physics (only one drone in this env)
        drone_id = self.DRONE_IDS[0]
        mass = p.getDynamicsInfo(drone_id, -1, physicsClientId=self.CLIENT)[0]
        p.changeDynamics(drone_id, -1,
                         mass=mass * mass_scale,
                         physicsClientId=self.CLIENT)
        # Scale inertia (principal moments)
        dyn = p.getDynamicsInfo(drone_id, -1, physicsClientId=self.CLIENT)
        inertia = np.array(dyn[2])
        p.changeDynamics(drone_id, -1,
                         localInertiaDiagonal=(inertia * inertia_scale).tolist(),
                         physicsClientId=self.CLIENT)
        # Motor thrust/torque coefficients inside env
        self.KF *= kf_scale
        self.KM *= km_scale
        # Drag (array [xy, xy, z])
        if hasattr(self, "DRAG_COEFF"):
            self.DRAG_COEFF = self.DRAG_COEFF * drag_scale
        # Wind force applied as external force at base
        p.applyExternalForce(drone_id, -1, forceObj=wind.tolist(),
                             posObj=[0, 0, 0], flags=p.WORLD_FRAME,
                             physicsClientId=self.CLIENT)


def make_randomized_env(seed: int = 0, rand_cfg: dict | None = None):
    rng = np.random.default_rng(seed)
    def _init():
        env = RandomizedSemiCircleAviary(
            gui=False,
            record=False,
            obs=ObservationType.KIN,
            act=ActionType.RPM,
            rand_cfg=rand_cfg,
        )
        env._rng = rng
        return env
    return _init
