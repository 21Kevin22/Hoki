"""Dynamic exposure: rather than only inferring the hidden object's
future position, plan an arm displacement toward the direction the
occluding object is moving away from, to visually re-expose the target
sooner."""

from dataclasses import dataclass

import numpy as np

from occ_vla.pklp.kinematics import KinematicState


@dataclass
class ExposureWaypoint:
    delta_ee_position: np.ndarray  # (3,) end-effector offset to apply


class DynamicExposurePlanner:
    def plan(self, occluder_state: KinematicState, current_ee_position: np.ndarray) -> ExposureWaypoint:
        raise NotImplementedError
