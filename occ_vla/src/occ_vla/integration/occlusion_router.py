"""Routes between the world model (self-occlusion) and PKLP (scene-induced
dynamic occlusion), with a WM -> PKLP fallback on high uncertainty."""

from dataclasses import dataclass
from enum import Enum, auto

from occ_vla.world_model.arm_free_subgoal import ARM_OCC_THRESHOLD


@dataclass
class OcclusionSignals:
    arm_s_occ: float  # fraction of target occluded by the arm itself
    scene_dyn_occ: bool  # a moving scene object is currently occluding the target


class OcclusionSource(Enum):
    NONE = auto()
    SELF = auto()  # -> world model
    SCENE = auto()  # -> PKLP


class OcclusionRouter:
    def __init__(self, arm_occ_threshold: float = ARM_OCC_THRESHOLD):
        self.arm_occ_threshold = arm_occ_threshold

    def route(self, signals: OcclusionSignals) -> OcclusionSource:
        if signals.arm_s_occ > self.arm_occ_threshold:
            return OcclusionSource.SELF
        if signals.scene_dyn_occ:
            return OcclusionSource.SCENE
        return OcclusionSource.NONE
