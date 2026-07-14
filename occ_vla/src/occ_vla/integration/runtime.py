"""Main control loop: perceive -> route occlusion -> resolve (WM or
PKLP, with uncertainty fallback) -> condition policy -> act.

Segmentation and target tracking (where the arm is, which patch is the
target) are separate concerns this loop doesn't implement — the caller
supplies them each step via `PerceptionInputs`.
"""

from dataclasses import dataclass

import numpy as np

from occ_vla.control.pi05_policy import Pi05Observation, Pi05Policy
from occ_vla.integration.occlusion_router import OcclusionRouter, OcclusionSignals, OcclusionSource
from occ_vla.integration.uncertainty import PlausibilityChecker
from occ_vla.pklp.kinematics import KinematicExtrapolator, KinematicState
from occ_vla.pklp.optical_flow import RaftFlowEstimator
from occ_vla.world_model.arm_free_subgoal import ArmFreeSubgoalGenerator
from occ_vla.world_model.cot_anchor import CotAnchorGenerator

FRAME_HISTORY_LEN = 3  # RAFT needs (t-2, t-1, t) for the two flow samples kinematics.py::estimate_state wants


@dataclass
class ControlLoopComponents:
    policy: Pi05Policy
    subgoal_generator: ArmFreeSubgoalGenerator
    cot_generator: CotAnchorGenerator
    flow_estimator: RaftFlowEstimator
    kinematic_extrapolator: KinematicExtrapolator
    router: OcclusionRouter
    plausibility_checker: PlausibilityChecker


@dataclass
class PerceptionInputs:
    """Per-step upstream perception this loop consumes but doesn't
    produce itself."""

    arm_pixel_mask: np.ndarray  # HxW bool, current frame
    arm_s_occ: float  # fraction of the target occluded by the arm
    scene_dyn_occ: bool  # a moving scene object is occluding the target right now
    target_patch_idx: int | None  # index into the RAFT 16x16 patch grid tracking the target; None if never seen clearly


class ControlLoop:
    def __init__(self, components: ControlLoopComponents):
        self.c = components
        self._frame_history: list[np.ndarray] = []
        self._last_clear_state: KinematicState | None = None

    def _push_frame(self, image: np.ndarray) -> None:
        self._frame_history.append(image)
        if len(self._frame_history) > FRAME_HISTORY_LEN:
            self._frame_history.pop(0)

    def step(self, obs: Pi05Observation, perception: PerceptionInputs) -> np.ndarray:
        self._push_frame(obs.base_image)

        signals = OcclusionSignals(arm_s_occ=perception.arm_s_occ, scene_dyn_occ=perception.scene_dyn_occ)
        source = self.c.router.route(signals)

        if source == OcclusionSource.SELF:
            subgoal = self.c.subgoal_generator.sample_arm_free_image(
                obs.base_image, perception.arm_pixel_mask, obs.prompt
            )
            physical_context = {"original_image": obs.base_image, "arm_pixel_mask": perception.arm_pixel_mask}
            if self.c.plausibility_checker.should_fallback(subgoal.image, physical_context):
                source = OcclusionSource.SCENE  # fall through to PKLP below
            else:
                obs.subgoal_image = subgoal.image
                if self._last_clear_state is None:
                    # Pseudo-anchor: seed PKLP's frame history from the
                    # world model's arm-free image instead of leaving
                    # last_clear at None, so a *subsequent* scene
                    # occlusion still has somewhere to extrapolate from.
                    self._push_frame(subgoal.image)

        if source == OcclusionSource.SCENE:
            have_enough_history = perception.target_patch_idx is not None and len(self._frame_history) >= FRAME_HISTORY_LEN
            if have_enough_history:
                t2, t1, t0 = self._frame_history[-3:]
                flow_earlier, flow_latest = self.c.flow_estimator.three_frame_patch_flow(t2, t1, t0)
                state = self.c.kinematic_extrapolator.estimate_state(
                    [flow_earlier, flow_latest], perception.target_patch_idx
                )
                self._last_clear_state = state
                predicted_position = self.c.kinematic_extrapolator.extrapolate(state)
                obs.cot_anchor = self.c.cot_generator.generate(
                    f"The occluded target's estimated position is now {predicted_position.tolist()}.",
                    history=[],
                )
            # else: not enough frame history yet (e.g. right after the
            # pseudo-anchor seed above) — nothing more PKLP can do this
            # step; fall through to the policy on the raw observation.

        return self.c.policy.step(obs)
