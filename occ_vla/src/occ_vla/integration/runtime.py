"""Main control loop: perceive -> route occlusion -> resolve (WM or
PKLP, with uncertainty fallback) -> condition policy -> act.

Segmentation and target tracking (where the arm is, which patch is the
target) are separate concerns this loop doesn't implement — the caller
supplies them each step via `PerceptionInputs`.
"""

from dataclasses import dataclass

import numpy as np

from occ_vla.control.occlusion_gating import apply_soft_gate, gate_scale
from occ_vla.control.pi05_policy import Pi05Observation, Pi05Policy
from occ_vla.integration.occlusion_router import OcclusionRouter, OcclusionSignals, OcclusionSource
from occ_vla.integration.uncertainty import PlausibilityChecker
from occ_vla.pklp.kinematics import KinematicExtrapolator, KinematicState
from occ_vla.pklp.optical_flow import RaftFlowEstimator
from occ_vla.pklp.pixel_to_action import CameraProjector, pklp_pixel_delta_to_world_delta
from occ_vla.pklp.visual_overlay import draw_kinematic_overlay
from occ_vla.world_model.arm_free_subgoal import ArmFreeSubgoalGenerator
from occ_vla.world_model.cot_anchor import CotAnchorGenerator

FRAME_HISTORY_LEN = 3  # RAFT needs (t-2, t-1, t) for the two flow samples kinematics.py::estimate_state wants

# robosuite's default OSC_POSE controller (confirmed 2026-07-18: LIBERO/our
# env wrapper never overrides controller_configs, so this default is what's
# actually active) maps a normalized action component in [-1, 1] linearly to
# a +-0.05m per-step end-effector position delta (`output_max`/`output_min`
# in robosuite.controllers.load_controller_config(default_controller="OSC_POSE")).
# pixel_to_action.pixel_delta_to_world_delta returns a delta in meters, so it
# must be divided by this to land in the same normalized units as pi0.5's
# own action output before blending.
OSC_POSE_MAX_DELTA_M = 0.05

# Fixed blend weight for the SCENE-branch action blend (Plan 3). NOT
# derived from gate_scale(arm_s_occ): OcclusionRouter.route only ever
# reaches SCENE when arm_s_occ <= arm_occ_threshold (else it routes to
# SELF instead) -- exactly the range where gate_scale always returns 1.0,
# so `1 - gate_scale(arm_s_occ)` would be identically 0 here (caught by
# tests/integration/test_runtime.py while wiring this up). arm_s_occ
# measures the ARM's self-occlusion of the target; SCENE fires for a
# *different* occlusion source (perception.scene_dyn_occ), which the
# current perception contract only reports as a bool, with no continuous
# severity signal to scale alpha by. Fixed at 0.5 (untested; a knob to
# tune once the n>=10 paired comparison actually runs) until a real
# scene-occlusion severity metric exists.
SCENE_BLEND_ALPHA = 0.5


def gated_blend_xy(vla_xy: np.ndarray, pklp_xy: np.ndarray, alpha: float = SCENE_BLEND_ALPHA) -> np.ndarray:
    """Conflict-avoidance gate (found 2026-07-18 via
    scripts/diagnose_blend_vector_conflict.py on mug_in_microwave, see
    occ_vla/CLAUDE.md): only blend toward pklp_xy when it still agrees
    with pi0.5's own direction (cos_angle > 0); otherwise trust pi0.5's
    own action_xy unmodified.

    The diagnostic showed both vectors stay near full strength through
    an occluded episode (measured ~0.5-0.6 magnitude each -- this is NOT
    pi0.5 losing confidence/going low-norm near the occlusion boundary),
    but PKLP's target is `last_known_position`, frozen at the moment
    occlusion began; for an *insertion* motion (e.g. mug_in_microwave)
    the arm correctly keeps moving past that frozen 2D point, so
    cos_angle drifts from agreement (+) to opposition (-) the longer
    occlusion persists. A naive fixed-alpha blend of two opposing
    full-strength vectors destructively cancels the net commanded
    motion (measured net norm ~0.04-0.10 vs. ~0.5-0.6 unblended),
    freezing the arm right at the boundary -- this is the confirmed
    mechanism behind the "stuck hovering" failure. Gating on cos_angle
    keeps PKLP's correction only while it's still reinforcing, not
    fighting, pi0.5's own intent."""
    vla_norm = np.linalg.norm(vla_xy)
    pklp_norm = np.linalg.norm(pklp_xy)
    if vla_norm < 1e-8 or pklp_norm < 1e-8:
        return vla_xy
    cos_angle = np.dot(vla_xy, pklp_xy) / (vla_norm * pklp_norm)
    if cos_angle <= 0:
        return vla_xy
    return (1.0 - alpha) * vla_xy + alpha * pklp_xy


@dataclass
class ControlLoopComponents:
    policy: Pi05Policy
    subgoal_generator: ArmFreeSubgoalGenerator
    cot_generator: CotAnchorGenerator
    flow_estimator: RaftFlowEstimator
    kinematic_extrapolator: KinematicExtrapolator
    router: OcclusionRouter
    plausibility_checker: PlausibilityChecker
    # MMaDA's arm-free subgoal image generation has an unresolved
    # generation-quality problem (see occ_vla/CLAUDE.md, "MMaDA arm-free
    # generation quality investigation" -- 6+ independent attempts across
    # schedule/CFG/prompt/mask-geometry variations all produced the same
    # collapsed-blob artifact on the held object). Defaults to off: the
    # SELF branch below uses a text-only cot_anchor (same mechanism as
    # the SCENE branch) instead of depending on that broken image path.
    # Flip to True only to re-test a specific generation-quality fix.
    enable_subgoal_image_generation: bool = False
    # MOKA-style marker (pklp/visual_overlay.py) drawn on obs.base_image
    # at PKLP's kinematic-extrapolated predicted position, for the SCENE
    # (scene-induced occlusion) branch. Non-generative and cheap, so on
    # by default unlike enable_subgoal_image_generation above.
    enable_visual_overlay: bool = True
    # Plan 3 (see occ_vla/CLAUDE.md, "camera calibration validated,
    # Jacobian-based pixel->action bridge" -- 2026-07-18): blend pi0.5's
    # own XY translation output with a PKLP-derived correction in the
    # SCENE branch, weighted by a fixed alpha (SCENE_BLEND_ALPHA; see its
    # docstring for why this isn't arm_s_occ-derived). Geometry-validated
    # (camera projection, Jacobian inversion, direction sanity) but not
    # yet empirically tested for task success rate -- defaults to False,
    # same caution as enable_subgoal_image_generation, until an n>=10
    # paired comparison against baseline pi0.5 has actually run.
    enable_action_blending: bool = False


@dataclass
class PerceptionInputs:
    """Per-step upstream perception this loop consumes but doesn't
    produce itself."""

    arm_pixel_mask: np.ndarray  # HxW bool, current frame
    arm_s_occ: float  # fraction of the target occluded by the arm
    scene_dyn_occ: bool  # a moving scene object is occluding the target right now
    target_patch_idx: int | None  # index into the RAFT 16x16 patch grid tracking the target; None if never seen clearly
    # Only needed when ControlLoopComponents.enable_action_blending is
    # True; None is safe (blending is skipped) otherwise. camera_projector
    # is static per episode (fixed camera pose) so the caller can build it
    # once per reset via CameraProjector.from_sim(sim, "agentview",
    # resolution=<obs.base_image side>) and pass it through every step.
    camera_projector: CameraProjector | None = None
    eef_pos_world: np.ndarray | None = None  # (3,), current end-effector world position


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
        pklp_action_delta_xy: np.ndarray | None = None  # normalized [-1, 1] units, set below if blending fires

        signals = OcclusionSignals(arm_s_occ=perception.arm_s_occ, scene_dyn_occ=perception.scene_dyn_occ)
        source = self.c.router.route(signals)

        if source == OcclusionSource.SELF:
            if self.c.enable_subgoal_image_generation:
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
            else:
                # Text-only CoT anchor instead of image generation --
                # see enable_subgoal_image_generation's docstring above.
                obs.cot_anchor = self.c.cot_generator.generate(
                    f"{obs.prompt}. The robot's own arm currently occludes the target from the main "
                    f"camera (S_occ={perception.arm_s_occ:.2f}); reason about the expected next state "
                    f"using the wrist camera view.",
                    history=[],
                )

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
                if self.c.enable_visual_overlay:
                    obs.base_image = draw_kinematic_overlay(obs.base_image, state.position, predicted_position)

                if self.c.enable_action_blending and perception.camera_projector is not None and perception.eef_pos_world is not None:
                    # state.position/predicted_position are in obs.base_image's
                    # pixel space (pushed to frame_history unmodified above) --
                    # same convention perception.camera_projector must be built
                    # with (see PerceptionInputs.camera_projector docstring).
                    world_delta_m = pklp_pixel_delta_to_world_delta(
                        perception.camera_projector,
                        perception.eef_pos_world,
                        state.position,
                        predicted_position,
                    )
                    pklp_action_delta_xy = world_delta_m[:2] / OSC_POSE_MAX_DELTA_M
            # else: not enough frame history yet (e.g. right after the
            # pseudo-anchor seed above) — nothing more PKLP can do this
            # step; fall through to the policy on the raw observation.

        # Soft gating (Phase 4): attenuate, don't erase, the agentview
        # frame as arm_s_occ rises -- applied after frame_history/flow
        # use the real pixels above, so RAFT/kinematics never see a
        # dimmed frame, only the policy's own input does.
        scale = gate_scale(perception.arm_s_occ)
        if scale < 1.0:
            obs.base_image = apply_soft_gate(obs.base_image, scale)

        actions = self.c.policy.step(obs)

        if pklp_action_delta_xy is not None:
            # Plan 3 blend, applied only to the immediately-next action
            # (actions[0]) -- the rest of the chunk corresponds to future
            # steps where this step's PKLP geometry no longer applies; the
            # caller is expected to re-run this (and thus re-blend) on
            # every replan, same as it already does for actions[0] alone.
            # See SCENE_BLEND_ALPHA's docstring for why this is a fixed
            # weight, not gate_scale(arm_s_occ) (that would be identically
            # 0 whenever this branch can even run). gated_blend_xy adds
            # the conflict-avoidance gate found 2026-07-18 (see its
            # docstring) on top of the fixed-alpha blend.
            blended = gated_blend_xy(actions[0, :2], pklp_action_delta_xy, SCENE_BLEND_ALPHA)
            actions[0, :2] = np.clip(blended, -1.0, 1.0)

        return actions
