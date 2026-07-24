import numpy as np

from occ_vla.control.pi05_policy import Pi05Observation
from occ_vla.integration.occlusion_router import OcclusionRouter
from occ_vla.integration.runtime import OSC_POSE_MAX_DELTA_M, ControlLoop, ControlLoopComponents, PerceptionInputs
from occ_vla.pklp.kinematics import KinematicState
from occ_vla.pklp.pixel_to_action import CameraProjector
from occ_vla.world_model.arm_free_subgoal import SubgoalResult


class _FakePolicy:
    def __init__(self):
        self.last_obs = None
        self.calls = 0

    def step(self, obs):
        self.calls += 1
        self.last_obs = obs
        return np.array([[0.0] * 7])


class _FakeSubgoalGenerator:
    def __init__(self, image=None):
        self.image = image if image is not None else np.full((4, 4, 3), 9, dtype=np.uint8)
        self.calls = 0

    def sample_arm_free_image(self, image, arm_pixel_mask, instruction, horizon=5):
        self.calls += 1
        return SubgoalResult(image=self.image)


class _FakeCotGenerator:
    def __init__(self):
        self.calls = 0

    def generate(self, instruction, history):
        self.calls += 1
        return "anchor text"


class _FakeFlowEstimator:
    def __init__(self):
        self.calls = 0

    def three_frame_patch_flow(self, t2, t1, t0):
        self.calls += 1
        return "flow_earlier", "flow_latest"


class _FakeKinematicExtrapolator:
    def __init__(self):
        self.calls = 0

    def estimate_state(self, flow_history, target_patch_idx):
        self.calls += 1
        return KinematicState(position=np.array([1.0, 2.0]), velocity=np.zeros(2), acceleration=np.zeros(2))

    def extrapolate(self, state, steps=5):
        return state.position


class _FakePlausibilityChecker:
    def __init__(self, fallback=False):
        self._fallback = fallback

    def should_fallback(self, generated_image, physical_context):
        return self._fallback


def _obs():
    return Pi05Observation(
        base_image=np.zeros((4, 4, 3), dtype=np.uint8),
        wrist_image=np.zeros((4, 4, 3), dtype=np.uint8),
        state=np.zeros(8),
        prompt="pick up the pot",
    )


def _loop(fallback=False, enable_subgoal_image_generation=False):
    components = ControlLoopComponents(
        policy=_FakePolicy(),
        subgoal_generator=_FakeSubgoalGenerator(),
        cot_generator=_FakeCotGenerator(),
        flow_estimator=_FakeFlowEstimator(),
        kinematic_extrapolator=_FakeKinematicExtrapolator(),
        router=OcclusionRouter(),
        plausibility_checker=_FakePlausibilityChecker(fallback=fallback),
        enable_subgoal_image_generation=enable_subgoal_image_generation,
    )
    return ControlLoop(components)


def test_no_occlusion_calls_policy_directly():
    loop = _loop()
    perception = PerceptionInputs(
        arm_pixel_mask=np.zeros((4, 4), dtype=bool), arm_s_occ=0.0, scene_dyn_occ=False, target_patch_idx=None
    )
    loop.step(_obs(), perception)

    assert loop.c.policy.calls == 1
    assert loop.c.subgoal_generator.calls == 0
    assert loop.c.policy.last_obs.subgoal_image is None


def test_self_occlusion_uses_text_cot_by_default():
    # enable_subgoal_image_generation defaults to False -- MMaDA's
    # arm-free generation has an unresolved quality problem (blob
    # artifact), so the default SELF path no longer depends on it.
    loop = _loop()
    perception = PerceptionInputs(
        arm_pixel_mask=np.ones((4, 4), dtype=bool), arm_s_occ=0.5, scene_dyn_occ=False, target_patch_idx=None
    )
    loop.step(_obs(), perception)

    assert loop.c.subgoal_generator.calls == 0
    assert loop.c.policy.last_obs.subgoal_image is None
    assert loop.c.cot_generator.calls == 1
    assert loop.c.policy.last_obs.cot_anchor == "anchor text"


def test_self_occlusion_injects_subgoal_image_when_enabled():
    loop = _loop(enable_subgoal_image_generation=True)
    perception = PerceptionInputs(
        arm_pixel_mask=np.ones((4, 4), dtype=bool), arm_s_occ=0.5, scene_dyn_occ=False, target_patch_idx=None
    )
    loop.step(_obs(), perception)

    assert loop.c.subgoal_generator.calls == 1
    assert loop.c.policy.last_obs.subgoal_image is not None


def test_self_occlusion_falls_back_to_pklp_on_implausible_subgoal():
    loop = _loop(fallback=True, enable_subgoal_image_generation=True)
    perception = PerceptionInputs(
        arm_pixel_mask=np.ones((4, 4), dtype=bool), arm_s_occ=0.5, scene_dyn_occ=False, target_patch_idx=None
    )
    loop.step(_obs(), perception)

    # subgoal was generated but rejected -> not attached to obs
    assert loop.c.subgoal_generator.calls == 1
    assert loop.c.policy.last_obs.subgoal_image is None


def test_scene_occlusion_skips_pklp_without_enough_frame_history():
    loop = _loop()
    perception = PerceptionInputs(
        arm_pixel_mask=np.zeros((4, 4), dtype=bool), arm_s_occ=0.0, scene_dyn_occ=True, target_patch_idx=3
    )
    loop.step(_obs(), perception)  # only 1 frame pushed so far

    assert loop.c.flow_estimator.calls == 0
    assert loop.c.policy.last_obs.cot_anchor is None


def test_scene_occlusion_runs_pklp_once_history_is_full():
    loop = _loop()
    perception = PerceptionInputs(
        arm_pixel_mask=np.zeros((4, 4), dtype=bool), arm_s_occ=0.0, scene_dyn_occ=True, target_patch_idx=3
    )
    loop.step(_obs(), perception)
    loop.step(_obs(), perception)
    loop.step(_obs(), perception)  # 3rd frame -> history is now full

    assert loop.c.flow_estimator.calls == 1
    assert loop.c.kinematic_extrapolator.calls == 1
    assert loop.c.cot_generator.calls == 1
    assert loop.c.policy.last_obs.cot_anchor == "anchor text"
    # visual overlay (Phase 2) is on by default -- drawn at the fake
    # extrapolator's predicted point [1, 2] (xy) -> image[y=2, x=1]
    assert loop.c.policy.last_obs.base_image[2, 1].tolist() == [255, 0, 0]


def test_scene_occlusion_skips_overlay_when_disabled():
    components_kwargs = dict(
        policy=_FakePolicy(),
        subgoal_generator=_FakeSubgoalGenerator(),
        cot_generator=_FakeCotGenerator(),
        flow_estimator=_FakeFlowEstimator(),
        kinematic_extrapolator=_FakeKinematicExtrapolator(),
        router=OcclusionRouter(),
        plausibility_checker=_FakePlausibilityChecker(),
        enable_visual_overlay=False,
    )
    loop = ControlLoop(ControlLoopComponents(**components_kwargs))
    perception = PerceptionInputs(
        arm_pixel_mask=np.zeros((4, 4), dtype=bool), arm_s_occ=0.0, scene_dyn_occ=True, target_patch_idx=3
    )
    loop.step(_obs(), perception)
    loop.step(_obs(), perception)
    loop.step(_obs(), perception)

    assert loop.c.policy.last_obs.base_image[2, 1].tolist() == [0, 0, 0]


def test_self_occlusion_pseudo_anchors_frame_history_when_last_clear_is_none():
    loop = _loop(enable_subgoal_image_generation=True)
    perception = PerceptionInputs(
        arm_pixel_mask=np.ones((4, 4), dtype=bool), arm_s_occ=0.5, scene_dyn_occ=False, target_patch_idx=None
    )
    loop.step(_obs(), perception)
    # base frame + pseudo-anchor subgoal frame == 2 pushed in one step
    assert len(loop._frame_history) == 2  # noqa: SLF001


def test_soft_gate_attenuates_base_image_above_threshold():
    loop = _loop()
    obs = _obs()
    obs.base_image[:] = 200
    perception = PerceptionInputs(
        arm_pixel_mask=np.ones((4, 4), dtype=bool), arm_s_occ=0.8, scene_dyn_occ=False, target_patch_idx=None
    )
    loop.step(obs, perception)

    gated = loop.c.policy.last_obs.base_image
    assert gated.max() < 200
    assert gated.max() > 0  # attenuated, not zeroed


def test_soft_gate_is_noop_below_threshold():
    loop = _loop()
    obs = _obs()
    obs.base_image[:] = 200
    perception = PerceptionInputs(
        arm_pixel_mask=np.zeros((4, 4), dtype=bool), arm_s_occ=0.1, scene_dyn_occ=False, target_patch_idx=None
    )
    loop.step(obs, perception)

    assert (loop.c.policy.last_obs.base_image == 200).all()


class _FakeKinematicExtrapolatorWithDelta:
    """Unlike _FakeKinematicExtrapolator, predicted != current, so
    pklp_action_delta_xy in ControlLoop.step is nonzero -- needed to
    actually exercise the action-blending path below."""

    def __init__(self, current=(1.0, 2.0), predicted=(9.0, 6.0)):
        self.current = np.array(current)
        self.predicted = np.array(predicted)
        self.calls = 0

    def estimate_state(self, flow_history, target_patch_idx):
        self.calls += 1
        return KinematicState(position=self.current.copy(), velocity=np.zeros(2), acceleration=np.zeros(2))

    def extrapolate(self, state, steps=5):
        return self.predicted.copy()


class _FakePolicyWithAction:
    def __init__(self, action_xy=(0.2, -0.1)):
        self.action_xy = action_xy
        self.last_obs = None
        self.calls = 0

    def step(self, obs):
        self.calls += 1
        self.last_obs = obs
        action = np.zeros((1, 7))
        action[0, 0], action[0, 1] = self.action_xy
        return action


def _overhead_projector():
    return CameraProjector(cam_pos=np.array([0.0, 0.0, 1.0]), cam_mat=np.eye(3), fovy_deg=90.0, resolution=100)


def _blending_loop(enable_action_blending=True, action_xy=(0.2, -0.1)):
    components = ControlLoopComponents(
        policy=_FakePolicyWithAction(action_xy=action_xy),
        subgoal_generator=_FakeSubgoalGenerator(),
        cot_generator=_FakeCotGenerator(),
        flow_estimator=_FakeFlowEstimator(),
        kinematic_extrapolator=_FakeKinematicExtrapolatorWithDelta(),
        router=OcclusionRouter(),
        plausibility_checker=_FakePlausibilityChecker(),
        enable_action_blending=enable_action_blending,
    )
    return ControlLoop(components)


def test_action_blending_off_by_default_even_with_geometry_supplied():
    # arm_s_occ must stay <= OcclusionRouter's arm_occ_threshold, or
    # routing goes to SELF and the SCENE/blending branch never runs at
    # all (see SCENE_BLEND_ALPHA's docstring for why this matters).
    loop = _blending_loop(enable_action_blending=False)
    perception = PerceptionInputs(
        arm_pixel_mask=np.zeros((4, 4), dtype=bool), arm_s_occ=0.0, scene_dyn_occ=True, target_patch_idx=3,
        camera_projector=_overhead_projector(), eef_pos_world=np.zeros(3),
    )
    loop.step(_obs(), perception)
    loop.step(_obs(), perception)
    actions = loop.step(_obs(), perception)

    np.testing.assert_allclose(actions[0, :2], [0.2, -0.1])


def test_action_blending_skipped_without_geometry_even_when_enabled():
    loop = _blending_loop(enable_action_blending=True)
    perception = PerceptionInputs(
        arm_pixel_mask=np.zeros((4, 4), dtype=bool), arm_s_occ=0.0, scene_dyn_occ=True, target_patch_idx=3,
        # camera_projector/eef_pos_world left as default None
    )
    loop.step(_obs(), perception)
    loop.step(_obs(), perception)
    actions = loop.step(_obs(), perception)

    np.testing.assert_allclose(actions[0, :2], [0.2, -0.1])


def test_action_blending_matches_manual_computation_when_enabled():
    from occ_vla.integration.runtime import SCENE_BLEND_ALPHA  # noqa: PLC0415
    from occ_vla.pklp.pixel_to_action import pklp_pixel_delta_to_world_delta  # noqa: PLC0415

    projector = _overhead_projector()
    eef_pos = np.zeros(3)
    action_xy = np.array([0.2, -0.1])

    loop = _blending_loop(enable_action_blending=True, action_xy=tuple(action_xy))
    perception = PerceptionInputs(
        arm_pixel_mask=np.zeros((4, 4), dtype=bool), arm_s_occ=0.0, scene_dyn_occ=True, target_patch_idx=3,
        camera_projector=projector, eef_pos_world=eef_pos,
    )
    loop.step(_obs(), perception)
    loop.step(_obs(), perception)
    actions = loop.step(_obs(), perception)  # 3rd frame -> history full -> PKLP + blending fire

    extrapolator = loop.c.kinematic_extrapolator
    world_delta = pklp_pixel_delta_to_world_delta(projector, eef_pos, extrapolator.current, extrapolator.predicted)
    pklp_delta_xy = world_delta[:2] / OSC_POSE_MAX_DELTA_M
    expected = np.clip((1 - SCENE_BLEND_ALPHA) * action_xy + SCENE_BLEND_ALPHA * pklp_delta_xy, -1.0, 1.0)

    np.testing.assert_allclose(actions[0, :2], expected)
    assert not np.allclose(actions[0, :2], action_xy)  # blending actually changed something


def test_action_blending_gate_skips_blend_when_directions_conflict():
    # Conflict-avoidance gate (2026-07-18, see gated_blend_xy docstring):
    # current=(1,2) -> predicted=(9,6) pushes pklp_xy toward +x/-y (see
    # test_action_blending_matches_manual_computation_when_enabled), so
    # pointing pi0.5's own action the opposite way (-x/+y) makes the two
    # vectors conflict (cos_angle < 0) -- the gate should then leave
    # pi0.5's own action completely untouched instead of destructively
    # cancelling it against pklp_xy.
    projector = _overhead_projector()
    eef_pos = np.zeros(3)
    action_xy = np.array([-0.2, 0.1])

    loop = _blending_loop(enable_action_blending=True, action_xy=tuple(action_xy))
    perception = PerceptionInputs(
        arm_pixel_mask=np.zeros((4, 4), dtype=bool), arm_s_occ=0.0, scene_dyn_occ=True, target_patch_idx=3,
        camera_projector=projector, eef_pos_world=eef_pos,
    )
    loop.step(_obs(), perception)
    loop.step(_obs(), perception)
    actions = loop.step(_obs(), perception)  # 3rd frame -> history full -> PKLP + blending fire

    np.testing.assert_allclose(actions[0, :2], action_xy)
