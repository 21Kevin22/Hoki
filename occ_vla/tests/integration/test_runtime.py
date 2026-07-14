import numpy as np

from occ_vla.control.pi05_policy import Pi05Observation
from occ_vla.integration.occlusion_router import OcclusionRouter
from occ_vla.integration.runtime import ControlLoop, ControlLoopComponents, PerceptionInputs
from occ_vla.pklp.kinematics import KinematicState
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


def _loop(fallback=False):
    components = ControlLoopComponents(
        policy=_FakePolicy(),
        subgoal_generator=_FakeSubgoalGenerator(),
        cot_generator=_FakeCotGenerator(),
        flow_estimator=_FakeFlowEstimator(),
        kinematic_extrapolator=_FakeKinematicExtrapolator(),
        router=OcclusionRouter(),
        plausibility_checker=_FakePlausibilityChecker(fallback=fallback),
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


def test_self_occlusion_injects_subgoal_image():
    loop = _loop()
    perception = PerceptionInputs(
        arm_pixel_mask=np.ones((4, 4), dtype=bool), arm_s_occ=0.5, scene_dyn_occ=False, target_patch_idx=None
    )
    loop.step(_obs(), perception)

    assert loop.c.subgoal_generator.calls == 1
    assert loop.c.policy.last_obs.subgoal_image is not None


def test_self_occlusion_falls_back_to_pklp_on_implausible_subgoal():
    loop = _loop(fallback=True)
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


def test_self_occlusion_pseudo_anchors_frame_history_when_last_clear_is_none():
    loop = _loop()
    perception = PerceptionInputs(
        arm_pixel_mask=np.ones((4, 4), dtype=bool), arm_s_occ=0.5, scene_dyn_occ=False, target_patch_idx=None
    )
    loop.step(_obs(), perception)
    # base frame + pseudo-anchor subgoal frame == 2 pushed in one step
    assert len(loop._frame_history) == 2  # noqa: SLF001
