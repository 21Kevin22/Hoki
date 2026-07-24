import re

import numpy as np

from occ_vla.eval.libero_occ_env import AGENTVIEW_KEY, AGENTVIEW_SEGMENTATION_KEY, LiberoOccEnv, LiberoOccEnvConfig
from occ_vla.eval.metrics import DIFFICULTY_BANDS, Difficulty

TARGET_MASK = np.ones((8, 8), dtype=bool)


class _FakeModel:
    def body_name2id(self, name):
        return {"table": 0, "target_obj": 1}[name]

    def camera_name2id(self, name):
        return 0

    def geom_name2id(self, name):
        return 99

    def get_xml(self):
        return '<mujoco><worldbody><body name="table"/></worldbody></mujoco>'


class _FakeSimData:
    def __init__(self):
        self.body_xpos = {0: np.array([0.0, 0.0, 0.0]), 1: np.array([0.0, 0.0, 1.0])}
        self.cam_xpos = {0: np.array([0.0, -1.0, 1.0])}
        self.contact = []
        self.ncon = 0


class _FakeSim:
    def __init__(self):
        self.model = _FakeModel()
        self.data = _FakeSimData()


class _FakeDomainEnv:
    def __init__(self):
        self.sim = _FakeSim()
        self.obj_body_id = {"target_obj": 1}


class _FakeLiberoEnv:
    """Stands in for libero.libero.envs.SegmentationRenderEnv."""

    def __init__(self, bddl_file_name, camera_names):
        self.bddl_file_name = bddl_file_name
        self.env = _FakeDomainEnv()
        self.obj_of_interest = ["target_obj"]
        self.segmentation_robot_id = 0
        self._last_half_extent = 0.0
        self.step_calls = 0
        self.arm_occluding = False  # tests flip this to simulate the arm covering the target

    @property
    def sim(self):
        return self.env.sim

    def reset(self):
        return {}

    def set_init_state(self, init_state):
        return {}

    def get_sim_state(self):
        return "state"

    def reset_from_xml_string(self, xml):
        match = re.search(r'size="([\d.]+) ([\d.]+) ([\d.]+)"', xml)
        self._last_half_extent = float(match.group(1)) if match else 0.0

    def regenerate_obs_from_state(self, state):
        return {AGENTVIEW_SEGMENTATION_KEY: None}

    def get_segmentation_instances(self, segmentation_image):
        occlusion_fraction = min(1.0, 3.0 * self._last_half_extent)
        n_occluded = int(round(occlusion_fraction * TARGET_MASK.sum()))
        mask = TARGET_MASK.copy().reshape(-1)
        mask[:n_occluded] = False
        target_mask_now = mask.reshape(TARGET_MASK.shape)
        # Arm mask covers whatever part of the target is currently not
        # visible, if arm_occluding is set -- otherwise all-False (the
        # occlusion, if any, is attributed to something else, e.g. a
        # placed physical occluder).
        robot_mask = (~target_mask_now) if self.arm_occluding else np.zeros_like(target_mask_now)
        return {
            "target_obj": target_mask_now[..., None],
            "robot": robot_mask[..., None],
        }

    def step(self, action):
        self.step_calls += 1
        obs = {
            AGENTVIEW_KEY: np.full(TARGET_MASK.shape + (3,), 100, dtype=np.uint8),
            AGENTVIEW_SEGMENTATION_KEY: None,
        }
        return obs, 0.0, False, {}


class _FakeBenchmark:
    def get_task_bddl_file_path(self, task_id):
        return "/fake/task.bddl"

    def get_task_init_states(self, task_id):
        return ["init_state_0"]


def _build_env_with_fakes(config: LiberoOccEnvConfig) -> LiberoOccEnv:
    occ_env = LiberoOccEnv(config, libero_root="/fake/libero")
    occ_env._benchmark = _FakeBenchmark()  # noqa: SLF001
    occ_env._env = _FakeLiberoEnv(bddl_file_name="/fake/task.bddl", camera_names=[])  # noqa: SLF001
    occ_env._build_env = lambda: None  # noqa: SLF001 -- already built above, skip real libero import
    return occ_env


def test_reset_places_occluder_and_records_s_occ_in_band():
    config = LiberoOccEnvConfig(benchmark_suite="libero_spatial", task_id=0, difficulty=Difficulty.MEDIUM)
    occ_env = _build_env_with_fakes(config)

    obs = occ_env.reset()

    assert obs is not None
    lo, hi = DIFFICULTY_BANDS[Difficulty.MEDIUM]
    assert lo <= occ_env.last_s_occ < hi


def test_step_delegates_to_underlying_env():
    config = LiberoOccEnvConfig(benchmark_suite="libero_spatial", task_id=0, difficulty=Difficulty.LIGHT)
    occ_env = _build_env_with_fakes(config)
    occ_env.reset()

    occ_env.step(np.zeros(7))
    assert occ_env._env.step_calls == 1  # noqa: SLF001


def test_arm_s_occ_zero_with_no_arm_occlusion():
    config = LiberoOccEnvConfig(
        benchmark_suite="libero_spatial", task_id=0, difficulty=Difficulty.LIGHT, place_occluder=False
    )
    occ_env = _build_env_with_fakes(config)
    obs = occ_env.reset()
    occ_env.capture_clear_baseline(obs)

    assert occ_env.compute_arm_s_occ(obs) == 0.0
    assert occ_env.compute_total_occ(obs) == 0.0


def test_arm_s_occ_reflects_arm_occlusion_against_clear_baseline():
    config = LiberoOccEnvConfig(
        benchmark_suite="libero_spatial", task_id=0, difficulty=Difficulty.LIGHT, place_occluder=False
    )
    occ_env = _build_env_with_fakes(config)
    obs = occ_env.reset()
    occ_env.capture_clear_baseline(obs)  # clear: target fully visible (half_extent starts at 0)

    occ_env._env.arm_occluding = True  # noqa: SLF001
    occ_env._env._last_half_extent = 1.0  # noqa: SLF001 -- now fully covered
    obs_now = occ_env._env.regenerate_obs_from_state(occ_env._env.get_sim_state())

    assert occ_env.compute_arm_s_occ(obs_now) == 1.0
    assert occ_env.compute_total_occ(obs_now) == 1.0


def test_compute_arm_s_occ_requires_baseline_first():
    config = LiberoOccEnvConfig(
        benchmark_suite="libero_spatial", task_id=0, difficulty=Difficulty.LIGHT, place_occluder=False
    )
    occ_env = _build_env_with_fakes(config)
    obs = occ_env.reset()

    try:
        occ_env.compute_arm_s_occ(obs)
        raise AssertionError("expected RuntimeError")
    except RuntimeError:
        pass


def test_pixel_mask_is_noop_before_baseline_captured():
    config = LiberoOccEnvConfig(
        benchmark_suite="libero_spatial", task_id=0, difficulty=Difficulty.LIGHT, place_occluder=False, pixel_mask=True
    )
    occ_env = _build_env_with_fakes(config)
    occ_env.reset()

    obs, _, _, _ = occ_env.step(np.zeros(7))
    assert (obs[AGENTVIEW_KEY] == 100).all()


def test_pixel_mask_blackens_clear_footprint_after_baseline_captured():
    config = LiberoOccEnvConfig(
        benchmark_suite="libero_spatial", task_id=0, difficulty=Difficulty.LIGHT, place_occluder=False, pixel_mask=True
    )
    occ_env = _build_env_with_fakes(config)
    obs = occ_env.reset()
    occ_env.capture_clear_baseline(obs)

    obs2, _, _, _ = occ_env.step(np.zeros(7))
    # TARGET_MASK covers the whole synthetic frame here, so the clean
    # footprint blackens every pixel -- the point being it's masked at
    # all (and wasn't, above, before capture_clear_baseline).
    assert (obs2[AGENTVIEW_KEY] == 0).all()
