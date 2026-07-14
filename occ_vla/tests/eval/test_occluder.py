import re

import numpy as np
import pytest

from occ_vla.eval.metrics import DIFFICULTY_BANDS, Difficulty
from occ_vla.eval.occluder import CAMERA_TARGET_FRACTION, OccluderPlacer, OccluderSpec


def test_build_occluder_xml_fragment_contains_pose_and_size():
    spec = OccluderSpec(size=(0.05, 0.05, 0.1), position=(0.1, 0.2, 0.3), target_difficulty=Difficulty.MEDIUM)
    xml = OccluderPlacer().build_occluder_xml_fragment(spec)
    assert '<body name="occluder" pos="0.1 0.2 0.3">' in xml
    assert 'size="0.05 0.05 0.1"' in xml
    assert 'name="occluder_geom"' in xml
    assert xml.count("<body") == xml.count("</body>")


def test_insert_before_worldbody_close_places_fragment_inside():
    xml = "<mujoco><worldbody><body name=\"table\"/></worldbody></mujoco>"
    fragment = '<body name="occluder"/>'
    result = OccluderPlacer()._insert_before_worldbody_close(xml, fragment)  # noqa: SLF001
    assert result == "<mujoco><worldbody><body name=\"table\"/><body name=\"occluder\"/></worldbody></mujoco>"


def test_insert_before_worldbody_close_raises_without_worldbody():
    with pytest.raises(ValueError):
        OccluderPlacer()._insert_before_worldbody_close("<mujoco></mujoco>", "<body/>")  # noqa: SLF001


class _FakeModel:
    def __init__(self):
        self._bodies = {"table": 0, "target_obj": 1}
        self._cameras = {"agentview": 0}

    def body_name2id(self, name):
        return self._bodies[name]

    def camera_name2id(self, name):
        return self._cameras[name]

    def geom_name2id(self, name):
        assert name == "occluder_geom"
        return 99

    def get_xml(self):
        return "<mujoco><worldbody><body name=\"table\"/></worldbody></mujoco>"


class _FakeContact:
    def __init__(self, geom1, geom2):
        self.geom1 = geom1
        self.geom2 = geom2


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


TARGET_MASK = np.ones((8, 8), dtype=bool)


class _FakeSegmentationEnv:
    """Occlusion grows monotonically with the requested occluder
    half-extent: at half_extent, `occlusion_fraction * 64` of the
    target's 64 pixels are covered. `occlusion_fraction` is read back
    out of the XML the placer builds, so this exercises the real
    binary-search loop end to end."""

    def __init__(self, occlusion_per_half_extent: float, always_in_contact: bool = False):
        self.env = _FakeDomainEnv()
        self._occlusion_per_half_extent = occlusion_per_half_extent
        self._always_in_contact = always_in_contact
        self._last_half_extent = 0.0

    @property
    def sim(self):
        return self.env.sim

    def get_sim_state(self):
        return "baseline_state"

    def reset_from_xml_string(self, xml):
        match = re.search(r'size="([\d.]+) ([\d.]+) ([\d.]+)"', xml)
        self._last_half_extent = float(match.group(1)) if match else 0.0
        self.env.sim.data.ncon = 1 if self._always_in_contact else 0
        self.env.sim.data.contact = [_FakeContact(99, 0)] if self._always_in_contact else []

    def regenerate_obs_from_state(self, state):
        return {"agentview_segmentation_instance": None}

    def get_segmentation_instances(self, segmentation_image):
        occlusion_fraction = min(1.0, self._occlusion_per_half_extent * self._last_half_extent)
        n_occluded = int(round(occlusion_fraction * TARGET_MASK.sum()))
        mask = TARGET_MASK.copy().reshape(-1)
        mask[:n_occluded] = False
        return {"target_obj": mask.reshape(TARGET_MASK.shape)}


def test_search_converges_to_requested_difficulty_band():
    # occlusion_per_half_extent chosen so some half_extent in [0.01, 0.15] lands in MEDIUM (0.3-0.6)
    env = _FakeSegmentationEnv(occlusion_per_half_extent=3.0)
    spec = OccluderPlacer().search(env, "target_obj", Difficulty.MEDIUM)

    lo, hi = DIFFICULTY_BANDS[Difficulty.MEDIUM]
    achieved = min(1.0, 3.0 * spec.size[0])
    assert lo <= achieved < hi


def test_search_places_occluder_along_camera_to_target_line():
    env = _FakeSegmentationEnv(occlusion_per_half_extent=3.0)
    spec = OccluderPlacer().search(env, "target_obj", Difficulty.MEDIUM)

    camera_pos = np.array([0.0, -1.0, 1.0])
    target_pos = np.array([0.0, 0.0, 1.0])
    expected_position = camera_pos + CAMERA_TARGET_FRACTION * (target_pos - camera_pos)
    np.testing.assert_allclose(spec.position, expected_position)


def test_search_raises_when_target_not_visible_at_baseline():
    env = _FakeSegmentationEnv(occlusion_per_half_extent=3.0)

    def empty_target(segmentation_image):
        return {"target_obj": np.zeros((8, 8), dtype=bool)}

    env.get_segmentation_instances = empty_target
    with pytest.raises(RuntimeError, match="isn't visible"):
        OccluderPlacer().search(env, "target_obj", Difficulty.MEDIUM)


def test_search_rejects_candidates_in_contact():
    env = _FakeSegmentationEnv(occlusion_per_half_extent=3.0, always_in_contact=True)
    with pytest.raises(RuntimeError, match="did not converge"):
        OccluderPlacer(max_search_iters=4).search(env, "target_obj", Difficulty.MEDIUM)


def test_search_raises_when_band_unreachable_within_size_bounds():
    # occlusion_per_half_extent tiny -> even MAX_HALF_EXTENT never reaches HEAVY band
    env = _FakeSegmentationEnv(occlusion_per_half_extent=0.001)
    with pytest.raises(RuntimeError, match="did not converge"):
        OccluderPlacer(max_search_iters=6).search(env, "target_obj", Difficulty.HEAVY)
