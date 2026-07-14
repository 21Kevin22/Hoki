import numpy as np
import pytest

from occ_vla.pklp.kinematics import KinematicExtrapolator, KinematicState, motion_descriptor_tokens
from occ_vla.pklp.optical_flow import PatchFlow


def test_extrapolate_constant_velocity():
    state = KinematicState(
        position=np.array([0.0, 0.0]),
        velocity=np.array([1.0, 0.0]),
        acceleration=np.array([0.0, 0.0]),
    )
    result = KinematicExtrapolator().extrapolate(state, steps=5)
    np.testing.assert_allclose(result, [5.0, 0.0])


def test_extrapolate_with_acceleration():
    state = KinematicState(
        position=np.array([0.0, 0.0]),
        velocity=np.array([0.0, 0.0]),
        acceleration=np.array([2.0, 0.0]),
    )
    result = KinematicExtrapolator().extrapolate(state, steps=3)
    # x = 0 + 0*3 + 0.5*2*9 = 9
    np.testing.assert_allclose(result, [9.0, 0.0])


def _patch_flow(centers, flow):
    centers = np.asarray(centers, dtype=float)
    flow = np.asarray(flow, dtype=float)
    return PatchFlow(patch_centers=centers, flow=flow, grid_shape=(1, len(centers)))


def test_estimate_state_constant_velocity_gives_zero_acceleration():
    earlier = _patch_flow([[10.0, 10.0]], [[2.0, 0.0]])
    latest = _patch_flow([[12.0, 10.0]], [[2.0, 0.0]])
    state = KinematicExtrapolator().estimate_state([earlier, latest], target_patch_idx=0)
    np.testing.assert_allclose(state.position, [12.0, 10.0])
    np.testing.assert_allclose(state.velocity, [2.0, 0.0])
    np.testing.assert_allclose(state.acceleration, [0.0, 0.0])


def test_estimate_state_finite_difference_acceleration():
    earlier = _patch_flow([[0.0, 0.0]], [[1.0, 0.0]])
    latest = _patch_flow([[1.0, 0.0]], [[3.0, 0.0]])
    state = KinematicExtrapolator().estimate_state([earlier, latest], target_patch_idx=0)
    # A = (v_latest - v_earlier) / dt = (3 - 1) / 1
    np.testing.assert_allclose(state.acceleration, [2.0, 0.0])


def test_estimate_state_rejects_wrong_history_length():
    only_one = _patch_flow([[0.0, 0.0]], [[1.0, 0.0]])
    with pytest.raises(ValueError):
        KinematicExtrapolator().estimate_state([only_one], target_patch_idx=0)


def test_motion_descriptor_tokens_concatenates_v_and_a():
    visual = np.zeros((3, 8))
    velocity = np.ones((3, 2)) * 2.0
    acceleration = np.ones((3, 2)) * -1.0
    out = motion_descriptor_tokens(visual, velocity, acceleration)
    assert out.shape == (3, 12)
    np.testing.assert_allclose(out[:, 8:10], velocity)
    np.testing.assert_allclose(out[:, 10:12], acceleration)


def test_motion_descriptor_tokens_rejects_shape_mismatch():
    visual = np.zeros((3, 8))
    with pytest.raises(ValueError):
        motion_descriptor_tokens(visual, np.ones((2, 2)), np.ones((3, 2)))
