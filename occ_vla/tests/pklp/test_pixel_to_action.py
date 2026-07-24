import numpy as np
import pytest

from occ_vla.pklp.pixel_to_action import (
    CameraProjector,
    pixel_delta_to_world_delta,
    pklp_pixel_delta_to_world_delta,
    projection_jacobian,
)


def _overhead_projector(resolution: int = 256, fovy_deg: float = 90.0) -> CameraProjector:
    """Camera at (0, 0, 1) looking straight down (-world-z) with the
    camera frame aligned to the world frame -- the simplest case where
    the expected projection can be hand-verified."""
    return CameraProjector(
        cam_pos=np.array([0.0, 0.0, 1.0]),
        cam_mat=np.eye(3),
        fovy_deg=fovy_deg,
        resolution=resolution,
    )


def test_project_origin_maps_to_image_center():
    projector = _overhead_projector()
    px = projector.project(np.array([0.0, 0.0, 0.0]))
    np.testing.assert_allclose(px, [128.0, 128.0])


def test_project_positive_world_x_moves_pixel_right():
    projector = _overhead_projector()
    px = projector.project(np.array([0.1, 0.0, 0.0]))
    assert px[0] > 128.0
    np.testing.assert_allclose(px[1], 128.0, atol=1e-6)


def test_projection_jacobian_axes_decouple_at_boresight():
    """Directly overhead, on the optical axis: world-x only moves pixel-x,
    world-y only moves pixel-y, and world-z (depth, along the viewing ray)
    has ~zero effect since the point stays centered regardless of depth."""
    projector = _overhead_projector()
    jac = projection_jacobian(projector, np.array([0.0, 0.0, 0.0]))
    assert jac.shape == (2, 3)
    assert jac[0, 0] != pytest.approx(0.0)
    assert jac[1, 0] == pytest.approx(0.0, abs=1e-6)
    assert jac[1, 1] != pytest.approx(0.0)
    assert jac[0, 1] == pytest.approx(0.0, abs=1e-6)
    np.testing.assert_allclose(jac[:, 2], [0.0, 0.0], atol=1e-6)


def test_pixel_delta_to_world_delta_recovers_expected_direction():
    projector = _overhead_projector()
    jac = projection_jacobian(projector, np.array([0.0, 0.0, 0.0]))
    # px increases with +world-x (test above), so a desired +px shift
    # should be explained by a +world-x delta.
    world_delta = pixel_delta_to_world_delta(jac, pixel_delta=np.array([10.0, 0.0]), max_step_m=10.0)
    assert world_delta[0] > 0.0
    assert world_delta[1] == pytest.approx(0.0, abs=1e-6)


def test_zero_z_masks_vertical_component():
    jac = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5]])
    world_delta = pixel_delta_to_world_delta(jac, pixel_delta=np.array([1.0, 1.0]), zero_z=True, max_step_m=10.0)
    assert world_delta[2] == 0.0


def test_z_not_masked_when_disabled():
    jac = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5]])
    world_delta = pixel_delta_to_world_delta(jac, pixel_delta=np.array([1.0, 1.0]), zero_z=False, max_step_m=10.0)
    assert world_delta[2] != 0.0


def test_max_step_clips_norm():
    jac = np.array([[0.01, 0.0, 0.0], [0.0, 0.01, 0.0]])  # tiny sensitivity -> large raw delta
    world_delta = pixel_delta_to_world_delta(jac, pixel_delta=np.array([100.0, 0.0]), max_step_m=0.03)
    assert np.linalg.norm(world_delta) == pytest.approx(0.03, rel=1e-6)


def test_damping_keeps_singular_jacobian_finite():
    # Both rows identical -> J @ J.T is singular (rank-deficient), the
    # near-edge-on-camera case this damping term exists for.
    jac = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    world_delta = pixel_delta_to_world_delta(jac, pixel_delta=np.array([1.0, 1.0]), damping=1e-2, max_step_m=10.0)
    assert np.all(np.isfinite(world_delta))


def test_zero_damping_on_singular_jacobian_raises():
    jac = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    with pytest.raises(np.linalg.LinAlgError):
        pixel_delta_to_world_delta(jac, pixel_delta=np.array([1.0, 1.0]), damping=0.0, max_step_m=10.0)


def test_pklp_wrapper_matches_manual_composition():
    projector = _overhead_projector()
    eef = np.array([0.0, 0.0, 0.0])
    current_pixel = projector.project(eef)
    predicted_pixel = current_pixel + np.array([10.0, 0.0])

    expected_jac = projection_jacobian(projector, eef)
    expected = pixel_delta_to_world_delta(expected_jac, predicted_pixel - current_pixel, max_step_m=0.03)

    got = pklp_pixel_delta_to_world_delta(projector, eef, current_pixel, predicted_pixel, max_step_m=0.03)
    np.testing.assert_allclose(got, expected)


def test_pklp_wrapper_zero_pixel_delta_gives_zero_world_delta():
    projector = _overhead_projector()
    eef = np.array([0.0, 0.0, 0.0])
    pixel = projector.project(eef)
    world_delta = pklp_pixel_delta_to_world_delta(projector, eef, pixel, pixel)
    np.testing.assert_allclose(world_delta, [0.0, 0.0, 0.0], atol=1e-9)
