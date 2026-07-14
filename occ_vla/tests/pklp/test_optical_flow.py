import numpy as np
import pytest

from occ_vla.pklp.optical_flow import patch_pool_flow


def test_patch_pool_flow_uniform_field():
    flow = np.zeros((32, 32, 2), dtype=np.float32)
    flow[..., 0] = 2.0  # constant x-flow
    flow[..., 1] = -1.0  # constant y-flow
    result = patch_pool_flow(flow, patch_size=16)
    assert result.grid_shape == (2, 2)
    assert result.flow.shape == (4, 2)
    np.testing.assert_allclose(result.flow, np.tile([2.0, -1.0], (4, 1)))


def test_patch_pool_flow_patch_centers():
    flow = np.zeros((32, 16, 2), dtype=np.float32)
    result = patch_pool_flow(flow, patch_size=16)
    assert result.grid_shape == (2, 1)
    # rows at y=8 and y=24, single column at x=8
    np.testing.assert_allclose(sorted(result.patch_centers[:, 1]), [8.0, 24.0])
    assert np.all(result.patch_centers[:, 0] == 8.0)


def test_patch_pool_flow_averages_within_patch():
    flow = np.zeros((16, 16, 2), dtype=np.float32)
    flow[0, 0] = [16.0, 0.0]  # single outlier pixel in a 16x16=256-pixel patch
    result = patch_pool_flow(flow, patch_size=16)
    np.testing.assert_allclose(result.flow[0], [16.0 / 256, 0.0])


def test_patch_pool_flow_rejects_non_divisible_shape():
    flow = np.zeros((30, 32, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        patch_pool_flow(flow, patch_size=16)
