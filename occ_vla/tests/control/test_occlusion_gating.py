import numpy as np

from occ_vla.control.occlusion_gating import DEFAULT_MIN_SCALE, apply_soft_gate, gate_scale


def test_gate_scale_no_attenuation_below_threshold():
    assert gate_scale(0.0) == 1.0
    assert gate_scale(0.30) == 1.0


def test_gate_scale_ramps_down_above_threshold():
    mid = gate_scale(0.65, threshold=0.30, min_scale=0.3)
    assert 0.3 < mid < 1.0


def test_gate_scale_floors_at_min_scale():
    assert gate_scale(1.0) == DEFAULT_MIN_SCALE
    assert gate_scale(5.0) == DEFAULT_MIN_SCALE  # clamps even past 1.0


def test_apply_soft_gate_noop_at_full_scale():
    image = np.full((4, 4, 3), 200, dtype=np.uint8)
    assert apply_soft_gate(image, 1.0) is image


def test_apply_soft_gate_attenuates_not_zeroes():
    image = np.full((4, 4, 3), 200, dtype=np.uint8)
    gated = apply_soft_gate(image, 0.3)
    assert gated.dtype == np.uint8
    assert gated.max() == 60  # 200 * 0.3
    assert gated.max() > 0
