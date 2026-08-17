import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from oft_occlusion_gt import (
    GRID_SIDE,
    PATCH_PX,
    occ_gt_for_camera_block,
    occ_gt_scalar,
    patch_labels,
    patch_overlap_fractions,
)

# Matches run_oft_camera_dropout_eval.py's _apply_partial_patch geometry:
# PARTIAL_PATCH_FRAC=0.59 on a 224px side.
FULL_SIDE = GRID_SIDE * PATCH_PX  # 224


def _fixed_partial_patch_bounds():
    ph = pw = int(FULL_SIDE * 0.59)
    r0, c0 = (FULL_SIDE - ph) // 2, (FULL_SIDE - pw) // 2
    return (r0, r0 + ph, c0, c0 + pw)


def test_no_occlusion_gives_zero_everywhere():
    fractions = patch_overlap_fractions((0, 0, 0, 0))
    assert fractions.shape == (GRID_SIDE, GRID_SIDE)
    assert np.all(fractions == 0.0)
    assert occ_gt_scalar(fractions) == 0.0


def test_full_frame_occlusion_gives_one_everywhere():
    fractions = patch_overlap_fractions((0, FULL_SIDE, 0, FULL_SIDE))
    assert np.allclose(fractions, 1.0)
    assert occ_gt_scalar(fractions) == 1.0


def test_single_patch_exact_alignment():
    # occlude exactly patch (i=2, j=5)'s pixel footprint -- fraction there
    # should be exactly 1.0, and exactly 0.0 everywhere else.
    r0, c0 = 2 * PATCH_PX, 5 * PATCH_PX
    fractions = patch_overlap_fractions((r0, r0 + PATCH_PX, c0, c0 + PATCH_PX))
    assert fractions[2, 5] == 1.0
    assert fractions.sum() == 1.0  # nothing else touched


def test_partial_overlap_is_fractional_not_boolean():
    # occlude half of one patch's width (7 of 14 px) -> exactly 0.5, not
    # rounded to 0 or 1 -- this is the whole point vs. the eval harness's
    # own center-in/center-out `_build_patch_token_mask` approximation.
    r0, c0 = 0, 0
    fractions = patch_overlap_fractions((r0, r0 + PATCH_PX, c0, c0 + PATCH_PX // 2))
    assert fractions[0, 0] == 7 / 14


def test_patch_labels_thresholds_correctly():
    fractions = np.array([[0.0, 0.49], [0.5, 0.51]])
    labels = patch_labels(fractions, patch_threshold=0.5)
    assert labels.tolist() == [[False, False], [False, True]]  # strictly greater than 0.5


def test_occ_gt_for_camera_block_matches_realistic_bounds():
    bounds = _fixed_partial_patch_bounds()
    occ_gt = occ_gt_for_camera_block(bounds)
    # PARTIAL_PATCH_FRAC=0.59 -> 34.7% raw pixel area, but occ_gt uses the
    # patch->0.5-area hard-label rule (occ_gt_scalar), which lands
    # noticeably higher (0.390625 = 25/64 patches) because a patch only
    # needs >50% of ITS OWN area covered to count, not >50% of the whole
    # frame -- exact value pinned here as a regression check, not a guess.
    assert abs(occ_gt - 0.390625) < 1e-9


def test_occ_gt_for_camera_block_none_bounds_is_zero():
    assert occ_gt_for_camera_block(None) == 0.0


def test_fractions_never_exceed_unit_interval():
    # sweep several overlapping/edge-case rectangles, confirm no overflow
    rng = np.random.default_rng(0)
    for _ in range(50):
        r0, r1 = sorted(rng.integers(-20, FULL_SIDE + 20, size=2).tolist())
        c0, c1 = sorted(rng.integers(-20, FULL_SIDE + 20, size=2).tolist())
        fractions = patch_overlap_fractions((r0, r1, c0, c1))
        assert fractions.min() >= 0.0
        assert fractions.max() <= 1.0
