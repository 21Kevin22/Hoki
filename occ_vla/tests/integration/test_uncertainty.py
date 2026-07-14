import numpy as np
import pytest

from occ_vla.integration.uncertainty import PLAUSIBILITY_FALLBACK_THRESHOLD, PlausibilityChecker


def _images(background_shift=0):
    original = np.zeros((8, 8, 3), dtype=np.uint8)
    original[:, 4:, :] = 200  # some texture so background isn't trivially all-zero
    arm_mask = np.zeros((8, 8), dtype=bool)
    arm_mask[:4, :4] = True

    generated = original.copy()
    generated[:4, :4] = 50  # arm region legitimately changed
    if background_shift:
        generated[:, 4:] = np.clip(generated[:, 4:].astype(int) + background_shift, 0, 255).astype(np.uint8)
    return generated, original, arm_mask


def test_score_high_when_only_arm_region_changed():
    generated, original, arm_mask = _images(background_shift=0)
    checker = PlausibilityChecker()
    score = checker.score(generated, {"original_image": original, "arm_pixel_mask": arm_mask})
    assert score > PLAUSIBILITY_FALLBACK_THRESHOLD


def test_score_low_when_background_also_changed():
    generated, original, arm_mask = _images(background_shift=150)
    checker = PlausibilityChecker()
    score = checker.score(generated, {"original_image": original, "arm_pixel_mask": arm_mask})
    assert score < PLAUSIBILITY_FALLBACK_THRESHOLD


def test_should_fallback_matches_threshold():
    generated, original, arm_mask = _images(background_shift=150)
    checker = PlausibilityChecker()
    assert checker.should_fallback(generated, {"original_image": original, "arm_pixel_mask": arm_mask})


def test_score_zero_when_arm_mask_covers_whole_frame():
    original = np.zeros((8, 8, 3), dtype=np.uint8)
    generated = original.copy()
    arm_mask = np.ones((8, 8), dtype=bool)
    checker = PlausibilityChecker()
    score = checker.score(generated, {"original_image": original, "arm_pixel_mask": arm_mask})
    assert score == 0.0


def test_score_rejects_shape_mismatch():
    original = np.zeros((8, 8, 3), dtype=np.uint8)
    generated = np.zeros((4, 4, 3), dtype=np.uint8)
    arm_mask = np.zeros((8, 8), dtype=bool)
    checker = PlausibilityChecker()
    with pytest.raises(ValueError):
        checker.score(generated, {"original_image": original, "arm_pixel_mask": arm_mask})
