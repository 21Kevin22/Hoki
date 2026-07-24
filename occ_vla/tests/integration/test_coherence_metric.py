import numpy as np

from occ_vla.integration.coherence_metric import coherence_score


def _checkerboard(size: int, square: int = 4) -> np.ndarray:
    """High-frequency synthetic texture, as a stand-in for 'real detailed content'."""
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    pattern = ((yy // square + xx // square) % 2).astype(np.uint8) * 255
    return np.stack([pattern] * 3, axis=-1)


def test_coherence_score_high_when_region_matches_reference_detail():
    image = _checkerboard(64)
    mask = np.zeros((64, 64), dtype=bool)
    mask[16:48, 16:48] = True  # same checkerboard texture inside and outside

    score = coherence_score(image, mask)
    assert score > 0.85


def test_coherence_score_low_when_region_is_flat_blob():
    image = _checkerboard(64)
    image = image.copy()
    image[16:48, 16:48] = 128  # flat/smeared region -- the "blob" failure mode

    mask = np.zeros((64, 64), dtype=bool)
    mask[16:48, 16:48] = True

    score = coherence_score(image, mask)
    assert score < 0.3


def test_coherence_score_low_when_region_is_much_noisier_than_reference():
    rng = np.random.default_rng(0)
    image = np.full((64, 64, 3), 128, dtype=np.uint8)  # flat, low-detail reference
    noisy_patch = rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8)
    image[16:48, 16:48] = noisy_patch  # chaotic garbled-noise region

    mask = np.zeros((64, 64), dtype=bool)
    mask[16:48, 16:48] = True

    score = coherence_score(image, mask)
    assert score < 0.3


def test_coherence_score_zero_when_region_too_small_for_a_tile():
    image = _checkerboard(64)
    mask = np.zeros((64, 64), dtype=bool)
    mask[0:3, 0:3] = True  # smaller than DEFAULT_TILE_SIZE=8

    assert coherence_score(image, mask) == 0.0
