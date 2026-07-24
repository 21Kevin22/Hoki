"""Prototype, NOT YET VALIDATED, alternative/companion signal to
`PlausibilityChecker` (uncertainty.py).

Motivation (occ_vla/CLAUDE.md, "Mask-area & temperature-schedule
investigation" + "Multi-frame/multi-task area-sweep follow-up"):
`PlausibilityChecker`'s background-MSE score is dominated by a ~100-135
MSE noise floor from MAGVIT-v2's own encode/decode roundtrip on
*untouched* tokens -- a near-constant offset unrelated to whether the
*masked* (actually-regenerated) region looks like a coherent object or
a smeared/garbled blob. It also compares against a fixed absolute
scale (`DEFAULT_MSE_SCALE`), which doesn't calibrate per-frame.

This metric instead looks only at the masked region itself, and
compares its local structural detail (Laplacian variance -- a standard
focus/detail measure) against the SAME image's own real, unmasked
content as the reference scale, rather than an absolute constant. A
well-formed object surface should have a detail level in the same
ballpark as other real rendered surfaces in the same frame; a
flat/smeared blob reads far lower, and a blocky/chaotic garble artifact
reads far higher.

**Status: prototype, cross-checked only against a handful of already-
generated images from this session with informal visual judgments (see
scripts/validate_coherence_metric.py) -- not wired into ControlLoop or
PlausibilityChecker's own gating decision. Do not swap this in for
`PlausibilityChecker` without a real validation pass (more frames, a
holdout set, ideally against human-labeled good/bad judgments, not just
the same handful of images used to design it).**

Known blind spot: this only asks "does the masked region look like
plausible real image content," not "was the right thing actually
generated there" -- e.g. a well-placed mask that produces a totally
different (but locally coherent-looking) object would still score well
here. It also can't tell "arm never removed because the mask didn't
cover it" (moka_pots_step060 in the area-sweep) apart from "arm removed
successfully" if what's left in view is real, unaltered, in-focus
content either way -- that failure mode needs a different check
entirely (e.g. does the masked region still overlap what a live arm
segmentation would show).
"""

import cv2
import numpy as np

DEFAULT_TILE_SIZE = 8


def _local_detail_level(image: np.ndarray, region_mask: np.ndarray, tile_size: int = DEFAULT_TILE_SIZE) -> float:
    """Mean Laplacian-variance over `tile_size`-square tiles fully
    inside `region_mask`. Returns 0.0 if no tile fits entirely inside
    the region (region too small/thin for this tile size)."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float64) if image.ndim == 3 else image.astype(np.float64)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    h, w = region_mask.shape
    tile_variances = []
    for y in range(0, h - tile_size + 1, tile_size):
        for x in range(0, w - tile_size + 1, tile_size):
            tile_region = region_mask[y : y + tile_size, x : x + tile_size]
            if tile_region.all():
                tile_lap = laplacian[y : y + tile_size, x : x + tile_size]
                tile_variances.append(float(tile_lap.var()))
    if not tile_variances:
        return 0.0
    return float(np.mean(tile_variances))


def coherence_score(generated_image: np.ndarray, region_mask: np.ndarray, tile_size: int = DEFAULT_TILE_SIZE) -> float:
    """Score in (0, 1]: 1.0 when the masked region's local detail level
    matches the rest of the same image's real content; falls off
    (toward 0) the further it deviates in either direction (too flat =
    smeared blob, too chaotic = garbled noise). Per-frame calibrated --
    no fixed absolute scale, unlike PlausibilityChecker's
    DEFAULT_MSE_SCALE. Returns 0.0 if the region is too small for even
    one tile, or if the reference (unmasked) area has zero detail
    (degenerate flat image, e.g. a synthetic test frame)."""
    region_detail = _local_detail_level(generated_image, region_mask, tile_size)
    reference_detail = _local_detail_level(generated_image, ~region_mask, tile_size)
    if region_detail == 0.0 or reference_detail <= 1e-9:
        return 0.0
    ratio = region_detail / reference_detail
    return float(np.exp(-abs(np.log(ratio))))
