"""
oft_occlusion_gt.py

Ground-truth occlusion labeling for the SYNTHETIC occlusion conditions used
throughout run_oft_camera_dropout_eval.py (`*_partial`, `*_full`, etc.):
since the occluded pixel rectangle is a script-chosen constant (not
estimated), the true occlusion state of every 14x14 vision-transformer
patch is exactly computable, not something that needs a real
detector/oracle-mask trick.

This underlies A3's evaluation setup: `occ_gt` (this module) is the label
S_occ (produced by whatever detector, currently a placeholder -- see
run_oft_camera_dropout_eval.py's `--s-occ-source`) is being scored against.
Do NOT use this module's output as S_occ itself -- that would make
precision/recall against occ_gt trivially perfect and prove nothing about
detector quality (see run_oft_camera_dropout_eval.py's `--s-occ-source
oracle` docstring for why that mode is a *pipeline scaffold*, not a stand-in
for a real A3 result).

Deliberately numpy-only (no torch) so this is testable and reusable outside
the GPU eval loop -- e.g. for a standalone offline patch-level PR-curve
script once patch-level detector output exists.
"""

from __future__ import annotations

import numpy as np

GRID_SIDE = 16  # 224 / 14
PATCH_PX = 14
NUM_PATCHES_PER_IMAGE = GRID_SIDE * GRID_SIDE  # 256, matches vision_backbone.get_num_patches()


def patch_overlap_fractions(
    pixel_bounds: tuple[int, int, int, int],
    grid_side: int = GRID_SIDE,
    patch_px: int = PATCH_PX,
) -> np.ndarray:
    """Exact area-overlap fraction between the occluded pixel rectangle and
    each patch in the grid, NOT just a center-in/center-out boolean (that
    center-only test is what `_build_patch_token_mask` in
    run_oft_camera_dropout_eval.py uses for *building the correction mask*,
    which is a fine approximation for "should this patch be corrected" --
    but a coarser ground-truth label than we want for scoring a detector).

    Args:
        pixel_bounds: (r0, r1, c0, c1) occluded rectangle in pixel space,
            same convention as `_apply_partial_patch`'s return value.
        grid_side: patches per side (16 for a 224px image at 14px patches).
        patch_px: patch side length in pixels.

    Returns:
        (grid_side, grid_side) float array, each entry in [0, 1] = fraction
        of that patch's pixel area covered by the occluded rectangle.
    """
    r0, r1, c0, c1 = pixel_bounds
    fractions = np.zeros((grid_side, grid_side), dtype=np.float64)
    for i in range(grid_side):
        patch_r0, patch_r1 = i * patch_px, (i + 1) * patch_px
        row_overlap = max(0, min(patch_r1, r1) - max(patch_r0, r0))
        if row_overlap == 0:
            continue
        for j in range(grid_side):
            patch_c0, patch_c1 = j * patch_px, (j + 1) * patch_px
            col_overlap = max(0, min(patch_c1, c1) - max(patch_c0, c0))
            if col_overlap == 0:
                continue
            fractions[i, j] = (row_overlap * col_overlap) / (patch_px * patch_px)
    return fractions


def patch_labels(fractions: np.ndarray, patch_threshold: float = 0.5) -> np.ndarray:
    """Boolean (grid_side, grid_side) hard label per patch: True if more
    than `patch_threshold` of the patch's area is occluded. This is the
    "patch-level ground truth" the advisor plan asks for -- lets a future
    patch-level detector be scored patch-by-patch (precision/recall over
    which of the 256 patches SHOULD have been corrected), not just a
    single per-step scalar."""
    return fractions > patch_threshold


def occ_gt_scalar(fractions: np.ndarray, patch_threshold: float = 0.5) -> float:
    """Single-number occ_gt for the step log: fraction of patches whose
    occlusion fraction exceeds `patch_threshold` (patch-hard-labeled, then
    averaged) -- i.e. "what fraction of this camera's 256 patches were
    truly occluded this step," matching what the user's own plan describes
    ("パッチ単位で遮蔽率を出し、閾値でパッチ単位の真値にする"). Distinct from
    just averaging the raw per-patch fractions (which would under-penalize
    many barely-touched patches) -- patch_threshold controls that tradeoff,
    default 0.5 (majority-of-patch-area rule)."""
    labels = patch_labels(fractions, patch_threshold=patch_threshold)
    return float(labels.mean())


def occ_gt_for_camera_block(
    pixel_bounds: tuple[int, int, int, int] | None,
    grid_side: int = GRID_SIDE,
    patch_px: int = PATCH_PX,
    patch_threshold: float = 0.5,
) -> float:
    """Convenience wrapper: pixel_bounds=None (no synthetic occlusion this
    step/camera) -> occ_gt=0.0, matching the "baseline"/unoccluded
    condition without a special case at every call site."""
    if pixel_bounds is None:
        return 0.0
    fractions = patch_overlap_fractions(pixel_bounds, grid_side=grid_side, patch_px=patch_px)
    return occ_gt_scalar(fractions, patch_threshold=patch_threshold)
