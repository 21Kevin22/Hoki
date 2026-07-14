"""S_occ: fraction of the target object's visible-from-camera area that
is occluded by the occluder box. Buckets tasks into difficulty tiers.

Occlusion is measured against LIBERO's `agentview_image` camera
(Lifelong-Robot-Learning/LIBERO: libero/lifelong/evaluate.py uses this
as the canonical third-person view) via robosuite segmentation
rendering (`camera_segmentations` on `ControlEnv`, see
libero_occ_env.py) rather than a 2D box heuristic, so S_occ reflects
true visibility, not just bounding-box overlap.
"""

from enum import Enum

import numpy as np


class Difficulty(str, Enum):
    LIGHT = "light"
    MEDIUM = "medium"
    HEAVY = "heavy"


# (lower_bound_inclusive, upper_bound_exclusive) on S_occ
DIFFICULTY_BANDS: dict[Difficulty, tuple[float, float]] = {
    Difficulty.LIGHT: (0.0, 0.3),
    Difficulty.MEDIUM: (0.3, 0.6),
    Difficulty.HEAVY: (0.6, 1.0),
}


class SoccMetric:
    def compute(self, target_mask: np.ndarray, occluder_mask: np.ndarray) -> float:
        """S_occ = |target_mask & occluder_mask| / |target_mask|.
        target_mask/occluder_mask: boolean segmentation masks from the
        `agentview_image`-aligned segmentation camera, target_mask
        being the target's *unoccluded* (no-occluder-in-scene) extent."""
        target_area = target_mask.sum()
        if target_area == 0:
            return 0.0
        return float((target_mask & occluder_mask).sum()) / float(target_area)

    def difficulty(self, s_occ: float) -> Difficulty:
        for tier, (lo, hi) in DIFFICULTY_BANDS.items():
            if lo <= s_occ < hi:
                return tier
        return Difficulty.HEAVY
