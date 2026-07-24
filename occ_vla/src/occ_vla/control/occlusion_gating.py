"""Soft attenuation of the agentview frame under self-occlusion.

Replaces the previously-tested hard full-frame zeroing
(`collect_multiview_data.py --gate-occ-threshold`), which zeroed the
*entire* agentview frame once arm_s_occ crossed a low threshold and
measurably made things worse: 3/3 -> 0/1 task success on the same seed
(see occ_vla/CLAUDE.md, "Attention gating tried and made things worse,
not better"). The likely cause was that a full-zero frame discards
global scene context (e.g. where the stove or second pot are) that the
wrist camera's narrow, close-up FOV can't substitute for -- not just
the occluded region itself.

This module scales pixel intensities toward a floor instead of erasing
them to a flat value, so coarse scene structure stays visible at
reduced magnitude. The policy still gets a photometric cue to weight
its always-present wrist view more heavily as arm_s_occ rises, without
losing the spatial context a hard gate discarded.
"""

import numpy as np

# Matches ARM_OCC_THRESHOLD in world_model/arm_free_subgoal.py -- same
# S_occ definition, same trigger point as the rest of the pipeline.
DEFAULT_GATE_THRESHOLD = 0.30
DEFAULT_MIN_SCALE = 0.3  # floor attenuation factor, not zero


def gate_scale(
    s_occ: float, threshold: float = DEFAULT_GATE_THRESHOLD, min_scale: float = DEFAULT_MIN_SCALE
) -> float:
    """1.0 (no attenuation) at/below threshold; ramps linearly down to
    min_scale as s_occ goes from threshold to 1.0. Linear rather than a
    step function so the transition isn't a discontinuity right at the
    threshold value."""
    if s_occ <= threshold:
        return 1.0
    if s_occ >= 1.0:
        return min_scale
    frac = (s_occ - threshold) / (1.0 - threshold)
    return 1.0 - frac * (1.0 - min_scale)


def apply_soft_gate(image: np.ndarray, scale: float) -> np.ndarray:
    """Scale pixel intensities by `scale`, clipped back to the input
    dtype's range. scale >= 1.0 is a no-op (returns `image` unchanged,
    not a copy)."""
    if scale >= 1.0:
        return image
    info = np.iinfo(image.dtype) if np.issubdtype(image.dtype, np.integer) else None
    scaled = image.astype(np.float32) * scale
    if info is not None:
        scaled = scaled.clip(info.min, info.max)
    return scaled.astype(image.dtype)
