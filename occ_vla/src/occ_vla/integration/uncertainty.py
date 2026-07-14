"""Physical-plausibility check on a world-model-generated subgoal image.
Used to decide whether to trust it or fall back to PKLP's kinematic
estimate.

Heuristic, not a learned classifier: `sample_arm_free_image()`
(world_model/arm_free_subgoal.py) only masks the arm-covered MAGVIT-v2
tokens before calling MMaDA's t2i_generate, so everything *outside* the
arm mask should come back close to the original — MAGVIT-v2's own
encode/decode round-trip isn't pixel-exact even for untouched tokens,
but a genuinely successful generation should stay close to that
reconstruction-only baseline. A generation that also changed the
background substantially (a "hallucination" that invented a different
scene rather than just filling in the arm region) means the model
overreached, and PKLP's kinematic estimate is the safer fallback.

This does NOT verify physical correctness of what was generated *inside*
the arm mask (e.g. whether the depicted target position is physically
reachable) — only that the model behaved like an inpainter and not like
an unconstrained generator. A real physical-plausibility check (e.g.
comparing the generated target position against PKLP's own kinematic
extrapolation) is a natural follow-up but needs that comparison signal
plumbed in via `physical_context`, which callers don't populate yet.
"""

import numpy as np

PLAUSIBILITY_FALLBACK_THRESHOLD = 0.5
DEFAULT_MSE_SCALE = 40.0  # denominator in exp(-mse/scale); larger = more tolerant of background drift


class PlausibilityChecker:
    def __init__(self, mse_scale: float = DEFAULT_MSE_SCALE):
        self.mse_scale = mse_scale

    def score(self, generated_image: np.ndarray, physical_context: dict) -> float:
        """physical_context must carry `original_image` (HWC, same shape
        as generated_image) and `arm_pixel_mask` (HxW bool, True where
        the arm was — and so where a real change is expected). Returns
        plausibility in [0, 1]; below PLAUSIBILITY_FALLBACK_THRESHOLD,
        the caller should discard the world-model subgoal and use PKLP
        instead."""
        original_image = physical_context["original_image"]
        arm_pixel_mask = physical_context["arm_pixel_mask"]
        if generated_image.shape != original_image.shape:
            raise ValueError(
                f"generated_image shape {generated_image.shape} != original_image shape {original_image.shape}"
            )

        background_mask = ~arm_pixel_mask
        if not background_mask.any():
            # arm covers the whole frame -- nothing to compare against, treat as maximally uncertain
            return 0.0

        diff = generated_image.astype(np.float64) - original_image.astype(np.float64)
        background_mse = float((diff[background_mask] ** 2).mean())
        return float(np.exp(-background_mse / self.mse_scale))

    def should_fallback(self, generated_image: np.ndarray, physical_context: dict) -> bool:
        return self.score(generated_image, physical_context) < PLAUSIBILITY_FALLBACK_THRESHOLD
