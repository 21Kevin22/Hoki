"""Per-patch velocity/acceleration from a two-sample flow history, and
constant-acceleration extrapolation of an occluded target's position:

    x_{t+K} = x_last_clear + V*K + 0.5*A*K^2

V is the RAFT patch flow itself (pixels displaced per frame == pixels
per step, matching K's units in the formula above). A is the finite
difference between the two flow samples in `flow_history`
(optical_flow.py::RaftFlowEstimator.three_frame_patch_flow gives
exactly this: [flow[t-2->t-1], flow[t-1->t]]), divided by dt = 1 frame.
"""

from dataclasses import dataclass

import numpy as np

from occ_vla.pklp.optical_flow import PatchFlow

DEFAULT_EXTRAPOLATION_STEPS = 5
DT_FRAMES = 1.0  # finite-difference step between the two flow samples


@dataclass
class KinematicState:
    position: np.ndarray  # (2,) xy at last_clear
    velocity: np.ndarray  # (2,)
    acceleration: np.ndarray  # (2,)


class KinematicExtrapolator:
    def estimate_state(self, flow_history: list[PatchFlow], target_patch_idx: int) -> KinematicState:
        """flow_history: [flow[t-2->t-1], flow[t-1->t]] (see
        RaftFlowEstimator.three_frame_patch_flow). V is read from the
        most recent sample; A is the finite difference between the two."""
        if len(flow_history) != 2:
            raise ValueError(f"expected a 2-sample flow history, got {len(flow_history)}")
        earlier, latest = flow_history

        position = latest.patch_centers[target_patch_idx]
        v_latest = latest.flow[target_patch_idx]
        v_earlier = earlier.flow[target_patch_idx]
        acceleration = (v_latest - v_earlier) / DT_FRAMES

        return KinematicState(position=position, velocity=v_latest, acceleration=acceleration)

    def extrapolate(self, state: KinematicState, steps: int = DEFAULT_EXTRAPOLATION_STEPS) -> np.ndarray:
        k = steps
        return state.position + state.velocity * k + 0.5 * state.acceleration * (k**2)


def motion_descriptor_tokens(visual_tokens: np.ndarray, velocity: np.ndarray, acceleration: np.ndarray) -> np.ndarray:
    """Concatenate [V; A] onto each patch's visual token, so the model
    can read off "which region is accelerating which way" per-patch.
    visual_tokens: (N, D); velocity, acceleration: (N, 2) each.
    Returns (N, D + 4)."""
    n = visual_tokens.shape[0]
    if velocity.shape != (n, 2) or acceleration.shape != (n, 2):
        raise ValueError(
            f"velocity/acceleration must be ({n}, 2), got {velocity.shape} and {acceleration.shape}"
        )
    return np.concatenate([visual_tokens, velocity, acceleration], axis=-1)
